// cx-kb-extract: codexek KB behavior extractor (KB-6).
//
// Runs as an -add-plugin AST action alongside the main action during the
// CTU prepare step (clang ... -emit-ast), enumerating function definitions
// and extracting provable behavior records in the codexek KB record shape
// (field names mirror the product's encodeRecord in engine/src/kb.cpp).
//
// v1 scope — deliberately conservative; every accepted function must be a
// full-coverage, all-exits-provable contract:
//
//   Accepted function shape:
//     pointer-returning plain C function, non-variadic, non-template,
//     whose body is a linear sequence of:
//       - constant-initialized local declarations,
//       - `if (<condition>) return <constant-or-&object>;`   (no else),
//       - `if (<condition>) return V1; else return V2;`,
//       - `return <constant-or-&object>;`                    (the tail),
//     or a single `return <pointer param>;` with an unchanged pointer type.
//     where <condition> is one of
//       <integer param> == <int literal> | != | !<param> | bare <param>
//     and every reachable path ends in a return.
//
//   <object> is a file-scope variable or a function-scope static local
//   (static storage, no weak/alias/asm-label/TLS/volatile), used bare as a
//   constant array or via & of the whole object, optionally refined by a
//   constant chain of named fields and array subscripts rooted at it
//   (`&s.f`, `&s.arr[2]`, `sa[1].f` — stage B2; the leaf-most subscript
//   becomes the offset, earlier steps travel in the optional `path`).
//
//   Everything else => coverage "unsupported" with a reason. The extractor
//   never emits a half-proven contract (the analogue of patch semantics:
//   a merge-style "partial extraction" would silently change analyzer
//   behavior; unsupported entries keep the original analysis).
//
// RET entry predicates require integer parameters. A direct pointer return
// uses an unconditional RETARG record.
//
// Output (path passed via -plugin-arg-cx-kb-extract <path>):
//   {"schema":1, "file":"<source path>", "entries":[
//     {"usr","name","signature","linkage","line",
//      "coverage":"complete"|"unsupported","reason",
//      "records":[<JSON objects with the encodeRecord field names>]}]}

#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/RecordLayout.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/Type.h"
#include "clang/Basic/Builtins.h"
#include "clang/Analysis/CodexekKBInline.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendPluginRegistry.h"
#include "clang/UnifiedSymbolResolution/USRGeneration.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/FileSystem.h"

#include <optional>
#include <string>
#include <vector>

using namespace clang;

namespace {

// --- JSON helpers; field names mirror the product encodeRecord -------------

llvm::json::Value predOperand(unsigned index) {
  return llvm::json::Object{
      {"slot", "argument"}, {"index", static_cast<int64_t>(index)}};
}

llvm::json::Value predReturn() { return llvm::json::Object{{"slot", "return"}}; }

llvm::json::Value predAlways() { return llvm::json::Object{{"always", true}}; }

llvm::json::Value predEq(unsigned param, int64_t value) {
  return llvm::json::Object{{"operand", predOperand(param)},
                            {"op", "eq"},
                            {"value", value}};
}

llvm::json::Value predNe(unsigned param, int64_t value) {
  return llvm::json::Object{{"operand", predOperand(param)},
                            {"op", "ne"},
                            {"value", value}};
}

llvm::json::Value retSuccessEq(int64_t value) {
  return llvm::json::Object{{"operand", predReturn()},
                            {"op", "eq"},
                            {"value", value}};
}

llvm::json::Value retSuccessNonNull() {
  return llvm::json::Object{{"operand", predReturn()},
                            {"op", "ne"},
                            {"value", 0}};
}

// --- record language recognition -------------------------------------------

struct Condition {
  unsigned param = 0; // 1-based
  bool equal = true;
  int64_t value = 0;
  bool operator==(const Condition &o) const {
    return param == o.param && equal == o.equal && value == o.value;
  }
};

const VarDecl *paramRef(const Expr *E, const FunctionDecl &FD) {
  E = E->IgnoreParenImpCasts();
  const auto *DRE = dyn_cast<DeclRefExpr>(E);
  if (!DRE) return nullptr;
  const auto *VD = dyn_cast<VarDecl>(DRE->getDecl());
  if (!VD) return nullptr;
  for (unsigned i = 0, n = FD.getNumParams(); i < n; ++i)
    if (FD.getParamDecl(i) == VD) return VD;
  return nullptr;
}

const VarDecl *integerGuardParam(const Expr *E, const FunctionDecl &FD) {
  const auto *Param = paramRef(E, FD);
  if (!Param || !Param->getType()->isIntegerType() ||
      Param->getType().isVolatileQualified())
    return nullptr;
  return Param;
}

// Recognize <param> ==/!= <literal> (either orientation), !<param>, or bare
// <param>. Returns false when the shape is outside the record language.
bool recognizeCondition(const Expr *Cond, const FunctionDecl &FD,
                        Condition &C, std::string &why) {
  Cond = Cond->IgnoreParenImpCasts();
  if (const auto *BO = dyn_cast<BinaryOperator>(Cond)) {
    if (!BO->isEqualityOp()) {
      why = "relational or compound condition";
      return false;
    }
    const auto *Lit =
        dyn_cast<IntegerLiteral>(BO->getRHS()->IgnoreParenImpCasts());
    const auto *Ref = integerGuardParam(BO->getLHS(), FD);
    if (!Ref || !Lit) {
      Ref = integerGuardParam(BO->getRHS(), FD);
      Lit = dyn_cast<IntegerLiteral>(BO->getLHS()->IgnoreParenImpCasts());
    }
    if (!Ref || !Lit) {
      why = "condition is not <param> ==/!= <literal>";
      return false;
    }
    for (unsigned i = 0, n = FD.getNumParams(); i < n; ++i)
      if (FD.getParamDecl(i) == Ref) {
        C.param = i + 1;
        break;
      }
    C.equal = BO->getOpcode() == BO_EQ;
    llvm::APSInt Literal(Lit->getValue(), Lit->getType()->isUnsignedIntegerType());
    auto Value = Literal.tryExtValue();
    auto Type = Ref->getType();
    unsigned Width = FD.getASTContext().getIntWidth(Type);
    bool Fits = Value && (Type->isUnsignedIntegerType()
                              ? *Value >= 0 && Literal.isIntN(Width)
                              : llvm::APInt(64, *Value).isSignedIntN(Width));
    if (!Fits) {
      why = "comparison constant outside parameter or KB integer range";
      return false;
    }
    C.value = *Value;
    return true;
  }
  if (const auto *UO = dyn_cast<UnaryOperator>(Cond)) {
    if (UO->getOpcode() != UO_LNot) {
      why = "unrecognized unary condition";
      return false;
    }
    const auto *Ref = integerGuardParam(UO->getSubExpr(), FD);
    if (!Ref) {
      why = "condition !<expr> with expr not a parameter";
      return false;
    }
    for (unsigned i = 0, n = FD.getNumParams(); i < n; ++i)
      if (FD.getParamDecl(i) == Ref) {
        C.param = i + 1;
        break;
      }
    C.equal = true; // !p means p == 0
    C.value = 0;
    return true;
  }
  if (const auto *Ref = integerGuardParam(Cond, FD)) {
    for (unsigned i = 0, n = FD.getNumParams(); i < n; ++i)
      if (FD.getParamDecl(i) == Ref) {
        C.param = i + 1;
        break;
      }
    C.equal = false; // bare p means p != 0
    C.value = 0;
    return true;
  }
  why = "unrecognized condition shape";
  return false;
}

/* Stage D1: guard disjunctive-normal-form for ORIGIN joint cases
 * (design §4.1 合取/析取). A clause is a conjunction of atoms; the case's
 * guard is the disjunction of clauses. Legacy single-atom guards are the
 * 1-clause 1-atom special case, so the record encoding stays backward
 * compatible. Grammar budget: ≤ 8 clauses, ≤ 8 atoms per clause, nesting
 * depth ≤ 4 — anything beyond fails the extraction (unsupported), never
 * widens the frozen grammar. */
struct GuardDnf {
  std::vector<std::vector<Condition>> clauses;
};

constexpr unsigned GuardMaxClauses = 8;
constexpr unsigned GuardMaxAtoms = 8;
// Depth guards the RECURSION only; the clause/atom budgets below are the
// real grammar bound. Counting raw AST nesting at 4 would silently cap a
// flat `a1||...||aN` chain (depth N-1) at 5 atoms — far below the 8-clause
// budget the DNF transform is prepared to hold (8-depth recursion is still
// trivially safe). 审查任务2 问题3。
constexpr unsigned GuardMaxDepth = 8;

// Dedupe clauses; an empty (vacuous) clause makes the whole DNF TRUE.
void normalizeDnf(GuardDnf &G) {
  for (auto &clause : G.clauses) {
    std::sort(clause.begin(), clause.end(), [](const Condition &a, const Condition &b) {
      if (a.param != b.param) return a.param < b.param;
      if (a.equal != b.equal) return a.equal < b.equal;
      return a.value < b.value;
    });
    clause.erase(std::unique(clause.begin(), clause.end(),
                             [](const Condition &a, const Condition &b) {
                               return a.param == b.param && a.equal == b.equal &&
                                      a.value == b.value;
                             }),
                 clause.end());
  }
  std::sort(G.clauses.begin(), G.clauses.end(),
            [](const std::vector<Condition> &a, const std::vector<Condition> &b) {
              if (a.size() != b.size()) return a.size() < b.size();
              for (size_t i = 0; i < a.size(); ++i)
                if (a[i].param != b[i].param) return a[i].param < b[i].param;
              return false;
            });
  G.clauses.erase(std::unique(G.clauses.begin(), G.clauses.end(),
                              [](const std::vector<Condition> &a,
                                 const std::vector<Condition> &b) {
                                return a == b;
                              }),
                  G.clauses.end());
}

// Conjunction = cross product of clauses. Fails (nullopt) when the product
// would exceed the clause budget — conservative unsupported, never widening.
std::optional<GuardDnf> conjDnf(const GuardDnf &A, const GuardDnf &B) {
  if (A.clauses.size() * B.clauses.size() > GuardMaxClauses) return std::nullopt;
  GuardDnf Out;
  for (const auto &a : A.clauses)
    for (const auto &b : B.clauses) {
      std::vector<Condition> merged = a;
      merged.insert(merged.end(), b.begin(), b.end());
      if (merged.size() > GuardMaxAtoms) return std::nullopt;
      Out.clauses.push_back(std::move(merged));
    }
  normalizeDnf(Out);
  return Out;
}

// Disjunction = clause union.
std::optional<GuardDnf> disjDnf(const GuardDnf &A, const GuardDnf &B) {
  if (A.clauses.size() + B.clauses.size() > GuardMaxClauses) return std::nullopt;
  GuardDnf Out = A;
  Out.clauses.insert(Out.clauses.end(), B.clauses.begin(), B.clauses.end());
  normalizeDnf(Out);
  return Out;
}

// Negation by De Morgan: flip every atom (eq<->ne stays in grammar) and take
// the cross product over clauses. Fails when the product would blow the
// budget — the caller treats a negation failure as unsupported.
std::optional<GuardDnf> negateDnf(const GuardDnf &G) {
  std::vector<std::vector<Condition>> negated;
  negated.reserve(G.clauses.size());
  for (const auto &clause : G.clauses) {
    std::vector<Condition> flipped;
    flipped.reserve(clause.size());
    for (const auto &atom : clause) {
      Condition F = atom;
      F.equal = !F.equal;
      flipped.push_back(F);
    }
    negated.push_back(std::move(flipped));
  }
  GuardDnf Out;
  Out.clauses.push_back({});
  for (const auto &flipped : negated) {
    if (Out.clauses.size() * flipped.size() > GuardMaxClauses) return std::nullopt;
    std::vector<std::vector<Condition>> next;
    for (const auto &partial : Out.clauses)
      for (const auto &atom : flipped) {
        std::vector<Condition> merged = partial;
        merged.push_back(atom);
        if (merged.size() > GuardMaxAtoms) return std::nullopt;
        next.push_back(std::move(merged));
      }
    Out.clauses = std::move(next);
  }
  normalizeDnf(Out);
  return Out;
}

// Recognize a compound guard expression: parenthesized leaf conditions
// joined by && / ||, with !prefix delegating to atom negation + De Morgan.
bool recognizeGuardDnf(const Expr *E, const FunctionDecl &FD, GuardDnf &G,
                       std::string &why, unsigned depth = 0) {
  if (depth > GuardMaxDepth) {
    why = "compound guard nesting beyond budget";
    return false;
  }
  const Expr *Stripped = E ? E->IgnoreParenImpCasts() : nullptr;
  if (const auto *BO = dyn_cast_or_null<BinaryOperator>(Stripped)) {
    if (BO->getOpcode() == BO_LAnd || BO->getOpcode() == BO_LOr) {
      GuardDnf L, R;
      if (!recognizeGuardDnf(BO->getLHS(), FD, L, why, depth + 1) ||
          !recognizeGuardDnf(BO->getRHS(), FD, R, why, depth + 1))
        return false;
      auto Merged = BO->getOpcode() == BO_LAnd ? conjDnf(L, R) : disjDnf(L, R);
      if (!Merged) {
        why = "compound guard clause budget exceeded";
        return false;
      }
      G = std::move(*Merged);
      return true;
    }
  }
  if (const auto *UO = dyn_cast_or_null<UnaryOperator>(Stripped)) {
    if (UO->getOpcode() == UO_LNot) {
      GuardDnf Inner;
      if (!recognizeGuardDnf(UO->getSubExpr(), FD, Inner, why, depth + 1))
        return false;
      auto Negated = negateDnf(Inner);
      if (!Negated) {
        why = "compound guard clause budget exceeded";
        return false;
      }
      G = std::move(*Negated);
      return true;
    }
  }
  Condition C;
  if (!recognizeCondition(Stripped, FD, C, why)) return false;
  G.clauses.push_back({C});
  return true;
}

// Joint-case guard encoding: single atom keeps the legacy flat record;
// anything compound becomes {"any": [{"all": [atom...]}, ...]}.
llvm::json::Object encodeGuard(const std::optional<GuardDnf> &guard) {
  if (!guard) return llvm::json::Object{{"always", true}};
  auto atom = [](const Condition &C) {
    return llvm::json::Object{{"argument", int64_t(C.param)},
                              {"op", C.equal ? "eq" : "ne"}, {"value", C.value}};
  };
  if (guard->clauses.size() == 1 && guard->clauses[0].size() == 1)
    return atom(guard->clauses[0][0]);
  llvm::json::Array Any;
  for (const auto &clause : guard->clauses) {
    llvm::json::Array All;
    for (const auto &C : clause) All.push_back(llvm::json::Value(atom(C)));
    Any.push_back(llvm::json::Value(llvm::json::Object{{"all", std::move(All)}}));
  }
  return llvm::json::Object{{"any", std::move(Any)}};
}

// Null and the address of a strong, non-thread-local object with static
// storage can be represented as pointer nullness facts.
bool recognizeReturnValue(const Expr *E, ASTContext &Ctx,
                          bool &ReturnsNonNull) {
  if (E->isNullPointerConstant(Ctx, Expr::NPC_ValueDependentIsNotNull)) {
    ReturnsNonNull = false;
    return true;
  }
  const auto *Op = dyn_cast<UnaryOperator>(E->IgnoreParenImpCasts());
  if (!Op || Op->getOpcode() != UO_AddrOf)
    return false;
  const auto *Ref = dyn_cast<DeclRefExpr>(Op->getSubExpr()->IgnoreParens());
  const auto *Object = Ref ? dyn_cast<VarDecl>(Ref->getDecl()) : nullptr;
  if (!Object || !Object->hasGlobalStorage() || Object->isWeak() ||
      Object->getTLSKind() != VarDecl::TLS_None)
    return false;
  ReturnsNonNull = true;
  return true;
}

struct Exit {
  std::optional<Condition> condition;
  unsigned returnArgument = 0;
  bool returnsNonNull = false;
  unsigned line = 0;
};

struct Entry {
  std::string usr;
  std::string name;
  std::string signature;
  std::string linkage = "external";
  unsigned line = 0;
  std::string coverage;
  std::string reason;
  std::vector<Exit> exits;
};

// Linear-sequence body walker. Returns "terminates" for statements that end
// every path they govern (a return). Any other shape => unsupported.
class BodyWalker {
public:
  enum class Flow { Terminates, FallsThrough, Unsupported };

  BodyWalker(const FunctionDecl &FD, ASTContext &Ctx, Entry &E)
      : FD(FD), Ctx(Ctx), E(E) {}

  bool hasFailed() const { return failed; }
  const std::string &failureReason() const { return why; }

  /* Analyzes the body. Flow::Terminates means every path provably ends in a
   * recognized constant return (E.exits is the complete contract, with each
   * exit carrying its accumulated path condition). Flow::FallsThrough means
   * some path leaves the function without a return — not full coverage. */
  Flow walkBody(const Stmt *Body) {
    if (const auto *CS = dyn_cast<CompoundStmt>(Body)) {
      if (CS->size() == 1)
        if (const auto *RS = dyn_cast<ReturnStmt>(*CS->body_begin()))
          if (emitAliasReturn(RS))
            return Flow::Terminates;
      return walkSequence(CS);
    }
    fail("body is not a statement sequence");
    return Flow::FallsThrough;
  }

private:
  const FunctionDecl &FD;
  ASTContext &Ctx;
  Entry &E;
  bool failed = false;
  std::string why;

  void fail(const std::string &reason) {
    if (!failed) {
      failed = true;
      why = reason;
    }
  }

  bool emitAliasReturn(const ReturnStmt *RS) {
    const auto *Value = RS->getRetValue();
    const auto *Param = Value ? paramRef(Value, FD) : nullptr;
    if (!Param || Param->getType().isVolatileQualified() ||
        !Ctx.hasSameType(Param->getType(), FD.getReturnType()))
      return false;
    for (unsigned I = 0; I < FD.getNumParams(); ++I)
      if (FD.getParamDecl(I) == Param) {
        Exit X;
        X.returnArgument = I + 1;
        X.line = Ctx.getSourceManager().getPresumedLoc(RS->getReturnLoc()).getLine();
        E.exits.push_back(X);
        return true;
      }
    return false;
  }

  Flow emitReturn(const ReturnStmt *RS,
                  std::optional<Condition> condition) {
    Exit exit;
    exit.line =
        Ctx.getSourceManager().getPresumedLoc(RS->getReturnLoc()).getLine();
    if (!RS->getRetValue()) {
      fail("void return in pointer-returning function");
      return Flow::Unsupported;
    }
    if (!recognizeReturnValue(RS->getRetValue(), Ctx, exit.returnsNonNull)) {
      fail("return is neither null nor the address of a static object");
      return Flow::Unsupported;
    }
    exit.condition = condition;
    E.exits.push_back(exit);
    return Flow::Terminates;
  }

  Flow walkSequence(const CompoundStmt *CS) {
    std::optional<Condition> remaining;
    for (const Stmt *Child : CS->body()) {
      if (const auto *IS = dyn_cast<IfStmt>(Child)) {
        // A second guard would need the conjunction with the first guard's
        // negation. RET records can express only one entry predicate.
        if (remaining) {
          fail("multiple guards require a compound entry predicate");
          return Flow::Unsupported;
        }
        if (IS->getInit() || IS->getConditionVariable()) {
          fail("if with init/condition variable");
          return Flow::Unsupported;
        }
        Condition C;
        if (!recognizeCondition(IS->getCond(), FD, C, why)) {
          fail("unextractable condition: " + why);
          return Flow::Unsupported;
        }
        const Stmt *Then = IS->getThen();
        const ReturnStmt *RS = dyn_cast<ReturnStmt>(Then);
        if (!RS && isa_and_nonnull<CompoundStmt>(Then)) {
          const auto *CS2 = cast<CompoundStmt>(Then);
          if (CS2->size() == 1) RS = dyn_cast<ReturnStmt>(*CS2->body().begin());
        }
        if (!RS) {
          fail("if body is not a single return statement");
          return Flow::Unsupported;
        }
        if (emitReturn(RS, C) == Flow::Unsupported)
          return Flow::Unsupported;

        C.equal = !C.equal;
        if (const Stmt *Else = IS->getElse()) {
          const auto *ER = dyn_cast<ReturnStmt>(Else);
          if (!ER) {
            fail("if else body is not a single return statement");
            return Flow::Unsupported;
          }
          return emitReturn(ER, C);
        }
        remaining = C;
        continue;
      }
      Flow childFlow = walkStatement(Child, remaining);
      if (childFlow == Flow::Unsupported) return Flow::Unsupported;
      if (childFlow == Flow::Terminates) return Flow::Terminates;
    }
    return Flow::FallsThrough;
  }

  Flow walkStatement(const Stmt *S, std::optional<Condition> condition) {
    switch (S->getStmtClass()) {
    case Stmt::NullStmtClass:
    case Stmt::DeclStmtClass: {
      if (const auto *DS = dyn_cast<DeclStmt>(S))
        for (const auto *D : DS->decls())
          if (const auto *VD = dyn_cast<VarDecl>(D))
            if (VD->hasInit() &&
                !isa<IntegerLiteral>(VD->getInit()->IgnoreParenImpCasts())) {
              fail("local declaration with non-constant initializer");
              return Flow::Unsupported;
            }
      return Flow::FallsThrough; // inert
    }
    case Stmt::ReturnStmtClass:
      return emitReturn(cast<ReturnStmt>(S), condition);
    default: {
      if (const auto *ExprS = dyn_cast<Expr>(S)) {
        const Expr *Stripped = ExprS->IgnoreParenImpCasts();
        if (isa<CallExpr>(Stripped)) {
          fail("contains a call");
          return Flow::Unsupported;
        }
        if (const auto *BO = dyn_cast<BinaryOperator>(Stripped))
          if (BO->isAssignmentOp()) {
            fail("contains an assignment");
            return Flow::Unsupported;
          }
      }
      fail("unsupported statement");
      return Flow::Unsupported;
    }
    }
  }
};


// A separate full-coverage leaf transfer retains return/output correlation.
// Unsupported statements abort the whole contract, never just one exit.
// Stage D2 forward declarations (defined after the extract plumbing; the
// OriginBody walker emits forwarded return sources from within result()).
std::optional<llvm::json::Object> forwardedReturnSource(
    const Expr *Expression, const FunctionDecl &FD, ASTContext &Ctx);
std::string generateUSR(const FunctionDecl &FD);
std::string canonicalSignature(const FunctionDecl &FD, ASTContext &Ctx);
bool forwardArgumentMapping(const CallExpr &Call, const FunctionDecl &FD,
                            const FunctionDecl &Callee, ASTContext &Ctx,
                            llvm::json::Array &Arguments);

/* Stage D3: known-target table for STATIC function-pointer globals (design
 * §5 间接调用). A pointer participates only while its target set is
 * CLOSED: every write visible in this TU resolves to a direct function and
 * the pointer's own address never escapes. Internal (static) linkage makes
 * the TU the whole assignment domain, so closure here is a sound claim;
 * external function pointers stay out (cross-TU writes are invisible). */
class FpTargetTable {
public:
  static bool isTrackable(const VarDecl &V, ASTContext &Ctx) {
    const auto &canonical = *V.getCanonicalDecl();
    if (Ctx.getLangOpts().CPlusPlus || canonical.getStorageClass() != SC_Static ||
        !canonical.hasGlobalStorage() || canonical.getType().isVolatileQualified() ||
        canonical.hasAttr<AsmLabelAttr>())
      return false;
    if (canonical.isExternallyVisible()) return false;
    const auto *pt = canonical.getType()->getAs<PointerType>();
    return pt && pt->getPointeeType()->isFunctionProtoType();
  }

  bool closed(const VarDecl &fp) const {
    return !unmodeledStorage && targets.count(&fp) > 0 && open.count(&fp) == 0;
  }

  const std::set<const FunctionDecl *> *targetsOf(const VarDecl &fp) const {
    auto it = targets.find(&fp);
    return it == targets.end() ? nullptr : &it->second;
  }

  bool isTarget(const FunctionDecl &FD) const {
    return functionTargets.count(FD.getCanonicalDecl()) != 0;
  }

  bool unmodeledStorage = false;
  llvm::SmallDenseSet<const FunctionDecl *, 16> functionTargets;
  std::map<const VarDecl *, std::set<const FunctionDecl *>> targets;
  std::set<const VarDecl *> open;
};

class OriginBody {
  using Object = llvm::json::Object;
  using Array = llvm::json::Array;
  struct Path {
    std::optional<GuardDnf> guard; // nullopt = always (vacuous conjunction)
    Array writes;
    Array effects;
  };
  const FunctionDecl &FD;
  ASTContext &Ctx;
  std::string TU;
  const FpTargetTable *Fps = nullptr;
  Array Cases;
  bool Failed = false;
  std::string Why;
  unsigned Steps = 0;
  bool HasDispatch = false;

  void fail(StringRef reason) {
    if (!Failed) {
      Failed = true;
      Why = reason.str();
    }
  }

  // Stage D3: one forwarded origin per known target of a CLOSED static
  // function pointer (design §5: 已知目标合并为案例并集、未知目标保守).
  // The dispatch guard stays with each case; the union across targets is the
  // case set, and regions outside the dispatch guard stay runtime-
  // conservative. Arguments map through the frozen grammar once (the
  // targets share the pointer's prototype).
  std::vector<Object> dispatchedReturnSources(const Expr *Expression) {
    const auto *Call =
        Expression ? dyn_cast<CallExpr>(Expression->IgnoreParens()) : nullptr;

    if (!Call || !Fps) return {};
    const auto *Ref =
        dyn_cast<DeclRefExpr>(Call->getCallee()->IgnoreParenImpCasts());
    const auto *fp = Ref ? dyn_cast<VarDecl>(Ref->getDecl()) : nullptr;
    if (!fp) return {};
    fp = fp->getCanonicalDecl();
    if (!Fps->closed(*fp)) return {};
    const auto *known = Fps->targetsOf(*fp);
    if (!known || known->empty()) return {};
    const auto *proto =
        fp->getType()->getPointeeType()->getAs<FunctionProtoType>();
    if (!proto || proto->getNumParams() != Call->getNumArgs()) return {};
    std::map<std::string, Object> ordered;
    for (const auto *target : *known) {
      if (!Ctx.hasSameType(fp->getType()->getPointeeType(), target->getType()) ||
          target->isWeak() || target->hasAttr<AsmLabelAttr>())
        return {};
      llvm::json::Array arguments;
      if (!forwardArgumentMapping(*Call, FD, *target, Ctx, arguments))
        return {};
      auto USR = generateUSR(*target);
      if (USR.empty()) return {};
      ordered.try_emplace(USR, Object{{"kind", "forwarded"},
                                      {"callee_usr", USR},
                                      {"signature", canonicalSignature(*target, Ctx)},
                                      {"arguments", std::move(arguments)}});
    }
    std::vector<Object> out;
    for (auto &[usr, source] : ordered)
      out.push_back(std::move(source));
    HasDispatch = !out.empty();
    return out;
  }

  std::string path(StringRef Name) {
    SmallString<256> P(Name);
    Ctx.getSourceManager().getFileManager().makeAbsolutePath(P);
    llvm::sys::path::remove_dots(P, true);
    return P.str().str();
  }
  // Stage B2: a constant field/array-element chain rooted at a static
  // object (`&s.f`, `&s.arr[2]`, `sa[1].f`). Steps are collected leaf-first
  // and reversed at the end; the leaf-most array subscript becomes the
  // record's legacy `offset`, everything before it travels in `path`.
  // Arrow dereferences, anonymous members and non-constant indices abort
  // the chain (parameter pointee aliasing stays stage C).
  std::optional<std::pair<const VarDecl *, llvm::json::Array>>
  reduceChain(const Expr *Expression) {
    llvm::json::Array LeafFirst;
    const Expr *E = Expression->IgnoreParenImpCasts();
    for (unsigned Guard = 0;; ++Guard) {
      if (Guard > 16) return std::nullopt;
      if (const auto *Sub = dyn_cast<ArraySubscriptExpr>(E)) {
        const auto *Idx = Sub->getIdx()->IgnoreParenImpCasts();
        Expr::EvalResult R;
        if (!Idx->EvaluateAsInt(R, Ctx) || Idx->HasSideEffects(Ctx))
          return std::nullopt;
        auto V = R.Val.getInt().tryExtValue();
        if (!V || *V < 0) return std::nullopt;
        LeafFirst.push_back(Object{{"index", int64_t(*V)}});
        E = Sub->getLHS()->IgnoreParenImpCasts();
        continue;
      }
      if (const auto *M = dyn_cast<MemberExpr>(E)) {
        if (M->isArrow()) return std::nullopt;
        const auto *Field = dyn_cast<FieldDecl>(M->getMemberDecl());
        if (!Field || !Field->getIdentifier() ||
            Field->isAnonymousStructOrUnion())
          return std::nullopt;
        LeafFirst.push_back(Object{{"field", Field->getName().str()}});
        E = M->getBase()->IgnoreParenImpCasts();
        continue;
      }
      // Constant pointer arithmetic (`h + 1`) is the subscript form written
      // with operators; negative results reject.
      if (const auto *BO = dyn_cast<BinaryOperator>(E)) {
        if (BO->getOpcode() != BO_Add && BO->getOpcode() != BO_Sub)
          return std::nullopt;
        const auto *Idx = BO->getRHS()->IgnoreParenImpCasts();
        Expr::EvalResult R;
        if (!Idx->EvaluateAsInt(R, Ctx) || Idx->HasSideEffects(Ctx))
          return std::nullopt;
        auto V = R.Val.getInt().tryExtValue();
        if (!V) return std::nullopt;
        int64_t Index = BO->getOpcode() == BO_Add ? int64_t(*V) : -int64_t(*V);
        if (Index < 0) return std::nullopt;
        LeafFirst.push_back(Object{{"index", Index}});
        E = BO->getLHS()->IgnoreParenImpCasts();
        continue;
      }
      break;
    }
    const auto *Ref = dyn_cast<DeclRefExpr>(E);
    const auto *V = Ref ? dyn_cast<VarDecl>(Ref->getDecl()) : nullptr;
    if (!V) return std::nullopt;
    std::reverse(LeafFirst.begin(), LeafFirst.end());
    return std::make_pair(V, std::move(LeafFirst));
  }
  // Root-to-leaf type after every path step; the caller then applies the
  // final array offset. Field lookup requires the complete record
  // definition and named members only.
  QualType terminalType(QualType Type, const llvm::json::Array &Path) const {
    for (const auto &Step : Path) {
      const auto *Object = Step.getAsObject();
      if (!Object) return {};
      if (const auto Field = Object->getString("field")) {
        const auto *Record = Type->getAs<RecordType>();
        if (!Record) return {};
        const auto *Definition = Record->getDecl()->getDefinition();
        if (!Definition) return {};
        bool Found = false;
        for (const auto *Member : Definition->fields()) {
          if (Member->getIdentifier() && Member->getName() == *Field) {
            Type = Member->getType();
            Found = true;
            break;
          }
        }
        if (!Found) return {};
      } else if (auto Index = Object->getInteger("index")) {
        const auto *Array = Ctx.getAsConstantArrayType(Type);
        if (!Array || *Index < 0 ||
            llvm::APInt(64, uint64_t(*Index)).uge(Array->getSize().zextOrTrunc(64)))
          return {};
        Type = Array->getElementType();
      } else {
        return {};
      }
    }
    return Type;
  }
  std::optional<Object> source(const Expr *Expression, QualType Target) {
    if (!Expression || Target.isVolatileQualified() || Target->isAtomicType())
      return std::nullopt;
    if (Target->isIntegerType()) {
      Expr::EvalResult R;
      if (Expression->EvaluateAsInt(R, Ctx) && !Expression->HasSideEffects(Ctx)) {
        // Constant-folded pointer arithmetic (e.g. the layout constant of
        // `a[1] - a[0]`) evaluates fine but is not a source-identity fact;
        // integer sources are status-code / write-slot literals only.
        std::function<bool(const Stmt *)> HasPointerOperand = [&](const Stmt *S) {
          if (const auto *E = dyn_cast<Expr>(S))
            if (E->getType()->isPointerType() || E->getType()->isArrayType() ||
                E->getType()->isReferenceType())
              return true;
          for (const Stmt *Child : S->children())
            if (HasPointerOperand(Child)) return true;
          return false;
        };
        if (!HasPointerOperand(Expression)) {
          if (auto V = R.Val.getInt().tryExtValue())
            return Object{{"kind", "integer"}, {"value", *V}};
        }
      }
      // Stage C1: an unmodified integer parameter feeds a write effect with
      // its call-site value; any other expression stays unsupported.
      if (const auto *P = paramRef(Expression, FD)) {
        if (!P->getType().isVolatileQualified() &&
            Ctx.hasSameUnqualifiedType(P->getType(), Target)) {
          for (unsigned I = 0; I < FD.getNumParams(); ++I)
            if (FD.getParamDecl(I) == P)
              return Object{{"kind", "argument"}, {"index", int64_t(I + 1)}};
        }
      }
      return std::nullopt;
    }
    if (!Target->isPointerType()) return std::nullopt;
    if (Expression->isNullPointerConstant(Ctx, Expr::NPC_ValueDependentIsNotNull))
      return Object{{"kind", "null"}};
    // Stage C2: a direct allocator call returns a fresh heap object whose
    // identity is created per call instance — never merged across calls.
    // void* implicitly converts to the requested pointee, so the call type
    // may be void* or already the target pointer type.
    if (const auto *Call = dyn_cast<CallExpr>(Expression->IgnoreParenImpCasts())) {
      if (const auto *Callee = Call->getDirectCallee()) {
        const auto *II = Callee->getIdentifier();
        // A user definition of malloc/calloc/realloc is an allocator with
        // unknown semantics — only the unimplemented (extern) standard
        // declaration may carry heap-instance meaning.
        if (II && !Callee->hasBody() &&
            (II->isStr("malloc") || II->isStr("calloc") ||
             II->isStr("realloc"))) {
          const auto &Returned = Call->getType();
          if (Ctx.hasSameUnqualifiedType(Returned, Target) ||
              Ctx.hasSameType(Returned, Ctx.VoidPtrTy))
            return Object{{"kind", "heap"}};
        }
      }
    }
    const auto *E = Expression->IgnoreParenImpCasts();
    if (const auto *P = paramRef(E, FD)) {
      if (P->getType().isVolatileQualified() || !Ctx.hasSameType(P->getType(), Target))
        return std::nullopt;
      for (unsigned I = 0; I < FD.getNumParams(); ++I)
        if (FD.getParamDecl(I) == P)
          return Object{{"kind", "argument"}, {"index", int64_t(I + 1)}};
    }
    bool Address = false;
    if (const auto *U = dyn_cast<UnaryOperator>(E)) {
      if (U->getOpcode() != UO_AddrOf) return std::nullopt;
      E = U->getSubExpr()->IgnoreParens();
      Address = true;
    }
    auto Chain = reduceChain(E);
    const auto *V = Chain ? Chain->first : nullptr;
    // Static locals have static storage and a USR rooted at their function,
    // so the consumer's TU + USR identity disambiguates same-named objects;
    // they are accepted alongside file-scope objects (stage B1).
    if (!V || !V->hasGlobalStorage() || V->isWeak() ||
        V->hasAttr<AliasAttr>() || V->hasAttr<AsmLabelAttr>() ||
        V->getTLSKind() != VarDecl::TLS_None || V->getType().isVolatileQualified())
      return std::nullopt;
    // Path form (stage B2): root-to-leaf steps with the leaf-most array
    // subscript carried as the legacy offset (== size stays a one-past
    // source, matching the consumer bound). A chain with steps needs the
    // address-of form only when the terminal is not an array decay.
    llvm::json::Array Path;
    int64_t Offset = 0;
    bool LeafIndex = false;
    QualType Element;
    if (Chain && !Chain->second.empty()) {
      Path = std::move(Chain->second);
      // Leaf-most array subscript (original last after reverse) is the
      // offset; == size stays a one-past source, > size is out of bounds
      // and never extracted (never emit a half-proven contract).
      if (auto Last = Path.back().getAsObject())
        if (auto Index = Last->getInteger("index")) {
          Offset = *Index;
          LeafIndex = true;
          Path.pop_back();
        }
      Element = terminalType(V->getType(), Path);
      if (Element.isNull()) return std::nullopt;
      if (LeafIndex || Offset) {
        const auto *A = Ctx.getAsConstantArrayType(Element);
        if (!A) return std::nullopt;
        if (llvm::APInt(64, uint64_t(Offset)).ugt(A->getSize().zextOrTrunc(64)))
          return std::nullopt;
        Element = A->getElementType();
      } else if (!Address) {
        // Bare value form decays a terminal array to its first element
        // (return s.arr), same as the root-only legacy branch.
        if (const auto *A = Ctx.getAsConstantArrayType(Element))
          Element = A->getElementType();
      }
    } else {
      Element = V->getType();
      if (!Address) {
        const auto *A = Ctx.getAsConstantArrayType(Element);
        if (!A) return std::nullopt;
        Element = A->getElementType();
      }
    }
    if (!Ctx.hasSameType(Ctx.getPointerType(Element), Target)) return std::nullopt;
    SmallString<128> USR;
    if (index::generateUSRForDecl(V, USR)) return std::nullopt;
    auto &SM = Ctx.getSourceManager();
    PrintingPolicy Policy(Ctx.getLangOpts());
    Policy.SuppressTagKeyword = true;
    Policy.Bool = true;
    Object Result{{"kind", "object"}, {"usr", USR.str().str()},
                  {"file", path(SM.getFilename(SM.getSpellingLoc(V->getLocation())))},
                  {"tu", TU}, {"linkage", V->isExternallyVisible() ? "external" : "internal"},
                  {"type", V->getType().getCanonicalType().getAsString(Policy)},
                  {"offset", Offset}};
    if (!Path.empty()) Result["path"] = std::move(Path);
    return Result;
  }
  bool split(const Expr *E, const Path &P, Path &T, Path &F) {
    GuardDnf C;
    std::string Why;
    // Stage D1: compound guards are recognized as a DNF within the frozen
    // grammar budget. A nested guard conjuncts with the live path guard
    // (the previous "reject rather than erase" rule — the conjunction now
    // keeps the earlier guard instead of dropping it); a budget overflow
    // fails the extraction conservatively.
    if (!recognizeGuardDnf(E, FD, C, Why)) {
      fail("joint origin guard unsupported: " + Why);
      return false;
    }
    T = P;
    F = P;
    if (P.guard) {
      auto Conj = conjDnf(*P.guard, C);
      auto Negated = negateDnf(C);
      if (!Conj || !Negated) return false;
      auto FConj = conjDnf(*P.guard, *Negated);
      if (!FConj) return false;
      T.guard = std::move(*Conj);
      F.guard = std::move(*FConj);
      return true;
    }
    T.guard = std::move(C);
    auto Negated = negateDnf(*T.guard);
    if (!Negated) return false;
    F.guard = std::move(*Negated);
    return true;
  }
  void result(const Expr *E, Path P) {
    if (const auto *C = dyn_cast_or_null<ConditionalOperator>(E ? E->IgnoreParens() : nullptr)) {
      Path T, F;
      if (!split(C->getCond(), P, T, F)) { fail("nested ternary guard beyond the frozen budget"); return; }
      result(C->getTrueExpr(), std::move(T));
      result(C->getFalseExpr(), std::move(F));
      return;
    }
    std::vector<Object> values;
    if (FD.getReturnType()->isVoidType() && !E) {
      values.push_back(Object{{"kind", "unknown"}});
    } else if (auto V = source(E, FD.getReturnType())) {
      values.push_back(std::move(*V));
    } else if (auto V = forwardedReturnSource(E, FD, Ctx)) {
      // Stage D2: a return of `callee(args)` inside the frozen mapping
      // grammar is a forwarded source — the SCC transfer composes it
      // against the callee candidate; an unresolved callee keeps the case
      // and the composition drops it (partial), never fabricates.
      values.push_back(std::move(*V));
    } else {
      // Stage D3: indirect dispatch through a closed static function
      // pointer — one forwarded case per known target; an open set
      // yields nothing and the extraction fails conservatively.
      values = dispatchedReturnSources(E);
    }
    if (values.empty() || Cases.size() + values.size() > 32) {
      fail(values.empty() ? "unrepresentable return source"
                          : "case count beyond budget");
      return;
    }
    Object Guard = encodeGuard(P.guard);
    const bool approximate = values.size() > 1;
    for (auto &V : values) {
      Object Row{{"guard", Object(Guard)}, {"return", std::move(V)},
                 {"writes", llvm::json::Array(P.writes)}, {"unknown_effects", false}};
      if (approximate) Row["approximate"] = true;
      if (!P.effects.empty()) Row["effects"] = llvm::json::Array(P.effects);
      Cases.push_back(std::move(Row));
    }
  }
  std::vector<Path> walk(const Stmt *S, std::vector<Path> Live, unsigned Depth) {
    if (!S || Failed) return Live;
    if (++Steps > 256 || Depth > 64) { fail("statement walk beyond budget"); return {}; }
    if (const auto *Block = dyn_cast<CompoundStmt>(S)) {
      for (const auto *Child : Block->body()) {
        Live = walk(Child, std::move(Live), Depth + 1);
        if (Live.empty() || Failed) break;
      }
      return Live;
    }
    if (isa<NullStmt>(S)) return Live;
    // Local declarations are inert for pointer origins: constant-initialized
    // (or uninitialized) objects never produce a source by themselves, and a
    // returned local without static storage is rejected in source(). A
    // variably-modified declaration evaluates a size expression with
    // unmodeled effects, so it aborts the contract (stage B1).
    if (const auto *DS = dyn_cast<DeclStmt>(S)) {
      for (const auto *D : DS->decls())
        if (const auto *VD = dyn_cast<VarDecl>(D)) {
          if (VD->getType()->isVariablyModifiedType()) { fail("variably modified local declaration"); return {}; }
          if (VD->hasInit() &&
              !isa<IntegerLiteral>(VD->getInit()->IgnoreParenImpCasts())) {
            fail("local declaration with non-constant initializer");
            return {};
          }
        }
      return Live;
    }
    if (const auto *R = dyn_cast<ReturnStmt>(S)) {
      for (auto &P : Live) result(R->getRetValue(), std::move(P));
      return {};
    }
    if (const auto *If = dyn_cast<IfStmt>(S)) {
      if (If->getInit() || If->getConditionVariable()) { fail("if with init/condition variable"); return {}; }
      std::vector<Path> Next;
      for (const auto &P : Live) {
        Path T, F;
        if (!split(If->getCond(), P, T, F)) { fail("if condition beyond the frozen guard grammar"); return {}; }
        auto A = walk(If->getThen(), {std::move(T)}, Depth + 1);
        auto B = walk(If->getElse(), {std::move(F)}, Depth + 1);
        for (auto &X : A) Next.push_back(std::move(X));
        for (auto &X : B) Next.push_back(std::move(X));
      }
      return Next;
    }
    if (const auto *Call = dyn_cast<CallExpr>(S)) {
      const auto *Callee = Call->getDirectCallee();
      if (!Callee || !Callee->getIdentifier() || Callee->getBuiltinID() != Builtin::BIfree ||
          Callee->isDefined() ||
          Callee->isWeak() || Callee->hasAttr<AsmLabelAttr>() ||
          Callee->isVariadic() || !Callee->getDeclContext()->isTranslationUnit() ||
          Callee->getNumParams() != 1 || Call->getNumArgs() != 1 ||
          !Callee->getReturnType()->isVoidType() ||
          !Ctx.hasSameType(Callee->getParamDecl(0)->getType(), Ctx.VoidPtrTy)) {
        fail("unsupported call effect");
        return {};
      }
      const auto *Arg = Call->getArg(0)->IgnoreParenImpCasts();
      QualType type = Arg->getType();
      if (type->isArrayType()) type = Ctx.getArrayDecayedType(type);
      auto target = source(Arg, type->isPointerType() ? type : Ctx.VoidPtrTy);
      if (!target || target->getString("kind") == "heap") {
        fail("free target is not a stable input or object");
        return {};
      }
      if (target->getString("kind") == "null") return Live;
      auto freeUsr = generateUSR(*Callee);
      if (freeUsr.empty()) {
        fail("free declaration identity is unavailable");
        return {};
      }
      for (auto &P : Live) {
        if (!P.writes.empty() || !P.effects.empty()) {
          fail("free combined with another memory effect");
          return {};
        }
        P.effects.push_back(Object{{"kind", "free"}, {"target", Object(*target)},
                                   {"callee_usr", freeUsr}});
      }
      return Live;
    }
    const auto *Assignment = dyn_cast<BinaryOperator>(S);
    const auto *Deref = Assignment && Assignment->getOpcode() == BO_Assign
                            ? dyn_cast<UnaryOperator>(Assignment->getLHS()->IgnoreParens()) : nullptr;
    const auto *Param = Deref && Deref->getOpcode() == UO_Deref
                            ? paramRef(Deref->getSubExpr(), FD) : nullptr;
    // Stage C1: a plain-object assignment (`g.x = v`, `g.arr[1] = v`) is a
    // global/static write effect. The target must reduce to a static root
    // with a constant chain (no dereferenced pointer fields — those alias
    // and stay unmodeled); the effect travels per case in `effects`.
    if (Assignment && Assignment->getOpcode() == BO_Assign && !Param) {
      auto Target = reduceChain(Assignment->getLHS()->IgnoreParenImpCasts());
      const auto *Root = Target ? Target->first : nullptr;
      if (!Root || !Root->hasGlobalStorage() || Root->isWeak() ||
          Root->hasAttr<AliasAttr>() || Root->hasAttr<AsmLabelAttr>() ||
          Root->getTLSKind() != VarDecl::TLS_None ||
          Root->getType().isVolatileQualified()) { fail("unmodeled write target (weak/alias/TLS/volatile/non-global)"); return {}; }
      llvm::json::Array Steps = std::move(Target->second);
      int64_t LeafOffset = 0;
      if (!Steps.empty())
        if (auto Last = Steps.back().getAsObject())
          if (auto Index = Last->getInteger("index")) {
            LeafOffset = *Index;
            Steps.pop_back();
          }
      QualType Terminal = terminalType(Root->getType(), Steps);
      if (Terminal.isNull()) { fail("write chain type does not reduce"); return {}; }
      QualType ElementType = Terminal;
      if (LeafOffset || isa<ConstantArrayType>(Terminal)) {
        const auto *A = Ctx.getAsConstantArrayType(Terminal);
        if (!A || llvm::APInt(64, uint64_t(LeafOffset)).ugt(A->getSize().zextOrTrunc(64)))
          { fail("effect offset outside the array bounds"); return {}; }
        ElementType = A->getElementType();
      }
      if (ElementType.isConstQualified() || ElementType.isVolatileQualified())
        { fail("write target element is const/volatile qualified"); return {}; }
      auto Value = source(Assignment->getRHS(), ElementType);
      if (!Value) { fail("effect value is unrepresentable"); return {}; }
      SmallString<128> USR;
      if (index::generateUSRForDecl(Root, USR)) { fail("USR generation failed for the write target"); return {}; }
      auto &SM = Ctx.getSourceManager();
      PrintingPolicy Policy(Ctx.getLangOpts());
      Policy.SuppressTagKeyword = true;
      Policy.Bool = true;
      Object TargetDescriptor{{"kind", "object"}, {"usr", USR.str().str()},
          {"file", path(SM.getFilename(SM.getSpellingLoc(Root->getLocation())))},
          {"tu", TU}, {"linkage", Root->isExternallyVisible() ? "external" : "internal"},
          {"type", Root->getType().getCanonicalType().getAsString(Policy)},
          {"offset", LeafOffset}};
      if (!Steps.empty()) TargetDescriptor["path"] = std::move(Steps);
      for (auto &P : Live) {
        if (!P.effects.empty()) { fail("one write effect per path is representable"); return {}; }
        P.effects.push_back(Object{{"target", std::move(TargetDescriptor)},
                                   {"value", Object(*Value)}});
      }
      return Live;
    }
    if (!Param || !Param->getType()->isPointerType() ||
        Param->getType().isVolatileQualified()) { fail("assignment target is not a pointee parameter"); return {}; }
    const auto Type = Param->getType()->getPointeeType();
    auto Value = source(Assignment->getRHS(), Type);
    if (!Value || Type.isConstQualified() || Type.isVolatileQualified()) {
      fail("output write value unrepresentable or target qualified");
      return {};
    }
    for (auto &P : Live) {
      // Until ordering/alias effects are represented, one write per path.
      if (!P.writes.empty() ||
          (!P.effects.empty() && P.effects.front().getAsObject()->getString("kind") == "free")) {
        fail("output write combined with another write or release");
        return {};
      }
      const auto I = cast<ParmVarDecl>(Param)->getFunctionScopeIndex() + 1;
      P.writes.push_back(Object{{"argument", int64_t(I)}, {"value", Object(*Value)}});
    }
    return Live;
  }
public:
  OriginBody(const FunctionDecl &F, ASTContext &C, StringRef MainFile,
             const FpTargetTable *Fps = nullptr)
      : FD(F), Ctx(C), TU(path(MainFile)), Fps(Fps) {}
  bool hasFailed() const { return Failed; }
  bool hasDispatch() const { return HasDispatch; }
  const std::string &failureWhy() const { return Why; }
  std::optional<Object> extract() {
    if ((!FD.isExternallyVisible() && (!Fps || !Fps->isTarget(FD))) ||
        FD.isWeak() || FD.hasAttr<AsmLabelAttr>() || FD.isNoReturn() || FD.isVariadic() || isa<CXXMethodDecl>(FD) ||
        FD.getTemplatedKind() != FunctionDecl::TK_NonTemplate || Ctx.getLangOpts().CPlusPlus)
      return std::nullopt;
    auto Remaining = walk(FD.getBody(), {Path{}}, 0);
    if (FD.getReturnType()->isVoidType())
      for (auto &P : Remaining) result(nullptr, std::move(P));
    else if (!Remaining.empty())
      fail("fall-through exit without return");
    if (Failed || Cases.empty()) return std::nullopt;
    // The kb codec bounds the whole summary JSON at 65536 bytes; a DNF
    // grammar can newly reach that envelope, so pre-check here and degrade
    // to unsupported instead of failing loudly in prepare (审查任务 2 问题5).
    Object Summary{{"version", int64_t(1)}, {"cases", std::move(Cases)}};
    llvm::SmallString<8192> Rendered;
    llvm::raw_svector_ostream Stream(Rendered);
    Stream << llvm::json::Value(llvm::json::Object(Summary));
    if (Rendered.size() > 65536) {
      fail("origin summary beyond the record size budget");
      return std::nullopt;
    }
    return Summary;
  }
};

std::string canonicalSignature(const FunctionDecl &FD, ASTContext &Ctx) {
  PrintingPolicy Policy(Ctx.getLangOpts());
  Policy.SuppressTagKeyword = true;
  Policy.Bool = true;
  return FD.getType().getCanonicalType().getAsString(Policy);
}

std::string generateUSR(const FunctionDecl &FD) {
  SmallString<128> Buf;
  if (!index::generateUSRForDecl(&FD, Buf)) return Buf.str().str();
  return {};
}

bool supportedCppForwardingFunction(const FunctionDecl &FD) {
  if (isa<CXXMethodDecl>(FD) || !FD.getIdentifier() ||
      FD.getTemplatedKind() != FunctionDecl::TK_NonTemplate)
    return false;
  const auto *Context = FD.getDeclContext()->getRedeclContext();
  if (!Context->isFileContext())
    return false;
  for (const auto *Declaration : Context->lookup(FD.getDeclName()))
    if (Declaration->getCanonicalDecl() != FD.getCanonicalDecl())
      return false;
  auto isValue = [](QualType T) {
    return !T.isVolatileQualified() &&
           (T->isIntegerType() || T->isPointerType());
  };
  if (!FD.getReturnType()->isVoidType() && !isValue(FD.getReturnType()))
    return false;
  // Even an unused by-value object parameter can run a destructor.
  return llvm::all_of(FD.parameters(), [&](const ParmVarDecl *P) {
    return isValue(P->getType());
  });
}

// Capture direct forwarding independently of the configured KB. Prepare can
// recompose the record after a model edit using the cached extraction.
/* Stage D2: shared argument-mapping grammar for forwarded returns and pure
 * wrappers — identity param, `!param`, or an integer constant (design
 * §5.1 in-component transfer input). Pointer arguments stay identity so
 * output locations map 1:1; arithmetic mappings are outside the frozen
 * grammar and reject (the case then drops conservatively upstream). */
bool forwardArgumentMapping(const CallExpr &Call, const FunctionDecl &FD,
                            const FunctionDecl &Callee, ASTContext &Ctx,
                            llvm::json::Array &Arguments) {
  llvm::SmallDenseSet<const VarDecl *, 8> Used;
  for (unsigned I = 0; I < Call.getNumArgs(); ++I) {
    const auto *Arg = Call.getArg(I)->IgnoreParenImpCasts();
    const auto *Param = paramRef(Arg, FD);
    if (Param && !Param->getType().isVolatileQualified() &&
        Ctx.hasSameType(Param->getType(), Callee.getParamDecl(I)->getType()) &&
        Used.insert(Param).second) {
      const auto *Parm = cast<ParmVarDecl>(Param);
      Arguments.push_back(llvm::json::Object{
          {"param", static_cast<int64_t>(Parm->getFunctionScopeIndex()) + 1}});
      continue;
    }
    if (const auto *U = dyn_cast<UnaryOperator>(Arg)) {
      if (U->getOpcode() == UO_LNot &&
          Callee.getParamDecl(I)->getType()->isIntegerType()) {
        const auto *Inner = paramRef(U->getSubExpr()->IgnoreParenImpCasts(), FD);
        if (Inner && !Inner->getType().isVolatileQualified() &&
            Inner->getType()->isIntegerType() &&
            Ctx.hasSameUnqualifiedType(Inner->getType(),
                                       Callee.getParamDecl(I)->getType()) &&
            Used.insert(Inner).second) {
          const auto *Parm = cast<ParmVarDecl>(Inner);
          Arguments.push_back(llvm::json::Object{
              {"not", static_cast<int64_t>(Parm->getFunctionScopeIndex()) + 1}});
          continue;
        }
      }
    }
    Expr::EvalResult R;
    if (Arg && Callee.getParamDecl(I)->getType()->isIntegerType() &&
        Call.getArg(I)->EvaluateAsInt(R, Ctx) && !Call.getArg(I)->HasSideEffects(Ctx)) {
      if (auto V = R.Val.getInt().tryExtValue()) {
        Arguments.push_back(llvm::json::Object{{"const", *V}});
        continue;
      }
    }
    return false;
  }
  return true;
}

// Stage D2: a return value of the shape `callee(args)` becomes a forwarded
// origin source — the callee's summary applied to the captured mapping.
// Only direct, prototype, non-variadic project callees qualify; the kb codec
// and the composition layer share the callee_usr identity.
std::optional<llvm::json::Object> forwardedReturnSource(
    const Expr *Expression, const FunctionDecl &FD, ASTContext &Ctx) {
  const auto *Call =
      Expression ? dyn_cast<CallExpr>(Expression->IgnoreParens()) : nullptr;
  const auto *Callee = Call ? Call->getDirectCallee() : nullptr;
  // The resolver checks the declaration against the selected project
  // definition. Missing targets remain uncovered.
  if (!Callee || Callee->isWeak() || Callee->isVariadic() ||
      Callee->isNoReturn() || !Callee->getType()->isFunctionProtoType() ||
      Callee->getBuiltinID() || Callee->hasAttr<AsmLabelAttr>() ||
      Call->getNumArgs() != Callee->getNumParams() ||
      !Ctx.hasSameType(FD.getReturnType(), Callee->getReturnType()))
    return std::nullopt;
  auto USR = generateUSR(*Callee);
  if (USR.empty())
    return std::nullopt;
  llvm::json::Array Arguments;
  if (!forwardArgumentMapping(*Call, FD, *Callee, Ctx, Arguments))
    return std::nullopt;
  return llvm::json::Object{{"kind", "forwarded"},
                            {"callee_usr", std::move(USR)},
                            {"signature", canonicalSignature(*Callee, Ctx)},
                            {"arguments", std::move(Arguments)}};
}

std::optional<llvm::json::Object> forwardingCall(const FunctionDecl &FD,
                                                ASTContext &Ctx) {
  if (!FD.isExternallyVisible() || FD.isWeak() || FD.hasAttr<AsmLabelAttr>() ||
      FD.isNoReturn() || !FD.getType()->isFunctionProtoType() ||
      (Ctx.getLangOpts().CPlusPlus && !supportedCppForwardingFunction(FD)))
    return std::nullopt;
  const auto *Body = dyn_cast<CompoundStmt>(FD.getBody());
  if (!Body || Body->size() != 1)
    return std::nullopt;
  const Stmt *Statement = *Body->body_begin();
  const Expr *Expression = dyn_cast<Expr>(Statement);
  if (const auto *Return = dyn_cast<ReturnStmt>(Statement))
    Expression = Return->getRetValue();
  else if (!FD.getReturnType()->isVoidType())
    return std::nullopt;
  const auto *Call =
      Expression ? dyn_cast<CallExpr>(Expression->IgnoreParens()) : nullptr;
  const auto *Callee = Call ? Call->getDirectCallee() : nullptr;
  if (Callee && Callee->getIdentifier() && Callee->getIdentifier()->isStr("free") &&
      !Callee->isDefined())
    return std::nullopt;
  // Stage B3: the callee may itself be a defined project leaf — prepare
  // composes the wrapper contract from the leaf's extracted ORIGIN records
  // (external callees keep the configured-contract composition path).
  if (!Callee || !Callee->isExternallyVisible() ||
      Callee->isWeak() || Callee->isVariadic() || Callee->isNoReturn() ||
      !Callee->getType()->isFunctionProtoType() || Callee->getBuiltinID() ||
      Callee->hasAttr<AsmLabelAttr>() ||
      (Ctx.getLangOpts().CPlusPlus && !supportedCppForwardingFunction(*Callee)) ||
      Call->getNumArgs() != Callee->getNumParams() ||
      !Ctx.hasSameType(FD.getReturnType(), Callee->getReturnType()))
    return std::nullopt;
  auto USR = generateUSR(*Callee);
  if (USR.empty())
    return std::nullopt;
  llvm::json::Array Arguments;
  if (!forwardArgumentMapping(*Call, FD, *Callee, Ctx, Arguments))
    return std::nullopt;
  llvm::json::Object Forward{{"callee_usr", std::move(USR)},
                             {"callee", Callee->getQualifiedNameAsString()},
                             {"signature", canonicalSignature(*Callee, Ctx)},
                             {"arguments", std::move(Arguments)}};
  if (!Ctx.getLangOpts().CPlusPlus && FD.getReturnType()->isPointerType())
    Forward["origin_member"] = true;
  return Forward;
}

class KBExtractConsumer : public ASTConsumer {
  std::string OutPath;
  std::string SourcePath;

  void extractDefinitions(const DeclContext &Context, ASTContext &Ctx,
                          llvm::json::Array &Entries, FpTargetTable &Fps) {
    auto &SM = Ctx.getSourceManager();
    for (Decl *D : Context.decls()) {
      if (isa<NamespaceDecl, LinkageSpecDecl>(D)) {
        extractDefinitions(*cast<DeclContext>(D), Ctx, Entries, Fps);
        continue;
      }
      const auto *FD = dyn_cast<FunctionDecl>(D);
      if (!FD) continue;
      if (!FD->isThisDeclarationADefinition() || !FD->hasBody()) continue;
      auto Location = SM.getPresumedLoc(FD->getLocation());
      Entry E;
      E.usr = generateUSR(*FD);
      E.name = FD->getQualifiedNameAsString();
      E.signature = canonicalSignature(*FD, Ctx);
      E.line = Location.getLine();
      E.linkage = FD->isExternallyVisible() ? "external" : "internal";
      llvm::json::Object JE{{"usr", E.usr}, {"name", E.name},
                            {"signature", E.signature}, {"linkage", E.linkage},
                            {"file", Location.isValid() ? Location.getFilename() : SourcePath},
                            {"line", static_cast<int64_t>(E.line)}};
      // Variably modified declarators are evaluated on function entry.
      // In particular, a bound such as c++ changes the value against which
      // body guards are evaluated. The leaf language has no entry effects.
      if (llvm::any_of(FD->parameters(), [](const ParmVarDecl *P) {
            return P->getOriginalType()->isVariablyModifiedType();
          })) {
        JE["coverage"] = "unsupported";
        JE["reason"] = "variably modified parameter has unmodeled entry evaluation";
        Entries.push_back(std::move(JE));
        continue;
      }
      if (!FD->isVariadic())
        if (auto Forward = forwardingCall(*FD, Ctx)) {
          JE["coverage"] = "unsupported";
          JE["reason"] = "forwarded call needs a configured contract";
          JE["forward"] = std::move(*Forward);
          Entries.push_back(std::move(JE));
          continue;
        }
      // Preserve existing RET/RETARG emission where the legacy leaf proof
      // succeeds. The joint IR adds only the formerly unsupported shapes.
      Entry Legacy = E;
      BodyWalker LegacyWalker(*FD, Ctx, Legacy);
      bool LegacyComplete = FD->getReturnType()->isPointerType() &&
          LegacyWalker.walkBody(FD->getBody()) == BodyWalker::Flow::Terminates &&
          !LegacyWalker.hasFailed();
      if (!LegacyComplete || (!FD->isExternallyVisible() && Fps.isTarget(*FD))) {
        OriginBody Origin(*FD, Ctx, SourcePath, &Fps);
        if (auto Summary = Origin.extract()) {
          llvm::json::Object Rec{{"schema", int64_t(2)}, {"function", E.name},
              {"signature", E.signature}, {"unoverloaded", false}, {"kind", "ORIGIN"},
              {"file", SourcePath}, {"line", int64_t(E.line)}, {"origins", std::move(*Summary)}};
          JE["coverage"] = "complete";
          if (Origin.hasDispatch()) JE["indirect_dispatch"] = true;
          if (!FD->isExternallyVisible()) JE["solver_only"] = true;
          JE["records"] = llvm::json::Array{std::move(Rec)};
          Entries.push_back(std::move(JE));
          continue;
        }
        // The joint path attempted and failed: its reason (e.g. compound
        // guard beyond the frozen budget) is the truthful diagnostic; the
        // legacy walker's message would only describe the fallback attempt.
        if (Origin.hasFailed() && !LegacyComplete) {
          JE["coverage"] = "unsupported";
          JE["reason"] = Origin.failureWhy();
          Entries.push_back(std::move(JE));
          continue;
        }
      }
      if (FD->isVariadic() || isa<CXXMethodDecl>(FD) ||
          FD->getTemplatedKind() != FunctionDecl::TK_NonTemplate ||
          !FD->getReturnType()->isPointerType()) {
        JE["coverage"] = "unsupported";
        JE["reason"] = FD->getReturnType()->isPointerType()
                           ? "C++ method, template or variadic"
                           : "non-pointer return";
        Entries.push_back(std::move(JE));
        continue;
      }

      if (Ctx.getLangOpts().CPlusPlus &&
          (!Context.isTranslationUnit() || !supportedCppForwardingFunction(*FD))) {
        JE["coverage"] = "unsupported";
        JE["reason"] = "C++ declaration outside the supported leaf-function subset";
        Entries.push_back(std::move(JE));
        continue;
      }

      BodyWalker Walker(*FD, Ctx, E);
      auto flow = Walker.walkBody(FD->getBody());

      if (Walker.hasFailed() || flow != BodyWalker::Flow::Terminates) {
        JE["coverage"] = "unsupported";
        JE["reason"] = Walker.hasFailed()
                           ? Walker.failureReason()
                           : "fall-through exit without return";
        Entries.push_back(std::move(JE));
        continue;
      }
      JE["coverage"] = "complete";
      if (auto Leaf = codexek::exactInlineLeaf(*FD, Ctx))
        JE["inline_key"] = std::move(Leaf->Key);
      llvm::json::Array Records;
      for (const auto &X : E.exits) {
        llvm::json::Object Rec;
        Rec["schema"] = 1;
        Rec["function"] = E.name;
        Rec["signature"] = E.signature;
        Rec["unoverloaded"] = false; /* canonical signature present */
        Rec["kind"] = X.returnArgument ? "RETARG" : "RET";
        Rec["file"] = SourcePath;
        Rec["line"] = static_cast<int64_t>(X.line);
        Rec["entry"] = predAlways();
        if (const auto &C = X.condition)
          Rec["entry"] = C->equal ? predEq(C->param, C->value)
                                 : predNe(C->param, C->value);
        if (X.returnArgument) {
          Rec["target"] = predReturn();
          Rec["source"] = predOperand(X.returnArgument);
        } else {
          Rec["success"] = X.returnsNonNull ? retSuccessNonNull()
                                            : retSuccessEq(0);
        }
        Records.push_back(std::move(Rec));
      }
      JE["records"] = std::move(Records);
      Entries.push_back(std::move(JE));
    }
  }

  // Stage D3: recursive statement scan for writes to / address-takes of
  // trackable function-pointer globals.
  // Resolve an assignment RHS to a set of direct function targets: a direct
  // reference (with or without &), or a ternary over recognizable branches
  // (the D3 dispatch union). nullopt = unrecognized write (opens the set).
  static std::optional<std::set<const FunctionDecl *>>
  resolveFpTargets(const Expr *E, unsigned depth = 0) {
    if (!E || depth > 8) return std::nullopt;
    const Expr *rhs = E->IgnoreParenImpCasts();
    if (const auto *rr = dyn_cast<DeclRefExpr>(rhs)) {
      if (const auto *target = dyn_cast<FunctionDecl>(rr->getDecl()))
        if (!target->isVariadic() && !target->isNoReturn() &&
            target->getType()->isFunctionProtoType() && !target->getBuiltinID())
          return std::set<const FunctionDecl *>{target};
      return std::nullopt;
    }
    if (const auto *uo = dyn_cast<UnaryOperator>(rhs))
      if (uo->getOpcode() == UO_AddrOf)
        return resolveFpTargets(uo->getSubExpr(), depth + 1);
    if (const auto *cond = dyn_cast<ConditionalOperator>(rhs)) {
      auto t = resolveFpTargets(cond->getTrueExpr(), depth + 1);
      auto f = resolveFpTargets(cond->getFalseExpr(), depth + 1);
      if (!t || !f) return std::nullopt;
      std::set<const FunctionDecl *> out = *t;
      out.insert(f->begin(), f->end());
      return out;
    }
    return std::nullopt;
  }

  class FpTargetScanner : public RecursiveASTVisitor<FpTargetScanner> {
    ASTContext &Ctx;
    FpTargetTable &Table;
    llvm::json::Array &Objects;
    llvm::SmallDenseSet<const Expr *, 16> Classified;

    void appendLayout(QualType type, llvm::raw_ostream &out,
                      llvm::SmallPtrSetImpl<const RecordDecl *> &seen) {
      if (const auto *array = Ctx.getAsArrayType(type)) {
        appendLayout(array->getElementType(), out, seen);
        return;
      }
      const auto *record = type->getAs<RecordType>();
      const auto *definition = record ? record->getDecl()->getDefinition() : nullptr;
      if (!definition || definition->isInvalidDecl() || !seen.insert(definition).second)
        return;
      const auto &layout = Ctx.getASTRecordLayout(definition);
      out << "\nlayout " << definition->isUnion() << ' '
          << layout.getSize().getQuantity() << ' ' << layout.getAlignment().getQuantity();
      unsigned index = 0;
      for (const auto *field : definition->fields()) {
        out << '\n' << layout.getFieldOffset(index++) << ' ' << field->getName()
            << ' ' << field->getType().getCanonicalType().getAsString();
        if (field->isBitField()) out << ' ' << field->getBitWidthValue();
        appendLayout(field->getType(), out, seen);
      }
    }

    void recordWrite(const VarDecl &fp, const Expr *value) {
      const auto *key = fp.getCanonicalDecl();
      if (auto targets = resolveFpTargets(value)) {
        Table.targets[key].insert(targets->begin(), targets->end());
        for (const auto *target : *targets)
          Table.functionTargets.insert(target->getCanonicalDecl());
      } else {
        Table.open.insert(key);
      }
    }

  public:
    FpTargetScanner(ASTContext &ctx, FpTargetTable &table,
                    llvm::json::Array &objects)
        : Ctx(ctx), Table(table), Objects(objects) {}

    bool VisitVarDecl(VarDecl *decl) {
      if (decl->hasAttr<AliasAttr>() || decl->hasAttr<WeakRefAttr>())
        Table.unmodeledStorage = true;
      if (FpTargetTable::isTrackable(*decl, Ctx) && decl->hasInit())
        recordWrite(*decl, decl->getInit());
      if ((decl->isFileVarDecl() || decl->isStaticLocal()) &&
          decl->isThisDeclarationADefinition()) {
        llvm::SmallString<128> usr;
        if (!index::generateUSRForDecl(decl, usr)) {
          std::string content;
          llvm::raw_string_ostream stream(content);
          decl->print(stream, Ctx.getPrintingPolicy());
          stream << '\n' << decl->getType().getCanonicalType().getAsString();
          llvm::SmallPtrSet<const RecordDecl *, 8> seen;
          appendLayout(decl->getType(), stream, seen);
          llvm::SHA256 digest;
          digest.update(content);
          Objects.push_back(llvm::json::Object{
              {"usr", usr.str().str()}, {"external", decl->isExternallyVisible()},
              {"sha256", llvm::toHex(digest.final(), /*LowerCase=*/true)}});
        }
      }
      return true;
    }

    bool VisitFileScopeAsmDecl(FileScopeAsmDecl *) {
      Table.unmodeledStorage = true;
      return true;
    }

    bool VisitAsmStmt(AsmStmt *) {
      Table.unmodeledStorage = true;
      return true;
    }

    bool VisitBinaryOperator(BinaryOperator *assignment) {
      if (assignment->getOpcode() != BO_Assign) return true;
      const auto *ref = dyn_cast<DeclRefExpr>(assignment->getLHS()->IgnoreParens());
      const auto *fp = ref ? dyn_cast<VarDecl>(ref->getDecl()) : nullptr;
      if (fp && FpTargetTable::isTrackable(*fp, Ctx)) {
        Classified.insert(ref);
        recordWrite(*fp, assignment->getRHS());
      }
      return true;
    }

    bool VisitImplicitCastExpr(ImplicitCastExpr *cast) {
      if (cast->getCastKind() == CK_LValueToRValue)
        if (const auto *ref = dyn_cast<DeclRefExpr>(cast->getSubExpr()->IgnoreParens()))
          if (const auto *fp = dyn_cast<VarDecl>(ref->getDecl()))
            if (FpTargetTable::isTrackable(*fp, Ctx))
              Classified.insert(ref); // Reading the value cannot mutate storage.
      return true;
    }

    bool VisitDeclRefExpr(DeclRefExpr *ref) {
      if (const auto *fp = dyn_cast<VarDecl>(ref->getDecl()))
        if (FpTargetTable::isTrackable(*fp, Ctx) && !Classified.count(ref))
          Table.open.insert(fp->getCanonicalDecl());
      return true;
    }
  };

  void HandleTranslationUnit(ASTContext &Ctx) override {
    llvm::json::Array Entries;
    FpTargetTable Fps;
    llvm::json::Array Objects;
    if (!Ctx.getLangOpts().CPlusPlus)
      FpTargetScanner(Ctx, Fps, Objects).TraverseDecl(Ctx.getTranslationUnitDecl());
    extractDefinitions(*Ctx.getTranslationUnitDecl(), Ctx, Entries, Fps);
    llvm::json::Object Root;
    Root["schema"] = 1;
    Root["file"] = SourcePath;
    Root["entries"] = std::move(Entries);
    Root["objects"] = std::move(Objects);
    std::error_code EC;
    llvm::raw_fd_ostream Out(OutPath, EC, llvm::sys::fs::OF_Text);
    if (EC) {
      llvm::errs() << "cx-kb-extract: cannot write " << OutPath << "\n";
      return;
    }
    Out << llvm::formatv("{0}", llvm::json::Value(std::move(Root)));
    Out.flush();
  }

public:
  KBExtractConsumer(StringRef InFile, StringRef Out)
      : OutPath(Out.str()), SourcePath(InFile.str()) {}
};

class KBExtractAction : public PluginASTAction {
  std::string OutPath;
  std::string SourcePath;

protected:
  std::unique_ptr<ASTConsumer>
  CreateASTConsumer(CompilerInstance &CI, StringRef InFile) override {
    SourcePath = InFile.str();
    return std::make_unique<KBExtractConsumer>(InFile, OutPath);
  }

  bool ParseArgs(const CompilerInstance &CI,
                 const std::vector<std::string> &arg) override {
    /* Positional output path (a "-out=..." value would be swallowed by the
     * clang option parser). Extra args ignored for forward compatibility. */
    if (!arg.empty()) OutPath = arg[0];
    if (OutPath.empty()) {
      llvm::errs() << "cx-kb-extract: missing output path argument\n";
      return false;
    }
    return true;
  }

  ActionType getActionType() override { return AddAfterMainAction; }
};

} // namespace

static FrontendPluginRegistry::Add<KBExtractAction>
    RegisterKBExtract("cx-kb-extract", "codexek KB behavior extractor (KB-6)");

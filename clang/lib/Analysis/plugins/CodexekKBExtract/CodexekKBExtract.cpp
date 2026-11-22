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
#include "clang/AST/Stmt.h"
#include "clang/AST/Type.h"
#include "clang/Analysis/CodexekKBInline.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendPluginRegistry.h"
#include "clang/UnifiedSymbolResolution/USRGeneration.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/JSON.h"
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
class OriginBody {
  using Object = llvm::json::Object;
  using Array = llvm::json::Array;
  struct Path {
    std::optional<Condition> guard;
    Array writes;
    Array effects;
  };
  const FunctionDecl &FD;
  ASTContext &Ctx;
  std::string TU;
  Array Cases;
  bool Failed = false;
  unsigned Steps = 0;

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
    Condition C;
    std::string Why;
    // The v1 codec carries one atom. Reject conjunctions rather than erase
    // an earlier guard and publish an unsound complete contract.
    if (P.guard || !recognizeCondition(E, FD, C, Why)) return false;
    T = P; F = P;
    T.guard = C;
    C.equal = !C.equal;
    F.guard = C;
    return true;
  }
  void result(const Expr *E, Path P) {
    if (const auto *C = dyn_cast_or_null<ConditionalOperator>(E ? E->IgnoreParens() : nullptr)) {
      Path T, F;
      if (!split(C->getCond(), P, T, F)) { Failed = true; return; }
      result(C->getTrueExpr(), std::move(T));
      result(C->getFalseExpr(), std::move(F));
      return;
    }
    auto Value = FD.getReturnType()->isVoidType() && !E
                     ? std::optional<Object>(Object{{"kind", "unknown"}})
                     : source(E, FD.getReturnType());
    if (!Value || Cases.size() >= 32) { Failed = true; return; }
    Object Guard{{"always", true}};
    if (P.guard)
      Guard = Object{{"argument", int64_t(P.guard->param)},
                     {"op", P.guard->equal ? "eq" : "ne"}, {"value", P.guard->value}};
    Object Row{{"guard", std::move(Guard)}, {"return", std::move(*Value)},
               {"writes", std::move(P.writes)}, {"unknown_effects", false}};
    if (!P.effects.empty()) Row["effects"] = std::move(P.effects);
    Cases.push_back(std::move(Row));
  }
  std::vector<Path> walk(const Stmt *S, std::vector<Path> Live, unsigned Depth) {
    if (!S || Failed) return Live;
    if (++Steps > 256 || Depth > 64) { Failed = true; return {}; }
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
          if (VD->getType()->isVariablyModifiedType()) { Failed = true; return {}; }
          if (VD->hasInit() &&
              !isa<IntegerLiteral>(VD->getInit()->IgnoreParenImpCasts())) {
            Failed = true; return {};
          }
        }
      return Live;
    }
    if (const auto *R = dyn_cast<ReturnStmt>(S)) {
      for (auto &P : Live) result(R->getRetValue(), std::move(P));
      return {};
    }
    if (const auto *If = dyn_cast<IfStmt>(S)) {
      if (If->getInit() || If->getConditionVariable()) { Failed = true; return {}; }
      std::vector<Path> Next;
      for (const auto &P : Live) {
        Path T, F;
        if (!split(If->getCond(), P, T, F)) { Failed = true; return {}; }
        auto A = walk(If->getThen(), {std::move(T)}, Depth + 1);
        auto B = walk(If->getElse(), {std::move(F)}, Depth + 1);
        for (auto &X : A) Next.push_back(std::move(X));
        for (auto &X : B) Next.push_back(std::move(X));
      }
      return Next;
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
          Root->getType().isVolatileQualified()) { Failed = true; return {}; }
      llvm::json::Array Steps = std::move(Target->second);
      int64_t LeafOffset = 0;
      if (!Steps.empty())
        if (auto Last = Steps.back().getAsObject())
          if (auto Index = Last->getInteger("index")) {
            LeafOffset = *Index;
            Steps.pop_back();
          }
      QualType Terminal = terminalType(Root->getType(), Steps);
      if (Terminal.isNull()) { Failed = true; return {}; }
      QualType ElementType = Terminal;
      if (LeafOffset || isa<ConstantArrayType>(Terminal)) {
        const auto *A = Ctx.getAsConstantArrayType(Terminal);
        if (!A || llvm::APInt(64, uint64_t(LeafOffset)).ugt(A->getSize().zextOrTrunc(64)))
          { Failed = true; return {}; }
        ElementType = A->getElementType();
      }
      if (ElementType.isConstQualified() || ElementType.isVolatileQualified())
        { Failed = true; return {}; }
      auto Value = source(Assignment->getRHS(), ElementType);
      if (!Value) { Failed = true; return {}; }
      SmallString<128> USR;
      if (index::generateUSRForDecl(Root, USR)) { Failed = true; return {}; }
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
        if (!P.effects.empty()) { Failed = true; return {}; }
        P.effects.push_back(Object{{"target", std::move(TargetDescriptor)},
                                   {"value", Object(*Value)}});
      }
      return Live;
    }
    if (!Param || !Param->getType()->isPointerType() ||
        Param->getType().isVolatileQualified()) { Failed = true; return {}; }
    const auto Type = Param->getType()->getPointeeType();
    auto Value = source(Assignment->getRHS(), Type);
    if (!Value || Type.isConstQualified() || Type.isVolatileQualified()) {
      Failed = true; return {};
    }
    for (auto &P : Live) {
      // Until ordering/alias effects are represented, one write per path.
      if (!P.writes.empty()) { Failed = true; return {}; }
      const auto I = cast<ParmVarDecl>(Param)->getFunctionScopeIndex() + 1;
      P.writes.push_back(Object{{"argument", int64_t(I)}, {"value", Object(*Value)}});
    }
    return Live;
  }
public:
  OriginBody(const FunctionDecl &F, ASTContext &C, StringRef MainFile)
      : FD(F), Ctx(C), TU(path(MainFile)) {}
  std::optional<Object> extract() {
    if (!FD.isExternallyVisible() || FD.isWeak() || FD.hasAttr<AsmLabelAttr>() ||
        FD.isNoReturn() || FD.isVariadic() || isa<CXXMethodDecl>(FD) ||
        FD.getTemplatedKind() != FunctionDecl::TK_NonTemplate || Ctx.getLangOpts().CPlusPlus)
      return std::nullopt;
    auto Remaining = walk(FD.getBody(), {Path{}}, 0);
    if (FD.getReturnType()->isVoidType())
      for (auto &P : Remaining) result(nullptr, std::move(P));
    else if (!Remaining.empty()) Failed = true;
    if (Failed || Cases.empty()) return std::nullopt;
    return Object{{"version", int64_t(1)}, {"cases", std::move(Cases)}};
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
  llvm::SmallDenseSet<const VarDecl *, 8> Used;
  llvm::json::Array Arguments;
  for (unsigned I = 0; I < Call->getNumArgs(); ++I) {
    const auto *Arg = Call->getArg(I)->IgnoreParenImpCasts();
    // Stage B3: integer arguments may also be `!param` (guard flip) or an
    // integer literal (constant folding); pointer arguments stay identity
    // so output locations map 1:1.
    const auto *Param = paramRef(Arg, FD);
    if (Param && !Param->getType().isVolatileQualified() &&
        Ctx.hasSameType(Param->getType(), Callee->getParamDecl(I)->getType()) &&
        Used.insert(Param).second) {
      const auto *Parm = cast<ParmVarDecl>(Param);
      Arguments.push_back(llvm::json::Object{
          {"param", static_cast<int64_t>(Parm->getFunctionScopeIndex()) + 1}});
      continue;
    }
    if (const auto *U = dyn_cast<UnaryOperator>(Arg)) {
      if (U->getOpcode() == UO_LNot &&
          Callee->getParamDecl(I)->getType()->isIntegerType()) {
        const auto *Inner = paramRef(U->getSubExpr()->IgnoreParenImpCasts(), FD);
        if (Inner && !Inner->getType().isVolatileQualified() &&
            Inner->getType()->isIntegerType() &&
            Ctx.hasSameUnqualifiedType(Inner->getType(),
                                       Callee->getParamDecl(I)->getType()) &&
            Used.insert(Inner).second) {
          const auto *Parm = cast<ParmVarDecl>(Inner);
          Arguments.push_back(llvm::json::Object{
              {"not", static_cast<int64_t>(Parm->getFunctionScopeIndex()) + 1}});
          continue;
        }
      }
    }
    Expr::EvalResult R;
    if (Arg && Callee->getParamDecl(I)->getType()->isIntegerType() &&
        Arg->EvaluateAsInt(R, Ctx) && !Arg->HasSideEffects(Ctx)) {
      if (auto V = R.Val.getInt().tryExtValue()) {
        Arguments.push_back(llvm::json::Object{{"const", *V}});
        continue;
      }
    }
    return std::nullopt;
  }
  return llvm::json::Object{{"callee_usr", std::move(USR)},
                            {"callee", Callee->getQualifiedNameAsString()},
                            {"signature", canonicalSignature(*Callee, Ctx)},
                            {"arguments", std::move(Arguments)}};
}

class KBExtractConsumer : public ASTConsumer {
  std::string OutPath;
  std::string SourcePath;

  void extractDefinitions(const DeclContext &Context, ASTContext &Ctx,
                          llvm::json::Array &Entries) {
    auto &SM = Ctx.getSourceManager();
    for (Decl *D : Context.decls()) {
      if (isa<NamespaceDecl, LinkageSpecDecl>(D)) {
        extractDefinitions(*cast<DeclContext>(D), Ctx, Entries);
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
      if (!LegacyComplete) {
        OriginBody Origin(*FD, Ctx, SourcePath);
        if (auto Summary = Origin.extract()) {
          llvm::json::Object Rec{{"schema", int64_t(2)}, {"function", E.name},
              {"signature", E.signature}, {"unoverloaded", false}, {"kind", "ORIGIN"},
              {"file", SourcePath}, {"line", int64_t(E.line)}, {"origins", std::move(*Summary)}};
          JE["coverage"] = "complete";
          JE["records"] = llvm::json::Array{std::move(Rec)};
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

  void HandleTranslationUnit(ASTContext &Ctx) override {
    llvm::json::Array Entries;
    extractDefinitions(*Ctx.getTranslationUnitDecl(), Ctx, Entries);
    llvm::json::Object Root;
    Root["schema"] = 1;
    Root["file"] = SourcePath;
    Root["entries"] = std::move(Entries);
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

//===---------- IssueHash.cpp - Generate identification hashes --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/IssueHash.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/Lexer.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/LineIterator.h"
#include "llvm/Support/MD5.h"

#include <optional>
#include <sstream>
#include <string>

using namespace clang;

// Get a string representation of the parts of the signature that can be
// overloaded on.
static std::string GetSignature(const FunctionDecl *Target) {
  if (!Target)
    return "";
  std::string Signature;

  // When a flow sensitive bug happens in templated code we should not generate
  // distinct hash value for every instantiation. Use the signature from the
  // primary template.
  if (const FunctionDecl *InstantiatedFrom =
          Target->getTemplateInstantiationPattern())
    Target = InstantiatedFrom;

  if (!isa<CXXConstructorDecl>(Target) && !isa<CXXDestructorDecl>(Target) &&
      !isa<CXXConversionDecl>(Target))
    Signature.append(Target->getReturnType().getAsString()).append(" ");
  Signature.append(Target->getQualifiedNameAsString()).append("(");

  for (int i = 0, paramsCount = Target->getNumParams(); i < paramsCount; ++i) {
    if (i)
      Signature.append(", ");
    Signature.append(Target->getParamDecl(i)->getType().getAsString());
  }

  if (Target->isVariadic())
    Signature.append(", ...");
  Signature.append(")");

  const auto *TargetT =
      llvm::dyn_cast_or_null<FunctionType>(Target->getType().getTypePtr());

  if (!TargetT || !isa<CXXMethodDecl>(Target))
    return Signature;

  if (TargetT->isConst())
    Signature.append(" const");
  if (TargetT->isVolatile())
    Signature.append(" volatile");
  if (TargetT->isRestrict())
    Signature.append(" restrict");

  if (const auto *TargetPT =
          dyn_cast_or_null<FunctionProtoType>(Target->getType().getTypePtr())) {
    switch (TargetPT->getRefQualifier()) {
    case RQ_LValue:
      Signature.append(" &");
      break;
    case RQ_RValue:
      Signature.append(" &&");
      break;
    default:
      break;
    }
  }

  return Signature;
}

static std::string GetEnclosingDeclContextSignature(const Decl *EnclosingDecl) {
  if (const auto *ND = dyn_cast_or_null<NamedDecl>(EnclosingDecl)) {
    if (const auto *FD = dyn_cast<FunctionDecl>(ND))
      // To distinguish overloads we need to use the signature.
      return GetSignature(FD);
    return ND->getQualifiedNameAsString();
  }
  return "";
}

static StringRef GetNthLineOfFile(std::optional<llvm::MemoryBufferRef> Buffer,
                                  int Line) {
  if (!Buffer)
    return "";

  llvm::line_iterator LI(*Buffer, false);
  for (; !LI.is_at_eof() && LI.line_number() != Line; ++LI)
    ;

  return *LI;
}

static std::string NormalizeLine(const SourceManager &SM, const FullSourceLoc &L,
                                 const LangOptions &LangOpts) {
  static StringRef Whitespaces = " \t\n";

  StringRef Str = GetNthLineOfFile(SM.getBufferOrNone(L.getFileID(), L),
                                   L.getExpansionLineNumber());
  StringRef::size_type col = Str.find_first_not_of(Whitespaces);
  if (col == StringRef::npos)
    col = 1; // The line only contains whitespace.
  else
    col++;
  SourceLocation StartOfLine =
      SM.translateLineCol(SM.getFileID(L), L.getExpansionLineNumber(), col);
  std::optional<llvm::MemoryBufferRef> Buffer =
      SM.getBufferOrNone(SM.getFileID(StartOfLine), StartOfLine);
  if (!Buffer)
    return {};

  const char *BufferPos = SM.getCharacterData(StartOfLine);

  Token Token;
  Lexer Lexer(SM.getLocForStartOfFile(SM.getFileID(StartOfLine)), LangOpts,
              Buffer->getBufferStart(), BufferPos, Buffer->getBufferEnd());

  size_t NextStart = 0;
  std::ostringstream LineBuff;
  while (!Lexer.LexFromRawLexer(Token) && NextStart < 2) {
    if (Token.isAtStartOfLine() && NextStart++ > 0)
      continue;
    LineBuff << std::string(SM.getCharacterData(Token.getLocation()),
                            Token.getLength());
  }

  return LineBuff.str();
}

static llvm::SmallString<32> GetMD5HashOfContent(StringRef Content) {
  llvm::MD5 Hash;
  llvm::MD5::MD5Result MD5Res;
  SmallString<32> Res;

  Hash.update(Content);
  Hash.final(MD5Res);
  llvm::MD5::stringifyResult(MD5Res, Res);

  return Res;
}

// Tokenizes the issue line with the exact same walk NormalizeLine performs
// and additionally determines, within that single walk, the byte offset of
// the issue's starting token inside the concatenation of the line's tokens.
// The V2 hash must take the relative column and the line content from one
// shared tokenization: two independent computations could reintroduce
// sensitivity to whitespace alignment.
//
// Keep this walk in lockstep with NormalizeLine: same start (first
// non-whitespace character of the issue's expansion line) and same stopping
// rule (first token that starts a new line).
static std::pair<std::string, size_t>
GetNormalizedLineAndIssueColumnOffset(const SourceManager &SM,
                                      const FullSourceLoc &L,
                                      const LangOptions &LangOpts) {
  static StringRef Whitespaces = " \t\n";

  StringRef Str = GetNthLineOfFile(SM.getBufferOrNone(L.getFileID(), L),
                                   L.getExpansionLineNumber());
  StringRef::size_type col = Str.find_first_not_of(Whitespaces);
  if (col == StringRef::npos)
    col = 1; // The line only contains whitespace.
  else
    col++;
  SourceLocation StartOfLine =
      SM.translateLineCol(SM.getFileID(L), L.getExpansionLineNumber(), col);
  std::optional<llvm::MemoryBufferRef> Buffer =
      SM.getBufferOrNone(SM.getFileID(StartOfLine), StartOfLine);
  if (!Buffer)
    return {};

  const char *BufferPos = SM.getCharacterData(StartOfLine);
  // The issue position shares the line's buffer, so raw pointer comparison
  // against token extents is valid. Same FileID as the line by construction.
  const char *IssuePos = SM.getCharacterData(L);

  Token Token;
  Lexer Lexer(SM.getLocForStartOfFile(SM.getFileID(StartOfLine)), LangOpts,
              Buffer->getBufferStart(), BufferPos, Buffer->getBufferEnd());

  size_t NextStart = 0;
  size_t ColumnOffset = 0;
  size_t IssueColumnOffset = 0;
  bool FoundIssueToken = false;
  std::ostringstream LineBuff;
  while (!Lexer.LexFromRawLexer(Token) && NextStart < 2) {
    if (Token.isAtStartOfLine() && NextStart++ > 0)
      continue;
    const char *TokStart = SM.getCharacterData(Token.getLocation());
    if (!FoundIssueToken && TokStart + Token.getLength() > IssuePos) {
      // First token that is not entirely before the issue location: this is
      // the token the issue starts at (covers the issue pointing at a token
      // start, into a token, or at whitespace preceding it).
      IssueColumnOffset = ColumnOffset;
      FoundIssueToken = true;
    }
    LineBuff << std::string(TokStart, Token.getLength());
    ColumnOffset += Token.getLength();
  }
  // The issue location lies past every token of the line (e.g. inside a
  // trailing comment): attribute it to the end of the concatenation.
  if (!FoundIssueToken)
    IssueColumnOffset = ColumnOffset;

  return {LineBuff.str(), IssueColumnOffset};
}

std::string clang::getIssueString(const FullSourceLoc &IssueLoc,
                                  StringRef CheckerName,
                                  StringRef WarningMessage,
                                  const Decl *IssueDecl,
                                  const LangOptions &LangOpts) {
  static StringRef Delimiter = "$";

  return (llvm::Twine(CheckerName) + Delimiter +
          GetEnclosingDeclContextSignature(IssueDecl) + Delimiter +
          Twine(IssueLoc.getExpansionColumnNumber()) + Delimiter +
          NormalizeLine(IssueLoc.getManager(), IssueLoc, LangOpts) +
          Delimiter + WarningMessage)
      .str();
}

SmallString<32> clang::getIssueHash(const FullSourceLoc &IssueLoc,
                                    StringRef CheckerName,
                                    StringRef WarningMessage,
                                    const Decl *IssueDecl,
                                    const LangOptions &LangOpts) {

  return GetMD5HashOfContent(getIssueString(
      IssueLoc, CheckerName, WarningMessage, IssueDecl, LangOpts));
}

std::string clang::getIssueStringV2(const FullSourceLoc &IssueLoc,
                                    StringRef CheckerName,
                                    const Decl *IssueDecl,
                                    const LangOptions &LangOpts) {
  static StringRef Delimiter = "$";

  auto [Line, ColumnOffset] = GetNormalizedLineAndIssueColumnOffset(
      IssueLoc.getManager(), IssueLoc, LangOpts);

  return (llvm::Twine(CheckerName) + Delimiter +
          GetEnclosingDeclContextSignature(IssueDecl) + Delimiter +
          llvm::Twine(ColumnOffset) + Delimiter + Line)
      .str();
}

SmallString<32> clang::getIssueHashV2(const FullSourceLoc &IssueLoc,
                                      StringRef CheckerName,
                                      const Decl *IssueDecl,
                                      const LangOptions &LangOpts) {

  return GetMD5HashOfContent(
      getIssueStringV2(IssueLoc, CheckerName, IssueDecl, LangOpts));
}

// Source-token identities are independent of line layout. The AST supplies
// only a bounded written anchor/role; compiler-generated nodes are not hashed.
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/ParentMapContext.h"
#include "llvm/Support/JSON.h"
#include <algorithm>

namespace {
static std::string pack(llvm::ArrayRef<std::string> Parts) {
  std::string Out;
  for (const auto &P : Parts) Out += std::to_string(P.size()) + ":" + P;
  return Out;
}
static std::string digest(llvm::StringRef Text) {
  return GetMD5HashOfContent(Text).str().str();
}
static IssueIdentitySource &sourceIdentity(SourceLocation L,
    const SourceManager &SM, const LangOptions &LO, IssueIdentityCache &Cache) {
  auto F = SM.getFileID(SM.getSpellingLoc(L));
  auto [It, Inserted] = Cache.sources.try_emplace(F.getHashValue());
  auto &S = It->second;
  if (!Inserted) return S;
  bool Invalid = false;
  auto Text = SM.getBufferData(F, &Invalid);
  if (Invalid) return S;
  Lexer Lex(SM.getLocForStartOfFile(F), LO, Text.begin(), Text.begin(), Text.end());
  Token T;
  while (!Lex.LexFromRawLexer(T)) {
    if (T.is(tok::comment)) continue;
    S.offsets.push_back(SM.getFileOffset(T.getLocation()));
    S.tokens.emplace_back(Text.substr(S.offsets.back(), T.getLength()).str());
  }
  S.digest = digest(pack(S.tokens));
  return S;
}
static size_t tokenIndex(const IssueIdentitySource &S, unsigned Offset) {
  auto I = std::upper_bound(S.offsets.begin(), S.offsets.end(), Offset);
  return I == S.offsets.begin() ? 0 : size_t(I - S.offsets.begin() - 1);
}
static std::string tokenWindow(const IssueIdentitySource &S, size_t A, size_t B) {
  B = std::min(B, S.tokens.size()); A = std::min(A, B);
  return pack(llvm::ArrayRef(S.tokens).slice(A, B-A));
}
static std::pair<size_t, size_t> macroDefinition(const IssueIdentitySource &S,
    size_t Token, llvm::StringRef Name, SourceLocation L, const SourceManager &SM) {
  for (size_t N = Token + 1; N-- > 0;) {
    if (N+2 >= S.tokens.size() || S.tokens[N] != "#" ||
        S.tokens[N+1] != "define" || S.tokens[N+2] != Name) continue;
    auto Text = SM.getBufferData(SM.getFileID(L));
    size_t End = Text.find('\n', S.offsets[N]);
    while (End != llvm::StringRef::npos) {
      size_t Prev = End;
      if (Prev && Text[Prev-1] == '\r') --Prev;
      if (!Prev || Text[Prev-1] != '\\') break;
      End = Text.find('\n', End+1);
    }
    size_t Last = End == llvm::StringRef::npos ? S.tokens.size() :
        std::lower_bound(S.offsets.begin(), S.offsets.end(), End)-S.offsets.begin();
    if (Token < Last) return {N, Last};
  }
  return {Token, std::min(Token+1, S.tokens.size())};
}
class AnchorFinder : public RecursiveASTVisitor<AnchorFinder> {
  std::map<unsigned, std::pair<unsigned, const Stmt *>> &Index;
  unsigned Visited = 0;
  void add(SourceLocation L, const Stmt *S, unsigned Rank) {
    if (L.isInvalid()) return;
    auto &Entry = Index[L.getRawEncoding()];
    if (Rank > Entry.first) Entry = {Rank, S};
  }
public:
  explicit AnchorFinder(std::map<unsigned, std::pair<unsigned, const Stmt *>> &I) : Index(I) {}
  bool shouldWalkTypesOfTypeLocs() const { return false; }
  bool VisitStmt(Stmt *S) {
    if (++Visited > 1000000) return false;
    add(S->getBeginLoc(), S, isa<Expr>(S) ? 10 : 5);
    if (auto *B = dyn_cast<BinaryOperator>(S)) add(B->getOperatorLoc(), S, 100);
    else if (auto *U = dyn_cast<UnaryOperator>(S)) add(U->getOperatorLoc(), S, 100);
    else if (isa<UnaryExprOrTypeTraitExpr, SwitchStmt>(S)) add(S->getBeginLoc(), S, 100);
    return true;
  }
};
static bool contextualBuiltin(llvm::StringRef Name) {
  return Name=="__LINE__" || Name=="__FILE__" || Name=="__FILE_NAME__" ||
      Name=="__BASE_FILE__" || Name=="__COUNTER__" || Name=="__DATE__" ||
      Name=="__TIME__" || Name=="__TIMESTAMP__";
}
static std::string macroAnchor(const Stmt *S, const SourceManager &SM,
    const LangOptions &LO, IssueIdentityCache &Cache, unsigned Depth, bool &Complete) {
  if (!S) return "";
  if (Depth > 256) { Complete = false; return ""; }
  if (const auto *E = dyn_cast<Expr>(S)) S = E->IgnoreParenImpCasts();
  if (const auto *W = dyn_cast<SwitchStmt>(S)) S = W->getCond()->IgnoreParenImpCasts();
  std::vector<std::string> Parts{S->getStmtClassName()};
  if (const auto *B = dyn_cast<BinaryOperator>(S)) Parts.push_back(B->getOpcodeStr().str());
  if (const auto *U = dyn_cast<UnaryOperator>(S)) {
    Parts.push_back(UnaryOperator::getOpcodeStr(U->getOpcode()).str());
    Parts.push_back(U->isPostfix() ? "postfix" : "prefix");
  }
  if (const auto *D = dyn_cast<DeclRefExpr>(S)) Parts.push_back(D->getDecl()->getNameAsString());
  if (const auto *M = dyn_cast<MemberExpr>(S)) {
    Parts.push_back(M->getMemberDecl()->getNameAsString());
    Parts.push_back(M->isArrow() ? "arrow" : "dot");
  }
  if (const auto *C = dyn_cast<ExplicitCastExpr>(S)) {
    auto T = C->getTypeInfoAsWritten()->getTypeLoc();
    auto A = SM.getSpellingLoc(T.getBeginLoc()), B = SM.getSpellingLoc(T.getEndLoc());
    if (A.isValid() && B.isValid() && SM.getFileID(A)==SM.getFileID(B) && SM.getFileOffset(A)<=SM.getFileOffset(B)) {
      auto &Source = sourceIdentity(A,SM,LO,Cache);
      Parts.push_back(tokenWindow(Source,tokenIndex(Source,SM.getFileOffset(A)),tokenIndex(Source,SM.getFileOffset(B))+1));
    } else Complete = false;
  }
  if (isa<IntegerLiteral, FloatingLiteral, CharacterLiteral, StringLiteral>(S)) {
    auto AddLiteral = [&](SourceLocation Written) {
      auto MacroName = Written.isMacroID() ? Lexer::getImmediateMacroName(Written,SM,LO).str() : std::string();
      if (contextualBuiltin(MacroName)) Parts.push_back(MacroName);
      else {
        auto L = SM.getSpellingLoc(Written);
        auto &Source = sourceIdentity(L, SM, LO, Cache);
        if (!Source.tokens.empty()) Parts.push_back(Source.tokens[tokenIndex(Source, SM.getFileOffset(L))]);
      }
    };
    if (const auto *Text = dyn_cast<StringLiteral>(S))
      for (unsigned N = 0; N < Text->getNumConcatenated(); ++N) AddLiteral(Text->getStrTokenLoc(N));
    else AddLiteral(S->getBeginLoc());
  }
  if (const auto *U = dyn_cast<UnaryExprOrTypeTraitExpr>(S)) {
    Parts.push_back(std::to_string(U->getKind()));
    if (U->isArgumentType()) {
      // Written type tokens remain independent of compiler-inferred widths.
      auto T = U->getArgumentTypeInfo()->getTypeLoc();
      auto A = SM.getSpellingLoc(T.getBeginLoc()), B = SM.getSpellingLoc(T.getEndLoc());
      if (A.isValid() && B.isValid() && SM.getFileID(A)==SM.getFileID(B) && SM.getFileOffset(A)<=SM.getFileOffset(B)) {
        auto &Source = sourceIdentity(A,SM,LO,Cache);
        Parts.push_back(tokenWindow(Source,tokenIndex(Source,SM.getFileOffset(A)),tokenIndex(Source,SM.getFileOffset(B))+1));
      }
    }
  }
  for (const auto *Child : S->children()) if (Child) Parts.push_back(macroAnchor(Child,SM,LO,Cache,Depth+1,Complete));
  return pack(Parts);
}
static std::string writtenDecl(const Decl *D, const SourceManager &SM,
    const LangOptions &LO, IssueIdentityCache &Cache) {
  if (!D) return "";
  const auto *N = dyn_cast<NamedDecl>(D);
  std::string Name = N ? N->getQualifiedNameAsString() : D->getDeclKindName();
  // Local variable names alone do not distinguish functions.
  for (auto *C = D->getDeclContext(); C && !C->isTranslationUnit(); C = C->getParent())
    if (auto *F = dyn_cast<FunctionDecl>(C)) Name = writtenDecl(F, SM, LO, Cache)+"/"+Name;
  if (auto *F = dyn_cast<FunctionDecl>(D)) {
    Name = F->getQualifiedNameAsString();
    for (auto *P : F->parameters()) {
      auto A = SM.getExpansionLoc(P->getBeginLoc()), B = SM.getExpansionLoc(P->getEndLoc());
      if (A.isValid() && B.isValid() && SM.getFileID(A)==SM.getFileID(B) && SM.getFileOffset(A)<=SM.getFileOffset(B)) {
        auto &Source = sourceIdentity(A,SM,LO,Cache);
        Name += "/"+tokenWindow(Source,tokenIndex(Source,SM.getFileOffset(A)),tokenIndex(Source,SM.getFileOffset(B))+1);
      } else Name += "/parameter";
    }
    if (F->isVariadic()) Name += "/variadic";
    if (const auto *M = dyn_cast<CXXMethodDecl>(F)) {
      if (M->isConst()) Name += "/const";
      if (M->isVolatile()) Name += "/volatile";
      if (M->getRefQualifier()==RQ_LValue) Name += "/lvalue";
      if (M->getRefQualifier()==RQ_RValue) Name += "/rvalue";
    }
  }
  return Name;
}
} // namespace

std::string IssueIdentity::metadata() const {
  llvm::json::Object O{{"quality", quality}, {"declaration", declaration},
      {"anchor", anchor}, {"role", role}, {"macro", macro},
      {"before", before}, {"after", after}, {"source", sourceDigest},
      {"offset", offset}, {"occurrence", occurrence}, {"tokenKey", tokenKey}, {"macroRole", macroRole}, {"macroDigest", macroDigest}, {"macroOrder", macroOrder}};
  std::string Out; llvm::raw_string_ostream Stream(Out);
  Stream << llvm::json::Value(std::move(O));
  return Out;
}

IssueIdentity clang::getIssueIdentity(SourceLocation L, llvm::StringRef Checker,
    llvm::StringRef Problem, const Decl *Owner, const SourceManager &SM,
    const LangOptions &LO, IssueIdentityCache &Cache) {
  IssueIdentity I;
  if (L.isInvalid()) return I;
  auto Physical = SM.getExpansionLoc(L);
  if (Physical.isInvalid() || !Physical.isFileID()) return I;
  auto &Source = sourceIdentity(Physical, SM, LO, Cache);
  if (Source.tokens.empty()) return I;
  I.offset = SM.getFileOffset(Physical);
  size_t Index = tokenIndex(Source, I.offset);
  I.sourceDigest = Source.digest;
  I.before = tokenWindow(Source, Index > 24 ? Index-24 : 0, Index);
  I.after = tokenWindow(Source, Index+1, Index+25);
  I.declaration = writtenDecl(Owner, SM, LO, Cache);
  I.quality = "token";
  std::vector<std::string> Occurrence{Checker.str(), Problem.str(), std::to_string(I.offset)};
  SourceLocation Current = L;
  std::vector<std::string> Macro;
  std::vector<std::string> Tokens{std::to_string(Index)}, Snapshots{Source.digest}, Roles, Definitions, Orders;
  while (Current.isMacroID()) {
    auto Spelling = SM.getImmediateSpellingLoc(Current);
    auto FileLoc = SM.getSpellingLoc(Spelling);
    auto &Spelled = sourceIdentity(FileLoc, SM, LO, Cache);
    if (Spelled.tokens.empty()) return {};
    size_t Token = tokenIndex(Spelled, SM.getFileOffset(FileLoc));
    auto Name = Lexer::getImmediateMacroName(Current, SM, LO).str();
    auto SlotLoc = SM.getSpellingLoc(SM.getImmediateExpansionRange(Current).getBegin());
    auto &SlotSource = sourceIdentity(SlotLoc, SM, LO, Cache);
    if (SlotSource.tokens.empty()) return {};
    auto Slot = tokenIndex(SlotSource, SM.getFileOffset(SlotLoc));
    bool WrittenFile = bool(SM.getFileEntryRefForID(SM.getFileID(FileLoc)));
    Tokens.push_back(pack({Name, WrittenFile ? std::to_string(Token) : Spelled.tokens[Token], std::to_string(Slot)}));
    Snapshots.push_back(Spelled.digest); Snapshots.push_back(SlotSource.digest);
    auto Caller = SM.getImmediateMacroCallerLoc(Current);
    auto CallFile = SM.getExpansionLoc(Caller);
    auto SpelledValue = contextualBuiltin(Name) ? Name : Spelled.tokens[Token];
    // Raw offsets distinguish expansion instances within this snapshot only.
    // They never enter the semantic fingerprint.
    Occurrence.push_back(pack({Name, WrittenFile ? std::to_string(SM.getFileOffset(FileLoc)) : SpelledValue,
        std::to_string(SM.getFileOffset(CallFile)), std::to_string(SM.getFileOffset(SlotLoc)), SM.isMacroArgExpansion(Current) ? "arg" : "body"}));
    bool Argument = SM.isMacroArgExpansion(Current);
    auto Definition = Argument ? macroDefinition(SlotSource, Slot, Name, SlotLoc, SM) :
        macroDefinition(Spelled, Token, Name, FileLoc, SM);
    auto &DefSource = Argument ? SlotSource : Spelled;
    auto DefToken = Argument ? Slot : Token;
    Roles.push_back(pack({Name, Argument ? "arg" : "body", std::to_string(DefToken-Definition.first)}));
    auto Position = std::to_string(DefToken-Definition.first);
    Orders.push_back(std::string(20-std::min<size_t>(20,Position.size()), '0')+Position);
    Occurrence.push_back(Roles.back());
    Tokens.push_back(Roles.back());
    Definitions.push_back(tokenWindow(DefSource, Definition.first, Definition.second));
    Macro.push_back(pack({Name, Argument ? "arg" : "body", SpelledValue}));
    Current = Caller;
  }
  I.macro = pack(Macro);
  I.sourceDigest = digest(pack(Snapshots));
  I.tokenKey = digest(pack(Tokens));
  I.macroRole = pack(Roles);
  I.macroDigest = digest(pack(Definitions));
  for (auto P = Orders.rbegin(); P != Orders.rend(); ++P) I.macroOrder += *P;
  // Recover the source node for basic reports that carry only SourceLocation.
  const Stmt *S = nullptr;
  if (Owner) {
    auto [Entry, Inserted] = Cache.anchors.try_emplace(Owner);
    if (Inserted) {
      AnchorFinder Finder(Entry->second);
      Finder.TraverseDecl(const_cast<Decl *>(Owner));
    }
    auto Node = Entry->second.find(L.getRawEncoding());
    if (Node != Entry->second.end()) S = Node->second.second;
  }
  if (S) {
    auto A = SM.getSpellingLoc(S->getBeginLoc()), B = SM.getSpellingLoc(S->getEndLoc());
    if (auto *Switch = dyn_cast<SwitchStmt>(S)) {
      A = SM.getSpellingLoc(Switch->getCond()->getBeginLoc());
      B = SM.getSpellingLoc(Switch->getCond()->getEndLoc());
    }
    if (A.isValid() && B.isValid() && SM.getFileID(A) == SM.getFileID(B) &&
        SM.getFileOffset(A) <= SM.getFileOffset(B)) {
      auto &Written = sourceIdentity(A, SM, LO, Cache);
      auto First = tokenIndex(Written, SM.getFileOffset(A));
      auto Last = tokenIndex(Written, SM.getFileOffset(B));
      if (Last-First <= 256) {
        I.anchor = tokenWindow(Written, First, Last+1);
        I.role = S->getStmtClassName(); I.quality = "ast";
        // A written guard/branch is a local semantic role, not a statement
        // ordinal. It distinguishes equal expressions in different branches
        // without hashing the entire function or compiler-inferred types.
        const Stmt *Child = S;
        for (unsigned Depth = 0; Owner && Depth < 12; ++Depth) {
          auto Parents = Owner->getASTContext().getParents(*Child);
          if (Parents.empty()) break;
          const Stmt *Parent = Parents[0].get<Stmt>();
          if (!Parent) break;
          if (const auto *If = dyn_cast<IfStmt>(Parent)) {
            std::string GuardText;
            if (If->getCond()->getBeginLoc().isMacroID()) {
              bool Complete = true;
              GuardText = macroAnchor(If->getCond(),SM,LO,Cache,0,Complete);
              if (!Complete) GuardText.clear();
            } else {
              auto CA = SM.getSpellingLoc(If->getCond()->getBeginLoc());
              auto CB = SM.getSpellingLoc(If->getCond()->getEndLoc());
              if (CA.isValid() && CB.isValid() && SM.getFileID(CA) == SM.getFileID(CB)) {
                auto &Guard = sourceIdentity(CA, SM, LO, Cache);
                auto GA = tokenIndex(Guard, SM.getFileOffset(CA));
                auto GB = tokenIndex(Guard, SM.getFileOffset(CB));
                if (GB >= GA && GB-GA < 128) GuardText = tokenWindow(Guard,GA,GB+1);
              }
            }
            if (!GuardText.empty()) I.role += pack({Child == If->getThen() ? "then" : Child == If->getElse() ? "else" : "condition", GuardText});
            break;
          }
          Child = Parent;
        }
      }
    }
  }
  if (S && L.isMacroID()) {
    // Spelling endpoints may belong to a replacement body and an argument:
    // a raw contiguous source range would then swallow unrelated code.
    bool Complete = true;
    I.anchor = macroAnchor(S, SM, LO, Cache, 0, Complete);
    if (!Complete) I.anchor.clear();
    I.quality = Complete ? "ast" : "token";
  }
  if (I.anchor.empty()) {
    // Preprocessor/diagnostics without an AST use the written token line.
    // No source coordinates or diagnostic wording enter this fallback.
    auto Line = SM.getSpellingLineNumber(Physical);
    size_t A = Index, B = Index+1;
    auto Start = SM.getLocForStartOfFile(SM.getFileID(Physical));
    while (A && SM.getSpellingLineNumber(Start.getLocWithOffset(Source.offsets[A-1])) == Line) --A;
    while (B < Source.tokens.size() && SM.getSpellingLineNumber(Start.getLocWithOffset(Source.offsets[B])) == Line) ++B;
    I.anchor = tokenWindow(Source, A, B);
  }
  I.semantic = digest(pack({Checker.str(), Problem.str(), I.declaration, I.anchor, I.role, I.macro}));
  I.occurrence = digest(pack(Occurrence));
  return I;
}

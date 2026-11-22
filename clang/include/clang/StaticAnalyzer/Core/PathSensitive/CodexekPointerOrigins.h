//===- CodexekPointerOrigins.h - Pointer origin query SDK -------*- C++ -*-===//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef LLVM_CLANG_STATICANALYZER_CORE_PATHSENSITIVE_CODEXEKPOINTERORIGINS_H
#define LLVM_CLANG_STATICANALYZER_CORE_PATHSENSITIVE_CODEXEKPOINTERORIGINS_H
#include "clang/AST/Attr.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ProgramState.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/MemRegion.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/SValBuilder.h"
namespace clang::ento::codexek {
inline constexpr unsigned PointerOriginsVersion = 1;
enum class ArrayRelation { Unknown, SameArray, DifferentArrays };
struct OriginResult {
  const MemRegion *Object = nullptr;
  const MemRegion *Array = nullptr;
};
namespace detail {
inline const clang::ento::MemRegion *arrayDomain(clang::ento::SVal value, clang::ASTContext &ctx) {
  using namespace clang;
  using namespace clang::ento;
  const auto *region = value.getAsRegion();
  if (const auto *element = dyn_cast_or_null<ElementRegion>(region)) {
    const auto *parent = element->getSuperRegion();
    if (const auto *typed = dyn_cast<TypedValueRegion>(parent)) {
      const auto type = typed->getValueType();
      const auto *array = ctx.getAsArrayType(type);
      if (!ctx.hasSameUnqualifiedType(array ? array->getElementType() : type,
                                      element->getElementType()))
        return nullptr;
      return parent;
    }
    return isa<AllocaRegion>(parent) ? parent : nullptr;
  }
  // Stage C2: a heap allocation acts as its own array domain (per-instance
  // object identity, §4).
  if (const auto *symbolic = dyn_cast_or_null<SymbolicRegion>(region))
    return isa<HeapSpaceRegion>(symbolic->getSuperRegion()) ? region : nullptr;
  return isa_and_nonnull<VarRegion, FieldRegion, CompoundLiteralRegion>(region) ? region : nullptr;
}

inline bool concreteObject(const clang::ento::MemRegion *region) {
  using namespace clang;
  using namespace clang::ento;
  if (const auto *var = dyn_cast<VarRegion>(region)) {
    const auto *decl = var->getDecl();
    return !decl->hasAttr<AliasAttr>() && !decl->hasAttr<WeakAttr>() &&
           !decl->hasAttr<WeakRefAttr>() && !decl->hasAttr<AsmLabelAttr>();
  }
  // Stage C2: a heap allocation is a distinct object per call instance
  // (§4); its conjured symbol region never aliases another heap object.
  if (const auto *symbolic = dyn_cast<SymbolicRegion>(region))
    return isa<HeapSpaceRegion>(symbolic->getSuperRegion());
  return isa<AllocaRegion, CompoundLiteralRegion>(region);
}

inline bool disjointDomains(const clang::ento::MemRegion *left,
                                const clang::ento::MemRegion *right,
                                clang::ento::ProgramStateRef state, unsigned depth = 0) {
  using namespace clang;
  using namespace clang::ento;
  if (left == right || depth == 64) return false;
  const auto *leftBase = left->getBaseRegion(), *rightBase = right->getBaseRegion();
  if (leftBase != rightBase) {
    if (!concreteObject(leftBase) || !concreteObject(rightBase)) return false;
    const auto *leftVar = dyn_cast<VarRegion>(leftBase);
    const auto *rightVar = dyn_cast<VarRegion>(rightBase);
    if (leftVar && rightVar) {
      const auto *a = leftVar->getDecl(), *b = rightVar->getDecl();
      // Imported declarations can represent the same externally linked object.
      if (a->isExternallyVisible() && b->isExternallyVisible() &&
          a->getIdentifier() == b->getIdentifier())
        return false;
    }
    return true;
  }
  const auto *leftField = dyn_cast<FieldRegion>(left), *rightField = dyn_cast<FieldRegion>(right);
  if (leftField && rightField) {
    if (leftField->getSuperRegion() == rightField->getSuperRegion())
      return leftField->getDecl() != rightField->getDecl() &&
             !leftField->getDecl()->getParent()->isUnion() &&
             !rightField->getDecl()->getParent()->isUnion();
    return disjointDomains(leftField->getSuperRegion(),rightField->getSuperRegion(),
                                      state,depth+1);
  }
  // Mixed member/index shapes on a common base (s.x vs s.arr[2]): peel the
  // element indexing to reach the comparable member pair. Both branches
  // strictly reduce depth, so the shared guard bounds the recursion.
  const auto *leftElement = dyn_cast<ElementRegion>(left);
  const auto *rightElement = dyn_cast<ElementRegion>(right);
  if (leftField && rightElement)
    return disjointDomains(left, rightElement->getSuperRegion(), state, depth+1);
  if (leftElement && rightField)
    return disjointDomains(leftElement->getSuperRegion(), right, state, depth+1);
  const auto *a = dyn_cast<ElementRegion>(left), *b = dyn_cast<ElementRegion>(right);
  if (!a || !b || !state->getStateManager().getContext().hasSameUnqualifiedType(a->getElementType(),b->getElementType()))
    return false;
  if (a->getSuperRegion() != b->getSuperRegion())
    return disjointDomains(a->getSuperRegion(),b->getSuperRegion(),state,depth+1);
  auto &builder = state->getStateManager().getSValBuilder();
  const auto equal = builder.evalEQ(state,a->getIndex(),b->getIndex()).getAs<DefinedSVal>();
  return equal && !state->assume(*equal,true);
}

} // namespace detail
// Query only: native and summary-produced SVals use the same nominal C array
// hierarchy. Unknown values never acquire an invented source identity.
inline OriginResult resolveOrigin(ProgramStateRef State, SVal Value,
                                  QualType UseType,
                                  const StackFrame *) {
  if (!UseType->isPointerType()) return {};
  const auto *Array = detail::arrayDomain(Value, State->getStateManager().getContext());
  return {Array ? Array->getBaseRegion() : nullptr, Array};
}
inline ArrayRelation compareArrayOrigins(ProgramStateRef State,
                                         OriginResult Left, OriginResult Right) {
  if (!Left.Array || !Right.Array) return ArrayRelation::Unknown;
  if (Left.Array == Right.Array) return ArrayRelation::SameArray;
  return detail::disjointDomains(Left.Array,Right.Array,State)
             ? ArrayRelation::DifferentArrays : ArrayRelation::Unknown;
}
} // namespace clang::ento::codexek
#endif

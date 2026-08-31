"""Balanced selection from an already ranked H4 candidate pool.

No retrieval, scoring, or source-specific policy lives here.  The composer
only selects from the finite list it receives before ContextBuilder applies
its established token and neighbour safeguards.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable

from app.services.retrieval_service import RetrievalResult


class CompositionCategory:
    CONCRETE = "CONCRETE_CLASSROOM_MATERIAL"
    GUIDANCE = "PEDAGOGICAL_GUIDANCE"
    SUPPORTING = "SUPPORTING_RESOURCE"


@dataclass(frozen=True)
class CompositionPolicy:
    max_chunks: int = 6
    soft_max_per_document: int = 2
    primary_pool_size: int = 10
    max_pool_size: int = 20


@dataclass(frozen=True)
class CompositionResult:
    selected: list[RetrievalResult]
    candidate_pool_size: int
    category_counts: dict[str, int]
    concrete_material_available: bool
    diagnostics: dict[str, object]


class PedagogicalContextComposer:
    def __init__(self, policy: CompositionPolicy | None = None) -> None:
        self.policy = policy or CompositionPolicy()

    @staticmethod
    def category(result: RetrievalResult) -> str:
        if result.is_concrete_classroom_material:
            return CompositionCategory.CONCRETE
        if result.pedagogical_role == "METHODOLOGY":
            return CompositionCategory.GUIDANCE
        return CompositionCategory.SUPPORTING

    def compose(self, candidates: Iterable[RetrievalResult], *, intent: str) -> CompositionResult:
        pool = list(candidates)[:self.policy.max_pool_size]
        primary = pool[:self.policy.primary_pool_size]
        extended = pool[self.policy.primary_pool_size:]
        categories = {item.chunk_id: self.category(item) for item in pool}
        concrete_available = any(category == CompositionCategory.CONCRETE for category in categories.values())
        primary_concrete = any(categories[item.chunk_id] == CompositionCategory.CONCRETE for item in primary)
        extended_concrete = any(categories[item.chunk_id] == CompositionCategory.CONCRETE for item in pool)
        priority = self._category_priority(intent)
        selected: list[RetrievalResult] = []
        reasons: dict[int, str] = {}
        per_document: dict[int, int] = {}

        # Select category representatives first, retaining original H4 order.
        for category in priority:
            for item in primary:
                if categories[item.chunk_id] != category or not self._can_add(item, selected, per_document, enforce_document_cap=True):
                    continue
                selected.append(item)
                per_document[item.document_id] = per_document.get(item.document_id, 0) + 1
                reasons[item.chunk_id] = self._reason(category, intent)
                break

        # An extended candidate may supply a category absent from Top-10, but
        # receives only one bounded representative slot before normal filling.
        for category in [item for item in priority if not any(categories[selected_item.chunk_id] == item for selected_item in selected)]:
            for item in extended:
                if categories[item.chunk_id] != category or not self._can_add(item, selected, per_document, enforce_document_cap=False):
                    continue
                selected.append(item)
                per_document[item.document_id] = per_document.get(item.document_id, 0) + 1
                reasons[item.chunk_id] = "selected_extended_concrete_material" if category == CompositionCategory.CONCRETE else "extended_fallback_fill"
                break

        # Fill from H4 order, preferring a new document when one remains.
        for prefer_new_document, enforce_cap, reason in ((True, True, "selected_for_document_diversity"), (False, True, "fallback_fill"), (False, False, "fallback_fill_same_document")):
            for item in primary:
                if len(selected) >= self.policy.max_chunks:
                    break
                if item in selected or not self._can_add(item, selected, per_document, enforce_document_cap=enforce_cap):
                    continue
                if prefer_new_document and item.document_id in per_document:
                    continue
                selected.append(item)
                per_document[item.document_id] = per_document.get(item.document_id, 0) + 1
                reasons[item.chunk_id] = reason
            if len(selected) >= self.policy.max_chunks:
                break

        # Deep candidates are only considered for a missing preferred category
        # or after the primary window could not fill the context.
        needed_categories = [category for category in priority if not any(categories[item.chunk_id] == category for item in selected)]
        if extended and (len(selected) < self.policy.max_chunks or needed_categories):
            for item in extended:
                if len(selected) >= self.policy.max_chunks:
                    break
                category = categories[item.chunk_id]
                if category not in needed_categories and len(selected) >= self.policy.max_chunks:
                    continue
                if not self._can_add(item, selected, per_document, enforce_document_cap=False):
                    continue
                selected.append(item)
                per_document[item.document_id] = per_document.get(item.document_id, 0) + 1
                reasons[item.chunk_id] = "selected_extended_concrete_material" if category == CompositionCategory.CONCRETE and not primary_concrete else "extended_fallback_fill"

        selected_with_provenance = [
            replace(
                item,
                composition_category=categories[item.chunk_id],
                composition_rank=index,
                composition_selection_reason=reasons[item.chunk_id],
                composition_diversity_override=reasons[item.chunk_id] == "selected_for_document_diversity",
            )
            for index, item in enumerate(selected, start=1)
        ]
        counts = {category: sum(categories[item.chunk_id] == category for item in selected) for category in (CompositionCategory.CONCRETE, CompositionCategory.GUIDANCE, CompositionCategory.SUPPORTING)}
        underfilled = len(selected) < self.policy.max_chunks
        underfill_reason = "none" if not underfilled else "insufficient_unique_candidates" if len(selected) == len(pool) else "duplicate_rejection"
        return CompositionResult(selected_with_provenance, len(pool), counts, concrete_available, {
            "primary_pool_size": len(primary), "extended_pool_size": len(extended),
            "strong_concrete_available_primary_top10": primary_concrete,
            "strong_concrete_available_extended_top20": extended_concrete,
            "strong_concrete_selected_by_composer": any(item.is_concrete_classroom_material for item in selected),
            "composer_underfilled": underfilled, "underfill_reason": underfill_reason,
            "selected_from_extended_pool_count": sum((item.final_rank or item.rank) > self.policy.primary_pool_size for item in selected),
            "same_document_fallback_fill_count": sum(reason == "fallback_fill_same_document" for reason in reasons.values()),
        })

    def _can_add(self, item: RetrievalResult, selected: list[RetrievalResult], per_document: dict[int, int], *, enforce_document_cap: bool) -> bool:
        if len(selected) >= self.policy.max_chunks:
            return False
        if enforce_document_cap and per_document.get(item.document_id, 0) >= self.policy.soft_max_per_document:
            return False
        return not any(
            existing.document_id == item.document_id
            and existing.source_page_start == item.source_page_start
            and existing.source_page_end == item.source_page_end
            and existing.heading_context == item.heading_context
            for existing in selected
        )

    @staticmethod
    def _category_priority(intent: str) -> tuple[str, ...]:
        if intent in {"concrete_activity", "concrete_exercise", "role_play", "listening_activity", "task", "dialogue"}:
            return (CompositionCategory.CONCRETE, CompositionCategory.GUIDANCE, CompositionCategory.SUPPORTING)
        if intent == "methodology":
            return (CompositionCategory.GUIDANCE, CompositionCategory.SUPPORTING, CompositionCategory.CONCRETE)
        return (CompositionCategory.SUPPORTING, CompositionCategory.CONCRETE, CompositionCategory.GUIDANCE)

    @staticmethod
    def _reason(category: str, intent: str) -> str:
        if category == CompositionCategory.CONCRETE:
            return "selected_concrete_material"
        if category == CompositionCategory.GUIDANCE:
            return "selected_pedagogical_guidance"
        return "selected_supporting_resource" if intent != "general" else "selected_top_relevance"

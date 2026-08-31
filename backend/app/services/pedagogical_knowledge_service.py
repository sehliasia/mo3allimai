"""Deterministic orchestration of structured CEFR and semantic resources.

This is deliberately a retrieval boundary: it never calls an LLM and never
alters CEFR, chunk, vector, or ingestion data.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from time import perf_counter
from typing import Literal, Sequence

from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.services.cefr_knowledge_service import CEFRKnowledgeService
from app.services.context_builder import ContextBuilder, ContextSourceBlock
from app.services.retrieval_service import RetrievalFilters, RetrievalService
from app.services.pedagogical_retrieval_ranker import PedagogicalRankingRequest
from app.services.pedagogical_context_composer import PedagogicalContextComposer
from app.services.retrieval_pipeline import resolve_effective_retrieval_pipeline


logger = logging.getLogger(__name__)


SUPPORTED_CEFR_LEVELS = (
    "PRE-A1", "A1", "A2", "A2+", "B1", "B1+", "B2", "B2+", "C1", "C2"
)

# Exact CEFR scale names are kept as source labels.  A skill expands to a
# small, deterministic set rather than assuming one universal scale.
SKILL_SCALE_MAPPING: dict[str, tuple[str, ...]] = {
    "listening": (
        "Compréhension générale de l'oral",
        "Comprendre des annonces et des instructions orales",
        "Comprendre des médias audio ou signés et des enregistrements",
    ),
    "reading": (
        "Compréhension générale de l'écrit",
        "Lire pour s'orienter",
        "Lire pour s'informer et discuter",
    ),
    "speaking": (
        "Interaction orale générale",
        "Production orale générale",
        "Prise de parole",
    ),
    "writing": (
        "Production écrite générale",
        "Interaction écrite générale",
        "Écriture créative",
    ),
}

# Resource-query wording only.  It deliberately does not alter CEFR scale
# resolution, which remains exact and PostgreSQL-backed above.
_RESOURCE_SKILL_LABELS: dict[str, tuple[str, ...]] = {
    "speaking": ("expression orale", "interaction orale", "production orale"),
    "listening": ("compréhension orale", "écoute", "activité d'écoute"),
    "reading": ("compréhension écrite", "lecture"),
    "writing": ("expression écrite", "production écrite"),
}

_SKILL_ALIASES = {
    "oral_interaction": "speaking", "oral-interaction": "speaking",
    "oral_production": "speaking", "speaking": "speaking",
    "listening": "listening", "oral_comprehension": "listening",
    "oral-comprehension": "listening", "reading": "reading",
    "written_comprehension": "reading", "written-comprehension": "reading",
    "writing": "writing", "written_production": "writing",
    "written-production": "writing",
}


class PedagogicalRole:
    ACTIVITY = "ACTIVITY"
    EXERCISE = "EXERCISE"
    DIALOGUE = "DIALOGUE"
    TASK = "TASK"
    ASSESSMENT = "ASSESSMENT"
    METHODOLOGY = "METHODOLOGY"
    REFERENCE = "REFERENCE"
    OTHER = "OTHER"


class PedagogicalRequestIntent:
    CONCRETE_ACTIVITY = "concrete_activity"
    CONCRETE_EXERCISE = "concrete_exercise"
    ROLE_PLAY = "role_play"
    LISTENING_ACTIVITY = "listening_activity"
    METHODOLOGY = "methodology"
    GENERAL = "general"


_ROLE_MARKERS: dict[str, tuple[str, ...]] = {
    PedagogicalRole.DIALOGUE: ("dialogue", "حوار"),
    PedagogicalRole.EXERCISE: ("exercice", "exercise", "worksheet_exercise", "تمرين"),
    PedagogicalRole.TASK: ("jeu de rôle", "role play", "role-play", "tâche", "mission", "consigne", "مهمة", "تعليمات"),
    PedagogicalRole.ASSESSMENT: ("évaluation", "evaluation", "correction", "assessment", "تقويم", "تصحيح"),
    PedagogicalRole.ACTIVITY: ("activité", "activity", "نشاط"),
    PedagogicalRole.METHODOLOGY: ("méthodolog", "methodolog", "démarche", "déroulement", "pedagogical guide", "guide pédagogique"),
}

_CONCRETE_ROLE_PREFERENCES: dict[str, tuple[str, ...]] = {
    PedagogicalRequestIntent.CONCRETE_ACTIVITY: (
        PedagogicalRole.ACTIVITY, PedagogicalRole.EXERCISE, PedagogicalRole.DIALOGUE, PedagogicalRole.TASK,
    ),
    PedagogicalRequestIntent.CONCRETE_EXERCISE: (
        PedagogicalRole.EXERCISE, PedagogicalRole.ACTIVITY, PedagogicalRole.TASK, PedagogicalRole.DIALOGUE,
    ),
    PedagogicalRequestIntent.ROLE_PLAY: (
        PedagogicalRole.DIALOGUE, PedagogicalRole.TASK, PedagogicalRole.ACTIVITY, PedagogicalRole.EXERCISE,
    ),
    PedagogicalRequestIntent.LISTENING_ACTIVITY: (
        PedagogicalRole.DIALOGUE, PedagogicalRole.ACTIVITY, PedagogicalRole.EXERCISE, PedagogicalRole.TASK,
    ),
    PedagogicalRequestIntent.METHODOLOGY: (PedagogicalRole.METHODOLOGY, PedagogicalRole.REFERENCE),
}


class PedagogicalKnowledgeValidationError(ValueError):
    """A request cannot be represented safely by the orchestrator."""


@dataclass(frozen=True)
class PedagogicalKnowledgeRequest:
    cefr_level: str | None
    topic: str
    objective: str | None = None
    language: str | None = None
    skills: tuple[str, ...] = ()
    competencies: tuple[str, ...] = ()
    activity_type: str | None = None
    source_document_ids: tuple[int, ...] = ()
    # These constrain indexed source metadata only.  They deliberately differ
    # from the requested output language and learner CEFR level.
    source_language: str | None = None
    source_cefr_level: str | None = None
    retrieval_top_k: int | None = None
    include_cefr: bool = True
    include_resources: bool = True
    rerank: bool | None = None
    # Assistant follow-ups use this to distinguish a bounded conversational
    # retrieval context from an explicit literal topic supplied by a caller.
    topic_is_context: bool = False


@dataclass(frozen=True)
class CEFRSourceProvenance:
    document_id: int
    page_start: int | None
    page_end: int | None
    chunk_id: int
    source_order: int


@dataclass(frozen=True)
class PedagogicalCEFRDescriptor:
    level: str
    scale: str
    status: str
    descriptor_text: str | None
    reference_level: str | None
    sources: list[CEFRSourceProvenance]


@dataclass(frozen=True)
class PedagogicalCEFRMissing:
    level: str
    scale: str
    status: Literal["NO_STRUCTURED_DESCRIPTOR_FOUND"] = "NO_STRUCTURED_DESCRIPTOR_FOUND"


@dataclass(frozen=True)
class PedagogicalResourceBlock:
    source_number: int
    document_id: int
    document_title: str
    chunk_ids: list[int]
    page_start: int | None
    page_end: int | None
    heading_context: list[str]
    content_type: str
    structural_quality: str | None
    content: str
    requires_vision: bool
    image_not_interpreted: bool
    vector_scores: list[float]
    reranker_scores: list[float | None]
    original_ranks: list[int]
    reranked_ranks: list[int | None]


@dataclass(frozen=True)
class PedagogicalSource:
    source_type: Literal["cefr_structured", "pedagogical_resource"]
    document_id: int
    page_start: int | None
    page_end: int | None
    chunk_ids: list[int]
    descriptor_scale: str | None = None


@dataclass(frozen=True)
class PedagogicalContext:
    request_summary: dict[str, object]
    cefr_descriptors: list[PedagogicalCEFRDescriptor]
    cefr_missing: list[PedagogicalCEFRMissing]
    resource_blocks: list[PedagogicalResourceBlock]
    retrieved_count: int
    selected_count: int
    sources: list[PedagogicalSource]
    warnings: list[str]
    requires_vision_count: int


class PedagogicalKnowledgeService:
    """Composes existing CEFR, retrieval, and context services without generation."""

    def __init__(
        self,
        *,
        cefr: CEFRKnowledgeService,
        retrieval: RetrievalService | None = None,
        context_builder: ContextBuilder | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.cefr = cefr
        self.retrieval = retrieval
        self.context_builder = context_builder
        self.settings = settings or get_settings()
        self.pipeline = resolve_effective_retrieval_pipeline(self.settings)
        self.context_composer = PedagogicalContextComposer()

    @staticmethod
    def _normalized_level(level: str) -> str:
        normalized = level.strip().upper()
        if normalized not in SUPPORTED_CEFR_LEVELS:
            raise PedagogicalKnowledgeValidationError(
                "Unsupported CEFR level. Use one of: " + ", ".join(SUPPORTED_CEFR_LEVELS)
            )
        return normalized

    @staticmethod
    def _normal_skills(skills: Sequence[str]) -> tuple[list[str], list[str]]:
        recognized: list[str] = []
        unknown: list[str] = []
        for skill in skills:
            key = skill.strip().casefold().replace(" ", "_")
            mapped = _SKILL_ALIASES.get(key)
            if mapped is None:
                if skill.strip():
                    unknown.append(skill.strip())
            elif mapped not in recognized:
                recognized.append(mapped)
        return sorted(recognized), sorted(set(unknown), key=str.casefold)

    def _validate(self, request: PedagogicalKnowledgeRequest) -> tuple[str | None, list[str], list[str]]:
        level = self._normalized_level(request.cefr_level) if request.cefr_level else None
        if not request.topic.strip():
            raise PedagogicalKnowledgeValidationError("topic is required.")
        if request.retrieval_top_k is not None and request.retrieval_top_k < 1:
            raise PedagogicalKnowledgeValidationError("retrieval_top_k must be at least 1.")
        if any(document_id < 1 for document_id in request.source_document_ids):
            raise PedagogicalKnowledgeValidationError("source_document_ids must contain positive IDs.")
        if request.source_cefr_level is not None:
            self._normalized_level(request.source_cefr_level)
        skills, unknown_skills = self._normal_skills([*request.skills, *request.competencies])
        return level, skills, unknown_skills

    @staticmethod
    def _pedagogical_role(result: "RetrievalResult") -> str:
        """Classify retrieved evidence without persisting or changing canonical chunks."""
        evidence = " ".join([
            result.content_type,
            *result.heading_context,
            result.content,
        ]).casefold()
        if result.content_type == "worksheet_exercise":
            return PedagogicalRole.EXERCISE
        for role in (
            PedagogicalRole.DIALOGUE,
            PedagogicalRole.EXERCISE,
            PedagogicalRole.TASK,
            PedagogicalRole.ASSESSMENT,
            PedagogicalRole.ACTIVITY,
            PedagogicalRole.METHODOLOGY,
        ):
            if any(marker in evidence for marker in _ROLE_MARKERS[role]):
                return role
        return PedagogicalRole.REFERENCE if result.content_type in {"table", "list"} else PedagogicalRole.OTHER

    @staticmethod
    def _request_intent(request: PedagogicalKnowledgeRequest) -> str:
        """Use small request wording cues; this is not CEFR resolution or general NLP."""
        evidence = " ".join(value for value in (request.topic, request.objective, request.activity_type) if value).casefold()
        if any(marker in evidence for marker in ("jeu de rôle", "role play", "role-play", "لعب دور")):
            return PedagogicalRequestIntent.ROLE_PLAY
        if any(marker in evidence for marker in ("comment enseigner", "méthode", "method", "كيف أدرّس", "كيف ادرس")):
            return PedagogicalRequestIntent.METHODOLOGY
        if any(marker in evidence for marker in ("exercice", "exercise", "تمرين")):
            return PedagogicalRequestIntent.CONCRETE_EXERCISE
        if any(marker in evidence for marker in ("compréhension orale", "listening", "activité d'écoute", "استماع", "فهم المسموع")):
            return PedagogicalRequestIntent.LISTENING_ACTIVITY
        if any(marker in evidence for marker in ("activité", "activity", "tâche", "task", "نشاط")):
            return PedagogicalRequestIntent.CONCRETE_ACTIVITY
        return PedagogicalRequestIntent.GENERAL

    @classmethod
    def _soft_prioritize_pedagogical_roles(
        cls, results: list["RetrievalResult"], intent: str
    ) -> list["RetrievalResult"]:
        """Prefer concrete evidence only among similarly ranked semantic results."""
        preferred_roles = _CONCRETE_ROLE_PREFERENCES.get(intent)
        if not preferred_roles:
            return results
        role_priority = {role: index for index, role in enumerate(preferred_roles)}
        return sorted(
            results,
            key=lambda result: (
                # Never let a low-ranked concrete item leap across a semantic rank band.
                (max(result.rank, 1) - 1) // 3,
                role_priority.get(cls._pedagogical_role(result), len(preferred_roles)),
                result.rank,
                result.chunk_id,
            ),
        )

    @staticmethod
    def _resource_semantic_query(
        request: PedagogicalKnowledgeRequest, level: str | None, skills: list[str]
    ) -> str:
        """Build resource-search intent, independently from CEFR and UI language."""
        parts = ["enseignement de l'arabe", "activité pédagogique"]
        if level:
            parts.append(f"niveau {level}")
        topic_label = "contexte pédagogique" if request.topic_is_context else "thème"
        parts.append(f"{topic_label}: {request.topic.strip()}")
        if request.objective and request.objective.strip():
            parts.append(f"objective: {request.objective.strip()}")
        if skills:
            labels = tuple(
                dict.fromkeys(label for skill in skills for label in _RESOURCE_SKILL_LABELS[skill])
            )
            parts.append("compétences: " + ", ".join(labels))
        if request.activity_type and request.activity_type.strip():
            parts.append(f"activity: {request.activity_type.strip()}")
        return "; ".join(parts)

    def _cefr_context(
        self, db: Session, level: str, skills: list[str]
    ) -> tuple[list[PedagogicalCEFRDescriptor], list[PedagogicalCEFRMissing], list[PedagogicalSource]]:
        descriptors: list[PedagogicalCEFRDescriptor] = []
        missing: list[PedagogicalCEFRMissing] = []
        sources: list[PedagogicalSource] = []
        requested_scales = tuple(
            dict.fromkeys(scale for skill in skills for scale in SKILL_SCALE_MAPPING[skill])
        )
        for scale in requested_scales:
            rows = self.cefr.get_descriptors(db, level_code=level, scale_name=scale)
            if not rows:
                missing.append(PedagogicalCEFRMissing(level=level, scale=scale))
                continue
            for row in rows:
                provenance = [
                    CEFRSourceProvenance(
                        document_id=source.document_id,
                        page_start=source.page_start,
                        page_end=source.page_end,
                        chunk_id=source.chunk_id,
                        source_order=source.source_order,
                    )
                    for source in self.cefr.get_descriptor_sources(db, row.id)
                ]
                descriptor = PedagogicalCEFRDescriptor(
                    level=row.level.code,
                    scale=row.scale.name,
                    status=row.status,
                    descriptor_text=row.descriptor_text,
                    reference_level=row.reference_level.code if row.reference_level else None,
                    sources=provenance,
                )
                descriptors.append(descriptor)
                for source in provenance:
                    sources.append(PedagogicalSource(
                        source_type="cefr_structured", document_id=source.document_id,
                        page_start=source.page_start, page_end=source.page_end,
                        chunk_ids=[source.chunk_id], descriptor_scale=row.scale.name,
                    ))
        return descriptors, missing, sources

    @staticmethod
    def _resource_block(block: ContextSourceBlock) -> PedagogicalResourceBlock:
        return PedagogicalResourceBlock(
            source_number=block.source_number, document_id=block.document_id,
            document_title=block.document_title, chunk_ids=block.chunk_ids,
            page_start=block.page_start, page_end=block.page_end,
            heading_context=block.heading_context, content_type=block.content_type,
            structural_quality=block.structural_quality,
            content=block.content, requires_vision=block.requires_vision,
            image_not_interpreted=block.image_not_interpreted,
            vector_scores=block.vector_scores, reranker_scores=block.reranker_scores,
            original_ranks=block.original_ranks, reranked_ranks=block.reranked_ranks,
        )

    def build_context(self, db: Session, request: PedagogicalKnowledgeRequest) -> PedagogicalContext:
        started = perf_counter()
        level, skills, unknown_skills = self._validate(request)
        warnings = [f"Unmapped pedagogical skill: {skill}" for skill in unknown_skills]
        descriptors: list[PedagogicalCEFRDescriptor] = []
        missing: list[PedagogicalCEFRMissing] = []
        sources: list[PedagogicalSource] = []
        if request.include_cefr and level:
            cefr_started = perf_counter()
            descriptors, missing, sources = self._cefr_context(db, level, skills)
            structured_cefr_ms = round((perf_counter() - cefr_started) * 1000)
            if not skills:
                warnings.append("No mapped skill was supplied; structured CEFR lookup was not broadened.")
        elif request.include_cefr:
            structured_cefr_ms = 0
            warnings.append("No CEFR level was supplied; structured CEFR lookup was skipped.")
        else:
            structured_cefr_ms = 0

        resource_blocks: list[PedagogicalResourceBlock] = []
        retrieved_count = 0
        request_intent = self._request_intent(request)
        composition_diagnostics: dict[str, object] | None = None
        retrieval_trace: dict[str, object] = {}
        if request.include_resources:
            if self.retrieval is None or self.context_builder is None:
                raise PedagogicalKnowledgeValidationError("Retrieval and ContextBuilder are required when include_resources is true.")
            response = self.retrieval.search(
                db, self._resource_semantic_query(request, level, skills),
                top_k=request.retrieval_top_k or self.settings.rag_retrieval_top_k,
                # Explicit production profiles own reranker state; legacy callers
                # retain the established optional per-request behavior.
                rerank=(
                    (self.pipeline.reranker if request.rerank is None else request.rerank)
                    if self.pipeline.profile == "legacy"
                    else self.pipeline.reranker
                ),
                filters=RetrievalFilters(
                    document_ids=list(request.source_document_ids) or None,
                    language=(
                        request.source_language.strip()
                        if request.source_language and request.source_language.strip()
                        else None
                    ),
                    cefr_level=(
                        self._normalized_level(request.source_cefr_level)
                        if request.source_cefr_level
                        else None
                    ),
                ),
                pedagogical_request=PedagogicalRankingRequest(
                    intent=request_intent, cefr_level=level, skills=tuple(skills),
                ),
                composition_pool_size=(
                    self.pipeline.composition_pool_size
                    if self.pipeline.context_composition
                    else None
                ),
            )
            retrieved_count = len(response.results)
            retrieval_trace = dict(getattr(response, "pipeline_trace", None) or {})
            retrieval_trace.update({
                "pipeline_version": self.pipeline.pipeline_version,
                "pipeline_profile": self.pipeline.profile,
                "retrieval_mode": getattr(response, "retrieval_mode", "dense"),
                "dense_candidate_count": getattr(response, "dense_candidate_count", 0),
                "sparse_candidate_count": getattr(response, "sparse_candidate_count", 0),
                "union_candidate_count": getattr(response, "union_candidate_count", 0),
                "h4_ranked_count": len(response.results) if getattr(self.retrieval, "pedagogical_ranking_enabled", False) else 0,
                "fallback_used": bool(getattr(response, "fallback_reason", None)),
                "fallback_reason": getattr(response, "fallback_reason", None),
            })
            conflicting_chunks = [
                result.chunk_id for result in response.results
                if result.cefr_level and result.cefr_level.upper() != level
            ]
            if level and conflicting_chunks and descriptors:
                warnings.append(
                    "Structured CEFR descriptors remain authoritative; semantic resource CEFR metadata "
                    f"differs for chunks: {', '.join(map(str, conflicting_chunks))}."
                )
            prioritized_results = (
                response.results if (
                    getattr(self.retrieval, "pedagogical_ranking_enabled", False)
                    and getattr(response, "retrieval_mode", "dense") == "hybrid"
                )
                else self._soft_prioritize_pedagogical_roles(response.results, request_intent)
            )
            composition_enabled = self.pipeline.context_composition
            if composition_enabled:
                composition_pool = getattr(response, "composition_candidates", None) or prioritized_results
                compose_started = perf_counter()
                try:
                    composition = self.context_composer.compose(composition_pool, intent=request_intent)
                    prioritized_results = composition.selected
                    composition_diagnostics = {
                        "candidate_pool_size": composition.candidate_pool_size,
                        "category_counts": composition.category_counts,
                        "concrete_material_available": composition.concrete_material_available,
                        "selected_chunk_ids": [result.chunk_id for result in composition.selected],
                        **composition.diagnostics,
                    }
                except Exception:
                    composition_diagnostics = {
                        "candidate_pool_size": len(composition_pool),
                        "fallback_used": True,
                        "fallback_reason": "context_composition_fallback",
                    }
                    retrieval_trace.update({"fallback_used": True, "fallback_reason": "context_composition_fallback"})
                retrieval_trace["h5_composition_ms"] = round((perf_counter() - compose_started) * 1000)
            context_started = perf_counter()
            assembled = self.context_builder.build(response.query, prioritized_results, db=db)
            retrieval_trace.update({
                "context_builder_ms": round((perf_counter() - context_started) * 1000),
                "composition_pool_size": len(getattr(response, "composition_candidates", None) or []),
                "composer_selected_count": len(prioritized_results),
                "final_context_chunk_count": len(assembled.included_chunk_ids),
                "final_context_token_estimate": assembled.estimated_token_count,
                "distinct_document_count": len({block.document_id for block in assembled.source_blocks}),
                "strong_concrete_available_primary": (composition_diagnostics or {}).get("strong_concrete_available_primary_top10"),
                "strong_concrete_available_extended": (composition_diagnostics or {}).get("strong_concrete_available_extended_top20"),
                "strong_concrete_retained": any(result.is_concrete_classroom_material for result in assembled.included_results),
                "extended_pool_used": bool((composition_diagnostics or {}).get("selected_from_extended_pool_count")),
                "strong_concrete_selected": (composition_diagnostics or {}).get("strong_concrete_selected"),
                "neighbor_count": sum(
                    1 for result in assembled.included_results if getattr(result, "neighbor_of", None) is not None
                ),
                "structured_cefr_ms": structured_cefr_ms,
                "total_pedagogical_knowledge_ms": round((perf_counter() - started) * 1000),
            })
            logger.info("pedagogical_retrieval_trace=%s", retrieval_trace)
            resource_blocks = [self._resource_block(block) for block in assembled.source_blocks]
            warnings.extend(assembled.warnings)
            for block in resource_blocks:
                sources.append(PedagogicalSource(
                    source_type="pedagogical_resource", document_id=block.document_id,
                    page_start=block.page_start, page_end=block.page_end,
                    chunk_ids=block.chunk_ids,
                ))

        return PedagogicalContext(
            request_summary={
                "cefr_level": level, "topic": request.topic.strip(), "objective": request.objective,
                "language": request.language, "skills": skills, "activity_type": request.activity_type,
                "pedagogical_request_intent": request_intent,
                "source_document_ids": list(request.source_document_ids),
                "source_language": request.source_language,
                "source_cefr_level": self._normalized_level(request.source_cefr_level)
                if request.source_cefr_level else None,
                "context_composition": composition_diagnostics,
                "retrieval_trace": retrieval_trace,
                "pipeline_version": self.pipeline.pipeline_version,
                "pipeline_profile": self.pipeline.profile,
                "include_cefr": request.include_cefr, "include_resources": request.include_resources,
            },
            cefr_descriptors=descriptors, cefr_missing=missing, resource_blocks=resource_blocks,
            retrieved_count=retrieved_count, selected_count=len(resource_blocks), sources=sources,
            warnings=list(dict.fromkeys(warnings)),
            requires_vision_count=sum(1 for block in resource_blocks if block.requires_vision),
        )

"""Intelligent, KB-first exercise search.

The search understands natural-language teacher queries ("exercices A1 de
vocabulaire sur l'école") WITHOUT any LLM call: constraints (level, skill,
type) are parsed deterministically and only kept when the user actually states
them. Retrieval reuses the existing hybrid pipeline (dense + BM25 + RRF via
RetrievalService) and multi-chunk reconstruction (ContextBuilder merges
adjacent chunks and expands to immediate neighbors). Detected exercises are
typed and ranked with configurable weights, and facets are computed from the
results actually found — never from imagined theme lists.

Hard guarantee: search never calls the LLM (llm_calls is always 0).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.models.knowledge_document import KnowledgeChunk
from app.schemas.exercise_generator import (
    ExerciseFilterFacet,
    ExerciseSearchIn,
    ExerciseSearchItem,
    ExerciseSearchMeta,
    ExerciseSearchOut,
)
from app.services.context_builder import ContextSourceBlock
from app.services.exercise_cefr import (
    detect_explicit_level,
    estimate_level,
    normalize_level,
)
from app.services.exercise_detection import _TITLE_RE, score_exercise
from app.services.retrieval_service import RetrievalFilters, RetrievalService


_LEVEL_RE = re.compile(
    r"\b(?:(?:niveau|niv\.|level)\s*)?(?:pre-)?[A-C][12]\b|"
    r"\b(?:مستوى|المستوى)\s*[أبج][12]",
    re.IGNORECASE,
)
_ARABIC_LEVEL_TAIL = re.compile(r"[أبج][12]")
_ARABIC_LETTER_MAP = {"أ": "A", "ب": "B", "ج": "C"}

_SKILL_LEXICON: dict[str, tuple[str, ...]] = {
    "reading": ("compréhension écrite", "lecture", "قراءة", "فهم المقروء", "reading"),
    "listening": ("compréhension orale", "écoute", "استماع", "فهم المسموع", "listening"),
    "speaking": ("expression orale", "interaction orale", "production orale", "تحدث", "تعبير شفهي", "speaking"),
    "writing": ("expression écrite", "production écrite", "écriture", "كتابة", "تعبير كتابي", "writing"),
    "vocabulary": ("vocabulaire", "lexique", "mots", "مفردات", "كلمات", "vocabulary"),
    "grammar": ("grammaire", "conjugaison", "syntaxe", "قواعد", "تصريف", "grammaire"),
    "spelling": ("orthographe", "dictée", "إملاء", "إملاء", "orthographe"),
}
_SKILL_LABELS: dict[str, str] = {
    "reading": "Compréhension écrite",
    "listening": "Compréhension orale",
    "speaking": "Expression orale",
    "writing": "Expression écrite",
    "vocabulary": "Vocabulaire",
    "grammar": "Grammaire",
    "spelling": "Orthographe",
}

_TYPE_LEXICON: dict[str, tuple[str, ...]] = {
    "qcm": ("qcm", "choix multiple", "q.c.m", "اختيار من متعدد", "test à choix"),
    "true_false": ("vrai ou faux", "vrai/faux", "صواب وخطأ", "صح وخطأ", "الصحيح والخطأ"),
    "matching": ("relier", "apparier", "associer", "correspondance", "وصل", "طابق", "اربط"),
    "fill_blank": ("compléter", "trous", "complète", "الفراغ", "أكمل", "املأ"),
    "ordering": ("remettre en ordre", "réordonner", "ranger", "رتب", "رتّب"),
    "transformation": ("transformation", "transforme", "conjugue", "حوّل", "صرف"),
    "production": ("production écrite", "rédaction", "expression écrite", "تعبير كتابي", "أنتج"),
    "open_question": ("question ouverte", "réponds", "question libre", "سؤال مفتوح", "أجب"),
}

_FILLER_WORDS = (
    "je", "cherche", "cherché", "cherchez", "trouve", "trouvez", "donne", "donnez",
    "exercices", "exercice", "des", "les", "la", "le", "l", "les", "d'une", "d'un", "de", "du", "des",
    "sur", "à", "au", "aux", "en", "pour", "avec", "niveau", "niveaux", "m'", "me", "s'il", "s'il",
    "plait", "plaît", "svp", "s'ilvousplaît", "activités", "activité", "travailler", "travaille",
    "ainsi", "faut", "savoir", "peux",
    "أبحث", "عن", "تمارين", "تمرين", "في", "مستوى", "المستوى", "هام", "من فضلك",
)
_LEVEL_HINT_WORDS = {"a1", "a2", "b1", "b2", "c1", "c2"}


@dataclass
class StructuredExerciseQuery:
    raw_query: str
    level: str | None = None
    skills: list[str] = field(default_factory=list)
    exercise_type: str | None = None
    theme_tokens: tuple[str, ...] = ()
    objective: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "level": self.level,
            "skills": list(self.skills),
            "exercise_type": self.exercise_type,
            "theme_tokens": list(self.theme_tokens),
        }


def parse_query(text: str) -> StructuredExerciseQuery:
    """Deterministic constraint extraction. Constraints stay None/empty unless
    the user explicitly states them — nothing is invented."""
    raw = " ".join(text.strip().split())
    if not raw:
        raise ValueError("query is required")

    level = None
    for match in _LEVEL_RE.finditer(raw):
        token = match.group(0)
        tail = _ARABIC_LEVEL_TAIL.search(token)
        if tail:
            simple = tail.group(0)
        else:
            simple = re.sub(r"^(?:niveau|niv\.|level)\s*", "", token, flags=re.IGNORECASE)
        simple = simple.strip()
        if simple[:1] in _ARABIC_LETTER_MAP:
            simple = f"{_ARABIC_LETTER_MAP[simple[:1]]}{simple[1:]}"
        simple = simple.upper().replace("PRE-", "")
        if simple.casefold() in _LEVEL_HINT_WORDS:
            level = simple
            break

    skills: list[str] = []
    exercise_type: str | None = None
    remaining = raw
    lowered = raw.casefold()
    for skill, markers in _SKILL_LEXICON.items():
        if any(marker.casefold() in lowered for marker in markers):
            skills.append(skill)
    for type_name, markers in _TYPE_LEXICON.items():
        if any(marker.casefold() in lowered for marker in markers):
            exercise_type = type_name
            break

    # Remove constraint tokens and filler words to recover the theme.
    stripped = re.sub(_LEVEL_RE, " ", raw)
    for markers in (*_SKILL_LEXICON.values(), *_TYPE_LEXICON.values()):
        for marker in markers:
            if marker.casefold() in stripped.casefold():
                stripped = re.sub(re.escape(marker), " ", stripped, flags=re.IGNORECASE)
    tokens = []
    for token in re.findall(r"\w+", stripped, re.UNICODE):
        key = token.casefold()
        key = "".join(character for character in key if character.isalnum())
        if key and key not in _FILLER_WORDS and key not in _LEVEL_HINT_WORDS:
            tokens.append(key)
    return StructuredExerciseQuery(
        raw_query=raw, level=level, skills=skills, exercise_type=exercise_type,
        theme_tokens=tuple(dict.fromkeys(tokens))[:6],
    )


def build_retrieval_query(parsed: StructuredExerciseQuery) -> str:
    parts = ["exercices", "fiche d'exercices", "activité d'entraînement"]
    if parsed.level:
        parts.append(f"niveau {parsed.level}")
    if parsed.skills:
        labels = ", ".join(_SKILL_LABELS[skill] for skill in parsed.skills)
        parts.append(f"compétence : {labels}")
    if parsed.exercise_type:
        parts.append(f"type : {parsed.exercise_type}")
    theme = " ".join(parsed.theme_tokens) if parsed.theme_tokens else parsed.raw_query
    parts.append(f"thème : {theme}")
    return "; ".join(parts)


def _detect_skill(content: str, heading_context: Sequence[str], requested: Sequence[str]) -> tuple[str, str | None, str]:
    """Return (skill_label, skill_key, skill_source). Explicit only when the
    document text itself states a skill; otherwise inferred from the request."""
    evidence = " ".join([*heading_context, content]).casefold()
    for skill, markers in _SKILL_LEXICON.items():
        if any(marker.casefold() in evidence for marker in markers):
            return _SKILL_LABELS[skill], skill, "explicit"
    for skill in requested:
        if any(marker.casefold() in evidence for marker in _SKILL_LEXICON[skill]):
            return _SKILL_LABELS[skill], skill, "explicit"
    if requested:
        return _SKILL_LABELS[requested[0]], requested[0], "inferred"
    return "", None, "inferred"


def _detect_theme_tokens(content: str, heading_context: Sequence[str]) -> str | None:
    heading = " ".join(heading_context).strip()
    if heading:
        return heading[:120]
    return None


def _extract_exercise_title(content: str, heading_context: Sequence[str]) -> str:
    """Prefer the document's own exercise title line over a generic section."""
    from app.services.exercise_detection import _TITLE_RE
    for line in content.splitlines():
        if _TITLE_RE.search(line):
            return re.sub(r"\s+", " ", line).strip(" :—–")
    for heading in heading_context:
        if heading.strip():
            return heading.strip()
    return "Exercice"


def _normalize_retrieval_score(rank: int | None, top_k: int) -> float:
    if rank is None or rank < 1:
        return 0.0
    return max(0.0, 1.0 - ((rank - 1) / max(top_k, 1)))


# A chunk is an explicit exercise-start title only when it is a *title line* and
# we have not yet seen the directive/items that make an exerize real. Merging
# "تمرين 1" + "تمرين 2" is forbidden: a new title always closes the section.
_EXERCISE_TITLE_ONLY_RE = _TITLE_RE


def _merged_source_block(previous: ContextSourceBlock, result) -> ContextSourceBlock:
    """Recreate ContextBuilder's frozen-block merge for an appended chunk."""
    return ContextSourceBlock(
        **{**previous.__dict__,
           "chunk_ids": [*previous.chunk_ids, result["chunk_id"]],
           "page_start": min(filter(None, [previous.page_start, result["page_start"]]), default=None),
           "page_end": max(filter(None, [previous.page_end, result["page_end"]]), default=None),
           "has_image": previous.has_image or result["has_image"],
           "requires_vision": previous.requires_vision or result["requires_vision"],
           "image_not_interpreted": previous.image_not_interpreted or result["requires_vision"],
           "content": previous.content + "\n\n" + result["content"],
           "estimated_token_count": previous.estimated_token_count + result["token_count"]},
    )


def enrich_exercise_blocks(
    blocks: Sequence[ContextSourceBlock],
    *,
    db: Session,
    window: int = 4,
    max_tokens: int = 1800,
) -> list[ContextSourceBlock]:
    """In-memory reconstruction of fragmented exercises (no reindexing).

    ContextBuilder expands to at most one immediate neighbor for the top-2
    candidates, which does not reliably put an exercise's title, directive and
    options back together. This pass re-groups the *already retrieved* chunks by
    walking their same-document ``chunk_index`` neighbourhood in memory and
    merging contiguous rows into a single complete block. It never creates or
    edits stored chunks, never invents provenance, and never crosses a new
    explicit exercise title (so "تمرين 1" + "تمرين 2" are never fused).
    """
    if not blocks:
        return list(blocks)

    # Fetch the chunk_index neighbourhood of every block's member chunk ids so
    # we can walk the *stored* same-document rows (including fragments not
    # themselves returned by retrieval) and reunite them in memory.
    member_ids = {cid for block in blocks for cid in block.chunk_ids}
    anchors = db.scalars(
        select(KnowledgeChunk)
        .where(KnowledgeChunk.id.in_(member_ids))
    ).all()
    by_id = {row.id: row for row in anchors}
    # A small query per distinct (document, area) to pull the surrounding rows.
    query_by_area: dict[tuple[int, int, int], set[int]] = {}
    for row in anchors:
        query_by_area[(row.document_id, row.chunk_index - window, row.chunk_index + window)] = set()
    if query_by_area:
        window_rows = db.scalars(
            select(KnowledgeChunk)
            .where(
                KnowledgeChunk.document_id.in_({area[0] for area in query_by_area}),
                KnowledgeChunk.chunk_index >= min(a[1] for a in query_by_area),
                KnowledgeChunk.chunk_index <= max(a[2] for a in query_by_area),
            )
        ).all()
    else:
        window_rows = []
    doc_to_indices: dict[int, set[int]] = {}
    index_to_row: dict[tuple[int, int], KnowledgeChunk] = {}
    for row in window_rows:
        doc_to_indices.setdefault(row.document_id, set()).add(row.chunk_index)
        index_to_row[(row.document_id, row.chunk_index)] = row

    def _exercise_title_only(content: str) -> bool:
        for line in content.splitlines():
            if _EXERCISE_TITLE_ONLY_RE.search(line):
                return True
        return False

    enriched: list[ContextSourceBlock] = []
    used_ids: set[int] = set()
    for block in blocks:
        if block.chunk_ids and block.chunk_ids[0] in used_ids:
            continue
        working = block
        for cid in block.chunk_ids:
            row = by_id.get(cid)
            if row is None:
                continue
            document_id = row.document_id
            indices = sorted(doc_to_indices.get(document_id, set()))
            base_index = row.chunk_index
            for offset in range(-window, window + 1):
                index = base_index + offset
                if index == base_index:
                    continue
                if index not in indices:
                    continue
                neighbor = index_to_row.get((document_id, index))
                if neighbor is None or neighbor.id in used_ids:
                    continue
                metadata = neighbor.chunk_metadata or {}
                if (metadata.get("structural_quality") or neighbor.quality_status) == "layout_unreliable":
                    continue
                # Stop when the neighbour opens a *new* exercise section once we
                # already hold the current exercise (title already seen).
                if _exercise_title_only(neighbor.content) and len(working.chunk_ids) >= 1:
                    break
                merged = _merged_source_block(working, {
                    "chunk_id": neighbor.id,
                    "page_start": neighbor.source_page_start,
                    "page_end": neighbor.source_page_end,
                    "has_image": bool(metadata.get("has_image")),
                    "requires_vision": bool(metadata.get("requires_vision")),
                    "content": neighbor.content,
                    "token_count": neighbor.token_count,
                })
                if merged.estimated_token_count > max_tokens and working.chunk_ids:
                    break
                working = merged
                used_ids.add(neighbor.id)
        used_ids.update(working.chunk_ids)
        enriched.append(working)

    # Re-number the source blocks so the rendered [SOURCE n] labels stay valid.
    for number, item in enumerate(enriched, start=1):
        enriched[number - 1] = ContextSourceBlock(
            **{**item.__dict__, "source_number": number},
        )
    return enriched


class ExerciseSearchService:
    """0-LLM hybrid search over the teacher knowledge base."""

    def __init__(
        self,
        *,
        retrieval: RetrievalService,
        context_builder=None,
        settings: Settings | None = None,
    ) -> None:
        self.retrieval = retrieval
        self.context_builder = context_builder
        self.settings = settings or get_settings()

    def _blended_score(self, *, block, retrieval_item) -> float:
        weights = self.settings.exercise_ranking_weights
        total = (
            weights.get("retrieval", 0.0) * retrieval_item["retrieval_norm"]
            + weights.get("structure", 0.0) * block["detection"].confidence
            + weights.get("theme", 0.0) * block["theme_match"]
            + weights.get("level", 0.0) * block["level_match"]
            + weights.get("skill", 0.0) * block["skill_match"]
            + weights.get("type", 0.0) * block["type_match"]
            + weights.get("source", 0.0) * block["source_quality"]
        )
        return round(total, 4)

    def search(
        self, db: Session | None, request: ExerciseSearchIn,
        *, expanded_blocks: Sequence | None = None,
        prebuilt_response: object | None = None,
    ) -> ExerciseSearchOut:
        parsed = parse_query(request.query)
        # Hard refinements supplied by the UI take precedence; they are strict.
        level = request.level or parsed.level
        if level and not normalize_level(level):
            level = None
        skills = list(request.skills) or parsed.skills
        exercise_type = request.exercise_type or parsed.exercise_type

        if expanded_blocks is not None:
            # Reuse blocks already retrieved+expanded by an upstream pipeline
            # (e.g. the endpoint's LLM extraction stage). No second retrieval.
            response = prebuilt_response
            pool = list(getattr(response, "results", []) or [])
            assembly = list(expanded_blocks)
            query = build_retrieval_query(StructuredExerciseQuery(
                raw_query=request.query, level=level, skills=skills,
                exercise_type=exercise_type, theme_tokens=parsed.theme_tokens,
            ))
        else:
            query = build_retrieval_query(StructuredExerciseQuery(
                raw_query=request.query, level=level, skills=skills,
                exercise_type=exercise_type, theme_tokens=parsed.theme_tokens,
            ))
            response = self.retrieval.search(
                db,
                query,
                top_k=self.settings.exercise_search_top_k,
                rerank=False,
                filters=RetrievalFilters(
                    document_ids=list(request.source_document_ids) or None,
                ),
            )

            pool = response.results
            if self.context_builder is not None and pool:
                built = self.context_builder.build(
                    response.query, pool, db=db,
                )
                assembly = built.source_blocks
            else:
                assembly = []

        results: list[dict[str, object]] = []
        raw_facets: dict[str, dict[str, int]] = {
            "levels": {}, "themes": {}, "skills": {}, "types": {}, "documents": {},
        }
        for block in assembly:
            chunk_to_result = {result.chunk_id: result for result in pool}
            members = [chunk_to_result[cid] for cid in block.chunk_ids if cid in chunk_to_result]
            if not members:
                members = [pool[0]] if pool else []
            detection = score_exercise(
                block.content, block.heading_context,
                threshold=self.settings.exercise_detection_threshold,
            )
            if not detection.is_exercise:
                continue

            # The level stated in the exercise's own heading/text is
            # authoritative. Only when the source does NOT state a level do we
            # fall back to the document-level indexed metadata (an admin may tag
            # a whole workbook A2 while individual exercises are A1). This keeps
            # the strict "never relabel a provably-other-level exercise" rule
            # without letting a broad document tag hide relevant items.
            explicit_level = detect_explicit_level(
                content=block.content, heading_context=block.heading_context,
            )
            if explicit_level is None and members:
                explicit_level = detect_explicit_level(
                    content=block.content, heading_context=block.heading_context,
                    indexed_cefr_level=members[0].cefr_level,
                )
            if level is not None and explicit_level and explicit_level != level:
                # Strict CEFR filter: never present a provably-other-level chunk
                # as the requested level.
                continue
            estimated_level, estimate_confidence = estimate_level(block.content)
            if explicit_level:
                item_level, level_source = explicit_level, "explicit"
            elif estimated_level:
                item_level, level_source = estimated_level, "inferred"
            else:
                item_level, level_source = (level or "A1"), "inferred"
                estimate_confidence = 0.0

            skill_label, skill_key, skill_source = _detect_skill(
                block.content, block.heading_context, skills,
            )
            theme_label = _detect_theme_tokens(block.content, block.heading_context)
            type_source = "explicit" if block.heading_context else "inferred"

            best_member = min(members, key=lambda m: m.fused_rank or m.rank)
            retrieval_norm = _normalize_retrieval_score(best_member.fused_rank or best_member.rank, response.top_k)
            has_theme_tokens = bool(parsed.theme_tokens and any(
                token in block.content.casefold() for token in parsed.theme_tokens
            ))
            source_quality = 0.0 if block.structural_quality == "layout_unreliable" else 1.0

            item_score_block = {
                "detection": detection,
                "theme_match": 1.0 if has_theme_tokens else 0.0,
                "level_match": 1.0 if level is not None and item_level == level else 0.0,
                "skill_match": (
                    1.0 if (skills and skill_key in skills)
                    else (0.0 if skills else 0.5)
                ),
                "type_match": (
                    1.0 if (exercise_type and detection.exercise_type == exercise_type)
                    else (0.0 if exercise_type else 0.5)
                ),
                "source_quality": source_quality,
                "retrieval_norm": retrieval_norm,
            }
            result_item = ExerciseSearchItem(
                title=_extract_exercise_title(block.content, block.heading_context),
                skill=skill_label,
                skill_source=skill_source,
                exercise_type=detection.exercise_type or exercise_type or "mixed",
                type_source=type_source,
                prompt=block.content.strip(),
                context="",
                answer_expectation=None,
                level=item_level,
                level_source=level_source,
                theme=theme_label or " ".join(parsed.theme_tokens),
                theme_source="explicit" if theme_label else "inferred",
                status="kb_original",
                document_title=block.document_title,
                document_id=block.document_id,
                page_start=block.page_start,
                page_end=block.page_end,
                chunk_ids=list(block.chunk_ids),
                heading_context=list(block.heading_context),
                theme_label=theme_label,
                skill_label=skill_label,
                summary_score=0.0,
            )
            raw_facets["levels"][f"{item_level} ({level_source})"] = raw_facets["levels"].get(f"{item_level} ({level_source})", 0) + 1
            raw_facets["themes"][theme_label or "Sans thème"] = raw_facets["themes"].get(theme_label or "Sans thème", 0) + 1
            raw_facets["skills"][skill_label or "Sans compétence"] = raw_facets["skills"].get(skill_label or "Sans compétence", 0) + 1
            raw_facets["types"][detection.exercise_type or exercise_type or "mixed"] = raw_facets["types"].get(detection.exercise_type or exercise_type or "mixed", 0) + 1
            raw_facets["documents"][block.document_title] = raw_facets["documents"].get(block.document_title, 0) + 1
            results.append({"item": result_item, "score": item_score_block, "retrieval": best_member, "source_quality": source_quality})

        ranked = sorted(
            results,
            key=lambda entry: self._blended_score(block=entry["score"], retrieval_item=entry["score"]),
            reverse=True,
        )[: request.limit]
        for position, entry in enumerate(ranked):
            entry["item"].summary_score = round(entry["score"]["retrieval_norm"] + 0.0, 4)
            entry["item"].summary_score = self._blended_score(block=entry["score"], retrieval_item=entry["score"])

        facets = {
            key: [ExerciseFilterFacet(value=value, count=count) for value, count in sorted(items.items(), key=lambda kv: (-kv[1], str.casefold(kv[0])))]
            for key, items in raw_facets.items()
        }
        return ExerciseSearchOut(
            query=request.query,
            items=[entry["item"] for entry in ranked],
            total=len(ranked),
            facets=facets,
            meta=ExerciseSearchMeta(
                llm_calls=0,
                retrieval_mode=getattr(response, "retrieval_mode", "dense"),
                dense_candidate_count=getattr(response, "dense_candidate_count", 0),
                sparse_candidate_count=getattr(response, "sparse_candidate_count", 0),
                union_candidate_count=getattr(response, "union_candidate_count", 0),
                candidate_blocks=len(assembly),
                detected_blocks=len(results),
                parsed=parsed.to_dict(),
            ),
        )
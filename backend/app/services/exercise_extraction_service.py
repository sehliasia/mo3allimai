"""LLM-based exercise extraction (deep RAG + semantic structuring).

Retrieval stays 0 LLM: this service runs AFTER the hybrid search and the
multi-chunk expansion, on a token-bounded set of candidate blocks, and is the
place where the model is used to *understand* passages, decide whether they
contain a real exercise (never just the word "exercice"/"تمرين"), reconstruct
an exercise spread across several chunks, structure it and classify it.

Hard invariants:
- The LLM NEVER creates or modifies pedagogical content here. It only extracts
  and structures what the knowledge base provides. (Because extraction is
  verbatim from the KB, no Fusha guard applies here: the Arabic-guard belongs
  to the adaptation/generation services that may create or rewrite content.)
- Provenance (document, pages, chunk_ids) is NEVER invented: any value returned
  by the model that does not match an actual provided block is dropped/null.
- The result always carries status="kb_original".
- A provably-A2 source is never relabelled A1; a strict requested level is a
  hard filter (documented CEFR layer via exercise_cefr).
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Callable, Sequence

from app.core.config import Settings, get_settings
from app.schemas.exercise_generator import (
    ExerciseSearchIn,
    ExerciseSearchItem,
)
from app.services.context_builder import ContextSourceBlock
from app.services.exercise_cefr import (
    detect_explicit_level,
    estimate_level,
    normalize_level,
)
from app.services.exercise_detection import (
    classify_exercise_type,
)
from app.services.exercise_generation_service import (
    ExerciseGenerationService,
    _NO_PROVIDER_RETRY,
    _RATE_LIMIT_USER_MESSAGE,
)
from app.services.llm_provider import LLMGenerationOptions, LLMProvider, LLMProviderError

logger = logging.getLogger(__name__)

_MAX_PROVIDER_RETRIES = 1
_RATE_LIMIT_FALLBACK_DELAY_SECONDS = 2.0

_ACCEPTED_TYPES = {
    "qcm", "true_false", "matching", "fill_blank", "ordering",
    "transformation", "production", "open_question", "mixed",
}


class ExerciseExtractionError(RuntimeError):
    pass


class ExerciseExtractionRateLimitError(ExerciseExtractionError):
    pass


@dataclass
class ExtractionCandidate:
    """A token-bounded block with resolved provenance, ready for the LLM."""
    text: str
    chunk_ids: tuple[int, ...]
    document_id: int | None
    document_title: str | None
    page_start: int | None
    page_end: int | None
    heading_context: tuple[str, ...]
    structural_quality: str | None


@dataclass
class ExtractedExercise:
    title: str | None
    instruction: str
    exercise_type: str
    items: list[dict[str, object]]
    expected_answer: str | None
    level: str | None
    level_source: str
    theme: str | None
    skill: str | None
    source_chunk_ids: list[int]
    source_document_id: int | None
    source_document_title: str | None
    source_pages: list[int]
    reasons: tuple[str, ...] = ()

    def dedup_key(self) -> tuple[object, ...]:
        norm = " ".join((self.instruction or "").split()).casefold()
        return (self.source_document_id, norm[:160])


def _estimate_tokens(text: str) -> int:
    # ~4 chars/token is a safe upper bound for mixed French/Arabic.
    return max(1, round(len(text or "") / 4.0) + 1)


def _render_passage(block: ContextSourceBlock) -> ExtractionCandidate:
    heading = " > ".join(block.heading_context) if block.heading_context else ""
    header = heading or block.document_title or ""
    text = f"{header}\n{block.content}".strip() if header else block.content.strip()
    return ExtractionCandidate(
        text=text,
        chunk_ids=tuple(int(c) for c in block.chunk_ids),
        document_id=block.document_id,
        document_title=block.document_title,
        page_start=block.page_start,
        page_end=block.page_end,
        heading_context=tuple(block.heading_context or ()),
        structural_quality=block.structural_quality,
    )


def _reduce_candidates(
    blocks: Sequence[ContextSourceBlock],
    *,
    context_k: int,
    max_context_tokens: int,
) -> list[ExtractionCandidate]:
    """Rank-provided blocks are already ordered by relevance; reduce the LLM
    input to a bounded window while keeping complete (multi-chunk) blocks."""
    selected: list[ExtractionCandidate] = []
    used_tokens = 0
    for block in blocks:
        if len(selected) >= context_k:
            break
        candidate = _render_passage(block)
        if _estimate_tokens(candidate.text) > max_context_tokens // 2:
            continue
        if used_tokens + _estimate_tokens(candidate.text) > max_context_tokens:
            break
        selected.append(candidate)
        used_tokens += _estimate_tokens(candidate.text)
    return selected


def _resolve_provenance(
    item: dict[str, object],
    *,
    candidates: Sequence[ExtractionCandidate],
) -> dict[str, object]:
    """Accepts the model's provenance but ONLY when each referenced id exists in
    the actual candidate blocks; anything else is nulled, never invented."""
    known_document_ids = {c.document_id for c in candidates if c.document_id is not None}
    known_titles = {c.document_title for c in candidates if c.document_title}

    source = item.get("source")
    if not isinstance(source, dict):
        source = {}

    def _int_or_none(value: object) -> int | None:
        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    document_id = _int_or_none(source.get("document_id"))
    if document_id is not None and document_id not in known_document_ids:
        document_id = None
    chunk_ids_raw = source.get("chunk_ids", [])
    if not isinstance(chunk_ids_raw, list):
        chunk_ids_raw = []
    # Anchor to a real block: the model's verified doc id, else when there is
    # exactly one source document among the candidate blocks, the only block it
    # could have been reading (never an invented id).
    anchor = None
    if document_id is not None:
        anchor = next((c for c in candidates if c.document_id == document_id), None)
    if anchor is None:
        unique_docs = {c.document_id for c in candidates if c.document_id is not None}
        unique_titles = {c.document_title for c in candidates if c.document_title}
        if document_id is not None:
            matching = [c for c in candidates if c.document_id == document_id]
        elif len(unique_docs) == 1 and len(unique_titles) <= 1:
            matching = [c for c in candidates if c.document_id is not None]
        else:
            matching = []
        if len(matching) == 1:
            anchor = matching[0]
            document_id = anchor.document_id
    if anchor is None:
        # Only a single candidate block exists at all: that is the one true source.
        if len(candidates) == 1 and document_id is None:
            anchor = candidates[0]
            document_id = anchor.document_id

    known_chunk_ids = {cid for c in candidates for cid in c.chunk_ids}
    chunk_ids = [cid for cid in (_int_or_none(v) for v in chunk_ids_raw) if cid is not None and cid in known_chunk_ids]
    if not chunk_ids and anchor is not None:
        chunk_ids = list(anchor.chunk_ids)
    pages_raw = source.get("pages", [])
    if not isinstance(pages_raw, list):
        pages_raw = []
    pages = [int(p) for p in pages_raw if _int_or_none(p) is not None]
    if not pages and anchor is not None:
        pages = sorted({p for p in (anchor.page_start, anchor.page_end) if p is not None})
    title = source.get("document_name") or source.get("document_title")
    if title is None or (isinstance(title, str) and title not in known_titles):
        title = anchor.document_title if anchor is not None else None
    return {
        "chunk_ids": chunk_ids,
        "document_id": document_id,
        "document_title": title,
        "pages": pages,
    }


def _first_str(obj: object) -> str | None:
    if isinstance(obj, str):
        value = obj.strip()
        return value or None
    return None


def _parse_items(raw: object, instruction: str) -> list[dict[str, object]]:
    if not isinstance(raw, list) or not raw:
        return [{"content": instruction}]
    items: list[dict[str, object]] = []
    for entry in raw:
        if isinstance(entry, str):
            items.append({"content": entry.strip()})
            continue
        if isinstance(entry, dict):
            number = entry.get("number")
            content = _first_str(entry.get("content")) or _first_str(entry.get("text"))
            items.append({"number": number, "content": content or ""})
    if not any(item.get("content") for item in items):
        return [{"content": instruction}]
    return items


class ExerciseExtractionService:
    """Semantic extraction/structuring of retrieved KB exercise blocks."""

    def __init__(
        self, *,
        llm: LLMProvider,
        settings: Settings | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if llm is None:
            raise ValueError("ExerciseExtractionService requires an LLM provider.")
        self.llm = llm
        self.settings = settings or get_settings()
        self._sleep = sleep

    # -- LLM call (bounded, 429 retried once) -------------------------------
    def _extract_once(self, system: str, user: str):
        attempts = 0
        while True:
            started = time.perf_counter()
            try:
                result = self.llm.generate(
                    system_prompt=system, user_prompt=user, temperature=0.0,
                    max_tokens=self.settings.exercise_extraction_max_output_tokens,
                    retry_policy=_NO_PROVIDER_RETRY,
                    generation_options=LLMGenerationOptions(reasoning_effort="low", include_reasoning=False),
                )
            except LLMProviderError as exc:
                duration_seconds = time.perf_counter() - started
                if exc.status_code != 429:
                    ExerciseGenerationService._log_provider_request_failed(
                        exc, model_id=self.llm.model_id, duration_seconds=duration_seconds,
                    )
                    raise ExerciseExtractionError(exc.provider_message) from exc
                logger.warning("[LLM_RATE_LIMIT_RETRY] exercise-extraction provider=%s attempts=%s", self.llm.model_id, attempts)
                if attempts >= _MAX_PROVIDER_RETRIES:
                    raise ExerciseExtractionRateLimitError(_RATE_LIMIT_USER_MESSAGE) from exc
                self._sleep(_RATE_LIMIT_FALLBACK_DELAY_SECONDS)
                attempts += 1
                continue
            if (result.finish_reason or "").casefold() in {"length", "max_tokens", "max_output_tokens"}:
                raise ExerciseExtractionError("L'extraction des exercices a été tronquée. Élargissez la recherche.")
            return result

    # -- System prompt (dedicated to extraction, not generation) -------------
    @classmethod
    def _build_system_prompt(cls) -> str:
        return (
            "Tu es un expert en didactique de la langue arabe et en reconnaissance d'exercices dans "
            "des documents pédagogiques. Ton rôle est UNIQUEMENT d'analyser les passages fournis pour :\n"
            "1. comprendre chaque passage ;\n"
            "2. décider s'il contient RÉELLEMENT un exercice (une tâche à réaliser par l'apprenant) — "
            "et non seulement une phrase qui parle d'exercices ;\n"
            "3. extraire l'exercice en le recopiant EXACTEMENT tel qu'il apparaît, sans le réécrire ;\n"
            "4. reconstituer un exercice réparti sur plusieurs chunks/pages en UN seul exercice ;\n"
            "5. structurer la consigne, les items et le type ;\n"
            "6. indiquer le niveau CECRL UNIQUEMENT s'il est écrit dans le document, sinon ignorer.\n\n"
            "Ce qui n'est PAS un exercice : préface, introduction, description du manuel, présentation "
            "méthodologique, description du programme, 'ce manuel contient…', 'les exercices proposés…', "
            "un texte théorique sans tâche adressée à l'apprenant.\n\n"
            "RÈGLES STRICHTES\n"
            "- INTERDIT d'inventer un exercice, un item, une correction ou du contenu absent des passages.\n"
            "- INTERDIT d'ajouter une correction (expected_answer) si elle n'est pas dans le document : mettre null.\n"
            "- Recopie les items tels quels. Ne reformule pas, n'ajoute pas de mots.\n"
            "- La provenance source doit reprendre uniquement les document_id, document_name, pages et "
            "chunk_ids RÉELLEMENT présents dans les passages. Tout ID inconnu doit être null.\n"
            "- status est toujours \"kb_original\".\n"
            "- is_exercise=false pour tout passage qui parle d'exercices sans en contenir un.\n\n"
            "Réponds UNIQUEMENT avec un objet JSON valide (sans Markdown) de la forme :\n"
            "{\"exercises\": [{\"is_exercise\": true, \"title\": \"…\", \"instruction\": \"…\", "
            "\"exercise_type\": \"qcm|true_false|matching|fill_blank|ordering|transformation|production|open_question|mixed\", "
            "\"items\": [{\"number\": 1, \"content\": \"…\"}], \"expected_answer\": null, "
            "\"level\": null, \"level_source\": \"explicit_indexed|inferred\", \"theme\": null, \"skill\": null, "
            "\"source\": {\"document_id\": …, \"document_name\": …, \"pages\": [], \"chunk_ids\": []}}]}\n"
            "Si aucun passage ne contient d'exercice, renvoie {\"exercises\": []}.\n"
            "Ne jamais retourner de texte en dehors de cet objet JSON."
        )

    # -- Public API --------------------------------------------------------
    def extract(
        self,
        blocks: Sequence[ContextSourceBlock],
        *,
        request: ExerciseSearchIn,
        parsed_level: str | None = None,
    ) -> tuple[list[ExerciseSearchItem], int]:
        """LLM extraction over reduced candidate blocks. Returns
        (exercises, llm_calls). Raises on provider failure; a JSON that parses
        but yields no exercises is a legitimate empty result."""
        candidates = _reduce_candidates(
            blocks,
            context_k=self.settings.exercise_extraction_context_k,
            max_context_tokens=self.settings.exercise_extraction_max_context_tokens,
        )
        if not candidates:
            return [], 0

        passages = [
            {
                "chunk_ids": list(c.chunk_ids),
                "document_id": c.document_id,
                "document_name": c.document_title,
                "pages": [p for p in (c.page_start, c.page_end) if p is not None],
                "text": c.text[: _estimated_cap(self.settings)],
            }
            for c in candidates
        ]
        user = json.dumps({
            "query": request.query,
            "passages": passages,
        }, ensure_ascii=False)

        result = self._extract_once(self._build_system_prompt(), user)
        payload = _parse_extraction_json(result.text)

        raw_exercises = payload.get("exercises")
        if not isinstance(raw_exercises, list) or not raw_exercises:
            return [], 1

        extracted: list[ExtractedExercise] = []
        strict_level = normalize_level(request.level or parsed_level)
        for raw in raw_exercises:
            if not isinstance(raw, dict):
                continue
            is_exercise = raw.get("is_exercise", True)
            if is_exercise is False:
                continue
            parsed = self._to_extracted(raw, candidates, strict_level=strict_level)
            if parsed is not None:
                extracted.append(parsed)

        deduplicated = self._deduplicate(extracted)
        items = self._to_items(deduplicated, candidates)
        return items, 1

    # -- Validate + map one extracted exercise ------------------------------
    def _to_extracted(
        self, raw: dict[str, object], candidates: Sequence[ExtractionCandidate],
        *, strict_level: str | None,
    ) -> ExtractedExercise | None:
        instruction = (_first_str(raw.get("instruction"))
                       or _first_str(raw.get("prompt"))
                       or _first_str(raw.get("title")))
        if not instruction:
            return None

        # A strict requested level is a HARD filter: never relabel a
        # provably-other-level source as the requested level.
        doc_id = _doc_id_of(raw)
        anchor = next((c for c in candidates if c.document_id == doc_id), None)
        explicit = detect_explicit_level(
            content=instruction,
            heading_context=list(anchor.heading_context) if anchor is not None else [],
            indexed_cefr_level=None,
        )
        if strict_level and explicit and explicit != strict_level:
            return None

        prov = _resolve_provenance(raw, candidates=candidates)
        items = _parse_items(raw.get("items"), instruction)

        exercise_type = _first_str(raw.get("exercise_type"))
        if exercise_type not in _ACCEPTED_TYPES:
            exercise_type = classify_exercise_type(instruction, [])

        level = normalize_level(_first_str(raw.get("level")))
        level_source_str = _first_str(raw.get("level_source")) or "inferred"
        if level is None:
            estimated, _conf = estimate_level(instruction)
            level = estimated or strict_level
            level_source = "inferred"
        else:
            level_source = "explicit" if level_source_str == "explicit" else "inferred"

        return ExtractedExercise(
            title=_first_str(raw.get("title")) or instruction[:80],
            instruction=instruction,
            exercise_type=exercise_type or "mixed",
            items=items,
            expected_answer=(_first_str(raw.get("expected_answer"))
                             if raw.get("expected_answer") else None),
            level=level or "A1",
            level_source=level_source,
            theme=_first_str(raw.get("theme")),
            skill=_first_str(raw.get("skill")),
            source_chunk_ids=prov["chunk_ids"],
            source_document_id=prov["document_id"],
            source_document_title=prov["document_title"],
            source_pages=prov["pages"],
        )

    # -- Deduplication ------------------------------------------------------
    def _deduplicate(self, exercises: list[ExtractedExercise]) -> list[ExtractedExercise]:
        seen: set[tuple[object, ...]] = set()
        result: list[ExtractedExercise] = []
        for ex in exercises:
            key = ex.dedup_key()
            if key in seen:
                continue
            # Also drop raw text duplicates within the same document.
            raw_norm = " ".join((ex.instruction or "").split()).casefold()
            if any(existing.source_document_id == ex.source_document_id
                   and " ".join((existing.instruction or "").split()).casefold() == raw_norm
                   for existing in result):
                continue
            seen.add(key)
            result.append(ex)
        return result

    # -- Map to schema items ------------------------------------------------
    def _to_items(
        self, exercises: list[ExtractedExercise],
        candidates: Sequence[ExtractionCandidate],
    ) -> list[ExerciseSearchItem]:
        items: list[ExerciseSearchItem] = []
        for ex in exercises:
            anchor = next((c for c in candidates if c.document_id == ex.source_document_id), None)
            page_start = anchor.page_start if anchor is not None else (ex.source_pages[0] if ex.source_pages else None)
            page_end = anchor.page_end if anchor is not None else (ex.source_pages[-1] if ex.source_pages else None)
            prompt = ex.instruction
            if ex.items:
                prompt = ex.instruction + "\n" + "\n".join(
                    f"{i.get('number', '')} {i['content']}".strip()
                    if i.get("number") else str(i["content"])
                    for i in ex.items
                )
            items.append(ExerciseSearchItem(
                title=ex.title or "Exercice",
                skill=ex.skill or "",
                skill_source="inferred" if ex.skill else "inferred",
                exercise_type=ex.exercise_type or "mixed",
                type_source="inferred",
                prompt=prompt.strip(),
                context="",
                answer_expectation=ex.expected_answer,
                level=ex.level or "A1",
                level_source=ex.level_source,
                theme=ex.theme or "",
                theme_source="inferred" if ex.theme else "inferred",
                status="kb_original",
                document_title=ex.source_document_title,
                document_id=ex.source_document_id,
                page_start=page_start,
                page_end=page_end,
                chunk_ids=ex.source_chunk_ids,
                heading_context=[ex.source_document_title] if ex.source_document_title else [],
                theme_label=ex.theme,
                skill_label=ex.skill,
                summary_score=0.0,
            ))
        return items


def _parse_extraction_json(text_raw: str) -> dict:
    """Extract the extraction envelope, preferring the object that carries an
    ``exercises`` list. Robust against Markdown fences and stray prose, and
    unlike the generation parser it never confuses a nested exercise object
    (which also contains text) with the envelope root."""
    text = (text_raw or "").lstrip("\ufeff \t\r\n")
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", stripped).strip()
    try:
        loaded = json.loads(stripped)
        if isinstance(loaded, dict):
            return loaded
    except json.JSONDecodeError:
        pass
    # Fall back to scanning for the object with an 'exercises' list.
    decoder = json.JSONDecoder()
    start = text.find("{")
    while start >= 0:
        try:
            value, end = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            start = text.find("{", start + 1)
            continue
        if isinstance(value, dict):
            urtext = value.get("exercises")
            if isinstance(urtext, list):
                return value
        start = text.find("{", end)
    raise ExerciseExtractionError("La réponse d'extraction n'est pas un JSON valide.")


def _doc_id_of(raw: dict[str, object]) -> int | None:
    source = raw.get("source")
    if isinstance(source, dict):
        try:
            return int(source.get("document_id")) if source.get("document_id") is not None else None
        except (TypeError, ValueError):
            return None
    return None


def _estimated_cap(settings: Settings) -> int:
    return settings.exercise_extraction_max_context_tokens * 4


__all__ = [
    "ExerciseExtractionService",
    "ExerciseExtractionError",
    "ExerciseExtractionRateLimitError",
    "ExtractionCandidate",
]
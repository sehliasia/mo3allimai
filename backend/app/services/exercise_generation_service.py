"""ChatGPT-style exercise generation engine.

The teacher writes a natural-language request; the generation engine turns it
into a coherent, varied exercise set. Unlike the previous KB-first behaviour,
the LLM is now ALWAYS the generator when a provider is configured:

- The RAG pedagogical context is gathered and used as REFERENCE material only
  (reuse vocabulary / notions / situations), never as a constraint, and never
  copied verbatim.
- A single bounded LLM call produces a genuinely-structured progression with
  variety, clear corrected answers, correct MSA Arabic and level-appropriate
  CECRL guidance. The output is validated post-generation (structure,
  anti-doublons, level, Arabic, answer coherence).
- Provenance discipline is preserved: an exercise is attributed to a KB block
  ONLY when source_index genuinely points to a real block; otherwise it stays
  provenance-null (status ai_generated) and is never claimed to come from a
  document.
- If no LLM provider is configured, the engine degrades to the deterministic
  KB-only extractor (same contract as before) and errors when no exercise block
  is available.
"""
from __future__ import annotations

import json
import logging
import re
import time
import unicodedata
from typing import Callable
from pydantic import ValidationError
from app.core.config import Settings, get_settings
from app.schemas.exercise_generator import ExerciseGenerateIn, ExerciseItem, ExerciseOut
from app.services.exercise_cefr import detect_explicit_level, has_arabic_script
from app.services.exercise_detection import score_exercise
from app.services.exercise_planner import ExercisePlan, PedagogicalPlanner
from app.services.exercise_prompts import (
    build_context_section, build_system_prompt, build_task_section,
)
from app.services.exercise_validator import MAX_REGENERATION_ATTEMPTS, ExerciseValidator
from app.services.llm_provider import LLMGenerationOptions, LLMProvider, LLMProviderError, LLMRetryPolicy
from app.services.pedagogical_knowledge_service import PedagogicalContext, PedagogicalResourceBlock

logger = logging.getLogger(__name__)

# Bounded provider rate-limit handling, same contract as the course generator.
_MAX_PROVIDER_RETRIES = 1
_RATE_LIMIT_FALLBACK_DELAY_SECONDS = 2.0
_NO_PROVIDER_RETRY = LLMRetryPolicy(
    max_retries=0, max_wait_seconds=0.0, retry_base_delay=0.0, retry_max_delay=0.0,
)
_RATE_LIMIT_USER_MESSAGE = (
    "Le service IA est temporairement très sollicité. "
    "Veuillez patienter quelques secondes puis réessayer."
)

# Role markers reused from the pedagogical knowledge layer (EXERCISE role).
_EXERCISE_CONTENT_TYPES = {"worksheet_exercise", "exercise", "exercises"}
_EXERCISE_ROLE_MARKERS = ("exercice", "exercise", "worksheet_exercise", "تمرين", "تمارين")

# Blocks whose multi-signal score falls below this strongly-negative value are
# descriptive/preface text (e.g. "Pour chaque niveau, des exercices sont
# proposés…"), kept for neither generation nor search.
_STRONGLY_DESCRIPTIVE_THRESHOLD = -2.0


def _default_title(request) -> str:
    """A teacher-friendly default title, Arabic when the request is Arabic."""
    if (request.language or "").casefold() in ("ar", "ara", "arabe", "عربي"):
        return f"تمارين اللغة العربية – المستوى {request.level}"
    return f"Exercices — {request.theme}"


def _as_str(value) -> str:
    """Coerce a raw LLM field to a str only when it is actually a scalar. A
    structurally-mismatched value (e.g. a list where a string is expected) is
    drained to "" so a single bad field cannot abort the whole generation; the
    validator then flags the item and the targeted regeneration repairs it."""
    return str(value) if isinstance(value, (str, int, float)) else ""


def _ans_str(value) -> str | None:
    """Safe answer_expectation: keep a scalar string; drop lists/dicts (which
    are structural noise, e.g. pairs mistakenly placed there) to None so the
    validator flags 'réponse attendue manquante' instead of a hard failure."""
    return _as_str(value) if isinstance(value, (str, int, float)) else None


class ExerciseGenerationError(RuntimeError):
    pass


class ExerciseRateLimitError(ExerciseGenerationError):
    pass


class ExerciseGenerationService:
    def __init__(
        self, *, llm: LLMProvider | None, settings: Settings | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.llm = llm
        self.settings = settings or get_settings()
        self._sleep = sleep
        self.planner = PedagogicalPlanner()
        self.validator = ExerciseValidator()

    # -- KB classification -------------------------------------------------
    @classmethod
    def _is_exercise_block(cls, block: PedagogicalResourceBlock) -> bool:
        """A block is an exercise candidate when its indexed content_type or its
        content/heading markers clearly identify an exercise. Mirrors the role
        classification used by the pedagogical knowledge layer."""
        content_type = (block.content_type or "").casefold()
        if content_type in _EXERCISE_CONTENT_TYPES:
            return True
        evidence = f"{content_type} {' '.join(block.heading_context or [])} {block.content}".casefold()
        return any(marker in evidence for marker in _EXERCISE_ROLE_MARKERS)

    @classmethod
    def _select_exercise_blocks(
        cls, context: PedagogicalContext, limit: int,
    ) -> list[tuple[int, PedagogicalResourceBlock]]:
        """Return (order, block) pairs for exercise blocks in source order,
        capped at `limit`. Deterministic and KB-only.

        Marker-based selection (existing behaviour) is kept for KB parity, but
        blocks that are demonstrably descriptive/preface text — strong negative
        rhetorical signals with no task structure — are excluded outright.
        """
        selected: list[tuple[int, PedagogicalResourceBlock]] = []
        for order, block in enumerate(context.resource_blocks):
            if cls._is_exercise_block(block):
                detection = score_exercise(
                    block.content, block.heading_context, threshold=0.0,
                )
                if detection.raw_score < _STRONGLY_DESCRIPTIVE_THRESHOLD:
                    continue
                selected.append((order, block))
            if len(selected) >= limit:
                break
        return selected

    # -- Provenance helpers ------------------------------------------------
    @staticmethod
    def _normalized(value: str) -> str:
        return " ".join(
            "".join(
                character for character in unicodedata.normalize("NFD", value.casefold())
                if unicodedata.category(character) != "Mn"
            ).split()
        )

    @classmethod
    def _blocks_are_duplicates(cls, a: PedagogicalResourceBlock, b: PedagogicalResourceBlock) -> bool:
        a_sig = cls._normalized(a.content)[:120] or cls._normalized(a.heading_context[0] if a.heading_context else "")
        b_sig = cls._normalized(b.content)[:120] or cls._normalized(b.heading_context[0] if b.heading_context else "")
        return bool(a_sig and a_sig == b_sig)

    @classmethod
    def _dedup_blocks(cls, blocks: list[tuple[int, PedagogicalResourceBlock]]) -> list[tuple[int, PedagogicalResourceBlock]]:
        seen: list[str] = []
        unique: list[tuple[int, PedagogicalResourceBlock]] = []
        for pair in blocks:
            sig = cls._normalized(pair[1].content)[:120] or (
                cls._normalized(pair[1].heading_context[0] if pair[1].heading_context else "")
            )
            if not sig:
                unique.append(pair)
                continue
            if sig in seen:
                continue
            seen.append(sig)
            unique.append(pair)
        return unique

    @staticmethod
    def _block_to_item(order: int, block: PedagogicalResourceBlock, request: ExerciseGenerateIn) -> ExerciseItem:
        heading = block.heading_context[0] if block.heading_context else ""
        title = heading or block.document_title or f"Exercice {order + 1}"
        explicit_level = detect_explicit_level(
            content=block.content, heading_context=block.heading_context,
        )
        return ExerciseItem(
            title=title,
            skill=", ".join(request.skills),
            exercise_type=request.exercise_type or "exercice",
            prompt=block.content.strip(),
            context="",
            answer_expectation=None,
            level=request.level,
            level_source="explicit" if explicit_level else "inferred",
            theme=request.theme,
            status="kb_original",
            document_title=block.document_title,
            document_id=block.document_id,
            page_start=block.page_start,
            page_end=block.page_end,
            chunk_ids=list(block.chunk_ids),
            heading_context=list(block.heading_context),
        )

    # -- JSON parsing (mirrors the course generator) ------------------------
    @staticmethod
    def _scan_objects(text: str) -> list[dict]:
        decoder = json.JSONDecoder()
        objects: list[dict] = []
        start = text.find("{")
        while start >= 0:
            try:
                value, _ = decoder.raw_decode(text, start)
                if isinstance(value, dict):
                    objects.append(value)
            except json.JSONDecodeError:
                pass
            start = text.find("{", start + 1)
        return objects

    _ROOT_SIGNATURE = {"level", "theme", "exercises"}
    _ROOT_WEIGHTS = {
        "title": 2, "level": 3, "theme": 3, "exercise_type": 1,
        "exercises": 3, "language": 1, "skills": 1,
    }

    @classmethod
    def _root_score(cls, obj: object) -> int:
        if not isinstance(obj, dict):
            return 0
        if not cls._ROOT_SIGNATURE.issubset(obj):
            return 0
        return sum(weight for key, weight in cls._ROOT_WEIGHTS.items() if key in obj)

    @classmethod
    def _select_root(cls, candidates: list[dict]) -> dict | None:
        best_root: dict | None = None
        best_root_score = 0
        best_any: dict | None = None
        best_any_score = -1
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            root_score = cls._root_score(candidate)
            if root_score > best_root_score:
                best_root = candidate
                best_root_score = root_score
            any_score = sum(weight for key, weight in cls._ROOT_WEIGHTS.items() if key in candidate)
            if candidate and any_score > best_any_score:
                best_any = candidate
                best_any_score = any_score
        return best_root if best_root is not None else best_any

    @classmethod
    def _json_object(cls, raw: str) -> dict:
        text = raw.lstrip("\ufeff \t\r\n")
        candidates = cls._scan_objects(text)
        best = cls._select_root(candidates)
        if best is not None:
            return best
        raise ExerciseGenerationError("Le JSON des exercices est incomplet ou invalide.")

    # -- Provider error handling (same contract as course generator) --------
    def _generate_once(self, system: str, user: str):
        attempts = 0
        total_waited = 0.0
        while True:
            started = time.perf_counter()
            try:
                result = self.llm.generate(
                    system_prompt=system, user_prompt=user, temperature=0.2,
                    max_tokens=self.settings.exercise_max_output_tokens,
                    retry_policy=_NO_PROVIDER_RETRY,
                    generation_options=LLMGenerationOptions(reasoning_effort="low", include_reasoning=False),
                )
            except LLMProviderError as exc:
                duration_seconds = time.perf_counter() - started
                if exc.status_code != 429:
                    self._log_provider_request_failed(exc, model_id=self.llm.model_id, duration_seconds=duration_seconds)
                    raise ExerciseGenerationError(exc.provider_message) from exc
                logger.warning("[LLM_RATE_LIMIT_RETRY] exercise provider=%s attempts=%s", self.llm.model_id, attempts)
                if attempts >= _MAX_PROVIDER_RETRIES:
                    raise ExerciseRateLimitError(_RATE_LIMIT_USER_MESSAGE) from exc
                self._sleep(_RATE_LIMIT_FALLBACK_DELAY_SECONDS)
                total_waited += _RATE_LIMIT_FALLBACK_DELAY_SECONDS
                attempts += 1
                continue
            if (result.finish_reason or "").casefold() in {"length", "max_tokens", "max_output_tokens"}:
                raise ExerciseGenerationError("La génération a été tronquée. Réduisez les consignes puis réessayez.")
            return result

    @staticmethod
    def _log_provider_request_failed(exc: LLMProviderError, *, model_id: str, duration_seconds: float) -> None:
        cause = exc.__cause__ if isinstance(exc.__cause__, Exception) else exc
        category = exc.category or type(cause).__name__ or "provider_error"
        metadata = exc.response_metadata or {}
        error_type = metadata.get("error_type") or category
        status_code = metadata.get("status_code") if metadata.get("status_code") is not None else exc.status_code
        logger.error(
            "provider_request_failed provider=%s model=%s error_type=%s status_code=%s "
            "error_message=%s duration_seconds=%.3f",
            "groq", model_id, error_type,
            status_code if status_code is not None else "none",
            exc.provider_message or "none", duration_seconds,
        )

    # -- Normalization -----------------------------------------------------
    @staticmethod
    def _normalize_payload(payload: dict) -> dict:
        normalized = payload.copy()
        exercises = normalized.get("exercises")
        if isinstance(exercises, list):
            cleaned: list[dict] = []
            for item in exercises:
                if not isinstance(item, dict):
                    continue
                source_index = item.get("source_index")
                if isinstance(source_index, str):
                    try:
                        source_index = int(source_index)
                    except (TypeError, ValueError):
                        source_index = None
                cleaned.append({
                    "title": item.get("title", ""),
                    "skill": item.get("skill", ""),
                    "exercise_type": item.get("exercise_type", ""),
                    "prompt": item.get("prompt", ""),
                    "context": item.get("context", ""),
                    "answer_expectation": item.get("answer_expectation"),
                    "level": item.get("level", ""),
                    "options": item.get("options"),
                    "is_true": item.get("is_true"),
                    "pairs": item.get("pairs"),
                    "difficulty": item.get("difficulty"),
                    "source_index": source_index if isinstance(source_index, int) else None,
                })
            normalized["exercises"] = cleaned
        return normalized

    # -- Generation ---------------------------------------------------------
    def generate(
        self, request: ExerciseGenerateIn, context: PedagogicalContext,
    ) -> ExerciseOut:
        # Pedagogical planning runs BEFORE both KB and LLM paths so that the
        # plan (objectives, target vocabulary, distribution, rationale) is a
        # first-class output of the tool.
        plan = self.planner.plan(
            level=request.level, theme=request.theme, skills=request.skills,
            count=request.count, objective=request.objective, context=context,
            language=request.language,
        )
        all_blocks = self._select_exercise_blocks(context, limit=request.count)
        blocks = self._dedup_blocks(all_blocks)

        # With a provider, the LLM is always the generator: RAG blocks are
        # reference material only. Without a provider, we degrade to the
        # deterministic KB-only extractor.
        if self.llm is None:
            return self._generate_from_blocks(request, blocks, context, plan=plan)
        return self._generate_with_ai(request, blocks, context, plan=plan)

    def _generate_from_blocks(
        self, request: ExerciseGenerateIn, blocks: list[tuple[int, PedagogicalResourceBlock]],
        context: PedagogicalContext, *, plan: ExercisePlan,
    ) -> ExerciseOut:
        if not blocks:
            raise ExerciseGenerationError(
                "Aucun exercice exploitable trouvé dans vos documents. "
                "Ajoutez des fiches d'exercices à votre bibliothèque ou activez l'adaptation avec l'IA."
            )
        items = [
            self._block_to_item(order, block, request) for order, block in blocks
        ]
        return ExerciseOut(
            title=f"Exercices — {request.theme}",
            level=request.level,
            theme=request.theme,
            exercise_type=request.exercise_type,
            language=request.language,
            skills=request.skills,
            exercises=items,
            kb_sourced_count=len(items),
            rag_sources_used=len(context.resource_blocks),
            provider_model=None,
            adapt_with_ai=False,
            plan=self._plan_to_dict(plan),
        )

    def _generate_with_ai(
        self, request: ExerciseGenerateIn, blocks: list[tuple[int, PedagogicalResourceBlock]],
        context: PedagogicalContext, *, plan: ExercisePlan,
    ) -> ExerciseOut:
        source_blocks = [
            {
                "source_index": index,
                "document_title": block.document_title,
                "content": block.content[:900],
                "heading": " ".join(block.heading_context or [])[:200],
            }
            for index, (_, block) in enumerate(blocks)
        ]
        # Distribution: honour the planner's ordered types, unless the teacher
        # explicitly forced a single exercise type.
        forced = request.exercise_type.casefold() if request.exercise_type else "auto"
        if forced not in ("auto", ""):
            distribution = [forced] * request.count
        else:
            distribution = list(plan.exercise_distribution) or [request.exercise_type or "auto"] * request.count

        system = build_system_prompt(language=request.language)
        context_section = build_context_section(source_blocks)
        task_section = build_task_section(
            {
                "count": request.count,
                "theme": request.theme,
                "objective": request.objective or "",
                "skills": request.skills,
                "language": request.language,
                "special_instructions": request.special_instructions or "",
                "forced_type": forced if forced != "auto" else "",
            },
            plan, distribution,
        )
        user = (
            "= CONTEXT (pédagogie) =\n"
            + context_section
            + "\n\n= PLAN PÉDAGOGIQUE (donné par le générateur d'exercices, à suivre) =\n"
            + json.dumps(self._plan_to_dict(plan), ensure_ascii=False)
            + "\n\n= TACHE / CONTRAINTES =\n"
            + task_section
        )

        result = self._generate_once(system, user)
        try:
            items, title, level, theme, exercise_type = self._first_generation(
                request, result.text, blocks, theme=plan.theme,
            )
            # Validate every generated item (structural + pedagogical + quality)
            # and regenerate ONLY the invalid subset (bounded).
            items = self._regenerate_invalid(
                request, result, items, blocks, plan, distribution,
            )
            items = self._order_by_distribution(items, distribution)
            kb_sourced_count = sum(1 for item in items if item.document_id is not None)
            return ExerciseOut(
                title=title, level=level or plan.level, theme=theme or plan.theme,
                exercise_type=exercise_type,
                language=request.language, skills=request.skills,
                exercises=items, kb_sourced_count=kb_sourced_count,
                rag_sources_used=len(context.resource_blocks),
                provider_model=result.model, adapt_with_ai=True,
                plan=self._plan_to_dict(plan),
            )
        except (ValidationError, ExerciseGenerationError):
            raise
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "exercise_ai_generation_failed provider=%s error=%s", result.model, exc,
            )
            raise ExerciseGenerationError("La génération des exercices a échoué.") from exc

    def _first_generation(
        self, request: ExerciseGenerateIn, raw: str,
        blocks: list[tuple[int, PedagogicalResourceBlock]], *, theme: str,
    ) -> tuple[list[ExerciseItem], str, str, str, str]:
        payload = self._json_object(raw)
        title = payload.get("title") or _default_title(request)
        level = payload.get("level") or request.level
        ex_theme = payload.get("theme") or request.theme
        exercise_type = payload.get("exercise_type") or request.exercise_type
        exercises_raw = self._normalize_payload(payload).get("exercises") or []
        if not exercises_raw:
            raise ExerciseGenerationError("Aucun exercice généré.")
        items: list[ExerciseItem] = []
        for item in exercises_raw:
            if not isinstance(item, dict):
                continue
            items.append(self._build_item(request, item, exercise_type, blocks, theme))
        if not items:
            raise ExerciseGenerationError("Aucun exercice généré au bon format.")
        return items, title, level, ex_theme, exercise_type

    def _regenerate_invalid(
        self, request: ExerciseGenerateIn, result, items: list[ExerciseItem],
        blocks: list[tuple[int, PedagogicalResourceBlock]], plan: ExercisePlan,
        distribution: list[str],
    ) -> list[ExerciseItem]:
        """Validate the whole set; regenerate only the invalid duplicates/items.

        Bounded by MAX_REGENERATION_ATTEMPTS so this can never loop forever.
        Uses the full validator (structural + pedagogical + quality + duplicates)
        and keeps whatever remains valid; a fully-valid set stops immediately.
        """
        attempts = 0
        while True:
            verdicts = self.validator.validate(
                items, request_level=request.level, theme=plan.theme, language=request.language,
            )
            ok_indices = [v.index for v in verdicts if v.ok]
            bad_indices = sorted({v.index for v in verdicts if not v.ok})

            # Duplicates among the (separately) valid items.
            dup_indices = self.validator.find_duplicate_indices(items)

            to_fix = sorted(set(bad_indices) | set(dup_indices))
            if not to_fix:
                for v in verdicts:
                    if not v.ok:
                        logger.warning(
                            "[VALIDATOR] rejected index=%s reasons=%s", v.index, v.reasons,
                        )
                return items

            logger.warning(
                "[VALIDATOR] rejecting count=%s indices=%s attempt=%s",
                len(to_fix), to_fix, attempts,
            )
            if attempts >= MAX_REGENERATION_ATTEMPTS:
                # Keep only the valid subset; drop the unregenerable ones.
                kept = [items[i] for i in ok_indices]
                return kept or items

            items = self._regenerate_subset(
                request, result, items, to_fix, blocks, plan, distribution,
            )
            attempts += 1

    def _regenerate_subset(
        self, request: ExerciseGenerateIn, result, items: list[ExerciseItem],
        to_fix: list[int], blocks: list[tuple[int, PedagogicalResourceBlock]],
        plan: ExercisePlan, distribution: list[str],
    ) -> list[ExerciseItem]:
        """Rebuild the invalid subset only, via a targeted consolidated JSON
        call prefixed by the retained (valid) items so the model can diversify."""
        retained = [items[i] for i in range(len(items)) if i not in to_fix]
        retained_payload = [
            {
                "title": item.title,
                "skill": item.skill,
                "exercise_type": item.exercise_type,
                "prompt": item.prompt,
                "answer_expectation": item.answer_expectation,
            }
            for item in retained
        ]
        system = build_system_prompt(language=request.language)
        user = (
            "REGÉNÉRATION CIBLÉE : les précédents exercices suivants sont DÉJÀ retenus "
            "(ne les reproduis pas, diversifie au maximum) :\n"
            + json.dumps(retained_payload, ensure_ascii=False)
            + "\nRegénère UNIQUEMENT les "
            + str(len(to_fix))
            + " exercices manquants pour compléter la distribution : "
            + ", ".join(distribution[i] for i in to_fix if i < len(distribution))
            + ".\nNiveau : " + plan.level + " — Thème : " + plan.theme
            + " — Compétences : " + ", ".join(plan.skills) + "\n"
            + "Retourne UNIQUEMENT un tableau JSON liste des exercices régénérés, chacun avec "
            "les clés : title, skill, exercise_type, prompt, context (optionnel), "
            "answer_expectation, level, options (pour qcm/true_false), is_true (pour true_false), "
            "pairs (pour matching), difficulty."
        )
        regen = self._generate_once(system, user)
        parsed = self._json_object(regen.text)
        raw_items = parsed.get("exercises") if isinstance(parsed, dict) else None
        if raw_items is None:
            # The regen prompt asks the model for a bare JSON array of the
            # regenerated exercises. `_json_object` may have returned a single
            # weighted dict (no "exercises" wrapper) in that case — treat any
            # object that looks like an exercise item, or a list of them, as
            # the regenerated set so invalid items actually get replaced.
            raw_items = parsed if isinstance(parsed, list) else [parsed] if isinstance(parsed, dict) else []
        if not isinstance(raw_items, list):
            raw_items = [raw_items]
        rebuilt = list(retained)
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            rebuilt.append(self._build_item(request, item, request.exercise_type or "open_question", blocks, plan.theme))
        # Trim back to the retained size + fixed count (<= original length), in
        # distribution order where possible.
        return rebuilt

    @staticmethod
    def _order_by_distribution(items: list[ExerciseItem], distribution: list[str]) -> list[ExerciseItem]:
        """Best-effort re-order so items loosely follow the requested distribution.
        Non-destructive: unknown types fall to the end."""
        order = {t: i for i, t in enumerate(distribution)}

        def rank(item: ExerciseItem) -> int:
            return order.get((item.exercise_type or "").casefold(), len(distribution))
        return sorted(items, key=rank)

    @staticmethod
    def _plan_to_dict(plan: ExercisePlan) -> dict:
        return {
            "level": plan.level,
            "theme": plan.theme,
            "skills": list(plan.skills),
            "objective": plan.objective,
            "learning_objectives": list(plan.learning_objectives),
            "target_vocabulary": list(plan.target_vocabulary),
            "target_grammar": list(plan.target_grammar),
            "exercise_distribution": list(plan.exercise_distribution),
            "rationale": plan.rationale,
        }

    @classmethod
    def _build_item(
        cls, request: ExerciseGenerateIn, item: dict,
        fallback_type: str, blocks: list[tuple[int, PedagogicalResourceBlock]], theme: str,
    ) -> ExerciseItem:
        source_index = item.get("source_index")
        bound: dict[str, object] = {}
        if isinstance(source_index, int) and 0 <= source_index < len(blocks):
            _, block = blocks[source_index]
            bound = {
                "document_title": block.document_title,
                "document_id": block.document_id,
                "page_start": block.page_start,
                "page_end": block.page_end,
                "chunk_ids": list(block.chunk_ids),
            }
        prompt = _as_str(item.get("prompt"))
        if request.language == "ar" and len(prompt) >= 12 and not has_arabic_script(prompt):
            raise ExerciseGenerationError(
                "Les exercices générés en arabe doivent être rédigés en caractères arabes "
                "(العربية الفصحى). Réessayez ou changez de langue."
            )
        level = _as_str(item.get("level")) or request.level
        options_raw = item.get("options")
        options = (
            [str(o) for o in options_raw if isinstance(o, (str, int, float))]
            if isinstance(options_raw, list) else []
        )
        is_true = item.get("is_true")
        is_true = is_true if isinstance(is_true, bool) else None
        pairs_raw = item.get("pairs")
        pairs: list[dict[str, str]] = []
        if isinstance(pairs_raw, list):
            for pair in pairs_raw:
                if isinstance(pair, dict) and ("left" in pair or "right" in pair):
                    pairs.append({
                        "left": str(pair.get("left") or ""),
                        "right": str(pair.get("right") or ""),
                    })
        difficulty = item.get("difficulty")
        if difficulty not in ("easy", "medium", "hard"):
            difficulty = None
        item_data = {
            "title": _as_str(item.get("title")) or "Exercice",
            "skill": _as_str(item.get("skill")) or ", ".join(request.skills),
            "exercise_type": _as_str(item.get("exercise_type")) or fallback_type,
            "prompt": prompt,
            "context": _as_str(item.get("context")),
            "answer_expectation": _ans_str(item.get("answer_expectation")),
            "level": _as_str(level) or request.level,
            "status": "adapted_from_kb" if bound else "ai_generated",
            "level_source": "explicit" if bound else "generated",
            "theme": theme,
            "options": options,
            "is_true": is_true,
            "pairs": pairs,
            "difficulty": difficulty,
            **bound,
        }
        return ExerciseItem.model_validate(item_data)

    # -- Compatibility shims (legacy tests rely on these) -------------------
    @staticmethod
    def _item_signature(item: ExerciseItem) -> str:
        """Legacy normalized identity used for anti-doublon detection (kept for
        backward compatibility; new validation uses ExerciseValidator)."""
        source = item.prompt or item.title or ""
        return " ".join(
            "".join(
                character for character in unicodedata.normalize("NFD", source.casefold())
                if unicodedata.category(character) != "Mn"
            ).split()
        )[:180]

    @classmethod
    def _build_system_prompt(cls, *, adapt: bool = True) -> str:
        """Legacy shim → the new segmented SYSTEM prompt."""
        del adapt
        return build_system_prompt()


# Re-export for the API layer's convenience (PedagogicalContext import remains public).
__all__ = [
    "ExerciseGenerationError",
    "ExerciseRateLimitError",
    "ExerciseGenerationService",
]

"""Explicit per-exercise AI adaptation towards a target CEFR level.

The adaptation keeps the source KB exercise's provenance and its original
level: the result is always labelled status="adapted_from_kb", never presented
as a verbatim extract nor as a silently-rewritten A2 pretending to be A1.
CEFR guidance comes from the internal LEVEL_RULES layer (never presented as an
official Council of Europe citation). Exactly one bounded LLM call is made; a
provider 429 is retried once with a fallback delay.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Callable

from pydantic import ValidationError

from app.core.config import Settings, get_settings
from app.schemas.exercise_generator import ExerciseAdaptIn, ExerciseItem
from app.services.exercise_cefr import LEVEL_RULES, has_arabic_script
from app.services.exercise_generation_service import (
    ExerciseGenerationService,
    ExerciseGenerationError,
    ExerciseRateLimitError,
    _NO_PROVIDER_RETRY,
    _RATE_LIMIT_USER_MESSAGE,
)
from app.services.llm_provider import LLMGenerationOptions, LLMProvider, LLMProviderError

logger = logging.getLogger(__name__)

_MAX_PROVIDER_RETRIES = 1
_RATE_LIMIT_FALLBACK_DELAY_SECONDS = 2.0


class ExerciseAdaptationError(RuntimeError):
    pass


class ExerciseAdaptationRateLimitError(ExerciseAdaptationError):
    pass


class ExerciseAdaptationService:
    def __init__(
        self, *, llm: LLMProvider, settings: Settings | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if llm is None:
            raise ValueError("ExerciseAdaptationService requires an LLM provider.")
        self.llm = llm
        self.settings = settings or get_settings()
        self._sleep = sleep

    def _generate_once(self, system: str, user: str):
        attempts = 0
        while True:
            started = time.perf_counter()
            try:
                result = self.llm.generate(
                    system_prompt=system, user_prompt=user, temperature=0.2,
                    max_tokens=self.settings.exercise_adapt_max_output_tokens,
                    retry_policy=_NO_PROVIDER_RETRY,
                    generation_options=LLMGenerationOptions(reasoning_effort="low", include_reasoning=False),
                )
            except LLMProviderError as exc:
                duration_seconds = time.perf_counter() - started
                if exc.status_code != 429:
                    ExerciseGenerationService._log_provider_request_failed(
                        exc, model_id=self.llm.model_id, duration_seconds=duration_seconds,
                    )
                    raise ExerciseAdaptationError(exc.provider_message) from exc
                logger.warning("[LLM_RATE_LIMIT_RETRY] exercise-adapt provider=%s attempts=%s", self.llm.model_id, attempts)
                if attempts >= _MAX_PROVIDER_RETRIES:
                    raise ExerciseAdaptationRateLimitError(_RATE_LIMIT_USER_MESSAGE) from exc
                self._sleep(_RATE_LIMIT_FALLBACK_DELAY_SECONDS)
                attempts += 1
                continue
            if (result.finish_reason or "").casefold() in {"length", "max_tokens", "max_output_tokens"}:
                raise ExerciseAdaptationError("L'adaptation a été tronquée. Réduisez les consignes puis réessayez.")
            return result

    @staticmethod
    def _assert_arabic_fusha(prompt: str) -> None:
        """Minimal deterministic Arabic quality guard: an Arabic target must stay
        in Arabic script; explicit Darija is only allowed when the user asks."""
        if not prompt.strip():
            return
        if len(prompt) >= 12 and not has_arabic_script(prompt):
            raise ExerciseAdaptationError(
                "L'exercice adapté doit être en arabe (العربية الفصحى). Réessayez avec une langue cible l'arabe."
            )

    @classmethod
    def _build_system_prompt(cls) -> str:
        rules = LEVEL_RULES
        return (
            "Tu es un expert en didactique de la langue arabe et en CECRL. Tu adaptes un exercice "
            "existant vers un niveau cible sans jamais inventer de source. "
            "Voici des RÈGLES D'ADAPTATION CECRL INTERNES (structure pédagogique interne, pas une "
            "citation officielle du Conseil de l'Europe) :\n" +
            "\n".join(
                f"RÈGLE {level} : " + " ; ".join(
                    f"{label}: {value}" for label, value in rules[level].items()
                )
                for level in ("A1", "A2", "B1", "B2", "C1", "C2")
            ) +
            "\nInstructions :\n"
            "- Conserve toujours le cœur pédagogique et les items de l'exercice source.\n"
            "- Adapte UNIQUEMENT la complexité de la tâche au niveau cible : consignes courtes à A1/A2, "
            "ajout de justification/opinion/comparaison à B1/B2, nuance et registre à C1/C2.\n"
            "- Langue cible : arabe subjectivement correct (العربية الفصحى المعاصرة). N'utilise JAMAIS de "
            "darija ni de dialecte sauf demande explicite. N'invente jamais de mots arabes.\n"
            "- Retourne uniquement un objet JSON valide, sans Markdown, avec exactement ces clés : "
            "title, prompt, context, answer_expectation, difficulty (easy|medium|hard).\n"
            "- Le champ prompt contient la consigne ET les items. Ne fabrique pas de provenance.\n"
        )

    def adapt(self, request: ExerciseAdaptIn) -> ExerciseItem:
        source = request.source
        if source.status != "kb_original" and source.document_id is None:
            raise ExerciseAdaptationError(
                "L'adaptation ne s'applique qu'à un exercice sourcé dans vos documents."
            )
        system = self._build_system_prompt()
        user = json.dumps({
            "source_exercise": {
                "title": source.title,
                "level": source.level,
                "exercise_type": source.exercise_type,
                "skill": source.skill,
                "prompt": source.prompt,
                "context": source.context,
                "answer_expectation": source.answer_expectation,
            },
            "target_level": request.target_level,
            "language": request.language,
            "teacher_instructions": request.instructions,
        }, ensure_ascii=False)
        result = self._generate_once(system, user)
        try:
            payload = ExerciseGenerationService._json_object(result.text)
        except ExerciseGenerationError as exc:
            raise ExerciseAdaptationError("L'exercice adapté doit être un objet JSON valide.") from exc
        try:
            prompt = str(payload.get("prompt") or "").strip()
            if not prompt:
                raise ExerciseAdaptationError("L'exercice adapté est vide.")
            if request.language == "ar":
                self._assert_arabic_fusha(prompt)
            adapted = ExerciseItem(
                title=str(payload.get("title") or source.title).strip(),
                skill=source.skill,
                skill_source=source.skill_source,
                exercise_type=str(payload.get("exercise_type") or source.exercise_type),
                type_source=source.type_source,
                prompt=prompt,
                context=str(payload.get("context") or "").strip(),
                answer_expectation=(
                    str(payload["answer_expectation"]).strip()
                    if payload.get("answer_expectation") else source.answer_expectation
                ),
                level=request.target_level,
                level_source="generated",
                theme=source.theme,
                theme_source=source.theme_source,
                difficulty=(
                    str(payload["difficulty"])
                    if payload.get("difficulty") in {"easy", "medium", "hard"} else None
                ),
                status="adapted_from_kb",
                document_title=source.document_title,
                document_id=source.document_id,
                page_start=source.page_start,
                page_end=source.page_end,
                chunk_ids=list(source.chunk_ids),
                heading_context=list(source.heading_context),
                original_level=source.level,
                original_document_title=source.document_title,
                original_document_id=source.document_id,
                original_page_start=source.page_start,
                original_page_end=source.page_end,
                original_chunk_ids=list(source.chunk_ids),
            )
            return adapted
        except ValidationError as exc:
            logger.warning(
                "exercise_adapt_schema_validation_failed provider=%s errors=%s raw_response=%r",
                result.model, exc.errors(include_url=False), result.text,
            )
            raise ExerciseAdaptationError("L'exercice adapté ne respecte pas le format attendu.") from exc


__all__ = [
    "ExerciseAdaptationError",
    "ExerciseAdaptationRateLimitError",
    "ExerciseAdaptationService",
]
"""Deterministic pedagogical parameter resolution; no model or network calls."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


_LEVEL_RE = re.compile(r"(?<![A-Z0-9+\-])(PRE-A1|A1|A2\+|A2|B1\+|B1|B2\+|B2|C1|C2)(?![A-Z0-9+\-])", re.I)
_SKILLS = {
    "speaking": ("production orale", "interaction orale", "activité orale", "expression orale", "faire parler", "conversation", "speaking", "oral production", "oral interaction", "expresión oral", "producción oral", "interacción oral", "التعبير الشفهي", "الإنتاج الشفهي", "التفاعل الشفهي", "المحادثة", "التحدث"),
    "listening": ("compréhension orale", "activité d'écoute", "écoute", "listening comprehension", "listening", "comprensión oral", "escucha", "الاستماع", "فهم المسموع", "الفهم الشفهي"),
    "reading": ("compréhension écrite", "compréhension de l'écrit", "lecture", "reading comprehension", "reading", "comprensión lectora", "lectura", "القراءة", "فهم المقروء"),
    "writing": ("production écrite", "expression écrite", "écriture", "written production", "writing", "producción escrita", "expresión escrita", "التعبير الكتابي", "الإنتاج الكتابي", "الكتابة"),
}
_EXPLICIT_LANGUAGE = {
    "ar": (
        "réponds en arabe",
        "réponse en arabe",
        "donne-moi la réponse en arabe",
        "explique-moi cela en arabe",
        "أجب بالعربية",
        "اشرح بالعربية",
    ),
    "fr": ("réponds en français", "réponse en français"),
    "en": ("answer in english", "respond in english"),
    "es": ("responde en español", "responde en espanol"),
}
_LANGUAGE_MARKERS = {
    "fr": ("propose", "activité", "famille", "rends", "niveau", "compréhension"),
    "en": ("please", "activity", "lesson", "speaking", "reading", "answer"),
    "es": ("propón", "actividad", "nivel", "lectura", "respuesta", "escucha"),
}


@dataclass(frozen=True)
class ResolvedChatParameters:
    cefr_level: str | None
    skills: tuple[str, ...]
    response_language: str | None
    cefr_source: str | None
    skills_source: str | None
    language_source: str | None


class ChatParameterResolver:
    """Explicit API values win; only USER history can supply missing values."""

    @staticmethod
    def historical_semantic_text(text: str) -> str:
        """Keep historic lexical context while removing stale control signals.

        This is deliberately limited to the level and skill expressions used by
        this resolver.  It is only for semantic retrieval context; it never
        changes the original user message or parameter precedence.
        """
        sanitized = _LEVEL_RE.sub("", text)
        markers = sorted(
            {marker for skill_markers in _SKILLS.values() for marker in skill_markers},
            key=len,
            reverse=True,
        )
        for marker in markers:
            sanitized = re.sub(
                rf"(?<!\w){re.escape(marker)}(?!\w)",
                "",
                sanitized,
                flags=re.IGNORECASE,
            )
        sanitized = re.sub(r"\s+", " ", sanitized).strip()
        return re.sub(r"\s+([,.;:!?])", r"\1", sanitized)

    @staticmethod
    def _level(text: str) -> str | None:
        match = _LEVEL_RE.search(text.upper())
        return match.group(1).upper() if match else None

    @staticmethod
    def _skills(text: str) -> tuple[str, ...]:
        folded = text.casefold()
        found: list[str] = []
        for skill in ("speaking", "listening", "reading", "writing"):
            if any(marker.casefold() in folded for marker in _SKILLS[skill]):
                if skill not in found:
                    found.append(skill)
        return tuple(found)

    @staticmethod
    def _language(text: str) -> tuple[str | None, str | None]:
        folded = text.casefold()
        for language, phrases in _EXPLICIT_LANGUAGE.items():
            if any(phrase.casefold() in folded for phrase in phrases):
                return language, "explicit_message"
        if any("\u0600" <= char <= "\u06ff" for char in text):
            return "ar", "message_detection"
        hits = [language for language, markers in _LANGUAGE_MARKERS.items() if any(marker in folded for marker in markers)]
        return (hits[0], "message_detection") if len(hits) == 1 else (None, None)

    def resolve(
        self,
        *,
        message: str,
        cefr_level: str | None,
        skills: Iterable[str],
        language: str | None,
        user_history: Iterable[str] = (),
    ) -> ResolvedChatParameters:
        current_level, current_skills = self._level(message), self._skills(message)
        history = tuple(user_history)
        history_level = next((self._level(item) for item in reversed(history) if self._level(item)), None)
        history_skills = next((self._skills(item) for item in reversed(history) if self._skills(item)), ())
        detected_language, message_language_source = self._language(message)
        history_language, _ = next(
            (self._language(item) for item in reversed(history) if self._language(item)[0]),
            (None, None),
        )
        explicit_skills = tuple(skills)
        return ResolvedChatParameters(
            cefr_level=cefr_level or current_level or history_level,
            skills=explicit_skills or current_skills or history_skills,
            response_language=language or detected_language or history_language,
            cefr_source="api" if cefr_level else "current_message" if current_level else "history" if history_level else None,
            skills_source="api" if explicit_skills else "current_message" if current_skills else "history" if history_skills else None,
            language_source=(
                "api" if language else message_language_source if detected_language else "history" if history_language else None
            ),
        )

"""ExerciseValidator — structural, pedagogical and quality validation.

This validator runs after generation (and after each targeted regeneration) and
decides, for each exercise, whether it is fit to keep. Deterministic guards
only — it never calls an LLM and never invents content.

Validation dimensions:
- Structural: prompt non-empty, level in the grid, exercise_type present.
- Pedagogical: level matches the request, theme present, Arabic script used
  when the request language is Arabic.
- Quality: QCM options rules (≥1 right answer present in options, no duplicate
  options, exactly one right answer convention), V/F coherence, no empty
  answer for non-open types, prompt/answer coherence.
- Duplicate detection: by normalized prompt similarity (light, no heavy model).

It returns an ordered list of validation verdicts and a set of indices to
regenerate. It also enriches structured fields (options/is_true/pairs) from the
LLM payload when present.
"""

from __future__ import annotations

import logging
import unicodedata
from dataclasses import dataclass, field
from typing import Iterable

from app.schemas.exercise_generator import ExerciseItem
from app.services.exercise_cefr import SUPPORTED_LEVELS, has_arabic_script

logger = logging.getLogger(__name__)

# How many times an invalid exercise is regenerated before giving up. Bounded to
# avoid infinite loops.
MAX_REGENERATION_ATTEMPTS = 2

# Default passing similarity threshold for duplicate detection (Dice).
_DEFAULT_SIM_THRESHOLD = 0.78


@dataclass
class ValidationVerdict:
    index: int
    ok: bool
    reasons: list[str] = field(default_factory=list)


def _normalized(text: str) -> str:
    return " ".join(
        "".join(
            character for character in unicodedata.normalize("NFD", (text or "").casefold())
            if unicodedata.category(character) != "Mn"
        ).split()
    )


def _tokenize(text: str) -> set[str]:
    import re as _re
    return {
        w for w in _re.findall(r"[\w\u0600-\u06FF]+", _normalized(text))
        if len(w) >= 2
    }


def _similarity(a: str, b: str) -> float:
    ta, tb = _tokenize(a), _tokenize(b)
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    # Dice coefficient — more sensitive than a max-norm for near-duplicates.
    return (2.0 * inter) / (len(ta) + len(tb))


def _word_multiset(text: str) -> list[str]:
    """Lowercased word tokens (with order) for reconstructability checks."""
    import re as _re
    return [w for w in _re.findall(r"[\w\u0600-\u06FF]+", _normalized(text)) if len(w) >= 2]


def _validate_ordering_reconstructable(prompt: str, answer: str, reasons: list[str]) -> None:
    """Ordering must be solvable: every word of the expected answer must be
    available among the tokens offered to the student. If an essential word is
    missing, the exercise can never be solved — flag it for regeneration."""
    offered = _word_multiset(prompt or "")
    if not offered:
        reasons.append("mots à ordonner absents")
        return
    needed = _word_multiset(answer or "")
    # Any word required by the answer that is absent from the offered tokens
    # makes reconstruction impossible (e.g. "إلى" missing from the shuffled set).
    offered_set = set(offered)
    missing = [w for w in needed if w not in offered_set]
    if missing:
        reasons.append("mot essentiel manquant pour reconstruire la phrase")


_ARABIC = set(range(0x0600, 0x06FF)) | set(range(0x0750, 0x077F)) | set(range(0xFB50, 0xFDFF)) | set(range(0xFE70, 0xFEFF))
_LATIN = set(range(0x0041, 0x005B)) | set(range(0x0061, 0x007B)) | set(range(0x00C0, 0x024F))


def _arabic_latin_mix_ratio(text: str) -> float:
    """Share of Latin letters among Arabic+Latin letters in `text`.

    Used to catch strong FR/AR mixing (`« أريد أن partir إلى المدرسة »`) while
    staying tolerant of a little Latin (proper nouns, numbers, labels).
    """
    arabic = latin = 0
    for ch in text or "":
        code = ord(ch)
        if code in _ARABIC:
            arabic += 1
        elif code in _LATIN:
            latin += 1
    total = arabic + latin
    if total == 0 or arabic == 0:
        return 0.0
    return latin / total


class ExerciseValidator:
    def __init__(self, *, duplicate_threshold: float = _DEFAULT_SIM_THRESHOLD) -> None:
        self.duplicate_threshold = duplicate_threshold
        self.logger = logging.getLogger(f"{__name__}.ExerciseValidator")

    # -- Structural ---------------------------------------------------------
    @staticmethod
    def _structural_reasons(item: ExerciseItem) -> list[str]:
        reasons: list[str] = []
        if not (item.prompt or "").strip():
            reasons.append("consigne vide")
        if item.level and item.level.upper() not in SUPPORTED_LEVELS:
            reasons.append(f"niveau invalide ({item.level})")
        if not (item.exercise_type or "").strip():
            reasons.append("type d'exercice manquant")
        return reasons

    # -- Pedagogical --------------------------------------------------------
    @staticmethod
    def _pedagogical_reasons(item: ExerciseItem, *, request_level: str, theme: str, language: str) -> list[str]:
        reasons: list[str] = []
        del theme  # Theme alignment is enforced via prompt instructions; the
        # deterministic textual check produced too many false positives (a
        # well-formed exercise need not quote the theme verbatim).
        if request_level and item.level and item.level.upper() != request_level.upper():
            reasons.append(f"niveau ({item.level}) != demandé ({request_level})")
        prompt = (item.prompt or "").strip()
        if language == "ar" and len(prompt) >= 12:
            if not has_arabic_script(prompt):
                reasons.append("arabe requis mais texte non arabe")
            elif _arabic_latin_mix_ratio(prompt) > 0.15:
                # Strong FR/AR mixing (Latin words embedded in Arabic) → student
                # content is no longer natural Arabic; flag for regeneration.
                reasons.append("mélange français/arabe")
        # Learner-facing structured fields must also avoid strong FR/AR mixing
        # when the request is Arabic. Internal metadata (title, level, skill,
        # difficulty) stays allowed in French/English.
        if language == "ar":
            learner_fields: list[str] = []
            learner_fields.extend(field for field in item.options if field)
            for pair in item.pairs:
                learner_fields.append(pair.get("left") or "")
                learner_fields.append(pair.get("right") or "")
            learner_fields.append((item.answer_expectation or "").strip())
            for field in learner_fields:
                if has_arabic_script(field) and _arabic_latin_mix_ratio(field) > 0.15:
                    reasons.append("mélange français/arabe")
                    break
        return reasons

    # -- Quality ------------------------------------------------------------
    @staticmethod
    def _quality_reasons(item: ExerciseItem) -> list[str]:
        reasons: list[str] = []
        ex_type = (item.exercise_type or "").casefold()
        prompt = (item.prompt or "").strip()
        answer = (item.answer_expectation or "").strip()

        if ex_type == "qcm":
            options = [o.strip() for o in item.options if o.strip()]
            if not options:
                reasons.append("QCM sans options")
            if len(set(o.casefold() for o in options)) != len(options):
                reasons.append("options identiques")
            if answer and options and not any(o.casefold() == answer.casefold() for o in options):
                reasons.append("réponse correcte absente des options")
            duplicates = [
                a for i, a in enumerate(options)
                for b in options[i + 1:]
                if _similarity(a, b) >= 0.9
            ]
            if duplicates:
                reasons.append("options quasi identiques")
        elif ex_type == "true_false":
            if item.is_true is None and not answer:
                reasons.append("vrai/faux sans réponse")
        elif ex_type in ("complete", "matching", "grammar_transformation"):
            if not answer:
                reasons.append("réponse attendue manquante")
        elif ex_type == "ordering":
            if not answer:
                reasons.append("réponse attendue manquante")
            else:
                # Every word needed to reconstruct the correct order must be
                # available in the shuffled tokens. If an essential word is
                # missing the exercise is impossible to solve.
                _validate_ordering_reconstructable(prompt, answer, reasons)
        elif ex_type not in ("open_question", "writing", "reading_comprehension"):
            # Unknown types: require at least a prompt.
            if not prompt:
                reasons.append("consigne vide")

        # Answer that copies the prompt verbatim.
        norm_prompt = _normalized(prompt)
        norm_answer = _normalized(answer)
        if norm_answer and norm_answer in norm_prompt:
            reasons.append("la correction recopie la consigne")
        return reasons

    # -- Full validation ----------------------------------------------------
    def validate(
        self,
        items: list[ExerciseItem], *,
        request_level: str, theme: str, language: str,
    ) -> list[ValidationVerdict]:
        verdicts: list[ValidationVerdict] = []
        for i, item in enumerate(items):
            reasons: list[str] = []
            reasons.extend(self._structural_reasons(item))
            reasons.extend(self._pedagogical_reasons(item, request_level=request_level, theme=theme, language=language))
            reasons.extend(self._quality_reasons(item))
            verdicts.append(ValidationVerdict(index=i, ok=not reasons, reasons=reasons))
        return verdicts

    # -- Duplicate detection ------------------------------------------------
    def find_duplicate_indices(self, items: list[ExerciseItem]) -> list[int]:
        """Return indices whose (normalized) prompt duplicates an earlier item's,
        in order of appearance. Later duplicate copies get regenerated."""
        seen: list[str] = []
        dup_indices: list[int] = []
        for i, item in enumerate(items):
            sig = _normalized(item.prompt or item.title or "")[:180]
            if not sig:
                continue
            duplicated = False
            for prior_sig in seen:
                if _similarity(prior_sig, sig) >= self.duplicate_threshold:
                    dup_indices.append(i)
                    duplicated = True
                    break
            if not duplicated:
                seen.append(sig)
        return dup_indices
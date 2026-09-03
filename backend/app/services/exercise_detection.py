"""Deterministic multi-signal detection and typing of real exercises.

The rule is: the mere presence of the word "exercice"/"exercise"/"تمرين" is
NEVER sufficient. A chunk is a usable exercise only when it shows task
structure (a directive + items to perform). Descriptive sentences about
exercises (prefaces, methodology, catalogue text) are negative signals and are
excluded by a configurable scoring threshold.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

_TITLE_RE = re.compile(
    r"^\s*(?:exercice|exercise|activité|activity|تمرين|نشاط)\s*[0-9٠-٩]+\b",
    re.IGNORECASE,
)
_INSTRUCTION_RE = re.compile(
    r"^\s*(?:consigne|directive|instruction|consigne pédagogique|تعليمات|توجيه)\s*[:：]",
    re.IGNORECASE,
)
_NUMBERED_ITEM_RE = re.compile(
    r"^\s*(\d+[\.\)]|[٠-٩]+[\.\)]|[أ-ز][\.\))]|\(?[a-z]\))\s+",
    re.IGNORECASE,
)
_CHOICE_ITEM_RE = re.compile(
    r"^\s*\(?\s*[a-z]\)\s+", re.IGNORECASE,
)
_ARABIC_CHOICE_RE = re.compile(r"^\s*[أ-ز][\.\)]\s+")
# Real Arabic worksheets (Miftah / Sokkan alJarra) list options under a
# "الاختيارات:"/"الاختيارات" header, each prefixed by a hyphen.  A block that
# carries such a choices list is a genuine (often fragmented) multiple-choice
# exercise stem, distinct from a descriptive preface.
_ARABIC_CHOICES_HEADER_RE = re.compile(r"^\s*الاختيارات\s*[:：]?\s*$", re.IGNORECASE)
_ARABIC_OPTION_LINE_RE = re.compile(r"^\s*[-–—]\s*\S+")
_BLANK_RE = re.compile(r"_{2,}|\[\.\.\.\]|\(\.\.\.\)|\.\.\.\.")
_MCQ_HINT_RE = re.compile(r"\b(?:choisis|choisissez|entoure|entourez|bonne réponse|إختر|اختر|اختر الإجابة|اختر الإجابة الصحيحة|ضع دائرة)\b", re.IGNORECASE)
_MATCHING_RE = re.compile(r"\b(?:relie|reliez|associe|associez|trouve le correspondant|اربط|صِل|وصل|طابق|صل بين)\b", re.IGNORECASE)
_TRUE_FALSE_RE = re.compile(r"\b(?:vrai ou faux|vrai|faux|صواب|خطأ|غير صحيح|صح أو خطأ|صح وخطأ|الصحيح والخطأ)\b", re.IGNORECASE)
_ORDERING_RE = re.compile(r"\b(?:remettre|remettez|remets|mets en ordre|mettez en ordre|range|rangez|organise|organisez|en ordre|رتب|رتّب|رتّبي)\b", re.IGNORECASE)
_TRANSFORMATION_RE = re.compile(r"\b(?:transforme|transformez|conjugue|conjuguez|mets au|mettez au|حوّل|صرّف|صرف الأفعال)\b", re.IGNORECASE)
_OPEN_QUESTION_RE = re.compile(r"[؟?؛;]\s*$|(?:réponds|répondez|qui\b|que\b|quoi\b|où\b|quand\b|pourquoi\b|comment\b|أجب|أين|متى|لماذا|كيف|من\b|ماذا\b)", re.IGNORECASE)
_PRODUCTION_RE = re.compile(r"\b(?:production écrite|rédige|rédigez|écris un|écrivez|petit texte|أنتج نص|اكتب نص|تعبير كتابي|en quelques phrases)\b", re.IGNORECASE)
_FILL_BLANK_VERB_RE = re.compile(r"\b(?:complète|complétez|remplis|remplissez|compléter|أكمل|أكمل الفراغ|املأ|املا)\b", re.IGNORECASE)

# Descriptive / preface / methodology wording: strong negative signals.
_DESCRIPTIVE_RE = re.compile(
    r"(?:"
    r"ce manuel|ce cahier|cet ouvrage|la présente méthode"
    r"|des exercices (?:sont|seront) proposés|les exercices (?:proposés|présentés)"
    r"|permet.*aux élèves|permettent aux élèves"
    r"|nous avons conçu|années de (?:recherche|travail)"
    r"|pour chaque niveau|par niveau"
    r"|cahier d'activités"
    r"|présentation générale|avant-propos|introduction"
    r"|méthodologie|démarche pédagogique|guide pédagogique"
    r"|exercices de (?:production|vocabulaire|grammaire) (?:sont|seront) présentés"
    r"|يقدم هذا الكتاب|يشمل هذا|يتضمن هذا|سلسلة|منهجية|مقدمة|لاحظ أن"
    r")",
    re.IGNORECASE,
)

# Positive signal weights; negative signals subtract. Weights live in one place
# so behaviour is tunable/testable without touching the extraction loop.
POSITIVE_WEIGHTS = {
    "title": 3,
    "instruction": 3,
    "numbered_items": 2,
    "choice_items": 2,
    "blanks": 1,
    "fill_blank_verb": 1,
    "mcq_hint": 1,
    "matching": 1,
    "true_false": 1,
    "ordering": 1,
    "transformation": 1,
    "open_question": 1,
    "production": 1,
}
NEGATIVE_WEIGHTS = {
    "descriptive_rhetoric": 4,
    "no_items": 2,
}


@dataclass(frozen=True)
class ExerciseDetectionResult:
    is_exercise: bool
    confidence: float  # 0..1, from raw score via stronger-than-linear squash
    raw_score: float
    exercise_type: str | None
    reasons: tuple[str, ...] = ()


def _signals(content: str, heading_context: list[str]) -> dict[str, bool]:
    heading_text = " ".join(heading_context)
    text = f"{heading_text}\n{content}"
    lines = [line for line in content.splitlines() if line.strip()]
    numbered = 0
    choices = 0
    options_open = False
    for line in lines:
        if _NUMBERED_ITEM_RE.match(line):
            numbered += 1
        if _CHOICE_ITEM_RE.match(line) or _ARABIC_CHOICE_RE.match(line):
            choices += 1
        # Enter/exit an Arabic choices block and count its hyphen options once
        # the header has been seen (works with or without a preceding title).
        if _ARABIC_CHOICES_HEADER_RE.match(line):
            options_open = True
        elif line.casefold().startswith(("choisis", "choisissez", "entoure", "entourez", "complète", "complétez", "relie", "reliez", "associe", "associez", "إختر", "اختر", "أكمل", "اربط", "طابق")):
            options_open = False
        elif options_open and _ARABIC_OPTION_LINE_RE.match(line):
            choices += 1
    threshold_items = 1 if heading_text and len(lines) <= 2 else 2
    return {
        "title": bool(_TITLE_RE.search(text)),
        "instruction": bool(_INSTRUCTION_RE.search(text)),
        "numbered_items": numbered >= threshold_items,
        "choice_items": choices >= threshold_items,
        "blanks": bool(_BLANK_RE.search(content)),
        "fill_blank_verb": bool(_FILL_BLANK_VERB_RE.search(text)),
        "mcq_hint": bool(_MCQ_HINT_RE.search(text)),
        "matching": bool(_MATCHING_RE.search(text)),
        "true_false": bool(_TRUE_FALSE_RE.search(text)),
        "ordering": bool(_ORDERING_RE.search(text)),
        "transformation": bool(_TRANSFORMATION_RE.search(text)),
        "open_question": bool(len(lines) <= 4 and _OPEN_QUESTION_RE.search(text)),
        "production": bool(_PRODUCTION_RE.search(text)),
        "descriptive_rhetoric": bool(_DESCRIPTIVE_RE.search(text)),
        "no_items": numbered == 0 and choices == 0 and not _BLANK_RE.search(content),
    }


def score_exercise(content: str, heading_context: list[str] | None = None,
                   *, threshold: float = 2.0) -> ExerciseDetectionResult:
    """Score a chunk-like text; a real exercise must pass the threshold."""
    signals = _signals(content, heading_context or [])
    raw = sum(POSITIVE_WEIGHTS[name] for name, present in signals.items() if present and name in POSITIVE_WEIGHTS)
    raw -= sum(NEGATIVE_WEIGHTS[name] for name, present in signals.items() if present and name in NEGATIVE_WEIGHTS)
    confidence = 1.0 / (1.0 + math.exp(-raw))  # bounded [0,1], monotone in raw
    is_exercise = raw >= threshold
    reasons = tuple(
        name for name, present in signals.items()
        if present and (name in POSITIVE_WEIGHTS or name in NEGATIVE_WEIGHTS)
    )
    return ExerciseDetectionResult(
        is_exercise=is_exercise,
        confidence=round(confidence, 2),
        raw_score=round(raw, 2),
        exercise_type=classify_exercise_type(content, heading_context or []) if is_exercise else None,
        reasons=reasons,
    )


def classify_exercise_type(content: str, heading_context: list[str] | None = None) -> str:
    """Deterministic lexical typing, applied to blocks already accepted."""
    text = f"{' '.join(heading_context or [])}\n{content}"
    candidates: list[tuple[str, str]] = []
    for type_name, pattern in (
        ("matching", _MATCHING_RE),
        ("true_false", _TRUE_FALSE_RE),
        ("ordering", _ORDERING_RE),
        ("transformation", _TRANSFORMATION_RE),
        ("production", _PRODUCTION_RE),
    ):
        if pattern.search(text):
            candidates.append((type_name, pattern.pattern))
    if _BLANK_RE.search(content) or _FILL_BLANK_VERB_RE.search(text):
        candidates.append(("fill_blank", "blank_fill"))
    if _MCQ_HINT_RE.search(text) or _CHOICE_ITEM_RE.search(text) or _ARABIC_CHOICE_RE.search(text) or (
        re.search(r"^الاختيارات\s*[:：]?\s*$", text, re.IGNORECASE | re.MULTILINE)
        and re.search(r"^[-–—]\s*\S+", text, re.MULTILINE)
    ):
        candidates.append(("qcm", "mcq_hint"))
    if _OPEN_QUESTION_RE.search(text):
        candidates.append(("open_question", "open_question"))
    if not candidates:
        return "mixed"
    return candidates[0][0]
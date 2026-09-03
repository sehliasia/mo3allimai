"""PedagogicalPlanner — the pedagogical intelligence behind the exercise tool.

This module turns the teacher's coarse request (level, theme, skills, count)
plus the RAG/CEFR pedagogical context into a structured, actionable plan:

- per-objective learning goals;
- target vocabulary and grammar to reuse (drawn from the KB blocks when they
  provide relevant material, never invented);
- an ordered distribution of exercise types that fits the selected skill and
  the requested count;
- a progression rationale (guided → productive).

It is deliberately deterministic: it does NOT call an LLM. The LLM (in the
generation service) turns this plan into concrete exercises. Deterministic
rules here are guards/planning only and explicitly leverage the retrieved
pedagogical context rather than pretending to be "smart".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

from app.services.exercise_cefr import LEVEL_RULES
from app.services.pedagogical_knowledge_service import PedagogicalContext

logger = logging.getLogger(__name__)

SUPPORTED_LEVELS = ("A1", "A2", "B1", "B2", "C1", "C2")

# Ordered progression phases, guided → productive. Used as the default backbone
# of every distribution; skills then pick which types are relevant.
_PHASES = (
    "recognition",
    "comprehension",
    "application",
    "manipulation",
    "production",
)

# Helpers for building a natural-language objective from a skill label.
_SKILL_LABELS: dict[str, str] = {
    "vocabulaire": "le vocabulaire",
    "vocabulary": "le vocabulaire",
    "grammaire": "la grammaire",
    "grammar": "la grammaire",
    "compréhension écrite": "la compréhension écrite",
    "compréhension orale": "la compréhension orale",
    "expression écrite": "l'expression écrite",
    "expression orale": "l'expression orale",
    "production écrite": "l'expression écrite",
    "production orale": "l'expression orale",
    "lecture": "la lecture",
    "vocabulaire et grammaire": "le vocabulaire et la grammaire",
}


def _normalize_skill(value: str) -> str:
    return (value or "").strip().casefold()


def _skill_base(skills: Sequence[str]) -> str:
    """Return the dominant skill stem used to pick types, or ''."""
    if not skills:
        return ""
    first = _normalize_skill(skills[0])
    lowered = " ".join(_normalize_skill(s) for s in skills)
    if "vocabulaire" in lowered or "vocabulary" in lowered:
        return "vocabulary"
    if "grammaire" in lowered or "grammar" in lowered:
        return "grammar"
    if "compréhension écrite" in lowered or "written comprehension" in lowered or "lecture" in lowered:
        return "reading"
    if "compréhension orale" in lowered or "listening" in lowered:
        return "listening"
    if "expression écrite" in lowered or "production écrite" in lowered or "writing" in lowered:
        return "writing"
    if "expression orale" in lowered or "production orale" in lowered or "speaking" in lowered:
        return "speaking"
    _ = first
    return ""


# Exercise type catalog (canonical ids). Frontend maps these to labels/rendering.
EXERCISE_TYPES = (
    "qcm",
    "complete",
    "true_false",
    "matching",
    "ordering",
    "grammar_transformation",
    "reading_comprehension",
    "writing",
    "open_question",
)

# Which types suit which skill. Skill-specific distributions replace this only
# when the skill is explicit (e.g. "expression écrite" → guided writing).
_SKILL_TYPE_PREFS: dict[str, tuple[str, ...]] = {
    "vocabulary": ("recognition", "qcm", "matching", "complete"),
    "grammar": ("qcm", "grammar_transformation", "complete", "ordering"),
    "reading": ("reading_comprehension", "true_false", "ordering", "open_question"),
    "listening": ("true_false", "complete", "open_question"),
    "writing": ("writing", "open_question"),
    "speaking": ("open_question", "true_false"),
}

# Default ordered list, guided → productive (the core progression).
_DEFAULT_SEQUENCE = (
    "recognition",
    "qcm",
    "complete",
    "matching",
    "true_false",
    "grammar_transformation",
    "ordering",
    "reading_comprehension",
    "open_question",
    "writing",
)

# Count → recommended spread of phases (how many from each phase).
# Drives distribution length by target count.
_PHASE_SHARE_BY_COUNT: dict[int, tuple[int, int, int, int, int]] = {
    # (recognition, comprehension, application, manipulation, production)
    1: (1, 0, 0, 0, 0),
    2: (1, 1, 0, 0, 0),
    3: (1, 1, 1, 0, 0),
    4: (1, 1, 1, 1, 0),
    5: (1, 1, 1, 1, 1),
    6: (2, 1, 1, 1, 1),
    7: (2, 2, 1, 1, 1),
    8: (2, 2, 2, 1, 1),
    9: (2, 2, 2, 2, 1),
    10: (2, 2, 2, 2, 2),
    15: (3, 4, 3, 3, 2),
}


def _cap_count(count: int) -> int:
    return max(1, min(15, count))


def _shares_for(count: int) -> tuple[int, int, int, int, int]:
    target = _cap_count(count)
    return _PHASE_SHARE_BY_COUNT[target]


# Map each canonical type to its progression phase so a distribution can be
# ordered guided → productive while still honouring per-skill preferences.
_TYPE_PHASE: dict[str, str] = {
    "recognition": "recognition",
    "qcm": "comprehension",
    "true_false": "comprehension",
    "matching": "application",
    "complete": "application",
    "ordering": "manipulation",
    "grammar_transformation": "manipulation",
    "reading_comprehension": "comprehension",
    "open_question": "production",
    "writing": "production",
}


@dataclass(frozen=True)
class ExercisePlan:
    level: str
    theme: str
    skills: list[str] = field(default_factory=list)
    objective: str = ""
    learning_objectives: list[str] = field(default_factory=list)
    target_vocabulary: list[str] = field(default_factory=list)
    target_grammar: list[str] = field(default_factory=list)
    exercise_distribution: list[str] = field(default_factory=list)
    rationale: str = ""


def _pick_distribution(skill_base: str, count: int) -> list[str]:
    """Choose an ordered distribution that fits skill + count.

    Deterministic, guided → productive. Uses the requested count to control the
    length; uses the skill only to bias which types are relevant (never a mock:
    the ordering and selection still follow the pedagogical phases above).
    """
    target = _cap_count(count)
    if skill_base == "writing":
        # Expression écrite → guided writing sequence.
        seq = ("recognition", "grammar_transformation", "complete", "open_question", "writing")
    elif skill_base == "grammar":
        seq = ("recognition", "qcm", "complete", "grammar_transformation", "ordering", "writing")
    elif skill_base == "reading":
        seq = ("reading_comprehension", "true_false", "ordering", "open_question", "writing")
    else:
        seq = _DEFAULT_SEQUENCE

    # Fill the target count by cycling the preferred sequence.
    distribution: list[str] = []
    idx = 0
    while len(distribution) < target:
        distribution.append(seq[idx % len(seq)])
        idx += 1
    return distribution


# Curated, concrete A2-friendly themes (Arabic labels, student-facing). Used
# only when the RAG pool offers no usable theme keyword and the request was
# generic — never as a hard-coded FR/AR dictionary, just a safe fallback set.
_FALLBACK_THEMES: tuple[str, ...] = (
    "الحياة اليومية",
    "الأسرة",
    "المدرسة",
    "السفر",
    "الهوايات",
    "الطعام",
    "البيت",
    "العمل",
)

# Readable Latin labels mirroring the Arabic fallbacks (for non-Arabic UI
# headers); these are not translations of arbitrary user text.
_FALLBACK_THEME_LABELS: dict[str, str] = {
    "الحياة اليومية": "vie quotidienne",
    "الأسرة": "famille",
    "المدرسة": "école",
    "السفر": "voyage",
    "الهوايات": "loisirs",
    "الطعام": "nourriture",
    "البيت": "maison",
    "العمل": "travail",
}

# Generic request markers (French UI). A theme made of these words is a general
# request, not a real topic — never use it as the exercise theme.
_GENERIC_THEME_MARKERS = {
    "je", "veux", "veut", "faire", "crée", "créer", "génère", "générer", "donne",
    "donner", "propose", "proposer", "prépare", "souhaite", "ai", "des", "de",
    "exercice", "exercices", "activité", "activités", "fiche", "fiches", "séance",
    "niveau", "svp", "stp", "merci", "s'il", "ça", "un", "une", "le", "la", "les",
    "pour", "sur", "avec", "en", "et", "ou", "session", "the", "a", "an", "in", "of",
    # CEFR tokens are a level, not a topic — never carry them into a theme.
    "a1", "a2", "b1", "b2", "c1", "c2", "élémentaire", "intermédiaire",
}

# Theme keywords the planner searches for inside the RAG blocks (Arabic + a few
# Latin routing hints). Presence in a heading/content is a signal of the topic.
_RAG_THEME_KEYWORDS: dict[str, tuple[tuple[str, str], ...]] = {
    "الحياة اليومية": (("الحياة", "اليومية"), ("routine", "صَبَاح"), ("استيقظ",), ("biographie quotidienne",)),
    "الأسرة": (("الأسرة", "العائلة"), ("فاميل",), ("famille",)),
    "المدرسة": (("المدرسة", "الدرس", "الأستاذ"), ("école", "الطالب"), ("classe",)),
    "السفر": (("السفر", "السافرة"), ("رحلة", "سياحة"), ("voyage",)),
    "الهوايات": (("الهوايات", "ألعاب"), ("رياضة", "قراءة"), ("loisirs",)),
    "الطعام": (("الطعام", "الأكل"), ("غذاء", "مطبخ"), ("nourriture",)),
    "البيت": (("البيت", "المنزل"), ("غرفة", "طبخ"), ("maison",)),
    "العمل": (("العمل", "المهنة"), ("وظيفة", "شركة"), ("travail",)),
}


def _generic_theme(theme: str) -> bool:
    """Whether the incoming theme is essentially a general request, not a topic."""
    text = (theme or "").strip()
    if not text or len(text) <= 3:
        return True
    words = [w for w in text.split()]
    if not words:
        return True
    # A theme is "specific" if it carries at least one content word outside the
    # generic request markers / common stopwords (e.g. "Famille", "Voyage").
    meaningful = [
        w for w in words
        if w.casefold() not in _GENERIC_THEME_MARKERS
        and w.casefold() not in _STOP_TOKENS
    ]
    if meaningful:
        return False
    # Only markers/stopwords → a general request, not a real topic.
    return True


def _theme_from_rag(context: PedagogicalContext | None) -> str | None:
    """Pick a concrete Arabic A2 theme if the RAG blocks clearly reference one."""
    if context is None or not context.resource_blocks:
        return None
    harvested = []
    for block in context.resource_blocks[:12]:
        heading = " ".join(block.heading_context or [])
        content = (block.content or "")[:500]
        harvested.append((heading + " " + content).casefold())
    joined = " ".join(harvested)
    for theme, keywords in _RAG_THEME_KEYWORDS.items():
        for key_tuple in keywords:
            if any(k in joined for k in key_tuple):
                return theme
    return None


def _resolve_theme(theme: str, context: PedagogicalContext | None, *, language: str, level: str) -> str:
    """Return a usable theme, preferring a real one and avoiding prose leakage.

    - Keep the user's theme when it is a concrete topic (not a general request).
    - Otherwise try to surface a topic from the RAG pool.
    - Otherwise fall back to a familiar, concrete theme suited to `level`.
    """
    if not _generic_theme(theme):
        return theme.strip() or "اللغة العربية"
    derived = _theme_from_rag(context)
    if derived:
        return derived
    # Choose a level-appropriate familiar theme from the curated fallback list.
    label = _FALLBACK_THEMES[0]  # الحياة اليومية
    if level.upper() in ("B1", "B2", "C1", "C2"):
        label = _FALLBACK_THEMES[-1] if level.upper() in ("B2", "C1", "C2") else _FALLBACK_THEMES[3]
    return label


class PedagogicalPlanner:
    """Deterministic planner turning request + RAG context → ExercisePlan."""

    def __init__(self) -> None:
        self.logger = logging.getLogger(f"{__name__}.PedagogicalPlanner")

    def plan(
        self, *,
        level: str, theme: str, skills: Sequence[str],
        count: int, objective: str | None,
        context: PedagogicalContext | None,
        language: str = "",
    ) -> ExercisePlan:
        level = level.upper() if level in SUPPORTED_LEVELS else "A1"
        skills_norm = [s for s in skills]
        skill_base = _skill_base(skills_norm)

        # Resolve the theme: a generic request must never leak its prose as the
        # theme. When the incoming theme is essentially a general request (e.g.
        # "Je veux des exercices de niveau A2"), derive a coherent, concrete
        # theme from the RAG blocks, else a curated familiar A2 theme.
        theme = _resolve_theme(theme, context, language=language, level=level)

        # Target vocabulary / grammar: reuse relevant tokens from the KB blocks
        # (not invented). We collect a compact, deduplicated set, filtered for
        # the requested script (no Latin/French leakage when the language is
        # Arabic) and bounded by the CEFR level.
        target_vocab, target_grammar = self._collect_targets(
            context, language=language, level=level,
        )

        distribution = _pick_distribution(skill_base, count)
        learning_objectives = _learning_objectives(level, theme, skills_norm, skill_base)
        objective_sentence = objective or (
            f"Réaliser {count} exercices progressifs sur « {theme} » "
            f"au niveau {level} pour travailler {','.join(skills_norm) if skills_norm else 'la langue'}."
        )
        rationale = _rationale(level, skill_base, distribution, language=language)
        self.logger.info(
            "[PLANNER] plan level=%s theme=%s skill=%s count=%s distribution=%s",
            level, theme, skill_base,
            count, ",".join(distribution),
        )
        return ExercisePlan(
            level=level, theme=theme, skills=list(skills_norm),
            objective=objective_sentence,
            learning_objectives=learning_objectives,
            target_vocabulary=target_vocab,
            target_grammar=target_grammar,
            exercise_distribution=distribution,
            rationale=rationale,
        )

    @staticmethod
    def _collect_targets(
        context: PedagogicalContext | None, *,
        language: str = "", level: str = "A1",
    ) -> tuple[list[str], list[str]]:
        """Harvest a compact target vocabulary / grammar from the KB blocks.

        We reuse the content of the retrieved pedagogical blocks (theme-relevant
        vocabulary and grammar), deduplicated and capped, so the plan reflects
        the knowledge base rather than inventing arbitrary words.

        For an Arabic request we keep only Arabic-script tokens (no Latin /
        French leakage into target_vocabulary) and only Arabic grammar tokens,
        so a French conjunction such as "mais" can never become the target
        grammar of an Arabic exercise plan.
        """
        if context is None or not context.resource_blocks:
            return [], []
        want_arabic = (language or "").casefold() == "ar"
        vocab: list[str] = []
        grammar: list[str] = []
        seen_vocab: set[str] = set()
        seen_grammar: set[str] = set()
        for block in context.resource_blocks:
            heading = block.heading_context[0] if block.heading_context else ""
            content = (block.content or "")[:400]
            heading_lower = heading.casefold()
            is_grammar = any(k in heading_lower for k in ("gramm", "conjugaison", "syntaxe", "قواعد"))
            # Light token harvesting from Arabic + neutral tokens.
            tokens = _token_harvest(content + " " + heading, want_arabic=want_arabic)
            for token, is_grammar_word in tokens:
                if want_arabic and not _is_arabic_word(token):
                    # Keep the RAG as a source of knowledge, but never let a
                    # Latin/French token leak into an Arabic plan's targets.
                    continue
                if is_grammar or is_grammar_word:
                    key = token.casefold()
                    if key not in seen_grammar and len(grammar) < 8:
                        seen_grammar.add(key)
                        grammar.append(token)
                else:
                    key = token.casefold()
                    if key not in seen_vocab and len(vocab) < 12:
                        seen_vocab.add(key)
                        vocab.append(token)
            if len(vocab) >= 12 and len(grammar) >= 8:
                break
        return vocab, grammar


_ARABIC_RANGE = (range(0x0600, 0x06FF), range(0x0750, 0x077F), range(0xFB50, 0xFDFF), range(0xFE70, 0xFEFF))


def _is_arabic_word(word: str) -> bool:
    return any(any(ord(ch) in rng for rng in _ARABIC_RANGE) for ch in word)


_STOP_TOKENS = {
    "le", "la", "les", "un", "une", "des", "du", "de", "et", "ou", "à", "au",
    "aux", "pour", "avec", "dans", "sur", "il", "elle", "je", "tu", "nous",
    "vous", "ils", "elles", "ce", "cette", "ces", "cet", "qu", "que", "qui",
    "en", "pas", "plus", "très", "niveau", "exercice", "activité", "thème",
    "the", "a", "an", "and", "of", "to", "in", "for", "with",
}


def _token_harvest(text: str, *, want_arabic: bool = False) -> list[tuple[str, bool]]:
    """Cheap light token harvesting (no heavy model). Returns (token, is_grammarish).

    Grammar-ish is an heuristic on Latin stop-grammar verbs / Arabic particles;
    it is only a suggestion to the plan, not a strict classification. When the
    plan targets Arabic (`want_arabic`), Latin/French tokens are dropped entirely
    so a French conjunction such as "mais" can never surface as the target
    grammar of an Arabic exercise plan.
    """
    import re as _re
    tokens: list[tuple[str, bool]] = []
    lowered = text.casefold()
    for word in _re.findall(r"[\w\u0600-\u06FF]+", lowered):
        if not word or word in _STOP_TOKENS or len(word) < 2:
            continue
        if want_arabic and not _is_arabic_word(word):
            continue
        is_grammar = word in (
            "et", "ou", "mais", "car", "donc", "ne", "pas", "le", "la",
            "préposition", "conjugaison", "verbe", "nom", "أ", "في", "على",
            "من", "إلى", "عن", "و", "ثم", "لكن", "لأن",
        )
        tokens.append((word, is_grammar))
    return tokens


def _learning_objectives(level: str, theme: str, skills: Sequence[str], skill_base: str) -> list[str]:
    """Deterministic per-objective goals anchored in the level + selected skills."""
    skill_labels = [s.strip() for s in skills if s.strip()]
    if not skill_labels:
        skill_labels = ["la langue"]
    human_skills = " et ".join(_SKILL_LABELS.get(_normalize_skill(s), s) for s in skill_labels)
    objectives = [
        f"Identifier et réemployer le vocabulaire essentiel lié au thème « {theme} ».",
        f"Travailler {human_skills} au niveau {level} avec des consignes adaptées.",
        "Progresser du plus guidé au plus libre pour consolider la maîtrise.",
    ]
    if skill_base == "grammar":
        objectives.insert(1, "Reconnaître et appliquer les structures grammaticales ciblées.")
    elif skill_base == "writing":
        objectives.insert(1, "Produire des énoncés écrits courts et structurés.")
    return objectives


def _rationale(level: str, skill_base: str, distribution: list[str], *, language: str = "") -> str:
    phase_count: dict[str, int] = {}
    for t in distribution:
        phase = _TYPE_PHASE.get(t, "application")
        phase_count[phase] = phase_count.get(phase, 0) + 1
    phases_used = [p for p in _PHASES if phase_count.get(p, 0) > 0]
    base = (
        f"Distribution ({len(distribution)} exercices) ordonnée des phases "
        f"{' → '.join(phases_used)} : on reconnaît d'abord, puis on comprend, on applique, "
        f"on manipule et enfin on produit."
    )
    if (language or "").casefold() == "ar":
        # Keep the Arabic-facing rationale short and neutral; drop the French
        # skill/level annotations that don't fit an Arabic plan.
        return base
    skill_note = {
        "writing": "La compétence « expression écrite » conduit à privilégier rédaction guidée et production.",
        "grammar": "La grammaire est travaillée par reconnaissance puis transformation avant la production.",
        "reading": "La compréhension écrite s'appuie sur un court texte adapté puis des questions progressives.",
    }.get(skill_base, "")
    rule = LEVEL_RULES.get(level, {})
    level_note = "; ".join(
        f"{label}: {value}" for label, value in list(rule.items())[:4]
    )
    return (
        base
        + (f" {skill_note}" if skill_note else "")
        + f" Contrainte CECRL {level} : {level_note}."
    ).strip()

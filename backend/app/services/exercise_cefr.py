"""Internal CEFR constraint layer for the exercise tool.

This layer is explicitly PEDAGOGICAL-INTERNAL: it is NOT an official Council of
Europe / CEFR citation and nothing here claims to be one. Whenever the
knowledge base already contains structured CEFR descriptors
(CEFRKnowledgeService backed by PostgreSQL), those descriptors remain the
authoritative source. This module only:

- documents internal "adaptation rules" that constrain how the LLM may reshape
  a task towards an explicit target level; and
- provides a deterministic, clearly-labelled *estimate* for items whose level
  is not stated by the source document (level_source="inferred"), never
  presenting an estimate as a proven value.
"""

from __future__ import annotations

import re

SUPPORTED_LEVELS: tuple[str, ...] = ("A1", "A2", "B1", "B2", "C1", "C2")

_ARABIC_SCRIPT_RE = re.compile(r"[\u0600-\u06FF]")


def has_arabic_script(text: str) -> bool:
    return bool(_ARABIC_SCRIPT_RE.search(text or ""))

# Arabic level labels map onto the same letters: أ= A, ب= B, ج= C.
_ARABIC_LEVEL_LETTERS = {"أ": "A", "ب": "B", "ج": "C"}

# Internal adaptation rules per level. These are guidance for the generator /
# adapter; they deliberately live in one place so the CEFR dimension is a real
# pedagogical layer instead of scattered prompt text.
LEVEL_RULES: dict[str, dict[str, str]] = {
    "A1": {
        "lexical": "vocabulaire fréquent et concret de la vie immédiate uniquement",
        "sentence": "phrases simples, ordre SVO/SVO direct, pas de subordination longue",
        "situations": "situations familières et concrètes (famille, école, objets quotidiens)",
        "tasks": "tâches très guidées : compléter, relier, choisir, répéter",
        "guidance": "consignes courtes et explicites ; un seul item par consigne",
        "interaction": "interaction élémentaire : se présenter, nommer, saluer",
    },
    "A2": {
        "lexical": "vocabulaire des situations quotidiennes (courses, routine, déplacements)",
        "sentence": "phrases plus développées avec connecteurs simples (et, ou, puis, parce que)",
        "situations": "situations de la vie courante prévisibles",
        "tasks": "tâches guidées avec une part d'initiative : décrire, donner des informations simples",
        "guidance": "consignes simples ; items courts",
        "interaction": "autonomie limitée mais croissante dans des échanges simples",
    },
    "B1": {
        "lexical": "lexique de la narration, de la description et des goûts",
        "sentence": "phrases structurées avec connecteurs logiques (donc, cependant, d'abord ensuite)",
        "situations": "situations variées dont certaines imprévues",
        "tasks": "narration, description, expression d'une opinion simple, comparaison, justification simple",
        "guidance": "consignes claires permettant une production personnelle courte",
        "interaction": "interaction autonome dans des situations familières et nouvelles",
    },
    "B2": {
        "lexical": "lexique précis et varié, y compris abstrait",
        "sentence": "phrases complexes et nuancées ; connecteurs argumentatifs (en revanche, par conséquent, en effet)",
        "situations": "sujets d'actualité et thèmes abstraits",
        "tasks": "argumentation, justification développée, comparaison fine, synthèse",
        "guidance": "consignes qui exigent une prise de position et une argumentation étayée",
        "interaction": "interaction plus autonome avec aisance relative",
    },
    "C1": {
        "lexical": "lexique étendu et registres variés",
        "sentence": "structures complexes maîtrisées ; implicite et nuance",
        "situations": "sujets complexes, abstraits ou techniques",
        "tasks": "argumentation complexe, analyse critique, reformulation, registre adapté",
        "guidance": "consignes ouvertes favorisant précision et nuance",
        "interaction": "autonomie élevée avec souplesse linguistique",
    },
    "C2": {
        "lexical": "lexique très étendu, idiomatismes, registre soutenu",
        "sentence": "maîtrise fine de la grammaire, de la nuance et de l'implicite",
        "situations": "tous les domaines, y compris académiques",
        "tasks": "argumentation complexe, synthèse critique, création, style personnel",
        "guidance": "consignes minimales laissant une liberté de production élevée",
        "interaction": "autonomie totale, stratégies de compensation invisibles",
    },
}

_LEVEL_HEADING_RE = re.compile(
    r"\b(?P<level>[A-C][12])(?i:)\b"
    r"|\b(?:niveau|niv\.)\s*(?P<lvl2>[A-C][12])"
    r"|\b(?:مستوى|المستوى)\s*(?P<ara>[أبج][12])",
    re.IGNORECASE,
)

_SIMPLE_CONNECTORS = (
    "et", "ou", "puis", "alors", "parce que", "avec", "و", "ثم", "أو", "لكن", "لأن",
)
_COMPLEX_CONNECTORS = (
    "cependant", "néanmoins", "par conséquent", "en revanche", "de plus", "ainsi",
    "bien que", "toutefois", "en effet", "auparavant", "par ailleurs",
    "وبالتالي", "مع ذلك", "علاوة على ذلك", "على الرغم من", "بالتالي", "بالمقابل",
)
_OPINION_MARKERS = (
    "opinion", "argumenter", "argument", "point de vue", "justifiez", "selon vous",
    "comparez", "discutez", "raison", "avis",
    "رأي", "حجة", "برر", "قارن", "ناقش", "دافع", "برّر",
)
_ABSTRACT_MARKERS = ("nuance", "registre", "implicite", "critique", "nuancé", "دقيق", "محكم", "نقد")

_WORD_RE = re.compile(r"\w+", re.UNICODE)
_SENTENCE_SPLIT_RE = re.compile(r"[.!?…]+|[\u060C\u061B]")

_MIN_SENTENCE_TOKENS = 4
_MAX_SENTENCE_TOKENS = 60


def normalize_level(value: str | None) -> str | None:
    """Return the canonical A1..C2 form, or None when the value is not usable."""
    if not value:
        return None
    normalized = value.strip().upper()
    if normalized in SUPPORTED_LEVELS:
        return normalized
    if normalized.startswith("PRE-"):
        candidate = normalized.replace("PRE-", "")
        if candidate in SUPPORTED_LEVELS:
            return candidate
    return None


def detect_explicit_level(*, content: str, heading_context: list[str] | None = None,
                          indexed_cefr_level: str | None = None) -> str | None:
    """Return an explicit source level when the document truly states one.

    Priority: indexed CEFR metadata (from the pipeline) > level stated in the
    heading > level stated in the content. Any explicit claim must map to the
    A1..C2 grid; otherwise the level is NOT explicit.
    """
    indexed = normalize_level(indexed_cefr_level)
    if indexed:
        return indexed
    evidence = "\n".join([*(heading_context or []), content[:500]])
    for match in _LEVEL_HEADING_RE.finditer(evidence):
        raw = match.group("level") or match.group("lvl2")
        if not raw:
            raw_value = match.group("ara")
            if raw_value:
                letter = _ARABIC_LEVEL_LETTERS.get(raw_value[0])
                if letter:
                    raw = f"{letter}{raw_value[1]}"
        if raw:
            normalized = normalize_level(raw)
            if normalized:
                return normalized
    return None


def estimate_level(content: str) -> tuple[str | None, float]:
    """Deterministic, clearly-labelled estimate. Never a proof of level."""
    text = re.sub(r"\[[^\]]*\]", " ", content)
    sentences = [s for s in _SENTENCE_SPLIT_RE.split(text) if len(_WORD_RE.findall(s)) >= _MIN_SENTENCE_TOKENS]
    if not sentences:
        return None, 0.0
    words_per_sentence = sum(len(_WORD_RE.findall(s)) for s in sentences) / len(sentences)
    lowered = content.casefold()
    complex_count = sum(1 for marker in _COMPLEX_CONNECTORS if marker in lowered)
    simple_count = sum(1 for marker in _SIMPLE_CONNECTORS if marker in lowered)
    opinion_count = sum(1 for marker in _OPINION_MARKERS if marker in lowered)
    abstract_count = sum(1 for marker in _ABSTRACT_MARKERS if marker in lowered)

    confidence = 0.0
    level: str | None = None
    # Anchor on strong productive markers first: opinion/argumentation imply B2+.
    if opinion_count >= 1 or abstract_count >= 2:
        level = "B2" if opinion_count >= 1 else "C1"
        confidence = 0.5
    elif complex_count >= 2 or words_per_sentence >= _MAX_SENTENCE_TOKENS:
        level = "B1"
        confidence = 0.45
    elif words_per_sentence <= 8 and simple_count <= 2:
        level = "A1"
        confidence = 0.55
    elif words_per_sentence <= 12:
        level = "A2"
        confidence = 0.5
    else:
        level = "B1"
        confidence = 0.4
    return level, round(min(confidence, 0.9), 2)
"""Deterministic parser for explicit, serialized CEFR descriptor rows."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field


_LEVEL = r"(?:PR(?:E|É)[- ]?A1|A1|A2\+?|B1\+?|B2\+?|C1|C2)"
# A row marker may be serialized on the same physical line as the previous
# descriptor.  Boundaries must therefore be based on the explicit ``LEVEL,
# scale =`` structure rather than line breaks introduced by a serializer.
_ROW_HEADER = re.compile(
    rf"(?<![\w+\-])(?P<level>{_LEVEL})(?![\w+\-])\s*[,;]\s*(?P<scale>[^\n=|]{{1,180}}?)\s*=",
    re.IGNORECASE,
)
_TABLE_ROW = re.compile(rf"(?m)^\s*(?P<level>{_LEVEL})\s*\|\s*(?P<text>[^\n|].*?)\s*$", re.IGNORECASE)
_NO_DESCRIPTOR = re.compile(
    rf"^pas\s+de\s+descripteur\s+disponible\s*(?:[,;:]\s*voir\s+(?P<reference>{_LEVEL}))?\s*[.!…]*$",
    re.IGNORECASE,
)
_TRAILING_FRAGMENT = re.compile(r"[,;:/-]$")
_NARRATIVE_START = re.compile(
    r"^(?:des\s+descripteurs\b|les\s+descripteurs\b|ce\s+(?:niveau|tableau)\b)",
    re.IGNORECASE,
)
_CANDIDATE_SIGNAL = re.compile(rf"(?<![\w+\-]){_LEVEL}(?![\w+\-])\s*(?:[,;|])", re.IGNORECASE)
_MID_SENTENCE_FRAGMENT = re.compile(r"^(?:[,;:]|(?:exemple|information|ainsi|mais|et|ou|avec|pour|dans|sur)\b)")
_EDITORIAL_COMMENTARY = re.compile(
    r"\b(?:plusieurs\s+modifications?|modifications?\s+(?:présentées?|apportées?|introduites?)|"
    r"liste\s+de\s+l['’]annexe|consulter\s+l['’]annexe|changements?\s+(?:dans|introduits?)|"
    r"(?:cette|la)\s+(?:édition|annexe)|version\s+(?:précédente|révisée))\b",
    re.IGNORECASE,
)
_SCALE_CONTAMINATION = re.compile(
    r"\b(?:annexe|consulter|page|section|chapitre|modifications?|a\s+été|a\s+ete)\b",
    re.IGNORECASE,
)
_PARENTHETICAL = re.compile(r"\([^()]*\)")
_SCALE_OUTSIDE_SENTENCE = re.compile(r"[.!?]")
_ORPHAN_SCALE_TAIL = re.compile(r"(?:[.!…]{2,})\s*,\s*[A-ZÀ-ÖØ-Þ].{3,}$", re.DOTALL)
_SERIALIZED_CELL = re.compile(r"(?:[.!…]\s*)?,\s*[A-ZÀ-ÖØ-Þ][^=\n|]{2,180}\s*=", re.DOTALL)
_LEADING_FRAGMENT = re.compile(
    r"^(?:[,;:]|[\)\]]|(?:enregistr[ée]s?|de|du|des|ainsi\s+que|et|ou|mais|avec|pour|dans|sur|à|au)\b)",
)


@dataclass(frozen=True)
class ParsedCEFRDescriptor:
    level_code: str
    scale_name: str
    descriptor_text: str | None
    normalized_text: str
    descriptor_hash: str
    status: str
    source_chunk_ids: list[int]
    reference_level: str | None = None
    reconstructed_from_fragments: bool = False


@dataclass(frozen=True)
class RejectedCEFRCandidate:
    reason: str
    source_chunk_ids: list[int]
    level_code: str | None = None
    scale_name: str | None = None
    text_preview: str = ""


@dataclass
class CEFRParseResult:
    records: list[ParsedCEFRDescriptor] = field(default_factory=list)
    rejected: list[RejectedCEFRCandidate] = field(default_factory=list)
    candidate_rows_detected: int = 0


def normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFC", value).replace("|", " ")
    normalized = " ".join(normalized.split())
    return re.sub(r"\s+([.!…])", r"\1", normalized)


def normalize_level(value: str) -> str:
    value = unicodedata.normalize("NFC", value).upper().replace(" ", "-")
    return value.replace("PRÉ", "PRE")


def descriptor_hash(level_code: str, scale_name: str, normalized_text: str, status: str, reference_level: str | None = None) -> str:
    payload = "\x1f".join((level_code, normalize_text(scale_name).casefold(), normalized_text, status, reference_level or ""))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_scale_name(value: str) -> str | None:
    """Return a deterministic rejection reason, or ``None`` for a usable scale.

    This intentionally does not keep a book-specific scale whitelist. It only
    rejects structural fragments that cannot safely be authoritative labels.
    """
    scale = normalize_text(value)
    if not any(character.isalpha() for character in scale):
        return "NUMERIC_OR_PUNCTUATION_SCALE"
    if _TRAILING_FRAGMENT.search(scale):
        return "INCOMPLETE_SCALE_FRAGMENT"
    if len(scale) > 220 or _SCALE_CONTAMINATION.search(scale):
        return "CONTAMINATED_SCALE"
    if scale.count("(") != scale.count(")"):
        return "INCOMPLETE_SCALE_FRAGMENT"
    # Parenthetical examples are legitimate CEFR scale context.  Narration
    # remains unsafe only when it leaks into the title outside that complete,
    # balanced structural context.
    outside_parentheses = _PARENTHETICAL.sub("", scale)
    if _SCALE_OUTSIDE_SENTENCE.search(outside_parentheses):
        return "CONTAMINATED_SCALE"
    if _CANDIDATE_SIGNAL.search(scale):
        return "CONTAMINATED_SCALE"
    words = scale.split()
    if len(words) == 1 and words[0].islower() and words[0].isascii():
        return "INCOMPLETE_SCALE_FRAGMENT"
    return None


def is_valid_long_scale(value: str) -> bool:
    """Identify complete explanatory titles accepted after structural checks."""
    scale = normalize_text(value)
    return "(" in scale and ")" in scale and validate_scale_name(scale) is None


def is_leading_fragment(value: str) -> bool:
    """Recognise structural continuation openings without rejecting normal text."""
    return bool(_LEADING_FRAGMENT.match(normalize_text(value)))


def has_incomplete_descriptor_tail(value: str) -> bool:
    text = normalize_text(value)
    return bool(
        _TRAILING_FRAGMENT.search(text)
        or text.count("(") > text.count(")")
        or text.count("[") > text.count("]")
    )


def classify_no_descriptor(value: str) -> str | None:
    """Return an optional referenced level for a deterministic absence marker."""
    match = _NO_DESCRIPTOR.fullmatch(normalize_text(value))
    return normalize_level(match.group("reference")) if match and match.group("reference") else "" if match else None


def validate_available_descriptor(value: str, *, source_chunk_ids: list[int], reject_leading_fragment: bool = True, reject_incomplete_tail: bool = True) -> str | None:
    """Return an acceptance-gate failure reason, never guessing missing text."""
    text = normalize_text(value)
    if not source_chunk_ids:
        return "MISSING_PROVENANCE"
    if not text:
        return "TRUNCATED_DESCRIPTOR"
    if classify_no_descriptor(text) is not None:
        return "NO_DESCRIPTOR_MISCLASSIFIED"
    if _ROW_HEADER.search(text):
        return "EMBEDDED_NEXT_ROW_MARKER"
    if _SERIALIZED_CELL.search(text):
        return "EMBEDDED_SERIALIZED_CELL"
    if _ORPHAN_SCALE_TAIL.search(text):
        return "TAIL_CONTAMINATION"
    if _NARRATIVE_START.match(text) or _EDITORIAL_COMMENTARY.search(text):
        return "NARRATIVE_FALSE_POSITIVE"
    if reject_leading_fragment and is_leading_fragment(text):
        return "AVAILABLE_LEADING_FRAGMENT"
    if reject_incomplete_tail and has_incomplete_descriptor_tail(text):
        return "TRUNCATED_DESCRIPTOR"
    if _MID_SENTENCE_FRAGMENT.match(text):
        return "MID_SENTENCE_TRUNCATED"
    return None


class CEFRParser:
    @staticmethod
    def _headers(text: str) -> list[re.Match[str]]:
        return list(_ROW_HEADER.finditer(text))

    def _record(self, *, level: str, scale: str, value: str, source_chunk_ids: list[int], reconstructed_from_fragments: bool = False) -> ParsedCEFRDescriptor:
        reference_level = classify_no_descriptor(value)
        status = "NO_DESCRIPTOR_AVAILABLE" if reference_level is not None else "AVAILABLE"
        canonical = None if status == "NO_DESCRIPTOR_AVAILABLE" else value
        normalized = "" if canonical is None else value.casefold()
        return ParsedCEFRDescriptor(
            level,
            scale,
            canonical,
            normalized,
            descriptor_hash(level, scale, normalized, status, reference_level or None),
            status,
            source_chunk_ids,
            reference_level or None,
            reconstructed_from_fragments,
        )

    def parse(self, text: str, *, source_chunk_ids: list[int], default_scale: str | None = None) -> CEFRParseResult:
        result = CEFRParseResult()
        headers = self._headers(text)
        rows = []
        for position, match in enumerate(headers):
            next_start = headers[position + 1].start() if position + 1 < len(headers) else len(text)
            rows.append((normalize_level(match.group("level")), normalize_text(match.group("scale")), normalize_text(text[match.end():next_start])))

        position = 0
        while position < len(rows):
            level, scale, value = rows[position]
            result.candidate_rows_detected += 1
            reconstructed = False
            # Repeated same-level/same-scale headers can be table-cell
            # fragments.  Reconstruct only a clean opening with a provably
            # unfinished tail; never infer text missing before the fragment.
            if not is_leading_fragment(value):
                while position + 1 < len(rows):
                    next_level, next_scale, next_value = rows[position + 1]
                    if (next_level, next_scale) != (level, scale) or not is_leading_fragment(next_value):
                        break
                    if not has_incomplete_descriptor_tail(value):
                        break
                    value = normalize_text(f"{value} {next_value}")
                    reconstructed = True
                    position += 1
            rejection = validate_scale_name(scale)
            if rejection:
                result.rejected.append(RejectedCEFRCandidate(rejection, source_chunk_ids, level, scale, value[:240]))
            elif not value:
                result.rejected.append(RejectedCEFRCandidate("TRUNCATED_DESCRIPTOR", source_chunk_ids, level, scale))
            elif not source_chunk_ids:
                result.rejected.append(RejectedCEFRCandidate("MISSING_PROVENANCE", source_chunk_ids, level, scale, value[:240]))
            elif classify_no_descriptor(value) is not None:
                result.records.append(self._record(level=level, scale=scale, value=value, source_chunk_ids=source_chunk_ids, reconstructed_from_fragments=reconstructed))
            elif failure := validate_available_descriptor(value, source_chunk_ids=source_chunk_ids):
                result.rejected.append(RejectedCEFRCandidate(failure, source_chunk_ids, level, scale, value[:240]))
            else:
                result.records.append(self._record(level=level, scale=scale, value=value, source_chunk_ids=source_chunk_ids, reconstructed_from_fragments=reconstructed))
            position += 1

        if not result.records and not result.rejected and default_scale:
            normalized_scale = normalize_text(default_scale)
            scale_rejection = validate_scale_name(normalized_scale)
            for match in _TABLE_ROW.finditer(text):
                result.candidate_rows_detected += 1
                level, value = normalize_level(match.group("level")), normalize_text(match.group("text"))
                if scale_rejection:
                    result.rejected.append(RejectedCEFRCandidate(scale_rejection, source_chunk_ids, level, normalized_scale, value[:240]))
                elif not value:
                    result.rejected.append(RejectedCEFRCandidate("TRUNCATED_DESCRIPTOR", source_chunk_ids, level, normalized_scale))
                elif not source_chunk_ids:
                    result.rejected.append(RejectedCEFRCandidate("MISSING_PROVENANCE", source_chunk_ids, level, normalized_scale, value[:240]))
                elif classify_no_descriptor(value) is not None:
                    result.records.append(self._record(level=level, scale=normalized_scale, value=value, source_chunk_ids=source_chunk_ids))
                elif failure := validate_available_descriptor(value, source_chunk_ids=source_chunk_ids):
                    result.rejected.append(RejectedCEFRCandidate(failure, source_chunk_ids, level, normalized_scale, value[:240]))
                else:
                    result.records.append(self._record(level=level, scale=normalized_scale, value=value, source_chunk_ids=source_chunk_ids))

        if not result.records and not result.rejected and default_scale and _MID_SENTENCE_FRAGMENT.match(normalize_text(text)):
            result.candidate_rows_detected += 1
            result.rejected.append(RejectedCEFRCandidate("MID_SENTENCE_TRUNCATED", source_chunk_ids, text_preview=normalize_text(text)[:240]))
        elif not result.records and not result.rejected and _CANDIDATE_SIGNAL.search(text):
            # Do not count every ordinary document chunk as a CEFR candidate.
            # Only a level-shaped structure belongs in candidate outcomes.
            result.candidate_rows_detected += 1
            result.rejected.append(RejectedCEFRCandidate("UNSUPPORTED_STRUCTURE", source_chunk_ids, text_preview=normalize_text(text)[:240]))
        return result

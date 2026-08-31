"""Deterministic multilingual lexical sparse encoding for KnowledgeChunks.

The encoder keeps Arabic/French Unicode tokens intact and uses stable hashed
term IDs.  Qdrant applies server-side IDF to the stored term-frequency values;
this is a compact BM25-compatible lexical foundation without English stemming.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass


_TOKEN_RE = re.compile(r"[^\W_]+(?:['’\-][^\W_]+)*", flags=re.UNICODE)


@dataclass(frozen=True)
class SparseVector:
    indices: list[int]
    values: list[float]


class MultilingualSparseEncoder:
    """Unicode-preserving lexical encoder. Canonical text is never rewritten."""

    @staticmethod
    def indexed_text(*, heading_context: list[str], content: str) -> str:
        headings = "\n".join(item.strip() for item in heading_context if item and item.strip())
        body = (content or "").strip()
        return "\n".join(part for part in (headings, body) if part)

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return [token.casefold() for token in _TOKEN_RE.findall(text or "")]

    @staticmethod
    def _term_id(token: str) -> int:
        # 31-bit positive IDs are stable across processes and fit Qdrant sparse indices.
        return int.from_bytes(hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest(), "big") & 0x7FFFFFFF

    def encode(self, text: str) -> SparseVector | None:
        counts = Counter(self.tokenize(text))
        if not counts:
            return None
        combined: dict[int, float] = {}
        for token, count in counts.items():
            term_id = self._term_id(token)
            # Sub-linear TF keeps long worksheets from overwhelming exact terms.
            combined[term_id] = combined.get(term_id, 0.0) + (1.0 + math.log(count))
        indices = sorted(combined)
        return SparseVector(indices=indices, values=[combined[index] for index in indices])

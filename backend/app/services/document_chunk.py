"""Typed, retrieval-ready-but-not-indexed document chunk representation."""

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class DocumentChunk:
    id: str
    document_id: int
    chunk_index: int
    text_original: str
    text_for_embedding: str
    page_start: int | None
    page_end: int | None
    section: str | None
    headings: list[str]
    content_type: str
    metadata: dict[str, Any]
    token_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

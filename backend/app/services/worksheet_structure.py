"""Geometry-only reconstruction for OCR worksheet pages.

This deliberately orders *blocks*, never words inside a recognized block.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class WorksheetBlock:
    text: str
    label: str
    bbox: dict[str, float]


@dataclass(frozen=True)
class WorksheetSection:
    number: str | None
    text: str
    has_image: bool
    requires_vision: bool = False
    structural_quality: str = "structured"


class WorksheetStructureBuilder:
    _NUMBER = re.compile(r"(?:^|\s)([0-9٠-٩]{1,2})(?:[.)]|\s|$)")
    _VISION_HINT = re.compile(r"(?:الرسم|الصورة|شكل)")

    @staticmethod
    def _bbox(document: Any, item: Any) -> dict[str, float] | None:
        prov = next(iter(getattr(item, "prov", []) or []), None)
        bbox = getattr(prov, "bbox", None)
        if bbox is None:
            return None
        page = (getattr(document, "pages", {}) or {}).get(getattr(prov, "page_no", None))
        height = getattr(getattr(page, "size", None), "height", None)
        if height is not None and hasattr(bbox, "to_top_left_origin"):
            bbox = bbox.to_top_left_origin(page_height=float(height))
        l, t, r, b = (float(getattr(bbox, key)) for key in ("l", "t", "r", "b"))
        return {"l": min(l, r), "r": max(l, r), "t": min(t, b), "b": max(t, b)}

    @staticmethod
    def _arabic_dominant(values: list[WorksheetBlock]) -> bool:
        text = " ".join(value.text for value in values)
        return sum("\u0600" <= char <= "\u08ff" for char in text) > sum(char.isascii() and char.isalpha() for char in text)

    def blocks(self, document: Any, page_number: int) -> list[WorksheetBlock]:
        values: list[WorksheetBlock] = []
        page = (getattr(document, "pages", {}) or {}).get(page_number)
        cells = next((list(getattr(page, name, []) or []) for name in ("word_cells", "ocr_cells", "textline_cells", "cells") if getattr(page, name, None)), [])
        # Word/cell geometry is the only safe basis for repairing a TextItem whose
        # internal OCR order is wrong.  Reconstruct visual lines, never characters.
        if cells:
            for cell in cells:
                text = str(getattr(cell, "text", "") or "").strip()
                bbox = getattr(cell, "bbox", getattr(cell, "rect", None))
                if not text or bbox is None:
                    continue
                try:
                    l, t, r, b = (float(getattr(bbox, key)) for key in ("l", "t", "r", "b"))
                except (TypeError, ValueError, AttributeError):
                    continue
                values.append(WorksheetBlock(text, "ocr_cell", {"l": min(l, r), "r": max(l, r), "t": min(t, b), "b": max(t, b)}))
            if values:
                return self._line_blocks(values)
        for item in getattr(document, "texts", []) or []:
            prov = next(iter(getattr(item, "prov", []) or []), None)
            text = str(getattr(item, "text", "") or "").strip()
            bbox = self._bbox(document, item)
            if text and bbox and getattr(prov, "page_no", None) == page_number:
                values.append(WorksheetBlock(text, str(getattr(item, "label", "text")), bbox))
        return self._line_blocks(values)

    def _line_blocks(self, values: list[WorksheetBlock]) -> list[WorksheetBlock]:
        lines: list[list[WorksheetBlock]] = []
        for value in sorted(values, key=lambda block: (block.bbox["t"], block.bbox["l"])):
            if not lines or abs(lines[-1][0].bbox["t"] - value.bbox["t"]) > 18:
                lines.append([value])
            else:
                lines[-1].append(value)
        ordered: list[WorksheetBlock] = []
        for line in lines:
            rtl = self._arabic_dominant(line)
            line = sorted(line, key=lambda block: block.bbox["r"] if rtl else block.bbox["l"], reverse=rtl)
            if all(block.label == "ocr_cell" for block in line):
                ordered.append(WorksheetBlock(
                    " ".join(block.text for block in line), "text",
                    {"l": min(block.bbox["l"] for block in line), "r": max(block.bbox["r"] for block in line),
                     "t": min(block.bbox["t"] for block in line), "b": max(block.bbox["b"] for block in line)},
                ))
            else:
                ordered.extend(line)
        return ordered

    def sections(self, document: Any, page_number: int, *, has_image: bool) -> list[WorksheetSection]:
        blocks = self.blocks(document, page_number)
        page = (getattr(document, "pages", {}) or {}).get(page_number)
        fine_geometry = any(getattr(page, name, None) for name in ("word_cells", "ocr_cells", "textline_cells", "cells"))
        if len(blocks) < 3:
            return []
        starts = [0]
        for index, block in enumerate(blocks[1:], 1):
            if block.label in {"section_header", "title"} or self._NUMBER.search(block.text):
                starts.append(index)
        starts = sorted(set(starts))
        sections: list[WorksheetSection] = []
        for position, start in enumerate(starts):
            group = blocks[start:starts[position + 1] if position + 1 < len(starts) else len(blocks)]
            if not group:
                continue
            number = next((match.group(1) for block in group[:2] if (match := self._NUMBER.search(block.text))), None)
            quality = "structured" if fine_geometry else "partially_structured"
            requires_vision = has_image and bool(self._VISION_HINT.search(" ".join(block.text for block in group)))
            # Preserve a genuine two-column activity as two spatial groups.  The
            # labels deliberately describe position, never an invented answer pair.
            content = group[1:] if (group[0].label in {"section_header", "title"} or number is not None) else group
            centers = sorted((block.bbox["l"] + block.bbox["r"]) / 2 for block in content)
            gaps = [(centers[index + 1] - value, index) for index, value in enumerate(centers[:-1])]
            largest_gap, split_at = max(gaps, default=(0.0, -1))
            if (group[0].label in {"section_header", "title"} or number is not None) and largest_gap >= 90 and split_at >= 1 and len(centers) - split_at - 1 >= 2:
                divider = (centers[split_at] + centers[split_at + 1]) / 2
                left = [block for block in content if (block.bbox["l"] + block.bbox["r"]) / 2 < divider]
                right = [block for block in content if block not in left]
                rendered = [group[0].text, "العمود الأيمن:\n" + "\n".join(f"- {block.text}" for block in right), "العمود الأيسر:\n" + "\n".join(f"- {block.text}" for block in left)]
                prefix = f"تمرين {number}\n" if number else ""
                sections.append(WorksheetSection(number, prefix + "\n".join(rendered), has_image, requires_vision, quality))
                continue
            # Preserve same-line choices as an explicit list.
            lines: dict[int, list[WorksheetBlock]] = {}
            for block in group:
                lines.setdefault(round(block.bbox["t"] / 18), []).append(block)
            rendered: list[str] = []
            for line in lines.values():
                if len(line) > 1:
                    rendered.append("الاختيارات:\n" + "\n".join(f"- {block.text}" for block in line))
                else:
                    rendered.append(line[0].text)
            prefix = f"تمرين {number}\n" if number else ""
            sections.append(WorksheetSection(number, prefix + "\n".join(rendered), has_image, requires_vision, quality))
        return sections

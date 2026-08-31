"""Conservative Unicode-safe cleanup for extracted document content."""

import re


class DocumentCleaner:
    _control_characters = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
    _horizontal_whitespace = re.compile(r"[\t\f\v ]+")
    _blank_lines = re.compile(r"\n[ \t]*\n[ \t]*\n+")

    def clean(self, text: str) -> str:
        """Remove extraction noise without rewriting language-specific content."""
        cleaned = self._control_characters.sub("", text or "")
        cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
        cleaned = "\n".join(self._horizontal_whitespace.sub(" ", line).strip() for line in cleaned.split("\n"))
        return self._blank_lines.sub("\n\n", cleaned).strip()

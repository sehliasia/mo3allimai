"""Lazy, process-cached Qwen3 cross-encoder provider."""

from __future__ import annotations

from threading import Lock
from typing import Any, Callable

from app.services.reranker_service import RerankScore


class Qwen3RerankerProvider:
    _models: dict[tuple[str, str, str], Any] = {}
    _lock = Lock()

    def __init__(
        self,
        *,
        model_id: str = "Qwen/Qwen3-Reranker-0.6B",
        device: str = "auto",
        batch_size: int = 8,
        instruction: str = (
            "Given a pedagogical query, rank passages by how directly and completely "
            "they answer the query. Prefer passages containing explicit requested "
            "information such as learning objectives, lesson goals, competencies, "
            "instructions, assessment criteria, or pedagogical content over passages "
            "that are only topically related."
        ),
        loader: Callable[[str, str, str], Any] | None = None,
    ) -> None:
        self.model_id, self.device, self.batch_size, self.instruction = model_id, device, batch_size, instruction
        self._loader = loader or self._default_loader

    def _resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    @staticmethod
    def _default_loader(model_id: str, device: str, instruction: str) -> Any:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError("sentence-transformers is required for reranking.") from exc
        # Qwen's supplied default prompt is web-search oriented.  Override it at
        # construction so CrossEncoder applies this task-specific instruction to
        # every predict() call through the model chat template.
        return CrossEncoder(
            model_id,
            device=device,
            prompts={"pedagogical": instruction},
            default_prompt_name="pedagogical",
        )

    def _model(self) -> Any:
        key = (self.model_id, self._resolved_device(), self.instruction)
        with self._lock:
            if key not in self._models:
                self._models[key] = self._loader(*key)
            return self._models[key]

    def rerank(self, query: str, documents: list[str], top_k: int) -> list[RerankScore]:
        if not documents or top_k < 1:
            return []
        scores = self._model().predict(
            [(query, document) for document in documents],
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        if len(scores) != len(documents):
            raise RuntimeError("Reranker returned an unexpected number of scores.")
        ranked = sorted(enumerate(scores), key=lambda item: (-float(item[1]), item[0]))
        return [RerankScore(index=index, score=float(score)) for index, score in ranked[:top_k]]

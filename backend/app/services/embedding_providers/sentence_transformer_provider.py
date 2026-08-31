"""Shared lazy SentenceTransformer adapter.

The adapter deliberately has no import-time model or network side effect. A
model is loaded only when an embedding method is called, and then cached per
model/device pair for the lifetime of the process.
"""

from __future__ import annotations

from collections.abc import Callable
import logging
from threading import Lock
from time import perf_counter
from typing import Any


logger = logging.getLogger(__name__)


class EmbeddingProviderError(RuntimeError):
    """Base error for a real embedding provider."""


class EmbeddingModelLoadError(EmbeddingProviderError):
    """The selected local/Hugging Face model could not be initialized."""


class EmbeddingDimensionError(EmbeddingProviderError):
    """The model did not return the configured vector dimension."""


ModelLoader = Callable[[str, str], Any]


class SentenceTransformerEmbeddingProvider:
    """Dense-only provider with explicit document/query paths.

    `truncate_dim` is delegated to SentenceTransformers and is therefore only
    used when a model supports Matryoshka output dimensions. No vector is
    manually truncated by this application.
    """

    _models: dict[tuple[str, str], Any] = {}
    _models_lock = Lock()
    # One Qwen model is shared per process. Concurrent CPU inference competes
    # for the same cores and can make every teacher document look stalled.
    # Queue whole encode operations instead of loading duplicate models or
    # oversubscribing the CPU.
    _inference_lock = Lock()

    def __init__(
        self,
        *,
        model_id: str,
        dimension: int,
        device: str = "auto",
        batch_size: int = 32,
        query_instruction: str = "",
        max_seq_length: int | None = None,
        cpu_threads: int | None = None,
        model_loader: ModelLoader | None = None,
    ) -> None:
        if dimension <= 0:
            raise ValueError("Embedding dimension must be positive.")
        if device not in {"auto", "cpu", "cuda"}:
            raise ValueError("Embedding device must be auto, cpu, or cuda.")
        self.model_id = model_id
        self.dimension = dimension
        self.device = device
        self.batch_size = batch_size
        self.query_instruction = query_instruction.strip()
        self.max_seq_length = max_seq_length
        self.cpu_threads = cpu_threads
        self._model_loader = model_loader or self._default_model_loader
        self._model: Any | None = None
        self._last_diagnostics: dict[str, object] = {}

    def _resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - dependency error path
            raise EmbeddingModelLoadError("PyTorch is required for real embeddings.") from exc
        return "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def _default_model_loader(model_id: str, device: str) -> Any:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - only when dependency is absent
            raise EmbeddingModelLoadError(
                "sentence-transformers is required for the configured embedding provider."
            ) from exc
        try:
            # Ingestion runs in a FastAPI background task, where an on-demand
            # Hugging Face request can be blocked by deployment networking and
            # hide the real failure behind a generic ingestion error.  Models
            # are provisioned ahead of time; use the local Hugging Face cache
            # exclusively at runtime.
            kwargs: dict[str, Any] = {
                "device": device,
                "trust_remote_code": True,
                "local_files_only": True,
            }
            if device == "cuda":
                # Let the installed, model-compatible Transformers backend pick
                # a supported accelerated dtype; CPU remains full precision.
                kwargs["model_kwargs"] = {"torch_dtype": "auto"}
            return SentenceTransformer(model_id, **kwargs)
        except Exception as exc:  # preserve a provider-level, actionable failure
            raise EmbeddingModelLoadError(f"Could not load embedding model {model_id!r}.") from exc

    def _get_model(self) -> Any:
        if self._model is not None:
            self._last_diagnostics["embedding_model_cache_hit"] = True
            self._last_diagnostics["embedding_model_load_ms"] = 0
            return self._model
        cache_key = (self.model_id, self._resolved_device())
        load_started = perf_counter()
        with self._models_lock:
            model = self._models.get(cache_key)
            cache_hit = model is not None
            if model is None:
                logger.info("[EMBEDDING] model_load_start model=%s device=%s", self.model_id, cache_key[1])
                if cache_key[1] == "cpu" and self.cpu_threads:
                    import torch

                    # CPU-only PyTorch defaults can differ between Windows
                    # processes. Keep the worker bounded and predictable.
                    torch.set_num_threads(self.cpu_threads)
                model = self._model_loader(*cache_key)
                if self.max_seq_length is not None:
                    current_limit = getattr(model, "max_seq_length", self.max_seq_length)
                    model.max_seq_length = min(current_limit, self.max_seq_length)
                self._models[cache_key] = model
                logger.info(
                    "[EMBEDDING] model_load_completed model=%s device=%s max_seq_length=%s duration_seconds=%.3f",
                    self.model_id,
                    cache_key[1],
                    getattr(model, "max_seq_length", None),
                    perf_counter() - load_started,
                )
            else:
                logger.info("[EMBEDDING] model_cache_hit model=%s device=%s", self.model_id, cache_key[1])
        self._model = model
        self._last_diagnostics.update({
            "embedding_model_cache_hit": cache_hit,
            "embedding_model_load_ms": round((perf_counter() - load_started) * 1000),
            "embedding_device": cache_key[1],
        })
        return model

    def _encode(
        self,
        texts: list[str],
        *,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[list[float]]:
        if not texts:
            return []
        model = self._get_model()
        encode_started = perf_counter()
        try:
            import torch

            logger.info("[EMBEDDING] inference_wait model=%s", self.model_id)
            with self._inference_lock, torch.inference_mode():
                logger.info("[EMBEDDING] inference_acquired model=%s", self.model_id)
                # SentenceTransformer batches internally, but does not expose
                # batch completion.  Explicit bounded batches keep CPU/RAM
                # usage predictable and make persistent ingestion progress
                # truthful rather than estimated.
                batches = []
                total_batches = (len(texts) + self.batch_size - 1) // self.batch_size
                logger.info("[EMBEDDING] encode_start chunks=%s batch_size=%s", len(texts), self.batch_size)
                for batch_number, start in enumerate(range(0, len(texts), self.batch_size), start=1):
                    batch = texts[start:start + self.batch_size]
                    batch_started = perf_counter()
                    logger.info("[EMBEDDING] batch_start batch=%s/%s chunks=%s", batch_number, total_batches, len(batch))
                    batches.extend(model.encode(
                        batch,
                        batch_size=len(batch),
                        convert_to_numpy=True,
                        normalize_embeddings=True,
                        show_progress_bar=False,
                        truncate_dim=self.dimension,
                    ))
                    logger.info("[EMBEDDING] batch_completed batch=%s/%s duration_seconds=%.3f", batch_number, total_batches, perf_counter() - batch_started)
                    if progress_callback is not None:
                        progress_callback(min(start + len(batch), len(texts)), len(texts))
                vectors = batches
        except EmbeddingProviderError:
            raise
        except RuntimeError as exc:
            # Includes CUDA out-of-memory. The caller can report/retry without
            # mutating the persistent indexed state.
            raise EmbeddingProviderError("Embedding inference failed for this batch.") from exc
        except Exception as exc:
            raise EmbeddingProviderError("Embedding inference failed for this batch.") from exc

        encode_ms = round((perf_counter() - encode_started) * 1000)
        logger.info("[EMBEDDING] encode_completed chunks=%s duration_seconds=%.3f", len(texts), encode_ms / 1000)
        postprocess_started = perf_counter()
        # Per-batch NumPy rows are converted exactly once at the public
        # boundary, after all inference is finished.
        rows = vectors.tolist() if hasattr(vectors, "tolist") else vectors
        if len(rows) != len(texts) or any(len(row) != self.dimension for row in rows):
            actual = len(rows[0]) if rows else 0
            raise EmbeddingDimensionError(
                f"Model {self.model_id!r} returned dimension {actual}; expected {self.dimension}."
            )
        output = [[float(value) for value in row] for row in rows]
        self._last_diagnostics.update({
            # SentenceTransformer exposes tokenization and inference as one
            # encode call.  Keep that boundary honest instead of tokenizing twice.
            "embedding_tokenization_ms": None,
            "embedding_encode_ms": encode_ms,
            "embedding_inference_ms": encode_ms,
            "embedding_postprocess_ms": round((perf_counter() - postprocess_started) * 1000),
            "embedding_query_count": len(texts),
        })
        return output

    def last_diagnostics(self) -> dict[str, object]:
        """Return timing-only data from the last embedding operation."""
        return dict(self._last_diagnostics)

    def embed_documents(
        self,
        texts: list[str],
        *,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[list[float]]:
        """Embed canonical `content_for_embedding` unchanged."""
        return self._encode(texts, progress_callback=progress_callback)

    def embed_queries(self, queries: list[str]) -> list[list[float]]:
        return self._encode([self.format_query(query) for query in queries])

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Compatibility alias; new callers must choose documents or queries."""
        return self.embed_documents(texts)

    def format_query(self, query: str) -> str:
        return query

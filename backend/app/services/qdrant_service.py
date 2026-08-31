"""Small, safe Qdrant adapter for the canonical KnowledgeChunk vector index."""

from __future__ import annotations

import re
from typing import Any

from app.core.config import Settings, get_settings


class QdrantServiceError(RuntimeError):
    """A collection, compatibility, or confirmed-write operation failed."""


class QdrantCollectionCompatibilityError(QdrantServiceError):
    """An existing collection does not satisfy the immutable vector contract."""


def _version_at_least(value: str, minimum: tuple[int, int, int]) -> bool:
    match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)(?:[-+].*)?", value.strip())
    if match is None:
        raise QdrantCollectionCompatibilityError("Qdrant server returned an invalid version.")
    return tuple(int(part) for part in match.groups()) >= minimum


class QdrantService:
    """Qdrant boundary only: no embedding generation and no ORM state changes."""

    _payload_indexes = {
        "document_id": "INTEGER",
        "language": "KEYWORD",
        "cefr_level": "KEYWORD",
        "content_type": "KEYWORD",
        "structural_quality": "KEYWORD",
        "requires_vision": "BOOL",
        "source_type": "KEYWORD",
        "owner_id": "INTEGER",
    }

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        client: Any | None = None,
        models_module: Any | None = None,
    ) -> None:
        settings = settings or get_settings()
        self.collection_name = settings.qdrant_collection_name
        self.dimension = settings.rag_embedding_dimension
        self.timeout = settings.qdrant_timeout
        self.sparse_vector_name = settings.rag_sparse_vector_name
        self._client = client
        self._models_module = models_module
        self._url = settings.qdrant_url
        self._api_key = settings.qdrant_api_key

    @property
    def client(self) -> Any:
        if self._client is None:
            try:
                from qdrant_client import QdrantClient
            except ImportError as exc:  # pragma: no cover - dependency deployment failure
                raise QdrantServiceError("qdrant-client is required for vector indexing.") from exc
            self._client = QdrantClient(url=self._url, api_key=self._api_key, timeout=self.timeout)
        return self._client

    @property
    def models(self) -> Any:
        if self._models_module is None:
            try:
                from qdrant_client import models
            except ImportError as exc:  # pragma: no cover - dependency deployment failure
                raise QdrantServiceError("qdrant-client is required for vector indexing.") from exc
            self._models_module = models
        return self._models_module

    @staticmethod
    def _unwrap(result: Any) -> Any:
        return getattr(result, "result", result)

    @staticmethod
    def _is_not_found(error: Exception) -> bool:
        return getattr(error, "status_code", None) == 404 or "not found" in str(error).lower()

    @staticmethod
    def _value(value: Any) -> Any:
        return getattr(value, "value", value)

    def ensure_collection(self) -> None:
        """Create once or validate; never recreate an incompatible collection."""
        try:
            info = self._unwrap(self.client.get_collection(self.collection_name))
        except Exception as exc:
            if not self._is_not_found(exc):
                raise QdrantServiceError("Could not inspect the Qdrant collection.") from exc
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=self.models.VectorParams(
                    size=self.dimension,
                    distance=self.models.Distance.COSINE,
                ),
            )
            info = self._unwrap(self.client.get_collection(self.collection_name))

        self._validate_collection_info(info)

        for field, schema_name in self._payload_indexes.items():
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field,
                field_schema=getattr(self.models.PayloadSchemaType, schema_name),
                wait=True,
            )

    def validate_collection(self) -> None:
        """Validate an existing collection for querying without creating one."""
        try:
            info = self._unwrap(self.client.get_collection(self.collection_name))
        except Exception as exc:
            if self._is_not_found(exc):
                raise QdrantServiceError("Qdrant collection does not exist.") from exc
            raise QdrantServiceError("Could not inspect the Qdrant collection.") from exc
        self._validate_collection_info(info)

    def _validate_collection_info(self, info: Any) -> None:
        vectors = info.config.params.vectors
        if isinstance(vectors, dict):
            raise QdrantCollectionCompatibilityError("Named vectors are not supported by this collection contract.")
        actual_size = getattr(vectors, "size", None)
        actual_distance = str(self._value(getattr(vectors, "distance", None))).lower()
        if actual_size != self.dimension or actual_distance != "cosine":
            raise QdrantCollectionCompatibilityError(
                "Qdrant collection is incompatible: "
                f"expected size={self.dimension}, distance=cosine; "
                f"got size={actual_size}, distance={actual_distance}."
            )

    def _collection_info(self) -> Any:
        try:
            return self._unwrap(self.client.get_collection(self.collection_name))
        except Exception as exc:
            raise QdrantServiceError("Could not inspect the Qdrant collection.") from exc

    def server_version(self) -> str:
        """Return the server version from Qdrant itself; never infer it from the client."""
        try:
            info = self._unwrap(self.client.info())
            version = getattr(info, "version", None)
        except Exception as exc:
            raise QdrantServiceError("Could not inspect the Qdrant server version.") from exc
        if not isinstance(version, str) or not version.strip():
            raise QdrantServiceError("Qdrant server did not provide a usable version.")
        return version.strip()

    def sparse_vector_state(self) -> str:
        """Read the real typed collection config: absent, configured, or incompatible."""
        info = self._collection_info()
        self._validate_collection_info(info)
        sparse = getattr(info.config.params, "sparse_vectors", None) or {}
        config = sparse.get(self.sparse_vector_name)
        if config is None:
            return "absent"
        modifier = str(self._value(getattr(config, "modifier", None))).lower()
        return "configured" if modifier == "idf" else "incompatible"

    def sparse_vector_configured(self) -> bool:
        state = self.sparse_vector_state()
        if state == "incompatible":
            raise QdrantCollectionCompatibilityError(
                f"Sparse vector {self.sparse_vector_name!r} exists but is not configured with IDF."
            )
        return state == "configured"

    def ensure_sparse_vector(self) -> None:
        """Explicitly add the named lexical sparse vector after dense validation.

        This is intentionally *not* called from ``ensure_collection`` so the
        deployed dense-only path cannot migrate a collection unexpectedly.
        """
        state = self.sparse_vector_state()
        if state == "configured":
            return
        if state == "incompatible":
            raise QdrantCollectionCompatibilityError(
                f"Sparse vector {self.sparse_vector_name!r} exists with an incompatible configuration."
            )
        try:
            if not _version_at_least(self.server_version(), (1, 18, 0)):
                raise QdrantCollectionCompatibilityError(
                    "Existing Qdrant server does not support adding a sparse vector schema to this collection. "
                    "Collection rebuild/migration is required."
                )
            self.client.create_vector_name(
                collection_name=self.collection_name,
                vector_name=self.sparse_vector_name,
                vector_name_config=self.models.SparseVectorNameConfig(
                    sparse=self.models.SparseVectorConfig(modifier=self.models.Modifier.IDF)
                ),
                wait=True,
            )
        except AttributeError as exc:
            raise QdrantCollectionCompatibilityError(
                "Installed qdrant-client/server does not expose the required sparse-vector API."
            ) from exc
        except QdrantCollectionCompatibilityError:
            raise
        except Exception as exc:
            raise QdrantServiceError("Qdrant did not confirm sparse-vector configuration.") from exc
        if not self.sparse_vector_configured():
            raise QdrantCollectionCompatibilityError("Qdrant did not retain the requested sparse-vector configuration.")

    def update_sparse_vectors(self, points: list[dict[str, Any]]) -> None:
        """Attach a named sparse vector to existing dense points only."""
        if not points:
            return
        structs = [
            self.models.PointVectors(
                id=point["id"],
                vector={self.sparse_vector_name: self.models.SparseVector(indices=point["indices"], values=point["values"])},
            )
            for point in points
        ]
        try:
            self.client.update_vectors(collection_name=self.collection_name, points=structs, wait=True)
        except Exception as exc:
            raise QdrantServiceError("Qdrant did not confirm sparse-vector update.") from exc

    def sparse_vector_present(self, point_id: str) -> bool:
        """Read one point with vectors so repeat sparse indexing is idempotent."""
        try:
            points = self.client.retrieve(
                collection_name=self.collection_name, ids=[point_id], with_payload=False, with_vectors=True,
            )
        except Exception as exc:
            raise QdrantServiceError("Could not inspect an existing Qdrant point.") from exc
        if not points:
            return False
        vectors = getattr(points[0], "vector", None) or {}
        return isinstance(vectors, dict) and self.sparse_vector_name in vectors

    def sparse_point_state(self, point_id: str) -> str:
        """Read-only state for one existing dense point: missing/present/without_sparse."""
        try:
            points = self.client.retrieve(
                collection_name=self.collection_name, ids=[point_id], with_payload=False, with_vectors=True,
            )
        except Exception as exc:
            raise QdrantServiceError("Could not inspect an existing Qdrant point.") from exc
        if not points:
            return "missing"
        vectors = getattr(points[0], "vector", None) or {}
        return "sparse_present" if isinstance(vectors, dict) and self.sparse_vector_name in vectors else "sparse_missing"

    def collection_point_ids(self) -> set[str]:
        """Read all point IDs for a bounded diagnostic; never delete or update points."""
        ids: set[str] = set()
        offset: Any | None = None
        try:
            while True:
                points, offset = self.client.scroll(
                    collection_name=self.collection_name, with_payload=False, with_vectors=False,
                    limit=10_000, offset=offset,
                )
                ids.update(str(point.id) for point in points)
                if offset is None:
                    return ids
        except Exception as exc:
            raise QdrantServiceError("Could not inspect Qdrant point identities.") from exc

    def point_count(self) -> int:
        try:
            count = self.client.count(collection_name=self.collection_name, exact=True)
            return int(getattr(self._unwrap(count), "count", self._unwrap(count)))
        except Exception as exc:
            raise QdrantServiceError("Could not count Qdrant points.") from exc

    def upsert_points(self, points: list[dict[str, Any]]) -> None:
        if not points:
            return
        point_structs = [
            self.models.PointStruct(id=point["id"], vector=point["vector"], payload=point["payload"])
            for point in points
        ]
        try:
            self.client.upsert(collection_name=self.collection_name, points=point_structs, wait=True)
        except Exception as exc:
            raise QdrantServiceError("Qdrant did not confirm the vector upsert.") from exc

    def delete_points(self, point_ids: list[str]) -> int:
        if not point_ids:
            return 0
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=self.models.PointIdsList(points=point_ids),
                wait=True,
            )
        except Exception as exc:
            raise QdrantServiceError("Qdrant did not confirm deletion of stale vectors.") from exc
        return len(point_ids)

    def document_point_ids(self, document_id: int) -> set[str]:
        """Read only IDs for one document; never scan or wipe the whole collection."""
        return {str(point.id) for point in self.document_points(document_id, with_vectors=False)}

    def document_points(self, document_id: int, *, with_vectors: bool = False) -> list[Any]:
        """Read points for one document, including payload for a targeted audit."""
        try:
            points, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=self.models.Filter(
                    must=[
                        self.models.FieldCondition(
                            key="document_id",
                            match=self.models.MatchValue(value=document_id),
                        )
                    ]
                ),
                with_payload=True,
                with_vectors=with_vectors,
                limit=10_000,
            )
        except Exception as exc:
            raise QdrantServiceError("Could not inspect document vectors in Qdrant.") from exc
        return list(points)

    def retrieve_points(self, point_ids: list[str]) -> list[Any]:
        if not point_ids:
            return []
        return self.client.retrieve(
            collection_name=self.collection_name,
            ids=point_ids,
            with_payload=True,
            with_vectors=False,
        )

    def search_points(
        self,
        query_vector: list[float],
        *,
        top_k: int,
        document_ids: list[int] | None = None,
        language: str | None = None,
        cefr_level: str | None = None,
        content_type: str | None = None,
        requires_vision: bool | None = None,
        owner_id: int | None = None,
        source_type: str | None = None,
    ) -> list[Any]:
        """Return Qdrant-ranked points only; canonical text is hydrated elsewhere."""
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")
        if len(query_vector) != self.dimension:
            raise ValueError(
                f"Query vector dimension {len(query_vector)} does not match {self.dimension}."
            )
        self.validate_collection()
        must: list[Any] = []
        if document_ids:
            match = (
                self.models.MatchValue(value=document_ids[0])
                if len(document_ids) == 1
                else self.models.MatchAny(any=document_ids)
            )
            must.append(self.models.FieldCondition(key="document_id", match=match))
        for key, value in (
            ("language", language),
            ("cefr_level", cefr_level),
            ("content_type", content_type),
            ("requires_vision", requires_vision),
            ("owner_id", owner_id),
            ("source_type", source_type),
        ):
            if value is not None:
                must.append(self.models.FieldCondition(key=key, match=self.models.MatchValue(value=value)))
        query_filter = self.models.Filter(must=must) if must else None
        try:
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                query_filter=query_filter,
                limit=top_k,
                with_payload=True,
                with_vectors=False,
            )
        except Exception as exc:
            raise QdrantServiceError("Qdrant similarity search failed.") from exc
        return list(getattr(self._unwrap(response), "points", self._unwrap(response)))

    def search_sparse_points(self, indices: list[int], values: list[float], *, top_k: int, document_ids: list[int] | None = None, language: str | None = None, cefr_level: str | None = None, content_type: str | None = None, requires_vision: bool | None = None, owner_id: int | None = None, source_type: str | None = None) -> list[Any]:
        """Query only the H2 named sparse vector; dense configuration is untouched."""
        if top_k < 1 or not indices or len(indices) != len(values):
            raise ValueError("Sparse query requires non-empty matching indices and values.")
        if not self.sparse_vector_configured():
            raise QdrantCollectionCompatibilityError("Configured sparse vector is unavailable.")
        must: list[Any] = []
        if document_ids:
            match = self.models.MatchValue(value=document_ids[0]) if len(document_ids) == 1 else self.models.MatchAny(any=document_ids)
            must.append(self.models.FieldCondition(key="document_id", match=match))
        for key, value in (("language", language), ("cefr_level", cefr_level), ("content_type", content_type), ("requires_vision", requires_vision), ("owner_id", owner_id), ("source_type", source_type)):
            if value is not None:
                must.append(self.models.FieldCondition(key=key, match=self.models.MatchValue(value=value)))
        try:
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=self.models.SparseVector(indices=indices, values=values), using=self.sparse_vector_name,
                query_filter=self.models.Filter(must=must) if must else None, limit=top_k,
                with_payload=True, with_vectors=False,
            )
        except Exception as exc:
            raise QdrantServiceError("Qdrant sparse similarity search failed.") from exc
        return list(getattr(self._unwrap(response), "points", self._unwrap(response)))

"""Embedding-provider contracts used by the ingestion pipeline."""

import hashlib
from pathlib import Path
from typing import Protocol, Sequence

from sentence_transformers import SentenceTransformer


class EmbeddingProvider(Protocol):
    """The small interface ingestion needs from any embedding model."""

    model_name: str
    dimensions: int
    normalized: bool

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        ...


class SentenceTransformerProvider:
    """Local wrapper around a sentence-transformers model."""

    normalized = True

    def __init__(
        self,
        model_name: str,
        model_dir: Path,
        dimensions: int,
    ) -> None:
        self.model_name = model_name
        self.dimensions = dimensions
        self._model = SentenceTransformer(
            model_name,
            cache_folder=str(model_dir),
            local_files_only=True,
        )

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self._model.encode(list(texts), normalize_embeddings=True)
        result = [vector.tolist() for vector in vectors]
        for vector in result:
            if len(vector) != self.dimensions:
                raise ValueError(
                    f"embedding provider returned {len(vector)} dimensions; "
                    f"expected {self.dimensions}"
                )
        return result


class DeterministicEmbeddingProvider:
    """Small deterministic provider for unit tests and local experiments."""

    normalized = False

    def __init__(self, dimensions: int = 8) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self.dimensions = dimensions
        self.model_name = "deterministic-test-embedder"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            vector = [float(digest[index] + 1) for index in range(self.dimensions)]
            vectors.append(vector)
        return vectors

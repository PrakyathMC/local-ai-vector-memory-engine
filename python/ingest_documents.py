"""Ingest Markdown or plain-text files into the local memory engine."""

import argparse
import hashlib
import time
from pathlib import Path

from embedding import EmbeddingProvider, SentenceTransformerProvider
from local_ai_memory import Memory, MemoryStore


MODEL_NAME = "BAAI/bge-small-en-v1.5"
MODEL_DIR = Path(__file__).parents[1] / "models"
MODEL_DIMENSIONS = 384


def chunk_text(text: str, chunk_size: int = 200, overlap: int = 40) -> list[tuple[str, int, int]]:
    """Split text deterministically into overlapping word chunks.

    Returns ``(chunk_text, start_word, end_word)`` tuples. The end position is
    exclusive, which makes the boundaries unambiguous for later evaluation.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    words = text.split()
    chunks: list[tuple[str, int, int]] = []
    step = chunk_size - overlap
    for start in range(0, len(words), step):
        end = min(start + chunk_size, len(words))
        if start >= end:
            break
        chunks.append((" ".join(words[start:end]), start, end))
        if end == len(words):
            break
    return chunks


def _source_key(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]


def ingest_text(
    text: str,
    source: str,
    store: MemoryStore,
    provider: EmbeddingProvider,
    namespace: str = "default",
    chunk_size: int = 200,
    overlap: int = 40,
) -> int:
    """Embed and upsert all chunks from one source. Returns chunk count."""
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    source_key = _source_key(source)
    now_ms = int(time.time() * 1000)
    expected_ids: set[str] = set()

    if not chunks:
        return 0

    vectors = provider.embed([chunk for chunk, _, _ in chunks])
    if len(vectors) != len(chunks):
        raise ValueError("embedding provider returned the wrong number of vectors")

    for index, ((content, start, end), vector) in enumerate(zip(chunks, vectors)):
        memory_id = f"{source_key}-chunk-{index:04d}"
        expected_ids.add(memory_id)
        memory = Memory(
            memory_id,
            content,
            vector,
            namespace=namespace,
            model=provider.model_name,
            source=source,
            metadata={
                "chunk_index": str(index),
                "start_word": str(start),
                "end_word": str(end),
            },
            created_at_ms=now_ms,
            updated_at_ms=now_ms,
            normalized=provider.normalized,
        )
        if store.get(memory_id) is None:
            store.insert(memory)
        else:
            store.update(memory)

    # Remove stale chunks when a source becomes shorter after editing.
    for existing in store.list(namespace):
        if existing.source == source and existing.id.startswith(f"{source_key}-chunk-"):
            if existing.id not in expected_ids:
                store.delete(existing.id)

    return len(chunks)


def ingest_file(
    path: Path,
    database: str,
    namespace: str = "default",
    chunk_size: int = 200,
    overlap: int = 40,
) -> int:
    """Load, embed, and store one Markdown or text file."""
    if path.suffix.lower() not in {".md", ".markdown", ".txt"}:
        raise ValueError("only Markdown and plain-text files are supported")

    source = str(path.resolve())
    text = path.read_text(encoding="utf-8")
    provider = SentenceTransformerProvider(
        MODEL_NAME,
        MODEL_DIR,
        MODEL_DIMENSIONS,
    )
    store = MemoryStore(database)
    return ingest_text(text, source, store, provider, namespace, chunk_size, overlap)


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed and ingest a Markdown/text file.")
    parser.add_argument("file", type=Path)
    parser.add_argument("--db", default="data/ingested-memory.db")
    parser.add_argument("--namespace", default="default")
    parser.add_argument("--chunk-size", type=int, default=200)
    parser.add_argument("--overlap", type=int, default=40)
    args = parser.parse_args()

    count = ingest_file(
        args.file,
        args.db,
        args.namespace,
        args.chunk_size,
        args.overlap,
    )
    print(f"Ingested {count} chunks from {args.file}")
    print(f"Database: {args.db}")
    print(f"Embedding model: {MODEL_NAME}")


if __name__ == "__main__":
    main()

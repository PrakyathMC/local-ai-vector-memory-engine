from pathlib import Path
import sys
import gc

from local_ai_memory import Memory, MemoryStore

sys.path.insert(0, str(Path(__file__).parents[1]))
from embedding import DeterministicEmbeddingProvider
from ingest_documents import chunk_text, ingest_text


def test_python_can_create_rust_backed_memory():
    memory = Memory(
        "memory-1",
        "Rust handles retrieval",
        [0.1, 0.2, 0.3],
    )

    assert memory.id == "memory-1"
    assert memory.namespace == "default"
    assert memory.dimensions == 3
    assert memory.embedding_model == "test-embedder"
    memory.validate()


def test_invalid_python_memory_is_rejected_by_rust():
    try:
        Memory("", "Invalid ID", [0.1, 0.2, 0.3])
    except ValueError as error:
        assert "memory ID cannot be empty" in str(error)
    else:
        raise AssertionError("invalid memory should have been rejected")


def test_python_can_persist_and_read_memory():
    database = Path(__file__).parents[2] / "data/python-test-memory.db"
    if database.exists():
        database.unlink()

    try:
        store = MemoryStore(str(database))
        original = Memory(
            "persistent-1",
            "This memory is stored in SQLite.",
            [0.1, 0.2, 0.3],
        )

        store.insert(original)
        del store

        reopened_store = MemoryStore(str(database))
        restored = reopened_store.get("persistent-1")

        assert restored is not None
        assert restored.id == original.id
        assert restored.content == original.content
        assert restored.dimensions == original.dimensions
        del reopened_store
    finally:
        if database.exists():
            database.unlink()


def test_python_can_manage_memory_lifecycle():
    database = Path(__file__).parents[2] / "data/python-lifecycle-test.db"
    if database.exists():
        database.unlink()

    store = None
    try:
        store = MemoryStore(str(database))
        original = Memory("lifecycle-1", "Original content.", [0.1, 0.2, 0.3])
        store.insert(original)

        assert len(store.list("default")) == 1

        updated = Memory("lifecycle-1", "Updated content.", [0.3, 0.2, 0.1])
        store.update(updated)
        assert store.get("lifecycle-1").content == "Updated content."

        store.archive("lifecycle-1")
        assert store.get("lifecycle-1").status == "archived"

        store.delete("lifecycle-1")
        assert store.get("lifecycle-1").status == "deleted"
    finally:
        del store
        gc.collect()
        if database.exists():
            database.unlink()


def test_python_can_search_active_memories_by_similarity():
    database = Path(__file__).parents[2] / "data/python-search-test.db"
    if database.exists():
        database.unlink()

    store = None
    try:
        store = MemoryStore(str(database))
        store.insert(Memory("exact", "Exact match", [1.0, 0.0]))
        store.insert(Memory("different", "Different direction", [0.0, 1.0]))
        store.insert(Memory("archived", "Should not be returned", [1.0, 0.0]))
        store.archive("archived")

        results = store.search([1.0, 0.0], top_k=5)

        assert [result.id for result in results] == ["exact", "different"]
        assert results[0].score > results[1].score
        assert results[0].content == "Exact match"
    finally:
        del store
        gc.collect()
        if database.exists():
            database.unlink()


def test_chunking_is_deterministic_and_ingestion_records_boundaries():
    chunks = chunk_text("one two three four five six", chunk_size=4, overlap=1)
    assert chunks == [
        ("one two three four", 0, 4),
        ("four five six", 3, 6),
    ]

    database = Path(__file__).parents[2] / "data/python-ingestion-test.db"
    if database.exists():
        database.unlink()

    store = None
    try:
        store = MemoryStore(str(database))
        count = ingest_text(
            "one two three four five six",
            "notes/example.md",
            store,
            DeterministicEmbeddingProvider(dimensions=2),
            chunk_size=4,
            overlap=1,
        )
        assert count == 2
        memories = store.list("default")
        assert len(memories) == 2
        assert memories[0].source == "notes/example.md"
    finally:
        del store
        gc.collect()
        if database.exists():
            database.unlink()


def test_reingestion_updates_chunks_and_soft_deletes_stale_chunks():
    database = Path(__file__).parents[2] / "data/python-reingestion-test.db"
    if database.exists():
        database.unlink()

    store = None
    try:
        store = MemoryStore(str(database))
        provider = DeterministicEmbeddingProvider(dimensions=2)
        source = "notes/changing.md"

        ingest_text("one two three four five six", source, store, provider, chunk_size=3, overlap=0)
        first_ids = [memory.id for memory in store.list("default")]
        assert len(first_ids) == 2

        ingest_text("one two three four five six", source, store, provider, chunk_size=3, overlap=0)
        assert len(store.list("default")) == 2

        ingest_text("one two three", source, store, provider, chunk_size=3, overlap=0)
        memories = store.list("default")
        assert len(memories) == 2
        assert sum(memory.status == "deleted" for memory in memories) == 1
    finally:
        del store
        gc.collect()
        if database.exists():
            database.unlink()

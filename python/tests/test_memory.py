from pathlib import Path

from local_ai_memory import Memory, MemoryStore


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
        if database.exists():
            database.unlink()

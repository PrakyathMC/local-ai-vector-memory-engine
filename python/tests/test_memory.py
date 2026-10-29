from local_ai_memory import Memory


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

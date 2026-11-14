import gc
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from context_builder import build_context
from local_ai_memory import Memory, MemoryStore


def make_store(name: str) -> tuple[Path, MemoryStore]:
    database = Path(__file__).parents[2] / f"data/{name}.db"
    if database.exists():
        database.unlink()
    return database, MemoryStore(str(database))


def test_context_contains_rank_score_source_and_only_active_evidence():
    database, store = make_store("python-context-test")
    try:
        store.insert(
            Memory(
                "rust-memory",
                "Rust owns the performance-critical retrieval core.",
                [1.0, 0.0],
                source="docs/architecture.md",
                metadata={"topic": "rust"},
            )
        )
        store.insert(Memory("archived-memory", "Do not include this.", [1.0, 0.0]))
        store.archive("archived-memory")

        package = build_context(
            store,
            "What owns retrieval?",
            [1.0, 0.0],
            top_k=5,
            max_chars=1_000,
        )

        assert package.retrieval["selected_count"] == 1
        assert package.evidence[0]["memory_id"] == "rust-memory"
        assert package.evidence[0]["source"] == "docs/architecture.md"
        assert package.evidence[0]["score"] == 1.0
        assert package.evidence[0]["citation"].startswith("[1]")
        assert "Do not include" not in package.context_text
        assert json.loads(package.to_json())["schema_version"] == "1"
    finally:
        del store
        gc.collect()
        if database.exists():
            database.unlink()


def test_context_is_deterministically_bounded_and_marks_truncation():
    database, store = make_store("python-context-budget-test")
    try:
        store.insert(Memory("long", "abcdefghij" * 20, [1.0, 0.0]))
        package = build_context(
            store,
            "bounded query",
            [1.0, 0.0],
            max_chars=80,
        )

        assert package.retrieval["selected_count"] == 1
        assert package.evidence[0]["truncated"] is True
        assert package.context_text.endswith("…")
        assert package.retrieval["used_chars"] <= 80
    finally:
        del store
        gc.collect()
        if database.exists():
            database.unlink()


def test_context_validates_inputs():
    database, store = make_store("python-context-validation-test")
    try:
        with pytest.raises(ValueError, match="query cannot be empty"):
            build_context(store, " ", [1.0, 0.0])
        with pytest.raises(ValueError, match="mode"):
            build_context(store, "query", [1.0, 0.0], mode="unknown")
    finally:
        del store
        gc.collect()
        if database.exists():
            database.unlink()


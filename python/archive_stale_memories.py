"""Archive memories created from a stale source document.

Archiving is reversible and keeps the old rows available for inspection while
removing them from normal exact and HNSW retrieval.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from local_ai_memory import MemoryStore


def is_matching_source(source: str | None, source_name: str) -> bool:
    if not source:
        return False
    normalized_source = source.replace("\\", "/").lower()
    normalized_name = source_name.replace("\\", "/").lower().lstrip("/")
    return normalized_source == normalized_name or normalized_source.endswith(
        f"/{normalized_name}"
    )


def archive_source(database: str, source_name: str) -> int:
    store = MemoryStore(database)
    candidates = [
        memory
        for memory in store.list(None)
        if is_matching_source(memory.source, source_name)
        and memory.status == "active"
    ]
    for memory in candidates:
        store.archive(memory.id)
    return len(candidates)


def main() -> None:
    parser = argparse.ArgumentParser(description="Archive stale source memories.")
    parser.add_argument("--db", default="data/ingestion-demo.db")
    parser.add_argument("--source", default="README.md")
    args = parser.parse_args()

    count = archive_source(args.db, args.source)
    print(f"Archived {count} active memories matching {args.source!r} in {args.db}")


if __name__ == "__main__":
    main()


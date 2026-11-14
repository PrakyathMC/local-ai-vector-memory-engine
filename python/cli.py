"""Command-line workflows for the local memory engine."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(Path(__file__).parent))

from context_builder import build_context  # noqa: E402
from embedding import SentenceTransformerProvider  # noqa: E402
from ingest_documents import (  # noqa: E402
    MODEL_DIMENSIONS,
    MODEL_DIR,
    MODEL_NAME,
    ingest_file,
)
from local_ai_memory import MemoryStore  # noqa: E402


def provider() -> SentenceTransformerProvider:
    return SentenceTransformerProvider(MODEL_NAME, MODEL_DIR, MODEL_DIMENSIONS)


def database_path(value: str) -> str:
    path = Path(value)
    return str(path if path.is_absolute() else PROJECT_ROOT / path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Local AI memory engine CLI")
    parser.add_argument("--db", default="data/ingestion-demo.db")
    commands = parser.add_subparsers(dest="command", required=True)

    inspect = commands.add_parser("inspect")
    inspect.set_defaults(action="inspect")

    ingest = commands.add_parser("ingest")
    ingest.add_argument("file", type=Path)
    ingest.add_argument("--namespace", default="default")
    ingest.set_defaults(action="ingest")

    search = commands.add_parser("search")
    search.add_argument("query")
    search.add_argument("--top-k", type=int, default=5)
    search.add_argument("--mode", choices=["exact", "hnsw"], default="exact")
    search.set_defaults(action="search")

    context = commands.add_parser("context")
    context.add_argument("query")
    context.add_argument("--top-k", type=int, default=5)
    context.add_argument("--max-chars", type=int, default=6000)
    context.add_argument("--mode", choices=["exact", "hnsw"], default="exact")
    context.set_defaults(action="context")

    feedback = commands.add_parser("feedback")
    feedback.add_argument("memory_id")
    feedback_group = feedback.add_mutually_exclusive_group(required=True)
    feedback_group.add_argument("--useful", action="store_true")
    feedback_group.add_argument("--not-useful", action="store_true")
    feedback.set_defaults(action="feedback")

    args = parser.parse_args()
    db = database_path(args.db)

    if args.action == "inspect":
        store = MemoryStore(db)
        memories = store.list(None)
        print(json.dumps({"database": db, "memories": len(memories), "active": sum(m.status == "active" for m in memories)}, indent=2))
    elif args.action == "ingest":
        count = ingest_file(args.file, db, namespace=args.namespace)
        print(json.dumps({"database": db, "accepted_chunks": count}, indent=2))
    elif args.action in {"search", "context"}:
        model = provider()
        vector = model.embed([args.query])[0]
        store = MemoryStore(db)
        if args.action == "search":
            if args.mode == "hnsw":
                results = store.search_hnsw(vector, top_k=args.top_k, model=model.model_name)
            else:
                results = store.search(vector, top_k=args.top_k, model=model.model_name)
            print(json.dumps({"mode": args.mode, "results": [{"id": r.id, "score": r.score, "content": r.content, "source": r.source} for r in results]}, indent=2))
        else:
            print(build_context(store, args.query, vector, top_k=args.top_k, max_chars=args.max_chars, mode=args.mode, model=model.model_name).to_json())
    elif args.action == "feedback":
        MemoryStore(db).record_feedback(args.memory_id, args.useful, int(time.time() * 1000))
        print(json.dumps({"id": args.memory_id, "status": "recorded"}, indent=2))


if __name__ == "__main__":
    main()


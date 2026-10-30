"""Proof of concept: embed one piece of text and store it in SQLite."""

import argparse
from pathlib import Path

from sentence_transformers import SentenceTransformer

from local_ai_memory import Memory, MemoryStore


MODEL_NAME = "BAAI/bge-small-en-v1.5"
MODEL_DIR = Path(__file__).parents[1] / "models"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a local embedding and store one memory."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", help="Text to remember.")
    source.add_argument("--file", type=Path, help="Text or Markdown file to remember.")
    parser.add_argument("--id", required=True, help="Stable memory ID.")
    parser.add_argument("--db", default="data/demo-memory.db")
    parser.add_argument("--namespace", default="default")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    content = args.text if args.text is not None else args.file.read_text(encoding="utf-8")

    model = SentenceTransformer(
        MODEL_NAME,
        cache_folder=str(MODEL_DIR),
        local_files_only=True,
    )
    vector = model.encode(content, normalize_embeddings=True).tolist()

    memory = Memory(
        args.id,
        content,
        vector,
        namespace=args.namespace,
        model=MODEL_NAME,
    )

    store = MemoryStore(args.db)
    if store.get(args.id) is None:
        store.insert(memory)
        operation = "inserted"
    else:
        store.update(memory)
        operation = "updated"

    print(f"Memory {operation}: {memory.id}")
    print(f"Database: {args.db}")
    print(f"Embedding model: {MODEL_NAME}")
    print(f"Embedding dimensions: {memory.dimensions}")
    print(f"Namespace: {memory.namespace}")


if __name__ == "__main__":
    main()

"""Evaluate exact retrieval against a small hand-labeled fixture."""

import argparse
import json
from pathlib import Path

from embedding import SentenceTransformerProvider
from ingest_documents import MODEL_DIMENSIONS, MODEL_DIR, MODEL_NAME, ingest_text
from local_ai_memory import MemoryStore


PROJECT_ROOT = Path(__file__).parents[1]
FIXTURE_PATH = Path(__file__).with_name("evaluation_fixture.json")


def load_fixture() -> list[dict[str, str]]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def evaluate(
    database: str,
    top_k: int = 3,
) -> tuple[list[dict[str, object]], dict[str, float]]:
    fixture = load_fixture()
    provider = SentenceTransformerProvider(MODEL_NAME, MODEL_DIR, MODEL_DIMENSIONS)
    store = MemoryStore(database)

    for item in fixture:
        ingest_text(
            item["text"],
            item["source"],
            store,
            provider,
            namespace="evaluation",
            chunk_size=80,
            overlap=0,
            memory_type="document",
        )

    results = []
    top_one_hits = 0
    top_k_hits = 0

    for item in fixture:
        query_vector = provider.embed([item["query"]])[0]
        hits = store.search(
            query_vector,
            top_k=top_k,
            namespace="evaluation",
            model=provider.model_name,
        )
        hit_sources = [hit.source for hit in hits]
        top_one = bool(hit_sources) and hit_sources[0] == item["source"]
        top_k_match = item["source"] in hit_sources
        top_one_hits += int(top_one)
        top_k_hits += int(top_k_match)
        results.append(
            {
                "id": item["id"],
                "query": item["query"],
                "expected_source": item["source"],
                "returned_sources": hit_sources,
                "top_score": hits[0].score if hits else None,
                "top1_correct": top_one,
                "top_k_correct": top_k_match,
            }
        )

    total = len(fixture)
    metrics = {
        "top1_accuracy": top_one_hits / total if total else 0.0,
        "topk_accuracy": top_k_hits / total if total else 0.0,
        "cases": float(total),
    }
    return results, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate exact retrieval quality.")
    parser.add_argument("--db", default=str(PROJECT_ROOT / "data" / "retrieval-evaluation.db"))
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    results, metrics = evaluate(args.db, args.top_k)
    print("Retrieval evaluation")
    print(f"Database: {args.db}")
    print(f"Top-1 accuracy: {metrics['top1_accuracy']:.1%}")
    print(f"Top-{args.top_k} accuracy: {metrics['topk_accuracy']:.1%}")
    print()
    for result in results:
        status = "PASS" if result["top_k_correct"] else "FAIL"
        print(
            f"[{status}] {result['id']}: score={result['top_score']!r}, "
            f"returned={result['returned_sources']}"
        )


if __name__ == "__main__":
    main()

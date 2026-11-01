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


def evaluate_hnsw(
    database: str,
    top_k: int = 3,
) -> tuple[list[dict[str, object]], dict[str, float]]:
    """Evaluate the HNSW path against the same labeled fixture."""
    fixture = load_fixture()
    provider = SentenceTransformerProvider(MODEL_NAME, MODEL_DIR, MODEL_DIMENSIONS)
    store = MemoryStore(database)
    results = []
    top_one_hits = 0
    top_k_hits = 0

    for item in fixture:
        query_vector = provider.embed([item["query"]])[0]
        hits = store.search_hnsw(
            query_vector,
            top_k=top_k,
            ef_search=max(64, top_k),
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
    hnsw_results, hnsw_metrics = evaluate_hnsw(args.db, args.top_k)
    exact_top1_ids = [result["returned_sources"][0] for result in results if result["returned_sources"]]
    hnsw_top1_ids = [result["returned_sources"][0] for result in hnsw_results if result["returned_sources"]]
    exact_top1_agreement = sum(
        exact_id == hnsw_id for exact_id, hnsw_id in zip(exact_top1_ids, hnsw_top1_ids)
    ) / len(exact_top1_ids) if exact_top1_ids else 0.0
    top_k_recall_values = []
    for exact_result, hnsw_result in zip(results, hnsw_results):
        exact_sources = set(exact_result["returned_sources"])
        hnsw_sources = set(hnsw_result["returned_sources"])
        if exact_sources:
            top_k_recall_values.append(len(exact_sources & hnsw_sources) / len(exact_sources))
    top_k_recall = sum(top_k_recall_values) / len(top_k_recall_values) if top_k_recall_values else 0.0
    print("Retrieval evaluation")
    print(f"Database: {args.db}")
    print(f"Exact top-1 accuracy: {metrics['top1_accuracy']:.1%}")
    print(f"Exact top-{args.top_k} accuracy: {metrics['topk_accuracy']:.1%}")
    print(f"HNSW top-1 accuracy: {hnsw_metrics['top1_accuracy']:.1%}")
    print(f"HNSW top-{args.top_k} accuracy: {hnsw_metrics['topk_accuracy']:.1%}")
    print(f"HNSW agreement with exact: top-1={exact_top1_agreement:.1%}")
    print(f"HNSW recall of exact top-{args.top_k}: {top_k_recall:.1%}")
    print()
    for exact_result, hnsw_result in zip(results, hnsw_results):
        status = "PASS" if hnsw_result["top_k_correct"] else "FAIL"
        print(
            f"[{status}] {exact_result['id']}: "
            f"exact={exact_result['returned_sources']} | "
            f"hnsw={hnsw_result['returned_sources']}"
        )


if __name__ == "__main__":
    main()

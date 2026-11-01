"""Benchmark exact retrieval against the cached HNSW path on synthetic vectors."""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import numpy as np

from local_ai_memory import Memory, MemoryStore


PROJECT_ROOT = Path(__file__).parents[1]


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((len(ordered) - 1) * fraction)))
    return ordered[index]


def run_benchmark(database: str, count: int, dimensions: int, queries: int, top_k: int) -> None:
    rng = np.random.default_rng(42)
    vectors = rng.normal(size=(count, dimensions)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    query_vectors = vectors[:queries].copy()

    database_path = Path(database)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    if database_path.exists():
        database_path.unlink()

    store = MemoryStore(str(database_path))
    for index, vector in enumerate(vectors):
        store.insert(
            Memory(
                f"benchmark-{index:05d}",
                f"Synthetic benchmark memory {index}",
                vector.tolist(),
                namespace="benchmark",
                model="benchmark-embedder",
            )
        )

    exact_times: list[float] = []
    exact_results = []
    for query in query_vectors:
        started = time.perf_counter()
        hits = store.search(
            query.tolist(),
            top_k=top_k,
            namespace="benchmark",
            model="benchmark-embedder",
        )
        exact_times.append((time.perf_counter() - started) * 1000)
        exact_results.append({hit.id for hit in hits})

    started = time.perf_counter()
    store.search_hnsw(
        query_vectors[0].tolist(),
        top_k=top_k,
        ef_search=max(64, top_k),
        namespace="benchmark",
        model="benchmark-embedder",
    )
    hnsw_build_query_ms = (time.perf_counter() - started) * 1000

    hnsw_times: list[float] = []
    hnsw_results = []
    for query in query_vectors:
        started = time.perf_counter()
        hits = store.search_hnsw(
            query.tolist(),
            top_k=top_k,
            ef_search=max(64, top_k),
            namespace="benchmark",
            model="benchmark-embedder",
        )
        hnsw_times.append((time.perf_counter() - started) * 1000)
        hnsw_results.append({hit.id for hit in hits})

    recall = [
        len(exact_ids & hnsw_ids) / len(exact_ids)
        for exact_ids, hnsw_ids in zip(exact_results, hnsw_results)
        if exact_ids
    ]

    print("Retrieval benchmark")
    print(f"Database: {database}")
    print(f"Memories: {count:,} | Dimensions: {dimensions} | Queries: {queries} | Top-k: {top_k}")
    print(f"Exact p50/p95: {percentile(exact_times, 0.50):.3f} / {percentile(exact_times, 0.95):.3f} ms")
    print(
        f"HNSW build + first query: {hnsw_build_query_ms:.3f} ms | "
        f"warm p50/p95: {percentile(hnsw_times, 0.50):.3f} / "
        f"{percentile(hnsw_times, 0.95):.3f} ms"
    )
    print(f"HNSW recall of exact top-{top_k}: {statistics.mean(recall):.1%}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(PROJECT_ROOT / "data" / "retrieval-benchmark.db"))
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--dimensions", type=int, default=384)
    parser.add_argument("--queries", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()
    run_benchmark(args.db, args.count, args.dimensions, args.queries, args.top_k)


if __name__ == "__main__":
    main()

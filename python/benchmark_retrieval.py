"""Benchmark exact retrieval against the cached HNSW path on synthetic vectors."""

from __future__ import annotations

import argparse
import ctypes
import os
import statistics
import sys
from ctypes import wintypes
import time
from pathlib import Path

import numpy as np

from local_ai_memory import Memory, MemoryStore


PROJECT_ROOT = Path(__file__).parents[1]


def process_rss_bytes() -> int | None:
    """Return current process RSS where the operating system exposes it."""
    if sys.platform == "win32":
        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("page_fault_count", ctypes.c_ulong),
                ("peak_working_set_size", ctypes.c_size_t),
                ("working_set_size", ctypes.c_size_t),
                ("quota_peak_paged_pool_usage", ctypes.c_size_t),
                ("quota_paged_pool_usage", ctypes.c_size_t),
                ("quota_peak_non_paged_pool_usage", ctypes.c_size_t),
                ("quota_non_paged_pool_usage", ctypes.c_size_t),
                ("pagefile_usage", ctypes.c_size_t),
                ("peak_pagefile_usage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        process = ctypes.WinDLL("kernel32.dll").GetCurrentProcess()
        get_memory_info = ctypes.WinDLL("psapi.dll").GetProcessMemoryInfo
        get_memory_info.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(ProcessMemoryCounters),
            wintypes.DWORD,
        ]
        get_memory_info.restype = wintypes.BOOL
        success = get_memory_info(
            process,
            ctypes.byref(counters),
            counters.cb,
        )
        return int(counters.working_set_size) if success else None

    try:
        import resource

        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(value * (1024 if sys.platform != "darwin" else 1))
    except (ImportError, AttributeError):
        return None


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

    sqlite_bytes = database_path.stat().st_size
    vector_bytes = count * dimensions * 4
    rss_before_hnsw = process_rss_bytes()

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
    rss_after_hnsw = process_rss_bytes()
    hnsw_rss_delta_mb = (
        (rss_after_hnsw - rss_before_hnsw) / (1024 * 1024)
        if rss_before_hnsw is not None and rss_after_hnsw is not None
        else None
    )

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
    print(f"SQLite file size: {sqlite_bytes / (1024 * 1024):.3f} MB")
    print(f"Raw vector payload: {vector_bytes / (1024 * 1024):.3f} MB")
    print(f"Exact p50/p95: {percentile(exact_times, 0.50):.3f} / {percentile(exact_times, 0.95):.3f} ms")
    print(
        f"HNSW build + first query: {hnsw_build_query_ms:.3f} ms | "
        f"warm p50/p95: {percentile(hnsw_times, 0.50):.3f} / "
        f"{percentile(hnsw_times, 0.95):.3f} ms"
    )
    print(
        "HNSW process RSS increase during first build: "
        + (f"{hnsw_rss_delta_mb:.3f} MB" if hnsw_rss_delta_mb is not None else "unavailable")
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

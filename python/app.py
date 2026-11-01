"""Lightweight Streamlit inspection UI for the local memory engine."""

import json
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(Path(__file__).parent))

from embedding import SentenceTransformerProvider  # noqa: E402
from ingest_documents import (  # noqa: E402
    MODEL_DIMENSIONS,
    MODEL_DIR,
    MODEL_NAME,
    chunk_text,
    ingest_text,
)
from local_ai_memory import MemoryStore  # noqa: E402


st.set_page_config(
    page_title="Local AI Memory Inspector",
    page_icon="🧠",
    layout="wide",
)


def database_path() -> Path:
    configured = st.sidebar.text_input(
        "SQLite database",
        value=str(PROJECT_ROOT / "data" / "ingestion-demo.db"),
    )
    return Path(configured)


def read_database(path: Path) -> tuple[dict[str, str], list[dict[str, object]]]:
    if not path.exists():
        return {}, []

    with sqlite3.connect(path) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        if "memories" not in tables:
            return {}, []

        metadata = {}
        if "engine_metadata" in tables:
            metadata = dict(connection.execute("SELECT key, value FROM engine_metadata"))

        rows = connection.execute(
            """
            SELECT id, content, source, memory_type, namespace,
                   created_at_ms, updated_at_ms, status, metadata_json,
                   embedding_model, embedding_dimensions, embedding_normalized
            FROM memories
            ORDER BY created_at_ms, id
            """
        ).fetchall()

    memories = []
    for row in rows:
        metadata_json = row[8]
        try:
            chunk_metadata = json.loads(metadata_json)
        except json.JSONDecodeError:
            chunk_metadata = {}
        memories.append(
            {
                "id": row[0],
                "content": row[1],
                "source": row[2],
                "type": row[3],
                "namespace": row[4],
                "created_at_ms": row[5],
                "updated_at_ms": row[6],
                "status": row[7],
                "metadata": chunk_metadata,
                "model": row[9],
                "dimensions": row[10],
                "normalized": bool(row[11]),
            }
        )
    return metadata, memories


@st.cache_resource(show_spinner="Loading the local embedding model...")
def get_provider() -> SentenceTransformerProvider:
    return SentenceTransformerProvider(MODEL_NAME, MODEL_DIR, MODEL_DIMENSIONS)


def load_store(path: Path) -> MemoryStore:
    path.parent.mkdir(parents=True, exist_ok=True)
    return MemoryStore(str(path))


def status_badge(status: str) -> None:
    if status == "active":
        st.success("ACTIVE")
    elif status == "archived":
        st.warning("ARCHIVED")
    else:
        st.error("DELETED")


def dashboard(path: Path, metadata: dict[str, str], memories: list[dict[str, object]]) -> None:
    st.title("🧠 Local AI Memory Inspector")
    st.caption("A small observability UI for the Rust memory engine and Python AI workflow.")

    if not path.exists():
        st.info("This database does not exist yet. Use Ingest & Chunking to create it.")
        return

    active = sum(memory["status"] == "active" for memory in memories)
    archived = sum(memory["status"] == "archived" for memory in memories)
    deleted = sum(memory["status"] == "deleted" for memory in memories)

    columns = st.columns(6)
    columns[0].metric("Total memories", len(memories))
    columns[1].metric("Active", active)
    columns[2].metric("Archived", archived)
    columns[3].metric("Deleted", deleted)
    columns[4].metric("Dimensions", metadata.get("embedding_dimensions", "—"))
    columns[5].metric("Distance", metadata.get("distance_metric", "—"))

    st.subheader("Database configuration")
    st.json(
        {
            "database": str(path),
            "schema_version": metadata.get("schema_version", "legacy/not initialized"),
            "embedding_model": metadata.get("embedding_model", "legacy/not initialized"),
            "embedding_dimensions": metadata.get("embedding_dimensions", "legacy/not initialized"),
            "distance_metric": metadata.get("distance_metric", "legacy/not initialized"),
        }
    )


def memory_explorer(memories: list[dict[str, object]]) -> None:
    st.header("Memory explorer")
    if not memories:
        st.info("No memories found.")
        return

    table = pd.DataFrame(
        [
            {
                "ID": memory["id"],
                "Status": memory["status"],
                "Source": memory["source"],
                "Namespace": memory["namespace"],
                "Type": memory["type"],
                "Created (ms)": memory["created_at_ms"],
                "Dimensions": memory["dimensions"],
                "Content preview": str(memory["content"])[:120],
            }
            for memory in memories
        ]
    )
    st.dataframe(table, use_container_width=True, hide_index=True)

    selected_id = st.selectbox("Inspect memory", [memory["id"] for memory in memories])
    selected = next(memory for memory in memories if memory["id"] == selected_id)
    st.subheader("Selected memory")
    left, right = st.columns([2, 1])
    with left:
        st.text_area("Content", str(selected["content"]), height=220, disabled=True)
    with right:
        status_badge(str(selected["status"]))
        st.json(
            {
                "id": selected["id"],
                "source": selected["source"],
                "namespace": selected["namespace"],
                "type": selected["type"],
                "model": selected["model"],
                "dimensions": selected["dimensions"],
                "normalized": selected["normalized"],
                "metadata": selected["metadata"],
            }
        )


def ingestion_view(path: Path) -> None:
    st.header("Ingest & chunking")
    st.write("Preview the chunks before embedding them into SQLite.")
    uploaded = st.file_uploader("Choose a Markdown or text file", type=["md", "markdown", "txt"])
    source = st.text_input("Source label", value=uploaded.name if uploaded else "notes/example.md")
    namespace = st.text_input("Namespace", value="default")
    memory_type = st.selectbox(
        "Memory type",
        ["document", "fact", "decision", "event", "note", "procedure"],
    )
    chunk_size = st.number_input("Chunk size (words)", min_value=10, value=200, step=10)
    overlap = st.number_input("Overlap (words)", min_value=0, value=40, step=5)
    text = uploaded.getvalue().decode("utf-8") if uploaded else st.text_area("Or paste text", height=180)

    if text:
        try:
            chunks = chunk_text(text, int(chunk_size), int(overlap))
        except ValueError as error:
            st.error(str(error))
            return

        st.info(f"Preview: {len(chunks)} chunks will be created.")
        for index, (content, start, end) in enumerate(chunks):
            with st.expander(f"Chunk {index:04d} · words {start}–{end}"):
                st.write(content)

        if st.button("Embed and store chunks", type="primary"):
            provider = get_provider()
            store = load_store(path)
            count = ingest_text(
                text,
                source,
                store,
                provider,
                namespace=namespace,
                chunk_size=int(chunk_size),
                overlap=int(overlap),
                memory_type=memory_type,
            )
            st.success(f"Stored {count} chunks in {path}.")
            st.rerun()


def search_view(path: Path) -> None:
    st.header("Search playground")
    query = st.text_area("Natural-language query", placeholder="How does this project store memories?")
    top_k = st.slider("Results", min_value=1, max_value=20, value=5)
    search_mode = st.selectbox(
        "Search mode",
        ["Exact baseline", "HNSW (ANN)"],
        help="Exact search checks every matching memory. HNSW searches an in-memory approximate index.",
    )
    ef_search = st.slider(
        "HNSW ef_search",
        min_value=8,
        max_value=256,
        value=64,
        step=8,
        help="Higher values inspect more candidates and usually improve recall at the cost of more work.",
        disabled=search_mode == "Exact baseline",
    )
    _, records = read_database(path)
    sources = sorted({str(record["source"]) for record in records if record["source"]})
    namespaces = sorted({str(record["namespace"]) for record in records})
    memory_types = sorted({str(record["type"]) for record in records})
    filter_columns = st.columns(4)
    with filter_columns[0]:
        namespace_filter = st.selectbox("Namespace", ["Any"] + namespaces)
    with filter_columns[1]:
        source_filter = st.selectbox("Source", ["Any"] + sources)
    with filter_columns[2]:
        type_filter = st.selectbox("Memory type", ["Any"] + memory_types)
    with filter_columns[3]:
        status_filter = st.selectbox("Status", ["active", "archived", "deleted"])

    time_columns = st.columns(2)
    with time_columns[0]:
        min_time = st.number_input("Minimum created_at_ms (0 = any)", min_value=0, value=0)
    with time_columns[1]:
        max_time = st.number_input("Maximum created_at_ms (0 = any)", min_value=0, value=0)

    if st.button("Search", type="primary") and query.strip():
        try:
            provider = get_provider()
            vector = provider.embed([query])[0]
            store = load_store(path)
            namespace = None if namespace_filter == "Any" else namespace_filter
            if search_mode == "HNSW (ANN)":
                unsupported_filters = []
                if source_filter != "Any":
                    unsupported_filters.append("source")
                if type_filter != "Any":
                    unsupported_filters.append("memory type")
                if status_filter != "active":
                    unsupported_filters.append("status")
                if min_time != 0 or max_time != 0:
                    unsupported_filters.append("created time")
                if unsupported_filters:
                    st.warning(
                        "HNSW v1 supports only namespace filtering. Use Exact baseline for: "
                        + ", ".join(unsupported_filters)
                    )
                    return
                results = store.search_hnsw(
                    vector,
                    top_k=top_k,
                    ef_search=ef_search,
                    namespace=namespace,
                    model=provider.model_name,
                )
            else:
                results = store.search(
                    vector,
                    top_k=top_k,
                    namespace=namespace,
                    source=None if source_filter == "Any" else source_filter,
                    memory_type=None if type_filter == "Any" else type_filter,
                    status=status_filter,
                    min_created_at_ms=None if min_time == 0 else int(min_time),
                    max_created_at_ms=None if max_time == 0 else int(max_time),
                    model=provider.model_name,
                )
        except ValueError as error:
            st.error(str(error))
            return

        if not results:
            st.info("No active memories matched this query.")
            return
        for rank, result in enumerate(results, start=1):
            with st.container(border=True):
                st.subheader(f"#{rank} · score {result.score:.4f}")
                st.write(result.content)
                st.caption(
                    f"ID: {result.id} · Source: {result.source or 'none'} · "
                    f"Model: {result.embedding_model} · Dimensions: {result.dimensions}"
                )
                if result.metadata:
                    st.json(result.metadata)


def hnsw_observatory(path: Path) -> None:
    st.header("HNSW observatory")
    st.write(
        "Run the same query through exact cosine search and the cached HNSW index, "
        "then inspect latency, agreement, and a 2D projection of the full embeddings."
    )
    st.info(
        "The projection is only a visual aid. Retrieval and HNSW still operate on the full "
        "384-dimensional vectors. The first HNSW query includes index construction; the "
        "second query demonstrates warm-cache behavior."
    )

    _, records = read_database(path)
    active_records = [record for record in records if record["status"] == "active"]
    namespaces = sorted({str(record["namespace"]) for record in active_records})
    query = st.text_area(
        "Query to inspect",
        value="How does the project retrieve relevant memories?",
        key="hnsw_observatory_query",
    )
    controls = st.columns(3)
    with controls[0]:
        namespace_filter = st.selectbox("Namespace", ["Any"] + namespaces, key="hnsw_namespace")
    with controls[1]:
        top_k = st.slider("Top-k", 1, 20, 5, key="hnsw_top_k")
    with controls[2]:
        ef_search = st.slider("ef_search", 8, 256, 64, step=8, key="hnsw_ef")

    if not st.button("Run exact + HNSW comparison", type="primary") or not query.strip():
        return
    if not active_records:
        st.warning("No active memories are available. Ingest some text first.")
        return

    try:
        provider = get_provider()
        vector = provider.embed([query])[0]
        store = load_store(path)
        namespace = None if namespace_filter == "Any" else namespace_filter

        exact_started = time.perf_counter()
        exact_results = store.search(
            vector,
            top_k=top_k,
            namespace=namespace,
            status="active",
            model=provider.model_name,
        )
        exact_ms = (time.perf_counter() - exact_started) * 1000

        first_hnsw_started = time.perf_counter()
        hnsw_results = store.search_hnsw(
            vector,
            top_k=top_k,
            ef_search=ef_search,
            namespace=namespace,
            model=provider.model_name,
        )
        first_hnsw_ms = (time.perf_counter() - first_hnsw_started) * 1000

        warm_hnsw_started = time.perf_counter()
        store.search_hnsw(
            vector,
            top_k=top_k,
            ef_search=ef_search,
            namespace=namespace,
            model=provider.model_name,
        )
        warm_hnsw_ms = (time.perf_counter() - warm_hnsw_started) * 1000
    except ValueError as error:
        st.error(str(error))
        return

    exact_ids = [result.id for result in exact_results]
    hnsw_ids = [result.id for result in hnsw_results]
    exact_set = set(exact_ids)
    hnsw_set = set(hnsw_ids)
    overlap = exact_set & hnsw_set
    recall = len(overlap) / len(exact_set) if exact_set else 0.0
    top1_agreement = bool(exact_ids and hnsw_ids and exact_ids[0] == hnsw_ids[0])

    st.subheader("What happened")
    metrics = st.columns(5)
    metrics[0].metric("Exact latency", f"{exact_ms:.2f} ms")
    metrics[1].metric("HNSW first query", f"{first_hnsw_ms:.2f} ms")
    metrics[2].metric("HNSW warm query", f"{warm_hnsw_ms:.2f} ms")
    metrics[3].metric(f"Top-{top_k} overlap", f"{recall:.0%}")
    metrics[4].metric("Top-1 agreement", "Yes" if top1_agreement else "No")

    comparison_rows = []
    for rank, result in enumerate(exact_results, start=1):
        comparison_rows.append(
            {
                "Exact rank": rank,
                "Exact ID": result.id,
                "Exact score": round(result.score, 4),
                "HNSW rank": hnsw_ids.index(result.id) + 1 if result.id in hnsw_ids else None,
                "HNSW score": round(next((hit.score for hit in hnsw_results if hit.id == result.id), 0.0), 4),
                "Agreement": "both" if result.id in hnsw_set else "exact only",
            }
        )
    for rank, result in enumerate(hnsw_results, start=1):
        if result.id not in exact_set:
            comparison_rows.append(
                {
                    "Exact rank": None,
                    "Exact ID": None,
                    "Exact score": None,
                    "HNSW rank": rank,
                    "HNSW score": round(result.score, 4),
                    "Agreement": "HNSW only",
                }
            )
    st.subheader("Exact versus HNSW results")
    st.dataframe(pd.DataFrame(comparison_rows), use_container_width=True, hide_index=True)

    projection_memories = [memory for memory in store.list(namespace) if memory.status == "active"][:2000]
    if len(projection_memories) >= 2:
        matrix = np.asarray(
            [memory.embedding for memory in projection_memories],
            dtype=np.float64,
        )
        centered = matrix - matrix.mean(axis=0)
        _, _, components = np.linalg.svd(centered, full_matrices=False)
        points = centered @ components[:2].T
        query_centered = np.asarray(vector, dtype=np.float64) - matrix.mean(axis=0)
        query_point = query_centered @ components[:2].T
        exact_set = set(exact_ids)
        hnsw_set = set(hnsw_ids)
        projection = pd.DataFrame(
            {
                "x": points[:, 0],
                "y": points[:, 1],
                "kind": [
                    "both" if memory.id in (exact_set & hnsw_set)
                    else "exact only" if memory.id in exact_set
                    else "HNSW only" if memory.id in hnsw_set
                    else "other active memory"
                    for memory in projection_memories
                ],
                "id": [memory.id for memory in projection_memories],
            }
        )
        query_row = pd.DataFrame(
            {"x": [query_point[0]], "y": [query_point[1]], "kind": ["query"], "id": ["QUERY"]}
        )
        st.subheader("Embedding space projection")
        st.caption("PCA projection of active vectors. The query and returned memories are highlighted.")
        st.scatter_chart(pd.concat([projection, query_row], ignore_index=True), x="x", y="y", color="kind")
        st.dataframe(
            projection[projection["kind"] != "other active memory"],
            use_container_width=True,
            hide_index=True,
        )


def embedding_view(path: Path) -> None:
    st.header("Embedding inspector")
    store = load_store(path)
    memories = [memory for memory in store.list(None) if memory.status == "active"]
    if not memories:
        st.info("No active memories to inspect.")
        return

    selected_id = st.selectbox("Memory", [memory.id for memory in memories])
    selected = next(memory for memory in memories if memory.id == selected_id)
    vector = np.asarray(selected.embedding, dtype=np.float32)
    st.metric("Vector dimensions", len(vector))
    st.metric("Vector norm", f"{np.linalg.norm(vector):.6f}")
    st.write("First 16 values")
    st.code([round(float(value), 6) for value in vector[:16]])

    if len(memories) >= 2:
        matrix = np.asarray([memory.embedding for memory in memories], dtype=np.float64)
        centered = matrix - matrix.mean(axis=0)
        _, _, components = np.linalg.svd(centered, full_matrices=False)
        points = centered @ components[:2].T
        projection = pd.DataFrame(
            {"x": points[:, 0], "y": points[:, 1], "id": [memory.id for memory in memories]}
        )
        st.subheader("2D projection of active embeddings")
        st.caption("This is for visual inspection only; retrieval still uses the full vector.")
        st.scatter_chart(projection, x="x", y="y")


def lifecycle_view(path: Path) -> None:
    st.header("Memory lifecycle")
    store = load_store(path)
    memories = store.list(None)
    if not memories:
        st.info("No memories found.")
        return

    selected_id = st.selectbox("Memory", [memory.id for memory in memories])
    selected = next(memory for memory in memories if memory.id == selected_id)
    st.write(selected.content)
    status_badge(selected.status)

    first, second, third = st.columns(3)
    with first:
        if st.button("Restore / activate"):
            store.restore(selected.id)
            st.success("Memory restored.")
            st.rerun()
    with second:
        if st.button("Archive"):
            store.archive(selected.id)
            st.warning("Memory archived.")
            st.rerun()
    with third:
        if st.button("Soft delete"):
            store.delete(selected.id)
            st.error("Memory soft-deleted.")
            st.rerun()


def main() -> None:
    path = database_path()
    if path.exists():
        try:
            load_store(path)
        except ValueError as error:
            st.error(f"Database validation failed: {error}")
            st.stop()
    metadata, memories = read_database(path)
    view = st.sidebar.radio(
        "View",
        [
            "Dashboard",
            "Memory explorer",
            "Ingest & chunking",
            "Search playground",
            "HNSW observatory",
            "Embedding inspector",
            "Memory lifecycle",
        ],
    )

    if view == "Dashboard":
        dashboard(path, metadata, memories)
    elif view == "Memory explorer":
        memory_explorer(memories)
    elif view == "Ingest & chunking":
        ingestion_view(path)
    elif view == "Search playground":
        search_view(path)
    elif view == "HNSW observatory":
        hnsw_observatory(path)
    elif view == "Embedding inspector":
        embedding_view(path)
    else:
        lifecycle_view(path)


if __name__ == "__main__":
    main()

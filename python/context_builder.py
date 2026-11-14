"""Build deterministic, evidence-grounded context packages for RAG workflows.

This module deliberately stops before answer generation. Its job is to make the
retrieved evidence explicit, bounded, and easy for another model or service to
consume.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from local_ai_memory import MemoryStore


CONTEXT_SCHEMA_VERSION = "1"


@dataclass(frozen=True)
class ContextPackage:
    """A JSON-ready package of retrieved evidence and its retrieval trace."""

    query: str
    retrieval: dict[str, Any]
    evidence: list[dict[str, Any]]
    context_text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CONTEXT_SCHEMA_VERSION,
            "query": self.query,
            "retrieval": self.retrieval,
            "evidence": self.evidence,
            "context_text": self.context_text,
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


def build_context(
    store: MemoryStore,
    query: str,
    query_embedding: list[float],
    *,
    top_k: int = 5,
    max_chars: int = 6_000,
    mode: str = "exact",
    ef_search: int = 64,
    namespace: str | None = None,
    model: str | None = None,
) -> ContextPackage:
    """Retrieve memories and package them into bounded RAG evidence.

    Results are already deterministically ordered by the Rust retrieval layer.
    Archived, deleted, and expired memories are excluded by normal retrieval.
    ``max_chars`` limits the combined evidence text, while metadata remains
    available for every selected item.
    """
    if not query.strip():
        raise ValueError("query cannot be empty")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if mode not in {"exact", "hnsw"}:
        raise ValueError("mode must be 'exact' or 'hnsw'")

    if mode == "hnsw":
        results = store.search_hnsw(
            query_embedding,
            top_k=top_k,
            ef_search=ef_search,
            namespace=namespace,
            model=model,
        )
    else:
        results = store.search(
            query_embedding,
            top_k=top_k,
            namespace=namespace,
            status="active",
            model=model,
        )

    evidence: list[dict[str, Any]] = []
    context_sections: list[str] = []
    remaining = max_chars
    truncated_count = 0

    for rank, result in enumerate(results, start=1):
        if remaining <= 0:
            break

        source = result.source or "unknown source"
        citation = f"[{rank}] {source} · memory:{result.id}"
        available = remaining - len(citation) - 1
        if available <= 0:
            break

        text = result.content
        truncated = len(text) > available
        text_budget = available - 1 if truncated else available
        selected_text = text[:max(text_budget, 0)].rstrip()
        if truncated:
            selected_text = selected_text.rstrip() + "…"
            truncated_count += 1

        evidence.append(
            {
                "rank": rank,
                "memory_id": result.id,
                "text": selected_text,
                "source": result.source,
                "score": round(float(result.score), 6),
                "namespace": result.namespace,
                "status": result.status,
                "metadata": dict(result.metadata),
                "citation": citation,
                "truncated": truncated,
            }
        )
        context_sections.append(f"{citation}\n{selected_text}")
        remaining -= len(citation) + len(selected_text) + 1

    return ContextPackage(
        query=query,
        retrieval={
            "mode": mode,
            "requested_top_k": top_k,
            "returned_count": len(results),
            "selected_count": len(evidence),
            "max_chars": max_chars,
            "used_chars": max_chars - remaining,
            "ef_search": ef_search if mode == "hnsw" else None,
            "namespace": namespace,
            "truncated_count": truncated_count,
        },
        evidence=evidence,
        context_text="\n\n".join(context_sections),
    )


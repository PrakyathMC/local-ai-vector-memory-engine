"""Local HTTP API for the memory engine."""

from __future__ import annotations

import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(Path(__file__).parent))

from answer_generation import (  # noqa: E402
    AnswerGenerationError,
    MockAnswerGenerator,
    OllamaAnswerGenerator,
    generate_grounded_answer,
)
from context_builder import build_context  # noqa: E402
from embedding import SentenceTransformerProvider  # noqa: E402
from ingest_documents import (  # noqa: E402
    MODEL_DIMENSIONS,
    MODEL_DIR,
    MODEL_NAME,
)
from local_ai_memory import Memory, MemoryStore  # noqa: E402


app = FastAPI(title="Local AI Memory Engine API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class MemoryInput(BaseModel):
    id: str
    content: str
    embedding: list[float]
    namespace: str = "default"
    model: str = "test-embedder"
    source: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)
    created_at_ms: int = 0
    updated_at_ms: int = 0
    normalized: bool = False
    memory_type: str = "note"
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    expires_at_ms: int | None = None
    admission_reason: str = "http api"


class SearchRequest(BaseModel):
    query_embedding: list[float]
    top_k: int = Field(default=5, ge=1, le=100)
    mode: str = "exact"
    ef_search: int = Field(default=64, ge=1, le=1000)
    namespace: str | None = None
    source: str | None = None
    memory_type: str | None = None
    status: str = "active"
    min_created_at_ms: int | None = None
    max_created_at_ms: int | None = None
    model: str | None = None


class ContextRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=100)
    max_chars: int = Field(default=6_000, ge=100, le=100_000)
    mode: str = "exact"
    ef_search: int = Field(default=64, ge=1, le=1000)
    namespace: str | None = None


class AnswerRequest(ContextRequest):
    provider: str = "mock"
    ollama_model: str = "llama3.2:3b"
    ollama_url: str = "http://localhost:11434"


class FeedbackRequest(BaseModel):
    useful: bool


class LifecycleRequest(BaseModel):
    importance: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    expires_at_ms: int | None = None
    admission_reason: str = "http api"


def database_path(db: str) -> str:
    return str((PROJECT_ROOT / db).resolve()) if not Path(db).is_absolute() else db


@lru_cache(maxsize=1)
def provider() -> SentenceTransformerProvider:
    return SentenceTransformerProvider(MODEL_NAME, MODEL_DIR, MODEL_DIMENSIONS)


def store_for(db: str) -> MemoryStore:
    return MemoryStore(database_path(db))


def serialize_result(result: Any) -> dict[str, Any]:
    return {
        "id": result.id,
        "content": result.content,
        "score": result.score,
        "namespace": result.namespace,
        "source": result.source,
        "status": result.status,
        "embedding_model": result.embedding_model,
        "dimensions": result.dimensions,
        "metadata": dict(result.metadata),
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "local-ai-memory"}


@app.get("/memories")
def list_memories(
    db: str = Query("data/ingestion-demo.db"),
    namespace: str | None = None,
) -> list[dict[str, Any]]:
    memories = store_for(db).list(namespace)
    return [
        {
            "id": memory.id,
            "content": memory.content,
            "namespace": memory.namespace,
            "source": memory.source,
            "status": memory.status,
        }
        for memory in memories
    ]


@app.post("/memories")
def admit_memory(payload: MemoryInput, db: str = Query("data/ingestion-demo.db")) -> dict[str, Any]:
    store = store_for(db)
    memory = Memory(
        payload.id,
        payload.content,
        payload.embedding,
        namespace=payload.namespace,
        model=payload.model,
        source=payload.source,
        metadata=payload.metadata,
        created_at_ms=payload.created_at_ms,
        updated_at_ms=payload.updated_at_ms,
        normalized=payload.normalized,
        memory_type=payload.memory_type,
    )
    if store.get(payload.id) is not None:
        raise HTTPException(status_code=409, detail="memory ID already exists")
    admitted = store.admit(
        memory,
        importance=payload.importance,
        confidence=payload.confidence,
        expires_at_ms=payload.expires_at_ms,
        admission_reason=payload.admission_reason,
    )
    return {"id": payload.id, "admitted": admitted, "status": "active" if admitted else "rejected"}


@app.post("/search")
def search(payload: SearchRequest, db: str = Query("data/ingestion-demo.db")) -> dict[str, Any]:
    store = store_for(db)
    if payload.mode not in {"exact", "hnsw"}:
        raise HTTPException(status_code=400, detail="mode must be exact or hnsw")
    if payload.mode == "hnsw":
        results = store.search_hnsw(
            payload.query_embedding,
            top_k=payload.top_k,
            ef_search=payload.ef_search,
            namespace=payload.namespace,
            model=payload.model,
        )
    else:
        results = store.search(
            payload.query_embedding,
            top_k=payload.top_k,
            namespace=payload.namespace,
            source=payload.source,
            memory_type=payload.memory_type,
            status=payload.status,
            min_created_at_ms=payload.min_created_at_ms,
            max_created_at_ms=payload.max_created_at_ms,
            model=payload.model,
        )
    return {"mode": payload.mode, "results": [serialize_result(result) for result in results]}


def build_api_context(payload: ContextRequest, db: str) -> Any:
    model = provider()
    query_embedding = model.embed([payload.query])[0]
    return build_context(
        store_for(db),
        payload.query,
        query_embedding,
        top_k=payload.top_k,
        max_chars=payload.max_chars,
        mode=payload.mode,
        ef_search=payload.ef_search,
        namespace=payload.namespace,
        model=model.model_name,
    )


@app.post("/context")
def context(payload: ContextRequest, db: str = Query("data/ingestion-demo.db")) -> dict[str, Any]:
    return build_api_context(payload, db).to_dict()


@app.post("/trace")
def trace(payload: ContextRequest, db: str = Query("data/ingestion-demo.db")) -> dict[str, Any]:
    """Return a browser-friendly trace of the retrieval path."""
    started = time.perf_counter()
    package = build_api_context(payload, db)
    elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
    evidence_ids = [item["memory_id"] for item in package.evidence]
    return {
        "trace_version": "1",
        "query": payload.query,
        "stages": [
            {"name": "query", "status": "complete", "details": {"text": payload.query}},
            {
                "name": "embedding",
                "status": "complete",
                "details": {"model": MODEL_NAME, "dimensions": MODEL_DIMENSIONS},
            },
            {
                "name": "retrieval",
                "status": "complete",
                "details": package.retrieval,
            },
            {
                "name": "evidence",
                "status": "complete",
                "details": {"selected_ids": evidence_ids},
            },
        ],
        "latency_ms": elapsed_ms,
        "context": package.to_dict(),
    }


@app.post("/answer")
def answer(payload: AnswerRequest, db: str = Query("data/ingestion-demo.db")) -> dict[str, Any]:
    package = build_api_context(payload, db)
    if payload.provider == "mock":
        generator = MockAnswerGenerator()
    elif payload.provider == "ollama":
        generator = OllamaAnswerGenerator(payload.ollama_model, payload.ollama_url)
    else:
        raise HTTPException(status_code=400, detail="provider must be mock or ollama")
    try:
        generated = generate_grounded_answer(package, generator)
    except AnswerGenerationError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
    return {"answer": generated, "provider": generator.provider_name, "model": generator.model_name, "context": package.to_dict()}


@app.post("/memories/{memory_id}/feedback")
def feedback(memory_id: str, payload: FeedbackRequest, db: str = Query("data/ingestion-demo.db")) -> dict[str, str]:
    try:
        store_for(db).record_feedback(memory_id, payload.useful, int(time.time() * 1000))
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    return {"id": memory_id, "status": "recorded"}


@app.patch("/memories/{memory_id}/lifecycle")
def lifecycle(memory_id: str, payload: LifecycleRequest, db: str = Query("data/ingestion-demo.db")) -> dict[str, str]:
    try:
        store_for(db).set_lifecycle(
            memory_id,
            payload.importance,
            payload.confidence,
            payload.expires_at_ms,
            payload.admission_reason,
        )
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    return {"id": memory_id, "status": "updated"}


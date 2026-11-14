from pathlib import Path

from fastapi.testclient import TestClient

import sys

sys.path.insert(0, str(Path(__file__).parents[1]))

from api import app  # noqa: E402


def test_health_endpoint():
    response = TestClient(app).get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_memory_can_be_admitted_listed_and_searched(tmp_path):
    database = tmp_path / "api-test.db"
    client = TestClient(app)
    params = {"db": str(database)}
    payload = {
        "id": "api-memory-1",
        "content": "The API stores a memory and makes it searchable.",
        "embedding": [1.0, 0.0, 0.0],
        "namespace": "api-tests",
        "model": "test-embedder",
        "source": "api-test",
        "importance": 0.9,
        "confidence": 0.9,
    }

    admitted = client.post("/memories", params=params, json=payload)
    assert admitted.status_code == 200
    assert admitted.json()["admitted"] is True

    listed = client.get("/memories", params={**params, "namespace": "api-tests"})
    assert listed.status_code == 200
    assert listed.json()[0]["id"] == "api-memory-1"

    searched = client.post(
        "/search",
        params=params,
        json={
            "query_embedding": [1.0, 0.0, 0.0],
            "top_k": 1,
            "namespace": "api-tests",
            "model": "test-embedder",
        },
    )
    assert searched.status_code == 200
    assert searched.json()["results"][0]["id"] == "api-memory-1"


def test_search_rejects_unknown_mode():
    client = TestClient(app)

    response = client.post(
        "/search",
        json={"query_embedding": [1.0, 0.0], "mode": "unknown"},
    )

    assert response.status_code == 400

"""Integration tests for the FastAPI endpoints."""

from __future__ import annotations

import io
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# App fixture with mocked singletons
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    """Create a FastAPI test client with mocked RAG chain and indexer."""
    mock_chain = MagicMock()
    mock_chain.query.return_value = {
        "answer": "Hybrid search combines dense and sparse retrieval.",
        "sources": [
            {
                "content": "Hybrid search uses FAISS + BM25.",
                "source": "doc.txt",
                "page": 1,
                "chunk_id": 0,
                "score": 0.9432,
            }
        ],
        "chat_history": [],
    }
    mock_chain.clear_memory.return_value = None

    mock_indexer = MagicMock()
    mock_indexer.documents = [MagicMock()] * 42

    with patch("api.main._rag_chain", mock_chain), \
         patch("api.main._indexer", mock_indexer), \
         patch("api.main._agent_router", MagicMock()):
        from api.main import app
        yield TestClient(app)


# ---------------------------------------------------------------------------
# /health tests
# ---------------------------------------------------------------------------

class TestHealthEndpoint:

    def test_health_ok(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["index_loaded"] is True
        assert data["num_indexed_docs"] == 42
        assert data["version"] == "1.0.0"


# ---------------------------------------------------------------------------
# /query tests
# ---------------------------------------------------------------------------

class TestQueryEndpoint:

    def test_query_returns_answer(self, client):
        response = client.post("/query", json={
            "query": "What is hybrid search?",
            "mode": "hybrid_rag",
            "top_k": 5,
            "stream": False,
        })
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert len(data["answer"]) > 0

    def test_query_returns_sources(self, client):
        response = client.post("/query", json={
            "query": "What is RRF?",
            "stream": False,
        })
        data = response.json()
        assert "sources" in data
        assert isinstance(data["sources"], list)

    def test_query_returns_latency(self, client):
        response = client.post("/query", json={"query": "Test?", "stream": False})
        data = response.json()
        assert "latency_ms" in data
        assert data["latency_ms"] >= 0

    def test_query_too_short_rejected(self, client):
        response = client.post("/query", json={"query": "Hi", "stream": False})
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# /memory tests
# ---------------------------------------------------------------------------

class TestMemoryEndpoint:

    def test_clear_memory(self, client):
        response = client.delete("/memory")
        assert response.status_code == 200
        assert response.json()["message"] == "Conversation memory cleared."

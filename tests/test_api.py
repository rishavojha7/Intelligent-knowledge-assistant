"""
End-to-end API tests.

Tests the FastAPI endpoints using TestClient. Uses mocked LLM
calls to avoid requiring real API keys during testing.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app


@pytest.fixture
def client():
    """Create a FastAPI TestClient."""
    app = create_app()
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for GET /api/v1/health."""

    def test_health_returns_200(self, client):
        """Health endpoint should return 200."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200

    def test_health_response_schema(self, client):
        """Health response should have expected fields."""
        response = client.get("/api/v1/health")
        data = response.json()
        assert "status" in data
        assert "documents_count" in data
        assert "total_chunks" in data


class TestRootEndpoint:
    """Tests for GET /."""

    def test_root_returns_200(self, client):
        """Root should return 200 with a welcome message."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data


class TestAskEndpoint:
    """Tests for POST /api/v1/ask."""

    def test_ask_validates_empty_question(self, client):
        """Empty question should be rejected."""
        response = client.post("/api/v1/ask", json={"question": ""})
        assert response.status_code == 422  # Validation error

    def test_ask_validates_missing_question(self, client):
        """Missing question field should be rejected."""
        response = client.post("/api/v1/ask", json={})
        assert response.status_code == 422

    @patch("src.api.routes.run_query")
    def test_ask_returns_expected_schema(self, mock_run_query, client):
        """Ask endpoint should return the expected response schema."""
        from src.models import QueryResult, SourceReference

        mock_run_query.return_value = QueryResult(
            answer="RAG stands for Retrieval-Augmented Generation.",
            sources=[
                SourceReference(
                    document="rag_architecture_patterns.pdf",
                    page=1,
                    chunk_id="test::page0::chunk0",
                    relevance_score=0.85,
                )
            ],
            workflow="simple_rag",
            reasoning_steps=["Retrieved 3 relevant chunks"],
            confidence="high",
        )

        response = client.post(
            "/api/v1/ask",
            json={"question": "What is RAG?"},
        )
        assert response.status_code == 200

        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "workflow" in data
        assert "reasoning_steps" in data
        assert "confidence" in data
        assert data["workflow"] == "simple_rag"
        assert len(data["sources"]) == 1
        assert data["sources"][0]["document"] == "rag_architecture_patterns.pdf"

    @patch("src.api.routes.run_query")
    def test_ask_with_use_agent(self, mock_run_query, client):
        """use_agent parameter should be passed through."""
        from src.models import QueryResult

        mock_run_query.return_value = QueryResult(
            answer="Comparison result",
            workflow="agentic",
            reasoning_steps=["Step 1", "Step 2"],
        )

        response = client.post(
            "/api/v1/ask",
            json={"question": "Compare FAISS and Pinecone", "use_agent": True},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["workflow"] == "agentic"

        # Verify use_agent was passed
        mock_run_query.assert_called_once_with(
            question="Compare FAISS and Pinecone",
            use_agent=True,
        )


class TestDocumentsEndpoint:
    """Tests for GET /api/v1/documents."""

    def test_documents_returns_200(self, client):
        """Documents endpoint should return 200."""
        response = client.get("/api/v1/documents")
        assert response.status_code == 200

    def test_documents_response_schema(self, client):
        """Documents response should have expected fields."""
        response = client.get("/api/v1/documents")
        data = response.json()
        assert "documents" in data
        assert "total_documents" in data
        assert "total_chunks" in data

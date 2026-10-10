"""Shared test fixtures for the Knowledge Assistant test suite."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Ensure tests don't require real Azure credentials or network access
os.environ.setdefault("AZURE_OPENAI_API_KEY", "test-key-not-real")
os.environ.setdefault("AZURE_OPENAI_ENDPOINT", "https://test.openai.azure.com/")
os.environ.setdefault("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
os.environ.setdefault("AZURE_OPENAI_DEPLOYMENT_EMBED", "text-embedding-3-small")
os.environ.setdefault("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
os.environ.setdefault("EMBEDDING_PROVIDER", "huggingface")
os.environ.setdefault("RETRIEVAL_MODE", "dense")
os.environ.setdefault("RERANKER_ENABLED", "false")
os.environ.setdefault("LOG_LEVEL", "WARNING")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


@pytest.fixture
def corpus_dir():
    """Return the path to the test corpus directory."""
    corpus = Path(__file__).resolve().parent.parent / "corpus"
    if corpus.is_dir():
        return corpus
    pytest.skip("Corpus directory not found")


@pytest.fixture
def sample_pdf(corpus_dir):
    """Return the path to one of the corpus PDFs for testing."""
    pdf = corpus_dir / "rag_architecture_patterns.pdf"
    if pdf.exists():
        return pdf
    pdfs = list(corpus_dir.glob("*.pdf"))
    if pdfs:
        return pdfs[0]
    pytest.skip("No PDF files found in corpus")


@pytest.fixture
def temp_chroma_dir():
    """Provide a temporary directory for ChromaDB in tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def test_settings(temp_chroma_dir, corpus_dir):
    """Create test Settings with temporary ChromaDB directory."""
    from src.config import Settings

    return Settings(
        azure_openai_api_key="test-key-not-real",
        azure_openai_endpoint="https://test.openai.azure.com/",
        azure_openai_deployment="gpt-4o",
        azure_openai_deployment_embed="text-embedding-3-small",
        azure_openai_api_version="2024-08-01-preview",
        embedding_provider="huggingface",
        chroma_persist_dir=temp_chroma_dir,
        chroma_collection_name="test_collection",
        corpus_dir=str(corpus_dir),
        chunk_size=500,
        chunk_overlap=100,
        log_level="WARNING",
    )


@pytest.fixture
def mock_llm():
    """Create a mock LLM that returns a predetermined response."""
    mock = MagicMock()
    mock.invoke.return_value = MagicMock(
        content="This is a test answer. [rag_architecture_patterns.pdf, Page 1]"
    )
    return mock

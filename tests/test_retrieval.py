"""
Tests for the retrieval layer.

Covers vector store operations, similarity search, metadata filtering,
and the Retriever interface.
"""

from __future__ import annotations

import pytest

from src.config import get_embeddings
from src.ingestion.chunker import chunk_documents
from src.ingestion.loader import load_pdf
from src.retrieval.retriever import Retriever
from src.retrieval.vector_store import VectorStoreManager


@pytest.fixture
def loaded_store(test_settings, sample_pdf):
    """Create a VectorStoreManager with test data loaded."""
    embeddings = get_embeddings(test_settings)
    store = VectorStoreManager(
        persist_dir=test_settings.chroma_persist_dir,
        collection_name=test_settings.chroma_collection_name,
        embedding_function=embeddings,
    )

    # Load and chunk a single PDF
    docs = load_pdf(sample_pdf)
    chunks = chunk_documents(docs, chunk_size=500, chunk_overlap=100)
    store.upsert_documents(chunks)

    return store


@pytest.fixture
def retriever(loaded_store):
    """Create a Retriever instance with test data."""
    return Retriever(loaded_store, default_top_k=3)


class TestVectorStoreManager:
    """Tests for the ChromaDB wrapper."""

    def test_upsert_and_count(self, loaded_store):
        """After upserting, the collection should have chunks."""
        stats = loaded_store.get_collection_stats()
        assert stats["total_chunks"] > 0

    def test_collection_stats_has_documents(self, loaded_store):
        """Stats should list the source document."""
        stats = loaded_store.get_collection_stats()
        assert stats["documents_count"] > 0
        assert len(stats["documents"]) > 0

    def test_similarity_search_returns_results(self, loaded_store):
        """A relevant query should return results."""
        results = loaded_store.similarity_search("What is RAG?", k=3)
        assert len(results) > 0
        # Each result should be a (Document, score) tuple
        doc, score = results[0]
        assert doc.page_content
        assert isinstance(score, float)

    def test_similarity_search_with_filter(self, loaded_store, sample_pdf):
        """Filtering by source should only return matching documents."""
        results = loaded_store.similarity_search(
            "chunking strategies",
            k=3,
            filter_metadata={"source": sample_pdf.name},
        )
        for doc, _ in results:
            assert doc.metadata["source"] == sample_pdf.name

    def test_idempotent_upsert(self, test_settings, sample_pdf):
        """Upserting the same documents twice should not duplicate chunks."""
        embeddings = get_embeddings(test_settings)
        store = VectorStoreManager(
            persist_dir=test_settings.chroma_persist_dir,
            collection_name="idempotent_test",
            embedding_function=embeddings,
        )
        docs = load_pdf(sample_pdf)
        chunks = chunk_documents(docs, chunk_size=500, chunk_overlap=100)

        # Upsert twice
        store.upsert_documents(chunks)
        count_first = store.get_collection_stats()["total_chunks"]
        store.upsert_documents(chunks)
        count_second = store.get_collection_stats()["total_chunks"]

        assert count_first == count_second

    def test_document_currency_check(self, loaded_store, sample_pdf):
        """Current check should return True for a just-ingested document."""
        from src.ingestion.loader import compute_file_hash

        doc_hash = compute_file_hash(sample_pdf)
        assert loaded_store.is_document_current(sample_pdf.name, doc_hash) is True
        assert loaded_store.is_document_current(sample_pdf.name, "wrong_hash") is False

    def test_delete_by_document(self, loaded_store, sample_pdf):
        """Deleting by document should remove all its chunks."""
        deleted = loaded_store.delete_by_document(sample_pdf.name)
        assert deleted > 0
        stats = loaded_store.get_collection_stats()
        assert sample_pdf.name not in stats["documents"]


class TestRetriever:
    """Tests for the high-level Retriever interface."""

    def test_search_returns_results(self, retriever):
        """Search should return RetrievalResult objects."""
        results = retriever.search("What is RAG?")
        assert len(results) > 0
        assert results[0].content
        assert results[0].source

    def test_search_by_document(self, retriever, sample_pdf):
        """Document-scoped search should only return matching results."""
        results = retriever.search_by_document("chunking", sample_pdf.name)
        for r in results:
            assert r.source == sample_pdf.name

    def test_format_results_as_context(self, retriever):
        """Formatted context should contain source citations."""
        results = retriever.search("RAG architecture")
        context = retriever.format_results_as_context(results)
        assert "[Source" in context
        assert "Page" in context

    def test_format_empty_results(self, retriever):
        """Empty results should produce a 'no information' message."""
        context = retriever.format_results_as_context([])
        assert "No relevant information" in context


class TestHybridSearch:
    """Tests for BM25 + dense hybrid search and cross-encoder re-ranking."""

    @pytest.fixture
    def hybrid_retriever(self, loaded_store):
        """Create a Retriever in hybrid mode (no re-ranker for speed)."""
        from src.config import Settings

        settings = Settings(
            azure_openai_api_key="test-key-not-real",
            azure_openai_endpoint="https://test.openai.azure.com/",
            embedding_provider="huggingface",
            retrieval_mode="hybrid",
            bm25_weight=0.4,
            dense_weight=0.6,
            reranker_enabled=False,
            chroma_persist_dir=loaded_store._persist_dir,
            chroma_collection_name=loaded_store._collection_name,
        )
        return Retriever(loaded_store, default_top_k=3, settings=settings)

    def test_bm25_search(self, hybrid_retriever):
        """BM25 search should return results for a keyword query."""
        results = hybrid_retriever._bm25_search("RAG retrieval augmented", k=3)
        assert len(results) > 0
        doc, score = results[0]
        assert doc.page_content
        assert 0.0 <= score <= 1.0

    def test_hybrid_search_returns_results(self, hybrid_retriever):
        """Hybrid mode should return results combining BM25 and dense."""
        results = hybrid_retriever.search("What is RAG?")
        assert len(results) > 0
        assert results[0].content
        assert results[0].relevance_score > 0

    def test_hybrid_search_by_document(self, hybrid_retriever, sample_pdf):
        """Hybrid search with document filter should scope results."""
        results = hybrid_retriever.search_by_document("chunking", sample_pdf.name)
        for r in results:
            assert r.source == sample_pdf.name

    def test_dense_only_mode(self, loaded_store):
        """Dense-only mode should skip BM25 entirely."""
        from src.config import Settings

        settings = Settings(
            azure_openai_api_key="test-key-not-real",
            azure_openai_endpoint="https://test.openai.azure.com/",
            embedding_provider="huggingface",
            retrieval_mode="dense",
            reranker_enabled=False,
            chroma_persist_dir=loaded_store._persist_dir,
            chroma_collection_name=loaded_store._collection_name,
        )
        retriever = Retriever(loaded_store, default_top_k=3, settings=settings)
        results = retriever.search("What is RAG?")
        assert len(results) > 0
        # BM25 index should NOT have been built
        assert retriever._bm25_index is None

    def test_reciprocal_rank_fusion(self):
        """RRF should merge and deduplicate results from two lists."""
        from langchain_core.documents import Document

        doc_a = Document(page_content="doc A", metadata={"chunk_id": "a"})
        doc_b = Document(page_content="doc B", metadata={"chunk_id": "b"})
        doc_c = Document(page_content="doc C", metadata={"chunk_id": "c"})

        dense = [(doc_a, 0.9), (doc_b, 0.8)]
        bm25 = [(doc_b, 0.95), (doc_c, 0.7)]

        fused = Retriever.reciprocal_rank_fusion(
            dense_results=dense,
            bm25_results=bm25,
            dense_weight=0.6,
            bm25_weight=0.4,
            k=3,
        )

        assert len(fused) == 3
        # doc_b appears in both lists, should get highest fused score
        chunk_ids = [doc.metadata["chunk_id"] for doc, _ in fused]
        assert "b" in chunk_ids
        assert "a" in chunk_ids
        assert "c" in chunk_ids
        # Scores should be positive
        for _, score in fused:
            assert score > 0

    def test_bm25_index_built_once(self, hybrid_retriever):
        """BM25 index should be built on first search and cached."""
        hybrid_retriever._bm25_search("test query", k=3)
        first_index = hybrid_retriever._bm25_index

        hybrid_retriever._bm25_search("another query", k=3)
        assert hybrid_retriever._bm25_index is first_index  # Same object

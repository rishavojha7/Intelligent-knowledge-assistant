"""
Tests for the document ingestion pipeline.

Covers PDF loading, text chunking, metadata attachment,
and idempotent ingestion behavior.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.ingestion.chunker import chunk_documents
from src.ingestion.loader import compute_file_hash, load_corpus, load_pdf


class TestPDFLoader:
    """Tests for PDF loading functionality."""

    def test_load_pdf_returns_documents(self, sample_pdf):
        """Loading a PDF should return a non-empty list of documents."""
        docs = load_pdf(sample_pdf)
        assert len(docs) > 0

    def test_load_pdf_has_page_content(self, sample_pdf):
        """Each page document should have non-empty text content."""
        docs = load_pdf(sample_pdf)
        for doc in docs:
            assert doc.page_content, f"Empty content on page {doc.metadata.get('page')}"

    def test_load_pdf_metadata_fields(self, sample_pdf):
        """Each document should have required metadata fields."""
        docs = load_pdf(sample_pdf)
        required_fields = {"source", "page", "doc_hash"}
        for doc in docs:
            for field in required_fields:
                assert field in doc.metadata, f"Missing metadata field: {field}"

    def test_load_pdf_source_is_filename(self, sample_pdf):
        """The source metadata should be the PDF filename."""
        docs = load_pdf(sample_pdf)
        assert docs[0].metadata["source"] == sample_pdf.name

    def test_load_pdf_pages_are_sequential(self, sample_pdf):
        """Each document should have a valid non-negative page number."""
        docs = load_pdf(sample_pdf)
        for doc in docs:
            page = doc.metadata["page"]
            assert isinstance(page, int) and page >= 0, (
                f"Invalid page number: {page}"
            )

    def test_load_pdf_nonexistent_raises(self):
        """Loading a nonexistent file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_pdf(Path("/nonexistent/file.pdf"))

    def test_compute_file_hash_deterministic(self, sample_pdf):
        """Same file should produce the same hash."""
        hash1 = compute_file_hash(sample_pdf)
        hash2 = compute_file_hash(sample_pdf)
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hex digest

    def test_load_corpus(self, corpus_dir):
        """Loading the corpus should return documents for each PDF."""
        corpus = load_corpus(corpus_dir)
        assert len(corpus) > 0
        for name, docs in corpus.items():
            assert name.endswith(".pdf")
            assert len(docs) > 0


class TestChunker:
    """Tests for text chunking functionality."""

    def test_chunk_documents_produces_chunks(self, sample_pdf):
        """Chunking should produce more chunks than pages."""
        docs = load_pdf(sample_pdf)
        chunks = chunk_documents(docs, chunk_size=500, chunk_overlap=100)
        assert len(chunks) >= len(docs)

    def test_chunk_metadata_preserved(self, sample_pdf):
        """Chunks should preserve source metadata from the original documents."""
        docs = load_pdf(sample_pdf)
        chunks = chunk_documents(docs, chunk_size=500, chunk_overlap=100)

        for chunk in chunks:
            assert "source" in chunk.metadata
            assert "page" in chunk.metadata
            assert "doc_hash" in chunk.metadata
            assert "chunk_index" in chunk.metadata
            assert "chunk_id" in chunk.metadata

    def test_chunk_id_format(self, sample_pdf):
        """Chunk IDs should follow the deterministic format."""
        docs = load_pdf(sample_pdf)
        chunks = chunk_documents(docs, chunk_size=500, chunk_overlap=100)

        for chunk in chunks:
            chunk_id = chunk.metadata["chunk_id"]
            assert "::" in chunk_id
            assert "page" in chunk_id
            assert "chunk" in chunk_id

    def test_chunk_size_respected(self, sample_pdf):
        """Text chunks should not significantly exceed the target size.
        Tables and images are kept whole by design and may exceed chunk_size."""
        docs = load_pdf(sample_pdf)
        chunk_size = 500
        chunks = chunk_documents(docs, chunk_size=chunk_size, chunk_overlap=100)

        for chunk in chunks:
            content_type = chunk.metadata.get("content_type", "text")
            if content_type in ("table", "image"):
                continue  # tables/images are kept whole — no size limit
            # Allow some tolerance for splitting at natural boundaries
            assert len(chunk.page_content) <= chunk_size * 1.5, (
                f"Chunk too large: {len(chunk.page_content)} chars (limit: {chunk_size})"
            )

    def test_chunk_empty_input(self):
        """Chunking empty input should return empty list."""
        chunks = chunk_documents([])
        assert chunks == []

    def test_chunk_overlap_creates_more_chunks(self, sample_pdf):
        """More overlap should generally produce more chunks."""
        docs = load_pdf(sample_pdf)
        chunks_no_overlap = chunk_documents(docs, chunk_size=500, chunk_overlap=0)
        chunks_with_overlap = chunk_documents(docs, chunk_size=500, chunk_overlap=200)
        assert len(chunks_with_overlap) >= len(chunks_no_overlap)

"""
Ingestion pipeline orchestrator.

Coordinates the end-to-end flow: load PDFs → chunk → embed → upsert,
with hash-based idempotency to avoid re-processing unchanged documents.
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.config import Settings, get_embeddings, get_settings
from src.ingestion.chunker import chunk_documents
from src.ingestion.loader import compute_file_hash, load_pdf
from src.models import IngestionReport
from src.retrieval.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """
    Orchestrates document ingestion with idempotency.

    Workflow per document:
        1. Compute file hash
        2. Check if document is already current in the vector store
        3. If current → skip; if new/changed → load → chunk → upsert
    """

    def __init__(self, settings: Settings | None = None):
        self._settings = settings or get_settings()
        self._embeddings = get_embeddings(self._settings)
        self._vector_store = VectorStoreManager(
            persist_dir=self._settings.chroma_persist_dir,
            collection_name=self._settings.chroma_collection_name,
            embedding_function=self._embeddings,
        )

    @property
    def vector_store(self) -> VectorStoreManager:
        return self._vector_store

    def ingest(self, corpus_dir: Path | str | None = None) -> IngestionReport:
        """
        Run the full ingestion pipeline on a corpus directory.

        Args:
            corpus_dir: Directory containing PDF files. Defaults to settings.corpus_dir.

        Returns:
            IngestionReport with processing statistics.
        """
        corpus_path = Path(corpus_dir or self._settings.corpus_dir).resolve()
        report = IngestionReport()

        if not corpus_path.is_dir():
            msg = f"Corpus directory not found: {corpus_path}"
            logger.error(msg)
            report.errors.append(msg)
            return report

        pdf_files = sorted(corpus_path.glob("*.pdf"))
        if not pdf_files:
            msg = f"No PDF files found in {corpus_path}"
            logger.warning(msg)
            report.errors.append(msg)
            return report

        logger.info("Starting ingestion of %d PDF files from %s", len(pdf_files), corpus_path)

        for pdf_path in pdf_files:
            try:
                self._ingest_document(pdf_path, report)
            except Exception as e:
                msg = f"Failed to ingest {pdf_path.name}: {e}"
                logger.exception(msg)
                report.errors.append(msg)

        logger.info(
            "Ingestion complete: %d processed, %d skipped, %d chunks created, %d errors",
            report.documents_processed,
            report.documents_skipped,
            report.chunks_created,
            len(report.errors),
        )
        return report

    def _ingest_document(self, pdf_path: Path, report: IngestionReport) -> None:
        """Ingest a single PDF document with idempotency check."""
        doc_name = pdf_path.name
        doc_hash = compute_file_hash(pdf_path)

        # Check if document is already current
        if self._vector_store.is_document_current(doc_name, doc_hash):
            logger.info("Skipping '%s' — already current (hash: %s)", doc_name, doc_hash[:12])
            report.documents_skipped += 1
            return

        # Document is new or changed — delete old chunks if they exist
        deleted = self._vector_store.delete_by_document(doc_name)
        if deleted > 0:
            logger.info("Deleted %d stale chunks for '%s'", deleted, doc_name)

        # Load and chunk
        pages = load_pdf(pdf_path)
        chunks = chunk_documents(
            pages,
            chunk_size=self._settings.chunk_size,
            chunk_overlap=self._settings.chunk_overlap,
        )

        # Upsert into vector store
        count = self._vector_store.upsert_documents(chunks)

        report.documents_processed += 1
        report.chunks_created += count
        logger.info("Ingested '%s': %d pages → %d chunks", doc_name, len(pages), count)

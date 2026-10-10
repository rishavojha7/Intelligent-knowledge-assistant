"""
API route handlers for the Knowledge Assistant.

Implements REST endpoints for question answering, health checks,
document management, and ingestion triggers.
"""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, HTTPException

from src.agent.graph import run_query
from src.api.schemas import (
    AskRequest,
    AskResponse,
    DocumentInfo,
    DocumentsResponse,
    HealthResponse,
    IngestResponse,
    SourceReferenceResponse,
)
from src.config import get_embeddings, get_settings
from src.ingestion.pipeline import IngestionPipeline
from src.retrieval.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Knowledge Assistant"])


def _get_vector_store() -> VectorStoreManager:
    """Get a VectorStoreManager instance."""
    settings = get_settings()
    embeddings = get_embeddings(settings)
    return VectorStoreManager(
        persist_dir=settings.chroma_persist_dir,
        collection_name=settings.chroma_collection_name,
        embedding_function=embeddings,
    )


# ---------------------------------------------------------------------------
# POST /api/v1/ask
# ---------------------------------------------------------------------------

@router.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest) -> AskResponse:
    """
    Ask a question and get an answer from the knowledge base.

    The system automatically routes simple questions through a direct RAG chain
    and complex questions through an agentic workflow with multi-step reasoning.
    Override routing with the `use_agent` parameter.
    """
    start_time = time.time()
    question = request.question.strip()

    logger.info("Received question: '%s' (use_agent=%s)", question[:80], request.use_agent)

    try:
        result = run_query(question=question, use_agent=request.use_agent)

        elapsed = time.time() - start_time
        logger.info(
            "Query answered in %.2fs via %s (confidence: %s)",
            elapsed,
            result.workflow,
            result.confidence,
        )

        return AskResponse(
            answer=result.answer,
            sources=[
                SourceReferenceResponse(
                    document=s.document,
                    page=s.page,
                    chunk_id=s.chunk_id,
                    relevance_score=s.relevance_score,
                )
                for s in result.sources
            ],
            workflow=result.workflow,
            reasoning_steps=result.reasoning_steps,
            confidence=result.confidence,
        )

    except Exception:
        logger.exception("Error processing question")
        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing your question. Please try again.",
        )


# ---------------------------------------------------------------------------
# GET /api/v1/health
# ---------------------------------------------------------------------------

@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Check the health of the service and knowledge base status."""
    try:
        store = _get_vector_store()
        stats = store.get_collection_stats()
        return HealthResponse(
            status="healthy",
            documents_count=stats["documents_count"],
            total_chunks=stats["total_chunks"],
            documents=stats["documents"],
        )
    except Exception:
        logger.exception("Health check failed")
        return HealthResponse(status="unhealthy")


# ---------------------------------------------------------------------------
# POST /api/v1/ingest
# ---------------------------------------------------------------------------

@router.post("/ingest", response_model=IngestResponse)
async def trigger_ingestion() -> IngestResponse:
    """
    Trigger document ingestion from the corpus directory.

    Idempotent: unchanged documents are skipped automatically.
    """
    logger.info("Ingestion triggered via API")
    try:
        settings = get_settings()
        pipeline = IngestionPipeline(settings=settings)
        report = pipeline.ingest()

        status = "completed"
        if report.errors:
            status = "completed_with_errors"

        return IngestResponse(
            status=status,
            documents_processed=report.documents_processed,
            documents_skipped=report.documents_skipped,
            chunks_created=report.chunks_created,
            errors=report.errors,
        )
    except Exception:
        logger.exception("Ingestion failed")
        raise HTTPException(status_code=500, detail="Ingestion failed. Check server logs.")


# ---------------------------------------------------------------------------
# GET /api/v1/documents
# ---------------------------------------------------------------------------

@router.get("/documents", response_model=DocumentsResponse)
async def list_documents() -> DocumentsResponse:
    """List all ingested documents with their chunk counts."""
    try:
        store = _get_vector_store()
        stats = store.get_collection_stats()

        # Get per-document chunk counts
        documents = []
        collection = store._client.get_or_create_collection(
            name=get_settings().chroma_collection_name,
        )

        for doc_name in stats["documents"]:
            result = collection.get(where={"source": doc_name})
            chunk_count = len(result.get("ids", []))
            documents.append(DocumentInfo(name=doc_name, chunk_count=chunk_count))

        return DocumentsResponse(
            documents=documents,
            total_documents=len(documents),
            total_chunks=stats["total_chunks"],
        )
    except Exception:
        logger.exception("Failed to list documents")
        raise HTTPException(status_code=500, detail="Failed to retrieve document list.")

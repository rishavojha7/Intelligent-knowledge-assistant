"""
Pydantic request/response schemas for the REST API.

Defines the external contract for all API endpoints.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    """Request body for the /ask endpoint."""

    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The question to answer from the knowledge base.",
        examples=["What is RAG and how does it work?"],
    )
    use_agent: bool | None = Field(
        default=None,
        description=(
            "Force workflow selection. True = agentic, False = simple RAG, "
            "None = let the router decide automatically."
        ),
    )


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class SourceReferenceResponse(BaseModel):
    """A citation reference to a source document."""

    document: str = Field(description="Source document filename")
    page: int = Field(description="Page number (1-indexed)")
    chunk_id: str = Field(default="", description="Internal chunk identifier")
    relevance_score: float = Field(default=0.0, description="Retrieval relevance score (0-1)")


class AskResponse(BaseModel):
    """Response body for the /ask endpoint."""

    answer: str = Field(description="Generated answer grounded in the knowledge base")
    sources: list[SourceReferenceResponse] = Field(
        default_factory=list,
        description="Source citations for the answer",
    )
    workflow: str = Field(
        description="Workflow used: 'simple_rag' or 'agentic'",
        examples=["simple_rag", "agentic"],
    )
    reasoning_steps: list[str] = Field(
        default_factory=list,
        description="Step-by-step reasoning trace (populated for agentic workflow)",
    )
    confidence: str = Field(
        default="high",
        description="Confidence level: 'high', 'medium', or 'low'",
    )


class HealthResponse(BaseModel):
    """Response body for the /health endpoint."""

    status: str = Field(default="healthy", description="Service status")
    documents_count: int = Field(default=0, description="Number of ingested documents")
    total_chunks: int = Field(default=0, description="Total chunks in the vector store")
    documents: list[str] = Field(default_factory=list, description="List of ingested document names")


class IngestResponse(BaseModel):
    """Response body for the /ingest endpoint."""

    status: str = Field(description="Ingestion status")
    documents_processed: int = Field(default=0)
    documents_skipped: int = Field(default=0)
    chunks_created: int = Field(default=0)
    errors: list[str] = Field(default_factory=list)


class DocumentInfo(BaseModel):
    """Information about an ingested document."""

    name: str
    chunk_count: int = 0


class DocumentsResponse(BaseModel):
    """Response body for the /documents endpoint."""

    documents: list[DocumentInfo] = Field(default_factory=list)
    total_documents: int = Field(default=0)
    total_chunks: int = Field(default=0)


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str = Field(description="Error type")
    detail: str = Field(description="Human-readable error message")

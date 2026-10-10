"""
Shared domain models used across the application.

These models define the data contracts between ingestion, retrieval,
RAG chain, agent, and API layers.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ChunkMetadata:
    """Metadata attached to each text chunk during ingestion."""

    source: str  # Document filename, e.g. "rag_architecture_patterns.pdf"
    page: int  # 0-indexed page number from the PDF
    chunk_index: int  # Position of this chunk within the page/document
    chunk_id: str  # Deterministic ID: "{doc}::page{N}::chunk{M}"
    doc_hash: str  # SHA-256 of the source file for idempotency


@dataclass
class RetrievalResult:
    """A single retrieved chunk with its content and metadata."""

    content: str
    source: str
    page: int
    chunk_id: str
    relevance_score: float = 0.0


@dataclass
class SourceReference:
    """Citation reference returned to the user."""

    document: str
    page: int
    chunk_id: str
    relevance_score: float = 0.0


@dataclass
class QueryResult:
    """Complete result of a question-answering query."""

    answer: str
    sources: list[SourceReference] = field(default_factory=list)
    workflow: str = "simple_rag"  # "simple_rag" or "agentic"
    reasoning_steps: list[str] = field(default_factory=list)
    confidence: str = "high"  # "high", "medium", or "low"


@dataclass
class IngestionReport:
    """Summary of an ingestion run."""

    documents_processed: int = 0
    documents_skipped: int = 0
    chunks_created: int = 0
    errors: list[str] = field(default_factory=list)

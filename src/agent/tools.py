"""
Agent tools for the LangGraph agentic workflow.

Four tools that provide the agent with different retrieval strategies
for the knowledge base. Each tool returns formatted text with source
citations that the agent can use to compose its final answer.
"""

from __future__ import annotations

import logging
from typing import Optional

from langchain_core.tools import tool

from src.retrieval.retriever import Retriever

logger = logging.getLogger(__name__)

# Module-level retriever instance, set during initialization
_retriever: Retriever | None = None

AVAILABLE_DOCUMENTS = [
    "rag_architecture_patterns.pdf",
    "vector_database_comparison.pdf",
    "agentic_ai_frameworks.pdf",
]


def initialize_tools(retriever: Retriever) -> None:
    """Set the retriever instance used by all tools."""
    global _retriever
    _retriever = retriever


def _get_retriever() -> Retriever:
    if _retriever is None:
        raise RuntimeError("Tools not initialized. Call initialize_tools() first.")
    return _retriever


@tool
def search_knowledge_base(query: str, top_k: int = 5) -> str:
    """Search the entire knowledge base for information relevant to the query.

    Use this for general questions that may span multiple documents.

    Args:
        query: Natural language search query.
        top_k: Number of results to return (default: 5).
    """
    retriever = _get_retriever()
    results = retriever.search(query=query, top_k=top_k)

    if not results:
        return "No relevant information found in the knowledge base for this query."

    return retriever.format_results_as_context(results)


@tool
def search_by_document(query: str, document_name: str) -> str:
    """Search within a specific document for relevant information.

    Use this when you know which document contains the information you need.

    Available documents:
    - rag_architecture_patterns.pdf: RAG design, chunking, retrieval, evaluation
    - vector_database_comparison.pdf: FAISS, Pinecone, Weaviate, pgvector, Chroma
    - agentic_ai_frameworks.pdf: LangChain, LangGraph, CrewAI, AutoGen

    Args:
        query: Natural language search query.
        document_name: Exact filename of the document to search.
    """
    retriever = _get_retriever()

    # Validate document name
    if document_name not in AVAILABLE_DOCUMENTS:
        return (
            f"Document '{document_name}' not found. "
            f"Available documents: {', '.join(AVAILABLE_DOCUMENTS)}"
        )

    results = retriever.search_by_document(
        query=query,
        document_name=document_name,
        top_k=5,
    )

    if not results:
        return f"No relevant information found in '{document_name}' for this query."

    return retriever.format_results_as_context(results)


@tool
def compare_across_documents(
    topic: str,
    document_1: str = "",
    document_2: str = "",
) -> str:
    """Retrieve information about a topic from multiple documents for comparison.

    Use this when the question requires comparing concepts, technologies,
    or approaches discussed in different documents.

    Args:
        topic: The topic to compare across documents.
        document_1: First document to search (optional, searches all if empty).
        document_2: Second document to search (optional, searches all if empty).
    """
    retriever = _get_retriever()
    sections = []

    if document_1 and document_2:
        # Search specific documents
        for doc_name in [document_1, document_2]:
            if doc_name not in AVAILABLE_DOCUMENTS:
                sections.append(f"⚠️ Document '{doc_name}' not found.")
                continue
            results = retriever.search_by_document(query=topic, document_name=doc_name, top_k=3)
            if results:
                sections.append(f"## From {doc_name}:\n{retriever.format_results_as_context(results)}")
            else:
                sections.append(f"## From {doc_name}:\nNo relevant information found.")
    else:
        # Search all documents and group by source
        results = retriever.search(query=topic, top_k=10)
        if not results:
            return f"No relevant information found about '{topic}' in any document."

        by_source: dict[str, list] = {}
        for r in results:
            by_source.setdefault(r.source, []).append(r)

        for source, source_results in by_source.items():
            sections.append(
                f"## From {source}:\n{retriever.format_results_as_context(source_results)}"
            )

    return "\n\n" + "\n\n".join(sections)


@tool
def get_document_summary(document_name: str) -> str:
    """Get a high-level overview of what a specific document covers.

    Use this to understand the scope of a document before doing detailed searches.

    Available documents:
    - rag_architecture_patterns.pdf: RAG design, chunking, retrieval, evaluation
    - vector_database_comparison.pdf: FAISS, Pinecone, Weaviate, pgvector, Chroma
    - agentic_ai_frameworks.pdf: LangChain, LangGraph, CrewAI, AutoGen

    Args:
        document_name: Exact filename of the document.
    """
    retriever = _get_retriever()

    if document_name not in AVAILABLE_DOCUMENTS:
        return (
            f"Document '{document_name}' not found. "
            f"Available documents: {', '.join(AVAILABLE_DOCUMENTS)}"
        )

    results = retriever.get_document_chunks(document_name=document_name, top_k=3)

    if not results:
        return f"No content found for '{document_name}'."

    return f"Overview of {document_name}:\n\n{retriever.format_results_as_context(results)}"


def get_all_tools() -> list:
    """Return all agent tools as a list."""
    return [
        search_knowledge_base,
        search_by_document,
        compare_across_documents,
        get_document_summary,
    ]

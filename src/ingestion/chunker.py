"""
Content-aware text chunking with metadata preservation.

Handles three content types from the document loader:
    - text: Split normally with RecursiveCharacterTextSplitter
    - table: Keep whole — never split a table across chunks
    - image: Keep whole — each figure summary is one chunk

All chunks receive deterministic IDs and preserve source metadata.
"""

from __future__ import annotations

import logging

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


def chunk_documents(
    documents: list[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 200,
) -> list[Document]:
    """
    Split documents into chunks, respecting content type.

    - content_type="text" → split with RecursiveCharacterTextSplitter
    - content_type="table" → pass through as a single chunk (never split)
    - content_type="image" → pass through as a single chunk (never split)

    Each chunk receives:
        - All metadata from the source document
        - chunk_index: global position within the document's chunks
        - chunk_id: deterministic ID "{source}::page{N}::chunk{M}"

    Args:
        documents: Documents from the loader (with content_type metadata).
        chunk_size: Target chunk size in characters (for text only).
        chunk_overlap: Overlap between consecutive text chunks.

    Returns:
        List of chunked Documents with enriched metadata.
    """
    if not documents:
        logger.warning("No documents provided for chunking")
        return []

    text_docs = []
    passthrough_docs = []

    for doc in documents:
        content_type = doc.metadata.get("content_type", "text")
        if content_type in ("table", "image"):
            passthrough_docs.append(doc)
        else:
            text_docs.append(doc)

    # --- Chunk text documents ---
    text_chunks = []
    if text_docs:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n## ", "\n\n", "\n", ". ", " ", ""],
            is_separator_regex=False,
        )
        text_chunks = splitter.split_documents(text_docs)

    # --- Combine: text chunks + passthrough (tables, images) ---
    # Sort by page number to maintain document order
    all_chunks = text_chunks + passthrough_docs
    all_chunks.sort(key=lambda d: (
        d.metadata.get("source", ""),
        d.metadata.get("page", 0),
        # tables/images before text on the same page for better context
        0 if d.metadata.get("content_type") in ("table", "image") else 1,
    ))

    # --- Assign deterministic chunk IDs ---
    doc_chunk_counters: dict[str, int] = {}

    for chunk in all_chunks:
        source = chunk.metadata.get("source", "unknown")
        page = chunk.metadata.get("page", 0)
        content_type = chunk.metadata.get("content_type", "text")

        counter_key = source
        idx = doc_chunk_counters.get(counter_key, 0)
        doc_chunk_counters[counter_key] = idx + 1

        chunk.metadata["chunk_index"] = idx
        chunk.metadata["chunk_id"] = f"{source}::page{page}::chunk{idx}"

        # Ensure content_type is set (for pypdf fallback docs that may lack it)
        if "content_type" not in chunk.metadata:
            chunk.metadata["content_type"] = "text"

    # Log summary
    text_count = sum(1 for c in all_chunks if c.metadata.get("content_type") == "text")
    table_count = sum(1 for c in all_chunks if c.metadata.get("content_type") == "table")
    image_count = sum(1 for c in all_chunks if c.metadata.get("content_type") == "image")

    logger.info(
        "Chunked %d documents into %d chunks: %d text (size=%d, overlap=%d), "
        "%d tables (whole), %d images (whole)",
        len(documents),
        len(all_chunks),
        text_count,
        chunk_size,
        chunk_overlap,
        table_count,
        image_count,
    )
    return all_chunks

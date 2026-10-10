"""
ChromaDB vector store management.

Provides a wrapper around ChromaDB for storing and retrieving
document embeddings with metadata filtering and idempotent upsert.
"""

from __future__ import annotations

import logging
from pathlib import Path

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """
    Manages ChromaDB operations for the knowledge base.

    Supports:
        - Persistent storage with idempotent upsert
        - Document-level change detection via hashes
        - Metadata-filtered retrieval
        - Collection statistics
    """

    def __init__(
        self,
        persist_dir: str | Path,
        collection_name: str,
        embedding_function,
    ):
        self._persist_dir = str(Path(persist_dir).resolve())
        self._collection_name = collection_name
        self._embedding_function = embedding_function

        # Initialize persistent ChromaDB client
        self._client = chromadb.PersistentClient(path=self._persist_dir)
        logger.info("ChromaDB initialized at %s", self._persist_dir)

        # LangChain Chroma wrapper for retrieval
        self._langchain_store = Chroma(
            client=self._client,
            collection_name=self._collection_name,
            embedding_function=self._embedding_function,
        )

    @property
    def langchain_store(self) -> Chroma:
        """Access the underlying LangChain Chroma instance."""
        return self._langchain_store

    def get_collection_stats(self) -> dict:
        """Return collection statistics: document count, unique sources, etc."""
        try:
            collection = self._client.get_or_create_collection(
                name=self._collection_name,
            )
            count = collection.count()

            # Get unique source documents
            sources = set()
            if count > 0:
                result = collection.get(include=["metadatas"])
                for meta in result.get("metadatas", []):
                    if meta and "source" in meta:
                        sources.add(meta["source"])

            return {
                "total_chunks": count,
                "documents": sorted(sources),
                "documents_count": len(sources),
            }
        except Exception:
            logger.exception("Failed to get collection stats")
            return {"total_chunks": 0, "documents": [], "documents_count": 0}

    def is_document_current(self, doc_name: str, doc_hash: str) -> bool:
        """
        Check if a document is already ingested with the same hash.

        Returns True if the document exists in the store with a matching hash,
        meaning it does not need re-ingestion.
        """
        try:
            collection = self._client.get_or_create_collection(
                name=self._collection_name,
            )
            result = collection.get(
                where={"source": doc_name},
                include=["metadatas"],
                limit=1,
            )
            if result["metadatas"]:
                stored_hash = result["metadatas"][0].get("doc_hash", "")
                return stored_hash == doc_hash
        except Exception:
            logger.exception("Error checking document currency for %s", doc_name)
        return False

    def delete_by_document(self, doc_name: str) -> int:
        """Delete all chunks for a given document. Returns count deleted."""
        try:
            collection = self._client.get_or_create_collection(
                name=self._collection_name,
            )
            # Get IDs for this document
            result = collection.get(where={"source": doc_name})
            ids = result.get("ids", [])
            if ids:
                collection.delete(ids=ids)
                logger.info("Deleted %d chunks for document '%s'", len(ids), doc_name)
            return len(ids)
        except Exception:
            logger.exception("Error deleting chunks for %s", doc_name)
            return 0

    def upsert_documents(self, chunks: list[Document]) -> int:
        """
        Upsert document chunks into the vector store.

        Uses chunk_id from metadata as the deterministic ID.
        Returns the number of chunks upserted.
        """
        if not chunks:
            return 0

        ids = [chunk.metadata.get("chunk_id", f"chunk_{i}") for i, chunk in enumerate(chunks)]
        texts = [chunk.page_content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]

        # Sanitize metadata — ChromaDB only supports str, int, float, bool
        clean_metadatas = []
        for meta in metadatas:
            clean = {}
            for k, v in meta.items():
                if isinstance(v, (str, int, float, bool)):
                    clean[k] = v
                else:
                    clean[k] = str(v)
            clean_metadatas.append(clean)

        collection = self._client.get_or_create_collection(
            name=self._collection_name,
        )

        # Embed texts
        embeddings = self._embedding_function.embed_documents(texts)

        collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=clean_metadatas,
        )

        logger.info("Upserted %d chunks into collection '%s'", len(chunks), self._collection_name)
        return len(chunks)

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter_metadata: dict | None = None,
        score_threshold: float | None = None,
    ) -> list[tuple[Document, float]]:
        """
        Search for similar documents with relevance scores.

        Args:
            query: Search query text.
            k: Number of results to return.
            filter_metadata: Optional metadata filter (e.g., {"source": "doc.pdf"}).
            score_threshold: Minimum relevance score (0-1). None disables filtering.

        Returns:
            List of (Document, score) tuples, sorted by relevance.
        """
        kwargs = {"k": k}
        if filter_metadata:
            kwargs["filter"] = filter_metadata

        results = self._langchain_store.similarity_search_with_relevance_scores(
            query, **kwargs
        )

        if score_threshold is not None:
            results = [(doc, score) for doc, score in results if score >= score_threshold]

        return results

    def as_retriever(self, k: int = 5, filter_metadata: dict | None = None):
        """Return a LangChain retriever interface for use in chains."""
        search_kwargs = {"k": k}
        if filter_metadata:
            search_kwargs["filter"] = filter_metadata

        return self._langchain_store.as_retriever(
            search_type="similarity",
            search_kwargs=search_kwargs,
        )

    def get_all_documents(
        self,
        source_filter: str | None = None,
    ) -> list[tuple[str, dict]]:
        """
        Return all (text, metadata) pairs from the collection.

        Used to build the BM25 index for hybrid search.

        Args:
            source_filter: Optional document name to restrict results.

        Returns:
            List of (document_text, metadata_dict) tuples.
        """
        try:
            collection = self._client.get_or_create_collection(
                name=self._collection_name,
            )
            kwargs: dict = {"include": ["documents", "metadatas"]}
            if source_filter:
                kwargs["where"] = {"source": source_filter}

            result = collection.get(**kwargs)

            docs = result.get("documents", []) or []
            metas = result.get("metadatas", []) or []

            return list(zip(docs, metas))
        except Exception:
            logger.exception("Failed to get all documents")
            return []

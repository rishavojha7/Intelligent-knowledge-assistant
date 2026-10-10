"""
Retrieval logic with hybrid search (BM25 + dense) and cross-encoder re-ranking.

Wraps the VectorStoreManager to provide a clean retrieval interface
for the RAG chain and agent tools. Supports three modes:
    - dense: vector similarity search only (ChromaDB)
    - hybrid: BM25 sparse + dense embeddings fused via Reciprocal Rank Fusion
    - optional cross-encoder re-ranking on top of either mode
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict

from langchain_core.documents import Document

from src.models import RetrievalResult
from src.retrieval.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)


class Retriever:
    """
    High-level retrieval interface for the knowledge base.

    Provides methods for semantic search with score filtering,
    document-scoped search, and multi-document comparison retrieval.
    Supports hybrid BM25+dense search and cross-encoder re-ranking.
    """

    def __init__(
        self,
        vector_store: VectorStoreManager,
        default_top_k: int = 5,
        settings=None,
    ):
        self._store = vector_store
        self._default_top_k = default_top_k

        # Load settings for retrieval mode and re-ranking
        if settings is None:
            from src.config import get_settings
            settings = get_settings()

        self._retrieval_mode = getattr(settings, "retrieval_mode", "dense")
        self._bm25_weight = getattr(settings, "bm25_weight", 0.4)
        self._dense_weight = getattr(settings, "dense_weight", 0.6)
        self._reranker_enabled = getattr(settings, "reranker_enabled", False)
        self._reranker_model_name = getattr(settings, "reranker_model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        self._reranker_top_k = getattr(settings, "reranker_top_k", 5)

        # Lazy-initialized caches
        self._bm25_index = None
        self._bm25_corpus: list[tuple[str, dict]] = []
        self._reranker = None

    # ------------------------------------------------------------------
    # BM25 index management
    # ------------------------------------------------------------------

    def _ensure_bm25_index(self, source_filter: str | None = None) -> None:
        """Build or rebuild the BM25 index from stored documents."""
        if self._bm25_index is not None and source_filter is None:
            return  # Already built for full corpus

        from rank_bm25 import BM25Okapi

        all_docs = self._store.get_all_documents(source_filter=source_filter)
        if not all_docs:
            logger.warning("No documents found for BM25 index")
            self._bm25_index = None
            self._bm25_corpus = []
            return

        self._bm25_corpus = all_docs
        tokenized = [self._tokenize(text) for text, _ in all_docs]
        self._bm25_index = BM25Okapi(tokenized)

        logger.debug(
            "Built BM25 index: %d documents%s",
            len(all_docs),
            f" (source={source_filter})" if source_filter else "",
        )

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple whitespace tokenizer with lowercasing and punctuation removal."""
        text = text.lower()
        tokens = re.findall(r"\b\w+\b", text)
        return tokens

    def _bm25_search(
        self,
        query: str,
        k: int,
        source_filter: str | None = None,
    ) -> list[tuple[Document, float]]:
        """Run BM25 search and return (Document, normalized_score) tuples."""
        self._ensure_bm25_index(source_filter=source_filter)

        if self._bm25_index is None or not self._bm25_corpus:
            return []

        query_tokens = self._tokenize(query)
        scores = self._bm25_index.get_scores(query_tokens)

        # Normalize scores to 0-1 range
        max_score = max(scores) if len(scores) > 0 and max(scores) > 0 else 1.0
        normalized = scores / max_score

        # Get top-k indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

        results = []
        for idx in top_indices:
            if normalized[idx] <= 0:
                continue
            text, meta = self._bm25_corpus[idx]
            doc = Document(page_content=text, metadata=meta)
            results.append((doc, float(normalized[idx])))

        return results

    # ------------------------------------------------------------------
    # Reciprocal Rank Fusion
    # ------------------------------------------------------------------

    @staticmethod
    def reciprocal_rank_fusion(
        dense_results: list[tuple[Document, float]],
        bm25_results: list[tuple[Document, float]],
        dense_weight: float = 0.6,
        bm25_weight: float = 0.4,
        k: int = 10,
        rrf_k: int = 60,
    ) -> list[tuple[Document, float]]:
        """
        Fuse dense and BM25 results using Reciprocal Rank Fusion.

        RRF score for document d = Σ (weight / (rank + rrf_k))
        where rank is 1-indexed position in each result list.

        Args:
            dense_results: (Document, score) from dense vector search.
            bm25_results: (Document, score) from BM25 search.
            dense_weight: Weight for dense retrieval scores.
            bm25_weight: Weight for BM25 retrieval scores.
            k: Number of fused results to return.
            rrf_k: RRF constant (default 60, standard value from the RRF paper).

        Returns:
            Fused (Document, rrf_score) list sorted by score descending.
        """
        # Track by chunk_id to deduplicate
        doc_scores: dict[str, float] = defaultdict(float)
        doc_map: dict[str, Document] = {}

        for rank, (doc, _score) in enumerate(dense_results, start=1):
            chunk_id = doc.metadata.get("chunk_id", doc.page_content[:50])
            doc_scores[chunk_id] += dense_weight / (rank + rrf_k)
            doc_map[chunk_id] = doc

        for rank, (doc, _score) in enumerate(bm25_results, start=1):
            chunk_id = doc.metadata.get("chunk_id", doc.page_content[:50])
            doc_scores[chunk_id] += bm25_weight / (rank + rrf_k)
            if chunk_id not in doc_map:
                doc_map[chunk_id] = doc

        # Sort by fused score
        sorted_ids = sorted(doc_scores.keys(), key=lambda cid: doc_scores[cid], reverse=True)

        results = []
        for chunk_id in sorted_ids[:k]:
            doc = doc_map[chunk_id]
            # Normalize RRF score to approximate 0-1 range
            raw_score = doc_scores[chunk_id]
            max_possible = (dense_weight + bm25_weight) / (1 + rrf_k)
            normalized = min(raw_score / max_possible, 1.0) if max_possible > 0 else 0.0
            results.append((doc, normalized))

        return results

    # ------------------------------------------------------------------
    # Cross-encoder re-ranking
    # ------------------------------------------------------------------

    def _get_reranker(self):
        """Lazy-load the cross-encoder re-ranking model."""
        if self._reranker is None:
            try:
                from sentence_transformers import CrossEncoder

                self._reranker = CrossEncoder(
                    self._reranker_model_name,
                    local_files_only=True,
                )
                logger.info("Loaded cross-encoder re-ranker: %s", self._reranker_model_name)
            except Exception:
                logger.warning(
                    "Cross-encoder model '%s' not available locally — disabling re-ranking. "
                    "Download it first with: python -c \"from sentence_transformers import CrossEncoder; CrossEncoder('%s')\"",
                    self._reranker_model_name,
                    self._reranker_model_name,
                )
                self._reranker_enabled = False
                return None
        return self._reranker

    def _rerank(
        self,
        query: str,
        results: list[tuple[Document, float]],
        top_k: int,
    ) -> list[tuple[Document, float]]:
        """Re-rank results using a cross-encoder model."""
        reranker = self._get_reranker()
        if reranker is None or not results:
            return results[:top_k]

        # Score each (query, document) pair
        pairs = [(query, doc.page_content) for doc, _ in results]
        scores = reranker.predict(pairs)

        # Normalize scores to 0-1 range (cross-encoder outputs can be arbitrary)
        min_s, max_s = float(min(scores)), float(max(scores))
        score_range = max_s - min_s if max_s > min_s else 1.0

        scored = []
        for (doc, _original_score), ce_score in zip(results, scores):
            normalized = (float(ce_score) - min_s) / score_range
            scored.append((doc, round(normalized, 4)))

        # Sort by re-ranked score descending
        scored.sort(key=lambda x: x[1], reverse=True)

        logger.debug("Re-ranked %d results → top %d", len(scored), top_k)
        return scored[:top_k]

    # ------------------------------------------------------------------
    # Main search interface
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int | None = None,
        score_threshold: float | None = None,
        source_filter: str | None = None,
    ) -> list[RetrievalResult]:
        """
        Search the knowledge base with optional hybrid retrieval and re-ranking.

        Flow:
            1. Dense vector search (always)
            2. BM25 search + Reciprocal Rank Fusion (if hybrid mode)
            3. Cross-encoder re-ranking (if enabled)
            4. Score threshold filtering
            5. Convert to RetrievalResult

        Args:
            query: Natural language search query.
            top_k: Number of results to return.
            score_threshold: Minimum relevance score (0-1).
            source_filter: Optional document name to restrict search.

        Returns:
            List of RetrievalResult sorted by relevance.
        """
        k = top_k or self._default_top_k
        filter_meta = {"source": source_filter} if source_filter else None

        # Over-fetch when re-ranking or fusing — give the re-ranker a larger pool
        fetch_k = k * 2 if (self._reranker_enabled or self._retrieval_mode == "hybrid") else k

        # --- Step 1: Dense vector search ---
        dense_results = self._store.similarity_search(
            query=query,
            k=fetch_k,
            filter_metadata=filter_meta,
        )

        # --- Step 2: Hybrid fusion (if enabled) ---
        if self._retrieval_mode == "hybrid":
            bm25_results = self._bm25_search(
                query=query,
                k=fetch_k,
                source_filter=source_filter,
            )
            fused_results = self.reciprocal_rank_fusion(
                dense_results=dense_results,
                bm25_results=bm25_results,
                dense_weight=self._dense_weight,
                bm25_weight=self._bm25_weight,
                k=fetch_k,
            )
        else:
            fused_results = dense_results

        # --- Step 3: Cross-encoder re-ranking (if enabled) ---
        if self._reranker_enabled:
            final_results = self._rerank(query, fused_results, top_k=k)
        else:
            final_results = fused_results[:k]

        # --- Step 4: Score threshold filtering ---
        if score_threshold is not None:
            final_results = [(doc, score) for doc, score in final_results if score >= score_threshold]

        # --- Step 5: Convert to RetrievalResult ---
        retrieval_results = []
        for doc, score in final_results:
            retrieval_results.append(
                RetrievalResult(
                    content=doc.page_content,
                    source=doc.metadata.get("source", "unknown"),
                    page=doc.metadata.get("page", 0),
                    chunk_id=doc.metadata.get("chunk_id", ""),
                    relevance_score=round(score, 4),
                )
            )

        logger.debug(
            "Search for '%s' returned %d results (mode=%s, reranker=%s, k=%d, threshold=%s)",
            query[:50],
            len(retrieval_results),
            self._retrieval_mode,
            self._reranker_enabled,
            k,
            score_threshold,
        )
        return retrieval_results

    def search_by_document(
        self,
        query: str,
        document_name: str,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """Search within a specific document."""
        return self.search(
            query=query,
            top_k=top_k,
            source_filter=document_name,
        )

    def get_document_chunks(
        self,
        document_name: str,
        top_k: int = 3,
    ) -> list[RetrievalResult]:
        """
        Retrieve the first chunks of a document for summarization.

        Uses a generic query to get the most representative chunks.
        """
        return self.search(
            query=f"overview summary introduction of {document_name}",
            top_k=top_k,
            source_filter=document_name,
        )

    def format_results_as_context(self, results: list[RetrievalResult]) -> str:
        """Format retrieval results as a numbered context string for LLM prompts."""
        if not results:
            return "No relevant information found in the knowledge base."

        parts = []
        for i, r in enumerate(results, 1):
            parts.append(
                f"[Source {i}: {r.source}, Page {r.page + 1}]\n{r.content}"
            )
        return "\n\n---\n\n".join(parts)

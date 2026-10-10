"""
Simple RAG chain using LangChain Expression Language (LCEL).

Implements: retriever → format context → prompt → LLM → parse response.
Returns structured QueryResult with answer, sources, and confidence.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from src.models import QueryResult, SourceReference
from src.rag.prompts import RAG_SYSTEM_PROMPT, RAG_USER_PROMPT
from src.retrieval.retriever import Retriever

logger = logging.getLogger(__name__)


def build_rag_chain(retriever: Retriever, llm: Any) -> Any:
    """
    Build an LCEL RAG chain that retrieves, formats, prompts, and parses.

    This is used for simple single-retrieval questions.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", RAG_SYSTEM_PROMPT),
        ("human", RAG_USER_PROMPT),
    ])

    chain = prompt | llm | StrOutputParser()
    return chain


def run_simple_rag(
    question: str,
    retriever: Retriever,
    llm: Any,
    top_k: int = 5,
    score_threshold: float = 0.3,
) -> QueryResult:
    """
    Execute a simple RAG query: retrieve → generate → return with sources.

    Args:
        question: User's question.
        retriever: Configured Retriever instance.
        llm: Language model instance.
        top_k: Number of chunks to retrieve.
        score_threshold: Minimum relevance score.

    Returns:
        QueryResult with answer, sources, and confidence.
    """
    logger.info("Simple RAG query: '%s'", question[:80])

    # Step 1: Retrieve relevant chunks
    results = retriever.search(
        query=question,
        top_k=top_k,
        score_threshold=score_threshold,
    )

    # Step 2: Handle no results
    if not results:
        logger.info("No relevant chunks found for query")
        return QueryResult(
            answer="I don't have enough information in the available documents to answer this question. The knowledge base does not contain relevant content for this query.",
            sources=[],
            workflow="simple_rag",
            reasoning_steps=["Retrieved 0 relevant chunks above threshold"],
            confidence="low",
        )

    # Step 3: Format context with source citations
    context = retriever.format_results_as_context(results)

    # Step 4: Build and invoke chain
    chain = build_rag_chain(retriever, llm)
    answer = chain.invoke({"context": context, "question": question})

    # Step 5: Build source references
    sources = [
        SourceReference(
            document=r.source,
            page=r.page + 1,  # Convert to 1-indexed for user display
            chunk_id=r.chunk_id,
            relevance_score=r.relevance_score,
        )
        for r in results
    ]

    # Step 6: Assess confidence based on retrieval scores
    avg_score = sum(r.relevance_score for r in results) / len(results)
    if avg_score >= 0.6:
        confidence = "high"
    elif avg_score >= 0.4:
        confidence = "medium"
    else:
        confidence = "low"

    # Check if the LLM itself indicated uncertainty
    uncertainty_phrases = [
        "don't have enough information",
        "not enough information",
        "cannot answer",
        "no relevant information",
        "outside the scope",
    ]
    if any(phrase in answer.lower() for phrase in uncertainty_phrases):
        confidence = "low"

    return QueryResult(
        answer=answer,
        sources=sources,
        workflow="simple_rag",
        reasoning_steps=[
            f"Retrieved {len(results)} relevant chunks",
            f"Average relevance score: {avg_score:.3f}",
            "Generated answer from context with citations",
        ],
        confidence=confidence,
    )

"""
Tests for the RAG chain.

Covers prompt formatting, chain assembly, and the simple RAG flow.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.rag.chain import run_simple_rag
from src.rag.prompts import RAG_SYSTEM_PROMPT, RAG_USER_PROMPT


class TestPrompts:
    """Tests for prompt template content."""

    def test_rag_system_prompt_contains_instructions(self):
        """System prompt should contain key instructions."""
        assert "provided context" in RAG_SYSTEM_PROMPT.lower()
        assert "cite" in RAG_SYSTEM_PROMPT.lower()

    def test_rag_user_prompt_has_placeholders(self):
        """User prompt should have context and question placeholders."""
        assert "{context}" in RAG_USER_PROMPT
        assert "{question}" in RAG_USER_PROMPT

    def test_rag_system_prompt_instructs_refusal(self):
        """System prompt should instruct graceful refusal."""
        lower = RAG_SYSTEM_PROMPT.lower()
        assert "don't have enough information" in lower or "do not" in lower


class TestSimpleRAG:
    """Tests for the simple RAG chain execution."""

    def test_no_results_returns_low_confidence(self):
        """When retrieval finds nothing, result should have low confidence."""
        mock_retriever = MagicMock()
        mock_retriever.search.return_value = []

        result = run_simple_rag(
            question="What is something not in the corpus?",
            retriever=mock_retriever,
            llm=MagicMock(),
        )

        assert result.confidence == "low"
        assert result.workflow == "simple_rag"
        assert "don't have enough information" in result.answer.lower() or \
               "does not contain" in result.answer.lower()

    def test_result_has_correct_structure(self):
        """Result should have all required fields."""
        mock_retriever = MagicMock()
        mock_retriever.search.return_value = []

        result = run_simple_rag(
            question="test question",
            retriever=mock_retriever,
            llm=MagicMock(),
        )

        assert hasattr(result, "answer")
        assert hasattr(result, "sources")
        assert hasattr(result, "workflow")
        assert hasattr(result, "reasoning_steps")
        assert hasattr(result, "confidence")
        assert isinstance(result.sources, list)
        assert isinstance(result.reasoning_steps, list)

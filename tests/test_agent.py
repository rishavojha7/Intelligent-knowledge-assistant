"""
Tests for the agent layer.

Covers router classification, tool definitions, and graph compilation.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.agent.router import RouteDecision
from src.agent.tools import AVAILABLE_DOCUMENTS, get_all_tools


class TestRouter:
    """Tests for the question router."""

    def test_route_decision_model(self):
        """RouteDecision should accept valid values."""
        decision = RouteDecision(route="simple_rag", reasoning="Simple factual question")
        assert decision.route == "simple_rag"

        decision = RouteDecision(route="agentic", reasoning="Multi-step comparison")
        assert decision.route == "agentic"

    def test_route_decision_rejects_invalid(self):
        """RouteDecision should reject invalid route values."""
        with pytest.raises(Exception):
            RouteDecision(route="invalid", reasoning="test")


class TestTools:
    """Tests for agent tool definitions."""

    def test_all_tools_registered(self):
        """get_all_tools should return all 4 tools."""
        tools = get_all_tools()
        assert len(tools) == 4

    def test_tool_names(self):
        """Tools should have the expected names."""
        tools = get_all_tools()
        names = {t.name for t in tools}
        expected = {
            "search_knowledge_base",
            "search_by_document",
            "compare_across_documents",
            "get_document_summary",
        }
        assert names == expected

    def test_tools_have_descriptions(self):
        """Each tool should have a non-empty description."""
        tools = get_all_tools()
        for tool in tools:
            assert tool.description, f"Tool {tool.name} has no description"

    def test_available_documents_list(self):
        """Available documents list should contain expected PDFs."""
        assert "rag_architecture_patterns.pdf" in AVAILABLE_DOCUMENTS
        assert "vector_database_comparison.pdf" in AVAILABLE_DOCUMENTS
        assert "agentic_ai_frameworks.pdf" in AVAILABLE_DOCUMENTS


class TestGraph:
    """Tests for the LangGraph workflow graph."""

    def test_graph_compiles(self):
        """The graph should compile without errors."""
        from src.agent.tools import initialize_tools

        # Initialize tools with a mock retriever
        mock_retriever = MagicMock()
        initialize_tools(mock_retriever)

        from src.agent.graph import build_graph

        graph = build_graph()
        assert graph is not None

    def test_agent_state_structure(self):
        """AgentState should have expected keys."""
        from src.agent.graph import AgentState

        # TypedDict should define expected keys
        expected_keys = {
            "messages", "question", "route", "route_reasoning",
            "iteration_count", "reasoning_steps", "final_answer",
            "sources", "confidence",
        }
        assert expected_keys == set(AgentState.__annotations__.keys())

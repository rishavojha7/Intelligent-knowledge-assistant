"""
Question router — classifies questions as simple_rag or agentic.

Uses an LLM with structured output to determine the appropriate
workflow for each incoming question.
"""

from __future__ import annotations

import logging
from typing import Literal

from pydantic import BaseModel, Field

from src.rag.prompts import ROUTER_SYSTEM_PROMPT, ROUTER_USER_PROMPT

logger = logging.getLogger(__name__)


class RouteDecision(BaseModel):
    """Structured classification of a user question."""

    route: Literal["simple_rag", "agentic"] = Field(
        description="The workflow to use: 'simple_rag' for straightforward questions, "
        "'agentic' for multi-step or comparison questions."
    )
    reasoning: str = Field(
        description="Brief explanation of why this route was chosen."
    )


def classify_question(question: str, llm) -> RouteDecision:
    """
    Classify a question to determine the appropriate workflow.

    Args:
        question: The user's question.
        llm: ChatOpenAI (or compatible) LLM instance.

    Returns:
        RouteDecision with route and reasoning.
    """
    try:
        structured_llm = llm.with_structured_output(RouteDecision)

        from langchain_core.prompts import ChatPromptTemplate

        prompt = ChatPromptTemplate.from_messages([
            ("system", ROUTER_SYSTEM_PROMPT),
            ("human", ROUTER_USER_PROMPT),
        ])

        chain = prompt | structured_llm
        decision = chain.invoke({"question": question})

        logger.info(
            "Router decision for '%s': %s (%s)",
            question[:50],
            decision.route,
            decision.reasoning[:80],
        )
        return decision

    except Exception:
        # Default to agentic on classification failure — safer, more thorough
        logger.exception("Router classification failed, defaulting to agentic")
        return RouteDecision(
            route="agentic",
            reasoning="Classification failed; defaulting to agentic for safety.",
        )

"""
LangGraph agentic workflow for multi-step question answering.

Implements a stateful graph with:
    - Router node: classifies question → simple_rag or agentic
    - Simple RAG node: single-retrieval chain
    - Agent node: ReAct-style agent with tools
    - Tool node: executes tool calls
    - Response formatting: produces structured QueryResult

The graph handles iteration limits, error recovery, and state management.
"""

from __future__ import annotations

import logging
from typing import Annotated, Any, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from src.agent.router import RouteDecision, classify_question
from src.agent.tools import get_all_tools, initialize_tools
from src.config import Settings, get_embeddings, get_llm, get_settings
from src.models import QueryResult, SourceReference
from src.rag.chain import run_simple_rag
from src.rag.prompts import AGENT_SYSTEM_PROMPT
from src.retrieval.retriever import Retriever
from src.retrieval.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)

MAX_AGENT_ITERATIONS = 5


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    """State flowing through the LangGraph workflow."""

    messages: Annotated[list, add_messages]
    question: str
    route: str  # "simple_rag" or "agentic"
    route_reasoning: str
    iteration_count: int
    reasoning_steps: list[str]
    final_answer: str
    sources: list[dict]
    confidence: str


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------

def router_node(state: AgentState) -> dict:
    """Classify the question and decide the workflow."""
    question = state["question"]
    llm = get_llm()

    decision: RouteDecision = classify_question(question, llm)

    return {
        "route": decision.route,
        "route_reasoning": decision.reasoning,
        "reasoning_steps": [f"Router: classified as '{decision.route}' — {decision.reasoning}"],
    }


def simple_rag_node(state: AgentState) -> dict:
    """Execute the simple RAG chain."""
    question = state["question"]
    settings = get_settings()

    # Build retriever
    embeddings = get_embeddings(settings)
    store = VectorStoreManager(
        persist_dir=settings.chroma_persist_dir,
        collection_name=settings.chroma_collection_name,
        embedding_function=embeddings,
    )
    retriever = Retriever(store, default_top_k=settings.retrieval_top_k, settings=settings)
    llm = get_llm()

    result = run_simple_rag(
        question=question,
        retriever=retriever,
        llm=llm,
        top_k=settings.retrieval_top_k,
        score_threshold=settings.similarity_threshold,
    )

    return {
        "final_answer": result.answer,
        "sources": [
            {
                "document": s.document,
                "page": s.page,
                "chunk_id": s.chunk_id,
                "relevance_score": s.relevance_score,
            }
            for s in result.sources
        ],
        "confidence": result.confidence,
        "reasoning_steps": state.get("reasoning_steps", []) + result.reasoning_steps,
    }


def agent_node(state: AgentState) -> dict:
    """Invoke the LLM with tools bound for agentic reasoning."""
    llm = get_llm()
    tools = get_all_tools()
    llm_with_tools = llm.bind_tools(tools)

    messages = state.get("messages", [])
    if not messages:
        # First invocation — set up the system message and question
        messages = [
            SystemMessage(content=AGENT_SYSTEM_PROMPT),
            HumanMessage(content=state["question"]),
        ]

    response = llm_with_tools.invoke(messages)
    iteration = state.get("iteration_count", 0) + 1

    step_description = f"Agent iteration {iteration}"
    if response.tool_calls:
        tool_names = [tc["name"] for tc in response.tool_calls]
        step_description += f": calling {', '.join(tool_names)}"
    else:
        step_description += ": generating final response"

    return {
        "messages": [response],
        "iteration_count": iteration,
        "reasoning_steps": state.get("reasoning_steps", []) + [step_description],
    }


def format_agent_response(state: AgentState) -> dict:
    """Extract the final answer and sources from the agent's messages."""
    messages = state.get("messages", [])

    # Find the last AI message (the final answer)
    final_text = ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
            final_text = msg.content
            break

    if not final_text:
        final_text = "I was unable to generate a complete answer. Please try rephrasing your question."

    # Extract source references from the answer text
    # Parse [DocName, Page N] patterns
    import re

    source_pattern = r"\[([^,\]]+\.pdf),\s*Page\s*(\d+)\]"
    found_sources = re.findall(source_pattern, final_text)

    sources = []
    seen = set()
    for doc, page in found_sources:
        key = (doc, int(page))
        if key not in seen:
            seen.add(key)
            sources.append({
                "document": doc,
                "page": int(page),
                "chunk_id": "",
                "relevance_score": 0.0,
            })

    # Assess confidence
    uncertainty_phrases = [
        "don't have enough information",
        "not enough information",
        "cannot answer",
        "no relevant information",
        "outside the scope",
    ]
    confidence = "high"
    if any(phrase in final_text.lower() for phrase in uncertainty_phrases):
        confidence = "low"
    elif not sources:
        confidence = "medium"

    return {
        "final_answer": final_text,
        "sources": sources,
        "confidence": confidence,
    }


# ---------------------------------------------------------------------------
# Routing edges
# ---------------------------------------------------------------------------

def route_after_classification(state: AgentState) -> Literal["simple_rag_node", "agent_node"]:
    """Route based on the classifier's decision."""
    if state.get("route") == "simple_rag":
        return "simple_rag_node"
    return "agent_node"


def should_continue_agent(state: AgentState) -> Literal["tool_node", "format_agent_response"]:
    """Decide whether the agent should keep iterating or finish."""
    messages = state.get("messages", [])
    iteration = state.get("iteration_count", 0)

    # Check iteration limit
    if iteration >= MAX_AGENT_ITERATIONS:
        logger.warning("Agent hit max iterations (%d), forcing completion", MAX_AGENT_ITERATIONS)
        return "format_agent_response"

    # Check if the last message has tool calls
    if messages:
        last_message = messages[-1]
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "tool_node"

    return "format_agent_response"


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    """Build and compile the LangGraph workflow."""
    tools = get_all_tools()
    tool_node = ToolNode(tools)

    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("router_node", router_node)
    graph.add_node("simple_rag_node", simple_rag_node)
    graph.add_node("agent_node", agent_node)
    graph.add_node("tool_node", tool_node)
    graph.add_node("format_agent_response", format_agent_response)

    # Add edges
    graph.add_edge(START, "router_node")
    graph.add_conditional_edges(
        "router_node",
        route_after_classification,
        {
            "simple_rag_node": "simple_rag_node",
            "agent_node": "agent_node",
        },
    )
    graph.add_edge("simple_rag_node", END)
    graph.add_conditional_edges(
        "agent_node",
        should_continue_agent,
        {
            "tool_node": "tool_node",
            "format_agent_response": "format_agent_response",
        },
    )
    graph.add_edge("tool_node", "agent_node")
    graph.add_edge("format_agent_response", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

_compiled_graph = None


def _get_graph():
    """Lazy-initialize and cache the compiled graph."""
    global _compiled_graph
    if _compiled_graph is None:
        settings = get_settings()
        embeddings = get_embeddings(settings)
        store = VectorStoreManager(
            persist_dir=settings.chroma_persist_dir,
            collection_name=settings.chroma_collection_name,
            embedding_function=embeddings,
        )
        retriever = Retriever(store, default_top_k=settings.retrieval_top_k, settings=settings)
        initialize_tools(retriever)
        _compiled_graph = build_graph()
    return _compiled_graph


def run_query(question: str, use_agent: bool | None = None) -> QueryResult:
    """
    Main entry point for answering questions.

    Args:
        question: User's question.
        use_agent: If True, force agentic workflow. If False, force simple RAG.
                   If None, let the router decide.

    Returns:
        QueryResult with answer, sources, workflow, and metadata.
    """
    logger.info("Processing query: '%s' (use_agent=%s)", question[:80], use_agent)

    graph = _get_graph()

    initial_state: AgentState = {
        "messages": [],
        "question": question,
        "route": "",
        "route_reasoning": "",
        "iteration_count": 0,
        "reasoning_steps": [],
        "final_answer": "",
        "sources": [],
        "confidence": "high",
    }

    # Override routing if explicitly requested
    if use_agent is True:
        initial_state["route"] = "agentic"
        initial_state["route_reasoning"] = "User requested agentic workflow"
        initial_state["reasoning_steps"] = ["User forced agentic workflow"]
        # Skip router, go directly to agent
        settings = get_settings()
        embeddings = get_embeddings(settings)
        store = VectorStoreManager(
            persist_dir=settings.chroma_persist_dir,
            collection_name=settings.chroma_collection_name,
            embedding_function=embeddings,
        )
        retriever = Retriever(store, default_top_k=settings.retrieval_top_k, settings=settings)
        initialize_tools(retriever)
        return _run_agentic(question, initial_state)

    if use_agent is False:
        settings = get_settings()
        embeddings = get_embeddings(settings)
        store = VectorStoreManager(
            persist_dir=settings.chroma_persist_dir,
            collection_name=settings.chroma_collection_name,
            embedding_function=embeddings,
        )
        retriever = Retriever(store, default_top_k=settings.retrieval_top_k, settings=settings)
        llm = get_llm()
        result = run_simple_rag(
            question=question,
            retriever=retriever,
            llm=llm,
            top_k=settings.retrieval_top_k,
            score_threshold=settings.similarity_threshold,
        )
        result.reasoning_steps.insert(0, "User forced simple RAG workflow")
        return result

    # Let the graph handle routing
    final_state = graph.invoke(initial_state)

    return QueryResult(
        answer=final_state.get("final_answer", "Unable to generate an answer."),
        sources=[
            SourceReference(**s) if isinstance(s, dict) else s
            for s in final_state.get("sources", [])
        ],
        workflow=final_state.get("route", "unknown"),
        reasoning_steps=final_state.get("reasoning_steps", []),
        confidence=final_state.get("confidence", "medium"),
    )


def _run_agentic(question: str, state: AgentState) -> QueryResult:
    """Run the agentic workflow directly, bypassing the router."""
    from langchain_core.messages import HumanMessage, SystemMessage

    tools = get_all_tools()
    llm = get_llm()
    llm_with_tools = llm.bind_tools(tools)
    tool_node = ToolNode(tools)

    messages = [
        SystemMessage(content=AGENT_SYSTEM_PROMPT),
        HumanMessage(content=question),
    ]

    reasoning_steps = state.get("reasoning_steps", [])
    iteration = 0

    while iteration < MAX_AGENT_ITERATIONS:
        iteration += 1
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        if response.tool_calls:
            tool_names = [tc["name"] for tc in response.tool_calls]
            reasoning_steps.append(f"Agent iteration {iteration}: calling {', '.join(tool_names)}")

            # Execute tools
            tool_results = tool_node.invoke({"messages": messages})
            tool_messages = tool_results.get("messages", [])
            messages.extend(tool_messages)
        else:
            reasoning_steps.append(f"Agent iteration {iteration}: generating final response")
            break

    # Extract final answer
    final_text = ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content and not getattr(msg, "tool_calls", None):
            final_text = msg.content
            break

    if not final_text:
        final_text = "I was unable to generate a complete answer."

    # Parse sources from answer
    import re

    source_pattern = r"\[([^,\]]+\.pdf),\s*Page\s*(\d+)\]"
    found_sources = re.findall(source_pattern, final_text)

    sources = []
    seen = set()
    for doc, page in found_sources:
        key = (doc, int(page))
        if key not in seen:
            seen.add(key)
            sources.append(SourceReference(document=doc, page=int(page), chunk_id="", relevance_score=0.0))

    confidence = "high"
    uncertainty_phrases = ["don't have enough information", "cannot answer", "no relevant information"]
    if any(p in final_text.lower() for p in uncertainty_phrases):
        confidence = "low"
    elif not sources:
        confidence = "medium"

    return QueryResult(
        answer=final_text,
        sources=sources,
        workflow="agentic",
        reasoning_steps=reasoning_steps,
        confidence=confidence,
    )

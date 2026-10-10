"""
Streamlit UI for the Intelligent Knowledge Assistant.

Provides a lightweight web interface for asking questions and
viewing answers with source citations.

Usage:
    streamlit run ui/app.py
"""

from __future__ import annotations

import os
import sys

import httpx
import streamlit as st

# API configuration
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
API_ASK_URL = f"{API_BASE_URL}/api/v1/ask"
API_HEALTH_URL = f"{API_BASE_URL}/api/v1/health"
API_DOCUMENTS_URL = f"{API_BASE_URL}/api/v1/documents"

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Knowledge Assistant",
    page_icon="🧠",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🧠 Knowledge Assistant")
    st.markdown("---")

    # Health check
    st.subheader("📊 System Status")
    try:
        health = httpx.get(API_HEALTH_URL, timeout=5.0).json()
        status = health.get("status", "unknown")
        if status == "healthy":
            st.success(f"✅ Status: {status}")
        else:
            st.error(f"❌ Status: {status}")
        st.metric("Documents", health.get("documents_count", 0))
        st.metric("Total Chunks", health.get("total_chunks", 0))

        docs = health.get("documents", [])
        if docs:
            st.markdown("**Ingested Documents:**")
            for doc in docs:
                st.markdown(f"- 📄 {doc}")
    except Exception:
        st.error("⚠️ Cannot connect to API. Is the server running?")

    st.markdown("---")

    # Workflow toggle
    st.subheader("⚙️ Settings")
    workflow_mode = st.radio(
        "Workflow Mode",
        ["Auto (Router decides)", "Force Simple RAG", "Force Agentic"],
        index=0,
        help="Auto lets the system decide the best workflow for each question.",
    )

    st.markdown("---")
    st.markdown(
        "**About:** This assistant answers questions from a knowledge base "
        "of technical documents about RAG, vector databases, and agentic AI."
    )

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------

st.title("💬 Ask the Knowledge Assistant")
st.markdown(
    "Ask questions about RAG architecture, vector databases, or agentic AI frameworks. "
    "The system will retrieve relevant information and generate a grounded answer with citations."
)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("metadata"):
            meta = msg["metadata"]
            cols = st.columns(3)
            with cols[0]:
                st.caption(f"🔄 Workflow: {meta.get('workflow', 'N/A')}")
            with cols[1]:
                st.caption(f"📊 Confidence: {meta.get('confidence', 'N/A')}")
            with cols[2]:
                st.caption(f"📚 Sources: {meta.get('source_count', 0)}")

# Question input
question = st.chat_input("Ask a question about the knowledge base...")

if question:
    # Display user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Determine use_agent parameter
    use_agent = None
    if workflow_mode == "Force Simple RAG":
        use_agent = False
    elif workflow_mode == "Force Agentic":
        use_agent = True

    # Call API
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = httpx.post(
                    API_ASK_URL,
                    json={"question": question, "use_agent": use_agent},
                    timeout=120.0,
                )
                response.raise_for_status()
                data = response.json()

                # Display answer
                st.markdown(data["answer"])

                # Metadata bar
                cols = st.columns(3)
                with cols[0]:
                    workflow_icon = "🤖" if data["workflow"] == "agentic" else "📖"
                    st.caption(f"{workflow_icon} Workflow: {data['workflow']}")
                with cols[1]:
                    conf = data.get("confidence", "N/A")
                    conf_icon = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(conf, "⚪")
                    st.caption(f"{conf_icon} Confidence: {conf}")
                with cols[2]:
                    st.caption(f"📚 Sources: {len(data.get('sources', []))}")

                # Sources expander
                sources = data.get("sources", [])
                if sources:
                    with st.expander("📄 View Sources", expanded=False):
                        for i, src in enumerate(sources, 1):
                            score = src.get("relevance_score", 0)
                            score_str = f" (score: {score:.3f})" if score > 0 else ""
                            st.markdown(
                                f"**{i}.** {src['document']}, Page {src['page']}{score_str}"
                            )

                # Reasoning steps expander (for agentic workflow)
                steps = data.get("reasoning_steps", [])
                if steps:
                    with st.expander("🔍 Reasoning Steps", expanded=False):
                        for i, step in enumerate(steps, 1):
                            st.markdown(f"{i}. {step}")

                # Save to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": data["answer"],
                    "metadata": {
                        "workflow": data["workflow"],
                        "confidence": data.get("confidence", "N/A"),
                        "source_count": len(sources),
                    },
                })

            except httpx.ConnectError:
                st.error(
                    "❌ Cannot connect to the API server. "
                    "Make sure the FastAPI server is running on "
                    f"{API_BASE_URL}"
                )
            except httpx.HTTPStatusError as e:
                st.error(f"❌ API error: {e.response.status_code} — {e.response.text}")
            except Exception as e:
                st.error(f"❌ Unexpected error: {e}")

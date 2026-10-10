
# Intelligent Knowledge Assistant — Implementation Plan

## Context

Build a production-oriented RAG + Agentic AI application from scratch for a Senior AI Engineer technical assessment. The workspace contains only the 3 PDF corpus documents — all code, config, tests, Docker, and documentation must be created. The system must ingest documents, answer questions with source citations, route complex questions to an agentic workflow, and expose everything via a REST API with a Streamlit demo UI.

**Environment**: Python 3.11.8, all key packages pre-installed (LangChain, ChromaDB, FastAPI, Streamlit, sentence-transformers, pytest, pypdf, tiktoken). Docker available.

---

## Architecture Overview

```
User Question
     │
     ▼
┌──────────────┐   HTTP    ┌─────────────────────────┐
│ Streamlit UI │──────────▶│  FastAPI /api/v1/ask     │
└──────────────┘           └────────────┬────────────┘
                                        │
                                        ▼
                               ┌─────────────────┐
                               │  Router (LLM)   │
                               │  classifies Q   │
                               └───┬─────────┬───┘
                          simple   │         │   agentic
                              ▼              ▼
                    ┌──────────────┐  ┌─────────────────┐
                    │  RAG Chain   │  │ LangGraph Agent  │
                    │  (LCEL)      │  │ (ReAct + tools)  │
                    └──────┬───────┘  └──┬──┬──┬──┬─────┘
                           │             │  │  │  │
                           ▼             ▼  ▼  ▼  ▼
                      ┌──────────────────────────────┐
                      │   ChromaDB (persistent)      │
                      │   with metadata filtering    │
                      └──────────────┬───────────────┘
                                     ▲
                      ┌──────────────┴───────────────┐
                      │   Ingestion Pipeline         │
                      │   PDF → chunks → embeddings  │
                      └──────────────┬───────────────┘
                                     ▲
                      ┌──────────────┴───────────────┐
                      │   PDF Corpus (3 documents)   │
                      └──────────────────────────────┘
```

---

## Key Design Decisions

| Decision                  | Choice                                   | Rationale                                                                                            |
| ------------------------- | ---------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| **LLM**             | OpenAI GPT-4o (configurable)             | Strong LangChain integration, reliable structured output, widely used                                |
| **Embeddings**      | OpenAI text-embedding-3-small (1536d)    | Good quality/cost ratio; HuggingFace fallback for offline dev                                        |
| **Vector Store**    | ChromaDB (persistent)                    | Already installed, no external service, metadata filtering, sufficient for this scale                |
| **Chunking**        | RecursiveCharacterTextSplitter (800/200) | Smaller chunks improve precision on dense technical content; ~100 chunks total                       |
| **Agent Framework** | LangGraph                                | Stateful graph, conditional routing, tool nodes, iteration limits — the corpus itself recommends it |
| **Router**          | LLM classifier with structured output    | GPT-4o +`with_structured_output()` for reliable simple_rag / agentic classification                |
| **API**             | FastAPI                                  | Async-native, auto-docs, Pydantic integration                                                        |
| **UI**              | Streamlit                                | Pre-installed, minimal code for demo purposes                                                        |

---

## Project Structure

```
RAG_PRD_SAMPLE/
├── src/
│   ├── __init__.py
│   ├── config.py                 # pydantic-settings, model/embedding factories
│   ├── models.py                 # Shared domain models (ChunkMetadata, QueryResult)
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── loader.py             # PDF loading with pypdf, metadata + doc hash
│   │   ├── chunker.py            # RecursiveCharacterTextSplitter + metadata
│   │   └── pipeline.py           # Orchestrator: hash-check → load → chunk → upsert
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── vector_store.py       # ChromaDB wrapper: init, upsert, query, idempotency
│   │   └── retriever.py          # Retrieval logic with score thresholds
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── prompts.py            # All prompt templates
│   │   └── chain.py              # LCEL RAG chain: retrieve → format → LLM → parse
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── tools.py              # 4 @tool functions for the agent
│   │   ├── router.py             # LLM classifier → RouteDecision
│   │   └── graph.py              # LangGraph StateGraph, main run_query() entry point
│   └── api/
│       ├── __init__.py
│       ├── schemas.py            # Pydantic request/response models
│       ├── app.py                # FastAPI factory with lifespan, CORS, error handlers
│       └── routes.py             # REST endpoints
├── tests/
│   ├── __init__.py
│   ├── conftest.py               # Shared fixtures (settings, mock LLM, test ChromaDB)
│   ├── test_ingestion.py         # PDF loading, chunking, metadata, idempotency
│   ├── test_retrieval.py         # Vector store add/query, metadata filtering, thresholds
│   ├── test_rag.py               # Chain assembly, prompt formatting, citation extraction
│   ├── test_agent.py             # Router classification, tool output, graph compilation
│   └── test_api.py               # E2E: health, ask, documents, error handling
├── scripts/
│   ├── ingest.py                 # CLI: python -m scripts.ingest
│   └── evaluate.py               # Evaluation runner with metrics
├── configs/
│   └── eval_questions.json       # 15 evaluation questions with expected answers
├── corpus/                       # PDF documents (copied from workspace root)
├── ui/
│   └── app.py                    # Streamlit demo
├── docs/
│   └── architecture.md           # Architecture diagram + design decisions
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── pyproject.toml
├── .env.example
├── .gitignore
├── start.sh                      # Entrypoint: runs ingestion + API + Streamlit
└── README.md
```

---

## Implementation Phases (in order)

### Phase 1 — Foundation

Create project skeleton, configuration management, shared models.

**Files**: `pyproject.toml`, `requirements.txt`, `.env.example`, `.gitignore`, `src/__init__.py`, `src/config.py`, `src/models.py`

- `Settings` class via pydantic-settings with `.env` loading
- Factory functions: `get_settings()` (cached), `get_llm()`, `get_embeddings()`
- Early validation: fail fast with helpful error if `OPENAI_API_KEY` is missing
- Domain models: `ChunkMetadata`, `RetrievalResult`, `SourceReference`, `QueryResult`
- Copy corpus PDFs into `corpus/` directory

### Phase 2 — Ingestion Pipeline

Load PDFs, chunk, embed, store in ChromaDB with idempotency.

**Files**: `src/ingestion/loader.py`, `src/ingestion/chunker.py`, `src/retrieval/vector_store.py`, `src/ingestion/pipeline.py`, `scripts/ingest.py`

- `load_pdf()` using `PyPDFLoader` — attaches `source`, `page`, `doc_hash` (SHA-256) metadata
- `chunk_documents()` with `RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)`
- Deterministic chunk IDs: `{doc_name}::page{N}::chunk{M}`
- `VectorStoreManager` class wrapping ChromaDB with `PersistentClient`
- Idempotency: check `doc_hash` → skip unchanged docs, delete+re-ingest changed docs, `upsert()` as write primitive
- `IngestionPipeline.ingest()` returns structured report (processed/skipped/chunk counts)

### Phase 3 — Simple RAG Chain

Retrieval + grounded answer generation with citations.

**Files**: `src/retrieval/retriever.py`, `src/rag/prompts.py`, `src/rag/chain.py`

- Retriever wraps ChromaDB `.as_retriever()` with `search_kwargs={"k": 5}`
- Similarity score threshold filtering (0.3 minimum)
- LCEL chain: `retriever | format_docs_with_citations | prompt | llm | parse_response`
- Prompt instructs: cite as `[DocName, Page N]`, refuse when context is insufficient
- Returns structured output: answer + sources list + confidence level

### Phase 4 — Agentic Workflow

Router + LangGraph agent with tools for multi-step reasoning.

**Files**: `src/agent/tools.py`, `src/agent/router.py`, `src/agent/graph.py`

**Router** (`router.py`):

- `RouteDecision` Pydantic model: `route: Literal["simple_rag", "agentic"]`, `reasoning: str`
- Uses `ChatOpenAI.with_structured_output(RouteDecision)` with few-shot examples
- Simple factual → simple_rag; comparison/multi-step/cross-document → agentic

**Tools** (`tools.py`) — 4 `@tool` functions:

1. `search_knowledge_base(query, top_k=5)` — semantic search across all documents
2. `search_by_document(query, document_name)` — filtered search within one document
3. `compare_across_documents(topic, doc1, doc2)` — retrieve + format side-by-side
4. `get_document_summary(document_name)` — first chunks for high-level context

**Graph** (`graph.py`):

- `AgentState(TypedDict)`: messages, question, reasoning_steps, final_answer, sources
- Graph flow: `START → router → [simple_rag_node | agent_node → tool_node ⟲] → format_response → END`
- Max 5 iterations to prevent infinite loops
- `run_query(question, use_agent=None)` — main entry point used by API

### Phase 5 — REST API

FastAPI application with clear request/response contracts.

**Files**: `src/api/schemas.py`, `src/api/app.py`, `src/api/routes.py`

**Endpoints**:

- `POST /api/v1/ask` — `{question, use_agent?}` → `{answer, sources[], workflow, reasoning_steps[], confidence}`
- `GET /api/v1/health` — `{status, documents_count, chunks_count}`
- `POST /api/v1/ingest` — trigger ingestion → `{status, documents_processed, chunks_created}`
- `GET /api/v1/documents` — list ingested documents with chunk counts

**Production concerns**:

- Lifespan context manager: initialize vector store, verify corpus ingested
- CORS middleware for Streamlit
- Global exception handlers: LLM API errors, timeouts, empty corpus → structured error responses
- Request validation via Pydantic
- Structured logging throughout

### Phase 6 — Streamlit UI

Lightweight demo interface.

**File**: `ui/app.py`

- Text input + submit button
- "Use agentic workflow" checkbox toggle
- Displays answer with expandable source citations
- Shows reasoning steps for agentic responses
- Sidebar: document list, system health, workflow indicator

### Phase 7 — Docker & Deployment

Containerize for single-command startup.

**Files**: `Dockerfile`, `docker-compose.yml`, `start.sh`

- `python:3.11-slim` base, multi-stage build
- `start.sh`: runs ingestion → starts FastAPI (port 8000) + Streamlit (port 8501)
- docker-compose: single `app` service, volume for ChromaDB persistence, corpus mount, `.env` file
- Health check on `/api/v1/health`

### Phase 8 — Evaluation, Tests & Documentation

Complete the assessment requirements.

**Files**: `configs/eval_questions.json`, `scripts/evaluate.py`, `tests/*`, `docs/architecture.md`, `README.md`

**Evaluation set (15 questions)**:

- 5 simple RAG (single-doc factual)
- 5 multi-step/comparison (cross-document)
- 3 ambiguous (vague/missing referent)
- 2 unanswerable (outside corpus)

**Metrics**: retrieval recall@5, answer faithfulness (LLM-as-judge), citation accuracy, graceful refusal rate, router accuracy

**Tests**:

- Unit: ingestion, chunking, retrieval, router, tools, chain assembly
- E2E: FastAPI TestClient hitting all endpoints
- Fixtures: mock LLM, ephemeral ChromaDB for isolation

**Documentation**:

- `README.md`: problem understanding, architecture, setup instructions, API usage, design decisions, eval results, limitations, improvements
- `docs/architecture.md`: component diagram, data flow, design rationale

---

## Verification

After implementation, verify end-to-end:

1. **Ingestion**: `python -m scripts.ingest` — confirm 3 docs processed, ~100 chunks created, re-run confirms skip
2. **Simple RAG**: `curl -X POST localhost:8000/api/v1/ask -d '{"question": "What is RAG?"}'` — grounded answer with citations
3. **Agentic**: `curl -X POST localhost:8000/api/v1/ask -d '{"question": "Compare ChromaDB and Pinecone for production RAG"}'` — multi-step with reasoning_steps
4. **Unanswerable**: `curl -X POST localhost:8000/api/v1/ask -d '{"question": "How do I deploy on Kubernetes?"}'` — graceful refusal, no hallucination
5. **Tests**: `pytest tests/ -v` — all pass
6. **Evaluation**: `python -m scripts.evaluate` — metrics report generated
7. **Docker**: `docker-compose up` — both UI (8501) and API (8000) accessible
8. **Streamlit**: Open `localhost:8501` — ask questions, see answers with sources

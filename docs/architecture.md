# Intelligent Knowledge Assistant — Architecture

> **Author:** Rishav  
> **Date:** October 2026  
> **Status:** Production-ready reference implementation  
> **Audience:** Senior Engineering Managers, Architecture Review Board

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Component Architecture](#component-architecture)
3. [Architectural Decision Records (ADRs)](#architectural-decision-records-adrs)
4. [Data Flow Diagrams](#data-flow-diagrams)
5. [Security & Production Considerations](#security--production-considerations)

---

## System Overview

The Intelligent Knowledge Assistant is a production-oriented AI application that answers questions over a corpus of unstructured PDF documents using Retrieval-Augmented Generation (RAG) and agentic workflows. It combines Azure Document Intelligence for structured document parsing, ChromaDB for vector storage, Azure OpenAI (GPT-4o) for generation, and LangGraph for orchestrating multi-step reasoning. An LLM-based router dynamically selects between a lightweight single-retrieval RAG chain and a full agentic workflow with tool use, optimizing for both latency and answer quality.

### High-Level Architecture

```
+----------------------------------------------------------------------+
|                         USER INTERFACE                                |
|                                                                       |
|   +---------------------------+     +-----------------------------+   |
|   |    Streamlit UI (:8501)   |---->|   FastAPI REST API (:8000)  |   |
|   |  - Chat interface         |     |  - POST /api/v1/ask         |   |
|   |  - Workflow toggle        |     |  - POST /api/v1/ingest      |   |
|   |  - Source viewer          |     |  - GET  /api/v1/health      |   |
|   |  - Confidence display     |     |  - GET  /api/v1/documents   |   |
|   +---------------------------+     +----+------------------------+   |
+----------------------------------------------------------------------+
                                           |
                                           v
+----------------------------------------------------------------------+
|                       ORCHESTRATION LAYER                             |
|                                                                       |
|   +------------------------+                                          |
|   |     LLM Router         |  Structured output classification        |
|   |  (RouteDecision)       |  "simple_rag" | "agentic"               |
|   +-----------+------------+                                          |
|               |                                                       |
|       +-------+--------+                                              |
|       |                |                                              |
|       v                v                                              |
|   +--------+   +-------------------+                                  |
|   | Simple  |   | LangGraph Agent  |                                  |
|   | RAG     |   | (StateGraph)     |                                  |
|   | Chain   |   |                  |                                  |
|   | (LCEL)  |   | ReAct loop with  |                                  |
|   |         |   | 4 tools, max 5   |                                  |
|   |         |   | iterations       |                                  |
|   +----+----+   +--------+---------+                                  |
|        |                 |                                             |
+----------------------------------------------------------------------+
                           |
                           v
+----------------------------------------------------------------------+
|                       RETRIEVAL LAYER                                 |
|                                                                       |
|   +------------------+     +------------------+     +--------------+  |
|   |   Retriever      |---->|  VectorStore     |     | BM25 Index   |  |
|   |  - hybrid search |     |  Manager         |     | (rank-bm25)  |  |
|   |  - BM25 + dense  |     |  (ChromaDB)      |     | Sparse search|  |
|   |  - RRF fusion    |     |  - upsert        |     +--------------+  |
|   |  - cross-encoder |     |  - similarity    |                       |
|   |    re-ranking    |     |  - idempotency   |     +--------------+  |
|   |  - search_by_doc |     |  - get_all_docs  |     | Cross-Encoder|  |
|   +------------------+     +--------+---------+     | Re-Ranker    |  |
|                                                     | (ms-marco)   |  |
|                                                     +--------------+  |
+----------------------------------------------------------------------+
                                      |
                                      v
+----------------------------------------------------------------------+
|                      INGESTION PIPELINE                               |
|                                                                       |
|   +----------+    +-----------+    +----------+    +-----------+      |
|   |  PDF     |--->| Content   |--->| Content- |--->| Embed &   |      |
|   |  Corpus  |    | Parser    |    | Aware    |    | Upsert    |      |
|   |  (/*.pdf)|    | (Azure DI |    | Chunker  |    | (ChromaDB)|      |
|   |          |    |  + pypdf  |    |          |    |           |      |
|   |          |    |  fallback)|    | text:    |    | SHA-256   |      |
|   |          |    |           |    |  split   |    | hash IDs  |      |
|   |          |    | Extracts: |    | table:   |    |           |      |
|   |          |    |  - text   |    |  whole   |    | Determ.   |      |
|   |          |    |  - tables |    | image:   |    | chunk IDs |      |
|   |          |    |  - images |    |  whole   |    |           |      |
|   +----------+    +-----------+    +----------+    +-----------+      |
+----------------------------------------------------------------------+
                                                          |
                                                          v
+----------------------------------------------------------------------+
|                     EXTERNAL SERVICES                                 |
|                                                                       |
|   +------------------------+    +----------------------------+        |
|   | Azure OpenAI           |    | Azure Document Intelligence|        |
|   | - GPT-4o (generation)  |    | - prebuilt-layout model    |        |
|   | - GPT-4o (vision)      |    | - Markdown output          |        |
|   | - text-embedding-3-    |    | - Table extraction         |        |
|   |   small (embeddings)   |    | - Figure extraction        |        |
|   +------------------------+    +----------------------------+        |
+----------------------------------------------------------------------+
```

### Key Design Principles

| Principle | How It Manifests |
|---|---|
| **Graceful Degradation** | Azure DI falls back to pypdf; Azure embeddings fall back to HuggingFace local; vision summarization degrades to caption placeholders; cross-encoder re-ranker degrades to retrieval-score ordering if model unavailable |
| **Idempotent Operations** | SHA-256 file hashing prevents re-processing unchanged documents; deterministic chunk IDs enable safe upsert |
| **Content Awareness** | Tables are never split; images are summarized by vision LLM; text is chunked with section-boundary awareness |
| **Multi-Signal Retrieval** | Hybrid search (BM25 + dense) with Reciprocal Rank Fusion captures both keyword and semantic matches; cross-encoder re-ranking improves precision |
| **Separation of Concerns** | Ingestion, retrieval, RAG, agent, and API are fully decoupled modules with explicit data contracts |
| **Configuration as Code** | All settings via pydantic-settings with .env file; zero hardcoded secrets; validated at startup |

---

## Component Architecture

### 1. Configuration Management

**File:** `src/config.py`

**Purpose:** Centralizes all application configuration using pydantic-settings, providing validated, type-safe settings loaded from environment variables and `.env` files.

**How It Works:**

The `Settings` class extends `BaseSettings` from pydantic-settings, which automatically maps environment variables to class fields. Field validators enforce constraints at load time (e.g., `embedding_provider` must be `"azure_openai"` or `"huggingface"`; `chunk_size` must be 100-4000). The `get_settings()` function is `@lru_cache`-decorated, ensuring a single settings instance across the application lifecycle.

Factory functions `get_llm()` and `get_embeddings()` use lazy imports to construct configured Azure OpenAI instances, preventing import-time failures when optional dependencies are missing.

**Interfaces:**

```
Settings (BaseSettings)
  Fields: azure_openai_endpoint, azure_openai_api_key, azure_openai_deployment,
          parser_backend, chroma_persist_dir, chunk_size, chunk_overlap,
          retrieval_top_k, similarity_threshold, ...
  Properties: effective_api_version, corpus_path, chroma_path

get_settings() -> Settings           # Cached singleton
get_llm(settings?) -> AzureChatOpenAI
get_embeddings(settings?) -> AzureOpenAIEmbeddings | HuggingFaceEmbeddings
```

**Key Design Choices:**
- `lru_cache` on `get_settings()` prevents re-reading `.env` on every call while allowing test injection via parameter override
- Automatic fallback: if `parser_backend == "azure_di"` but credentials are missing, it silently downgrades to `"pypdf"` at startup
- The `extra="ignore"` model config prevents crashes when unrecognized environment variables are present

---

### 2. Document Ingestion Pipeline

**Files:** `src/ingestion/loader.py`, `src/ingestion/pipeline.py`

**Purpose:** Loads PDF documents with structured content extraction (text, tables, figures), using Azure Document Intelligence as the primary parser and pypdf as a resilient fallback.

**How It Works Internally:**

The `IngestionPipeline` orchestrator iterates over all PDFs in the corpus directory. For each file:

1. **Hash Check:** Computes SHA-256 of the file and queries ChromaDB for existing chunks with that hash. If the hash matches, the document is skipped entirely.
2. **Stale Cleanup:** If the hash differs (document was updated), all old chunks for that source are deleted before re-ingestion.
3. **Parsing:** The `load_pdf()` dispatcher selects the parser based on `settings.parser_backend`:
   - **Azure Document Intelligence (primary):** Sends the PDF to the `prebuilt-layout` model, which returns structured markdown with paragraph roles, bounding regions, table cells, and figure coordinates. The loader then:
     - Reconstructs tables as markdown from cell grid data, preserving row/column structure
     - Tracks character spans of tables and figures to exclude them from paragraph extraction (preventing duplication)
     - Extracts figure regions using PyMuPDF bounding-box clipping, encodes as base64 PNG, and sends to GPT-4o vision for summarization
     - Groups remaining paragraphs by page, filtering out headers/footers/page numbers
   - **pypdf (fallback):** Extracts raw text per page with no structural awareness. All content is tagged as `content_type="text"`.
4. **Chunking and Embedding:** Delegated to the chunker and vector store (described below).

**Interfaces:**

```
load_pdf(file_path, settings?) -> list[Document]
  Each Document has metadata: {source, page, content_type, doc_hash, file_path}

load_corpus(corpus_dir, settings?) -> dict[str, list[Document]]

IngestionPipeline(settings?)
  .ingest(corpus_dir?) -> IngestionReport
  .vector_store -> VectorStoreManager
```

**Content Type Tags:**

| content_type | Source | Downstream Handling |
|---|---|---|
| `"text"` | Paragraphs (Azure DI) or full pages (pypdf) | Split by `RecursiveCharacterTextSplitter` |
| `"table"` | Azure DI table extraction | Passed through whole — never split |
| `"image"` | Azure DI figure extraction + GPT-4o vision summary | Passed through whole — one chunk per figure |

---

### 3. Content-Aware Chunking

**File:** `src/ingestion/chunker.py`

**Purpose:** Splits documents into retrieval-optimized chunks while preserving the semantic integrity of structured content (tables and figures).

**How It Works Internally:**

The `chunk_documents()` function routes documents by their `content_type` metadata:

1. **Text documents** are split using LangChain's `RecursiveCharacterTextSplitter` with a tuned separator hierarchy:
   ```
   ["\n## ", "\n\n", "\n", ". ", " ", ""]
   ```
   This hierarchy prioritizes splitting at section headings, then paragraph boundaries, then sentence boundaries — preserving the most coherent context windows possible. Default chunk size is 800 characters with 200-character overlap.

2. **Table documents** pass through unmodified. A table that is split across two chunks loses its relational meaning — a row separated from its header is noise. Tables are kept as atomic units regardless of size.

3. **Image documents** (vision LLM summaries) pass through unmodified. Each figure summary is a self-contained description.

After splitting, all chunks are sorted by `(source, page, content_type_priority)` to maintain document reading order, with tables and images ordered before text on the same page for better retrieval context.

**Deterministic Chunk IDs:**

Each chunk receives a deterministic ID in the format `{source}::page{N}::chunk{M}`, where `M` is a monotonically increasing counter per source document. This enables:
- **Idempotent upsert:** Re-ingesting the same unchanged document produces identical IDs, so ChromaDB's upsert is a no-op
- **Reproducibility:** The same input always produces the same chunk set
- **Debuggability:** A chunk ID tells you exactly where it came from

**Interface:**

```
chunk_documents(documents, chunk_size=800, chunk_overlap=200) -> list[Document]
  Input:  Documents with content_type metadata from the loader
  Output: Chunked documents with chunk_id and chunk_index in metadata
```

---

### 4. Vector Store (ChromaDB)

**File:** `src/retrieval/vector_store.py`

**Purpose:** Manages persistent vector storage with idempotent upsert, document-level change detection, metadata filtering, and LangChain retriever integration.

**How It Works Internally:**

`VectorStoreManager` wraps a `chromadb.PersistentClient` for durable on-disk storage and a `langchain_chroma.Chroma` instance for retrieval integration. Key operations:

- **Upsert:** Uses deterministic `chunk_id` values as ChromaDB document IDs. Calls `collection.upsert()` which inserts new chunks and overwrites existing ones with the same ID. Metadata is sanitized to ChromaDB-compatible types (str, int, float, bool only).
- **Change Detection:** `is_document_current(doc_name, doc_hash)` queries the collection for any chunk with `source == doc_name`, then compares the stored `doc_hash` against the current file hash. This is an O(1) check that avoids re-processing unchanged documents.
- **Deletion:** `delete_by_document(doc_name)` removes all chunks for a given source, used when a document has been updated and needs re-ingestion.
- **Search:** `similarity_search()` delegates to LangChain's `similarity_search_with_relevance_scores()` with optional metadata filtering and score thresholding.

**Interface:**

```
VectorStoreManager(persist_dir, collection_name, embedding_function)
  .upsert_documents(chunks) -> int
  .is_document_current(doc_name, doc_hash) -> bool
  .delete_by_document(doc_name) -> int
  .similarity_search(query, k=5, filter_metadata?, score_threshold?) -> list[(Document, float)]
  .as_retriever(k=5, filter_metadata?) -> LangChain Retriever
  .get_collection_stats() -> dict
  .langchain_store -> Chroma
```

---

### 5. Retrieval System (Hybrid Search + Cross-Encoder Re-Ranking)

**File:** `src/retrieval/retriever.py`

**Purpose:** Provides a high-level retrieval interface combining dense vector search with BM25 sparse retrieval (Reciprocal Rank Fusion) and cross-encoder re-ranking for maximum retrieval quality.

**How It Works Internally:**

The `Retriever` class sits above `VectorStoreManager` and implements a multi-stage retrieval pipeline:

**Stage 1 — Dense Vector Search (always):**
ChromaDB cosine similarity search embeds the query and finds the nearest neighbors in embedding space. Good at capturing semantic meaning ("machine learning" matches "neural network training").

**Stage 2 — BM25 Sparse Search + Reciprocal Rank Fusion (hybrid mode):**
When `RETRIEVAL_MODE=hybrid`, BM25 (Okapi BM25 from `rank-bm25`) performs keyword-based term-frequency scoring across all stored documents. BM25 excels at exact keyword matches that dense search may miss ("HNSW" as a precise acronym). Both result sets are fused using **Reciprocal Rank Fusion (RRF)**:

```
RRF_score(d) = Σ (weight / (rank + k))
```

where `k=60` (standard RRF constant from the original paper), and each retriever's contribution is weighted (`dense_weight=0.6`, `bm25_weight=0.4`). RRF deduplicates by `chunk_id`, so documents found by both retrievers get boosted scores.

**Stage 3 — Cross-Encoder Re-Ranking (if enabled):**
When `RERANKER_ENABLED=true`, a cross-encoder model (`cross-encoder/ms-marco-MiniLM-L-6-v2`) re-scores each (query, document) pair. Unlike bi-encoders (which embed query and document independently), cross-encoders process the concatenated pair through a single transformer — much more accurate but slower. The re-ranker receives an over-fetched pool (k×2 candidates) and selects the final top-k.

**Stage 4 — Score Threshold Filtering:**
Results below the configured `SIMILARITY_THRESHOLD` (default 0.3) are removed before returning.

**Additional retrieval patterns:**
- **`search_by_document()`:** Scoped search within a single document, implemented via ChromaDB metadata filtering on the `source` field.
- **`get_document_chunks()`:** Retrieves representative chunks for document summarization by searching for "overview summary introduction" within that document.
- **`format_results_as_context()`:** Formats results as numbered context blocks with source attribution (`[Source N: filename, Page M]`) for injection into LLM prompts.

**BM25 Index Management:**
The BM25 index is built lazily on first hybrid search by calling `VectorStoreManager.get_all_documents()` to retrieve all stored texts. Documents are tokenized with a simple whitespace + lowercasing tokenizer. The index is cached and reused for subsequent queries.

**Interface:**

```
Retriever(vector_store, default_top_k=5, settings?)
  .search(query, top_k?, score_threshold?, source_filter?) -> list[RetrievalResult]
  .search_by_document(query, document_name, top_k?) -> list[RetrievalResult]
  .get_document_chunks(document_name, top_k=3) -> list[RetrievalResult]
  .format_results_as_context(results) -> str

  # Internal pipeline methods:
  ._bm25_search(query, k, source_filter?) -> list[(Document, float)]
  .reciprocal_rank_fusion(dense, bm25, weights, k) -> list[(Document, float)]
  ._rerank(query, results, top_k) -> list[(Document, float)]
```

**Configuration:**

| Setting | Default | Description |
|---|---|---|
| `RETRIEVAL_MODE` | `hybrid` | `"dense"` (vector only) or `"hybrid"` (BM25 + dense + RRF) |
| `BM25_WEIGHT` | `0.4` | RRF weight for BM25 scores |
| `DENSE_WEIGHT` | `0.6` | RRF weight for dense vector scores |
| `RERANKER_ENABLED` | `true` | Enable cross-encoder re-ranking |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | HuggingFace cross-encoder model |
| `RERANKER_TOP_K` | `5` | Final result count after re-ranking |

---

### 6. Simple RAG Chain

**Files:** `src/rag/chain.py`, `src/rag/prompts.py`

**Purpose:** Implements a straightforward retrieve-then-generate pipeline using LangChain Expression Language (LCEL) for single-retrieval questions.

**How It Works Internally:**

`run_simple_rag()` executes a five-step pipeline:

1. **Retrieve:** Calls `retriever.search()` with configured `top_k` and `score_threshold`
2. **Handle empty results:** Returns a structured "insufficient information" response with `confidence="low"` if no chunks pass the threshold
3. **Format context:** Converts results to numbered citation blocks via `format_results_as_context()`
4. **Generate:** Invokes the LCEL chain (`ChatPromptTemplate | AzureChatOpenAI | StrOutputParser`) with the context and question
5. **Assess confidence:** Computes average relevance score across retrieved chunks:
   - >= 0.6: `"high"`
   - >= 0.4: `"medium"`
   - < 0.4: `"low"`
   - Overridden to `"low"` if the LLM response contains uncertainty phrases ("don't have enough information", "cannot answer", etc.)

**Prompt Engineering:**

The system prompt enforces strict grounding rules:
- Answer ONLY from provided context (no hallucination)
- Cite every claim as `[DocumentName, Page N]`
- Explicitly refuse when context is insufficient
- Preserve technical accuracy for numbers and benchmarks

**Interface:**

```
run_simple_rag(question, retriever, llm, top_k=5, score_threshold=0.3) -> QueryResult
build_rag_chain(retriever, llm) -> LCEL Runnable
```

---

### 7. LLM Router

**File:** `src/agent/router.py`

**Purpose:** Classifies incoming questions to select the optimal workflow (simple RAG vs. agentic) using LLM structured output.

**How It Works Internally:**

The `classify_question()` function:

1. Binds a `RouteDecision` Pydantic model to the LLM via `llm.with_structured_output()`, which constrains the output to a JSON object matching the schema
2. Constructs a prompt that defines the two categories with concrete examples:
   - **`simple_rag`:** Single-retrieval questions ("What is RAG?", "What is HNSW?")
   - **`agentic`:** Multi-step, cross-document, or synthesis questions ("Compare FAISS and Pinecone", "How do chunking strategies affect retrieval quality across different vector databases?")
3. Invokes the chain and returns a `RouteDecision` with both the route and the reasoning
4. On any exception, defaults to `"agentic"` — the safer choice, since agentic provides a more thorough answer at the cost of higher latency

**Structured Output Schema:**

```python
class RouteDecision(BaseModel):
    route: Literal["simple_rag", "agentic"]
    reasoning: str
```

**Interface:**

```
classify_question(question, llm) -> RouteDecision
```

---

### 8. Agentic Workflow (LangGraph StateGraph)

**File:** `src/agent/graph.py`, `src/agent/tools.py`

**Purpose:** Implements a multi-step reasoning workflow using LangGraph's StateGraph for complex questions that require multiple retrievals, cross-document synthesis, or iterative refinement.

**How It Works Internally:**

The workflow is a compiled LangGraph `StateGraph` with five nodes and conditional edges:

```
                +-------+
                | START |
                +---+---+
                    |
                    v
            +-------+--------+
            |  router_node   |   LLM classifies question
            +-------+--------+
                    |
         +----------+-----------+
         |                      |
         v                      v
  +------+-------+    +---------+---------+
  | simple_rag   |    |   agent_node      |  LLM + bound tools
  | _node        |    +---------+---------+
  +------+-------+              |
         |            +---------+---------+
         |            |                   |
         v            v                   v
       +---+    +-----+------+    +-------+----------+
       |END|    | tool_node  |    | format_agent     |
       +---+    +-----+------+    | _response        |
                      |           +-------+----------+
                      |                   |
                      +----> agent_node   v
                        (loop back)     +---+
                                        |END|
                                        +---+
```

**State Schema (`AgentState`):**

```python
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]  # LangGraph message accumulator
    question: str
    route: str                               # "simple_rag" | "agentic"
    route_reasoning: str
    iteration_count: int                     # Tracks ReAct loop iterations
    reasoning_steps: list[str]               # Human-readable reasoning trace
    final_answer: str
    sources: list[dict]
    confidence: str
```

**ReAct Loop Mechanics:**

The agent node invokes the LLM with four bound tools. If the LLM returns tool calls, the `should_continue_agent` edge routes to `tool_node` (LangGraph's `ToolNode`), which executes the tools and appends results to the message history. Control then returns to `agent_node` for the next LLM invocation. This loop continues until:
- The LLM generates a response without tool calls (natural completion), or
- `iteration_count >= MAX_AGENT_ITERATIONS (5)` (safety limit)

**The Four Agent Tools:**

| Tool | Purpose | When Used |
|---|---|---|
| `search_knowledge_base` | General semantic search across all documents | Broad questions spanning multiple topics |
| `search_by_document` | Scoped search within a named document | When the agent knows which document to query |
| `compare_across_documents` | Retrieve from multiple documents for comparison | Cross-document comparison questions |
| `get_document_summary` | Get representative chunks for a document overview | Understanding document scope before deep search |

**Source Extraction:**

After the agent completes, `format_agent_response()` parses `[DocumentName.pdf, Page N]` citation patterns from the final response using regex, deduplicates them, and packages them as `SourceReference` objects.

**Interface:**

```
run_query(question, use_agent=None) -> QueryResult
  use_agent=None:  Router decides
  use_agent=True:  Force agentic (bypass router)
  use_agent=False: Force simple RAG (bypass router)

build_graph() -> Compiled StateGraph
```

---

### 9. REST API (FastAPI)

**Files:** `src/api/app.py`, `src/api/routes.py`, `src/api/schemas.py`

**Purpose:** Exposes the knowledge assistant as a RESTful service with structured request/response schemas, health monitoring, and lifecycle management.

**How It Works Internally:**

The application uses FastAPI's `lifespan` async context manager for startup/shutdown lifecycle. At startup, it verifies the corpus directory exists and checks whether the knowledge base has been populated. All routes are mounted under the `/api/v1` prefix.

**Endpoints:**

| Method | Path | Purpose | Request | Response |
|---|---|---|---|---|
| `POST` | `/api/v1/ask` | Ask a question | `AskRequest` (question, use_agent?) | `AskResponse` (answer, sources, workflow, reasoning_steps, confidence) |
| `POST` | `/api/v1/ingest` | Trigger ingestion | — | `IngestResponse` (status, counts, errors) |
| `GET` | `/api/v1/health` | Health check | — | `HealthResponse` (status, document/chunk counts) |
| `GET` | `/api/v1/documents` | List documents | — | `DocumentsResponse` (per-document chunk counts) |
| `GET` | `/docs` | OpenAPI/Swagger UI | — | Auto-generated |
| `GET` | `/redoc` | ReDoc API docs | — | Auto-generated |

**Error Handling:**

- Per-endpoint try/catch with `HTTPException` for known errors
- Global `@app.exception_handler(Exception)` for unhandled exceptions, returning a safe `500` response without leaking internal details
- All exceptions are logged with full tracebacks via `logger.exception()`

**Request/Response Contracts:**

All schemas are defined as Pydantic `BaseModel` classes in `src/api/schemas.py`, providing automatic request validation, serialization, and OpenAPI schema generation. The `AskRequest` enforces `min_length=1` and `max_length=2000` on questions.

---

### 10. Streamlit UI

**File:** `ui/app.py`

**Purpose:** Provides a conversational web interface for interacting with the knowledge assistant, communicating with the backend exclusively through the REST API.

**How It Works Internally:**

The UI is a stateless HTTP client that delegates all logic to the FastAPI backend via `httpx`:

- **Sidebar:** Displays system health (document count, chunk count, ingested document names) via the `/health` endpoint. Provides a workflow toggle (Auto / Force Simple RAG / Force Agentic).
- **Chat Interface:** Uses Streamlit's `st.chat_message` and `st.chat_input` for a conversational UX. Chat history is maintained in `st.session_state.messages`.
- **Response Display:** Shows the answer, workflow used (with icon), confidence level (color-coded), source count, expandable source citations with relevance scores, and expandable reasoning steps for agentic workflows.
- **Error Handling:** Catches `httpx.ConnectError` (server down), `httpx.HTTPStatusError` (API errors), and generic exceptions with user-friendly messages.

The UI has zero business logic — it is a pure presentation layer. This separation means the backend can be consumed by any HTTP client (curl, Postman, other UIs, CI pipelines).

---

## Architectural Decision Records (ADRs)

### ADR-001: Azure OpenAI vs. Direct OpenAI API

**Context:**  
The system requires LLM capabilities for generation (GPT-4o), embeddings (text-embedding-3-small), vision summarization (GPT-4o), and structured output (router classification). Both direct OpenAI and Azure OpenAI offer these capabilities.

**Decision:**  
Use Azure OpenAI as the exclusive LLM provider.

**Alternatives Considered:**
| Alternative | Pros | Cons |
|---|---|---|
| Direct OpenAI API | Simpler setup; latest models immediately; single API key | No VNet integration; data leaves org boundary; no enterprise SLA; limited compliance certifications |
| Azure OpenAI | VNet-peerable; data stays in tenant; SOC2/HIPAA/FedRAMP; AAD auth option; regional deployment | Model availability lag; Azure-specific SDK patterns; deployment management overhead |
| Self-hosted (vLLM/TGI) | Full data control; no API costs at scale | Massive infrastructure burden; GPU procurement; model licensing; operational complexity |

**Trade-offs:**
- Azure OpenAI adds deployment management overhead (you must create deployments for each model)
- Model availability lags behind direct OpenAI by weeks to months
- Requires Azure subscription and resource provisioning

**Why This Choice:**  
Enterprise compliance is non-negotiable. Azure OpenAI provides VNet integration so traffic never traverses the public internet, data residency guarantees (data stays within the Azure tenant boundary), enterprise SLAs (99.9% uptime), and integration with Azure Active Directory for identity federation. For a system handling potentially sensitive internal documents, these properties outweigh the convenience of direct OpenAI access.

---

### ADR-002: Azure Document Intelligence vs. pypdf Alone

**Context:**  
PDF documents contain structured content — tables with comparative data, figures with architectural diagrams, section headings — that is lost when extracted as flat text.

**Decision:**  
Use Azure Document Intelligence (`prebuilt-layout` model) as the primary PDF parser, with pypdf as an automatic fallback.

**Alternatives Considered:**
| Alternative | Pros | Cons |
|---|---|---|
| pypdf only | Zero cost; no external dependency; fast | Loses all structure: tables become garbled text; figures are invisible; no heading detection |
| Azure Document Intelligence | Structured markdown output; table cell extraction with row/column indices; figure bounding boxes; paragraph role classification (title, sectionHeading, pageHeader) | Costs per page; requires Azure subscription; network latency; potential rate limits |
| Unstructured.io | Open-source; local execution; multi-format | Less accurate table extraction than Azure DI; heavier local dependencies; inconsistent figure handling |
| LlamaParse | Good structure extraction; cloud-based | Another vendor dependency; less mature; limited enterprise compliance story |

**Trade-offs:**
- Azure DI costs approximately $1.50 per 1,000 pages (S0 tier)
- Adds a network dependency to the ingestion pipeline
- Requires separate Azure Cognitive Services resource provisioning

**Why This Choice:**  
The corpus documents contain dense comparison tables (vector database feature matrices, framework capability comparisons) that are critical for accurate retrieval. pypdf renders these tables as space-separated text where column alignment is destroyed — making them useless for answering "Compare X and Y" questions. Azure DI preserves the table grid structure (row indices, column indices, cell content), enabling the chunker to keep tables as atomic retrieval units. The figure extraction capability further enables vision-based summarization of architectural diagrams.

---

### ADR-003: Content-Aware Chunking vs. Uniform Chunking

**Context:**  
Standard text splitters treat all content identically, splitting at character boundaries regardless of content structure. This causes two specific problems: (1) tables split across chunks lose their column headers, making individual rows meaningless; (2) figure references become orphaned from their context.

**Decision:**  
Implement content-aware chunking that routes by content type: text is split normally, tables are kept whole, images are kept whole.

**Alternatives Considered:**
| Alternative | Pros | Cons |
|---|---|---|
| Uniform `RecursiveCharacterTextSplitter` | Simple; predictable chunk sizes; well-understood | Destroys table structure; splits figures mid-description |
| Semantic chunking (embedding-based) | Produces coherent topic-level chunks | Expensive (requires embedding every sentence); unpredictable chunk sizes; still splits tables |
| Document-level (one chunk per page) | Preserves all structure | Chunks too large for precise retrieval; exceeds embedding model context |
| Content-aware (this approach) | Tables atomic; images atomic; text optimally split | Variable chunk sizes; tables may be very large; requires content-type metadata from parser |

**Trade-offs:**
- Some table chunks may be significantly larger than the target chunk size (a 20-row table could be 2000+ characters), which increases embedding input and may reduce retrieval precision for non-table queries
- Requires a parser that classifies content types (Azure DI), so the benefit is lost when falling back to pypdf

**Why This Choice:**  
A table with its header row separated from its data rows is worse than useless — it is actively misleading. When a user asks "Which vector database supports HNSW indexing?", the answer is a specific cell in a comparison table. If that table is split, the chunk containing "Yes" has no column header to identify what "Yes" refers to. Keeping tables whole preserves their relational semantics at the cost of variable chunk sizes. The separator hierarchy for text (`["\n## ", "\n\n", "\n", ". ", " ", ""]`) further ensures that section boundaries are preferred over mid-paragraph splits.

---

### ADR-004: ChromaDB vs. Pinecone / Weaviate / pgvector

**Context:**  
The system needs a vector database for storing and retrieving document embeddings. The corpus is moderate in size (3 PDFs, ~hundreds of chunks), and the system should be deployable with minimal infrastructure.

**Decision:**  
Use ChromaDB with `PersistentClient` for local on-disk vector storage.

**Alternatives Considered:**
| Alternative | Pros | Cons |
|---|---|---|
| **ChromaDB** | Zero infrastructure; embedded (in-process); persistent local storage; Python-native; free | Not horizontally scalable; no built-in replication; no managed service; limited filtering capabilities |
| **Pinecone** | Fully managed; globally distributed; excellent filtering; enterprise SLA | Vendor lock-in; costs scale with vector count; network dependency for every query; data leaves local boundary |
| **Weaviate** | Rich filtering; hybrid search (BM25 + vector); GraphQL API; self-hostable | Requires separate server process; more complex deployment; heavier resource footprint |
| **pgvector** | Leverages existing PostgreSQL infrastructure; ACID transactions; mature ecosystem | Requires PostgreSQL; vector search performance inferior at scale; additional ops burden |
| **FAISS** | Extremely fast; Facebook-backed; mature | No built-in persistence; no metadata filtering; requires custom wrapper for production use |

**Trade-offs:**
- ChromaDB's in-process architecture means the vector store scales with the application process (no independent scaling)
- No built-in replication or backup (must back up the `data/chromadb` directory)
- Limited query capabilities compared to Pinecone's metadata filtering or Weaviate's hybrid search

**Why This Choice:**  
For a corpus of 3 documents producing hundreds of chunks, the operational complexity of a managed vector database is unjustified. ChromaDB's `PersistentClient` stores data to disk with zero configuration, starts in milliseconds (no separate process), and supports the exact operations needed: upsert by ID, metadata filtering by source document, and similarity search with relevance scores. The Docker Compose configuration persists ChromaDB's data directory as a named volume, surviving container restarts. Notably, the corpus documents themselves discuss these exact tradeoffs — the system practices what it teaches. If scale requirements grow beyond single-node capacity, migration to Pinecone or Weaviate is straightforward since all vector operations are abstracted behind `VectorStoreManager`.

---

### ADR-005: LangGraph vs. LangChain AgentExecutor vs. CrewAI

**Context:**  
Complex questions require multi-step reasoning: searching multiple documents, comparing results, synthesizing findings. This requires an orchestration framework that supports iterative tool use with state management.

**Decision:**  
Use LangGraph's `StateGraph` with explicit nodes, conditional edges, and typed state.

**Alternatives Considered:**
| Alternative | Pros | Cons |
|---|---|---|
| **LangGraph StateGraph** | Explicit state machine; typed state; conditional routing; visible iteration control; debuggable graph structure | More boilerplate than AgentExecutor; steeper learning curve; newer ecosystem |
| **LangChain AgentExecutor** | Simple API; less code; well-documented | Opaque execution loop; limited state visibility; deprecated in favor of LangGraph; no conditional routing |
| **CrewAI** | Multi-agent orchestration; role-based agents; good for diverse teams | Overkill for single-agent tool use; less control over iteration; additional dependency; less mature |
| **AutoGen** | Microsoft-backed; conversation-based orchestration; multi-agent | Heavy framework; conversation-centric (not tool-centric); complex setup |
| **Custom ReAct loop** | Full control; no framework dependency | Must implement message management, tool dispatch, error handling, iteration limits from scratch |

**Trade-offs:**
- LangGraph requires defining explicit state schemas, nodes, and edges — more upfront code than AgentExecutor
- The graph structure is harder to modify at runtime compared to a plain loop
- Debugging requires understanding the graph execution model

**Why This Choice:**  
LangGraph provides three properties that AgentExecutor lacks: (1) **Explicit state management** — the `AgentState` TypedDict makes every piece of state visible and typed, eliminating the "what does the agent know?" debugging problem; (2) **Conditional routing** — the `route_after_classification` edge cleanly branches between simple RAG and agentic workflows within a single graph; (3) **Iteration control** — `should_continue_agent` enforces `MAX_AGENT_ITERATIONS=5` as a graph edge condition, not a buried parameter. The graph structure is also self-documenting: reading `build_graph()` shows the complete execution topology. CrewAI was rejected because the system has a single agent with multiple tools, not multiple agents with different roles.

---

### ADR-006: LLM Router vs. Keyword/Rule-Based Routing

**Context:**  
The system must decide whether each question requires simple single-retrieval RAG or multi-step agentic reasoning. This routing decision directly impacts latency (simple RAG is ~2s; agentic is ~10-15s) and answer quality (agentic is more thorough for complex questions).

**Decision:**  
Use an LLM with structured output (`with_structured_output(RouteDecision)`) to classify questions.

**Alternatives Considered:**
| Alternative | Pros | Cons |
|---|---|---|
| **LLM structured output** | Understands nuance; handles novel phrasings; explains its reasoning; adapts without code changes | Adds one LLM call (~500ms) to every query; costs tokens; can misclassify |
| **Keyword matching** | Zero latency; zero cost; deterministic | Brittle: "compare" triggers agentic even for "compare this definition to RAG"; misses implicit complexity ("What should I use for production?") |
| **Embedding-based classifier** | Fast; trainable | Requires labeled training data; doesn't explain decisions; must be retrained for new question patterns |
| **Always agentic** | Simplest; always thorough | Unnecessary latency for simple questions; wastes tokens; higher cost |
| **Always simple RAG** | Fastest; cheapest | Fails on cross-document and multi-step questions |

**Trade-offs:**
- Adds ~500ms and ~200 tokens to every request for the classification call
- LLM classification is non-deterministic — the same question may occasionally route differently
- Default-to-agentic on error adds safety but increases average latency when errors occur

**Why This Choice:**  
The boundary between simple and complex questions is semantic, not syntactic. "What is HNSW?" is simple, but "What is HNSW and how does its performance compare across Pinecone and FAISS?" is complex — despite both starting with "What is". Keyword rules cannot reliably distinguish these. The structured output approach constrains the LLM to return exactly `{"route": "simple_rag"|"agentic", "reasoning": "..."}`, providing both a decision and an audit trail. The reasoning field is logged and included in the response's `reasoning_steps`, making the routing decision transparent and debuggable.

---

### ADR-007: Idempotent Ingestion with SHA-256 Hashing vs. Re-Index Everything

**Context:**  
The ingestion pipeline may be triggered multiple times (on container restart, via API, manually). Re-processing unchanged documents wastes compute (Azure DI API calls, embedding generation) and risks temporary data unavailability during delete-then-reinsert.

**Decision:**  
Compute SHA-256 hash of each PDF file and compare against the hash stored in ChromaDB metadata. Skip unchanged documents entirely.

**Alternatives Considered:**
| Alternative | Pros | Cons |
|---|---|---|
| **SHA-256 hash comparison** | Detects content changes precisely; O(1) check per document; no re-processing of unchanged docs | Requires hash storage in chunk metadata; does not detect metadata-only changes (e.g., updated filename with same content) |
| **Re-index everything** | Simplest implementation; always consistent | Wastes Azure DI API calls; wastes embedding compute; causes temporary data gap during re-ingestion |
| **File modification timestamp** | Simple; filesystem-native | Unreliable across containers (volume mounts reset timestamps); git clone resets timestamps; copies preserve content but change mtime |
| **Content hash per chunk** | Detects sub-document changes | Much more complex; requires diffing chunk sets; overkill for document-level updates |

**Trade-offs:**
- Hash comparison adds SHA-256 computation (~10ms per file) and one ChromaDB metadata query per document
- If the chunking strategy changes (e.g., `chunk_size` is updated), unchanged documents will not be re-chunked — requires a manual force-reindex

**Why This Choice:**  
The `start.sh` script runs ingestion on every container start. Without idempotency, every restart would re-process the entire corpus through Azure DI ($1.50/1000 pages) and re-embed all chunks. SHA-256 is a content-addressed check: if the file bytes are identical, the hash is identical, and processing is skipped. This makes `docker-compose restart` effectively free. The hash is stored in every chunk's metadata (`doc_hash` field), so the check requires only a single `collection.get(where={"source": name}, limit=1)` call.

---

### ADR-008: FastAPI vs. Flask

**Context:**  
The system needs an HTTP API to serve the Streamlit UI and potentially other clients. The API must handle potentially long-running LLM calls (10-15 seconds for agentic workflows).

**Decision:**  
Use FastAPI as the web framework.

**Alternatives Considered:**
| Alternative | Pros | Cons |
|---|---|---|
| **FastAPI** | Native async; auto-generated OpenAPI docs; Pydantic request/response validation; modern Python typing; lifespan management | Younger ecosystem than Flask; async can be confusing; Pydantic v2 migration friction |
| **Flask** | Mature; massive ecosystem; simple mental model; extensive documentation | No native async (requires gevent/gunicorn workers); no built-in validation; manual OpenAPI generation |
| **Django REST Framework** | Batteries-included; ORM; admin panel | Massively over-engineered for an API with 4 endpoints; ORM is irrelevant (no relational data) |
| **Raw ASGI (Starlette)** | Maximum control; minimal overhead | Must implement routing, validation, docs manually |

**Trade-offs:**
- FastAPI's async model is not fully utilized here (LLM calls are synchronous via LangChain), but it does not hurt and positions the system for future async integration
- The auto-generated `/docs` and `/redoc` endpoints provide immediate API documentation without maintenance

**Why This Choice:**  
FastAPI's Pydantic integration eliminates an entire class of bugs: request validation (`min_length=1`, `max_length=2000` on questions), response serialization, and OpenAPI schema generation are all derived from the same type definitions. The `lifespan` async context manager provides clean startup/shutdown hooks for verifying the knowledge base state. Auto-generated Swagger UI at `/docs` enables interactive API testing without Postman or curl. For a system where the API contract between frontend and backend is critical, having that contract enforced by the framework — not by convention — is a significant reliability win.

---

### ADR-009: Deterministic Chunk IDs vs. UUIDs

**Context:**  
Each chunk stored in ChromaDB requires a unique identifier. The choice of ID strategy affects upsert behavior, reproducibility, and debuggability.

**Decision:**  
Use deterministic IDs in the format `{source}::page{N}::chunk{M}` where `M` is a monotonically increasing counter per source document.

**Alternatives Considered:**
| Alternative | Pros | Cons |
|---|---|---|
| **Deterministic `{source}::page{N}::chunk{M}`** | Idempotent upsert (same input = same IDs); human-readable; debuggable; reproducible | Requires stable chunking order; counter depends on processing sequence; rename-sensitive |
| **UUID4** | Globally unique; no collisions; simple | Every re-ingestion creates new IDs; cannot upsert (must delete + insert); not debuggable |
| **Content hash (SHA-256 of chunk text)** | Content-addressed; deduplicates identical chunks | Hash collisions between similar chunks from different documents; not human-readable; doesn't handle metadata changes |
| **Composite key (source + page + content_hash)** | Content-aware; deduplicates | Overly complex; same content on different pages gets same ID |

**Trade-offs:**
- Deterministic IDs depend on stable document ordering (`sorted(corpus_path.glob("*.pdf"))`) and stable chunking — changing `chunk_size` or `chunk_overlap` changes all IDs
- The counter is per-document, not per-page, so page-level ID stability is not guaranteed if earlier pages change

**Why This Choice:**  
Deterministic IDs enable ChromaDB's `upsert()` to function as a true idempotent operation: if the same document is processed twice with the same chunking parameters, every chunk produces the same ID, and the upsert is a no-op. With UUIDs, re-ingesting the same document would create duplicate chunks unless all old chunks are deleted first — introducing a window where the document is absent from the index. The `{source}::page{N}::chunk{M}` format is also immediately interpretable during debugging: seeing `rag_architecture_patterns.pdf::page3::chunk7` in a retrieval result tells you exactly where to look in the source document.

---

### ADR-010: Dual-Parser Architecture (Azure DI + pypdf Fallback)

**Context:**  
Azure Document Intelligence requires Azure credentials and network access. Developers may not have Azure credentials configured locally, and CI/CD environments may not have network access to Azure Cognitive Services.

**Decision:**  
Implement a dual-parser architecture where Azure DI is the primary parser and pypdf is an automatic fallback at two levels: configuration-time and runtime.

**Alternatives Considered:**
| Alternative | Pros | Cons |
|---|---|---|
| **Dual-parser with automatic fallback** | Works everywhere; graceful degradation; developers can run locally without Azure; CI-friendly | Two code paths to maintain; pypdf results are lower quality; behavior differs between environments |
| **Azure DI only** | Single code path; consistent behavior | Cannot run without Azure credentials; blocks local development; CI requires secrets |
| **pypdf only** | Zero external dependencies; works everywhere | Loses all structural parsing benefits (tables, figures, headings) |
| **Configurable with hard failure** | Explicit about what parser is in use | Poor developer experience; "it works on my machine" when credentials differ |

**Trade-offs:**
- The system behaves differently depending on whether Azure DI credentials are available: Azure DI produces content-typed chunks (text/table/image) while pypdf produces only text chunks
- This means content-aware chunking benefits are silently lost when running with pypdf fallback
- Two fallback levels exist: (1) at `get_settings()` time, if credentials are missing, `parser_backend` is silently changed to `"pypdf"`; (2) at `load_pdf()` time, if Azure DI throws an exception, it catches and falls back to `load_pdf_pypdf()`

**Why This Choice:**  
The system should always be runnable. A developer cloning the repository should be able to `pip install` and `python -m scripts.ingest` without provisioning Azure resources. The fallback produces lower-quality results (no table preservation, no figure extraction), but it produces results. The configuration logs clear warnings when falling back (`"Document Intelligence credentials not set... Falling back to pypdf parser"`) so the degradation is visible, not silent. For production deployments, Azure DI credentials are always provided via `.env`, ensuring full-fidelity parsing.

---

## Data Flow Diagrams

### Ingestion Flow

```
PDF File (e.g., vector_database_comparison.pdf)
    |
    v
[SHA-256 Hash Computation]
    |
    v
[Is document current in ChromaDB?]---YES--->[SKIP - Document unchanged]
    |
    NO
    |
    v
[Delete stale chunks for this source]
    |
    v
[Parser Selection: azure_di or pypdf?]
    |                    |
    azure_di             pypdf
    |                    |
    v                    v
[Azure Document      [PyPDFLoader]
 Intelligence API]       |
    |                    v
    v               [Documents with
[Structured           content_type="text"]
 Markdown Output]        |
    |                    |
    +--[Tables]--------->+
    |  (content_type=    |
    |   "table")         |
    |                    |
    +--[Figures]-------->+
    |  (base64 ->        |
    |   GPT-4o Vision -> |
    |   summary text,    |
    |   content_type=    |
    |   "image")         |
    |                    |
    +--[Paragraphs]----->+
       (grouped by page, |
        content_type=    |
        "text")          |
                         |
                         v
              [Content-Aware Chunker]
                    |
         +----------+-----------+
         |          |           |
      text       table       image
      chunks     (whole)     (whole)
         |          |           |
         +----------+-----------+
                    |
                    v
         [Assign Deterministic IDs]
         "{source}::page{N}::chunk{M}"
                    |
                    v
         [Embed via text-embedding-3-small]
                    |
                    v
         [ChromaDB collection.upsert()]
                    |
                    v
         [Persistent Storage: data/chromadb/]
```

### Query Flow

```
User Question: "Compare FAISS and Pinecone for production use"
    |
    v
[FastAPI POST /api/v1/ask]
    |
    v
[LangGraph StateGraph.invoke()]
    |
    v
+--[Router Node]--+
|  LLM classifies |
|  -> "agentic"   |
|  (comparison     |
|   question)      |
+--------+--------+
         |
    +----+----+
    |         |
    v         v
[simple_rag] [agent_node]
    |             |
    v             v
[LCEL chain] [LLM + tools]
    |             |
    v             v
[QueryResult] [Tool calls:]
              |  1. search_by_document("FAISS",
              |     "vector_database_comparison.pdf")
              |  2. search_by_document("Pinecone",
              |     "vector_database_comparison.pdf")
              |
              v
         [tool_node executes]
              |
              v
         [agent_node (iteration 2)]
         [LLM synthesizes comparison]
              |
              v
         [format_agent_response]
         [Extract [Doc.pdf, Page N] citations]
              |
              v
         [QueryResult]
              |
              v
         [AskResponse JSON]
              |
              v
         [Streamlit displays answer,
          sources, confidence, reasoning]
```

### Retrieval Pipeline Detail (Hybrid Search + Re-Ranking)

```
Query: "What is HNSW?"
    |
    v
+---[Stage 1: Dense Vector Search]---+
|  Embed query with text-embedding-  |
|  3-small → cosine similarity in    |
|  ChromaDB → top k×2 results        |
|  (over-fetch for re-ranker pool)   |
+-----------------+------------------+
                  |
                  v
+---[Stage 2: BM25 + RRF Fusion]----+  (hybrid mode only)
|  Tokenize query → BM25Okapi score  |
|  all docs → top k×2 BM25 results   |
|                                     |
|  Reciprocal Rank Fusion:            |
|  score(d) = Σ weight/(rank + 60)    |
|  dense_weight=0.6, bm25_weight=0.4 |
|  Deduplicate by chunk_id            |
+-----------------+------------------+
                  |
                  v
+---[Stage 3: Cross-Encoder Rerank]--+  (if reranker enabled)
|  cross-encoder/ms-marco-MiniLM-L-  |
|  6-v2 scores each (query, doc)     |
|  pair through a single transformer |
|  Sort by cross-encoder score →     |
|  select final top-k                |
+-----------------+------------------+
                  |
                  v
+---[Stage 4: Threshold Filter]------+
|  Remove results below 0.3          |
|  similarity threshold              |
+-----------------+------------------+
                  |
                  v
         [list[RetrievalResult]]
```

### Agentic Workflow Detail

```
+------------------+
|   agent_node     |   Iteration 1
|  LLM reasons:    |
|  "I need info    |
|   about FAISS    |
|   and Pinecone"  |
+--------+---------+
         |
         | tool_calls: [
         |   search_by_document("FAISS features", "vector_database_comparison.pdf"),
         |   search_by_document("Pinecone features", "vector_database_comparison.pdf")
         | ]
         v
+------------------+
|   tool_node      |   Execute tools
|  Returns chunks  |
|  with citations  |
+--------+---------+
         |
         v
+------------------+
|   agent_node     |   Iteration 2
|  LLM reviews     |
|  tool results,   |
|  needs more info |
+--------+---------+
         |
         | tool_calls: [
         |   compare_across_documents("production deployment", ...)
         | ]
         v
+------------------+
|   tool_node      |   Execute tools
+--------+---------+
         |
         v
+------------------+
|   agent_node     |   Iteration 3
|  LLM has enough  |
|  context, writes |
|  final answer    |
|  with citations  |
|  (no tool_calls) |
+--------+---------+
         |
         v (should_continue_agent -> "format_agent_response")
+------------------+
| format_agent     |
| _response        |
|  Parse citations |
|  Assess confid.  |
+--------+---------+
         |
         v
       [END]
```

---

## Security & Production Considerations

### Secrets Management

| Concern | Implementation |
|---|---|
| **No secrets in code** | All credentials loaded from environment variables via pydantic-settings; `.env` file used for local development |
| **`.gitignore`** | `.env` file excluded from version control; `.env.example` provides a template with placeholder values |
| **Startup validation** | `get_settings()` logs warnings when critical credentials are missing, enabling early detection of misconfiguration |
| **No secret logging** | Pydantic `SecretStr` could be adopted for additional protection; currently credentials are `str` fields but never logged |

### CORS Configuration

The FastAPI application configures CORS middleware with `allow_origins=["*"]` for development flexibility. For production deployment:

```python
# Current (development):
allow_origins=["*"]

# Recommended production hardening:
allow_origins=["https://your-streamlit-domain.com"]
```

The current permissive configuration is appropriate for the Docker Compose deployment where both services run in the same network, but should be restricted when exposing the API externally.

### Error Handling and Graceful Degradation

The system implements defense-in-depth error handling at every layer:

| Layer | Error Handling Strategy |
|---|---|
| **Configuration** | Missing Azure DI credentials: auto-fallback to pypdf; Missing OpenAI credentials: logged warning at startup |
| **Document Parsing** | Azure DI failure per-file: catch and fallback to pypdf for that file; continues processing remaining files |
| **Figure Extraction** | PyMuPDF import failure: skip figures gracefully; Vision LLM failure: return caption placeholder text |
| **Ingestion** | Per-document try/catch: one document failure does not abort the batch; errors collected in `IngestionReport` |
| **Router** | Classification failure: default to agentic (more thorough, safer default) |
| **Agent** | Iteration limit (`MAX_AGENT_ITERATIONS=5`) prevents runaway tool-calling loops |
| **API** | Per-endpoint `HTTPException`; global `exception_handler(Exception)` catches unhandled errors; returns safe 500 without leaking internals |
| **UI** | `httpx.ConnectError` catch for server-down scenario; `HTTPStatusError` for API errors; generic fallback for unexpected exceptions |

### Containerization

**Multi-Stage Docker Build:**

```
Stage 1 (builder): python:3.11-slim
  - Installs build-essential for native dependencies
  - Installs Python packages from requirements.txt
  - Discarded after build (build tools not in runtime image)

Stage 2 (runtime): python:3.11-slim
  - Copies only installed packages from builder
  - Copies application code
  - Creates data directory for ChromaDB persistence
  - Exposes ports 8000 (API) and 8501 (Streamlit)
  - Health check: curl /api/v1/health every 30s
  - Entrypoint: start.sh (ingest -> FastAPI -> Streamlit)
```

**Docker Compose Configuration:**

| Feature | Implementation |
|---|---|
| **Data persistence** | Named volume `chromadb_data` mounted at `/app/data/chromadb` survives container restarts |
| **Corpus hot-swap** | `./corpus` mounted as read-only (`:ro`), allowing PDF updates without rebuilding the image |
| **Environment injection** | `env_file: .env` loads all configuration; overrides set explicitly for container-specific paths |
| **Health monitoring** | Health check with 60s start period (allows ingestion to complete), 30s interval, 3 retries |
| **Restart policy** | `restart: unless-stopped` ensures the service recovers from crashes automatically |

### Startup Sequence

The `start.sh` script orchestrates a deterministic startup:

```
1. Run ingestion (idempotent — skips unchanged documents)
2. Start FastAPI in background
3. Wait for API health check (up to 30s)
4. Start Streamlit in background
5. Wait for either process to exit (fail-fast)
```

This ensures the knowledge base is populated before any queries are served, while the idempotent ingestion means restarts do not re-process unchanged documents.

### Observability

| Capability | Implementation |
|---|---|
| **Structured logging** | `%(asctime)s | %(levelname)-8s | %(name)s | %(message)s` format across all modules |
| **Configurable log level** | `LOG_LEVEL` environment variable (default: INFO) |
| **LangSmith tracing** | Optional `LANGCHAIN_TRACING_V2=true` enables full chain/agent execution tracing in LangSmith |
| **Health endpoint** | `/api/v1/health` returns document count, chunk count, and ingested document names for monitoring |
| **Reasoning trace** | Every `QueryResult` includes `reasoning_steps` — a human-readable log of what the system did to answer the question |

---

## Appendix: Domain Model Summary

```
+------------------+     +------------------+     +------------------+
| ChunkMetadata    |     | RetrievalResult  |     | SourceReference  |
|                  |     |                  |     |                  |
| source: str      |     | content: str     |     | document: str    |
| page: int        |     | source: str      |     | page: int        |
| chunk_index: int |     | page: int        |     | chunk_id: str    |
| chunk_id: str    |     | chunk_id: str    |     | relevance_score  |
| doc_hash: str    |     | relevance_score  |     +------------------+
+------------------+     +------------------+
                                                   +------------------+
+------------------+     +------------------+      | RouteDecision    |
| QueryResult      |     | IngestionReport  |      |                  |
|                  |     |                  |      | route: Literal   |
| answer: str      |     | docs_processed   |      |  ["simple_rag",  |
| sources: list    |     | docs_skipped     |      |   "agentic"]     |
| workflow: str    |     | chunks_created   |      | reasoning: str   |
| reasoning_steps  |     | errors: list     |      +------------------+
| confidence: str  |     +------------------+
+------------------+
```

---

*This document reflects the architecture as implemented. All code references correspond to actual source files in the repository.*

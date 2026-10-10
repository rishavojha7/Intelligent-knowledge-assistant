# Intelligent Knowledge Assistant — Project Deep Dive

---

## What Is This Project

### Problem Statement

Enterprises accumulate vast amounts of domain knowledge inside PDF documents — architecture guides, vendor comparisons, framework evaluations — but this knowledge is effectively locked away from modern AI systems. A standard Large Language Model (LLM) cannot access these documents at inference time; it can only rely on whatever it learned during pre-training, which means it cannot answer questions about proprietary or recently created content, and it has no way to cite where an answer came from.

The core problem: **how do you let an LLM answer questions grounded in your organization's actual documents, with traceable citations, while gracefully refusing when the answer is not in the corpus?**

### Solution

This project is an **Intelligent Knowledge Assistant** — a production-oriented application that:

1. **Ingests** PDF documents into a vector knowledge base using Azure Document Intelligence (with pypdf fallback), preserving document structure (tables, figures, text).
2. **Chunks** content intelligently — text is split with overlap for context continuity; tables are kept whole; images are summarized by a vision LLM.
3. **Embeds** chunks into a ChromaDB vector store with deterministic IDs for idempotent re-ingestion.
4. **Routes** incoming questions through an LLM-powered classifier that decides whether a question needs simple single-retrieval RAG or a multi-step agentic workflow.
5. **Answers** simple questions via an LCEL RAG chain (retrieve → format → prompt → generate) with inline source citations.
6. **Handles complex questions** via a LangGraph StateGraph agent that iteratively calls specialized retrieval tools, reasons over results, and synthesizes a final answer.
7. **Serves** everything through a FastAPI REST API and a Streamlit chat UI.

### Key Capabilities

| Capability | Implementation |
|---|---|
| PDF ingestion with structure extraction | Azure Document Intelligence (`prebuilt-layout` model) with pypdf fallback |
| Content-aware chunking | Text split with `RecursiveCharacterTextSplitter`; tables kept whole; images vision-summarized |
| Semantic search | ChromaDB with Azure OpenAI `text-embedding-3-small` (or HuggingFace `all-MiniLM-L6-v2` offline) |
| Hybrid search + re-ranking | BM25 sparse + dense embeddings fused via Reciprocal Rank Fusion; cross-encoder (`ms-marco-MiniLM-L-6-v2`) re-ranking |
| Simple RAG with citations | LCEL chain: retrieve top-k → format context → prompt with citation instructions → generate |
| Agentic multi-step reasoning | LangGraph `StateGraph` with 4 retrieval tools, iteration limits, and conditional routing |
| Intelligent question routing | LLM structured output classifies `simple_rag` vs `agentic` |
| Idempotent ingestion | SHA-256 file hashing + deterministic chunk IDs = safe to re-run |
| REST API | FastAPI with Pydantic validation, auto-docs, CORS, health checks |
| Chat UI | Streamlit with session state, workflow indicators, source expanders |
| Evaluation framework | 15-question eval set with routing accuracy, source recall, content recall, graceful refusal metrics |

---

## Project Structure

```
RAG_PRD_SAMPLE/
├── src/                            # Core application source code
│   ├── __init__.py                 # Package docstring
│   ├── config.py                   # Configuration management (pydantic-settings)
│   ├── models.py                   # Shared domain models (dataclasses)
│   ├── ingestion/                  # Document ingestion pipeline
│   │   ├── __init__.py             # Package docstring
│   │   ├── loader.py               # PDF loading (Azure DI + pypdf fallback)
│   │   ├── chunker.py              # Content-aware text chunking
│   │   └── pipeline.py             # Orchestrator with idempotency
│   ├── retrieval/                  # Vector store and search
│   │   ├── __init__.py             # Package docstring
│   │   ├── vector_store.py         # ChromaDB wrapper (VectorStoreManager)
│   │   └── retriever.py            # Hybrid search + re-ranking pipeline (Retriever)
│   ├── rag/                        # Retrieval-Augmented Generation chain
│   │   ├── __init__.py             # Package docstring
│   │   ├── prompts.py              # All LLM prompt templates
│   │   └── chain.py                # LCEL RAG chain + run_simple_rag()
│   ├── agent/                      # Agentic workflow layer
│   │   ├── __init__.py             # Package docstring
│   │   ├── tools.py                # 4 LangChain @tool functions
│   │   ├── router.py               # LLM question classifier
│   │   └── graph.py                # LangGraph StateGraph definition + run_query()
│   └── api/                        # REST API
│       ├── __init__.py             # Package docstring
│       ├── schemas.py              # Pydantic request/response models
│       ├── app.py                  # FastAPI application factory
│       └── routes.py               # Endpoint handlers
├── ui/
│   └── app.py                      # Streamlit chat interface
├── scripts/
│   ├── ingest.py                   # CLI ingestion script
│   └── evaluate.py                 # Evaluation framework
├── tests/
│   ├── conftest.py                 # Shared fixtures (mock LLM, temp ChromaDB, etc.)
│   ├── test_ingestion.py           # Ingestion pipeline tests
│   ├── test_retrieval.py           # Vector store + retriever tests
│   ├── test_rag.py                 # RAG chain + prompt tests
│   ├── test_agent.py               # Router + tools + graph tests
│   └── test_api.py                 # FastAPI endpoint tests
├── configs/
│   └── eval_questions.json         # 15 evaluation questions (4 categories)
├── corpus/                         # PDF document corpus (3 PDFs)
│   ├── rag_architecture_patterns.pdf
│   ├── vector_database_comparison.pdf
│   └── agentic_ai_frameworks.pdf
├── docs/                           # Documentation
├── Dockerfile                      # Multi-stage container build
├── docker-compose.yml              # Single-command deployment
├── start.sh                        # Startup script (ingest → API → UI)
├── requirements.txt                # Python dependencies
├── pyproject.toml                  # Build/test configuration
├── .env.example                    # Environment variable template
└── .gitignore                      # Git exclusions
```

---

### 1. Configuration — `src/config.py`

**What it does:** Centralizes all application configuration. Every tunable parameter — Azure endpoints, API keys, chunk sizes, retrieval thresholds, port numbers — is loaded from environment variables and/or a `.env` file using `pydantic-settings`.

**Key classes and functions:**

| Symbol | Purpose |
|---|---|
| `Settings(BaseSettings)` | Pydantic settings model with ~25 fields, validators, and computed properties |
| `get_settings()` | `@lru_cache` singleton that loads settings once, configures logging, and validates credentials |
| `get_llm()` | Factory that returns a configured `AzureChatOpenAI` instance |
| `get_embeddings()` | Factory that returns either `AzureOpenAIEmbeddings` or `HuggingFaceEmbeddings` based on `embedding_provider` |
| `_configure_logging()` | Sets up structured logging with timestamps |

**How it connects to other files:**
- Every module in the project imports `get_settings()`, `get_llm()`, or `get_embeddings()` from this file.
- The `Settings` class is passed into `IngestionPipeline`, `VectorStoreManager`, and graph nodes.
- The `get_settings()` function uses `@lru_cache` so the settings object is created exactly once per process.

**Notable design patterns:**
- **Singleton via `@lru_cache`**: Ensures settings are loaded and validated once.
- **Lazy imports**: `get_llm()` and `get_embeddings()` import LangChain classes inside the function body. This avoids import-time failures if optional dependencies are missing.
- **Graceful degradation**: If Azure Document Intelligence credentials are missing, `get_settings()` automatically downgrades `parser_backend` from `"azure_di"` to `"pypdf"` with a warning.
- **Dual embedding provider**: The `embedding_provider` field lets you switch between Azure OpenAI embeddings (production) and HuggingFace `all-MiniLM-L6-v2` (free, offline, used in tests).
- **Computed properties**: `effective_api_version` and `corpus_path` provide derived values without duplicating logic.

**Field validators** enforce that `embedding_provider` is one of `("azure_openai", "huggingface")` and `parser_backend` is one of `("azure_di", "pypdf")`.

---

### 2. Shared Domain Models — `src/models.py`

**What it does:** Defines the data contracts used across all layers of the application as Python `dataclass` objects. These are the "lingua franca" that ingestion, retrieval, RAG, agent, and API layers all speak.

**Key classes:**

| Class | Purpose |
|---|---|
| `ChunkMetadata` | Metadata attached to each chunk: `source`, `page`, `chunk_index`, `chunk_id`, `doc_hash` |
| `RetrievalResult` | A single retrieved chunk with `content`, `source`, `page`, `chunk_id`, `relevance_score` |
| `SourceReference` | A citation reference returned to the user: `document`, `page`, `chunk_id`, `relevance_score` |
| `QueryResult` | Complete query response: `answer`, `sources`, `workflow`, `reasoning_steps`, `confidence` |
| `IngestionReport` | Summary of an ingestion run: `documents_processed`, `documents_skipped`, `chunks_created`, `errors` |

**How it connects:**
- `IngestionReport` is produced by `IngestionPipeline.ingest()` and consumed by `routes.py` and `scripts/ingest.py`.
- `RetrievalResult` is produced by `Retriever.search()` and consumed by `run_simple_rag()` and agent tools.
- `QueryResult` is produced by `run_simple_rag()` and `run_query()` and consumed by `routes.py` to build the API response.
- `SourceReference` is nested inside `QueryResult` and maps to `SourceReferenceResponse` in the API schema.

**Design pattern:** Using plain `dataclass` objects (not Pydantic `BaseModel`) keeps the domain layer lightweight and framework-agnostic. The API layer has its own Pydantic schemas (`schemas.py`) for serialization/validation.

---

### 3. Ingestion Pipeline — `src/ingestion/`

#### `src/ingestion/loader.py`

**What it does:** Loads PDF documents and extracts structured content. This is the most complex module in the project — it handles two parser backends and three content types.

**Key functions:**

| Function | Purpose |
|---|---|
| `compute_file_hash(file_path)` | Computes SHA-256 hash of a file for idempotent change detection |
| `load_pdf_azure_di(...)` | Primary loader: uses Azure Document Intelligence to extract structured markdown with tables, paragraphs, and figures |
| `load_pdf_pypdf(file_path)` | Fallback loader: uses pypdf for raw text extraction (no structure awareness) |
| `load_pdf(file_path, settings)` | Public API: auto-selects parser based on config, with automatic fallback on failure |
| `load_corpus(corpus_dir, settings)` | Loads all PDFs in a directory, returns `dict[filename, list[Document]]` |
| `_reconstruct_table_markdown(table)` | Rebuilds a markdown table from Azure DI's `DocumentTable` cell grid |
| `_extract_text_by_page(...)` | Extracts text paragraphs grouped by page, excluding regions occupied by tables and figures |
| `_extract_figures(...)` | Extracts figure bounding regions from the PDF using PyMuPDF, renders them as PNG, encodes as base64 |
| `_summarize_image(img_b64, caption, ...)` | Sends base64-encoded image to GPT-4o vision for summarization |
| `_spans_overlap(para_spans, excluded_spans)` | Checks if a paragraph's character spans overlap with table/figure spans |

**How Azure DI extraction works in detail:**

1. The PDF is sent to Azure Document Intelligence's `prebuilt-layout` model.
2. The response contains structured data: `paragraphs`, `tables`, and `figures` with character `spans` (offsets into the full content string) and `bounding_regions` (page coordinates).
3. **Tables**: Each table's cells are reconstructed into a markdown table via `_reconstruct_table_markdown()`. The table's character spans are tracked in `table_spans` so the same text is not also extracted as paragraphs.
4. **Figures**: Each figure's bounding region is used to clip the relevant area from the PDF page via PyMuPDF. The clipped image is rendered at 150 DPI, base64-encoded, and sent to GPT-4o vision for summarization. The summary becomes the chunk's `page_content`.
5. **Text**: Paragraphs are grouped by page number. Headers, footers, and page numbers are skipped. Paragraphs whose spans overlap with tables or figures are excluded. Section headings get `## ` prefixed.

Each `Document` object is tagged with `content_type` metadata: `"text"`, `"table"`, or `"image"`. This tag drives the chunker's behavior downstream.

**How it connects:**
- Called by `IngestionPipeline._ingest_document()` in `pipeline.py`.
- Returns `list[Document]` (LangChain `Document` objects) consumed by `chunk_documents()` in `chunker.py`.
- `compute_file_hash()` is also called directly by `pipeline.py` for idempotency checks.

**Notable design patterns:**
- **Dual parser with automatic fallback**: If Azure DI fails (network error, bad credentials), the code catches the exception and falls back to pypdf transparently.
- **Span-based deduplication**: Table and figure character spans are tracked so the same content is not extracted twice (once as a table and once as paragraph text).
- **Graceful image summarization**: If Azure OpenAI credentials are missing, image summaries fall back to a simple `[Figure: caption]` placeholder rather than failing.

---

#### `src/ingestion/chunker.py`

**What it does:** Splits documents into chunks for embedding, but respects content type boundaries. This is "content-aware chunking" — the chunker treats text, tables, and images differently.

**Key function:** `chunk_documents(documents, chunk_size=800, chunk_overlap=200)`

**Content-aware chunking logic:**

| Content Type | Behavior |
|---|---|
| `text` | Split with `RecursiveCharacterTextSplitter` using separators `["\n## ", "\n\n", "\n", ". ", " ", ""]` |
| `table` | Pass through as a single chunk — never split a table |
| `image` | Pass through as a single chunk — each figure summary is one chunk |

**After splitting:**

1. All chunks (text splits + passthrough tables/images) are sorted by `(source, page, type_priority)` where tables and images sort before text on the same page.
2. Each chunk receives a **deterministic chunk ID** in the format `{source}::page{N}::chunk{M}` where `M` is a global counter per source document.
3. Each chunk inherits all metadata from its parent document (`source`, `page`, `doc_hash`, `content_type`) plus new fields `chunk_index` and `chunk_id`.

**How it connects:**
- Called by `IngestionPipeline._ingest_document()` after `load_pdf()`.
- Returns `list[Document]` consumed by `VectorStoreManager.upsert_documents()`.

**Notable design patterns:**
- **Deterministic IDs**: The `chunk_id` format `"{source}::page{N}::chunk{M}"` means re-ingesting the same document with the same content produces the same IDs. ChromaDB's `upsert` then overwrites rather than duplicates.
- **Custom separator hierarchy**: `["\n## ", "\n\n", "\n", ". ", " ", ""]` means the splitter tries to break at section headings first, then paragraphs, then lines, then sentences, then words — preserving semantic coherence.

---

#### `src/ingestion/pipeline.py`

**What it does:** Orchestrates the end-to-end ingestion flow with hash-based idempotency.

**Key class:** `IngestionPipeline`

```python
class IngestionPipeline:
    def __init__(self, settings):
        # Creates embeddings model and VectorStoreManager
    
    def ingest(self, corpus_dir) -> IngestionReport:
        # Iterates over PDFs, calls _ingest_document() for each
    
    def _ingest_document(self, pdf_path, report):
        # 1. compute_file_hash(pdf_path)
        # 2. vector_store.is_document_current(doc_name, doc_hash) → skip if True
        # 3. vector_store.delete_by_document(doc_name) → remove stale chunks
        # 4. load_pdf(pdf_path) → get Documents
        # 5. chunk_documents(pages, chunk_size, chunk_overlap) → get chunks
        # 6. vector_store.upsert_documents(chunks) → embed and store
```

**Idempotency flow:**

```
For each PDF in corpus/:
  ├─ Compute SHA-256 hash of file
  ├─ Query ChromaDB: do chunks exist for this filename with this hash?
  │   ├─ YES → skip (document unchanged)
  │   └─ NO → 
  │       ├─ Delete any existing chunks for this filename (handles updates)
  │       ├─ Load PDF → chunk → embed → upsert
  │       └─ New chunks carry the new hash in metadata
```

**How it connects:**
- Instantiated by `scripts/ingest.py` (CLI), `routes.py` (API `/ingest` endpoint).
- Uses `VectorStoreManager` for all vector store operations.
- Uses `load_pdf()` and `chunk_documents()` from the ingestion sub-modules.
- Returns `IngestionReport` consumed by the caller for status reporting.

**Notable design patterns:**
- **Orchestrator pattern**: The pipeline coordinates multiple subsystems (loader, chunker, vector store) without containing business logic itself.
- **Error isolation**: Each document is processed in a try/except block. A failure on one PDF does not stop the rest of the corpus.

---

### 4. Retrieval System — `src/retrieval/`

#### `src/retrieval/vector_store.py`

**What it does:** Wraps ChromaDB operations behind a clean interface. This is the persistence layer for the entire knowledge base.

**Key class:** `VectorStoreManager`

| Method | Purpose |
|---|---|
| `__init__(persist_dir, collection_name, embedding_function)` | Initializes `chromadb.PersistentClient` and `langchain_chroma.Chroma` wrapper |
| `get_collection_stats()` | Returns `{total_chunks, documents, documents_count}` |
| `is_document_current(doc_name, doc_hash)` | Checks if a document's stored hash matches the provided hash |
| `delete_by_document(doc_name)` | Deletes all chunks for a document by `source` metadata filter |
| `upsert_documents(chunks)` | Embeds texts, upserts into ChromaDB with deterministic IDs and sanitized metadata |
| `similarity_search(query, k, filter_metadata, score_threshold)` | Semantic search with optional metadata filtering and score thresholding |
| `as_retriever(k, filter_metadata)` | Returns a LangChain `VectorStoreRetriever` for use in LCEL chains |

**How `upsert_documents` works:**

1. Extract `chunk_id` from each chunk's metadata as the deterministic ID.
2. Extract `page_content` as the text to embed.
3. Sanitize metadata — ChromaDB only supports `str`, `int`, `float`, `bool` values. Any other types are cast to `str`.
4. Call `self._embedding_function.embed_documents(texts)` to compute embeddings.
5. Call `collection.upsert(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)`.

Because IDs are deterministic (from `chunk_id`), calling `upsert` twice with the same content overwrites rather than duplicates.

**How it connects:**
- Used by `IngestionPipeline` for ingestion operations.
- Used by `Retriever` for search operations.
- Used by `routes.py` for health checks and document listing.
- Used by `graph.py` for building the retriever in workflow nodes.

**Notable design patterns:**
- **Facade pattern**: Wraps the ChromaDB native client and the LangChain Chroma wrapper behind a unified interface.
- **Dual client**: Uses `chromadb.PersistentClient` for low-level operations (upsert, delete, get) and `langchain_chroma.Chroma` for LangChain-compatible retrieval.

---

#### `src/retrieval/retriever.py`

**What it does:** Provides a multi-stage retrieval pipeline on top of `VectorStoreManager`. Combines dense vector search with BM25 sparse search (hybrid mode), fuses results via Reciprocal Rank Fusion, and optionally re-ranks with a cross-encoder model. This is what the RAG chain and agent tools actually call.

**Key class:** `Retriever`

| Method | Purpose |
|---|---|
| `search(query, top_k, score_threshold, source_filter)` | Multi-stage retrieval: dense → optional BM25+RRF → optional cross-encoder re-rank → threshold filter → `list[RetrievalResult]` |
| `search_by_document(query, document_name, top_k)` | Search scoped to a single document |
| `get_document_chunks(document_name, top_k)` | Retrieve representative chunks for summarization |
| `format_results_as_context(results)` | Format results as numbered context string: `[Source 1: doc.pdf, Page 3]\ncontent...` |
| `_bm25_search(query, k, source_filter)` | BM25 keyword search using `rank-bm25` (internal) |
| `reciprocal_rank_fusion(dense, bm25, ...)` | Fuse two ranked lists using RRF: `score(d) = Σ weight/(rank + 60)` (static method) |
| `_rerank(query, results, top_k)` | Cross-encoder re-ranking using `ms-marco-MiniLM-L-6-v2` (internal) |

**How it connects:**
- Instantiated in `graph.py` (both `simple_rag_node` and `_get_graph`), `routes.py` helper, and agent tool initialization. All call sites pass `settings=settings` to enable hybrid/re-ranking configuration.
- Called by `run_simple_rag()` in `chain.py` for single-retrieval queries.
- Called by all four agent tools in `tools.py` for agentic retrieval.

**Notable design patterns:**
- **Multi-stage pipeline**: Dense search → BM25+RRF fusion → cross-encoder re-ranking → threshold filter. Each stage is independent and can be enabled/disabled via config.
- **Over-fetching**: Retrieves k×2 candidates from both dense and BM25, then narrows via fusion and re-ranking — gives the re-ranker a larger candidate pool for better precision.
- **Adapter pattern**: Converts `(Document, score)` tuples from ChromaDB into `RetrievalResult` domain objects, decoupling the rest of the application from the vector store's return format.
- **Lazy initialization**: BM25 index and cross-encoder model are loaded on first use and cached. The cross-encoder loads with `local_files_only=True` to avoid network dependency.
- **Graceful degradation**: If the cross-encoder model isn't available locally, re-ranking is disabled with a warning and retrieval falls back to bi-encoder/RRF scores.
- **Context formatting**: The `format_results_as_context()` method produces a standardized format `[Source N: filename, Page M]` that the LLM prompt is designed to parse for citations.

---

### 5. RAG Chain — `src/rag/`

#### `src/rag/prompts.py`

**What it does:** Centralizes every LLM prompt in the application. There are four prompt templates:

| Prompt | Used By | Purpose |
|---|---|---|
| `RAG_SYSTEM_PROMPT` | Simple RAG chain | Instructs the LLM to answer only from context, cite sources as `[DocumentName, Page N]`, and refuse gracefully |
| `RAG_USER_PROMPT` | Simple RAG chain | Template with `{context}` and `{question}` placeholders |
| `ROUTER_SYSTEM_PROMPT` | Question router | Defines the `simple_rag` vs `agentic` classification criteria with examples |
| `ROUTER_USER_PROMPT` | Question router | Template with `{question}` placeholder for classification |
| `AGENT_SYSTEM_PROMPT` | LangGraph agent | Describes available documents, instructions for multi-step reasoning, and available tools |

**Key prompt engineering decisions:**

- **Grounding**: `RAG_SYSTEM_PROMPT` explicitly says "answer ONLY from the provided context" and "Do NOT use prior knowledge."
- **Citation format**: Mandates `[DocumentName, Page N]` format so citations can be parsed programmatically by `format_agent_response()`.
- **Graceful refusal**: Instructs "If the context does not contain enough information to answer the question, state clearly: 'I don't have enough information in the available documents to answer this question.'"
- **Router calibration**: The router prompt includes concrete examples for both categories and says "When in doubt, prefer 'agentic'" — biasing toward thoroughness over speed.

---

#### `src/rag/chain.py`

**What it does:** Implements the simple (non-agentic) RAG pipeline using LangChain Expression Language (LCEL).

**Key functions:**

| Function | Purpose |
|---|---|
| `build_rag_chain(retriever, llm)` | Builds the LCEL chain: `ChatPromptTemplate | llm | StrOutputParser()` |
| `run_simple_rag(question, retriever, llm, top_k, score_threshold)` | Executes the full simple RAG flow and returns a `QueryResult` |

**`run_simple_rag` step by step:**

```
1. Retrieve: retriever.search(query=question, top_k=5, score_threshold=0.3)
2. Handle empty results: return low-confidence "don't have enough information" answer
3. Format context: retriever.format_results_as_context(results)
4. Build chain: ChatPromptTemplate(system + user) | llm | StrOutputParser
5. Invoke chain: chain.invoke({"context": context, "question": question})
6. Build source references: map RetrievalResult → SourceReference (convert page to 1-indexed)
7. Assess confidence:
   - avg_score >= 0.6 → "high"
   - avg_score >= 0.4 → "medium"
   - else → "low"
   - Override to "low" if answer contains uncertainty phrases
8. Return QueryResult(answer, sources, workflow="simple_rag", reasoning_steps, confidence)
```

**How it connects:**
- Called by `simple_rag_node()` in `graph.py` when the router decides `simple_rag`.
- Called directly by `run_query()` when `use_agent=False` is explicitly set.

**Notable design patterns:**
- **LCEL composition**: The chain `prompt | llm | StrOutputParser()` is a LangChain Expression Language pipeline that composes cleanly and is easy to extend (e.g., adding a re-ranker stage).
- **Confidence heuristic**: Combines retrieval scores (quantitative) with uncertainty phrase detection (qualitative) for a multi-signal confidence assessment.

---

### 6. Agentic Workflow — `src/agent/`

#### `src/agent/router.py`

**What it does:** Classifies incoming questions as either `simple_rag` or `agentic` using the LLM with structured output.

**Key class and function:**

| Symbol | Purpose |
|---|---|
| `RouteDecision(BaseModel)` | Pydantic model with `route: Literal["simple_rag", "agentic"]` and `reasoning: str` |
| `classify_question(question, llm)` | Invokes the LLM with `ROUTER_SYSTEM_PROMPT` + `ROUTER_USER_PROMPT`, returns `RouteDecision` |

**How structured output works:**

```python
structured_llm = llm.with_structured_output(RouteDecision)
chain = prompt | structured_llm
decision = chain.invoke({"question": question})
# decision is a validated RouteDecision instance — no parsing needed
```

The LLM is forced to return a JSON object matching the `RouteDecision` schema. LangChain handles the function-calling / tool-use format under the hood.

**Fallback behavior:** If classification fails for any reason (LLM error, parsing error), the function defaults to `"agentic"` — the safer, more thorough path.

**How it connects:**
- Called by `router_node()` in `graph.py`.
- Uses prompts from `prompts.py`.

---

#### `src/agent/tools.py`

**What it does:** Defines four LangChain `@tool` functions that the LangGraph agent can call during its reasoning loop. Each tool wraps a different retrieval strategy.

**Module-level state:**

```python
_retriever: Retriever | None = None  # Set by initialize_tools()
```

Tools use a module-level `_retriever` instance that must be set by calling `initialize_tools(retriever)` before the agent is used.

**The four tools:**

| Tool | Function | When the Agent Uses It |
|---|---|---|
| `search_knowledge_base(query, top_k=5)` | General semantic search across all documents | For broad questions that may span multiple documents |
| `search_by_document(query, document_name)` | Search within a specific named document | When the agent knows which document contains the answer |
| `compare_across_documents(topic, document_1, document_2)` | Retrieve from multiple documents and group by source | For comparison questions ("Compare X and Y") |
| `get_document_summary(document_name)` | Get representative chunks from a document | To understand scope before detailed searches |

`compare_across_documents` has two modes:
- If both `document_1` and `document_2` are specified, it searches each document separately and returns grouped results.
- If neither is specified, it searches all documents (top_k=10) and groups results by source.

**The `AVAILABLE_DOCUMENTS` list** validates document names:
```python
AVAILABLE_DOCUMENTS = [
    "rag_architecture_patterns.pdf",
    "vector_database_comparison.pdf",
    "agentic_ai_frameworks.pdf",
]
```

**How it connects:**
- Tool functions are collected by `get_all_tools()` and bound to the LLM via `llm.bind_tools(tools)` in `graph.py`.
- Each tool calls methods on the module-level `_retriever` instance.
- `initialize_tools()` is called by `_get_graph()` in `graph.py`.

---

#### `src/agent/graph.py`

**What it does:** This is the heart of the application. It defines a LangGraph `StateGraph` that orchestrates the entire question-answering workflow — from routing through retrieval to response generation.

**State definition:**

```python
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]  # LangGraph message accumulator
    question: str                             # Original user question
    route: str                                # "simple_rag" or "agentic"
    route_reasoning: str                      # Why the router chose this route
    iteration_count: int                      # Agent loop counter
    reasoning_steps: list[str]                # Human-readable trace
    final_answer: str                         # Generated answer text
    sources: list[dict]                       # Parsed source citations
    confidence: str                           # "high", "medium", or "low"
```

**Graph nodes:**

| Node | Function | Responsibility |
|---|---|---|
| `router_node` | `router_node(state)` | Calls `classify_question()` → sets `route` and `route_reasoning` |
| `simple_rag_node` | `simple_rag_node(state)` | Builds retriever + LLM, calls `run_simple_rag()`, populates answer/sources/confidence |
| `agent_node` | `agent_node(state)` | Invokes `llm.bind_tools(tools)` with the message history. On first call, injects system message + question |
| `tool_node` | `ToolNode(tools)` | LangGraph's built-in tool executor — runs whatever tools the agent requested |
| `format_agent_response` | `format_agent_response(state)` | Extracts the last AI message content, parses `[DocName, Page N]` citations via regex, assesses confidence |

**Graph edges (the workflow):**

```
START → router_node
router_node → [conditional] → simple_rag_node  (if route == "simple_rag")
                             → agent_node       (if route == "agentic")

simple_rag_node → END

agent_node → [conditional] → tool_node              (if last message has tool_calls)
                            → format_agent_response  (if no tool_calls OR iteration >= 5)

tool_node → agent_node  (loop back for next reasoning step)

format_agent_response → END
```

**The agentic loop in detail:**

```
Iteration 1: agent_node invokes LLM → LLM decides to call search_knowledge_base("RAG patterns")
           → tool_node executes the tool → returns formatted results
Iteration 2: agent_node invokes LLM with tool results → LLM calls search_by_document("Pinecone", "vector_database_comparison.pdf")
           → tool_node executes → returns results  
Iteration 3: agent_node invokes LLM → LLM has enough info → generates final answer (no tool_calls)
           → should_continue_agent returns "format_agent_response"
           → format_agent_response parses citations → END
```

**Safety mechanisms:**
- `MAX_AGENT_ITERATIONS = 5` — the agent is forced to stop after 5 LLM calls to prevent runaway loops.
- If the agent hits the limit, `should_continue_agent` routes to `format_agent_response` regardless.

**Main entry point: `run_query(question, use_agent=None)`**

This is the single function the API calls. It handles three modes:

| `use_agent` | Behavior |
|---|---|
| `None` | Run the full graph (router decides) |
| `True` | Bypass router, run `_run_agentic()` directly |
| `False` | Bypass graph entirely, call `run_simple_rag()` directly |

**Lazy initialization:** `_get_graph()` initializes the compiled graph on first call:
1. Loads settings and embeddings.
2. Creates `VectorStoreManager` and `Retriever`.
3. Calls `initialize_tools(retriever)` to set the module-level retriever for tools.
4. Calls `build_graph()` to compile the `StateGraph`.
5. Caches the compiled graph in `_compiled_graph`.

**The `_run_agentic()` function** provides a standalone agentic loop (used when `use_agent=True` bypasses the graph). It manually implements the same agent_node → tool_node → agent_node loop with the same iteration limit.

**How it connects:**
- Called by `routes.py` endpoint handler (`run_query()`).
- Uses all other modules: `router.py`, `tools.py`, `chain.py`, `prompts.py`, `retriever.py`, `vector_store.py`, `config.py`.

---

### 7. REST API — `src/api/`

#### `src/api/schemas.py`

**What it does:** Defines the external API contract using Pydantic models. These are separate from the internal domain models in `models.py` — a clean boundary between internal and external representations.

**Request models:**

| Model | Fields |
|---|---|
| `AskRequest` | `question: str` (1-2000 chars), `use_agent: bool | None` |

**Response models:**

| Model | Purpose |
|---|---|
| `AskResponse` | `answer`, `sources`, `workflow`, `reasoning_steps`, `confidence` |
| `SourceReferenceResponse` | `document`, `page`, `chunk_id`, `relevance_score` |
| `HealthResponse` | `status`, `documents_count`, `total_chunks`, `documents` |
| `IngestResponse` | `status`, `documents_processed`, `documents_skipped`, `chunks_created`, `errors` |
| `DocumentsResponse` | `documents: list[DocumentInfo]`, `total_documents`, `total_chunks` |
| `DocumentInfo` | `name`, `chunk_count` |
| `ErrorResponse` | `error`, `detail` |

**Notable design patterns:**
- **Separate API schemas from domain models**: Internal code uses `QueryResult` (dataclass); the API layer converts to `AskResponse` (Pydantic). This allows the API contract to evolve independently of internal refactoring.
- **Pydantic validation**: `AskRequest.question` has `min_length=1, max_length=2000`, rejecting empty or excessively long questions at the serialization layer.

---

#### `src/api/app.py`

**What it does:** FastAPI application factory with lifecycle management.

**Key components:**

| Symbol | Purpose |
|---|---|
| `create_app()` | Factory function that creates and configures the FastAPI app |
| `lifespan()` | Async context manager for startup/shutdown. On startup: validates corpus directory, checks knowledge base status, logs readiness |
| `app` | Module-level `FastAPI` instance for uvicorn |

**Configuration applied:**
- **CORS middleware**: `allow_origins=["*"]` — allows the Streamlit UI (running on a different port) to call the API.
- **Global exception handler**: Catches all unhandled exceptions and returns a structured JSON error instead of a stack trace.
- **Root redirect**: `GET /` returns `{"message": "Knowledge Assistant API", "docs": "/docs"}`.
- **Auto-docs**: Swagger UI at `/docs`, ReDoc at `/redoc`.

**How it connects:**
- Mounts the `router` from `routes.py` via `app.include_router(router)`.
- Referenced by uvicorn: `uvicorn src.api.app:app`.
- The `lifespan` context manager initializes a `VectorStoreManager` at startup to verify the knowledge base has data.

---

#### `src/api/routes.py`

**What it does:** Implements the four API endpoints.

**Endpoints:**

| Endpoint | Method | Handler | Description |
|---|---|---|---|
| `/api/v1/ask` | POST | `ask_question()` | Accepts `AskRequest`, calls `run_query()`, returns `AskResponse` with timing |
| `/api/v1/health` | GET | `health_check()` | Returns knowledge base status (chunk count, document list) |
| `/api/v1/ingest` | POST | `trigger_ingestion()` | Creates `IngestionPipeline`, runs ingestion, returns report |
| `/api/v1/documents` | GET | `list_documents()` | Lists all ingested documents with per-document chunk counts |

**The `/ask` handler in detail:**

```python
async def ask_question(request: AskRequest) -> AskResponse:
    start_time = time.time()
    result = run_query(question=request.question, use_agent=request.use_agent)
    elapsed = time.time() - start_time
    # Log: "Query answered in 2.34s via simple_rag (confidence: high)"
    return AskResponse(
        answer=result.answer,
        sources=[SourceReferenceResponse(...) for s in result.sources],
        workflow=result.workflow,
        reasoning_steps=result.reasoning_steps,
        confidence=result.confidence,
    )
```

**How it connects:**
- All endpoints live under the `/api/v1` prefix via `APIRouter(prefix="/api/v1")`.
- The `/ask` endpoint calls `run_query()` from `graph.py` — the single entry point for all query processing.
- The `/ingest` endpoint creates a fresh `IngestionPipeline` and runs it.
- Helper `_get_vector_store()` creates a `VectorStoreManager` for health and document endpoints.

---

### 8. Streamlit UI — `ui/app.py`

**What it does:** Provides a chat-style web interface for the Knowledge Assistant. It communicates with the FastAPI backend via HTTP (using `httpx`).

**Layout:**

| Section | Content |
|---|---|
| **Sidebar** | System status (health check), document metrics, workflow mode toggle (Auto / Force Simple RAG / Force Agentic), about text |
| **Main area** | Chat title, description, chat history with metadata, chat input |

**Features:**

- **Chat history**: Uses `st.session_state.messages` to persist conversation across rerenders. Each message stores role, content, and metadata (workflow, confidence, source count).
- **Workflow toggle**: Radio button with three options that maps to `use_agent` parameter: `None` (Auto), `False` (Force Simple RAG), `True` (Force Agentic).
- **Real-time health check**: On every page load, calls `GET /api/v1/health` and displays status, document count, and chunk count.
- **Source expander**: After each answer, a collapsible "View Sources" section lists each source with document name, page number, and relevance score.
- **Reasoning steps expander**: For agentic workflows, shows the step-by-step reasoning trace.
- **Error handling**: Separate handlers for connection errors (API down), HTTP errors, and unexpected errors.

**How it connects:**
- Communicates exclusively via HTTP to the FastAPI backend (default `http://localhost:8000`).
- The `API_BASE_URL` is configurable via the `API_BASE_URL` environment variable.
- Uses `httpx.post()` with a 120-second timeout for the `/ask` endpoint (agentic workflows can be slow).

---

### 9. Scripts — `scripts/`

#### `scripts/ingest.py`

**What it does:** CLI script for document ingestion. This is the primary way to populate the knowledge base.

**Usage:**
```bash
python -m scripts.ingest                    # Use defaults from settings
python -m scripts.ingest --corpus-dir ./my_pdfs  # Custom corpus directory
python -m scripts.ingest --force            # Force re-ingestion (delete all, then ingest)
```

**What it outputs:**
- A header showing corpus directory, ChromaDB path, embedding model, and chunk size.
- An ingestion report showing documents processed, skipped, chunks created, and errors.
- Final collection stats showing total chunks and document names.

**Force mode:** When `--force` is passed, the script first deletes all existing chunks for every document in the collection, then runs a normal ingestion. This effectively rebuilds the entire knowledge base.

**How it connects:**
- Creates an `IngestionPipeline` from `pipeline.py` and calls `pipeline.ingest()`.
- Also used by `start.sh` to auto-ingest on container startup.

---

#### `scripts/evaluate.py`

**What it does:** Runs the evaluation question set against the live system and computes quality metrics.

**Usage:**
```bash
python -m scripts.evaluate                             # Defaults
python -m scripts.evaluate --eval-set configs/eval_questions.json --output results.json
```

**Key functions:**

| Function | Purpose |
|---|---|
| `load_eval_set(path)` | Loads questions from JSON |
| `evaluate_single(question_data)` | Runs one question through `run_query()` and computes per-question metrics |
| `compute_aggregate_metrics(results)` | Aggregates across all questions |
| `print_report(metrics, results)` | Prints a human-readable report to stdout |

**Per-question metrics computed:**

| Metric | How |
|---|---|
| `route_correct` | `result.workflow == expected_workflow` |
| `source_recall` | Fraction of expected source documents found in actual sources |
| `content_recall` | Fraction of expected answer terms found in the actual answer |
| `graceful_refusal` | For unanswerable questions: does the answer contain refusal phrases? |
| `latency_seconds` | Wall-clock time for the query |

**Aggregate metrics:**
- Routing accuracy (fraction of correctly routed questions)
- Average source recall
- Average content recall
- Unanswerable refusal rate
- Average latency
- Per-category breakdown

---

### 10. Tests — `tests/`

#### `tests/conftest.py`

**What it does:** Provides shared fixtures for the entire test suite. Sets up environment variables to ensure tests run offline without real Azure credentials.

**Key environment overrides:**
```python
os.environ.setdefault("EMBEDDING_PROVIDER", "huggingface")  # Use local embeddings, no API key needed
os.environ.setdefault("HF_HUB_OFFLINE", "1")                # Don't download models
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")           # Don't download models
```

**Fixtures:**

| Fixture | Scope | Purpose |
|---|---|---|
| `corpus_dir` | Per-test | Path to the `corpus/` directory (skips if missing) |
| `sample_pdf` | Per-test | Path to a single test PDF (prefers `rag_architecture_patterns.pdf`) |
| `temp_chroma_dir` | Per-test | `tempfile.TemporaryDirectory` for isolated ChromaDB (auto-cleaned) |
| `test_settings` | Per-test | `Settings` instance with HuggingFace embeddings, temp ChromaDB, 500-char chunks |
| `mock_llm` | Per-test | `MagicMock` that returns a predetermined answer with a citation |

---

#### `tests/test_ingestion.py`

**Tests:** 9 tests in 2 classes.

`TestPDFLoader` (7 tests):
- PDF returns non-empty document list
- Every page has non-empty content
- Required metadata fields exist (`source`, `page`, `doc_hash`)
- Source metadata is the filename
- Page numbers are valid non-negative integers
- Nonexistent file raises `FileNotFoundError`
- `compute_file_hash` is deterministic (same file → same SHA-256)
- `load_corpus` returns documents for each PDF

`TestChunker` (5 tests):
- Chunking produces more chunks than input documents
- Metadata preserved through chunking (all 5 required fields)
- Chunk IDs follow the `{source}::page{N}::chunk{M}` format
- Text chunk sizes respect the limit (within 1.5x tolerance; tables/images exempt)
- Empty input returns empty output
- More overlap produces more (or equal) chunks

---

#### `tests/test_retrieval.py`

**Tests:** 11 tests in 2 classes.

`TestVectorStoreManager` (7 tests):
- Upsert populates the collection
- Stats include document names
- Similarity search returns `(Document, score)` tuples
- Metadata filtering restricts results to the specified source
- **Idempotent upsert**: upserting the same chunks twice produces the same count
- `is_document_current` returns True for matching hash, False for wrong hash
- `delete_by_document` removes all chunks and they disappear from stats

`TestRetriever` (4 tests):
- Search returns `RetrievalResult` objects with content and source
- Document-scoped search only returns matching results
- Formatted context contains `[Source` and `Page` markers
- Empty results produce a "No relevant information" message

The `loaded_store` fixture creates a fresh ChromaDB in a temp directory, loads and chunks a real PDF, and upserts the chunks — providing realistic test data.

---

#### `tests/test_rag.py`

**Tests:** 5 tests in 2 classes.

`TestPrompts` (3 tests):
- System prompt contains "provided context" (grounding instruction)
- User prompt has `{context}` and `{question}` placeholders
- System prompt instructs graceful refusal

`TestSimpleRAG` (2 tests):
- No results → low confidence, workflow is `simple_rag`, answer mentions lack of information
- Result has all required fields (`answer`, `sources`, `workflow`, `reasoning_steps`, `confidence`)

Uses `MagicMock` for both retriever and LLM — no real API calls.

---

#### `tests/test_agent.py`

**Tests:** 7 tests in 3 classes.

`TestRouter` (2 tests):
- `RouteDecision` accepts `"simple_rag"` and `"agentic"`
- `RouteDecision` rejects invalid values (Pydantic `Literal` validation)

`TestTools` (4 tests):
- `get_all_tools()` returns exactly 4 tools
- Tool names match expected set
- Every tool has a non-empty description
- `AVAILABLE_DOCUMENTS` contains the three expected PDFs

`TestGraph` (2 tests):
- Graph compiles without errors (uses mock retriever for tool initialization)
- `AgentState` TypedDict has all 9 expected keys

---

#### `tests/test_api.py`

**Tests:** 7 tests in 4 classes.

`TestHealthEndpoint` (2 tests):
- `GET /api/v1/health` returns 200
- Response has `status`, `documents_count`, `total_chunks` fields

`TestRootEndpoint` (1 test):
- `GET /` returns 200 with `message` field

`TestAskEndpoint` (3 tests):
- Empty question returns 422 (Pydantic validation error)
- Missing question field returns 422
- With mocked `run_query()`: returns correct schema, workflow, sources, `use_agent` is passed through

`TestDocumentsEndpoint` (2 tests):
- `GET /api/v1/documents` returns 200
- Response has `documents`, `total_documents`, `total_chunks` fields

Uses `fastapi.testclient.TestClient` for synchronous endpoint testing. LLM calls are mocked with `@patch("src.api.routes.run_query")`.

---

### 11. Evaluation Question Set — `configs/eval_questions.json`

**What it does:** Defines 15 evaluation questions across 4 categories, each with expected workflow, expected source documents, and expected answer content keywords.

**Categories:**

| Category | Count | IDs | Purpose |
|---|---|---|---|
| `simple_rag` | 5 | `simple_01` to `simple_05` | Single-document factual questions (e.g., "What is RAG?") |
| `multi_step` | 5 | `multi_01` to `multi_05` | Cross-document comparison/synthesis (e.g., "Compare ChromaDB and Pinecone") |
| `ambiguous` | 3 | `ambiguous_01` to `ambiguous_03` | Vague/incomplete questions (e.g., "What's the best approach?") |
| `unanswerable` | 2 | `unanswerable_01` to `unanswerable_02` | Questions outside the corpus (e.g., "What is GPT-4o API pricing?") |

Each question has:
- `expected_workflow`: What the router should classify it as
- `expected_sources`: Which PDFs should appear in the sources
- `expected_answer_contains`: Keywords that should appear in the answer
- `notes`: Human explanation of what makes this question interesting

---

### 12. Configuration Files

#### `Dockerfile`

**What it does:** Multi-stage Docker build for production deployment.

**Stage 1 — Builder:**
- Base: `python:3.11-slim`
- Installs system dependencies (`build-essential`, `curl`)
- Installs Python dependencies from `requirements.txt`

**Stage 2 — Runtime:**
- Base: `python:3.11-slim` (clean, no build tools)
- Copies installed packages from builder stage
- Copies application code: `src/`, `scripts/`, `ui/`, `corpus/`, `configs/`, `start.sh`
- Creates `data/chromadb/` directory for persistence
- Exposes ports 8000 (API) and 8501 (Streamlit)
- Health check: `curl -f http://localhost:8000/api/v1/health` every 30s
- Entry point: `./start.sh`

**Notable design patterns:**
- **Multi-stage build**: The builder stage has `build-essential` for compiling C extensions; the runtime stage doesn't, reducing image size.
- **60-second start period**: The health check gives the application 60 seconds to start before counting failures (ingestion can take time).

---

#### `docker-compose.yml`

**What it does:** Single-command deployment of the entire application.

```bash
docker-compose up --build
```

**Key configuration:**
- Builds from the `Dockerfile` in the current directory.
- Maps ports: `${API_PORT:-8000}:8000` and `${STREAMLIT_PORT:-8501}:8501`.
- **Named volume** `chromadb_data` persists the vector store across container restarts.
- **Read-only bind mount** `./corpus:/app/corpus:ro` allows updating PDFs without rebuilding the image.
- Loads environment variables from `.env` file.
- Health check and `restart: unless-stopped` policy for resilience.

---

#### `start.sh`

**What it does:** Startup orchestration script that runs inside the container.

**Sequence:**
1. Run `python -I -m scripts.ingest` — idempotent ingestion (skips unchanged docs).
2. Start `uvicorn src.api.app:app` in the background.
3. Wait up to 30 seconds for the API health check to pass.
4. Start `streamlit run ui/app.py` in the background.
5. Print service URLs.
6. `wait -n` — wait for either process to exit, then exit with its code.

The `-I` flag on Python prevents the interpreter from loading code from the current directory, which is a security measure when working with untrusted data.

---

#### `requirements.txt`

**Dependencies by category:**

| Category | Packages |
|---|---|
| Core LLM & RAG | `langchain`, `langchain-core`, `langchain-community`, `langchain-openai`, `langchain-chroma`, `langchain-text-splitters`, `langgraph`, `openai` |
| Embeddings | `sentence-transformers`, `langchain-huggingface` |
| Vector Store | `chromadb` |
| PDF Processing | `pypdf`, `azure-ai-documentintelligence`, `azure-core`, `azure-identity`, `PyMuPDF`, `pillow` |
| Tokenization | `tiktoken` |
| API | `fastapi`, `uvicorn[standard]`, `pydantic`, `pydantic-settings` |
| UI | `streamlit`, `httpx` |
| Testing | `pytest`, `pytest-asyncio` |
| Utilities | `python-dotenv` |

---

#### `pyproject.toml`

- Build system: `setuptools >= 68.0`
- Project name: `knowledge-assistant`, version `1.0.0`
- Requires Python 3.11+
- Pytest configuration: test paths = `tests/`, async mode = `auto`, suppress deprecation warnings
- Package discovery: includes `src*`

---

#### `.env.example`

Template for all environment variables. Key sections:
- Azure OpenAI (endpoint, API key, deployment names, temperature)
- Embedding provider selection (`azure_openai` or `huggingface`)
- Document parser selection (`azure_di` or `pypdf`)
- Azure Document Intelligence credentials
- ChromaDB storage path and collection name
- Retrieval tuning (chunk size, overlap, top-k, similarity threshold)
- API and Streamlit ports
- Optional LangSmith tracing configuration
- Logging level

---

#### `.gitignore`

Excludes: Python bytecode, virtual environments, `.env` (secrets), `data/chromadb/` (vector store data), IDE files, test cache, evaluation outputs, and OS files.

---

## How Everything Works Together

### End-to-End Ingestion Flow

```
corpus/*.pdf
    │
    ▼
scripts/ingest.py (or POST /api/v1/ingest)
    │
    ▼
IngestionPipeline.__init__()
    ├── get_settings() → loads .env configuration
    ├── get_embeddings() → AzureOpenAIEmbeddings or HuggingFaceEmbeddings
    └── VectorStoreManager() → chromadb.PersistentClient at ./data/chromadb
    │
    ▼
IngestionPipeline.ingest(corpus_dir)
    │
    ├── For each *.pdf in corpus_dir (sorted):
    │   │
    │   ▼
    │   _ingest_document(pdf_path, report)
    │       │
    │       ├── compute_file_hash(pdf_path) → SHA-256 hex digest
    │       │
    │       ├── vector_store.is_document_current(name, hash)
    │       │   ├── YES → skip, report.documents_skipped += 1
    │       │   └── NO ↓
    │       │
    │       ├── vector_store.delete_by_document(name) → remove stale chunks
    │       │
    │       ├── load_pdf(pdf_path, settings)
    │       │   ├── If parser_backend == "azure_di":
    │       │   │   ├── Azure DI: analyze document → structured markdown
    │       │   │   ├── Extract tables → Document(content_type="table")
    │       │   │   ├── Extract figures → base64 → GPT-4o vision → Document(content_type="image")
    │       │   │   └── Extract text paragraphs (excluding table/figure spans) → Document(content_type="text")
    │       │   │   (Falls back to pypdf on failure)
    │       │   └── If parser_backend == "pypdf":
    │       │       └── PyPDFLoader → Document(content_type="text") per page
    │       │
    │       ├── chunk_documents(pages, chunk_size=800, chunk_overlap=200)
    │       │   ├── Text docs → RecursiveCharacterTextSplitter → split chunks
    │       │   ├── Table docs → pass through (never split)
    │       │   ├── Image docs → pass through (never split)
    │       │   ├── Sort all by (source, page, type_priority)
    │       │   └── Assign deterministic chunk_id = "{source}::page{N}::chunk{M}"
    │       │
    │       └── vector_store.upsert_documents(chunks)
    │           ├── Sanitize metadata (ChromaDB type restrictions)
    │           ├── embedding_function.embed_documents(texts) → vectors
    │           └── collection.upsert(ids, documents, embeddings, metadatas)
    │
    └── Return IngestionReport(processed, skipped, chunks_created, errors)
```

### End-to-End Query Flow

```
User question (via Streamlit UI or POST /api/v1/ask)
    │
    ▼
routes.py: ask_question()
    ├── Validates request (Pydantic: 1-2000 chars, optional use_agent)
    ├── Starts timer
    │
    ▼
graph.py: run_query(question, use_agent)
    │
    ├── If use_agent is None → invoke full LangGraph:
    │   │
    │   ▼
    │   router_node(state)
    │       ├── LLM.with_structured_output(RouteDecision)
    │       ├── Prompt: "Classify this question: {question}"
    │       └── Returns: route="simple_rag"|"agentic", reasoning="..."
    │       │
    │       ▼
    │   route_after_classification(state)
    │       ├── If route == "simple_rag" → simple_rag_node
    │       └── If route == "agentic" → agent_node
    │
    │   ┌─────────────────────────┐    ┌──────────────────────────────┐
    │   │   SIMPLE RAG PATH       │    │   AGENTIC PATH               │
    │   │                         │    │                              │
    │   │ simple_rag_node(state)  │    │ agent_node(state)            │
    │   │ ├ Build retriever       │    │ ├ LLM.bind_tools(4 tools)   │
    │   │ ├ retriever.search()    │    │ ├ Invoke with messages       │
    │   │ ├ format_context()      │    │ └ Returns AIMessage          │
    │   │ ├ LCEL chain.invoke()   │    │     │                        │
    │   │ ├ Build sources list    │    │     ▼                        │
    │   │ ├ Assess confidence     │    │ should_continue_agent()      │
    │   │ └ → END                 │    │ ├ Has tool_calls? → tool_node│
    │   └─────────────────────────┘    │ └ No tool_calls? → format    │
    │                                  │     │         │              │
    │                                  │     ▼         ▼              │
    │                                  │ tool_node  format_agent_     │
    │                                  │ (execute)  response()        │
    │                                  │     │      ├ Parse citations  │
    │                                  │     │      ├ Assess confidence│
    │                                  │     ▼      └ → END           │
    │                                  │ agent_node (loop back)       │
    │                                  │ (max 5 iterations)           │
    │                                  └──────────────────────────────┘
    │
    ├── If use_agent is True → _run_agentic() (bypass router)
    └── If use_agent is False → run_simple_rag() (bypass graph)
    │
    ▼
Returns QueryResult(answer, sources, workflow, reasoning_steps, confidence)
    │
    ▼
routes.py: converts to AskResponse → JSON → HTTP 200
    │
    ▼
Streamlit UI: displays answer, sources expander, reasoning steps expander
```

### How the Router Decides `simple_rag` vs `agentic`

The router uses the LLM itself (GPT-4o) with **structured output** to classify each question. The prompt (`ROUTER_SYSTEM_PROMPT` in `prompts.py`) provides:

**Criteria for `simple_rag`:**
- Can be answered with a single retrieval from one or two document sections
- Examples: "What is RAG?", "What is HNSW?", "List pros/cons of semantic chunking"

**Criteria for `agentic`:**
- Requires multiple retrieval steps, cross-document comparison, synthesis from different topics, or multi-step reasoning
- Examples: "Compare FAISS and Pinecone", "Which vector DB should I use with LangGraph?"

**Bias toward safety:** "When in doubt, prefer 'agentic'" — because the agentic path always produces a correct (if slower) answer, while the simple path may miss information.

The classification typically takes ~0.5 seconds and saves 5-10 seconds on simple queries by avoiding unnecessary tool-calling loops.

**Fallback:** If classification fails (LLM error, network issue), the system defaults to `"agentic"`.

### How the Agentic Loop Works Step by Step

Using the example question: *"Compare ChromaDB and Pinecone for a production RAG system"*

1. **Router classifies** → `agentic` (comparison question requiring multiple sources)

2. **Iteration 1 — agent_node:**
   - LLM receives: system prompt (document descriptions, tool list) + user question
   - LLM decides: call `compare_across_documents(topic="ChromaDB vs Pinecone comparison", document_1="vector_database_comparison.pdf", document_2="rag_architecture_patterns.pdf")`
   - → routes to `tool_node`

3. **tool_node executes:**
   - `compare_across_documents` calls `retriever.search_by_document()` for each document
   - Returns formatted context grouped by source: `## From vector_database_comparison.pdf: [Source 1: ...] ... ## From rag_architecture_patterns.pdf: [Source 1: ...] ...`
   - Tool result appended to messages

4. **Iteration 2 — agent_node:**
   - LLM reviews tool results, decides it needs more detail
   - Calls `search_by_document(query="Pinecone production scalability pricing", document_name="vector_database_comparison.pdf")`
   - → routes to `tool_node`

5. **tool_node executes:**
   - Returns additional relevant chunks about Pinecone

6. **Iteration 3 — agent_node:**
   - LLM has sufficient information
   - Generates a comprehensive comparison answer with `[vector_database_comparison.pdf, Page 5]` citations
   - No tool_calls in the response
   - `should_continue_agent()` returns `"format_agent_response"`

7. **format_agent_response:**
   - Extracts the last AI message content
   - Parses `[DocName.pdf, Page N]` citations via regex
   - Deduplicates sources
   - Assesses confidence (has sources → "high")
   - → END

### How Idempotent Ingestion Works (SHA-256 Hashing)

The ingestion pipeline uses a two-level idempotency mechanism:

**Level 1 — Document-level change detection:**
```python
# In pipeline.py: _ingest_document()
doc_hash = compute_file_hash(pdf_path)                    # SHA-256 of file bytes
if self._vector_store.is_document_current(doc_name, doc_hash):
    # Hash matches what's stored → document unchanged → skip
    return
```

The `is_document_current()` method queries ChromaDB for any chunk with `source == doc_name` and checks if its `doc_hash` metadata matches. If the file has not changed, the entire document is skipped.

**Level 2 — Chunk-level deduplication via deterministic IDs:**
```python
# In chunker.py: chunk_documents()
chunk.metadata["chunk_id"] = f"{source}::page{page}::chunk{idx}"

# In vector_store.py: upsert_documents()
ids = [chunk.metadata.get("chunk_id", f"chunk_{i}") for i, chunk in enumerate(chunks)]
collection.upsert(ids=ids, ...)  # ChromaDB upsert = insert-or-update by ID
```

Even if the document-level check is bypassed (e.g., with `--force`), the deterministic chunk IDs mean ChromaDB will overwrite existing chunks rather than creating duplicates.

**Update flow:** When a PDF file changes (different SHA-256):
1. `is_document_current()` returns `False`
2. `delete_by_document(doc_name)` removes all old chunks
3. New chunks are loaded, chunked, and upserted with the new hash

### How Content-Aware Chunking Preserves Table/Image Structure

Traditional text splitters blindly split at character boundaries, which destroys table structure and separates figure captions from their descriptions. This system uses content-type-aware chunking:

**During loading (`loader.py`):**
- Azure Document Intelligence identifies content types in the PDF
- Tables are extracted as complete markdown tables with headers and separators
- Figures are extracted as bounding-region crops, rendered to PNG, and summarized by GPT-4o vision
- Each `Document` object carries `content_type` metadata: `"text"`, `"table"`, or `"image"`

**During chunking (`chunker.py`):**
```python
for doc in documents:
    content_type = doc.metadata.get("content_type", "text")
    if content_type in ("table", "image"):
        passthrough_docs.append(doc)  # Never split
    else:
        text_docs.append(doc)         # Split normally
```

- **Text** is split with `RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)` using a hierarchy of separators: section headings → paragraphs → lines → sentences → words.
- **Tables** pass through as single chunks regardless of size. A 2000-character table stays as one chunk.
- **Images** pass through as single chunks (GPT-4o summaries are typically under 500 characters).

This means a search for "Q3 revenue" can match an entire table row rather than a fragment of a broken table.

---

## Key Implementation Details

### Azure Document Intelligence Integration (Dual-Parser with Fallback)

The system uses Azure Document Intelligence as the primary PDF parser, with pypdf as an automatic fallback:

```python
# In config.py: get_settings()
if settings.parser_backend == "azure_di" and (
    not settings.documentintelligence_endpoint or not settings.documentintelligence_api_key
):
    settings.parser_backend = "pypdf"  # Auto-downgrade if credentials missing

# In loader.py: load_pdf()
if s.parser_backend == "azure_di":
    try:
        return load_pdf_azure_di(...)
    except Exception:
        return load_pdf_pypdf(file_path)  # Fallback on any Azure DI failure
```

Azure DI (`prebuilt-layout` model) provides:
- Structured markdown output with paragraph roles (title, sectionHeading, pageHeader, pageFooter)
- Table extraction with cell-level data (row/column indices, content, bounding regions)
- Figure detection with bounding polygons
- Character-level span tracking for deduplication

pypdf provides:
- Raw text extraction per page
- No structure awareness (tables become flattened text)
- No image extraction
- Works offline, no API key needed

### Content-Aware Chunking (Text Split, Tables Whole, Images Vision-Summarized)

Three distinct processing paths based on `content_type`:

1. **Text (`content_type="text"`)**: Split using `RecursiveCharacterTextSplitter` with custom separators that prioritize section boundaries. Chunk size: 800 characters, overlap: 200 characters.

2. **Tables (`content_type="table"`)**: Kept whole as a single chunk. The markdown table format is preserved, including headers, separators, and all rows. Table captions are prepended as bold text.

3. **Images (`content_type="image"`)**: Each figure is:
   - Extracted from the PDF page using PyMuPDF (`fitz`) with the bounding polygon from Azure DI
   - Rendered at 150 DPI as PNG
   - Base64-encoded
   - Sent to GPT-4o vision with the prompt: "Describe this figure from a technical document in detail. Include all data points, labels, relationships, and key takeaways."
   - The vision summary becomes the chunk's `page_content`, making it searchable via text embeddings

### Deterministic Chunk IDs for Idempotency

Every chunk gets a deterministic ID computed from its position in the document:

```python
chunk.metadata["chunk_id"] = f"{source}::page{page}::chunk{idx}"
# Example: "rag_architecture_patterns.pdf::page3::chunk12"
```

This ID is used as the primary key in ChromaDB's `upsert()` operation. If the same document is ingested twice with the same content:
- The same chunks are produced with the same IDs
- ChromaDB's `upsert` overwrites in place instead of creating duplicates
- The collection size remains unchanged

### LangGraph StateGraph with Conditional Edges

The workflow is implemented as a compiled LangGraph `StateGraph`:

```python
graph = StateGraph(AgentState)

graph.add_node("router_node", router_node)
graph.add_node("simple_rag_node", simple_rag_node)
graph.add_node("agent_node", agent_node)
graph.add_node("tool_node", ToolNode(tools))
graph.add_node("format_agent_response", format_agent_response)

graph.add_edge(START, "router_node")
graph.add_conditional_edges("router_node", route_after_classification, {...})
graph.add_edge("simple_rag_node", END)
graph.add_conditional_edges("agent_node", should_continue_agent, {...})
graph.add_edge("tool_node", "agent_node")
graph.add_edge("format_agent_response", END)

compiled = graph.compile()
```

**Why LangGraph over plain LangChain agents:**
- Explicit state management via `TypedDict`
- Conditional routing with named edges (debuggable)
- Iteration limits enforced at the graph level
- Clean separation of concerns: each node is a pure function `AgentState → dict`
- The `add_messages` annotation handles LangGraph's message accumulation automatically

### LCEL RAG Chain with Citation Parsing

The simple RAG chain uses LangChain Expression Language:

```python
chain = ChatPromptTemplate.from_messages([
    ("system", RAG_SYSTEM_PROMPT),
    ("human", RAG_USER_PROMPT),
]) | llm | StrOutputParser()
```

The prompt instructs the LLM to cite sources as `[DocumentName, Page N]`. After the agentic workflow, citations are parsed back out via regex:

```python
source_pattern = r"\[([^,\]]+\.pdf),\s*Page\s*(\d+)\]"
found_sources = re.findall(source_pattern, final_text)
```

This creates a round-trip: context is formatted with source markers → LLM includes them in the answer → regex extracts them as structured `SourceReference` objects.

### Structured Output for Router Classification

The router uses LangChain's `with_structured_output()` to force the LLM to return a validated Pydantic model:

```python
class RouteDecision(BaseModel):
    route: Literal["simple_rag", "agentic"]
    reasoning: str

structured_llm = llm.with_structured_output(RouteDecision)
decision = chain.invoke({"question": question})
# decision.route is guaranteed to be "simple_rag" or "agentic"
```

Under the hood, this uses OpenAI's function calling / tool use to constrain the output to the schema. No string parsing is needed.

### ChromaDB with PersistentClient and Metadata Filtering

The vector store uses `chromadb.PersistentClient` for disk-backed persistence:

```python
self._client = chromadb.PersistentClient(path=self._persist_dir)
```

Data survives process restarts. The Docker Compose configuration maps this to a named volume (`chromadb_data`) for container persistence.

**Metadata filtering** is used for document-scoped search:

```python
results = collection.get(where={"source": doc_name})  # Get by document
results = store.similarity_search(query, filter_metadata={"source": "doc.pdf"})  # Search within document
```

### Error Handling and Graceful Degradation Patterns

The system degrades gracefully at multiple levels:

| Failure | Degradation |
|---|---|
| Azure DI credentials missing | Auto-downgrade to pypdf parser |
| Azure DI API call fails | Catch exception, fall back to pypdf for that document |
| Azure OpenAI vision fails | Image summary falls back to `[Figure: caption]` placeholder |
| Router classification fails | Default to `agentic` (safer, more thorough) |
| Agent hits iteration limit | Force `format_agent_response` with whatever info gathered so far |
| No relevant chunks found | Return low-confidence answer stating "don't have enough information" |
| Single document ingestion fails | Log error, continue with remaining documents |
| API global exception | Return structured JSON error, not a stack trace |

### Test Architecture (Fixtures, Mocks, Offline HuggingFace Embeddings)

The test suite is designed to run without any API keys or network access:

**Environment isolation** (`conftest.py`):
- `EMBEDDING_PROVIDER=huggingface` → uses local `all-MiniLM-L6-v2` model, no API calls
- `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` → prevents model downloads during tests
- Fake Azure credentials prevent accidental real API calls

**Fixture hierarchy:**
```
corpus_dir → sample_pdf → (used by test_ingestion, test_retrieval)
temp_chroma_dir → test_settings → loaded_store → retriever (used by test_retrieval)
mock_llm (used by test_rag)
client (TestClient, used by test_api)
```

**Mocking strategy:**
- Retrieval tests use **real embeddings** (HuggingFace) and **real ChromaDB** (in temp directory) with **real PDFs** for end-to-end retrieval validation.
- RAG and API tests use `MagicMock` and `@patch` to mock LLM calls, testing logic without API costs.
- Agent tests verify tool definitions and graph compilation without invoking LLMs.

---

## Running the Project

### Prerequisites

- **Python 3.11+** (required by `pyproject.toml`)
- **Azure OpenAI** API access with a deployed GPT-4o model and text-embedding-3-small
- **Azure Document Intelligence** (optional — falls back to pypdf)
- **Docker and Docker Compose** (optional — for containerized deployment)

### Environment Setup

```bash
# 1. Clone the repository
git clone <repo-url>
cd RAG_PRD_SAMPLE

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env with your credentials:
#   AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
#   AZURE_OPENAI_API_KEY=your-key
#   AZURE_OPENAI_DEPLOYMENT=gpt-4o
#   AZURE_OPENAI_DEPLOYMENT_EMBED=text-embedding-3-small
#   (Optional) DOCUMENTINTELLIGENCE_ENDPOINT and KEY for Azure DI
#   (Optional) EMBEDDING_PROVIDER=huggingface for local embeddings

# 5. Place PDF documents in corpus/
ls corpus/
# rag_architecture_patterns.pdf
# vector_database_comparison.pdf
# agentic_ai_frameworks.pdf
```

### Ingestion Command and What It Outputs

```bash
python -m scripts.ingest
```

**Sample output:**
```
============================================================
Knowledge Base — Document Ingestion
============================================================
  Corpus directory : /path/to/RAG_PRD_SAMPLE/corpus
  ChromaDB storage : /path/to/RAG_PRD_SAMPLE/data/chromadb
  Embedding model  : azure_openai/text-embedding-3-small
  Chunk size       : 800 (overlap: 200)
============================================================

────────────────────────────────────
Ingestion Report:
  Documents processed : 3
  Documents skipped   : 0
  Chunks created      : 87
  Errors              : 0
────────────────────────────────────

Collection 'knowledge_base':
  Total chunks : 87
  Documents    : agentic_ai_frameworks.pdf, rag_architecture_patterns.pdf, vector_database_comparison.pdf
```

**Re-running is safe:**
```bash
python -m scripts.ingest
# Documents skipped: 3 (all unchanged)
# Chunks created: 0
```

**Force re-ingestion:**
```bash
python -m scripts.ingest --force
# Clears collection, re-processes all documents
```

### Starting the API and UI

**Terminal 1 — FastAPI:**
```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
# API docs at http://localhost:8000/docs
```

**Terminal 2 — Streamlit:**
```bash
streamlit run ui/app.py --server.port 8501
# UI at http://localhost:8501
```

**Or use the startup script (runs both):**
```bash
./start.sh
# API at http://localhost:8000/docs
# UI at http://localhost:8501
```

### Docker Deployment

```bash
# Build and run everything in one command
docker-compose up --build

# Access:
#   API:  http://localhost:8000/docs
#   UI:   http://localhost:8501

# The container automatically:
# 1. Runs ingestion (idempotent)
# 2. Starts FastAPI
# 3. Starts Streamlit

# Data persists in the chromadb_data Docker volume
# Update PDFs by modifying files in ./corpus/ (bind-mounted read-only)
# Then restart: docker-compose restart (re-ingestion picks up changes)
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific modules
pytest tests/test_ingestion.py -v    # PDF loading + chunking
pytest tests/test_retrieval.py -v    # Vector store + retriever
pytest tests/test_rag.py -v          # RAG chain + prompts
pytest tests/test_agent.py -v        # Router + tools + graph
pytest tests/test_api.py -v          # FastAPI endpoints

# Tests use HuggingFace embeddings (offline, no API key needed)
# Tests use real ChromaDB in temp directories (auto-cleaned)
# Tests mock LLM calls where needed
```

### Running Evaluation

```bash
# Requires a running knowledge base (run ingestion first) and AZURE_OPENAI_API_KEY
python -m scripts.evaluate

# Custom eval set or output:
python -m scripts.evaluate --eval-set configs/eval_questions.json --output results.json
```

This runs all 15 evaluation questions through the live system and reports routing accuracy, source recall, content recall, graceful refusal rate, and latency.

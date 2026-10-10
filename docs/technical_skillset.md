# Intelligent Knowledge Assistant — Technical Skillset Guide

> **Purpose**: Comprehensive reference of every technology used in this project, with explanations, usage details, and interview preparation Q&A.

---

## 1. LLM & AI Frameworks

### 1.1 LangChain

**What it is**: An open-source framework for building applications with LLMs. Provides abstractions for chains, prompts, document loaders, text splitters, retrievers, and output parsers.

**How we use it**:
- **LCEL (LangChain Expression Language)** — our RAG chain is composed declaratively with the pipe `|` operator: `retriever | format_docs | prompt | llm | parser`. Each component is a `Runnable` with `.invoke()`, `.ainvoke()`, and `.stream()`.
- **RecursiveCharacterTextSplitter** — splits text chunks with markdown-aware separators: `["\n## ", "\n\n", "\n", ". ", " ", ""]`. Tries the largest separator first, falls back to smaller ones.
- **Prompt Templates** — `ChatPromptTemplate.from_messages()` creates system + user message pairs with `{context}` and `{question}` placeholders.
- **Document model** — `langchain_core.documents.Document` with `page_content` and `metadata` dict is used throughout ingestion, retrieval, and chain.

**Key concepts**:
- **Runnables**: composable units with input/output types; can be chained, parallelized, or branched
- **RunnablePassthrough**: passes input through unchanged; used to carry the question alongside retrieved docs
- **RunnableLambda**: wraps a plain function as a Runnable

**Interview Q&A**:

**Q: What is LCEL and why use it over legacy chains?**
A: LCEL is LangChain's declarative composition syntax using the `|` operator. Each step is a Runnable with typed input/output. Unlike legacy `LLMChain`/`SequentialChain`, LCEL chains support streaming, async, and batch natively. They're also inspectable — you can see every step's schema.

**Q: How does RecursiveCharacterTextSplitter work?**
A: It tries to split text at the largest separator first (e.g., `\n## ` for markdown headings). If the resulting chunk exceeds `chunk_size`, it recursively tries smaller separators. This preserves document structure — headings and paragraphs stay intact when possible. Overlap is applied between consecutive chunks.

**Q: What's the difference between `invoke()`, `ainvoke()`, and `stream()`?**
A: `invoke()` is synchronous — call and wait. `ainvoke()` is async — returns a coroutine for concurrent execution. `stream()` yields output tokens incrementally, enabling real-time display. All Runnables support all three.

**Alternatives**: LlamaIndex (more index-focused), Haystack (pipeline-based), raw OpenAI SDK (no abstractions).

---

### 1.2 LangGraph

**What it is**: A library for building stateful, multi-step agent workflows as directed graphs. Built on top of LangChain but provides explicit state management and conditional routing.

**How we use it**:
- **StateGraph** — our agent workflow is a `StateGraph(AgentState)` where `AgentState` is a `TypedDict` with: `messages`, `question`, `route`, `reasoning_steps`, `iteration_count`, `final_answer`, `sources`.
- **Nodes**: `router_node` (classifies question), `simple_rag_node` (LCEL chain), `agent_node` (LLM with tools), `tool_node` (executes tool calls), `format_response` (structures output).
- **Conditional edges**: `should_continue()` checks if agent has a final answer or exceeded max iterations; `route_question()` reads the classified route and branches.
- **Compilation**: `graph.compile()` produces an executable that takes initial state and runs to `END`.

**Key concepts**:
- **State**: a `TypedDict` that flows through nodes; each node reads and returns updates to it
- **Conditional edges**: functions that inspect state and return the name of the next node
- **Channels**: how state updates are merged (default: overwrite; `messages` uses `add_messages` to append)
- **ReAct loop**: agent → should_continue → tool → agent cycle

**Interview Q&A**:

**Q: Why LangGraph over AgentExecutor?**
A: AgentExecutor is a black-box loop — you can't inspect state, add custom routing, or control iteration. LangGraph makes the state machine explicit: every node, edge, and state field is visible and testable. AgentExecutor was deprecated in favor of LangGraph.

**Q: How do you handle state in LangGraph?**
A: State is a `TypedDict` passed through the graph. Each node function receives the full state and returns a dict of updates. For the `messages` field, I use `add_messages` annotation so new messages append rather than overwrite. Other fields like `iteration_count` are overwritten.

**Q: What happens if a node throws an exception?**
A: In our graph, tool execution is wrapped in try/catch — tool errors become error messages in the conversation history rather than crashing the graph. The agent then sees the error and can try a different approach.

**Alternatives**: LangChain AgentExecutor (deprecated), CrewAI (multi-agent, higher abstraction), AutoGen (Microsoft, multi-agent).

---

### 1.3 Azure OpenAI

**What it is**: Microsoft's managed deployment of OpenAI models within Azure's cloud infrastructure. Provides the same models (GPT-4o, embeddings) with enterprise security, compliance, and regional data residency.

**How we use it**:
- **Chat completions** — `AzureChatOpenAI` from `langchain-openai` for RAG generation, routing, and agent reasoning
- **Embeddings** — `AzureOpenAIEmbeddings` for document and query embedding with `text-embedding-3-small`
- **Vision** — `AzureOpenAI` from the `openai` SDK for GPT-4o vision calls to summarize extracted figures
- **Structured output** — `.with_structured_output(RouteDecision)` for type-safe router classification
- **Configuration** — endpoint, deployment name, API key, and API version in `pydantic-settings`

**Key concepts**:
- **Deployments**: named model instances (not model strings); you deploy `gpt-4o` as deployment `gpt-4o`
- **API versions**: explicit versioning (`2024-08-01-preview`) — different versions expose different features
- **Difference from direct OpenAI**: Azure adds deployment names, explicit endpoints, VNet support, DPA, no training on your data

**Interview Q&A**:

**Q: What parameters does AzureChatOpenAI need that ChatOpenAI doesn't?**
A: `azure_endpoint` (the resource URL), `azure_deployment` (deployment name), and `api_version`. Direct OpenAI just needs `model` and `api_key`.

**Q: What is structured output and how does it work?**
A: `.with_structured_output(PydanticModel)` constrains the LLM to produce JSON matching the model's schema. Under the hood, it sets `response_format` with a JSON schema derived from Pydantic. The response is parsed and validated automatically. This is more reliable than regex-parsing free text.

**Q: How do you handle API version differences?**
A: I have both `azure_openai_api_version` and `openai_api_version` in settings with a property `effective_api_version` that picks the right one. This handles cases where different Azure services expect different version formats.

---

### 1.4 Prompt Engineering

**What it is**: The craft of designing LLM instructions to produce reliable, grounded, and well-formatted outputs.

**How we use it** (all prompts centralized in `src/rag/prompts.py`):
- **RAG system prompt**: instructs grounded answers, citation format `[DocName, Page N]`, explicit refusal when context is insufficient
- **RAG user prompt**: `{context}` and `{question}` template
- **Router prompt**: classification criteria with examples — simple factual → `simple_rag`, comparison/multi-step → `agentic`
- **Agent system prompt**: ReAct instructions, available tools with descriptions, iteration awareness, formatting rules

**Key concepts**:
- **Grounding**: "Answer ONLY based on the provided context" — prevents hallucination
- **Citation format**: forces verifiable references by specifying exact format
- **Refusal pattern**: explicit instruction to say "I don't know" when context is insufficient
- **Few-shot examples**: router prompt includes example classifications for consistent routing

**Interview Q&A**:

**Q: How do you prevent hallucination through prompting?**
A: Three techniques: (1) grounding instruction — "answer only from provided context," (2) citation requirement — forces the LLM to point at specific sources, (3) explicit refusal instruction — "if context is insufficient, say so." These work together to constrain generation.

**Q: Why centralize prompts in a separate file?**
A: Prompts are the most-changed part of a RAG system. Centralizing them makes them version-controlled, testable, and reviewable without touching business logic. You can A/B test prompts by swapping the import.

---

## 2. Vector Databases & Embeddings

### 2.1 ChromaDB

**What it is**: An open-source embedding database with persistent local storage. Uses HNSW (Hierarchical Navigable Small World) indexing for approximate nearest neighbor search.

**How we use it**:
- **PersistentClient** — stores data on disk at `./data/chromadb`, survives restarts
- **Collections** — single collection `knowledge_base` holding all document chunks
- **Upsert** — `collection.upsert(ids, documents, embeddings, metadatas)` for idempotent writes
- **Similarity search** — `similarity_search_with_relevance_scores(query, k=5)` returns (Document, score) tuples
- **Metadata filtering** — `where={"source": "document.pdf"}` for document-specific search
- **VectorStoreManager** — wrapper class in `src/retrieval/vector_store.py` providing `upsert()`, `similarity_search()`, `delete_by_document()`, `get_collection_stats()`

**Key concepts**:
- **HNSW index**: multi-layer graph for O(log N) approximate nearest neighbor — trades small recall loss for massive speed gain
- **Cosine similarity**: measures directional alignment of vectors (0 to 1 after normalization)
- **Metadata**: arbitrary key-value pairs stored alongside vectors for filtering
- **Persistence**: SQLite for metadata + HNSW binary files on disk

**Interview Q&A**:

**Q: How does HNSW work at a high level?**
A: HNSW builds a multi-layer proximity graph. Bottom layer has all points; each upper layer has progressively fewer. Search starts at the top (fewest points, coarsest navigation), finds the nearest node, then descends to the next layer for finer search. This gives O(log N) query time vs O(N) for brute force.

**Q: What's the difference between `add` and `upsert` in ChromaDB?**
A: `add` fails if an ID already exists. `upsert` inserts if new, updates if existing. We use `upsert` exclusively for idempotent ingestion — re-running ingestion with the same deterministic IDs is safe.

**Q: What are ChromaDB's limitations?**
A: Single-node only (no replication), limited to ~1M vectors for good performance, no built-in access control. For production scale, migrate to Pinecone or Weaviate. Note: ChromaDB lacks native hybrid search (BM25 + dense), so we implemented BM25 as a separate index on top via `rank-bm25` with Reciprocal Rank Fusion.

---

### 2.2 Embeddings

**What it is**: Dense vector representations of text that capture semantic meaning. Similar texts have similar vectors (high cosine similarity).

**How we use it**:
- **Production**: Azure OpenAI `text-embedding-3-small` (1536 dimensions) — good quality/cost ratio, hosted in Azure tenant
- **Testing**: HuggingFace `all-MiniLM-L6-v2` (384 dimensions) — runs locally, no API needed, cached at `~/.cache/huggingface/`
- **Provider switch**: `EMBEDDING_PROVIDER` config controls which model is used; `get_embeddings()` factory returns the configured instance

**Key concepts**:
- **Dimensionality**: number of values in each vector — higher = more nuance but more storage/compute
- **Model consistency**: MUST use the same model for indexing and querying — cannot mix dimensions
- **Normalization**: some models output normalized vectors (unit length); cosine similarity on normalized vectors equals dot product

**Interview Q&A**:

**Q: Why can't you mix embedding models in one collection?**
A: Each model maps text to a different vector space with different dimensions and semantic geometry. A query embedded with model A lives in a completely different space than documents embedded with model B — cosine similarity between them is meaningless.

**Q: Why text-embedding-3-small over text-embedding-3-large?**
A: 3-small (1536d) gives 90%+ of 3-large's (3072d) quality at half the dimensions and lower cost. For our corpus size, the quality difference is negligible, but the cost and storage savings are meaningful.

---

### 2.3 Hybrid Search (BM25 + Dense + Reciprocal Rank Fusion)

**What it is**: A retrieval strategy that combines dense vector search (semantic similarity) with BM25 sparse search (keyword/term-frequency matching) and fuses results using Reciprocal Rank Fusion.

**How we use it**:
- **Dense search**: ChromaDB cosine similarity via `text-embedding-3-small` embeddings — captures semantic meaning ("machine learning" matches "neural network training")
- **BM25 search**: `BM25Okapi` from `rank-bm25` scores all documents by term frequency — excels at exact keyword matches ("HNSW", version numbers, acronyms)
- **Reciprocal Rank Fusion**: `score(d) = Σ (weight / (rank + 60))` — merges both ranked lists using only rank positions (score-agnostic), deduplicates by `chunk_id`, weights dense at 0.6 and BM25 at 0.4
- **BM25 index**: built lazily from `VectorStoreManager.get_all_documents()`, cached after first build, tokenized with whitespace + lowercasing

**Key concepts**:
- **Bi-encoder vs sparse**: dense embeddings capture meaning but may miss exact terms; BM25 captures exact terms but misses paraphrases. Together they cover both failure modes.
- **RRF constant k=60**: from the Cormack et al. paper — smooths the rank contribution curve so rank 1 isn't overwhelmingly dominant over rank 5
- **Score-agnostic fusion**: RRF uses rank positions only, avoiding the normalization problem of combining cosine similarity (0-1) with BM25 scores (unbounded)
- **Over-fetching**: both retrievers fetch k×2 candidates; fusion and re-ranking narrow to final k

**Interview Q&A**:

**Q: Why not just use BM25 or just use dense search?**
A: Dense search misses exact keyword matches — "HNSW" as an acronym may not embed close to documents about hierarchical navigable small world graphs. BM25 misses semantic connections — "retrieval augmented generation" won't match "RAG" well. Hybrid combines both: dense provides semantic understanding, BM25 provides keyword precision. The RRF fusion boosts documents found by both retrievers.

**Q: Why Reciprocal Rank Fusion over other fusion methods?**
A: RRF is score-agnostic — it uses only rank positions. This matters because BM25 scores and cosine similarities live on incompatible scales. CombSUM (summing normalized scores) requires careful per-distribution normalization. Linear interpolation requires tuning weight parameters. RRF works out of the box with any pair of rankers.

**Q: How is the BM25 index managed?**
A: Built lazily on first hybrid search call by fetching all documents from ChromaDB via `get_all_documents()`. Cached in memory for subsequent queries. If a source filter is applied, a filtered index is built for that specific document. The index uses simple whitespace + lowercasing tokenization — sufficient for technical English content.

**Alternatives**: Weaviate has native hybrid search built in; Elasticsearch/OpenSearch provide BM25 natively alongside vector search. Our approach layers BM25 on top of ChromaDB, avoiding a vector store migration.

---

### 2.4 Cross-Encoder Re-Ranking

**What it is**: A second-stage retrieval model that re-scores (query, document) pairs using a cross-attention transformer for higher accuracy than bi-encoder similarity.

**How we use it**:
- **Model**: `cross-encoder/ms-marco-MiniLM-L-6-v2` from sentence-transformers — trained on MS MARCO passage ranking dataset
- **Two-stage pipeline**: bi-encoder + BM25 retrieve a broad candidate pool (k×2), then the cross-encoder re-ranks to select the final top-k
- **Lazy loading**: model loaded on first use, cached for subsequent queries
- **Offline mode**: loaded with `local_files_only=True` to use cached model weights without network access (critical in corporate proxy environments)
- **Graceful degradation**: if the model isn't available locally, re-ranking is disabled and retrieval falls back to bi-encoder/RRF scores with a logged warning
- **Score normalization**: cross-encoder outputs arbitrary logits, so we normalize to 0-1 range using min-max scaling

**Key concepts**:
- **Bi-encoder vs cross-encoder**: bi-encoders embed query and document independently (fast, O(1) per pre-embedded doc). Cross-encoders process the concatenated pair through a single transformer with full cross-attention (accurate, O(N) per query). The two-stage approach gets the best of both.
- **MS MARCO**: Microsoft's large-scale passage ranking dataset — the standard benchmark for training retrieval models
- **MiniLM**: a distilled transformer that's much smaller/faster than BERT-large while retaining most accuracy — ideal for a re-ranking stage where you process dozens of pairs, not millions

**Interview Q&A**:

**Q: Why not use the cross-encoder for all retrieval?**
A: Cross-encoders can't pre-compute embeddings — they must process every (query, document) pair at query time. With 130 chunks, that's 130 transformer forward passes per query. With 10K chunks, it's 10K passes — far too slow. The bi-encoder narrows candidates to ~10, making cross-encoder scoring feasible.

**Q: How does the cross-encoder improve results in practice?**
A: The bi-encoder embeds query and document independently — it can't attend to specific query terms within the document. The cross-encoder sees both together, so it can recognize that "What indexing algorithm does Pinecone use?" is highly relevant to a passage containing "Pinecone uses a proprietary HNSW variant" even if the surface-level vocabulary differs. In our system, re-ranking typically reorders 2-3 positions in the top-5, which meaningfully improves answer quality.

**Q: How did you handle the corporate proxy SSL issue?**
A: The `sentence-transformers` library tries to check HuggingFace Hub for model updates even when the model is cached locally. Behind a corporate proxy, this fails with `SSL: CERTIFICATE_VERIFY_FAILED`. I fixed it by passing `local_files_only=True` to the `CrossEncoder` constructor, which skips the network check entirely. If the model isn't cached, it logs a warning and disables re-ranking gracefully.

---

## 3. Azure Services

### 3.1 Azure Document Intelligence

**What it is**: A cloud AI service that extracts text, tables, figures, and structure from documents. Formerly called Azure Form Recognizer.

**How we use it**:
- **Client**: `DocumentIntelligenceClient` with `AzureKeyCredential` authentication
- **Model**: `prebuilt-layout` — extracts paragraphs, tables, figures, selection marks with spatial information
- **Output format**: `DocumentContentFormat.MARKDOWN` — returns full document as structured markdown
- **Table extraction**: `result.tables` gives structured cells with row/column indices; we reconstruct markdown tables
- **Figure extraction**: `result.figures` gives bounding polygons; we convert to PyMuPDF coordinates for pixel extraction
- **Span tracking**: paragraphs, tables, and figures have character spans (offset + length) — we track these to avoid duplicating content between text and table chunks

**Key concepts**:
- **Prebuilt models**: `prebuilt-layout` (structure), `prebuilt-read` (text only), `prebuilt-document` (key-value pairs)
- **Bounding regions**: polygon coordinates in inches from page top-left
- **Paragraph roles**: `title`, `sectionHeading`, `pageHeader`, `pageFooter`, `pageNumber` — we filter headers/footers and promote headings with `## ` prefix
- **Analyze operation**: async — `begin_analyze_document()` returns a poller; `.result()` blocks until complete

**Interview Q&A**:

**Q: Why prebuilt-layout over prebuilt-read?**
A: `prebuilt-read` extracts text only — no table structure, no figure regions. `prebuilt-layout` understands document structure: it identifies table boundaries with cell-level granularity, detects figures with bounding boxes, and classifies paragraphs by role. This structural information is what enables content-aware chunking.

**Q: How do you handle the span overlap between paragraphs and tables?**
A: Tables and figures have character span ranges (offset, length) in the full markdown. I collect all table and figure spans into a set, then when processing paragraphs, I check if each paragraph's span overlaps any excluded span. Overlapping paragraphs are skipped since their content is already captured in the table/figure chunks.

**Q: What happens if the DI API returns unexpected results?**
A: Multiple fallbacks: if `result.paragraphs` is empty, I use `result.content` as a single text block. If `result.tables` is empty, we just get text chunks with no tables. If `result.figures` is empty or PyMuPDF isn't installed, figures are skipped. If the entire API call fails, we fall back to pypdf.

---

## 4. Document Processing

### 4.1 Content-Aware Chunking Pipeline

**What it is**: A processing pipeline that treats different content types differently rather than splitting everything uniformly.

**How it works end-to-end**:
1. Azure DI parses PDF → returns paragraphs, tables, figures with spans
2. **Tables** → reconstructed as markdown, stored as single chunks with `content_type: "table"`
3. **Figures** → PyMuPDF extracts bounding region as PNG → base64 encode → GPT-4o vision summarizes → summary stored as chunk with `content_type: "image"`
4. **Text** → paragraphs grouped by page (excluding table/figure spans) → split with `RecursiveCharacterTextSplitter(800, 200)` → stored with `content_type: "text"`
5. All chunks sorted by (source, page, content_type) → assigned deterministic IDs

**Key components**:
- **PyMuPDF (fitz)**: opens PDF, extracts page regions as pixmaps using `page.get_pixmap(clip=rect, dpi=150)`
- **Coordinate conversion**: Azure DI uses inches, PyMuPDF uses points (1 inch = 72 points)
- **Base64 encoding**: `base64.b64encode(png_bytes).decode("utf-8")` for vision API input

**Interview Q&A**:

**Q: Why not just split the full markdown output uniformly?**
A: A 14-row comparison table split in half loses the header row in the second chunk — it becomes incomprehensible. An image's alt-text split across chunks loses context. Content-aware chunking preserves the semantic integrity of structured elements.

**Q: How do you convert Azure DI coordinates to PyMuPDF coordinates?**
A: Azure DI returns polygon coordinates in inches from the top-left corner. PyMuPDF uses points (72 points per inch). I multiply each coordinate by 72 to convert, create a `fitz.Rect`, then clip it to the page bounds with `clip & page.rect` to handle edge cases.

---

## 5. Python Ecosystem

### 5.1 Pydantic & pydantic-settings

**What it is**: Pydantic is a data validation library using Python type hints. pydantic-settings extends it for configuration management from environment variables and `.env` files.

**How we use it**:
- **Settings** (pydantic-settings): `class Settings(BaseSettings)` with `SettingsConfigDict(env_file=".env")` — loads all config from environment variables automatically
- **Field validation**: `@field_validator("embedding_provider")` ensures only valid values
- **API schemas**: `AskRequest`, `AskResponse`, `HealthResponse` Pydantic models for FastAPI request/response validation
- **Structured output**: `RouteDecision(BaseModel)` with `route: Literal["simple_rag", "agentic"]` for LLM classification

**Interview Q&A**:

**Q: How does pydantic-settings resolve configuration values?**
A: Priority order: (1) constructor arguments, (2) environment variables, (3) `.env` file values, (4) field defaults. Environment variables are case-insensitive and matched by field name. `extra="ignore"` skips unknown env vars.

**Q: Why use Pydantic models for API schemas?**
A: FastAPI uses Pydantic for automatic request validation, serialization, and OpenAPI schema generation. An invalid request body returns a 422 with specific field errors — zero custom validation code needed.

---

### 5.2 FastAPI

**What it is**: A modern Python web framework built on Starlette (ASGI) and Pydantic, with automatic OpenAPI documentation.

**How we use it**:
- **App factory**: `create_app()` with `@asynccontextmanager` lifespan for startup initialization
- **APIRouter**: routes in `src/api/routes.py`, mounted with prefix `/api/v1`
- **Lifespan**: initializes ChromaDB, verifies corpus is ingested at startup
- **CORS**: `CORSMiddleware` allows Streamlit (port 8501) to call the API (port 8000)
- **Error handling**: global exception handler returns structured JSON for all unhandled errors
- **Endpoints**: `POST /api/v1/ask`, `GET /api/v1/health`, `POST /api/v1/ingest`, `GET /api/v1/documents`

**Interview Q&A**:

**Q: What is the lifespan context manager?**
A: It runs code at startup (before first request) and shutdown (after last request). I use it to initialize the vector store and verify the corpus is ingested. Resources created in lifespan are available to all request handlers via `app.state`.

**Q: How does FastAPI handle concurrent requests?**
A: FastAPI runs on uvicorn's async event loop. Each request is a coroutine. I/O-bound operations (LLM calls, ChromaDB queries) yield the event loop, allowing other requests to proceed. For CPU-bound work, you'd use `run_in_executor()`.

---

### 5.3 Streamlit

**What it is**: A Python framework for building data apps with minimal frontend code. Renders Python scripts as interactive web pages.

**How we use it**:
- **Chat interface**: `st.chat_message()` for user/assistant message display
- **Session state**: `st.session_state.messages` persists chat history across reruns
- **Sidebar**: document list, system health, workflow indicator
- **Expandable sections**: `st.expander()` for source citations and reasoning steps
- **API client**: `httpx.Client` calls FastAPI backend

**Interview Q&A**:

**Q: How does Streamlit's execution model work?**
A: Streamlit reruns the entire script on every user interaction. `st.session_state` persists data across reruns — without it, chat history would reset on each message. This is different from traditional frameworks where handlers execute on specific events.

---

### 5.4 pytest

**What it is**: Python's most popular testing framework, with powerful fixture and plugin system.

**How we use it**:
- **Fixtures**: `temp_chroma_dir` (tempdir), `test_settings` (overridden config), `mock_llm` (MagicMock), `corpus_dir`, `sample_pdf`
- **conftest.py**: shared fixtures; sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` for offline testing
- **TestClient**: FastAPI's test client for HTTP testing without a running server
- **Test isolation**: each test gets fresh temp directories; ChromaDB collections are isolated

**Interview Q&A**:

**Q: How do you test without Azure credentials?**
A: Tests set `EMBEDDING_PROVIDER=huggingface` to use local embeddings and don't set DI credentials (so parser falls back to pypdf). The mock LLM returns predetermined responses. The test env vars are set in `tests/__init__.py` via `os.environ.setdefault()`.

**Q: What's the purpose of `conftest.py`?**
A: It defines fixtures shared across all test modules. pytest automatically discovers it. Fixtures like `temp_chroma_dir` provide test isolation — each test gets a fresh temporary directory that's cleaned up afterward.

---

## 6. Infrastructure & DevOps

### 6.1 Docker

**What it is**: Container platform that packages applications with their dependencies for consistent deployment.

**How we use it**:
- **Base image**: `python:3.11-slim` for minimal size
- **Layer caching**: `COPY requirements.txt` and `pip install` before copying source code — dependencies are cached unless requirements change
- **Volume mounts**: `./data:/app/data` for ChromaDB persistence, `./corpus:/app/corpus` for PDF files
- **Health check**: `curl http://localhost:8000/api/v1/health` in Dockerfile

**How docker-compose works**:
- Single `app` service with both API (8000) and Streamlit (8501)
- `env_file: .env` injects secrets without baking them into the image
- `start.sh` entrypoint: runs ingestion → starts uvicorn + streamlit

**Interview Q&A**:

**Q: Why `python:3.11-slim` over `python:3.11`?**
A: The slim variant excludes development headers, documentation, and rarely-used packages — image size drops from ~900MB to ~150MB. For ML apps that add large pip packages anyway, slim is the right balance of size and compatibility.

**Q: How do you handle secrets in Docker?**
A: Secrets are in `.env` (not in the image). `docker-compose.yml` references `env_file: .env`. In production, you'd use Docker secrets, Azure Key Vault, or Kubernetes secrets instead of env files.

---

## 7. Design Patterns

### 7.1 Idempotent Operations

**Pattern**: Operations that produce the same result regardless of how many times they're executed.

**How we implement it**:
- **SHA-256 file hashing**: `compute_file_hash()` hashes the PDF bytes; unchanged files produce the same hash
- **Deterministic chunk IDs**: `{source}::page{N}::chunk{M}` — same content always gets the same ID
- **Upsert**: ChromaDB `upsert()` inserts or updates, never duplicates
- **Pipeline check**: before processing, check if document hash already exists in collection metadata

**Why it matters**: You can re-run ingestion after deployments, crashes, or additions without corrupting the knowledge base.

---

### 7.2 Graceful Degradation

**Pattern**: System continues operating with reduced functionality when a component fails.

**How we implement it**:
- **Parser**: Azure DI → pypdf fallback (config-time and runtime)
- **Embeddings**: Azure OpenAI → HuggingFace fallback (config-based)
- **Figures**: PyMuPDF missing → skip figures; vision API fails → store caption only
- **Re-ranker**: cross-encoder model unavailable locally → disable re-ranking, fall back to retrieval-score ordering
- **Router**: structured output fails → default to `simple_rag`
- **Agent**: max iterations → force format response; tool error → error message to agent

---

### 7.3 Factory Pattern

**Pattern**: Functions that create and return configured objects, hiding construction complexity.

**How we implement it**:
- `get_settings()` — cached singleton, loads `.env`, applies fallbacks
- `get_llm()` — returns configured `AzureChatOpenAI` with deployment, endpoint, key
- `get_embeddings()` — returns Azure or HuggingFace embeddings based on config

**Why it matters**: Consumers call `get_llm()` without knowing about Azure endpoints or API versions. Configuration is centralized. Testing is easy — pass mock settings to the factory.

---

## 8. RAG Architecture Concepts

### 8.1 Retrieval-Augmented Generation

**What it is**: A pattern that augments LLM generation with retrieved context from an external knowledge base. The LLM answers based on provided context rather than parametric memory alone.

**Canonical pipeline**: Query → Embed → **Hybrid Search** (Dense + BM25 → RRF Fusion) → **Cross-Encoder Re-Rank** → Threshold Filter → Inject into prompt → Generate → Return with citations

**Why it matters**:
- **Reduces hallucination**: LLM is grounded in actual documents
- **Enables knowledge updates**: swap corpus, no retraining needed
- **Provides citations**: answers are verifiable against sources
- **Domain-specific**: works with proprietary data without fine-tuning

**Interview Q&A**:

**Q: What are the failure modes of RAG?**
A: (1) **Retrieval failure** — relevant chunks aren't found (wrong query embedding, poor chunking). (2) **Context window overflow** — too many chunks dilute signal. (3) **Generation unfaithfulness** — LLM ignores context and hallucinates anyway. (4) **Citation errors** — LLM cites wrong source. Each requires different mitigation.

**Q: How do you decompose a bad answer — retrieval vs generation problem?**
A: Check the retrieved chunks. If they contain the answer but the LLM missed it → generation problem (fix prompt, temperature, model). If the chunks don't contain the answer → retrieval problem (fix chunking, embedding model, or search strategy). Log chunk IDs and scores for every request to enable this analysis.

---

### 8.2 Chunking Strategies Comparison

| Strategy | How it works | Pros | Cons |
|---|---|---|---|
| **Fixed-size** | Split every N characters | Simple, predictable | Breaks mid-sentence, ignores structure |
| **Recursive** | Try large separators first, fall back | Respects boundaries | Still splits tables/lists |
| **Semantic** | Split when embedding similarity drops | Topically coherent | Slow, unpredictable sizes |
| **Content-aware** (ours) | Classify content type, handle each differently | Preserves structure | More complex, parser-dependent |

---

### 8.3 Evaluation Concepts

**How we evaluate**:
- **Retrieval recall@5**: do the top-5 chunks contain the answer?
- **Answer faithfulness**: does the answer only use information from context? (LLM-as-judge)
- **Citation accuracy**: do citations point to chunks that support the claim?
- **Graceful refusal**: does the system refuse unanswerable questions?
- **Router accuracy**: does the router classify correctly?

**Framework reference**: RAGAS (Retrieval Augmented Generation Assessment) provides automated metrics: faithfulness, answer relevancy, context precision, context recall.

**Interview Q&A**:

**Q: How would you set up continuous evaluation?**
A: Maintain a test set of 200+ QA pairs with ground truth. Run evaluation in CI/CD on every code change. Track metrics over time. Alert on regressions. This catches chunking changes that break retrieval, prompt changes that increase hallucination, or model updates that shift behavior.

**Q: What's the difference between context precision and context recall?**
A: Context precision = what fraction of retrieved chunks are relevant (are we fetching noise?). Context recall = what fraction of relevant chunks are retrieved (are we missing information?). You want both high — precision avoids context pollution, recall ensures the answer is complete.

---

*This guide covers every technology in the project. For each one, understand not just what it does, but why we chose it and what we'd switch to at scale.*

# Intelligent Knowledge Assistant — Interview Preparation Guide

> **Purpose**: Prepare for promotion panel questions about this project. Each question includes a strong answer and what it demonstrates to the panel.

---

## 1. Architecture & Design Decisions

### Q1: Walk us through the system architecture at a high level.

**Answer**: The system follows a modular pipeline architecture with five layers. First, the **Ingestion Pipeline** processes PDFs using Azure Document Intelligence as the primary parser — it extracts structured markdown, tables, and figures separately. Text is chunked with RecursiveCharacterTextSplitter while tables are kept whole and images are summarized by GPT-4o vision. All chunks are embedded and stored in **ChromaDB** with deterministic IDs for idempotent upsert.

At query time, the **Retrieval Layer** uses a multi-stage pipeline: dense vector search runs in parallel with BM25 sparse search (hybrid mode), results are fused via **Reciprocal Rank Fusion**, then re-ranked by a **cross-encoder model** for maximum precision. An **LLM Router** classifies the question using structured output into `simple_rag` or `agentic`. Simple questions go through an **LCEL RAG chain** — retrieve → format with citations → LLM → parse. Complex questions go to a **LangGraph StateGraph** that runs a ReAct loop with 4 specialized tools and a max of 5 iterations. Everything is exposed via **FastAPI** and a **Streamlit** chat UI.

**Why impressive**: Shows end-to-end system thinking, not just component knowledge.

---

### Q2: Why Azure OpenAI instead of direct OpenAI API?

**Answer**: Three reasons: (1) **Enterprise compliance** — Azure OpenAI runs within Microsoft's Azure trust boundary, giving us SOC 2, HIPAA eligibility, and data residency guarantees that Eli Lilly requires. (2) **Network security** — Azure deployments can sit behind VNet/Private Endpoints, so our data never traverses the public internet. (3) **Data processing agreements** — Microsoft's DPA explicitly states that prompts and completions are not used for model training, which is critical for pharmaceutical IP. The trade-off is slightly more complex configuration (deployment names, API versions), but `langchain-openai`'s `AzureChatOpenAI` abstracts most of that.

**Why impressive**: Shows awareness of enterprise/regulatory requirements beyond pure engineering.

---

### Q3: Why ChromaDB over Pinecone, Weaviate, or pgvector?

**Answer**: ChromaDB was chosen for this scope because: (1) **Zero infrastructure** — `PersistentClient` stores on local disk, no separate service to manage. (2) **Good LangChain integration** — `langchain-chroma` provides native retriever support. (3) **Sufficient scale** — our corpus is 3 documents / ~130 chunks; ChromaDB handles up to ~1M vectors well. (4) **Fast iteration** — no API keys or cloud setup needed.

I'd switch to **pgvector** for early production (if we already have PostgreSQL) or **Pinecone** for scaled production (>2M vectors, need replication, SLAs). The corpus itself discusses this migration path — prototype with Chroma, early production with pgvector, scaled production with Pinecone/Weaviate.

**Why impressive**: Shows pragmatic decision-making and awareness of the scaling path.

---

### Q4: Why LangGraph over LangChain's AgentExecutor?

**Answer**: AgentExecutor is a black box — it runs a loop internally but gives you little control over state, routing, or iteration. LangGraph gives me: (1) **Explicit state management** via `TypedDict` — I can track `reasoning_steps`, `iteration_count`, `sources` across the graph. (2) **Conditional edges** — the router node decides whether to go to `simple_rag_node` or `agent_node` based on the classification. (3) **Iteration control** — I cap the ReAct loop at 5 iterations with a simple counter check in my `should_continue` function. (4) **Debuggability** — every node transition is visible and testable. AgentExecutor was deprecated in favor of LangGraph precisely because of these limitations.

**Why impressive**: Demonstrates understanding of the LangChain ecosystem evolution and why explicit state matters.

---

### Q5: Why content-aware chunking instead of uniform splitting?

**Answer**: Uniform splitting destroys structured content. If you naively split a 14-row markdown table across two chunks, neither chunk makes sense — you lose the header row in the second half, and queries about data in either half retrieve garbage. My approach classifies content by type at parse time: **text** → split with RecursiveCharacterTextSplitter (800 chars, 200 overlap), **tables** → pass through whole (a table is a single atomic chunk regardless of size), **images** → extract via PyMuPDF, summarize with GPT-4o vision, store the summary as a chunk. The chunker sorts by (source, page, content_type) then assigns deterministic IDs.

**Why impressive**: Shows deep understanding of how chunking quality directly affects retrieval quality.

---

### Q6: How does the dual-parser architecture work?

**Answer**: Azure Document Intelligence is the primary parser — it produces structured markdown with explicit table, paragraph, and figure elements. If DI credentials are missing or the API call fails, we fall back to pypdf which does raw text extraction. The fallback decision happens at two levels: (1) **Config time** — `get_settings()` checks if `DOCUMENTINTELLIGENCE_ENDPOINT` and `DOCUMENTINTELLIGENCE_API_KEY` are set; if not, it overrides `parser_backend` to `"pypdf"`. (2) **Runtime** — even if configured for Azure DI, if the API call throws any exception, the `load_pdf()` function catches it and falls back to pypdf for that specific file. This means ingestion never fails completely — you always get content, just with less structure.

**Why impressive**: Demonstrates defensive programming and graceful degradation thinking.

---

### Q7: Why deterministic chunk IDs instead of UUIDs?

**Answer**: Deterministic IDs (format: `{source}::page{N}::chunk{M}`) enable idempotent upsert. If I re-ingest the same document and the content hasn't changed, ChromaDB's `upsert()` with the same ID is a no-op — no duplicate chunks. With UUIDs, every re-ingestion creates new chunks alongside the old ones. This is critical for a production system where you might re-run ingestion after a deployment or add new documents to an existing corpus. Combined with SHA-256 file hashing for change detection, unchanged documents are skipped entirely.

**Why impressive**: Shows production thinking — idempotency is a key principle for reliable data pipelines.

---

### Q8: How does the LLM Router work?

**Answer**: The router uses `AzureChatOpenAI.with_structured_output(RouteDecision)` where `RouteDecision` is a Pydantic model with `route: Literal["simple_rag", "agentic"]` and `reasoning: str`. The prompt includes classification criteria: simple factual questions → `simple_rag`; comparisons, multi-step reasoning, cross-document synthesis → `agentic`. The LLM returns a JSON object that Pydantic validates, so we get guaranteed type-safe routing. If structured output parsing fails, we default to `simple_rag` as a safe fallback.

**Why impressive**: Shows use of structured output for reliable control flow, not just text generation.

---

### Q9: What's your error handling strategy?

**Answer**: Defense in depth at every layer: (1) **Ingestion** — per-file try/catch so one bad PDF doesn't stop the batch; Azure DI failure falls back to pypdf. (2) **Retrieval** — empty results return a "no relevant information found" response instead of crashing. (3) **Router** — parsing failures default to `simple_rag`. (4) **Agent** — max 5 iterations prevents infinite loops; tool exceptions are caught and returned as error messages to the agent. (5) **API** — global exception handler returns structured JSON errors; CORS middleware prevents cross-origin issues. (6) **Config** — missing credentials trigger warnings and automatic fallback (DI → pypdf, Azure embeddings → HuggingFace in tests).

**Why impressive**: Shows systematic thinking about failure modes across the entire stack.

---

### Q10: What trade-offs did you make and why?

**Answer**: (1) **ChromaDB over Pinecone** — traded scalability for simplicity; right for this scope but I'd revisit at scale. (2) **LLM router over rules** — traded latency (extra LLM call) for accuracy on ambiguous queries. (3) **Tables kept whole** — traded chunk size consistency for semantic integrity; a 1000-char table is one chunk even though my target is 800. (4) **Vision summarization** — traded ingestion speed for figure searchability; each figure adds ~3s for the GPT-4o call. (5) **Pypdf fallback** — traded structure awareness for availability; pypdf gives flat text but always works.

**Why impressive**: Shows ability to articulate trade-offs explicitly — a key senior engineering skill.

---

### Q11: How would you scale this system for production?

**Answer**: Several dimensions: (1) **Vector store** — migrate from ChromaDB to Pinecone or Weaviate for horizontal scaling and replication. (2) **Ingestion** — add async/parallel processing with Celery or Azure Functions for concurrent PDF processing. (3) **API** — deploy multiple FastAPI instances behind a load balancer; the app is stateless since ChromaDB is external. (4) **Caching** — add Redis for frequent query caching and embedding cache. (5) **Monitoring** — integrate LangSmith for LLM tracing, Prometheus/Grafana for API metrics. (6) **Multi-tenancy** — add tenant metadata to chunks and enforce tenant-level filtering at query time.

**Why impressive**: Shows production readiness thinking beyond the prototype.

---

### Q12: How do you handle secrets management?

**Answer**: All secrets are loaded from environment variables via `pydantic-settings`. The `Settings` class reads from a `.env` file (configured in `SettingsConfigDict`). The `.env` file is in `.gitignore` so it's never committed. For deployment, Docker Compose references the `.env` file, and in production you'd use Azure Key Vault or Kubernetes secrets. The `.env.example` file shows the variable names without values. No secrets appear anywhere in the codebase.

**Why impressive**: Shows security awareness and proper secrets management patterns.

---

### Q13: Why FastAPI over Flask?

**Answer**: (1) **Async-native** — FastAPI runs on Starlette/uvicorn with native async support, important since LLM calls are I/O-bound. (2) **Auto-documentation** — Swagger UI at `/docs` is generated from Pydantic schemas. (3) **Pydantic integration** — request/response validation is built-in, matching our Settings and schema models. (4) **Lifespan management** — FastAPI's `@asynccontextmanager` lifespan lets me initialize ChromaDB and verify the corpus at startup. Flask would need extensions for all of these.

**Why impressive**: Shows framework selection based on concrete technical requirements.

---

### Q14: How did you design the evaluation framework?

**Answer**: I created 15 evaluation questions across 4 categories: 5 simple factual (single-doc RAG), 5 cross-document comparisons (agentic), 3 ambiguous queries, and 2 unanswerable questions. The evaluation script measures: retrieval recall@5, answer faithfulness via LLM-as-judge, citation accuracy (do cited pages contain the claimed information), and graceful refusal rate for out-of-scope questions. This covers both retrieval quality and generation quality independently.

**Why impressive**: Shows systematic evaluation thinking, not just "it works."

---

### Q15: How do you ensure the system doesn't hallucinate?

**Answer**: Multiple layers: (1) **Grounded prompts** — the system prompt explicitly says "Answer ONLY based on the provided context." (2) **Citation requirement** — the prompt requires `[DocName, Page N]` format citations, forcing the LLM to point at specific sources. (3) **Confidence scoring** — the chain returns high/medium/low confidence based on retrieval scores and answer quality. (4) **Refusal pattern** — when context is insufficient, the system says so rather than guessing. (5) **Similarity threshold** — chunks below 0.3 cosine similarity are filtered out before reaching the LLM, preventing irrelevant context from inducing hallucination.

**Why impressive**: Shows systematic anti-hallucination strategy, not just prompt engineering.

---

## 2. RAG Deep Dive

### Q16: Explain the RAG pipeline end to end.

**Answer**: **Ingestion**: PDF → Azure DI parses to structured markdown → content classifier separates text/tables/images → text chunks via RecursiveCharacterTextSplitter(800/200) → tables pass through whole → images extracted via PyMuPDF, summarized by GPT-4o vision → all chunks embedded with Azure OpenAI text-embedding-3-small → upserted into ChromaDB with deterministic IDs.

**Query**: Question → LLM router classifies → simple_rag path: embed query → **hybrid retrieval** (dense cosine similarity + BM25 keyword search fused via Reciprocal Rank Fusion) → **cross-encoder re-ranking** (ms-marco-MiniLM-L-6-v2 re-scores top candidates) → threshold filter (0.3 minimum) → format retrieved chunks with source citations → inject into prompt → AzureChatOpenAI generates grounded answer → parse citations and confidence → return structured response.

**Why impressive**: Shows complete mental model of both data ingestion and query-time flow.

---

### Q17: Why 800-character chunks with 200-character overlap?

**Answer**: 800 characters is approximately 150-200 tokens, which balances two concerns: (1) **Precision** — smaller chunks mean search results are more focused on the query topic, reducing noise. (2) **Context** — chunks need enough surrounding text to be self-contained. The 200-character overlap (25%) ensures that information split across chunk boundaries is captured in at least one chunk. For dense technical content like our PDFs, these values outperform the common 1000/200 default. I use `RecursiveCharacterTextSplitter` with markdown-aware separators (`\n## `, `\n\n`, `\n`, `. `) to split at natural boundaries.

**Why impressive**: Shows empirical reasoning about chunking parameters, not just defaults.

---

### Q18: How do you handle tables and images differently from text?

**Answer**: Tables are extracted as complete markdown tables by Azure DI and stored as single chunks with `content_type: "table"` metadata — the chunker skips them entirely. A 14-row comparison table stays as one 1200-character chunk even though the target is 800 chars, because splitting it would destroy the relational structure. Images are extracted via PyMuPDF using Azure DI's bounding region coordinates (polygon → pixel rect), encoded as base64 PNG, then summarized by GPT-4o vision. The summary text becomes a chunk with `content_type: "image"`. This means you can search for "performance comparison chart" and find the figure's description.

**Why impressive**: Demonstrates understanding of why naive text splitting fails for structured content.

---

### Q19: What embedding model do you use and why?

**Answer**: Azure OpenAI `text-embedding-3-small` (1536 dimensions) for production — it offers a good quality-to-cost ratio and is hosted within our Azure tenant. For tests, I use HuggingFace `all-MiniLM-L6-v2` (384 dimensions) offline — it's cached locally and runs without network access, keeping tests fast and deterministic. The switch is controlled by `EMBEDDING_PROVIDER` in the config. Important: you cannot mix embedding models in the same collection — if you change models, you must re-index everything.

**Why impressive**: Shows awareness of the embedding model lifecycle and testing strategy.

---

### Q20: What is the similarity threshold and how did you choose 0.3?

**Answer**: The similarity threshold (0.3 cosine similarity) filters out chunks that are retrieved but not meaningfully relevant. ChromaDB always returns top-k results even if they're poor matches. Without a threshold, a question about Kubernetes would still get 5 chunks about RAG — low similarity but returned anyway. 0.3 is a conservative threshold — it filters obvious noise while keeping borderline-relevant results. In production, I'd tune this empirically by analyzing retrieval precision/recall on the evaluation set.

**Why impressive**: Shows understanding of retrieval quality beyond "just return top-5."

---

### Q21: How does the LCEL RAG chain work?

**Answer**: It's composed with LangChain Expression Language: `retrieve_docs | format_context_with_citations | prompt_template | llm | parse_output`. Each step is a `Runnable`. `RunnablePassthrough` carries the original question alongside retrieved docs. The formatter inserts `[Source: docname, Page N]` markers into the context. The prompt template has a system message instructing grounded answers with citations and a user message with `{context}` and `{question}` placeholders. The LLM output is parsed to extract cited sources and estimate confidence. This is declarative — you can inspect, test, and modify each stage independently.

**Why impressive**: Shows mastery of LangChain's composition model.

---

### Q22: How did you improve retrieval quality beyond basic vector search?

**Answer**: I implemented two key enhancements: (1) **Hybrid search** — BM25 sparse retrieval runs in parallel with dense vector search. BM25 uses term-frequency scoring (Okapi BM25 from `rank-bm25`) which excels at exact keyword matches that dense embeddings sometimes miss (e.g., acronyms like "HNSW" or specific version numbers). The two result sets are fused using **Reciprocal Rank Fusion (RRF)** — `score(d) = Σ (weight / (rank + 60))` — which deduplicates by chunk_id and boosts documents found by both retrievers. I weight dense at 0.6 and BM25 at 0.4 since semantic matching is generally more valuable, but BM25 rescues keyword-heavy queries. (2) **Cross-encoder re-ranking** — after hybrid fusion, a cross-encoder model (`cross-encoder/ms-marco-MiniLM-L-6-v2`) re-scores each (query, document) pair. Unlike bi-encoders that embed query and document independently, the cross-encoder processes the concatenated pair through a single transformer, giving much more accurate relevance judgments. I over-fetch k×2 candidates and let the re-ranker select the final top-k.

**Why impressive**: Shows advanced retrieval optimization with concrete implementation, not just theoretical knowledge.

---

### Q22a: What is Reciprocal Rank Fusion and why use it over other fusion methods?

**Answer**: RRF is a rank-based fusion algorithm: `score(d) = Σ (weight / (rank + k))` where `k=60` (from the original Cormack et al. paper). It's elegant because it's **score-agnostic** — it uses only rank positions, not raw scores. This matters because BM25 scores and cosine similarity scores live on completely different scales — you can't just add them. Alternatives like CombSUM (sum normalized scores) require careful normalization; linear interpolation requires tuning a weight parameter per score distribution. RRF works out of the box because rank is a universal currency. The `k=60` constant prevents top-ranked documents from dominating — it smooths the contribution curve so rank 1 isn't overwhelmingly more valuable than rank 5.

**Why impressive**: Shows mathematical understanding of the fusion algorithm, not just "I used hybrid search."

---

### Q22b: How does cross-encoder re-ranking differ from bi-encoder search?

**Answer**: A **bi-encoder** (like text-embedding-3-small) embeds query and document independently, then compares with cosine similarity. This is fast (embed once, compare with any query) but loses cross-attention — the model never sees query and document together. A **cross-encoder** (like ms-marco-MiniLM-L-6-v2) concatenates query + document and processes the pair through a single transformer with full cross-attention. This gives much more accurate relevance judgments — the model can attend to specific query terms in the document — but it's O(N) per query since you can't pre-compute. That's why we use a two-stage pipeline: bi-encoder retrieves a broad candidate pool (k×2), then the cross-encoder re-ranks just those candidates down to the final top-k. The re-ranker loads from local cache (`local_files_only=True`) to avoid network dependency in corporate environments.

**Why impressive**: Shows understanding of the precision/speed tradeoff and the two-stage retrieval architecture pattern.

---

### Q23: What happens when the context is insufficient?

**Answer**: The system handles this at multiple levels. First, if retrieval returns no chunks above the 0.3 threshold, the RAG chain returns a low-confidence response stating no relevant information was found. Second, the system prompt explicitly instructs the LLM: "If the provided context does not contain enough information to answer the question, say so clearly — do not fabricate an answer." Third, the confidence field in the response signals to the caller/UI whether the answer should be trusted. The agentic path has an additional advantage — if initial search returns poor results, the agent can reformulate and try different tools.

**Why impressive**: Shows multi-layer approach to handling knowledge gaps.

---

### Q24: How does prompt engineering work in this system?

**Answer**: I have separate prompts for each role: (1) **RAG system prompt** — instructs grounded answers, citation format `[DocName, Page N]`, refusal when context is insufficient. (2) **RAG user prompt** — template with `{context}` and `{question}` placeholders. (3) **Router prompt** — classification criteria with examples of simple vs agentic questions. (4) **Agent system prompt** — ReAct instructions with available tools and their use cases, max iterations, and formatting rules. All prompts are in `src/rag/prompts.py` as constants — centralized, testable, and version-controlled. No prompts are hardcoded in business logic.

**Why impressive**: Shows disciplined prompt management as engineering, not ad-hoc strings.

---

### Q25: How do you extract and validate citations?

**Answer**: The RAG prompt instructs the LLM to cite sources as `[DocumentName, Page N]`. After generation, the chain parses the response text with regex to extract these citation markers. Each citation is cross-referenced against the retrieved chunks' metadata to verify the cited document and page were actually in the context. The response includes a `sources` array with `document`, `page`, `chunk_id`, and `relevance_score` for every chunk used. This gives the user both inline citations in the text and a structured source list for verification.

**Why impressive**: Shows citation as a verifiable system, not just a prompt instruction.

---

## 3. Agentic AI / LangGraph

### Q26: Explain the agentic workflow step by step.

**Answer**: (1) User question enters `run_query()`. (2) **Router node** classifies it — if `simple_rag`, goes straight to the RAG chain. (3) If `agentic`, the question goes to the **agent node** which creates a prompt with the system message, tools, and conversation history. (4) The LLM decides which tool(s) to call — e.g., `search_by_document("ChromaDB features", "vector_database_comparison.pdf")`. (5) The **tool node** executes the tool and returns results. (6) The `should_continue` edge checks: did the agent produce a final answer, or does it want more tool calls? Also checks iteration count < 5. (7) If more tools needed, loops back to agent node. (8) When done, the **format_response node** structures the final answer with sources and reasoning steps.

**Why impressive**: Shows ability to trace the full execution path through a state machine.

---

### Q27: What is a ReAct loop and how do you implement it?

**Answer**: ReAct (Reason + Act) is a pattern where the LLM alternates between reasoning about what to do and taking action via tools. In each iteration, the LLM sees: the original question, its previous reasoning, and tool results. It then either calls another tool or produces a final answer. I implement this as a LangGraph cycle: `agent_node → should_continue → tool_node → agent_node`. The agent node runs the LLM with the tool schema; if the response contains `tool_calls`, the tool node executes them and adds results to the message history; then the agent runs again with the updated history. The cycle breaks when the LLM responds without tool calls (final answer) or iteration count hits 5.

**Why impressive**: Shows understanding of the reasoning pattern, not just the API.

---

### Q28: What 4 tools did you build and why those specific ones?

**Answer**: (1) **search_knowledge_base(query, top_k)** — broad semantic search across all documents; the general-purpose retrieval tool. (2) **search_by_document(query, document_name)** — filtered search within a single document; critical for "what does document X say about Y" questions. (3) **compare_across_documents(topic, doc1, doc2)** — retrieves from two specific documents and formats side-by-side; enables comparison questions that are the primary reason for the agentic path. (4) **get_document_summary(document_name)** — retrieves the first chunks of a document for high-level context; useful when the agent needs to understand what a document covers before diving deep.

These cover the core retrieval patterns: broad search, targeted search, comparison, and summarization.

**Why impressive**: Shows tools designed for specific retrieval patterns, not generic wrappers.

---

### Q29: How do you prevent infinite agent loops?

**Answer**: Two mechanisms: (1) **Iteration counter** — `AgentState` includes `iteration_count`, incremented each cycle. The `should_continue` conditional edge checks `iteration_count < 5` — if exceeded, it forces transition to `format_response`. (2) **Convergence** — the agent prompt instructs it to produce a final answer once it has sufficient information. In practice, most questions resolve in 2-3 iterations. The max of 5 is a safety net, not a target.

**Why impressive**: Shows defensive design for autonomous systems.

---

### Q30: When would simple RAG fail and the agent succeed?

**Answer**: Three cases: (1) **Cross-document comparison** — "Compare ChromaDB and Pinecone" requires searching two documents separately and synthesizing; simple RAG searches once and may miss one side. (2) **Multi-step reasoning** — "What are the best practices for chunking and how do vector databases handle them?" needs information from different sections across documents. (3) **Iterative refinement** — if the first search returns poor results, the agent can reformulate the query and try a different tool. Simple RAG has one shot — retrieve and answer. The router identifies these patterns and routes accordingly.

**Why impressive**: Shows understanding of when agentic adds value vs. unnecessary complexity.

---

### Q31: How does structured output work for routing?

**Answer**: `AzureChatOpenAI.with_structured_output(RouteDecision)` constrains the LLM to output a JSON object matching the `RouteDecision` Pydantic model: `{"route": "simple_rag" | "agentic", "reasoning": "..."}`. Under the hood, this uses OpenAI's `response_format` with a JSON schema derived from the Pydantic model. The LLM is constrained at the token level to produce valid JSON. This is more reliable than parsing free-text output — no regex needed, no edge cases. If parsing fails anyway, we catch the exception and default to `simple_rag`.

**Why impressive**: Shows use of LLM capabilities for reliable system integration.

---

### Q32: What are conditional edges in LangGraph?

**Answer**: Conditional edges are routing functions that determine the next node based on the current state. In my graph, `add_conditional_edges("router", route_question)` calls `route_question(state)` which reads `state["route"]` and returns either `"simple_rag"` or `"agent"`. Similarly, `add_conditional_edges("agent", should_continue)` checks if the agent wants more tools or is done. This is what makes LangGraph a directed graph rather than a linear chain — the execution path is dynamic based on state.

**Why impressive**: Shows understanding of graph execution semantics, not just API calls.

---

### Q33: How would you add a new tool to the agent?

**Answer**: Three steps: (1) Write a `@tool` decorated function in `src/agent/tools.py` with a clear docstring (the docstring is the tool description the LLM sees). (2) Add it to the `get_tools()` list. (3) Update the agent system prompt in `src/rag/prompts.py` to describe when to use the new tool. No graph changes needed — the tool node dynamically executes whatever tools the agent calls. For example, adding a `search_by_topic(topic)` tool that uses metadata filtering would take ~20 lines of code.

**Why impressive**: Shows the system is extensible by design.

---

## 4. Azure & Cloud Services

### Q34: How does Azure Document Intelligence work?

**Answer**: Azure DI is a cloud-based document analysis service. I send the PDF bytes to the `begin_analyze_document` API with the `prebuilt-layout` model. It returns a structured `AnalyzeResult` containing: (1) **paragraphs** — text blocks with bounding regions, roles (title, sectionHeading, pageHeader), and character spans. (2) **tables** — structured with rows, columns, cells, captions, and spans. (3) **figures** — bounding regions with polygon coordinates and optional captions. I request `DocumentContentFormat.MARKDOWN` output, which gives me the full document as markdown. I then separate tables and figures from text using the span offsets to avoid duplication.

**Why impressive**: Shows deep understanding of the DI API, not surface-level usage.

---

### Q35: What is the prebuilt-layout model?

**Answer**: It's one of Azure DI's prebuilt models, optimized for extracting document structure: text, tables, figures, selection marks, and layout information. Unlike `prebuilt-read` (text-only) or `prebuilt-document` (key-value pairs), `prebuilt-layout` understands document structure — it identifies table boundaries, column/row structure, section headings, and figure regions. This is why I chose it over `prebuilt-read` — I need the structural elements for content-aware chunking.

**Why impressive**: Shows model selection based on specific capability needs.

---

### Q36: How did you handle figure extraction with Azure DI + PyMuPDF?

**Answer**: Azure DI identifies figure regions and returns bounding polygon coordinates (in inches from top-left). I convert these to PyMuPDF's point system (1 inch = 72 points), create a `fitz.Rect` clip region, and extract just that region as a PNG at 150 DPI. The image is base64-encoded and sent to Azure OpenAI's GPT-4o vision API for summarization. The summary text becomes a searchable chunk. If PyMuPDF isn't installed or the vision call fails, I fall back to storing just the figure caption.

**Why impressive**: Shows integration of three services (DI, PyMuPDF, GPT-4o) for a single feature.

---

### Q37: How does Azure OpenAI differ from direct OpenAI?

**Answer**: (1) **Deployment model** — Azure OpenAI uses named deployments (e.g., `gpt-4o`, `text-embedding-ada-002`) rather than model strings. (2) **Authentication** — API key + endpoint URL, not just a key. (3) **API versioning** — explicit `api-version` parameter in every call. (4) **Data privacy** — prompts/completions are not used for training; data stays in your Azure region. (5) **Network** — can be placed behind VNet/Private Endpoints. In code, I use `AzureChatOpenAI` and `AzureOpenAIEmbeddings` from `langchain-openai`, and `AzureOpenAI` from the `openai` SDK for vision calls.

**Why impressive**: Shows practical Azure OpenAI experience, not just OpenAI.

---

### Q38: What's your fallback strategy when Azure services are unavailable?

**Answer**: Two-tier fallback: (1) **Parser** — if Azure DI credentials are missing, `get_settings()` automatically sets `parser_backend` to `"pypdf"`. If DI is configured but the API call fails at runtime, `load_pdf()` catches the exception and falls back to pypdf for that file. (2) **Embeddings** — `EMBEDDING_PROVIDER` can be set to `"huggingface"` to use local `all-MiniLM-L6-v2` — no network needed. (3) **Tests** — all tests run with HuggingFace embeddings offline (`HF_HUB_OFFLINE=1`) so they never need Azure credentials.

**Why impressive**: Shows graceful degradation at every external dependency.

---

## 5. Vector Databases & Embeddings

### Q39: How does ChromaDB store and retrieve vectors?

**Answer**: ChromaDB uses `PersistentClient` with SQLite for metadata and HNSW (Hierarchical Navigable Small World) for vector indexing. When I call `upsert()`, it stores the embedding vector, document text, and metadata. For retrieval, `similarity_search_with_relevance_scores()` computes cosine similarity between the query embedding and stored vectors using HNSW for approximate nearest neighbor search. HNSW builds a multi-layer graph where each layer has progressively fewer nodes — search starts at the top layer and descends, giving O(log N) lookup. Results are returned with similarity scores.

**Why impressive**: Shows understanding of the underlying indexing algorithm, not just the API.

---

### Q40: What is cosine similarity and why use it?

**Answer**: Cosine similarity measures the angle between two vectors, returning a value from -1 to 1 (1 = identical direction, 0 = orthogonal). It's preferred for text embeddings because it's magnitude-invariant — a longer document doesn't get higher similarity just because its embedding vector has larger values. This matters because embedding models produce vectors of varying magnitudes depending on input length. Euclidean distance, by contrast, is affected by magnitude. My threshold of 0.3 means "at least 30% directionally aligned."

**Why impressive**: Shows mathematical understanding behind the metric choice.

---

### Q41: How does idempotent upsert work in your system?

**Answer**: Three mechanisms work together: (1) **File-level** — `compute_file_hash()` generates a SHA-256 hash of each PDF. The pipeline checks if this hash already exists in ChromaDB metadata — if so, the file is skipped entirely. (2) **Chunk-level** — each chunk gets a deterministic ID (`{source}::page{N}::chunk{M}`). ChromaDB's `upsert()` updates existing records with matching IDs instead of creating duplicates. (3) **Change detection** — if a file's hash has changed, all its old chunks are deleted by source metadata filter, then new chunks are upserted. This means you can re-run ingestion safely at any time — unchanged docs are skipped, changed docs are re-indexed.

**Why impressive**: Shows production-grade data pipeline thinking.

---

### Q42: How would you scale beyond ChromaDB?

**Answer**: The vector store is abstracted behind `VectorStoreManager` in `src/retrieval/vector_store.py`. To swap in Pinecone: (1) Replace `Chroma` with `Pinecone` in the manager class. (2) Update the config with Pinecone API key and index name. (3) Re-index (embeddings must be regenerated for the new store). The retriever layer (`src/retrieval/retriever.py`) wouldn't change at all — it calls `vector_store.similarity_search()` regardless of the backend. This is why the abstraction layer exists.

**Why impressive**: Shows the abstraction was intentional, not accidental.

---

### Q43: What are embedding dimensions and why do they matter?

**Answer**: Dimensions are the length of the vector produced by the embedding model — `text-embedding-3-small` produces 1536-dimensional vectors, `all-MiniLM-L6-v2` produces 384-dimensional. Higher dimensions can capture more semantic nuance but use more storage and slower to search. The critical constraint: you cannot mix dimensions in one collection. If you embed documents with a 1536-d model and query with a 384-d model, the math doesn't work. This is why I pin the embedding model in config and would re-index if changing models.

**Why impressive**: Shows awareness of a common production pitfall.

---

## 6. Production & Engineering Quality

### Q44: How did you structure your tests?

**Answer**: 53 tests across 5 modules, organized by component: `test_ingestion.py` (loader + chunker), `test_retrieval.py` (vector store + retriever + hybrid search), `test_rag.py` (prompts + chain), `test_agent.py` (router + tools + graph), `test_api.py` (FastAPI endpoints). Tests use HuggingFace embeddings offline (`HF_HUB_OFFLINE=1`) so they never call Azure. `conftest.py` provides shared fixtures: `temp_chroma_dir` for isolated vector store, `test_settings` with overridden config, `mock_llm` for chain tests. The API tests use FastAPI's `TestClient` for full request/response testing without a running server. The retrieval tests include 6 dedicated hybrid search tests covering BM25 search, RRF fusion, hybrid mode, dense-only mode, and index caching.

**Why impressive**: Shows disciplined test architecture with proper isolation.

---

### Q45: How does the Docker setup work?

**Answer**: The `Dockerfile` uses `python:3.11-slim` base, copies requirements first (layer caching), then copies the source. `docker-compose.yml` defines a single `app` service with: `.env` file for secrets, volume mount for ChromaDB persistence (`./data:/app/data`), corpus mount (`./corpus:/app/corpus`), ports 8000 (API) and 8501 (Streamlit). The `start.sh` entrypoint runs ingestion first, then starts both uvicorn and streamlit. Health check hits `/api/v1/health`.

**Why impressive**: Shows containerization best practices for ML applications.

---

### Q46: How would you add monitoring and observability?

**Answer**: Three layers: (1) **LLM tracing** — LangSmith integration is already stubbed (env vars in `.env.example`); it traces every chain/agent call with inputs, outputs, latency, and token usage. (2) **API metrics** — add Prometheus middleware to FastAPI for request count, latency, error rate; expose `/metrics` endpoint; visualize in Grafana. (3) **Retrieval quality** — log chunk IDs and similarity scores for every query; build a dashboard tracking average relevance scores over time to detect embedding drift. (4) **Alerting** — alert on low average similarity scores (indicating stale index) or high error rates.

**Why impressive**: Shows observability thinking specific to RAG systems.

---

### Q47: What security considerations are important?

**Answer**: (1) **Secrets** — all credentials in `.env`, never in code, `.gitignore` enforced. (2) **Prompt injection** — the system prompt instructs the LLM to answer only from context, limiting injection surface. In production, add input sanitization. (3) **Data access** — currently single-tenant; for multi-tenant, add tenant metadata to chunks and enforce tenant-level filtering. (4) **CORS** — configured to restrict origins in production. (5) **API validation** — Pydantic schemas reject malformed requests. (6) **Dependency security** — pinned version ranges in requirements.txt.

**Why impressive**: Shows security awareness beyond "it's behind a firewall."

---

### Q48: How do you handle concurrent API requests?

**Answer**: FastAPI runs on uvicorn with async event loop. Each request gets its own coroutine. ChromaDB's `PersistentClient` is thread-safe for reads. The LLM calls are I/O-bound (waiting for Azure), so async handles many concurrent requests efficiently. For write operations (ingestion), I'd add a task queue (Celery) in production to prevent concurrent writes to the same collection. The current setup handles typical demo/evaluation load; production would add worker scaling via gunicorn with multiple uvicorn workers.

**Why impressive**: Shows understanding of concurrency model and its limitations.

---

## 7. Curveball / Stress Questions

### Q49: What would you do differently if starting over?

**Answer**: (1) **Use async throughout** — make the ingestion pipeline async for parallel PDF processing. (2) **Implement streaming** — stream the LLM response token-by-token instead of waiting for the full answer. (3) **Better evaluation** — build the evaluation set first and use it to tune chunking parameters and thresholds empirically. (4) **Query expansion** — use the LLM to generate multiple query variants before searching for broader recall. (5) **HyDE (Hypothetical Document Embeddings)** — generate a hypothetical answer, embed it, and use that for retrieval instead of the raw question.

**Why impressive**: Shows self-awareness and growth mindset — and notably, hybrid search and re-ranking are already implemented.

---

### Q50: What are the biggest limitations?

**Answer**: (1) **Single-node ChromaDB** — no replication, max ~1M vectors. (2) **Synchronous ingestion** — processes one PDF at a time. (3) **No streaming** — user waits for full response. (4) **Fixed chunking parameters** — 800/200 works for these PDFs but may not be optimal for all document types. (5) **No user feedback loop** — no way to learn from user corrections. (6) **No query expansion** — single query embedding may miss alternative phrasings. I'm aware of all these and have a migration path for each.

**Why impressive**: Acknowledging limitations honestly shows senior maturity — and the biggest former limitation (dense-only search) has already been addressed with hybrid search + re-ranking.

---

### Q51: How would you handle a 10,000-document corpus?

**Answer**: Major changes: (1) **Vector store** — migrate to Pinecone or Weaviate for horizontal scaling. (2) **Ingestion** — parallel processing with Celery workers; batch embedding calls. (3) **Chunking** — might need document-type-specific strategies. (4) **Retrieval** — add metadata pre-filtering to narrow search scope before similarity search. (5) **Indexing** — incremental ingestion becomes critical; the SHA-256 hash system already supports this. (6) **Cost** — at 10K docs you'd have millions of chunks; optimize embedding model choice and consider dimensionality reduction.

**Why impressive**: Shows scaling thinking with specific numbers and strategies.

---

### Q52: What if Azure DI fails mid-ingestion on a batch of 100 PDFs?

**Answer**: The current architecture handles this gracefully. Each PDF is processed independently with per-file error handling. If Azure DI fails on PDF #37, it falls back to pypdf for that file, logs the error, and continues with PDF #38. The ingestion report tracks processed, skipped, and errored files. Already-ingested PDFs aren't re-processed on retry thanks to SHA-256 hash checking. In production, I'd add a dead-letter queue for failed files and retry logic with exponential backoff.

**Why impressive**: Shows the resilience is already built in.

---

### Q53: How would you add streaming responses?

**Answer**: FastAPI supports `StreamingResponse` with server-sent events (SSE). I'd change the `/api/v1/ask` endpoint to optionally return an SSE stream. On the LangChain side, `AzureChatOpenAI` supports `.astream()` which yields tokens as they're generated. The chain would stream the LLM output while buffering the full response for citation extraction. The Streamlit UI already has `st.write_stream()` for rendering streamed output. The agentic path is trickier — you'd stream intermediate reasoning steps as events, then the final answer.

**Why impressive**: Shows knowledge of the streaming pattern across the full stack.

---

### Q54: How would you add multi-tenancy?

**Answer**: (1) Add `tenant_id` to chunk metadata during ingestion. (2) Extract tenant from the API request (JWT token or header). (3) Add automatic metadata filter `{"tenant_id": current_tenant}` to every vector search query. (4) This ensures tenant A never sees tenant B's documents. ChromaDB supports metadata filtering natively via the `where` parameter. For stricter isolation, use separate ChromaDB collections per tenant, or separate Pinecone namespaces.

**Why impressive**: Shows security-first multi-tenancy thinking.

---

### Q55: How would you handle versioned documents?

**Answer**: Add `version` and `effective_date` to document metadata. When a new version is ingested: (1) Don't delete old chunks — mark them with `is_latest: false`. (2) Default retrieval filters to `is_latest: true`. (3) Add a tool `search_document_version(query, doc_name, version)` for the agent to search specific versions. (4) Track version history in a separate metadata store. The SHA-256 hash already detects content changes — extend it to trigger versioning instead of replacement.

**Why impressive**: Shows document lifecycle thinking relevant to pharma (regulatory documents have versions).

---

### Q56: What if the LLM gives wrong citations?

**Answer**: Citations can be wrong in two ways: (1) **Fabricated source** — cites a document that wasn't retrieved. I prevent this by cross-referencing citations against the `sources` array from actual retrieval. (2) **Wrong page** — cites the right document but wrong page. This is harder to prevent purely with prompting. In production, I'd add a verification step: for each citation, check if the cited page's chunk actually contains content related to the claim. If not, flag it as unverified. The structured `sources` response already gives the user the actual retrieved chunks for manual verification.

**Why impressive**: Shows awareness of citation reliability as an ongoing challenge, not a solved problem.

---

## 8. Behavioral / Impact Questions

### Q57: Why did you build this project?

**Answer**: I wanted to demonstrate end-to-end AI engineering capability — not just using an API, but building a production-oriented system that handles real-world concerns: structured document parsing, content-aware processing, intelligent routing, agentic reasoning, idempotent data pipelines, proper testing, and containerized deployment. The domain (pharmaceutical knowledge management) is directly relevant to Eli Lilly — we have massive amounts of domain knowledge in documents that could be made accessible through systems like this.

**Why impressive**: Connects technical work to business impact.

---

### Q58: What was the hardest part?

**Answer**: The Azure Document Intelligence integration was the most complex. Getting structured extraction right required understanding the DI response model — paragraphs, tables, and figures come with overlapping character spans, and you have to carefully track which spans belong to which content type to avoid duplication. The figure extraction pipeline (DI bounding regions → PyMuPDF pixel coordinates → base64 → GPT-4o vision) required coordinating three different coordinate systems and two external APIs. Getting the fallback chain right (DI → pypdf, with per-file error handling) added another layer of complexity.

**Why impressive**: Shows honest reflection on technical challenges.

---

### Q59: How does this demonstrate senior-level engineering?

**Answer**: Several ways: (1) **Architecture** — I designed a modular, extensible system with clear separation of concerns, not a monolithic script. (2) **Decision-making** — every component choice has a documented rationale with trade-offs (the ADRs). (3) **Production thinking** — idempotent ingestion, graceful degradation, error handling at every layer, secrets management, containerization. (4) **Testing** — 47 tests with proper isolation, mocking, and offline execution. (5) **Extensibility** — adding a new tool, swapping the vector store, or changing the embedding model requires minimal changes. (6) **Documentation** — architecture docs, evaluation framework, clear code structure.

**Why impressive**: Explicitly maps technical choices to senior engineering expectations.

---

### Q60: What impact could this have at Eli Lilly?

**Answer**: Pharmaceutical companies have enormous amounts of domain knowledge in research papers, regulatory documents, SOPs, clinical trial protocols, and internal reports. This system demonstrates how to make that knowledge searchable and answerable with AI — with proper citation for auditability, agentic reasoning for complex queries, and enterprise-grade security via Azure. Specific applications: (1) regulatory intelligence — quickly find relevant guidance across thousands of FDA documents, (2) clinical trial literature review, (3) internal knowledge management for R&D teams, (4) onboarding — new team members can ask questions about internal processes.

**Why impressive**: Shows business acumen and domain awareness.

---

### Q61: How would you extend this to production at Lilly?

**Answer**: (1) **Security** — Azure Private Endpoints, AAD authentication, RBAC. (2) **Scale** — migrate to Pinecone/Weaviate, add Redis caching, deploy on AKS. (3) **Data** — integrate with Lilly's document management systems (SharePoint, Veeva) for automatic ingestion. (4) **Compliance** — add audit logging, data retention policies, access controls per document sensitivity. (5) **Quality** — continuous evaluation pipeline, user feedback loop, retrieval quality monitoring. (6) **Team** — CI/CD pipeline, code review process, staged deployment (dev → staging → prod).

**Why impressive**: Shows a realistic production roadmap, not hand-waving.

---

*Good luck with your promotion panel! Know every answer here cold, but also be ready to say "I don't know, but here's how I'd find out" — that's a senior engineer's most powerful answer.*

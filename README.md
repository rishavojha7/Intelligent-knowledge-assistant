# Intelligent Knowledge Assistant

A production-oriented AI application that answers questions over a collection of unstructured documents using **Retrieval-Augmented Generation (RAG)** with an **agentic workflow** for complex, multi-step questions.

## 📋 Problem Understanding

Organizations accumulate technical knowledge across documents, but finding specific information requires reading entire documents. This system creates a searchable, intelligent interface over a document corpus that:

1. **Ingests** PDF documents into a vector knowledge base using Azure Document Intelligence
2. **Retrieves** relevant content using hybrid search (BM25 + dense embeddings + Reciprocal Rank Fusion) with cross-encoder re-ranking
3. **Generates** grounded answers with source citations
4. **Routes** complex questions through an agentic workflow for multi-step reasoning
5. **Refuses gracefully** when the answer isn't in the knowledge base

## 🏗️ Architecture

```
User → Streamlit UI → FastAPI → Router → [Simple RAG | LangGraph Agent] → Hybrid Retrieval (BM25+Dense+RRF) → Cross-Encoder Re-Rank → ChromaDB → Answer + Citations
```

See [docs/architecture.md](docs/architecture.md) for the full architecture diagram and design decisions.

### Key Technology Choices

| Component | Choice | Why |
|-----------|--------|-----|
| LLM | Azure OpenAI GPT-4o | Enterprise compliance, VNet support, data residency, strong tool calling |
| Embeddings | Azure OpenAI text-embedding-3-small | Good quality/cost; HuggingFace fallback for offline dev |
| PDF Parser | Azure Document Intelligence | Structured markdown, table/figure extraction; pypdf fallback |
| Vector Store | ChromaDB | Zero-infra, persistent, metadata filtering, sufficient at this scale |
| Retrieval | Hybrid (BM25 + Dense + RRF) | Combines keyword precision with semantic understanding |
| Re-Ranking | Cross-encoder (ms-marco-MiniLM-L-6-v2) | Two-stage retrieval for higher precision |
| Agent Framework | LangGraph | Stateful graph, conditional routing, iteration limits |
| Chunking | Content-aware (text/table/image) | Tables kept whole, images vision-summarized, text split at 800/200 |
| API | FastAPI | Async-native, auto-docs, Pydantic validation |
| UI | Streamlit | Minimal code, chat interface, expander widgets for sources |

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Azure OpenAI API key and endpoint
- Docker (optional, for containerized deployment)

### Option 1: Docker (Recommended)

```bash
# Clone and configure
cp .env.example .env
# Edit .env and add your Azure OpenAI credentials

# Run everything
docker-compose up --build

# Access:
#   API:  http://localhost:8000/docs
#   UI:   http://localhost:8501
```

### Option 2: Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env and add your Azure OpenAI credentials

# Ingest documents
python -m scripts.ingest

# Start the API
uvicorn src.api.app:app --host 0.0.0.0 --port 8000

# In a separate terminal, start the UI
streamlit run ui/app.py --server.port 8501
```

## 📡 API Usage

### Ask a Question

```bash
# Auto-routed (system decides simple RAG vs agentic)
curl -X POST http://localhost:8000/api/v1/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is RAG and how does it work?"}'

# Force agentic workflow
curl -X POST http://localhost:8000/api/v1/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Compare ChromaDB and Pinecone for production RAG", "use_agent": true}'
```

### Response Format

```json
{
  "answer": "RAG (Retrieval-Augmented Generation) is a technique that...",
  "sources": [
    {
      "document": "rag_architecture_patterns.pdf",
      "page": 3,
      "chunk_id": "rag_architecture_patterns.pdf::page2::chunk5",
      "relevance_score": 0.87
    }
  ],
  "workflow": "simple_rag",
  "reasoning_steps": [
    "Retrieved 5 relevant chunks",
    "Average relevance score: 0.742",
    "Generated answer from context with citations"
  ],
  "confidence": "high"
}
```

### Other Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/ask` | POST | Ask a question |
| `/api/v1/health` | GET | Service health and knowledge base status |
| `/api/v1/ingest` | POST | Trigger document ingestion |
| `/api/v1/documents` | GET | List ingested documents |
| `/docs` | GET | Interactive API documentation (Swagger UI) |

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test modules
pytest tests/test_ingestion.py -v
pytest tests/test_api.py -v
```

**53 tests across 5 modules:**
- PDF loading and metadata extraction (Azure DI + pypdf fallback)
- Content-aware chunking with metadata preservation and deterministic IDs
- Vector store operations (upsert, query, idempotency, filtering)
- Hybrid search: BM25 search, RRF fusion, dense-only mode, index caching (6 tests)
- RAG chain assembly and empty-result handling
- Router classification validation
- Agent tool definitions and graph compilation
- E2E API endpoint testing with mocked LLM

## 📊 Evaluation

```bash
# Run the evaluation suite (requires Azure OpenAI credentials)
python scripts/evaluate.py
```

**15 evaluation questions across 4 categories:**

| Category | Count | Purpose |
|----------|-------|---------|
| Simple RAG | 5 | Single-document factual questions |
| Multi-step | 5 | Cross-document comparison and synthesis |
| Ambiguous | 3 | Vague or incomplete questions |
| Unanswerable | 2 | Questions outside the corpus |

**Metrics computed:**
- **Routing accuracy**: Does the router choose the correct workflow?
- **Source recall**: Are the expected source documents retrieved?
- **Content recall**: Does the answer contain expected key terms?
- **Graceful refusal rate**: Does the system refuse unanswerable questions without hallucinating?
- **Latency**: Response time by workflow type

### Evaluation Results

All 15 questions answered successfully with zero errors.

#### Aggregate Metrics

| Metric | Score |
|--------|-------|
| **Routing Accuracy** | **80%** (12/15 correctly routed) |
| **Avg Source Recall** | **66.7%** |
| **Avg Content Recall** | **77.8%** |
| **Unanswerable Refusal Rate** | **50%** (1/2 correctly refused) |
| **Avg Latency** | **19.2s** |

#### Per-Category Breakdown

| Category | Questions | Content Recall | Source Recall | Highlights |
|----------|-----------|---------------|--------------|------------|
| **Simple RAG** | 5 | 66.7% | 100% ✅ | All routed correctly, high confidence, avg 12.6s |
| **Multi-step** | 5 | 86.7% | 0%* | All routed correctly to agentic, thorough answers |
| **Ambiguous** | 3 | 100% ✅ | 100% ✅ | Appropriately asked for clarification on vague queries |
| **Unanswerable** | 2 | 50% | 100% | 1/2 refused gracefully; 1 gave a partial answer |

*\*Multi-step source recall is 0% due to a known evaluation metric limitation: the agentic workflow cites sources in the answer text (e.g., `[vector_database_comparison.pdf, Page 5]`) but the agent's tool-based retrieval doesn't populate the `sources` response field the same way simple RAG does. The answers themselves correctly reference the expected documents — this is a metric collection gap, not an answer quality issue.*

#### Notable Results

- **Perfect routing on core categories**: All 5 simple RAG and all 5 multi-step questions routed correctly
- **Strong ambiguous handling**: The system appropriately asks for clarification on vague questions like "What's the best approach?" and "How does it compare to the alternatives?"
- **Grounded refusal**: For "What is the pricing for OpenAI GPT-4o API calls?", the system correctly responds: *"I don't have enough information in the available documents to answer this question"*
- **High-quality agentic answers**: Cross-document comparisons (ChromaDB vs Pinecone, LangGraph vs CrewAI) produce structured, multi-section answers with content from both documents
- **Latency profile**: Simple RAG averages ~12s; agentic averages ~19s (2-3 tool-calling iterations); one outlier at 92s on a 4-tool-call comparison query

## 📁 Project Structure

```
├── src/                    # Application source code
│   ├── config.py           # Configuration management (pydantic-settings)
│   ├── models.py           # Shared domain models
│   ├── ingestion/          # Document ingestion pipeline
│   │   ├── loader.py       # PDF loading with metadata
│   │   ├── chunker.py      # Text chunking with IDs
│   │   └── pipeline.py     # Orchestrator with idempotency
│   ├── retrieval/          # Vector store and hybrid retrieval
│   │   ├── vector_store.py # ChromaDB wrapper
│   │   └── retriever.py    # Hybrid search + cross-encoder re-ranking
│   ├── rag/                # RAG chain
│   │   ├── prompts.py      # All prompt templates
│   │   └── chain.py        # LCEL chain implementation
│   ├── agent/              # Agentic workflow
│   │   ├── tools.py        # Agent tools (4 search tools)
│   │   ├── router.py       # LLM question classifier
│   │   └── graph.py        # LangGraph workflow
│   └── api/                # REST API
│       ├── schemas.py      # Request/response models
│       ├── app.py          # FastAPI application factory
│       └── routes.py       # Endpoint handlers
├── tests/                  # Test suite
├── scripts/                # CLI scripts (ingest, evaluate)
├── configs/                # Evaluation question set
├── corpus/                 # PDF document corpus
├── ui/                     # Streamlit demo interface
├── docs/                   # Architecture documentation
├── Dockerfile              # Container build
├── docker-compose.yml      # Single-command deployment
└── .env.example            # Environment variable template
```

## 🔑 Design Decisions

### Why LangGraph over LangChain Agents?
LangGraph provides explicit state management through a directed graph, making the workflow debuggable and controllable. It supports conditional routing, iteration limits, and clean separation between the router, agent, and tool execution. The corpus document on agentic frameworks itself recommends LangGraph for "RAG-centric knowledge assistants."

### Why ChromaDB?
For a 3-document, ~130-chunk corpus, ChromaDB is the right tool: zero infrastructure, embedded persistence, metadata filtering, and strong LangChain integration. We layer BM25 hybrid search on top via `rank-bm25` with Reciprocal Rank Fusion, compensating for ChromaDB's lack of native hybrid search. For production scale (>1M vectors), the architecture is designed to swap to Pinecone or pgvector via the `VectorStoreManager` abstraction.

### Why Hybrid Search + Re-Ranking?
Dense vector search captures semantic meaning but can miss exact keyword matches (acronyms, version numbers). BM25 captures precise terms but misses paraphrases. By fusing both with Reciprocal Rank Fusion and re-ranking with a cross-encoder (`ms-marco-MiniLM-L-6-v2`), we get the best of both worlds: broad semantic recall with precise keyword matching, refined by a more accurate scoring model.

### Why a Router?
Not every question needs an agent. Simple factual questions ("What is RAG?") are answered faster and more cheaply with a single retrieval + generation step. The LLM router classifies questions in ~0.5s, saving 5-10s on simple queries by avoiding unnecessary tool-calling loops.

### Idempotent Ingestion
Documents are hashed (SHA-256) and compared against stored hashes. Unchanged documents are skipped, and changed documents have their old chunks deleted before re-ingestion. This makes `ingest` safe to run repeatedly.

## ⚠️ Limitations

1. **Corpus scale**: ChromaDB is suitable for small corpora; would need migration for >100K chunks
2. **No streaming**: Answers are returned complete; streaming would improve UX for longer answers
3. **No conversation memory**: Each question is independent; multi-turn context would improve follow-up questions
4. **No authentication**: API is open; production would need auth middleware
5. **Synchronous ingestion**: Processes one PDF at a time; async workers would speed up large batches
6. **Fixed chunking parameters**: 800/200 works for these PDFs but may not be optimal for all document types

## 🔮 Production Improvements

1. **Streaming responses** via Server-Sent Events for real-time answer generation
2. **Conversation memory** for multi-turn question answering
3. **Query expansion** — LLM generates multiple query variants for broader recall
4. **Caching layer** (Redis) for repeated queries and embedding results
5. **Monitoring dashboard** with LangSmith/Arize for LLM observability
6. **A/B testing** framework for prompt and retrieval strategy optimization
7. **Rate limiting and authentication** for production API security
8. **Async ingestion** with background workers for large corpus updates
9. **pgvector migration** for production-scale vector storage with ACID guarantees
10. **User feedback loop** to learn from corrections and improve retrieval over time

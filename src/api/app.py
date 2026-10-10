"""
FastAPI application factory.

Creates and configures the FastAPI application with:
    - CORS middleware for Streamlit access
    - Lifespan context manager for startup/shutdown
    - Global exception handlers
    - API router mounting
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.routes import router
from src.config import get_settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle management."""
    # Startup
    logger.info("Starting Knowledge Assistant API...")
    settings = get_settings()

    # Verify corpus directory exists
    if not settings.corpus_path.is_dir():
        logger.warning("Corpus directory not found: %s", settings.corpus_path)

    # Check if knowledge base has data
    try:
        from src.config import get_embeddings
        from src.retrieval.vector_store import VectorStoreManager

        store = VectorStoreManager(
            persist_dir=settings.chroma_persist_dir,
            collection_name=settings.chroma_collection_name,
            embedding_function=get_embeddings(settings),
        )
        stats = store.get_collection_stats()
        if stats["total_chunks"] == 0:
            logger.warning(
                "Knowledge base is empty. Run ingestion first: "
                "python -m scripts.ingest"
            )
        else:
            logger.info(
                "Knowledge base ready: %d chunks from %d documents",
                stats["total_chunks"],
                stats["documents_count"],
            )
    except Exception:
        logger.exception("Failed to check knowledge base status at startup")

    logger.info("API ready at http://%s:%d", settings.api_host, settings.api_port)

    yield

    # Shutdown
    logger.info("Shutting down Knowledge Assistant API...")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Intelligent Knowledge Assistant",
        description=(
            "A production-oriented AI application that answers questions over a collection "
            "of unstructured documents using RAG and agentic workflows."
        ),
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS — allow Streamlit and local development
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.exception("Unhandled exception: %s", exc)
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_server_error",
                "detail": "An unexpected error occurred. Please try again later.",
            },
        )

    # Mount API routes
    app.include_router(router)

    # Root redirect to docs
    @app.get("/", include_in_schema=False)
    async def root():
        return {"message": "Knowledge Assistant API", "docs": "/docs"}

    return app


# Application instance for uvicorn
app = create_app()

"""
Configuration management for the Knowledge Assistant.

All settings are loaded from environment variables / .env file using pydantic-settings.
Factory functions provide configured Azure OpenAI LLM and embedding model instances.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Azure OpenAI ---
    azure_openai_endpoint: str = Field(default="", description="Azure OpenAI endpoint URL")
    azure_openai_api_key: str = Field(default="", description="Azure OpenAI API key")
    azure_openai_api_version: str = Field(default="2024-08-01-preview", description="Azure OpenAI API version")
    openai_api_version: str = Field(default="", description="OpenAI API version (fallback)")
    azure_openai_deployment: str = Field(default="gpt-4o", description="Azure OpenAI LLM deployment name")
    azure_openai_deployment_embed: str = Field(default="text-embedding-3-small", description="Azure OpenAI embedding deployment name")
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)

    # --- Embedding fallback ---
    embedding_provider: str = Field(
        default="azure_openai",
        description="Embedding provider: 'azure_openai' or 'huggingface'",
    )

    # --- Document Parser ---
    parser_backend: str = Field(
        default="azure_di",
        description="PDF parser: 'azure_di' (primary) or 'pypdf' (fallback)",
    )
    documentintelligence_endpoint: str = Field(default="", description="Azure Document Intelligence endpoint URL")
    documentintelligence_api_key: str = Field(default="", description="Azure Document Intelligence API key")
    documentintelligence_model: str = Field(default="prebuilt-layout", description="Azure DI model ID")
    image_summary_enabled: bool = Field(
        default=True,
        description="Summarize extracted images with vision LLM",
    )

    # --- ChromaDB ---
    chroma_persist_dir: str = Field(default="./data/chromadb")
    chroma_collection_name: str = Field(default="knowledge_base")

    # --- Retrieval ---
    chunk_size: int = Field(default=800, ge=100, le=4000)
    chunk_overlap: int = Field(default=200, ge=0, le=1000)
    retrieval_top_k: int = Field(default=5, ge=1, le=20)
    similarity_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    retrieval_mode: str = Field(
        default="hybrid",
        description="Retrieval mode: 'dense' (vector only) or 'hybrid' (BM25 + dense with RRF)",
    )
    bm25_weight: float = Field(default=0.4, ge=0.0, le=1.0, description="BM25 weight in reciprocal rank fusion")
    dense_weight: float = Field(default=0.6, ge=0.0, le=1.0, description="Dense weight in reciprocal rank fusion")
    reranker_enabled: bool = Field(default=True, description="Enable cross-encoder re-ranking")
    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Cross-encoder model for re-ranking",
    )
    reranker_top_k: int = Field(default=5, ge=1, le=20, description="Number of results after re-ranking")

    # --- Corpus ---
    corpus_dir: str = Field(default="./corpus")

    # --- API ---
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)

    # --- Streamlit ---
    streamlit_port: int = Field(default=8501)

    # --- Logging ---
    log_level: str = Field(default="INFO")

    @field_validator("embedding_provider")
    @classmethod
    def validate_embedding_provider(cls, v: str) -> str:
        v = v.lower()
        if v not in ("azure_openai", "huggingface"):
            raise ValueError(f"embedding_provider must be 'azure_openai' or 'huggingface', got '{v}'")
        return v

    @field_validator("parser_backend")
    @classmethod
    def validate_parser_backend(cls, v: str) -> str:
        v = v.lower()
        if v not in ("azure_di", "pypdf"):
            raise ValueError(f"parser_backend must be 'azure_di' or 'pypdf', got '{v}'")
        return v

    @field_validator("retrieval_mode")
    @classmethod
    def validate_retrieval_mode(cls, v: str) -> str:
        v = v.lower()
        if v not in ("dense", "hybrid"):
            raise ValueError(f"retrieval_mode must be 'dense' or 'hybrid', got '{v}'")
        return v

    @property
    def effective_api_version(self) -> str:
        """Return the API version, preferring azure_openai_api_version over openai_api_version."""
        return self.azure_openai_api_version or self.openai_api_version or "2024-08-01-preview"

    @property
    def corpus_path(self) -> Path:
        return Path(self.corpus_dir).resolve()

    @property
    def chroma_path(self) -> Path:
        return Path(self.chroma_persist_dir).resolve()


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings. Call once at startup."""
    settings = Settings()
    _configure_logging(settings.log_level)

    if not settings.azure_openai_api_key:
        logger.warning(
            "AZURE_OPENAI_API_KEY is not set. Azure OpenAI LLM and embedding calls will fail."
        )
    if not settings.azure_openai_endpoint:
        logger.warning(
            "AZURE_OPENAI_ENDPOINT is not set. Azure OpenAI calls will fail."
        )
    if settings.parser_backend == "azure_di" and (
        not settings.documentintelligence_endpoint or not settings.documentintelligence_api_key
    ):
        logger.warning(
            "Document Intelligence credentials not set "
            "(DOCUMENTINTELLIGENCE_ENDPOINT / DOCUMENTINTELLIGENCE_API_KEY). "
            "Falling back to pypdf parser."
        )
        settings.parser_backend = "pypdf"

    return settings


def get_llm(settings: Settings | None = None):
    """Return a configured AzureChatOpenAI instance."""
    from langchain_openai import AzureChatOpenAI

    s = settings or get_settings()
    return AzureChatOpenAI(
        azure_endpoint=s.azure_openai_endpoint,
        azure_deployment=s.azure_openai_deployment,
        api_key=s.azure_openai_api_key,
        api_version=s.effective_api_version,
        temperature=s.llm_temperature,
    )


def get_embeddings(settings: Settings | None = None):
    """Return a configured embedding model based on provider setting."""
    s = settings or get_settings()

    if s.embedding_provider == "azure_openai":
        from langchain_openai import AzureOpenAIEmbeddings

        return AzureOpenAIEmbeddings(
            azure_endpoint=s.azure_openai_endpoint,
            azure_deployment=s.azure_openai_deployment_embed,
            api_key=s.azure_openai_api_key,
            api_version=s.effective_api_version,
        )

    # HuggingFace / sentence-transformers (local, no API key needed)
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def _configure_logging(level: str) -> None:
    """Set up structured logging for the application."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

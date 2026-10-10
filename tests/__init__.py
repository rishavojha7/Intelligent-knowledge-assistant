"""Tests package — sets environment defaults before any test imports."""

import os

# Ensure tests don't require real Azure credentials or network access
os.environ.setdefault("AZURE_OPENAI_API_KEY", "test-key-not-real")
os.environ.setdefault("AZURE_OPENAI_ENDPOINT", "https://test.openai.azure.com/")
os.environ.setdefault("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
os.environ.setdefault("AZURE_OPENAI_DEPLOYMENT_EMBED", "text-embedding-3-small")
os.environ.setdefault("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
os.environ.setdefault("EMBEDDING_PROVIDER", "huggingface")
os.environ.setdefault("RETRIEVAL_MODE", "dense")
os.environ.setdefault("RERANKER_ENABLED", "false")
os.environ.setdefault("LOG_LEVEL", "WARNING")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

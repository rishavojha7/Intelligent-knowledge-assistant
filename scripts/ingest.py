"""
CLI script for document ingestion.

Usage:
    python -m scripts.ingest [--corpus-dir ./corpus] [--force]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.config import get_settings
from src.ingestion.pipeline import IngestionPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest PDF documents into the knowledge base")
    parser.add_argument(
        "--corpus-dir",
        type=str,
        default=None,
        help="Directory containing PDF files (default: from settings)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-ingestion of all documents (ignores hash check)",
    )
    args = parser.parse_args()

    settings = get_settings()
    corpus_dir = args.corpus_dir or settings.corpus_dir

    print(f"{'=' * 60}")
    print("Knowledge Base — Document Ingestion")
    print(f"{'=' * 60}")
    print(f"  Corpus directory : {Path(corpus_dir).resolve()}")
    print(f"  ChromaDB storage : {settings.chroma_path}")
    embed_label = (
        f"{settings.azure_openai_deployment_embed}"
        if settings.embedding_provider == "azure_openai"
        else "all-MiniLM-L6-v2"
    )
    print(f"  Embedding model  : {settings.embedding_provider}/{embed_label}")
    print(f"  Chunk size       : {settings.chunk_size} (overlap: {settings.chunk_overlap})")
    print(f"{'=' * 60}")

    pipeline = IngestionPipeline(settings=settings)

    if args.force:
        print("\n⚠️  Force mode: clearing existing collection...")
        stats = pipeline.vector_store.get_collection_stats()
        for doc_name in stats["documents"]:
            pipeline.vector_store.delete_by_document(doc_name)

    report = pipeline.ingest(corpus_dir)

    print(f"\n{'─' * 40}")
    print("Ingestion Report:")
    print(f"  Documents processed : {report.documents_processed}")
    print(f"  Documents skipped   : {report.documents_skipped}")
    print(f"  Chunks created      : {report.chunks_created}")
    if report.errors:
        print(f"  Errors              : {len(report.errors)}")
        for err in report.errors:
            print(f"    ✗ {err}")
    else:
        print("  Errors              : 0")
    print(f"{'─' * 40}")

    # Show final collection stats
    stats = pipeline.vector_store.get_collection_stats()
    print(f"\nCollection '{settings.chroma_collection_name}':")
    print(f"  Total chunks : {stats['total_chunks']}")
    print(f"  Documents    : {', '.join(stats['documents']) or 'none'}")
    print()


if __name__ == "__main__":
    main()

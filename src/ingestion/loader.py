"""
PDF document loader with Azure Document Intelligence (primary) and pypdf (fallback).

Azure DI extracts structured markdown with tables, paragraphs, and figures.
Content is classified by type so the chunker can handle each appropriately:
    - text: regular paragraphs → chunked normally
    - table: markdown tables → kept whole (never split)
    - image: figure regions → extracted as base64, summarized by vision LLM
"""

from __future__ import annotations

import base64
import hashlib
import io
import logging
import re
from pathlib import Path

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def compute_file_hash(file_path: Path) -> str:
    """Compute SHA-256 hash of a file for change detection."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            sha256.update(block)
    return sha256.hexdigest()


# ---------------------------------------------------------------------------
# Azure Document Intelligence loader (primary)
# ---------------------------------------------------------------------------

def load_pdf_azure_di(
    file_path: Path,
    endpoint: str,
    key: str,
    model_id: str = "prebuilt-layout",
    summarize_images: bool = True,
    azure_openai_endpoint: str = "",
    azure_openai_api_key: str = "",
    azure_openai_deployment: str = "",
    azure_openai_api_version: str = "",
) -> list[Document]:
    """
    Load a PDF using Azure Document Intelligence.

    Returns Document objects tagged with content_type metadata:
        - "text": paragraph content (to be chunked)
        - "table": full markdown table (kept as single chunk)
        - "image": vision LLM summary of an extracted figure

    Args:
        file_path: Path to the PDF file.
        endpoint: Azure DI endpoint URL.
        key: Azure DI API key.
        model_id: Azure DI model (default: prebuilt-layout).
        summarize_images: Whether to extract and summarize figures.
        azure_openai_endpoint: Azure OpenAI endpoint for vision summarization.
        azure_openai_api_key: Azure OpenAI API key for vision summarization.
        azure_openai_deployment: Azure OpenAI deployment name for vision model.
        azure_openai_api_version: Azure OpenAI API version.
    """
    from azure.ai.documentintelligence import DocumentIntelligenceClient
    from azure.ai.documentintelligence.models import (
        AnalyzeDocumentRequest,
        AnalyzeOutputOption,
        DocumentContentFormat,
    )
    from azure.core.credentials import AzureKeyCredential

    file_path = Path(file_path).resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    doc_hash = compute_file_hash(file_path)
    source_name = file_path.name

    logger.info("Loading PDF with Azure DI: %s", source_name)

    # Initialize client and analyze
    client = DocumentIntelligenceClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(key),
    )

    with open(file_path, "rb") as f:
        pdf_bytes = f.read()

    output_options = [AnalyzeOutputOption.FIGURES] if summarize_images else []

    poller = client.begin_analyze_document(
        model_id=model_id,
        body=AnalyzeDocumentRequest(bytes_source=pdf_bytes),
        output_content_format=DocumentContentFormat.MARKDOWN,
        output=output_options or None,
    )
    result = poller.result()

    documents: list[Document] = []

    # --- Extract tables as whole units ---
    table_spans = set()  # Track character spans occupied by tables
    if result.tables:
        for table_idx, table in enumerate(result.tables):
            table_md = _reconstruct_table_markdown(table)
            if not table_md.strip():
                continue

            # Determine which page this table is on
            page_num = 0
            if table.bounding_regions:
                page_num = table.bounding_regions[0].page_number - 1  # 0-indexed

            # Track spans so we can exclude table text from paragraph chunks
            if table.spans:
                for span in table.spans:
                    table_spans.add((span.offset, span.offset + span.length))

            caption_text = ""
            if table.caption:
                caption_text = f"**{table.caption.content}**\n\n"

            documents.append(Document(
                page_content=f"{caption_text}{table_md}",
                metadata={
                    "source": source_name,
                    "page": page_num,
                    "content_type": "table",
                    "table_index": table_idx,
                    "row_count": table.row_count,
                    "column_count": table.column_count,
                    "doc_hash": doc_hash,
                    "file_path": str(file_path),
                },
            ))

    logger.info("Extracted %d tables from %s", len(documents), source_name)

    # --- Extract figures/images ---
    figure_spans = set()
    if summarize_images and result.figures:
        figure_docs = _extract_figures(
            result=result,
            file_path=file_path,
            source_name=source_name,
            doc_hash=doc_hash,
            azure_endpoint=azure_openai_endpoint,
            azure_api_key=azure_openai_api_key,
            azure_deployment=azure_openai_deployment,
            azure_api_version=azure_openai_api_version,
        )
        documents.extend(figure_docs)

        # Track figure spans
        for fig in result.figures:
            if fig.spans:
                for span in fig.spans:
                    figure_spans.add((span.offset, span.offset + span.length))

    # --- Extract text paragraphs (excluding table and figure regions) ---
    excluded_spans = table_spans | figure_spans
    text_docs = _extract_text_by_page(
        result=result,
        excluded_spans=excluded_spans,
        source_name=source_name,
        doc_hash=doc_hash,
        file_path=file_path,
    )
    documents.extend(text_docs)

    logger.info(
        "Azure DI loaded %s: %d text blocks, %d tables, %d figures",
        source_name,
        len(text_docs),
        sum(1 for d in documents if d.metadata.get("content_type") == "table"),
        sum(1 for d in documents if d.metadata.get("content_type") == "image"),
    )
    return documents


def _reconstruct_table_markdown(table) -> str:
    """Reconstruct a markdown table from Azure DI DocumentTable cells."""
    if not table.cells:
        return ""

    # Build a grid
    grid: dict[tuple[int, int], str] = {}
    for cell in table.cells:
        grid[(cell.row_index, cell.column_index)] = cell.content or ""

    rows = []
    for r in range(table.row_count):
        row_cells = [grid.get((r, c), "") for c in range(table.column_count)]
        rows.append("| " + " | ".join(row_cells) + " |")

        # Add header separator after first row
        if r == 0:
            rows.append("| " + " | ".join(["---"] * table.column_count) + " |")

    return "\n".join(rows)


def _extract_text_by_page(
    result,
    excluded_spans: set[tuple[int, int]],
    source_name: str,
    doc_hash: str,
    file_path: Path,
) -> list[Document]:
    """Extract text paragraphs grouped by page, excluding table/figure spans."""
    documents = []

    if not result.paragraphs:
        # Fallback: use full content split by pages
        if result.content:
            documents.append(Document(
                page_content=result.content,
                metadata={
                    "source": source_name,
                    "page": 0,
                    "content_type": "text",
                    "doc_hash": doc_hash,
                    "file_path": str(file_path),
                },
            ))
        return documents

    # Group paragraphs by page
    page_paragraphs: dict[int, list[str]] = {}

    for para in result.paragraphs:
        # Skip page headers/footers/page numbers
        if para.role in ("pageHeader", "pageFooter", "pageNumber"):
            continue

        # Check if this paragraph overlaps with excluded spans (tables/figures)
        if para.spans and _spans_overlap(para.spans, excluded_spans):
            continue

        # Determine page number
        page_num = 0
        if para.bounding_regions:
            page_num = para.bounding_regions[0].page_number - 1

        content = para.content or ""
        if not content.strip():
            continue

        # Add heading markers for section headings
        if para.role in ("title", "sectionHeading"):
            content = f"## {content}"

        page_paragraphs.setdefault(page_num, []).append(content)

    for page_num in sorted(page_paragraphs.keys()):
        paragraphs = page_paragraphs[page_num]
        page_text = "\n\n".join(paragraphs)

        if page_text.strip():
            documents.append(Document(
                page_content=page_text,
                metadata={
                    "source": source_name,
                    "page": page_num,
                    "content_type": "text",
                    "doc_hash": doc_hash,
                    "file_path": str(file_path),
                },
            ))

    return documents


def _spans_overlap(para_spans, excluded_spans: set[tuple[int, int]]) -> bool:
    """Check if any paragraph span overlaps with excluded spans."""
    for ps in para_spans:
        p_start, p_end = ps.offset, ps.offset + ps.length
        for e_start, e_end in excluded_spans:
            if p_start < e_end and p_end > e_start:
                return True
    return False


def _extract_figures(
    result,
    file_path: Path,
    source_name: str,
    doc_hash: str,
    azure_endpoint: str = "",
    azure_api_key: str = "",
    azure_deployment: str = "",
    azure_api_version: str = "",
) -> list[Document]:
    """Extract figures from the PDF, encode as base64, and summarize with vision LLM."""
    documents = []

    if not result.figures:
        return documents

    try:
        import fitz  # PyMuPDF
    except ImportError:
        logger.warning("PyMuPDF not installed — skipping figure extraction")
        return documents

    pdf_doc = fitz.open(str(file_path))

    for fig_idx, figure in enumerate(result.figures):
        if not figure.bounding_regions:
            continue

        region = figure.bounding_regions[0]
        page_num = region.page_number - 1  # 0-indexed

        if page_num >= len(pdf_doc):
            continue

        try:
            # Extract the figure region as an image
            page = pdf_doc[page_num]
            polygon = region.polygon  # list of x,y coordinates

            if polygon and len(polygon) >= 4:
                # polygon is [x1,y1, x2,y2, x3,y3, x4,y4] — get bounding box
                xs = [polygon[i] for i in range(0, len(polygon), 2)]
                ys = [polygon[i] for i in range(1, len(polygon), 2)]

                # Azure DI coordinates are in inches from top-left
                # PyMuPDF uses points (1 inch = 72 points)
                clip = fitz.Rect(
                    min(xs) * 72,
                    min(ys) * 72,
                    max(xs) * 72,
                    max(ys) * 72,
                )

                # Clip to page bounds
                clip = clip & page.rect

                pix = page.get_pixmap(clip=clip, dpi=150)
                img_bytes = pix.tobytes("png")
                img_b64 = base64.b64encode(img_bytes).decode("utf-8")
            else:
                # No polygon — render full page
                pix = page.get_pixmap(dpi=150)
                img_bytes = pix.tobytes("png")
                img_b64 = base64.b64encode(img_bytes).decode("utf-8")

            # Summarize with vision LLM
            caption = figure.caption.content if figure.caption else ""
            summary = _summarize_image(
                img_b64, caption,
                azure_endpoint=azure_endpoint,
                azure_api_key=azure_api_key,
                azure_deployment=azure_deployment,
                azure_api_version=azure_api_version,
            )

            if summary:
                documents.append(Document(
                    page_content=summary,
                    metadata={
                        "source": source_name,
                        "page": page_num,
                        "content_type": "image",
                        "figure_index": fig_idx,
                        "image_b64_preview": img_b64[:200] + "...",  # truncated for metadata
                        "caption": caption,
                        "doc_hash": doc_hash,
                        "file_path": str(file_path),
                    },
                ))

        except Exception:
            logger.exception("Failed to extract figure %d from page %d", fig_idx, page_num)

    pdf_doc.close()
    logger.info("Extracted %d figures from %s", len(documents), source_name)
    return documents


def _summarize_image(
    img_b64: str,
    caption: str,
    azure_endpoint: str = "",
    azure_api_key: str = "",
    azure_deployment: str = "",
    azure_api_version: str = "",
) -> str:
    """Summarize an image using Azure OpenAI GPT-4o vision."""
    if not azure_api_key or not azure_endpoint:
        return f"[Figure: {caption}]" if caption else "[Figure: image extracted from document]"

    try:
        from openai import AzureOpenAI

        client = AzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=azure_api_key,
            api_version=azure_api_version or "2024-08-01-preview",
        )

        prompt = (
            "Describe this figure from a technical document in detail. "
            "Include all data points, labels, relationships, and key takeaways. "
            "Be precise and thorough — your description will be used for search and retrieval."
        )
        if caption:
            prompt += f"\n\nThe figure caption is: {caption}"

        response = client.chat.completions.create(
            model=azure_deployment,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_b64}",
                                "detail": "high",
                            },
                        },
                    ],
                }
            ],
            max_tokens=500,
        )
        summary = response.choices[0].message.content
        logger.debug("Image summary: %s", summary[:100])
        return f"[Figure: {caption}]\n\n{summary}" if caption else summary

    except Exception:
        logger.exception("Vision LLM summarization failed")
        return f"[Figure: {caption}]" if caption else "[Figure: image extracted from document]"


# ---------------------------------------------------------------------------
# pypdf loader (fallback)
# ---------------------------------------------------------------------------

def load_pdf_pypdf(file_path: Path) -> list[Document]:
    """
    Load a PDF using pypdf. Fallback when Azure DI is not configured.

    Returns Document objects with content_type="text" (no table/image separation).
    """
    from langchain_community.document_loaders import PyPDFLoader

    file_path = Path(file_path).resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    doc_hash = compute_file_hash(file_path)
    source_name = file_path.name

    logger.info("Loading PDF with pypdf (fallback): %s", source_name)

    loader = PyPDFLoader(str(file_path))
    pages = loader.load()

    for i, page_doc in enumerate(pages):
        page_doc.metadata.update({
            "source": source_name,
            "page": i,
            "content_type": "text",
            "file_path": str(file_path),
            "doc_hash": doc_hash,
        })

    logger.info("Loaded %d pages from %s (pypdf)", len(pages), source_name)
    return pages


# ---------------------------------------------------------------------------
# Public API — auto-selects parser based on config
# ---------------------------------------------------------------------------

def load_pdf(file_path: Path, settings=None) -> list[Document]:
    """
    Load a PDF using the configured parser backend.

    Primary: Azure Document Intelligence (structured markdown, table/image extraction)
    Fallback: pypdf (raw text)
    """
    from src.config import get_settings

    s = settings or get_settings()
    file_path = Path(file_path).resolve()

    if not file_path.suffix.lower() == ".pdf":
        raise ValueError(f"Not a PDF file: {file_path}")

    if s.parser_backend == "azure_di":
        try:
            return load_pdf_azure_di(
                file_path=file_path,
                endpoint=s.documentintelligence_endpoint,
                key=s.documentintelligence_api_key,
                model_id=s.documentintelligence_model,
                summarize_images=s.image_summary_enabled,
                azure_openai_endpoint=s.azure_openai_endpoint,
                azure_openai_api_key=s.azure_openai_api_key,
                azure_openai_deployment=s.azure_openai_deployment,
                azure_openai_api_version=s.effective_api_version,
            )
        except Exception:
            logger.exception(
                "Azure DI failed for %s — falling back to pypdf", file_path.name
            )
            return load_pdf_pypdf(file_path)

    return load_pdf_pypdf(file_path)


def load_corpus(corpus_dir: Path, settings=None) -> dict[str, list[Document]]:
    """
    Load all PDF files in a directory using the configured parser.

    Returns:
        Dict mapping filename -> list of Documents (with content_type metadata).
    """
    corpus_dir = Path(corpus_dir).resolve()
    if not corpus_dir.is_dir():
        raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")

    pdf_files = sorted(corpus_dir.glob("*.pdf"))
    if not pdf_files:
        logger.warning("No PDF files found in %s", corpus_dir)
        return {}

    logger.info("Found %d PDF files in %s", len(pdf_files), corpus_dir)

    corpus: dict[str, list[Document]] = {}
    for pdf_path in pdf_files:
        try:
            corpus[pdf_path.name] = load_pdf(pdf_path, settings=settings)
        except Exception:
            logger.exception("Failed to load %s", pdf_path.name)

    return corpus

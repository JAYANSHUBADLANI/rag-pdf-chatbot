"""
pdf_processor.py
────────────────
Handles PDF loading, text extraction, cleaning, and chunking.

Author : Jayanshu Badlani
GitHub : https://github.com/JAYANSHUBADLANI
"""

from __future__ import annotations

import re
import logging
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
CHUNK_SIZE    = 1000   # characters per chunk
CHUNK_OVERLAP = 200    # overlap between consecutive chunks
MIN_CHUNK_LEN = 50     # discard chunks shorter than this


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _clean_text(text: str) -> str:
    """
    Normalise whitespace, remove control characters, and strip leading/
    trailing blank lines from a raw PDF text extraction.
    """
    # Replace non-breaking spaces and other unicode whitespace
    text = re.sub(r"[\xa0\u200b\ufeff]", " ", text)
    # Collapse runs of whitespace (excluding newlines) to a single space
    text = re.sub(r"[^\S\n]+", " ", text)
    # Collapse 3+ consecutive newlines to two (preserve paragraph breaks)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip leading / trailing whitespace per line
    lines = [line.strip() for line in text.splitlines()]
    text  = "\n".join(lines)
    return text.strip()


def _add_metadata(doc: Document, pdf_path: str, page_count: int) -> Document:
    """Enrich a Document's metadata with source path and page context."""
    doc.metadata.setdefault("source", Path(pdf_path).name)
    doc.metadata["total_pages"]   = page_count
    doc.metadata["page_label"]    = f"p.{doc.metadata.get('page', 0) + 1}"
    return doc


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def load_pdf(pdf_path: str) -> List[Document]:
    """
    Load a PDF file and return a list of raw LangChain Documents (one per page).

    Parameters
    ----------
    pdf_path : str
        Absolute or relative path to the PDF file.

    Returns
    -------
    List[Document]
        One Document per page, with page-level metadata.

    Raises
    ------
    FileNotFoundError
        If the PDF does not exist at the given path.
    ValueError
        If the PDF contains no extractable text.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    logger.info("Loading PDF: %s", path.name)
    loader = PyPDFLoader(str(path))
    pages  = loader.load()

    if not pages:
        raise ValueError(f"No pages extracted from: {pdf_path}")

    page_count = len(pages)
    logger.info("Loaded %d page(s) from '%s'", page_count, path.name)

    # Clean text and enrich metadata for every page
    cleaned: List[Document] = []
    for doc in pages:
        doc.page_content = _clean_text(doc.page_content)
        doc = _add_metadata(doc, str(path), page_count)
        if len(doc.page_content) >= MIN_CHUNK_LEN:
            cleaned.append(doc)

    logger.info("%d page(s) retained after cleaning (min length %d chars)",
                len(cleaned), MIN_CHUNK_LEN)
    return cleaned


def split_documents(
    documents: List[Document],
    chunk_size:    int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Document]:
    """
    Split a list of Documents into smaller, overlapping chunks suitable for
    embedding and retrieval.

    Parameters
    ----------
    documents : List[Document]
        Raw Documents (e.g. output of :func:`load_pdf`).
    chunk_size : int
        Maximum number of characters per chunk.
    chunk_overlap : int
        Number of characters to overlap between consecutive chunks.

    Returns
    -------
    List[Document]
        Chunked Documents with inherited metadata and a `chunk_index` field.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size        = chunk_size,
        chunk_overlap     = chunk_overlap,
        length_function   = len,
        separators        = ["\n\n", "\n", ". ", "! ", "? ", " ", ""],
    )

    chunks = splitter.split_documents(documents)

    # Tag each chunk with its index and character count
    valid_chunks: List[Document] = []
    for i, chunk in enumerate(chunks):
        if len(chunk.page_content.strip()) < MIN_CHUNK_LEN:
            continue
        chunk.metadata["chunk_index"] = i
        chunk.metadata["chunk_chars"] = len(chunk.page_content)
        valid_chunks.append(chunk)

    logger.info(
        "Split into %d chunks (size=%d, overlap=%d)",
        len(valid_chunks), chunk_size, chunk_overlap,
    )
    return valid_chunks


def process_pdf(
    pdf_path: str,
    chunk_size:    int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Document]:
    """
    End-to-end convenience function: load → clean → split.

    Parameters
    ----------
    pdf_path : str
        Path to the PDF file.
    chunk_size : int
        Characters per chunk (default 1000).
    chunk_overlap : int
        Character overlap between chunks (default 200).

    Returns
    -------
    List[Document]
        Ready-to-embed document chunks.
    """
    pages  = load_pdf(pdf_path)
    chunks = split_documents(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    logger.info(
        "process_pdf complete — %d chunks ready for embedding.", len(chunks)
    )
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# CLI smoke-test  (python -m src.pdf_processor <path/to/file.pdf>)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python pdf_processor.py <path/to/file.pdf>")
        sys.exit(1)

    chunks = process_pdf(sys.argv[1])
    print(f"\nTotal chunks : {len(chunks)}")
    print(f"First chunk  :\n{chunks[0].page_content[:300]}")
    print(f"Metadata     : {chunks[0].metadata}")

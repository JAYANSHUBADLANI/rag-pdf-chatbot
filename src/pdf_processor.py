"""PDF loading, text cleaning, and chunking."""

import logging
import re
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# NOTE: no logging.basicConfig() here; library modules should only create
# loggers; the application entry point (app.py / __main__) owns configuration.
logger = logging.getLogger(__name__)

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MIN_CHUNK_LEN = 50


class EmptyPdfError(ValueError):
    """Raised when a PDF yields no extractable text.

    Typically means the file is a scanned/image-only PDF that would need OCR,
    or every page was shorter than MIN_CHUNK_LEN after cleaning.
    """


def _clean_text(text: str) -> str:
    # normalize invisible unicode: non-breaking space (\xa0) and BOM (﻿)
    text = re.sub(r"[\xa0﻿]", " ", text)
    # collapse runs of whitespace but keep newlines
    text = re.sub(r"[^\S\n]+", " ", text)
    # max two consecutive newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join(lines).strip()


def _add_metadata(doc: Document, pdf_path: str, page_count: int) -> Document:
    doc.metadata["source"] = Path(pdf_path).name
    doc.metadata["total_pages"] = page_count
    doc.metadata["page_label"] = f"p.{doc.metadata.get('page', 0) + 1}"
    return doc


def load_pdf(pdf_path: str) -> List[Document]:
    """Load a PDF and return cleaned per-page Documents.

    Raises:
        FileNotFoundError: if the path does not exist.
        EmptyPdfError: if no page contains extractable text.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    logger.info("Loading PDF: %s", path.name)
    loader = PyPDFLoader(str(path))
    pages = loader.load()

    if not pages:
        raise EmptyPdfError(f"No pages could be extracted from '{path.name}'.")

    page_count = len(pages)
    logger.info("Loaded %d page(s) from '%s'", page_count, path.name)

    cleaned = []
    for doc in pages:
        doc.page_content = _clean_text(doc.page_content)
        doc = _add_metadata(doc, str(path), page_count)
        if len(doc.page_content) >= MIN_CHUNK_LEN:
            cleaned.append(doc)

    if not cleaned:
        raise EmptyPdfError(
            f"'{path.name}' contains no extractable text. "
            "It may be a scanned/image-only PDF that requires OCR."
        )

    logger.info("%d page(s) retained after cleaning", len(cleaned))
    return cleaned


def split_documents(
    documents: List[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Document]:
    """Split page Documents into overlapping chunks ready for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
    )

    chunks = splitter.split_documents(documents)

    # Filter first, then index, so chunk_index is contiguous (0..n-1).
    valid_chunks = [
        chunk for chunk in chunks if len(chunk.page_content.strip()) >= MIN_CHUNK_LEN
    ]
    for i, chunk in enumerate(valid_chunks):
        chunk.metadata["chunk_index"] = i
        chunk.metadata["chunk_chars"] = len(chunk.page_content)

    logger.info(
        "Split into %d chunks (size=%d, overlap=%d)",
        len(valid_chunks), chunk_size, chunk_overlap,
    )
    return valid_chunks


def process_pdf(
    pdf_path: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Document]:
    """Full pipeline: load → clean → chunk. Raises EmptyPdfError when unusable."""
    pages = load_pdf(pdf_path)
    chunks = split_documents(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if not chunks:
        raise EmptyPdfError(
            f"'{Path(pdf_path).name}' produced no usable chunks after splitting."
        )
    logger.info("process_pdf complete, %d chunks ready for embedding.", len(chunks))
    return chunks


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    if len(sys.argv) < 2:
        print("Usage: python pdf_processor.py <path/to/file.pdf>")
        sys.exit(1)

    result = process_pdf(sys.argv[1])
    print(f"\nTotal chunks : {len(result)}")
    print(f"First chunk  :\n{result[0].page_content[:300]}")
    print(f"Metadata     : {result[0].metadata}")

import re
import logging
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MIN_CHUNK_LEN = 50


def _clean_text(text: str) -> str:
    # collapse unicode whitespace variants
    text = re.sub(r"[\xa0​﻿]", " ", text)
    # collapse whitespace but keep newlines
    text = re.sub(r"[^\S\n]+", " ", text)
    # max two consecutive newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join(lines).strip()


def _add_metadata(doc: Document, pdf_path: str, page_count: int) -> Document:
    doc.metadata.setdefault("source", Path(pdf_path).name)
    doc.metadata["total_pages"] = page_count
    doc.metadata["page_label"] = f"p.{doc.metadata.get('page', 0) + 1}"
    return doc


def load_pdf(pdf_path: str) -> List[Document]:
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    logger.info("Loading PDF: %s", path.name)
    loader = PyPDFLoader(str(path))
    pages = loader.load()

    if not pages:
        raise ValueError(f"No pages extracted from: {pdf_path}")

    page_count = len(pages)
    logger.info("Loaded %d page(s) from '%s'", page_count, path.name)

    cleaned = []
    for doc in pages:
        doc.page_content = _clean_text(doc.page_content)
        doc = _add_metadata(doc, str(path), page_count)
        if len(doc.page_content) >= MIN_CHUNK_LEN:
            cleaned.append(doc)

    logger.info("%d page(s) retained after cleaning", len(cleaned))
    return cleaned


def split_documents(
    documents: List[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
    )

    chunks = splitter.split_documents(documents)

    valid_chunks = []
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
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Document]:
    pages = load_pdf(pdf_path)
    chunks = split_documents(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    logger.info("process_pdf complete — %d chunks ready for embedding.", len(chunks))
    return chunks


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python pdf_processor.py <path/to/file.pdf>")
        sys.exit(1)

    chunks = process_pdf(sys.argv[1])
    print(f"\nTotal chunks : {len(chunks)}")
    print(f"First chunk  :\n{chunks[0].page_content[:300]}")
    print(f"Metadata     : {chunks[0].metadata}")

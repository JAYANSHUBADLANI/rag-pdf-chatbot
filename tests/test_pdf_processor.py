"""Tests for PDF loading, cleaning and chunking."""

from pathlib import Path

import pytest
from langchain_core.documents import Document

from src.pdf_processor import (
    MIN_CHUNK_LEN,
    EmptyPdfError,
    _clean_text,
    load_pdf,
    process_pdf,
    split_documents,
)

SAMPLE_PDF = Path(__file__).resolve().parents[1] / "assets" / "sample.pdf"


def test_clean_text_collapses_spaces_and_tabs():
    assert _clean_text("hello     world\tagain") == "hello world again"


def test_clean_text_replaces_invisible_unicode():
    # non-breaking space and BOM both survive PDF extraction and break matching
    cleaned = _clean_text("page\xa0one﻿two")
    assert "\xa0" not in cleaned
    assert "﻿" not in cleaned
    assert cleaned == "page one two"


def test_clean_text_caps_consecutive_newlines():
    assert _clean_text("a\n\n\n\n\nb") == "a\n\nb"


def test_clean_text_strips_each_line():
    assert _clean_text("  first  \n   second   ") == "first\nsecond"


def test_split_documents_indexes_chunks_contiguously():
    long_text = "Retrieval augmented generation grounds answers in documents. " * 60
    chunks = split_documents([Document(page_content=long_text, metadata={"page": 0})])

    assert len(chunks) > 1
    assert [c.metadata["chunk_index"] for c in chunks] == list(range(len(chunks)))


def test_split_documents_records_chunk_length():
    long_text = "Vector search finds nearest neighbours in embedding space. " * 40
    chunks = split_documents([Document(page_content=long_text, metadata={"page": 0})])

    for chunk in chunks:
        assert chunk.metadata["chunk_chars"] == len(chunk.page_content)


def test_split_documents_respects_chunk_size():
    long_text = "Cross encoders rerank candidates returned by the retriever. " * 40
    chunks = split_documents(
        [Document(page_content=long_text, metadata={"page": 0})],
        chunk_size=300,
        chunk_overlap=50,
    )

    assert chunks
    # the splitter may overshoot slightly on an unsplittable run, so allow a margin
    assert max(len(c.page_content) for c in chunks) <= 360


def test_split_documents_drops_chunks_below_minimum():
    tiny = Document(page_content="too short", metadata={"page": 0})
    assert split_documents([tiny]) == []


def test_split_documents_keeps_page_metadata():
    text = "Chunk metadata has to survive splitting for citations to work. " * 20
    chunks = split_documents([Document(page_content=text, metadata={"page": 4})])

    assert chunks
    assert all(c.metadata["page"] == 4 for c in chunks)


def test_load_pdf_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_pdf("does/not/exist.pdf")


def test_load_pdf_on_blank_pdf_raises_empty_pdf_error(tmp_path):
    # a page with no text layer is the scanned-PDF case the error exists for
    pypdf = pytest.importorskip("pypdf")
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=200, height=200)
    blank = tmp_path / "blank.pdf"
    with open(blank, "wb") as handle:
        writer.write(handle)

    with pytest.raises(EmptyPdfError):
        load_pdf(str(blank))


@pytest.mark.skipif(not SAMPLE_PDF.exists(), reason="bundled sample PDF is missing")
def test_process_pdf_on_bundled_sample():
    chunks = process_pdf(str(SAMPLE_PDF))

    assert chunks
    assert all(len(c.page_content) >= MIN_CHUNK_LEN for c in chunks)
    assert all(c.metadata["source"] == "sample.pdf" for c in chunks)
    assert all(c.metadata["page_label"].startswith("p.") for c in chunks)
    # total_pages is what the UI shows after indexing
    assert chunks[0].metadata["total_pages"] >= 1


@pytest.mark.skipif(not SAMPLE_PDF.exists(), reason="bundled sample PDF is missing")
def test_page_labels_are_one_based():
    pages = load_pdf(str(SAMPLE_PDF))
    first = pages[0]
    assert first.metadata["page_label"] == f"p.{first.metadata['page'] + 1}"

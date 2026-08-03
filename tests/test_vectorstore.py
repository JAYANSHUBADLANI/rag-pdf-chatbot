"""Tests for FAISS index build, add, persist and load."""

import pytest

from src.vectorstore import (
    add_to_vectorstore,
    build_vectorstore,
    load_vectorstore,
    vectorstore_exists,
)

from conftest import make_doc


def test_build_rejects_empty_chunks(embeddings):
    with pytest.raises(ValueError):
        build_vectorstore([], embedding_model=embeddings)


def test_similarity_search_returns_the_relevant_chunk(sample_chunks, embeddings):
    vs = build_vectorstore(sample_chunks, embedding_model=embeddings)

    hits = vs.similarity_search("how does photosynthesis work", k=1)

    assert "Photosynthesis" in hits[0].page_content


def test_build_does_not_persist_by_default(sample_chunks, embeddings, tmp_path):
    build_vectorstore(sample_chunks, embedding_model=embeddings)

    # the Streamlit app relies on this: a shared index dir would let concurrent
    # sessions overwrite each other
    assert not vectorstore_exists(save_dir=str(tmp_path))


def test_save_and_load_roundtrip(sample_chunks, embeddings, tmp_path):
    build_vectorstore(sample_chunks, embedding_model=embeddings, save_dir=str(tmp_path))
    assert vectorstore_exists(save_dir=str(tmp_path))

    reloaded = load_vectorstore(save_dir=str(tmp_path), embedding_model=embeddings)
    hits = reloaded.similarity_search("similarity search library", k=1)

    assert "FAISS" in hits[0].page_content


def test_load_missing_index_raises(tmp_path, embeddings):
    with pytest.raises(FileNotFoundError):
        load_vectorstore(save_dir=str(tmp_path / "nothing"), embedding_model=embeddings)


def test_add_empty_chunk_list_is_a_noop(sample_chunks, embeddings):
    vs = build_vectorstore(sample_chunks, embedding_model=embeddings)
    before = vs.index.ntotal

    add_to_vectorstore(vs, [])

    assert vs.index.ntotal == before


def test_add_makes_new_chunks_retrievable(sample_chunks, embeddings):
    vs = build_vectorstore(sample_chunks, embedding_model=embeddings)
    extra = make_doc("Groq serves Llama models with very low latency.", page=3)

    add_to_vectorstore(vs, [extra])
    hits = vs.similarity_search("Groq Llama latency", k=1)

    assert "Groq" in hits[0].page_content
    assert vs.index.ntotal == len(sample_chunks) + 1


def test_add_persists_when_save_dir_given(sample_chunks, embeddings, tmp_path):
    vs = build_vectorstore(sample_chunks, embedding_model=embeddings)
    extra = make_doc("Reranking reorders the candidate passages.", page=4)

    add_to_vectorstore(vs, [extra], save_dir=str(tmp_path))

    assert vectorstore_exists(save_dir=str(tmp_path))
    reloaded = load_vectorstore(save_dir=str(tmp_path), embedding_model=embeddings)
    assert reloaded.index.ntotal == len(sample_chunks) + 1

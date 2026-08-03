"""Tests for the RAG pipeline: prompt assembly, citations, reranking, streaming."""

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from src.rag_chain import (
    HISTORY_CONTENT_TRUNCATE,
    RagPipeline,
    _build_messages,
    _dedup_pages,
    _extract_text,
    _format_docs,
    _format_history_block,
    _format_history_for_rewrite,
    _truncate,
    get_llm,
)
from src.vectorstore import build_vectorstore

from conftest import StubCrossEncoder, make_chat_model, make_doc

ANSWER = "photosynthesis converts light energy (p.1)"


@pytest.fixture
def pipeline(monkeypatch, sample_chunks, embeddings):
    """A RagPipeline wired to stand-ins, so no key and no downloads are needed."""
    # get_llm is called twice in __init__: answer model first, then rewriter
    llms = iter([make_chat_model(ANSWER), make_chat_model("how does photosynthesis work")])
    monkeypatch.setattr("src.rag_chain.get_llm", lambda *a, **kw: next(llms))
    monkeypatch.setattr(
        RagPipeline,
        "_get_cross_encoder",
        classmethod(lambda cls, model_name=None: StubCrossEncoder()),
    )

    vs = build_vectorstore(sample_chunks, embedding_model=embeddings)
    return RagPipeline(vs, top_k=2, rerank_candidates=3)


def test_truncate_leaves_short_text_alone():
    assert _truncate("short", 20) == "short"


def test_truncate_marks_clipped_text():
    out = _truncate("a" * 50, 10)
    assert len(out) == 11
    assert out.endswith("…")


def test_format_docs_prefixes_source_and_page():
    formatted = _format_docs([make_doc("body text", page=2, source="report.pdf")])
    assert "report.pdf" in formatted
    assert "p.3" in formatted
    assert "body text" in formatted


def test_format_docs_separates_multiple_chunks():
    formatted = _format_docs([make_doc("first", page=0), make_doc("second", page=1)])
    assert "---" in formatted


def test_dedup_pages_removes_repeats_and_keeps_order():
    docs = [make_doc("a", page=2), make_doc("b", page=0), make_doc("c", page=2)]
    assert _dedup_pages(docs) == ["p.3", "p.1"]


def test_dedup_pages_qualifies_labels_when_sources_differ():
    docs = [
        make_doc("a", page=0, source="alpha.pdf"),
        make_doc("b", page=0, source="beta.pdf"),
    ]
    labels = _dedup_pages(docs)

    # same page number in two files must not collapse into one citation
    assert labels == ["alpha · p.1", "beta · p.1"]


def test_dedup_pages_ignores_docs_without_a_label():
    docs = [make_doc("a", page=0)]
    docs[0].metadata["page_label"] = ""
    assert _dedup_pages(docs) == []


def test_format_history_for_rewrite_keeps_only_recent_turns(chat_history):
    formatted = _format_history_for_rewrite(chat_history)

    assert "what about overlap" in formatted
    assert formatted.startswith("User:") or formatted.startswith("Assistant:")


def test_format_history_for_rewrite_empty_history():
    assert _format_history_for_rewrite([]) == ""


def test_format_history_truncates_long_messages():
    history = [{"role": "assistant", "content": "x" * (HISTORY_CONTENT_TRUNCATE + 200)}]
    formatted = _format_history_for_rewrite(history)
    assert len(formatted) < HISTORY_CONTENT_TRUNCATE + 100


def test_format_history_block_empty_history():
    assert _format_history_block([]) == ""


def test_format_history_block_is_labelled(chat_history):
    assert _format_history_block(chat_history).startswith("RECENT CONVERSATION:")


def test_extract_text_from_plain_string():
    assert _extract_text("hello") == "hello"


def test_extract_text_from_content_blocks():
    blocks = [{"type": "text", "text": "part one "}, {"type": "text", "text": "part two"}]
    assert _extract_text(blocks) == "part one part two"


def test_extract_text_ignores_non_text_blocks():
    blocks = [{"type": "image", "url": "x"}, {"type": "text", "text": "kept"}]
    assert _extract_text(blocks) == "kept"


def test_extract_text_on_unexpected_type():
    assert _extract_text(None) == ""


def test_build_messages_puts_rules_in_system_slot():
    messages = _build_messages("some context", "what is this", [])

    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[1], HumanMessage)
    assert "cite the page number" in messages[0].content


def test_build_messages_carries_context_and_question():
    messages = _build_messages("CTX-MARKER", "QUESTION-MARKER", [])
    user_content = messages[1].content

    assert "CTX-MARKER" in user_content
    assert "QUESTION-MARKER" in user_content


def test_get_llm_without_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        get_llm()


def test_get_llm_rejects_unedited_placeholder(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "your_groq_api_key_here")
    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        get_llm()


def test_rerank_orders_by_cross_encoder_score(pipeline):
    docs = [
        make_doc("unrelated text about weather", page=0),
        make_doc("photosynthesis converts light energy", page=1),
    ]

    ranked = pipeline._rerank("photosynthesis light", docs)

    assert "photosynthesis" in ranked[0].page_content


def test_rerank_on_empty_input(pipeline):
    assert pipeline._rerank("anything", []) == []


def test_rewrite_query_without_history_returns_original(pipeline):
    assert pipeline._rewrite_query("what is chunking", []) == "what is chunking"


def test_rewrite_query_falls_back_when_rewriter_fails(pipeline, chat_history, monkeypatch):
    class Boom:
        def invoke(self, *a, **kw):
            raise RuntimeError("groq is down")

    monkeypatch.setattr(pipeline, "_rewrite_chain", Boom())

    # a failing rewrite must degrade to the raw question, not break the turn
    assert pipeline._rewrite_query("what about it", chat_history) == "what about it"


def test_query_returns_answer_and_citations(pipeline):
    result = pipeline.query("how does photosynthesis work")

    assert result["answer"] == ANSWER
    assert result["question"] == "how does photosynthesis work"
    assert len(result["source_chunks"]) <= pipeline.top_k
    assert result["source_pages"]


def test_query_rejects_blank_question(pipeline):
    with pytest.raises(ValueError):
        pipeline.query("   ")


def test_query_respects_top_k(pipeline):
    result = pipeline.query("photosynthesis")
    assert len(result["source_chunks"]) == 2


def test_stream_emits_sources_before_tokens(pipeline):
    events = list(pipeline.stream("how does photosynthesis work"))

    assert events[0]["type"] == "sources"
    assert all(event["type"] == "token" for event in events[1:])


def test_stream_tokens_reassemble_into_the_answer(pipeline):
    events = list(pipeline.stream("how does photosynthesis work"))
    text = "".join(event["text"] for event in events if event["type"] == "token")

    assert text == ANSWER


def test_stream_rejects_blank_question(pipeline):
    with pytest.raises(ValueError):
        list(pipeline.stream(""))

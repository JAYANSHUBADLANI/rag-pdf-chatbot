"""RAG pipeline: query rewriting (fast 8B model), retrieval, cross-encoder
reranking, and streamed answers (Llama 3.3 70B via Groq)."""

import logging
import os
from typing import Any, Dict, Iterator, List, Optional, Tuple

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv()

logger = logging.getLogger(__name__)

ANSWER_MODEL = "llama-3.3-70b-versatile"
REWRITE_MODEL = "llama-3.1-8b-instant"
MAX_TOKENS = 1024
TEMPERATURE = 0.2
TOP_K_RETRIEVE = 4
RERANK_CANDIDATES = 12
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
HISTORY_TURNS_FOR_REWRITE = 2
HISTORY_TURNS_FOR_ANSWER = 3
HISTORY_CONTENT_TRUNCATE = 500


SYSTEM_INSTRUCTIONS = """You are a precise, helpful assistant that answers questions strictly based on the provided document context.

Rules:
- Answer only from the context provided in the user's message. Do not use outside knowledge.
- If the context does not contain enough information to answer the question, say "I don't have enough information in this document to answer that."
- Be concise and direct. Use bullet points when listing multiple items.
- Always cite the page number(s) you referenced, e.g. "(p.3, p.7)"."""


REWRITE_PROMPT_TEMPLATE = """Given the conversation below and a follow-up question, rewrite the follow-up as a standalone question that can be understood without prior context.
- If the follow-up is already self-contained, return it unchanged.
- Return ONLY the rewritten question with no preamble, no quotes, no explanation.

Conversation:
{history}

Follow-up: {question}

Standalone question:"""

REWRITE_PROMPT = PromptTemplate(
    template=REWRITE_PROMPT_TEMPLATE,
    input_variables=["history", "question"],
)


def get_llm(
    model: str = ANSWER_MODEL,
    temperature: float = TEMPERATURE,
    max_tokens: int = MAX_TOKENS,
) -> ChatGroq:
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key or api_key.startswith("your_"):
        raise ValueError(
            "GROQ_API_KEY is not set. Copy .env.example → .env and add your key."
        )
    logger.info("Initialising LLM: %s (temp=%.1f, max_tokens=%d)", model, temperature, max_tokens)
    return ChatGroq(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=api_key,
    )


def _format_docs(docs: List[Document]) -> str:
    parts = []
    for doc in docs:
        label = doc.metadata.get("page_label", "")
        source = doc.metadata.get("source", "")
        prefix_parts = [p for p in [source, label] if p]
        prefix = f"[{' · '.join(prefix_parts)}] " if prefix_parts else ""
        parts.append(prefix + doc.page_content)
    return "\n\n---\n\n".join(parts)


def _dedup_pages(docs: List[Document]) -> List[str]:
    sources = {doc.metadata.get("source", "") for doc in docs}
    multi_source = len(sources) > 1
    seen = set()
    labels: List[str] = []
    for doc in docs:
        page = doc.metadata.get("page_label", "")
        source = doc.metadata.get("source", "")
        if not page:
            continue
        if multi_source and source:
            stem = source.rsplit(".", 1)[0]
            label = f"{stem} · {page}"
        else:
            label = page
        if label in seen:
            continue
        seen.add(label)
        labels.append(label)
    return labels


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


def _format_history_for_rewrite(history: List[Dict[str, str]]) -> str:
    if not history:
        return ""
    recent = history[-(HISTORY_TURNS_FOR_REWRITE * 2):]
    lines = []
    for msg in recent:
        role = "User" if msg.get("role") == "user" else "Assistant"
        content = _truncate(msg.get("content", ""), HISTORY_CONTENT_TRUNCATE)
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _format_history_block(history: List[Dict[str, str]]) -> str:
    if not history:
        return ""
    recent = history[-(HISTORY_TURNS_FOR_ANSWER * 2):]
    lines = ["RECENT CONVERSATION:"]
    for msg in recent:
        role = "User" if msg.get("role") == "user" else "Assistant"
        content = _truncate(msg.get("content", ""), HISTORY_CONTENT_TRUNCATE)
        lines.append(f"{role}: {content}")
    return "\n".join(lines) + "\n\n"


def _build_messages(
    context: str,
    question: str,
    history: List[Dict[str, str]],
) -> List[Any]:
    # Prompt caching intentionally NOT used here. The retrieved context changes
    # with almost every question, so a cache_control breakpoint on it would pay
    # the cache-write premium and virtually never hit; the static system prompt
    # is well under the minimum cacheable size. If you later switch to
    # full-document stuffing (same large context every turn), re-add
    # {"cache_control": {"type": "ephemeral"}} on the context block; that is
    # where caching pays off.
    history_text = _format_history_block(history)
    user_content = (
        f"DOCUMENT CONTEXT:\n{context}\n\n"
        f"{history_text}"
        f"QUESTION:\n{question}\n\n"
        f"ANSWER:"
    )
    return [SystemMessage(content=SYSTEM_INSTRUCTIONS), HumanMessage(content=user_content)]


def _extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return ""


class RagPipeline:
    """End-to-end RAG pipeline.

    Note: `chat_history` passed to query()/stream() should contain only
    successful turns; the caller is expected to filter out error messages so
    they don't pollute query rewriting or the answer prompt.
    """

    _cross_encoder_cache: Any = None

    @classmethod
    def _get_cross_encoder(cls, model_name: str = CROSS_ENCODER_MODEL):
        if cls._cross_encoder_cache is None:
            from sentence_transformers import CrossEncoder
            cls._cross_encoder_cache = CrossEncoder(model_name)
            logger.info("Loaded cross-encoder: %s", model_name)
        return cls._cross_encoder_cache

    def __init__(
        self,
        vectorstore: FAISS,
        model: str = ANSWER_MODEL,
        rewrite_model: str = REWRITE_MODEL,
        temperature: float = TEMPERATURE,
        max_tokens: int = MAX_TOKENS,
        top_k: int = TOP_K_RETRIEVE,
        enable_rerank: bool = True,
        rerank_candidates: int = RERANK_CANDIDATES,
    ):
        self.llm = get_llm(model=model, temperature=temperature, max_tokens=max_tokens)
        self.rewrite_llm = get_llm(model=rewrite_model, temperature=0.0, max_tokens=200)
        self.top_k = top_k
        self.enable_rerank = enable_rerank
        self.rerank_candidates = max(rerank_candidates, top_k)
        retrieval_k = self.rerank_candidates if enable_rerank else top_k
        self.retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": retrieval_k},
        )
        self._rewrite_chain = REWRITE_PROMPT | self.rewrite_llm | StrOutputParser()
        if enable_rerank:
            self._get_cross_encoder()
        logger.info(
            "RAG pipeline ready (answer=%s, rewrite=%s, top_k=%d, rerank=%s)",
            model, rewrite_model, top_k, enable_rerank,
        )

    def _rewrite_query(self, question: str, history: List[Dict[str, str]]) -> str:
        if not history:
            return question
        formatted = _format_history_for_rewrite(history)
        try:
            rewritten = self._rewrite_chain.invoke(
                {"history": formatted, "question": question}
            ).strip()
        except Exception as exc:
            logger.warning("Query rewrite failed (%s); using original question.", exc)
            return question
        if not rewritten or rewritten.lower() == question.lower():
            return question
        logger.info("Rewrote query: %r -> %r", question, rewritten)
        return rewritten

    def _rerank(self, query: str, docs: List[Document]) -> List[Document]:
        if not docs:
            return docs
        pairs = [(query, doc.page_content) for doc in docs]
        scores = self._get_cross_encoder().predict(pairs)
        ranked = sorted(zip(scores, docs), key=lambda pair: pair[0], reverse=True)
        return [doc for _, doc in ranked]

    def _retrieve(self, query: str) -> Tuple[List[Document], List[str]]:
        docs = self.retriever.invoke(query)
        if self.enable_rerank and docs:
            docs = self._rerank(query, docs)
        docs = docs[: self.top_k]
        return docs, _dedup_pages(docs)

    def query(
        self,
        question: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        if not question.strip():
            raise ValueError("Question must not be empty.")
        history = chat_history or []
        search_query = self._rewrite_query(question, history)
        docs, pages = self._retrieve(search_query)
        context = _format_docs(docs)
        messages = _build_messages(context, question, history)
        response = self.llm.invoke(messages)
        answer = _extract_text(response.content)
        return {
            "question": question,
            "search_query": search_query,
            "answer": answer.strip(),
            "source_chunks": docs,
            "source_pages": pages,
        }

    def stream(
        self,
        question: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> Iterator[Dict[str, Any]]:
        """Yields one {"type": "sources", ...} event, then {"type": "token", ...} events."""
        if not question.strip():
            raise ValueError("Question must not be empty.")
        history = chat_history or []
        search_query = self._rewrite_query(question, history)
        docs, pages = self._retrieve(search_query)
        yield {
            "type": "sources",
            "source_chunks": docs,
            "source_pages": pages,
            "search_query": search_query,
        }
        context = _format_docs(docs)
        messages = _build_messages(context, question, history)
        for chunk in self.llm.stream(messages):
            text = _extract_text(chunk.content)
            if text:
                yield {"type": "token", "text": text}

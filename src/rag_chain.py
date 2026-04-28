"""
rag_chain.py
────────────
LangChain RAG pipeline using Claude claude-opus-4-6 as the LLM.

Flow:
    user query
        → FAISS retriever (top-k chunks)
        → custom prompt template
        → Claude claude-opus-4-6  (via Anthropic API)
        → answer + source chunks returned to caller

Author : Jayanshu Badlani
GitHub : https://github.com/JAYANSHUBADLANI
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain_community.vectorstores import FAISS

load_dotenv()

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ── Model config ──────────────────────────────────────────────────────────────
CLAUDE_MODEL   = "claude-opus-4-6"
MAX_TOKENS     = 1024
TEMPERATURE    = 0.2     # low temperature → factual, grounded answers
TOP_K_RETRIEVE = 4       # chunks retrieved per query

# ─────────────────────────────────────────────────────────────────────────────
# Prompt Template
# ─────────────────────────────────────────────────────────────────────────────

RAG_PROMPT_TEMPLATE = """You are a precise, helpful assistant that answers questions \
strictly based on the provided document context.

Rules:
- Answer only from the context below. Do not use outside knowledge.
- If the context does not contain enough information to answer the question, \
say "I don't have enough information in this document to answer that."
- Be concise and direct. Use bullet points when listing multiple items.
- Always cite the page number(s) you referenced, e.g. "(p.3, p.7)".

─────────────────────────
CONTEXT:
{context}
─────────────────────────

QUESTION:
{question}

ANSWER:"""

RAG_PROMPT = PromptTemplate(
    template       = RAG_PROMPT_TEMPLATE,
    input_variables = ["context", "question"],
)


# ─────────────────────────────────────────────────────────────────────────────
# LLM factory
# ─────────────────────────────────────────────────────────────────────────────

def get_llm(
    model: str       = CLAUDE_MODEL,
    temperature: float = TEMPERATURE,
    max_tokens: int  = MAX_TOKENS,
) -> ChatAnthropic:
    """
    Instantiate and return a Claude LLM via the Anthropic API.

    The API key is read from the ``ANTHROPIC_API_KEY`` environment variable
    (populated by ``python-dotenv`` from ``.env``).

    Parameters
    ----------
    model : str
        Claude model identifier (default ``"claude-opus-4-6"``).
    temperature : float
        Sampling temperature — lower = more deterministic.
    max_tokens : int
        Maximum tokens in the model's response.

    Returns
    -------
    ChatAnthropic

    Raises
    ------
    ValueError
        If ``ANTHROPIC_API_KEY`` is not set.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key or api_key.startswith("your_"):
        raise ValueError(
            "ANTHROPIC_API_KEY is not set. "
            "Copy .env.example → .env and add your key."
        )

    logger.info("Initialising LLM: %s (temp=%.1f, max_tokens=%d)",
                model, temperature, max_tokens)

    return ChatAnthropic(
        model        = model,
        temperature  = temperature,
        max_tokens   = max_tokens,
        anthropic_api_key = api_key,
    )


# ─────────────────────────────────────────────────────────────────────────────
# RAG Chain builder
# ─────────────────────────────────────────────────────────────────────────────

def build_rag_chain(
    vectorstore: FAISS,
    top_k: int         = TOP_K_RETRIEVE,
    model: str         = CLAUDE_MODEL,
    temperature: float = TEMPERATURE,
    max_tokens: int    = MAX_TOKENS,
) -> RetrievalQA:
    """
    Build and return a LangChain ``RetrievalQA`` chain.

    The chain:
        1. Retrieves the top-k most relevant chunks from ``vectorstore``.
        2. Injects them into :data:`RAG_PROMPT`.
        3. Sends the filled prompt to Claude.
        4. Returns the answer string **and** the source documents.

    Parameters
    ----------
    vectorstore : FAISS
        Populated FAISS vector store (from :mod:`src.vectorstore`).
    top_k : int
        Number of chunks to retrieve per query.
    model : str
        Claude model string.
    temperature : float
        Sampling temperature.
    max_tokens : int
        Max tokens in Claude's response.

    Returns
    -------
    RetrievalQA
    """
    llm       = get_llm(model=model, temperature=temperature, max_tokens=max_tokens)
    retriever = vectorstore.as_retriever(
        search_type   = "similarity",
        search_kwargs = {"k": top_k},
    )

    chain = RetrievalQA.from_chain_type(
        llm              = llm,
        chain_type       = "stuff",          # concatenate all chunks into one prompt
        retriever        = retriever,
        return_source_documents = True,      # include source chunks in result
        chain_type_kwargs = {"prompt": RAG_PROMPT},
    )

    logger.info("RAG chain built (model=%s, top_k=%d).", model, top_k)
    return chain


# ─────────────────────────────────────────────────────────────────────────────
# Query helper
# ─────────────────────────────────────────────────────────────────────────────

def query_rag(
    chain: RetrievalQA,
    question: str,
) -> Dict[str, Any]:
    """
    Run a question through the RAG chain and return a structured result.

    Parameters
    ----------
    chain : RetrievalQA
        Built chain from :func:`build_rag_chain`.
    question : str
        Natural-language question about the document.

    Returns
    -------
    dict with keys:
        - ``answer``          : str — model's answer
        - ``source_chunks``   : List[Document] — retrieved context chunks
        - ``source_pages``    : List[str] — human-readable page labels
        - ``question``        : str — echoed for logging / display
    """
    if not question.strip():
        raise ValueError("Question must not be empty.")

    logger.info("RAG query: %r", question[:80])
    result = chain.invoke({"query": question})

    answer        = result.get("result", "").strip()
    source_docs: List[Document] = result.get("source_documents", [])

    # Deduplicate page labels
    seen_pages: set = set()
    source_pages: List[str] = []
    for doc in source_docs:
        label = doc.metadata.get("page_label", "")
        if label and label not in seen_pages:
            source_pages.append(label)
            seen_pages.add(label)

    logger.info(
        "Answer generated (%d chars) from %d source chunk(s).",
        len(answer), len(source_docs),
    )

    return {
        "question":     question,
        "answer":       answer,
        "source_chunks": source_docs,
        "source_pages":  source_pages,
    }



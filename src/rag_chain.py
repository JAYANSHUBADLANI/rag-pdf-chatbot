import logging
import os
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_community.vectorstores import FAISS

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

CLAUDE_MODEL = "claude-opus-4-6"
MAX_TOKENS = 1024
TEMPERATURE = 0.2
TOP_K_RETRIEVE = 4


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
    template=RAG_PROMPT_TEMPLATE,
    input_variables=["context", "question"],
)


def get_llm(
    model: str = CLAUDE_MODEL,
    temperature: float = TEMPERATURE,
    max_tokens: int = MAX_TOKENS,
) -> ChatAnthropic:
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key or api_key.startswith("your_"):
        raise ValueError(
            "ANTHROPIC_API_KEY is not set. Copy .env.example → .env and add your key."
        )
    logger.info("Initialising LLM: %s (temp=%.1f, max_tokens=%d)", model, temperature, max_tokens)
    return ChatAnthropic(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        anthropic_api_key=api_key,
    )


def _format_docs(docs: List[Document]) -> str:
    parts = []
    for doc in docs:
        label = doc.metadata.get("page_label", "")
        prefix = f"[{label}] " if label else ""
        parts.append(prefix + doc.page_content)
    return "\n\n---\n\n".join(parts)


def build_rag_chain(
    vectorstore: FAISS,
    top_k: int = TOP_K_RETRIEVE,
    model: str = CLAUDE_MODEL,
    temperature: float = TEMPERATURE,
    max_tokens: int = MAX_TOKENS,
) -> Any:
    llm = get_llm(model=model, temperature=temperature, max_tokens=max_tokens)
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k},
    )

    # retrieve → format → prompt → LLM → parse
    chain = (
        RunnablePassthrough.assign(
            context=(lambda x: x["question"]) | retriever | _format_docs,
            _docs=(lambda x: x["question"]) | retriever,
        )
        | {
            "answer": RAG_PROMPT | llm | StrOutputParser(),
            "source_documents": RunnableLambda(lambda x: x.get("_docs", [])),
        }
    )

    logger.info("RAG chain built (model=%s, top_k=%d).", model, top_k)
    return chain, retriever


def query_rag(chain_tuple: Any, question: str) -> Dict[str, Any]:
    if not question.strip():
        raise ValueError("Question must not be empty.")

    chain, retriever = chain_tuple
    logger.info("RAG query: %r", question[:80])

    source_docs: List[Document] = retriever.invoke(question)
    result = chain.invoke({"question": question})
    answer = result.get("answer", "").strip()

    # deduplicate page labels for display
    seen: set = set()
    source_pages: List[str] = []
    for doc in source_docs:
        label = doc.metadata.get("page_label", "")
        if label and label not in seen:
            source_pages.append(label)
            seen.add(label)

    logger.info(
        "Answer generated (%d chars) from %d source chunk(s).", len(answer), len(source_docs)
    )

    return {
        "question": question,
        "answer": answer,
        "source_chunks": source_docs,
        "source_pages": source_pages,
    }

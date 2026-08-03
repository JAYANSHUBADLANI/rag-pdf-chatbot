"""Shared fixtures and test doubles.

The whole suite runs offline: no GROQ_API_KEY, no model downloads, no network.
That is deliberate. Tests that need an embedding model, a cross-encoder or an
LLM get a deterministic stand-in from this file instead, so the suite stays
fast and reproducible on a fresh clone.
"""

import hashlib
import re
from typing import Dict, List

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.fake_chat_models import FakeListChatModel

TOKEN_RE = re.compile(r"[a-z0-9]+")


class StubEmbeddings(Embeddings):
    """Hashed bag-of-words embeddings.

    Each token is hashed into one of `dim` buckets and the vector is L2
    normalised, so two texts that share vocabulary end up close together.
    That makes similarity search behave sensibly in tests (a query about
    photosynthesis retrieves the photosynthesis chunk) without pulling
    MiniLM off the hub.
    """

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim

    def _vector(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        for token in TOKEN_RE.findall(text.lower()):
            digest = hashlib.md5(token.encode()).hexdigest()
            vec[int(digest, 16) % self.dim] += 1.0
        norm = sum(value * value for value in vec) ** 0.5
        if norm == 0:
            return vec
        return [value / norm for value in vec]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._vector(text)


class StubCrossEncoder:
    """Cross-encoder stand-in that scores by shared-token count.

    Returns a score per (query, passage) pair, same contract as the real
    sentence-transformers CrossEncoder.predict.
    """

    def predict(self, pairs) -> List[float]:
        scores = []
        for query, passage in pairs:
            query_tokens = set(TOKEN_RE.findall(query.lower()))
            passage_tokens = set(TOKEN_RE.findall(passage.lower()))
            scores.append(float(len(query_tokens & passage_tokens)))
        return scores


def make_chat_model(response: str) -> FakeListChatModel:
    """A canned chat model.

    FakeListChatModel is a real BaseChatModel, so it composes with the
    prompt | llm | parser chain in rag_chain and streams like the genuine
    article, which a hand-rolled stub would not.
    """
    return FakeListChatModel(responses=[response])


def make_doc(text: str, page: int = 0, source: str = "doc.pdf") -> Document:
    """Build a Document with the metadata the pipeline expects downstream."""
    return Document(
        page_content=text,
        metadata={"source": source, "page": page, "page_label": f"p.{page + 1}"},
    )


@pytest.fixture
def embeddings() -> StubEmbeddings:
    return StubEmbeddings()


@pytest.fixture
def sample_chunks() -> List[Document]:
    return [
        make_doc("Photosynthesis converts light energy into chemical energy.", page=0),
        make_doc("The Amazon river carries more water than any other river.", page=1),
        make_doc("FAISS is a library for efficient similarity search.", page=2),
    ]


@pytest.fixture
def chat_history() -> List[Dict[str, str]]:
    return [
        {"role": "user", "content": "what is chunking"},
        {"role": "assistant", "content": "splitting a document into pieces"},
        {"role": "user", "content": "what about overlap"},
        {"role": "assistant", "content": "chunks share trailing text"},
    ]

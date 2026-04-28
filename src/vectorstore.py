"""
vectorstore.py
──────────────
FAISS-backed vector store: create, persist, load, and query.

Uses `sentence-transformers/all-MiniLM-L6-v2` for local, free embeddings so
no additional API key is required for indexing.  Swap the embedding model
for any HuggingFace-compatible model without changing the rest of the stack.

Author : Jayanshu Badlani
GitHub : https://github.com/JAYANSHUBADLANI
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple, Optional

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_VECTORSTORE_DIR = "vectorstore"
DEFAULT_INDEX_NAME      = "faiss_index"
DEFAULT_TOP_K           = 4          # documents returned per query
DEFAULT_SCORE_THRESHOLD = 0.0        # minimum relevance score (0 = no filter)


# ─────────────────────────────────────────────────────────────────────────────
# Embedding model factory
# ─────────────────────────────────────────────────────────────────────────────

def get_embeddings(
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    device: str = "cpu",
) -> HuggingFaceEmbeddings:
    """
    Return a HuggingFace embedding model.

    Parameters
    ----------
    model_name : str
        Any sentence-transformers model name, e.g.
        ``"sentence-transformers/all-MiniLM-L6-v2"`` (384-dim, fast) or
        ``"sentence-transformers/all-mpnet-base-v2"`` (768-dim, higher quality).
    device : str
        ``"cpu"`` or ``"cuda"``.

    Returns
    -------
    HuggingFaceEmbeddings
    """
    logger.info("Loading embedding model: %s (device=%s)", model_name, device)
    return HuggingFaceEmbeddings(
        model_name      = model_name,
        model_kwargs    = {"device": device},
        encode_kwargs   = {"normalize_embeddings": True},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Build / save / load
# ─────────────────────────────────────────────────────────────────────────────

def build_vectorstore(
    chunks: List[Document],
    embedding_model: Optional[HuggingFaceEmbeddings] = None,
    save_dir: str  = DEFAULT_VECTORSTORE_DIR,
    index_name: str = DEFAULT_INDEX_NAME,
) -> FAISS:
    """
    Embed a list of document chunks and build a FAISS index.

    The index is immediately persisted to ``save_dir/index_name`` so it can be
    reloaded without re-embedding.

    Parameters
    ----------
    chunks : List[Document]
        Chunked documents from :func:`~src.pdf_processor.split_documents`.
    embedding_model : HuggingFaceEmbeddings, optional
        Pre-instantiated embedding model.  Created fresh if not provided.
    save_dir : str
        Directory where the FAISS index will be saved.
    index_name : str
        Sub-folder name for this index (allows multiple indices in one dir).

    Returns
    -------
    FAISS
        The populated and persisted vector store.

    Raises
    ------
    ValueError
        If ``chunks`` is empty.
    """
    if not chunks:
        raise ValueError("Cannot build a vector store from an empty chunk list.")

    if embedding_model is None:
        embedding_model = get_embeddings()

    logger.info("Building FAISS index from %d chunks …", len(chunks))
    vectorstore = FAISS.from_documents(chunks, embedding_model)

    # Persist to disk
    save_path = Path(save_dir) / index_name
    save_path.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(save_path))
    logger.info("FAISS index saved → %s", save_path)

    return vectorstore


def load_vectorstore(
    save_dir: str   = DEFAULT_VECTORSTORE_DIR,
    index_name: str = DEFAULT_INDEX_NAME,
    embedding_model: Optional[HuggingFaceEmbeddings] = None,
) -> FAISS:
    """
    Load a previously saved FAISS index from disk.

    Parameters
    ----------
    save_dir : str
        Directory containing the saved index.
    index_name : str
        Sub-folder name used when the index was saved.
    embedding_model : HuggingFaceEmbeddings, optional
        Must use the **same** model that was used during :func:`build_vectorstore`.

    Returns
    -------
    FAISS

    Raises
    ------
    FileNotFoundError
        If the index directory does not exist.
    """
    index_path = Path(save_dir) / index_name
    if not index_path.exists():
        raise FileNotFoundError(
            f"No FAISS index found at '{index_path}'. "
            "Run build_vectorstore() first."
        )

    if embedding_model is None:
        embedding_model = get_embeddings()

    logger.info("Loading FAISS index from %s …", index_path)
    vectorstore = FAISS.load_local(
        str(index_path),
        embedding_model,
        allow_dangerous_deserialization=True,  # safe: our own index files
    )
    logger.info("FAISS index loaded successfully.")
    return vectorstore


def vectorstore_exists(
    save_dir: str   = DEFAULT_VECTORSTORE_DIR,
    index_name: str = DEFAULT_INDEX_NAME,
) -> bool:
    """Return True if a persisted FAISS index already exists on disk."""
    return (Path(save_dir) / index_name).exists()


# ─────────────────────────────────────────────────────────────────────────────
# Retrieval
# ─────────────────────────────────────────────────────────────────────────────

def similarity_search(
    vectorstore: FAISS,
    query: str,
    top_k: int           = DEFAULT_TOP_K,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
) -> List[Tuple[Document, float]]:
    """
    Retrieve the top-k most relevant document chunks for a query.

    Parameters
    ----------
    vectorstore : FAISS
        A populated FAISS vector store.
    query : str
        Natural-language question or search string.
    top_k : int
        Number of results to return.
    score_threshold : float
        Minimum relevance score (cosine similarity, 0–1).  Chunks below this
        threshold are filtered out.  Set to 0.0 to disable filtering.

    Returns
    -------
    List[Tuple[Document, float]]
        List of (document, score) pairs sorted by descending relevance.
    """
    if not query.strip():
        raise ValueError("Query must not be empty.")

    logger.info("Similarity search | query=%r | top_k=%d", query[:60], top_k)
    results: List[Tuple[Document, float]] = (
        vectorstore.similarity_search_with_score(query, k=top_k)
    )

    # FAISS returns L2 distance; lower = more similar. Convert to a 0-1 score.
    # score = 1 / (1 + distance)  →  higher score = more relevant
    scored = [(doc, float(1 / (1 + dist))) for doc, dist in results]

    if score_threshold > 0.0:
        scored = [(doc, sc) for doc, sc in scored if sc >= score_threshold]
        logger.info("%d chunk(s) above score threshold %.2f", len(scored), score_threshold)

    return scored


def get_retriever(
    vectorstore: FAISS,
    top_k: int = DEFAULT_TOP_K,
):
    """
    Return a LangChain-compatible retriever from a FAISS vector store.

    This is the interface used by :mod:`src.rag_chain`.

    Parameters
    ----------
    vectorstore : FAISS
    top_k : int
        Number of chunks to retrieve per query.

    Returns
    -------
    VectorStoreRetriever
    """
    return vectorstore.as_retriever(
        search_type   = "similarity",
        search_kwargs = {"k": top_k},
    )

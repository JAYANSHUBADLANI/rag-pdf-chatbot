"""FAISS index build / save / load, using local sentence-transformers embeddings."""

import logging
from pathlib import Path
from typing import List, Optional

# langchain_community.embeddings.HuggingFaceEmbeddings is deprecated;
# the maintained implementation lives in the langchain-huggingface package.
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTORSTORE_DIR = "vectorstore"
INDEX_NAME = "faiss_index"
DEFAULT_TOP_K = 4


def get_embeddings(model_name: str = EMBEDDING_MODEL, device: str = "cpu") -> HuggingFaceEmbeddings:
    logger.info("Loading embedding model: %s", model_name)
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )


def _persist(vs: FAISS, save_dir: str, index_name: str) -> None:
    save_path = Path(save_dir) / index_name
    save_path.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(save_path))
    logger.info("FAISS index saved → %s", save_path)


def build_vectorstore(
    chunks: List[Document],
    embedding_model: Optional[HuggingFaceEmbeddings] = None,
    save_dir: Optional[str] = None,
    index_name: str = INDEX_NAME,
) -> FAISS:
    """Build a FAISS index from chunks.

    Persistence is opt-in: pass save_dir (e.g. VECTORSTORE_DIR) to write the
    index to disk. The Streamlit app deliberately does NOT persist; a single
    shared directory would let concurrent sessions overwrite each other's
    indexes. Use persistence for CLI/batch workflows only.
    """
    if not chunks:
        raise ValueError("Cannot build a vector store from an empty chunk list.")

    if embedding_model is None:
        embedding_model = get_embeddings()

    logger.info("Building FAISS index from %d chunks …", len(chunks))
    vs = FAISS.from_documents(chunks, embedding_model)

    if save_dir:
        _persist(vs, save_dir, index_name)

    return vs


def add_to_vectorstore(
    vs: FAISS,
    chunks: List[Document],
    save_dir: Optional[str] = None,
    index_name: str = INDEX_NAME,
) -> FAISS:
    """Add chunks to an existing index. Persists only when save_dir is given."""
    if not chunks:
        return vs
    vs.add_documents(chunks)
    logger.info("Added %d chunks to vectorstore", len(chunks))
    if save_dir:
        _persist(vs, save_dir, index_name)
    return vs


def load_vectorstore(
    save_dir: str = VECTORSTORE_DIR,
    index_name: str = INDEX_NAME,
    embedding_model: Optional[HuggingFaceEmbeddings] = None,
) -> FAISS:
    """Load a previously persisted FAISS index.

    Security note: FAISS deserialization uses pickle, so
    allow_dangerous_deserialization=True must ONLY be used on index files this
    application wrote itself. Never point this at an index from an untrusted
    source; pickle can execute arbitrary code on load.
    """
    index_path = Path(save_dir) / index_name
    if not index_path.exists():
        raise FileNotFoundError(
            f"No FAISS index found at '{index_path}'. Run build_vectorstore() first."
        )

    if embedding_model is None:
        embedding_model = get_embeddings()

    logger.info("Loading FAISS index from %s …", index_path)
    vs = FAISS.load_local(
        str(index_path),
        embedding_model,
        allow_dangerous_deserialization=True,  # see security note in docstring
    )
    logger.info("FAISS index loaded.")
    return vs


def vectorstore_exists(save_dir: str = VECTORSTORE_DIR, index_name: str = INDEX_NAME) -> bool:
    return (Path(save_dir) / index_name).exists()

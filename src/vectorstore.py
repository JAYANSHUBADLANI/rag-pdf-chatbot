import logging
from pathlib import Path
from typing import List, Optional

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
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


def build_vectorstore(
    chunks: List[Document],
    embedding_model: Optional[HuggingFaceEmbeddings] = None,
    save_dir: str = VECTORSTORE_DIR,
    index_name: str = INDEX_NAME,
) -> FAISS:
    if not chunks:
        raise ValueError("Cannot build a vector store from an empty chunk list.")

    if embedding_model is None:
        embedding_model = get_embeddings()

    logger.info("Building FAISS index from %d chunks …", len(chunks))
    vs = FAISS.from_documents(chunks, embedding_model)

    # persist so we don't re-embed on reload
    save_path = Path(save_dir) / index_name
    save_path.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(save_path))
    logger.info("FAISS index saved → %s", save_path)

    return vs


def load_vectorstore(
    save_dir: str = VECTORSTORE_DIR,
    index_name: str = INDEX_NAME,
    embedding_model: Optional[HuggingFaceEmbeddings] = None,
) -> FAISS:
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
        allow_dangerous_deserialization=True,  # safe: index files we wrote ourselves
    )
    logger.info("FAISS index loaded.")
    return vs


def vectorstore_exists(save_dir: str = VECTORSTORE_DIR, index_name: str = INDEX_NAME) -> bool:
    return (Path(save_dir) / index_name).exists()

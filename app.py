"""
app.py
──────
Streamlit chat interface for the RAG PDF Chatbot.

Features:
  • PDF upload (sidebar) — processed and indexed on upload
  • Chat input with message history (session state)
  • Each assistant response shows answer + expandable source references
  • Reset button clears history and vectorstore

Author : Jayanshu Badlani
GitHub : https://github.com/JAYANSHUBADLANI
"""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from src.pdf_processor import process_pdf
from src.vectorstore   import build_vectorstore, get_embeddings, vectorstore_exists, load_vectorstore
from src.rag_chain     import build_rag_chain, query_rag

# ── Environment ───────────────────────────────────────────────────────────────
load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "RAG PDF Chatbot",
    page_icon  = "🤖",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Global background ── */
    .stApp { background-color: #0f1117; }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background-color: #1a1d2e;
        border-right: 1px solid #2e3250;
    }

    /* ── Chat messages ── */
    .stChatMessage { border-radius: 12px; margin-bottom: 8px; }

    /* ── Source chip ── */
    .source-chip {
        display: inline-block;
        background: #2e3250;
        color: #a0aec0;
        border-radius: 6px;
        padding: 2px 8px;
        font-size: 0.75rem;
        margin: 2px 2px 0 0;
    }

    /* ── Status pill ── */
    .status-ready   { color: #43e97b; font-weight: 600; }
    .status-pending { color: #f7971e; font-weight: 600; }

    /* ── Expander header ── */
    details summary { font-size: 0.85rem; color: #6e8efb !important; cursor: pointer; }

    /* ── Scrollable source block ── */
    .source-block {
        background: #1a1d2e;
        border: 1px solid #2e3250;
        border-radius: 8px;
        padding: 10px 14px;
        font-size: 0.82rem;
        color: #c9d1d9;
        max-height: 200px;
        overflow-y: auto;
        white-space: pre-wrap;
        margin-top: 4px;
    }
</style>
""", unsafe_allow_html=True)

# ── Session state defaults ────────────────────────────────────────────────────
if "messages"       not in st.session_state: st.session_state.messages       = []
if "vectorstore"    not in st.session_state: st.session_state.vectorstore    = None
if "rag_chain"      not in st.session_state: st.session_state.rag_chain      = None
if "embeddings"     not in st.session_state: st.session_state.embeddings     = None
if "uploaded_name"  not in st.session_state: st.session_state.uploaded_name  = None
if "processing"     not in st.session_state: st.session_state.processing     = False


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _get_embeddings():
    """Load the embedding model once and cache across sessions."""
    return get_embeddings()


def _process_and_index(pdf_bytes: bytes, filename: str) -> None:
    """Write PDF to a temp file, chunk it, embed it, build FAISS index."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    try:
        with st.spinner(f"📄 Extracting text from **{filename}** …"):
            chunks = process_pdf(tmp_path)

        with st.spinner(f"🧠 Building vector index ({len(chunks)} chunks) …"):
            emb = _get_embeddings()
            vs  = build_vectorstore(chunks, embedding_model=emb)

        st.session_state.vectorstore   = vs
        st.session_state.embeddings    = emb
        st.session_state.rag_chain     = build_rag_chain(vs)
        st.session_state.uploaded_name = filename
        st.session_state.messages      = []   # fresh chat for new document

        st.success(f"✅ **{filename}** indexed — {len(chunks)} chunks ready.")
    finally:
        os.unlink(tmp_path)


def _format_sources(source_chunks) -> str:
    """Return a compact comma-separated list of cited page labels."""
    seen, labels = set(), []
    for doc in source_chunks:
        lbl = doc.metadata.get("page_label", "")
        if lbl and lbl not in seen:
            labels.append(lbl)
            seen.add(lbl)
    return ", ".join(labels) if labels else "—"


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🤖 RAG PDF Chatbot")
    st.markdown(
        "Upload a PDF and ask questions about its contents.  "
        "Powered by **Claude claude-opus-4-6** + **FAISS**."
    )
    st.divider()

    # ── API key check ──
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key or api_key.startswith("your_"):
        st.warning("⚠️ **ANTHROPIC_API_KEY** not set.  "
                   "Add it to `.env` before querying.")
    else:
        st.markdown('<span class="status-ready">● API key loaded</span>',
                    unsafe_allow_html=True)

    st.divider()

    # ── PDF uploader ──
    st.markdown("### 📄 Upload PDF")
    uploaded = st.file_uploader(
        label       = "Choose a PDF file",
        type        = ["pdf"],
        label_visibility = "collapsed",
    )

    if uploaded and uploaded.name != st.session_state.uploaded_name:
        _process_and_index(uploaded.read(), uploaded.name)

    # ── Document status ──
    if st.session_state.uploaded_name:
        st.markdown(
            f'<span class="status-ready">● {st.session_state.uploaded_name}</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span class="status-pending">○ No document loaded</span>',
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Reset ──
    if st.button("🗑️ Clear chat & reset", use_container_width=True):
        st.session_state.messages      = []
        st.session_state.vectorstore   = None
        st.session_state.rag_chain     = None
        st.session_state.uploaded_name = None
        st.rerun()

    st.divider()
    st.markdown(
        "**Author:** [Jayanshu Badlani](https://github.com/JAYANSHUBADLANI)  \n"
        "[LinkedIn](https://linkedin.com/in/jayanshu-badlani)",
        unsafe_allow_html=False,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main — Chat UI
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("## 💬 Chat with your PDF")

# ── Replay existing messages ──
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Show source references for assistant messages
        if msg["role"] == "assistant" and msg.get("source_chunks"):
            with st.expander(f"📚 Sources — {msg.get('source_pages', '—')}"):
                for i, chunk in enumerate(msg["source_chunks"], 1):
                    meta = chunk.metadata
                    label = (
                        f"Chunk {i} | {meta.get('source','?')} | "
                        f"{meta.get('page_label','?')} | "
                        f"{meta.get('chunk_chars','?')} chars"
                    )
                    st.markdown(f"**{label}**")
                    st.markdown(
                        f'<div class="source-block">{chunk.page_content}</div>',
                        unsafe_allow_html=True,
                    )

# ── Chat input ──
placeholder = (
    "Ask a question about your PDF…"
    if st.session_state.rag_chain
    else "Upload a PDF in the sidebar first…"
)

if prompt := st.chat_input(placeholder, disabled=st.session_state.rag_chain is None):
    # ── Append and display user message ──
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # ── Generate assistant response ──
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            t0 = time.time()
            try:
                result = query_rag(st.session_state.rag_chain, prompt)
                elapsed = round(time.time() - t0, 1)

                answer        = result["answer"]
                source_chunks = result["source_chunks"]
                source_pages  = ", ".join(result["source_pages"]) or "—"

                st.markdown(answer)

                with st.expander(f"📚 Sources — {source_pages}  ·  _{elapsed}s_"):
                    for i, chunk in enumerate(source_chunks, 1):
                        meta  = chunk.metadata
                        label = (
                            f"Chunk {i} | {meta.get('source','?')} | "
                            f"{meta.get('page_label','?')} | "
                            f"{meta.get('chunk_chars','?')} chars"
                        )
                        st.markdown(f"**{label}**")
                        st.markdown(
                            f'<div class="source-block">{chunk.page_content}</div>',
                            unsafe_allow_html=True,
                        )

                # Persist in session
                st.session_state.messages.append({
                    "role":         "assistant",
                    "content":      answer,
                    "source_chunks": source_chunks,
                    "source_pages":  source_pages,
                })

            except Exception as exc:
                err_msg = f"⚠️ Error: {exc}"
                st.error(err_msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": err_msg}
                )

# ── Empty state nudge ──
if not st.session_state.messages and not st.session_state.uploaded_name:
    st.info("👆 Upload a PDF in the sidebar to get started.")
elif not st.session_state.messages and st.session_state.uploaded_name:
    st.info("✅ PDF indexed! Ask your first question above.")

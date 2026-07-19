import html
import logging
import os
import tempfile
import time
from typing import Any, Dict, List, Optional

import streamlit as st
from dotenv import load_dotenv

from src.pdf_processor import EmptyPdfError, process_pdf
from src.vectorstore import add_to_vectorstore, build_vectorstore, get_embeddings
from src.rag_chain import RagPipeline

load_dotenv()

# Logging is configured once, here, at the application entry point.
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

GENERIC_ERROR_MSG = (
    "Something went wrong while generating the answer. "
    "Please try again. Details are in the server logs."
)

SUGGESTED_QUESTIONS = [
    "Summarize this document",
    "What are the key takeaways?",
    "List important numbers & dates",
]

st.set_page_config(
    page_title="DocMind: PDF Intelligence",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }

:root {
    --bg: #070b14;
    --surface: rgba(255,255,255,0.03);
    --surface-2: rgba(255,255,255,0.05);
    --line: rgba(255,255,255,0.07);
    --line-soft: rgba(255,255,255,0.05);
    --text: #e7eaf2;
    --text-2: #98a0b3;
    --text-3: #6b7590;
    --accent: #2dd4a7;
    --accent-soft: rgba(45,212,167,0.1);
    --accent-line: rgba(45,212,167,0.22);
    --grad: linear-gradient(135deg, #2dd4bf 0%, #34d399 100%);
}

html, body, .stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    background-color: var(--bg) !important;
    color: var(--text);
}

/* aurora glow */
.stApp {
    background-image:
        radial-gradient(1100px 520px at 82% -12%, rgba(45,212,191,0.075), transparent 60%),
        radial-gradient(900px 480px at -12% 108%, rgba(124,106,247,0.055), transparent 60%) !important;
    background-attachment: fixed !important;
}

#MainMenu, footer, header { visibility: hidden; }

/* keep the sidebar re-open chevron reachable even with the header hidden,
   pinned to the viewport's top-left corner so it lines up where the pane was */
[data-testid="stExpandSidebarButton"] {
    visibility: visible !important;
    position: fixed !important;
    top: 12px !important;
    left: 12px !important;
    z-index: 999 !important;
    background: var(--surface-2) !important;
    border: 1px solid var(--line) !important;
    border-radius: 8px !important;
}
[data-testid="stExpandSidebarButton"] svg { color: var(--text-2) !important; }
[data-testid="stExpandSidebarButton"]:hover { border-color: var(--accent-line) !important; }
[data-testid="stExpandSidebarButton"]:hover svg { color: var(--accent) !important; }
.stDeployButton { display: none !important; }

@keyframes fadeUp {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: none; }
}

/* sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a101d 0%, #070b14 100%) !important;
    border-right: 1px solid var(--line-soft) !important;
    position: relative !important;
}
/* fixed width only while expanded; forcing it unconditionally leaves a
   300px ghost column when collapsed and the content sits right of center */
section[data-testid="stSidebar"][aria-expanded="true"] {
    width: 300px !important;
    min-width: 300px !important;
}
section[data-testid="stSidebar"] > div:first-child {
    padding: 28px 20px !important;
}

/* main content */
.main .block-container {
    max-width: 860px !important;
    padding: 0 32px 96px 32px !important;
    margin: 0 auto !important;
}

/* wordmark */
.wm-logo {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 28px;
}
.wm-diamond {
    width: 36px;
    height: 36px;
    background: var(--grad);
    border-radius: 11px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: 'Fraunces', serif;
    font-size: 19px;
    font-weight: 600;
    color: #04251c;
    flex-shrink: 0;
    box-shadow: 0 4px 18px rgba(45,212,167,0.25);
}
.wm-name {
    font-family: 'Fraunces', serif;
    font-size: 18px;
    font-weight: 600;
    color: var(--text);
    letter-spacing: -0.2px;
}
.wm-tagline {
    font-size: 10px;
    color: var(--text-3);
    font-weight: 500;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    margin-top: 1px;
}

.sb-label {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: var(--text-3);
    margin: 20px 0 8px 0;
}
.doc-card {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 12px;
    padding: 12px 14px;
    margin-top: 6px;
    transition: border-color 0.15s ease;
}
.doc-card:hover { border-color: var(--accent-line); }
.doc-card-name {
    font-size: 13px;
    font-weight: 500;
    color: #c9cfdd;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.doc-card-meta {
    font-size: 11px;
    color: var(--text-3);
    margin-top: 3px;
}
.doc-status-dot {
    display: inline-block;
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--accent);
    margin-right: 6px;
    vertical-align: middle;
    box-shadow: 0 0 8px rgba(45,212,167,0.55);
}
.doc-status-idle {
    display: inline-block;
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--text-3);
    margin-right: 6px;
    vertical-align: middle;
}
.doc-status-fail {
    display: inline-block;
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #ef4444;
    margin-right: 6px;
    vertical-align: middle;
    box-shadow: 0 0 8px rgba(239,68,68,0.4);
}
.api-warn {
    background: rgba(234,179,8,0.07);
    border: 1px solid rgba(234,179,8,0.18);
    border-radius: 10px;
    padding: 10px 13px;
    font-size: 12px;
    color: #d1a512;
    line-height: 1.5;
}
.api-ok {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    background: var(--accent-soft);
    border: 1px solid var(--accent-line);
    border-radius: 20px;
    padding: 6px 14px;
    font-size: 12px;
    color: var(--accent);
    font-weight: 500;
}
.api-ok::before {
    content: "";
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--accent);
    box-shadow: 0 0 8px rgba(45,212,167,0.7);
}
.sb-divider {
    height: 1px;
    background: var(--line-soft);
    margin: 20px 0;
}
/* pin the author credit to the sidebar's bottom-left corner. Streamlit wraps
   the markdown in a zero-height, position:relative stElementContainer, so we
   neutralise that wrapper and anchor against the full-height sidebar content. */
section[data-testid="stSidebar"] [data-testid="stSidebarContent"] { min-height: 100vh !important; }
section[data-testid="stSidebar"] [data-testid="stElementContainer"]:has(.sb-author) {
    position: static !important;
}
.sb-author {
    position: absolute;
    bottom: 22px;
    left: 20px;
    font-size: 11px;
    color: #333a4e;
    line-height: 1.7;
}
.sb-author a { color: var(--text-3); text-decoration: none; transition: color 0.15s ease; }
.sb-author a:hover { color: var(--accent); }

/* page header */
.page-header {
    padding: 60px 0 34px 0;
    border-bottom: 1px solid var(--line-soft);
    margin-bottom: 34px;
    animation: fadeUp 0.45s ease both;
}
.page-title {
    font-family: 'Fraunces', serif;
    font-size: 36px;
    font-weight: 550;
    color: var(--text);
    letter-spacing: -0.5px;
    line-height: 1.15;
    margin: 0 0 8px 0;
}
.page-title .grad {
    background: var(--grad);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    color: transparent;
}
.page-subtitle {
    font-size: 14px;
    color: var(--text-2);
    font-weight: 400;
}
.page-doc-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: var(--accent-soft);
    border: 1px solid var(--accent-line);
    border-radius: 20px;
    padding: 4px 12px 4px 9px;
    font-size: 12px;
    color: var(--accent);
    font-weight: 500;
    margin-top: 14px;
}

/* empty state */
.empty-state {
    text-align: center;
    padding: 64px 40px 28px 40px;
    color: var(--text-3);
    animation: fadeUp 0.5s ease both;
}
.empty-state-icon {
    width: 52px;
    height: 52px;
    margin: 0 auto 18px auto;
    border-radius: 15px;
    background: var(--accent-soft);
    border: 1px solid var(--accent-line);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 22px;
    color: var(--accent);
}
.empty-state-title {
    font-family: 'Fraunces', serif;
    font-size: 19px;
    font-weight: 550;
    color: #aab1c4;
    margin-bottom: 8px;
}
.empty-state-sub { font-size: 13px; color: var(--text-3); line-height: 1.7; }
.suggest-label {
    text-align: center;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: var(--text-3);
    margin: 6px 0 10px 0;
}

/* chat */
.stChatMessage {
    background: transparent !important;
    border: none !important;
    padding: 6px 0 !important;
    animation: fadeUp 0.35s ease both;
}
/* user turns get a soft bubble; assistant turns stay open on the canvas */
.stChatMessage:has([data-testid="chatAvatarIcon-user"]) {
    background: var(--surface) !important;
    border: 1px solid var(--line) !important;
    border-radius: 16px !important;
    padding: 12px 16px !important;
}
[data-testid="chatAvatarIcon-user"] {
    background: var(--grad) !important;
    color: #04251c !important;
}
[data-testid="chatAvatarIcon-assistant"] {
    background: var(--accent-soft) !important;
    color: var(--accent) !important;
    border: 1px solid var(--accent-line) !important;
}
[data-testid="stChatMessageContent"] { font-size: 14.5px; line-height: 1.75; }

.cite-row {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-top: 14px;
    padding-top: 14px;
    border-top: 1px solid var(--line-soft);
}
.cite-chip {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: var(--accent-soft);
    border: 1px solid var(--accent-line);
    border-radius: 6px;
    padding: 3px 9px;
    font-size: 11px;
    color: var(--accent);
    font-weight: 500;
    letter-spacing: 0.1px;
    cursor: default;
    transition: background 0.15s ease;
}
.cite-chip:hover { background: rgba(45,212,167,0.16); }
.cite-chip::before { content: "¶"; opacity: 0.7; font-size: 10px; }
.source-drawer {
    background: rgba(255,255,255,0.02);
    border: 1px solid var(--line-soft);
    border-left: 2px solid var(--accent-line);
    border-radius: 8px;
    padding: 14px 16px;
    margin-top: 8px;
    font-size: 12px;
    color: #7d8598;
    line-height: 1.7;
    font-family: 'JetBrains Mono', monospace;
    max-height: 190px;
    overflow-y: auto;
    white-space: pre-wrap;
}
.source-drawer-label {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: var(--text-3);
    margin-bottom: 6px;
}
.answer-error {
    font-size: 13px;
    color: #ef4444;
    background: rgba(239,68,68,0.06);
    border: 1px solid rgba(239,68,68,0.18);
    border-radius: 10px;
    padding: 10px 14px;
}

/* upload widget */
[data-testid="stFileUploader"] {
    background: var(--surface) !important;
    border: 1px dashed rgba(255,255,255,0.14) !important;
    border-radius: 12px !important;
    padding: 14px !important;
    transition: border-color 0.2s ease !important;
}
[data-testid="stFileUploader"]:hover { border-color: var(--accent-line) !important; }
[data-testid="stFileUploader"] section { border: none !important; padding: 0 !important; background: transparent !important; }
[data-testid="stFileUploader"] label { display: none !important; }

/* buttons (sidebar actions + suggestion chips) */
.stButton > button {
    background: var(--surface) !important;
    border: 1px solid rgba(255,255,255,0.13) !important;
    color: var(--text-2) !important;
    font-size: 12.5px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    border-radius: 20px !important;
    padding: 7px 15px !important;
    transition: all 0.18s ease !important;
    letter-spacing: 0.1px !important;
}
.stButton > button:hover {
    border-color: var(--accent-line) !important;
    color: var(--accent) !important;
    background: var(--accent-soft) !important;
    transform: translateY(-1px) !important;
}

/* chat input */
[data-testid="stBottom"],
[data-testid="stBottom"] > div {
    background: var(--bg) !important;
}
[data-testid="stChatInput"] {
    background: rgba(10,16,29,0.88) !important;
    backdrop-filter: blur(12px) !important;
    border-top: 1px solid var(--line-soft) !important;
    padding: 16px 32px !important;
}
/* one dark pill: border lives on the wrapper so the send button sits inside it */
[data-testid="stChatInput"] > div {
    background: var(--surface-2) !important;
    border: 1px solid rgba(255,255,255,0.11) !important;
    border-radius: 14px !important;
    align-items: center !important;
    transition: border-color 0.15s ease, box-shadow 0.15s ease !important;
}
[data-testid="stChatInput"] > div:focus-within {
    border-color: rgba(45,212,167,0.5) !important;
    box-shadow: 0 0 0 3px rgba(45,212,167,0.1) !important;
}
[data-testid="stChatInput"] textarea {
    background: transparent !important;
    border: none !important;
    color: var(--text) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 14px !important;
}
[data-testid="stChatInput"] textarea:focus {
    outline: none !important;
    border: none !important;
    box-shadow: none !important;
}
[data-testid="stChatInputSubmitButton"] {
    background: var(--grad) !important;
    border: none !important;
    border-radius: 50% !important;
    width: 32px !important;
    height: 32px !important;
    margin-right: 7px !important;
    flex-shrink: 0 !important;
}
[data-testid="stChatInputSubmitButton"]:hover { filter: brightness(1.12) !important; }
/* replace streamlit's up-arrow icon with a right arrow */
[data-testid="stChatInputSubmitButton"] svg { display: none !important; }
[data-testid="stChatInputSubmitButton"]::after {
    content: "→";
    color: #04251c;
    font-size: 17px;
    font-weight: 600;
    line-height: 1;
}

.stSpinner > div { border-top-color: var(--accent) !important; }

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #2a3145; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #3d465f; }

.stSuccess, .stInfo, .stWarning, .stError {
    font-size: 13px !important;
    border-radius: 10px !important;
}

details {
    border: 1px solid var(--line-soft) !important;
    border-radius: 10px !important;
    background: rgba(255,255,255,0.02) !important;
    margin-top: 10px !important;
}
details summary {
    font-size: 12px !important;
    color: var(--text-3) !important;
    padding: 9px 13px !important;
    cursor: pointer !important;
    letter-spacing: 0.1px !important;
    transition: color 0.15s ease !important;
}
details summary:hover { color: var(--accent) !important; }
details > div { padding: 0 13px 13px 13px !important; }
</style>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "rag_pipeline" not in st.session_state:
    st.session_state.rag_pipeline = None
if "indexed_pdfs" not in st.session_state:
    st.session_state.indexed_pdfs = []
if "failed_pdfs" not in st.session_state:
    st.session_state.failed_pdfs = {}  # filename -> short reason

# A suggestion chip click from the previous run, consumed exactly once.
pending_prompt: Optional[str] = st.session_state.pop("pending_prompt", None)


def _api_key_configured() -> bool:
    api_key = os.getenv("GROQ_API_KEY", "")
    return bool(api_key) and not api_key.startswith("your_")


@st.cache_resource(show_spinner=False)
def _get_embeddings():
    return get_embeddings()


def _process_and_index(pdf_bytes: bytes, filename: str) -> None:
    """Chunk and index one uploaded PDF. Raises on failure (caller handles UX)."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name
    try:
        with st.spinner(f"Reading {filename}…"):
            chunks = process_pdf(tmp_path)
            for chunk in chunks:
                # replace the temp-file name with the real upload name
                chunk.metadata["source"] = filename
        with st.spinner(f"Indexing {len(chunks)} passages…"):
            emb = _get_embeddings()
            if st.session_state.vectorstore is None:
                # No save_dir: keep the index in memory only. Persisting to a
                # shared directory would let concurrent sessions clobber
                # each other's indexes.
                vs = build_vectorstore(chunks, embedding_model=emb)
            else:
                vs = add_to_vectorstore(st.session_state.vectorstore, chunks)

        st.session_state.vectorstore = vs
        st.session_state.rag_pipeline = None  # rebuild retriever on next use
        st.session_state.indexed_pdfs.append({"name": filename, "chunks": len(chunks)})
    finally:
        os.unlink(tmp_path)


def _prewarm_pipeline() -> None:
    """Build the RAG pipeline right after indexing so the first question is fast.

    Best-effort: on failure we fall back to lazy init at question time.
    """
    if st.session_state.vectorstore is None or st.session_state.rag_pipeline is not None:
        return
    if not _api_key_configured():
        return
    try:
        with st.spinner("Preparing answer pipeline…"):
            st.session_state.rag_pipeline = RagPipeline(st.session_state.vectorstore)
    except Exception:
        logger.exception("Pipeline pre-warm failed; will retry on first question.")


def _render_sources(
    source_chunks: List[Any],
    source_pages: List[str],
    elapsed: Optional[float] = None,
) -> None:
    """Render citation chips + source-passage drawer (used for live and history)."""
    if source_pages:
        chips_html = "".join(
            f'<span class="cite-chip">{html.escape(str(p))}</span>'
            for p in source_pages
        )
        st.markdown(f'<div class="cite-row">{chips_html}</div>', unsafe_allow_html=True)

    if not source_chunks:
        return

    title = f"View {len(source_chunks)} source passage(s)"
    if elapsed is not None:
        title += f" · {elapsed}s"
    with st.expander(title):
        for i, chunk in enumerate(source_chunks, 1):
            meta = chunk.metadata
            label = html.escape(
                f"{meta.get('source', '-')} · {meta.get('page_label', '-')} · "
                f"{meta.get('chunk_chars', '?')} chars"
            )
            # PDF text is untrusted input; escape before injecting into HTML.
            content = html.escape(chunk.page_content)
            st.markdown(
                f'<div class="source-drawer-label">Passage {i}: {label}</div>'
                f'<div class="source-drawer">{content}</div>',
                unsafe_allow_html=True,
            )
            if i < len(source_chunks):
                st.markdown('<div style="height:10px;"></div>', unsafe_allow_html=True)


def _chat_history_for_pipeline() -> List[Dict[str, str]]:
    """History for the RAG pipeline: everything before the current turn,
    excluding error turns so they don't pollute rewriting/answering."""
    return [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[:-1]
        if not m.get("error")
    ]


# sidebar
with st.sidebar:
    st.markdown("""
    <div class="wm-logo">
        <div class="wm-diamond">D</div>
        <div>
            <div class="wm-name">DocMind</div>
            <div class="wm-tagline">PDF Intelligence</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not _api_key_configured():
        st.markdown(
            '<div class="api-warn">API key not configured.<br>'
            'Add <code>GROQ_API_KEY</code> to <code>.env</code></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<div class="api-ok">Groq connected</div>', unsafe_allow_html=True)

    st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-label">Documents</div>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        label="Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    if uploaded_files:
        indexed_names = {p["name"] for p in st.session_state.indexed_pdfs}
        skip = indexed_names | set(st.session_state.failed_pdfs)
        new_files = [f for f in uploaded_files if f.name not in skip]
        processed_any = False
        for file in new_files:
            try:
                _process_and_index(file.read(), file.name)
                processed_any = True
            except EmptyPdfError:
                # Track failures so we don't re-attempt them on every rerun.
                st.session_state.failed_pdfs[file.name] = "No extractable text (scanned PDF?)"
                logger.warning("Skipped '%s': no extractable text", file.name)
            except Exception:
                st.session_state.failed_pdfs[file.name] = "Indexing failed, see server logs"
                logger.exception("Failed to index '%s'", file.name)
        if processed_any:
            _prewarm_pipeline()
        if new_files:
            st.rerun()

    if st.session_state.indexed_pdfs:
        cards_html = "".join(
            f"""
            <div class="doc-card">
                <div class="doc-card-name">
                    <span class="doc-status-dot"></span>{html.escape(pdf["name"])}
                </div>
                <div class="doc-card-meta">{pdf["chunks"]} passages</div>
            </div>
            """
            for pdf in st.session_state.indexed_pdfs
        )
        total = sum(p["chunks"] for p in st.session_state.indexed_pdfs)
        st.markdown(cards_html, unsafe_allow_html=True)
        st.markdown(
            f'<div class="doc-card-meta" style="margin-top:8px;text-align:center;">'
            f'{len(st.session_state.indexed_pdfs)} '
            f'{"document" if len(st.session_state.indexed_pdfs) == 1 else "documents"}'
            f' · {total} total passages</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown("""
        <div class="doc-card">
            <div class="doc-card-name" style="color:#333a4e;">
                <span class="doc-status-idle"></span>No documents loaded
            </div>
            <div class="doc-card-meta">Upload PDFs above to begin</div>
        </div>
        """, unsafe_allow_html=True)

    if st.session_state.failed_pdfs:
        failed_html = "".join(
            f"""
            <div class="doc-card" style="border-color:rgba(239,68,68,0.22);">
                <div class="doc-card-name" style="color:#b0596a;">
                    <span class="doc-status-fail"></span>{html.escape(name)}
                </div>
                <div class="doc-card-meta">{html.escape(reason)}</div>
            </div>
            """
            for name, reason in st.session_state.failed_pdfs.items()
        )
        st.markdown(failed_html, unsafe_allow_html=True)

    st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with col_b:
        if st.button("Reset all", use_container_width=True):
            st.session_state.messages = []
            st.session_state.vectorstore = None
            st.session_state.rag_pipeline = None
            st.session_state.indexed_pdfs = []
            st.session_state.failed_pdfs = {}
            st.rerun()

    st.markdown("""
    <div class="sb-author">
        <a href="https://github.com/JAYANSHUBADLANI" target="_blank">Jayanshu Badlani</a>
        &nbsp;·&nbsp;
        <a href="https://www.linkedin.com/in/jayanshu-badlani-b77478185/" target="_blank">LinkedIn</a>
    </div>
    """, unsafe_allow_html=True)


indexed = st.session_state.indexed_pdfs
ready_text = ""
if indexed:
    if len(indexed) == 1:
        badge_text = html.escape(indexed[0]["name"])
        ready_text = f"{html.escape(indexed[0]['name'])} has been indexed."
    else:
        badge_text = f"{len(indexed)} documents"
        ready_text = f"{len(indexed)} documents indexed."

    st.markdown(f"""
    <div class="page-header">
        <div class="page-title">Ask your <span class="grad">documents</span> anything.</div>
        <div class="page-subtitle">Every answer is grounded in your PDFs, with page citations.</div>
        <div class="page-doc-badge">
            <svg width="10" height="10" viewBox="0 0 12 12" fill="none">
                <rect x="1" y="1" width="10" height="10" rx="2" stroke="#2dd4a7" stroke-width="1.5"/>
                <line x1="3" y1="4" x2="9" y2="4" stroke="#2dd4a7" stroke-width="1.2"/>
                <line x1="3" y1="6.5" x2="7" y2="6.5" stroke="#2dd4a7" stroke-width="1.2"/>
            </svg>
            {badge_text}
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="page-header">
        <div class="page-title">Ask your <span class="grad">documents</span> anything.</div>
        <div class="page-subtitle">Upload PDFs in the sidebar to get started.</div>
    </div>
    """, unsafe_allow_html=True)


if not st.session_state.messages and not indexed:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-state-icon">◆</div>
        <div class="empty-state-title">No documents loaded</div>
        <div class="empty-state-sub">
            Upload PDFs using the sidebar panel.<br>
            DocMind will index them and let you ask questions in plain language.
        </div>
    </div>
    """, unsafe_allow_html=True)
elif not st.session_state.messages and indexed and not pending_prompt:
    st.markdown(f"""
    <div class="empty-state">
        <div class="empty-state-icon">◆</div>
        <div class="empty-state-title">Ready when you are</div>
        <div class="empty-state-sub">
            {ready_text}<br>
            Type a question below, or start with one of these:
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="suggest-label">Try asking</div>', unsafe_allow_html=True)
    cols = st.columns(len(SUGGESTED_QUESTIONS))
    for col, question in zip(cols, SUGGESTED_QUESTIONS):
        with col:
            if st.button(question, key=f"suggest_{question}", use_container_width=True):
                st.session_state.pending_prompt = question
                st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("source_chunks"):
            _render_sources(msg["source_chunks"], msg.get("source_pages", []))


# chat input
typed = st.chat_input(
    placeholder="Ask a question about your document…" if st.session_state.vectorstore else "Upload a document to begin…",
    disabled=st.session_state.vectorstore is None,
)
prompt = typed or pending_prompt

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        t0 = time.time()
        answer = ""
        source_chunks: List[Any] = []
        source_pages: List[str] = []
        answer_placeholder = None
        try:
            if st.session_state.rag_pipeline is None:
                with st.spinner("Connecting…"):
                    st.session_state.rag_pipeline = RagPipeline(st.session_state.vectorstore)

            answer_placeholder = st.empty()
            chat_history = _chat_history_for_pipeline()

            with st.spinner("Searching document…"):
                stream = st.session_state.rag_pipeline.stream(prompt, chat_history=chat_history)
                first_event = next(stream, None)

            if first_event is None:
                raise RuntimeError("Pipeline stream produced no events.")
            if first_event["type"] == "sources":
                source_chunks = first_event["source_chunks"]
                source_pages = first_event["source_pages"]

            for event in stream:
                if event["type"] == "token":
                    answer += event["text"]
                    answer_placeholder.markdown(answer + "▌")

            answer_placeholder.markdown(answer)
            elapsed = round(time.time() - t0, 1)

            _render_sources(source_chunks, source_pages, elapsed=elapsed)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "source_chunks": source_chunks,
                "source_pages": source_pages,
            })

        except Exception:
            # Log the full traceback server-side; show the user a generic
            # message so internals (paths, key fragments, stack) never leak.
            logger.exception("Answer generation failed")

            if answer and answer_placeholder is not None:
                # Keep the partial streamed answer instead of discarding it.
                partial = answer + "\n\n*(answer interrupted by an error)*"
                answer_placeholder.markdown(partial)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": partial,
                    "source_chunks": source_chunks,
                    "source_pages": source_pages,
                })
            else:
                st.markdown(
                    f'<div class="answer-error">{GENERIC_ERROR_MSG}</div>',
                    unsafe_allow_html=True,
                )
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": GENERIC_ERROR_MSG,
                    "error": True,
                })

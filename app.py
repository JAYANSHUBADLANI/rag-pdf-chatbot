import os
import tempfile
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from src.pdf_processor import process_pdf
from src.vectorstore import build_vectorstore, get_embeddings
from src.rag_chain import build_rag_chain, query_rag

load_dotenv()

st.set_page_config(
    page_title="DocMind — PDF Intelligence",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, .stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    background-color: #0b0d11 !important;
    color: #d4d8e1;
}

#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none !important; }

/* sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1117 0%, #0b0d11 100%) !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
    width: 280px !important;
    min-width: 280px !important;
}
section[data-testid="stSidebar"] > div:first-child {
    padding: 28px 20px !important;
}

/* main content */
.main .block-container {
    max-width: 820px !important;
    padding: 0 32px 80px 32px !important;
    margin: 0 auto !important;
}

/* wordmark */
.wm-logo {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 32px;
}
.wm-diamond {
    width: 32px;
    height: 32px;
    background: linear-gradient(135deg, #7c6af7, #a78bfa);
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 14px;
    font-weight: 700;
    color: white;
    letter-spacing: -0.5px;
    flex-shrink: 0;
}
.wm-name {
    font-size: 15px;
    font-weight: 600;
    color: #e2e5ec;
    letter-spacing: -0.2px;
}
.wm-tagline {
    font-size: 11px;
    color: #4b5265;
    font-weight: 400;
    letter-spacing: 0.3px;
    text-transform: uppercase;
}

.sb-label {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    color: #3d4357;
    margin: 20px 0 8px 0;
}
.doc-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px;
    padding: 12px 14px;
    margin-top: 4px;
}
.doc-card-name {
    font-size: 13px;
    font-weight: 500;
    color: #c8cdd8;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.doc-card-meta {
    font-size: 11px;
    color: #454d62;
    margin-top: 3px;
}
.doc-status-dot {
    display: inline-block;
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #22c55e;
    margin-right: 5px;
    vertical-align: middle;
    box-shadow: 0 0 6px rgba(34,197,94,0.4);
}
.doc-status-idle {
    display: inline-block;
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #3d4357;
    margin-right: 5px;
    vertical-align: middle;
}
.api-warn {
    background: rgba(234,179,8,0.07);
    border: 1px solid rgba(234,179,8,0.18);
    border-radius: 8px;
    padding: 10px 13px;
    font-size: 12px;
    color: #ca8a04;
    line-height: 1.5;
}
.api-ok {
    background: rgba(34,197,94,0.06);
    border: 1px solid rgba(34,197,94,0.14);
    border-radius: 8px;
    padding: 10px 13px;
    font-size: 12px;
    color: #16a34a;
    line-height: 1.5;
}
.sb-divider {
    height: 1px;
    background: rgba(255,255,255,0.05);
    margin: 20px 0;
}
.sb-author {
    font-size: 11px;
    color: #2e3347;
    line-height: 1.7;
}
.sb-author a { color: #4b5265; text-decoration: none; }
.sb-author a:hover { color: #7c6af7; }

/* page header */
.page-header {
    padding: 48px 0 32px 0;
    border-bottom: 1px solid rgba(255,255,255,0.05);
    margin-bottom: 32px;
}
.page-title {
    font-size: 22px;
    font-weight: 600;
    color: #e2e5ec;
    letter-spacing: -0.4px;
    margin: 0 0 4px 0;
}
.page-subtitle {
    font-size: 13px;
    color: #454d62;
    font-weight: 400;
}
.page-doc-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(124,106,247,0.1);
    border: 1px solid rgba(124,106,247,0.2);
    border-radius: 20px;
    padding: 3px 10px 3px 8px;
    font-size: 12px;
    color: #a78bfa;
    font-weight: 500;
    margin-top: 10px;
}

/* empty state */
.empty-state {
    text-align: center;
    padding: 80px 40px;
    color: #3d4357;
}
.empty-state-icon { font-size: 28px; margin-bottom: 16px; opacity: 0.5; }
.empty-state-title {
    font-size: 17px;
    font-weight: 500;
    color: #4b5265;
    margin-bottom: 8px;
}
.empty-state-sub { font-size: 13px; color: #2e3347; line-height: 1.6; }

/* chat */
.stChatMessage { background: transparent !important; border: none !important; padding: 0 !important; }
[data-testid="stChatMessageContent"] { font-size: 14px; line-height: 1.7; }

.cite-row {
    display: flex;
    flex-wrap: wrap;
    gap: 5px;
    margin-top: 12px;
    padding-top: 12px;
    border-top: 1px solid rgba(255,255,255,0.05);
}
.cite-chip {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    background: rgba(124,106,247,0.08);
    border: 1px solid rgba(124,106,247,0.16);
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 11px;
    color: #7c6af7;
    font-weight: 500;
    letter-spacing: 0.1px;
    cursor: default;
}
.source-drawer {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 8px;
    padding: 14px 16px;
    margin-top: 8px;
    font-size: 12.5px;
    color: #6b7280;
    line-height: 1.65;
    font-family: 'Inter', monospace;
    max-height: 180px;
    overflow-y: auto;
    white-space: pre-wrap;
}
.source-drawer-label {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    color: #3d4357;
    margin-bottom: 6px;
}

/* upload widget */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 10px !important;
    padding: 14px !important;
}
[data-testid="stFileUploader"] section { border: none !important; padding: 0 !important; }
[data-testid="stFileUploader"] label { display: none !important; }

/* buttons */
.stButton > button {
    background: transparent !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    color: #6b7280 !important;
    font-size: 12px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    border-radius: 7px !important;
    padding: 6px 14px !important;
    transition: all 0.15s ease !important;
    letter-spacing: 0.1px !important;
}
.stButton > button:hover {
    border-color: rgba(124,106,247,0.35) !important;
    color: #a78bfa !important;
    background: rgba(124,106,247,0.06) !important;
}

/* chat input */
[data-testid="stChatInput"] {
    background: #0f1117 !important;
    border-top: 1px solid rgba(255,255,255,0.06) !important;
    padding: 16px 32px !important;
}
[data-testid="stChatInput"] textarea {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    color: #d4d8e1 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 14px !important;
    transition: border-color 0.15s ease !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: rgba(124,106,247,0.45) !important;
    box-shadow: 0 0 0 3px rgba(124,106,247,0.08) !important;
    outline: none !important;
}

.stSpinner > div { border-top-color: #7c6af7 !important; }

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #2e3347; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #454d62; }

.stSuccess, .stInfo, .stWarning, .stError {
    font-size: 13px !important;
    border-radius: 8px !important;
}

details {
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 8px !important;
    background: rgba(255,255,255,0.02) !important;
    margin-top: 8px !important;
}
details summary {
    font-size: 12px !important;
    color: #454d62 !important;
    padding: 8px 12px !important;
    cursor: pointer !important;
    letter-spacing: 0.1px !important;
}
details summary:hover { color: #7c6af7 !important; }
details > div { padding: 0 12px 12px 12px !important; }
</style>
""", unsafe_allow_html=True)

# session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "uploaded_name" not in st.session_state:
    st.session_state.uploaded_name = None
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0


@st.cache_resource(show_spinner=False)
def _get_embeddings():
    return get_embeddings()


def _process_and_index(pdf_bytes: bytes, filename: str) -> None:
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name
    try:
        with st.spinner(f"Reading {filename}…"):
            chunks = process_pdf(tmp_path)
        with st.spinner(f"Indexing {len(chunks)} passages…"):
            emb = _get_embeddings()
            vs = build_vectorstore(chunks, embedding_model=emb)

        st.session_state.vectorstore = vs
        st.session_state.rag_chain = None
        st.session_state.uploaded_name = filename
        st.session_state.chunk_count = len(chunks)
        st.session_state.messages = []
    finally:
        os.unlink(tmp_path)


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

    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key or api_key.startswith("your_"):
        st.markdown(
            '<div class="api-warn">API key not configured.<br>'
            'Add <code>ANTHROPIC_API_KEY</code> to <code>.env</code></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<div class="api-ok">Claude connected</div>', unsafe_allow_html=True)

    st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-label">Document</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        label="Upload PDF",
        type=["pdf"],
        label_visibility="collapsed",
    )
    if uploaded and uploaded.name != st.session_state.uploaded_name:
        _process_and_index(uploaded.read(), uploaded.name)
        st.rerun()

    if st.session_state.uploaded_name:
        st.markdown(f"""
        <div class="doc-card">
            <div class="doc-card-name">
                <span class="doc-status-dot"></span>{st.session_state.uploaded_name}
            </div>
            <div class="doc-card-meta">{st.session_state.chunk_count} passages indexed</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="doc-card">
            <div class="doc-card-name" style="color:#2e3347;">
                <span class="doc-status-idle"></span>No document loaded
            </div>
            <div class="doc-card-meta">Upload a PDF above to begin</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)

    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.vectorstore = None
        st.session_state.rag_chain = None
        st.session_state.uploaded_name = None
        st.session_state.chunk_count = 0
        st.rerun()

    st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sb-author">
        <a href="https://github.com/JAYANSHUBADLANI" target="_blank">Jayanshu Badlani</a>
        &nbsp;·&nbsp;
        <a href="https://linkedin.com/in/jayanshu-badlani" target="_blank">LinkedIn</a>
    </div>
    """, unsafe_allow_html=True)


# main header
if st.session_state.uploaded_name:
    st.markdown(f"""
    <div class="page-header">
        <div class="page-title">Document Q&amp;A</div>
        <div class="page-subtitle">Ask anything — answers are grounded in your document</div>
        <div class="page-doc-badge">
            <svg width="10" height="10" viewBox="0 0 12 12" fill="none">
                <rect x="1" y="1" width="10" height="10" rx="2" stroke="#a78bfa" stroke-width="1.5"/>
                <line x1="3" y1="4" x2="9" y2="4" stroke="#a78bfa" stroke-width="1.2"/>
                <line x1="3" y1="6.5" x2="7" y2="6.5" stroke="#a78bfa" stroke-width="1.2"/>
            </svg>
            {st.session_state.uploaded_name}
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="page-header">
        <div class="page-title">Document Q&amp;A</div>
        <div class="page-subtitle">Upload a PDF in the sidebar to get started</div>
    </div>
    """, unsafe_allow_html=True)


# chat history
if not st.session_state.messages and not st.session_state.uploaded_name:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-state-icon">◆</div>
        <div class="empty-state-title">No document loaded</div>
        <div class="empty-state-sub">
            Upload a PDF using the sidebar panel.<br>
            DocMind will index it and let you ask questions in plain language.
        </div>
    </div>
    """, unsafe_allow_html=True)
elif not st.session_state.messages and st.session_state.uploaded_name:
    st.markdown(f"""
    <div class="empty-state">
        <div class="empty-state-icon">◆</div>
        <div class="empty-state-title">Ready</div>
        <div class="empty-state-sub">
            {st.session_state.uploaded_name} has been indexed.<br>
            Type your first question below.
        </div>
    </div>
    """, unsafe_allow_html=True)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg["role"] == "assistant" and msg.get("source_chunks"):
            pages = msg.get("source_pages", [])
            if pages:
                chips_html = "".join(
                    f'<span class="cite-chip">p. {p.replace("p.", "").strip()}</span>'
                    for p in pages
                )
                st.markdown(f'<div class="cite-row">{chips_html}</div>', unsafe_allow_html=True)

            with st.expander(f"View {len(msg['source_chunks'])} source passage(s)"):
                for i, chunk in enumerate(msg["source_chunks"], 1):
                    meta = chunk.metadata
                    label = f"{meta.get('source','—')} · {meta.get('page_label','—')} · {meta.get('chunk_chars','?')} chars"
                    st.markdown(
                        f'<div class="source-drawer-label">Passage {i} — {label}</div>'
                        f'<div class="source-drawer">{chunk.page_content}</div>',
                        unsafe_allow_html=True,
                    )
                    if i < len(msg["source_chunks"]):
                        st.markdown('<div style="height:10px;"></div>', unsafe_allow_html=True)


# chat input
prompt = st.chat_input(
    placeholder="Ask a question about your document…" if st.session_state.vectorstore else "Upload a document to begin…",
    disabled=st.session_state.vectorstore is None,
)

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner(""):
            t0 = time.time()
            try:
                if st.session_state.rag_chain is None:
                    st.session_state.rag_chain = build_rag_chain(st.session_state.vectorstore)

                result = query_rag(st.session_state.rag_chain, prompt)
                elapsed = round(time.time() - t0, 1)

                answer = result["answer"]
                source_chunks = result["source_chunks"]
                source_pages = result["source_pages"]

                st.markdown(answer)

                if source_pages:
                    chips_html = "".join(
                        f'<span class="cite-chip">p. {p.replace("p.", "").strip()}</span>'
                        for p in source_pages
                    )
                    st.markdown(f'<div class="cite-row">{chips_html}</div>', unsafe_allow_html=True)

                with st.expander(f"View {len(source_chunks)} source passage(s) · {elapsed}s"):
                    for i, chunk in enumerate(source_chunks, 1):
                        meta = chunk.metadata
                        label = f"{meta.get('source','—')} · {meta.get('page_label','—')} · {meta.get('chunk_chars','?')} chars"
                        st.markdown(
                            f'<div class="source-drawer-label">Passage {i} — {label}</div>'
                            f'<div class="source-drawer">{chunk.page_content}</div>',
                            unsafe_allow_html=True,
                        )
                        if i < len(source_chunks):
                            st.markdown('<div style="height:10px;"></div>', unsafe_allow_html=True)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "source_chunks": source_chunks,
                    "source_pages": source_pages,
                })

            except Exception as exc:
                err = f"Error: {exc}"
                st.markdown(
                    f'<div style="font-size:13px;color:#ef4444;">{err}</div>',
                    unsafe_allow_html=True,
                )
                st.session_state.messages.append({"role": "assistant", "content": err})

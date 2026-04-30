# DocMind — PDF Intelligence

**Author:** Jayanshu Badlani
**GitHub:** [JAYANSHUBADLANI](https://github.com/JAYANSHUBADLANI)
**LinkedIn:** [jayanshu-badlani](https://linkedin.com/in/jayanshu-badlani)

---

## Overview

A production-ready **Retrieval-Augmented Generation (RAG)** application that lets you upload any PDF and interrogate it in plain English. Powered by **Claude Opus** (Anthropic), **FAISS** vector similarity search, local **sentence-transformer** embeddings, and a clean dark-mode **Streamlit** interface.

---

## 📸 Screenshot

![RAG PDF Chatbot — PDF indexed and ready](assets/screenshot.png)

---

## Demo

> Upload `assets/sample.pdf`, ask questions like *"What is FAISS?"* or *"How does chunking affect RAG quality?"* — answers cite page numbers and show retrieved source passages.

---

## ✨ Features

| Feature | Detail |
|---------|--------|
| 📄 PDF Upload | Drag-and-drop or browse, any PDF |
| ✂️ Smart Chunking | RecursiveCharacterTextSplitter (1000 chars, 200 overlap) |
| 🧠 Semantic Search | FAISS + sentence-transformers embeddings |
| 💬 Claude Opus LLM | claude-opus-4-6 — grounded, factual answers |
| 📚 Source Transparency | Every answer shows exact retrieved chunks + page numbers |
| 🗂️ Chat History | Full conversation persisted in session state |
| 💾 Persistent Index | FAISS saved to disk — no re-indexing on reload |
| 🌑 Dark Mode UI | Clean, modern dark theme |

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | Claude claude-opus-4-6 (Anthropic) |
| Orchestration | LangChain + LangChain-Community |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Store | FAISS (CPU) |
| PDF Parsing | PyPDF / PyPDFLoader |
| Text Splitting | RecursiveCharacterTextSplitter |
| UI | Streamlit |
| Config | python-dotenv |

---

## 📁 Repository Structure

```
rag-pdf-chatbot/
│
├── src/
│   ├── __init__.py
│   ├── pdf_processor.py       # PDF loading, cleaning & chunking
│   ├── vectorstore.py         # FAISS embedding, build/save/load, retrieval
│   └── rag_chain.py           # LangChain RAG pipeline with Claude Opus
│
├── app.py                     # Streamlit chat interface
│
├── assets/
│   ├── sample.pdf             # 8-page RAG technical overview (demo document)
│   └── screenshot.png         # UI screenshot
│
├── data/                      # Place your PDFs here (git-ignored)
├── vectorstore/               # FAISS index saved here (git-ignored)
│
├── .env.example               # API key template
├── .gitignore
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/JAYANSHUBADLANI/rag-pdf-chatbot.git
cd rag-pdf-chatbot
```

### 2. Create a Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure API Keys
```bash
cp .env.example .env
```
Open `.env` and paste your Anthropic API key:
```env
ANTHROPIC_API_KEY=sk-ant-...
```
Get a key at [console.anthropic.com](https://console.anthropic.com).

---

## 🚀 How to Run

```bash
streamlit run app.py
```

Opens at **http://localhost:8501**.

1. Click **"Upload PDF"** in the sidebar (or try `assets/sample.pdf`)
2. Wait ~5–15 s for chunking + indexing
3. Type your question in the chat input
4. View the answer + expand **📚 Sources** for retrieved chunks

---

## ☁️ Deploy to Streamlit Cloud

1. **Push to GitHub** (this repo is already set up)
2. Go to **[share.streamlit.io](https://share.streamlit.io)** → _New app_
3. Select:
   - Repository: `JAYANSHUBADLANI/rag-pdf-chatbot`
   - Branch: `main`
   - Main file: `app.py`
4. Under **Advanced settings → Secrets**, add:
   ```toml
   ANTHROPIC_API_KEY = "sk-ant-..."
   ```
5. Click **Deploy** — live URL in ~2 minutes

> **Note on persistence:** Streamlit Cloud's free tier resets the filesystem each session.  
> The FAISS vectorstore is rebuilt automatically on each new upload — no action needed.  
> For production use, cache the FAISS index to S3/GCS and load it at startup.

---

## 💡 Configuration

| Parameter | Default | File | Description |
|-----------|---------|------|-------------|
| `CHUNK_SIZE` | 1000 | `pdf_processor.py` | Characters per chunk |
| `CHUNK_OVERLAP` | 200 | `pdf_processor.py` | Overlap between chunks |
| `TOP_K_RETRIEVE` | 4 | `rag_chain.py` | Chunks retrieved per query |
| `CLAUDE_MODEL` | `claude-opus-4-6` | `rag_chain.py` | Anthropic model |
| `TEMPERATURE` | 0.2 | `rag_chain.py` | Sampling temperature |
| `MAX_TOKENS` | 1024 | `rag_chain.py` | Max response tokens |
| Embedding model | `all-MiniLM-L6-v2` | `vectorstore.py` | HuggingFace model name |

---

## 📜 License

MIT License — free to use and adapt with attribution.

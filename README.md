# 🤖 RAG PDF Chatbot

**Author:** Jayanshu Badlani
**GitHub:** [JAYANSHUBADLANI](https://github.com/JAYANSHUBADLANI)
**LinkedIn:** [jayanshu-badlani](https://linkedin.com/in/jayanshu-badlani)

---

## 🔍 Project Overview

A production-ready **Retrieval-Augmented Generation (RAG)** chatbot that lets you upload any PDF and ask questions about it in natural language. Powered by **Claude Opus** via the Anthropic API, **FAISS** vector similarity search, and a clean **Streamlit** chat interface.

---

## ✨ Features

- 📄 **PDF Upload** — drag-and-drop or browse to upload any PDF
- ✂️ **Smart Chunking** — splits documents into overlapping chunks for better context retention
- 🧠 **Semantic Search** — FAISS vector store with sentence-transformer embeddings for fast, accurate retrieval
- 💬 **Claude Opus LLM** — state-of-the-art language model for grounded, accurate answers
- 📚 **Source Transparency** — every answer shows the exact source chunks used
- 🗂️ **Chat History** — full conversation history persisted in session state
- 💾 **Persistent Vectorstore** — FAISS index saved to disk so you don't re-index on every reload

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
| Tokenisation | tiktoken |

---

## 📁 Repository Structure

```
rag-pdf-chatbot/
│
├── src/
│   ├── pdf_processor.py       # PDF loading & text chunking
│   ├── vectorstore.py         # FAISS embedding & retrieval
│   └── rag_chain.py           # LangChain RAG pipeline with Claude
│
├── app.py                     # Streamlit chat interface
│
├── assets/
│   ├── sample.pdf             # Sample PDF for demo
│   └── screenshot.png         # App screenshot (add after running)
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
Open `.env` and add your Anthropic API key:
```
ANTHROPIC_API_KEY=sk-ant-...
```
Get a key at [console.anthropic.com](https://console.anthropic.com).

---

## 🚀 How to Run

```bash
streamlit run app.py
```

The app opens at **http://localhost:8501**.

1. Click **"Upload PDF"** and select a PDF file
2. Wait for processing (chunking + indexing takes ~5–15 seconds)
3. Type your question in the chat input
4. View the answer + source references below each response

---

## 💡 Configuration

You can customise chunking and retrieval in `src/pdf_processor.py` and `src/vectorstore.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CHUNK_SIZE` | 1000 | Characters per chunk |
| `CHUNK_OVERLAP` | 200 | Overlap between consecutive chunks |
| `MAX_RETRIEVAL_DOCS` | 4 | Top-k chunks retrieved per query |
| Embedding model | `all-MiniLM-L6-v2` | Swap for any HuggingFace model |

---

## 📜 License

MIT License — free to use and adapt with attribution.

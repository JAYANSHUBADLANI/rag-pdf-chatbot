"""Generate assets/sample.pdf, the demo document used by the app and README.

Requires reportlab (not an app runtime dependency): pip install reportlab
Run with: python scripts/generate_sample_pdf.py
"""
from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, PageBreak

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "assets" / "sample.pdf"

styles = getSampleStyleSheet()
title_style = ParagraphStyle(
    "DocTitle", parent=styles["Heading1"], fontSize=14, spaceAfter=2,
)
subtitle_style = ParagraphStyle(
    "DocSubtitle", parent=styles["Normal"], fontSize=10, textColor="#444444", spaceAfter=14,
)
heading_style = ParagraphStyle(
    "SectionHeading", parent=styles["Heading2"], fontSize=12, spaceBefore=0, spaceAfter=10,
)
body_style = ParagraphStyle(
    "Body", parent=styles["Normal"], fontSize=10.5, leading=15,
)
footer_style = ParagraphStyle(
    "Footer", parent=styles["Normal"], fontSize=9, textColor="#888888", spaceBefore=16,
)

TITLE = "Retrieval-Augmented Generation: Technical Overview"
SUBTITLE = "Sample Document for RAG PDF Chatbot Demo"

PAGES = [
    (
        "1. What is Retrieval-Augmented Generation?",
        "Retrieval-Augmented Generation (RAG) is an AI architecture that combines information "
        "retrieval with large language model (LLM) generation. Instead of relying solely on the "
        "knowledge baked into the LLM during training, RAG first fetches relevant documents from "
        "an external knowledge base and then uses those documents as grounding context for the "
        "model answer. This approach significantly reduces hallucination and keeps answers "
        "factual and up to date.",
    ),
    (
        "2. Core Components of a RAG Pipeline",
        "A typical RAG pipeline consists of four stages: "
        "(a) Document Ingestion: PDFs, web pages, or other sources are loaded and split into "
        "smaller, overlapping text chunks (e.g. 1000 characters with 200-character overlap). "
        "(b) Embedding: Each chunk is converted into a dense vector representation using an "
        "embedding model such as sentence-transformers/all-MiniLM-L6-v2 (384 dimensions) or "
        "OpenAI text-embedding-3-small (1536 dimensions). "
        "(c) Vector Store: Embeddings are stored in a vector database such as FAISS, Pinecone, "
        "or Chroma, enabling millisecond-speed approximate nearest-neighbour search at query "
        "time. "
        "(d) Generation: The user query is embedded, the top-k most similar chunks are "
        "retrieved, and they are injected into a prompt alongside the query before being sent "
        "to the LLM.",
    ),
    (
        "3. FAISS: Facebook AI Similarity Search",
        "FAISS is an open-source library developed by Meta AI Research for efficient similarity "
        "search over dense vectors. It supports billions of vectors on GPU and millions on CPU. "
        "In this project we use the CPU-only faiss-cpu build with the IndexFlatL2 index (exact "
        "L2 search). For larger corpora, the IVFFlat or HNSW indices provide an accuracy/speed "
        "trade-off. FAISS indices are serialisable to disk and can be reloaded without "
        "re-embedding, making them ideal for serverless or Streamlit Cloud deployments.",
    ),
    (
        "4. Llama 3.3 70B Versatile: The Generation Model",
        "Llama 3.3 70B Versatile, served through Groq's free API, is the generation model for "
        "this project. It offers a 128000-token context window and strong performance on "
        "reasoning and long-document tasks, with Groq's inference hardware returning tokens "
        "fast enough to stream answers with almost no perceptible delay. For RAG, a low sampling "
        "temperature (0.2) is recommended to keep answers grounded and consistent. The model is "
        "accessed via the Groq API and integrated into this project through LangChain's ChatGroq "
        "wrapper. Query rewriting for follow-up questions uses the smaller, faster Llama 3.1 8B "
        "Instant model on the same API.",
    ),
    (
        "5. LangChain Integration",
        "LangChain is an open-source framework that provides composable building blocks for LLM "
        "applications. In this project it supplies: "
        "PyPDFLoader: wraps PyPDF for document loading. "
        "RecursiveCharacterTextSplitter: hierarchical text chunking. "
        "HuggingFaceEmbeddings: wrapper around sentence-transformers. "
        "FAISS vector store wrapper: with save/load and as_retriever(). "
        "ChatGroq: chat model wrapper for the Groq API. "
        "PromptTemplate: custom prompt construction.",
    ),
    (
        "6. Chunking Strategy",
        "Text chunking has a significant impact on RAG quality. A chunk that is too large may "
        "exceed the embedding model token limit and dilute the semantic signal. A chunk that is "
        "too small may lack the necessary surrounding context. The RecursiveCharacterTextSplitter "
        "used in this project uses a separator hierarchy: paragraph breaks, line breaks, sentence "
        "endings, and finally individual characters. Chunk size is set to 1000 characters with an "
        "overlap of 200 to prevent information loss at chunk boundaries.",
    ),
    (
        "7. Evaluation Metrics",
        "Common RAG evaluation metrics include: "
        "Faithfulness: does the answer contain only information present in the retrieved "
        "context? "
        "Answer Relevancy: how closely does the answer address the question? "
        "Context Precision: what proportion of retrieved chunks were actually useful? "
        "Context Recall: were all relevant chunks retrieved? "
        "Frameworks such as RAGAS and TruLens provide automated scoring using an LLM-as-a-judge "
        "approach, enabling continuous quality monitoring in production.",
    ),
    (
        "8. Deployment on Streamlit Cloud",
        "This application can be deployed for free on Streamlit Community Cloud: "
        "1. Push the repository to GitHub (public or private). "
        "2. Visit share.streamlit.io and connect your GitHub account. "
        "3. Select the repository, branch (main), and entry point (app.py). "
        "4. Add GROQ_API_KEY under Settings, Secrets. "
        "5. Click Deploy, the app will be live at a public URL in under two minutes. "
        "Note: Streamlit Cloud free tier does not persist the filesystem between sessions, so "
        "the vectorstore/ directory is rebuilt on each new session. For persistence, use a cloud "
        "object store such as AWS S3 or GCS to cache the FAISS index.",
    ),
]

def build() -> None:
    doc = SimpleDocTemplate(
        str(OUTPUT_PATH),
        pagesize=letter,
        topMargin=0.85 * inch,
        bottomMargin=0.85 * inch,
        leftMargin=0.85 * inch,
        rightMargin=0.85 * inch,
    )
    story = []
    for i, (heading, body) in enumerate(PAGES, start=1):
        story.append(Paragraph(TITLE, title_style))
        story.append(Paragraph(SUBTITLE, subtitle_style))
        story.append(Paragraph(heading, heading_style))
        story.append(Paragraph(body, body_style))
        story.append(Paragraph(f"Page {i}", footer_style))
        if i < len(PAGES):
            story.append(PageBreak())
    doc.build(story)
    print(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    build()

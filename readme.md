# rag-core

> An end-to-end Retrieval-Augmented Generation system built from scratch.

`rag-core` transforms unstructured documents into a searchable knowledge base using modern retrieval techniques — chunking, embeddings, vector databases, and semantic search. Retrieved context is then passed to a Large Language Model (Gemini) to generate grounded, context-aware responses.

Unlike framework-heavy implementations, `rag-core` focuses on understanding the **core mechanics** behind RAG systems by building each component manually — no LangChain, no abstractions hiding the logic.

---

## Features

- PDF document ingestion and text extraction
- Recursive and overlapping chunking strategies
- Semantic embeddings via Sentence Transformers
- Vector storage and similarity search using Qdrant
- Approximate Nearest Neighbor (ANN) retrieval with cosine similarity
- Top-K semantic document retrieval
- Context-aware prompt construction
- Gemini-powered answer generation
- Modular architecture designed for experimentation and extension

---

## Architecture

```
Documents (PDFs)
       │
       ▼
Text Extraction          ← PyMuPDF parses raw PDF content
       │
       ▼
   Chunking              ← Recursive + overlapping chunking strategies
       │
       ▼
  Embeddings             ← Sentence Transformers → dense vectors
       │
       ▼
    Qdrant               ← Persistent vector storage
(Vector Store)
       │
       ▼
Semantic Search          ← ANN search via cosine similarity
       │
       ▼
 Top-K Chunks            ← Most relevant context retrieved
       │
       ▼
Prompt Builder           ← Context + question → grounded prompt
       │
       ▼
  Gemini LLM             ← Answer generation
       │
       ▼
Grounded Answer
```

---

## Project Structure

```
rag-core/
├── data/                  # Input PDF documents
├── src/
│   ├── pdf_loader.py      # PDF parsing and text extraction
│   ├── chunker.py         # Fixed-size and overlapping chunking
│   ├── embedder.py        # Sentence Transformer embeddings
│   └── gemini.py          # Gemini LLM response generation
├── main.py                # End-to-end pipeline entry point
├── .env                   # API keys (not committed)
├── requirements.txt
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.9+
- A [Google Gemini API key](https://aistudio.google.com/)

### Installation

```bash
git clone https://github.com/your-username/rag-core.git
cd rag-core
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_api_key_here
```

### Running the pipeline

Place your PDF files in the `data/` directory, then:

```bash
python main.py
```

---

## How It Works

### 1. Document ingestion

PDFs are parsed using PyMuPDF and cleaned to remove noise (extra whitespace, headers, unicode artifacts) before processing.

### 2. Chunking

Text is split using an overlapping chunking strategy. Each chunk maintains a configurable overlap with the previous one to preserve context across boundaries.

```python
chunk_size = 500      # tokens per chunk
overlap_size = 100    # shared tokens between adjacent chunks
```

### 3. Embeddings

Each chunk is converted to a dense vector using a Sentence Transformer model (`all-MiniLM-L6-v2`, 384 dimensions). Semantically similar chunks produce vectors that are close in vector space.

### 4. Vector storage

Vectors and their corresponding text are stored in a Qdrant collection using cosine distance. The collection is persisted to disk and reused across runs.

### 5. Retrieval

At query time, the question is embedded using the same model. Qdrant performs ANN (Approximate Nearest Neighbour) search to return the top-K most semantically similar chunks.

### 6. Generation

Retrieved chunks are assembled into a grounded prompt and sent to Gemini. The model is instructed to answer only from the provided context, preventing hallucination.

```
You are a helpful assistant.
Answer ONLY from the provided context.
If the answer is not present, say: "I could not find that information in the documents."
```

---

## Technologies Used

| Component | Library |
|---|---|
| PDF parsing | PyMuPDF (`fitz`) |
| Embeddings | Sentence Transformers |
| Vector database | Qdrant |
| Numerical ops | NumPy |
| LLM | Google Gemini API |
| Config | Python Dotenv |

---

## Concepts Implemented

**Data ingestion** — PDF parsing, text extraction, data preprocessing

**Chunking** — Fixed-size chunking, chunk overlap, context preservation

**Embeddings** — Dense vector representations, semantic similarity, vector space modeling

**Retrieval** — Top-K retrieval, cosine similarity, ANN search, HNSW concepts

**Generation** — Retrieval-Augmented Generation, context grounding, prompt engineering

---

## Roadmap

This project serves as a foundation for exploring advanced RAG techniques:

- [ ] Re-ranking (cross-encoder)
- [ ] Hybrid search (dense + BM25)
- [ ] Query expansion and rewriting
- [ ] Multi-query retrieval
- [ ] HyDE (Hypothetical Document Embeddings)
- [ ] Parent document retrieval
- [ ] GraphRAG
- [ ] Conversation memory
- [ ] RAGAs evaluation
- [ ] LangChain integration
- [ ] LangGraph agentic RAG

---

## Learning Context

`rag-core` was built as part of a deep dive into Retrieval-Augmented Generation systems. The goal was to understand what happens *inside* frameworks like LangChain by implementing each layer from scratch — from byte streams out of a PDF all the way to a grounded LLM response.

---

## License

MIT
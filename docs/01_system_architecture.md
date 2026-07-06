# System Architecture

## What This System Does

This is a **local Retrieval-Augmented Generation (RAG)** application. It lets you upload PDF documents and then ask natural-language questions about them. The system finds the most relevant passages from your PDFs and feeds them to a language model, which then generates a grounded answer — entirely on your own machine, with no internet connection or API keys required.

---

## High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER (Browser)                           │
│                    http://localhost:8501                         │
└───────────────────────────┬─────────────────────────────────────┘
                            │ HTTP
┌───────────────────────────▼─────────────────────────────────────┐
│                     Streamlit UI  (app.py)                       │
│                                                                  │
│   Sidebar                          Main Area                     │
│   ┌──────────────────┐             ┌──────────────────────────┐  │
│   │ Upload PDFs      │             │ Chat interface           │  │
│   │ Select document  │             │ Display conversation     │  │
│   │ Delete document  │             │ Show source citations    │  │
│   │ Clear history    │             └──────────────────────────┘  │
│   └──────────────────┘                                           │
└──────────┬──────────────────────────────┬────────────────────────┘
           │                              │
           ▼                              ▼
┌──────────────────────┐      ┌───────────────────────────────────┐
│   ingest.py          │      │   LangChain RAG Chain             │
│                      │      │                                   │
│  1. Load PDF pages   │      │  1. History-aware retriever       │
│  2. Clean text       │      │     (rewrites follow-up Qs)       │
│  3. Split into chunks│      │  2. MMR vector retrieval          │
│  4. Embed chunks     │      │  3. LongContextReorder            │
│  5. Upsert to Chroma │      │  4. LLM answer generation         │
└──────────┬───────────┘      └──────────────┬────────────────────┘
           │                                 │
           ▼                                 ▼
┌──────────────────────┐      ┌───────────────────────────────────┐
│   ChromaDB           │      │   Ollama (local process)          │
│   ./VectorDB/        │      │                                   │
│   <collection>/      │◄─────│  • nomic-embed-text (embeddings)  │
│                      │      │  • llama3.2:latest (LLM)          │
│  Persisted on disk   │      │                                   │
└──────────────────────┘      └───────────────────────────────────┘
           │
           ▼
┌──────────────────────┐
│   chat_history/      │
│   <collection>.json  │
│                      │
│  Persisted on disk   │
└──────────────────────┘
```

---

## The Two Main Phases

### Phase 1 — Ingestion (one-time per document)

This happens when you upload a PDF. The goal is to convert raw PDF text into a searchable vector index.

```
PDF File
   │
   ▼
PyPDFLoader          → extracts text page-by-page
   │
   ▼
_clean_text()        → normalises whitespace, collapses blank lines
   │
   ▼
RecursiveCharacterTextSplitter
                     → splits pages into overlapping chunks (~1000 chars)
   │
   ▼
OllamaEmbeddings     → converts each chunk to a 768-dim float vector
(nomic-embed-text)     using the local Ollama process
   │
   ▼
ChromaDB.add_documents()
                     → stores vectors + text + metadata on disk
                       under VectorDB/<collection-name>/
```

After ingestion, the PDF's content lives permanently on disk. You never need to re-ingest unless the file changes.

### Phase 2 — Query (every time you ask a question)

```
User question
   │
   ▼
History-aware retriever
   │  Uses the last N chat turns to rewrite the question
   │  into a self-contained standalone question
   │  (so follow-up questions like "tell me more" work correctly)
   │
   ▼
MMR Retrieval from ChromaDB
   │  Finds the top-k most relevant chunks
   │  MMR ensures diversity — avoids returning near-duplicate chunks
   │
   ▼
LongContextReorder
   │  Reorders retrieved chunks so the most relevant ones
   │  appear at the start and end of the context window
   │  (LLMs tend to ignore content in the middle)
   │
   ▼
LLM (llama3.2:latest)
   │  Receives: system prompt + chat history + context chunks + question
   │  Generates a grounded answer
   │
   ▼
Answer + source citations displayed in UI
```

---

## Component Responsibilities

| Component | File | Responsibility |
|---|---|---|
| Configuration | `config.py` | Single source of truth for all tunable parameters |
| Ingestion pipeline | `ingest.py` | PDF → chunks → embeddings → ChromaDB |
| Streamlit UI | `app.py` | User interface, chain orchestration, session management |
| App launcher | `run.py` | Convenience wrapper to start Streamlit |
| Vector store | `VectorDB/` | Persisted ChromaDB collections, one per PDF |
| Chat history | `chat_history/` | JSON files storing conversation per PDF |
| LLM + Embeddings | Ollama (external) | Local model inference, no cloud dependency |

---

## Data Flow Summary

```
WRITE path (ingestion):
  PDF → text → chunks → embeddings → VectorDB/<name>/

READ path (query):
  question + chat_history → standalone question
                          → vector search → top-k chunks
                          → LLM → answer
                          → chat_history/<name>.json
```

---

## Why RAG Instead of Just Asking the LLM?

A plain LLM only knows what it was trained on. It cannot answer questions about your specific documents. RAG solves this by:

1. Storing your document content in a searchable vector database
2. At query time, retrieving only the relevant passages
3. Injecting those passages into the LLM's context window as grounding evidence

This means the LLM answers based on your actual document content, not hallucinated knowledge.

---

## Storage Layout

```
project root/
├── VectorDB/
│   └── KTM___Wikipedia/        ← one folder per indexed PDF
│       ├── chroma.sqlite3       ← ChromaDB metadata + index
│       └── ...                  ← binary vector data
├── chat_history/
│   └── KTM___Wikipedia.json    ← persisted conversation for that PDF
```

Collection names are derived from the PDF filename: special characters are replaced with underscores, and the name is capped at 60 characters to satisfy ChromaDB's naming rules.

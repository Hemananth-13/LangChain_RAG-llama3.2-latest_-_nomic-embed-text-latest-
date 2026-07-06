# LangChain RAG — llama3.2 + nomic-embed-text

A local Retrieval-Augmented Generation (RAG) app that lets you chat with your PDF documents. Runs entirely on your machine using Ollama — no API keys required.

## Stack

| Layer | Tool |
|---|---|
| LLM | `llama3.2:latest` via Ollama |
| Embeddings | `nomic-embed-text:latest` via Ollama |
| Vector store | ChromaDB (persisted locally at `./VectorDB`) |
| Framework | LangChain |
| UI | Streamlit |

## Prerequisites

- Python ≥ 3.11
- [Ollama](https://ollama.com) installed and running

## Setup

**1. Pull the required Ollama models**
```bash
ollama pull llama3.2:latest
ollama pull nomic-embed-text:latest
```

**2. Install dependencies**

Using uv (recommended):
```bash
uv sync
```

Or pip:
```bash
pip install -r requirements.txt
```

**3. Run the app**
```bash
python run.py
```

Or directly:
```bash
streamlit run app.py
```

## Usage

1. Open the app in your browser (default: http://localhost:8501)
2. Upload one or more PDFs via the sidebar and click **Ingest uploaded PDF(s)**
3. Once indexed, type your question in the chat input

## Ingesting PDFs from the command line

```bash
# Single file
python ingest.py "KTM - Wikipedia.pdf"

# Entire folder
python ingest.py ./my_pdfs_folder

# With custom chunk settings and a test query
python ingest.py "KTM - Wikipedia.pdf" --chunk-size 1200 --chunk-overlap 200 --test-query "founding year"
```

## Configuration

All tunable parameters live in `config.py`:

| Parameter | Default | Description |
|---|---|---|
| `LLM_MODEL` | `llama3.2:latest` | Ollama LLM model |
| `EMBEDDING_MODEL` | `nomic-embed-text:latest` | Ollama embedding model |
| `RETRIEVER_K` | `4` | Top-k docs retrieved per query |
| `HISTORY_TURNS_KEPT` | `3` | Prior Q&A turns fed into the chain |
| `DEFAULT_CHUNK_SIZE` | `1000` | Characters per chunk |
| `DEFAULT_CHUNK_OVERLAP` | `150` | Overlap between chunks |

## Project structure

```
├── app.py          # Streamlit chat UI
├── ingest.py       # PDF ingestion pipeline
├── config.py       # Centralized configuration
├── run.py          # App launcher
├── requirements.txt
├── pyproject.toml
└── VectorDB/       # Auto-created on first ingest (gitignored)
```

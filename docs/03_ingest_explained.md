# ingest.py — Line-by-Line Explanation

`ingest.py` is the ingestion pipeline. Its job is to take raw PDF files and transform them into a searchable vector database. It can be used in two ways:

1. **Imported by `app.py`** — called automatically when you upload a PDF through the Streamlit UI
2. **Run directly from the command line** — for batch ingestion or testing

---

## Imports

```python
import argparse
```
Standard library module for parsing command-line arguments. Used only when `ingest.py` is run directly (not when imported by `app.py`).

```python
import hashlib
```
Used to generate deterministic chunk IDs via SHA-256 hashing. This ensures that re-ingesting the same file produces the same IDs, enabling upsert behaviour (update if exists, insert if not).

```python
import re
```
Regular expressions. Used in `_clean_text()` to normalise whitespace in extracted PDF text.

```python
from pathlib import Path
```
Object-oriented filesystem path handling.

```python
from langchain_community.document_loaders import PyPDFLoader
```
LangChain's PDF loader. Internally uses the `pypdf` library to extract text from each page of a PDF. Returns a list of `Document` objects, one per page, each with `page_content` (the text) and `metadata` (including `source` filepath and `page` number).

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
```
Splits long documents into smaller chunks. "Recursive" means it tries to split on paragraph breaks first, then sentences, then words, then characters — always preferring natural boundaries over arbitrary cuts.

```python
from langchain_ollama import OllamaEmbeddings
```
LangChain's wrapper around Ollama's embedding endpoint. Sends text to the locally running Ollama process and receives back float vectors.

```python
from langchain_chroma import Chroma
```
LangChain's wrapper around ChromaDB. Handles storing, retrieving, and searching vectors on disk.

```python
from config import (
    VECTORDB_DIR,
    PERSIST_DIRECTORY,
    EMBEDDING_MODEL,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
)
```
Imports all relevant settings from the central config file.

---

## Private Helper Functions

### `_clean_text(text: str) -> str`

```python
def _clean_text(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
```
Collapses any sequence of spaces or tabs into a single space. PDF extraction often produces multiple consecutive spaces, especially around formatted content like tables or columns.

```python
    text = re.sub(r"\n{3,}", "\n\n")
```
Collapses three or more consecutive newlines into exactly two. This preserves paragraph breaks (double newline) while removing excessive blank lines that PDFs often contain between sections.

```python
    return text.strip()
```
Removes leading and trailing whitespace from the entire page text.

**Why this matters:** Cleaner text produces better embeddings. Noise like `"  "` (multiple spaces) or `"\n\n\n\n"` (many blank lines) adds no semantic meaning but wastes tokens and can confuse the embedding model.

---

### `_make_id(source, page, chunk_index, content) -> str`

```python
def _make_id(source: str, page: int, chunk_index: int, content: str) -> str:
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
    return f"{Path(source).stem}-p{page}-c{chunk_index}-{digest}"
```

Generates a unique, deterministic string ID for each chunk. The ID encodes:
- `Path(source).stem` — the PDF filename without extension (e.g. `KTM - Wikipedia`)
- `p{page}` — the page number the chunk came from
- `c{chunk_index}` — the index of this chunk within that page
- `{digest}` — first 16 hex characters of the SHA-256 hash of the chunk's text content

**Example output:** `KTM - Wikipedia-p3-c1-a4f2b8c1d9e0f123`

**Why deterministic IDs matter:** ChromaDB's `add_documents()` with explicit IDs performs an **upsert** — if a document with that ID already exists, it updates it; otherwise it inserts it. This means you can safely re-ingest the same PDF without creating duplicate entries in the database.

---

### `_get_embedding_model() -> OllamaEmbeddings`

```python
def _get_embedding_model() -> OllamaEmbeddings:
    return OllamaEmbeddings(model=EMBEDDING_MODEL)
```

A simple factory function that creates a fresh `OllamaEmbeddings` instance. Used in `ingest_pdfs()` and in the CLI test-query path. Kept separate so the embedding model name is always sourced from `config.py`.

---

## Public Functions

### `load_pdfs(paths) -> list`

```python
def load_pdfs(paths) -> list:
    pdf_files = []
    for raw_path in paths:
        p = Path(raw_path)
        if p.is_dir():
            pdf_files.extend(sorted(p.glob("*.pdf")))
        elif p.suffix.lower() == ".pdf":
            pdf_files.append(p)
        else:
            raise ValueError(f"Not a PDF file or directory: {p}")
```
Accepts a list of paths (files or directories). For each:
- If it's a directory: finds all `.pdf` files inside it, sorted alphabetically
- If it's a `.pdf` file: adds it directly
- Otherwise: raises an error immediately rather than silently skipping

```python
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in: {paths}")
```
Fails fast if no PDFs were found — better than silently producing an empty index.

```python
    pages = []
    for pdf_path in pdf_files:
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        if pdf_path.stat().st_size == 0:
            raise ValueError(f"PDF is empty: {pdf_path}")
        loaded = PyPDFLoader(str(pdf_path)).load()
        for page in loaded:
            page.page_content = _clean_text(page.page_content)
        pages.extend(loaded)
        print(f"Loaded {len(loaded)} page(s) from {pdf_path.name}")
```
For each PDF:
1. Verifies the file exists and is non-empty
2. Loads all pages using `PyPDFLoader` — each page becomes a `Document` object
3. Cleans the text of each page in-place using `_clean_text()`
4. Appends all pages to the running list
5. Prints progress to the terminal

Returns a flat list of `Document` objects, one per page across all input PDFs.

---

### `collection_name_for(pdf_name: str) -> str`

```python
def collection_name_for(pdf_name: str) -> str:
    stem = Path(pdf_name).stem
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", stem)[:60]
    return safe or "default"
```

Converts a PDF filename into a safe ChromaDB collection name:
1. Takes the filename stem (no extension): `"KTM - Wikipedia.pdf"` → `"KTM - Wikipedia"`
2. Replaces any character that isn't alphanumeric, underscore, or hyphen with `_`: `"KTM___Wikipedia"`
3. Truncates to 60 characters (ChromaDB has a 63-character limit)
4. Falls back to `"default"` if the result is empty

**Example:** `"My Report (2024).pdf"` → `"My_Report__2024_"`

---

### `persist_dir_for(pdf_name: str) -> str`

```python
def persist_dir_for(pdf_name: str) -> str:
    return str(VECTORDB_DIR / collection_name_for(pdf_name))
```

Returns the full filesystem path where a PDF's ChromaDB data will be stored. Each PDF gets its own isolated sub-directory under `VectorDB/`.

**Example:** `"KTM - Wikipedia.pdf"` → `"C:/.../VectorDB/KTM___Wikipedia"`

---

### `list_indexed_files() -> list[str]`

```python
def list_indexed_files() -> list[str]:
    if not VECTORDB_DIR.exists():
        return []
    return sorted(p.name for p in VECTORDB_DIR.iterdir() if p.is_dir())
```

Scans the `VectorDB/` directory and returns the names of all sub-directories (each representing an indexed PDF). Returns an empty list if `VectorDB/` doesn't exist yet. Used by `app.py` to populate the sidebar document list.

---

### `delete_indexed_file(collection: str) -> None`

```python
def delete_indexed_file(collection: str) -> None:
    import shutil
    target = VECTORDB_DIR / collection
    if target.exists():
        shutil.rmtree(target)
```

Permanently deletes a collection's ChromaDB directory. `shutil.rmtree` removes the directory and all its contents recursively. The `import shutil` is inside the function because this is a rarely-called destructive operation — keeping the import local makes it clear this function has side effects.

---

### `ingest_pdfs(paths, persist_directory, chunk_size, chunk_overlap) -> dict`

This is the core function. It orchestrates the entire ingestion pipeline.

```python
def ingest_pdfs(
    paths,
    persist_directory: str | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> dict:
```
Parameters:
- `paths` — list of PDF file paths or directory paths
- `persist_directory` — where to store the ChromaDB data; if `None`, derived from the first PDF's name
- `chunk_size` / `chunk_overlap` — override the defaults from `config.py`

```python
    pages = load_pdfs(paths)
```
Step 1: Load and clean all PDF pages.

```python
    if persist_directory is None:
        first_pdf = Path(paths[0]) if not Path(paths[0]).is_dir() else next(Path(paths[0]).glob("*.pdf"))
        persist_directory = persist_dir_for(first_pdf.name)
```
Step 2: Determine where to store the data. If `persist_directory` wasn't explicitly provided, derive it from the first PDF's filename. Handles both file and directory inputs.

```python
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(pages)
```
Step 3: Split all pages into chunks. `split_documents()` preserves the `metadata` from each page (source path, page number) and copies it to every chunk derived from that page.

```python
    chunk_counts_per_page: dict = {}
    ids = []
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        page = chunk.metadata.get("page", 0)
        key = (source, page)
        chunk_counts_per_page[key] = chunk_counts_per_page.get(key, -1) + 1
        ids.append(_make_id(source, page, chunk_counts_per_page[key], chunk.page_content))
```
Step 4: Generate a deterministic ID for every chunk. The `chunk_counts_per_page` dictionary tracks how many chunks have been generated from each (source, page) pair so far, providing the `chunk_index` component of the ID. Starting from `-1` and incrementing means the first chunk gets index `0`.

```python
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=_get_embedding_model(),
    )
    vectorstore.add_documents(documents=chunks, ids=ids)
```
Step 5: Open (or create) the ChromaDB collection at the target directory, then upsert all chunks. ChromaDB automatically calls the embedding model on each chunk's text and stores the resulting vectors alongside the text and metadata.

```python
    return {
        "files": len({c.metadata.get("source") for c in pages}),
        "pages": len(pages),
        "chunks": len(chunks),
        "persist_directory": persist_directory,
    }
```
Returns a summary dict. The `files` count uses a set comprehension to count unique source paths (deduplicates pages from the same file).

---

## CLI Entry Point

### `_parse_args()`

Defines the command-line interface:
- `paths` (positional, one or more) — PDF files or folders to ingest
- `--persist-directory` — override the storage location
- `--chunk-size` / `--chunk-overlap` — override chunking parameters
- `--test-query` — after ingesting, run a retrieval test with this query string

### `main()`

Called when the script is run directly (`python ingest.py ...`). Parses arguments, calls `ingest_pdfs()`, prints the summary, and optionally runs a test retrieval to verify the index is working.

---

## Complete Ingestion Flow Diagram

```
paths (list of PDF files/dirs)
         │
         ▼
    load_pdfs()
    ┌─────────────────────────────────────┐
    │ For each PDF:                       │
    │   PyPDFLoader.load()                │
    │     → [Document(page=0), ...]       │
    │   _clean_text() on each page        │
    └─────────────────────────────────────┘
         │ flat list of Document objects
         ▼
    RecursiveCharacterTextSplitter
    .split_documents()
         │ list of chunk Documents
         ▼
    Generate IDs with _make_id()
    (deterministic, content-based)
         │
         ▼
    Chroma.add_documents(chunks, ids)
    ┌─────────────────────────────────────┐
    │ For each chunk:                     │
    │   OllamaEmbeddings.embed(text)      │
    │     → 768-dim float vector          │
    │   Store (id, vector, text, metadata)│
    │   in SQLite + binary files on disk  │
    └─────────────────────────────────────┘
         │
         ▼
    VectorDB/<collection-name>/
    (persisted, ready for retrieval)
```

"""
Ingest PDF documents into the local Chroma vector store used by app.py.

Run directly from the command line to build (or update) the index:

    python ingest.py "KTM - Wikipedia.pdf"
    python ingest.py ./my_pdfs_folder --chunk-size 1200 --chunk-overlap 200
    python ingest.py "KTM - Wikipedia.pdf" --test-query "November 2025"

Or import `ingest_pdfs()` to add documents from inside app.py (e.g. from a
Streamlit file-uploader), which is exactly what app.py's sidebar does.

Note: this file used to be named "Vector Formation.py". It's renamed to
ingest.py because Python can't cleanly import a module whose filename has a
space in it, and app.py now imports this module directly.
"""

import argparse
import hashlib
import re
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# --------------------------------------------------------------------------
# Config - shared with app.py so both files always agree on where/how the
# index is built.
# --------------------------------------------------------------------------
PERSIST_DIRECTORY = "./VectorDB"
EMBEDDING_MODEL = "nomic-embed-text:latest"

# 120/24 (the original values) is roughly one short sentence per chunk, which
# starves the LLM of context at answer time. ~1000 characters is closer to a
# paragraph and retrieves far more usefully for Q&A.
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 150


def _clean_text(text: str) -> str:
    """Collapse the extra whitespace/line breaks PDF extraction tends to leave behind."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _make_id(source: str, page: int, chunk_index: int, content: str) -> str:
    """
    Deterministic ID for a chunk, so re-ingesting the same file updates its
    entries instead of piling up duplicates in Chroma. If your chromadb
    version doesn't upsert cleanly on repeated IDs, delete ./VectorDB and
    re-run to rebuild from scratch.
    """
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
    return f"{Path(source).stem}-p{page}-c{chunk_index}-{digest}"


def load_pdfs(paths) -> list:
    """Load one or more PDFs (files, or directories containing PDFs) into Document pages."""
    pdf_files = []
    for raw_path in paths:
        p = Path(raw_path)
        if p.is_dir():
            pdf_files.extend(sorted(p.glob("*.pdf")))
        elif p.suffix.lower() == ".pdf":
            pdf_files.append(p)
        else:
            raise ValueError(f"Not a PDF file or directory: {p}")

    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in: {paths}")

    pages = []
    for pdf_path in pdf_files:
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        loaded = PyPDFLoader(str(pdf_path)).load()
        for page in loaded:
            page.page_content = _clean_text(page.page_content)
        pages.extend(loaded)
        print(f"Loaded {len(loaded)} page(s) from {pdf_path.name}")

    return pages


def ingest_pdfs(
    paths,
    persist_directory: str = PERSIST_DIRECTORY,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> dict:
    """
    Load, split, embed, and store PDF(s) in the persistent Chroma vector store.

    Safe to call repeatedly (e.g. once per uploaded file): chunks get
    deterministic ids, so re-ingesting the same file upserts rather than
    duplicating entries. Returns a small stats dict for display in the UI.
    """
    pages = load_pdfs(paths)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(pages)

    # Build a deterministic id per chunk: source file + page + position on
    # that page + content hash. This is what makes re-ingestion idempotent.
    chunk_counts_per_page: dict = {}
    ids = []
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        page = chunk.metadata.get("page", 0)
        key = (source, page)
        chunk_counts_per_page[key] = chunk_counts_per_page.get(key, -1) + 1
        ids.append(_make_id(source, page, chunk_counts_per_page[key], chunk.page_content))

    embedding_model = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model,
    )
    vectorstore.add_documents(documents=chunks, ids=ids)

    return {
        "files": len({c.metadata.get("source") for c in pages}),
        "pages": len(pages),
        "chunks": len(chunks),
        "persist_directory": persist_directory,
    }


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", help="PDF file(s) or folder(s) of PDFs to ingest")
    parser.add_argument("--persist-directory", default=PERSIST_DIRECTORY)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    parser.add_argument(
        "--test-query",
        default=None,
        help="Optional: run a sample retrieval after ingesting, to sanity-check the index",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    stats = ingest_pdfs(
        args.paths,
        persist_directory=args.persist_directory,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    print(
        f"\nIndexed {stats['files']} file(s), {stats['pages']} page(s), "
        f"{stats['chunks']} chunk(s) -> {stats['persist_directory']}"
    )

    if args.test_query:
        embedding_model = OllamaEmbeddings(model=EMBEDDING_MODEL)
        vectorstore = Chroma(
            persist_directory=args.persist_directory,
            embedding_function=embedding_model,
        )
        retriever = vectorstore.as_retriever(search_type="mmr")
        results = retriever.invoke(args.test_query)
        print(f"\nTop results for test query: {args.test_query!r}")
        print("-" * 50)
        for i, doc in enumerate(results, start=1):
            print(f"\nResult {i} (page {doc.metadata.get('page')})")
            print(doc.page_content)


if __name__ == "__main__":
    main()

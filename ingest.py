"""
Ingest PDF documents into the local Chroma vector store used by app.py.

Run directly from the command line to build (or update) the index:

    python ingest.py "KTM - Wikipedia.pdf"
    python ingest.py ./my_pdfs_folder --chunk-size 1200 --chunk-overlap 200
    python ingest.py "KTM - Wikipedia.pdf" --test-query "November 2025"

Or import `ingest_pdfs()` to add documents from inside app.py (e.g. from a
Streamlit file-uploader), which is exactly what app.py's sidebar does.
"""

import argparse
import hashlib
import re
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

from config import (
    VECTORDB_DIR,
    PERSIST_DIRECTORY,
    EMBEDDING_MODEL,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
)


def _clean_text(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _make_id(source: str, page: int, chunk_index: int, content: str) -> str:
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
    return f"{Path(source).stem}-p{page}-c{chunk_index}-{digest}"


def _get_embedding_model() -> OllamaEmbeddings:
    return OllamaEmbeddings(model=EMBEDDING_MODEL)


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

    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in: {paths}")

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

    return pages


def collection_name_for(pdf_name: str) -> str:
    """Derive a safe ChromaDB collection name from a PDF filename."""
    stem = Path(pdf_name).stem
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", stem)[:60]
    return safe or "default"


def persist_dir_for(pdf_name: str) -> str:
    """Each PDF gets its own sub-directory inside VectorDB."""
    return str(VECTORDB_DIR / collection_name_for(pdf_name))


def list_indexed_files() -> list[str]:
    """Return collection names (== PDF stems) that have been indexed."""
    if not VECTORDB_DIR.exists():
        return []
    return sorted(p.name for p in VECTORDB_DIR.iterdir() if p.is_dir())


def delete_indexed_file(collection: str) -> None:
    """Delete the ChromaDB directory for a given collection name."""
    import shutil
    target = VECTORDB_DIR / collection
    if target.exists():
        shutil.rmtree(target)


def ingest_pdfs(
    paths,
    persist_directory: str | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> dict:
    """
    Load, split, embed, and store PDF(s) in the persistent Chroma vector store.
    Each PDF gets its own sub-directory under VectorDB (derived from its filename).
    Safe to call repeatedly — chunks get deterministic ids, so re-ingesting
    the same file upserts rather than duplicating entries.
    """
    pages = load_pdfs(paths)

    # Derive persist directory from the first PDF's name if not explicitly given
    if persist_directory is None:
        first_pdf = Path(paths[0]) if not Path(paths[0]).is_dir() else next(Path(paths[0]).glob("*.pdf"))
        persist_directory = persist_dir_for(first_pdf.name)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(pages)

    chunk_counts_per_page: dict = {}
    ids = []
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        page = chunk.metadata.get("page", 0)
        key = (source, page)
        chunk_counts_per_page[key] = chunk_counts_per_page.get(key, -1) + 1
        ids.append(_make_id(source, page, chunk_counts_per_page[key], chunk.page_content))

    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=_get_embedding_model(),
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
    parser.add_argument("--test-query", default=None, help="Run a sample retrieval after ingesting")
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
        vectorstore = Chroma(
            persist_directory=args.persist_directory,
            embedding_function=_get_embedding_model(),
        )
        results = vectorstore.as_retriever(search_type="mmr").invoke(args.test_query)
        print(f"\nTop results for test query: {args.test_query!r}")
        print("-" * 50)
        for i, doc in enumerate(results, start=1):
            print(f"\nResult {i} (page {doc.metadata.get('page')})")
            print(doc.page_content)


if __name__ == "__main__":
    main()

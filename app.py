import json
import os
import tempfile
from pathlib import Path

from annotated_types import doc

import streamlit as st
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import DocumentCompressorPipeline
from langchain_community.document_transformers import LongContextReorder
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma

from config import (
    LLM_MODEL,
    EMBEDDING_MODEL,
    RETRIEVER_K,
    HISTORY_TURNS_KEPT,
    LLM_PARAMS,
    CHAT_HISTORY_DIR,
)
from ingest import ingest_pdfs, list_indexed_files, delete_indexed_file, persist_dir_for

st.set_page_config(page_title="Ask me anything", page_icon="📄")

# ---------------------------------------------------------------------------
# Chat history persistence
# ---------------------------------------------------------------------------

def _history_path(collection: str) -> Path:
    CHAT_HISTORY_DIR.mkdir(exist_ok=True)
    return CHAT_HISTORY_DIR / f"{collection}.json"


def load_chat_history(collection: str) -> tuple[list, list]:
    """Returns (conversation, chat_history) for a given collection."""
    path = _history_path(collection)
    if not path.exists():
        return [], []
    data = json.loads(path.read_text(encoding="utf-8"))
    conversation = data.get("conversation", [])
    chat_history = []
    for msg in data.get("chat_history", []):
        if msg["type"] == "human":
            chat_history.append(HumanMessage(content=msg["content"]))
        else:
            chat_history.append(AIMessage(content=msg["content"]))
    return conversation, chat_history


def save_chat_history(collection: str, conversation: list, chat_history: list) -> None:
    path = _history_path(collection)
    serialized_history = [
        {"type": "human" if isinstance(m, HumanMessage) else "ai", "content": m.content}
        for m in chat_history
    ]
    path.write_text(
        json.dumps({"conversation": conversation, "chat_history": serialized_history}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def delete_chat_history(collection: str) -> None:
    path = _history_path(collection)
    if path.exists():
        path.unlink()


# ---------------------------------------------------------------------------
# LLM / retriever / chain (cached, keyed by collection)
# ---------------------------------------------------------------------------

@st.cache_resource()
def load_llm():
    return ChatOllama(
        model=LLM_MODEL,
        verbose=True,
        num_thread=os.cpu_count() or 8,
        **LLM_PARAMS,
    )


@st.cache_resource()
def load_embedding_model():
    return OllamaEmbeddings(model=EMBEDDING_MODEL)


@st.cache_resource()
def load_vectorstore(collection: str):
    return Chroma(
        persist_directory=persist_dir_for(collection),
        embedding_function=load_embedding_model(),
    )


@st.cache_resource()
def load_chain(collection: str):
    vectorstore = load_vectorstore(collection)
    base_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": RETRIEVER_K, "fetch_k": RETRIEVER_K * 3},
    )
    pipeline_compressor = DocumentCompressorPipeline(transformers=[LongContextReorder()])
    retriever = ContextualCompressionRetriever(
        base_compressor=pipeline_compressor, base_retriever=base_retriever
    )

    llm = load_llm()

    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Given the chat history and the latest user question, rewrite it as a "
         "standalone question that can be understood without the chat history. "
         "Do NOT answer the question. If it's already standalone, return it as-is."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    answer_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an assistant for question-answering tasks. Use the retrieved "
         "context below to answer the latest user question. If you don't know "
         "the answer, say you don't know, do NOT make one up.\n\nContext:\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_prompt)
    question_answer_chain = create_stuff_documents_chain(llm, answer_prompt)
    return create_retrieval_chain(history_aware_retriever, question_answer_chain)


# ---------------------------------------------------------------------------
# Session state bootstrap
# ---------------------------------------------------------------------------

if "active_collection" not in st.session_state:
    st.session_state.active_collection = None
if "conversation" not in st.session_state:
    st.session_state.conversation = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


def switch_collection(collection: str) -> None:
    """Save current history, then load the selected collection's history."""
    current = st.session_state.active_collection
    if current:
        save_chat_history(current, st.session_state.conversation, st.session_state.chat_history)
    st.session_state.active_collection = collection
    conv, hist = load_chat_history(collection)
    st.session_state.conversation = conv
    st.session_state.chat_history = hist


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def generate_answer(question: str, chain, container) -> None:
    trimmed_history = st.session_state.chat_history[-(HISTORY_TURNS_KEPT * 2):]
    callback = StreamlitCallbackHandler(container)
    try:
        response = chain.invoke(
            {"input": question, "chat_history": trimmed_history},
            config={"callbacks": [callback]},
        )
    except Exception as e:
        st.error(f"Failed to generate an answer: {e}")
        return

    answer_text = response["answer"]
    sources = [
    {
        "page_content": doc.page_content,
        "metadata": doc.metadata,
    }
    for doc in response.get("context", [])
]

    st.session_state.conversation.append(
        {"question": question, "answer": answer_text, "sources": sources}
    )
    st.session_state.chat_history.append(HumanMessage(content=question))
    st.session_state.chat_history.append(AIMessage(content=answer_text))
    save_chat_history(
        st.session_state.active_collection,
        st.session_state.conversation,
        st.session_state.chat_history,
    )


def display_conversation() -> None:
    for entry in st.session_state.conversation:
        with st.chat_message("Human"):
            st.write(entry["question"])
        with st.chat_message("AI"):
            st.write(entry["answer"])
            sources = entry.get("sources")
            if sources:
                with st.expander(f"Sources ({len(sources)} chunk(s))"):
                    seen = set()
                    for doc in sources:
                        page = doc["metadata"].get("page")
                        name = Path(doc["metadata"].get("source", "unknown")).name
                        key = (name, page)
                        if key in seen:
                            continue
                        seen.add(key)
                        st.markdown(f"**{name} — page {page}**")
                        snippet = doc["page_content"][:300]
                        if len(doc["page_content"]) > 300:
                            snippet += "…"
                        st.caption(snippet)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.subheader("Knowledge base")

    # --- Upload & ingest ---
    uploaded_files = st.file_uploader("Add PDFs to the index", type="pdf", accept_multiple_files=True)
    if uploaded_files and st.button("Ingest uploaded PDF(s)"):
        tmp_dir = Path(tempfile.mkdtemp())
        progress = st.progress(0, text="Starting ingestion…")
        try:
            for i, f in enumerate(uploaded_files):
                progress.progress((i + 1) / len(uploaded_files), text=f"Indexing {f.name}…")
                dest = tmp_dir / f.name
                dest.write_bytes(f.getvalue())
                stats = ingest_pdfs([dest])
            progress.empty()
            st.success(f"Indexed {stats['pages']} page(s) into {stats['chunks']} chunk(s).")
            load_vectorstore.clear()
            load_chain.clear()
            st.rerun()
        except Exception as e:
            progress.empty()
            st.error(f"Ingestion failed: {e}")

    st.divider()

    # --- File selector ---
    indexed = list_indexed_files()
    if indexed:
        st.markdown("**Indexed documents**")
        for collection in indexed:
            col1, col2 = st.columns([4, 1])
            with col1:
                is_active = st.session_state.active_collection == collection
                label = f"{'▶ ' if is_active else ''}{collection}"
                if st.button(label, key=f"sel_{collection}", use_container_width=True):
                    switch_collection(collection)
                    st.rerun()
            with col2:
                if st.button("🗑", key=f"del_{collection}", help=f"Delete {collection}"):
                    # Save current history before deleting if it's the active one
                    if st.session_state.active_collection == collection:
                        st.session_state.active_collection = None
                        st.session_state.conversation = []
                        st.session_state.chat_history = []
                    delete_indexed_file(collection)
                    delete_chat_history(collection)
                    load_vectorstore.clear()
                    load_chain.clear()
                    st.rerun()

    st.divider()

    # --- Clear conversation ---
    if st.session_state.conversation and st.button("Clear conversation"):
        st.session_state.conversation = []
        st.session_state.chat_history = []
        if st.session_state.active_collection:
            save_chat_history(st.session_state.active_collection, [], [])
        st.rerun()


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

indexed = list_indexed_files()

if not indexed:
    st.info("No documents indexed yet. Upload a PDF in the sidebar to get started.")
else:
    # Auto-select first file if nothing is active
    if st.session_state.active_collection not in indexed:
        switch_collection(indexed[0])

    active = st.session_state.active_collection
    st.header(f"📄 {active}")

    chain = load_chain(active)
    display_conversation()

    user_query = st.chat_input("Ask me anything: ")
    if user_query:
        with st.chat_message("Human"):
            st.write(user_query)
        with st.chat_message("AI"):
            callback_container = st.container()
            with st.spinner("Generating answer..."):
                generate_answer(user_query, chain, callback_container)
        st.rerun()

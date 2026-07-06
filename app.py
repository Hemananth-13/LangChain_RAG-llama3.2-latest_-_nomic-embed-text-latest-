import os
import tempfile
from pathlib import Path

import streamlit as st
# NOTE: as of langchain v1.0, "classic" building blocks (chains, retrievers)
# moved out of the core `langchain` package into `langchain-classic`.
# Run: pip install -U langchain-classic langchain-chroma
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import DocumentCompressorPipeline
from langchain_community.document_transformers import LongContextReorder
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma

from ingest import ingest_pdfs, PERSIST_DIRECTORY, EMBEDDING_MODEL

# --------------------------------------------------------------------------
# Config - tune these for your machine / models
# --------------------------------------------------------------------------
LLM_MODEL = "llama3.2:latest"
RETRIEVER_K = 4          # docs pulled per retriever, before the ensemble merges/dedupes them
HISTORY_TURNS_KEPT = 3   # how many prior Q&A turns get fed back into the chain

st.set_page_config(page_title="Ask me anything", page_icon="📄")
st.header("Ask me anything")


def vectorstore_ready(persist_directory: str = PERSIST_DIRECTORY) -> bool:
    path = Path(persist_directory)
    return path.exists() and any(path.iterdir())


@st.cache_resource()
def load_llm():
    llm = ChatOllama(
        model=LLM_MODEL,
        verbose=True,
        num_gpu=-1,                      # offload all layers to GPU if one is available
        num_ctx=16000,
        num_predict=500,
        num_batch=8192,
        num_thread=os.cpu_count() or 8,  # tune to your machine's physical core count
        temperature=0.01,
        top_p=0.95,
        top_k=40,
        repeat_penalty=1.2,
    )
    return llm


@st.cache_resource()
def local_embedding_model():
    return OllamaEmbeddings(model=EMBEDDING_MODEL)


@st.cache_resource()
def load_vectorstore():
    return Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=local_embedding_model())


@st.cache_resource()
def load_final_retriever():
    vectorstore = load_vectorstore()
    retriever_1 = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": RETRIEVER_K})
    retriever_2 = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": RETRIEVER_K})
    ensembled_retriever = EnsembleRetriever(retrievers=[retriever_1, retriever_2], weights=[0.5, 0.5])

    reordering = LongContextReorder()
    pipeline_compressor = DocumentCompressorPipeline(transformers=[reordering])
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=pipeline_compressor, base_retriever=ensembled_retriever
    )
    return compression_retriever


def build_contextualize_prompt():
    """Turns a follow-up question + chat history into a standalone search query."""
    system_prompt = (
        "Given the chat history and the latest user question, rewrite it as a "
        "standalone question that can be understood without the chat history. "
        "Do NOT answer the question. If it's already standalone, return it as-is."
    )
    return ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )


def build_answer_prompt():
    """Turns retrieved context + chat history + the question into a final answer."""
    system_prompt = (
        "You are an assistant for question-answering tasks. Use the retrieved "
        "context below to answer the latest user question. If you don't know "
        "the answer, say you don't know, do NOT make one up.\n\n"
        "Context:\n{context}"
    )
    return ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )


@st.cache_resource()
def chains_of_qa():
    llm = load_llm()
    retriever = load_final_retriever()

    # This is what keeps conversation history OUT of the retrieval query itself:
    # the retriever only ever sees a clean, standalone question, so multi-turn
    # chats don't dilute vector search with the whole chat log.
    history_aware_retriever = create_history_aware_retriever(llm, retriever, build_contextualize_prompt())
    question_answer_chain = create_stuff_documents_chain(llm, build_answer_prompt())
    return create_retrieval_chain(history_aware_retriever, question_answer_chain)


if "conversation" not in st.session_state:
    st.session_state.conversation = []  # list of {"question", "answer", "sources"} for display
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of HumanMessage/AIMessage for the chain


def generate_answer(question, container):
    if not question:
        return

    trimmed_history = st.session_state.chat_history[-(HISTORY_TURNS_KEPT * 2):]

    callback = StreamlitCallbackHandler(container)
    response = chain.invoke(
        {"input": question, "chat_history": trimmed_history},
        config={"callbacks": [callback]},
    )
    answer_text = response["answer"]
    sources = response.get("context", [])

    st.session_state.conversation.append(
        {"question": question, "answer": answer_text, "sources": sources}
    )
    st.session_state.chat_history.append(HumanMessage(content=question))
    st.session_state.chat_history.append(AIMessage(content=answer_text))


def display_answer():
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
                        page = doc.metadata.get("page")
                        name = Path(doc.metadata.get("source", "unknown")).name
                        key = (name, page)
                        if key in seen:
                            continue
                        seen.add(key)
                        st.markdown(f"**{name} — page {page}**")
                        snippet = doc.page_content[:300]
                        if len(doc.page_content) > 300:
                            snippet += "…"
                        st.caption(snippet)


with st.sidebar:
    st.subheader("Knowledge base")

    uploaded_files = st.file_uploader("Add PDFs to the index", type="pdf", accept_multiple_files=True)
    if uploaded_files and st.button("Ingest uploaded PDF(s)"):
        with st.spinner("Reading and indexing PDF(s)... this can take a while on a large file."):
            tmp_dir = Path(tempfile.mkdtemp())
            saved_paths = []
            for f in uploaded_files:
                dest = tmp_dir / f.name
                dest.write_bytes(f.getvalue())
                saved_paths.append(dest)
            stats = ingest_pdfs(saved_paths)
        st.success(f"Indexed {stats['pages']} page(s) into {stats['chunks']} chunk(s).")
        st.cache_resource.clear()  # drop cached retrievers/chain so the new chunks are picked up
        st.rerun()

    if st.session_state.conversation and st.button("Clear conversation"):
        st.session_state.conversation = []
        st.session_state.chat_history = []
        st.rerun()

if not vectorstore_ready():
    st.info("No documents indexed yet. Upload a PDF in the sidebar to get started.")
else:
    chain = chains_of_qa()

    display_answer()

    user_query = st.chat_input("Ask me anything: ")

    if user_query:
        with st.chat_message("Human"):
            st.write(user_query)
        with st.chat_message("AI"):
            callback_container = st.container()
            with st.spinner("Generating answer..."):
                generate_answer(user_query, callback_container)
        # Rerun so display_answer() renders the new turn as a normal chat
        # bubble (with its sources expander) instead of leaving it stuck
        # inside the collapsed "Thinking..." trace above.
        st.rerun()

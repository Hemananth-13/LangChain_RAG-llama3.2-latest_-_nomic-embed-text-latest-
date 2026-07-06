# app.py — Line-by-Line Explanation

`app.py` is the main application file. It defines the Streamlit web UI, manages session state, builds the LangChain RAG chain, and handles all user interactions. It is the entry point that users interact with directly.

---

## Imports

```python
import json
import os
import tempfile
from pathlib import Path
```
Standard library imports:
- `json` — for serialising/deserialising chat history to/from disk
- `os` — used to call `os.cpu_count()` when configuring the LLM thread count
- `tempfile` — creates a temporary directory to hold uploaded PDFs before ingestion
- `Path` — filesystem path handling

```python
import streamlit as st
```
The Streamlit framework. `st` is the conventional alias. Every UI element (buttons, text inputs, sidebars, chat messages) is created by calling functions on this object.

```python
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import DocumentCompressorPipeline
```
LangChain chain-building utilities:
- `create_history_aware_retriever` — wraps a retriever so it first rewrites the user's question using chat history before searching
- `create_retrieval_chain` — combines a retriever and a document QA chain into a single end-to-end chain
- `create_stuff_documents_chain` — takes retrieved documents and "stuffs" them all into the LLM's context window as a single prompt
- `ContextualCompressionRetriever` — wraps a base retriever with a post-processing step (in this case, reordering)
- `DocumentCompressorPipeline` — chains multiple document transformers together

```python
from langchain_community.document_transformers import LongContextReorder
```
A transformer that reorders a list of retrieved documents so the most relevant ones appear at the beginning and end. This counteracts the "lost in the middle" problem — LLMs tend to pay less attention to content in the middle of a long context window.

```python
from langchain_community.callbacks import StreamlitCallbackHandler
```
A LangChain callback that streams intermediate chain steps (like "searching documents…", "generating answer…") into a Streamlit container in real time, giving the user visual feedback during processing.

```python
from langchain_core.messages import AIMessage, HumanMessage
```
LangChain's message types for representing chat history. `HumanMessage` wraps user input; `AIMessage` wraps model responses. These are the objects that get passed into the chain's `chat_history` slot.

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
```
- `ChatPromptTemplate` — builds a structured prompt from a list of (role, content) tuples
- `MessagesPlaceholder` — a slot in the prompt template that gets filled with a list of messages at runtime (used for injecting chat history)

```python
from langchain_ollama import ChatOllama, OllamaEmbeddings
```
LangChain's Ollama integrations:
- `ChatOllama` — the LLM client; sends prompts to the local Ollama process and streams back responses
- `OllamaEmbeddings` — the embedding client; sends text to Ollama and receives back float vectors

```python
from langchain_chroma import Chroma
```
LangChain's ChromaDB integration. Used here to open an existing persisted vector store for querying (ingestion also uses this, but from `ingest.py`).

```python
from config import (
    LLM_MODEL, EMBEDDING_MODEL, RETRIEVER_K,
    HISTORY_TURNS_KEPT, LLM_PARAMS, CHAT_HISTORY_DIR,
)
from ingest import ingest_pdfs, list_indexed_files, delete_indexed_file, persist_dir_for
```
Imports all settings from `config.py` and the four public functions from `ingest.py` that the UI needs.

---

## Page Configuration

```python
st.set_page_config(page_title="Ask me anything", page_icon="📄")
```
Must be the first Streamlit call in the script. Sets the browser tab title and favicon. If this is called after any other `st.*` call, Streamlit raises an error.

---

## Chat History Persistence

These four functions handle saving and loading conversation history to/from JSON files on disk. This means conversations survive app restarts.

### `_history_path(collection: str) -> Path`

```python
def _history_path(collection: str) -> Path:
    CHAT_HISTORY_DIR.mkdir(exist_ok=True)
    return CHAT_HISTORY_DIR / f"{collection}.json"
```
Returns the path to the JSON file for a given collection. Also creates the `chat_history/` directory if it doesn't exist yet (`exist_ok=True` prevents an error if it already exists). Called by every other history function.

---

### `load_chat_history(collection: str) -> tuple[list, list]`

```python
def load_chat_history(collection: str) -> tuple[list, list]:
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
```
Reads the JSON file and reconstructs two separate data structures:

1. `conversation` — a list of dicts with keys `question`, `answer`, and `sources`. This is what the UI renders (the visible chat bubbles).
2. `chat_history` — a list of `HumanMessage` / `AIMessage` objects. This is what gets passed into the LangChain chain to provide context for follow-up questions.

The JSON file stores messages as plain dicts (`{"type": "human", "content": "..."}`) because LangChain message objects aren't directly JSON-serialisable. This function reconstructs the proper objects on load.

---

### `save_chat_history(collection, conversation, chat_history) -> None`

```python
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
```
Serialises both data structures to a single JSON file. The `chat_history` list of LangChain message objects is converted to plain dicts first. `ensure_ascii=False` preserves non-ASCII characters (e.g. accented letters, CJK characters). `indent=2` makes the file human-readable.

---

### `delete_chat_history(collection: str) -> None`

```python
def delete_chat_history(collection: str) -> None:
    path = _history_path(collection)
    if path.exists():
        path.unlink()
```
Deletes the JSON file for a collection. Called when the user clicks the 🗑 button in the sidebar. `path.unlink()` is Python's way of deleting a file.

---

## Cached Resource Loaders

These four functions are decorated with `@st.cache_resource()`, which means Streamlit calls them only once per unique set of arguments and caches the result in memory for the lifetime of the server process. This prevents the LLM, embedding model, vector store, and chain from being re-initialised on every page interaction.

### `load_llm()`

```python
@st.cache_resource()
def load_llm():
    return ChatOllama(
        model=LLM_MODEL,
        verbose=True,
        num_thread=os.cpu_count() or 8,
        **LLM_PARAMS,
    )
```
Creates the LLM client. `num_thread` is set to the number of logical CPU cores on the machine (`os.cpu_count()`), with a fallback of 8 if the count can't be determined. `**LLM_PARAMS` unpacks all the generation parameters from `config.py` (temperature, context window size, etc.).

---

### `load_embedding_model()`

```python
@st.cache_resource()
def load_embedding_model():
    return OllamaEmbeddings(model=EMBEDDING_MODEL)
```
Creates the embedding model client. Cached globally (no arguments) because the same embedding model is used for all collections.

---

### `load_vectorstore(collection: str)`

```python
@st.cache_resource()
def load_vectorstore(collection: str):
    return Chroma(
        persist_directory=persist_dir_for(collection),
        embedding_function=load_embedding_model(),
    )
```
Opens the ChromaDB vector store for a specific collection. Keyed by `collection` name, so each PDF gets its own cached store. `persist_dir_for(collection)` resolves to `VectorDB/<collection>/`. The `embedding_function` is needed so ChromaDB knows how to embed query strings at retrieval time.

---

### `load_chain(collection: str)`

This is the most complex function — it assembles the full RAG chain.

```python
@st.cache_resource()
def load_chain(collection: str):
    vectorstore = load_vectorstore(collection)
    base_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": RETRIEVER_K, "fetch_k": RETRIEVER_K * 3},
    )
```
Creates the base retriever using **MMR (Maximal Marginal Relevance)**. MMR balances relevance and diversity:
- `fetch_k = 12` — initially fetches 12 candidate chunks from the vector store
- `k = 4` — from those 12, selects the 4 that are most relevant AND most diverse from each other
- This prevents returning 4 near-identical chunks when the answer is spread across different parts of the document

```python
    pipeline_compressor = DocumentCompressorPipeline(transformers=[LongContextReorder()])
    retriever = ContextualCompressionRetriever(
        base_compressor=pipeline_compressor, base_retriever=base_retriever
    )
```
Wraps the base retriever with a post-processing pipeline. Currently the pipeline contains only `LongContextReorder`, which reorders the 4 retrieved chunks so the most relevant ones are at the start and end (not buried in the middle). The `DocumentCompressorPipeline` makes it easy to add more transformers later (e.g. a relevance filter).

```python
    llm = load_llm()

    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Given the chat history and the latest user question, rewrite it as a "
         "standalone question that can be understood without the chat history. "
         "Do NOT answer the question. If it's already standalone, return it as-is."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
```
The **contextualisation prompt** is used by the history-aware retriever. Before searching the vector store, the LLM is asked to rewrite the user's question into a standalone form. For example:
- User asks: "When was it founded?"
- With history showing the previous topic was KTM motorcycles
- The LLM rewrites it to: "When was KTM founded?"
- This standalone question is then used for vector search

```python
    answer_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an assistant for question-answering tasks. Use the retrieved "
         "context below to answer the latest user question. If you don't know "
         "the answer, say you don't know, do NOT make one up.\n\nContext:\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
```
The **answer prompt** is used for the final answer generation step. It instructs the LLM to:
- Answer based only on the provided context
- Admit when it doesn't know rather than hallucinating
- The `{context}` placeholder is filled with the retrieved chunks
- The `MessagesPlaceholder("chat_history")` is filled with recent conversation turns

```python
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_prompt)
    question_answer_chain = create_stuff_documents_chain(llm, answer_prompt)
    return create_retrieval_chain(history_aware_retriever, question_answer_chain)
```
Assembles the final chain:
1. `history_aware_retriever` — LLM + base retriever + contextualisation prompt
2. `question_answer_chain` — LLM + answer prompt (receives retrieved docs as `{context}`)
3. `create_retrieval_chain` — connects them: retriever output feeds into the QA chain

---

## Session State Bootstrap

```python
if "active_collection" not in st.session_state:
    st.session_state.active_collection = None
if "conversation" not in st.session_state:
    st.session_state.conversation = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
```
Streamlit re-runs the entire script from top to bottom on every user interaction. `st.session_state` is a dictionary that persists across these re-runs within a single browser session. These three lines initialise the state keys only if they don't already exist (the `not in` guard prevents resetting them on every re-run).

- `active_collection` — the name of the currently selected PDF collection
- `conversation` — list of `{question, answer, sources}` dicts for display
- `chat_history` — list of `HumanMessage`/`AIMessage` objects for the chain

---

### `switch_collection(collection: str) -> None`

```python
def switch_collection(collection: str) -> None:
    current = st.session_state.active_collection
    if current:
        save_chat_history(current, st.session_state.conversation, st.session_state.chat_history)
    st.session_state.active_collection = collection
    conv, hist = load_chat_history(collection)
    st.session_state.conversation = conv
    st.session_state.chat_history = hist
```
Called when the user clicks a different document in the sidebar. It:
1. Saves the current conversation to disk before switching (so it's not lost)
2. Updates `active_collection` to the new selection
3. Loads the new collection's saved conversation from disk into session state

---

## UI Helper Functions

### `generate_answer(question, chain, container) -> None`

```python
def generate_answer(question: str, chain, container) -> None:
    trimmed_history = st.session_state.chat_history[-(HISTORY_TURNS_KEPT * 2):]
```
Slices the chat history to keep only the last `HISTORY_TURNS_KEPT * 2` messages (e.g. 3 turns × 2 messages/turn = 6 messages). The negative index `[-(6):]` means "last 6 items". This prevents the context window from overflowing on long conversations.

```python
    callback = StreamlitCallbackHandler(container)
    try:
        response = chain.invoke(
            {"input": question, "chat_history": trimmed_history},
            config={"callbacks": [callback]},
        )
    except Exception as e:
        st.error(f"Failed to generate an answer: {e}")
        return
```
Invokes the chain with the user's question and trimmed history. The `StreamlitCallbackHandler` streams intermediate steps into the `container` element in real time. If anything fails (Ollama not running, model not pulled, etc.), the error is displayed in the UI rather than crashing the app.

```python
    answer_text = response["answer"]
    sources = [
        {"page_content": d.page_content, "metadata": d.metadata}
        for d in response.get("context", [])
    ]
```
Extracts the answer text and the source documents from the chain's response dict. The source `Document` objects are converted to plain dicts (`page_content` + `metadata`) so they can be JSON-serialised and saved to disk. The `metadata` dict contains `source` (file path) and `page` (page number).

```python
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
```
Updates both session state lists and immediately persists them to disk. The conversation is updated for display; the chat history is updated for the next chain invocation.

---

### `display_conversation() -> None`

```python
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
```
Renders the full conversation history. For each entry:
- Displays the question in a "Human" chat bubble
- Displays the answer in an "AI" chat bubble
- If sources exist, shows a collapsible expander with deduplicated source citations

The `seen` set deduplicates sources — if two chunks came from the same page of the same file, only one citation is shown. Each citation shows the filename, page number, and a 300-character preview of the chunk text.

---

## Sidebar

```python
with st.sidebar:
```
Everything indented under this context manager appears in the left sidebar panel.

### Upload & Ingest

```python
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
```
The file uploader widget accepts multiple PDFs. When the button is clicked:
1. A temporary directory is created to hold the uploaded files (Streamlit provides files as in-memory bytes, not disk paths)
2. Each file is written to disk in the temp directory
3. `ingest_pdfs()` is called for each file
4. After all files are ingested, the cached vectorstore and chain are cleared (`.clear()`) so they'll be rebuilt with the new data on the next access
5. `st.rerun()` triggers a full page re-run so the new document appears in the sidebar list

### Document Selector

```python
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
                ...
                delete_indexed_file(collection)
                delete_chat_history(collection)
                load_vectorstore.clear()
                load_chain.clear()
                st.rerun()
```
Renders one row per indexed document. Each row has:
- A wide button (4/5 of the width) showing the collection name with a `▶` prefix if it's currently active
- A narrow trash button (1/5 of the width) to delete the collection and its history

Each button has a unique `key` (required by Streamlit when creating multiple buttons in a loop).

---

## Main Area

```python
indexed = list_indexed_files()

if not indexed:
    st.info("No documents indexed yet. Upload a PDF in the sidebar to get started.")
else:
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
```
The main content area logic:
1. If no documents are indexed, shows an info message
2. If the active collection is no longer valid (e.g. was just deleted), auto-selects the first available document
3. Displays the active document name as a header
4. Loads the RAG chain for the active collection (from cache if available)
5. Renders the conversation history
6. Shows the chat input box at the bottom
7. When the user submits a question: immediately shows their message, then generates and displays the answer, then calls `st.rerun()` to refresh the full page (which re-renders the conversation cleanly)

---

## Streamlit Re-run Model

Understanding Streamlit's execution model is key to understanding `app.py`:

```
User action (button click, text input, etc.)
         │
         ▼
Streamlit re-runs app.py from top to bottom
         │
         ▼
Session state (st.session_state) persists between re-runs
         │
         ▼
@st.cache_resource functions return cached objects (not re-created)
         │
         ▼
UI is re-rendered based on current state
```

Every time the user does anything, the entire script runs again. This is why:
- Session state guards (`if "key" not in st.session_state`) are needed to avoid resetting values
- `@st.cache_resource` is critical — without it, the LLM and chain would be re-created on every keystroke
- `st.rerun()` is called explicitly after state-changing operations to force an immediate re-render

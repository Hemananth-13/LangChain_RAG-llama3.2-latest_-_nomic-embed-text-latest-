# Operational Mechanics — End-to-End Walkthrough

This document traces exactly what happens at every step of the system's operation, from startup to answering a question. It is intended to give a complete, concrete picture of how all components interact.

---

## 1. Starting the Application

### What you run
```bash
python run.py
# or
streamlit run app.py
```

### What happens

**`run.py` execution:**
```python
app = Path(__file__).parent / "app.py"
cmd = [sys.executable, "-m", "streamlit", "run", str(app)]
subprocess.run(cmd, check=True)
```
`run.py` is a thin wrapper. It constructs the equivalent of `python -m streamlit run app.py` using the current Python interpreter path, then hands off to Streamlit. The only reason it exists is convenience.

**Streamlit startup:**
1. Streamlit starts an HTTP server on port 8501
2. It executes `app.py` from top to bottom for the first time
3. The browser opens at `http://localhost:8501`

**First execution of `app.py`:**
- All imports are resolved
- `st.set_page_config()` sets the browser tab title
- Session state is initialised: `active_collection = None`, `conversation = []`, `chat_history = []`
- The sidebar is rendered
- `list_indexed_files()` scans `VectorDB/` for existing collections
- If no collections exist: the info message is shown
- If collections exist: `switch_collection(indexed[0])` auto-selects the first one

---

## 2. Uploading and Ingesting a PDF

### User action
User drags a PDF into the file uploader widget and clicks "Ingest uploaded PDF(s)".

### Step-by-step execution

**Step 1 — File is received by Streamlit**

The uploaded file is held in memory as an `UploadedFile` object with a `.name` attribute and `.getvalue()` method returning raw bytes.

**Step 2 — Temporary file is written to disk**
```python
tmp_dir = Path(tempfile.mkdtemp())   # e.g. C:\Users\...\Temp\tmpXXXXXX
dest = tmp_dir / f.name              # e.g. .../tmpXXXXXX/KTM - Wikipedia.pdf
dest.write_bytes(f.getvalue())       # writes the PDF bytes to disk
```
This is necessary because `PyPDFLoader` requires a file path, not in-memory bytes.

**Step 3 — `ingest_pdfs([dest])` is called**

3a. `load_pdfs([dest])` runs:
- `PyPDFLoader("KTM - Wikipedia.pdf").load()` extracts text from every page
- Each page becomes a `Document(page_content="...", metadata={"source": "...", "page": 0})`
- `_clean_text()` normalises whitespace on each page

3b. `persist_directory` is derived:
- `collection_name_for("KTM - Wikipedia.pdf")` produces `"KTM___Wikipedia"`
- `persist_dir_for("KTM - Wikipedia.pdf")` produces `"VectorDB/KTM___Wikipedia"`

3c. `RecursiveCharacterTextSplitter` splits pages into chunks:
- Each page's text is split into ~1000-character pieces with 150-character overlap
- Metadata (`source`, `page`) is copied from the parent page to every chunk derived from it

3d. Deterministic IDs are generated for every chunk:
- `_make_id("KTM - Wikipedia.pdf", 3, 1, "...text...")` produces `"KTM - Wikipedia-p3-c1-a4f2b8c1"`

3e. `Chroma(persist_directory="VectorDB/KTM___Wikipedia", embedding_function=...)` opens or creates the store

3f. `vectorstore.add_documents(chunks, ids=ids)` runs:
- For each chunk, Ollama's `nomic-embed-text` converts the text to a 768-dimensional float vector
- The vector, text, and metadata are stored in ChromaDB's SQLite database on disk
- If a chunk ID already exists it is updated (upsert); otherwise it is inserted

**Step 4 — Cache is cleared and page re-runs**
```python
load_vectorstore.clear()
load_chain.clear()
st.rerun()
```
On the re-run, `list_indexed_files()` finds `VectorDB/KTM___Wikipedia/` and the new document appears in the sidebar.

---

## 3. Selecting a Document

### User action
User clicks the "KTM___Wikipedia" button in the sidebar.

### What happens

```python
switch_collection("KTM___Wikipedia")
st.rerun()
```

Inside `switch_collection`:
1. The current collection's conversation is saved to `chat_history/<current>.json`
2. `active_collection` is updated to `"KTM___Wikipedia"`
3. `load_chat_history("KTM___Wikipedia")` reads `chat_history/KTM___Wikipedia.json`
   - First time: file doesn't exist, returns `[], []`
   - Subsequent times: deserialises the conversation and reconstructs `HumanMessage`/`AIMessage` objects
4. Session state is updated with the loaded history

On the re-run:
- The sidebar re-renders with `▶ KTM___Wikipedia` showing the active indicator
- `load_chain("KTM___Wikipedia")` returns the cached chain or builds it fresh
- `display_conversation()` renders any previously saved messages

---

## 4. Asking a Question

### User action
User types "When was KTM founded?" and presses Enter.

### Step-by-step execution

**Step 1 — Input is captured**
```python
user_query = st.chat_input("Ask me anything: ")
# user_query = "When was KTM founded?"
```

**Step 2 — Human message is displayed immediately**
```python
with st.chat_message("Human"):
    st.write(user_query)
```
The user's message appears before the answer is generated, providing instant feedback.

**Step 3 — `generate_answer()` is called**

History is trimmed:
```python
trimmed_history = st.session_state.chat_history[-(HISTORY_TURNS_KEPT * 2):]
# = last 6 messages (3 turns x 2 messages each)
# On first question: trimmed_history = []
```

The chain is invoked:
```python
response = chain.invoke(
    {"input": "When was KTM founded?", "chat_history": []},
    config={"callbacks": [callback]},
)
```

**Step 4 — Inside the chain**

**4a. History-aware retriever runs first**

Since `chat_history` is empty (first question), the contextualisation step is skipped and the original question is used as-is.

If this were a follow-up (e.g. "What about their racing history?"), the LLM would first rewrite it using the contextualisation prompt into a standalone question like "What is KTM's racing history?" before searching.

**4b. Vector search runs**

The question is embedded by `nomic-embed-text`:
```
"When was KTM founded?" -> [0.023, -0.147, 0.891, ..., 0.034]  (768 floats)
```

ChromaDB performs MMR search:
- Fetches 12 candidate chunks by cosine similarity (`fetch_k = RETRIEVER_K * 3`)
- Selects the 4 that maximise both relevance and diversity (`k = RETRIEVER_K`)
- Returns 4 `Document` objects with text and metadata

**4c. LongContextReorder runs**

The 4 chunks are reordered so the most relevant ones sit at positions 0 and 3 (start and end). LLMs have weaker recall for content in the middle of a long context window, so the most important chunks are placed at the extremes.

**4d. Answer generation runs**

The answer prompt is assembled and sent to `llama3.2:latest`:

```
System: You are an assistant for question-answering tasks. Use the retrieved
        context below to answer the latest user question. If you don't know
        the answer, say you don't know, do NOT make one up.

        Context:
        [Chunk 1 text - most relevant]
        [Chunk 2 text]
        [Chunk 3 text]
        [Chunk 4 text - second most relevant]

Human: When was KTM founded?

The LLM reads the context and generates a grounded answer. With temperature=0.01
the output is near-deterministic — the same question asked twice will produce
nearly identical answers.

**Step 5 — Response is processed and saved**

```python
answer_text = response["answer"]
sources = [
    {"page_content": d.page_content, "metadata": d.metadata}
    for d in response.get("context", [])
]
```

The answer string and source documents are extracted. Source `Document` objects
are converted to plain dicts so they can be JSON-serialised.

```python
st.session_state.conversation.append(
    {"question": question, "answer": answer_text, "sources": sources}
)
st.session_state.chat_history.append(HumanMessage(content=question))
st.session_state.chat_history.append(AIMessage(content=answer_text))
save_chat_history(...)
```

Both session state lists are updated and immediately written to
`chat_history/KTM___Wikipedia.json`.

**Step 6 — Page re-runs**

`st.rerun()` triggers a full re-run of `app.py`. `display_conversation()` now
renders the new entry (question + answer + sources expander) as part of the
complete conversation history.

---

## 5. Asking a Follow-up Question

### User action
User types "What racing championships have they won?" — a question that only
makes sense in the context of the previous answer about KTM.

### What is different

This time `trimmed_history` is not empty:
```python
trimmed_history = [
    HumanMessage(content="When was KTM founded?"),
    AIMessage(content="KTM was founded in 1934..."),
]
```

The history-aware retriever now calls the LLM with the contextualisation prompt:
```
System: Given the chat history and the latest user question, rewrite it as a
        standalone question...

Human (history): When was KTM founded?
AI (history):    KTM was founded in 1934...
Human (now):     What racing championships have they won?
```

The LLM rewrites this to: `"What racing championships has KTM won?"`

This standalone question is then used for vector search, ensuring the retrieval
step finds relevant chunks even though the original question used the pronoun
"they" with no explicit subject.

---

## 6. Deleting a Document

### User action
User clicks the trash icon next to a document in the sidebar.

### What happens

```python
if st.session_state.active_collection == collection:
    st.session_state.active_collection = None
    st.session_state.conversation = []
    st.session_state.chat_history = []
delete_indexed_file(collection)   # shutil.rmtree(VectorDB/<collection>)
delete_chat_history(collection)   # unlinks chat_history/<collection>.json
load_vectorstore.clear()
load_chain.clear()
st.rerun()
```

If the deleted document was the active one, session state is reset to avoid
referencing a now-nonexistent collection. Both the vector store directory and
the chat history JSON file are permanently deleted from disk.

---

## 7. Restarting the App

Because all state is persisted to disk, restarting the app is seamless:

- `VectorDB/<collection>/` — ChromaDB data survives; no re-ingestion needed
- `chat_history/<collection>.json` — conversation history survives; previous
  Q&A is restored when the collection is selected
- `@st.cache_resource` objects — rebuilt in memory on first access after restart

The only thing lost on restart is the in-memory Streamlit session state, which
is immediately restored from disk by `switch_collection()` on the first re-run.

---

## 8. Complete Data Flow Reference

```
INGESTION
---------
PDF file
  -> PyPDFLoader          (text extraction, one Document per page)
  -> _clean_text()        (whitespace normalisation)
  -> RecursiveCharacterTextSplitter  (chunking, ~1000 chars, 150 overlap)
  -> _make_id()           (deterministic SHA-256 chunk IDs)
  -> OllamaEmbeddings     (nomic-embed-text: text -> 768-dim vector)
  -> Chroma.add_documents (upsert vectors + text + metadata to SQLite)
  -> VectorDB/<name>/     (persisted on disk)

QUERY
-----
User question + chat_history
  -> History-aware retriever
       -> [if follow-up] LLM rewrites question to standalone form
       -> OllamaEmbeddings  (embed the standalone question)
       -> Chroma MMR search (fetch 12, return 4 diverse relevant chunks)
       -> LongContextReorder (reorder: best chunks at start and end)
  -> create_stuff_documents_chain
       -> answer_prompt filled with context chunks + history + question
       -> ChatOllama (llama3.2: generate answer, max 500 tokens)
  -> response["answer"]   (answer text)
  -> response["context"]  (source Document objects)
  -> save to session_state + chat_history/<name>.json
  -> display in UI with source citations
```

---

## 9. Key Design Decisions Explained

**Why one ChromaDB collection per PDF?**
Isolating each PDF in its own sub-directory means you can delete a single
document without affecting others, and the active document's vector store is
loaded independently. It also makes the sidebar document list trivial to
implement — just list the sub-directories.

**Why deterministic chunk IDs?**
If you re-upload the same PDF (e.g. after editing it), the ingestion pipeline
will upsert changed chunks and leave unchanged ones intact, rather than
creating duplicate entries. This keeps the index clean without requiring a
delete-and-rebuild step.

**Why serialize sources as dicts instead of Document objects?**
LangChain `Document` objects are not JSON-serialisable. Storing them as plain
dicts (`{"page_content": ..., "metadata": ...}`) allows the full conversation
including source citations to be saved to disk and restored on app restart.

**Why trim chat history to the last N turns?**
The LLM has a fixed context window (16,000 tokens). Including the entire
conversation history would eventually overflow it. Keeping only the last 3
turns (6 messages) provides enough context for coherent follow-up questions
while leaving room for the retrieved chunks and the answer.

**Why MMR instead of plain similarity search?**
Plain cosine similarity often returns near-duplicate chunks (e.g. the same
sentence appearing in slightly different contexts). MMR penalises chunks that
are too similar to already-selected ones, ensuring the 4 returned chunks cover
different aspects of the answer rather than repeating the same information.

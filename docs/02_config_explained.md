# config.py — Line-by-Line Explanation

`config.py` is the single source of truth for every tunable parameter in the project. All other files import from here — nothing is hardcoded elsewhere.

---

## Full File

```python
from pathlib import Path
```
Imports Python's `Path` class, which provides an object-oriented way to work with filesystem paths. It handles OS differences (forward vs back slashes) automatically.

---

```python
BASE_DIR = Path(__file__).parent
```
`__file__` is a Python built-in that holds the absolute path of the current script. `.parent` navigates one level up to the directory containing it. So `BASE_DIR` always points to the project root, regardless of where you run the app from.

---

```python
VECTORDB_DIR = BASE_DIR / "VectorDB"
```
Defines the root directory for all ChromaDB data. The `/` operator on `Path` objects is overloaded to mean path joining — equivalent to `os.path.join(BASE_DIR, "VectorDB")`. This directory is created automatically on first ingest; you never need to create it manually.

---

```python
PERSIST_DIRECTORY = str(VECTORDB_DIR)
```
A string version of `VECTORDB_DIR`. Kept for backwards compatibility with the CLI (`ingest.py --persist-directory`), which was written before the per-PDF sub-directory system was introduced. New code uses `persist_dir_for()` from `ingest.py` instead.

---

```python
CHAT_HISTORY_DIR = BASE_DIR / "chat_history"
```
Directory where per-document conversation history is saved as JSON files. Each PDF gets its own file: `chat_history/<collection-name>.json`. This directory is created on demand by `_history_path()` in `app.py`.

---

```python
LLM_MODEL = "llama3.2:latest"
```
The Ollama model tag for the language model used to generate answers. `llama3.2:latest` is Meta's 3.2B parameter model — fast enough to run on a CPU while still producing quality answers. Change this to any model you have pulled locally (e.g. `"mistral:latest"`, `"llama3.1:8b"`).

---

```python
EMBEDDING_MODEL = "nomic-embed-text:latest"
```
The Ollama model tag for the embedding model. Embeddings convert text into numerical vectors so that semantic similarity can be computed. `nomic-embed-text` produces 768-dimensional vectors and is optimised for retrieval tasks. This must match the model used when the VectorDB was originally built — changing it after ingestion will produce incompatible vectors.

---

```python
DEFAULT_CHUNK_SIZE = 1000
```
The maximum number of characters per text chunk during ingestion. Smaller chunks = more precise retrieval but less context per chunk. Larger chunks = more context but potentially noisier retrieval. 1000 characters is roughly 150–200 words, which fits comfortably within the embedding model's context window.

---

```python
DEFAULT_CHUNK_OVERLAP = 150
```
The number of characters that adjacent chunks share. Overlap prevents important information from being split across a chunk boundary and lost. For example, if a sentence starts at character 990 of a chunk, it will also appear at the start of the next chunk. 150 characters is ~15% of the chunk size, which is a standard ratio.

---

```python
RETRIEVER_K = 4
```
How many chunks to return per query. The retriever fetches `RETRIEVER_K * 3 = 12` candidates using MMR, then returns the best 4. Increasing this gives the LLM more context but also increases the chance of including irrelevant content and slows down inference.

---

```python
HISTORY_TURNS_KEPT = 3
```
How many prior question-answer pairs are included in the LLM's context window. Each "turn" = 1 human message + 1 AI message, so `3` turns = 6 messages. Keeping the full history would eventually overflow the context window; this sliding window keeps the conversation grounded without growing unboundedly.

---

```python
LLM_PARAMS = {
    "num_gpu": -1,
```
`-1` tells Ollama to use all available GPU layers. If no GPU is present, Ollama falls back to CPU automatically. Set to `0` to force CPU-only.

---

```python
    "num_ctx": 16000,
```
The context window size in tokens. This is the maximum total number of tokens the LLM can "see" at once — including the system prompt, chat history, retrieved chunks, and the user's question. 16,000 tokens is generous enough to hold several retrieved chunks plus a multi-turn conversation.

---

```python
    "num_predict": 500,
```
The maximum number of tokens the LLM will generate in its response. 500 tokens is roughly 375 words — enough for a detailed answer. Increase this if you need longer responses.

---

```python
    "num_batch": 8192,
```
The number of tokens processed in parallel during the prompt evaluation phase (before generation starts). Higher values use more memory but process the input faster. 8192 is a good default for machines with ≥16 GB RAM.

---

```python
    "temperature": 0.01,
```
Controls randomness in the LLM's output. `0.0` = fully deterministic (always picks the highest-probability token). `1.0` = highly random. `0.01` is near-deterministic, which is appropriate for a factual Q&A system where you want consistent, grounded answers rather than creative variation.

---

```python
    "top_p": 0.95,
```
Nucleus sampling threshold. The LLM only considers tokens whose cumulative probability mass reaches 95%. This filters out very low-probability (nonsensical) tokens while still allowing some variation. Works in conjunction with `temperature`.

---

```python
    "top_k": 40,
```
At each generation step, only the top 40 most probable tokens are considered. This is a hard cap that prevents the model from ever choosing very unlikely tokens, regardless of `temperature` or `top_p`.

---

```python
    "repeat_penalty": 1.2,
```
Penalises the model for repeating tokens it has already generated. `1.0` = no penalty. `1.2` = a 20% reduction in probability for any token that appeared earlier in the output. This prevents the model from getting stuck in repetitive loops.

---

## How Parameters Interact

```
User question
      │
      ▼
  Context window (num_ctx = 16000 tokens)
  ┌─────────────────────────────────────────┐
  │ System prompt          (~100 tokens)    │
  │ Chat history           (~300 tokens)    │  ← HISTORY_TURNS_KEPT controls size
  │ Retrieved chunks       (~800 tokens)    │  ← RETRIEVER_K controls count
  │ User question          (~50 tokens)     │
  └─────────────────────────────────────────┘
      │
      ▼
  LLM generates answer (up to num_predict = 500 tokens)
  with temperature=0.01, top_p=0.95, top_k=40, repeat_penalty=1.2
```

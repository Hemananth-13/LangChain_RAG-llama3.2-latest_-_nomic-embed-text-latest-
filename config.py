from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent
VECTORDB_DIR = BASE_DIR / "VectorDB"
PERSIST_DIRECTORY = str(VECTORDB_DIR)  # kept for CLI backwards-compat
CHAT_HISTORY_DIR = BASE_DIR / "chat_history"

# Models
LLM_MODEL = "llama3.2:latest"
EMBEDDING_MODEL = "nomic-embed-text:latest"

# Ingestion
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 150

# Retrieval
RETRIEVER_K = 4
HISTORY_TURNS_KEPT = 3

# LLM params
LLM_PARAMS = {
    "num_gpu": -1,
    "num_ctx": 16000,
    "num_predict": 500,
    "num_batch": 8192,
    "temperature": 0.01,
    "top_p": 0.95,
    "top_k": 40,
    "repeat_penalty": 1.2,
}

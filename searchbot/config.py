"""Central config for the search-bot RAG engine."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # project root
SEARCH_DIR = ROOT / "search"                           # /{search} folders live here
DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "searchbot.db"
LIBGEN_BIN = os.environ.get("SEARCHBOT_LIBGEN_BIN", str(ROOT / "bin" / "libgen-mcp"))
LIBGEN_DOWNLOADS = DATA_DIR / "libgen_downloads"

CHAT_URL = os.environ.get("SEARCHBOT_CHAT_URL", "http://127.0.0.1:8080/v1")
# When SEARCHBOT_CHAT_MODEL is unset, llm.py auto-detects the loaded model
# from /v1/models, so swapping models doesn't need config changes.
CHAT_MODEL = os.environ.get("SEARCHBOT_CHAT_MODEL", "")
EMBED_URL = os.environ.get("SEARCHBOT_EMBED_URL", "http://127.0.0.1:8082/v1")

# Retrieval / prompt budget
CHUNK_CHARS = 1100          # ~300 tokens per chunk
CHUNK_OVERLAP = 140
MAX_CHUNKS_PER_DOC = 40     # cap pathological giant docs
RECALL_K = 30               # per lane (vec / fts) before fusion
FINAL_K = 8                 # chunks in evidence block
RECENT_TURNS = 8            # raw turns re-injected each call
SUMMARY_TRIGGER = 12        # unsummarized turns before compaction
QUERY_TIMEOUT_S = 180

# embeddinggemma retrieval instruction prefix for QUERIES only
QUERY_INSTRUCT = ("Instruct: Given a scientific question, retrieve relevant "
                  "passages from research papers\nQuery: ")

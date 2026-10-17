"""
Engine Configuration Settings
=============================
Defines paths, model identifiers, Elasticsearch/Redis connection parameters,
and retrieval thresholds.
"""

import os
import ssl
from pathlib import Path
import torch

try:
    ssl.create_default_context()
except Exception:
    try:
        import certifi
        _orig_create_default_context = ssl.create_default_context
        def _safe_create_default_context(*args, **kwargs):
            if "cafile" not in kwargs:
                kwargs["cafile"] = certifi.where()
            return _orig_create_default_context(*args, **kwargs)
        ssl.create_default_context = _safe_create_default_context
    except Exception:
        pass

# Locate workspace root dynamically
_curr = Path(__file__).resolve().parent
for _cand in [_curr, _curr.parent, _curr.parent.parent, _curr.parent.parent.parent]:
    if (_cand / "goodreads_books.json.gz").exists() or (_cand / "catalog_2m_embeddings_fp16.pt").exists() or (_cand / "authors.sqlite3").exists():
        BASE_DIR = _cand
        break
else:
    BASE_DIR = Path(__file__).resolve().parent.parent

# Database & Catalog paths
SQLITE_DB_PATH = BASE_DIR / "books_metadata.sqlite3"
GOODREADS_GZ_PATH = BASE_DIR / "goodreads_books.json.gz"
CATALOG_2M_EMB = BASE_DIR / "catalog_2m_embeddings_fp16.pt"
CATALOG_2M_IDS = BASE_DIR / "catalog_2m_ids.json"

CATALOG_EMB_CACHE = CATALOG_2M_EMB if (CATALOG_2M_EMB.exists() and CATALOG_2M_EMB.stat().st_size > 0) else (BASE_DIR / "catalog_bge_embeddings_fp16.pt")
CATALOG_IDS_CACHE = CATALOG_2M_IDS if (CATALOG_2M_IDS.exists() and CATALOG_2M_IDS.stat().st_size > 0) else (BASE_DIR / "catalog_ids.json")

# Models (Prefers local fine-tuned checkpoints, falls back to Hugging Face Hub)
LOCAL_EMBED_DIR = BASE_DIR / "bge_small_v5_output" / "best_model"
HF_EMBED_REPO = "kanishshandilya/book_recs_embed"
EMBED_MODEL_PATH = str(LOCAL_EMBED_DIR) if LOCAL_EMBED_DIR.exists() else HF_EMBED_REPO

LOCAL_RERANKER_DIR = BASE_DIR / "bge_reranker_base_output" / "best_model"
HF_RERANKER_REPO = "kanishshandilya/book_recs_reranker"
RERANKER_MODEL_PATH = str(LOCAL_RERANKER_DIR) if LOCAL_RERANKER_DIR.exists() else HF_RERANKER_REPO

# Hardware
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_FP16 = (DEVICE == "cuda")

# Retrieval Hyperparameters
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
MAX_SEQ_LENGTH_EMBED = 256
MAX_SEQ_LENGTH_RERANK = 384

DEFAULT_SEMANTIC_TOP_K = 50
DEFAULT_BM25_TOP_K = 50
DEFAULT_FINAL_TOP_K = 10

# Elasticsearch Settings
ES_HOST = os.getenv("ELASTICSEARCH_HOST", "http://localhost:9200")
ES_INDEX_NAME = os.getenv("ELASTICSEARCH_INDEX", "goodreads_books")
ES_TIMEOUT = int(os.getenv("ELASTICSEARCH_TIMEOUT", 10))

# Redis Settings
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", 3600))  # 1 hour default

"""
Book Recommendation & Semantic Search Engine Package
=====================================================
Exports:
  - HybridSearchPipeline
  - DenseRetriever
  - ESRetriever
  - RerankerEngine
  - CacheEngine
"""

# Windows SSL cert store patch (resolves [ASN1: NOT_ENOUGH_DATA] in aiohttp / elastic_transport)
import ssl
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

from .cache import CacheEngine
from .dense_retriever import DenseRetriever
from .es_retriever import ESRetriever
from .reranker import RerankerEngine
from .pipeline import HybridSearchPipeline

__all__ = [
    "HybridSearchPipeline",
    "DenseRetriever",
    "ESRetriever",
    "RerankerEngine",
    "CacheEngine",
]

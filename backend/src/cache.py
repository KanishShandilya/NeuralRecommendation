"""
Query & Metadata Cache Engine (Exact Redis + Semantic Vector Cache)
===================================================================
Provides high-speed caching using:
  1. Tier 1 (Exact Query Match via Redis / In-Memory LRU) -> <1ms
  2. Tier 2 (Semantic Vector Match via Cosine Similarity >= 0.95) -> ~3ms
"""

import hashlib
import json
import logging
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import torch

try:
    import redis
except ImportError:
    redis = None

from .config import CACHE_TTL_SECONDS, REDIS_URL

logger = logging.getLogger("BookSearch.Cache")


class InMemoryLRUCache:
    """Fallback in-memory LRU cache with TTL expiration."""
    def __init__(self, capacity: int = 2000):
        self.capacity = capacity
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        with self.lock:
            if key not in self.cache:
                return None
            item = self.cache[key]
            if time.time() > item["expires_at"]:
                del self.cache[key]
                return None
            return item["data"]

    def set(self, key: str, value: Any, ttl: int = CACHE_TTL_SECONDS):
        with self.lock:
            if len(self.cache) >= self.capacity:
                oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k]["created_at"])
                del self.cache[oldest_key]
            self.cache[key] = {
                "data": value,
                "created_at": time.time(),
                "expires_at": time.time() + ttl,
            }

    def clear(self):
        with self.lock:
            self.cache.clear()


class SemanticVectorCache:
    """Vector similarity cache for recognizing semantically equivalent queries."""
    def __init__(self, max_entries: int = 1000, default_threshold: float = 0.95):
        self.max_entries = max_entries
        self.threshold = default_threshold
        self.entries: List[Dict[str, Any]] = []
        self.lock = threading.Lock()

    def find_match(self, query_emb: torch.Tensor, threshold: Optional[float] = None) -> Optional[Tuple[Dict, float, str]]:
        thresh = threshold or self.threshold
        with self.lock:
            if not self.entries:
                return None

            q_vec = query_emb.detach().cpu().float()
            if q_vec.dim() == 1:
                q_vec = q_vec.unsqueeze(0)

            # Stack cached normalized query embeddings
            cached_vectors = torch.stack([e["emb"] for e in self.entries])
            sims = torch.matmul(q_vec, cached_vectors.T).squeeze(0)

            best_idx = int(torch.argmax(sims).item())
            best_sim = float(sims[best_idx].item())

            if best_sim >= thresh:
                matched = self.entries[best_idx]
                return matched["data"], best_sim, matched["query"]

            return None

    def store(self, query_emb: torch.Tensor, query_str: str, data: Any):
        with self.lock:
            if len(self.entries) >= self.max_entries:
                self.entries.pop(0)
            self.entries.append({
                "emb": query_emb.detach().cpu().float().squeeze(0),
                "query": query_str,
                "data": data,
            })


class CacheEngine:
    """Unified cache engine: Redis exact match + Semantic Vector similarity cache."""
    def __init__(self, redis_url: str = REDIS_URL, ttl: int = CACHE_TTL_SECONDS):
        self.ttl = ttl
        self.redis_client = None
        self.is_redis_active = False
        self.in_memory = InMemoryLRUCache(capacity=2000)
        self.semantic_cache = SemanticVectorCache(max_entries=1000, default_threshold=0.95)

        if redis is not None:
            try:
                client = redis.Redis.from_url(redis_url, socket_connect_timeout=1.5, socket_timeout=1.5, protocol=2)
                client.ping()
                self.redis_client = client
                self.is_redis_active = True
                logger.info(f"Connected to Redis at {redis_url}")
            except Exception as e:
                logger.warning(f"Redis unavailable ({e}). Falling back to in-memory LRU cache.")

    def make_cache_key(self, query: str, filters: Optional[Dict] = None, top_k: int = 10) -> str:
        payload = {
            "query": (query or "").lower().strip(),
            "filters": sorted((filters or {}).items()),
            "top_k": top_k,
        }
        raw_bytes = json.dumps(payload, sort_keys=True).encode("utf-8")
        return f"book_search:{hashlib.sha256(raw_bytes).hexdigest()}"

    def get_exact(self, key: str) -> Optional[Any]:
        if self.is_redis_active and self.redis_client:
            try:
                data = self.redis_client.get(key)
                if data:
                    return json.loads(data.decode("utf-8"))
            except Exception as e:
                logger.warning(f"Redis get failed: {e}. Checking in-memory.")

        return self.in_memory.get(key)

    def set_exact(self, key: str, value: Any, ttl: Optional[int] = None):
        ttl_seconds = ttl or self.ttl
        self.in_memory.set(key, value, ttl=ttl_seconds)

        if self.is_redis_active and self.redis_client:
            try:
                serialized = json.dumps(value, ensure_ascii=False)
                self.redis_client.setex(key, ttl_seconds, serialized)
            except Exception as e:
                logger.warning(f"Redis set failed: {e}")

    # Aliases
    get = get_exact
    set = set_exact

    def get_semantic(self, query_emb: torch.Tensor, threshold: float = 0.95) -> Optional[Tuple[Dict, float, str]]:
        return self.semantic_cache.find_match(query_emb, threshold=threshold)

    def set_semantic(self, query_emb: torch.Tensor, query_str: str, data: Any):
        self.semantic_cache.store(query_emb, query_str, data)

    def status(self) -> Dict[str, Any]:
        return {
            "backend": "redis" if self.is_redis_active else "in_memory_lru",
            "redis_connected": self.is_redis_active,
            "semantic_cached_queries": len(self.semantic_cache.entries),
            "default_ttl_seconds": self.ttl,
        }

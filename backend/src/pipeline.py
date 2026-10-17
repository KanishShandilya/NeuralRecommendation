"""
Two-Stage Hybrid Search Pipeline Engine
========================================
Orchestrates:
  1. Redis / In-Memory Cache Lookup
  2. Stage 1a (Dense Vector ANN Search) + Stage 1b (Elasticsearch BM25 Search)
  3. Reciprocal Rank Fusion (RRF) & Candidate Deduplication
  4. Stage 2 (Cross-Encoder Precision Reranking)
  5. Cache Update & Detailed Latency Profiling
"""

import json
import logging
import sqlite3
import time
from typing import Any, Dict, List, Optional

from .cache import CacheEngine
from .config import (
    DEFAULT_BM25_TOP_K,
    DEFAULT_FINAL_TOP_K,
    DEFAULT_SEMANTIC_TOP_K,
    SQLITE_DB_PATH,
)
from .dense_retriever import DenseRetriever
from .es_retriever import ESRetriever
from .reranker import RerankerEngine

logger = logging.getLogger("BookSearch.Pipeline")


class HybridSearchPipeline:
    def __init__(
        self,
        cache: Optional[CacheEngine] = None,
        dense: Optional[DenseRetriever] = None,
        es: Optional[ESRetriever] = None,
        reranker: Optional[RerankerEngine] = None,
    ):
        logger.info("Initializing Hybrid Search Pipeline...")
        self.cache = cache or CacheEngine()
        self.dense = dense or DenseRetriever()
        self.es = es or ESRetriever()
        self.reranker = reranker or RerankerEngine()

        self.sqlite_path = SQLITE_DB_PATH
        self._book_metadata_cache = {}
        if not self.es.is_es_active:
            self._load_sqlite_catalog_index()

    def search(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = DEFAULT_FINAL_TOP_K,
        semantic_top_k: int = DEFAULT_SEMANTIC_TOP_K,
        bm25_top_k: int = DEFAULT_BM25_TOP_K,
        use_cache: bool = True,
        use_reranker: bool = True,
    ) -> Dict[str, Any]:
        """Executes the full hybrid search & reranking workflow."""
        start_total = time.time()
        timings = {}

        # 1. Tier 1: Exact Cache Check (Redis)
        t0 = time.time()
        cache_key = self.cache.make_cache_key(query, filters, top_k)
        if use_cache:
            cached_result = self.cache.get(cache_key)
            if cached_result:
                timings["cache_lookup_ms"] = round((time.time() - t0) * 1000, 2)
                timings["total_ms"] = round((time.time() - start_total) * 1000, 2)
                cached_result["from_cache"] = True
                cached_result["cache_tier"] = "Tier 1 (Exact Match)"
                cached_result["latency"] = timings
                return cached_result
        timings["cache_lookup_ms"] = round((time.time() - t0) * 1000, 2)

        # 2. Stage 1a — Dense Vector Encoding
        t0 = time.time()
        q_emb = self.dense.encode_query(query)

        # Tier 2: Semantic Vector Cache Check (Cosine similarity >= 0.95)
        if use_cache and not filters:
            sem_match = self.cache.get_semantic(q_emb, threshold=0.95)
            if sem_match:
                cached_data, sim_score, matched_q = sem_match
                timings["cache_lookup_ms"] = round((time.time() - start_total) * 1000, 2)
                timings["total_ms"] = timings["cache_lookup_ms"]
                import copy
                res_copy = copy.deepcopy(cached_data)
                res_copy["from_cache"] = True
                res_copy["cache_tier"] = f"Tier 2 (Semantic Match: {sim_score:.3f} to '{matched_q}')"
                res_copy["latency"] = timings
                return res_copy

        dense_results = self.dense.search_vector(q_emb, top_k=semantic_top_k)
        timings["dense_search_ms"] = round((time.time() - t0) * 1000, 2)

        # 3. Stage 1b — BM25 Keyword Retrieval
        t0 = time.time()
        bm25_results = self.es.search(query, filters=filters, top_k=bm25_top_k)
        timings["bm25_search_ms"] = round((time.time() - t0) * 1000, 2)

        # 4. Fusion via Reciprocal Rank Fusion (RRF)
        t0 = time.time()
        rrf_scores = {}
        merged_candidates = {}
        k_rrf = 60.0

        for item in dense_results:
            bid = item["book_id"]
            rank = item["dense_rank"]
            rrf_scores[bid] = rrf_scores.get(bid, 0.0) + (1.0 / (k_rrf + rank))
            merged_candidates[bid] = {
                "book_id": bid,
                "dense_score": item["dense_score"],
                "dense_rank": rank,
            }

        for item in bm25_results:
            bid = item["book_id"]
            rank = item["bm25_rank"]
            rrf_scores[bid] = rrf_scores.get(bid, 0.0) + (1.0 / (k_rrf + rank))
            if bid not in merged_candidates:
                merged_candidates[bid] = {"book_id": bid}
            merged_candidates[bid]["bm25_score"] = item["bm25_score"]
            merged_candidates[bid]["bm25_rank"] = rank

        # Hydrate metadata from Elasticsearch (or SQLite fallback)
        bids_to_fetch = list(merged_candidates.keys())
        es_meta = {}
        if self.es.is_es_active and self.es.es_client:
            try:
                mget_res = self.es.es_client.mget(index=self.es.index_name, body={"ids": bids_to_fetch})
                for doc in mget_res.get("docs", []):
                    if doc.get("found"):
                        es_meta[doc["_id"]] = doc.get("_source", {})
            except Exception as e:
                logger.error(f"Failed to mget from ES: {e}")

        hydrated_list = []
        for bid, rrf_sc in rrf_scores.items():
            cand = merged_candidates[bid]
            cand["rrf_score"] = rrf_sc

            meta = es_meta.get(bid) or self._book_metadata_cache.get(bid, {})
            cand["title"] = meta.get("title", "")
            cand["authors"] = meta.get("authors", [])
            cand["popular_shelves"] = meta.get("popular_shelves", [])
            cand["description"] = meta.get("description", "")
            cand["image_url"] = meta.get("image_url", "")
            cand["average_rating"] = meta.get("average_rating", 0.0)
            hydrated_list.append(cand)

        # Sort by RRF score descending
        hydrated_list.sort(key=lambda x: x["rrf_score"], reverse=True)
        timings["fusion_ms"] = round((time.time() - t0) * 1000, 2)

        # 5. Stage 2 — Cross-Encoder Reranking
        t0 = time.time()
        if use_reranker and self.reranker is not None:
            # Rerank the top-K merged candidates (e.g. up to 70 candidates)
            final_results = self.reranker.rerank(query, hydrated_list[:70], top_k=top_k)
        else:
            final_results = hydrated_list[:top_k]
            for i, item in enumerate(final_results, start=1):
                item["final_rank"] = i
        timings["reranking_ms"] = round((time.time() - t0) * 1000, 2)

        timings["total_ms"] = round((time.time() - start_total) * 1000, 2)

        response = {
            "query": query,
            "filters": filters or {},
            "total_candidates_retrieved": len(hydrated_list),
            "dense_count": len(dense_results),
            "bm25_count": len(bm25_results),
            "top_k": len(final_results),
            "from_cache": False,
            "latency": timings,
            "results": final_results,
        }

        # 6. Cache write (Tier 1 Exact + Tier 2 Semantic)
        if use_cache:
            self.cache.set(cache_key, response)
            if not filters:
                self.cache.set_semantic(q_emb, query, response)

        return response

"""
Stage 1b — Elasticsearch BM25 & Keyword Retriever Engine
=========================================================
Executes BM25 full-text keyword retrieval, author-specific filtering,
and genre facet aggregation via Elasticsearch.
Falls back seamlessly to in-memory BM25Okapi if Elasticsearch daemon is offline.
"""

import json
import logging
import re
import sqlite3
import time
from typing import Any, Dict, List, Optional

try:
    from elasticsearch import Elasticsearch, helpers
except ImportError:
    Elasticsearch = None

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

from .config import ES_HOST, ES_INDEX_NAME, ES_TIMEOUT, SQLITE_DB_PATH

logger = logging.getLogger("BookSearch.ESRetriever")


class ESRetriever:
    def __init__(self, es_host: str = ES_HOST, index_name: str = ES_INDEX_NAME):
        self.es_host = es_host
        self.index_name = index_name
        self.es_client: Optional[Elasticsearch] = None
        self.is_es_active = False

        # In-memory fallback structures
        self.fallback_bm25: Optional[BM25Okapi] = None
        self.fallback_docs: List[Dict[str, Any]] = []
        self.fallback_ids: List[str] = []

        self._init_connection()

    def _init_connection(self):
        if Elasticsearch is not None:
            try:
                client = Elasticsearch(self.es_host, request_timeout=ES_TIMEOUT)
                if client.ping():
                    self.es_client = client
                    self.is_es_active = True
                    logger.info(f"Connected to Elasticsearch at {self.es_host}")
                    return
            except Exception as e:
                logger.warning(f"Elasticsearch unavailable at {self.es_host} ({e}). Using in-memory BM25.")

        self._init_fallback_bm25()

    def _init_fallback_bm25(self):
        """Loads SQLite metadata into in-memory BM25Okapi for zero-dependency execution."""
        if not SQLITE_DB_PATH.exists() or BM25Okapi is None:
            logger.warning("Fallback BM25 cannot initialize (missing SQLite or rank_bm25).")
            return

        logger.info("Initializing in-memory BM25 fallback from SQLite...")
        t0 = time.time()
        conn = sqlite3.connect(SQLITE_DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT book_id, title, authors, description, popular_shelves FROM books")
        rows = cur.fetchall()
        conn.close()

        corpus_tokens = []
        for bid, title, authors, desc, shelves in rows:
            s_bid = str(bid)
            parts = [title or ""]
            try:
                auth_list = json.loads(authors) if authors and authors.startswith("[") else [authors]
                parts.extend(auth_list)
            except Exception:
                pass
            try:
                shelf_list = json.loads(shelves) if shelves and shelves.startswith("[") else []
                parts.extend(shelf_list[:8])
            except Exception:
                pass
            if desc:
                parts.append(desc[:400])

            combined_text = " ".join(parts).lower()
            tokens = re.findall(r"\w+", combined_text)
            corpus_tokens.append(tokens)

            self.fallback_docs.append({
                "book_id": s_bid,
                "title": title or "",
                "authors": auth_list if "auth_list" in locals() else [],
                "description": desc or "",
                "popular_shelves": shelf_list if "shelf_list" in locals() else [],
            })
            self.fallback_ids.append(s_bid)

        self.fallback_bm25 = BM25Okapi(corpus_tokens)
        logger.info(f"Fallback BM25 initialized with {len(self.fallback_ids):,} books in {time.time() - t0:.2f}s")

    def create_index_if_missing(self):
        """Creates an Elasticsearch index with BM25 similarity and edge-ngram author analyzer."""
        if not self.is_es_active or not self.es_client:
            return

        if self.es_client.indices.exists(index=self.index_name):
            logger.info(f"Elasticsearch index '{self.index_name}' already exists.")
            return

        mapping = {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "analysis": {
                    "analyzer": {
                        "author_autocomplete": {
                            "tokenizer": "author_edge_ngram",
                            "filter": ["lowercase"]
                        }
                    },
                    "tokenizer": {
                        "author_edge_ngram": {
                            "type": "edge_ngram",
                            "min_gram": 2,
                            "max_gram": 15,
                            "token_chars": ["letter", "digit"]
                        }
                    }
                }
            },
            "mappings": {
                "properties": {
                    "book_id": {"type": "keyword"},
                    "title": {
                        "type": "text",
                        "analyzer": "english",
                        "fields": {"keyword": {"type": "keyword"}}
                    },
                    "authors": {
                        "type": "text",
                        "analyzer": "author_autocomplete",
                        "search_analyzer": "standard",
                        "fields": {"keyword": {"type": "keyword"}}
                    },
                    "description": {"type": "text", "analyzer": "english"},
                    "popular_shelves": {"type": "keyword"},
                    "average_rating": {"type": "float"},
                    "ratings_count": {"type": "integer"},
                    "publication_year": {"type": "integer"}
                }
            }
        }
        self.es_client.indices.create(index=self.index_name, body=mapping)
        logger.info(f"Created Elasticsearch index '{self.index_name}'.")

    def search(self, query: str, filters: Optional[Dict] = None, top_k: int = 50) -> List[Dict]:
        """Performs BM25 keyword search across title, authors, shelves, and description."""
        if self.is_es_active and self.es_client:
            return self._search_elasticsearch(query, filters, top_k)
        return self._search_fallback(query, filters, top_k)

    def _search_elasticsearch(self, query: str, filters: Optional[Dict], top_k: int) -> List[Dict]:
        must_clauses = [{
            "multi_match": {
                "query": query,
                "fields": [
                    "title^3.5",
                    "authors^3.0",
                    "popular_shelves^1.5",
                    "description^1.0"
                ],
                "fuzziness": "AUTO",
                "operator": "or"
            }
        }]

        filter_clauses = []
        if filters:
            if "author" in filters:
                filter_clauses.append({"match_phrase": {"authors": filters["author"]}})
            if "genre" in filters:
                filter_clauses.append({"term": {"popular_shelves": filters["genre"].lower()}})
            if "min_rating" in filters:
                filter_clauses.append({"range": {"average_rating": {"gte": float(filters["min_rating"])}}})

        body = {
            "size": top_k,
            "query": {
                "bool": {
                    "must": must_clauses,
                    "filter": filter_clauses
                }
            }
        }

        try:
            res = self.es_client.search(index=self.index_name, body=body)
            results = []
            for rank, hit in enumerate(res["hits"]["hits"], start=1):
                src = hit["_source"]
                results.append({
                    "book_id": src["book_id"],
                    "bm25_score": float(hit["_score"]),
                    "bm25_rank": rank,
                    "title": src.get("title", ""),
                    "authors": src.get("authors", []),
                    "popular_shelves": src.get("popular_shelves", []),
                    "description": src.get("description", ""),
                })
            return results
        except Exception as e:
            logger.error(f"Elasticsearch search error: {e}. Falling back.")
            return self._search_fallback(query, filters, top_k)

    def _search_fallback(self, query: str, filters: Optional[Dict], top_k: int) -> List[Dict]:
        if self.fallback_bm25 is None or not self.fallback_ids:
            return []

        tokens = re.findall(r"\w+", query.lower())
        scores = self.fallback_bm25.get_scores(tokens)

        # Top K indices
        import numpy as np
        top_indices = np.argpartition(-scores, min(top_k, len(scores)))[:top_k]
        top_indices = top_indices[np.argsort(-scores[top_indices])]

        results = []
        for rank, idx in enumerate(top_indices, start=1):
            sc = float(scores[idx])
            if sc <= 0:
                continue
            doc = self.fallback_docs[idx]
            results.append({
                "book_id": doc["book_id"],
                "bm25_score": sc,
                "bm25_rank": rank,
                "title": doc["title"],
                "authors": doc["authors"],
                "popular_shelves": doc["popular_shelves"],
                "description": doc["description"],
            })

        return results

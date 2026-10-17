"""
Stage 1a — Dense Bi-Encoder Retriever Engine
============================================
Performs approximate nearest neighbor (ANN) vector search over the catalog
using the fine-tuned BGE-small bi-encoder.
"""

import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from .config import (
    CATALOG_EMB_CACHE,
    CATALOG_IDS_CACHE,
    DEVICE,
    EMBED_MODEL_PATH,
    MAX_SEQ_LENGTH_EMBED,
    QUERY_INSTRUCTION,
    SQLITE_DB_PATH,
    USE_FP16,
)

logger = logging.getLogger("BookSearch.DenseRetriever")


class DenseRetriever:
    def __init__(self, model_path: str = EMBED_MODEL_PATH, device: str = DEVICE):
        self.device = device
        self.model_path = model_path
        logger.info(f"Loading dense embedding model from '{model_path}' on {device}...")
        t0 = time.time()
        self.model = SentenceTransformer(model_path, device=device)
        self.model.max_seq_length = MAX_SEQ_LENGTH_EMBED
        logger.info(f"Dense model loaded in {time.time() - t0:.2f}s")

        self.catalog_ids: List[str] = []
        self.catalog_matrix: Optional[torch.Tensor] = None
        self._load_or_build_catalog()

    def _load_or_build_catalog(self):
        """Loads cached FP16 catalog embeddings into GPU/CPU memory."""
        if CATALOG_EMB_CACHE.exists() and CATALOG_IDS_CACHE.exists():
            logger.info(f"Loading cached catalog embeddings from {CATALOG_EMB_CACHE.name}...")
            t0 = time.time()
            with open(CATALOG_IDS_CACHE, "r", encoding="utf-8") as f:
                self.catalog_ids = json.load(f)

            self.catalog_matrix = torch.load(CATALOG_EMB_CACHE, map_location=self.device)
            if USE_FP16 and self.device == "cuda":
                self.catalog_matrix = self.catalog_matrix.half()
            else:
                self.catalog_matrix = self.catalog_matrix.float()

            logger.info(f"Loaded {len(self.catalog_ids):,} embeddings in {time.time() - t0:.2f}s (Shape: {self.catalog_matrix.shape})")
        elif SQLITE_DB_PATH.exists():
            logger.info("Catalog embeddings cache not found. Building from SQLite database...")
            self.build_catalog_from_sqlite(SQLITE_DB_PATH)

    def build_catalog_from_sqlite(self, db_path: Path):
        """Builds and caches embeddings for all books in SQLite database."""
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT book_id, title, authors, description, popular_shelves FROM books")
        rows = cur.fetchall()
        conn.close()

        logger.info(f"Encoding {len(rows):,} books from SQLite...")
        book_ids = []
        formatted_texts = []

        for bid, title, authors, desc, shelves in rows:
            s_bid = str(bid)
            parts = []
            if title:
                parts.append(f"Title: {title}")
            if authors:
                try:
                    alist = json.loads(authors) if authors.startswith("[") else [authors]
                    parts.append(f"Author: {', '.join(alist)}")
                except Exception:
                    parts.append(f"Author: {authors}")
            if shelves:
                try:
                    slist = json.loads(shelves) if shelves.startswith("[") else []
                    parts.append(f"Genres: {', '.join(slist[:8])}")
                except Exception:
                    pass
            if desc:
                parts.append(f"Description: {desc[:600]}")

            book_ids.append(s_bid)
            formatted_texts.append(" | ".join(parts))

        embs = self.model.encode(
            formatted_texts,
            batch_size=64,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_tensor=True,
            device=self.device,
        )
        if USE_FP16 and self.device == "cuda":
            embs = embs.half()

        torch.save(embs.cpu(), CATALOG_EMB_CACHE)
        with open(CATALOG_IDS_CACHE, "w", encoding="utf-8") as f:
            json.dump(book_ids, f)

        self.catalog_ids = book_ids
        self.catalog_matrix = embs.to(self.device)
        logger.info(f"Catalog embeddings built and cached to {CATALOG_EMB_CACHE.name}")

    def encode_query(self, query: str) -> torch.Tensor:
        """Encodes query into normalized embedding vector."""
        prefixed_query = QUERY_INSTRUCTION + query.strip()
        q_emb = self.model.encode(
            prefixed_query,
            normalize_embeddings=True,
            convert_to_tensor=True,
            device=self.device,
        )
        if USE_FP16 and self.device == "cuda":
            return q_emb.half()
        return q_emb.float()

    def search_vector(self, q_emb: torch.Tensor, top_k: int = 50) -> List[Dict]:
        """Performs vector search directly with a precomputed query embedding."""
        if self.catalog_matrix is None or not self.catalog_ids:
            logger.error("Catalog matrix is not initialized!")
            return []

        sims = torch.matmul(q_emb.unsqueeze(0), self.catalog_matrix.T).squeeze(0)
        topk_scores, topk_indices = torch.topk(sims, k=min(top_k, len(self.catalog_ids)), largest=True)

        scores = topk_scores.cpu().tolist()
        indices = topk_indices.cpu().tolist()

        results = []
        for rank, (idx, score) in enumerate(zip(indices, scores), start=1):
            bid = self.catalog_ids[idx]
            results.append({
                "book_id": bid,
                "dense_score": float(score),
                "dense_rank": rank,
            })

        return results

    def search(self, query: str, top_k: int = 50) -> List[Dict]:
        """Encodes query and retrieves top_k closest catalog books by cosine similarity."""
        q_emb = self.encode_query(query)
        return self.search_vector(q_emb, top_k=top_k)

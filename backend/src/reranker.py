"""
Stage 2 — Cross-Encoder Reranker Engine
=======================================
Performs full cross-attention scoring over (query, book_metadata) candidate pairs
using the fine-tuned BGE-reranker-base model (88.7% MRR@10).
"""

import json
import logging
import time
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import DEVICE, MAX_SEQ_LENGTH_RERANK, RERANKER_MODEL_PATH, USE_FP16

logger = logging.getLogger("BookSearch.Reranker")


class RerankerEngine:
    def __init__(self, model_path: str = RERANKER_MODEL_PATH, device: str = DEVICE):
        self.device = device
        self.model_path = model_path
        logger.info(f"Loading Cross-Encoder Reranker from '{model_path}' on {device}...")
        t0 = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=1)
        self.model.to(device)
        self.model.eval()
        logger.info(f"Reranker model loaded in {time.time() - t0:.2f}s")

    @staticmethod
    def format_candidate_text(candidate: Dict) -> str:
        parts = []
        if candidate.get("title"):
            parts.append(f"Title: {candidate['title']}")
        if candidate.get("authors"):
            auths = candidate["authors"]
            if isinstance(auths, list):
                parts.append(f"Author: {', '.join(auths)}")
            else:
                parts.append(f"Author: {auths}")
        if candidate.get("popular_shelves"):
            shelves = candidate["popular_shelves"]
            if isinstance(shelves, list):
                parts.append(f"Genres: {', '.join(shelves[:8])}")
        if candidate.get("description"):
            parts.append(f"Description: {candidate['description'][:600]}")

        return " | ".join(parts)

    def rerank(self, query: str, candidates: List[Dict], batch_size: int = 32, top_k: int = 10) -> List[Dict]:
        """Reranks candidate books by cross-attention relevance score."""
        if not candidates:
            return []

        t0 = time.time()
        cand_texts = [self.format_candidate_text(c) for c in candidates]
        pairs = [[query, text] for text in cand_texts]

        scores = []
        with torch.no_grad():
            for i in range(0, len(pairs), batch_size):
                batch_pairs = pairs[i:i + batch_size]
                inputs = self.tokenizer(
                    [p[0] for p in batch_pairs],
                    [p[1] for p in batch_pairs],
                    padding=True,
                    truncation=True,
                    max_length=MAX_SEQ_LENGTH_RERANK,
                    return_tensors="pt"
                ).to(self.device)

                with torch.cuda.amp.autocast(enabled=USE_FP16):
                    logits = self.model(**inputs).logits.squeeze(-1)
                    batch_scores = logits.cpu().tolist()
                    if not isinstance(batch_scores, list):
                        batch_scores = [batch_scores]
                    scores.extend(batch_scores)

        # Attach scores to candidates
        scored_candidates = []
        for cand, score in zip(candidates, scores):
            item = dict(cand)
            item["rerank_score"] = float(score)
            scored_candidates.append(item)

        # Sort descending by cross-encoder score
        scored_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

        for rank, cand in enumerate(scored_candidates, start=1):
            cand["final_rank"] = rank

        logger.debug(f"Reranked {len(candidates)} candidates in {time.time() - t0:.3f}s")
        return scored_candidates[:top_k]

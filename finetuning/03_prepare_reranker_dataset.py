"""
03 — Prepare Reranker Fine-Tuning Dataset
==========================================
Prepares high-quality training and validation data for fine-tuning a Cross-Encoder
reranker (e.g., BAAI/bge-reranker-base).

Inputs Required:
  - books_metadata.sqlite3:          SQLite DB containing book catalog metadata
  - train.jsonl / val.jsonl:         Output from Step 1 containing queries & positive book IDs
  - bge_small_v5_output/best_model:  Best bi-encoder retriever from Step 2 (or v3/v4 fallback)
  - catalog_goodreads_metadata.json: (Auto-extracted from goodreads_books.json.gz if missing)

Key Features:
  1. Goldilocks Hard Negative Mining (Ranks 10–45):
     - Bypasses ranks 0–9 entirely to eliminate false-negative poisoning
       (unlabelled sequels, adaptations, and near-duplicates).
     - Mines genuine hard distractors from ranks 10–45 that the bi-encoder
       retrieved in its top 0.05% candidates.
  2. Multi-Layer False-Negative Suppression:
     - Direct & cross-query positive ID exclusion.
     - Title normalization and base-title matching.
     - Significant word overlap threshold (< 60%).
     - Author ID and Series ID exclusion via Goodreads metadata.
     - Work ID exclusion (alternate editions / translations).
     - Goodreads reader-endorsed co-read / similar books exclusion.
     - Hard cosine similarity ceiling (<= 0.82).
  3. Frequency-Capped Distractor Sampling:
     - Prevents universal distractors from dominating gradient updates.
  4. Dual-Format Output:
     - reranker_train_grouped.jsonl: {query, positive_id, positive_text, negatives: [...]}
     - reranker_train_pairs.jsonl:   {query, text, label: 1.0/0.0} (ready for CrossEncoder training)
     - reranker_val_pairs.jsonl:     Validation pairs for MRR@10 and MAP evaluation

Usage:
  python 03_prepare_reranker_dataset.py
  python 03_prepare_reranker_dataset.py --model bge_small_v5_output/best_model --negs-per-query 4
  python 03_prepare_reranker_dataset.py --max-train 60000 --max-val 3000
"""

import argparse
import gzip
import json
import math
import os
import random
import re
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

# ==============================================================================
# Configuration & Paths
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent

DB_FILE = BASE_DIR / "books_metadata.sqlite3"
TRAIN_RAW = BASE_DIR / "train.jsonl"
VAL_RAW = BASE_DIR / "val.jsonl"
GOODREADS_META_FILE = BASE_DIR / "catalog_goodreads_metadata.json"
GZ_DUMP_FILE = BASE_DIR / "goodreads_books.json.gz"
CATALOG_EMB_CACHE = BASE_DIR / "catalog_bge_embeddings_fp16.pt"

# Output files
OUT_TRAIN_GROUPED = BASE_DIR / "reranker_train_grouped.jsonl"
OUT_TRAIN_PAIRS = BASE_DIR / "reranker_train_pairs.jsonl"
OUT_VAL_PAIRS = BASE_DIR / "reranker_val_pairs.jsonl"

QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "in", "on", "for", "to", "with",
    "about", "is", "are", "that", "this", "i", "my", "me", "book", "books",
    "novel", "novels", "story", "stories", "series", "vol", "volume", "part"
}


def norm_text(text: str) -> str:
    """Lowercase and normalize whitespace."""
    return re.sub(r"\s+", " ", (text or "").lower().strip())


def extract_base_title(title: str) -> str:
    """Strip subtitles, editions, and parentheticals to extract core title."""
    t = re.sub(r"\(.*?\)", "", title)
    t = re.sub(r"\[.*?\]", "", t)
    t = re.split(r"[:\-–—;]", t)[0]
    return norm_text(t)


def get_significant_words(title: str) -> set:
    """Extract informative non-stopword tokens of length >= 3."""
    words = re.findall(r"[a-z0-9']+", norm_text(title))
    return {w for w in words if len(w) >= 3 and w not in STOPWORDS}


# ==============================================================================
# Metadata Loading & Extraction
# ==============================================================================
def load_book_metadata(db_path: Path, goodreads_meta_path: Path, gz_dump_path: Path):
    """Loads text catalog and authoritative Goodreads relations (authors, series, work IDs)."""
    print(f"\n[Metadata] Loading book catalog from {db_path.name}...")
    t0 = time.time()
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    book_texts = {}
    title_of = {}
    base_title_of = {}
    sig_words_of = {}

    query = "SELECT book_id, title, authors, description, popular_shelves FROM books"
    for row in cur.execute(query):
        bid = str(row[0])
        title = row[1] or ""
        desc = (row[3] or "").strip()
        shelves_raw = row[4] or "[]"

        parts = []
        if title.strip():
            parts.append(f"Title: {title.strip()}")
        try:
            shelf_list = json.loads(shelves_raw)
            if shelf_list:
                parts.append(f"Genres: {', '.join(shelf_list)}")
        except Exception:
            pass
        if desc:
            parts.append(f"Description: {desc[:900]}")
        book_texts[bid] = " | ".join(parts) if parts else "Untitled"

        norm_t = norm_text(title)
        title_of[bid] = norm_t
        base_title_of[bid] = extract_base_title(title)
        sig_words_of[bid] = get_significant_words(title)

    conn.close()
    print(f"[Metadata] Loaded {len(book_texts):,} books from SQLite in {time.time() - t0:.2f}s.")

    # Extract Goodreads relations from gzip dump if cached JSON doesn't exist
    if not goodreads_meta_path.exists() and gz_dump_path.exists():
        print(f"[Metadata] Extracting Goodreads relations from {gz_dump_path.name}...")
        cat_ids = set(book_texts.keys())
        catalog_meta = {}
        with gzip.open(gz_dump_path, "rt", encoding="utf-8") as gz:
            for line in gz:
                row = json.loads(line)
                bid = str(row.get("book_id", ""))
                if bid in cat_ids:
                    authors = [a.get("author_id") for a in row.get("authors", []) if a.get("author_id")]
                    series = row.get("series", [])
                    work_id = str(row.get("work_id", ""))
                    sim_books = row.get("similar_books", [])
                    shelves = [sh.get("name", "").lower() for sh in row.get("popular_shelves", []) if int(sh.get("count", 0)) >= 20]
                    catalog_meta[bid] = {
                        "authors": authors,
                        "series": series,
                        "work_id": work_id,
                        "similar_books": sim_books,
                        "shelves": shelves[:30]
                    }
                    if len(catalog_meta) == len(cat_ids):
                        break
        with open(goodreads_meta_path, "w", encoding="utf-8") as out_f:
            json.dump(catalog_meta, out_f)
        print(f"[Metadata] Saved {len(catalog_meta):,} records to {goodreads_meta_path.name}.")

    authors_of = {}
    series_of = {}
    work_id_of = {}
    similar_books_of = {}
    shelves_of = {}

    if goodreads_meta_path.exists():
        print(f"[Metadata] Loading Goodreads relations from {goodreads_meta_path.name}...")
        t1 = time.time()
        with open(goodreads_meta_path, "r", encoding="utf-8") as f:
            gr_meta = json.load(f)

        for bid in book_texts.keys():
            d = gr_meta.get(bid, {})
            authors_of[bid] = set(d.get("authors", []))
            series_of[bid] = set(d.get("series", []))
            work_id_of[bid] = str(d.get("work_id", ""))
            similar_books_of[bid] = set(d.get("similar_books", []))
            shelves_of[bid] = set(d.get("shelves", []))
        print(f"[Metadata] Loaded relations for {len(gr_meta):,} books in {time.time() - t1:.2f}s.")
    else:
        print("[Metadata] Warning: Goodreads relations file not found. Title/cosine filtering only.")

    return {
        "book_texts": book_texts,
        "title_of": title_of,
        "base_title_of": base_title_of,
        "sig_words_of": sig_words_of,
        "authors_of": authors_of,
        "series_of": series_of,
        "work_id_of": work_id_of,
        "similar_books_of": similar_books_of,
        "shelves_of": shelves_of,
    }


# ==============================================================================
# False-Negative Detection
# ==============================================================================
def is_false_negative(
    cand_id: str,
    cand_score: float,
    meta: dict,
    banned_ids: set,
    banned_authors: set,
    banned_series: set,
    banned_works: set,
    banned_similar: set,
    max_cosine_sim: float
) -> tuple:
    """Returns (True, reason) if candidate is a false negative; otherwise (False, None)."""
    # 1. Direct ID & similarity ceiling check
    if cand_id in banned_ids:
        return True, "banned_id"
    if cand_score > max_cosine_sim:
        return True, "cosine_ceiling"

    # 2. Author overlap (eliminates sequels/companion novels by same author)
    cand_authors = meta["authors_of"].get(cand_id, set())
    if cand_authors and banned_authors and (cand_authors & banned_authors):
        return True, "same_author"

    # 3. Series overlap (eliminates sequels, prequels, novellas)
    cand_series = meta["series_of"].get(cand_id, set())
    if cand_series and banned_series and (cand_series & banned_series):
        return True, "same_series"

    # 4. Work ID overlap (eliminates alternate editions & translations)
    cand_work = meta["work_id_of"].get(cand_id, "")
    if cand_work and banned_works and (cand_work in banned_works):
        return True, "same_work_id"

    # 5. Goodreads reader-endorsed co-reads
    cand_sim = meta["similar_books_of"].get(cand_id, set())
    if (banned_similar and cand_id in banned_similar) or (cand_sim and (banned_ids & cand_sim)):
        return True, "goodreads_similar_book"

    # 6. Title and base-title overlap
    cand_title = meta["title_of"].get(cand_id, "")
    cand_base = meta["base_title_of"].get(cand_id, "")
    cand_words = meta["sig_words_of"].get(cand_id, set())

    for pid in banned_ids:
        pos_title = meta["title_of"].get(pid, "")
        if pos_title and cand_title and pos_title == cand_title:
            return True, "identical_title"

        pos_base = meta["base_title_of"].get(pid, "")
        if pos_base and cand_base and pos_base == cand_base and len(pos_base) > 3:
            return True, "base_title_match"

        pos_words = meta["sig_words_of"].get(pid, set())
        if pos_words and cand_words:
            shared = pos_words & cand_words
            overlap = len(shared) / max(len(pos_words), len(cand_words))
            if overlap >= 0.60:
                return True, "high_title_overlap"

    return False, None


# ==============================================================================
# Catalog Embedding Cache
# ==============================================================================
def get_catalog_embeddings(model, book_ids: list, book_texts: dict, device: str, cache_path: Path):
    """Computes or loads cached normalized catalog embeddings."""
    if cache_path.exists():
        print(f"[Catalog] Loading cached catalog embeddings from {cache_path.name}...")
        t0 = time.time()
        embs = torch.load(cache_path, map_location=device)
        print(f"[Catalog] Loaded shape {embs.shape} in {time.time() - t0:.2f}s.")
        return embs

    print(f"[Catalog] Encoding {len(book_ids):,} catalog items with {device}...")
    t0 = time.time()
    texts = [book_texts[bid] for bid in book_ids]
    raw_embs = model.encode(
        texts,
        batch_size=256,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_tensor=True,
        device=device
    )
    embs = raw_embs.to(torch.float16 if device == "cuda" else torch.float32)
    torch.save(embs, cache_path)
    print(f"[Catalog] Encoded and saved {embs.shape} in {time.time() - t0:.2f}s.")
    return embs


# ==============================================================================
# Reranker Dataset Mining Pipeline
# ==============================================================================
def mine_dataset(
    split_name: str,
    raw_file: Path,
    meta: dict,
    model: SentenceTransformer,
    catalog_ids: list,
    catalog_embs: torch.Tensor,
    device: str,
    args
):
    print(f"\n{'='*70}")
    print(f"Mining Reranker Data for {split_name.upper()} split")
    print(f"{'='*70}")

    # Load raw queries and group by query text
    query_to_bids = defaultdict(set)
    total_raw = 0
    with open(raw_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            q = row.get("query", "").strip()
            bid = str(row.get("book_id", "")).strip()
            if q and bid and bid in meta["book_texts"]:
                query_to_bids[q].add(bid)
                total_raw += 1

    distinct_queries = list(query_to_bids.keys())
    print(f"[{split_name}] Found {len(distinct_queries):,} distinct queries across {total_raw:,} records.")

    # Limit query count if requested
    max_q = args.max_train_queries if split_name == "train" else args.max_val_queries
    if max_q and len(distinct_queries) > max_q:
        random.seed(args.seed)
        random.shuffle(distinct_queries)
        distinct_queries = distinct_queries[:max_q]
        print(f"[{split_name}] Sampled {len(distinct_queries):,} queries according to budget.")

    # Encode queries
    print(f"[{split_name}] Encoding queries with instruction prefix...")
    instructed_queries = [f"{QUERY_INSTRUCTION}{q}" for q in distinct_queries]
    q_embs = model.encode(
        instructed_queries,
        batch_size=256,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_tensor=True,
        device=device
    ).to(catalog_embs.dtype)

    neg_freq = Counter()
    audit_counter = Counter()
    grouped_records = []
    pairs_records = []

    batch_size = 256
    num_queries = len(distinct_queries)

    print(f"[{split_name}] Mining Goldilocks hard negatives (ranks {args.min_rank}–{args.max_rank})...")
    for start in tqdm(range(0, num_queries, batch_size), desc=f"Mining {split_name}"):
        end = min(start + batch_size, num_queries)
        sub_q_embs = q_embs[start:end]

        # Cosine similarity matrix: [batch_q, catalog_size]
        sim_matrix = torch.matmul(sub_q_embs, catalog_embs.T)
        top_scores, top_indices = torch.topk(sim_matrix, k=args.top_k, dim=1)

        top_scores_cpu = top_scores.cpu().numpy()
        top_indices_cpu = top_indices.cpu().numpy()

        for idx, q_idx in enumerate(range(start, end)):
            query_str = distinct_queries[q_idx]
            banned_ids = query_to_bids[query_str]

            # Collect relation bans across all associated positive books
            banned_authors = set()
            banned_series = set()
            banned_works = set()
            banned_similar = set()

            for bid in banned_ids:
                banned_authors.update(meta["authors_of"].get(bid, set()))
                banned_series.update(meta["series_of"].get(bid, set()))
                w = meta["work_id_of"].get(bid, "")
                if w:
                    banned_works.add(w)
                banned_similar.update(meta["similar_books_of"].get(bid, set()))

            pos_id = next(iter(banned_ids))
            pos_text = meta["book_texts"].get(pos_id, "")
            if not pos_text:
                continue

            # Scan top retrieved candidates for Goldilocks distractors
            chosen_negs = []
            chosen_neg_ids = []
            chosen_ranks = []
            chosen_sims = []

            for rank in range(args.top_k):
                cand_idx = int(top_indices_cpu[idx, rank])
                cand_id = catalog_ids[cand_idx]
                cand_score = float(top_scores_cpu[idx, rank])

                is_fn, reason = is_false_negative(
                    cand_id, cand_score, meta,
                    banned_ids, banned_authors, banned_series, banned_works, banned_similar,
                    max_cosine_sim=args.max_cosine_sim
                )

                if is_fn:
                    audit_counter[f"filtered_{reason}"] += 1
                    continue

                # Bypass ranks 0..(min_rank-1) to avoid subtle false-negative leakage
                if rank < args.min_rank:
                    audit_counter["skipped_rank_below_min"] += 1
                    continue

                # Candidate is beyond max_rank
                if rank > args.max_rank:
                    audit_counter["skipped_rank_above_max"] += 1
                    break

                # Apply frequency cap for negative reuse
                if neg_freq[cand_id] >= args.max_neg_freq:
                    audit_counter["skipped_freq_cap"] += 1
                    continue

                neg_text = meta["book_texts"].get(cand_id, "")
                if not neg_text:
                    continue

                chosen_negs.append(neg_text)
                chosen_neg_ids.append(cand_id)
                chosen_ranks.append(rank)
                chosen_sims.append(round(cand_score, 4))
                neg_freq[cand_id] += 1

                if len(chosen_negs) == args.negs_per_query:
                    break

            if len(chosen_negs) < args.negs_per_query:
                audit_counter["insufficient_negatives"] += 1
                continue

            audit_counter["valid_queries"] += 1

            # Grouped format
            grouped_records.append({
                "query": query_str,
                "positive_id": pos_id,
                "positive_text": pos_text,
                "negatives": chosen_negs,
                "negative_ids": chosen_neg_ids,
                "negative_ranks": chosen_ranks,
                "negative_sims": chosen_sims
            })

            # Pair format (1 positive pair + N negative pairs)
            pairs_records.append({"query": query_str, "text": pos_text, "label": 1.0})
            for n_txt in chosen_negs:
                pairs_records.append({"query": query_str, "text": n_txt, "label": 0.0})

    print(f"\n[{split_name}] Mining Statistics:")
    for k, v in audit_counter.most_common():
        print(f"  {k:30s}: {v:,}")

    return grouped_records, pairs_records


# ==============================================================================
# Main Routine
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="03 — Prepare Reranker Fine-Tuning Dataset")
    parser.add_argument("--model", type=str, default="", help="Path or HuggingFace ID of bi-encoder retriever")
    parser.add_argument("--max-train-queries", type=int, default=60000, help="Max queries to mine for training")
    parser.add_argument("--max-val-queries", type=int, default=3000, help="Max queries to mine for validation")
    parser.add_argument("--negs-per-query", type=int, default=4, help="Hard negatives per query (default: 4)")
    parser.add_argument("--min-rank", type=int, default=10, help="Minimum retriever rank (Goldilocks floor)")
    parser.add_argument("--max-rank", type=int, default=45, help="Maximum retriever rank (Goldilocks ceiling)")
    parser.add_argument("--top-k", type=int, default=70, help="Retrieval candidate pool size")
    parser.add_argument("--max-cosine-sim", type=float, default=0.82, help="Max allowed cosine similarity for negatives")
    parser.add_argument("--max-neg-freq", type=int, default=35, help="Max occurrences of any single negative book")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Active Device: {device} | PyTorch: {torch.__version__}")

    # Determine bi-encoder model path
    model_path = args.model
    if not model_path:
        for candidate in [
            BASE_DIR / "bge_small_v5_output" / "best_model",
            BASE_DIR / "bge_small_v4_output" / "best_model",
            BASE_DIR / "bge_small_v3_output" / "best_model",
            "BAAI/bge-small-en-v1.5"
        ]:
            if isinstance(candidate, Path) and candidate.exists():
                model_path = str(candidate)
                break
            elif isinstance(candidate, str):
                model_path = candidate
                break

    print(f"Using Retriever for Hard Negative Mining: {model_path}")

    # 1. Load book metadata & authoritative Goodreads relations
    meta = load_book_metadata(DB_FILE, GOODREADS_META_FILE, GZ_DUMP_FILE)
    catalog_ids = sorted(list(meta["book_texts"].keys()))

    # 2. Load retriever model & catalog embeddings
    print(f"\n[Model] Loading bi-encoder from {model_path}...")
    model = SentenceTransformer(model_path, device=device)
    catalog_embs = get_catalog_embeddings(model, catalog_ids, meta["book_texts"], device, CATALOG_EMB_CACHE)

    # 3. Mine Training Split
    train_grouped, train_pairs = mine_dataset(
        "train", TRAIN_RAW, meta, model, catalog_ids, catalog_embs, device, args
    )

    print(f"\nSaving training files...")
    with open(OUT_TRAIN_GROUPED, "w", encoding="utf-8") as f:
        for row in train_grouped:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  Saved {len(train_grouped):,} queries to {OUT_TRAIN_GROUPED.name}")

    with open(OUT_TRAIN_PAIRS, "w", encoding="utf-8") as f:
        for row in train_pairs:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  Saved {len(train_pairs):,} pairs to {OUT_TRAIN_PAIRS.name}")

    # 4. Mine Validation Split
    val_grouped, val_pairs = mine_dataset(
        "val", VAL_RAW, meta, model, catalog_ids, catalog_embs, device, args
    )

    print(f"\nSaving validation files...")
    with open(OUT_VAL_PAIRS, "w", encoding="utf-8") as f:
        for row in val_pairs:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  Saved {len(val_pairs):,} pairs to {OUT_VAL_PAIRS.name}")

    print("\n" + "="*70)
    print("STEP 3 COMPLETE: Reranker Datasets Successfully Generated!")
    print(f"  Train: {len(train_pairs):,} pairs ({len(train_grouped):,} queries)")
    print(f"  Val:   {len(val_pairs):,} pairs ({len(val_grouped):,} queries)")
    print(f"Ready for training with: python 04_train_reranker.py")
    print("="*70)


if __name__ == "__main__":
    main()

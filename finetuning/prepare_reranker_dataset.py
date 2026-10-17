"""
Reranker Dataset Preparation Script
===================================
Prepares high-quality training and validation data for fine-tuning a Cross-Encoder
reranker (e.g., BAAI/bge-reranker-base or ms-marco-MiniLM-L-6-v2).

Key Features:
1. Goldilocks Hard Negative Mining (Ranks 10–40):
   - Bypasses ranks 0–9 entirely to eliminate false-negative poisoning
     (unlabelled sequels, adaptations, and near-duplicates).
   - Mines genuine hard distractors from ranks 10–40 that the bi-encoder
     retrieved in its top 0.05% candidates.
2. Multi-Layer False-Negative Filtering:
   - Excludes direct & cross-query positive IDs.
   - Normalizes & strips subtitles/editions.
   - Filters high title word overlap (>= 60%).
   - Excludes books with matching series or author tags.
   - Hard cosine ceiling at 0.82.
3. Dual-Format Output:
   - Grouped JSONL: {query, positive_id, positive_text, negatives: [...], negative_ids: [...], negative_ranks: [...]}
   - Pair JSONL: Individual (query, book_text, label) pairs ready for standard CrossEncoder training.
"""

import os
import sys
import json
import time
import math
import random
import re
import sqlite3
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

# Paths
BASE_DIR = Path(__file__).resolve().parent
DB_FILE = BASE_DIR / "books_metadata.sqlite3"
TRAIN_RAW = BASE_DIR / "train.jsonl"
VAL_RAW = BASE_DIR / "val.jsonl"
MODEL_PATH = BASE_DIR / "bge_small_v5_output" / "best_model"

# Fallback to v3 if v5 does not exist
if not MODEL_PATH.exists():
    MODEL_PATH = BASE_DIR / "bge_small_v3_output" / "best_model"

OUT_TRAIN_GROUPED = BASE_DIR / "reranker_train_grouped.jsonl"
OUT_TRAIN_PAIRS = BASE_DIR / "reranker_train_pairs.jsonl"
OUT_VAL_PAIRS = BASE_DIR / "reranker_val_pairs.jsonl"

GOODREADS_META_FILE = BASE_DIR / "catalog_goodreads_metadata.json"
GZ_DUMP_FILE = BASE_DIR / "goodreads_books.json.gz"
CATALOG_EMB_CACHE = BASE_DIR / "catalog_bge_embeddings_fp16.pt"

# Mining Hyperparameters
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
TOP_K_RETRIEVE = 70      # Expanded to 70 so plenty of clean candidates remain after strict false-negative filtering
MIN_RANK = 10           # Goldilocks Zone: bypass ranks 0-9 to avoid unlabelled true positives
MAX_RANK = 45           # Hard distractor window (ranks 10-45)
NEGS_PER_QUERY = 4      # 1 positive + 4 hard negatives = 1:4 ratio
MAX_COSINE_SIM = 0.82   # Strict similarity cap for negatives
MAX_NEG_FREQUENCY = 35  # Cap negative book reuse
MAX_TRAIN_QUERIES = 60000  # 60k queries * 5 pairs = 300,000 pairs (ideal size for RTX 4050 training)
MAX_VAL_QUERIES = 3000     # 3k queries * 5 pairs = 15,000 pairs for validation
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_FP16 = (DEVICE == "cuda")

STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "in", "on", "for", "to", "with",
    "about", "is", "are", "that", "this", "i", "my", "me", "book", "books",
    "novel", "novels", "story", "stories", "series", "vol", "volume", "part"
}

def norm_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower().strip())

def extract_base_title(title: str) -> str:
    t = re.sub(r"\(.*?\)", "", title)
    t = re.sub(r"\[.*?\]", "", t)
    t = re.split(r"[:\-–—;]", t)[0]
    return norm_text(t)

def get_significant_words(title: str) -> set:
    words = re.findall(r"[a-z0-9']+", norm_text(title))
    return {w for w in words if len(w) >= 3 and w not in STOPWORDS}

def load_book_metadata(db_path: Path, goodreads_meta_path: Path):
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
            shelf_list = []
        if desc:
            parts.append(f"Description: {desc[:900]}")
        book_texts[bid] = " | ".join(parts) if parts else "Untitled"

        norm_t = norm_text(title)
        title_of[bid] = norm_t
        base_title_of[bid] = extract_base_title(title)
        sig_words_of[bid] = get_significant_words(title)

    conn.close()
    print(f"[Metadata] Loaded {len(book_texts):,} books from SQLite in {time.time() - t0:.2f}s.")

    # Load Goodreads author_id, series_id, work_id, and similar_books metadata
    if not goodreads_meta_path.exists() and GZ_DUMP_FILE.exists():
        print(f"[Metadata] Extracting Goodreads metadata from {GZ_DUMP_FILE.name}...")
        import gzip
        cat_ids = set(book_texts.keys())
        catalog_meta = {}
        with gzip.open(GZ_DUMP_FILE, "rt", encoding="utf-8") as gz:
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

    print(f"[Metadata] Loading authoritative Goodreads metadata from {goodreads_meta_path.name}...")
    t1 = time.time()
    with open(goodreads_meta_path, "r", encoding="utf-8") as f:
        gr_meta = json.load(f)

    authors_of = {}
    series_of = {}
    work_id_of = {}
    similar_books_of = {}
    shelves_of = {}

    for bid in book_texts.keys():
        d = gr_meta.get(bid, {})
        authors_of[bid] = set(d.get("authors", []))
        series_of[bid] = set(d.get("series", []))
        work_id_of[bid] = str(d.get("work_id", ""))
        similar_books_of[bid] = set(d.get("similar_books", []))
        shelves_of[bid] = set(d.get("shelves", []))

    print(f"[Metadata] Loaded Goodreads metadata for {len(gr_meta):,} books in {time.time() - t1:.2f}s.")
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

def is_false_negative(
    pos_id: str,
    cand_id: str,
    cand_score: float,
    meta: dict,
    banned_ids: set,
    banned_authors: set = None,
    banned_series: set = None,
    banned_works: set = None,
    banned_similar: set = None
) -> tuple:
    # 1. Banned IDs (query positives across duplicate query entries)
    if cand_id in banned_ids:
        return True, "banned_id"
    if cand_score > MAX_COSINE_SIM:
        return True, "cosine_ceiling"

    # 2. Author check across all positives for this query (eliminates sequels & spin-offs by same author)
    cand_authors = meta["authors_of"].get(cand_id, set())
    if cand_authors and banned_authors and (cand_authors & banned_authors):
        return True, "same_author"

    # 3. Series check across all positives for this query (eliminates sequels, prequels, novellas, companion books)
    cand_series = meta["series_of"].get(cand_id, set())
    if cand_series and banned_series and (cand_series & banned_series):
        return True, "same_series"

    # 4. Work check across all positives for this query (eliminates alternate editions, translations, graphic novels)
    cand_work = meta["work_id_of"].get(cand_id, "")
    if cand_work and banned_works and (cand_work in banned_works):
        return True, "same_work_id"

    # 5. Goodreads Reader-Endorsed Co-Read / Similar Books across all positives
    cand_sim = meta["similar_books_of"].get(cand_id, set())
    if (banned_similar and cand_id in banned_similar) or (cand_sim and banned_ids & cand_sim):
        return True, "goodreads_similar_book"

    # 6. Title matching against all positives of this query
    cand_title = meta["title_of"].get(cand_id, "")
    cand_base = meta["base_title_of"].get(cand_id, "")
    cand_words = meta["sig_words_of"].get(cand_id, set())
    cand_shelves = meta["shelves_of"].get(cand_id, set())

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
            min_len = min(len(pos_words), len(cand_words))
            if min_len >= 2 and (len(shared) / min_len) >= 0.50:
                return True, "title_word_overlap"

        # Shared Franchise Shelves
        pos_shelves = meta["shelves_of"].get(pid, set())
        if pos_shelves and cand_shelves:
            shared_shelves = pos_shelves & cand_shelves
            franchise_shelves = [s for s in shared_shelves if any(x in s for x in ["series", "potter", "ice-and-fire", "thrones", "tolkien", "rings", "sanderson", "martin", "westeros", "wheel-of-time"])]
            if franchise_shelves:
                return True, "shared_franchise_shelf"

    return False, ""

def generate_reranker_data(
    model: SentenceTransformer,
    catalog_tensor: torch.Tensor,
    all_book_ids: list,
    meta: dict,
    input_jsonl: Path,
    out_pairs_jsonl: Path,
    out_grouped_jsonl: Path = None,
    max_queries: int = None,
    split_name: str = "train",
    exclude_queries: set = None,
):
    print(f"\n{'='*70}")
    print(f"  MINING RERANKER DATA: {split_name.upper()} SPLIT")
    print(f"  Input:  {input_jsonl.name}")
    print(f"  Output: {out_pairs_jsonl.name}")
    print(f"{'='*70}")

    # Load queries and index query positives
    query_positives = defaultdict(set)
    raw_rows = []
    with open(input_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                norm_q = norm_text(row["query"])
                if exclude_queries and norm_q in exclude_queries:
                    continue
                raw_rows.append(row)
                query_positives[norm_q].add(str(row["positive_id"]))

    print(f"[Loading] Loaded {len(raw_rows):,} queries from {input_jsonl.name}.")

    # Subsample if max_queries specified
    if max_queries and len(raw_rows) > max_queries:
        # Prioritize queries where retriever positive_rank > 0 (the harder queries where reranker matters most)
        hard_queries = [r for r in raw_rows if r.get("positive_rank", 0) > 0]
        easy_queries = [r for r in raw_rows if r.get("positive_rank", 0) == 0]
        random.shuffle(hard_queries)
        random.shuffle(easy_queries)

        n_hard = min(len(hard_queries), int(max_queries * 0.70))
        n_easy = max_queries - n_hard
        selected_rows = hard_queries[:n_hard] + easy_queries[:n_easy]
        random.shuffle(selected_rows)
        print(f"[Subsampling] Selected {len(selected_rows):,} queries ({n_hard:,} hard queries + {n_easy:,} easy queries).")
    else:
        selected_rows = raw_rows
        random.shuffle(selected_rows)

    batch_size = 256
    total = len(selected_rows)
    neg_frequency = Counter()
    total_pairs = 0
    grouped_count = 0
    blocked_counts = Counter()
    rank_distribution = Counter()

    t0 = time.time()
    pairs_file = open(out_pairs_jsonl, "w", encoding="utf-8")
    grouped_file = open(out_grouped_jsonl, "w", encoding="utf-8") if out_grouped_jsonl else None

    for b_start in tqdm(range(0, total, batch_size), desc=f"Mining {split_name}"):
        b_end = min(b_start + batch_size, total)
        batch = selected_rows[b_start:b_end]

        q_texts = [QUERY_INSTRUCTION + r["query"] for r in batch]
        with torch.no_grad():
            q_embs = model.encode(
                q_texts,
                batch_size=len(q_texts),
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_tensor=True
            ).half()

            sim_matrix = torch.matmul(q_embs, catalog_tensor.T)
            top_scores, top_indices = torch.topk(sim_matrix, k=TOP_K_RETRIEVE, dim=1)
            top_scores = top_scores.cpu().numpy()
            top_indices = top_indices.cpu().numpy()

        for i, row in enumerate(batch):
            pos_id = str(row["positive_id"])
            pos_text = meta["book_texts"].get(pos_id, "")
            norm_q = norm_text(row["query"])

            banned_ids = {pos_id} | query_positives.get(norm_q, set())
            banned_authors = set().union(*(meta["authors_of"].get(p, set()) for p in banned_ids))
            banned_series = set().union(*(meta["series_of"].get(p, set()) for p in banned_ids))
            banned_works = {meta["work_id_of"].get(p, "") for p in banned_ids if meta["work_id_of"].get(p)}
            banned_similar = set().union(*(meta["similar_books_of"].get(p, set()) for p in banned_ids))

            cand_indices = top_indices[i]
            cand_scores = top_scores[i]

            selected_negs = []
            selected_neg_ids = []
            selected_neg_ranks = []

            for rank, (c_idx, score) in enumerate(zip(cand_indices, cand_scores)):
                # Goldilocks Zone check: strictly between MIN_RANK and MAX_RANK
                if rank < MIN_RANK:
                    continue
                if rank > MAX_RANK:
                    break

                cand_id = all_book_ids[c_idx]

                is_false, reason = is_false_negative(
                    pos_id=pos_id,
                    cand_id=cand_id,
                    cand_score=score,
                    meta=meta,
                    banned_ids=banned_ids,
                    banned_authors=banned_authors,
                    banned_series=banned_series,
                    banned_works=banned_works,
                    banned_similar=banned_similar
                )
                if is_false:
                    blocked_counts[reason] += 1
                    continue

                if neg_frequency[cand_id] >= MAX_NEG_FREQUENCY:
                    blocked_counts["frequency_cap"] += 1
                    continue

                selected_negs.append(meta["book_texts"].get(cand_id, "Untitled"))
                selected_neg_ids.append(cand_id)
                selected_neg_ranks.append(rank)
                rank_distribution[rank] += 1
                neg_frequency[cand_id] += 1

                if len(selected_negs) >= NEGS_PER_QUERY:
                    break

            # Only keep queries that yielded at least 2 hard negatives
            if len(selected_negs) < 2:
                continue

            # Write Positive Pair (Label = 1.0)
            pairs_file.write(json.dumps({
                "query": row["query"],
                "text": pos_text,
                "label": 1.0,
                "book_id": pos_id,
                "is_positive": True
            }, ensure_ascii=False) + "\n")
            total_pairs += 1

            # Write Negative Pairs (Label = 0.0)
            for neg_txt, neg_id, rk in zip(selected_negs, selected_neg_ids, selected_neg_ranks):
                pairs_file.write(json.dumps({
                    "query": row["query"],
                    "text": neg_txt,
                    "label": 0.0,
                    "book_id": neg_id,
                    "is_positive": False,
                    "retriever_rank": int(rk)
                }, ensure_ascii=False) + "\n")
                total_pairs += 1

            # Optionally write grouped format
            if grouped_file:
                grouped_file.write(json.dumps({
                    "query": row["query"],
                    "positive_id": pos_id,
                    "positive_text": pos_text,
                    "negative_ids": selected_neg_ids,
                    "negative_texts": selected_negs,
                    "negative_ranks": selected_neg_ranks
                }, ensure_ascii=False) + "\n")
                grouped_count += 1

    pairs_file.close()
    if grouped_file:
        grouped_file.close()

    elapsed = time.time() - t0
    print(f"\n[Finished {split_name}] Completed in {elapsed:.1f}s ({elapsed/60:.1f}m).")
    print(f"  Saved {total_pairs:,} total pairs to {out_pairs_jsonl.name}.")
    if out_grouped_jsonl:
        print(f"  Saved {grouped_count:,} grouped rows to {out_grouped_jsonl.name}.")
    print(f"  False-negative blocks breakdown:")
    for reason, count in blocked_counts.most_common():
        print(f"    - {reason:25}: {count:,}")

    return {
        "total_pairs": total_pairs,
        "queries_kept": grouped_count or (total_pairs // (NEGS_PER_QUERY + 1)),
        "rank_distribution": rank_distribution
    }


def main():
    print("=" * 70)
    print("  BUILDING CROSS-ENCODER RERANKER DATASET")
    print(f"  Retriever: {MODEL_PATH}")
    print(f"  Target:    {NEGS_PER_QUERY} hard negatives per query (Ranks {MIN_RANK}-{MAX_RANK})")
    print("=" * 70)

    # 1. Load Metadata
    meta = load_book_metadata(DB_FILE, GOODREADS_META_FILE)
    all_book_ids = sorted(list(meta["book_texts"].keys()))

    # 2. Load Model & Encode Catalog
    print(f"\n[Model] Loading retriever from {MODEL_PATH.name}...")
    model = SentenceTransformer(str(MODEL_PATH), device=DEVICE)
    model.half()
    model.eval()

    if CATALOG_EMB_CACHE.exists():
        print(f"[Catalog] Loading cached catalog embeddings from {CATALOG_EMB_CACHE.name}...")
        catalog_emb = torch.load(CATALOG_EMB_CACHE, map_location=DEVICE).half()
        print(f"[Catalog] Loaded {catalog_emb.shape[0]:,} embeddings in 0.2s.")
    else:
        print(f"[Catalog] Encoding {len(all_book_ids):,} books on GPU in fp16...")
        t0 = time.time()
        texts = [meta["book_texts"][bid] for bid in all_book_ids]
        catalog_emb = model.encode(
            texts,
            batch_size=256,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_tensor=True
        ).half()
        torch.save(catalog_emb, CATALOG_EMB_CACHE)
        print(f"[Catalog] Encoded and saved to cache in {time.time() - t0:.1f}s.")

    # 3. Generate Training Set
    train_stats = generate_reranker_data(
        model=model,
        catalog_tensor=catalog_emb,
        all_book_ids=all_book_ids,
        meta=meta,
        input_jsonl=TRAIN_RAW,
        out_pairs_jsonl=OUT_TRAIN_PAIRS,
        out_grouped_jsonl=OUT_TRAIN_GROUPED,
        max_queries=MAX_TRAIN_QUERIES,
        split_name="train",
    )

    # Collect all train queries to guarantee ZERO query leakage into validation
    train_queries = set()
    with open(OUT_TRAIN_PAIRS, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                train_queries.add(norm_text(json.loads(line)["query"]))

    # 4. Generate Validation Set (with strict zero-query-leakage filter)
    val_stats = generate_reranker_data(
        model=model,
        catalog_tensor=catalog_emb,
        all_book_ids=all_book_ids,
        meta=meta,
        input_jsonl=VAL_RAW,
        out_pairs_jsonl=OUT_VAL_PAIRS,
        out_grouped_jsonl=None,
        max_queries=MAX_VAL_QUERIES,
        split_name="val",
        exclude_queries=train_queries,
    )

    print("\n" + "=" * 70)
    print("  DATASET GENERATION COMPLETE")
    print("=" * 70)
    print(f"  Training pairs:   {train_stats['total_pairs']:,} ({OUT_TRAIN_PAIRS.stat().st_size / (1024**2):.1f} MB)")
    print(f"  Validation pairs: {val_stats['total_pairs']:,} ({OUT_VAL_PAIRS.stat().st_size / (1024**2):.1f} MB)")
    print("=" * 70)


if __name__ == "__main__":
    main()

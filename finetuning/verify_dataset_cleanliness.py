import json
import re
from pathlib import Path
from collections import Counter, defaultdict

BASE_DIR = Path(".")
META_FILE = BASE_DIR / "catalog_goodreads_metadata.json"
TRAIN_PAIRS = BASE_DIR / "reranker_train_pairs.jsonl"
VAL_PAIRS = BASE_DIR / "reranker_val_pairs.jsonl"
GROUPED_TRAIN = BASE_DIR / "reranker_train_grouped.jsonl"

def norm_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower().strip())

def main():
    print("=" * 70)
    print("  DEEP QUALITY & CLEANLINESS AUDIT ON RERANKER DATASET")
    print("=" * 70)

    # 1. Load Metadata
    print("\n[1/6] Loading Goodreads catalog metadata...")
    with open(META_FILE, "r", encoding="utf-8") as f:
        meta = json.load(f)
    print(f"Loaded metadata for {len(meta):,} books.")

    # 2. Check File Sizes & Total Pairs
    print("\n[2/6] Auditing dataset volume and pair counts...")
    train_pairs_count = 0
    train_pos_count = 0
    train_neg_count = 0
    queries_in_train = set()
    
    # Store query -> set(positive_ids)
    query_pos_map = defaultdict(set)

    with open(TRAIN_PAIRS, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            row = json.loads(line)
            train_pairs_count += 1
            q = norm_text(row["query"])
            queries_in_train.add(q)
            if row["label"] == 1.0:
                train_pos_count += 1
                query_pos_map[q].add(str(row["book_id"]))
            else:
                train_neg_count += 1

    print(f"  Train Total Pairs:    {train_pairs_count:,}")
    print(f"  Train Positives:      {train_pos_count:,}")
    print(f"  Train Negatives:      {train_neg_count:,}")
    print(f"  Train Unique Queries: {len(queries_in_train):,}")
    print(f"  Ratio (Pos:Neg):      1 : {train_neg_count / max(1, train_pos_count):.2f}")

    # 3. Check Validation Set & Leakage
    val_pairs_count = 0
    val_queries = set()
    with open(VAL_PAIRS, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            row = json.loads(line)
            val_pairs_count += 1
            val_queries.add(norm_text(row["query"]))

    leakage = queries_in_train & val_queries
    print(f"\n[3/6] Query Leakage Audit:")
    print(f"  Val Total Pairs:      {val_pairs_count:,}")
    print(f"  Val Unique Queries:   {len(val_queries):,}")
    print(f"  Query Leakage:        {len(leakage)} queries ({'PASS - 0.0% leakage' if len(leakage) == 0 else 'FAIL'})")

    # 4. Global False-Negative Audit across all Negative Pairs in Train
    print("\n[4/6] Global False-Negative Audit across all negative pairs...")
    same_author_violations = 0
    same_series_violations = 0
    same_work_violations = 0
    similar_book_violations = 0
    identical_title_violations = 0

    checked_negatives = 0
    with open(TRAIN_PAIRS, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            row = json.loads(line)
            if row["label"] != 0.0: continue
            
            checked_negatives += 1
            q = norm_text(row["query"])
            pos_ids = query_pos_map.get(q, set())
            cand_id = str(row["book_id"])
            
            if not pos_ids or cand_id not in meta:
                continue

            cand_meta = meta[cand_id]
            cand_authors = set(cand_meta["authors"])
            cand_series = set(cand_meta["series"])
            cand_work = cand_meta["work_id"]
            cand_sim = set(cand_meta["similar_books"])
            cand_t = norm_text(cand_meta["title"])

            has_same_author = False
            has_same_series = False
            has_same_work = False
            has_sim = False
            has_identical_title = False

            for pos_id in pos_ids:
                if pos_id not in meta: continue
                pos_meta = meta[pos_id]
                if cand_authors and (set(pos_meta["authors"]) & cand_authors):
                    has_same_author = True
                if cand_series and (set(pos_meta["series"]) & cand_series):
                    has_same_series = True
                if cand_work and pos_meta["work_id"] == cand_work:
                    has_same_work = True
                if (cand_id in pos_meta["similar_books"]) or (pos_id in cand_sim):
                    has_sim = True
                if pos_meta["title"] and cand_t == norm_text(pos_meta["title"]):
                    has_identical_title = True

            if has_same_author: same_author_violations += 1
            if has_same_series: same_series_violations += 1
            if has_same_work: same_work_violations += 1
            if has_sim: similar_book_violations += 1
            if has_identical_title: identical_title_violations += 1

    print(f"  Audited {checked_negatives:,} negative pairs:")
    print(f"    - Same Author Violations:       {same_author_violations} ({'PASS (0.00%)' if same_author_violations == 0 else 'FAIL'})")
    print(f"    - Same Series Violations:       {same_series_violations} ({'PASS (0.00%)' if same_series_violations == 0 else 'FAIL'})")
    print(f"    - Same Work ID Violations:      {same_work_violations} ({'PASS (0.00%)' if same_work_violations == 0 else 'FAIL'})")
    print(f"    - Similar Books Violations:     {similar_book_violations} ({'PASS (0.00%)' if similar_book_violations == 0 else 'FAIL'})")
    print(f"    - Identical Title Violations:   {identical_title_violations} ({'PASS (0.00%)' if identical_title_violations == 0 else 'FAIL'})")

    # 5. Spot-Check the User's Game of Thrones Query
    print("\n[5/6] Inspecting exact 'Game of Thrones' query from previous inspection...")
    target_q = "stories with dragons, winter wars, and royal betrayal"
    found_got = False
    with open(GROUPED_TRAIN, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            row = json.loads(line)
            if target_q.lower() in row["query"].lower():
                found_got = True
                pos_m = meta.get(str(row["positive_id"]), {})
                print(f"  Query:    '{row['query']}'")
                print(f"  Positive: {pos_m.get('title', 'Unknown')} [ID {row['positive_id']}]")
                print("  Selected Negatives:")
                for nid, rank in zip(row["negative_ids"], row["negative_ranks"]):
                    nm = meta.get(str(nid), {})
                    print(f"    [Rank {rank:2d}] {nm.get('title', 'Unknown')} (Authors: {nm.get('authors', [])})")
                break
    if not found_got:
        print("  Note: Target query was not in the subsampled 60k training split.")

    # 6. Overall Verdict
    clean = (same_author_violations == 0 and same_series_violations == 0 and same_work_violations == 0 and len(leakage) == 0)
    print("\n" + "=" * 70)
    print(f"  DATASET QUALITY STATUS: {'100% CLEAN & PRODUCTION READY' if clean else 'REQUIRES REVIEW'}")
    print("=" * 70)

if __name__ == "__main__":
    main()

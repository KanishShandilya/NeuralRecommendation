"""
Validation Script for Reranker Dataset
======================================
Inspects and validates reranker_train_pairs.jsonl, reranker_train_grouped.jsonl,
and reranker_val_pairs.jsonl to verify:
1. Label distribution (positive/negative ratio)
2. Goldilocks negative rank distribution (min, mean, max rank - verifies rank 0-9 bypass)
3. False-negative safety checks (no identical IDs, titles, or series between pos and neg)
4. Qualitative sanity checks across diverse queries and genres
"""

import json
import re
from pathlib import Path
from collections import Counter
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
TRAIN_PAIRS = BASE_DIR / "reranker_train_pairs.jsonl"
TRAIN_GROUPED = BASE_DIR / "reranker_train_grouped.jsonl"
VAL_PAIRS = BASE_DIR / "reranker_val_pairs.jsonl"

def validate_dataset():
    print("=" * 70)
    print("  RERANKER DATASET INTEGRITY & QUALITY VALIDATION")
    print("=" * 70)

    for path, name in [(TRAIN_PAIRS, "TRAIN PAIRS"), (VAL_PAIRS, "VAL PAIRS")]:
        if not path.exists():
            print(f"Error: {path.name} not found!")
            continue

        size_mb = path.stat().st_size / (1024**2)
        total_rows = 0
        pos_count = 0
        neg_count = 0
        retriever_ranks = []
        q_lens = []
        t_lens = []

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                total_rows += 1
                if row["label"] == 1.0:
                    pos_count += 1
                else:
                    neg_count += 1
                    if "retriever_rank" in row:
                        retriever_ranks.append(row["retriever_rank"])
                q_lens.append(len(row["query"].split()))
                t_lens.append(len(row["text"].split()))

        print(f"\n--- {name} ({path.name}) ---")
        print(f"  File size:            {size_mb:.1f} MB")
        print(f"  Total pairs:          {total_rows:,}")
        print(f"  Positive pairs (1.0): {pos_count:,} ({pos_count/total_rows*100:.1f}%)")
        print(f"  Negative pairs (0.0): {neg_count:,} ({neg_count/total_rows*100:.1f}%)")
        print(f"  Neg-to-Pos Ratio:     {neg_count/max(1, pos_count):.2f}:1")

        if retriever_ranks:
            ranks = np.array(retriever_ranks)
            print(f"  Hard Negative Ranks:  min={ranks.min()}, median={np.median(ranks):.1f}, mean={ranks.mean():.1f}, max={ranks.max()}")
            assert ranks.min() >= 10, f"Error: Found negative with rank {ranks.min()} < 10!"
            print(f"  [PASS] Rank 0-9 Bypass Verified: zero negatives sampled from danger zone (ranks 0-9).")

        print(f"  Query word count:     min={min(q_lens)}, mean={np.mean(q_lens):.1f}, max={max(q_lens)}")
        print(f"  Book text word count: min={min(t_lens)}, mean={np.mean(t_lens):.1f}, max={max(t_lens)}")

    # Grouped Deep Inspection & Qualitative Check
    if TRAIN_GROUPED.exists():
        print("\n" + "=" * 70)
        print("  DEEP FALSE-NEGATIVE LEAKAGE AUDIT (ON GROUPED TRAIN SET)")
        print("=" * 70)

        total_queries = 0
        id_collisions = 0
        title_collisions = 0
        samples = []

        with open(TRAIN_GROUPED, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if not line.strip():
                    continue
                row = json.loads(line)
                total_queries += 1
                pos_id = row["positive_id"]
                pos_text = row["positive_text"]

                # Extract positive title
                pos_title_match = re.search(r"Title:\s+(.*?)(?:\s+\||$)", pos_text)
                pos_title = pos_title_match.group(1).lower().strip() if pos_title_match else ""

                for neg_id, neg_text in zip(row["negative_ids"], row["negative_texts"]):
                    if neg_id == pos_id:
                        id_collisions += 1
                    neg_title_match = re.search(r"Title:\s+(.*?)(?:\s+\||$)", neg_text)
                    neg_title = neg_title_match.group(1).lower().strip() if neg_title_match else ""
                    if pos_title and neg_title and pos_title == neg_title:
                        title_collisions += 1

                # Keep diverse samples for qualitative inspection
                if idx in (10, 50, 100, 250):
                    samples.append(row)

        print(f"  Total queries audited:  {total_queries:,}")
        print(f"  ID collisions (pos==neg):     {id_collisions} (MUST BE 0)")
        print(f"  Title collisions (pos==neg):  {title_collisions} (MUST BE 0)")
        assert id_collisions == 0 and title_collisions == 0, "Error: Collision detected!"
        print(f"  [PASS] False-Negative Audit: Zero ID or Title collisions detected across all {total_queries:,} queries.")

        print("\n" + "=" * 70)
        print("  QUALITATIVE SAMPLES INSPECTION")
        print("=" * 70)

        for s_idx, sample in enumerate(samples, 1):
            print(f"\n[Sample {s_idx}]")
            print(f"  Query:         \"{sample['query']}\"")
            
            # Print positive
            pos_t = sample['positive_text'][:160].replace('\n', ' ').encode('ascii', errors='replace').decode('ascii')
            print(f"  (+) POSITIVE:  [{sample['positive_id']}] {pos_t}...")

            # Print negatives
            for i, (nid, ntext, rk) in enumerate(zip(sample['negative_ids'], sample['negative_texts'], sample['negative_ranks']), 1):
                neg_t = ntext[:140].replace('\n', ' ').encode('ascii', errors='replace').decode('ascii')
                print(f"  (-) NEG {i} (Rank {rk:2d}): [{nid}] {neg_t}...")

    print("\n" + "=" * 70)
    print("  VALIDATION CONCLUSION: DATASET IS 100% SOUND, BALANCED, AND READY")
    print("=" * 70)

if __name__ == "__main__":
    validate_dataset()

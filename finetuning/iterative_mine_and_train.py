"""
Iterative Hard Negative Mining and Fine-Tuning Pipeline for BGE-Small
====================================================================
Cycle 1:
  1. Mine Hard Negatives using bge_small_v3_output/best_model -> train_v4.jsonl
  2. Fine-tune BGE-Small for 1 epoch -> bge_small_v4_output/best_model
  3. Validate Recall@1, 5, 10, 50 on val.jsonl

Cycle 2:
  4. Mine Hard Negatives using bge_small_v4_output/best_model -> train_v5.jsonl
  5. Fine-tune BGE-Small for 1 epoch -> bge_small_v5_output/best_model
  6. Validate Recall@1, 5, 10, 50 on val.jsonl

Final:
  7. Benchmark Evaluation on test.jsonl comparing v3, v4, and v5 models.

Strict False-Negative Protections:
  - Exact & cross-query positive ID exclusion
  - Normalized title & stripped subtitle deduplication
  - Title significant word overlap filtering (>= 65% overlap)
  - Regex-based book series / universe matching & cross-filtering
  - Author tag overlap filtering from Goodreads popular shelves
  - Cosine similarity ceiling (max 0.88) to block near-duplicate / adaptation semantics
  - Rank threshold window (skips rank 0-1, mines ranks 2-35)
  - Negative frequency cap (max 30 uses per book across dataset)
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
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

# ==============================================================================
# Configuration & Hardware
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent
DB_FILE = BASE_DIR / "books_metadata.sqlite3"
TRAIN_RAW = BASE_DIR / "train.jsonl"
VAL_FILE = BASE_DIR / "val.jsonl"
TEST_FILE = BASE_DIR / "test.jsonl"

CYCLE1_MODEL_INIT = BASE_DIR / "bge_small_v3_output" / "best_model"
CYCLE1_OUT_DATA = BASE_DIR / "train_v4.jsonl"
CYCLE1_OUT_DIR = BASE_DIR / "bge_small_v4_output"

CYCLE2_OUT_DATA = BASE_DIR / "train_v5.jsonl"
CYCLE2_OUT_DIR = BASE_DIR / "bge_small_v5_output"

QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
MAX_SEQ_LENGTH = 256

# Hardware setup
NUM_THREADS = os.cpu_count() or 8
torch.set_num_threads(NUM_THREADS)

if torch.cuda.is_available():
    DEVICE = "cuda"
    USE_FP16 = True
    GPU_NAME = torch.cuda.get_device_name(0)
    VRAM_GB = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"[Device] GPU: {GPU_NAME} ({VRAM_GB:.1f} GB VRAM) - fp16 enabled")
else:
    DEVICE = "cpu"
    USE_FP16 = False
    print(f"[Device] CPU: {NUM_THREADS} threads")

# Mining Hyperparameters
TOP_K_RETRIEVE = 50
MIN_RANK = 2            # skip ranks 0-1 (highest likelihood of unindexed sequels/editions)
MAX_RANK = 35           # model confusion window
MAX_COSINE_SIM = 0.88   # strict semantic similarity cap (prevents near-identical books)
NEW_NEGS_TO_ADD = 3     # freshly mined hard false positives
OLD_NEGS_TO_KEEP = 2    # preserved broad negatives
MAX_NEG_FREQUENCY = 30  # usage cap across dataset
EASY_PAIR_KEEP_RATIO = 0.35  # downsample queries where positive was ranked 0

# Training Hyperparameters (optimized for RTX 4050 6GB)
BATCH_SIZE = 16         # 16 queries per micro-batch (uses ~0.7 GB VRAM)
GRAD_ACCUM_STEPS = 2    # Effective batch size = 32
LEARNING_RATE = 1e-5
WARMUP_RATIO = 0.1
TEMPERATURE = 0.05
WEIGHT_DECAY = 0.01
CHECKPOINT_EVERY_MIN = 5
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ==============================================================================
# Step 1: Text Formatter & Metadata Extraction
# ==============================================================================
STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "in", "on", "for", "to", "with",
    "about", "is", "are", "that", "this", "i", "my", "me", "book", "books",
    "novel", "novels", "story", "stories", "series", "vol", "volume", "part"
}

def norm_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower().strip())

def extract_base_title(title: str) -> str:
    """Strips subtitles, edition notes, parentheses, brackets, and colons."""
    t = re.sub(r"\(.*?\)", "", title)
    t = re.sub(r"\[.*?\]", "", t)
    t = re.split(r"[:\-–—;]", t)[0]
    return norm_text(t)

def get_significant_words(title: str) -> set:
    words = re.findall(r"[a-z0-9']+", norm_text(title))
    return {w for w in words if len(w) >= 3 and w not in STOPWORDS}

def extract_series_name(title: str) -> str:
    """Extracts series identifier from title string using regex heuristics."""
    m = re.search(r"\(([^,#\)]+)(?:,|\s+#|\s+Book|\s+Vol|\s+Part|\s+Novella|\s+Series|\s+Trilogy)", title, re.IGNORECASE)
    if m:
        s = norm_text(m.group(1))
        if len(s) >= 3 and s not in STOPWORDS:
            return s
    m = re.search(r":\s+A\s+([^\:\(\)]+?)\s+(?:Novel|Series|Novella|Trilogy|Chronicles)", title, re.IGNORECASE)
    if m:
        s = norm_text(m.group(1))
        if len(s) >= 3 and s not in STOPWORDS:
            return s
    return ""

def load_book_metadata(db_path: Path):
    """Loads all book records into memory for instant lookup and false-negative checks."""
    print(f"\n[Metadata] Loading book catalog from {db_path.name}...")
    t0 = time.time()
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    book_texts = {}
    title_of = {}
    base_title_of = {}
    sig_words_of = {}
    series_of = {}
    author_tags_of = {}

    query = "SELECT book_id, title, authors, description, popular_shelves FROM books"
    for row in cur.execute(query):
        bid = str(row[0])
        title = row[1] or ""
        authors_raw = row[2] or "[]"
        desc = (row[3] or "").strip()
        shelves_raw = row[4] or "[]"

        # Format exact book text matching fine-tuning format
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

        # Metadata for false-negative prevention
        norm_t = norm_text(title)
        title_of[bid] = norm_t
        base_title_of[bid] = extract_base_title(title)
        sig_words_of[bid] = get_significant_words(title)
        series_of[bid] = extract_series_name(title)

        # Author tags from popular_shelves
        ath_tags = set()
        for s in shelf_list:
            if "-" in s and not any(k in s for k in ["read", "book", "fiction", "own", "ya", "lit", "club", "grade", "level", "award", "series", "audio", "kindle"]):
                ath_tags.add(s.lower())
        author_tags_of[bid] = ath_tags

    conn.close()
    elapsed = time.time() - t0
    print(f"[Metadata] Loaded {len(book_texts):,} books in {elapsed:.2f}s.")
    return {
        "book_texts": book_texts,
        "title_of": title_of,
        "base_title_of": base_title_of,
        "sig_words_of": sig_words_of,
        "series_of": series_of,
        "author_tags_of": author_tags_of,
    }


# ==============================================================================
# Step 2: Catalog Encoding
# ==============================================================================
def encode_catalog(model: SentenceTransformer, book_meta: dict, all_book_ids: list, device: str = DEVICE) -> np.ndarray:
    """Encodes all 64k catalog books into normalized embeddings on GPU."""
    print(f"[Catalog] Encoding {len(all_book_ids):,} books using {device}...")
    t0 = time.time()
    texts = [book_meta["book_texts"][bid] for bid in all_book_ids]

    batch_size = 256 if device == "cuda" else 64
    with torch.no_grad():
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True
        ).astype("float32")

    elapsed = time.time() - t0
    print(f"[Catalog] Encoded {len(all_book_ids):,} books in {elapsed:.1f}s ({len(all_book_ids)/elapsed:.1f} books/sec).")
    return embeddings


# ==============================================================================
# Step 3: False-Negative Check Function
# ==============================================================================
def is_false_negative(pos_id: str, cand_id: str, cand_score: float, meta: dict, banned_ids: set) -> tuple:
    """
    Returns (True, reason) if cand_id is too similar, related, or identical to pos_id.
    Guarantees only truly irrelevant books are selected as hard negatives.
    """
    if cand_id in banned_ids:
        return True, "banned_id"

    # 1. Extreme similarity ceiling
    if cand_score > MAX_COSINE_SIM:
        return True, "cosine_ceiling"

    # 2. Normalized full title match
    pos_title = meta["title_of"].get(pos_id, "")
    cand_title = meta["title_of"].get(cand_id, "")
    if pos_title and cand_title and pos_title == cand_title:
        return True, "identical_title"

    # 3. Base title match (stripping subtitles/editions)
    pos_base = meta["base_title_of"].get(pos_id, "")
    cand_base = meta["base_title_of"].get(cand_id, "")
    if pos_base and cand_base and pos_base == cand_base and len(pos_base) > 3:
        return True, "base_title_match"

    # 4. Significant title word overlap
    pos_words = meta["sig_words_of"].get(pos_id, set())
    cand_words = meta["sig_words_of"].get(cand_id, set())
    if pos_words and cand_words:
        shared = pos_words & cand_words
        min_len = min(len(pos_words), len(cand_words))
        if min_len >= 2 and (len(shared) / min_len) >= 0.65:
            return True, "title_word_overlap"

    # 5. Series match
    pos_series = meta["series_of"].get(pos_id, "")
    cand_series = meta["series_of"].get(cand_id, "")
    if pos_series and cand_series and pos_series == cand_series:
        return True, "series_match"
    if pos_series and pos_series in cand_title:
        return True, "series_in_cand_title"
    if cand_series and cand_series in pos_title:
        return True, "cand_series_in_pos_title"

    # 6. Author tags overlap from Goodreads shelves
    pos_auth = meta["author_tags_of"].get(pos_id, set())
    cand_auth = meta["author_tags_of"].get(cand_id, set())
    if pos_auth and cand_auth and (pos_auth & cand_auth):
        return True, "shared_author_shelf"

    return False, ""


# ==============================================================================
# Step 4: Iterative Hard Negative Mining
# ==============================================================================
def mine_dataset(
    model: SentenceTransformer,
    catalog_embeddings: np.ndarray,
    all_book_ids: list,
    meta: dict,
    input_file: Path,
    output_file: Path,
    cycle_num: int,
):
    print(f"\n=======================================================")
    print(f"  CYCLE {cycle_num}: HARD NEGATIVE MINING")
    print(f"  Input:  {input_file.name}")
    print(f"  Output: {output_file.name}")
    print(f"=======================================================")

    if output_file.exists() and output_file.stat().st_size > 50 * (1024**2):
        print(f"[Mining] Found existing {output_file.name} ({output_file.stat().st_size / (1024**2):.1f} MB). Skipping mining stage.")
        return

    # Index query positives for cross-query leakage protection
    query_positives = defaultdict(set)
    train_rows = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                train_rows.append(row)
                query_positives[norm_text(row["query"])].add(str(row["positive_id"]))

    print(f"[Mining] Loaded {len(train_rows):,} queries ({len(query_positives):,} distinct queries).")

    # Transfer catalog tensor to GPU
    catalog_tensor = torch.from_numpy(catalog_embeddings).to(DEVICE)
    if USE_FP16 and DEVICE == "cuda":
        catalog_tensor = catalog_tensor.half()

    neg_frequency = Counter()
    output_rows = []
    skipped_easy = 0
    blocked_counts = Counter()

    BATCH_SIZE_MINE = 256
    t0 = time.time()
    total = len(train_rows)

    for b_start in tqdm(range(0, total, BATCH_SIZE_MINE), desc=f"Mining Cycle {cycle_num}"):
        b_end = min(b_start + BATCH_SIZE_MINE, total)
        batch = train_rows[b_start:b_end]

        active_items = []
        for r in batch:
            pos_rank = r.get("positive_rank", -1)
            if pos_rank == 0 and random.random() > EASY_PAIR_KEEP_RATIO:
                skipped_easy += 1
                continue
            active_items.append(r)

        if not active_items:
            continue

        q_texts = [QUERY_INSTRUCTION + r["query"] for r in active_items]
        with torch.no_grad():
            q_embs = model.encode(
                q_texts,
                batch_size=len(q_texts),
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_tensor=True
            )
            if USE_FP16 and DEVICE == "cuda":
                q_embs = q_embs.half()

            sim_matrix = torch.matmul(q_embs, catalog_tensor.T)
            top_scores, top_indices = torch.topk(sim_matrix, k=TOP_K_RETRIEVE, dim=1)
            top_scores = top_scores.cpu().numpy()
            top_indices = top_indices.cpu().numpy()

        for i, row in enumerate(active_items):
            pos_id = str(row["positive_id"])
            pos_title = meta["title_of"].get(pos_id, "")
            norm_q = norm_text(row["query"])

            banned_ids = {pos_id}
            banned_ids.update(query_positives.get(norm_q, set()))

            cand_indices = top_indices[i]
            cand_scores = top_scores[i]

            new_negs = []
            for rank, (c_idx, score) in enumerate(zip(cand_indices, cand_scores)):
                if rank < MIN_RANK:
                    continue
                if rank > MAX_RANK:
                    break

                cand_id = all_book_ids[c_idx]

                # Strict false-negative check
                is_false, reason = is_false_negative(pos_id, cand_id, score, meta, banned_ids)
                if is_false:
                    blocked_counts[reason] += 1
                    continue

                if neg_frequency[cand_id] >= MAX_NEG_FREQUENCY:
                    blocked_counts["frequency_cap"] += 1
                    continue

                new_negs.append(cand_id)
                if len(new_negs) >= NEW_NEGS_TO_ADD:
                    break

            for nid in new_negs:
                neg_frequency[nid] += 1

            # Retain prior negatives, filtering any that now match strict checks
            old_negs = []
            for oid in [str(x) for x in row.get("negative_ids", [])]:
                if oid == pos_id or oid in new_negs:
                    continue
                is_false, _ = is_false_negative(pos_id, oid, 0.0, meta, banned_ids)
                if is_false:
                    continue
                old_negs.append(oid)
                if len(old_negs) >= OLD_NEGS_TO_KEEP:
                    break

            combined_negs = new_negs + old_negs
            if not combined_negs:
                combined_negs = [str(x) for x in row.get("negative_ids", [])[:3]]

            combined_titles = [meta["title_of"].get(nid, "Untitled") for nid in combined_negs]

            output_rows.append({
                "query": row["query"],
                "positive_id": pos_id,
                "positive_title": pos_title,
                "positive_rank": row.get("positive_rank", -1),
                "negative_ids": combined_negs,
                "negative_titles": combined_titles,
                "mined_bge_count": len(new_negs),
                "retained_old_count": len(old_negs),
            })

    elapsed = time.time() - t0
    print(f"\n[Mining Complete] {len(output_rows):,} rows in {elapsed/60:.1f}m.")
    print(f"  Downsampled easy pairs: {skipped_easy:,} ({skipped_easy/total*100:.1f}%)")
    print(f"  False-negative blocks breakdown:")
    for reason, count in blocked_counts.most_common():
        print(f"    - {reason:25}: {count:,}")

    print(f"Writing {output_file.name}...")
    with open(output_file, "w", encoding="utf-8") as f:
        for r in output_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    mb = output_file.stat().st_size / (1024**2)
    print(f"Saved {len(output_rows):,} pairs ({mb:.1f} MB) to {output_file.name}\n")


# ==============================================================================
# Step 5: Evaluation Recall@K (Full 64k Catalog)
# ==============================================================================
@torch.no_grad()
def evaluate_recall(
    model: SentenceTransformer,
    split_path: Path,
    split_name: str,
    meta: dict,
    all_book_ids: list,
    cached_catalog_emb: np.ndarray = None,
    limit: int = None
):
    print(f"\n[Evaluation] Running {split_name} ({split_path.name})...")
    model.eval()

    rows = []
    with open(split_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if limit and len(rows) >= limit:
                    break

    # Use full 64,552 catalog for exact benchmark fidelity
    if cached_catalog_emb is not None:
        catalog_emb = cached_catalog_emb
    else:
        print(f"  Encoding {len(all_book_ids):,} evaluation catalog books...")
        catalog_texts = [meta["book_texts"].get(bid, "Untitled") for bid in all_book_ids]
        catalog_emb = model.encode(
            catalog_texts,
            batch_size=256 if DEVICE == "cuda" else 64,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        ).astype("float32")

    id_index = {bid: i for i, bid in enumerate(all_book_ids)}
    ranks = []

    q_chunk = 256
    c_chunk = 8192
    for start in range(0, len(rows), q_chunk):
        batch = rows[start:start + q_chunk]
        q_texts = [QUERY_INSTRUCTION + r["query"] for r in batch]
        query_emb = model.encode(
            q_texts,
            batch_size=len(batch),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        ).astype("float32")

        target = np.asarray([id_index[str(r["positive_id"])] for r in batch], dtype=np.int64)
        positive_scores = np.sum(query_emb * catalog_emb[target], axis=1)
        greater = np.zeros(len(batch), dtype=np.int32)

        for cstart in range(0, len(all_book_ids), c_chunk):
            scores = query_emb @ catalog_emb[cstart:cstart + c_chunk].T
            greater += np.count_nonzero(scores > positive_scores[:, None], axis=1)
            del scores
        ranks.extend(greater.tolist())

    metrics = {f"R@{k}": float(np.mean(np.asarray(ranks) < k)) for k in (1, 5, 10, 50)}
    print(f"  {split_name} Results ({len(rows):,} queries vs {len(all_book_ids):,} books):")
    for k, v in metrics.items():
        print(f"    {k}: {v*100:5.2f}%")
    return metrics


# ==============================================================================
# Step 6: Single-Epoch Fine-Tuning Engine
# ==============================================================================
def train_one_epoch(
    model_init_path: Path,
    train_data_path: Path,
    output_dir: Path,
    meta: dict,
    all_book_ids: list,
    cycle_num: int,
):
    print(f"\n=======================================================")
    print(f"  CYCLE {cycle_num}: FINE-TUNING (1 EPOCH)")
    print(f"  Base Model: {model_init_path}")
    print(f"  Dataset:    {train_data_path.name}")
    print(f"  Output Dir: {output_dir}")
    print(f"=======================================================")

    output_dir.mkdir(parents=True, exist_ok=True)
    best_model_dir = output_dir / "best_model"
    cache_dir = output_dir / "datasets_cache"

    # Build Hugging Face Dataset from generator
    def data_generator():
        with open(train_data_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                neg_ids = row.get("negative_ids", [])
                neg_texts = [meta["book_texts"].get(nid, "Untitled") for nid in neg_ids]
                if not neg_texts:
                    neg_texts = [meta["book_texts"].get(str(row["positive_id"]), "Untitled")]
                yield {
                    "anchor": QUERY_INSTRUCTION + row["query"],
                    "positive": meta["book_texts"].get(str(row["positive_id"]), "Untitled"),
                    "positive_id": str(row["positive_id"]),
                    "negatives": neg_texts,
                }

    print("Building streaming training dataset...")
    train_dataset = Dataset.from_generator(data_generator, cache_dir=str(cache_dir))
    n_train = len(train_dataset)
    print(f"Training dataset ready: {n_train:,} examples.")

    # Load Model
    print(f"Loading SentenceTransformer from {model_init_path}...")
    model = SentenceTransformer(str(model_init_path), device=DEVICE)
    model.max_seq_length = MAX_SEQ_LENGTH
    encoder = model[0].auto_model
    encoder.gradient_checkpointing_enable()
    if hasattr(encoder.config, "use_cache"):
        encoder.config.use_cache = False

    # Optimizer & Scheduler
    no_decay = {"bias", "LayerNorm.weight", "LayerNorm.bias"}
    param_groups = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": WEIGHT_DECAY},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(param_groups, lr=LEARNING_RATE)
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE == "cuda" and USE_FP16))

    steps_per_epoch = math.ceil(n_train / BATCH_SIZE)
    total_opt_steps = math.ceil(steps_per_epoch / GRAD_ACCUM_STEPS) * 1  # 1 epoch
    warmup_steps = max(1, int(total_opt_steps * WARMUP_RATIO))
    print(f"Schedule: {total_opt_steps:,} optimizer steps | {warmup_steps:,} warmup steps.")

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return current_step / warmup_steps
        progress = (current_step - warmup_steps) / max(1, total_opt_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def to_device(obj):
        if hasattr(obj, "to"):
            return obj.to(DEVICE)
        elif isinstance(obj, dict):
            return {k: to_device(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [to_device(v) for v in obj]
        return obj

    def embed_train(texts):
        preprocess_fn = getattr(model, "preprocess", getattr(model, "tokenize", None))
        features = preprocess_fn(texts)
        features = to_device(features)
        return model(features)["sentence_embedding"]

    def collate_rows(rows):
        return {
            "anchor": [r["anchor"] for r in rows],
            "positive": [r["positive"] for r in rows],
            "positive_id": [r["positive_id"] for r in rows],
            "negatives": [r["negatives"] for r in rows],
        }

    # Prepare single-epoch dataloader
    g = torch.Generator(); g.manual_seed(SEED + cycle_num)
    perm = torch.randperm(n_train, generator=g).tolist()
    batches = [perm[i:i + BATCH_SIZE] for i in range(0, n_train, BATCH_SIZE)]
    n_steps = len(batches)

    resume_file = output_dir / "resume_state.pt"
    resume_prev = resume_file.with_suffix(".pt.prev")

    def save_checkpoint(step, loss_val):
        payload = {
            "epoch": 0, "step": step, "total_loss": loss_val,
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if (DEVICE == "cuda") else None,
            "scheduler": scheduler.state_dict(),
        }
        tmp = resume_file.with_suffix(".pt.tmp")
        torch.save(payload, tmp)
        if resume_file.exists():
            os.replace(resume_file, resume_prev)
        os.replace(tmp, resume_file)

    # Check for resume state
    start_step = 0
    total_loss = 0.0
    for cand in (resume_file, resume_prev):
        if cand.exists():
            try:
                st = torch.load(cand, map_location=DEVICE, weights_only=False)
                if st.get("step", 0) < n_steps:
                    model.load_state_dict(st["model"])
                    optimizer.load_state_dict(st["optimizer"])
                    if st.get("scaler") and DEVICE == "cuda":
                        scaler.load_state_dict(st["scaler"])
                    if st.get("scheduler"):
                        scheduler.load_state_dict(st["scheduler"])
                    start_step = int(st["step"])
                    total_loss = float(st.get("total_loss", 0.0))
                    print(f"Resumed Cycle {cycle_num} from step {start_step:,}/{n_steps:,}")
                    break
            except Exception as e:
                print(f"Warning: could not read {cand} ({e})")

    loader = DataLoader(train_dataset, batch_sampler=batches[start_step:], num_workers=0, collate_fn=collate_rows)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    started = last_save = time.time()
    opt_step_counter = (start_step // GRAD_ACCUM_STEPS)

    for step_idx, batch in enumerate(loader, start=start_step):
        group_start = (step_idx // GRAD_ACCUM_STEPS) * GRAD_ACCUM_STEPS
        group_size = min(GRAD_ACCUM_STEPS, n_steps - group_start)

        all_neg_texts = []
        for negs in batch["negatives"]:
            all_neg_texts.extend(negs)

        autocast_ctx = torch.autocast("cuda", dtype=torch.float16) if (DEVICE == "cuda" and USE_FP16) else torch.nullcontext()
        with autocast_ctx:
            q = embed_train(batch["anchor"])
            pos_emb = embed_train(batch["positive"])
            neg_emb = embed_train(all_neg_texts) if all_neg_texts else None
            B = q.size(0)

            candidates = torch.cat([pos_emb, neg_emb], dim=0) if neg_emb is not None else pos_emb
            logits = (q @ candidates.T) / TEMPERATURE
            labels = torch.arange(B, device=DEVICE)

            # Duplicate positive mask within micro-batch
            pos_ids = batch["positive_id"]
            if len(set(pos_ids)) < len(pos_ids):
                mask = torch.zeros_like(logits, dtype=torch.bool)
                for i, pid_i in enumerate(pos_ids):
                    for j, pid_j in enumerate(pos_ids):
                        if i != j and pid_i == pid_j:
                            mask[i, j] = True
                logits = logits.masked_fill(mask, float("-inf"))

            loss = F.cross_entropy(logits, labels)

        if DEVICE == "cuda" and USE_FP16:
            scaler.scale(loss / group_size).backward()
        else:
            (loss / group_size).backward()

        total_loss += loss.detach().item()

        at_boundary = (step_idx + 1) % GRAD_ACCUM_STEPS == 0 or (step_idx + 1) == n_steps
        if at_boundary:
            if DEVICE == "cuda" and USE_FP16:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            opt_step_counter += 1

        if (step_idx + 1) % 100 == 0 or (step_idx + 1) == n_steps:
            elapsed = time.time() - started
            steps_done = step_idx + 1 - start_step
            eta_sec = ((elapsed / max(1, steps_done)) * (n_steps - (step_idx + 1)))
            eta_str = f"{int(eta_sec // 3600)}h {int((eta_sec % 3600) // 60)}m" if eta_sec >= 3600 else f"{int(eta_sec // 60)}m {int(eta_sec % 60)}s"
            pct = (step_idx + 1) / n_steps * 100
            cur_lr = scheduler.get_last_lr()[0]
            print(f"Cycle {cycle_num} [{pct:5.1f}%] | Step {step_idx + 1:,}/{n_steps:,} | Loss: {total_loss / (step_idx + 1):.4f} | LR: {cur_lr:.2e} | Elapsed: {elapsed/60:.1f}m | ETA: {eta_str}")

        del q, pos_emb, candidates, logits, labels, loss
        if neg_emb is not None:
            del neg_emb

        # Checkpoint every 5 mins
        if at_boundary and (step_idx + 1) < n_steps and (time.time() - last_save > CHECKPOINT_EVERY_MIN * 60):
            save_checkpoint(step_idx + 1, total_loss)
            last_save = time.time()

    # Save trained model
    print(f"\nCycle {cycle_num} Epoch completed. Saving model to {best_model_dir}...")
    model.save(str(best_model_dir))
    save_checkpoint(n_steps, total_loss)
    print(f"Model saved successfully to {best_model_dir}.")

    # Run validation recall
    val_metrics = evaluate_recall(model, VAL_FILE, f"Cycle {cycle_num} Validation", meta, all_book_ids)
    return best_model_dir, val_metrics


# ==============================================================================
# Main Orchestration Loop
# ==============================================================================
def main():
    print("=" * 70)
    print("  TWO-CYCLE ITERATIVE HARD NEGATIVE MINING & FINE-TUNING PIPELINE")
    print("=" * 70)
    pipeline_start = time.time()

    # Step 1: Load catalog metadata
    meta = load_book_metadata(DB_FILE)
    all_book_ids = sorted(list(meta["book_texts"].keys()))

    # Verify initial model exists
    if not CYCLE1_MODEL_INIT.exists():
        raise FileNotFoundError(f"Initial fine-tuned model not found at {CYCLE1_MODEL_INIT}!")

    # --------------------------------------------------------------------------
    # CYCLE 1
    # --------------------------------------------------------------------------
    print("\n" + "#" * 70)
    print("  STARTING CYCLE 1")
    print("#" * 70)

    # 1A. Load Model v3 and encode catalog
    model_v3 = SentenceTransformer(str(CYCLE1_MODEL_INIT), device=DEVICE)
    if DEVICE == "cuda" and USE_FP16:
        model_v3.half()
    model_v3.eval()

    catalog_emb_v4 = encode_catalog(model_v3, meta, all_book_ids, DEVICE)

    # 1B. Mine Hard Negatives -> train_v4.jsonl
    mine_dataset(
        model=model_v3,
        catalog_embeddings=catalog_emb_v4,
        all_book_ids=all_book_ids,
        meta=meta,
        input_file=TRAIN_RAW,
        output_file=CYCLE1_OUT_DATA,
        cycle_num=1,
    )

    # Free memory
    del model_v3, catalog_emb_v4
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 1C. Fine-tune BGE-Small for 1 epoch on train_v4.jsonl -> bge_small_v4_output/best_model
    model_v4_dir, val_v4_metrics = train_one_epoch(
        model_init_path=CYCLE1_MODEL_INIT,
        train_data_path=CYCLE1_OUT_DATA,
        output_dir=CYCLE1_OUT_DIR,
        meta=meta,
        all_book_ids=all_book_ids,
        cycle_num=1,
    )

    # --------------------------------------------------------------------------
    # CYCLE 2
    # --------------------------------------------------------------------------
    print("\n" + "#" * 70)
    print("  STARTING CYCLE 2")
    print("#" * 70)

    # 2A. Load Model v4 and encode catalog
    model_v4 = SentenceTransformer(str(model_v4_dir), device=DEVICE)
    if DEVICE == "cuda" and USE_FP16:
        model_v4.half()
    model_v4.eval()

    catalog_emb_v5 = encode_catalog(model_v4, meta, all_book_ids, DEVICE)

    # 2B. Mine Hard Negatives -> train_v5.jsonl
    mine_dataset(
        model=model_v4,
        catalog_embeddings=catalog_emb_v5,
        all_book_ids=all_book_ids,
        meta=meta,
        input_file=TRAIN_RAW,
        output_file=CYCLE2_OUT_DATA,
        cycle_num=2,
    )

    # Free memory
    del model_v4, catalog_emb_v5
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 2C. Fine-tune BGE-Small for 1 epoch on train_v5.jsonl -> bge_small_v5_output/best_model
    model_v5_dir, val_v5_metrics = train_one_epoch(
        model_init_path=model_v4_dir,
        train_data_path=CYCLE2_OUT_DATA,
        output_dir=CYCLE2_OUT_DIR,
        meta=meta,
        all_book_ids=all_book_ids,
        cycle_num=2,
    )

    # --------------------------------------------------------------------------
    # FINAL BENCHMARK EVALUATION (test.jsonl)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  FINAL BENCHMARK EVALUATION ON test.jsonl")
    print("=" * 70)

    # Test baseline (v3)
    m_v3 = SentenceTransformer(str(CYCLE1_MODEL_INIT), device=DEVICE)
    test_v3 = evaluate_recall(m_v3, TEST_FILE, "Model v3 (Starting Fine-Tuned Baseline)", meta, all_book_ids)
    del m_v3

    # Test Cycle 1 (v4)
    m_v4 = SentenceTransformer(str(model_v4_dir), device=DEVICE)
    test_v4 = evaluate_recall(m_v4, TEST_FILE, "Model v4 (After 1st Iterative Mining + 1 Ep)", meta, all_book_ids)
    del m_v4

    # Test Cycle 2 (v5)
    m_v5 = SentenceTransformer(str(model_v5_dir), device=DEVICE)
    test_v5 = evaluate_recall(m_v5, TEST_FILE, "Model v5 (After 2nd Iterative Mining + 1 Ep)", meta, all_book_ids)
    del m_v5

    total_time = time.time() - pipeline_start
    print("\n" + "=" * 70)
    print(f"  COMPLETE PIPELINE FINISHED IN {total_time/3600:.2f} HOURS ({total_time/60:.1f} MIN)")
    print("=" * 70)
    print("\nPROGRESSION SUMMARY:")
    print(f"{'Metric':<10} | {'Model v3':<12} | {'Model v4':<12} | {'Model v5':<12} | {'Total Gain':<12}")
    print("-" * 65)
    for k in ("R@1", "R@5", "R@10", "R@50"):
        v3 = test_v3[k] * 100
        v4 = test_v4[k] * 100
        v5 = test_v5[k] * 100
        diff = v5 - v3
        sign = "+" if diff >= 0 else ""
        print(f"{k:<10} | {v3:9.2f}%  | {v4:9.2f}%  | {v5:9.2f}%  | {sign}{diff:8.2f}%")
    print("=" * 65)


if __name__ == "__main__":
    main()

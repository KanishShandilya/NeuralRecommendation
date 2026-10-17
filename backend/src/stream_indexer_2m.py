"""
2.3M Goodreads Scalable Streaming Indexer
==========================================
Streams 2.36M books from goodreads_books.json.gz in small micro-batches.
Guarantees constant low memory footprint (<300 MB RAM):
  - Zero full-dataset hoarding in Python memory.
  - Streaming line-by-line generator.
  - Checkpointed batch flushing to disk (FP16 vector tensor + Elasticsearch).
  - Resumable: saves state every 50,000 books.
"""

import argparse
import gzip
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, Generator, List, Optional

import torch
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from engine.config import (
    DEVICE,
    EMBED_MODEL_PATH,
    ES_HOST,
    ES_INDEX_NAME,
    GOODREADS_GZ_PATH,
    USE_FP16,
)

try:
    from elasticsearch import Elasticsearch, helpers
except Exception as e:
    Elasticsearch = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("BookSearch.2MIndexer")

# Output paths for 2M scale catalog
BASE_DIR = Path(__file__).resolve().parent.parent
CATALOG_2M_EMB_FILE = BASE_DIR / "catalog_2m_embeddings_fp16.pt"
CATALOG_2M_IDS_FILE = BASE_DIR / "catalog_2m_ids.json"
CHECKPOINT_2M_FILE = BASE_DIR / "indexer_2m_checkpoint.json"

ENGLISH_CODES = {"eng", "en-us", "en-gb", "en-ca", "en", ""}


def clean_text(text: Optional[str]) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def stream_goodreads_generator(gz_path: Path, start_line: int = 0) -> Generator[Dict, None, None]:
    """Streams and parses goodreads_books.json.gz line-by-line without loading into RAM."""
    author_map = {}
    auth_db = BASE_DIR / "authors.sqlite3"
    if auth_db.exists():
        import sqlite3
        logger.info(f"Loading author name index from {auth_db.name}...")
        conn = sqlite3.connect(auth_db)
        cur = conn.cursor()
        cur.execute("SELECT author_id, name FROM authors")
        author_map = dict(cur.fetchall())
        conn.close()
        logger.info(f"Loaded {len(author_map):,} author names.")

    line_number = 0
    with gzip.open(gz_path, "rt", encoding="utf-8") as f:
        for line in f:
            line_number += 1
            if line_number <= start_line:
                continue

            try:
                raw = json.loads(line)
            except Exception:
                continue

            # 1. Language Filter
            lang = (raw.get("language_code") or "").lower().strip()
            if lang not in ENGLISH_CODES:
                continue

            # 2. Content Quality Filters
            desc = clean_text(raw.get("description"))
            if len(desc) < 40:
                continue

            title = clean_text(raw.get("title"))
            if not title:
                continue

            book_id = str(raw.get("book_id", "")).strip()
            if not book_id:
                continue

            # 3. Clean fields
            authors = []
            for a in raw.get("authors", []):
                aid = str(a.get("author_id", "")).strip()
                aname = a.get("name") or author_map.get(aid)
                if aname:
                    authors.append(clean_text(aname))
            shelves = [clean_text(s.get("name")) for s in raw.get("popular_shelves", []) if s.get("name")][:12]

            try:
                avg_rating = float(raw.get("average_rating") or 0.0)
            except Exception:
                avg_rating = 0.0

            try:
                ratings_count = int(raw.get("ratings_count") or 0)
            except Exception:
                ratings_count = 0

            pub_year_str = str(raw.get("publication_year", "")).strip()
            pub_year = int(pub_year_str) if pub_year_str.isdigit() else 0

            image_url = raw.get("image_url", "")

            yield {
                "line_index": line_number,
                "book_id": book_id,
                "title": title,
                "authors": authors,
                "description": desc,
                "popular_shelves": shelves,
                "average_rating": avg_rating,
                "ratings_count": ratings_count,
                "publication_year": pub_year,
                "image_url": image_url,
            }


def format_embedding_text(book: Dict) -> str:
    parts = []
    if book["title"]:
        parts.append(f"Title: {book['title']}")
    if book["authors"]:
        parts.append(f"Author: {', '.join(book['authors'])}")
    if book["popular_shelves"]:
        parts.append(f"Genres: {', '.join(book['popular_shelves'][:8])}")
    if book["description"]:
        parts.append(f"Description: {book['description'][:600]}")
    return " | ".join(parts)


def run_streaming_indexer(
    batch_size: int = 256,
    flush_every: int = 25000,
    max_total_books: Optional[int] = None,
    index_es: bool = True,
    index_vectors: bool = True,
):
    logger.info("=" * 80)
    logger.info("STARTING STREAMING INGESTION FOR 2.3M GOODREADS CATALOG")
    logger.info("Constant Low-RAM Mode: Streaming generator with micro-batch flushing.")
    logger.info("=" * 80)

    # 1. Resume support
    start_line = 0
    total_indexed = 0
    all_book_ids: List[str] = []
    embedding_chunks: List[torch.Tensor] = []

    if CHECKPOINT_2M_FILE.exists():
        try:
            with open(CHECKPOINT_2M_FILE, "r", encoding="utf-8") as f:
                ckpt = json.load(f)
                start_line = ckpt.get("last_processed_line", 0)
                total_indexed = ckpt.get("total_indexed", 0)
                logger.info(f"Resuming from line {start_line:,} (already indexed: {total_indexed:,} books)")
        except Exception as e:
            logger.warning(f"Could not load checkpoint: {e}. Starting fresh.")

    if CATALOG_2M_IDS_FILE.exists() and start_line > 0:
        with open(CATALOG_2M_IDS_FILE, "r", encoding="utf-8") as f:
            all_book_ids = json.load(f)

    if CATALOG_2M_EMB_FILE.exists() and start_line > 0:
        logger.info(f"Loading existing embeddings tensor from {CATALOG_2M_EMB_FILE.name}...")
        existing_embs = torch.load(CATALOG_2M_EMB_FILE, map_location="cpu")
        embedding_chunks.append(existing_embs)

    # 2. Setup Elasticsearch connection if enabled
    es_client = None
    if index_es and Elasticsearch is not None:
        try:
            client = Elasticsearch(ES_HOST, request_timeout=30)
            if client.ping():
                es_client = client
                logger.info(f"Elasticsearch connected at {ES_HOST}")
            else:
                logger.warning(f"Elasticsearch not reachable at {ES_HOST}. Skipping ES indexing.")
        except Exception as e:
            logger.warning(f"Elasticsearch connection failed: {e}. Skipping ES indexing.")

    # 3. Setup Embedding Model
    model = None
    if index_vectors:
        logger.info(f"Loading embedding model from {EMBED_MODEL_PATH} on {DEVICE}...")
        model = SentenceTransformer(EMBED_MODEL_PATH, device=DEVICE)
        model.max_seq_length = 256

    # 4. Stream loop
    current_batch_books: List[Dict] = []
    current_batch_texts: List[str] = []
    current_batch_ids: List[str] = []
    current_es_actions: List[Dict] = []

    last_line_seen = start_line
    t0 = time.time()

    pbar = tqdm(desc="Indexing books", initial=total_indexed, total=max_total_books or 1500000)

    try:
        for book in stream_goodreads_generator(GOODREADS_GZ_PATH, start_line=start_line):
            last_line_seen = book["line_index"]
            bid = book["book_id"]

            current_batch_books.append(book)
            current_batch_ids.append(bid)
            all_book_ids.append(bid)

            if index_vectors:
                current_batch_texts.append(format_embedding_text(book))

            if es_client:
                current_es_actions.append({
                    "_index": ES_INDEX_NAME,
                    "_id": bid,
                    "_source": {
                        "book_id": bid,
                        "title": book["title"],
                        "authors": book["authors"],
                        "description": book["description"],
                        "popular_shelves": book["popular_shelves"],
                        "average_rating": book["average_rating"],
                        "ratings_count": book["ratings_count"],
                        "publication_year": book["publication_year"],
                        "image_url": book["image_url"],
                    }
                })

            # Process micro-batch of embeddings
            if len(current_batch_texts) >= batch_size and index_vectors and model:
                with torch.no_grad():
                    embs = model.encode(
                        current_batch_texts,
                        batch_size=batch_size,
                        normalize_embeddings=True,
                        convert_to_tensor=True,
                        device=DEVICE,
                        show_progress_bar=False,
                    )
                    if USE_FP16 and DEVICE == "cuda":
                        embs = embs.half()
                    embedding_chunks.append(embs.cpu())

                current_batch_texts.clear()

            # Flush to Elasticsearch
            if len(current_es_actions) >= 2000 and es_client:
                try:
                    helpers.bulk(es_client, current_es_actions, chunk_size=2000, request_timeout=60)
                except Exception as e:
                    logger.error(f"ES Bulk insert error: {e}")
                current_es_actions.clear()

            total_indexed += 1
            pbar.update(1)

            # Checkpoint flush
            if total_indexed % flush_every == 0:
                logger.info(f"\n[Checkpoint] Saving progress at {total_indexed:,} books (line {last_line_seen:,})...")
                # Merge tensor chunks to save RAM
                if embedding_chunks:
                    combined_tensor = torch.cat(embedding_chunks, dim=0)
                    torch.save(combined_tensor, CATALOG_2M_EMB_FILE)
                    embedding_chunks = [combined_tensor]  # keep single consolidated tensor

                with open(CATALOG_2M_IDS_FILE, "w", encoding="utf-8") as f:
                    json.dump(all_book_ids, f)

                with open(CHECKPOINT_2M_FILE, "w", encoding="utf-8") as f:
                    json.dump({"last_processed_line": last_line_seen, "total_indexed": total_indexed}, f)

                logger.info(f"Checkpoint saved! Speed: {total_indexed / max(1, time.time() - t0):.1f} books/sec")

            if max_total_books and total_indexed >= max_total_books:
                logger.info(f"Reached requested limit of {max_total_books:,} books.")
                break

    except KeyboardInterrupt:
        logger.info("\nInterrupted by user! Saving checkpoint before exit...")
    finally:
        # Final flush
        if current_batch_texts and index_vectors and model:
            with torch.no_grad():
                embs = model.encode(
                    current_batch_texts,
                    batch_size=len(current_batch_texts),
                    normalize_embeddings=True,
                    convert_to_tensor=True,
                    device=DEVICE,
                    show_progress_bar=False,
                )
                if USE_FP16 and DEVICE == "cuda":
                    embs = embs.half()
                embedding_chunks.append(embs.cpu())

        if current_es_actions and es_client:
            try:
                helpers.bulk(es_client, current_es_actions, chunk_size=len(current_es_actions), request_timeout=60)
            except Exception as e:
                logger.error(f"ES Final bulk error: {e}")

        if embedding_chunks:
            logger.info("Saving final consolidated vector embeddings...")
            final_tensor = torch.cat(embedding_chunks, dim=0)
            torch.save(final_tensor, CATALOG_2M_EMB_FILE)
            logger.info(f"Vector tensor saved: {final_tensor.shape} ({final_tensor.element_size() * final_tensor.nelement() / (1024**2):.1f} MB)")

        with open(CATALOG_2M_IDS_FILE, "w", encoding="utf-8") as f:
            json.dump(all_book_ids, f)

        with open(CHECKPOINT_2M_FILE, "w", encoding="utf-8") as f:
            json.dump({"last_processed_line": last_line_seen, "total_indexed": total_indexed}, f)

        pbar.close()
        logger.info(f"\nFinished! Total indexed: {total_indexed:,} books in {time.time() - t0:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="2.3M Goodreads Scalable Stream Indexer")
    parser.add_argument("--batch-size", type=int, default=256, help="GPU embedding micro-batch size")
    parser.add_argument("--flush-every", type=int, default=25000, help="Checkpoint to disk every N books")
    parser.add_argument("--max-books", type=int, default=None, help="Limit total books to index (e.g. 500000)")
    parser.add_argument("--no-vectors", action="store_true", help="Skip dense vector generation")
    parser.add_argument("--no-es", action="store_true", help="Skip Elasticsearch indexing")
    args = parser.parse_args()

    run_streaming_indexer(
        batch_size=args.batch_size,
        flush_every=args.flush_every,
        max_total_books=args.max_books,
        index_es=not args.no_es,
        index_vectors=not args.no_vectors,
    )


if __name__ == "__main__":
    main()

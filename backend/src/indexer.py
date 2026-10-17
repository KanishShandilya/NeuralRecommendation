"""
Data Ingestion & Indexing Engine
=================================
Streams book records from goodreads_books.json.gz (~2.3M dump) or SQLite,
applies quality filtering, and bulk-indexes into:
  1. Elasticsearch (BM25 keyword index + author autocomplete)
  2. Dense Vector Store (FP16 embeddings tensor)
"""

import argparse
import gzip
import json
import logging
import re
import sqlite3
import time
from pathlib import Path
from typing import Generator, List, Optional

import torch
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

try:
    from elasticsearch import Elasticsearch, helpers
except ImportError:
    Elasticsearch = None

from .config import (
    CATALOG_EMB_CACHE,
    CATALOG_IDS_CACHE,
    DEVICE,
    EMBED_MODEL_PATH,
    ES_HOST,
    ES_INDEX_NAME,
    GOODREADS_GZ_PATH,
    SQLITE_DB_PATH,
    USE_FP16,
)
from .es_retriever import ESRetriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("BookSearch.Indexer")

ENGLISH_CODES = {"eng", "en-us", "en-gb", "en-ca", "en", ""}


def clean_text(text: Optional[str]) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def stream_goodreads_dump(gz_path: Path, max_books: Optional[int] = None) -> Generator[dict, None, None]:
    """Streams and filters records from the 2.36M Goodreads gzip dump."""
    logger.info(f"Streaming from {gz_path.name}...")
    count = 0
    with gzip.open(gz_path, "rt", encoding="utf-8") as f:
        for line in f:
            try:
                raw = json.loads(line)
            except Exception:
                continue

            # Quality filters
            lang = (raw.get("language_code") or "").lower().strip()
            if lang not in ENGLISH_CODES:
                continue

            desc = clean_text(raw.get("description"))
            if len(desc) < 40:
                continue

            title = clean_text(raw.get("title"))
            if not title:
                continue

            book_id = str(raw.get("book_id", ""))
            if not book_id:
                continue

            authors = [a.get("name") for a in raw.get("authors", []) if a.get("name")]
            shelves = [s.get("name") for s in raw.get("popular_shelves", []) if s.get("name")][:15]

            yield {
                "book_id": book_id,
                "title": title,
                "authors": authors,
                "description": desc,
                "popular_shelves": shelves,
                "average_rating": float(raw.get("average_rating") or 0.0),
                "ratings_count": int(raw.get("ratings_count") or 0),
                "publication_year": int(raw.get("publication_year") or 0) if str(raw.get("publication_year", "")).isdigit() else 0,
            }

            count += 1
            if max_books and count >= max_books:
                break


def stream_sqlite(db_path: Path, max_books: Optional[int] = None) -> Generator[dict, None, None]:
    """Streams records from books_metadata.sqlite3."""
    logger.info(f"Streaming from SQLite database {db_path.name}...")
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    query = "SELECT book_id, title, authors, description, popular_shelves FROM books"
    if max_books:
        query += f" LIMIT {max_books}"
    cur.execute(query)

    for bid, title, authors, desc, shelves in cur.fetchall():
        try:
            auth_list = json.loads(authors) if authors and authors.startswith("[") else [authors]
        except Exception:
            auth_list = [authors] if authors else []
        try:
            shelf_list = json.loads(shelves) if shelves and shelves.startswith("[") else []
        except Exception:
            shelf_list = []

        yield {
            "book_id": str(bid),
            "title": title or "",
            "authors": auth_list,
            "description": desc or "",
            "popular_shelves": shelf_list[:15],
            "average_rating": 4.0,
            "ratings_count": 100,
            "publication_year": 2000,
        }
    conn.close()


def bulk_index_elasticsearch(books: List[dict]):
    """Bulk indexes book documents into Elasticsearch."""
    if Elasticsearch is None:
        logger.error("elasticsearch package not installed.")
        return

    client = Elasticsearch(ES_HOST)
    if not client.ping():
        logger.error(f"Cannot connect to Elasticsearch at {ES_HOST}")
        return

    retriever = ESRetriever(ES_HOST, ES_INDEX_NAME)
    retriever.create_index_if_missing()

    logger.info(f"Bulk indexing {len(books):,} books into Elasticsearch index '{ES_INDEX_NAME}'...")
    actions = []
    for b in books:
        actions.append({
            "_index": ES_INDEX_NAME,
            "_id": b["book_id"],
            "_source": b,
        })

    success, failed = helpers.bulk(client, actions, chunk_size=2000, request_timeout=60)
    logger.info(f"Elasticsearch indexing complete: {success:,} indexed, {len(failed):,} failed.")


def build_dense_embeddings(books: List[dict]):
    """Encodes book metadata into FP16 dense vector tensor."""
    logger.info(f"Loading embedding model from {EMBED_MODEL_PATH} on {DEVICE}...")
    model = SentenceTransformer(EMBED_MODEL_PATH, device=DEVICE)

    logger.info(f"Formatting {len(books):,} books for vector encoding...")
    book_ids = []
    texts = []
    for b in books:
        book_ids.append(b["book_id"])
        parts = []
        if b["title"]:
            parts.append(f"Title: {b['title']}")
        if b["authors"]:
            parts.append(f"Author: {', '.join(b['authors'])}")
        if b["popular_shelves"]:
            parts.append(f"Genres: {', '.join(b['popular_shelves'][:8])}")
        if b["description"]:
            parts.append(f"Description: {b['description'][:600]}")
        texts.append(" | ".join(parts))

    logger.info(f"Generating embeddings with batch_size=128...")
    embs = model.encode(
        texts,
        batch_size=128,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_tensor=True,
        device=DEVICE,
    )
    if USE_FP16 and DEVICE == "cuda":
        embs = embs.half()

    torch.save(embs.cpu(), CATALOG_EMB_CACHE)
    with open(CATALOG_IDS_CACHE, "w", encoding="utf-8") as f:
        json.dump(book_ids, f)

    logger.info(f"Dense vector index saved to {CATALOG_EMB_CACHE.name} (Shape: {embs.shape})")


def main():
    parser = argparse.ArgumentParser(description="Ingest and Index Books")
    parser.add_argument("--source", choices=["sqlite", "gzip"], default="sqlite", help="Source data file")
    parser.add_argument("--max-books", type=int, default=None, help="Limit number of books to index")
    parser.add_argument("--index-es", action="store_true", help="Index into Elasticsearch")
    parser.add_argument("--index-vectors", action="store_true", help="Generate dense vector index")
    args = parser.parse_args()

    # Stream books
    streamer = stream_goodreads_dump(GOODREADS_GZ_PATH, args.max_books) if args.source == "gzip" else stream_sqlite(SQLITE_DB_PATH, args.max_books)
    books = list(tqdm(streamer, desc="Reading books"))
    logger.info(f"Loaded {len(books):,} qualified book records.")

    if args.index_es:
        bulk_index_elasticsearch(books)

    if args.index_vectors:
        build_dense_embeddings(books)

    if not args.index_es and not args.index_vectors:
        print("\nSpecify --index-es to index to Elasticsearch or --index-vectors to build vector embeddings.")


if __name__ == "__main__":
    main()

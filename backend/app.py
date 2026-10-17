"""
Book Recommendation & Semantic Search FastAPI Server
===================================================
Production-grade REST API powering the frontend interface.
Orchestrates:
  - Hybrid Search (BM25 + BGE Dense + Cross-Encoder Reranking)
  - Multi-tier Caching (Redis Exact + Semantic Cache)
  - Real-time Multi-Field Autocomplete (Title, Author, Genre)
  - Book Detail & Vector Similarity Recommendations
"""

import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.pipeline import HybridSearchPipeline
from src.config import ES_INDEX_NAME, ES_HOST

app = FastAPI(
    title="Goodreads 2.3M Book Semantic Search & Discovery API",
    description="Production-grade hybrid search with dense vector ANN, BM25, cross-encoder reranking, and Redis caching.",
    version="2.0.0",
)

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance
pipeline: Optional[HybridSearchPipeline] = None


@app.on_event("startup")
def startup_event():
    global pipeline
    print("🚀 Initializing Hybrid Search Pipeline in FastAPI...")
    pipeline = HybridSearchPipeline()
    print("✅ Engine pipeline ready for requests!")


class SearchRequest(BaseModel):
    query: str = Field(..., description="User search query or reading taste prompt")
    filters: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Metadata filters (genre, author, min_rating)")
    top_k: int = Field(default=10, ge=1, le=50, description="Number of results to return")
    use_cache: bool = Field(default=True, description="Enable Redis / Vector cache")
    use_reranker: bool = Field(default=True, description="Enable Cross-Encoder precision reranking")


@app.get("/health")
@app.get("/api/health")
def health_check():
    """Returns status of Elasticsearch, Redis, and search models."""
    global pipeline
    es_status = pipeline.es.is_es_active if pipeline else False
    redis_status = pipeline.cache.is_redis_active if pipeline else False
    dense_docs = len(pipeline.dense.catalog_ids) if pipeline and pipeline.dense else 0

    return {
        "status": "healthy",
        "elasticsearch": {
            "connected": es_status,
            "host": ES_HOST,
            "index": ES_INDEX_NAME,
        },
        "redis": {
            "connected": redis_status,
            "semantic_cache": True,
        },
        "vector_catalog": {
            "indexed_count": dense_docs,
            "model": "kanishshandilya/book_recs_embed",
            "device": pipeline.dense.device if pipeline else "unknown",
        },
        "reranker": {
            "model": "kanishshandilya/book_recs_reranker",
            "active": pipeline.reranker is not None if pipeline else False,
        }
    }


@app.post("/api/search")
def search_books(req: SearchRequest):
    """Executes full hybrid search pipeline."""
    global pipeline
    if not pipeline:
        raise HTTPException(status_code=503, detail="Search engine initializing")

    query = req.query.strip()
    if not query:
        return {"query": "", "results": [], "total_candidates_retrieved": 0, "latency": {"total_ms": 0.0}}

    results = pipeline.search(
        query=query,
        filters=req.filters,
        top_k=req.top_k,
        use_cache=req.use_cache,
        use_reranker=req.use_reranker,
    )
    return results


@app.get("/api/search")
def search_books_get(
    q: str = Query(..., description="Search query"),
    genre: Optional[str] = Query(None, description="Genre filter"),
    author: Optional[str] = Query(None, description="Author filter"),
    min_rating: Optional[float] = Query(None, description="Minimum average rating"),
    top_k: int = Query(10, ge=1, le=50),
    use_cache: bool = Query(True),
    use_reranker: bool = Query(True),
):
    """GET endpoint for easy browser or quick parameter queries."""
    filters = {}
    if genre:
        filters["genre"] = genre
    if author:
        filters["author"] = author
    if min_rating:
        filters["min_rating"] = min_rating

    req = SearchRequest(
        query=q,
        filters=filters,
        top_k=top_k,
        use_cache=use_cache,
        use_reranker=use_reranker,
    )
    return search_books(req)


@app.get("/api/autocomplete")
def autocomplete(
    q: str = Query(..., min_length=1, description="Prefix to autocomplete"),
    type: str = Query("all", description="Type of suggestions: all, title, author, genre"),
    limit: int = Query(6, ge=1, le=15),
):
    """
    Intelligent autocomplete across:
      - Titles (with covers & years)
      - Authors (using edge-ngram author analyzer)
      - Genres (popular shelf tags)
    """
    global pipeline
    if not pipeline or not pipeline.es.is_es_active or not pipeline.es.es_client:
        return {"titles": [], "authors": [], "genres": []}

    clean_q = q.strip().lower()
    es_client = pipeline.es.es_client
    index_name = pipeline.es.index_name

    titles = []
    authors = []
    genres = []

    # 1. Title prefix autocomplete
    if type in ("all", "title"):
        try:
            body = {
                "size": limit,
                "query": {
                    "match_phrase_prefix": {
                        "title": {
                            "query": clean_q,
                            "max_expansions": 10
                        }
                    }
                },
                "_source": ["book_id", "title", "publication_year", "authors", "image_url"]
            }
            res = es_client.search(index=index_name, body=body)
            for hit in res["hits"]["hits"]:
                src = hit["_source"]
                titles.append({
                    "book_id": src.get("book_id"),
                    "title": src.get("title"),
                    "authors": src.get("authors", []),
                    "year": src.get("publication_year"),
                    "image_url": src.get("image_url", ""),
                })
        except Exception:
            pass

    # 2. Author autocomplete (edge-ngram)
    if type in ("all", "author"):
        try:
            body = {
                "size": limit,
                "query": {
                    "match": {
                        "authors": {
                            "query": clean_q,
                            "analyzer": "standard"
                        }
                    }
                },
                "_source": ["authors"]
            }
            res = es_client.search(index=index_name, body=body)
            seen_authors = set()
            for hit in res["hits"]["hits"]:
                for a in hit["_source"].get("authors", []):
                    if clean_q in a.lower() and a not in seen_authors:
                        seen_authors.add(a)
                        authors.append({"name": a})
                        if len(authors) >= limit:
                            break
                if len(authors) >= limit:
                    break
        except Exception:
            pass

    # 3. Genre matching
    if type in ("all", "genre"):
        POPULAR_GENRES = [
            "fantasy", "science-fiction", "romance", "thriller", "mystery",
            "historical-fiction", "classics", "young-adult", "nonfiction",
            "horror", "dystopia", "crime", "biography", "memoir", "poetry",
            "graphic-novels", "philosophy", "psychology", "adventure"
        ]
        matched = [g for g in POPULAR_GENRES if clean_q in g or g.startswith(clean_q)]
        genres = [{"genre": g} for g in matched[:limit]]

    return {
        "query": q,
        "titles": titles,
        "authors": authors,
        "genres": genres,
    }


@app.get("/api/book/{book_id}")
def get_book_details(book_id: str):
    """Retrieves full metadata for a book from Elasticsearch."""
    global pipeline
    if not pipeline or not pipeline.es.is_es_active or not pipeline.es.es_client:
        raise HTTPException(status_code=503, detail="Elasticsearch unavailable")

    try:
        res = pipeline.es.es_client.get(index=pipeline.es.index_name, id=book_id)
        if res.get("found"):
            return res["_source"]
        raise HTTPException(status_code=404, detail="Book not found")
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Book not found ({e})")


@app.get("/api/recommend/{book_id}")
def recommend_similar_books(book_id: str, top_k: int = Query(6, ge=1, le=20)):
    """
    Finds nearest neighbors in dense embedding space for vector-based recommendations.
    """
    global pipeline
    if not pipeline or pipeline.dense.catalog_matrix is None:
        raise HTTPException(status_code=503, detail="Dense retrieval index unavailable")

    # Locate book in catalog
    dense = pipeline.dense
    try:
        idx = dense.catalog_ids.index(book_id)
        book_emb = dense.catalog_matrix[idx]
    except ValueError:
        # If not in catalog index, try fetching book from ES and encoding
        doc = get_book_details(book_id)
        parts = [f"Title: {doc.get('title', '')}"]
        if doc.get("authors"):
            parts.append(f"Author: {', '.join(doc['authors'])}")
        if doc.get("description"):
            parts.append(f"Description: {doc['description'][:400]}")
        text = " | ".join(parts)
        book_emb = dense.model.encode(text, normalize_embeddings=True, convert_to_tensor=True, device=dense.device)
        if dense.device == "cuda":
            book_emb = book_emb.half()

    # Vector search
    candidates = dense.search_vector(book_emb, top_k=top_k + 5)
    # Filter out current book
    filtered_bids = [c["book_id"] for c in candidates if c["book_id"] != book_id][:top_k]

    # Hydrate metadata from ES
    books = []
    if pipeline.es.is_es_active and pipeline.es.es_client:
        try:
            mres = pipeline.es.es_client.mget(index=pipeline.es.index_name, body={"ids": filtered_bids})
            for doc in mres.get("docs", []):
                if doc.get("found"):
                    src = doc["_source"]
                    books.append(src)
        except Exception:
            pass

    return {
        "source_book_id": book_id,
        "recommendations": books,
    }


@app.get("/api/genres")
def list_popular_genres():
    """Returns top genres for facet navigation."""
    genres = [
        {"name": "Fantasy", "id": "fantasy"},
        {"name": "Sci-Fi", "id": "science-fiction"},
        {"name": "Romance", "id": "romance"},
        {"name": "Thriller", "id": "thriller"},
        {"name": "Mystery", "id": "mystery"},
        {"name": "Historical Fiction", "id": "historical-fiction"},
        {"name": "Classics", "id": "classics"},
        {"name": "Young Adult", "id": "young-adult"},
        {"name": "Non-Fiction", "id": "nonfiction"},
        {"name": "Horror", "id": "horror"},
        {"name": "Dystopia", "id": "dystopia"},
    ]
    return {"genres": genres}


# Mount frontend static files to serve UI at root URL
from fastapi.staticfiles import StaticFiles
frontend_dir = backend_dir.parent / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

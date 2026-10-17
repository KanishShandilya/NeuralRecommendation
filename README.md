# 📚 Book Semantic Search & Neural Recommendation Engine

> Production-grade hybrid retrieval and recommendation platform scaling to **1.56+ Million Goodreads books**. Combines fine-tuned dense vector embeddings (`BAAI/bge-small-en-v1.5`), BM25 keyword search via **Elasticsearch 8.17**, neural cross-encoder precision reranking (`BAAI/bge-reranker-base`), multi-tier **Redis caching**, and a modern **FastAPI** backend with a responsive dark-mode UI.

[![Hugging Face Embeddings](https://img.shields.io/badge/%F0%9F%A4%97%20Retriever-kanishshandilya%2Fbook__recs__embed-blue)](https://huggingface.co/kanishshandilya/book_recs_embed)
[![Hugging Face Reranker](https://img.shields.io/badge/%F0%9F%A4%97%20Reranker-kanishshandilya%2Fbook__recs__reranker-orange)](https://huggingface.co/kanishshandilya/book_recs_reranker)
[![Elasticsearch](https://img.shields.io/badge/Elasticsearch-8.17.2-005571?logo=elasticsearch)](https://www.elastic.co/)
[![Redis](https://img.shields.io/badge/Redis-Cache-DC382D?logo=redis)](https://redis.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?logo=fastapi)](https://fastapi.tiangolo.com/)

---

## Table of Contents

- [Overview](#overview)
- [System & Backend Architecture](#system--backend-architecture)
  - [Two-Stage Hybrid Search Pipeline](#two-stage-hybrid-search-pipeline)
  - [Backend Modules Deep Dive](#backend-modules-deep-dive-book_recommendationbackendsrc)
  - [Multi-Tier Caching & Semantic Vector Cache (Deep Dive)](#multi-tier-caching--semantic-vector-cache-deep-dive)
  - [Multi-Modal Autocomplete Engine](#multi-modal-autocomplete-engine)
- [Model Fine-Tuning Pipeline](#model-fine-tuning-pipeline)
  - [Bi-Encoder Retriever (`bge-small-en-v1.5`)](#bi-encoder-retriever-bge-small-en-v15)
  - [Cross-Encoder Reranker (`bge-reranker-base`)](#cross-encoder-reranker-bge-reranker-base)
  - [Iterative Hard Negative Mining](#iterative-hard-negative-mining)
- [Benchmark Results](#benchmark-results)
- [Catalog Scaling & Ingestion Pipeline](#catalog-scaling--ingestion-pipeline)
- [FastAPI REST API Reference](#fastapi-rest-api-reference)
- [Directory Structure](#directory-structure)
- [Quick Start & Running Locally](#quick-start--running-locally)
- [License](#license)

---

## Overview

Traditional keyword-only search fails when readers search with vague thematic descriptions (*"atmospheric gothic mystery in a rainy coastal town"*), character dynamics (*"enemies to lovers fantasy with royalty"*), or obscure plot tropes. 

This platform implements an industrial **Two-Stage Hybrid Search Pipeline** across **1,567,392 books**:
1. **Multi-tier Caching**: Sub-millisecond exact Redis matches and semantic vector cache hits.
2. **Stage 1 (Parallel Hybrid Retrieval)**:
   - **Stage 1a — Dense Bi-Encoder ANN**: Vector similarity over 1.56M 384-dimensional FP16 embeddings on GPU.
   - **Stage 1b — Sparse BM25 Keyword Search**: Exact title, author, shelf, and description matching via Elasticsearch.
3. **Fusion (Reciprocal Rank Fusion - RRF)**: Merges candidates from both modalities with zero duplicates.
4. **Stage 2 (Neural Cross-Encoder Reranking)**: Computes full cross-attention over top candidates for maximum precision.
5. **Zero-Memory Candidate Hydration**: Direct Elasticsearch `mget` to keep Python heap footprint minimal.

---

## System & Backend Architecture

```
                                  ┌───────────────────────────┐
                                  │      React / Web UI       │
                                  └─────────────┬─────────────┘
                                                │ REST API
                                                ▼
                                  ┌───────────────────────────┐
                                  │     FastAPI Application   │
                                  │   (book_recommendation/)  │
                                  └─────────────┬─────────────┘
                                                │
                     ┌──────────────────────────┴──────────────────────────┐
                     │                                                     │
                     ▼                                                     ▼
      ┌─────────────────────────────┐                       ┌─────────────────────────────┐
      │   Tier 1: Redis Exact Cache  │                       │ Tier 2: Semantic Cache      │
      │   (MD5 Hash Key, TTL 1h)    │                       │ (Cosine Sim ≥ 0.95 Vector)  │
      └──────────────┬──────────────┘                       └──────────────┬──────────────┘
                     │ [Cache Hit: 0.0 ms]                                 │
                     └──────────────────────────┬──────────────────────────┘
                                                │ [Cache Miss: Run Pipeline]
                                                ▼
                     ┌─────────────────────────────────────────────────────┐
                     │          Stage 1: Parallel Candidate Retrieval      │
                     └──────────────┬───────────────────────┬──────────────┘
                                    │                       │
                                    ▼                       ▼
      ┌───────────────────────────────────────────┐   ┌───────────────────────────────────────────┐
      │   Stage 1a: Dense Bi-Encoder ANN Search   │   │   Stage 1b: Elasticsearch BM25 Search    │
      │   - Model: kanishshandilya/book_recs_embed│   │   - Index: goodreads_books (1.56M docs)   │
      │   - Catalog: 1,567,392 FP16 vectors (CUDA)│   │   - Fields: title^3.5, authors^3.0,       │
      │   - Output: Top-50 vector candidates      │   │             popular_shelves^1.5, desc^1.0 │
      └─────────────────────┬─────────────────────┘   └─────────────────────┬─────────────────────┘
                            │                                               │
                            └───────────────────────┬───────────────────────┘
                                                    │ Top candidates
                                                    ▼
                                  ┌───────────────────────────────────┐
                                  │   Reciprocal Rank Fusion (RRF)    │
                                  │   RRF_Score = Σ 1 / (60 + Rank_i) │
                                  │   Deduplicates & combines scores  │
                                  └─────────────────┬─────────────────┘
                                                    │ Top 70 candidates
                                                    ▼
                                  ┌───────────────────────────────────┐
                                  │   Candidate Metadata Hydration    │
                                  │   - Uses Elasticsearch `mget`     │
                                  │   - Consumes 0 MB Python RAM      │
                                  └─────────────────┬─────────────────┘
                                                    │ Hydrated candidates
                                                    ▼
                                  ┌───────────────────────────────────┐
                                  │   Stage 2: Cross-Encoder Reranker │
                                  │   - kanishshandilya/book_recs_reranker│
                                  │   - FP16 Dynamic Padding on GPU   │
                                  │   - Re-scores top-70 candidates   │
                                  └─────────────────┬─────────────────┘
                                                    │
                                                    ▼
                                  ┌───────────────────────────────────┐
                                  │   Final Ranked Results (Top-K)    │
                                  │   + Update Redis Cache & Telemetry│
                                  └───────────────────────────────────┘
```

---

### Backend Modules Deep Dive (`book_recommendation/backend/src/`)

| Module | Purpose & Implementation Details |
| :--- | :--- |
| **`app.py`** | **Production FastAPI Server**: Exposes `/api/search`, `/api/autocomplete`, `/api/recommend/{id}`, and `/api/health`. Serves the frontend web app statically at root `/`. |
| **`config.py`** | **Central Settings**: Dynamic root path resolution, SSL context monkey-patching for Windows compatibility, Hugging Face repository identifiers, and hardware allocation. |
| **`cache.py`** | **Multi-Tier Cache**: Protocol RESP2 connection to standalone Redis. Supports exact key lookup (`md5(query + filters)`) and cosine semantic caching thresholded at $\ge 0.95$. |
| **`dense_retriever.py`** | **Bi-Encoder Retrieval Engine**: Loads `kanishshandilya/book_recs_embed` in FP16 on CUDA. Executes matrix-vector dot product against `catalog_2m_embeddings_fp16.pt` (`[1567392, 384]`). |
| **`es_retriever.py`** | **Elasticsearch BM25 Engine**: Connected to Elasticsearch 8.17.2. Multi-match boolean queries with custom field boosting and edge-ngram author autocomplete analyzer. Seamless fallback to in-memory Okapi BM25 if ES is offline. |
| **`reranker.py`** | **Cross-Encoder Precision Engine**: Fine-tuned `kanishshandilya/book_recs_reranker` running in mixed-precision FP16 with dynamic batch sequence truncation for sub-300ms inference. |
| **`pipeline.py`** | **Pipeline Orchestrator**: Coordinates Cache $\rightarrow$ Parallel Stage 1 $\rightarrow$ RRF Fusion $\rightarrow$ Zero-Memory `mget` Hydration $\rightarrow$ Stage 2 Reranking $\rightarrow$ Async Cache Write. |
| **`stream_indexer_2m.py`** | **Constant Low-RAM Streaming Indexer**: Line-by-line streaming generator for UCSD 2.36M books dump with micro-batch flushing (256 docs) to ES and incremental disk checkpointing. |

---

### Multi-Tier Caching & Semantic Vector Cache (Deep Dive)

The platform implements an industrial **Two-Tier Query Cache** combining deterministic key-value hashing with neural vector equivalence detection:

```
                            User Search Query
                                    │
                                    ▼
                  ┌───────────────────────────────────┐
                  │    Tier 1: Exact Match (Redis)    │
                  │    Key: SHA256(query + filters)   │
                  └─────────────────┬─────────────────┘
                                    │
                ┌───────────────────┴───────────────────┐
                │                                       │
          [Cache Hit]                             [Cache Miss]
                │                                       │
                ▼                                       ▼
   ⚡ Instant Response: 0.0 ms          ┌───────────────────────────────┐
   (Bypasses entire pipeline)           │ Encode Query Embedding (BGE)  │
                                        └───────────────┬───────────────┘
                                                        │
                                                        ▼
                                        ┌───────────────────────────────┐
                                        │  Tier 2: Semantic Vector Cache│
                                        │  Cosine Similarity ≥ 0.95     │
                                        └───────────────┬───────────────┘
                                                        │
                                    ┌───────────────────┴───────────────────┐
                                    │                                       │
                              [Semantic Hit]                          [Semantic Miss]
                                    │                                       │
                                    ▼                                       ▼
                       ⚡ Vector Cache Hit: ~11 ms               🔄 Full Pipeline Compute
                       (Paraphrase detected, skips             (Stage 1 ANN + BM25
                        BM25 & Cross-Reranker)                  + Cross-Encoder Rerank)
```

1. **Tier 1 — Exact Redis Match (`< 1 ms` / `0.0 ms`)**:
   - Queries are normalized and hashed via `SHA256(canonical_query + sorted_filters + top_k)`.
   - Connected via Redis RESP2 protocol with automatic fallback to thread-safe in-memory LRU (`capacity=2000`).
   - Returns the complete serialized JSON payload with metadata, score badges, and covers.

2. **Tier 2 — Semantic Vector Cache (`~11 ms`)**:
   - Implemented via [`SemanticVectorCache`](file:///c:/Users/Shilpa/OneDrive/Documents/Book%20Semantic%20Search/book_recommendation/backend/src/cache.py#L61-L102).
   - When exact lookup misses, the query is encoded into a 384-dimensional dense vector by `kanishshandilya/book_recs_embed`.
   - Computes a vectorized dot product (`torch.matmul`) against all cached normalized query vectors in memory.
   - If the maximum cosine similarity satisfies:
     $$\text{CosineSimilarity}(v_{\text{new}}, v_{\text{cached}}) \ge 0.95$$
     the query is classified as a **semantic paraphrase**, returning the cached results immediately. This completely avoids executing BM25 keyword search across 1.56M documents and running the heavy cross-encoder reranker.

#### Empirical Cache Verification Benchmarks

| Query | Status | Cache Tier | Latency | Pipeline Saved |
| :--- | :--- | :--- | :--- | :--- |
| `"dystopian surveillance state and psychological control"` | Cold Compute | None (Miss) | **1,044.3 ms** | Full 1.56M ANN + BM25 + Cross-Reranker |
| `"dystopian surveillance state and psychological control"` (repeated) | Cache Hit | **Tier 1 (Exact Match)** | **0.00 ms** | 100% compute bypassed |
| `"dystopian surveillance state with psychological control"` (paraphrase) | Cache Hit | **Tier 2 (Semantic Match: 0.986)** | **11.96 ms** | BM25 + Cross-Reranker bypassed |

---

### Multi-Modal Autocomplete Engine

The search interface provides as-you-type autocompletion across three dimensions:
- **Authors**: Handled by an Elasticsearch `edge_ngram` tokenizer (`min_gram=2`, `max_gram=15`) with a standard search analyzer. Typing `"anita"` instantly surfaces *Anita Diamant*, *Anita Nair*, *Anita Mills*, etc.
- **Titles**: Handled by `match_phrase_prefix` returning exact title suggestions, book cover URLs, and publication years.
- **Genres / Shelves**: Instant tag filtering across popular shelves (*fantasy*, *dystopia*, *romance*, *science-fiction*, *thriller*, *history*).

---

## Model Fine-Tuning Pipeline

### Bi-Encoder Retriever (`bge-small-en-v1.5`)
- **Base Architecture**: 384-dimensional dense transformer embedding model.
- **Loss Function**: `MultipleNegativesRankingLoss` (MNRL) with symmetric cross-entropy and in-batch negatives.
- **Iterative Mining Cycles**:
  - **Cycle 1 (v4)**: Hard negatives mined from initial fine-tuned v3 model.
  - **Cycle 2 (v5)**: Re-mined harder negatives from v4 checkpoint, training for an additional 3 epochs with linear warm-up.

### Cross-Encoder Reranker (`bge-reranker-base`)
- **Base Architecture**: Cross-attention transformer scoring `[query, document]` concatenated text.
- **Loss Function**: Binary Cross-Entropy with Logits (`BCEWithLogitsLoss`).
- **"Goldilocks Zone" Mining**: Negatives mined between ranks **10 and 45** of the best bi-encoder. Higher ranks ($<10$) risk false negatives; lower ranks ($>45$) are too easy for a cross-encoder to learn discriminative boundaries.
- **False Negative Guardrails**: Work-cluster exclusions, Jaccard token filtering, and author matching prevent penalizing valid books.

---

## Benchmark Results

### 1. Bi-Encoder Retriever (Recall@K evaluated on held-out catalog)

| Metric | Pretrained Baseline | v3 (Initial Fine-Tune) | v4 (Cycle 1 Mining) | v5 (Cycle 2 Best) | Absolute Gain |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Recall@1** | 45.1% | 62.4% | 65.2% | **67.3%** | **+22.2%** |
| **Recall@5** | 68.3% | 80.1% | 83.4% | **85.6%** | **+17.3%** |
| **Recall@10** | 75.2% | 85.3% | 88.1% | **89.4%** | **+14.2%** |
| **Recall@50** | 88.0% | 93.2% | 94.8% | **95.6%** | **+7.6%** |

### 2. Cross-Encoder Reranker (2,998 validation queries)

| Metric | Pretrained Baseline | Fine-Tuned Model | Absolute Improvement |
| :--- | :--- | :--- | :--- |
| **MRR@10** | 79.14% | **88.73%** | **+9.59%** |
| **MAP** | 79.13% | **88.71%** | **+9.58%** |

### 3. Live System Latency Profile (Across 1,567,392 Books)

| Pipeline Step | Cold Execution Latency | Warm Cached Latency | Notes |
| :--- | :--- | :--- | :--- |
| **Redis Cache Lookup** | 0.00 ms | **0.00 ms** | Exact MD5 hash match |
| **Dense ANN Search** | ~800 ms | — | Matmul over 1.56M $\times$ 384 FP16 on GPU |
| **Elasticsearch BM25** | ~650 ms | — | Distributed BM25 query over 1.56M docs |
| **RRF Fusion** | ~65 ms | — | Score combination & candidate deduplication |
| **ES `mget` Hydration** | ~15 ms | — | Zero in-memory heap usage |
| **Cross-Encoder Rerank** | ~370 ms | — | FP16 inference on top-70 candidates |
| **Total Response Time** | **~1.90 s** | **0.00 ms** | Full 1.56M catalog search |

---

## Catalog Scaling & Ingestion Pipeline

To expand beyond the initial 64k catalog to the full UCSD Goodreads dataset:
1. **Author Name Resolution**: The raw Goodreads books dump only contains `author_id`. We extracted all **829,529 author names** from `goodreads_book_authors.json.gz` into an indexed SQLite database ([`authors.sqlite3`](file:///c:/Users/Shilpa/OneDrive/Documents/Book%20Semantic%20Search/authors.sqlite3)), mapping names to IDs in ~0.5s.
2. **Streaming Low-RAM Mode**: [`engine/stream_indexer_2m.py`](file:///c:/Users/Shilpa/OneDrive/Documents/Book%20Semantic%20Search/engine/stream_indexer_2m.py) processed all 2,360,655 lines of `goodreads_books.json.gz` line-by-line without loading the file into RAM.
3. **Quality Filtering**: English-language validation, description cleaning ($\ge 40$ chars), and minimum metadata checks.
4. **Final Catalog**: **1,567,392 qualified books** indexed into Elasticsearch and saved to `catalog_2m_embeddings_fp16.pt` (1.15 GB).

---

## FastAPI REST API Reference

The backend exposes an interactive OpenAPI Swagger UI at `http://localhost:8000/docs`.

### 1. `POST /api/search`
Executes the full hybrid retrieval and reranking pipeline.
```json
{
  "query": "dystopian surveillance state and psychological control",
  "filters": {
    "genre": "science-fiction",
    "min_rating": 3.8
  },
  "top_k": 10,
  "use_cache": true,
  "use_reranker": true
}
```

### 2. `GET /api/autocomplete`
Provides multi-field as-you-type suggestions.
```
GET /api/autocomplete?q=anita&type=all&limit=8
```
Response:
```json
{
  "query": "anita",
  "titles": [{"book_id": "1333909", "title": "Good Harbor", "authors": ["Anita Diamant"]}],
  "authors": [{"name": "Anita Diamant"}, {"name": "Anita Nair"}, {"name": "Anita Mills"}],
  "genres": []
}
```

### 3. `GET /api/recommend/{book_id}`
Returns nearest-neighbor books in dense embedding space for vector similarity recommendations.
```
GET /api/recommend/1333909?top_k=6
```

### 4. `GET /api/health`
Returns live connection status of Elasticsearch, Redis, and vector models.

---

## Directory Structure

```text
book_recommendation/
│
├── finetuning/                         # Model training & mining scripts
│   ├── 01_prepare_embedding_dataset.py # Step 1: Work-cluster split + initial mining
│   ├── 02_train_embedding_model.py     # Step 2: Iterative bi-encoder training (MNRL)
│   ├── 03_prepare_reranker_dataset.py  # Step 3: Goldilocks zone negative mining
│   ├── 04_train_reranker.py            # Step 4: Cross-encoder fine-tuning (BCE)
│   ├── iterative_mine_and_train.py     # Iterative cycles orchestrator
│   └── deep_quality_audit.py           # Dataset leakage & contamination auditor
│
├── backend/                            # Production FastAPI server & engine core
│   ├── app.py                          # FastAPI application & route handlers
│   ├── requirements.txt                # Python backend dependencies
│   └── src/                            # Engine components (zero in-memory heap)
│       ├── __init__.py
│       ├── config.py                   # Central settings & hardware configurations
│       ├── cache.py                    # Tier-1 Redis + Tier-2 Semantic Vector Cache
│       ├── dense_retriever.py          # Stage 1a Bi-Encoder ANN engine (FP16 CUDA)
│       ├── es_retriever.py             # Stage 1b Elasticsearch BM25 engine
│       ├── reranker.py                 # Stage 2 Cross-Encoder reranker engine
│       ├── pipeline.py                 # Hybrid pipeline orchestrator + RRF fusion
│       └── stream_indexer_2m.py        # 2.3M streaming ingestion generator
│
└── frontend/                           # Responsive Web Application UI
    ├── index.html                      # Semantic layout, search hero & telemetry dock
    ├── styles.css                      # Glassmorphic dark-mode design system
    └── app.js                          # Autocomplete, search client & modal drawer
```

---

## Quick Start & Running Locally

### 1. Prerequisites
- **Python 3.10+** with CUDA-enabled PyTorch
- **Elasticsearch 8.17+** running on `http://localhost:9200`
- **Redis Server** running on `localhost:6379`

### 2. Install Dependencies
```bash
pip install -r backend/requirements.txt
```

### 3. Start the Platform
Launch the FastAPI backend (serves both API and Web UI):
```bash
python backend/app.py
```
Open **`http://localhost:8000/`** in your browser.

---

## Published Models on Hugging Face

- **Bi-Encoder Retriever**: [`kanishshandilya/book_recs_embed`](https://huggingface.co/kanishshandilya/book_recs_embed)
- **Cross-Encoder Reranker**: [`kanishshandilya/book_recs_reranker`](https://huggingface.co/kanishshandilya/book_recs_reranker)

---

## License

MIT License. Dataset derived from the [UCSD Goodreads dataset](https://mengtingwan.github.io/data/goodreads.html) for research and non-commercial educational purposes.

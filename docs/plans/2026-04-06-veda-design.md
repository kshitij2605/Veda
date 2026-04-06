# Veda — Multimodal Vector Database Design Document

**Date:** 2026-04-06
**Status:** Approved (brainstorming complete)
**Components:** Veda (database engine) + Niriksha (visualization dashboard)

---

## Table of Contents

1. [Overview & Design Decisions](#1-overview--design-decisions)
2. [System Architecture](#2-system-architecture)
3. [Data Model & Storage Format](#3-data-model--storage-format)
4. [Index Architecture](#4-index-architecture)
5. [Query Engine & API Design](#5-query-engine--api-design)
6. [Niriksha Dashboard](#6-niriksha-dashboard)
7. [Deployment & Operations](#7-deployment--operations)

---

## 1. Overview & Design Decisions

**Veda** is a next-generation vector database designed for the multimodal era of RAG. It supports storing, indexing, and searching embeddings from any modality: text, images, audio, video, and documents.

**Niriksha** is Veda's built-in data visualization and exploration dashboard — the "Kibana for multimodal vectors."

### Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Language | Rust core + Python SDK (PyO3) | Production-proven (Qdrant, LanceDB, ChromaDB), C++-level perf with memory safety |
| Architecture | Full disaggregation (Milvus-style) | Independent scaling of query/data/index/coordination layers |
| Scale target | Billion+ vectors | Designed for enterprise workloads from day one |
| Blob storage | Transparent proxy to MinIO/S3 | No custom blob engine; object storage handles durability/replication |
| Embedding support | Pre-computed vectors + optional auto-embedding | Pluggable model pipelines with modality validation |
| Index strategy | HNSW (primary) + DiskANN + IVF-PQ + Flat | Multiple index types, auto-selected by collection size |
| Quantization | RaBitQ (primary), SQ, PQ | RaBitQ is 2024 SIGMOD breakthrough, 3x faster than PQ |
| Text search | Suffix array-based fuzzy pattern matching (SoftMatcha-style) | No OSS vector DB has this; complements semantic search with precise lexical retrieval |
| Client interface | REST + gRPC + SSE + Python SDK | No CLI client; Niriksha for interactive use, SDKs for programmatic |
| Build approach | From scratch in Rust | Study OpenSearch distributed patterns as reference, no fork |
| Milestone priority | Engine first, then Niriksha |

---

## 2. System Architecture

### Component Diagram

```
                           ┌─────────────────────────────┐
                           │         Clients              │
                           │  Python SDK · REST · gRPC    │
                           └──────────────┬───────────────┘
                                          │
                           ┌──────────────▼───────────────┐
                           │       Gateway                 │
                           │  REST + gRPC + SSE + Niriksha │
                           └──────────────┬───────────────┘
                                          │
              ┌───────────────────────────┼───────────────────────────┐
              │                           │                           │
   ┌──────────▼──────────┐   ┌───────────▼──────────┐   ┌───────────▼──────────┐
   │    Query Nodes       │   │    Data Nodes         │   │    Index Nodes        │
   │                      │   │                       │   │                       │
   │ • Execute searches   │   │ • Ingest & write data │   │ • Build indexes       │
   │ • Score & rank       │   │ • WAL management      │   │ • HNSW/DiskANN/IVF   │
   │ • Filter & fuse      │   │ • Segment management  │   │ • Quantization        │
   │ • Multi-vector merge │   │ • Stream to object    │   │ • GPU acceleration    │
   │   (RRF fusion)       │   │   storage             │   │   (cuVS/CAGRA)        │
   └──────────────────────┘   └───────────────────────┘   └───────────────────────┘
              │                           │                           │
              └───────────────────────────┼───────────────────────────┘
                                          │
                           ┌──────────────▼───────────────┐
                           │   Coordination Service        │
                           │                               │
                           │ • Raft consensus              │
                           │ • Cluster state / topology    │
                           │ • Shard routing table         │
                           │ • Schema registry             │
                           │ • Leader election             │
                           └──────────────┬───────────────┘
                                          │
              ┌───────────────────────────┼───────────────────────────┐
              │                           │                           │
   ┌──────────▼──────────┐   ┌───────────▼──────────┐   ┌───────────▼──────────┐
   │   Message Queue      │   │   Object Storage      │   │   Embedding Service   │
   │   (NATS JetStream)   │   │   (MinIO/S3)          │   │   (Optional)          │
   │                      │   │                       │   │                       │
   │ • Write-ahead log    │   │ • Sealed segments     │   │ • Pluggable models    │
   │ • Event streaming    │   │ • Raw data blobs      │   │ • CLIP, ImageBind     │
   │ • Change capture     │   │   (images, audio...)  │   │ • External API or     │
   └──────────────────────┘   └───────────────────────┘   │   local inference     │
                                                          └───────────────────────┘
```

### Component Responsibilities

| Component | Responsibility | Scales By |
|-----------|---------------|-----------|
| **Gateway** | REST/gRPC/SSE routing, auth, rate limiting, serves Niriksha | Stateless replicas |
| **Query Nodes** | Search execution, scoring, filtering, multi-vector fusion (RRF) | Adding read replicas |
| **Data Nodes** | Ingest, WAL, segment creation, flush to object storage | Adding write shards |
| **Index Nodes** | Background index building, quantization | Adding GPU/CPU workers |
| **Coordination** | Raft consensus, cluster state, shard routing, schema registry | 3-5 node quorum |
| **Message Queue** | Write-ahead log, event streaming, change data capture | Partitioned topics |
| **Object Storage** | Sealed segments, raw multimodal blobs, index snapshots | Infinite (S3) |
| **Embedding Service** | Optional auto-embedding via pluggable models | Independent GPU pool |

### Key Design Principles

- **Stateless compute**: Query/Data/Index nodes hold no permanent state — all durable state lives in object storage + coordination service
- **Segment-based storage**: Data organized into immutable sealed segments, with an in-memory growing segment for recent writes
- **Multi-vector native**: Each record can hold multiple embedding fields with separate indexes per field
- **Cross-modal search**: Query in one modality, retrieve across others. RRF for combining scores across fields

---

## 3. Data Model & Storage Format

### Collection Schema

A **collection** is the top-level container. Each collection has a schema defining its fields:

```
Collection: "product_catalog"
├── Schema:
│   ├── id: UUID (primary key, auto-generated)
│   ├── text_embedding: Vector(768)        — HNSW index
│   ├── image_embedding: Vector(512)       — DiskANN index
│   ├── audio_embedding: Vector(1024)      — IVF-PQ index
│   ├── title: String (filterable)
│   ├── category: String (filterable, facetable)
│   ├── price: Float64 (filterable, sortable)
│   ├── created_at: Timestamp (sortable)
│   ├── tags: Array<String> (filterable)
│   ├── image_blob: Blob (stored in object storage)
│   └── audio_blob: Blob (stored in object storage)
│
├── Partitioning: by category (optional)
├── Shards: 16 (auto or manual)
└── Replication: factor 3
```

### Field Types

| Type | Description | Indexable | Filterable |
|------|-------------|-----------|------------|
| `Vector(dim)` | Dense float32/float16/binary vector | ANN indexes | Pre/post filter |
| `SparseVector` | Sparse vector (BM25-style, learned sparse) | Inverted index | Yes |
| `String` | UTF-8 text, keyword or full-text | Inverted index | Yes |
| `Int32/64` | Integer | Range index | Yes |
| `Float32/64` | Floating point | Range index | Yes |
| `Bool` | Boolean | Bitmap index | Yes |
| `Timestamp` | UTC timestamp | Range index | Yes |
| `Array<T>` | Array of any scalar type | Multi-value index | Yes |
| `JSON` | Arbitrary JSON payload | Selective indexing | Dot-path access |
| `Blob` | Raw binary (image, audio, video, PDF) | Not indexed | Not filterable |
| `UUID` | 128-bit unique identifier | Hash index | Yes |

### Blob Storage

Blobs are not stored inline with vectors. They go to object storage with a reference pointer:

```
Record in segment:
  { id: "abc-123",
    text_embedding: [0.12, -0.34, ...],  — stored inline
    image_blob: BlobRef("s3://veda-data/blobs/abc-123/image.jpg") }
```

Segments stay lean for fast vector scanning. Blobs can be arbitrarily large.

### Segment Format

```
Segment (sealed, immutable)
├── vectors/
│   ├── text_embedding.vec     — raw vector data (mmap-friendly, aligned)
│   ├── image_embedding.vec
│   └── audio_embedding.vec
├── indexes/
│   ├── text_embedding.hnsw    — HNSW graph
│   ├── image_embedding.diskann — DiskANN index
│   └── audio_embedding.ivfpq  — IVF-PQ index
├── scalars/
│   ├── title.idx              — inverted index
│   ├── category.idx           — inverted index + bitmap
│   ├── price.range            — range index (B-tree)
│   └── created_at.range
├── text_search/
│   └── content.sa             — suffix array index (pattern search)
├── metadata.json              — segment stats, min/max, row count, schema version
├── delete_bitmap.bin          — bitset of soft-deleted rows
└── blob_refs.bin              — blob reference pointers
```

### Growing vs Sealed Segments

| Property | Growing Segment | Sealed Segment |
|----------|----------------|----------------|
| Location | In-memory on Data Nodes | Object storage (S3/MinIO/local) |
| Mutability | Append-only writes | Immutable |
| Index | Flat brute-force (small) | Full ANN indexes |
| Durability | WAL in message queue | Flushed to object storage |
| Query | Scanned alongside sealed | Loaded/mmap'd by Query Nodes |

**Write path**: Insert → Data Node appends to growing segment + writes WAL to message queue → segment reaches size threshold (512MB) → seal → Index Node builds ANN indexes → upload to object storage → Coordination updates shard routing.

**Compaction**: Background process merges small sealed segments, applies delete bitmaps, rebuilds indexes.

### Multi-Vector Records

A single record can have multiple vector fields with different dimensions and index types:

```rust
// "Find products similar to this image that also match this text description"
query = MultiVectorQuery {
    fields: [
        VectorQuery { field: "image_embedding", vector: [...], weight: 0.7 },
        VectorQuery { field: "text_embedding", vector: [...], weight: 0.3 },
    ],
    fusion: RRF { k: 60 },
    filter: "category = 'electronics' AND price < 500.0",
    limit: 20,
}
```

### Distance Metrics

**Tier 1 — Launch:**

| Metric | Use Case |
|--------|----------|
| Cosine similarity | Normalized embeddings (CLIP, Gemini, Nova, all text models) |
| Euclidean (L2) | Raw/unnormalized embeddings |
| Inner product (dot) | MIPS, recommendation systems |

**Tier 2 — V1.1:**

| Metric | Use Case |
|--------|----------|
| Hamming | Binary vectors (XOR + popcount) |
| Manhattan (L1) | High-dim data, outlier robustness |
| Jaccard | Set/sparse similarity |

All metrics are SIMD-optimized (AVX-512/AVX2 on x86, NEON on ARM). Normalized vector detection auto-optimizes to dot product internally.

---

## 4. Index Architecture

### Supported Index Types

| Index | Algorithm | Best For | Memory | Build Speed | Query Speed | Recall |
|-------|-----------|----------|--------|-------------|-------------|--------|
| **HNSW** | Hierarchical graph | General purpose, <100M/shard | High | Medium | Fastest | ~95%+ |
| **DiskANN** | Vamana graph on SSD | Billion-scale, cost-optimized | Low | Slow | Fast (SSD) | ~95%+ |
| **IVF-PQ** | Clustering + PQ | Large scale, memory-constrained | Low | Fast | Medium | ~90% |
| **Flat** | Brute-force | Small datasets, ground truth | Minimal | Instant | Slow | 100% |

### Auto-Selection by Collection Size

```
< 10K vectors    → Flat (exact search is fast enough)
10K - 10M        → HNSW (best recall/speed tradeoff)
10M - 100M       → HNSW with RaBitQ quantization
100M - 1B        → DiskANN (cost-effective at scale)
> 1B             → IVF-PQ with DiskANN fallback
```

### Quantization

| Method | Bits/Dim | Memory Reduction | Recall Impact | Use When |
|--------|----------|-----------------|---------------|----------|
| None (FP32) | 32 | 1x | 100% | Small datasets, exact search |
| Scalar (SQ8) | 8 | 4x | ~99% | < 512 dimensions |
| RaBitQ | 1-7 | 5-32x | 90-99% | 512-1024 dimensions (primary method) |
| Product Quantization | Variable | 10-64x | ~90% | > 1024 dimensions |

RaBitQ is auto-tuned by dimension. It is a 2024 SIGMOD breakthrough with provable error bounds, already adopted by Elasticsearch.

### Filtered Search

Adaptive strategy selection to avoid the latency cliff that plagues competitors:

```
If filter matches >50% of data  → Pre-filter, then ANN search
If filter matches 1-50%         → In-algorithm filter (check during HNSW traversal)
If filter matches <1%           → Post-filter with oversampling (search 10x limit, then filter)
```

### Pattern Search Index (Suffix Array)

A suffix array index can be built on any `String` field to enable SoftMatcha-style fuzzy pattern matching:

```toml
[collection.documents.fields.content]
type = "string"
index = "suffix_array"           # enables pattern search
fuzzy_matching = true            # allow substitution/insertion/deletion
```

**How it works:**
- At segment seal time, a suffix array is built over the concatenated text content of the field
- Suffix arrays enable O(m log n) exact lookup (m = pattern length, n = corpus size)
- Fuzzy matching extends this with dynamic corpus-aware pruning to avoid exponential blowup
- Disk-aware design keeps memory footprint manageable at billion-token scale

**Performance target:** Sub-second pattern search over trillion-token corpora (per SoftMatcha 2 benchmarks).

**Unique differentiator:** No OSS vector database offers this. Users get semantic search (vectors) + precise lexical search (suffix arrays) in one system.

### GPU Acceleration

- **Index building**: cuVS/CAGRA for GPU-accelerated HNSW construction (12x faster)
- **Search**: GPU search for batch queries (8x+ throughput)
- **Strategy**: Build on GPU with CAGRA, export to CPU-optimized HNSW for serving
- **Optional**: CPU-only deployment is first-class

### Per-Field Indexes

Each vector field gets its own index, independently configurable:

```toml
[collection.product_catalog.fields.text_embedding]
type = "vector"
dimension = 768
metric = "cosine"
index = "hnsw"
quantization = "rabitq_4bit"

[collection.product_catalog.fields.image_embedding]
type = "vector"
dimension = 512
metric = "cosine"
index = "diskann"
quantization = "sq8"
```

---

## 5. Query Engine & API Design

### Query Types

**1. Single-Vector Search**
```json
{
  "collection": "product_catalog",
  "query": {
    "vector": { "field": "image_embedding", "values": [0.12, -0.34] }
  },
  "filter": "category = 'electronics'",
  "limit": 20
}
```

**2. Multi-Vector Fusion Search**
```json
{
  "query": {
    "multi_vector": [
      { "field": "text_embedding", "values": [...], "weight": 0.4 },
      { "field": "image_embedding", "values": [...], "weight": 0.6 }
    ],
    "fusion": "rrf",
    "fusion_params": { "k": 60 }
  }
}
```

Fusion strategies: RRF (default), Weighted Sum, Intersection.

**3. Cross-Modal Search** — query in one modality, retrieve across others via shared embedding space.

**4. Hybrid Search (Dense + Sparse)** — combine ANN vector search with BM25/sparse matching.

**5. Scalar Filter-Only Query** — no vector search, just metadata filtering (for Niriksha browsing).

**6. Pattern Search (SoftMatcha-style)**
Suffix array-based fuzzy pattern matching across text fields. Handles semantic variations (substitution, insertion, deletion) while preserving word order. Sub-second search over billion-token corpora.
```json
{
  "collection": "documents",
  "query": {
    "pattern": {
      "field": "content",
      "text": "section 230 of the communications decency act",
      "fuzzy": true,
      "max_edits": 2,
      "preserve_order": true
    }
  },
  "limit": 50
}
```
No existing OSS vector database offers this. It complements semantic vector search with precise lexical retrieval — users can do both in one system.

### Filter Expression Language

SQL-like syntax parsed into an AST:

```
field = 'value'                    -- equality
field != 'value'                   -- not equal
field > 100                        -- numeric comparison
field BETWEEN 100 AND 500          -- range
field IN ('a', 'b', 'c')          -- set membership
field IS NULL / IS NOT NULL        -- null checks
field CONTAINS 'value'             -- array contains
field STARTS_WITH 'prefix'         -- string prefix
expr AND expr / expr OR expr       -- logical operators
NOT expr                           -- negation
json_field.nested.path = 'value'   -- JSON dot-path access
```

### API Protocols

| Protocol | Use Case |
|----------|----------|
| **gRPC** | SDK clients, service-to-service, lowest latency |
| **REST** | Convenience, curl-friendly, Niriksha backend calls |
| **SSE** | Niriksha streaming — embedding progress, index builds, live cluster events |

No CLI client. All interaction through Niriksha, SDKs, or REST/gRPC.

### REST Endpoints

```
# Collections
POST   /v1/collections                     -- create collection
DELETE /v1/collections/{name}              -- drop collection
GET    /v1/collections                     -- list collections
GET    /v1/collections/{name}              -- describe collection

# Data
POST   /v1/collections/{name}/search       -- vector search
POST   /v1/collections/{name}/insert       -- insert records
POST   /v1/collections/{name}/upsert       -- upsert records
DELETE /v1/collections/{name}/records       -- delete records
GET    /v1/collections/{name}/records/{id}  -- get by ID

# Auto-Embedding
POST   /v1/collections/{name}/embed-insert  -- upload raw data → auto-embed → insert

# Streaming (SSE)
GET    /v1/collections/{name}/embed-insert/stream   -- embedding progress
GET    /v1/collections/{name}/index/build/stream    -- index build progress
GET    /v1/collections/{name}/search/stream         -- streaming search results
GET    /v1/system/events                            -- cluster events

# Admin
GET    /v1/admin/health                    -- cluster health
GET    /v1/admin/nodes                     -- list nodes
POST   /v1/admin/nodes/{id}/drain          -- drain a node
POST   /v1/admin/backup                    -- create backup
GET    /v1/admin/backup                    -- list backups
POST   /v1/admin/backup/{id}/restore       -- restore backup
GET    /v1/admin/metrics                   -- Prometheus metrics
```

### Auto-Embedding with Modality Validation

Each embedding model registers supported modalities. Veda validates before sending data:

```toml
[embedding.models.clip-vit-l-14]
endpoint = "http://localhost:8080/v1/embeddings"
supported_modalities = ["text", "image"]
output_dimension = 768

[embedding.models.gemini-embedding-2]
endpoint = "https://api.google.com/v1/embeddings"
supported_modalities = ["text", "image", "video", "audio", "pdf"]
output_dimension = 3072
```

**Error responses:**

| Error Code | When |
|-----------|------|
| `MODALITY_NOT_SUPPORTED` | Model doesn't support the data type |
| `DIMENSION_MISMATCH` | Model output dim doesn't match field schema |
| `INPUT_TOO_LARGE` | Raw data exceeds model's max input size |
| `INVALID_FORMAT` | Unrecognized file format |
| `MODEL_UNAVAILABLE` | Embedding service endpoint is down |
| `EMBEDDING_TIMEOUT` | Model took too long |
| `PARTIAL_FAILURE` | Batch: some records succeeded, some failed |

Error responses include actionable suggestions:

```json
{
  "error": {
    "code": "MODALITY_NOT_SUPPORTED",
    "message": "Model 'clip-vit-l-14' does not support modality 'audio'.",
    "details": {
      "model": "clip-vit-l-14",
      "supported_modalities": ["text", "image"],
      "requested_modality": "audio"
    },
    "suggestion": "Configure a model that supports audio (e.g., 'gemini-embedding-2') or use /insert with pre-computed vectors."
  }
}
```

### Query Execution Flow

```
Client sends search request
  │
  ▼
Gateway → parse, validate, authenticate
  │
  ▼
Coordination Service → resolve shard routing
  │
  ▼
Fan-out to relevant Query Nodes (parallel)
  │
  ▼
Each Query Node:
  ├── Parse filter → build bitmap
  ├── Select search strategy (pre/in/post filter)
  ├── Search growing segment (brute-force)
  ├── Search sealed segments (ANN index)
  ├── Apply delete bitmap
  ├── Merge results across segments
  └── Return top-K to coordinator
  │
  ▼
Coordination Service:
  ├── Merge results across shards
  ├── Apply fusion (RRF/weighted) if multi-vector
  ├── Global top-K selection
  ├── Fetch full records if requested
  └── Resolve BlobRefs if include_blobs=true
  │
  ▼
Return response to client
```

---

## 6. Niriksha Dashboard

### Tech Stack

| Layer | Choice | Reason |
|-------|--------|--------|
| Framework | React 19 + TypeScript | Proven (data-viewer pattern) |
| Build | Vite | Fast HMR |
| State | Zustand | Per-item SSE updates (data-viewer pattern) |
| Animations | Framer Motion | Smooth transitions |
| Charts | D3.js + visx | Time-series, distributions |
| Scatter/3D | deck.gl or regl | GPU-accelerated Vector Atlas (50K+ points) |
| Tables | TanStack Table | Sortable, filterable, paginated |
| Styling | Tailwind CSS + CSS variables | Dark/light mode |
| HTTP | Axios + EventSource | REST + SSE |
| Audio | WaveSurfer.js | Waveform visualization |
| Video | Native HTML5 | Blob playback with time-segment markers |
| PDF | pdfjs-dist | Inline PDF rendering |
| Diff | react-diff-viewer | Schema/embedding comparison |

### Pages

#### 1. Dashboard Home

Cluster overview: collection count, total vectors, node count, uptime, avg latency, ingestion rate, search QPS, storage usage, recent activity feed.

#### 2. Collection Browser

Browse collections, inspect records with inline media rendering (image thumbnails, audio players, video players). Schema editor, bulk import/export, paginated record table with search.

#### 3. Search Playground

Interactive multimodal search: type text, upload images, record audio, upload video. Auto-embeds queries via Veda's embedding service. Results rendered by modality (image cards, audio waveforms, video players). Score visualization with color-coded badges. "Find Similar" button, query history, A/B comparison mode. **Pattern Search tab**: suffix array-based fuzzy text search with word-order preservation — search for exact phrases or near-variants across massive text corpora, complementing semantic vector search.

#### 4. Vector Atlas

Flagship visualization — explore embedding spaces:
- Dimensionality reduction: UMAP (default), t-SNE, PCA (computed server-side, streamed via SSE)
- Interactive scatter plot: pan, zoom, lasso select, hover to preview multimodal content
- Color by any scalar field
- Cross-field view: toggle between text_embedding and image_embedding spaces
- Cluster analysis with automatic boundary detection
- Lasso → Search: select a region, find all records

#### 5. Embedding Pipeline Monitor

Real-time SSE-streamed progress for auto-embedding jobs. Per-item status, error drill-down, model comparison, re-embedding triggers.

#### 6. Index Manager

Visualize and manage indexes: type, dimensions, metric, quantization, memory/disk usage, recall estimates, latency, build history. Rebuild, change type, tune parameters.

#### 7. System Health

Live cluster monitoring (SSE): per-node CPU/RAM/disk/GPU, QPS, ingest rate, shard distribution, replication status, WAL lag.

### Data Quality & RAG Evaluation Engine

**Layer 1 — Collection-Level Analysis ("AI Review")**

Analyzes entire collections to provide strategic recommendations:
- **Chunking recommendations per modality**: optimal strategy, segment duration, overlap
- **Embedding model recommendations**: suggest better models per data type
- **Data quality issues**: near-duplicates, short text, noisy audio, blank images, corrupt files
- **Embedding space health**: entropy, clustering analysis, cross-modal alignment scores

**Layer 2 — Per-Record Evaluation (LLM-as-a-Judge, SSE-streamed)**

Scores individual records across quality dimensions:

| Dimension | What It Measures |
|-----------|-----------------|
| RAG Usability (0-10) | Will this record be useful when retrieved as context for an LLM? |
| FAQ Usability (0-10) | Can standalone Q&A pairs be derived? |
| Data Quality (0-10) | Resolution, SNR, coherence, encoding — per modality |
| Cross-Modal Alignment (0-1) | Do paired modalities (text + image) agree semantically? |
| Embedding Quality (0-10) | Is the embedding representative of the content? |

**Data Quality scores** computed locally (no LLM): blur detection, SNR estimation, length checks, near-duplicate detection, language detection, entropy checks.

**RAG/FAQ scores** via LLM-as-a-Judge: structured rubric, 0-10 scale, streamed per-record via SSE, parallel evaluation.

**Multimodal AI Suggestion**: sends all modalities of a record to multimodal LLM, asks for alignment assessment and specific improvement suggestions.

**Actionable outputs**: low scores trigger suggestions (re-chunk, remove, re-embed, preprocess, deduplicate).

**User feedback loop**: emoji feedback (correct/incorrect/needs_edit/unclear/partial) + comments, stored per-record for rubric refinement and quality tracking.

### Authentication & Roles

- JWT-based authentication
- Roles: `admin` (full access), `editor` (search, insert, manage embeddings), `viewer` (read-only)

### Deployment

Niriksha ships with Veda. The Gateway serves the built frontend:

```
http://localhost:6340/niriksha/    (port 6340 = "VEDA" on phone keypad)
```

---

## 7. Deployment & Operations

### Server Binary

Minimal binary — server commands only, no client CLI:

```bash
veda serve                        # All-in-one dev mode (coordinator+query+data+index)
veda serve --role=coordinator     # Start as coordinator
veda serve --role=query           # Start as query node
veda serve --role=data            # Start as data node
veda serve --role=index           # Start as index node
veda serve --standalone           # All-in-one, local filesystem, no MinIO/NATS
veda init                         # Generate default veda.toml
veda version                      # Print version
veda validate-config              # Check veda.toml for errors
```

All data operations, admin tasks, and management go through REST/gRPC/Niriksha. No client CLI.

### Tier 1: Single-Node Development

```bash
# Install
curl -sSL https://get.veda.dev | sh

# Start (bundles MinIO + NATS in Docker containers)
veda serve
# ✓ MinIO started on :9000
# ✓ NATS started on :4222
# ✓ Veda started on :6340
# ✓ Niriksha: http://localhost:6340/niriksha/

# Without dependencies
veda serve --standalone

# Without bundled MinIO
veda serve --no-minio --storage.endpoint=s3.amazonaws.com

# Without Niriksha
veda serve --no-dashboard
```

### Tier 2: Docker Compose (Production)

```yaml
services:
  minio:
    image: minio/minio:latest
    command: server /data --console-address ":9001"
    ports: ["9000:9000", "9001:9001"]
    volumes: ["minio-data:/data"]

  nats:
    image: nats:latest
    command: ["--jetstream", "--store_dir=/data"]
    ports: ["4222:4222"]
    volumes: ["nats-data:/data"]

  coordinator:
    image: veda/veda:latest
    command: veda serve --role=coordinator
    ports: ["6340:6340", "6341:6341"]

  query-node:
    image: veda/veda:latest
    command: veda serve --role=query
    deploy:
      replicas: 2

  data-node:
    image: veda/veda:latest
    command: veda serve --role=data
    volumes: ["data:/var/veda/data"]

  index-node:
    image: veda/veda:latest
    command: veda serve --role=index
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### Tier 3: Kubernetes Helm Chart

```bash
helm repo add veda https://charts.veda.dev
helm install veda veda/veda

# With external S3
helm install veda veda/veda \
  --set minio.enabled=false \
  --set storage.backend=s3 \
  --set storage.endpoint=s3.amazonaws.com

# Scale
kubectl scale deployment veda-query --replicas=8
```

### Configuration — `veda.toml`

```toml
[cluster]
name = "production"
data_dir = "/var/veda/data"

[coordinator]
raft_port = 6342
raft_peers = ["node1:6342", "node2:6342", "node3:6342"]

[gateway]
rest_port = 6340
grpc_port = 6341
max_request_size = "256MB"

[storage]
backend = "minio"                 # minio | s3 | gcs | azure | local
endpoint = "localhost:9000"
bucket = "veda-data"
access_key = "${VEDA_STORAGE_ACCESS_KEY}"
secret_key = "${VEDA_STORAGE_SECRET_KEY}"

[message_queue]
backend = "nats"                  # nats | kafka
endpoint = "localhost:4222"

[query]
default_limit = 20
max_limit = 10000
timeout = "30s"
cache_size = "2GB"

[data]
segment_size = "512MB"
wal_sync = "fsync"
compaction_interval = "1h"

[index]
default_hnsw_m = 16
default_hnsw_ef_construction = 200
default_hnsw_ef_search = 128
gpu_enabled = false

[index.auto_select]
flat_threshold = 10000
hnsw_threshold = 100000000
diskann_threshold = 1000000000

[index.quantization]
auto = true
sq_threshold = 512
rabitq_threshold = 1024

[embedding]
enabled = false

[niriksha]
enabled = true
auth_enabled = true
jwt_secret = "${VEDA_JWT_SECRET}"

[niriksha.evaluation]
llm_endpoint = "http://localhost:8080/v1/chat/completions"
llm_model = "qwen-3.5-122b"
concurrency = 16

[logging]
level = "info"
format = "json"

[metrics]
enabled = true
prometheus_port = 9090
```

### Observability

**Prometheus Metrics:**
- `veda_query_latency_seconds` — histogram by collection, query type
- `veda_query_total` — counter by collection, status
- `veda_insert_total`, `veda_insert_latency_seconds`
- `veda_segment_count` — gauge by collection, state
- `veda_vector_count` — gauge by collection
- `veda_storage_bytes` — gauge by type (vector/scalar/blob)
- `veda_index_build_duration_seconds` — histogram by index type
- `veda_raft_leader_changes_total`
- `veda_node_status` — gauge
- `veda_embedding_requests_total`, `veda_embedding_latency_seconds`

**OpenTelemetry Tracing:**
- Distributed traces across all node types
- Span per: query parse, shard fan-out, ANN search, filter, fusion, fetch

**Structured Logging:**
- JSON format, configurable verbosity
- Slow query log, error logs with context

### Backup & Recovery

All via REST endpoints (accessible from Niriksha or curl):

```
POST /v1/admin/backup              -- create snapshot
GET  /v1/admin/backup              -- list snapshots
POST /v1/admin/backup/{id}/restore -- restore snapshot
```

Snapshots include: sealed segments, schema registry, shard routing table. Point-in-time recovery via WAL replay.

### Security

| Feature | Implementation |
|---------|---------------|
| Authentication | JWT (Niriksha), API keys (SDK/REST), mTLS (gRPC) |
| Authorization | RBAC — admin, editor, viewer per collection |
| Encryption at rest | Object storage encryption (S3 SSE, MinIO KMS) |
| Encryption in transit | TLS for external endpoints, optional mTLS inter-node |
| Secrets | Environment variables or Vault |
| Audit log | All mutations logged with user, timestamp, action |

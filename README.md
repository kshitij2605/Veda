<h1 align="center">Veda</h1>

<p align="center">
  <strong>A next-generation multimodal vector database for the era of RAG</strong>
</p>

<p align="center">
  <a href="#features">Features</a> |
  <a href="#quick-start">Quick Start</a> |
  <a href="#api-examples">API</a> |
  <a href="#python-sdk">Python SDK</a> |
  <a href="#niriksha-dashboard">Niriksha</a> |
  <a href="#benchmarks">Benchmarks</a> |
  <a href="#monitoring--observability">Monitoring</a> |
  <a href="CONTRIBUTING.md">Contributing</a>
</p>

---

## Features

### Storage + retrieval
- **Multimodal native** -- first-class support for text, image, audio, video, and PDF embeddings
- **4 ANN index types** -- HNSW, DiskANN (Vamana graph), IVF-PQ, and Flat (brute-force)
- **RaBitQ + SQ8 quantization** -- memory-efficient vector compression with minimal recall loss
- **SIMD-optimized distances** -- AVX2 fast paths with scalar fallback; 6 metrics (Cosine, Euclidean, DotProduct, Hamming, Manhattan, Jaccard)
- **SQL-like filter expressions** -- combine vector search with structured metadata filters
- **Multi-vector fusion** -- RRF, WeightedSum, and Intersection strategies for cross-modal retrieval
- **Pattern search** -- suffix array-backed substring and fuzzy matching on text fields
- **Built-in blob storage** -- store raw files alongside embeddings via MinIO/S3-compatible backends

### Distributed data plane (FoundationDB-shape)
- **Per-shard append-only log** (`veda-log`) -- record writes go through `veda-log`, not the metadata Raft. BookKeeper-style quorum replication (E, Qw, Qa) with epoch fencing; Tonic gRPC transport for cross-process followers. BookKeeper open-and-recover truncate at writer handover; `rebuild_ensemble` swaps out failed nodes; `ShardLog::copy_to` ships a snapshot to a joining follower.
- **`FollowerHealthMonitor`** -- edge-triggered probe loop drives ensemble change on follower failure (one orchestration call per outage, not per probe interval).
- **Hybrid Logical Clock + leased TSO** (`veda-time`) -- monotonic commit timestamps issued by a lease holder named in `_meta`. `MetaLeasedTso` swaps in automatically when Raft is configured.
- **MVCC versioning** -- every shard tracks per-doc `commit_ts` from growing through sealed segments. Reads at `?at=<ts>` return the snapshot view; the executor filters superseded / future versions inside the search walk. Every index (HNSW, Flat, IVF-PQ, DiskANN) has a native `search_filtered` that consults the visibility bitmap inside the candidate walk -- no 10×-oversample post-filter, no recall collapse on sparse snapshots. Disk-durable via per-shard MVCC WAL with periodic atomic checkpoint (snapshot + WAL truncate).
- **Percolator snapshot-isolation transactions** -- begin/commit/abort on `/v1/txn`, primary-and-secondary locks on the per-shard log, commit record in `_meta`. Replaces the v0 single-Raft txn stub. Background orphan-lock resolver included. Adya anomaly suite (write-skew allowed, lost-update prevented, read-only consistent) verifies SI invariants.
- **Atomic create-and-load** -- `POST /v1/collections` accepts `seed_records`; insert failure rolls back the collection via a compensating drop.
- **Real reindex pipeline** -- `POST /v1/collections/{name}/reindex` accepts `{chunker, re_embed, ...}`. Groups source records by `parent_doc_id`, sorts by `chunk_index`, reconstructs parent text, re-chunks via the supplied strategy, and (when `re_embed: true`) re-embeds via the registry's text-modality model. The atomic alias swap finalises the cutover at commit_ts. `SetCollectionAlias` is the same primitive underneath -- a name can be pointed at a sibling (e.g. `docs` → `docs@v2`) atomically. Returns `424 FAILED_DEPENDENCY` when `re_embed` is requested with no embedding endpoint configured.
- **Cross-region WAN log shipping** -- async replication of per-shard logs to replica regions with bounded staleness. Cross-region reads call `check_read_ts` on the live `/records` and `/search` path; reads ahead of the replica's `safe_ts` return `503` with explicit lag information. Operator-driven `promote_replica` cutover drains pending writes then bumps the topology epoch (fences old primaries). Admin REST: `GET/PUT /v1/admin/region-topology`, `POST /v1/admin/promote-region`, `GET/PUT /v1/admin/replica-safe-ts`.

### Pipelines + tooling
- **7 chunking strategies** -- MarkdownRecursive, MarkdownHeading (3 levels), Semantic, Sentence, FixedSize, Page, None
- **FlashParse OCR integration** -- end-to-end document processing with configurable chunking
- **Auto-embedding pipeline** -- automatic embedding generation with modality validation
- **Niriksha dashboard** -- built-in visualization and exploration UI for multimodal vector data
- **JWT authentication + RBAC** -- Admin, Editor, and Viewer roles with token-based auth
- **Full observability stack** -- Prometheus metrics, OpenTelemetry tracing, query explain, health checks, and a pre-built Grafana dashboard
- **Python SDK** -- `veda-client` package for idiomatic Python access
- **`veda-admin` CLI** -- offline maintenance tool (currently scaffold)

## Quick Start

### Docker Compose (recommended)

```bash
# Copy and configure environment
cp .env.example .env

# Core services (Veda + MinIO + NATS)
docker compose up

# With embedding service (HuggingFace TEI)
docker compose --profile embedding up

# With document OCR (FlashParse + GLM-OCR, text-only, ~8GB VRAM)
docker compose --profile ocr up

# With full OCR + VLM image captioning (~48GB+ VRAM)
docker compose --profile ocr --profile ocr-vlm up
```

This starts the Veda server, MinIO blob storage, and NATS message queue.

- Veda API: `http://localhost:6340`
- Niriksha dashboard: `http://localhost:5173`
- MinIO console: `http://localhost:9001`

**Optional profiles:**

| Profile | Services | GPU VRAM | Purpose |
|---|---|---|---|
| `embedding` | HuggingFace TEI + Multimodal Embedding (vLLM) | CPU + ~10GB | Text embedding (BGE-M3) + multimodal embedding (Qwen3-VL) |
| `ocr` | FlashParse + GLM-OCR | ~8GB | Text/table/formula OCR from documents |
| `ocr-vlm` | VLM (Qwen3.5-35B-A3B) | ~48GB | Image captioning and classification (use with `ocr`) |
| `monitoring` | OTel Collector, Prometheus, Mimir, Tempo, Loki, Grafana | CPU | Metrics, tracing, and log aggregation |

### Build from source

```bash
# Build the Rust engine
cargo build --release

# Run the server
cargo run --release --bin veda-server

# (Optional) Build the Niriksha dashboard
cd niriksha && npm install && npm run build
```

## Architecture

```
                  +-----------+
                  |  Niriksha |   (React dashboard)
                  +-----+-----+
                        |
                  +-----v-----+
 Clients -------> |  Gateway   |   REST, auth, MVCC reads at ?at=<ts>
                  +--+--+--+---+
                     |  |  |
       +-------------+  |  +-------------+
       |                |                |
       v                v                v
  +---------+    +--------------+   +------------------+
  | TSO/HLC |    |  _meta RAFT  |   |  PER-SHARD LOGS  |
  | veda-   |    |  schema /    |   |  veda-log        |
  | time    |    |  shard plan /|   |  - quorum write  |
  | leased  |    |  TSO lease / |   |    (E, Qw, Qa)   |
  | alloc.  |    |  txn commits |   |  - tonic gRPC    |
  +---------+    |  / aliases / |   |  - epoch-fenced  |
                 |  RegionTopo  |   |  - WAN ship      |
                 +------+-------+   +---------+--------+
                        |                     |
                        +----------+----------+
                                   v
                  +----------------------------------+
                  |  STORAGE / APPLIER (veda-storage)|
                  |  - LogApplier tails shard logs   |
                  |  - growing segment (commit_ts)   |
                  |  - sealed segment (commit_ts     |
                  |    sidecar) + MVCC WAL durable   |
                  |  - executor filters by read_ts   |
                  +----------------------------------+
```

The metadata Raft (`_meta`) carries only schemas, the shard plan, nodes, the TSO lease, txn commit records, collection aliases, and the region topology — NOT record bodies. Record writes go to per-shard `veda-log`; reads go to local `SegmentManager`s with per-doc `commit_ts` filtering.

**Crates:**

| Crate | Role |
|---|---|
| `veda-gateway` | REST API (axum), authentication, MVCC snapshot reads, txn endpoints, LogApplier bootstrap |
| `veda-query` | Query planning, multi-vector fusion, distributed fan-out, executor-side visibility filter |
| `veda-index` | ANN indexes (HNSW, DiskANN, IVF-PQ, Flat), scalar indexes — visibility-aware `search_filtered` |
| `veda-storage` | Vector storage, blob storage, growing + sealed segments with `commit_ts`, MVCC store + WAL, `LogApplier` |
| `veda-core` | Shared types, distance functions, filter expressions, schema (incl. `use_log_primary` / `use_mvcc`) |
| `veda-coordinator` | `_meta` state machine + apply path, Percolator `TxnCoordinator`, shard router, Raft transport |
| `veda-time` | Hybrid Logical Clock, `Tso` trait, `LocalTso`, `MetaLeasedTso`, `SwappableTso` |
| `veda-log` | Per-shard `ShardLog`, BookKeeper-style replication, Tonic transport for follower + WAN, cutover primitive |
| `veda-server` | Binary entry point, configuration, startup |
| `veda-admin` | Offline maintenance CLI (scaffold) |

See [documentation/rearchitecture-overview.md](documentation/rearchitecture-overview.md) for the layered design and current phase status.

## API Examples

### Create a collection

```bash
curl -X POST http://localhost:6340/v1/collections \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "name": "products",
    "fields": [
      {"name": "embedding", "type": "vector", "dimension": 768, "metric": "cosine"},
      {"name": "category", "type": "string", "filterable": true},
      {"name": "price", "type": "float64", "filterable": true},
      {"name": "description", "type": "string"}
    ]
  }'
```

### Insert records

```bash
curl -X POST http://localhost:6340/v1/collections/products/insert \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "records": [
      {
        "embedding": [0.1, 0.2, ...],
        "category": "electronics",
        "price": 299.99,
        "description": "Wireless headphones"
      }
    ]
  }'
```

### Embed + Insert (auto-embedding)

Let Veda generate embeddings automatically. The `auto_embed` map specifies which source fields to embed into which vector fields. Omit `model` to let Veda auto-select based on data modality.

```bash
curl -X POST http://localhost:6340/v1/collections/products/embed-insert \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "records": [
      {
        "description": "Wireless headphones with noise cancellation",
        "category": "electronics",
        "price": 299.99
      }
    ],
    "auto_embed": {
      "embedding": {
        "source_field": "description"
      }
    }
  }'
```

**Auto-selection rules:**
- Text-only data uses the text model (BGE-M3, CPU-friendly)
- If any field is non-text (image, video), **all** auto-selected mappings use the multimodal model (Qwen3-VL) to keep embeddings in a unified vector space
- Specify `"model": "model-name"` in any mapping to override auto-selection

### Search

```bash
curl -X POST http://localhost:6340/v1/collections/products/search \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "query": {"vector": {"field": "embedding", "values": [0.15, 0.22, ...]}},
    "limit": 10
  }'
```

### Filtered search

```bash
curl -X POST http://localhost:6340/v1/collections/products/search \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "query": {"vector": {"field": "embedding", "values": [0.15, 0.22, ...]}},
    "limit": 10,
    "filter": "category = '\''electronics'\'' AND price < 500"
  }'
```

## Python SDK

```bash
pip install veda-client
```

```python
from veda_client import VedaClient

client = VedaClient("http://localhost:6340")

# Create a collection
client.create_collection({
    "name": "products",
    "fields": [
        {"name": "embedding", "type": "vector", "dimension": 768, "metric": "cosine"},
        {"name": "category", "type": "string", "filterable": True},
        {"name": "price", "type": "float64", "filterable": True},
    ],
})

# Insert records
client.insert("products", [
    {"embedding": [0.1, 0.2, ...], "category": "electronics", "price": 299.99},
])

# Search
response = client.search(
    collection="products",
    field="embedding",
    vector=[0.15, 0.22, ...],
    limit=10,
    filter="category = 'electronics'",
)

for hit in response.results:
    print(f"{hit.id}: {hit.distance:.4f}")
```

## Document Processing

Veda provides **end-to-end document processing**: upload a document and Veda handles the entire pipeline -- OCR, chunking, embedding, and insertion into your collection. This is powered by [FlashParse](https://github.com/kshitij2605/FlashParse), which converts PDFs, DOCX, PPTX, images, and other document formats into structured markdown with extracted images.

**Pipeline:** Document upload --> FlashParse OCR --> markdown + images --> chunk (configurable strategy) --> embed (optional) --> insert into collection

### Chunking Strategies

| Strategy | Description | Default Size | Default Overlap |
|---|---|---|---|
| `markdown_recursive` | Recursively splits markdown by structure (default) | 2000 chars | 300 |
| `heading_1` | Splits on H1 headings | -- | -- |
| `heading_2` | Splits on H2 headings | -- | -- |
| `heading_3` | Splits on H3 headings | -- | -- |
| `semantic` | Word-overlap Jaccard similarity; splits on topic change | 2000 chars | 300 |
| `sentence` | Splits on sentence boundaries | 2000 chars | 300 |
| `fixed_size` | Fixed character count chunks | 2000 chars | 300 |
| `page` | Splits on page separators | -- | -- |
| `none` | No chunking; entire document as one chunk | -- | -- |

### Setup

1. Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

2. Start Veda with OCR enabled:

```bash
# Text-only OCR (FlashParse + GLM-OCR, ~8GB VRAM)
docker compose --profile ocr up

# Full OCR with VLM image captioning (~48GB+ VRAM)
docker compose --profile ocr --profile ocr-vlm up
```

The `ocr` profile starts FlashParse and GLM-OCR for text, table, and formula extraction. Add the `ocr-vlm` profile to also start the VLM service (Qwen/Qwen3.5-35B-A3B-FP8 on vLLM) for image captioning and classification.

**VLM configuration:** If you prefer to use an external VLM instead of the Docker-managed one, set these environment variables to point to any OpenAI-compatible API:
- `VLM_ENDPOINT` -- e.g., `http://your-server:8008/v1/chat/completions`
- `VLM_API_KEY` -- your API key (leave empty for local models)
- `VLM_MODEL` -- the model name to use

### Supported Formats

PDF, DOCX, PPTX, images (PNG, JPG, TIFF, BMP), and other formats supported by FlashParse.

### Single Document Processing

```bash
# 1. Create a collection for documents
curl -X POST http://localhost:6340/v1/collections \
  -H "Content-Type: application/json" \
  -d '{
  "name": "my-docs",
  "fields": [
    {"name": "content", "type": "string", "filterable": true},
    {"name": "content_embedding", "type": "vector", "dimension": 768, "metric": "cosine"},
    {"name": "source_file", "type": "string", "filterable": true},
    {"name": "chunk_index", "type": "int32"},
    {"name": "image_blob", "type": "blob"}
  ]
}'

# 2. Process a document (end-to-end: OCR -> chunk -> embed -> insert)
# Default (markdown-recursive chunking)
curl -X POST http://localhost:6340/v1/collections/my-docs/process-document \
  -F "file=@report.pdf"

# Semantic chunking
curl -X POST http://localhost:6340/v1/collections/my-docs/process-document \
  -F "file=@report.pdf" \
  -F "chunking_strategy=semantic"

# Split by H2 headings with custom chunk size
curl -X POST http://localhost:6340/v1/collections/my-docs/process-document \
  -F "file=@report.pdf" \
  -F "chunking_strategy=heading_2" \
  -F "chunk_size=3000"

# 3. Search the processed content
curl -X POST http://localhost:6340/v1/collections/my-docs/search -d '{
  "query": {"vector": {"field": "content_embedding", "values": [...]}},
  "limit": 5
}'
```

**Options:**
- `chunking_strategy=<strategy>` -- chunking strategy (default: `markdown_recursive`)
- `chunk_size=<chars>` -- maximum chunk size in characters (default: 2000)
- `chunk_overlap=<chars>` -- overlap between chunks in characters (default: 300)
- `skip_captions=true` -- skip VLM-based image captioning (faster processing)
- `dpi=300` -- set DPI for PDF rasterization (default: automatic)

```bash
curl -X POST http://localhost:6340/v1/collections/my-docs/process-document \
  -F "file=@report.pdf" \
  -F "skip_captions=true" \
  -F "dpi=300"
```

### Batch / Multiple File Processing

Process multiple documents in a single request:

```bash
curl -X POST http://localhost:6340/v1/collections/my-docs/process-documents \
  -F "files=@report1.pdf" \
  -F "files=@report2.pdf" \
  -F "files=@slides.pptx"
```

Returns aggregate stats with per-file results.

### Python SDK

```python
from veda_client import VedaClient

client = VedaClient("http://localhost:6340")

# Process a single document
with open("report.pdf", "rb") as f:
    result = client.process_document("my-docs", f)
    print(f"Inserted {result['records_inserted']} records")
    for chunk in result['chunks']:
        print(f"  [{chunk['type']}] {chunk['id']}: {chunk['preview'][:60]}")
```

If FlashParse is not running, the endpoint returns a clear error with setup instructions.

## Niriksha Dashboard

**Niriksha** is Veda's built-in data visualization and exploration dashboard for multimodal vector data. It provides:

- Collection overview with real-time statistics
- Interactive vector space visualization (2D/3D projections)
- Point-and-click search with filter builder
- Embedding inspector for any modality
- Cluster health and shard status monitoring
- Query performance analytics
- User and role management

Access it at `http://localhost:5173` when running via Docker Compose.

## Benchmarks

Measured on 10,000 vectors of dimension 768, single node, recall@10:

| Index | Recall@10 | Latency (p99) |
|---|---|---|
| HNSW (M=16, ef=200) | 95.5% | ~2ms |
| DiskANN (R=64, L=100) | 99.9% | ~5ms |
| IVF-PQ (nlist=256, m=48) | 89.2% | ~1ms |
| Flat (brute-force) | 100.0% | ~12ms |

Run the benchmark suite yourself:

```bash
cargo test --test ann_benchmark -- --ignored --nocapture
```

## Project Structure

```
project-mdb/
  crates/
    veda-core/          # Shared types, distances, filters
    veda-index/         # ANN and scalar indexes
    veda-storage/       # Vector storage, blobs, WAL
    veda-query/         # Query planning and fusion
    veda-gateway/       # REST API and auth
    veda-coordinator/   # Shards, cluster, compaction
    veda-server/        # Binary entry point
  niriksha/             # Visualization dashboard (React + TypeScript)
  sdk/
    python/             # Python SDK (veda-client)
  documentation/        # Architecture docs and ADRs
  research/             # Research notes and analysis
```

## Configuration

### Embedding Models

| Variable | Default | Description |
|---|---|---|
| `EMBEDDING_ENDPOINT` | `http://embedding:80` | Text embedding endpoint (OpenAI-compatible) |
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | Text embedding model name |
| `EMBEDDING_DIMENSION` | `1024` | Text embedding output dimension |
| `MULTIMODAL_EMBEDDING_ENDPOINT` | `http://multimodal-embedding:8000` | Multimodal embedding endpoint |
| `MULTIMODAL_EMBEDDING_MODEL` | `Qwen/Qwen3-VL-Embedding-2B` | Multimodal embedding model name |
| `MULTIMODAL_EMBEDDING_DIMENSION` | `2048` | Multimodal embedding output dimension |

### Blob Storage

| Variable | Default | Description |
|---|---|---|
| `VEDA_BLOB_BACKEND` | `minio` | `"minio"` (path-style, auto-create bucket) or `"s3"` (virtual-hosted, TLS) |
| `VEDA_BLOB_ENDPOINT` | `http://minio:9000` | Object store endpoint |
| `VEDA_BLOB_BUCKET` | `veda-blobs` | Bucket name |
| `VEDA_BLOB_ACCESS_KEY` | — | Access key |
| `VEDA_BLOB_SECRET_KEY` | — | Secret key |
| `VEDA_BLOB_REGION` | `us-east-1` | AWS region (for S3 mode) |

If `VEDA_BLOB_BACKEND` is unset, Veda falls back to local filesystem blob storage.

## Monitoring & Observability

### Starting the monitoring stack

```bash
docker compose --profile monitoring up
```

This starts six services alongside Veda:

| Service | Port | Purpose |
|---|---|---|
| OTel Collector | `:4317` (gRPC) | Receives traces, metrics, and logs via OTLP |
| Prometheus | `:9090` | Scrapes Veda's `/v1/admin/metrics` endpoint |
| Mimir | `:9009` | Long-term metrics storage (Prometheus remote-write) |
| Tempo | `:3200` | Distributed trace storage |
| Loki | `:3100` | Log aggregation |
| Grafana | `:3000` | Dashboards and exploration (pre-provisioned datasources and dashboard) |

### OpenTelemetry configuration

Veda exports traces and logs via OTLP when the following environment variable is set:

| Variable | Default | Description |
|---|---|---|
| `OTEL_EXPORTER_OTLP_ENDPOINT` | *(unset)* | OTLP gRPC endpoint (e.g. `http://otel-collector:4317`). OTel export is disabled when unset. |
| `OTEL_SERVICE_NAME` | `veda` | Service name attached to all telemetry |

When running with `--profile monitoring`, these are pre-configured in `docker-compose.yml`.

### Prometheus metrics

Veda exposes Prometheus metrics at `GET /v1/admin/metrics`. Metrics include request counts and durations, search latency, insert counts, embedding and blob operation durations, segment and compaction gauges, and error totals.

### Health checks

```bash
# Liveness -- always returns 200
curl http://localhost:6340/v1/admin/health

# Readiness -- returns 503 if unhealthy
curl http://localhost:6340/v1/admin/readiness
```

Both endpoints return:

```json
{
  "status": "healthy",
  "version": "0.1.0",
  "checks": {
    "minio": "healthy",
    "nats": "healthy",
    "embedding": "unconfigured",
    "multimodal_embedding": "unconfigured"
  }
}
```

Status is one of `healthy`, `degraded` (non-critical service down), or `unhealthy` (MinIO or NATS down). The readiness endpoint returns HTTP 503 when unhealthy.

### Query explain

Add `"explain": true` to any search request to get per-phase timing:

```bash
curl -X POST http://localhost:6340/v1/collections/products/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": {"vector": {"field": "embedding", "values": [0.15, 0.22, ...]}},
    "limit": 10,
    "explain": true
  }'
```

The response includes an `explain` object with `filter_ms`, `ann_search_ms`, `scoring_ms`, `fusion_ms`, and `total_ms` fields.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, coding standards, and the pull request process.

## License

Licensed under the [Apache License 2.0](LICENSE).

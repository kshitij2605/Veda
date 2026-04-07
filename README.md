<p align="center">
  <!-- TODO: Replace with actual Veda logo -->
  <img src="docs/assets/veda-logo-placeholder.png" alt="Veda" width="200" />
</p>

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
  <a href="CONTRIBUTING.md">Contributing</a>
</p>

---

## Features

- **Multimodal native** -- first-class support for text, image, audio, video, and PDF embeddings
- **4 ANN index types** -- HNSW, DiskANN (Vamana graph), IVF-PQ, and Flat (brute-force)
- **RaBitQ + SQ8 quantization** -- memory-efficient vector compression with minimal recall loss
- **SIMD-optimized distances** -- AVX2 fast paths with scalar fallback; 6 metrics (Cosine, Euclidean, DotProduct, Hamming, Manhattan, Jaccard)
- **SQL-like filter expressions** -- combine vector search with structured metadata filters
- **Multi-vector fusion** -- RRF, WeightedSum, and Intersection strategies for cross-modal retrieval
- **Pattern search** -- suffix array-backed substring and fuzzy matching on text fields
- **Built-in blob storage** -- store raw files alongside embeddings via MinIO/S3-compatible backends
- **Auto-embedding pipeline** -- automatic embedding generation with modality validation
- **Niriksha dashboard** -- built-in visualization and exploration UI for multimodal vector data
- **JWT authentication + RBAC** -- Admin, Editor, and Viewer roles with token-based auth
- **Prometheus metrics** -- operational observability out of the box
- **Python SDK** -- `veda-client` package for idiomatic Python access

## Quick Start

### Docker Compose (recommended)

```bash
docker compose up
```

This starts the Veda server, MinIO blob storage, and the Niriksha dashboard.

- Veda API: `http://localhost:8080`
- Niriksha dashboard: `http://localhost:5173`
- MinIO console: `http://localhost:9001`

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
   Clients ------> |  Gateway   |   (REST API, auth, routing)
                    +-----+-----+
                          |
                    +-----v-----+
                    |   Query    |   (planning, fusion, fan-out)
                    +-----+-----+
                          |
              +-----------+-----------+
              |                       |
        +-----v-----+          +-----v-----+
        |   Index    |          |  Storage   |
        | HNSW/Disk  |          | vectors,   |
        | ANN/IVF-PQ |          | blobs, WAL |
        +-----+-----+          +-----+-----+
              |                       |
        +-----v-----------------------v-----+
        |          Coordinator               |
        |   (shards, cluster, compaction)    |
        +------------------------------------+
```

**Crates:**

| Crate | Role |
|---|---|
| `veda-gateway` | REST API (axum), authentication, request routing |
| `veda-query` | Query planning, multi-vector fusion, distributed fan-out |
| `veda-index` | ANN indexes (HNSW, DiskANN, IVF-PQ, Flat), scalar indexes |
| `veda-storage` | Vector storage, blob storage, WAL, mmap I/O |
| `veda-core` | Shared types, distance functions, filter expressions |
| `veda-coordinator` | Shard management, cluster coordination, compaction |
| `veda-server` | Binary entry point, configuration, startup |

## API Examples

### Create a collection

```bash
curl -X POST http://localhost:8080/api/v1/collections \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "name": "products",
    "dimension": 768,
    "distance": "cosine",
    "modality": "text",
    "index_type": "hnsw"
  }'
```

### Insert vectors

```bash
curl -X POST http://localhost:8080/api/v1/collections/products/points \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "points": [
      {
        "id": "p1",
        "vector": [0.1, 0.2, ...],
        "payload": {"category": "electronics", "price": 299.99}
      }
    ]
  }'
```

### Search

```bash
curl -X POST http://localhost:8080/api/v1/collections/products/search \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "vector": [0.15, 0.22, ...],
    "top_k": 10
  }'
```

### Filtered search

```bash
curl -X POST http://localhost:8080/api/v1/collections/products/search \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "vector": [0.15, 0.22, ...],
    "top_k": 10,
    "filter": "category = '\''electronics'\'' AND price < 500"
  }'
```

## Python SDK

```bash
pip install veda-client
```

```python
from veda_client import VedaClient

client = VedaClient("http://localhost:8080", api_key="your-token")

# Create a collection
client.create_collection(
    name="products",
    dimension=768,
    distance="cosine",
    modality="text",
    index_type="hnsw",
)

# Insert vectors
client.insert(
    collection="products",
    points=[
        {"id": "p1", "vector": [0.1, 0.2, ...], "payload": {"category": "electronics"}},
    ],
)

# Search
results = client.search(
    collection="products",
    vector=[0.15, 0.22, ...],
    top_k=10,
    filter="category = 'electronics'",
)

for hit in results:
    print(f"{hit.id}: {hit.score:.4f}")
```

## Document Processing

Veda provides **end-to-end document processing**: upload a document and Veda handles the entire pipeline -- OCR, chunking, embedding, and insertion into your collection. This is powered by [FlashParse](https://github.com/kshitij2605/FlashParse), which converts PDFs, DOCX, PPTX, images, and other document formats into structured markdown with extracted images.

**Pipeline:** Document upload --> FlashParse OCR --> markdown + images --> chunk by headings --> embed (optional) --> insert into collection

### Setup

1. Copy `.env.example` to `.env` and configure your VLM backend:

```bash
cp .env.example .env
# Edit .env to set VLM_ENDPOINT, VLM_API_KEY, and VLM_MODEL
```

2. Start Veda with FlashParse enabled:

```bash
docker compose --profile ocr up
```

**VLM configuration:** FlashParse uses a vision-language model for OCR. Set these environment variables to point to any OpenAI-compatible API (vLLM, SGLang, OpenAI, Together AI, etc.):
- `VLM_ENDPOINT` -- e.g., `http://host.docker.internal:8008/v1/chat/completions`
- `VLM_API_KEY` -- your API key (leave empty for local models)
- `VLM_MODEL` -- the model name to use

### Supported Formats

PDF, DOCX, PPTX, images (PNG, JPG, TIFF, BMP), and other formats supported by FlashParse.

### Single Document Processing

```bash
# 1. Create a collection for documents
curl -X POST http://localhost:6340/v1/collections -d '{
  "name": "my-docs",
  "fields": [
    {"name": "content", "field_type": "string", "filterable": true},
    {"name": "content_embedding", "field_type": "vector", "dimension": 768, "metric": "cosine"},
    {"name": "source_file", "field_type": "string", "filterable": true},
    {"name": "chunk_index", "field_type": "integer"},
    {"name": "image_blob", "field_type": "blob"}
  ]
}'

# 2. Process a document (end-to-end: OCR -> chunk -> embed -> insert)
curl -X POST http://localhost:6340/v1/collections/my-docs/process-document \
  -F "file=@report.pdf"

# 3. Search the processed content
curl -X POST http://localhost:6340/v1/collections/my-docs/search -d '{
  "query": {"vector": {"field": "content_embedding", "values": [...]}},
  "limit": 5
}'
```

**Options:**
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

<!-- TODO: Replace with actual screenshot -->
![Niriksha Dashboard](docs/assets/niriksha-screenshot-placeholder.png)

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
  docs/                 # Design documents and plans
  documentation/        # ADRs and reference docs
  research/             # Research notes and analysis
  scripts/              # Build and deployment scripts
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, coding standards, and the pull request process.

## License

Licensed under the [Apache License 2.0](LICENSE).

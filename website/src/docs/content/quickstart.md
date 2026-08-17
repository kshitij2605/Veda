# Quick start

Get a working Veda cluster and run your first vector search in a few minutes. You need **Docker with Compose** and **git** — there is no published Veda image yet, so Compose builds the server from source in this repo.

![Two API calls against a live Veda — create a collection, insert three sentences — then a search in plain English finds the right one without sharing a single keyword.](/videos/quickstart.mp4)

## Get the repository

Compose builds Veda from this repo, so clone it first — `docker compose` has nothing to read otherwise.

```bash
git clone https://github.com/kshitij2605/Veda.git
cd Veda
```

## Start the core services

```bash
# Copy and configure environment
cp .env.example .env

# Core services (Veda + MinIO + NATS)
docker compose up
```

The **first** `up` compiles the server, so expect it to take a few minutes and a few GB of disk. Later starts reuse the built image and come up in seconds.

This starts the Veda server, MinIO blob storage, and the NATS message queue:

| Service | URL |
|---|---|
| Veda API | `http://localhost:6340` |
| Niriksha dashboard | `http://localhost:5173` |
| MinIO console | `http://localhost:9001` |

## Optional profiles

Add capabilities with Compose profiles:

```bash
# With embedding service (HuggingFace TEI)
docker compose --profile embedding up

# With document OCR (FlashParse + GLM-OCR, text-only, ~8GB VRAM)
docker compose --profile ocr up

# With full OCR + VLM image captioning (~48GB+ VRAM)
docker compose --profile ocr --profile ocr-vlm up

# With self-hosted web crawl + meta-search (crawl4ai + searxng)
docker compose --profile web up
```

| Profile | Services | GPU VRAM | Purpose |
|---|---|---|---|
| `embedding` | HuggingFace TEI + Multimodal Embedding (vLLM) | CPU + ~10GB | Text embedding (BGE-M3) + multimodal embedding (Qwen3-VL) |
| `ocr` | FlashParse + GLM-OCR | ~8GB | Text/table/formula OCR from documents |
| `ocr-vlm` | VLM (Qwen3.5-35B-A3B) | ~48GB | Image captioning and classification (use with `ocr`) |
| `web` | crawl4ai + searxng | CPU | Self-hosted web crawl + meta-search for [web connectors & agentic web search](/docs/web-connectors) |
| `monitoring` | OTel Collector, Prometheus, Mimir, Tempo, Loki, Grafana | CPU | Metrics, tracing, and log aggregation |

## Authenticate

Veda uses JWT authentication with role-based access (Admin, Editor, Viewer). Set `VEDA_BOOTSTRAP_ADMIN_PASSWORD` before the first start, then obtain a token:

```bash
curl -X POST http://localhost:6340/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "<your-bootstrap-password>"}'
```

Pass the returned JWT on subsequent requests as `Authorization: Bearer $TOKEN`.

## Create a collection

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

## Insert records

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

> With the `embedding` profile running you can skip pre-computing vectors entirely — see [auto-embedding](/docs/collections#auto-embedding-embed-insert).

## Search

```bash
curl -X POST http://localhost:6340/v1/collections/products/search \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "query": {"vector": {"field": "embedding", "values": [0.15, 0.22, ...]}},
    "limit": 10
  }'
```

## Next steps

- Model your data in [Collections & fields](/docs/collections)
- Pick the right index in [Indexes](/docs/indexes)
- Go beyond dense search in [Search & fusion](/docs/search)
- Turn PDFs into searchable vectors with the [Document pipeline](/docs/document-pipeline)

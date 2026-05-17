# Veda System Requirements

This document describes the hardware, software, and infrastructure requirements to run the Veda multimodal vector database system.

---

## Minimum Requirements (Core Only)

The core system runs **Veda + MinIO + NATS** — enough for vector storage, indexing, and search with pre-computed embeddings (BYOE: Bring Your Own Embeddings).

| Resource | Requirement |
|----------|-------------|
| **CPU** | 4+ cores (x86_64). AVX2 recommended for SIMD-accelerated distance computation |
| **RAM** | 4 GB minimum, 8 GB recommended |
| **Disk** | 10 GB for application + data volumes |
| **OS** | Linux (Debian/Ubuntu recommended). macOS for development |
| **Docker** | Docker Engine 24+ with Docker Compose v2 |
| **GPU** | Not required |
| **Network** | Ports 6340 (Veda API), 9200-9201 (MinIO), 4222 (NATS) |

**Start command:**
```bash
docker compose up -d
```

This launches 3 services: `veda`, `minio`, `nats`.

---

## Optional: Text Embedding Service

Adds automatic text embedding via HuggingFace Text-Embeddings-Inference. Enables `embed-insert` and `document` endpoints for text content.

| Resource | Requirement |
|----------|-------------|
| **CPU** | 4+ additional cores (runs on CPU, no GPU needed) |
| **RAM** | +4 GB (model loading + inference) |
| **Disk** | +2 GB for model cache (BAAI/bge-m3 default) |
| **Network** | Port 8092 |

**Start command:**
```bash
docker compose --profile embedding up -d
```

Adds 2 services: `embedding` (text, CPU), `multimodal-embedding` (GPU, see below).

---

## Optional: Multimodal Embedding Service

Adds image, audio, and video embedding via vLLM serving Qwen3-VL-Embedding-2B. Required for multimodal `embed-insert` and cross-modal search.

| Resource | Requirement |
|----------|-------------|
| **GPU** | 1x NVIDIA GPU with 4+ GB VRAM |
| **CUDA** | CUDA 12.0+ with NVIDIA Container Toolkit |
| **RAM** | +2 GB system RAM |
| **Disk** | +8 GB for model cache |
| **Network** | Port 8094 |

Included in the `embedding` profile alongside the text embedding service.

**Environment variables:**
| Variable | Default | Description |
|----------|---------|-------------|
| `MULTIMODAL_EMBEDDING_MODEL` | `Qwen/Qwen3-VL-Embedding-2B` | vLLM-compatible model |
| `MULTIMODAL_EMBEDDING_DIMENSION` | `2048` | Output vector dimension |
| `MULTIMODAL_EMBEDDING_GPU` | `3` | NVIDIA GPU device ID |
| `MULTIMODAL_EMBEDDING_GPU_UTIL` | `0.50` | GPU memory utilization (0.0-1.0) |
| `MULTIMODAL_EMBEDDING_MAX_MODEL_LEN` | `8192` | Max sequence length |

---

## Optional: OCR Services

Adds document processing (PDF, scanned images) via FlashParse + GLM-OCR. Required for the `/v1/collections/{name}/document` endpoint.

### FlashParse (Document Parser)

| Resource | Requirement |
|----------|-------------|
| **CPU** | 2+ cores |
| **RAM** | +2 GB |
| **Disk** | Minimal (processing is ephemeral) |
| **Network** | Port 8000 |

### GLM-OCR (OCR Model)

| Resource | Requirement |
|----------|-------------|
| **GPU** | 1x NVIDIA GPU with 8+ GB VRAM |
| **CUDA** | CUDA 12.0+ with NVIDIA Container Toolkit |
| **RAM** | +2 GB system RAM |
| **Disk** | +6 GB for model cache |
| **Network** | Port 8082 |

**Start command:**
```bash
docker compose --profile ocr up -d
```

Adds 2 services: `flashparse`, `glm-ocr`.

**Environment variables:**
| Variable | Default | Description |
|----------|---------|-------------|
| `GLM_OCR_GPU` | `1` | NVIDIA GPU device ID |
| `GLMOCR_API_HOST` | `glm-ocr` | GLM-OCR service hostname |
| `GLMOCR_API_PORT` | `8080` | GLM-OCR service port |

---

## Optional: Vision Language Model (VLM)

Adds image captioning and classification for FlashParse. Required only when OCR processing includes image understanding (not needed for text-only OCR).

| Resource | Requirement |
|----------|-------------|
| **GPU** | 1x NVIDIA GPU with 4+ GB VRAM |
| **CUDA** | CUDA 12.0+ with NVIDIA Container Toolkit |
| **RAM** | +2 GB system RAM |
| **Disk** | +8 GB for model cache |
| **Network** | Port 8008 |

**Start command:**
```bash
docker compose --profile ocr-vlm up -d
```

Adds 1 service: `vlm`.

**Environment variables:**
| Variable | Default | Description |
|----------|---------|-------------|
| `VLM_MODEL` | `Qwen/Qwen3.5-2B` | vLLM-compatible VLM model |
| `VLM_GPU` | `2` | NVIDIA GPU device ID |

---

## Optional: Monitoring Stack

Adds full observability with metrics, traces, and logs via the Grafana LGTM stack.

| Resource | Requirement |
|----------|-------------|
| **CPU** | 4+ additional cores |
| **RAM** | +4 GB (6 additional services) |
| **Disk** | +5 GB for metrics/trace/log storage |
| **GPU** | Not required |
| **Network** | Ports 4317-4318 (OTel), 9190 (Prometheus), 9109 (Mimir), 3200 (Tempo), 3100 (Loki), 3010 (Grafana) |

**Start command:**
```bash
docker compose --profile monitoring up -d
```

Adds 6 services: `otel-collector`, `prometheus`, `mimir`, `tempo`, `loki`, `grafana`.

---

## Full System (All Profiles)

Running every service requires:

| Resource | Requirement |
|----------|-------------|
| **CPU** | 16+ cores (x86_64 with AVX2) |
| **RAM** | 24+ GB system RAM |
| **GPU** | 3x NVIDIA GPUs (20+ GB total VRAM) |
| **Disk** | 50+ GB (application + model caches + monitoring data) |
| **CUDA** | 12.0+ with NVIDIA Container Toolkit |
| **Docker** | Docker Engine 24+ with Compose v2, NVIDIA container runtime |

**GPU allocation (default):**
| GPU | Service | VRAM |
|-----|---------|------|
| GPU 1 | GLM-OCR (0.9B) | ~8 GB |
| GPU 2 | VLM (Qwen3.5-2B) | ~4 GB |
| GPU 3 | Multimodal Embedding (Qwen3-VL-Embedding-2B) | ~4 GB |

All GPU assignments are configurable via environment variables (`GLM_OCR_GPU`, `VLM_GPU`, `MULTIMODAL_EMBEDDING_GPU`). Multiple services can share a GPU if VRAM permits — adjust `--gpu-memory-utilization` accordingly.

**Start command:**
```bash
docker compose --profile embedding --profile ocr --profile ocr-vlm --profile monitoring up -d
```

This launches all 14 services.

---

## Service Port Summary

| Service | Host Port | Purpose |
|---------|-----------|---------|
| Veda API | 6340 | REST API + Niriksha dashboard |
| MinIO API | 9200 | S3-compatible blob storage |
| MinIO Console | 9201 | MinIO web UI |
| NATS | 4222 | Message queue |
| NATS Monitor | 8222 | NATS monitoring dashboard |
| Text Embedding | 8092 | HuggingFace TEI (CPU) |
| Multimodal Embedding | 8094 | vLLM multimodal (GPU) |
| FlashParse | 8000 | Document parsing |
| GLM-OCR | 8082 | OCR model (GPU) |
| VLM | 8008 | Vision language model (GPU) |
| OTel Collector | 4317/4318 | OpenTelemetry gRPC/HTTP |
| Prometheus | 9190 | Metrics scraper |
| Mimir | 9109 | Long-term metrics storage |
| Tempo | 3200 | Distributed tracing |
| Loki | 3100 | Log aggregation |
| Grafana | 3010 | Visualization dashboard |

---

## Build from Source

If building Veda from source (without Docker):

| Requirement | Version |
|-------------|---------|
| **Rust** | Stable (edition 2024, Debian Bookworm toolchain recommended) |
| **Node.js** | 22+ (for Niriksha dashboard) |
| **npm** | Bundled with Node.js |
| **OS** | Linux x86_64 (Debian/Ubuntu). macOS for development |

```bash
# Build
cargo build --release --bin veda-server
cd niriksha && npm ci && npm run build

# Run
./target/release/veda-server /path/to/data 0.0.0.0:6340
```

The binary expects MinIO and NATS to be available at the endpoints configured via environment variables (`VEDA_STORAGE_ENDPOINT`, `VEDA_MQ_ENDPOINT`).

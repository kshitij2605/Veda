# Object Blob Store + Multimodal Embedding Design

**Date:** 2026-04-09
**Status:** Approved

## Problem

1. `S3BlobStore` is a stub that returns "not yet implemented". Blob storage only works on local filesystem (`LocalBlobStore`), despite MinIO running in Docker Compose with credentials wired into Veda.
2. `EmbeddingClient::embed_bytes()` returns mock deterministic vectors for non-text modalities (images, video). No real multimodal embedding inference exists.
3. Users cannot bring their own embedding models — the system hardcodes assumptions about the embedding service.

## Design

Three changes, all building on existing abstractions.

### 1. ObjectBlobStore

Replace `S3BlobStore` stub with `ObjectBlobStore` — a real implementation of the `BlobStore` trait backed by any S3-compatible object storage.

**Crate:** `veda-storage`. Add `rust-s3` crate dependency.

**Struct:** `ObjectBlobStore` implements `BlobStore` trait (`put/get/delete/exists/list`).

**Config via env vars:**

| Variable | Default | Description |
|----------|---------|-------------|
| `VEDA_BLOB_BACKEND` | `minio` | `"minio"` or `"s3"` |
| `VEDA_BLOB_ENDPOINT` | `minio:9000` | Object store endpoint (required for minio) |
| `VEDA_BLOB_BUCKET` | `veda-blobs` | Bucket name (auto-created for minio) |
| `VEDA_BLOB_ACCESS_KEY` | — | Access key |
| `VEDA_BLOB_SECRET_KEY` | — | Secret key |
| `VEDA_BLOB_REGION` | `us-east-1` | AWS region (for s3 mode) |

**Backend modes:**
- `minio`: Path-style URLs, custom endpoint, no TLS by default, auto-create bucket on init.
- `s3`: Virtual-hosted URLs, AWS region-based endpoint, TLS enabled.
- If `VEDA_BLOB_BACKEND` is unset, fall back to `LocalBlobStore` (dev/test).

**Wiring:** Add `blob_store: Arc<dyn BlobStore>` to `AppStateInner`. Construct in `start_server()` from env vars. Pass to `BlobManager` in insert/document handlers.

**Key format:** `{collection}/{record_id}/{field_name}` (unchanged).

**Naming:** No "S3" in our code surface. `rust-s3` is an internal dependency only.

### 2. Multimodal Embedding Client

Make `embed_bytes()` call real HTTP endpoints for image/video modalities.

**Qwen3-VL-Embedding-2B capabilities:**
- Supported: Text, Image, Screenshot, Video, mixed multimodal
- NOT supported: PDF (PDFs go through FlashParse OCR first, then text/images are embedded separately)
- Output dimension: 2048 (flexible 64-2048 via MRL)

**vLLM multimodal embedding API format** — chat messages on `/v1/embeddings`:

```json
{
  "model": "Qwen/Qwen3-VL-Embedding-2B",
  "messages": [{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,<b64>"}},
    {"type": "text", "text": "Represent the given image."}
  ]}],
  "encoding_format": "float"
}
```

Text-only uses standard OpenAI format: `{"model": "...", "input": "text"}`.

**Changes to `embedding_client.rs`:**

- `embed_bytes()`: If `is_real_endpoint()`, call new `embed_bytes_http(data, modality)`.
- `embed_bytes_http()`: Build chat-messages payload with base64 data URI based on modality:
  - `Image` → `data:image/jpeg;base64,...` in `image_url` content part
  - `Video` → `data:video/mp4;base64,...` in `video_url` content part
  - `Pdf` → error (should not reach here; FlashParse converts to text+images first)
- `is_real_endpoint()`: Remove `!contains("localhost")` exclusion so Docker-internal endpoints work.
- Response parsing stays the same: `{"data": [{"embedding": [...]}]}`.

### 3. Smart Model Auto-Selection

Users can explicitly choose a model, or Veda auto-selects based on data modality.

**3a. Environment-based registration:**

Existing text model env vars unchanged:
```
EMBEDDING_ENDPOINT=http://embedding:80
EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_DIMENSION=1024
```

New multimodal model env vars:
```
MULTIMODAL_EMBEDDING_ENDPOINT=http://multimodal-embedding:8000
MULTIMODAL_EMBEDDING_MODEL=Qwen/Qwen3-VL-Embedding-2B
MULTIMODAL_EMBEDDING_DIMENSION=2048
```

`load_embedding_from_env()` registers both if configured:
- Text model → `supported_modalities: [Text]`
- Multimodal model → `supported_modalities: [Text, Image, Video]`

Users deploy their own models by changing these env vars. Veda makes no assumptions about model names.

**3b. Selection logic (in `embed-insert` and `process-document`):**

1. User specifies `"model": "my-model"` → use that model (explicit override).
2. User omits model → auto-select:
   - Detect modality from field name or content.
   - `registry.find_by_modality(modality)` → pick first match.
   - For `Text` → BGE-M3 (registered first, CPU, cheaper).
   - For `Image`/`Video` → Qwen3-VL (only model supporting those modalities).
3. No model supports required modality → error listing available models.

**3c. Docker Compose — new `multimodal-embedding` service:**

```yaml
multimodal-embedding:
  profiles: ["embedding"]
  image: vllm/vllm-openai:nightly
  command:
    - "--model"
    - "${MULTIMODAL_EMBEDDING_MODEL:-Qwen/Qwen3-VL-Embedding-2B}"
    - "--runner"
    - "pooling"
    - "--port"
    - "8000"
    - "--trust-remote-code"
    - "--max-model-len"
    - "8192"
    - "--gpu-memory-utilization"
    - "0.50"
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            device_ids: ["${MULTIMODAL_EMBEDDING_GPU:-3}"]
            capabilities: [gpu]
```

**3d. PDF handling:**

PDFs flow through FlashParse (OCR → markdown + extracted images). Text chunks embed via text model, extracted images embed via multimodal model. No change to pipeline logic — just real embeddings instead of mocks.

## Files Changed

| File | Change |
|------|--------|
| `crates/veda-storage/Cargo.toml` | Add `rust-s3` dependency |
| `crates/veda-storage/src/blob_store.rs` | Replace `S3BlobStore` stub with `ObjectBlobStore` |
| `crates/veda-storage/tests/blob_test.rs` | Add `ObjectBlobStore` unit tests |
| `crates/veda-core/src/embedding_client.rs` | Implement `embed_bytes_http()`, fix `is_real_endpoint()` |
| `crates/veda-gateway/src/rest/state.rs` | Add `blob_store` to `AppStateInner`, extend `load_embedding_from_env()` |
| `crates/veda-gateway/src/server.rs` | Construct blob store from env vars |
| `crates/veda-gateway/src/rest/embed_insert.rs` | Auto-select model when omitted |
| `crates/veda-gateway/src/rest/document.rs` | Wire blob store, use auto-selected models |
| `crates/veda-gateway/src/rest/records.rs` | Wire blob store for insert/retrieve |
| `docker-compose.yml` | Add `multimodal-embedding` service |
| `.env` | Add `MULTIMODAL_EMBEDDING_*` vars |

## What's NOT Changing

- `BlobStore` trait — stays as-is
- `BlobManager` — stays as-is (already generic over `BlobStore`)
- `Modality` enum — stays as-is
- `EmbeddingModelConfig` / `EmbeddingModelRegistry` — stays as-is
- `AutoEmbedConfig` — stays as-is
- Search handlers — no changes
- Filter system — no changes

# Installation & config

Docker Compose is the recommended way to run Veda (see [Quick start](/docs/quickstart)). This page covers building from source, cluster modes, and the configuration surface.

## Build from source

```bash
# Build the Rust engine
cargo build --release

# Run the server
cargo run --release --bin veda-server

# (Optional) Build the Niriksha dashboard
cd niriksha && npm install && npm run build
```

## Cluster modes

Veda runs in two modes; the admin REST surface differs depending on which one is active.

- **Single-node** (default, no Raft) — Veda comes up with `VEDA_COORDINATOR` unset. Schemas, segments, MVCC, the transaction coordinator, atomic reindex, and the per-collection log all work. The `_meta`-backed admin endpoints (region topology, replica promotion, cluster membership) return `503` in this mode.
- **Raft** — set `VEDA_COORDINATOR=raft` to start `_meta` as a single-node openraft group. The 503 endpoints light up, and `GET /v1/admin/nodes` flips its `mode` field from `"single"` to `"raft"`.

| Endpoint | Single-node | Raft |
|---|---|---|
| `GET /v1/admin/health` / `readiness` | ✓ | ✓ |
| `GET /v1/admin/metrics` | ✓ | ✓ |
| `GET /v1/admin/nodes` | ✓ (1 node) | ✓ (N nodes) |
| `GET /v1/admin/region_topology` | ✓ (returns `{topology:null}`) | ✓ |
| `PUT /v1/admin/region_topology` | ✗ 503 | ✓ |
| `POST /v1/admin/region_topology/promote` | ✗ 503 | ✓ |
| `GET` / `POST /v1/admin/replica/safe_ts` | ✓ (local-only) | ✓ |
| `POST /v1/admin/cluster/join` / `DELETE /cluster/nodes/{id}` | ✗ 503 | ✓ |
| `POST /v1/admin/collections/{name}/mvcc_checkpoint` | ✓ | ✓ |
| `POST /v1/auth/login` | ✓ | ✓ |

If you don't intend to run multi-node, leave `VEDA_COORDINATOR` unset and ignore the 503 endpoints.

## Embedding models

Veda generates embeddings through OpenAI-compatible endpoints:

| Variable | Default | Description |
|---|---|---|
| `EMBEDDING_ENDPOINT` | `http://embedding:80` | Text embedding endpoint (OpenAI-compatible) |
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | Text embedding model name |
| `EMBEDDING_DIMENSION` | `1024` | Text embedding output dimension |
| `MULTIMODAL_EMBEDDING_ENDPOINT` | `http://multimodal-embedding:8000` | Multimodal embedding endpoint |
| `MULTIMODAL_EMBEDDING_MODEL` | `Qwen/Qwen3-VL-Embedding-2B` | Multimodal embedding model name |
| `MULTIMODAL_EMBEDDING_DIMENSION` | `2048` | Multimodal embedding output dimension |

## Blob storage

| Variable | Default | Description |
|---|---|---|
| `VEDA_BLOB_BACKEND` | `minio` | `"minio"` (path-style, auto-create bucket) or `"s3"` (virtual-hosted, TLS) |
| `VEDA_BLOB_ENDPOINT` | `http://minio:9000` | Object store endpoint |
| `VEDA_BLOB_BUCKET` | `veda-blobs` | Bucket name |
| `VEDA_BLOB_ACCESS_KEY` | — | Access key |
| `VEDA_BLOB_SECRET_KEY` | — | Secret key |
| `VEDA_BLOB_REGION` | `us-east-1` | AWS region (for S3 mode) |

If `VEDA_BLOB_BACKEND` is unset, Veda falls back to local filesystem blob storage.

## Memory scaling

By default Veda keeps all records in memory; RSS grows linearly with record count (roughly ~80 bytes of overhead per record plus the vector data itself). For collections that will exceed **~50,000 records**, enable the on-disk log path:

```bash
curl -X POST http://localhost:6340/v1/collections \
  -H "Content-Type: application/json" \
  -d '{
    "name": "large-collection",
    "fields": [ ... ],
    "use_log_primary": true
  }'
```

With `use_log_primary` enabled, record bodies are stored on disk via the per-shard append-only log and only the index structures are held in memory — the memory profile changes from O(records) to O(index).

## Authentication

Veda ships JWT authentication with three roles: **Admin**, **Editor**, and **Viewer**. Set `VEDA_BOOTSTRAP_ADMIN_PASSWORD` before the first start, then log in via `POST /v1/auth/login` and pass the returned token as `Authorization: Bearer <token>`.

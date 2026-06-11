# REST API

All endpoints are served under `http://<host>:6340/v1`. Authenticated requests pass a JWT as `Authorization: Bearer <token>` (obtain one via `POST /v1/auth/login`). The full machine-readable spec lives at `documentation/openapi.yaml` in the repository.

## Collections

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/collections` | Create a collection (supports `seed_records` for atomic create-and-load, `use_log_primary` for on-disk records; vector fields take `index`: `hnsw` · `diskann` · `ivf_pq` · `flat` · `rabitq` · `sq4` · `sq8`) |
| `GET` | `/v1/collections` | List collections |
| `GET` | `/v1/collections/{name}` | Get a collection's schema and stats |
| `DELETE` | `/v1/collections/{name}` | Drop a collection |
| `POST` | `/v1/collections/{name}/reindex` | Re-chunk / re-embed into a sibling and atomically swap the alias |

## Records & search

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/collections/{name}/insert` | Insert records (flat field→value maps) |
| `POST` | `/v1/collections/{name}/embed-insert` | Insert with server-side auto-embedding (`auto_embed` map) |
| `GET` | `/v1/collections/{name}/records` | List records (paginated; supports `?at=<ts>` snapshot reads) |
| `GET` | `/v1/collections/{name}/records/{id}` | Fetch one record |
| `POST` | `/v1/collections/{name}/search` | Search: `vector`, `bm25`, `sparse_vector`, `hybrid`, or `multi_vector` queries + `filter`, `rerank`, `explain`, `?at=<ts>` |
| `GET` | `/v1/collections/{name}/coverage` | Field coverage statistics |

### Search request shape

```json
{
  "query": {
    "vector": { "field": "embedding", "values": [0.15, 0.22] }
  },
  "limit": 10,
  "filter": "category = 'electronics' AND price < 500",
  "rerank": false,
  "explain": true
}
```

See [Search & fusion](/docs/search) for `multi_vector` fusion, hybrid queries, and snapshot reads.

## Documents

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/collections/{name}/process-document` | OCR → chunk → embed → insert one file (multipart `file`) |
| `POST` | `/v1/collections/{name}/process-documents` | Batch variant (multipart `files`) |
| `POST` | `/v1/chunk/preview` | Preview how text would be chunked under a strategy |

## Embedding & models

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/embed` | Embed text (or multimodal input) directly; optional `model` override |
| `POST` | `/v1/llm/complete` | LLM completion passthrough (used by Niriksha features) |
| `POST` | `/v1/route` | Modality routing helper |

## Transactions

Percolator-style snapshot-isolation transactions:

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/txn` | Begin / commit / abort transactional operations |

Write skew is allowed (snapshot isolation); lost updates are prevented; read-only transactions are consistent.

## Auth & saved objects

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/auth/login` | Exchange username/password for a JWT (Argon2id verification) |
| `GET/POST` | `/v1/saved_objects` | Saved objects used by the Niriksha dashboard |
| `GET/PUT/DELETE` | `/v1/saved_objects/{id}` | Manage one saved object |

## Admin

| Method | Path | Description |
|---|---|---|
| `GET` | `/v1/admin/health` | Liveness — always 200, with per-dependency checks |
| `GET` | `/v1/admin/readiness` | Readiness — 503 when unhealthy |
| `GET` | `/v1/admin/metrics` | Prometheus text-format metrics |
| `GET` | `/v1/admin/nodes` | Node list; `mode` reports `"single"` or `"raft"` |
| `GET/PUT` | `/v1/admin/region_topology` | Cross-region replication topology (writes require Raft) |
| `POST` | `/v1/admin/region_topology/promote` | Promote a replica region (Raft only) |
| `GET/POST` | `/v1/admin/replica/safe_ts` | Replica read watermark |
| `POST` | `/v1/admin/cluster/join` · `DELETE /v1/admin/cluster/nodes/{id}` | Raft membership changes |
| `POST` | `/v1/admin/collections/{name}/mvcc_checkpoint` | Atomic MVCC snapshot + WAL truncate |

Endpoints marked Raft-only return `503` in single-node mode — see [Cluster modes](/docs/installation#cluster-modes).

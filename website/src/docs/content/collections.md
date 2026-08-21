# Collections & fields

A collection is a named schema of typed fields plus the records stored against it. Vector fields hold embeddings; scalar fields hold filterable metadata; blob fields hold raw files.

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

## Field types

| Type | Notes |
|---|---|
| `vector` | Dense embedding; requires `dimension`, takes `metric`, optionally `model` for auto-embedding, and `index` to pick the ANN index (`hnsw` default; see [Indexes](/docs/indexes#choosing-an-index)) |
| `sparse_vector` | Sparse embedding (e.g. SPLADE-style term weights) |
| `multi_vector` | Multiple vectors per record |
| `string` | Text; supports `filterable: true` and `bm25: true` for lexical search |
| `int32` / `int64` / `float32` / `float64` | Numeric scalars; `filterable: true` enables range predicates |
| `bool` | Boolean scalar |
| `timestamp` | Time values |
| `json` | Arbitrary JSON payloads |
| `uuid` | UUID values |
| `blob` | Raw file bytes, stored in the configured blob backend (MinIO/S3) |

**Distance metrics** for vector fields: `cosine`, `euclidean`, `dot_product`, `hamming`, `manhattan`, `jaccard`.

## Field annotations

A field's type says how a value is stored. It does not say what the value
*means* — a `string` may hold an ISO country code, an email address, or free
prose, and nothing in the type distinguishes them. Annotations carry that, and
they are optional: a collection without them behaves exactly as before.

```json
{
  "name": "ctry_cd",
  "type": "string",
  "filterable": true,
  "annotations": {
    "description": "Country where the supplier is headquartered.",
    "semantic_type": "iso3166_alpha2",
    "value_domain": ["EE", "VN", "CL", "PL"],
    "sensitivity": "internal"
  }
}
```

| Key | Purpose |
|---|---|
| `description` | Human-readable meaning, shown to agents and MCP clients |
| `semantic_type` | What the value *is* — `iso3166_alpha2`, `email`, `currency_cents_usd` |
| `value_domain` | The values the field takes, or a representative sample |
| `sensitivity` | `public` (default), `internal`, or `restricted` |
| `pii` | `true` when the field holds personally-identifiable information |

`description`, `semantic_type` and `value_domain` are rendered by the
`describe_collection` tool, so an agent reading your schema learns that
`ctry_cd` holds `"EE"` rather than `"Estonia"` without having to discover it by
trial and error.

### Sensitivity is enforced, not advisory

`sensitivity: "restricted"` and `pii: true` are the two that change behaviour.
A field marked either way is **withheld from non-admin callers on every
retrieval path** — `search`, `filter_query`, `get_record` and `aggregate` —
and a query that *names* it in a filter or an aggregation is refused rather
than answered.

That second half matters. Withholding a column from results is not enough on
its own: a caller who cannot see `salary` could otherwise recover it by asking
for `salary > 100000` and reading the answer off the result count. Filtering
and grouping on a restricted field are therefore rejected outright.

`internal` is deliberately **advisory** — it is carried to agents and operators
and never enforced — so it can be used to record a distinction without changing
what anyone sees.

Because Veda owns ingestion, embedding and retrieval, this is enforced in the
engine. A stack that assembles a separate embedder and vector store has no
single place to put it.

## Insert records

Records are flat maps of field name to value. An `id` is generated when omitted.

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

## Auto-embedding (embed-insert)

Let Veda generate embeddings for you. The `auto_embed` map specifies which source fields to embed into which vector fields; omit `model` to let Veda auto-select based on data modality.

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

- Text-only data uses the text model (BGE-M3, CPU-friendly).
- If any field is non-text (image, video), **all** auto-selected mappings use the multimodal model (Qwen3-VL) to keep embeddings in a unified vector space.
- Specify `"model": "model-name"` in any mapping to override auto-selection.

## Read records

```bash
# List records (paginated)
curl "http://localhost:6340/v1/collections/products/records?limit=50"

# Fetch one record by id
curl http://localhost:6340/v1/collections/products/records/<id>
```

## Row-level security

A collection may declare an `rls_owner_field` at creation. When set, every read (search, get, list) is automatically restricted to records whose owner field equals the authenticated JWT subject — the predicate is AND-combined with any filter you pass — and inserts must set the owner to the caller's subject. The Admin role bypasses both, mirroring RBAC.

```bash
curl -X POST http://localhost:6340/v1/collections \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "name": "notes",
    "rls_owner_field": "owner",
    "fields": [
      {"name": "embedding", "type": "vector", "dimension": 1024, "metric": "cosine"},
      {"name": "owner", "type": "string", "filterable": true},
      {"name": "body", "type": "string"}
    ]
  }'
```

## Record TTL

Give any record an expiry by setting the reserved `_expires_at` field (unix seconds) — no schema change needed:

```bash
curl -X POST http://localhost:6340/v1/collections/cache/insert \
  -H "Content-Type: application/json" \
  -d '{
    "records": [
      {"embedding": [...], "key": "session-42", "_expires_at": 1765532800}
    ]
  }'
```

Expired records are excluded at **query time** from search, rerank, `GET /records/{id}`, and record listings. To physically purge them, schedule a `ttl_sweep` [job](/docs/rest-api#scheduled-jobs) for the collection.

## Atomic create-and-load

`POST /v1/collections` accepts `seed_records` to create a collection and load initial data atomically — if the insert fails, the collection creation is rolled back via a compensating drop.

## Large collections

For collections that will exceed ~50,000 records, set `use_log_primary: true` at creation so record bodies live on disk and memory stays O(index). See [Memory scaling](/docs/installation#memory-scaling).

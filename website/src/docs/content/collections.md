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
| `vector` | Dense embedding; requires `dimension`, takes `metric` and optionally `model` for auto-embedding |
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

## Atomic create-and-load

`POST /v1/collections` accepts `seed_records` to create a collection and load initial data atomically — if the insert fails, the collection creation is rolled back via a compensating drop.

## Large collections

For collections that will exceed ~50,000 records, set `use_log_primary: true` at creation so record bodies live on disk and memory stays O(index). See [Memory scaling](/docs/installation#memory-scaling).

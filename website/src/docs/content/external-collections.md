# External collections (read-in-place Parquet)

Point Veda at Parquet files you already have and query them **where they are** — no ingestion step, no second copy of your data, no duplicate storage bill.

This is a deliberate trade: you give up writes and the index-backed query types, and you get exactness, zero ingestion, and one less copy of your data to keep in sync.

![A live run: one call registers a collection over a Parquet file — dimension and row count read from the footer — then a write returns 405 and a BM25 query returns 400 with the reason.](/videos/external-collections.mp4)

## Register a collection over your files

```bash
curl -X POST http://localhost:6340/v1/collections \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -d '{
    "name": "lake_vectors",
    "fields": [],
    "external": {
      "format": "parquet",
      "uri": "embeddings/2026/",
      "id_field": "doc_id",
      "vector_field": "embedding",
      "metric": "Cosine"
    }
  }'
```

`fields` must be `[]` — an external collection infers its schema from the Parquet footer, and declaring a field alongside `external` is rejected with `400`.

| Field | Notes |
|---|---|
| `format` | Only `"parquet"` today. |
| `uri` | A key or key prefix in the server's configured object store, or `file:///abs/path`. Anything not ending in `.parquet` is treated as a **prefix** and selects every `*.parquet` object beneath it — anchored on `/`, so `embeddings/` never matches a sibling like `embeddings-v1/…`. |
| `id_field` | Utf8, LargeUtf8, Int32 or Int64; stringified into Veda's record id. Nulls are rejected. |
| `vector_field` | Must be `FixedSizeList<Float32>`. |
| `metric` | `Cosine`, `DotProduct` or `Euclidean`. **Note the casing** — this field deserializes the enum directly and, unlike an ordinary vector field's lowercase `metric`, rejects `"cosine"`. |

Registration requires **Admin**; without it the call returns `403`.

## Everything is validated up front

Schema, column types and vector dimension are read from the Parquet footer at registration. A missing column, a wrong Arrow type, a dimension that disagrees across files, or an unreadable URI **fails immediately with `400`** — not later, at query time, in front of a user.

## What you can query

**Dense vector search and scalar filters.** That is the whole surface.

- **Exact search, recall 1.0** — a linear scan using the same SIMD distance kernels as internal collections.
- **Scalar filtering uses the same filter evaluator** as internal collections, so external and internal filtering agree by construction rather than by convention.
- **Row-level security is enforced identically** to internal collections.
- **Never stale** — every query re-stamps the source and reloads on change, so a rewritten source file is never served from a stale copy. Concurrent reloads are single-flighted.
- Works against local files or any S3-compatible store (MinIO, AWS S3).

## What it deliberately does not do

- **It is read-only.** 18 mutating endpoints return `405`, including insert, delete, reindex, seal, optimize, `process-document` and the tuning-campaign routes. Dropping the collection and managing aliases still work.
- **`bm25`, `match_phrase`, `sparse`, `hybrid` and `rerank` all return `400`.** Parquet carries no inverted index and no sparse index.
- **Five read endpoints need locally stored records** and return `400`: `GET /records`, `/aggregate`, `/histogram`, `/field-stats` and `/evaluate`.
- **It is not as fast as an indexed collection.** Search is a linear scan; it trades speed for exactness and zero ingestion.

## Sizing

The vector column is memory-resident and search is a linear scan, so budget roughly **3 GB per 1M rows at 768 dimensions**. This is not a "scales to any size" feature — it is a "query the data you already have, exactly" feature.

Over S3/MinIO, reads are whole-object rather than ranged.

## Known rough edges

Published rather than discovered later:

- `explain: true` and `truncate_to` are silently **ignored** on an external collection rather than rejected — `explain` comes back `null`.
- A dense query's `field` is not validated against the registered `vector_field` (an internal collection does validate it).
- A hit's row id appears both as `SearchHit.id` and again as a column inside `fields`.
- External hits carry no `_guardrail_class` annotation, so [retrieval guardrails](/docs/retrieval-guardrails) do not annotate them.
- In Niriksha, only **Dense** applies to an external collection. The Sparse, BM25 and Hybrid modes are still offered but the server refuses them with a `400`, since Parquet carries no inverted or sparse index.

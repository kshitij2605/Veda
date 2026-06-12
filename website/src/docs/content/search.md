# Search & fusion

One search endpoint covers dense vectors, lexical BM25, sparse vectors, hybrid combinations, and cross-modal multi-vector fusion — with SQL-like filtering, reranking, per-phase explain timing, and snapshot reads.

## Dense vector search

```bash
curl -X POST http://localhost:6340/v1/collections/products/search \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "query": {"vector": {"field": "embedding", "values": [0.15, 0.22, ...]}},
    "limit": 10
  }'
```

## Filtered search

Combine vector similarity with SQL-like metadata predicates over `filterable` fields:

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

Scalar indexes (range / inverted / bitmap) pre-filter candidates before the ANN walk.

### JSON-path filtering

Filters can reach *into* `json` fields with dotted paths — the leading segment names the field, the rest traverse the nested document, and the leaf is compared as a scalar:

```bash
curl -X POST http://localhost:6340/v1/collections/products/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": {"vector": {"field": "embedding", "values": [...]}},
    "limit": 10,
    "filter": "meta.user.tier = '\''gold'\''"
  }'
```

A missing path simply matches nothing. JSON-path filters work on every filter site — search, record listing, and the [analytics endpoints](/docs/rest-api#analytics).

## Lexical, sparse, and hybrid queries

The `query` object accepts other retrieval modes over appropriately-typed fields:

- `bm25` — lexical ranking over `string` fields created with `bm25: true`.
- `sparse_vector` — sparse embeddings (e.g. learned term weights).
- `hybrid` — combine dense and lexical signals in one query; robust for rare names, acronyms, and precise technical vocabulary.

## Multi-vector fusion

Search several embedding spaces at once — text, image, and more — and fuse the ranked lists server-side:

```bash
curl -X POST http://localhost:6340/v1/collections/products/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": {
      "multi_vector": [
        {"field": "text_embedding",  "values": [...], "weight": 0.7},
        {"field": "image_embedding", "values": [...], "weight": 0.3}
      ],
      "fusion": "rrf",
      "fusion_params": {"k": 60}
    },
    "filter": "category = '\''electronics'\''",
    "limit": 10
  }'
```

| Strategy | How it works | When to use |
|---|---|---|
| `rrf` | Reciprocal Rank Fusion: `score(d) = Σ 1/(k + rank_i(d))` | Default. Robust when score scales differ across modalities |
| `weighted_sum` | Min-max normalize per field, then weighted linear combination | When you know the relative importance of each modality |
| `intersection` | Only returns results appearing in **all** field result lists | High-precision queries requiring cross-modal agreement |

## Reranking

Add `"rerank": true` to re-score the top candidates with a cross-encoder before returning — slower, but sharper precision at the very top of the list.

## Query explain

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

The response includes an `explain` object with `filter_ms`, `ann_search_ms`, `scoring_ms`, `fusion_ms`, and `total_ms`.

## Snapshot reads (time travel)

Every shard tracks per-document commit timestamps (MVCC). Read the collection as it existed at a point in time by adding `?at=<ts>` to `/records` and `/search` requests — either a raw HLC timestamp or a relative expression like `now-1h`:

```bash
curl -X POST "http://localhost:6340/v1/collections/products/search?at=now-1h" \
  -H "Content-Type: application/json" \
  -d '{"query": {"vector": {"field": "embedding", "values": [...]}}, "limit": 10}'
```

The executor filters superseded and future versions inside the search walk, so snapshot recall stays intact.

## Pattern search

Suffix-array-backed substring and fuzzy matching runs over text fields, enabling hybrid exact-text + vector retrieval in one engine.

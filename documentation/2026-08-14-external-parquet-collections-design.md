# External Parquet Collections (read-in-place) — Design

**Date:** 2026-08-14
**Status:** Approved for implementation
**Closes:** gap 1 of the 2026-08-12 competitive refresh — "storage format / lake
interoperability", the top-ranked gap by biggest-gap × highest-payoff.

## Context

`research/2026-08-12-competitive-refresh/00-summary.md` states the gap precisely:

> **Storage format / lake interoperability.** No Lance, no Vortex, no query-in-place.
> Milvus 3.0 points at a customer's existing lake with zero copy; **we require
> re-ingestion.** Widened since May — it moved from a performance footgun to an
> interoperability gap.

Its recommendation #8 reads "Lance as a third `StoredFieldsFormat` — and, beyond it,
read-in-place over a customer's existing lake", filed under "larger, needs a design
discussion first". This is that discussion, and it reaches a different conclusion about
which half to build.

### Why not Lance, and why not a new `StoredFieldsFormat`

Three findings from exploring the codebase, each measured rather than assumed:

**1. The stated gap is re-ingestion, and the `StoredFieldsFormat` half does not close
it.** `StoredFormat` (`crates/veda-storage/src/segment.rs:102`) selects the physical
layout of a segment's stored row payload — `Jsonl` or `Columnar`. Writing that payload
as Lance changes how *Veda's own* segments are encoded. It does nothing about querying
data Veda did not write, which is what "we require re-ingestion" means.

**2. It would give only partial interoperability anyway.** A Veda segment is a directory
of many files: `.vec` (dense vectors), `.spr` (sparse), `.mvec` (multi-vector), `.inv` /
`.bmp` / `.rng` (scalar indexes), `.bm25`, plus stored fields and `metadata.json`. Only
the stored fields are governed by `StoredFormat`. An external tool reading a Lance
stored-fields file would get the scalar columns and **not the embeddings** — most of what
makes a vector dataset worth reading.

**3. The dependency cost is disproportionate.** Measured by resolving each crate in an
empty project on the build host:

| dependency | transitive crates |
|---|---|
| Veda workspace today | 606 |
| `lance` | **684** |
| `arrow` | 102 |
| `parquet` | **91** |

`lance` alone resolves to more crates than the entire Veda workspace, dragging in the
full AWS, Azure and GCP SDKs, all of arrow-rs, tantivy, and 2D geometry libraries
(`i_overlay`, `i_shape`, `goosefs-sdk`). Roughly doubling the dependency surface is a
poor trade for a project whose stated differentiators include breadth of self-hosting
and Apache-2.0 end to end.

Parquet is also the right *first* format on the merits, not merely the cheap one:
Iceberg, Delta Lake and Hudi all store Parquet underneath, so "point at the customer's
existing lake" is overwhelmingly a Parquet problem. Lance is one vendor's format layered
above it.

**Lance is therefore deferred, not rejected.** Once external collections exist, adding a
second reader behind the same interface is additive. Revisit when a customer asks for
Lance specifically.

## Goals

1. Register a Parquet dataset as a read-only collection and query it **without
   ingesting or copying it**.
2. Support scalar filtering and **exact** vector search over it.
3. Work against both local paths and S3/MinIO, reusing existing machinery.
4. Never serve silently stale results when the source file changes.
5. State the scale and I/O limits plainly rather than implying the feature scales
   indefinitely.

## Non-goals

- Lance, Vortex, Iceberg or Delta table formats (later, additive).
- Changing `StoredFormat` or anything about how Veda writes its own segments.
- ANN over external data. No index is built, so there is no index to go stale.
- Writes of any kind to an external source.
- BM25, sparse or hybrid search over external data — see "Rejected query types".

## Design

### Component 1 — registration

`POST /v1/collections` gains one optional block, alongside the existing optional
per-collection opt-ins on `CollectionSchema` (`stored_format`, `use_log_primary`,
`use_mvcc`, `rls_owner_field` — all `#[serde(default)] Option<...>`, at
`crates/veda-core/src/schema.rs:226`):

```json
{
  "name": "lake_docs",
  "external": {
    "format": "parquet",
    "uri": "embeddings/",
    "id_field": "doc_id",
    "vector_field": "embedding",
    "metric": "cosine"
  }
}
```

```rust
/// Registration of a collection whose data Veda does not own.
/// `None` for every ordinary collection, so existing schemas
/// deserialize unchanged.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ExternalSource {
    /// Only `"parquet"` today. Present so a second reader can be added
    /// without changing the request shape.
    pub format: String,
    /// A key or key prefix **within the server's configured object store**,
    /// or `file:///abs/path` for a local file. A prefix — anything not
    /// ending in `.parquet` — selects every `*.parquet` object beneath it.
    ///
    /// A URI naming a bucket (`s3://bucket/prefix`) is **rejected**: see
    /// "Which bucket" below.
    pub uri: String,
    /// Column holding the record identifier. Any Arrow primitive or
    /// string type; stringified into Veda's record id.
    pub id_field: String,
    /// Column holding the dense vector: `FixedSizeList<Float32>`.
    pub vector_field: String,
    /// Distance metric. Not inferable from Parquet, so required.
    pub metric: DistanceMetric,
}
```

**Which bucket, and why a bucket cannot be named in the URI.** Veda holds exactly one
`ObjectBlobStore`, constructed against a single bucket from server configuration. An
authority in the URI would therefore be discarded, and `s3://customer-lake/embeddings/`
would silently read key `embeddings/` from **whatever bucket the server was configured
with** — succeeding against unrelated data if such a prefix happened to exist there.

That is the same silent-wrong-data failure this design works hard to avoid elsewhere, so
a URI with a non-empty authority is **rejected at registration** with a message saying the
bucket belongs in server configuration. External URIs are key prefixes within the store
Veda already has. Pointing Veda at a customer's lake means configuring its object store
against that bucket.

Two related rules that exist for the same reason:

- **A prefix is listed with a trailing `/`.** `ObjectBlobStore::list` is a bare
  `ListObjectsV2` with no delimiter, i.e. `starts_with` semantics, so listing
  `embeddings` without the slash would also match `embeddings-v1/…` and silently merge a
  stale snapshot into the dataset.
- **An empty prefix is an error**, not a whole-bucket sweep.

**Inferred vs required.** Column names, Arrow types and the vector dimension are read
from the Parquet footer at registration. `vector_field` and `metric` are required,
because neither can be inferred and guessing either wrong produces results that are
wrong without being obviously wrong.

**Validated at registration, not at query time:**

- the URI resolves and at least one Parquet file exists behind it → else `400`
- `id_field` and `vector_field` exist in the Arrow schema → else `400`, naming the
  columns that do exist
- `vector_field` is `FixedSizeList<Float32>` → else `400` naming the actual Arrow type
- every file behind a prefix agrees on schema and vector dimension → else `400` naming
  the first disagreeing file

The registered schema records the inferred dimension and the scalar columns, so
`GET /v1/collections/{name}` describes an external collection the same way it describes
an ordinary one.

### Component 2 — the reader

New module `crates/veda-storage/src/external_parquet.rs`. Sole owner of everything
Parquet-shaped; nothing else in the tree learns about Arrow.

```rust
/// A Parquet dataset registered as a read-only collection.
pub struct ExternalDataset {
    stamp: SourceStamp,
    dim: u32,
    /// Materialized vector column, row-aligned with `ids`.
    vectors: Vec<f32>,
    ids: Vec<String>,
    /// Scalar columns, materialized for predicate evaluation and
    /// for returning field values on a hit.
    scalars: HashMap<String, Vec<FieldValue>>,
}

impl ExternalDataset {
    /// Read and materialize from `store` at `uri`.
    pub fn load(store: &dyn BlobStore, source: &ExternalSource) -> Result<Self>;

    /// Exact top-k by `metric`, honouring `filter` if present.
    /// Returns `(id, distance)` sorted ascending by distance, ties
    /// broken by id, truncated to `k`.
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        metric: DistanceMetric,
        filter: Option<&FilterExpr>,
    ) -> Result<Vec<(String, f32)>>;

    pub fn dim(&self) -> u32;
    pub fn row_count(&self) -> usize;
    pub fn stamp(&self) -> &SourceStamp;
}
```

Scoring reuses the existing kernels in `crates/veda-core/src/simd.rs` —
`cosine_distance` and `dot_product` at lines 407 and 311 — so external search and
internal `Flat` search agree by construction rather than by a second implementation of
the same arithmetic.

Predicates reuse `FilterExpr` and the existing parser. Row-group statistics in the
Parquet footer prune whole row groups before decode where the predicate allows it;
correctness never depends on that pruning, only speed.

### Component 3 — freshness

```rust
/// Identity of a source at a point in time. Compared, never interpreted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceStamp {
    /// One entry per file behind the URI, sorted by key so the
    /// comparison is order-independent.
    entries: Vec<(String, BlobStat)>,
}

/// What a backend can cheaply report about an object without reading it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlobStat {
    pub size: u64,
    /// Local: mtime in milliseconds. Object storage: ETag if the backend
    /// supplies one, else Last-Modified. Opaque — only equality matters.
    pub version: String,
}
```

`BlobStore` (`crates/veda-storage/src/blob_store.rs:10`) currently exposes `put`, `get`,
`delete`, `exists` and `list`. It gains one method:

```rust
/// Cheap metadata for a key, without transferring the object.
fn stat(&self, key: &str) -> Result<BlobStat>;
```

implemented on both `LocalBlobStore` (`fs::metadata`) and `ObjectBlobStore` (HEAD).

Before serving any query, the handler re-stamps the source and compares. On mismatch the
cached `ExternalDataset` is dropped and reloaded. The cost is one metadata round-trip per
query; the benefit is that a changed lake file can never be served as though it were
current.

Caches live in `AppState` as `RwLock<HashMap<String, Arc<ExternalDataset>>>` keyed by
resolved collection name, mirroring how `segment_managers` is held.

### Component 4 — query dispatch

`AppState` holds `schemas` and `segment_managers` (`rest/state.rs:200-203`). An external
collection has a schema and no segment manager, so the search handler branches on the
schema **after** `tenant::scope` and `state.inner.aliases.resolve` — the ordering that
has produced three separate silent-data-loss incidents in this project and is therefore
called out explicitly here rather than left to the implementer.

- **Dense vector search** and **scalar filters** are served from `ExternalDataset`.
- **`limit`**, field projection and the ordinary response shape are unchanged, so an
  external collection is a drop-in for read traffic.

### Rejected query types, and why

Each of these needs an index that a Parquet file does not contain, so each returns `400`
with a message naming the reason rather than failing obscurely or returning empty:

| query | reason |
|---|---|
| `bm25`, `match_phrase` | needs the inverted index Veda builds at ingest |
| `sparse` | needs stored `SparseVec` columns and their index |
| `hybrid` | composes a dense leg with a bm25 leg; the bm25 leg is unavailable |
| `rerank` | resolves each candidate's text via internal record-storage lookups; external
collections hold only the dense vector and scalar columns read from the source Parquet
file, with no local record text to resolve |

`rerank` is checked before the dense search even runs, because `search`'s rerank-oversampling
already multiplies `req.limit` by 4 upstream of this dispatch — silently ignoring `rerank`
here would return that oversampled, untruncated, unreranked candidate set instead of the
caller's requested `limit`, which is a worse failure than a clear `400`.

### Writes

`insert`, `embed-insert`, `delete`, `update`, `reindex`, `seal` and
`process-document` against an external collection return **`405 Method Not Allowed`**
with a message saying the collection is external and read-only. Enforced in each
handler, not by convention — an unenforced read-only contract is the same class of
mistake as an unenforced sortedness contract.

## Stated limits

These belong in the API documentation, not only in this file:

- **Exact search is linear.** No index exists, so query latency scales with row count.
  Recall is 1.0 by construction. This is a deliberate trade: an ANN index over external
  data would be a derived copy of the vectors and would introduce staleness against a
  lake that can change underneath it.
- **The vector column is resident.** `row_count * dim * 4` bytes is the floor for the
  vector buffer alone — **~2.86 GiB (~3.07 GB decimal) for 1M × 768-dim `f32`**, confirmed
  by measurement (see "Measured result" below); the original "~3 GB" estimate held.
  Actual process memory is somewhat higher: ids (one heap-allocated `String` per row) and
  every scalar column (one `Vec<FieldValue>` per column, one heap allocation per string
  value) are additional, unmeasured overhead on top of this floor. Registering a dataset
  whose vector column alone exceeds available memory will fail on load rather than
  degrade; a dataset with many wide scalar columns can also exhaust memory well before
  the vector-column floor is reached, since that overhead is not bounded here.
- **Object reads are whole-object.** `BlobStore::get` returns the entire object, so
  Parquet's range-request advantage is lost over S3/MinIO. Column pruning still saves
  decode CPU and memory, but not I/O. Fixing this means either extending `BlobStore` with
  ranged reads or adopting `object_store`; neither is in scope here.
- **Change detection has a narrow blind spot on local files.** `BlobStore::stat` reports
  size plus a version derived from the filesystem mtime. On Linux, inode mtime is stamped
  from a *coarse* clock (`ktime_get_coarse_real_ts64`) that advances once per timer tick,
  so the nanosecond field is full-width but not full-resolution. A rewrite that keeps the
  file's **exact byte length** and completes **within a single tick** is therefore
  indistinguishable from no change, and a cached dataset would be served stale.

  This was measured, not assumed: raising the version from millisecond to nanosecond
  precision did not help, and a test writing two same-size payloads back to back failed
  ~87% of runs. It is pinned by a characterization test in `blob_store.rs` and documented
  on `BlobStat` itself.

  Accepted deliberately. Realistic external datasets are megabytes to gigabytes, so their
  writes span many ticks and the mtime does move; and Parquet is compressed, so a genuine
  content change producing a byte-identical file size is rare. Closing the gap properly
  means hashing the object's contents, which requires reading it in full on every query —
  destroying the entire point of a cheap `stat`. Size changes, and content changes
  separated by more than a tick, are detected reliably.

  Object storage does not share this limitation where the backend returns an ETag, which
  MinIO and S3 both do. A backend supplying neither ETag nor Last-Modified nor
  Content-Length would have the same blind spot.

- **A URI naming a bucket is rejected.** External URIs are key prefixes within the
  server's single configured object store, never a bucket in their own right — see
  "Which bucket, and why a bucket cannot be named in the URI" under Component 1. Pointing
  Veda at a customer's lake means configuring its object store against that bucket, then
  registering a bare key prefix.

- **Unsupported query types return `400` with a reason.** `bm25`, `match_phrase`,
  `sparse` and `hybrid` each need an index a Parquet file does not carry; `rerank` needs
  record text external collections do not hold locally. See "Rejected query types" above.

- **The collection is read-only: 18 mutating endpoints return `405`.** `insert`,
  `embed-insert`, `delete`, `reindex`, `rescan-guardrails`, `seal`, `optimize`,
  `process-document`, `process-documents` (batch), `ingest-url`, `ingest-media`,
  `ingest-logs`, `mvcc-checkpoint`, the tuning-campaign `start` and `apply-trial` routes,
  the web-connector run route, eval-set generation, and `set_auto_optimization` all reject
  an external collection with `405` and a message naming it external and read-only.
  Dropping the collection (`DELETE /v1/collections/{name}`) and alias management
  (`PUT /v1/admin/collections/{name}/alias`) still succeed, because both touch only
  Veda's own registration metadata, never the source data.

- **Two other search entry points do not support external collections.** Both reach a
  `DistributedQueryExecutor` directly rather than going through the dispatch branch in
  `rest/search.rs` that checks `schema.external()`:
  - **Auto-tuning campaigns** (`POST .../tuning/campaigns` and the trial-apply route) are
    rejected cleanly with `405` before a campaign ever starts — `start_campaign` and
    `apply_trial` both call `reject_if_external` up front. Auto-tuning simply does not run
    against an external collection; there is no silent partial behaviour.
  - **The agent retrieval tool** (`agent/tools/builtins.rs`) has no such guard. It looks
    up the collection's `DistributedQueryExecutor` in `AppState::query_executors`, which
    is never populated for an external collection (there is no segment manager, no
    executor, to register), and fails with a generic `"no query executor"` internal
    error rather than an explained rejection. An agent (or the MCP tool surface) calling
    retrieval against an external collection gets an unhelpful 500-class failure, not a
    `405` naming the reason.

- **Observability parity is partial.** A search against an external collection records:
  the `db.system` / `db.collection` / `retrieval.query_type` / `retrieval.top_k` /
  `retrieval.reranked` / `retrieval.has_filter` / `retrieval.result_count` span
  attributes, a query-log entry (`record_success`/`record_error`, exactly one per
  request, with the returned doc ids), and the `veda_search_duration{collection=...}`
  histogram. It does **not** record `retrieval.index_type` (correctly — no index exists)
  or the per-phase timing breakdown (`retrieval.filter_ms` / `.ann_search_ms` /
  `.scoring_ms` / `.fusion_ms` / `.total_ms`), because that breakdown is only computed
  when `explain: true` is honoured, and it is not — see the next bullet. Before Task 7's
  fix round, external searches recorded none of this at all and were invisible to
  per-collection observability, the query log, and any advisor/RCA tooling that reads
  from it; that gap is closed for everything except the `explain` timing breakdown.

- **Three request parameters are silently ignored rather than rejected**, where an
  internal collection validates all three:
  - `truncate_to` (Matryoshka/MRL dimensionality reduction) is dropped; the query always
    scores the full vector.
  - `explain: true` has no effect; the response's `explain` field is always `null`, and
    no per-phase timing is computed.
  - A dense query's `field` is never checked against the collection's registered
    `vector_field`. `validate_dense_query`, which performs this check for internal
    collections, sits downstream of the point where the external branch already
    returned.

- **Two hit-shape divergences from internal search results.** A row's id appears twice
  in an external hit: once as the top-level `SearchHit.id`, and again as the id column
  inside `fields` (`record_at` includes the id field alongside every other scalar column,
  and nothing strips it back out before the hit is built). And an external hit carries no
  `_guardrail_class` annotation — that annotation is only ever inserted on the internal
  search path.

## Measured result

Figures below are from `crates/veda-storage/tests/external_parquet_bench.rs`, run with
`PROTOC=/home/mac/bin/protoc timeout 3600 /home/mac/.cargo/bin/cargo test -p veda-storage --test external_parquet_bench -- --ignored --nocapture`
on the build host, local-disk `LocalBlobStore`, 768 dimensions throughout.

| rows | load | stamp | vector-column floor | parquet file size | floor / file size |
|---|---|---|---|---|---|
| 10,000 | 3.455837059s | 502.626µs | 30,720,000 B (29.297 MB) | 9,798,818 B (9.345 MB) | 3.135x |
| 100,000 | 20.748620277s | 535.033µs | 307,200,000 B (292.969 MB) | 98,044,409 B (93.502 MB) | 3.133x |

Filtered vs. unfiltered search, 100,000 rows, k=10, a ~50%-selective predicate on one
scalar column:

| | time | hits |
|---|---|---|
| unfiltered | 1.689646697s | 10 |
| filtered | 1.19760238s | 10 |

**Did the ~3 GB / 1M × 768-dim estimate hold? Yes.** `row_count * dim * 4` is exactly
linear by construction — 10k→100k rows scaled the floor 10.02x (29.297 MB → 292.969 MB) —
so it extrapolates cleanly to **2,929.69 MB (2.861 GiB, 3.072 GB decimal) at 1M rows**,
matching "~3 GB" closely. The "Stated limits" section above has been reworded to state
this precisely as a **floor for the vector column alone**, not total resident memory: ids
and scalar columns add further heap allocations this benchmark does not measure (no RSS
sampling is wired up here — see the caveat in the benchmark's own doc comment). The
original estimate was directionally correct and is not corrected further, only clarified.

**`stamp` cost.** ~500µs regardless of row count (502.626µs at 10k rows, 535.033µs at
100k), because `stamp` only calls `BlobStore::stat` per file behind the URI — filesystem
metadata, never the file's content. This is a per-query tax on every search against an
external collection (see Component 3), and at this cost it is negligible relative to the
linear scan `search` itself performs. This number is local-disk only: an S3/MinIO `stat`
is a network HEAD request, and its latency was not measured here — the "Risks" table's
"revisit only if measurement says otherwise" for the per-query `stat` round-trip remains
open for the object-storage case.

**Filtering cost.** Filtered search was **faster** than unfiltered in this benchmark
(1.198s vs. 1.690s, 0.709x), not slower. The review's concern — that `record_at`
materialises a full `Record` (a `HashMap` plus a `String` clone per column) for every row
scanned, ahead of the predicate's own selectivity — is real and does add work per row,
but at one scalar column it is cheaper than the 768-dimension cosine-distance computation
it lets a ~50%-selective predicate skip for excluded rows. This benchmark used a single
scalar column; it does not stress-test the case the review actually flagged, which is
column *count*, not filter selectivity — a source with many wide scalar columns would
make `record_at`'s per-row allocation cost scale with column count regardless of how
selective the predicate is, and that case is not measured here.

- Registration failures are `400` with the specific cause (missing column, wrong Arrow
  type, dimension disagreement across files, unreadable URI).
- A source that disappears between registration and query is `503` with the URI, not a
  panic and not an empty result — an empty result set is indistinguishable from a
  correct answer.
- A dimension mismatch between the query vector and the registered dimension is `400`,
  matching the existing `validate_dense_query` behaviour for ordinary collections.
- Load failures release the cache entry rather than caching a poisoned one.

## Testing

1. **Round-trip.** Write a Parquet file with the `parquet` crate inside the test, register
   it, search it, assert the expected ids come back. No fixture files checked in.
2. **Exactness.** Differential: score the same vectors directly with
   `veda_core::simd::cosine_distance` and assert the external search returns an identical
   ordering and identical distances. This is the same oracle discipline used for the
   block-max work.
3. **Filtering.** A predicate that excludes rows the vector search would otherwise rank
   first, asserting they are absent — so the filter cannot be silently ignored.
4. **Staleness.** Register, query, rewrite the Parquet file with different vectors, query
   again, assert the new data is visible. Verified to fail when the stamp comparison is
   removed, so the test cannot pass with the freshness check disabled.
5. **Prefix datasets.** Multiple Parquet files behind one prefix are searched as one
   collection; a file whose schema disagrees is rejected at registration.
6. **Write rejection.** Every mutating endpoint returns `405` on an external collection.
7. **Rejected query types.** `bm25`, `match_phrase`, `sparse` and `hybrid` each return
   `400` with a message naming the missing index.
8. **Alias and tenant resolution.** An external collection reached through an alias
   resolves correctly, guarding the invariant named in Component 4.

Plus the standing gates: `cargo fmt --all --check`, and
`cargo clippy -p veda-storage -p veda-gateway --all-targets -- -D warnings`.

## Risks

| Risk | Mitigation |
|---|---|
| `parquet` + `arrow` add ~91 crates and meaningful build time to a workspace that currently has neither. | Measured before committing to it, and an order of magnitude below `lance`. Confined to `veda-storage`; nothing else in the tree gains an Arrow dependency. |
| Memory residency makes a large registration an OOM risk. | Fail on load with a clear error rather than degrade. The limit is documented, and row count is reported at registration so an operator can see what they are about to load. |
| The per-query `stat` round-trip adds latency to object-storage queries. | One HEAD against a query that already scans the whole vector column linearly; it will not dominate. Revisit only if measurement says otherwise. |
| Read-only is enforced in many handlers, so one could be missed. | Test 6 covers every mutating endpoint by name; the same "one of several entry points" mistake was caught twice on the previous branch. |
| Scope creep toward "just add ANN". | Explicit non-goal. An index over external data is a derived copy and reintroduces staleness — the thing this design avoids. |

## Deferred, with the trigger for revisiting

- **Lance reader** — when a customer asks for Lance specifically. Additive behind the
  same interface.
- **Ranged object reads** — when object-storage query latency is measured and shown to be
  I/O-bound.
- **ANN over external data** — only with an explicit answer to what happens when the lake
  changes under a built index.
- **Iceberg / Delta manifests** — this design reads Parquet files, not table formats with
  snapshot semantics.

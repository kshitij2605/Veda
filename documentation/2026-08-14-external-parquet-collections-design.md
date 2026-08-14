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
    "uri": "s3://customer-lake/embeddings/",
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
    /// `file:///abs/path`, a bare local path, or `s3://bucket/prefix`.
    /// A prefix selects every `*.parquet` object beneath it.
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
- **The vector column is resident.** ~3 GB for 1M × 768-dim `f32`. Registering a dataset
  larger than available memory will fail on load rather than degrade.
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

## Error handling

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

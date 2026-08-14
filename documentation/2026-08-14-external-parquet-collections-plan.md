# External Parquet Collections — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Register a Parquet dataset as a read-only collection and query it in place — scalar filtering plus exact vector search — with no ingestion and no copy.

**Architecture:** One new module, `crates/veda-storage/src/external_parquet.rs`, owns everything Arrow-shaped: schema introspection, materialisation, freshness stamping and exact search. Everything else is reuse — `BlobStore` for local and S3/MinIO I/O, `veda_core::simd` for distance, `evaluate_record` for predicates. The gateway gains an `external` block on collection creation, a dispatch branch in search, and read-only enforcement on mutating handlers.

**Tech Stack:** Rust 2024, `parquet` + `arrow` (added to `veda-storage` only), existing `BlobStore`, `FilterExpr`/`evaluate_record`, `veda_core::simd`.

**Spec:** `documentation/2026-08-14-external-parquet-collections-design.md` (committed at `36c4548`).

## Global Constraints

- Toolchain pinned to **1.96.0** via `rust-toolchain.toml`. Do not change it.
- **Prefix every cargo command with `PROTOC=/home/mac/bin/protoc`.** The crate graph includes `veda-log`, whose build script needs protoc. Without it you get a confusing "Could not find protoc" error that is NOT a code problem.
- **Source of truth is remote:** `iitgpu11.hon.otsuka-shokai.co.jp:/home/mac/test/project-mdb`. Cargo is `/home/mac/.cargo/bin/cargo`. Edit locally, `scp` across, build and test over SSH. A local edit has no effect until copied back.
- **Run every cargo command in the FOREGROUND.** Never background a build or test, never poll. Adding Arrow makes first builds slow — many minutes is expected and is not a hang.
- **Never** `pkill`/`kill -9` cargo, rustc, or any process you did not start. This previously wedged the shared host into a reboot.
- **Never** run an unscoped workspace-wide `cargo test`/`cargo build`. Scope with `-p veda-storage` or `-p veda-gateway`.
- **Never** `git clean`, `git checkout -- .`, or `git stash` — a human works in this checkout concurrently. `git add` explicit paths only.
- Commit messages use conventional-commit prefixes and **must never mention Claude, AI assistance, or include a `Co-Authored-By` line**. Verify with `git log`.
- Branch `feat/external-parquet-collections` is checked out. Do not switch branches.
- **`arrow` and `parquet` are dependencies of `veda-storage` ONLY.** No other crate in the workspace may gain an Arrow dependency. This confines a 91-crate addition to one module and is a deliberate constraint from the spec.
- **Exact search only.** No ANN index over external data, ever, in this plan. It is an explicit non-goal: an index would be a derived copy and would go stale against a lake that changes.
- External collections are **read-only**. Every mutating endpoint rejects them.

---

### Task 1: Dependencies and the `ExternalSource` schema type

**Files:**
- Modify: `Cargo.toml` (workspace `[workspace.dependencies]`)
- Modify: `crates/veda-storage/Cargo.toml`
- Modify: `crates/veda-core/src/schema.rs` (`CollectionSchema` around line 226, `CollectionSchemaBuilder` around line 391)

**Interfaces:**
- Produces:
  - `pub struct ExternalSource { pub format: String, pub uri: String, pub id_field: String, pub vector_field: String, pub metric: DistanceMetric }` in `veda-core::schema`
  - `CollectionSchema::external(&self) -> Option<&ExternalSource>`
  - `CollectionSchema::is_external(&self) -> bool`
  - `CollectionSchemaBuilder::external(self, src: ExternalSource) -> Self`

  Every later task consumes these.

- [ ] **Step 1: Add the dependencies**

Run, on the remote host:

```bash
cd /home/mac/test/project-mdb
PROTOC=/home/mac/bin/protoc /home/mac/.cargo/bin/cargo add parquet arrow -p veda-storage
```

Then move both to `[workspace.dependencies]` in the root `Cargo.toml` and reference them
as `{ workspace = true }` in `crates/veda-storage/Cargo.toml`, matching how every other
shared dependency in this workspace is declared.

**Record the resolved versions in your report.** Do not invent a version number; use
whatever `cargo add` resolves and state it.

Confirm no other crate gained them:

```bash
grep -rn 'arrow\|parquet' crates/*/Cargo.toml
```

Expected: hits only in `crates/veda-storage/Cargo.toml`.

- [ ] **Step 2: Write the failing test**

Add to the `#[cfg(test)] mod tests` block in `crates/veda-core/src/schema.rs`:

```rust
#[test]
fn external_source_round_trips_through_serde() {
    let src = ExternalSource {
        format: "parquet".into(),
        uri: "s3://lake/embeddings/".into(),
        id_field: "doc_id".into(),
        vector_field: "embedding".into(),
        metric: DistanceMetric::Cosine,
    };
    let schema = CollectionSchema::builder("lake_docs")
        .external(src.clone())
        .build();
    let json = serde_json::to_string(&schema).unwrap();
    let back: CollectionSchema = serde_json::from_str(&json).unwrap();
    assert_eq!(back.external(), Some(&src));
    assert!(back.is_external());
}

#[test]
fn an_ordinary_collection_is_not_external() {
    let schema = CollectionSchema::builder("ordinary").build();
    assert_eq!(schema.external(), None);
    assert!(!schema.is_external());
}

#[test]
fn a_schema_saved_before_this_feature_still_loads() {
    // Older persisted schemas have no `external` key at all. `#[serde(default)]`
    // must let them load unchanged, so existing collections need no migration.
    let json = r#"{"id":"00000000-0000-0000-0000-000000000001","name":"legacy",
        "fields":[],"shard_count":1,"replication_factor":1}"#;
    let schema: CollectionSchema = serde_json::from_str(json).unwrap();
    assert!(!schema.is_external());
    assert_eq!(schema.name(), "legacy");
}
```

If `CollectionSchema::builder(...)` is not the actual constructor name, read the file and
use the real one — do not guess. Report any such correction.

- [ ] **Step 3: Run the test to verify it fails**

Run: `PROTOC=/home/mac/bin/protoc /home/mac/.cargo/bin/cargo test -p veda-core --lib schema`

Expected: compile error — `ExternalSource` does not exist.

- [ ] **Step 4: Add the type and the accessors**

In `crates/veda-core/src/schema.rs`, add above `CollectionSchema`:

```rust
/// Registration of a collection whose data Veda does not own and never
/// writes to. `None` for every ordinary collection.
///
/// The referenced files are queried in place: no ingestion, no copy, and
/// no index is built over them.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExternalSource {
    /// Only `"parquet"` is supported today. Present so a second reader
    /// can be added without changing the request shape.
    pub format: String,
    /// `file:///abs/path`, a bare local path, or `s3://bucket/prefix`.
    /// A prefix selects every `*.parquet` object beneath it.
    pub uri: String,
    /// Column holding the record identifier.
    pub id_field: String,
    /// Column holding the dense vector. Must be `FixedSizeList<Float32>`.
    pub vector_field: String,
    /// Distance metric. Not inferable from Parquet, so required.
    pub metric: DistanceMetric,
}
```

Add the field to `CollectionSchema`, following the existing optional-opt-in pattern
(`stored_format`, `use_log_primary`, `use_mvcc`, `rls_owner_field`):

```rust
    /// Set when this collection reads data Veda does not own. Absent for
    /// ordinary collections, so schemas written before this feature load
    /// unchanged.
    #[serde(default)]
    external: Option<ExternalSource>,
```

Add the accessors to `impl CollectionSchema`:

```rust
    /// The external source, if this collection reads data Veda does not own.
    pub fn external(&self) -> Option<&ExternalSource> {
        self.external.as_ref()
    }

    /// Whether this collection reads data Veda does not own. External
    /// collections are read-only and support only dense search and
    /// scalar filters.
    pub fn is_external(&self) -> bool {
        self.external.is_some()
    }
```

Add the builder method and the corresponding `external: Option<ExternalSource>` field on
`CollectionSchemaBuilder`, defaulting to `None`, and thread it through `build()`.

```rust
    /// Register this collection as reading an external dataset in place.
    pub fn external(mut self, src: ExternalSource) -> Self {
        self.external = Some(src);
        self
    }
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `PROTOC=/home/mac/bin/protoc /home/mac/.cargo/bin/cargo test -p veda-core --lib schema`
Expected: PASS, including every pre-existing schema test.

- [ ] **Step 6: Gates and commit**

```bash
PROTOC=/home/mac/bin/protoc /home/mac/.cargo/bin/cargo fmt --all --check
PROTOC=/home/mac/bin/protoc /home/mac/.cargo/bin/cargo clippy -p veda-core -p veda-storage --all-targets -- -D warnings
git add Cargo.toml Cargo.lock crates/veda-storage/Cargo.toml crates/veda-core/src/schema.rs
git commit -m "feat(schema): external source registration for read-in-place collections"
```

---

### Task 2: `BlobStore::stat`

**Files:**
- Modify: `crates/veda-storage/src/blob_store.rs` (trait at line 10, `ArcBlobStore` at 28, `LocalBlobStore` at 49, `ObjectBlobStore` at 158)

**Interfaces:**
- Produces:
  - `pub struct BlobStat { pub size: u64, pub version: String }`
  - `fn stat(&self, key: &str) -> Result<BlobStat>` on the `BlobStore` trait

  Task 5 consumes both.

- [ ] **Step 1: Write the failing test**

Add to the `#[cfg(test)] mod tests` block in `crates/veda-storage/src/blob_store.rs`:

```rust
#[test]
fn local_stat_reports_size_and_changes_with_content() {
    let dir = tempfile::tempdir().unwrap();
    let store = LocalBlobStore::new(dir.path());
    store.put("a.bin", b"hello").unwrap();
    let first = store.stat("a.bin").unwrap();
    assert_eq!(first.size, 5);

    // A different payload of a different length must produce a
    // different stat, or cache invalidation cannot work.
    store.put("a.bin", b"hello world").unwrap();
    let second = store.stat("a.bin").unwrap();
    assert_eq!(second.size, 11);
    assert_ne!(first, second);
}

#[test]
fn local_stat_on_a_missing_key_is_an_error() {
    let dir = tempfile::tempdir().unwrap();
    let store = LocalBlobStore::new(dir.path());
    assert!(store.stat("nope.bin").is_err());
}

#[test]
fn arc_blob_store_forwards_stat() {
    let dir = tempfile::tempdir().unwrap();
    let inner = std::sync::Arc::new(LocalBlobStore::new(dir.path()));
    inner.put("a.bin", b"xyz").unwrap();
    let wrapped = ArcBlobStore(inner.clone());
    assert_eq!(wrapped.stat("a.bin").unwrap(), inner.stat("a.bin").unwrap());
}
```

If `LocalBlobStore::new` takes different arguments, read the file and use the real
signature. Do not guess.

- [ ] **Step 2: Run the test to verify it fails**

Run: `PROTOC=/home/mac/bin/protoc /home/mac/.cargo/bin/cargo test -p veda-storage --lib blob_store`
Expected: compile error — `stat` and `BlobStat` do not exist.

- [ ] **Step 3: Add `BlobStat` and the trait method**

```rust
/// Cheap identity of a stored object, used to detect that a source
/// changed without transferring it.
///
/// `version` is opaque: only equality is meaningful. Local stores put an
/// mtime in it; object stores put an ETag, or Last-Modified when the
/// backend supplies no ETag.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlobStat {
    pub size: u64,
    pub version: String,
}
```

Add to the `BlobStore` trait:

```rust
    /// Metadata for a key without transferring the object.
    fn stat(&self, key: &str) -> Result<BlobStat>;
```

Implement on `LocalBlobStore` using `fs::metadata`, deriving `version` from the modified
time in milliseconds. Implement on `ArcBlobStore` by forwarding. Implement on
`ObjectBlobStore` using the `s3` crate's head-object call; prefer the ETag, fall back to
Last-Modified, and fall back to the empty string only if the backend supplies neither —
in which case `size` alone still detects most changes.

**Read the `s3` crate's actual head-object API in the version this workspace pins before
writing that call.** If it does not expose one, report that rather than inventing a
method name.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PROTOC=/home/mac/bin/protoc /home/mac/.cargo/bin/cargo test -p veda-storage --lib blob_store`
Expected: PASS.

Then confirm nothing else broke — every existing implementor of `BlobStore` must now
provide `stat`:

Run: `PROTOC=/home/mac/bin/protoc /home/mac/.cargo/bin/cargo test -p veda-storage`
Expected: PASS. If a test double elsewhere implements `BlobStore`, it needs `stat` too;
add it there and mention it in your report.

- [ ] **Step 5: Gates and commit**

```bash
PROTOC=/home/mac/bin/protoc /home/mac/.cargo/bin/cargo fmt --all --check
PROTOC=/home/mac/bin/protoc /home/mac/.cargo/bin/cargo clippy -p veda-storage --all-targets -- -D warnings
git add crates/veda-storage/src/blob_store.rs
git commit -m "feat(storage): BlobStore::stat for cheap change detection"
```

---

### Task 3: Read and validate a Parquet dataset

**Files:**
- Create: `crates/veda-storage/src/external_parquet.rs`
- Modify: `crates/veda-storage/src/lib.rs` (add `pub mod external_parquet;`)

**Interfaces:**
- Consumes: `ExternalSource` (Task 1), `BlobStore` (existing).
- Produces:
  - `pub struct ExternalDataset`
  - `pub fn load(store: &dyn BlobStore, source: &ExternalSource) -> Result<ExternalDataset>`
  - `ExternalDataset::{dim() -> u32, row_count() -> usize, ids() -> &[String]}`

  Tasks 4 and 5 consume these.

- [ ] **Step 1: Write the failing tests**

Create `crates/veda-storage/src/external_parquet.rs` with only a test module for now.
The tests write their own Parquet with the `parquet` crate — **no fixture files are
checked in**, so the tests are self-describing and cannot drift from a stale binary.

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc as StdArc;

    use arrow::array::{ArrayRef, FixedSizeListArray, Float32Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::ArrowWriter;

    use crate::blob_store::LocalBlobStore;
    use veda_core::types::DistanceMetric;

    /// Build a Parquet file with `n` rows: a string id column and a
    /// fixed-size-list vector column of dimension `dim`.
    fn write_parquet(path: &std::path::Path, n: usize, dim: usize) {
        let ids: Vec<String> = (0..n).map(|i| format!("doc{i}")).collect();
        let mut flat: Vec<f32> = Vec::with_capacity(n * dim);
        for i in 0..n {
            for d in 0..dim {
                flat.push((i * dim + d) as f32 / 10.0);
            }
        }
        let values = Float32Array::from(flat);
        let vec_field = StdArc::new(Field::new("item", DataType::Float32, false));
        let vectors =
            FixedSizeListArray::try_new(vec_field.clone(), dim as i32, StdArc::new(values), None)
                .unwrap();
        let schema = StdArc::new(Schema::new(vec![
            Field::new("doc_id", DataType::Utf8, false),
            Field::new(
                "embedding",
                DataType::FixedSizeList(vec_field, dim as i32),
                false,
            ),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                StdArc::new(StringArray::from(ids)) as ArrayRef,
                StdArc::new(vectors) as ArrayRef,
            ],
        )
        .unwrap();
        let file = std::fs::File::create(path).unwrap();
        let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
    }

    fn source(uri: &str) -> ExternalSource {
        ExternalSource {
            format: "parquet".into(),
            uri: uri.into(),
            id_field: "doc_id".into(),
            vector_field: "embedding".into(),
            metric: DistanceMetric::Cosine,
        }
    }

    #[test]
    fn loads_ids_and_vectors() {
        let dir = tempfile::tempdir().unwrap();
        write_parquet(&dir.path().join("a.parquet"), 10, 4);
        let store = LocalBlobStore::new(dir.path());
        let ds = load(&store, &source("a.parquet")).unwrap();
        assert_eq!(ds.row_count(), 10);
        assert_eq!(ds.dim(), 4);
        assert_eq!(ds.ids()[0], "doc0");
        assert_eq!(ds.ids()[9], "doc9");
    }

    #[test]
    fn rejects_a_missing_vector_column() {
        let dir = tempfile::tempdir().unwrap();
        write_parquet(&dir.path().join("a.parquet"), 4, 4);
        let store = LocalBlobStore::new(dir.path());
        let mut src = source("a.parquet");
        src.vector_field = "not_there".into();
        let err = load(&store, &src).unwrap_err().to_string();
        assert!(
            err.contains("not_there") && err.contains("embedding"),
            "error must name the missing column AND what is available, got: {err}"
        );
    }

    #[test]
    fn rejects_a_vector_column_of_the_wrong_arrow_type() {
        let dir = tempfile::tempdir().unwrap();
        write_parquet(&dir.path().join("a.parquet"), 4, 4);
        let store = LocalBlobStore::new(dir.path());
        let mut src = source("a.parquet");
        // The id column is Utf8, not FixedSizeList<Float32>.
        src.vector_field = "doc_id".into();
        let err = load(&store, &src).unwrap_err().to_string();
        assert!(
            err.contains("doc_id"),
            "error must name the offending column, got: {err}"
        );
    }

    #[test]
    fn rejects_a_missing_id_column() {
        let dir = tempfile::tempdir().unwrap();
        write_parquet(&dir.path().join("a.parquet"), 4, 4);
        let store = LocalBlobStore::new(dir.path());
        let mut src = source("a.parquet");
        src.id_field = "nope".into();
        assert!(load(&store, &src).is_err());
    }

    #[test]
    fn reads_every_parquet_file_behind_a_prefix() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("part")).unwrap();
        write_parquet(&dir.path().join("part/a.parquet"), 5, 4);
        write_parquet(&dir.path().join("part/b.parquet"), 7, 4);
        let store = LocalBlobStore::new(dir.path());
        let ds = load(&store, &source("part/")).unwrap();
        assert_eq!(ds.row_count(), 12, "both files must be read as one dataset");
    }

    #[test]
    fn rejects_files_behind_a_prefix_that_disagree_on_dimension() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("part")).unwrap();
        write_parquet(&dir.path().join("part/a.parquet"), 5, 4);
        write_parquet(&dir.path().join("part/b.parquet"), 5, 8);
        let store = LocalBlobStore::new(dir.path());
        let err = load(&store, &source("part/")).unwrap_err().to_string();
        assert!(
            err.contains("4") && err.contains("8"),
            "error must name both dimensions, got: {err}"
        );
    }

    #[test]
    fn an_empty_prefix_is_an_error_not_an_empty_dataset() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("part")).unwrap();
        let store = LocalBlobStore::new(dir.path());
        // An empty result set is indistinguishable from a correct answer,
        // so registering a URI with no Parquet behind it must fail loudly.
        assert!(load(&store, &source("part/")).is_err());
    }
}
```

Add `pub mod external_parquet;` to `crates/veda-storage/src/lib.rs`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PROTOC=/home/mac/bin/protoc /home/mac/.cargo/bin/cargo test -p veda-storage --lib external_parquet`
Expected: compile errors — `load` and `ExternalDataset` do not exist.

- [ ] **Step 3: Implement loading**

Prepend to `crates/veda-storage/src/external_parquet.rs`:

```rust
//! Read-in-place Parquet datasets.
//!
//! A collection registered with an [`ExternalSource`] is queried directly
//! from the customer's files: nothing is ingested, nothing is copied into a
//! segment, and no index is built. Exact search only — see the design doc
//! at `documentation/2026-08-14-external-parquet-collections-design.md`.
//!
//! This module is the ONLY place in the workspace that knows about Arrow.

use std::collections::HashMap;

use bytes::Bytes;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

use veda_core::error::{Result, VedaError};
use veda_core::record::FieldValue;
use veda_core::schema::ExternalSource;

use crate::blob_store::BlobStore;

/// A Parquet dataset registered as a read-only collection.
pub struct ExternalDataset {
    dim: u32,
    /// Flattened vector column, `dim` floats per row, row-aligned with `ids`.
    vectors: Vec<f32>,
    ids: Vec<String>,
    /// Scalar columns, row-aligned, for predicates and for returning
    /// field values on a hit.
    scalars: HashMap<String, Vec<FieldValue>>,
}

impl ExternalDataset {
    pub fn dim(&self) -> u32 {
        self.dim
    }

    pub fn row_count(&self) -> usize {
        self.ids.len()
    }

    pub fn ids(&self) -> &[String] {
        &self.ids
    }
}

/// Resolve `source.uri` to the object keys to read, in sorted order so a
/// dataset is assembled deterministically.
///
/// A URI ending in `/` is a prefix and selects every `*.parquet` beneath
/// it. Anything else is a single object key.
fn resolve_keys(store: &dyn BlobStore, uri: &str) -> Result<Vec<String>> {
    let cleaned = uri.strip_prefix("file://").unwrap_or(uri);
    if cleaned.ends_with('/') {
        let mut keys: Vec<String> = store
            .list(cleaned)?
            .into_iter()
            .filter(|k| k.ends_with(".parquet"))
            .collect();
        keys.sort();
        if keys.is_empty() {
            return Err(VedaError::Storage(format!(
                "no .parquet objects found under '{uri}'"
            )));
        }
        Ok(keys)
    } else {
        Ok(vec![cleaned.to_owned()])
    }
}

/// Read and materialize the dataset described by `source`.
pub fn load(store: &dyn BlobStore, source: &ExternalSource) -> Result<ExternalDataset> {
    if source.format != "parquet" {
        return Err(VedaError::Storage(format!(
            "unsupported external format '{}': only 'parquet' is supported",
            source.format
        )));
    }
    let keys = resolve_keys(store, &source.uri)?;

    let mut dim: Option<u32> = None;
    let mut vectors: Vec<f32> = Vec::new();
    let mut ids: Vec<String> = Vec::new();
    let mut scalars: HashMap<String, Vec<FieldValue>> = HashMap::new();

    for key in &keys {
        let raw = store.get(key)?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(Bytes::from(raw))
            .map_err(|e| VedaError::Storage(format!("parquet open '{key}': {e}")))?;
        let schema = builder.schema().clone();

        let available: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
        if !available.contains(&source.id_field.as_str()) {
            return Err(VedaError::Storage(format!(
                "id_field '{}' not found in '{key}'; available columns: {}",
                source.id_field,
                available.join(", ")
            )));
        }
        if !available.contains(&source.vector_field.as_str()) {
            return Err(VedaError::Storage(format!(
                "vector_field '{}' not found in '{key}'; available columns: {}",
                source.vector_field,
                available.join(", ")
            )));
        }

        let file_dim = fixed_size_list_dim(&schema, &source.vector_field, key)?;
        match dim {
            None => dim = Some(file_dim),
            Some(d) if d != file_dim => {
                return Err(VedaError::Storage(format!(
                    "vector dimension disagreement: earlier files have {d}, '{key}' has {file_dim}"
                )));
            }
            Some(_) => {}
        }

        read_batches(builder, source, &mut ids, &mut vectors, &mut scalars)?;
    }

    let dim = dim.ok_or_else(|| VedaError::Storage(format!("no data behind '{}'", source.uri)))?;
    Ok(ExternalDataset {
        dim,
        vectors,
        ids,
        scalars,
    })
}
```

Then implement the two helpers:

- `fixed_size_list_dim(schema, field_name, key) -> Result<u32>` — looks up the field,
  matches `DataType::FixedSizeList(inner, len)` with `inner.data_type() == &DataType::Float32`,
  and returns `len as u32`. Any other Arrow type is an error naming the column and the
  actual type it found.
- `read_batches(builder, source, ids, vectors, scalars)` — iterates
  `builder.build()?`, and per `RecordBatch` appends: the id column stringified, the
  flattened vector values, and every remaining column converted to `FieldValue`.

For the id column accept `Utf8`, `LargeUtf8`, `Int32` and `Int64`, stringifying each —
external ids are commonly integers, and `SearchHit.id` is a `String`, so no synthetic
identifier is needed. For scalar columns map `Utf8`/`LargeUtf8` → `FieldValue::String`,
`Int32` → `Int32`, `Int64` → `Int64`, `Float32` → `Float32`, `Float64` → `Float64`,
`Boolean` → `Bool`; skip any column whose type is not in that set rather than failing,
so an unusual column elsewhere in a customer's file does not make the dataset
unloadable. **Report in your task report which types you chose to skip.**

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PROTOC=/home/mac/bin/protoc /home/mac/.cargo/bin/cargo test -p veda-storage --lib external_parquet`
Expected: PASS, all seven tests.

- [ ] **Step 5: Gates and commit**

```bash
PROTOC=/home/mac/bin/protoc /home/mac/.cargo/bin/cargo fmt --all --check
PROTOC=/home/mac/bin/protoc /home/mac/.cargo/bin/cargo clippy -p veda-storage --all-targets -- -D warnings
git add crates/veda-storage/src/external_parquet.rs crates/veda-storage/src/lib.rs
git commit -m "feat(storage): read and validate external Parquet datasets"
```

---

### Task 4: Exact search with filtering

**Files:**
- Modify: `crates/veda-storage/src/external_parquet.rs`

**Interfaces:**
- Consumes: `ExternalDataset` (Task 3), `veda_core::simd`, `veda_core::record::Record`.
- Produces:
  - `ExternalDataset::search(&self, query: &[f32], k: usize, metric: DistanceMetric, allow: Option<&dyn Fn(&Record) -> bool>) -> Result<Vec<(String, f32)>>`
  - `ExternalDataset::record_at(&self, row: usize) -> Record`
  - `ExternalDataset::vector_at(&self, row: usize) -> &[f32]`

  Task 7 consumes all three.

**Why the predicate is a closure and not a `FilterExpr`.** `veda-query` already depends
on `veda-storage` (`crates/veda-query/Cargo.toml:9`), so a storage→query dependency
would be a **dependency cycle**, which Cargo rejects outright — not merely a layering
inversion. `ExternalDataset` therefore cannot name `FilterExpr` at all.

The predicate is passed in as `Option<&dyn Fn(&Record) -> bool>`. The gateway depends on
both crates, so **Task 7 supplies `evaluate_record`** as that closure. Filter semantics
are still shared with the internal path exactly — the same function evaluates them —
without inverting the crate graph. `Record` lives in `veda-core`, which both crates
already depend on.

Do not attempt to add `veda-query` to `veda-storage`; it will not compile.

- [ ] **Step 1: Write the failing tests**

Add to the test module in `crates/veda-storage/src/external_parquet.rs`:

```rust
#[test]
fn search_is_exact_and_matches_direct_scoring() {
    let dir = tempfile::tempdir().unwrap();
    write_parquet(&dir.path().join("a.parquet"), 200, 8);
    let store = LocalBlobStore::new(dir.path());
    let ds = load(&store, &source("a.parquet")).unwrap();

    let query: Vec<f32> = (0..8).map(|d| d as f32 / 3.0).collect();
    let got = ds.search(&query, 10, DistanceMetric::Cosine, None).unwrap();

    // Oracle: score every row directly with the same kernel the
    // implementation uses, sort, truncate. If these disagree, the
    // external path is not exact.
    let mut want: Vec<(String, f32)> = (0..ds.row_count())
        .map(|r| {
            let v = ds.vector_at(r);
            (ds.ids()[r].clone(), veda_core::simd::cosine_distance(&query, v))
        })
        .collect();
    want.sort_by(|a, b| {
        a.1.partial_cmp(&b.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });
    want.truncate(10);
    assert_eq!(got, want);
}

#[test]
fn search_honours_a_scalar_filter() {
    let dir = tempfile::tempdir().unwrap();
    write_parquet(&dir.path().join("a.parquet"), 50, 4);
    let store = LocalBlobStore::new(dir.path());
    let ds = load(&store, &source("a.parquet")).unwrap();

    let query = vec![0.0f32, 0.0, 0.0, 0.0];
    let unfiltered = ds.search(&query, 5, DistanceMetric::Cosine, None).unwrap();
    assert!(unfiltered.iter().any(|(id, _)| id == "doc0"));

    // Exclude the row the unfiltered search ranks first, so the filter
    // cannot be silently ignored. In storage's own tests the predicate is
    // a plain closure; the gateway supplies `evaluate_record` in Task 7.
    let deny_doc0 = |rec: &Record| {
        rec.get("doc_id") != Some(&FieldValue::String("doc0".into()))
    };
    let filtered = ds
        .search(&query, 5, DistanceMetric::Cosine, Some(&deny_doc0))
        .unwrap();
    assert!(
        !filtered.iter().any(|(id, _)| id == "doc0"),
        "filter was ignored: doc0 is still present"
    );
    assert_eq!(filtered.len(), 5);
}

#[test]
fn search_with_k_larger_than_the_dataset_returns_everything() {
    let dir = tempfile::tempdir().unwrap();
    write_parquet(&dir.path().join("a.parquet"), 7, 4);
    let store = LocalBlobStore::new(dir.path());
    let ds = load(&store, &source("a.parquet")).unwrap();
    let hits = ds
        .search(&[1.0, 0.0, 0.0, 0.0], 100, DistanceMetric::Cosine, None)
        .unwrap();
    assert_eq!(hits.len(), 7);
}

#[test]
fn search_rejects_a_query_of_the_wrong_dimension() {
    let dir = tempfile::tempdir().unwrap();
    write_parquet(&dir.path().join("a.parquet"), 4, 8);
    let store = LocalBlobStore::new(dir.path());
    let ds = load(&store, &source("a.parquet")).unwrap();
    let err = ds
        .search(&[1.0, 0.0], 3, DistanceMetric::Cosine, None)
        .unwrap_err()
        .to_string();
    assert!(err.contains('8') && err.contains('2'), "got: {err}");
}

#[test]
fn record_at_exposes_scalar_columns_for_filtering() {
    let dir = tempfile::tempdir().unwrap();
    write_parquet(&dir.path().join("a.parquet"), 3, 4);
    let store = LocalBlobStore::new(dir.path());
    let ds = load(&store, &source("a.parquet")).unwrap();
    let rec = ds.record_at(1);
    assert_eq!(
        rec.get("doc_id"),
        Some(&FieldValue::String("doc1".into()))
    );
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PROTOC=/home/mac/bin/protoc /home/mac/.cargo/bin/cargo test -p veda-storage --lib external_parquet`
Expected: compile errors — `search`, `vector_at` and `record_at` do not exist.

- [ ] **Step 3: Implement search**

```rust
impl ExternalDataset {
    /// The vector for row `row`.
    pub fn vector_at(&self, row: usize) -> &[f32] {
        let d = self.dim as usize;
        &self.vectors[row * d..(row + 1) * d]
    }

    /// A `Record` view of row `row`, so predicates reuse the same
    /// `evaluate_record` the internal path uses rather than a second
    /// implementation that could drift from it.
    pub fn record_at(&self, row: usize) -> Record {
        let mut rec = Record::new();
        for (name, column) in &self.scalars {
            if let Some(v) = column.get(row) {
                rec.set(name, v.clone());
            }
        }
        rec
    }

    /// Exact top-k. Returns `(id, distance)` sorted ascending by distance,
    /// ties broken by id, truncated to `k`.
    ///
    /// Linear in `row_count()` by design — no index exists over external
    /// data, so recall is 1.0 and latency scales with dataset size.
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        metric: DistanceMetric,
        allow: Option<&dyn Fn(&Record) -> bool>,
    ) -> Result<Vec<(String, f32)>> {
        if query.len() != self.dim as usize {
            return Err(VedaError::Storage(format!(
                "query dimension {} does not match dataset dimension {}",
                query.len(),
                self.dim
            )));
        }
        if k == 0 {
            return Ok(Vec::new());
        }
        let mut scored: Vec<(String, f32)> = Vec::with_capacity(self.row_count().min(k * 4));
        for row in 0..self.row_count() {
            if let Some(pred) = allow
                && !pred(&self.record_at(row))
            {
                continue;
            }
            let d = match metric {
                DistanceMetric::Cosine => cosine_distance(query, self.vector_at(row)),
                DistanceMetric::DotProduct => -dot_product(query, self.vector_at(row)),
                DistanceMetric::Euclidean => euclidean_distance(query, self.vector_at(row)),
                other => {
                    return Err(VedaError::Storage(format!(
                        "metric {other:?} is not supported for external collections"
                    )));
                }
            };
            scored.push((self.ids[row].clone(), d));
        }
        scored.sort_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        scored.truncate(k);
        Ok(scored)
    }
}
```

Use whatever euclidean kernel `veda_core::simd` actually exposes; if it exposes none,
report that and use the squared-distance form consistently, noting it in your report.
Ties break by id so results are deterministic, matching the ordering discipline used
elsewhere in this codebase.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PROTOC=/home/mac/bin/protoc /home/mac/.cargo/bin/cargo test -p veda-storage --lib external_parquet`
Expected: PASS.

- [ ] **Step 5: Gates and commit**

```bash
PROTOC=/home/mac/bin/protoc /home/mac/.cargo/bin/cargo fmt --all --check
PROTOC=/home/mac/bin/protoc /home/mac/.cargo/bin/cargo clippy -p veda-storage --all-targets -- -D warnings
git add crates/veda-storage/src/external_parquet.rs
git commit -m "feat(storage): exact filtered search over external Parquet"
```

---

### Task 5: Freshness stamping and the dataset cache

**Files:**
- Modify: `crates/veda-storage/src/external_parquet.rs`
- Modify: `crates/veda-gateway/src/rest/state.rs` (alongside `segment_managers`, line 203)

**Interfaces:**
- Consumes: `BlobStat` / `BlobStore::stat` (Task 2), `load` (Task 3).
- Produces:
  - `pub struct SourceStamp` with `PartialEq`
  - `pub fn stamp(store: &dyn BlobStore, source: &ExternalSource) -> Result<SourceStamp>`
  - `ExternalDataset::source_stamp(&self) -> &SourceStamp`
  - `AppState.external_datasets: RwLock<HashMap<String, Arc<ExternalDataset>>>`

  Task 7 consumes all of these.

- [ ] **Step 1: Write the failing tests**

```rust
#[test]
fn a_rewritten_file_produces_a_different_stamp() {
    let dir = tempfile::tempdir().unwrap();
    write_parquet(&dir.path().join("a.parquet"), 10, 4);
    let store = LocalBlobStore::new(dir.path());
    let before = stamp(&store, &source("a.parquet")).unwrap();

    // Different row count -> different bytes -> different stamp.
    write_parquet(&dir.path().join("a.parquet"), 20, 4);
    let after = stamp(&store, &source("a.parquet")).unwrap();
    assert_ne!(before, after);
}

#[test]
fn an_unchanged_file_produces_an_equal_stamp() {
    let dir = tempfile::tempdir().unwrap();
    write_parquet(&dir.path().join("a.parquet"), 10, 4);
    let store = LocalBlobStore::new(dir.path());
    let a = stamp(&store, &source("a.parquet")).unwrap();
    let b = stamp(&store, &source("a.parquet")).unwrap();
    assert_eq!(a, b, "stamping must be stable when nothing changed");
}

#[test]
fn a_new_file_behind_a_prefix_changes_the_stamp() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(dir.path().join("part")).unwrap();
    write_parquet(&dir.path().join("part/a.parquet"), 5, 4);
    let store = LocalBlobStore::new(dir.path());
    let before = stamp(&store, &source("part/")).unwrap();
    write_parquet(&dir.path().join("part/b.parquet"), 5, 4);
    let after = stamp(&store, &source("part/")).unwrap();
    assert_ne!(before, after, "an added file must invalidate the cache");
}

#[test]
fn a_loaded_dataset_carries_the_stamp_of_its_source() {
    let dir = tempfile::tempdir().unwrap();
    write_parquet(&dir.path().join("a.parquet"), 6, 4);
    let store = LocalBlobStore::new(dir.path());
    let ds = load(&store, &source("a.parquet")).unwrap();
    assert_eq!(*ds.source_stamp(), stamp(&store, &source("a.parquet")).unwrap());
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PROTOC=/home/mac/bin/protoc /home/mac/.cargo/bin/cargo test -p veda-storage --lib external_parquet`
Expected: compile errors — `stamp` and `source_stamp` do not exist.

- [ ] **Step 3: Implement stamping**

```rust
/// Identity of a source at a point in time. Compared, never interpreted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceStamp {
    /// One entry per object behind the URI, sorted by key so the
    /// comparison does not depend on listing order.
    entries: Vec<(String, BlobStat)>,
}

/// Cheap identity of `source` without transferring any data.
pub fn stamp(store: &dyn BlobStore, source: &ExternalSource) -> Result<SourceStamp> {
    let keys = resolve_keys(store, &source.uri)?;
    let mut entries = Vec::with_capacity(keys.len());
    for key in keys {
        let st = store.stat(&key)?;
        entries.push((key, st));
    }
    entries.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(SourceStamp { entries })
}
```

Add a `stamp: SourceStamp` field to `ExternalDataset`, populate it inside `load` (call
`stamp(store, source)?` before reading the data, so the stamp describes the bytes that
were read), and expose `source_stamp(&self) -> &SourceStamp`.

- [ ] **Step 4: Add the cache to `AppState`**

In `crates/veda-gateway/src/rest/state.rs`, alongside `segment_managers`:

```rust
    /// Materialized external datasets, keyed by resolved collection name.
    /// Dropped and reloaded whenever the source's stamp changes, so a
    /// changed lake file is never served as though it were current.
    pub external_datasets: RwLock<HashMap<String, Arc<ExternalDataset>>>,
```

Initialise it to an empty map at **every** `AppState` construction site — there are at
least three (around lines 392, 468 and 525). Grep for `segment_managers: RwLock::new`
and add the field next to each occurrence. Missing one causes a compile error, so this
is self-checking, but check anyway and say how many you found.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `PROTOC=/home/mac/bin/protoc /home/mac/.cargo/bin/cargo test -p veda-storage --lib external_parquet`
Expected: PASS.

Run: `PROTOC=/home/mac/bin/protoc /home/mac/.cargo/bin/cargo test -p veda-gateway --lib`
Expected: PASS.

- [ ] **Step 6: Gates and commit**

```bash
PROTOC=/home/mac/bin/protoc /home/mac/.cargo/bin/cargo fmt --all --check
PROTOC=/home/mac/bin/protoc /home/mac/.cargo/bin/cargo clippy -p veda-storage -p veda-gateway --all-targets -- -D warnings
git add crates/veda-storage/src/external_parquet.rs crates/veda-gateway/src/rest/state.rs
git commit -m "feat(storage): stamp external sources and cache materialized datasets"
```

---

### Task 6: Registration endpoint

**Files:**
- Modify: `crates/veda-gateway/src/rest/collections.rs` (`create_collection`, line 263)
- Test: `crates/veda-gateway/tests/external_parquet_test.rs` (create)

**Interfaces:**
- Consumes: `ExternalSource` (Task 1), `load` (Task 3).
- Produces: a collection whose schema carries `external`, validated at registration.

**Convention:** this crate's integration tests are **self-contained per file** — there is
no `tests/common/` and you must not create one. Copy the harness from
`crates/veda-gateway/tests/bm25_filter_test.rs`, which is the closest existing example.
Read it first and confirm the actual request/response shapes rather than guessing.

- [ ] **Step 1: Write the failing tests**

Create `crates/veda-gateway/tests/external_parquet_test.rs`. Write a helper that creates
a Parquet file in a `tempfile::TempDir`, then:

1. `registering_a_valid_parquet_dataset_succeeds` — `POST /v1/collections` with an
   `external` block pointing at the file; assert `201`, then `GET /v1/collections/{name}`
   reports it as external with the inferred dimension.
2. `registering_with_a_missing_vector_column_is_rejected` — assert `400` and that the
   body names both the missing column and the available ones.
3. `registering_with_a_nonexistent_uri_is_rejected` — assert `400`.
4. `registering_a_wrong_typed_vector_column_is_rejected` — point `vector_field` at the
   string id column; assert `400`.

Every non-2xx assertion must print the response body in its failure message, e.g.
`assert_eq!(r.status(), 400, "body: {}", r.text().await.unwrap());` — otherwise a
malformed request is indistinguishable from a working guard.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PROTOC=/home/mac/bin/protoc timeout 2400 /home/mac/.cargo/bin/cargo test -p veda-gateway --test external_parquet_test`
Expected: failures — the `external` block is ignored today, so registration succeeds where
it should 400. Note that `serde` silently drops unknown keys unless
`deny_unknown_fields` is set, so the RED state here is a runtime status mismatch, not a
compile error.

- [ ] **Step 3: Wire registration**

In `create_collection`, after the schema is assembled and before it is persisted: if the
request carries an `external` block, call `veda_storage::external_parquet::load` against
the state's blob store. Map any error to `400` with the message from `VedaError`
verbatim, so the specific cause reaches the caller. On success, record the inferred
dimension and row count in the response.

Do **not** cache the dataset here — Task 7 owns the cache and its freshness check. Loading
at registration is a validation step whose result is discarded; say so in a comment, so a
later reader does not "optimise" it into a cache write that skips stamping.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PROTOC=/home/mac/bin/protoc timeout 2400 /home/mac/.cargo/bin/cargo test -p veda-gateway --test external_parquet_test`
Expected: PASS.

- [ ] **Step 5: Gates and commit**

```bash
PROTOC=/home/mac/bin/protoc /home/mac/.cargo/bin/cargo fmt --all --check
PROTOC=/home/mac/bin/protoc /home/mac/.cargo/bin/cargo clippy -p veda-gateway --all-targets -- -D warnings
git add crates/veda-gateway/src/rest/collections.rs crates/veda-gateway/tests/external_parquet_test.rs
git commit -m "feat(collections): register external Parquet datasets"
```

---

### Task 7: Search dispatch for external collections

**Files:**
- Modify: `crates/veda-gateway/src/rest/search.rs` (the dispatch chain around line 660)
- Test: `crates/veda-gateway/tests/external_parquet_test.rs` (extend)

**Interfaces:**
- Consumes: `ExternalDataset::search` (Task 4), `stamp` / `source_stamp` (Task 5),
  `AppState.external_datasets` (Task 5).

**The ordering that matters:** the external branch must run **after** `tenant::scope` and
`state.inner.aliases.resolve`, exactly like every other collection-addressed handler.
Resolving in the wrong order has caused three separate silent-data-loss incidents in this
project. The cache is keyed by the **resolved** name.

- [ ] **Step 1: Write the failing tests**

Extend `crates/veda-gateway/tests/external_parquet_test.rs`:

1. `external_dense_search_returns_expected_ids` — register, search, assert the nearest
   ids come back and the hit list is non-empty.
2. `external_search_honours_a_filter` — a filter that excludes the top hit; assert it is
   absent and the response is still `200`.
3. `a_rewritten_source_is_visible_to_the_next_query` — register, search, **rewrite the
   Parquet file with different vectors**, search again, assert the new data is returned.
   This is the staleness gate.
4. `external_bm25_is_rejected` / `external_sparse_is_rejected` /
   `external_hybrid_is_rejected` — each asserts `400` and that the body explains the
   missing index rather than returning an empty result set.
5. `external_search_through_an_alias_resolves` — create an alias to the external
   collection and search through it; assert the same hits.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PROTOC=/home/mac/bin/protoc timeout 2400 /home/mac/.cargo/bin/cargo test -p veda-gateway --test external_parquet_test`
Expected: failures — external search is not implemented, so these 404 or return empty.

- [ ] **Step 3: Implement the branch**

In the search handler, after scope and alias resolution and before the ordinary
`segment_managers` lookup:

```rust
    // External collections are read in place: no segments, no index.
    // Checked after tenant scoping and alias resolution so the cache is
    // keyed by the same resolved name every other handler uses.
    if let Some(src) = schema.external() {
        return execute_external(state, &name, src, req).await;
    }
```

`execute_external` must:

1. `stamp(store, src)?` and compare against the cached dataset's `source_stamp()`;
   reload on mismatch or on a cache miss, and insert under the resolved name.
2. Reject `bm25`, `match_phrase`, `sparse` and `hybrid` with `400` and a message naming
   the missing index — for example: *"bm25 search is not available on external collection
   'lake_docs': it requires the inverted index Veda builds at ingest, and external data
   is read in place."*
3. Serve a dense query via `ExternalDataset::search`, honouring `req.limit` and the
   effective filter, and map each `(id, distance)` into the ordinary `SearchHit` shape,
   populating `fields` from `record_at`.

   The filter is supplied as the closure `ExternalDataset::search` expects. This is where
   the internal and external paths are made to share filter semantics exactly — the
   gateway can see both crates, so it hands storage the very same evaluator the segment
   path uses:

   ```rust
   let eff = effective_filter(req);
   let pred = eff
       .as_ref()
       .map(|expr| move |rec: &Record| evaluate_record(expr, rec));
   let hits = dataset.search(
       &query_vector,
       req.limit,
       src.metric,
       pred.as_ref().map(|p| p as &dyn Fn(&Record) -> bool),
   )?;
   ```
4. Return `503` naming the URI if the source has disappeared, never an empty result —
   an empty result set is indistinguishable from a correct answer.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PROTOC=/home/mac/bin/protoc timeout 2400 /home/mac/.cargo/bin/cargo test -p veda-gateway --test external_parquet_test`
Expected: PASS.

- [ ] **Step 5: Prove the staleness test has teeth**

Temporarily make the stamp comparison always report "unchanged" (e.g. return early from
the comparison), re-run `a_rewritten_source_is_visible_to_the_next_query`, and confirm it
**FAILS**. Then restore. A freshness test that passes with the freshness check disabled
is worthless, and this project has shipped two such tests already.

Record both the failure output and the restoration in your report.

- [ ] **Step 6: Gates and commit**

```bash
PROTOC=/home/mac/bin/protoc /home/mac/.cargo/bin/cargo fmt --all --check
PROTOC=/home/mac/bin/protoc /home/mac/.cargo/bin/cargo clippy -p veda-gateway --all-targets -- -D warnings
git add crates/veda-gateway/src/rest/search.rs crates/veda-gateway/tests/external_parquet_test.rs
git commit -m "feat(search): serve external Parquet collections in place"
```

---

### Task 8: Read-only enforcement

**Files:**
- Modify: the mutating handlers in `crates/veda-gateway/src/rest/` — `records.rs`
  (insert/update/delete), `embed_insert.rs`, `ingest_common.rs`, `collections.rs`
  (reindex, seal), `document.rs` (process-document)
- Test: `crates/veda-gateway/tests/external_parquet_test.rs` (extend)

**Interfaces:**
- Consumes: `CollectionSchema::is_external()` (Task 1).

- [ ] **Step 1: Find every mutating entry point**

Grep for the handlers registered on this collection's routes and list them. **Report the
complete list you found**; the plan names the files it expects, and a handler present in
the code but missing from that list is a finding that matters more than the code.

- [ ] **Step 2: Write the failing tests**

Add one test per mutating endpoint, each registering an external collection and asserting
`405` with a body explaining the collection is external and read-only. Cover at minimum:
`POST .../insert`, `POST .../embed-insert`, `POST .../delete`, `PATCH`/`PUT` update,
`POST .../reindex`, `POST .../seal`, `POST .../process-document`.

- [ ] **Step 3: Run the tests to verify they fail**

Run: `PROTOC=/home/mac/bin/protoc timeout 2400 /home/mac/.cargo/bin/cargo test -p veda-gateway --test external_parquet_test`
Expected: failures — these currently attempt the write.

- [ ] **Step 4: Add the guard**

Add one shared helper next to the other validation helpers in `search.rs` or a shared
module, and call it from each handler after scope and alias resolution:

```rust
/// Reject a mutating request against a collection Veda does not own.
///
/// Enforced per handler rather than by convention: an unenforced
/// read-only contract is the same class of mistake as an unenforced
/// sortedness contract, and this codebase has been bitten by guards
/// applied at only some of several entry points.
fn reject_if_external(
    schema: &CollectionSchema,
    collection: &str,
) -> Result<(), (StatusCode, Json<ErrorResponse>)> {
    if schema.is_external() {
        return Err((
            StatusCode::METHOD_NOT_ALLOWED,
            Json(ErrorResponse {
                error: format!(
                    "collection '{collection}' is external and read-only: \
                     it reads data Veda does not own, in place"
                ),
            }),
        ));
    }
    Ok(())
}
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `PROTOC=/home/mac/bin/protoc timeout 2400 /home/mac/.cargo/bin/cargo test -p veda-gateway --test external_parquet_test`
Expected: PASS.

- [ ] **Step 6: Gates and commit**

```bash
PROTOC=/home/mac/bin/protoc /home/mac/.cargo/bin/cargo fmt --all --check
PROTOC=/home/mac/bin/protoc /home/mac/.cargo/bin/cargo clippy -p veda-gateway --all-targets -- -D warnings
git add crates/veda-gateway/src/rest/ crates/veda-gateway/tests/external_parquet_test.rs
git commit -m "feat(collections): reject writes to external read-only collections"
```

---

### Task 9: Documentation, limits, and full verification

**Files:**
- Modify: `CHANGELOG.md`
- Modify: `documentation/2026-08-14-external-parquet-collections-design.md` (append)
- Modify: the REST API documentation under `documentation/` (find the file that documents
  `POST /v1/collections`; if none exists, say so in your report rather than creating a new
  convention)

- [ ] **Step 1: Measure what registration costs**

Add an `#[ignore]` benchmark to `crates/veda-storage/tests/` that writes Parquet files of
10,000 and 100,000 rows at 768 dimensions, then reports load time and resident vector
bytes for each. Run it and record the numbers.

Run: `PROTOC=/home/mac/bin/protoc timeout 1800 /home/mac/.cargo/bin/cargo test -p veda-storage --test external_parquet_bench -- --ignored --nocapture`

- [ ] **Step 2: Append a "Measured result" section to the design doc**

Record the load times and memory figures verbatim, and state plainly whether the ~3 GB
per million 768-dim vectors estimate in the spec held. **If it is materially wrong, say
so and correct the spec's stated limit** — the limit exists to be honest about the
feature, not to flatter it.

- [ ] **Step 3: Document the limits where users will see them**

The spec's "Stated limits" section must reach the API documentation, not only the design
doc: exact search is linear in row count; the vector column is resident; object reads are
whole-object so Parquet's range-request advantage is lost; bm25/sparse/hybrid are
unavailable; the collection is read-only.

- [ ] **Step 4: Update the changelog**

Read `CHANGELOG.md` and follow its existing structure. Cover: external Parquet
collections registered via an `external` block; exact filtered vector search in place;
auto-invalidation on source change; read-only enforcement; and the stated limits.

- [ ] **Step 5: Full verification**

Run each in the foreground and report actual pass/fail counts, not "all green":

- `PROTOC=/home/mac/bin/protoc timeout 1800 /home/mac/.cargo/bin/cargo test -p veda-storage`
- `PROTOC=/home/mac/bin/protoc timeout 3600 /home/mac/.cargo/bin/cargo test -p veda-gateway`
- `PROTOC=/home/mac/bin/protoc timeout 900 /home/mac/.cargo/bin/cargo test -p veda-core`
- `PROTOC=/home/mac/bin/protoc /home/mac/.cargo/bin/cargo fmt --all --check`
- `PROTOC=/home/mac/bin/protoc /home/mac/.cargo/bin/cargo clippy -p veda-core -p veda-storage -p veda-gateway --all-targets -- -D warnings`

- [ ] **Step 6: Commit**

```bash
git add CHANGELOG.md documentation/ crates/veda-storage/tests/
git commit -m "docs: external Parquet collections, measured limits"
```

---

## Plan self-review

**Spec coverage:**

| Spec requirement | Task |
|---|---|
| `external` block on `POST /v1/collections`; `ExternalSource` type | 1, 6 |
| `arrow`/`parquet` confined to `veda-storage` | 1 (asserted by grep) |
| Registration validation: missing column, wrong Arrow type, dimension disagreement, unreadable URI | 3, 6 |
| Inferred dimension and scalar columns | 3, 6 |
| `external_parquet.rs` as sole Arrow owner | 3 |
| Reuse of `veda_core::simd` kernels | 4 |
| Reuse of `evaluate_record`, supplied by the gateway as a closure over `record_at` | 4, 7 |
| Prefix datasets read as one collection | 3 |
| `BlobStore::stat` + `BlobStat` | 2 |
| `SourceStamp`, per-query comparison, auto-invalidation | 5, 7 |
| Cache in `AppState` keyed by resolved name | 5, 7 |
| Dispatch after `tenant::scope` + alias resolve | 7 |
| bm25 / match_phrase / sparse / hybrid rejected with `400` | 7 |
| Writes rejected with `405` on every mutating endpoint | 8 |
| `503` when the source disappears | 7 |
| Query/dataset dimension mismatch is an error | 4, 6 |
| Exactness verified differentially | 4 |
| Staleness test verified to fail with the check disabled | 7 |
| Stated limits reach the API docs | 9 |
| Memory estimate validated against measurement | 9 |

**Type consistency:** `ExternalSource` fields are identical in Tasks 1, 3, 4 and 6.
`load(&dyn BlobStore, &ExternalSource) -> Result<ExternalDataset>` is consistent in Tasks
3, 5, 6. `search(&[f32], usize, DistanceMetric, Option<&dyn Fn(&Record) -> bool>) -> Result<Vec<(String, f32)>>`
is consistent in Tasks 4 and 7. `stamp(&dyn BlobStore, &ExternalSource) -> Result<SourceStamp>`
and `source_stamp(&self) -> &SourceStamp` are consistent in Tasks 5 and 7.
`BlobStat { size, version }` is consistent in Tasks 2 and 5.

**Crate-graph constraint, resolved before execution.** `veda-query` depends on
`veda-storage`, so `veda-storage` cannot reference `FilterExpr` — that would be a
dependency cycle. Predicate evaluation therefore crosses the boundary as a closure over
`&Record`, with the gateway supplying `evaluate_record`. Task 4 states this explicitly so
no implementer tries to add the dependency and burns a build cycle discovering it.

**One deliberate read-then-decide point, not a placeholder.** Task 8 asks for the
complete list of mutating endpoints, because a handler that exists in the code but is
absent from the plan's list is a finding that matters more than the code — this codebase
has twice shipped a guard applied at only some of several entry points.

# System Architecture

## 1. Overview

Veda is a multimodal vector database designed for the RAG (Retrieval-Augmented Generation) era. It stores, indexes, and searches embeddings across any modality -- text, images, audio, video, and documents -- through a unified REST API.

The system comprises three major components:

- **Core Engine** -- 7 Rust crates organized in a layered dependency graph, compiled into a single `veda-server` binary
- **Niriksha Dashboard** -- A React/TypeScript visualization and exploration frontend
- **Python SDK** -- A pure-Python client wrapping the REST API

The default listen port is **6340**.

```
                ┌──────────────────────────┐
                │      Client Layer        │
                │  Python SDK / curl / UI  │
                └──────────┬───────────────┘
                           │ HTTP :6340
                ┌──────────▼───────────────┐
                │     veda-gateway         │
                │   REST API + Auth + RBAC │
                └──────────┬───────────────┘
          ┌────────────────┼────────────────┐
          │                │                │
   ┌──────▼──────┐ ┌──────▼──────┐ ┌───────▼───────┐
   │ veda-query   │ │veda-storage │ │veda-coordinator│
   │ Filter+Search│ │ Segments+WAL│ │ Cluster State  │
   └──────┬──────┘ └──────┬──────┘ └───────┬───────┘
          │                │                │
   ┌──────▼──────┐        │                │
   │ veda-index   │        │                │
   │ ANN Indexes  │        │                │
   └──────┬──────┘        │                │
          └────────┬───────┘                │
            ┌──────▼──────┐                 │
            │  veda-core   │─────────────────┘
            │ Types+SIMD   │
            └──────────────┘
```

---

## 2. Crate Dependency Graph

The workspace is defined in the root `Cargo.toml` and contains 7 crates. Dependencies flow strictly upward; no cycles exist.

```
veda-core          (no internal deps)
    │
    ├── veda-storage        (depends on: veda-core)
    │       │
    │       ├── veda-index  (depends on: veda-core, veda-storage)
    │       │
    │       └── veda-query  (depends on: veda-core, veda-storage, veda-index)
    │
    ├── veda-coordinator    (depends on: veda-core)
    │
    └── veda-gateway        (depends on: veda-core, veda-query, veda-storage,
    │                                     veda-index, veda-coordinator)
    │
    └── veda-server         (depends on: all six crates above)
```

| Crate | Key third-party deps |
|---|---|
| `veda-core` | `serde`, `serde_json`, `uuid`, `thiserror` |
| `veda-storage` | `memmap2`, `roaring`, `base64` |
| `veda-index` | `rand`, `roaring` |
| `veda-query` | `roaring`, `uuid` |
| `veda-coordinator` | `tokio`, `tracing`, `uuid` |
| `veda-gateway` | `axum`, `jsonwebtoken`, `tower`, `tower-http` |
| `veda-server` | `tracing-subscriber` |

All crates share workspace-level dependency versions via `[workspace.dependencies]` in the root `Cargo.toml`. The workspace edition is Rust 2024 under Apache-2.0 license.

---

## 3. veda-core -- Shared Types & Utilities

**Path:** `crates/veda-core/src/`

This crate has zero internal dependencies and provides the foundational types, distance computation, metrics collection, and embedding infrastructure used by every other crate.

### 3.1 Type Aliases and Enums

**File:** `types.rs`

```rust
pub type RecordId = Uuid;
pub type CollectionId = Uuid;
pub type SegmentId = Uuid;
pub type ShardId = u32;
```

**`DistanceMetric`** -- 6 variants:
- `Cosine`, `Euclidean`, `DotProduct`, `Hamming`, `Manhattan`, `Jaccard`

**`FieldType`** -- 13 variants:
- `Vector { dimension: u32, metric: DistanceMetric }`, `SparseVector`, `String`, `Int32`, `Int64`, `Float32`, `Float64`, `Bool`, `Timestamp`, `Array(Box<FieldType>)`, `Json`, `Blob`, `Uuid`

**`IndexType`** -- 10 variants:
- `Hnsw`, `DiskAnn`, `IvfPq`, `Flat`, `SuffixArray`, `Inverted`, `Range`, `Bitmap`, `Hash`, `None`

### 3.2 DenseVector

**File:** `vector.rs`

A wrapper around `Vec<f32>` providing dimension-aware distance computation.

**Methods:**
- `new(data: Vec<f32>)` -- constructor
- `dimension()` -- returns `data.len()`
- `as_slice()` -- returns `&[f32]`
- `as_bytes()` -- reinterprets the float array as raw `&[u8]` (unsafe, zero-copy)
- `normalize()` -- L2-normalizes in place
- `normalized()` -- returns a new L2-normalized copy
- `is_normalized()` -- checks if norm is within 1e-5 of 1.0
- `cosine_distance(other)` -- delegates to `simd::cosine_distance`
- `euclidean_distance(other)` -- delegates to `simd::euclidean_distance`
- `dot_product(other)` -- delegates to `simd::dot_product`
- `manhattan_distance(other)` -- delegates to `simd::manhattan_distance`

All distance methods assert matching dimensions before computing.

### 3.3 CollectionSchema

**File:** `schema.rs`

Defines the shape of a collection: its fields, shard count, and replication factor.

**`FieldDefinition`** fields: `name: String`, `field_type: FieldType`, `filterable: bool`, `index_type: IndexType`. Constructors:
- `new(name, field_type)` -- defaults to not filterable, `IndexType::None`
- `vector(name, dimension, metric)` -- defaults to `IndexType::Hnsw`
- `blob(name)` -- blob field with no index

Builder-pattern methods: `with_filterable(bool)`, `with_index_type(IndexType)`.

**`CollectionSchema`** fields: `id: CollectionId`, `name: String`, `fields: Vec<FieldDefinition>`, `shard_count: u32`, `replication_factor: u32`.

Accessors: `id()`, `name()`, `fields()`, `vector_fields()`, `scalar_fields()`, `blob_fields()`, `get_field(name)`, `shard_count()`, `replication_factor()`.

**`CollectionSchemaBuilder`** constructed via `CollectionSchema::builder(name)`:
- `add_field(FieldDefinition)`, `shard_count(u32)`, `replication_factor(u32)`
- `build() -> Result<CollectionSchema>` -- validates:
  - Name must not be empty
  - Field names must be unique (checked via `HashSet`)
  - Vector fields must have `dimension > 0`
  - On success, assigns a random `Uuid::new_v4()` as the collection ID

### 3.4 Record and FieldValue

**File:** `record.rs`

**`FieldValue`** -- 13 variants: `String(String)`, `Int32(i32)`, `Int64(i64)`, `Float32(f32)`, `Float64(f64)`, `Bool(bool)`, `Timestamp(i64)` (unix ms), `Vector(DenseVector)`, `Json(serde_json::Value)`, `BlobRef(String)` (URI), `Uuid(Uuid)`, `Array(Vec<FieldValue>)`, `Null`.

**`Record`** -- a UUID-keyed row with a `HashMap<String, FieldValue>` payload:
- `new()` -- auto-generates `Uuid::new_v4()`
- `with_id(id)` -- explicit ID
- `set(field, value)`, `get(field)`, `remove(field)`, `fields()`, `field_names()`, `has_field(field)`

### 3.5 VedaError

**File:** `error.rs`

11 variants via `thiserror::Error`:

| Variant | Template |
|---|---|
| `InvalidSchema(String)` | "Invalid schema: {0}" |
| `FieldNotFound(String)` | "Field not found: {0}" |
| `DimensionMismatch { expected, actual }` | "Dimension mismatch: expected {expected}, got {actual}" |
| `CollectionNotFound(String)` | "Collection not found: {0}" |
| `CollectionAlreadyExists(String)` | "Collection already exists: {0}" |
| `Storage(String)` | "Storage error: {0}" |
| `Index(String)` | "Index error: {0}" |
| `Query(String)` | "Query error: {0}" |
| `ModalityNotSupported { model, modality }` | "Modality not supported: model '{model}' does not support '{modality}'" |
| `Io(std::io::Error)` | "IO error: {0}" (auto-`From`) |
| `Serialization(serde_json::Error)` | "Serialization error: {0}" (auto-`From`) |

A type alias `pub type Result<T> = std::result::Result<T, VedaError>` is provided.

### 3.6 SIMD Module

**File:** `simd.rs`

Provides hardware-accelerated distance computation with runtime CPU feature detection.

**Architecture:** On `x86_64`, each public function checks `is_x86_feature_detected!("avx2")` and `is_x86_feature_detected!("fma")` at runtime. If both are available, the AVX2+FMA implementation is called; otherwise, the scalar fallback is used.

**AVX2+FMA implementations** (in `mod avx2`, gated by `#[cfg(target_arch = "x86_64")]`):
- `dot_product_avx2` -- `_mm256_fmadd_ps` in 8-wide chunks, scalar tail
- `l2_squared_avx2` -- subtract, fmadd of diff*diff, scalar tail
- `manhattan_avx2` -- subtract, `_mm256_andnot_ps` with sign mask for abs, scalar tail
- `l2_norm_squared_avx2` -- fmadd of `va*va`, scalar tail

All use a shared `hsum_avx` helper that horizontally sums a `__m256` to a scalar via `_mm256_extractf128_ps` / `_mm_movehdup_ps` / `_mm_movehl_ps` / `_mm_add_ss`.

**Scalar fallback implementations:** `dot_product_scalar`, `l2_squared_scalar`, `manhattan_scalar`, `l2_norm_squared_scalar` -- simple `iter().zip().map().sum()` patterns.

**Public dispatch functions:**
- `dot_product(a, b) -> f32`
- `l2_squared(a, b) -> f32`
- `manhattan_distance(a, b) -> f32`
- `l2_norm_squared(a) -> f32`
- `cosine_distance(a, b) -> f32` -- `1.0 - dot / (norm_a * norm_b)`, returns 1.0 if denom is zero
- `euclidean_distance(a, b) -> f32` -- `l2_squared(a, b).sqrt()`

### 3.7 Metrics

**File:** `metrics.rs`

A global singleton (`static GLOBAL_METRICS`) using `AtomicU64` counters with `Ordering::Relaxed`.

**Counters:** `query_count`, `query_latency_sum_us`, `insert_count`, `insert_latency_sum_us`, `error_count`.

**Methods:**
- `Metrics::global()` -- returns `&'static Metrics`
- `record_query(latency: Duration)` -- increments `query_count`, adds `latency.as_micros()` to sum
- `record_insert(latency: Duration)` -- same pattern for inserts
- `record_error()` -- increments `error_count`
- `export_prometheus()` -- generates Prometheus text exposition format with metrics:
  - `veda_query_total`, `veda_query_latency_microseconds_total`
  - `veda_insert_total`, `veda_insert_latency_microseconds_total`
  - `veda_errors_total`

### 3.8 Embedding

**File:** `embedding.rs`

**`Modality`** -- 5 variants: `Text`, `Image`, `Audio`, `Video`, `Pdf`.
- `from_mime(mime: &str)` -- detects modality from MIME prefix (`text/`, `image/`, `audio/`, `video/`, `application/pdf`)
- `from_field_name(name: &str)` -- heuristic detection from field name substrings (e.g. "image", "photo", "picture" map to `Image`)

**`EmbeddingModelConfig`** -- model configuration:
- `name: String`, `endpoint: String`, `supported_modalities: Vec<Modality>`, `output_dimension: u32`, `max_input_size_bytes: usize`, `timeout_seconds: u32`, `batch_size: u32`

**`EmbeddingModelRegistry`** -- registry of model configurations:
- `register(config)`, `get(name)`, `find_by_modality(modality)`, `list()`
- `validate_modality(model_name, modality)` -- returns `Ok(())` or `Err(ModalityNotSupported)` with a suggestion for an alternative model

**`AutoEmbedConfig`** -- ties a source field to a target vector field via a model:
- `source_field`, `target_field`, `model`, `modality`

**`validate_auto_embed_config(config, registry, schema)`** -- validates: source field exists, target field is a vector, model exists and supports the modality, output dimension matches target field dimension.

### 3.9 EmbeddingClient

**File:** `embedding_client.rs`

A mock embedding client that produces deterministic vectors for testing without external model services.

**`EmbeddingClient`** methods:
- `new(config: EmbeddingModelConfig)` -- stores the config
- `embed_text(text: &str)` -- validates input, returns deterministic vector
- `embed_bytes(data: &[u8], modality: Modality)` -- validates input, returns deterministic vector
- `validate_input(data, modality)` -- checks modality is supported and input size is within `max_input_size_bytes`

**Deterministic vector generation** via `deterministic_vector(data, dimension)`:
1. Hash input bytes with FNV-1a-like `simple_hash` (seed `2166136261`, multiplier `16777619`)
2. Feed hash as seed to `xorshift32` PRNG (`state ^= state << 13; state ^= state >> 17; state ^= state << 5`)
3. Map each u32 output to `[-1.0, 1.0]` range
4. L2-normalize the resulting vector

### 3.10 Chunking Module

**File:** `chunking.rs`

Provides 7 configurable strategies for splitting document text into chunks suitable for embedding and vector search.

**`ChunkingStrategy`** -- 7 variants:

| Strategy | Description |
|---|---|
| `MarkdownRecursive` | Recursively splits markdown by structure (headings, paragraphs, sentences). Default strategy with 2000-char chunks and 300-char overlap. |
| `MarkdownHeading(1)` | Splits on H1 (`#`) headings. Each section becomes one chunk. |
| `MarkdownHeading(2)` | Splits on H2 (`##`) headings. |
| `MarkdownHeading(3)` | Splits on H3 (`###`) headings. |
| `Semantic` | Computes word-overlap Jaccard similarity between adjacent windows. Splits where similarity drops below a threshold, indicating a topic change. |
| `Sentence` | Splits on sentence boundaries (`.`, `!`, `?`) respecting chunk size limits. |
| `FixedSize` | Fixed character count chunks with configurable overlap. |
| `Page` | Splits on page separator markers (e.g., `---` or form feed characters). |
| `None` | No chunking; the entire document is returned as a single chunk. |

**Semantic chunking algorithm:**
1. Tokenize text into words
2. Create sliding windows of consecutive words
3. For each pair of adjacent windows, compute Jaccard similarity: `|intersection| / |union|` of the word sets
4. Where similarity drops below the threshold, insert a chunk boundary
5. Merge small chunks that fall below the minimum size

**Default parameters:** `chunk_size = 2000` characters, `chunk_overlap = 300` characters (~15% overlap).

---

## 4. veda-storage -- Storage Engine

**Path:** `crates/veda-storage/src/`

Implements the persistence layer: on-disk vector storage, scalar indexes, segment lifecycle, WAL, blob storage, and compaction.

### 4.1 Vector Storage Format

**File:** `vector_storage.rs`

A custom binary format with a 32-byte header followed by packed `f32` body.

**Header layout** (all fields little-endian):

| Offset | Size | Field |
|---|---|---|
| 0 | 4 | Magic bytes: `b"VEDA"` |
| 4 | 4 | Version: `u32 = 1` |
| 8 | 4 | Dimension: `u32` |
| 12 | 8 | Count: `u64` |
| 20 | 4 | Data type: `u32` (0 = f32) |
| 24 | 8 | Reserved: zero bytes |

**`VectorStorageWriter`** -- writes vectors sequentially using `BufWriter<File>`:
- `create(path, dimension)` -- creates file, writes header with count=0
- `append(vector: &[f32]) -> Result<u32>` -- validates dimension, writes raw f32 bytes, returns 0-indexed ID
- `flush()`, `count()`
- `close()` -- flushes, seeks to offset 0, rewrites header with final count

**`VectorStorageReader`** -- memory-mapped reader via `memmap2::Mmap`:
- `open(path)` -- validates magic/version, verifies body size matches `count * dimension * 4` bytes
- `get(id: u32) -> Result<&[f32]>` -- returns a zero-copy slice into mmap'd memory, asserts f32 alignment at runtime
- `get_batch(ids: &[u32])`, `iter()` -- batch and sequential access
- `dimension()`, `count()`

### 4.2 Scalar Indexes

**`InvertedIndex`** (`inverted_index.rs`) -- `BTreeMap<String, RoaringBitmap>`:
- `insert(doc_id, value)`, `search_exact(value)`, `search_prefix(prefix)`, `search_in(values)`, `search_contains(substring)`, `contains(value, doc_id)`, `all_doc_ids()`, `term_count()`, `doc_count()`
- Persistence: `save(path)` / `load(path)` via JSON with serialized bitmap bytes

**`RangeIndex`** (`range_index.rs`) -- sorted `Vec<(f64, u32)>` with binary search:
- `insert(doc_id, value)` -- appends, marks as unsorted
- `build()` -- sorts by value, must be called before queries
- `search_eq(value)`, `search_range(min, max)`, `search_gt(value)`, `search_gte(value)`, `search_lt(value)`, `search_lte(value)` -- all use `partition_point` for O(log n) lower bound
- Persistence: `save(path)` / `load(path)`

**`BitmapIndex`** (`bitmap_index.rs`) -- `HashMap<String, RoaringBitmap>`:
- `insert(doc_id, value)`, `search_eq(value)`, `search_in(values)`, `search_not(value)`, `distinct_values()`, `doc_count()`
- Tracks `total_docs` as max doc_id + 1
- Persistence: `save(path)` / `load(path)`

### 4.3 DeleteBitmap

**File:** `delete_bitmap.rs`

A `RoaringBitmap` wrapper for soft-delete tracking. Each set bit indicates a logically deleted document.

Methods: `mark_deleted(doc_id)`, `is_deleted(doc_id)`, `deleted_count()`, `save(path)`, `load(path)`.

### 4.4 Segment Format

**File:** `segment.rs`

A sealed segment is an immutable directory with the following structure:

```
segment_{uuid}/
  vectors/
    {field_name}.vec          # VectorStorageWriter output per vector field
  scalars/
    {field_name}.inv          # InvertedIndex per string/uuid/json field
    {field_name}.rng          # RangeIndex per numeric/timestamp field
    {field_name}.bmp          # BitmapIndex per bool field
  metadata.json               # SegmentMetadata
  schema.json                 # CollectionSchema
  records.jsonl               # One JSON object per row (scalar + blob fields)
  delete_bitmap.bin           # Serialized RoaringBitmap
  blob_refs.json              # Per-doc blob reference mappings
```

**`SegmentMetadata`**: `id: SegmentId`, `row_count: u64`, `schema_version: u32`, `created_at: i64` (unix seconds), `sealed: bool`.

**Scalar index type selection** (`scalar_index_kind` function):
- `String`, `Uuid`, `Json` --> `Inverted`
- `Bool` --> `Bitmap`
- `Int32`, `Int64`, `Float32`, `Float64`, `Timestamp` --> `Range`
- All other types --> `Inverted` (fallback)

### 4.5 SegmentWriter

**File:** `segment.rs`

Creates the directory structure and writes data per-field as records are inserted.

**`create(dir, schema)`** -- creates `vectors/` and `scalars/` subdirectories, initializes `VectorStorageWriter` per vector field and scalar index builders per scalar field, opens JSONL writer.

**`insert(record) -> Result<u32>`** -- for each record:
1. Writes vector fields to corresponding `.vec` files
2. Indexes scalar fields into the appropriate scalar index (inverted/range/bitmap) based on `scalar_index_kind`
3. Collects blob references
4. Writes a JSONL line with `__record_id__` plus all scalar and blob field values

**`seal()`** -- consumes the writer:
1. Closes all vector writers (finalizes headers with correct count)
2. Flushes the JSONL writer
3. Builds and saves all range indexes (calls `build()` then `save()`)
4. Saves all inverted and bitmap indexes
5. Creates and saves an empty `DeleteBitmap`
6. Writes `blob_refs.json`
7. Writes `metadata.json` with `sealed: true`
8. Writes `schema.json`

### 4.6 SegmentReader

**File:** `segment.rs`

Opens a sealed segment directory for reading.

**`open(dir)`** -- loads `metadata.json` (rejects unsealed segments), `schema.json`, opens `VectorStorageReader` per vector field, loads scalar indexes per field, loads delete bitmap.

**Methods:**
- `get_vector(field, doc_id) -> Result<&[f32]>` -- zero-copy via mmap
- `get_record(doc_id)` -- reads JSONL line at position `doc_id`, parses scalar fields, restores vector fields from readers, returns a full `Record`
- `delete_bitmap()` / `delete_bitmap_mut()` -- access the soft-delete tracking
- `inverted_index(field)`, `range_index(field)`, `bitmap_index(field)` -- access scalar indexes
- `schema()`, `row_count()`, `active_row_count()` (total minus deleted)

### 4.7 GrowingSegment

**File:** `growing_segment.rs`

The active, mutable segment that receives writes. All data lives in memory.

**Internal state:**
- `records: Vec<Record>` -- full record data
- `vectors: HashMap<String, Vec<Vec<f32>>>` -- per vector field, flattened storage for fast scan
- `size_bytes: usize` -- estimated memory usage
- `max_size_bytes: usize` -- seal threshold (default `DEFAULT_MAX_SIZE_BYTES = 512 * 1024 * 1024` = 512 MiB)
- `delete_bitmap: DeleteBitmap` -- soft-delete tracking
- Overhead estimate: `SCALAR_OVERHEAD_BYTES = 200` bytes per record for scalar fields

**Methods:**
- `new(schema)` -- creates with default 512 MiB threshold
- `with_max_size(schema, max_size_bytes)` -- custom threshold
- `insert(record) -> Result<u32>` -- validates dimension match, extracts and stores vectors separately, tracks memory usage
- `should_seal()` -- returns `size_bytes >= max_size_bytes`
- `search(field, query, k, metric, filter)` -- brute-force scan using a `BinaryHeap` max-heap of `OrdF32Entry { dist, doc_id }` for top-k maintenance. Skips deleted docs and docs not in filter bitmap.
- `seal(dir)` -- creates a `SegmentWriter`, inserts all non-deleted records, calls `seal()`. Returns `SegmentMetadata`.
- `mark_deleted(doc_id)`, `is_deleted(doc_id)`, `get_record(doc_id)`, `get_vector(field, doc_id)`

**Distance function selection** (`select_distance_fn`):
- `Cosine` --> `simd::cosine_distance`
- `Euclidean` --> `simd::euclidean_distance`
- `DotProduct` --> `-simd::dot_product(a, b)` (negated for "lower is better")
- `Manhattan` --> `simd::manhattan_distance`
- `Hamming` / `Jaccard` --> falls back to `simd::euclidean_distance`

### 4.8 SegmentManager

**File:** `segment_manager.rs`

Manages the lifecycle of segments for a single shard.

**State:** `schema`, `shard_id`, `base_dir` (at `base_dir/shard_{shard_id}/`), `growing: GrowingSegment`, `sealed_segments: Vec<SealedSegmentHandle>`, `seal_threshold_bytes`.

**`SealedSegmentHandle`**: `metadata: SegmentMetadata`, `reader: SegmentReader`, `dir: PathBuf`.

**Methods:**
- `new(schema, shard_id, base_dir, seal_threshold_bytes)` -- creates shard directory, initializes growing segment
- `insert(record)` -- auto-seals if threshold reached, then inserts
- `seal_growing()` -- seals current growing segment to `segment_{uuid}/` directory, opens `SegmentReader`, creates new growing segment. Returns `None` if growing segment is empty.
- `delete_record(record_id)` -- scans growing segment then sealed segments for the record, marks as deleted. Returns `true` if found.
- `load_sealed_segments()` -- scans shard directory for `segment_*` subdirectories, opens each as a `SegmentReader`. Used for recovery on restart.
- `growing_segment()`, `sealed_segments()`, `total_records()`, `segment_ids()`

### 4.9 WAL (Write-Ahead Log)

**File:** `wal.rs`

**`WalEntry`** enum: `Insert { collection, shard_id, record }`, `Delete { collection, shard_id, record_id }`.

**Traits:**
- `WalWriter: Send + Sync` -- `append(entry) -> Result<u64>` (returns sequence number)
- `WalReader: Send + Sync` -- `read_from(seq, max_entries)`, `latest_seq()`, `acknowledge(seq)` (allows purge)

**`InMemoryWal`** -- implementation using `RwLock<Vec<(u64, WalEntry)>>` and `AtomicU64` for sequence numbering:
- `append` -- `next_seq.fetch_add(1, SeqCst)`, pushes to entries
- `read_from` -- filters entries with `seq >= start`
- `acknowledge(seq)` -- retains only entries with sequence > seq

**`NatsWal`** -- placeholder struct for future NATS JetStream integration.

### 4.10 Blob Storage

**File:** `blob_store.rs`

**`BlobStore` trait** (`Send + Sync`): `put(key, data)`, `get(key)`, `delete(key)`, `exists(key)`, `list(prefix)`.

**`LocalBlobStore`** -- filesystem-based implementation:
- `new(base_dir)` -- creates directory
- `put` -- writes file at `base_dir/key`, creating parent directories
- `get` -- reads file, returns `VedaError::Storage` if not found
- `list` -- recursively walks directory, returns keys matching prefix

**`S3BlobStore`** -- placeholder, returns `VedaError::Storage("not yet implemented")`.

### 4.11 BlobManager

**File:** `blob_manager.rs`

Orchestrates blob lifecycle for a collection.

**`BlobManager`**: wraps a `Box<dyn BlobStore>` and a `collection_name`.

**Methods:**
- `process_insert(record, schema)` -- for each blob field in the schema: decodes `FieldValue::String` as base64, uploads via `store.put()` with key `{collection}/{record_id}/{field_name}`, replaces the field with `FieldValue::BlobRef(key)`
- `resolve_blobs(record, schema)` -- fetches blob data via `store.get()`, encodes as base64, replaces `BlobRef` with `FieldValue::String`
- `delete_blobs(record_id)` -- lists all keys under `{collection}/{record_id}/`, deletes each

### 4.12 Compaction

**File:** `compaction.rs`

**`CompactionConfig`** defaults: `min_segments: 3`, `max_merge_segments: 10`, `delete_ratio_threshold: 0.2`.

**`select_segments_for_compaction(segments, config)`**:
1. Select segments where `deleted_count / row_count >= delete_ratio_threshold`
2. If total segment count >= `min_segments`, also select smallest segments (by `row_count`) up to `max_merge_segments`
3. Cap result at `max_merge_segments`

**`compact_segments(segments, schema, output_dir)`**:
- Creates a `SegmentWriter`
- Iterates all input segments, skips deleted rows, writes non-deleted records
- Seals the writer
- Returns `CompactionResult { new_segment, merged_segment_ids, record_count, purged_count }`

---

## 5. veda-index -- Indexing Engine

**Path:** `crates/veda-index/src/`

Implements multiple ANN index algorithms and quantization techniques.

### 5.1 VectorIndex Trait

**File:** `traits.rs`

```rust
pub trait VectorIndex: Send + Sync {
    fn build(&mut self, vectors: &[&[f32]]) -> Result<()>;
    fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>>;
    fn search_filtered(&self, query: &[f32], k: usize, filter: &RoaringBitmap) -> Result<Vec<SearchResult>>;
    fn insert(&mut self, id: u32, vector: &[f32]) -> Result<()>;
    fn delete(&mut self, id: u32) -> Result<()>;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool;
    fn save(&self, path: &Path) -> Result<()>;
    fn load(path: &Path, metric: DistanceMetric, dimension: u32) -> Result<Self>;
    fn metric(&self) -> DistanceMetric;
    fn dimension(&self) -> u32;
}
```

**`SearchResult`**: `id: u32`, `distance: f32`. Lower distance = greater similarity for all metrics (dot product is negated).

The default `search_filtered` implementation oversamples by 10x and post-filters.

### 5.2 Distance Module

**File:** `distance.rs`

`compute_distance(a, b, metric)` dispatches to the appropriate `veda_core::simd` function:
- `Cosine` --> `simd::cosine_distance`
- `Euclidean` --> `simd::euclidean_distance`
- `DotProduct` --> `-simd::dot_product` (negated)
- `Manhattan` --> `simd::manhattan_distance`
- Other metrics --> panics

### 5.3 FlatIndex

**File:** `flat.rs`

Brute-force exact nearest neighbor search. Computes distance from query to every non-deleted vector. Ideal for small datasets and as ground truth for testing.

**State:** `vectors: Vec<Vec<f32>>`, `deleted: HashSet<u32>`, `metric`, `dimension`.

**Search:** Uses a `BinaryHeap` (max-heap by distance) of size k. For each non-deleted vector, computes distance and evicts the farthest candidate when the heap exceeds k.

**Persistence:** Serialized via `FlatIndexData` to/from JSON.

### 5.4 HnswIndex

**Files:** `hnsw/mod.rs`, `hnsw/graph.rs`, `hnsw/builder.rs`, `hnsw/searcher.rs`

Hierarchical Navigable Small World graph (Malkov & Yashunin, 2016). Provides approximate nearest neighbor search with logarithmic search complexity and >95% recall with default parameters.

**Default parameters:**
- `m = 16` (connections per node per layer)
- `m_max0 = 2 * m = 32` (connections at layer 0)
- `ef_construction = 200` (construction beam width)
- `ef_search = 200` (search beam width)
- `ml = 1.0 / ln(m)` (level generation multiplier)

**Graph structure** (`HnswGraph`):
- `nodes: Vec<Option<HnswNode>>` -- sparse node storage
- `entry_point: Option<u32>` -- global entry point
- `max_level: usize`
- Each `HnswNode` has `id`, `level`, `neighbors: Vec<Vec<u32>>` (per-layer neighbor lists)

**Level selection** (`HnswBuilder::random_level`): `floor(-ln(uniform(0,1)) * ml)`, capped at 16.

**Build algorithm** (Algorithm 1 from the paper):
1. Select random level for new node
2. Add node to graph
3. If first node, set as entry point and return
4. Phase 1: Greedy descent from top level to `new_level + 1` -- at each layer, find the single nearest neighbor
5. Phase 2: For layers `min(new_level, top)` down to 0:
   - Search the layer with `ef_construction` to find candidates
   - Select up to `m_max0` (layer 0) or `m` (other layers) nearest neighbors
   - Set bidirectional connections
   - Prune over-full neighbors by keeping only the closest `max_conn` (simple nearest selection, no RobustPrune needed)
6. If `new_level > current_max_level`, update entry point and max level

**Search algorithm:**
1. Greedy descent from entry point through layers `max_level` down to 1
2. At layer 0: `search_layer` with `ef = max(ef_search, k)` using dual heaps (min-heap for candidates, max-heap for results bounded at ef)
3. Return top-k from results

**Adaptive filtered search** (`search_layer_filtered`) -- selects strategy based on filter selectivity:
- **>50% selectivity (PreFilter):** Normal search with `ef * 2`, then discard non-matching results
- **1-50% selectivity (InAlgorithm):** Traverse graph normally (all neighbors, preserving connectivity) but only add to result set if candidate passes filter. Increases ef inversely proportional to selectivity, capped at `ef * 10`.
- **<1% selectivity (PostFilter):** Heavy oversampling with `ef * 20`, then filter. If too few results, falls back to brute-force scan of filtered IDs only.

**Persistence:** Serialized via `HnswIndexData` to/from JSON (includes full graph, all vectors, and parameters).

### 5.5 DiskAnnIndex

**Files:** `diskann/mod.rs`, `diskann/graph.rs`, `diskann/builder.rs`, `diskann/searcher.rs`

Vamana graph index for high-recall ANN search on large datasets. Batch-build only -- does not support incremental inserts.

**Default parameters:**
- `r = 64` (max degree / out-neighbors per node)
- `l_build = 100` (build-time search list size)
- `alpha = 1.2` (RobustPrune diversity parameter)
- `l_search = 100` (search-time list size)

**Graph structure** (`VamanaGraph`): simple `adjacency: Vec<Vec<u32>>` with a `medoid: Option<u32>` entry point.

**Build algorithm** (`build_vamana`):
1. **Medoid computation** (`compute_medoid`): compute centroid of all vectors, find vector closest to centroid
2. **Random graph init:** For each node, pick `min(r, n-1)` random neighbors (no duplicates)
3. **Iterative refinement:** Shuffle vector order, then for each vector p:
   - Greedy search from medoid to find L nearest neighbors of p
   - **RobustPrune** to select up to R neighbors: greedily pick closest, then remove candidates where `dist(v, p*) * alpha <= dist(v, p)`. This encourages diverse neighbor sets.
   - Set bidirectional connections; if any neighbor exceeds R connections, prune it via RobustPrune

**Search** (`greedy_search`): Beam search maintaining a sorted candidate list of size `l_search`. Iteratively expands the closest unvisited candidate's neighbors.

### 5.6 IvfPqIndex

**File:** `ivfpq/mod.rs`

Inverted File with Product Quantization for memory-efficient ANN search.

**Default parameters:**
- `nlist = 256` (number of IVF partitions)
- `nprobe = 8` (number of partitions to probe at search time)
- `m_pq = 8` (number of PQ sub-quantizers)
- `nbits = 8` (bits per sub-quantizer code, so 256 centroids per sub-space)

**Build pipeline:**
1. **Train IVF centroids** -- k-means with `max_iter = 25` on all vectors. k-means uses random initialization with `k = min(nlist, n)`.
2. **Assign vectors to centroids** -- each vector assigned to nearest IVF centroid
3. **Train PQ codebooks** -- computes residuals `(vector - centroid)`, trains per-sub-quantizer k-means on the residual sub-vectors. Residual PQ significantly improves quantization quality.
4. **Encode vectors** -- each residual is quantized to `m_pq` uint8 PQ codes

**Search pipeline:**
1. Find `nprobe` nearest IVF centroids to the query
2. For each probed centroid: compute query residual `(query - centroid)`, precompute Asymmetric Distance Computation (ADC) table `[sub_quantizer][code] = squared_L2`, then sum ADC distances for each vector in the inverted list
3. Sort by approximate distance, take top `k * 10` candidates
4. **Exact reranking:** recompute exact distances for top candidates, return top-k

### 5.7 RaBitQuantizer

**File:** `quantization/rabitq.rs`

Random Bit Quantization: quantizes D-dimensional float vectors into compact binary codes (1 bit per dimension, packed into `Vec<u64>`).

**Training:**
1. Compute centroid of all vectors
2. Center each vector (`v - centroid`), compute its L2 norm
3. Binarize: bit = 1 if centered component >= 0, 0 otherwise. Packed into u64 words.
4. Compute `mean_abs_component`: average `|v_hat_i|` across all normalized centered vectors and all dimensions. This scaling factor corrects the sign approximation.

**Distance estimation** (`estimate_distance`):
- Center query: `q' = q - centroid`
- Asymmetric dot product: `raw_dot = 2 * sum_pos - sum_all` where `sum_pos` sums query components at set-bit positions (iterating via `trailing_zeros` / bit clearing)
- Estimated inner product: `v_norm * mean_abs_component * raw_dot`
- Final distance depends on metric:
  - `Cosine`: `1.0 - estimated_dot / (q_norm * v_norm)`
  - `Euclidean`: `sqrt(q_norm^2 + v_norm^2 - 2 * estimated_dot)`
  - `DotProduct`: `-estimated_dot`

### 5.8 ScalarQuantizer

**File:** `quantization/scalar.rs`

Maps each float dimension to uint8 in `[0, 255]`.

**Training:**
1. Compute per-dimension `min` and `max` across all vectors
2. Encode: `((val - min) / (max - min)).clamp(0, 1) * 255`, rounded to u8. Returns 128 if range < 1e-10.

**Methods:**
- `decode(code: &[u8]) -> Vec<f32>` -- approximate reconstruction: `min + (code / 255.0) * range`
- `estimate_distance(query, code_idx, metric)` -- decodes stored code, computes exact distance on reconstruction

### 5.9 SuffixArrayIndex

**Files:** `suffix_array/mod.rs`, `suffix_array/builder.rs`, `suffix_array/searcher.rs`

Full-text search index over string field content.

**Data structure:** `text: Vec<u8>` (concatenated documents with `\0` sentinels), `sa: Vec<u32>` (suffix array), `doc_boundaries: Vec<(start, end, doc_id)>`.

**Construction** (`build`): O(n log^2 n) prefix-doubling algorithm:
1. Concatenate all document texts with `\0` separator
2. Initialize ranks by character value
3. Repeatedly double comparison length, sort suffixes, recompute ranks
4. Stop when all ranks are unique

**Exact search** (`search_exact`): Binary search on the suffix array to find the range `[lo, hi]` of suffixes starting with the pattern. Uses `sa_range` with two binary searches for lower and upper bounds.

**Fuzzy search** (`search_fuzzy`):
1. Split pattern into words
2. For each word: find exact matches, then generate edit-distance variants (substitutions, deletions, insertions up to `max_edits`) and search each
3. Find documents containing ALL words
4. If `preserve_order`, verify words appear in ascending offset order within each document (greedy ascending sequence check)

**`search_doc_ids(pattern, config) -> RoaringBitmap`** -- returns unique matching document IDs as a bitmap. Config: `fuzzy: bool`, `max_edits: usize`, `preserve_order: bool`, `max_results: usize`.

---

## 6. veda-query -- Query Engine

**Path:** `crates/veda-query/src/`

Implements the filter language, search execution, multi-vector fusion, and distributed query coordination.

### 6.1 Filter AST

**File:** `filter/ast.rs`

**`FilterExpr`** -- 15 variants:
- Comparisons: `Eq`, `Ne`, `Gt`, `Gte`, `Lt`, `Lte` -- each with `field: String` and `value: FilterValue`
- Range: `Between { field, low, high }`
- Set: `In { field, values: Vec<FilterValue> }`
- String ops: `Contains { field, value: String }`, `StartsWith { field, value: String }`
- Null checks: `IsNull { field }`, `IsNotNull { field }`
- Logical: `And(Box, Box)`, `Or(Box, Box)`, `Not(Box)`

**`FilterValue`** -- 4 variants: `String(String)`, `Int(i64)`, `Float(f64)`, `Bool(bool)`.
- `as_f64()` -- converts numeric/bool to f64 for range queries
- `as_str_repr()` -- string representation for inverted/bitmap queries

### 6.2 Filter Parser

**File:** `filter/parser.rs`

Tokenizer + recursive descent parser.

**Tokenizer** -- hand-written character-by-character scanner producing:
- `Ident`, `StringLit` (single-quoted), `IntLit`, `FloatLit`, `BoolLit`
- Keywords (case-insensitive): `AND`, `OR`, `NOT`, `BETWEEN`, `IN`, `CONTAINS`, `STARTS_WITH`, `IS`, `NULL`, `TRUE`, `FALSE`
- Operators: `=`, `!=`, `>`, `>=`, `<`, `<=`
- Punctuation: `(`, `)`, `,`, `.`

**Grammar** (recursive descent):
```
or_expr     = and_expr ("OR" and_expr)*
and_expr    = not_expr ("AND" not_expr)*
not_expr    = "NOT" not_expr | primary
primary     = comparison | "(" or_expr ")"
comparison  = field op value
            | field "BETWEEN" value "AND" value
            | field "IN" "(" value ("," value)* ")"
            | field "CONTAINS" string
            | field "STARTS_WITH" string
            | field "IS" ["NOT"] "NULL"
```

Field names support dotted paths (e.g., `metadata.color`).

### 6.3 Filter Evaluator

**File:** `filter/evaluator.rs`

**`FilterContext`** -- holds references to scalar indexes per field:
- `inverted_indexes: HashMap<String, &InvertedIndex>`
- `range_indexes: HashMap<String, &RangeIndex>`
- `bitmap_indexes: HashMap<String, &BitmapIndex>`
- `total_docs: u32`

**`evaluate(expr) -> Result<RoaringBitmap>`** -- recursive evaluation:
- `Eq` -- tries range index first (for numeric values), then bitmap index, then inverted index
- `Ne` -- evaluates Eq, then subtracts from universe (`0..total_docs`)
- `Gt/Gte/Lt/Lte` -- requires range index, calls corresponding `search_*` method
- `Between` -- range index `search_range(lo, hi)`
- `In` -- tries bitmap index `search_in`, then inverted index `search_in`
- `Contains` -- inverted index `search_contains`
- `StartsWith` -- inverted index `search_prefix`
- `IsNull` -- universe minus all indexed docs for that field
- `IsNotNull` -- all indexed docs for that field
- `And` -- bitmap intersection (`&`)
- `Or` -- bitmap union (`|`)
- `Not` -- XOR with universe

### 6.4 Record Evaluator

**File:** `filter/record_eval.rs`

**`evaluate_record(expr, record) -> bool`** -- evaluates a filter expression directly against a `Record` without pre-built indexes. Used for the growing segment where no scalar indexes exist.

Supports all 15 expression types with runtime type coercion:
- Numeric comparisons extract f64 from `Int32/Int64/Float32/Float64/Timestamp`
- String comparisons use lexicographic ordering as fallback
- Cross-type equality (e.g., Float64 field vs Int value) handled via epsilon comparison

### 6.5 SearchExecutor

**File:** `search.rs`

**Request/response types:**
- `VectorSearchRequest`: `field`, `query_vector`, `k`, `metric`, `filter: Option<String>`
- `SearchResponse`: `results: Vec<ScoredRecord>`, `total_candidates`
- `ScoredRecord`: `id: Uuid`, `segment_id: Uuid`, `doc_id: u32`, `distance: f32`

**`SearchExecutor::search(request, growing_segment, sealed_segments)`:**
1. Parse filter expression (if provided)
2. Search growing segment: build filter bitmap by scanning records with `evaluate_record`, then brute-force vector search with filter
3. Search each sealed segment: build `FilterContext` from segment's scalar indexes, evaluate filter to bitmap, exclude deleted docs, call `index.search_filtered`
4. Merge all results: sort by distance ascending, truncate to k

### 6.6 Multi-Vector Fusion

**File:** `fusion.rs`

**`FusionStrategy`** -- 3 variants:
- `Rrf { k: u32 }` -- Reciprocal Rank Fusion
- `WeightedSum` -- weighted normalized scores
- `Intersection` -- only documents in ALL fields

**`fuse_results(field_results, strategy, weights, limit)`:**

Documents are matched by UUID across fields and segments.

**RRF:** `score(d) = sum over all fields i: 1.0 / (k + rank_i(d))` where `rank_i(d)` is the 1-based rank. Documents appearing in only some fields simply skip missing fields.

**WeightedSum:**
1. Normalize each field's distances to `[0, 1]` using per-field min/max
2. Convert to similarity: `sim = 1.0 - norm_dist`
3. `score(d) = sum(weight_i * sim_i(d))`

**Intersection:** Filter to documents appearing in ALL field results, then score via RRF with `k = 60`.

All strategies sort by score descending and truncate to limit.

### 6.7 DistributedQueryExecutor

**File:** `distributed.rs`

**`ShardSearcher` trait:** `search_shard(shard_id, request) -> Result<Vec<ScoredRecord>>`.

**`DistributedQueryExecutor`:**
- `shard_searchers: HashMap<u32, Box<dyn ShardSearcher>>`
- `register_shard(shard_id, searcher)`
- `search(request)` -- fans out to all registered shards, collects results, sorts by distance ascending, returns global top-k
- `search_multi_vector(query, filter, metric)` -- for each field query, fans out to all shards, merges per-field results, then calls `fuse_results`

### 6.8 LocalShardSearcher

**File:** `local_shard.rs`

In-process shard searcher wrapping a `SegmentManager`.

**State:** `segment_manager`, `indexes: HashMap<Uuid, HashMap<String, Box<dyn VectorIndex>>>` (per segment, per field).

**`build_indexes()`:** For each sealed segment, for each vector field, builds a `FlatIndex` by iterating all stored vectors.

**`search_shard()`:** Searches growing segment via `SearchExecutor`, then each sealed segment individually using the pre-built vector indexes. Merges and returns top-k.

---

## 7. veda-coordinator -- Cluster Coordination

**Path:** `crates/veda-coordinator/src/`

Manages cluster-wide state through a replicated state machine pattern.

### 7.1 ClusterCommand

**File:** `raft_types.rs`

6 variants:
- `CreateCollection { name, schema_json }` -- create a new collection
- `DropCollection { name }` -- drop a collection and its shard assignments
- `RegisterNode { node_id: u64, address, roles: Vec<String> }` -- add a cluster node
- `DeregisterNode { node_id }` -- remove a node
- `AssignShard { collection, shard_id, node_id }` -- assign a shard to a node
- `UnassignShard { collection, shard_id, node_id }` -- remove a shard assignment

**`ClusterResponse`:** `Ok` or `Error(String)`.

**`ClusterState`** (read-only snapshot):
- `collections: HashMap<String, String>` (name -> schema JSON)
- `nodes: HashMap<u64, NodeInfo>` (node_id -> info)
- `shard_assignments: HashMap<String, HashMap<u32, Vec<u64>>>` (collection -> shard_id -> [node_ids])
- `last_applied_index: u64`

**`NodeInfo`:** `address`, `roles: Vec<String>`, `status: NodeStatus` (`Active`, `Draining`, `Inactive`).

### 7.2 LogStore and ClusterStateMachine

**File:** `store.rs`

**`LogStore`** -- append-only `Vec<LogEntry>` with monotonically increasing indices starting at 1:
- `append(command) -> u64` -- returns assigned index
- `get_range(start, end)`, `get_from(start)`
- `purge(up_to_inclusive)` -- retains only entries with index > given value, bounds memory
- `next_index()`, `last_index()`, `len()`, `is_empty()`

**`ClusterStateMachine`** -- deterministic state machine:
- `apply(entry) -> ClusterResponse` -- enforces sequential application: entry index must equal `last_applied + 1`
- Business logic per command:
  - `CreateCollection` -- rejects duplicates
  - `DropCollection` -- also removes shard assignments
  - `RegisterNode` -- rejects already-registered nodes
  - `DeregisterNode` -- removes node
  - `AssignShard` -- validates collection and node exist, adds node to shard's replica list (no duplicates)
  - `UnassignShard` -- removes node from shard's replica list
- `snapshot() -> ClusterState` -- cloned read-only view
- Accessors: `collections()`, `nodes()`, `shard_assignments()`, `nodes_for_shard(collection, shard_id)`

### 7.3 CoordinatorHandle

**File:** `network.rs`

Async channel-based interface to the coordinator loop.

**`Coordinator::start() -> CoordinatorHandle`** -- spawns a tokio task running `coordinator_loop` with an `mpsc` channel (capacity 256).

**`coordinator_loop`** -- receives `CoordinatorMsg` variants:
- `Command` -- appends to `LogStore`, applies to `ClusterStateMachine`, sends response via `oneshot`
- `Query` -- returns `sm.snapshot()` via `oneshot`
- `Shutdown` -- breaks the loop

**`CoordinatorHandle`** (cloneable via `mpsc::Sender`):
- `execute(command) -> Result<ClusterResponse, CoordinatorError>` -- submit and await
- `query_state() -> Result<ClusterState, CoordinatorError>` -- read current state
- `shutdown()` -- graceful shutdown

**`ChannelNetwork`** -- in-process transport for future multi-node Raft testing. Maps `node_id: u64 -> mpsc::Sender<Vec<u8>>` behind a `RwLock`. Methods: `register`, `deregister`, `send`, `node_count`.

The current implementation uses a single-leader channel-based approach. The types and state machine are designed so that `openraft` can replace the channel transport without changing the command/response interface.

### 7.4 ShardRouter

**File:** `shard_router.rs`

Routes queries and writes to the correct nodes.

**`ReplicaSelection`** -- 3 strategies: `RoundRobin`, `Random`, `First`.
- `RoundRobin` -- `AtomicU64` counter mod active replicas
- `Random` -- counter + bit-mixing hash (`0xff51afd7ed558ccd` / `0xc4ceb9fe1a85ec53` Murmur-style)
- `First` -- always picks first active replica (deterministic testing)

**Methods:**
- `route_query(collection, state)` -- returns one `ShardLocation` per shard, selecting one active node per shard based on strategy. Returns error if any shard has no active replicas.
- `route_write(collection, state)` -- returns ALL active replica locations for all shards (writes go to every replica)
- `get_shard_replicas(collection, shard_id, state)` -- returns all active replicas for a specific shard
- `assign_shard(record_id, shard_count) -> u32` -- hash-based: extracts first 8 bytes of UUID as u64, computes `hash % shard_count`

---

## 8. veda-gateway -- API Gateway

**Path:** `crates/veda-gateway/src/`

HTTP API server built on `axum`.

### 8.1 AppState

**File:** `rest/state.rs`

**`AppState`** wraps `Arc<AppStateInner>`:
- `schemas: RwLock<HashMap<String, CollectionSchema>>`
- `segment_managers: RwLock<HashMap<String, HashMap<u32, SegmentManager>>>` -- keyed by (collection_name, shard_id)
- `query_executors: RwLock<HashMap<String, DistributedQueryExecutor>>`
- `embedding_registry: RwLock<EmbeddingModelRegistry>`
- `data_dir: PathBuf`
- `auth_enabled: bool`
- `jwt_secret: String`
- `users: HashMap<String, UserRecord>` -- `{ password: String, role: Role }`

**Constructors:**
- `AppState::new(data_dir)` -- auth disabled, default test secret
- `AppState::with_auth(data_dir, jwt_secret)` -- auth enabled, default admin/admin user
- `AppState::with_auth_and_users(data_dir, jwt_secret, users)` -- custom users

### 8.2 Authentication

**File:** `auth.rs`

JWT-based authentication via the `jsonwebtoken` crate.

**`Claims`:** `sub: String` (user ID), `role: Role`, `exp: u64` (unix timestamp).

**`Role`** enum: `Admin`, `Editor`, `Viewer`.

**Functions:**
- `generate_token(user_id, role, secret) -> Result<String, StatusCode>` -- creates JWT with 1-hour expiry using HMAC-SHA256 (`Header::default()`)
- `verify_token(token, secret) -> Result<Claims, StatusCode>` -- decodes and validates

**`auth_middleware`** -- Axum middleware:
- If `auth_enabled` is false, passes through (no-op)
- Extracts `Bearer` token from `Authorization` header
- Verifies token, inserts `Claims` into request extensions
- Returns 401 if missing or invalid

### 8.3 RBAC

**File:** `rbac.rs`

**`check_permission(role, action) -> bool`:**
- `Admin` -- all actions
- `Editor` -- `search`, `insert`, `upsert`, `delete_records`, `embed_insert`, `list_collections`, `describe_collection`, `get_record`
- `Viewer` -- `search`, `get_record`, `list_collections`, `describe_collection`

Per-action middleware functions: `require_admin`, `require_search`, `require_insert`, `require_delete_records`, `require_list_collections`, `require_describe_collection`, `require_get_record`, `require_embed_insert`.

### 8.4 REST Endpoints

**File:** `rest/mod.rs`

**Public routes** (no auth):

| Method | Path | Handler | Description |
|---|---|---|---|
| GET | `/v1/admin/health` | `admin::health` | Health check, returns `{"status":"healthy","version":"0.1.0"}` |
| POST | `/v1/auth/login` | `admin::login` | Authenticate, returns JWT token |

**Protected routes** (auth middleware when enabled):

| Method | Path | Handler | Description |
|---|---|---|---|
| GET | `/v1/admin/nodes` | `admin::list_nodes` | List cluster nodes |
| GET | `/v1/admin/metrics` | `admin::metrics` | Prometheus metrics export |
| POST | `/v1/collections` | `collections::create_collection` | Create collection |
| GET | `/v1/collections` | `collections::list_collections` | List all collections |
| GET | `/v1/collections/{name}` | `collections::describe_collection` | Describe collection schema |
| DELETE | `/v1/collections/{name}` | `collections::drop_collection` | Drop collection |
| POST | `/v1/collections/{name}/search` | `search::search` | Vector search (single or multi) |
| POST | `/v1/collections/{name}/insert` | `records::insert` | Insert records |
| POST | `/v1/collections/{name}/embed-insert` | `embed_insert::embed_insert` | Insert with auto-embedding |
| GET | `/v1/collections/{name}/records/{id}` | `records::get_record` | Get record by UUID |
| DELETE | `/v1/collections/{name}/records` | `records::delete_records` | Delete records by ID list |

### 8.5 Insert Flow

**Handler:** `records::insert`

1. Look up `CollectionSchema` by name (404 if not found)
2. For each record in the request:
   - Parse JSON map into `Record` using schema field definitions
   - Hash-based shard routing: `record.id().as_u128() as u32 % shard_count`
   - `SegmentManager::insert(record)`
3. Seal all growing segments (`seal_growing()`)
4. Rebuild query executors via `rebuild_executors`:
   - Creates new `SegmentManager` per shard
   - Calls `load_sealed_segments()` to recover from disk
   - Creates `LocalShardSearcher`, calls `build_indexes()`
   - Registers with new `DistributedQueryExecutor`

### 8.6 Search Flow

**Handler:** `search::search`

1. Look up `CollectionSchema` by name (404 if not found)
2. Resolve distance metric from schema for the queried vector field (defaults to `Cosine`)
3. **Single-vector:** Build `VectorSearchRequest`, call `DistributedQueryExecutor::search()`
4. **Multi-vector:** Parse fusion strategy (`rrf` with configurable k, `weighted_sum`, or `intersection`), build `MultiVectorQuery`, call `search_multi_vector()`
5. Format response with `SearchHit { id, distance, fields }` and `took_ms`

### 8.7 Error Handling

All handlers return structured JSON errors:
- **400 Bad Request** -- validation failures (empty name, unknown type, missing required field, invalid base64)
- **404 Not Found** -- collection or record not found
- **422 Unprocessable Entity** -- modality not supported (from embed-insert)
- **500 Internal Server Error** -- segment manager failures, index build failures

Error format: `{"error": "description string"}`

---

## 9. veda-server -- Binary Entry Point

**Path:** `crates/veda-server/src/main.rs`

The `veda-server` binary:

1. **Tracing initialization** -- `tracing_subscriber::fmt()` with JSON format and `RUST_LOG` environment variable filter (defaults to `info`)
2. **CLI argument parsing** (positional, no clap):
   - `args[1]` -- data directory (default: `./veda-data`)
   - `args[2]` -- listen address (default: `0.0.0.0:6340`)
3. **Startup banner** -- printed to stderr with version and config
4. **Create data directory** -- `fs::create_dir_all`
5. **Call `start_server(config)`** -- from `veda-gateway::server`
   - Creates `AppState::new(data_dir)` (auth disabled by default)
   - Builds router via `build_router`
   - Binds `TcpListener` and runs `axum::serve`

---

## 10. Niriksha Dashboard

**Path:** `niriksha/`

A data visualization and exploration dashboard for multimodal vector data.

### 10.1 Technology Stack

- **React 19** + **TypeScript**
- **Vite** (build tool, v8)
- **Zustand** (state management)
- **Tailwind CSS v4** (styling)
- **Framer Motion** (animations)
- **Axios** (HTTP client)
- **React Router v7** (routing)

### 10.2 Pages

7 pages in `niriksha/src/pages/`:

| Page | File | Purpose |
|---|---|---|
| Dashboard Home | `DashboardHome.tsx` | Overview metrics and status |
| Collection Browser | `CollectionBrowser.tsx` | Browse and manage collections |
| Search Playground | `SearchPlayground.tsx` | Interactive vector search testing |
| Vector Atlas | `VectorAtlas.tsx` | Vector space visualization |
| System Health | `SystemHealth.tsx` | Node and cluster monitoring |
| Index Manager | `IndexManager.tsx` | Index configuration and status |
| Evaluation View | `EvaluationView.tsx` | Search quality evaluation |

### 10.3 Design

- Dark theme with CSS custom properties
- Inter font family
- Component-based architecture with shared stores

### 10.4 API Proxy

The Vite dev server proxies `/v1` requests to `http://localhost:6340`, enabling seamless development against a local Veda instance:

```typescript
// vite.config.ts
server: {
  proxy: {
    '/v1': 'http://localhost:6340',
  },
}
```

---

## 11. Python SDK

**Path:** `sdk/python/`

Pure Python client wrapping the Veda REST API. Published as `veda-client` via hatchling. Requires Python >= 3.9 and `requests >= 2.28`.

### 11.1 VedaClient

**File:** `src/veda_client/client.py`

```python
client = VedaClient(url="http://localhost:6340", token=None)
```

**Methods:**
- `login(username, password)` -- POST `/v1/auth/login`, stores token
- `health()` -- GET `/v1/admin/health`
- `create_collection(schema)` -- POST `/v1/collections`
- `list_collections()` -- GET `/v1/collections`
- `describe_collection(name)` -- GET `/v1/collections/{name}`
- `drop_collection(name)` -- DELETE `/v1/collections/{name}`
- `insert(collection, records)` -- POST `/v1/collections/{collection}/insert`
- `get(collection, record_id)` -- GET `/v1/collections/{collection}/records/{record_id}`
- `delete(collection, ids)` -- DELETE `/v1/collections/{collection}/records`
- `search(collection, field, vector, limit, filter)` -- POST `.../search` (single-vector)
- `search_multi(collection, queries, fusion, limit, filter)` -- POST `.../search` (multi-vector)
- `embed_insert(collection, records, auto_embed)` -- POST `.../embed-insert`

### 11.2 Models

**File:** `src/veda_client/models.py`

Dataclasses: `FieldDefinition`, `CollectionSchema`, `SearchResult` (`id`, `distance`, `fields`), `SearchResponse` (`results`, `count`, `took_ms`).

### 11.3 Error Mapping

**File:** `src/veda_client/exceptions.py`

HTTP status codes are mapped to typed exceptions:
- `404` --> `CollectionNotFoundError`
- `400` --> `ValidationError`
- `422` --> `ModalityNotSupportedError`
- Other 4xx/5xx --> `VedaError`

All exceptions carry `status_code` and optional `details`.

---

## 12. Data Flow Diagrams

### 12.1 Write Path

```
Client
  │ POST /v1/collections/{name}/insert
  │ JSON body: { records: [...] }
  ▼
veda-gateway (records::insert)
  │ 1. Look up CollectionSchema
  │ 2. Parse JSON → Record
  │ 3. Hash shard: record.id().as_u128() % shard_count
  ▼
SegmentManager (per shard)
  │ 4. Check should_seal() → if true, seal first
  │ 5. GrowingSegment::insert(record)
  │    ├── Extract vectors → HashMap<String, Vec<Vec<f32>>>
  │    └── Store full Record in Vec<Record>
  │ 6. After all inserts: seal_growing()
  ▼
GrowingSegment::seal(dir)
  │ 7. SegmentWriter::create(dir, schema)
  │ 8. For each non-deleted record:
  │    ├── Write vectors to .vec files (VectorStorageWriter)
  │    ├── Index scalars (InvertedIndex/RangeIndex/BitmapIndex)
  │    └── Write JSONL line
  │ 9. SegmentWriter::seal()
  │    ├── Close vector writers (finalize headers)
  │    ├── Build + save range indexes
  │    ├── Save inverted + bitmap indexes
  │    ├── Save delete bitmap + metadata + schema
  ▼
Sealed segment on disk
  │ 10. rebuild_executors()
  │     ├── SegmentManager::load_sealed_segments()
  │     ├── LocalShardSearcher::build_indexes() (FlatIndex per field)
  │     └── Register with DistributedQueryExecutor
  ▼
Ready for search
```

### 12.2 Read Path

```
Client
  │ POST /v1/collections/{name}/search
  │ JSON body: { query: { vector: { field, values } }, filter, limit }
  ▼
veda-gateway (search::search)
  │ 1. Look up CollectionSchema
  │ 2. Resolve distance metric from schema
  │ 3. Build VectorSearchRequest
  ▼
DistributedQueryExecutor::search()
  │ 4. Fan out to all registered ShardSearchers
  ▼
LocalShardSearcher::search_shard()
  │
  ├── Growing segment:
  │     5a. If filter: scan records with evaluate_record() → bitmap
  │     5b. Brute-force search with filter + delete bitmap
  │
  └── Sealed segments (each):
        6a. Build FilterContext from scalar indexes
        6b. FilterParser::parse(filter_str) → FilterExpr AST
        6c. FilterContext::evaluate(expr) → RoaringBitmap
        6d. Exclude deleted docs
        6e. VectorIndex::search_filtered(query, k, bitmap)
            ├── HNSW: adaptive strategy by selectivity
            ├── DiskANN: beam search + filter
            ├── IVF-PQ: ADC search + rerank
            └── Flat: brute-force with filter
  │
  ▼
Merge: sort by distance ascending, truncate to k
  │
  ▼ (if multi-vector)
fuse_results(field_results, strategy, weights, limit)
  │
  ▼
SearchResponse → Client
```

### 12.3 Compaction

```
Background trigger (periodic or manual)
  │
  ▼
select_segments_for_compaction()
  │ 1. Select segments with delete_ratio >= 0.2
  │ 2. If total >= min_segments (3), add smallest
  │ 3. Cap at max_merge_segments (10)
  ▼
compact_segments(selected, schema, output_dir)
  │ 4. SegmentWriter::create(output_dir, schema)
  │ 5. For each input segment:
  │    └── For each doc: skip if deleted, else writer.insert(record)
  │ 6. writer.seal()
  ▼
CompactionResult
  │ 7. Replace old segments with new merged segment
  │ 8. Delete old segment directories
  │ 9. Rebuild indexes and executors
```

---

## 13. Configuration

The system uses environment variables and CLI arguments. A `veda.toml` configuration file is planned with these sections:

| Section | Description |
|---|---|
| `[cluster]` | Cluster mode (standalone/distributed), node ID, peers |
| `[coordinator]` | Consensus settings, election timeout |
| `[gateway]` | Listen address (default `0.0.0.0:6340`), TLS, CORS |
| `[storage]` | Object storage endpoint, credentials, bucket |
| `[message_queue]` | NATS/JetStream endpoint, stream names |
| `[query]` | Default search parameters (ef_search, nprobe) |
| `[data]` | Local data directory path |
| `[index]` | Default index types and parameters per modality |
| `[embedding]` | Model registry configuration (endpoints, dimensions) |
| `[niriksha]` | Dashboard settings (enabled, port) |
| `[logging]` | Log level, format (JSON/text), RUST_LOG override |
| `[metrics]` | Prometheus endpoint, push gateway |

Currently, configuration is done via:
- CLI args: `veda-server [data_dir] [listen_addr]`
- Environment: `RUST_LOG` (tracing filter), `VEDA_STORAGE_ENDPOINT`, `VEDA_STORAGE_ACCESS_KEY`, `VEDA_STORAGE_SECRET_KEY`, `VEDA_MQ_ENDPOINT`

---

## 14. Deployment

### 14.1 Standalone

Run the `veda-server` binary directly:

```bash
veda-server ./data 0.0.0.0:6340
```

Creates the data directory, starts the HTTP server. Auth is disabled by default. All data stored locally.

### 14.2 Docker Compose

**File:** `docker-compose.yml`

Three services:
- **MinIO** -- S3-compatible object storage for blobs (ports 9000/9001)
- **NATS** -- JetStream message queue for WAL replication (ports 4222/8222)
- **Veda** -- The database server (port 6340)

Named volumes: `minio-data`, `nats-data`, `veda-data`.

Veda connects to MinIO and NATS via environment variables. MinIO has a health check; Veda depends on MinIO being healthy and NATS being started.

### 14.3 FlashParse OCR (Optional Sidecar)

Veda provides end-to-end document processing: upload a document and the system handles OCR, chunking, and insertion into the target collection. This is powered by FlashParse, deployed as an optional Docker Compose service using the `ocr` profile.

```
                ┌──────────────────────────┐
                │      Client Layer        │
                │  POST /process-document  │
                │  POST /process-documents │
                └──────────┬───────────────┘
                           │ HTTP :6340
                ┌──────────▼───────────────┐
                │     veda-gateway         │
                │   document.rs handler    │
                └──────────┬───────────────┘
                           │ HTTP :8000 (internal)
                ┌──────────▼───────────────┐
                │     FlashParse           │
                │   (optional sidecar)     │
                │   OCR + VLM pipeline     │
                └──────────┬───────────────┘
                           │ ZIP (markdown + images)
                ┌──────────▼───────────────┐
                │     veda-gateway         │
                │   extract ZIP            │
                │   chunk markdown         │
                │   build Record objects   │
                │   shard-route + insert   │
                │   seal + rebuild indexes │
                └──────────────────────────┘
```

**Data flow:**

1. Client uploads a document (PDF, DOCX, image, etc.) to `POST /v1/collections/{name}/process-document`
2. Gateway forwards the file to FlashParse at `http://flashparse:8000/api/v1/pdf/process`
3. FlashParse processes the document using its VLM-based OCR pipeline and returns a ZIP containing structured markdown, extracted images, and metadata
4. Gateway extracts the ZIP, chunks the markdown using the configured chunking strategy (default: MarkdownRecursive)
5. For each text chunk: creates a `Record` with `content`, `source_file`, and `chunk_index` fields, routes to the correct shard via hash-based routing, and inserts via `SegmentManager::insert()`
6. For each extracted image: creates a `Record` with an `image_blob` BlobRef field and metadata
7. After all inserts: seals growing segments and rebuilds `DistributedQueryExecutor` (same as the insert endpoint)
8. Returns response with actual inserted record IDs

**Endpoints:**

| Method | Path | Description |
|---|---|---|
| POST | `/v1/collections/{name}/process-document` | Process a single document end-to-end |
| POST | `/v1/collections/{name}/process-documents` | Batch process multiple documents |

The batch endpoint (`process-documents`) accepts multiple files in the multipart form, processes each through FlashParse sequentially, inserts all resulting chunks and images, and returns aggregate stats with per-file results.

**Docker Compose profile mechanism:**

- FlashParse is defined under `profiles: ["ocr"]` in `docker-compose.yml`
- Standard `docker compose up` does NOT start FlashParse
- `docker compose --profile ocr up` starts FlashParse alongside the core services
- If FlashParse is not running, the `/process-document` endpoint returns `OCR_SERVICE_UNAVAILABLE` with setup instructions

**Configuration:**

- `VEDA_FLASHPARSE_ENDPOINT` environment variable overrides the default endpoint (`http://flashparse:8000`)
- FlashParse requires `VLM_ENDPOINT`, `VLM_API_KEY`, and `VLM_MODEL` for the vision-language model backend
- See `.env.example` at the project root for all configuration options

### 14.4 Kubernetes (Planned)

A Helm chart is planned for Kubernetes deployment with:
- StatefulSet for Veda nodes
- Persistent volumes for data
- ConfigMap for `veda.toml`
- Service for load balancing
- Optional MinIO and NATS sub-charts

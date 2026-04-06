# Veda Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build Veda, a production-grade distributed multimodal vector database with Niriksha visualization dashboard, from scratch in Rust.

**Architecture:** Fully disaggregated system with separate query/data/index/coordination nodes, NATS JetStream for WAL, MinIO/S3 for object storage, and a React/TypeScript dashboard (Niriksha). See `docs/plans/2026-04-06-veda-design.md` for full design.

**Tech Stack:** Rust (core engine), Tokio (async runtime), tonic (gRPC), axum (REST/SSE), serde (serialization), NATS JetStream (message queue), MinIO/S3 (object storage), React 19 + TypeScript + Vite (Niriksha), PyO3 + Maturin (Python SDK)

---

## Phase Overview

| Phase | What | Milestone |
|-------|------|-----------|
| **Phase 1** | Project scaffolding & core data types | Rust workspace compiles, basic types defined |
| **Phase 2** | Storage engine — segments, vectors, scalars | Can write/read vectors and scalar data to/from segments |
| **Phase 3** | Index engine — HNSW, Flat | Can build HNSW index and run ANN search on a single node |
| **Phase 4** | Query engine — filter, search, fusion | Can execute filtered vector search with multi-vector fusion |
| **Phase 5** | Coordination — Raft, cluster state, schema registry | Multi-node cluster forms and maintains consensus |
| **Phase 6** | Data nodes — WAL, ingestion, segment lifecycle | Can ingest data, flush to object storage, compact segments |
| **Phase 7** | Distributed query — shard routing, fan-out, merge | Full distributed search across shards/nodes |
| **Phase 8** | Gateway — REST API, gRPC API, SSE | Clients can interact via REST/gRPC |
| **Phase 9** | Additional indexes — DiskANN, IVF-PQ, RaBitQ | Full index type support with quantization |
| **Phase 10** | Blob storage — MinIO integration, BlobRef | Can store/retrieve multimodal blobs |
| **Phase 11** | Auto-embedding service — pluggable models, modality validation | Can auto-embed raw data on ingest |
| **Phase 12** | Pattern search — suffix array index | SoftMatcha-style fuzzy text search |
| **Phase 13** | Auth, security, observability | JWT, RBAC, Prometheus metrics, OpenTelemetry |
| **Phase 14** | Niriksha — Dashboard, Collection Browser, Search Playground | Core dashboard with search and browsing |
| **Phase 15** | Niriksha — Vector Atlas, Evaluation Engine | Visualization and data quality evaluation |
| **Phase 16** | Python SDK — PyO3 bindings | `pip install veda-client` |
| **Phase 17** | Deployment — Docker, Compose, Helm | Production deployment artifacts |
| **Phase 18** | Benchmarks, testing, hardening | Performance validation, stress testing |

---

## Phase 1: Project Scaffolding & Core Data Types

### Task 1.1: Initialize Rust Workspace

**Files:**
- Create: `Cargo.toml` (workspace root)
- Create: `crates/veda-core/Cargo.toml`
- Create: `crates/veda-core/src/lib.rs`
- Create: `crates/veda-storage/Cargo.toml`
- Create: `crates/veda-storage/src/lib.rs`
- Create: `crates/veda-index/Cargo.toml`
- Create: `crates/veda-index/src/lib.rs`
- Create: `crates/veda-query/Cargo.toml`
- Create: `crates/veda-query/src/lib.rs`
- Create: `crates/veda-coordinator/Cargo.toml`
- Create: `crates/veda-coordinator/src/lib.rs`
- Create: `crates/veda-gateway/Cargo.toml`
- Create: `crates/veda-gateway/src/lib.rs`
- Create: `crates/veda-server/Cargo.toml`
- Create: `crates/veda-server/src/main.rs`
- Create: `.gitignore`

**Step 1: Create workspace Cargo.toml**

```toml
[workspace]
resolver = "2"
members = [
    "crates/veda-core",
    "crates/veda-storage",
    "crates/veda-index",
    "crates/veda-query",
    "crates/veda-coordinator",
    "crates/veda-gateway",
    "crates/veda-server",
]

[workspace.package]
version = "0.1.0"
edition = "2024"
license = "Apache-2.0"
repository = "https://github.com/veda-db/veda"

[workspace.dependencies]
tokio = { version = "1", features = ["full"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
uuid = { version = "1", features = ["v4", "serde"] }
thiserror = "2"
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter", "json"] }
bytes = "1"
```

Each crate's `Cargo.toml` should reference workspace dependencies and declare inter-crate dependencies:
- `veda-core`: no internal deps (defines shared types)
- `veda-storage`: depends on `veda-core`
- `veda-index`: depends on `veda-core`, `veda-storage`
- `veda-query`: depends on `veda-core`, `veda-storage`, `veda-index`
- `veda-coordinator`: depends on `veda-core`
- `veda-gateway`: depends on `veda-core`, `veda-query`
- `veda-server`: depends on all crates

**Step 2: Create .gitignore**

```
/target
.env
*.swp
*.swo
.DS_Store
```

**Step 3: Create stub lib.rs for each crate**

Each `lib.rs` starts empty or with a module declaration. `veda-server/src/main.rs`:

```rust
fn main() {
    println!("Veda v0.1.0");
}
```

**Step 4: Verify workspace compiles**

Run: `cargo build`
Expected: Compiles successfully with no errors.

**Step 5: Initialize git and commit**

```bash
git init
git add -A
git commit -m "feat: initialize Rust workspace with crate structure"
```

---

### Task 1.2: Define Core Data Types — Field Types & Schema

**Files:**
- Create: `crates/veda-core/src/schema.rs`
- Create: `crates/veda-core/src/types.rs`
- Create: `crates/veda-core/src/error.rs`
- Modify: `crates/veda-core/src/lib.rs`
- Test: `crates/veda-core/tests/schema_test.rs`

**Step 1: Write failing tests for schema types**

```rust
// crates/veda-core/tests/schema_test.rs
use veda_core::schema::*;
use veda_core::types::*;

#[test]
fn test_create_collection_schema() {
    let schema = CollectionSchema::builder("product_catalog")
        .add_field(FieldDefinition::new("title", FieldType::String))
        .add_field(FieldDefinition::new("price", FieldType::Float64))
        .add_field(FieldDefinition::vector("text_embedding", 768, DistanceMetric::Cosine))
        .add_field(FieldDefinition::vector("image_embedding", 512, DistanceMetric::Cosine))
        .add_field(FieldDefinition::blob("image_blob"))
        .build()
        .unwrap();

    assert_eq!(schema.name(), "product_catalog");
    assert_eq!(schema.fields().len(), 5);
    assert!(schema.vector_fields().len() == 2);
    assert!(schema.blob_fields().len() == 1);
}

#[test]
fn test_field_type_validation() {
    let result = FieldDefinition::vector("bad_vec", 0, DistanceMetric::Cosine);
    // Dimension 0 should fail validation
    // (handle via builder validation or separate validate method)
}

#[test]
fn test_distance_metric_variants() {
    assert_ne!(DistanceMetric::Cosine, DistanceMetric::Euclidean);
    assert_ne!(DistanceMetric::Cosine, DistanceMetric::DotProduct);
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p veda-core`
Expected: FAIL — types don't exist yet.

**Step 3: Implement core types**

`crates/veda-core/src/types.rs`:
```rust
use serde::{Deserialize, Serialize};
use uuid::Uuid;

pub type RecordId = Uuid;
pub type CollectionId = Uuid;
pub type SegmentId = Uuid;
pub type ShardId = u32;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DistanceMetric {
    Cosine,
    Euclidean,
    DotProduct,
    Hamming,
    Manhattan,
    Jaccard,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FieldType {
    Vector { dimension: u32, metric: DistanceMetric },
    SparseVector,
    String,
    Int32,
    Int64,
    Float32,
    Float64,
    Bool,
    Timestamp,
    Array(Box<FieldType>),
    Json,
    Blob,
    Uuid,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IndexType {
    Hnsw,
    DiskAnn,
    IvfPq,
    Flat,
    SuffixArray,
    Inverted,
    Range,
    Bitmap,
    Hash,
    None,
}
```

`crates/veda-core/src/schema.rs`:
```rust
use crate::types::*;
use crate::error::VedaError;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldDefinition {
    name: String,
    field_type: FieldType,
    filterable: bool,
    index_type: IndexType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionSchema {
    id: CollectionId,
    name: String,
    fields: Vec<FieldDefinition>,
    shard_count: u32,
    replication_factor: u32,
}

// Builder pattern, field constructors, accessor methods
```

`crates/veda-core/src/error.rs`:
```rust
use thiserror::Error;

#[derive(Error, Debug)]
pub enum VedaError {
    #[error("Invalid schema: {0}")]
    InvalidSchema(String),
    #[error("Field not found: {0}")]
    FieldNotFound(String),
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: u32, actual: u32 },
    #[error("Collection not found: {0}")]
    CollectionNotFound(String),
    #[error("Collection already exists: {0}")]
    CollectionAlreadyExists(String),
    #[error("Storage error: {0}")]
    Storage(String),
    #[error("Index error: {0}")]
    Index(String),
    #[error("Query error: {0}")]
    Query(String),
    #[error("Modality not supported: model '{model}' does not support '{modality}'")]
    ModalityNotSupported { model: String, modality: String },
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p veda-core`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/veda-core/
git commit -m "feat: define core data types — schema, field types, distance metrics, errors"
```

---

### Task 1.3: Define Core Data Types — Records & Vectors

**Files:**
- Create: `crates/veda-core/src/record.rs`
- Create: `crates/veda-core/src/vector.rs`
- Test: `crates/veda-core/tests/record_test.rs`

**Step 1: Write failing tests**

```rust
// crates/veda-core/tests/record_test.rs
use veda_core::vector::DenseVector;
use veda_core::record::{Record, FieldValue};

#[test]
fn test_dense_vector_creation() {
    let v = DenseVector::new(vec![0.1, 0.2, 0.3]);
    assert_eq!(v.dimension(), 3);
    assert_eq!(v.as_slice(), &[0.1, 0.2, 0.3]);
}

#[test]
fn test_dense_vector_distance_cosine() {
    let a = DenseVector::new(vec![1.0, 0.0]);
    let b = DenseVector::new(vec![0.0, 1.0]);
    let dist = a.cosine_distance(&b);
    assert!((dist - 1.0).abs() < 1e-6); // orthogonal = distance 1.0
}

#[test]
fn test_dense_vector_distance_euclidean() {
    let a = DenseVector::new(vec![0.0, 0.0]);
    let b = DenseVector::new(vec![3.0, 4.0]);
    let dist = a.euclidean_distance(&b);
    assert!((dist - 5.0).abs() < 1e-6);
}

#[test]
fn test_record_with_multiple_fields() {
    let mut record = Record::new();
    record.set("title", FieldValue::String("Red Shoes".into()));
    record.set("price", FieldValue::Float64(89.99));
    record.set("embedding", FieldValue::Vector(DenseVector::new(vec![0.1, 0.2])));

    assert_eq!(record.get("title"), Some(&FieldValue::String("Red Shoes".into())));
    assert!(record.id().is_some()); // auto-generated UUID
}
```

**Step 2: Run tests — expect failure**

Run: `cargo test -p veda-core`

**Step 3: Implement vector and record types**

`crates/veda-core/src/vector.rs` — DenseVector with SIMD-ready distance functions (start with scalar, optimize later):
- `cosine_distance`, `euclidean_distance`, `dot_product`, `hamming_distance`, `manhattan_distance`
- `normalize`, `dimension`, `as_slice`, `as_bytes`

`crates/veda-core/src/record.rs` — Record as a HashMap<String, FieldValue> with auto-generated UUID:
- `FieldValue` enum: String, Int32, Int64, Float32, Float64, Bool, Timestamp, Vector, SparseVector, Json, BlobRef, Uuid, Array, Null

**Step 4: Run tests — expect pass**

Run: `cargo test -p veda-core`

**Step 5: Commit**

```bash
git add crates/veda-core/
git commit -m "feat: add dense vector with distance functions and record types"
```

---

### Task 1.4: SIMD-Optimized Distance Functions

**Files:**
- Modify: `crates/veda-core/src/vector.rs`
- Create: `crates/veda-core/src/simd.rs`
- Test: `crates/veda-core/tests/simd_test.rs`
- Create: `crates/veda-core/benches/distance_bench.rs`

**Step 1: Write benchmark and correctness tests**

Tests that verify SIMD and scalar produce identical results for all distance metrics across various dimensions (128, 256, 512, 768, 1024, 1536).

**Step 2: Implement SIMD distance functions**

Use `std::arch` for x86_64 (AVX2/AVX-512) and aarch64 (NEON). Feature-gated with runtime detection:

```rust
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub fn cosine_distance_simd(a: &[f32], b: &[f32]) -> f32 {
    // AVX2 implementation: process 8 f32s at a time
    // Fallback to scalar for remainder
}
```

**Step 3: Run tests + benchmarks**

Run: `cargo test -p veda-core && cargo bench -p veda-core`

**Step 4: Commit**

```bash
git commit -m "feat: add SIMD-optimized distance functions (AVX2, NEON)"
```

---

## Phase 2: Storage Engine

### Task 2.1: Vector Storage — Write & Read Raw Vectors

**Files:**
- Create: `crates/veda-storage/src/vector_storage.rs`
- Create: `crates/veda-storage/src/segment.rs`
- Test: `crates/veda-storage/tests/vector_storage_test.rs`

Implement mmap-friendly vector file format:
- Header: magic bytes, version, dimension, count, data type
- Body: tightly packed f32 arrays, aligned for mmap
- Support append (growing segment) and read (sealed segment)

**Step 1: Write tests** — create vector storage, append 1000 vectors, read back, verify equality.

**Step 2: Implement** — `VectorStorage` struct with `append`, `get`, `len`, `flush`, `mmap_open`.

**Step 3: Test + commit**

---

### Task 2.2: Scalar Storage — Inverted Index, Range Index, Bitmap

**Files:**
- Create: `crates/veda-storage/src/scalar_storage.rs`
- Create: `crates/veda-storage/src/inverted_index.rs`
- Create: `crates/veda-storage/src/range_index.rs`
- Create: `crates/veda-storage/src/bitmap_index.rs`
- Test: `crates/veda-storage/tests/scalar_storage_test.rs`

Implement scalar indexes for filter queries:
- Inverted index: String fields → posting lists (sorted doc IDs)
- Range index: Numeric/Timestamp fields → B-tree with range scan
- Bitmap index: Bool/low-cardinality fields → roaring bitmaps

---

### Task 2.3: Segment — Combine Vectors + Scalars + Metadata

**Files:**
- Modify: `crates/veda-storage/src/segment.rs`
- Create: `crates/veda-storage/src/delete_bitmap.rs`
- Test: `crates/veda-storage/tests/segment_test.rs`

Implement `Segment` struct that combines:
- Vector storage files (one per vector field)
- Scalar index files
- Delete bitmap (roaring bitmap)
- Metadata JSON (row count, min/max, schema version)
- Blob reference file

Support: `insert_record`, `get_record`, `delete_record` (soft delete via bitmap), `seal`, `is_sealed`.

---

### Task 2.4: Growing Segment — In-Memory Append-Only Buffer

**Files:**
- Create: `crates/veda-storage/src/growing_segment.rs`
- Test: `crates/veda-storage/tests/growing_segment_test.rs`

In-memory segment for recent writes:
- Append-only (no updates)
- Flat brute-force scan for search (small enough)
- `seal()` → flushes to disk as a sealed segment
- Configurable size threshold (default 512MB)

---

## Phase 3: Index Engine

### Task 3.1: Flat Index (Brute-Force)

**Files:**
- Create: `crates/veda-index/src/flat.rs`
- Create: `crates/veda-index/src/traits.rs`
- Test: `crates/veda-index/tests/flat_test.rs`

Define the `VectorIndex` trait and implement Flat index:

```rust
pub trait VectorIndex: Send + Sync {
    fn build(&mut self, vectors: &[&[f32]]) -> Result<(), VedaError>;
    fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>, VedaError>;
    fn insert(&mut self, id: u32, vector: &[f32]) -> Result<(), VedaError>;
    fn delete(&mut self, id: u32) -> Result<(), VedaError>;
    fn len(&self) -> usize;
    fn save(&self, path: &Path) -> Result<(), VedaError>;
    fn load(path: &Path) -> Result<Self, VedaError> where Self: Sized;
}
```

Flat index: linear scan with SIMD distance. Baseline for correctness testing.

---

### Task 3.2: HNSW Index — Core Implementation

**Files:**
- Create: `crates/veda-index/src/hnsw/mod.rs`
- Create: `crates/veda-index/src/hnsw/graph.rs`
- Create: `crates/veda-index/src/hnsw/builder.rs`
- Create: `crates/veda-index/src/hnsw/searcher.rs`
- Test: `crates/veda-index/tests/hnsw_test.rs`

Implement HNSW from scratch per Malkov & Yashunin (2016):
- Multi-layer graph with skip-list-inspired hierarchy
- Parameters: M (max connections), ef_construction, ef_search
- Support all distance metrics via the metric trait
- Incremental insertion (no full rebuild needed)
- Serialization to/from disk

Tests:
- Build with 10K random vectors, verify recall@10 > 95% vs flat index ground truth
- Test incremental insertion
- Test serialization round-trip
- Test with different distance metrics

---

### Task 3.3: HNSW — Filtered Search

**Files:**
- Modify: `crates/veda-index/src/hnsw/searcher.rs`
- Test: `crates/veda-index/tests/hnsw_filtered_test.rs`

Implement adaptive filtered search:
- Accept a filter bitmap (roaring bitmap of valid doc IDs)
- Strategy auto-selection based on filter selectivity:
  - `>50%` match → pre-filter then search
  - `1-50%` match → in-algorithm filter (check bitmap during traversal)
  - `<1%` match → post-filter with oversampling
- Tests: verify filtered recall vs brute-force filtered ground truth

---

## Phase 4: Query Engine

### Task 4.1: Filter Expression Parser

**Files:**
- Create: `crates/veda-query/src/filter/mod.rs`
- Create: `crates/veda-query/src/filter/parser.rs`
- Create: `crates/veda-query/src/filter/ast.rs`
- Create: `crates/veda-query/src/filter/evaluator.rs`
- Test: `crates/veda-query/tests/filter_test.rs`

Parse SQL-like filter expressions into an AST and evaluate against scalar indexes:

```
"category = 'electronics' AND price BETWEEN 100 AND 500 AND tags CONTAINS 'sale'"
```

Use a recursive descent parser or `nom` crate. Evaluate against inverted/range/bitmap indexes to produce a roaring bitmap of matching doc IDs.

---

### Task 4.2: Single-Vector Search Executor

**Files:**
- Create: `crates/veda-query/src/search.rs`
- Test: `crates/veda-query/tests/search_test.rs`

Execute a single-vector search across multiple segments:
1. Parse filter → evaluate to bitmap
2. Search growing segment (flat scan with filter)
3. Search each sealed segment (ANN index with filter bitmap)
4. Merge results across segments (top-K merge)

---

### Task 4.3: Multi-Vector Fusion Search

**Files:**
- Create: `crates/veda-query/src/fusion.rs`
- Test: `crates/veda-query/tests/fusion_test.rs`

Implement fusion strategies:
- **RRF**: `score(d) = sum(1 / (k + rank_i(d)))` across fields
- **Weighted Sum**: normalize scores per field, weighted combination
- **Intersection**: only results appearing in all field results

---

### Task 4.4: Hybrid Search (Dense + Sparse/BM25)

**Files:**
- Create: `crates/veda-query/src/hybrid.rs`
- Create: `crates/veda-index/src/bm25.rs`
- Test: `crates/veda-query/tests/hybrid_test.rs`

Implement BM25 scoring on inverted indexes + fusion with dense vector results via alpha parameter.

---

## Phase 5: Coordination Service

### Task 5.1: Raft Consensus — Leader Election & Log Replication

**Files:**
- Create: `crates/veda-coordinator/src/raft/mod.rs`
- Create: `crates/veda-coordinator/src/raft/state.rs`
- Create: `crates/veda-coordinator/src/raft/log.rs`
- Create: `crates/veda-coordinator/src/raft/rpc.rs`
- Test: `crates/veda-coordinator/tests/raft_test.rs`

Use the `openraft` crate (battle-tested Raft implementation in Rust, used by Databend) rather than implementing from scratch. Configure:
- Election timeout, heartbeat interval
- Log storage (append-only, snapshotting)
- State machine for cluster state changes

---

### Task 5.2: Cluster State & Schema Registry

**Files:**
- Create: `crates/veda-coordinator/src/cluster_state.rs`
- Create: `crates/veda-coordinator/src/schema_registry.rs`
- Test: `crates/veda-coordinator/tests/cluster_state_test.rs`

The Raft state machine manages:
- Collection schemas (create, drop, modify)
- Shard routing table (which shards live on which nodes)
- Node registry (join, leave, health status)
- All state changes go through Raft log for consistency

---

### Task 5.3: Shard Routing

**Files:**
- Create: `crates/veda-coordinator/src/shard_router.rs`
- Test: `crates/veda-coordinator/tests/shard_router_test.rs`

Given a collection and optional partition key:
- Determine which shards to query
- Determine which nodes host those shards
- Handle replica selection (round-robin or latency-based)

---

## Phase 6: Data Nodes

### Task 6.1: WAL — Write-Ahead Log via NATS JetStream

**Files:**
- Create: `crates/veda-storage/src/wal.rs`
- Test: `crates/veda-storage/tests/wal_test.rs`

Integrate with NATS JetStream:
- Each shard has a NATS stream for its WAL
- Inserts are published to the stream, consumed by data nodes
- On recovery: replay WAL from last checkpoint
- Use `async-nats` crate

---

### Task 6.2: Segment Lifecycle — Flush, Seal, Upload

**Files:**
- Create: `crates/veda-storage/src/segment_manager.rs`
- Test: `crates/veda-storage/tests/segment_manager_test.rs`

Manage the full segment lifecycle:
1. Growing segment accepts writes
2. When size threshold reached → seal
3. Notify Index Node to build ANN indexes
4. Upload sealed segment to object storage (MinIO/S3)
5. Update coordination service with new segment metadata

---

### Task 6.3: Compaction

**Files:**
- Create: `crates/veda-storage/src/compaction.rs`
- Test: `crates/veda-storage/tests/compaction_test.rs`

Background process:
- Merge small sealed segments into larger ones
- Apply delete bitmaps (physically remove deleted rows)
- Rebuild indexes on merged segment
- Upload merged segment, remove old segments

---

## Phase 7: Distributed Query

### Task 7.1: Query Node — Shard Fan-Out & Merge

**Files:**
- Create: `crates/veda-query/src/distributed.rs`
- Test: `crates/veda-query/tests/distributed_test.rs`

The query coordinator:
1. Receives search request
2. Resolves shard routing (from coordination service)
3. Fans out search to relevant query node shards (parallel)
4. Each shard searches its segments (growing + sealed)
5. Merges results across shards (global top-K)
6. Applies fusion if multi-vector
7. Fetches full records if requested

---

### Task 7.2: Inter-Node Communication — gRPC Internal API

**Files:**
- Create: `proto/internal.proto`
- Create: `crates/veda-core/src/internal_rpc.rs`

Internal gRPC service for node-to-node communication:
- `SearchShard` — query node asks another node to search a shard
- `FetchRecords` — query node asks data node for full records
- `BuildIndex` — coordination asks index node to build
- `ReplicateShard` — data node replicates to another data node

Use `tonic` crate with `prost` for code generation.

---

## Phase 8: Gateway

### Task 8.1: REST API — axum Server

**Files:**
- Create: `crates/veda-gateway/src/rest/mod.rs`
- Create: `crates/veda-gateway/src/rest/collections.rs`
- Create: `crates/veda-gateway/src/rest/search.rs`
- Create: `crates/veda-gateway/src/rest/records.rs`
- Create: `crates/veda-gateway/src/rest/admin.rs`
- Test: `crates/veda-gateway/tests/rest_test.rs`

Implement all REST endpoints from the design doc using `axum`:
- Collection CRUD
- Search (all 6 query types)
- Record insert/upsert/delete/get
- Admin (health, nodes, backup)
- Proper error responses with codes and suggestions

---

### Task 8.2: gRPC API — tonic Server

**Files:**
- Create: `proto/veda.proto`
- Create: `crates/veda-gateway/src/grpc.rs`
- Test: `crates/veda-gateway/tests/grpc_test.rs`

Mirror REST endpoints as gRPC service methods. Share handler logic with REST layer.

---

### Task 8.3: SSE Streaming Endpoints

**Files:**
- Create: `crates/veda-gateway/src/sse.rs`
- Test: `crates/veda-gateway/tests/sse_test.rs`

Implement SSE endpoints using `axum`'s SSE support:
- Embedding progress streaming
- Index build progress streaming
- Streaming search results
- System events (cluster health changes)

---

## Phase 9: Additional Indexes

### Task 9.1: DiskANN (Vamana Graph)

**Files:**
- Create: `crates/veda-index/src/diskann/mod.rs`
- Create: `crates/veda-index/src/diskann/graph.rs`
- Create: `crates/veda-index/src/diskann/builder.rs`
- Create: `crates/veda-index/src/diskann/searcher.rs`
- Test: `crates/veda-index/tests/diskann_test.rs`

Implement Vamana graph algorithm for SSD-resident search:
- Build graph with R (max degree), L (search list size), alpha (pruning parameter)
- Disk-optimized: graph fits on SSD, vectors loaded on demand
- Implement beam search with SSD-aware prefetching

---

### Task 9.2: IVF-PQ Index

**Files:**
- Create: `crates/veda-index/src/ivfpq/mod.rs`
- Test: `crates/veda-index/tests/ivfpq_test.rs`

Implement:
- K-means clustering for IVF partitioning
- Product Quantization for vector compression
- Asymmetric Distance Computation (ADC) for search
- Support `nlist` (number of clusters) and `nprobe` (clusters to search)

---

### Task 9.3: RaBitQ Quantization

**Files:**
- Create: `crates/veda-index/src/quantization/mod.rs`
- Create: `crates/veda-index/src/quantization/rabitq.rs`
- Create: `crates/veda-index/src/quantization/scalar.rs`
- Test: `crates/veda-index/tests/rabitq_test.rs`

Implement RaBitQ (SIGMOD 2024):
- Quantize D-dimensional vectors into D-bit strings
- Bitwise distance computation (XOR + popcount)
- Corrective factors for unbiased estimation
- Support 1-bit through 7-bit quantization
- Auto-select bit-width by dimension

---

### Task 9.4: Auto Index Selection

**Files:**
- Create: `crates/veda-index/src/auto_select.rs`
- Test: `crates/veda-index/tests/auto_select_test.rs`

Implement the auto-selection logic:
- `<10K` → Flat
- `10K-10M` → HNSW
- `10M-100M` → HNSW + RaBitQ
- `100M-1B` → DiskANN
- `>1B` → IVF-PQ

---

## Phase 10: Blob Storage

### Task 10.1: MinIO/S3 Client Integration

**Files:**
- Create: `crates/veda-storage/src/object_store.rs`
- Test: `crates/veda-storage/tests/object_store_test.rs`

Use `object_store` crate (Apache Arrow project) for S3-compatible storage:
- `put_blob`, `get_blob`, `delete_blob`, `list_blobs`
- Support: MinIO, S3, GCS, Azure, local filesystem
- Configurable via `veda.toml` `[storage]` section

---

### Task 10.2: Transparent Blob Proxy

**Files:**
- Modify: `crates/veda-gateway/src/rest/records.rs`
- Create: `crates/veda-storage/src/blob_manager.rs`
- Test: `crates/veda-storage/tests/blob_manager_test.rs`

On insert with raw data:
1. Extract blob fields from request
2. Upload to object storage → get BlobRef
3. Replace blob data with BlobRef in record
4. Insert record with BlobRef

On retrieval with `include_blobs=true`:
1. Fetch record with BlobRef
2. Resolve BlobRef → fetch from object storage
3. Return blob data inline

---

## Phase 11: Auto-Embedding Service

### Task 11.1: Embedding Model Registry & Modality Validation

**Files:**
- Create: `crates/veda-core/src/embedding.rs`
- Test: `crates/veda-core/tests/embedding_test.rs`

Define model registry from config:
- Parse `[embedding.models.*]` from `veda.toml`
- Validate modality support before sending data
- Return proper `MODALITY_NOT_SUPPORTED` errors with suggestions

---

### Task 11.2: Embed-Insert Pipeline

**Files:**
- Create: `crates/veda-gateway/src/rest/embed_insert.rs`
- Test: `crates/veda-gateway/tests/embed_insert_test.rs`

`POST /v1/collections/{name}/embed-insert`:
1. Parse raw data from request
2. For each field mapped to an embedding model:
   - Validate modality support → error if unsupported
   - Send to embedding endpoint (OpenAI-compatible API)
   - Receive vector, attach to record
3. Insert record with computed vectors
4. Stream progress via SSE for batch inserts

---

## Phase 12: Pattern Search

### Task 12.1: Suffix Array Construction

**Files:**
- Create: `crates/veda-index/src/suffix_array/mod.rs`
- Create: `crates/veda-index/src/suffix_array/builder.rs`
- Test: `crates/veda-index/tests/suffix_array_test.rs`

Build suffix array over concatenated text content at segment seal time:
- Use SA-IS algorithm (linear time construction)
- Disk-aware design for large corpora
- Support exact pattern search via binary search on suffix array

---

### Task 12.2: Fuzzy Pattern Matching

**Files:**
- Create: `crates/veda-index/src/suffix_array/fuzzy.rs`
- Test: `crates/veda-index/tests/suffix_array_fuzzy_test.rs`

Extend suffix array with SoftMatcha-style fuzzy matching:
- Support substitution, insertion, deletion
- Word-order preservation
- Dynamic corpus-aware pruning to prevent exponential blowup
- Configurable `max_edits` parameter

---

### Task 12.3: Pattern Search Query Integration

**Files:**
- Modify: `crates/veda-query/src/search.rs`
- Modify: `crates/veda-gateway/src/rest/search.rs`
- Test: `crates/veda-query/tests/pattern_search_test.rs`

Wire pattern search into the query engine:
- New query type `pattern` in REST/gRPC API
- Execute across segments, merge results
- Can be combined with filters

---

## Phase 13: Auth, Security, Observability

### Task 13.1: JWT Authentication

**Files:**
- Create: `crates/veda-gateway/src/auth.rs`
- Test: `crates/veda-gateway/tests/auth_test.rs`

JWT-based auth for REST/SSE (Niriksha), API keys for SDK/REST, mTLS option for gRPC.

---

### Task 13.2: RBAC Authorization

**Files:**
- Create: `crates/veda-gateway/src/rbac.rs`
- Test: `crates/veda-gateway/tests/rbac_test.rs`

Roles: admin, editor, viewer. Per-collection permissions.

---

### Task 13.3: Prometheus Metrics & OpenTelemetry

**Files:**
- Create: `crates/veda-core/src/metrics.rs`
- Create: `crates/veda-core/src/tracing_setup.rs`

Use `metrics` + `metrics-exporter-prometheus` crates. Instrument all hot paths. OpenTelemetry via `tracing-opentelemetry`.

---

## Phase 14: Niriksha — Core Dashboard

### Task 14.1: Scaffold Niriksha (React + TypeScript + Vite)

**Files:**
- Create: `niriksha/package.json`
- Create: `niriksha/vite.config.ts`
- Create: `niriksha/tsconfig.json`
- Create: `niriksha/src/App.tsx`
- Create: `niriksha/src/main.tsx`
- Create: `niriksha/src/api/client.ts`
- Create: `niriksha/src/stores/authStore.ts`

Initialize with: React 19, TypeScript, Vite, Zustand, Tailwind CSS, Framer Motion, Axios.

Design system: dark/light mode via CSS variables, inspired by data-viewer aesthetic.

---

### Task 14.2: Dashboard Home Page

**Files:**
- Create: `niriksha/src/pages/DashboardHome.tsx`
- Create: `niriksha/src/components/ClusterOverview.tsx`
- Create: `niriksha/src/components/MetricsCards.tsx`
- Create: `niriksha/src/components/ActivityFeed.tsx`

Cluster overview: collection count, vector count, node count, uptime, latency, QPS, storage usage. Real-time updates via SSE.

---

### Task 14.3: Collection Browser

**Files:**
- Create: `niriksha/src/pages/CollectionBrowser.tsx`
- Create: `niriksha/src/components/CollectionList.tsx`
- Create: `niriksha/src/components/RecordTable.tsx`
- Create: `niriksha/src/components/RecordDetail.tsx`
- Create: `niriksha/src/components/SchemaEditor.tsx`
- Create: `niriksha/src/stores/collectionStore.ts`

Browse collections, paginated record table with inline media rendering (image thumbnails, audio waveform, video player), record detail modal with resizable split pane, schema editor.

---

### Task 14.4: Search Playground

**Files:**
- Create: `niriksha/src/pages/SearchPlayground.tsx`
- Create: `niriksha/src/components/QueryBuilder.tsx`
- Create: `niriksha/src/components/SearchResults.tsx`
- Create: `niriksha/src/components/ResultCard.tsx`
- Create: `niriksha/src/stores/searchStore.ts`

Multi-mode search: single vector, multi-vector, hybrid, pattern search. Multimodal query input (text, image upload, audio record/upload). Result rendering by modality with similarity score badges. "Find Similar" button, query history.

---

### Task 14.5: Embedding Pipeline Monitor

**Files:**
- Create: `niriksha/src/pages/EmbeddingMonitor.tsx`
- Create: `niriksha/src/components/PipelineProgress.tsx`
- Create: `niriksha/src/components/ErrorDrilldown.tsx`

SSE-streamed progress bars, per-item status, error drill-down.

---

### Task 14.6: Index Manager

**Files:**
- Create: `niriksha/src/pages/IndexManager.tsx`
- Create: `niriksha/src/components/IndexCard.tsx`

Visualize indexes per collection: type, metrics, quantization, memory/disk, recall, latency. Rebuild, tune parameters.

---

### Task 14.7: System Health

**Files:**
- Create: `niriksha/src/pages/SystemHealth.tsx`
- Create: `niriksha/src/components/NodeCard.tsx`
- Create: `niriksha/src/components/ShardMap.tsx`

Live cluster monitoring via SSE. Per-node resource usage, shard distribution, replication status.

---

### Task 14.8: Auth — Login & RBAC

**Files:**
- Create: `niriksha/src/pages/LoginPage.tsx`
- Modify: `niriksha/src/api/client.ts`
- Create: `niriksha/src/components/ProtectedRoute.tsx`

JWT login, role-based UI (hide admin features for non-admins).

---

### Task 14.9: Serve Niriksha from Gateway

**Files:**
- Modify: `crates/veda-gateway/src/rest/mod.rs`

Mount `niriksha/dist/` as static files from the gateway. Serve `index.html` as SPA fallback. Available at `http://localhost:6340/niriksha/`.

---

## Phase 15: Niriksha — Vector Atlas & Evaluation Engine

### Task 15.1: Vector Atlas — UMAP/t-SNE Projection

**Files:**
- Create: `crates/veda-query/src/projection.rs` (server-side UMAP/t-SNE)
- Create: `niriksha/src/pages/VectorAtlas.tsx`
- Create: `niriksha/src/components/ScatterPlot.tsx`
- Create: `niriksha/src/components/AtlasControls.tsx`

Server computes dimensionality reduction (use `linfa` crate for UMAP/t-SNE in Rust), streams iterations via SSE. Frontend renders with deck.gl for GPU-accelerated 50K+ point scatter plots. Color by field, hover preview, lasso select, click to inspect.

---

### Task 15.2: Evaluation Engine — Data Quality Scoring

**Files:**
- Create: `crates/veda-query/src/evaluation/mod.rs`
- Create: `crates/veda-query/src/evaluation/quality.rs`
- Create: `crates/veda-query/src/evaluation/llm_judge.rs`
- Create: `niriksha/src/pages/EvaluationView.tsx`
- Create: `niriksha/src/components/EvaluationTable.tsx`
- Create: `niriksha/src/components/AIReview.tsx`

Two layers:
1. **Local quality scores** (no LLM): blur detection, SNR, length, duplicates, entropy
2. **LLM-as-a-Judge** (streamed via SSE): RAG usability, FAQ usability, cross-modal alignment

Collection-level AI Review: analyze entire collection, suggest chunking strategies, embedding models, flag data quality issues.

---

### Task 15.3: User Feedback System

**Files:**
- Create: `niriksha/src/components/FeedbackPicker.tsx`
- Create: `niriksha/src/stores/feedbackStore.ts`

Emoji feedback + comments per record, stored in Veda. Same pattern as data-viewer.

---

## Phase 16: Python SDK

### Task 16.1: PyO3 Bindings — veda-client

**Files:**
- Create: `crates/veda-python/Cargo.toml`
- Create: `crates/veda-python/src/lib.rs`
- Create: `crates/veda-python/python/veda/__init__.py`
- Create: `crates/veda-python/python/veda/client.py`
- Create: `crates/veda-python/pyproject.toml`
- Test: `crates/veda-python/tests/test_client.py`

Python SDK wrapping REST/gRPC client:

```python
from veda import VedaClient

client = VedaClient("http://localhost:6340")
client.create_collection("products", schema={...})
client.insert("products", records=[...])
results = client.search("products", vector=[...], limit=10)
```

Build with Maturin. Publish to PyPI as `veda-client`.

---

## Phase 17: Deployment

### Task 17.1: Dockerfile

**Files:**
- Create: `Dockerfile`
- Create: `.dockerignore`

Multi-stage build: Rust builder → minimal runtime image with Niriksha static files.

---

### Task 17.2: Docker Compose

**Files:**
- Create: `docker-compose.yml`
- Create: `docker-compose.dev.yml`

Production compose with MinIO, NATS, coordinator, query nodes, data nodes, index nodes. Dev compose with `--standalone` mode.

---

### Task 17.3: Helm Chart

**Files:**
- Create: `deploy/helm/veda/Chart.yaml`
- Create: `deploy/helm/veda/values.yaml`
- Create: `deploy/helm/veda/templates/` (all K8s manifests)

Kubernetes deployment with MinIO and NATS as subcharts.

---

### Task 17.4: Configuration & Install Script

**Files:**
- Create: `scripts/install.sh`
- Create: `config/veda.toml.example`

`curl -sSL https://get.veda.dev | sh` installer. Example config file.

---

## Phase 18: Benchmarks & Hardening

### Task 18.1: ANN Benchmark Suite

**Files:**
- Create: `benchmarks/ann_benchmark.rs`

Test against standard datasets (SIFT1M, GloVe, Deep1M):
- Measure recall@10, recall@100 at various QPS levels
- Compare HNSW, DiskANN, IVF-PQ, Flat
- Compare quantization impact (none, SQ8, RaBitQ, PQ)

---

### Task 18.2: End-to-End Integration Tests

**Files:**
- Create: `tests/integration/`

Full pipeline tests:
- Start cluster (docker compose)
- Create collection, insert data, build index, search, verify results
- Test multi-node scenarios (node failure, recovery, rebalancing)
- Test blob upload/retrieval
- Test auto-embedding pipeline

---

### Task 18.3: Stress Testing

**Files:**
- Create: `benchmarks/stress_test.rs`

Concurrent read/write workloads:
- 100 concurrent writers + 100 concurrent readers
- Measure latency percentiles (p50, p95, p99)
- Test with 10M, 100M, 1B vectors
- Memory and disk usage monitoring

---

## Dependency Graph

```
Phase 1 (scaffolding + types)
  ├── Phase 2 (storage)
  │     ├── Phase 3 (indexes)
  │     │     └── Phase 4 (query engine)
  │     │           └── Phase 7 (distributed query)
  │     │                 └── Phase 8 (gateway)
  │     │                       ├── Phase 10 (blobs)
  │     │                       ├── Phase 11 (auto-embedding)
  │     │                       └── Phase 13 (auth/metrics)
  │     └── Phase 6 (data nodes)
  │           └── Phase 7
  ├── Phase 5 (coordination)
  │     └── Phase 7
  ├── Phase 9 (additional indexes) — can parallel with Phase 5-7
  ├── Phase 12 (pattern search) — can parallel with Phase 5-7
  └── Phase 14-15 (Niriksha) — after Phase 8
        └── Phase 16 (Python SDK) — after Phase 8
              └── Phase 17 (deployment) — after all
                    └── Phase 18 (benchmarks) — after all
```

Phases 9, 12 can be developed in parallel with Phases 5-7 since they only depend on the index trait from Phase 3.

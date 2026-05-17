# GPU Index Building via NVIDIA cuVS

Plan for adding optional GPU acceleration to Veda's ANN index construction.

## Why

CPU HNSW build is the bottleneck for ingest-heavy workloads. Published
benchmarks (May 2025 - April 2026) show:

- NVIDIA cuVS CAGRA: 10-50× faster index build than CPU HNSW on
  equivalent hardware.
- Milvus + cuVS (Mar 2026): 3× total ingest throughput.
- Elasticsearch 9.3 GPU vector indexing (Feb 2026 preview, GA April 9.4):
  12× indexing throughput, 7× faster force-merge.
- OpenSearch 3.5+ cuVS (Q1 2026): 14× faster, 4× cheaper at scale.

This is the single biggest gap in Veda's competitive matrix as of May
2026 — competitors have it; we don't.

## Status

Currently a placeholder: a `gpu` cargo feature flag exists on
`veda-index`, and `crates/veda-index/src/gpu.rs` reserves the module path.
No `cuvs-rs` dependency is declared yet — adding it requires the CUDA
toolkit at build time and would break CPU-only builds without further
conditional compilation work.

## Phased plan

### Phase 1 — Build-system plumbing

- Add `cuvs-rs` as an optional dependency: `cuvs = { version = "...",
  optional = true }`, behind the existing `gpu` feature flag.
- CI: split jobs into `cpu` (default) and `gpu` (NVIDIA runner).
- Document the build requirement (CUDA 12.x + NVIDIA driver ≥ 535).

### Phase 2 — CAGRA index wrapper

CAGRA is cuVS's GPU-native graph index. It's similar to HNSW but
GPU-optimised; recall@10 is comparable for typical embedding dims.

```rust
pub struct CagraIndex {
    inner: cuvs::CagraIndex,
    dimension: u32,
    metric: DistanceMetric,
}

impl VectorIndex for CagraIndex {
    fn build(&mut self, vectors: &[&[f32]]) -> Result<()> {
        // Upload to GPU, build, keep handle.
    }
    fn search(&self, q: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        // Run on GPU if loaded, else copy index back to CPU (heuristic TBD).
    }
}
```

Implementation work: ~2-3 days, mostly cuvs-rs FFI plumbing + error
mapping.

### Phase 3 — Hybrid CPU/GPU build path

Most production deployments don't have a GPU per node. The right pattern
is: build on GPU when available, persist to disk in CPU-friendly format,
serve queries from CPU. This means:

- CAGRA build → CAGRA serialise → reload as HNSW-compatible format.
- cuVS has a `to_hnsw()` exporter that does this.

Implementation work: ~1 day on top of Phase 2.

### Phase 4 — Selection policy

When should we use GPU vs CPU? Heuristics:

- Segment ≥ 100K rows AND dim ≥ 256 AND a GPU is present → CAGRA.
- Otherwise → existing CPU HNSW.

Surface this as a per-collection config: `index_build_target:
"cpu" | "gpu" | "auto"`. Default `auto`.

### Phase 5 — Query path (deferred)

For most single-node deployments, GPU query is not necessary; CPU search
on a well-built HNSW handles ~1ms latency comfortably. GPU query is
worth it only at >1M vectors and high QPS. Defer until users ask.

## Alternative: optional NVIDIA NIM endpoint

A cheaper alternative for users who don't want CUDA in the Veda binary:
point at an NVIDIA NIM container that hosts the GPU-accelerated index
service. This is the "managed" version — same speedup, no build
complexity. Implementation is just an HTTP client to NIM's API.

This is probably the right first step for Veda: NIM client first, then
in-binary cuVS later.

## What this is *not*

- Not a query-path speedup. Veda's HNSW search is already ~1ms; the cost
  there is filter evaluation, not vector math. GPU there is a marginal
  win at extreme scale only.
- Not a substitute for cuVS proper if you're running >100M vectors
  single-node. At that scale, run Milvus / OpenSearch / Qdrant which
  already have mature GPU paths.
- Not free. CUDA toolkit, driver matching, and a GPU at runtime are real
  ops cost. Veda's value prop is "easy to operate"; adding GPU as a hard
  requirement would undermine that. Keep it firmly opt-in.

## Estimated effort

| Phase | Effort |
|-------|--------|
| 1 (build plumbing) | 0.5 day |
| 2 (CAGRA wrapper) | 2-3 days |
| 3 (CPU export) | 1 day |
| 4 (policy) | 0.5 day |
| 5 (query path) | Deferred |
| NIM-client alternative | 0.5 day |

**Recommended first step**: ship the NIM client (4-6 hours of focused
work). Then revisit in-binary cuVS only if customers specifically need it.

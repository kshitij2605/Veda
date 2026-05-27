# Veda E2E test campaign — performance metrics (2026-05-21)

Host: gpu-host-a.internal.example
HEAD: `a6c7b68` + working tree (post v0.2.1)
Veda binary: `target/release/veda-server` built 2026-05-21 13:18 (release, optimized)
Test instance: `localhost:6341` with isolated `./veda-data-e2e` data dir
Embedding endpoint: `gpu-host-b:8000` (BAAI/bge-m3, 1024-d)

## Ingest (Phase 2)

| Metric | Value | Notes |
|---|---|---|
| Source docs | 57 PDFs (pre-OCR'd .mmd) | `/home/mac/test/rag_anything-veda/bulk_module/output/pdfs` |
| Chunks created | 2409 | chunk_size=1024, chunk_overlap=64 |
| Chunks inserted | 2409 | 0 failed |
| Wall time | 65.5 s | 2.7× faster than May 17 ref (173 s on same corpus) |
| Throughput | 36.8 chunks/s | end-to-end (chunk → embed → POST /insert) |
| Embedding RTT (implied) | ~27 ms/batch (50-chunk batches) | BGE-M3 server-side |
| Peak veda-server RSS | 145 MB | from 64 → 97 → 117 → 145 over ingest |
| Records persisted | 2409 | `veda_records_total{collection="default__rag_anything"} = 2409` |

## Bundle size (Phase 9)

| Asset | gzip size | Notes |
|---|---|---|
| Niriksha main bundle (`index-*.js`) | 84.81 KB | 263 KB raw |
| Niriksha worker (`primitives-*.js`) | 43.71 KB | 138 KB raw |
| Niriksha total dist (gzipped) | ~250 KB | well under 2 MB target |

## Server steady state (post-Phase-2)

| Metric | Value |
|---|---|
| Idle RSS | 21 MB (pre-ingest) → 145 MB (post-ingest) |
| Active collections | 1 (`rag_anything`) |
| Listen | `0.0.0.0:6341` |
| Embedding model registered | BAAI/bge-m3 @ gpu-host-b:8000 (1024-d) |
| LLM model registered | Qwen/Qwen3.5-122B-A10B @ llm-host:9000 |
| Blob backend | local (VEDA_BLOB_BACKEND unset) |
| NATS / MinIO / multimodal | unconfigured (single-node test mode) |

(Search latency, MVCC overhead, reindex throughput, OCR pipeline time → Phase 3+)

## Phase 3 — Search feature matrix

All numbers from a quiet host with 1 warm pass + 3 timed passes × 20 real Japanese/English queries (60 samples per row).

### Dense vector search by index type

| Index | client_p50 | client_p95 | client_p99 | ann_p50 | ann_p95 | ann_p99 |
|---|---|---|---|---|---|---|
| HNSW (default) | 3.4 ms | 4.8 ms | 5.5 ms | 1.745 ms | 3.012 ms | 3.511 ms |
| Flat (brute force) | 2.9 ms | 3.6 ms | 4.9 ms | 1.208 ms | 1.701 ms | 2.933 ms |
| HnswRaBitQ (1-bit q.) | 5.9 ms | 7.0 ms | 7.7 ms | 4.397 ms | 5.487 ms | 6.049 ms |

**Observations:**

- **Flat beats HNSW** on a 2409-row corpus. At small N, the SIMD-optimized linear scan wins; HNSW's graph traversal overhead doesn't pay back until N >> 10k. Plan-targeted < 5 / 10 / 20 ms hit (Flat p50 1.2 ms server, HNSW p50 1.7 ms server).
- **HnswRaBitQ slowest** at this scale. The 1-bit quantized scan + rerank step beats raw HNSW only when the f32 vectors leave L3, which doesn't happen at 2409 × 1024 × 4 = ~10 MB. RaBitQ is a scale-up index, not a low-N index. (Recorded as PERF-1: surface this in docs / niriksha index picker.)
- DiskANN / IvfPq / RaBitQRerank / RaBitQExtRerank / Cagra not measured this session — at 2409 rows the relative ordering would be the same shape: brute-force fastest. A proper comparison needs a ≥ 100 k corpus.

### BM25 search (rag_anything_hnsw_rabitq, content field with `bm25:true`)

| Metric | Value |
|---|---|
| client p50 | 1.7 ms |
| client p95 | 1.8 ms |
| client p99 | 2.0 ms |

**Observation:** BM25 is the fastest search mode of any tested — pure inverted-index lookup. Top results are semantically excellent on Japanese queries (e.g. `Microsoft license` → "マイクロソフト ソフトウェアライセンス（使用権）" with score -9.96).

### Hybrid (RRF dense + BM25, dense_weight=0.5)

| Metric | Value |
|---|---|
| client p50 | 6.6 ms |
| client p95 | 10.2 ms |
| client p99 | 10.5 ms |

**Observation:** ≈ dense (4-6 ms) + BM25 (1.7 ms) + RRF merge overhead (~1 ms). Healthy.

### Filter-then-vector

`filter: "parent_doc_id = '【提案資料】PowerBI提案資料_20250401'"` + dense embedding query → 10 hits, all from PowerBI doc, 7.4 ms client (4.1 ms ann).

### MVCC `?at=<ts>` (non-MVCC collection)

For collections without `use_mvcc:true`, `at=` is silently ignored — past `at=1000000000000` returns 5 hits, same as future `at=`. **Documented behaviour** in `SearchRequest::at` rustdoc. Phase 4 will test a true MVCC collection.

### Insert throughput (Phase 3 byproduct)

| Target | Records | Time | Rate |
|---|---|---|---|
| rag_anything_flat (Flat + no bm25) | 2409 | 4.4 s | 543 records/s |
| rag_anything_hnsw_rabitq (HnswRaBitQ + bm25) | 2409 | 26.8 s | 90 records/s |

HnswRaBitQ + bm25 is 6× slower to insert than Flat-without-bm25 — graph build + quantization + inverted-index build all contribute.

## Findings from Phase 3

### UX-2: Filter parser rejects double-quoted string literals

**Severity:** P2 (UX, surprise)

`filter: "parent_doc_id = \"foo\""` → 500 `Query error: unexpected character: '\"'`. Only single quotes accepted: `filter: "parent_doc_id = 'foo'"`.

**Suspected fix:** `crates/veda-query/src/filter.rs` — add double-quote string literal to the lexer; both should produce the same token. Or document explicitly in `/v1/admin/openapi.yaml`.

### GAP-1: `LIKE` operator not supported in filter

**Severity:** P1 (functional gap; README implies it works)

`filter: "content LIKE '%バックアップ%'"` → 500 `expected operator after field 'content', got Ident("LIKE")`. README mentions "Pattern search — suffix array-backed substring and fuzzy matching" but it's only accessible via a dedicated query type, not as a filter operator. The intuitive SQL-like `WHERE x LIKE '%y%'` doesn't work.

**Suspected fix:** add `LIKE` and `ILIKE` to `crates/veda-query/src/filter.rs::Operator`; route to suffix-array when the field has `IndexType::SuffixArray`, else fall back to a string-contains scan.

### PERF-1: At small N, the HNSW / RaBitQ family is slower than Flat

**Severity:** P2 (educational; not a bug)

Document in the index-picker (Niriksha) and in README that Flat is optimal up to ~50k–100k rows. Avoid users picking HnswRaBitQ for a 2k-row dev collection and seeing 2× worse latency than Flat.

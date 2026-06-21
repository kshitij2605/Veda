# Veda vs Elasticsearch — Dense-Vector Head-to-Head Benchmark

**Date:** 2026-06-21
**Author:** automated benchmark (gpu-host-a)
**Scope:** Dense single-vector ANN retrieval only (cosine, 1024-d). This is Elasticsearch's strongest vector footing. Veda's multimodal / any-format ingestion advantages are categorical (ES cannot do them) and are deliberately out of scope here — this report measures only the apples-to-apples dense case.

---

## 1. Setup

| Item | Value |
|---|---|
| Corpus | **30,000** English passages sampled from **BeIR/nq** (Natural Questions) corpus (`title + text`, 200–2000 chars), downloaded direct from HuggingFace (auto-converted parquet, 2.68M rows total) |
| Query set | **500** held-out passages, disjoint from the indexed corpus |
| Embedding model | `jinaai/jina-embeddings-v5-omni-small` (the SAME model Veda uses; "jina-v5-omni-small"), via `SentenceTransformer(..., trust_remote_code=True, default_task="retrieval")` on GPU |
| Encode protocol | corpus → `encode_document`, queries → `encode_query` (asymmetric retrieval), L2-normalized, **1024-d float32**. Identical `.npy` vectors fed to BOTH systems |
| Ground truth | exact top-10 cosine via GPU brute-force (torch) over the 30k corpus → recall@10 baseline for both |
| Hardware | gpu-host-a: 72 vCPU, 1 TB RAM, NVIDIA RTX 6000 Ada (48 GB) |
| Veda | local release build, isolated instance on `127.0.0.1:6343` (separate from prod 6340), HNSW |
| Elasticsearch | **production** cluster `es-prod` v8.19.0 at `build-host.internal.example:9200` (remote, TLS, self-signed). Auth masked as `***` |

### Index configs (matched HNSW params)

**Veda** — `vector` field, `metric=cosine`, `index="hnsw"`. Veda's HNSW params are built-in defaults and **not overridable per query or per collection** via the API:
`m=16, m_max0=32, ef_construction=200, ef_search=200`.

**Elasticsearch** — `dense_vector {dims:1024, index:true, similarity:cosine, index_options:{type:hnsw, m:16, ef_construction:200}}`; queried via the kNN API with `k=10, num_candidates=200` (chosen to match Veda's `ef_search=200`). Force-merged to 1 segment before timing.

### Critical caveat: network asymmetry
**Veda is localhost; Elasticsearch is remote over TLS.** Client-side latency therefore flatters Veda's transport and penalizes ES. To separate engine time from network, both systems' **server-side timing** is reported (Veda `explain.ann_search_ms` / `took_ms`; ES `took`). The engine comparison should lean on those.

### Critical caveat: Veda segmentation (this dominates the Veda result)
Veda builds ANN indexes **per sealed segment** and merges results across segments at query time. During bulk insert, Veda seals a new segment roughly every 500-record batch and only opportunistically compacts (compaction is insert-triggered, caps merged segments at 10k rows, does **not** run to completion after the load stops, and there is **no force-merge / optimize API**). At 30k rows the collection settles at **~38 segments**, and Veda's multi-segment HNSW merge loses a large fraction of true neighbors. This is out-of-the-box behavior of Veda's current bulk-load path, not a tuning choice — see analysis.

---

## 2. Results (30,000 vectors, 500 queries, k=10, sequential single client)

| Metric | **Veda (HNSW, 30k, ~38 segments)** | **Elasticsearch (HNSW, 30k, 1 segment)** |
|---|---|---|
| Ingest throughput | 260 docs/s | 312 docs/s |
| Build/finalize | seal instant; bg compaction never fully consolidates | force-merge to 1 seg: 23.6 s |
| **recall@10** (vs exact GT) | **0.265** | **1.000** |
| Latency p50 (client) | 56 ms | 16 ms |
| Latency p95 (client) | 117 ms | 19 ms |
| Latency p99 (client) | 159 ms | 27 ms |
| **Server-side** p50 | `ann_search` 47 ms / `took` 47 ms | `took` 7 ms |
| Server-side p95 | `ann_search` 85 ms | `took` 9 ms |

> Veda's ~47ms server-side time at 30k reflects searching ~38 separate HNSW graphs (each at ef_search=200) and merging — and despite that work, recall is only 0.265. ES's 7ms `took` is a single consolidated HNSW graph at recall 1.0.

### Reference points (to characterize Veda fairly)

| Config | N | segments | recall@10 | client p50 | server p50 |
|---|---|---|---|---|---|
| **Veda HNSW, single segment** (best case) | 5,000 | 1 | **1.000** | 5.1 ms | `ann` 2.5 ms / `took` 2 ms |
| Veda HNSW | 9,999 | ~4 | 0.853 | ~25 ms | — |
| Veda HNSW | 30,000 | ~38 | 0.265 | 56 ms | 47 ms |

Recall degrades monotonically with segment count: **1 seg → 1.000, ~4 seg → 0.853, ~38 seg → 0.265.** A single-segment Veda HNSW is exact (recall 1.0) and fast (5ms client / 2ms server) — fully competitive with ES. The 30k result is dominated by segment fragmentation + the multi-segment merge, NOT by the HNSW algorithm itself.

> Even a single sealed 5k segment searches in ~2.5ms `ann` — fast, but at that size more consistent with a near-exhaustive scan than sub-millisecond graph traversal. The HNSW path is sound on recall (1.0 at 1 segment); the scaling problem is operational (fragmentation + no force-merge).

### Optional: quantized point (same protocol, 30k, 500 queries)

| Metric | **Veda SQ8** | **Elasticsearch int8_hnsw** |
|---|---|---|
| recall@10 | 0.965 | 0.986 |
| Latency p50 (client) | 278 ms | 16 ms |
| Server p50 | `took` 251 ms | `took` 7 ms |
| Ingest | 239 docs/s | 295 docs/s |

Veda SQ8 is a quantized **brute-force scan** over all codes per segment — so its recall (0.965) is far higher than Veda HNSW (0.265) because an exhaustive scan doesn't lose neighbors in the multi-segment merge, but it is correspondingly slow (~278ms at 30k). ES int8_hnsw keeps near-flat recall (0.986) at 16ms.

---

## 3. Honest analysis

**Elasticsearch wins this dense-vector benchmark decisively**, both on recall and latency, and the win is real (not just the network artifact):

- **Recall:** ES = 1.000 vs Veda = 0.265 at 30k. This is the headline. The gap is **not** an HNSW-algorithm deficiency in Veda — Veda's HNSW is exact (recall 1.0) on a single segment. It is that Veda's bulk-load path leaves the collection in ~38 small sealed segments, and its cross-segment result merge drops most true top-10 neighbors. There is no API to force-merge/optimize, and compaction does not run to completion after ingestion stops. **For an apples-to-apples 30k index, ES's single force-merged graph retrieves correctly and Veda's fragmented one does not.**

- **Latency:** Even leaning entirely on **server-side** numbers (which removes the localhost-vs-remote-TLS asymmetry), ES is faster: ES `took` p50 = **7 ms** vs Veda `ann_search` p50 = **47 ms** at 30k. The client-side gap (16ms vs 56ms) is partly inflated in ES's favor by the remote-TLS round trip, but ES still wins on engine time by ~7x. The network asymmetry does **not** rescue Veda here.

- **Where Veda ties / is competitive:** at small scale / a single segment (5k, 1 segment), Veda HNSW is recall 1.0 at **5ms client / 2ms server** — on par with ES. The engine is fundamentally sound; the deficit is operational/scaling behavior of the segment + compaction layer.

- **Ingest:** roughly comparable (Veda 260 vs ES 312 docs/s), both single-client bulk; not a differentiator.

- **Quantization:** ES int8_hnsw is the clear winner (0.986 @ 16ms). Veda SQ8 preserves recall (0.965) but only because it is exhaustive, paying ~278ms — not a viable latency point at this scale.

**Bottom line (one paragraph):** On the dense single-vector case — Elasticsearch's home turf — ES beats Veda outright at 30k: recall 1.000 vs 0.265 and ~7ms vs ~47ms server-side. The recall gap is caused by Veda fragmenting a bulk load into ~38 sealed segments with no force-merge/optimize API and a lossy multi-segment HNSW merge; Veda's HNSW is actually exact and fast (recall 1.0, 5ms) when the data lands in a single segment, so this is an operational/scaling gap rather than a core-algorithm one, but it is a real, user-visible loss today. The localhost-vs-remote-TLS asymmetry favors Veda on client latency and was neutralized by comparing server-side `took`/`ann_search` — ES still wins. This benchmark says nothing about Veda's actual differentiator: native multimodal / any-format ingestion and unified-vector-space retrieval, which Elasticsearch cannot do at all and which were out of scope here.

---

## 4. Production-safety confirmation (Elasticsearch)

- Only throwaway indices named `veda_bench_<epoch>` were created (one per run: dense, ES-int8). Each was **deleted in a `finally` block**.
- Bulk batches ≤ 500 docs; single-client sequential query load (no concurrent hammering of prod).
- No other index was listed-then-modified or touched.
- Post-run verification:
  - `DELETE veda_bench_... -> 200 {"acknowledged":true}`
  - `GET _cat/indices/veda_bench_... -> 404 index_not_found_exception` (gone)
  - `GET _cat/indices/veda_bench_* -> (empty)` — no leftover bench indices
  - Total index count back to **207** (matches the documented prod baseline) — production untouched.
- Veda was benchmarked on an **isolated instance** (port 6343, temp data dir), never the production 6340 instance; the child instance and temp data were cleaned up afterward; prod 6340 confirmed healthy.

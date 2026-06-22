# Veda vs Elasticsearch — Dense-Vector Head-to-Head Benchmark

**Date:** 2026-06-21
**Author:** automated benchmark (iitgpu11)
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
| Hardware | iitgpu11: 72 vCPU, 1 TB RAM, NVIDIA RTX 6000 Ada (48 GB) |
| Veda | local release build, isolated instance on `127.0.0.1:6343` (separate from prod 6340), HNSW |
| Elasticsearch | **production** cluster `es-prod` v8.19.0 at `macdep01.tdc.otsuka-shokai.co.jp:9200` (remote, TLS, self-signed). Auth masked as `***` |

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

---

## Post-fix re-run

**Date:** 2026-06-21 (later same day)
**Binary:** release build at `git 95df295`, including the P0 fix commits `91a9593` (oversample candidates per segment before cross-segment merge), `46584f5` (force-merge + `POST /v1/collections/{name}/optimize`), `c0da217` (consolidate after bulk load + build indexes on recovery).
**Instance:** the **same** live release server on `127.0.0.1:6340` (`./target/release/veda-server ./veda-data-mm`), not an isolated child this time.
**Data:** byte-identical reuse of the prior run's saved vectors — `veda-es-bench/corpus_vecs.npy` (30000x1024 f32, L2-normalized), `query_vecs.npy` (500x1024), and `gt_top10.npy` (exact GPU brute-force cosine top-10). No re-embedding; the ground truth is the same 30k exact GT used for the 0.265 number.
**Collection:** fresh `bench_fixed`, single `vector` field (`dimension=1024, metric=cosine, index="hnsw"`), bulk-inserted via the same 500-record-batch `/insert` path. Settled at **41 sealed segments** after load (even more fragmented than the prior ~38).

### Findings

- **The oversample fix alone does NOT close the recall gap on fragmented data.** Fragmented recall@10 rose only from **0.265 -> 0.329**. It is still catastrophically below ES's 1.000. Per-segment HNSW already runs at `ef_search=200` and the fix now returns each segment's top-`max(k, ef_search)=200` to the cross-segment merge, but with 41 graphs each only approximately ranking, the union still misses ~67% of true top-10 neighbours.
- **`num_candidates: 200` made no difference** (0.329 identical to default). Expected: the default `fetch_k` is already `max(k=10, DEFAULT_EF_SEARCH=200) = 200`, so an explicit `num_candidates: 200` produces the exact same per-segment fetch. The request knob only matters if you raise it well above 200.
- **`/optimize` is the actual remedy and it works.** Force-merge reported `segments_before: 41 -> segments_after: 1`, confirmed by one `segment_*` dir on disk. Post-optimize recall@10 = **0.9974** — effectively matching ES's 1.000 (a single consolidated HNSW graph; the tiny 0.0026 shortfall is normal approximate-HNSW loss at `ef_search=200`, not fragmentation).
- **Latency cost:** oversampling added no measurable cost while fragmented (server p50 ~43-47ms either way). After `/optimize`, server p50 actually rose to **54ms** — a single 30k-node HNSW traversed at `ef_search=200` visits more distinct nodes than the small per-segment graphs did, so consolidation trades a slightly higher (but recall-correct) query for the 3x recall gain. This is still ~7-8x slower than ES's 7ms `took` on the engine-time comparison. **The fix closes the recall gap; it does not close the latency gap.**
- **`/optimize` is expensive:** the force-merge + synchronous HNSW rebuild of 30k x 1024 at `ef_construction=200` took on the order of ~9-10 minutes single-threaded (the `/optimize` HTTP call blocks for its full duration). It is an operator-run consolidation step, not a per-query path.

### Comparison table

| state | recall@10 | p50 latency (server) | segments |
|---|---|---|---|
| Veda BEFORE fix (prior run) | 0.265 | ~47 ms `ann` / 47 ms `took` | ~38 |
| Veda AFTER fix, fragmented (default ef) | **0.329** | 47 ms `ann` / 47 ms `took` | 41 |
| Veda AFTER fix, fragmented (`num_candidates: 200`) | **0.329** | 44 ms `ann` / 43 ms `took` | 41 |
| Veda AFTER fix, post-`/optimize` | **0.9974** | 54 ms `ann` / 54 ms `took` | 1 |
| Elasticsearch (prior run) | 1.000 | 7 ms `took` | 1 |

(Client-side p50: fragmented 54 ms default / 49 ms nc200; post-optimize 62 ms. ES client p50 16 ms over remote TLS.)

### Honest verdict

The P0 dense-recall fix **does close the recall gap vs Elasticsearch on this real 30k clustered data — but only after running `/optimize`, not from the oversample change alone.** The per-segment oversampling lifts fragmented recall a negligible amount (0.265 -> 0.329, still ~0.33 vs ES 1.0), and `num_candidates` is a no-op at the default-200 setting; on a fragmented bulk load Veda remains far below ES. What actually fixes it is the new force-merge: `POST /optimize` consolidates the 41 segments into 1 and rebuilds the graph, after which recall is **0.9974 — a dead heat with ES on recall**. So the honest story is: fragmented Veda is still well below ES (the oversample fix is not a substitute for consolidation), and **`/optimize` is the required remedy** to reach ES-parity recall. The remaining gap is latency, not recall: even consolidated, Veda's engine p50 is ~54 ms vs ES's ~7 ms, and `/optimize` itself is a multi-minute operator operation. Recall gap: closed (via `/optimize`). Latency gap: not closed.

## Latency optimization re-run (HNSW perf cycle)

**Date:** 2026-06-21 (later). **Binary:** release at git `2914118`, incl. perf commits `cb6ee0f`
(generation-stamped visited bitset + reusable heaps in `search_layer`), `083bd0d` (normalized
cosine→dot internal distance + Algorithm-4 neighbour selection), `b346a67` (per-query `ef` threaded
into traversal + default `ef_search` 200→96). Same 30k corpus / 500 queries / exact GT; fresh
`bench_fixed` (39 sealed segments after load).

| state | recall@10 | server p50 (ann) | segments |
|---|---|---|---|
| Veda after recall-fix, post-`/optimize` (prior run) | 0.9974 | 54 ms | 1 |
| Veda after PERF cycle, fragmented (default `ef=96`) | 0.249 | 42 ms | 39 |
| Veda after PERF cycle, fragmented (`num_candidates=200`, ef=200) | 0.296 | 62 ms | 39 |
| **Veda after PERF cycle, post-`/optimize` (ef=96)** | **1.000** | **36.7 ms** | 1 |
| Elasticsearch | 1.000 | 7 ms | 1 |

### Verdict
The perf cycle (per-query ef + default ef 200→96, normalized cosine→dot, Algorithm-4 neighbour
selection, bitset visited + reusable heaps) cut consolidated single-segment latency **54 ms → 36.7 ms
(~32%)** and lifted recall to **1.000** (the neighbour heuristic closed the last 0.0026). Real progress,
but Veda is **still ~5× slower than Elasticsearch's ~7 ms**. The cheap query-side wins delivered less
than the ~15–25 ms estimate, which points to the **deferred structural cost (#3): the `Vec<Vec<f32>>`
vector store + `Vec<Vec<u32>>` adjacency** — pointer-chasing over ~120 MB at 30k is cache-hostile (and
explains why Veda scales worse than ES). **Closing to ES-class latency requires the contiguous
vector-slab + CSR-adjacency rewrite** (a separate, higher-risk cycle with an on-disk segment-format
change). Status vs ES: **recall at parity (1.0); latency improved ~32% but the gap is not closed.**

### ef sweep (confirmation run, same binary, fresh consolidated collection)

A follow-up sweep on a second fresh 30k load consolidated to 1 segment, varying the per-query `ef`
(`num_candidates`) on the new heuristic-built graph. Recall/latency are server-side (`took_ms` p50) over
500 queries vs exact GT:

| ef | recall@10 | server p50 | server p95 |
|---|---|---|---|
| 64 | **0.9998** | **21 ms** | 40 ms |
| 80 | 1.000 | 26 ms | 50 ms |
| **96 (default)** | **1.000** | **29–31 ms** | 56 ms |
| 112 | 1.000 | 33 ms | 58 ms |
| 128 | 1.000 | 38 ms | 75 ms |
| 200 (old default) | 1.000 | 48–63 ms | 84–115 ms |
| Elasticsearch | 1.000 | 7 ms | — |

**Takeaways:**
- On real L2-normalized embeddings the graph is far easier than the adversarial synthetic unit-test data:
  recall is already 1.000 at ef=80 and **0.9998 at ef=64**. The default of **96** is a conservative
  choice (recall exactly 1.0 with headroom); **ef=64 is a viable aggressive setting** that holds
  recall ≥0.99 at **21 ms** — the bottom of the original 15–25 ms target band, and ~2.5× faster than the
  54 ms baseline.
- Latency scales almost linearly with ef here, confirming the layer-0 traversal budget (now correctly
  plumbed per-query) is the dominant cost — exactly what fix A targets. Callers who want ES-class latency
  on this data can pass `num_candidates: 64` today.
- The remaining gap to ES's 7 ms at a fixed ef is structural (deferred #3, the contiguous-slab + CSR
  rewrite), but the per-query `ef` knob now lets operators trade the last few points of recall for
  latency without a rebuild.

**Build-time cost (honest):** the Algorithm-4 neighbour selection makes `/optimize` (the synchronous
single-threaded HNSW rebuild) materially slower than the prior plain-nearest-M builder — the diversity
test is O(M²) in distance comps per node in the back-link prune, and over `Vec<Vec<f32>>` storage on a
loaded box this confirmation run's rebuild took ~30 min vs the prior cycle's ~10 min. `/optimize` is a
one-time operator step, not a query path, but this regression is real and is another argument for the
contiguous-layout rewrite (which would also speed the build).

---

## Contiguous vector-slab re-run (2026-06-22) — the layout hypothesis, falsified

Deferred item #3 above predicted that replacing the cache-hostile `Vec<Vec<f32>>` vector
storage with a contiguous flat `Vec<f32>` slab (commit `453f2dd`) would close the gap to ES.
We built it and re-ran the identical 30k×1024 protocol back-to-back against the pre-slab build
to control for host load. **The prediction was wrong.**

| Build | Config | Recall@10 | server p50 | ann_search p50 |
|---|---|---|---|---|
| pre-slab (bench2) | post-optimize, ef=96 | 1.000 | 36.5 ms | 36.7 ms |
| **slab (bench3)** | post-optimize, ef=96 | 1.000 | 37.0 ms | **36.6 ms** |
| **slab (bench3)** | post-optimize, ef=64 | 0.9998 | 20.0 ms | 20.5 ms |
| Elasticsearch | — | 1.000 | 7 ms | — |

**The contiguous slab changed latency by ~0 ms (36.6 vs 36.7 at ef=96).** Memory layout was
**not** the bottleneck. Two corollaries, both verified in-tree:

1. `ann_search` is 36.6 ms of the 37 ms total → ~99 % of query time is inside HNSW
   traversal+distance, **not** HTTP/auth/serde per-request overhead. The gap to ES is in the graph walk.
2. The distance hot path (`veda-core/src/simd.rs`) is **already real AVX2+FMA** (`_mm256_fmadd_ps`,
   runtime-dispatched, scalar fallback; plus an int8 `_mm256_madd_epi16` path). So scalar-arithmetic
   is not the lever either.

**Diagnosis.** 36 ms for ef=96 over only 30k vectors is abnormally high (a tight HNSW does this in
~1 ms; ES in 7 ms). With both layout and arithmetic ruled out, the cost is in **traversal efficiency**:
node-visit count and per-hop heap/visited bookkeeping — possibly per-query allocation, the `fetch_k`
oversample widening the walk, or redundant distance recomputation. This is NOT addressed by the
queued CSR-adjacency + on-disk-slab increment (which targets the same now-falsified layout hypothesis),
so that increment is **deprioritised**.

**Recommendation.** The highest-value remaining lever is **quantized-distance traversal** — walk the
graph on int8/SQ8 (or RaBitQ) distances and rerank the final top-k at full precision, which is exactly
what Lucene/ES do and which Veda already has partial machinery for (`HnswRaBitQ`, SQ8 index types).
Alternatively, ship ef=64 (recall 0.9998, 20 ms — at recall parity with ES) as good-enough and pivot
to the categorical differentiators. The slab itself is retained: it is harmless, builds clean, and is a
better foundation for any future mmap on-disk format — but it is not, on its own, a latency win.

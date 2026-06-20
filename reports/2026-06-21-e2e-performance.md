# Veda Multimodal E2E — Phase 4 Performance Report (2026-06-21)

Measured live against the FINAL release `veda-server` on `:6340` (data `./veda-data-mm`, auth OFF, local-FS blob).
Unified embedder: ONE `jina-embeddings-v5-omni-small` sidecar at `http://localhost:8001` (1024-d, text/image/audio/video).
Reranker `cl-nagoya/ruri-v3-reranker-310m` (TEI) at `iitgpu03:9001`. LLM `Qwen/Qwen3.5-122B-A10B` (vLLM) at `difgpu01:9000`.
Embedder warmed before measuring (omni's first audio inference is a one-time ~60 s; steady-state numbers below).

## Method notes & caveats (READ FIRST)
- **Corpus is small, so search is Flat brute-force.** `txt` = 400 real English passages; `mm` = 23 multimodal records. Both stay in the *growing* segment (seal threshold hardcoded 512 MB, no seal REST route — Finding 4 in the bug log), so every dense query is **exact brute-force over the growing segment**, NOT HNSW/IVF-PQ/etc. Search latencies below are Flat-scan numbers; ANN index latencies are unreachable via the live API.
- **Search server-side timing** comes from `explain.ann_search_ms` / `explain.total_ms`. **Wall RTT** is `curl %{time_total}` from the same host (loopback). For BM25 / match_phrase / hybrid, `explain` is null (timings only populated on the dense/vector path — known), so those rows are **wall-clock only**.
- **Embed RTT** uses the server-reported `elapsed_ms` in the `/v1/embed` response (sidecar inference + transport), plus the curl wall RTT. p50/p95 over 20 calls each, warm.
- **Cold vs warm:** all numbers are WARM. Cold start of the omni audio path is ~60 s (one-time model/codepath warmup), then ~0.5–1.3 s steady.
- **LLM via Veda is currently broken by a host run-script env defect** (see the LLM section + Finding LLM-1): `VEDA_LLM_ENDPOINT` in `start_veda_mm.sh` was set to `…:9000/v1` but Veda's `LlmClient` POSTs the endpoint verbatim (no `/chat/completions` suffix appended), so Veda to `POST …/v1` to **404**. The vLLM server itself is healthy (`…/v1/chat/completions` to 200). LLM numbers below were therefore measured **directly against vLLM** and characterize the model, not the (currently-broken) Veda agent path. I did **not** restart the running `:6340` server (host-safety: single-flight, never disturb the live binary).

## 1. Omni embed RTT per modality (`/v1/embed`, warm, n=20)

| Modality | server `elapsed_ms` p50 | p95 | min–max | wall RTT p50 / p95 (ms) | Method |
|---|---|---|---|---|---|
| text  | **584.7** | 943.3 | 439–967 | 609 / 973 | 20x `/v1/embed {text}` |
| image | **2687.2** | 3964.7 | 1863–3992 | 2722 / 3998 | 20x `/v1/embed {image_b64}` (OmniDocBench JPG, ~411 KB) |
| audio | **1090.9** | 1564.0 | 704–2362 | 1124 / 1609 | 20x `/v1/embed {audio_b64}` (ESC-50 WAV, ~431 KB) |
| video | **1304.3** | 1697.8 | 1054–2161 | 1328 / 1760 | 20x `/v1/embed {video_b64}` (ffmpeg testsrc, 2 s 320x240 MP4, ~11 KB) |

Corroborated by Prometheus `veda_embedding_duration_seconds` (sum/count averages): text 0.57 s, image 2.82 s, audio 0.93 s, video 1.29 s — consistent with the p50s above. Image is the slowest modality (largest payload + vision tower); text the fastest.

## 2. Ingest throughput

| Path | Modality | Throughput | Method |
|---|---|---|---|
| `/v1/collections/txt/embed-insert` | text | **1.75–2.39 rec/s** (10 recs to 2.39/s; 50 recs to 1.75/s) | serial batch; embed-bound (~0.5 s/text embed done sequentially) |
| `/v1/collections/{c}/ingest-media` | image | **0.33 items/s** | 8x multipart upload (OmniDocBench JPG); embed-bound (~2.8 s/image) |
| `/v1/collections/{c}/ingest-media` | audio | **1.19 items/s** | 8x multipart upload (ESC-50 WAV); ~0.85 s/audio |
| `/v1/collections/{c}/ingest-media` | video | **0.70 items/s** | 8x multipart upload (synth MP4); ~1.3 s/video |

Throughput is dominated by the (single, serial) omni embedder, not by Veda's storage path. Image is slowest (heaviest embed). Media collections used a vector field with `auto_embed{source:media_blob, modality:…}` + a `media_blob` Blob field (the shape `ingest-media` requires; pinning to one modality each). Test collections were deleted after measuring.

## 3. Dense search latency (`txt`, k=10, n=50, explain)

| Metric | p50 | p95 | p99 | min–max |
|---|---|---|---|---|
| `explain.ann_search_ms` (server) | **9.10** | 17.42 | 18.20 | 5.69–18.76 |
| wall RTT (loopback, per-call body rebuild) | 18.68 | 36.32 | 42.24 | 10.95–43.59 |

Sample explain: `{filter_ms 0.0, ann_search_ms 10.4, scoring_ms 0.009, fusion_ms 0.0, total_ms 10.4}` — pure brute-force scan over 400x1024-d vectors. (Wall RTT here is inflated by a per-iteration body re-serialization in the harness; the server-side `ann_search_ms` is the reliable figure.)

## 4. Per query-mode latency (`txt`, k=10, n=20, wall RTT ms)

| Mode | p50 | p95 | Notes |
|---|---|---|---|
| dense (vector) | **2.75** | 7.20 | body built once; explain `ann_search_ms` p50 ~ 9 ms (see section 3) |
| bm25 | **1.47** | 2.61 | tantivy full-text; `explain` null on this path |
| match_phrase (slop=2) | **1.58** | 2.75 | phrase proximity; `explain` null |
| hybrid (dense+bm25, RRF) | **2.74** | 10.72 | fuses both legs; `explain` returns null (known: only vector path fills explain) |

All four return correct hits (verified semantically in Phase 2.B). BM25/match_phrase are fastest (small index); hybrid ~ dense (dominated by the dense leg).

## 5. Cross-modal search latency (text to media, `mm`, n=30)

| Metric | p50 | p95 |
|---|---|---|
| `explain.ann_search_ms` | **0.33 ms** | 1.65 ms |
| wall RTT | 2.98 ms | 13.04 ms |

Sub-millisecond scan (only 23 records). Semantically correct: text query "a dog barking" to top non-text hit is **audio/dog @ dist 0.496** (matches Phase 2.A ground truth: text-to-audio strong). (Video-to-text remains weak — quality, not latency; see Phase 2 Finding 2.)

## 6. Rerank latency (ruri-v3 TEI, 20 candidates, n=20)

| Metric | p50 | p95 |
|---|---|---|
| server `took_ms` | **106.5 ms** | 131.3 ms |
| wall RTT | 115.0 ms | 138.0 ms |

20 docs reranked by the cross-encoder in ~106 ms p50 (~5 ms/doc). Reorders correctly (Phase 2.B verified).

## 7. AIOps endpoint latency (wall-clock, n=3 each)

| Endpoint | latencies (ms) | Note |
|---|---|---|
| `POST /admin/insights/run` | 116 / 165 / 211 | ticks the whole rule engine on demand (37 rules) — heaviest |
| `GET /admin/correlations` | 4.7 / 3.2 / 3.1 | 300 s window aggregation |
| `GET /admin/resources` | 1.4 / 1.0 / 0.9 | disk/cpu/mem/rss snapshot |
| `GET /admin/forecast` | 1.2 / 1.1 / 5.4 | empty (insufficient series) — cheap |
| `GET /admin/index-advisor` | 21.4 / 2.3 / 1.2 | first call warms per-collection stats |
| `GET /admin/depmap` | 1.2 / 0.7 / 1.0 | static topology (9 nodes) |

All sub-250 ms; the read-only AIOps GETs are single-digit ms. `insights/run` is the only one doing real per-rule work.

## 8. LLM: converse total + TTFT

**Veda agent path is currently BROKEN (404) — see Finding LLM-1.** Numbers below measured **directly against the healthy vLLM** (`difgpu01:9000/v1/chat/completions`), n=3 each:

| Metric | values | Note |
|---|---|---|
| non-stream total (~60–74 completion tok) | 0.40 / 0.50 / 0.56 s | prompt ~45 tok, completion 57–74 tok |
| stream TTFT (`time_starttransfer`) | 23.6 / 26.3 / 21.9 ms | first-token latency |

For reference, when the Veda agent path WAS healthy (Phase 2.D), `veda_llm_ttft_seconds` summary read p50 ~ 0.38 s for full agent runs (which include tool-calls + RAG before the LLM), and a grounded triage RCA took total ~ 2.6 s (llm_ms 2.63 s, 1893 prompt / 361 completion tok). The raw model TTFT (~22 ms) is far below the agent-run TTFT because the agent does retrieval/tool work first.

### Finding LLM-1 (BUG, P1 — host run-script env, not a repo defect)
`start_veda_mm.sh` exports `VEDA_LLM_ENDPOINT=http://difgpu01.tdc.otsuka-shokai.co.jp:9000/v1`. Veda's `veda_core::llm_client::LlmClient` uses `VEDA_LLM_ENDPOINT` **verbatim** as the POST URL (`.post(&self.config.endpoint)` in `crates/veda-core/src/llm_client.rs`; it does NOT append `/chat/completions`). So Veda POSTs to `…/v1` to vLLM **404 Not Found**; the LLM doc-comment and the (correct) gitignored `.env` use the full `…/v1/chat/completions`. After the Phase 4-pre restart used the start-script's `/v1`, and because the `.env` loader (post-fix `ea3292a`) no longer overrides an already-exported var, the wrong value won. **Same class as the searxng-port bug** (host script env, no Veda code change). Fix: set the start-script `VEDA_LLM_ENDPOINT` to the full `…/v1/chat/completions` and restart. NOT done here — host-safety forbids disturbing the live `:6340` process mid-campaign. (Optionally harden `LlmClient::from_env` to append `/chat/completions` when the endpoint ends in `/v1` — a P2 robustness fix.)

## 9. Resource footprint

| Metric | Value | Source |
|---|---|---|
| Veda server peak RSS | **75.3 MB** | `/v1/admin/resources` `rss_bytes` (75,309,056) |
| Omni sidecar GPU mem (GPU 0) | **4618 MiB** / 49140 MiB | `nvidia-smi` (GPUs 1–3 idle 2 MiB) |
| GPU utilization (idle between calls) | 0 % | `nvidia-smi` (bursts during inference) |
| Host: disk used | 77.98 % (1.75 TB free of 7.94 TB) | `/v1/admin/resources` |
| Host: load1/5/15 | 22.6 / 30.3 / 57.5 (72 CPUs) | shared host (other tenants) |

Veda's own process is tiny (~75 MB RSS); the GPU cost is the single omni sidecar at ~4.6 GB (one model serving all four modalities — the point of the unified embedder).

## 10. Niriksha bundle size (from `niriksha/dist`, vite 8 build)

| Asset | raw | gzip |
|---|---|---|
| DashboardHome (largest chunk) | 439.3 KB | 125.6 KB |
| index (entry) | 271.6 KB | 86.4 KB |
| primitives | 149.7 KB | 47.1 KB |
| umapWorker (Atlas) | 102.8 KB | 27.8 KB |
| proxy | 99.1 KB | — |
| CollectionBrowser | 78.3 KB | — |
| **Total `dist/`** | **1.8 MB** | — |

Largest chunk DashboardHome 439 KB (126 KB gz); entry 272 KB (86 KB gz). Consistent with the Phase 3 build report.

## Headline
- **Search is fast** on this (small, Flat) corpus: dense `ann_search_ms` p50 ~ 9 ms, cross-modal sub-ms, BM25/phrase ~1.5 ms.
- **The omni embedder is the bottleneck for ingest** (text ~0.57 s, image ~2.8 s, audio ~0.9 s, video ~1.3 s per item, served serially) to ingest 0.33–2.4 items/s.
- **Rerank** ~106 ms for 20 docs; **AIOps** GETs single-digit ms, `insights/run` ~150 ms.
- **LLM model is healthy** (~0.4–0.56 s/answer, TTFT ~22 ms) but the **Veda agent LLM path is currently 404** due to a host start-script env defect (Finding LLM-1) — fixable by one env correction + restart.
- **Footprint:** Veda 75 MB RSS; one omni sidecar 4.6 GB GPU; Niriksha 1.8 MB dist (126 KB gz largest chunk).
- **Caveat:** ANN index latency/recall is NOT measurable via the live API on small data (everything brute-forces Flat in the growing segment — Phase 2 Finding 4).

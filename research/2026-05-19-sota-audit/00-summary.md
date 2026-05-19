# Veda — 2026 SOTA audit, consolidated summary

Generated 2026-05-19 by eight parallel research agents. Each file in
this directory is a deep dive on one subsystem; this is the TL;DR
with verdicts and the cross-cutting recommendation order.

Every audit asks the same question: **Is Veda's choice state of the
art as of May 2026? If not, what's ahead, by how much, and is the
swap worth it?** Sources, version numbers, and benchmark scores are
cited inline in the per-topic files.

## Headline scorecard

| # | Subsystem | Status | Headline gap | Per-topic file |
|---|---|---|---|---|
| 1 | Vector indexes (ANN) | Mixed | DiskANN + IVF-PQ behind family roadmaps; HNSW/Flat/SQ8 par; RaBitQ ahead. No first-class GPU index (CAGRA scaffolded but not promoted), no ACORN filtered traversal, no SPANN/SPFresh for object-storage. | [01-vector-indexes.md](01-vector-indexes.md) |
| 2 | Quantization | SOTA-leaning | RaBitQ 1-bit still SOTA recall-per-bit. Missing the **Extended-RaBitQ** multi-bit ladder for the rerank stage. SQ8 lacks Walsh-Hadamard rotation. No GPU quantizer path. | [02-quantization.md](02-quantization.md) |
| 3 | Text embeddings | Behind frontier, defensible default | BGE-M3 ~7-9 pts behind MTEB-EN frontier (NV-Embed-v2 72.31, Qwen3-Embedding-8B 70.58 vs BGE-M3 ~63.0). But unique: only model shipping dense + sparse + multi-vector from one forward pass. | [03-text-embeddings.md](03-text-embeddings.md) |
| 4 | Multimodal embeddings | Top-3 in class | Qwen3-VL-Embedding-2B is defensible. Modality gap 0.25 is the smallest in the field (vs Voyage 0.59, Gemini 0.73) — directly validates Veda's "one unified vector space" design. Add 8B opt-in. | [04-multimodal-embeddings.md](04-multimodal-embeddings.md) |
| 5 | Late interaction | Architecture right, engines aging | ColPali + MUVERA + PLAID still defensible. **PLAID is dominated by WARP (SIGIR'25, 3x speedup, 2-4x smaller index).** MaxSim not SIMD-optimized. ColQwen2.5 / ColModernVBERT adapters missing. | [05-late-interaction.md](05-late-interaction.md) |
| 6 | Sparse + fusion + rerank | Largely SOTA | SPLADE-v3 + BM25 + RRF k=60 are the right defaults. Generic Voyage/Cohere-shape rerank HTTP API matches what major DBs ship. Gap: no `instruction:` field passthrough blocks Voyage rerank-2.5's value-add. No CC/DBSF fusion options. | [06-sparse-fusion-rerank.md](06-sparse-fusion-rerank.md) |
| 7 | Chunking + OCR | Chunking SOTA, OCR slipping | 7 chunkers cover SOTA technique surface. Defaults overshoot: `MarkdownRecursive {2000c, 300o}` is past the 2.5k-token "context cliff" recent studies surface. **FlashParse no longer SOTA — replace with PaddleOCR-VL-1.5 (#1 OmniDocBench, Apache-2.0, 0.9B params) or Granite-Docling-258M (IBM, Jan 2026).** | [07-chunking-ocr.md](07-chunking-ocr.md) |
| 8 | Storage + GPU + system | Architecture par, format behind | JSONL below SOTA — adopt Lance 2.2 as third format option. cuVS env-gating right direction; gap is promoting CAGRA from optional fallback to first-class index. Field converged on **object-storage-native + LSM compaction** (Turbopuffer, Pinecone, Milvus 2.6, S3 Vectors). Milvus's **Woodpecker** (object-storage-backed WAL, Jan 2026) is the architecturally distinctive 2026 move. | [08-storage-gpu-system.md](08-storage-gpu-system.md) |

## Where Veda is genuinely ahead

1. **RaBitQ shipped early.** Most peers caught up in 2025; Veda was there in the first wave.
2. **Unified vector space across modalities.** The "any non-text input → multimodal model, single shared dim" auto-routing matches where Cohere / Gemini / Voyage are moving. Veda's already there.
3. **Multi-index quiver in one workspace.** 7 ANN index types under one Apache-2.0 Rust crate is rare. Lance and Qdrant are the closest peers.
4. **First-class explain timings (`"explain": true`).** Few vector DBs expose per-phase latency natively at the API level.
5. **Open dense + sparse + multi-vector triplet** from BGE-M3 in one forward pass — no frontier model duplicates this.

## Where Veda is genuinely behind

In order of "biggest gap × highest payoff":

1. **No first-class GPU index.** `gpu_cuvs.rs` already wraps cuVS with CAGRA build + `serialize_to_hnswlib` — the work is done; promote it. Faiss v1.10 (Feb 2025), Milvus, OpenSearch 3.0, Elasticsearch 9.3 all shipped CAGRA as a default option.
2. **Storage format JSONL.** Lance 2.2 (2026) is the production-ready columnar format for vector + scalar + versioning. Single biggest performance footgun in the storage layer.
3. **OCR FlashParse.** Doesn't appear on any 2025/2026 leaderboard; PaddleOCR-VL-1.5 dominates at the same hardware budget.
4. **PLAID engine internals.** WARP delivers 3x speedup at 2-4x smaller index; semantically a drop-in.
5. **DiskANN/IVF-PQ on object storage.** Veda's MinIO/S3 backend doesn't match DiskANN's NVMe-random-4KB access pattern. SPANN-class centroid+posting is the right shape.

## Where Veda is fine

- **HNSW, Flat, SQ8** — par with 2026 mainstream.
- **RaBitQ 1-bit** — still SOTA recall-per-bit.
- **SPLADE + BM25 + RRF (k=60)** — exactly what every major DB ships.
- **Cohere/Voyage-shape rerank HTTP API** — the convergent API surface.
- **Auto-modality embedding routing** — the unified-vector-space design is where the field is converging.
- **NIM as embedding-side only** — correct positioning (not as a vector search engine).

## Cross-cutting recommendation order

If you want to close the gaps in a sensible sequence, in priority order:

1. **Promote `gpu_cuvs::CagraIndex` to a first-class index.** The Rust wrapper exists; lift it from env-gated fallback to a `IndexType::Cagra` variant. Largest perf delta for least architectural change. (See [01](01-vector-indexes.md), [08](08-storage-gpu-system.md).)
2. **Adopt Lance 2.2 as third stored-fields format option** alongside JSONL + columnar. Versioning + zero-copy + columnar vectors. ([08](08-storage-gpu-system.md).)
3. **Replace FlashParse with PaddleOCR-VL-1.5** as the default OCR engine. Same VRAM, Apache-2.0, top of OmniDocBench. ([07](07-chunking-ocr.md).)
4. **Add Extended-RaBitQ multi-bit ladder (3/4/5/7/8/9-bit)** as the rerank stage; current full-float rerank wastes bandwidth. ([02](02-quantization.md).)
5. **Add ACORN-style filtered graph traversal to HNSW.** Small implementation, large RAG quality win on filtered ANN. Qdrant 1.16 (Nov 2025), Weaviate 1.27, Vespa Sep 2025 all shipped it. ([01](01-vector-indexes.md).)
6. **Add `Qwen3-Embedding-8B` and `voyage-3-large` to the embedding registry** as opt-in upgrades; keep BGE-M3 default for the hybrid-retrieval triple-output workload. ([03](03-text-embeddings.md).)
7. **Swap PLAID engine internals for WARP semantics** — 3x speedup, 2-4x smaller index, same API surface to the multi-vector code path. ([05](05-late-interaction.md).)
8. **Add `instruction:` field passthrough to the rerank HTTP body** — one-line gateway change that unlocks Voyage rerank-2.5's instruction-following. ([06](06-sparse-fusion-rerank.md).)
9. **SIMD-optimize MaxSim** by porting Mixedbread's `maxsim-cpu` (libxsmm + AVX2) into `veda-core::simd`. 10-20x over the current PyTorch-CPU-equivalent. ([05](05-late-interaction.md).)
10. **Retune `MarkdownRecursive` defaults** from `{2000, 300}` to `{1500, 0}` per the Jan 2026 systematic chunking study; expose factoid vs analytical presets. Document the 2.5k-token context cliff. ([07](07-chunking-ocr.md).)
11. **Add Granite-Docling-258M as second OCR option** for the page-image embedding path; complements PaddleOCR-VL-1.5 on long documents. ([07](07-chunking-ocr.md).)
12. **Wire CC + DBSF as fusion options** (`veda-query::fusion`) — RRF stays default but the labeled-query regime benefits substantially. ([06](06-sparse-fusion-rerank.md).)

## What deliberately does NOT need to change

- **Consensus crate (openraft 0.9).** Covered by `documentation/consensus-swap-decision.md` — the answer is the FoundationDB-shape rearchitecture, not a swap.
- **Auto-modality routing logic.** SOTA.
- **`"explain": true` per-phase timings.** Differentiator; don't drop it.
- **Apache-2.0 license posture.** Avoid pulling in NSCL / CC-BY-NC dependencies (NV-Embed-v2, Omni-Embed-Nemotron, Llama-Embed-Nemotron-8B all flagged for license traps).
- **The 7 chunking strategies as a set.** Cover the SOTA technique surface; just retune defaults.

## What's NOT in this audit

These were either covered separately in earlier session work, or are infrastructure choices for which "is this SOTA?" isn't the right framing:

- Raft / consensus (covered: `documentation/consensus-swap-decision.md`).
- Linearisability checker (covered: `research/2026-05-19-multi-week-survey/02-linearisability-checkers.md`).
- Transactions (covered: `documentation/transactions-v0.md`).
- Multi-tenancy (covered: `research/2026-05-19-multi-week-survey/04-multi-tenancy.md`).
- Auth (covered: `research/2026-05-19-multi-week-survey/05-auth-rust-crates.md`).
- The Phase A/B+ rearchitecture itself (covered: `documentation/rearchitecture-overview.md`).
- The Niriksha frontend (covered: `.claude/plans/structured-mixing-turing.md`).

## Methodology

- 8 parallel general-purpose research agents.
- Each agent independently performed WebSearch + WebFetch against arXiv, vendor blogs, HuggingFace leaderboards, GitHub release pages, and ANN-Benchmarks.
- Each agent confirmed Veda's current state via the existing codebase + the project's CLAUDE.md.
- Findings recorded as ~3000-4000 word reports with version numbers, release dates, and per-claim citations.
- Total ~26,000 words across the 8 reports.
- Source distribution: arXiv papers 2024-2026, vendor changelogs (Cohere / Voyage / Jina / Qdrant / Milvus / Vespa / LanceDB / Pinecone / NVIDIA), HuggingFace MTEB / MMEB-V2 / OmniDocBench leaderboards as of May 2026.

# State-of-the-Art Audit: Late Interaction & Multi-Vector Retrieval (2026-05)

**Scope**: Veda currently ships ColPali (vision late interaction), MUVERA (FDE compression), PLAID (centroid pruning), `MultiVector` field type in `veda-core`, and MaxSim scoring in `veda-query::local_shard.rs`. This audit checks whether those choices are still defensible against the 2026 frontier and where the gaps are.

---

## Verdict (TL;DR)

**Are ColPali + MUVERA + PLAID still the right choices in 2026?** Mostly yes, but each is showing its age in a specific way.

- **ColPali (the architecture/recipe)** — still correct. **ColPali-v1.2 (PaliGemma-3B base)** — no longer SOTA. The ViDoRe v2 leaderboard is now topped by **ColQwen2.5** and **jina-embeddings-v4** (multi-vector mode). Veda must adopt at least the ColQwen family or lose ~5-7 nDCG@5 on visual document benchmarks.
- **MUVERA** — still the right primitive (FDE that reduces multi-vector to single-vector MIPS). It is now shipped in production by **Qdrant (FastEmbed 0.7.2, Sep 2025)** and **Weaviate 1.31+** with first-class support, and a **Rust crate (`muvera-rs`, `ruvector`)** exists. No published successor that we should swap in; the live research direction is integrating MUVERA with **token pooling** and **binary quantisation** rather than replacing the FDE itself.
- **PLAID** — was the right choice in 2023. In 2026 it is dominated by **WARP (SIGIR'25, Stanford/UC Berkeley/ETH)** which delivers **3x speedup over PLAID, 41x over XTR**, plus **2-4x smaller index** with no quality loss. WARP should be considered a drop-in successor.

**Top 3 gaps**:
1. **ColQwen / ModernVBERT family is missing.** Veda is one major-version behind on the vision late-interaction model.
2. **PLAID engine is behind WARP.** Same algorithmic family, materially better engineering, paper is open-source (`stanford-futuredata/colbert` + `jlscheerer/xtr-warp`).
3. **No `WARP_SELECT`-style dynamic similarity imputation in `local_shard.rs`**, and no native `mxbai-edge-colbert`-style small/CPU path. Veda's MaxSim is naive in comparison to `maxsim-cpu` (libxsmm + AVX2, ~10-20x over PyTorch CPU).

**Specific recommendation**:
- **Keep** ColPali path as the legacy/reference encoder, **add** ColQwen2.5 + ColModernVBERT model adapters (no schema change needed — `MultiVector` is general).
- **Keep** MUVERA implementation; expose `k_sim`/`d_proj`/`r_reps` as collection-level config (matching Weaviate's API surface) and add **MUVERA + binary quantisation** as a stacked option (this is what Vespa does to scale ColPali to billions).
- **Swap PLAID engine internals to WARP semantics** (`WARP_SELECT`, implicit decompression, two-stage reduction). The Veda type system already exposes the right knobs; the change is inside `veda-query::plaid.rs`.
- **Add a `maxsim-cpu`-equivalent fast path** for the in-process scoring loop (libxsmm on x86, NEON on Apple/ARM); GPU MaxSim only matters at >10M doc tiers.

---

## 1. ColBERT v2: still the canonical text late-interaction baseline

ColBERTv2 (Stanford, 2021/2022) remains the reference single-modal late-interaction model in 2026. It introduced **residual compression** and **denoised supervision**, cutting the storage footprint of vanilla ColBERT by 6-10x while holding (or improving) MS MARCO / BEIR quality. Critically, **there is no "ColBERTv3"**. The research community went in three different directions instead:

1. **Better base encoders**: `answerai-colbert-small-v1` (33M params, outperforms ColBERTv2 on BEIR despite being one-third the size); `mxbai-edge-colbert-v0` (17M/32M, projection dims 48/64, built on Ettin/ModernBERT, **outperforms ColBERTv2 on BEIR and long-context tasks at one-sixth the parameter budget**); `GTE-ModernColBERT` and `Reason-ModernColBERT` trained via PyLate.
2. **Multilingual coverage**: `jina-colbert-v2` (Jina, Aug 2024) — 89 languages, **+6.5% over ColBERTv2**, Matryoshka output dims (64/96/128), 8192-token context, 50% smaller index than v1.
3. **Faster engines**: PLAID (2022), then EMVB (ECIR 2024, 2.8x faster than PLAID via SIMD column-wise centroid + PQ), then **WARP (SIGIR'25)** which now leads the field.

**Takeaway for Veda**: ColBERTv2 is fine as a reference; for new collections in 2026 prefer one of the ModernBERT-derived ColBERTs. Veda doesn't have to ship a model — but the test data and `crates/veda-query/tests` should validate against at least `answerai-colbert-small` to stay relevant.

## 2. ColPali: still the architecture, but the model has been overtaken

ColPali (ILLuin Technology + ETH, arXiv 2407.01449, ICLR 2025) crystallised "page-as-image + ColBERT-style late interaction over patch embeddings" as the dominant visual document retrieval pattern. The original ColPali used **PaliGemma-3B + SigLIP-So400m/14**, producing 1024 patch vectors per page at 128 dims each (~16KB per page in float).

**Successor models in the ColVision family** (all use the `colpali-engine` repo `illuin-tech/colpali`):

- **ColQwen2** (Qwen2-VL backbone): +5 nDCG@5 over ColPali on ViDoRe, **768 patch vectors** (vs 1024), Apache 2.0/MIT licence, supports arbitrary image resolutions/aspect ratios (no forced square resize).
- **ColQwen2.5** (Qwen2.5-VL backbone): tops ViDoRe v2 (harder, multilingual, May 2025 release). Currently the SOTA open ColVision model.
- **ColQwen3**: support added to the `transformers` library and to `colpali-engine` releases.
- **ColSmol / ColSmolVLM** (256M and 500M variants): for cost-sensitive deployments — outperforms ColPali at one-tenth the parameter budget on a subset of ViDoRe.
- **ColNomic Embed Multimodal**: Apache 2.0, OCR-free, competitive on ViDoRe.
- **ColModernVBERT** (Teiletche et al., arXiv 2510.01149, Oct 2025): **250M params**, only **0.6 nDCG@5 below ColPali** on aggregated ViDoRe despite being 10x smaller. Built on ModernVBERT, which is itself a ModernBERT-style vision encoder.
- **ColIDeFics**: the late-interaction adapter applied to IDEFICS-class VLMs.

**Single-vector counterparts catching up**: `jina-embeddings-v4` (3.8B params, Qwen2.5-VL backbone, June 2025) — **84.11 avg ViDoRe in single-vector mode vs ColPali's 83.90**, jumps to **90.17 in multi-vector mode**. Cohere Embed 4 and Voyage Multimodal 3.5 (Jan 2026) similarly close the gap for single-vector visual retrieval. The narrative that "you need late interaction for visually-rich documents" no longer holds unconditionally — **single-vector vision models are within 1-2 nDCG points on ViDoRe v1**, though they still lose on ViDoRe v2 and on table/chart-heavy corpora.

**Hybrid approaches**: HEAVEN (Hybrid-Vector Retrieval for Visually Rich Documents, ACL 2026 Findings, arXiv 2510.22215) is a two-stage system that uses single-vector retrieval over "Visually-Summarized Pages" for stage-1 then multi-vector rerank with query-token filtering. Reports **99.87% of multi-vector Recall@1 at 99.82% lower per-query compute**.

**Takeaway for Veda**: the `MultiVector` field type and MaxSim scoring path are correct and reusable — they were never tied to ColPali's specific patch count. What Veda needs is **a model adapter / embedding gateway** that can route to ColQwen2.5, ColModernVBERT, and jina-v4 multi-vector, ideally selected automatically based on collection modality (text only / mixed / visual-heavy). The `gateway` crate's embedding auto-selection logic is the right place.

## 3. PLAID is functional; WARP is materially better

**PLAID** (Santhanam et al., CIKM'22) made ColBERTv2 production-viable: **centroid interaction** treats each passage as a bag of centroid IDs, **centroid pruning** discards low-scoring docs cheaply, and **residual compression** stores delta-from-centroid in 2-bit codes. Result: **7x faster on GPU, 45x faster on CPU** vs vanilla ColBERTv2.

But the late-interaction engine race kept moving:

- **EMVB** (Rulli et al., ECIR 2024, [arXiv 2404.02805](https://arxiv.org/abs/2404.02805)): adds **bit-vector pre-filtering**, **SIMD column-wise centroid interaction**, **PQ for residuals**, and **per-document term filtering**. **2.8x faster than PLAID with 1.8x smaller index, zero quality loss** on MS MARCO + LoTTE.
- **XTR** (Lee et al., Google DeepMind, NeurIPS'23): rethinks the training objective so the model retrieves the *most important* tokens first, enabling rank-from-retrieved-tokens. Refinement stage is **4,000x cheaper** than ColBERT's. +3.3 nDCG@10 over ColBERTv2 on BEIR.
- **WARP** (Scheerer et al., SIGIR'25, [arXiv 2501.17788](https://arxiv.org/abs/2501.17788), repo `jlscheerer/xtr-warp`): combines XTR's objective with PLAID's index layout, then adds:
  - `WARP_SELECT` — dynamic similarity imputation (no constant-fill for missing query-token centroids).
  - **Implicit decompression** — score directly from quantised residuals; no full vector reconstruction during retrieval.
  - **Two-stage reduction** for score aggregation.
  - Result: **3x over PLAID, 41x over XTR reference impl, 2-4x smaller index, same quality**.
- **ConstBERT** (Frei et al., ECIR 2025, [arXiv 2504.01818](https://arxiv.org/abs/2504.01818)): a *learned* projection that pools per-document token embeddings down to a **fixed small set** (e.g. 32 vectors regardless of doc length). **50%+ index reduction** with near-zero quality loss. Different axis from WARP — orthogonal optimisation.
- **Token pooling** (Clavié et al., NAACL'25, [arXiv 2409.14683](https://arxiv.org/abs/2409.14683)): training-free hierarchical agglomerative clustering on patch/token embeddings; for ColPali, **pool factor 3** keeps **97.8% of original quality at one-third the vectors**.

**Takeaway for Veda**: `veda-query::plaid.rs` is the correct module location — the engine choice can be swapped without disturbing the schema. Migrating semantics to WARP-style (`WARP_SELECT` + implicit decompression) is a one-month engineering project that buys ~3x QPS or ~50% memory at iso-quality. Adding **token pooling** as a pre-index step is a one-week project that buys ~60% storage reduction with negligible loss.

## 4. MUVERA: still the right FDE primitive

MUVERA (Dhulipala et al., Google Research, NeurIPS'24, [arXiv 2405.19504](https://arxiv.org/abs/2405.19504)) is the only mature method that **reduces multi-vector retrieval back to single-vector MIPS** with a formal ε-approximation on Chamfer/MaxSim. The recipe: SimHash-style space partitioning into 2^k_sim buckets + Rademacher random projections per bucket + r_reps repetitions, asymmetric for query vs document.

**Performance** (per the original Google blog and Qdrant/Weaviate validations):
- **90% lower latency vs PLAID** on BEIR at +10% recall.
- **2-5x fewer candidates retrieved** to hit the same recall as previous heuristics.
- Memory: **16x reduction per document** (e.g. 1KB vs 16KB for 32-token ColBERT doc).
- QPS: **329x improvement over brute-force MaxSim** in `ruvector` Rust benchmarks.

**Production status in 2026**:
- **Qdrant**: shipped via `FastEmbed 0.7.2` (Sep 5, 2025) as a post-processing wrapper around any multi-vector model (`Muvera.from_multivector_model()`); ~7x faster than full multi-vector search on `BeIR/nfcorpus` (0.18s vs 1.27s) with MUVERA-only achieving ~70% of full quality and **MUVERA + rerank recovering 99% of NDCG@10** (0.343 vs 0.347).
- **Weaviate 1.31+**: native integration with user-tunable `k_sim` / `d_proj` / `r_reps`; on LoTTE lifestyle (~119k docs, 130 vectors/doc) reports **70-80% memory reduction** (12GB → <1GB), 3-6 minute import vs 20+ minute baseline.
- **Vespa**: doesn't expose MUVERA by name but achieves a comparable result via **binary-quantised hamming MaxSim** for stage-1 + float MaxSim for stage-2 (32x memory reduction, 3.5x faster MaxSim, 49.5 vs 52.4 nDCG@5 on DocVQA pre-rerank, 51.6 post-rerank).
- **Pinecone**: positions multi-vector models as rerankers only inside "cascading retrieval"; published the **ConstBERT** paper as their preferred multi-vector compression direction.
- **Rust ecosystem**: `muvera-rs` (NewBornRustacean) and `ruvector` (gist by ruvnet) are unofficial but functional Rust implementations of the FDE construction.

**Successors to MUVERA**: there isn't a clean successor yet. The 2025-2026 research direction is **composing MUVERA with**:
- **Binary quantisation** of the FDE output (Vespa-style, ~32x further compression).
- **Token pooling** *before* FDE (reduces input set size).
- **Reranking with the original multi-vectors** (this is the canonical recommended deployment).

**Takeaway for Veda**: `veda-query::muvera.rs` is well-placed. Three improvements are worth doing:
1. Expose `k_sim` / `d_proj` / `r_reps` per collection (Weaviate parity).
2. Stack MUVERA → binary quantisation → HNSW for stage-1 (Vespa-style) for collections above ~10M docs.
3. Mandate a rerank stage in `local_shard.rs` when MUVERA is the stage-1 path; bare-MUVERA at ~70% quality is not acceptable for default Veda behaviour.

## 5. ESPRESSO / coffe / other recent IR research

The name "ESPRESSO" in 2025-2026 IR research refers to a **fixed-length video projector** (spatial + temporal token compression for long video understanding) — not directly a multi-vector retrieval engine. It's relevant to Veda only if a "Veda for video" multimodal extension is on the roadmap.

More relevant recent work:
- **`PRUNE-THEN-MERGE`** ([arXiv 2602.19549](https://arxiv.org/abs/2602.19549)): two-stage compression for visual document retrieval — refine first (remove noise patches), merge second (cluster + pool). **+10 percentage points of near-lossless compression range** over plain pooling.
- **`Col-Bandit`** ([arXiv 2602.02827](https://arxiv.org/abs/2602.02827)): zero-shot query-time pruning for late interaction — multi-armed bandit picks which document tokens to score for each query token; **5x fewer MaxSim FLOPs** at iso-quality on text and multimodal benchmarks.
- **`Working Notes on Late Interaction Dynamics`** ([arXiv 2603.26259](https://arxiv.org/abs/2603.26259)): empirical study of length bias in MaxSim, motivates moving beyond pure best-score pooling.
- **`ColBERT-Att`** ([arXiv 2603.25248](https://arxiv.org/abs/2603.25248)): merges late interaction with attention, ECIR-2026-track work.
- **LIR workshop @ ECIR 2026** ([arXiv 2511.00444](https://arxiv.org/abs/2511.00444)): the field's first dedicated workshop. Explicit themes: efficiency, system integration, cross-modal reasoning retrieval, negative results.

The signal: **2026 IR research is mostly engineering / compression / pruning, not new training objectives**. WARP, ConstBERT, MUVERA, token pooling, and Col-Bandit are all in the same family — different ways to throw away redundant work at index or query time.

## 6. Single-vector vs late-interaction: when to use which (2026 view)

| Axis | Single-vector dense | Late interaction |
|------|---------------------|------------------|
| Storage per doc | 1 vector × dim (~1.5-4 KB) | N vectors × dim (~16-100 KB raw, 1-4 KB with MUVERA+BQ) |
| Query latency (1M docs, HNSW) | ~1-5 ms | ~5-50 ms (MUVERA), ~50-500 ms (raw MaxSim) |
| Out-of-domain robustness | Weak — needs domain fine-tune | Strong — token-level matching transfers |
| Long-document retention | Loses information at chunk boundaries | Captures every token / patch |
| Visual document quality | Catching up (jina-v4, Cohere Embed 4 within 1-2 pts on ViDoRe v1) | Still leads on ViDoRe v2, table/chart-heavy corpora |
| Implementation complexity | Trivial (ANN over flat vectors) | High (MaxSim, compression, two-stage pipelines) |

**2026 consensus**: use **single-vector dense** for the default case, **late interaction as a reranker** in cascading retrieval (Pinecone's published stance, and what Vespa/Qdrant/Weaviate all recommend for production at scale), **late interaction as primary** only for visually-rich documents with tables/charts/diagrams or for highly out-of-domain corpora where single-vector models haven't been fine-tuned.

For Veda, this argues for a **default cascading pipeline**: BM25 or single-vector → MUVERA-compressed multi-vector candidate set → full MaxSim rerank on top-k. The schema supports this today; what's missing is the orchestration in the query planner.

## 7. Document-image retrieval alternatives to ColPali

The three architectures dominating 2026 visual document RAG:
1. **Caption-and-index** (simplest): VLM generates a caption, embed the caption with a text model. Cheap, lossy.
2. **Unified vision embeddings**: single dense vector for the full page image. Cohere Embed 4, Voyage Multimodal 3.5, jina-embeddings-v4 (single-vector mode), `MrLight/dse-qwen2-2b-mrl-v1`, `MrLight/dse-phi35-vidore-ft`.
3. **Page-as-image with late interaction**: ColPali, ColQwen2.5, ColModernVBERT, ColNomic, ColSmolVLM.

**DSE (Document Screenshot Embedding)** family: bi-encoder, single-vector per page screenshot, treats the document as an image. Simpler than ColPali, no MaxSim required, but loses the fine-grained patch-level matching. On ViDoRe v1, DSE-Qwen2 is competitive; on ViDoRe v2 it trails ColQwen2.5 by 3-5 points.

**Donut / DocLLM** are *layout-aware document understanding* models — they're upstream of retrieval (text extraction + structure), not direct retrieval alternatives. Veda's existing FlashParse → markdown → chunk → embed pipeline serves the same purpose.

**Security note**: DSE and ColPali are both **vulnerable to pixel poisoning attacks** ([arXiv 2501.16902](https://arxiv.org/abs/2501.16902)) — imperceptible image perturbations can bias retrieval. Worth a defensive note in Veda's threat model if these encoders are exposed to user-supplied images.

## 8. Production vector DB support for late interaction in 2026

| System | Multi-vector schema | Native MaxSim | MUVERA / FDE | Binary-quantised MaxSim | Notes |
|--------|---------------------|---------------|--------------|--------------------------|-------|
| **Vespa** | Mixed dense + sparse tensor | Tensor expression in rank profile | No (does BQ + hamming MaxSim instead) | Yes, the canonical recipe | Scales ColPali to billions; in-content-node compute |
| **Qdrant** | Named vectors, multivector field with `MaxSim` comparator | Yes | FastEmbed post-processor (0.7.2+, Sep 2025) | Yes, BQ on multivector | Two-stage prefetch+rerank API; ECIR 2025 ColPali blog |
| **Weaviate 1.31+** | Multi-vector embeddings + ColBERT/ColPali/ColQwen tutorials | Yes | Native config (`k_sim`, `d_proj`, `r_reps`) | In progress | Public MUVERA benchmark on LoTTE |
| **Milvus 2.5+** | Multi-vector hybrid search, ColPali docs | Via reranker plugin | Not yet native | Yes, on dense fields | Strong multimodal hybrid; multi-field ANN |
| **LanceDB** | Multi-vector record support | Yes (post-rerank step) | Not yet native | Yes | Best columnar-storage story; multimodal lakehouse positioning |
| **Pinecone** | Cascading retrieval pipeline | As reranker only | Not exposed; ConstBERT research direction | Proprietary sparse encoding | Managed-only; explicit "multi-vector is not stage-1" stance |
| **pgvector 0.9** | Single-vector only | N/A | N/A | scalar quantisation | No multi-vector path |
| **Veda** | `MultiVector` field type | `local_shard.rs` MaxSim | `veda-query::muvera.rs` | Not yet | PLAID-style engine in `veda-query::plaid.rs` |

The two systems Veda should benchmark against directly are **Qdrant** (closest in shape — Rust, multi-vector + named-vectors, MUVERA postprocessor) and **Vespa** (closest in production-scale ambition — billions of ColPali pages).

## 9. MaxSim scoring optimisations

The state of MaxSim execution in 2026:

- **`maxsim-cpu`** (Mixedbread, [blog](https://www.mixedbread.com/blog/maxsim-cpu)): Rust + `libxsmm` (AVX2 small-matrix-mul kernels) on x86; custom Apple Silicon path. **Reduces per-query overhead from 50-100 ms to ~5 ms on ~1000 docs** vs PyTorch CPU. Fused operations, no intermediate distance materialisation, custom variable-length batch handling (no padding waste).
- **GPU MaxSim**: large-batch is where GPUs win; CPU is competitive at low batch / low latency. The crossover for ColBERT-128-dim is around batch size 32 on modern AVX-512.
- **Approximate MaxSim** via Col-Bandit (5x fewer FLOPs), token pooling (3x fewer tokens), or MUVERA FDE approximation (1 inner product instead of N×M MaxSim).
- **Quantised MaxSim**: Vespa's binary-binary hamming MaxSim is **3.5x faster than float dot product** and only **2.9 nDCG@5 worse** pre-rerank.

**Takeaway for Veda**: `veda-core::simd` already exists for distance functions but Grep shows MaxSim lives in `veda-query::local_shard.rs` rather than as a SIMD kernel in `veda-core`. Moving the inner MaxSim loop to a `veda-core::simd::maxsim` module with AVX2/NEON paths is the obvious follow-up. This is a self-contained ~500-line change.

## 10. Tooling: PyLate, RAGatouille, colpali-engine

For training and evaluation (not runtime), the 2026 standard tools are:
- **PyLate** (LightOn, [arXiv 2508.03555](https://arxiv.org/abs/2508.03555)) — Sentence-Transformers wrapper for fine-tuning ColBERT-style models on any HF base; built `GTE-ModernColBERT`, `Reason-ModernColBERT`, `mxbai-edge-colbert`.
- **RAGatouille** (Answer.AI) — `pip install ragatouille`, indexes-on-disk, "very viable production deployment" per the author. Good for evaluating Veda's quality against an off-the-shelf reference.
- **`colpali-engine`** (ILLuin) — canonical training + inference path for ColPali / ColQwen2 / ColQwen3 / ColSmol.

Veda doesn't need to host these inside the crates; an `embedding-providers` config that calls `colpali-engine` over the docker-compose embedding profile is the pragmatic integration.

---

## Recommended action items for Veda (prioritised)

1. **Add ColQwen2.5 + ColModernVBERT model adapters** to the embedding gateway. No schema change; the `MultiVector` field type is general. (Quality win: +5-7 nDCG@5 on visual docs.)
2. **Swap PLAID engine semantics to WARP** inside `veda-query::plaid.rs` (`WARP_SELECT`, implicit decompression, two-stage reduction). (Perf win: ~3x QPS, ~50% memory at iso-quality.)
3. **Move MaxSim into `veda-core::simd`** with AVX2/NEON paths, modelled on `maxsim-cpu`. (Perf win: ~10-20x CPU MaxSim.)
4. **Expose MUVERA `k_sim` / `d_proj` / `r_reps` per collection** (Weaviate-parity API). Default to a profile that targets 70-80% memory reduction + rerank.
5. **Add MUVERA + binary-quantised hamming MaxSim as a stacked stage-1** (Vespa-parity) for >10M-doc collections.
6. **Add token pooling as an optional index-time pre-processor** for multi-vector collections (one-week project, ~60% storage reduction).
7. **Add a defensive note in the threat model** about pixel poisoning attacks against ColPali/DSE-class encoders.
8. **Future watch**: ConstBERT (learned fixed-set pooling), HEAVEN (VS-Pages stage-1), and the ECIR 2026 LIR workshop proceedings (expected June 2026).

The architecture choices Veda made in v0.1.0 still hold; the engines underneath need a 2026 refresh.

---

## Sources

- ColPali — Faysse et al., [arXiv 2407.01449](https://arxiv.org/abs/2407.01449)
- ColBERTv2 — Santhanam et al., [arXiv 2112.01488](https://arxiv.org/abs/2112.01488)
- PLAID — Santhanam et al., [arXiv 2205.09707](https://arxiv.org/abs/2205.09707)
- MUVERA — Dhulipala et al., [arXiv 2405.19504](https://arxiv.org/abs/2405.19504), [Google Research blog](https://research.google/blog/muvera-making-multi-vector-retrieval-as-fast-as-single-vector-search/)
- WARP — Scheerer et al., [arXiv 2501.17788](https://arxiv.org/abs/2501.17788), [repo](https://github.com/jlscheerer/xtr-warp)
- XTR — Lee et al., [arXiv 2304.01982](https://arxiv.org/abs/2304.01982)
- EMVB — Rulli et al., [arXiv 2404.02805](https://arxiv.org/abs/2404.02805)
- ConstBERT — Frei et al., [arXiv 2504.01818](https://arxiv.org/abs/2504.01818)
- Token pooling — Clavié et al., [arXiv 2409.14683](https://arxiv.org/abs/2409.14683)
- Jina-ColBERT-v2 — Jha et al., [arXiv 2408.16672](https://arxiv.org/abs/2408.16672)
- Jina-embeddings-v4 — [arXiv 2506.18902](https://arxiv.org/abs/2506.18902), [release](https://jina.ai/news/jina-embeddings-v4-universal-embeddings-for-multimodal-multilingual-retrieval/)
- mxbai-edge-colbert — [Mixedbread blog](https://www.mixedbread.com/blog/edge-v0), [arXiv 2510.14880](https://arxiv.org/abs/2510.14880)
- maxsim-cpu — [Mixedbread blog](https://www.mixedbread.com/blog/maxsim-cpu)
- ModernVBERT / ColModernVBERT — Teiletche et al., [arXiv 2510.01149](https://arxiv.org/abs/2510.01149)
- HEAVEN — [arXiv 2510.22215](https://arxiv.org/abs/2510.22215)
- DSE — [arXiv 2406.11251](https://arxiv.org/abs/2406.11251); pixel poisoning [arXiv 2501.16902](https://arxiv.org/abs/2501.16902)
- ColPali Hierarchical Patch Compression — [arXiv 2506.21601](https://arxiv.org/abs/2506.21601)
- Vespa scaling ColPali — [Vespa blog](https://blog.vespa.ai/scaling-colpali-to-billions/)
- Qdrant MUVERA — [Qdrant article](https://qdrant.tech/articles/muvera-embeddings/)
- Weaviate MUVERA — [Weaviate blog](https://weaviate.io/blog/muvera)
- Pinecone cascading retrieval — [Pinecone blog](https://www.pinecone.io/blog/cascading-retrieval-with-multi-vector-representations/)
- LanceDB late-interaction post — [LanceDB blog](https://www.lancedb.com/blog/late-interaction-efficient-multi-modal-retrievers-need-more-than-just-a-vector-index)
- LIR workshop @ ECIR 2026 — [arXiv 2511.00444](https://arxiv.org/abs/2511.00444)
- PyLate — [arXiv 2508.03555](https://arxiv.org/abs/2508.03555)
- answerai-colbert-small — [Answer.AI blog](https://www.answer.ai/posts/2024-08-13-small-but-mighty-colbert.html)
- Voyage Multimodal 3.5 — [Voyage AI blog](https://blog.voyageai.com/2026/01/15/voyage-multimodal-3-5/)

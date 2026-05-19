# SOTA Audit: Text Embedding Models for Veda (2026-05-19)

**Scope.** This audit evaluates Veda's current default text embedding model (BGE-M3, served via HuggingFace TEI on CPU) against the 2026 frontier and recommends a forward-looking model strategy for the `EmbeddingModelRegistry`. Every model cited includes a version, release date, and the source benchmark.

---

## Verdict (TL;DR)

**Is BGE-M3 still defensible as Veda's default in 2026?** Partially. As a *baseline* it is acceptable; as a *flagship default* it is no longer SOTA.

- BGE-M3 (BAAI, released Jan 30, 2024) sits around **MTEB-EN ~63.0** on the v1 leaderboard, roughly **~9 points behind the open-weight frontier** (Qwen3-Embedding-8B at 70.58 multilingual, NV-Embed-v2 at 72.31 English MTEB v1) and **~5 points behind the Gemini/Voyage proprietary frontier** (Gemini-Embedding-001 at 68.32, voyage-3-large at ~66.8) (sources: BAAI/bge-m3 HF card; awesomeagents.ai MTEB April 2026 snapshot; NVIDIA blog; Voyage blog Jan 2025; Google Gemini Embedding paper arXiv:2503.07891).
- BGE-M3's *unique* selling points — dense + sparse + multi-vector outputs in one model, 8192-token context, 100+ languages, Apache-2.0 license, runs on CPU — remain genuinely differentiated. None of the frontier models offer all four properties simultaneously.
- BGE-M3 is therefore defensible as a **floor / CPU-friendly default**, but Veda should ship at least one frontier text model in the `EmbeddingModelRegistry` for users who want top-quality retrieval.

**Top 3 alternatives Veda should support (in priority order):**

1. **Qwen3-Embedding (0.6B / 4B / 8B family)** — Apache-2.0 open-weight, MRL-truncatable, 32K context, ranked #1 on MMTEB at release (June 2025, score 70.58), still in the top 4 as of April 2026, has 3 size tiers that mirror Veda's "small/medium/large" deployment story, and is fully supported by TEI and vLLM. This is the single highest-impact addition.
2. **Voyage-3-large** (or its successor voyage-3.5) — proprietary API but Anthropic's officially recommended embedding partner; 32K context; Matryoshka 2048/1024/512/256; quantization-aware training enabling int8 and binary at ~1/200th storage; ~9.74% over OpenAI-3-large average across 100 datasets. This is the right "managed/proprietary" option for customers who do not want to self-host.
3. **Gemini-Embedding-001** — Google's API-only flagship, MTEB-Multilingual #1 at 68.32 (release; +5.09 above second-best), 100+ languages, 3072-dim Matryoshka down to 768. Best choice when the user is already on GCP / Vertex AI.

**If you had to pick one in 2026:** **Qwen3-Embedding-8B** for self-hosted Veda deployments and **voyage-3-large** (or voyage-3.5) for API-only deployments. Qwen3-8B is the right *default* upgrade because it is open-weight, Apache-2.0, runs on TEI/vLLM, supports MRL truncation (so Veda's existing 1024-dim storage can be preserved), is multilingual, and is within striking distance of the proprietary frontier. The tradeoff is GPU memory: 8B params at fp16 needs ~16 GB VRAM — meaningfully heavier than BGE-M3's ~2.3 GB. For CPU-only deployments, keep BGE-M3 (or move to Qwen3-Embedding-0.6B which is the same architecture at a fraction of the cost).

---

## 1. The 2026 Leaderboard Landscape

The benchmark to use as of May 2026 is **MMTEB** (Massive Multilingual Text Embedding Benchmark, OpenReview zl3pfz4VCV, arXiv:2502.13595), which superseded the original MTEB by expanding to >250 languages and >500 tasks. Aggregation is by Borda count, which penalises models that win only on a few tasks. MTEB-EN v1 (the legacy 56-task English benchmark) is still cited but should be treated as a sub-benchmark, not the headline.

Two caveats matter for any number quoted below:

1. **MTEB v2 scores are not comparable to MTEB v1.** Papers and blog posts conflate them constantly; readers must check which version produced a given number.
2. **Self-reported scores dominate the leaderboard.** Model authors submit their own results; the evaluation code is open but there is no independent re-running.

### English MTEB (v1) — April 2026 snapshot

Per the *Awesome Agents* April 2026 leaderboard mirror and corroborated by the underlying HF model cards:

| Rank | Model | MTEB-EN | Params | Dim | Max tokens | License |
|------|-------|---------|--------|-----|------------|---------|
| 1 | NV-Embed-v2 (NVIDIA, Sep 2024) | 72.31 | 7B | 4096 | 32K | **CC-BY-NC-4.0 (non-commercial)** |
| 2 | BGE-en-ICL (BAAI, late 2024) | 71.24 | 7B | 4096 | 32K | Apache-2.0 |
| 3 | Qwen3-Embedding-8B (June 5, 2025) | 70.58 (multilingual) | 8B | 4096 | 32K | Apache-2.0 |
| 4 | GTE-Qwen2-7B-instruct (Alibaba, June 2024) | 70.24 | 7B | 3584 | 32K | Apache-2.0 |
| 5 | Gemini Embedding 001 (GA July 14, 2025) | 68.32 (Multilingual) | n/d | 3072 (MRL) | 2048 | **Proprietary API** |
| 6 | voyage-3.1-large (Voyage AI) | 67.40 | n/d | 2048 | 32K | **Proprietary API** |
| 7 | Jina Embeddings v4 (June 2025) | 66.81 / 66.49 MMTEB | 3.8B | 1024+ML | 32K | Apache-2.0 (weights) + API |
| 8 | voyage-3-large (Jan 7, 2025) | 66.80 | n/d | 2048 | 32K | **Proprietary API** |
| 9 | Cohere Embed v4 (Apr 15, 2025) | 65.20 | n/d | 1024 (MRL 256/512/1024/1536) | 128K | **Proprietary API** |
| 10 | OpenAI text-embedding-3-large (Jan 25, 2024) | 64.60 | n/d | 3072 (MRL) | 8191 | **Proprietary API** |
| 11 | **BGE-M3 (BAAI, Jan 30, 2024)** | **63.00** | 568M | 1024 | 8192 | **Apache-2.0** |
| 12 | Nomic Embed v1.5 | 62.39 | 137M | 768 | 8192 | Apache-2.0 |
| 13 | OpenAI text-embedding-3-small | 62.26 | n/d | 1536 (MRL) | 8191 | Proprietary API |

Sources: awesomeagents.ai April-2026 leaderboard; HF model cards for Qwen/Qwen3-Embedding-8B, nvidia/NV-Embed-v2, BAAI/bge-m3, jinaai/jina-embeddings-v4, Voyage blog 2025-01-07 and 2025-05-20; OpenAI 2024-01-25 launch post; Cohere Embed v4 changelog; Google developers blog 2025-07-14.

### Multilingual MMTEB — late-2025 / 2026 snapshot

For multilingual retrieval the picture differs:

- **Llama-Embed-Nemotron-8B** (NVIDIA, fine-tune of Llama-3.1-8B, 4096-dim bidirectional, SOTA on multilingual MTEB as of October 21, 2025) — **non-commercial only** (Customized NSCLv1 + Llama-3.1 Community License).
- **Qwen3-Embedding-8B** — Apache-2.0, score 70.58, French 69.8; remains the best commercially-usable open-weight multilingual model.
- **Gemini Embedding 001** — top-1 on the public MMTEB leaderboard at experimental launch with 68.32 mean and +5.09 lead over multilingual-e5-large-instruct (per the Gemini Embedding paper, arXiv:2503.07891).
- **BGE-M3** — still strong for cross-lingual retrieval; rivals most commercial services on MIRACL-style queries, but is no longer leading on aggregate MMTEB.

---

## 2. Per-model Analysis

### 2.1 BGE-M3 (Veda's current default)

- **Release:** Jan 30, 2024, BAAI; paper arXiv:2402.03216.
- **Architecture:** XLM-RoBERTa-large fine-tune, 568M params.
- **Outputs:** Dense (1024-dim), sparse (lexical weights), multi-vector (ColBERT-style late interaction) — all from one forward pass. This is unique among production embedding models. No other model on this list ships sparse + multi-vector + dense in a single artifact.
- **Context:** 8192 tokens.
- **Languages:** 100+, very strong on Chinese and low-resource languages.
- **License:** MIT for code, Apache-2.0 for weights.
- **Serving:** First-class on TEI (huggingface/text-embeddings-inference); CPU inference is the headline use case (which is why Veda picked it). vLLM support landed in 2024.
- **MTEB-EN:** 63.0. Solid but eclipsed by the 7B/8B generation.
- **Pros for Veda:** triple-output (dense + sparse + multi-vector) is exactly what Veda's hybrid retrieval needs without running two models; long context; commercial-friendly license; CPU-runnable.
- **Cons for Veda:** roughly 7-9 MTEB points behind the frontier; dimension 1024 is fine but no MRL truncation; outdated training mix (no Llama-3 / Qwen-3 era synthetic data).

**Verdict:** keep as floor option, especially for CPU-only / hybrid-retrieval users; not defensible as the sole default in 2026.

### 2.2 Qwen3-Embedding family (0.6B / 4B / 8B)

- **Release:** June 5, 2025; paper arXiv:2506.05176; QwenLM/Qwen3-Embedding on GitHub.
- **Architecture:** Qwen3 LLM backbone, instruction-tunable via task prompts.
- **Sizes:** 0.6B (1024-dim, 28 layers, fastest), 4B (intermediate), 8B (4096-dim, top-ranked).
- **Context:** 32K tokens (vs BGE-M3's 8K).
- **MTEB:** 8B variant ranked **#1 on MMTEB multilingual at 70.58** at release; still in top 4 as of April 2026.
- **License:** Apache-2.0 across the family.
- **MRL:** Yes — trained with Matryoshka representation learning, so vectors can be truncated at inference time without re-embedding.
- **Serving:** TEI support, vLLM support, GGUF quantizations published officially by Qwen; works in Milvus tutorials and Ollama out of the box.
- **Languages:** 100+, including code retrieval as a first-class task.
- **Tradeoff:** 8B at fp16 is ~16 GB VRAM; quantized GGUF brings this down to ~5-6 GB for the Q4_K_M variant.

**Verdict:** **highest-impact open-weight addition for Veda.** Add all three sizes to `EmbeddingModelRegistry` so users can pick by deployment budget; recommend 8B as the new flagship and 0.6B as the CPU-replacement for BGE-M3.

### 2.3 NV-Embed-v2 (NVIDIA, Mistral-7B base)

- **Release:** Sep 2024; paper arXiv:2405.17428 (v3).
- **MTEB-EN v1:** 72.31 — held #1 at release; retrieval sub-score 62.65 on 15 tasks.
- **Architecture:** Mistral-7B with bidirectional attention, latent attention pooling.
- **License:** **CC-BY-NC-4.0 (non-commercial).** NVIDIA explicitly states it should not be used for commercial purposes and points commercial users to NeMo Retriever NIM (nv-embedqa-mistral-7b-v2).
- **Implication for Veda:** *Do not ship NV-Embed-v2 as a default.* Mention it in the registry as an "evaluation-only / research" option, but block it from production registration. The commercial NIM equivalent (nv-embedqa-mistral-7b-v2) is API-only via build.nvidia.com — a poor fit for self-hosted Veda.

### 2.4 Llama-Embed-Nemotron-8B (NVIDIA, Oct 2025)

- **Architecture:** Llama-3.1-8B with bidirectional attention, 4096-dim.
- **Score:** SOTA on multilingual MTEB as of Oct 21, 2025.
- **License:** Customized NSCLv1 + Llama-3.1 Community License — **non-commercial / research only.**
- **Implication for Veda:** same as NV-Embed-v2 — research evaluation only.

### 2.5 Voyage-3-large and Voyage-3.5

- **voyage-3-large release:** Jan 7, 2025. **voyage-3.5 / voyage-3.5-lite:** May 20, 2025. **voyage-context-3:** July 23, 2025 (adds chunk-level + global document context).
- **MTEB-EN v1:** voyage-3-large 66.80; voyage-3.1-large 67.40 (per April 2026 snapshot).
- **Context:** 32K tokens.
- **Dimensions:** Matryoshka 2048 / 1024 / 512 / 256; quantization-aware training enables int8 and binary precision.
- **Headline claim:** binary 512-dim variant matches OpenAI-3-large's float 3072-dim at ~**1/200th storage cost**.
- **Pricing:** voyage-3-large $0.18 / 1M tokens; voyage-3.5 $0.06 / 1M; voyage-3.5-lite $0.02 / 1M.
- **License:** **Proprietary API.** No open weights. Anthropic-affiliated and Anthropic's officially recommended embeddings provider.
- **Serving:** API only; not deployable inside Veda.

**Verdict:** **best proprietary option to wire up in `EmbeddingModelRegistry`**, primarily because (a) Voyage publishes Matryoshka + quantization specs that map directly to Veda's RaBitQ/SQ8/binary index path, (b) the price-per-quality is class-leading at the 3.5-lite tier, and (c) Anthropic users will expect this integration.

### 2.6 Gemini Embedding 001 (and Embedding 2 preview)

- **gemini-embedding-001:** experimental launch March 2025; GA July 14, 2025.
- **gemini-embedding-2-preview:** March 10, 2026 — Google's first natively multimodal embedding model (text + image + video + audio + document into one space).
- **MTEB-Multilingual:** 68.32 mean, +5.09 above the second-best at release; top-1 on Borda rank (Gemini Embedding paper, arXiv:2503.07891).
- **Dimensions:** Default 3072; MRL down to 1536 / 768.
- **Context:** 2048 tokens — **notably short.** This is the key weakness vs Voyage / Cohere / Qwen3, all of which offer 8K-128K.
- **Languages:** 100+.
- **Pricing:** $0.15 / 1M input tokens.
- **License:** **Proprietary API** (Gemini API + Vertex AI). No open weights, no self-hosting.

**Verdict:** worth adding to `EmbeddingModelRegistry` for the GCP-native customer segment. The 2048-token cap is awkward for the document-chunking pipeline Veda already runs (default 2000-char MarkdownRecursive chunks usually fit, but the 32K headroom of competitors is gone). The upcoming Embedding 2 is the more interesting target because it unifies modalities — relevant to Veda's multimodal positioning.

### 2.7 Cohere Embed v4 (and Embed v3 multilingual)

- **Release:** April 15, 2025; Bedrock + Azure AI Foundry + Cohere Platform.
- **MTEB:** 65.20 (ahead of OpenAI-3-large at 64.60).
- **Context:** **128K tokens** — by far the longest of any production embedding model in 2026.
- **Dimensions:** Matryoshka 256 / 512 / 1024 / 1536.
- **Languages:** 100+, with cross-lingual semantic search in a single index.
- **Modalities:** multimodal (text + image + interleaved).
- **License:** **Proprietary API.**

**Verdict:** Cohere Embed v4 is the obvious "long-document" model in the registry — wire it in if any Veda customer is ingesting >32K-token documents without chunking. Below 32K, voyage-3.5 wins on price and Qwen3-8B wins on openness.

### 2.8 Jina Embeddings v3 and v4

- **v3 (Sep 2024, arXiv:2409.10173):** 570M params, 8192 tokens, multilingual, task LoRA adapters; outperforms OpenAI/Cohere on English at release.
- **v4 (June 2025, arXiv:2506.18902):** **3.8B params**, 32K context, **unifies text + image**, supports **both single-vector and ColBERT-style multi-vector outputs** (multi-vector tokens are 128-dim each), task-specific LoRA adapters (~60M params each, all three loadable with <2% memory overhead). MMTEB 66.49 (+14% over v3); SOTA on ViDoRe visual document retrieval.
- **License:** Apache-2.0 weights + API tiers.
- **Serving:** TEI support, HF native.

**Verdict:** **strongest "drop-in v4 / ColBERT-style" model for Veda's multi-vector index path.** If Veda wants to lean into its multi-vector / late-interaction index work, Jina v4 is the cleanest open-weight bet. It does *not* outperform Qwen3-8B on MTEB single-vector, so it complements rather than replaces Qwen3.

### 2.9 OpenAI text-embedding-3-large / -small

- **Release:** Jan 25, 2024.
- **text-embedding-3-large:** 3072-dim with MRL truncation; 8191-token context; MTEB-EN 64.60.
- **text-embedding-3-small:** 1536-dim with MRL truncation; MTEB-EN 62.26.
- **Pricing:** $0.13 / 1M (3-large), $0.02 / 1M (3-small).
- **License:** **Proprietary API.**
- **2026 status:** OpenAI has **not** released a "text-embedding-4" series at the time of writing. The -3 family is the live API. This is now the *worst* value-for-quality among proprietary options in 2026 — Voyage-3.5 matches the price and beats the quality, Cohere v4 has 16x the context, Gemini is +3.7 MTEB points.

**Verdict:** keep in registry for compatibility with existing OpenAI-pipeline customers; do not recommend as a default.

### 2.10 Stella-1.5B-V5 and Jasper

- **Stella 1.5B v5 (NovaSearch):** Apache-2.0, MRL dimensions 512 / 768 / 1024 / 2048 / 4096 / 6144 / 8192. The 1024-dim MTEB is within 0.001 of 8192-dim — exceptional MRL collapse.
- **Jasper** (Dec 2024, arXiv:2412.19048): distilled from Stella, **#3 on MTEB at release, score 71.54.**
- **License:** Apache-2.0.

**Verdict:** Stella/Jasper is the strongest "small + Matryoshka" open-weight model and a clean replacement for BGE-M3's CPU role if Qwen3-0.6B feels too LLM-flavored. Worth shipping in the registry as a "compact + MRL" option.

### 2.11 GTE-Qwen2-7B-instruct

- **Release:** June 16, 2024 (Alibaba NLP).
- **MTEB:** 70.24 on the v1 leaderboard, retrieval nDCG@10 60.25; ranked #1 English + #1 Chinese at release.
- **Architecture:** Qwen2-7B with bidirectional attention; 3584-dim, 32K context.
- **License:** Apache-2.0.

**Verdict:** essentially superseded by Qwen3-Embedding (same lineage, newer base model). Keep in registry only if existing Veda users have indexes built with it.

### 2.12 Linq-Embed-Mistral and SFR-Embedding-Mistral

- **SFR-Embedding-Mistral (Salesforce, early 2024):** MTEB-EN 67.6 at release.
- **Linq-Embed-Mistral (May 29, 2024):** further fine-tune of E5-mistral-7b-instruct with refined data crafting; top open-source on MTEB at release.

**Verdict:** both are now ~2 years old and beaten by Qwen3/NV-Embed. Historical interest only.

### 2.13 BGE-Multilingual-Gemma2

- **Release:** July 26, 2024.
- **Architecture:** Gemma-2-9B base.
- **Score:** SOTA on multilingual benchmarks (MIRACL, MTEB-fr, MTEB-pl) at release.
- **License:** Gemma terms.
- **Implication:** This is **not** a direct successor to BGE-M3 — it is a complementary multilingual model in the BGE family. BAAI's public 2026 roadmap does not yet name a BGE-M4. The closest "next-gen BGE" is **BGE-en-ICL** at MTEB-EN 71.24.

---

## 3. Comparison Across the Required Axes

### Context length
- 128K: Cohere Embed v4 (unique)
- 32K: Voyage-3-large/3.5, Qwen3-Embedding family, Jina v4, GTE-Qwen2-7B, NV-Embed-v2
- 8K: BGE-M3, Jina v3, OpenAI -3 family, Stella, Nomic v1.5
- 2K: Gemini Embedding 001 (notable weakness)

### Matryoshka / truncation
- Voyage-3-large, Voyage-3.5, Cohere v4, Gemini Embedding 001, OpenAI -3 family, Qwen3-Embedding, Stella, Jina v4 — all native MRL.
- BGE-M3 — no MRL (fixed 1024-dim dense).

### Multimodal in-band
- Cohere Embed v4, Jina v4, Gemini Embedding 2 (preview): text + image (+ more) into one vector space.
- All other text models are text-only. (Veda already routes non-text modalities to `Qwen3-VL-Embedding-2B` for the multimodal pipeline, so a text-only model is the right slot here.)

### Inference cost — open-weight self-hosted
| Model | Params | fp16 VRAM | Q4 VRAM | TEI / vLLM |
|-------|--------|-----------|---------|------------|
| BGE-M3 | 568M | ~2.3 GB | ~0.5 GB | both |
| Qwen3-Embedding-0.6B | 600M | ~1.2 GB | ~0.4 GB | both |
| Qwen3-Embedding-4B | 4B | ~8 GB | ~2.5 GB | both |
| Qwen3-Embedding-8B | 8B | ~16 GB | ~5 GB | both |
| Stella-1.5B-V5 | 1.5B | ~3 GB | ~1 GB | both |
| Jina v4 | 3.8B | ~8 GB | ~2.5 GB | TEI |
| BGE-en-ICL | 7B | ~14 GB | ~4.5 GB | both |
| NV-Embed-v2 | 7B | ~14 GB | ~4.5 GB | TEI / vLLM (research only) |
| GTE-Qwen2-7B | 7B | ~14 GB | ~4.5 GB | both |

INT8 / binary quantization at the *vector* level (orthogonal to model quantization) cuts storage by 4-32x with <0.5% MTEB drop per HuggingFace's quantization benchmark — Veda already has the SQ8/RaBitQ index types to exploit this; the missing piece is asking embedding models to emit quantization-friendly vectors, which Voyage and Stella do natively.

### Licensing for commercial production
- **Apache-2.0 / MIT (safe):** BGE-M3, BGE-en-ICL, Qwen3-Embedding (all sizes), GTE-Qwen2-7B, Stella, Jina v3/v4 weights, BGE-Multilingual-Gemma2 (Gemma terms — quasi-permissive).
- **Non-commercial only (do NOT default-ship):** NV-Embed-v2, Llama-Embed-Nemotron-8B.
- **Proprietary API only:** Gemini Embedding 001/2, Voyage-3-large/3.5, Cohere Embed v4, OpenAI -3 family.

---

## 4. Implications for Veda's `EmbeddingModelRegistry`

Concrete additions to consider (no code changes in this audit — research-only):

1. **Add Qwen3-Embedding-0.6B, -4B, -8B** as Apache-2.0 self-hosted options. Default for new GPU deployments should move to Qwen3-Embedding-8B (or 4B for memory-constrained sites). 0.6B replaces BGE-M3 as the new CPU default.
2. **Keep BGE-M3** as the "hybrid-retrieval default" specifically because of its sparse + multi-vector outputs — Veda's hybrid (dense + sparse) and multi-vector index paths benefit uniquely from BGE-M3. Document this clearly so users do not assume BGE-M3 is "the legacy option."
3. **Add Voyage-3.5 and Voyage-3-large** as proprietary API providers. The API integration also unlocks Voyage's int8/binary precision endpoint, which is the best match for Veda's RaBitQ/SQ8 storage paths.
4. **Add Gemini-Embedding-001** for GCP-native deployments. Document the 2048-token cap so the chunking pipeline (current default 2000-char MarkdownRecursive) is explicitly checked against it.
5. **Add Cohere Embed v4** for the 128K-context long-document use case.
6. **Add Jina Embeddings v4** as the open-weight ColBERT/multi-vector option (complement to BGE-M3 multi-vector; Jina v4 has 128-dim per-token vectors vs BGE-M3's ColBERT head).
7. **Add Stella-1.5B-V5** as the small-MRL open-weight option (1024-dim ~= 8192-dim quality).
8. **Block NV-Embed-v2 and Llama-Embed-Nemotron-8B** from the *default* registry path due to non-commercial license. Allow as research-mode toggles only.
9. **Demote OpenAI text-embedding-3-large/-small** from "recommended" to "compatibility." They are no longer competitive on quality-per-dollar in 2026.

### Risks to watch
- **Self-reported leaderboards.** Treat the +0.5-1 MTEB-point gaps between adjacent models as inside the noise floor; pick by license, context, and serving fit, not by raw MTEB rank.
- **MTEB v1 vs v2.** Cross-version comparisons mislead. When migrating Veda's docs/benchmarks, lock the version.
- **Tokenizer compatibility.** Qwen3-Embedding uses Qwen3's tokenizer; switching from BGE-M3's XLM-R tokenizer changes chunk sizing in practice. Re-tune the chunker defaults when changing default model.
- **Sparse + multi-vector outputs.** Only BGE-M3 ships all three from a single forward pass. If Veda drops BGE-M3 entirely, the hybrid + multi-vector pipelines will need a second model loaded alongside Qwen3 (e.g., SPLADE-v3 for sparse, Jina v4 for multi-vector), which is more memory and more orchestration. This is the single strongest argument for keeping BGE-M3 in the registry indefinitely.

---

## 5. Recommendation Summary

| Veda role | Today (2026-05-19) | After audit |
|-----------|--------------------|-------------|
| Default text embedding (GPU) | BGE-M3 | **Qwen3-Embedding-8B** (8K-32K context, MRL, Apache-2.0) |
| Default text embedding (CPU) | BGE-M3 | **Qwen3-Embedding-0.6B** or keep BGE-M3 if hybrid+multi-vector is in use |
| Hybrid-retrieval (dense+sparse+multi-vector) default | BGE-M3 | **Keep BGE-M3** — unique capability |
| Long-document (>32K) | n/a | **Cohere Embed v4** (128K) |
| Proprietary / managed | n/a | **voyage-3.5 / voyage-3-large** |
| GCP-native | n/a | **gemini-embedding-001** |
| Multi-vector open-weight | BGE-M3 ColBERT head | **Jina v4 multi-vector** as alternative |
| Compact + MRL | BGE-M3 | **Stella-1.5B-V5** |

**Bottom line.** BGE-M3 remains an excellent *specialised* tool for hybrid + multi-vector retrieval on CPU and should not be removed from Veda — but it should no longer be the *sole headline default*. The 2026 frontier for general-purpose text embedding is Qwen3-Embedding-8B (open-weight) and voyage-3-large / gemini-embedding-001 (proprietary). Veda's registry should expose all three lanes (open-weight flagship, proprietary frontier, BGE-M3 hybrid specialist) and let deployers choose.

---

## Sources

- MTEB / MMTEB leaderboard mirror, April 2026: <https://awesomeagents.ai/leaderboards/embedding-model-leaderboard-mteb-april-2026/>
- MMTEB paper: arXiv:2502.13595, OpenReview zl3pfz4VCV
- BGE-M3 model card: <https://huggingface.co/BAAI/bge-m3>; FlagEmbedding GitHub
- Qwen3-Embedding paper: arXiv:2506.05176; QwenLM/Qwen3-Embedding GitHub; HF cards for Qwen3-Embedding-0.6B / -4B / -8B
- NV-Embed-v2 model card: <https://huggingface.co/nvidia/NV-Embed-v2>; arXiv:2405.17428; NVIDIA blog "NVIDIA Text Embedding Model Tops MTEB Leaderboard"
- Llama-Embed-Nemotron-8B model card: <https://huggingface.co/nvidia/llama-embed-nemotron-8b>
- Voyage-3-large blog: <https://blog.voyageai.com/2025/01/07/voyage-3-large/>; Voyage-3.5 blog: <https://blog.voyageai.com/2025/05/20/voyage-3-5/>; voyage-context-3 blog (2025-07-23)
- Gemini Embedding paper: arXiv:2503.07891; Google Developers Blog (2025-07-14 GA announcement)
- Cohere Embed v4 changelog: <https://docs.cohere.com/changelog/embed-multimodal-v4>; AWS Bedrock docs for Embed v4
- Jina Embeddings v3 paper: arXiv:2409.10173; v4 paper: arXiv:2506.18902; Jina blog posts on v3/v4 and ColBERT v2
- OpenAI new embedding models announcement (2024-01-25): <https://openai.com/index/new-embedding-models-and-api-updates/>
- Stella 1.5B v5 model card: <https://huggingface.co/NovaSearch/stella_en_1.5B_v5>; Jasper/Stella paper arXiv:2412.19048
- GTE-Qwen2-7B-instruct: <https://huggingface.co/Alibaba-NLP/gte-Qwen2-7B-instruct>
- SFR-Embedding-Mistral: <https://huggingface.co/Salesforce/SFR-Embedding-Mistral>; Linq-Embed-Mistral: <https://huggingface.co/Linq-AI-Research/Linq-Embed-Mistral>
- BGE-Multilingual-Gemma2: <https://huggingface.co/BAAI/bge-multilingual-gemma2>
- HuggingFace embedding quantization blog (int8 + binary): <https://huggingface.co/blog/embedding-quantization>

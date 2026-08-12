# SOTA refresh — embeddings and late interaction

**Research date:** 2026-08-12. Updates
`research/2026-05-19-sota-audit/{03-text-embeddings,04-multimodal-embeddings,05-late-interaction}.md`.

**Veda baseline verified in-repo today** (`crates/veda-core/src/embedding.rs`):

| Kind | Presets |
|---|---|
| Text | `bge-m3` (1024), `qwen3-embedding-8b` (4096), `qwen3-embedding-4b` (2560), `qwen3-embedding-0.6b` (1024), `voyage-3-large` (1024), `voyage-3.5` (1024) |
| Multimodal | `qwen3-vl-embedding-2b` (2048), `qwen3-vl-embedding-8b` (3584), **`jina-v5-omni-small` (1024)** |

Deployed default in the live stack: `jina-v5-omni-small` for **both** `EMBEDDING_*` and
`MULTIMODAL_*`, giving one 1024-d cross-modal space.

---

## 1. Text embeddings — the frontier moved again; BGE-M3 fell further behind

| Model | Score | Board | Note |
|---|---|---|---|
| **KaLM-Embedding-Gemma3-12B** (Tencent) | **72.32** | MMTEB multilingual, **#1 open**, Jul 2026 | new since the May audit |
| Microsoft **Harrier-OSS-v1** (27B, MIT) | **74.3** | MTEB v2 | MIT licence — no NC trap |
| **Qwen3-Embedding-8B** | 70.58 | MTEB | **Veda ships this preset** |
| Google **gemini-embedding-001** | 68.32 (retrieval 67.71) | MTEB multilingual | proprietary/API |
| voyage-3-large | — | retrieval-optimized commercial | **Veda ships this preset** |
| **BGE-M3** | ~63.0 | MTEB-EN | **Veda's documented default** |

The May audit put BGE-M3 "~7–9 pts behind the frontier". With KaLM-Gemma3-12B at 72.32 and
Harrier-OSS at 74.3 that spread is now **~9–11 pts**.

**Two caveats that matter more than the ranking:**

1. **MTEB v2 scores are not comparable to v1**, and rankings differ across boards (English v2 vs
   MMTEB). Any comparison table we publish must state which board and which version.
2. **Leaderboard rank does not transfer in-domain.** On a legal-contract retrieval eval of 200
   queries, the MTEB top-3 placed **5th, 7th and 2nd**, and **BGE-large-en-v1.5 — ranked 11th on
   MTEB — won**.

Sources: [PremAI 2026 ranking](https://www.premai.io/blog/best-embedding-models-for-rag-2026-ranked-by-mteb-score-cost-and-self-hosting/) ·
[Modal](https://modal.com/blog/mteb-leaderboard-article) ·
[Zhang benchmark](https://zc277584121.github.io/rag/2026/03/20/embedding-models-benchmark-2026.html)

> **This is a marketing asset, not just a caveat.** Point 2 is the clearest third-party argument
> for Veda's auto-tuning pipeline: the field's own benchmark does not predict which embedder
> wins on *your* corpus, and Veda is the only system here that will search that space against
> your own data. The May audit's framing ("BGE-M3 is behind, defensible default") understates
> this — the right claim is not "our default is fine", it is "the default doesn't matter because
> we tune it per corpus".

**Verdict: BEHIND on default model quality, but the registry mitigates it** — `qwen3-embedding-8b`
(70.58) is one env var away. **Recommendation: add `kalm-embedding-gemma3-12b` and
`harrier-oss-v1` as presets** (both open; Harrier is MIT) and, more importantly, **stop leading
with the default's MTEB score** in positioning.

---

## 2. Multimodal embeddings — Veda's default is well chosen and independently validated

`jina-embeddings-v5-omni-small` (~1.74 B params) accepts text, image, video and audio into a
shared space aligned with `jina-embeddings-v5-text-small`.

**GELATO** (Geometry-preserving Embeddings via Locked Aligned TOwers) is the method: freeze the
non-text encoders, keep the text tower's geometry, and train only the small connecting
components. It is a direct attack on the **modality gap** — the very metric the May audit used
to justify Veda's unified-space design.

Reported results: **best overall four-modality score among models under 5 B params (53.93)**,
strongest text-only performance in that class, slightly above LCO-Embedding-Omni-3B and behind
only the larger LCO-Embedding-Omni-7B. Benchmarks: MMEB-Video (video), MAEB (audio), MMTEB (text).

**Video is independently confirmed weak:** "video performance is notably weaker compared to
baselines."

Sources: [arXiv 2605.08384](https://arxiv.org/abs/2605.08384) ·
[Jina announcement](https://jina.ai/news/jina-embeddings-v5-omni-multimodal-embeddings-for-text-image-audio-and-video/) ·
[HF model card](https://huggingface.co/jinaai/jina-embeddings-v5-omni-small) ·
[Qwen3-VL-Embedding paper](https://arxiv.org/pdf/2601.04720)

> **Two consequences.**
> 1. `competitive_advantages.md`'s caveat — "text→video alignment is currently weak… a property
>    of the embedding model, not the architecture" — is **CONFIRMED by the model's own paper**.
>    That claim is safe to keep and can now be cited rather than asserted.
> 2. The model is **no longer ours alone**. Elastic ships the whole Jina family on-prem
>    (2026-07-27) and MongoDB ships `voyage-multimodal-3.5` (text+image+**video**). Choosing a
>    good omni model is no longer a differentiator; it is a procurement decision anyone can make.

**Verdict: PAR.** Good choice, well validated, no longer exclusive.

---

## 3. Late interaction — Veda closed its gap; the field opened new ones

### May audit items — status

| May item | Status |
|---|---|
| "PLAID is dominated by WARP (SIGIR'25, 3× speedup, 2–4× smaller index)" | **CLOSED.** `crates/veda-query/src/plaid.rs` implements **WARP_SELECT**; its own comments list remaining un-ported WARP pieces as follow-up work |
| "MaxSim not SIMD-optimized" | **PARTIALLY CLOSED.** `veda-core/src/simd.rs` has `maxsim_distance` + a pre-normalised cosine fast path, with a comment referencing Mixedbread's `maxsim-cpu` (libxsmm/AVX2) shape |
| "ColQwen2.5 / ColModernVBERT adapters missing" | **STILL OPEN.** Weaviate ships ModernVBERT/ColModernVBERT for visual-document retrieval |
| "`instruction:` passthrough blocks Voyage rerank-2.5" | **CLOSED.** `rerank.rs` forwards `instruction` when supplied |

### What is new in the field since May

- **First LIR workshop at ECIR 2026** — late interaction and multi-vector retrieval now has its
  own venue. The stated motivation: late interaction is "particularly well-suited for novel use
  cases such as reasoning-based or **cross-modality** retrieval" — i.e. it is moving *toward*
  Veda's lane ([arXiv 2511.00444](https://arxiv.org/abs/2511.00444)).
- **Token pruning is the active optimization axis**: prune-then-merge for visual document
  retrieval ([arXiv 2602.19549](https://arxiv.org/pdf/2602.19549)), a Voronoi-cell formulation
  for principled token pruning ([arXiv 2603.09933](https://arxiv.org/pdf/2603.09933)).
- **`TileMaxSim`** — IO-aware **GPU** MaxSim scoring with dimension tiling and fused product
  quantization ([arXiv 2606.26439](https://arxiv.org/pdf/2606.26439)). The GPU answer to the
  MaxSim cost problem Veda addressed on CPU.
- **Vespa** ships a native ColBERT embedder with **32× compression** of token-level vectors
  "without significant impact on ranking accuracy", plus long-context ColBERT.

**Verdict: PAR.** Veda's architecture (ColPali + MUVERA + PLAID/WARP_SELECT) is current. The
specific number we cannot match with a citation is Vespa's 32× ColBERT compression.

---

## 4. Rerankers — the generic HTTP API remains the right call

| Model | BEIR nDCG@10 | Params | Note |
|---|---|---|---|
| **jina-reranker-v3.5** | **63.20** | ~0.6 B | current SOTA-for-size |
| Qwen3-Reranker-4B | 62.28 | 4 B | ~7× larger for lower score |
| jina-reranker-v3 | 61.94 | 0.6 B | Qwen3-0.6B base, listwise, 64 docs in a 131 K context |
| mxbai-rerank-large | 61.44 | 1.5 B | |
| bge-reranker-v2-m3 | — | — | v3 is +5.43 % at the same scale |

jina-reranker-v3 multilingual: **66.50 MIRACL** across 18 languages. Architecturally it is
*not* ColBERT-style — causal self-attention across query and up to 64 documents in one window,
then contextual embeddings per document ("last but not late interaction").

Sources: [jina-reranker-v3](https://jina.ai/news/jina-reranker-v3-0-6b-listwise-reranker-for-sota-multilingual-retrieval/) ·
[arXiv 2509.25085](https://arxiv.org/pdf/2509.25085) ·
[v3.5 model card](https://jina.ai/models/jina-reranker-v3.5/) ·
[aimultiple benchmark](https://aimultiple.com/rerankers)

**Verdict: FINE.** Veda's generic Cohere/Voyage/TEI-shape rerank HTTP API with `instruction`
passthrough means adopting jina-reranker-v3.5 is a config change, not a code change. Note that
Elastic now bundles `jina-reranker-v3` in its on-prem package — the same model, one fewer moving
part for their users.

---

## Recommendation order for this file's scope

1. **Reframe the embedding story around auto-tuning, not the default model.** The
   leaderboard-vs-in-domain divergence is third-party evidence for Veda's core differentiator.
   Highest-value, zero-code change.
2. **Add `kalm-embedding-gemma3-12b` and `harrier-oss-v1` (MIT) presets** — registry additions,
   same shape as the existing `qwen3-embedding-*` entries.
3. **Add a `jina-reranker-v3.5` preset/doc example** — config-level.
4. **Add ColModernVBERT / ColQwen2.5 adapters** — still open from May; Weaviate ships it.
5. **Watch `TileMaxSim`** for a GPU MaxSim path once `Cagra` is validated (see `03`).

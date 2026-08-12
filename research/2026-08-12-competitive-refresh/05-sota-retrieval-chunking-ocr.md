# SOTA refresh — sparse/fusion/rerank, chunking, OCR

**Research date:** 2026-08-12. Updates
`research/2026-05-19-sota-audit/{06-sparse-fusion-rerank,07-chunking-ocr}.md`.

**Veda baseline verified in-repo today:**

- Chunking (`crates/veda-core/src/chunking.rs`): `FixedSize`, `MarkdownRecursive`, `Sentence`,
  `Page`, `MarkdownHeading`, **`Hierarchical`** (emits coarse doc-section *and* fine chunks),
  **late chunking** (documented against Jina Sept-2024 / voyage-context-3 July-2025, tags chunks
  `content_types = ["late_chunked"]`)
- Fusion (`crates/veda-query/src/fusion.rs`): RRF, weighted-sum, **DBSF**
- Rerank (`crates/veda-gateway/src/rest/rerank.rs`): generic Cohere/Voyage/TEI-shape HTTP with
  **`instruction` passthrough**
- OCR: **FlashParse** wrapping **GLM-OCR** + PP-DocLayoutV3 layout analysis

---

## 1. OCR — the May recommendation is REVERSED. Do not act on it.

The May audit's **#3 cross-cutting recommendation** was:

> *"Replace FlashParse with PaddleOCR-VL-1.5 as the default OCR engine. Same VRAM, Apache-2.0,
> top of OmniDocBench."*

**That is no longer correct.** As of 2026, on **OmniDocBench v1.5**:

| Model | OmniDocBench v1.5 | Note |
|---|---|---|
| **GLM-OCR** (Zhipu / Z.ai) | **94.62 — SOTA** | **0.9 B params**; CogViT encoder + GLM decoder + Multi-Token Prediction; 1.86 PDF pages/s at ~$0.03/M tokens |
| PaddleOCR-VL(-1.5/1.6) | 94.50 | ~0.9 B (native-res encoder + ERNIE-4.5-0.3B); **100+ languages; the recommended start for CJK / Arabic / non-Latin** |
| Gemini 3 Pro | 90.33 | frontier, closed |
| Qwen3-VL-235B | 89.15 | 260× larger than GLM-OCR |
| GPT-5.2 | 85.4 | frontier, closed |
| dots.ocr | — | 1.7 B, unifies layout + recognition + reading order, 100+ languages, "outperforms models 20× its size" |
| DeepSeek-OCR | — | MoE decoder, ~570 M active params/token → substantially higher throughput at the same batch size |
| HunyuanOCR-1.5 | — | new (2026), "lightweight OCR VLMs faster and better" — track for the next refresh |

**Veda's pipeline is GLM-OCR-backed**, so Veda's document ingestion currently sits at the **top
of OmniDocBench v1.5**. The May audit's "FlashParse doesn't appear on any 2025/2026
leaderboard" observation conflated the *wrapper* (FlashParse) with the *engine* (GLM-OCR).

Sources: **[GLM-OCR Technical Report, arXiv 2603.10910](https://arxiv.org/pdf/2603.10910)** ·
[Z.ai GLM-OCR docs](https://docs.z.ai/guides/vlm/glm-ocr) ·
[PaddleOCR-VL-1.5, arXiv 2601.21957](https://arxiv.org/pdf/2601.21957) ·
[HunyuanOCR-1.5, arXiv 2607.04884](https://arxiv.org/pdf/2607.04884) ·
[Spheron self-host comparison](https://www.spheron.network/blog/best-open-source-ocr-vlm-self-host-gpu-cloud-2026/) ·
[OmniDocBench](https://github.com/opendatalab/OmniDocBench) ·
[decodethefuture GLM-OCR vs PaddleOCR](https://decodethefuture.org/en/glm-ocr-explained/)

**Two caveats before this becomes a marketing line:**

1. **The benchmark is saturated.** LlamaIndex published *"OmniDocBench is Saturated, What's Next
   for OCR Benchmarks?"* — scores above 94 should be treated as "within noise" until a private
   eval confirms a delta. A 94.62-vs-94.50 lead is **not** a defensible claim.
   ([LlamaIndex](https://www.llamaindex.ai/blog/omnidocbench-is-saturated-what-s-next-for-ocr-benchmarks))
2. **CJK is called out separately.** PaddleOCR-VL-1.6 is the recommended starting point for
   corpora with CJK, Arabic or other non-Latin scripts — and Veda's own live testing corpus is
   **Japanese business documents**. Worth an in-domain head-to-head on our own corpus rather
   than trusting either leaderboard number.

**Verdict: AHEAD (unexpectedly), but on a saturated benchmark.** Action: **withdraw the May
recommendation**, and run a private GLM-OCR vs PaddleOCR-VL-1.6 eval on the Japanese corpus
before choosing a default for CJK-heavy deployments.

---

## 2. Chunking — Veda covers the 2026 production patterns

The May audit's chunking recommendation (retune `MarkdownRecursive` from `{2000, 300}` to
`{1500, 0}`) is **CLOSED** — that is the shipped default.

What the field settled on in 2026, and Veda's status:

| 2026 pattern | Evidence | Veda |
|---|---|---|
| **Hierarchical / parent-child** — "the most widely adopted production pattern in 2025–2026 because it resolves the precision-context trade-off: small chunks for finding, larger chunks for understanding" | [Atlan](https://atlan.com/know/chunking-strategies-rag/), [FutureAGI](https://futureagi.com/blog/advanced-chunking-techniques-for-rag/) | **HAS** — `Hierarchical` emits both coarse and fine chunks |
| **Late chunking** — embed the whole document first, split the token embeddings after; **+10–12 % retrieval accuracy on documents with anaphoric references** | [Medium/Mysore](https://medium.com/@visrow/chunking-strategies-for-rag-early-late-and-contextual-chunking-explained-with-code-71b88e4709f9), [arXiv 2504.19754](https://arxiv.org/abs/2504.19754) | **HAS** — documented against Jina/voyage-context-3 |
| **Contextual retrieval** — best paired with BM25 + reranking or late chunking when boundary context loss is the bottleneck | [Firecrawl](https://www.firecrawl.dev/blog/best-chunking-strategies-rag) | partial — has BM25 + rerank + late chunking; no per-chunk LLM context prefixing |
| Structure-aware over naive splitting — **chunking choice can swing recall by up to 9 % on the same corpus** | [DigitalApplied](https://www.digitalapplied.com/blog/rag-chunking-strategies-2026-retrieval-quality-playbook) | **HAS** — 7+ strategies, and uniquely **searches** them via tier-3 auto-tuning |

Also new: a 2026 cost-effectiveness evaluation of chunking methods
([arXiv 2606.00881](https://arxiv.org/html/2606.00881v1)) and SLIDE for sliding localized
extraction ([arXiv 2503.17952](https://arxiv.org/pdf/2503.17952)).

**Verdict: AHEAD.** Veda not only ships the two patterns 2026 converged on, it is the only
system in this survey that **automatically searches chunking strategy as a tuning dimension**
(tier-3 of the campaign engine). Given the "up to 9 % recall swing" finding, that is a
substantive, defensible differentiator — and it is currently under-marketed.

> **Caveat carried from the 2026-08-12 fix:** tier-3 recall scoring only became meaningful with
> the content-overlap change (`5d24dea`). Before it, re-chunked trials scored a structural 0.000,
> so the differentiator was real in design but broken in measurement. Any claim we make about
> "searching chunking strategies" should date from that fix, not from the original feature.

---

## 3. Sparse, fusion and rerank

### Still standard, still correct

SPLADE-v3 + BM25 + RRF remain the right defaults. SPLADE outperforms BM25 on most BEIR tasks via
learned query expansion, at the cost of GPU inference; the standard mitigation is exactly what
Veda does — **precompute document sparse vectors at index time, run SPLADE only on the query**
(query-time SPLADE otherwise adds 100–300 ms, "not tolerable for interactive RAG").

Fusion: RRF remains the default; the May audit's gap "no CC/DBSF fusion options" is **CLOSED**
(`fusion.rs` has DBSF). Newer work is on LLM-query-expansion-driven fusion
([Exp4Fuse, arXiv 2506.04760](https://arxiv.org/pdf/2506.04760)) and projection fusion with
diversity reranking ([arXiv 2604.13728](https://arxiv.org/pdf/2604.13728)) — research-stage, not
table stakes.

Sources: [PremAI hybrid search](https://www.premai.io/blog/hybrid-search-for-rag-bm25-splade-and-vector-search-combined/) ·
[DigitalApplied hybrid reference](https://www.digitalapplied.com/blog/hybrid-search-bm25-vector-reranking-reference-2026)

**Verdict: PAR.**

### The one concrete gap: sparse execution

Veda's sparse search is, by its own comment in `crates/veda-query/src/search.rs`,
**brute-force**: *"Sparse search is brute-force in v1: every non-deleted, non-empty sparse vector
is scored against the query and merged into a global top-k."*

The field moved past this within the last quarter:

| System | Sparse execution | Date |
|---|---|---|
| Milvus 3.0 | **SINDI + Block-Max WAND + Block-Max MaxScore**, inverted-list compression + quantization; index ~3× smaller at comparable recall | 2026-07-27 |
| Weaviate 1.39 | BM25 via **BlockMax WAND** | 2026-08-04 |
| Qdrant 1.19 | **per-query IDF corpus** for sparse | 2026-08-05 |
| **Veda** | brute-force scan over all sparse vectors | — |

**Verdict: BEHIND.** This is the most concrete, smallest-scope engineering gap this refresh
found. Block-Max WAND is a well-documented algorithm with reference implementations, it is
confined to the sparse execution path, and three competitors shipped it in a three-week window.

---

## Recommendation order for this file's scope

1. **Withdraw the "replace FlashParse with PaddleOCR-VL" recommendation** from the May audit —
   GLM-OCR leads OmniDocBench v1.5. Replace it with a **private CJK eval** (GLM-OCR vs
   PaddleOCR-VL-1.6) on the Japanese corpus.
2. **Implement Block-Max WAND (or MaxScore) for sparse retrieval.** Smallest well-defined gap
   with three competitors ahead; contained to `veda-query`'s sparse path.
3. **Market the chunking-search differentiator.** "Chunking swings recall up to 9 %; we search it
   for you" is a claim backed by third-party evidence and a shipped feature.
4. Optional: per-chunk LLM context prefixing (Anthropic-style contextual retrieval) as an eighth
   strategy — it composes with the existing tier-3 search rather than replacing anything.

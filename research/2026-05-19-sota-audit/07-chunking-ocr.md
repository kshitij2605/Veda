# Veda SOTA Audit — Chunking, Document Processing, OCR (2026)

Date: 2026-05-19
Scope: `crates/veda-core/src/chunking.rs`, `POST /v1/collections/{name}/process-document`, FlashParse + GLM-OCR + VLM augmentation profiles.
Method: literature scan (LumberChunker EMNLP 2024 + 2026 follow-ups, Late Chunking arXiv 2409.04701 v3 Jul 2025, Anthropic Contextual Retrieval Sept 2024 + 2026 adoption studies, voyage-context-3 Jul 2025), OmniDocBench v1.5/v1.7 leaderboard, Docling/Granite-Docling, Marker, MinerU 2.5, Mistral OCR 3 (Dec 2025), PaddleOCR-VL-1.5, GLM-OCR, DeepSeek-OCR.

---

## Verdict

**Chunking — Veda is at the 2026 frontier.** The seven strategies plus the `LateChunking` and `ContextualChunking` variants (already wired in `chunking.rs`) cover the *technique* surface area. The two gaps are (1) absence of a *contextualized chunk embedding* path that competes with `voyage-context-3` (Jul 2025) — currently the highest-scoring chunk-aware embedding approach published — and (2) absence of an LLM-driven boundary chunker in the LumberChunker / TopoChunker family. Defaults are also tuned hot: 2000 chars / 300 overlap sits above the empirically-validated 2026 "context cliff" (~2.5k tokens / ~10k chars where retrieval quality collapses); recent multi-dataset studies push the sweet spot down to **256-512 tokens (~1000-2000 chars) with 10-20 % overlap**, or *zero* overlap when paired with semantic boundaries (overlap "provides no measurable benefit and increases indexing cost" per the Jan 2026 systematic analysis arXiv:2601.14123).

**OCR — FlashParse + GLM-OCR + VLM augmentation is *competitive but no longer SOTA*.** GLM-OCR is the current OmniDocBench v1.5 leader (94.62 overall), so keep it. FlashParse, by contrast, does not appear on any 2025/2026 leaderboard and is dominated by PaddleOCR-VL-1.5 (94.5, 0.9 B params, ~5 GB VRAM vs FlashParse's ~8 GB), MinerU 2.5 (90.67, best on PubTabNet scientific tables, integrates TableFormer + RT-DETR), and Granite-Docling-258M (Jan 2026, Apache-2.0, end-to-end, IBM, Linux Foundation AAIF donation). For VLM augmentation, Qwen2.5-VL / Qwen3-VL is still defensible but PaddleOCR-VL or Granite-Docling now subsume both the OCR and the "captioning" roles for ~1/30th the VRAM.

**Top 3 gaps / wins**:
1. **Replace FlashParse with PaddleOCR-VL-1.5 (or Granite-Docling-258M)** as the default OCR engine. Win: +SOTA accuracy, ~3-5x lower VRAM, Apache-2.0, multilingual.
2. **Add `ContextualizedEmbedding` chunk-aware embedding path** (voyage-context-3-style) — chunks share global doc context *natively* in the embedding model, beating Anthropic Contextual Retrieval by 20.54 % @ chunk-level on Voyage's own benches and removing the per-chunk LLM call cost.
3. **Tune defaults**: drop `MarkdownRecursive` default from 2000/300 to **1500/0** (≈ 400-token chunks) or **2000/200** (semantic-boundary aware); add a "no-overlap + sentence boundary" preset; document the 2.5k-token "context cliff".

---

## A. Chunking

### A1. Late Chunking (Jina, Sept 2024)
Verdict: **already implemented (`LateChunking`), but underutilized.**
Late chunking — embed the full document with a long-context encoder (jina-v3, jina-v4, BGE-M3 8k), then mean-pool per chunk window — has gone from research curiosity to *expected* feature in 2026 vector DBs. Jina v3 (8k), jina-embeddings-v4 (multimodal), voyage-context-3 all natively expose it. The arXiv v3 update (Jul 2025) reports a 1.5-3.6 % absolute nDCG@10 lift over naive chunking across BEIR retrieval, and "improved across all boundary strategies" — the boundary strategy itself becomes nearly irrelevant when late pooling is on. Production usage: Milvus, MongoDB Atlas Vector Search, Weaviate all ship integrated tutorials. Veda's wiring (chunker emits offsets, embedding service mean-pools) is the correct minimal contract.

Gap: Veda currently *tags* chunks `late_chunked` and trusts the embedding service. There is no built-in voyage-context-3 client and no documented quickstart pointing users at jina-embeddings-v3. Add a config preset and validation that `EMBEDDING_ENDPOINT` advertises late-pooling support.

### A2. LumberChunker (EMNLP 2024) + 2026 follow-ups
Verdict: **not in Veda; conditional add.**
LumberChunker iteratively prompts an LLM to find the point in a sliding window of passages where the content shifts. Reported +7.37 % DCG@20 vs the best baseline on GutenQA (3000 narrative-book QA pairs). The 2026 follow-up "Beyond Chunk-Then-Embed" (arXiv:2602.16974) and the Mar 2026 CMU ML Blog re-analysis are blunt: **task-dependent**. Structure-based (Markdown-recursive) wins for *in-corpus* retrieval, LumberChunker wins only for *in-document* retrieval over long narrative text. Optimal θ ≈ 550 tokens.

Recommendation: add a `LLMChunker { boundary_model_endpoint, window_size, max_chunk_size }` variant, but **do not make it default**. Position it for narrative-heavy corpora (legal opinions, novels, long-form journalism). Cost: one LLM call per ~550-token boundary decision; with prompt caching ~$0.20-0.50 per million doc tokens.

### A3. Agentic chunking
Verdict: **experimental; skip for now.**
TopoChunker (arXiv:2603.18409) proposes "topology-aware agentic" chunking with explicit hierarchies, and current 2025-2026 surveys (langcopilot Oct 2025, Firecrawl 2026) summarize the field as "5x accuracy in best-case multi-step retrieval, but complexity and cost limit real-world adoption." Veda's `Hierarchical` strategy already covers the multi-granularity emission story; an "agent" that picks the chunker per doc is sugar, not substance.

Recommendation: defer. Document that `Hierarchical` (coarse + fine) is Veda's answer to multi-granularity until an agent harness becomes warranted.

### A4. Semantic chunking — done right
Verdict: **Veda's current `Semantic` is a text-similarity heuristic, not embedding-based.**
The 2026 picture: embedding-distance semantic chunking (LangChain `SemanticChunker`, Llama-Index `SemanticSplitterNodeParser`) is the literature baseline, but a 2025 IEEE benchmark across 7 arXiv domains found a sentence-aware splitter with 512-token windows + 200-token overlap *outperformed* breakpoint-based semantic chunking on overall retrieval. Milvus's Apr 2026 Max-Min semantic chunking re-establishes embedding-based as the SOTA when paired with min-pool aggregation.

The `chunking.rs` comment is honest: "Uses a lightweight text-based similarity heuristic (word overlap) since we can't call embedding models synchronously." This is a *real* limitation. Two routes:

a. **Fast**: call the embedding service inside the chunker via the same async runtime, behind a feature flag. Cost: roughly N+1 embedding calls instead of 1 (where N = sentence count); for a 4 KB doc maybe 30-40 small embedding calls, batchable.
b. **Better**: emit `Semantic` chunk *candidates* (lots of small sentence-level chunks), then let the embedding service collapse adjacent ones whose cosine similarity is above threshold. Requires a coordinator-side post-pass — substantial change.

Recommendation: implement (a). The synchronous-embedding ban inside the chunker is the actual blocker; it's a design choice, not a hard constraint.

### A5. Contextual Retrieval (Anthropic, Sept 2024)
Verdict: **`ContextualChunking` implemented; prompt-caching guardrails missing.**
Anthropic's Sept 2024 blog reports 49 % retrieval-error reduction solo, 67 % paired with reranking. By 2026 this is standard practice in production RAG (CoderCops "RAG is Dead, Long Live RAG", Apr 2026; AWS Bedrock Knowledge Bases shipped contextual chunking integrated with Claude in Q4 2025). Cost: ~$1.02 per million doc tokens *with prompt caching*; without caching it's 10x.

Veda's docstring already says "prompt caching is essential at scale" but doesn't enforce it. Add:
- A sanity check that `CONTEXTUAL_LLM_ENDPOINT` is a caching-aware endpoint (Anthropic, OpenAI, Bedrock — refuse vLLM without `--enable-prefix-caching`).
- A telemetry counter for `contextual_chunking.cache_hit_rate`; alert if < 70 %.
- Batch the per-chunk LLM call (4-8 chunks per request) to amortize the cached prefix.

### A6. Markdown-aware / structure-aware chunking
Verdict: **strongest area for Veda.** `MarkdownRecursive` + `MarkdownHeading` (the `SectionChunker` in your description) is essentially what LlamaIndex `MarkdownNodeParser` + `HierarchicalNodeParser` + Unstructured.io's chunk-by-title implement. The fact that document ingest goes OCR → markdown → chunker means structure is preserved end-to-end.

Microsoft MarkItDown (a 2024 open-source tool to normalize Office docs → markdown) is a *prep* layer, not a chunker. Recommendation: if Veda's process-document doesn't already accept .docx/.pptx/.xlsx natively, integrate MarkItDown as a pre-OCR conversion step for digital-native Office docs — skip OCR entirely for these.

### A7. Recursive character vs token-aware vs semantic — 2026 verdict
The Jan 2026 systematic analysis (arXiv:2601.14123, 90 chunker-model configs) lands on this hierarchy:
1. **Sentence-aware splitting** is the most cost-effective and matches semantic chunking up to ~5k tokens.
2. **Semantic chunking** wins only on small datasets with high coherence between adjacent sentences.
3. **Token-aware** > **character-aware** when the embedding model has a hard token cap (so 512 tokens, not 2000 chars; embedding-aware token counting matters).
4. **Overlap** provides **no measurable benefit** in their study — and increases indexing cost.

Recommendation: Veda's character-aware sizing should be supplemented by a token-aware mode keyed off the embedding model's tokenizer. The current 2000-char default works for ~500-token English chunks under BGE-M3 / jina-v3 (4 chars/token) but breaks for CJK (~1 char/token = 2000 tokens, well past the cliff). Surface `tokenizer_aware_max_tokens: usize` as a sibling to `max_chunk_size`.

### A8. Chunk size sweet spot
2026 consensus across Milvus, LlamaIndex, the arXiv:2505.21700 multi-dataset analysis, and the Jan 2026 systematic study:
- **Factoid queries**: 256-512 tokens (~1000-2000 chars).
- **Analytical queries**: 1024+ tokens.
- **Mixed**: start at 400-512 tokens with 10-20 % overlap, OR 0 % overlap if using sentence boundaries.
- **Context cliff**: retrieval quality drops sharply past ~2.5 k tokens.

Veda's `MarkdownRecursive { 2000, 300 }` default: 2000 chars ≈ 500 English tokens (good), but 300 char overlap = 15 % (high). Recommendation:
- New default: `MarkdownRecursive { 1500, 0 }` (boundary-aware, no overlap) — matches 2026 best practice.
- Add a `MarkdownRecursive::factoid()` preset (800 / 0) and `MarkdownRecursive::analytical()` preset (3000 / 300, with a `warn_above_cliff: true` flag that logs when chunks exceed 2500 tokens).

---

## B. Document Processing / OCR

### B1. Docling (IBM, 2024 → 2026)
Verdict: **production-grade; consider as alternative pipeline.**
58.6 k GitHub stars as of early 2026 (was 37 k in Aug 2025). Donated to Linux Foundation AAIF Q1 2026. Anchored by **Granite-Docling-258M** (Jan 2026, Apache-2.0): a 258 M-parameter VLM (Idefics3 + Granite 165 M LLM + SigLIP2-base-patch16-512 vision encoder) that does end-to-end PDF → DocTags/markdown. Replaces the experimental SmolDocling-256M-preview. Table extraction uses TableFormer + RT-DETR detection + EasyOCR. Red Hat OpenShift Operator targets banking/insurance.

Recommendation: add Docling as a parallel `--profile docling` in `docker-compose.yml`. It's a *one-shot* PDF → markdown alternative to Veda's FlashParse → markdown chain, with substantially lower VRAM (~2-3 GB for the 258 M model) and Apache-2.0 license. Position FlashParse vs Docling as user choice based on table-density.

### B2. Marker (Vik Paruchuri / datalab-to)
Verdict: **strong PDF-to-markdown, but not on OmniDocBench leaderboard.**
PyMuPDF + Tesseract + Surya (GPU) base. Active development. Notable for structure fidelity and image/table handling. Now under the commercial `datalab-to` org. License: GPL-3.0 (with paid commercial license available) — *this is a deal-breaker for many enterprise Veda users*.

Recommendation: do not adopt as default. Consider documenting as a community plug-in via Veda's pipeline abstraction.

### B3. MinerU (OpenDataLab)
Verdict: **#1 on English subset, #2 on overall PubTabNet (88.4 TEDS).**
MinerU 2.5 leads OmniDocBench English subset in reading-order and column-layout. Best PubTabNet score among the three leading open systems (vs 85.2 GLM-OCR, lower for PaddleOCR-VL). MIT-licensed core.

Recommendation: this is the right choice for *table-heavy* English corpora. Add as a `--profile mineru` alongside Docling and FlashParse. Auto-selection heuristic: if `text-only` collection and tables suspected, route to MinerU; if multilingual / messy scans, route to PaddleOCR-VL; if banking / contracts, route to Granite-Docling.

### B4. Mistral OCR / Mistral OCR 3 (Dec 2025)
Verdict: **closed API, fast, accuracy contested.**
Mistral OCR 2 hit ~94.9 % on internal benchmarks (vs Google Document AI 83.4 %, Azure 89.5 %). Mistral OCR 3 (Dec 2025): 88.9 % handwriting (vs Azure 78.2 %), 96.6 % tables (vs Textract 84.8 %), 2000 pages/min. Real-world tests show a "significant gap between theoretical and observed" — community reports place it behind open Qwen2.5-VL on JSON extraction (72.2 % Mistral OCR 2 vs ~75 % Qwen2.5-VL).

Recommendation: do not depend on Mistral OCR for a self-hostable vector DB. Document as a `MISTRAL_OCR_API_KEY` optional pipeline path for users who want commercial reliability.

### B5. OmniDocBench / DOLPHIN / DocVQA — what scores SOTA in 2026?
OmniDocBench v1.7 (Apr 2026, post v1.6 transition; Qianfan-OCR leaderboard added):
- **GLM-OCR (ZhipuAI)**: 94.62 overall — #1 open-source.
- **PaddleOCR-VL-1.5 (0.9 B)**: 94.50 — #2, dominant on multi-column, headers/footers, ArXiv.
- **MinerU 2.5**: 90.67 — wins PubTabNet scientific tables (88.4 TEDS).
- **Gemini 3 Pro**: 90.3.
- **GPT-5.2**: below leading OSS.
- **DeepSeek-OCR (Gundam-M)**: 75.7, but 4.65 pages/sec — speed leader.

Critical caveat (LlamaIndex blog, 2026): **OmniDocBench is saturated.** Top scores cluster within 4 points; exact-match metrics penalize semantically-equivalent formatting differences; the 1,355-page corpus is academia-heavy. Watch for `socOCRbench` (Noah Dasanaike) and `olmOCR-Bench` (1,402 PDFs, 7,010 cases) as supplementary signals.

Recommendation: Veda should pick from {GLM-OCR, PaddleOCR-VL-1.5, MinerU 2.5} per use case rather than chase the leaderboard.

### B6. Nougat / Donut / DePlot
Verdict: **legacy; deprecate from active pipeline.**
Nougat (Meta, Aug 2023) — still actively maintained for academic-paper math/formulas (Arabic-Nougat fine-tune Nov 2024) but architecturally a Donut variant, which is now five generations behind the VLM frontier. DePlot (Google chart-to-table) — niche, no longer maintained at the pace of competitors.

Recommendation: do not add. If a user needs ArXiv-equation OCR, route to PaddleOCR-VL-1.5 (FormulaCDM SOTA) or MinerU.

### B7. VLM-based OCR (Qwen2.5-VL, Phi-3.5-vision, Pixtral, GPT-4o, Granite-Docling)
Verdict: **VLMs have absorbed dedicated OCR for many use cases.**
2026 finding: Qwen2.5-72B-VL and Qwen3-VL-235B match or exceed Mistral OCR / dedicated OCR on JSON extraction, table extraction, markdown output, and arbitrary-orientation robustness. Granite-Docling at 258 M params matches multi-billion-param systems for document-specific OCR. The interesting inversion: **dedicated OCR models (Qwen-VL-OCR) underperform their general-purpose siblings (Qwen3-VL)** — confirming that VLM-as-OCR is no longer a compromise.

When VLM > dedicated OCR:
- Mixed layouts, screenshots, web pages, slides.
- Need structured output (JSON / markdown / DocTags) directly.
- Need image-region captioning alongside OCR.

When dedicated OCR > VLM:
- High-volume throughput (PaddleOCR-VL-1.5 ~10 pages/sec on a single A100; VLM-72B ~0.5).
- Pure text accuracy on scanned books (PaddleOCR-VL OCRBench Text 94.0).
- Predictable cost / latency.

Recommendation: Veda's current `--profile ocr-vlm` design (VLM as a *secondary captioner* on top of dedicated OCR) is **the right architecture**. Just swap the dedicated OCR from FlashParse → PaddleOCR-VL-1.5, and standardize the VLM on Qwen2.5-VL-7B or Granite-Docling-258M (both small enough to coexist with the index).

### B8. Table extraction
Verdict: **TableFormer + TATR remain the SOTA backbone, used internally by both Docling and modern stacks.**
Docling's pipeline: RT-DETR table detection → TableFormer table-structure recognition → EasyOCR (now optionally PaddleOCR) text. Microsoft's `table-transformer-structure-recognition-v1.1-all` is the strongest open standalone. For pure scientific tables, MinerU 2.5 wins PubTabNet (88.4 TEDS). For mixed business tables, GLM-OCR's end-to-end TableTEDS-S is best.

Recommendation: do not build a table extractor inside Veda. Trust the OCR engine's output; if a user wants specialized table accuracy, expose `OCR_BACKEND=mineru` (best for PubTabNet-style) or `OCR_BACKEND=glm-ocr` (best for messy business tables).

### B9. Layout-aware embeddings (LayoutLMv3) — still SOTA?
Verdict: **superseded for end-to-end pipelines.**
LayoutLMv3 (2022) was the discriminative-pretraining benchmark for form / receipt understanding. By 2026, VLM-based document encoders (ColPali, ColQwen, GME-Qwen2-VL, jina-embeddings-v4 multimodal) have effectively absorbed layout-aware embedding into vision-language pretraining: the model sees the page as an image and produces a layout-aware vector natively.

Recommendation: Veda's vision-language embedding path (Qwen3-VL-Embedding-2B in `--profile embedding`) already does this. LayoutLMv3 is not a gap. Tactical note: consider adding ColPali / ColQwen-style **late-interaction multi-vector** document embeddings (MaxSim instead of cosine) as an opt-in for visually-rich documents; this is the highest-impact 2026 unlock for PDF/slide collections and a clear differentiator vs Milvus/Qdrant/Weaviate which still require manual setup.

### B10. End-to-end Document QA — does the field bypass chunking?
Verdict: **partially, and not for vector-DB use cases.**
Mamba retrievers (130 M / 1.3 B, arXiv 2026) and discourse-aware hierarchical retrieval (arXiv:2506.06313) skip chunk boundaries by reading the whole document with linear-time recurrence and emitting sentence-level retrievals. Gemini 1.5/2.5 Pro with 1-2 M-token context windows make "stuff the whole corpus into the LLM" tractable for small enterprises.

But these approaches don't *replace* a vector DB — they replace small-scale RAG. For Veda's target (millions to billions of vectors, billions of tokens of content), chunking remains structurally required; the right question is "*how* to chunk for best recall," not "do we still chunk."

Recommendation: monitor Mamba-retriever family. No engineering action this cycle.

---

## Specific recommendations (engineering deltas)

### Chunking — chunking.rs / veda-core
1. Reduce `MarkdownRecursive` default to `{ max_chunk_size: 1500, chunk_overlap: 0 }` and require an explicit opt-in for overlap > 0. Log a warning if any chunk exceeds 2500 tokens (the "cliff").
2. Add `TokenAware` companion field on all character-sized chunkers, keyed off the embedding model's tokenizer (BGE-M3 / jina-v3 / Qwen3-VL-Embedding tokenizers cached in `veda-core`).
3. Make `Semantic` actually call the embedding service (feature-gated `async-semantic-chunking`). Document the trade-off; default off.
4. Add `LumberChunker { boundary_model_endpoint, theta_tokens: 550 }` for narrative-heavy corpora. Document as opt-in, not default.
5. Add a `ContextualizedEmbedding` strategy: emits the *same* `MarkdownRecursive`-style chunks but tags them so the embedding service uses voyage-context-3 / jina-v3-late or a custom contextualizer that processes the full doc in one shot. This subsumes both `LateChunking` and `ContextualChunking` for users on voyage-context-3, at lower cost.
6. Add a tokenizer-aware "factoid" preset (`{ 800, 0 }`) and "analytical" preset (`{ 3000, 300 }`) on `MarkdownRecursive` with an `intent` enum.

### OCR / document pipeline
1. **Swap FlashParse for PaddleOCR-VL-1.5** as the default OCR engine. Reasons: SOTA OmniDocBench v1.5 (94.5), Apache-2.0, 0.9 B params (~3-5 GB VRAM), multilingual, robust on warped/skewed scans, fastest among top-3.
2. **Add Granite-Docling-258M as an alternative profile** (`--profile granite-docling`). Reasons: end-to-end PDF → DocTags / markdown, IBM-backed, Linux Foundation AAIF, 258 M params (~2 GB VRAM), Apache-2.0. Position for banking / contracts.
3. **Add MinerU 2.5 as a third option** for scientific-table-heavy corpora.
4. **Keep GLM-OCR** as the high-accuracy / general-purpose option. It's the current OmniDocBench v1.5 #1 (94.62), and Veda already has it.
5. **Auto-routing**: surface an `OcrBackend::Auto` mode that picks per document — page count, language detection, table density — between `PaddleOCR-VL`, `GLM-OCR`, `MinerU`, `Granite-Docling`. Heuristics encoded in `veda-gateway` document pipeline.
6. **Deprecate FlashParse** from the default `docker-compose.yml`. Keep as `--profile ocr-legacy` for one release for backwards compatibility, then remove.
7. **VLM augmentation**: keep Qwen3-VL for image captioning, but document that with Granite-Docling or PaddleOCR-VL as the primary OCR, the captioner only needs to handle non-text image regions — much smaller, ~2 GB VRAM model suffices (Qwen2.5-VL-3B or Granite-Docling-258M doubling up).

### Embedding pipeline
1. **Add ColPali / ColQwen multi-vector late-interaction support** for visually-rich documents. This is a 2026 differentiator (Milvus/Qdrant/Weaviate require manual setup for MaxSim). Wire it as a tag on `MultiModalCollection` configurations.
2. **Add voyage-context-3 / jina-embeddings-v3 late-pool client** in `veda-embedding`; surface as a first-class `EmbeddingModel::Contextualized` enum variant so users don't have to set raw env vars.

---

## Closing observations

Veda's chunking surface is already past the 2026 SOTA for most enterprise RAG workloads — the seven listed strategies plus the two scaffolds (`LateChunking`, `ContextualChunking`) miss only one true technique (LumberChunker / LLM-boundary), and that one is task-conditional. The OCR layer is where the realistic wins are: FlashParse predates the 2025 wave of compact VLM-OCR (PaddleOCR-VL, Granite-Docling) that delivered both higher accuracy *and* substantially lower VRAM, and standardizing on a top-2 OmniDocBench engine costs little engineering effort.

The Niriksha-side / observability angle is worth a footnote: per-document chunking telemetry (chunk count, avg token length, OCR confidence distribution, contextual-chunking cache hit rate) is exactly the kind of cross-cutting signal Niriksha could surface as a "document quality" panel — many of the gaps named above only become *visible* with the right dashboard.

---

## Sources

Chunking
- [Late Chunking arXiv 2409.04701 v3 (Jul 2025)](https://arxiv.org/pdf/2409.04701)
- [Jina AI Late Chunking blog (Sept 2024)](https://jina.ai/news/late-chunking-in-long-context-embedding-models/)
- [Milvus — Late Chunking with Jina v2](https://milvus.io/blog/smarter-retrieval-for-rag-late-chunking-with-jina-embeddings-v2-and-milvus.md)
- [LumberChunker EMNLP 2024 / arXiv 2406.17526](https://arxiv.org/html/2406.17526)
- [CMU ML Blog — LumberChunker re-analysis (Mar 2026)](https://blog.ml.cmu.edu/2026/03/17/lumberchunker-long-form-narrative-document-segmentation/)
- [Beyond Chunk-Then-Embed taxonomy (arXiv 2602.16974)](https://arxiv.org/abs/2602.16974)
- [TopoChunker — topology-aware agentic chunking (arXiv 2603.18409)](https://arxiv.org/html/2603.18409)
- [Anthropic Contextual Retrieval (Sept 2024)](https://www.anthropic.com/news/contextual-retrieval)
- [voyage-context-3 (Jul 2025)](https://blog.voyageai.com/2025/07/23/voyage-context-3/)
- [Systematic Analysis of Chunking Strategies (arXiv 2601.14123, Jan 2026)](https://arxiv.org/abs/2601.14123)
- [Rethinking Chunk Size (arXiv 2505.21700)](https://arxiv.org/html/2505.21700v2)
- [Discourse-Aware Hierarchical Retrieval (arXiv 2506.06313)](https://arxiv.org/abs/2506.06313)
- [Best Chunking Strategies for RAG and LLMs in 2026 — Firecrawl](https://www.firecrawl.dev/blog/best-chunking-strategies-rag)
- [Document Chunking for RAG — langcopilot (Oct 2025)](https://langcopilot.com/posts/2025-10-11-document-chunking-for-rag-practical-guide)
- [Max-Min Semantic Chunking — Milvus blog](https://milvus.io/blog/embedding-first-chunking-second-smarter-rag-retrieval-with-max-min-semantic-chunking.md)
- [LlamaIndex HierarchicalNodeParser docs](https://developers.llamaindex.ai/python/framework-api-reference/node_parsers/hierarchical/)
- [Anthropic Prompt Caching 2026 cost/TTL guide](https://aicheckerhub.com/anthropic-prompt-caching-2026-cost-latency-guide)

OCR / Document Processing
- [OmniDocBench GitHub (CVPR 2025, v1.7 Apr 2026)](https://github.com/opendatalab/OmniDocBench)
- [OmniDocBench paper (arXiv 2412.07626)](https://arxiv.org/html/2412.07626v1)
- [LlamaIndex — OmniDocBench is saturated (2026)](https://www.llamaindex.ai/blog/omnidocbench-is-saturated-what-s-next-for-ocr-benchmarks)
- [CodeSOTA — OmniDocBench leaderboard 2026](https://www.codesota.com/browse/computer-vision/document-parsing/omnidocbench)
- [Docling GitHub (docling-project)](https://github.com/docling-project/docling)
- [IBM — Docling's rise (Linux Foundation AAIF, 2026)](https://www.ibm.com/think/news/doclings-rise-llm-ready-data)
- [IBM Granite-Docling-258M (Jan 2026 announcement)](https://www.ibm.com/new/announcements/granite-docling-end-to-end-document-conversion)
- [Granite-Docling-258M on Hugging Face](https://huggingface.co/ibm-granite/granite-docling-258M)
- [PaddleOCR-VL-1.5 (arXiv 2510.14528 / 2601.21957)](https://arxiv.org/pdf/2510.14528)
- [GLM-OCR vs PaddleOCR-VL-1.5 review — Houdao AI](https://www.houdao.com/d/1665-Zhipu-AI-Releases-GLMOCR-A-HandsOn-Comparative-Review-with-PaddleOCRVL1-5)
- [DeepSeek-OCR-2 vs GLM-OCR vs PaddleOCR benchmark 2026 — regolo.ai](https://regolo.ai/deepseek-ocr-vs-glm-ocr-vs-paddleocr-benchmark-2026/)
- [Marker GitHub (datalab-to/marker)](https://github.com/VikParuchuri/marker)
- [Mistral OCR (mistral.ai/news)](https://mistral.ai/news/mistral-ocr)
- [Mistral OCR 3 review (PyImageSearch, Dec 2025)](https://pyimagesearch.com/2025/12/23/mistral-ocr-3-technical-review-sota-document-parsing-at-commodity-pricing/)
- [Mistral OCR 3 launch (MarkTechPost, Dec 2025)](https://www.marktechpost.com/2025/12/19/mistral-ai-releases-ocr-3-a-smaller-optical-character-recognition-ocr-model-for-structured-document-ai-at-scale/)
- [OCR vs VLM — f22labs](https://www.f22labs.com/blogs/ocr-vs-vlm-vision-language-models-key-comparison/)
- [Qwen 2.5/3-VL OCR — apidog setup](https://apidog.com/blog/qwen-2-5-72b-open-source-ocr/)
- [OCR SOTA Router 2026 — CodeSOTA](https://www.codesota.com/ocr)
- [Microsoft Table-Transformer (TATR)](https://github.com/microsoft/table-transformer)
- [Best PDF Parsers 2026 — Firecrawl](https://www.firecrawl.dev/blog/best-pdf-parsers)
- [Open Source PDF-to-Markdown deep dive (Jimmy Song, 2026)](https://jimmysong.io/blog/pdf-to-markdown-open-source-deep-dive/)
- [Best Open-Source VLMs 2026 — Labellerr](https://www.labellerr.com/blog/top-open-source-vision-language-models/)
- [Supercharge OCR pipelines with open models — Hugging Face blog](https://huggingface.co/blog/ocr-open-models)
- [Nougat — Hugging Face docs](https://huggingface.co/docs/transformers/en/model_doc/nougat)

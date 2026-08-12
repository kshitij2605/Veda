# Veda's position today, and what is stale in `competitive_advantages.md`

**Research date:** 2026-08-12. This file does **not** edit `competitive_advantages.md` — it
lists, claim by claim, what the refresh contradicts or can no longer verify, so acting on it
later is mechanical.

`competitive_advantages.md` was last written **2026-07-02**, and its `[verified]` tags trace to
the **2026-06-21** multimodal E2E campaign. The two moves that invalidate parts of it —
Elastic's Jina On-Prem and Milvus 3.0 — both landed **2026-07-27**, i.e. *after* the doc was
written.

---

## Part A — claim-by-claim staleness list

### Claim #1 — "Multimodal, unified cross-modal retrieval — the wedge"

> *"ES has `dense_vector` + kNN and ELSER, but **no unified text+image+audio+video vector space
> and no native audio/video retrieval** — non-text is handled as opaque attachments."*

**Status: FALSE as written. Highest-priority correction.**

- Elastic Stack **9.5** (2026-08-04): the `semantic` field type "supports all modalities, such as
  text, images, video, audio, and documents".
- Elastic **Jina On-Prem** (2026-07-27): 28 Jina models incl. **`jina-embeddings-v5-omni`** —
  "text, images, audio, and video **in a single embedding space**" — running **fully air-gapped**.
- That is **the same model family Veda's `jina-v5-omni-small` preset uses**.

Also narrowing it: Weaviate (text+image+audio, plus video via TwelveLabs Marengo), MongoDB
(`voyage-multimodal-3.5`, adds video), Marqo (text/image/audio/video, end-to-end), LanceDB
("multimodal lakehouse" as its identity).

**What survives:** not the *capability*, but the *integration*. Elastic assembles a semantic
field + an embedding endpoint + a reranker as configuration; Veda ships one opinionated pipeline
(auto-modality routing, `auto_embed`, `/ingest-media`, cross-modal search) with the RAG-quality
layer attached. Rewrite the claim as **integration and defaults**, not as exclusivity.

### Claim #2 — "Ingest ANY document format, natively"

> *"ES has **no built-in document ingestion** — you bolt on Tika / Unstructured / `ingest-attachment`."*

**Status: WEAKENED.** Elastic's `semantic` field now accepts **PDFs** directly with a multimodal
embedding endpoint. Veda's advantage narrows to the *depth* of the pipeline — OCR with layout
analysis, **table and image extraction**, 7+ chunking strategies, and per-page structure — rather
than to ingestion existing at all. MongoDB's `voyage-multimodal-3.5` also explicitly targets
"tables, graphics, figures, slides, PDFs".

**Bonus, in our favour:** see the OCR finding below — the engine underneath is stronger than the
May audit believed.

### Claim #3 — "Owns the whole RAG stack → trustworthy RAG ES can't offer"

**Status: HOLDS, and is now the strongest claim in the document.** No competitor surveyed ships
faithfulness + citations + eval harness + chunker A/B + coverage/orphan detection + semantic
cache + guardrails + **auto-tuning** as one system. The community evidence in `06` (43 % of
answers carry unsupported claims; 43 % vendor-vs-NIST accuracy gap; MTEB-92 model at 61 %
in-domain precision) is far stronger support than the doc currently uses.

**Action:** promote this from claim #3 to the lead, and cite `06`'s numbers.

### Claim #4 — "Vector-native AIOps & LLM observability"

**Status: HOLDS, with a caveat.** OpenSearch 3.5 added agentic conversation memory and expanded
Prometheus; Elastic 9.5 added agent observability as OTel traces. The specific *semantic*
framing (embedding-based log categorization, semantic incident correlation, spike attribution)
is still not matched. But the adjacency is being contested — this is no longer a quiet lane.

### Claim #5 — "Self-hosted, private, LLM-agnostic — the full stack on your infra"

**Status: SIGNIFICANTLY WEAKENED.** This was a structural claim against Elastic. As of
2026-07-27, Jina On-Prem runs **all 28 models** — including the omni multimodal embedder and
`jina-reranker-v3` — inside the customer's environment with **"no calls to the outside once
deployed"**, on CPU or GPU, small models on a single 8 GB GPU.

**What survives:** Veda self-hosts *more* of the stack (OCR, web-RAG via searxng + crawl4ai, the
LLM endpoint, the whole RAG-quality layer), and is Apache-2.0 throughout, whereas Elastic's
package is distributed under Elastic's terms via an access token. Rewrite as **scope and
licensing**, not as "they can't".

### Claim #6 — "Niriksha — a RAG/vector-first console"

**Status: NOT RE-VERIFIED.** The console/UI stream was deliberately out of scope for this
refresh (see the approved plan). The June 2026 frontend research still stands as the reference.
Note only that Weaviate/Attu/Qdrant consoles continue to evolve, and Elastic 9.5 added
human-in-the-loop approvals reaching outside Kibana.

### Claim #7 — "Agent-native interop with security"

**Status: WEAKENED.** Weaviate shipped **the first native MCP server in any vector database**
(v1.37, April 2026) and took it GA. Elastic ships MCP apps and Agent Builder with Skills /
Attachments / Connectors / Plugins, plus agent observability as OTel traces. OpenSearch 3.5 has
agentic conversation memory.

**What survives:** MCP + A2A + converse **with JWT + tenant + row-level security enforced at the
agent/MCP layer**. The security enforcement is the differentiator now, not the protocol support.
Note the doc's own honest caveat (auth bypass + missing admin RBAC found and fixed in June)
should be extended: the 2026-08-10/12 pass found and fixed seven more defects, including
**silent write loss on aliased collections** — see `reports` and commits `7acede3`, `5d24dea`.

### Performance & index breadth — the benchmark table

> *"Veda (quiet host) ann_search p50 **0.53 ms** … Elasticsearch 8.19 ~7 ms … roughly **13×
> faster**"*

**Status: STALE — benchmarked against a superseded version.** That table used **ES 8.19**. Since
then Elasticsearch shipped, in 9.4 (2026-05-05) and 9.5 (2026-08-04):

- native SIMD kernels — "almost 3×" vector-operation throughput
- **≥3× lower query latency on restrictive filters**
- **cuVS GPU indexing GA** — 12× indexing throughput, 7× faster force-merge
- BBQ 2/4/7-bit quantization + DiskBBQ **auto-calibration**
- a **`VectorDB` index mode** with pre-optimized defaults

**Do not repeat the 13× claim.** It compares current Veda against a version Elastic has since
substantially reworked on exactly the axis measured. Either re-run against 9.5 or retire the
number. The **recall@10 = 1.000 parity** finding is more robust and less likely to have moved.

Also stale: *"11 CPU ANN / quantization index types"* is correct but no longer distinctive —
see `03`.

### Part 2 — "vs pure vector databases"

| Bullet | Status |
|---|---|
| "End-to-end multimodal + document ingestion — they require you to run embedding/OCR/chunking externally" | **WEAKENED.** Marqo does end-to-end multimodal with built-in embedding, open-source; MongoDB auto-embeds on write; Pinecone ships an Inference API |
| "Index breadth — 11+ types; most competitors ship 1–3" | **STALE.** Still true numerically, but Milvus 3.0 added EmbList + a redesigned sparse stack, and the field standardized on RaBitQ + CAGRA. Breadth is no longer scarce |
| "Multi-vector fusion (RRF / weighted-sum / intersection), hybrid dense + sparse + BM25" | **HOLDS** (and DBSF was added). But see `05` — our sparse **execution** is brute-force while three competitors ship Block-Max WAND |
| "Built-in blob storage (MinIO/S3)" | **HOLDS** but is now the weakest form of an object-storage story — competitors put the *index* there |
| "The RAG + AIOps + agent layer" | **HOLDS — the core differentiator** |
| "Niriksha — a first-class console" | not re-verified this pass |

---

## Part B — where Veda actually stands today

### Genuinely ahead

1. **The integrated RAG-quality layer.** Eval-set generation from the user's own corpus →
   campaign search over chunker/embedder/index/query params → Pareto front → safe apply. Nobody
   else has this. The community evidence in `06` says it addresses the top three practitioner
   pains simultaneously.
2. **Chunking breadth *and* automatic chunker search.** `Hierarchical` + late chunking are the
   two patterns 2026 converged on, and Veda is alone in *searching* them per corpus — worth up to
   9 % recall by the field's own estimate.
3. **OCR — unexpectedly.** Veda's GLM-OCR-backed pipeline sits at the top of OmniDocBench v1.5
   (94.62, above Gemini 3 Pro and GPT-5.2). **The May audit's "replace FlashParse" recommendation
   is withdrawn.** Caveat: the benchmark is saturated; a CJK-specific eval is still warranted.
4. **Explain timings (`"explain": true`) per phase.** Still rare at the API level.
5. **Breadth of self-hosting.** OCR + web-RAG + embedder + reranker + LLM + the quality layer,
   Apache-2.0.

### Genuinely behind

1. **Storage format / lake interoperability.** No Lance, no Vortex, no read-in-place. Milvus 3.0
   queries a customer's existing Parquet/Lance/Iceberg lake with zero copy; Veda requires
   ingestion. *Widened since May.*
2. **Object-storage-native index tier.** The category default; Veda has blobs on S3, not an
   index tier. Longest lead time of anything on this list.
3. **Sparse execution is brute-force.** Milvus (SINDI + Block-Max WAND + MaxScore), Weaviate
   (BlockMax WAND) and Qdrant (per-query IDF) all shipped better in a three-week window.
   Smallest well-defined gap here.
4. **`Cagra` is unvalidated.** The index exists; the one E2E that tried it recorded `SKIP — no
   cuVS in this build`. Everyone else has published GPU numbers.
5. **RAG-specific security.** Guardrails are input-side and detect-only; OWASP's July 2026
   additions are about *retrieved* content.
6. **Multi-hop reasoning.** Loudest unmet need in the category; Veda has no explicit answer.

### Fine / par

Quantization (caught up, no longer ahead), CAGRA-on-paper, ACORN filtered traversal,
WARP_SELECT late interaction, SPLADE + BM25 + RRF/DBSF fusion, the generic rerank HTTP API,
multimodal embedding model choice, auto-modality routing.

---

## Part C — the honest one-liner

The 2026-07-02 positioning rests on **capability exclusivity** ("ES cannot do multimodal; pure
vector DBs cannot do end-to-end"). Both halves eroded in the four weeks after it was written.

What did **not** erode is **integration**: no competitor ships document-OCR **and** cross-modal
retrieval **and** a closed-loop RAG-quality system (generate a test set from your corpus, search
the configuration space, apply the winner) **and** vector-native AIOps, self-hostable and
Apache-2.0. That is a narrower and more defensible claim than the current one, and the community
evidence in `06` supports it far better than the exclusivity framing ever did.

Suggested replacement one-liner, for whoever rewrites the doc:

> *Others give you a place to put vectors. Veda gives you a retrieval system that measures itself
> on your data and tunes itself to it — across text, images, audio, video and any document you
> throw at it, entirely on your own infrastructure.*

# Veda — 2026-08-12 competitive / SOTA / community refresh, consolidated summary

Refreshes the April 2026 competitor comparisons, the
[2026-05-19 SOTA audit](../2026-05-19-sota-audit/00-summary.md), and the positioning in
`competitive_advantages.md` (2026-07-02). Each file in this directory is a deep dive; this is
the TL;DR with verdicts and the recommendation order.

Every audit asks the same question the May one did: **is Veda's choice state of the art as of
today, and are we still ahead of the field?** Sources, version numbers and dates are cited
inline in the per-topic files.

**Scope:** competitor landscape, SOTA technology, community demand. The console/UI stream was
deliberately excluded; `research/frontend/2026-06-10-competitive-research.md` remains the
reference there.

---

## The two things that happened while we weren't looking

Both landed **after** `competitive_advantages.md` was written on 2026-07-02.

1. **2026-07-27 — Elastic shipped unified multimodal, self-hosted.** *Jina On-Prem*: 28 Jina
   models including **`jina-embeddings-v5-omni`** (text, image, audio, video in **one** embedding
   space) and `jina-reranker-v3`, running **fully air-gapped**, "no calls to the outside once
   deployed". Elastic Stack 9.5 (2026-08-04) added a `semantic` field accepting text, images,
   video, audio and documents. **This is the same model family Veda uses.**
2. **2026-07-27 — Milvus 3.0 went lake-native.** Reads Parquet / **Lance** / Iceberg / **Vortex**
   in place, zero-copy; new manifest-based columnar storage engine on object storage; sparse
   index overhaul (SINDI + Block-Max WAND + MaxScore, ~3× smaller); server-side sorting,
   aggregation and faceted search.

Together they invalidate the two claims Veda's positioning was built on — *"Elasticsearch can't
do multimodal"* and *"pure vector DBs can't do end-to-end"* — within four weeks of the doc that
asserts them.

---

## Headline scorecard

| # | Area | Status | Headline gap | File |
|---|---|---|---|---|
| 1 | Competitor landscape (incumbents) | **Eroded** | Elastic 9.5 + Jina On-Prem matches the multimodal + self-hosted claims; Milvus 3.0 matches aggregations and beats us on storage; Weaviate shipped the first native MCP server | [01](01-competitors-incumbents.md) |
| 2 | New entrants / contested lanes | **Both lanes closed** | Object-storage-native is now the category default (Turbopuffer ~$100 M ARR, S3 Vectors GA in 31 regions); multimodal-native is contested by ≥5, incl. **Marqo** doing end-to-end multimodal open-source | [02](02-competitors-new-entrants.md) |
| 3 | Indexes / quantization / storage / GPU | **Mixed → behind on storage** | Quantization lead gone (RaBitQ is now standard; everyone shipped multi-bit). CAGRA closed but **never validated on hardware**. **Storage format gap widened** — no Lance/Vortex, no read-in-place | [03](03-sota-indexes-quantization-storage.md) |
| 4 | Embeddings / late interaction | **Par** | BGE-M3 now ~9–11 pts behind frontier, but the registry mitigates. jina-omni validated as best-under-5B four-modality — and now also shipped by Elastic. WARP + instruction-passthrough closed | [04](04-sota-embeddings-late-interaction.md) |
| 5 | Sparse / fusion / chunking / OCR | **Ahead on chunking + OCR, behind on sparse execution** | **May's "replace FlashParse" recommendation is REVERSED** — GLM-OCR leads OmniDocBench v1.5 at 94.62. But sparse search is **brute-force** while three competitors shipped Block-Max WAND | [05](05-sota-retrieval-chunking-ocr.md) |
| 6 | Community demand | **We already answer most of it, badly stated** | Top 7 asks are mostly shipped features. Clearest gap: **RAG-specific security** (OWASP added 3 RAG threat vectors 2026-07-12) and **multi-hop reasoning** | [06](06-community-demand.md) |
| 7 | Position + stale claims | **Rebuild around integration, not exclusivity** | Claims #1, #2, #5, #7 and the benchmark table all need rewriting | [07](07-veda-position-and-stale-claims.md) |

---

## Where Veda is genuinely ahead

1. **The closed-loop RAG-quality system.** Generate a labeled eval set from the user's own
   corpus → search chunker/embedder/index/query-param space → Pareto front → safe apply. No
   competitor surveyed has this, and the community's own numbers say it targets their top three
   pains at once.
2. **Chunking breadth *and* automatic chunker search.** `Hierarchical` + late chunking are the
   two patterns 2026 converged on; chunking swings recall up to 9 %; Veda is alone in searching
   it per corpus.
3. **OCR — the pleasant surprise.** GLM-OCR (what FlashParse wraps) tops OmniDocBench v1.5 at
   **94.62**, above PaddleOCR-VL, Gemini 3 Pro and GPT-5.2. **Withdraw the May recommendation to
   replace it.**
4. **Per-phase explain timings** at the API level — still rare.
5. **Breadth of self-hosting**, Apache-2.0 end to end.

## Where Veda is genuinely behind

In order of "biggest gap × highest payoff":

1. **Storage format / lake interoperability.** No Lance, no Vortex, no query-in-place. Milvus 3.0
   points at a customer's existing lake with zero copy; we require re-ingestion. **Widened since
   May** — it moved from a performance footgun to an interoperability gap.
2. **RAG-specific security.** OWASP added prompt-injection-via-retrieved-documents, indirect
   context poisoning and temporal inconsistency on 2026-07-12; reportedly 89 % of orgs exposed.
   Veda's guardrails are **input-side and detect-only**. We own the ingestion and retrieval path,
   so we can defend this in a way bring-your-own-vector-DB stacks structurally cannot.
3. **Sparse execution is brute-force.** Block-Max WAND is well documented, contained to
   `veda-query`, and three competitors shipped it in three weeks.
4. **`Cagra` unvalidated on real hardware.** The one E2E that tried it recorded `SKIP`.
   Cheapest credibility win available.
5. **Object-storage-native index tier.** Category default; longest lead time; a rearchitecture.
6. **Multi-hop reasoning.** Loudest unmet need in the category (reported 73 % failure rate);
   nobody answers it well, including us.

## Where Veda is fine

Quantization (caught up — RaBitQ standard, multi-bit ladder shipped), ACORN filtered traversal,
WARP_SELECT late interaction, SPLADE + BM25 + RRF/DBSF fusion, the generic rerank HTTP API with
`instruction` passthrough, multimodal embedding model choice, auto-modality routing, hierarchical
+ late chunking.

## May-audit items — closure status

| May item | Status |
|---|---|
| Promote CAGRA to first-class index | **CLOSED** (unvalidated) |
| Adopt Lance 2.2 as a stored-fields format | **STILL OPEN** — now the top gap |
| Replace FlashParse with PaddleOCR-VL-1.5 | **WITHDRAWN** — GLM-OCR now leads OmniDocBench v1.5 |
| Extended-RaBitQ multi-bit ladder | **CLOSED** (`RaBitQExtRerank`) |
| ACORN filtered graph traversal | **CLOSED** |
| Qwen3-Embedding-8B / voyage-3-large presets | **CLOSED** |
| WARP semantics for PLAID | **CLOSED** (`WARP_SELECT`) |
| `instruction:` rerank passthrough | **CLOSED** |
| SIMD-optimize MaxSim | **PARTIAL** |
| Retune MarkdownRecursive to {1500, 0} | **CLOSED** |
| Granite-Docling as a second OCR option | **STILL OPEN** (lower priority given the OCR finding) |
| CC + DBSF fusion options | **CLOSED** (DBSF) |
| SPANN-class object-storage index | **STILL OPEN** — more urgent than in May |
| ColQwen2.5 / ColModernVBERT adapters | **STILL OPEN** |

---

## Cross-cutting recommendation order

**Zero-engineering, highest return:**

1. **Rewrite `competitive_advantages.md` around integration, not exclusivity.** Claims #1, #2,
   #5, #7 are false or weakened as written; the benchmark table cites a superseded ES version.
   `07` lists them claim by claim. Lead with claim #3 (the RAG-quality layer) and cite the
   community's own numbers from `06` — 43 % unsupported claims, 43 % vendor-vs-NIST gap, an
   MTEB-92 model at 61 % in-domain precision, 9 % recall swing from chunking, 43 % cost saving
   from semantic caching.
2. **Withdraw the OCR swap recommendation** and replace it with a private GLM-OCR vs
   PaddleOCR-VL-1.6 eval on the Japanese corpus (CJK is the one place PaddleOCR is specifically
   recommended).
3. **Stop repeating the "13× faster than Elasticsearch" number** until it is re-run against 9.5.

**Small, well-defined engineering:**

4. **Validate `Cagra` on real GPU hardware and publish a number.**
5. **Block-Max WAND (or MaxScore) for sparse retrieval.**
6. **Guardrails for retrieved content** + a mode beyond Flag — addresses OWASP 2026-07-12 and is
   structurally advantaged for us.
7. **Add `kalm-embedding-gemma3-12b` (72.32 MMTEB) and `harrier-oss-v1` (MIT) presets**;
   add a `jina-reranker-v3.5` example.

**Larger, needs a design discussion first:**

8. **Lance as a third `StoredFieldsFormat`** — and, beyond it, read-in-place over a customer's
   existing lake.
9. **In-place schema evolution** (add/drop non-primary fields) — the #2 community demand signal.
10. **Object-storage-native index tier** (SPANN-class).
11. **Multi-hop retrieval strategy.**

## What deliberately does NOT need to change

- **The chunking strategy set** — it covers the 2026 patterns, including the two the field
  converged on.
- **Auto-modality routing and the unified vector space** — still the right design, now also the
  industry's direction.
- **`"explain": true` per-phase timings** — differentiator; keep.
- **The generic rerank HTTP API** — model-agnostic by design; adopting jina-reranker-v3.5 is a
  config change.
- **Apache-2.0 posture** — now a genuine differentiator against Elastic's access-token-gated
  on-prem package.
- **GLM-OCR as the default engine** — pending the CJK eval.

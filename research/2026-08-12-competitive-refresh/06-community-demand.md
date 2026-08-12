# Community feature demand — what users are actually asking for

**Research date:** 2026-08-12. Sources: competitor GitHub issue trackers (reaction counts read
via the GitHub API on this date), aggregated Reddit/practitioner reporting, and 2026 industry
analyses.

## How to read the evidence column

| Grade | Meaning |
|---|---|
| **A** | Primary source with a countable signal — GitHub reactions, official release notes |
| **B** | Secondary aggregation citing specific numbers, but the underlying study was not read directly |
| **C** | Single-source assertion; directionally useful, not citable in customer-facing material |

Anything graded **B** or **C** should be re-verified before it appears in marketing. The
2026-08-12 refresh deliberately grades rather than launders.

---

## The ranked demand table

| # | What users want | Evidence | Grade | Veda status |
|---|---|---|---|---|
| 1 | **Hybrid search (BM25 + dense) built in** | Chroma [#1330](https://github.com/chroma-core/chroma/issues/1330) — **78 👍**, open since Nov 2023, still active Jun 2026; the single most-reacted open request in the repos surveyed | **A** | **HAS** — BM25 + SPLADE + dense, RRF/weighted-sum/DBSF |
| 2 | **Schema evolution on a non-empty collection** | Milvus [#20405](https://github.com/milvus-io/milvus/issues/20405) — top enhancement request ("add/drop fields without recreating and re-ingesting"); Milvus 3.0 answered part of it (dynamic field addition on live collections); Weaviate 1.39 shipped alter-schema reindex/drop-vector-index in preview | **A** | **PARTIAL** — `/reindex` builds a sibling + swaps the alias, which re-ingests. No in-place add/drop field |
| 3 | **Embedding-model selection is guesswork** | Reddit-derived: a model scoring **92 on MTEB reached 61 % precision on biomedical**; switching to a domain model reached 84 % at $3,400/mo. Independently: MTEB top-3 placed 5th/7th/2nd on an in-domain legal eval, where **11th-ranked BGE-large-en-v1.5 won** | **B** | **HAS — and under-marketed.** Tier-2/3 auto-tuning searches embedder + chunker against *your* corpus |
| 4 | **Chunking silently destroys accuracy** | Reddit: 400-comment r/MachineLearning thread on fixed-size chunking; one financial-services firm went **71 % → 89 %** moving to semantic splitting. Separately: chunking choice can swing recall **up to 9 %** on the same corpus | **B** | **HAS** — 7+ strategies incl. `Hierarchical` and late chunking, **and** chunker A/B + tier-3 search |
| 5 | **Hallucination detection that works** | Reddit: "**43 % of RAG answers contain at least one unsupported claim** when evaluated against retrieved context"; post-hoc verification catches 68–73 % | **B** | **HAS** — faithfulness + citation scoring, LLM-judge grounding on `/converse` |
| 6 | **Evaluation you can trust** | Reddit: "**43 % gap between vendor-reported accuracy and NIST framework scores**"; one system fell from 90 % to below 70 % under full NIST evaluation | **B** | **HAS** — eval harness, recall@k / faithfulness / answer-relevancy / pass-rate, eval-set generation from the user's own corpus |
| 7 | **Cost attribution and control** | Reddit: a team budgeted **$8,000 and received a $47,000 bill**; semantic caching cut costs **43 %**; re-embedding costs run **3–5× initial indexing annually** | **B** | **HAS** — semantic response cache (verified `cache_hit`, 0 LLM tokens), `veda_llm_cost_usd_total`, token/cost metrics |
| 8 | **RAG-specific security** | **OWASP LLM Top 10 updated 2026-07-12** with three new RAG threat vectors: prompt injection **through retrieved documents**, **indirect context poisoning**, **temporal inconsistency attacks**; reportedly **89 % of surveyed organizations exposed**; a PoC showed a malicious Confluence document causing exfiltration | **B** | **PARTIAL — the clearest product gap.** Guardrails exist but run in **Flag (detect-only)** mode at a 0.65 cosine threshold, targeting prompt injection/PII in the *user's* input. Nothing defends against poisoned *retrieved* documents or temporal inconsistency |
| 9 | **Multi-hop / multi-document reasoning** | Reported NIST analysis (Jul 2026): **73 % of enterprise RAG deployments fail informal accuracy audits on multi-hop reasoning**; multi-hop showed a 62 % discrepancy vs vendor claims. TreeRAG and knowledge-graph hybrids are the proposed answers | **B/C** | **GAP** — single-shot retrieval + an agent loop. No graph/tree retrieval, no explicit multi-hop planner |
| 10 | **Object-storage persistence** | Chroma [#1736](https://github.com/chroma-core/chroma/issues/1736) — 11 👍, explicitly citing LanceDB as having it. Reinforced by the whole object-storage-native shift in `02`/`03` | **A** | **PARTIAL** — MinIO/S3 for blobs; the index tier is not object-storage-native |
| 11 | **Sorting / ordering / aggregation in queries** | Chroma [#469](https://github.com/chroma-core/chroma/issues/469) — 18 reactions, opened by a Chroma collaborator; Milvus 3.0 shipped server-side sorting, aggregation and faceted search; OpenSearch star-tree | **A** | **HAS** — `/aggregate`, `/histogram`, `/field-stats`; **no faceted search** |
| 12 | **Multi-agent orchestration cost** | Reddit: 1→3 agents increased latency **2.7×**; 7 agents spent more time coordinating than retrieving; one incident burned $180 in credits before timeout | **B** | **PARTIAL** — single-agent converse + MCP/A2A; no orchestration cost controls |
| 13 | **Hybrid search is expensive** | Reported: hybrid costs **2.3× storage and 1.7× retrieval latency** — enough that teams debate whether the accuracy is worth it | **B** | **OPPORTUNITY** — Performance-mode tuning already finds the cheapest config above a recall floor. This is the exact objection it answers |
| 14 | Async client support | Chroma [#2200](https://github.com/chroma-core/chroma/issues/2200) — small but persistent | **A** | N/A — Veda is HTTP-first; the Python SDK should be checked |

---

## The three conclusions that matter

### 1. Veda already answers most of the top demand — and says so badly

Items **1, 3, 4, 5, 6, 7, 11** are all shipped. Items **3, 4, 5, 6, 7** are precisely the
"RAG quality" layer that `competitive_advantages.md` lists as claim #3, and the community
evidence for how badly they hurt is much stronger than anything in our own positioning:

> A model scoring **92 on MTEB** delivered **61 % precision** on biomedical. **43 %** of RAG
> answers carry an unsupported claim. A **43 % gap** between vendor-reported and NIST-measured
> accuracy. Chunking swings recall **9 %**. Semantic caching cuts cost **43 %**.

Every one of those numbers is an argument for a feature Veda already has. **The highest-return
action from this entire refresh is not engineering — it is restating the RAG-quality layer in
terms of these numbers.**

### 2. The clearest product gap is RAG-specific security

The OWASP LLM Top 10 added three RAG threat vectors on 2026-07-12. Veda's guardrails are
embedding-similarity checks on **input**, in **Flag** mode. The new threats are about
**retrieved content** — a poisoned document in the corpus, not a hostile user prompt.

Veda is unusually well placed to fix this: it owns the ingestion pipeline, the embedder and the
retrieval path, so it can score *retrieved* chunks for injection patterns before they reach the
LLM — something a bring-your-own-vector-DB stack structurally cannot do. This is a genuine
"only Veda can do this" feature, and it is currently missing.

### 3. Multi-hop is the loudest unmet need in the category

Reported at 73 % failure rate. Neither Veda nor the mainstream vector DBs answer it; the answers
being proposed are TreeRAG, knowledge graphs and agentic decomposition. Veda has the pieces
(agent loop, tools, hierarchical chunks, eval harness to measure it) but no explicit multi-hop
retrieval strategy. Worth a design discussion before it becomes table stakes.

---

## Recommendation order for this file's scope

1. **Reposition the RAG-quality layer around the community's own numbers.** Zero engineering.
2. **Guardrails for retrieved content** (indirect/contextual injection) + move guardrails beyond
   Flag-only. Addresses OWASP 2026-07-12; structurally advantaged for Veda.
3. **In-place schema evolution** (add/drop non-primary fields without a full reindex) — the #2
   demand signal, and Weaviate/Milvus both moved on it in the last quarter.
4. **Faceted search** to complete the analytics story against Milvus 3.0 / OpenSearch star-tree.
5. **Scope multi-hop retrieval** — design discussion, not implementation.
6. Verify the **Python SDK** exposes async.

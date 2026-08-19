# Graph and memory layers: does ontology-grounding actually work, and is multimodal viable?

**Research date:** 2026-08-20. This file exists to answer two questions the plan named, both of
which could have overturned the prior anti-knowledge-graph verdict in
`documentation/2026-08-17-multi-hop-retrieval-scope.md`. One is answered yes-with-conditions;
the other is answered more favourably than that document assumed.

---

## Q1. Does ontology-grounding beat plain GraphRAG, on a primary source?

### Yes — OG-RAG, and it is peer-reviewed (grade A)

**[OG-RAG: Ontology-Grounded Retrieval-Augmented Generation For Large Language Models](https://arxiv.org/abs/2412.15235)**,
Kartik Sharma, Peeyush Kumar, Yunqing Li. arXiv:2412.15235, submitted 2024-12-12; published at
**[EMNLP 2025 main](https://aclanthology.org/2025.emnlp-main.1674/)**. Abstract re-fetched and
verified 2026-08-20.

Mechanism: build a **hypergraph** over domain documents where each hyperedge is a cluster of
facts grounded in a domain-specific ontology, then run an optimisation that retrieves the
*minimal* set of hyperedges covering the query. Reported in the abstract:

- **+55%** recall of accurate facts
- **+40%** response correctness
- **30%** faster attribution of responses to context
- **+27%** fact-based reasoning accuracy

Three details decide how much this transfers to Veda, and all three cut the same way:

1. **The ontology is an input, not an output.** The abstract "anchors retrieval processes in
   domain-specific ontologies" — pre-existing ones are leveraged, not built. A Veda user who
   already owns an ontology is a different customer from one who does not, and the second is the
   common case.
2. **It is a hypergraph over documents, not a triple store.** This is the single most useful
   architectural finding in this file: the win came from *structured grouping of facts plus a
   coverage-optimising retrieval objective*, not from RDF, OWL or a graph database. That
   decouples the result from ontobricks' layers 2–3.
3. **The baseline is standard RAG.** The gain is measured against vanilla retrieval, not against
   a tuned hybrid retriever with reranking — which is what Veda ships.

### Weakly — Graphwise "Semantic GraphRAG" on MuSiQue (grade C)

The number circulating as "ontology halves the errors" traces to
[Graphwise's blog](https://graphwise.ai/blog/from-retrieval-to-reasoning-enhancing-hipporag-with-graph-based-semantics/)
(accessed 2026-08-20). Chased to source, the caveats are disclosed by the authors themselves and
are large:

| System | 2-hop | multi-hop (2+3+4) |
|---|---|---|
| LLM, no RAG | 71% | — |
| Vector RAG | 79% | ~75% |
| HippoRAG | 86% | ~75% |
| Semantic GraphRAG | **95%** | **86%** |

- **n = 100** randomly sampled MuSiQue questions.
- **LLM-as-judge**, not exact match.
- The authors state **"approximately a third of the questions are not answerable — usually the
  expected answers are wrong or impossible to derive"**, and unanswerables were filtered out.
- The ontology was **LLM-generated** by "an LLM agent that acts as a schema architect", not
  hand-authored — which is interesting for Veda (it means the ontology need not be a human
  deliverable) and simultaneously means the "ontology" is a model artifact of unaudited quality.
- Published by the vendor whose platform wins.

**Grade C. Do not cite this number in Veda material.** It is directionally consistent with
OG-RAG, and that is all it supports.

### What both results are actually about

Both target **entity-join multi-hop retrieval on text**. That is precisely the workload Veda
measured at **100%** (one-to-one 12/12, one-to-many 6/6) in `05`. The externally-demonstrated
win is in a place Veda's measured numbers say is not broken. This is the crux of the whole
cycle, and it holds regardless of how the two sources are graded.

---

## Q2. Is multimodal knowledge-graph construction viable in 2026?

The prior scoping document's strongest argument against a knowledge graph was:

> "multimodal corpora (Veda's differentiator) have no good entity extractor for images/audio"

That was asserted without a source. **It is now too strong, and the doc should be narrowed.**

- **[M³KG-RAG: Multi-hop Multimodal Knowledge Graph-enhanced RAG](https://arxiv.org/abs/2512.20136)**,
  **CVPR 2026** (submitted 2025-12-23, final 2026-04-13; project page
  `kuai-lab.github.io/cvpr2026m3kgrag`). Builds an audio-visual MMKG via "a lightweight
  multi-agent pipeline" producing "context-enriched triplets of multimodal entities", plus GRASP
  (Grounded Retrieval And Selective Pruning) for entity grounding and redundancy removal.
  Reports improvements on "diverse multimodal benchmarks". **Grade B** — peer-reviewed venue,
  but the accessible abstract does not give per-benchmark numbers or extraction-quality figures.
- Supporting context (grade B/C): TIVA-KG (text/image/video/audio MKG), the CVPR 2026 CASTLE
  challenge entry using hierarchical KG retrieval over long-context video, and the
  *Knowledge Graphs Meet Multi-Modal Learning* survey (arXiv:2402.05391).

**Corrected position:** multimodal KG construction is an **active, published research area with
working pipelines**, not an unsolved problem. What remains true, and is documented in the same
literature, is that quality is bounded by the extractors — surveys name ontology alignment, data
quality, interpretability and query standardisation as open, and note that KG coverage
"underrepresents long-tail entities, rare relations, or domains not well captured in source
datasets", with cross-modal miscalibration surfacing off-topic neighbours.

So the honest version of the anti-KG argument is not "it is impossible". It is: **research-grade,
extractor-bounded, and it would add a second quality-critical model to Veda's ingest path whose
failures are silent.** That is still a strong argument against building it now — but it is a
cost argument, not an impossibility argument, and `00-summary.md` states it that way.

---

## The adjacent product category: agent-memory graphs

Relevant because it is where the OSS energy is, and because it is **not** what Veda is.

| System | Bet | Notes |
|---|---|---|
| **Graphiti / Zep** | *Temporal* depth — how facts change over time | Zep reports 63.8% on LongMemEval with GPT-4o (**grade C**, vendor-published) |
| **Cognee** | *Source breadth* — PDFs, Slack, Notion, images, audio into one queryable KG | Benchmarks itself against Mem0/Graphiti/LightRAG on HotPotQA (**grade C**, vendor-published) |
| **Microsoft GraphRAG / LightRAG** | Batch KG construction over a static corpus | The batch foil to Graphiti's incremental ingestion |

Two things to take from the category:

1. **They are memory layers you bolt onto an agent, and each runs a paid cloud on top of the
   OSS.** They sit *above* a vector store; several use one underneath. They are potential
   consumers of Veda, not head-to-head competitors — and a Veda that emits good semantics is
   more useful to them, which is the composability argument from `02`.
2. **Every benchmark in the category is published by the party that wins it.** Cognee
   benchmarks Cognee; Zep benchmarks Zep. All grade C. There is no neutral leaderboard for
   agent-memory KGs comparable to MTEB or BEIR, and that absence is itself worth knowing before
   entering the category.

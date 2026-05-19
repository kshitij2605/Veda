# SOTA Audit: Sparse Retrieval, Fusion, and Reranking (2026-05-19)

> Scope: Veda's current choices (SPLADE, BM25, RRF + weighted-sum, generic HTTP cross-encoder) measured against state-of-the-art alternatives as of mid-2026.

---

## Verdict (TL;DR)

**Sparse retrieval.** SPLADE-v3 (Naver, March 2024) is still the strongest open learned-sparse model in production use today. SPLADE++ / "SPLADE-v3" achieves nDCG@10 ~51.7 on BEIR-13 and MRR@10 ~40.2 on MS MARCO, beating BM25 by 5-7 points on most BEIR slices. The 2025 follow-up SPLARE has shown competitive results against Qwen3-8B-Embed but is research-grade and not yet packaged. The Sentence-Transformers v5 `SparseEncoder` API (June 2025) has standardised SPLADE/inference-free-SPLADE loading. **Verdict:** SPLADE is still SOTA for learned sparse - keep it. BM25 remains load-bearing as the cold-start / out-of-domain anchor (it does not need training). uniCOIL/DeepImpact/TILDEv2 are deprecated relative to SPLADE-v3.

**Fusion.** RRF with k=60 remains the right unsupervised default and what Pinecone/Qdrant/Vespa/Elastic/Azure-AI-Search all ship. **However**, the published literature is clear: when even a handful of labelled queries exist, **convex combination (weighted sum) beats RRF on every BEIR slice tested** (Bruch et al., TOIS 2023; replicated in 2024-2025). The 2025 DAT (Dynamic Alpha Tuning) paper shows per-query α tuning via an LLM beats fixed-weight CC. Distribution-Based Score Fusion (DBSF) and Sigmoid-normalised CC are competitive with CC and cheaper than learned fusion. **Verdict:** Keep RRF as default (zero-shot, robust), expose CC as opt-in, and add DBSF as a third option. Don't ship "learned fusion" - it duplicates what a reranker already does.

**Reranking.** This is the area with the most movement in 2025-2026. As of today the open-source SOTA on BEIR nDCG@10 is **jina-reranker-v3** (0.6B, 61.94 nDCG@10) and **mxbai-rerank-large-v2** (1.5B, 57.49 nDCG@10), both released 2025. On proprietary APIs **Voyage rerank-2.5** (Aug 2025) leads, claiming +7.94% over Cohere rerank-3.5 on Voyage's 93-dataset suite, with 32K context. **Cohere rerank-3.5** (Dec 2024) is the current incumbent default in most production stacks. **ZeroEntropy zerank-1** (4B open-weight, Apache-2.0 for the small variant) is the new entrant on a Voyage/Cohere-shape API. The right 2026 default depends on closed/open: closed-source Voyage rerank-2.5 (or rerank-2.5-lite at half the cost); open-weight jina-reranker-v3 if listwise is acceptable, mxbai-rerank-large-v2 if pairwise cross-encoder is preferred. Cohere remains best for "no decisions" production reliability.

### Top 3 Gaps

1. **Veda has no native rerank-2.5 / instruction-following reranker support.** The current rerank.rs is a generic Voyage/Cohere-shape POST. It works with Voyage rerank-2.5 today but does not expose the 32K-context or `instruction` parameter that is the entire value-add of 2.5. Adding an optional `instruction` field unlocks a documented +12.7% vs Cohere on MAIR.
2. **Fusion is stuck at RRF + weighted-sum.** No CC (with proper score normalisation) and no DBSF. CC is documented to be sample-efficient (needs ~50 labelled queries) and outperforms RRF on every measured slice. DBSF is the cheaper, label-free middle ground.
3. **No late-interaction (ColBERT/ColPali) rerank path.** For multimodal/visual-document RAG (a stated Veda goal) the SOTA pipeline in 2026 is `dense → ColPali MaxSim → optional cross-encoder`. Veda's pipeline currently caps out at dense + cross-encoder text rerank, which leaves visual-document precision on the table.

### Top 3 Wins (Veda is already competitive)

1. **SPLADE choice is correct.** SPLADE-v3 is still SOTA for learned sparse. The external-endpoint architecture (env-configured) is exactly right: SPLADE is a 100M-parameter BERT model that wants its own GPU, and Veda's `veda-storage` only needs the resulting sparse vector.
2. **RRF as default is correct.** Every major vector DB defaults to RRF k=60 in 2026 (Pinecone, Qdrant, Vespa, Elastic, OpenSearch, Azure AI Search). Veda picking the same default means it composes with the rest of the ecosystem.
3. **Generic Voyage/Cohere-shape rerank API.** This is the right abstraction. Voyage rerank-2.5, Cohere rerank-3.5, ZeroEntropy zerank-1, mxbai-rerank-large-v2 (via TEI/vLLM), and bge-reranker-v2.5-gemma all conform to roughly the same JSON shape. Veda already supports any of them by env config.

### Recommendation (single sentence)

**Keep SPLADE + BM25 + RRF as defaults; add (a) an `instruction:` passthrough to the rerank HTTP body for Voyage-2.5 / zerank-2 compatibility, (b) a CC + DBSF fusion option in `veda-query::fusion`, and (c) document Cohere rerank-3.5 as the recommended production default and jina-reranker-v3 as the recommended open-weight default in `documentation/`.**

---

## A. Sparse Retrieval

### A.1. SPLADE-v3 (current SOTA learned sparse)

SPLADE-v3 was released March 11, 2024 by Naver Labs (Lassance, Déjean, Formal, Clinchant). It is the public successor to SPLADE++ and trained with:

- 8 negatives per query
- KL-Div + MarginMSE distillation losses
- Ensemble teacher distillation

Reported numbers (from the SPLADE-v3 paper, arXiv:2403.06789):

| Model | MS MARCO MRR@10 | BEIR-13 avg nDCG@10 |
|---|---|---|
| BM25 | ~18.4 | ~43.0 |
| SPLADE++ (SelfDistil) | 38.9 | 50.7 |
| **SPLADE-v3** | **40.2** | **51.7** |
| SPLADE-v3-Doc (inference-free) | 38.0 | 50.0 |

Output dimension is 30,522 (BERT vocab). Sparsity ratio at inference is typically 100-200 non-zero terms per document.

**Verdict:** SPLADE-v3 is the right learned-sparse for Veda in 2026. Naver's "SPLARE" (Jan 2025 WSDM Cup, arXiv:2602.20986) and "SPLATE" (sparse late interaction adapter on ColBERTv2) exist but are not packaged for production. The Sentence-Transformers v5.0 `SparseEncoder` (June 2025) provides a standardised loader, including "Inference-Free SPLADE" variants where queries are computed by a small CPU model and documents are pre-computed.

### A.2. SPLADE++ Sparse-Encoder-1-v1 and "next SPLADE"

The community-rebadged "sparse-encoder-1-v1" releases on Hugging Face are repackagings of SPLADE-v3 + Inference-Free-SPLADE under the Sentence-Transformers Sparse Encoder API. No fundamentally newer SPLADE generation has shipped as of May 2026. SPLARE (Naver, 2025) showed competitive numbers against Qwen3-8B-Embed but is research-only.

### A.3. uniCOIL / DeepImpact / TILDEv2 (deprecated relative to SPLADE-v3)

Per the unified learned-sparse-retrieval framework (Nguyen et al., SIGIR 2023):
- DeepImpact and TILDEv2 are essentially uniCOIL without a query encoder.
- They have near-instant query encoding (no neural pass at query time) but no query expansion → strictly worse recall than SPLADE-v3.
- DeeperImpact (arXiv:2405.17093, May 2024) optimises the sparse index structures but does not move SOTA on relevance.

**Verdict for Veda:** Do not add uniCOIL/DeepImpact/TILDEv2. If "fast-query-encoding" matters (e.g. on-CPU clients), use Inference-Free SPLADE instead - it is in the same package as SPLADE-v3 and has stronger relevance numbers.

### A.4. BM25 variants (BM25F, BM25+, ATIRE BM25)

The classic BM25 in `veda-storage` is what every system in 2026 still ships. The named variants:

- **BM25F** - field-weighted BM25 (title vs body). Useful if Veda exposes per-field weights via the schema. Worth considering as a feature flag, but Pinecone/Qdrant/Vespa hybrid all default to plain BM25.
- **BM25+** (Lv & Zhai 2011) - addresses the "long document penalty" of BM25. Negligible practical benefit in modern hybrid pipelines.
- **ATIRE BM25** - a numerical-stability variant. Not worth the engineering.

**Qdrant's BM42** (mid-2024) attempted to replace BM25 with an attention-derived sparse representation. The community evaluations (Q3-Q4 2024) raised methodological concerns and most users reverted to BM25 + SPLADE. Don't follow Qdrant down this path.

**Verdict:** Keep classic BM25. The simplest correct choice.

### A.5. Sparse-only vs Hybrid in 2026

Dense embeddings have **not** replaced sparse. Every major hybrid evaluation in 2024-2026 (Elastic, Vespa, Weaviate, MongoDB, OpenSearch) reports the same finding:

- Dense alone: strong on semantic / paraphrase queries, weak on rare terms, named entities, codes.
- Sparse alone: strong on exact match, weak on synonymy.
- Hybrid (dense + sparse + fusion): 30-40% recall improvement vs either alone in production e-commerce / enterprise search.

Fine-tuned SPLADE alone achieves nDCG@10=0.388 on Amazon ESCI vs BM25 0.301 (a 29% relative gain), but a SPLADE+dense hybrid beats SPLADE-alone by another 5-10%.

**Conclusion:** Sparse is still load-bearing. Veda's choice to keep BM25 + SPLADE alongside dense is correct.

---

## B. Fusion

### B.1. RRF tuning - k=60 is still the right default

The original Cormack et al. (2009) paper proposed k=60 from a single empirical sweep. Replications in 2024-2025 (OpenSearch, Azure AI Search, Elastic) all converge on k ∈ [40, 80] as the practical sweet spot, with k=60 the dominant default.

Mechanically:
- Low k (k<10): top-1 from a single retriever dominates the fusion. Brittle.
- High k (k>200): differences between rank-1 and rank-50 nearly disappear. Consensus voting.
- k=60: rank-1 contributes 1/61 ≈ 0.0164; rank-10 contributes 1/70 ≈ 0.0143. ~13% premium for rank-1, which empirically matches what humans want.

**Verdict:** k=60 default in Veda is correct. Expose `k` as a tunable parameter (likely already done) but no reason to change the default.

### B.2. Weighted sum (convex combination) - when it beats RRF

The Bruch et al. paper "An Analysis of Fusion Functions for Hybrid Retrieval" (arXiv:2210.11934, TOIS 2023) is the canonical reference. Key findings, replicated in 2024-2025:

1. **CC beats RRF on every BEIR slice tested** in supervised (with-labels) settings, by an average of 2-4 nDCG@10 points.
2. **CC is sample-efficient** - tuning α typically requires only 50-200 labelled queries.
3. **CC is robust to score normalisation choice** (min-max, z-score, sigmoid all work).
4. **RRF wins in fully zero-shot scenarios** where no labels exist and score distributions are wildly different.

The 2025 follow-up by AutoRAG ("Weights in Hybrid Retrieval") confirms α=0.5 (equal weighting) is a strong default when you can normalise scores. The 2025 DAT paper (arXiv:2503.23013) goes further: an LLM picks α per query, beating fixed-α CC.

**Verdict for Veda:**
- Default: RRF k=60 (current).
- Opt-in: CC with min-max normalisation, default α=0.5. Document that 50 labelled queries beats RRF.
- Do not implement DAT - the latency of an LLM call per query defeats the point of fast first-stage fusion.

### B.3. Sigmoid normalisation / Distribution-Based Score Fusion (DBSF)

DBSF (Mazzeschi, 2023) normalises scores using mean ± 3σ per retriever, then sums. The Qdrant 1.11 release (Q3 2024) shipped DBSF as a first-class option. Findings:

- DBSF is competitive with CC when no labels are available.
- DBSF is more sensitive to outliers in the score distribution than RRF.
- DBSF is strictly better than naive min-max sum when the two retrievers have wildly different score scales (e.g. BM25 raw vs cosine similarity).

**Verdict:** DBSF is a cheap addition (~30 lines of Rust in `veda-query::fusion`). Worth shipping as a third option alongside RRF and CC. It is what Qdrant defaults to when CC is not tuned.

### B.4. Learned fusion (neural rerankers as fusers)

Some papers propose using a small neural network to learn the fusion weights end-to-end. Empirical finding: this is a strictly worse use of the same neural budget than just running a cross-encoder reranker on top-k. A reranker sees the query and document text; a learned fuser only sees the two scalar scores.

**Verdict:** Do not implement learned fusion in Veda. The compute should go into the rerank stage, not the fusion stage.

### B.5. Cross-modal fusion (text + image)

The honest answer: **there is no consensus SOTA for cross-modal score fusion in 2026.** Two dominant paths:

1. **Unified embedding space** (e.g. CLIP, SigLIP, Qwen3-VL-Embedding): one vector per item, text and image embedded into the same space, normal vector search. Score fusion is unnecessary. This is what most multimodal RAG systems do in 2026.
2. **Late-interaction multimodal** (ColPali, ColQwen, Nemotron-ColEmbed-v2): query tokens vs document patches via MaxSim. The "fusion" happens inside the model.

If you keep separate text and image retrievers (which Veda does), RRF k=60 is the default cross-modal fuser used by Vespa, LanceDB, and Weaviate. The Elasticsearch Labs hybrid post (2024) confirms RRF is robust across modalities.

**Verdict:** RRF k=60 is the right cross-modal fuser default for Veda's current architecture. If Veda later supports ColPali, fusion of patch-level scores becomes the model's responsibility, not Veda's.

---

## C. Reranking

This is the area with the most 2025-2026 movement. The current landscape:

### C.1. Proprietary cross-encoders

| Model | Released | Context | BEIR nDCG@10 | Pricing (per 1M tokens) | Notes |
|---|---|---|---|---|---|
| Cohere rerank-3.5 | Dec 2024 | 4096 | ~55.4 | $2.00 | 100+ languages, multilingual SOTA at release |
| Voyage rerank-2.5 | Aug 2025 | 32K | ~58-60 (claimed) | $0.05 | Instruction-following, +7.94% vs Cohere v3.5 |
| Voyage rerank-2.5-lite | Aug 2025 | 32K | ~57-58 (claimed) | $0.02 | Same arch, smaller; +7.16% vs Cohere |
| ZeroEntropy zerank-1 | Q1 2025 | 4096 | competitive | $0.025 | 4B params, ELO-pairwise training |
| ZeroEntropy zerank-1-small | 2025 | 4096 | competitive | open-weight | Apache-2.0 |

**Cohere rerank-3.5** (Dec 2024) is the safe enterprise default. AWS Bedrock, Azure, Oracle all ship it. The biggest win vs rerank-3 is multilingual reasoning quality on Finance/E-commerce/Hospitality domains.

**Voyage rerank-2.5** (Aug 2025) is the current proprietary leader. Two big differentiators:
1. **32K context** vs Cohere's 4K - actually useful for long document chunks.
2. **Instruction-following** via an `instruction` field in the API. Voyage claims +12.7% vs Cohere on the MAIR (Massive Instructed Retrieval) benchmark.

Voyage rerank-2.5-lite is half the cost and gives most of the quality. This is the right default-of-defaults for cost-sensitive deployments.

**ZeroEntropy zerank-1** is the new entrant. Trained with their "zELO" pairwise method. Claims +28% NDCG@10 over first-stage retrieval and >+5% over Cohere v3.5 on every domain. Half the cost of Cohere. The small variant (zerank-1-small) is open-weight Apache-2.0.

### C.2. Open-weight cross-encoders

| Model | Params | Released | BEIR nDCG@10 | License |
|---|---|---|---|---|
| bge-reranker-v2-m3 | 568M | Mid 2024 | ~54-55 | Apache-2.0 |
| bge-reranker-v2-gemma | 2B | Mid 2024 | ~55-56 | Apache-2.0 |
| **bge-reranker-v2.5-gemma2-lightweight** | 9B | Jul 2024 | claimed SOTA at release | Apache-2.0 |
| **mxbai-rerank-base-v2** | 0.5B | Q1 2025 | 55.57 | Apache-2.0 |
| **mxbai-rerank-large-v2** | 1.5B | Q1 2025 | **57.49** | Apache-2.0 |
| jina-reranker-v2-base-multilingual | 278M | 2024 | 57.06 | CC BY-NC (commercial-restricted) |
| **jina-reranker-v3** | 600M | Q3 2025 | **61.94** | CC BY-NC |
| Qwen3-Reranker-0.6B | 600M | Jun 2025 | competitive | Apache-2.0 |
| Qwen3-Reranker-4B | 4B | Jun 2025 | ~SOTA-class | Apache-2.0 |
| Qwen3-Reranker-8B | 8B | Jun 2025 | matches Contextual rerank-v2 | Apache-2.0 |

**bge-reranker-v2-m3** is the workhorse of the open ecosystem - multilingual, 568M params, fits on a single T4. Recommended for "I need an open reranker that just works on multilingual data and runs cheaply."

**bge-reranker-v2.5-gemma2-lightweight** was BAAI's SOTA play in 2024 (9B params, gemma2-based, with token compression + layerwise lightweight inference). Has not been overtaken by a bge-v3 series in 2026.

**mxbai-rerank-large-v2** (Q1 2025) sits at 57.49 BEIR - the highest open-weight cross-encoder by a small margin until jina-v3 shipped. Trained with GRPO + contrastive + preference learning. Apache-2.0. 1.5B params. 8K context. Said to be 8x faster than bge-reranker-v2-gemma despite being smaller.

**jina-reranker-v3** (Sep 2025, arXiv:2509.25085) is the current open-weight SOTA: 61.94 BEIR nDCG@10. Architecture is a "last but not late" interaction - causal attention between the query and all candidates in the same context, then extracting embeddings from the last token of each document. 600M params, 2.5x fewer than mxbai-large-v2 yet stronger. **License is CC BY-NC** - not Apache - which is a real concern for commercial use of Veda. Document this.

**Qwen3-Reranker** family (Jun 2025) - 0.6B / 4B / 8B sizes, all Apache-2.0. The 8B matches Contextual AI's open-source rerank-v2. The 4B is the recommended quality/cost balance.

### C.3. Listwise rerankers (RankGPT / RankVicuna / RankZephyr)

The listwise paradigm has the LLM see all candidates simultaneously and reorder them in one pass. State of play:

- **RankGPT (GPT-4)** - reference quality but $$$ and 5+ second latency.
- **RankZephyr** (arXiv:2312.02724) - open-source 7B model, rivals and occasionally beats RankGPT4 on BEIR's NEWS and COVID slices.
- **RankLLM toolkit** (Castorini, accepted SIGIR 2025) - the canonical implementation. FirstMistral inside RankLLM achieved the highest nDCG@10 on 8/11 datasets in their evaluation.

ZeroEntropy's 2026 guide concedes listwise wins on NDCG@10 (often 0.90+ for small k) but at 5+ seconds latency and 10-100x the cost of cross-encoders.

**Verdict for Veda:** Do not make listwise the default. It is a research/specialised choice. The generic HTTP rerank API in `veda-gateway` already supports it (any backend that scores top-k can be called); document this as a "high-precision mode" but do not ship it as default.

### C.4. Late-interaction as rerank (ColBERT / ColPali)

The 2026 multi-stage SOTA pipeline for visual-document retrieval:

```
1. Dense retrieve top-1000 (cheap bi-encoder)
2. ColPali/ColQwen MaxSim rerank to top-100 (multi-vector late interaction)
3. (Optional) cross-encoder rerank to top-10
```

For text-only retrieval, the equivalent pipeline uses ColBERTv2 instead of ColPali at step 2. The MaxSim rerank step is sublinear in candidate count and very fast on GPU.

The Weaviate/LanceDB/Vespa late-interaction blog posts (2024-2025) converge on the same finding: late-interaction as rerank is the right architecture when you have multi-vector indices, but the storage cost is 100-1000x a single-vector index. For most production RAG, a 1.5B cross-encoder is the better trade.

**Verdict for Veda:** Late-interaction is a separate index type (multi-vector). It is in scope for Veda eventually (the project has DiskANN/HNSW listed for multi-vector). But as a rerank choice it is orthogonal to the rerank.rs HTTP path - it would live in `veda-index`. Not a 2026 priority unless visual-document RAG is the main use case.

### C.5. Multi-vector vs single-vector rerank

| Approach | Storage cost | Latency | Quality | Use case |
|---|---|---|---|---|
| Single-vector cross-encoder | 1x | 100-500ms | High (BEIR 55-62) | Default RAG |
| Multi-vector late interaction | 50-200x | 10-50ms | High (similar BEIR) | Multimodal, low-latency |
| Listwise LLM | 1x (no index) | 2-10s | Highest | Research, top-precision |

**Verdict:** Default to single-vector cross-encoder reranker. Multi-vector late interaction is a separate (large) project.

---

## D. Hybrid Retrieval as a Whole - what the leaders default to in 2026

| Vector DB | Default sparse | Default dense | Default fusion | Rerank shipped? |
|---|---|---|---|---|
| **Pinecone** | proprietary sparse format | proprietary | RRF (server-side) | Yes (Cohere/own) |
| **Qdrant** | BM25 (server-side IDF since 1.15.2) or SPLADE | dense HNSW | RRF or DBSF | No (call out to provider) |
| **Vespa** | weakAnd over inverted index | nearestNeighbor (HNSW) | Configurable rank-profile, RRF available | No (rank-profile is the rerank) |
| **LanceDB** | BM25 (FTS) | dense | RRF | Yes (built-in rerankers) |
| **Weaviate** | BM25 | dense | RRF (k=60) | Yes (Cohere/Voyage/Jina) |
| **Elasticsearch** | BM25 | dense (HNSW) | RRF (k=60) | Yes (inference API) |
| **OpenSearch** | BM25 | dense | RRF (k=60) | Yes (inference connectors) |
| **MongoDB Atlas** | BM25 (text) | dense | RRF | Yes (Voyage native) |

**Pattern observed:**
1. Every major system ships RRF as the default fusion.
2. Most default to BM25 for sparse. Only Pinecone and Qdrant ship learned-sparse first-class.
3. Most ship rerank as a separately-configured stage, not part of the core query.
4. The Voyage / Cohere / Jina HTTP shape is the universal rerank ABI.

**Veda's current architecture matches the leader-pack defaults.**

---

## Concrete Recommendations for Veda

In priority order:

### P0 - Document the recommended defaults (no code changes)

In `documentation/`, add a "Recommended Reranker" section:
- **Closed-source default:** Cohere rerank-3.5 (4K context, 100+ languages, AWS/Azure/Oracle availability).
- **Cost-optimised closed-source:** Voyage rerank-2.5-lite ($0.02/1M tokens, 32K context).
- **Open-weight default (Apache-2.0):** mxbai-rerank-large-v2 (1.5B, 57.49 BEIR) or Qwen3-Reranker-4B.
- **Open-weight SOTA (non-commercial only):** jina-reranker-v3 (CC BY-NC license).
- **Multilingual workhorse:** bge-reranker-v2-m3 (568M, Apache-2.0).

### P1 - Add `instruction` field passthrough in `veda-gateway::rest::rerank.rs`

Voyage rerank-2.5 takes an `instruction: string` field for natural-language steering. Veda's generic HTTP body should optionally forward this. ~10 lines.

### P2 - Add CC (convex combination) and DBSF to `veda-query::fusion`

Side-by-side with RRF. Default still RRF k=60.

- CC: takes `weight_dense`, `weight_sparse` (default 0.5/0.5), normalises via min-max.
- DBSF: takes no parameters, normalises each retriever's scores by mean ± 3σ.

Documented use:
- RRF: no labels available, default.
- DBSF: no labels, score distributions wildly different.
- CC: ≥50 labelled queries available, want max nDCG.

### P3 - (Future) Late-interaction as a separate retrieval path

Track ColPali/ColQwen support as an index type in `veda-index`. Not a 2026 sprint goal.

---

## References

- SPLADE-v3 paper, arXiv:2403.06789 (Naver Labs, Mar 2024)
- "An Analysis of Fusion Functions for Hybrid Retrieval," Bruch et al., TOIS 2023, arXiv:2210.11934
- "DAT: Dynamic Alpha Tuning for Hybrid Retrieval," arXiv:2503.23013 (2025)
- jina-reranker-v3 paper, arXiv:2509.25085 (Sep 2025)
- "RankZephyr: Effective and Robust Zero-Shot Listwise Reranking," arXiv:2312.02724
- "RankLLM: A Python Package for Reranking with LLMs," SIGIR 2025, arXiv:2505.19284
- Cohere rerank-3.5 announcement (Dec 2024)
- Voyage rerank-2.5 release blog (Aug 2025)
- MixedBread mxbai-rerank-v2 blog (Q1 2025)
- ZeroEntropy "Ultimate Guide to Choosing the Best Reranking Model in 2026"
- BAAI bge-reranker-v2.5-gemma2-lightweight model card (Jul 2024)
- Qdrant DBSF documentation (1.11+, 2024)
- Sentence-Transformers v5.0 release notes (Jun 2025) - SparseEncoder API

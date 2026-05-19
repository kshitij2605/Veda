# SOTA Audit: Vector Quantization (2026-05-19)

**Scope.** This document audits Veda's quantization choices (RaBitQ, SQ8, PQ, RaBitQ-rerank) against the 2026 state of the art for approximate nearest neighbour (ANN) search. It is purely a research report; no code is changed.

---

## TL;DR Verdict

| Veda quantizer | 2026 status | One-line judgment |
|---|---|---|
| **RaBitQ (1-bit)** | **SOTA** for recall-per-bit at >=768 dimensions | Still the algorithm everyone copies; Milvus, LanceDB, Elasticsearch, Lucene, VectorChord, CockroachDB, OpenSearch all integrated it during 2024-2026. |
| **RaBitQ-rerank** (binary -> full-precision rerank) | **SOTA pattern**, but Veda's particular implementation is *behind* the multi-bit refinement now favoured industry-wide | Two-stage retrieval is the right shape, but the second stage should be Extended-RaBitQ 4-7 bit or SQ8 with FastScan, not plain float reranking, for best Pareto. |
| **SQ8 (int8 scalar)** | **Behind** | Linear scalar quantization is no longer the right "balanced" baseline. **8-bit Rotational Quantization (RQ)** -- random rotation + scalar -- consistently dominates plain SQ8 on every benchmark since late 2024. Weaviate, Qdrant (via TurboQuant), and the Pinecone serverless tier all moved away from naive SQ8. |
| **PQ (classic Jegou)** | **Legacy / behind** | Still useful as a fallback and as a building block (asymmetric distance, IVF-PQ on cuVS), but PQ alone has no theoretical bound and exhibits ~0.72 regression slope bias that distorts ranking. Most new index work in 2025-2026 either skips PQ outright or replaces it with QINCo2 / RaBitQ / TurboQuant. |

**Headline finding.** *RaBitQ remains the state-of-the-art recall-per-bit quantizer in 2026; its 2025 multi-bit successor (Extended-RaBitQ, SIGMOD '25) closes the recall gap to float at 4-7 bits without reranking. The only credible alternative families are (a) **TurboQuant** (ICLR '26, Google) -- a data-oblivious rotation+Lloyd-Max scheme that ships in Qdrant 1.18, and (b) **SAQ** (SIGMOD '26, segmented code adjustment) -- which claims up to 80x faster encoding and 2.5-5.4x lower error than Extended-RaBitQ at B=4. Anisotropic quantization (ScaNN) is no longer competitive on the modern recall-latency frontier for general vectors, and binary/1.58-bit ternary remain the right choice only when you accept aggressive rerank.*

**Top 3 gaps:**
1. Veda has 1-bit RaBitQ but **not Extended-RaBitQ multi-bit** (3/4/5/7/8/9-bit). Closing this gap would let users hit 95-99% recall without a rerank pass.
2. Veda's **SQ8** is plain linear int8. Adding a fast Walsh-Hadamard rotation in front of it (i.e., 8-bit Rotational Quantization) is a near-zero-cost upgrade with measurable accuracy gain.
3. Veda has **no GPU-native quantization path** (cuVS IVF-RaBitQ / IVF-PQ on the GPU). For deployments with H100/A100 nodes (which Veda's docker-compose embedding profile already implies), this is the single biggest throughput win available.

**Top 3 wins Veda already has:**
1. Picking RaBitQ at all -- many competing OSS DBs only added it in 2025.
2. Two-stage (RaBitQ-coarse -> full-precision rerank) pattern is conceptually right and matches what Elasticsearch, Lucene and Milvus do in production.
3. Keeping PQ as an option for legacy interop is correct -- not deleting it.

**Single most important recommendation.** Add **Extended-RaBitQ (3-9 bit)** as a first-class quantizer and make it the default rerank stage replacing full-precision floats. This is one algorithm to add, one config flag to expose, and it moves Veda from "RaBitQ-only" to "the full RaBitQ family" -- which is what the SIGMOD '25 paper and the RaBitQ-Library (Nanyang Technological University) recommend as the production configuration.

---

## 1. RaBitQ family: original, extended, IVF-, library

The original **RaBitQ** (Gao & Long, SIGMOD '24, arXiv 2405.12497) quantizes a D-dim vector to D bits via random rotation + sign extraction + two float32 correction factors (distance-to-centroid and quantized-to-original dot product). It proves a sharp error bound that *no other practical quantizer offers*, and the bound shrinks as O(1/sqrt(D)) -- which is precisely why it dominates at higher dims where SQ8 / PQ get worse.

**What changed in 2025-2026:**

- **Extended-RaBitQ** (Gao et al., SIGMOD '25, arXiv 2409.09913). Generalises 1-bit RaBitQ to 3/4/5/7/8/9 bits per dimension while keeping computation identical to classical scalar quantization. Quote from the authors: it gives "especially significant improvement in the setting from 2-bit to 6-bit, which helps an algorithm achieve high recall *without reranking*". On the official library page (vectordb-ntu.github.io/RaBitQ-Library), 4-bit hits ~90% recall, 5-bit ~95%, 7-bit ~99% -- all without rerank.
- **IVF-RaBitQ** (arXiv 2602.23999). GPU-native re-engineering of the IVF + RaBitQ pipeline; integrated into NVIDIA cuVS. Build time drops from hours to minutes, search throughput in the thousands of QPS at 90%+ recall.
- **The RaBitQ Library** (Nanyang Technological University, openreview OeZHhOsFir). Packages 1-bit + multi-bit + HNSW+RaBitQ + IVF+RaBitQ + SymphonyQG (proximity-graph variant) into one reference C++ codebase. Industry uptake list as of mid-2026: Milvus (`IVF_RABITQ`), Faiss, VSAG, VectorChord, Volcengine OpenSearch, CockroachDB, Elasticsearch, Lucene, LanceDB. This is unusual breadth for a 24-month-old algorithm.
- **Theoretical optimality.** The extended paper proves the method is *asymptotically optimal* in the Alon-Klartag (FOCS '17) sense -- the same lower bound a 2026 challenger would have to beat. This matters because it gives an information-theoretic stopping point for the recall-per-bit race.

**Practical caveat documented in the field** (DEV.to "tradeoffs of extreme vector compression"):
- Raw 1-bit RaBitQ without refinement: ~76% recall on Glove-1M, 898 QPS, 96 B/vec.
- IVF_RaBitQ + SQ8 refinement: 94.7% recall, 864 QPS, 864 B/vec (matches uncompressed within statistical noise).
- IVF_FLAT (FP32) baseline: 95.2% recall, 236 QPS, 3072 B/vec.

The takeaway: RaBitQ alone is a *coarse* stage, not a standalone retriever. Veda already understands this (it has RaBitQ-rerank). The remaining question is **what** to rerank with.

## 2. TurboQuant (ICLR '26, Google Research)

TurboQuant is the headline 2026 challenger. It is also a rotation-then-scalar method: random rotation -> induce a Beta distribution on coordinates -> apply Lloyd-Max optimal scalar quantizer per dimension -> optional 1-bit QJL residual sketch for unbiased inner-product estimation.

**Strengths:**
- **Data-oblivious.** No codebook training, no calibration set, no per-dataset state -- a huge operational win vs PQ/OPQ.
- **Sub-millisecond quantization.** Indexing time is essentially zero (0.0007-0.0021 s for the rotation+Lloyd-Max step in published benchmarks).
- **Strong recall at 3-bit** on dbpedia-text-embedding-3-large (1536 and 3072 dim), GloVe, openAI3 embeddings.
- **Already in Qdrant 1.18** (May 2026 release): selectable at 1, 1.5, 2, or 4 bits via `quantization_config`. Qdrant added length renormalization, anisotropy compensation (P-Square quantile estimation), and L2/dot support on top.

**Qdrant-measured numbers (vs scalar quantization and binary):**
- TurboQuant 4-bit: within 1-2 pp of SQ8 on most datasets, at *half* the storage (8x vs 4x compression).
- TurboQuant 2-bit: beats binary 2-bit by 9-24 pp at identical 16x compression.
- TurboQuant 1-bit: beats vanilla binary by 9-21 pp at 32x compression.

**Controversy worth knowing about.** Jianyang Gao (author of RaBitQ) wrote a public rebuttal ("TurboQuant and RaBitQ: What the Public Story Gets Wrong", March 2026) arguing that (i) both methods share the random-rotation primitive and TurboQuant's paper omits this, (ii) RaBitQ's optimality theorem in the SIGMOD '25 paper matches Alon-Klartag so the "TurboQuant is asymptotically optimal, RaBitQ is suboptimal" framing is wrong, and (iii) the original TurboQuant benchmarks ran RaBitQ single-core CPU vs TurboQuant on an A100 GPU. An arXiv pre-print (2604.19528, "Revisiting RaBitQ and TurboQuant: A Symmetric Comparison") performs an apples-to-apples re-benchmark; the headline result is that the two methods are essentially tied within experimental noise on the recall-bit Pareto frontier, with TurboQuant slightly faster to build and RaBitQ slightly better at very low bit-widths.

**Net verdict for Veda.** TurboQuant is *not* a strict win over Extended-RaBitQ. But the data-oblivious property (no codebook training) is genuinely attractive and makes it a useful additional option for streaming / online ingest scenarios. Adoption priority: lower than Extended-RaBitQ, higher than e.g. additional PQ variants.

## 3. SAQ (SIGMOD '26, segmented code adjustment)

The dark-horse 2026 entrant. SAQ (arXiv 2509.12086) does PCA + dimension segmentation + bit-budget allocation per segment + coordinate-descent code refinement.

Reported numbers (from the paper, vs Extended-RaBitQ on DEEP, GIST, MSMARCO, OpenAI-1536):
- Up to **80x faster encoding** than Extended-RaBitQ (O(D) vs O(2^B * D * log D)).
- Up to **80% reduction in quantization error**.
- At B=4: 2.5-5.4x lower error.
- Reaches 95%+ recall with B=2-3 bits where RaBitQ needs B=3-5.

Caveats: tested only with IVF, not HNSW; segments are constrained to multiples of 64; gains diminish on perfectly uniform eigenvalue distributions; no large-scale production deployment confirmed as of May 2026.

**Net verdict.** Promising but not yet a default. Watch this one; it may be the next-gen replacement if results survive independent replication.

## 4. Anisotropic Quantization (ScaNN)

Google's ScaNN (Guo et al., ICML '20, anisotropic VQ) was the previous king of MIPS recall. The idea: weight the quantization loss so that high-inner-product candidates have smaller error than low-inner-product ones. It dominated ann-benchmarks.com glove-100 for several years.

**2026 status: outclassed for general-purpose ANN.**

- On the recall-latency Pareto frontier reported by ann-benchmarks.com on modern embeddings (768d-3072d), RaBitQ-family and TurboQuant push the curve further than anisotropic-PQ.
- The asymmetry trick is *specifically* tuned for MIPS (cosine / inner-product) and is less useful for L2.
- ScaNN's anisotropic loss requires per-codebook training; Extended-RaBitQ / TurboQuant do not.
- Anisotropic Additive Quantization (AAAI '22) is the natural extension but has not seen meaningful production adoption beyond Google internal stacks.

**Net verdict for Veda.** Not worth implementing. ScaNN remains relevant as a *library*, not as a quantization algorithm to copy.

## 5. Additive Quantization (AQ) / Local Search Quantization (LSQ) / QINCo2

The "codebook stacking" family. AQ (CVPR '14) and LSQ (ECCV '16/'18) replace PQ's per-subspace codebooks with a sum of codewords from multiple full-dimensional codebooks, encoded via simulated annealing or iterated local search.

**2026 status: alive, but specialised.**

- Faiss continues to ship `LocalSearchQuantizer` and `ResidualQuantizer` and treats them as serious baselines.
- **QINCo2** (ICLR '25, Vallaeys, Muckley, Verbeek, Douze -- Meta FAIR). Neural residual quantizer where each step's codebook is the output of a small NN conditioned on the previous reconstruction. Improves SOTA reconstruction MSE by **34% for 16-byte vectors on BigANN**, search accuracy by **24% with 8-byte encodings on Deep1M**.
- These methods *win at very low byte budgets* (e.g. 8-16 bytes per vector for a 1024-dim embedding) where RaBitQ has too many bits to spare. For Veda's likely operating point (>=96 bytes/vec for 768d+ embeddings) the gap is small.

**Net verdict for Veda.** AQ/LSQ are not worth chasing; if you want neural codebooks, QINCo2 is the one to track. But for the document/text RAG workloads Veda targets, multi-bit RaBitQ at >=4 bit beats LSQ on simplicity at roughly equivalent recall.

## 6. Residual Quantization (RQ)

Two distinct things share this name in 2026, which causes confusion:

1. **Classical Residual Quantization** (Faiss `ResidualQuantizer`). Encode v, then encode v - reconstruction(v), then encode the residual of *that*, and so on. Has been in Faiss for a decade. Useful but unremarkable in 2026.
2. **"Rotational Quantization (RQ)"** as marketed by Weaviate (and embraced by the broader DB market in 2025-2026): random rotation -> int8 scalar -> store float32 (min, max, norm) metadata. *This is essentially a rebranding of SQ8 with a Walsh-Hadamard rotation in front of it.*

**Weaviate's 8-bit RQ benchmarks (their May 2025 blog "8-bit Rotational Quantization"):**
- MSMARCO 1M: 99.40% recall@10.
- DBpedia 1M: 99.02% recall@10.
- GloVe 1M: 98.82% recall@10.
- 4x compression while *improving* throughput and recall vs uncompressed.
- vs binary: on GIST, BQ gets 0.02 recall@10 (essentially useless without rerank), RQ gets 96.62%.
- Replaced RaBitQ's exponential-time encoding with cheap scalar quantization and substituted random rotations with Walsh-Hadamard, reducing rotation time from 1700 us to 7 us for 1536-dim vectors.

**Net verdict for Veda.** This is the most actionable single improvement to Veda's SQ8. Adding a Walsh-Hadamard fast rotation in front of the existing int8 quantization is a few hundred lines of code and matches or beats Weaviate's default. Worth doing even if you do not add Extended-RaBitQ.

## 7. Binary embeddings / 1-bit quantization (Cohere, Jina, mxbai)

Distinct from RaBitQ in important ways: these are *model-side* binary embeddings -- the encoder is trained to produce embeddings that round well to 1 bit -- rather than post-hoc quantization of float embeddings.

- **Cohere embed-v3 / v4** (binary, int8, ubinary native outputs). Reported retention: 90-98% of original search quality after 32x compression; int8 retention up to 100% on some MTEB tasks.
- **Sentence-Transformers / HuggingFace** (binary precision in `model.encode`). mxbai-embed-large-v1 retains ~96% of float32 retrieval performance with binary + rescore (multiplier 4), int8 retains ~99% with rescore (multiplier 4-5). 24.76x retrieval-speedup measured.
- **Jina binary embeddings**: comparable behaviour, 96-97% retention with rescore.
- **Cohere BinaryVectorDB**: open-source reference for hundred-million-scale binary ANN.

**vs RaBitQ.** Cohere/Jina/mxbai binary is *complementary*, not a replacement: their binary embeddings round well by design, but they still benefit from the bias-correction tricks RaBitQ uses (centroid offset, random rotation, two correction floats). The 2026 best-practice pipeline is: model emits binary -> RaBitQ-style centroid-relative coding -> rerank with int8 or full float.

**Net verdict for Veda.** Veda should treat "the embedding is already binary" as an input modality and *still* use RaBitQ's correction factors when storing it. That is the lowest-risk way to support binary-native embeddings without giving up the theoretical bound.

## 8. BitNet-style ternary (1.58-bit) embeddings

BitNet b1.58 (Ma et al. 2024) constrains LLM weights to {-1, 0, +1}; the 1.58 bits comes from log2(3). The 2025 paper "Ultra-Quantisation: Efficient Embedding Search via 1.58-bit Encodings" (arXiv 2506.00528) ports the idea to embeddings: store coordinates as {-1, 0, +1}.

**Reported results.** Better than binary, better than RaBitQ (in the paper's own benchmarks), on retrieval recall at similar memory. *No production vector DB has shipped this as of May 2026.* Status is "promising research".

**Caveat.** The "better than RaBitQ" claim is on specific datasets; independent replication is still pending and Gao's response to similar claims (see TurboQuant section) suggests the apples-to-apples picture may be closer than the paper suggests.

**Net verdict for Veda.** Watch list. Do not implement.

## 9. Matryoshka quantization (MatQuant + MRL)

Two orthogonal compression axes have converged in 2025-2026:

- **Matryoshka Representation Learning (MRL)** -- truncating the embedding dimension (first N dims form a valid embedding by themselves). Most SOTA embedding models in 2024-2026 are trained MRL-style (OpenAI text-embedding-3, Cohere embed-v3, Snowflake Arctic, BGE-M3, mxbai).
- **Matryoshka Quantization (MatQuant, arXiv 2502.06786, Feb 2025)**. Trains a single quantized model that can be served at int8, int4, or int2 by leveraging the nested integer structure (lower bit-widths are the MSBs of higher). Reported: extracted int2 outperforms standard int2 quantization by 4-7%.

**Combined.** MRL truncates dims; MatQuant quantizes within each dim. The two stack. A 2025 Towards Data Science piece reports up to 80% cost reduction from combining the two.

**vs RaBitQ.** MatQuant is mostly aimed at LLM weights; MRL is an embedding-model-side technique, not a quantizer. The vector DB layer's job is to (a) accept truncated MRL embeddings as first-class (Veda likely already does, since vector dim is just metadata) and (b) apply a strong quantizer (RaBitQ family) on top.

**Net verdict for Veda.** Already implicitly supported via vector dimension flexibility. No new quantizer needed. Worth documenting that "MRL + RaBitQ" is the recommended combined-compression mode.

## 10. GPU-friendly quantization: cuVS, NVIDIA NIM

- **cuVS** (NVIDIA's GPU vector search library) ships **IVF-PQ**, **IVF-Flat**, **CAGRA**, and as of 2026 **IVF-RaBitQ**. The latter is the same paper as the GPU-native IVF-RaBitQ above; it has been merged into cuVS and benchmarked there.
- **Elasticsearch + NVIDIA cuVS** (announced 2025): GPU-accelerated vector indexing using cuVS IVF-PQ and CAGRA.
- **NVIDIA NIM** (production model-serving): uses TensorRT-LLM for the model side; for the retrieval side it points to cuVS. Quantization choices are PQ for legacy compatibility and RaBitQ for new deployments.
- For GPU, **PQ is still relevant** because the IVF-PQ ADC distance table fits beautifully in shared memory and GPU PQ throughput remains formidable. So the GPU/CPU best choices diverge: CPU prefers RaBitQ-family, GPU is fine with either PQ or RaBitQ-family.

**Net verdict for Veda.** Veda's docker-compose embedding profile already implies GPU. Wiring cuVS in for the index path -- IVF-RaBitQ specifically -- would slot perfectly into the existing RaBitQ choice and inherit Veda's correctness-preserving rerank pattern. This is the single largest throughput win.

## 11. What top vector DBs default to in 2026

| Database | Default quantizer | Notable options |
|---|---|---|
| **Pinecone** | PQ (under the hood in Serverless tier; IVF+PQ for large RAG deployments). HNSW is the default *index*; PQ is the default *compression*. | int8, binary; the higher-tier "P-series" pods use proprietary quantization. |
| **Milvus / Zilliz** | Per-index choice: `IVF_PQ`, `IVF_SQ8`, `IVF_FLAT`, **`IVF_RABITQ`** (added late 2024). Recommends RaBitQ for max compression on ultra-large datasets and SQ8 for balanced workloads. | Hybrid configs combining methods. |
| **Qdrant** | Scalar quantization (int8) recommended default. **TurboQuant** in 1.18 (May 2026) as a new opt-in. Binary supported for stored vectors with int8 query asymmetry. | Product quantization available but not recommended for new deployments. |
| **Weaviate** | **8-bit Rotational Quantization (RQ)** -- their default since 2025. PQ legacy, binary supported. | They explicitly de-emphasised RaBitQ in favour of their simpler rotational SQ8 approach. |
| **Elasticsearch / Lucene** | int8 by default; **BBQ (RaBitQ binary)** integrated 2024 for opt-in 1-bit, with int8 rerank. | NVIDIA cuVS for GPU acceleration. |
| **LanceDB** | IVF_PQ default; **IVF-RaBitQ** added in 2025 as opt-in, no training needed. | Reports 96%+ recall@10 on DBpedia vs ~92% for IVF_PQ, 495 vs ~350 QPS, 32x compression. |
| **OpenSearch (Volcengine fork)** | RaBitQ integrated. | -- |
| **CockroachDB / VectorChord / VSAG** | RaBitQ integrated. | -- |
| **pgvector** | No quantization in core; uses HNSW with float32. Binary via `bit` type for users who want it. | -- |

**Pattern.** Every major OSS vector DB has either adopted RaBitQ outright (Milvus, LanceDB, Elasticsearch, Lucene, CockroachDB, VectorChord, VSAG, OpenSearch fork) or shipped a simplified rotated-scalar variant (Weaviate RQ, Qdrant TurboQuant). Plain PQ-only and plain SQ8-only databases are now in the minority.

---

## Concrete recommendations for Veda

Listed in priority order. Each item is independent and can be done in isolation.

1. **Add Extended-RaBitQ as a multi-bit quantizer** (3/4/5/7/8/9 bit). Replace Veda's RaBitQ-rerank "rerank with full float" path with "rerank with Extended-RaBitQ 4-7 bit". Same theoretical bound, ~10x less rerank cost than full float. The C++ reference implementation in `VectorDB-NTU/RaBitQ-Library` is permissively licensed and small (~3k LoC). Estimated effort: 2-3 weeks of one engineer.
2. **Add Walsh-Hadamard fast rotation in front of SQ8** to get 8-bit Rotational Quantization. Few hundred lines of code, no training, matches Weaviate's default behaviour. Estimated effort: 1 week.
3. **GPU path via cuVS IVF-RaBitQ.** Veda already requires a GPU for the multimodal embedding profile; reusing the same GPU for index search via cuVS is natural. Estimated effort: 4-6 weeks (FFI + index lifecycle wiring).
4. **Document the MRL + RaBitQ combination** as the recommended production setup. No code; just guidance.
5. **Watch list (do not implement yet).** TurboQuant, SAQ, 1.58-bit ternary, QINCo2. Reconsider in 6 months once independent benchmarks land.
6. **Leave plain PQ in place** as a legacy option for users coming from Faiss/Milvus IVF_PQ indexes -- do not delete it.

**Single-line elevator answer to "is Veda 2026 SOTA on quantization?"** *Veda is on the SOTA frontier (RaBitQ + SQ8 + PQ + rerank) but is missing the Extended-RaBitQ multi-bit ladder that lets users dial recall vs storage without a separate rerank stage. Adding that one algorithm closes the gap to the leading published configurations (Milvus, LanceDB, Elasticsearch) without changing Veda's existing architecture.*

---

## Sources

- [RaBitQ original paper (SIGMOD '24, arXiv 2405.12497)](https://arxiv.org/abs/2405.12497)
- [Extended-RaBitQ (SIGMOD '25)](https://github.com/VectorDB-NTU/Extended-RaBitQ)
- [The RaBitQ Library (NTU)](https://vectordb-ntu.github.io/RaBitQ-Library/)
- [GPU-Native IVF-RaBitQ (arXiv 2602.23999)](https://arxiv.org/pdf/2602.23999)
- [TurboQuant paper (ICLR '26, arXiv 2504.19874)](https://arxiv.org/html/2504.19874v1)
- [Google Research blog: TurboQuant](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)
- [Gao's rebuttal: "TurboQuant and RaBitQ: What the Public Story Gets Wrong"](https://dev.to/gaoj0017/turboquant-and-rabitq-what-the-public-story-gets-wrong-1i00)
- [Symmetric comparison paper (arXiv 2604.19528)](https://arxiv.org/html/2604.19528v1)
- [SAQ paper (SIGMOD '26, arXiv 2509.12086)](https://arxiv.org/html/2509.12086v1)
- [QINCo2 (ICLR '25, Meta FAIR)](https://github.com/facebookresearch/Qinco)
- [Matryoshka Quantization (MatQuant, arXiv 2502.06786)](https://arxiv.org/pdf/2502.06786)
- [Ultra-Quantisation: 1.58-bit embeddings (arXiv 2506.00528)](https://arxiv.org/pdf/2506.00528)
- [Weaviate 8-bit Rotational Quantization blog](https://weaviate.io/blog/8-bit-rotational-quantization)
- [Qdrant TurboQuant article](https://qdrant.tech/articles/turboquant-quantization/)
- [Qdrant Issue #8524 (TurboQuant integration)](https://github.com/qdrant/qdrant/issues/8524)
- [Milvus IVF_RABITQ docs](https://milvus.io/docs/ivf-rabitq.md)
- [Milvus blog: TurboQuant vs RaBitQ debate](https://milvus.io/blog/turboquant-rabitq-vector-database-cost.md)
- [LanceDB RaBitQ blog](https://www.lancedb.com/blog/feature-rabitq-quantization)
- [Elastic RaBitQ explainer 101](https://www.elastic.co/search-labs/blog/rabitq-explainer-101)
- [HuggingFace embedding quantization blog (Cohere/mxbai)](https://huggingface.co/blog/embedding-quantization)
- [Cohere int8 & binary embeddings](https://cohere.com/blog/int8-binary-embeddings)
- [NVIDIA cuVS IVF-PQ deep dive](https://developer.nvidia.com/blog/accelerating-vector-search-nvidia-cuvs-ivf-pq-deep-dive-part-1/)
- [ScaNN / anisotropic vector quantization (ICML '20)](https://arxiv.org/pdf/1908.10396)
- [Faiss library paper (arXiv 2401.08281)](https://arxiv.org/pdf/2401.08281)
- [Practical Tradeoffs of Extreme Vector Compression at Scale (DEV.to)](https://dev.to/schiffer_kate_18420bf9766/the-practical-tradeoffs-of-extreme-vector-compression-testing-rabitq-at-scale-10h8)

# State-of-the-Art Audit: Vector Indexes (2026-05-19)

**Subject:** Veda's vector-index portfolio vs. 2026 industry state of the art.
**Scope:** `crates/veda-index/` — HNSW, DiskANN, IVF-PQ, Flat, RaBitQ, SQ8, HNSW-RaBitQ.
**Method:** Primary-source review of papers (arXiv, SIGMOD/SOSP/VLDB/WWW 2023-2026), GitHub releases (Milvus, Qdrant, DiskANN, cuVS, Weaviate), and vendor engineering blogs (Pinecone, Turbopuffer, Vespa, NVIDIA, Meta).

---

## Verdict Summary

| Veda Index | 2026 SOTA Position | Verdict |
|---|---|---|
| **HNSW** | Default of Qdrant, Weaviate, Milvus, Vespa, Elasticsearch. Still strong on CPU for <100M, but now routinely augmented with ACORN-style filter traversal and RQ/RaBitQ compression. | **Same** — table-stakes. |
| **DiskANN (Vamana)** | Active development by Microsoft. Default for Azure Cosmos DB, SQL Server 2025, Bing. Has evolved into a family (Fresh/Filtered/IP-/Distributed-) that Veda does not yet track. | **Behind** — Veda has the original; the family has moved on. |
| **IVF-PQ** | Largely superseded by IVF-RaBitQ, ScaNN (anisotropic VQ + SOAR), and SPANN for disk-resident workloads. Still used internally by Pinecone, LanceDB, Vertex/AlloyDB (via ScaNN). | **Behind** — kept for compatibility; better-quality quantizers exist. |
| **Flat / brute-force** | Universal. Used by everybody for ground-truth and small collections. cuVS Flat on GPU is the new ceiling. | **Same** (CPU); **behind** (no GPU path productionised). |
| **RaBitQ** | SIGMOD 2024 paper, SIGMOD 2025 follow-ups (Extended RaBitQ, SymphonyQG). Adopted by Milvus 2.5+, LanceDB, Elasticsearch, Weaviate (as RQ). | **Ahead of average** — Veda shipped this early; needs to ride the 2025 improvements. |
| **SQ8** | Universal commodity; every major DB has it. Adequate baseline. | **Same**. |
| **HNSW-RaBitQ** | Equivalent to NGT-QG / SymphonyQG / Weaviate-HNSW-RQ. Strong choice, but the 2025 papers add FastScan and remove explicit rerank. | **Same** — needs the FastScan-aware refinement. |

### Top 3 gaps
1. **No GPU-native graph index productionised.** The `gpu_cuvs.rs` scaffolding exists but CAGRA is not a first-class index. Faiss (Meta) shipped this in v1.10 (Feb 2025), Milvus shipped it, OpenSearch 3.0 (Tech Preview) and Elasticsearch 9.3 (Tech Preview, early 2026) shipped it. CAGRA is 12.3x faster to build and 4.7x lower latency than CPU HNSW at the same recall (NVIDIA, May 2025).
2. **No filtered-search algorithm beyond naive pre/post-filtering.** ACORN (SIGMOD 2024) is now in Qdrant 1.16 (Nov 19, 2025), Weaviate 1.27, and Vespa (Sep 2025) as `filterFirstThreshold`/Acorn-1. Filtered-DiskANN (WWW 2023) is the standard for filtered ANN on disk. Veda has neither.
3. **No SPANN-class disk-resident inverted index.** DiskANN is graph-on-SSD; SPANN is centroids-in-RAM-postings-on-SSD and is >2x faster on the low-latency Pareto frontier. Pinecone, Turbopuffer, and Vespa all use SPANN-derived "centroid + posting" architectures rather than DiskANN on object storage. Veda's DiskANN is the wrong shape for cheap S3-backed serverless tiers.

### Top 3 wins
1. **RaBitQ shipped early.** Veda was an early adopter of the SIGMOD 2024 binary-quantization-with-error-bound approach. Most competitors caught up only in 2025 (Weaviate 1.32 RQ in July 2025, LanceDB feature blog Aug 2025).
2. **Seven indexes already.** Few DBs offer this many backends behind one schema — Milvus is the only obvious comparator, and Qdrant/Weaviate intentionally restrict choice.
3. **Apache-2.0 Rust + workspace decomposition.** Lance (Apache-2.0) and Qdrant (Apache-2.0) are the only competitors with a comparable license and language stance; everyone else is Go (Milvus, Weaviate, Vespa) or proprietary (Pinecone, Vectara, Turbopuffer).

### Recommendation in one paragraph
**Add three indexes, modernise one.** (1) Promote `gpu_cuvs::CagraIndex` to a first-class index variant with on-GPU build then CPU HNSW serialisation (the "build on GPU, serve on CPU" pattern Faiss adopted in v1.10). (2) Add a Filtered-DiskANN / ACORN-style traversal mode to HNSW for predicate-heavy RAG. (3) Add a SPANN-style centroid+posting index for the S3/MinIO cold tier — this is what Pinecone Serverless, Turbopuffer, and Vespa HNSW-IF use, and it is the natural fit for Veda's existing MinIO blob backend. (4) Refresh RaBitQ with the SymphonyQG/Extended-RaBitQ 2025 integration (FastScan kernels, implicit rerank).

---

## 1. NVIDIA cuVS / CAGRA — GPU-native graph ANN

**Primary source.** Ootomo et al., *"CAGRA: Highly Parallel Graph Construction and Approximate Nearest Neighbor Search for GPUs"* (arXiv:2308.15136, v2 July 9 2024). Quote: *"In graph construction time, CAGRA is 2.2-27x faster than HNSW... in large-batch query throughput in the 90-95% recall range, CAGRA is 33-77x faster than HNSW, and 3.8-8.8x faster than state-of-the-art GPU implementations. For a single query, CAGRA is 3.4-53x faster than HNSW at 95% recall."*

**Production status.**
- **Faiss v1.10.0 (Feb 2025)** — opt-in cuVS backend for Flat, IVF-Flat, IVF-PQ, and `GpuIndexCagra`. Meta and NVIDIA jointly announced this at https://engineering.fb.com/2025/05/08/data-infrastructure/accelerating-gpu-indexes-in-faiss-with-nvidia-cuvs/.
- **Milvus 2.5+** uses cuVS CAGRA via Knowhere for GPU index builds (Zilliz blog, 2025).
- **OpenSearch 3.0** ships GPU-accelerated indexing (Tech Preview) via cuVS.
- **Elasticsearch 9.3** (scheduled early 2026) ships GPU-accelerated vector indexing in Tech Preview.
- **cuVS releases:** active monthly cadence at https://github.com/rapidsai/cuvs/releases (RAPIDS calendar).

**Key 2025 trick — "build on GPU, serve on CPU":** cuVS supports serialising a CAGRA graph into a `hnswlib`-format index file. This means you can use a GPU instance only for the build step (minutes vs. hours on CPU) and serve from cheap CPU memory — exactly the indexing-vs-serving cost split RAG workloads need.

**Veda position.** `gpu_cuvs.rs` already wraps `cuvs-rs` (CAGRA build + search + `serialize_to_hnswlib`) but it is feature-gated and not exposed through the index registry. **Recommendation:** promote to a first-class variant; this is the single highest-leverage gap to close.

---

## 2. SPANN — disk-resident inverted index for billion-scale

**Primary source.** Chen et al., *"SPANN: Highly-efficient Billion-scale Approximate Nearest Neighbor Search"* (NeurIPS 2021, arXiv:2111.08566). Quote: *"SPANN follows the inverted index methodology... only the centroid points of the posting lists in memory while putting the large posting lists on disk... SPANN significantly outperforms DiskANN in both recall@1 and recall@10 especially in the low query latency budget (less than 4ms). Additionally, SPANN is more than two times faster than DiskANN."*

**2025 evolution — SPFresh:** Xu et al., *"SPFresh: Incremental In-Place Update for Billion-Scale Vector Search"* (SOSP 2023; arXiv:2410.14452 — Oct 2024 cleanup). Quote: *"With LIRE, SPFresh provides superior query latency and accuracy to solutions based on global rebuild, with only 1% of DRAM and less than 10% cores needed at the peak compared to the state-of-the-art, in a billion scale disk-based vector index with a 1% of daily vector update rate."*

**Production usage.**
- **Pinecone Serverless** (Jan 2024 GA, "evolving" 2025-2026) — "Ananas (FJLT) for small slabs, PQFS (Product Quantization Fast Scan) for medium slabs, IVF indexing for large slabs" (Pinecone blog, 2024). This is a slab-of-postings architecture inspired by SPANN, not a graph-on-SSD.
- **Turbopuffer** — "clustered indexes rather than graph-based approaches like HNSW that require many round trips. Vectors are grouped semantically into clusters with cluster centroids stored together" (Jason Liu, *"TurboPuffer: Object Storage-First"*, Sep 11 2025). Cold query 200-500ms, warm 8-10ms.
- **Vespa "HNSW-IF" (Hybrid HNSW + Inverted File)** — "Billion-scale vector search using hybrid HNSW-IF" (Vespa blog).

**Why this matters for Veda.** Veda already has MinIO/S3 as a backend, but DiskANN's design assumes local NVMe with random 4KB reads. SPANN's coarse-grained posting reads (megabyte-scale sequential) is the right primitive for object storage. **Recommendation:** add a SPANN/SPFresh-style index variant targeting the existing blob storage.

---

## 3. FreshDiskANN / Filtered-DiskANN / IP-DiskANN / DistributedANN

The DiskANN family has fractured into separate algorithms — Veda has only the 2019 base.

- **FreshDiskANN** (arXiv:2105.09613, 2021). Streaming inserts/deletes with an in-memory buffer and periodic merge. Foundation for Cosmos DB streaming.
- **Filtered-DiskANN / FilteredVamana** (WWW 2023, https://harsha-simhadri.org/pubs/Filtered-DiskANN23.pdf). Builds the graph with label-aware pruning so the search can natively follow predicate edges. Reports clicks/revenue uplift in production Bing.
- **IP-DiskANN** ("In-Place Updates of a Graph Index for Streaming ANN", Xu et al., arXiv:2502.13826, Feb 19 2025). First algorithm to avoid batch consolidation entirely.
- **DistributedANN** (Xu et al., arXiv:2509.06046, Sep 2025; ICML Vector DBs Workshop 2025). Quote: *"a distributed vector search service that enables searching over a single 50 billion vector graph index spread across over a thousand machines, offering 26ms median query latency and processing over 100,000 queries per second... has replaced conventional scale-out architectures for serving the Bing search engine. Compared to the previous production system, it delivered over 6x headroom in query throughput."*
- **microsoft/DiskANN Rust rewrite.** The repo is now 99.9% Rust (Apache-2.0). Latest release **v0.52.0 (May 12, 2026)** ships AVX-512 4-bit kernels and tightens the PQ contract. README states: *"Real-time updates (using logic from IP-DiskANN and Fresh-DiskANN) that support stable recall under long update streams — no merges, rebuilds, patches needed."*

**Veda position.** Implements original DiskANN only. **Recommendation:** when refactoring DiskANN, adopt the Rust upstream as a dependency rather than maintaining a parallel implementation — the codebase is now permissively licensed and Rust-native. At minimum, add filtered-Vamana edges for predicate-aware search.

---

## 4. ScaNN — anisotropic quantization

**Primary source.** Guo et al., *"Accelerating Large-Scale Inference with Anisotropic Vector Quantization"* (ICML 2020). Follow-up: *"SOAR: New algorithms for even faster vector search with ScaNN"* (Google Research blog, 2024).

**Production.** Powers **Vertex AI Vector Search** and is now built into **AlloyDB** as the `scann` Postgres extension (Google Cloud blog, Aug 22 2025: "AlloyDB's ScaNN Vector Index Unifies Your Data & AI"). ScaNN is consistently near the Pareto front on `ann-benchmarks.com`.

**Key insight.** Anisotropic loss preserves dot-product ranking for *likely* top-k pairs at the cost of arbitrary pairs — the opposite of PQ's uniform-error assumption. This is provably better for retrieval whenever the loss function matches the query distribution.

**Veda position.** No anisotropic-aware quantizer. RaBitQ partially addresses this (with theoretical error bound on inner-product), but ScaNN's tree-AH-reorder pipeline is a separate design point.

---

## 5. ACORN — predicate-aware graph search

**Primary source.** Patel et al., *"ACORN: Performant and Predicate-Agnostic Search Over Vector Embeddings and Structured Data"* (SIGMOD 2024, arXiv:2403.04871). Quote: *"ACORN achieves state-of-the-art performance on all datasets, outperforming prior methods with 2-1,000x higher throughput at a fixed recall."*

**Production adoption (1.5 years from paper to ubiquity):**
- **Qdrant 1.16** (Nov 19, 2025) — per-query `acorn` parameter (https://qdrant.tech/blog/qdrant-1.16.x/).
- **Weaviate 1.27** (late 2025) — `filterStrategy` HNSW option.
- **Vespa** (Sep 2025) — `filterFirstThreshold` / `filterFirstExploration` / `explorationSlack` parameters, branded "Acorn 1" (https://blog.vespa.ai/additions-to-hnsw/).

**Veda position.** `crates/veda-query/` has a filter parser, but the index does not see predicates. **Recommendation:** add ACORN-style "neighbors of neighbors" traversal as an opt-in to the HNSW searcher. Implementation is small (~200 LOC) and the upside on RAG workloads with metadata filtering is large.

---

## 6. Vamana family & ParlayANN

**Primary source.** Manohar et al., *"ParlayANN: Scalable and Deterministic Parallel Graph-Based ANN Search Algorithms"* (PPoPP 2024). The ParlayANN paper is the canonical comparative-analysis benchmark for graph indexes — it reimplements DiskANN, HNSW, HCNNG, and pyNNDescent under one harness with the same search routine.

**Takeaway for tuning.** ParlayANN's Vamana is described as "the fastest and highest quality implementation of a shared-memory graph-based ANNS algorithm available today." If Veda's HNSW or DiskANN benchmarks worse than the same dataset on ParlayANN, the issue is implementation, not algorithm. **Recommendation:** add ParlayANN as a benchmark reference.

**Other 2024-2025 Vamana descendants worth tracking:**
- **B+ANN** (arXiv:2511.15557, late 2025) — fast billion-scale disk index, page-aligned.
- **PiPNN** (arXiv:2602.21247) — ultra-scalable graph-based NN indexing.

---

## 7. Hybrid scalar+vector indexes (filtered search SOTA)

The 2026 consensus pattern, from Qdrant/Weaviate/Vespa/Milvus:

1. **Inline scalar columns alongside vector storage** — Qdrant 1.16's "Inline Storage" mode stores vector data directly inside HNSW nodes (https://qdrant.tech/blog/qdrant-1.16.x/). Weaviate's HNSW links can be skipped per payload-indexed field (Qdrant 1.17, PR #7887).
2. **Predicate-aware graph traversal** — ACORN (above).
3. **Pre-filter when selectivity is high, post-filter when low** — Vespa's `filterFirstThreshold` makes this a tunable threshold.
4. **Label-aware index construction** — Filtered-DiskANN's FilteredVamana actually builds different neighbour sets per label.

**Veda position.** No native interleaving of payload filters with graph traversal. The query layer currently appears to do post-filtering only.

---

## 8. RaBitQ & SymphonyQG — what's new in 2025

Veda is ahead of most DBs by *having* RaBitQ. But the 2025 follow-ups change the picture:

- **Extended RaBitQ** (SIGMOD 2025, https://github.com/VectorDB-NTU/Extended-RaBitQ) — generalises RaBitQ from 1-bit to b-bit codes with optimal scalar quantization, closing the gap with PQ at higher bit budgets while keeping the theoretical guarantee.
- **SymphonyQG** (SIGMOD 2025, https://vectordb-ntu.github.io/news/symqg/) — *"avoids the explicit re-ranking step and refining the graph structure to be more aligned with FastScan"*. Implements graph+RaBitQ with shorter codes than NGT-QG.
- **Weaviate 1.32 RQ** (July 2025) — rotational quantization, 1-bit RQ in 1.33 (Oct 2025) as preview.
- **LanceDB RaBitQ** — feature blog (Aug 2025), shipped in `lance` columnar format.

**Veda position.** Has the original 2024 RaBitQ. **Recommendation:** track Extended-RaBitQ and SymphonyQG patches in 2026.

---

## 9. What top vector DBs actually ship in 2026

| DB | Default index (2026) | Filtered search | Notable |
|---|---|---|---|
| **Pinecone Serverless** | Proprietary (Ananas/PQFS/IVF slabs) | Built-in | Read units pricing (Apr 2026), Dedicated Read Nodes (Dec 2025). |
| **Milvus 2.6.16** (May 14 2026) | HNSW, IVF, DISKANN, AUTOINDEX. CAGRA via Knowhere/cuVS. Milvus 3.0-beta (May 9 2026) adds **EmbList + DISKANN** for multi-vector / late-interaction (ColBERT-style). | Scalar + vector co-indexes | Has External Collection (zero-copy Iceberg/Lance/Parquet), Storage V3 (manifest-based S3 columnar). |
| **Qdrant 1.18.0** (May 11 2026) | HNSW + RocksDB *removed*. Adds **TurboQuant** (8x compression, 1.18), ACORN (1.16). | ACORN per-query | GPU HNSW builds, Inline Storage, Qdrant Edge. |
| **Weaviate 1.34/1.35** (late 2025/early 2026) | HNSW, Flat. RQ (rotational quantization) as default compression in 1.32-1.34. ACORN filter strategy (1.27+). | ACORN | Async indexing with on-disk queue. |
| **Vespa** (Sep/Dec 2025 newsletters) | HNSW, HNSW-IF (hybrid). | "Acorn 1" + `filterFirstThreshold` (Sep 2025); Google Highway SIMD (Dec 2025). | Billion-scale references in production. |
| **Vectara** | Proprietary (managed). | Built-in. | Closed source. |
| **LanceDB** | IVF-PQ on Lance columnar format; RaBitQ (Aug 2025); HNSW for in-memory. | Pushdown SQL via DuckDB (2026). | Disk-first. |
| **Turbopuffer** | Centroid + posting on object storage (no HNSW). | Built-in. | "100x cheaper" cold storage. 200-500ms cold, 8-10ms warm. |
| **Elasticsearch 9.3** (early 2026, Tech Preview) | HNSW + cuVS GPU build. | Mature, via Lucene. | RaBitQ binary quantization (Elastic Search Labs). |
| **OpenSearch 3.0** | HNSW + cuVS GPU build (Tech Preview). | Built-in. | |
| **SQL Server 2025** | DiskANN. | Yes, T-SQL filters. | Microsoft's all-in DiskANN bet. |
| **Azure Cosmos DB** | DiskANN with streaming/IP updates (Apr 2026 blog post). | Yes. | Stable recall under long update streams. |
| **AlloyDB / Vertex AI** | ScaNN (with SOAR). | Yes (PostgreSQL predicates). | Aug 22 2025 GA. |

**Pattern.** The 2026 mainstream is *HNSW + RaBitQ/RQ for compression + ACORN for filters + cuVS for GPU build*. Veda has the first two pieces and the scaffolding for the fourth; ACORN is the missing first-tier piece.

---

## Recommendations (priority order)

1. **(High)** Promote `gpu_cuvs::CagraIndex` to a first-class index variant. Wire it into the index registry, with "build on GPU, serve on CPU" via `serialize_to_hnswlib`. This single change closes the largest gap.
2. **(High)** Add ACORN-style traversal as a per-query HNSW option (parity with Qdrant 1.16, Weaviate 1.27, Vespa). Small implementation, large RAG-quality upside.
3. **(Medium)** Add a SPANN/SPFresh-style centroid+posting index for the MinIO/S3 tier — replaces DiskANN as the natural cheap-storage index. Pinecone, Turbopuffer, Vespa-HNSW-IF all use this shape; DiskANN-on-S3 is the wrong access pattern.
4. **(Medium)** Replace the in-tree DiskANN with the `microsoft/DiskANN` Rust crate (Apache-2.0, v0.52.0 as of May 12 2026). Avoids tracking IP-DiskANN/Fresh-DiskANN ourselves.
5. **(Medium)** Add Filtered-DiskANN edges for predicate-aware search on the disk path.
6. **(Low)** Upgrade RaBitQ to the SymphonyQG / Extended-RaBitQ 2025 integration (FastScan kernels, implicit rerank).
7. **(Low)** Adopt ParlayANN as a reference benchmark in the index test suite.

---

## Primary-source bibliography

- CAGRA: arXiv:2308.15136 (v2 Jul 9 2024).
- Filtered-DiskANN: WWW 2023, https://harsha-simhadri.org/pubs/Filtered-DiskANN23.pdf.
- FreshDiskANN: arXiv:2105.09613 (2021).
- IP-DiskANN: arXiv:2502.13826 (Feb 19 2025).
- DistributedANN: arXiv:2509.06046 (Sep 2025; ICML VDB Workshop 2025).
- ACORN: SIGMOD 2024, arXiv:2403.04871.
- SPANN: NeurIPS 2021, arXiv:2111.08566.
- SPFresh: SOSP 2023, arXiv:2410.14452.
- RaBitQ: SIGMOD 2024, arXiv:2405.12497.
- SymphonyQG: SIGMOD 2025, https://vectordb-ntu.github.io/news/symqg/.
- ScaNN/SOAR: ICML 2020 + Google Research blog (2024).
- ParlayANN: PPoPP 2024, https://www.cs.ucr.edu/~yihans/papers/2024/PPoPP24/parlayann.pdf.
- Faiss + cuVS: Meta Engineering blog (May 8 2025), https://engineering.fb.com/2025/05/08/data-infrastructure/accelerating-gpu-indexes-in-faiss-with-nvidia-cuvs/.
- Qdrant 1.16 ACORN: https://qdrant.tech/blog/qdrant-1.16.x/ (Nov 19 2025).
- Qdrant 1.18 TurboQuant: https://github.com/qdrant/qdrant/releases/tag/v1.18.0 (May 11 2026).
- Milvus 2.6.16: https://github.com/milvus-io/milvus/releases/tag/v2.6.16 (May 14 2026).
- Milvus 3.0-beta EmbList+DISKANN: https://github.com/milvus-io/milvus/releases/tag/v3.0-beta (May 9 2026).
- Weaviate 1.32/1.33: https://weaviate.io/blog/weaviate-1-32-release, https://weaviate.io/blog/weaviate-1-33-release.
- Vespa Acorn-1: https://blog.vespa.ai/additions-to-hnsw/ (Sep 2025).
- Turbopuffer architecture: https://jxnl.co/writing/2025/09/11/turbopuffer-object-storage-first-vector-database-architecture/ (Sep 11 2025).
- microsoft/DiskANN v0.52.0 (Rust): https://github.com/microsoft/DiskANN/releases/tag/v0.52.0 (May 12 2026).

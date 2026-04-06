# ANN Algorithm Research for Multimodal Vector Database

> Research compiled April 2026. Covers algorithms, benchmarks, latest research, and recommendations for a vector DB handling text, image, audio, and video embeddings (128-2048+ dimensions).

---

## Table of Contents

1. [Current ANN Algorithms -- Deep Dive](#1-current-ann-algorithms--deep-dive)
2. [Latest Research (2024-2026)](#2-latest-research-2024-2026)
3. [Benchmark Comparisons](#3-benchmark-comparisons)
4. [Multimodal-Specific Considerations](#4-multimodal-specific-considerations)
5. [Hybrid Approaches](#5-hybrid-approaches)
6. [What Existing Vector DBs Chose and Why](#6-what-existing-vector-dbs-chose-and-why)
7. [Recommendation](#7-recommendation)

---

## 1. Current ANN Algorithms -- Deep Dive

### 1.1 HNSW (Hierarchical Navigable Small World)

**Used by:** Elasticsearch, Qdrant, Weaviate, pgvector, Milvus, Chroma, Redis

**How it works:**
HNSW combines the "small world" property of certain graphs (where any node can reach any other in surprisingly few hops) with a hierarchical layering scheme inspired by skip lists. It builds multiple layers of graphs: Layer 0 contains all data points, each subsequent layer contains a subset. Higher layers act as "expressways" for long-range jumps; lower layers enable fine-grained exploration. Search starts at the top layer and descends, greedily following the nearest neighbor at each step.

**Complexity:**
- **Search:** O(log N) -- logarithmic scaling with dataset size
- **Build:** O(N * log N) -- each insertion requires a graph traversal
- **Space:** O(N * M) where M is the max connections per node (typically 16-48)

**Key Parameters:**
| Parameter | Range | Effect |
|---|---|---|
| `M` (max connections) | 5-48 | Higher = better recall, more memory. Low-dim: 5-12, High-dim: 16-48 |
| `efConstruction` | 100-500 | Higher = better graph quality, slower build |
| `ef` / `efSearch` | 50-500 | Higher = better recall, slower query. Scales linearly with latency |

**Performance:**
- ~95% recall@10 on SIFT1M in 1-2 ms per query on CPU
- Well-constructed graphs achieve >95% recall even with moderate `ef` settings
- Recall can degrade at scale: 99% recall at 10K vectors can drop to 85% at 10M with same config
- Insertion order can cause up to 17% relative change in recall

**Memory footprint (1B vectors, 128 dims, M=16):** ~1,408 GB with one replica

**Strengths:**
- Best general-purpose recall/speed tradeoff
- Supports incremental insertions
- Tunable speed-recall via `ef` at query time
- Dominant algorithm in production vector databases

**Limitations:**
- High memory overhead (graph structure)
- Recall degrades at scale without tuning adjustments
- Frequent inserts/deletes degrade graph quality ("unreachable points" problem)
- Build time is slower than IVF-based approaches
- No native filtered search support (requires modifications like filterable HNSW)

**References:**
- Original paper: Malkov & Yashunin, "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs" (2016) -- [arXiv](https://arxiv.org/pdf/1603.09320)
- [Pinecone HNSW Guide](https://www.pinecone.io/learn/series/faiss/hnsw/)
- [Scaling HNSW Recall in RAG Systems](https://explore.n1n.ai/blog/scaling-hnsw-vector-search-recall-rag-2026-01-10)

---

### 1.2 IVF (Inverted File Index)

**Used by:** FAISS, Milvus

**How it works:**
IVF partitions the vector space into Voronoi cells using k-means clustering. At query time, only the nearest `nprobe` cells are searched, dramatically reducing the search scope. Vectors are assigned to clusters based on their nearest centroid.

**Complexity:**
- **Build:** O(N * k * iterations) for k-means (fast relative to graph methods)
- **Search:** O(nprobe * N/k) -- linear within each probed cell
- **Space:** O(N * d) + O(k * d) for centroids

**Key Parameters:**
| Parameter | Typical | Effect |
|---|---|---|
| `nlist` (num clusters) | sqrt(N) to 4*sqrt(N) | More clusters = finer partitioning |
| `nprobe` (clusters searched) | 1-nlist | Higher = better recall, slower query |

**Strengths:**
- Fast build time (k-means is parallelizable)
- Lower memory overhead than graph methods
- Handles filtered search efficiently (cluster-level filtering first, then fine-grained)
- Predictable performance characteristics

**Limitations:**
- Recall depends heavily on `nprobe` tuning
- Cluster boundary effects reduce recall for boundary vectors
- Less effective than HNSW at low-latency, high-recall operating points
- Requires rebuild when data distribution shifts significantly

---

### 1.3 IVF-PQ (IVF + Product Quantization)

**Used by:** FAISS, Milvus, LanceDB

**How it works:**
Combines IVF partitioning with Product Quantization compression. PQ divides each vector into `m` sub-vectors and quantizes each independently using a codebook (typically 256 entries = 8 bits). Distance calculations operate on compressed codes, enabling massive memory savings. Residuals (vector minus centroid) are PQ-encoded for better accuracy.

**Complexity:**
- **Search:** ~92x faster than flat search
- **Space:** ~96% reduction vs. flat index. 1M vectors at 256 dims with PQ: ~0.171 GB

**Key Parameters:**
| Parameter | Typical | Effect |
|---|---|---|
| `m` (num sub-vectors) | dimension/8 | More = better recall, more memory |
| `code_size` (bits per sub-vector) | 8 | 4-bit PQ enables FastScan with SIMD |
| `nprobe` | tunable | Same as IVF |

**Performance:**
- IVF-PQ: ~50% recall (tunable to higher via nprobe)
- With re-ranking (IVF-PQ + refine): competitive with HNSW
- FastScan (4-bit PQ): SIMD-optimized, even faster on modern CPUs
- GPU (cuVS IVF-PQ): 3-4x better than IVF-Flat on large batches

**Strengths:**
- Extreme memory efficiency
- Suitable for billion-scale on commodity hardware
- Well-suited for GPU acceleration
- Can be combined with re-ranking for high recall

**Limitations:**
- Base recall lower than HNSW without re-ranking
- PQ encoding introduces quantization error
- Not ideal for very low-dimensional vectors (PQ overhead not worth it below ~64 dims)
- Additive quantizers (AQ) are more accurate for small code sizes

**References:**
- Jegou et al., "Product quantization for nearest neighbor search" (PAMI 2011)
- [FAISS Guidelines](https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index)
- [NVIDIA cuVS IVF-PQ Deep Dive](https://developer.nvidia.com/blog/accelerating-vector-search-nvidia-cuvs-ivf-pq-deep-dive-part-1/)

---

### 1.4 ScaNN (Scalable Nearest Neighbors)

**Used by:** Google (internal), AlloyDB, Vertex AI

**How it works:**
ScaNN uses anisotropic vector quantization for Maximum Inner Product Search (MIPS). It combines tree-based partitioning with learned quantization, where the quantization objective accounts for the directional importance of errors (anisotropic) rather than treating all dimensions equally. The SOAR enhancement adds redundant routing to improve recall when clustering fails.

**Complexity:**
- ~2x more QPS than next-fastest library on glove-100-angular benchmark
- SOAR: best query speed/indexing speed tradeoff among all benchmarked libraries
- Smallest memory footprint among top performers

**Performance (with SOAR, NeurIPS 2023):**
- Highest result in Big-ANN 2023 out-of-distribution and streaming tracks
- Only libraries near ScaNN's querying speed require >10x memory and >50x indexing time
- AlloyDB ScaNN: 4x faster vector queries, 8x faster index builds, 3-4x smaller memory than HNSW in PostgreSQL

**Strengths:**
- Best overall tradeoff (speed, memory, build time) with SOAR
- Optimized for x86 with AVX
- Strong theoretical foundation (anisotropic quantization)
- Production-proven at Google scale

**Limitations:**
- Primarily optimized for inner product / cosine similarity
- Less community adoption than HNSW/FAISS
- Google-centric ecosystem
- Not as widely supported across vector databases

**References:**
- Guo et al., "Accelerating Large-Scale Inference with Anisotropic Vector Quantization" (ICML 2020)
- [SOAR: New algorithms for even faster vector search](https://research.google/blog/soar-new-algorithms-for-even-faster-vector-search-with-scann/)
- [ScaNN on ANN-Benchmarks](https://ann-benchmarks.com/scann.html)

---

### 1.5 DiskANN

**Used by:** Microsoft (Bing), Milvus, SQL Server 2025

**How it works:**
DiskANN uses the Vamana graph algorithm with product quantization for SSD-based billion-scale search. PQ-compressed vectors are cached in RAM for approximate distance computation; full-precision vectors and the graph index reside on SSD. Typically requires only 2-3 disk reads per query. Construction uses a divide-and-conquer approach: data is clustered, per-cluster Vamana graphs are built, then merged.

**Performance:**
- 1B vectors with 95% recall@1, latency <5 ms, on a single node with 64 GB RAM
- RAM-based HNSW: max ~200M vectors, 500 GB memory, ~2 ms p95 latency, ~$2,000/month
- DiskANN: 1B+ vectors, 32 GB RAM + SSD, ~8 ms p95 latency, ~$400/month
- Up to 90% RAM cost reduction vs. in-memory approaches

**Complexity:**
- **Build:** Moderate-slow (two-pass Vamana graph + PQ encoding)
- **Search:** O(log N) with 2-3 SSD reads per query
- **Space:** PQ codes in RAM (~30-60 bytes/vector) + full vectors on SSD

**Strengths:**
- Billion-scale on commodity hardware (single node)
- 90% cost reduction vs. RAM-based HNSW
- Strong recall (>95%) with low latency
- Integrated into SQL Server 2025, Milvus

**Limitations:**
- Latency higher than pure in-memory approaches (~5-8 ms vs. ~1-2 ms)
- SSD I/O can be a bottleneck; requires NVMe for best performance
- Dynamic updates (FreshDiskANN) still being evaluated
- Build time is significant for very large datasets

**References:**
- Jayaram Subramanya et al., "DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node" (NeurIPS 2019) -- [PDF](https://suhasjs.github.io/files/diskann_neurips19.pdf)
- [DiskANN Explained (Milvus)](https://milvus.io/blog/diskann-explained.md)
- [SQL Server 2025 DiskANN Guide](https://www.mytechmantra.com/sql-server/sql-server-2025-diskann-vector-indexing-guide/)

---

### 1.6 SPANN

**Used by:** Microsoft (SPTAG), PlanetScale, Chroma (distributed mode)

**How it works:**
SPANN is a memory-disk hybrid following the inverted index methodology. Centroid points are stored in memory; large posting lists are stored on SSD. Uses hierarchical balanced clustering for index building, closure-based posting list expansion for boundary vectors, and query-aware dynamic pruning to minimize disk reads.

**Performance:**
- 2x faster than DiskANN to reach 90% recall with same memory cost
- 90% recall@1 and recall@10 in ~1 ms with ~10% of original memory cost
- Evaluated on SIFT1B, SPACEV1B, DEEP1B

**Strengths:**
- Superior speed/recall tradeoff vs. DiskANN in benchmarks
- Very low memory footprint (centroid-only in RAM)
- Full-precision distance calculations (no quantization loss during search)
- Practical billion-scale on commodity hardware

**Limitations:**
- Less widely adopted than DiskANN or HNSW
- Posting list management adds complexity
- Update handling less mature than graph-based approaches

**References:**
- Chen et al., "SPANN: Highly-efficient Billion-scale Approximate Nearest Neighbor Search" (NeurIPS 2021) -- [Paper](https://www.microsoft.com/en-us/research/wp-content/uploads/2021/11/SPANN_finalversion1.pdf)
- [Code: Microsoft SPTAG](https://github.com/microsoft/SPTAG)

---

### 1.7 Vamana

**Used by:** DiskANN (core graph algorithm), NVIDIA cuVS (GPU variant)

**How it works:**
Vamana is the graph construction algorithm underlying DiskANN. It initializes a random directed graph with R out-neighbors per node, calculates the medoid (approximate centroid that is a dataset member) as the navigation starting point, then iteratively refines edges using GreedySearch and RobustPrune routines. Makes two passes over the dataset (vs. one for HNSW/NSG), which improves graph quality. Produces a flat (single-layer) graph optimized for minimal disk access.

**Key difference from HNSW:** Single-layer graph (vs. hierarchical), smaller search radius, optimized for disk I/O patterns.

**References:**
- [DiskANN and the Vamana Algorithm (Zilliz)](https://zilliz.com/learn/DiskANN-and-the-Vamana-Algorithm)

---

### 1.8 NSG (Navigating Spreading-out Graph)

**Used by:** Alibaba (Taobao search engine)

**How it works:**
NSG approximates a Monotonic Relative Neighborhood Graph (MRNG) which guarantees near-logarithmic search complexity. Uses a fixed "Navigating Node" (approximate medoid) as the search starting point. Focuses on four principles: ensuring graph connectivity, lowering average out-degree, shortening search paths, and reducing index size.

**Performance:**
- Smallest index size among graph-based algorithms
- Best search performance in original benchmarks
- Integrated into Alibaba's billion-scale product search

**Strengths:**
- Compact index (smallest among graph-based methods)
- Strong theoretical guarantees (MRNG approximation)
- Fast search (near-logarithmic complexity)

**Limitations:**
- Requires a pre-built k-NN graph (adds build time)
- Less widely adopted than HNSW
- Single-layer graph (no hierarchical speedup for initial search)

**References:**
- Fu et al., "Fast Approximate Nearest Neighbor Search With The Navigating Spreading-out Graphs" (PVLDB 2019)

---

### 1.9 ANNOY (Approximate Nearest Neighbors Oh Yeah)

**Used by:** Formerly Spotify (now replaced by Voyager/HNSW)

**How it works:**
Builds a forest of random projection trees. Each tree recursively splits the space by picking two random points and creating a hyperplane between them. At query time, multiple trees are traversed in parallel and results are combined.

**Complexity:**
- **Search:** O(log N) per tree, with multiple trees
- **Space:** O(N * num_trees)
- **Build:** O(N * num_trees * log N)

**Performance:**
- Spotify's own testing: Voyager (HNSW-based) is 10x faster at same recall, or 50% more accurate at same speed
- Now "solidly in the middle of the pack" on ANN-Benchmarks

**Strengths:**
- Memory-mapped file support (share indexes across processes)
- Simple, well-understood algorithm
- Good for static, read-only workloads

**Limitations:**
- No incremental updates (must rebuild to add points)
- No GPU support or batch processing
- Significantly outperformed by HNSW on all metrics
- High dimensionality degrades performance
- **Spotify itself replaced ANNOY with Voyager (HNSW-based)**

**References:**
- [GitHub: spotify/annoy](https://github.com/spotify/annoy)
- [Spotify: Introducing Voyager](https://engineering.atspotify.com/2023/10/introducing-voyager-spotifys-new-nearest-neighbor-search-library)

---

### 1.10 LSH (Locality-Sensitive Hashing)

**How it works:**
Maps similar vectors to the same hash buckets with high probability using locality-sensitive hash functions. Multiple hash tables are used to improve recall. For cosine similarity, random hyperplane projections are common; for Euclidean distance, random projections with quantized buckets.

**Complexity:**
- **Search:** Sub-linear O(N^rho) where rho < 1
- **Build:** O(N * L) for L hash tables
- **Space:** O(N * L * K) for L tables with K hash functions

**Strengths:**
- Theoretical guarantees on query accuracy
- Fast, near-constant-time insertions (easy dynamic updates)
- Simple and interpretable
- Good for custom similarity metrics

**Limitations:**
- Requires careful tuning (number of tables, hash function parameters)
- Generally lower recall than graph-based methods out-of-the-box
- Can require high memory for multiple hash tables
- No longer the default choice in production vector databases

**2025 Status:**
Modern LSH variants (like DET-LSH) have closed the gap with HNSW on some benchmarks, especially for workloads requiring frequent updates. Hybrid approaches combining LSH for candidate generation with re-ranking are common. However, graph-based methods remain the default in most production systems.

**References:**
- [Pinecone LSH Guide](https://www.pinecone.io/learn/series/faiss/locality-sensitive-hashing/)

---

### 1.11 Flat / Brute-Force

**How it works:**
Computes exact distances between the query and every vector in the dataset. No index structure needed.

**Complexity:**
- **Search:** O(N * d)
- **Build:** O(1)
- **Space:** O(N * d)

**Use cases:**
- Ground truth computation for benchmarking
- Small datasets (<10K vectors)
- When 100% recall is required (e.g., GPU_BRUTE_FORCE in Milvus)
- Validation and testing

---

## 2. Latest Research (2024-2026)

### 2.1 RaBitQ -- Breakthrough Quantization (SIGMOD 2024 & 2025)

**What:** Randomized Binary Quantization that compresses D-dimensional vectors into D-bit strings with theoretical error bounds.

**Key results:**
- 3x faster than Product Quantization at same accuracy
- First practically deployable algorithm with asymptotically optimal space-accuracy tradeoff
- 1024-dim float32 vector (4 KB) compressed to ~136 bytes with corrective factors

**Extended RaBitQ (SIGMOD 2025):** Supports arbitrary compression rates (2-bit to 6-bit per dimension), with computation identical to classical scalar quantization but dominant accuracy.

**SymphonyQG:** Combines RaBitQ with graph-based indices, achieving state-of-the-art query performance.

**Adoption:**
- LanceDB: Available as `IVF_RQ` index type
- Elastic: Integrated as "BBQ" (Binary Quantization) feature
- TensorChord: Used in cost-effective vector search systems
- Nuclia: Reports up to 75% latency reduction

**References:**
- Gao & Long, "RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound" (SIGMOD 2024) -- [GitHub](https://github.com/gaoj0017/RaBitQ)
- [Extended RaBitQ (SIGMOD 2025)](https://github.com/VectorDB-NTU/Extended-RaBitQ)
- [LanceDB RaBitQ](https://www.lancedb.com/blog/feature-rabitq-quantization)

---

### 2.2 GPU Acceleration: NVIDIA cuVS & CAGRA

**CAGRA (CUDA ANN Graph):** GPU-native graph-based ANN algorithm built from the ground up for GPU acceleration.

**Benchmark results vs. CPU HNSW:**
| Metric | Improvement |
|---|---|
| Build time | 12.3x faster |
| Search latency | 4.7x lower |
| Offline throughput (image) | 18x higher |
| Offline throughput (text) | 8x higher |
| Online search (dynamic batching) | 10x latency improvement |

**Key capability:** CAGRA graph can be converted to HNSW graph (build on GPU, search on CPU) and to DiskANN/Vamana format.

**Integrations (2025-2026):**
| Platform | Status | Key Result |
|---|---|---|
| FAISS (v1.10.0+) | GA | 12x CPU build acceleration, 8x GPU index speedup |
| Elasticsearch | Tech Preview (9.3, early 2026) | 12x indexing throughput, 7x force-merge improvement |
| OpenSearch | Preview (3.0) | 9.3x indexing speed, 3.75x cost reduction |
| Weaviate | Integrated | 8x index build time reduction |
| Milvus (DDN) | Production | 22x faster index build |
| Apache Lucene | In progress (SearchScale) | 40x GPU build acceleration |

**cuVS APIs:** C, C++, Python, Rust, Go, Java

**AMD response:** hipVS (HIP port of cuVS) for AMD Instinct GPUs, API-compatible with cuVS.

**References:**
- [NVIDIA cuVS Technical Blog](https://developer.nvidia.com/blog/optimizing-vector-search-for-indexing-and-real-time-retrieval-with-nvidia-cuvs/)
- [Meta + cuVS in FAISS](https://engineering.fb.com/2025/05/08/data-infrastructure/accelerating-gpu-indexes-in-faiss-with-nvidia-cuvs/)
- [Elasticsearch GPU Acceleration](https://www.elastic.co/search-labs/blog/elasticsearch-gpu-accelerated-vector-indexing-nvidia)

---

### 2.3 Streaming/Dynamic HNSW Improvements

**The Problem:** Standard HNSW only supports insert + query; deletions create "unreachable points" that degrade recall over time. Industry standard is tombstone marking + periodic full rebuild.

**MN-RU Algorithm (2024):** Mutual Neighbor Replaced Update -- repairs only mutual neighbors of deleted points, suppressing recall degradation. Uses dual-index architecture for streaming workloads. Available at [GitHub](https://github.com/xwt1/ICPADS-MN-RU.git).

**Random Walk Deletion (2024-2025):** Theoretical random-walk framework for graph-based ANN. Produces a deterministic deletion algorithm that preserves hitting time statistics. Better tradeoff between query latency, recall, deletion time, and memory usage.

**hnswlib updates (2024):** Support for replacing deleted elements with newly inserted ones; fixed data races/deadlocks in multithreaded updates.

---

### 2.4 Filtered Vector Search Advances

**ACORN (Weaviate):** Maintains unpruned graph edges, uses two-hop jumps to skip filtered-out nodes. Significantly faster for restrictive, low-correlation filters on large datasets. Integrated as ACORN-1 in Lucene/Elasticsearch (up to 5x faster filtered search).

**Filterable HNSW (Qdrant):** Adds intra-category links; planner estimates filter cardinality to select strategy. Engines with in-algorithm filtering actually get faster under restrictive filters.

**Filter-Centric Vector Indexing (FCVI, 2025):** Geometric transformations encode filters into vectors. FCVI-HNSW variant shows best overall performance.

**Key finding:** IVF-based indexes outperform HNSW for low-selectivity (tight) filtered queries due to efficient cluster-level filtering.

---

### 2.5 New Algorithms and Research Directions (2024-2026)

| Paper/System | Venue | Key Contribution |
|---|---|---|
| **MARGO** | VLDB 2025 | Monotonic path-aware graph layout for disk-based ANN. Up to 26.6% efficiency improvement over Starling, 5.5x faster layout optimization |
| **ANSMET** | ISCA 2025 | First DIMM-based near-data processing for ANNS. Hybrid partial-dimension/bit early termination |
| **TANNS** | ICDE 2025 | Timestamp-aware ANN search incorporating time attributes |
| **VIBE** | 2025 | New benchmark for modern embeddings (OOD-aware), replacing outdated SIFT/GloVe datasets |
| **ParaGraph** | SIGMOD 2025 | GPU-CPU parallel graph indexing for cross-modal ANN |
| **SymphonyQG** | 2025 | Integration of RaBitQ with graph indices (state-of-the-art query performance) |
| **Filtered Vector Search** | VLDB 2025 | Microsoft Research: filters + diversity for RAG pipelines |
| **pgvectorscale** | 2025 | PostgreSQL: 471 QPS vs. Qdrant's 41 QPS at 99% recall on 50M vectors |

**Field direction (2025-2026):** The field is converging on:
1. **Graph-based indexes** (HNSW/Vamana) as the primary search structure
2. **Advanced quantization** (RaBitQ, SOAR) replacing PQ for compression
3. **GPU acceleration** (cuVS/CAGRA) for build + search
4. **Disk-aware designs** (DiskANN, MARGO, SPANN) for cost efficiency
5. **Integrated filtering** as a first-class capability
6. **Streaming/dynamic** workload support

---

## 3. Benchmark Comparisons

### 3.1 ANN-Benchmarks (ann-benchmarks.com)

**Setup:** r6i.16xlarge AWS, single-CPU, parallelism=31, hyperthreading disabled. Latest update: April 2025.

**Top performers on glove-100-angular:**
1. ScaNN (with SOAR) -- highest QPS at given recall
2. HNSW (hnswlib) -- consistently near Pareto front
3. FAISS IVF-PQ with FastScan -- excellent at high throughput

**Key findings across datasets:**
- Graph-based (HNSW) and tree-quantization (ScaNN) methods dominate
- ANNOY is "middle of the pack"
- LSH variants have closed the gap but still trail HNSW

---

### 3.2 Big-ANN-Benchmarks (NeurIPS 2023 Competition)

**Tracks:** Filtered, Out-of-Distribution, Sparse, Streaming

**Ongoing leaderboard (March 2024):**
- ScaNN with SOAR: top in streaming and OOD tracks
- Pinecone and Zilliz entries noted (closed-source)
- Significant improvement over 2021 competition baselines

**Key insight:** NeurIPS 2023 shifted focus from pure billion-scale to practical variants (filtered, streaming, OOD) -- reflecting real-world needs.

---

### 3.3 Performance by Dimensionality

| Dimensionality | Typical Source | Memory per 1M vectors (float32) | Notes |
|---|---|---|---|
| 128 | SIFT, legacy | ~488 MB | All algorithms perform well |
| 256 | Various | ~976 MB | PQ starts showing benefits |
| 384 | MiniLM, small models | ~1.46 GB | Good QPS/quality sweet spot |
| 768 | Cohere, CLIP ViT-L | ~2.93 GB | Standard benchmark dimension |
| 1024 | ImageBind, Titan | ~3.91 GB | "Sweet spot" for text-embedding-3-large |
| 1536 | OpenAI ada-002 | ~5.86 GB | Memory pressure significant |
| 2048+ | Large vision models | ~7.81+ GB | PQ/quantization essential |

**Key findings:**
- Reducing dimensionality from 768 to 384 improves QPS by ~1.5x and reduces latency by ~20%
- 1024 dims provides "pretty much the same performance" as 3072 dims for text-embedding-3-large
- At 1024 dims, HNSW graph overhead is only ~3% of vector storage (128 bytes vs. 4 KB per vector)
- Graph-based indexes read 95-97% fewer bytes than cluster-based indexes on most datasets

---

### 3.4 Performance by Scale

| Scale | Recommended Index | Typical QPS | Memory Approach |
|---|---|---|---|
| <100K | Flat or HNSW | 10K-100K+ | All in RAM |
| 100K-1M | HNSW | 1K-10K | RAM, optional PQ |
| 1M-10M | HNSW or IVF-PQ | 500-5K | RAM with quantization |
| 10M-100M | HNSW+PQ or IVF-PQ | 100-1K | RAM + quantization, or mmap |
| 100M-1B | DiskANN or SPANN | 50-500 | SSD + PQ in RAM |
| 1B+ | DiskANN or SPANN | 20-200 | Distributed + SSD |

---

### 3.5 Filtered Search Performance

**Benchmark across 6 vector engines (2025):**

| Engine | Unfiltered QPS (16 threads) | Filtered QPS (10% selectivity) | Notes |
|---|---|---|---|
| Pinecone | ~800 | ~600 | Integrated filtering |
| Zilliz (Milvus) | ~750 | ~700 | IVF + integrated filtering |
| Qdrant | High | High | Filterable HNSW |
| LanceDB IVF-PQ | Lower | Drops | Brute-force fallback |
| OpenSearch HNSW | Lower | Drops | Post-filter overhead |
| pgvector HNSW | Lower | Drops | Pre-filter overhead |

**Bottom line:** Engines with in-algorithm filtering (Pinecone, Zilliz, Qdrant) actually get faster under restrictive filters. Post-filter and pre-filter approaches degrade.

---

### 3.6 Vector Database Benchmarks (VectorDBBench)

**1M vectors, 768 dims (Cohere):**
- Purpose-built vector databases (Milvus, Qdrant, Zilliz) significantly outperform traditional database add-ons in both QPS and recall

**50M vectors, 768 dims (pgvector vs. Qdrant):**
- pgvectorscale: 471.57 QPS (11.4x higher throughput)
- Qdrant: 41.47 QPS (but 48% better p99 latency: 38.71 ms vs. 74.60 ms)

---

## 4. Multimodal-Specific Considerations

### 4.1 Embedding Dimensions by Modality

| Modality | Model | Dimensions | Notes |
|---|---|---|---|
| **Text** | MiniLM | 384 | Lightweight, fast |
| **Text** | Cohere embed-v3 | 768 | Standard quality |
| **Text** | OpenAI ada-002 | 1536 | High quality, expensive |
| **Text** | text-embedding-3-large | 256-3072 | Matryoshka (variable dim) |
| **Image** | CLIP ViT-B/32 | 512 | Standard CLIP |
| **Image** | CLIP ViT-L/14 | 768 | Higher quality |
| **Audio** | CLAP | 512 | Audio-text aligned |
| **Video** | VideoMAE, InternVideo | 512-768 | Frame-level or clip-level |
| **Cross-modal** | ImageBind | 1024 | 6 modalities unified |
| **Cross-modal** | Gemini embedding 2 | 128-1408 | Variable, multimodal |
| **Cross-modal** | Titan Multimodal | 256-1024 | Text+Image |
| **Cross-modal** | Voyage multimodal-3 | varies | 32K token limit |

### 4.2 Should Different Modalities Use Different Index Types?

**Answer: It depends on scale, not modality.**

The key insight is that dimensionality affects performance, not the modality itself. A 768-dim CLIP vector and a 768-dim Cohere text vector behave identically from the index perspective. The decision factors are:

1. **Collection size per modality:** If text embeddings are 100M+ vectors but image embeddings are only 1M, you might use DiskANN for text and HNSW for images.

2. **Query patterns:** If cross-modal search is primary (text-to-image, image-to-audio), all modalities should be in the same embedding space (e.g., ImageBind, CLIP) and indexed together.

3. **Dimensionality:** Higher-dim vectors (1536+) benefit more from PQ compression. Lower-dim vectors (384) may not need it.

**Practical recommendation:**
- Use a **unified index type** (HNSW) for collections sharing an embedding space (cross-modal models)
- Use **separate collections** for fundamentally different embedding spaces (a text-specific model vs. an image-specific model)
- Apply **quantization aggressiveness** based on dimensionality (more compression for higher dims)

### 4.3 Multi-Vector Search

For multimodal retrieval, multi-vector search is essential. Real-world items often need multiple embedding representations:

- **Same item, different views:** A product has title embedding (text), photo embedding (image), audio description (audio)
- **Cross-modal retrieval:** User uploads image + text query, both must be searched simultaneously
- **Independent semantic spaces:** Face embedding + voiceprint + fingerprint (no overlap)

**Approaches:**
| Platform | Method |
|---|---|
| Azure AI Search | Multiple vector fields, highest score across fields |
| Milvus | Parallel ANN searches with RRF reranking |
| Vespa | Multi-vector HNSW (multiple vectors per document) |
| Elasticsearch | Separate vector fields with hybrid scoring |
| Superlinked | Unified store with query-time attribute weighting |

**MUVERA (Google, 2025):** Transforms multi-vector retrieval into single-vector search via Fixed Dimensional Encodings (FDEs). Makes multi-vector retrieval as fast as single-vector search.

### 4.4 Unified vs. Separate Embedding Spaces

**Unified (ImageBind, Gemini Embedding 2, CLIP):**
- Pros: Single index, cross-modal search trivial, simpler architecture
- Cons: Jack-of-all-trades quality, limited to modalities the model supports

**Separate (specialized models per modality):**
- Pros: Best-in-class quality per modality, flexibility to swap models
- Cons: Cross-modal search requires fusion, multiple indexes to manage

**Recommendation for multimodal DB:** Support both. Use unified embeddings for cross-modal search use cases, and separate specialized embeddings when per-modality quality matters. The DB should support multiple vector fields per document.

---

## 5. Hybrid Approaches

### 5.1 Combining Algorithms by Scale

| Data Tier | Size | Index Type | Rationale |
|---|---|---|---|
| Hot (frequent queries) | <10M | HNSW (in-memory) | Lowest latency |
| Warm (moderate access) | 10M-100M | HNSW+PQ or IVF-PQ | Balanced cost/performance |
| Cold (infrequent) | 100M-1B+ | DiskANN or SPANN | Lowest cost, SSD-based |

**Real-world examples:**
- **Milvus 2.6 Tiered Storage:** Treats local storage as cache; hot data in RAM, warm data mmap'd, cold data on object storage. Reports 80% cost reduction.
- **Elasticsearch:** Hot/warm/cold/frozen tiers. Frozen tier uses searchable snapshots, extending capacity 20x over warm tier.
- **Weaviate Dynamic Index:** Starts with flat index, automatically switches to HNSW at 10K objects.

### 5.2 Algorithm Selection by Data Characteristics

| Characteristic | Best Algorithm | Alternative |
|---|---|---|
| Small collection (<50K) | Flat (brute-force) | Flat index |
| Medium, low-latency | HNSW | ScaNN |
| Large, memory-constrained | IVF-PQ | DiskANN |
| Billion-scale, SSD | DiskANN | SPANN |
| Frequent updates | HNSW (with MN-RU) | LSH |
| Heavy filtering | IVF or Filterable HNSW | ACORN |
| GPU available (build) | CAGRA -> export to HNSW | cuVS IVF-PQ |
| GPU available (search) | CAGRA | GPU IVF-PQ |

### 5.3 Pinecone's Adaptive Approach (Worth Emulating)

Pinecone's architecture demonstrates an effective hybrid strategy:
- **Small slabs (<100K):** Brute-force (exact, simple)
- **Medium slabs (<1M):** Proprietary FJLT-based algorithm ("Ananas")
- **Large slabs (>1M):** IVF with per-cluster sub-indexes
- Selection is **automatic** based on data characteristics -- users never configure it

This "right algorithm for the right scale" approach is a strong model for a new vector database.

### 5.4 Hybrid Search (Dense + Sparse)

Combining vector similarity (dense) with keyword search (sparse/BM25) consistently outperforms either alone:
- Reciprocal Rank Fusion (RRF) is the most common combination method
- Weaviate, Vespa, Elasticsearch all support natively
- Essential for RAG workloads where keyword precision matters alongside semantic recall

---

## 6. What Existing Vector DBs Chose and Why

### 6.1 Elasticsearch / OpenSearch: HNSW (via Lucene)

**Why HNSW:** Integrates naturally with Lucene's segment-based architecture. Lock-free search, incremental updates, parallel search across segments. HNSW was the natural fit because it supports incremental insertions (critical for Lucene's segment merging) and provides the best recall/speed tradeoff.

**Why Lucene (Elasticsearch):** "Lucene Is All You Need" thesis -- tight integration enables hybrid search (BM25 + vector), filtered search (ACORN), and leverages decades of search infrastructure investment.

**Why Multiple Engines (OpenSearch):** After the fork, OpenSearch added NMSLIB and FAISS engines alongside Lucene. FAISS for large-scale (IVF + HNSW), NMSLIB for HNSW, Lucene for smart filtering. Each has different trade-offs.

**Recent:** GPU acceleration via cuVS coming in Elasticsearch 9.3 and OpenSearch 3.0.

### 6.2 Milvus: Multiple Index Types (IVF, HNSW, DiskANN, GPU)

**Why multiple:** "No single index algorithm is optimal for all workloads." Milvus deliberately decouples the search engine from storage to support flexible index selection.

- **IVF:** Large-scale, filtered searches, faster builds, lower memory
- **HNSW:** Low-latency, high-accuracy, smaller datasets
- **DiskANN:** Datasets exceeding RAM capacity
- **GPU (CAGRA, IVF-PQ, Brute-Force):** Massively parallel throughput

**Architecture:** C++ search engine (30-70% faster than FAISS/hnswlib). Distributed with etcd + object storage + Kafka/Pulsar. Independent scaling of query nodes (reads) and data nodes (writes).

### 6.3 Qdrant: HNSW Only

**Why HNSW only:** Philosophy of deep optimization over breadth. Entire architecture (optimizers, payload indexes, filterable HNSW) is structured around serving one index type extremely well.

**Key innovations:**
- Filterable HNSW with query planner (auto-selects filter strategy)
- Rust implementation (memory-safe, low latency)
- Delta Encoding for HNSW graph compression
- GPU-accelerated indexing (NVIDIA, AMD, Intel) -- up to 10x faster
- Multiple quantization strategies (scalar, binary, asymmetric)

**Trade-off:** Less flexibility than Milvus for workloads better suited to IVF or DiskANN.

### 6.4 Weaviate: HNSW (with alternatives)

**Why HNSW default:** Scalable, fast, allows incremental updates, efficient for high-dimensional vectors. Architecture places vector index next to object store within a shard for self-contained units.

**Key design:** HNSW indexes cannot be merged efficiently, so Weaviate keeps one large HNSW per shard (unlike Lucene's segment approach). Uses WAL + snapshots for persistence.

**Additional indexes:**
- **Flat:** For small multi-tenant collections
- **Dynamic:** Auto-switches flat -> HNSW at 10K objects
- **HFresh (Preview):** Cluster-based (SPFresh algorithm) for large datasets with memory efficiency

**Recent:** cuVS integration (8x build speedup), ACORN for filtered search, HNSW+PQ for memory reduction.

### 6.5 Pinecone: Proprietary (Ananas + IVF + PQFS)

**Architecture:** Slab-based, log-structured merge tree. No user-facing index configuration.

**Algorithms:**
- Small slabs: Brute-force
- Medium slabs: "Ananas" (FJLT-based, proprietary)
- Large slabs: IVF with per-cluster Ananas sub-indexes
- Emerging: PQFS replacing Ananas at certain scales

**Key design:** Users never choose or tune algorithms. System auto-selects based on slab characteristics. Upgrades happen transparently via background compaction.

**Filtering:** Roaring bitmaps for metadata, pre-filtering approach. Low-cardinality bitmaps cached in memory; high-cardinality streamed from disk.

### 6.6 LanceDB: IVF-PQ (disk-first)

**Why IVF-PQ:** Built on Lance columnar format designed for disk-based ML workloads. IVF-PQ naturally fits disk storage patterns. Pure HNSW is too memory-intensive for LanceDB's disk-first philosophy.

**Variant indexes:**
- `IVF_PQ`: Default, good general purpose
- `IVF_SQ`: Scalar quantization (4x compression, simpler than PQ)
- `IVF_RQ`: RaBitQ quantization (extreme compression, 1-bit per dim)
- `IVF_HNSW_PQ` / `IVF_HNSW_SQ`: HNSW as sub-index within IVF partitions (disk-friendly + high recall)

**Design insight:** HNSW is available but only nested inside IVF partitions -- never as standalone. This preserves disk efficiency while gaining HNSW's accuracy within partitions.

### 6.7 Chroma: HNSW (via hnswlib)

**Why HNSW:** Simplicity and developer experience. Chroma targets the "easy to get started" segment. hnswlib provides a well-tested, performant HNSW implementation out of the box.

**Distributed mode:** Uses SPANN (SPFresh-based) for large-scale distributed scenarios.

**Limitation:** HNSW index grows unbounded with inserts/deletes (no pruning). Entire index (including deleted IDs) loaded into memory.

---

## 7. Recommendation

### 7.1 Primary Index Algorithm: HNSW

**Rationale:**
- Dominant algorithm across the industry for good reason: best general-purpose recall/speed tradeoff
- Supports incremental inserts (essential for a database)
- Well-understood tuning model (M, efConstruction, ef)
- Massive ecosystem (hnswlib, FAISS, cuVS integration)
- GPU-accelerated build via CAGRA (export CAGRA graph -> HNSW for CPU search)
- Works well across all dimensionalities relevant to multimodal (128-2048)
- Filterable HNSW implementations are mature (Qdrant, Weaviate ACORN patterns)

**Implementation strategy:**
- Custom Rust/C++ implementation (like Qdrant) for maximum performance and control
- Integrate filterable HNSW from day one
- Support HNSW+PQ / HNSW+SQ for memory-constrained deployments
- Build index on GPU (CAGRA via cuVS), convert to HNSW for CPU search

### 7.2 Secondary Index: DiskANN (Vamana)

**Rationale:**
- Essential for billion-scale and memory-constrained scenarios
- 90% cost reduction vs. pure HNSW at similar recall
- SQL Server 2025 integration validates enterprise readiness
- GPU build support via cuVS
- Natural complement to HNSW (same data, different cost/latency tradeoff)

**When to use:** Collections exceeding ~25% of available RAM, or when cost optimization is critical.

### 7.3 Tertiary Index: IVF-PQ / IVF-RaBitQ

**Rationale:**
- Best for extremely large-scale batch/analytics workloads
- Efficient filtered search (cluster-level filtering)
- GPU-accelerated via cuVS
- IVF-RaBitQ (Extended RaBitQ) is the emerging replacement for IVF-PQ with better accuracy

### 7.4 Support Multiple Index Types (Like Milvus, Not Like Qdrant)

**Recommendation: Yes, support at least 3 index types.**

The Milvus approach of supporting multiple indexes is correct for a multimodal database because:
1. Different collections have wildly different sizes (text corpus at 1B vs. audio clips at 100K)
2. Different cost/latency requirements per use case
3. Users should be able to pick the right tradeoff

However, avoid the complexity of supporting too many variants. Recommended set:

| Index Type | Use Case | Priority |
|---|---|---|
| **HNSW** | Default, low-latency, high-recall | P0 (launch) |
| **HNSW+Quantization** | Memory-optimized HNSW | P0 (launch) |
| **DiskANN** | Billion-scale, cost-optimized | P1 (fast follow) |
| **Flat** | Small collections, ground truth | P0 (launch) |
| **IVF-PQ/RaBitQ** | Batch analytics, extreme scale | P2 |
| **GPU (CAGRA)** | Build acceleration | P1 |

### 7.5 GPU Acceleration Strategy

**Phase 1 (Build acceleration):**
- Integrate cuVS for GPU-accelerated index building
- Build CAGRA on GPU, export to HNSW for CPU search (supported by cuVS)
- 12x+ build speedup is transformative for large collections
- Support both NVIDIA (cuVS) and AMD (hipVS) where possible

**Phase 2 (Search acceleration):**
- GPU-accelerated CAGRA search for high-throughput pipelines
- GPU brute-force for small collections requiring exact results
- cuVS IVF-PQ for batch GPU search workloads

**Phase 3 (Integrated):**
- Dynamic batching API for online GPU search
- GPU-CPU hybrid: route queries to GPU or CPU based on load

### 7.6 Handling Varying Dimensionalities Across Modalities

**Architecture:**

```
Collection: "products"
  |-- vector_field: "text_embedding"    (768 dims, Cohere)     -> HNSW index
  |-- vector_field: "image_embedding"   (512 dims, CLIP)       -> HNSW index  
  |-- vector_field: "audio_embedding"   (512 dims, CLAP)       -> HNSW index
  |-- vector_field: "cross_modal"       (1024 dims, ImageBind) -> HNSW index
  |-- scalar_fields: metadata, timestamps, etc.
```

**Per-field index configuration:**
- Each vector field gets its own index (potentially different type/parameters)
- Dimensionality-aware defaults: lower `M` for 128-dim vectors, higher for 1024+
- Quantization settings scaled to dimensionality:
  - 128-384 dims: Scalar quantization (int8) sufficient
  - 512-768 dims: Scalar or binary quantization (RaBitQ)
  - 1024+ dims: Product quantization or RaBitQ for significant savings

**Multi-vector search support:**
- Support querying multiple vector fields in a single request
- Implement score fusion (RRF, weighted combination)
- Allow per-field search parameters (ef, nprobe, etc.)

**Auto-tuning:**
- Auto-select quantization based on dimensionality and collection size
- Dynamic index switching (flat -> HNSW at threshold, like Weaviate)
- Suggest index type based on data characteristics (like Pinecone's automatic selection)

### 7.7 Summary Decision Matrix

| Decision | Choice | Reasoning |
|---|---|---|
| Primary index | HNSW | Best recall/speed, ecosystem support, GPU build |
| Disk-based index | DiskANN (Vamana) | Proven billion-scale, 90% cost reduction |
| Quantization | RaBitQ (primary), PQ (fallback) | RaBitQ is theoretically optimal, 3x faster than PQ |
| GPU strategy | cuVS (CAGRA build, IVF-PQ search) | 12x build, 8x search improvements |
| Filtering | Integrated filterable HNSW | 1.5x+ faster than post-filter |
| Multi-vector | Per-field indexes + score fusion | Essential for multimodal |
| Implementation language | Rust (core) + C++ (cuVS bindings) | Safety + performance |
| Dynamic updates | MN-RU or similar streaming HNSW | Avoid full rebuilds |

---

## Appendix: Key Sources

### Research Papers
- Malkov & Yashunin, "Efficient and robust approximate nearest neighbor search using HNSW" (2016)
- Jayaram Subramanya et al., "DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node" (NeurIPS 2019)
- Chen et al., "SPANN: Highly-efficient Billion-scale Approximate Nearest Neighbor Search" (NeurIPS 2021)
- Guo et al., "Accelerating Large-Scale Inference with Anisotropic Vector Quantization" (ICML 2020)
- Gao & Long, "RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound" (SIGMOD 2024)
- Extended RaBitQ (SIGMOD 2025)
- Fu et al., "Fast Approximate Nearest Neighbor Search With The Navigating Spreading-out Graphs" (PVLDB 2019)
- SOAR: "Spilling with Orthogonality-Amplified Residuals" (NeurIPS 2023)
- MARGO: "Monotonic Path Aware Graph Layout Optimization for Disk-Based ANN" (VLDB 2025)
- "Enhancing HNSW Index for Real-Time Updates" (2024)

### Benchmarks
- [ANN-Benchmarks](https://ann-benchmarks.com/)
- [Big-ANN-Benchmarks](https://big-ann-benchmarks.com/)
- [VectorDBBench](https://github.com/zilliztech/VectorDBBench)
- [Qdrant Benchmarks](https://qdrant.tech/benchmarks/)

### Industry Sources
- [NVIDIA cuVS Blog](https://developer.nvidia.com/blog/optimizing-vector-search-for-indexing-and-real-time-retrieval-with-nvidia-cuvs/)
- [Meta FAISS + cuVS](https://engineering.fb.com/2025/05/08/data-infrastructure/accelerating-gpu-indexes-in-faiss-with-nvidia-cuvs/)
- [Elasticsearch Vector Search Rationale](https://www.elastic.co/search-labs/blog/vector-search-elasticsearch-rationale)
- [Milvus Index Guide](https://milvus.io/docs/index-explained.md)
- [Qdrant Architecture](https://qdrant.tech/articles/dedicated-vector-search/)
- [Weaviate Vector Indexing](https://docs.weaviate.io/weaviate/concepts/vector-index)
- [Pinecone Architecture](https://www.pinecone.io/how-pinecone-works/)
- [LanceDB Vector Indexes](https://docs.lancedb.com/indexing/vector-index)
- [Google SOAR Blog](https://research.google/blog/soar-new-algorithms-for-even-faster-vector-search-with-scann/)
- [Google MUVERA Blog](https://research.google/blog/muvera-making-multi-vector-retrieval-as-fast-as-single-vector-search/)
- [DiskANN Explained (Milvus)](https://milvus.io/blog/diskann-explained.md)
- [RaBitQ (LanceDB)](https://www.lancedb.com/blog/feature-rabitq-quantization)
- [Filtered Vector Search (Weaviate ACORN)](https://weaviate.io/blog/speed-up-filtered-vector-search)
- [Filtered Vector Search (Qdrant)](https://qdrant.tech/articles/vector-search-filtering/)
- [Milvus Tiered Storage](https://milvus.io/blog/milvus-tiered-storage-80-less-vector-search-cost-with-on-demand-hot%E2%80%93cold-data-loading.md)

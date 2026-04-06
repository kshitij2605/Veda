# Vector Database Research: OpenSearch, FAISS, and LanceDB

**Research Date:** April 2026
**Purpose:** Evaluate vector databases for a new multimodal vector database project

---

## Table of Contents

1. [OpenSearch](#1-opensearch)
2. [FAISS (Facebook AI Similarity Search)](#2-faiss-facebook-ai-similarity-search)
3. [LanceDB](#3-lancedb)
4. [Comparative Summary](#4-comparative-summary)

---

## 1. OpenSearch

### 1.1 Overview

| Attribute | Detail |
|---|---|
| **What it is** | Distributed search and analytics engine with vector search capabilities via the k-NN plugin |
| **Origin** | Fork of Elasticsearch 7.10.2, created by Amazon Web Services in 2021 |
| **Maintainer** | OpenSearch Software Foundation (a Linux Foundation project) |
| **License** | Apache License 2.0 |
| **Primary Language** | Java |
| **Search Engine Base** | Apache Lucene 10 (since v3.0) |
| **Latest Version** | 3.5.0 (March 15, 2026) |
| **Named** | Leader and Fast Mover in the 2025 GigaOm Radar for Vector Databases |

OpenSearch is a full-featured distributed search, analytics, and observability platform that has expanded into vector search territory. Its vector capabilities are delivered primarily through the **k-NN plugin** and the **Neural Search plugin**. The k-NN plugin provides core approximate nearest neighbor (ANN) search, while the Neural Search plugin handles automatic embedding generation at both indexing and search time.

### 1.2 Architecture

#### Vector Engine

OpenSearch's vector search is powered by its **Vector Engine**, which handles unstructured and high-dimensional data using numerical embeddings. The engine supports three underlying libraries:

| Engine | License | Supported Algorithms | Notes |
|---|---|---|---|
| **NMSLIB** | Apache 2.0 | HNSW | Does NOT support filters; oldest engine |
| **FAISS** | MIT | HNSW, IVF, IVF-PQ | Most feature-rich; GPU acceleration support |
| **Lucene** | Apache 2.0 | HNSW | Native to OpenSearch; tightest integration |

#### Index Types and Algorithms

- **HNSW (Hierarchical Navigable Small World):** Best query latency/quality tradeoff when memory is not a concern. Graph-based approach with configurable `m` (connections per node) and `ef_construction` / `ef_search` parameters.
- **IVF (Inverted File Index):** Uses less memory and indexes faster than HNSW with similar query quality. Clusters vectors into Voronoi cells.
- **IVF-PQ (Inverted File + Product Quantization):** Adds lossy vector compression on top of IVF for significant memory reduction at the cost of some recall.

#### Search Methods

1. **Approximate search (ANN):** Sacrifices some accuracy for lower latency and smaller memory footprint. Best for most use cases.
2. **Exact search (brute-force):** Exhaustive k-NN search. Guaranteed correct results but does not scale.
3. **Radial search:** Searches based on distance or score thresholds rather than fixed k.

#### Supported Space Types (Distance Metrics)

| Space Type | Description | Engines |
|---|---|---|
| `l2` | Euclidean distance (default) | NMSLIB, FAISS, Lucene |
| `cosinesimil` | Cosine similarity | NMSLIB, FAISS, Lucene |
| `innerproduct` | Dot product | NMSLIB, FAISS, Lucene |
| `hamming` | Hamming distance (binary vectors) | FAISS (v2.16+) |
| `hammingbit` | Hamming bit distance (scoring script) | Script-based |

#### Vector Data Types

- **`float`** (default): 4 bytes per dimension, up to 16,000 dimensions
- **`byte`**: Each dimension in [-128, 127] range (since v2.9)
- **`binary`**: Binary vectors with Hamming distance

#### Memory Architecture

OpenSearch uses half of instance RAM for the Java heap (up to 32 GiB). The k-NN plugin uses up to 50% of the remaining half for graph data. Example: a 32 GiB RAM instance accommodates ~8 GiB of graph data (32 x 0.5 x 0.5). Performance degrades if graph memory exceeds this threshold.

#### Storage Engine

OpenSearch uses a segment-based Lucene storage engine. Vector indices are stored alongside traditional inverted indices in Lucene segments. The k-NN plugin manages off-heap native memory for NMSLIB and FAISS engines, while Lucene-based vectors use Lucene's native HNSW implementation.

#### GPU Acceleration (Experimental, v3.0+)

OpenSearch 3.0 introduced GPU-accelerated vector indexing using **NVIDIA cuVS**:
- Uses the **CAGRA** algorithm (CUDA ANN GRAph-based), built from the ground up for GPU acceleration
- Achieves up to **9.3x faster** index builds and **3.75x cost reduction**
- GPU workers construct a graph representation using IVF-PQ or NN-DESCENT, then optimize by removing redundant paths
- Falls back to CPU-based indexing on error
- Vectors are uploaded to an object store during flush/merge, then a GPU fleet builds the index

#### Concurrent Segment Search

Enabled by default for k-NN since v3.0, delivering up to **2.5x faster query performance** by searching across segments in parallel.

### 1.3 Multimodal Support

OpenSearch provides multimodal search through the **Neural Search plugin** (introduced in v2.11):

**Natively Supported Modalities:**
- **Text:** Full native support via text embedding processors
- **Images:** Supported via `text_image_embedding` processor with models like CLIP and Amazon Titan Multimodal Embeddings

**Not Yet Natively Supported (require external embedding APIs):**
- **Audio:** Must use external embedding APIs (e.g., TwelveLabs) and index resulting vectors
- **Video:** Workaround available via TwelveLabs Embed API (generates 1024-dimensional embeddings from video content capturing visual, audio, and textual elements)

**Metadata/Payload Handling:**
- Vectors are stored alongside traditional document fields in the same index
- Full support for structured metadata, nested objects, arrays
- Hybrid search combines vector similarity with keyword/filter queries
- ML Connectors integrate with Amazon Bedrock, SageMaker, OpenAI, Cohere for embedding generation

**Supported Multimodal Embedding Models:**
- Amazon Titan Multimodal Embeddings (1024, 384, or 256 dimensions)
- OpenAI CLIP
- TwelveLabs Embed API (video)
- Any model via custom ML connectors

### 1.4 Shortcomings

1. **Slower vector search vs. competitors:** Elastic's benchmarks claim Elasticsearch is up to 12x faster than OpenSearch for pure vector search (though OpenSearch disputes this).
2. **Engine-specific limitations:** NMSLIB does not support filters. Serverless vector collections only support HNSW with FAISS (no IVF, no Lucene engine).
3. **60-second refresh interval** on Serverless vector collections -- problematic for near-real-time use cases.
4. **Compute-intensive index building at scale:** Uber reported >10 hours to build a base index for billion-level vector data with default settings.
5. **Heavy memory requirements:** High-dimensional vectors consume significant off-heap memory. The 16,000-dimension limit constrains some use cases.
6. **Algorithm diversity gaps:** Uber noted being "limited by the lack of algorithm options, which hindered our ability to fine-tune trade-offs for different scenarios."
7. **Feature gaps in older versions:** Semantic sentence highlighting only arrived in v3.0. Concurrent segment search was optional pre-3.0.
8. **Operational complexity:** Being a distributed system, it requires cluster management, shard tuning, and capacity planning.
9. **Filtered search performance:** Both OpenSearch-HNSW and brute-force fallbacks can spike to 200-300ms latency under tight filter conditions.
10. **GPU support still experimental:** GPU-accelerated indexing is a preview feature as of v3.0.

### 1.5 Paid Tier / Cloud Offering

#### Amazon OpenSearch Service -- Managed Clusters (Provisioned)

Pay for always-on infrastructure: instance hours, EBS storage, data transfer, and managed S3-backed storage.

| Instance Family | Released | Key Benefit | Sample Rate |
|---|---|---|---|
| **OR2** (compute-optimized) | March 2025 | 26% higher indexing throughput than OR1 | or2.2xlarge: $0.80/hr |
| **OM2** (memory-optimized) | March 2025 | 15% higher throughput than OR1 | om2.large: $0.153/hr |
| **OI2** (I/O-optimized) | December 2025 | 9% higher throughput than OR2, writable warm tier | -- |

**Reserved Instance Discounts:**
- 1-year No Upfront: 31% discount
- 3-year No Upfront: 48% discount
- 1-year All Upfront: 35% discount
- 3-year All Upfront: 52% discount

**Storage Tiers:**
- **Hot:** Standard EBS-backed storage
- **UltraWarm:** S3-backed, ~90% cheaper per GB than hot storage, interactive queries
- **Cold:** Long-term retention, lowest cost

**Free Tier:** 750 instance-hours/month on t2/t3.small.search (single-AZ) + 10 GB EBS for first 12 months.

#### Amazon OpenSearch Serverless

- Uses OpenSearch Compute Units (OCUs): **$0.24/OCU-hour**
- Minimum production: 2 OCUs = ~**$350/month**
- Dev/test (no redundancy): 1 OCU = ~**$175/month**
- Storage: $0.024/GB/month on S3
- GPU Vector Acceleration OCU: 6 GB GPU memory, 8 GiB RAM, 2 vCPUs per OCU

#### Database Savings Plans

Launched March 5, 2026, covering both Managed Clusters and Serverless.

#### Alternative Managed Providers

- **Aiven:** Multi-cloud (AWS, GCP, Azure, OCI), starting ~$19/month, running OpenSearch 3.3.2 as of January 2026
- **Instaclustr (NetApp):** On-premises option, 99.999% uptime SLA, broadest compliance

### 1.6 GUI / Dashboard

**OpenSearch Dashboards** is the primary visualization and management UI, forked from Kibana 7.10.2 and significantly enhanced:

#### Visualization Types

| Type | Description |
|---|---|
| **Area/Line/Bar Charts** | Standard time-series and categorical visualizations |
| **Pie Charts** | Proportional data display |
| **Heat Maps** | Density-based visualizations |
| **Coordinate Maps** | GPS/latitude-longitude geographic data on maps |
| **Region Maps** | Custom vector maps with geographic boundaries |
| **Vega/Vega-Lite** | Declarative grammar for advanced custom visualizations |
| **VisBuilder** | Drag-and-drop visualization builder (no code required) |
| **TSVB** | Time-series visual builder (Area, Line, Metric, Gauge, Markdown, Data Table) |
| **Markdown** | Embed formatted text/documentation in dashboards |
| **Data Tables** | Tabular data representation |

#### Key Features

- **Discover Tool:** Interactive data exploration with DQL (Dashboards Query Language), search, sort, filter
- **Workspaces:** Isolated environments for different teams/use cases (e.g., observability, security)
- **Alerting:** Built-in alert configuration and monitoring
- **Anomaly Detection:** ML-powered anomaly detection visualization
- **Security Analytics:** SIEM-like security event dashboards
- **Query Monitoring:** Live Queries API for real-time tracking (v3.0+)
- **SQL/PPL Support:** Query data using SQL or Piped Processing Language

### 1.7 Data Visualization Platform (Kibana Equivalent)

**OpenSearch Dashboards is the direct equivalent of Kibana** for the OpenSearch ecosystem. Key comparison:

| Feature | OpenSearch Dashboards | Kibana |
|---|---|---|
| **License** | Apache 2.0 (fully open-source) | Elastic License / SSPL |
| **Drag-and-drop builder** | VisBuilder | Lens |
| **Presentation mode** | Not available | Canvas |
| **Query languages** | Lucene, DSL, SQL, PPL | Lucene, DSL, SQL, KQL, EQL, ES\|QL |
| **Geospatial** | Coordinate & Region Maps | Elastic Maps (more advanced) |
| **Plugins** | Community-driven (SQL, alerting, anomaly detection, etc.) | Larger ecosystem (ML, SIEM, enterprise search) |
| **Custom visualizations** | Vega/Vega-Lite + community plugins (Google Charts, D3.js, Apache ECharts) | Vega/Vega-Lite + Elastic plugins |

**OpenSearch Dashboards Strengths:**
- Fully open-source with no feature gating
- SOC 2, CMMC, NIST compliance
- Can import Kibana visualization objects (migration from `.kibana*` indexes)
- Active community contributing visualization plugins (e.g., custom Mustache transforms, D3.js integration)

**OpenSearch Dashboards Limitations vs. Kibana:**
- No Canvas (presentation mode) equivalent
- No Lens (AI-suggested visualizations) equivalent
- Smaller plugin ecosystem overall
- No ES|QL equivalent (though PPL serves a similar role)

**For Vector Data Exploration Specifically:**
- No dedicated vector/embedding visualization tool (e.g., no t-SNE/UMAP projector)
- Vector data can be explored through standard document views in Discover
- Custom Vega visualizations can be built for embedding analysis
- Third-party tools (e.g., TensorFlow Projector) would be needed for embedding space visualization

### 1.8 API and SDK Support

| Language | Client |
|---|---|
| **Java** | opensearch-java (recommended; high-level REST client deprecated in 3.0) |
| **Python** | opensearch-py (Apache 2.0, fork of elasticsearch-py) |
| **JavaScript/Node.js** | opensearch-js |
| **Go** | opensearch-go |
| **Ruby** | opensearch-ruby |
| **PHP** | opensearch-php |
| **.NET** | opensearch-net |
| **Rust** | opensearch-rs |
| **Hadoop** | opensearch-hadoop |

**Protocols:**
- **REST API:** Primary interface, comprehensive OpenAPI 3.1.0 specification
- **gRPC API (Experimental, v3.0+):** Protocol buffers over gRPC, concurrent requests over single TCP connections, reduced serialization overhead vs. JSON

**Note:** Elasticsearch clients are NOT compatible with OpenSearch 2.0+.

### 1.9 Performance Benchmarks

- **OpenSearch 3.0 benchmarks:** 10x improvement in general search performance, 2.5x improvement in vector search, 25% faster range queries, 75% reduction in p90 latency for high-cardinality aggregations
- **GigaOm 2025 Radar:** Up to 17% throughput improvement and 22% latency improvement for vector search
- **Filtered search:** Spikes to 200-300ms latency under tight filter conditions (similar to LanceDB)
- **Uber scale:** Successfully deployed at billion-scale vector search after significant tuning of ingestion and query parameters
- **Elastic's claim:** Elasticsearch is up to 12x faster for pure vector search (Elastic-run benchmarks, disputed by OpenSearch community)
- **GPU-accelerated indexing:** 9.3x faster index builds with NVIDIA cuVS (preview)

---

## 2. FAISS (Facebook AI Similarity Search)

### 2.1 Overview

| Attribute | Detail |
|---|---|
| **What it is** | A library (NOT a database) for efficient similarity search and clustering of dense vectors |
| **Maintainer** | Meta's Fundamental AI Research (FAIR) group |
| **License** | MIT |
| **Primary Language** | C++ (59.4%), with Python (22.0%) and CUDA (14.6%) |
| **Latest Version** | v1.14.1 (March 6, 2026) |
| **GitHub Stars** | 39.6k |
| **Scale** | Used at Meta to index 1.5 trillion 144-dimensional vectors |

FAISS is explicitly NOT a database. From the official paper: "Faiss is not a database -- it does not provide concurrent write access, load balancing, sharding, transaction management or query optimization." It is a focused, high-performance library for approximate nearest neighbor search (ANNS) that is commonly used as the core engine inside actual vector databases (OpenSearch, Milvus, Vearch).

### 2.2 Architecture

#### Core Design

FAISS builds an in-memory data structure (an "index") from a set of vectors. Given a query vector, it efficiently searches this index. The architecture is built around the concept of composable index types that can be stacked and combined.

#### Index Types

| Index | Description | Memory | Speed | Accuracy |
|---|---|---|---|---|
| **IndexFlatL2 / IndexFlatIP** | Brute-force exact search | Full dataset in RAM | Slow at scale | 100% |
| **IndexIVFFlat** | IVF clustering + flat search within clusters | Full dataset + centroids | Fast (only searches relevant clusters) | High |
| **IndexIVFPQ** | IVF + Product Quantization compression | Compressed (~10-50x smaller) | Very fast | Moderate (lossy) |
| **IndexIVFScalarQuantizer** | IVF + Scalar Quantization | Reduced precision | Fast | High |
| **IndexHNSW** | Hierarchical Navigable Small World graph | Full dataset + graph | Very fast queries, slow build | Very high |
| **IndexNSG** | Navigating Spreading-out Graph | Full dataset + graph | Fast | High |
| **IndexBinaryFlat** | Binary vector brute-force | Binary vectors | Fast | 100% |
| **IndexBinaryIVF** | Binary vector IVF | Binary vectors + centroids | Fast | High |
| **IndexPQ** | Pure Product Quantization (no IVF) | Very compressed | Fast | Moderate |
| **IndexLSH** | Locality-Sensitive Hashing | Hashes | Fast | Low-moderate |

**New in v1.13-v1.14 (2025-2026):**
- **RaBitQ (Random Bit Quantization):** New quantization method with multi-bit support (2-9 bits), SIMD-optimized inner products, and FastScan variants
- **Panorama architecture:** New IndexFlatL2Panorama, IndexFlatIPPanorama, integrated into HNSW variants
- **Hadamard transformation as an index for IVF**
- **Binary CAGRA** for graph-based search on binary vectors
- **bfloat16 brute-force kNN support** on GPU

#### Distance Metrics

- **L2 (Euclidean distance):** Primary metric
- **Inner Product (dot product):** Primary metric
- **Cosine similarity:** Via normalized vectors + inner product
- **L1 (Manhattan distance):** Limited support
- **Linf (Chebyshev distance):** Limited support
- **Hamming distance:** For binary vectors

#### Hardware Acceleration

- **CPU:** Multi-threaded via OpenMP, BLAS (OpenBLAS or Intel MKL), custom SIMD kernels (x86 SSE/AVX/AVX-512, ARM Neon)
- **GPU (NVIDIA CUDA):** Single and multi-GPU support, automatic CPU/GPU memory transfers
- **GPU (AMD ROCm):** Supported since recent releases
- **NVIDIA cuVS:** Optional backend for IVF-Flat, IVF-PQ, and CAGRA indices
- **Intel SVS:** Scalable Vector Search integration (v1.13+)

### 2.3 Multimodal Support

**FAISS has NO native multimodal support.**

- FAISS works exclusively with dense numerical vectors (float32, float16, bfloat16, int8, binary)
- It does NOT extract features -- you must generate embeddings externally using models like CLIP, BERT, etc.
- It does NOT store metadata or payloads alongside vectors
- It does NOT handle raw data (images, audio, video, text) in any way
- To build a multimodal system with FAISS, you need a separate embedding pipeline and a separate metadata store

### 2.4 Shortcomings

1. **Not a database:** No persistence, concurrency, sharding, transactions, or query optimization
2. **Not a service:** No built-in server, API, or network interface -- runs only in the calling process
3. **No metadata filtering:** Cannot combine vector search with structured data lookups natively
4. **No feature extraction:** Requires external embedding generation
5. **Memory-intensive:** Large-scale deployments require careful index selection to fit in RAM. For compressed indexes beyond 10M vectors, IVF is the only practical option.
6. **Limited incremental indexing:** Adding vectors without full index rebuild is limited. Rebuilding billion-scale IVF-PQ indices is extremely time-consuming and compute-expensive.
7. **No built-in persistence:** Index must be explicitly saved/loaded from disk; no automatic durability
8. **Complex parameter tuning:** Choosing the right index type, nprobe, nlist, m, efConstruction, etc. requires expertise
9. **Construction time at scale:** HNSW graph construction becomes the bottleneck beyond ~10M vectors
10. **Limited distance metrics:** Primarily L2 and inner product; other metrics have limited support

### 2.5 Paid Tier / Cloud Offering

**FAISS has no paid tier or cloud offering.** It is entirely free and open-source under the MIT license.

FAISS is used as the underlying engine in several commercial products:
- Amazon OpenSearch Service (uses FAISS as one of its k-NN engines)
- Milvus / Zilliz Cloud (uses FAISS for some index types)
- Various other vector databases embed FAISS internally

### 2.6 GUI / Dashboard

**FAISS has no GUI or dashboard.** It is a pure library with a C++ and Python API. All interaction is programmatic.

For visualization of FAISS-indexed data, users typically use:
- Matplotlib / Seaborn for plotting
- TensorBoard Projector for embedding visualization
- Custom Jupyter notebook workflows
- Third-party tools built on top of FAISS

### 2.7 Data Visualization Platform

**No equivalent to Kibana exists for FAISS.** Since FAISS is a library, not a platform, there is no built-in visualization or exploration tool.

### 2.8 API and SDK Support

| Language | Support Level |
|---|---|
| **C++** | Native (primary implementation) |
| **Python** | Full wrappers via SWIG (complete numpy integration) |
| **C** | Optional C API (enabled via CMake flag) |

**Community bindings (via C API):**
- Go bindings exist (community-maintained)
- Rust bindings exist (community-maintained)
- Java/JNI bindings exist (used by OpenSearch's k-NN plugin)

**Installation Methods:**
- **Conda** (official/recommended): `faiss-cpu`, `faiss-gpu`, `faiss-gpu-cuvs` via pytorch channel
- **PyPI:** `faiss-cpu` (official), `faiss-gpu-cu12` / `faiss-gpu-cu11` (unofficial GPU wheels)
- **Build from source:** CMake-based, highly configurable

**Supported Python versions:** 3.10-3.13

### 2.9 Performance Benchmarks

- **Meta production scale:** Indexes 1.5 trillion 144-dimensional vectors
- **Speed claim:** 8.5x faster than previous best methods for similarity search
- **NeurIPS'23 Filtered Search Track:** FAISS baseline hit ~3k QPS at 90% recall on 10M vectors; optimized methods reached 30-40k QPS at same recall (10x gap)
- **In-memory HNSW:** Generally the fastest option for pure vector search when properly configured
- **GPU acceleration:** Significant speedups for IVF-based indices, especially with cuVS/CAGRA
- **Key advantage:** Raw, unfiltered vector search speed is FAISS's core strength
- **Key weakness:** No native filtering means filtered search requires post-filtering or integration with external systems

---

## 3. LanceDB

### 3.1 Overview

| Attribute | Detail |
|---|---|
| **What it is** | Open-source, embedded/serverless multimodal vector database ("AI-native lakehouse") |
| **Maintainer** | LanceDB, Inc. |
| **License** | Apache License 2.0 |
| **Primary Language** | Rust (core), with Python, TypeScript, and Java SDKs |
| **Latest Milestone** | SDK 1.0.0 (Rust, Python, Java); Lance format 2.2 |
| **Funding** | $30M Series A (June 2025); total raised includes earlier $11M |
| **Key Customers** | Midjourney, Uber, Netflix, Hugging Face, Bytedance, Huawei, Tencent, Alibaba |

LanceDB is purpose-built for multimodal AI workloads. It stores, indexes, and retrieves billions of vectors alongside raw images, video, audio, point clouds, JSON metadata, and text without separate storage systems. It operates as an embedded database (like SQLite for vectors) or as a serverless cloud service.

### 3.2 Architecture

#### Embedded & Serverless Design

LanceDB runs in-process (embedded) or as a serverless cloud service. No separate database server is required for the embedded mode. This makes it uniquely suitable for edge deployment, local development, and Lambda/serverless architectures.

#### Lance Columnar Format

The foundation is **Lance**, a modern columnar data format implemented in Rust:
- **~100x faster random access** than Parquet
- Cloud-friendly scan performance on object storage (S3, GCS, Azure Blob)
- Zero-copy versioning with ACID transactions
- Fine-grained schema evolution
- Native blob storage for multimodal data (images, video, audio)

**Lance Format 2.2** introduced:
- **Blob V2:** Redesigned blob storage with four semantics (Inline, Packed, Dedicated, External)
- Flexible nested field evolution
- Native Map types
- Broad compression uplift
- 1-2 round-trip random-access I/O

#### Query Engine

Built on **Apache Arrow** for in-memory operations and **Apache DataFusion** for query execution:
- Full SIMD instruction set utilization for vectorized computation
- Columnar storage optimized for batch reading
- Disk-based queries at near in-memory speeds via memory-mapped files
- DuckDB integration for Lance-native SQL retrieval (January 2026)

#### Indexing Methods

| Index Type | Description |
|---|---|
| **IVF-PQ** | Inverted File Index + Product Quantization |
| **IVF-HNSW-PQ** | Sub-HNSW indices within each IVF partition (unique approach) |
| **RaBitQ** | Higher compression, faster indexing, better recall on high-dimensional embeddings |
| **Full-text index** | BM25-based full-text search |
| **Scalar index** | B-tree index for metadata filtering |
| **R-Tree index** | Geospatial indexing (new) |

**Unique IVF-HNSW-PQ approach:** Unlike standard HNSW implementations, LanceDB does NOT build a single HNSW graph over the entire dataset. Instead, it builds sub-HNSW indices within each IVF partition, which reduces required partitions and improves search efficiency.

#### Versioning and Data Management

- **Git-style branching and tagging** for ML/AI experimentation
- **Time travel** to query any historical version
- **Shallow clone** for efficient dataset sharing
- Automatic versioning with ACID transactions

### 3.3 Multimodal Support

**LanceDB has the strongest native multimodal support of the three databases researched.**

| Capability | Detail |
|---|---|
| **Text** | Native storage and embedding search |
| **Images** | Stored as blobs inline with metadata and embeddings; CLIP integration |
| **Video** | Native blob storage; TwelveLabs integration for video embeddings |
| **Audio** | Native blob storage; extensible embedding model support |
| **Point clouds** | Supported as native data type |
| **Metadata/Payloads** | Full structured metadata co-located with vectors and blobs in single tables |

**Key Multimodal Features:**
- Blobs are **first-class citizens** in Lance format -- images, video, audio stored alongside embeddings and metadata in the same table
- No external file system or object store needed for raw data
- Blob V2 provides four storage semantics that automatically adapt to workloads
- CLIP embedding workflows are built-in (e.g., multimodal fashion search)
- Hugging Face Hub integration for distributing multimodal datasets as single searchable artifacts
- Planned native media type support in the type system (image width/height, audio sample rate, etc.)

**Metadata Handling:**
- Full Apache Arrow type system (nested structs, lists, maps, etc.)
- SQL filtering via DuckDB integration
- Scalar indices for fast metadata filtering
- Vector search + metadata filter in single query

### 3.4 Shortcomings

1. **Concurrent write limitations:** Too many concurrent writers can lead to failing writes due to limited commit retries
2. **Single-record insert performance:** Each insert creates a new data fragment; batching is strongly recommended
3. **Row ID stability:** Only experimental support for stable row IDs through compaction; unstable through updates
4. **Index rebuild overhead:** Adding/modifying records often requires reindexing for optimal query performance
5. **Batch size configuration complexity:** The team acknowledges batch sizes are an "annoying nuisance" -- hard to set correctly across reads, compaction, merge inserts
6. **Filtered search performance degradation:** IVF-PQ with tight filters can spike to 200-300ms latency
7. **Curse of dimensionality:** High-dimensional vectors degrade search accuracy/efficiency; requires careful parameter tuning
8. **Reliance on external embedding models:** No built-in embedding generation; depends on OpenAI, Hugging Face, etc.
9. **Python multiprocessing incompatibility:** Fork-based multiprocessing is not supported; must use spawn
10. **Limited query capabilities vs. SQL databases:** Focus on vector operations; complex relational queries are limited
11. **Lossy PQ compression:** Product Quantization reduces recall; users must carefully configure refinement factors
12. **Younger ecosystem:** Less battle-tested than OpenSearch or FAISS at massive scale (though Uber and Midjourney are production users)

### 3.5 Paid Tier / Cloud Offering

| Tier | Price | Key Features |
|---|---|---|
| **LanceDB OSS** | Free | 100% open source, embedded, runs locally or in your cloud, no vendor lock-in |
| **LanceDB Cloud** | Starting at **$16.03/month** | Serverless, automatic compaction/indexing/cleanup, same SDK as local |
| **LanceDB Enterprise** | Custom pricing (annual commitment) | Self-managed, enterprise security (SOC 2 Type II, HIPAA, GDPR), dedicated support, flexible deployment |

**LanceDB Enterprise** is available on the **AWS Marketplace** with annual commitment for volume-based discounts.

**Cloud features:** Decoupled compute and storage leveraging S3, seamless local-to-cloud transition with same codebase, managed service with technical support (Slack, email, phone).

### 3.6 GUI / Dashboard

#### Lance Data Viewer

LanceDB's primary GUI tool for exploring data:
- **Lightweight web UI** for browsing Lance datasets, viewing schemas, and browsing table data
- **Vector visualization:** Compact sparkline/heatstrip previews for vector columns (spots anomalies visually)
- **CLIP embedding detection** and vector column highlighting
- **Schema analysis** with type detection
- **Server-side pagination** with column filtering
- **Read-only** browsing (no editing capability)
- **Docker-based deployment:** Mount a folder and go

```bash
docker pull ghcr.io/lancedb/lance-data-viewer:latest
docker run --rm -p 8080:8080 -v /path/to/data:/data:ro ghcr.io/lancedb/lance-data-viewer:latest
```

#### LanceDB Cloud Dashboard

Web-based management dashboard at cloud.lancedb.com/dashboard for managing projects, tables, and usage analytics.

#### YOLOExplorer (Computer Vision)

Specialized tool for CV datasets: SQL filters, vector similarity search, Pandas integration, and built-in visualization with `plot_similarity_index`.

#### New Relic Integration

Monitoring dashboard for LanceDB performance: latency, queries, documents retrieved, content relevance.

### 3.7 Data Visualization Platform

**LanceDB does not have a Kibana-equivalent dashboard.** The Lance Data Viewer provides basic data exploration, but there is no full-featured analytics/visualization platform comparable to OpenSearch Dashboards.

For visualization, LanceDB users typically rely on:
- **Lance Data Viewer** for basic browsing and vector sparklines
- **DuckDB** integration for SQL-based analysis
- **Pandas / Polars** for data manipulation
- **Apache Arrow** for in-memory analytics
- **Custom Jupyter notebooks** for embedding visualization (t-SNE, UMAP)
- **Streamlit / Gradio** for building custom search UIs

### 3.8 API and SDK Support

| Language | Package | Status |
|---|---|---|
| **Rust** | `lancedb` crate | Core SDK, v1.0.0 |
| **Python** | `lancedb` (PyPI) | v1.0.0, sync + async APIs |
| **JavaScript/TypeScript** | `@lancedb/lancedb` (npm) | Thin wrapper around Rust core |
| **Java** | Java SDK | v1.0.0 |
| **REST API** | HTTP endpoints | Available for LanceDB Cloud |

**Architecture:** All non-Rust SDKs are thin wrappers around the Rust core, ensuring consistent feature parity and performance across languages.

**Ecosystem Integrations:**
- LangChain, LlamaIndex (RAG frameworks)
- Apache Arrow, Pandas, Polars (data processing)
- DuckDB (SQL queries)
- Apache Spark, Ray, Trino, Apache Flink (distributed processing)
- Hugging Face Hub (dataset distribution)

### 3.9 Performance Benchmarks

- **1.5M IOPS** benchmark (announced January 2026)
- **~95% accuracy** with only a few milliseconds of latency for ANN search
- **Disk-based performance:** Approaches in-memory speeds via memory-mapped files and SIMD optimization
- **GIST1M benchmark** (1M 960-dimensional vectors): Compared favorably to FAISS in-memory HNSW, especially for datasets exceeding available memory
- **Filtered search weakness:** IVF-PQ with tight filters can spike to 200-300ms (similar to OpenSearch)
- **GPU-accelerated index building:** Supported with CUDA and Apple Silicon (MPS) for IVF/PQ training
- **Uber-scale multi-bucket storage:** Production-proven at billion scale

---

## 4. Comparative Summary

### Feature Matrix

| Feature | OpenSearch | FAISS | LanceDB |
|---|---|---|---|
| **Type** | Distributed search platform | Library | Embedded/serverless DB |
| **License** | Apache 2.0 | MIT | Apache 2.0 |
| **Primary Language** | Java | C++ | Rust |
| **Latest Version** | 3.5.0 (Mar 2026) | 1.14.1 (Mar 2026) | SDK 1.0.0 |
| **Persistence** | Built-in (Lucene segments) | Manual (save/load) | Built-in (Lance format) |
| **Distribution/Sharding** | Native (distributed by design) | None | File-based (S3-compatible) |
| **Concurrency** | Full ACID | None (single-process) | Concurrent reads; limited concurrent writes |
| **Metadata Filtering** | Full (hybrid search) | None | Full (SQL, scalar indices) |
| **Full-text Search** | Full (BM25, neural) | None | BM25-based |
| **Max Dimensions** | 16,000 | Unlimited (memory-bound) | Unlimited (memory-bound) |
| **GPU Support** | Experimental (v3.0, NVIDIA cuVS) | Native (CUDA, ROCm, cuVS) | Index building only (CUDA, MPS) |
| **Multimodal (Text)** | Native (Neural Search) | None (vectors only) | Native |
| **Multimodal (Images)** | Native (CLIP, Titan) | None | Native (first-class blobs) |
| **Multimodal (Audio)** | Via external APIs | None | Native (blob storage) |
| **Multimodal (Video)** | Via external APIs | None | Native (blob storage) |
| **GUI/Dashboard** | OpenSearch Dashboards (full) | None | Lance Data Viewer (basic) |
| **Kibana Equivalent** | OpenSearch Dashboards | None | None |
| **Cloud Offering** | AWS, Aiven, Instaclustr | None | LanceDB Cloud ($16.03/mo+) |
| **Min Cloud Cost** | ~$175/mo (Serverless) | N/A | ~$16.03/mo |

### Index Type Support

| Algorithm | OpenSearch | FAISS | LanceDB |
|---|---|---|---|
| **HNSW** | Yes (all engines) | Yes | Yes (within IVF partitions) |
| **IVF** | Yes (FAISS engine) | Yes | Yes |
| **Product Quantization** | Yes (via FAISS) | Yes | Yes |
| **Scalar Quantization** | Yes (via FAISS) | Yes | No |
| **RaBitQ** | No | Yes (v1.13+) | Yes |
| **CAGRA (GPU)** | Yes (v3.0, experimental) | Yes | No |
| **Binary indices** | Yes (v2.16+) | Yes | No |
| **NSG** | No | Yes | No |
| **LSH** | No | Yes | No |
| **Full-text (BM25)** | Yes | No | Yes |
| **R-Tree (geospatial)** | No | No | Yes |

### Best Use Cases

| Use Case | Best Choice | Rationale |
|---|---|---|
| **Hybrid text + vector search** | OpenSearch | Full-text search + vector search in one platform |
| **Raw vector search speed** | FAISS | Fastest in-memory search, GPU acceleration |
| **Multimodal AI applications** | LanceDB | Native blob storage, embedded deployment, multimodal-first design |
| **Enterprise log/analytics + vectors** | OpenSearch | Full observability stack + vector search |
| **Edge/embedded deployment** | LanceDB | No server needed, runs in-process |
| **Research/prototyping** | FAISS | Simple API, extensive index type options |
| **Billion-scale production** | OpenSearch or LanceDB | OpenSearch: distributed; LanceDB: S3-backed serverless |
| **Building a custom vector DB** | FAISS | Designed to be embedded as a library inside other systems |

### Key Takeaways for a New Multimodal Vector Database Project

1. **LanceDB is the strongest multimodal-native option:** Its Lance format treats blobs as first-class citizens, storing images, video, audio alongside embeddings and metadata in a single table. This is architecturally closest to what a multimodal vector database should look like.

2. **FAISS is the gold standard for raw vector search algorithms:** Its RaBitQ, HNSW, IVF-PQ, and CAGRA implementations set the performance bar. Any new vector database should consider using FAISS (or its algorithms) as the underlying search engine.

3. **OpenSearch demonstrates what a production-grade distributed vector platform looks like:** Its three-engine architecture (NMSLIB, FAISS, Lucene), hybrid search capabilities, and OpenSearch Dashboards visualization provide a reference implementation for enterprise features.

4. **All three struggle with filtered search:** This is a known industry-wide pain point. Engines relying on pre- or post-filter fallbacks see significant performance degradation under tight filters.

5. **GPU acceleration is the frontier:** FAISS has the most mature GPU support. OpenSearch's cuVS integration is experimental. LanceDB supports GPU for index building only. A new database should plan for first-class GPU support from the start.

6. **The storage format matters enormously:** LanceDB's Lance format (100x faster random access than Parquet, native blob support) shows that the right storage format can be a major differentiator for multimodal workloads.

---

## Sources

### OpenSearch
- [OpenSearch Vector Search Guide (Instaclustr)](https://www.instaclustr.com/education/opensearch/opensearch-vector-search-the-basics-and-a-quick-tutorial-2026-guide/)
- [k-NN Search in Amazon OpenSearch Service (AWS Docs)](https://docs.aws.amazon.com/opensearch-service/latest/developerguide/knn.html)
- [k-NN Vector Documentation](https://docs.opensearch.org/latest/mappings/supported-field-types/knn-vector/)
- [Vector Search Techniques](https://docs.opensearch.org/latest/vector-search/vector-search-techniques/index/)
- [GPU-Accelerated Vector Search Blog](https://opensearch.org/blog/gpu-accelerated-vector-search-opensearch-new-frontier/)
- [OpenSearch k-NN GitHub](https://github.com/opensearch-project/k-NN)
- [Amazon OpenSearch Service Pricing](https://aws.amazon.com/opensearch-service/pricing/)
- [OpenSearch Pricing Guide (BigData Boutique)](https://bigdataboutique.com/blog/opensearch-and-elasticsearch-pricing-guide)
- [OpenSearch Dashboards Platform](https://opensearch.org/platform/opensearch-dashboards/)
- [OpenSearch Dashboards Documentation](https://docs.opensearch.org/latest/dashboards/)
- [OpenSearch Language Clients](https://docs.opensearch.org/latest/clients/)
- [Spaces Documentation](https://docs.opensearch.org/latest/mappings/supported-field-types/knn-spaces/)
- [Multimodal Semantic Search Blog](https://opensearch.org/blog/multimodal-semantic-search/)
- [Multimodal Search Documentation](https://docs.opensearch.org/2.11/search-plugins/multimodal-search/)
- [OpenSearch 3.0 Release](https://cloudnativenow.com/features/opensearch-3-0-vector-search-gets-the-open-source-performance-boost-ai-applications-need/)
- [OpenSearch Version History](https://docs.opensearch.org/latest/version-history/)
- [Search Limitations (Sease)](https://sease.io/2025/07/limitations-and-workarounds-in-opensearch.html)
- [Uber Billion-Scale Vector Search](https://www.uber.com/us/en/blog/powering-billion-scale-vector-search-with-opensearch/)
- [Elasticsearch vs OpenSearch Vector Performance (Elastic)](https://www.elastic.co/search-labs/blog/elasticsearch-opensearch-vector-search-performance-comparison)
- [OpenSearch Dashboards vs Kibana (Opster)](https://opster.com/guides/opensearch/opensearch-basics/opensearch-dashboards-vs-kibana-comparison/)
- [OpenSearch Dashboards Visualizations (Instaclustr)](https://www.instaclustr.com/blog/how-to-use-opensearch-3-0-dashboards-creating-advanced-visualizations/)

### FAISS
- [FAISS GitHub Repository](https://github.com/facebookresearch/faiss)
- [FAISS Documentation](https://faiss.ai/index.html)
- [FAISS Changelog](https://github.com/facebookresearch/faiss/blob/main/CHANGELOG.md)
- [FAISS Meta AI Page](https://ai.meta.com/tools/faiss/)
- [FAISS Tutorial (Pinecone)](https://www.pinecone.io/learn/series/faiss/faiss-tutorial/)
- [FAISS Complete Guide (DataCamp)](https://www.datacamp.com/blog/faiss-facebook-ai-similarity-search)
- [FAISS Wikipedia](https://en.wikipedia.org/wiki/FAISS)
- [FAISS Limitations (Medium)](https://medium.com/@ssuparnataneja/unlocking-faiss-limitations-cutting-edge-solutions-and-tangible-illustrations-bf94021be873)
- [FAISS arxiv Paper](https://arxiv.org/abs/2401.08281)
- [Scaling FAISS to Billion-Scale (Medium)](https://medium.com/@deveshbajaj59/scaling-semantic-search-with-faiss-challenges-and-solutions-for-billion-scale-datasets-1cacb6f87f95)
- [faiss-gpu-cu12 PyPI](https://pypi.org/project/faiss-gpu-cu12/)

### LanceDB
- [LanceDB Homepage](https://www.lancedb.com/)
- [LanceDB Documentation](https://docs.lancedb.com/)
- [LanceDB GitHub](https://github.com/lancedb/lancedb)
- [Lance Format GitHub](https://github.com/lance-format/lance)
- [LanceDB Pricing](https://lancedb.com/pricing/)
- [LanceDB AWS Architecture Blog](https://aws.amazon.com/blogs/architecture/a-scalable-elastic-database-and-search-solution-for-1b-vectors-built-on-lancedb-and-amazon-s3/)
- [Lance Data Viewer Blog](https://lancedb.com/blog/lance-data-viewer/)
- [Lance Data Viewer GitHub](https://github.com/lance-format/lance-data-viewer)
- [LanceDB SDK Streamlining Blog](https://blog.lancedb.com/streamlining-our-sdks/)
- [LanceDB Client SDKs](https://docs.lancedb.com/api-reference)
- [LanceDB Technical Analysis (Oreate AI)](https://www.oreateai.com/blog/technical-analysis-and-practical-applications-of-vector-database-lancedb/892f4eb7f18a69710b4b3f5cde5546be)
- [LanceDB Newsletter January 2026](https://lancedb.com/blog/newsletter-january-2026/)
- [LanceDB Newsletter June 2025](https://lancedb.com/blog/newsletter-june-2025/)
- [Lance x Hugging Face Blog](https://lancedb.com/blog/lance-x-huggingface-a-new-era-of-sharing-multimodal-data/)
- [LanceDB TechCrunch Article](https://techcrunch.com/2024/05/15/lancedb-which-counts-midjourney-as-a-customer-is-building-databases-for-multimodal-ai/)
- [Lance Format 2.2 Blog](https://lancedb.com/blog/lance-file-format-2-2-taming-complex-data/)
- [2025 Roadmap (GitHub)](https://github.com/lance-format/lance/issues/3730)

### Comparative
- [FAISS vs LanceDB (Zilliz)](https://zilliz.com/comparison/faiss-vs-lancedb)
- [LanceDB vs OpenSearch (Zilliz)](https://zilliz.com/comparison/lancedb-vs-opensearch)
- [Vector Database Comparison 2026 (4xxi)](https://4xxi.com/articles/vector-database-comparison/)
- [Filtered Search Achilles Heel](https://yudhiesh.github.io/2025/05/09/the-achilles-heel-of-vector-search-filters.html)
- [Best Vector Databases 2025 (Firecrawl)](https://www.firecrawl.dev/blog/best-vector-databases)

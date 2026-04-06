# Vector Database Research: Milvus vs. Weaviate vs. Qdrant

*Research compiled April 2026 to inform the design of a new multimodal vector database.*

---

## Table of Contents

1. [Milvus (by Zilliz)](#1-milvus-by-zilliz)
2. [Weaviate](#2-weaviate)
3. [Qdrant](#3-qdrant)
4. [Comparative Summary](#4-comparative-summary)

---

## 1. Milvus (by Zilliz)

### 1.1 Overview

| Attribute | Detail |
|---|---|
| **Maintainer** | Zilliz Inc.; governed by the LF AI & Data Foundation |
| **License** | Apache License 2.0 |
| **Primary Language** | Go (coordination/control plane), C++ (search engine core) |
| **Latest Version** | v2.6.10 (March 2026); v2.5.27 (LTS security patch) |
| **GitHub Stars** | 43,000+ |
| **First Release** | 2019 (v1.0); v2.0 architecture rewrite January 2022 |

Milvus is a high-performance, cloud-native vector database designed for scalable ANN search at billion-vector scale. It is the largest open-source vector database by community size, used in production by NVIDIA, Salesforce, eBay, Airbnb, and DoorDash, among 10,000+ enterprise teams.

### 1.2 Architecture

Milvus uses a **fully disaggregated architecture** that separates compute, storage, and coordination into independent microservices:

**Four Layers:**

1. **Access Layer** -- Stateless proxies handle client connections, load balancing, authentication, and static/dynamic verification. Exposes both gRPC and REST APIs.

2. **Coordinator Service** -- Four coordinator types:
   - *Root Coordinator*: metadata, collection schemas, timestamps, ID allocation
   - *Data Coordinator*: data nodes and segment management
   - *Query Coordinator*: query node management and segment distribution
   - *Streaming Coordinator* (new in 2.6): streaming nodes and message channel assignments

3. **Worker Nodes** -- Fully stateless, independently scalable:
   - *Query Nodes*: execute vector/scalar search
   - *Data Nodes*: handle ingestion and compaction
   - *Index Nodes*: build vector/scalar indexes

4. **Storage**:
   - **Object Storage**: MinIO (default), AWS S3, Azure Blob -- stores snapshots, index files, intermediate results
   - **WAL**: Kafka, Pulsar, or **Woodpecker** (new cloud-native, zero-disk WAL that writes directly to object storage)
   - **Metadata Store**: etcd

**Deployment Modes:**
- **Milvus Lite**: Lightweight Python library for prototyping (embeddable, no server)
- **Milvus Standalone**: Single Docker container for development/small production
- **Milvus Distributed**: Full Kubernetes deployment for billion-scale workloads

**Supported Index Types (most comprehensive of the three):**

| Category | Index Types |
|---|---|
| **CPU Floating-Point** | FLAT, IVF_FLAT, IVF_PQ, IVF_SQ8, HNSW, HNSW_SQ, HNSW_PQ, HNSW_PRQ, SCANN, DiskANN |
| **GPU-Accelerated** | GPU_IVF_FLAT, GPU_IVF_PQ, GPU_CAGRA (NVIDIA), GPU_BRUTE_FORCE |
| **Binary Vectors** | BIN_FLAT, BIN_IVF_FLAT |
| **Sparse Vectors** | SPARSE_INVERTED_INDEX (with DAAT_WAND for BM25 full-text search) |

**Distance Metrics:** L2 (Euclidean), IP (Inner Product), Cosine, JACCARD, HAMMING.

**Storage Features:**
- Hot/cold tiered storage for cost optimization
- mmap support for memory-mapped indexes
- Segment-based storage with background compaction

### 1.3 Multimodal Support

Milvus does not natively run embedding models but provides strong infrastructure-level multimodal support:

- **Multi-vector fields per collection**: Store dense vectors, sparse vectors, and scalar metadata in the same collection. E.g., a product can have CLIP image embeddings + text encoder embeddings + BM25 sparse vectors simultaneously.
- **Unified Tensor/StructList data type** (v2.6+): New data type for multi-vector embedding structures, enabling compatibility with ColBERT, ColQwen, video frame embeddings, and multimodal vectors.
- **Built-in hybrid search**: Combines dense vector similarity, sparse vector (BM25/SPLADE), and metadata filtering in a single query with configurable reranking (RRF, weighted).
- **pymilvus[model] utility**: Wrapper code for transforming text/images/video into embeddings using OpenAI, HuggingFace, CLIP, etc.
- **Metadata/Payloads**: Supports rich scalar fields (INT, FLOAT, VARCHAR, BOOL, JSON, ARRAY, GEO), with scalar indexing for filtered search. New in 2.6: geometry fields, timestamptz, array of structs.

**Limitations**: No built-in vectorizer/embedding generation -- users must bring their own embeddings or use the pymilvus model utility. However, Zilliz Cloud Pipelines offer managed embedding generation.

### 1.4 Shortcomings

1. **Deployment Complexity**: Running distributed Milvus requires Kafka/Pulsar + MinIO + etcd + the Milvus components themselves. Up to 120 configuration parameters. Users report initial cluster setup taking hours.
2. **Steep Learning Curve**: G2 reviewers consistently cite complexity for new users, especially in distributed setups.
3. **Documentation Quality**: Some users find documentation less user-friendly compared to simpler alternatives like Chroma or Qdrant.
4. **Security Vulnerabilities**: Two critical CVEs in 2025-2026 -- CVE-2025-64513 (auth bypass via forged headers, CVSS 9.8) and CVE-2026-26190 (metrics port auth bypass, CVSS 9.8). Both patched.
5. **Version/Tooling Fragmentation**: Related tools (milvus-backup, birdwatch, helm charts, operator) version correlation is not always clear.
6. **Write-Heavy Stability**: Reports of standalone container becoming unhealthy under frequent periodic writes.
7. **Delayed Space Reclamation**: Deleted data is only logically deleted; actual space recovery depends on background compaction.
8. **gRPC Insert Size Limit**: Inserts capped at 1,024 MB per operation.
9. **Not Ideal for Small Projects**: Overkill for simple use cases; Chroma or pgvector are simpler alternatives for small-scale needs.

### 1.5 Paid Tier / Cloud Offering (Zilliz Cloud)

| Tier | Price | Details |
|---|---|---|
| **Free** | $0 | 5 GB storage, 2.5M vCUs/month; or 30-day trial of Serverless/Dedicated |
| **Serverless** | $4 per 1M vCUs | Pay-per-query, auto-scaling, no cluster management |
| **Standard (Dedicated)** | From ~$99/month | Managed essentials for non-critical workloads |
| **Business Critical** | Custom | Cross-region backups, HIPAA/SOC2/ISO 27001, for regulated industries |
| **BYOC** | Custom | Bring Your Own Cloud -- runs in customer's VPC |

**Key pricing details:**
- Storage: $0.04/GB/month (standardized Jan 2026 across AWS/Azure/GCP)
- Compute types: Performance-optimized (1.5M vectors/CU), Capacity-optimized (5M/CU), Tiered-storage (20M/CU)
- Annual commitment discounts available
- Claims 70-80% TCO savings vs. self-hosted Milvus
- SLA: 99.95% for dedicated; SOC 2 Type II, ISO 27001, GDPR, HIPAA-ready

### 1.6 GUI / Dashboard

**Attu** (open-source, Apache 2.0 up to v2.5.12):
- Visual schema designer -- design, view, modify collection schemas
- Full lifecycle control -- manage databases, collections, partitions, properties
- Smart data operations -- filters, syntax highlighting, inline editing, import/export
- **Vector search interface** -- perform similarity queries with advanced filters
- **Data visualization** -- explore vector relationships in a knowledge-graph-like structure; click to recursively explore neighbors
- **System monitoring** -- topology view, node metrics (CPU, memory, disk), performance tracking
- User management with RBAC
- Built-in AI assistant (requires OpenAI API key, v2.6.0+)
- Connects to both Milvus Standalone and Zilliz Cloud
- Access via web browser at `http://{host}:8000`

**Milvus WebUI** (built-in, for observability): Accessible at `:9091/webui`, provides system diagnostics and cluster health monitoring.

### 1.7 Data Visualization Platform

- **No Kibana-equivalent built-in**. Attu's vector visualization (knowledge-graph explorer, top-K result visualization) is the closest analog.
- **Prometheus + Grafana**: Milvus exposes Prometheus-compatible metrics. Pre-built Grafana dashboards available for monitoring latency, throughput, memory, storage.
- **DataDog integration**: Official integration for observability.
- **FiftyOne integration**: For visual exploration of image/video vector datasets.
- **VectorDBBench** (by Zilliz): Open-source benchmarking tool for vector databases.

### 1.8 API and SDK Support

| Language | SDK | Status |
|---|---|---|
| **Python** | pymilvus v2.6.10 | Official, async support, most feature-rich |
| **Java** | milvus-sdk-java | Official |
| **Go** | milvus-sdk-go | Official |
| **Node.js** | milvus-sdk-node | Official, gRPC + HTTP (for edge runtimes) |
| **C#** | milvus-sdk-csharp | Community (contributed by Microsoft) |

**Protocols:**
- **gRPC**: Primary protocol, highest performance, full feature parity
- **RESTful HTTP**: Near feature parity in SDK v2, easier to get started
- Both APIs share the same internal request handling after parsing

**SDK v2 Highlights** (2025): Unified `MilvusClient` class across all languages, native async support, consistent method naming. SDK v1/ORM deprecated; v1 support ends with Milvus 3.0.

### 1.9 Performance Benchmarks

- **Indexing speed**: Fastest among the three -- 6 minutes for 1M vectors (benchmark), vs. 8 minutes for Weaviate.
- **Query latency**: <10ms p50 on 768-dim text embeddings (Zilliz Cloud); 12ms p50 at 1,200 QPS in independent benchmark.
- **vs. Elasticsearch**: Milvus 2.5 BM25 full-text search returned results in 6ms vs. Elasticsearch's 200ms (30x improvement) on 1M vectors.
- **vs. FAISS/HNSWLib**: Claims 30-70% better performance than popular standalone libraries.
- **Production**: Sub-50ms retrieval latency across billions of vectors reported by enterprise users.
- **GPU acceleration**: NVIDIA CAGRA support delivers further speedups for GPU-equipped deployments.

---

## 2. Weaviate

### 2.1 Overview

| Attribute | Detail |
|---|---|
| **Maintainer** | Weaviate B.V. (Amsterdam, Netherlands) |
| **License** | BSD 3-Clause "New" or "Revised" License |
| **Primary Language** | Go |
| **Latest Version** | v1.35.16 (March 26, 2026) |
| **GitHub Stars** | ~16,000 |
| **First Release** | 2019 |

Weaviate is an open-source vector database that stores both objects and vectors, combining vector search with structured filtering and knowledge graph capabilities. It differentiates by offering built-in vectorization through model provider integrations and a strong focus on developer experience.

### 2.2 Architecture

Weaviate uses a **monolithic-with-sharding** architecture (not disaggregated like Milvus):

**Storage Structure:**
- Each **index** (collection/class) contains multiple **shards**
- Each shard contains: object storage, inverted index, and vector storage
- Shards can be distributed and/or replicated across cluster nodes
- Vector storage is independent of LSM segmentation

**Index Types:**

| Index Type | Description | Best For |
|---|---|---|
| **HNSW** (default) | Multi-layered graph for fast ANN search; stored in RAM | Large datasets needing speed + recall |
| **Flat** | Brute-force search, memory-efficient | Small datasets, multi-tenancy with few objects per tenant |
| **Dynamic** | Auto-switches from Flat to HNSW at threshold (default 10,000 objects) | Multi-tenant with variable tenant sizes |
| **HFresh** (new) | Cluster-based index using SPFresh algorithm with HNSW for centroid search | Large datasets needing memory efficiency; only cosine/L2 |

**Quantization/Compression Methods:**

| Method | Compression | Notes |
|---|---|---|
| **Rotational Quantization (RQ)** | 4x (8-bit) or 32x (1-bit) | Recommended default (v1.33+). No training needed, 98-99% recall |
| **Product Quantization (PQ)** | Configurable | HNSW only. More complex to tune |
| **Binary Quantization (BQ)** | 32x | No training step. Fast for flat indexes |
| **Scalar Quantization (SQ)** | 4x | HNSW only. Simpler than PQ |

**Distance Metrics:** Cosine, L2-squared, Dot product, Manhattan, Hamming.

**Key Architectural Features:**
- Inverted index with append-only writes (no bitmap rewrites) for high write throughput
- Bit-sliced indexing for range filtering -- fixed-operation-count range queries regardless of dataset size
- Native full-text search via BlockMax WAND (GA in 2025)
- Async and sync replication (GA in 2025)
- Multi-target vector search and MUVERA encoding

### 2.3 Multimodal Support

Weaviate has the **strongest native multimodal integration** of the three through built-in vectorizer modules:

**Built-in Multimodal Vectorizer Modules:**

| Module | Modalities | Model |
|---|---|---|
| **multi2vec-bind** | Text, image, audio, video, thermal, IMU, depth (7 modalities) | Meta ImageBind |
| **multi2vec-clip** | Text, image | OpenAI CLIP |
| **img2vec-neural** (legacy) | Image only | ResNet-50 |
| **Vertex AI multimodal** | Text, image, video | Google Vertex AI |
| **Gemini Embedding 2** | Text, image, audio, video, documents | Google Gemini |

**Cross-Modal Search:** With `multi2vec-bind`, you can query with audio (e.g., lion roaring sound) and retrieve images (lion photo). All modalities map to the same embedding space, enabling cross-modal retrieval.

**Multimodal RAG:** Retrieve images/audio/video from the vector database and pass them to a Large Multimodal Model for generation.

**At-Import Vectorization:** Weaviate can generate embeddings at import time using integrated vectorizers (OpenAI, Cohere, HuggingFace, Google, etc.), removing the need for external embedding pipelines.

**Metadata/Payloads:** Objects stored alongside vectors. Schema-based with strong typing. Supports cross-references between objects (knowledge graph). Properties can be filtered, aggregated, and grouped.

### 2.4 Shortcomings

1. **Scalability at Large Scale**: Performance at >100M vectors requires significant tuning. Reports of 5-15 second query times at 18M objects on 256 GB RAM / 128 core machines. Milvus and Qdrant handle pure vector scale better.
2. **Resource Intensity**: Needs more memory and compute than alternatives at very large scale. Memory usage increases heavily with data in multi-node Kubernetes clusters.
3. **Query Pagination Limit**: Default limit of offset+limit <= 10,000 objects. Increasing is "not recommended" due to memory explosion risk.
4. **Self-Hosting Complexity**: Managing, scaling, and maintaining Weaviate requires significant engineering time and Kubernetes expertise.
5. **Not a General-Purpose DB**: Cannot replace relational databases for structured data; requires managing an additional data store.
6. **RBAC/Integration Challenges**: Users report difficulties integrating with directory services and implementing proper RBAC.
7. **Single Batch Thread Per Client**: Python client supports only one batch import thread per client object, constraining parallel ingestion throughput.
8. **Graph Features Add Latency**: The knowledge graph layer, while powerful, can slow complex queries that combine vector search with relationship traversals.
9. **Pricing Unpredictability**: Usage-based cloud pricing hard to forecast; 14-day free sandbox is short.
10. **Migration/Upgrade Risk**: Historical HNSW index corruption bugs mean upgrades sometimes recommend fresh reimport rather than migration.
11. **Benchmark Performance**: Generally trails Milvus and Qdrant in pure vector search benchmarks; described as "improved the least" in recent Qdrant benchmarks.

### 2.5 Paid Tier / Cloud Offering (Weaviate Cloud)

| Tier | Price | Details |
|---|---|---|
| **Free Trial** | $0 (14 days) | Sandbox cluster with full core features; no permanent free tier |
| **Flex** | $45/month minimum | Pay-as-you-go, shared cloud, 99.5% SLA |
| **Plus** | $280/month | Annual commitment, more resources |
| **Premium** | $400/month | Dedicated deployment, 99.95% SLA, phone/Slack support |
| **Enterprise** | From ~$10,000/year | Annual contracts, dedicated environment, premium support |

**Pricing Dimensions (usage-based):**
- Vector Dimensions: $0.00975 - $0.01668 per million
- Storage: $0.2125 - $0.31875 per GiB
- Backup Storage: $0.022 - $0.0264 per GiB
- Rates vary by plan, cloud provider, and region

**Additional Cloud Services:**
- Weaviate Embeddings: Managed embedding inference service
- Weaviate Agents: Pre-built agentic services (Query Agent, etc.)
- HIPAA compliance available on cloud

**Self-Hosting Crossover**: Cloud is cheaper below ~5M vectors; self-hosting becomes cheaper above that if you already have Kubernetes.

### 2.6 GUI / Dashboard

**Weaviate Cloud Console** (official, cloud-only):
- Cluster management and monitoring
- GraphQL query editor with variables and headers panel
- Collection creation and management via GUI (Workbench)
- Query Agent access
- Data Import Tool

**Weaviate-UI** (third-party, open-source):
- Community Docker-based GUI (`naaive/weaviate-ui` on GitHub)
- Connects to self-hosted Weaviate instances
- GraphQL query interface with schema explorer

**No built-in GUI for self-hosted Weaviate** -- the official Console is cloud-only. Self-hosted users must use the third-party Weaviate-UI or interact via API/CLI.

### 2.7 Data Visualization Platform

- **No Kibana-equivalent built-in**.
- **Prometheus + Grafana**: Weaviate exposes Prometheus-compatible metrics at `<hostname>:2112/metrics`. Pre-configured Grafana dashboards available in the Weaviate examples repo for request latency, import speed, memory usage, vector vs. object storage time.
- **New Relic Integration**: Pre-built dashboard for monitoring vector searches, latency, query content, and document relevance.
- **No native vector space visualization** (unlike Qdrant's built-in t-SNE/UMAP viewer or Milvus Attu's knowledge graph explorer).

### 2.8 API and SDK Support

| Language | SDK | Status |
|---|---|---|
| **Python** | weaviate-client v4.20.4 | Official, sync + async (v4.7.0+) |
| **TypeScript/JavaScript** | v3 | Official (replaced JS client in 2023) |
| **Go** | v5 | Official |
| **Java** | v6 | Official (complete rewrite, idiomatic API) |
| **C#** | v1 | Official (first C# client) |

**Protocols:**
- **GraphQL**: Primary query interface; unique among vector databases
- **REST API**: Full CRUD, schema management, batch operations
- **gRPC**: Used internally by modern client libraries for search/query performance; automatically negotiated
- All clients support all API functions; modern clients auto-select gRPC where optimal

### 2.9 Performance Benchmarks

- **Query latency**: 15ms p50 at 800 QPS in one benchmark; higher latency (~20-50ms+) on 768-dim embeddings in other tests.
- **Index build**: ~8 minutes for 1M vectors in comparative benchmark.
- **Memory**: Higher memory consumption than alternatives at scale; efficient below 50M vectors.
- **Hybrid search**: BlockMax WAND BM25 implementation is competitive; strong combined vector+keyword retrieval.
- **Compression**: RQ compression (default since v1.33) can reduce cloud costs by 80%+ while maintaining 98-99% recall.
- **Docker pulls**: >1M monthly (highest among the three).
- **Noted weakness**: "Improved the least since the last benchmark run" per Qdrant's published benchmarks.

---

## 3. Qdrant

### 3.1 Overview

| Attribute | Detail |
|---|---|
| **Maintainer** | Qdrant Solutions GmbH (Berlin, Germany) |
| **License** | Apache License 2.0 |
| **Primary Language** | Rust |
| **Latest Version** | v1.17.1 (March 26, 2026) |
| **GitHub Stars** | ~24,000 |
| **First Release** | 2021 |

Qdrant is a vector similarity search engine and database built entirely in Rust. It emphasizes raw performance, low-overhead operations, ACID-compliant point-in-time consistency, and sophisticated metadata filtering.

### 3.2 Architecture

Qdrant uses a **single-binary, segment-based** architecture:

**Core Components:**
- **Collections**: Named sets of points sharing the same vector configuration, managed by `TableOfContent` (TOC) storage orchestrator
- **Shards**: Distribute collection data across nodes for horizontal scaling
- **Segments**: Fundamental storage units within shards; each maintains its own HNSW index and payload indexes
- **Gridstore**: Custom storage engine (replaced RocksDB entirely in v1.17)

**Index Types:**

Qdrant uses HNSW as its sole dense vector index (unlike Milvus which offers many). Differentiation comes through quantization:

| Quantization Method | Compression | Memory Savings | Speed Gain | Notes |
|---|---|---|---|---|
| **Scalar (SQ)** | 4x | 75% | Moderate | Default choice; float32 to uint8 |
| **Binary (BQ)** | 32x | ~97% | Up to 40x faster | Best with 1024+ dimensions |
| **Product (PQ)** | Configurable | High | Variable | Codebook trained automatically |
| **2-bit** (v1.15+) | 16x | ~94% | High | Better accuracy than BQ for smaller dims |
| **1.5-bit** (v1.15+) | 24x | ~96% | High | Intermediate between SQ and BQ |
| **Asymmetric** (v1.15+) | Varies | High | High | Binary storage + scalar queries |

**HNSW Key Parameters:** `m` (connections per node), `hnsw_ef` (search quality), `ef_construct` (build quality).

**HNSW Healing** (v1.15+): Reuses existing graph structure during optimization instead of full rebuilds.

**Distance Metrics:** Cosine, Dot Product, Euclidean, Manhattan.

**Key Architectural Features:**
- **SIMD hardware acceleration** (x86-64, ARM Neon)
- **io_uring async I/O** for maximizing disk throughput
- **Write-Ahead Logging** for data persistence guarantees
- **ACORN algorithm** (2025) for higher-quality filtered HNSW queries
- **Real-time indexing**: Vectors searchable immediately upon insertion
- **On-disk storage**: Vectors/payloads can be offloaded to disk with quantized indexes in RAM (oversampling + rescoring from disk)
- **Point-in-time consistency** and ACID-compliant operations

**Deployment Modes:**
- **Single node**: Docker or binary
- **Distributed cluster**: Sharding + replication on Kubernetes
- **Qdrant Edge**: On-device retrieval for low-latency applications
- **Hybrid Cloud**: Customer infrastructure + Qdrant management plane

### 3.3 Multimodal Support

Qdrant handles multimodal search through its **Named Vectors** feature and modality-agnostic architecture:

- **Named Vectors**: Store multiple named vector fields per point within a single collection. E.g., `{"text": [0.1, ...], "image": [0.3, ...]}` per data point. Each named vector can have its own dimensionality, distance metric, and HNSW configuration.
- **Multivector Representations**: Support for ColBERT/ColPali-style late interaction, where a single document is encoded as multiple vectors (token-level or segment-level). Used for advanced reranking and document retrieval.
- **Cross-Modal Search**: Using unified embedding models (ImageBind, CLIP, Gemini Embedding 2), text/image/audio/video can all be stored in the same vector space and cross-queried.
- **Two-Pass Pipeline**: Lower-dimensional embeddings for fast candidate retrieval, then full-dimensional rescoring -- enabled naturally by named vectors in a single collection.

**Metadata/Payloads**: Rich payload support with typed fields (keyword, integer, float, bool, text, datetime, geo, UUID). Payload indexing for filtered search. Supports nested JSON. Full-text filtering with native multilingual tokenization, stemming, and phrase matching (2025).

**Limitation**: No built-in vectorizer -- users must generate embeddings externally. However, Qdrant Cloud Inference (2025) provides managed embedding generation for unified embedding + search.

### 3.4 Shortcomings

1. **Horizontal Scaling Still Maturing**: Distributed features are evolving. Operational tooling for large-scale clustering is more limited than Milvus. Billion-scale is achievable but requires careful planning.
2. **Shard Management Complexity**: Shards cannot be split across nodes; shard count must be planned upfront. Recommendation: start with 12 shards for flexibility, but this can limit throughput on small clusters.
3. **Memory-Intensive by Default**: Vectors stored in RAM by default. Nodes can OOM if not properly configured. Without high availability, OOM causes downtime.
4. **No Global ID Uniqueness with Custom Sharding**: IDs only unique within a shard key. Multiple points with same ID possible across shard keys -- a known anti-pattern.
5. **Limited Built-in Visualization**: While the Web UI has basic t-SNE/UMAP visualization, users report wanting richer operations (batch delete, pattern-based collection management) from the UI.
6. **Default Config Not Production-Ready**: Out-of-box defaults frequently fail under production loads. WAL paths, memory limits, and index sizing need explicit tuning.
7. **Vertical Scaling Requires Downtime**: Without replication, vertical scaling (CPU/memory/disk) requires cluster restart.
8. **Disk Space Cannot Scale Down**: Cloud limitation from underlying providers.
9. **Single Index Algorithm**: Only HNSW for dense vectors (no IVF, no DiskANN, no GPU indexes). Quantization variants compensate, but less flexibility than Milvus.
10. **RocksDB Migration** (completed v1.17): Migration to Gridstore means direct upgrade from v1.15.x to v1.17.x is not possible.
11. **Relevance Degradation Under Complex Queries**: Qdrant's own 2025 recap acknowledged that relevance breaks down under complex queries at scale.
12. **Lower Throughput in Some Benchmarks**: Some 2025 benchmarks showed only 41 QPS for certain workloads (though highly configuration-dependent).

### 3.5 Paid Tier / Cloud Offering (Qdrant Cloud)

| Tier | Price | Details |
|---|---|---|
| **Free** | $0 (permanent) | 1 node: 0.5 vCPU, 1 GB RAM, 4 GB disk. No credit card required. Free inference with selected embedding models |
| **Standard** | ~$150-200/month (8GB/2vCPU) | Pay-as-you-go, hourly billing, 99.5% SLA |
| **Premium** | Custom (minimum spend) | 24/7 support, 99.9% SLA, SSO, private networking, customer-managed encryption keys |
| **Hybrid Cloud** | From $0.014/hour | Customer infrastructure + Qdrant management plane |
| **Enterprise/Private Cloud** | ~$2,000-5,000+/month | SOC2, private links, dedicated support, custom quote |

**Billing Model:**
- Charged for: vCPU, memory (GB), storage (GB), backup storage (GB), inference tokens
- Billed hourly, charged monthly
- Available on AWS, GCP, Azure Marketplaces
- Interactive pricing calculator on website

**Unique Offering: Qdrant Edge** -- Run retrieval directly on edge devices for low-latency, offline-capable search.

### 3.6 GUI / Dashboard

**Qdrant Web UI** (built-in, open-source):
- Accessible at `localhost:6333/dashboard` (self-hosted) or via cloud URL
- **Console**: Run REST API calls in-browser, test endpoints, inspect responses
- **Collections**: Create, view, manage collections; upload snapshots
- **Points Tab**: Search/filter points by ID or payload; view payloads and vectors
- **Visualize Tab**: Interactive 2D projection of vector space using t-SNE or UMAP (via DruidJS); see clusters, spot outliers
- **Graph Tab**: Explore HNSW graph interactively; follow nearest neighbors, see graph structure
- **Tutorial**: Interactive walkthrough with sample data
- **Datasets**: Bulk-load preconfigured public datasets

The Qdrant Web UI is self-described as "similar to Kibana for Elasticsearch" -- it is the most complete built-in visualization among the three databases.

**Third-Party: 1bench** -- Full-featured desktop GUI client for Qdrant with visual vector search, payload filtering, and multi-database support.

### 3.7 Data Visualization Platform

- **Built-in vector space visualization**: The Web UI's Visualize tab provides interactive t-SNE/UMAP 2D projections with Chart.js scatter plots. Dimensionality reduction runs in a web worker to avoid UI blocking.
- **HNSW graph explorer**: Interactive navigation of the HNSW graph structure from any point.
- **No Grafana/Prometheus by default**, but Qdrant Cloud includes built-in monitoring and observability dashboards.
- Qdrant explicitly positions its Web UI as "similar to Kibana for Elasticsearch."

### 3.8 API and SDK Support

| Language | SDK | Status |
|---|---|---|
| **Python** | qdrant-client v1.17.1 | Official, sync + async, gRPC + REST, local mode |
| **JavaScript/TypeScript** | @qdrant/js-client-rest | Official |
| **Rust** | qdrant-client (Rust) | Official |
| **Go** | go-client | Official |
| **.NET** | Qdrant.Client | Official |
| **Java** | qdrant-java-client (Maven) | Official |

**Six official client libraries** -- the most language coverage of the three.

**Protocols:**
- **REST API**: Primary interface, recommended for getting started; OpenAPI spec available
- **gRPC**: Higher performance, recommended for production; protobuf definitions available
- All clients transparently support both protocols; gRPC enabled with `prefer_grpc=True`

**Python Client Special Features:**
- Local mode: Run Qdrant in-memory without a server (for development/testing)
- Parallel upload with configurable batch sizes
- Full type definitions for all API methods

### 3.9 Performance Benchmarks

- **Query latency**: 8ms p50 at 1,500 QPS in comparative benchmark (fastest of the three).
- **Index build**: 4 minutes for 1M vectors using 3 GB memory (fastest of the three).
- **Memory efficiency**: 64x reduction possible with binary quantization; scalar quantization gives 4x with minimal accuracy loss.
- **Binary quantization**: Up to 40x faster searches with 32x memory reduction on high-dimensional vectors.
- **Filtered search**: Particularly strong; ACORN algorithm (2025) improves filtered HNSW quality.
- **HNSW healing** (v1.15+): Faster optimization by reusing existing graph structure.
- **MMR (Maximal Marginal Relevance)**: Native support for diversity-balanced retrieval.
- **Benchmark methodology note**: Qdrant publishes their own benchmarks at qdrant.tech/benchmarks comparing against Milvus, Weaviate, and Elasticsearch across multiple datasets and configurations.

---

## 4. Comparative Summary

### 4.1 Head-to-Head Comparison

| Dimension | Milvus | Weaviate | Qdrant |
|---|---|---|---|
| **Language** | Go + C++ | Go | Rust |
| **License** | Apache 2.0 | BSD 3-Clause | Apache 2.0 |
| **Latest Version** | v2.6.10 | v1.35.16 | v1.17.1 |
| **GitHub Stars** | 43,000+ | ~16,000 | ~24,000 |
| **Architecture** | Fully disaggregated microservices | Monolithic with sharding | Single binary, segment-based |
| **Deployment Complexity** | High (Kafka + MinIO + etcd + Milvus) | Medium (single binary or K8s) | Low (single Docker container) |
| **Max Proven Scale** | Billions of vectors | ~100M vectors comfortably | Hundreds of millions |
| **Index Types** | 15+ (HNSW, IVF, DiskANN, SCANN, GPU, binary, sparse) | 4 (HNSW, Flat, Dynamic, HFresh) | 1 (HNSW) + extensive quantization |
| **GPU Acceleration** | Yes (NVIDIA CAGRA, GPU_IVF) | No | No (planned) |
| **Sparse Vectors/BM25** | Yes (native) | Yes (BlockMax WAND) | Yes (named vectors) |
| **Built-in Vectorizers** | No (pymilvus[model] utility) | Yes (multi2vec-bind, CLIP, etc.) | No (Cloud Inference available) |
| **Multimodal Native** | Multi-vector fields + Tensor type | 7-modality ImageBind integration | Named vectors per modality |
| **Quantization** | PQ, SQ within IVF/HNSW variants | RQ, PQ, BQ, SQ | SQ, BQ, PQ, 2-bit, 1.5-bit, asymmetric |
| **Metadata Filtering** | During search, with scalar indexes | During search, bit-sliced indexing | During search, ACORN for HNSW |
| **Full-Text Search** | BM25 via sparse vectors | BlockMax WAND | Native tokenization + stemming |
| **Multi-Tenancy** | DB/collection/partition/partition key | Built-in, with dynamic indexes | Tiered multitenancy (2025) |
| **RBAC** | Yes | Yes (cluster RBAC, OIDC) | Yes (API keys, SSO in Premium) |
| **Hot/Cold Storage** | Yes (native tiered storage) | Compression-based cost savings | On-disk vectors with RAM quantized indexes |

### 4.2 SDK & API Comparison

| Feature | Milvus | Weaviate | Qdrant |
|---|---|---|---|
| **Official SDKs** | 4 (Python, Java, Go, Node.js) + C# community | 5 (Python, TS/JS, Go, Java, C#) | 6 (Python, TS/JS, Rust, Go, .NET, Java) |
| **REST API** | Yes | Yes | Yes |
| **gRPC API** | Yes (primary) | Yes (auto-negotiated) | Yes |
| **GraphQL** | No | Yes (primary query interface) | No |
| **Async Support** | Yes (SDK v2) | Yes (Python v4.7+) | Yes |
| **Local/Embedded Mode** | Milvus Lite (Python) | No | qdrant-client local mode |

### 4.3 Pricing Comparison

| Feature | Milvus (Zilliz Cloud) | Weaviate Cloud | Qdrant Cloud |
|---|---|---|---|
| **Permanent Free Tier** | Yes (5GB, 2.5M vCUs) | No (14-day trial only) | Yes (1GB RAM, 4GB disk) |
| **Minimum Paid** | ~$99/month (Dedicated) or $4/M vCUs (Serverless) | $45/month (Flex) | ~$150-200/month (Standard) |
| **Enterprise** | Custom (BYOC available) | From ~$10K/year | ~$2K-5K+/month |
| **Top SLA** | 99.95% | 99.95% | 99.9% |

### 4.4 GUI & Visualization Comparison

| Feature | Milvus (Attu) | Weaviate | Qdrant |
|---|---|---|---|
| **Built-in Self-Hosted GUI** | Yes (Attu, separate container) | No (cloud console only; third-party for self-hosted) | Yes (built into server at :6333/dashboard) |
| **Vector Space Visualization** | Knowledge-graph explorer, neighbor exploration | None built-in | t-SNE/UMAP 2D projections |
| **HNSW Graph Explorer** | No | No | Yes |
| **Query Interface** | Visual search with filters | GraphQL editor | REST console |
| **System Monitoring** | Topology, node metrics, resource usage | Cloud dashboard | Cloud monitoring |
| **Prometheus/Grafana** | Yes | Yes | Cloud only |
| **Kibana Equivalent** | Attu (partial) | None | Web UI (self-described) |

### 4.5 Performance Summary

| Metric | Milvus | Weaviate | Qdrant |
|---|---|---|---|
| **p50 Latency (1M, 768d)** | ~12ms | ~15ms | ~8ms |
| **Throughput (QPS)** | ~1,200 | ~800 | ~1,500 |
| **Index Build (1M vectors)** | ~6 min | ~8 min | ~4 min |
| **Memory (1M vectors)** | ~5 GB | ~4 GB | ~3 GB |
| **Best At** | Billion-scale, GPU workloads | Hybrid search, <50M vectors | Filtered search, low latency |

### 4.6 Key Takeaways for Designing a New Multimodal Vector Database

1. **Architecture**: Milvus proves that disaggregated storage/compute scales to billions but adds operational complexity. Qdrant shows that a simpler architecture with Rust can deliver excellent single-node performance. A new database should consider a middle ground -- simple deployment with optional disaggregation.

2. **Multimodal**: Weaviate's built-in vectorizer approach (multi2vec-bind with 7 modalities) is the most developer-friendly for multimodal, but couples the database to specific models. Milvus's multi-vector fields and Qdrant's named vectors are more flexible but require external embedding pipelines. A new database could offer both: built-in vectorizers as a convenience layer over a flexible multi-vector storage core.

3. **Index Diversity**: Milvus's 15+ index types (including GPU, DiskANN, IVF) give users the most tuning options. Qdrant compensates with advanced quantization (1.5-bit, 2-bit, asymmetric). A new database should support at least HNSW, IVF, and disk-based indexes, with first-class quantization.

4. **Filtering**: All three integrate filtering into the search process (not post-filtering), but Qdrant's ACORN algorithm and Weaviate's bit-sliced indexing show the state of the art. This is table stakes for a modern vector database.

5. **Developer Experience**: Qdrant's built-in Web UI with vector visualization, HNSW graph exploration, and tutorials is the gold standard. Milvus's Attu is powerful but separate. Weaviate's self-hosted GUI story is weakest. A new database should ship with a built-in dashboard.

6. **Shortcomings to Avoid**: All three struggle with some combination of deployment complexity (Milvus), scaling limitations (Weaviate/Qdrant), and production-readiness of defaults (Qdrant). A new database should prioritize sensible defaults, simple deployment, and transparent scalability.

7. **Edge/Hybrid**: Qdrant Edge for on-device search is unique. Milvus Lite for embedded use is practical. These deployment flexibility patterns are increasingly important.

---

## Sources

### Milvus
- [Milvus GitHub Repository](https://github.com/milvus-io/milvus)
- [Milvus Architecture Overview](https://milvus.io/docs/architecture_overview.md)
- [Milvus Documentation](https://milvus.io/docs)
- [Milvus Roadmap](https://milvus.io/docs/roadmap.md)
- [What is Milvus](https://milvus.io/docs/overview.md)
- [Milvus In-Memory Index Types](https://milvus.io/docs/index.md)
- [Milvus Limits](https://milvus.io/docs/limitations.md)
- [Milvus SDK v2 Announcement](https://milvus.io/blog/introducing-milvus-sdk-v2-native-async-support-unified-apis-and-superior-performance.md)
- [Zilliz Cloud Pricing](https://zilliz.com/pricing)
- [Zilliz Cloud October 2025 Update](https://zilliz.com/blog/zilliz-cloud-oct-2025-update)
- [Attu GUI for Milvus](https://zilliz.com/attu)
- [Attu GitHub](https://github.com/zilliztech/attu)
- [Milvus WebUI](https://milvus.io/docs/milvus-webui.md)
- [Milvus Wikipedia](https://en.wikipedia.org/wiki/Milvus_(vector_database))
- [Milvus 40K Stars Announcement](https://www.manilatimes.net/2025/12/19/tmt-newswire/pr-newswire/milvus-surpasses-40000-github-stars-reinforcing-leadership-in-open-source-vector-databases/2246697)
- [Milvus Security Vulnerability Report](https://cyberpress.org/milvus-proxy-vulnerability/)
- [Milvus G2 Reviews](https://www.g2.com/products/milvus/reviews)
- [VectorDBBench](https://github.com/zilliztech/VectorDBBench)

### Weaviate
- [Weaviate GitHub Repository](https://github.com/weaviate/weaviate)
- [Weaviate Documentation](https://docs.weaviate.io/weaviate)
- [Weaviate in 2025](https://weaviate.io/blog/weaviate-in-2025)
- [Weaviate Pricing](https://weaviate.io/pricing)
- [Weaviate Cloud Pricing Update](https://weaviate.io/blog/weaviate-cloud-pricing-update)
- [Weaviate Vector Indexing](https://docs.weaviate.io/weaviate/concepts/vector-index)
- [Weaviate Vector Quantization](https://docs.weaviate.io/weaviate/concepts/vector-quantization)
- [Weaviate Client Libraries](https://docs.weaviate.io/weaviate/client-libraries)
- [Weaviate Multimodal Embeddings (ImageBind)](https://weaviate.io/developers/weaviate/model-providers/imagebind/embeddings-multimodal)
- [Weaviate Multimodal Guide](https://weaviate.io/blog/multimodal-guide)
- [Weaviate Workbench](https://weaviate.io/workbench)
- [Weaviate Monitoring](https://weaviate.io/developers/weaviate/configuration/monitoring)
- [Weaviate Release Notes](https://docs.weaviate.io/weaviate/release-notes)
- [Weaviate Scaling Blog](https://weaviate.io/blog/scaling-and-weaviate)
- [Weaviate Performance Issue (GitHub)](https://github.com/weaviate/weaviate/issues/4572)
- [Weaviate Support Plans](https://weaviate.io/support-plans)

### Qdrant
- [Qdrant GitHub Repository](https://github.com/qdrant/qdrant)
- [Qdrant Documentation Overview](https://qdrant.tech/documentation/overview/)
- [Qdrant Vector Database](https://qdrant.tech/qdrant-vector-database/)
- [Qdrant 2025 Recap](https://qdrant.tech/blog/2025-recap/)
- [Qdrant Benchmarks](https://qdrant.tech/benchmarks/)
- [Qdrant Cloud Pricing](https://qdrant.tech/pricing/)
- [Qdrant Web UI Documentation](https://qdrant.tech/documentation/web-ui/)
- [Qdrant Web UI GitHub](https://github.com/qdrant/qdrant-web-ui)
- [Qdrant API & SDKs](https://qdrant.tech/documentation/interfaces/)
- [Qdrant Binary Quantization](https://qdrant.tech/articles/binary-quantization/)
- [Qdrant What is Vector Quantization](https://qdrant.tech/articles/what-is-vector-quantization/)
- [Qdrant v1.15 Release (Quantization)](https://qdrant.tech/blog/qdrant-1.15.x/)
- [Qdrant v1.16 Release (Tiered Multitenancy)](https://qdrant.tech/blog/qdrant-1.16.x/)
- [Qdrant Cloud Premium Tier](https://qdrant.tech/documentation/cloud-premium/)
- [Qdrant Capacity Planning](https://qdrant.tech/documentation/guides/capacity-planning/)
- [Qdrant Distributed Deployment](https://qdrant.tech/documentation/guides/distributed_deployment/)
- [Qdrant Gemini Embedding 2 Integration](https://qdrant.tech/blog/qdrant-gemini-embedding-2/)
- [Qdrant Multimodal Search Documentation](https://qdrant.tech/documentation/multimodal-search/)
- [Qdrant ColPali Integration](https://qdrant.tech/blog/qdrant-colpali/)

### Comparison Sources
- [Vector Database Comparison 2026 (Firecrawl)](https://www.firecrawl.dev/blog/best-vector-databases)
- [Vector Database Comparison 2026 (Jishu Labs)](https://jishulabs.com/blog/vector-database-comparison-2026)
- [Top 5 Open Source Vector Databases 2025 (Medium)](https://medium.com/@fendylike/top-5-open-source-vector-search-engines-a-comprehensive-comparison-guide-for-2025-e10110b47aa3)
- [Vector Database Comparison (TensorBlue)](https://tensorblue.com/blog/vector-database-comparison-pinecone-weaviate-qdrant-milvus-2025)
- [Vector Database Comparison (LiquidMetal AI)](https://liquidmetal.ai/casesAndBlogs/vector-comparison/)
- [Top 7 Vector Databases 2026 (DataCamp)](https://www.datacamp.com/blog/the-top-5-vector-databases)
- [10 Reproducible Benchmarks (Medium)](https://medium.com/@Nexumo_/10-reproducible-benchmarks-for-milvus-qdrant-weaviate-02723160b89d)

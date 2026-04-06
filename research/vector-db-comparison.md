# Vector Database Research: ChromaDB vs. Pinecone vs. pgvector

*Research compiled April 2026*

---

## Table of Contents

1. [ChromaDB](#1-chromadb)
2. [Pinecone](#2-pinecone)
3. [pgvector](#3-pgvector)
4. [Side-by-Side Comparison Table](#4-side-by-side-comparison-table)
5. [Recommendations for a Multimodal Vector Database Project](#5-recommendations)
6. [Sources](#sources)

---

## 1. ChromaDB

### 1.1 Overview

| Attribute | Detail |
|---|---|
| **What it is** | Open-source vector database / "AI-native data infrastructure" designed for storing and retrieving embeddings |
| **Maintainer** | Chroma, Inc. (founded by Jeff Huber and Anton Troynikov, San Francisco). Raised $18M seed in April 2023 |
| **License** | Apache 2.0 (same codebase powers open-source and Chroma Cloud -- no open-core split) |
| **Primary languages** | Rust (execution engine) + Python (API layer, client). The 2025 Rust-core rewrite delivered 4x faster writes/queries |
| **Latest version** | v1.5.5 (March 10, 2026) |
| **GitHub stars** | 26,000+; used in 90,000+ open-source codebases; 11M+ monthly downloads |

### 1.2 Architecture

**Storage layer:**
- Combines SQLite for metadata with a custom binary format for vectors
- Schema-less design -- collections need no predefined schema
- Extremely portable: copy one directory to migrate the entire database

**Execution engine:**
- Push-based, morsel-driven execution engine written in Rust
- In-memory by default; persistent storage (SQLite or PostgreSQL backend) requires configuration

**Deployment modes:**
- **Embedded** -- runs in-process inside your Python application, zero network latency
- **Client/Server** -- standalone server mode, deployable via Docker
- **Chroma Cloud** -- fully managed serverless offering on AWS, GCP, Azure

**Index types:**
- **HNSW only** -- no IVF, DiskANN, or GPU-accelerated indexing
- Similarity metrics: cosine similarity, Euclidean distance, inner product

**Search capabilities (as of v1.5.x):**
- Dense vector search
- Full-text search (BM25)
- Sparse vector search (SPLADE)
- Regex search
- Metadata filtering
- Hybrid search (combining multiple of the above)

**Scaling characteristics:**
- Single-server architecture. Manual multi-instance setups possible but no built-in clustering or automatic failover
- Works well up to a few million vectors with moderate query volume
- Beyond 10M vectors or high-throughput scenarios, scalability becomes a concern

### 1.3 Multimodal Support

| Capability | Status |
|---|---|
| Text embeddings | Fully supported |
| Image embeddings | Natively supported via built-in OpenCLIP embedding function |
| Audio/Video embeddings | Not natively supported. You must generate embeddings externally and store as pre-computed vectors |
| Raw file storage | ChromaDB does NOT store raw multimodal files; it stores URLs. A data loader (e.g., `ImageLoader`) accesses the files from external storage |
| Built-in embedding models | OpenCLIP (text + image). Native integrations with OpenAI, HuggingFace, Google, Cohere models |
| Multimodal API availability | **Python only** -- JS/TS and other SDKs lack multimodal capabilities |
| Modality per entry | One modality per entry ID. Cannot attach both an image and text to the same record |

**Metadata/Payload handling:**
- Arbitrary key-value metadata attached to each embedding
- Metadata filtering in queries (equality, inequality, `$in`, `$nin`, `$contains`, array filters as of v1.5.0)
- Documents (raw text) can be stored alongside embeddings
- No documented hard limit on metadata size per record

### 1.4 Shortcomings

1. **Single index type (HNSW only)** -- no IVF, DiskANN, GPU-accelerated search, or quantized index options
2. **Single-server architecture** -- no built-in horizontal scaling, clustering, or automatic failover
3. **Limited production readiness** -- no built-in RBAC, audit logging, or multi-tenancy; requires application-layer implementation
4. **Security gaps** -- lacks authentication by default; exposed instances are publicly readable/writable
5. **Multimodal limitations** -- only images as a native non-text modality; Python-only multimodal API; one modality per entry
6. **Performance ceiling** -- recommended only for small-to-medium datasets (under ~10M vectors)
7. **Limited tuning knobs** -- fewer performance optimization levers compared to Milvus, Qdrant, or Weaviate
8. **In-memory default** -- persistent storage requires extra configuration
9. **Smaller ecosystem** -- fewer third-party integrations and tools than Pinecone or Milvus

### 1.5 Paid Tier / Cloud Offering

**Chroma Cloud pricing (April 2026):**

| Plan | Base Price | Included Credits | Databases | Team Members | Key Features |
|---|---|---|---|---|---|
| **Starter** | $0/month + usage | $5 free credits | 10 | 10 | Community Slack |
| **Team** | $250/month + usage | $100 credits (non-rolling) | 100 | 30 | Slack support, SOC 2 Type II, volume discounts |
| **Enterprise** | Custom | Custom | Unlimited | Unlimited | Dedicated support, single-tenant clusters, BYOC, SLAs |

**Usage-based rates:**

| Service | Rate |
|---|---|
| Write | $2.50 / GiB |
| Storage | $0.33 / GiB / month |
| Query | $0.0075 / TiB |
| Network egress | $0.09 / GiB returned |
| Fork calls | $0.03 / call |
| Sync processing | $0.04 / GiB |
| Document extraction | $0.01 / page |
| Web scraping | $0.01 / webpage |

**Example:** ~1M-6M documents with 10M queries/month estimated at ~$79/month.

**Self-hosted:** Completely free under Apache 2.0. Third-party managed hosting (e.g., Elestio) starts at ~$14/month.

### 1.6 GUI / Dashboard

ChromaDB has **no built-in GUI**. All management is via API or CLI. However, a rich ecosystem of community-built tools exists:

| Tool | Type | Description |
|---|---|---|
| **Chroma Flow Studio** | Python GUI (Flask) | Full management dashboard with collection visualization, data export, embedding generation |
| **ChromaDash** | Web UI (HTML/JS) | Lightweight HTTP client for CRUD operations on collections/documents |
| **Chroma UI** (thakkaryash94) | Web app (Vercel) | Connect to and manage remote ChromaDB instances |
| **Chromagraphic** | Desktop GUI (Eel/Python) | GUI for viewing collections with OpenAI embedding support |
| **ChromaDB Viewer** | Streamlit app | Visualize, add, update, delete items in collections |
| **ChromaDB Manager** | Chrome extension | Browser-based data query tool for local ChromaDB instances |

### 1.7 Data Visualization Platform

There is **no Kibana equivalent** for ChromaDB. Options for exploring/visualizing vector data:

- **New Relic ChromaDB Quickstart** -- production monitoring dashboard with interactive visualizations for search latency, usage, errors, and retrieval metrics
- **Community GUI tools** (above) provide basic collection browsing and data point visualization
- For embedding cluster visualization, users typically export vectors and use external tools (e.g., Python + matplotlib with PCA/t-SNE, Nomic Atlas)

### 1.8 API and SDK Support

| Language | Status | Notes |
|---|---|---|
| **Python** | First-party (official) | `pip install chromadb` -- full feature parity |
| **JavaScript/TypeScript** | First-party (official) | `npm install chromadb` -- V3 client rewrite in 2025 |
| **Go** | Official client | Local/persistent client examples added March 2026 |
| **Rust** | Official client | Keyword search docs with Rust examples (Feb 2026) |
| **Ruby** | Community/ecosystem | Available via OpenAPI generator |
| **Java** | Community/ecosystem | Available via OpenAPI generator |
| **PHP** | Community/ecosystem | Available via OpenAPI generator |
| **C#** | Community/ecosystem | Available via OpenAPI generator |
| **Elixir** | Community/ecosystem | Listed as supported |

**Protocol:** HTTP REST API. OpenAPI spec available for generating clients in any language.

**Integrations:** LangChain, LlamaIndex, Braintrust, Streamlit, HuggingFace, OpenAI, Google, Cohere.

### 1.9 Performance Benchmarks

| Metric | Value | Conditions |
|---|---|---|
| Median search latency (p50) | ~20ms | 100K vectors, 384 dimensions |
| Benchmark snapshot | p50: 3ms, p99: 10ms, Recall@10: 96% | datastores.ai comparison |
| Insertion rate | ~25,000 vectors/sec | 1M vectors, 1536 dimensions |
| Query rate | ~2,000 queries/sec | 1M vectors, 1536 dimensions |
| Filtered query rate | ~1,000 queries/sec | 1M vectors, 1536 dimensions |

The 2025 Rust rewrite delivered 4x faster writes and queries vs. the prior Python implementation. ChromaDB is fastest for prototyping but lags behind Pinecone, Qdrant, and pgvector+pgvectorscale in raw throughput at scale.

---

## 2. Pinecone

### 2.1 Overview

| Attribute | Detail |
|---|---|
| **What it is** | Fully managed, cloud-native vector database purpose-built for AI/ML similarity search at scale |
| **Maintainer** | Pinecone Systems, Inc. (founded 2019 by Edo Liberty, former head of Amazon AI Labs) |
| **License** | Proprietary / closed-source. No self-hosted option (except BYOC which is still Pinecone-managed) |
| **Primary language** | Rust (engine), with managed infrastructure |
| **Latest SDK versions** | Python SDK v8 (April 2026), Java SDK v6.0.0, Go SDK v5, .NET and Rust clients also available |

### 2.2 Architecture

**Core design:**
- Proprietary, distributed architecture optimized for cloud deployment
- Automatic sharding, replication, and scaling without user intervention
- Storage-compute separation

**Deployment models:**
- **Serverless indexes** -- auto-scaling infrastructure, pay-per-use (default for new customers)
- **Pod-based indexes** -- dedicated resources (p1, p2, s1 pods) with predictable performance
- **BYOC** -- data plane runs in your AWS, GCP, or Azure account with zero-access operating model

**Indexing:**
- Proprietary algorithms -- users do NOT choose index types (unlike Milvus with 11+ options)
- Specify similarity metric at index creation: cosine, Euclidean, or dot product
- Pinecone handles all optimization internally

**Search capabilities:**
- Dense vector search
- Sparse vector search (keyword/BM25)
- Hybrid search (dense + sparse)
- Integrated reranking
- Metadata filtering (MongoDB-style query language)
- Real-time indexing

**Limits:**
- Max dense vector dimensions: 20,000
- Max sparse vector non-zero values: 1,000
- Max metadata per record: 40 KB
- Max upsert size: 2 MB or 1,000 records
- Max top_k: 10,000
- Result size limit per query: 4 MB
- Starter plan: 100,000 records, 100 namespaces

### 2.3 Multimodal Support

| Capability | Status |
|---|---|
| Text embeddings | Fully supported |
| Image embeddings | Supported via external models (CLIP, Gemini Embedding 2, etc.) |
| Audio/Video embeddings | Supported via external models -- store pre-computed vectors |
| Raw file storage | NOT stored in Pinecone. Raw media files kept in S3, GCS, CDN etc. Metadata field links back to source |
| Integrated embedding | Pinecone can generate embeddings automatically for text via integrated inference |
| Cross-modal search | Fully supported if embeddings share a vector space (e.g., CLIP text-to-image) |

**Metadata/Payload handling:**
- JSON key-value metadata per vector (max 40 KB)
- Supported types: strings, numbers, booleans, lists of strings
- No null values allowed (remove the key instead)
- Metadata filtering during search (pre-filter or post-filter depending on architecture)
- Fetch-by-metadata now supports pagination (2026 addition)

**Key distinction:** Pinecone is modality-agnostic at the storage layer. It stores vectors regardless of what they represent. Multimodal capability depends entirely on the embedding model used upstream.

### 2.4 Shortcomings

1. **Closed-source / vendor lock-in** -- no self-hosted option; can't inspect the engine, run locally for free, or avoid lock-in
2. **Unpredictable costs** -- teams report surprise bills up to $5,000; costs scale with data volume in non-obvious ways (read units, write units, storage, egress)
3. **No custom indexing** -- cannot choose between HNSW, IVF, DiskANN, SCANN etc.; Pinecone abstracts it entirely
4. **No SQL/relational features** -- no joins, aggregations, or transactional semantics; must run alongside a relational DB
5. **Metadata limitations** -- 40 KB cap per record; forces a two-step architecture (vector search in Pinecone, then data fetch from main DB)
6. **Network latency bottleneck** -- being cloud-only means every query incurs network RTT; the bottleneck is often the network, not the search
7. **Security concerns** -- IronCore Labs gave Pinecone a "weak" security maturity score in 2024; RBAC and CMEK improvements have since been added
8. **Regional restrictions** -- trial/Starter plan limited to US regions, which creates compliance issues for regulated industries
9. **Limited advanced filtering** -- competitors like Qdrant offer "payload-aware" HNSW filtering that can outperform Pinecone's approach on complex hybrid queries

### 2.5 Paid Tier / Cloud Offering

Everything is cloud-based. There is no free self-hosted version.

| Plan | Base Price | Included Resources | Key Features |
|---|---|---|---|
| **Starter** | Free | 2 GB storage, 2M write units/month, 5 indexes, 100K records | US regions only |
| **Standard** | From $50/month | Multi-cloud (AWS, Azure, GCP) | 21-day trial with $300 credits |
| **Enterprise** | From $500/month | Higher limits | 99.95% uptime SLA, private networking |
| **BYOC** | Custom | Data plane in your cloud account | Zero-access model, SOC 2, HIPAA |

**Usage-based overage rates:**

| Resource | Rate |
|---|---|
| Write units | $4 / million |
| Read units | $16 / million |
| Storage | $0.33 / GB / month |
| Egress | ~$0.08 / GB |

**Cost example:** 100M vectors (1024-dim), 150M queries, 10M writes/month = approximately $5,000-$6,000/month.

### 2.6 GUI / Dashboard

Pinecone includes a **built-in web console** at [app.pinecone.io](https://app.pinecone.io):

| Feature | Description |
|---|---|
| **Index management** | Create, configure, delete indexes; choose serverless vs. pod-based |
| **Collections & Indexes tabs** | View status, size, dimensions, metrics for all resources |
| **Explorer / Query Builder** | Visualize data and construct queries via GUI |
| **API key management** | Create and rotate API keys |
| **Billing & usage** | Monitor consumption and costs |
| **Playground** | Chat interface using Pinecone Assistants (supports Claude Sonnet 4.5, GPT-5) |
| **Sample data loading** | One-click sample datasets for exploration |

**Third-party GUI:**
- **Pinecone Explorer** -- native macOS desktop GUI for browsing indexes, namespaces, vectors with search/reranking/debugging (launched January 2026, featured on Hacker News)

### 2.7 Data Visualization Platform

There is **no built-in Kibana equivalent**. Options:

- **Nomic Atlas** -- export embeddings from Pinecone and visualize clusters, hover over points to see text, debug search results. The closest thing to a "Kibana for vectors"
- **Datadog integration** -- production monitoring dashboard for anomaly detection, latency visualization, usage trends. Supports both pod-based and serverless metrics
- **Pinecone Console Explorer** -- basic built-in data browsing in the web console
- No native scatter plot, t-SNE, or UMAP visualization built into Pinecone

### 2.8 API and SDK Support

| Language | Status | Latest Version |
|---|---|---|
| **Python** | Official | v8 (April 2026). Requires Python 3.10+. gRPC + async support |
| **TypeScript/Node.js** | Official | Active development |
| **Java** | Official | v6.0.0. Supports Kotlin, Groovy, Scala |
| **Go** | Official | v5. Also powers the Pinecone CLI |
| **.NET (C#)** | Official | Supports .NET Standard, .NET Core, .NET Framework |
| **Rust** | Official | `raker` client, last updated April 2026 |
| **CLI** | Official | Written in Go |

**Protocol:** REST API + gRPC. All SDKs pinned to specific API versions.

**Integrations:** LangChain, LlamaIndex, OpenAI, Cohere, HuggingFace, Vercel AI SDK, AWS ecosystem, MCP server for AI agents. 50+ total integrations.

### 2.9 Performance Benchmarks

| Metric | Value | Conditions |
|---|---|---|
| Typical query latency | 20-100ms | General workloads |
| p99 latency (p2 pods) | ~30ms | Pod-based indexes |
| Serverless latency | 50-100ms | Normal conditions |
| Customer report | 600 QPS sustained | 135M vectors, P50: 45ms, P99: 96ms |
| Dense vector benchmark | p99: 33ms, p50: 16ms, p90: 21ms | 10M vectors (Aloa 2026) |
| Insertion rate | ~50,000 vectors/sec | 1M vectors, 1536 dimensions |
| Query rate | ~5,000 queries/sec | 1M vectors, 1536 dimensions |
| Filtered query rate | ~4,000 queries/sec | 1M vectors, 1536 dimensions |

**vs. pgvector+pgvectorscale (50M vectors):** Pinecone's storage-optimized index (s1) showed 28x higher p95 latency and 16x lower throughput than pgvector+pgvectorscale at 99% recall. Pinecone's performance-optimized index (p2) showed 1.4x higher p95 latency and 1.5x lower throughput at 90% recall.

---

## 3. pgvector

### 3.1 Overview

| Attribute | Detail |
|---|---|
| **What it is** | Open-source PostgreSQL extension that adds vector data types and similarity search to any PostgreSQL database |
| **Maintainer** | Andrew Kane (independent developer). Community contributions via GitHub |
| **License** | PostgreSQL License (permissive, BSD-like). Free for commercial and non-commercial use |
| **Primary language** | C (integrates at the PostgreSQL engine level via C APIs) |
| **Latest version** | v0.8.2 (February 26, 2026) -- security fix for CVE-2026-3172 |
| **Companion project** | pgvectorscale (by Timescale) -- Rust-based extension adding DiskANN indexes and SBQ compression |

### 3.2 Architecture

**Core design:**
- PostgreSQL extension -- not a standalone database. Vectors are a first-class data type (`vector`) within PostgreSQL tables
- Inherits PostgreSQL's single-node architecture, ACID transactions, WAL (write-ahead log), replication, and point-in-time recovery
- Vectors stored in PostgreSQL's 8 KB page format

**Index types:**

| Index | Description | Best For |
|---|---|---|
| **HNSW** | Hierarchical Navigable Small World graph. Best recall-performance tradeoff. Preferred for production | Low-latency, high-recall workloads |
| **IVFFlat** | Inverted file with flat compression. Faster build times, lower memory | Batch updates, smaller datasets |
| **Streaming Disk ANN** | New index type (via pgvectorscale). Designed for larger-than-memory datasets | Large-scale, cost-efficient search |

**Dimension limits:**

| Vector Type | Max Storage | Max Indexed |
|---|---|---|
| `vector` (float32) | 16,000 | 2,000 |
| `halfvec` (float16) | 16,000 | 4,000 |
| `bit` (binary) | 64,000 | 64,000 |
| Sparse vectors | 16,000 non-zero elements | -- |

**Distance metrics:** L2 (Euclidean), inner product, cosine distance, L1 (Manhattan), Hamming, Jaccard.

**Scaling:**
- Single-node by default (inherits PostgreSQL's scaling model)
- Read replicas for query scaling
- Table partitioning for data scaling
- Connection pooling (PgBouncer) for concurrency
- Sharding across multiple PostgreSQL instances for very large deployments

### 3.3 Multimodal Support

| Capability | Status |
|---|---|
| Text embeddings | Fully supported -- store any embedding as a `vector` column |
| Image embeddings | Supported -- store pre-computed vectors from CLIP, Gemini Embedding 2, etc. |
| Audio/Video embeddings | Supported -- store pre-computed vectors from any multimodal model |
| Raw file storage | PostgreSQL can store binary data (`BYTEA`), file paths, or URLs alongside vectors |
| Integrated embedding | No built-in embedding generation. Use external models (OpenAI, Gemini, HuggingFace) |
| Cross-modal search | Fully supported if embeddings share a vector space |

**Metadata/Payload handling:**
- PostgreSQL's full relational model -- metadata is stored as regular columns (`TEXT`, `INTEGER`, `JSONB`, `TIMESTAMP`, etc.)
- No artificial metadata size limits (bounded only by PostgreSQL's row size limit of ~1.6 GB)
- Full SQL for filtering, joining, aggregating -- far more powerful than any dedicated vector DB's metadata filtering
- JSONB columns for flexible, schema-less metadata alongside typed columns
- Can JOIN vector search results with any other PostgreSQL table

**Key advantage:** pgvector is the only option of the three that can combine vector search with full relational queries (JOINs, GROUP BY, CTEs, window functions) in a single SQL statement.

### 3.4 Shortcomings

1. **Scalability ceiling** -- performance degrades noticeably beyond 10-20M vectors depending on dimensionality and hardware. Billion-scale requires significant engineering
2. **Index build times** -- HNSW index builds on large datasets can take hours and consume multiple times the dataset size in memory; can crash the database during builds
3. **Memory requirements** -- HNSW index must fit in `shared_buffers` for optimal performance. A 50M vector dataset at 768 dimensions requires ~852 GB total memory
4. **Limited index types** -- only HNSW and IVFFlat natively (DiskANN via separate pgvectorscale extension). No GPU-accelerated search
5. **Dimension indexing limits** -- float32 vectors capped at 2,000 dimensions for indexing (4,000 with half-precision)
6. **Pre-filter vs. post-filter problem** -- combining metadata filters with vector search can produce incomplete results or 100x latency variance depending on the query planner's choice
7. **Real-time ingestion challenges** -- PostgreSQL was not designed for high-velocity vector ingestion; WAL and MVCC overhead affects write throughput
8. **Not a first-class vector citizen** -- vectors are "bolted on" to a relational database; at scale, the abstractions leak
9. **ORM support gaps** -- Prisma still doesn't fully support pgvector or table partitioning as of late 2025

### 3.5 Paid Tier / Cloud Offering

pgvector itself is free and open-source. Managed PostgreSQL services that include pgvector:

| Provider | Free Tier | Paid Starting At | pgvector Support | Notes |
|---|---|---|---|---|
| **Supabase** | 500 MB storage, 50K MAU | $25/month (Pro) | Built-in, plus Gemini multimodal embedding integration | Full backend platform (auth, storage, edge functions) |
| **Neon** | Usage-based free tier | Usage-based (compute hours + storage) | Built-in | Serverless Postgres, auto-scaling, instant branching. Acquired by Databricks (May 2025) |
| **Timescale Cloud** | Limited free trial | Usage-based | Built-in + pgvectorscale | Optimized for time-series + vector workloads |
| **AWS RDS for PostgreSQL** | Free tier (12 months) | ~$15-50/month (db.t3.micro+) | Supported extension | Standard managed PostgreSQL |
| **Google Cloud SQL** | $300 credits | ~$30-80/month | Supported extension | Managed PostgreSQL |
| **Azure Database for PostgreSQL** | $200 credits | ~$25-60/month | Supported extension | Managed PostgreSQL |

**Cost advantage:** If you already run PostgreSQL, adding pgvector is effectively $0 incremental cost. A self-hosted setup handling millions of vectors can run on a $25-80/month VPS.

### 3.6 GUI / Dashboard

pgvector has **no dedicated GUI**. It inherits PostgreSQL's ecosystem of tools:

| Tool | Description |
|---|---|
| **pgAdmin 4** | Standard PostgreSQL GUI -- run vector queries, view results in tabular format, inspect table structures |
| **psql** | PostgreSQL CLI -- full SQL access to vector operations |
| **DBeaver** | Universal database tool with PostgreSQL support |
| **DataGrip** | JetBrains IDE with PostgreSQL support |
| **VS Code PostgreSQL extensions** | Query and manage directly from the editor |
| **Apache Superset** | Dashboard/BI tool that can query pgvector tables via SQL |
| **Adminer / phpMyAdmin** | Lightweight web-based database managers |

All of these show vector data in tabular/raw form. None provide native vector-specific visualization (cluster maps, embedding plots).

### 3.7 Data Visualization Platform

There is **no Kibana equivalent** purpose-built for pgvector. Options:

- **PostgreSQL monitoring tools** (pg_stat_statements, pganalyze, Datadog PostgreSQL integration) for query performance monitoring
- **Apache Superset / Grafana** -- connect to PostgreSQL and build custom dashboards over vector query results
- **Python + matplotlib/plotly** -- export vectors, apply PCA/t-SNE/UMAP dimensionality reduction, and create scatter plots
- **Nomic Atlas** -- export vectors and visualize embedding clusters (same tool usable with Pinecone)
- The PostgreSQL ecosystem has no dedicated "vector data explorer" as of 2026

### 3.8 API and SDK Support

pgvector works with **any language that has a PostgreSQL client**. Official pgvector-specific libraries:

| Language | Library | Integrations |
|---|---|---|
| **Python** | `pgvector-python` (v0.4.2) | Django, SQLAlchemy, psycopg2/3, Peewee |
| **Node.js/TypeScript** | `pgvector-node` (174K weekly npm downloads) | node-postgres, Knex, Sequelize, Prisma, TypeORM, Drizzle ORM, MikroORM, Kysely, pg-promise, Bun SQL, deno-postgres |
| **Java/Kotlin/Groovy/Scala** | `pgvector-java` | JDBC, Spring JdbcTemplate, Hibernate 6.4+, R2DBC. Spring AI PGVector support |
| **Go** | `pgvector-go` | pgx, pg, Bun, Ent, GORM, sqlx. LangChainGo integration |
| **C#/.NET** | Via Npgsql | Entity Framework, Dapper |
| **Ruby** | `neighbor` gem | ActiveRecord, Sequel |
| **Rust** | Via standard PG clients | sqlx, diesel |
| **PHP** | Via standard PG clients | Laravel, Doctrine |
| **Elixir** | `pgvector-elixir` | Ecto |

**Protocol:** Standard PostgreSQL wire protocol (libpq). Accessible via any PostgreSQL-compatible driver, connection pooler, or ORM in any language.

**Framework integrations:** LangChain (Python + JS), LlamaIndex, Spring AI, Quarkus LangChain4j, Google GenKit.

### 3.9 Performance Benchmarks

| Metric | Value | Conditions |
|---|---|---|
| **pgvector + pgvectorscale** | 471 QPS at 99% recall | 50M vectors, 768 dimensions |
| vs. Pinecone s1 | 28x lower p95 latency, 16x higher throughput | 50M vectors, 99% recall, 75% lower cost |
| vs. Pinecone p2 | 1.4x lower p95 latency, 1.5x higher throughput | 50M vectors, 90% recall, 79% lower cost |
| vs. Qdrant | 11.4x higher QPS (471 vs. 41) | 50M vectors, 99% recall |
| Insertion rate | ~30,000 vectors/sec | 1M vectors, 1536 dimensions |
| Query rate | ~3,000 queries/sec | 1M vectors, 1536 dimensions |
| Filtered query rate | ~2,000 queries/sec | 1M vectors, 1536 dimensions |

**Important caveat:** The headline pgvectorscale benchmarks use Timescale's DiskANN extension on top of pgvector, not vanilla pgvector alone. Vanilla pgvector HNSW performance is lower, particularly beyond 10M vectors.

**Scaling trajectory (real-world):**
- 10K vectors: queries in single-digit ms, index builds in seconds
- 500K vectors: acceptable latency, index builds in minutes
- 5M vectors: inconsistent latency (50ms to 5s), index builds take hours, memory spikes

---

## 4. Side-by-Side Comparison Table

| Feature | ChromaDB | Pinecone | pgvector |
|---|---|---|---|
| **Type** | Standalone vector DB | Managed cloud vector DB | PostgreSQL extension |
| **License** | Apache 2.0 | Proprietary | PostgreSQL License (BSD-like) |
| **Self-hosted** | Yes (free) | No (BYOC is Pinecone-managed) | Yes (free) |
| **Primary language** | Rust + Python | Rust (proprietary) | C |
| **Latest version** | v1.5.5 (Mar 2026) | Proprietary (SDK v8) | v0.8.2 (Feb 2026) |
| **Max dimensions** | No hard documented limit | 20,000 | 16,000 (storage), 2,000 (indexed) |
| **Index types** | HNSW only | Proprietary (automatic) | HNSW, IVFFlat, DiskANN (via pgvectorscale) |
| **Hybrid search** | Dense + sparse + full-text + regex | Dense + sparse + reranking | Dense + full-text (via PostgreSQL `tsvector`) |
| **SQL support** | No | No | Full SQL |
| **ACID transactions** | No | No | Yes (PostgreSQL) |
| **Multimodal (native)** | Images via OpenCLIP (Python only) | Modality-agnostic (store any vector) | Modality-agnostic (store any vector) |
| **Metadata limit** | No documented limit | 40 KB per record | PostgreSQL row limit (~1.6 GB) |
| **Built-in GUI** | No | Yes (web console) | No (uses pgAdmin, etc.) |
| **Authentication** | None by default | API key, RBAC, CMEK | PostgreSQL auth (roles, SSL, LDAP) |
| **Clustering/HA** | No | Built-in (managed) | Via PostgreSQL replication |
| **Free tier** | Self-hosted unlimited; Cloud: $5 credits | 2 GB storage, 100K records | Completely free (extension) |
| **Paid entry point** | $250/month (Team) | $50/month (Standard) | Cost of PostgreSQL hosting |
| **Official SDKs** | Python, JS/TS, Go, Rust | Python, JS/TS, Java, Go, .NET, Rust | Python, JS/TS, Java, Go (+ any PG client) |

---

## 5. Recommendations for a Multimodal Vector Database Project <a name="5-recommendations"></a>

### Key findings for building a new multimodal vector database:

**1. None of the three have deep native multimodal support.**
All three treat multimodal as "store vectors from any embedding model." ChromaDB goes slightly further with built-in OpenCLIP for text+image, but it is Python-only and limited to images. The real multimodal intelligence lives in the embedding model (CLIP, Gemini Embedding 2, etc.), not the database. This represents a significant gap in the market.

**2. Metadata handling is a major differentiator.**
pgvector's full SQL capabilities (JOINs, JSONB, arbitrary column types) vastly outperform ChromaDB's and Pinecone's key-value metadata stores. Pinecone's 40KB metadata limit is a real constraint for multimodal use cases where rich metadata (bounding boxes, transcripts, timestamps, multi-label annotations) is essential.

**3. Scalability architecture varies enormously.**
- ChromaDB: Single-node, best under 10M vectors. Prototyping-oriented.
- Pinecone: Fully managed, scales to billions, but closed-source and expensive.
- pgvector: Single-node PostgreSQL, competitive up to ~10-20M vectors; pgvectorscale pushes this to 50M+ but with significant engineering.

**4. Performance at scale favors pgvector+pgvectorscale or Pinecone.**
pgvector+pgvectorscale's 50M-vector benchmarks (28x better than Pinecone s1, competitive with Pinecone p2 at 79% lower cost) are impressive. ChromaDB is not competitive at this scale.

**5. The market is shifting toward "vectors as a data type" rather than dedicated vector databases.**
PostgreSQL, MongoDB, Oracle, and Elasticsearch all now support vectors natively. This trend suggests building on a general-purpose database foundation may be more future-proof than building another standalone vector DB.

**6. GUI and visualization tooling is universally weak.**
None of the three has a Kibana-equivalent for vector data exploration. This is an opportunity area. Tools like Nomic Atlas exist but are separate products. A multimodal vector database with built-in embedding visualization, cluster exploration, and cross-modal search debugging would be a significant differentiator.

---

## Sources

### ChromaDB
- [Chroma Official Site](https://www.trychroma.com/)
- [Chroma GitHub Repository](https://github.com/chroma-core/chroma)
- [Chroma Pricing](https://www.trychroma.com/pricing)
- [Chroma Documentation](https://docs.trychroma.com/)
- [Chroma Multimodal Embeddings Docs](https://docs.trychroma.com/docs/embeddings/multimodal)
- [Chroma Cookbook](https://cookbook.chromadb.dev/)
- [The Good and Bad of ChromaDB for RAG - AltexSoft](https://www.altexsoft.com/blog/chroma-pros-and-cons/)
- [ChromaDB Wikipedia](https://en.wikipedia.org/wiki/Chroma_(vector_database))
- [Chroma Flow Studio GUI](https://github.com/coffeecodeconverter/ChromaFlowStudio)
- [ChromaDash GUI](https://github.com/coffeecodeconverter/ChromaDash)
- [New Relic ChromaDB Observability](https://newrelic.com/instant-observability/chromadb)
- [Chroma Pricing 2026 - PE Collective](https://pecollective.com/tools/chroma-pricing/)
- [Introduction to ChromaDB 2026 - TheLinuxCode](https://thelinuxcode.com/introduction-to-chromadb-2026-a-practical-docsfirst-guide-to-semantic-search/)

### Pinecone
- [Pinecone Official Site](https://www.pinecone.io/)
- [Pinecone Pricing](https://www.pinecone.io/pricing/)
- [Pinecone Documentation](https://docs.pinecone.io/guides/get-started/overview)
- [Pinecone Database Limits](https://docs.pinecone.io/reference/api/database-limits)
- [Pinecone Quotas and Limits](https://docs.pinecone.io/reference/quotas-and-limits)
- [Pinecone Metadata Docs](https://docs.pinecone.io/guides/data/understanding-metadata)
- [Pinecone SDKs Reference](https://docs.pinecone.io/reference/pinecone-sdks)
- [Pinecone 2025 Release Notes](https://docs.pinecone.io/release-notes/2025)
- [Pinecone 2026 Release Notes](https://docs.pinecone.io/release-notes/2026)
- [Pinecone Explorer - Desktop GUI](https://www.pinecone-explorer.com/)
- [Visualizing Pinecone with Atlas - Andriy Mulyar](https://andriymulyar.com/blog/how-to-visualize-pinecone-vector-database)
- [The True Cost of Pinecone - MetaCTO](https://www.metacto.com/blogs/the-true-cost-of-pinecone-a-deep-dive-into-pricing-integration-and-maintenance)
- [Why We Replaced Pinecone with PGVector - Confident AI](https://www.confident-ai.com/blog/why-we-replaced-pinecone-with-pgvector)
- [Pinecone Security Analysis - IronCore Labs](https://ironcorelabs.com/vectordbs/pinecone-security/)
- [Pinecone Pricing 2026 - CostBench](https://costbench.com/software/vector-databases/pinecone/)
- [Multimodal Vector DB with Gemini and Pinecone - MindStudio](https://www.mindstudio.ai/blog/multimodal-vector-database-gemini-embedding-2-pinecone)
- [Pinecone Datadog Integration](https://docs.datadoghq.com/integrations/pinecone/)
- [Pinecone vs Qdrant Comparison 2026](https://is4.ai/blog/our-blog-1/pinecone-vs-qdrant-comparison-2026-319)
- [Pinecone G2 Reviews 2026](https://www.g2.com/products/pinecone/reviews)

### pgvector
- [pgvector GitHub Repository](https://github.com/pgvector/pgvector)
- [pgvector 2026 Guide - Instaclustr](https://www.instaclustr.com/education/vector-database/pgvector-key-features-tutorial-and-pros-and-cons-2026-guide/)
- [pgvector DeepWiki](https://deepwiki.com/pgvector/pgvector)
- [pgvector Complete Guide 2026 - Calmops](https://calmops.com/database/postgresql-vector-search-pgvector-2026/)
- [pgvector-python](https://github.com/pgvector/pgvector-python)
- [pgvector-node](https://github.com/pgvector/pgvector-node)
- [pgvector-java](https://github.com/pgvector/pgvector-java)
- [pgvector-go](https://github.com/pgvector/pgvector-go)
- [The Case Against pgvector - Alex Jacobs](https://alex-jacobs.com/posts/the-case-against-pgvector/)
- [pgvector Performance Benchmarks - Instaclustr](https://www.instaclustr.com/education/vector-database/pgvector-performance-benchmark-results-and-5-ways-to-boost-performance/)
- [pgvector Guide for DBA Part 2 - March 2026](https://www.dbi-services.com/blog/pgvector-a-guide-for-dba-part-2-indexes-update-march-2026/)
- [pgvector Neon Docs](https://neon.com/docs/extensions/pgvector)
- [pgvectorscale GitHub](https://github.com/timescale/pgvectorscale)
- [Multimodal Search with pgvector - Yugabyte](https://www.yugabyte.com/blog/postgresql-pgvector-multimodal-search/)
- [Supabase Gemini-Powered Multimodal Embeddings](https://www.createwith.com/tool/supabase/updates/supabase-adds-gemini-powered-multimodal-embeddings-for-cross-format-similarity-s)
- [Supabase Pricing](https://supabase.com/pricing)

### Comparative / Benchmark Sources
- [Vector Database Comparison 2026 - 4xxi](https://4xxi.com/articles/vector-database-comparison/)
- [Best Vector Databases 2026 - Firecrawl](https://www.firecrawl.dev/blog/best-vector-databases)
- [Best Vector Databases 2026 - Encore](https://encore.dev/articles/best-vector-databases)
- [Vector Database Benchmarks - Sesame Disk](https://sesamedisk.com/vector-databases-benchmarks-performance/)
- [Top 9 Vector Databases March 2026 - Shakudo](https://www.shakudo.io/blog/top-9-vector-databases)
- [pgvector vs Pinecone - Tiger Data](https://www.tigerdata.com/blog/pgvector-vs-pinecone)
- [What's Changing in Vector Databases 2026 - DEV Community](https://dev.to/actiandev/whats-changing-in-vector-databases-in-2026-3pbo)
- [Benchmarking Vector Databases 500K Vectors - Medium](https://medium.com/everyday-ai/benchmarking-vector-databases-what-i-learned-from-pushing-500k-vectors-through-their-limits-cda16bbbda39)

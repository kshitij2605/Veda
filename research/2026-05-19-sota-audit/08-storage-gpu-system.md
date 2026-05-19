# Veda SOTA Audit — Storage, GPU, System Architecture (May 2026)

## Verdict (TL;DR)

| Dimension | Veda today | 2026 SOTA | Gap severity |
|---|---|---|---|
| **Storage format** | JSONL (default) + per-field columnar | Lance 2.2 (LanceDB) / Storage v2 (Milvus) / Slab L0-L3 (Pinecone) / Object-storage-native (Turbopuffer, S3 Vectors) | **Medium-High**. JSONL is unambiguously below SOTA. Custom columnar is workable but Lance / Parquet integration would be table stakes for the data-lake story. |
| **Blob/object store** | MinIO + S3 (basic) | Same primitives, but Milvus/Pinecone/Turbopuffer treat object storage as **the** source of truth with hot/warm/cold tiers and slab-style immutables | **Medium**. Veda has the wires (S3) but no tiered cache, no slab-style compaction, no on-demand load. |
| **WAL** | Per-entry CRC32-framed WAL, tonic gRPC Raft transport | Milvus's Woodpecker (object-storage-native WAL replaces Kafka/Pulsar), batched group-commit, log-as-a-service | **Medium**. Per-entry WAL is correct but throughput-limited. Object-storage WAL (Woodpecker pattern) is the frontier. |
| **GPU indexes** | cuVS env-gated, ColPali/MUVERA/PLAID wired | cuVS 25.12+ with CAGRA (graph), IVF-PQ, IVF-Flat, brute-force, IVF-RaBitQ (new in 2026, 2.2x QPS over CAGRA), CAGRA dynamic batching (10x latency win) & persistent search (8x throughput) | **Medium**. cuVS is wired but CAGRA itself, dynamic batching, IVF-RaBitQ likely not yet first-class. |
| **NIM** | NIM client for embedding/search | NeMo Retriever Embedding NIM is the de-facto standard; native HF TEI/vLLM also fine. NIM-for-search is not a thing — NIM is embedding-side | **Low**. Veda's NIM client correctly positions NIM as embedding-side; "NIM vector search" was never a real product. |
| **Architecture shape** | Segment-based, single-machine first, S3 optional | Two converged shapes: (a) **slab/LSM on object storage** (Pinecone, Turbopuffer, Milvus 2.6) and (b) **Lance-style columnar lakehouse** (LanceDB, Apache Polaris+Lance) | **Medium**. Segments are fine and Milvus/Vespa still use them; but the field has clearly moved towards object-storage-native + LSM-style compaction. |

**Top 5 gaps / wins:**
1. **JSONL stored fields → Lance 2.2** would close the multimodal-lakehouse gap in one move. Highest ROI.
2. **CAGRA-first index path** with cuVS dynamic batching is currently table stakes; Veda's cuVS gating is half the work.
3. **Object-storage-native WAL** (Woodpecker pattern) — Veda's per-entry CRC32 WAL works but does not scale to billions of writes/sec the way Milvus 2.6 demonstrates.
4. **Hot/warm/cold tiering** with on-demand segment loading (Milvus 2.6, Turbopuffer) — Veda treats S3 as a backup target, not a query target.
5. **IVF-RaBitQ on GPU** — 2026 SIGMOD-quality result with 2.2x QPS over CAGRA at 95% recall. Veda has RaBitQ on CPU; the GPU version is a natural next step.

**Specific recommendations:** see bottom of file.

---

## A. Storage Layer / File Format

### A1. The Apache Parquet / Arrow / Feather question

Parquet was designed for batch SQL on tabular data and is not a great fit for vector embeddings as a primary format. Specifically:

- Parquet has no logical `VECTOR` type; embeddings must be encoded as a list-of-float column, which loses the fixed-width guarantee and disables many encodings. A February 2026 GitHub issue on `apache/parquet-format` explicitly raises this as an open problem ([parquet-format#553](https://github.com/apache/parquet-format/issues/553)).
- Parquet's row-group model is tuned for full-column scans, not point lookups by row offset — exactly what vector search needs (fetch the raw vector for a candidate ID returned by the index).
- Parquet's variant work, image-blob support, and the Iceberg v3 push are real but lag behind purpose-built formats by roughly two file-format generations.

Arrow IPC / Feather V2 (with Zstd/LZ4) is a perfectly fine **wire** format for vectors but is not a storage format — it doesn't have row-level random access or pages.

**Verdict A1:** Parquet/Arrow/Feather are not the right primary format for vectors in 2026. They are excellent **interchange** formats (Veda should be able to import/export Parquet/Arrow), but storing vectors there as the source of truth is not where the field is.

### A2. Lance (LanceDB) — the new colossus

Lance 2.1 went stable in October 2025; Lance 2.2 shipped in early 2026 with Blob V2 (`Inline`/`Packed`/`Dedicated`/`External` blob semantics), Map type support, nested schema evolution, and "50% storage cut vs Parquet, up to 68x faster blob reads" (per LanceDB's [v2.2 benchmark post](https://www.lancedb.com/blog/lance-format-v2-2-benchmarks-half-the-storage-none-of-the-slowdown)).

Key design choices:
- **Disk pages, not row groups.** Each column has independent pages, typically ~8 MB, sized for cloud-storage I/O efficiency.
- **Cascading encoding without giving up random access.** Two structural encodings (mini-block for small types, full-zip for large multimodal), with the explicit goal that selective reads remain a fixed small number of I/Os ([lance.org/format/file](https://lance.org/format/file/)).
- **Blob storage as a first-class citizen.** Image/video/PDF bytes live next to their embeddings without forcing the table itself to bloat.
- **Versioning baked in.** Lance is also a table format with snapshots / time travel.
- **Lakehouse story.** Apache Polaris+Lance landed in January 2026 ([blog](https://polaris.apache.org/blog/2026/01/06/apache-polaris-and-lance-bringing-ai-native-storage-to-the-open-multimodal-lakehouse/)) — Lance is now a credible open-lakehouse columnar format.

Production benchmarks (April 2026) put LanceDB into the top-5 production vector DBs alongside pgvector, Qdrant, Weaviate, and Milvus ([CallSphere 2026](https://callsphere.ai/blog/vector-database-benchmarks-2026-pgvector-qdrant-weaviate-milvus-lancedb)). LanceDB also announced 1.5M IOPS benchmarks and Uber-scale multi-bucket storage for 2026.

**Verdict A2:** Lance is production-ready in 2026 and is the closest thing the field has to a "default" multimodal columnar format. Veda's per-field columnar layout solves a similar problem in a similar way but loses the interop / lakehouse story.

### A3. Vespa's mixed tensor format

Vespa stores tensors with a mix of mapped (sparse) and indexed (dense) dimensions, e.g., `tensor<float>(p{}, x[384])`. This is structurally similar to how ColBERT-style multi-vector documents need to be stored (variable number of token vectors per document). On disk, Vespa Proton uses a doc-store of compressed serialized blobs, paired with a memory-resident attribute store for the dense parts.

Mixed tensors give Vespa a clean way to express ColBERT/ColPali multi-vector documents natively, which is a real architectural advantage Veda does not have today. Vespa's current MaxSim implementation for multi-vector retrieval is built on this primitive ([Vespa multi-vector HNSW](https://blog.vespa.ai/semantic-search-with-multi-vector-indexing/)).

**Verdict A3:** Vespa's mixed-tensor model is elegant for multi-vector. Veda's ColPali path through MUVERA/PLAID achieves similar capability but the storage primitive (per-token vectors stored as separate rows? as a packed blob?) deserves explicit design.

### A4. Milvus's segment format — Storage v2 and Woodpecker

Milvus 2.6 (May 2025 / GA late 2025; v3.0 targeted for late 2026) introduced two major storage changes:

1. **Storage v2** — A new on-disk segment layout described as "optimized structure that reduces IOPS and memory requirements." Hot/cold tiered storage on top of v2 cuts memory cost up to 80% and lets the same hardware serve 4.3× more data (2.2 TB / ~590M vectors) ([Milvus 2.6 blog](https://milvus.io/blog/introduce-milvus-2-6-built-for-scale-designed-to-reduce-costs.md)).
2. **Woodpecker WAL** — Replaces Kafka/Pulsar with a purpose-built object-storage-backed WAL. "Zero-disk architecture": log data lives in S3/GCS/OSS, metadata in etcd. This is the most architecturally distinctive 2026 vector-DB move ([Milvus blog](https://milvus.io/blog/we-replaced-kafka-pulsar-with-a-woodpecker-for-milvus.md)).

Milvus segments are still segments — immutable on flush, compacted in the background — and the lineage is much closer to FAISS / older Milvus than to Lance. The novelty is not the segment format itself but the **tiered placement** of segments across memory / NVMe / object storage.

**Verdict A4:** Milvus did *not* adopt Lance. It evolved its own segment format (Storage v2) plus an object-storage-native WAL (Woodpecker). The tiered loading story is the headline feature.

### A5. Iceberg / Delta Lake / Hudi for vectors?

The data-lakehouse-with-vectors story is real but partial in 2026:

- **Iceberg v3** (Databricks public preview Q1 2026) is the most aggressive: Variant type, deletion vectors, row-level deletes, and "wide ML table" optimizations. Confluent's Tableflow generates embeddings into Iceberg/Delta tables in real time.
- **Lance** is already a lakehouse format on its own (file + table + catalog spec), and Apache Polaris integration makes it interoperable with the Iceberg world.
- **Spice** combines vector search and SQL over Iceberg/Delta/Hudi tables at the edge.

But: nobody is using raw Iceberg/Delta Parquet files as a *primary* index store today. The pattern is "Iceberg/Delta for the documents and metadata; Lance or a purpose-built vector DB for the embeddings + index." A unified single-format story is still aspirational.

**Verdict A5:** Veda does not need to be an Iceberg participant in 2026, but should be able to **read** Parquet/Iceberg as an ingest source. Writing Lance natively is the better unification bet than Iceberg.

### A6. Custom columnar (Veda) vs Parquet vs Lance — explicit tradeoffs

| | Veda custom columnar | Parquet | Lance 2.2 |
|---|---|---|---|
| Vector encoding | Per-field file (good) | List-of-float, no `VECTOR` type | Native fixed-width with blob support |
| Random access by row | Likely O(1) with offsets | Row-group scan (slow) | Page-indexed O(1) |
| Compression | Not specified | Zstd/Snappy global | Per-page cascading w/o random-access loss |
| Schema evolution | Custom | Limited | First-class incl. nested |
| Time travel | None | None | Yes |
| Lakehouse interop | None | Iceberg/Delta | Polaris, Iceberg interop |
| Ecosystem | None | Massive | Growing fast |

**Verdict A6:** Veda's custom columnar is correct **for the use case** but loses the ecosystem, lakehouse interop, and time-travel that Lance brings essentially for free. The honest answer is: **adopt Lance 2.2 as a second stored-fields format option**, alongside the existing JSONL and custom-columnar. Wire Lance via `lance-format/lance` (Rust-native).

### A7. mmap'd index files

Standard pattern across the field. Qdrant uses immutable memmaps for HNSW after segment optimization; Weaviate snapshots HNSW indexes (avoiding WAL rebuild) since 1.31; DiskANN/Vamana paged-graph design *is* the mmap pattern. Milvus's tiered storage treats the local disk as a cache and mmaps when hot.

**Verdict A7:** mmap is standard, Veda almost certainly already does this for its index payloads. No gap.

### A8. Compressed columnar vector storage

State of the art for the raw vector bytes:

- **Zstd-3** for high-ratio cold storage (Lance, Parquet default).
- **LZ4** for low-CPU warm tier (ZFS default, Feather V2).
- **Bit-packed quantization** at the index layer (RaBitQ, PQ, SQ8) — this is where the real compression happens. RaBitQ-style 1-bit-per-dim quantization is now in production at Milvus, Qdrant (Rotational Quantization in Weaviate 1.32), and as a GPU index in cuVS via IVF-RaBitQ.
- **MUVERA / PLAID** for multi-vector compression — already in Veda's pipeline.

**Verdict A8:** Zstd at the file level + RaBitQ/PQ at the index level is the right combo. Veda has the index-level pieces; the file-level Zstd is a one-line config win.

---

## B. GPU Acceleration

### B1. NVIDIA cuVS — 2026 status

cuVS follows a CalVer 2-month release cycle. Versions today (May 2026): nightly **26.02**, stable **25.12**, legacy **25.10**. Officially supported indexes:

- **CAGRA** — graph ANN on GPU. Build via IVF-PQ or NN-DESCENT. The reference 2026 GPU ANN index.
- **IVF-PQ** — inverted file with product quantization.
- **IVF-Flat** — inverted file with brute-force per-list.
- **Brute force** — exhaustive search, GPU-vectorized.
- HNSW (as a fallback / interop format), Vamana, NN-Descent (auxiliary).

Notable 2026 cuVS features:
- **Dynamic batching API** for CAGRA: up to 10× latency improvement for high-volume online search.
- **Persistent search**: 8× throughput improvement maintaining low single-query latency.
- **Binary-quantized CAGRA**: graph built directly on 1-bit quantized vectors.
- **High-selectivity prefiltering**: CAGRA now keeps recall when 99% of candidates are pre-excluded by a filter (this is the killer filtered-search feature).

Production adoption:
- **Milvus 2.4+** uses cuVS for `GPU_CAGRA`, `GPU_IVF_PQ`, `GPU_IVF_FLAT`, `GPU_BRUTE_FORCE` (RAFT → cuVS migration completed).
- **FAISS** integrates cuVS for GPU indexes (May 2025 Meta announcement).
- **OpenSearch / Elasticsearch** have GPU vector indexing via cuVS as of 25.06.
- **Apache Lucene** accelerated via cuVS (Searchscale, 25.06).

**Verdict B1:** cuVS is *the* GPU vector library in 2026. Veda's env-flag-gated cuVS path is the right direction. The gap is that CAGRA-first is now table stakes — cuVS dynamic batching and persistent search are unambiguous wins for QPS and tail latency.

### B2. NVIDIA NIM — current capabilities

NIM is **inference-side**, not search-side. The relevant NIM for a vector DB is **NeMo Retriever Embedding NIM**, which serves text and image embedding models behind an OpenAI-compatible API, accelerated by TensorRT + Triton.

- Wikimedia migrated 10M Wikidata entries in <3 days using NeMo Retriever NIM (was 30 days previously).
- Vector DB integrations (Datastax, FAISS, Kinetica, Milvus, Redis, Weaviate) treat NIM as the embedding source, not the search engine.

There is **no NIM product** that does vector search itself. cuVS does that.

**Verdict B2:** Veda's NIM client correctly targets the embedding side. No architectural gap — the design is right.

### B3. RAFT — predecessor to cuVS, still supported?

Effectively superseded by cuVS in 2025. Milvus migrated GPU support from RAFT → cuVS. cuVS is the supported library going forward. Veda should not be writing new code against RAFT.

**Verdict B3:** No action. Veda's cuVS-first stance is correct.

### B4. GPUDirect Storage / GPUDirect RDMA for vectors

2026 has finally turned this from a curiosity into a real production pattern:
- **Cloudian S3 RDMA + GPUDirect** = 8× faster Milvus indexing (2 hours → 16 minutes).
- **VAST + NVIDIA InsightEngine** uses GPUDirect to pull vectors straight to GPU memory, bypassing CPU.
- **VDURA** added RDMA across its platform for GPU servers.
- **MinIO AIStor on BlueField-4** (NVIDIA STX reference architecture) — GA H2 2026.

The pattern: GPU-side ANN index needs vectors and metadata streamed at high rates during build/recall. GPUDirect Storage lets the GPU DMA directly from NVMe or RDMA-backed object storage, sparing the CPU and PCIe bottleneck.

**Verdict B4:** GPUDirect Storage is no longer experimental for vector workloads. It's not yet a default; it requires specific NICs (ConnectX-7/8, BlueField), specific filesystems (WekaFS, VAST, MinIO AIStor), and a kernel-mode driver. For Veda, this is a "phase 3" target — meaningful when the GPU index build becomes the bottleneck.

### B5. CAGRA bench results vs HNSW

The most relevant 2026 numbers (from the original CAGRA paper [arXiv:2308.15136](https://arxiv.org/abs/2308.15136) and the GPU IVF-RaBitQ paper [arXiv:2602.23999](https://arxiv.org/abs/2602.23999)):

- **Large-batch throughput at 90–95% recall:** CAGRA is **33–77× faster** than HNSW, **3.8–8.8×** faster than other SOTA GPU implementations.
- **Single query at 95% recall:** CAGRA is **3.4–53×** faster than HNSW.
- **Index build:** CAGRA builds up to **12.3×** faster than HNSW.
- **Caveat:** small-batch / cold queries lose much of the GPU advantage because of kernel launch overhead. CAGRA persistent search (2025) and dynamic batching (2025) mitigate this substantially.

**Verdict B5:** CAGRA is genuinely better than HNSW for the workloads where you can keep the GPU saturated. For latency-critical single-query online search with low arrival rate, HNSW on CPU still wins. The right answer in Veda is "have both, route by load."

### B6. AMD ROCm — competitive GPU ANN story?

Yes, since 2025. **hipVS + hipRAFT** is AMD's cuVS clone, API-compatible. Supports IVF-Flat, IVF-PQ, CAGRA, brute force. Has C/C++/Python/Rust bindings ([ROCm Blogs](https://rocm.blogs.amd.com/software-tools-optimization/hipvs/README.html)).

Real-world: hipVS is on AMD MI300/MI325X GPUs. The API compatibility means a cuVS-targeted code path can compile against hipVS with minor adjustments. We have not yet seen large-scale production deployments of hipVS in named vector DBs (no Milvus / Vespa / Qdrant adoption announcement).

**Verdict B6:** hipVS exists and is credible. For Veda, this is a "do not lock yourself to cuVS-specific APIs" recommendation — the rest is wait-and-see.

### B7. TensorRT-LLM / vLLM for embedding inference

For dense embedding inference on H100 / B100 (May 2026):

- **TensorRT-LLM**: 15–30% higher peak throughput than vLLM after a 10–30-minute compile step. Lower p50/p95 latency at all concurrencies (e.g., p95 TTFT 1280 ms vs vLLM 1450 ms @ 100 concurrent on GPT-OSS-120B / 2× H100). Best for a single model in long-term production.
- **vLLM**: Better multi-hardware support, faster deployment, model-flexibility. 4,741 tok/s at 100 concurrent on GPT-OSS-120B / 2× H100.
- **SGLang**: Competitive at low concurrencies; structured-generation winner.

For pure embedding inference (sequence-classification head, no generation), TEI (HuggingFace) is also still extremely competitive and is what Veda already uses for BGE-M3 on CPU.

**Verdict B7:** Veda's docker-compose split (TEI for BGE-M3 CPU, vLLM for multimodal Qwen3-VL) is exactly what the field is doing. No change. NIM (TensorRT-LLM under the hood) is the option for SLA-critical single-model deployments.

---

## C. System Architecture — what shape do vector DBs take in 2026?

### C1. Turbopuffer — object-storage-native search engine

Architecture ([Turbopuffer blog](https://turbopuffer.com/blog/turbopuffer), [567-labs talk](https://567-labs.github.io/systematically-improving-rag/talks/turbopuffer-engine/)):

- **S3 / GCS is the source of truth.** No replicated SSDs, no replicated WAL. Writes are durably committed to object storage.
- **Two-tier cache**: in-memory + NVMe SSD. Warm queries hit ~10–18 ms p90. Cold queries hit 250–450 ms p90.
- **Indexes inside slabs** (or namespaces) chosen per-slab based on size: scalar quantization or random projections for small, IVF for large.
- **Query plan minimizes object-storage roundtrips** (target: ≤3 for cold).
- **Stateless query nodes** — any node can serve any namespace; horizontal scale = cheap.
- **Production scale**: 2.5T+ documents, 10M+ writes/s, 10k+ QPS. Cursor, Notion, Linear are reference users.

Key enabling primitives (none of which existed in 2018):
- S3 strong consistency (2020)
- S3 compare-and-swap (Dec 2023)
- NVMe SSD only 4–5× slower than RAM, 100× cheaper

**Verdict C1:** Turbopuffer is the cleanest expression of "object storage is the new disk" for vector search. The cold-query latency tradeoff (250–450 ms) is acceptable for the long-tail-of-namespaces use case (per-tenant SaaS, codebase indexes). Not a fit for a single hot index with hard sub-50 ms p99.

### C2. LanceDB — embedded multimodal lakehouse

Architecture:
- Built on Lance file format (covered in A2).
- **Embedded-first** (you can use it from a Python script with no server) but supports server mode.
- Vector index types: IVF-PQ, IVF-Flat, HNSW. CAGRA support is requested ([GitHub issue #6534](https://github.com/lance-format/lance/issues/6534)).
- **Lance-native SQL via DuckDB** (2026).
- Latency claims: 25 ms vector search; 50 ms with metadata filtering ([LanceDB perf benchmarks](https://docs.lancedb.com/enterprise/benchmark/benchmark)).
- Multi-bucket "uber-scale" storage announced for 2026.

**Verdict C2:** LanceDB is closer to a *file-based library* than a database server. It is the strongest argument for "store your vectors in a columnar file format and put a thin server on top" — exactly the gap Veda's JSONL+custom-columnar choice could close.

### C3. Pinecone Serverless (v2, Q1 2026) — slab LSM on object storage

Architecture ([Pinecone slab blog](https://www.pinecone.io/learn/slab-architecture/)):
- **Memtable + immutable slabs (L0 → L1 → L2 → L3).** Writes hit a memtable, flush to L0 slabs, compact into L1/L2/L3.
- **Per-level indexing**: memtable = brute-force in RAM (good for ~10k); small slabs use Pinecone's proprietary FJLT-based "ananas"; large slabs (>1M) use IVF.
- **Storage in object storage, executor cache** in memory + local SSD. Cache miss = fetch slab from object storage.
- **Separated reads, writes, and compaction**: writes go to new L0 slabs, queries fan out across all slabs, compaction is background.
- **Dedicated Read Nodes (DRN)** for read-heavy workloads (2025–2026 addition).
- **Pinecone Inference** (hosted embedding/reranker models integrated, 2026).

This is **exactly the LSM-tree pattern** ported to vector search. Memtable for hot writes, levels with increasingly heavy indexing as data ages.

**Verdict C3:** Pinecone is the most direct application of LSM principles to vector search. The slab/level model is what most production vector DBs are converging on (Milvus's segments + tiered storage approximate it; Turbopuffer's namespaces approximate it).

### C4. Milvus 2.5+ / Milvus Lite

- **Milvus 2.5** (late 2024 / 2025): incremental polish. Better object-storage retry under rate-limiting.
- **Milvus 2.6** (May 2025 / GA late 2025): the big release. Storage v2, Woodpecker WAL (replaces Kafka/Pulsar), tiered storage. 4× faster than Elasticsearch on full-text, 72% memory reduction without recall loss, 4.3× more data per host.
- **Milvus 3.0** (target late 2026): refactored streaming, deeper hot/cold tiering, upgraded data model.
- **Milvus Lite**: embedded mode (Python pip-installable), Lance-like positioning, for prototyping.

**Verdict C4:** Milvus is the most engineered open-source vector DB in 2026 and is doing the most ambitious architectural work (Woodpecker is the standout). Worth tracking closely.

### C5. Qdrant 1.16+ — Tiered Multitenancy

Released Nov 2025 ([Qdrant blog](https://qdrant.tech/blog/qdrant-1.16.x/)):
- **User-defined sharding** + **Fallback Shards** + **Tenant Promotion**. Small tenants share a fallback shard; promoted tenants get dedicated shards. Solves the noisy-neighbor problem for SaaS multi-tenant.
- **Inline storage**: quantized vector data inside HNSW nodes. 33 reads/iter → 2 reads/iter. 20 QPS → 211 QPS on disk-bound workloads.
- **io_uring** for async I/O on Linux.
- **Immutable memmaps** post-optimization.
- **ACORN** for filtered HNSW search at low filter selectivity.

**Verdict C5:** Qdrant has the strongest multi-tenancy story among open-source vector DBs in 2026. The fallback-shard + tenant-promotion pattern is genuinely new.

### C6. Vespa — long-standing, still distinctive

- **Proton search core**: per-node process maintaining doc-store + memory attributes per document type.
- **Mixed tensor format**: `tensor<float>(p{}, x[384])` for multi-vector documents (ColBERT/ColPali fit naturally).
- **Per-node transaction log** (append-only, fsync-style); document store is compressed serialized blobs in immutable file pairs.
- **HNSW for ANN**, multi-stage ranking, real-time ingest. Hybrid sparse+dense at the core.
- **Tensor-based retrieval** as the abstraction (not vector-only) — this is where Vespa is genuinely ahead.

**Verdict C6:** Vespa's mixed-tensor abstraction is the cleanest way to represent ColPali-style multi-vector documents. Veda's choice (MUVERA/PLAID compression on top of raw multi-vec) is fine but worth comparing.

### C7. Weaviate v1.27+ → 1.32 / 1.34

Recent release cadence ([Weaviate releases](https://weaviate.io/blog/tags/release)):
- **1.26**: tenant offloading to S3.
- **1.31**: MUVERA encoding (matches Veda), HNSW snapshotting (avoid WAL replay), Cohere v3.5/v4, model2vec, BM25 new operators.
- **1.32**: Collection aliases for migration, **Rotational Quantization (RQ)** (Weaviate's RaBitQ-equivalent), compressed HNSW connections, replica movement GA, cost-aware sorting (2–200× faster filtered/sorted queries).
- **1.34**: further refinement.

Weaviate's architecture is per-tenant-shard (every tenant a shard) and the move toward offloading inactive tenants to cold S3 is now mature.

**Verdict C7:** Weaviate has converged on a similar feature set: multi-tenancy with offloading, quantization (RQ), MUVERA for multi-vec. Veda already has MUVERA; RQ is essentially RaBitQ which Veda also has.

### C8. Vectara — proprietary search-as-a-service

- Proprietary embedding model: **Boomerang** (strong cross-lingual).
- Proprietary hallucination detector: **HHEM**.
- 2026 update: structured-knowledge-graph + multimodal hybrid retrieval.

Not architecturally distinctive on the storage side — Vectara is positioned as a managed RAG service, not a foundation library.

**Verdict C8:** Out of scope for an architectural comparison. Veda is not competing with Vectara at the product level.

### C9. AWS S3 Vectors — GA Dec 2025

The biggest 2026 surprise. AWS made S3 itself a vector store:
- **2 billion vectors per index, 20 trillion per bucket** (40× preview capacity).
- **<1s cold, ~100ms warm** query latency.
- **1,000 PUT/s** writes.
- **First-class Bedrock + OpenSearch integration**.
- 14 AWS regions.

This is the "object storage IS the vector DB" story taken to its logical conclusion. The latency is "cold" by any other vector DB's standard, but the cost (no compute layer to provision) and capacity (trillions of vectors) shift the conversation.

**Verdict C9:** S3 Vectors is a major architectural data point. It validates Turbopuffer's bet. For Veda the question is: should Veda be able to *use* S3 Vectors as a cold tier? Probably yes, eventually.

### C10. Common patterns — what shape do 2026 vector DBs have?

Two convergent shapes:

**Shape A — Slab/LSM on object storage** (Pinecone, Turbopuffer, Milvus 2.6, S3 Vectors)
- Writes → memtable → flush to immutable slab in S3.
- Per-slab indexing: lighter for small slabs, heavier for large.
- Background compaction: L0 → L1 → L2.
- Read path: cache (RAM + NVMe) → object storage on miss.
- Separation of write, read, compaction paths.

**Shape B — Columnar lakehouse format** (LanceDB, Lance + Polaris)
- Lance format files in S3.
- Native vector + scalar + blob columns.
- Time travel via snapshots.
- Index is auxiliary, not the source of truth.
- Lakehouse-style read via Arrow/DuckDB/Polars.

**Veda today is Shape C — segment-based, single-shard-first, S3 as backup.** This is the 2022 shape. Segments are still fine (Milvus and Vespa use them) but the modern variant is "segments live in S3, RAM/NVMe is a cache, log is in object storage too."

Tiering is universal in 2026:
- **Hot** (RAM): recent / high-QPS data.
- **Warm** (local NVMe): bulk of working set.
- **Cold** (object storage): inactive tenants, archived data.

Sharding-vs-single-shard split:
- Single shard: LanceDB embedded, Turbopuffer per-namespace, Pinecone per-namespace.
- Multi-shard: Milvus (DataNode/QueryNode), Weaviate (per-tenant), Qdrant (user-defined + fallback).

---

## D. Specific Recommendations for Veda

### D1. Storage format
1. **Adopt Lance 2.2 as a third stored-fields format option**, alongside JSONL and custom columnar. Use the official Rust `lance` crate. Default-keep JSONL for backwards compat, default-new for Lance for the multimodal + lakehouse story.
2. **Add Zstd-3 page-level compression to the custom columnar format** if Lance isn't adopted immediately. One-line ROI.
3. **Add Parquet/Arrow IPC as ingest formats** (read-only). Iceberg/Delta as read-only sources for future.
4. **Keep JSONL** as the debug/diagnostic format — its strength is human-readability.

### D2. WAL
5. **Group-commit the existing per-entry CRC32 WAL**. Batch N entries per fsync. Easy 5–10× write throughput.
6. **Investigate an object-storage-backed WAL** (Woodpecker pattern) as a Veda 0.3 / 0.4 target. Not urgent but the field is moving here.

### D3. GPU
7. **Make CAGRA a first-class index type**, not just an env-flag fallback. Wire cuVS dynamic batching for online workloads and persistent search for sustained QPS.
8. **Investigate IVF-RaBitQ on GPU** (the 2026 GPU paper claims 2.2× QPS vs CAGRA at 95% recall). Veda already has CPU RaBitQ; the GPU port is small.
9. **Don't lock to cuVS-only APIs**. Keep the index trait portable so hipVS (AMD) can be swapped later.
10. **Keep NIM client as embedding-side only**. The architecture is correct; no need to chase "NIM vector search."

### D4. Architecture shape
11. **Build out the hot/warm/cold tier**. Today S3 is a backup; make it a query-path target with NVMe + RAM as caches. Borrow Milvus 2.6's pattern.
12. **Slab-style compaction is a future-proofing investment.** Veda's segments are conceptually L1; a memtable + L0-flush + background compaction gets you the LSM pattern without rewriting.
13. **Per-namespace stateless query** (Turbopuffer pattern) is worth designing for even before implementing — keeps you cloud-native.
14. **Multi-tenancy: take Qdrant's tiered approach** (fallback shard + promotable tenants). This is the cleanest pattern in 2026.
15. **Mixed-tensor / multi-vector storage**: explicitly design how ColPali documents are stored on disk. Today MUVERA+PLAID compresses them; the question is the layout (packed blob vs per-token-row vs Vespa-style mapped+indexed tensor).

### D5. Observability + GPUDirect (phase 3)
16. **GPUDirect Storage** is a credible 2026 path for build-time speedups (Cloudian + Milvus showed 8×). Not urgent; wait for the build to become the bottleneck.

---

## E. Five Most Important Items to Steal in the Next 6 Months

1. **Lance 2.2 as a stored-fields option.** Biggest single architectural win. Buys the lakehouse story, time travel, columnar reads, and 50% storage cut vs Parquet — all in one library.
2. **CAGRA-first GPU index path** with dynamic batching. Veda already gates cuVS; promoting CAGRA from "optional" to "preferred for high-QPS / batch" is the right 2026 default.
3. **Hot/warm/cold tiering with object storage as a query target.** Milvus 2.6's pattern. Single-largest cost-saving win for users with >100M vectors.
4. **Group-commit WAL.** A trivial change with 5–10× write throughput. Best ROI per LOC.
5. **IVF-RaBitQ on GPU.** Veda already has RaBitQ on CPU; the 2026 SIGMOD-class paper gives a 2.2× QPS-at-95%-recall win over CAGRA. Strong "competitive differentiator vs Milvus" pitch.

---

## Sources

**Storage formats and lakehouse**
- [LanceDB Lance Format 2.1 stable](https://www.lancedb.com/blog/lance-file-2-1-stable)
- [Lance Format v2.2 benchmarks](https://www.lancedb.com/blog/lance-format-v2-2-benchmarks-half-the-storage-none-of-the-slowdown)
- [Lance file format spec](https://lance.org/format/file/)
- [Apache Polaris + Lance (Jan 2026)](https://polaris.apache.org/blog/2026/01/06/apache-polaris-and-lance-bringing-ai-native-storage-to-the-open-multimodal-lakehouse/)
- [Parquet for vectors GitHub discussion](https://github.com/apache/parquet-format/issues/553)
- [Iceberg v3 public preview on Databricks](https://www.databricks.com/blog/next-era-open-lakehouse-apache-icebergtm-v3-public-preview-databricks)
- [Apache Parquet vs Lance/Vortex/BtrBlocks (2026)](https://dipankar-tnt.medium.com/apache-parquet-vs-newer-file-formats-btrblocks-fastlanes-lance-vortex-cdf02130182c)
- [Arrow IPC / Feather V2](https://arrow.apache.org/docs/python/feather.html)

**Vector DB architectures**
- [Turbopuffer architecture blog](https://turbopuffer.com/blog/turbopuffer)
- [Pinecone slab architecture](https://www.pinecone.io/learn/slab-architecture/)
- [Pinecone serverless v2 (2026)](https://www.pinecone.io/blog/serverless-architecture/)
- [Milvus 2.6 announcement](https://milvus.io/blog/introduce-milvus-2-6-built-for-scale-designed-to-reduce-costs.md)
- [Milvus tiered storage](https://milvus.io/blog/milvus-tiered-storage-80-less-vector-search-cost-with-on-demand-hot%E2%80%93cold-data-loading.md)
- [Milvus Woodpecker WAL](https://milvus.io/blog/we-replaced-kafka-pulsar-with-a-woodpecker-for-milvus.md)
- [Qdrant 1.16 tiered multitenancy](https://qdrant.tech/blog/qdrant-1.16.x/)
- [Weaviate 1.31 release](https://weaviate.io/blog/weaviate-1-31-release)
- [Weaviate 1.32 release](https://weaviate.io/blog/weaviate-1-32-release)
- [Vespa Proton storage](https://docs.vespa.ai/en/proton.html)
- [Vespa multi-vector HNSW](https://blog.vespa.ai/semantic-search-with-multi-vector-indexing/)
- [AWS S3 Vectors GA](https://aws.amazon.com/blogs/aws/amazon-s3-vectors-now-generally-available-with-increased-scale-and-performance/)
- [Turbopuffer vs Pinecone in 2026](https://particula.tech/blog/turbopuffer-vs-pinecone-vector-database-migration)

**GPU / cuVS / NIM**
- [cuVS Milvus integration docs](https://docs.rapids.ai/api/cuvs/nightly/integrations/milvus/)
- [Optimizing vector search with cuVS](https://developer.nvidia.com/blog/optimizing-vector-search-for-indexing-and-real-time-retrieval-with-nvidia-cuvs/)
- [Faiss + cuVS (Meta engineering)](https://engineering.fb.com/2025/05/08/data-infrastructure/accelerating-gpu-indexes-in-faiss-with-nvidia-cuvs/)
- [CAGRA paper arXiv:2308.15136](https://arxiv.org/abs/2308.15136)
- [GPU IVF-RaBitQ arXiv:2602.23999](https://arxiv.org/abs/2602.23999)
- [hipVS / hipRAFT on AMD ROCm](https://rocm.blogs.amd.com/software-tools-optimization/hipvs/README.html)
- [NVIDIA NeMo Retriever Embedding NIM docs](https://docs.nvidia.com/nim/nemo-retriever/text-embedding/latest/overview.html)
- [Cloudian S3 RDMA + Milvus 8× speedup](https://cloudian.com/blog/supercharging-vector-database-indexing-8x-faster-with-cloudian-s3-rdma-and-nvidia/)
- [vLLM vs TensorRT-LLM 2026 benchmarks](https://medium.com/synthetic-futures/vllm-vs-tensorrt-llm-the-definitive-2026-comparison-for-llm-inference-ed0943fb81d2)

**Other**
- [Starling: I/O-efficient disk graph index SIGMOD'24](https://arxiv.org/pdf/2401.02116)
- [Vector DB benchmarks 2026 (CallSphere)](https://callsphere.ai/blog/vector-database-benchmarks-2026-pgvector-qdrant-weaviate-milvus-lancedb)
- [Best vector databases in 2026 (MarkTechPost)](https://www.marktechpost.com/2026/05/10/best-vector-databases-in-2026-pricing-scale-limits-and-architecture-tradeoffs-across-nine-leading-systems/)

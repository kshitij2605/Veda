# Veda vs Pinecone

Pinecone is a proprietary, fully-managed vector database — you send it vectors, it serves
similarity search. Veda is an open-source, self-hostable engine that owns the whole pipeline
from raw documents to a grounded answer. The two solve different problems; here's how to
think about the trade-off.

## The core difference: managed SaaS vs open, self-hostable

Pinecone is **cloud-only** — there is no self-hosted or on-prem option, and your vectors
live in Pinecone's infrastructure under Pinecone's terms. Veda is **open source (AGPL-3.0)**
and runs anywhere: your laptop, your data center, an air-gapped network, or your own cloud
account. If data sovereignty, on-prem/air-gapped deployment, or avoiding vendor lock-in
matters to you, that alone decides it.

| | Veda | Pinecone |
|---|---|---|
| Deployment | Self-hosted or your cloud — open source | Managed SaaS only — proprietary |
| License / source | AGPL-3.0 (+ commercial option) | Closed source |
| Data locality | Anywhere you run it, incl. air-gapped | Pinecone's cloud infrastructure |
| Modalities | Unified text · image · audio · video space | Vectors you provide (bring your own embeddings) |
| Document ingestion | Built-in: PDF/DOCX/PPTX/XLSX/scans → OCR → chunk → embed | None — you embed and upload vectors yourself |
| Search modes | Dense + sparse (SPLADE) + BM25 + hybrid fusion + reranking | Dense (+ sparse in some tiers) |
| Index types | 12 tunable (HNSW, DiskANN, IVF-PQ, RaBitQ, SPANN, SQ4/SQ8) | Managed, internals not user-tunable |
| Pricing | Free (self-hosted) or a flat commercial license | Usage-based (storage + read/write units) |

## Why teams pick Veda over a managed vector store

- **You don't need to build the ingestion pipeline yourself.** Pinecone stores vectors —
  getting from a folder of PDFs, images, and recordings to searchable vectors is entirely
  on you (typically stitched together from Unstructured, LangChain, and a separate OCR
  service). Veda does OCR → chunk → embed → insert in one API call across every modality.
- **One embedding space, every modality.** Index a document's text and its diagrams
  together; query with a photo and get back the matching manual page. Pinecone has no
  native concept of cross-modal retrieval — you'd need a separate system per modality.
- **RAG quality is built in, not stitched on.** Faithfulness/citation scoring, an eval
  harness with chunker A/B testing, a semantic response cache, and embedding-based
  guardrails ship with the engine — not a separate LangSmith/Phoenix/Galileo integration.
- **No vendor lock-in, no usage-metered bill for every query.** Self-host on infrastructure
  you already pay for, with a predictable flat license instead of per-read/write-unit
  pricing that scales with traffic.
- **A rigorously engineered ANN core.** Pure Rust, SIMD (AVX2+FMA) distance kernels, mmap
  zero-copy reads, and 12 index types with an explicit recall/latency dial — we benchmark
  this seriously (see our [head-to-head against Elasticsearch](/docs/vs-elasticsearch),
  where Veda's dense search came out ~13× faster than a mature production engine on equal
  hardware).

## Where Pinecone is a reasonable choice

If you already have embeddings and just need a managed, zero-ops similarity index — no
self-hosting, no document pipeline, no multimodal requirement — Pinecone's simplicity and
managed-infrastructure model is a legitimate fit, especially for small teams who'd rather
not operate any database themselves. Veda's advantage shows up once you need multimodal
ingestion, self-hosting/sovereignty, hybrid search, or the surrounding RAG-quality tooling
that a pure vector index doesn't provide.

## Try it yourself

Bring a folder of mixed PDFs, images, and audio and run them through
[the document pipeline](/docs/document-pipeline) — no separate embedding step required.
See the [quick start](/docs/quickstart) to get running in a few minutes.

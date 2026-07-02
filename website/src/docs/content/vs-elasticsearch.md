# Veda vs Elasticsearch

Elasticsearch added `dense_vector` + kNN and ELSER on top of a text-search-first engine.
Veda is a vector database built from the ground up for the multimodal RAG era. Here's an
honest, benchmarked comparison.

## The short version

> Elasticsearch is a mature, battle-tested platform for **text and log search at scale**.
> Veda is the place to **ingest any document — PDF, Office, or scans — and search and reason
> across text, images, audio, and video together**, with RAG quality controls built in, and,
> on equal hardware, dense-vector search that's faster too.

## Dense-vector performance — benchmarked head-to-head

We ran the same 30,000-passage corpus, the same embeddings, and the same queries against
Veda and a production Elasticsearch 8.19 cluster. Recall is identical; latency was measured
twice — once on a shared, contended host, once on a quiet dedicated one — to isolate the
index from the machine it runs on:

| | Veda (dedicated host) | Veda (contended dev host) | Elasticsearch 8.19 |
|---|---:|---:|---:|
| recall@10 | 1.000 | 1.000 | 1.000 |
| search p50 (ef=96) | **0.53 ms** | 36.6 ms | ~7 ms |
| ingest throughput | 1,837 docs/s | ~200 docs/s | ~250–310 docs/s |

On comparable hardware, Veda's dense search is **roughly 13× faster than Elasticsearch** at
equal recall. The 36.6 ms figure on the contended host wasn't an index problem — direct
instrumentation showed the HNSW graph visiting only ~1.5% of nodes per query (healthy, not
near-exhaustive); the gap was memory-bandwidth contention from other workloads sharing that
box. **Always benchmark on dedicated hardware** — the same code can look dramatically
different depending on what else is running on the machine.

## Where Veda is categorically ahead

Elasticsearch's architecture makes these hard to bolt on; Veda was designed around them:

- **One vector space for every modality.** Text, image, audio, and video share a single
  1024-dimensional embedding space — index with text, query with an image, get the audio
  clip back. Elasticsearch has no native audio/video retrieval; non-text is handled as
  opaque attachments, not first-class searchable vectors.
- **Any document format, one API call.** Drop in a PDF, DOCX, PPTX, XLSX, or a scanned
  image — Veda runs OCR, extracts tables and images, chunks, embeds, and makes it searchable
  in one request. Elasticsearch has no built-in document pipeline; you bolt on Tika or
  Unstructured and feed pre-extracted text back yourself.
- **RAG quality, not just retrieval.** Faithfulness/citation scoring, an eval harness,
  chunker A/B testing, orphan-chunk / coverage detection, a semantic response cache, and
  embedding-based guardrails ship with the engine. Elastic's Agent Builder is explicitly
  text-centric and doesn't score faithfulness, quality, or drift.
- **Vector-native ops intelligence.** Because Veda owns the embedder, its observability is
  semantic: embedding-based log categorization (not template matching), semantic incident
  correlation, spike attribution, and LLM-grounded root-cause analysis — versus Elastic
  AutoOps's rule-based/statistical approach.
- **Self-hosted, private, LLM-agnostic.** Embedder, OCR, reranker, and LLM endpoint are all
  swappable and self-hostable — the whole stack runs air-gapped, for regulated or
  sovereignty-bound deployments.
- **11 tunable ANN/quantization index types** (HNSW, DiskANN, IVF-PQ, RaBitQ family, SQ4,
  SQ8) with an explicit recall/latency dial, versus Elasticsearch's more opaque internals.

## Where Elasticsearch is still ahead

We'd rather tell you this than have you find out the hard way: Elasticsearch wins
decisively on **maturity, operational scale, and full-text/log depth** — petabyte-scale log
analytics, decades of Lucene analyzer richness, Beats/Logstash/APM agents, a huge ecosystem,
enterprise support, and compliance certifications. Veda is young. **Don't replace a mature
Elasticsearch log-analytics deployment with Veda** — that's not the target. Veda's lane is
multimodal ingestion and RAG, where Elasticsearch fundamentally cannot follow, plus now
comparable-or-better dense-vector performance on equal hardware.

## Try it yourself

The fastest way to evaluate this is on your own data: bring a folder of mixed PDFs, images,
and audio, run them through [the document pipeline](/docs/document-pipeline), and search
across all of them from one query box. See the [quick start](/docs/quickstart) to get running
in a few minutes.

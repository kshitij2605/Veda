# SOTA refresh — indexes, quantization, storage, GPU

**Research date:** 2026-08-12. Updates
`research/2026-05-19-sota-audit/{01-vector-indexes,02-quantization,08-storage-gpu-system}.md`.

**Veda baseline verified in-repo today** (`crates/veda-core/src/types.rs`,
`crates/veda-storage/src/segment.rs`):

- 11 vector/quantization index types: `Hnsw, DiskAnn, IvfPq, Flat, RaBitQ, RaBitQRerank,
  HnswRaBitQ, Cagra, RaBitQExtRerank, Sq4, Sq8` + 5 scalar (`SuffixArray, Inverted, Range,
  Bitmap, Hash`)
- ACORN filtered traversal present in `crates/veda-index/src/hnsw/searcher.rs`
- Stored-fields formats: **`Jsonl` and `Columnar` only** — no Lance, no Vortex

---

## 1. Quantization — Veda has caught up; the lead is gone

The May audit said *"RaBitQ 1-bit still SOTA recall-per-bit"* and listed Veda as **ahead** for
shipping it early. That is no longer true — **RaBitQ is now the industry standard**, shipped in
**Milvus 2.6, Faiss, VSAG and Elasticsearch**
([Milvus](https://milvus.io/blog/turboquant-rabitq-vector-database-cost.md)).

The May audit's gap — *"Missing the Extended-RaBitQ multi-bit ladder"* — is **CLOSED**
(`RaBitQExtRerank`). It closed just as the whole field moved to multi-bit:

| Vendor | Multi-bit quantization | Date |
|---|---|---|
| Elasticsearch | **BBQ 2/4/7-bit** (was 1-bit only); DiskBBQ **auto-calibration** of depth/preconditioning/oversampling | 9.4 (2026-05-05), 9.5 (2026-08-04) |
| Qdrant | **TurboQuant** 8× compression; then **4-bit as a primary-storage datatype** (drop the originals) | 1.18.0 (2026-05-11), 1.19.0 (2026-08-05) |
| Weaviate | **RQ4** — 4-bit rotational quantization | 1.39.0 (2026-08-04) |
| Milvus | sparse-index quantization + inverted-list compression, ~3× smaller | 3.0 (2026-07-27) |
| **Veda** | `RaBitQExtRerank` (Extended-RaBitQ ladder), `Sq4`, `Sq8` | shipped post-audit |

Extended RaBitQ itself: "supports quantization at arbitrary compression rates… accuracy
dominantly better than SOTA scalar quantization at the same compression rate". Reported
guidance: 4/5/7-bit suffices for ~90/95/99 % recall without reranking; multi-bit hits 96 %
recall without refine overhead
([DEV](https://dev.to/gaoj0017/extended-rabitq-an-optimized-scalar-quantization-method-83m),
[RaBitQ Library](https://vectordb-ntu.github.io/RaBitQ-Library/)).

**New since May: `IVF-RaBitQ`** — a **GPU-native** IVF + RaBitQ combination for scalable ANN,
reported to beat OPQ-IVF and HNSW on QPS–recall at 80–100 % recall. Veda has `IvfPq` and
`RaBitQ` separately and `Cagra` for GPU, but **not** this combination.

> **Naming caution.** Qdrant ships quantization branded *TurboQuant*. The **TurboQuant paper**
> (Google, ICLR 2026) is a *KV-cache* compression method for LLM inference (3-bit keys / 2-bit
> values, ~6× KV reduction), a different problem from persistent vector indexes. Do not treat
> Qdrant's feature and the paper as interchangeable when writing comparisons.

**Verdict: PAR.** Veda is no longer ahead on quantization; it is level with the field, and its
one distinctive combination (`HnswRaBitQ` deep-quantized graph) is matched in spirit by Qdrant's
4-bit-as-primary-storage.

---

## 2. Vector indexes — CAGRA is now table stakes, filtered ANN is the live frontier

### GPU / CAGRA — gap CLOSED, and it was closed just in time

The May audit's **#1 gap** was *"No first-class GPU index"*. Veda now has `IndexType::Cagra`.
The field arrived at the same place:

| System | GPU status | Numbers |
|---|---|---|
| FAISS | v1.10 + cuVS; `GpuIndexCagra`; `faiss-gpu-cuvs` 1.14.1 (Mar 2026) | IVF build ↑4.7×, search ↓8.1×; CAGRA build ↑12.3× vs CPU HNSW, search ↓4.7× |
| Elasticsearch | cuVS indexing **GA** in 9.4 (preview 9.3) | ↑12× indexing throughput, 7× faster force-merge |
| OpenSearch | preview 3.0 → **GA 3.1** | ↑9.3–9.5× indexing, ~3.75× lower cost vs CPU |
| Milvus | GPU indexes ongoing | — |
| Qdrant | GPU-accelerated indexing (Cloud) | — |
| **Veda** | `Cagra` behind the `gpu_cuvs` Cargo feature; rejected at create-time when cuVS is absent | not benchmarked in-house (CHANGELOG 0.5.1 records `cagra` SKIP — no cuVS in that build) |

**Verdict: PAR on paper, BEHIND in evidence.** Everyone ships cuVS/CAGRA. Veda's is
compile-time gated and has **never been validated end-to-end on this hardware** — the 0.5.1 E2E
explicitly skipped it. That is a claim we currently cannot defend in a POC.

### Filtered ANN — the live research frontier

ACORN (which Veda now implements) has become one of a family: **NHO, ACORN, UNG, RWalks,
Filtered-DiskANN**. 2026 brought **RACORN-1** — "adaptive recall-preserving speedup for
low-selectivity filtered vector search" ([arXiv 2607.00768](https://arxiv.org/pdf/2607.00768)) —
plus a hardness-controlled benchmark generator ([arXiv 2606.14193](https://arxiv.org/pdf/2606.14193))
and a dedicated **ACM SIGMOD programming contest** on filtered ANN.

Meanwhile Elasticsearch 9.4 reports **≥3× lower latency on restrictive filters** — the same
problem, solved at the engine level.

**Verdict: PAR.** Veda's ACORN adoption was the right call and is current. The next move in this
lane is RACORN-1-style adaptivity for low-selectivity filters, not another index type.

### What the May audit listed as behind — status

| May gap | Status today |
|---|---|
| No first-class GPU index | **CLOSED** (`Cagra`) — but unvalidated on real hardware |
| DiskANN/IVF-PQ on object storage (SPANN-class) | **STILL OPEN.** And now more urgent: the whole category moved to object storage |
| No ACORN filtered traversal | **CLOSED** |
| Storage format JSONL | **STILL OPEN** — see §3, now the single clearest technical gap |
| PLAID engine internals (WARP) | **CLOSED** (see `04`) |

---

## 3. Storage format — the gap widened

This was the May audit's *"single biggest performance footgun in the storage layer."* It is now
worse, because the format question got answered by the market while Veda stood still.

### The 2026 format landscape

| Format | Owner | Design centre | Vector index in-format? |
|---|---|---|---|
| **Parquet** | Apache | the estate / BI source of truth | no |
| **Lance** | LanceDB (Rust) | **fast random access**, zero-copy versioning, **native vector index** | **yes** |
| **Vortex** | Linux Foundation / Spiral | BtrBlocks-inspired encodings, **compression + fast scans**, zero-copy Arrow | **no — deliberately** |
| Nimble, BtrBlocks, FastLanes | various | research/adjacent | no |

The stated deciding factor: **random-access ML serving → Lance; compression, scans and
lake-native serving → Vortex**. Lance "bundles the index with the format… the storage layer and
the retrieval layer are one decision." Vortex is "a credible prototype rather than a proven
estate" as of 2026 — engine integrations are young.

The architecture most teams land on in 2026: **Iceberg-governed Parquet as source of truth,
a Lance dataset derived from it serving retrieval and training, Arrow as interchange.**

Sources: [Zilliz Lance vs Vortex](https://zilliz.com/ai-faq/lance-vs-vortex) ·
[File Format Renaissance](https://iceberglakehouse.com/posts/file-formats-renaissance/) ·
[Dremio](https://www.dremio.com/blog/exploring-the-evolving-file-format-landscape-in-ai-era-parquet-lance-nimble-and-vortex-and-what-it-means-for-apache-iceberg/) ·
[lance discussion #3130](https://github.com/lance-format/lance/discussions/3130)

### Who adopted what

- **Milvus 3.0**: Loon storage engine with **Vortex as default format**; reads Parquet / Lance /
  Iceberg / Vortex **in place, zero-copy** via External Collections.
- **LanceDB**: Lance format **2.2** — Blob V2, nested schema evolution, native Map type, more
  compression; DuckDB SQL over Lance.
- **Elasticsearch 9.5**: **Columnar Mode** (tech preview) — column store, no inverted index by
  default.
- **Veda**: `Jsonl` (legacy default) + `Columnar`.

**Verdict: BEHIND, and the gap widened.** In May this was a performance footgun. Now it is an
*interoperability* gap: competitors can point an index at data already sitting in a customer's
lakehouse, with no copy. Veda requires ingestion into its own format. For a buyer with an
existing Iceberg/Parquet estate this is the difference between "point it at your data" and
"re-ingest your corpus".

---

## 4. Object-storage-native architecture

The May audit called Milvus's Woodpecker "the architecturally distinctive 2026 move". It is now
the category default, and Woodpecker itself graduated to an independently deployable service in
Milvus 3.0. Turbopuffer proves the economics commercially ($100 M ARR, 1 T+ documents);
S3 Vectors sets the price floor from the hyperscaler. See `02`.

**Verdict: BEHIND.** Veda's MinIO/S3 backend is blob storage for originals, not an
object-storage-native index tier. This is the deepest architectural gap in the list, and the one
with the longest lead time — it is a rearchitecture, not a feature.

---

## Cross-cutting: recommendation order for this file's scope

1. **Validate `Cagra` on real GPU hardware** and publish a number. The index exists but the one
   E2E that tried it skipped it. Cheapest credibility win available.
2. **Adopt Lance as a third `StoredFieldsFormat`.** Still the top storage recommendation from
   May, now with an added interop argument: it is the format Milvus 3.0 reads natively.
3. **Scope an object-storage-native index tier** (SPANN-class centroid + posting lists on S3).
   Long lead time — start the design discussion now rather than the implementation.
4. **Watch RACORN-1** for low-selectivity filtered search; Veda's ACORN is current, so this is a
   monitor-not-act item.
5. **Consider IVF-RaBitQ** as a GPU-native quantized index once `Cagra` is validated.

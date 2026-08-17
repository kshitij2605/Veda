# Veda — new features for the website (handoff brief)

**Prepared:** 2026-08-15
**For:** the agent updating `website/` (marketing site + `/docs` site)
**Scope:** everything shipped since the website was last refreshed, plus one
feature still in flight. Nothing here is aspirational — each item names the
design doc and the code path it rests on.

---

## 0. How to use this document

Every feature below has three blocks:

- **What it is** — plain-language, safe to paraphrase on a marketing page.
- **Claims you may make** — statements that are backed by measurement or by
  code. Numbers are quoted with the conditions they were measured under. **Do
  not quote a number without its conditions.**
- **Do NOT claim** — things that are false, unmeasured, or true only under
  conditions a reader would not assume. This block exists because the previous
  positioning doc accumulated claims nobody could trace; treat it as binding.

Where a feature has a documented limitation that a prospective user would
reasonably care about, it is listed. Publishing the limitation alongside the
capability is the point — a limitation discovered after adoption costs far
more than one read on the docs page.

**Current website coverage** (measured 2026-08-15 by grepping `website/src`):

| Feature | Marketing site | Docs site |
|---|---|---|
| Retrieval guardrails | absent | one passing mention in `rest-api.md` |
| Block-Max WAND | absent | absent |
| External Parquet collections | absent | absent |
| CAGRA GPU index | absent | absent (correct — unreleased) |

So items 1–3 are net-new website content, not edits.

---

## 1. Retrieval guardrails — shipped in v0.6.0

**What it is.** Veda screens *retrieved documents* for hostile instructions
before they reach an LLM. This is a different threat from prompt injection in
the user's message: the attack text lives inside the corpus — a PDF, a scraped
page, a vendor deck — and arrives through retrieval. Input/output guardrails
(v0.5.0) cannot see it, because by the time the text is in the context window
it looks like legitimate retrieved content.

Every chunk is screened at ingest against its already-computed embedding and
stamped with four reserved fields. Every LLM-facing surface — `/converse`,
`/converse/stream`, MCP `tools/call`, `/agent-tools/{id}/execute`,
`/a2a/{agent_id}` — drops hostile chunks before the model sees them and
attaches a `retrieval_guard` report.

**Claims you may make.**

- Screens retrieved documents, not just user input and model output. This is
  the OWASP 2026-07-12 indirect-injection vector.
- Enforcement happens at the tool-dispatch choke point, so it covers every
  LLM-facing surface at once rather than being bolted onto each endpoint.
- Screening reuses the embedding Veda already computed for the chunk, so it
  adds no extra model call on the ingest path.
- Calibrated against a real corpus: 10 PDFs / 149 chunks of known-clean
  Japanese business documents. The semantic layer flagged 0 of 149 false
  positives, with the nearest miss at 0.354 against a 0.60 threshold.
- Fails closed on an unrecognised or missing category on a flagged chunk.
- Raw `POST /search` annotates but never drops, so retrieval quality stays
  measurable and the security layer cannot silently degrade recall.
- `VEDA_RETRIEVAL_GUARD=off` degrades to annotate-only. The switch can only
  weaken enforcement, never enable it by accident.

**Do NOT claim.**

- ~~"Blocks all prompt injection"~~ — it targets *retrieved-document*
  injection. Input-side injection is the separate v0.5.0 feature.
- ~~"Every record is screened"~~ — plain `POST /v1/collections/{name}/insert`
  takes client-supplied vectors, so there is no Veda-computed embedding to
  screen and those records are stamped `skipped`. **This includes the Python
  SDK's normal client-side-embedding flow.** Screening happens on
  `process-document`, `ingest-url` and `embed-insert`.
- ~~"PII is blocked"~~ — PII is detected and reported, then **passed
  through**. This was deliberate: live calibration found PII-only chunks (a
  contact detail in an otherwise-benign deck) being dropped identically to
  attacks, and silently deleting a real phone number from an answer was judged
  the worse failure. Say "detected and reported", never "blocked".
- ~~"Works on existing data automatically"~~ — chunks indexed before v0.6.0
  carry no verdict, report as `unscreened`, and pass through. Enforcement
  fails open there by design, because treating unstamped chunks as hostile
  would empty every existing collection on upgrade. Closing that gap requires
  running `POST /v1/admin/collections/{name}/rescan-guardrails`.

**Limitations worth publishing.** The re-scan is a manual admin action, not
schedulable — each run materialises a full copy of the collection and
superseded copies are not reclaimed, so repeated re-scans grow storage.
Temporal conflict detection is active on `filter_query` and inert on `search`.

**Source:** `documentation/2026-08-12-retrieval-guardrails-design.md`

---

## 2. Block-Max WAND pruning for BM25 — merged, unreleased

**What it is.** Keyword (BM25) search on sealed segments now skips blocks of
the posting list that provably cannot contain a top-k result, instead of
scoring every document that contains any query term. The classic Ding & Suel
technique; per-term block-max impact bounds are computed when a segment opens.

**Claims you may make.**

- **2.01x–2.53x faster** per-query BM25 than exhaustive scoring — measured on
  synthetic corpora of 10,000 and 100,000 documents. Always quote the corpus
  sizes; this is not a claim about arbitrary scale.
- **Rank-safe by default.** At the default setting, results are byte-for-byte
  identical to exhaustive scoring. Verified by comparison, not asserted.
- No index format change and no migration. Block metadata is derived when a
  segment opens.
- Opt-in aggressive mode via `prune_theta` on a `bm25` query, trading
  rank-safety for more latency reduction.
- Tie-breaking is now deterministic: equal scores order by document id, so
  repeated identical queries return identical orderings.

**Do NOT claim.**

- ~~"2.5x faster search"~~ unqualified. It is BM25 keyword search only —
  not vector search, not hybrid end-to-end.
- ~~"All BM25 queries are faster"~~ — growing (unsealed) segments carry no
  block metadata and use the existing exhaustive path unchanged. Freshly
  ingested data below the seal threshold sees no speedup.
- ~~"prune_theta below 1.0 is more aggressive"~~ — backwards, and the API
  rejects it with `400`. `1.0` (or omitted) is exact; **above** `1.0` prunes
  harder and is not rank-safe.

**Source:** `documentation/2026-08-13-block-max-wand-design.md`

---

## 3. External Parquet collections (read-in-place) — merged, unreleased

**What it is.** Register a collection over a customer's existing Parquet files
and query them **directly — no ingestion, no copy, no duplicate storage**. The
schema, column types and vector dimension are inferred from the Parquet footer
at registration and validated up front.

This is the "point Veda at your data lake" story, and it is probably the
strongest net-new marketing item in this batch.

**Claims you may make.**

- Query Parquet in place: no ingestion step, no second copy of the data.
- Schema, types and vector dimension inferred from the Parquet footer and
  validated at registration — a missing column, a wrong Arrow type, a
  dimension disagreement across files, or an unreadable URI all fail
  immediately with `400` rather than at query time.
- **Exact search, recall 1.0** — linear scan using the same SIMD distance
  kernels as internal collections.
- Scalar filtering uses the *same* filter evaluator as internal collections,
  so external and internal filtering agree by construction rather than by
  convention.
- Row-level security is enforced identically to internal collections.
- Freshness: every query re-stamps the source and reloads on change, so a
  rewritten source file is never served stale. Concurrent reloads are
  single-flighted.
- Works against local files or any S3-compatible store (MinIO, AWS S3).

**Do NOT claim.**

- ~~"Full Veda functionality on external data"~~ — it is deliberately
  **read-only**. 18 mutating endpoints return `405`, including insert, delete,
  reindex, seal, optimize, process-document and the tuning campaign routes.
- ~~"All query types supported"~~ — `bm25`, `match_phrase`, `sparse`,
  `hybrid` and `rerank` all return `400`. Parquet carries no inverted index
  and no sparse index. **Dense vector search and scalar filters only.**
- ~~"Scales to any size"~~ — search is a linear scan and the vector column is
  memory-resident. Budget roughly **3 GB per 1M rows at 768 dimensions**.
- ~~"As fast as an indexed collection"~~ — it trades speed for exactness and
  zero ingestion. Say that plainly; it is an honest and attractive trade.

**Limitations worth publishing.** `explain: true` and `truncate_to` are
silently ignored rather than rejected. A dense query's `field` is not
validated against the registered `vector_field`. A hit's row id appears both
as `SearchHit.id` and again inside `fields`. External hits carry no
`_guardrail_class` annotation. Over S3/MinIO, reads are whole-object.

**Source:** `documentation/2026-08-14-external-parquet-collections-design.md`

---

## 4. CAGRA GPU index — IN FLIGHT, DO NOT PUBLISH YET

**Status: not merged, and the benchmark has not produced numbers.** Included
so you know it is coming; **put nothing about it on the website in this
pass.**

Veda's in-binary GPU CAGRA index (NVIDIA cuVS) previously did not compile at
all — it was never built by anyone, and CI only ever compiled a different GPU
feature. It now compiles, builds, searches, persists and is safe to share
across threads, with GPU tests passing on RTX 6000 Ada hardware.

Note that `cagra` already appears in some existing material as one of Veda's
index types. That was accurate as an enum variant and is **rejected at
collection-create with a 400** on builds without cuVS — so no existing claim
is broken, but do not upgrade it to a working-feature claim yet.

Publish only once: the branch is merged, and the recall-qualified benchmark
has produced build time, search latency **and recall@10** together. A latency
number without its recall is meaningless for an ANN index and must not ship.

**Source:** `documentation/2026-08-15-gpu-cuvs-cagra-design.md`

---

## 5. Corrections to existing website content

Found while auditing, unrelated to the features above:

- `website/src/content/features.ts` claims **"Nine index algorithms."** That
  undercounts. `IndexType` in `crates/veda-core/src/types.rs` has 11 *vector*
  index types — `Hnsw`, `DiskAnn`, `IvfPq`, `Flat`, `RaBitQ`, `RaBitQRerank`,
  `HnswRaBitQ`, `RaBitQExtRerank`, `Sq4`, `Sq8`, `Cagra` — plus separate
  scalar index types (`Inverted`, `Range`, `Bitmap`, `Hash`, `SuffixArray`)
  which are a different category and should not be added to the same count.

  **Use "10 CPU vector index types"** as the safe public number: the v0.5.1
  campaign validated exactly those 10 end-to-end through the API, with
  measured recall@10 vs exact search (flat / hnsw / diskann / rabitq_rerank /
  sq8 at 1.00, sq4 0.99, ivfpq 0.97, rabitq / hnsw_rabitq / rabitq_ext_rerank
  0.90). The 11th, `cagra`, is GPU-only and is **rejected at collection-create
  with a 400** on builds without cuVS — so do not include it in a headline
  count until §4 lands.
- `features.ts` lists **"Semantic LLM cache & guardrails."** That entry
  describes the v0.5.0 input/output guardrails. It should either be updated to
  cover retrieval guardrails as well, or a separate entry added — the
  retrieved-document threat is a distinct and more differentiating story.

---

## 6. Suggested docs-site pages

`website/src/docs/content/` currently has ~24 pages and a sidebar. Suggested
additions, following the existing page conventions:

| Page | Covers |
|---|---|
| `retrieval-guardrails.md` | §1 — the threat, what is screened, the `retrieval_guard` report, the re-scan endpoint, the `/insert` gap |
| `external-collections.md` | §3 — registration, the read-only contract, supported query types, sizing |
| `search.md` (edit) | §2 — `prune_theta` and rank-safety on BM25 |

`documentation/openapi.yaml` was updated in the same pass as this brief and now
documents `prune_theta` and the full `RetrievalGuardReport` schema — use it as
the reference for exact field names and semantics rather than transcribing
from this document.

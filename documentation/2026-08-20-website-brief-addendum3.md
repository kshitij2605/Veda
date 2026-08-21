# Veda — website handoff brief, addendum 3

**Prepared:** 2026-08-20
**For:** the agent updating `website/` (marketing site + `/docs` site)
**Supplements:** `documentation/2026-08-15-website-feature-brief.md` — same rules apply.
Read section 0 of that document first. Every claim below is backed by a measurement whose
conditions are quoted; **do not quote a number without its conditions.**

Sections continue the numbering from addendum 2 (`2026-08-18-website-brief-multihop.md`).

**Two corrections in here are to text already live on the site.** §14 fixes an index count that
is wrong in four places and was self-contradictory before this cycle; §13 stops the RaBitQ family
being presented as uniformly usable. Do those two even if nothing else in this addendum ships.

---

## 11. Object-storage-native index tier — merged, unreleased

**This closes the last open gap from the 2026-08-12 competitive refresh.** That refresh listed
three: sparse execution, lake storage format, and an object-storage-native index tier. The first
two had already shipped (Block-Max WAND, external Parquet read-in-place) — see §16 for why that
matters to how we talk about the roadmap.

**What it is.** A new index type, `"index": "spann"` on a vector field. Centroids stay resident;
posting lists live in the configured object store and are fetched by byte range, so a query reads
only the lists it probes.

**The distinction to draw carefully, because it is the whole point.** MinIO has shipped in the
core compose since day one and holds **blobs** — images, PDFs, media. The *index* was still built
and searched locally, so a node had to hold it. This is the index itself living in object storage.
Do not let these blur together in copy; "we already had MinIO" is exactly the confusion to avoid.

**Measured** (release build, clustered corpus, 768 dimensions — the dimension the product
embeds at):

| corpus | index size | resident | recall@10 | read per query |
|---|---|---|---|---|
| 10,000 vectors | 30.8 MB | 0.31 MB (**1.01%**) | 0.950 | **2.96%** |
| 50,000 vectors | 153.8 MB | 0.69 MB (**0.45%**) | 0.912 | 3.83% |

The corpus grows 5× and the resident *share* falls. Resident memory tracks the centroid count,
not the corpus.

**Safe to say:** the index lives in object storage; a node holds ~0.5–1% of it; recall 0.950 while
reading under 3% of the index per query, at 768d on a clustered corpus.

**Do NOT say:** that it makes ingest cheaper. **Build is not incremental** — the build still
receives the whole batch in memory, so peak *build* memory is unchanged from any other index
type. The property is about serving. Streaming the build is named as the next step and is not
done.

> **Update 2026-08-21.** A streaming build now exists and is measured: peak build memory 4.5% of
> the corpus at 768d, producing a byte-identical index. The claim above still holds for anything
> customer-facing, because the path is **opt-in** — `IndexType::Spann` still builds through the
> in-memory route, so ingest cost is unchanged for anyone selecting `spann` on a collection. Say
> ingest is cheaper only once a segment-backed `VectorSource` is wired into `build_indexes`.

**Do NOT publish latency comparisons for it.** The exact-scan baseline moved 100× between two
runs on the build host. Only the recall and byte figures above are trustworthy.

Code: `crates/veda-index/src/spann.rs`. Enabling primitive: `BlobStore::get_range`.

---

## 12. Field semantics, and sensitivity enforced at retrieval — merged, unreleased

**Two things, and only the second is a differentiator.**

**The first is catch-up, and should be framed as such.** Collections can now carry per-field
`description`, `semantic_type`, `value_domain` and `sensitivity`. Milvus has had per-field
descriptions since v2.2.x, Weaviate has them and makes them updatable on a live collection, and
Elasticsearch has per-field `meta` with a standardised `unit`/`metric_type` convention. **Do not
present field descriptions as something Veda invented.** If the site currently implies our schema
is more expressive than competitors', that is wrong.

**The second has no equivalent anywhere surveyed.** `sensitivity: "restricted"` or `pii: true` on
a field is **enforced at retrieval**: the field is withheld from non-admin callers on every path —
`search`, `filter_query`, `get_record`, `aggregate` — and a filter or aggregation naming it is
refused rather than silently answered. That last part matters: a caller who cannot see `salary`
could otherwise binary-search it with `salary > 100000` and read the answer off the result count.

Veda can do this because it owns ingestion, embedding and retrieval. A bring-your-own-vector-DB
stack structurally cannot. **That is the claim worth making**, and it is the same structural
argument the 2026-08-12 refresh made about RAG-specific security.

**Measured** — 4 cases × 5 repeats × 2 arms, against Qwen3.5-122B, on a schema whose stored
encodings differ from how the question phrases them (`ctry_cd` holds ISO-2 codes, `amt_2` holds
cents):

| | correct | used the wrong plaintext guess |
|---|---|---|
| unannotated schema | 7/20 (35%) | 11/20 |
| annotated schema | **19/20 (95%)** | **0/20** |

**Quote the conditions with this or not at all.** Five records, four questions, one model, and
field names chosen to be maximally opaque. The honest claim is *"annotations help when the schema
is not self-describing"* — **not** "annotations add 60 points". A schema whose fields are already
named `country` and `amount_usd` should be expected to show little gain.

Code: `crates/veda-core/src/schema.rs`, `crates/veda-gateway/src/agent/field_guard.rs`.
Evidence: `documentation/annotation-eval/gate_a.py`.

---

## 13. RaBitQ — correction to live copy

`website/src/docs/content/introduction.md` advertises "three RaBitQ variants" and
`vs-elasticsearch.md` lists a "RaBitQ family", both without qualification. That is now known to
be misleading for one of them.

**`IndexType::RaBitQ` — the pure 1-bit scan with no rerank — scores ~2% recall@10 on a clustered
corpus** at both 128d and 768d. This is an information limit, not a tuning problem: one bit per
dimension cannot order points inside a tight cluster, because the distance-estimate error is a
large fraction of the entire intra-cluster spread. A k-means residual codebook was built and
measured: it cut reconstruction error 20× and left recall unchanged.

`RaBitQRerank` is fine — **recall 1.000** on the same corpus after this cycle's fix to the
candidate depth (`DEFAULT_RERANK_MULTIPLIER` 10 → 100, which was previously returning a quarter
of the true neighbours). `Sq8` scores 0.94.

**Action:** wherever the RaBitQ variants are listed, do not present the bare scan as
interchangeable with the rerank variants. The index reference should say plainly that the pure
scan is for uniform-ish corpora and that clustered data wants `RaBitQRerank` or `Sq8`. The
type's own doc comment now says this.

Evidence: `documentation/2026-08-20-rabitq-clustered-finding.md`.

---

## 14. The index count on the site is wrong in four places

Authoritative, counted from the `IndexType` enum today: **12 dense vector index types** — Hnsw,
DiskAnn, IvfPq, Flat, RaBitQ, RaBitQRerank, HnswRaBitQ, Cagra, RaBitQExtRerank, Sq4, Sq8, Spann.
(Plus SuffixArray and the scalar index kinds, which are a separate category and should not be
folded into the vector-index number.)

Live copy says:

| file | says |
|---|---|
| `website/src/sections/Architecture.tsx:29` | "Ten vector index types" |
| `website/src/content/features.ts:53` | "10 vector index types" |
| `website/src/docs/content/introduction.md:10` | "10 vector index types" + an explicit list |
| `website/src/docs/content/vs-elasticsearch.md:57` | "11 tunable ANN/quantization index types" |
| `website/src/docs/content/vs-pinecone.md:43` | "11 index types" |

These already disagreed with each other before this cycle. Fix all five to 12, add Spann to the
explicit list in `introduction.md`, and — since this has now drifted twice — consider whether the
number belongs in prose at all, or whether the index reference page should be the single place it
appears.

---

## 15. What is safe to publish in this pass — summary

**Publish:**
- The object-storage index tier (§11), with the residency and read-fraction figures and their
  conditions, and **without** any claim about build cost or latency.
- Field sensitivity enforced at retrieval (§12), as a structural advantage. Field descriptions
  themselves as parity, not as a lead.
- The corrected index count (§14).

**Correct:**
- The RaBitQ framing (§13).
- The index count (§14).

**Do not publish:**
- Any build-time or latency number measured on the build host this cycle. The DiskANN build
  measured 382.8s where 2026-08-19 recorded ~509s for identical work; neither run had exclusive
  use of the machine. Treat build times from that host as ±30% unless both arms ran back to back.
- The annotation figure without its conditions (§12).
- Anything about answer-quality auto-tuning. A new `answer_correctness` objective was added to
  the tuning campaign this cycle, but it is opt-in, unmeasured on a real corpus, and nothing has
  yet been tuned with it. It is not a feature until it has a number.

---

## 16. Outstanding, and not mine to decide: `competitive_advantages.md`

The 2026-08-12 refresh's **recommendation #1** — "zero-engineering, highest return" — was to
rewrite `competitive_advantages.md` around integration rather than exclusivity, because claims
#1, #2, #5 and #7 are false or weakened as written and the benchmark table cites a superseded
Elasticsearch version. **That has not been done.** The document still opens dated 2026-07-02 and
still carries the "roughly 13× faster than Elasticsearch" figure at line 95, which the refresh
said to stop repeating until it is re-run against ES 9.5.

Two things have changed since that recommendation, and both make it more urgent rather than less:

1. **Two of the three gaps that refresh identified are now closed** (sparse execution, lake
   storage) and the third closed this cycle (§11). Positioning written against a three-gap
   roadmap is stale in our favour, which is the more embarrassing direction.
2. This cycle added a structural security claim (§12) that the current document does not mention.

This is a positioning decision rather than a website edit, so it is flagged here rather than
acted on. It needs someone to decide what Veda leads with now that the "we can do what they
can't" framing has been overtaken by "we integrate what they've each got".

# Veda — website handoff brief, addendum 2

**Prepared:** 2026-08-18
**For:** the agent updating `website/` (marketing site + `/docs` site)
**Supplements:** `documentation/2026-08-15-website-feature-brief.md` — same rules apply.
Read section 0 of that document first. Every claim below is backed by a measurement whose
conditions are quoted; **do not quote a number without its conditions.**

**Two items in the 2026-08-15 brief are now out of date. Sections 8 and 9 below supersede
them.** Do not act on the older text for CAGRA or for index recall numbers.

---

## 7. Multi-hop agentic retrieval — merged, unreleased

### What it is

Veda's agent loop answers questions whose answer is not in any single document. Asked "is the
plant that assembles Product Halcyon certified to ISO 14001?", it searches for the plant that
assembles Halcyon, then searches again for that plant's certifications, and combines the two.
The agent decides how many searches to run; nothing is pre-declared.

This is ordinary tool-calling over Veda's own `search` tool, not a separate subsystem. There is
no knowledge graph, no query planner, and no new index.

### Claims you may make

Measured 2026-08-18 on a **2,012-document corpus** — 12 fact documents buried in 2,000 filler
documents sharing the same business vocabulary, so no single search can return both halves of an
answer. Qwen3.5-122B as the reasoning model, `jina-v5-omni-small` embeddings, 3 repeats per
question. **The whole suite was run twice on separate days and returned identical figures.**

| question type | correct | mean searches |
|---|---|---|
| single-hop | 9/9 (100%) | 1.0 |
| two-document join ("does X also appear in Y") | 12/12 (100%) | 1.8 |
| one-to-many join ("list *every* supplier where…") | 6/6 (100%) | 2.2 |

- "Veda's agent answers multi-hop questions by issuing follow-up searches, without a knowledge
  graph or a query planner."
- "Measured 100% on single-hop, two-document joins and one-to-many joins over a 2,012-document
  corpus (Qwen3.5-122B, 3 repeats, reproduced on two separate runs)."
- "One-to-many questions — the kind that require finding *all* matches rather than one — were
  answered completely, in about two searches on average."
- The retrieval itself was not the limiting factor in any of these cases.

Quote the corpus size and the model whenever you quote a percentage. A reader who assumes these
figures come from a million-document corpus has been misled.

### Do NOT claim

- **Do not claim any figure for questions whose answer is absent from the corpus.** Those
  measured **3/6 (50%)** — the agent sometimes recites what it found instead of answering
  "none". This is a real, published limitation. If the page describes multi-hop, it should say
  plainly that answering "the corpus does not contain this" is less reliable than answering a
  question that has an answer.
- **Do not repeat the earlier "83% multi-hop" figure** if you find it anywhere. It came from a
  15-document corpus that a single search returned in full, so it measured how well a model
  reads its own context — not multi-hop retrieval. It is invalid and was retracted.
- Do not claim a knowledge graph, GraphRAG, TreeRAG, or query planning. None exists, and the
  measurement gives no reason to build one.
- Do not claim benchmark standing (HotpotQA, MuSiQue, 2WikiMultihopQA). This is Veda's own
  suite on a synthetic corpus; no public multi-hop benchmark has been run.
- Do not describe the number of hops as configurable or guaranteed. The loop is bounded by a
  step limit; how many searches it spends is the model's choice.
- Do not imply latency figures. Multi-hop answers took roughly 2-9 seconds here, but that is
  dominated by an external LLM endpoint and is not a Veda performance number.

**Source:** `documentation/2026-08-17-multi-hop-retrieval-scope.md`; harness and raw results in
`documentation/multihop-eval/`.

---

## 8. CAGRA GPU index — supersedes §4 of the 2026-08-15 brief

**§4 said "IN FLIGHT, DO NOT PUBLISH YET". That is no longer true** — it merged (PR #8), and the
recall-qualified benchmark §4 required has produced numbers. The condition §4 set was "build
time, search latency **and** recall@10 together". That condition is now met, so it may be
published.

Measured on RTX 6000 Ada, clustered corpus, 128 dimensions, k=10, on an otherwise idle host:

| corpus | index | build | search | recall@10 |
|---|---|---|---|---|
| 100k | CAGRA | 16.4s | 1018µs | 1.000 |
| 100k | HNSW | 171.2s | 550µs | 0.990 |
| 1M | CAGRA | 86.2s | 878µs | 0.930 |
| 1M | HNSW | 4184.4s | 4488µs | 0.913 |

**Claims you may make.**

- "CAGRA builds a 1M-vector index in 86 seconds against HNSW's 70 minutes — roughly 49x faster
  — at equal or better recall (0.930 vs 0.913)."
- "CAGRA's search latency is nearly flat as the corpus grows (1018µs at 100k, 878µs at 1M),
  while HNSW's grows 8.2x over the same range."
- Index build is where the GPU pays off most decisively.

**Do NOT claim.**

- **Do not claim CAGRA is faster to search at small corpus sizes.** At 100k, HNSW is 1.85x
  *faster* per query. CAGRA wins on latency only once the corpus is large enough to amortise
  fixed GPU overhead; the crossover is between 100k and 1M here.
- Do not quote latency without recall, and do not quote either without the corpus size.
- Do not present per-query latency as CAGRA's ceiling. These figures come from a one-query-at-
  a-time interface, which is the worst case for a GPU; batched throughput was not measured.
- Do not claim it works without a cuVS build. On builds without cuVS, `cagra` is **rejected at
  collection-create with a 400**.
- The build-time ratio has real error bars — some earlier runs were measured on a contended
  host. The figures above are from an idle host and are the ones to use.

**Source:** `documentation/2026-08-15-gpu-cuvs-cagra-design.md`

---

## 9. Recall numbers are corpus-dependent — read before publishing any recall figure

**This supersedes the recall list in §5 of the 2026-08-15 brief.** That section told you to
publish per-index recall figures (`diskann` 1.00, `rabitq` 0.90, and so on). **Do not publish
those numbers, or any per-index recall number, without checking this section first.**

Veda's index benchmark generated every corpus from uniform-random vectors. In high dimension
those are all roughly equidistant, so recall measured on them describes the corpus more than the
index. Re-measuring the same indexes on a clustered corpus — which is what real embeddings look
like — moved the results **in both directions**:

| index | 10k uniform → clustered | 50k uniform → clustered |
|---|---|---|
| HNSW | 89.3 → 100.0 | 56.8 → 99.1 (was understated) |
| IVF-PQ | 34.4 → 98.0 | 24.3 → 100.0 (was understated) |
| DiskANN | 97.2 → **20.3** | 73.4 → **20.1** (was overstated) |
| RaBitQ | 35.3 → **3.2** | (was overstated) |

**DiskANN at ~20% and RaBitQ at ~3% on representative data are undiagnosed and are flagged as
probable defects.** They are under investigation and are not fixed.

**What this means for the website.**

- **Do not publish a recall figure for DiskANN or RaBitQ in any form right now.**
- HNSW and IVF-PQ recall is *better* on realistic data than the old numbers suggested, so
  nothing published about them is overstated — but still quote the corpus type.
- Any recall claim must state the corpus it was measured on. "recall@10 of 0.99" with no corpus
  is not a meaningful claim, and this is exactly how the old numbers became misleading.
- The safe public index count from §5 — **"10 CPU vector index types"** — is unaffected and
  still correct. Counting index types is not a recall claim.

**Source:** `documentation/2026-08-18-corpus-shape-index-ranking.md`

---

## 10. What is safe to publish in this pass — summary

| item | publish? |
|---|---|
| Multi-hop agentic retrieval (§7) | **yes**, with the negative-answer limitation stated |
| CAGRA GPU index (§8) | **yes** — supersedes the §4 hold |
| Retrieval guardrails, Block-Max WAND, external Parquet (§§1-3 of the 2026-08-15 brief) | yes, unchanged |
| Per-index recall numbers (§5 of that brief) | **no** — see §9 |
| "10 CPU vector index types" | yes |

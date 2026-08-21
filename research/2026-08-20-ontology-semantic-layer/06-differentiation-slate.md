# Differentiation slate

**Research date:** 2026-08-20.

**Admission rule, applied strictly:** every candidate states the measurement that would show it
worked **and** the measurement that would show it did not. A candidate that cannot state its own
disconfirming evidence is not in the table. Two of the four hypotheses this cycle set out to
test failed that rule or failed on evidence, and both are recorded below rather than deleted —
the point of naming hypotheses in advance is that killing them counts as a result.

The ranking is by *(differentiation × evidence) ÷ cost*, not by appeal.

---

## D1. Semantics as a **tuned** artifact, not a documented one

**Rank 1 — the only candidate in this cycle that is not catch-up.**

**Mechanism.** Extend the tier-3 tuning campaign (`veda-gateway/src/tuning/config.rs`) so that
the field annotations rendered into `describe_collection` are a **searched parameter**, scored
against an eval set generated from the user's own corpus
(`rest/rag_eval.rs::generate_eval_set_for_collection`). The campaign proposes candidate
annotations (LLM-drafted from sampled values, as Graphwise's "schema architect" does in `03`),
measures agent task accuracy with and without them, and keeps them only if they pay.

**Why it differentiates.** Per `04`, Milvus, Weaviate and Elasticsearch all let you *write*
field descriptions; not one tells you whether writing them helped. They are documentation
accepted on faith. Veda already owns the generate-eval-set → search-space → Pareto-front loop.
Nothing else surveyed can measure its own semantic layer.

**Reuses.** `tuning/{config,search,campaign}.rs`, `rest/rag_eval.rs`, `ToolCatalog` renderer.
No storage change, no new index, no graph.

**Cost.** Medium. A new tier-3 dimension, an annotation proposer, and a renderer.

| | Measurement |
|---|---|
| **Confirms** | On a corpus with non-obvious field names, agent task accuracy is higher with searched annotations than with none, **and** higher than with naive LLM-drafted annotations that were not scored — the second comparison is the one that isolates *searching* from *annotating* |
| **Disconfirms** | Annotations move accuracy by less than run-to-run variance (`03`/`05` both show the agent loop is nondeterministic — so ≥5 repeats, error bars, before any claim), **or** the searched annotations beat none but do not beat unscored drafts, which would mean the tuning adds nothing and the cheap version wins |

**Precondition, non-negotiable:** instrument how often an annotation is actually *read* by the
model before measuring win rate. The stagnation detector was correct, tested and inert — it
fired in 1 run of 6 — and looked merely ineffective. Same trap here.

---

## D2. Field-declared sensitivity, enforced at retrieval

**Rank 2 — smallest, and the only one with a security story competitors structurally cannot copy.**

**Mechanism.** A `sensitivity` / `pii` declaration on `FieldDefinition`. At retrieval, fields so
marked are redacted or withheld based on the caller's role, using the enforcement path that
already exists for `GuardrailCategory::Pii` and `GuardrailMode::{Flag,Block}`
(`agent/guardrails.rs`, `agent/retrieval_guard.rs`), and the role plumbing already present in
`rls.rs` / `rbac.rs`.

**Why it differentiates.** Per `05`, Veda today re-discovers PII **per record, probabilistically,
at ingest**, for something the operator knows once at schema-design time. Per `04`, no surveyed
product declares sensitivity on the field. Veda owns ingestion, embedding *and* retrieval —
the structural advantage `06-community-demand.md` already identified — so it can enforce where
a bring-your-own-vector-DB stack cannot.

**Reuses.** `rls.rs`, `rbac.rs`, `retrieval_guard.rs`, the `GuardrailMode` enum.

**Cost.** Small. One schema field, one enforcement point, one renderer change.

| | Measurement |
|---|---|
| **Confirms** | A field marked `pii` never appears in agent tool output or `/search` hits for a non-privileged caller, across every retrieval path — `search`, `filter_query`, `get_record`, `aggregate`, `/converse` citations. Plus: **delete the enforcement and confirm the test fails**, per the standing rule that this project has repeatedly shipped tests that pass with the mechanism removed |
| **Disconfirms** | Any path leaks the field, or the declaration is unenforceable on `external` Parquet collections where Veda does not control the write path (`schema.rs:230`) — a real risk that must be checked, not assumed |

**Caveat worth stating before anyone builds it:** the detector still has to run, because the
declaration only covers PII in fields the operator anticipated. This *narrows* the detector's
job; it does not replace it.

---

## D3. Constraint validation (SHACL's idea, without SHACL)

**Rank 3 — real but unproven demand.**

**Mechanism.** Declarative per-field constraints — enumeration, regex, range, required-with —
evaluated at ingest and reportable over existing data, using `veda-query/src/filter/` as the
evaluator. Violations reported, not silently dropped.

**Why it might differentiate.** `01` identifies this as the one genuinely transferable idea from
ontobricks' reasoning layer: a shape is "field X must satisfy predicate P, here is the violation
report", and it needs no graph, no RDF, no OWL. Nobody in `04` has it.

**Reuses.** `filter/{ast,parser,evaluator}.rs`, ingest path.

**Cost.** Small-to-medium.

| | Measurement |
|---|---|
| **Confirms** | On a deliberately dirty corpus, validation catches injected violations with no false positives on clean records — **and** a user-facing signal that this is wanted (see below) |
| **Disconfirms** | No demand signal exists. `06-community-demand.md` ranked 14 asks by evidence and **data validation is not among them.** Schema *evolution* was #2; validation was absent |

**Honest position: this is a solution looking for its demand.** It should not be built on the
strength of being architecturally tidy. Ranked third for that reason, and it needs a demand
check before a design.

---

## Rejected: relations between collections

**Failed on cost-vs-evidence, not on appeal.**

Weaviate has cross-references; Veda's `FilterExpr` has no join and no reference type (`05`). So
this is a genuine capability gap against the closest comparator, and it is the natural home for
"ontology" in reading A.

It is rejected for this cycle because the workload it serves — cross-collection entity joins —
is the workload Veda **already measured at 100%** via the agent loop (one-to-one 12/12,
one-to-many 6/6, `05`). Building a join to serve a saturated metric is the same mistake as
implementing the one-to-many enumeration fix that v2 of the multi-hop work proved was a
non-problem. **Revisit if and when a measured workload fails that the agent loop cannot serve.**

---

## Killed by evidence: two hypotheses this cycle set out to test

**"Field descriptions in `describe_collection` are a cheap differentiating win."**
**Dead.** Per `04`, Milvus has had per-field descriptions since v2.2.x, Weaviate has them and
makes them updatable live, and Elasticsearch has per-field `meta` with a *standardised
convention* for `unit` and `metric_type`. Veda lacking them is **catch-up, not differentiation**.
It is still worth doing — it is a prerequisite for D1 and D2, both of which need somewhere to put
the annotation — but it must be described as closing a gap, never as a lead.

**"Multimodal ontology is open ground because nobody has one."**
**Dead as stated.** M³KG-RAG (CVPR 2026) builds audio-visual multimodal KGs; TIVA-KG covers
text/image/video/audio; the CASTLE challenge work does hierarchical KG retrieval over video
(`03`). The surviving, weaker claim is that this is **research-grade and extractor-bounded** —
which is a cost argument against building it, not a differentiation opportunity. The prior
scoping doc's phrasing ("no good entity extractor for images/audio") is too strong and should be
narrowed where it is quoted.

---

## Ranked summary

| Rank | Candidate | Differentiates? | Evidence | Cost | Verdict |
|---|---|---|---|---|---|
| 1 | **D1** semantics as a tuned artifact | **Yes — uniquely** | Reasoned from `04`; no competitor measures its semantic layer | Medium | Build, gated on a firing-rate check |
| 2 | **D2** field-declared sensitivity | **Yes** — structural advantage | OWASP 2026-07-12 lineage; `04` shows nobody declares it | Small | Build |
| 3 | **D3** constraint validation | Partly | Architecturally sound, **no demand signal** | Small-med | Demand check first |
| — | Relations between collections | Yes, vs Weaviate | Serves a metric already at 100% | Large | Defer |
| ✗ | Field descriptions alone | **No — catch-up** | Milvus/Weaviate/Elastic all ship it | Tiny | Do it, as a prerequisite, not a headline |
| ✗ | Multimodal ontology | No | M³KG-RAG et al. exist | Very large | Drop |
| ✗ | Triple store / OWL / SWRL | No | Serves saturated retrieval | Very large | Drop (`01`, `03`) |

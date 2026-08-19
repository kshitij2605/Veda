# Ontology in the field: what the word means, and who is actually buying it

**Research date:** 2026-08-20. Evidence grades per
`research/2026-08-12-competitive-refresh/06-community-demand.md`:
**A** primary + countable, **B** secondary citing numbers, **C** single-source assertion.

---

## The word is doing three jobs at once

Most of the confusion in "do we need an ontology" is that the term names three different
artifacts with three different costs. Separating them is the single most useful thing this file
does, because the answer differs for each.

| Sense | What it is | Typical cost | Who it serves |
|---|---|---|---|
| **S1 — semantic annotation** | Descriptions, business types, units, value domains, sensitivity classes attached to existing fields | Days | Anything reading the schema: agents, NL→query, docs, governance |
| **S2 — semantic model / composite semantic layer** | Entities, relationships and metrics defined *over* physical storage; consistent business logic across tools | Weeks–months, ongoing | BI, analytics, agents that must agree on "revenue" |
| **S3 — formal ontology** | OWL/RDFS classes with logical axioms, machine inference, SHACL constraints, a triple store | Quarters, plus a standing governance body | Reasoning, regulatory modelling, cross-organisation data exchange |

OntoBricks spans S2 and S3 and includes a slice of S1. Palantir's Ontology is S2 with governed
actions bolted on. Most "we added an ontology" blog posts are S1. **A recommendation that does
not say which sense it means is not a recommendation.**

---

## Gartner (grade A)

Gartner's *Top Trends in Data and Analytics for 2026*
([press release, 2026-06-16](https://www.gartner.com/en/newsroom/press-releases/2026-06-16-gartner-identifies-the-top-trends-for-data-and-analytics),
accessed 2026-08-20) names the leading trends as **AI agents, advances in semantics, and
platform convergence**. Two of the trend documents are directly on point:

- *[Making Composite Semantic Layers Interoperable](https://www.gartner.com/en/documents/7440462)* —
  the framing is that "the goal of a single, universal semantic layer continues to be elusive",
  so leaders are pivoting to a **composite** layer coordinating diverse semantic artifacts.
- *[Handling Complex Use Cases With GraphRAG](https://www.gartner.com/en/documents/7444326)* —
  "most retrieval-augmented generation initiatives fail when high thresholds of accuracy are
  required."
- Prediction: **40% of enterprises will have leveraged GraphRAG techniques by 2029.**

**How to read this, and it is not "build a knowledge graph".** Gartner's own framing is that the
universal semantic layer *failed* and the surviving pattern is composition — many partial
semantic artifacts made interoperable. For a component vendor like Veda, that argues for
**emitting good semantics about your own data** and being consumable, not for becoming the
enterprise's semantic authority. The 40%-by-2029 figure is a Gartner prediction, i.e. grade A
for *what Gartner said* and unverifiable as a fact about the future; do not quote it as
evidence that GraphRAG works.

---

## Palantir Ontology / Ontology-Augmented Generation (grade C)

The commercial reference everyone points at. The claimed mechanism: model every noun,
relationship, rule, action and permission as a governed layer; let agents operate on that layer
rather than on raw data. The slogan is "OAG retrieves typed Ontology objects and triggers
deterministic tools before LLM synthesis", against RAG "retrieving unstructured text chunks".

**Every source found for this is secondary** — Substack posts, DEV articles, vendor-adjacent
analyses. No benchmark, no reproducible comparison, no published methodology. It is graded **C**
and should not appear in Veda customer-facing material as evidence of anything.

Two observations that survive the grading anyway, because they are architectural rather than
empirical:

1. **The value claimed is determinism, not retrieval.** "Trigger deterministic tools before LLM
   synthesis" is a statement about *actions and typed access*, not about finding documents. It
   is not a claim that vector search is inaccurate.
2. **It is inseparable from governance and permissions.** Palantir's ontology is valuable
   because it is the same object model that enforces access control and drives workflows.
   Extracting "the ontology part" and adding it to a retrieval engine keeps the modelling cost
   and drops the mechanism that pays for it.

---

## The honest summary of the field-level evidence

- **S1 is uncontroversial and cheap**, and per `04` most competitors already ship a piece of it.
- **S2 is where the enterprise money and the Gartner attention are**, and it belongs to
  warehouse/BI/platform vendors who own the governed model. Veda is a component beneath that
  layer, not a candidate to be it.
- **S3 has real peer-reviewed results in retrieval** (see `03` — OG-RAG, EMNLP 2025), and those
  results are about *grounding retrieval in an ontology the domain already has*, not about
  authoring one from scratch inside the database.

The recurring rhetorical move in this literature — "ontology beat vector embeddings" — is
comparing an ontology-augmented pipeline against a naive vector baseline. Where the baseline is
a tuned hybrid retriever with reranking, the comparison is not published. Veda's own numbers in
`05` are a tuned baseline scoring 100% on the joins these systems target, which is the single
most relevant datum this cycle has and the one no external source can supply.

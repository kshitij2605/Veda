# Does Veda need an ontology? — consolidated summary

**Research date:** 2026-08-20. Prompted by
[databrickslabs/ontobricks](https://github.com/databrickslabs/ontobricks) v0.7.1 and the question
of what could be built in that direction to differentiate from competitors.

Evidence grades follow `research/2026-08-12-competitive-refresh/06-community-demand.md`:
**A** primary + countable, **B** secondary citing numbers, **C** single-source assertion.
Each file here is a deep dive; this is the verdict.

---

## The answer in one line

**No — not in the sense ontobricks means, and the reason is a Veda measurement rather than a
judgement about ontologies.** But Veda is *behind* on the cheap semantic annotation the field
takes for granted, and there is exactly one ontology-adjacent thing it could do that no
competitor can.

---

## The word means three things, and the answer differs for each

| Sense | Verdict | Why |
|---|---|---|
| **S1 — semantic annotation** (descriptions, types, units, sensitivity on fields) | **Behind. Close it.** | Milvus has had per-field `description` since v2.2.x; Weaviate has collection + property descriptions, updatable live; Elasticsearch has per-field `meta` with a standardised `unit`/`metric_type` convention. Veda has none of it. **Grade A** ([`04`](04-competitor-semantic-surface.md)) |
| **S2 — semantic model / composite semantic layer** | **Not Veda's job.** | Gartner's own 2026 framing is that the universal semantic layer *failed* and the surviving pattern is **composition**. That argues for emitting good semantics and being consumable, not for becoming the enterprise's semantic authority — which belongs to whoever owns the governed model. **Grade A** ([`02`](02-ontology-in-the-field.md)) |
| **S3 — formal ontology** (OWL, triples, inference, a triple store) | **No.** | It targets retrieval, and Veda's retrieval is not what is failing (below) ([`01`](01-ontobricks-teardown.md), [`03`](03-graph-and-memory-layers.md)) |

A recommendation that does not say which sense it means is not a recommendation. Most of the
"ontology beats RAG" discourse silently switches between them.

---

## Why S3 is a no: our own number, not an opinion

`documentation/2026-08-17-multi-hop-retrieval-scope.md` measured the shipped agent loop on a
2,012-document corpus built so that no top-k result set can hold both halves of an answer,
3 repeats per case:

| case type | correct |
|---|---|
| single-hop control | 9/9 (100%) |
| one-to-one join | 12/12 (100%) |
| one-to-many join | 6/6 (100%) |
| **negative (no answer exists)** | **3/6 (50%)** |

Everything the ontology/GraphRAG literature is optimising — entity-join multi-hop retrieval —
is **already at 100%** here. The measured deficit is the model failing to *conclude* that a fact
is absent, which no graph addresses. OG-RAG's +55% and Graphwise's MuSiQue numbers are both
gains on entity-join retrieval against a plain-RAG baseline; Veda's baseline is a tuned hybrid
retriever that is already saturated on that workload.

**This is the whole argument.** Not "ontologies do not work" — they demonstrably do, on the
thing they target. "That thing is not what is broken here."

---

## Scorecard

| # | Area | Status | Headline | File |
|---|---|---|---|---|
| 1 | Ontobricks as a template | **Mostly inapplicable** | 4 of 6 layers have no referent in Veda. R2RML exists to bridge *someone else's governed relational tables*; Veda's collections hold chunks it wrote itself. Ontobricks is downstream of a Unity Catalog governance layer Veda does not have | [01](01-ontobricks-teardown.md) |
| 2 | Field-level evidence | **Split** | Gartner grade A and directionally *against* becoming a semantic authority; Palantir/OAG is grade C with no published methodology | [02](02-ontology-in-the-field.md) |
| 3 | Does ontology-grounding work? | **Yes, on a workload we've saturated** | OG-RAG is peer-reviewed (EMNLP 2025, **grade A**): +55% fact recall, +40% response correctness, +27% reasoning accuracy — but it grounds in an ontology the domain *already has*, and is a **hypergraph over documents, not a triple store**. The circulating MuSiQue figure is **grade C**: n=100, LLM-as-judge, vendor-published, ~⅓ of questions unanswerable by the authors' own admission | [03](03-graph-and-memory-layers.md) |
| 4 | Competitor semantic surface | **We are behind on the cheap part** | Milvus, Weaviate, Elastic all ship field descriptions. Weaviate also has cross-references *and* shipped the first native vector-DB MCP server | [04](04-competitor-semantic-surface.md) |
| 5 | Veda's semantic surface | **Empty** | `FieldDefinition` carries `name`, type, `filterable`, `index_type`, `bm25`, tokenizer. `describe_collection` — the orientation tool for every agent and MCP client — returns `{name, type, filterable}` | [05](05-veda-current-semantic-surface.md) |
| 6 | Differentiation | **One real candidate** | Nobody measures whether their semantic layer helps. Veda alone has the machinery to | [06](06-differentiation-slate.md) |

---

## Relationship to the prior verdict: **upheld, and narrowed on one point**

`documentation/2026-08-17-multi-hop-retrieval-scope.md` rejected approach B (knowledge graph).
This cycle **upholds that rejection** and strengthens its main argument — the retrieval metric a
graph would serve is at 100%.

**One of its supporting arguments must be narrowed.** It claimed multimodal corpora "have no
good entity extractor for images/audio". That was asserted without a source and is now too
strong: M³KG-RAG (CVPR 2026) builds audio-visual multimodal KGs, TIVA-KG spans
text/image/video/audio, and the CVPR 2026 CASTLE work does hierarchical KG retrieval over video
(**grade B**). The correct argument is that multimodal KG construction is **research-grade and
extractor-bounded** — a second quality-critical model in the ingest path whose failures are
silent. That is still a strong reason not to build it. It is a cost argument, not an
impossibility argument, and it should be quoted that way from now on.

## A correction to this cycle's own planning note

The plan asserted that grepping `crates/` for ontology/RDF/graph terms returns **zero hits**.
Re-run, it returns **six files**. All are false positives — five are the *homonym* "semantic
layer" (the embedding-similarity stage of the guardrail check) and one is `rdf` matching inside
the identifier `OrdF32Entry`, 17 times. The claim to cite is "zero **true** hits". Details in
[`05`](05-veda-current-semantic-surface.md).

## A stale claim to retire elsewhere

`06-community-demand.md` records Veda's guardrails as "input-side and detect-only". That is no
longer true: `GuardrailCategory::IndirectInjection` exists, documented in-code against the OWASP
2026-07-12 vector, and `GuardrailMode::Block` exists. Anything repeating the old gap is stale.

---

## Recommendation order

**Do, in this order:**

1. **Field annotations on `FieldDefinition`** — description, semantic type, value domain,
   sensitivity — rendered through `ToolCatalog` so they reach the MCP server and the agent
   executor at once. **Frame it as closing a gap, never as a lead**; three competitors shipped it
   years ago. It is a prerequisite for 2 and 3.
2. **[D2] Field-declared sensitivity, enforced at retrieval.** Smallest real differentiator.
   Veda owns ingest, embedding and retrieval, so it can enforce where a bring-your-own-vector-DB
   stack cannot. Today PII is re-discovered probabilistically per record for something the
   operator knows once.
3. **[D1] Make the annotations a *tuned* artifact.** The one thing no competitor can copy:
   Milvus, Weaviate and Elastic let you write field descriptions and none tells you whether it
   helped. Veda's `tuning/` + `rag_eval.rs` loop can search annotations against an eval set
   generated from the user's own corpus and keep only what pays.

**Do not build:** a triple store, OWL 2 RL / SWRL inference, an ontology authoring canvas with
its review/quorum governance, `invoke_entity_action`-style governed writes, or a multimodal
knowledge graph.

**Defer:** cross-collection relations. A genuine gap against Weaviate, but it serves a workload
already measured at 100%.

**Demand-check before designing:** constraint validation (D3). Architecturally the one clean
idea inside ontobricks' reasoning layer, but data validation appears nowhere in the 14 ranked
community asks — schema *evolution* was #2, validation was absent.

---

## Binding conditions on whatever gets built

Both come from failures this project has already paid for:

- **Instrument the firing rate before measuring the win rate.** The stagnation detector was
  correct, tested, and fired in 1 run of 6 — it looked merely ineffective. Any annotation
  feature must first report how often the model actually reads the annotation.
- **Disable the mechanism and confirm the test fails.** Veda has repeatedly shipped tests that
  pass with the guarded mechanism deleted.

And one from this cycle: the agent loop is **nondeterministic** (`03`, `05`). Any accuracy claim
needs repeats and error bars, or it is indistinguishable from run-to-run variance.

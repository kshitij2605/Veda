# Ontology for Veda — decision

**Date:** 2026-08-20
**Status:** decision + one measurement gate. **No implementation is proposed here beyond the
first slice, and that slice is gated on a measurement that has not been run.**
**Source:** `research/2026-08-20-ontology-semantic-layer/`, prompted by
[databrickslabs/ontobricks](https://github.com/databrickslabs/ontobricks) v0.7.1.

---

## 1. The decision

**Veda does not need an ontology in the sense ontobricks means** — no triple store, no OWL 2 RL
or SWRL inference, no authoring canvas, no review/quorum governance, no multimodal knowledge
graph.

The reason is one measurement, not a view about ontologies. On a 2,012-document corpus built so
no top-k result set can hold both halves of an answer
(`documentation/2026-08-17-multi-hop-retrieval-scope.md`, 3 repeats per case), the shipped agent
loop scores **100% on single-hop, one-to-one and one-to-many entity joins**. That is the exact
workload every ontology/GraphRAG result targets. The measured deficit is elsewhere — negative
answers at 3/6, a failure to *conclude* that a fact is absent — and no graph addresses it.

The external evidence for ontology-grounding is real and does not change this. OG-RAG is
peer-reviewed (EMNLP 2025, arXiv:2412.15235) and reports +55% fact recall, +40% response
correctness and +27% reasoning accuracy — against a plain-RAG baseline, grounding in an ontology
the domain already owns, using a hypergraph over documents rather than a triple store. The circulating MuSiQue figure (95% vs 86% vs 79%) is
vendor-published, n=100, LLM-as-judge, with the authors disclosing that ~⅓ of questions are
unanswerable. Neither is evidence about a tuned hybrid retriever that is already saturated.

**What Veda does need** is the cheap semantic annotation the rest of the category shipped years
ago, and which Veda entirely lacks.

---

## 2. What Veda lacks, precisely

`FieldDefinition` (`crates/veda-core/src/schema.rs:10-27`) carries `name`, `field_type`,
`filterable`, `index_type`, `bm25`, `bm25_tokenizer`. Nothing else. Consequently
`describe_collection` (`crates/veda-gateway/src/agent/tools/builtins.rs:186-238`) — the
orientation tool for **every agent turn and every MCP client** — returns per field exactly:

```json
{ "name": "supplier_country", "type": "string", "filterable": true }
```

The model must infer from the name alone what the field means and whether `"DE"`, `"Germany"` or
`"DEU"` is stored.

**This is catch-up, and the research says so plainly.** Milvus has carried per-field
`description` since v2.2.x; Weaviate has collection and property descriptions, updatable on a
live collection; Elasticsearch has per-field `meta` with a standardised `unit` / `metric_type`
convention. It must be positioned as closing a gap, never as a lead.

---

## 3. The one first slice

**Field annotations on `FieldDefinition`, rendered through `ToolCatalog`.**

`ToolCatalog` (`crates/veda-gateway/src/agent/catalog.rs`) backs both the MCP server
(`ToolSpec::to_mcp_entry`) and the agent executor (`to_openai_tooldef`), so a single renderer
change reaches both surfaces. There is no second place to update — this is why the slice is this
one and not something adjacent.

Scope: additive, optional, `#[serde(default)]` so existing schemas load unchanged. Four
annotations, chosen because each has a named downstream consumer rather than because a taxonomy
suggested them:

| Annotation | Consumer | Precedent |
|---|---|---|
| `description` | `describe_collection` → agent + MCP | Milvus, Weaviate, Elastic |
| `semantic_type` (e.g. `iso3166_alpha2`, `email`, `currency_code`) | agent; later, filter validation | **None surveyed has this** |
| `value_domain` (enumeration or exemplar values) | agent — removes exploratory queries | **None surveyed has this** |
| `sensitivity` (`public` / `internal` / `restricted`, and a `pii` flag) | retrieval enforcement (§5) | **None surveyed has this** |

Nothing else. No relations, no constraints, no inference. Cross-collection relations are a
genuine gap against Weaviate and are **deferred**, because they serve the workload already
measured at 100%.

---

## 4. The measurement gate — run this before writing the slice

The slice must not be built on the assumption that annotations help. Two things get measured,
in this order, and the second is only meaningful if the first passes.

### Gate A — does the model read the annotation at all? (firing rate)

Reuse the harness pattern in `documentation/multihop-eval/multihop_v2.py`: a corpus large enough
that no single context window holds the answer, controls that isolate the mechanism, and repeats
because the loop is nondeterministic.

Build a corpus whose field names are **deliberately non-obvious** (`ctry_cd`, `amt_2`,
`ref_no`), pose questions answerable only if the model knows what the fields mean, and log
whether the annotation text appears in the model's reasoning or changes its tool arguments.

**Pass:** the annotation demonstrably changes tool arguments in a clear majority of runs.
**Fail:** it fires rarely — *stop and rethink the surface*, do not proceed to Gate B.

This gate exists because of a specific, recent failure: the stagnation detector was correct,
unit-tested and load-bearing, and fired in **1 run out of 6**. Measured only by win rate, it
looked ineffective rather than inert, and the wrong lesson would have been drawn. A change that
never triggers cannot be evaluated by whether it helps.

### Gate B — does it improve accuracy beyond variance?

Three arms on the same corpus, **≥5 repeats per case**, with error bars:

1. no annotations (baseline)
2. hand-written annotations
3. LLM-drafted annotations

**Pass:** arms 2 and 3 beat arm 1 by more than the run-to-run spread.
**Fail:** the difference is inside the spread. `documentation/2026-08-17-multi-hop-retrieval-scope.md`
already recorded a two-arm instruction test where a 13/16-vs-12/16 difference was correctly
called noise at n=4 per cell. Do not repeat that error in the favourable direction.

**Regression guard for both gates:** the one-to-one and one-to-many join cases must stay at
100%. The v2 negative-answer work showed that pressing the model harder on one axis degraded
another — arm B made `one2many-1` worse while not helping negatives.

---

## 5. What follows the slice, if the gates pass

In order, each its own decision:

1. **Enforce `sensitivity` at retrieval.** Fields marked `pii`/`restricted` are withheld or
   redacted for non-privileged callers, through the existing enforcement path
   (`agent/retrieval_guard.rs`, `agent/guardrails.rs`, `rls.rs`, `rbac.rs`). This is the smallest
   real differentiator in the slate: no surveyed product declares sensitivity on the field, and
   Veda owns ingest, embedding and retrieval, so it can enforce where a bring-your-own-vector-DB
   stack structurally cannot. Today PII is re-discovered probabilistically per record
   (`GuardrailCategory::Pii`) for something the operator knows once at schema-design time.
   - **Must be verified on every path** — `search`, `filter_query`, `get_record`, `aggregate`,
     `/converse` citations — and on `external` Parquet collections (`schema.rs:230`), where Veda
     does not control the write path and the declaration may be unenforceable. Check, don't assume.
   - **Delete the enforcement and confirm the test fails.** This project has repeatedly shipped
     tests that pass with the guarded mechanism removed.
2. **Make annotations a tuned artifact.** The only candidate in the slate that no competitor can
   copy. Milvus, Weaviate and Elastic all let you *write* field descriptions; not one reports
   whether writing them helped — they are documentation accepted on faith. Veda's tier-3 tuning
   campaign (`crates/veda-gateway/src/tuning/`) plus eval-set generation from the user's own
   corpus (`crates/veda-gateway/src/rest/rag_eval.rs::generate_eval_set_for_collection`) can
   search annotations and keep only what pays. Gate B's arm 3 is the pilot for this.

---

## 6. Explicitly not doing

- Triple store, OWL 2 RL, SWRL, SHACL inference — serve a saturated metric.
- Ontology authoring canvas and its DRAFT→IN-REVIEW→PUBLISHED / quorum governance — roughly a
  third of ontobricks exists so humans can negotiate an ontology. Veda's differentiator is
  deriving things from the corpus and measuring them; copying the authoring path takes the cost
  without the differentiator.
- `invoke_entity_action`-style governed writes on typed entities — a different product.
- Multimodal knowledge graph — viable in research (M³KG-RAG, CVPR 2026) but extractor-bounded,
  and it adds a second quality-critical model to the ingest path whose failures are silent.
- Becoming an enterprise semantic layer — Gartner's own 2026 framing is that the universal
  semantic layer failed and composition won. Veda is a component beneath that layer.

## 7. Open questions this decision does not settle

1. **Is there demand for constraint validation?** Architecturally it is the one clean idea in
   ontobricks' reasoning layer, needs no graph, and nobody has it — but data validation appears
   nowhere in the 14 ranked community asks in
   `research/2026-08-12-competitive-refresh/06-community-demand.md`. Schema *evolution* was #2;
   validation was absent. Needs a demand signal before a design.
2. **When do cross-collection relations become justified?** Only when a measured workload fails
   that the agent loop cannot serve. That workload has not been found.
3. **Does `semantic_type` want a closed vocabulary or free text?** A closed set is enforceable
   and machine-usable; free text is adoptable. Elasticsearch's `unit`/`metric_type` convention
   suggests the closed set is what makes the feature useful rather than decorative. Decide with
   Gate A's data, not before it.

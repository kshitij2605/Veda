# OntoBricks, layer by layer — and which layers have a referent in Veda

**Research date:** 2026-08-20.
**Source:** [databrickslabs/ontobricks](https://github.com/databrickslabs/ontobricks), repo tree
and `README.md` read at commit `7b27209` (accessed 2026-08-20). **Grade A** — primary repo, not
a marketing page.

---

## What it is

> "Knowledge Graph Builder for Databricks" — transforms Databricks tables into a materialized
> graph, with ontology design and reasoning exposed as tools via MCP.

Version 0.7.1. Python (FastAPI 0.109+, `pyproject.toml`, `uv.lock`), with a React frontend
("OntoViz" canvas, graph viewer, AI assistant). Shipped as a Databricks Asset Bundle
(`databricks.yml`, `app.yaml.template`).

**License and support posture, quoted:**

> "All projects in the /databrickslabs github account are provided for your exploration only,
> and are not formally supported by Databricks with Service Level Agreements (SLAs). They are
> provided AS-IS and we do not make any guarantees of any kind."

This matters for how much weight the design deserves: it is a Labs exploration with real
engineering behind it (821 commits, CI, mypy baseline, pre-commit, commitlint), not a supported
product whose shape has been validated by paying customers.

---

## The layers, and whether Veda has anything they could attach to

| # | OntoBricks layer | What it does | Referent in Veda? |
|---|---|---|---|
| 1 | **Ontology designer** (OntoViz canvas + AI assistant) | Visual authoring of classes, properties, relationships, *bridges* (cross-domain links), *class actions* (Unity Catalog function bindings), *cohorts* (rule-based entity groupings). Imports OWL, RDFS, FIBO, CDISC, IOF, HL7 FHIR R4/R4B/R5 | **No referent.** Veda has no object model above `FieldDefinition` to design against. Building the canvas first would be designing a schema for a schema that does not exist |
| 2 | **R2RML mapper** | Maps ontology entities/relations onto Unity Catalog tables at **column granularity**; auto-generates the SQL; materializes triples | **No referent, and this is the load-bearing mismatch.** R2RML's whole job is bridging *someone else's governed relational tables* to a semantic model. Veda's ordinary collections hold chunks and embeddings **Veda itself wrote** — there is no foreign relational schema to map. The single partial exception is `ExternalSource` (`veda-core/src/schema.rs:230`), where Veda reads Parquet it does not own; that is the only place a mapping layer would have anything to map |
| 3 | **Triple store** (Lakebase Postgres / Delta / Neo4j, pluggable) | Physical storage of materialized triples | **No referent.** Veda stores segments of vectors + stored fields. A triple store would be a new storage engine beside the existing one, with its own consistency, compaction and deletion story |
| 4 | **Reasoning engine** — OWL 2 RL inference, SWRL rules, SHACL constraint validation, a 19-point ontology pitfall detector | Derives implied facts; validates the ontology and the data against constraints | **Partial referent, and the most transferable idea in the repo.** Not the OWL inference — the **SHACL half**. Constraint validation over records ("`country` must be ISO-3166-alpha-2", "`amount` requires `currency`") maps directly onto data Veda already stores, and needs no graph. The OWL 2 RL / SWRL half needs a graph and has no referent |
| 5 | **Registry + governance** — per-domain versions, DRAFT ↔ IN-REVIEW → PUBLISHED, sign-off quorum, admin override, append-only audit in `domain_review_events`, single-editor lease with a ~10-min TTL | Makes the ontology a reviewed artifact rather than a config file | **No referent, and worth noticing anyway.** This is roughly a third of the product. It exists because an ontology is a *negotiated organisational asset*. Any Veda feature where a human authors semantics inherits this cost; a feature where semantics are **derived and measured** largely escapes it |
| 6 | **Query + MCP** — auto-generated GraphQL over published versions; MCP server exposing `get_entity_context` and `invoke_entity_action` | Lets agents read typed entity context and *invoke governed actions* on entities | **Strong referent.** Veda already has `ToolCatalog` serving both MCP and the agent executor (`veda-gateway/src/agent/catalog.rs`), and `describe_collection` is the direct analogue of `get_entity_context` — just far thinner. This is the layer where the two products actually meet |

---

## The three things worth taking, and the reason for each

**1. `get_entity_context` is the right shape and Veda's version is impoverished.** OntoBricks
hands an agent a *typed object with its relationships*; `describe_collection` hands it
`{name, type, filterable}`. The gap is not that Veda lacks a triple store — it is that Veda's
orientation tool carries no meaning at all. That is fixable without layers 1–3.

**2. SHACL-style constraint validation is separable from the graph.** SHACL is usually discussed
as an RDF technology, but a shape is just "field X must satisfy predicate P, and here is the
violation report". Veda has the data and a filter evaluator (`veda-query/src/filter/`); it lacks
only the declaration and the report.

**3. `invoke_entity_action` has no Veda analogue and should not acquire one.** OntoBricks binds
Unity Catalog functions to ontology classes so agents can *act* on business objects. Veda is a
retrieval substrate; adding governed write-actions on typed entities would be a different
product. Naming this explicitly so it is not smuggled in later.

## The three things not to take

- **The triple store.** A second storage engine, and Veda's own multi-hop measurement
  (`05-veda-current-semantic-surface.md`) shows retrieval at 100% on the joins a graph would
  serve.
- **OWL 2 RL / SWRL inference.** Requires the graph, and answers a question ("what else is
  implied?") that no measured Veda deficit asks.
- **The authoring canvas and its governance apparatus.** Two thirds of ontobricks' surface
  exists to let humans negotiate an ontology. Veda's differentiator — per `06` and the tuning
  campaign — is deriving things from the corpus and *measuring* them. Copying the human-authoring
  path adopts the cost without the differentiator.

## Note on the Databricks context, which does not transfer

OntoBricks is coherent because Unity Catalog already holds governed, typed, related tables with
lineage and permissions. The ontology is a *semantic re-description of an existing governed
model*. Veda has no such substrate: a collection is a flat bag of typed fields with one optional
RLS owner column. **OntoBricks' architecture is downstream of a governance layer Veda does not
have**, and any plan that copies its shape inherits an assumption Veda cannot satisfy.

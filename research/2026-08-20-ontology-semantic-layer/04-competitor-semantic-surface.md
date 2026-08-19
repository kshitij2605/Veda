# Competitor semantic surface: who already carries meaning in the schema

**Research date:** 2026-08-20. The question per product: **does the schema carry field-level
meaning beyond the physical type, is there any entity/relation concept, and does the agent/MCP
surface expose it?**

This file contains the most uncomfortable finding of the cycle and it is stated up front.

---

## The finding: Veda is behind on the cheap part, not ahead of it

The plan for this cycle listed "field descriptions surfaced through `describe_collection`" as
the *cheapest concrete win*, implicitly as an opportunity. It is not an opportunity. It is
**catch-up**: the three largest comparators have shipped per-field descriptive metadata for
years, and Elasticsearch has shipped standardised *units* on fields.

| Product | Field description? | Beyond description | Entity / relation concept | Exposed to agents |
|---|---|---|---|---|
| **Milvus** | **Yes** — `FieldSchema(description=...)` and `CollectionSchema(description=...)`, documented since **v2.2.x** ([API ref](https://milvus.io/api-reference/pymilvus/v2.5.x/MilvusClient/CollectionSchema/CollectionSchema.md), accessed 2026-08-20) | Struct fields with element-level search (2.6); nullable vectors | No | Via SDK/schema reads |
| **Weaviate** | **Yes** — collection `description` and `property_descriptions`, and they are **updatable** on a live collection ([docs](https://docs.weaviate.io/weaviate/manage-collections/collection-operations), accessed 2026-08-20) | Tokenization, module config, nested properties | **Yes — cross-references**, typed links between collections, filterable | **Yes — native MCP server in v1.37 (Apr 2026)**, the first in the category |
| **Elasticsearch** | **Yes** — per-field `meta`, plus a documented convention with standard entries: **`unit`** (`percent`, `byte`, time units) and **`metric_type`** (`gauge`/`counter`); updatable via mapping update; not supported on `object`/`nested` ([mapping-field-meta](https://www.elastic.co/docs/reference/elasticsearch/mapping-reference/mapping-field-meta), accessed 2026-08-20). Index-level `_meta` too | Runtime fields, ES\|QL | Parent/join field type | Via ES\|QL / integrations |
| **Qdrant** | **No** — payload is schemaless; indexes are declared per key, no descriptive metadata | Rich boolean/geo payload filtering; Relevance Feedback Query (Mar 2026) | No | MCP servers exist (community/official), schema-thin |
| **Pinecone** | **No** — metadata is schemaless key/value | Metadata filtering | No | MCP server exists |
| **LanceDB** | Arrow schema — **field metadata is expressible** because Arrow fields carry key/value metadata, but there is no product-level semantic concept above it | Lance format, versioning | No | — |
| **Neo4j** (not a vector DB, the ontology comparator) | Property graph; node labels and relationship types **are** the semantic model. Ontology via neosemantics/RDF | Full graph traversal + Cypher | **Yes, natively** | Graphiti builds on it |
| **Veda** | **No** — `FieldDefinition` carries `name`, `field_type`, `filterable`, `index_type`, `bm25`, `bm25_tokenizer` and nothing else (`veda-core/src/schema.rs:10-27`) | Auto-embed modality; RLS owner field | No | `describe_collection` returns `{name, type, filterable}` |

**Grades:** Milvus, Weaviate and Elasticsearch rows are **A** (vendor API reference read
directly). Qdrant/Pinecone/LanceDB absence-of-feature rows are **B** — absence is harder to
prove than presence, and they are inferred from the documented schema model rather than from a
statement that the feature does not exist. Weaviate's MCP-first claim is **B** (secondary
reporting, consistent with the 2026-08-12 refresh).

---

## Per-product verdict: does this erode a Veda claim, or open one?

- **Milvus — erodes.** Per-field descriptions since 2.2.x. Any Veda messaging implying it is
  ahead on schema expressiveness is wrong.
- **Weaviate — erodes twice, and is the closest comparator.** It has descriptions, it has
  **cross-references** (typed links between collections — a genuine relation concept that Veda's
  `FilterExpr` cannot express at all), and it shipped a **native MCP server first**. Weaviate is
  the product that most nearly already does "reading A of ontology", and it is the one to
  benchmark against rather than ontobricks.
- **Elasticsearch — erodes, in a specific and instructive way.** Its per-field `meta` with a
  *documented convention* for `unit` and `metric_type` is exactly the "semantic annotation"
  design, already standardised and already shipping. It is also a reminder that the useful
  version of this feature is a **convention plus consumers**, not a free-text blob.
- **Qdrant / Pinecone — neutral.** Schemaless by design; they made the opposite bet.
- **LanceDB — neutral, with a note.** Arrow field metadata means the *mechanism* exists free;
  nobody has built a product concept on it. That is a real signal about demand.
- **Neo4j — not a competitor, a substrate.** It is what the agent-memory graph layer in `03`
  builds on. Competing with it is a category error.

---

## What no one in this table has

Verified as absent across every product surveyed:

1. **A semantic type distinct from the physical type.** Nobody records that a `string` field is
   an ISO-3166 country code, an email address, or a supplier identifier.
2. **A sensitivity / PII classification declared on the field.** Veda classifies PII per *record*
   at ingest via a detector (`GuardrailCategory::Pii`), which is probabilistic re-discovery of
   something an operator already knows. Nobody declares it on the schema and enforces from there.
3. **Value domains / enumerations** exposed to an agent, so a model knows `"DE"` not `"Germany"`
   is stored without issuing exploratory queries.
4. **Any measurement that the semantics helped.** This is the largest gap in the whole survey.
   Milvus, Weaviate and Elastic all let you *write* field descriptions. Not one of them tells
   you whether writing them changed retrieval quality, agent accuracy, or anything else. They
   are documentation, accepted on faith.

Point 4 is where Veda's existing machinery is genuinely unusual: `tuning/` + `rag_eval.rs`
already generate an eval set from the user's own corpus and search a configuration space against
it (`05`). Veda is the only product surveyed that could **treat schema semantics as a tunable
and report whether they paid**. Whether that is worth building is `06`'s question — but it is
the only candidate in this cycle that is not catch-up.

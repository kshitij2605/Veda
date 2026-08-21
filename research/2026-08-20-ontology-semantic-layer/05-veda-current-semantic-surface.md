# What Veda knows about meaning today

**Research date:** 2026-08-20. Read against `docs/ontology-semantic-layer-research` at `9b007ea`.
Every row cites a file path. Nothing here is quoted from a prior research document.

---

## The grep, and a correction

The planning note for this cycle asserted that
`grep -ril "ontology|knowledge.graph|rdf|sparql|owl2|taxonomy|semantic.layer"` over `crates/`
returns **zero hits**. Re-run, it returns **six files**. Every one is a false positive, and the
two reasons are worth recording because both would mislead a future search:

| File | What matched | Why it is not a hit |
|---|---|---|
| `crates/veda-gateway/src/agent/guardrails.rs` (13 lines) | `semantic layer` | A **homonym**. Here "semantic layer" is the embedding-similarity stage of the guardrail check — "a curated, in-code seed list of bad patterns" scored by cosine distance (`guardrails.rs:12`). Nothing to do with a data semantic layer. |
| `crates/veda-gateway/src/rest/collections.rs:1816` | `semantic layer` | Same homonym, in a test comment. |
| `crates/veda-gateway/tests/retrieval_guard_*.rs` (3 files) | `semantic layer` | Same homonym. |
| `crates/veda-storage/src/growing_segment.rs` | `rdf` × 17 | A **substring artifact**: all 17 are `rdF` inside the identifier `OrdF32Entry`. Confirmed with `grep -oi rdf ... \| sort \| uniq -c` → `17 rdF`. |

Over `documentation/` and `research/`, the same terms hit four files, all of which *discuss*
knowledge graphs as a candidate rather than describing anything built:
`2026-08-17-multi-hop-retrieval-scope.md`, `2026-08-18-website-brief-multihop.md`,
`research/2026-08-12-competitive-refresh/06-community-demand.md`, and
`research/vector-database-comparison.md`.

**Corrected claim, which is the one to cite:** Veda contains no ontology, RDF, triple, graph or
taxonomy concept in any crate. The correct statement is "zero true hits", not "zero hits", and
anyone re-running the grep should expect the six above.

---

## What meaning Veda actually holds

| Meaning held | Where it lives | What consumes it |
|---|---|---|
| Field **name** (an opaque string) | `FieldDefinition.name`, `veda-core/src/schema.rs:11` | Everything. It is the only carrier of human intent in the schema. |
| Physical **type** | `FieldDefinition.field_type` → `FieldType`, `veda-core/src/types.rs:23` | Validation, index selection, filter type-checking |
| **Filterable** flag | `FieldDefinition.filterable`, `schema.rs:13` | Scalar index build; `filter_query` |
| **Index type** | `FieldDefinition.index_type`, `schema.rs:14` | Index construction |
| **BM25** on/off + tokenizer name | `schema.rs:20,26` | Full-text index; `cjk_bigram` default |
| Embedding **model binding** | `FieldType::Vector{model_id}`, `types.rs:32` | Rejects embed-insert with a mismatched model |
| MRL truncation lengths | `FieldType::Vector{mrl_dims}`, `types.rs:39` | `truncate_to` at search |
| Auto-embed **source field + modality** | `AutoEmbedConfig`, `types.rs:~90-110`; `AutoEmbedModality::{Text,Image,Video,Audio}` | Document pipeline routing — **the closest thing Veda has to a semantic annotation** |
| RLS **owner field** | `CollectionSchema.rls_owner_field`, `schema.rs:290`; consumed in `veda-gateway/src/rls.rs` | Injects `owner == user` into every non-Admin read |
| **External source** binding | `ExternalSource{format,uri,id_field,vector_field,metric}`, `schema.rs:230-251` | Read-in-place Parquet collections |
| Per-record **guardrail verdict** | `_guardrail_flagged` / `_guardrail_score` / `_guardrail_category`, `veda-core/src/record.rs:22-33` | Retrieval-time enforcement |
| Per-record **time** | `_expires_at`, `_indexed_at`, `record.rs:15,40` | TTL; temporal-anomaly checks |

**What is absent, and it is everything an ontology would add:** no field description, no semantic
or business type, no unit, no value domain or enumeration, no sensitivity/PII class **on the
schema**, no relation between fields, no relation between collections, no entity type, no
cardinality, no constraint beyond the physical type.

### The one place a vocabulary already exists

`GuardrailCategory` (`veda-gateway/src/agent/guardrails.rs:84`) is a real, small, closed
semantic vocabulary — `PromptInjection`, `Jailbreak`, `Unsafe`, `Pii`, `IndirectInjection` — and
`GuardrailMode` is `Off | Flag | Block`. Two consequences:

1. **PII is classified per *record*, at ingest, by a regex+embedding detector — never declared
   per *field*, where the operator already knows the answer.** A field named `ssn` is
   re-discovered probabilistically on every row instead of being declared once.
2. **The August refresh's "guardrails are input-side and detect-only" gap is closed.**
   `IndirectInjection` is documented in-code as "hostile instructions embedded in *retrieved*
   content rather than in the user's prompt — the OWASP 2026-07-12 indirect-injection vector",
   and `Block` mode exists. Any document repeating that gap is now stale.

---

## What the agent and MCP surface is told

`describe_collection` (`veda-gateway/src/agent/tools/builtins.rs:186-238`) is the orientation
tool for every agent turn and every MCP client. It returns, per field, exactly:

```json
{ "name": "...", "type": "string", "filterable": true }
```

plus `record_count` and `shard_count`. So a model working on a supplier corpus receives
`supplier_country: string, filterable: true` and must infer from the *name alone* what the field
means, what values it can take, whether `"DE"` or `"Germany"` or `"DEU"` is stored, and how it
relates to anything in another collection. The values are discoverable only by issuing queries.

This is the highest-leverage insertion point in the codebase, for a structural reason:
`ToolCatalog` (`veda-gateway/src/agent/catalog.rs`) backs **both** the MCP server
(`rest/mcp.rs`, via `ToolSpec::to_mcp_entry`) and the agent executor (via
`to_openai_tooldef`). One renderer change reaches both surfaces; there is no second place to
update.

## What a filter can express

`FilterExpr` (`veda-query/src/filter/ast.rs:7`) covers `Eq`, `Ne`, `Gt`, `Gte`, `Lt`, `Lte`,
`Between`, `In`, `Contains`, `StartsWith`, and boolean combinators — all **within one
collection**. There is no join, no traversal, no reference type. A cross-collection question is
answerable today only by the agent issuing several `search` / `filter_query` calls and combining
the results in the model's head, which is precisely what
`documentation/2026-08-17-multi-hop-retrieval-scope.md` measured.

## The closed-loop machinery any proposal must reuse

Two subsystems already do the "derive something from the user's own corpus, then measure it"
motion. Anything this cycle recommends should extend them rather than start beside them.

| Subsystem | Where | What it already does |
|---|---|---|
| Eval-set generation | `veda-gateway/src/rest/rag_eval.rs` — `generate_eval_set_for_collection` (1348), `generate_cases` (816), `sample_diverse_chunks` (737), `document_groups` (634), `judge_case` (415), `run_eval` (1058) | Reads the user's own chunks, samples them for diversity, generates labeled cases with an LLM, judges them |
| Auto-tuning campaign | `veda-gateway/src/tuning/{config,search,campaign}.rs` | Searches a 3-tier space against that eval set: **tier 1** `ef_search`, `fusion_mode`, `dense_weight`, `rerank`, `top_k`, `truncate_to`; **tier 2** `index_type` (6 choices), `ef_construction`, `m`; **tier 3** `chunker_strategy` (7 choices), `chunk_size`, `overlap` |

**The tiering is the reusable idea.** Tiers are ordered by re-computation cost — query-time
only, reindex-without-re-embed, full re-chunk-and-re-embed. Any corpus-derived semantics would
sit at tier 3 or beyond, and the campaign engine already knows how to budget that.

## Measurements already on the record

From `documentation/2026-08-17-multi-hop-retrieval-scope.md`, on a 2,012-document corpus built
so no top-k result set can hold both halves of an answer, 3 repeats per case:

| case type | correct | mean searches |
|---|---|---|
| single-hop control | 9/9 (100%) | 1.0 |
| one-to-one join | 12/12 (100%) | 1.9 |
| one-to-many join | 6/6 (100%) | 2.0 |
| negative (no answer exists) | **3/6 (50%)** | 4.8 |

Harness: `documentation/multihop-eval/multihop_v2.py`. This is the baseline any ontology
proposal has to beat, and the shape of it — retrieval saturated, *conclusion* failing — is what
makes most of the ontology literature aim at the wrong target for Veda. See `00-summary.md`.

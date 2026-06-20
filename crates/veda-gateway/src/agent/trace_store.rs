//! Searchable semantic LLM trace store + failure clustering (Phase 3:
//! LLM observability & optimization).
//!
//! **Opt-in** (privacy): an agent with `trace == true` has each completed
//! converse run persisted as an embedded record into a per-tenant
//! `_llm_traces` collection — `{ prompt, response, tool_ids, faithfulness,
//! cost_usd, cache_hit, guardrail_flagged, latency_ms, ts }` plus an
//! embedding of `prompt + "\n" + response`. The traces can then be
//! semantically searched (NL query → relevant past runs) and clustered into
//! failure groups (low-faithfulness runs collapsed by cosine similarity).
//! This generalizes the eval flywheel and is purely vector-native — an
//! inverted-index store can only keyword-search traces.
//!
//! The whole feature is **best-effort, off-lock, and fail-open**, mirroring
//! `agent/cache.rs` + `agent/memory.rs`:
//!   * embedding is done OFF any lock (clone the text `EmbeddingClient`
//!     config out from under the registry read-lock, drop the lock, then
//!     `embed_text(..).await` with no lock held);
//!   * a store failure is swallowed (logged at debug) and NEVER fails the
//!     turn;
//!   * `search_traces` / `cluster_traces` degrade to empty (never error) on
//!     a missing collection, absent embedder, or embed/search failure.
//!
//! Tenant/RLS-scoped: `tenant::scope` is applied so each tenant gets its
//! own isolated `_llm_traces` collection.

use serde::{Deserialize, Serialize};

use veda_core::embedding::Modality;
use veda_core::embedding_client::EmbeddingClient;
use veda_core::record::{EXPIRES_AT_FIELD, Record};
use veda_core::schema::{CollectionSchema, FieldDefinition};
use veda_core::types::{AutoEmbedConfig, AutoEmbedModality, DistanceMetric, FieldType};
use veda_query::search::VectorSearchRequest;

use super::model::Agent;
use crate::logs::categorize::cluster_embedded;
use crate::rest::state::AppState;
use crate::tenant::{self, TenantId};

/// User-facing name of the per-tenant LLM trace collection. Tenant scoping
/// (`tenant::scope`) is applied on top, so each tenant gets its own
/// isolated `_llm_traces` collection.
pub const TRACE_COLLECTION: &str = "_llm_traces";

/// The user prompt (input) for the run.
const PROMPT_FIELD: &str = "prompt";
/// The final assistant response.
const RESPONSE_FIELD: &str = "response";
/// Comma-joined tool ids invoked during the run (empty string when none).
const TOOL_IDS_FIELD: &str = "tool_ids";
/// Faithfulness/grounding score in `[0,1]`; `-1.0` when none was scored.
const FAITHFULNESS_FIELD: &str = "faithfulness";
/// Estimated cost of the run in USD (`0.0` when the model is unpriced).
const COST_USD_FIELD: &str = "cost_usd";
/// Whether this run was served from the semantic response cache.
const CACHE_HIT_FIELD: &str = "cache_hit";
/// Whether a safety guardrail flagged this run (input or output).
const GUARDRAIL_FLAGGED_FIELD: &str = "guardrail_flagged";
/// Wall-clock latency of the run in milliseconds.
const LATENCY_MS_FIELD: &str = "latency_ms";
/// Creation timestamp (epoch ms).
const TS_FIELD: &str = "ts";
/// Hidden auto-embed source: `prompt + "\n" + response`. Not returned in
/// hits; it exists only so the schema is self-describing for other tooling
/// (this path supplies the vector explicitly on write).
const EMBED_SRC_FIELD: &str = "embed_src";
/// The vector field auto-embedded over [`EMBED_SRC_FIELD`].
const EMBEDDING_FIELD: &str = "embedding";

/// Sentinel faithfulness value meaning "not scored".
pub const FAITHFULNESS_NONE: f64 = -1.0;

/// Default number of traces to pull / return.
pub const DEFAULT_LIMIT: usize = 20;
/// Maximum number of traces considered for a search / cluster call.
pub const MAX_LIMIT: usize = 500;
/// Default cosine threshold for failure clustering (mirrors the log
/// categorizer's embedding-path default).
pub const DEFAULT_CLUSTER_THRESHOLD: f32 = 0.82;

/// Default trace retention: 30 days, in SECONDS. Persisted prompts/responses
/// auto-expire after this so the trace store does not keep them indefinitely
/// (privacy). Mirrors the cache's TTL posture; the cache uses ms (24h), while
/// traces use the reserved `_expires_at` field which is epoch SECONDS.
pub const DEFAULT_TRACE_TTL_SECS: i64 = 30 * 24 * 60 * 60; // 2_592_000

/// Environment override for the trace retention TTL, in seconds.
const TRACE_TTL_ENV: &str = "VEDA_LLM_TRACE_TTL_SECS";

/// Resolve the configured trace TTL in seconds: `VEDA_LLM_TRACE_TTL_SECS`
/// when set to a parseable positive integer, else [`DEFAULT_TRACE_TTL_SECS`].
/// A value of `0` or an unparseable value falls back to the default (TTL is
/// always on — there is no "never expires" for stored prompts/responses).
fn trace_ttl_secs() -> i64 {
    std::env::var(TRACE_TTL_ENV)
        .ok()
        .and_then(|v| v.trim().parse::<i64>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(DEFAULT_TRACE_TTL_SECS)
}

/// One trace returned from a semantic search.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct TraceHit {
    pub prompt: String,
    pub response: String,
    pub tool_ids: String,
    pub faithfulness: f64,
    pub cost_usd: f64,
    pub cache_hit: bool,
    pub guardrail_flagged: bool,
    pub latency_ms: f64,
    pub ts: i64,
    /// Cosine similarity of the stored trace to the query, in `[-1, 1]`.
    pub score: f32,
}

/// One failure cluster: a group of semantically similar trace prompts.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct TraceCluster {
    /// Representative (normalized) prompt for the cluster.
    pub representative: String,
    /// Number of member traces.
    pub size: u64,
    /// Example member prompts (up to a small cap).
    pub member_ids: Vec<String>,
    /// Mean faithfulness across the cluster's scored members; `-1.0` when
    /// no member carried a score.
    pub avg_faithfulness: f64,
}

/// Whether the given agent has opted into the LLM trace store. Kept as a
/// tiny named guard so the wiring in `run_agent_turn` is unit-testable
/// without standing up an LLM.
#[inline]
pub fn should_store_trace(agent: &Agent) -> bool {
    agent.trace
}

/// Resolve a text embedder by cloning its config out from under the
/// registry read-lock (so the lock is never held across the embed await).
/// `None` when the deployment cannot embed text.
async fn text_embedder(state: &AppState) -> Option<EmbeddingClient> {
    let registry = state.inner.embedding_registry.read().await;
    registry
        .find_by_modality(Modality::Text)
        .first()
        .map(|c| EmbeddingClient::new((*c).clone()))
}

/// The text embedded for one trace: the prompt and response joined by a
/// newline (so a search query matches against both sides of the turn).
fn embed_source(prompt: &str, response: &str) -> String {
    format!("{prompt}\n{response}")
}

/// Build the tenant-scoped trace collection schema for a given embedding
/// dimension. Pure — no I/O, so it is directly unit-testable.
pub fn trace_schema(scoped_name: &str, dimension: u32) -> Result<CollectionSchema, String> {
    CollectionSchema::builder(scoped_name)
        .add_field(FieldDefinition::new(PROMPT_FIELD, FieldType::String))
        .add_field(FieldDefinition::new(RESPONSE_FIELD, FieldType::String))
        .add_field(FieldDefinition::new(TOOL_IDS_FIELD, FieldType::String).with_filterable(true))
        .add_field(
            FieldDefinition::new(FAITHFULNESS_FIELD, FieldType::Float64).with_filterable(true),
        )
        .add_field(FieldDefinition::new(COST_USD_FIELD, FieldType::Float64))
        .add_field(FieldDefinition::new(CACHE_HIT_FIELD, FieldType::Bool).with_filterable(true))
        .add_field(
            FieldDefinition::new(GUARDRAIL_FLAGGED_FIELD, FieldType::Bool).with_filterable(true),
        )
        .add_field(FieldDefinition::new(LATENCY_MS_FIELD, FieldType::Float64))
        .add_field(FieldDefinition::new(TS_FIELD, FieldType::Int64).with_filterable(true))
        // Reserved per-record TTL (epoch SECONDS). Declared so the raw-map
        // value survives `parse_record_public` (which only keeps schema
        // fields); honored by `lookup_record_at` (search) + `is_expired_at`
        // (cluster) + the TTL sweeper, so expired traces auto-purge.
        .add_field(FieldDefinition::new(EXPIRES_AT_FIELD, FieldType::Int64).with_filterable(true))
        .add_field(FieldDefinition::new(EMBED_SRC_FIELD, FieldType::String))
        .add_field(
            FieldDefinition::vector(EMBEDDING_FIELD, dimension, DistanceMetric::Cosine)
                .with_auto_embed(AutoEmbedConfig {
                    source: EMBED_SRC_FIELD.to_owned(),
                    modality: AutoEmbedModality::Text,
                }),
        )
        .build()
        .map_err(|e| e.to_string())
}

/// All the per-run fields persisted for one trace.
#[derive(Clone, Debug)]
pub struct TraceFields {
    pub prompt: String,
    pub response: String,
    pub tool_ids: String,
    pub faithfulness: f64,
    pub cost_usd: f64,
    pub cache_hit: bool,
    pub guardrail_flagged: bool,
    pub latency_ms: f64,
    pub ts: i64,
}

/// Build the JSON record map for one trace (the shape passed to
/// `parse_record_public`). Pure — the `embedding` is supplied explicitly.
/// `ttl_secs` stamps the reserved `_expires_at` (epoch SECONDS) at
/// `ts/1000 + ttl_secs` so the trace auto-expires (privacy retention). `ts`
/// is epoch MILLISECONDS, while `_expires_at` is epoch SECONDS.
fn trace_record_map(
    f: &TraceFields,
    ttl_secs: i64,
    embedding: Vec<f32>,
) -> serde_json::Map<String, serde_json::Value> {
    let num = |x: f64| {
        serde_json::Number::from_f64(x)
            .map(serde_json::Value::Number)
            .unwrap_or(serde_json::Value::Null)
    };
    let mut m = serde_json::Map::new();
    m.insert(
        PROMPT_FIELD.to_owned(),
        serde_json::Value::String(f.prompt.clone()),
    );
    m.insert(
        RESPONSE_FIELD.to_owned(),
        serde_json::Value::String(f.response.clone()),
    );
    m.insert(
        TOOL_IDS_FIELD.to_owned(),
        serde_json::Value::String(f.tool_ids.clone()),
    );
    m.insert(FAITHFULNESS_FIELD.to_owned(), num(f.faithfulness));
    m.insert(COST_USD_FIELD.to_owned(), num(f.cost_usd));
    m.insert(
        CACHE_HIT_FIELD.to_owned(),
        serde_json::Value::Bool(f.cache_hit),
    );
    m.insert(
        GUARDRAIL_FLAGGED_FIELD.to_owned(),
        serde_json::Value::Bool(f.guardrail_flagged),
    );
    m.insert(LATENCY_MS_FIELD.to_owned(), num(f.latency_ms));
    m.insert(TS_FIELD.to_owned(), serde_json::Value::Number(f.ts.into()));
    // Reserved TTL field is epoch SECONDS; trace `ts` is epoch MILLISECONDS.
    let expires_at_secs = (f.ts / 1000).saturating_add(ttl_secs);
    m.insert(
        EXPIRES_AT_FIELD.to_owned(),
        serde_json::Value::Number(expires_at_secs.into()),
    );
    m.insert(
        EMBED_SRC_FIELD.to_owned(),
        serde_json::Value::String(embed_source(&f.prompt, &f.response)),
    );
    m.insert(
        EMBEDDING_FIELD.to_owned(),
        serde_json::Value::Array(
            embedding
                .into_iter()
                .map(|x| {
                    serde_json::Number::from_f64(x as f64)
                        .map(serde_json::Value::Number)
                        .unwrap_or(serde_json::Value::Null)
                })
                .collect(),
        ),
    );
    m
}

/// Ensure the tenant's trace collection exists, creating it (schema
/// persisted to disk + materialized into in-process state) sized for the
/// given embedding `dimension` if absent. Best-effort: returns the scoped
/// collection name on success, `None` on any failure. Idempotent. Mirrors
/// `cache::ensure_cache_collection` / `memory::ensure_memory_collection`.
async fn ensure_trace_collection(
    state: &AppState,
    tenant: &TenantId,
    dimension: u32,
) -> Option<String> {
    let scoped = tenant::scope(tenant, TRACE_COLLECTION);
    {
        let schemas = state.inner.schemas.read().await;
        if schemas.contains_key(&scoped) {
            return Some(scoped);
        }
    }

    let schema = trace_schema(&scoped, dimension).ok()?;

    let collection_dir = state.inner.data_dir.join(&scoped);
    std::fs::create_dir_all(&collection_dir).ok()?;
    let schema_json = serde_json::to_vec_pretty(&schema).ok()?;
    std::fs::write(collection_dir.join("collection.json"), &schema_json).ok()?;

    crate::rest::collections::materialize_collection_locally(state, &schema, &collection_dir)
        .await
        .ok()?;
    Some(scoped)
}

/// Persist one completed run as an embedded trace, auto-creating the
/// collection on first write. **Best-effort and non-fatal**: embeds
/// `prompt + "\n" + response` off-lock, then inserts the record (with the
/// explicitly-computed vector) directly into the segment manager. Any
/// failure is logged at debug and swallowed — the turn is never failed.
pub async fn store_trace(state: &AppState, tenant: &TenantId, fields: TraceFields) {
    if fields.prompt.trim().is_empty() && fields.response.trim().is_empty() {
        return;
    }

    let Some(client) = text_embedder(state).await else {
        return;
    };
    let src = embed_source(&fields.prompt, &fields.response);
    let embedding = match client.embed_text(&src).await {
        Ok(v) => v,
        Err(e) => {
            tracing::debug!(error = %e, "llm trace: embed failed; skipping store");
            return;
        }
    };

    store_trace_embedded(state, tenant, fields, embedding).await;
}

/// Persist one already-embedded trace and make it **immediately
/// searchable**. Ensures the collection, parses the record, then inserts it
/// via the shared seal + executor-rebuild path so a subsequent semantic
/// trace search finds it (a bare `SegmentManager::insert` would leave it
/// only in the growing segment the executor never sees). Best-effort. The
/// vector is supplied explicitly so this is unit-testable without a live
/// embedder.
async fn store_trace_embedded(
    state: &AppState,
    tenant: &TenantId,
    fields: TraceFields,
    embedding: Vec<f32>,
) {
    let dimension = embedding.len() as u32;

    let Some(scoped) = ensure_trace_collection(state, tenant, dimension).await else {
        tracing::debug!("llm trace: could not ensure trace collection; skipping store");
        return;
    };

    let schema = {
        let schemas = state.inner.schemas.read().await;
        match schemas.get(&scoped).cloned() {
            Some(s) => s,
            None => return,
        }
    };
    let raw = trace_record_map(&fields, trace_ttl_secs(), embedding);
    let record = match crate::rest::records::parse_record_public(&raw, &schema) {
        Ok(r) => r,
        Err(e) => {
            tracing::debug!(error = %e, "llm trace: record parse failed; skipping store");
            return;
        }
    };

    crate::agent::index::insert_only(state, &scoped, &schema, record, "llm trace").await;
}

/// Semantic search over the tenant's traces. Embeds `query` off-lock,
/// vector-searches `_llm_traces`, and returns up to `limit` hits with the
/// stored fields + cosine score. Returns empty (never errors) when the
/// collection is absent, no embedder is configured, or the search fails.
pub async fn search_traces(
    state: &AppState,
    tenant: &TenantId,
    query: &str,
    limit: usize,
) -> Vec<TraceHit> {
    if query.trim().is_empty() || limit == 0 {
        return Vec::new();
    }
    let limit = limit.min(MAX_LIMIT);
    let scoped = tenant::scope(tenant, TRACE_COLLECTION);

    {
        let schemas = state.inner.schemas.read().await;
        if !schemas.contains_key(&scoped) {
            return Vec::new();
        }
    }

    let Some(client) = text_embedder(state).await else {
        return Vec::new();
    };
    let vector = match client.embed_text(query).await {
        Ok(v) => v,
        Err(e) => {
            tracing::debug!(error = %e, "llm trace: query embed failed; empty search");
            return Vec::new();
        }
    };

    let request = VectorSearchRequest {
        field: EMBEDDING_FIELD.to_owned(),
        query_vector: vector,
        k: limit,
        metric: DistanceMetric::Cosine,
        filter: None,
        filter_expr: None,
        truncate_to: None,
        read_ts: None,
    };

    let executors = state.inner.query_executors.read().await;
    let Some(executor) = executors.get(&scoped) else {
        return Vec::new();
    };
    let response = match executor.search(&request) {
        Ok((response, _)) => response,
        Err(e) => {
            tracing::debug!(error = %e, "llm trace: search failed; empty search");
            return Vec::new();
        }
    };
    drop(executors);

    // The executor only sees sealed segments. Merge in the live growing
    // segment (where a just-stored trace lives until it seals) so a fresh
    // trace is searchable immediately, with no per-write seal/rebuild.
    let growing = super::index::search_live_growing(state, &scoped, &request).await;
    let merged = super::index::merge_top_k(response.results, growing, request.k);
    let response = veda_query::search::SearchResponse {
        results: merged,
        total_candidates: 0,
    };

    let now_secs = now_ms() / 1000;
    let managers = state.inner.segment_managers.read().await;
    let shard_map = managers.get(&scoped);

    let mut out = Vec::new();
    for r in response.results {
        let Some(shard_map) = shard_map else { break };
        let Some(rec) = shard_map
            .values()
            .find_map(|sm| sm.lookup_record_at(r.id, now_secs))
        else {
            continue;
        };
        out.push(TraceHit {
            prompt: string_field(&rec, PROMPT_FIELD),
            response: string_field(&rec, RESPONSE_FIELD),
            tool_ids: string_field(&rec, TOOL_IDS_FIELD),
            faithfulness: float_field(&rec, FAITHFULNESS_FIELD),
            cost_usd: float_field(&rec, COST_USD_FIELD),
            cache_hit: bool_field(&rec, CACHE_HIT_FIELD),
            guardrail_flagged: bool_field(&rec, GUARDRAIL_FLAGGED_FIELD),
            latency_ms: float_field(&rec, LATENCY_MS_FIELD),
            ts: int_field(&rec, TS_FIELD),
            score: 1.0 - r.distance,
        });
    }
    out
}

/// One enumerated trace record's clustering inputs.
struct RawTrace {
    prompt: String,
    faithfulness: f64,
}

/// Enumerate up to `limit` recent (non-deleted, non-expired) trace records
/// from the collection's segment managers, mirroring `list_records`. Brief
/// read-lock; no await held across it. Returns the prompt/response/
/// faithfulness for each.
async fn recent_traces(state: &AppState, scoped: &str, limit: usize) -> Vec<RawTrace> {
    let now_secs = now_ms() / 1000;
    let managers = state.inner.segment_managers.read().await;
    let Some(shard_map) = managers.get(scoped) else {
        return Vec::new();
    };
    let mut out: Vec<RawTrace> = Vec::new();
    for sm in shard_map.values() {
        let gs = sm.growing_segment();
        for doc_id in 0..gs.count() {
            if gs.is_deleted(doc_id) {
                continue;
            }
            if let Some(rec) = gs.get_record(doc_id) {
                if rec.is_expired_at(now_secs) {
                    continue;
                }
                out.push(RawTrace {
                    prompt: string_field(rec, PROMPT_FIELD),
                    faithfulness: float_field(rec, FAITHFULNESS_FIELD),
                });
            }
        }
        for handle in sm.sealed_segments() {
            for doc_id in 0..handle.metadata.row_count as u32 {
                if handle.reader.delete_bitmap().is_deleted(doc_id) {
                    continue;
                }
                if let Ok(rec) = handle.reader.get_record(doc_id) {
                    if rec.is_expired_at(now_secs) {
                        continue;
                    }
                    out.push(RawTrace {
                        prompt: string_field(&rec, PROMPT_FIELD),
                        faithfulness: float_field(&rec, FAITHFULNESS_FIELD),
                    });
                }
            }
        }
    }
    // Keep the most recent `limit` (records accumulate roughly in insert
    // order; take the tail).
    if out.len() > limit {
        out.drain(0..out.len() - limit);
    }
    out
}

/// Cluster recent traces into failure groups. Pulls up to `limit` recent
/// traces (optionally only those with `faithfulness < faithfulness_max`,
/// i.e. "failures"), re-embeds each prompt off-lock, and clusters by cosine
/// similarity (reusing the log categorizer's `cluster_embedded`). Returns
/// each cluster's representative prompt, size, member examples, and mean
/// faithfulness. Empty (never errors) when the collection is absent, no
/// embedder is configured, or embedding fails.
pub async fn cluster_traces(
    state: &AppState,
    tenant: &TenantId,
    limit: usize,
    faithfulness_max: Option<f64>,
) -> Vec<TraceCluster> {
    if limit == 0 {
        return Vec::new();
    }
    let limit = limit.min(MAX_LIMIT);
    let scoped = tenant::scope(tenant, TRACE_COLLECTION);

    {
        let schemas = state.inner.schemas.read().await;
        if !schemas.contains_key(&scoped) {
            return Vec::new();
        }
    }

    let traces = recent_traces(state, &scoped, limit).await;
    // Keep only "failures" when a faithfulness ceiling is requested. The
    // unscored sentinel (-1) is excluded from the failure view (it has no
    // measured faithfulness to call a failure).
    let traces: Vec<RawTrace> = traces
        .into_iter()
        .filter(|t| match faithfulness_max {
            Some(max) => t.faithfulness >= 0.0 && t.faithfulness < max,
            None => true,
        })
        .filter(|t| !t.prompt.trim().is_empty())
        .collect();
    if traces.is_empty() {
        return Vec::new();
    }

    let Some(client) = text_embedder(state).await else {
        return Vec::new();
    };

    // Embed each prompt off-lock; abort to empty on the first failure (the
    // stored embedding is over prompt+response, so we re-embed the prompt
    // for a prompt-similarity clustering view).
    let mut items: Vec<(String, Vec<f32>, i64, String)> = Vec::with_capacity(traces.len());
    for t in &traces {
        match client.embed_text(&t.prompt).await {
            Ok(emb) => items.push((
                t.prompt.clone(),
                emb,
                t.faithfulness.to_bits() as i64,
                "WARN".to_owned(),
            )),
            Err(e) => {
                tracing::debug!(error = %e, "llm trace: cluster embed failed; empty");
                return Vec::new();
            }
        }
    }

    // Reuse the greedy cosine clustering from the log categorizer. We smuggle
    // each trace's faithfulness through the `ts` slot (f64 bits) so we can
    // average it per cluster afterwards by matching member prompts.
    let cats = cluster_embedded(&items, DEFAULT_CLUSTER_THRESHOLD);

    cats.into_iter()
        .map(|c| {
            // Average faithfulness across this cluster's member prompts.
            let mut sum = 0.0;
            let mut n = 0u64;
            for ex in &c.examples {
                if let Some(t) = traces.iter().find(|t| &t.prompt == ex)
                    && t.faithfulness >= 0.0
                {
                    sum += t.faithfulness;
                    n += 1;
                }
            }
            let avg_faithfulness = if n > 0 {
                sum / n as f64
            } else {
                FAITHFULNESS_NONE
            };
            TraceCluster {
                representative: c.representative,
                size: c.count,
                member_ids: c.examples,
                avg_faithfulness,
            }
        })
        .collect()
}

/// Current wall-clock time in epoch milliseconds.
fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

fn string_field(rec: &Record, name: &str) -> String {
    match rec.get(name) {
        Some(veda_core::record::FieldValue::String(s)) => s.clone(),
        _ => String::new(),
    }
}

fn int_field(rec: &Record, name: &str) -> i64 {
    match rec.get(name) {
        Some(veda_core::record::FieldValue::Int64(v)) => *v,
        Some(veda_core::record::FieldValue::Int32(v)) => *v as i64,
        _ => 0,
    }
}

fn float_field(rec: &Record, name: &str) -> f64 {
    match rec.get(name) {
        Some(veda_core::record::FieldValue::Float64(v)) => *v,
        Some(veda_core::record::FieldValue::Float32(v)) => *v as f64,
        _ => 0.0,
    }
}

fn bool_field(rec: &Record, name: &str) -> bool {
    matches!(
        rec.get(name),
        Some(veda_core::record::FieldValue::Bool(true))
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fields() -> TraceFields {
        TraceFields {
            prompt: "what index does veda use?".to_owned(),
            response: "Veda uses HNSW.".to_owned(),
            tool_ids: "search,aggregate".to_owned(),
            faithfulness: 0.91,
            cost_usd: 0.0023,
            cache_hit: false,
            guardrail_flagged: false,
            latency_ms: 1234.5,
            ts: 1_700_000_000_000,
        }
    }

    #[test]
    fn should_store_trace_reflects_flag() {
        let mut a = crate::agent::model::default_agent();
        assert!(!should_store_trace(&a), "default agent has trace off");
        a.trace = true;
        assert!(should_store_trace(&a), "trace=true agent stores traces");
    }

    #[test]
    fn trace_schema_marks_filterables_and_vector_dim() {
        let schema = trace_schema("t1::_llm_traces", 8).expect("schema builds");
        assert!(schema.get_field(FAITHFULNESS_FIELD).unwrap().filterable());
        assert!(schema.get_field(CACHE_HIT_FIELD).unwrap().filterable());
        assert!(
            schema
                .get_field(GUARDRAIL_FLAGGED_FIELD)
                .unwrap()
                .filterable()
        );
        assert!(schema.get_field(TS_FIELD).unwrap().filterable());
        assert!(
            schema.get_field(EXPIRES_AT_FIELD).unwrap().filterable(),
            "_expires_at must be present + filterable for the TTL sweeper"
        );
        let vec_field = schema.get_field(EMBEDDING_FIELD).expect("embedding field");
        match vec_field.field_type() {
            FieldType::Vector {
                dimension,
                auto_embed,
                ..
            } => {
                assert_eq!(*dimension, 8);
                assert!(
                    auto_embed.is_some(),
                    "embedding field must carry auto_embed config"
                );
            }
            other => panic!("expected vector field type, got {other:?}"),
        }
    }

    #[test]
    fn trace_record_map_round_trips_through_schema() {
        let dim = 4u32;
        let schema = trace_schema("t1::_llm_traces", dim).expect("schema builds");
        let raw = trace_record_map(&fields(), DEFAULT_TRACE_TTL_SECS, vec![0.1, 0.2, 0.3, 0.4]);
        let rec = crate::rest::records::parse_record_public(&raw, &schema).expect("record parses");
        assert_eq!(
            string_field(&rec, PROMPT_FIELD),
            "what index does veda use?"
        );
        assert_eq!(string_field(&rec, RESPONSE_FIELD), "Veda uses HNSW.");
        assert_eq!(string_field(&rec, TOOL_IDS_FIELD), "search,aggregate");
        assert!((float_field(&rec, FAITHFULNESS_FIELD) - 0.91).abs() < 1e-9);
        assert!((float_field(&rec, COST_USD_FIELD) - 0.0023).abs() < 1e-9);
        assert!(!bool_field(&rec, CACHE_HIT_FIELD));
        assert!(!bool_field(&rec, GUARDRAIL_FLAGGED_FIELD));
        assert!((float_field(&rec, LATENCY_MS_FIELD) - 1234.5).abs() < 1e-6);
        assert_eq!(int_field(&rec, TS_FIELD), 1_700_000_000_000);
        // The reserved TTL field is stamped at ts(secs) + ttl, in SECONDS.
        assert_eq!(
            int_field(&rec, EXPIRES_AT_FIELD),
            1_700_000_000_000 / 1000 + DEFAULT_TRACE_TTL_SECS,
            "_expires_at must be ts(secs) + ttl_secs"
        );
        // The hidden embed source joins prompt + response.
        assert_eq!(
            string_field(&rec, EMBED_SRC_FIELD),
            "what index does veda use?\nVeda uses HNSW."
        );
        match rec.get(EMBEDDING_FIELD) {
            Some(veda_core::record::FieldValue::Vector(v)) => assert_eq!(v.dimension(), 4),
            other => panic!("expected vector field, got {other:?}"),
        }
    }

    #[test]
    fn embed_source_joins_prompt_and_response() {
        assert_eq!(embed_source("a", "b"), "a\nb");
    }

    // ── Best-effort / off-lock wiring against a bare AppState ────────────
    //
    // The test env has no live embedder, so the entry points must degrade
    // gracefully: `store_trace` is a no-op that does NOT create the
    // collection, and `search_traces` / `cluster_traces` return empty. The
    // request SHAPE is exercised by the pure tests above; the live
    // embed/search path is validated against a deployment.

    #[tokio::test]
    async fn store_is_noop_without_embedder() {
        let tmp = tempfile::tempdir().unwrap();
        let state = AppState::new(tmp.path());
        let tenant = TenantId::default_tenant();
        store_trace(&state, &tenant, fields()).await;
        let scoped = tenant::scope(&tenant, TRACE_COLLECTION);
        assert!(
            !state.inner.schemas.read().await.contains_key(&scoped),
            "store_trace must not create the collection when it cannot embed",
        );
    }

    #[tokio::test]
    async fn search_returns_empty_without_embedder_or_collection() {
        let tmp = tempfile::tempdir().unwrap();
        let state = AppState::new(tmp.path());
        let tenant = TenantId::default_tenant();
        let hits = search_traces(&state, &tenant, "anything", DEFAULT_LIMIT).await;
        assert!(
            hits.is_empty(),
            "no embedder/collection → empty, never an error"
        );
    }

    #[tokio::test]
    async fn cluster_returns_empty_without_collection() {
        let tmp = tempfile::tempdir().unwrap();
        let state = AppState::new(tmp.path());
        let tenant = TenantId::default_tenant();
        let clusters = cluster_traces(&state, &tenant, DEFAULT_LIMIT, None).await;
        assert!(clusters.is_empty(), "no collection → empty, never an error");
        let clusters = cluster_traces(&state, &tenant, DEFAULT_LIMIT, Some(0.5)).await;
        assert!(clusters.is_empty());
    }

    // Searchability regression: a stored trace must be returned by a
    // semantic search for its own embedding immediately after store.
    // Supplies the vector via the `store_trace_embedded` seam (no live
    // embedder). Before the seal + executor-rebuild fix this returned 0.
    #[tokio::test]
    async fn stored_trace_is_immediately_searchable() {
        let tmp = tempfile::tempdir().unwrap();
        let state = AppState::new(tmp.path());
        let tenant = TenantId::default_tenant();
        let embedding = vec![0.0f32, 0.0, 1.0, 0.0];

        store_trace_embedded(&state, &tenant, fields(), embedding.clone()).await;

        let scoped = tenant::scope(&tenant, TRACE_COLLECTION);
        assert!(state.inner.schemas.read().await.contains_key(&scoped));

        let request = VectorSearchRequest {
            field: EMBEDDING_FIELD.to_owned(),
            query_vector: embedding,
            k: DEFAULT_LIMIT,
            metric: DistanceMetric::Cosine,
            filter: None,
            filter_expr: None,
            truncate_to: None,
            read_ts: None,
        };
        let sealed = {
            let executors = state.inner.query_executors.read().await;
            let executor = executors.get(&scoped).expect("executor exists");
            executor.search(&request).expect("search runs").0.results
        };
        let growing = super::super::index::search_live_growing(&state, &scoped, &request).await;
        let results = super::super::index::merge_top_k(sealed, growing, request.k);
        assert_eq!(
            results.len(),
            1,
            "the stored trace must be returned by a semantic search for its own embedding"
        );
    }

    // Bounded-cost regression: storing N distinct traces must NOT seal a
    // segment per write (no O(n)-per-write rebuild). All traces land in the
    // live growing segment; nothing seals until the size threshold.
    #[tokio::test]
    async fn storing_many_traces_does_not_explode_sealed_segments() {
        let tmp = tempfile::tempdir().unwrap();
        let state = AppState::new(tmp.path());
        let tenant = TenantId::default_tenant();

        const N: usize = 30;
        for i in 0..N {
            let mut embedding = vec![0.0f32; 8];
            embedding[i % 8] = 1.0 + i as f32;
            let mut f = fields();
            f.prompt = format!("prompt {i}");
            f.response = format!("response {i}");
            f.ts = 1_700_000_000_000 + i as i64;
            store_trace_embedded(&state, &tenant, f, embedding).await;
        }

        let scoped = tenant::scope(&tenant, TRACE_COLLECTION);
        let managers = state.inner.segment_managers.read().await;
        let shard_map = managers.get(&scoped).expect("trace collection exists");
        let sealed_total: usize = shard_map
            .values()
            .map(|sm| sm.sealed_segments().len())
            .sum();
        let growing_total: u32 = shard_map
            .values()
            .map(|sm| sm.growing_segment().count())
            .sum();
        assert_eq!(
            sealed_total, 0,
            "storing {N} traces must not seal per write (got {sealed_total})"
        );
        assert_eq!(
            growing_total as usize, N,
            "all {N} traces must live in the growing segment"
        );
    }

    // ── Retention TTL (privacy) ─────────────────────────────────────────

    // The env override is honored when set to a positive integer; an unset /
    // zero / unparseable value falls back to the 30-day default. Serialized
    // because it mutates a process-global env var.
    #[test]
    fn trace_ttl_secs_reads_env_with_default_fallback() {
        // Default when unset.
        unsafe { std::env::remove_var(TRACE_TTL_ENV) };
        assert_eq!(trace_ttl_secs(), DEFAULT_TRACE_TTL_SECS);
        assert_eq!(DEFAULT_TRACE_TTL_SECS, 2_592_000, "default is 30 days");

        // Positive override is honored.
        unsafe { std::env::set_var(TRACE_TTL_ENV, "3600") };
        assert_eq!(trace_ttl_secs(), 3600);

        // Zero / unparseable / negative fall back to the default (TTL is
        // always on for stored prompts/responses).
        unsafe { std::env::set_var(TRACE_TTL_ENV, "0") };
        assert_eq!(trace_ttl_secs(), DEFAULT_TRACE_TTL_SECS);
        unsafe { std::env::set_var(TRACE_TTL_ENV, "nonsense") };
        assert_eq!(trace_ttl_secs(), DEFAULT_TRACE_TTL_SECS);

        unsafe { std::env::remove_var(TRACE_TTL_ENV) };
    }

    // An expired trace (its `_expires_at` already in the past) is excluded
    // from BOTH read paths — the `search_traces` hydration (`lookup_record_at`)
    // and the `cluster_traces` enumeration (`recent_traces` / `is_expired_at`)
    // — while a fresh trace (long TTL) is included. The vectors are supplied
    // via the `store_trace_embedded` seam, with the `_expires_at` stamped by a
    // negative effective TTL so it lands in the past without a live embedder.
    #[tokio::test]
    async fn expired_trace_excluded_from_search_and_cluster() {
        let tmp = tempfile::tempdir().unwrap();
        let state = AppState::new(tmp.path());
        let tenant = TenantId::default_tenant();

        let now_secs = now_ms() / 1000;

        // Store an EXPIRED trace directly: build the raw map with an
        // `_expires_at` already in the past, then insert via the same parse +
        // insert_only path `store_trace_embedded` uses.
        let scoped = tenant::scope(&tenant, TRACE_COLLECTION);
        let mut expired = fields();
        expired.prompt = "expired prompt".to_owned();
        expired.response = "expired response".to_owned();
        // ts far in the past so ts/1000 + ttl < now even with a large ttl.
        expired.ts = 1_000; // ~1970, 1 second past epoch (ms)

        let fresh_embedding = vec![0.0f32, 0.0, 1.0, 0.0];
        // Seed the collection + a FRESH trace via the normal seam (its
        // _expires_at = ts/1000 + 30d, well in the future).
        let mut fresh = fields();
        fresh.prompt = "fresh prompt".to_owned();
        fresh.response = "fresh response".to_owned();
        fresh.ts = now_secs.saturating_mul(1000); // ~now (ms)
        store_trace_embedded(&state, &tenant, fresh, fresh_embedding.clone()).await;
        assert!(state.inner.schemas.read().await.contains_key(&scoped));

        // Now insert the EXPIRED trace with an explicit past `_expires_at`.
        let schema = state
            .inner
            .schemas
            .read()
            .await
            .get(&scoped)
            .cloned()
            .unwrap();
        // ttl small enough that ts(secs=1) + ttl is still in the past.
        let raw = trace_record_map(&expired, 10, vec![1.0f32, 0.0, 0.0, 0.0]);
        // Sanity: the stamped expiry is in the past.
        assert!(matches!(
            raw.get(EXPIRES_AT_FIELD),
            Some(serde_json::Value::Number(_))
        ),);
        let record = crate::rest::records::parse_record_public(&raw, &schema).unwrap();
        assert!(
            record.is_expired_at(now_secs),
            "the seeded expired trace must read as expired at now"
        );
        crate::agent::index::insert_only(&state, &scoped, &schema, record, "test").await;

        // recent_traces (the cluster_traces enumeration) must drop the expired
        // trace and keep the fresh one.
        let recent = recent_traces(&state, &scoped, MAX_LIMIT).await;
        let prompts: Vec<&str> = recent.iter().map(|t| t.prompt.as_str()).collect();
        assert!(
            prompts.contains(&"fresh prompt"),
            "fresh trace must be enumerated; got {prompts:?}"
        );
        assert!(
            !prompts.contains(&"expired prompt"),
            "expired trace must be excluded from recent_traces; got {prompts:?}"
        );

        // The search hydration path (`lookup_record_at`) must also drop the
        // expired record: a vector search for the expired record's own
        // embedding returns it as a candidate id, but hydration filters it out
        // by `_expires_at`, so it never becomes a TraceHit.
        let request = VectorSearchRequest {
            field: EMBEDDING_FIELD.to_owned(),
            query_vector: vec![1.0f32, 0.0, 0.0, 0.0],
            k: MAX_LIMIT,
            metric: DistanceMetric::Cosine,
            filter: None,
            filter_expr: None,
            truncate_to: None,
            read_ts: None,
        };
        let sealed = {
            let executors = state.inner.query_executors.read().await;
            let executor = executors.get(&scoped).expect("executor exists");
            executor.search(&request).expect("search runs").0.results
        };
        let growing = super::super::index::search_live_growing(&state, &scoped, &request).await;
        let merged = super::super::index::merge_top_k(sealed, growing, request.k);
        let now_secs2 = now_ms() / 1000;
        let managers = state.inner.segment_managers.read().await;
        let shard_map = managers.get(&scoped).unwrap();
        let hydrated: Vec<String> = merged
            .iter()
            .filter_map(|r| {
                shard_map
                    .values()
                    .find_map(|sm| sm.lookup_record_at(r.id, now_secs2))
            })
            .map(|rec| string_field(&rec, PROMPT_FIELD))
            .collect();
        assert!(
            !hydrated.contains(&"expired prompt".to_owned()),
            "expired trace must be filtered out by lookup_record_at; got {hydrated:?}"
        );
        assert!(
            hydrated.contains(&"fresh prompt".to_owned()),
            "fresh trace must survive hydration; got {hydrated:?}"
        );
    }

    #[tokio::test]
    async fn empty_query_and_zero_limit_short_circuit() {
        let tmp = tempfile::tempdir().unwrap();
        let state = AppState::new(tmp.path());
        let tenant = TenantId::default_tenant();
        assert!(
            search_traces(&state, &tenant, "   ", DEFAULT_LIMIT)
                .await
                .is_empty()
        );
        assert!(search_traces(&state, &tenant, "q", 0).await.is_empty());
        assert!(cluster_traces(&state, &tenant, 0, None).await.is_empty());
    }
}

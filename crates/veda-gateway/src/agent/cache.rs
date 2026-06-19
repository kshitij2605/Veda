//! Semantic response cache (Phase 1: LLM observability & optimization).
//!
//! Opt-in: an agent with `cache == true` has each prompt embedded and
//! used as a vector key into Veda's own store. A near-duplicate prompt
//! (top-1 cosine similarity at or above a HIGH threshold, not expired) is
//! served straight from cache — skipping the LLM entirely (cost + latency
//! win). This is purely vector-native; an inverted-index store cannot do
//! semantic near-duplicate matching.
//!
//! The whole feature is **best-effort, off-lock, and fail-open**, mirroring
//! `agent/memory.rs`:
//!   * embedding is done OFF any lock (clone the text `EmbeddingClient`
//!     config out from under the registry read-lock, drop the lock, then
//!     `embed_text(..).await` with no lock held);
//!   * a lookup failure (missing collection, no embedder, embed/search
//!     error, below threshold, expired) degrades to a **miss** (`None`) and
//!     never errors — the turn runs normally;
//!   * a store failure is swallowed (logged at debug).
//!
//! The prompt is embedded (not the response) because the prompt is the
//! cache *key*; the response is stored verbatim and returned on a hit.
//!
//! TTL / bounds: each entry carries an `expires_at` (epoch ms). A hit whose
//! `expires_at` is in the past is treated as a miss (expiry is checked in
//! Rust after the search, so it needs no OR-filter support). The default
//! TTL is [`DEFAULT_TTL_MS`] (24h). The cache collection otherwise grows
//! within the same per-tenant collection bounds as any other collection;
//! periodic compaction / eviction is out of scope for Phase 1 (documented).

use serde::{Deserialize, Serialize};

use veda_core::embedding::Modality;
use veda_core::embedding_client::EmbeddingClient;
use veda_core::record::Record;
use veda_core::schema::{CollectionSchema, FieldDefinition};
use veda_core::types::{AutoEmbedConfig, AutoEmbedModality, DistanceMetric, FieldType};
use veda_query::search::VectorSearchRequest;

use super::model::Agent;
use crate::rest::state::AppState;
use crate::tenant::{self, TenantId};

/// User-facing name of the per-tenant semantic-cache collection. Tenant
/// scoping (`tenant::scope`) is applied on top, so each tenant gets its
/// own isolated `_llm_cache` collection.
pub const CACHE_COLLECTION: &str = "_llm_cache";

/// The prompt field — the cache key text, embedded into [`EMBEDDING_FIELD`].
const PROMPT_FIELD: &str = "prompt";
/// The cached response served verbatim on a hit.
const RESPONSE_FIELD: &str = "response";
/// The model the response was produced by (filtered on lookup so a hit is
/// never served from a different model's answer).
const MODEL_FIELD: &str = "model";
/// Creation timestamp (epoch ms).
const TS_FIELD: &str = "ts";
/// Expiry timestamp (epoch ms); `0` means "never expires".
const EXPIRES_AT_FIELD: &str = "expires_at";
/// The vector field auto-embedded over [`PROMPT_FIELD`].
const EMBEDDING_FIELD: &str = "embedding";

/// Default cosine-similarity threshold for serving a cached response.
/// Deliberately HIGH: a cache hit must be a near-duplicate prompt, since a
/// false hit serves a *wrong* answer without ever calling the model.
pub const DEFAULT_SIMILARITY_THRESHOLD: f32 = 0.97;

/// Default entry time-to-live: 24 hours, in milliseconds.
pub const DEFAULT_TTL_MS: i64 = 24 * 60 * 60 * 1000;

/// A cached response served on a lookup hit.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct CachedResponse {
    pub response: String,
    /// Cosine similarity of the stored prompt to the lookup prompt, in
    /// `[0, 1]` (`1.0` == identical direction).
    pub similarity: f32,
    /// Creation timestamp of the cached entry (epoch ms).
    pub cached_at_ms: i64,
}

/// Whether the given agent has opted into the semantic response cache.
/// Kept as a tiny named guard so the wiring in `run_agent_turn` is
/// unit-testable without standing up an LLM.
#[inline]
pub fn should_use_cache(agent: &Agent) -> bool {
    agent.cache
}

/// Convert a cosine *distance* (as returned by the vector search, where
/// `0.0` is identical and `2.0` is opposite) into a cosine *similarity*
/// in `[-1, 1]` (`1.0` == identical). Pure.
#[inline]
pub fn distance_to_similarity(distance: f32) -> f32 {
    1.0 - distance
}

/// Pure cache decision: given the top hit's cosine *distance*, the
/// configured `threshold`, the entry's `expires_at` (epoch ms; `0` == never)
/// and the current time `now_ms`, decide whether to serve the cache.
/// Returns `Some(similarity)` on a hit, `None` on a miss (below threshold
/// or expired). Pure + directly testable.
pub fn cache_decision(
    distance: f32,
    threshold: f32,
    expires_at: i64,
    now_ms: i64,
) -> Option<f32> {
    let similarity = distance_to_similarity(distance);
    if similarity < threshold {
        return None;
    }
    if expires_at != 0 && expires_at <= now_ms {
        return None;
    }
    Some(similarity)
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

/// Build the tenant-scoped cache collection schema for a given embedding
/// dimension. Pure — no I/O, so it is directly unit-testable.
pub fn cache_schema(scoped_name: &str, dimension: u32) -> Result<CollectionSchema, String> {
    CollectionSchema::builder(scoped_name)
        .add_field(FieldDefinition::new(PROMPT_FIELD, FieldType::String))
        .add_field(FieldDefinition::new(RESPONSE_FIELD, FieldType::String))
        .add_field(FieldDefinition::new(MODEL_FIELD, FieldType::String).with_filterable(true))
        .add_field(FieldDefinition::new(TS_FIELD, FieldType::Int64).with_filterable(true))
        .add_field(
            FieldDefinition::new(EXPIRES_AT_FIELD, FieldType::Int64).with_filterable(true),
        )
        .add_field(
            FieldDefinition::vector(EMBEDDING_FIELD, dimension, DistanceMetric::Cosine)
                .with_auto_embed(AutoEmbedConfig {
                    source: PROMPT_FIELD.to_owned(),
                    modality: AutoEmbedModality::Text,
                }),
        )
        .build()
        .map_err(|e| e.to_string())
}

/// Build the JSON record map for one cache entry (the shape passed to
/// `parse_record_public`). Pure — the `embedding` is supplied explicitly.
fn entry_record_map(
    prompt: &str,
    response: &str,
    model: &str,
    ts: i64,
    expires_at: i64,
    embedding: Vec<f32>,
) -> serde_json::Map<String, serde_json::Value> {
    let mut m = serde_json::Map::new();
    m.insert(PROMPT_FIELD.to_owned(), serde_json::Value::String(prompt.to_owned()));
    m.insert(
        RESPONSE_FIELD.to_owned(),
        serde_json::Value::String(response.to_owned()),
    );
    m.insert(MODEL_FIELD.to_owned(), serde_json::Value::String(model.to_owned()));
    m.insert(TS_FIELD.to_owned(), serde_json::Value::Number(ts.into()));
    m.insert(
        EXPIRES_AT_FIELD.to_owned(),
        serde_json::Value::Number(expires_at.into()),
    );
    m.insert(
        EMBEDDING_FIELD.to_owned(),
        serde_json::Value::Array(
            embedding
                .into_iter()
                .map(|f| {
                    serde_json::Number::from_f64(f as f64)
                        .map(serde_json::Value::Number)
                        .unwrap_or(serde_json::Value::Null)
                })
                .collect(),
        ),
    );
    m
}

/// Ensure the tenant's cache collection exists, creating it (schema
/// persisted to disk + materialized into in-process state) sized for the
/// given embedding `dimension` if absent. Best-effort: returns the scoped
/// collection name on success, `None` on any failure. Idempotent — an
/// already-present collection is left alone. Mirrors
/// `memory::ensure_memory_collection`.
async fn ensure_cache_collection(
    state: &AppState,
    tenant: &TenantId,
    dimension: u32,
) -> Option<String> {
    let scoped = tenant::scope(tenant, CACHE_COLLECTION);
    {
        let schemas = state.inner.schemas.read().await;
        if schemas.contains_key(&scoped) {
            return Some(scoped);
        }
    }

    let schema = cache_schema(&scoped, dimension).ok()?;

    // Persist the schema marker so the collection survives a restart,
    // mirroring `create_collection_impl`.
    let collection_dir = state.inner.data_dir.join(&scoped);
    std::fs::create_dir_all(&collection_dir).ok()?;
    let schema_json = serde_json::to_vec_pretty(&schema).ok()?;
    std::fs::write(collection_dir.join("collection.json"), &schema_json).ok()?;

    crate::rest::collections::materialize_collection_locally(state, &schema, &collection_dir)
        .await
        .ok()?;
    Some(scoped)
}

/// Look up a near-duplicate prompt in the tenant's cache for `model`.
///
/// Embeds `prompt` off-lock, vector-searches the cache collection filtered
/// to `model`, takes the top-1 hit, and returns it when the cosine
/// similarity is at or above `threshold` AND the entry has not expired.
/// Returns `None` (never errors) on: cache collection absent, no embedder,
/// embed/search failure, below threshold, expired. **Fail-open** — a lookup
/// problem yields a miss and the turn runs normally.
pub async fn lookup(
    state: &AppState,
    tenant: &TenantId,
    model: &str,
    prompt: &str,
    threshold: f32,
) -> Option<CachedResponse> {
    if prompt.trim().is_empty() {
        return None;
    }
    let scoped = tenant::scope(tenant, CACHE_COLLECTION);

    // The cache collection must already exist; never create it on a read.
    {
        let schemas = state.inner.schemas.read().await;
        if !schemas.contains_key(&scoped) {
            return None;
        }
    }

    // Embed the prompt off any lock.
    let client = text_embedder(state).await?;
    let vector = match client.embed_text(prompt).await {
        Ok(v) => v,
        Err(e) => {
            tracing::debug!(error = %e, "llm cache: prompt embed failed; treating as miss");
            return None;
        }
    };

    // Filter to this model so a hit is never served from a different
    // model's answer. The model name is operator-controlled config.
    // NB: the filter grammar's equality operator is a single `=` (see
    // veda-query FilterParser); `==` fails to parse and would silently
    // turn every lookup into a miss.
    let filter = format!("{MODEL_FIELD} = \"{model}\"");
    let request = VectorSearchRequest {
        field: EMBEDDING_FIELD.to_owned(),
        query_vector: vector,
        k: 1,
        metric: DistanceMetric::Cosine,
        filter: Some(filter),
        filter_expr: None,
        truncate_to: None,
        read_ts: None,
    };

    let executors = state.inner.query_executors.read().await;
    let executor = executors.get(&scoped)?;
    let response = match executor.search(&request) {
        Ok((response, _)) => response,
        Err(e) => {
            tracing::debug!(error = %e, "llm cache: lookup search failed; treating as miss");
            return None;
        }
    };
    drop(executors);

    // The executor only sees sealed segments. Merge in the live growing
    // segment (where a just-stored entry lives until it seals) so a fresh
    // entry is found immediately, without any per-write seal/rebuild.
    let growing = super::index::search_live_growing(state, &scoped, &request).await;
    let results = super::index::merge_top_k(response.results, growing, request.k);

    let top = results.first()?;
    let now_ms = now_ms();
    let now_secs = now_ms / 1000;

    // Hydrate the stored entry via an O(1) record lookup (mirrors search).
    let managers = state.inner.segment_managers.read().await;
    let shard_map = managers.get(&scoped)?;
    let rec = shard_map
        .values()
        .find_map(|sm| sm.lookup_record_at(top.id, now_secs))?;

    let expires_at = int_field(&rec, EXPIRES_AT_FIELD);
    let similarity = cache_decision(top.distance, threshold, expires_at, now_ms)?;

    Some(CachedResponse {
        response: string_field(&rec, RESPONSE_FIELD),
        similarity,
        cached_at_ms: int_field(&rec, TS_FIELD),
    })
}

/// Store one prompt → response entry in the tenant's cache, auto-creating
/// the collection on first write. Best-effort and non-fatal: embeds the
/// prompt off-lock, then inserts the record (with the explicitly-computed
/// vector) directly into the segment manager — the same in-process insert
/// path `upsert_record` uses. The entry expires at `ts + DEFAULT_TTL_MS`.
/// Any failure is logged at debug and swallowed.
pub async fn store(
    state: &AppState,
    tenant: &TenantId,
    model: &str,
    prompt: &str,
    response: &str,
    ts: i64,
) {
    if prompt.trim().is_empty() || response.trim().is_empty() {
        return;
    }

    // Embed the PROMPT (the cache key) off any lock.
    let Some(client) = text_embedder(state).await else {
        return;
    };
    let embedding = match client.embed_text(prompt).await {
        Ok(v) => v,
        Err(e) => {
            tracing::debug!(error = %e, "llm cache: prompt embed failed; skipping store");
            return;
        }
    };

    store_embedded(state, tenant, model, prompt, response, ts, embedding).await;
}

/// Insert one already-embedded cache entry and make it **immediately
/// searchable**. Ensures the collection, parses the record, inserts it into
/// the routed shard, then seals the growing segment and rebuilds the
/// collection's query executor — the same seal + executor-rebuild the
/// gateway `embed_insert` path performs. Without that final step the record
/// only lives in the live segment manager's growing segment, which the
/// executor (a separate manager built from disk) never sees, so a vector
/// search misses it. Best-effort: any failure is logged at debug and
/// swallowed. The vector is supplied explicitly so this is unit-testable
/// without a live embedder.
async fn store_embedded(
    state: &AppState,
    tenant: &TenantId,
    model: &str,
    prompt: &str,
    response: &str,
    ts: i64,
    embedding: Vec<f32>,
) {
    let dimension = embedding.len() as u32;

    let Some(scoped) = ensure_cache_collection(state, tenant, dimension).await else {
        tracing::debug!("llm cache: could not ensure cache collection; skipping store");
        return;
    };

    let schema = {
        let schemas = state.inner.schemas.read().await;
        match schemas.get(&scoped).cloned() {
            Some(s) => s,
            None => return,
        }
    };
    let expires_at = ts.saturating_add(DEFAULT_TTL_MS);
    let raw = entry_record_map(prompt, response, model, ts, expires_at, embedding);
    let record = match crate::rest::records::parse_record_public(&raw, &schema) {
        Ok(r) => r,
        Err(e) => {
            tracing::debug!(error = %e, "llm cache: record parse failed; skipping store");
            return;
        }
    };

    crate::agent::index::insert_only(state, &scoped, &schema, record, "llm cache").await;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn should_use_cache_reflects_flag() {
        let mut a = crate::agent::model::default_agent();
        assert!(!should_use_cache(&a), "default agent has cache off");
        a.cache = true;
        assert!(should_use_cache(&a), "cache=true agent uses cache");
    }

    #[test]
    fn distance_similarity_round_trip() {
        // identical → distance 0 → similarity 1; orthogonal → 1 → 0.
        assert!((distance_to_similarity(0.0) - 1.0).abs() < 1e-6);
        assert!((distance_to_similarity(1.0) - 0.0).abs() < 1e-6);
        assert!((distance_to_similarity(0.02) - 0.98).abs() < 1e-6);
    }

    #[test]
    fn decision_hit_at_or_above_threshold() {
        // similarity 0.98 (distance 0.02) ≥ 0.97 → hit, no expiry.
        let s = cache_decision(0.02, DEFAULT_SIMILARITY_THRESHOLD, 0, 1_000).expect("hit");
        assert!((s - 0.98).abs() < 1e-6);
        // exactly at threshold is a hit (>=).
        assert!(cache_decision(0.03, 0.97, 0, 1_000).is_some());
    }

    #[test]
    fn decision_miss_below_threshold() {
        // similarity 0.90 (distance 0.10) < 0.97 → miss.
        assert!(cache_decision(0.10, DEFAULT_SIMILARITY_THRESHOLD, 0, 1_000).is_none());
    }

    #[test]
    fn decision_miss_when_expired() {
        // similarity is high enough, but the entry expired before now.
        assert!(
            cache_decision(0.0, DEFAULT_SIMILARITY_THRESHOLD, 500, 1_000).is_none(),
            "expired entry must be a miss even at perfect similarity"
        );
        // expires_at == 0 means never-expires → still a hit.
        assert!(cache_decision(0.0, DEFAULT_SIMILARITY_THRESHOLD, 0, 1_000).is_some());
        // expires_at in the future → hit.
        assert!(cache_decision(0.0, DEFAULT_SIMILARITY_THRESHOLD, 2_000, 1_000).is_some());
    }

    #[test]
    fn cache_schema_marks_model_filterable_and_vector_dim() {
        let schema = cache_schema("t1::_llm_cache", 8).expect("schema builds");
        let model_field = schema.get_field(MODEL_FIELD).expect("model field");
        assert!(model_field.filterable(), "model must be filterable for per-model lookup");
        let expires_field = schema.get_field(EXPIRES_AT_FIELD).expect("expires_at field");
        assert!(expires_field.filterable(), "expires_at must be filterable");
        let vec_field = schema.get_field(EMBEDDING_FIELD).expect("embedding field");
        match vec_field.field_type() {
            FieldType::Vector { dimension, auto_embed, .. } => {
                assert_eq!(*dimension, 8);
                assert!(auto_embed.is_some(), "embedding field must carry auto_embed config");
            }
            other => panic!("expected vector field type, got {other:?}"),
        }
    }

    #[test]
    fn entry_record_map_round_trips_through_schema() {
        let dim = 4u32;
        let schema = cache_schema("t1::_llm_cache", dim).expect("schema builds");
        let raw = entry_record_map(
            "what index does veda use?",
            "Veda uses HNSW.",
            "gpt-4o-mini",
            12345,
            12345 + DEFAULT_TTL_MS,
            vec![0.1, 0.2, 0.3, 0.4],
        );
        let rec = crate::rest::records::parse_record_public(&raw, &schema).expect("record parses");
        assert_eq!(string_field(&rec, PROMPT_FIELD), "what index does veda use?");
        assert_eq!(string_field(&rec, RESPONSE_FIELD), "Veda uses HNSW.");
        assert_eq!(string_field(&rec, MODEL_FIELD), "gpt-4o-mini");
        assert_eq!(int_field(&rec, TS_FIELD), 12345);
        assert_eq!(int_field(&rec, EXPIRES_AT_FIELD), 12345 + DEFAULT_TTL_MS);
        match rec.get(EMBEDDING_FIELD) {
            Some(veda_core::record::FieldValue::Vector(v)) => assert_eq!(v.dimension(), 4),
            other => panic!("expected vector field, got {other:?}"),
        }
    }

    // ── Best-effort / off-lock wiring against a bare AppState ────────────
    //
    // The test env has no live embedder registered, so both entry points
    // must degrade gracefully (never panic, never error): `lookup` returns
    // None (a miss → the turn runs normally) and `store` is a no-op that
    // does NOT create the cache collection. This mirrors how the memory
    // tests handle the absent embedder; the live embed/search path needs a
    // running embedder and is validated against a deployment.

    #[tokio::test]
    async fn lookup_returns_none_without_embedder_or_collection() {
        let tmp = tempfile::tempdir().unwrap();
        let state = AppState::new(tmp.path());
        let tenant = TenantId::default_tenant();
        let hit = lookup(&state, &tenant, "gpt-4o-mini", "anything", DEFAULT_SIMILARITY_THRESHOLD)
            .await;
        assert!(hit.is_none(), "no embedder/collection → miss (None), never an error");
    }

    #[tokio::test]
    async fn store_is_noop_without_embedder() {
        let tmp = tempfile::tempdir().unwrap();
        let state = AppState::new(tmp.path());
        let tenant = TenantId::default_tenant();
        // Must not panic; best-effort no-op when nothing can embed.
        store(&state, &tenant, "gpt-4o-mini", "a prompt", "a response", 1).await;
        // No embedder → the collection is never created (store bails before
        // ensure_cache_collection).
        let scoped = tenant::scope(&tenant, CACHE_COLLECTION);
        let schemas = state.inner.schemas.read().await;
        assert!(
            !schemas.contains_key(&scoped),
            "store must not create the cache collection when it cannot embed",
        );
    }

    // ── Searchability regression test (the missing test) ────────────────
    //
    // This is the test that would have caught the shipped bug: a stored
    // cache entry must be findable by a vector search for its own embedding
    // *immediately* after store. It supplies the embedding directly (no live
    // embedder needed) via the `store_embedded` seam, then runs the same
    // executor + live-growing merge `lookup` uses and asserts the record is
    // returned. The fresh entry lives in the live growing segment (the
    // executor is sealed-only), so the live-growing search is what surfaces
    // it — with NO per-write seal/rebuild.
    #[tokio::test]
    async fn stored_entry_is_immediately_searchable() {
        let tmp = tempfile::tempdir().unwrap();
        let state = AppState::new(tmp.path());
        let tenant = TenantId::default_tenant();
        let model = "test-model";
        let embedding = vec![1.0f32, 0.0, 0.0, 0.0];

        store_embedded(
            &state,
            &tenant,
            model,
            "what index does veda use?",
            "Veda uses HNSW.",
            1_000,
            embedding.clone(),
        )
        .await;

        let scoped = tenant::scope(&tenant, CACHE_COLLECTION);
        assert!(
            state.inner.schemas.read().await.contains_key(&scoped),
            "store_embedded must create the cache collection"
        );

        // Run the same executor + live-growing merge `lookup` performs.
        let request = VectorSearchRequest {
            field: EMBEDDING_FIELD.to_owned(),
            query_vector: embedding,
            k: 1,
            metric: DistanceMetric::Cosine,
            filter: Some(format!("{MODEL_FIELD} = \"{model}\"")),
            filter_expr: None,
            truncate_to: None,
            read_ts: None,
        };
        let sealed = {
            let executors = state.inner.query_executors.read().await;
            let executor = executors.get(&scoped).expect("executor exists for cache collection");
            executor.search(&request).expect("search runs").0.results
        };
        let growing = super::super::index::search_live_growing(&state, &scoped, &request).await;
        let results = super::super::index::merge_top_k(sealed, growing, request.k);
        assert_eq!(
            results.len(),
            1,
            "the stored cache entry must be returned by a vector search for its own embedding"
        );
    }

    // Bounded-cost regression: storing N distinct entries must NOT produce N
    // sealed segments and must NOT trigger a per-write seal/rebuild. With the
    // O(n)-per-write seal+rebuild removed, every store lands in the growing
    // segment and nothing seals until the size threshold — so the sealed
    // segment count stays 0 for these tiny records. This is the test that
    // pins the perf fix.
    #[tokio::test]
    async fn storing_many_entries_does_not_explode_sealed_segments() {
        let tmp = tempfile::tempdir().unwrap();
        let state = AppState::new(tmp.path());
        let tenant = TenantId::default_tenant();
        let model = "test-model";

        const N: usize = 30;
        for i in 0..N {
            // Distinct unit vectors so each is a separate entry.
            let mut embedding = vec![0.0f32; 8];
            embedding[i % 8] = 1.0 + i as f32; // distinct directions/magnitudes
            store_embedded(
                &state,
                &tenant,
                model,
                &format!("prompt number {i}"),
                &format!("response number {i}"),
                1_000 + i as i64,
                embedding,
            )
            .await;
        }

        let scoped = tenant::scope(&tenant, CACHE_COLLECTION);
        let managers = state.inner.segment_managers.read().await;
        let shard_map = managers.get(&scoped).expect("cache collection exists");
        let sealed_total: usize = shard_map.values().map(|sm| sm.sealed_segments().len()).sum();
        let growing_total: u32 = shard_map.values().map(|sm| sm.growing_segment().count()).sum();
        assert_eq!(
            sealed_total, 0,
            "storing {N} tiny entries must not seal a segment per write (got {sealed_total} sealed)"
        );
        assert_eq!(
            growing_total as usize, N,
            "all {N} entries must live in the live growing segment"
        );
    }

    #[tokio::test]
    async fn empty_prompt_and_empty_response_short_circuit() {
        let tmp = tempfile::tempdir().unwrap();
        let state = AppState::new(tmp.path());
        let tenant = TenantId::default_tenant();
        assert!(
            lookup(&state, &tenant, "m", "   ", DEFAULT_SIMILARITY_THRESHOLD).await.is_none()
        );
        // Empty prompt/response store is a no-op (returns before embedding).
        store(&state, &tenant, "m", "   ", "resp", 1).await;
        store(&state, &tenant, "m", "prompt", "   ", 1).await;
        let scoped = tenant::scope(&tenant, CACHE_COLLECTION);
        assert!(!state.inner.schemas.read().await.contains_key(&scoped));
    }
}

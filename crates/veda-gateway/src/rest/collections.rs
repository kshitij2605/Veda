use std::collections::HashMap;

use axum::Json;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use serde::{Deserialize, Serialize};
use veda_core::metrics as m;
use veda_core::schema::{CollectionSchema, FieldDefinition};
use veda_core::types::{DistanceMetric, FieldType};
use veda_query::distributed::DistributedQueryExecutor;
use veda_query::local_shard::LocalShardSearcher;
use veda_storage::segment_manager::SegmentManager;

use super::state::AppState;
use crate::tenant::{self, Tenant};

/// 307 Temporary Redirect pointing at `{leader_base}{path}`.
/// Used by mutating handlers when this node is a Raft follower.
/// 307 (not 302) is the right code: it preserves the request
/// method + body, so `POST` stays a `POST` on the leader.
pub(crate) fn raft_redirect(leader_base: &str, path: &str) -> axum::response::Response {
    use axum::http::{HeaderValue, header};
    use axum::response::IntoResponse;
    let url = format!("{}{}", leader_base.trim_end_matches('/'), path);
    let mut resp = (
        StatusCode::TEMPORARY_REDIRECT,
        Json(serde_json::json!({
            "redirect": url,
            "reason": "this node is a raft follower; resubmit to the leader",
        })),
    )
        .into_response();
    if let Ok(v) = HeaderValue::from_str(&url) {
        resp.headers_mut().insert(header::LOCATION, v);
    }
    resp
}

// ── Request / Response types ─────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct CreateCollectionRequest {
    pub name: String,
    pub fields: Vec<FieldDef>,
    #[serde(default = "default_shard_count")]
    pub shard_count: u32,
    #[serde(default = "default_replication_factor")]
    pub replication_factor: u32,
    /// Default physical layout for the stored-fields data of new
    /// sealed segments. `"jsonl"` (default — legacy + small records)
    /// or `"columnar"` (per-field column files, projection-friendly).
    /// Per-segment metadata still records the format that was
    /// actually written, so this can be changed later for new
    /// segments without rewriting existing ones.
    #[serde(default)]
    pub stored_format: Option<String>,
    /// Per-collection opt-in for the Phase C primary-write path.
    /// When `Some(true)`, writes go through the per-shard
    /// `veda-log` for THIS collection regardless of the
    /// `VEDA_LOG_PRIMARY` global env flag. When `None` (the
    /// default), the env flag decides.
    #[serde(default)]
    pub use_log_primary: Option<bool>,
    /// Per-collection opt-in for the Phase F MVCC overlay.
    /// `Some(true)` enables it for this collection regardless
    /// of the `VEDA_MVCC` env flag. `None` falls back to env.
    #[serde(default)]
    pub use_mvcc: Option<bool>,
    /// Phase G: atomic create-and-load. When present, these
    /// records are bulk-inserted immediately after the
    /// collection materialises. If any insert fails the
    /// collection is rolled back with a `DropCollection`
    /// compensating action — the client sees one
    /// success-or-failure response and either everything lands
    /// or nothing does.
    #[serde(default)]
    pub seed_records: Option<Vec<serde_json::Map<String, serde_json::Value>>>,
    /// Per-collection Row-Level Security. When set, non-Admin users can only
    /// read/write records where this metadata field equals their own user id
    /// (the JWT `sub` claim). Admin users bypass the filter entirely.
    #[serde(default)]
    pub rls_owner_field: Option<String>,
}

/// Default growing-segment seal threshold: 512 MiB. Vector ANN indexes
/// (HNSW / IVF-PQ / DiskANN / RaBitQ* / SQ) build only on *sealed*
/// segments; until the growing segment hits this size, search brute-forces
/// the unsealed data with Flat. The default is fine for production but far
/// too large for small datasets / tests, which would otherwise never seal
/// and never exercise their configured index type.
pub(crate) const DEFAULT_SEAL_THRESHOLD_BYTES: usize = 512 * 1024 * 1024;

/// Resolve the growing-segment seal threshold in bytes. Reads
/// `VEDA_SEAL_THRESHOLD` (a byte count); falls back to
/// [`DEFAULT_SEAL_THRESHOLD_BYTES`] when unset or unparseable. Lowering it
/// (e.g. `VEDA_SEAL_THRESHOLD=4194304` for 4 MiB) lets small datasets seal
/// so the collection's declared vector index actually builds.
pub(crate) fn seal_threshold_bytes() -> usize {
    parse_seal_threshold(std::env::var("VEDA_SEAL_THRESHOLD").ok())
}

/// Pure parse of the `VEDA_SEAL_THRESHOLD` value. A positive integer wins;
/// anything else (absent, blank, non-numeric, zero) falls back to the
/// default. Split out so it's testable without mutating the process env.
fn parse_seal_threshold(raw: Option<String>) -> usize {
    raw.and_then(|s| s.trim().parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(DEFAULT_SEAL_THRESHOLD_BYTES)
}

fn default_shard_count() -> u32 {
    1
}

fn default_replication_factor() -> u32 {
    1
}

#[derive(Debug, Deserialize)]
pub struct FieldDef {
    pub name: String,
    #[serde(rename = "type")]
    pub field_type: String,
    #[serde(default)]
    pub filterable: bool,
    pub dimension: Option<u32>,
    pub metric: Option<String>,
    /// Optional embedding-model identifier for vector fields. When set,
    /// embed-insert calls must use the same model.
    pub model: Option<String>,
    /// Maximum number of vectors per record for multi-vector fields.
    /// 0 / absent = unbounded.
    pub max_vectors: Option<u32>,
    /// Matryoshka truncation ladder. When set, per-dim HNSW indexes are
    /// built so search with `truncate_to=N` (for N in this list) hits a
    /// real ANN index instead of brute-force scanning.
    pub mrl_dims: Option<Vec<u32>>,
    /// Per-field auto-embed config. JSON form:
    /// `{"source": "content", "modality": "text"}`. When present, the
    /// process-document handler uses this to pick the source field and
    /// embedder for this vector field instead of the name-based
    /// heuristic.
    pub auto_embed: Option<veda_core::types::AutoEmbedConfig>,
    /// Build a full-text BM25 index for this field. Only meaningful for
    /// `string` fields. When true, the schema exposes a `bm25` query
    /// type and the `hybrid` (dense + BM25) query type.
    #[serde(default)]
    pub bm25: bool,
    /// Tokenizer for BM25. Recognized: `"cjk_bigram"` (default, good
    /// for mixed-script and CJK), `"whitespace"` (cheaper for
    /// ASCII-only). Unknown names fall back to the default.
    #[serde(default)]
    pub bm25_tokenizer: Option<String>,
    /// Vector index type. Recognized: `"hnsw"` (default for vector
    /// fields), `"diskann"`, `"ivfpq"`, `"flat"`, `"rabitq"` (1-bit
    /// quantized brute-force scan; ~32× memory savings at the cost of
    /// some recall, well-suited for ≥ 1 M-row collections). Unknown
    /// values fall back to `hnsw`.
    #[serde(default)]
    pub index: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct CreateCollectionResponse {
    pub name: String,
    pub status: String,
    /// Set only when the request supplied `seed_records`. Mirrors
    /// the count from the bulk-insert that ran atomically with
    /// the create. Absent / null when no seeds were requested.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seeded: Option<usize>,
}

/// Per-shard outcome of a force-seal: how many segments that shard
/// sealed (0 if its growing segment was empty).
#[derive(Debug, Serialize)]
pub struct ShardSealResult {
    pub shard_id: u32,
    pub sealed_segments: usize,
}

#[derive(Debug, Serialize)]
pub struct SealResponse {
    pub collection: String,
    /// Total number of segments sealed across all shards.
    pub sealed_segments: usize,
    pub per_shard: Vec<ShardSealResult>,
}

/// Per-shard outcome of a force-merge: sealed-segment count before and after
/// consolidation.
#[derive(Debug, Serialize)]
pub struct ShardOptimizeResult {
    pub shard_id: u32,
    pub segments_before: usize,
    pub segments_after: usize,
}

#[derive(Debug, Serialize)]
pub struct OptimizeResponse {
    pub collection: String,
    pub per_shard: Vec<ShardOptimizeResult>,
}

#[derive(Debug, Serialize)]
pub struct ListCollectionsResponse {
    pub collections: Vec<CollectionSummary>,
}

#[derive(Debug, Serialize)]
pub struct CollectionSummary {
    pub name: String,
    pub field_count: usize,
    pub shard_count: u32,
}

#[derive(Debug, Serialize)]
pub struct DescribeCollectionResponse {
    pub name: String,
    pub fields: Vec<FieldInfo>,
    pub shard_count: u32,
    pub replication_factor: u32,
}

#[derive(Debug, Serialize)]
pub struct FieldInfo {
    pub name: String,
    #[serde(rename = "type")]
    pub field_type: String,
    pub filterable: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dimension: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metric: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// The field's ANN index type ("Hnsw", "DiskAnn", "Flat", ...) — only
    /// present for vector/sparse_vector/multi_vector fields. Lets a caller
    /// confirm a `/reindex` request's `index_type` genuinely took effect.
    /// NOTE: this is `{:?}` (Debug) formatted, NOT the same string
    /// `ReindexRequest.index_type`/`FieldDef.index` accept as input — for
    /// compound variants (e.g. `RaBitQRerank` Debug-formats to
    /// "RaBitQRerank", but `parse_index_type` only accepts the underscored
    /// alias "rabitq_rerank") the two do NOT round-trip. This field is
    /// read-only observability; never feed it back into `parse_index_type`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub index_type: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct DropCollectionResponse {
    pub name: String,
    pub status: String,
}

#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    pub error: String,
}

// ── Handlers ─────────────────────────────────────────────────────────────

pub async fn create_collection(
    State(state): State<AppState>,
    Tenant(tenant): Tenant,
    Json(req): Json<CreateCollectionRequest>,
) -> axum::response::Response {
    if let Some(leader_url) = state.leader_rest_address().await {
        return raft_redirect(&leader_url, "/v1/collections");
    }
    create_collection_inner(state, tenant, req).await
}

async fn create_collection_inner(
    state: AppState,
    tenant: tenant::TenantId,
    req: CreateCollectionRequest,
) -> axum::response::Response {
    use axum::response::IntoResponse;
    match create_collection_impl(state, tenant, req).await {
        Ok((status, body)) => (status, body).into_response(),
        Err((status, body)) => (status, body).into_response(),
    }
}

async fn create_collection_impl(
    state: AppState,
    tenant: tenant::TenantId,
    req: CreateCollectionRequest,
) -> Result<(StatusCode, Json<CreateCollectionResponse>), (StatusCode, Json<ErrorResponse>)> {
    // Validate user-facing name first.
    let user_name = req.name.trim();
    if user_name.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "collection name must not be empty".into(),
            }),
        ));
    }
    // Scope the name to the caller's tenant. Two tenants with the
    // same user-facing name get distinct collections internally.
    let scoped_name = tenant::scope(&tenant, user_name);

    // Check for duplicate — in-memory state OR an existing on-disk
    // collection directory with our schema marker.
    let collection_dir = state.inner.data_dir.join(&scoped_name);
    let schema_marker = collection_dir.join("collection.json");
    {
        let schemas = state.inner.schemas.read().await;
        if schemas.contains_key(&scoped_name) || schema_marker.exists() {
            return Err((
                StatusCode::CONFLICT,
                Json(ErrorResponse {
                    error: format!("collection '{user_name}' already exists"),
                }),
            ));
        }
    }

    // Build schema under the scoped name. schema.name() will return
    // the scoped name; responses strip it back to user_name.
    let mut builder = CollectionSchema::builder(&scoped_name)
        .shard_count(req.shard_count)
        .replication_factor(req.replication_factor)
        .use_log_primary(req.use_log_primary)
        .use_mvcc(req.use_mvcc)
        .rls_owner_field(req.rls_owner_field.clone());
    if let Some(fmt) = req.stored_format.as_deref() {
        match fmt.to_lowercase().as_str() {
            "jsonl" | "columnar" => builder = builder.stored_format(fmt.to_lowercase()),
            other => {
                return Err((
                    StatusCode::BAD_REQUEST,
                    Json(ErrorResponse {
                        error: format!(
                            "stored_format must be \"jsonl\" or \"columnar\", got {other:?}"
                        ),
                    }),
                ));
            }
        }
    }

    for f in &req.fields {
        let field_def = parse_field_def(f)
            .map_err(|e| (StatusCode::BAD_REQUEST, Json(ErrorResponse { error: e })))?;
        builder = builder.add_field(field_def);
    }

    let schema = builder.build().map_err(|e| {
        (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: e.to_string(),
            }),
        )
    })?;

    // Persist the schema at the collection root so a restart can recover
    // both the in-memory state and the on-disk segments under each shard.
    // This is the source of truth for `recover_collections_from_disk`.
    std::fs::create_dir_all(&collection_dir).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: format!("failed to create collection dir: {e}"),
            }),
        )
    })?;
    let schema_json = serde_json::to_vec_pretty(&schema).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: format!("serialize schema: {e}"),
            }),
        )
    })?;
    std::fs::write(&schema_marker, &schema_json).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: format!("persist schema: {e}"),
            }),
        )
    })?;

    // Raft mode: route the creation through the consensus log
    // BEFORE materializing locally. Raft sees the scoped name —
    // it stays tenant-oblivious; the gateway is what binds tenants.
    {
        let raft_guard = state.inner.raft.read().await;
        if let Some(coord) = raft_guard.as_ref() {
            let cmd = veda_coordinator::raft_types::ClusterCommand::CreateCollection {
                name: scoped_name.clone(),
                schema_json: String::from_utf8(schema_json.clone()).unwrap_or_default(),
            };
            if let Err(e) = coord.execute(cmd).await {
                return Err((
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse {
                        error: format!("raft submit: {e}"),
                    }),
                ));
            }
        }
    }

    if let Err((status, e)) = materialize_collection_locally(&state, &schema, &collection_dir).await
    {
        return Err((status, e));
    }

    // Phase C: when the per-shard log applier is on, spawn one
    // task per shard of the newly-created collection so writes
    // submitted later through `insert_via_log` get applied.
    if crate::log_appliers::applier_enabled() {
        for shard_id in 0..schema.shard_count() {
            crate::log_appliers::spawn_one(
                state.clone(),
                scoped_name.clone(),
                shard_id,
                std::time::Duration::from_millis(20),
            );
        }
    }

    // Phase G: atomic create-and-load. The bulk insert runs
    // immediately after materialise; any failure rolls back the
    // collection via the regular drop path so the client never
    // sees a half-created collection. `seed_records` is the
    // user-facing field name, ordinary insert validation /
    // shard-routing applies — this is the same path POST
    // /v1/collections/:name/insert uses.
    let seeded_count = if let Some(records) = req.seed_records {
        let count = records.len();
        let seed_req = crate::rest::records::InsertRequest { records };
        match crate::rest::records::insert_impl(state.clone(), scoped_name.clone(), None, seed_req)
            .await
        {
            Ok(_) => Some(count),
            Err((seed_status, seed_err)) => {
                // Compensating action: drop the collection so the
                // create+seed pair is observable as
                // all-or-nothing.
                let _ =
                    drop_collection_impl(state.clone(), tenant.clone(), user_name.to_owned()).await;
                return Err((
                    seed_status,
                    Json(ErrorResponse {
                        error: format!("create-and-seed rolled back: {}", seed_err.0.error),
                    }),
                ));
            }
        }
    } else {
        None
    };

    Ok((
        StatusCode::CREATED,
        Json(CreateCollectionResponse {
            name: user_name.to_owned(),
            status: "created".into(),
            seeded: seeded_count,
        }),
    ))
}

/// Wire a parsed `CollectionSchema` into the local in-process
/// state: segment managers + query executor per shard, then insert
/// into `state.inner.schemas`. Idempotent — a collection already
/// present locally is left alone.
///
/// This is the shared code path between:
///   - direct REST creation (non-Raft mode)
///   - REST creation after a Raft commit (leader)
///   - the background reconcile task picking up entries from the
///     Raft state machine (followers)
pub(crate) async fn materialize_collection_locally(
    state: &AppState,
    schema: &CollectionSchema,
    collection_dir: &std::path::Path,
) -> Result<(), (StatusCode, Json<ErrorResponse>)> {
    let name = schema.name().to_owned();
    {
        let schemas = state.inner.schemas.read().await;
        if schemas.contains_key(&name) {
            return Ok(()); // already materialized
        }
    }
    let mut shard_managers: HashMap<u32, SegmentManager> = HashMap::new();
    let mut executor = DistributedQueryExecutor::new();
    let mvcc_on = schema.effective_use_mvcc();
    for shard_id in 0..schema.shard_count() {
        let mut sm = SegmentManager::new(
            schema.clone(),
            shard_id,
            collection_dir,
            seal_threshold_bytes(),
        )
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: format!("failed to create segment manager: {e}"),
                }),
            )
        })?;
        if mvcc_on {
            sm.enable_mvcc();
        }
        let searcher = LocalShardSearcher::new(
            SegmentManager::new(
                schema.clone(),
                shard_id,
                collection_dir,
                seal_threshold_bytes(),
            )
            .unwrap(),
        );
        executor.register_shard(shard_id, Box::new(searcher));
        shard_managers.insert(shard_id, sm);
    }
    {
        let mut schemas = state.inner.schemas.write().await;
        schemas.insert(name.clone(), schema.clone());
    }
    {
        let mut managers = state.inner.segment_managers.write().await;
        managers.insert(name.clone(), shard_managers);
    }
    {
        let mut executors = state.inner.query_executors.write().await;
        executors.insert(name.clone(), executor);
    }
    let collection_count = state.inner.schemas.read().await.len();
    metrics::gauge!(m::ACTIVE_COLLECTIONS).set(collection_count as f64);
    Ok(())
}

/// Reconcile the local Veda state with the Raft state machine's
/// view. Materializes any collection that exists in Raft state
/// but not locally; drops any local collection no longer in Raft
/// state. Called by the periodic reconcile task in raft mode and
/// also (synchronously, fall-through) by the create/drop handlers
/// so the leader's response reflects the new state.
pub async fn reconcile_collections_from_raft(state: &AppState) -> Result<(), String> {
    let raft_state = {
        let guard = state.inner.raft.read().await;
        match guard.as_ref() {
            Some(c) => c.query_state().await.map_err(|e| format!("{e}"))?,
            None => return Ok(()),
        }
    };

    let local_names: std::collections::HashSet<String> = {
        let schemas = state.inner.schemas.read().await;
        schemas.keys().cloned().collect()
    };
    let raft_names: std::collections::HashSet<String> =
        raft_state.collections.keys().cloned().collect();

    // Materialize: in Raft, not in local.
    for name in raft_names.difference(&local_names) {
        let Some(schema_json) = raft_state.collections.get(name) else {
            continue;
        };
        let schema: CollectionSchema = match serde_json::from_str(schema_json) {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!(
                    collection = %name,
                    error = %e,
                    "reconcile: schema JSON parse failed; skipping",
                );
                continue;
            }
        };
        let collection_dir = state.inner.data_dir.join(name);
        // Persist the schema marker so a restart can recover the
        // collection from disk independently of the Raft log.
        let _ = std::fs::create_dir_all(&collection_dir);
        let _ = std::fs::write(
            collection_dir.join("collection.json"),
            schema_json.as_bytes(),
        );
        if let Err(e) = materialize_collection_locally(state, &schema, &collection_dir).await {
            tracing::warn!(
                collection = %name,
                error = ?e.1.0.error,
                "reconcile: materialize failed",
            );
        } else {
            tracing::info!(collection = %name, "reconcile: materialized from raft state");
        }
    }

    // Drop: in local, not in Raft.
    for name in local_names.difference(&raft_names) {
        tracing::info!(collection = %name, "reconcile: dropping (no longer in raft state)");
        let _ = state.inner.schemas.write().await.remove(name);
        let _ = state.inner.segment_managers.write().await.remove(name);
        let _ = state.inner.query_executors.write().await.remove(name);
        let _ = state
            .inner
            .raft_materialized_records
            .write()
            .await
            .remove(name);
    }

    // Phase E hard-remove: record materialisation from
    // `raft_state.records` is gone. Records live in per-shard
    // `veda-log` now; follower nodes pick them up via the Phase
    // C `LogApplier`, not the metadata-Raft reconcile loop.
    let _ = &raft_state;
    Ok(())
}

pub async fn list_collections(
    State(state): State<AppState>,
    Tenant(tenant): Tenant,
) -> Json<ListCollectionsResponse> {
    let schemas = state.inner.schemas.read().await;
    let collections = schemas
        .iter()
        .filter_map(|(key, s)| {
            tenant::strip_prefix(&tenant, key).map(|user_name| CollectionSummary {
                name: user_name.to_owned(),
                field_count: s.fields().len(),
                shard_count: s.shard_count(),
            })
        })
        .collect();

    Json(ListCollectionsResponse { collections })
}

pub async fn describe_collection(
    State(state): State<AppState>,
    Tenant(tenant): Tenant,
    Path(name): Path<String>,
) -> Result<Json<DescribeCollectionResponse>, (StatusCode, Json<ErrorResponse>)> {
    let scoped = tenant::scope(&tenant, &name);
    // Resolve the alias — a tier-2/3 reindex swaps in a sibling whose schema
    // can differ (different embedding model, so a different vector dimension),
    // and describing the abandoned base would report a schema that no longer
    // matches what `/search` and `/insert` on this same name operate on.
    // `GET /v1/admin/collections/{name}/alias` still reports the mapping.
    let scoped = state.inner.aliases.resolve(&scoped).await;
    let schemas = state.inner.schemas.read().await;
    let schema = schemas.get(&scoped).ok_or_else(|| {
        (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: format!("collection '{name}' not found"),
            }),
        )
    })?;

    let fields = schema.fields().iter().map(field_to_info).collect();

    Ok(Json(DescribeCollectionResponse {
        name,
        fields,
        shard_count: schema.shard_count(),
        replication_factor: schema.replication_factor(),
    }))
}

pub async fn drop_collection(
    State(state): State<AppState>,
    Tenant(tenant): Tenant,
    Path(name): Path<String>,
) -> axum::response::Response {
    if let Some(leader_url) = state.leader_rest_address().await {
        return raft_redirect(&leader_url, &format!("/v1/collections/{name}"));
    }
    use axum::response::IntoResponse;
    match drop_collection_impl(state, tenant, name).await {
        Ok(body) => body.into_response(),
        Err((status, body)) => (status, body).into_response(),
    }
}

// ---------------------------------------------------------------
// Phase G — alias administration + reindex stub.
// ---------------------------------------------------------------

#[derive(Debug, Serialize, Deserialize)]
pub struct SetAliasRequest {
    pub target: String,
}

#[derive(Debug, Serialize)]
pub struct AliasResponse {
    pub alias: String,
    pub target: String,
}

/// Phase G reindex request body. All fields optional; the
/// default (no body, or `{}`) is the straight-copy behaviour.
/// Set `chunker` to re-chunk the source's parent documents
/// with a new strategy; set `re_embed: true` to also re-embed
/// the new chunks via the configured embedding client.
#[derive(Debug, Default, Deserialize)]
pub struct ReindexRequest {
    /// New chunking strategy to apply. `None` = straight copy.
    #[serde(default)]
    pub chunker: Option<veda_core::chunking::ChunkingStrategy>,
    /// Re-run the embedding model against each new chunk's
    /// text. Requires the gateway's embedding client to be
    /// configured (`VEDA_EMBED_ENDPOINT` etc.). When false,
    /// new chunks land with no embedding field set.
    #[serde(default)]
    pub re_embed: bool,
    /// Source field that holds chunk text. Default: "content".
    #[serde(default)]
    pub text_field: Option<String>,
    /// Source field that holds the parent document id (records
    /// sharing a parent get reconstructed + re-chunked
    /// together). Default: "parent_doc_id".
    #[serde(default)]
    pub parent_id_field: Option<String>,
    /// Source field that holds the chunk index within a parent
    /// (used for reconstruction order). Default: "chunk_index".
    #[serde(default)]
    pub chunk_index_field: Option<String>,
    /// Destination field for the re-embedded vector. Default:
    /// the schema's first vector field.
    #[serde(default)]
    pub embedding_field: Option<String>,
    /// New vector index type to apply to every vector field on the
    /// sibling (same accepted values as collection-create's `FieldDef.index`
    /// — "hnsw", "diskann", "ivfpq", "flat", "rabitq", "rabitq_rerank",
    /// "hnsw_rabitq", "cagra", "rabitq_ext_rerank", "sq4", "sq8"). `None` =
    /// keep the source's index type unchanged (the pre-existing behaviour).
    /// NOTE: HNSW build parameters (`ef_construction`/`m`) are NOT
    /// schema-configurable anywhere in Veda today — they're hardcoded in
    /// `veda-index`'s HNSW builder (`crates/veda-index/src/hnsw/mod.rs`).
    /// This field only changes WHICH index algorithm the sibling uses, not
    /// that algorithm's internal build tuning; that is a separate,
    /// pre-existing platform limitation out of this field's scope.
    #[serde(default)]
    pub index_type: Option<String>,
    /// Internal-only override for the sibling's name suffix (e.g.
    /// `@tuning-base1-<uuid>`), bypassing the default `@vN` bump.
    /// `skip_deserializing` — never settable from the public `/reindex`
    /// HTTP body. Used by the tuning campaign's `materialize_bases` to
    /// guarantee each base lands on its own distinct sibling: the default
    /// bump derives purely from the SOURCE name, so calling
    /// `reindex_collection_internal` more than once against the same live
    /// collection (one call per base) would otherwise always compute the
    /// same `@v2` target, silently merging different bases' data into one
    /// sibling.
    #[serde(default, skip_deserializing)]
    pub sibling_suffix: Option<String>,
    /// Internal-only: skip the alias swap in step 6, leaving the live
    /// collection's alias untouched. `skip_deserializing` — never
    /// settable from the public `/reindex` HTTP body (real `/reindex`
    /// calls and `apply_trial`'s genuine applies always want the swap).
    /// Used by `materialize_bases`, which only needs a sibling collection
    /// to search against for trial scoring — it must NOT redirect the
    /// live collection's public alias just because a campaign is
    /// evaluating candidate configs.
    #[serde(default, skip_deserializing)]
    pub skip_alias_swap: bool,
    /// Internal-only: re-screen every copied record for retrieval
    /// guardrails as it is written into the sibling, using the schema's
    /// text-modality `auto_embed` vector field (the record's own already-
    /// stored vector — no re-embed call, no embedder dependency).
    /// `skip_deserializing` — never settable from the public `/reindex`
    /// HTTP body; set only by `rescan_guardrails_impl`. Only takes effect
    /// on the straight-copy path (`chunker: None`) — a guardrail rescan
    /// never re-chunks.
    #[serde(default, skip_deserializing)]
    pub rescreen_guardrails: bool,
}

#[derive(Debug, Serialize)]
pub struct ReindexResponse {
    /// New collection that was created with the same schema.
    pub target: String,
    /// Status — always "alias_swapped (copied N records)" in both Raft
    /// and non-Raft mode (the local AliasMap makes the non-Raft swap
    /// genuinely work too, see Task 0.2's "Phase 0 fix").
    pub status: String,
}

pub async fn get_alias(
    State(state): State<AppState>,
    Tenant(tenant): Tenant,
    Path(name): Path<String>,
) -> Result<Json<AliasResponse>, (StatusCode, Json<ErrorResponse>)> {
    let scoped = tenant::scope(&tenant, &name);
    // No Raft → resolve against the local AliasMap (Phase 0 fix).
    // In Raft mode, also check the replicated SM snapshot. Surface
    // alias = target when no alias is set.
    let raft_guard = state.inner.raft.read().await;
    let target = if let Some(coord) = raft_guard.as_ref() {
        let snap = coord.linearisable_query_state().await.map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: format!("query state: {e}"),
                }),
            )
        })?;
        snap.aliases.get(&scoped).cloned().unwrap_or(scoped.clone())
    } else {
        state.inner.aliases.resolve(&scoped).await
    };
    let target_user = tenant::strip_prefix(&tenant, &target)
        .unwrap_or(&target)
        .to_owned();
    Ok(Json(AliasResponse {
        alias: name,
        target: target_user,
    }))
}

pub async fn set_alias(
    State(state): State<AppState>,
    Tenant(tenant): Tenant,
    Path(name): Path<String>,
    Json(req): Json<SetAliasRequest>,
) -> Result<Json<AliasResponse>, (StatusCode, Json<ErrorResponse>)> {
    let scoped_alias = tenant::scope(&tenant, &name);
    let scoped_target = tenant::scope(&tenant, &req.target);
    {
        let raft_guard = state.inner.raft.read().await;
        if let Some(coord) = raft_guard.as_ref() {
            let cmd = veda_coordinator::raft_types::ClusterCommand::SetCollectionAlias {
                alias: scoped_alias.clone(),
                target: scoped_target.clone(),
            };
            if let Err(e) = coord.execute(cmd).await {
                let lower = format!("{e}").to_lowercase();
                let status = if lower.contains("does not exist") {
                    StatusCode::NOT_FOUND
                } else {
                    StatusCode::INTERNAL_SERVER_ERROR
                };
                return Err((
                    status,
                    Json(ErrorResponse {
                        error: format!("set_alias: {e}"),
                    }),
                ));
            }
        }
        // Non-Raft mode: no Raft SM to write through, but we still
        // persist the alias locally so resolution genuinely works
        // without a coordinator (see Phase 0 of the auto-tuning plan).
        if raft_guard.as_ref().is_none() {
            state
                .inner
                .aliases
                .set(scoped_alias.clone(), scoped_target.clone())
                .await;
        }
    }
    Ok(Json(AliasResponse {
        alias: name,
        target: req.target,
    }))
}

/// Force-seal the current growing segment(s) of every shard in a
/// collection, then rebuild the query executors so the collection's
/// configured vector index type is built on the freshly sealed segments.
///
/// Vector ANN indexes build only on sealed segments. With the default
/// 512 MiB seal threshold a small dataset never seals on its own, so search
/// always brute-forces the growing segment (Flat) and the declared index
/// type (HNSW / IVF-PQ / RaBitQ / …) is unreachable. This endpoint forces
/// the seal so the next search uses the real index. Pair it with a low
/// `VEDA_SEAL_THRESHOLD` for testing, or call it explicitly after a bulk
/// load. Scoped like the other collection routes (not admin-only).
pub async fn seal_collection(
    State(state): State<AppState>,
    Tenant(tenant): Tenant,
    Path(name): Path<String>,
) -> Result<Json<SealResponse>, (StatusCode, Json<ErrorResponse>)> {
    if let Some(leader_url) = state.leader_rest_address().await {
        return Err((
            StatusCode::TEMPORARY_REDIRECT,
            Json(ErrorResponse {
                error: format!("seal must be submitted to the raft leader at {leader_url}"),
            }),
        ));
    }
    let scoped = tenant::scope(&tenant, &name);
    // Resolve the alias — sealing the abandoned base is a no-op that leaves
    // the live collection unsealed, so its declared index types never build.
    let scoped = state.inner.aliases.resolve(&scoped).await;

    // Confirm the collection exists before touching segment managers.
    {
        let schemas = state.inner.schemas.read().await;
        if !schemas.contains_key(&scoped) {
            return Err((
                StatusCode::NOT_FOUND,
                Json(ErrorResponse {
                    error: format!("collection '{name}' not found"),
                }),
            ));
        }
    }

    let mut per_shard: Vec<ShardSealResult> = Vec::new();
    let mut total_sealed = 0usize;
    {
        let mut managers = state.inner.segment_managers.write().await;
        let shard_map = managers.get_mut(&scoped).ok_or_else(|| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: "no segment managers for collection".into(),
                }),
            )
        })?;
        for (shard_id, sm) in shard_map.iter_mut() {
            let sealed = sm.seal_growing().map_err(|e| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse {
                        error: format!("failed to seal segment: {e}"),
                    }),
                )
            })?;
            let n = usize::from(sealed.is_some());
            total_sealed += n;
            per_shard.push(ShardSealResult {
                shard_id: *shard_id,
                sealed_segments: n,
            });
        }
    }

    // Rebuild executors so the configured vector index type is built on the
    // newly sealed segments (Flat fallback is replaced by the real index).
    if total_sealed > 0 {
        crate::rest::records::rebuild_executors(&state, &scoped).await?;
    }

    per_shard.sort_by_key(|s| s.shard_id);
    Ok(Json(SealResponse {
        collection: name,
        sealed_segments: total_sealed,
        per_shard,
    }))
}

/// Force-merge every shard's sealed segments down to a single segment, then
/// rebuild the query executors so one ANN graph per shard serves the whole
/// collection.
///
/// A bulk load fragments a collection into many small sealed segments, and
/// the cross-segment merge loses recall as the count grows. `/optimize`
/// consolidates them (the operator-facing equivalent of a Lucene force-merge):
/// each shard's segments are merged via the compaction path, bypassing the
/// `max_merge_segments` cap, after which search runs a single graph at full
/// recall. Scoped like the other collection routes (not admin-only).
pub async fn optimize_collection(
    State(state): State<AppState>,
    Tenant(tenant): Tenant,
    Path(name): Path<String>,
) -> Result<Json<OptimizeResponse>, (StatusCode, Json<ErrorResponse>)> {
    if let Some(leader_url) = state.leader_rest_address().await {
        return Err((
            StatusCode::TEMPORARY_REDIRECT,
            Json(ErrorResponse {
                error: format!("optimize must be submitted to the raft leader at {leader_url}"),
            }),
        ));
    }
    let scoped = tenant::scope(&tenant, &name);
    // Resolve the alias — see `seal_collection`.
    let scoped = state.inner.aliases.resolve(&scoped).await;

    {
        let schemas = state.inner.schemas.read().await;
        if !schemas.contains_key(&scoped) {
            return Err((
                StatusCode::NOT_FOUND,
                Json(ErrorResponse {
                    error: format!("collection '{name}' not found"),
                }),
            ));
        }
    }

    // Single-flight against the background compactor: take the same
    // per-collection compaction lock so a force-merge never races an
    // insert-triggered compaction (overlapping segment-dir writes would
    // corrupt state).
    let lock = {
        let mut locks = state.inner.compaction_locks.write().await;
        locks
            .entry(scoped.clone())
            .or_insert_with(|| std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)))
            .clone()
    };
    if lock
        .compare_exchange(
            false,
            true,
            std::sync::atomic::Ordering::AcqRel,
            std::sync::atomic::Ordering::Relaxed,
        )
        .is_err()
    {
        return Err((
            StatusCode::CONFLICT,
            Json(ErrorResponse {
                error: "a compaction or optimize is already running for this collection".into(),
            }),
        ));
    }
    struct OptimizeGuard(std::sync::Arc<std::sync::atomic::AtomicBool>);
    impl Drop for OptimizeGuard {
        fn drop(&mut self) {
            self.0.store(false, std::sync::atomic::Ordering::Release);
        }
    }
    let _guard = OptimizeGuard(lock);

    let mut per_shard: Vec<ShardOptimizeResult> = Vec::new();
    {
        let mut managers = state.inner.segment_managers.write().await;
        let shard_map = managers.get_mut(&scoped).ok_or_else(|| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: "no segment managers for collection".into(),
                }),
            )
        })?;
        for (shard_id, sm) in shard_map.iter_mut() {
            let before = sm.force_merge_all().map_err(|e| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse {
                        error: format!("force-merge failed: {e}"),
                    }),
                )
            })?;
            per_shard.push(ShardOptimizeResult {
                shard_id: *shard_id,
                segments_before: before,
                segments_after: sm.sealed_segments().len(),
            });
        }
    }

    // Rebuild executors so the consolidated segment's ANN index replaces the
    // per-segment indexes the searcher held.
    crate::rest::records::rebuild_executors(&state, &scoped).await?;

    per_shard.sort_by_key(|s| s.shard_id);
    Ok(Json(OptimizeResponse {
        collection: name,
        per_shard,
    }))
}

/// Phase G stub: creates `name@v2` (next version if it exists)
/// with the SAME schema as `name`, then submits the alias swap.
/// Does NOT copy records or re-chunk — that's the full
/// reindex pipeline (separate follow-up). Today this proves the
/// atomic-swap primitive end-to-end via REST.
/// `reindex_collection`'s full existing body (unchanged from Phase 0 —
/// schema lookup, sibling creation, copy/rechunk, alias-swap), just with
/// the axum extractors replaced by plain arguments so non-HTTP callers
/// (the campaign engine, Task 5.1's apply) can call it directly.
/// `scoped_name` is already tenant-scoped; `ReindexResponse.target` is
/// still the UNSCOPED display name (via `tenant::display_name`) to match
/// this function's existing HTTP-facing contract/tests.
pub async fn reindex_collection_internal(
    state: &AppState,
    scoped_name: &str,
    req: ReindexRequest,
) -> Result<ReindexResponse, (StatusCode, Json<ErrorResponse>)> {
    let mut scan_counts = GuardrailScanCounts::default();
    // Ordinary reindex callers never pre-resolve an alias (`reindex_collection`
    // below uses `tenant::scope` only), so `scoped_name` is both the alias
    // key (`base_name`) and the data source (`read_from`) — unchanged from
    // before this function grew that distinction for the guardrail rescan
    // path below.
    reindex_collection_internal_ex(state, scoped_name, scoped_name, req, &mut scan_counts).await
}

/// Guardrail-scan counters produced when `ReindexRequest.rescreen_guardrails`
/// is set. Always left at its default (all zero) for an ordinary reindex.
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct GuardrailScanCounts {
    pub scanned: usize,
    pub flagged: usize,
    pub skipped: usize,
}

/// `reindex_collection_internal`'s real body, plus an out-parameter that
/// `rescan_guardrails_impl` reads to get the `scanned`/`flagged`/`skipped`
/// counts out of the copy step without changing `ReindexResponse`'s public
/// shape (which every other `/reindex` caller depends on).
///
/// `base_name` and `read_from` are deliberately separate. `base_name` is
/// the ALIAS KEY — the stable, never-resolved scoped name (e.g. `"X"`) —
/// and is what sibling-name derivation and the final alias swap both use.
/// `read_from` is where the CURRENT live data actually is: for an
/// ordinary reindex the two are identical (see `reindex_collection_internal`
/// above), but a guardrail rescan resolves the alias first, so on a
/// SECOND rescan `base_name` is still `"X"` while `read_from` is `"X@v2"`.
/// Collapsing them into one parameter (an earlier version of this
/// function did) is a real bug: `AliasMap::resolve` is single-hop, so if
/// the alias key were also the resolved name, the second rescan would set
/// `"X@v2" -> "X@v3"` instead of `"X" -> "X@v3"`, leaving `"X"` still
/// pointing at the stale `"X@v2"` and every verdict from the second rescan
/// onward permanently unreachable — while the endpoint keeps returning
/// 200 with full counts. Using `read_from` (not `base_name`) as the data
/// source is equally required: rebuilding from the base name would use
/// whatever was materialized under `"X"` BEFORE the first swap, silently
/// dropping everything ingested since (all writes since the first swap
/// went through the alias to `"X@v2"`, not to `"X"`).
async fn reindex_collection_internal_ex(
    state: &AppState,
    base_name: &str,
    read_from: &str,
    req: ReindexRequest,
    scan_counts: &mut GuardrailScanCounts,
) -> Result<ReindexResponse, (StatusCode, Json<ErrorResponse>)> {
    // 1. Look up the current schema — from `read_from`, the actual
    //    current data, not `base_name`.
    let schema_json = {
        let schemas = state.inner.schemas.read().await;
        match schemas.get(read_from) {
            Some(s) => serde_json::to_string(s).map_err(|e| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse {
                        error: format!("serialize schema: {e}"),
                    }),
                )
            })?,
            None => {
                return Err((
                    StatusCode::NOT_FOUND,
                    Json(ErrorResponse {
                        error: format!(
                            "collection '{}' not found",
                            tenant::display_name(read_from)
                        ),
                    }),
                ));
            }
        }
    };

    // 2. Pick the next sibling version. Today we just append
    //    `@v2`; a v2 of v2 would be `@v3`. Keep it simple. This operates
    //    directly on `base_name` (already tenant-prefixed) — tenant ids
    //    can't contain '@' (see `TenantId::parse`), so bumping the whole
    //    scoped string is equivalent to bumping the unscoped display name
    //    and re-adding the prefix. Using `base_name` here (not
    //    `read_from`) is what keeps repeated rescans/reindexes bumping
    //    "X@v2", "X@v3", "X@v4", … relative to the STABLE base, rather
    //    than the collision guard below trying to bump off of whatever
    //    the alias currently resolves to.
    let mut scoped_new = if let Some(suffix) = req.sibling_suffix.as_deref() {
        format!("{base_name}{suffix}")
    } else if let Some(prefix) = base_name.strip_suffix("@v2") {
        format!("{prefix}@v3")
    } else if base_name.contains("@v") {
        // generic vN bump

        bump_version_suffix(base_name).unwrap_or_else(|| format!("{base_name}@v2"))
    } else {
        format!("{base_name}@v2")
    };
    // Guard against colliding with an already-materialized sibling from a
    // PRIOR /reindex call on this same base — the computation above only
    // looks at `base_name`'s OWN suffix, not at what siblings already
    // exist, so reindexing "rag_test" (no `@v` suffix) twice always
    // computes "rag_test@v2" both times. On collision,
    // `materialize_collection_locally` silently no-ops (it only checks
    // whether the name is already in `state.inner.schemas`), so the
    // schema rewrite (e.g. a new index_type) never applies, and the
    // record-copy step re-inserts every source record into the STALE
    // sibling, duplicating them. Found via a live end-to-end trial-apply
    // on real data: a second reindex onto an existing "@v2" silently kept
    // reporting the old index type and doubled the sibling's record count.
    {
        let schemas = state.inner.schemas.read().await;
        while schemas.contains_key(&scoped_new) {
            scoped_new =
                bump_version_suffix(&scoped_new).unwrap_or_else(|| format!("{scoped_new}@v2"));
        }
    }

    // 3. Materialize the sibling locally with the rewritten
    //    schema. In Raft mode we ALSO submit a CreateCollection
    //    so peers learn about it; in non-Raft mode the local
    //    materialise IS the create. We always materialise so
    //    the record-copy step below has somewhere to put rows.
    let mut rewritten: serde_json::Value = serde_json::from_str(&schema_json).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: format!("rewrite schema: {e}"),
            }),
        )
    })?;
    rewritten["name"] = serde_json::Value::String(scoped_new.clone());
    if let Some(idx_str) = &req.index_type {
        let idx_type = parse_index_type(idx_str).map_err(|e| {
            (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: format!("reindex index_type: {e}"),
                }),
            )
        })?;
        let idx_value = serde_json::to_value(idx_type).unwrap_or(serde_json::Value::Null);
        if let Some(fields) = rewritten["fields"].as_array_mut() {
            for field in fields.iter_mut() {
                let is_vector = field
                    .get("field_type")
                    .and_then(|ft| ft.get("Vector"))
                    .is_some();
                if is_vector {
                    field["index_type"] = idx_value.clone();
                }
            }
        }
    }
    let new_schema_json = serde_json::to_string(&rewritten).unwrap_or(schema_json);
    let new_schema: CollectionSchema = serde_json::from_str(&new_schema_json).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: format!("parse rewritten schema: {e}"),
            }),
        )
    })?;

    // Write the schema marker for the sibling so
    // recover_collections_from_disk picks it up across restart.
    let sibling_dir = state.inner.data_dir.join(&scoped_new);
    std::fs::create_dir_all(&sibling_dir).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: format!("create sibling dir: {e}"),
            }),
        )
    })?;
    let schema_bytes = serde_json::to_vec_pretty(&new_schema).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: format!("serialize sibling schema: {e}"),
            }),
        )
    })?;
    std::fs::write(sibling_dir.join("collection.json"), &schema_bytes).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: format!("persist sibling schema: {e}"),
            }),
        )
    })?;
    if let Err((status, e)) = materialize_collection_locally(state, &new_schema, &sibling_dir).await
    {
        return Err((status, e));
    }

    // 4. Raft-mode CreateCollection so the sibling exists on
    //    every node, then continue with the copy locally.
    {
        let raft_guard = state.inner.raft.read().await;
        if let Some(coord) = raft_guard.as_ref() {
            let create = veda_coordinator::raft_types::ClusterCommand::CreateCollection {
                name: scoped_new.clone(),
                schema_json: new_schema_json,
            };
            if let Err(e) = coord.execute(create).await {
                // The state machine returns "already exists" if
                // a previous attempt got this far; treat that as
                // continue rather than fatal.
                let msg = format!("{e}");
                if !msg.contains("already exists") {
                    return Err((
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(ErrorResponse {
                            error: format!("create sibling: {e}"),
                        }),
                    ));
                }
            }
        }
    }

    // 5. Move records from source into the sibling. Three
    //    modes:
    //    - No chunker config → straight copy (legacy path).
    //    - Chunker config, no re_embed → reconstruct each
    //      parent document by concatenating its source chunks,
    //      run the new chunker, emit new chunk records WITHOUT
    //      embeddings (caller's choice; useful for offline
    //      bulk re-chunk before a separate re-embed pass).
    //    - Chunker config + re_embed → as above, plus embed
    //      each new chunk via the gateway's embedding client.
    let copied = if req.chunker.is_some() {
        rechunk_source_into_sibling(state, read_from, &scoped_new, &req).await?
    } else {
        let (n, counts) =
            copy_source_records_to_sibling(state, read_from, &scoped_new, req.rescreen_guardrails)
                .await?;
        *scan_counts = counts;
        n
    };

    // Seal the sibling's growing segments so the copied/rechunked data is
    // actually on disk before the alias swap below. Without this, a crash
    // right after a "successful" reindex (this function already returned
    // `Ok`, alias already swapped) loses everything: `recover_collections_
    // from_disk` only ever reconstructs SEALED segments, never a growing
    // one (mirrors `records::insert_impl`'s existing "seal growing
    // segments so data is on disk" step after an insert batch).
    {
        let mut managers = state.inner.segment_managers.write().await;
        if let Some(shard_map) = managers.get_mut(&scoped_new) {
            for sm in shard_map.values_mut() {
                sm.seal_growing().map_err(|e| {
                    (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(ErrorResponse {
                            error: format!("failed to seal sibling segment: {e}"),
                        }),
                    )
                })?;
            }
        }
    }

    // `materialize_collection_locally` registered the sibling's query
    // executor from a freshly constructed, empty `SegmentManager` — a
    // distinct in-memory instance from the one just populated above, so it
    // never observed the copied/rechunked records. Without this rebuild,
    // every search against the sibling (directly, or via the alias swap
    // below) returns zero results regardless of index type — found via
    // live E2E testing, where a DiskAnn- AND a Flat-reindexed sibling of a
    // real collection both returned empty results despite holding the
    // correct record count.
    crate::rest::records::rebuild_executors(state, &scoped_new).await?;

    // 6. Alias swap: Raft mode uses the replicated command; non-Raft mode
    //    writes directly to the local AliasMap (Phase 0 fix) so the swap
    //    genuinely redirects traffic in single-node deployments too.
    //    Skipped when `req.skip_alias_swap` — a tuning campaign's
    //    `materialize_bases` only needs the sibling to exist and be
    //    populated for internal trial scoring; it must never redirect the
    //    live collection's public alias as a side effect of merely
    //    evaluating candidate configs (only a genuine `apply_trial` or a
    //    real `/reindex` call should ever do that).
    let target = tenant::display_name(&scoped_new).to_owned();
    if req.skip_alias_swap {
        return Ok(ReindexResponse {
            target,
            status: format!(
                "materialized (copied {copied} records), alias not swapped (internal use only)"
            ),
        });
    }
    {
        let raft_guard = state.inner.raft.read().await;
        if let Some(coord) = raft_guard.as_ref() {
            let swap = veda_coordinator::raft_types::ClusterCommand::SetCollectionAlias {
                alias: base_name.to_owned(),
                target: scoped_new.clone(),
            };
            if let Err(e) = coord.execute(swap).await {
                return Err((
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse {
                        error: format!("swap alias: {e}"),
                    }),
                ));
            }
            return Ok(ReindexResponse {
                target,
                status: format!("alias_swapped (copied {copied} records)"),
            });
        }
    }
    state
        .inner
        .aliases
        .set(base_name.to_owned(), scoped_new.clone())
        .await;
    Ok(ReindexResponse {
        target,
        status: format!("alias_swapped (copied {copied} records)"),
    })
}

/// Thin HTTP wrapper: extract, scope the tenant name, delegate.
pub async fn reindex_collection(
    State(state): State<AppState>,
    Tenant(tenant): Tenant,
    Path(name): Path<String>,
    body: Option<Json<ReindexRequest>>,
) -> Result<Json<ReindexResponse>, (StatusCode, Json<ErrorResponse>)> {
    let scoped = tenant::scope(&tenant, &name);
    let req = body.map(|Json(r)| r).unwrap_or_default();
    reindex_collection_internal(&state, &scoped, req)
        .await
        .map(Json)
}

/// Result of a guardrail re-scan.
#[derive(Debug, Serialize)]
pub struct RescanGuardrailsResponse {
    pub scanned: usize,
    pub flagged: usize,
    pub skipped: usize,
}

/// Re-screen every record in a collection and PERSIST the guardrail
/// verdict onto each record.
///
/// Needed for two cases the ingest path cannot cover: corpora indexed
/// before retrieval guardrails shipped, and applying an updated pattern
/// set. Counting without writing the verdict back would leave the
/// collection exactly as exposed as before while reporting success — an
/// unstamped chunk classifies as `unscreened` and passes through
/// enforcement unscreened.
pub async fn rescan_guardrails(
    State(state): State<AppState>,
    Tenant(tenant): Tenant,
    Path(name): Path<String>,
) -> Result<Json<RescanGuardrailsResponse>, (StatusCode, Json<ErrorResponse>)> {
    let base = tenant::scope(&tenant, &name);
    let resolved = state.inner.aliases.resolve(&base).await;
    {
        let schemas = state.inner.schemas.read().await;
        if !schemas.contains_key(&resolved) {
            return Err((
                StatusCode::NOT_FOUND,
                Json(ErrorResponse {
                    error: format!("collection '{name}' not found"),
                }),
            ));
        }
    }
    rescan_guardrails_impl(&state, &base, &resolved)
        .await
        .map(Json)
}

/// Shared re-scan body, used by both the HTTP handler above and
/// `JobKind::RescanGuardrails`'s scheduler dispatch.
///
/// `base` is the tenant-scoped name BEFORE alias resolution (the alias
/// KEY, e.g. `"X"`); `resolved` is what that alias currently points at
/// (the data source, e.g. `"X"` on the first rescan, `"X@v2"` on the
/// second, and so on). Callers must pass both — collapsing them into one
/// value is exactly the bug a previous version of this function had (see
/// `reindex_collection_internal_ex`'s doc comment): `AliasMap::resolve`
/// is single-hop, so re-keying the ALREADY-RESOLVED name on a second
/// rescan sets `"X@v2" -> "X@v3"` instead of `"X" -> "X@v3"`, leaving
/// `"X"` stuck pointing at the stale `"X@v2"` forever after, with every
/// later rescan's verdicts unreachable despite a 200 response.
///
/// Segments are immutable once sealed, so there is no in-place field
/// update. An earlier version of this function hand-rolled one anyway
/// (soft-delete the live doc, re-insert a stamped clone under the same
/// id, directly against the live collection) and that shape had real
/// bugs: a non-deterministic text-field choice that could re-screen the
/// record's own `_guardrail_category` string on a second rescan and
/// silently un-flag it; `let _ = …` on every persistence call, so a
/// partial write reported success while leaving stale and stamped copies
/// both live; shard re-assignment against records already correctly
/// placed, which could permanently duplicate a record a rerun couldn't
/// heal; and a crash window between sealing and persisting bitmaps.
///
/// Instead this reuses `/reindex`'s existing, tested, crash-safe
/// mechanism: rebuild into a sibling collection and atomically swap the
/// alias onto it (`reindex_collection_internal_ex` with
/// `ReindexRequest.rescreen_guardrails: true`, which makes
/// `copy_source_records_to_sibling` screen and stamp each record as it
/// writes it into the sibling — see that function for how it picks a
/// deterministic text field, off-lock screening, and per-record error
/// isolation). The live collection is never mutated in place; a failure
/// anywhere in the rebuild propagates as an error and the alias is never
/// swapped, so a failed rescan can't report success.
pub(crate) async fn rescan_guardrails_impl(
    state: &AppState,
    base: &str,
    resolved: &str,
) -> Result<RescanGuardrailsResponse, (StatusCode, Json<ErrorResponse>)> {
    let req = ReindexRequest {
        rescreen_guardrails: true,
        ..Default::default()
    };
    let mut scan_counts = GuardrailScanCounts::default();
    reindex_collection_internal_ex(state, base, resolved, req, &mut scan_counts).await?;

    Ok(RescanGuardrailsResponse {
        scanned: scan_counts.scanned,
        flagged: scan_counts.flagged,
        skipped: scan_counts.skipped,
    })
}

/// Phase G reindex pipeline helper. Iterates the source
/// collection's segment managers, pulls every non-tombstoned
/// record out of every shard's growing + sealed segments, and
/// inserts each into the corresponding sibling segment manager.
/// Today it's a straight copy; the chunker / embedder
/// transforms slot in here when the real pipeline integration
/// lands. Returns the count of records copied.
async fn copy_source_records_to_sibling(
    state: &AppState,
    src: &str,
    dst: &str,
    rescreen_guardrails: bool,
) -> Result<(usize, GuardrailScanCounts), (StatusCode, Json<ErrorResponse>)> {
    // Pull records from source under a read lock, then release
    // before grabbing the write lock for the destination.
    let records: Vec<veda_core::record::Record> = {
        let managers = state.inner.segment_managers.read().await;
        let Some(src_shards) = managers.get(src) else {
            return Ok((0, GuardrailScanCounts::default())); // source has nothing materialised
        };
        let mut out = Vec::new();
        for sm in src_shards.values() {
            let gs = sm.growing_segment();
            for doc_id in 0..gs.count() {
                if gs.is_deleted(doc_id) {
                    continue;
                }
                if let Some(rec) = gs.get_record(doc_id) {
                    out.push(rec.clone());
                }
            }
            for handle in sm.sealed_segments() {
                for doc_id in 0..handle.metadata.row_count as u32 {
                    if handle.reader.delete_bitmap().is_deleted(doc_id) {
                        continue;
                    }
                    if let Ok(rec) = handle.reader.get_record(doc_id) {
                        out.push(rec);
                    }
                }
            }
        }
        out
    };

    let dst_schema = {
        let schemas = state.inner.schemas.read().await;
        schemas.get(dst).cloned()
    };
    let Some(dst_schema) = dst_schema else {
        return Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: format!("sibling '{dst}' not materialised"),
            }),
        ));
    };

    // When re-screening, identify the one text-modality `auto_embed`
    // vector field the schema declares (mirrors `ingest_common::
    // ingest_markdown`'s `primary_text_vector_field` selection). This is
    // a DETERMINISTIC, schema-driven choice — not "any string field on
    // the record" (a prior version of this code picked the first string
    // field off a `HashMap`, which on a SECOND rescan could pick up the
    // record's own reserved `_guardrail_category` field, e.g. the literal
    // string "pii", screen THAT, come back clean, and overwrite a
    // genuinely flagged record to `_guardrail_flagged = false`).
    //
    // When found, screening reuses the record's own ALREADY-STORED vector
    // for that field rather than re-embedding: no embedder needed at
    // rescan time, and it's the vector actually indexed. This is NOT
    // drift-proof, though: `screen_chunk` embeds the guardrail PATTERN
    // set with whatever text-modality model is CURRENTLY registered. If
    // the record's stored vector was produced by a different (e.g. older)
    // model, comparing it against the current patterns compares two
    // different embedding spaces — `categorize::cosine` returns `0.0` on
    // a dimension mismatch, so the semantic layer simply never fires
    // rather than firing on a meaningless score; the lexical/pii layer
    // (regex-based, not embedding-based) is unaffected either way.
    let screen_target: Option<(String, String)> = if rescreen_guardrails {
        dst_schema.fields().iter().find_map(|f| {
            if let veda_core::types::FieldType::Vector {
                auto_embed: Some(cfg),
                ..
            } = f.field_type()
                && matches!(cfg.modality, veda_core::types::AutoEmbedModality::Text)
            {
                return Some((f.name().to_owned(), cfg.source.clone()));
            }
            None
        })
    } else {
        None
    };

    // Collections whose vectors are CLIENT-supplied (no schema `auto_embed`
    // at all) are exactly the pre-guardrails corpora this endpoint exists
    // to remediate, and `screen_target` above is `None` for every one of
    // them — without a fallback, every record in such a collection would
    // report `skipped` with no way to actually screen it. Fall back to
    // embedding the record's text with whatever text-modality model is
    // currently registered, same as `screen_chunk`'s own pattern-embedding
    // path does. The fallback text field is still chosen deterministically
    // from the SCHEMA (the first `FieldType::String` field, in schema
    // declaration order) rather than the record's own field map — the
    // reserved guardrail fields are never declared on a `CollectionSchema`,
    // so this can't reintroduce the same "picks up `_guardrail_category`"
    // failure the primary path avoids.
    let fallback_text_field: Option<String> = if rescreen_guardrails && screen_target.is_none() {
        dst_schema.fields().iter().find_map(|f| {
            matches!(f.field_type(), veda_core::types::FieldType::String)
                .then(|| f.name().to_owned())
        })
    } else {
        None
    };
    let fallback_embedder = if fallback_text_field.is_some() {
        crate::rest::rag_eval::text_embedder_pub(state).await
    } else {
        None
    };

    // Screen off any lock — `screen_chunk` can do a real embed call (to
    // embed the guardrail pattern set on a cache miss), and the fallback
    // path above embeds the record's own text too; neither may run while
    // holding `segment_managers`.
    let mut scan_counts = GuardrailScanCounts::default();
    let mut staged: Vec<veda_core::record::Record> = Vec::with_capacity(records.len());
    for mut record in records {
        if rescreen_guardrails {
            scan_counts.scanned += 1;
            let verdict = match &screen_target {
                Some((vec_field, text_field)) => {
                    let text = match record.get(text_field) {
                        Some(veda_core::record::FieldValue::String(s)) => Some(s.clone()),
                        _ => None,
                    };
                    let vector = match record.get(vec_field) {
                        Some(veda_core::record::FieldValue::Vector(v)) => {
                            Some(v.as_slice().to_vec())
                        }
                        _ => None,
                    };
                    match (text, vector) {
                        (Some(t), Some(v)) => {
                            crate::agent::guardrails::screen_chunk(state, &v, &t).await
                        }
                        _ => crate::agent::guardrails::ChunkVerdict::skipped(),
                    }
                }
                None => match (&fallback_text_field, &fallback_embedder) {
                    (Some(text_field), Some(client)) => {
                        let text = match record.get(text_field) {
                            Some(veda_core::record::FieldValue::String(s))
                                if !s.trim().is_empty() =>
                            {
                                Some(s.clone())
                            }
                            _ => None,
                        };
                        match text {
                            Some(t) => match client.embed_text(&t).await {
                                Ok(v) => {
                                    crate::agent::guardrails::screen_chunk(state, &v, &t).await
                                }
                                Err(_) => crate::agent::guardrails::ChunkVerdict::skipped(),
                            },
                            None => crate::agent::guardrails::ChunkVerdict::skipped(),
                        }
                    }
                    _ => crate::agent::guardrails::ChunkVerdict::skipped(),
                },
            };
            if !verdict.screened {
                scan_counts.skipped += 1;
            } else if verdict.flagged {
                scan_counts.flagged += 1;
            }
            crate::rest::ingest_common::stamp_guardrail_fields(
                &mut record,
                &verdict,
                crate::rest::ingest_common::now_secs(),
            );
        }
        staged.push(record);
    }

    let shard_count = dst_schema.shard_count();
    let mut managers = state.inner.segment_managers.write().await;
    let Some(dst_shards) = managers.get_mut(dst) else {
        return Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: format!("sibling '{dst}' segment managers missing"),
            }),
        ));
    };
    let mut copied = 0usize;
    for record in staged {
        let shard_id =
            veda_coordinator::shard_router::ShardRouter::assign_shard(&record.id(), shard_count);
        let Some(sm) = dst_shards.get_mut(&shard_id) else {
            continue;
        };
        // Best-effort — duplicates between retries collide; ignore. This
        // is pre-existing `/reindex` behaviour, unchanged by rescreening:
        // `scan_counts` above already reflects what was screened
        // regardless of whether an individual record's insert into the
        // sibling later fails.
        if sm.insert(record).is_ok() {
            copied += 1;
        }
    }
    Ok((copied, scan_counts))
}

/// Phase G real pipeline: re-chunk source records via the
/// supplied chunker strategy, optionally re-embed, then write
/// the new chunks to the sibling. Records are grouped by their
/// `parent_id_field` value (default "parent_doc_id"); each
/// group is concatenated in chunk-index order to reconstruct
/// the parent text, then the new chunker runs over that text
/// and emits the new chunks. Records WITHOUT a parent_doc_id
/// (root documents themselves, or single-chunk insertions)
/// are copied through unchanged. Returns the count of records
/// inserted into the sibling.
async fn rechunk_source_into_sibling(
    state: &AppState,
    src: &str,
    dst: &str,
    req: &ReindexRequest,
) -> Result<usize, (StatusCode, Json<ErrorResponse>)> {
    use std::collections::BTreeMap;
    use veda_core::record::FieldValue;

    let strategy = req.chunker.clone().ok_or_else(|| {
        (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "rechunk requires chunker".into(),
            }),
        )
    })?;
    let text_field = req.text_field.as_deref().unwrap_or("content");
    let parent_field = req.parent_id_field.as_deref().unwrap_or("parent_doc_id");
    let chunk_idx_field = req.chunk_index_field.as_deref().unwrap_or("chunk_index");

    // 1. Pull every source record under a read lock; group by
    //    parent_id (None bucket = orphans / copy-as-is).
    let (parented_groups, orphans): (
        BTreeMap<String, Vec<veda_core::record::Record>>,
        Vec<veda_core::record::Record>,
    ) = {
        let managers = state.inner.segment_managers.read().await;
        let Some(src_shards) = managers.get(src) else {
            return Ok(0);
        };
        let mut groups: BTreeMap<String, Vec<veda_core::record::Record>> = BTreeMap::new();
        let mut orphans: Vec<veda_core::record::Record> = Vec::new();
        let take_record = |out_groups: &mut BTreeMap<String, Vec<veda_core::record::Record>>,
                           out_orphans: &mut Vec<veda_core::record::Record>,
                           rec: veda_core::record::Record| {
            let parent_id = rec.get(parent_field).and_then(|v| match v {
                FieldValue::String(s) => Some(s.clone()),
                _ => None,
            });
            match parent_id {
                Some(pid) => out_groups.entry(pid).or_default().push(rec),
                None => out_orphans.push(rec),
            }
        };
        for sm in src_shards.values() {
            let gs = sm.growing_segment();
            for doc_id in 0..gs.count() {
                if gs.is_deleted(doc_id) {
                    continue;
                }
                if let Some(rec) = gs.get_record(doc_id) {
                    take_record(&mut groups, &mut orphans, rec.clone());
                }
            }
            for handle in sm.sealed_segments() {
                for doc_id in 0..handle.metadata.row_count as u32 {
                    if handle.reader.delete_bitmap().is_deleted(doc_id) {
                        continue;
                    }
                    if let Ok(rec) = handle.reader.get_record(doc_id) {
                        take_record(&mut groups, &mut orphans, rec);
                    }
                }
            }
        }
        (groups, orphans)
    };

    // 2. Look up the sibling's schema to find the embedding
    //    field (defaults to the first vector field).
    let dst_schema = {
        let schemas = state.inner.schemas.read().await;
        schemas.get(dst).cloned()
    };
    let Some(dst_schema) = dst_schema else {
        return Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: format!("sibling '{dst}' not materialised"),
            }),
        ));
    };
    let embedding_field: Option<String> = req.embedding_field.clone().or_else(|| {
        dst_schema
            .vector_fields()
            .first()
            .map(|f| f.name().to_owned())
    });
    let embedding_dim: Option<u32> = embedding_field.as_ref().and_then(|name| {
        dst_schema.vector_fields().iter().find_map(|f| {
            if f.name() == name {
                match f.field_type() {
                    veda_core::types::FieldType::Vector { dimension, .. } => Some(*dimension),
                    _ => None,
                }
            } else {
                None
            }
        })
    });
    let embed_client_opt: Option<veda_core::embedding_client::EmbeddingClient> = if req.re_embed {
        let registry = state.inner.embedding_registry.read().await;
        // Pick a text-modality model (re-chunk inputs are text).
        registry
            .find_by_modality(veda_core::embedding::Modality::Text)
            .first()
            .map(|cfg| veda_core::embedding_client::EmbeddingClient::new((*cfg).clone()))
    } else {
        None
    };
    if req.re_embed && embed_client_opt.is_none() {
        return Err((
            StatusCode::FAILED_DEPENDENCY,
            Json(ErrorResponse {
                error: "re_embed=true requires a text-modality model registered in the embedding registry".into(),
            }),
        ));
    }

    let shard_count = dst_schema.shard_count();

    // 3. For each parent group: reconstruct text, re-chunk,
    //    build new Records, insert. Orphans get copied through.
    let mut new_records: Vec<veda_core::record::Record> = Vec::new();
    for (parent_id, mut chunks) in parented_groups {
        // Sort by chunk_index (numeric) — None falls to the end.
        chunks.sort_by_key(|r| match r.get(chunk_idx_field) {
            Some(FieldValue::Int32(i)) => *i as i64,
            Some(FieldValue::Int64(i)) => *i,
            _ => i64::MAX,
        });
        let reconstructed: String = chunks
            .iter()
            .filter_map(|r| match r.get(text_field) {
                Some(FieldValue::String(s)) => Some(s.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n");
        let new_chunks = veda_core::chunking::chunk_markdown(&reconstructed, &strategy);
        // Inherit one of the source chunks' non-text scalar
        // fields as a template (parent_doc_id, page_number, etc.
        // stay sensible). We use the FIRST chunk in the sorted
        // group as the template.
        let template = chunks.first();
        for (i, tc) in new_chunks.into_iter().enumerate() {
            let mut rec = veda_core::record::Record::new();
            // Carry forward scalar / non-vector / non-text
            // fields from the template so things like
            // collection_name, source_uri, language survive
            // the re-chunk.
            if let Some(tpl) = template {
                for f in dst_schema.scalar_fields() {
                    if f.name() == text_field || f.name() == chunk_idx_field {
                        continue;
                    }
                    if let Some(v) = tpl.get(f.name()) {
                        rec.set(f.name(), v.clone());
                    }
                }
            }
            rec.set(text_field, FieldValue::String(tc.text.clone()));
            rec.set(parent_field, FieldValue::String(parent_id.clone()));
            rec.set(chunk_idx_field, FieldValue::Int64(i as i64));
            if let Some(ef) = embedding_field.as_ref()
                && let (Some(client), Some(dim)) = (embed_client_opt.as_ref(), embedding_dim)
            {
                match client.embed_text(&tc.text).await {
                    Ok(v) if v.len() == dim as usize => {
                        rec.set(
                            ef,
                            FieldValue::Vector(veda_core::vector::DenseVector::new(v)),
                        );
                    }
                    Ok(v) => {
                        return Err((
                            StatusCode::FAILED_DEPENDENCY,
                            Json(ErrorResponse {
                                error: format!("embed dim mismatch: schema={dim} got={}", v.len()),
                            }),
                        ));
                    }
                    Err(e) => {
                        return Err((
                            StatusCode::FAILED_DEPENDENCY,
                            Json(ErrorResponse {
                                error: format!("embed_text failed: {e}"),
                            }),
                        ));
                    }
                }
            } else if let Some(ef) = embedding_field.as_ref()
                && let Some(dim) = embedding_dim
            {
                // No re_embed: zero-vector placeholder so the
                // schema's required vector field is present.
                rec.set(
                    ef,
                    FieldValue::Vector(veda_core::vector::DenseVector::new(vec![
                        0.0;
                        dim as usize
                    ])),
                );
            }
            new_records.push(rec);
        }
    }
    new_records.extend(orphans);

    // 4. Insert into sibling segment managers.
    let mut managers = state.inner.segment_managers.write().await;
    let Some(dst_shards) = managers.get_mut(dst) else {
        return Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: format!("sibling '{dst}' segment managers missing"),
            }),
        ));
    };
    let mut inserted = 0usize;
    for rec in new_records {
        let shard_id =
            veda_coordinator::shard_router::ShardRouter::assign_shard(&rec.id(), shard_count);
        if let Some(sm) = dst_shards.get_mut(&shard_id)
            && sm.insert(rec).is_ok()
        {
            inserted += 1;
        }
    }
    Ok(inserted)
}

fn bump_version_suffix(name: &str) -> Option<String> {
    let at = name.rfind("@v")?;
    let suffix = &name[at + 2..];
    let n: u32 = suffix.parse().ok()?;
    Some(format!("{}@v{}", &name[..at], n + 1))
}

async fn drop_collection_impl(
    state: AppState,
    tenant: tenant::TenantId,
    name: String,
) -> Result<Json<DropCollectionResponse>, (StatusCode, Json<ErrorResponse>)> {
    // Raft mode: submit the drop to the consensus log first.
    // After the entry commits + applies on every node, the local
    // reconcile (here or in the periodic task) tears down
    // segment managers + executors.
    let scoped = tenant::scope(&tenant, &name);
    {
        let raft_guard = state.inner.raft.read().await;
        if let Some(coord) = raft_guard.as_ref() {
            let cmd = veda_coordinator::raft_types::ClusterCommand::DropCollection {
                name: scoped.clone(),
            };
            if let Err(e) = coord.execute(cmd).await {
                return Err((
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse {
                        error: format!("raft submit: {e}"),
                    }),
                ));
            }
        }
    }

    let removed = {
        let mut schemas = state.inner.schemas.write().await;
        schemas.remove(&scoped)
    };

    if removed.is_none() {
        return Err((
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: format!("collection '{name}' not found"),
            }),
        ));
    }

    // Stop this collection's applier tasks before their SegmentManager
    // and shard logs are removed, so they don't tick into missing state.
    state.abort_appliers(&scoped);

    {
        let mut managers = state.inner.segment_managers.write().await;
        managers.remove(&scoped);
    }
    {
        let mut executors = state.inner.query_executors.write().await;
        executors.remove(&scoped);
    }
    state.inner.shard_logs.remove_collection(&scoped);
    if let Err(e) = state.inner.shard_logs.purge_collection_dir(&scoped) {
        tracing::warn!(collection = %scoped, error = %e, "failed to purge shard-log dir on drop");
    }

    let collection_dir = state.inner.data_dir.join(&scoped);
    if collection_dir.exists() {
        let _ = std::fs::remove_dir_all(&collection_dir);
    }

    let collection_count = {
        let schemas = state.inner.schemas.read().await;
        schemas.len()
    };
    metrics::gauge!(m::ACTIVE_COLLECTIONS).set(collection_count as f64);

    Ok(Json(DropCollectionResponse {
        name,
        status: "dropped".into(),
    }))
}

// ── Helpers ──────────────────────────────────────────────────────────────

/// Parse a vector index-type string (accepted at collection-create time via
/// `FieldDef.index`, and on `/reindex` via `ReindexRequest.index_type`) into
/// the real `veda_core::types::IndexType`. Factored out of `parse_field_def`
/// so both call sites share one alias list and one CAGRA GPU-availability
/// check rather than risking two copies drifting apart.
fn parse_index_type(idx: &str) -> Result<veda_core::types::IndexType, String> {
    Ok(match idx.to_lowercase().as_str() {
        "hnsw" => veda_core::types::IndexType::Hnsw,
        "diskann" => veda_core::types::IndexType::DiskAnn,
        "ivfpq" | "ivf_pq" => veda_core::types::IndexType::IvfPq,
        "flat" => veda_core::types::IndexType::Flat,
        "rabitq" | "rabit_q" => veda_core::types::IndexType::RaBitQ,
        "rabitq_rerank" | "rabit_q_rerank" => veda_core::types::IndexType::RaBitQRerank,
        "hnsw_rabitq" | "hnsw_rabit_q" | "quantized_hnsw" => {
            veda_core::types::IndexType::HnswRaBitQ
        }
        "cagra" | "gpu_cagra" | "gpu" => {
            // CAGRA needs the GPU cuVS backend. Without the `gpu_cuvs`
            // feature the index build fails after insert, leaving an
            // unsearchable collection. Reject up-front with a clear error.
            if !veda_index::gpu_cuvs::gpu_cuvs_available() {
                return Err(
                    "index type 'cagra' requires a GPU build (compile with --features veda-index/gpu_cuvs); this binary has no cuVS backend"
                        .to_owned(),
                );
            }
            veda_core::types::IndexType::Cagra
        }
        "rabitq_ext_rerank" | "ext_rabitq_rerank" | "rabitq4_rerank" => {
            veda_core::types::IndexType::RaBitQExtRerank
        }
        "sq4" => veda_core::types::IndexType::Sq4,
        "sq8" => veda_core::types::IndexType::Sq8,
        other => return Err(format!("unknown vector index type '{other}'")),
    })
}

fn parse_field_def(f: &FieldDef) -> Result<FieldDefinition, String> {
    match f.field_type.as_str() {
        "vector" => {
            let dim = f
                .dimension
                .ok_or_else(|| format!("field '{}': vector type requires 'dimension'", f.name))?;
            let metric = match f.metric.as_deref().unwrap_or("cosine") {
                "cosine" => DistanceMetric::Cosine,
                "euclidean" => DistanceMetric::Euclidean,
                "dot_product" | "dotproduct" => DistanceMetric::DotProduct,
                "manhattan" => DistanceMetric::Manhattan,
                other => return Err(format!("field '{}': unknown metric '{other}'", f.name)),
            };
            let mut fd = match &f.model {
                Some(m) if !m.is_empty() => {
                    FieldDefinition::vector_with_model(&f.name, dim, metric, m)
                }
                _ => FieldDefinition::vector(&f.name, dim, metric),
            };
            if let Some(dims) = &f.mrl_dims {
                fd = fd.with_mrl_dims(dims.clone());
            }
            if let Some(cfg) = &f.auto_embed {
                fd = fd.with_auto_embed(cfg.clone());
            }
            if let Some(idx) = f.index.as_deref() {
                let idx_type =
                    parse_index_type(idx).map_err(|e| format!("field '{}': {e}", f.name))?;
                fd = fd.with_index_type(idx_type);
            }
            Ok(fd.with_filterable(f.filterable))
        }
        "multi_vector" | "multivector" => {
            let dim = f.dimension.ok_or_else(|| {
                format!("field '{}': multi_vector type requires 'dimension'", f.name)
            })?;
            let metric = match f.metric.as_deref().unwrap_or("cosine") {
                "cosine" => DistanceMetric::Cosine,
                "euclidean" => DistanceMetric::Euclidean,
                "dot_product" | "dotproduct" => DistanceMetric::DotProduct,
                other => {
                    return Err(format!(
                        "field '{}': multi_vector unsupported metric '{other}'",
                        f.name
                    ));
                }
            };
            // max_vectors is optional; 0 = unbounded.
            let max_vectors = f.max_vectors.unwrap_or(0);
            Ok(
                FieldDefinition::multi_vector(&f.name, dim, max_vectors, metric)
                    .with_filterable(f.filterable),
            )
        }
        "sparse_vector" | "sparsevector" => {
            let dim = f.dimension.ok_or_else(|| {
                format!(
                    "field '{}': sparse_vector type requires 'dimension'",
                    f.name
                )
            })?;
            let metric = match f.metric.as_deref().unwrap_or("cosine") {
                "cosine" => DistanceMetric::Cosine,
                "dot_product" | "dotproduct" => DistanceMetric::DotProduct,
                other => {
                    return Err(format!(
                        "field '{}': sparse_vector supports only 'cosine' or 'dot_product', got '{other}'",
                        f.name
                    ));
                }
            };
            let mut fd = FieldDefinition::sparse_vector(&f.name, dim, metric);
            if let Some(cfg) = &f.auto_embed {
                fd = fd.with_sparse_auto_embed(cfg.clone());
            }
            Ok(fd.with_filterable(f.filterable))
        }
        "blob" => Ok(FieldDefinition::blob(&f.name).with_filterable(f.filterable)),
        "string" => Ok(FieldDefinition::new(&f.name, FieldType::String)
            .with_filterable(f.filterable)
            .with_bm25(f.bm25)
            .with_bm25_tokenizer(f.bm25_tokenizer.clone())),
        "int32" => {
            Ok(FieldDefinition::new(&f.name, FieldType::Int32).with_filterable(f.filterable))
        }
        "int64" => {
            Ok(FieldDefinition::new(&f.name, FieldType::Int64).with_filterable(f.filterable))
        }
        "float32" => {
            Ok(FieldDefinition::new(&f.name, FieldType::Float32).with_filterable(f.filterable))
        }
        "float64" => {
            Ok(FieldDefinition::new(&f.name, FieldType::Float64).with_filterable(f.filterable))
        }
        "bool" => Ok(FieldDefinition::new(&f.name, FieldType::Bool).with_filterable(f.filterable)),
        "timestamp" => {
            Ok(FieldDefinition::new(&f.name, FieldType::Timestamp).with_filterable(f.filterable))
        }
        "json" => Ok(FieldDefinition::new(&f.name, FieldType::Json).with_filterable(f.filterable)),
        "uuid" => Ok(FieldDefinition::new(&f.name, FieldType::Uuid).with_filterable(f.filterable)),
        "array_string" | "stringarray" | "string_array" => Ok(FieldDefinition::new(
            &f.name,
            FieldType::Array(Box::new(FieldType::String)),
        )
        .with_filterable(f.filterable)),
        "array_int32" | "int32_array" => Ok(FieldDefinition::new(
            &f.name,
            FieldType::Array(Box::new(FieldType::Int32)),
        )
        .with_filterable(f.filterable)),
        "array_int64" | "int64_array" => Ok(FieldDefinition::new(
            &f.name,
            FieldType::Array(Box::new(FieldType::Int64)),
        )
        .with_filterable(f.filterable)),
        other => Err(format!("field '{}': unknown type '{other}'", f.name)),
    }
}

fn field_to_info(f: &FieldDefinition) -> FieldInfo {
    let (type_str, dimension, metric, model) = match f.field_type() {
        FieldType::Vector {
            dimension,
            metric,
            model_id,
            ..
        } => {
            let m = match metric {
                DistanceMetric::Cosine => "cosine",
                DistanceMetric::Euclidean => "euclidean",
                DistanceMetric::DotProduct => "dot_product",
                DistanceMetric::Manhattan => "manhattan",
                DistanceMetric::Hamming => "hamming",
                DistanceMetric::Jaccard => "jaccard",
            };
            (
                "vector",
                Some(*dimension),
                Some(m.to_owned()),
                model_id.clone(),
            )
        }
        FieldType::Blob => ("blob", None, None, None),
        FieldType::String => ("string", None, None, None),
        FieldType::Int32 => ("int32", None, None, None),
        FieldType::Int64 => ("int64", None, None, None),
        FieldType::Float32 => ("float32", None, None, None),
        FieldType::Float64 => ("float64", None, None, None),
        FieldType::Bool => ("bool", None, None, None),
        FieldType::Timestamp => ("timestamp", None, None, None),
        FieldType::Json => ("json", None, None, None),
        FieldType::Uuid => ("uuid", None, None, None),
        FieldType::SparseVector {
            dimension, metric, ..
        } => {
            let m = match metric {
                DistanceMetric::Cosine => "cosine",
                DistanceMetric::DotProduct => "dot_product",
                _ => "cosine",
            };
            ("sparse_vector", Some(*dimension), Some(m.to_owned()), None)
        }
        FieldType::MultiVector {
            dimension, metric, ..
        } => {
            let m = match metric {
                DistanceMetric::Cosine => "cosine",
                DistanceMetric::Euclidean => "euclidean",
                DistanceMetric::DotProduct => "dot_product",
                _ => "cosine",
            };
            ("multi_vector", Some(*dimension), Some(m.to_owned()), None)
        }
        FieldType::Array(_) => ("array", None, None, None),
    };

    let index_type = matches!(type_str, "vector" | "sparse_vector" | "multi_vector")
        .then(|| format!("{:?}", f.index_type()));

    FieldInfo {
        name: f.name().to_owned(),
        field_type: type_str.to_owned(),
        filterable: f.filterable(),
        dimension,
        metric,
        model,
        index_type,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seal_threshold_parses_positive_override() {
        // 4 MiB override is honored.
        assert_eq!(parse_seal_threshold(Some("4194304".to_owned())), 4_194_304);
        // Surrounding whitespace is tolerated.
        assert_eq!(parse_seal_threshold(Some("  1024 ".to_owned())), 1024);
    }

    #[test]
    fn seal_threshold_falls_back_to_default() {
        // Unset, blank, non-numeric, and zero all fall back to the default.
        assert_eq!(parse_seal_threshold(None), DEFAULT_SEAL_THRESHOLD_BYTES);
        assert_eq!(
            parse_seal_threshold(Some("".to_owned())),
            DEFAULT_SEAL_THRESHOLD_BYTES
        );
        assert_eq!(
            parse_seal_threshold(Some("not-a-number".to_owned())),
            DEFAULT_SEAL_THRESHOLD_BYTES
        );
        assert_eq!(
            parse_seal_threshold(Some("0".to_owned())),
            DEFAULT_SEAL_THRESHOLD_BYTES
        );
    }

    fn test_state() -> AppState {
        let tmp = tempfile::tempdir().unwrap();
        AppState::new(tmp.path())
    }

    #[tokio::test]
    async fn set_alias_then_get_alias_resolves_in_non_raft_mode() {
        let state = test_state();
        let tenant = crate::tenant::TenantId::default_tenant();

        // No Raft configured on `state` -> this exercises the non-Raft path.
        let resp = set_alias(
            State(state.clone()),
            Tenant(tenant.clone()),
            Path("mycol".to_owned()),
            Json(SetAliasRequest {
                target: "mycol@v2".to_owned(),
            }),
        )
        .await
        .expect("set_alias should succeed without Raft");
        assert_eq!(resp.target, "mycol@v2");

        let got = get_alias(
            State(state.clone()),
            Tenant(tenant),
            Path("mycol".to_owned()),
        )
        .await
        .expect("get_alias should succeed");
        assert_eq!(
            got.target, "mycol@v2",
            "non-Raft get_alias must now reflect the swap, not just echo back a no-op"
        );
    }

    async fn seed_test_collection(state: &AppState, name: &str) {
        let scoped = crate::tenant::scope(&crate::tenant::TenantId::default_tenant(), name);
        let schema = CollectionSchema::builder(scoped.clone())
            .add_field(FieldDefinition::new("id", FieldType::String))
            .shard_count(1)
            .replication_factor(1)
            .build()
            .expect("test schema should build");
        {
            let mut schemas = state.inner.schemas.write().await;
            schemas.insert(scoped.clone(), schema);
        }
    }

    #[tokio::test]
    async fn reindex_then_get_alias_shows_the_swap_without_raft() {
        let state = test_state();
        let tenant = crate::tenant::TenantId::default_tenant();
        // Seed a minimal collection so reindex_collection has a schema to copy.
        seed_test_collection(&state, "mycol").await;

        let resp = reindex_collection(
            State(state.clone()),
            Tenant(tenant.clone()),
            Path("mycol".to_owned()),
            None,
        )
        .await
        .expect("reindex should succeed");
        assert!(resp.status.starts_with("alias_swapped"));

        let alias = get_alias(
            State(state.clone()),
            Tenant(tenant),
            Path("mycol".to_owned()),
        )
        .await
        .expect("get_alias should succeed");
        assert_eq!(
            alias.target, resp.target,
            "alias must now point at the new sibling"
        );
    }
}

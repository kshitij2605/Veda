use std::collections::HashMap;

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::Json;
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
    use axum::http::{header, HeaderValue};
    use axum::response::IntoResponse;
    let url = format!(
        "{}{}",
        leader_base.trim_end_matches('/'),
        path
    );
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
        .use_mvcc(req.use_mvcc);
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
        let field_def = parse_field_def(f).map_err(|e| {
            (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse { error: e }),
            )
        })?;
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
                let _ = drop_collection_impl(state.clone(), tenant.clone(), user_name.to_owned()).await;
                return Err((
                    seed_status,
                    Json(ErrorResponse {
                        error: format!(
                            "create-and-seed rolled back: {}",
                            seed_err.0.error
                        ),
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
        let mut sm =
            SegmentManager::new(schema.clone(), shard_id, collection_dir, 512 * 1024 * 1024)
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
            SegmentManager::new(schema.clone(), shard_id, collection_dir, 512 * 1024 * 1024)
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
pub async fn reconcile_collections_from_raft(
    state: &AppState,
) -> Result<(), String> {
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
        let _ = state.inner.raft_materialized_records.write().await.remove(name);
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
    let schemas = state.inner.schemas.read().await;
    let schema = schemas.get(&scoped).ok_or_else(|| {
        (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: format!("collection '{name}' not found"),
            }),
        )
    })?;

    let fields = schema
        .fields()
        .iter()
        .map(field_to_info)
        .collect();

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
}

#[derive(Debug, Serialize)]
pub struct ReindexResponse {
    /// New collection that was created with the same schema.
    pub target: String,
    /// Status — "alias_swapped" or "no_raft" depending on mode.
    pub status: String,
}

pub async fn get_alias(
    State(state): State<AppState>,
    Tenant(tenant): Tenant,
    Path(name): Path<String>,
) -> Result<Json<AliasResponse>, (StatusCode, Json<ErrorResponse>)> {
    let scoped = tenant::scope(&tenant, &name);
    // No Raft → just resolve against the in-memory schemas
    // (the alias map round-trips through state-machine snapshot,
    // but in non-Raft mode there's no SM to consult). Surface
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
        scoped.clone()
    };
    let target_user = tenant::strip_prefix(&tenant, &target).unwrap_or(&target).to_owned();
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
        // Non-Raft mode: there's no SM to write to. The handler
        // is a no-op that surfaces the requested target as if
        // the swap happened — alias resolution in non-Raft mode
        // is a follow-up.
    }
    Ok(Json(AliasResponse {
        alias: name,
        target: req.target,
    }))
}

/// Phase G stub: creates `name@v2` (next version if it exists)
/// with the SAME schema as `name`, then submits the alias swap.
/// Does NOT copy records or re-chunk — that's the full
/// reindex pipeline (separate follow-up). Today this proves the
/// atomic-swap primitive end-to-end via REST.
pub async fn reindex_collection(
    State(state): State<AppState>,
    Tenant(tenant): Tenant,
    Path(name): Path<String>,
    body: Option<Json<ReindexRequest>>,
) -> Result<Json<ReindexResponse>, (StatusCode, Json<ErrorResponse>)> {
    let scoped = tenant::scope(&tenant, &name);
    let req = body.map(|Json(r)| r).unwrap_or_default();

    // 1. Look up the current schema.
    let schema_json = {
        let schemas = state.inner.schemas.read().await;
        match schemas.get(&scoped) {
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
                        error: format!("collection '{name}' not found"),
                    }),
                ));
            }
        }
    };

    // 2. Pick the next sibling version. Today we just append
    //    `@v2`; a v2 of v2 would be `@v3`. Keep it simple.
    let new_user_name = if let Some(prefix) = name.strip_suffix("@v2") {
        format!("{prefix}@v3")
    } else if name.contains("@v") {
        // generic vN bump
        let bumped = bump_version_suffix(&name).unwrap_or_else(|| format!("{name}@v2"));
        bumped
    } else {
        format!("{name}@v2")
    };
    let scoped_new = tenant::scope(&tenant, &new_user_name);

    // 3. Materialize the sibling locally with the rewritten
    //    schema. In Raft mode we ALSO submit a CreateCollection
    //    so peers learn about it; in non-Raft mode the local
    //    materialise IS the create. We always materialise so
    //    the record-copy step below has somewhere to put rows.
    let mut rewritten: serde_json::Value = serde_json::from_str(&schema_json)
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: format!("rewrite schema: {e}"),
                }),
            )
        })?;
    rewritten["name"] = serde_json::Value::String(scoped_new.clone());
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
    if let Err((status, e)) = materialize_collection_locally(&state, &new_schema, &sibling_dir).await
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
        rechunk_source_into_sibling(&state, &scoped, &scoped_new, &req).await?
    } else {
        copy_source_records_to_sibling(&state, &scoped, &scoped_new).await?
    };

    // 6. Atomic alias swap. Raft mode only — non-Raft mode
    //    falls through with "no_raft" as before, but at least
    //    the sibling now has every source record.
    {
        let raft_guard = state.inner.raft.read().await;
        if let Some(coord) = raft_guard.as_ref() {
            let swap = veda_coordinator::raft_types::ClusterCommand::SetCollectionAlias {
                alias: scoped.clone(),
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
            return Ok(Json(ReindexResponse {
                target: new_user_name,
                status: format!("alias_swapped (copied {copied} records)"),
            }));
        }
    }
    Ok(Json(ReindexResponse {
        target: new_user_name,
        status: format!("no_raft (copied {copied} records)"),
    }))
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
) -> Result<usize, (StatusCode, Json<ErrorResponse>)> {
    // Pull records from source under a read lock, then release
    // before grabbing the write lock for the destination.
    let records: Vec<veda_core::record::Record> = {
        let managers = state.inner.segment_managers.read().await;
        let Some(src_shards) = managers.get(src) else {
            return Ok(0); // source has nothing materialised
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
    for record in records {
        let shard_id = veda_coordinator::shard_router::ShardRouter::assign_shard(
            &record.id(),
            shard_count,
        );
        let Some(sm) = dst_shards.get_mut(&shard_id) else {
            continue;
        };
        // Best-effort — duplicates between retries collide; ignore.
        if sm.insert(record).is_ok() {
            copied += 1;
        }
    }
    Ok(copied)
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
    use veda_core::record::FieldValue;
    use std::collections::BTreeMap;

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
    let embedding_field: Option<String> = req
        .embedding_field
        .clone()
        .or_else(|| dst_schema.vector_fields().first().map(|f| f.name().to_owned()));
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
    let embed_client_opt: Option<veda_core::embedding_client::EmbeddingClient> = if req.re_embed
    {
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
                    if f.name() == text_field
                        || f.name() == chunk_idx_field
                    {
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
                match client.embed_text(&tc.text) {
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
                                error: format!(
                                    "embed dim mismatch: schema={dim} got={}",
                                    v.len()
                                ),
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
        let shard_id = veda_coordinator::shard_router::ShardRouter::assign_shard(
            &rec.id(),
            shard_count,
        );
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

    {
        let mut managers = state.inner.segment_managers.write().await;
        managers.remove(&scoped);
    }
    {
        let mut executors = state.inner.query_executors.write().await;
        executors.remove(&scoped);
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
                let idx_type = match idx.to_lowercase().as_str() {
                    "hnsw" => veda_core::types::IndexType::Hnsw,
                    "diskann" => veda_core::types::IndexType::DiskAnn,
                    "ivfpq" | "ivf_pq" => veda_core::types::IndexType::IvfPq,
                    "flat" => veda_core::types::IndexType::Flat,
                    "rabitq" | "rabit_q" => veda_core::types::IndexType::RaBitQ,
                    "rabitq_rerank" | "rabit_q_rerank" => {
                        veda_core::types::IndexType::RaBitQRerank
                    }
                    "hnsw_rabitq" | "hnsw_rabit_q" | "quantized_hnsw" => {
                        veda_core::types::IndexType::HnswRaBitQ
                    }
                    "cagra" | "gpu_cagra" | "gpu" => {
                        veda_core::types::IndexType::Cagra
                    }
                    "rabitq_ext_rerank" | "ext_rabitq_rerank" | "rabitq4_rerank" => {
                        veda_core::types::IndexType::RaBitQExtRerank
                    }
                    other => {
                        return Err(format!(
                            "field '{}': unknown vector index type '{other}'",
                            f.name
                        ));
                    }
                };
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
            Ok(FieldDefinition::multi_vector(&f.name, dim, max_vectors, metric)
                .with_filterable(f.filterable))
        }
        "sparse_vector" | "sparsevector" => {
            let dim = f.dimension.ok_or_else(|| {
                format!("field '{}': sparse_vector type requires 'dimension'", f.name)
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
        "string" => Ok(
            FieldDefinition::new(&f.name, FieldType::String)
                .with_filterable(f.filterable)
                .with_bm25(f.bm25)
                .with_bm25_tokenizer(f.bm25_tokenizer.clone()),
        ),
        "int32" => Ok(
            FieldDefinition::new(&f.name, FieldType::Int32).with_filterable(f.filterable),
        ),
        "int64" => Ok(
            FieldDefinition::new(&f.name, FieldType::Int64).with_filterable(f.filterable),
        ),
        "float32" => Ok(
            FieldDefinition::new(&f.name, FieldType::Float32).with_filterable(f.filterable),
        ),
        "float64" => Ok(
            FieldDefinition::new(&f.name, FieldType::Float64).with_filterable(f.filterable),
        ),
        "bool" => Ok(
            FieldDefinition::new(&f.name, FieldType::Bool).with_filterable(f.filterable),
        ),
        "timestamp" => Ok(
            FieldDefinition::new(&f.name, FieldType::Timestamp).with_filterable(f.filterable),
        ),
        "json" => Ok(
            FieldDefinition::new(&f.name, FieldType::Json).with_filterable(f.filterable),
        ),
        "uuid" => Ok(
            FieldDefinition::new(&f.name, FieldType::Uuid).with_filterable(f.filterable),
        ),
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
        FieldType::Vector { dimension, metric, model_id, .. } => {
            let m = match metric {
                DistanceMetric::Cosine => "cosine",
                DistanceMetric::Euclidean => "euclidean",
                DistanceMetric::DotProduct => "dot_product",
                DistanceMetric::Manhattan => "manhattan",
                DistanceMetric::Hamming => "hamming",
                DistanceMetric::Jaccard => "jaccard",
            };
            ("vector", Some(*dimension), Some(m.to_owned()), model_id.clone())
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
        FieldType::SparseVector { dimension, metric, .. } => {
            let m = match metric {
                DistanceMetric::Cosine => "cosine",
                DistanceMetric::DotProduct => "dot_product",
                _ => "cosine",
            };
            ("sparse_vector", Some(*dimension), Some(m.to_owned()), None)
        }
        FieldType::MultiVector { dimension, metric, .. } => {
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

    FieldInfo {
        name: f.name().to_owned(),
        field_type: type_str.to_owned(),
        filterable: f.filterable(),
        dimension,
        metric,
        model,
    }
}

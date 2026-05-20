use std::time::Instant;

use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::Json;
use serde::{Deserialize, Serialize};
use tracing::instrument;
use uuid::Uuid;
use veda_core::metrics as m;
use veda_core::record::{FieldValue, Record};
use veda_core::sparse::SparseVec;
use veda_core::types::FieldType;
use veda_core::vector::DenseVector;
use veda_query::distributed::DistributedQueryExecutor;
use veda_query::local_shard::LocalShardSearcher;

use veda_storage::blob_manager::BlobManager;
use veda_storage::blob_store::ArcBlobStore;

use super::collections::ErrorResponse;
use super::state::AppState;
use crate::tenant::{self, Tenant};

// ── Request / Response types ─────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct InsertRequest {
    pub records: Vec<serde_json::Map<String, serde_json::Value>>,
}

#[derive(Debug, Serialize)]
pub struct InsertResponse {
    pub inserted: usize,
    /// IDs assigned to the inserted records. Populated in the
    /// Raft-routed path (so the linearisability harness can read
    /// exactly the rows it just wrote). May be empty on the
    /// non-Raft path until that's threaded similarly.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub ids: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct GetRecordResponse {
    pub id: String,
    pub fields: serde_json::Map<String, serde_json::Value>,
}

#[derive(Debug, Deserialize)]
pub struct DeleteRecordsRequest {
    pub ids: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct DeleteRecordsResponse {
    pub deleted: usize,
}

#[derive(Debug, Deserialize)]
pub struct ListRecordsQuery {
    #[serde(default = "default_list_limit")]
    pub limit: usize,
    #[serde(default)]
    pub offset: usize,
}

fn default_list_limit() -> usize {
    50
}

#[derive(Debug, Serialize)]
pub struct ListRecordsResponse {
    pub records: Vec<GetRecordResponse>,
    pub total: u64,
}

// ── Handlers ─────────────────────────────────────────────────────────────

#[instrument(name = "insert", skip(state, req, headers), fields(collection = %name, record_count = req.records.len()))]
pub async fn insert(
    State(state): State<AppState>,
    Tenant(tenant): Tenant,
    Path(name): Path<String>,
    headers: axum::http::HeaderMap,
    Json(req): Json<InsertRequest>,
) -> axum::response::Response {
    if let Some(leader_url) = state.leader_rest_address().await {
        return crate::rest::collections::raft_redirect(
            &leader_url,
            &format!("/v1/collections/{name}/insert"),
        );
    }
    use axum::response::IntoResponse;
    let scoped = tenant::scope(&tenant, &name);
    // Optional transactional enrollment — when X-Veda-Txn-Id is
    // present, every record in this request is staged under that
    // txn instead of going straight into the records map. See
    // crates/veda-gateway/src/rest/txn.rs for the lifecycle.
    let txn_id = headers
        .get(crate::rest::txn::TXN_HEADER)
        .and_then(|v| v.to_str().ok())
        .map(str::to_owned);
    match insert_impl(state, scoped, txn_id, req).await {
        Ok((status, body)) => (status, body).into_response(),
        Err((status, body)) => (status, body).into_response(),
    }
}

// Phase E deprecation window: the Raft-fallback block below still
// constructs `ClusterCommand::InsertRecord` / `TxnWrite`, both
// `#[deprecated]`. Allowed for one release while collections
// flip `use_log_primary`; removed at the next major along with
// the apply arms in `store::apply_command`.
#[allow(deprecated)]
pub(crate) async fn insert_impl(
    state: AppState,
    name: String,
    txn_id: Option<String>,
    mut req: InsertRequest,
) -> Result<(StatusCode, Json<InsertResponse>), (StatusCode, Json<ErrorResponse>)> {
    let start = Instant::now();
    let schema = {
        let schemas = state.inner.schemas.read().await;
        schemas.get(&name).cloned().ok_or_else(|| {
            (
                StatusCode::NOT_FOUND,
                Json(ErrorResponse {
                    error: format!("collection '{name}' not found"),
                }),
            )
        })?
    };

    // Phase C: route this collection's writes through
    // `veda-log` when EITHER the schema explicitly opted in
    // (`use_log_primary: Some(true)`) OR `VEDA_LOG_PRIMARY=1`
    // is set as a global rollout flag.
    if schema.effective_use_log_primary() {
        let _ = start;
        return insert_via_log(state, name, txn_id, req, schema).await;
    }

    // Raft mode: submit each record as a ClusterCommand::InsertRecord
    // and return. All materialisation (local segment manager
    // inserts) happens uniformly through the reconcile task on
    // every node, including the receiver — that keeps the
    // segment-manager view identical across the cluster instead
    // of having the receiver run a slightly different code path.
    //
    // (Phase E: this block is inside an `#[allow(deprecated)]`
    // function; see the attribute on `insert_impl` above.)
    {
        let raft_guard = state.inner.raft.read().await;
        if let Some(coord) = raft_guard.as_ref() {
            // Per-shard group routing. When the
            // MultiGroupCoordinator has a group for
            // (collection, shard_id), submit there; otherwise
            // fall back to the cluster-wide _meta group.
            //
            // Today the per-shard groups are in-process single-
            // node Raft instances spawned by reconcile; they
            // don't yet replicate across nodes (that needs the
            // multiplexing Tonic transport). The _meta group
            // remains the source of truth for cross-node
            // replication. This wires the dispatching code path
            // so the future multi-node multi-group transport
            // drops in without changing this handler.
            let mg = state.inner.multi_group.read().await.clone();
            let shard_count = schema.shard_count();
            let mut applied = 0usize;
            let mut applied_ids: Vec<String> = Vec::with_capacity(req.records.len());
            for raw in req.records.iter() {
                let record = parse_record_public(raw, &schema).map_err(|e| {
                    (
                        StatusCode::BAD_REQUEST,
                        Json(ErrorResponse { error: e }),
                    )
                })?;
                let rid = record.id().to_string();
                let rjson = serde_json::to_string(&record).map_err(|e| {
                    (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(ErrorResponse {
                            error: format!("serialize record: {e}"),
                        }),
                    )
                })?;
                let cmd = match &txn_id {
                    Some(tid) => veda_coordinator::raft_types::ClusterCommand::TxnWrite {
                        txn_id: tid.clone(),
                        collection: name.clone(),
                        record_id: rid.clone(),
                        record_json: rjson,
                    },
                    None => veda_coordinator::raft_types::ClusterCommand::InsertRecord {
                        collection: name.clone(),
                        record_id: rid.clone(),
                        record_json: rjson,
                    },
                };

                // Compute the shard id for this record. Falls
                // through to the cluster-wide group below if
                // multi_group isn't configured or the group
                // hasn't been spawned yet.
                let shard_id = veda_coordinator::shard_router::ShardRouter::assign_shard(
                    &record.id(),
                    shard_count,
                );
                let group_id = format!("{}:{}", name, shard_id);
                let mut routed_to_shard = false;
                if let Some(ref mg) = mg
                    && mg.get(&group_id).await.is_some()
                {
                    if let Err(e) = mg.execute(&group_id, cmd.clone()).await {
                        tracing::warn!(
                            group_id = %group_id,
                            error = %e,
                            "per-shard submit failed; falling back to _meta",
                        );
                    } else {
                        routed_to_shard = true;
                    }
                }

                // Always also submit to the cluster-wide group —
                // that's where cross-node replication lives until
                // the multi-group transport ships. Skip when the
                // per-shard submit was sufficient and the
                // cluster-wide group is the same logical
                // destination (single-node), to avoid double-
                // counting. We detect single-node via "no peers"
                // — coord.peers().snapshot() empty.
                let peers_empty = coord.peers().snapshot().await.is_empty();
                if !routed_to_shard || !peers_empty {
                    if let Err(e) = coord.execute(cmd).await {
                        return Err((
                            StatusCode::INTERNAL_SERVER_ERROR,
                            Json(ErrorResponse {
                                error: format!("raft submit: {e}"),
                            }),
                        ));
                    }
                }
                // Dark-write to per-shard ShardLog (Phase B).
                // Errors logged but not surfaced — the Raft path
                // above is still the source of truth in Phase A/B.
                if dark_write_enabled() {
                    let log_arc = ensure_shard_log(&state, &name, shard_id).await;
                    if let Some(log) = log_arc {
                        let hlc = state.inner.tso.now().await.unwrap_or_default();
                        let payload = bytes::Bytes::from(serde_json::to_vec(raw).unwrap_or_default());
                        let key = bytes::Bytes::from(record.id().as_bytes().to_vec());
                        match log.append_one(hlc, 0, key, veda_log::LogOp::Put, payload) {
                            Ok(seq) => tracing::debug!(
                                collection = %name,
                                shard = shard_id,
                                seq_id = seq,
                                hlc_phys = hlc.physical_ns,
                                "dark-write to ShardLog ok"
                            ),
                            Err(e) => tracing::warn!(
                                collection = %name,
                                shard = shard_id,
                                error = %e,
                                "dark-write to ShardLog failed (ignored)"
                            ),
                        }
                    }
                }
                applied_ids.push(rid);
                applied += 1;
            }
            // Trigger an immediate reconcile so the receiver's
            // segment manager sees the new records before this
            // response returns. Without this, the caller would
            // have to wait up to VEDA_RAFT_RECONCILE_INTERVAL_SECS
            // before a follow-up GET /records returns the data.
            let _ = crate::rest::collections::reconcile_collections_from_raft(&state).await;
            let _ = start;
            return Ok((
                StatusCode::CREATED,
                Json(InsertResponse {
                    inserted: applied,
                    ids: applied_ids,
                }),
            ));
        }
    }

    let mut inserted = 0usize;
    let mut managers = state.inner.segment_managers.write().await;
    let shard_map = managers.get_mut(&name).ok_or_else(|| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: "no segment managers for collection".into(),
            }),
        )
    })?;

    let shard_count = schema.shard_count();

    // Get blob store for processing blob fields
    let blob_store = state.inner.blob_store.read().await.clone();

    for raw in &req.records {
        let mut record = parse_record(raw, &schema).map_err(|e| {
            (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse { error: e }),
            )
        })?;

        if !schema.blob_fields().is_empty() {
            let blob_manager = BlobManager::new(Box::new(ArcBlobStore(blob_store.clone())), &name);
            blob_manager.process_insert(&mut record, &schema).map_err(|e| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse {
                        error: format!("blob processing failed: {e}"),
                    }),
                )
            })?;
        }

        // Jump consistent hash for stable routing across shard-count changes.
        let shard_id =
            veda_coordinator::shard_router::ShardRouter::assign_shard(&record.id(), shard_count);
        let sm = shard_map.get_mut(&shard_id).ok_or_else(|| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: format!("shard {shard_id} not found"),
                }),
            )
        })?;

        let record_id_for_log = record.id();
        sm.insert(record).map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: format!("insert failed: {e}"),
                }),
            )
        })?;

        // Dark-write to per-shard ShardLog (Phase B). Same
        // behaviour + error handling as the Raft branch above.
        if dark_write_enabled() {
            let log_arc = ensure_shard_log(&state, &name, shard_id).await;
            if let Some(log) = log_arc {
                let hlc = state.inner.tso.now().await.unwrap_or_default();
                let payload = bytes::Bytes::from(serde_json::to_vec(raw).unwrap_or_default());
                let key = bytes::Bytes::from(record_id_for_log.as_bytes().to_vec());
                match log.append_one(hlc, 0, key, veda_log::LogOp::Put, payload) {
                    Ok(seq) => tracing::debug!(
                        collection = %name,
                        shard = shard_id,
                        seq_id = seq,
                        hlc_phys = hlc.physical_ns,
                        "dark-write to ShardLog ok (non-raft branch)"
                    ),
                    Err(e) => tracing::warn!(
                        collection = %name,
                        shard = shard_id,
                        error = %e,
                        "dark-write to ShardLog failed (ignored)"
                    ),
                }
            }
        }

        inserted += 1;
    }

    // Seal growing segments so data is on disk. Track whether any shard
    // actually sealed — if nothing sealed, skip compaction and rebuild.
    let mut any_sealed = false;
    for sm in shard_map.values_mut() {
        let seal_start = Instant::now();
        let sealed = sm.seal_growing().map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: format!("failed to seal segment: {e}"),
                }),
            )
        })?;
        if sealed.is_some() {
            any_sealed = true;
            metrics::histogram!(m::SEAL_DURATION, "collection" => name.clone())
                .record(seal_start.elapsed().as_secs_f64());
        }
    }

    // Run compaction only if we actually sealed something.
    if any_sealed {
        let config = veda_storage::compaction::CompactionConfig::default();
        for sm in shard_map.values_mut() {
            metrics::gauge!(m::COMPACTION_IN_PROGRESS, "collection" => name.clone()).set(1.0);
            let compact_start = Instant::now();
            match sm.compact_if_needed(&config) {
                Ok(removed) if removed > 0 => {
                    metrics::histogram!(m::COMPACTION_DURATION, "collection" => name.clone())
                        .record(compact_start.elapsed().as_secs_f64());
                    tracing::info!(removed_segments = removed, "compaction completed");
                }
                Ok(_) => {}
                Err(e) => {
                    tracing::warn!(error = %e, "compaction failed");
                }
            }
            metrics::gauge!(m::COMPACTION_IN_PROGRESS, "collection" => name.clone()).set(0.0);
        }
        let total_segments: usize = shard_map.values().map(|sm| sm.sealed_segments().len()).sum();
        metrics::gauge!(m::SEGMENTS_TOTAL, "collection" => name.clone()).set(total_segments as f64);
    }

    drop(managers);

    if any_sealed {
        metrics::gauge!(m::INDEX_REBUILD_IN_PROGRESS, "collection" => name.clone()).set(1.0);
        let rebuild_start = Instant::now();
        rebuild_executors(&state, &name).await?;
        metrics::histogram!(m::INDEX_BUILD_DURATION, "collection" => name.clone(), "index_type" => "rebuild")
            .record(rebuild_start.elapsed().as_secs_f64());
        metrics::gauge!(m::INDEX_REBUILD_IN_PROGRESS, "collection" => name.clone()).set(0.0);
    }

    metrics::histogram!(m::INSERT_DURATION, "collection" => name.clone())
        .record(start.elapsed().as_secs_f64());
    metrics::counter!(m::INSERT_TOTAL, "collection" => name.clone())
        .increment(inserted as u64);

    // Update record count gauge from segment managers.
    {
        let managers = state.inner.segment_managers.read().await;
        if let Some(shard_managers) = managers.get(&name) {
            let total: u64 = shard_managers.values().map(|sm| sm.total_records()).sum();
            metrics::gauge!(m::RECORDS_TOTAL, "collection" => name.clone()).set(total as f64);
        }
    }

    Ok((
        StatusCode::CREATED,
        Json(InsertResponse {
            inserted,
            ids: Vec::new(),
        }),
    ))
}

#[instrument(name = "list_records", skip(state), fields(collection = %name, limit = q.limit, offset = q.offset))]
pub async fn list_records(
    State(state): State<AppState>,
    Tenant(tenant): Tenant,
    Path(name): Path<String>,
    Query(q): Query<ListRecordsQuery>,
) -> Result<Json<ListRecordsResponse>, (StatusCode, Json<ErrorResponse>)> {
    let name = tenant::scope(&tenant, &name);
    {
        let schemas = state.inner.schemas.read().await;
        if !schemas.contains_key(&name) {
            return Err((
                StatusCode::NOT_FOUND,
                Json(ErrorResponse {
                    error: format!("collection '{name}' not found"),
                }),
            ));
        }
    }

    let managers = state.inner.segment_managers.read().await;
    let shard_map = managers.get(&name).ok_or_else(|| {
        (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: format!("collection '{name}' not found"),
            }),
        )
    })?;

    // Two-pass: count non-deleted records, then collect window [offset, offset+limit).
    let mut total: u64 = 0;
    let mut seen: usize = 0;
    let mut records: Vec<GetRecordResponse> = Vec::with_capacity(q.limit);

    for sm in shard_map.values() {
        let gs = sm.growing_segment();
        for doc_id in 0..gs.count() {
            if gs.is_deleted(doc_id) {
                continue;
            }
            if let Some(rec) = gs.get_record(doc_id) {
                total += 1;
                if seen >= q.offset && records.len() < q.limit {
                    records.push(record_to_response(rec));
                }
                seen += 1;
            }
        }

        for handle in sm.sealed_segments() {
            for doc_id in 0..handle.metadata.row_count as u32 {
                if handle.reader.delete_bitmap().is_deleted(doc_id) {
                    continue;
                }
                if let Ok(rec) = handle.reader.get_record(doc_id) {
                    total += 1;
                    if seen >= q.offset && records.len() < q.limit {
                        records.push(record_to_response(&rec));
                    }
                    seen += 1;
                }
            }
        }
    }

    Ok(Json(ListRecordsResponse { records, total }))
}

#[derive(Debug, Deserialize, Default)]
pub struct GetRecordQuery {
    /// When true and the gateway is in Raft mode, read the
    /// record from the Raft state machine after a barrier
    /// `Noop` client_write. This gives linearisable
    /// semantics: the returned value reflects every Insert
    /// that committed before this GET arrived. Without the
    /// flag, the read goes straight to the local SegmentManager
    /// — fast but bounded by the reconcile interval on
    /// followers.
    #[serde(default)]
    pub linearisable: bool,
    /// Phase F: MVCC snapshot timestamp (HLC u128, decimal). When
    /// the collection has `use_mvcc` on, the read returns the
    /// latest version with `commit_ts <= at`. Omitted = TSO.now()
    /// at read time. Ignored when MVCC is off for this collection.
    /// Wire type is String because Axum's `serde_urlencoded` does
    /// not support `u128`.
    #[serde(default)]
    pub at: Option<String>,
}

pub async fn get_record(
    State(state): State<AppState>,
    Tenant(tenant): Tenant,
    Path((name, id_str)): Path<(String, String)>,
    Query(q): Query<GetRecordQuery>,
) -> Result<Json<GetRecordResponse>, (StatusCode, Json<ErrorResponse>)> {
    let name = tenant::scope(&tenant, &name);
    let record_id: Uuid = id_str.parse().map_err(|_| {
        (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: format!("invalid UUID: '{id_str}'"),
            }),
        )
    })?;

    // Linearisable read path. Issue a Noop barrier through
    // Raft, then consult the state machine's records map. This
    // bypasses the local SegmentManager (which lags behind the
    // Raft state by up to VEDA_RAFT_RECONCILE_INTERVAL_SECS on
    // followers) and gives strict linearisability at the cost
    // of one Raft round trip per read.
    if q.linearisable {
        let raft_guard = state.inner.raft.read().await;
        if let Some(coord) = raft_guard.as_ref() {
            let cs = coord.linearisable_query_state().await.map_err(|e| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse {
                        error: format!("linearisable read: {e}"),
                    }),
                )
            })?;
            if !cs.collections.contains_key(&name) {
                return Err((
                    StatusCode::NOT_FOUND,
                    Json(ErrorResponse {
                        error: format!("collection '{name}' not found"),
                    }),
                ));
            }
            if let Some(records) = cs.records.get(&name) {
                if let Some(rjson) = records.get(&record_id.to_string()) {
                    let rec: veda_core::record::Record = serde_json::from_str(rjson)
                        .map_err(|e| {
                            (
                                StatusCode::INTERNAL_SERVER_ERROR,
                                Json(ErrorResponse {
                                    error: format!(
                                        "raft record deserialise: {e}"
                                    ),
                                }),
                            )
                        })?;
                    return Ok(Json(record_to_response(&rec)));
                }
            }
            return Err((
                StatusCode::NOT_FOUND,
                Json(ErrorResponse {
                    error: format!("record '{record_id}' not found"),
                }),
            ));
        }
        // Raft not configured — fall through to the local path
        // (linearisability flag is effectively a no-op when
        // there's only one node).
    }

    // Check collection exists
    {
        let schemas = state.inner.schemas.read().await;
        if !schemas.contains_key(&name) {
            return Err((
                StatusCode::NOT_FOUND,
                Json(ErrorResponse {
                    error: format!("collection '{name}' not found"),
                }),
            ));
        }
    }

    let managers = state.inner.segment_managers.read().await;
    let shard_map = managers.get(&name).ok_or_else(|| {
        (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: format!("collection '{name}' not found"),
            }),
        )
    })?;

    // Phase F: when any shard has MVCC enabled, route the read
    // through `read_at_ts`. The `?at=<ts>` query param selects the
    // snapshot; omitted defaults to TSO.now() (so the default is
    // "latest visible version", matching non-MVCC behaviour
    // semantics).
    let mvcc_active = shard_map.values().any(|sm| sm.has_mvcc());
    if mvcc_active {
        let read_ts: u128 = match q.at.as_deref() {
            Some(s) => s.parse::<u128>().map_err(|_| {
                (
                    StatusCode::BAD_REQUEST,
                    Json(ErrorResponse {
                        error: format!("invalid ?at= value '{s}': expected decimal u128"),
                    }),
                )
            })?,
            None => {
                let hlc = state.inner.tso.now().await.unwrap_or_default();
                ((hlc.physical_ns as u128) << 32) | (hlc.logical as u128)
            }
        };
        for sm in shard_map.values() {
            if !sm.has_mvcc() {
                continue;
            }
            if let Some(bytes) = sm.read_at_ts(&record_id, read_ts) {
                let rec: veda_core::record::Record = serde_json::from_slice(&bytes)
                    .map_err(|e| {
                        (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            Json(ErrorResponse {
                                error: format!("mvcc record deserialise: {e}"),
                            }),
                        )
                    })?;
                return Ok(Json(record_to_response(&rec)));
            }
        }
        return Err((
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: format!("record '{record_id}' not visible at read_ts={read_ts}"),
            }),
        ));
    }

    // Non-MVCC (legacy) path: scan segments for the record.
    for sm in shard_map.values() {
        // Check growing segment
        let gs = sm.growing_segment();
        for doc_id in 0..gs.count() {
            if let Some(rec) = gs.get_record(doc_id)
                && rec.id() == record_id
                && !gs.is_deleted(doc_id)
            {
                return Ok(Json(record_to_response(rec)));
            }
        }

        // Check sealed segments
        for handle in sm.sealed_segments() {
            for doc_id in 0..handle.metadata.row_count as u32 {
                if handle.reader.delete_bitmap().is_deleted(doc_id) {
                    continue;
                }
                if let Ok(rec) = handle.reader.get_record(doc_id)
                    && rec.id() == record_id
                {
                    return Ok(Json(record_to_response(&rec)));
                }
            }
        }
    }

    Err((
        StatusCode::NOT_FOUND,
        Json(ErrorResponse {
            error: format!("record '{record_id}' not found"),
        }),
    ))
}

pub async fn delete_records(
    State(state): State<AppState>,
    Tenant(tenant): Tenant,
    Path(name): Path<String>,
    Json(req): Json<DeleteRecordsRequest>,
) -> Result<Json<DeleteRecordsResponse>, (StatusCode, Json<ErrorResponse>)> {
    let name = tenant::scope(&tenant, &name);
    {
        let schemas = state.inner.schemas.read().await;
        if !schemas.contains_key(&name) {
            return Err((
                StatusCode::NOT_FOUND,
                Json(ErrorResponse {
                    error: format!("collection '{name}' not found"),
                }),
            ));
        }
    }

    let mut deleted = 0usize;
    let mut managers = state.inner.segment_managers.write().await;
    let shard_map = managers.get_mut(&name).ok_or_else(|| {
        (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: format!("collection '{name}' not found"),
            }),
        )
    })?;

    for id_str in &req.ids {
        let record_id: Uuid = id_str.parse().map_err(|_| {
            (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: format!("invalid UUID: '{id_str}'"),
                }),
            )
        })?;

        for sm in shard_map.values_mut() {
            if sm.delete_record(record_id).unwrap_or(false) {
                deleted += 1;
                break;
            }
        }
    }

    Ok(Json(DeleteRecordsResponse { deleted }))
}

// ── Helpers ──────────────────────────────────────────────────────────────

/// True when `VEDA_LOG_DARK_WRITE=1` (or any truthy value). The
/// dark-write path is OFF by default so existing deployments don't
/// pay the extra fsync — flip on in test / soak to exercise the
/// future Phase C primary write path.
fn dark_write_enabled() -> bool {
    matches!(
        std::env::var("VEDA_LOG_DARK_WRITE").as_deref(),
        Ok("1") | Ok("true") | Ok("TRUE")
    )
}

// Note: per-collection log-primary decision is made via
// `CollectionSchema::effective_use_log_primary()` — see
// `crates/veda-core/src/schema.rs`. This file used to have a
// helper that consulted the env flag directly; it's been
// rolled into the schema accessor so per-collection overrides
// land cleanly.

/// Phase C primary write path. Each record:
///   1. Parses + serialises (same shape the Raft path used).
///   2. Acquires a commit_ts from the TSO (if a TSO is configured).
///   3. Appends to the per-shard `ShardLog` — this is the
///      durability ack point.
///   4. Inserts into the local `SegmentManager` — immediate
///      visibility for follow-up reads.
///
/// `TxnWrite` enrollment via `X-Veda-Txn-Id` still routes through
/// the cluster-wide Raft because the txn coordinator state still
/// lives in the metadata SM. Phase G moves it.
async fn insert_via_log(
    state: AppState,
    name: String,
    txn_id: Option<String>,
    req: InsertRequest,
    schema: veda_core::schema::CollectionSchema,
) -> Result<(StatusCode, Json<InsertResponse>), (StatusCode, Json<ErrorResponse>)> {
    if let Some(tid) = txn_id.as_deref() {
        // Phase G integration: buffer mutations into the open
        // TxnHandle. Caller drives prewrite + commit via
        // `POST /v1/txn/{id}/commit`.
        let mut handles = state.inner.open_txn_handles.write().await;
        let handle = handles.get_mut(tid).ok_or_else(|| {
            (
                StatusCode::NOT_FOUND,
                Json(ErrorResponse {
                    error: format!("txn '{tid}' not open"),
                }),
            )
        })?;
        let mut applied_ids = Vec::with_capacity(req.records.len());
        for raw in req.records.iter() {
            let record = parse_record_public(raw, &schema).map_err(|e| {
                (
                    StatusCode::BAD_REQUEST,
                    Json(ErrorResponse { error: e }),
                )
            })?;
            let rid = record.id().to_string();
            let payload = bytes::Bytes::from(serde_json::to_vec(&record).map_err(|e| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse {
                        error: format!("serialize record: {e}"),
                    }),
                )
            })?);
            // Key as `{scoped_collection}/{record_id}` — the
            // MVCC key shape that pairs with how Phase F's
            // SegmentManager integration looks up versioned rows.
            let key = format!("{name}/{rid}").into_bytes();
            handle.buffer(veda_coordinator::percolator::Mutation {
                key,
                value: Some(payload),
            });
            applied_ids.push(rid);
        }
        let inserted = applied_ids.len();
        return Ok((
            StatusCode::ACCEPTED,
            Json(InsertResponse {
                inserted,
                ids: applied_ids,
            }),
        ));
    }
    let shard_count = schema.shard_count();
    let mut applied_ids: Vec<String> = Vec::with_capacity(req.records.len());
    let mut managers = state.inner.segment_managers.write().await;
    let shard_map = managers.get_mut(&name).ok_or_else(|| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: format!("no segment managers for collection '{name}'"),
            }),
        )
    })?;

    for raw in req.records.iter() {
        let record = parse_record_public(raw, &schema).map_err(|e| {
            (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse { error: e }),
            )
        })?;
        let rid = record.id().to_string();
        let shard_id = veda_coordinator::shard_router::ShardRouter::assign_shard(
            &record.id(),
            shard_count,
        );

        // (1) Acquire commit_ts from the TSO.
        let hlc = state.inner.tso.now().await.unwrap_or_default();

        // (2) Append to the per-shard log (durability).
        let log = ensure_shard_log(&state, &name, shard_id).await.ok_or_else(|| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: format!("VEDA_LOG_PRIMARY: failed to open shard log for {name}:{shard_id}"),
                }),
            )
        })?;
        let payload = bytes::Bytes::from(
            serde_json::to_vec(&record).map_err(|e| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse {
                        error: format!("serialize record: {e}"),
                    }),
                )
            })?,
        );
        let key = bytes::Bytes::from(record.id().as_bytes().to_vec());
        log.append_one(hlc, 0, key, veda_log::LogOp::Put, payload)
            .map_err(|e| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse {
                        error: format!("VEDA_LOG_PRIMARY: log append failed: {e}"),
                    }),
                )
            })?;

        // (3) Apply to local SegmentManager (visibility).
        let sm = shard_map.get_mut(&shard_id).ok_or_else(|| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: format!("shard {shard_id} not found for '{name}'"),
                }),
            )
        })?;
        // Phase F: when MVCC is enabled for this shard, insert
        // through the MVCC overlay (which also mirrors into the
        // growing segment for legacy reads). Otherwise take the
        // single-version path.
        let commit_ts = ((hlc.physical_ns as u128) << 32) | (hlc.logical as u128);
        if sm.has_mvcc() {
            sm.insert_mvcc(commit_ts, record).map_err(|e| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse {
                        error: format!("VEDA_LOG_PRIMARY: mvcc insert failed: {e}"),
                    }),
                )
            })?;
        } else {
            sm.insert(record).map_err(|e| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse {
                        error: format!("VEDA_LOG_PRIMARY: segment insert failed: {e}"),
                    }),
                )
            })?;
        }

        applied_ids.push(rid);
    }

    let inserted = applied_ids.len();
    Ok((
        StatusCode::CREATED,
        Json(InsertResponse {
            inserted,
            ids: applied_ids,
        }),
    ))
}

/// Look up or create a ShardLog for `(collection_scoped, shard_id)`.
/// Logs live under `{data_dir}/_logs/{scoped}/{shard_id}/`.
async fn ensure_shard_log(
    state: &AppState,
    scoped_collection: &str,
    shard_id: u32,
) -> Option<std::sync::Arc<veda_log::ShardLog>> {
    let key = format!("{scoped_collection}:{shard_id}");
    {
        let guard = state.inner.shard_logs.read().await;
        if let Some(log) = guard.get(&key) {
            return Some(log.clone());
        }
    }
    // Slow path — construct the log on disk.
    let dir = state
        .inner
        .data_dir
        .join("_logs")
        .join(scoped_collection)
        .join(format!("{shard_id:04}"));
    let log = match veda_log::ShardLog::open(&dir) {
        Ok(l) => std::sync::Arc::new(l),
        Err(e) => {
            tracing::warn!(
                dir = ?dir,
                error = %e,
                "ShardLog::open failed; dark-write skipped for this shard"
            );
            return None;
        }
    };
    let mut guard = state.inner.shard_logs.write().await;
    Some(guard.entry(key).or_insert(log).clone())
}

/// Public wrapper exposing the record-parsing logic to other gateway modules
/// (notably `streaming`). Internal callers should keep using `parse_record`.
pub fn parse_record_public(
    raw: &serde_json::Map<String, serde_json::Value>,
    schema: &veda_core::schema::CollectionSchema,
) -> Result<Record, String> {
    parse_record(raw, schema)
}

fn parse_record(
    raw: &serde_json::Map<String, serde_json::Value>,
    schema: &veda_core::schema::CollectionSchema,
) -> Result<Record, String> {
    let mut record = Record::new();

    for field_def in schema.fields() {
        let name = field_def.name();
        let value = match raw.get(name) {
            Some(v) => v,
            None => {
                // Vector fields are required
                if matches!(field_def.field_type(), FieldType::Vector { .. }) {
                    return Err(format!("missing required vector field '{name}'"));
                }
                continue;
            }
        };

        let fv = json_to_field_value(value, field_def.field_type())
            .map_err(|e| format!("field '{name}': {e}"))?;
        record.set(name, fv);
    }

    Ok(record)
}

fn json_to_field_value(
    value: &serde_json::Value,
    field_type: &FieldType,
) -> Result<FieldValue, String> {
    match field_type {
        FieldType::Vector { .. } => {
            let arr = value
                .as_array()
                .ok_or("expected array for vector field")?;
            let floats: Vec<f32> = arr
                .iter()
                .map(|v| {
                    v.as_f64()
                        .map(|f| f as f32)
                        .ok_or("vector elements must be numbers")
                })
                .collect::<Result<_, _>>()?;
            Ok(FieldValue::Vector(DenseVector::new(floats)))
        }
        FieldType::SparseVector { dimension, .. } => {
            // Accept either { "indices": [...], "values": [...] } or
            // { "<idx>": <val>, ... } (object form).
            let sparse = parse_sparse_value(value, *dimension)?;
            Ok(FieldValue::SparseVector(sparse))
        }
        FieldType::MultiVector { dimension, max_vectors, .. } => {
            let arr = value
                .as_array()
                .ok_or("expected array-of-arrays for multi_vector field")?;
            if *max_vectors > 0 && arr.len() as u32 > *max_vectors {
                return Err(format!(
                    "multi_vector has {} rows; field max_vectors={max_vectors}",
                    arr.len()
                ));
            }
            let mut rows: Vec<Vec<f32>> = Vec::with_capacity(arr.len());
            for row in arr {
                let inner = row
                    .as_array()
                    .ok_or("each multi_vector row must be an array of numbers")?;
                if inner.len() as u32 != *dimension {
                    return Err(format!(
                        "multi_vector row length {} does not match dimension {dimension}",
                        inner.len()
                    ));
                }
                let floats: Vec<f32> = inner
                    .iter()
                    .map(|v| v.as_f64().map(|f| f as f32).ok_or("row elements must be numbers"))
                    .collect::<Result<_, _>>()?;
                rows.push(floats);
            }
            Ok(FieldValue::MultiVector(rows))
        }
        FieldType::String => {
            let s = value.as_str().ok_or("expected string")?;
            Ok(FieldValue::String(s.to_owned()))
        }
        FieldType::Int32 => {
            let n = value.as_i64().ok_or("expected integer")?;
            Ok(FieldValue::Int32(n as i32))
        }
        FieldType::Int64 => {
            let n = value.as_i64().ok_or("expected integer")?;
            Ok(FieldValue::Int64(n))
        }
        FieldType::Float32 => {
            let n = value.as_f64().ok_or("expected number")?;
            Ok(FieldValue::Float32(n as f32))
        }
        FieldType::Float64 => {
            let n = value.as_f64().ok_or("expected number")?;
            Ok(FieldValue::Float64(n))
        }
        FieldType::Bool => {
            let b = value.as_bool().ok_or("expected boolean")?;
            Ok(FieldValue::Bool(b))
        }
        FieldType::Timestamp => {
            let n = value.as_i64().ok_or("expected integer timestamp")?;
            Ok(FieldValue::Timestamp(n))
        }
        FieldType::Blob => {
            // User sends base64 data as a string; store as String so BlobManager
            // can detect it, decode it, store the bytes, and replace with BlobRef.
            let s = value.as_str().ok_or("expected base64 string for blob field")?;
            Ok(FieldValue::String(s.to_owned()))
        }
        FieldType::Json => Ok(FieldValue::Json(value.clone())),
        FieldType::Uuid => {
            let s = value.as_str().ok_or("expected string for uuid")?;
            let u: uuid::Uuid = s.parse().map_err(|_| "invalid UUID")?;
            Ok(FieldValue::Uuid(u))
        }
        FieldType::Array(inner) => {
            let arr = value.as_array().ok_or("expected JSON array for array field")?;
            let elems: Result<Vec<FieldValue>, String> =
                arr.iter().map(|v| json_to_field_value(v, inner)).collect();
            Ok(FieldValue::Array(elems?))
        }
    }
}

/// Parse a sparse vector from either `{"indices":[..],"values":[..]}` or
/// `{"42": 0.1, "100": 0.5}` (string-keyed object) form. Validates that
/// all indices are within the declared dimension.
fn parse_sparse_value(value: &serde_json::Value, dimension: u32) -> Result<SparseVec, String> {
    if let Some(obj) = value.as_object() {
        // Two cases: indices/values arrays, or term:weight pairs.
        if let (Some(idx_v), Some(val_v)) = (obj.get("indices"), obj.get("values")) {
            let idx_arr = idx_v.as_array().ok_or("'indices' must be an array")?;
            let val_arr = val_v.as_array().ok_or("'values' must be an array")?;
            let indices: Vec<u32> = idx_arr
                .iter()
                .map(|v| v.as_u64().and_then(|n| u32::try_from(n).ok()).ok_or("indices must be non-negative integers fitting in u32"))
                .collect::<Result<_, _>>()?;
            let values: Vec<f32> = val_arr
                .iter()
                .map(|v| v.as_f64().map(|f| f as f32).ok_or("values must be numbers"))
                .collect::<Result<_, _>>()?;
            for &i in &indices {
                if i >= dimension {
                    return Err(format!(
                        "sparse vector index {i} out of range (dimension={dimension})"
                    ));
                }
            }
            SparseVec::new(indices, values).map_err(|e| e.to_string())
        } else {
            // term:weight object.
            let mut pairs: Vec<(u32, f32)> = Vec::with_capacity(obj.len());
            for (k, v) in obj {
                let idx: u32 = k.parse().map_err(|_| format!("sparse vector key '{k}' must be a u32 index"))?;
                if idx >= dimension {
                    return Err(format!(
                        "sparse vector index {idx} out of range (dimension={dimension})"
                    ));
                }
                let val = v.as_f64().ok_or_else(|| format!("sparse vector value at '{k}' must be a number"))?;
                pairs.push((idx, val as f32));
            }
            SparseVec::from_pairs(pairs).map_err(|e| e.to_string())
        }
    } else {
        Err("expected object for sparse_vector field".into())
    }
}

fn record_to_response(rec: &Record) -> GetRecordResponse {
    let mut fields = serde_json::Map::new();
    for (key, value) in rec.fields() {
        fields.insert(key.clone(), field_value_to_json(value));
    }

    GetRecordResponse {
        id: rec.id().to_string(),
        fields,
    }
}

pub(crate) fn field_value_to_json(fv: &FieldValue) -> serde_json::Value {
    match fv {
        FieldValue::String(s) => serde_json::Value::String(s.clone()),
        FieldValue::Int32(n) => serde_json::json!(*n),
        FieldValue::Int64(n) => serde_json::json!(*n),
        FieldValue::Float32(n) => serde_json::json!(*n),
        FieldValue::Float64(n) => serde_json::json!(*n),
        FieldValue::Bool(b) => serde_json::Value::Bool(*b),
        FieldValue::Timestamp(t) => serde_json::json!(*t),
        FieldValue::Vector(v) => serde_json::json!(v.as_slice()),
        FieldValue::SparseVector(sv) => serde_json::json!({
            "indices": sv.indices(),
            "values": sv.values(),
        }),
        FieldValue::MultiVector(rows) => serde_json::Value::Array(
            rows.iter()
                .map(|row| serde_json::Value::Array(row.iter().map(|f| serde_json::json!(*f)).collect()))
                .collect(),
        ),
        FieldValue::Json(j) => j.clone(),
        FieldValue::BlobRef(s) => serde_json::Value::String(s.clone()),
        FieldValue::Uuid(u) => serde_json::Value::String(u.to_string()),
        FieldValue::Array(arr) => {
            serde_json::Value::Array(arr.iter().map(field_value_to_json).collect())
        }
        FieldValue::Null => serde_json::Value::Null,
    }
}

/// Rebuild the DistributedQueryExecutor for a collection after data changes.
///
/// Creates new LocalShardSearcher instances that load sealed segments from
/// disk, then builds vector indexes so search can find the data.
#[instrument(name = "rebuild_executors", skip(state), fields(collection = %collection_name))]
pub async fn rebuild_executors(
    state: &AppState,
    collection_name: &str,
) -> Result<(), (StatusCode, Json<ErrorResponse>)> {
    let schema = {
        let schemas = state.inner.schemas.read().await;
        schemas.get(collection_name).cloned().ok_or_else(|| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: "schema not found during rebuild".into(),
                }),
            )
        })?
    };

    let collection_dir = state.inner.data_dir.join(collection_name);
    let mut executor = DistributedQueryExecutor::new();

    for shard_id in 0..schema.shard_count() {
        let mut sm = veda_storage::segment_manager::SegmentManager::new(
            schema.clone(),
            shard_id,
            &collection_dir,
            512 * 1024 * 1024,
        )
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: format!("failed to recreate segment manager: {e}"),
                }),
            )
        })?;

        // Load sealed segments from disk
        sm.load_sealed_segments().map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: format!("failed to load sealed segments: {e}"),
                }),
            )
        })?;

        let mut searcher = LocalShardSearcher::new(sm);
        searcher.build_indexes().map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: format!("failed to build indexes: {e}"),
                }),
            )
        })?;
        executor.register_shard(shard_id, Box::new(searcher));
    }

    let mut executors = state.inner.query_executors.write().await;
    executors.insert(collection_name.to_owned(), executor);

    Ok(())
}

//! NATS-driven streaming ingestion worker.
//!
//! When `VEDA_STREAMING_NATS` is set at startup, a background task subscribes
//! to subjects matching `veda.ingest.<collection>` and inserts JSON-encoded
//! records into the matching collection. The body of each message must be a
//! JSON record object identical to one element of the REST `insert` payload.
//!
//! Records are appended to the in-memory growing segment only — no automatic
//! seal/rebuild. They become searchable after the next REST `insert` or
//! `embed-insert` call on the same collection triggers a seal. This avoids
//! the per-message executor rebuild that would otherwise make streaming
//! O(N²).

use std::sync::Arc;

use futures::StreamExt;
use tracing::{error, info, warn};
use veda_core::error::{Result, VedaError};

use crate::rest::state::AppState;

/// Spawn a NATS subscriber that ingests records into Veda collections.
///
/// Returns once the connection is established; the actual consumption runs
/// in the spawned task. Errors during message handling are logged but do
/// not stop the worker.
pub async fn spawn_ingestion_worker(state: AppState, nats_url: String) -> Result<()> {
    let client = async_nats::connect(&nats_url)
        .await
        .map_err(|e| VedaError::Storage(format!("streaming: nats connect failed: {e}")))?;
    info!(url = %nats_url, "streaming ingestion worker connected");

    let mut subscriber = client
        .subscribe("veda.ingest.>")
        .await
        .map_err(|e| VedaError::Storage(format!("streaming: subscribe failed: {e}")))?;

    let state = Arc::new(state);

    tokio::spawn(async move {
        while let Some(msg) = subscriber.next().await {
            let subject = msg.subject.as_str();
            let collection = match subject.strip_prefix("veda.ingest.") {
                Some(s) if !s.is_empty() => s.to_owned(),
                _ => {
                    warn!(subject, "streaming: ignoring malformed subject");
                    continue;
                }
            };
            if let Err(e) = handle_message(&state, &collection, &msg.payload).await {
                error!(collection, error = %e, "streaming: insert failed");
            }
        }
        info!("streaming ingestion worker stopped");
    });

    Ok(())
}

async fn handle_message(
    state: &AppState,
    collection: &str,
    payload: &[u8],
) -> Result<()> {
    use veda_core::schema::CollectionSchema;

    let raw: serde_json::Map<String, serde_json::Value> =
        serde_json::from_slice(payload)
            .map_err(|e| VedaError::Storage(format!("streaming: bad json: {e}")))?;

    let schema: CollectionSchema = {
        let schemas = state.inner.schemas.read().await;
        match schemas.get(collection) {
            Some(s) => s.clone(),
            None => {
                return Err(VedaError::CollectionNotFound(collection.to_owned()));
            }
        }
    };

    let record = crate::rest::records::parse_record_public(&raw, &schema)
        .map_err(VedaError::Storage)?;

    let mut managers = state.inner.segment_managers.write().await;
    let shard_map = managers
        .get_mut(collection)
        .ok_or_else(|| VedaError::CollectionNotFound(collection.to_owned()))?;
    let shard_id = veda_coordinator::shard_router::ShardRouter::assign_shard(
        &record.id(),
        schema.shard_count(),
    );
    let sm = shard_map
        .get_mut(&shard_id)
        .ok_or_else(|| VedaError::Storage(format!("shard {shard_id} not found")))?;
    sm.insert(record)?;
    Ok(())
}

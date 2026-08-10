//! Logs-as-a-collection ingest — `POST /v1/collections/{name}/ingest-logs`.
//!
//! Turns Veda into a searchable log archive. The caller POSTs a batch of
//! raw log lines (JSON or plain text); each line is normalized by
//! [`crate::logs::parse::parse_log_line`] into `{ message, level, ts, ...fields }`
//! and inserted through the **existing** insert path so it picks up the
//! collection's indexes (BM25 full-text on `message`, scalar filters on
//! `level`/`ts`) and — when the schema declares a vector field with
//! `auto_embed` over `message` — semantic embeddings.
//!
//! ## Recommended logs-collection schema
//!
//! Create the target collection like this so ingest + search work well:
//!
//! ```json
//! {
//!   "name": "app_logs",
//!   "fields": [
//!     { "name": "message", "type": "string", "bm25": true },
//!     { "name": "level",   "type": "string", "filterable": true },
//!     { "name": "ts",      "type": "int64",  "filterable": true }
//!     // optional, for semantic search:
//!     // { "name": "embedding", "type": "vector", "dimension": <d>,
//!     //   "metric": "cosine", "auto_embed": { "source": "message" } }
//!   ]
//! }
//! ```
//!
//! `_expires_at` is a reserved TTL field (unix seconds) — no schema entry
//! is needed; pass `ttl_secs` on the request to stamp it.
//!
//! ## Field handling
//!
//! Insertion reuses [`crate::rest::records`] / [`crate::rest::embed_insert`],
//! which map raw keys to schema fields and **silently drop keys not present
//! in the schema** (same as `/insert`). So `message`/`level`/`ts` (and
//! `_expires_at`) land in the record; any extra `fields` parsed off a JSON
//! log line are kept only if the collection declares a matching field.
//!
//! ## Auth / tenant
//!
//! This is a WRITE, gated exactly like `/insert`: the shared auth +
//! tenant middleware run first, and the collection name is tenant-scoped.
//! It is **not** admin-only.

use std::time::{SystemTime, UNIX_EPOCH};

use axum::Json;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tracing::instrument;
use veda_core::record::EXPIRES_AT_FIELD;
use veda_core::types::FieldType;

use super::collections::ErrorResponse;
use super::embed_insert::{EmbedInsertRequest, embed_insert};
use super::records::{InsertRequest, insert_impl};
use super::state::AppState;
use crate::logs::parse::parse_log_line;
use crate::tenant::{self, Tenant};

fn default_level() -> String {
    "INFO".to_owned()
}

#[derive(Debug, Deserialize)]
pub struct IngestLogsRequest {
    /// Raw log lines, JSON objects or plain text.
    pub lines: Vec<String>,
    /// Level applied when a line carries none. Default `INFO`.
    #[serde(default = "default_level")]
    pub default_level: String,
    /// Optional time-to-live in seconds. When set, each record gets
    /// `_expires_at = now_secs + ttl_secs` so it's swept at query time.
    #[serde(default)]
    pub ttl_secs: Option<u64>,
}

#[derive(Debug, Serialize)]
pub struct IngestLogsResponse {
    pub ingested: usize,
}

#[instrument(name = "ingest_logs", skip(state, req), fields(collection = %name, line_count = req.lines.len()))]
pub async fn ingest_logs(
    State(state): State<AppState>,
    Tenant(tenant): Tenant,
    Path(name): Path<String>,
    Json(req): Json<IngestLogsRequest>,
) -> Result<(StatusCode, Json<IngestLogsResponse>), (StatusCode, Json<ErrorResponse>)> {
    let scoped = tenant::scope(&tenant, &name);
    // Resolve the alias like `/insert` does — otherwise lines are written
    // into the abandoned base collection while reads follow the alias.
    let scoped = state.inner.aliases.resolve(&scoped).await;

    let schema = {
        let schemas = state.inner.schemas.read().await;
        schemas.get(&scoped).cloned().ok_or_else(|| {
            (
                StatusCode::NOT_FOUND,
                Json(ErrorResponse {
                    error: format!("collection '{}' not found", tenant::display_name(&scoped)),
                }),
            )
        })?
    };

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let now_ms = now.as_millis() as i64;
    let now_secs = now.as_secs() as i64;
    let expires_at = req.ttl_secs.map(|ttl| now_secs + ttl as i64);

    // Build one raw record map per line via the pure parser.
    let mut records: Vec<Map<String, Value>> = Vec::with_capacity(req.lines.len());
    for line in &req.lines {
        let parsed = parse_log_line(line, &req.default_level, now_ms);
        let mut rec = parsed.fields; // extra scalar fields first
        rec.insert("message".into(), Value::String(parsed.message));
        rec.insert("level".into(), Value::String(parsed.level));
        rec.insert("ts".into(), Value::Number(parsed.ts_ms.into()));
        if let Some(exp) = expires_at {
            rec.insert(EXPIRES_AT_FIELD.into(), Value::Number(exp.into()));
        }
        records.push(rec);
    }

    let ingested = records.len();

    // Reuse the existing insert path. If the collection has a vector
    // field with a schema-level `auto_embed`, route through embed-insert
    // (so messages get embedded for semantic search); otherwise plain
    // insert. Both paths drop keys not present in the schema, exactly
    // like `/insert`.
    let has_auto_embed = schema.vector_fields().iter().any(|fd| {
        matches!(
            fd.field_type(),
            FieldType::Vector {
                auto_embed: Some(_),
                ..
            }
        )
    });

    if has_auto_embed {
        let embed_req = EmbedInsertRequest {
            records,
            auto_embed: Default::default(), // inferred from schema
        };
        // Delegate to the embed-insert handler (same auth/tenant already
        // applied here; it re-scopes the bare name, so pass `name`).
        let (_status, resp) = embed_insert(
            State(state.clone()),
            Tenant(tenant.clone()),
            Path(name.clone()),
            Json(embed_req),
        )
        .await?;
        return Ok((
            StatusCode::CREATED,
            Json(IngestLogsResponse {
                ingested: resp.0.inserted,
            }),
        ));
    }

    // Plain insert path (no embedding required).
    let _ = ingested;
    let insert_req = InsertRequest { records };
    let (_status, resp) = insert_impl(state.clone(), scoped, None, insert_req).await?;
    Ok((
        StatusCode::CREATED,
        Json(IngestLogsResponse {
            ingested: resp.0.inserted,
        }),
    ))
}

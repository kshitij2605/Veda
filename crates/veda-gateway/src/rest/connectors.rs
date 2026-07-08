//! Admin REST API for data-source connectors (`/v1/admin/connectors*`).
//!
//! A connector is a persisted, manageable data source. Two kinds today:
//!   * `Web`        — a set of seed URLs crawled (via crawl4ai) into a
//!     target collection. Has a `/run` "crawl now" action.
//!   * `WebSearch`  — a lightweight marker that agentic web search is
//!     enabled for this tenant; the actual tool is the
//!     Phase-2 `web_search` agent tool. No `/run` action.
//!
//! Admin-only, mirroring [`crate::rest::slos`]: registered under the
//! protected router with an extra `rbac::require_admin` route layer.
//! Persisted through the existing [`SavedObjectStore`] with
//! `type = "connector"`, tenant-scoped exactly like RAG eval-sets
//! ([`crate::rest::rag_eval`]): the stored id is `{tenant}__{id}` and the
//! user-facing id has the prefix stripped.
//!
//!   GET    /v1/admin/connectors          — list connectors
//!   POST   /v1/admin/connectors          — create (server-assigns id if absent)
//!   GET    /v1/admin/connectors/{id}      — get one
//!   PUT    /v1/admin/connectors/{id}      — replace one
//!   DELETE /v1/admin/connectors/{id}      — delete one
//!   POST   /v1/admin/connectors/{id}/run  — crawl a Web connector now

use axum::Json;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use serde::{Deserialize, Serialize};

use super::collections::ErrorResponse;
use super::document::parse_chunking_strategy;
use super::ingest_url::crawl_and_ingest;
use super::saved_objects::{SavedObject, SavedObjectStore};
use super::state::AppState;
use crate::tenant::{self, Tenant};

/// `SavedObject.type` tag for connectors.
const TYPE_CONNECTOR: &str = "connector";

/// Kind of data-source connector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ConnectorKind {
    /// A crawl4ai web source: seed URLs → target collection.
    Web,
    /// A marker that agentic web search is enabled.
    WebSearch,
}

/// A persisted data-source connector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Connector {
    /// User-facing id. Optional on create — the server assigns
    /// `conn-{uuid}` when empty.
    #[serde(default)]
    pub id: String,
    pub kind: ConnectorKind,
    pub name: String,
    /// Kind-specific config (see module docs). Opaque to the store.
    #[serde(default)]
    pub config: serde_json::Value,
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default)]
    pub created_at: i64,
    #[serde(default)]
    pub updated_at: i64,
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Serialize)]
pub struct ConnectorsResponse {
    pub connectors: Vec<Connector>,
}

#[derive(Debug, Serialize)]
pub struct RunResponse {
    pub pages: usize,
    pub chunks_inserted: usize,
    pub per_url: Vec<RunPerUrl>,
}

#[derive(Debug, Serialize)]
pub struct RunPerUrl {
    pub url: String,
    pub title: Option<String>,
    pub chunks_inserted: usize,
}

fn err(code: StatusCode, msg: String) -> (StatusCode, Json<ErrorResponse>) {
    (code, Json(ErrorResponse { error: msg }))
}

fn now_ms() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

// ── Persistence helpers (SavedObjects, tenant-scoped) ─────────────────────

fn to_saved(c: &Connector) -> SavedObject {
    SavedObject {
        id: c.id.clone(),
        kind: TYPE_CONNECTOR.to_owned(),
        name: c.name.clone(),
        body: serde_json::to_value(c).unwrap_or(serde_json::Value::Null),
        owner: None,
        created_at: c.created_at,
        updated_at: c.updated_at,
        last_used_at: None,
    }
}

async fn save(store: &SavedObjectStore, c: &Connector) {
    store.put(to_saved(c)).await;
}

/// Fetch a connector by its already-scoped id, decoding the body.
async fn get_scoped(store: &SavedObjectStore, scoped_id: &str) -> Option<Connector> {
    let obj = store.get(scoped_id).await?;
    if obj.kind != TYPE_CONNECTOR {
        return None;
    }
    serde_json::from_value(obj.body).ok()
}

// ── CRUD handlers ─────────────────────────────────────────────────────────

/// `GET /v1/admin/connectors` — connectors for the caller's tenant.
pub async fn list_connectors(
    State(state): State<AppState>,
    Tenant(tenant): Tenant,
) -> Json<ConnectorsResponse> {
    let mut connectors: Vec<Connector> = state
        .inner
        .saved_objects
        .list_by_type(TYPE_CONNECTOR)
        .await
        .into_iter()
        .filter_map(|o| serde_json::from_value::<Connector>(o.body).ok())
        .collect();
    // Keep only this tenant's connectors and strip the prefix for display.
    connectors.retain(|c| tenant::strip_prefix(&tenant, &c.id).is_some());
    for c in &mut connectors {
        if let Some(name) = tenant::strip_prefix(&tenant, &c.id) {
            c.id = name.to_owned();
        }
    }
    Json(ConnectorsResponse { connectors })
}

/// `POST /v1/admin/connectors` — create. The server assigns a `conn-…`
/// id when the body's `id` is empty. Returns the created connector (201).
pub async fn create_connector(
    State(state): State<AppState>,
    Tenant(tenant): Tenant,
    Json(mut c): Json<Connector>,
) -> Result<(StatusCode, Json<Connector>), (StatusCode, Json<ErrorResponse>)> {
    if c.id.trim().is_empty() {
        c.id = format!("conn-{}", uuid::Uuid::new_v4());
    }
    let now = now_ms();
    c.created_at = now;
    c.updated_at = now;
    let user_id = c.id.clone();
    c.id = tenant::scope(&tenant, &user_id);
    save(&state.inner.saved_objects, &c).await;
    c.id = user_id;
    Ok((StatusCode::CREATED, Json(c)))
}

/// `GET /v1/admin/connectors/{id}` — one connector (404 if absent / not
/// owned by this tenant).
pub async fn get_connector(
    State(state): State<AppState>,
    Tenant(tenant): Tenant,
    Path(id): Path<String>,
) -> Result<Json<Connector>, (StatusCode, Json<ErrorResponse>)> {
    let scoped = tenant::scope(&tenant, &id);
    let mut c = get_scoped(&state.inner.saved_objects, &scoped)
        .await
        .ok_or_else(|| {
            err(
                StatusCode::NOT_FOUND,
                format!("no connector with id '{id}'"),
            )
        })?;
    c.id = id;
    Ok(Json(c))
}

/// `PUT /v1/admin/connectors/{id}` — replace. The path id wins; 404 if it
/// does not already exist for this tenant. `created_at` is preserved.
pub async fn put_connector(
    State(state): State<AppState>,
    Tenant(tenant): Tenant,
    Path(id): Path<String>,
    Json(mut c): Json<Connector>,
) -> Result<Json<Connector>, (StatusCode, Json<ErrorResponse>)> {
    let scoped = tenant::scope(&tenant, &id);
    let existing = get_scoped(&state.inner.saved_objects, &scoped)
        .await
        .ok_or_else(|| {
            err(
                StatusCode::NOT_FOUND,
                format!("no connector with id '{id}'"),
            )
        })?;
    c.created_at = existing.created_at;
    c.updated_at = now_ms();
    c.id = scoped;
    save(&state.inner.saved_objects, &c).await;
    c.id = id;
    Ok(Json(c))
}

/// `DELETE /v1/admin/connectors/{id}` — remove (204, idempotent).
pub async fn delete_connector(
    State(state): State<AppState>,
    Tenant(tenant): Tenant,
    Path(id): Path<String>,
) -> StatusCode {
    let scoped = tenant::scope(&tenant, &id);
    state.inner.saved_objects.remove(&scoped).await;
    StatusCode::NO_CONTENT
}

// ── Web config + run ──────────────────────────────────────────────────────

/// Config shape for a `Web` connector.
#[derive(Debug, Deserialize, Default)]
struct WebConfig {
    #[serde(default)]
    urls: Vec<String>,
    #[serde(default)]
    collection: String,
    #[serde(default)]
    deep_crawl: bool,
    #[serde(default)]
    max_pages: Option<u32>,
    #[serde(default)]
    chunker: Option<String>,
    #[serde(default)]
    chunk_size: Option<usize>,
    #[serde(default)]
    chunk_overlap: Option<usize>,
    #[serde(default)]
    ttl_secs: Option<u64>,
}

/// `POST /v1/admin/connectors/{id}/run` — for a `Web` connector, crawl its
/// configured `urls` into its `collection` via the shared Phase-1 ingest
/// path. 404 unknown id, 400 if not a Web connector / missing config,
/// 503 if crawl4ai is unconfigured.
pub async fn run_connector(
    State(state): State<AppState>,
    Tenant(tenant): Tenant,
    Path(id): Path<String>,
) -> Result<Json<RunResponse>, (StatusCode, Json<ErrorResponse>)> {
    let scoped = tenant::scope(&tenant, &id);
    let connector = get_scoped(&state.inner.saved_objects, &scoped)
        .await
        .ok_or_else(|| {
            err(
                StatusCode::NOT_FOUND,
                format!("no connector with id '{id}'"),
            )
        })?;

    if connector.kind != ConnectorKind::Web {
        return Err(err(
            StatusCode::BAD_REQUEST,
            "run is only supported for Web connectors".to_owned(),
        ));
    }

    let cfg: WebConfig = serde_json::from_value(connector.config.clone()).map_err(|e| {
        err(
            StatusCode::BAD_REQUEST,
            format!("invalid Web connector config: {e}"),
        )
    })?;
    if cfg.collection.trim().is_empty() {
        return Err(err(
            StatusCode::BAD_REQUEST,
            "Web connector config is missing 'collection'".to_owned(),
        ));
    }
    let urls: Vec<String> = cfg.urls.into_iter().filter(|s| !s.is_empty()).collect();
    if urls.is_empty() {
        return Err(err(
            StatusCode::BAD_REQUEST,
            "Web connector config has no 'urls'".to_owned(),
        ));
    }

    // Resolve the (tenant-scoped) target collection schema.
    let collection_name = tenant::scope(&tenant, &cfg.collection);
    let schema = {
        let schemas = state.inner.schemas.read().await;
        schemas.get(&collection_name).cloned().ok_or_else(|| {
            err(
                StatusCode::NOT_FOUND,
                format!("Collection '{}' not found", cfg.collection),
            )
        })?
    };

    let strategy =
        parse_chunking_strategy(cfg.chunker.as_deref(), cfg.chunk_size, cfg.chunk_overlap);
    let expires_at = cfg.ttl_secs.map(|ttl| now_ms() / 1000 + ttl as i64);

    let outcome = crawl_and_ingest(
        &state,
        &tenant,
        &collection_name,
        &schema,
        &urls,
        cfg.deep_crawl,
        cfg.max_pages,
        &strategy,
        expires_at,
    )
    .await
    .map_err(|e| err(e.status, e.message))?;

    Ok(Json(RunResponse {
        pages: outcome.pages,
        chunks_inserted: outcome.chunks_inserted,
        per_url: outcome
            .per_url
            .into_iter()
            .map(|p| RunPerUrl {
                url: p.url,
                title: p.title,
                chunks_inserted: p.chunks_inserted,
            })
            .collect(),
    }))
}

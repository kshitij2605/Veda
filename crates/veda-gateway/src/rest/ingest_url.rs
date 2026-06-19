//! `POST /v1/collections/{name}/ingest-url` — crawl one or more URLs via
//! crawl4ai, clean to markdown, and run each page through the shared
//! [`super::ingest_common::ingest_markdown`] pipeline (chunk -> embed ->
//! insert). Tenant/RLS-scoped like `process-document`.
//!
//! Opt-in: when `VEDA_CRAWL4AI_ENDPOINT` is unset the route returns a
//! clean 503 "not configured" error rather than panicking.

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::Json;
use serde::{Deserialize, Serialize};
use tracing::instrument;

use super::crawl4ai::Crawl4aiClient;
use super::document::parse_chunking_strategy;
use super::ingest_common::{ingest_markdown, SourceMeta};
use super::state::AppState;
use crate::tenant::{self, Tenant};

#[derive(Debug, Deserialize)]
pub struct IngestUrlRequest {
    /// Single URL to crawl. Either `url` or `urls` must be present.
    #[serde(default)]
    pub url: Option<String>,
    /// Multiple URLs to crawl.
    #[serde(default)]
    pub urls: Option<Vec<String>>,
    /// Follow links from the seed URL(s) (best-effort; crawl4ai caps it).
    #[serde(default)]
    pub deep_crawl: bool,
    /// Page cap for deep crawls.
    #[serde(default)]
    pub max_pages: Option<u32>,
    /// Chunker name (same vocabulary as `process-document`).
    #[serde(default)]
    pub chunker: Option<String>,
    /// Chunk size override.
    #[serde(default)]
    pub chunk_size: Option<usize>,
    /// Chunk overlap override.
    #[serde(default)]
    pub chunk_overlap: Option<usize>,
    /// When set, stamp `_expires_at = now + ttl_secs` on every record.
    #[serde(default)]
    pub ttl_secs: Option<u64>,
}

#[derive(Debug, Serialize)]
pub struct PerUrlResult {
    pub url: String,
    pub title: Option<String>,
    pub chunks_inserted: usize,
}

#[derive(Debug, Serialize)]
pub struct IngestUrlResponse {
    pub status: String,
    pub pages: usize,
    pub chunks_inserted: usize,
    pub per_url: Vec<PerUrlResult>,
}

#[derive(Debug, Serialize)]
pub struct IngestUrlErrorResponse {
    pub error: IngestUrlError,
}

#[derive(Debug, Serialize)]
pub struct IngestUrlError {
    pub code: String,
    pub message: String,
}

fn err(status: StatusCode, code: &str, message: String) -> (StatusCode, Json<IngestUrlErrorResponse>) {
    (
        status,
        Json(IngestUrlErrorResponse {
            error: IngestUrlError {
                code: code.to_owned(),
                message,
            },
        }),
    )
}

#[instrument(name = "ingest_url", skip(state, req), fields(collection = %name))]
pub async fn ingest_url(
    State(state): State<AppState>,
    Tenant(tenant): Tenant,
    Path(name): Path<String>,
    Json(req): Json<IngestUrlRequest>,
) -> Result<(StatusCode, Json<IngestUrlResponse>), (StatusCode, Json<IngestUrlErrorResponse>)> {
    let name = tenant::scope(&tenant, &name);

    let schema = {
        let schemas = state.inner.schemas.read().await;
        schemas.get(&name).cloned().ok_or_else(|| {
            err(
                StatusCode::NOT_FOUND,
                "COLLECTION_NOT_FOUND",
                format!("Collection '{}' not found", tenant::display_name(&name)),
            )
        })?
    };

    // crawl4ai is opt-in. Unset endpoint -> clean 503, no panic.
    let client = Crawl4aiClient::from_env().ok_or_else(|| {
        err(
            StatusCode::SERVICE_UNAVAILABLE,
            "CRAWL_SERVICE_UNAVAILABLE",
            "Web ingestion requires crawl4ai. Start Veda with: docker compose --profile web up, and set VEDA_CRAWL4AI_ENDPOINT.".to_owned(),
        )
    })?;

    // Collect the seed URLs.
    let mut urls: Vec<String> = Vec::new();
    if let Some(u) = req.url.as_ref().filter(|s| !s.is_empty()) {
        urls.push(u.clone());
    }
    if let Some(list) = &req.urls {
        urls.extend(list.iter().filter(|s| !s.is_empty()).cloned());
    }
    if urls.is_empty() {
        return Err(err(
            StatusCode::BAD_REQUEST,
            "MISSING_URL",
            "Provide a 'url' string or a non-empty 'urls' array.".to_owned(),
        ));
    }

    let strategy = parse_chunking_strategy(req.chunker.as_deref(), req.chunk_size, req.chunk_overlap);

    let expires_at = req.ttl_secs.map(|ttl| {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;
        now + ttl as i64
    });

    let pages = client
        .crawl_many(&urls, req.deep_crawl, req.max_pages)
        .await
        .map_err(|e| {
            let status = match e.code {
                "CRAWL_UNAVAILABLE" => StatusCode::SERVICE_UNAVAILABLE,
                "CRAWL_EMPTY" => StatusCode::UNPROCESSABLE_ENTITY,
                _ => StatusCode::BAD_GATEWAY,
            };
            err(status, e.code, e.message)
        })?;

    let mut per_url = Vec::new();
    let mut total = 0usize;
    for page in pages {
        let source_url = if page.url.is_empty() {
            urls.first().cloned()
        } else {
            Some(page.url.clone())
        };
        let meta = SourceMeta {
            source_file: source_url.clone(),
            source_url,
            title: page.title.clone(),
            expires_at,
        };
        let result = ingest_markdown(&state, &name, &schema, &page.markdown, &meta, &strategy)
            .await
            .map_err(|(status, msg)| err(status, "INGEST_FAILED", msg))?;
        total += result.chunks_inserted;
        per_url.push(PerUrlResult {
            url: page.url,
            title: page.title,
            chunks_inserted: result.chunks_inserted,
        });
    }

    Ok((
        StatusCode::CREATED,
        Json(IngestUrlResponse {
            status: "ingested".to_owned(),
            pages: per_url.len(),
            chunks_inserted: total,
            per_url,
        }),
    ))
}

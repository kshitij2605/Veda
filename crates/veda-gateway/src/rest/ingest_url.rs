//! `POST /v1/collections/{name}/ingest-url` — crawl one or more URLs via
//! crawl4ai, clean to markdown, and run each page through the shared
//! [`super::ingest_common::ingest_markdown`] pipeline (chunk -> embed ->
//! insert). Tenant/RLS-scoped like `process-document`.
//!
//! Opt-in: when `VEDA_CRAWL4AI_ENDPOINT` is unset the route returns a
//! clean 503 "not configured" error rather than panicking.
//!
//! The crawl -> ingest core is factored into [`crawl_and_ingest`] so the
//! connectors framework (`/v1/admin/connectors/{id}/run`) can drive the
//! exact same path without duplicating it.

use axum::Json;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use serde::{Deserialize, Serialize};
use tracing::instrument;

use super::collections;
use super::crawl4ai::Crawl4aiClient;
use super::document::parse_chunking_strategy;
use super::ingest_common::{SourceMeta, ingest_markdown};
use super::state::AppState;
use super::url_guard::check_url_allowed;
use crate::tenant::{self, Tenant};
use veda_core::chunking::ChunkingStrategy;
use veda_core::schema::CollectionSchema;

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

fn err(
    status: StatusCode,
    code: &str,
    message: String,
) -> (StatusCode, Json<IngestUrlErrorResponse>) {
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

/// Default server cap on pages crawled for a deep crawl. The requested
/// `max_pages` is clamped to `min(requested, cap)`. Overridable via the
/// `VEDA_WEB_MAX_PAGES` env var.
const MAX_PAGES_CAP: u32 = 50;

/// Resolve the effective page cap from `VEDA_WEB_MAX_PAGES` (falling back
/// to [`MAX_PAGES_CAP`]) and clamp the caller's request to it. `None`
/// (no explicit request) is clamped to the cap so a deep crawl never
/// exceeds it even when the caller omitted `max_pages`.
fn clamp_max_pages(requested: Option<u32>) -> Option<u32> {
    let cap = std::env::var("VEDA_WEB_MAX_PAGES")
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(MAX_PAGES_CAP);
    Some(requested.unwrap_or(cap).min(cap))
}

/// Outcome of a crawl -> ingest run, shared by the `/ingest-url` handler
/// and the connector `/run` action.
#[derive(Debug, Default)]
pub struct CrawlIngestOutcome {
    pub pages: usize,
    pub chunks_inserted: usize,
    pub per_url: Vec<PerUrlResult>,
}

/// Error from a crawl -> ingest run, carrying a status + machine code +
/// message so each caller can map it into its own error envelope.
#[derive(Debug)]
pub struct CrawlIngestError {
    pub status: StatusCode,
    pub code: &'static str,
    pub message: String,
}

/// Crawl `urls` via crawl4ai and ingest each resulting page into the
/// already-scoped `collection_name` through [`ingest_markdown`]. This is
/// the shared core: both `/ingest-url` and the connectors `/run` action
/// call it so the crawl + chunk/embed/insert logic lives in one place.
///
/// `collection_name` must already be tenant-scoped by the caller; `schema`
/// is that collection's schema. Returns a clean 503 when crawl4ai is
/// unconfigured.
#[allow(clippy::too_many_arguments)]
pub async fn crawl_and_ingest(
    state: &AppState,
    tenant: &crate::tenant::TenantId,
    collection_name: &str,
    schema: &CollectionSchema,
    urls: &[String],
    deep_crawl: bool,
    max_pages: Option<u32>,
    strategy: &ChunkingStrategy,
    expires_at: Option<i64>,
) -> Result<CrawlIngestOutcome, CrawlIngestError> {
    if urls.is_empty() {
        return Err(CrawlIngestError {
            status: StatusCode::BAD_REQUEST,
            code: "MISSING_URL",
            message: "Provide a 'url' string or a non-empty 'urls' array.".to_owned(),
        });
    }

    // SSRF guard: validate every seed URL BEFORE contacting crawl4ai, so a
    // blocked (internal/private) URL is rejected with a clean error and
    // never fetched. Defense-in-depth alongside running crawl4ai in a
    // network-isolated container.
    for url in urls {
        check_url_allowed(url).map_err(|e| CrawlIngestError {
            status: StatusCode::FORBIDDEN,
            code: e.code(),
            message: e.to_string(),
        })?;
    }

    // Server-enforced page cap for deep crawls: clamp the requested
    // `max_pages` to `MAX_PAGES_CAP` (overridable via VEDA_WEB_MAX_PAGES) so
    // a request cannot drive an unbounded crawl regardless of what the
    // caller asked for.
    let max_pages = clamp_max_pages(max_pages);

    // crawl4ai is opt-in. Unset endpoint -> clean 503, no panic.
    let client = Crawl4aiClient::from_env().ok_or_else(|| CrawlIngestError {
        status: StatusCode::SERVICE_UNAVAILABLE,
        code: "CRAWL_SERVICE_UNAVAILABLE",
        message: "Web ingestion requires crawl4ai. Start Veda with: docker compose --profile web up, and set VEDA_CRAWL4AI_ENDPOINT.".to_owned(),
    })?;

    let pages = client
        .crawl_many(urls, deep_crawl, max_pages)
        .await
        .map_err(|e| {
            let status = match e.code {
                "CRAWL_UNAVAILABLE" => StatusCode::SERVICE_UNAVAILABLE,
                "CRAWL_EMPTY" => StatusCode::UNPROCESSABLE_ENTITY,
                _ => StatusCode::BAD_GATEWAY,
            };
            CrawlIngestError {
                status,
                code: e.code,
                message: e.message,
            }
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
            tenant: tenant.clone(),
        };
        let result = ingest_markdown(
            state,
            collection_name,
            schema,
            &page.markdown,
            &meta,
            strategy,
        )
        .await
        .map_err(|(status, msg)| CrawlIngestError {
            status,
            code: "INGEST_FAILED",
            message: msg,
        })?;
        total += result.chunks_inserted;
        per_url.push(PerUrlResult {
            url: page.url,
            title: page.title,
            chunks_inserted: result.chunks_inserted,
        });
    }

    Ok(CrawlIngestOutcome {
        pages: per_url.len(),
        chunks_inserted: total,
        per_url,
    })
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
    if let Err((status, Json(e))) =
        collections::reject_if_external(&schema, tenant::display_name(&name))
    {
        return Err(err(status, "EXTERNAL_READ_ONLY", e.error));
    }

    // Collect the seed URLs.
    let mut urls: Vec<String> = Vec::new();
    if let Some(u) = req.url.as_ref().filter(|s| !s.is_empty()) {
        urls.push(u.clone());
    }
    if let Some(list) = &req.urls {
        urls.extend(list.iter().filter(|s| !s.is_empty()).cloned());
    }

    let strategy =
        parse_chunking_strategy(req.chunker.as_deref(), req.chunk_size, req.chunk_overlap);

    let expires_at = req.ttl_secs.map(|ttl| {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;
        now + ttl as i64
    });

    let outcome = crawl_and_ingest(
        &state,
        &tenant,
        &name,
        &schema,
        &urls,
        req.deep_crawl,
        req.max_pages,
        &strategy,
        expires_at,
    )
    .await
    .map_err(|e| err(e.status, e.code, e.message))?;

    Ok((
        StatusCode::CREATED,
        Json(IngestUrlResponse {
            status: "ingested".to_owned(),
            pages: outcome.pages,
            chunks_inserted: outcome.chunks_inserted,
            per_url: outcome.per_url,
        }),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clamps_excessive_request_to_default_cap() {
        // No VEDA_WEB_MAX_PAGES override → cap is MAX_PAGES_CAP (50).
        // A wildly large request is clamped down to the cap.
        assert_eq!(clamp_max_pages(Some(1000)), Some(MAX_PAGES_CAP));
    }

    #[test]
    fn keeps_request_below_cap() {
        assert_eq!(clamp_max_pages(Some(5)), Some(5));
    }

    #[test]
    fn defaults_unspecified_to_cap() {
        assert_eq!(clamp_max_pages(None), Some(MAX_PAGES_CAP));
    }
}

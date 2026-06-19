//! Client for the crawl4ai web-crawling service (Docker
//! `unclecode/crawl4ai`). Mirrors the FlashParse OCR client pattern:
//! an async `reqwest::Client`, a generous timeout (crawls are slow),
//! W3C trace-context propagation, and an optional bearer token.
//!
//! Configured from the environment:
//!   - `VEDA_CRAWL4AI_ENDPOINT` — base URL of the crawl4ai service
//!     (e.g. `http://crawl4ai:11235`). When unset the client is `None`
//!     and callers return a clean "not configured" error.
//!   - `VEDA_CRAWL4AI_TOKEN` — optional bearer token.
//!
//! crawl4ai's `POST {endpoint}/crawl` accepts `{ "urls": [...] }` and
//! returns `{ "results": [{ "url", "markdown", "metadata": { "title" } }] }`.
//! The `markdown` field can itself be either a string or an object with a
//! `raw_markdown` / `fit_markdown` sub-field depending on version, so the
//! parser is defensive and extracts whichever string it finds.

use std::time::Duration;

/// One crawled page reduced to clean markdown.
#[derive(Debug, Clone)]
pub struct CrawledPage {
    pub url: String,
    pub title: Option<String>,
    pub markdown: String,
}

/// Error from a crawl attempt. Carries a short machine code plus a
/// human message so the REST layer can map it to a status + payload.
#[derive(Debug)]
pub struct CrawlError {
    pub code: &'static str,
    pub message: String,
}

impl CrawlError {
    fn new(code: &'static str, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }
}

/// Async client for crawl4ai. Holds the resolved endpoint + optional
/// bearer token and a shared `reqwest::Client`.
#[derive(Clone)]
pub struct Crawl4aiClient {
    endpoint: String,
    token: Option<String>,
    http: reqwest::Client,
}

impl Crawl4aiClient {
    /// Build a client for `endpoint` (no trailing slash required).
    pub fn new(endpoint: impl Into<String>, token: Option<String>) -> Self {
        Self {
            endpoint: endpoint.into().trim_end_matches('/').to_owned(),
            token,
            http: reqwest::Client::new(),
        }
    }

    /// Build a client from `VEDA_CRAWL4AI_ENDPOINT` (+ optional
    /// `VEDA_CRAWL4AI_TOKEN`). Returns `None` when the endpoint env var
    /// is unset or empty — the connector is opt-in.
    pub fn from_env() -> Option<Self> {
        let endpoint = std::env::var("VEDA_CRAWL4AI_ENDPOINT")
            .ok()
            .filter(|s| !s.is_empty())?;
        let token = std::env::var("VEDA_CRAWL4AI_TOKEN")
            .ok()
            .filter(|s| !s.is_empty());
        Some(Self::new(endpoint, token))
    }

    /// Crawl a single URL and return its cleaned markdown.
    pub async fn crawl(&self, url: &str) -> Result<CrawledPage, CrawlError> {
        let mut pages = self.crawl_many(&[url.to_owned()], false, None).await?;
        pages.pop().ok_or_else(|| {
            CrawlError::new("CRAWL_EMPTY", format!("crawl4ai returned no result for {url}"))
        })
    }

    /// Crawl one or more URLs. When `deep_crawl` is set, ask crawl4ai to
    /// follow links up to `max_pages` (best-effort; crawl4ai caps it).
    /// Returns one [`CrawledPage`] per result in the response.
    pub async fn crawl_many(
        &self,
        urls: &[String],
        deep_crawl: bool,
        max_pages: Option<u32>,
    ) -> Result<Vec<CrawledPage>, CrawlError> {
        if urls.is_empty() {
            return Ok(Vec::new());
        }

        let mut body = serde_json::json!({ "urls": urls });
        if deep_crawl
            && let Some(obj) = body.as_object_mut()
        {
            // crawl4ai's deep-crawl knobs live under crawler_config; pass
            // the page cap through if given. Unknown fields are ignored by
            // older server builds, so this is safe to always send.
            obj.insert(
                "crawler_config".to_owned(),
                serde_json::json!({
                    "deep_crawl_strategy": "bfs",
                    "max_pages": max_pages.unwrap_or(10),
                }),
            );
        }

        let url = format!("{}/crawl", self.endpoint);
        let mut trace_headers = reqwest::header::HeaderMap::new();
        veda_core::trace_propagation::inject_current_context(&mut trace_headers);

        let mut req = self
            .http
            .post(&url)
            .headers(trace_headers)
            .json(&body)
            // Crawls are slow (network + JS render). Be generous.
            .timeout(Duration::from_secs(180));
        if let Some(tok) = &self.token {
            req = req.bearer_auth(tok);
        }

        let resp = req.send().await.map_err(|e| {
            CrawlError::new("CRAWL_UNAVAILABLE", format!("crawl4ai request failed: {e}"))
        })?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(CrawlError::new(
                "CRAWL_FAILED",
                format!("crawl4ai returned {status}: {text}"),
            ));
        }

        let json: serde_json::Value = resp.json().await.map_err(|e| {
            CrawlError::new("CRAWL_PARSE", format!("crawl4ai response parse failed: {e}"))
        })?;

        let pages = parse_crawl_response(&json);
        if pages.is_empty() {
            return Err(CrawlError::new(
                "CRAWL_EMPTY",
                "crawl4ai returned no markdown for the requested URL(s)".to_owned(),
            ));
        }
        Ok(pages)
    }
}

/// Defensively extract `CrawledPage`s from a crawl4ai response.
///
/// Handles the documented shape `{ "results": [ {url, markdown, metadata} ] }`
/// as well as a bare top-level result object and a top-level array. The
/// `markdown` value may be a string or an object containing
/// `raw_markdown` / `fit_markdown` / `markdown`.
fn parse_crawl_response(json: &serde_json::Value) -> Vec<CrawledPage> {
    let results: Vec<&serde_json::Value> = if let Some(arr) = json.get("results").and_then(|v| v.as_array()) {
        arr.iter().collect()
    } else if let Some(arr) = json.as_array() {
        arr.iter().collect()
    } else {
        vec![json]
    };

    let mut pages = Vec::new();
    for r in results {
        let Some(md) = extract_markdown(r) else {
            continue;
        };
        if md.trim().is_empty() {
            continue;
        }
        let url = r
            .get("url")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_owned();
        let title = r
            .get("metadata")
            .and_then(|m| m.get("title"))
            .or_else(|| r.get("title"))
            .and_then(|v| v.as_str())
            .map(|s| s.to_owned());
        pages.push(CrawledPage { url, title, markdown: md });
    }
    pages
}

/// Pull a markdown string out of a single result value, tolerating both
/// `"markdown": "..."` and `"markdown": { "raw_markdown": "..." }`.
fn extract_markdown(result: &serde_json::Value) -> Option<String> {
    let md = result.get("markdown")?;
    if let Some(s) = md.as_str() {
        return Some(s.to_owned());
    }
    if let Some(obj) = md.as_object() {
        for key in ["fit_markdown", "raw_markdown", "markdown"] {
            if let Some(s) = obj.get(key).and_then(|v| v.as_str()) {
                return Some(s.to_owned());
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_results_array_with_string_markdown() {
        let json = serde_json::json!({
            "results": [
                { "url": "https://a.com", "markdown": "# A\n\nbody", "metadata": { "title": "A" } }
            ]
        });
        let pages = parse_crawl_response(&json);
        assert_eq!(pages.len(), 1);
        assert_eq!(pages[0].url, "https://a.com");
        assert_eq!(pages[0].title.as_deref(), Some("A"));
        assert_eq!(pages[0].markdown, "# A\n\nbody");
    }

    #[test]
    fn parses_object_markdown_with_raw_field() {
        let json = serde_json::json!({
            "results": [
                { "url": "https://b.com", "markdown": { "raw_markdown": "raw body" } }
            ]
        });
        let pages = parse_crawl_response(&json);
        assert_eq!(pages.len(), 1);
        assert_eq!(pages[0].markdown, "raw body");
        assert!(pages[0].title.is_none());
    }

    #[test]
    fn skips_empty_markdown() {
        let json = serde_json::json!({ "results": [ { "url": "x", "markdown": "   " } ] });
        assert!(parse_crawl_response(&json).is_empty());
    }
}

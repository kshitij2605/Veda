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
//! crawl4ai's synchronous "URL → clean markdown" endpoint is
//! `POST {endpoint}/md`, which accepts a single `{ "url": "...", "f": "fit" }`
//! (`f` selects the markdown filter; `"fit"` = LLM-ready filtered markdown)
//! and returns a top-level `{ "url", "markdown": "<string>", "success": true }`.
//! The `markdown` field is normally a string but may, on some builds, be an
//! object with a `fit_markdown` / `raw_markdown` / `markdown` sub-field, so the
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

    /// Crawl a single URL and return its cleaned markdown via `POST /md`.
    pub async fn crawl(&self, url: &str) -> Result<CrawledPage, CrawlError> {
        let body = serde_json::json!({ "url": url, "f": "fit" });

        let endpoint = format!("{}/md", self.endpoint);
        let mut trace_headers = reqwest::header::HeaderMap::new();
        veda_core::trace_propagation::inject_current_context(&mut trace_headers);

        let mut req = self
            .http
            .post(&endpoint)
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

        parse_md_response(&json, url).ok_or_else(|| {
            CrawlError::new(
                "CRAWL_EMPTY",
                format!("crawl4ai returned no markdown for {url}"),
            )
        })
    }

    /// Crawl one or more URLs, one [`CrawledPage`] per URL.
    ///
    /// crawl4ai's per-URL `/md` endpoint returns a single page; recursive
    /// link-following (deep crawl) is served by the async `/crawl/job` API,
    /// which is a documented follow-up and NOT wired up here. When
    /// `deep_crawl` is requested we still fetch each provided URL via `/md`
    /// (single page each) — `deep_crawl` / `max_pages` are accepted for API
    /// compatibility but intentionally ignored.
    pub async fn crawl_many(
        &self,
        urls: &[String],
        _deep_crawl: bool,
        _max_pages: Option<u32>,
    ) -> Result<Vec<CrawledPage>, CrawlError> {
        if urls.is_empty() {
            return Ok(Vec::new());
        }

        let mut pages = Vec::with_capacity(urls.len());
        for url in urls {
            pages.push(self.crawl(url).await?);
        }
        Ok(pages)
    }
}

/// Defensively extract a [`CrawledPage`] from a crawl4ai `/md` response.
///
/// The documented shape is a top-level object
/// `{ "url", "markdown": "<string>", "success": true, ... }`. The
/// `markdown` value may be a string or an object containing
/// `fit_markdown` / `raw_markdown` / `markdown`. The request `url` is used
/// as a fallback when the response omits one. Returns `None` when no
/// non-empty markdown is present.
fn parse_md_response(json: &serde_json::Value, request_url: &str) -> Option<CrawledPage> {
    let md = extract_markdown(json)?;
    if md.trim().is_empty() {
        return None;
    }
    let url = json
        .get("url")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .unwrap_or(request_url)
        .to_owned();
    let title = json
        .get("metadata")
        .and_then(|m| m.get("title"))
        .or_else(|| json.get("title"))
        .and_then(|v| v.as_str())
        .map(|s| s.to_owned());
    Some(CrawledPage { url, title, markdown: md })
}

/// Pull a markdown string out of a `/md` response value, tolerating both
/// a top-level `"markdown": "..."` string and
/// `"markdown": { "fit_markdown": "..." }`.
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
    fn parses_md_response_with_string_markdown() {
        let json = serde_json::json!({
            "url": "https://a.com",
            "markdown": "# A\n\nbody",
            "success": true
        });
        let page = parse_md_response(&json, "https://a.com").unwrap();
        assert_eq!(page.url, "https://a.com");
        assert_eq!(page.markdown, "# A\n\nbody");
    }

    #[test]
    fn parses_md_response_title_when_present() {
        let json = serde_json::json!({
            "url": "https://a.com",
            "markdown": "# A\n\nbody",
            "title": "A"
        });
        let page = parse_md_response(&json, "https://a.com").unwrap();
        assert_eq!(page.title.as_deref(), Some("A"));
    }

    #[test]
    fn parses_object_markdown_with_fit_field() {
        let json = serde_json::json!({
            "url": "https://b.com",
            "markdown": { "fit_markdown": "fit body", "raw_markdown": "raw body" }
        });
        let page = parse_md_response(&json, "https://b.com").unwrap();
        assert_eq!(page.markdown, "fit body");
        assert!(page.title.is_none());
    }

    #[test]
    fn falls_back_to_request_url_when_response_omits_it() {
        let json = serde_json::json!({ "markdown": "body" });
        let page = parse_md_response(&json, "https://req.example").unwrap();
        assert_eq!(page.url, "https://req.example");
    }

    #[test]
    fn skips_empty_markdown() {
        let json = serde_json::json!({ "url": "x", "markdown": "   " });
        assert!(parse_md_response(&json, "x").is_none());
    }
}

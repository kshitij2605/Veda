//! Client for a searxng metasearch instance (Docker
//! `searxng/searxng`). Mirrors the crawl4ai client pattern: an async
//! `reqwest::Client`, a timeout, W3C trace-context propagation, and an
//! optional bearer token.
//!
//! Configured from the environment:
//!   - `VEDA_SEARXNG_ENDPOINT` — base URL of the searxng service
//!     (e.g. `http://searxng:8080`). When unset the client is `None`
//!     and callers return a clean "not configured" error.
//!   - `VEDA_SEARXNG_TOKEN` — optional bearer token.
//!
//! searxng's JSON API is `GET {endpoint}/search?q=<query>&format=json`
//! and returns `{ "results": [{ "title", "url", "content", "engine" }] }`.
//! The `format=json` output must be enabled in the instance's
//! `settings.yml` (`search.formats: [html, json]`). The parser is
//! defensive: missing fields are tolerated and the result list is capped
//! to the requested `limit`.

use std::time::Duration;

/// One web search result reduced to title / url / snippet (+ engine).
#[derive(Debug, Clone, PartialEq)]
pub struct WebResult {
    pub title: String,
    pub url: String,
    pub snippet: String,
    pub engine: Option<String>,
}

/// Error from a search attempt. Carries a short machine code plus a
/// human message so callers can map it to a clean tool result.
#[derive(Debug)]
pub struct SearchError {
    pub code: &'static str,
    pub message: String,
}

impl SearchError {
    fn new(code: &'static str, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }
}

/// Async client for searxng. Holds the resolved endpoint + optional
/// bearer token and a shared `reqwest::Client`.
#[derive(Clone)]
pub struct SearxngClient {
    endpoint: String,
    token: Option<String>,
    http: reqwest::Client,
}

impl SearxngClient {
    /// Build a client for `endpoint` (no trailing slash required).
    pub fn new(endpoint: impl Into<String>, token: Option<String>) -> Self {
        Self {
            endpoint: endpoint.into().trim_end_matches('/').to_owned(),
            token,
            http: reqwest::Client::new(),
        }
    }

    /// Build a client from `VEDA_SEARXNG_ENDPOINT` (+ optional
    /// `VEDA_SEARXNG_TOKEN`). Returns `None` when the endpoint env var
    /// is unset or empty — the connector is opt-in.
    pub fn from_env() -> Option<Self> {
        let endpoint = std::env::var("VEDA_SEARXNG_ENDPOINT")
            .ok()
            .filter(|s| !s.is_empty())?;
        let token = std::env::var("VEDA_SEARXNG_TOKEN")
            .ok()
            .filter(|s| !s.is_empty());
        Some(Self::new(endpoint, token))
    }

    /// Run a web search and return up to `limit` results.
    pub async fn search(&self, query: &str, limit: usize) -> Result<Vec<WebResult>, SearchError> {
        let url = format!("{}/search", self.endpoint);
        let mut trace_headers = reqwest::header::HeaderMap::new();
        veda_core::trace_propagation::inject_current_context(&mut trace_headers);

        let mut req = self
            .http
            .get(&url)
            .headers(trace_headers)
            .query(&[("q", query), ("format", "json")])
            .timeout(Duration::from_secs(30));
        if let Some(tok) = &self.token {
            req = req.bearer_auth(tok);
        }

        let resp = req.send().await.map_err(|e| {
            SearchError::new("SEARCH_UNAVAILABLE", format!("searxng request failed: {e}"))
        })?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(SearchError::new(
                "SEARCH_FAILED",
                format!("searxng returned {status}: {text}"),
            ));
        }

        let json: serde_json::Value = resp.json().await.map_err(|e| {
            SearchError::new(
                "SEARCH_PARSE",
                format!("searxng response parse failed: {e}"),
            )
        })?;

        let mut results = parse_search_response(&json);
        results.truncate(limit);
        Ok(results)
    }
}

/// Defensively extract `WebResult`s from a searxng JSON response,
/// `{ "results": [ {title, url, content, engine} ] }`. Tolerates a
/// bare top-level array as well, and skips entries without a `url`.
fn parse_search_response(json: &serde_json::Value) -> Vec<WebResult> {
    let results: Vec<&serde_json::Value> =
        if let Some(arr) = json.get("results").and_then(|v| v.as_array()) {
            arr.iter().collect()
        } else if let Some(arr) = json.as_array() {
            arr.iter().collect()
        } else {
            Vec::new()
        };

    let mut out = Vec::new();
    for r in results {
        let Some(url) = r.get("url").and_then(|v| v.as_str()) else {
            continue;
        };
        if url.is_empty() {
            continue;
        }
        let title = r
            .get("title")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_owned();
        let snippet = r
            .get("content")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_owned();
        let engine = r
            .get("engine")
            .and_then(|v| v.as_str())
            .map(|s| s.to_owned());
        out.push(WebResult {
            title,
            url: url.to_owned(),
            snippet,
            engine,
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_results_array() {
        let json = serde_json::json!({
            "results": [
                { "title": "A", "url": "https://a.com", "content": "body a", "engine": "google" },
                { "title": "B", "url": "https://b.com", "content": "body b" }
            ]
        });
        let r = parse_search_response(&json);
        assert_eq!(r.len(), 2);
        assert_eq!(
            r[0],
            WebResult {
                title: "A".to_owned(),
                url: "https://a.com".to_owned(),
                snippet: "body a".to_owned(),
                engine: Some("google".to_owned()),
            }
        );
        assert_eq!(r[1].engine, None);
    }

    #[test]
    fn skips_entries_without_url() {
        let json = serde_json::json!({
            "results": [ { "title": "no url", "content": "x" }, { "url": "" } ]
        });
        assert!(parse_search_response(&json).is_empty());
    }

    #[test]
    fn non_object_response_is_empty() {
        assert!(parse_search_response(&serde_json::json!("nope")).is_empty());
    }
}

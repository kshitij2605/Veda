//! Agentic web-search tools for the ToolCatalog.
//!
//! Two read-only tools let an agent ground its answers on the live web:
//!
//!   * `web_search` — query a searxng metasearch instance
//!     ([`crate::rest::searxng::SearxngClient`]) and return a list of
//!     `{title, url, snippet}` results.
//!   * `fetch_url` — fetch a single URL as clean markdown via the
//!     crawl4ai client ([`crate::rest::crawl4ai::Crawl4aiClient`]),
//!     reusing the Phase-1 ingestion connector.
//!
//! Both are **opt-in**: their backing services are configured from the
//! environment (`VEDA_SEARXNG_ENDPOINT` / `VEDA_CRAWL4AI_ENDPOINT`). When
//! the relevant endpoint is unset the tool returns a clean
//! `{ "not_configured": true, "message": … }` result (not an error), so
//! an agent can explain the gap rather than crash the turn. Neither tool
//! is included in the default agent's auto-whitelist — web access is
//! granted explicitly (see the built-in `veda-web` agent).

use crate::agent::catalog::{ToolContext, ToolOutcome, ToolSpec};
use crate::rest::crawl4ai::Crawl4aiClient;
use crate::rest::searxng::SearxngClient;
use crate::rest::url_guard::check_url_allowed;

/// Default number of search results when `limit` is omitted.
const DEFAULT_SEARCH_LIMIT: usize = 5;
/// Cap on `limit` to keep tool payloads bounded.
const MAX_SEARCH_LIMIT: usize = 20;
/// Markdown is truncated to this many characters in `fetch_url` to keep
/// a single page from blowing the model's context window.
const MARKDOWN_CHAR_CAP: usize = 8000;

/// The opt-in web tool specs (id == OpenAI/MCP tool name). Appended to
/// the builtin catalog but excluded from the "empty whitelist == all"
/// expansion (see [`crate::agent::catalog::OPT_IN_BUILTINS`]).
pub fn web_specs() -> Vec<ToolSpec> {
    vec![
        ToolSpec {
            id: "web_search".to_owned(),
            name: "web_search".to_owned(),
            description: "Search the live web via a searxng metasearch instance. Returns a list of results, each with title, url and a short snippet. Use this to find sources, then `fetch_url` to read the most relevant ones. Requires VEDA_SEARXNG_ENDPOINT; returns not_configured when unset.".to_owned(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string", "description": "Search query" },
                    "limit": { "type": "integer", "minimum": 1, "maximum": MAX_SEARCH_LIMIT, "default": DEFAULT_SEARCH_LIMIT, "description": "Max results to return" }
                },
                "required": ["query"]
            }),
            kind: "builtin".to_owned(),
            read_only: true,
            destructive: false,
            title: Some("Web search".to_owned()),
        },
        ToolSpec {
            id: "fetch_url".to_owned(),
            name: "fetch_url".to_owned(),
            description: "Fetch a single web page and return its main content as clean markdown (title + body, truncated to a sane length). Use after `web_search` to read a result. Requires VEDA_CRAWL4AI_ENDPOINT; returns not_configured when unset.".to_owned(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "url": { "type": "string", "description": "Absolute URL of the page to fetch" }
                },
                "required": ["url"]
            }),
            kind: "builtin".to_owned(),
            read_only: true,
            destructive: false,
            title: Some("Fetch URL".to_owned()),
        },
    ]
}

// ── Implementations ─────────────────────────────────────────────────────────

/// A clean "service not configured" success result. Not an error: the
/// agent reads it and explains rather than failing the turn.
fn not_configured(message: &str) -> ToolOutcome {
    ToolOutcome::ok(serde_json::json!({ "not_configured": true, "message": message }))
}

pub async fn web_search(_ctx: &ToolContext<'_>, args: serde_json::Value) -> ToolOutcome {
    let Some(query) = args["query"].as_str().filter(|s| !s.trim().is_empty()) else {
        return ToolOutcome::invalid_args("query is required");
    };
    let limit = args["limit"]
        .as_u64()
        .map(|n| (n as usize).clamp(1, MAX_SEARCH_LIMIT))
        .unwrap_or(DEFAULT_SEARCH_LIMIT);

    let Some(client) = SearxngClient::from_env() else {
        return not_configured(
            "Web search requires searxng. Start Veda with: docker compose --profile web up, and set VEDA_SEARXNG_ENDPOINT.",
        );
    };

    match client.search(query, limit).await {
        Ok(results) => {
            let items: Vec<serde_json::Value> = results
                .into_iter()
                .map(
                    |r| serde_json::json!({ "title": r.title, "url": r.url, "snippet": r.snippet }),
                )
                .collect();
            ToolOutcome::ok(serde_json::json!({ "results": items, "count": items.len() }))
        }
        Err(e) => ToolOutcome::error(format!("web_search failed [{}]: {}", e.code, e.message)),
    }
}

pub async fn fetch_url(_ctx: &ToolContext<'_>, args: serde_json::Value) -> ToolOutcome {
    let Some(url) = args["url"].as_str().filter(|s| !s.trim().is_empty()) else {
        return ToolOutcome::invalid_args("url is required");
    };

    // SSRF guard: reject internal/private/non-http(s) URLs BEFORE fetching.
    // Returns a clean `{ blocked: true, reason }` outcome (not a hard
    // error) so the agent can explain rather than crash the turn.
    if let Err(e) = check_url_allowed(url) {
        return ToolOutcome::ok(serde_json::json!({
            "blocked": true,
            "reason": e.to_string(),
            "code": e.code(),
        }));
    }

    let Some(client) = Crawl4aiClient::from_env() else {
        return not_configured(
            "Fetching URLs requires crawl4ai. Start Veda with: docker compose --profile web up, and set VEDA_CRAWL4AI_ENDPOINT.",
        );
    };

    match client.crawl(url).await {
        Ok(page) => {
            let mut markdown = page.markdown;
            let truncated = markdown.len() > MARKDOWN_CHAR_CAP;
            if truncated {
                // Truncate on a char boundary at/below the cap.
                let mut cut = MARKDOWN_CHAR_CAP;
                while cut > 0 && !markdown.is_char_boundary(cut) {
                    cut -= 1;
                }
                markdown.truncate(cut);
            }
            ToolOutcome::ok(serde_json::json!({
                "url": page.url,
                "title": page.title,
                "markdown": markdown,
                "truncated": truncated,
            }))
        }
        Err(e) => ToolOutcome::error(format!("fetch_url failed [{}]: {}", e.code, e.message)),
    }
}

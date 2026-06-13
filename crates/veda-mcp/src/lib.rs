//! `veda-mcp` — a stdio↔HTTP bridge for Veda's MCP server.
//!
//! Veda exposes an MCP server over HTTP at `POST <url>/v1/mcp` (JSON-RPC 2.0,
//! optionally bearer-authenticated). Desktop MCP clients (Claude Desktop,
//! Cursor, VS Code) speak MCP over **stdio** instead. This crate bridges the
//! two: it reads newline-delimited JSON-RPC messages from stdin, forwards each
//! to the Veda HTTP endpoint, and writes responses back to stdout.
//!
//! The core forwarding logic lives in [`forward_message`], which is kept
//! free of any I/O on stdin/stdout so it can be unit-tested directly.

use reqwest::Client;

/// The MCP protocol version this bridge advertises to the server.
pub const MCP_PROTOCOL_VERSION: &str = "2025-06-18";

/// Returns `true` if `line` parses as a JSON-RPC **request** — i.e. an object
/// carrying a non-null `"id"` field. Notifications (no `id` / null `id`) and
/// non-objects return `false`.
///
/// Returns `None` when the line is not valid JSON at all (caller skips it).
pub fn is_request(line: &str) -> Option<bool> {
    let value: serde_json::Value = serde_json::from_str(line).ok()?;
    let has_id = value
        .get("id")
        .map(|id| !id.is_null())
        .unwrap_or(false);
    Some(has_id)
}

/// Builds a JSON-RPC error response (as a single-line string) echoing `id`,
/// used when the HTTP forward fails so the client gets a reply instead of a
/// hang. `id` is the raw JSON value of the request's `"id"` field.
pub fn error_response(id: &serde_json::Value, message: &str) -> String {
    let body = serde_json::json!({
        "jsonrpc": "2.0",
        "id": id,
        "error": {
            "code": -32000,
            "message": message,
        }
    });
    body.to_string()
}

/// Forward a single newline-delimited JSON-RPC `line` to the Veda MCP HTTP
/// `endpoint`.
///
/// - Malformed (non-JSON) lines are skipped → `None`.
/// - Notifications (no/null `id`) are POSTed but produce no reply → `None`.
/// - Requests (non-null `id`) return `Some(response_line)`: either the
///   server's response body, or a synthesized JSON-RPC error if the HTTP call
///   fails.
pub async fn forward_message(
    client: &Client,
    endpoint: &str,
    token: Option<&str>,
    line: &str,
) -> Option<String> {
    // Classify first so malformed input never reaches the network.
    let request = match is_request(line) {
        Some(r) => r,
        None => {
            eprintln!("veda-mcp: skipping malformed (non-JSON) line");
            return None;
        }
    };

    let mut builder = client
        .post(endpoint)
        .header("Content-Type", "application/json")
        .header("MCP-Protocol-Version", MCP_PROTOCOL_VERSION)
        .body(line.to_owned());
    if let Some(token) = token {
        builder = builder.bearer_auth(token);
    }

    let result = builder.send().await;

    if !request {
        // Notification: forwarded, but no response is written to stdout.
        if let Err(e) = result {
            eprintln!("veda-mcp: notification forward failed: {e}");
        }
        return None;
    }

    // Request: we must reply on stdout no matter what.
    let id = serde_json::from_str::<serde_json::Value>(line)
        .ok()
        .and_then(|v| v.get("id").cloned())
        .unwrap_or(serde_json::Value::Null);

    match result {
        Ok(resp) => match resp.text().await {
            Ok(text) => Some(text.replace(['\n', '\r'], " ")),
            Err(e) => Some(error_response(&id, &format!("failed to read response body: {e}"))),
        },
        Err(e) => Some(error_response(&id, &format!("HTTP request to Veda failed: {e}"))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener as StdTcpListener;

    #[test]
    fn classifies_request_with_id() {
        assert_eq!(
            is_request(r#"{"jsonrpc":"2.0","id":1,"method":"ping"}"#),
            Some(true)
        );
        assert_eq!(
            is_request(r#"{"jsonrpc":"2.0","id":"abc","method":"ping"}"#),
            Some(true)
        );
    }

    #[test]
    fn classifies_notification_without_id() {
        assert_eq!(
            is_request(r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#),
            Some(false)
        );
        assert_eq!(
            is_request(r#"{"jsonrpc":"2.0","id":null,"method":"x"}"#),
            Some(false)
        );
    }

    #[test]
    fn classifies_malformed_as_none() {
        assert_eq!(is_request("not json at all"), None);
        assert_eq!(is_request(""), None);
    }

    #[test]
    fn error_response_echoes_id() {
        let id = serde_json::json!(7);
        let out = error_response(&id, "boom");
        let v: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["id"], serde_json::json!(7));
        assert_eq!(v["jsonrpc"], "2.0");
        assert_eq!(v["error"]["message"], "boom");
    }

    /// A throwaway HTTP/1.1 mock that handles exactly one `POST /v1/mcp`.
    /// It captures the inbound request (headers + body) and replies with a
    /// canned JSON-RPC result echoing the request id. Returns the bound
    /// `http://127.0.0.1:PORT` base URL and a join handle yielding the raw
    /// request text the server received.
    fn spawn_mock() -> (String, std::thread::JoinHandle<String>) {
        let listener = StdTcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let handle = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            // Read until we have headers + the full body (Content-Length).
            let mut buf = Vec::new();
            let mut tmp = [0u8; 1024];
            loop {
                let n = stream.read(&mut tmp).unwrap();
                if n == 0 {
                    break;
                }
                buf.extend_from_slice(&tmp[..n]);
                let text = String::from_utf8_lossy(&buf);
                if let Some(hdr_end) = text.find("\r\n\r\n") {
                    let headers = &text[..hdr_end];
                    let body_start = hdr_end + 4;
                    let content_len = headers
                        .lines()
                        .find_map(|l| {
                            l.to_ascii_lowercase()
                                .strip_prefix("content-length:")
                                .map(|v| v.trim().parse::<usize>().unwrap_or(0))
                        })
                        .unwrap_or(0);
                    if buf.len() - body_start >= content_len {
                        break;
                    }
                }
            }
            let received = String::from_utf8_lossy(&buf).to_string();

            // Echo the request id back inside a JSON-RPC result.
            let body = String::from_utf8_lossy(&buf);
            let json_body = body.split("\r\n\r\n").nth(1).unwrap_or("");
            let id = serde_json::from_str::<serde_json::Value>(json_body)
                .ok()
                .and_then(|v| v.get("id").cloned())
                .unwrap_or(serde_json::Value::Null);
            let reply = serde_json::json!({
                "jsonrpc": "2.0",
                "id": id,
                "result": {"ok": true}
            })
            .to_string();
            let http = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                reply.len(),
                reply
            );
            stream.write_all(http.as_bytes()).unwrap();
            stream.flush().unwrap();
            received
        });
        (url, handle)
    }

    #[tokio::test]
    async fn request_returns_response_with_same_id() {
        let (url, handle) = spawn_mock();
        let endpoint = format!("{url}/v1/mcp");
        let client = Client::new();
        let line = r#"{"jsonrpc":"2.0","id":42,"method":"tools/list"}"#;

        let out = forward_message(&client, &endpoint, Some("secret-jwt"), line)
            .await
            .expect("request must yield a response");

        let v: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["id"], serde_json::json!(42));
        assert_eq!(v["result"]["ok"], true);

        let received = handle.join().unwrap();
        let low = received.to_ascii_lowercase();
        assert!(low.contains("authorization: bearer secret-jwt"));
        assert!(low.contains("mcp-protocol-version: 2025-06-18"));
    }

    #[tokio::test]
    async fn notification_returns_none_but_is_posted() {
        let (url, handle) = spawn_mock();
        let endpoint = format!("{url}/v1/mcp");
        let client = Client::new();
        let line = r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#;

        let out = forward_message(&client, &endpoint, None, line).await;
        assert!(out.is_none(), "notifications get no reply");

        let received = handle.join().unwrap();
        assert!(received.contains("notifications/initialized"));
        assert!(!received.to_ascii_lowercase().contains("authorization:"));
    }

    #[tokio::test]
    async fn malformed_line_returns_none_without_network() {
        // No mock server: a malformed line must short-circuit before any HTTP.
        let client = Client::new();
        let out = forward_message(&client, "http://127.0.0.1:1/v1/mcp", None, "}{ not json")
            .await;
        assert!(out.is_none());
    }

    #[tokio::test]
    async fn request_to_dead_endpoint_yields_error_response() {
        let client = Client::new();
        // Port 1 is unbindable/unreachable → connection error.
        let line = r#"{"jsonrpc":"2.0","id":"x1","method":"ping"}"#;
        let out = forward_message(&client, "http://127.0.0.1:1/v1/mcp", None, line)
            .await
            .expect("requests always get a reply, even on failure");
        let v: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["id"], serde_json::json!("x1"));
        assert!(v["error"]["message"].as_str().unwrap().contains("failed"));
    }
}

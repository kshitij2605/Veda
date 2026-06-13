//! `veda-mcp` binary: bridge MCP **stdio** to Veda's MCP **HTTP** endpoint.
//!
//! Reads newline-delimited JSON-RPC messages from stdin, forwards each to
//! `<url>/v1/mcp`, and writes server responses (for requests only) back to
//! stdout. Diagnostics go to stderr — stdout carries ONLY JSON-RPC messages.
//!
//! ## Claude Desktop config
//! ```json
//! { "mcpServers": { "veda": {
//!   "command": "veda-mcp",
//!   "args": ["--url", "http://localhost:6340"],
//!   "env": { "VEDA_MCP_TOKEN": "<jwt>" } } } }
//! ```

use clap::Parser;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use veda_mcp::forward_message;

#[derive(Parser, Debug)]
#[command(
    name = "veda-mcp",
    about = "Bridge MCP stdio clients to Veda's MCP HTTP endpoint"
)]
struct Args {
    /// Veda base URL (the `/v1/mcp` path is appended automatically).
    #[arg(long, env = "VEDA_MCP_URL")]
    url: String,

    /// Bearer token (JWT) for authenticated Veda servers. Omit for no-auth.
    #[arg(long, env = "VEDA_MCP_TOKEN")]
    token: Option<String>,
}

#[tokio::main]
async fn main() {
    let args = Args::parse();
    let endpoint = format!("{}/v1/mcp", args.url.trim_end_matches('/'));
    let token = args.token.as_deref();
    let client = reqwest::Client::new();

    eprintln!("veda-mcp: bridging stdio → {endpoint}");

    let stdin = tokio::io::stdin();
    let mut lines = BufReader::new(stdin).lines();
    let mut stdout = tokio::io::stdout();

    while let Ok(Some(line)) = lines.next_line().await {
        if line.trim().is_empty() {
            continue;
        }
        if let Some(response) = forward_message(&client, &endpoint, token, &line).await {
            if stdout.write_all(response.as_bytes()).await.is_err()
                || stdout.write_all(b"\n").await.is_err()
                || stdout.flush().await.is_err()
            {
                eprintln!("veda-mcp: stdout closed, exiting");
                break;
            }
        }
    }

    eprintln!("veda-mcp: stdin closed, exiting");
}

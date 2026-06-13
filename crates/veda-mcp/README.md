# veda-mcp

A stdio↔HTTP bridge that lets standard desktop MCP clients (Claude Desktop,
Cursor, VS Code) use Veda as an MCP server.

Veda exposes its MCP server over HTTP at `POST <url>/v1/mcp` (JSON-RPC 2.0,
optionally bearer-authenticated). Desktop MCP clients speak MCP over **stdio**.
`veda-mcp` reads newline-delimited JSON-RPC messages from stdin, forwards each
to Veda over HTTP (adding `MCP-Protocol-Version` and an optional
`Authorization: Bearer <token>` header), and writes responses back to stdout.

- **Requests** (messages with a non-null `id`) get the server's response on stdout.
- **Notifications** (no `id`) are forwarded but produce no stdout output.
- **Malformed lines** are skipped (logged to stderr); the loop never crashes.
- stdout carries **only** JSON-RPC messages; all diagnostics go to stderr.

## Usage

```sh
veda-mcp --url http://localhost:6340            # no-auth server
VEDA_MCP_TOKEN=<jwt> veda-mcp --url http://localhost:6340   # bearer auth
```

CLI flags also read from env: `--url` ← `VEDA_MCP_URL`, `--token` ← `VEDA_MCP_TOKEN`.

## Claude Desktop config

```json
{
  "mcpServers": {
    "veda": {
      "command": "veda-mcp",
      "args": ["--url", "http://localhost:6340"],
      "env": { "VEDA_MCP_TOKEN": "<jwt>" }
    }
  }
}
```

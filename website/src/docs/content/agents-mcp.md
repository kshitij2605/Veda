# Agents & MCP

Veda ships an agentic layer and a Model Context Protocol (MCP) server, so an LLM — or any MCP client like Claude Desktop, Cursor, or VS Code — can answer questions over your data using Veda's retrieval tools.

Because Veda is multimodal-native, its agent and MCP tools do **cross-modal retrieval**: a text query can return image, audio, video, or PDF hits, and an image query can return text hits, all through one unified embedding space. That is a retrieval tool no text-only vector database can offer. Every tool call also runs through Veda's JWT auth, tenant scoping, and row-level security.

## The built-in agent

A general-purpose agent, `veda-agent`, is always available with access to all tools — no setup required. Ask it a question and it runs a tool-calling loop: it calls the LLM, dispatches any tool calls (search, aggregate, …), feeds the results back, and repeats until it has an answer.

```bash
# one-shot against the built-in agent
curl -X POST http://localhost:6340/v1/agents/veda-agent/converse \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "input": "What index does Veda use for nearest-neighbor search? Search the docs collection."
  }'
```

The response includes the grounded answer, a `conversation_id`, and the full step trace:

```json
{
  "conversation_id": "…",
  "response": "Veda uses an HNSW graph index … the dashboard is Niriksha.",
  "steps": [
    { "kind": "tool_call",   "tool_id": "search", "args": {"collection": "docs", "query": "…"} },
    { "kind": "tool_result", "is_error": false,   "result": {"hits": [ … ]} },
    { "kind": "reasoning",   "text": "Based on the search results …" }
  ]
}
```

The agent's system prompt is augmented with a **modality-awareness block** listing the tenant's collections and which modalities each is searchable with — so the agent knows when it can search a non-text corpus.

The loop is bounded by `max_steps` (default 8) and truncates large tool results to keep context in check. `converse` returns `503` when no LLM is configured.

> Each run also returns `citations` + a `grounding` faithfulness score, there's a [streaming SSE variant](/docs/rag-aiops#streaming-converse-sse), opt-in [vector-backed memory](/docs/rag-aiops#vector-backed-agent-memory), and an [eval harness](/docs/rag-aiops#eval-harness-the-production-trace-flywheel) — see [RAG & AIOps Intelligence](/docs/rag-aiops).

## The tool catalog

Eight built-in tools back both the agent and the MCP server:

| Tool | Read-only | What it does |
|---|---|---|
| `search` | yes | Multimodal retrieval. Embeds a text **or** image query (`query` / `query_image_b64`) and runs distributed search; returns hits `{id, distance, fields}`. Cross-modal via the unified space; RLS owner filter injected. |
| `aggregate` | yes | OLAP aggregation (`count`/`sum`/`avg`/`min`/`max`, optional `group_by`) over the columnar store. |
| `filter_query` | yes | Structured record retrieval via the filter DSL; tenant + TTL + RLS enforced. |
| `get_record` | yes | Fetch one record by id (TTL-excluded, RLS-enforced). |
| `describe_collection` | yes | Schema + record count. |
| `list_collections` | yes | Tenant-scoped collection names. |
| `route` | yes | Adaptive retrieval-strategy recommendation. |
| `upsert_record` | no (destructive) | Insert one record, auto-embedding when the vector is omitted. |

The filter DSL supports dotted [JSON paths](/docs/search#json-path-filtering) (e.g. `meta.tier = "gold"`). Tools are also callable directly, outside any agent:

```bash
curl -X POST http://localhost:6340/v1/agent-tools/search/execute \
  -H "Content-Type: application/json" \
  -d '{"collection": "docs", "query": "hybrid search", "top_k": 3}'
```

> The `search` `mode` of `dense`/`auto` execute today; `sparse`/`hybrid`/`colpali` return a clear `invalid_args` until their query encoders are wired in. Cross-modal **image** queries require the multimodal embedder (`MULTIMODAL_EMBEDDING_*`) to be configured.

## Custom agents

An agent is a saved object you can create, update, and delete:

```json
{
  "id": "support-bot",
  "name": "Support Bot",
  "description": "Answers product questions from the docs.",
  "instructions": "You are a concise assistant. Ground answers in the docs collection.",
  "tool_ids": ["search", "aggregate"],
  "model": { "endpoint": null, "model": null }
}
```

- `tool_ids` is a whitelist of tools the agent may call; an empty list means **all** built-ins.
- `model` optionally overrides the global LLM endpoint/model for that agent.

```bash
curl -X POST http://localhost:6340/v1/agents \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{ "id": "support-bot", "instructions": "…", "tool_ids": ["search"] }'
```

Agents, custom tools, and conversations are all tenant-scoped saved objects. See the [REST API](/docs/rest-api#agents-tools-mcp) for the full endpoint list.

## MCP server

`/v1/mcp` is a JSON-RPC 2.0 MCP server (protocol `2025-06-18`) advertising the `tools` capability. Methods: `initialize`, `tools/list`, `tools/call`, `ping`.

- `tools/list` returns all eight tools with annotations (`readOnlyHint`, `destructiveHint`, `title`) and the multimodal `search` schema (advertising `query_image_b64` and `mode`).
- `tools/call` success returns a text block **and** `structuredContent` (the raw tool JSON). Tool-execution failures return a result with `isError: true` so the model can self-correct; only protocol errors (unknown method/tool, malformed) use JSON-RPC error codes.

### Connect a client over stdio

The `veda-mcp` bridge speaks stdio for Claude Desktop, Cursor, and VS Code. Point it at your Veda server and pass a JWT:

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

The bridge reads newline-delimited JSON-RPC from stdin, forwards each message to `<url>/v1/mcp` with `Authorization: Bearer <token>`, and writes responses to stdout (diagnostics go to stderr to keep the channel clean).

### Connect over Streamable HTTP

For `mcp-remote` and HTTP MCP clients, POST JSON-RPC directly to `/v1/mcp` with `Authorization: Bearer <jwt>`. The server is stateless (no session id), honors the `MCP-Protocol-Version` header (unsupported → `400`), and returns `405` on `GET` (there is no server-initiated stream).

```bash
curl -X POST http://localhost:6340/v1/mcp \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -H "MCP-Protocol-Version: 2025-06-18" \
  -d '{"jsonrpc": "2.0", "id": 1, "method": "tools/list"}'
```

## Agent-to-Agent (A2A)

Expose a Veda agent as a peer to external orchestrators:

- `GET /v1/a2a/{agent_id}/card` returns an A2A **AgentCard** — `name`, `description`, `version`, `url`, `capabilities` (`streaming: false`), default input/output modes, and `skills` derived one-to-one from the agent's tools.
- `POST /v1/a2a/{agent_id}` is JSON-RPC 2.0 with a single method, `message/send`: it concatenates the message's text parts, runs a one-shot agent turn, and returns an A2A message `{ role: "agent", parts: [{ kind: "text", text: "…" }], kind: "message" }` with the step trace under `metadata`. No LLM → `503`.

A2A is synchronous (`message/send`) only — there is no streaming.

## Configuration

| Variable | Purpose |
|---|---|
| `VEDA_LLM_ENDPOINT` | OpenAI-compatible `chat/completions` endpoint |
| `VEDA_LLM_MODEL` | Model name |
| `VEDA_LLM_API_KEY` | API key (optional for local models) |
| `VEDA_LLM_TIMEOUT_SECS` | Per-call timeout |
| `EMBEDDING_*` | Text embedding (used by the `search` tool) |
| `MULTIMODAL_EMBEDDING_*` | Image/video embedding — required for image queries |

Without an LLM, `/converse` and A2A `message/send` return `503`, but the tools and MCP `tools/call` still work — they don't need an LLM. When auth is unconfigured (single-node dev), requests run as admin and tools bypass RLS.

## Current limitations

- `search` executes `dense`/`auto` modes; `sparse`/`hybrid`/`colpali` are gated to `invalid_args` until their query encoders are wired in.
- Image queries require `MULTIMODAL_EMBEDDING_*` (text retrieval works with a text embedder alone). Converse image attachments are accepted but image-attachment-as-retrieval is deferred — use the `search` tool's `query_image_b64` for cross-modal queries today.
- MCP advertises only the `tools` capability (no resources/prompts/sampling), by design.
- Custom tools can be stored but are not yet dispatched by the catalog (built-in tools only today).

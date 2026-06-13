# Veda Agents & MCP — Functional Guide

**Date:** 2026-06-13
**Status:** Shipped on `main` (v0.2.4). Validated end-to-end against a live LLM
(Qwen3.5-122B) and embedder (BGE-M3).

This document describes Veda's agentic layer and its Model Context Protocol (MCP)
surface: what they are, how they're built, the full API, how external MCP/A2A clients
connect, and the known limitations.

---

## 1. Why this exists

Elastic shipped *Agent Builder* — conversational agents that answer questions over your
data using an LLM plus tools that query the store, exposed over MCP (and A2A). Its weak
point is that it is **text-centric**: non-text is handled only as opaque LLM attachments,
with no multimodal retrieval and no unified vector space.

Veda is multimodal-native. So Veda's agent and MCP tools can do **cross-modal
retrieval** — a text query can return image/audio/video/PDF hits, and an image query can
return text hits — through one unified embedding space. That is the differentiator: an
MCP client connected to Veda gets a retrieval tool no text-only vector DB can offer.

Veda is also **stronger on security** than the standalone Elastic MCP server: every tool
call goes through Veda's JWT auth, tenant scoping, and row-level security.

---

## 2. Architecture

All agent code lives in the gateway (`crates/veda-gateway/src/agent/`) because it
orchestrates REST-level operations and needs `AppState`.

```
                 ┌─────────────────────────────────────────────┐
   MCP clients   │  REST surface (axum, protected router)        │
   (stdio/HTTP)  │   /v1/agents/*   /v1/agent-tools/*            │
        │        │   /v1/conversations/*   /v1/a2a/*   /v1/mcp   │
        ▼        └───────────────┬───────────────┬──────────────┘
  veda-mcp bridge                │               │
   (stdio↔HTTP)                  ▼               ▼
                        ┌──────────────┐  ┌───────────────┐
                        │ AgentExecutor│  │  MCP server   │
                        │  (ReAct loop)│  │  (JSON-RPC)   │
                        └──────┬───────┘  └───────┬───────┘
                               │   both use       │
                               ▼                  ▼
                        ┌──────────────────────────────┐
                        │        ToolCatalog            │
                        │  8 builtins; dispatch(ctx,..) │
                        └──────┬─────────────┬──────────┘
                               ▼             ▼
                     search / embed    aggregate / filter /
                     (distributed)     get_record / route / ...
```

Key components:

- **`ToolCatalog`** (`agent/catalog.rs`) — the single registry of tools, used by **both**
  the agent executor and the MCP server. A `ToolSpec` is `{ id, name, description,
  input_schema, kind, read_only, destructive, title }`. `dispatch(ctx, tool_id, args)`
  runs a tool and returns a `ToolOutcome { content, is_error, error_kind }`
  (`error_kind ∈ {InvalidArgs, NotFound, Internal}` — transport-agnostic). `ToolContext`
  carries `&AppState`, the tenant, and the optional RLS `user`.
- **`AgentExecutor`** (`agent/executor.rs`) — `run(model, catalog, ctx, agent,
  system_prompt, history, input, max_steps)` runs the tool-calling loop: call the LLM
  with the agent's whitelisted tool specs → if it returns `tool_calls`, dispatch each via
  the catalog, append the results, loop; otherwise return the final answer. Bounded by
  `max_steps` (default 8) with tool-result truncation (~8 KB) to cap context blow-ups.
  Records every step.
- **LLM tool-calling** (`veda-core/src/llm_client.rs`) — `complete_with_tools` + a
  `ChatModel` trait (OpenAI-compatible `tools` / `tool_calls`). The executor is generic
  over `ChatModel`, so tests inject a fake (no network).
- **Persistence** — agents, custom tools, and conversations are stored as
  `SavedObject`s (`type ∈ {agent, agent_tool, agent_conversation}`), tenant-scoped.
- **Embedding** — `embed_text`/`embed_bytes` are async (`reqwest::Client`); the `search`
  tool auto-embeds text or image queries via the modality-appropriate model in the
  registry.

---

## 3. The tool catalog (8 built-in tools)

| Tool | read-only | Parameters | What it does |
|---|---|---|---|
| `search` | yes | `collection`, `query?` (text), `query_image_b64?` (base64 image), `mode` (`auto`/`dense`/`hybrid`/`sparse`/`colpali`, default `auto`), `top_k`, `field?`, `fields?` (projection), `filter?` | Multimodal retrieval. Embeds the text **or** image query and runs distributed search; returns hits `{id, distance, fields{…}}`. Cross-modal via the unified space. RLS owner filter injected. |
| `aggregate` | yes | `collection`, `field`, `metric` (`count`/`sum`/`avg`/`min`/`max`), `group_by?`, `filter?` | OLAP aggregation over the columnar store. |
| `filter_query` | yes | `collection`, `filter`, `limit?` | Structured record retrieval via the filter DSL; tenant + TTL + RLS enforced. |
| `get_record` | yes | `collection`, `id` | Fetch one record by id (TTL-excluded, RLS-enforced). |
| `describe_collection` | yes | `name` | Schema + record count. |
| `list_collections` | yes | — | Tenant-scoped collection names. |
| `route` | yes | `query` | Adaptive retrieval-strategy recommendation. |
| `upsert_record` | **no** (destructive) | `collection`, `record` | Insert one record (auto-embeds if the vector is omitted). |

- The same JSON-Schema `input_schema` is shown to the LLM (as an OpenAI tool def) and to
  MCP clients (as `inputSchema` + annotations). The MCP `search` entry advertises
  `query_image_b64` and `mode`, so external MCP clients get cross-modal retrieval.
- Tools are directly invocable outside an agent:
  `POST /v1/agent-tools/{id}/execute` with the args object (or `{ "args": {…} }`).
- **Filter DSL** supports dotted JSON paths (e.g. `meta.tier = "gold"`).

> **MVP note:** `search` `mode` of `sparse`/`hybrid`/`colpali` currently returns a clear
> `invalid_args` (only `dense`/`auto` execute) — these need the sparse/multi-vector query
> encoders wired in. Cross-modal **image** queries require the multimodal embedder
> (`MULTIMODAL_EMBEDDING_*`) to be configured.

---

## 4. Agents

An agent is:

```json
{
  "id": "support-bot",
  "name": "Support Bot",
  "description": "Answers product questions from the docs.",
  "instructions": "You are a concise assistant. Ground answers in the docs collection.",
  "tool_ids": ["search", "aggregate"],        // empty = all builtins
  "model": { "endpoint": null, "model": null } // optional per-agent LLM override
}
```

- A built-in **`veda-agent`** is always available (general-purpose, all tools) without
  creating anything — `converse` resolves it automatically when no stored agent has that
  id. It is in-memory unless you save one with the same id.
- `tool_ids` is a whitelist; an empty list means "all builtins".
- `model` overrides the global `VEDA_LLM_*` endpoint/model for that agent.

### Converse loop & trace

`POST /v1/agents/{id}/converse` runs the executor and returns:

```json
{
  "conversation_id": "…",
  "response": "Veda uses an HNSW graph index … the dashboard is Niriksha.",
  "steps": [
    { "kind": "tool_call",   "tool_call_id": "call_1", "tool_id": "search", "args": { "collection": "docs", "query": "…" } },
    { "kind": "tool_result", "tool_call_id": "call_1", "is_error": false, "result": { "hits": [ … ] } },
    { "kind": "reasoning",   "text": "Based on the search results …" }
  ]
}
```

The system prompt is the agent's `instructions` plus an auto-generated **modality
awareness** block listing the tenant's collections and which modalities each is
searchable with — so the agent knows it can search non-text corpora.

---

## 5. REST API reference

All endpoints are in the protected router (JWT auth + tenant + RLS apply).

**Agents**
- `GET /v1/agents` — list
- `POST /v1/agents` — create/upsert (empty `id` → generated)
- `GET|PUT|DELETE /v1/agents/{id}`
- `POST /v1/agents/{id}/converse` — body `ConverseRequest { input, conversation_id?, attachments? }` → `ConverseResponse`. `503` if no LLM; `404` unknown (non-default) agent.

**Tools**
- `GET /v1/agent-tools` / `POST /v1/agent-tools` — custom-tool CRUD list/create
- `GET|PUT|DELETE /v1/agent-tools/{id}`
- `GET /v1/agent-tools/catalog` — built-in tool specs (id, description, inputSchema, annotations)
- `POST /v1/agent-tools/{id}/execute` — run one tool directly; error mapping invalid→400, not-found→404, internal→500

**Conversations**
- `GET /v1/conversations` — list
- `GET|DELETE /v1/conversations/{id}`

### Example

```bash
# one-shot against the built-in agent
curl -X POST $BASE/v1/agents/veda-agent/converse \
  -H 'Content-Type: application/json' \
  -d '{"input":"What index does Veda use for nearest-neighbor search? Search the docs collection."}'

# run a tool directly
curl -X POST $BASE/v1/agent-tools/search/execute \
  -H 'Content-Type: application/json' \
  -d '{"collection":"docs","query":"hybrid search","top_k":3}'
```

---

## 6. MCP server

`/v1/mcp` is a JSON-RPC 2.0 MCP server, protocol **`2025-06-18`**, advertising the
`tools` capability. Methods: `initialize`, `tools/list`, `tools/call`, `ping`.

- **Full catalog:** `tools/list` returns all 8 tools with **annotations**
  (`readOnlyHint`, `destructiveHint`, `title`).
- **Results:** `tools/call` success returns a text block **and** `structuredContent`
  (the raw tool JSON). Tool-execution errors return a `CallToolResult` with
  **`isError: true`** so the model can self-correct; only protocol errors (unknown
  method/tool, malformed) use JSON-RPC error codes (`-32601`/`-32602`/…).

### Transports

The MCP message layer is reachable two standard ways:

**1. stdio — the `veda-mcp` bridge** (for Claude Desktop, Cursor, VS Code):

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

The bridge reads newline-delimited JSON-RPC from stdin, forwards each message to
`<url>/v1/mcp` with `Authorization: Bearer <token>` + `MCP-Protocol-Version`, and writes
responses to stdout (diagnostics go to stderr so the channel stays clean).

**2. Streamable HTTP** (for `mcp-remote` and HTTP MCP clients): POST JSON-RPC to
`/v1/mcp` with `Authorization: Bearer <jwt>`; the server is stateless (no
`Mcp-Session-Id`), honors the `MCP-Protocol-Version` header (unsupported → `400`), and
returns `405` on `GET` (no server-initiated stream).

---

## 7. A2A (Agent-to-Agent)

Expose a Veda agent as a peer to external orchestrators (sync subset, matching Elastic's
A2A):

- `GET /v1/a2a/{agent_id}/card` → an A2A **AgentCard**: `name`, `description`, `version`,
  `url`, `capabilities { streaming: false }`, `defaultInputModes`/`defaultOutputModes`,
  and `skills` derived one-to-one from the agent's tools.
- `POST /v1/a2a/{agent_id}` → JSON-RPC 2.0; the only method is `message/send`. It
  concatenates the user message's text parts, runs a one-shot agent turn, and returns an
  A2A Message `{ role: "agent", parts: [{ kind: "text", text: <answer> }], kind:
  "message" }` (with the step trace under `metadata`). Unknown method → `-32601`; no LLM
  → `503`.

---

## 8. Configuration

- **LLM:** `VEDA_LLM_ENDPOINT` (OpenAI-compatible `chat/completions`),
  `VEDA_LLM_MODEL`, `VEDA_LLM_API_KEY`, `VEDA_LLM_TIMEOUT_SECS`. Without an LLM,
  `/converse` and A2A `message/send` return `503` (the tools and MCP `tools/call` still
  work — they don't need an LLM).
- **Embedding:** `EMBEDDING_ENDPOINT` / `EMBEDDING_MODEL` / `EMBEDDING_DIMENSION` (text);
  `MULTIMODAL_EMBEDDING_*` (image/video — required for image queries).
- Auth: standard Veda JWT (`/v1/auth/login`); when auth is unconfigured (single-node
  dev), requests run as admin and tools bypass RLS.

---

## 9. Validation (real data)

Verified on iitgpu11 with the live BGE-M3 embedder and Qwen3.5-122B LLM:

1. `embed-insert` of 5 real docs → persisted (this is also the regression that proved the
   async-embedding-client fix — see §10).
2. MCP `initialize` → `2025-06-18`; `tools/list` → 8 tools with correct annotations and
   the multimodal `search` schema.
3. MCP `tools/call search` (real query embedding) → ranked the correct doc #1, returned
   content fields.
4. Agent `/converse` (real LLM tool-calling) → the agent called `search`, then answered
   correctly and grounded ("HNSW" + "Niriksha") with a full step trace.
5. `veda-mcp` stdio bridge → `initialize` + `tools/list` flow through.
6. A2A card (8 skills) + `message/send` → correct grounded answer.

---

## 10. Known limitations & follow-ups

- **`search` modes:** only `dense`/`auto` execute; `sparse`/`hybrid`/`colpali` are gated
  to `invalid_args` until the sparse/multi-vector query encoders are wired into the tool.
- **Image queries** need `MULTIMODAL_EMBEDDING_*` configured (text retrieval works with
  BGE-M3 alone).
- **Converse attachments** are accepted but image-attachment-as-retrieval is deferred
  (the cross-modal path is available now via the `search` tool's `query_image_b64`).
- **A2A** is sync (`message/send`) only — no streaming.
- **MCP** advertises only the `tools` capability (no resources/prompts/sampling) — by
  design for now.
- **Blocking-reqwest landmine:** `chunking.rs` and `splade_client.rs` still build
  `reqwest::blocking` clients (the same pattern that broke embedding) and must be made
  async before their live paths are exercised under the async server.
- **Custom `AgentTool` execution** is stored but not yet dispatched by the catalog
  (built-in tools only today).

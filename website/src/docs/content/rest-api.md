# REST API

All endpoints are served under `http://<host>:6340/v1`. Authenticated requests pass a JWT as `Authorization: Bearer <token>` (obtain one via `POST /v1/auth/login`). The full machine-readable spec lives at `documentation/openapi.yaml` in the repository.

## Collections

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/collections` | Create a collection (supports `seed_records` for atomic create-and-load, `use_log_primary` for on-disk records, `rls_owner_field` for [row-level security](/docs/collections#row-level-security); vector fields take `index`: `hnsw` · `diskann` · `ivf_pq` · `flat` · `rabitq` · `sq4` · `sq8`) |
| `GET` | `/v1/collections` | List collections |
| `GET` | `/v1/collections/{name}` | Get a collection's schema and stats |
| `DELETE` | `/v1/collections/{name}` | Drop a collection |
| `POST` | `/v1/collections/{name}/reindex` | Re-chunk / re-embed into a sibling and atomically swap the alias |

## Records & search

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/collections/{name}/insert` | Insert records (flat field→value maps) |
| `POST` | `/v1/collections/{name}/embed-insert` | Insert with server-side auto-embedding (`auto_embed` map) |
| `GET` | `/v1/collections/{name}/records` | List records (paginated; supports `?at=<ts>` snapshot reads) |
| `GET` | `/v1/collections/{name}/records/{id}` | Fetch one record |
| `POST` | `/v1/collections/{name}/search` | Search: `vector`, `bm25`, `sparse_vector`, `hybrid`, or `multi_vector` queries + `filter`, `rerank`, `explain`, `?at=<ts>` |
| `GET` | `/v1/collections/{name}/coverage` | Field coverage statistics |

### Search request shape

```json
{
  "query": {
    "vector": { "field": "embedding", "values": [0.15, 0.22] }
  },
  "limit": 10,
  "filter": "category = 'electronics' AND price < 500",
  "rerank": false,
  "explain": true
}
```

See [Search & fusion](/docs/search) for `multi_vector` fusion, hybrid queries, and snapshot reads.

## Analytics

Read-only OLAP-style aggregations over the columnar store. All three honor the filter DSL (including [JSON-path filters](/docs/search#json-path-filtering)) and row-level security.

| Method | Path | Description |
|---|---|---|
| `GET` | `/v1/collections/{name}/aggregate` | `?field=&metric={count\|sum\|avg\|min\|max}&group_by=&filter=` — scalar metric, optionally bucketed per group |
| `GET` | `/v1/collections/{name}/histogram` | `?field=&bins=&filter=` — equal-width numeric histogram (default 10 bins) |
| `GET` | `/v1/collections/{name}/field-stats` | `?field=&filter=` — total/null counts, cardinality, numeric or string-length summaries |

```bash
# Average price per category, electronics under $500 only
curl "http://localhost:6340/v1/collections/products/aggregate?field=price&metric=avg&group_by=category&filter=price%20%3C%20500"
```

## Scheduled jobs

Admin-only cron scheduler. A job is a cron expression plus a typed kind — `ttl_sweep`, `reindex`, `compact`, or `evaluate`, each scoped to a collection — deliberately not arbitrary code. Jobs persist in the data directory and reload on restart. (`ttl_sweep` is fully wired today; the other kinds are registered hooks.)

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/admin/jobs` | Register or overwrite a job (`400` on an invalid cron expression) |
| `GET` | `/v1/admin/jobs` | List registered jobs |
| `DELETE` | `/v1/admin/jobs/{name}` | Remove a job |

```bash
# Sweep expired records from the cache collection every 5 minutes
curl -X POST http://localhost:6340/v1/admin/jobs \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "name": "cache-ttl-sweep",
    "cron": "0 */5 * * * *",
    "kind": {"kind": "ttl_sweep", "collection": "cache"}
  }'
```

## Documents

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/collections/{name}/process-document` | OCR → chunk → embed → insert one file (multipart `file`) |
| `POST` | `/v1/collections/{name}/process-documents` | Batch variant (multipart `files`) |
| `POST` | `/v1/chunk/preview` | Preview how text would be chunked under a strategy |

## Embedding & models

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/embed` | Embed text (or multimodal input) directly; optional `model` override |
| `POST` | `/v1/llm/complete` | LLM completion passthrough (used by Niriksha features) |
| `POST` | `/v1/route` | Modality routing helper |

## Agents, tools & MCP

The agentic layer and MCP surface. See [Agents & MCP](/docs/agents-mcp) for the full guide.

**Agents**

| Method | Path | Description |
|---|---|---|
| `GET` | `/v1/agents` | List agents |
| `POST` | `/v1/agents` | Create or upsert an agent (empty `id` → generated) |
| `GET/PUT/DELETE` | `/v1/agents/{id}` | Manage one agent |
| `POST` | `/v1/agents/{id}/converse` | Run a tool-calling turn — body `{ input, conversation_id?, attachments? }`; `503` without an LLM, `404` for an unknown non-default agent |

**Tools**

| Method | Path | Description |
|---|---|---|
| `GET/POST` | `/v1/agent-tools` | List / create custom tools |
| `GET/PUT/DELETE` | `/v1/agent-tools/{id}` | Manage one custom tool |
| `GET` | `/v1/agent-tools/catalog` | Built-in tool specs (id, description, `inputSchema`, annotations) |
| `POST` | `/v1/agent-tools/{id}/execute` | Run one tool directly (invalid→400, not-found→404, internal→500) |

**Conversations**

| Method | Path | Description |
|---|---|---|
| `GET` | `/v1/conversations` | List conversations |
| `GET/DELETE` | `/v1/conversations/{id}` | Fetch / delete one conversation |

**MCP & A2A**

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/mcp` | JSON-RPC 2.0 MCP server (protocol `2025-06-18`): `initialize`, `tools/list`, `tools/call`, `ping`. `GET` → `405` |
| `GET` | `/v1/a2a/{agent_id}/card` | A2A AgentCard (skills derived from the agent's tools) |
| `POST` | `/v1/a2a/{agent_id}` | A2A JSON-RPC `message/send` — one-shot agent turn; `503` without an LLM |

## Transactions

Percolator-style snapshot-isolation transactions:

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/txn` | Begin / commit / abort transactional operations |

Write skew is allowed (snapshot isolation); lost updates are prevented; read-only transactions are consistent.

## Auth & saved objects

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/auth/login` | Exchange username/password for a JWT (Argon2id verification) |
| `GET/POST` | `/v1/saved_objects` | Saved objects used by the Niriksha dashboard |
| `GET/PUT/DELETE` | `/v1/saved_objects/{id}` | Manage one saved object |

## Admin

| Method | Path | Description |
|---|---|---|
| `GET` | `/v1/admin/health` | Liveness — always 200, with per-dependency checks |
| `GET` | `/v1/admin/readiness` | Readiness — 503 when unhealthy |
| `GET` | `/v1/admin/metrics` | Prometheus text-format metrics |
| `GET` | `/v1/admin/nodes` | Node list; `mode` reports `"single"` or `"raft"` |
| `GET/PUT` | `/v1/admin/region_topology` | Cross-region replication topology (writes require Raft) |
| `POST` | `/v1/admin/region_topology/promote` | Promote a replica region (Raft only) |
| `GET/POST` | `/v1/admin/replica/safe_ts` | Replica read watermark |
| `POST` | `/v1/admin/cluster/join` · `DELETE /v1/admin/cluster/nodes/{id}` | Raft membership changes |
| `POST` | `/v1/admin/collections/{name}/mvcc_checkpoint` | Atomic MVCC snapshot + WAL truncate |

Endpoints marked Raft-only return `503` in single-node mode — see [Cluster modes](/docs/installation#cluster-modes).

## Ops Advisor (Insights)

Admin-only automated-operations surface. See [Ops Advisor](/docs/ops-advisor) for the full guide.

| Method | Path | Description |
|---|---|---|
| `GET` | `/v1/admin/insights` | Open events (`?severity=`, `?scope=` filters), severity-sorted |
| `GET` | `/v1/admin/insights/history` | Closed events (bounded history) |
| `GET` | `/v1/admin/insights/rules` | Rule catalog with effective config per rule |
| `PUT` | `/v1/admin/insights/rules/{id}` | Tune `enabled` / `muted` / `severity_override` / `thresholds` / `auto_remediate` |
| `GET/PUT` | `/v1/admin/insights/channels` | Alert-channel config (secrets write-only on `GET`) |
| `POST` | `/v1/admin/insights/run` | Force one evaluation tick → `{ opened, closed }` |
| `POST` | `/v1/admin/insights/{event_id}/ack` | Acknowledge an open event |
| `POST` | `/v1/admin/insights/{event_id}/triage` | AI triage of an event — grounded explanation + tool trace (`404` unknown, `503` no LLM) |
| `GET` | `/v1/admin/resources` | Local node's latest resource snapshot — disk, CPU, load, fds, memory, RSS (see [Node Resource Signals](/docs/resource-signals)); always `200`, fields nullable |

## Service Reliability (SLOs & synthetics)

Admin-only SLO and synthetic-probe surface. See [Service Reliability](/docs/service-reliability) for the full guide.

| Method | Path | Description |
|---|---|---|
| `GET/POST` | `/v1/admin/slos` | List SLO defs / create-or-overwrite (POST needs an explicit `id`) |
| `GET/PUT/DELETE` | `/v1/admin/slos/{id}` | Get / update / delete one SLO (`404` if missing) |
| `GET` | `/v1/admin/slos/{id}/status` | Current SLI / error-budget / burn-rate status |
| `GET/POST` | `/v1/admin/synthetics` | List probe defs / create-or-overwrite |
| `GET/PUT/DELETE` | `/v1/admin/synthetics/{id}` | Get / update / delete one probe |
| `GET` | `/v1/admin/synthetics/{id}/results` | Recent per-probe results + summary |

## APM & performance

Admin-only application-performance surface. See [APM & Performance Observability](/docs/apm-observability) for the full guide.

| Method | Path | Description |
|---|---|---|
| `GET` | `/v1/admin/correlations` | Slow-query / error correlation, bucketed and ranked worst-first |
| `GET` | `/v1/admin/pprof` | On-demand CPU profile (`?seconds=`, `?frequency=`, `?format=flamegraph\|pprof`); `profiling` build; `409` if running |
| `GET` | `/v1/admin/depmap` | Declarative service dependency graph (9 nodes / 8 edges) |
| `GET` | `/v1/admin/index-advisor` | Per-collection index type / params / p95 / recall + recommendations + open events |

`POST /v1/agents/{id}/converse` and `POST /v1/admin/insights/{event_id}/triage` also return `metrics` (`RunMetrics`: timings + token counts).

## Logs (Log Insights)

Captured-log and log-ingest surface. See [Log Insights](/docs/log-insights) for the full guide.

| Method | Path | Description |
|---|---|---|
| `GET` | `/v1/admin/logs` | Recent captured logs (`?level=`, `?q=`, `?limit=`) — admin-only |
| `GET` | `/v1/admin/logs/categories` | Semantic log categories — admin-only |
| `GET` | `/v1/admin/slow-queries` | Slowest recent queries (`?collection=`, `?limit=`) — admin-only |
| `POST` | `/v1/collections/{name}/ingest-logs` | Parse raw log lines into records (`{ lines, default_level?, ttl_secs? }`); same auth as `/insert` |

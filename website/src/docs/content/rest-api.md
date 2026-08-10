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
| `POST` | `/v1/collections/{name}/ingest-media` | Multipart upload-and-embed for image/audio/video (`-F file=@…`) — infers modality, embeds, stores the blob, inserts; see [Multimodal](/docs/multimodal#upload-and-embed-ingest-media) |
| `GET` | `/v1/collections/{name}/records` | List records (paginated; supports `?at=<ts>` snapshot reads) |
| `GET` | `/v1/collections/{name}/records/{id}` | Fetch one record |
| `POST` | `/v1/collections/{name}/search` | Search: `vector`, `bm25`, `sparse_vector`, `hybrid`, or `multi_vector` queries + `filter`, `rerank`, `explain`, `?at=<ts>` |
| `POST` | `/v1/collections/{name}/seal` | Explicit seal/flush so declared index types build on sealed segments (also `VEDA_SEAL_THRESHOLD`) |
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

`cron` accepts a 5-field Unix expression (`0 3 * * *` = 03:00 daily) as well as
the 6- and 7-field forms with a leading seconds field. `kind` is an internally
tagged enum, so it nests as `{"kind": {"kind": "...", ...}}` — the same shape
SLO `kind` and synthetic-probe `kind` use.

## Documents

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/collections/{name}/process-document` | OCR → chunk → embed → insert one file (multipart `file`) |
| `POST` | `/v1/collections/{name}/process-documents` | Batch variant (multipart `files`) |
| `POST` | `/v1/collections/{name}/ingest-url` | Crawl a URL → clean markdown → chunk → embed → insert (`{ url\|urls, deep_crawl?, max_pages?, chunker?, ttl_secs? }`); see [Web Connectors](/docs/web-connectors); `503` if crawl4ai unconfigured |
| `POST` | `/v1/chunk/preview` | Preview how text would be chunked under a strategy |

## Connectors

Admin-only web data-source connectors (declarative, persisted, re-runnable). See [Web Connectors](/docs/web-connectors#connectors-framework).

| Method | Path | Description |
|---|---|---|
| `GET/POST` | `/v1/admin/connectors` | List / create a connector (`{ kind: "web", name, config, enabled }`; `id` server-assigned if omitted) |
| `GET/PUT/DELETE` | `/v1/admin/connectors/{id}` | Read / update / delete a connector |
| `POST` | `/v1/admin/connectors/{id}/run` | Run now — crawl the connector's URLs into its target collection (`503` if crawl4ai unconfigured) |

## Embedding & models

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/embed` | Embed `text` / `image_b64` / `audio_b64` / `video_b64` directly → `{model, modality, dimension, values, elapsed_ms}`; optional `model` override (see [Multimodal](/docs/multimodal#embedding-media-directly)) |
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
| `POST` | `/v1/agents/{id}/converse` | Run a tool-calling turn — body `{ input, conversation_id?, attachments? }`; returns `metrics` + [`citations` + `grounding`](/docs/rag-aiops#faithfulness-citations); `503` without an LLM, `404` for an unknown non-default agent |
| `POST` | `/v1/agents/{id}/converse/stream` | [Streaming converse over SSE](/docs/rag-aiops#streaming-converse-sse) — reasoning/tool/answer deltas then a `done` frame with `metrics` + `citations` + `grounding`; `503` no LLM |

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
| `POST` | `/v1/conversations/{id}/feedback` | Thumbs feedback `{ rating: up\|down, note? }` (see [LLM Observability](/docs/llm-observability)) |

**Eval harness** — see [RAG & AIOps Intelligence](/docs/rag-aiops#eval-harness-the-production-trace-flywheel).

| Method | Path | Description |
|---|---|---|
| `GET/POST` | `/v1/eval-sets` | List / create-or-overwrite eval sets (`id` server-assigned if omitted) |
| `GET/DELETE` | `/v1/eval-sets/{id}` | Get / delete an eval set |
| `POST` | `/v1/eval-sets/{id}/run` | Run → per-case + aggregate recall@k / faithfulness / answer-relevancy / pass_rate; `503` no LLM |
| `GET` | `/v1/eval-sets/{id}/runs` | List past runs |
| `POST` | `/v1/eval-sets/{id}/cases/from-conversation` | Graduate a production conversation into a regression case |

**Auto-tuning** — see [RAG Auto-Tuning](/docs/auto-tuning).

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/collections/{name}/eval-sets/generate` | Generate a labeled eval set from the collection's own documents (`{ name?, num_cases }`) |
| `GET` | `/v1/collections/{name}/eval-set-size-recommendation` | Corpus-aware `{ recommended, max, chunk_count, document_count }` |
| `GET/PUT` | `/v1/collections/{name}/auto-optimization` | Per-collection opt-in: regenerate + re-tune on every ingest (`{ enabled, target_questions?, mode }`) |
| `POST` | `/v1/collections/{name}/tuning-campaigns` | Run a nested config search → Pareto front (`{ eval_set_id, max_trials?, num_bases?, mode? }`) |
| `POST` | `/v1/tuning-campaigns/{campaign_id}/trials/{trial_id}/apply` | Apply a trial — tier-1 instant/reversible; tier-2/3 via `/reindex` + alias swap |

`mode` is `"Quality"` or `"Performance"` — capitalized exactly, on both endpoints. Any other spelling (`"quality"`) is rejected with `422`.

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
| `GET` | `/v1/saved_objects` | List saved objects used by the Niriksha dashboard |
| `PUT/DELETE` | `/v1/saved_objects/{id}` | Upsert (create-or-replace) / delete one saved object. There is no `POST` collection route and no `GET` by id — create with `PUT` on the id you choose, and read via the list |

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
| `POST` | `/v1/admin/insights/{event_id}/triage` | AI triage of an event — grounded explanation + tool trace, now enriched with the RCA bundle (`404` unknown, `503` no LLM) |
| `GET` | `/v1/admin/insights/{event_id}/rca` | [Cross-signal root-cause bundle](/docs/rag-aiops#cross-signal-rca) — logs + queries + traces + resources + incident |
| `GET` | `/v1/admin/incidents` | [Semantic incident clusters](/docs/rag-aiops#semantic-incident-correlation) — with `draft` (proactive RCA), `is_novel` / `novelty_score`, and a `noise_reduction` summary |
| `GET` | `/v1/admin/forecast` | [Capacity forecasts](/docs/rag-aiops#capacity-forecasting) — disk-full / memory-exhaustion / recall-decay ETAs |
| `GET` | `/v1/admin/change-points` | [Change-point detections](/docs/aiops#change-point-detection) — sustained level shifts / trend breaks (warm-up ≥20 pts, 1.5σ) |
| `GET` | `/v1/admin/log-rate-analysis` | [Semantic spike attribution](/docs/aiops#semantic-spike-root-cause-attribution) — meaning-clusters driving a log-rate spike (`?window_secs=&baseline_secs=&min_level=`) |
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

## LLM Observability

Admin-only LLM trace surface. See [LLM Observability](/docs/llm-observability) for the full guide.

| Method | Path | Description |
|---|---|---|
| `GET` | `/v1/admin/llm-traces` | Semantic search over the `_llm_traces` store (`?q=&limit=`) |
| `GET` | `/v1/admin/llm-traces/clusters` | Failure clustering — recurring failure modes (`category`, `representative`, `count`) |

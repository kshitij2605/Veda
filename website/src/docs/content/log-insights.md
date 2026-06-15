# Log Insights

**Log Insights** captures Veda's own logs, correlates them with traces, **categorizes them semantically**, alerts on log content through the [Ops Advisor](/docs/ops-advisor), and can store external logs as a searchable collection.

It is deliberately *not* a full logs platform (no ingest agents, ECS normalization, or ILM tiers — that's outside a vector database's lane). Instead it builds the pieces that reuse Veda's strengths — the embedder, BM25, the filter DSL, TTL — and adds the one thing a vector DB can do that template-based log tools can't: **semantic log categorization** that groups differently-worded messages about the same underlying failure.

## Log capture & trace correlation

A `tracing` layer captures every event at or above `VEDA_LOG_CAPTURE_LEVEL` (default `WARN`) into a bounded in-memory ring buffer. Each record is:

```json
{
  "ts_ms": 1718409600000,
  "level": "WARN",
  "target": "veda_gateway::notify",
  "message": "send failed",
  "fields": { "channel": "webhook" },
  "trace_id": "…",
  "span_id": "…"
}
```

`trace_id` / `span_id` are read best-effort from the active OpenTelemetry span — populated when an OTLP endpoint is configured (see [Observability](/docs/observability)), `null` otherwise. The per-query log carries a `trace_id` too, so logs and traces line up. Capture is synchronous, non-blocking, and panic-free; the same buffer feeds the REST API, the categorizer, and the Ops Advisor.

```bash
# recent captured logs, with level + substring filters
curl "http://localhost:6340/v1/admin/logs?level=WARN&q=send&limit=100"
```

## Semantic categorization

`GET /v1/admin/logs/categories` pulls recent WARN/ERROR messages, embeds each *distinct* message (cache-aware — only new messages embed), and greedily clusters them by cosine similarity:

```json
{
  "id": "…",
  "representative": "send failed: connection refused",
  "count": 7,
  "max_severity": "ERROR",
  "examples": ["send failed: connection refused", "delivery failed — host unreachable"],
  "first_seen_ms": 1718409600000,
  "last_seen_ms": 1718409930000
}
```

This groups *semantically similar* errors, not just identical templates. When no text embedder is configured, it falls back to grouping by a **normalized template** (numbers, ids, and quoted strings collapsed to placeholders).

## Log-content alerting

The [Ops Advisor](/docs/ops-advisor) snapshot gained `error_log_count` / `warn_log_count` (over a 5-minute window), powering three rules that open events and flow through the existing alert channels (webhook, Slack, PagerDuty, email):

| Rule | Severity | Fires when |
|---|---|---|
| `log_error_rate` | High | ERROR-log count over the window exceeds a threshold |
| `anomaly_error_logs` | Med/High | error-log rate spikes above its EWMA/σ baseline |
| `new_error_category` | Medium | a newly-appeared WARN/ERROR semantic category is detected |

All three are tunable, mutable, and auto-remediable like every other rule via `/v1/admin/insights/rules` — see [Ops Advisor](/docs/ops-advisor#rest-api).

## Slow queries

`GET /v1/admin/slow-queries?collection=&limit=` returns the slowest recent queries, server-backed from each collection's query log (tenant-scoped):

```bash
curl "http://localhost:6340/v1/admin/slow-queries?collection=products&limit=20"
```

## Logs as a searchable collection

Ship external logs into a collection and get a searchable archive — BM25 full-text **and** semantic similarity, plus the filter DSL, TTL retention, and [analytics aggregations](/docs/rest-api#analytics):

```bash
curl -X POST http://localhost:6340/v1/collections/applogs/ingest-logs \
  -H "Content-Type: application/json" \
  -d '{
    "lines": [
      "{\"ts\":\"2026-06-15T10:00:00Z\",\"level\":\"error\",\"message\":\"disk full on node 3\"}",
      "2026-06-15T10:00:01Z INFO request completed in 12ms"
    ],
    "default_level": "INFO",
    "ttl_secs": 604800
  }'
```

Each line is parsed and inserted via the standard insert / embed-insert path:

- **JSON line** — `message` (from `message`/`msg`/`body`), `level` (`level`/`severity`/`lvl`, upper-cased), `ts` (RFC3339 or epoch); remaining scalar keys become fields.
- **Plain line** — a leading RFC3339 timestamp + level token are parsed off, the rest is the message; otherwise the whole line is the message with `default_level` and the current time.
- `ttl_secs` sets `_expires_at` for retention (see [Record TTL](/docs/collections#record-ttl)).

When the collection has a vector field with `auto_embed` over `message`, ingested messages are embedded for **semantic** log search; otherwise it's a plain insert. Same auth and tenant scoping as `/insert`.

**Recommended schema for a logs collection:**

```json
{
  "name": "applogs",
  "fields": [
    {"name": "message", "type": "string", "bm25": true},
    {"name": "level", "type": "string", "filterable": true},
    {"name": "ts", "type": "int64", "filterable": true},
    {"name": "embedding", "type": "vector", "dimension": 1024, "metric": "cosine",
     "model": "BAAI/bge-m3", "auto_embed": {"source_field": "message"}}
  ]
}
```

## In Niriksha

The dashboard's **Logs** page gives a live tail (with level filter and text search), a `trace_id` column, the semantic-categories panel, the slow-query table, and a box to search a logs collection — plus nav and command-palette entries.

## Configuration

| Variable | Purpose |
|---|---|
| `VEDA_LOG_CAPTURE_LEVEL` | Capture threshold — `ERROR` / `WARN` / `INFO` / `DEBUG` / `TRACE` (default `WARN`) |
| `VEDA_ADVISOR_INTERVAL_SECS` | Advisor tick — drives the log rules and categorization cadence (default 30s) |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | When set, log records carry `trace_id` / `span_id` |
| `EMBEDDING_*` | Enables semantic categorization and semantic log search; without it, categorization falls back to normalized templates |

## Current limitations

- `trace_id` is `null` unless an OTLP endpoint is configured (best-effort by design).
- The capture buffer is in-memory and bounded (a recent window only) — for long-term history, ingest into a logs collection or use the external Loki stack.
- This is not a full log platform: no ingest agents, ECS normalization, ILM tiers, or a Logs Explorer — by design.

# Veda Log Insights

**Date:** 2026-06-15
**Status:** Shipped on `main` (v0.2.6). Validated end-to-end against a live server.

Log Insights is Veda's in-lane answer to a log-monitoring platform: capture the system's
own logs, **correlate** them with traces, **categorize** them semantically, **alert** on
log content, and optionally **store external logs as a searchable collection**. It does
NOT try to be Elasticsearch's full logs platform (ingest agents, ECS, ILM tiers, Logs
Explorer) — that's outside a vector DB's lane. It builds the pieces that reuse Veda's
strengths (the embedder, BM25 + analyzer, the filter DSL, TTL, the Ops Advisor) and adds
one thing a vector DB can do that template-based tools can't: **semantic log
categorization**.

---

## 1. Architecture

A foundational **in-process log capture** unlocks most of the feature.

```
  tracing events (≥ VEDA_LOG_CAPTURE_LEVEL, default WARN)
        │  CaptureLayer (sync, non-blocking, poison-safe)
        ▼
  LogBuffer  (bounded ring, on AppState)  ──────────────┐
        │                                               │
        ├── GET /v1/admin/logs            (live tail)    │
        ├── LogCategorizer.categorize()  → /logs/categories (embed + cluster)
        └── Ops Advisor snapshot          (error/warn counts) → log rules → events → alerts

  Per-collection query log (_query_log.jsonl) ── GET /v1/admin/slow-queries
  POST /v1/collections/{name}/ingest-logs  → records (BM25 + semantic + TTL)
```

- `crates/veda-gateway/src/logs/` — `buffer.rs` (`LogRecord`, `LogBuffer`), `layer.rs`
  (`CaptureLayer`), `categorize.rs` (semantic clustering), `parse.rs` (log-line parsing).
- The capture buffer is the SAME `Arc<LogBuffer>` installed in the tracing subscriber and
  stored on `AppState` — so the REST API, categorizer, and advisor all see captured logs.
- `on_event` is synchronous, non-blocking, panic-free, and recovers from a poisoned lock.

---

## 2. Capture + trace correlation (#1)

A `tracing_subscriber::Layer` captures events at/above `VEDA_LOG_CAPTURE_LEVEL` (default
`WARN`) into a bounded ring (`LogRecord { ts_ms, level, target, message, fields,
trace_id, span_id }`). `trace_id`/`span_id` are read best-effort from the current OpenTelemetry
span context — populated when an OTLP endpoint is configured, `null` otherwise. The
per-query `QueryLogEntry` also gained a `trace_id`. This is the basis for log↔trace
correlation.

---

## 3. Semantic log categorization (#4-core)

`LogCategorizer.categorize()` pulls recent WARN/ERROR messages, embeds each *distinct*
message via the (async) embedding client — cache-aware, so only new messages embed — and
greedily clusters them by cosine similarity into `LogCategory { id, representative,
count, max_severity, examples, first_seen_ms, last_seen_ms }`. When no text embedder is
configured it falls back to grouping by a **normalized template** (numbers/ids/quoted
strings collapsed to placeholders). The embedding awaits run **outside** the cache lock
(the lock is released around the network calls). `GET /v1/admin/logs/categories`.

This is the differentiator: it groups *semantically similar* errors (e.g. differently-worded
messages about the same failure), not just identical templates.

---

## 4. Log REST API + Niriksha Logs page (#2)

All admin-gated:
- `GET /v1/admin/logs?level=&q=&limit=` — recent captured logs (level + substring filters).
- `GET /v1/admin/slow-queries?collection=&limit=` — slowest recent queries, **server-backed**
  from each collection's `_query_log.jsonl` (tenant-scoped; replaces the old browser-local
  slow-query view).
- `GET /v1/admin/logs/categories` — the semantic categories.

The Niriksha **Logs** page provides a live tail (poll), level filter, text search, a
`trace_id` column, a categories panel, and the slow-query table — plus a nav entry and a
command-palette entry. A small "search a logs collection" box ties into #5.

---

## 5. Log-content alerting (#3)

The Ops Advisor was extended (it already does detection → recommendation → alerting →
opt-in remediation). The snapshot gained `error_log_count` / `warn_log_count` (over a
5-minute window from the buffer), and three rules were added:
- `log_error_rate` — fires when ERROR-log count over the window exceeds a threshold.
- `anomaly_error_logs` — EWMA/σ baseline spike on the error-log rate.
- `new_error_category` — a newly-appeared WARN/ERROR semantic category (detected in a
  **lock-free Phase 0** of the advisor tick, since categorize may embed over the network).

All three open events in the existing `EventStore`, flow through the existing alert
channels (webhook/Slack/PagerDuty/email), and are tunable/mutable via
`/v1/admin/insights/rules` like every other rule.

---

## 6. Logs-as-a-collection ingest (#5)

`POST /v1/collections/{name}/ingest-logs { lines: [..], default_level?, ttl_secs? }`
parses each raw line and inserts it as a record:
- **JSON line:** `message` (from `message`/`msg`/`body`), `level` (`level`/`severity`/`lvl`,
  upper-cased), `ts` (RFC3339 or epoch ms/secs); remaining scalar keys → fields.
- **Plain line:** leading RFC3339 timestamp + level token, rest = message; otherwise the
  whole line is the message with `default_level` and now.
- Optional `_expires_at = now + ttl_secs` for retention.

Records are inserted via the existing insert / embed-insert path — so when the collection
has a vector field with `auto_embed` over `message`, the messages are embedded for
**semantic** search; otherwise it's a plain insert. Same auth/tenant as `/insert`.

**Recommended logs-collection schema:** `message` (String, `bm25=true`), `level` (String,
filterable), `ts` (Int64, filterable), optional `embedding` (Vector, `auto_embed`
source=`message`). The result is a searchable log archive: BM25 full-text + semantic
similarity + the filter DSL + TTL retention + analytics aggregations.

---

## 7. BM25 / lexical filter + RLS fix (shipped alongside)

Previously the lexical search paths (`execute_bm25`, `execute_match_phrase`, and the BM25
leg of hybrid) **ignored the request `filter`/`filter_expr`** — so scalar filtering and,
more importantly, **row-level security were not applied to BM25 queries** (an RLS bypass
on the lexical path). Fixed: these paths now resolve the effective filter (preferring the
RLS-injected `filter_expr`, else parsing `filter`), oversample candidates, drop
non-matches via `evaluate_record`, and intersect MVCC visibility — matching the dense
path. So `{ query:{bm25...}, filter:"level='ERROR'" }` now returns only matching records,
and RLS-scoped users only see their own rows on BM25/hybrid queries too.

(Follow-up: the oversample window is a fixed `k×10` per posting list, so an extremely
common term combined with a highly selective filter could under-recall.)

---

## 8. Configuration

- `VEDA_LOG_CAPTURE_LEVEL` — capture threshold (`ERROR`|`WARN`|`INFO`|`DEBUG`|`TRACE`; default `WARN`).
- `VEDA_ADVISOR_INTERVAL_SECS` — advisor tick interval (default 30s); drives the log rules + categorization cadence.
- `OTEL_EXPORTER_OTLP_ENDPOINT` — when set, log records carry `trace_id`/`span_id`.
- Embedding endpoint (`EMBEDDING_*`) — enables *semantic* categorization + semantic log search; without it, categorization falls back to normalized templates and ingest is plain text.

---

## 9. Validation (live)

Verified against a running server with the BGE-M3 embedder:
- **Ingest:** ingested JSON + plain + garbage lines → parsed levels correctly →
  `aggregate group_by level` = `{ERROR:3, INFO:2, WARN:1}` → **BM25 search "disk" →
  "disk full on node 3"**; TTL applied (proven by the integration test).
- **Capture:** real server WARNs (`notify: send failed`) were captured into the buffer
  with structured fields; `GET /v1/admin/logs?level=WARN` returned them.
- **Categorization:** `/v1/admin/logs/categories` grouped them ("[WARN] ×2").
- **Alerting:** the advisor opened a **`new_error_category`** event live for the
  freshly-appeared category (and it routes to the alert channels).
- **BM25 filter/RLS fix:** BM25 + `level='ERROR'` returns only ERROR rows; a non-admin
  BM25 search on an RLS collection returns only the caller's records.

---

## 10. Limitations & follow-ups

- `trace_id` is `null` unless an OTLP endpoint is configured (best-effort by design).
- The capture buffer is in-memory + bounded (recent window only); long-term log history
  belongs in a logs *collection* (#5) or the external Loki stack.
- This is NOT a full log platform: no ingest agents, ECS normalization, ILM data tiers,
  or a Kibana-style Logs Explorer — out of a vector DB's lane.
- BM25 lexical filter oversampling (`k×10`) could under-recall on extreme term/filter
  combinations (see §7).
- `chunking.rs` / `splade_client.rs` still use `reqwest::blocking` (unrelated landmine,
  tracked separately) — not on the Log Insights paths.

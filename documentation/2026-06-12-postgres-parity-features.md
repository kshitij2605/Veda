# Postgres-Extension-Parity Features

**Date:** 2026-06-12
**Trigger:** A review of eight Postgres-extension-style capabilities (JSONB, pg_cron,
in-memory/TTL cache, pg_graphql, ElectricSQL realtime, pgcrypto+pgjwt+RLS,
pg_mooncake analytics, PostgREST) and the question: for each, is it feasible and
useful for Veda, and is it better to adapt the existing Postgres extension's code or
build from scratch?
**Status:** Complete. Five features shipped and merged to `main` (`v0.2.3`).

This document records the feasibility assessment for all eight, the cross-cutting
"adapt vs build" decision, and what shipped for the five selected features —
including a security issue caught in review before it could ship.

---

## 1. The cross-cutting finding — adapt vs build

Postgres extensions are written in C / SQL / PL-pgSQL / Elixir and are coupled to
Postgres internals: the SPI, the system catalogs, the SQL executor, and
WAL/logical-replication. Veda is a Rust vector database with no SQL engine. **None of
their code is reusable.**

But every *concept* ports, and Veda already owns the primitives each feature needs:
the filter DSL (`veda-query::filter`), columnar stored fields
(`veda-storage::columnar_stored_fields`), the `veda-log` change log, tokio background
tasks, `argon2` + `jsonwebtoken`, and tenant scoping. So the uniform decision was:
**build Rust-native, reusing Veda's existing infrastructure and standard Rust crates —
never fork a Postgres extension.**

### Feasibility assessment (all eight)

| # | Feature | In Veda before this run | Decision |
|---|---|---|---|
| 6a | Password hashing | **Present** — `argon2` | already done |
| 6b | JWT sessions | **Present** — `jsonwebtoken` | already done |
| 8 | PostgREST (REST-from-schema) | **Present** — hand-written, vector-aware REST | already done; only gap is auto-OpenAPI |
| 1 | JSONB | **Partial** — store/retrieve `FieldValue::Json`, no nested-path query | **built** — JSON-path filtering |
| 6c | Row-level security | **Absent** — had RBAC + tenant scoping, not per-record owner | **built** — RLS owner injection |
| 2 | pg_cron | **Absent** — internal tokio loops only | **built** — cron scheduler |
| 3 | Redis/TTL cache | **Partial** — RAM segments, JWKS TTL; no record TTL | **built** — record TTL + sweeper |
| 7 | pg_mooncake analytics | **Partial** — columnar store, `/evaluate`, `/coverage`; no OLAP aggregations | **built** — analytics aggregations |
| 5 | ElectricSQL realtime | **Absent** — `veda-log` CDC exists, no client subscribe | **deferred** (feasible: tail `veda-log` → SSE/WS) |
| 4 | pg_graphql | **Absent** — REST + MCP only | **deferred** (feasible but least essential for a vector DB) |

The five middle rows are what shipped this run.

---

## 2. JSONB-style path filtering (#1)

**Why:** Veda stored and returned JSON (`FieldType::Json` / `FieldValue::Json`) but
the filter DSL could not query *into* nested JSON.

**What shipped:** the filter DSL now accepts dotted field paths. When the leading
segment resolves to a `FieldValue::Json`, the remaining segments are traversed with
`serde_json::Value::pointer` and the leaf is coerced to a comparable scalar.

```
filter = meta.user.tier = "gold"
```

A missing path yields no match; a dotted path against a non-JSON field is unaffected.
Because the resolution lives in the shared `record_eval`, it works on every filter
site — search, record listing, and the new aggregations.

**Key file:** `crates/veda-query/src/filter/record_eval.rs` (`resolve_field`).
**Reuse:** existing `FilterExpr` / `FilterValue` comparison; `serde_json`'s built-in
`pointer()` (no new crate).

---

## 3. Row-level security (#6c)

**Why:** Veda had role-level RBAC and collection-level tenant scoping, but no
per-record owner access control (a user seeing only the rows they own).

**What shipped:** a collection may declare an optional `rls_owner_field` at create
time. When set, on every read (search/get/list) the gateway AND-injects
`<owner_field> == <jwt-subject>` into the parsed `FilterExpr`; on insert it enforces
the owner equals the subject. The Admin role bypasses both, mirroring the existing
RBAC admin checks.

**Key files:** `crates/veda-gateway/src/rls.rs` (`inject_owner_filter`,
`enforce_owner_on_insert`), wired in `rest/search.rs` and `rest/records.rs`;
`crates/veda-core/src/schema.rs` (`rls_owner_field`).

**Security note (caught in review):** an early draft serialized the trusted owner
predicate back to a *filter string* (`owner = 'alice'`) and re-parsed it. Because the
string formatter did not escape quotes and the parser did not handle escapes, a
username containing a quote could break out of the predicate and bypass RLS entirely.
The fix was to stop round-tripping through a string: the requests now carry a
`filter_expr: Option<FilterExpr>` and the owner clause is combined at the **AST
level**, never re-serialized. Review also surfaced a *second* vulnerable site in
`local_shard.rs::search_sealed_segment_ref` that had to prefer `filter_expr` over the
re-parsed string. Both were fixed before merge.

---

## 4. Cron-style scheduled jobs (#2)

**Why:** Veda had no user-facing scheduler — only hard-coded internal tokio loops.

**What shipped:** an admin-only scheduler.

- `POST /v1/admin/jobs` — register or overwrite a job (400 on an invalid cron expr).
- `GET /v1/admin/jobs` — list jobs.
- `DELETE /v1/admin/jobs/{name}` — remove a job.

A job is a cron expression plus a **typed `JobKind` enum** — `TtlSweep`, `Reindex`,
`Compact`, `Evaluate`, each carrying `{ collection }` — deliberately *not* arbitrary
code (safer than pg_cron's arbitrary SQL). The registry persists to JSON in the data
dir and reloads on restart. A single tokio loop ticks and dispatches due jobs;
`is_due(now)` is deterministic on an injected `now` for testability.

`TtlSweep` is wired to the record-TTL sweeper (below). `Reindex` / `Compact` /
`Evaluate` are registered hooks whose dispatch is stubbed pending later wiring.

**Key files:** `crates/veda-gateway/src/scheduler.rs`,
`crates/veda-gateway/src/rest/jobs.rs`.
**Reuse:** the `tokio::time::interval` background-task pattern; the `cron` crate; the
SavedObject-style persistence; admin RBAC.

---

## 5. Record TTL + expiry sweeper (#3)

**Why:** no per-record TTL/expiry, which blocks Redis-like ephemeral/cache use. (The
Postgres `UNLOGGED TABLE` model does *not* apply — Veda has no WAL cost for ephemeral
data, so RAM growing-segments plus a TTL sweeper is the correct Veda-native model.)

**What shipped:** a reserved record field `_expires_at` (unix seconds) flowing through
the existing `fields` map — no schema or struct change. Expired records are excluded
at **query time** from search (`populate_hit_fields` and rerank), `GET /records/{id}`,
and `GET /records`, via `SegmentManager::lookup_record_at(id, now)`. A `TtlSweep`
scheduled job calls `SegmentManager::sweep_expired(now)` so compaction physically
purges them.

**Completeness note (caught in review):** the first pass only filtered expired records
in the sweeper, so an expired record still appeared in search results until the next
sweep. Read-time exclusion was added to search, rerank, get, and list to close that
gap.

**Key files:** `crates/veda-core/src/record.rs` (`EXPIRES_AT_FIELD`,
`expires_at_secs`, `is_expired_at`); `crates/veda-storage/src/segment_manager.rs`
(`lookup_record_at`, `sweep_expired`); `rest/search.rs`, `rest/records.rs`. A separate
fix made the reserved `_expires_at` field persist through segment seal (a non-schema
field was previously dropped).

---

## 6. Analytics aggregations (#7)

**Why:** columnar storage existed but there was no OLAP aggregation API — only
`/evaluate` quality stats and `/coverage` retrieval stats.

**What shipped:** three read-only endpoints scanning the columnar reader across sealed
and growing segments, all honoring the filter DSL (and RLS injection):

- `GET /v1/collections/{name}/aggregate?field=&metric={count|sum|avg|min|max}&group_by=&filter=`
  — scalar metric with optional per-group buckets; numeric metrics validate the field
  is numeric.
- `GET /v1/collections/{name}/histogram?field=&bins=&filter=` — equal-width numeric
  histogram (default 10 bins).
- `GET /v1/collections/{name}/field-stats?field=&filter=` — total/null counts,
  cardinality, and numeric (min/max/avg) or string (min/max length) summaries.

No DuckDB or external engine — plain accumulation in a HashMap over the existing
columnar reader.

**Key file:** `crates/veda-gateway/src/rest/aggregate.rs`.
**Reuse:** the columnar reader; the filter DSL; the `/evaluate` and `/coverage`
endpoint patterns.

---

## 7. Net result and follow-ups

**Shipped to `main` (`v0.2.3`):** JSON-path filtering, row-level security, a cron
scheduler, record TTL + sweeper, and analytics aggregations — all Rust-native, each
independently committed and reviewed (a spec-compliance pass then a code-quality pass
per feature, plus a final review).

**Deliberately deferred (feasible, not selected this run):**
- ElectricSQL-style realtime — tail `veda-log` and push changes over SSE/WebSocket.
- pg_graphql-style GraphQL — `async-graphql` over `CollectionSchema`; least essential
  for a vector DB.

**Documented follow-ups within the shipped features:**
- Wire `Reindex` / `Compact` / `Evaluate` scheduler kinds to their existing operations
  (only `TtlSweep` dispatches today).
- A JSON containment operator (`@>`) for objects/arrays, if a use case appears.

**Process note:** the two-stage review per feature paid for itself here — it caught a
genuine RLS-bypass security vulnerability (the unescaped filter-string round-trip, plus
a second bypass site) and a TTL correctness gap (expired records visible in search
until the sweep), both before they reached `main`.

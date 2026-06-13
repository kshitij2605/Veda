# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.5] - 2026-06-14

Ops Advisor ("Insights") — an AutoOps-style automated operations layer (detection → root-cause → recommendation → alerting), validated end-to-end against a live server. See `documentation/2026-06-14-autoops-ops-advisor.md` for the full guide.

### Added
- **Ops advisor engine** — new `crates/veda-gateway/src/advisor/` module with a background tick (`spawn_advisor`, interval via `VEDA_ADVISOR_INTERVAL_SECS`, default 30s) that samples an `OpsSnapshot` (per-collection segments/records, compaction state, nodes, replica `safe_ts`, health probes, query-log latency, process RSS, Prometheus counters) into a bounded history buffer.
- **Rule catalog (15 rules)** — threshold+correlation rules: `compaction_backlog`, `seal_stalled`, `high_search_latency`, `slow_query_surge`, `high_error_rate`, `embedding_unhealthy`, `dependency_down`, `memory_pressure`, `replica_lag`, `node_unhealthy`, `no_compaction_progress`, plus 4 anomaly detectors (`anomaly_search_latency`, `anomaly_error_rate`, `anomaly_insert_rate`, `anomaly_rss`) using EWMA/σ baselines. Each rule yields an event with severity (High/Medium/Low), root-cause detail, and a recommendation.
- **Event lifecycle** — an `EventStore` opens an event on first fire (dedup per rule+scope), refreshes in place, and **auto-closes** when the condition clears; bounded history; persisted to `{data_dir}/_ops_events.json`.
- **Alerting** — notify on open/close transitions via **generic webhook, Slack, PagerDuty (Events API v2 trigger/resolve keyed by event id), and email (async SMTP via `lettre`)**. Severity-routed (per-channel `min_severity`), per-channel/per-event failure-isolated, async-only HTTP with a 10s client timeout. Channel config persisted; secrets are write-only (redacted on `GET`).
- **Opt-in auto-remediation** — per-rule `auto_remediate` (default **off**); when enabled, an opened event can trigger a safe `RemediationAction` (real compaction via the existing path; reindex is best-effort) — rate-limited (≤1/5min/scope) and recorded on the event.
- **Insights REST API (admin-only)** — `GET /v1/admin/insights` (open events), `/insights/history`, `/insights/rules` (catalog + effective config), `PUT /insights/rules/{id}` (tune threshold/severity/enabled/muted/auto_remediate), `GET|PUT /insights/channels` (secrets write-only), `POST /insights/run` (force a tick), `POST /insights/{event_id}/ack`. Config persisted to `{data_dir}/_advisor_config.json`.
- **Niriksha Insights page** — a dashboard "Insights" page (open/history events, severity badges, event detail drawer with root cause + recommendation + remediation affordance, rule settings, alert-channel config) + a sidebar open-event-count badge + a command-palette entry.

### Fixed
- **Advisor availability** — the engine tick previously held the `events` lock across the notification/remediation network awaits while the notifier had no HTTP timeout, so a hung webhook could freeze the Insights API and the tick loop. The reconcile phase is now synchronous under-lock; notify/remediation run lock-free; the `events` lock is re-acquired only briefly to annotate; and the notifier client has a 10s timeout.

## [0.2.4] - 2026-06-13

Agentic layer and a real Model Context Protocol (MCP) surface, validated end-to-end against a live LLM (Qwen3.5-122B) + embedder (BGE-M3). See `documentation/2026-06-13-agents-and-mcp.md` for the full guide.

### Added
- **Agent framework** — conversational agents that ground answers in your collections. `Agent { id, name, instructions, tool_ids, model? }` persisted as saved objects; a built-in in-memory `veda-agent` (general-purpose, all tools) is always available. New `crates/veda-gateway/src/agent/` module: `model`, `store`, `catalog`, `tools/`, `executor`.
  - `POST /v1/agents/{id}/converse` — runs a bounded multi-step tool-calling (ReAct-style) loop over the configured LLM (`AgentExecutor::run`, default max 8 steps, tool-result truncation), returning `{ conversation_id, response, steps[] }` where each step is `reasoning` / `tool_call` / `tool_result`.
  - Agent CRUD (`GET/POST /v1/agents`, `GET/PUT/DELETE /v1/agents/{id}`), custom-tool CRUD (`/v1/agent-tools`), and conversation history (`GET/DELETE /v1/conversations`). All tenant-scoped; row-level security enforced through tool calls via the caller's identity.
- **Shared tool catalog** — `ToolCatalog` with 8 built-in tools used by BOTH the agent executor and the MCP server: `search` (multimodal — text or base64-image query → cross-modal hits, with `fields` projection), `aggregate`, `filter_query`, `get_record`, `describe_collection`, `list_collections`, `route`, `upsert_record`. `GET /v1/agent-tools/catalog` lists their specs; `POST /v1/agent-tools/{id}/execute` invokes one directly.
- **LLM tool-calling** — `veda-core::llm_client` gained `complete_with_tools` + a `ChatModel` trait (OpenAI-compatible `tools`/`tool_calls`); the existing `complete()` is unchanged.
- **MCP transports (the standalone server is now actually connectable by real clients)**:
  - New `veda-mcp` crate — a stdio↔HTTP bridge binary so Claude Desktop / Cursor / VS Code (stdio MCP) can use Veda. Configure with `--url`/`VEDA_MCP_URL` + `--token`/`VEDA_MCP_TOKEN`.
  - `/v1/mcp` is now a spec-compliant **stateless Streamable HTTP** endpoint (`MCP-Protocol-Version` header negotiation, `GET → 405`) for `mcp-remote` and HTTP MCP clients.
- **A2A (Agent-to-Agent)** — `GET /v1/a2a/{agent_id}/card` returns an A2A AgentCard (skills derived from the agent's tools); `POST /v1/a2a/{agent_id}` handles JSON-RPC `message/send` (sync, one-shot), so external orchestrators can consume a Veda agent as a peer.

### Changed
- **MCP server upgraded** — `/v1/mcp` now exposes the **full 8-tool catalog** (was 4) with MCP tool **annotations** (`readOnlyHint`/`destructiveHint`/`title`); tool results carry **`structuredContent`** alongside the text block; tool-execution errors return a correctable **`isError: true`** result instead of a JSON-RPC error (protocol errors stay JSON-RPC); protocol version bumped to **`2025-06-18`** with version negotiation on `initialize`.
- **Agent/MCP `search` now returns record content** — previously the search tool returned only `{id, distance}` (no fields); it now hydrates each hit's non-vector fields (tenant-scoped, TTL-excluded, RLS-filtered) and supports an optional `fields` projection to bound LLM context.

### Fixed
- **Embedding client panicked against a real endpoint** — `veda-core::embedding_client` built a `reqwest::blocking::Client` inside async handlers, which panics (*"Cannot drop a runtime in a context where blocking is not allowed"*), breaking `embed-insert`, `/v1/embed`, the document pipeline, and agent/MCP query embedding against any live embedder. Converted `embed_text`/`embed_bytes` to async `reqwest::Client`. (The mock test path hid this; it was caught by the first real-data end-to-end test.) Note: `chunking.rs` and `splade_client.rs` still use `reqwest::blocking` and must be migrated before their live paths are exercised.

## [0.2.3] - 2026-06-12

Postgres-extension-parity features, built Rust-native on Veda's existing primitives (filter DSL, columnar store, tokio background tasks, RBAC/JWT) — no Postgres extension code was forked. See `documentation/2026-06-12-postgres-parity-features.md` for the full feasibility assessment and design notes.

### Added
- **JSONB-style path filtering** — the filter DSL now accepts dotted field paths (`meta.user.tier = "gold"`) that traverse into nested `FieldValue::Json` values via `serde_json::Value::pointer`, coercing the leaf to a comparable scalar. Non-JSON fields and missing paths are unaffected (missing path → no match). Implemented in `veda-query::filter::record_eval::resolve_field`; works on every filter site (search, records, aggregations).
- **Row-level security (RLS)** — a collection may declare an optional `rls_owner_field` at create time (`POST /v1/collections`). When set, the gateway AND-injects `<owner_field> == <jwt-subject>` into the parsed `FilterExpr` on every read (search/get/list) and enforces `owner == subject` on insert; the Admin role bypasses. Injection happens at the AST level (`veda-gateway::rls::inject_owner_filter`) via a new `filter_expr: Option<FilterExpr>` carried on the search requests — the trusted predicate is never round-tripped through an unescaped filter string.
- **Cron-style scheduled jobs** — admin-only scheduler with `POST /v1/admin/jobs` (register/overwrite), `GET /v1/admin/jobs` (list), `DELETE /v1/admin/jobs/{name}`. Jobs carry a cron expression (validated by the `cron` crate; 400 on invalid) and a typed `JobKind` enum (`TtlSweep`, `Reindex`, `Compact`, `Evaluate`) — not arbitrary code. The registry persists to JSON in the data dir and reloads on restart; a single tokio loop ticks and dispatches due jobs (`is_due` is deterministic on an injected `now` for testability). `TtlSweep` is wired to the record-TTL sweeper; `Reindex`/`Compact`/`Evaluate` are registered hooks whose dispatch is stubbed pending later wiring.
- **Record TTL + expiry sweeper** — a reserved record field `_expires_at` (unix seconds) marks a record's expiry. Expired records are excluded at query time from search (`populate_hit_fields` + rerank), `GET /records/{id}`, and `GET /records` via `SegmentManager::lookup_record_at(id, now)`, and are physically purged by `SegmentManager::sweep_expired(now)` driven by a `TtlSweep` scheduled job. `Record::expires_at_secs()` / `is_expired_at(now)` and `veda_core::record::EXPIRES_AT_FIELD` back the convention; the reserved field now persists through segment seal (previously a non-schema field was dropped).
- **Analytics aggregations** — three read-only OLAP endpoints over the columnar store, all honoring the filter DSL (and RLS injection):
  - `GET /v1/collections/{name}/aggregate?field=&metric={count|sum|avg|min|max}&group_by=&filter=` — scalar metric with optional grouping; numeric metrics validate the field is numeric.
  - `GET /v1/collections/{name}/histogram?field=&bins=&filter=` — equal-width numeric histogram (default 10 bins).
  - `GET /v1/collections/{name}/field-stats?field=&filter=` — total/null counts, cardinality, and numeric (min/max/avg) or string (min/max length) summaries.

## [0.2.2] - 2026-05-26

Quality and DX improvements from the E2E test campaign.

### Added
- **Relative time expressions for `?at=` parameter** — MVCC snapshot reads now accept `now`, `now-1h`, `now-30m`, `now-5s` in addition to raw u128 HLC timestamps. Parses in `veda-gateway::rest::time_parse`.
- **CJK sentence splitting** — `split_sentences` in the chunking engine now recognizes `。` (U+3002), `？` (U+FF1F), and `！` (U+FF01) as self-delimiting sentence boundaries (no trailing whitespace required, unlike ASCII `.?!`).
- **MVCC GC stress benchmark** — `mvcc_gc_bench.rs` exercises GC correctness and throughput at 1M and 10M version scales (~400–470k versions/sec in debug mode).
- **100k-vector scale benchmark** — `scale_benchmark.rs` now runs 100k×1024-dim vectors in release mode across Flat, HNSW, RaBitQ, RaBitQ+Rerank, and HNSW+RaBitQ. Gated behind `#[ignore]`.
- **Multi-node cluster test plan** — `documentation/multi-node-test-plan.md` documents 3-node cluster test prerequisites and 6 test scenarios.
- **RSS scaling limit documentation** — README documents non-log-path memory growth and recommends `use_log_primary: true` above ~50k records.

### Changed
- **SIMD acceleration via SimSIMD** — the f32 distance kernels (`dot_product`, `cosine_distance`, `euclidean_distance`, `l2_squared`, `l2_norm_squared`) and `i8_dot_product` now dispatch to the SimSIMD backend, using AVX-512 / ARM NEON / SVE selected at runtime instead of AVX2-only. Enabled by default; `--no-default-features` keeps the prior AVX2/scalar path. `manhattan_distance` stays on the AVX2/scalar path (SimSIMD has no f32 L1 kernel). Measured ~9–11× faster than the release-mode scalar baseline for dot product at dims 768–3072 on the AVX-512 build host.
- **Scalar-quant indexes `Sq4` + `Sq8`** — new standalone `IndexType` variants: `Sq4` (4-bit per-dimension scalar quantization, ~D/2 bytes/vec) and `Sq8` (8-bit per-dimension, ~D bytes/vec, surfacing the previously internal-only `ScalarQuantizer` as a first-class index). Measured recall@10 at d=1536: Sq8 0.995, Sq4 0.879 (synthetic) / both 1.000 on real BGE-M3 embeddings — Sq4 beats the turbovec competitor's 4-bit (0.848) at equal memory. Accepted by the REST collection parser as `"sq4"` / `"sq8"`.
- **Durable applier cursor** — the per-shard log applier persists a CRC-framed cursor at the sealed (durable) watermark and resumes from it on restart, instead of replaying the whole log from seq 1. New `veda-log::ShardLogRegistry` + `veda-log::cursor`.
- **ShardLogRegistry replaces MultiGroupCoordinator** — the per-shard log map is now a first-class `veda-log::ShardLogRegistry`; the vestigial `MultiGroupCoordinator` (per-shard Raft groups unreachable since Phase E) is deleted. Collection drop now evicts + purges shard logs and cancels the collection's running applier tasks (previously they kept ticking after the collection was gone).
- **Insert response always returns `ids`** — the `InsertResponse.ids` array is now populated on both the log-primary and non-log paths (previously only on log-primary).
- **Background compaction** — segment compaction and index rebuild after insert are now spawned via `tokio::spawn` instead of running synchronously. A per-collection `AtomicBool` guard prevents concurrent compaction pileup. Insert latency is no longer blocked by I/O-heavy segment merges.
- **Prometheus metric labels** — collection labels in all metrics now use the user-facing display name (tenant prefix stripped) instead of the internal scoped name (e.g. `my_collection` instead of `default__my_collection`).

### Fixed
- **Sentence chunker UTF-8 overlap** — `chunk_by_sentence` overlap calculation now uses `floor_char_boundary` to avoid splitting multi-byte characters.

## [0.2.1] - 2026-05-20

Production-grade follow-ups to the Phase A-H rearchitecture: BookKeeper open-and-recover, auto-orchestrated ensemble changes, MVCC WAL checkpoint, native filtered search on every index, cross-region reader gating in the live request path, admin REST endpoints for region operations, and a real chunker/embedder reindex pipeline.

### Added — Phase D production-grade replication
- **BookKeeper open-and-recover** — `ShardLog::truncate_above(target)` walks the active segment via `LogEntryV2::decode` to find the first frame past `target` and truncates the file there. `LogFollower::truncate_above` mirrors the cut on followers. `LogWriter::open_and_recover` now runs Phase 3 truncate so a minority-acked entry cannot resurrect after a writer handover.
- **Ensemble change** — `rebuild_ensemble(current, failed, replacement)` produces a new `LogEnsemble` with the failed node swapped out and the epoch bumped; ready to feed into `RegisterShardEnsemble` on `_meta`.
- **Snapshot install for joining followers** — `ShardLog::copy_to(dest_dir)` fsyncs and copies rolled-over + active segment files so a new follower can catch up mid-stream cleanly.
- **`FollowerHealthMonitor` auto-orchestrator** — probes followers on an interval, edge-triggered (fires exactly once per consecutive-failure threshold crossing, not on every probe). `spawn(on_failure)` runs the probe loop and feeds the orchestrator hook without operator action.
- **`quorum_partition.rs`** — split-brain integration test with a `PartitionableFollower` wrapper; verifies the writer stalls on `Qa`-side, minority fences on heal, and no acked write is lost.

### Added — Phase F MVCC durability + native index visibility
- **`MvccWal::checkpoint(store, snap_path)`** — atomic snapshot file (`.tmp` + `fsync` + `rename`) followed by WAL truncate. `MvccStore::snapshot_entries()` / `install_snapshot_entries()` are the in-memory primitives. `SegmentManager::mvcc_checkpoint()` is the operator hook; `enable_mvcc_durable(path)` loads the snapshot first and then replays the trailing WAL so checkpoint + WAL together form the durable record without unbounded WAL growth.
- **Native `search_filtered`** for `Flat`, `IVF-PQ`, and `DiskANN` — the bitmap is consulted INSIDE the candidate walk, so invisible docs never occupy top-k slots and the 10× post-filter oversample disappears. DiskANN uses `l = self.l_search.max(k * 4)` to keep recall at high read-ts ratios.
- **`mvcc_isolation_gc_safety.rs`** — 8 linearisability tests covering SI invariants (reader at T never sees write at T+1; cross-key interleavings; tombstone handling) and GC safety (floor version always preserved; tombstone floor not dropped; above-safe-ts always kept; snapshot at safe_ts after GC returns the floor).

### Added — Phase G real reindex pipeline + SI verification
- **`POST /v1/collections/{name}/reindex`** now accepts a JSON body `{chunker?, re_embed?, text_field?, parent_id_field?, chunk_index_field?, embedding_field?}`. With `chunker` set, the handler groups source records by `parent_doc_id` (configurable), sorts by `chunk_index` (configurable), reconstructs the parent document text, re-chunks via `veda_core::chunking::chunk_markdown`, and inserts the new chunks into the sibling. With `re_embed: true`, embeds each new chunk via the registry's text-modality model. No body / empty body remains the straight-copy behaviour.
- **424 `FAILED_DEPENDENCY`** when `re_embed: true` is requested but no embedding endpoint is configured — explicit failure instead of zero-vector placeholders.
- **`snapshot_isolation.rs`** — Adya anomaly suite: write-skew allowed (SI is not serialisable), lost-update prevented, read-only consistent, abort releases locks.
- **`atomic_reindex.rs`** — kill-mid-reindex recovery: alias unchanged on partial failure; sibling either fully populated or absent.

### Added — Phase H operator surface + live gating
- **Admin REST endpoints**:
  - `GET /v1/admin/region-topology` / `PUT /v1/admin/region-topology` — read or replace the `RegionTopology` stored under the sentinel key in `_meta::shard_ensembles`.
  - `POST /v1/admin/promote-region` — operator-driven region cutover; returns 409 when `new_epoch ≤ current_epoch`.
  - `GET /v1/admin/replica-safe-ts` / `PUT /v1/admin/replica-safe-ts` — read or advance the local replica `safe_ts` (monotonic-forward; sentinel `0` = no gating). Body uses JSON-string encoding for u128.
- **Cross-region reader gating in the live path** — `GET /records/:id` and `POST /search` consult `replica_safe_ts` before serving. When `safe > 0 && read_ts > safe`, they return `503` with `replica behind: read_ts=X exceeds replica_safe_ts=Y by Z` so callers can choose to retry, fall back, or wait.
- **`planned_failover_with_zero_acked_write_loss`** + **`old_primary_fences_after_planned_failover`** — controlled-cutover test confirming the drain + epoch-bump + fence semantics; the stale-primary ship returns `EpochMismatch` after the receiver advances to the new epoch.

### Changed
- `ReadVisibility` enum now derives `Debug + Clone` so test diagnostics print the discriminant.

## [0.2.0] - 2026-05-20

FoundationDB-shape data plane: record writes leave the cluster-wide Raft, the metadata Raft becomes a small "_meta" group, and Phases A through H of `.claude/plans/structured-mixing-turing.md` land in production-integrated form.

### Added — new crates
- **`veda-time`** — Hybrid Logical Clock, `Tso` trait, `LocalTso` (single-node monotonic), `MetaLeasedTso` (talks to `_meta` via a lease grant), `SwappableTso` wrapper (hot-swap behind `Arc<dyn Tso>`).
- **`veda-log`** — Per-shard append-only `ShardLog` with the `[crc32][len][bincode]` framing inherited from `raft_wal`. Group commit (1 ms / 64 KB), 256 MB segment rollover with 4 KB sidecar index. `LogTailer` for gapless replay. BookKeeper-style replication module (`LogEnsemble`, `LogWriter`, `LogFollower`) with epoch fencing. Tonic gRPC transport for cross-process followers AND for WAN log shipping (`ShipBatchRpc` + `safe_ts` gating). Operator-driven `promote_replica` cutover primitive.
- **`veda-admin`** — Offline maintenance CLI scaffold; the migration subcommand is now a no-op stub (Phase E hard-remove dropped the v0 fields the tool used to drain).

### Added — gateway + storage architecture
- **Phase A** — `AppState.tso: Arc<dyn Tso>`; `load_meta_leased_tso` swaps `MetaLeasedTso` in when Raft is up. `GrantTsLease` / `RenewTsLease` `ClusterCommand`s with strict epoch monotonicity.
- **Phase B → C** — Record writes go through `veda-log` instead of `ClusterCommand::InsertRecord`. Per-collection `use_log_primary: Option<bool>` schema flag (env fallback `VEDA_LOG_PRIMARY`). Per-shard `LogApplier` in `veda-storage` that drains a `ShardLog` into the local `SegmentManager`; gateway bootstrap auto-spawns one per shard when `VEDA_LOG_APPLIER=1` and the writer's inline-apply is then skipped to avoid double-apply.
- **Phase E (hard-remove)** — The five v0 `ClusterCommand` variants (`InsertRecord`, `BeginTxn`, `TxnWrite`, `CommitTxn`, `AbortTxn`) and the `records` + `open_txns` fields on `ClusterStateMachine` are removed entirely, along with the gateway's Raft fallback block, the dark-write path, and the `cs.records` reconcile loop.
- **Phase F (MVCC, end to end)** — `MvccStore` with `(user_key, commit_ts)` descending-ts encoding, in-memory `BTreeMap` core, max-versions-per-key cap, safe_ts-driven GC. `MvccWal` per-shard durability with torn-tail-safe `[crc32][len][bincode]` frames; `enable_mvcc_durable(path)` replays on open. Per-doc `commit_ts: Option<u128>` in `GrowingSegment` and a `commit_ts.bin` sidecar in sealed segments. `is_visible_at(doc_id, read_ts)` + `visible_doc_ids_at(read_ts) -> RoaringBitmap` on both. `LocalShardSearcher` (dense / sparse / multi-vector) intersects the visibility bitmap into the `allowed` set BEFORE the index walk runs, so HNSW / IVF-PQ / Flat / DiskANN never spend candidate budget on invisible docs. `?at=<u128>` on `GET /records` and on `POST /search` selects the snapshot timestamp.
- **Phase G** — Real Percolator `TxnCoordinator` (begin/commit/abort) replaces the v0 stub on `/v1/txn`. Background orphan-lock resolver. `seed_records` on `POST /v1/collections` for atomic create-and-load (compensating drop on seed failure). `SetCollectionAlias` `ClusterCommand` + `aliases` map in `_meta` + `POST /v1/collections/{name}/reindex` that creates `name@vN` and atomically swaps the alias.
- **Phase H** — WAN log shipping with `LogShipper` + `LocalReceiver`. `ShipBatchRpc` carries `primary_safe_ts`; replicas advance their local `safe_ts` as batches durably land. `check_read_ts` on the replica `LogReceiver` gates cross-region snapshot reads. `promote_replica(shipper, new_epoch, max_drain_rounds)` operator primitive: drain → refuse-if-incomplete → bump epoch. `RegionTopology` persisted in `_meta::shard_ensembles` under a sentinel key.

### Tests added
- 9 LogApplier unit tests; 5 MVCC WAL unit tests; 5 `mvcc_executor_test` integration tests (incl. sealed-segment seal→reopen→search) + 4 segment-manager MVCC durability tests; 4 atomic-create / rollback tests; 4 reindex / alias tests; 3 mvcc_test gateway tests; 1 log_applier gateway test; 3 wan cutover + lag-recovery tests; 2 tonic_log + 2 tonic_wan round-trip tests; 6 store::tests covering aliases + region topology + ts lease.

### Changed
- `ClusterStateMachine::snapshot()` no longer carries `records` / `open_txns`. Metadata snapshot bounded by schema count, not record count.
- `insert_via_log` is now the only data-plane write path. The non-Raft fall-through that writes directly to `SegmentManager` is kept for single-node deployments without log-primary; multi-node operates via the log path universally.

### Removed
- `dark_write_enabled` and the dark-write blocks. Phase B's `VEDA_LOG_DARK_WRITE` knob existed to mirror writes to the log alongside Raft during pre-Phase-C rollout; with Phase E hard-remove the log IS the only path.
- `migrate_persisted_blob` + `build_persisted_blob_for_state` from `veda-coordinator::raft_storage` (no in-process migration to express now that the v0 fields are gone).
- `PendingTxn` struct from `veda-coordinator::store` (transaction state lives in the Percolator `TxnCoordinator`).
- Gateway per-record post-filter for MVCC visibility in `search.rs` — the executor handles every visibility decision precisely.

### Notes
- The metadata Raft snapshot for a workload of 100k collections + 10M records is now bounded under 100 MB (was unbounded; previously grew linearly with record count).
- Backwards-compatible on-disk format: pre-MVCC sealed segments stay byte-identical (no `commit_ts.bin` written when no doc was stamped); pre-Phase-F `raft_state.bin` no longer loads (rebuild from logs or use the pre-Phase-E binary to migrate first).

## [0.1.0] - 2026-04-07

### Added
- Core vector database engine with 7 Rust crates
- HNSW index with adaptive filtered search (95.5% recall@10)
- DiskANN (Vamana graph) index (99.9% recall@10)
- IVF-PQ index with residual product quantization
- RaBitQ and scalar quantization
- SIMD-optimized distance functions (AVX2 + scalar fallback)
- 6 distance metrics: Cosine, Euclidean, DotProduct, Hamming, Manhattan, Jaccard
- mmap-backed vector storage format
- Scalar indexes: inverted, range, bitmap
- Growing + sealed segment architecture with compaction
- SQL-like filter expression parser and evaluator
- Multi-vector fusion search (RRF, WeightedSum, Intersection)
- Distributed query with shard fan-out and merge
- Suffix array pattern search with fuzzy matching
- REST API gateway (axum) with full CRUD
- Auto-embedding service with modality validation
- Blob storage with MinIO/S3-compatible and local filesystem backends
- JWT authentication and RBAC (Admin/Editor/Viewer)
- Full observability stack: Prometheus metrics (16 metric types), OpenTelemetry tracing and log export, health/readiness endpoints, query explain with per-phase timing
- Pre-built Grafana dashboard with provisioned datasources (Mimir, Tempo, Loki)
- Docker Compose `monitoring` profile (OTel Collector, Prometheus, Mimir, Tempo, Loki, Grafana)
- W3C Trace Context propagation for incoming traceparent headers
- HTTP request metrics middleware with automatic error counting
- `#[instrument]` tracing spans on handlers, embedding client, blob store, segment manager, and search executor
- Cluster coordination with shard routing
- WAL abstraction with in-memory backend
- Niriksha dashboard with 7 pages (React + TypeScript)
- Python SDK (veda-client)
- 7 configurable chunking strategies: MarkdownRecursive (default), MarkdownHeading (levels 1/2/3), Semantic, Sentence, FixedSize, Page, None
- Semantic chunking strategy with word-overlap Jaccard similarity for topic-based splitting
- Markdown heading-level chunking (split on H1, H2, or H3 depth)
- FlashParse OCR integration with Docker Compose profiles (`docker compose --profile ocr up`)
- End-to-end document processing pipeline (parse, chunk, embed, insert) via `/v1/collections/{name}/process-document` and `/v1/collections/{name}/process-documents`
- `.env.example` configuration template for VLM and service settings
- Docker and Docker Compose deployment
- ANN benchmark suite
- Comprehensive integration and stress tests

<h1 align="center">Veda</h1>

<p align="center">
  <strong>A next-generation multimodal vector database for the era of RAG</strong>
</p>

<p align="center">
  <a href="#features">Features</a> |
  <a href="#quick-start">Quick Start</a> |
  <a href="#api-examples">API</a> |
  <a href="#python-sdk">Python SDK</a> |
  <a href="#niriksha-dashboard">Niriksha</a> |
  <a href="#benchmarks">Benchmarks</a> |
  <a href="#monitoring--observability">Monitoring</a> |
  <a href="CONTRIBUTING.md">Contributing</a>
</p>

---

## Features

### Storage + retrieval
- **Multimodal native** -- first-class support for text, image, audio, video, and PDF embeddings
- **4 ANN index types** -- HNSW, DiskANN (Vamana graph), IVF-PQ, and Flat (brute-force)
- **Quantization indexes** -- RaBitQ (1-bit), RaBitQ+rerank, RaBitQExt (4-bit rerank), plus standalone `Sq4` (4-bit/dim) and `Sq8` (8-bit/dim) scalar quantization -- memory-efficient compression with minimal recall loss
- **SIMD-optimized distances** -- SimSIMD-backed AVX-512 / ARM NEON / SVE selected at runtime (AVX2/scalar fallback via `--no-default-features`); 6 metrics (Cosine, Euclidean, DotProduct, Hamming, Manhattan, Jaccard)
- **SQL-like filter expressions** -- combine vector search with structured metadata filters, including dotted-path queries into nested JSON fields (e.g. `meta.user.tier = "gold"`)
- **Multi-vector fusion** -- RRF, WeightedSum, and Intersection strategies for cross-modal retrieval
- **Full-text BM25 + analyzer** -- Lucene-style BM25 scoring with a configurable analyzer pipeline (char filters, tokenizer, **stemming** + **stopword** removal) applied identically at index and query time. The scalar filter DSL and row-level security apply to BM25, `match_phrase`, and hybrid queries (not just the dense path)
- **Phrase / proximity queries** -- positional postings power `match_phrase` (exact-adjacency and slop-N proximity), BM25-ranked
- **Hybrid search** -- fuse dense vectors, sparse SPLADE vectors, and BM25 full-text in one query (RRF / weighted)
- **Segment compression** -- sealed stored-field columns are `zstd`-compressed at seal time (the mmap'd vector `.vec` fast path is deliberately left uncompressed)
- **Pattern search** -- suffix array-backed substring and fuzzy matching on text fields
- **Built-in blob storage** -- store raw files alongside embeddings via MinIO/S3-compatible backends
- **Record TTL** -- set a reserved `_expires_at` (unix seconds) field on any record; expired records are excluded from search/get/list at query time and physically purged by a background sweeper
- **Analytics aggregations** -- OLAP-style `count`/`sum`/`avg`/`min`/`max` with optional `group_by`, plus numeric histograms and per-field statistics, computed over the columnar store and honoring the filter DSL

### Distributed data plane (FoundationDB-shape)
- **Per-shard append-only log** (`veda-log`) -- record writes go through `veda-log`, not the metadata Raft. BookKeeper-style quorum replication (E, Qw, Qa) with epoch fencing; Tonic gRPC transport for cross-process followers. BookKeeper open-and-recover truncate at writer handover; `rebuild_ensemble` swaps out failed nodes; `ShardLog::copy_to` ships a snapshot to a joining follower.
- **`FollowerHealthMonitor`** -- edge-triggered probe loop drives ensemble change on follower failure (one orchestration call per outage, not per probe interval).
- **Hybrid Logical Clock + leased TSO** (`veda-time`) -- monotonic commit timestamps issued by a lease holder named in `_meta`. `MetaLeasedTso` swaps in automatically when Raft is configured.
- **MVCC versioning** -- every shard tracks per-doc `commit_ts` from growing through sealed segments. Reads at `?at=<ts>` (raw u128 HLC or relative expression like `now-1h`) return the snapshot view; the executor filters superseded / future versions inside the search walk. Every index (HNSW, Flat, IVF-PQ, DiskANN) has a native `search_filtered` that consults the visibility bitmap inside the candidate walk -- no 10×-oversample post-filter, no recall collapse on sparse snapshots. Disk-durable via per-shard MVCC WAL with periodic atomic checkpoint (snapshot + WAL truncate).
- **Percolator snapshot-isolation transactions** -- begin/commit/abort on `/v1/txn`, primary-and-secondary locks on the per-shard log, commit record in `_meta`. Replaces the v0 single-Raft txn stub. Background orphan-lock resolver included. Adya anomaly suite (write-skew allowed, lost-update prevented, read-only consistent) verifies SI invariants.
- **Atomic create-and-load** -- `POST /v1/collections` accepts `seed_records`; insert failure rolls back the collection via a compensating drop.
- **Real reindex pipeline** -- `POST /v1/collections/{name}/reindex` accepts `{chunker, re_embed, ...}`. Groups source records by `parent_doc_id`, sorts by `chunk_index`, reconstructs parent text, re-chunks via the supplied strategy, and (when `re_embed: true`) re-embeds via the registry's text-modality model. The atomic alias swap finalises the cutover at commit_ts. `SetCollectionAlias` is the same primitive underneath -- a name can be pointed at a sibling (e.g. `docs` → `docs@v2`) atomically. Returns `424 FAILED_DEPENDENCY` when `re_embed` is requested with no embedding endpoint configured.
- **Cross-region WAN log shipping** -- async replication of per-shard logs to replica regions with bounded staleness. Cross-region reads call `check_read_ts` on the live `/records` and `/search` path; reads ahead of the replica's `safe_ts` return `503` with explicit lag information. Operator-driven `promote_replica` cutover drains pending writes then bumps the topology epoch (fences old primaries). Admin REST: `GET/PUT /v1/admin/region-topology`, `POST /v1/admin/promote-region`, `GET/PUT /v1/admin/replica-safe-ts`.
- **True multi-node distributed search** -- a gRPC search service (`RemoteShardSearcher`) does scatter-gather across shards with a query-then-fetch split: Phase 1 returns lightweight `ScoredRecord`s for the global top-k merge, Phase 2 fetches the full records from the owning node. Per-shard placement comes from `route_query`; local and remote shards interleave behind the same `ShardSearcher` trait.

### Pipelines + tooling
- **7 chunking strategies** -- MarkdownRecursive, MarkdownHeading (3 levels), Semantic, Sentence (with CJK punctuation support), FixedSize, Page, None
- **FlashParse OCR integration** -- end-to-end document processing with configurable chunking
- **Auto-embedding pipeline** -- automatic embedding generation with modality validation
- **Niriksha dashboard** -- built-in visualization and exploration UI for multimodal vector data
- **JWT authentication + RBAC** -- Admin, Editor, and Viewer roles with token-based auth
- **Row-level security** -- a collection may declare an `rls_owner_field`; the gateway then AND-injects `owner == <jwt-subject>` into every read filter and enforces owner-on-write at the AST level, so a user sees and mutates only the records they own (Admin bypasses)
- **Scheduled jobs** -- cron-style admin scheduler (`/v1/admin/jobs`) with a typed job-kind enum (`TtlSweep`, `Reindex`, `Compact`, `Evaluate`); schedule persists across restart. `TtlSweep` is wired to the record-TTL sweeper; the other kinds are registered hooks pending their dispatch wiring
- **Full observability stack** -- Prometheus metrics, OpenTelemetry tracing, query explain, health checks, and a pre-built Grafana dashboard
- **Python SDK** -- `veda-client` package for idiomatic Python access
- **`veda-admin` CLI** -- offline maintenance tool (currently scaffold)

### Agents & MCP
- **Agent framework** -- build conversational agents that ground answers in your data. An agent is `{ instructions, tool_ids, model? }`; a built-in `veda-agent` works out of the box. `POST /v1/agents/{id}/converse` runs a multi-step tool-calling (ReAct-style) loop over your configured LLM and returns the final answer plus a full `steps[]` trace (reasoning / tool_call / tool_result). CRUD at `/v1/agents`, history at `/v1/conversations`.
- **Tool catalog** -- 8 built-in tools agents (and MCP clients) can call: `search` (multimodal — text **or** image query → cross-modal hits, with field projection), `aggregate`, `filter_query`, `get_record`, `describe_collection`, `list_collections`, `route`, `upsert_record`. Tools are also directly invocable via `POST /v1/agent-tools/{id}/execute`; auth + tenant + row-level security apply to every tool call.
- **MCP server** -- full Model Context Protocol server (JSON-RPC 2.0, protocol `2025-06-18`) at `/v1/mcp`, exposing the whole tool catalog with `readOnlyHint`/`destructiveHint` annotations, `structuredContent` results, and correctable `isError` tool errors. Two standard transports: **stdio** (via the `veda-mcp` bridge) and **stateless Streamable HTTP** — so Claude Desktop, Cursor, VS Code, and `mcp-remote` connect directly. Veda's multimodal `search` tool gives MCP clients cross-modal retrieval no text-only vector DB can.
- **A2A (Agent-to-Agent)** -- expose a Veda agent as a peer: `GET /v1/a2a/{id}/card` returns an A2A AgentCard (skills = the agent's tools) and `POST /v1/a2a/{id}` handles JSON-RPC `message/send`, so external orchestrators can drive a Veda agent with no custom API.

See [documentation/2026-06-13-agents-and-mcp.md](documentation/2026-06-13-agents-and-mcp.md) for the full agentic + MCP guide.

### Ops Advisor (Insights)
- **Automated issue detection** -- a background advisor samples an operational snapshot (per-collection segments/records, compaction state, nodes, replica lag, health, query latency, RSS, request/error counters, node resources, log content, SLOs) and evaluates a catalog of **21 threshold/correlation rules plus 7 EWMA/σ anomaly detectors** spanning operational health (compaction backlog, seal stalled, high search latency, slow-query surge, high error rate, embedding/dependency down, replica lag, memory pressure, node unhealthy), **node resources** (disk-space-low, high-CPU, high-load, fd-exhaustion, system-memory-low), **log content** (log error rate, new error category), and **SLO/reliability** (burn-rate fast/slow, error-budget exhausted, synthetic-probe failing). Each detected condition opens an **event** with a severity (High/Medium/Low), a root-cause **detail**, and a concrete **recommendation**; events **auto-close** when the condition clears.
- **Anomaly detection** -- EWMA/σ baselines over search latency, error rate, insert rate, RSS, CPU, disk usage, and error-log rate flag deviations beyond a configurable sigma once warmed up.
- **Alerting** -- on event open/close transitions, notify via **generic webhook, Slack, PagerDuty (Events API v2), and email (SMTP)** — async, severity-routed, per-channel failure-isolated. Channel secrets are write-only (redacted on read).
- **Opt-in auto-remediation** -- per rule (off by default), the advisor can trigger a safe action (e.g. compaction) on an open event, rate-limited per scope.
- **REST + UI** -- admin API at `/v1/admin/insights` (open events), `/insights/history`, `/insights/rules` (+ `PUT` to tune thresholds/severity/mute/auto-remediate), `/insights/channels`, `POST /insights/run`, `POST /insights/{id}/ack`; plus a Niriksha **Insights** page (event list + detail drawer + rule settings + channel config) with a sidebar open-event badge.

See [documentation/2026-06-14-autoops-ops-advisor.md](documentation/2026-06-14-autoops-ops-advisor.md) for the full Ops Advisor guide.

### Log Insights
- **In-process log capture** -- a `tracing` layer captures Veda's own logs (≥ `VEDA_LOG_CAPTURE_LEVEL`, default WARN) into a bounded in-memory buffer with structured fields and best-effort `trace_id`/`span_id` correlation (when an OTLP endpoint is configured); query logs also carry `trace_id`.
- **Semantic log categorization** -- recent WARN/ERROR messages are embedded and clustered by cosine similarity into categories (the vector-DB differentiator over template-only grouping), with a normalized-template fallback when no embedder is configured. `GET /v1/admin/logs/categories`.
- **Log REST API + Niriksha Logs page** -- `GET /v1/admin/logs` (live tail, level + text filters), `GET /v1/admin/slow-queries` (server-backed, from the per-collection query log), `GET /v1/admin/logs/categories`; plus a Niriksha **Logs** page (tail + level filter + text search + trace_id + categories + slow queries).
- **Log-content alerting** -- the Ops Advisor gained log rules: `log_error_rate` (ERROR-log rate), `anomaly_error_logs` (baseline/σ spike), and `new_error_category` (a newly-appeared error category) — flowing through the same events + alert channels.
- **Logs-as-a-collection** -- `POST /v1/collections/{name}/ingest-logs` parses raw log lines (JSON + plain-text) into records (message + level + ts + optional `_expires_at` TTL, auto-embedded when the collection has a vector field), turning Veda into a **searchable log archive** (BM25 + semantic + filter + retention).

See [documentation/2026-06-15-log-insights.md](documentation/2026-06-15-log-insights.md) for the full Log Insights guide.

### Node Resource Signals
- **Host/process resources in the snapshot** -- the Ops Advisor snapshot gained best-effort node-resource signals, read dependency-free from `/proc` (Linux) + `statvfs`: data-dir **disk** total/free/used%, **CPU** utilization (from `/proc/stat` jiffy deltas), **load** average (1/5/15 + core count), open **file descriptors** vs the `RLIMIT_NOFILE` limit, and **system memory** total/available. Every signal is optional and degrades to `null` off-Linux (no false alerts).
- **Resource rules** -- `disk_space_low` (High, data-dir used % > 90 — warns *before* writes/seals/compaction fail when the disk fills), `high_cpu` (> 85%), `high_load` (load1/core > 2.0), `fd_exhaustion` (open/limit > 0.8), and `system_memory_low` (available/total < 10%); plus anomaly detectors `anomaly_cpu` and `anomaly_disk_used` (EWMA/σ, catches a rapid fill). Each skips cleanly when its signal is absent, and all flow through the existing events + alert channels and are tunable via `/v1/admin/insights/rules`.
- **REST + UI** -- `GET /v1/admin/resources` (admin-only) returns the local node's latest resource snapshot (`disk`, `cpu_pct`, `load`, `num_cpus`, `fds`, `memory`, `rss_bytes`); plus a **Resources** panel on the Niriksha System Health page (disk used %/free, CPU, load, fds, system memory, RSS — warning-colored past threshold).

See [documentation/2026-06-16-node-resource-signals.md](documentation/2026-06-16-node-resource-signals.md) for the full Node Resource Signals guide.

### Service Reliability (SLOs)
- **SLOs with error budgets** -- define service-level objectives (`{ kind, target_pct, window_secs, fast_burn, slow_burn }`) for **latency** (% of queries under a threshold), **availability**, **error rate**, or a **synthetic** probe. Each advisor tick records good/total into a persisted 60s-bucketed rolling counter (up to 30 days) and computes `sli_pct`, `error_budget_remaining_pct` (can go negative = over budget), and **Google-SRE multi-window burn rate** (`burn_rate = (bad/total)/allowed_bad`; 1.0 = exactly on budget) over the fast and slow windows. Three rules — `slo_burn_rate_fast` (High), `slo_burn_rate_slow` (Medium), `slo_budget_exhausted` (High) — open events through the same events + alert channels.
- **Synthetic self-checks** -- in-process probes that run at the start of each advisor tick: `health` (the readiness dependency checks), `search` (the internal query executor, top-1), and `embed` (the async embedding client). Results feed a per-probe uptime/consecutive-failure summary, the `synthetic` SLO SLI, and a `synthetic_failing` rule (High, fires after K consecutive failures, default 3). No external load generator or browser — Veda checks its own critical paths.
- **AI ops triage** -- `POST /v1/admin/insights/{event_id}/triage` runs a built-in read-only **`veda-ops`** agent over the event: it autonomously calls 4 ops tools (`get_insights`, `get_logs`, `get_resources`, `get_slo_status`) and returns `{ explanation, steps[] }` (a root cause + next steps with a full reasoning/tool trace). Returns **503** when no LLM is configured (`VEDA_LLM_ENDPOINT` unset).
- **REST + UI** -- admin API at `/v1/admin/slos` (+ `/slos/{id}`, `/slos/{id}/status`), `/v1/admin/synthetics` (+ `/synthetics/{id}`, `/synthetics/{id}/results`), and `POST /v1/admin/insights/{event_id}/triage`; plus a Niriksha **SLOs** page (SLI %, error-budget gauge, fast/slow burn) in the Observe nav, a **Synthetics** panel on System Health, and a **"Triage with AI"** button on the Insights event drawer.

See [documentation/2026-06-16-service-reliability-slo.md](documentation/2026-06-16-service-reliability-slo.md) for the full Service Reliability guide.

### APM & Performance Observability
- **Distributed tracing + sampling + dependency spans** -- the OTel pipeline takes a configurable sampler (`OTEL_TRACES_SAMPLER` / `OTEL_TRACES_SAMPLER_ARG`, default `parentbased_traceidratio` @ 1.0) and **injects W3C trace context into outbound calls** (LLM / embedding / rerank HTTP + remote-shard gRPC metadata), so a Veda trace stitches into the calling app's and out to its dependencies. The previously-missing dependency spans (remote-shard gRPC, rerank, local-shard, vector-index search, NATS) are now emitted, and the Python SDK propagates context too (`veda-client[otel]`, no-op without OTel).
- **On-demand CPU profiling** -- `GET /v1/admin/pprof?seconds=&frequency=&format=` samples the running server and returns a flamegraph **SVG** or **pprof protobuf** (409 if one is already running; empty when idle by design). Behind a `profiling` cargo feature, so default builds carry no `pprof` dependency (`cargo build -p veda-server --features profiling`).
- **Agent / GenAI-semconv tracing + RunMetrics** -- the agent loop emits OTel **GenAI semantic-convention** spans: `invoke_agent` → `chat {model}` (`gen_ai.system`, `gen_ai.request.model`, `gen_ai.usage.*_tokens`, `gen_ai.response.finish_reasons`) → `execute_tool {id}` (`gen_ai.tool.*`), so RAG/agent traces drop natively into Datadog / Grafana / Phoenix / Langfuse. Each run also returns `metrics { total_ms, llm_ms, tool_ms, steps, prompt_tokens, completion_tokens, total_tokens }` (per-step `duration_ms` on tool results).
- **Cost / token observability** -- Prometheus `veda_llm_tokens_total{type,model}`, `veda_agent_run_duration_seconds{model}`, and `veda_llm_cost_usd_total{model}` (accumulated in **micro-USD** — divide by 1e6). Pricing via `VEDA_LLM_PRICES` JSON (`{"model":{"input":x,"output":y}}` per-1k-tokens) else a built-in default; an unpriced model still emits tokens and just skips cost (never fails).
- **Slow-query / error correlation** -- `GET /v1/admin/correlations` groups the query-log window by collection / query_type / has_filter / reranked / index_type / result_count + top_k buckets and reports per-bucket count, p50, p95, slow_count, and error_rate, ranked worst-first — so you can see *which kind of query* is slow or failing. The query log gained the enriching fields and now **logs errors too** (was success-only), independent of trace sampling.
- **Retrieval-quality traces + dependency map** -- the search span carries `db.system=veda`, `db.collection`, and `retrieval.query_type`/`top_k`/`result_count`/`reranked`/`has_filter`/`index_type` (and the `ExplainTiming` `filter_ms`/`ann_search_ms`/`scoring_ms`/`fusion_ms`/`total_ms` under `explain`); `GET /v1/admin/depmap` returns a declarative 9-node/8-edge service graph.
- **ANN index advisor** -- a low-cadence proxy **recall probe** (`VEDA_RECALL_PROBE_INTERVAL_SECS`, default 600) compares ANN top-k against an exact brute-force scan over a sampled subset → `recall@k` (`is_proxy=true`) + p95, exported as `veda_index_recall{collection}` / `veda_index_p95_ms{collection}`. Rules `index_recall_degraded` (High, `recall < 0.90`) and `index_latency_high_for_recall` (Medium, `p95_ms > 50` at `recall >= 0.95`) open events with directional recommendations (raise/lower efSearch, reindex, or switch index type). `GET /v1/admin/index-advisor` surfaces per-collection index_type/params/p95/recall/recommendations/open events; the Niriksha **IndexAdvisor** panel adds opt-in reindex.

See [documentation/2026-06-17-apm-observability.md](documentation/2026-06-17-apm-observability.md) for the full APM & Performance Observability guide.

### RAG & AIOps Intelligence
- **Answer citations + faithfulness/relevancy** -- every agent run is scored for grounding. `ConverseResponse` (and the triage / A2A responses) carry `citations: [{ chunk_id, collection, snippet, score }]` and `grounding: { faithfulness, answer_relevancy, method }` (0–1). Faithfulness is **LLM-as-judge** (RAGAS-style) with an **embedding-overlap** fallback when no LLM is configured, and `no_context` when the run did no retrieval. Scoring is post-run, best-effort, and never fails the answer. Metrics `veda_agent_faithfulness` / `veda_agent_citations`.
- **Retrieval & faithfulness quality SLOs** -- `SloKind` gains **`recall`** (`{ collection, k? }`, SLI from the ANN recall probe — proxy) and **`faithfulness`** (`{ agent? }`, SLI from a persisted faithfulness sample of converse turns), so retrieval quality and answer grounding get the same multi-window burn-rate / error-budget treatment as latency and availability — at `/v1/admin/slos`.
- **RAG/agent eval harness + flywheel** -- define `eval-sets` of cases `{ query, expected_chunk_ids?, gold_answer? }`; `POST /v1/eval-sets/{id}/run` reports per-case + aggregate **recall@k, faithfulness, answer-relevancy, and pass_rate**, `GET /v1/eval-sets/{id}/runs` lists past runs, and `POST /v1/eval-sets/{id}/cases/from-conversation` graduates a production trace into a regression case (the flywheel). 503 without an LLM. A Niriksha **Eval Harness** page renders it.
- **Vector-backed agent memory** -- opt-in `Agent.memory` (default off; existing agents unchanged). When enabled, each turn is embedded into a per-tenant `_agent_memory` collection and the most semantically-similar prior turns are retrieved and injected (bounded) into the system prompt, giving the agent recall across conversations. Best-effort and off the request lock.
- **Streaming `/converse` over SSE** -- `POST /v1/agents/{id}/converse/stream` emits `event: reasoning | tool_call | tool_result | answer` deltas as the loop runs, then a final `event: done` carrying `{ conversation_id, response, metrics, citations, grounding }`. The blocking `/converse` is unchanged and A2A stays sync. 503 without an LLM.
- **Semantic incident correlation & dedup** -- ops-event title+detail are embedded and clustered by cosine into `Incident { representative, member_event_ids, rule_ids, count, max_severity, first_seen, last_seen }` (rule_id+scope fallback without an embedder), collapsing a storm of related events into one incident. Off-tick runner (`VEDA_INCIDENT_INTERVAL_SECS`, default 300). `GET /v1/admin/incidents`; a Niriksha Insights **"Incidents"** tab.
- **Capacity forecasting** -- a least-squares trend over the snapshot history projects threshold-crossing ETAs (disk-full date, memory exhaustion, recall decay) with warm-up + min-R² guards so it never fires on a flat/noisy series. Rules `disk_full_forecast` (High), `memory_growth_forecast` (Medium), `recall_decay_forecast` (Medium). `GET /v1/admin/forecast`; a Niriksha forecast panel.
- **Cross-signal RCA bundle** -- `GET /v1/admin/insights/{event_id}/rca` joins the correlated logs + slow/errored queries + traces (by `trace_id` + the event time window) + the nearest resource snapshot + the related incident into one bundle; a `correlate_incident` ops tool lets the `veda-ops` triage agent pull it, and the triage prompt is enriched with a bounded RCA summary. Niriksha event-drawer **"Root-cause bundle"**.

See [documentation/2026-06-18-rag-aiops-intelligence.md](documentation/2026-06-18-rag-aiops-intelligence.md) for the full RAG & AIOps Intelligence guide.

### Agentic AIOps
Where Elastic's AIOps is statistical and template-based over text, Veda's is **semantic and embedding-based over every signal it owns**.
- **Change-point detection** -- `advisor/changepoint.rs` detects sustained **LEVEL** shifts and **TREND** breaks (not single-point anomalies) in the advisor metric series (`disk_used_pct`, `rss_bytes`, query `p95`, error-log count, per-collection `recall`) via a least-squares / two-window statistic with a well-fit gate + warm-up (≥20 pts) + min-magnitude (1.5σ) guard, so it never fires on a flat/noisy series. Computed in-tick, attached to `OpsSnapshot.change_points`; rule `change_point_detected` (Medium). `GET /v1/admin/change-points`.
- **Semantic spike root-cause attribution** -- when the error/warn-log rate spikes, `advisor/spike_attribution.rs` attributes it to the semantic **categories** driving it by reusing the embedding `LogCategorizer` — `{ window, baseline, total_delta, drivers: [{ category, representative, spike_count, baseline_count, growth, share_of_growth, is_new }], method }`. The vector-native answer to a statistical "log rate analysis": it names *which meaning-cluster* drove the spike. Off-lock embedding; normalized-template fallback (`method="template"`) without an embedder. `GET /v1/admin/log-rate-analysis?window_secs=&baseline_secs=&min_level=`.
- **Proactive incident RCA drafts + noise reduction** -- on a newly-correlated incident (off-tick), the `veda-ops` agent auto-runs over the incident's cross-signal RCA bundle to draft a root-cause paragraph + runbook step into `Incident.draft` (best-effort, 503-safe → None without an LLM, new-incident-only). Notifications fire per **incident** not per event ("N alerts → 1 incident", deduped via persisted notified-ids), and `/v1/admin/incidents` gains a `noise_reduction { event_count, incident_count, suppressed }` summary.
- **Incident novelty detection** -- reuses the incident-clustering embeddings to flag genuinely **unprecedented** incidents — `Incident.is_novel` + `novelty_score = 1 − max_cosine_to_history` against a persisted embedding history (warm-up ≥5 historical incidents; embedder-free skip so no false novelty). Novel incidents escalate the per-incident notification ("⚠ UNPRECEDENTED pattern", severity → High). Surfaced on `/v1/admin/incidents` + a Niriksha badge.

See [documentation/2026-06-18-agentic-aiops.md](documentation/2026-06-18-agentic-aiops.md) for the full Agentic AIOps guide.

### LLM Observability & Optimization
Veda turns its vector engine into the LLM layer itself — it doesn't just *observe* LLM apps, it **optimizes and secures** them, which Elastic LLM Observability can't (no faithfulness, eval, or semantic caching).
- **Semantic response cache** -- opt-in `Agent.cache` (default off; existing agents unchanged). Before a run the prompt is embedded and vector-searched against a per-tenant `_llm_cache` collection for a near-duplicate (cosine **≥ 0.97**); on a hit the cached response is returned **without calling the LLM** (`cache_hit:true`, **0 steps**), on a miss the agent runs and the answer is stored. Pure vector-native. Metrics `veda_llm_cache_hits_total` / `veda_llm_cache_misses_total`.
- **Embedding-based safety guardrails** -- opt-in `Agent.guardrails` (default off). Input *and* output are screened for **prompt-injection / jailbreak / PII / unsafe** content via a PII regex plus max-cosine (**≥ 0.80**) to a curated bad-pattern embedding set, in mode **Flag** (annotate) or **Block** (short-circuit the LLM with a refusal). A `GuardrailReport` rides on the response; metrics `veda_llm_guardrail_blocked_total` / `veda_llm_guardrail_flagged_total`. Embedder-free → regex-only; fail-open (a guardrail error never fails the request).
- **Searchable semantic LLM trace store + failure clustering** -- opt-in `Agent.trace` (default off). Each run is written to a per-tenant `_llm_traces` collection (prompt / response / tool_ids / faithfulness / cost_usd / cache_hit / guardrail_flagged / latency_ms / ts + an embedding of prompt+response). `GET /v1/admin/llm-traces?q=` does **semantic search** over traces; `GET /v1/admin/llm-traces/clusters` clusters failing traces (reusing the embedding categorizer) into recurring failure modes. A Niriksha panel renders both.
- **TTFT + cost budgets + quality drift + feedback** -- `RunMetrics` gains `ttft_ms` (stream-only time-to-first-token; metric `veda_llm_ttft_seconds`); advisor rules `llm_cost_budget` (Medium, USD-per-window off the cost counter) and `quality_drift` (Medium, a sustained faithfulness drop via the two-window / change-point statistic over the faithfulness series); and `POST /v1/conversations/{id}/feedback { rating: up|down, note? }` (metric `veda_llm_feedback_total{rating}`).

See [documentation/2026-06-19-llm-observability.md](documentation/2026-06-19-llm-observability.md) for the full LLM Observability & Optimization guide.

### Web Connectors & Agentic Web Search
The in-lane, differentiating answer to Elastic's data-source connectors: rather than chasing 30+ SaaS connectors, Veda does **web-native RAG ingestion + agentic web search** over open-source, self-hosted services (crawl4ai + searxng) — privacy-respecting, **no external API**. Both services are **opt-in** (endpoints unset → clean "not configured"/503, no behavior change).
- **Web ingestion (crawl4ai)** -- `POST /v1/collections/{name}/ingest-url { url|urls, deep_crawl?, max_pages?, chunker?, ttl_secs? }` crawls a page (or a bounded deep crawl) via a `Crawl4aiClient` (`POST {VEDA_CRAWL4AI_ENDPOINT}/md` → clean markdown), then runs the shared **`ingest_markdown`** path (chunk → embed → insert) that `process-document` was refactored to reuse (no OCR regression). Records carry `source_url` / `title` / `chunk_index`. Tenant/RLS-scoped; **503** when crawl4ai is unconfigured. Fetches are **SSRF-guarded** — private/internal/reserved addresses (loopback, RFC1918, link-local incl. `169.254.169.254`) are blocked by default with **403 `URL_BLOCKED_PRIVATE`** (tunable via `VEDA_WEB_ALLOWLIST`/`VEDA_WEB_BLOCKLIST`/`VEDA_WEB_ALLOW_PRIVATE`).
- **Agentic web search (searxng)** -- two opt-in agent ToolCatalog tools — `web_search { query, limit }` (searxng results via `GET {VEDA_SEARXNG_ENDPOINT}/search?format=json`) and `fetch_url { url }` (crawl4ai markdown, truncated) — let an agent **search → fetch → ground** over the live web. A built-in **`veda-web`** agent whitelists `web_search` + `fetch_url` + `search`; the web tools are **excluded from the default `veda-agent`** (the catalog has 15 builtins, the default agent resolves to 13).
- **Connectors framework + Niriksha page** -- `Connector { id, kind: web|websearch, name, config, enabled }` persisted via SavedObjects (tenant-scoped). CRUD `GET`/`POST /v1/admin/connectors`, `GET`/`PUT`/`DELETE /v1/admin/connectors/{id}`, and `POST /v1/admin/connectors/{id}/run` (a Web connector crawls its configured URLs into its target collection). Niriksha `Settings.tsx` gained a **Data Source Connectors** CRUD section ("Crawl now").
- **Deployment** -- a new `docker compose --profile web` brings up `crawl4ai` (`unclecode/crawl4ai`) and `searxng` (`searxng/searxng`); `VEDA_CRAWL4AI_ENDPOINT` / `VEDA_SEARXNG_ENDPOINT` (default empty). crawl4ai requires **`VEDA_CRAWL4AI_TOKEN`** to be reachable — it binds loopback-only and is unreachable from Veda when the token is unset. License posture: crawl4ai is **Apache-2.0**; searxng is **AGPL-3.0** but is only HTTP-clienteed as a separate, self-hosted service — its network-copyleft binds modifications to searxng itself, not an API client, so there is no contamination of Veda's Apache-2.0.

See [documentation/2026-06-19-web-connectors.md](documentation/2026-06-19-web-connectors.md) for the full Web Connectors & Agentic Web Search guide.

## Quick Start

### Docker Compose (recommended)

```bash
# Copy and configure environment
cp .env.example .env

# Core services (Veda + MinIO + NATS)
docker compose up

# With embedding service (HuggingFace TEI)
docker compose --profile embedding up

# With document OCR (FlashParse + GLM-OCR, text-only, ~8GB VRAM)
docker compose --profile ocr up

# With full OCR + VLM image captioning (~48GB+ VRAM)
docker compose --profile ocr --profile ocr-vlm up
```

This starts the Veda server, MinIO blob storage, and NATS message queue.

- Veda API: `http://localhost:6340`
- Niriksha dashboard: `http://localhost:5173`
- MinIO console: `http://localhost:9001`

**Optional profiles:**

| Profile | Services | GPU VRAM | Purpose |
|---|---|---|---|
| `embedding` | HuggingFace TEI + Multimodal Embedding (vLLM) | CPU + ~10GB | Text embedding (BGE-M3) + multimodal embedding (Qwen3-VL) |
| `ocr` | FlashParse + GLM-OCR | ~8GB | Text/table/formula OCR from documents |
| `ocr-vlm` | VLM (Qwen3.5-35B-A3B) | ~48GB | Image captioning and classification (use with `ocr`) |
| `monitoring` | OTel Collector, Prometheus, Mimir, Tempo, Loki, Grafana | CPU | Metrics, tracing, and log aggregation |

### Build from source

```bash
# Build the Rust engine
cargo build --release

# Run the server
cargo run --release --bin veda-server

# (Optional) Build the Niriksha dashboard
cd niriksha && npm install && npm run build
```

## Architecture

```
                  +-----------+
                  |  Niriksha |   (React dashboard)
                  +-----+-----+
                        |
                  +-----v-----+
 Clients -------> |  Gateway   |   REST, auth, MVCC reads at ?at=<ts>
                  +--+--+--+---+
                     |  |  |
       +-------------+  |  +-------------+
       |                |                |
       v                v                v
  +---------+    +--------------+   +------------------+
  | TSO/HLC |    |  _meta RAFT  |   |  PER-SHARD LOGS  |
  | veda-   |    |  schema /    |   |  veda-log        |
  | time    |    |  shard plan /|   |  - quorum write  |
  | leased  |    |  TSO lease / |   |    (E, Qw, Qa)   |
  | alloc.  |    |  txn commits |   |  - tonic gRPC    |
  +---------+    |  / aliases / |   |  - epoch-fenced  |
                 |  RegionTopo  |   |  - WAN ship      |
                 +------+-------+   +---------+--------+
                        |                     |
                        +----------+----------+
                                   v
                  +----------------------------------+
                  |  STORAGE / APPLIER (veda-storage)|
                  |  - LogApplier tails shard logs   |
                  |  - growing segment (commit_ts)   |
                  |  - sealed segment (commit_ts     |
                  |    sidecar) + MVCC WAL durable   |
                  |  - executor filters by read_ts   |
                  +----------------------------------+
```

The metadata Raft (`_meta`) carries only schemas, the shard plan, nodes, the TSO lease, txn commit records, collection aliases, and the region topology — NOT record bodies. Record writes go to per-shard `veda-log`; reads go to local `SegmentManager`s with per-doc `commit_ts` filtering.

**Crates:**

| Crate | Role |
|---|---|
| `veda-gateway` | REST API (axum), authentication, MVCC snapshot reads, txn endpoints, LogApplier bootstrap |
| `veda-query` | Query planning, multi-vector fusion, distributed fan-out, executor-side visibility filter |
| `veda-index` | ANN indexes (HNSW, DiskANN, IVF-PQ, Flat), scalar indexes — visibility-aware `search_filtered` |
| `veda-storage` | Vector storage, blob storage, growing + sealed segments with `commit_ts`, MVCC store + WAL, `LogApplier` |
| `veda-core` | Shared types, distance functions, filter expressions, schema (incl. `use_log_primary` / `use_mvcc`) |
| `veda-coordinator` | `_meta` state machine + apply path, Percolator `TxnCoordinator`, shard router, Raft transport |
| `veda-time` | Hybrid Logical Clock, `Tso` trait, `LocalTso`, `MetaLeasedTso`, `SwappableTso` |
| `veda-log` | Per-shard `ShardLog`, BookKeeper-style replication, Tonic transport for follower + WAN, cutover primitive |
| `veda-server` | Binary entry point, configuration, startup |
| `veda-admin` | Offline maintenance CLI (scaffold) |
| `veda-mcp` | Stdio↔HTTP bridge binary so desktop MCP clients (Claude Desktop, Cursor, VS Code) can use Veda's MCP server |

See [documentation/rearchitecture-overview.md](documentation/rearchitecture-overview.md) for the layered design and current phase status.

## Cluster modes & admin endpoints

Veda runs in two modes; the admin REST surface differs depending on which one is active.

- **Single-node** (default, no Raft) — Veda comes up with `VEDA_COORDINATOR` unset (or anything ≠ `raft`). Schemas, segments, MVCC, the Percolator txn coordinator, atomic reindex, and the per-collection log all work. The `_meta`-backed admin endpoints (region topology, replica promotion, cluster membership) return **503** in this mode because there is no `_meta` to mutate.
- **Raft** — set `VEDA_COORDINATOR=raft` to start `_meta` as a single-node openraft group. The 503 endpoints below light up; `/v1/admin/nodes` flips its `mode` field from `"single"` to `"raft"`.

| Endpoint | Single-node | Raft | Notes |
|---|---|---|---|
| `GET /v1/admin/health` / `readiness` | ✓ | ✓ | Always reachable |
| `GET /v1/admin/metrics` | ✓ | ✓ | Prometheus text format |
| `GET /v1/admin/resources` | ✓ | ✓ | Latest node-resource snapshot (disk/CPU/load/fds/memory/RSS); fields `null` off-Linux |
| `GET /v1/admin/slos` | ✓ | ✓ | SLO defs + per-SLO SLI/error-budget/burn-rate (`/slos/{id}/status`); CRUD admin-only |
| `GET /v1/admin/synthetics` | ✓ | ✓ | Synthetic-probe defs + results (`/synthetics/{id}/results`); probes run in-process each tick |
| `POST /v1/admin/insights/{event_id}/triage` | ✓ | ✓ | AI triage of an event via the `veda-ops` agent; 404 unknown event, 503 when no LLM |
| `GET /v1/admin/correlations` | ✓ | ✓ | Slow-query/error correlation over the query-log window, bucketed and ranked worst-first |
| `GET /v1/admin/pprof` | ✓ | ✓ | On-demand CPU profile (flamegraph SVG / pprof proto); `profiling` feature build; 409 if already running |
| `GET /v1/admin/depmap` | ✓ | ✓ | Declarative service dependency graph (9 nodes / 8 edges) |
| `GET /v1/admin/index-advisor` | ✓ | ✓ | Per-collection index_type/params/p95/recall (proxy) + directional recommendations + open events |
| `GET /v1/admin/change-points` | ✓ | ✓ | Change-point detections (sustained level shifts / trend breaks) in the advisor metric series; warm-up + min-magnitude guarded |
| `GET /v1/admin/log-rate-analysis` | ✓ | ✓ | Semantic spike root-cause attribution — names the meaning-clusters driving a log-rate spike (`?window_secs=&baseline_secs=&min_level=`); template fallback w/o embedder |
| `GET /v1/admin/incidents` | ✓ | ✓ | Semantic incident clusters (dedup of correlated ops events); off-tick runner; now carries `draft` (veda-ops RCA), `is_novel`/`novelty_score`, and a `noise_reduction` summary |
| `GET /v1/admin/forecast` | ✓ | ✓ | Capacity forecasts — disk-full / memory-exhaustion / recall-decay ETAs (warm-up + min-R² guarded) |
| `GET /v1/admin/insights/{event_id}/rca` | ✓ | ✓ | Cross-signal root-cause bundle (logs + slow/errored queries + traces + resources + incident) |
| `GET /v1/admin/llm-traces` | ✓ | ✓ | Semantic search over the `_llm_traces` store (`?q=`); prompt/response/faithfulness/cost/cache_hit/guardrail_flagged/latency/ts per trace |
| `GET /v1/admin/llm-traces/clusters` | ✓ | ✓ | Failure clustering over LLM traces (reuses the embedding categorizer) — recurring failure modes |
| `POST /v1/conversations/{id}/feedback` | ✓ | ✓ | Thumbs `{ rating: up\|down, note? }` on a conversation; emits `veda_llm_feedback_total{rating}`. Tenant/RLS-scoped |
| `POST /v1/agents/{id}/converse/stream` | ✓ | ✓ | Streaming converse over SSE (reasoning/tool_call/tool_result/answer deltas + `done` w/ citations+grounding); 503 no LLM. Tenant/RLS-scoped |
| `GET` / `POST /v1/eval-sets`, `/{id}`, `/{id}/run`, `/{id}/runs`, `/{id}/cases/from-conversation` | ✓ | ✓ | RAG/agent eval harness + production-trace flywheel (recall@k/faithfulness/relevancy/pass_rate); 503 no LLM. Tenant/RLS-scoped |
| `POST /v1/collections/{name}/ingest-url` | ✓ | ✓ | Web ingestion via crawl4ai — `{ url\|urls, deep_crawl?, max_pages?, chunker?, ttl_secs? }` → crawl → markdown → chunk/embed/insert (records carry `source_url`/`title`/`chunk_index`); 503 if crawl4ai unconfigured. Tenant/RLS-scoped |
| `GET` / `POST /v1/admin/connectors`, `GET`/`PUT`/`DELETE /v1/admin/connectors/{id}`, `POST /v1/admin/connectors/{id}/run` | ✓ | ✓ | Connectors framework — `Connector { id, kind: web\|websearch, name, config, enabled }` (SavedObjects, tenant-scoped); `run` crawls a Web connector's URLs into its target collection. Admin-only |
| `GET /v1/admin/nodes` | ✓ (1 node) | ✓ (N nodes) | `mode` field reports which path is active |
| `GET /v1/admin/region_topology` | ✓ (returns `{topology:null}`) | ✓ | Read works even without `_meta` |
| `PUT /v1/admin/region_topology` | ✗ 503 | ✓ | Mutates `_meta::shard_ensembles` |
| `POST /v1/admin/region_topology/promote` | ✗ 503 | ✓ | Requires `_meta` for epoch monotonicity |
| `GET` / `POST /v1/admin/replica/safe_ts` | ✓ (local-only) | ✓ | Per-process watermark, sentinel 0 = no gating |
| `POST /v1/admin/cluster/join` / `DELETE /cluster/nodes/{id}` | ✗ 503 | ✓ | Openraft membership change |
| `POST /v1/admin/cluster/noop` | ✗ 503 | ✓ | Drives the Raft log for test harnesses |
| `POST /v1/admin/collections/{name}/mvcc_checkpoint` | ✓ | ✓ | Atomic snapshot + WAL truncate; per-shard fan-out |
| `GET` / `POST /v1/admin/jobs`, `DELETE /v1/admin/jobs/{name}` | ✓ | ✓ | Cron-style scheduled jobs (admin-only); schedule persists in the data dir |
| `POST /v1/auth/login` | ✓ | ✓ | Argon2id verify; set `VEDA_BOOTSTRAP_ADMIN_PASSWORD` before first start |

If you don't intend to run multi-node, leave `VEDA_COORDINATOR` unset and ignore the 503 endpoints — every Phase E-onward write path goes through `veda-log` regardless.

## API Examples

### Create a collection

```bash
curl -X POST http://localhost:6340/v1/collections \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "name": "products",
    "fields": [
      {"name": "embedding", "type": "vector", "dimension": 768, "metric": "cosine"},
      {"name": "category", "type": "string", "filterable": true},
      {"name": "price", "type": "float64", "filterable": true},
      {"name": "description", "type": "string"}
    ]
  }'
```

### Insert records

```bash
curl -X POST http://localhost:6340/v1/collections/products/insert \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "records": [
      {
        "embedding": [0.1, 0.2, ...],
        "category": "electronics",
        "price": 299.99,
        "description": "Wireless headphones"
      }
    ]
  }'
```

### Embed + Insert (auto-embedding)

Let Veda generate embeddings automatically. The `auto_embed` map specifies which source fields to embed into which vector fields. Omit `model` to let Veda auto-select based on data modality.

```bash
curl -X POST http://localhost:6340/v1/collections/products/embed-insert \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "records": [
      {
        "description": "Wireless headphones with noise cancellation",
        "category": "electronics",
        "price": 299.99
      }
    ],
    "auto_embed": {
      "embedding": {
        "source_field": "description"
      }
    }
  }'
```

**Auto-selection rules:**
- Text-only data uses the text model (BGE-M3, CPU-friendly)
- If any field is non-text (image, video), **all** auto-selected mappings use the multimodal model (Qwen3-VL) to keep embeddings in a unified vector space
- Specify `"model": "model-name"` in any mapping to override auto-selection

### Search

```bash
curl -X POST http://localhost:6340/v1/collections/products/search \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "query": {"vector": {"field": "embedding", "values": [0.15, 0.22, ...]}},
    "limit": 10
  }'
```

### Filtered search

```bash
curl -X POST http://localhost:6340/v1/collections/products/search \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "query": {"vector": {"field": "embedding", "values": [0.15, 0.22, ...]}},
    "limit": 10,
    "filter": "category = '\''electronics'\'' AND price < 500"
  }'
```

## Python SDK

```bash
pip install veda-client
```

```python
from veda_client import VedaClient

client = VedaClient("http://localhost:6340")

# Create a collection
client.create_collection({
    "name": "products",
    "fields": [
        {"name": "embedding", "type": "vector", "dimension": 768, "metric": "cosine"},
        {"name": "category", "type": "string", "filterable": True},
        {"name": "price", "type": "float64", "filterable": True},
    ],
})

# Insert records
client.insert("products", [
    {"embedding": [0.1, 0.2, ...], "category": "electronics", "price": 299.99},
])

# Search
response = client.search(
    collection="products",
    field="embedding",
    vector=[0.15, 0.22, ...],
    limit=10,
    filter="category = 'electronics'",
)

for hit in response.results:
    print(f"{hit.id}: {hit.distance:.4f}")
```

## Document Processing

Veda provides **end-to-end document processing**: upload a document and Veda handles the entire pipeline -- OCR, chunking, embedding, and insertion into your collection. This is powered by [FlashParse](https://github.com/kshitij2605/FlashParse), which converts PDFs, DOCX, PPTX, images, and other document formats into structured markdown with extracted images.

**Pipeline:** Document upload --> FlashParse OCR --> markdown + images --> chunk (configurable strategy) --> embed (optional) --> insert into collection

### Chunking Strategies

| Strategy | Description | Default Size | Default Overlap |
|---|---|---|---|
| `markdown_recursive` | Recursively splits markdown by structure (default) | 2000 chars | 300 |
| `heading_1` | Splits on H1 headings | -- | -- |
| `heading_2` | Splits on H2 headings | -- | -- |
| `heading_3` | Splits on H3 headings | -- | -- |
| `semantic` | Word-overlap Jaccard similarity; splits on topic change | 2000 chars | 300 |
| `sentence` | Splits on sentence boundaries | 2000 chars | 300 |
| `fixed_size` | Fixed character count chunks | 2000 chars | 300 |
| `page` | Splits on page separators | -- | -- |
| `none` | No chunking; entire document as one chunk | -- | -- |

### Setup

1. Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

2. Start Veda with OCR enabled:

```bash
# Text-only OCR (FlashParse + GLM-OCR, ~8GB VRAM)
docker compose --profile ocr up

# Full OCR with VLM image captioning (~48GB+ VRAM)
docker compose --profile ocr --profile ocr-vlm up
```

The `ocr` profile starts FlashParse and GLM-OCR for text, table, and formula extraction. Add the `ocr-vlm` profile to also start the VLM service (Qwen/Qwen3.5-35B-A3B-FP8 on vLLM) for image captioning and classification.

**VLM configuration:** If you prefer to use an external VLM instead of the Docker-managed one, set these environment variables to point to any OpenAI-compatible API:
- `VLM_ENDPOINT` -- e.g., `http://your-server:8008/v1/chat/completions`
- `VLM_API_KEY` -- your API key (leave empty for local models)
- `VLM_MODEL` -- the model name to use

### Supported Formats

PDF, DOCX, PPTX, images (PNG, JPG, TIFF, BMP), and other formats supported by FlashParse.

### Single Document Processing

```bash
# 1. Create a collection for documents
curl -X POST http://localhost:6340/v1/collections \
  -H "Content-Type: application/json" \
  -d '{
  "name": "my-docs",
  "fields": [
    {"name": "content", "type": "string", "filterable": true},
    {"name": "content_embedding", "type": "vector", "dimension": 768, "metric": "cosine"},
    {"name": "source_file", "type": "string", "filterable": true},
    {"name": "chunk_index", "type": "int32"},
    {"name": "image_blob", "type": "blob"}
  ]
}'

# 2. Process a document (end-to-end: OCR -> chunk -> embed -> insert)
# Default (markdown-recursive chunking)
curl -X POST http://localhost:6340/v1/collections/my-docs/process-document \
  -F "file=@report.pdf"

# Semantic chunking
curl -X POST http://localhost:6340/v1/collections/my-docs/process-document \
  -F "file=@report.pdf" \
  -F "chunking_strategy=semantic"

# Split by H2 headings with custom chunk size
curl -X POST http://localhost:6340/v1/collections/my-docs/process-document \
  -F "file=@report.pdf" \
  -F "chunking_strategy=heading_2" \
  -F "chunk_size=3000"

# 3. Search the processed content
curl -X POST http://localhost:6340/v1/collections/my-docs/search -d '{
  "query": {"vector": {"field": "content_embedding", "values": [...]}},
  "limit": 5
}'
```

**Options:**
- `chunking_strategy=<strategy>` -- chunking strategy (default: `markdown_recursive`)
- `chunk_size=<chars>` -- maximum chunk size in characters (default: 2000)
- `chunk_overlap=<chars>` -- overlap between chunks in characters (default: 300)
- `skip_captions=true` -- skip VLM-based image captioning (faster processing)
- `dpi=300` -- set DPI for PDF rasterization (default: automatic)

```bash
curl -X POST http://localhost:6340/v1/collections/my-docs/process-document \
  -F "file=@report.pdf" \
  -F "skip_captions=true" \
  -F "dpi=300"
```

### Batch / Multiple File Processing

Process multiple documents in a single request:

```bash
curl -X POST http://localhost:6340/v1/collections/my-docs/process-documents \
  -F "files=@report1.pdf" \
  -F "files=@report2.pdf" \
  -F "files=@slides.pptx"
```

Returns aggregate stats with per-file results.

### Python SDK

```python
from veda_client import VedaClient

client = VedaClient("http://localhost:6340")

# Process a single document
with open("report.pdf", "rb") as f:
    result = client.process_document("my-docs", f)
    print(f"Inserted {result['records_inserted']} records")
    for chunk in result['chunks']:
        print(f"  [{chunk['type']}] {chunk['id']}: {chunk['preview'][:60]}")
```

If FlashParse is not running, the endpoint returns a clear error with setup instructions.

## Niriksha Dashboard

**Niriksha** is Veda's built-in data visualization and exploration dashboard for multimodal vector data. It provides:

- Collection overview with real-time statistics
- Interactive vector space visualization (2D/3D projections)
- Point-and-click search with filter builder
- Embedding inspector for any modality
- Cluster health and shard status monitoring
- Query performance analytics
- User and role management

Access it at `http://localhost:5173` when running via Docker Compose.

## Benchmarks

Measured on 10,000 vectors of dimension 768, single node, recall@10:

| Index | Recall@10 | Latency (p99) |
|---|---|---|
| HNSW (M=16, ef=200) | 95.5% | ~2ms |
| DiskANN (R=64, L=100) | 99.9% | ~5ms |
| IVF-PQ (nlist=256, m=48) | 89.2% | ~1ms |
| Flat (brute-force) | 100.0% | ~12ms |

Run the benchmark suite yourself:

```bash
cargo test --test ann_benchmark -- --ignored --nocapture
```

## Project Structure

```
project-mdb/
  crates/
    veda-core/          # Shared types, distances, filters
    veda-index/         # ANN and scalar indexes
    veda-storage/       # Vector storage, blobs, WAL
    veda-query/         # Query planning and fusion
    veda-gateway/       # REST API and auth
    veda-coordinator/   # Shards, cluster, compaction
    veda-server/        # Binary entry point
  niriksha/             # Visualization dashboard (React + TypeScript)
  sdk/
    python/             # Python SDK (veda-client)
  documentation/        # Architecture docs and ADRs
  research/             # Research notes and analysis
```

## Configuration

### Embedding Models

| Variable | Default | Description |
|---|---|---|
| `EMBEDDING_ENDPOINT` | `http://embedding:80` | Text embedding endpoint (OpenAI-compatible) |
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | Text embedding model name |
| `EMBEDDING_DIMENSION` | `1024` | Text embedding output dimension |
| `MULTIMODAL_EMBEDDING_ENDPOINT` | `http://multimodal-embedding:8000` | Multimodal embedding endpoint |
| `MULTIMODAL_EMBEDDING_MODEL` | `Qwen/Qwen3-VL-Embedding-2B` | Multimodal embedding model name |
| `MULTIMODAL_EMBEDDING_DIMENSION` | `2048` | Multimodal embedding output dimension |

### Blob Storage

| Variable | Default | Description |
|---|---|---|
| `VEDA_BLOB_BACKEND` | `minio` | `"minio"` (path-style, auto-create bucket) or `"s3"` (virtual-hosted, TLS) |
| `VEDA_BLOB_ENDPOINT` | `http://minio:9000` | Object store endpoint |
| `VEDA_BLOB_BUCKET` | `veda-blobs` | Bucket name |
| `VEDA_BLOB_ACCESS_KEY` | — | Access key |
| `VEDA_BLOB_SECRET_KEY` | — | Secret key |
| `VEDA_BLOB_REGION` | `us-east-1` | AWS region (for S3 mode) |

If `VEDA_BLOB_BACKEND` is unset, Veda falls back to local filesystem blob storage.

## Memory Scaling

By default, Veda keeps all records in the `ClusterStateMachine.records` HashMap (the non-log storage path). RSS grows linearly with record count: each record costs roughly **~80 bytes of overhead plus the vector data itself**. A 24-hour soak test measured 474 MB RSS at 3,338 records growing to 1,257 MB at 13,037 records (~80 MB/h growth rate, proportional to insert throughput).

For collections that will exceed **~50 000 records**, enable the on-disk log path by setting `use_log_primary: true` in the collection configuration:

```bash
curl -X POST http://localhost:6340/v1/collections   -H "Content-Type: application/json"   -d '{
    "name": "large-collection",
    "fields": [ ... ],
    "use_log_primary": true
  }'
```

With `use_log_primary` enabled, record bodies are stored on disk via the per-shard append-only log and only the index structures are held in memory. This changes the memory profile from O(records) to O(index), which is dramatically smaller for large collections.

## Monitoring & Observability

### Starting the monitoring stack

```bash
docker compose --profile monitoring up
```

This starts six services alongside Veda:

| Service | Port | Purpose |
|---|---|---|
| OTel Collector | `:4317` (gRPC) | Receives traces, metrics, and logs via OTLP |
| Prometheus | `:9090` | Scrapes Veda's `/v1/admin/metrics` endpoint |
| Mimir | `:9009` | Long-term metrics storage (Prometheus remote-write) |
| Tempo | `:3200` | Distributed trace storage |
| Loki | `:3100` | Log aggregation |
| Grafana | `:3000` | Dashboards and exploration (pre-provisioned datasources and dashboard) |

### OpenTelemetry configuration

Veda exports traces and logs via OTLP when the following environment variable is set:

| Variable | Default | Description |
|---|---|---|
| `OTEL_EXPORTER_OTLP_ENDPOINT` | *(unset)* | OTLP gRPC endpoint (e.g. `http://otel-collector:4317`). OTel export is disabled when unset. |
| `OTEL_SERVICE_NAME` | `veda` | Service name attached to all telemetry |

When running with `--profile monitoring`, these are pre-configured in `docker-compose.yml`.

### Prometheus metrics

Veda exposes Prometheus metrics at `GET /v1/admin/metrics`. Metrics include request counts and durations, search latency, insert counts, embedding and blob operation durations, segment and compaction gauges, and error totals.

### Health checks

```bash
# Liveness -- always returns 200
curl http://localhost:6340/v1/admin/health

# Readiness -- returns 503 if unhealthy
curl http://localhost:6340/v1/admin/readiness
```

Both endpoints return:

```json
{
  "status": "healthy",
  "version": "0.1.0",
  "checks": {
    "minio": "healthy",
    "nats": "healthy",
    "embedding": "unconfigured",
    "multimodal_embedding": "unconfigured"
  }
}
```

Status is one of `healthy`, `degraded` (non-critical service down), or `unhealthy` (MinIO or NATS down). The readiness endpoint returns HTTP 503 when unhealthy.

### Query explain

Add `"explain": true` to any search request to get per-phase timing:

```bash
curl -X POST http://localhost:6340/v1/collections/products/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": {"vector": {"field": "embedding", "values": [0.15, 0.22, ...]}},
    "limit": 10,
    "explain": true
  }'
```

The response includes an `explain` object with `filter_ms`, `ann_search_ms`, `scoring_ms`, `fusion_ms`, and `total_ms` fields.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, coding standards, and the pull request process.

## License

Licensed under the [Apache License 2.0](LICENSE).

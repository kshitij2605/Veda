# RAG & AIOps Intelligence

This is the vector-native intelligence layer that comes from owning the whole pipeline. A text-search engine indexes documents you push to it, but the corpus boundaries, the chunker, the embedding model, the retriever, the agent loop, and the operational telemetry live in other people's systems. **Veda owns all of them** — so it can apply semantic intelligence to *both* sides of the house: to **agents** (citations, faithfulness scoring, evaluation, memory, streaming) and to **ops** (semantic incident correlation, capacity forecasting, cross-signal root-cause), using the same primitives. An engine that doesn't own the embeddings can't faithfully score grounding; a metrics store that doesn't own an embedder can't semantically cluster incidents. Veda can, because it *is* the whole pipeline.

LLM-dependent features degrade gracefully — an embedding fallback, or a clear `503` — when no LLM is configured.

## Faithfulness & citations

Every agent run now returns *why you should trust the answer*, not just the answer.

**Citations.** Veda walks the run's retrieval tool calls and extracts the supporting chunks into `citations: [{ chunk_id, collection, snippet, score }]`. Because Veda owns the retriever and the chunk identity, the citation points at the exact chunk that grounded the answer — not a re-found approximation.

**Grounding scores.** Each run carries `grounding: { faithfulness, answer_relevancy, method }`, each in `0–1`:

- **Faithfulness** — how grounded the answer is in the retrieved context. The primary method is **LLM-as-judge** (RAGAS-style): the model is asked whether each claim is supported by the retrieved chunks. With no LLM configured, Veda falls back to **embedding-overlap** (cosine similarity between answer and context, using the embedder it already owns). A run with no retrieval reports `method: no_context` rather than a fabricated score.
- **Answer relevancy** — how relevant the answer is to the original query.

`method` (`llm_judge` / `embedding_overlap` / `no_context`) always reports which path produced the score, so the cheap fallback is never mistaken for the judge. Scoring is **post-run, best-effort, and non-fatal** — a scoring failure never fails the answer. The scores ride on `/converse`, the triage response, and the A2A response, and aggregate into the `veda_agent_faithfulness` and `veda_agent_citations` Prometheus metrics.

## Quality SLOs

The [SLO engine](/docs/service-reliability) gains two RAG-native SLI kinds, so retrieval quality and answer grounding get the same error-budget + burn-rate treatment as latency and availability:

- **`recall { collection, k? }`** — sourced from the ANN [recall probe](/docs/apm-observability#ann-index-advisor). Labeled **proxy** (ANN top-k vs an exact scan over a sampled subset, not a labeled relevance set) — a drift guard, not a benchmark.
- **`faithfulness { agent? }`** — sourced from a persisted faithfulness sample of converse turns, optionally scoped to one agent.

Both reuse the existing burn-rate / error-budget rules and surface through `/v1/admin/slos` and the Niriksha SLOs page — no new alerting machinery, because the SLO layer just reads SLIs Veda already produces.

## Eval harness & the production-trace flywheel

A regression-test harness for retrieval and generation quality.

- **Eval sets** — a named set of cases `{ query, expected_chunk_ids?, gold_answer? }`. `expected_chunk_ids` drive **recall@k**; `gold_answer` is an optional reference. CRUD at `POST/GET/DELETE /v1/eval-sets` (+ `/{id}`); the `id` is server-assigned when omitted.
- **Running** — `POST /v1/eval-sets/{id}/run` runs every case through the agent and returns per-case and aggregate **recall@k, faithfulness, answer-relevancy, and pass_rate**; `GET /v1/eval-sets/{id}/runs` tracks quality over time. The run needs an LLM (faithfulness/relevancy lean on the judge), so it returns `503` when none is configured.
- **The flywheel** — `POST /v1/eval-sets/{id}/cases/from-conversation` graduates a recorded production conversation into a regression case (its query, the retrieved chunk ids as `expected_chunk_ids`, the answer as `gold_answer`). Because Veda owns the conversation history *and* the eval harness, "production query → permanent test case" is a single call.

A Niriksha **Eval Harness** page renders sets, runs, and aggregate scores.

> Beyond graduating conversations, Veda can **generate an eval set from your own documents** and **auto-tune the pipeline** (chunker, embedder, index, fusion, rerank) against it — see [RAG Auto-Tuning](/docs/auto-tuning).

## Vector-backed agent memory

Opt-in long-term memory for agents, backed by Veda's own vector store. `Agent.memory` defaults to **false**, so existing agents are byte-for-byte unchanged. When enabled, each turn is embedded into a per-tenant `_agent_memory` collection, and before each new turn the most semantically-similar prior turns are retrieved and injected — **bounded** — into the system prompt. The whole path is best-effort and off the request lock: a memory read/write failure degrades to a memory-less turn rather than failing the request. There is no external vector DB to provision, because Veda *is* the vector DB.

## Streaming converse (SSE)

`POST /v1/agents/{id}/converse/stream` runs the same ReAct loop as the blocking `/converse` but streams progress over Server-Sent Events:

```
event: reasoning      data: {…}      (model thinking deltas)
event: tool_call      data: {…}      (a tool invocation)
event: tool_result    data: {…}      (its result)
event: answer         data: {…}      (answer-token deltas)
event: done           data: { conversation_id, response, metrics, citations, grounding }
```

The terminal `done` frame carries the same `metrics`, `citations`, and `grounding` as the blocking response. The blocking `/converse` is unchanged and A2A stays synchronous; streaming returns `503` without an LLM, and is tenant-scoped with row-level security enforced through the caller's identity.

## Semantic incident correlation

The ops side gets the same vector intelligence as the agent side: a storm of related advisor events (a flapping rule, a cascading dependency failure) shouldn't read as twelve separate pages. Veda **embeds each event's title + detail** and clusters them by cosine similarity into an `Incident`:

```json
{
  "representative": "…", "member_event_ids": [ … ], "rule_ids": [ … ],
  "count": 4, "max_severity": "High", "first_seen": "…", "last_seen": "…"
}
```

With no embedder, it falls back to a `rule_id + scope` grouping, so dedup still works without semantics. Correlation runs on a separate off-tick runner (`VEDA_INCIDENT_INTERVAL_SECS`, default 300s) so it never stalls the advisor tick. `GET /v1/admin/incidents` returns the clusters; a Niriksha Insights **Incidents** tab renders them. It's a clustering surface that sits *beside* the [rule catalog](/docs/ops-advisor#the-rule-catalog), not a rule inside it.

Each incident also carries a proactive `veda-ops`-drafted RCA, an `is_novel` / `novelty_score` flag (unprecedented patterns escalate to High), and a `noise_reduction` summary — notifications fire **once per incident**, not per event. See [AIOps](/docs/aiops#root-cause-agentic) for proactive drafts, novelty detection, and alert-storm noise reduction.

## Capacity forecasting

Veda already records a time series of resource snapshots (disk, memory, recall). Forecasting fits a **least-squares trend** over that history and extrapolates to each rule's threshold, producing a threshold-crossing ETA — *when* the disk fills, *when* memory exhausts, *when* recall decays below target. Two guards keep it honest: a **warm-up** (won't forecast without enough history) and a **min-R²** (won't forecast off a flat or noisy series), so it never fires on flat/noisy data.

Three rules join the [advisor catalog](/docs/ops-advisor#the-rule-catalog):

| Rule | Severity | Projects |
|---|---|---|
| `disk_full_forecast` | High | data-dir disk exhaustion |
| `memory_growth_forecast` | Medium | memory exhaustion |
| `recall_decay_forecast` | Medium | recall dropping below target |

`GET /v1/admin/forecast` returns the per-metric trend (slope, R², threshold, ETA); a Niriksha forecast panel renders it. These flow through the existing event store + alert channels like every other rule.

## Cross-signal RCA

The payoff of owning every signal: one call that assembles the whole picture around an event. `GET /v1/admin/insights/{event_id}/rca` joins five signals for one event into a single bundle:

- the **correlated logs** (from the [log buffer](/docs/log-insights)),
- the **slow / errored queries** in the event window (from the [query log](/docs/apm-observability#slow-query-error-correlation)),
- the **traces** matched by `trace_id` and the event time window,
- the **nearest resource snapshot** ([node resource signals](/docs/resource-signals)), and
- the **related incident** (above).

This is the cross-signal correlation an APM tool charges for — but Veda already holds every signal, so the join is local. A `correlate_incident` ops tool exposes the same bundle to the [`veda-ops` triage agent](/docs/service-reliability#ai-ops-triage), and the triage prompt is enriched with a bounded RCA summary — so AI triage reasons over logs + queries + traces + resources + the incident at once, not a single event in isolation. The Niriksha event drawer renders a **Root-cause bundle** section.

## REST API

Agent routes are tenant/RLS-scoped; `/admin/*` are admin-only.

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/agents/{id}/converse/stream` | Streaming converse over SSE; ends with a `done` frame carrying `metrics` + `citations` + `grounding`; `503` no LLM |
| `GET/POST` | `/v1/eval-sets` | List / create-or-overwrite eval sets (`id` server-assigned if omitted) |
| `GET/DELETE` | `/v1/eval-sets/{id}` | Get / delete an eval set |
| `POST` | `/v1/eval-sets/{id}/run` | Run → per-case + aggregate recall@k / faithfulness / answer-relevancy / pass_rate; `503` no LLM |
| `GET` | `/v1/eval-sets/{id}/runs` | List past runs |
| `POST` | `/v1/eval-sets/{id}/cases/from-conversation` | Graduate a production conversation into a regression case |
| `GET` | `/v1/admin/incidents` | Semantic incident clusters (dedup of correlated ops events) |
| `GET` | `/v1/admin/forecast` | Capacity forecasts — disk-full / memory-exhaustion / recall-decay ETAs |
| `GET` | `/v1/admin/insights/{event_id}/rca` | Cross-signal root-cause bundle (logs + queries + traces + resources + incident) |

`POST /v1/agents/{id}/converse` and `POST /v1/admin/insights/{event_id}/triage` now also return `citations` + `grounding`; `POST /v1/admin/slos` accepts the new `recall` and `faithfulness` SLO kinds.

## Metrics & configuration

| Metric (`/v1/admin/metrics`) | Type | Notes |
|---|---|---|
| `veda_agent_faithfulness` | gauge | latest faithfulness score (0–1) |
| `veda_agent_citations` | counter | citation accounting per run |

| Variable | Default | Purpose |
|---|---|---|
| `VEDA_INCIDENT_INTERVAL_SECS` | `300` | Cadence of the off-tick semantic incident-correlation runner |
| `VEDA_LLM_ENDPOINT` | *(unset)* | When unset, LLM-judge faithfulness falls back to embedding-overlap; eval-run and streaming-converse return `503` |
| `VEDA_RECALL_PROBE_INTERVAL_SECS` | `600` | Recall-probe cadence — sources the `recall` SLO SLI (proxy) |

## Current limitations

- **LLM-judge faithfulness needs an LLM** — without `VEDA_LLM_ENDPOINT`, faithfulness/relevancy fall back to embedding-overlap, a cheaper, less precise proxy (`method` always reports which path was used).
- **Proxy recall ≠ labeled recall@k** — the `recall` SLO's SLI comes from the proxy recall probe (a drift signal, not a ground-truth benchmark against a labeled relevance set).
- **`gen_ai.system` is inferred** from the model name, so exotic/self-hosted names may be mis-attributed (carried over from the APM cycle).

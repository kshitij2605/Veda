# Veda RAG & AIOps Intelligence — Citations & Faithfulness, Quality SLOs, Eval Harness, Agent Memory, Streaming Converse, Incident Correlation, Forecasting & Cross-Signal RCA

**Date:** 2026-06-18
**Status:** Shipped on `feat/rag-aiops-intelligence` (also on `main`), v0.3.0. Validated end-to-end against a live server.

RAG & AIOps Intelligence is the vector-native layer Elasticsearch can't build. The framing matters:
Elastic is **text-search-core** and does not own the RAG stack — it indexes documents you push to
it and answers queries, but the corpus boundaries, the chunker, the embedding model, the retriever,
the agent loop, and the operational telemetry of the whole pipeline live in other people's systems.
**Veda owns all of them.** It owns the corpus, the chunker, the embeddings, the retriever, the
agent, *and* every ops signal it emits. That single fact lets Veda apply vector / semantic / RAG
intelligence to **both** sides of the house — to **agents** (citations, faithfulness scoring,
evaluation, memory, streaming) and to **ops** (semantic incident correlation, capacity forecasting,
cross-signal root-cause) — using the same primitives. A text-search engine that doesn't own the
embeddings or the agent can't faithfully score grounding, and a metrics store that doesn't own an
embedder can't semantically cluster incidents. Veda can, because it is the whole pipeline.

Everything here builds on existing primitives: the collections + embeddings + retriever for the
agent side, and the Ops Advisor pipeline (snapshot → rules → events → channels → Niriksha) for the
ops side. LLM-dependent features degrade gracefully (an embedding fallback, or a clear **503**) when
no LLM is configured, and the OTLP export path is unchanged. The Ops Advisor catalog is now ~33
rules (was 30: +3 forecast rules; incident correlation is a separate clustering surface, not a
rule).

---

## 1. Faithfulness + citations

Veda's RAG/agent loop now returns *why you should trust the answer*, not just the answer.

**Citations.** After a run, Veda walks the run's retrieval tool calls and extracts the supporting
chunks into `citations: [{ chunk_id, collection, snippet, score }]`. Because Veda owns the retriever
and the chunk identity, the citation points at the exact chunk that grounded the answer — not a
re-found approximation.

**Grounding scores.** Each run also carries `grounding: { faithfulness, answer_relevancy, method }`,
each in `0–1`:
- **Faithfulness** — how grounded the answer is in the retrieved context. The **primary** method is
  **LLM-as-judge** (RAGAS-style): the model is asked whether each claim in the answer is supported
  by the retrieved chunks. When no LLM is configured, Veda falls back to **embedding-overlap** —
  cosine similarity between the answer and its retrieved context, using the embedder Veda already
  owns. When the run did no retrieval at all, `method` is `no_context` and the score is reported as
  such rather than fabricated.
- **Answer relevancy** — how relevant the answer is to the original query.

`method` tells you which path produced the score (`llm_judge`, `embedding_overlap`, or
`no_context`), so a consumer never mistakes the cheap fallback for the judge.

Scoring is **post-run, best-effort, and non-fatal** — a scoring failure never fails the answer. The
scores ride on `ConverseResponse`, the triage response, and the A2A response. Two Prometheus metrics
expose the aggregate: `veda_agent_faithfulness` and `veda_agent_citations`.

---

## 2. Retrieval & faithfulness quality SLOs

The existing SLO engine (error budgets + Google-SRE multi-window burn-rate alerting) gains two
RAG-native SLI kinds, so retrieval quality and answer grounding get the same treatment as latency
and availability.

- **`SloKind::Recall { collection, k? }`** — the SLI is sourced from the ANN **recall probe**
  shipped in v0.2.9. It is labeled **proxy**: the probe measures ANN top-k against an exact
  brute-force scan over a sampled subset, not against a labeled relevance set. A recall SLO is a
  drift guard ("alert me when retrieval recall on collection X burns its budget"), not a
  ground-truth benchmark.
- **`SloKind::Faithfulness { agent? }`** — the SLI is sourced from a **persisted faithfulness
  sample** of converse turns (the scores from §1). An optional `agent` filter scopes the SLO to one
  agent.

Both reuse the existing burn-rate / error-budget rules (`slo_burn_rate_fast`, `slo_burn_rate_slow`,
`slo_budget_exhausted`) and surface through `/v1/admin/slos` and the Niriksha SLOs page — no new
alerting machinery. This is only possible because Veda owns both the recall probe and the
faithfulness scorer; the SLO layer just reads SLIs that already exist.

---

## 3. RAG/agent eval harness + production-trace flywheel

A regression-test harness for retrieval and generation quality.

**Eval sets.** An `eval-set` is a named collection of cases `{ query, expected_chunk_ids?,
gold_answer? }`. `expected_chunk_ids` drive **recall@k**; `gold_answer` is an optional reference.
CRUD lives at `POST/GET/DELETE /v1/eval-sets` (+ `/{id}`).

**Running.** `POST /v1/eval-sets/{id}/run` runs every case through the agent and returns per-case
and aggregate scores: **recall@k, faithfulness, answer-relevancy, and pass_rate**. `GET
/v1/eval-sets/{id}/runs` lists past runs so you can track quality over time. The run path needs an
LLM (faithfulness/relevancy lean on the judge), so it returns **503** when none is configured.

**The flywheel.** `POST /v1/eval-sets/{id}/cases/from-conversation` **graduates a production trace
into a regression case** — it takes a recorded conversation and turns it into an `EvalCase` (its
query, the retrieved chunk ids as `expected_chunk_ids`, the produced answer as `gold_answer`). A
real query that went well (or badly) becomes a permanent regression guard. Because Veda owns the
conversation history *and* the eval harness, the loop from "production query" to "named test case"
is a single call.

A Niriksha **Eval Harness** page renders the sets, runs, and aggregate scores.

---

## 4. Vector-backed agent memory

Opt-in long-term memory for agents, backed by Veda's own vector store.

`Agent.memory: bool` defaults to **false**, so every existing agent is byte-for-byte unchanged. When
enabled:
- each conversational turn is embedded and written into a **per-tenant `_agent_memory`
  collection**, and
- before each new turn, the most **semantically-similar prior turns** are retrieved and injected —
  **bounded** — into the system prompt.

So an agent recalls relevant context from earlier conversations without stuffing the whole history
into the prompt. The whole path is **best-effort and off the request lock** — a memory read/write
failure degrades to a memory-less turn rather than failing the request. Veda is its own memory
store; there is no external vector DB to provision because Veda *is* the vector DB.

---

## 5. Streaming converse over SSE

`POST /v1/agents/{id}/converse/stream` runs the same ReAct-style loop as the blocking `/converse`
but streams progress over **Server-Sent Events**:

```
event: reasoning      data: {…}      (model thinking deltas)
event: tool_call      data: {…}      (a tool invocation)
event: tool_result    data: {…}      (its result)
event: answer         data: {…}      (answer-token deltas)
…
event: done           data: { conversation_id, response, metrics, citations, grounding }
```

The terminal `done` frame carries the same `metrics`, `citations`, and `grounding` as the blocking
response, so a streaming client gets the full grounded result at the end. The LLM client gained
`complete_stream` to back this. The **blocking `/converse` is unchanged** and **A2A stays sync**;
streaming returns **503** without an LLM. Tenant-scoped, with row-level security enforced through
tool calls via the caller's identity.

---

## 6. Semantic incident correlation & dedup

The ops side gets the same vector intelligence as the agent side. A storm of related ops-advisor
events (a flapping rule, a cascading dependency failure) shouldn't read as twelve separate pages.

Veda **embeds each ops-event's title+detail** and clusters them by cosine similarity into an
`Incident`:

```
Incident {
  representative,         // the representative event
  member_event_ids,       // the events folded into it
  rule_ids,               // which rules contributed
  count,
  max_severity,           // High > Medium > Low
  first_seen, last_seen,
}
```

When no embedder is configured, it falls back to a **rule_id + scope** grouping, so dedup still
works without semantics. Correlation runs on a **separate off-tick runner**
(`VEDA_INCIDENT_INTERVAL_SECS`, default **300**) so it never stalls the advisor tick. `GET
/v1/admin/incidents` returns the clusters; a Niriksha Insights **"Incidents"** tab renders them.
Incident correlation is a clustering surface, not a rule — it sits beside the rule catalog rather
than inside it.

---

## 7. Capacity forecasting

Veda already records a time series of resource snapshots (disk, memory, recall). Forecasting turns
that history into **threshold-crossing ETAs**.

A **least-squares trend** is fit over the snapshot-history series and extrapolated to the rule's
threshold, producing an ETA: *when* the disk fills, *when* memory exhausts, *when* recall decays
below target. Two guards keep it honest:
- **warm-up** — it won't forecast until it has enough history, and
- **min-R²** — it won't forecast off a flat or noisy series (a low-fit trend produces no ETA),

so it **never fires on flat/noisy data**. Three rules join the advisor catalog (bringing it to ~33):

- **`disk_full_forecast`** (High) — projected data-dir disk exhaustion.
- **`memory_growth_forecast`** (Medium) — projected memory exhaustion.
- **`recall_decay_forecast`** (Medium) — projected recall drop below target.

`GET /v1/admin/forecast` returns the per-metric trend (slope, R², threshold, ETA); a Niriksha
forecast panel renders it. These flow through the existing event store + alert channels like every
other rule.

---

## 8. Cross-signal RCA bundle

The payoff of owning every signal: a single call that assembles the whole picture around one event.

`GET /v1/admin/insights/{event_id}/rca` **joins** five signals for one event into one bundle:
- the **correlated logs** (from the log buffer),
- the **slow / errored queries** in the event window (from the query log),
- the **traces** matched by `trace_id` and the event time window,
- the **nearest resource snapshot**, and
- the **related incident** (from §6).

This is the cross-signal correlation an APM tool charges for — but Veda already holds every one of
these signals, so the join is local. A **`correlate_incident` ops tool** exposes the same bundle to
the `veda-ops` triage agent, and the triage prompt is **enriched with a bounded RCA summary**, so AI
triage reasons over logs + queries + traces + resources + the incident at once instead of a single
event in isolation. The Niriksha event-drawer renders a **"Root-cause bundle"** section.

---

## 9. REST / metrics / env reference

### REST (agent routes are tenant/RLS-scoped; `/admin/*` are admin-only — all `bearerAuth`)

| Endpoint | Method | Description |
|---|---|---|
| `/v1/agents/{id}/converse` | POST | Now also returns `citations` + `grounding` alongside `metrics` + the step trace |
| `/v1/agents/{id}/converse/stream` | POST | Streaming converse over SSE; reasoning/tool_call/tool_result/answer deltas then a `done` frame with `{conversation_id, response, metrics, citations, grounding}`; 503 no LLM |
| `/v1/eval-sets` | GET / POST | List / create-or-overwrite eval sets (explicit `id` required today) |
| `/v1/eval-sets/{id}` | GET / DELETE | Get / delete an eval set |
| `/v1/eval-sets/{id}/run` | POST | Run → per-case + aggregate recall@k / faithfulness / answer-relevancy / pass_rate; 503 no LLM |
| `/v1/eval-sets/{id}/runs` | GET | List past runs |
| `/v1/eval-sets/{id}/cases/from-conversation` | POST | Graduate a production conversation into a regression case (the flywheel) |
| `/v1/admin/incidents` | GET | Semantic incident clusters (dedup of correlated ops events) |
| `/v1/admin/forecast` | GET | Capacity forecasts — disk-full / memory-exhaustion / recall-decay ETAs |
| `/v1/admin/insights/{event_id}/rca` | GET | Cross-signal root-cause bundle (logs + queries + traces + resources + incident) |
| `/v1/admin/insights/{event_id}/triage` | POST | Triage now also returns `citations` + `grounding`; prompt enriched with the RCA summary |
| `/v1/admin/slos` | GET / POST | `SloKind` now includes `recall { collection, k? }` and `faithfulness { agent? }` |

### Metrics (Prometheus, at `/v1/admin/metrics`)

| Metric | Type | Notes |
|---|---|---|
| `veda_agent_faithfulness` | gauge | latest faithfulness score (0–1) |
| `veda_agent_citations` | counter / gauge | citation accounting per run |

### Configuration (env vars)

| Variable | Default | Purpose |
|---|---|---|
| `VEDA_INCIDENT_INTERVAL_SECS` | `300` | Cadence of the off-tick semantic incident-correlation runner |
| `VEDA_LLM_ENDPOINT` | *(unset)* | When unset, LLM-judge faithfulness falls back to embedding-overlap, and eval-run / streaming-converse return **503** |
| `VEDA_RECALL_PROBE_INTERVAL_SECS` | `600` | Recall-probe cadence — sources the `recall` SLO SLI (proxy) |

---

## 10. Validation (live)

Verified against a running server on the Linux host:

- **Citations + grounding:** created an auto-embed `kb` collection + **4 facts**; `/converse`
  searched it (**3 search calls**) and returned **citations** with snippets plus **grounding
  faithfulness 1.0 / answer_relevancy 1.0 (`llm_judge`)** alongside RunMetrics.
- **Streaming converse:** `/converse/stream` emitted **64 answer-token frames** plus
  `reasoning` / `tool_call` / `tool_result` events, then a final `done` frame carrying the citations
  + grounding.
- **Eval harness:** an eval-set run scored **faithfulness 1.0, answer_relevancy 1.0, pass_rate 1.0**.
- **AIOps surfaces:** `/v1/admin/forecast`, `/v1/admin/incidents`, and `/v1/eval-sets/{id}/runs` all
  returned **200**.

---

## 11. Limitations & follow-ups

- **LLM-judge faithfulness needs an LLM.** Without `VEDA_LLM_ENDPOINT`, faithfulness/relevancy fall
  back to **embedding-overlap** — a cheaper, less precise proxy. `method` always reports which path
  produced a score, so the fallback is never mistaken for the judge.
- **Proxy recall ≠ labeled recall@k.** The `recall` SLO's SLI comes from the proxy recall probe
  (ANN top-k vs an exact scan over a sampled subset), which is a drift signal, not a ground-truth
  benchmark against a labeled relevance set.
- **`EvalCase` / `SloDef` / `SyntheticProbe` require an explicit `id` on create today.** Supplying
  the id is on the caller; an id-optional (server-assigned) create is a known follow-up.
- **Agent-memory `conversation_id` is not yet threaded.** Memory is retrieved by semantic similarity
  across the tenant's turns; per-conversation scoping/threading of memory is a follow-up.
- **Streaming final-answer is chunked.** Real per-token streaming via `complete_stream` is wired,
  but the loop currently chunks the already-computed answer into `answer` frames rather than
  streaming tokens straight from the model as they arrive — a follow-up.
- **A pre-existing flaky GenAI-span unit test.**
  `agent::executor::tests::emits_genai_semconv_spans_for_chat_and_tool_calls` is flaky under
  parallel runs (the GenAI span exporter is process-global); it **passes single-threaded** and is
  slated for a separate test-isolation fix. It is unrelated to this feature line.
- **`gen_ai.system` is inferred** from the model name rather than a configured provider identity, so
  exotic/self-hosted model names may be mis-attributed (carried over from the APM cycle).

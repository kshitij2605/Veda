# Veda LLM Observability & Optimization — Semantic Response Cache, Embedding-Based Guardrails, Searchable Trace Store + Failure Clustering, TTFT / Cost Budgets / Quality Drift / Feedback

**Date:** 2026-06-19
**Status:** Shipped on `feat/llm-observability` (also on `main`), v0.3.2. Validated end-to-end against a live server.

LLM Observability is the lane where Elastic's offering is thinnest: Elastic LLM Observability
ingests the GenAI traces/metrics an app emits, but it explicitly **lacks faithfulness scoring,
an eval harness, and semantic caching** — it observes the LLM from the outside. Veda already
shipped the half Elastic can't (answer citations + faithfulness/relevancy, a RAG/agent eval
harness, native GenAI-semconv spans, and cost/token metrics) in the prior RAG/AIOps cycle. This
cycle leans on the one structural advantage nobody else in this lane has: **Veda owns the
embedder, the retriever, the agent loop, and the faithfulness scorer**, so it can turn its
vector engine *into* the LLM layer. A near-duplicate prompt is a vector search away from a
**cached answer with no LLM call**; an unsafe prompt is a cosine away from a **blocked
request**; a recurring bad answer is a cluster away from a **named failure mode**. Where Elastic
*observes* LLM apps, Veda **observes *and* optimizes/secures** them — semantic cache, semantic
guardrails, semantic trace search, plus the faithfulness/eval it already had. Every feature here
reuses an existing primitive (collections + embeddings + the agent executor + the faithfulness
scorer + the Ops Advisor) and degrades gracefully — regex-only guardrails and fail-open without
an embedder, warm-up off real data for the cost/quality rules. All three new agent flags
(`cache` / `guardrails` / `trace`) are **opt-in (default false)**, so existing agents are
unchanged. The Ops Advisor catalog is now ~36 rules (+`llm_cost_budget`, +`quality_drift`).

---

## 1. Semantic response cache

LLM apps re-ask the same thing constantly — paraphrased, re-worded, but *meaning* the same.
A string-keyed cache misses all of it ("How do I tune the ANN index?" ≠ "what's the best way to
configure HNSW?"). Veda already embeds text and searches it by cosine, so a semantic cache is
the natural thing to build, and Elastic — which doesn't own the embedder or the retriever in
this lane — can't.

`agent/cache.rs` is opt-in via the `Agent.cache` flag (default **false**, so existing agents
are untouched). Before a run, the executor **embeds the prompt** and **vector-searches a
per-tenant `_llm_cache` collection** for a near-duplicate:

- on a **hit** (cosine **≥ 0.97**) it returns the cached response **without calling the LLM** —
  the `ConverseResponse` carries `cache_hit:true` and **`steps` is 0** (no reasoning, no tool
  calls, no model latency, no cost);
- on a **miss** it runs the agent normally and **stores** the `{prompt, response}` (embedded)
  back into `_llm_cache` for next time.

The threshold is deliberately high (0.97) so only genuinely equivalent prompts collapse — a
0.90-ish "related but different" question still runs. Metrics `veda_llm_cache_hits_total` and
`veda_llm_cache_misses_total` make the hit-rate (and the LLM-spend you avoided) observable. This
is pure vector-native optimization: the cache *is* a collection, the lookup *is* a search, and
it reuses the same embed → insert → search path everything else in Veda uses.

---

## 2. Embedding-based safety guardrails

Prompt-injection, jailbreaks, PII leakage, and unsafe content are the LLM-app threat surface.
A keyword blocklist is brittle (attackers paraphrase); a trained classifier is heavy and
out-of-lane for a vector DB. But Veda already turns text into vectors, so "does this *mean*
something like a known attack?" is, again, a cosine.

`agent/guardrails.rs` is opt-in via the `Agent.guardrails` flag (default **false**). It screens
**both the input and the output** for four categories — **prompt-injection / jailbreak / PII /
unsafe** — by combining:

- a **PII regex** (emails, phone numbers, card-shaped numbers, etc.); and
- **max-cosine to a curated bad-pattern embedding set** — if the input/output embeds within
  **≥ 0.80** of any curated bad pattern, that category trips.

Two modes control the response:

- **Flag** — annotate the result and move on (the answer still returns); and
- **Block** — **short-circuit the LLM with a refusal** (the model is never called for a blocked
  input; a blocked output is replaced with a refusal).

A `GuardrailReport` (input verdict, output verdict, triggered categories, mode, method) rides on
the response. Metrics `veda_llm_guardrail_blocked_total` and `veda_llm_guardrail_flagged_total`
track enforcement. Two safety properties matter: with **no embedder** it degrades to
**regex-only** (`method="regex"`, coarser — it can't catch semantic paraphrases, but PII still
fires), and the whole path is **fail-open** — a guardrail evaluation error never fails the
request. The default mode is **Flag**, so enabling guardrails surfaces risk before it ever
blocks a user.

---

## 3. Searchable semantic LLM trace store + failure clustering

GenAI spans answer "what did this one call do?". The operational question is the *aggregate*
one: "show me every answer about X", "which prompts are leaking PII", "what's the recurring way
this agent fails". A span store you query by trace-id can't answer that; a vector store you
query by meaning can.

`agent/trace_store.rs` is opt-in via the `Agent.trace` flag (default **false**). Each completed
run is written to a per-tenant `_llm_traces` collection with the full operational record —
**prompt, response, tool_ids, faithfulness, cost_usd, cache_hit, guardrail_flagged, latency_ms,
ts** — plus an **embedding of prompt+response**. Two surfaces (in `rest/llm_traces.rs`) read it:

- `GET /v1/admin/llm-traces?q=` — **semantic search** over the trace store: embed the query and
  vector-search the stored embeddings, so "ANN index questions" finds the run that asked about
  HNSW tuning even if the words differ; and
- `GET /v1/admin/llm-traces/clusters` — **failure clustering** that groups failing/low-quality
  traces (reusing the same embedding `LogCategorizer` the Log Insights / spike-attribution
  features use) into recurring failure modes, each with a `representative` and a `count`, with a
  normalized fallback when no embedder is configured.

A Niriksha panel renders both. This is the trace store Elastic's LLM Observability is, made
*vector-native*: you search traces by meaning and cluster failures by meaning, on the same
engine that stores them.

Trace records also carry a **retention TTL** so prompts/responses don't persist forever: each
trace sets the reserved `_expires_at` field to `ts + VEDA_LLM_TRACE_TTL_SECS` (default **30
days**), the TTL sweeper auto-purges expired traces, and `search`/`cluster` already exclude
expired records (via `lookup_record_at` / `is_expired_at`) — so a regulated deployment gets
bounded retention without bookkeeping.

---

## 4. TTFT + cost budgets + quality drift + feedback

The prior cycle added RunMetrics, cost/token counters, and faithfulness scoring. This cycle
closes the LLM-ops loop with the four signals an LLM SRE actually watches:

- **TTFT (time-to-first-token).** `RunMetrics` gains `ttft_ms` — the **stream-only** time to the
  first answer token (null on the blocking `/converse` path, which has no streaming notion of
  "first token") — exported as the histogram `veda_llm_ttft_seconds`. TTFT is the latency users
  *feel*, distinct from total run time.
- **Cost budgets.** A new advisor rule **`llm_cost_budget`** (Medium) reads the existing
  `veda_llm_cost_usd_total` counter and opens an event when spend over a window crosses a USD
  budget — so a runaway agent or an expensive model pages you before the bill does.
- **Quality drift.** A new advisor rule **`quality_drift`** (Medium) runs the same two-window /
  change-point statistic the Agentic AIOps cycle introduced, but over the **faithfulness
  results series** — it fires on a *sustained* faithfulness drop (a regression in answer
  grounding), not a single bad turn.
- **Feedback.** `POST /v1/conversations/{id}/feedback { rating: up|down, note? }` records a
  thumbs rating on a conversation and increments `veda_llm_feedback_total{rating}` — the
  human-judgment signal that complements the automated faithfulness score, and a natural source
  for graduating traces into the eval harness.

Both new rules reuse machinery that already existed (the cost counter, the change-point
statistic, the persisted faithfulness series) and flow through the same event store + alert
channels as every other advisor rule.

---

## 5. The vector-indexing fix (insert → seal → rebuild)

This cycle also repaired a coupled pair of bugs that made the cache, agent-memory, and trace
store **silently non-functional** — and the fix is load-bearing for everything above, so it's
worth explaining.

The cache, agent-memory, and trace store all *stored* their vectors by inserting into a
collection's **growing** (unsealed) segment. But Veda's query executor only loads **sealed**
segments — the growing segment isn't searchable until it's sealed and the executors are rebuilt.
So the stored vectors were **written but never searchable**: the cache never hit, agent-memory
was never recalled, and `GET /v1/admin/llm-traces?q=` returned empty. Worse, the lookup filter
used `==`, but the filter parser only accepts a single `=`, so even the metadata lookup was
malformed.

The fix is a shared helper, `agent/index.rs::insert_and_index`, used by all three stores: it
**inserts → seals the growing segment → rebuilds the executors** (off-lock, best-effort) on
every store, and the lookup filter was corrected from `==` to `=`. A searchability regression
test was added (red → green) to lock it in.

**Update (follow-up, RESOLVED):** the per-write seal+rebuild this fix introduced has since been
replaced by **live-growing search** — see the perf follow-up in §8. Stores now `insert_only`
into the growing segment (no per-write seal/rebuild; the growing segment seals on the normal
size threshold), and lookups call `search_live_growing` to brute-force the bounded live growing
segment and **merge** those results with the sealed-executor results. Fresh entries are
searchable immediately, with **zero per-write rebuild**.

Note: this same bug had silently shipped on `main` since the RAG/AIOps cycle, so **agent-memory
was also broken in production** until this fix — the tests had mocked the store path, so they
never exercised the seal-and-search round-trip. The new test now does. (See the perf follow-up
in §8: seal-and-rebuild on *every* store is a real per-write cost.)

---

## 6. REST / metrics / env reference

### REST

| Endpoint | Method | Auth | Description |
|---|---|---|---|
| `/v1/admin/llm-traces` | GET | admin (`bearerAuth`) | Semantic search over the `_llm_traces` store (`?q=&limit=`); prompt/response/tool_ids/faithfulness/cost_usd/cache_hit/guardrail_flagged/latency_ms/ts + score per trace |
| `/v1/admin/llm-traces/clusters` | GET | admin (`bearerAuth`) | Failure clustering over LLM traces (reuses the embedding categorizer) — recurring failure modes (`category`, `representative`, `count`) |
| `/v1/conversations/{id}/feedback` | POST | tenant (`bearerAuth`) | Thumbs feedback `{ rating: up\|down, note? }` on a conversation; emits `veda_llm_feedback_total{rating}` |

`ConverseResponse` (and the streamed `done` frame) now also carry **`cache_hit`** (bool) and a
**`guardrails`** `GuardrailReport` when those flags are enabled; `RunMetrics` carries
**`ttft_ms`**.

### Metrics

| Metric | Type | Notes |
|---|---|---|
| `veda_llm_cache_hits_total` / `veda_llm_cache_misses_total` | counter | Semantic-cache hit-rate (and avoided LLM spend) |
| `veda_llm_guardrail_blocked_total` / `veda_llm_guardrail_flagged_total` | counter | Guardrail enforcement |
| `veda_llm_ttft_seconds` | histogram | Stream-only time-to-first-token |
| `veda_llm_feedback_total{rating}` | counter | Human thumbs up/down |

### Advisor rules

| Rule | Severity | Notes |
|---|---|---|
| `llm_cost_budget` | Medium | USD-per-window over `veda_llm_cost_usd_total` crosses a budget |
| `quality_drift` | Medium | Sustained faithfulness drop via the two-window / change-point statistic over the faithfulness series; brings the catalog to ~36 rules |

### Agent flags (all opt-in, default `false`)

| Flag | Effect |
|---|---|
| `cache` | Enables the semantic response cache (per-tenant `_llm_cache`, cosine ≥ 0.97) |
| `guardrails` | Enables embedding+regex input/output safety screening (mode Flag\|Block) |
| `trace` | Enables writing each run to the searchable `_llm_traces` store |

Cache/guardrails/trace are per-agent flags; the cost-budget threshold is configured like any
advisor rule via `/v1/admin/insights/rules`. The one dedicated env knob is the trace-store
retention TTL:

### Env

| Variable | Default | Notes |
|---|---|---|
| `VEDA_LLM_TRACE_TTL_SECS` | `2592000` (30d) | Retention TTL for the `_llm_traces` store — each trace's reserved `_expires_at` = `ts + this`; the TTL sweeper auto-purges expired traces and `search`/`cluster` exclude them |

---

## 7. Validation (live)

Verified against a running server on the Linux host (**after** the §5 vector-indexing fix —
before it, the cache never hit):

- **Semantic cache (end-to-end):** an agent with `cache` + `guardrails` + `trace` enabled was
  asked the same prompt twice. The first run executed normally; the **second returned
  `cache_hit:true` in 0 steps with no LLM call** — the near-duplicate prompt was served straight
  from `_llm_cache`.
- **Guardrails:** the run returned a `GuardrailReport` with **allow/flag verdicts on both the
  input and the output** — the screening path ran on both sides.
- **Trace search:** `GET /v1/admin/llm-traces?q=ANN index algorithm` returned the stored trace
  at **score ~0.61** — semantic search over the trace store works, confirming the
  insert→seal→rebuild fix.

---

## 8. Limitations & follow-ups

- **The cache and trace store persist prompts and responses.** Both are **opt-in**, written to
  **per-tenant** collections, and **tenant/RLS-scoped** on read — so enabling them remains a
  privacy decision, but the trace store no longer retains text indefinitely: it **gained a
  retention TTL** (**RESOLVED**, follow-up `6caf0fe`) — each trace
  sets the reserved `_expires_at` = `ts + VEDA_LLM_TRACE_TTL_SECS` (default **30 days**), the TTL
  sweeper auto-purges expired traces, and `search`/`cluster` exclude expired records (via
  `lookup_record_at` / `is_expired_at`), so prompts/responses auto-expire for regulated data.
- **Cache threshold + TTL.** The 0.97 cosine threshold trades recall (fewer hits) for safety
  (no wrong-answer reuse); it is fixed today and should become tunable. Cached entries carry a
  TTL (`DEFAULT_TTL_MS`, 24h) — an entry whose `expires_at` has passed is treated as a miss — so
  a stale answer can be served only within that window before re-generation.
- **Guardrails are embedding-similarity + regex, not a trained classifier.** Max-cosine to a
  curated bad-pattern set catches paraphrased attacks the curated set covers, but it is not a
  fine-tuned safety model — novel attack phrasings outside the set, and adversarial evasion, are
  out of reach. The default mode is **Flag** (observe before you block) precisely because this
  is a heuristic, not a verdict.
- **Per-write seal + rebuild on every store — RESOLVED via live-growing search** (follow-up
  `891d6c7`). The §5 fix made stored vectors searchable by sealing the growing segment and
  rebuilding the executors on **every** cache/memory/trace write — correct, but O(collection)
  and expensive for high-volume traffic (many tiny sealed segments + a rebuild each time). The
  follow-up removes the per-write seal/rebuild entirely: stores now `insert_only` into the
  growing segment (which seals on the normal size threshold), and lookups call
  `search_live_growing` to brute-force the **bounded** live growing segment and **merge** with
  the sealed-executor results. Fresh entries are searchable immediately with **zero per-write
  rebuild**. Live-validated: a repeated prompt still returns `cache_hit:true`, and 15 distinct
  stores produced **0 sealed segments** (was ~15).
- **Guardrails / cache / trace need an embedder.** All three are vector-native, so without an
  embedder the cache and trace search are inert and guardrails fall back to **regex-only**
  (PII still fires; semantic paraphrase detection does not). A CPU embedder (BGE-M3) is fine —
  no GPU is required — and `method` always reports which path ran.
- **Cost-budget and quality-drift need data and warm-up.** `llm_cost_budget` needs accumulated
  spend in the window, and `quality_drift` needs enough faithfulness samples for the two-window
  statistic to clear its warm-up — neither fires on a fresh, idle deployment by design.

# LLM Observability & Optimization

This is the lane where most observability tools are thinnest: they *ingest* the GenAI traces an app emits but can't score grounding, can't cache by meaning, and can't screen by meaning — they observe the LLM from the outside. Veda already shipped the half that needs the model's own signals (answer [citations + faithfulness scoring](/docs/rag-aiops#faithfulness-citations), a [RAG/agent eval harness](/docs/rag-aiops#eval-harness-the-production-trace-flywheel), native [GenAI-semconv spans + cost/token metrics](/docs/apm-observability#genai-rag-tracing-run-metrics-cost)). This layer adds the part nobody else in the lane can: because **Veda owns the embedder, the retriever, the agent loop, and the faithfulness scorer**, it can turn its vector engine *into* the LLM layer.

A near-duplicate prompt is a vector search away from a **cached answer with no LLM call**; an unsafe prompt is a cosine away from a **blocked request**; a recurring bad answer is a cluster away from a **named failure mode**. Where most tools *observe* LLM apps, Veda **observes *and* optimizes/secures** them. Every feature reuses an existing primitive (collections + embeddings + the agent executor + the faithfulness scorer + the [Ops Advisor](/docs/ops-advisor)) and degrades gracefully — regex-only guardrails and a fail-open path without an embedder. The three new agent flags (`cache` / `guardrails` / `trace`) are **opt-in (default false)**, so existing agents are unchanged.

## Semantic response cache

LLM apps re-ask the same thing constantly — paraphrased, re-worded, but *meaning* the same. A string-keyed cache misses all of it ("How do I tune the ANN index?" ≠ "what's the best way to configure HNSW?"). Veda already embeds text and searches it by cosine, so a semantic cache is the natural build.

Enabled via the `cache` agent flag. Before a run, the executor **embeds the prompt** and **vector-searches a per-tenant `_llm_cache` collection** for a near-duplicate:

- on a **hit** (cosine **≥ 0.97**) it returns the cached response **without calling the LLM** — the response carries `cache_hit: true` and **`steps` is 0** (no reasoning, no tool calls, no model latency, no cost). In a live test a repeated prompt came back **~2× faster with zero LLM tokens**;
- on a **miss** it runs the agent normally and **stores** the `{prompt, response}` (embedded) back into `_llm_cache` for next time.

The threshold is deliberately high (0.97) so only genuinely equivalent prompts collapse — a "related but different" question still runs. `veda_llm_cache_hits_total` / `veda_llm_cache_misses_total` make the hit-rate (and the spend you avoided) observable. The cache *is* a collection, the lookup *is* a search.

## Embedding-based guardrails

Prompt-injection, jailbreaks, PII leakage, and unsafe content are the LLM-app threat surface. A keyword blocklist is brittle (attackers paraphrase); a trained classifier is heavy and out-of-lane for a vector DB. But "does this *mean* something like a known attack?" is, again, a cosine.

Enabled via the `guardrails` flag, it screens **both the input and the output** for four categories — **prompt-injection / jailbreak / PII / unsafe** — by combining a **PII regex** (emails, phone numbers, card-shaped numbers) with **max-cosine to a curated bad-pattern embedding set** (trips above a threshold, default **0.65**, recalibrated per embedder). The verdict always surfaces the strongest observed cosine in `score`, so sub-threshold near-misses are observable. Two modes:

- **Flag** *(default)* — annotate the result and move on (the answer still returns);
- **Block** — short-circuit the LLM with a refusal (the model is never called for a blocked input; a blocked output is replaced with a refusal).

A `GuardrailReport` (input verdict, output verdict, triggered categories, mode, method) rides on the response, and `veda_llm_guardrail_blocked_total` / `veda_llm_guardrail_flagged_total` track enforcement. Two safety properties: with **no embedder** it degrades to **regex-only** (`method: "regex"` — PII still fires, but it can't catch semantic paraphrases), and the whole path is **fail-open** — a guardrail evaluation error never fails the request. The default Flag mode surfaces risk before it ever blocks a user.

## Searchable trace store & failure clustering

GenAI spans answer "what did this one call do?". The operational question is the *aggregate* one: "show me every answer about X", "which prompts are leaking PII", "what's the recurring way this agent fails". A span store you query by trace-id can't answer that; a vector store you query by meaning can.

Enabled via the `trace` flag, each completed run is written to a per-tenant `_llm_traces` collection with the full operational record — prompt, response, `tool_ids`, `faithfulness`, `cost_usd`, `cache_hit`, `guardrail_flagged`, `latency_ms`, `ts` — plus an **embedding of prompt+response**. Two surfaces read it:

- `GET /v1/admin/llm-traces?q=` — **semantic search** over the trace store: embed the query and vector-search the stored embeddings, so "ANN index questions" finds the run that asked about HNSW tuning even if the words differ;
- `GET /v1/admin/llm-traces/clusters` — **failure clustering** that groups failing / low-quality traces (reusing the same embedding [categorizer](/docs/log-insights#semantic-categorization) the Log Insights features use) into recurring failure modes, each with a `representative` and a `count`.

Trace records carry a **retention TTL** so prompts/responses don't persist forever: each trace sets the reserved [`_expires_at`](/docs/collections#record-ttl) to `ts + VEDA_LLM_TRACE_TTL_SECS` (default **30 days**), the TTL sweeper auto-purges expired traces, and search/cluster exclude expired records — so a regulated deployment gets bounded retention without bookkeeping.

## TTFT, cost budgets, quality drift & feedback

Four more signals close the LLM-ops loop:

- **TTFT (time-to-first-token).** `RunMetrics` gains `ttft_ms` — the **stream-only** time to the first answer token (null on the blocking `/converse` path) — exported as the histogram `veda_llm_ttft_seconds`. TTFT is the latency users *feel*, distinct from total run time.
- **Cost budgets.** The advisor rule **`llm_cost_budget`** (Medium) reads the existing `veda_llm_cost_usd_total` counter and opens an event when spend over a window crosses a USD budget — so a runaway agent or an expensive model pages you before the bill does.
- **Quality drift.** The advisor rule **`quality_drift`** (Medium) runs the same [two-window / change-point statistic](/docs/aiops#change-point-detection) over the **faithfulness results series** — it fires on a *sustained* grounding regression, not a single bad turn.
- **Feedback.** `POST /v1/conversations/{id}/feedback { rating: up|down, note? }` records a thumbs rating and increments `veda_llm_feedback_total{rating}` — the human-judgment signal that complements the automated faithfulness score, and a natural source for graduating traces into the [eval harness](/docs/rag-aiops#eval-harness-the-production-trace-flywheel).

Both new rules flow through the same event store + alert channels as every other [advisor rule](/docs/ops-advisor#the-rule-catalog).

## Agent flags

All three are per-agent and **opt-in** (default `false`), so existing agents are byte-for-byte unchanged:

| Flag | Effect |
|---|---|
| `cache` | Semantic response cache (per-tenant `_llm_cache`, cosine ≥ 0.97) |
| `guardrails` | Embedding + regex input/output safety screening (mode Flag \| Block) |
| `trace` | Writes each run to the searchable `_llm_traces` store |

Set them on an [agent definition](/docs/agents-mcp#custom-agents). The cost-budget threshold is tuned like any rule via `/v1/admin/insights/rules`.

## REST, metrics & configuration

| Method | Path | Description |
|---|---|---|
| `GET` | `/v1/admin/llm-traces` | Semantic search over the trace store (`?q=&limit=`) — admin-only |
| `GET` | `/v1/admin/llm-traces/clusters` | Failure clustering — recurring failure modes (`category`, `representative`, `count`) — admin-only |
| `POST` | `/v1/conversations/{id}/feedback` | Thumbs feedback `{ rating: up\|down, note? }` on a conversation |

`/converse` (and the streamed `done` frame) now also carry `cache_hit` and a `guardrails` report when those flags are enabled; `RunMetrics` carries `ttft_ms`.

| Metric | Type |
|---|---|
| `veda_llm_cache_hits_total` / `veda_llm_cache_misses_total` | counter |
| `veda_llm_guardrail_blocked_total` / `veda_llm_guardrail_flagged_total` | counter |
| `veda_llm_ttft_seconds` | histogram |
| `veda_llm_feedback_total{rating}` | counter |

| Variable | Default | Purpose |
|---|---|---|
| `VEDA_LLM_TRACE_TTL_SECS` | `2592000` (30d) | Retention TTL for the `_llm_traces` store — each trace's `_expires_at` = `ts + this` |

## Current limitations

- **The cache and trace store persist prompts and responses** — both are opt-in, per-tenant, and tenant/RLS-scoped on read, so enabling them is a privacy decision; the trace store auto-expires text via the retention TTL above.
- **Cache threshold + TTL are fixed today** — the 0.97 cosine threshold trades recall for safety (no wrong-answer reuse); cached entries also carry a 24h TTL, after which an entry is treated as a miss. Both should become tunable.
- **Guardrails are embedding-similarity + regex, not a trained classifier** — max-cosine to a curated bad-pattern set catches paraphrased attacks the set covers, but novel phrasings and adversarial evasion are out of reach. The default Flag mode is deliberate: observe before you block.
- **Cache / guardrails / trace need an embedder** — all three are vector-native, so without one the cache and trace search are inert and guardrails fall back to regex-only (PII still fires). A CPU embedder (BGE-M3) is fine; no GPU is required.
- **Cost-budget and quality-drift need data and warm-up** — neither fires on a fresh, idle deployment by design.

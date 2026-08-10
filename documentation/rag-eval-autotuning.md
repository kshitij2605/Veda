# Veda RAG Eval & Auto-Tuning — Pipeline Reference

**Status:** Living reference. Consolidates the auto-tuning story shipped across two cycles —
the v1 MVP (`2026-07-02-automated-rag-eval-autotuning-{design,plan}.md`: synthetic text QA
generation, the nested campaign engine, safe-apply) and v2 (`2026-07-05-rag-eval-autogen-v2-
{design,plan}.md`: dedicated production-safe LLM config, corpus-diverse sampling, the
ingestion auto-trigger, dual Quality/Performance tuning modes). This is the single overview of
how the pipeline actually works today, including two real bugs found and fixed during live
end-to-end testing against a real multi-document corpus. The dated design/plan docs carry the
phase-by-phase build history; this doc carries the current architecture.

Veda is the only vector DB in this space that owns the whole RAG stack — the corpus, the
chunker, the embedder, the retriever, the reranker, and the LLM judge. That means it can
**generate its own labeled test set from a user's real data** and **search its own
configuration space** for the best pipeline, instead of asking the operator to hand-pick a
chunk size and hope. Nothing else in this market does both halves in one product.

## The pipeline, end to end

```
  ingest a document (process-document / ingest-url)
        │  ingest_markdown: chunk → embed → insert → seal → rebuild_executors
        ▼
  tokio::spawn (fire-and-forget, never blocks the ingest response)
        │
        ▼
  maybe_trigger_auto_pipeline   (auto_optimization.rs)
        │  only if the collection opted in via PUT .../auto-optimization
        │  and an ANSWER_GEN_* LLM endpoint is configured
        ▼
  1. generate_eval_set_for_collection   (rag_eval.rs)
        │  sample_diverse_chunks: document- and element-type-weighted, easy/medium/hard
        │  semaphore(5)-bounded LLM calls generate grounded QA pairs
        │  persists as EvalSet id "auto-eval-<collection>" (deterministic — overwrites,
        │  doesn't accumulate, on every re-trigger)
        ▼
  2. run_and_persist_campaign_with_mode   (tuning.rs / tuning/campaign.rs)
        │  nested cost-aware search: LHS over tier2/3 bases (each = one /reindex),
        │  NSGA-II over tier1 within each base, global Pareto front across all trials
        │  mode = Quality (multi-objective front) or Performance (fastest above a
        │  recall floor from an explicit baseline evaluation)
        ▼
  TuningCampaign persisted (SavedObject, type "tuning_campaign")
        │
        ▼
  operator reviews the Pareto front in Niriksha → Quality Lab → Campaigns
        │  "Apply" a trial: tier1-only = instant/reversible; tier2/3 = /reindex + alias-swap
        ▼
  collection alias now points at the applied config's sibling (revert = swap back)
```

Every step reuses an existing Veda primitive — the embedder, the LLM client, `/reindex`, the
`SavedObjectStore`, the scheduler's off-tick pattern — rather than a bespoke subsystem. See
`crates/veda-gateway/src/rest/{rag_eval.rs, auto_optimization.rs, tuning.rs}` and
`crates/veda-gateway/src/tuning/campaign.rs`.

---

## 1. Test-set generation

**`POST /v1/collections/{name}/eval-sets/generate`** (`rag_eval.rs`), body
`{ name?: string, num_cases: number }`.

### Diverse, weighted sampling

A flat random sample of chunks tends to over-represent large documents and text-only content.
`sample_diverse_chunks` fixes this in three steps:

1. **`document_groups`** — groups chunks by `parent_doc_id` (orphans become singleton groups),
   weighted by total text length.
2. **`apportion`** — largest-remainder apportionment of the target case count across document
   weights, with a fairness pass: if `target_count >= number_of_documents`, any document that
   would organically round to zero is bumped to 1 by stealing from whichever document holds the
   most — every real document gets represented at least once when the budget allows it.
3. **`classify_element`** — heuristically tags each chunk as `Table` (≥2 lines with ≥2 pipe
   characters), `Image` (`![...](...)` markdown), or `Text`, and rounds robin across element
   types within each document's allocation — so a document with both prose and embedded tables
   gets questions asked from *both*, not just whichever the sampler happened to see first.

### Difficulty

Cases are labeled `easy` / `medium` / `hard` in equal thirds (`default_difficulty_mix`,
1.0/1.0/1.0 weights — the round-robin sampling already produces near-equal thirds).

### The LLM call

Each sampled chunk becomes one `generate_one_qa` call to a **dedicated** answer-generation LLM
— see [§4](#4-the-dedicated-llm-config--semaphore) — producing a grounded
`{ query, expected_chunk_ids, gold_answer, difficulty }` case.

### Size recommendation

**`GET /v1/collections/{name}/eval-set-size-recommendation`** returns
`{ recommended, max, chunk_count, document_count }` where
`recommended = max(document_count * 2, 4)` capped at `chunk_count`, and `max = chunk_count`
(never let an operator ask for more questions than the corpus has chunks). Surfaced live in the
Niriksha "Generate from documents" modal — typing a collection name shows the caption
`"(N recommended for this index's size — D documents, C chunks)"` and the number input is
clamped to `max`.

---

## 2. The auto-trigger

**`GET/PUT /v1/collections/{name}/auto-optimization`** (`auto_optimization.rs`), body
`{ enabled: bool, target_questions?: number, mode: "Quality" | "Performance" }`, persisted as a
`SavedObject` (type `auto_optimization_config`, id = the scoped collection name). Default is
`enabled: false` — a collection is opt-in only.

When a document finishes ingesting (`ingest_common.rs::ingest_markdown`, used by both
`process-document` and `ingest-url`), its tail spawns `maybe_trigger_auto_pipeline` via
`tokio::spawn` — **fire-and-forget**: the ingest HTTP response returns immediately regardless
of how long generation + the campaign take, and every failure path inside the trigger is a
`tracing::warn!` and an early return, never a panic that could surface to the caller.

If the collection is opted in and an `ANSWER_GEN_*` endpoint is configured, the trigger:

1. Computes `target_questions` from the config override or the same
   `document_count * 2` (capped) formula as the size-recommendation endpoint.
2. Calls `generate_eval_set_for_collection` with a **deterministic** `preferred_id`
   (`auto-eval-<collection>`) — so re-triggering on every subsequent ingestion *overwrites* the
   same eval set with a fresh one reflecting the now-larger corpus, rather than accumulating a
   new orphaned set on every upload. (The human/UI-triggered generate endpoint still gets a
   fresh random `eval-<uuid>` each time — only the auto-trigger path is deterministic.)
3. Calls `run_and_persist_campaign_with_mode` with the configured `mode` and a small default
   budget (3 outer bases).

This re-runs on **every** ingestion into an opted-in collection, not just once — the eval set
and the tuning recommendation always reflect the corpus's current state.

---

## 3. Dual tuning modes

Both modes share the same nested search loop (`tuning/campaign.rs::run_campaign_with_mode`);
they differ only in how the winning trial(s) are selected.

**`POST /v1/collections/{name}/tuning-campaigns`**, body
`{ eval_set_id, max_trials?, num_bases?, mode? }` (`mode` defaults to `Quality`).

### The search itself

A **nested, cost-aware** loop avoids re-chunking/re-embedding on every trial:

1. **Outer sampling** (tier 2/3 — chunker strategy, embedding model, index type): Latin
   Hypercube Sampling picks a handful of distinct "base configurations" (`num_bases`, default
   3–5). Each base gets exactly **one** `/reindex` call materializing a sibling collection —
   the expensive step, sampled sparingly.
2. **Inner search** (tier 1 — `ef_search`, fusion mode/weights, rerank, `top_k`): within each
   materialized base, `scirs2-optimize`'s NSGA-II runs many generations, entirely query-time,
   no rebuild.
3. Every trial across every base carries the full combined config and an objective vector
   `{ recall, faithfulness, latency_p50_ms }`.

### Scoring recall when the base re-chunked

Eval cases pin their ground truth as `expected_chunk_ids`. A **tier-3** base re-chunks the
corpus, and a re-chunk mints fresh record ids, so id matching can never match anything such a
trial retrieves.

Left alone this is not a rounding error — it scores *every* chunker-varying trial `recall =
0.000` by construction, no matter how good the chunking is. The tier-3 dimension of the search
then can't be won on recall at all, and a chunker that is genuinely worse still reaches the
Pareto front on another objective and gets offered to the operator as a trade-off worth
applying. (Found live: a `Page`-chunked base scored a flat 0.000 across all of its trials while
the two `MarkdownRecursive` bases scored 0.75–1.00, and the 0.000 trial sat on the front on
faithfulness alone.)

So when a trial's base differs from the live config in tier 3, recall is scored on **content**
instead of ids:

- The live collection's `chunk_id → text` map is resolved **once per campaign**
  (`load_source_chunks`), not per trial.
- Each expected chunk id resolves to its gold text; the trial's top-k results are reduced to a
  union of case-folded, whitespace-stripped **character 5-grams** (character-level because the
  corpora include Japanese, which has no whitespace token boundaries).
- A gold chunk counts as retrieved when **≥ 50%** of its shingles appear in that union. The
  union — rather than any single retrieved chunk — is the right model: a re-chunk legitimately
  spreads one gold chunk across several retrieved ones, and surrounding text in a larger chunk
  must not dilute the score, since containment is measured against the *gold's* shingles.
- If no expected id resolves to gold text, the case is left **unscored** rather than counted as
  a miss — the same "unscored signals don't fail" convention the faithfulness proxy uses.

Trials on the live chunking (tier-1 and tier-2 bases, where a straight-copy reindex preserves
record ids) keep exact id matching. After the fix the same `Page` base scored a real **0.375** —
still worse than `MarkdownRecursive` on that corpus, which is the correct conclusion, but now
measured rather than structurally impossible.

### Quality mode

The classic case: the global Pareto front is computed over *every* trial from *every* base — a
genuinely joint optimum across the whole tier1×tier2×tier3 space, despite staged execution.
`pareto_front` on the returned `TuningCampaign` can hold multiple trial ids, each representing
a different, non-dominated recall/faithfulness/latency trade-off.

### Performance mode

Selects a **single** winner: the lowest-`latency_p50_ms` trial among those whose `recall` does
not fall below a **recall floor**. The floor is *not* found by searching for a trial that
happens to match the default config — the search loop perturbs tier1 on every trial (including
ones descended from the default outer base), so no trial is ever guaranteed to organically
reproduce `PipelineConfig::default()` exactly. Instead, `mark_and_collect_performance_winner`
runs one **explicit, guaranteed baseline evaluation** of `PipelineConfig::default()` before
selecting, and uses *that* trial's recall as the floor. (An earlier version of this logic tried
to find the default config among the organic trials and silently fell back to a floor of `0.0`
when it didn't — defeating the entire point of the guard. Caught by a flaky test during
development; fixed by making the baseline explicit and unconditional.)

`pareto_front` on a Performance-mode campaign always holds exactly one trial id.

---

## 4. The dedicated LLM config + semaphore

`ANSWER_GEN_BASE_URL` / `ANSWER_GEN_API_KEY` / `ANSWER_GEN_MODEL` /
`ANSWER_GEN_TIMEOUT_SECS` (`LlmConfig::from_env_answer_gen`, `crates/veda-core/src/
llm_client.rs`) configure a **separate** LLM connection from the agent-facing `VEDA_LLM_*`
config — a deliberate choice so eval-set generation and campaign scoring never compete with (or
get starved by) an operator's chat/agent traffic on a shared production LLM endpoint.

Every call through this config acquires a permit from `AppStateInner.answer_gen_semaphore`
(`tokio::sync::Semaphore`, capacity 5) before making a request — bounding concurrent load
against the answer-gen endpoint regardless of how many chunks a generation batch or how many
concurrent generation requests are in flight. Verified live: 3 concurrent 8-case generation
requests (24 total LLM calls) completed in ~7s with no deadlock and no more than 5 concurrent
requests observed against the endpoint.

If `ANSWER_GEN_BASE_URL` is unset, `from_env_answer_gen` returns `None` and the auto-trigger
is a clean no-op (checked explicitly before doing any work) — the feature is fully opt-in at
the infrastructure level, not just the per-collection level.

---

## 5. Safe apply

Recommend-only — a campaign never auto-applies a trial. **`POST /v1/tuning-campaigns/
{campaign_id}/trials/{trial_id}/apply`**:

- **Tier1-only diff** (the trial's config differs from the collection's live default only in
  `ef_search`/fusion/rerank/`top_k`): applied **instantly and reversibly** — just updates the
  collection's default query parameters, no data touched.
- **Tier2/3 involved**: triggers the existing `/reindex` primitive, which — as it already did
  before this feature existed — produces a **new sibling** collection (`name@vN`) and never
  overwrites the original. Revert = swap the alias back to the previous version; the prior
  sibling is never deleted by this feature.

This is why alias correctness (§6) is load-bearing for the entire "apply" half of the pipeline,
not a side detail.

---

## 6. Niriksha UI

**Eval Harness** (`pages/quality/EvalHarness.tsx`) — a "Generate from documents" modal
alongside the existing manual set-creation flow: type a collection name, see the live size
recommendation, optionally check "Auto-regenerate this testset and re-tune every time this
collection grows" (calls `setAutoOptimizationConfig` right after generation — the discoverable
opt-in point for §2's trigger).

**Quality Lab → Campaigns** (`pages/quality/Campaigns.tsx` + `QualityLab.tsx`) — a per-
collection campaign list, a "+ New campaign" form with an eval-set picker, max-trials, and an
**Optimization mode** selector (`Quality` / `Performance`), and a `CampaignDetail` view showing
the Pareto front as a sortable table (recall / faithfulness / p50 latency / Apply) — for
Performance mode this table always has exactly one row.

---

## Two real bugs found via live end-to-end testing

Both of the following were found only once this pipeline was run against a real, large,
OCR'd multi-document corpus rather than a small hand-crafted eval set — the kind of testing
this reference exists partly to make repeatable.

### `chunk_by_page` unbounded output

`ChunkingStrategy::Page` (`crates/veda-core/src/chunking.rs`) split on literal `"\n---\n"`
page separators with no size cap, unlike every other chunking strategy in the file. An OCR'd
page with a large embedded table/figure could produce a single chunk exceeding an embedding
model's max input size (bge-m3's 32768 bytes) — and because a Quality-mode campaign's default
outer bases always include one `ChunkerStrategyChoice::Page` base, *any* default Quality
campaign against a corpus with one sufficiently large OCR'd page would hard-fail its `/reindex`
call, and with it the whole campaign. Fixed by sub-splitting oversized pages via the existing
`chunk_fixed_size` helper, matching the pattern already used by `chunk_markdown_recursive` and
`chunk_markdown_heading`.

### `ingest_markdown` never resolved collection aliases

The more serious of the two. Every *read* path — `list_records`, `get_record`, `search`, the
manual `/insert` — resolves the caller's collection name through the alias map
(`state.inner.aliases.resolve`) before touching `segment_managers`, because §5's apply flow can
repoint a collection's alias at any time. `ingest_markdown` — the shared helper behind both
`process-document` and `ingest-url` — never did. A collection whose alias had been swapped (by
an earlier `/reindex`, or an applied tuning trial) would silently accept new documents into the
**abandoned base collection**, while every read continued to hit the alias target. The ingest
endpoint reported `"status":"processed"` with a plausible `records_inserted` count, and the
data was structurally valid and correctly sealed to disk — genuinely, permanently unreachable
by any query.

This was, in fact, exactly the risk the v1 design doc's Phase 0 called out by name — non-Raft
alias resolution was flagged as "correctness-critical, wide-blast-radius" and required for
*every* collection-name lookup path — but the actual Phase 0 implementation covered the read
paths and missed this one write path. Fixed by resolving the alias once at the top of
`ingest_markdown`, matching the existing read-path pattern; a regression test
(`test_ingest_markdown_follows_an_alias_to_the_swapped_collection`,
`crates/veda-gateway/tests/ingest_url_test.rs`) creates a collection, swaps its alias, ingests
through the original name, and asserts the record is queryable through it — mirroring the
existing `get_record_follows_an_alias_to_the_swapped_collection` test for the read side.

---

## Verified live, end to end

Against a real ingested corpus (7 real OCR'd documents, ~190 chunks, real BGE-M3 embeddings,
real `Qwen/Qwen3.5-122B-A10B` answer generation and LLM-judge scoring — no mocks):

- A fresh ingestion into an opted-in collection auto-generated a real 14-case eval set
  (`document_count * 2` for 7 documents) spanning all 7 documents and an image-sourced
  question, and auto-started a Quality-mode campaign that completed with a genuine 2-point
  Pareto front (recall spanning 0.0–0.714, latency 0.78–3.17ms) — not the degenerate
  "recall 1.0 everywhere" result a small, low-diversity eval set produces.
- A Performance-mode campaign against the same eval set found a trial with **identical**
  recall (0.643) and faithfulness (0.531) to the explicit default-config baseline, at **2.27×
  lower latency** (4.42ms vs. 10.03ms) — the recall floor held exactly as designed while
  finding a genuine, non-trivial latency win.

Neither result is a claim of the mathematically provable global optimum (`num_bases`/
`max_trials` bound a heuristic search, not an exhaustive one), but both demonstrate the search
finding real, materially different, defensible configurations on real data — which is the
substantive property this pipeline exists to provide.

---

## Key files

| Concern | File |
|---|---|
| Dedicated LLM config + semaphore | `crates/veda-core/src/llm_client.rs` |
| Diverse sampling, difficulty, eval-set generation | `crates/veda-gateway/src/rest/rag_eval.rs` |
| Auto-optimization config + trigger | `crates/veda-gateway/src/rest/auto_optimization.rs` |
| Shared ingest → chunk → embed → insert → seal pipeline | `crates/veda-gateway/src/rest/ingest_common.rs` |
| Campaign REST surface, apply | `crates/veda-gateway/src/rest/tuning.rs` |
| Nested search loop, dual-mode selection | `crates/veda-gateway/src/tuning/campaign.rs` |
| Search-space parameter encoding | `crates/veda-gateway/src/tuning/config.rs` |
| Niriksha eval-set generation UI | `niriksha/src/pages/quality/EvalHarness.tsx` |
| Niriksha campaign UI | `niriksha/src/pages/quality/{Campaigns,QualityLab}.tsx` |

## Related docs

- `2026-07-02-automated-rag-eval-autotuning-design.md` / `-plan.md` — the v1 MVP design (test-set
  generation, the campaign engine, safe apply).
- `2026-07-05-rag-eval-autogen-v2-design.md` / `-plan.md` — the v2 design (dedicated LLM config,
  diverse sampling, the auto-trigger, dual tuning modes).

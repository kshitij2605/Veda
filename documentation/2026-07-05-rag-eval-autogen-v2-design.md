# RAG Eval Auto-Generation v2 + Dual-Mode Tuning — Design

> Extends `2026-07-02-automated-rag-eval-autotuning-{design,plan}.md` (already shipped).
> This is the next iteration, covering: a dedicated answer-generation LLM, diverse/leveled
> testset generation, opt-in auto-trigger on ingestion, a Niriksha sizing control, and a
> second "performance-only" tuning mode.

## Context

Live end-to-end testing against a real 184-chunk corpus (18 Japanese business PPTX decks,
see the 2026-07-04/05 real-data E2E campaign) found the shipped `generate_eval_set` feature
has three real gaps once you look past "does it run":

1. **Chunk sampling is document-blind.** `load_source_chunks` walks segments in raw storage
   order and takes the first `MAX_SOURCE_CHUNKS=200`. For any corpus larger than ~200 chunks
   this silently skews toward whichever documents were ingested first — the opposite of
   "diverse across all documents."
2. **No element-type awareness.** OCR-produced markdown chunks routinely contain image
   references (`![](images/...)`) and markdown tables inline, but nothing in the sampler
   distinguishes them from prose, so generated questions cluster on whichever type is most
   common in the corpus (usually prose).
3. **Difficulty taxonomy mismatch.** The existing `default_difficulty_mix()` is RAGAS-style
   (`simple`/`reasoning`/`multi_context`, nominally 40/40/20) rather than the
   easy/medium/hard split product wants. The actual sampling loop round-robins by index
   (not by weight), so real output is already ~equal thirds regardless of the stored
   weights — this needs relabeling, not a new algorithm.

Separately, the user has approved: a dedicated (semaphore-limited) LLM for generation
distinct from any agent-facing LLM config; an opt-in per-collection auto-trigger that
regenerates the testset and re-runs a tuning campaign every time the collection grows;
a Niriksha control for testset size with a corpus-aware recommendation; and a second,
explicitly "performance-only" tuning mode alongside the existing quality-weighted one.

## Decisions (confirmed with user)

- `ANSWER_GEN_BASE_URL` / `ANSWER_GEN_API_KEY` / `ANSWER_GEN_MODEL` is a **dedicated**
  config, separate from any other LLM config in the gateway (agents/triage use their own).
  Confirmed live: `Qwen/Qwen3.5-122B-A10B` at `http://difgpu01.tdc.otsuka-shokai.co.jp:9000/v1`,
  `max_model_len: 128000` (not 64k as assumed — no truncation concern for ~1500-char chunks).
- Concurrency to this endpoint is bounded by a **`tokio::sync::Semaphore(5)`** — shared
  production infra, never more than 5 in-flight requests regardless of corpus size.
- Auto-optimization is **opt-in per collection, default `false`**.
- When enabled + `ANSWER_GEN_*` configured: **fully hands-off** — testset generation
  automatically chains into starting a tuning campaign (no manual step).
- **Re-triggers on every ingestion** that adds data to the collection (not just the first).
  Each trigger **regenerates the testset from scratch** at the newly-recommended size
  (proportional to the grown corpus) rather than incrementally appending — diversity/
  difficulty balance has to be recomputed against the whole corpus anyway.
- Performance-only mode **keeps a quality guard**: it minimizes p50 latency but only among
  trials that clear a recall floor (default = the live collection's own current recall,
  so switching to performance mode can never make retrieval quality worse than today).
- Scope: auto-trigger hooks the **document ingestion paths** (`process-document(s)`,
  `ingest-url` — both funnel through the shared `ingest_common::ingest_markdown`), not raw
  `/insert`/`/embed-insert`, since "documents/tables/images" framing is document-pipeline-
  specific.

## Architecture

### 1. Dedicated answer-gen LLM config + semaphore (`veda-gateway`)

- `AppStateInner` gains `answer_gen_config: RwLock<Option<AnswerGenConfig>>` (same shape/
  read pattern as the existing `llm_config`, populated from `ANSWER_GEN_*` env vars at
  startup) and `answer_gen_semaphore: Arc<tokio::sync::Semaphore>` (5 permits, fixed —
  not user-configurable in v1, YAGNI).
- `generate_one_qa` (and any future answer-gen call site) acquires a permit
  (`semaphore.acquire().await`) before calling `LlmClient::complete`, drops it on return.
  This is the ONLY place concurrency is bounded — no separate rate limiter needed since a
  bounded semaphore already caps in-flight requests exactly.

### 2. Diverse, leveled chunk sampling (`rag_eval.rs`)

Replace `load_source_chunks`'s flat "first 200 in storage order" with a two-stage sampler:

**Stage A — group + weight by document.** Group all chunks by `parent_doc_id` (chunks
without one — e.g. from `/insert` records ingested outside the document pipeline — form
their own singleton group, keyed by chunk id, so the algorithm degrades gracefully rather
than crashing on non-document collections). Each group's weight = sum of its chunks' text
length (proportional-to-length, per the user's explicit ask — a short doc gets fewer
questions than a long one even with the same chunk count).

**Stage B — apportion target count across groups.** Given a target question count `N`
(from the Niriksha size control or the recommendation formula), allocate per-group counts
via largest-remainder apportionment on the weight fractions, so the allocations sum to
exactly `N` and every non-trivial document gets a fair share.

**Stage C — element-type tagging + in-group diversity.** Each chunk gets a cheap heuristic
tag: `Table` (contains ≥2 lines with ≥2 `|` pipe characters each, a standard markdown-table
shape), `Image` (contains `![...](...)` syntax), else `Text`. Within a document's allocated
count, chunks are drawn round-robin across whichever tags are actually present in that
document (so a doc with both prose and a table won't get all-prose questions), falling back
to plain round-robin over all its chunks once a tag bucket is exhausted.

**Stage D — difficulty (unchanged mechanism, relabeled).** `default_difficulty_mix()`
becomes `[("easy", 1.0), ("medium", 1.0), ("hard", 1.0)]` — equal-weighted labels; the
existing round-robin-by-index cycling already yields ~equal thirds (this was already true
before, just mislabeled), so no change to `generate_cases`'s core loop.

### 3. Per-collection auto-optimization config + trigger (`veda-gateway`)

- New lightweight store `_auto_optimization.json` (mirrors the `_slo_state.json`/
  `_synthetic_state.json` write-through pattern — NOT a `CollectionSchema` field, so this
  stays orthogonal to the vector/field schema and needs no schema-version bump):
  `AutoOptimizationConfig { collection: String, enabled: bool, target_questions:
  Option<usize>, mode: TuningMode }` (`target_questions: None` → use the recommendation
  formula each run; `Some(n)` → an explicit Niriksha-set override, still capped at the
  live chunk count at trigger time). `mode` defaults to `Quality` (see §5) since that's the
  more complete metric; the campaign engine's existing mode is what runs unless a user has
  explicitly asked for `Performance`.
- REST: `GET/PUT /v1/collections/{name}/auto-optimization` (mirrors the existing collection-
  scoped route pattern, not admin-only — matches `/reindex`'s access level).
- Trigger point: `ingest_common::ingest_markdown` (the shared tail for both
  `process-document(s)` and `ingest-url`) spawns a **detached background task**
  (`tokio::spawn`, matching the incident-correlator/recall-probe off-lock pattern — never
  blocks the ingest HTTP response) after its existing `rebuild_executors` call, IF
  `auto_optimization.enabled && answer_gen_config.is_some()`. The task: (a) computes the
  recommended question count from the current (post-ingest) corpus, (b) calls
  `generate_eval_set` fresh (replacing any prior auto-generated set for this collection —
  tracked via a stable, deterministic id like `auto-eval-{collection}` so regeneration
  overwrites rather than accumulates orphaned sets), (c) calls `run_and_persist_campaign`
  against it in the configured `mode`. All best-effort: any failure (LLM down, generation
  produced 0 cases) is logged via `tracing::warn!` and the task exits — it must never affect
  the ingest response or crash the server.

### 4. Niriksha: testset-size control + auto-optimization toggle

- New backend endpoint `GET /v1/collections/{name}/eval-set-size-recommendation` →
  `{ recommended: usize, max: usize, chunk_count: usize, document_count: usize }`, computed
  with the same per-document weighting as §2 (`recommended` = `2 × document_count` rounded
  per the user's "10 docs → 20 questions" example, length-adjusted the same way generation
  itself is; `max` = `chunk_count`, a hard ceiling enforced both client- and server-side).
- Niriksha's "Create/Generate eval set" flow (`CreateEvalSet` in `QualityLab.tsx`) gains a
  number input pre-filled with `recommended`, a caption ("20 recommended based on this
  index's size"), and a `max` bound wired from the same response — the existing manual
  eval-set creation flow is untouched; this only affects the "generate from documents" path.
- A small "Auto-optimize" toggle (calling the new `GET/PUT .../auto-optimization` endpoint)
  lives on the collection's settings surface, with a `mode` radio (Quality/Performance) —
  reachable from the same place, off by default.

### 5. Two tuning modes (`veda-gateway` tuning engine)

- `StartCampaignRequest` gains `mode: TuningMode` (`Quality` default, `Performance`
  variant), threaded into `run_and_persist_campaign`/`run_campaign`.
- **Quality mode** = today's behavior, unchanged: NSGA-II multi-objective search over
  (recall, faithfulness, latency), reporting the full Pareto front.
- **Performance mode** reuses the identical sampling/evaluation loop (same trials, same
  objectives computed) — only the **selection/reporting** step changes: compute a
  `recall_floor` once per campaign (the recall the live collection's own current tier1-
  default config already achieves against the eval set — never a fixed magic number, so
  switching modes can't regress below today's baseline), filter trials to
  `recall >= recall_floor`, and report the single lowest-`latency_p50_ms` survivor as
  the campaign's answer (still stored in the same `pareto_front`/`trials` shape for API
  compatibility — `pareto_front` just contains one id in this mode instead of many).

## Testing (real-data verification)

Once built: sync to iitgpu02, restart the live server with the real `ANSWER_GEN_*`
credentials, and re-run against the actual corpus already ingested there
(`/home/mac/25gitlab/rag_anything/bulk_module/tmp` → `rag_test`, 184 chunks / 18 docs) —
per the user's explicit instruction to validate end-to-end on this same real data. Confirm:
generated questions are grounded and diverse (spot-check across ≥3 distinct source
documents and at least one table/image-tagged chunk if present in this corpus); easy/
medium/hard counts are within 1 of equal thirds; enabling auto-optimization on `rag_test`
and re-ingesting one more real document triggers a fresh generation + campaign without
blocking the ingest call; Niriksha's size control shows a sane recommendation and enforces
the chunk-count cap; a Performance-mode campaign's single winner has `recall >=` the
Quality-mode baseline and the lowest latency among qualifying trials.

## Out of scope (this iteration)

Configurable semaphore limit (fixed at 5); incremental/append-only testset regeneration
(always full regenerate, per decision above); auto-trigger on non-document ingestion paths
(`/insert`, `/embed-insert`); debouncing rapid repeated ingestions (each ingestion call
triggers independently, per the user's explicit "re-trigger every time it grows").

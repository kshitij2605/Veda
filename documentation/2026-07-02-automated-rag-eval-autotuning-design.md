# Automated RAG Evaluation & Auto-Tuning — Design (Phase 1: text-only MVP)

## Context

Veda has all the *scoring* primitives for RAG quality (recall@k, faithfulness, answer-relevancy,
chunker preview, per-query adaptive routing) but nothing automating them: nothing reads a user's
own stored corpus and generates a labeled test set, and nothing sweeps pipeline configuration to
find what actually works best for that corpus. Every step today requires a human to pick a config
and click "run" (`crates/veda-gateway/src/rest/rag_eval.rs`, `niriksha/src/pages/quality/`).

A deep-research pass (106 agents, 4M tokens, 24 primary sources — arXiv + GitHub, adversarially
verified) grounded this design in the current state of the art:

- **Synthetic test-set generation** is a solved, well-documented pattern (RAGAS, ARES, DeepEval,
  DataMorgana): chunk/cluster → LLM-generate grounded QA → quality-filter → diversify. No tool
  surveyed does this for images/audio/video — that gap is real, unbuilt, and **out of scope for
  this cycle** (see Deferred Scope).
- **Per-component metrics**: RAGChecker (NeurIPS 2024, peer-reviewed) separates retriever- and
  generator-level diagnostics and correlates with human judgment far better than RAGAS or n-gram
  metrics (Pearson 61.93 vs. 48.31 vs. 31.75/38.89, human ceiling ~63–71). BEIR/MTEB confirm no
  universal winning retrieval model/metric — corpus-specific tuning is genuinely necessary, not
  cosmetic.
- **Avoiding local optima**: joint multi-objective search (Bayesian optimization, qLogNEHVI
  acquisition) beats greedy per-stage tuning in peer-reviewed evaluation (arXiv 2502.18635). The
  two most-adopted *actual* OSS tools people use today (AutoRAG, 4.9k★; RagBuilder, 1.5k★) both
  *market* themselves as "AutoML" but implement **per-stage/per-node optimization under the hood**
  — genuine joint optimization is a real, verified gap in the current tooling landscape.
- **Domain-conditioned metrics** (deferred to a later cycle): legal/medical/finance/technical
  domains each need a distinct supplementary metric, not just reweighted defaults (ClaimRAG-LAW,
  RAG-X, FinReflectKG/HalluBench, arXiv 2407.12873) — noted here for future design, not built now.

Full findings: see the research transcript referenced in this conversation (not persisted to a
separate doc — the confirmed/refuted claims and source list are reproduced in the summary above).

## Scope for this cycle

**In scope**: synthetic **text-only** QA generation from a collection's chunks; RAGChecker-style
per-component metrics; a bounded, multi-objective auto-tuning campaign over text-corpus pipeline
parameters; a Niriksha UI surface.

**Explicitly deferred** (separate future design cycles):
- Multimodal (image/audio/video) synthetic QA generation — requires new vision-LLM wiring in
  `LlmClient` (confirmed absent today: no `image_url`/vision content type in `llm_client.rs`).
  Also cross-modal alignment metrics (CLIPScore-style for image-text; no verified CLAP-style
  audio/video-text metric exists in the literature surveyed — Veda would need to build one).
- Domain-aware metric selection (a domain classifier/tag + pluggable domain-specific checks).

## Architecture — seven phases, each independently shippable

| Phase | Delivers |
|---|---|
| **0 — Alias resolution fix** | Prerequisite. `/reindex`'s alias-swap (`crates/veda-gateway/src/rest/collections.rs::reindex_collection`) is a **documented no-op in non-Raft/single-node mode** (`set_alias`: *"Non-Raft mode: there's no SM to write to... alias resolution in non-Raft mode is a follow-up"*). Add a persisted, non-Raft alias map on `AppState` that every collection-name lookup (search/records/insert/etc.) resolves through, so `set_alias`/`get_alias` genuinely redirect traffic without Raft. Without this, Phase 5's "safe apply" silently does nothing for most self-hosted (single-node) deployments. |
| **1 — Synthetic text QA generation** | `POST /v1/collections/{name}/eval-sets/generate`: chunk-cluster the collection → LLM generates grounded QA pairs → quality-filter (numeric gate, bounded retries) → diversify (evolution-operator categories: simple/reasoning/multi_context, mirroring DeepEval/RAGAS) → persists as an `EvalSet`/`EvalCase` via the **existing** `rag_eval.rs` model (additive fields only, see Data Model). |
| **2 — Richer per-component metrics** | Extend `CaseResult` with RAGChecker-style separated retriever/generator diagnostics, additive to the existing recall/faithfulness/relevancy fields. |
| **3 — Search engine integration** | Pull in `scirs2-optimize` (real crate: 33k downloads, Apache-2.0, confirmed via docs.rs to expose `multi_objective::NSGAII`/`NSGAIII` and `bayesian::BayesianOptimizer`) behind a small internal Rust trait — never called directly elsewhere, so the backend is swappable. A verification spike (toy 3-objective problem with a known Pareto front) must pass before Phase 4 builds on it. Defines the 3-tier parameter search space (below). |
| **4 — Campaign engine** | The bounded, nested multi-objective search loop (mechanics below). Produces a `TuningCampaign` with a global Pareto front of `Trial`s. |
| **5 — Safe apply** | Recommend-only. Tier 1 applies instantly and reversibly (query-time defaults only). Tier 2/3 applies via the existing `/reindex` → new sibling collection (never overwrites) → alias-swap (real, thanks to Phase 0). Revert = swap the alias back; the prior sibling is never deleted by this feature. |
| **6 — Niriksha UI** | A "Campaigns" sub-view inside the existing Quality Lab page. |

**Sequencing**: 0 is a hard prerequisite for 5 (apply needs working aliases in single-node mode).
1→2 share `rag_eval.rs` (edit in series). 3→4 (engine needs the search backend). 4→5 (apply needs
a result to apply). 6 last, wraps everything. 0, 1+2, and 3 are otherwise independent and can be
built in parallel; 4, 5, 6 are strictly sequential after their dependencies land.

## Data model

- **`EvalCase`** (`crates/veda-gateway/src/rest/rag_eval.rs`) gains two optional, `#[serde(default)]`
  fields — fully backward-compatible with existing persisted cases:
  - `source: Option<CaseSource>` — `{ generated_by: String, source_chunk_ids: Vec<String>, generated_at: i64 }`,
    provenance for synthetically-generated cases (audit trail — was this case human-authored,
    graduated from a conversation, or synthetically generated, and from what).
  - `difficulty: Option<String>` — `"simple" | "reasoning" | "multi_context"`, the diversity
    category a synthetic case was generated under.
- **`CaseResult`** gains additive fields for Phase 2's per-component diagnostics: `claim_recall`,
  `context_precision`, `context_utilization`, `noise_sensitivity`, `hallucination`,
  `self_knowledge` (all `Option<f64>`, `None` when not scored — mirrors the existing
  `recall_at_k`/`faithfulness` optionality pattern already in this struct).
- **Job execution**: reuses the **existing** `scheduler.rs` `JobRegistry`/`JobKind`/
  `/v1/admin/jobs` infrastructure rather than building new job-tracking:
  - New `JobKind::GenerateEvalSet { collection, request }`.
  - New `JobKind::TuningCampaign { collection, eval_set_id, config }`.
  - **`JobKind::Evaluate` gets wired up** — it is currently dead code
    (`scheduler.rs:237`, logs `"scheduler: would run evaluate — TODO: wire to evaluate handler"`
    and does nothing else). This is a pre-existing bug fixed as a natural part of this work, not
    new scope.
- **`TuningCampaign`** / **`Trial`** / **`PipelineConfig`** (new module, `rag_eval.rs` or a
  sibling `tuning.rs`): persisted as `SavedObject`s (`type = "tuning_campaign"`) — the same
  persistence pattern as `EvalRun`, no new storage subsystem.
  ```
  TuningCampaign { id, collection, eval_set_id, status, search_space, budget,
                   trials: Vec<Trial>, pareto_front: Vec<String> (trial ids),
                   created_at, completed_at }
  Trial { id, config: PipelineConfig, objectives: { recall, faithfulness, latency_p50_ms },
          is_pareto_optimal, started_at, duration_ms, sibling_collection: Option<String> }
  PipelineConfig {
    // Tier 1 (cheap — query-time only)
    ef_search, fusion: { mode: Rrf|Weighted, dense_weight, sparse_weight }, rerank, top_k, truncate_to,
    // Tier 2 (medium — reindex, no re-embed)
    index_type, ef_construction, m,
    // Tier 3 (expensive — re-chunk + re-embed)
    chunker: { strategy, chunk_size, overlap }, embedding_model,
  }
  ```

## Campaign engine mechanics

Naive NSGA-II over the full flat parameter vector would draw a random tier-1+2+3 combination for
every individual in every generation — a full re-chunk+re-embed for nearly every trial. Instead, a
**nested, cost-aware search**:

1. **Outer sampling (tier 2+3, expensive)**: Latin Hypercube Sampling picks a small, well-spread
   set of distinct "base configurations" (e.g. 5–8) across chunker strategy × chunk size ×
   embedding model × index type. Each base gets **one** `/reindex` call to materialize a sibling
   collection — tier-3 bases pass a `chunker` config (re-chunk, optionally re-embed); tier-2-only
   bases pass no chunker (straight copy, cheaper — only the index-type-related schema fields
   differ, so no re-chunk/re-embed).
2. **Inner search (tier 1, cheap)**: *within* each materialized base, `scirs2-optimize`'s
   NSGA-II/III runs over just the tier-1 params — many generations, all query-time, no rebuild.
3. **Global Pareto front**: every trial (across all bases) carries its full combined config and
   objective vector `(recall, faithfulness, latency_p50_ms)`; the front is computed over *all*
   trials globally — a genuinely joint optimum across the whole space, despite staged execution.

Convergence: stop at `budget.max_trials` or `budget.max_duration_secs`, whichever first, or early
if the front hasn't improved for K consecutive trials.

Tier 2 and tier 3 trials both go through the **same** `/reindex` primitive
(`reindex_collection`), differing only in request shape (chunker present or absent) — no new
reindex machinery, only this campaign layer orchestrating calls to the existing one.

## Safe apply

Recommend-only — a campaign report never auto-applies. Applying a chosen Pareto-front `Trial`:
- **Tier 1 only** (config differs from the live default only in tier-1 fields): applied instantly,
  reversibly — just updates the collection's default query params, no data touched.
- **Tier 2/3 involved**: triggers `/reindex` (already produces a **new sibling**, never overwrites
  — pre-existing, unmodified behavior) and a real alias-swap (Phase 0). **Revert** = swap the alias
  back to the previous version name; the prior sibling is never deleted by this feature.

## Niriksha UI

A "Campaigns" sub-view inside the existing Quality Lab page (`niriksha/src/pages/quality/`),
reusing its `SavedObject`-list + detail-drawer patterns:
- Campaign list with status (Running/Converged/BudgetExhausted/Failed); "New campaign" (pick or
  generate an eval set → set budget → submit).
- Live progress while running (trial count, best-so-far objectives — polled).
- Completed view: a sortable/filterable table of Pareto-optimal trials (recall/faithfulness/latency
  columns) plus a simple 2-axis scatter (recall vs. latency; faithfulness as point size/color). A
  full 3D Pareto visualization is out of scope for MVP.
- **Apply** button per trial → a confirmation dialog that clearly distinguishes "instant, reversible"
  (tier 1 only) from "triggers a reindex" (tier 2/3 involved) before executing.

## Testing strategy

Each phase TDD'd against fixtures/fakes, matching existing conventions (the fake-LLM injection
trait already used by agent tests, fake embedder, small in-memory collections):

- **Phase 0**: non-Raft alias map get/set/resolve unit tests; a `/reindex` followed by `/search` in
  single-node mode actually hits the new sibling (integration test); revert restores the old one.
- **Phase 1**: fake-LLM generates cases from fixture chunks; a mocked low-quality case is filtered
  out; category diversity is produced across a batch; persists as a valid `EvalSet`.
- **Phase 2**: each new metric scored against fixture retrieval/generation pairs with known-correct
  expected values.
- **Phase 3**: the verification spike *is* the test — a toy 3-objective synthetic function with a
  known Pareto front; confirm `scirs2-optimize` converges near it. Unit test for the trait wrapper.
- **Phase 4**: a tiny fixture collection + fake LLM/embedder; run a small bounded campaign; assert
  a valid Pareto front; **critically**, assert `/reindex` is called only once per distinct
  tier-2/3 base (not once per tier-1 trial) — the core cost-correctness property.
- **Phase 5**: apply a tier-1-only config (defaults change, reversible); apply a tier-2/3 config
  (new sibling created, alias swapped, old collection untouched and still independently queryable,
  revert works).
- **Phase 6**: component tests + a clean `npm run build`.
- **Live e2e** (mirrors every other major cycle's verification section in this project): a real
  seeded collection, real LLM + embedder, a campaign converges, applying a result measurably
  changes search behavior, revert restores it.

## Risks / gotchas

1. **`scirs2-optimize` is young** (created April 2025) and broad-scope (part of a larger "SciRS2"
   umbrella) — the verification spike (Phase 3) is not optional; do not build Phase 4 on an
   unverified assumption that its NSGA-II/III actually converges correctly.
2. **Embedding/re-chunk cost is real** — the nested search design exists specifically to bound it;
   any future change to the search strategy must preserve the "expensive params sampled sparingly"
   property, or campaigns become prohibitively slow/costly on large corpora.
3. **Phase 0 touches every collection-name lookup path** (search/records/insert/etc.) — this is a
   correctness-critical, wide-blast-radius change; needs careful review and cannot be rushed even
   though it's "just" a prerequisite.
4. **Job-based execution** (`JobKind::GenerateEvalSet`/`TuningCampaign`) must not block the
   scheduler's other ticks (TtlSweep, etc.) — run off the main tick loop, matching the existing
   `spawn_recall_prober`/`spawn_incident_correlator` off-tick pattern.
5. **No multimodal, no domain-awareness in this cycle** — both are real, verified gaps in the
   literature (not just deferred for time) and warrant their own dedicated design cycles once this
   MVP is proven.

## Verification (definition of done)

Phase 0–6 each pass their TDD suite (`cargo test -p veda-gateway`, targeted `--test` where
applicable) + `cargo build --workspace` + niriksha `npm run build`, clippy/fmt clean on the pinned
1.96.0 toolchain. Live e2e (single-flight, isolated data dir): generate a synthetic eval set from a
real seeded collection, run a bounded campaign, confirm a valid Pareto front, apply a tier-1 config
(instant), apply a tier-2/3 config (reindex + alias-swap), confirm search behavior changed, revert,
confirm it's restored.

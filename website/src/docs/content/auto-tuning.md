# RAG Auto-Tuning

Because Veda owns the whole RAG stack — the corpus, the chunker, the embedder, the retriever, the reranker, and the LLM judge — it can do two things end-to-end that a bring-your-own-pipeline setup can't: **generate a labeled test set from your real data**, and **search its own configuration space for the best pipeline** — instead of asking you to hand-pick a chunk size and hope.

Every step reuses an existing primitive — the embedder, the LLM client, [`/reindex`](/docs/collections#atomic-create-and-load), saved-object persistence — rather than a bespoke subsystem.

![A real run: Veda generates a test set from seven documents, searches its configuration space over eight trials, and the winning trade-off is applied in one click.](/videos/auto-tuning.mp4)

## Generate a test set from your corpus

`POST /v1/collections/{name}/eval-sets/generate` (`{ name?, num_cases }`) turns your ingested documents into a labeled [eval set](/docs/rag-aiops#eval-harness-the-production-trace-flywheel) of grounded `{ query, expected_chunk_ids, gold_answer, difficulty }` cases — no hand-labeling.

The sampler is deliberately **diverse and weighted** so the test set represents the whole corpus, not just its largest text-only documents:

- **Document-weighted** — chunks are grouped by parent document and apportioned by text length, with a fairness pass so every real document gets at least one question when the budget allows it.
- **Element-type aware** — each chunk is tagged `Table`, `Image`, or `Text`, and questions round-robin across types within a document, so a report with prose *and* embedded tables gets asked about both.
- **Difficulty-tiered** — cases are labeled `easy` / `medium` / `hard` in roughly equal thirds.

`GET /v1/collections/{name}/eval-set-size-recommendation` returns a corpus-aware `{ recommended, max, chunk_count, document_count }` (recommended ≈ `document_count × 2`, capped at the chunk count) so you never ask for more questions than the corpus can ground.

## Tuning campaigns

`POST /v1/collections/{name}/tuning-campaigns` (`{ eval_set_id, max_trials?, num_bases?, mode? }`) searches the pipeline's configuration space against an eval set and returns the winning trade-offs. Each trial carries an objective vector `{ recall, faithfulness, latency_p50_ms }`.

The search is **nested and cost-aware**, so it doesn't re-chunk and re-embed on every trial:

| Tier | Parameters | Cost |
|---|---|---|
| **Tier 2/3** (outer) | chunker strategy, embedding model, index type | one `/reindex` per base — sampled sparingly via Latin Hypercube Sampling (`num_bases`, default 3–5) |
| **Tier 1** (inner) | `ef_search`, fusion mode/weights, rerank, `top_k` | query-time only — NSGA-II runs many generations per base with no rebuild |

The global Pareto front is computed across *every* trial from *every* base — a genuinely joint optimum across the whole space, despite the staged execution.

### Two modes

- **Quality** *(default)* — returns the full Pareto front: multiple non-dominated recall / faithfulness / latency trade-offs to choose from.
- **Performance** — returns a single winner: the lowest-latency trial whose recall stays above a **recall floor**, where the floor comes from one explicit, guaranteed evaluation of the default config (not a lucky organic trial).

## Auto-optimize on ingest

`GET/PUT /v1/collections/{name}/auto-optimization` (`{ enabled, target_questions?, mode }`) opts a collection into **continuous** tuning. When enabled and a dedicated answer-gen LLM is configured, every finished ingestion (`process-document` or `ingest-url`) fires — **fire-and-forget, never blocking the ingest response** — a fresh eval-set generation followed by a tuning campaign, so the recommendation always reflects the corpus's *current* state.

The auto-generated eval set uses a deterministic id (`auto-eval-<collection>`), so re-triggering overwrites it with a fresh set for the now-larger corpus rather than accumulating orphaned sets. It's off by default — opt-in per collection, and only active at all when the answer-gen endpoint is configured.

## Safe apply

A campaign is **recommend-only** — it never auto-applies. `POST /v1/tuning-campaigns/{campaign_id}/trials/{trial_id}/apply`:

- **Tier-1-only change** (`ef_search` / fusion / rerank / `top_k`): applied **instantly and reversibly** — it just updates the collection's default query parameters; no data is touched.
- **Tier-2/3 change**: runs `/reindex`, which produces a **new sibling** collection (`name@vN`) and swaps the alias — the original is never overwritten. Revert is a swap back to the previous version.

## The dedicated answer-gen LLM

Eval-set generation and campaign scoring use a **separate** LLM connection (`ANSWER_GEN_BASE_URL` / `ANSWER_GEN_API_KEY` / `ANSWER_GEN_MODEL` / `ANSWER_GEN_TIMEOUT_SECS`) from the agent-facing `VEDA_LLM_*` config — a deliberate split so tuning work never competes with your chat/agent traffic on a shared production endpoint. Calls are bounded by a semaphore (capacity 5) against that endpoint. If `ANSWER_GEN_BASE_URL` is unset, the whole auto-tuning path is a clean no-op — it's fully opt-in at the infrastructure level.

## In Niriksha

The **Quality Lab** surfaces all of this: an **Eval Harness** with a "Generate from documents" modal (live size recommendation, and a checkbox to auto-regenerate + re-tune whenever the collection grows), and a **Campaigns** view with a per-collection campaign list, a mode selector, and a `CampaignDetail` Pareto-front table (recall / faithfulness / p50 latency / Apply).

## Validated on real data

Against a real ingested corpus (7 OCR'd documents, ~190 chunks, real BGE-M3 embeddings and a real LLM judge), a fresh ingestion auto-generated a 14-case eval set spanning all 7 documents (including an image-sourced question) and ran a Quality campaign to a genuine multi-point Pareto front. A Performance-mode campaign then found a trial with **identical recall and faithfulness at 2.27× lower latency** than the default config — the recall floor holding exactly as designed while surfacing a real latency win. These are heuristic searches (`num_bases` / `max_trials` bound the effort), not a provable global optimum — but they find real, materially different, defensible configurations on real data.

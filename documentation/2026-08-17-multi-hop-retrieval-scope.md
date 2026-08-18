# Multi-hop retrieval — scoping discussion

**Date:** 2026-08-17
**Status:** scoping only. No implementation is proposed here, and none should start from this
document alone.
**Source:** gap 5 of 5 from `research/2026-08-12-competitive-refresh/`, whose own
recommendation for this item is *"Scope multi-hop retrieval — design discussion, not
implementation"* (`06-community-demand.md`).

---

## 1. What the evidence actually says, and how strong it is

The refresh ranks multi-hop **9th** in community demand and grades its evidence **B/C** — the
weakest of the five gaps acted on in this cycle. The other four were graded A. That ranking is
the reason this is last, and it should stay in view.

The headline figure is:

> "Reported NIST analysis (Jul 2026): **73% of enterprise RAG deployments fail informal accuracy
> audits on multi-hop reasoning**; multi-hop showed a 62% discrepancy vs vendor claims."

Two hedges are load-bearing and are in the source itself: **"Reported"**, and **"informal
accuracy audits"**. We do not have the primary document. Before any engineering is committed,
someone should obtain the actual NIST analysis and confirm both numbers and methodology — a 73%
failure rate against *informal* audits is not the same claim as a 73% failure rate against a
benchmark, and the difference decides whether this is a product gap or a marketing artifact.

**Recommendation:** treat §1 as an open verification task, not as an established premise.

## 2. What "multi-hop" means, concretely

A query whose answer requires **combining facts that do not co-occur in any single chunk**:

- *"Which of our suppliers are in countries where the 2026 tariff schedule changed?"* — needs the
  supplier list from one document and the tariff schedule from another, joined on country.
- *"Did the vendor who signed the Q1 MSA also appear in the Q3 incident report?"* — two documents,
  joined on an entity that neither states as the answer.

Single-shot dense retrieval fails these structurally, not through poor tuning: the query embedding
is a single point, and no chunk is close to it, because no chunk contains the answer. Retrieving
top-k harder does not help; the required inference lives *between* chunks.

This is worth stating precisely because it explains why the fix cannot be "improve recall". Every
retrieval improvement this cycle shipped — Block-Max WAND, CAGRA, guardrails, read-in-place
Parquet — makes single-shot retrieval better and moves this need not at all.

## 3. What Veda already has

Verified against the code, not against the research's summary:

| Piece | Where | What it gives us |
|---|---|---|
| ReAct-style agent loop, bounded by `max_steps` | `crates/veda-gateway/src/agent/executor.rs` | A model can already issue several retrievals in sequence and condition each on the last |
| Retrieval tools: `search`, `filter_query`, `get_record`, `aggregate` | `crates/veda-gateway/src/agent/catalog.rs` | The primitives a decomposition strategy would call |
| `ChunkingStrategy::Hierarchical` — emits **both** coarse doc-section and fine chunks | `crates/veda-core/src/chunking.rs` | Coarse-to-fine narrowing, which is one of the two things TreeRAG provides |
| Eval harness: `/v1/eval-sets`, `/run`, `/generate`, faithfulness scoring | `crates/veda-gateway/src/rest/` | **We can measure multi-hop accuracy before choosing an approach** |
| Retrieval guardrails, citations, grounding | shipped v0.6.0 | Per-hop provenance already exists |

The honest reading: **Veda can already do multi-hop badly, via the agent loop.** What is missing
is an explicit strategy — nothing plans the decomposition, nothing tracks what has been resolved
across hops, and nothing measures whether the loop actually terminated with the facts it needed.

This matters for sequencing: the cheapest next step is not building a retriever, it is
**measuring the loop we already have**.

## 4. Three candidate approaches

### A. Agentic query decomposition (iterative retrieve-then-reason)

Decompose the query into sub-questions, retrieve per sub-question, feed resolved facts forward,
stop when the model reports sufficiency or `max_steps` trips.

- **Fits** the existing agent loop almost exactly; largest reuse of shipped surface.
- **Cost:** a planner prompt, a resolved-fact accumulator, per-hop provenance, a termination rule.
- **Risks:** latency multiplies by hop count; each hop is an LLM call, so cost scales with it;
  error compounds — a wrong hop 1 poisons everything after it; termination is genuinely hard and
  is where these systems usually fail in practice.
- **No index or storage change.**

### B. Knowledge-graph hybrid (GraphRAG-style)

Extract entities and relations at ingest, store a graph, traverse it to assemble multi-hop
context.

- **Strongest** on the entity-join queries in §2, and the traversal is deterministic and
  explainable rather than model-mediated.
- **Cost:** by far the largest. Entity/relation extraction on every ingest (another model in the
  pipeline), a graph store Veda does not have, graph maintenance under updates and deletes, and a
  new query surface. This is a subsystem, not a feature.
- **Risks:** extraction quality caps the whole thing; multimodal corpora (Veda's differentiator)
  have no good entity extractor for images/audio; graph staleness on update is a hard problem in
  its own right.

### C. TreeRAG / hierarchical summarisation

Summarise upward into a tree, retrieve at the coarse level, descend.

- **Partially shipped already** — `ChunkingStrategy::Hierarchical` emits the coarse/fine levels.
  The missing part is summarisation of parents and descent-based retrieval.
- **Cost:** moderate. Summarisation at ingest, a descent retrieval path, tree maintenance.
- **Risks:** helps *aggregative* questions ("what are the themes across these docs") far more than
  the *entity-join* questions in §2. It may not answer the demand it is being proposed for, which
  is the trap worth naming.

## 5. What I recommend, and what I do not

**Do first, and it is not engineering:**

1. **Get the primary NIST document** and verify the 73%/62% figures and their methodology (§1).
2. **Build a multi-hop eval set and measure the agent loop we already have.** The harness exists;
   `/v1/collections/{name}/eval-sets/generate` plus hand-written multi-hop cases would give a
   number. Without it, any approach in §4 is chosen blind and its improvement is unmeasurable.

That second item is the highest-value work in this document and the cheapest. It is also the
thing that decides between A, B and C — if the bounded agent loop already scores acceptably, this
gap is a *positioning* problem rather than an engineering one, which is exactly what the refresh
concluded about most of the community-demand list.

**If engineering is then warranted, A before C before B.** A reuses the most, measures the
fastest, and is reversible. B is a subsystem whose extraction step conflicts with Veda's
multimodal positioning and should not be entered without the numbers from step 2.

**Do not** start with B because it is the most impressive, and do not start with C because it is
closest to something already shipped — C most likely does not answer the demand in §2.

## 6. Explicitly out of scope here

No API surface, no schema, no index changes, no task breakdown. This document exists so that the
decision is informed; the refresh asked for a design discussion and this is that discussion, not
a plan. A plan should follow only after step 2 of §5 produces a number.

## 7. Open questions for the decision

1. Do we accept the 73% figure without the primary source? (I would not.)
2. Is the target the entity-join query in §2, the aggregative query, or both? The answer changes
   which of A/B/C is even relevant, and the three are not interchangeable.
3. What latency and cost budget is acceptable? A 4-hop query is 4 LLM calls; if the product's
   latency budget cannot absorb that, approach A is out before it is designed.
4. Does this need to work on multimodal corpora? If yes, B weakens sharply — entity extraction
   over images and audio is not a solved problem, and multimodality is Veda's differentiator.

---

# Measured: the existing agent loop on multi-hop questions

**Date:** 2026-08-18. **Supersedes the "do first" list in §5** — step 2 of it is now done.

§5 argued that the cheapest useful step was to measure the bounded ReAct loop that already
exists, because that decides whether any of approaches A/B/C is warranted. That measurement has
now been run against live infrastructure: `Qwen/Qwen3.5-122B-A10B` on vLLM, the
`jina-v5-omni-small` embedder, and a Veda server on an isolated port.

Harness and raw results are committed beside this document
(`multihop-eval/multihop_probe.py`, `multihop-eval/results-2026-08-18.json`).

## Design of the measurement

A 15-document corpus in which **no single document answers any multi-hop question**: each answer
needs a fact from one document joined to a fact in another (supplier -> country -> tariff change;
contract -> signatory -> incident report; product -> plant -> certification). Three distractor
documents share the vocabulary without carrying an answer.

Every multi-hop case is paired with **single-hop controls** whose answers sit in one document.
Without those, a low multi-hop score would be ambiguous between "cannot retrieve" and "cannot
combine". The controls isolate the second.

## Result

| | correct | mean latency | mean steps |
|---|---|---|---|
| single-hop control | **4/4 (100%)** | 1.8 s | 3.0 |
| multi-hop | **5/6 (83%)** | 2.4 s | 4.0 |

**The existing loop already performs 2-hop entity-join retrieval.** Controls at 100% confirm the
multi-hop figure measures combining rather than finding.

### The one failure is the informative part

`mh-1` asked which suppliers are headquartered in countries whose 2026 tariff rates changed
(Estonia and Chile -> Nordwind, Braewick, Meridian). It answered "Nordwind Components, Meridian
Alloys" — one supplier per country, **missing the second Estonian supplier**.

That is incomplete enumeration on a **one-to-many join**: the loop finds a path and stops instead
of exhausting the join. It is not a retrieval failure (the control proves the fact is findable)
and not a reasoning failure (the two it found are correct).

## What this does not support

- **`mh-1` is nondeterministic.** An earlier run of the same case returned all three names. The
  honest figure is 83–100% across runs, and with six cases run-to-run variance swamps the
  difference. This is directional, not conclusive.
- **The corpus is 15 documents.** Close to a best case: in a real corpus a wrong first hop is far
  likelier, and the distractor set here is three documents.
- **Only 2-hop, only entity joins.** No three-hop chains, no aggregative questions, no negative
  cases where no answer exists (which is where a decomposition planner usually fails, by
  confabulating a join).

## Revised recommendation

**The measurement argues against approaches B and C.** A knowledge graph (B) is a subsystem whose
ingest-time entity extraction conflicts with Veda's multimodal positioning, and TreeRAG (C) targets
aggregative questions rather than the entity joins that the loop already handles. Neither is
justified by a loop scoring 83–100% on the cases they would address.

It points instead at approach **A**, and at the narrowest form of it: **completeness on
one-to-many joins**, which is the single observed failure. That is a prompt-and-termination
concern inside the existing loop, not new retrieval machinery.

**The blocker is eval capacity, not implementation.** Six cases cannot validate an improvement —
a fix and LLM variance are indistinguishable at this sample size. Before any code changes:

1. Expand to a few dozen cases spanning join shapes: one-to-many, three-hop chains, aggregative,
   and **negative cases where no answer exists**.
2. Re-measure to establish a baseline with usable error bars, running each case several times so
   variance is quantified rather than assumed.
3. Only then change the loop, and only where the expanded baseline shows a deficit.

Doing it in the other order means shipping a change that cannot be shown to have helped — which is
the failure this project has already paid for repeatedly this cycle.

## Note on the harness

The grader initially used substring matching, under which a `"no"` answer matched `"not"`,
`"none"`, `"cannot"` and even `"Nordwind"`. `mh-1` scored 6/6 under it. It now matches on word
boundaries, and that is what moved the multi-hop figure from 6/6 to 5/6 — the stricter grader is
the reason the number is 83% and not 100%. Any future expansion of this eval set should keep the
boundary matching, and should assume a grader is wrong until it has been shown to reject a
plausible wrong answer.

---

# Re-measured on a corpus that actually forces hops (v2)

**Date:** 2026-08-18. **This supersedes the v1 result above.**

## Why v1 did not measure multi-hop

v1 used 15 documents. Dumping the loop's actual tool calls showed a single search with
`limit=20` returning the whole corpus: in the two failing runs the model had *every* fact in
context and still omitted one supplier; in the passing run it happened to issue three searches.

So v1 measured how reliably the LLM scans a context window it already holds — not retrieval. The
"83% multi-hop" figure and the one-to-many gap drawn from it do not describe multi-hop behaviour.

v2 buries the same 12 fact documents in **2,000 filler documents** sharing the business
vocabulary, so no top-k search can hold both halves of an answer. It adds one-to-many and
**negative** cases (correct answer "none"/"unknown"), and runs each case 3x because v1 showed the
loop is nondeterministic.

## Result (2,012 documents, 3 repeats per case)

| case type | correct | mean searches |
|---|---|---|
| single-hop control | 9/9 (100%) | 1.0 |
| one-to-one join | 12/12 (100%) | 1.9 |
| one-to-many join | **6/6 (100%)** | 2.0 |
| negative (no answer exists) | **3/6 (50%)** | 4.8 |

**The one-to-many gap was an artifact of the small corpus.** With hops actually forced it is 6/6.
Implementing the enumeration fix v1 recommended would have built machinery for a non-problem.

**The real gap is negative answers**, and it consumes the most searches. The failures are not
wrong facts; they are failures to *conclude*. `negative-1` returned "Based on the search results,
the suppliers mentioned and their headquarters are: ..." — reciting context instead of answering
"none", and listing suppliers that are not in Poland.

One nuance: `negative-2 r1` answered "The search results do not contain any information about who
signed the Q4 master service agreement" — semantically correct but not the literal word the
prompt demanded. That is instruction compliance, not reasoning, so the true reasoning failure is
2/6 rather than 3/6.

## The cheap fix was tested and does not work

Before proposing any loop change, the obvious candidate was that the probe's instructions never
said what to do when nothing is found. Two arms, same corpus and cases, 4 repeats:

| arm | negative-1 | negative-2 | one2many-1 | one2one-2 | total |
|---|---|---|---|---|---|
| A: baseline instructions | 2/4 | 4/4 | 3/4 | 4/4 | **13/16** |
| B: + "say so in the exact word rather than summarising" | 2/4 | 4/4 | 2/4 | 4/4 | **12/16** |

No measurable improvement (at n=4 per cell the difference is noise), and arm B used more searches.

**But it exposed the mechanism.** Two arm-B failures are new in kind: `negative-1 r2` made **14
searches** and returned "I reached the maximum number of reasoning steps before producing a final
answer"; `one2many-1 r2` made 8 and abandoned mid-reasoning. Pressing the model to try harder made
it search until `max_steps` and emit a non-answer.

Nothing in the loop notices that repeated searches are returning documents it has **already
seen**, so there is no signal to stop searching and answer. That is a loop-level deficiency, and
it is what the scoping section above predicted when it named termination as "genuinely hard and
where these systems usually fail in practice".

## Implementation this now supports

**Stagnation detection in the ReAct loop.** Track the record ids returned by retrieval tool calls
within a turn; when a call returns no ids that have not already been seen, tell the model that
further searching is returning nothing new, so it should answer from what it has — including
answering that the corpus does not contain the fact.

This is narrow (it touches the loop's tool-result handling, adds no retrieval machinery), it
targets the only measured deficit, and it is falsifiable against the baseline above: negative
cases 3/6 and mean 4.8 searches, with one-to-many at 6/6 as the regression guard, since arm B
showed that pushing on negatives can degrade positives.

It does **not** support building a knowledge graph or TreeRAG. Both target retrieval, and
retrieval measured 100% across control, one-to-one and one-to-many.

## Reproducing

`documentation/multihop-eval/multihop_v2.py` builds the corpus and runs the suite;
`neg_probe.py` runs the two-arm instruction test. Both need a Veda server with an embedder and
`VEDA_LLM_ENDPOINT` configured — they were run against Qwen3.5-122B on vLLM and
`jina-v5-omni-small`, with the server on an isolated port so a shared instance is untouched.

Note for whoever runs these: use `python3 -u`. Python block-buffers stdout to a file, so a long
run looks like it is producing nothing until it exits — which cost time here, and is the same
buffering trap already recorded for the Rust benchmark in the CAGRA design document.

---

# Stagnation detection: implemented, measured, and it does not work (v3)

**Date:** 2026-08-18. **This supersedes the "Implementation this now supports" section above.**

The v2 section proposed one narrow change and stated the baseline it had to beat. That change
was built, unit-tested, and measured against exactly that baseline. It does not move it.

## Result

Same 2,012-document corpus, same agent instructions, same grader, 3 repeats per case. Only
the server binary differs. The baseline column was measured twice, on separate runs, with
identical figures both times.

| case type | baseline (x2) | with stagnation detection |
|---|---|---|
| single-hop control | 9/9 (100%) | 9/9 (100%) |
| one-to-one join | 12/12 (100%) | 12/12 (100%) |
| one-to-many join | 6/6 (100%) | 6/6 (100%) |
| **negative** | **3/6 (50%), 4.8 searches** | **3/6 (50%), 4.7 searches** |

No improvement on the target, and no regression on the guard.

## Why: the trigger almost never fires

A null result has two causes that need different fixes - the signal reached the model and
changed nothing, or the signal was never sent. Replaying the loop's own bookkeeping over the
returned steps separates them. Six runs of the two negative cases:

| run | retrievals | times the note fired | answer |
|---|---|---|---|
| negative-1 r0 | 5 | **0** | recited context (fail) |
| negative-1 r1 | 8 | 2 | hit max_steps (fail) |
| negative-1 r2 | 7 | **0** | "none" (pass) |
| negative-2 r0 | 5 | **0** | "unknown" (pass) |
| negative-2 r1 | 6 | **0** | "unknown" (pass) |
| negative-2 r2 | 2 | **0** | "unknown" (pass) |

The note fired in **one run out of six**. The condition it was given - a retrieval call that
returns *no* record id the run has not already seen - is far too strict for a corpus of this
size. Every reworded query pulls back at least one document that has not appeared yet, even
when none of them are relevant. The loop is not re-reading the same documents; it is reading
new irrelevant ones.

So the v2 diagnosis was right that the loop cannot tell when searching has stopped paying,
and wrong about how to detect it. "No new documents" is not the signal. "No new *relevant*
documents" is, and that is a materially harder thing to compute - it needs a relevance
judgement, which is the retrieval problem again, not a bookkeeping problem.

## What this does and does not license

The code is correct, tested, and load-bearing (disabling the guard fails its test on the
intended assertion). It is also, by measurement, close to inert on real traffic. It should
not be merged as a fix for the negative-answer deficit, because it does not fix it.

Candidates a future attempt could measure, in rough order of cost:

- Fire on a *proportion* of new ids rather than requiring zero - e.g. under 20% new. Cheap,
  and directly testable against the table above.
- Fire on repeated near-identical queries rather than on returned ids.
- Score returned documents against the question and fire when the best score stops improving.
  This is the one that matches the actual failure, and the most expensive.

Whichever is tried, the firing-rate check above should be run first: a change that never
triggers cannot be evaluated by its win rate, and would otherwise have looked simply
ineffective.

## A note on what the failures now look like

The recited-context failure mode from v2 persists (`negative-1` twice returned "Based on the
search results, the suppliers mentioned and their headquarters are: ..."). The
instruction-compliance case also recurs: one `negative-2` run answered "The search results do
not contain any information about who signed the Q4 master service agreement" - correct in
substance, not the literal word the prompt demanded. As in v2, the true reasoning failure
count is below the raw score.

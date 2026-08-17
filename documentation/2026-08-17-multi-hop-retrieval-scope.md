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

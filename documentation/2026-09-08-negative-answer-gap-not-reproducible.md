# The negative-answer gap does not reproduce

**2026-09-08.** Task #72 asked which commit between 2026-08-18 and 2026-08-21
raised negative-answer accuracy from 3/6 to ~100%, after an A/B ruled out the
stagnation note and the embedding-strip commits were shown to be untestable on
that suite. The answer is that **no commit did, because under a working
embedder there is no gap to explain.**

## Measurement

Endpoints of the range, measured against an identical corpus — same 2,012
documents, same seed, same data directory, only the server binary differing:

| Commit | Date | Negatives | Controls |
|---|---|---|---|
| `7882b3c` "prune the Vamana graph over the visited set" | 2026-08-18 18:02 | 18/20, 19/20, 20/20 → **57/60 (95%)** | 18/18 |
| `356d6c9` "Merge docs/stagnation-ab-measurement" | 2026-08-21 11:31 | **20/20 (100%)** | 6/6 |

Fisher exact on 18/20 vs 20/20: **p = 0.487**. The before-endpoint reaches
20/20 unaided in one run of three. The historical claim was 50% vs 100%; what
is actually there is 95% vs 100%, which is nothing.

A bisect was set up and deliberately **not run**. Walking five steps across 31
commits to separate two arms that are statistically identical would have named
an arbitrary commit, and that commit would then have entered the changelog as a
cause.

## Why the earlier number was different

The corpus had to be re-embedded. The original was built with
`jina-v5-omni-small` via the sidecar on `localhost:8001`; that endpoint is down,
and the vectors it produced are orthogonal to anything else — cosine of a
stored vector against a fresh `BAAI/bge-m3` embedding of its own text measured
−0.007, −0.038, −0.013 across three records, against a +1.000000 self-consistency
control. Same 1024 dimensions, different space. A query in the wrong space is
*accepted* and returns ranked, plausible, meaningless results, so this had to be
checked before anything else. The schema records `model_id: null`, so nothing
in the system would have caught it.

Re-ingested with `BAAI/bge-m3` at `gpu-host-b:8000` — 2,012 documents in 175s —
retrieval is good: for a fact question the correct fact document is the top hit
above 2,000 filler documents.

**The most probable explanation for 3/6 is therefore that the original
measurement was retrieval-limited, not code-limited.** With poor or absent
retrieval the agent has no evidence and answers from the prior, which is exactly
what a bad negative-answer rate looks like. With retrieval working, refusing to
answer is easy — and it is easy on *both* sides of the range.

The two failures observed at the before-endpoint support that reading, because
neither is the failure mode the task described:

- `negative-1 r0` — "I reached the maximum number of reasoning steps before
  producing a final answer": step-limit exhaustion, i.e. the loop the stagnation
  note (#12) exists to cut.
- `negative-2 r3` — declined to answer but in prose, without emitting the literal
  token `unknown`: a compliance miss, not a hallucination.

Neither is "the model confidently named a supplier that does not exist", which is
what a 50% negative rate implies.

## What this does and does not establish

**Established:** no commit in `7882b3c..356d6c9` can be credited with closing the
gap, because with `BAAI/bge-m3` the gap is not present at either end.

**Not established:** that the code never mattered. The embedder was changed, which
is a confound — a deliberate one, since the original was unavailable. The open
question has moved from *"which commit fixed it?"* to *"was the 2026-08-18
measurement embedder-limited?"*, and the way to settle that is to restore
`jina-v5-omni-small` on `:8001`, re-ingest, and re-measure `7882b3c`. If it scores
3/6 there and 95% under bge-m3, the answer is the embedder and #72 closes for good.

## Method notes

- The probe puts **`filter_query` in `tool_ids`**, not just `search`. `stag_v2.py`
  ships `["search"]` alone, and with that the embedding-strip commits
  (`05661e7`/`214d5db`) are invisible — a `search` hit carries `content`, `doc` and
  `_guardrail_*` and no vector, so `strip_embeddings` is a no-op on that path.
  Running the stock harness would have silently excluded the leading candidate
  from its own bisect.
- Negatives run **10 repeats per case (20 samples)**, not 3 (6 samples). "3/6" is
  six coin flips; a bisect step decided on that is a random walk.
- Every probe re-runs the **controls as a validity gate**. Since the corpus was
  re-embedded, retrieval quality could not be assumed; a run whose controls fail
  reports `INVALID` instead of a negative rate that reflects a broken corpus.
  Controls were 6/6 on every run.

## Corrected costs

The task list estimated ~10 min per commit (6 min build + 4 min suite). Measured:
**`cargo build --release -p veda-server` is ~12m45s** (two samples: 12m37s,
12m48s) and **the probe is ~46s**. Both halves were wrong, in opposite
directions. A five-step bisect would have been ~70 minutes, essentially all
compile time.

Harness: `/home/mac/test/stag_bisect.py` (`ingest` / `probe`). Corpus:
`/home/mac/test/stag-data-bge`.

# SPANN ingest memory, measured at scale

**2026-09-11.** The website brief has carried "do NOT say it makes ingest
cheaper" since 2026-08-20, gated on one condition: *"Say ingest is cheaper
only once a segment-backed `VectorSource` is wired into `build_indexes`."*
That wiring landed (#73, PR #26). This is the measurement that was still
missing — the prior evidence was a 4,000-vector routing test, which proves
`build_indexes` takes the streaming path and says nothing about memory at
scale.

## Numbers

`cargo test --release -p veda-query --test spann_scale_measurement --
--ignored --nocapture`, at **dim=1024** (the dimension this product runs:
`EMBEDDING_DIMENSION=1024`, and jina-v5-omni and bge-m3 are both 1024), with
`VEDA_SPANN_BUILD_BUDGET_BYTES=8 MiB`.

| vectors | corpus | peak buffered | peak/corpus | passes | groups |
|---|---|---|---|---|---|
| 25,000 | 97.7 MiB | 8.0 MiB | 8.2% | 15 | 13 |
| 50,000 | 195.3 MiB | 6.3 MiB | 3.2% | 34 | 32 |
| 100,000 | 390.6 MiB | 12.2 MiB | 3.1% | 67 | 65 |

**Corpus grew 4.0×; peak grew 1.5×.** Memory tracks the budget, not the
dataset. The in-memory route holds the whole corpus, so at 100k × 1024d the
comparison is 12.2 MiB against 390.6 MiB.

## What can now be said, and what still cannot

**Can be said:** choosing `spann` bounds *peak build memory* near a
configurable budget instead of the corpus size. At 100k × 1024d that measured
3.1% of the corpus.

**Cannot be said — "ingest is cheaper" without qualification.** Bounded memory
is bought with more I/O: passes over the segment grew 15 → 34 → 67 across the
same three sizes. It is a memory/IO trade, not a free win. A claim that
implies ingest is *faster* is not supported by anything here.

**Not measured: wall-clock build time.** The run took 465s total across all
three sizes including corpus generation, and per-size build time was never
isolated. Do not quote a time figure from this.

**Peak can exceed the budget.** 12.2 MiB against an 8 MiB budget at 100k. That
is the documented floor, not a leak: a posting list is never split across
parts, so it is always buffered whole, and `nlist` governs the floor. The
budget is a target, not a cap — say "bounded near" rather than "capped at".

## Why this needed measuring rather than extrapolating

The 4k test asserts `peak_buffered_bytes * 4 < corpus_bytes` on a single
corpus size. That catches a regression in routing and cannot distinguish
"memory is bounded" from "memory is proportional but the constant is small".
The scale test asserts the growth *ratio* — quadruple the corpus and peak must
not quadruple — which is the property the claim actually rests on, and which
would have failed loudly if the streaming build were not bounding anything.

This project has retracted published index numbers before
(`2026-08-19-recall-remeasurement.md`), which is why the brief gated the claim
in the first place.

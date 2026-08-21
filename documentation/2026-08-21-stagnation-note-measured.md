# 2026-08-21 — What the stagnation note actually does

**Status:** measured, A/B'd, and it is not what the earlier reading suggested.

`dbc7279` shipped the relevance-based stagnation detector and deliberately declined to claim it
fixed the negative-answer deficit: *"This is NOT yet claimed to improve the negative-answer
deficit (3/6 in the v2 measurement). The firing rate should be read off the new counter on real
traffic before any such claim — which is exactly the step v1 skipped."*

This is that step. The conclusion is that the note is worth keeping, but **not for the reason it
was built.**

## Method

The v2 suite (11 cases × 3 repeats: control, one-to-one, one-to-many, negative) over the existing
2,012-document `stagv2` corpus, against Qwen3.5-122B. Two arms, differing **only** in whether
`STAGNATION_NOTE` is appended to the tool result the model sees — the counter is incremented
identically in both, so firing rates are comparable.

Run twice with the arm order **reversed** the second time, so a warmer LLM endpoint on one arm
cannot explain the difference. 99 conversations per arm.

Two measurement traps had to be cleared first:

- **The note is invisible in the API response.** It is appended to the `ChatMessage` sent to the
  model, while `AgentStep::ToolResult` is pushed earlier with the raw result. A client-side probe
  scanning `steps` for the note text reports `fired = 0` no matter what happens. The first pass
  here did exactly that and produced a clean, confident, entirely false zero.
  `veda_agent_stagnation_checks_total{fired}` is the only honest source, which is why it exists.
- **A single run cannot tell a fix from variance.** One arm's `one-to-many` scored 2/6 on one run
  and 6/6 on the next, same binary, same corpus.

## Results

| | note OFF | note ON | mean searches OFF → ON |
|---|---|---|---|
| control | 27/27, fires **0%** | 27/27, fires **0%** | 1.00 → 1.00 (+0%) |
| one-to-one | 36/36, fires 25% | 36/36, fires 25% | 1.78 → 1.75 (−2%) |
| one-to-many | 14/18, fires 62% | 14/18, fires 49% | 3.44 → 2.72 (**−21%**) |
| negative | 16/18, fires 54% | 18/18, fires 50% | 3.17 → 2.00 (**−37%**) |

### What holds

**The note cuts wasted retrieval.** 37% fewer searches on negative questions and 21% fewer on
one-to-many, reproduced across both replications, at no accuracy cost. That is the effect.

**The detector is specific.** Across 54 control conversations it fired **zero** times. Controls
answer from one search and there is nothing to stagnate on; a detector that fired there would be
noise. v1 fired about 1 run in 6 globally and changed nothing — v2 fires 25–62% on exactly the
categories that search repeatedly, and never on the ones that do not.

### What does not hold

**The note is not what fixed the negative-answer deficit.** With the note suppressed, negative
questions still score 16/18 (89%) — nowhere near the 3/6 (50%) baseline of 2026-08-18. Whatever
closed that gap is something else that landed between then and now; this A/B cannot say what, only
that it is not this.

**The accuracy difference is not established.** 18/18 versus 16/18 is two cases on n=18 per arm.
It points the right way and it is not significant. Do not quote it as an improvement.

## Consequence

Keep the note, and describe it as what it measurably is: **a retrieval-cost reduction on questions
whose answer is not in the corpus**, not an accuracy fix. The earlier informal reading — negative
answers 3/6 → 6/6 *because of* the stagnation change — was an artefact of comparing across builds
without an A/B, and is retracted here.

The open question is what did close the negative-answer gap between 2026-08-18 and 2026-08-21.

## One candidate tested and set aside

`05661e7` / `214d5db` stopped dumping raw 1024-dim embeddings into the model's context, and the
docstring on `strip_embeddings` records that removing them moved an annotation-probe case from
0/5 to 5/5. That is the most plausible candidate in the range, so it was tested the same way: a
binary with the strip suppressed, two runs.

Every category scored 100%, identical to the normal binary — which looks like a refutation and is
**not one**. Inspecting the step trace shows a `search` hit carries `content`, `doc` and the
`_guardrail_*` columns and **no vector at all**. `strip_embeddings` is a no-op on this suite, so
the arm was never different from the control. The measured 0/5 → 5/5 was on `filter_query`, and
this suite's agent is `tool_ids: ["search"]`.

The result is therefore *inconclusive by construction*, not evidence either way — the same class
of error as the `fired = 0` probe above, caught the same way, by checking whether the mechanism
could fire before believing that it did not.

**A bisect over the range is still the way to answer this**, and it must use an agent with
`filter_query` in its toolset if it wants the embedding-strip commits to be distinguishable at
all. Worth doing before the next agent-loop change claims credit for something it did not do.

## Reproducing

```bash
# on gpu-host-a, with the stagv2 corpus already ingested on :6357
python3 /home/mac/test/stag_repeat.py 2      # 2 runs x 3 repeats, reuses the corpus
```

`stag_repeat.py` reads the firing rate from `/v1/admin/metrics` rather than the response trace,
for the reason above. `stag_v2.py` re-ingests all 2,012 documents on every invocation and should
not be used for repeats.

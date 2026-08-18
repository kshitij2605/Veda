# Multi-hop retrieval

Some questions have no answer in any single document.

> *"Is the plant that assembles Product Halcyon certified to ISO 14001?"*

One document names the plant that assembles Halcyon. A different one lists that plant's certifications. No single search returns both, because until you know the plant's name you cannot search for its certificate.

Veda's agent handles this by searching again. It looks up the plant, reads the name out of the result, then searches for that plant's certifications, and answers from the two together.

## There is no graph and no planner

This is ordinary tool-calling over the same [`search` tool](/docs/agents-mcp) the agent already had. There is **no knowledge graph, no query planner, and no new index** — nothing to build, nothing to keep in sync with your data, and nothing extra to operate.

The agent decides how many searches to run. Nothing is pre-declared, and the number of hops is not something you configure or that Veda guarantees — the loop is bounded by a step limit, and how many searches it spends inside that limit is the model's choice.

## What was measured

On a **2,012-document corpus**: 12 fact documents buried in 2,000 filler documents sharing the same business vocabulary, so that no single search can return both halves of an answer. Reasoning model **Qwen3.5-122B**, `jina-v5-omni-small` embeddings, 3 repeats per question. The whole suite was run twice on separate days and returned identical figures.

| Question type | Correct | Mean searches |
|---|---|---|
| Single-hop (control) | 9/9 (100%) | 1.0 |
| Two-document join — *"does X also appear in Y"* | 12/12 (100%) | 1.8 |
| One-to-many join — *"list **every** supplier where…"* | 6/6 (100%) | 2.2 |

One-to-many questions are the interesting ones: they require finding *all* the matches rather than stopping at the first. Those were answered completely, in about two searches on average.

Retrieval was not the limiting factor in any of these cases.

Those figures belong to that corpus and that model. This is Veda's own suite on a synthetic corpus — no public multi-hop benchmark (HotpotQA, MuSiQue, 2WikiMultihopQA) has been run, so it carries no benchmark standing.

## Where it is weakest: questions with no answer

Asked something the corpus genuinely does not contain, the agent answered correctly **3 out of 6 times**. The other times it recited what it *did* find instead of saying "none".

This is the honest weak spot. Answering *"the corpus does not contain this"* is markedly less reliable than answering a question that has an answer, and it is worth knowing before you point this at a corpus where absence is meaningful — a compliance check, say, where "no supplier in Poland" and "here are some suppliers" are very different answers.

The agent also works harder on these: it spent noticeably more searches on unanswerable questions than on answerable ones before giving up.

## Timing

Multi-hop answers took roughly 2–9 seconds in this measurement. That is dominated by round-trips to an external LLM endpoint and is **not a Veda performance figure** — it will move with whatever model you point it at.

## Using it

Nothing to enable. Ask an [agent](/docs/agents-mcp) a question that needs more than one lookup, and watch the `steps` array in the response: each `tool_call` / `tool_result` pair is one hop, and the retrieved passages come back as [citations](/docs/rag-aiops#faithfulness-citations) so you can see which documents each part of the answer came from.

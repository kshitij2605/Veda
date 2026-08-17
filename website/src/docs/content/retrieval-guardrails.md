# Retrieval guardrails

Most guardrails screen what the user typed and what the model said back. Retrieval guardrails screen the third thing in the room: **the documents you retrieved**.

The attack text doesn't arrive in the prompt. It sits inside your corpus — a PDF someone uploaded, a page you crawled, a vendor deck — and rides into the context window through retrieval. By the time it gets there it looks like legitimate retrieved content, which is exactly why [input/output guardrails](/docs/llm-observability) cannot see it. This is the OWASP indirect prompt-injection vector.

![A live run: a handbook collection containing a planted instruction. Raw search returns it flagged, then an ordinary question to the agent comes back answered with the hostile chunk dropped — category indirect_injection, score 0.63.](/videos/retrieval-guardrails.mp4)

## What gets screened, and when

Every chunk is screened **at ingest**, against the embedding Veda has already computed for it. There is no extra model call on the ingest path — the screening reuses work that was happening anyway.

Each screened record is stamped with reserved fields:

| Field | Type | Meaning |
|---|---|---|
| `_guardrail_flagged` | bool | Screening flagged this record. Absent means never screened — which reports as `unscreened`, **not** as clean. |
| `_guardrail_score` | float | Strongest pattern similarity seen. Stamped even when *not* flagged, so sub-threshold near-misses stay measurable. |
| `_guardrail_category` | string | Category of the strongest match, when flagged. |
| `_indexed_at` | int | Ingest time, epoch seconds — used by the temporal staleness check. |

## Enforcement happens once, for every surface

Dropping happens at the tool-dispatch choke point rather than being bolted onto each endpoint, so every LLM-facing surface is covered at once: `/converse`, `/converse/stream`, MCP `tools/call`, `/agent-tools/{id}/execute` and `/a2a/{agent_id}`.

Chunks flagged `PromptInjection`, `Jailbreak`, `Unsafe` or `IndirectInjection` are dropped before the model sees them. A flagged chunk carrying an unrecognised or missing category **fails closed** and is dropped.

**PII is detected and reported, then passed through.** That is deliberate. Live calibration found PII-only chunks — a contact detail in an otherwise-benign deck — being dropped exactly like attacks, and silently deleting a real phone number from an answer was judged the worse failure.

## Raw search annotates, never drops

`POST /search` does not drop anything. It annotates hits with `_guardrail_class` and returns them. Retrieval quality therefore stays measurable, and the security layer cannot quietly degrade your recall without showing up as a security finding.

## The report

Every LLM-facing response carries a `retrieval_guard` report:

```json
{
  "kept": 8,
  "dropped": 1,
  "skipped": 0,
  "flagged_not_dropped": 1,
  "unscreened": 0,
  "dropped_chunks": [
    { "id": "…", "category": "prompt_injection", "score": 0.83 }
  ]
}
```

It is **omitted entirely when the turn made no retrieving tool call** — "did not retrieve" and "retrieved, all clean" are different facts and are not collapsed into an all-zero report.

`skipped` and `unscreened` are always serialized even at zero, so an operator can tell them apart: `skipped` means there was nothing to screen against, `unscreened` means a verdict is missing.

## Calibration, and what it does not tell you

The threshold was calibrated against a real corpus of **10 PDFs / 149 chunks** of known-clean Japanese business documents. The semantic layer produced **0 false positives out of 149**, with the nearest miss at **0.354** against a **0.60** threshold.

That corpus contained **no planted attacks**, so it measures one thing only: the screen does not fire on ordinary documents. It is not a measurement of how many attacks it catches.

**Treat this as a defence-in-depth layer, not a security boundary.** Detection is cosine similarity between a chunk's embedding and a curated seed list of hostile phrasings, and that has two consequences worth planning around:

- An attack phrased unlike the seed patterns scores low. Canonical "ignore all previous instructions" wording is what it catches best.
- **A hostile sentence buried in a longer, benign chunk is diluted.** The chunk's embedding is dominated by its bulk, so the similarity stays low even though the instruction is right there in the text.

Our own adversarial testing bears both out. If retrieved-document injection is part of your threat model, pair this with controls that do not depend on recognising the attack — least-privilege tools, human review of what enters the corpus, and treating retrieved text as untrusted input at the prompt layer.

## What it does not cover

These are the edges worth knowing before you rely on it.

- **Not every record is screened.** Plain `POST /v1/collections/{name}/insert` takes client-supplied vectors, so there is no Veda-computed embedding to screen against; those records are stamped `skipped`. **This includes the Python SDK's normal client-side-embedding flow.** Screening happens on `process-document`, `ingest-url` and `embed-insert`.
- **Existing data is not covered automatically.** Chunks indexed before this feature carry no verdict, report as `unscreened`, and pass through. Enforcement fails open there by design — treating unstamped chunks as hostile would empty every existing collection on upgrade. Close the gap with `POST /v1/admin/collections/{name}/rescan-guardrails`.
- **The re-scan is a manual admin action, not schedulable.** Each run materialises a full copy of the collection, and superseded copies are not reclaimed, so repeated re-scans grow storage.
- **Detection is similarity-based against a threshold**, so it is not a proof. An injection phrased far enough from the known patterns can score below the threshold and pass. The score is stamped on every chunk as `_guardrail_score` even when it is not flagged, so near-misses stay auditable — treat the guardrail as one layer, not the only one.
- **Temporal conflict detection** is active on `filter_query` and inert on `search`.
- This targets *retrieved-document* injection. Injection in the user's own message is the separate input/output guardrail layer.

## Turning it down

`VEDA_RETRIEVAL_GUARD=off` degrades every LLM-facing path to annotate-only: hits are still classified and still reported, none are removed. The switch can only ever **weaken** enforcement — it can never enable it by accident, because enforcement is on by default.

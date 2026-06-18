# Veda Agentic AIOps — Change-Point Detection, Semantic Spike Attribution, Proactive Incident Drafts + Noise Reduction, Incident Novelty Detection

**Date:** 2026-06-18
**Status:** Shipped on `feat/agentic-aiops` (also on `main`), v0.3.1. Validated end-to-end against a live server.

Agentic AIOps closes the gap against Elastic's AIOps-for-Observability. Veda already covered
~80% of it — the Ops Advisor (snapshot → rules → events → channels → Niriksha), Log Insights,
and the prior RAG/AIOps cycle (semantic incident correlation, capacity forecasting, the
cross-signal RCA bundle, and the `veda-ops` triage agent) — so this cycle is about the *last
mile*, and it leans on the one structural advantage Elastic doesn't have. **Elastic's AIOps is
statistical and template-based over text** — log-rate analysis bucketizes by message templates,
anomaly detection runs population/time-series statistics, and the "explain a spike" feature
counts term frequencies. **Veda's is semantic and embedding-based over every signal it owns.**
Because Veda owns the embedder, the log categorizer, the incident correlator, the RCA bundle,
*and* the agent, it can answer the questions a statistical engine can only approximate: not
"the log rate went up" but "*this meaning-cluster* drove the spike"; not "an incident opened"
but "this incident is *unprecedented*"; not "here is the event" but "here is the drafted
root-cause, grounded in the cross-signal bundle." Every feature here reuses an existing
primitive and degrades gracefully — a 503-safe draft, a template fallback, or a novelty skip —
when no LLM or embedder is configured. The Ops Advisor catalog is now ~34 rules
(+`change_point_detected`).

---

## 1. Change-point detection

The advisor already had EWMA/σ anomaly detectors — but those fire on a single outlier point. A
sustained regime change (the disk-used baseline steps up after a config change; RSS starts
climbing on a new trend) is not a spike, and a single-point detector either misses it or fires
once and resets. Change-point detection answers a different question: *did the series move to a
new level, or break onto a new trend, and stay there?*

`advisor/changepoint.rs` runs over the same advisor metric series the snapshot history already
records — `disk_used_pct`, `rss_bytes`, query `p95`, the error-log count, and per-collection
`recall`. For each series it computes a **two-window statistic**: a least-squares fit on each
side of a candidate split, detecting both a **LEVEL** shift (the mean steps up/down) and a
**TREND** break (the slope changes). Three guards keep it from crying wolf:

- a **well-fit gate** — the split is only accepted when the two-window model fits materially
  better than a single-window fit, so noise around a stable mean doesn't register as a shift;
- a **warm-up** (≥20 points) — it won't declare a change-point until it has enough history; and
- a **min-magnitude** guard (1.5σ) — a shift smaller than 1.5 baseline sigmas is ignored.

The result is **pure and in-tick** — no network, no LLM, just arithmetic over the series the
advisor already holds — so it costs nothing on the tick path. Detections are attached to
`OpsSnapshot.change_points` and surfaced two ways: a new rule **`change_point_detected`**
(Medium) opens an event through the existing event store + alert channels like every other
rule, and `GET /v1/admin/change-points` returns the raw detections (metric, scope, kind,
before/after, magnitude in σ, detection time). It reuses the SnapshotHistory series and the
anomaly machinery's baseline math; it adds the *sustained-shift* dimension the point detectors
lacked.

---

## 2. Semantic spike root-cause attribution

When the error/warn-log rate spikes, the operator's first question is "what is driving it?".
Elastic's "log rate analysis" answers this statistically — it bucketizes messages by template
and reports which templates grew. That works, but a template is a syntactic artifact: two
log lines that *mean* the same thing but differ in wording land in different buckets, and a
template count can't tell you a genuinely new *kind* of failure appeared.

`advisor/spike_attribution.rs` answers it **semantically** by reusing the embedding
`LogCategorizer` that Log Insights already ships. Given a spike `window` and a `baseline`
window, it embeds and clusters the logs in each, then attributes the `total_delta` to the
meaning-clusters driving it:

```
{
  window, baseline, total_delta,
  drivers: [
    { category, representative, spike_count, baseline_count,
      growth, share_of_growth, is_new }
  ],
  method
}
```

Each driver names a semantic **category** (with a `representative` line), its `spike_count`
vs `baseline_count`, its `growth` ratio, its `share_of_growth` (what fraction of the total
delta it accounts for), and `is_new` (the category had no presence in the baseline window — a
brand-new failure mode). This is the vector-native answer to a statistical log-rate analysis:
it names *which meaning-cluster* drove the spike, and it can say "this is new" rather than just
"this template's count rose."

Embedding runs **off the advisor lock**, so the analysis never stalls the tick. When no
embedder is configured it degrades to the same **normalized-template fallback** the
`LogCategorizer` uses elsewhere, and reports `method="template"` so a consumer never mistakes
the fallback for the semantic path (`method="embedding"`). `GET
/v1/admin/log-rate-analysis?window_secs=&baseline_secs=&min_level=` exposes it on demand.

---

## 3. Proactive incident drafts + alert-storm noise reduction

The prior cycle gave Veda semantic incident correlation (a storm of related events collapses
into one `Incident`), the cross-signal RCA bundle (logs + queries + traces + resources + the
incident, joined locally), and the `veda-ops` triage agent (read-only ops tools, grounded
explanations). On-demand triage was a button you pressed. This cycle makes it **proactive**.

On a **newly-correlated incident** — detected off-tick by the incident-correlation runner — the
`veda-ops` agent **auto-runs over the incident's cross-signal RCA bundle** and drafts a
root-cause paragraph plus a runbook step into `Incident.draft`. The operator opens an incident
and the first-pass RCA is already there, grounded in the same bundle a human would have pulled.
Three properties keep it safe and cheap:

- **best-effort and 503-safe** — when no LLM is configured the draft is simply `None`; the
  incident is unaffected and nothing fails;
- **new-incident-only** — a stable incident that's already been drafted is *not* re-drafted on
  every runner pass, so the LLM cost is one draft per incident, not one per tick; and
- it runs **off-tick** on the incident-correlation runner, so it never touches the advisor tick
  path.

Alongside the drafts, this cycle fixes the alert-storm problem at the **notification** layer.
Notifications now fire per **incident**, not per event: a storm of twelve correlated events that
folds into one incident pages you **once** ("N alerts → 1 incident"), deduped via a set of
persisted notified-ids so a given incident notifies on its transition rather than on every
member event. `GET /v1/admin/incidents` carries the accounting as a `noise_reduction {
event_count, incident_count, suppressed }` summary, where `suppressed = event_count −
incident_count` — the number of pages you *didn't* get. This reuses the incident correlator,
the RCA bundle, the `veda-ops` agent, and the existing alert-channel layer; the only new
machinery is the per-incident dedup set and the draft field.

---

## 4. Incident novelty detection

Not every incident is equally interesting. A flapping rule you've seen a hundred times deserves
a different response from a failure pattern that has *never occurred before*. A statistical
engine struggles here — "novel" is a semantic judgment about whether this incident *means*
something the system hasn't seen — but Veda already embeds every incident for correlation, so
the answer is one cosine away.

Novelty detection reuses the **incident-clustering embeddings**. The `IncidentStore` keeps a
persisted **embedding history** of past incidents; when a new incident is correlated, Veda
computes its `novelty_score = 1 − max_cosine_to_history` (1.0 = nothing in history looks like
it; 0.0 = a near-duplicate of a prior incident) and flags `is_novel` when the score clears the
threshold. Two guards prevent false novelty:

- a **warm-up** (≥5 historical incidents) — with too little history *everything* looks novel,
  so below the warm-up the incident is simply not flagged; and
- an **embedder-free skip** — with no embedder there are no embeddings to compare, so novelty is
  skipped (`novelty_score` is null) rather than fabricated.

A novel incident **escalates its per-incident notification**: the page is tagged
"⚠ UNPRECEDENTED pattern" and the severity is raised to **High**, so a genuinely new failure
mode jumps the queue. `is_novel` and `novelty_score` ride on `/v1/admin/incidents` and a
Niriksha badge marks novel incidents in the Insights "Incidents" tab. This is pure reuse of the
incident embeddings the correlator already computes — novelty is a read over the same vectors.

---

## 5. REST / metrics / env reference

### REST (all `/admin/*` are admin-only — `bearerAuth`)

| Endpoint | Method | Description |
|---|---|---|
| `/v1/admin/change-points` | GET | Change-point detections (sustained level shifts / trend breaks) in the advisor metric series; warm-up (≥20 pts) + min-magnitude (1.5σ) guarded |
| `/v1/admin/log-rate-analysis` | GET | Semantic spike root-cause attribution — the meaning-clusters driving a log-rate spike (`?window_secs=&baseline_secs=&min_level=`); `method` is `embedding` or the `template` fallback |
| `/v1/admin/incidents` | GET | Semantic incident clusters; now carries `draft` (veda-ops-drafted RCA, null w/o LLM or for a non-new incident), `is_novel` / `novelty_score`, and a `noise_reduction { event_count, incident_count, suppressed }` summary |

### Advisor rules

| Rule | Severity | Notes |
|---|---|---|
| `change_point_detected` | Medium | Fires on a sustained level shift / trend break in the advisor metric series; brings the catalog to ~34 rules |

### Configuration (env vars)

| Variable | Default | Purpose |
|---|---|---|
| `VEDA_INCIDENT_INTERVAL_SECS` | `300` | Cadence of the off-tick incident-correlation runner, which also drives proactive drafts + novelty scoring |
| `VEDA_LLM_ENDPOINT` | *(unset)* | When unset, proactive incident drafts are `None` (503-safe); the rest of the pipeline is unaffected |

(Change-point detection and spike attribution have no dedicated env knobs — change-points run
in-tick on the existing advisor cadence, and spike attribution is on-demand at the endpoint.)

---

## 6. Validation (live)

Verified against a running server on the Linux host:

- **Proactive incident draft (end-to-end):** tripped a `disk_space_low` event; within ~15s the
  incident correlator formed an incident and the `veda-ops` agent **auto-drafted a grounded
  RCA** — *"disk at ~75.7% … likely a false positive from a misconfigured threshold … verify
  via `get_resources` and adjust"* — attached to the incident.
- **Novelty:** the same incident reported **`is_novel=false`** — the warm-up (≥5 historical
  incidents) correctly suppressed a novelty flag rather than fabricating one.
- **Noise reduction:** the incidents response carried **`noise_reduction {1,1,0}`** (one event,
  one incident, zero suppressed) for the single-event case.
- **Surfaces:** `GET /v1/admin/change-points`, `GET /v1/admin/log-rate-analysis`, and `GET
  /v1/admin/incidents` all returned **200**.

---

## 7. Limitations & follow-ups

- **Live-flagging of change-points / spikes / novelty is exercised via unit tests.** Driving a
  real sustained step-change in the metric series, a real log-rate spike with distinct semantic
  clusters, or a genuinely novel incident inside a quick end-to-end run is impractical (it
  requires accumulating warm-up history and crafting distinct failure modes), so the detection
  logic itself is covered by unit tests; the live e2e exercised the proactive-draft +
  noise-reduction + novelty-warm-up path against a real incident.
- **Spike attribution and novelty need an embedder.** Without one, spike attribution falls back
  to the normalized-template path (`method="template"` — coarser, since it can't cluster by
  meaning), and novelty is **skipped** (`novelty_score` null) rather than guessed. `method` /
  the null score always report which path ran.
- **Proactive drafts need an LLM.** Without `VEDA_LLM_ENDPOINT` the draft is `None` (503-safe);
  the incident, correlation, and noise-reduction summary are all unaffected.
- **Novelty warm-up (≥5 incidents) means early incidents aren't flagged.** Until the
  `IncidentStore` has accumulated enough history, novelty is intentionally suppressed to avoid
  flagging everything as new in a fresh deployment.
- **Two pre-existing gateway warnings.** The `gateway` crate carries two compiler warnings that
  predate this cycle and are unrelated to it.

# Consensus crate swap — decision

**Decision: do not swap.** Stay on openraft 0.9 (or its eventual
stable 0.10), live with the residual disruptive-election failure
mode, and mitigate it at the application layer. Capture the swap
surface in `crates/veda-coordinator-raftrs/` as a design artifact so
the cost stays visible if the calculus ever changes.

## The actual choice

After Phase 2 (Raft migration) shipped, the visible failure mode that
keeps coming up in the soak suite is "disruptive elections on
partition heal" — a partitioned node times out, bumps its term, and
on heal forces the existing leader to step down. Bounded by one
election timeout of write unavailability (~1-2 s), no data loss, but
ugly in logs.

True Raft pre-vote (candidates poll for a quorum *before* bumping
term) eliminates this. openraft does not implement pre-vote and the
maintainers have publicly declined to add it.

The decision is a binary one — there is no "just wait for 0.10" path:

| Option | What you get | What you give up | Cost |
|---|---|---|---|
| **Stay on openraft + mitigate (chosen)** | Existing async actor architecture, active upstream, openraft's `Vote` design narrows the disruptive-election window | Residual ~1-2 s unavailability per partition heal | Hours-of-work mitigations |
| Swap to tikv/raft-rs 0.7 | First-class `Config::pre_vote = true`, true pre-vote elimination of disruptive elections | Fundamentally different API (sync `Ready` loop vs. async actor), library in maintenance mode since 2023-03-07, full rewrite of storage + transport + driver | 3-5 engineer-weeks (see stub) |
| Swap to riteraft / raftbare / little_raft / canonical-raft / paxakos | — | All ruled out for maintenance / API / scale reasons | — |
| FFI to hashicorp/raft or etcd/raft | — | Cross-language pain dominates any benefit | — |

The full survey is at
`research/2026-05-19-multi-week-survey/03-consensus-crates.md`.

## Why "wait for openraft 0.10" is not the answer

It would be convenient if a routine upgrade to openraft 0.10 closed
this gap. **It does not.** What we verified against the upstream
repo (2026-05-19):

- Latest crates.io publish: `0.10.0-alpha.20` (2026-05-07).
- Searching `openraft/src/core/raft_core.rs` for `pre_vote`,
  `PreVote`, or `PreCandidate` returns nothing.
- Discussion #15, the canonical thread on pre-vote, still rests on
  the maintainer's 2021 position: *"Just do not increase the term
  of a candidate unless it has to."* The intent is to **replace the
  need for pre-vote**, not to add it.
- 0.10's election-related changes are refinements of the existing
  `Vote` design — e.g. migrating `single-term-leader` from a feature
  flag to a type-level `LeaderId` config — plus bug fixes. Not the
  addition of a real pre-vote round.

So the disruptive-election window is narrowed by the `Vote` design
in both 0.9 and 0.10 but **never eliminated** in either. If we want
true pre-vote semantics, the only path is the raft-rs swap.

## Why we're still choosing to stay

The failure mode is real but bounded:

- One election timeout (~1-2 s) of write unavailability per
  partition heal event. No data loss. No consistency violation.
- The 24-hour soak observes it as transient log churn, not as data
  divergence. The bounded-lag convergence checker (`max_lag <= 10`)
  remains green across hundreds of injected partitions.
- Application-layer mitigations close most of the practical gap at
  hours-of-work cost:
  - Switch to the `single-term-leader` semantics (one leader per
    term, no leader_id-tuple log_id) — aligns with what most
    operational tooling expects, reduces some election conflicts.
  - Add an application-level CheckQuorum: a leader that fails to
    hear from a quorum within N election timeouts voluntarily steps
    down. ~100 lines watching `RaftMetrics`.
  - Wider random jitter on election timeouts (`min=2000ms,
    max=4000ms`) — reduces spurious elections in normal jitter.
  - Rate-limit term bumps in `peer_health.rs` so a flapping node
    can't repeatedly disrupt the cluster.

3-5 engineer-weeks of swap work to close a bounded, recoverable
failure mode is not the right trade today.

## What `crates/veda-coordinator-raftrs` contains

A skeleton crate, **not in the workspace**, that types-check against
raft-rs 0.7's API but where every method body is `unimplemented!()`.
Four pieces:

1. **`VedaRaftRsStorage`** — `raft::Storage` impl (read side:
   `initial_state`, `entries`, `term`, `first_index`, `last_index`,
   `snapshot`).
2. **`WriteableStorage` trait** — captures the write side that
   raft-rs delivers via `Ready` (`save_hard_state`, `append`,
   `apply_snapshot`, `save_conf_state`).
3. **`Transport` trait** — single-method send; the receive side is
   driven by `RawNode::step` in the Ready loop.
4. **`Driver`** — actor that would own the `RawNode<VedaRaftRsStorage>`,
   tick it, drain `Ready` each iteration, persist log entries via
   `WriteableStorage`, dispatch messages via `Transport`, and apply
   committed entries to the existing `ClusterStateMachine`.

The comments at each `unimplemented!()` site document what the body
needs to do. Filling them in is the bulk of the 3-5 engineer-week
swap cost.

## Why excluded from the workspace

raft-rs has heavyweight transitive deps (protobuf, slog,
slog-async, etc.). Pulling them into every developer's `cargo build`
to host a never-called stub is wasteful. Excluding the crate from
`[workspace.members]` (and listing it under `exclude`) keeps the
artifact on disk for reference without dragging the deps into normal
work.

To examine the swap surface:

```
cd crates/veda-coordinator-raftrs
cargo build
```

To delete the artifact entirely if we decide we'll never swap,
remove the directory and the `exclude` line in the workspace
`Cargo.toml`.

## Triggers that would revisit the decision

- The application-layer mitigations are deployed and the
  disruptive-election failure mode still causes operational pain —
  raft-rs's true pre-vote becomes worth the swap cost.
- A specific consistency / availability incident in production is
  diagnosed as fixable only by a real pre-vote round.
- openraft reverses its 2021 position and adds pre-vote in a future
  release. Unlikely (the design philosophy is explicitly against
  it), but if it happens the upgrade replaces the swap.
- raft-rs picks up new releases after its 2023-03-07 maintenance
  pause — the "library in maintenance" risk goes away, lowering the
  cost of staying on it post-swap.

Until one of those triggers fires, the order of operations is:

1. Ship the `single-term-leader` migration.
2. Add application-level CheckQuorum.
3. Widen election-timeout jitter + rate-limit term bumps.
4. Reassess after a soak under representative production load.

All four are hours-of-work changes, none touch the surface area
captured in `veda-coordinator-raftrs`. Cheapest path first.

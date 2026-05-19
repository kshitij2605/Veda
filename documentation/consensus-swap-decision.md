# Consensus crate swap — decision

**Decision: do not swap.** Keep openraft 0.9; upgrade to openraft 0.10
when it stabilises. Capture the swap surface in
`crates/veda-coordinator-raftrs/` as a design artifact so the cost
remains visible if the calculus ever changes.

## Background

After Phase 2 (Raft migration) shipped, the visible failure mode that
keeps coming up in the soak suite is "disruptive elections on partition
heal" — bounded to one election timeout of unavailability, no data
loss, but ugly in logs. Mitigating it requires "pre-vote": candidates
poll for a quorum of votes before bumping their term. openraft 0.9
deliberately doesn't expose pre-vote (per the openraft maintainers, the
existing `Vote` design — term + leader_id tuple + `effective_leader`
heartbeat tracking — prevents the worst of the issue).

We surveyed alternatives. The full notes are at
`research/2026-05-19-multi-week-survey/03-consensus-crates.md`. The
short version:

| Option | Verdict |
|---|---|
| Stay on openraft 0.9 | Pre-vote not exposed; live with bounded election storms. |
| **Upgrade to openraft 0.10 (alpha)** | **Recommended.** 0.10's roadmap explicitly addresses simplifying vote/pre-vote semantics. Upgrade cost: hours-to-days. |
| Swap to tikv/raft-rs 0.7 | `Config.pre_vote = true` available. But: maintenance mode since 2023-03-07, fundamentally different API (sync `Ready` loop vs. async actor), 3-5 engineer-week swap cost. |
| Swap to riteraft / raftbare / little_raft / canonical-raft / paxakos | All ruled out for maintenance / API / scale reasons. |
| FFI to hashicorp/raft or etcd/raft | Cross-language pain dominates any benefit. |

The remaining question is "what would we have to write if the swap *did*
become unavoidable?" That's what `crates/veda-coordinator-raftrs`
captures.

## What `crates/veda-coordinator-raftrs` contains

A skeleton crate, **not in the workspace**, that types-check against
raft-rs's API but every method body is `unimplemented!()`. Four pieces:

1. **`VedaRaftRsStorage`** — implements `raft::Storage` (the
   read side: `initial_state`, `entries`, `term`, `first_index`,
   `last_index`, `snapshot`).
2. **`WriteableStorage` trait** — captures the write side that
   raft-rs delivers via `Ready` (save_hard_state, append,
   apply_snapshot, save_conf_state).
3. **`Transport` trait** — single-method send; the receive side is
   driven by `RawNode::step` in the Ready loop.
4. **`Driver`** — actor that owns the `RawNode<VedaRaftRsStorage>`,
   ticks it, drains `Ready` each iteration, persists log entries via
   `WriteableStorage`, dispatches messages via `Transport`, applies
   committed entries to the existing `ClusterStateMachine`.

The comments at each `unimplemented!()` site document what the body
needs to do. Filling them in is the bulk of the 3-5 engineer-week
swap cost.

## Why excluded from the workspace

raft-rs has heavyweight transitive deps (protobuf, slog,
slog-async, etc.). Pulling them into every developer's `cargo build`
to host a never-compiled stub is wasteful. Excluding the crate from
`[workspace.members]` (and listing it under `exclude`) keeps the
artifact on disk for reference without dragging the deps into normal
work.

To examine the swap surface:

```
cd crates/veda-coordinator-raftrs
cargo build
```

To delete the artifact entirely if the decision is finally to stay on
openraft forever, simply remove the directory and the `exclude` line
in the workspace `Cargo.toml`.

## Triggers that would revisit the decision

- openraft 0.10 lands and **does not** address vote/pre-vote
  satisfactorily — the swap becomes a real option.
- raft-rs picks up new releases after its 2023-03-07 maintenance
  pause — the "in maintenance" risk goes away.
- A specific failure mode that's only fixable by swapping is
  diagnosed in production.

Until one of those triggers fires, the cheaper mitigations are:

- Enable openraft's `single-term-leader` feature flag.
- Add application-level `CheckQuorum` watching `RaftMetrics`.
- Tune election-timeout jitter.
- Rate-limit term bumps in `peer_health.rs`.

All four are hours-of-work changes, none touch the surface area
captured in `veda-coordinator-raftrs`. Cheapest path first.

# Rust Consensus Crate Alternatives to openraft 0.9

*Survey week: 2026-05-19. Veda currently uses `openraft = "0.9"` with `features = ["serde"]` in `veda-coordinator`. Approx. 4.7k LoC of openraft-coupled code in `crates/veda-coordinator/src/`.*

## Executive Summary

**Recommendation: do not swap. Stay on openraft 0.9 and either (a) wait for 0.10 stable, or (b) implement a CheckQuorum-style application-level guard. The swap cost is multi-week and the failure mode pre-vote prevents is partial and well-bounded.**

If a swap is forced, the only realistic candidate is **tikv/raft-rs 0.7.0** via the **`riteraft` wrapper** (or a bespoke wrapper). Estimated swap cost: **3-5 engineer-weeks** of focused work on `veda-coordinator`, plus a re-do of the linearisability harness and Tonic transport plumbing. raft-rs has real pre-vote (`Config::pre_vote = true`, one line), is Apache-2.0, and is battle-tested by TiKV in production since 2016 — but the API style is fundamentally different (pull-based Ready pattern, synchronous core, BYO log/network/state machine), and the project is in maintenance mode (last release 2023-03-07).

The honest conclusion: **the disruptive-election failure mode is real but rare**, occurs only on partition heal, and is bounded by a single election timeout. openraft 0.9's `Vote` design (term + leader_id tuple comparison + effective_leader heartbeat tracking) already prevents *most* of the practical pathology pre-vote was designed for. The remaining residual risk does not justify a multi-week swap to a less actively-developed library.

---

## Candidate Survey

### 1. openraft 0.9.x (status quo)

- **Pre-vote**: Removed from roadmap. Discussion #15 explicitly states the intent to "get rid of pre-vote RPC" entirely. Replaced by a partially-ordered `Vote` struct: `(term_and_leader, effective_leader, log_id)`. A candidate only increments its term when necessary; `effective_leader` empties on heartbeat timeout. Vote granted iff `request.vote >= local.vote` in vector order.
- **License**: Apache-2.0 / MIT dual.
- **Activity**: Very active. v0.9.24 released 2026-04-21. Notable 2025-2026 patches include v0.9.19 (leader-revert tolerance on restart), v0.9.20 (heartbeat-response conflict handling), v0.9.22 (single-voter election fix), v0.9.23 (bounded snapshot streaming).
- **API style**: Framework-style, fully async, event-driven (not tick-based). Type-erased via `RaftTypeConfig`. User implements `RaftLogStorage`, `RaftStateMachine`, `RaftNetwork` + `RaftNetworkFactory`. Veda uses v1 `RaftStorage` adapter wrapped via `openraft::storage::Adaptor::new(...)` to produce the v2 log+state-machine pair.
- **Production users**: Databend (meta-service), CnosDB, RobustMQ, Helyim (SeaweedFS in Rust), rrqlite.
- **Pre-vote *equivalence***: openraft's `Vote` design prevents the most common pathology — a partitioned minority repeatedly incrementing term and disrupting on heal. The maintainer's argument: "do not increase the term of a candidate unless it has to". Combined with `single-term-leader` feature flag (which switches to standard one-leader-per-term semantics), this gives substantially-similar guarantees to pre-vote, without the extra RPC round.
- **What's missing vs. true pre-vote**: A candidate that *has* lost contact with quorum will still eventually time out, increment its term, and start an election. On partition heal, that incremented term *can* still force the existing leader to step down. The window is narrowed but not eliminated. CheckQuorum (a complementary technique) is not exposed as a first-class config option in 0.9.

### 2. tikv/raft-rs 0.7.0 ("raft-rs")

- **Pre-vote**: **Yes, first-class.** `Config { pre_vote: true, ..Default::default() }`. Documented as "This enables a pre-election vote round on Candidates prior to disrupting the cluster. Enable this if greater cluster stability is preferred over faster elections." Plus `check_quorum: bool` as a complementary setting.
- **License**: Apache-2.0.
- **Activity**: **Maintenance mode.** v0.7.0 released 2023-03-07 — *no releases for 3+ years as of 2026-05*. Repository still receives sporadic commits (PRs through 2024-09, issue activity through 2024-12). TiKV depends on it in production, so it won't bit-rot, but new feature development is essentially zero.
- **API style**: **Low-level**, pull-based `Ready` pattern, synchronous core with async hooks (`advance_append_async`). User writes a loop: `tick()` every ~100ms, `step(msg)` on incoming RPC, `propose(data)` for writes, `has_ready()` -> `ready()` -> process messages/snapshots/committed-entries/hard-state -> `advance()` -> `advance_apply()`. User provides `Storage` trait impl (persists log, hard state, snapshots). **No network layer, no state machine framework, no log persistence helpers.**
- **Production users**: TiKV (since 2016), TiDB, multiple downstream forks.
- **Why TiKV stays on it**: TiKV has 8+ years of operational tuning on top of raft-rs (Multi-Raft sharding, raft-engine for log persistence, Joint Consensus, learner-mode rebalancing). Rewriting to openraft would be a multi-year project for them; the inverse calculus applies to us.

### 3. async-raft (archived) -> openraft fork

- **Status**: Archived. openraft is the de-facto successor and was forked from async-raft (with extensive bug fixes and a different `Vote`/safety model).
- **No reason to consider async-raft directly.**

### 4. riteraft 0.1.0

- **Pre-vote**: Inherits raft-rs (built on top of `tikv/raft-rs`). The wrapper exposes a simplified `Store::apply` + mailbox interface; `pre_vote` config is configurable through the underlying raft-rs `Config`.
- **License**: Apache-2.0 / MIT dual.
- **Activity**: **Dormant.** v0.1.0 released 2023-07-26, only 7 commits on `main`, no subsequent releases. Effectively abandoned but the code is small enough to fork.
- **API style**: Framework-ish wrapper around raft-rs. `Store::apply(entry)` is the only required method. Networking is mailbox-based, transport plug-in via a `Mailbox`. **Cuts raft-rs's BYO ceremony by ~80%, at the cost of being a thin layer over an unmaintained wrapper.**
- **Production users**: None we could verify. Used in `rmqtt-raft` (MQTT broker fork).
- **Verdict**: Useful as a reference *if* swapping to raft-rs, but the wrapper itself shouldn't be a dependency — vendor it or rewrite the ~500 lines into `veda-coordinator`.

### 5. raftbare 0.2.2 (sile)

- **Pre-vote**: Not documented. Code inspection would be needed. Crate philosophy is "core algorithm only — convenience left to user", suggesting pre-vote may need application-level scaffolding.
- **License**: MIT.
- **Activity**: Recent. v0.2.2 released 2026-01-11. Maintained by Takeru Ohta (sile, known from Erlang community).
- **API style**: **Extremely low-level, I/O-free.** `Node` generates `Action`s representing pending I/O — the user owns all execution. Smaller surface than even raft-rs. Companion crate `raftpico` provides reference I/O layer.
- **Production users**: None known. The crate is too new for production stories.
- **Verdict**: Promising for greenfield, but a worse swap target than raft-rs because we'd have to write *more* glue (every I/O action becomes our code), not less. Skip.

### 6. little_raft / raft-lite / raft-consensus / rast

- **Pre-vote**: Not exposed in any of them.
- **Activity**: All are educational/single-author projects with <2k lines and infrequent releases.
- **Verdict**: **Not viable for a production vector DB.** These would be net regressions vs. openraft.

### 7. canonical/raft (C library, unmaintained)

- **Pre-vote**: Issue #65 in upstream tracks pre-vote — not implemented.
- **Status**: Upstream archived. `Kerollmops/canonical-raft` Rust wrapper exists but was never released.
- **License**: LGPLv3 (modified for static linking).
- **Verdict**: Linux-only + C FFI + unmaintained + no pre-vote = no.

### 8. hashicorp/raft (Go) via FFI

- **Pre-vote**: Yes (since 2017, via `PreVote` config and the v2 protocol).
- **License**: MPL-2.0.
- **Activity**: Active.
- **Verdict**: Wrapping a Go library via cgo in a Rust process is **not viable** for Veda. cgo has its own runtime (goroutine scheduler, GC), TLS surprises, and we lose async-Rust integration. The Vault / Consul / Nomad fleet uses this, but always as Go-native — no production Rust+Go-FFI raft binding exists for a reason.

### 9. etcd-io/raft (Go)

- **Pre-vote**: Yes (`PreVote: true`).
- **License**: Apache-2.0.
- **Verdict**: Same FFI problem as hashicorp/raft. Skip.

### 10. paxakos / omnipaxos / consensus-rs / BFT engines

- **Algorithm**: Paxos / Multi-Paxos / BFT, not Raft.
- **Verdict**: Algorithmic swap (not just crate swap) — invalidates the "lowest swap cost" criterion. Not in scope.

---

## API Mapping: openraft -> raft-rs

If we *did* swap, here is where the changes would land in `veda-coordinator/`:

| Veda file (lines) | openraft surface used | raft-rs equivalent | Effort |
|---|---|---|---|
| `raft.rs` (64) | `declare_raft_types!`, `VedaTypeConfig`, `NodeId`, `VedaSnapshot` | Delete macro; define `Config`, `Storage` trait impls inline | Low (rewrite, 1 day) |
| `raft_types.rs` (175) | `ClusterCommand`/`ClusterResponse` (already crate-local) | Reusable as-is — these are app types | None |
| `raft_storage.rs` (954) | `RaftStorage`, `RaftLogReader`, `RaftSnapshotBuilder`, `LogState`, `Entry<VedaTypeConfig>`, `Vote`, `LogId`, `StorageError`, `Adaptor::new` | `raft::Storage` trait (single trait, 7 methods); manual snapshot/log-truncation orchestration in our loop | **High** (full rewrite, 1-2 weeks) |
| `raft_network.rs` (233) | `RaftNetwork` + `RaftNetworkFactory` traits, append/install/vote RPC types | No network trait — we write the message dispatcher directly. Tonic transport can be reused for envelope. | Medium (1 week) |
| `raft_tonic.rs` (427) | Tonic services for append/vote/snapshot | Replace proto messages with raft-rs `eraftpb::Message` (or keep ours and translate) | Medium (3-5 days) |
| `raft_wal.rs` (404) | Bespoke WAL behind openraft's storage trait | Reusable, but reshaped to feed raft-rs `Storage` | Low-Medium (2-3 days) |
| `raft_coordinator.rs` (638) | `openraft::Raft<C>::client_write`, `append_entries`, `vote`, `ensure_linearizable`, `RaftMetrics` watch channel | Manual `RawNode` driver loop with `propose()`, `step()`, `tick()`, `has_ready()`/`ready()`/`advance()`; metrics emitted by us | **High** (the core rewrite, 1-2 weeks) |
| `multi_group.rs` (277) | Multiple `Raft<C>` instances keyed by shard | Multiple `RawNode<S>` driven by a shared scheduler — raft-rs has no built-in multi-group; we write it | Medium (1 week) |
| `linearisability_checker.rs` (387) | Drives via the openraft-wrapped API | Adapt to new client surface; logic largely portable | Low (2-3 days) |
| `store.rs` (470) | In-memory store called from state machine | Mostly app-side, minor adaptor changes | Low (1 day) |

**Aggregate effort estimate**: **3-5 engineer-weeks**, dominated by the storage trait rewrite and the `RawNode` driver loop. Plus a re-validation cycle (run the linearisability harness, the existing raft_throughput bench, integration tests under partitions).

The fundamental shape change is: **openraft does the event loop for you; raft-rs makes you write it.** Our `raft_coordinator.rs` becomes substantially longer (the Ready-processing state machine adds ~300-400 lines), but our storage layer becomes thinner (one trait vs. v1/v2 adaptor dance).

A `riteraft`-style thin wrapper could shave ~30% off this estimate but adds a vendored dependency we'd have to maintain ourselves.

---

## Should We Even Swap?

**No, probably not.** Here is the honest cost-benefit.

### What pre-vote actually prevents

The pathology: a node or minority partition (e.g. `{A, B}` of a 5-node cluster) is isolated. Its election timer fires, it increments term, requests votes, fails to win, sleeps, repeats. After network heal, that node rejoins with a high term, forcing the existing leader to step down and triggering a fresh election. The cluster experiences **one election-timeout (~1-2 second) of write unavailability** per heal event.

This is real, but:
1. **It happens only on partition heal**, not steady state.
2. **The unavailability window is bounded** by `election_timeout_max` (which we control).
3. **No data is lost** — the cluster is correct, just briefly unavailable.
4. **It is rare in cloud deployments** where most partitions are short, asymmetric, or whole-datacentre (where the minority *should* lose anyway).

### What openraft 0.9 already mitigates

The `Vote` partial-order design **already prevents the worst version** of this. An isolated candidate cannot freely increment term; it only does so when its current vote is dominated by an incoming response. Combined with `effective_leader` heartbeat tracking, a follower will *refuse* to grant a vote if it's still seeing its leader's heartbeats — which means an isolated minority can't easily bump term without losing heartbeats first.

The leftover failure: a *long* partition where the minority's heartbeats genuinely time out, after which a re-elected minority candidate carries a high term back into the majority. This still triggers one disruptive election on heal.

### What a swap would buy us

Pre-vote eliminates that final disruption window — but in exchange for:
- **3-5 weeks of engineering time** off the roadmap.
- **A less actively-developed dependency** (raft-rs hasn't released since 2023).
- **More code we own** (the Ready loop), which is more code we have to test under Jepsen-style chaos.
- **Loss of features** we currently get for free from openraft 0.9: joint membership, snapshot streaming, async event-driven core, `ensure_linearizable()`, structured metrics via `watch` channel.

### Better alternatives to swapping

1. **Wait for openraft 0.10 stable.** It's on alpha.18 as of 2026-04-20. The roadmap explicitly addresses simplifying the vote/pre-vote path; some 0.10 alphas have already tightened election semantics. A point upgrade (0.9 -> 0.10) is hours-to-days of work vs. weeks for a crate swap.
2. **Enable the `single-term-leader` feature flag.** This switches openraft to standard Raft semantics (one leader per term, log_id is `(term, log_index)` not `(term, node_id, log_index)`), aligning behaviour with what most operational tooling expects. Reduces some classes of election conflict at modest cost to availability under burst-elections.
3. **Add application-level CheckQuorum.** A leader that fails to hear from a quorum in `N` election timeouts voluntarily steps down. Implement as a `tokio::spawn` watching `RaftMetrics` — ~100 lines in `raft_coordinator.rs`. This is what etcd does in addition to pre-vote; here it can be a partial substitute.
4. **Tune election timeouts.** Wider random jitter + higher minimum (e.g. `election_timeout_min = 2000ms, max = 4000ms`) reduces spurious elections in normal jitter without requiring pre-vote at all. Most disruption events come from over-aggressive timeouts.
5. **Pin term-bump rate.** Track per-node term increments; if a node bumps term more than K times within T seconds (clearly partitioned), gate it from rejoining via gateway-level health checks — `peer_health.rs` already has the scaffolding. This is a defence-in-depth measure but cheap.

### When would a swap make sense?

Only if:
- openraft 0.10 is delayed past 2026-Q4 AND
- We observe sustained operational pain from disruptive elections in production (Jepsen-style or real customer reports) AND
- A workaround (CheckQuorum, timeout tuning) provably fails to mitigate it.

None of those three are true today.

---

## Recommendation

1. **Keep openraft 0.9 for now.** It's actively maintained, the swap cost is high, and the residual risk is partial.
2. **Plan a 0.9 -> 0.10 upgrade for Q3 2026** once 0.10 stable lands.
3. **Implement application-level CheckQuorum + term-bump rate limiting** as a 2-3 day defence-in-depth task. Tracked separately from this survey.
4. **Reserve the raft-rs swap as a contingency** if openraft 0.10 stable slips past 2026-Q4 AND production pain materialises. Pre-write a one-page "swap playbook" referencing the file-level mapping in this doc, but do not start the work.
5. **Add a partition-heal scenario to the linearisability harness** (`linearisability_checker.rs`) to empirically measure how disruptive elections look on this cluster, rather than relying on theory.

The point of pre-vote is to make a rare, bounded failure even rarer. The point of engineering effort is to ship a vector database. These don't currently align in favour of swapping.

---

## Source Index

- openraft repo: https://github.com/databendlabs/openraft
- openraft discussion #15 (get rid of pre-vote): https://github.com/databendlabs/openraft/discussions/15
- openraft releases: https://github.com/databendlabs/openraft/releases
- tikv/raft-rs repo: https://github.com/tikv/raft-rs
- raft-rs pre_vote API: https://github.com/tikv/raft-rs/blob/master/src/raft.rs
- riteraft: https://github.com/PsiACE/riteraft
- raftbare: https://lib.rs/crates/raftbare
- async-raft (archived): https://github.com/async-raft/async-raft
- canonical/raft (unmaintained C): https://github.com/canonical/raft
- hashicorp/raft (Go): https://github.com/hashicorp/raft
- Rockraft choice rationale: https://www.codedump.info/en/post/20260412-rockraft/
- TiKV blog: implement raft in rust: https://tikv.org/blog/implement-raft-in-rust/
- openraft docs (Vote design, FAQ): https://databendlabs.github.io/openraft/

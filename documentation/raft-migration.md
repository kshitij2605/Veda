# Raft Consensus Migration Plan

> **PARTIALLY HISTORICAL — superseded in v0.2.0 (Phase C + Phase E hard-remove).**
>
> The original migration goal of "route every record write through a
> single cluster-wide openraft group" is intentionally NOT where Veda
> landed. In v0.2.0 the data plane split:
>
> - `_meta` openraft group — schema, shard plan, nodes, TSO lease,
>   txn commit records, aliases, region topology. Slim by design.
> - Per-shard `veda-log` — record writes. NOT Raft. BookKeeper-
>   style `(E, Qw, Qa)` quorum replication with epoch fencing.
>   Writer-leadership per shard named in `_meta`.
>
> The "current state" table below described the early scaffolding for
> the all-records-through-Raft target. That target was abandoned in
> favour of the FoundationDB-shape layered split when it became clear
> that funnelling every record through one Raft leader caps write
> throughput at single-leader speed and makes the metadata snapshot
> grow linearly with the dataset. See ADR-013 and
> [rearchitecture-overview.md](rearchitecture-overview.md).
>
> What this doc IS still useful for:
>
> - Historical record of the openraft 0.9 integration steps (state
>   machine, log store, snapshot, in-process transport, gateway env
>   switch). All of that scaffolding survived and powers `_meta`
>   today.
> - The "current state" table accurately describes the `_meta` Raft
>   group as it stands now.
>
> What this doc IS NOT useful for:
>
> - The "Phase 4" / "Phase 5" / "Phase 6" sections that talk about
>   routing record writes through Raft. That path was deprecated in
>   `70b4aa0` (Phase E deprecation) and hard-removed in `c8baf29`
>   (Phase E hard-remove). Record writes flow through `veda-log`,
>   not the Raft state machine.
> - `ClusterCommand::InsertRecord` / `TxnWrite` / `BeginTxn` /
>   `CommitTxn` / `AbortTxn` — all five variants are gone.
> - `ClusterStateMachine.records` / `open_txns` — both gone.
>
> ---

This document describes the staged migration from Veda's current single-leader
channel-based coordinator (`veda-coordinator/src/network.rs`) to a fully
distributed Raft consensus layer backed by [openraft](https://github.com/datafuselabs/openraft).

## Current state

| Component | Status |
|---|---|
| `ClusterCommand` / `ClusterResponse` enums | Stable; production interface |
| `ClusterStateMachine` | Deterministic, fully tested |
| `LogStore` | In-memory append-only log |
| `Coordinator` / `CoordinatorHandle` (channel-based) | Working for single-node |
| `ChannelNetwork` | Placeholder, multi-node transport stub |
| `raft` module (openraft scaffolding) | Type config + aliases declared |
| `raft_storage` module | `MemLogState` + `MemStateMachine` data structures, snapshot round-trip tested. `MemRaftStorage` implements v1 `RaftStorage<VedaTypeConfig>` end-to-end; `MemSnapshotBuilder` implements `RaftSnapshotBuilder`. `Adaptor::new(MemRaftStorage::new())` produces the v2 `(RaftLogStorage, RaftStateMachine)` pair `Raft::new` expects. |
| `raft_network` module | ✅ `Arc<InProcessNetwork>` now implements `RaftNetworkFactory<VedaTypeConfig>`; `InProcessConn` implements `RaftNetwork`. All three RPCs (append_entries / install_snapshot / vote) return `Unreachable` for any non-self peer — correct for single-node, recoverable behaviour for any future multi-node attempt over this transport. Tonic cross-process transport still deferred. |
| `raft_coordinator` module (Phase 4) | ✅ `RaftCoordinator` wraps `openraft::Raft<VedaTypeConfig>` with the same `execute / query_state / shutdown` API as the channel coordinator. `start_single_node(node_id)` bootstraps. Two tests pin the end-to-end CreateCollection round-trip. |
| `veda-gateway` env switch (Phase 5) | ✅ `VEDA_COORDINATOR=raft` boots a single-node Raft loop at startup via `AppState::load_raft_from_env()`. `VEDA_NODE_ID` / `VEDA_NODE_ADDRESS` env vars plumbed. |
| `GET /v1/admin/nodes` Raft fan-out (Phase 6 UI side) | ✅ When the gateway runs in raft mode, `list_nodes` reads peers from `ClusterState.nodes` via the coordinator and flips `mode` to `"raft"`. Peer status is `"unknown"` until the Tonic transport reports liveness. |

The async traits and types in `crates/veda-coordinator/src/raft.rs` exist so
the rest of the codebase can refer to a real Raft type and so the migration
is a localised change.

## Why openraft

- Pure Rust, no GC pauses.
- Pluggable storage and network — fits our existing `LogStore` + `ClusterStateMachine`.
- Production-proven (Databend, RisingWave, Datafuse use openraft).
- Async-first; matches our Tokio-based stack.

## Migration phases

### Phase 1 — Compile-time scaffolding (done in commit introducing this doc)

- Add `openraft = "0.9"` dependency.
- Declare `VedaTypeConfig` via `openraft::declare_raft_types!`.
- Type aliases: `NodeId = u64`, `VedaRaft = Raft<VedaTypeConfig>`.
- `VedaSnapshot` placeholder struct (just `last_applied_index` + serialised state).

This is the entrypoint surface. Nothing functionally changes.

### Phase 2 — Storage adapter ✅ (in-memory)

Implemented in `crates/veda-coordinator/src/raft_storage.rs`:

- `MemRaftStorage` composes `MemLogState` + `MemStateMachine` and
  implements v1 `RaftStorage<VedaTypeConfig>` end-to-end (vote, log,
  state-machine apply, snapshot build/install).
- `MemSnapshotBuilder` implements `RaftSnapshotBuilder` against a cloned
  `MemStateMachine` so snapshot building doesn't block the apply path.
- The combined `EntryPayload` dispatch handles `Normal` (forwarded to
  `ClusterStateMachine::apply`), `Membership` (recorded into
  `StoredMembership`), and `Blank` (bumps `last_applied` only).
- Test coverage in `raft_storage::tests`:
  - `mem_raft_storage_vote_log_apply_snapshot_cycle` — drives every v1
    trait method end-to-end.
  - `mem_raft_storage_adaptor_compiles` — proves
    `Adaptor::<VedaTypeConfig, MemRaftStorage>::new(...)` produces the v2
    `(RaftLogStorage, RaftStateMachine)` pair `Raft::new` requires.

What is **not** yet covered (deliberately left for follow-ups):

- Durable backing. Vote, log, and state machine all live in `Arc<RwLock>`
  over in-process structures. Plugging in a file-backed log + fsync'd
  vote is the next focused piece.
- Snapshot streaming via `begin_receiving_snapshot` returns an empty
  cursor; a chunked-disk variant is a future concern.

#### Why the v1 adapter path

openraft 0.9 split the legacy `RaftStorage` into `RaftLogStorage` +
`RaftStateMachine` v2 traits, but both are sealed under the
`storage-v2` feature. Without that feature, the supported path is to
implement v1 `RaftStorage` and wrap with
`openraft::storage::Adaptor<C, S>`. Our `MemRaftStorage` is exactly that
implementation; the Adaptor round-trip test pins the contract.

### Phase 3 — Network transport (in-process done; Tonic deferred)

In-process `RaftNetwork` / `RaftNetworkFactory` traits are now
implemented on `Arc<InProcessNetwork>` (commit `d4aebff`). All three
trait methods return `Unreachable` when asked to talk to a non-self
peer — correct semantics for single-node Raft (the leader never RPCs
itself) and recoverable for any future multi-node attempt over the
in-process transport (openraft marks the peer dead and backs off).

The remaining work for true multi-node Raft replication:

- gRPC service exposing the openraft RPCs (`AppendEntries`, `InstallSnapshot`,
  `Vote`).
- Use Tonic since we already use it for OTel.
- `RaftNetworkFactory::new_client(target)` resolves `NodeId` → `address`
  from the existing `ClusterState::nodes` map.

For testing, retain `ChannelNetwork` as the in-process transport so unit
tests don't need real TCP sockets.

### Phase 4 — Coordinator replacement ✅

Landed in commit `d4aebff`:

- `RaftCoordinator` wraps `Arc<openraft::Raft<VedaTypeConfig>>` and
  exposes the same `execute / query_state / shutdown` API as
  `CoordinatorHandle`.
- `execute` calls `raft.client_write(command)` and returns the
  applied `ClusterResponse`.
- `query_state` is a local read off `MemRaftStorage.sm.snapshot()`
  — follower-safe, no Raft round-trip. Linearisable reads would
  need a no-op `client_write` first; that is a follow-up if a
  caller needs them.
- The existing channel-based `Coordinator` stays the default. The
  switch lives in `AppState::load_raft_from_env()`:
  `VEDA_COORDINATOR=raft` boots the Raft path with
  `VEDA_NODE_ID` (default 1).

### Phase 5 — Cluster bootstrap (single-node done; multi-node deferred)

Single-node bootstrap done in commit `d4aebff`:
`RaftCoordinator::start_single_node(node_id)` constructs the type
config + storage + network, builds the Raft handle, and calls
`raft.initialize({node_id => BasicNode::default()})`. After this
returns, the node is leader.

Multi-node bootstrap (still TODO):

- `veda-server` startup: read `VEDA_RAFT_PEERS` (comma-separated
  `id@host:port`).
- On first launch with `--bootstrap`, run `raft.initialize(...)` with the
  full peer set.
- On subsequent launches, just `raft.add_voter(...)` is unnecessary —
  state is already in the durable log.

Multi-node bootstrap is blocked on the Tonic transport landing,
since `raft.initialize` with N > 1 voters immediately triggers
RPCs the in-process transport cannot deliver.

### Phase 6 — Membership changes & observability (UI side done; rest TODO)

`GET /v1/admin/nodes` now reads from `ClusterState.nodes` via the
Raft coordinator when one is configured, and flips `mode` to
`"raft"` in the response (commit `d4aebff`). The Niriksha
"Cluster Nodes" table needs zero changes — it already consumes
this shape.

Remaining:

- Expose `POST /v1/admin/cluster/join` and `DELETE
  /v1/admin/cluster/nodes/{id}` in `veda-gateway`. Backed by
  `raft.add_voter(...)` / `raft.change_membership(...)`.
- Add Prometheus metrics for `raft_term`, `raft_leader_id`,
  `raft_last_applied`, `raft_log_lag`. Most of these are
  available on `Raft::metrics()` already; the wiring is a
  background task that reads the metrics stream and pushes to
  the existing Prometheus recorder.
- Tracing spans on each Raft RPC (Tonic-layer concern; folds
  into Phase 3 completion).

### Phase 7 — Production hardening

- Pre-vote (`feature = "pre-vote"` in openraft) to avoid disruptive
  elections.
- Configurable snapshot interval.
- Backpressure on log growth.
- Failover testing under network partitions (use `tokio::time::pause` for
  determinism).

## Estimated effort

| Phase | Estimate | Status |
|---|---|---|
| 1 (scaffolding) | 1 hour | ✅ done |
| 2 (storage) | 1-2 days | ✅ in-memory + durable file-backed (commit `6ef5d46`) |
| 3 (network) | 2-3 days | ✅ Tonic gRPC transport (commit `7374a3e`) |
| 4 (coordinator) | 1 day | ✅ done |
| 5 (bootstrap) | 0.5 day | ✅ single-node + multi-node (`VEDA_RAFT_PEERS`) |
| 6 (membership + observability) | 1 day | ✅ POST/DELETE /v1/admin/cluster/* + Raft Prom gauges + UI fan-out |
| 7 (production hardening) | 1-2 weeks | ⚠ partial — see "remaining" below |

**Original estimate: ~3-4 weeks of focused engineering**, dominated
by phases 2-3 and production hardening in phase 7.

**Status today** (after commit `a7ca788`): the migration is
substantially complete. A 3-node Raft cluster boots,
elects leaders, fails over on node loss, recovers from a
wiped-clean follower (via WAL replay + Tonic
AppendEntries), and serves linearisable reads through a
no-op barrier. Each of the six hardening items the previous
revision of this doc called out has shipped except pre-vote
(genuinely unavailable in openraft 0.9):

- ✅ **Cross-node health probes** (`peer_health.rs`,
  commit `9095454`). Each node GETs every peer's
  `/admin/health` every 5s; the dashboard's Cluster Nodes
  table reports `healthy / degraded / unreachable` instead
  of `unknown`.
- ✅ **Per-entry WAL** (`raft_wal.rs`, commit `0641a1f`).
  Append-only `raft_log.wal` with crc32-framed records,
  torn-write recovery, and a one-shot migration from the
  legacy `raft_log.bin` snapshot file.
- ✅ **Snapshot soak smoke** (`scripts/raft-snapshot-soak.sh`,
  commit `a7ca788`). Drives 200 Noop entries through the
  log, wipes a follower's data dir, restarts it, asserts
  catch-up to the leader's high-water mark within 60s.
- ✅ **Partition smoke** (`scripts/raft-partition-smoke.sh`,
  commit `79c1e95`). Kills the leader, asserts a new one is
  elected from the surviving pair, restarts the killed
  node, asserts convergence. Also caught + fixed a Tonic
  cached-channel-reuse bug in the same commit.
- ✅ **Linearisable reads** (commit `9095454`).
  `RaftCoordinator::linearisable_query_state()` issues a
  `Noop` barrier write before reading; the test asserts
  `last_applied_index` advances through the barrier.
- ❌ **Pre-vote** — *available only via consensus-library swap.*
  Investigation v3 confirms:
    - openraft 0.9.24 source + changelog + roadmap: zero
      pre-vote support. Upstream has explicitly decided to
      "get rid of pre-vote RPC" in favour of the
      `single-term-leader` feature.
    - tikv/raft-rs 0.7.0: `Config::pre_vote: bool` exists
      and is documented as "Enables the Pre-Vote algorithm
      described in raft thesis section 9.6. This prevents
      disruption when a node that has been partitioned away
      rejoins the cluster."
  So pre-vote IS available in the Rust ecosystem — it just
  is not available in the consensus library Veda has built
  around. Getting it requires swapping openraft for raft-rs.
  Cost estimate:
    - raft-rs is lower-level than openraft (caller brings
      Log + State Machine + Transport — openraft provides
      these as trait implementations).
    - Our MemRaftStorage / TonicRaftNetwork / WAL would
      need to be rewritten against raft-rs traits.
    - raft-rs has not released since 2023; not necessarily
      abandoned, but slower-moving than openraft.
    - Realistic effort: 2–4 weeks of focused engineering.
  Verdict: this is a separate project, not a session-scope
  item. Filed as "deferred to a consensus-crate swap
  engagement"; in the meantime, the failure mode pre-vote
  prevents (partitioned node disrupting the cluster on
  rejoin) is partially mitigated by the Tonic
  channel-eviction-on-error fix from commit 79c1e95 — when
  the disrupted node rejoins, its outbound channels
  reconnect cleanly and openraft's existing term-comparison
  logic handles the rejoin via AppendEntries rather than a
  forced re-election.

What was added in the third follow-up pass (commits `d544634`,
`bf16879`):

- **Cross-node linearisable reads** (`?linearisable=true`):
  **SUPERSEDED — the flag is refused with 501 as of 2026-09-11.**
  It issued a Noop barrier and then read from
  `ClusterState.records`. ADR-013 Phase E removed that map
  (`c8baf29`); record writes go through per-shard `veda-log`.
  The flag outlived the mechanism and degraded to "pay a Raft
  round-trip, then do the ordinary local read" — no additional
  guarantee.

  The "7,065 invocations, 0 violations" below was measured
  BEFORE Phase E and does not describe current behaviour.
  Re-measured 2026-09-11 on a live 3-node cluster with the
  barrier enabled: **8,846 stale_read violations across 26,857
  invocations**, and a direct probe (insert on node-1, read on
  node-2 with the flag) returned 404 — identical to the plain
  read, while the writer node returned 200.

  Genuine cross-node linearisable reads would need a read-index
  against the per-shard log. Not built.

- **MultiGroupCoordinator (removed 2026-05-27)**: this was a
  coordinator that hosted N openraft instances per process keyed
  by `GroupId`, intended to carry per-(collection, shard)
  data-plane Raft groups. The FoundationDB-shape rearchitecture
  routed record writes through per-shard `veda-log` instead
  (Phase C), and Phase E removed the `InsertRecord` command those
  groups carried, leaving them vestigial. The struct and its
  `crates/veda-coordinator/src/multi_group.rs` file were deleted;
  the per-shard data plane is now `veda-log::ShardLogRegistry`.
  See `documentation/2026-05-27-applier-cursor-shardlog-registry-design.md`.

- **24-hour soak running**: `scripts/raft-soak-24h.sh` is
  in flight against the 3-node compose cluster on iitgpu11
  as of commit `d544634`. PID + monitoring command paths
  are in `/tmp/veda-soak-24h-wrapper.log`. Result tally
  available via `grep -c 'iter PASS' /tmp/veda-soak-24h/soak.log`.

What was added in the second follow-up pass (commits `98a09a3`,
linearisability checker, soak harness wrapper):

- **Inserts routed through Raft**: `POST
  /v1/collections/{name}/insert` now submits
  `ClusterCommand::InsertRecord` per record. On apply each
  node materialises the record via its SegmentManager
  through the existing reconcile path. State machine carries
  every record so chunked snapshots have substance. Wire
  trade-off explicitly documented inline:
  single-cluster-Raft scaling cap. Verified end-to-end:
  inserts on the leader appear with byte-identical UUIDs
  on all three follower nodes within the reconcile window.

- **Linearisability history checker + harness**:
  `crates/veda-coordinator/src/linearisability_checker.rs`
  records every operation's `(start_ns, end_ns, outcome)`
  and offline-checks every Read against the set of Inserts
  that could have completed by the Read's start. Three
  violation classes detected: `stale_read` (Read returned
  None after a confirmed Insert), `payload_mismatch` (Read
  saw a value no Insert produced), `ghost_read` (Read saw
  a value with no overlapping Insert). Four unit tests.
  Companion harness binary `veda-linearisability-harness`
  drives concurrent inserts + reads against the cluster
  and prints a JSONL history + a check report.

  **Honest findings from running it**:
    - Single-node (all ops against the leader): 3,524 ops,
      1,760 reads, **0 violations**. Confirms the
      Raft-replicated record store is linearisable WHEN
      every client lands on the node that processed its
      latest insert.
    - Cross-node (ops randomly distributed across 3 nodes):
      ~2,290 stale_read violations across ~29,000 ops in
      a 30s run. These are NOT correctness bugs — they
      are the visible 2s reconcile-interval window
      between insert-on-leader and follower-side
      SegmentManager materialisation. The system is
      eventually consistent across nodes; making it
      linearisable across nodes needs either a
      barrier-on-read (see linearisable_query_state
      pattern) or replacing follower reads with
      consultations of the Raft state machine. Documented
      as a follow-up.

- **24-hour soak workflow**: `scripts/raft-soak-24h.sh`
  wraps the existing fault-injection harness with the
  nohup-detach + structured-log pattern used by an operator
  for hours-long runs. Pre-flight check picks up the
  cluster's leader, validates every node responds, writes
  a header (git commit, parameters, host) into LOG_DIR,
  detaches the soak. PID + log paths printed for monitoring.

What was added in the first follow-up pass (commits `4a3408e`,
`967f5c0`, `26c4eda`, `126a8ac`):

- **WAL compaction**: `MemLogState::compact()` + a periodic
  background task (`VEDA_RAFT_WAL_COMPACT_INTERVAL_SECS`,
  `VEDA_RAFT_WAL_COMPACT_THRESHOLD_BYTES`). The append-only
  WAL now rewrites itself when it grows past 16 MB (default).
  Round-trip test in `raft_wal::tests`.
- **`tc qdisc` fault-injection rig**: `cap_add: [NET_ADMIN]`
  on every raft-cluster container, `iproute2` shipped in
  the runtime image, and `scripts/raft-fault-inject.sh`
  exposing loss / delay / asymmetric-partition / smoke. Real
  10% packet-loss smoke passes.
- **`CreateCollection` routed through Raft**:
  `POST /v1/collections` (and `DELETE`) now submit
  `ClusterCommand::CreateCollection { schema_json }` to the
  log; every node materialises locally on apply. A periodic
  reconcile task (every 2s by default) catches followers up
  after restart. Verified end-to-end: collection created on
  leader, visible on all 3 nodes within the reconcile window.
- **Sustained-load soak harness**: `scripts/raft-soak.sh`
  runs continuous Noops from 3 writer processes while
  randomly applying one of {kill, restart, loss, latency,
  partition} per iteration. 2-minute demo run with
  ITERATION_SECS=30 passed 4/4 iterations with writes
  flowing throughout (last_applied climbed 194 -> 3041).
  Designed for the operator to run for hours.

What is genuinely still deferred (production engineering
budget, not session-scope):

- **Hours-long soak runs.** The harness exists; the run
  itself is the operator's decision. A 24-hour soak with
  this harness is the Jepsen-style operator-confidence
  baseline.
- **Jepsen-class linearisability checker.** The soak
  harness asserts "all nodes agree on last_applied after
  each fault clears" — catches gross divergence, not
  subtle linearisability violations on individual writes.
  Adding a real client-history checker is a separate
  project.
- **Pre-vote.** Confirmed upstream openraft has decided
  not to implement it; achieving the semantics in Veda
  would require swapping consensus crates or layering
  above openraft's public API. Both out of scope.
- **Per-shard data routing through Raft.** Inserts
  (`POST /v1/collections/{name}/insert`) still hit one
  node's local segment manager directly. Routing inserts
  through Raft is the natural next step but is a bigger
  refactor (record IDs, ownership, etc.).
- **Follower-to-leader request forwarding.** A `POST`
  to a follower currently submits to its local raft handle
  which forwards via openraft; ideally we'd 307-redirect to
  the current leader's REST address so clients see the
  fast path directly.

A team can deploy multi-node Veda today and let
`scripts/raft-soak.sh` run continuously against a
production-shape cluster to build operational confidence
before launch.

## Why not openraft today

Each `RaftLogStorage` method requires careful semantics around log
truncation and persistent vote state. Getting these wrong silently corrupts
the cluster. This is the kind of work that benefits from dedicated time and
multiple rounds of testing — not what a quick session should attempt.

The scaffolding is in place so that work can start immediately when the
team chooses to allocate time.

## Concrete next-pass checklist (openraft 0.9)

When the implementation work starts, these are the trait method
signatures that need to land. Each one already has a Veda-internal
counterpart in `store.rs` / `raft_storage.rs` — the impl is mostly
adapter code mapping types back and forth, plus careful attention to
the persistence ordering rules openraft documents in its README.

### `impl RaftLogStorage<VedaTypeConfig> for MemRaftLogStorage`

```rust
type LogReader = Self; // shareable read view

async fn get_log_state(&mut self) -> Result<LogState<VedaTypeConfig>, StorageError<NodeId>>;
async fn read_vote(&mut self) -> Result<Option<Vote<NodeId>>, StorageError<NodeId>>;
async fn save_vote(&mut self, vote: &Vote<NodeId>) -> Result<(), StorageError<NodeId>>;

async fn append<I>(&mut self, entries: I, callback: LogFlushed<VedaTypeConfig>)
    -> Result<(), StorageError<NodeId>>
where
    I: IntoIterator<Item = Entry<VedaTypeConfig>> + Send;

async fn truncate(&mut self, log_id: LogId<NodeId>)
    -> Result<(), StorageError<NodeId>>;
async fn purge(&mut self, log_id: LogId<NodeId>)
    -> Result<(), StorageError<NodeId>>;

async fn get_log_reader(&mut self) -> Self::LogReader;
```

### `impl RaftLogReader<VedaTypeConfig> for MemRaftLogStorage`

```rust
async fn try_get_log_entries<RB: RangeBounds<u64> + Clone + Debug + Send>(
    &mut self,
    range: RB,
) -> Result<Vec<Entry<VedaTypeConfig>>, StorageError<NodeId>>;
```

### `impl RaftStateMachine<VedaTypeConfig> for MemRaftStateMachine`

```rust
type SnapshotBuilder = SnapshotBuilderType;

async fn applied_state(&mut self) -> Result<(
    Option<LogId<NodeId>>,
    StoredMembership<NodeId, BasicNode>,
), StorageError<NodeId>>;

async fn apply<I>(&mut self, entries: I) -> Result<Vec<ClusterResponse>, StorageError<NodeId>>
where I: IntoIterator<Item = Entry<VedaTypeConfig>> + Send;

async fn get_snapshot_builder(&mut self) -> Self::SnapshotBuilder;
async fn begin_receiving_snapshot(&mut self) -> Result<Box<Cursor<Vec<u8>>>, StorageError<NodeId>>;

async fn install_snapshot(
    &mut self,
    meta: &SnapshotMeta<NodeId, BasicNode>,
    snapshot: Box<Cursor<Vec<u8>>>,
) -> Result<(), StorageError<NodeId>>;

async fn get_current_snapshot(&mut self)
    -> Result<Option<Snapshot<VedaTypeConfig>>, StorageError<NodeId>>;
```

### `impl RaftNetworkFactory<VedaTypeConfig> for ChannelNetwork`

```rust
type Network = ChannelNetworkConn;
async fn new_client(&mut self, target: NodeId, node: &BasicNode) -> Self::Network;
```

### `impl RaftNetwork<VedaTypeConfig> for ChannelNetworkConn`

```rust
async fn send_append_entries(...)    -> Result<AppendEntriesResponse<NodeId>, ...>;
async fn send_install_snapshot(...)  -> Result<InstallSnapshotResponse<NodeId>, ...>;
async fn send_vote(...)              -> Result<VoteResponse<NodeId>, ...>;
```

### Persistence ordering (openraft README requirement)

1. `save_vote` returns only after the vote is fsync'd.
2. `append` returns only after entries are durable AND the `LogFlushed`
   callback has been invoked.
3. `purge(log_id)` may not remove entries above the latest applied
   log id; openraft will only call it with safe values, but defence-in-
   depth in the impl is wise.
4. `install_snapshot` is followed by a purge — the storage must accept
   that and clear log entries below the snapshot.

Get these four invariants right and the rest is mechanical mapping.

### What lands here (now)

This commit adds the trait-method checklist above plus a snapshot
round-trip integration test
(`crates/veda-coordinator/tests/snapshot_roundtrip_test.rs`) that
proves the in-memory log + state machine pieces compose into a
consistent snapshot before any openraft trait surface is in play.
The implementer can build on top of that contract.

## Linearisability harness — Elle cross-check

The native `linearisability_harness` binary records every insert /
read against the cluster and runs Veda's own sequential checker
(see `linearisability_checker::check_history`). To get an independent
read with a different algorithm, dump the history in Elle format
and feed it to `elle-cli` — Kingsbury's cycle-detection consistency
checker.

The harness now emits two files:

```
$ VEDA_LIN_DURATION_SECS=120 cargo run --release \
    -p veda-coordinator --bin veda-linearisability-harness

[lin] history has 4382 invocations
[lin] history dumped to linearisability-history.jsonl
[lin] elle-format history dumped to linearisability-history.jsonl.elle.jsonl
[lin] report:
  ...
[lin] PASS
```

### Run elle-cli against the Elle dump

`elle-cli` is published as a JAR + Docker image; running via Docker is
the lowest-friction path:

```
docker run --rm -v "$PWD:/w" -w /w jepsen/elle-cli \
  lein run check --model rw-register \
    --consistency-models linearizable \
    linearisability-history.jsonl.elle.jsonl
```

A passing run prints `{:valid? true}`. A failing run prints one or
more anomaly classes (`:G0`, `:G1a`, `:G1b`, `:G2-item`,
`:lost-update`, etc.) along with the implicated operations.

### Why both checkers

The native check_history is a simple "every read observes a
linearisable insert" sequential model — fast, in-process, runs in
unit tests. elle-cli is a graph-based cycle detector — slower,
independent algorithm, catches anomaly classes the simple model
doesn't enumerate (e.g. read-only G2 cycles).

When the two agree we have high confidence. When they disagree the
delta names exactly what's missing from the simple model. The Veda
checker is **not** intended to replace Elle — it's the smoke test
that runs every soak iteration; Elle is the offline weapon for
hunting subtle anomalies.

### Format we emit

Each Veda Invocation becomes two events (`:invoke` then `:ok` /
`:fail` / `:info`), keyed by `process` = client id. The mop shape
is single-key rw-register:

```
{"index":0,"time":1700000000000,"process":0,"type":"invoke","f":"write","value":[[":w","lin_smoke/abc","payload-7"]]}
{"index":1,"time":1700000000812,"process":0,"type":"ok","f":"write","value":[[":w","lin_smoke/abc","payload-7"]]}
{"index":2,"time":1700000000900,"process":1,"type":"invoke","f":"read","value":[[":r","lin_smoke/abc",null]]}
{"index":3,"time":1700000001050,"process":1,"type":"ok","f":"read","value":[[":r","lin_smoke/abc","payload-7"]]}
```

Same JSONL is consumable by Jepsen's Clojure tooling.

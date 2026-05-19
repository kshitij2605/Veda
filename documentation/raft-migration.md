# Raft Consensus Migration Plan

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
| 2 (storage, in-memory) | 1-2 days | ✅ done |
| 3 (network, in-process) | 2-3 days | ✅ in-process done; Tonic deferred |
| 4 (coordinator) | 1 day | ✅ done |
| 5 (single-node bootstrap) | 0.5 day | ✅ done |
| 6 (membership + observability) | 1 day | ⚠ UI fan-out done; join/leave + Prom metrics deferred |
| 7 (production hardening) | 1-2 weeks | ❌ deferred (Tonic, durable log, snapshot streaming, pre-vote, partition tests) |

**Original estimate: ~3-4 weeks of focused engineering**, dominated
by phases 2-3 and production hardening in phase 7.

**Status today** (after commit `d4aebff`): single-node openraft is
running end-to-end. Phases 1, 2, 4, 5 (single-node) are complete;
Phase 3 (in-process portion) and Phase 6 (UI fan-out) are
complete. The remaining work is the multi-node hardening pass:

- **Tonic gRPC transport** (~2-3 days). Replace the in-process
  `RaftNetwork` impl with a real cross-process one. Service the
  three openraft RPCs (`AppendEntries`, `InstallSnapshot`, `Vote`)
  + the chunked-snapshot path.
- **Durable storage** (~3-5 days). The current `MemRaftStorage`
  loses cluster state on restart. A file-backed log + fsync'd vote
  is the minimum honest durability story. Snapshot streaming on
  top of that.
- **Multi-node bootstrap** (~0.5 day after transport lands).
  `VEDA_RAFT_PEERS=1@host1,2@host2,3@host3` + a `--bootstrap` flag
  that calls `raft.initialize(...)` on first boot.
- **Membership API** (~0.5 day). `POST /v1/admin/cluster/join` +
  `DELETE /v1/admin/cluster/nodes/{id}`.
- **Observability** (~0.5 day). Raft Prometheus metrics scraping
  `Raft::metrics()` stream.
- **Production hardening** (1-2 weeks, mostly testing). Pre-vote,
  partition tests, snapshot soak, leader transfer.

A team can use the current single-node path in production
deployments where consensus matters less than "does it boot
cleanly under VEDA_COORDINATOR=raft" — but should not deploy
multi-node until the durable-storage + Tonic items land.

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

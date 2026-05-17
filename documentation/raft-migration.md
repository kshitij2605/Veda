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
| `raft_storage` module | `MemLogState` + `MemStateMachine` data structures, snapshot round-trip tested. `RaftLogStorage` / `RaftStateMachine` *trait* impls deferred — see "Phase 2 friction" below. |
| `raft_network` module | `NodeAddressBook` + `InProcessNetwork` channel-based transport. `RaftNetwork` trait impl deferred. |

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

### Phase 2 — Storage adapter

Implement `openraft::RaftLogStorage` and `openraft::RaftStateMachine` over
our existing types:

- `LogStore` → satisfies log read/append/truncate/purge methods.
- `ClusterStateMachine` → satisfies state-machine `apply` + snapshot build/install.
- Add persistent vote storage (currently in-memory; needs durable file or
  the WAL).
- Implement snapshot creation: serialise `ClusterStateMachine` via bincode.

#### Phase 2 friction (discovered in commits adding `raft_storage.rs`)

openraft 0.9 split the legacy `RaftStorage` trait into `RaftLogStorage` +
`RaftStateMachine`. **Both are sealed** — only implementable via the
`openraft::storage::Adaptor<C, S>` wrapper around the v1 `RaftStorage`
trait. So the actual path is:

1. Implement v1 `RaftStorage<VedaTypeConfig>` (12+ methods, single combined trait).
2. Wrap with `Adaptor::new(your_storage)` to get the v2 traits.
3. Pass to `openraft::Raft::new(...)`.

The structural state needed for the impl already lives in
`MemLogState` + `MemStateMachine`. The remaining work is purely
trait-method-to-plain-method translation, plus careful review against
the `examples/raft-kv-memstore` reference. Estimated 1-2 focused days.

### Phase 3 — Network transport

Replace `ChannelNetwork` with a real `RaftNetwork` implementation:

- gRPC service exposing the openraft RPCs (`AppendEntries`, `InstallSnapshot`,
  `Vote`).
- Use Tonic since we already use it for OTel.
- `RaftNetworkFactory::new_client(target)` resolves `NodeId` → `address`
  from the existing `ClusterState::nodes` map.

For testing, retain `ChannelNetwork` as the in-process transport so unit
tests don't need real TCP sockets.

### Phase 4 — Coordinator replacement

Add a `RaftCoordinator` that wraps `openraft::Raft<VedaTypeConfig>` and
exposes the same `CoordinatorHandle` API:

- `execute(ClusterCommand) -> Result<ClusterResponse>` → submits to the
  Raft client.
- `query_state() -> Result<ClusterState>` → reads from the local state
  machine snapshot (read-only, no Raft round-trip required for follower-safe
  reads).

The existing channel-based `Coordinator` stays as the single-node default;
selection via env var `VEDA_COORDINATOR=raft` switches to the Raft path.

### Phase 5 — Cluster bootstrap

- `veda-server` startup: read `VEDA_RAFT_PEERS` (comma-separated
  `id@host:port`).
- On first launch with `--bootstrap`, run `raft.initialize(...)` with the
  full peer set.
- On subsequent launches, just `raft.add_voter(...)` is unnecessary —
  state is already in the durable log.

### Phase 6 — Membership changes & observability

- Expose `POST /v1/admin/cluster/join` and `DELETE
  /v1/admin/cluster/nodes/{id}` in `veda-gateway`.
- Add Prometheus metrics for `raft_term`, `raft_leader_id`,
  `raft_last_applied`, `raft_log_lag`.
- Tracing spans on each Raft RPC.

### Phase 7 — Production hardening

- Pre-vote (`feature = "pre-vote"` in openraft) to avoid disruptive
  elections.
- Configurable snapshot interval.
- Backpressure on log growth.
- Failover testing under network partitions (use `tokio::time::pause` for
  determinism).

## Estimated effort

| Phase | Estimate |
|---|---|
| 1 (done) | 1 hour |
| 2 (storage) | 1-2 days |
| 3 (network) | 2-3 days |
| 4 (coordinator) | 1 day |
| 5 (bootstrap) | 0.5 day |
| 6 (membership + observability) | 1 day |
| 7 (hardening) | 1-2 weeks (mostly testing) |

**Total: ~3-4 weeks of focused engineering**, dominated by phases 2-3 and
production hardening in phase 7.

## Why not openraft today

Each `RaftLogStorage` method requires careful semantics around log
truncation and persistent vote state. Getting these wrong silently corrupts
the cluster. This is the kind of work that benefits from dedicated time and
multiple rounds of testing — not what a quick session should attempt.

The scaffolding is in place so that work can start immediately when the
team chooses to allocate time.

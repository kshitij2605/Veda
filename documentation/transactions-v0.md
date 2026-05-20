# Transactions v0 — design + roadmap

> **HISTORICAL — superseded in v0.2.0 (Phases E hard-remove + G).**
>
> The v0 single-Raft txn stub described below was removed entirely in
> commit `c8baf29` (Phase E hard-remove). All five `ClusterCommand`
> variants (`BeginTxn`, `TxnWrite`, `CommitTxn`, `AbortTxn`,
> `InsertRecord`) and the `open_txns: HashMap<String, PendingTxn>`
> field on `ClusterStateMachine` are gone.
>
> The production txn substrate is now Percolator-style snapshot
> isolation, delivered by `crates/veda-coordinator/src/percolator.rs::
> TxnCoordinator`. Two-phase commit: prewrite places primary +
> secondary locks on per-shard logs at `start_ts`, commit writes a
> single `RecordTxnCommit` row to `_meta::txn_commits` at `commit_ts`,
> async cleanup of secondary lock markers. Orphan locks resolved by a
> background task (`spawn_orphan_resolver`) that checks the primary
> in `_meta`.
>
> The REST surface (`/v1/txn` begin / commit / abort, `X-Veda-Txn-Id`
> header on inserts) is unchanged in shape; only the implementation
> moved from "stage on every node via Raft" to "Percolator locks on
> the per-shard log + one commit row in `_meta`".
>
> See [ADR-016](ADR.md#adr-016-percolator-style-snapshot-isolation-transactions)
> and [rearchitecture-overview.md](rearchitecture-overview.md) (Phase G).
>
> Kept for historical reference and to document the "what v0 deliberately
> did NOT do" list that drove the Phase G design.
>
> ---

This document captures what shipped in the v0 transaction
scaffolding and — much more importantly — what's deliberately
missing relative to a production system. Anyone reading this
should leave with a clear sense that v0 is the bare API surface,
not a Percolator implementation.

## What v0 ships

Three new `ClusterCommand` variants drive a strict stage-and-flip
lifecycle:

```
BeginTxn { txn_id } ──► TxnWrite { txn_id, ... } * ──► CommitTxn / AbortTxn
```

- `BeginTxn` opens a holder in `ClusterStateMachine::open_txns`.
  Idempotent — retrying with the same id is a no-op.
- `TxnWrite` stages one record into the holder. Rejected if the
  txn isn't open or the collection doesn't exist.
- `CommitTxn` moves every staged write into the main `records`
  map and drops the holder. Last-writer-wins on collisions.
- `AbortTxn` drops the holder. No side effects.

Every command goes through Raft, so all nodes converge to the same
post-commit state. The new variants are at the end of the enum
(no positional reuse) so existing bincode snapshots continue to
load.

### REST surface

| Method + path | Behaviour |
|---|---|
| `POST /v1/txn` | Mint a fresh `txn_id` (uuid v4), emit `BeginTxn` |
| `POST /v1/txn/{txn_id}/commit` | Emit `CommitTxn` |
| `POST /v1/txn/{txn_id}/abort` | Emit `AbortTxn` |
| `POST /v1/collections/{name}/insert` with header `X-Veda-Txn-Id: <id>` | Emits `TxnWrite` instead of `InsertRecord` for every record in the body |

All four require Raft mode (`VEDA_COORDINATOR=raft`). The handler
returns `503 Service Unavailable` otherwise — v0 has no
single-node fallback because the whole point of transactions is
durable + replicated staging.

### Test coverage

`crates/veda-coordinator/src/store.rs::tests`:

- `txn_commit_flips_staged_writes_into_records` — happy path
- `txn_abort_drops_staged_writes`
- `txn_write_against_unknown_txn_is_error`
- `txn_begin_is_idempotent`
- `last_writer_wins_on_concurrent_commit_to_same_key` — pins the v0
  conflict behaviour explicitly so anyone tightening it later sees
  this test fail.

## What v0 deliberately does NOT do

This is the **honest** list. Anyone considering using transactions in
production needs to read it.

| Missing | Why it matters | Roadmap |
|---|---|---|
| **Concurrency control** | Two concurrent transactions writing the same key both succeed; the last commit wins. No conflict detection, no abort-on-conflict. | Adopt Percolator-style optimistic locks (lock rows in a dedicated CF) — see TiKV's `txn_types::Lock`. |
| **MVCC / snapshot isolation** | Reads bypass the txn overlay entirely. Read-your-writes within a txn is NOT supported in v0. | Vendor TiKV's `mvcc::MvccTxn`, add a timestamp source (TSO or HLC), gate reads through the snapshot at txn start. |
| **Cross-shard coordination** | Every staged write goes through the cluster-wide group, so per-shard Raft groups don't yet participate. Single-shard transactions only. | Multi-group prepare/commit with a coordinator handles cross-shard; same architecture TiKV ships. |
| **Deadlock detection / retry / timeouts** | A txn that's opened but never committed/aborted holds resources forever. No automated cleanup. | Heartbeat + lease on txns; coordinator detects stale txns and aborts them. |
| **Conflict resolution** | Lost updates are silent. | Tied to MVCC + concurrency control above. |
| **Large transactions** | Every staged write sits in memory on every node until commit / abort. | Stream staged writes to a dedicated CF on disk; same TiKV pattern. |
| **Read isolation levels** | RC / RR / Serializable are all "read latest committed records" today. | MVCC again — once we have snapshot reads, the level is a knob. |

## Why this is "v0" rather than "Phase 0"

Phase numbering implies incremental progress on a known plan. v0 is
*scaffolding* — a working API surface and the wiring that makes the
ergonomic and operational changes (REST endpoints, Raft commands,
state-machine accounting) measurable before we choose how to fill in
the semantics. Replacing v0 with TiKV-vendored Percolator is a
mostly-orthogonal effort: the API stays, only the state-machine
internals get re-implemented against `txn_types::Mutation` /
`txn_types::Lock` / a real timestamp oracle.

If you want production transactions today, this isn't it.
research/2026-05-19-multi-week-survey/01-cross-shard-transactions.md
has the honest 28–43 engineer-week estimate for full Percolator
integration on top of Veda's existing storage layer.

## What lands here (now)

Commit `feat(txn): v0 transaction scaffolding`:

- `ClusterCommand::{BeginTxn, TxnWrite, CommitTxn, AbortTxn}`
- `ClusterStateMachine::open_txns` + `PendingTxn` holder
- 3 new REST routes (`POST /v1/txn`, `/v1/txn/{id}/commit`, `/v1/txn/{id}/abort`)
- `X-Veda-Txn-Id` header threading through insert
- 5 unit tests pinning the lifecycle invariants
- This design document

What ships is the API + a documented gap to Percolator.

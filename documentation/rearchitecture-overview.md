# Rearchitecture — FoundationDB-shape data plane

This is the engineer-onboarding companion to
`.claude/plans/structured-mixing-turing.md`. The plan file holds
the full multi-phase schedule and decision points; this doc
summarises what's true *today* and where each new piece slots in.

## Why we're rearchitecting

Today every record write funnels through one cluster-wide openraft
group. `ClusterStateMachine.records` at
`crates/veda-coordinator/src/store.rs` IS the data store. Every
replica holds every record in memory, the reconcile loop at
`crates/veda-gateway/src/rest/collections.rs` pulls them out, and
the local `SegmentManager` mirrors what cluster-wide Raft already
replicated. Correct, but caps write throughput at one Raft
leader's speed and makes the metadata snapshot grow linearly with
data.

The state of the art for a database in this shape is **not** a
different Raft library — it is a layered split:

1. **Small consensus group ("_meta")** — schema, shard plan, node
   membership, TSO leases, txn commit records. Low write rate.
   openraft 0.9 keeps doing the job here.
2. **Per-shard distributed append-only log (`veda-log`)** —
   record writes. BookKeeper-style quorum, NOT Raft. Writer
   leadership named in `_meta`.
3. **Hybrid Logical Clock + Timestamp Oracle (`veda-time`)** —
   monotonic logical time. Single leased allocator in `_meta`.
4. **MVCC versioning** in storage — records keyed by `(user_key,
   commit_ts)`.
5. **Snapshot-isolation transactions** — Percolator-style, replaces
   today's v0 stub.

Matches FoundationDB / Aurora / TiDB v8+ / YugabyteDB.

## Target layering

```
                       ┌─────────────────────────────────────────────┐
                       │  GATEWAY (veda-gateway)                     │
                       │   - shard routing (jump-consistent hash)    │
                       │   - acquires commit_ts / read_ts from TSO   │
                       │   - serves MVCC reads at requested ts       │
                       └─────────┬───────────────────┬───────────────┘
                                 │                   │
            ┌────────────────────┼──────────┐        │
            │                    │          │        │
            ▼                    ▼          ▼        ▼
  ┌──────────────────┐  ┌─────────────────┐  ┌──────────────────┐
  │ TSO + HLC        │  │ METADATA RAFT   │  │ PER-SHARD LOGS   │
  │ veda-time        │  │ ("_meta")       │  │ veda-log         │
  │ - HLC.now()      │  │ veda-coord.     │  │ - quorum write   │
  │ - leased TSO     │  │ - schemas       │  │ - tail readers   │
  │   (in _meta)     │  │ - shard plan    │  │ - BookKeeper-    │
  │ - monotonic ts   │  │ - nodes         │  │   style          │
  │                  │  │ - TSO lease     │  │ - epoch-fenced   │
  │                  │  │ - txn commits   │  │ - one writer     │
  │                  │  │   (small recs)  │  │   per epoch      │
  └────────┬─────────┘  └────────┬────────┘  └────────┬─────────┘
           │                     │                    │
           └────────────┬────────┴────────────────────┘
                        ▼
             ┌──────────────────────────────────────┐
             │ STORAGE / APPLIER (veda-storage)     │
             │ - tails assigned shard logs          │
             │ - applies (commit_ts, key) -> value  │
             │ - MVCC segments + LSM tier           │
             │ - publishes durable_ts watermark     │
             │ - GC of versions below safe_ts       │
             └──────────────────────────────────────┘
```

## Where we are today (Phase A foundations + start of Phase B)

Shipped in this rearchitecture commit:

| Piece | Location | Phase |
|---|---|---|
| Hybrid Logical Clock | `crates/veda-time/src/hlc.rs` | A |
| Tso trait + `LocalTso` | `crates/veda-time/src/tso.rs` | A |
| `TsLease` record | `crates/veda-time/src/lease.rs` | A |
| `GrantTsLease` / `RenewTsLease` cluster commands | `crates/veda-coordinator/src/raft_types.rs` | A |
| Lease apply path + epoch fencing | `crates/veda-coordinator/src/store.rs::apply_command` | A |
| `AppStateInner.tso: Arc<dyn Tso>` | `crates/veda-gateway/src/rest/state.rs` | A |
| `Invocation.commit_ts: Option<u128>` (harness scaffolding) | `crates/veda-coordinator/src/linearisability_checker.rs` | A |
| `ShardLog` single-node append-only log | `crates/veda-log/src/shard_log.rs` | B (start) |
| CRC32+len framing identical to `raft_wal.rs` | `crates/veda-log/src/format.rs` | B (start) |
| Segment file rollover (256MB) + 4KB sidecar index | `crates/veda-log/src/segment_file.rs` | B (start) |
| Polling `LogTailer` | `crates/veda-log/src/reader.rs` | B (start) |
| Gateway dark-write (`VEDA_LOG_DARK_WRITE=1`) | `crates/veda-gateway/src/rest/records.rs::insert` | B (start) |

What's deliberately NOT here yet (all later phases):

- `MetaLeasedTso` — the production TSO impl that talks to `_meta`
  via the lease grant. The data type ships now; the impl lands as
  a Phase A follow-up.
- Multi-node `ShardLog` replication (Phase D).
- Migration of the live data plane off `ClusterStateMachine.records`
  (Phase C).
- MVCC versioning of records (Phase F).
- Real Percolator-style snapshot isolation transactions (Phase G).
- WAN log shipping (Phase H).

## Reading order for new contributors

1. Read this file.
2. Read `.claude/plans/structured-mixing-turing.md` (top section) for
   the full multi-phase schedule + decision points.
3. Read `documentation/raft-migration.md` for the *current* (pre-
   rearchitecture) Raft path that we're about to migrate off.
4. Read `documentation/consensus-swap-decision.md` to understand why
   the answer is *not* "swap openraft for raft-rs" — it's this
   layered rearchitecture instead.
5. For the v0 transaction surface that becomes real Percolator SI
   in Phase G, see `documentation/transactions-v0.md`.

## Reference architecture sources

- **FoundationDB** — *FoundationDB: A Distributed Unbundled Transactional Key Value Store* (SIGMOD 2021).
- **Apache BookKeeper / DistributedLog** — *DistributedLog: A High Performance Replicated Log Service* (ICDE 2017).
- **Amazon Aurora** — *Amazon Aurora: Design Considerations for High Throughput Cloud-Native Relational Databases* (SIGMOD 2017).
- **Percolator** — *Large-scale Incremental Processing Using Distributed Transactions and Notifications* (OSDI 2010).
- **TiKV** — `components/txn_types` + `mvcc` packages (Apache-2.0).
- **CockroachDB HLC** — `pkg/util/hlc/hlc.go`. `veda-time` follows this.
- **Spanner TrueTime** — *Spanner* (OSDI 2012). We deliberately do not use TrueTime; the TSO is the alternative.

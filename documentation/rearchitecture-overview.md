# Rearchitecture — FoundationDB-shape data plane

This is the engineer-onboarding companion to
`.claude/plans/structured-mixing-turing.md`. The plan file holds
the full multi-phase schedule and decision points; this doc
summarises what's true *today* and where each new piece slots in.

## Why we rearchitected

Until Phase E hard-remove (commit `c8baf29`), every record write
funnelled through one cluster-wide openraft group.
`ClusterStateMachine.records` was literally the data store: every
replica held every record in memory, the reconcile loop pulled
them out, and the local `SegmentManager` mirrored what
cluster-wide Raft already replicated. Correct, but capped write
throughput at one Raft leader's speed and made the metadata
snapshot grow linearly with data. The Phase E hard-remove
deleted that path entirely; the doc below describes the
target state we're now in.

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

## Where we are today (Phases A-H landed in v0.2.0)

| Piece | Location | Phase | Status |
|---|---|---|---|
| Hybrid Logical Clock | `crates/veda-time/src/hlc.rs` | A | done |
| `Tso` trait + `LocalTso` + `SwappableTso` | `crates/veda-time/src/{tso,swappable}.rs` | A | done |
| `MetaLeasedTso` (talks to `_meta` lease) | `crates/veda-coordinator/src/meta_leased_tso.rs` | A | done |
| `GrantTsLease` / `RenewTsLease` with strict epoch monotonicity | `crates/veda-coordinator/src/raft_types.rs` + `store.rs` | A | done |
| `AppState.tso` + `load_meta_leased_tso()` auto-swap when Raft up | `crates/veda-gateway/src/rest/state.rs` + `server.rs` | A | done |
| `ShardLog` append-only log, 256 MB segments + 4 KB sidecar index | `crates/veda-log/src/{shard_log,segment_file}.rs` | B | done |
| `LogTailer` (gapless replay + `lag_seq` watermark) | `crates/veda-log/src/reader.rs` | B | done |
| Group commit (1 ms / 64 KB) | `crates/veda-log/src/shard_log.rs` | B | done |
| Per-collection `use_log_primary` schema flag + env fallback | `crates/veda-core/src/schema.rs` | C | done |
| `insert_via_log` is the only data-plane write path | `crates/veda-gateway/src/rest/records.rs` | C | done |
| `LogApplier` per-shard tail-and-apply task | `crates/veda-storage/src/log_applier.rs` | C | done |
| Gateway bootstrap auto-spawn (`VEDA_LOG_APPLIER=1`) | `crates/veda-gateway/src/log_appliers.rs` | C | done |
| BookKeeper-style replication (`LogEnsemble`, writer-leadership in `_meta`) | `crates/veda-log/src/replication/` | D | scaffolding done |
| Tonic gRPC transport for follower (`RawRpc` + bincode) | `crates/veda-log/src/tonic_log.rs` + `proto/log_service.proto` | D | done |
| `ChangeShardWriter` / `RegisterShardEnsemble` cluster commands | `crates/veda-coordinator/src/{raft_types,store}.rs` | D | done |
| `_meta` slimmed: `records` + `open_txns` + variants removed | `crates/veda-coordinator/src/{raft_types,store}.rs` | E | hard-removed |
| `RecordTxnCommit` for one-row-per-txn registry | `crates/veda-coordinator/src/raft_types.rs` | E | done |
| `MvccStore` with descending-commit_ts encoding | `crates/veda-storage/src/mvcc.rs` | F | done |
| `MvccWal` per-shard durability | `crates/veda-storage/src/mvcc_wal.rs` | F | done |
| `commit_ts` per doc in growing + sealed segments | `crates/veda-storage/src/{growing_segment,segment}.rs` | F | done |
| Per-doc visibility via `is_visible_at` / `visible_doc_ids_at` | both | F | done |
| `read_ts` plumbed through `SearchRequest` types | `crates/veda-query/src/search.rs` | F | done |
| Executor pre-filters candidates by visibility bitmap | `crates/veda-query/src/local_shard.rs` | F | done |
| `GET /records/:id?at=<ts>` + `POST /search { at: ts }` | `crates/veda-gateway/src/rest/{records,search}.rs` | F | done |
| Percolator `TxnCoordinator` (begin/commit/abort) | `crates/veda-coordinator/src/percolator.rs` | G | done |
| Background orphan-lock resolver | `crates/veda-coordinator/src/percolator.rs` + `state.rs` | G | done |
| `seed_records` for atomic create-and-load (with rollback) | `crates/veda-gateway/src/rest/collections.rs` | G | done |
| `SetCollectionAlias` + `_meta::aliases` + `POST /reindex` | `crates/veda-coordinator/src/{raft_types,store}.rs` + `gateway` | G | done |
| WAN log shipping (`LogShipper`, `LocalReceiver`) | `crates/veda-log/src/wan.rs` | H | done |
| Tonic WAN transport + `safe_ts` gating | `crates/veda-log/src/tonic_wan.rs` + `proto/wan_service.proto` | H | done |
| Operator `promote_replica` cutover primitive | `crates/veda-log/src/wan.rs` | H | done |
| `RegionTopology` persisted under sentinel key in `_meta::shard_ensembles` | `crates/veda-coordinator/src/store.rs::REGION_TOPOLOGY_KEY` | H | done |
| `veda-admin migrate-metadata` CLI | `crates/veda-admin/` | E | stub (no v0 to migrate post hard-remove) |
| BookKeeper open-and-recover truncate + ensemble change + snapshot install | `crates/veda-log/src/{shard_log,replication}.rs` | D | done (v0.2.1) |
| `FollowerHealthMonitor` edge-triggered auto-orchestrator | `crates/veda-log/src/replication.rs` | D | done (v0.2.1) |
| `quorum_partition.rs` split-brain integration test | `crates/veda-log/tests/quorum_partition.rs` | D | done (v0.2.1) |
| MVCC WAL checkpoint (atomic snapshot + WAL truncate) | `crates/veda-storage/src/{mvcc_wal,segment_manager}.rs` | F | done (v0.2.1) |
| Native `search_filtered` for Flat / IVF-PQ / DiskANN | `crates/veda-index/src/{flat,ivfpq,diskann}` | F | done (v0.2.1) |
| `mvcc_isolation` + `gc_safety` linearisability tests | `crates/veda-storage/tests/mvcc_isolation_gc_safety.rs` | F | done (v0.2.1) |
| Real re-chunk pipeline (chunker + embedder transforms) | `crates/veda-gateway/src/rest/collections.rs::rechunk_source_into_sibling` | G | done (v0.2.1) |
| `snapshot_isolation` Adya anomaly suite + `atomic_reindex` test | `crates/veda-coordinator/tests/{snapshot_isolation,atomic_reindex}.rs` | G | done (v0.2.1) |
| Admin REST: `GET/PUT /v1/admin/region-topology`, `POST /v1/admin/promote-region`, `GET/PUT /v1/admin/replica-safe-ts` | `crates/veda-gateway/src/rest/admin.rs` | H | done (v0.2.1) |
| Cross-region reader gating in live `/records` + `/search` (sentinel `0` = no gating; 503 when `read_ts > replica_safe_ts`) | `crates/veda-gateway/src/rest/{records,search,state}.rs` | H | done (v0.2.1) |
| `planned_failover` zero-acked-write-loss + `old_primary_fences_after_planned_failover` | `crates/veda-log/src/wan.rs` | H | done (v0.2.1) |

What's deliberately NOT here yet (tracked follow-ups, not in v0.2.1):

- Niriksha frontend redesign — separate 7-phase plan (~14 engineer-weeks). Not started.

(Done 2026-05-27: the durable `LogApplier` cursor, the `ShardLogRegistry` (which replaced the now-deleted vestigial `MultiGroupCoordinator`), and per-shard applier-task cancellation on collection drop all landed. Design: `documentation/2026-05-27-applier-cursor-shardlog-registry-design.md`.)

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

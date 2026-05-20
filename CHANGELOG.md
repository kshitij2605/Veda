# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-05-20

FoundationDB-shape data plane: record writes leave the cluster-wide Raft, the metadata Raft becomes a small "_meta" group, and Phases A through H of `.claude/plans/structured-mixing-turing.md` land in production-integrated form.

### Added — new crates
- **`veda-time`** — Hybrid Logical Clock, `Tso` trait, `LocalTso` (single-node monotonic), `MetaLeasedTso` (talks to `_meta` via a lease grant), `SwappableTso` wrapper (hot-swap behind `Arc<dyn Tso>`).
- **`veda-log`** — Per-shard append-only `ShardLog` with the `[crc32][len][bincode]` framing inherited from `raft_wal`. Group commit (1 ms / 64 KB), 256 MB segment rollover with 4 KB sidecar index. `LogTailer` for gapless replay. BookKeeper-style replication module (`LogEnsemble`, `LogWriter`, `LogFollower`) with epoch fencing. Tonic gRPC transport for cross-process followers AND for WAN log shipping (`ShipBatchRpc` + `safe_ts` gating). Operator-driven `promote_replica` cutover primitive.
- **`veda-admin`** — Offline maintenance CLI scaffold; the migration subcommand is now a no-op stub (Phase E hard-remove dropped the v0 fields the tool used to drain).

### Added — gateway + storage architecture
- **Phase A** — `AppState.tso: Arc<dyn Tso>`; `load_meta_leased_tso` swaps `MetaLeasedTso` in when Raft is up. `GrantTsLease` / `RenewTsLease` `ClusterCommand`s with strict epoch monotonicity.
- **Phase B → C** — Record writes go through `veda-log` instead of `ClusterCommand::InsertRecord`. Per-collection `use_log_primary: Option<bool>` schema flag (env fallback `VEDA_LOG_PRIMARY`). Per-shard `LogApplier` in `veda-storage` that drains a `ShardLog` into the local `SegmentManager`; gateway bootstrap auto-spawns one per shard when `VEDA_LOG_APPLIER=1` and the writer's inline-apply is then skipped to avoid double-apply.
- **Phase E (hard-remove)** — The five v0 `ClusterCommand` variants (`InsertRecord`, `BeginTxn`, `TxnWrite`, `CommitTxn`, `AbortTxn`) and the `records` + `open_txns` fields on `ClusterStateMachine` are removed entirely, along with the gateway's Raft fallback block, the dark-write path, and the `cs.records` reconcile loop.
- **Phase F (MVCC, end to end)** — `MvccStore` with `(user_key, commit_ts)` descending-ts encoding, in-memory `BTreeMap` core, max-versions-per-key cap, safe_ts-driven GC. `MvccWal` per-shard durability with torn-tail-safe `[crc32][len][bincode]` frames; `enable_mvcc_durable(path)` replays on open. Per-doc `commit_ts: Option<u128>` in `GrowingSegment` and a `commit_ts.bin` sidecar in sealed segments. `is_visible_at(doc_id, read_ts)` + `visible_doc_ids_at(read_ts) -> RoaringBitmap` on both. `LocalShardSearcher` (dense / sparse / multi-vector) intersects the visibility bitmap into the `allowed` set BEFORE the index walk runs, so HNSW / IVF-PQ / Flat / DiskANN never spend candidate budget on invisible docs. `?at=<u128>` on `GET /records` and on `POST /search` selects the snapshot timestamp.
- **Phase G** — Real Percolator `TxnCoordinator` (begin/commit/abort) replaces the v0 stub on `/v1/txn`. Background orphan-lock resolver. `seed_records` on `POST /v1/collections` for atomic create-and-load (compensating drop on seed failure). `SetCollectionAlias` `ClusterCommand` + `aliases` map in `_meta` + `POST /v1/collections/{name}/reindex` that creates `name@vN` and atomically swaps the alias.
- **Phase H** — WAN log shipping with `LogShipper` + `LocalReceiver`. `ShipBatchRpc` carries `primary_safe_ts`; replicas advance their local `safe_ts` as batches durably land. `check_read_ts` on the replica `LogReceiver` gates cross-region snapshot reads. `promote_replica(shipper, new_epoch, max_drain_rounds)` operator primitive: drain → refuse-if-incomplete → bump epoch. `RegionTopology` persisted in `_meta::shard_ensembles` under a sentinel key.

### Tests added
- 9 LogApplier unit tests; 5 MVCC WAL unit tests; 5 `mvcc_executor_test` integration tests (incl. sealed-segment seal→reopen→search) + 4 segment-manager MVCC durability tests; 4 atomic-create / rollback tests; 4 reindex / alias tests; 3 mvcc_test gateway tests; 1 log_applier gateway test; 3 wan cutover + lag-recovery tests; 2 tonic_log + 2 tonic_wan round-trip tests; 6 store::tests covering aliases + region topology + ts lease.

### Changed
- `ClusterStateMachine::snapshot()` no longer carries `records` / `open_txns`. Metadata snapshot bounded by schema count, not record count.
- `insert_via_log` is now the only data-plane write path. The non-Raft fall-through that writes directly to `SegmentManager` is kept for single-node deployments without log-primary; multi-node operates via the log path universally.

### Removed
- `dark_write_enabled` and the dark-write blocks. Phase B's `VEDA_LOG_DARK_WRITE` knob existed to mirror writes to the log alongside Raft during pre-Phase-C rollout; with Phase E hard-remove the log IS the only path.
- `migrate_persisted_blob` + `build_persisted_blob_for_state` from `veda-coordinator::raft_storage` (no in-process migration to express now that the v0 fields are gone).
- `PendingTxn` struct from `veda-coordinator::store` (transaction state lives in the Percolator `TxnCoordinator`).
- Gateway per-record post-filter for MVCC visibility in `search.rs` — the executor handles every visibility decision precisely.

### Notes
- The metadata Raft snapshot for a workload of 100k collections + 10M records is now bounded under 100 MB (was unbounded; previously grew linearly with record count).
- Backwards-compatible on-disk format: pre-MVCC sealed segments stay byte-identical (no `commit_ts.bin` written when no doc was stamped); pre-Phase-F `raft_state.bin` no longer loads (rebuild from logs or use the pre-Phase-E binary to migrate first).

## [0.1.0] - 2026-04-07

### Added
- Core vector database engine with 7 Rust crates
- HNSW index with adaptive filtered search (95.5% recall@10)
- DiskANN (Vamana graph) index (99.9% recall@10)
- IVF-PQ index with residual product quantization
- RaBitQ and scalar quantization
- SIMD-optimized distance functions (AVX2 + scalar fallback)
- 6 distance metrics: Cosine, Euclidean, DotProduct, Hamming, Manhattan, Jaccard
- mmap-backed vector storage format
- Scalar indexes: inverted, range, bitmap
- Growing + sealed segment architecture with compaction
- SQL-like filter expression parser and evaluator
- Multi-vector fusion search (RRF, WeightedSum, Intersection)
- Distributed query with shard fan-out and merge
- Suffix array pattern search with fuzzy matching
- REST API gateway (axum) with full CRUD
- Auto-embedding service with modality validation
- Blob storage with MinIO/S3-compatible and local filesystem backends
- JWT authentication and RBAC (Admin/Editor/Viewer)
- Full observability stack: Prometheus metrics (16 metric types), OpenTelemetry tracing and log export, health/readiness endpoints, query explain with per-phase timing
- Pre-built Grafana dashboard with provisioned datasources (Mimir, Tempo, Loki)
- Docker Compose `monitoring` profile (OTel Collector, Prometheus, Mimir, Tempo, Loki, Grafana)
- W3C Trace Context propagation for incoming traceparent headers
- HTTP request metrics middleware with automatic error counting
- `#[instrument]` tracing spans on handlers, embedding client, blob store, segment manager, and search executor
- Cluster coordination with shard routing
- WAL abstraction with in-memory backend
- Niriksha dashboard with 7 pages (React + TypeScript)
- Python SDK (veda-client)
- 7 configurable chunking strategies: MarkdownRecursive (default), MarkdownHeading (levels 1/2/3), Semantic, Sentence, FixedSize, Page, None
- Semantic chunking strategy with word-overlap Jaccard similarity for topic-based splitting
- Markdown heading-level chunking (split on H1, H2, or H3 depth)
- FlashParse OCR integration with Docker Compose profiles (`docker compose --profile ocr up`)
- End-to-end document processing pipeline (parse, chunk, embed, insert) via `/v1/collections/{name}/process-document` and `/v1/collections/{name}/process-documents`
- `.env.example` configuration template for VLM and service settings
- Docker and Docker Compose deployment
- ANN benchmark suite
- Comprehensive integration and stress tests

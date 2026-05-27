# Architectural Decision Records

This document captures the key architectural decisions made during the design and implementation of Veda.

---

## ADR-001: Rust as Implementation Language

**Status:** Accepted

### Context
A vector database has strict performance requirements: low-latency similarity search, efficient memory management for large vector sets, and safe concurrency for multi-threaded indexing and queries. We evaluated Rust, C++, Go, and Java.

### Decision
Use Rust as the primary implementation language for the core engine.

### Consequences
- **Positive:** Memory safety without garbage collection pauses, zero-cost abstractions, excellent SIMD intrinsics support, strong type system catches bugs at compile time, and Cargo provides a mature build and dependency system.
- **Negative:** Steeper learning curve for contributors, longer compile times compared to Go/Java, smaller ecosystem for some database-specific libraries compared to C++.

---

## ADR-002: Full Disaggregation Architecture

**Status:** Accepted

### Context
We needed to decide between a monolithic architecture and a disaggregated one where gateway, query, index, storage, and coordination are separate crates with clean interfaces.

### Decision
Adopt a fully disaggregated architecture with 7 crates: `veda-core`, `veda-index`, `veda-storage`, `veda-query`, `veda-gateway`, `veda-coordinator`, and `veda-server`.

### Consequences
- **Positive:** Each component can be developed and tested independently, clear ownership boundaries, easier to scale individual layers, enables future deployment as separate services.
- **Negative:** More complex dependency graph, cross-crate refactoring is more involved, potential overhead from trait-based abstractions at crate boundaries.

---

## ADR-003: HNSW as Primary Index

**Status:** Accepted

### Context
Multiple ANN index algorithms exist (HNSW, NSG, IVF variants, tree-based). We needed a default index type that offers the best balance of recall, speed, and simplicity for the common case.

### Decision
Use HNSW (Hierarchical Navigable Small World) as the primary and default index type, with DiskANN, IVF-PQ, and Flat as alternatives.

### Consequences
- **Positive:** HNSW is well-studied with predictable performance characteristics, achieves 95.5% recall@10 on our benchmarks, supports incremental insertions without full rebuilds, and integrates naturally with filtered search.
- **Negative:** Higher memory overhead compared to IVF-PQ due to graph structure, not ideal for extremely large datasets that cannot fit in memory (DiskANN fills this gap).

---

## ADR-004: RaBitQ for Quantization

**Status:** Accepted

### Context
Vector quantization is essential for reducing memory consumption. Common options include Product Quantization (PQ), Scalar Quantization (SQ), and the newer RaBitQ (Random Bit Quantization) approach.

### Decision
Implement RaBitQ as the primary quantization method alongside SQ8 (8-bit scalar quantization) as a simpler alternative.

### Consequences
- **Positive:** RaBitQ achieves strong compression ratios with minimal recall loss, works well with binary distance operations for fast candidate filtering, and pairs well with HNSW re-ranking.
- **Negative:** RaBitQ is less widely adopted than PQ, so fewer reference implementations and community resources. We mitigate this by also offering SQ8 for users who want a more conventional approach.

---

## ADR-005: Suffix Array Pattern Search

**Status:** Accepted

### Context
Users need to search text fields by substring patterns in addition to vector similarity. Traditional approaches include inverted indexes with n-grams, trigram indexes, or suffix-based structures.

### Decision
Use suffix arrays for pattern search, with support for fuzzy matching via edit distance.

### Consequences
- **Positive:** Suffix arrays provide O(m log n) substring search, lower memory overhead than suffix trees, and integrate cleanly with the existing filter pipeline. Fuzzy matching extends utility beyond exact substring search.
- **Negative:** Suffix array construction requires O(n) time and the array itself consumes O(n) space. Updates require rebuilding the array for the affected segment, which is acceptable under the growing/sealed segment model.

---

## ADR-006: No CLI Client (REST + SDK + Niriksha)

**Status:** Accepted

### Context
We considered whether to build a dedicated CLI client for interacting with Veda. Alternatives are the REST API directly (via curl or similar), a language SDK, or the Niriksha dashboard.

### Decision
Do not build a dedicated CLI client. Provide REST API, Python SDK, and the Niriksha dashboard as the three access methods.

### Consequences
- **Positive:** Less code to maintain, the REST API is universally accessible via curl/httpie, the Python SDK covers programmatic use cases, and Niriksha provides a visual interface for exploration and administration.
- **Negative:** Power users who prefer CLI workflows must use curl or write shell wrappers. We may revisit this decision if user demand is strong.

---

## ADR-007: MinIO as Default Blob Storage

**Status:** Accepted

### Context
Veda supports storing raw files (images, PDFs, audio, video) alongside their embeddings. We needed a blob storage backend that is self-contained for development and compatible with cloud deployments.

### Decision
Use MinIO as the default blob storage backend, accessed via the S3-compatible API. Also support a local filesystem backend for development.

### Consequences
- **Positive:** MinIO is S3-compatible, so the same code works with AWS S3, GCS (via compatibility layer), and other S3-compatible stores. MinIO is easy to run locally via Docker. The local filesystem backend keeps development simple.
- **Negative:** Adds MinIO as a runtime dependency for production deployments with blob storage. Users who do not need blob storage can disable it entirely.

---

## ADR-008: Channel-Based Coordination (Raft-Ready)

**Status:** Superseded by ADR-013 — see below

### Context
Cluster coordination requires consensus on shard assignments, leader election, and configuration changes. Full Raft consensus is complex to implement correctly but necessary for production-grade multi-node deployments.

### Decision
Implement coordination using Rust async channels for intra-process communication, with the architecture designed so that Raft can be plugged in later for multi-node consensus.

### Consequences
- **Positive:** Channel-based coordination is simple, correct for single-node and embedded multi-shard scenarios, and easy to test. The coordinator trait boundary makes future Raft integration a localized change.
- **Negative:** Multi-node deployments are not supported until Raft (or equivalent) is implemented. This is acceptable for the initial release.

---

## ADR-009: Axum for REST API

**Status:** Accepted

### Context
We evaluated Rust web frameworks for the REST API layer: axum, actix-web, warp, and rocket.

### Decision
Use axum as the HTTP framework for the gateway crate.

### Consequences
- **Positive:** Axum is built on tokio and tower, which we already use throughout the stack. Its extractor-based design produces clean handler signatures. Strong community adoption and active maintenance. Tower middleware (compression, rate limiting, tracing) works out of the box.
- **Negative:** Less "batteries included" than actix-web for some features (e.g., WebSocket support requires additional setup). This is a minor tradeoff given our REST-first API design.

---

## ADR-010: Pure Python SDK (Not PyO3)

**Status:** Accepted

### Context
The Python SDK could be implemented as a pure Python HTTP client or as a PyO3/Rust-backed native extension that embeds parts of the engine.

### Decision
Implement the Python SDK as a pure Python package (`veda-client`) that communicates with Veda over REST.

### Consequences
- **Positive:** Simple to install (`pip install veda-client`), no compilation step or Rust toolchain required for users, easy to distribute across all platforms, straightforward to maintain and debug.
- **Negative:** All operations go through HTTP, so there is no option for in-process embedding or direct index access. Latency is bounded by network round-trips. This is the correct tradeoff for a client SDK -- users who need in-process access can use the Rust crates directly.

---

## ADR-011: OpenTelemetry-First Observability

**Status:** Accepted

### Context
A vector database needs comprehensive observability: metrics for capacity planning, distributed tracing for latency diagnosis, and structured logs for debugging. We evaluated several approaches: custom metrics libraries, StatsD, Prometheus client libraries directly, and the OpenTelemetry ecosystem.

### Decision
Adopt an OpenTelemetry-first observability architecture with three pillars:
1. **Metrics:** Use the `metrics` crate (v0.24) as a facade with `metrics-exporter-prometheus` for Prometheus exposition at `/v1/admin/metrics`. 16 metric constants defined centrally in `veda-core::metrics`.
2. **Tracing:** Use `tracing` + `tracing-opentelemetry` for distributed tracing. OTel trace and log export is conditional on `OTEL_EXPORTER_OTLP_ENDPOINT` being set. W3C Trace Context propagation for incoming `traceparent` headers.
3. **Logs:** Use `opentelemetry-appender-tracing` to bridge `tracing` events to OTel log records. Structured JSON format via `tracing-subscriber`.

The monitoring stack runs as a Docker Compose profile (`monitoring`) with OTel Collector, Prometheus, Mimir, Tempo, Loki, and Grafana with pre-provisioned datasources and dashboard.

### Consequences
- **Positive:** Standard OTel protocol means any OTel-compatible backend works. Conditional initialization means zero overhead when monitoring is disabled. The `metrics` crate facade decouples metric recording from the exporter. Pre-built Grafana dashboard provides immediate visibility.
- **Negative:** Six additional Docker services for the monitoring stack. OTel dependency adds compile time. Mimir requires MinIO for long-term storage, adding another dependency to the monitoring profile.

---

## ADR-012: MarkdownRecursive as Default Chunking Strategy

**Status:** Accepted

### Context
The document processing pipeline needs to chunk OCR output into pieces suitable for embedding and vector search. Many chunking strategies exist, ranging from simple fixed-size splitting to model-dependent semantic chunking. The choice of default strategy affects retrieval quality, processing cost, and operational complexity.

### Decision
Use MarkdownRecursive as the default chunking strategy with 512 tokens (~2000 characters) chunk size and 15% overlap (~300 characters). Research shows that recursive markdown splitting outperforms semantic chunking in end-to-end RAG benchmarks while being approximately 10x cheaper, since it requires no embedding model calls during the chunking phase.

### Consequences
- **Positive:** Simple, fast, and deterministic. No model dependency during chunking. Respects document structure (headings, paragraphs) for natural break points. Works well with markdown output from FlashParse OCR.
- **Negative:** May miss some topic boundaries that semantic chunking catches, particularly in unstructured text without clear markdown formatting. Users can switch to semantic chunking via the `chunking_strategy` parameter when needed.

---

## ADR-013: FoundationDB-shape data plane (`_meta` Raft + per-shard log)

**Status:** Accepted (supersedes ADR-008's "Raft-ready" stance for record writes)

### Context
The original design funnelled every record write through a single cluster-wide openraft group. `ClusterStateMachine.records` was literally the data store: every replica held every record in memory, the reconcile loop pulled them out, and the local `SegmentManager` mirrored what Raft already replicated. Correct but caps write throughput at one leader's single-node speed, and makes the metadata snapshot grow linearly with the dataset.

The state of the art for this shape is not "swap openraft for raft-rs" but a layered split (FoundationDB / Aurora / TiDB v8+ / YugabyteDB shape): a small consensus group carrying only metadata, plus a per-shard distributed log carrying record writes.

### Decision
Split the data plane:

1. `_meta` openraft group — schema, shard plan, node membership, TSO leases, transaction commit records, collection aliases, region topology. Low write rate, small snapshot.
2. Per-shard `veda-log` — record writes. BookKeeper-style `(E, Qw, Qa)` quorum replication with epoch fencing. Writer-leadership per shard named in `_meta` (`ChangeShardWriter`). Data-plane RPCs do not touch openraft.
3. Tonic gRPC for cross-process replication, both intra-region (follower) and cross-region (WAN).

### Consequences
- **Positive:** Metadata snapshot bounded by schema count, not record count. Record write throughput scales with shard count, not leader speed. Cross-region replication via async log shipping with bounded staleness. Standard FoundationDB-shape architecture; references easily map to known systems.
- **Negative:** Two replication systems to operate (openraft + BookKeeper-style). Migration off the single-Raft data plane required a deprecation window (we ultimately hard-removed in the same release since no live deployments existed).

---

## ADR-014: Hybrid Logical Clock + leased Timestamp Oracle

**Status:** Accepted

### Context
The new data plane needs monotonic commit timestamps that order writes across shards and survive leader changes. Options: (a) Spanner-style TrueTime (requires GPS hardware), (b) per-shard HLC ordering (no cross-shard SI), (c) leased centralised TSO with HLC physical-time backbone.

### Decision
Adopt CockroachDB-style Hybrid Logical Clocks (`(physical_ns, logical)`) as the underlying time format. Issue them from a single leased Timestamp Oracle whose lease is granted via `GrantTsLease` in `_meta`. Default lease horizon 10 s; leader-change-driven rewinds prevented by the lease ceiling.

### Consequences
- **Positive:** Monotonic commit_ts across shards enables cross-shard snapshot isolation. Lease handoff sub-50 ms. Batched allocation (10k ts per RPC) keeps the TSO from becoming a hot path. `SwappableTso` wrapper enables non-disruptive swap from `LocalTso` (single-node) to `MetaLeasedTso` (multi-node) once Raft is up.
- **Negative:** Single allocator is a logical bottleneck; the batching mitigates throughput but operators must monitor lease handoff latency. Not as crisp as TrueTime for sub-millisecond external consistency — we explicitly do not target Spanner-class semantics.

---

## ADR-015: MVCC versioning with descending-commit_ts key encoding

**Status:** Accepted

### Context
Phase G snapshot-isolation transactions need MVCC: records keyed by `(user_key, commit_ts)` so reads at `read_ts = T` see the latest write whose `commit_ts ≤ T`. Two key-encoding directions are possible: ascending commit_ts (reverse-seek to find latest) or descending commit_ts (forward-seek finds latest directly).

### Decision
Encode commit_ts as `MAX_TS - real_ts` (descending) so that a forward iterator from `(user_key, read_ts)` finds the newest visible version as its first hit. Carry per-doc `commit_ts: Option<u128>` in both `GrowingSegment` and `SegmentReader` (sealed-segment `commit_ts.bin` sidecar). MVCC durability via a per-shard WAL (`MvccWal`) with the same torn-tail-safe framing the data log uses.

### Consequences
- **Positive:** RocksDB / TiKV convention; saves one reverse-seek per read. Bounded per-key version count (default 100) prevents pathological hot-key growth. Sealed-segment sidecar is omitted entirely when no doc was stamped, so pre-MVCC segments stay byte-identical on disk (backwards-compatible).
- **Negative:** GC must coordinate across `safe_ts` advancement to avoid dropping a version a live snapshot could observe. Read amplification under heavy churn (multiple versions per key) — bounded by `max_versions_per_key`.

---

## ADR-016: Percolator-style snapshot-isolation transactions

**Status:** Accepted (replaces the v0 single-Raft txn stub)

### Context
The v0 `BeginTxn` / `TxnWrite` / `CommitTxn` / `AbortTxn` `ClusterCommand`s funnelled every staged write through the cluster-wide Raft and held them as `PendingTxn` blobs on every node. No isolation, no conflict detection — last-writer-wins. Phase G requires a real transactional substrate.

### Decision
Vendor Percolator's two-phase commit protocol: optimistic concurrency control, primary-and-secondary lock rows on the per-shard log at prewrite (`start_ts`), single `RecordTxnCommit` entry in `_meta` at commit (`commit_ts`), async cleanup of secondary lock markers. Orphan locks resolved by checking the primary in `_meta`. Background orphan-lock resolver runs every 5 s.

### Consequences
- **Positive:** Snapshot-isolation reads (no anomalies except write-skew, which SI explicitly allows). Cross-shard atomicity via locks on each shard's log. `_meta` chatter is one commit record per txn, not one per write. TiKV's `concurrency_manager` + `txn_types` modules serve as the reference algorithm.
- **Negative:** Locks are real state that survives node restarts; the orphan-lock resolver must run or stuck locks accumulate. SI is not serialisable — applications that need it must use external coordination.

---

## ADR-017: Atomic re-chunk via collection aliases

**Status:** Accepted

### Context
Re-chunking / re-embedding a collection (different chunker config, different embedding model) must be atomic from the client's perspective — a partially-rebuilt index should never serve reads. Three options: in-place rebuild (locks the collection), copy-on-write into the same name (requires per-record versioning), or sibling-build + atomic swap.

### Decision
Use a sibling-build pattern. `POST /v1/collections/{name}/reindex` creates `name@vN` with the same schema, then `SetCollectionAlias { alias: "docs", target: "docs@v2" }` atomically points the alias at the rebuilt sibling. Rollback is `SetCollectionAlias { alias: "docs", target: "docs" }` (alias-to-self clears the entry). The alias map persists in `_meta::aliases` and round-trips through snapshots.

### Consequences
- **Positive:** Atomic from any reader's perspective — one `_meta` commit flips the alias. Rollback is the same primitive in reverse. Old data stays on disk (auditable / restorable) until an explicit drop.
- **Negative:** Disk footprint roughly doubles during the rebuild window. Read-side alias resolution adds one hop (single-hop chain by design — no chasing through multi-level aliases).

---

## ADR-018: WAN log shipping with operator-driven failover

**Status:** Accepted

### Context
Cross-region replication tradeoffs: synchronous commits across regions cost ~30 ms RTT per write (Spanner-class durability, prohibitive throughput); async log shipping with bounded staleness (FoundationDB / Aurora style) preserves throughput at the cost of a configurable RPO window.

### Decision
Async log shipping. `LogShipper` drains a source `ShardLog` to a replica `LocalReceiver` (or `TonicReceiver` cross-process) in batches. Each `ShipBatchRpc` carries the primary's `safe_ts`; replicas advance their local `safe_ts` once the batch is durable. Cross-region readers call `check_read_ts(read_ts)` which rejects reads ahead of the replica's `safe_ts`. Operator-driven `promote_replica` (no auto-failover): drains pending → refuses if drain incomplete → bumps the topology epoch → old primary's subsequent ships fence via `EpochMismatch`.

### Consequences
- **Positive:** Throughput stays at primary-region quorum speed. Bounded-staleness reads are explicit, not silent. Operator failover prevents split-brain — the same conservative stance FoundationDB and Spanner take.
- **Negative:** RPO > 0 in steady state (typically seconds). No auto-failover means longer RTO when the primary region is hard-down; operators must drive the cutover.

---

## ADR-019: Per-shard `LogApplier` for follower nodes

**Status:** Accepted

### Context
On the writer node, `insert_via_log` appends to the log AND applies inline to the local `SegmentManager` in the same call (fast feedback). On follower nodes the log entry arrives via Tonic-replicated `AppendRpc`; something must drive those entries into the follower's local `SegmentManager`.

### Decision
Per-shard `LogApplier` task (`crates/veda-storage/src/log_applier.rs`) that holds a `LogTailer` and a handle to the shard's `SegmentManager`, applying `LogOp::Put` / `LogOp::Delete` entries on each tick. Gated by `VEDA_LOG_APPLIER=1` env. When the env is on, the writer's inline-apply path is skipped to prevent double-apply; visibility comes via the applier tick (~20-50 ms).

### Consequences
- **Positive:** Clean separation — same log entries are the source of truth for every node's SegmentManager. Errors are logged but never propagate (an applier crash must not take the gateway down). The standalone module is fully unit-testable.
- **Negative:** Visibility latency on followers is bounded by the applier tick. *(Resolved 2026-05-27: the cold-start concern — the applier re-applying the whole log on restart because its cursor was in-memory only — is fixed. The gateway applier (`ShardApplier` in `crates/veda-gateway/src/log_appliers.rs`) now persists a durable CRC-framed cursor at the sealed watermark and resumes from it. See `documentation/2026-05-27-applier-cursor-shardlog-registry-design.md`.)*

---

## ADR-020: Pre-filter MVCC visibility inside the index walk

**Status:** Accepted

### Context
With MVCC live, a search at `read_ts` should not surface candidates whose `commit_ts > read_ts` or whose version has been superseded. Two implementations: (a) post-filter — execute the index walk normally, drop invisible hits afterwards; (b) pre-filter — intersect the visibility bitmap with the `allowed` set BEFORE calling `index.search_filtered`, so the index never spends candidate budget on invisible docs.

### Decision
Pre-filter. `LocalShardSearcher::search_shard` (and its sparse / multi-vector siblings) computes the visibility bitmap from `GrowingSegment::visible_doc_ids_at` and `SegmentReader::visible_doc_ids_at` and intersects it into the `allowed` set fed to `VectorIndex::search_filtered`. Brute-force scans check `mvcc_visible.contains(doc_id)` inside the loop. The gateway's old per-record post-filter is removed — redundant.

### Consequences
- **Positive:** HNSW / quantized-HNSW (which override `search_filtered` with native-walk implementations) never traverse invisible neighbours. Top-k slots aren't wasted. Correctness invariant preserved (the supersession test proves a better-matching superseded version is correctly dropped). The default `search_filtered` (10× oversample + post-filter) still works for IVF-PQ / DiskANN at v1 — they inherit correctness at the cost of some wasted retrieval.
- **Negative:** Building the visibility bitmap costs O(N) per search for segments with MVCC stamps. Cheap relative to the search itself, but worth bench-checking on very large segments.

---

## ADR-021: Edge-triggered follower-health monitor drives ensemble change

**Status:** Accepted

### Context
With BookKeeper-style replication, a single follower failure stalls the ack quorum until either the writer retries indefinitely (unbounded latency) or the ensemble is changed to swap out the failing node. Two failure-detector designs were considered: (a) level-triggered — fire on every probe while the follower is below the threshold; (b) edge-triggered — fire exactly once per crossing of the threshold, and not again until the follower recovers and fails again.

### Decision
Edge-triggered. `FollowerHealthMonitor::probe_once` tracks consecutive-failure counts per `NodeId` and only emits the `NodeId` when the count crosses `failure_threshold` from below. `spawn(on_failure)` runs the probe loop on a `tokio::interval` and invokes the user-supplied closure for each newly-failed node. The closure is responsible for driving `rebuild_ensemble(current, failed, replacement)` and persisting the new ensemble via `RegisterShardEnsemble` on `_meta`.

### Consequences
- **Positive:** A persistently-down follower yields exactly one orchestration call per outage, not one per probe interval. The orchestrator stays at human-driven escalation rates even when network partitions last hours. The monitor itself is dependency-light: a list of `Arc<dyn LogFollower>`, a threshold, a probe interval — embeddable in any operator process.
- **Negative:** A flapping follower (fail → recover → fail → recover) triggers one ensemble change per cycle, which is the intended behaviour but can amplify churn; the operator should set `failure_threshold` high enough (default 3 consecutive failures) to absorb single-probe blips. `_meta` rate-limiting on `ChangeShardWriter` / `RegisterShardEnsemble` is the second line of defence (cf. risk register in `.claude/plans/structured-mixing-turing.md`).

---

## ADR-022: MVCC WAL checkpoint = atomic snapshot file + WAL truncate

**Status:** Accepted

### Context
The MVCC WAL (`MvccWal` per shard) appends every `(key, ts, payload)` write so the in-memory `MvccStore` can be reconstructed on open. Without a checkpoint mechanism the WAL grows without bound, replay cost scales linearly with the number of writes ever made, and disk fills up under any sustained workload.

### Decision
Periodic checkpoint that writes a complete snapshot of the in-memory state to a separate `mvcc_snap` file using the same atomic-write idiom as the metadata snapshot: write `<path>.tmp`, `fsync`, then `rename`. After the snapshot is durable, the WAL is truncated to zero length. On open, `enable_mvcc_durable(path)` first calls `load_snapshot_into(snap, store)` and then replays the (now-short) trailing WAL. `SegmentManager::mvcc_checkpoint() -> Result<Option<usize>>` is the operator-facing trigger; returning `None` indicates MVCC durability is not enabled for that shard.

### Consequences
- **Positive:** Replay cost is bounded by `time-since-last-checkpoint`, not total history. Snapshot installation is atomic: a crash mid-write leaves the previous snapshot intact (the `.tmp` is orphaned and ignored). The same primitive supports future point-in-time recovery and follower snapshot install over WAN.
- **Negative:** Checkpoint cost is proportional to the live keyspace (every key-version pair is serialised). For workloads with very large keyspaces the operator should checkpoint less frequently, accepting longer replay in exchange for lower amortised cost.

---

## ADR-023: Native `search_filtered` overrides on every index

**Status:** Accepted

### Context
The default `VectorIndex::search_filtered` implementation runs the unfiltered top-k search at 10× oversample and then drops post-walk hits whose `doc_id` is not in the allowed bitmap. For sparse bitmaps (e.g. an `read_ts` snapshot that excludes 90% of docs), 10× oversample is insufficient — the recall collapses. For dense bitmaps it wastes work. ADR-020 introduces the bitmap-pre-filter strategy at the executor level; this ADR pushes the same logic into the index implementations themselves.

### Decision
Each index ships a native `search_filtered` that consults the bitmap INSIDE the candidate walk. `Flat` and `IVF-PQ` skip non-bitmap docs as the metric-comparison loop runs. `DiskANN` reuses the graph walk with `l = self.l_search.max(k * 4)` to keep recall stable when the bitmap is sparse, and tests every visited node against the bitmap. HNSW already had a native filtered walk; this ADR generalises the pattern.

### Consequences
- **Positive:** Recall stays at the unfiltered level regardless of bitmap density. Top-k slots are never spent on invisible docs. The 10× oversample fallback is no longer load-bearing.
- **Negative:** Each index now carries its own bitmap-aware walk; the abstraction "filtered search" leaks into the index implementations. Acceptable because the API surface is identical (one trait method) — only the body differs.

---

## ADR-024: Cross-region reader gating uses sentinel `0` for "no gating"

**Status:** Accepted

### Context
The replica `safe_ts` watermark gates cross-region snapshot reads: a read at `read_ts > replica_safe_ts` must be rejected, because the replica's `MvccStore` has not yet seen all writes ≤ `read_ts`. The default value (before any WAN batch lands) needs to encode "no gating in effect" cleanly. Two designs: (a) default to `u128::MAX`, gate is "`read_ts > safe_ts`" — semantically clean but the monotonic-forward setter cannot move `safe_ts` from `u128::MAX` to any lower value, so the first WAN batch can't set it; (b) default to `0`, gate is "`safe_ts > 0 && read_ts > safe_ts`" — one extra branch in the hot path but the monotonic setter works from defaults.

### Decision
Sentinel `0` = no gating. `replica_safe_ts: RwLock<u128>` initialised to `0`. The check on `/records/:id` and `/search` is `if safe > 0 && read_ts > safe { return 503 }`. The setter on `PUT /v1/admin/replica-safe-ts` and on the `LogReceiver` path is monotonic-forward (refuses to lower the watermark). Body uses JSON-string encoding for u128 since JavaScript can't safely round-trip values past 2^53.

### Consequences
- **Positive:** Operators can stand up a primary-only deployment and never touch `safe_ts`; cross-region gating activates the moment the first WAN batch lands. Test setup is trivial — fresh state means no gating, identical to single-region behaviour. The monotonic-forward invariant matches every other watermark in the system.
- **Negative:** One extra branch on every read. Negligible relative to the index walk. The `0` sentinel does mean "snapshot at `read_ts = 0`" is no longer a meaningful API call (which it never was — `0` is below every commit_ts).

---

## ADR-025: Reindex pipeline groups by `parent_doc_id`, reconstructs text, then re-chunks

**Status:** Accepted

### Context
The atomic-alias-swap reindex primitive (ADR-017) creates a sibling collection and points the alias at it once populated. The straight-copy version (v0.2.0) copies source records verbatim. The production version (v0.2.1) needs to re-chunk under a new strategy and optionally re-embed. The chunker takes parent-document text, not individual chunks, so we have three choices: (a) re-OCR the source PDFs from blob storage and re-chunk that; (b) store the unchunked parent text alongside every chunk and re-chunk that; (c) reconstruct the parent text by sorting the chunks by `chunk_index` and concatenating, then re-chunk that.

### Decision
(c). The reindex handler reads source records, groups them by the `parent_id_field` (default `parent_doc_id`), sorts each group by the `chunk_index_field` (default `chunk_index`), concatenates the `text_field` (default `content`) values, and runs `veda_core::chunking::chunk_markdown(reconstructed, &strategy)` on each. The new chunks land in the sibling with `parent_doc_id` preserved and `chunk_index` re-numbered. When `re_embed: true`, each new chunk's embedding comes from the registry's text-modality model; when no text-modality model is configured, the handler returns `424 FAILED_DEPENDENCY` instead of producing zero-vector placeholders.

### Consequences
- **Positive:** No dependency on the OCR pipeline staying healthy during reindex. No on-disk schema change (no separate parent-text storage). Lossless for any chunker that operates on flat text. Works the same for both text-only and multimodal collections (image chunks pass through untouched since they have no `chunk_index`).
- **Negative:** Original document structure (headings, page breaks) is lost — the concatenation produces a flat blob. Markdown-aware chunkers degrade to text-recursive behaviour. For collections where the source PDFs are still in blob storage, re-OCR (option (a)) would be lossless; that path is a future enhancement, not a v0.2.1 regression.

# Cross-Shard / Distributed Transactions: OSS Survey for Veda

**Date:** 2026-05-19
**Scope:** Identify open-source code Veda could vendor or adapt instead of writing a CockroachDB-class distributed transaction layer from scratch.
**Veda baseline:** Rust workspace; single cluster-wide Raft group (openraft); per-shard Raft groups planned but not implemented; vector + scalar storage layered on a custom segment format; REST gateway (axum).

---

## Executive Summary

After surveying the landscape, the realistic options are narrower than they first appear. **CockroachDB is effectively off the table for vendoring**: under the August 2024 license change (BSL -> CockroachDB Software License, CSL), the txn code is now source-available proprietary, with mandatory telemetry, license keys, and restrictions on benchmarking and derivative redistribution. Even the older BSL versions converted to Apache 2.0 only on a 3-year delay per-release, and the v25.x+ code we would actually want never reaches Apache 2.0 under the new license. Vendoring would require legal review and likely a paid license. **FoundationDB is too coupled** to its actor framework (Flow), bespoke deterministic scheduler, and C++ runtime to lift the transaction subsystem in isolation. **YugabyteDB is C++**, Spanner-style, and tightly bound to DocDB/RocksDB and the YB-Master service.

That leaves **TiKV** (`tikv/tikv`, Apache 2.0, Rust) as the only candidate that simultaneously satisfies license-permits-vendoring, same-language, and has a battle-tested Percolator implementation with async-commit and 1PC extensions. TiKV's transaction layer is not a clean, drop-in library — it is wired into `raftstore`, `pd_client`, `engine_traits` and the broader scheduler — but the *algorithmic core* (concurrency_manager, txn_types, src/storage/mvcc, src/storage/txn) is a coherent ~25-35k LoC unit that maps recognisably onto Veda's needs. Two adjunct assets help: the **pingcap/talent-plan** Percolator course (Apache 2.0) gives a ~2k LoC reference implementation we can read end-to-end as a pedagogical model, and TiKV's **sig-transaction** docs are essentially a free open design spec.

**Top recommendation:** Adapt TiKV's algorithms (Percolator + async-commit) into Veda using TiKV's design as a spec, vendor `txn_types` + `concurrency_manager` (~5-8k LoC) as starting points, and write the integration code ourselves. Estimated engineering effort: **12-18 engineer-weeks** for a single-region SI transaction layer covering 80% of the CockroachDB-class semantics list, vs. an estimated 30-50 weeks for a from-scratch implementation that re-derives the same lessons.

---

## Candidate Matrix

| Candidate | License | Language | Vendor-able? | LoC (txn core) | Maturity |
|---|---|---|---|---|---|
| **TiKV** (`tikv/tikv`) | Apache 2.0 | Rust | Yes (with care) | ~30-40k | Production (PingCAP TiDB, since 2017) |
| **CockroachDB** (`cockroachdb/cockroach`) | CSL (proprietary source-available, Nov 2024+) | Go | **No** (without commercial license) | ~50-80k | Production |
| **FoundationDB** (`apple/foundationdb`) | Apache 2.0 | C++ (Flow actor lang) | Partially (design-only) | ~100k+ | Production (Apple iCloud, Snowflake metadata) |
| **YugabyteDB** (`yugabyte/yugabyte-db`) | Apache 2.0 (core) | C++ | Theoretically | ~50k | Production |
| **Apache Omid** (`apache/phoenix-omid`) | Apache 2.0 | Java | No (HBase-coupled) | ~15-20k | Mature, narrow user base |
| **Apache Doris** (`apache/doris`) | Apache 2.0 | C++ | No (OLAP-specific 2PC) | ~5k | Production |
| **Apache Pegasus** | Apache 2.0 | C++ | No (no multi-key txns) | n/a | Production |
| **SurrealDB** (`surrealdb/surrealdb`) | BSL 1.1 | Rust | No (BSL + young txn code) | ~10k | Beta |
| **pingcap/talent-plan/percolator** | Apache 2.0 | Rust | Yes (reference only) | ~2k | Teaching code |

---

## Detailed Findings

### 1. TiKV (`github.com/tikv/tikv`)

**License:** Apache 2.0. Cleanly vendorable, modifiable, redistributable. No telemetry or use restrictions. CNCF graduated project.

**Language:** Rust (2021/2024 edition mix); the project even pioneered async Rust patterns we already use.

**Transaction shape.** TiKV implements **optimised Percolator** — Google's 2PC-on-top-of-Bigtable algorithm, extended with:

- **Parallel prewrite** — keys are batched and prewritten concurrently.
- **Async commit** — commit-ts is derived locally as `max(max_read_ts, start_ts, for_update_ts) + 1`, eliminating the second consensus round. This is TiKV's analogue to CockroachDB's Parallel Commits and gives the same single-round-trip property for distributed transactions.
- **1PC** — single-region transactions skip 2PC entirely and commit in the prewrite phase.
- **Pessimistic locking** — optional, for high-conflict workloads; mirrors the locktable concept in CockroachDB's `kv/kvserver/concurrency`.
- **Optimised Percolator** also embeds short values inline in the write record and supports point-read-without-timestamp.

**Code layout.** The txn-relevant code lives in three places:
- `src/storage/txn/` — command scheduler, latches, commands (Prewrite, Commit, Cleanup, ResolveLock, CheckTxnStatus, ~15+ command types), `actions/` (per-step state mutations).
- `src/storage/mvcc/` — MVCC reader/writer, lock detection, the actual key-write logic.
- `components/concurrency_manager/` (~3-4k LoC) — in-memory lock table, max-ts tracking. **Relatively self-contained**: depends on `txn_types`, `kvproto`, `pd_client`, `tikv_util` and external `crossbeam-skiplist`/`parking_lot`/`tokio`.
- `components/txn_types/` (~2-3k LoC) — Lock, Write, TimeStamp, Mutation types. Depends on `codec`, `kvproto`, `tikv_util`, `error_code`.

**Coupling.** This is the chief risk. TiKV's transaction code is not a tidy library — it talks to:
- `raftstore` for replication of writes (Veda already has openraft and would need to translate).
- `pd_client` for the Placement Driver — TiKV's centralised metadata + Timestamp Oracle. Veda has no PD; we either need to embed a TSO in the coordinator crate or use Hybrid Logical Clocks like Yugabyte/Cockroach.
- `engine_traits` / `engine_rocks` for storage. Veda has its own segment-based vector storage, so the MVCC writes need to be rerouted into our scalar column families (which today live in scalar-index code, not a generic KV engine).
- `kvproto` (the protobuf API). This sets the wire contract; vendoring requires either keeping the protos or rewriting against Veda's REST shapes.

**Test coverage.** Excellent. TiKV runs Jepsen periodically, has TLA+ specs for parts of async commit, and the codebase has thousands of integration tests in `tests/`. Maturity: production at PingCAP, AWS Aurora-Limitless, and as the storage for several Spanner-clones.

**Vendor cost estimate.**
- *Minimal* (vendor `txn_types` + `concurrency_manager` as utility crates, write our own scheduler): **6-8 weeks**, gives single-region SI but no cross-shard atomic writes.
- *Realistic* (port the Percolator state machine, write our own engine adapter, add async-commit later): **12-18 weeks** for snapshot-isolation cross-shard writes.
- *Full* (port async-commit, pessimistic lock, large-txn pipelining): **24-30 weeks**.

**Key risks.**
- TiKV's transaction code assumes a `kvproto` request shape and a per-region Raft state machine that is distinct from our segment model. The translation layer is non-trivial.
- Async commit specifically requires *all* MVCC reads to advance the local max-ts; this has to be threaded through every read path in Veda (search, fetch, scan). Easy to miss.
- TiKV's lock table is in-memory only — recovery on a node restart relies on scanning `CF_LOCK` from RocksDB. Veda needs an equivalent persistence story.

### 2. CockroachDB (`github.com/cockroachdb/cockroach`)

**License (critical).** As of **2024-11-18**, all of CockroachDB — including the `kv/kvserver/*`, `kv/kvclient/kvcoord/*`, and Parallel Commits code — is under the **CockroachDB Software License (CSL)**, a proprietary source-available license. It requires a paid license key for production use beyond a single node, enforces telemetry, restricts publishing benchmarks, and (per the FAQ) does not grant derivative-work redistribution rights. The earlier BSL-to-Apache-2.0 automatic conversion (3-year delay per minor version) was eliminated under the CSL transition. Older BSL revisions (e.g. v22.x, v23.x) are still subject to their original BSL terms and convert per their own schedules, but the modern Parallel-Commits-mature code is not.

**Verdict.** Cannot be vendored into Veda without a commercial license agreement with Cockroach Labs. Even the design ideas are fully documented elsewhere (the Parallel Commits blog, the RFC `docs/RFCS/20180324_parallel_commit.md`, the public TLA+ spec `docs/tla-plus/ParallelCommits/ParallelCommits.tla`), so we can re-implement the protocol without touching the code.

**Architecture worth borrowing as a spec.** Two CockroachDB packages are exceptionally well-designed:
- `pkg/kv/kvserver/concurrency` — concurrency manager combining `latchManager`, `lockTable`, `lockTableWaiter`, `txnWaitQueue`. Excellent reference design.
- `pkg/kv/kvclient/kvcoord` — the `TxnCoordSender` with its interceptor chain (TxnPipeliner, TxnCommitter, TxnSpanRefresher). This is a clean architecture for a transaction coordinator.

**Vendor cost.** N/A — re-implementation only. If we *do* re-implement from public docs, estimated 16-24 weeks for Parallel Commits + a concurrency manager, with the TLA+ spec as a verification anchor.

**Key risk.** Cockroach Labs has been litigious about close clones. Stick to public-paper-level design borrowing; do not copy code or comments.

### 3. FoundationDB (`github.com/apple/foundationdb`)

**License:** Apache 2.0.

**Language:** C++ with a custom actor-based language (`Flow`) that transpiles to C++. The transaction subsystem (`Master`, `Proxies`, `Resolvers`, `TransactionLogs`) is built on top of Flow's deterministic simulator — this is what makes FDB so reliable but also what makes it nearly impossible to extract a piece.

**What's interesting.** Strict serializability via OCC: read-set/write-set validation at commit time across all participants via `Resolvers`. The Resolver's conflict-detection algorithm (a key-range interval tree, sliding window) is elegant and would be reusable as a *design idea*. The deterministic simulation framework is the bigger lesson — it's why FoundationDB has had famously few correctness bugs.

**Vendor verdict.** Practical impossibility for code-level vendoring:
1. The protocol relies on Flow's actor abstractions; rewriting actor logic to Rust async/await is feasible but tedious.
2. The `Resolver` depends on the cluster topology being managed by FDB's `ClusterController` and `Master` roles — these are not separable.
3. The deterministic simulator (`fdbserver/SimulatedCluster.actor.cpp` + the `Sim2` runtime) is ~5-10k LoC of C++ tightly bound to Flow.

**Design borrowings.** Optimistic concurrency control with resolver-based commit-time validation is a viable alternative to Percolator for Veda — it has lower lock-contention cost. But implementing it from scratch is at least as much work as porting Percolator.

**Vendor cost:** Effectively infinite (full rewrite). Spec-borrowing cost: same as building Percolator from scratch.

### 4. YugabyteDB (`github.com/yugabyte/yugabyte-db`)

**License:** Apache 2.0 (core; enterprise features are not).

**Language:** C++. Hybrid Logical Clocks (HLC) for timestamps — like CockroachDB, not Spanner — so no atomic-clock dependency.

**Transaction model:** Spanner-inspired. Uses a per-shard Raft (DocDB) and a transaction status table. Multi-row writes go through a coordinator that pins a status record, writes provisional records to tablets, then atomically flips the status to COMMITTED.

**Coupling.** Very high. Conflict resolution lives in `src/yb/docdb/conflict_resolution.cc` and is tightly bound to DocDB's data layout (which embeds a sub-document model on top of RocksDB column families). Pulling this out without DocDB would mean reimplementing the data layout *and* the conflict logic — at which point you are writing Veda's own code anyway.

**Vendor cost:** Realistically 30+ weeks (and we would be porting C++ to Rust). Design-borrowing is fine; the YugabyteDB blog series on "Implementing Distributed Transactions the Google Way" is excellent reference material.

### 5. Apache Omid (`github.com/apache/phoenix-omid`)

**License:** Apache 2.0.

**Language:** Java.

**Relevance.** Omid is an open-source Percolator implementation for HBase, originally from Yahoo. ~15-20k LoC for the Transaction Manager + commit table + conflict checker. Snapshot isolation, lock-free conflict detection via a commit table.

**Vendor verdict.** Not directly usable — it is Java and tightly bound to HBase's coprocessor APIs. But Omid is a useful **second reference implementation** alongside TiKV; reading Omid's `HBaseTransactionManager.java` and `TSOServer` modules clarifies which Percolator components are essential vs. which are HBase-isms.

**Vendor cost:** N/A (read-only reference).

### 6. Apache Doris stream-load 2PC

**License:** Apache 2.0.

**Language:** C++.

**Relevance.** Doris implements 2PC for *bulk-load* transactions, not for general OLTP. The protocol is a PreCommit + Commit phase with an HTTP-level handshake (designed for Flink + exactly-once sinks). This is essentially a one-writer, table-level 2PC and lacks the multi-key, multi-shard guarantees Veda needs.

**Vendor verdict.** Not applicable. The semantics are too weak for our use case.

### 7. SurrealDB (`github.com/surrealdb/surrealdb`)

**License:** BSL 1.1.

**Language:** Rust.

**Relevance.** SurrealDB has its own KV abstraction (SurrealKV) and a transaction layer with snapshot isolation. Distributed-transaction support is recent and not yet production-mature; the 2PC across shards is reportedly less battle-tested than TiKV.

**Vendor verdict.** Avoid. BSL is more restrictive than Apache 2.0 (use-limitation clause), and the txn code is younger than TiKV's by 5+ years.

### 8. pingcap/talent-plan — Percolator course (Apache 2.0)

**Why mention it.** This is the *teaching* version of TiKV's Percolator: ~2k LoC of Rust implementing the full algorithm (TSO + MemoryStorage + transaction client + retry loop) with the algorithm split into readable lessons. **It is the cheapest way to onboard the team** before we touch the production TiKV code. Use it as the warm-up; do not vendor it as production code.

---

## Academic & Spec-Level Resources

These are not "code we can vendor" but they are the next-best alternative: protocols formally specified well enough to implement correctly.

1. **Percolator (Peng & Dabek, OSDI 2010)** — the foundational paper. Self-contained, well-written, 14 pages. Direct algorithm we would implement.
2. **TiKV "Optimised Percolator" deep-dive** (`tikv.org/deep-dive/distributed-transaction/optimized-percolator/`) — production refinements beyond the paper.
3. **TiKV sig-transaction async-commit design** (`tikv.github.io/sig-transaction/design/async-commit/`) — the open spec for the parallel-commits-equivalent in Percolator.
4. **CockroachDB Parallel Commits RFC** (`docs/RFCS/20180324_parallel_commit.md`) + **TLA+ spec** (`docs/tla-plus/ParallelCommits/ParallelCommits.tla`) — both originally BSL 1.1 -> partially Apache 2.0 reachable for older versions, both public. The TLA+ spec is verifiable locally; we can use it as a test oracle.
5. **Calvin (Thomson et al., SIGMOD 2012)** + **Detock (SIGMOD 2023)** — deterministic ordering as an *alternative* to 2PC. Avoids 2PC altogether by pre-ordering transactions across replicas. Interesting future direction but a different architectural commitment (requires a sequencer layer).
6. **HLC paper (Kulkarni et al., 2014)** — hybrid logical clocks. Use this if we want to avoid a centralised TSO.
7. **FoundationDB paper (SIGMOD 2021)** — for OCC/Resolver design ideas.

---

## Recommendation: Top 3 Picks Ranked by Realistic Adoption Cost

### Pick 1: TiKV-spec adoption + selective vendoring
- **Approach.** Read TiKV + Omid as references, use the talent-plan course to onboard. Vendor `txn_types` (~2k LoC) and `concurrency_manager` (~4k LoC) as starting utility crates under their Apache 2.0 licenses (clearly attributed). Write our own scheduler, MVCC reader/writer, and engine adapter from scratch, following TiKV's design but mapped onto our segment storage.
- **Cost:** ~12-18 engineer-weeks for SI cross-shard writes; +6-8 weeks for async commit.
- **Why it wins.** Same language, permissive license, mature reference, *and* we keep architectural flexibility for our segment model.

### Pick 2: Clean-room CockroachDB-style implementation
- **Approach.** Use CockroachDB's public design docs (Parallel Commits RFC, the concurrency package README, the TLA+ spec) as a spec, but no code copying. Implement against our own openraft-based per-shard groups.
- **Cost:** ~20-28 engineer-weeks. More expensive than Pick 1 because we do not have type-level starting points.
- **Why consider.** If we want HLC + Parallel Commits semantics specifically and prefer Cockroach's coordinator architecture (interceptor chain) over Percolator's lock-based model.

### Pick 3: FoundationDB-style OCC with resolvers
- **Approach.** Build a commit-time validation layer with range-interval conflict detection, OCC at the API. Avoids lock-table complexity. Reference the FDB paper and the Resolver source.
- **Cost:** ~24-32 engineer-weeks.
- **Why consider.** If our workload is read-heavy and write-conflicts are rare (likely true for vector DB ingest + query mix), OCC has lower latency than Percolator under low contention. Risk: high-conflict workloads (e.g. many concurrent updates to the same collection's metadata) regress badly.

---

## What Must Still Be Built (Even With Maximum Vendoring)

Even if we vendor every applicable line from TiKV, the following Veda-specific work is **not in any upstream**:

1. **Engine adapter.** TiKV writes to RocksDB column families (`CF_DEFAULT`, `CF_LOCK`, `CF_WRITE`). Veda has segment files + sealed segments + a custom WAL. The MVCC layer needs to be retargeted onto these. Estimate: 3-5 weeks.
2. **Per-shard Raft integration.** TiKV uses `raftstore`; Veda uses openraft. The Prewrite/Commit messages need to be proposals on our state machines, and the leader must own the local lock table. Estimate: 2-3 weeks.
3. **Timestamp source.** TiKV has the PD timestamp oracle; we either embed a TSO in coordinator or use HLC. HLC is simpler but requires per-node clock skew tolerance. Estimate: 2 weeks.
4. **Vector-aware semantics.** A transaction in Veda may need to atomically commit (a) a vector write, (b) a scalar-index update, (c) a blob-store upload reference. The standard Percolator key model assumes single-key conflicts; we need a story for blob references and async-uploaded vectors. This is novel design work; estimate 3-5 weeks.
5. **REST API integration.** Our existing `/v1/collections/.../points` endpoints do not have transaction handles. Adding an explicit `BEGIN/COMMIT` REST surface (or an implicit batched-write semantic) is a separate API design problem. Estimate: 1-2 weeks.
6. **Test surface.** Even with TiKV's test patterns to copy, we need our own Jepsen-style tests against our types, plus chaos tests against the segment storage, plus the talent-plan-style algorithmic tests. Estimate: 4-6 weeks.
7. **Observability hooks.** We have OTel; TiKV uses its own metrics. Tracing each phase of a cross-shard txn (prewrite, primary lock, async commit, async cleanup) needs to land in our existing `explain: true` infrastructure. Estimate: 1-2 weeks.

**Total integration tax over vendoring:** ~16-25 weeks. Combined with the Pick 1 vendoring cost (~12-18 weeks) this is **~28-43 engineer-weeks** for production-ready cross-shard SI semantics — still meaningfully cheaper than a from-scratch effort, but not the multi-week win that "just vendor TiKV" superficially suggests.

---

## Honest Caveats

- **We are not TiKV.** TiKV is a *general* transactional KV store. Veda's hot path is vector search; the txn surface is narrower (ingest, segment seal, schema mutation). It is plausible that a *scoped* solution — single-region SI with no cross-region, no pessimistic locks, no large-txn pipelining — covers 95% of Veda's needs at 30-40% of the cost. Before committing to full TiKV-feature parity, define which user-visible semantics we actually need.
- **License compliance is not optional.** Vendoring Apache 2.0 code requires preserving the LICENSE/NOTICE files and attributing the original authors in our own NOTICE. Several Apache projects (TiKV included) also have signed CLAs; we do not need to sign one for vendoring, only for upstreaming changes back.
- **The "1500-3000 word target" is a smell.** This survey deliberately stops at concrete artefacts rather than padding with secondary trees we already know do not apply (e.g. EventStore, Cassandra-LWT, etcd-tx — all checked, none give us cross-shard ACID with adaptable code). If a stakeholder wants a particular system surveyed in more depth, request it.

---

## Suggested Next Steps

1. Spend 2-3 days reading the `talent-plan/percolator` course end-to-end as a team primer.
2. Spike (1 week): Vendor `txn_types` + `concurrency_manager` into a fresh `veda-txn` crate; write a no-op scheduler that compiles and links against `veda-storage`. Establishes the integration tax empirically.
3. Decide commit-protocol axis: Percolator (lock-table heavy, snapshot-iso default) vs. OCC-Resolver (FDB-style, validation at commit) vs. Parallel-Commits-clean-room (HLC + Cockroach interceptor chain).
4. Write a Veda-specific design doc anchored to one of the three protocols, with a TLA+ sketch for the cross-shard commit decision. Aim for a 2-week design phase before any production code.

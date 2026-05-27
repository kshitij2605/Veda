# Design: Durable applier cursor + ShardLogRegistry (retire MultiGroupCoordinator)

**Date:** 2026-05-27
**Status:** Approved (brainstorming)
**Scope:** Two deferred follow-ups from `documentation/rearchitecture-overview.md`
(items "Persistent `last_applied` cursor for `LogApplier`" and
"`MultiGroupCoordinator` rewrite into a `ShardLogRegistry`"). Shipped as one
spec, committed as separable TDD steps.

## Goal

1. **#2 — Durable applier cursor.** A follower's `LogApplier` currently
   restarts tailing from seq=1 every boot (`LogTailer::new(log, 1)`),
   re-applying the entire log. Persist a durable cursor so it resumes from
   where it left off. Correct, not just fast: the cursor must track the
   *durable* (sealed) watermark, never the merely-applied one.
2. **#3 — ShardLogRegistry + delete MultiGroupCoordinator.** The per-shard
   data plane is an informal `RwLock<HashMap<String, Arc<ShardLog>>>` plus a
   free function `ensure_shard_log_for`. Promote it to a proper
   `ShardLogRegistry` type. The `MultiGroupCoordinator` it lived beside is
   vestigial post-Phase-E (it auto-spawns per-(collection,shard) single-node
   Raft groups whose state machines are never written to) — delete it.

No request-path behavior change. Followers recover faster; a class of dead
Raft instances disappears.

## Context (verified against HEAD)

- The live apply path is `veda-gateway/src/log_appliers.rs`
  (`spawn_one` + `apply_batch`), which uses `LogTailer` directly with a
  hardcoded `from = 1` and tracks no cursor. The `veda-storage::LogApplier`
  struct (which has a `last_applied` AtomicU64 + tests) is **not used by the
  gateway** — duplication, noted as a follow-up, not fixed here (YAGNI).
- `state.inner.raft: RwLock<Option<RaftCoordinator>>` is the real `_meta`
  consensus; all schema/membership/lease/ensemble commands go there. It is
  **out of scope** and untouched.
- `state.inner.multi_group` is referenced in exactly one functional place:
  `collections.rs:~494-523` auto-spawns `"{collection}:{shard}"` single-node
  Raft groups. Nothing ever calls `execute`/`query_state` on them
  (only `group_ids()` + `start_single_group()`). Phase E removed the
  `InsertRecord` command they were meant to carry. They are dead weight.
- `multi_group` / `MultiGroupCoordinator` / `META_GROUP` / `GroupId` have **no
  references** outside `multi_group.rs`, its `pub mod` line in
  `veda-coordinator/src/lib.rs`, and the two gateway files above — deletion is
  clean.
- The atomic-write pattern to reuse for the cursor file is
  `veda-storage/src/mvcc_wal.rs::checkpoint` (tmp + fsync + rename).

## #2 — Durable applier cursor

### Safety model (the crux)

A growing segment is in-memory until sealed. Therefore the persisted cursor
must track the **durable (sealed) watermark**, never the applied one — else a
crash before seal skips log entries whose data was lost.

Rule:

- The applier tracks `applied_max` (highest seq applied so far) in memory and
  persists `durable_seq` to disk.
- `apply_batch` detects a seal by comparing `sm.sealed_segments().len()`
  before and after the batch.
- **When a seal is observed during a batch, persist
  `durable_seq = applied_max recorded at the end of the PREVIOUS batch`**
  (call it `prev_batch_max`). The current batch's entries may straddle the
  seal boundary, so they are conservatively not counted yet.
- This is provably `<=` the true durable watermark, so we never persist a seq
  still living in a growing segment. Worst case: we under-claim and replay one
  extra batch on restart — harmless, because `insert` / `insert_mvcc` are
  idempotent on `(key, commit_ts)`.

Invariant: `persisted durable_seq <= max seq durably in a sealed segment`.

### Startup

Read `durable_seq` (0 if file absent or CRC/parse fails) →
`LogTailer::new(log, durable_seq + 1)`.

### Cursor file

`{shard_log_dir}/applied.cursor`: a single `u64`, framed `[crc32][u64 LE]`,
written atomically (tmp + fsync + rename) reusing the `mvcc_wal.rs`
checkpoint pattern. Torn/corrupt write → treated as absent → full replay
(safe).

One uniform rule for MVCC and non-MVCC shards. An MVCC-WAL-aware
"advance-every-tick" optimization is explicitly **out of scope** (adds a dual
code path for marginal restart-speed gain).

## #3 — ShardLogRegistry + delete MultiGroupCoordinator

### New type: `veda-log/src/registry.rs`

```rust
pub struct ShardLogRegistry {
    base_dir: PathBuf,
    logs: std::sync::RwLock<HashMap<(String, u32), Arc<ShardLog>>>,
}
impl ShardLogRegistry {
    pub fn new(base_dir: PathBuf) -> Self;
    pub fn get(&self, collection: &str, shard: u32) -> Option<Arc<ShardLog>>;
    pub fn get_or_open(&self, collection: &str, shard: u32) -> std::io::Result<Arc<ShardLog>>;
    pub fn list(&self) -> Vec<(String, u32)>;
    pub fn remove(&self, collection: &str, shard: u32);
}
```

- Sync internal `RwLock` (ShardLog ops are sync; `open` is a cheap file open).
- Dir layout `{base}/{collection}:{shard}/` — identical to today's
  `ensure_shard_log_for`, so on-disk layout is unchanged.

### Gateway rewiring

- `AppStateInner.shard_logs: RwLock<HashMap<String, Arc<ShardLog>>>` →
  `shard_logs: Arc<ShardLogRegistry>` (drops the outer tokio lock; the
  registry locks internally).
- `ensure_shard_log_for` body becomes `state.inner.shard_logs.get_or_open(..)`.
  Call sites in `insert_via_log` and `spawn_one` updated.
- Collection-drop path gains `registry.remove(..)` — resolves the existing
  `collections.rs:533` "MultiGroupCoordinator doesn't have a remove" note.

### Deletion

- Remove the `multi_group` field + always-on init (`state.rs:~744-753`) + the
  3 constructor inits.
- Remove the dead per-shard auto-spawn block (`collections.rs:~494-523`).
- Delete `crates/veda-coordinator/src/multi_group.rs` and `pub mod
  multi_group;` (`lib.rs:23`).
- `state.inner.raft` (the `_meta` Raft) is untouched.

## Testing (TDD, RED -> GREEN per step)

**#2**
- `applier_resumes_from_persisted_cursor_after_seal` — apply N, seal, persist,
  drop applier, reopen; confirm it resumes at `durable_seq + 1` and does not
  re-apply sealed entries.
- `applier_replays_unsealed_entries_after_crash` — apply into the growing
  segment without a seal, restart without a fresh cursor; confirm un-sealed
  entries are replayed (not lost).
- `cursor_torn_write_falls_back_to_full_replay` — corrupt the cursor file;
  confirm replay from seq=1.

**#3**
- `registry_get_or_open_caches_same_arc` — same `(collection, shard)` returns
  the same `Arc`.
- `registry_remove_drops_entry`.
- `registry_list_returns_open_shards`.
- Existing gateway insert/applier integration tests stay green (proves the
  swap is behavior-preserving). Workspace suite confirms the deletion left no
  dangling refs.

## File-by-file

- **New:** `crates/veda-log/src/registry.rs`;
  `crates/veda-storage/tests/applier_cursor_test.rs` (or extend the gateway
  applier tests — exact home decided in the plan); registry unit tests inline.
- **Modified:** `veda-log/src/lib.rs` (export registry);
  `veda-gateway/src/log_appliers.rs` (cursor);
  `veda-gateway/src/rest/state.rs` (registry field, drop multi_group);
  `veda-gateway/src/rest/records.rs` (`ensure_shard_log_for` -> registry);
  `veda-gateway/src/rest/collections.rs` (drop spawn, add remove-on-drop);
  `veda-coordinator/src/lib.rs` (drop mod).
- **Deleted:** `crates/veda-coordinator/src/multi_group.rs`.

## Out of scope

- Unifying the gateway apply loop with `veda-storage::LogApplier` (the
  duplicate). Noted for a future cleanup.
- MVCC-WAL-aware per-tick cursor advance.
- Any change to the `_meta` Raft (`state.inner.raft`).
- Multi-node transport / ensemble changes (separate phases).

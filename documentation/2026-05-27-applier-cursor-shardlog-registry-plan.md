# Durable Applier Cursor + ShardLogRegistry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist the follower log-applier cursor at the durable (sealed) watermark so restarts resume instead of replaying from seq 1, and promote the informal shard-log map into a `ShardLogRegistry` while deleting the vestigial `MultiGroupCoordinator`.

**Architecture:** `ShardLogRegistry` (new, in `veda-log`) owns the `{base}/{collection}/{shard:04}/` layout and the open-log cache. The gateway holds an `Arc<ShardLogRegistry>` instead of a raw `RwLock<HashMap>`. A small `veda-log::cursor` module provides atomic CRC-framed cursor read/write. The gateway applier (`spawn_one`) is refactored around a testable `ShardApplier` that persists the cursor at the previous-batch high-water mark whenever a seal makes prior entries durable. `MultiGroupCoordinator` and its dead per-shard Raft auto-spawn are deleted.

**Tech Stack:** Rust 2024, tokio, `crc32fast` + `bytes` (already `veda-log` deps). Build needs `PROTOC`. Every `cargo` command below assumes these are exported in the shell:

```bash
export PROTOC=/home/mac/mitrix69/load/.venv/lib/python3.12/site-packages/torch/bin/protoc
export PATH=/home/mac/.cargo/bin:$PATH
```

All work happens on `iitgpu11.hon.otsuka-shokai.co.jp:/home/mac/test/project-mdb` (the local Windows path is a stale mirror). Edit on the remote (or edit locally + scp); run `cargo` over SSH from the repo root.

Design reference: `documentation/2026-05-27-applier-cursor-shardlog-registry-design.md`.

---

## File Structure

- **Create** `crates/veda-log/src/registry.rs` — `ShardLogRegistry`: owns the on-disk layout convention + the open-log cache.
- **Create** `crates/veda-log/src/cursor.rs` — `read_cursor` / `write_cursor`: atomic, CRC-framed per-shard applier cursor file.
- **Modify** `crates/veda-log/src/lib.rs` — declare + export the two new modules.
- **Modify** `crates/veda-gateway/src/rest/state.rs` — swap `shard_logs` to `Arc<ShardLogRegistry>`; delete the `multi_group` field + its always-on init + import.
- **Modify** `crates/veda-gateway/src/rest/records.rs` — `ensure_shard_log` delegates to the registry.
- **Modify** `crates/veda-gateway/src/log_appliers.rs` — `ShardApplier` struct + cursor persistence; `apply_batch` returns a `BatchOutcome { max_seq, sealed }`; extract a testable `apply_entries`.
- **Modify** `crates/veda-gateway/src/rest/collections.rs` — delete the dead per-shard Raft auto-spawn block; add registry eviction + `_logs` dir purge to `drop_collection_impl`.
- **Create** `crates/veda-gateway/tests/applier_cursor_test.rs` — resume-after-seal + replay-unsealed integration tests.
- **Modify** `crates/veda-coordinator/src/lib.rs` — drop `pub mod multi_group;`.
- **Delete** `crates/veda-coordinator/src/multi_group.rs`.

Ordering: registry (Tasks 1-2), delete dead coordinator (Task 3), drop-cleanup (Task 4), cursor primitives (Task 5), applier wiring + tests (Task 6), verify + docs (Task 7). Each task leaves the workspace green and is committed independently.

API note (refined from the spec during planning): the registry surface is trimmed to what callers actually use — `new`, `get_or_open`, `remove_collection`, `purge_collection_dir`, `shard_dir`. The spec's `get`/`list` have no callers (YAGNI, dropped).

---

## Task 1: ShardLogRegistry type

**Files:**
- Create: `crates/veda-log/src/registry.rs`
- Modify: `crates/veda-log/src/lib.rs`

Owns the layout `{base}/{collection}/{shard:04}/` (gateway constructs it with `base = {data_dir}/_logs`, reproducing today's `{data_dir}/_logs/{scoped}/{shard:04}/`). Cache keyed by `(String, u32)`. Sync `RwLock` (ShardLog ops are sync; `open` is a cheap file open).

- [ ] **Step 1: Write the failing test**

Create `crates/veda-log/src/registry.rs` containing ONLY the test module for now:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn get_or_open_caches_same_arc() {
        let tmp = tempfile::tempdir().unwrap();
        let reg = ShardLogRegistry::new(tmp.path().join("_logs"));
        let a = reg.get_or_open("coll", 0).unwrap();
        let b = reg.get_or_open("coll", 0).unwrap();
        assert!(std::sync::Arc::ptr_eq(&a, &b), "same (collection,shard) returns the cached Arc");
    }

    #[test]
    fn shard_dir_matches_legacy_layout() {
        let reg = ShardLogRegistry::new(std::path::PathBuf::from("/data/_logs"));
        assert_eq!(
            reg.shard_dir("rag_anything", 3),
            std::path::PathBuf::from("/data/_logs/rag_anything/0003")
        );
    }

    #[test]
    fn remove_collection_drops_all_shards_for_that_collection() {
        let tmp = tempfile::tempdir().unwrap();
        let reg = ShardLogRegistry::new(tmp.path().join("_logs"));
        reg.get_or_open("a", 0).unwrap();
        reg.get_or_open("a", 1).unwrap();
        reg.get_or_open("b", 0).unwrap();
        assert_eq!(reg.remove_collection("a"), 2, "both shards of 'a' evicted");
        assert_eq!(reg.remove_collection("a"), 0, "idempotent: already gone");
    }

    #[test]
    fn purge_collection_dir_removes_on_disk_logs() {
        let tmp = tempfile::tempdir().unwrap();
        let reg = ShardLogRegistry::new(tmp.path().join("_logs"));
        reg.get_or_open("doomed", 0).unwrap(); // creates {base}/doomed/0000 on disk
        let dir = tmp.path().join("_logs").join("doomed");
        assert!(dir.exists());
        reg.purge_collection_dir("doomed").unwrap();
        assert!(!dir.exists(), "collection log dir removed");
    }
}
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd /home/mac/test/project-mdb && cargo test -p veda-log --lib registry:: 2>&1 | tail -20
```
Expected: FAIL — `cannot find type ShardLogRegistry in this scope`.

- [ ] **Step 3: Write the minimal implementation**

Prepend to `crates/veda-log/src/registry.rs` (above the test module):

```rust
//! Registry of per-shard [`ShardLog`]s. Owns the on-disk layout
//! convention `{base}/{collection}/{shard:04}/` and caches open
//! `Arc<ShardLog>` handles. Replaces the gateway's informal
//! `RwLock<HashMap<String, Arc<ShardLog>>>` + `ensure_shard_log`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use crate::shard_log::{ShardLog, ShardLogError};

pub struct ShardLogRegistry {
    base_dir: PathBuf,
    logs: RwLock<HashMap<(String, u32), Arc<ShardLog>>>,
}

impl ShardLogRegistry {
    /// `base_dir` is the directory under which per-shard log dirs
    /// live. The gateway passes `{data_dir}/_logs`.
    pub fn new(base_dir: impl AsRef<Path>) -> Self {
        Self {
            base_dir: base_dir.as_ref().to_owned(),
            logs: RwLock::new(HashMap::new()),
        }
    }

    /// On-disk directory for one shard's log:
    /// `{base}/{collection}/{shard:04}`.
    pub fn shard_dir(&self, collection: &str, shard: u32) -> PathBuf {
        self.base_dir.join(collection).join(format!("{shard:04}"))
    }

    /// Return the cached log for `(collection, shard)`, opening
    /// (and caching) it on first access.
    pub fn get_or_open(
        &self,
        collection: &str,
        shard: u32,
    ) -> Result<Arc<ShardLog>, ShardLogError> {
        let key = (collection.to_owned(), shard);
        if let Some(log) = self.logs.read().unwrap().get(&key) {
            return Ok(log.clone());
        }
        let dir = self.shard_dir(collection, shard);
        let log = Arc::new(ShardLog::open(&dir)?);
        let mut guard = self.logs.write().unwrap();
        Ok(guard.entry(key).or_insert(log).clone())
    }

    /// Evict every cached shard handle for `collection`. Returns
    /// the count evicted. Does not touch on-disk files.
    pub fn remove_collection(&self, collection: &str) -> usize {
        let mut guard = self.logs.write().unwrap();
        let before = guard.len();
        guard.retain(|(c, _), _| c != collection);
        before - guard.len()
    }

    /// Remove the on-disk log directory `{base}/{collection}` and
    /// all its shard subdirs. No-op if the dir is absent.
    pub fn purge_collection_dir(&self, collection: &str) -> std::io::Result<()> {
        let dir = self.base_dir.join(collection);
        if dir.exists() {
            std::fs::remove_dir_all(&dir)?;
        }
        Ok(())
    }
}
```

In `crates/veda-log/src/lib.rs`, add after `pub mod reader;`:

```rust
pub mod registry;
```

and after `pub use reader::LogTailer;`:

```rust
pub use registry::ShardLogRegistry;
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
cd /home/mac/test/project-mdb && cargo test -p veda-log --lib registry:: 2>&1 | tail -20
```
Expected: PASS — 4 tests.

- [ ] **Step 5: Commit**

```bash
cd /home/mac/test/project-mdb && git add crates/veda-log/src/registry.rs crates/veda-log/src/lib.rs && git commit -m "feat(veda-log): ShardLogRegistry owning shard-log layout + cache"
```

---

## Task 2: Swap the gateway to ShardLogRegistry

**Files:**
- Modify: `crates/veda-gateway/src/rest/state.rs` (field at ~221, import at ~21, inits at ~279/~342/~386)
- Modify: `crates/veda-gateway/src/rest/records.rs` (`ensure_shard_log`, ~858-895)

Behavior-preserving: same on-disk layout, same caching. The existing gateway suite is the regression guard.

- [ ] **Step 1: Change the field type + import in `state.rs`**

The import line currently reads `use veda_log::ShardLog;`. Change it to:

```rust
use veda_log::{ShardLog, ShardLogRegistry};
```

Replace the field (the line `pub shard_logs: RwLock<HashMap<String, Arc<ShardLog>>>,` plus its doc comment) with:

```rust
    /// Per-(collection, shard_id) append-only logs, behind a
    /// registry that owns the `{data_dir}/_logs/{scoped}/{shard:04}/`
    /// layout and caches open handles.
    pub shard_logs: Arc<ShardLogRegistry>,
```

- [ ] **Step 2: Update the three constructor inits**

In each of the three constructors (`new`, `with_auth`, and the third), replace:

```rust
                shard_logs: RwLock::new(HashMap::new()),
```

with:

```rust
                shard_logs: Arc::new(ShardLogRegistry::new(data_dir.join("_logs"))),
```

(`data_dir` is the `&Path` constructor param, in scope in all three.)

- [ ] **Step 3: Rewrite `ensure_shard_log` in `records.rs`**

Replace the entire body of the private `ensure_shard_log` function (the `key` build, the read-guard fast path, the slow-path `ShardLog::open`, and the write-guard insert) with a single registry call. The function becomes:

```rust
async fn ensure_shard_log(
    state: &AppState,
    scoped_collection: &str,
    shard_id: u32,
) -> Option<std::sync::Arc<veda_log::ShardLog>> {
    match state.inner.shard_logs.get_or_open(scoped_collection, shard_id) {
        Ok(log) => Some(log),
        Err(e) => {
            tracing::warn!(
                collection = %scoped_collection,
                shard = shard_id,
                error = %e,
                "ShardLog open failed via registry"
            );
            None
        }
    }
}
```

Leave the `pub(crate)` wrapper `ensure_shard_log_for` unchanged (it still calls `ensure_shard_log`).

- [ ] **Step 4: Build + run the gateway suite (regression)**

```bash
cd /home/mac/test/project-mdb && cargo test -p veda-gateway 2>&1 | tail -25
```
Expected: PASS — all existing gateway tests green. If a compile error mentions `HashMap`/`ShardLog` now unused in `state.rs`, remove the dangling import only if nothing else uses it (grep first).

- [ ] **Step 5: Commit**

```bash
cd /home/mac/test/project-mdb && git add crates/veda-gateway/src/rest/state.rs crates/veda-gateway/src/rest/records.rs && git commit -m "refactor(gateway): use ShardLogRegistry instead of inline shard-log map"
```

---

## Task 3: Delete MultiGroupCoordinator + dead per-shard Raft spawn

**Files:**
- Modify: `crates/veda-gateway/src/rest/collections.rs` (delete the `if let Some(mg) = ...` block, ~493-545)
- Modify: `crates/veda-gateway/src/rest/state.rs` (drop `multi_group` field ~183-191, import ~19, init ~744-753, three constructor `multi_group: RwLock::new(None),` lines)
- Modify: `crates/veda-coordinator/src/lib.rs` (drop `pub mod multi_group;`, ~23)
- Delete: `crates/veda-coordinator/src/multi_group.rs`

This is a pure deletion of code Phase E made unreachable (verified: no external refs to `MultiGroupCoordinator`/`META_GROUP`/`GroupId`). No new test; the regression guard is a clean build + green workspace.

- [ ] **Step 1: Delete the dead auto-spawn block in `collections.rs`**

In `reconcile_collections_from_raft`, delete the entire block that begins with the comment `// Auto-spawn per-(collection, shard_id) Raft groups in the` and the following `if let Some(mg) = state.inner.multi_group.read().await.clone() { ... }` (through its closing brace, immediately before the `// Materialize: in Raft, not in local.` comment). Remove the leading comment lines too. Nothing else in the function references `mg`.

- [ ] **Step 2: Delete the `multi_group` field, import, and init in `state.rs`**

- Remove the import line `use veda_coordinator::multi_group::MultiGroupCoordinator;`.
- Remove the `multi_group` field (its doc comment block + `pub multi_group: RwLock<Option<MultiGroupCoordinator>>,`).
- Remove the three `multi_group: RwLock::new(None),` lines in the constructors.
- Remove the always-on init block in `init_raft_from_env` (or wherever it lives, ~744-753): the lines from `// Always-on: a MultiGroupCoordinator under` through `*self.inner.multi_group.write().await = Some(mg);` (including the `let groups_dir = ...` and `let mg = MultiGroupCoordinator::new(&groups_dir);`).

- [ ] **Step 3: Delete the module**

```bash
cd /home/mac/test/project-mdb && git rm crates/veda-coordinator/src/multi_group.rs
```

In `crates/veda-coordinator/src/lib.rs`, delete the line `pub mod multi_group;`.

- [ ] **Step 4: Build the workspace + run both crates' suites**

```bash
cd /home/mac/test/project-mdb && cargo build -p veda-gateway -p veda-coordinator 2>&1 | tail -20
```
Expected: clean build, no `unresolved import` / `cannot find` errors. Then:

```bash
cd /home/mac/test/project-mdb && cargo test -p veda-coordinator -p veda-gateway 2>&1 | tail -25
```
Expected: PASS. (The `multi_group` unit tests are gone with the file; nothing else referenced them.)

- [ ] **Step 5: Commit**

```bash
cd /home/mac/test/project-mdb && git add crates/veda-gateway/src/rest/collections.rs crates/veda-gateway/src/rest/state.rs crates/veda-coordinator/src/lib.rs crates/veda-coordinator/src/multi_group.rs && git commit -m "refactor(coordinator): delete vestigial MultiGroupCoordinator + dead per-shard Raft spawn"
```

---

## Task 4: Evict + purge shard logs on collection drop

**Files:**
- Modify: `crates/veda-gateway/src/rest/collections.rs` (`drop_collection_impl`, ~1346-1358)

`drop_collection_impl` currently cleans `segment_managers`, `query_executors`, and the collection dir, but leaks the shard-log cache entry and the `_logs/{scoped}` dir. Wire in the registry's eviction + purge. The registry methods are already unit-tested (Task 1); this step is the integration.

- [ ] **Step 1: Add the cleanup calls**

In `drop_collection_impl`, immediately after the existing block that removes from `query_executors`:

```rust
    {
        let mut executors = state.inner.query_executors.write().await;
        executors.remove(&scoped);
    }
```

add:

```rust
    state.inner.shard_logs.remove_collection(&scoped);
    if let Err(e) = state.inner.shard_logs.purge_collection_dir(&scoped) {
        tracing::warn!(collection = %scoped, error = %e, "failed to purge shard-log dir on drop");
    }
```

- [ ] **Step 2: Build + run the gateway drop tests (regression)**

```bash
cd /home/mac/test/project-mdb && cargo test -p veda-gateway 2>&1 | tail -20
```
Expected: PASS — existing create/drop coverage green; new calls compile and don't regress drop behavior.

- [ ] **Step 3: Commit**

```bash
cd /home/mac/test/project-mdb && git add crates/veda-gateway/src/rest/collections.rs && git commit -m "fix(gateway): evict + purge shard logs on collection drop"
```

---

## Task 5: Atomic CRC-framed cursor primitives

**Files:**
- Create: `crates/veda-log/src/cursor.rs`
- Modify: `crates/veda-log/src/lib.rs`

`write_cursor` writes `[crc32 u32 LE][seq u64 LE]` atomically (tmp + fsync + rename). `read_cursor` returns the seq, or `0` if the file is missing, short, or CRC-mismatched (→ full replay, safe).

- [ ] **Step 1: Write the failing test**

Create `crates/veda-log/src/cursor.rs` containing ONLY the test module:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrips_a_seq() {
        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join("applied.cursor");
        write_cursor(&p, 42).unwrap();
        assert_eq!(read_cursor(&p), 42);
    }

    #[test]
    fn missing_file_reads_zero() {
        let tmp = tempfile::tempdir().unwrap();
        assert_eq!(read_cursor(tmp.path().join("nope.cursor")), 0);
    }

    #[test]
    fn corrupt_file_reads_zero() {
        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join("applied.cursor");
        write_cursor(&p, 99).unwrap();
        // Flip a payload byte so the CRC no longer matches.
        let mut bytes = std::fs::read(&p).unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 0xFF;
        std::fs::write(&p, &bytes).unwrap();
        assert_eq!(read_cursor(&p), 0, "CRC mismatch falls back to 0");
    }

    #[test]
    fn truncated_file_reads_zero() {
        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join("applied.cursor");
        write_cursor(&p, 7).unwrap();
        let bytes = std::fs::read(&p).unwrap();
        std::fs::write(&p, &bytes[..bytes.len() - 3]).unwrap(); // torn tail
        assert_eq!(read_cursor(&p), 0);
    }
}
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd /home/mac/test/project-mdb && cargo test -p veda-log --lib cursor:: 2>&1 | tail -20
```
Expected: FAIL — `cannot find function write_cursor`.

- [ ] **Step 3: Write the minimal implementation**

Prepend to `crates/veda-log/src/cursor.rs`:

```rust
//! Per-shard applier cursor: the highest log `seq_id` that is
//! durably reflected in the local SegmentManager. Written
//! atomically (tmp + fsync + rename), framed `[crc32][seq u64]`.
//! A missing or torn/corrupt file reads as 0 (full replay — safe,
//! because apply is idempotent on `(key, commit_ts)`).

use std::io::Write;
use std::path::Path;

/// Read the persisted cursor, or 0 if absent/short/corrupt.
pub fn read_cursor(path: impl AsRef<Path>) -> u64 {
    let bytes = match std::fs::read(path.as_ref()) {
        Ok(b) => b,
        Err(_) => return 0,
    };
    if bytes.len() != 12 {
        return 0;
    }
    let crc_stored = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    let payload = &bytes[4..12];
    if crc32fast::hash(payload) != crc_stored {
        return 0;
    }
    let mut seq = [0u8; 8];
    seq.copy_from_slice(payload);
    u64::from_le_bytes(seq)
}

/// Write the cursor atomically: tmp file + fsync + rename.
pub fn write_cursor(path: impl AsRef<Path>, seq: u64) -> std::io::Result<()> {
    let path = path.as_ref();
    let payload = seq.to_le_bytes();
    let crc = crc32fast::hash(&payload);
    let mut frame = Vec::with_capacity(12);
    frame.extend_from_slice(&crc.to_le_bytes());
    frame.extend_from_slice(&payload);

    let tmp = path.with_extension("cursor.tmp");
    {
        let mut f = std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&tmp)?;
        f.write_all(&frame)?;
        f.flush()?;
        f.sync_data()?;
    }
    std::fs::rename(&tmp, path)
}
```

In `crates/veda-log/src/lib.rs`, add after `pub mod cursor;`-worthy spot (next to `pub mod registry;`):

```rust
pub mod cursor;
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
cd /home/mac/test/project-mdb && cargo test -p veda-log --lib cursor:: 2>&1 | tail -20
```
Expected: PASS — 4 tests.

- [ ] **Step 5: Commit**

```bash
cd /home/mac/test/project-mdb && git add crates/veda-log/src/cursor.rs crates/veda-log/src/lib.rs && git commit -m "feat(veda-log): atomic CRC-framed applier cursor primitives"
```

---

## Task 6: ShardApplier with seal-watermark cursor persistence

**Files:**
- Modify: `crates/veda-gateway/src/log_appliers.rs`
- Create: `crates/veda-gateway/tests/applier_cursor_test.rs`

Refactor the apply loop around a testable `ShardApplier`. `apply_entries` (pure over `&mut SegmentManager`) applies a batch and returns the max seq. `apply_batch` locks the map, calls `apply_entries`, detects a seal via `sealed_segments().len()` delta, and returns `BatchOutcome`. `ShardApplier::tick` persists the cursor at `prev_batch_max` whenever a seal is observed (the previous batch's entries are now durable; the current batch may straddle the seal, so it is not counted yet). `spawn_one` loops `tick`.

- [ ] **Step 1: Write the failing integration test**

Create `crates/veda-gateway/tests/applier_cursor_test.rs`:

```rust
//! #2 — durable applier cursor: resume-after-seal + replay-unsealed.

use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use veda_core::record::{FieldValue, Record};
use veda_core::schema::{CollectionSchema, FieldDefinition};
use veda_core::types::DistanceMetric;
use veda_core::vector::DenseVector;
use veda_log::format::LogOp;
use veda_time::Hlc;

use veda_gateway::log_appliers::ShardApplier;
use veda_gateway::rest::state::AppState;

fn schema(name: &str) -> CollectionSchema {
    CollectionSchema::builder(name)
        .add_field(FieldDefinition::vector("v", 2, DistanceMetric::Cosine))
        .shard_count(1)
        .build()
        .unwrap()
}

fn record() -> Record {
    let mut r = Record::new();
    r.set("v", FieldValue::Vector(DenseVector::new(vec![1.0, 0.0])));
    r
}

// Register a SegmentManager for (collection, 0) with a tiny seal
// threshold so a few inserts force a seal.
async fn register_sm(state: &AppState, collection: &str, seal_bytes: usize) {
    let sm = veda_storage::segment_manager::SegmentManager::new(
        schema(collection),
        0,
        state.inner.data_dir.join(collection),
        seal_bytes,
    )
    .unwrap();
    let mut managers = state.inner.segment_managers.write().await;
    managers
        .entry(collection.to_owned())
        .or_default()
        .insert(0, sm);
}

async fn append_n(state: &AppState, collection: &str, n: u64) {
    let log = state.inner.shard_logs.get_or_open(collection, 0).unwrap();
    for i in 1..=n {
        let r = record();
        let payload = Bytes::from(serde_json::to_vec(&r).unwrap());
        let key = Bytes::copy_from_slice(r.id().as_bytes());
        log.append_one(Hlc::new(i, 0), 1, key, LogOp::Put, payload)
            .unwrap();
    }
}

#[tokio::test]
async fn resumes_from_persisted_cursor_after_seal() {
    let tmp = tempfile::tempdir().unwrap();
    let state = AppState::new(tmp.path());
    register_sm(&state, "coll", 256).await; // tiny -> seals quickly
    append_n(&state, "coll", 20).await;

    let mut applier = ShardApplier::open(&state, "coll", 0).await.unwrap();
    // Drive enough ticks to apply all 20 and trigger at least one seal.
    for _ in 0..20 {
        applier.tick(&state, "coll", 0).await.unwrap();
    }

    let cursor_path = state.inner.shard_logs.shard_dir("coll", 0).join("applied.cursor");
    let persisted = veda_log::cursor::read_cursor(&cursor_path);
    assert!(persisted > 0, "a seal should have advanced the durable cursor");

    // A fresh applier (simulating restart) resumes at persisted+1.
    let applier2 = ShardApplier::open(&state, "coll", 0).await.unwrap();
    assert_eq!(applier2.next_seq(), persisted + 1, "resume from cursor+1, not 1");
}

#[tokio::test]
async fn replays_unsealed_entries_after_crash() {
    let tmp = tempfile::tempdir().unwrap();
    let state = AppState::new(tmp.path());
    register_sm(&state, "coll", 64 * 1024 * 1024).await; // huge -> never seals
    append_n(&state, "coll", 5).await;

    let mut applier = ShardApplier::open(&state, "coll", 0).await.unwrap();
    let applied = applier.tick(&state, "coll", 0).await.unwrap();
    assert_eq!(applied, 5);

    // No seal happened, so the cursor must still be 0 (those entries
    // live only in the in-memory growing segment, not durable).
    let cursor_path = state.inner.shard_logs.shard_dir("coll", 0).join("applied.cursor");
    assert_eq!(veda_log::cursor::read_cursor(&cursor_path), 0);

    // Restart: a fresh applier resumes at 1 and re-applies all 5.
    let applier2 = ShardApplier::open(&state, "coll", 0).await.unwrap();
    assert_eq!(applier2.next_seq(), 1);
    let _ = Duration::from_millis(1); // silence unused import if needed
}
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd /home/mac/test/project-mdb && cargo test -p veda-gateway --test applier_cursor_test 2>&1 | tail -20
```
Expected: FAIL — `ShardApplier` not found / `next_seq` not found.

- [ ] **Step 3: Implement `ShardApplier` + refactor `apply_batch` in `log_appliers.rs`**

Add near the top of `log_appliers.rs` (after the existing imports), the testable pieces. First, the batch outcome + the pure `apply_entries`:

```rust
use std::path::PathBuf;
use veda_log::reader::LogTailer;
use veda_storage::segment_manager::SegmentManager;

/// Result of applying one batch to a SegmentManager.
pub(crate) struct BatchOutcome {
    pub max_seq: u64,
    pub sealed: bool,
}

/// Apply a batch of entries to `sm`; return the highest seq_id
/// applied. Pure over the SegmentManager — no AppState locking.
fn apply_entries(sm: &mut SegmentManager, batch: &[LogEntryV2]) -> Result<u64, String> {
    let mut max_seq = 0u64;
    for entry in batch {
        match entry.op {
            LogOp::Put => {
                let record: Record = serde_json::from_slice(&entry.payload)
                    .map_err(|e| format!("decode record @ seq={}: {e}", entry.seq_id))?;
                if sm.has_mvcc() {
                    let commit_ts =
                        ((entry.hlc_ts.physical_ns as u128) << 32) | (entry.hlc_ts.logical as u128);
                    sm.insert_mvcc(commit_ts, record).map_err(|e| format!("insert_mvcc: {e}"))?;
                } else {
                    sm.insert(record).map_err(|e| format!("insert: {e}"))?;
                }
            }
            LogOp::Delete => {
                let rid = uuid_from_key(&entry.key)?;
                if sm.has_mvcc() {
                    let commit_ts =
                        ((entry.hlc_ts.physical_ns as u128) << 32) | (entry.hlc_ts.logical as u128);
                    sm.delete_mvcc(commit_ts, &rid).map_err(|e| format!("delete_mvcc: {e}"))?;
                } else {
                    sm.delete_record(rid).map_err(|e| format!("delete_record: {e}"))?;
                }
            }
            other => {
                tracing::warn!(op = ?other, seq_id = entry.seq_id, "log_applier: skipping unrecognised op");
            }
        }
        max_seq = entry.seq_id;
    }
    Ok(max_seq)
}
```

Rewrite `apply_batch` to use `apply_entries` and report whether a seal happened:

```rust
async fn apply_batch(
    state: &AppState,
    collection: &str,
    shard_id: u32,
    batch: Vec<LogEntryV2>,
) -> Result<BatchOutcome, String> {
    let mut managers = state.inner.segment_managers.write().await;
    let shard_map = managers
        .get_mut(collection)
        .ok_or_else(|| format!("collection '{collection}' not in segment_managers"))?;
    let sm = shard_map
        .get_mut(&shard_id)
        .ok_or_else(|| format!("shard {shard_id} not in '{collection}'"))?;
    let sealed_before = sm.sealed_segments().len();
    let max_seq = apply_entries(sm, &batch)?;
    let sealed = sm.sealed_segments().len() > sealed_before;
    Ok(BatchOutcome { max_seq, sealed })
}
```

Add the `ShardApplier`:

```rust
/// Per-shard applier: tails the shard log and persists a durable
/// cursor at the sealed watermark. One `tick` drains whatever the
/// tailer can give right now.
pub struct ShardApplier {
    tailer: LogTailer,
    cursor_path: PathBuf,
    /// Highest seq applied as of the END of the previous batch.
    /// On observing a seal we persist THIS (prior batches are now
    /// durable; the current batch may straddle the seal boundary).
    prev_batch_max: u64,
}

impl ShardApplier {
    /// Open the applier for `(collection, shard)`: read the
    /// persisted cursor and resume tailing from `cursor + 1`.
    /// Returns `None` if the shard log can't be opened.
    pub async fn open(state: &AppState, collection: &str, shard_id: u32) -> Option<Self> {
        let log = ensure_shard_log_for(state, collection, shard_id).await?;
        let cursor_path = state
            .inner
            .shard_logs
            .shard_dir(collection, shard_id)
            .join("applied.cursor");
        let start = veda_log::cursor::read_cursor(&cursor_path);
        Some(Self {
            tailer: LogTailer::new(log, start + 1),
            cursor_path,
            prev_batch_max: start,
        })
    }

    /// The next seq_id this applier will attempt to read. Test
    /// observability for resume behavior.
    pub fn next_seq(&self) -> u64 {
        self.prev_batch_max + 1
    }

    /// Drain one batch into the SegmentManager; persist the cursor
    /// at the durable watermark if this batch caused a seal.
    /// Returns the number of entries applied.
    pub async fn tick(&mut self, state: &AppState, collection: &str, shard_id: u32) -> Result<usize, String> {
        let batch = self
            .tailer
            .next_batch()
            .map_err(|e| format!("tailer next_batch: {e}"))?;
        if batch.is_empty() {
            return Ok(0);
        }
        let n = batch.len();
        let outcome = apply_batch(state, collection, shard_id, batch).await?;
        if outcome.sealed {
            if let Err(e) = veda_log::cursor::write_cursor(&self.cursor_path, self.prev_batch_max) {
                tracing::warn!(error = %e, "log_applier: cursor persist failed");
            }
        }
        self.prev_batch_max = outcome.max_seq;
        Ok(n)
    }
}
```

Now replace the body of `spawn_one` so it uses `ShardApplier` (drop the inline `LogTailer` + match):

```rust
pub fn spawn_one(
    state: AppState,
    collection: String,
    shard_id: u32,
    tick: Duration,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        let mut applier = match ShardApplier::open(&state, &collection, shard_id).await {
            Some(a) => a,
            None => {
                tracing::warn!(collection = %collection, shard = shard_id, "log_applier: open failed; not spawning");
                return;
            }
        };
        let mut interval = tokio::time::interval(tick);
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            interval.tick().await;
            match applier.tick(&state, &collection, shard_id).await {
                Ok(0) => {}
                Ok(n) => tracing::debug!(applied = n, next = applier.next_seq(), "log_applier tick"),
                Err(e) => tracing::warn!(error = %e, "log_applier tick failed"),
            }
        }
    })
}
```

Ensure `log_appliers` is reachable from tests: in `crates/veda-gateway/src/lib.rs`, confirm `pub mod log_appliers;` exists (it must, since tests import `veda_gateway::log_appliers::ShardApplier`). If it is currently `mod log_appliers;` or `pub(crate)`, change it to `pub mod log_appliers;`. Likewise confirm `pub mod rest;` and that `rest::state` is reachable (`veda_gateway::rest::state::AppState`).

- [ ] **Step 4: Run the test to verify it passes**

```bash
cd /home/mac/test/project-mdb && cargo test -p veda-gateway --test applier_cursor_test 2>&1 | tail -25
```
Expected: PASS — 2 tests. Then run the full gateway suite to confirm no regression in the existing applier path:

```bash
cd /home/mac/test/project-mdb && cargo test -p veda-gateway 2>&1 | tail -25
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/mac/test/project-mdb && git add crates/veda-gateway/src/log_appliers.rs crates/veda-gateway/src/lib.rs crates/veda-gateway/tests/applier_cursor_test.rs && git commit -m "feat(gateway): durable applier cursor at the sealed watermark"
```

---

## Task 7: Verify workspace + update docs

**Files:**
- Modify: `documentation/rearchitecture-overview.md` (deferred list)
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Full workspace test**

```bash
cd /home/mac/test/project-mdb && cargo test --workspace --lib 2>&1 | tail -30
```
Expected: PASS across all crates. Then the gateway integration tests:

```bash
cd /home/mac/test/project-mdb && cargo test -p veda-gateway -p veda-log -p veda-storage -p veda-coordinator 2>&1 | tail -30
```
Expected: PASS.

- [ ] **Step 2: Update the deferred list in `rearchitecture-overview.md`**

In the section `What's deliberately NOT here yet (tracked follow-ups, not in v0.2.1):`, remove the two now-done bullets:
- the `Persistent last_applied cursor for LogApplier` bullet
- the `MultiGroupCoordinator rewrite into a ShardLogRegistry` bullet

Add a line under the done-items table (or a short note) recording that both landed: `ShardLogRegistry + durable applier cursor — done 2026-05-27`.

- [ ] **Step 3: Add a CHANGELOG entry**

Under the most recent unreleased/`[0.2.2]` heading (create an `### Added`/`### Changed` block if needed):

```markdown
- **Durable applier cursor** — the per-shard log applier persists a CRC-framed cursor at the sealed (durable) watermark and resumes from it on restart, instead of replaying the whole log from seq 1.
- **ShardLogRegistry** — the per-shard log map is now a first-class `veda-log::ShardLogRegistry`; the vestigial `MultiGroupCoordinator` (dead per-shard Raft groups, unreachable since Phase E) is deleted. Collection drop now evicts + purges shard logs.
```

- [ ] **Step 4: Commit**

```bash
cd /home/mac/test/project-mdb && git add documentation/rearchitecture-overview.md CHANGELOG.md && git commit -m "docs: record applier-cursor + ShardLogRegistry; drop deferred items"
```

---

## Self-Review notes (for the implementer)

- **Spec coverage:** Task 1 (registry type) + Task 2 (swap) + Task 4 (drop cleanup) cover spec #3's registry; Task 3 covers the deletion; Tasks 5-6 cover spec #2's cursor with the seal-watermark rule and torn-write fallback. All spec testing bullets map to a test (registry caching/remove/purge → Task 1; cursor roundtrip/corrupt → Task 5; resume-after-seal/replay-unsealed → Task 6).
- **Type consistency:** `get_or_open` returns `Result<Arc<ShardLog>, ShardLogError>`; `ensure_shard_log` maps `Err` → `None` (Task 2). `BatchOutcome { max_seq, sealed }` is produced by `apply_batch` and consumed by `ShardApplier::tick` (Task 6). `read_cursor`/`write_cursor` signatures match across Tasks 5 and 6.
- **Watermark safety invariant:** persisted cursor ≤ max seq durably sealed (Task 6 persists `prev_batch_max` only on seal). Under-claiming is safe (idempotent re-apply); over-claiming never happens.
- **If `data_dir` is not in scope** in some constructor variant, build the registry from the same path that constructor passes to `SavedObjectStore::new` / `LocalBlobStore::new` (all three use `data_dir`).

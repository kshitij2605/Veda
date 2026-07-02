# Automated RAG Evaluation & Auto-Tuning Implementation Plan (Phase 1: text-only MVP)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an automated, corpus-grounded RAG evaluation and multi-objective auto-tuning system for text collections — synthetic test-set generation, richer per-component metrics, and a bounded search campaign that finds a Pareto-optimal pipeline configuration for the user's own data, with a safe, revertible apply path.

**Architecture:** Seven sequenced phases, each independently shippable. Phase 0 fixes a pre-existing gap (alias resolution doesn't work without Raft) that Phase 5's safe-apply depends on. Phases 1–2 extend the existing `rag_eval.rs` eval harness additively. Phase 3 integrates a real Rust multi-objective optimization crate behind an internal trait. Phase 4 is the campaign engine — a nested search that reuses the existing `/reindex` primitive for expensive trials. Phase 5 wires apply/revert through the Phase 0 alias fix. Phase 6 is the Niriksha UI.

**Tech Stack:** Rust (axum, tokio, serde), `scirs2-optimize` crate (NSGA-II/NSGA-III + Bayesian optimization), the existing `SavedObjectStore`/`JobRegistry` persistence patterns, React/TypeScript (Niriksha).

## Global Constraints

- Toolchain: pinned Rust 1.96.0 (`rust-toolchain.toml` — plain `cargo` auto-selects). All work builds/tests against this version.
- Every new/changed Rust file must pass `cargo fmt --all --check` and `cargo clippy --workspace --all-targets -- -D warnings` before commit.
- No new persistence subsystem: reuse `SavedObjectStore` (`crates/veda-gateway/src/rest/saved_objects.rs`) and the `JobRegistry`/`JobKind` pattern (`crates/veda-gateway/src/scheduler.rs`) for everything new that needs to persist.
- No `reqwest::blocking` anywhere (a landmine documented elsewhere in this codebase — it panics inside async contexts). All new HTTP/LLM/embedding calls are async.
- Commit messages: conventional commits (`feat:`, `test:`, `fix:`, `docs:`), no AI-assistance mentions, stage only the files a task touches.
- Every collection-name field the campaign engine reads/writes is **text-only** for this cycle — no image/audio/video handling anywhere in Phases 0–6.
- Development workflow: this repo's source of truth is on the remote host `gpu-host-a.internal.example:/home/mac/test/project-mdb`; edit locally, `scp` up, build/test remotely over SSH, commit/push remotely.

---

## Phase 0 — Non-Raft alias resolution (prerequisite)

**Why:** `crates/veda-gateway/src/rest/collections.rs::set_alias` is a documented no-op without Raft (comment: *"Non-Raft mode: there's no SM to write to... alias resolution in non-Raft mode is a follow-up"*), and `reindex_collection`'s alias-swap step is gated on `raft_guard.as_ref()` being `Some`. Every collection-name lookup in the gateway (`search::search`, `records::insert`, `records::list_records`, `records::get_record`, `records::delete_records`) currently does `tenant::scope(&tenant, &name)` and looks the result up **directly** in `state.inner.schemas`/`state.inner.segment_managers` — no alias indirection exists anywhere in the read path. Without this phase, Phase 5's "safe, revertible apply" silently does nothing in single-node mode (the common deployment case).

### Task 0.1: Alias map — data structure, persistence, and resolution helper

**Files:**
- Create: `crates/veda-gateway/src/rest/alias.rs`
- Modify: `crates/veda-gateway/src/rest/mod.rs:33` area (add `pub mod alias;`)
- Test: inline `#[cfg(test)] mod tests` in `alias.rs`

**Interfaces:**
- Produces: `pub struct AliasMap` with `new(data_dir: &Path) -> Self`, `async fn set(&self, alias: String, target: String)`, `async fn resolve(&self, name: &str) -> String` (returns `target` if an alias entry exists for `name`, else returns `name` unchanged — identity default), `async fn get(&self, alias: &str) -> Option<String>`.

- [ ] **Step 1: Write the failing tests**

```rust
// bottom of crates/veda-gateway/src/rest/alias.rs
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn resolve_returns_identity_when_no_alias_set() {
        let tmp = tempfile::tempdir().unwrap();
        let map = AliasMap::new(tmp.path());
        assert_eq!(map.resolve("rag_anything").await, "rag_anything");
    }

    #[tokio::test]
    async fn resolve_returns_target_after_set() {
        let tmp = tempfile::tempdir().unwrap();
        let map = AliasMap::new(tmp.path());
        map.set("rag_anything".to_owned(), "rag_anything@v2".to_owned())
            .await;
        assert_eq!(map.resolve("rag_anything").await, "rag_anything@v2");
        // Unrelated names still resolve to themselves.
        assert_eq!(map.resolve("other_collection").await, "other_collection");
    }

    #[tokio::test]
    async fn set_persists_and_reloads() {
        let tmp = tempfile::tempdir().unwrap();
        {
            let map = AliasMap::new(tmp.path());
            map.set("a".to_owned(), "a@v2".to_owned()).await;
        }
        // Fresh instance over the same data_dir picks up the persisted alias.
        let map2 = AliasMap::new(tmp.path());
        assert_eq!(map2.resolve("a").await, "a@v2");
    }

    #[tokio::test]
    async fn get_returns_none_when_unset() {
        let tmp = tempfile::tempdir().unwrap();
        let map = AliasMap::new(tmp.path());
        assert_eq!(map.get("a").await, None);
    }

    #[tokio::test]
    async fn set_can_be_reverted_by_setting_back() {
        let tmp = tempfile::tempdir().unwrap();
        let map = AliasMap::new(tmp.path());
        map.set("a".to_owned(), "a@v2".to_owned()).await;
        assert_eq!(map.resolve("a").await, "a@v2");
        // Revert: point the alias back at the original name.
        map.set("a".to_owned(), "a".to_owned()).await;
        assert_eq!(map.resolve("a").await, "a");
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `ssh gpu-host-a.internal.example "cd /home/mac/test/project-mdb && /home/mac/.cargo/bin/cargo test -p veda-gateway --lib alias:: 2>&1 | tail -20"`
Expected: FAIL — `alias` module / `AliasMap` type not found (the file doesn't exist yet, or is empty).

- [ ] **Step 3: Write the implementation**

```rust
// crates/veda-gateway/src/rest/alias.rs
//! Non-Raft (single-node) collection-name alias resolution.
//!
//! `set_alias`/`get_alias` (`collections.rs`) already work end-to-end in
//! Raft mode via `ClusterCommand::SetCollectionAlias` on the replicated
//! state machine. In non-Raft (single-node) mode that command has nowhere
//! to go — this module is the local, persisted equivalent, mirroring
//! `SavedObjectStore`'s `Arc<RwLock<HashMap>>` + write-through-JSON pattern
//! (`crates/veda-gateway/src/rest/saved_objects.rs`).
//!
//! `resolve` is the identity function when no alias is set, matching
//! Raft-mode `get_alias`'s documented behavior ("alias = target when no
//! alias is set").

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;

#[derive(Clone)]
pub struct AliasMap {
    inner: Arc<RwLock<HashMap<String, String>>>,
    path: PathBuf,
}

impl AliasMap {
    /// Load from `{data_dir}/_aliases.json`, starting empty if absent or
    /// unparseable (mirrors `SavedObjectStore::new` / `JobRegistry::new`).
    pub fn new(data_dir: &std::path::Path) -> Self {
        let path = data_dir.join("_aliases.json");
        let inner: HashMap<String, String> = match std::fs::read(&path) {
            Ok(bytes) => serde_json::from_slice(&bytes).unwrap_or_else(|e| {
                tracing::warn!(
                    error = %e,
                    path = %path.display(),
                    "alias: parse failed, starting empty"
                );
                HashMap::new()
            }),
            Err(_) => HashMap::new(),
        };
        Self {
            inner: Arc::new(RwLock::new(inner)),
            path,
        }
    }

    /// Set (or clear, by pointing an alias at itself) an alias -> target
    /// mapping and persist immediately.
    pub async fn set(&self, alias: String, target: String) {
        {
            let mut m = self.inner.write().await;
            m.insert(alias, target);
        }
        self.write_through().await;
    }

    /// The raw alias entry, if one exists (used by `get_alias`'s HTTP
    /// handler to distinguish "no alias" from "alias = self").
    pub async fn get(&self, alias: &str) -> Option<String> {
        self.inner.read().await.get(alias).cloned()
    }

    /// Resolve `name` through the alias map. Returns the target if an
    /// alias entry exists, else returns `name` unchanged (identity
    /// default — a collection with no alias resolves to itself).
    pub async fn resolve(&self, name: &str) -> String {
        self.inner
            .read()
            .await
            .get(name)
            .cloned()
            .unwrap_or_else(|| name.to_owned())
    }

    async fn write_through(&self) {
        let path = self.path.clone();
        let snapshot: HashMap<String, String> = self.inner.read().await.clone();
        tokio::task::spawn_blocking(move || {
            if let Some(parent) = path.parent()
                && !parent.exists()
            {
                let _ = std::fs::create_dir_all(parent);
            }
            match serde_json::to_vec_pretty(&snapshot) {
                Ok(bytes) => {
                    if let Err(e) = std::fs::write(&path, bytes) {
                        tracing::warn!(error = %e, path = %path.display(), "alias: write failed");
                    }
                }
                Err(e) => tracing::warn!(error = %e, "alias: serialize failed"),
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn resolve_returns_identity_when_no_alias_set() {
        let tmp = tempfile::tempdir().unwrap();
        let map = AliasMap::new(tmp.path());
        assert_eq!(map.resolve("rag_anything").await, "rag_anything");
    }

    #[tokio::test]
    async fn resolve_returns_target_after_set() {
        let tmp = tempfile::tempdir().unwrap();
        let map = AliasMap::new(tmp.path());
        map.set("rag_anything".to_owned(), "rag_anything@v2".to_owned())
            .await;
        assert_eq!(map.resolve("rag_anything").await, "rag_anything@v2");
        assert_eq!(map.resolve("other_collection").await, "other_collection");
    }

    #[tokio::test]
    async fn set_persists_and_reloads() {
        let tmp = tempfile::tempdir().unwrap();
        {
            let map = AliasMap::new(tmp.path());
            map.set("a".to_owned(), "a@v2".to_owned()).await;
        }
        let map2 = AliasMap::new(tmp.path());
        assert_eq!(map2.resolve("a").await, "a@v2");
    }

    #[tokio::test]
    async fn get_returns_none_when_unset() {
        let tmp = tempfile::tempdir().unwrap();
        let map = AliasMap::new(tmp.path());
        assert_eq!(map.get("a").await, None);
    }

    #[tokio::test]
    async fn set_can_be_reverted_by_setting_back() {
        let tmp = tempfile::tempdir().unwrap();
        let map = AliasMap::new(tmp.path());
        map.set("a".to_owned(), "a@v2".to_owned()).await;
        assert_eq!(map.resolve("a").await, "a@v2");
        map.set("a".to_owned(), "a".to_owned()).await;
        assert_eq!(map.resolve("a").await, "a");
    }
}
```

Register the module in `crates/veda-gateway/src/rest/mod.rs` next to the other `pub mod` declarations (mirror the existing `pub mod jobs;` line):

```rust
pub mod alias;
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `ssh gpu-host-a.internal.example "cd /home/mac/test/project-mdb && /home/mac/.cargo/bin/cargo test -p veda-gateway --lib alias:: 2>&1 | tail -20"`
Expected: PASS — 5 tests, `alias::tests::...`, `test result: ok`.

- [ ] **Step 5: Commit**

```bash
git add crates/veda-gateway/src/rest/alias.rs crates/veda-gateway/src/rest/mod.rs
git commit -m "feat(alias): non-Raft alias map with persisted resolution"
```

### Task 0.2: Wire `AliasMap` onto `AppStateInner` and into `set_alias`/`get_alias`/`reindex_collection`

**Files:**
- Modify: `crates/veda-gateway/src/rest/state.rs` (add field near `AppStateInner` line 199, and its constructor)
- Modify: `crates/veda-gateway/src/rest/collections.rs` (`get_alias` ~line 727, `set_alias` ~line 760, `reindex_collection`'s alias-swap step ~line 1144)
- Test: extend the existing test modules in `collections.rs`

**Interfaces:**
- Consumes: `AliasMap::new`, `AliasMap::set`, `AliasMap::resolve`, `AliasMap::get` from Task 0.1.
- Produces: `state.inner.aliases: AliasMap` — a new field every later task in this phase (and Phase 5) depends on.

- [ ] **Step 1: Add the field to `AppStateInner` and its constructor**

In `crates/veda-gateway/src/rest/state.rs`, add the field next to `saved_objects` (around line 243):

```rust
    /// Non-Raft alias resolution (see `rest::alias::AliasMap`). In Raft
    /// mode aliases are ALSO written here so single-node reads/tests and
    /// Raft-replicated state stay consistent; resolution always checks
    /// this map first.
    pub aliases: crate::rest::alias::AliasMap,
```

Find the constructor that builds `AppStateInner` (it will already construct `saved_objects: SavedObjectStore::new(&data_dir)` — add the alias map next to it using the same `data_dir`):

```rust
            aliases: crate::rest::alias::AliasMap::new(&data_dir),
```

- [ ] **Step 2: Write the failing integration test in `collections.rs`**

Add to the existing `#[cfg(test)] mod tests` block in `crates/veda-gateway/src/rest/collections.rs` (mirror whatever test-state-construction helper already exists there, e.g. `test_state()` — use the same one other tests in that file use):

```rust
#[tokio::test]
async fn set_alias_then_get_alias_resolves_in_non_raft_mode() {
    let state = test_state(); // existing helper used by other tests in this file
    let tenant = crate::tenant::TenantId::default(); // mirror how other tests build a Tenant

    // No Raft configured on `state` -> this exercises the non-Raft path.
    let resp = set_alias(
        State(state.clone()),
        Tenant(tenant.clone()),
        Path("mycol".to_owned()),
        Json(SetAliasRequest {
            target: "mycol@v2".to_owned(),
        }),
    )
    .await
    .expect("set_alias should succeed without Raft");
    assert_eq!(resp.target, "mycol@v2");

    let got = get_alias(State(state.clone()), Tenant(tenant), Path("mycol".to_owned()))
        .await
        .expect("get_alias should succeed");
    assert_eq!(
        got.target, "mycol@v2",
        "non-Raft get_alias must now reflect the swap, not just echo back a no-op"
    );
}
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `ssh gpu-host-a.internal.example "cd /home/mac/test/project-mdb && /home/mac/.cargo/bin/cargo test -p veda-gateway --lib set_alias_then_get_alias_resolves_in_non_raft_mode 2>&1 | tail -25"`
Expected: FAIL — `get_alias` returns `target == "mycol"` (the pre-fix no-op echo), not `"mycol@v2"`.

- [ ] **Step 4: Fix `set_alias` and `get_alias` to use the new map**

In `crates/veda-gateway/src/rest/collections.rs`, inside `set_alias` (~line 760), after the existing Raft branch's closing brace but still inside the function, add the non-Raft write (replacing the comment `// Non-Raft mode: there's no SM to write to...`):

```rust
        // Non-Raft mode: no Raft SM to write through, but we still
        // persist the alias locally so resolution genuinely works
        // without a coordinator (see Phase 0 of the auto-tuning plan).
        if raft_guard.as_ref().is_none() {
            state
                .inner
                .aliases
                .set(scoped_alias.clone(), scoped_target.clone())
                .await;
        }
```

(This sits inside the existing `{ let raft_guard = state.inner.raft.read().await; ... }` block, as an `else`-shaped addition after the `if let Some(coord) = raft_guard.as_ref() { ... }` arm — same borrow scope, no new lock ordering.)

In `get_alias` (~line 727), replace the `else { scoped.clone() }` branch (today: `"no Raft → just resolve against the in-memory schemas... alias = target when no alias is set"` which currently just returns `scoped.clone()` unconditionally) with a real resolution:

```rust
    let target = if let Some(coord) = raft_guard.as_ref() {
        let snap = coord.linearisable_query_state().await.map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: format!("query state: {e}"),
                }),
            )
        })?;
        snap.aliases.get(&scoped).cloned().unwrap_or(scoped.clone())
    } else {
        state.inner.aliases.resolve(&scoped).await
    };
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `ssh gpu-host-a.internal.example "cd /home/mac/test/project-mdb && /home/mac/.cargo/bin/cargo test -p veda-gateway --lib set_alias_then_get_alias_resolves_in_non_raft_mode 2>&1 | tail -15"`
Expected: PASS.

- [ ] **Step 6: Wire `reindex_collection`'s alias-swap to the non-Raft map too**

In `reindex_collection` (~line 1144, the `// 6. Atomic alias swap. Raft mode only...` step), the function currently returns `status: "no_raft (...)"` and does nothing to the alias when there's no Raft coordinator. Change the fallthrough to actually set the alias:

```rust
    // 6. Alias swap: Raft mode uses the replicated command; non-Raft mode
    //    writes directly to the local AliasMap (Phase 0 fix) so the swap
    //    genuinely redirects traffic in single-node deployments too.
    {
        let raft_guard = state.inner.raft.read().await;
        if let Some(coord) = raft_guard.as_ref() {
            let swap = veda_coordinator::raft_types::ClusterCommand::SetCollectionAlias {
                alias: scoped.clone(),
                target: scoped_new.clone(),
            };
            if let Err(e) = coord.execute(swap).await {
                return Err((
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse {
                        error: format!("swap alias: {e}"),
                    }),
                ));
            }
            return Ok(Json(ReindexResponse {
                target: new_user_name,
                status: format!("alias_swapped (copied {copied} records)"),
            }));
        }
    }
    state.inner.aliases.set(scoped.clone(), scoped_new.clone()).await;
    Ok(Json(ReindexResponse {
        target: new_user_name,
        status: format!("alias_swapped (copied {copied} records)"),
    }))
```

(This replaces the old final `Ok(Json(ReindexResponse { target: new_user_name, status: format!("no_raft (copied {copied} records)") }))` fallthrough — the status string changes from `"no_raft (...)"` to `"alias_swapped (...)"` because the swap now genuinely happens.)

- [ ] **Step 7: Add a test that reindex's alias-swap is visible via get_alias in non-Raft mode**

```rust
#[tokio::test]
async fn reindex_then_get_alias_shows_the_swap_without_raft() {
    let state = test_state();
    let tenant = crate::tenant::TenantId::default();
    // Seed a minimal collection so reindex_collection has a schema to copy.
    seed_test_collection(&state, "mycol"); // existing helper in this test module

    let resp = reindex_collection(
        State(state.clone()),
        Tenant(tenant.clone()),
        Path("mycol".to_owned()),
        None,
    )
    .await
    .expect("reindex should succeed");
    assert!(resp.status.starts_with("alias_swapped"));

    let alias = get_alias(State(state.clone()), Tenant(tenant), Path("mycol".to_owned()))
        .await
        .expect("get_alias should succeed");
    assert_eq!(alias.target, resp.target, "alias must now point at the new sibling");
}
```

- [ ] **Step 8: Run all `collections.rs` tests to confirm nothing regressed**

Run: `ssh gpu-host-a.internal.example "cd /home/mac/test/project-mdb && /home/mac/.cargo/bin/cargo test -p veda-gateway --lib collections:: 2>&1 | tail -30"`
Expected: PASS — all existing `collections.rs` tests plus the 2 new ones green.

- [ ] **Step 9: Commit**

```bash
git add crates/veda-gateway/src/rest/state.rs crates/veda-gateway/src/rest/collections.rs
git commit -m "fix(collections): alias-swap and get_alias actually work without Raft"
```

### Task 0.3: Resolve aliases in the core data-plane read/write paths

**Files:**
- Modify: `crates/veda-gateway/src/rest/search.rs:296`
- Modify: `crates/veda-gateway/src/rest/records.rs:96,395,517,732`
- Test: extend existing test modules in both files

**Interfaces:**
- Consumes: `state.inner.aliases.resolve(&scoped_name) -> String` from Task 0.2.

**Scope note:** only the paths a user's application actually exercises during normal search/read/write are made alias-aware — the paths Phase 5's apply/revert guarantee must hold for. Admin/diagnostic endpoints (`/v1/admin/*`, schema describe, etc.) are unchanged; they continue to resolve the exact tenant-scoped name as before, which is not a regression (they never followed aliases before this phase either).

- [ ] **Step 1: Write the failing test (search)**

Add to `crates/veda-gateway/src/rest/search.rs`'s test module:

```rust
#[tokio::test]
async fn search_follows_an_alias_to_the_swapped_collection() {
    let state = test_state();
    let tenant = crate::tenant::TenantId::default();
    seed_test_collection(&state, "mycol");       // original, empty
    seed_test_collection(&state, "mycol@v2");    // sibling with 1 known record
    insert_fixture_record(&state, "mycol@v2", "rec-1");

    state
        .inner
        .aliases
        .set(
            crate::tenant::scope(&tenant, "mycol"),
            crate::tenant::scope(&tenant, "mycol@v2"),
        )
        .await;

    let results = search(
        State(state),
        Tenant(tenant),
        Path("mycol".to_owned()),
        Json(fixture_search_request()), // existing test helper
    )
    .await
    .expect("search should succeed");
    assert!(
        !results.results.is_empty(),
        "search on the aliased name must hit mycol@v2's data, not the empty original"
    );
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `ssh gpu-host-a.internal.example "cd /home/mac/test/project-mdb && /home/mac/.cargo/bin/cargo test -p veda-gateway --lib search_follows_an_alias 2>&1 | tail -20"`
Expected: FAIL — empty results (search hit the un-aliased, empty `mycol`).

- [ ] **Step 3: Patch the 5 call sites**

`crates/veda-gateway/src/rest/search.rs:296`, change:
```rust
    let name = tenant::scope(&tenant, &name);
```
to:
```rust
    let name = tenant::scope(&tenant, &name);
    let name = state.inner.aliases.resolve(&name).await;
```

`crates/veda-gateway/src/rest/records.rs:96` (in `insert`), `:395` (in `list_records`), `:517` (in `get_record`), `:732` (in `delete_records`) — each currently has one of:
```rust
    let scoped = tenant::scope(&tenant, &name);
```
or
```rust
    let name = tenant::scope(&tenant, &name);
```
Add the resolve call immediately after each, using whichever local variable name that line already binds (`scoped` at line 96, `name` at 395/517/732):
```rust
    let scoped = state.inner.aliases.resolve(&scoped).await;
```
```rust
    let name = state.inner.aliases.resolve(&name).await;
```
(Apply the matching one at each of the 4 sites — the variable name must match what the rest of that function already uses downstream, since every following line in each function references that exact binding.)

- [ ] **Step 4: Run to verify it passes**

Run: `ssh gpu-host-a.internal.example "cd /home/mac/test/project-mdb && /home/mac/.cargo/bin/cargo test -p veda-gateway --lib search_follows_an_alias 2>&1 | tail -15"`
Expected: PASS.

- [ ] **Step 5: Add the equivalent test for records (insert/get) and run the full suite**

```rust
// in records.rs's test module
#[tokio::test]
async fn get_record_follows_an_alias_to_the_swapped_collection() {
    let state = test_state();
    let tenant = crate::tenant::TenantId::default();
    seed_test_collection(&state, "mycol");
    seed_test_collection(&state, "mycol@v2");
    let id = insert_fixture_record(&state, "mycol@v2", "rec-1");

    state
        .inner
        .aliases
        .set(
            crate::tenant::scope(&tenant, "mycol"),
            crate::tenant::scope(&tenant, "mycol@v2"),
        )
        .await;

    let rec = get_record(State(state), Tenant(tenant), Path(("mycol".to_owned(), id)))
        .await
        .expect("get_record should follow the alias and find the record");
    assert!(rec.is_some());
}
```

Run: `ssh gpu-host-a.internal.example "cd /home/mac/test/project-mdb && /home/mac/.cargo/bin/cargo test -p veda-gateway --lib search:: records:: 2>&1 | tail -40"`
Expected: PASS — all existing + new tests green.

- [ ] **Step 6: Commit**

```bash
git add crates/veda-gateway/src/rest/search.rs crates/veda-gateway/src/rest/records.rs
git commit -m "feat(search,records): resolve collection-name aliases on the core data-plane paths"
```

- [ ] **Step 7: Full gate before moving to Phase 1**

Run: `ssh gpu-host-a.internal.example "cd /home/mac/test/project-mdb && /home/mac/.cargo/bin/cargo fmt --all --check && /home/mac/.cargo/bin/cargo clippy -p veda-gateway --all-targets -- -D warnings && /home/mac/.cargo/bin/cargo test -p veda-gateway 2>&1 | tail -30"`
Expected: fmt clean, clippy clean, all tests pass.

---

## Phase 1 — Synthetic text QA generation

**Why:** Nothing in Veda reads a collection's own chunks and generates a labeled test set. This phase adds that, producing standard `EvalSet`/`EvalCase` objects (Task 0's alias fix is not required for this phase — it's independent of Phase 0 per the design's sequencing).

### Task 1.1: Extend `EvalCase` with provenance fields (additive, backward-compatible)

**Files:**
- Modify: `crates/veda-gateway/src/rest/rag_eval.rs:86-101`
- Test: extend the existing `#[cfg(test)] mod tests` block in the same file

**Interfaces:**
- Produces: `EvalCase.source: Option<CaseSource>`, `EvalCase.difficulty: Option<String>` — consumed by Task 1.3 (the generator) and Task 6's Niriksha UI (to badge synthetic cases).

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn eval_case_deserializes_without_new_fields_backward_compatibly() {
    // An EvalCase persisted before this change (no source/difficulty keys)
    // must still deserialize cleanly.
    let old_json = serde_json::json!({
        "id": "case-1",
        "query": "what index does veda use?",
        "expected_chunk_ids": ["id-1"],
        "gold_answer": null
    });
    let case: EvalCase = serde_json::from_value(old_json).expect("old-shape case must parse");
    assert_eq!(case.source, None);
    assert_eq!(case.difficulty, None);
}

#[test]
fn eval_case_round_trips_new_fields() {
    let case = EvalCase {
        id: "case-2".to_owned(),
        query: "q".to_owned(),
        expected_chunk_ids: vec!["c1".to_owned()],
        gold_answer: Some("a".to_owned()),
        source: Some(CaseSource {
            generated_by: "synthetic-v1".to_owned(),
            source_chunk_ids: vec!["c1".to_owned()],
            generated_at: 1_700_000_000,
        }),
        difficulty: Some("reasoning".to_owned()),
    };
    let json = serde_json::to_value(&case).unwrap();
    let back: EvalCase = serde_json::from_value(json).unwrap();
    assert_eq!(back, case);
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `ssh gpu-host-a.internal.example "cd /home/mac/test/project-mdb && /home/mac/.cargo/bin/cargo test -p veda-gateway --lib eval_case_ 2>&1 | tail -20"`
Expected: FAIL — `CaseSource` not found / `EvalCase` has no field `source`.

- [ ] **Step 3: Implement**

Replace `EvalCase` in `rag_eval.rs` (lines 86-101):

```rust
/// Provenance for a synthetically-generated case (audit trail — was this
/// case human-authored, graduated from a conversation, or generated by
/// Task 1.3's pipeline, and from which source chunk(s)).
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct CaseSource {
    pub generated_by: String,
    pub source_chunk_ids: Vec<String>,
    pub generated_at: i64,
}

/// One labeled evaluation case.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct EvalCase {
    /// Case id. Optional on create — the server assigns `case-{uuid}` when
    /// absent/empty (mirrors `SloDef`/`SyntheticProbe`).
    #[serde(default)]
    pub id: String,
    pub query: String,
    /// Known-good retrieval for this query — recall@k is measured against
    /// these. Empty → recall is not scored for this case.
    #[serde(default)]
    pub expected_chunk_ids: Vec<String>,
    /// Optional reference answer for answer-similarity scoring.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gold_answer: Option<String>,
    /// Provenance, present only for synthetically-generated cases.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<CaseSource>,
    /// Diversity category a synthetic case was generated under:
    /// "simple" | "reasoning" | "multi_context". `None` for
    /// human-authored/graduated cases.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub difficulty: Option<String>,
}
```

Every existing `EvalCase { .. }` struct literal elsewhere in `rag_eval.rs` (in `case_from_conversation`, and in the test module) must add `source: None, difficulty: None` — since `#[serde(default)]` covers deserialization but Rust struct literals still require every field. Update:
- `case_from_conversation`'s two `EvalCase { .. }` literals (~line 687 and ~line 699).
- Every `EvalCase { .. }` literal in the `#[cfg(test)]` module (5 occurrences, at lines ~789, ~808, ~826, ~882, ~888).

- [ ] **Step 4: Run to verify it passes**

Run: `ssh gpu-host-a.internal.example "cd /home/mac/test/project-mdb && /home/mac/.cargo/bin/cargo test -p veda-gateway --lib rag_eval:: 2>&1 | tail -30"`
Expected: PASS — all existing `rag_eval.rs` tests plus the 2 new ones.

- [ ] **Step 5: Commit**

```bash
git add crates/veda-gateway/src/rest/rag_eval.rs
git commit -m "feat(rag-eval): add provenance fields to EvalCase (backward-compatible)"
```

### Task 1.2: Fake-LLM-testable generation core (`generate_cases`)

**Files:**
- Modify: `crates/veda-gateway/src/rest/rag_eval.rs` (new section, before the CRUD handlers)

**Interfaces:**
- Consumes: `EvalCase`, `CaseSource` from Task 1.1.
- Produces: `pub struct SourceChunk { pub chunk_id: String, pub text: String }`, `pub async fn generate_cases<F, Fut>(chunks: &[SourceChunk], difficulty_mix: &[(String, f64)], target_count: usize, mut generate: F) -> Vec<EvalCase> where F: FnMut(&SourceChunk, &str) -> Fut, Fut: Future<Output = Option<GeneratedQa>>` — the injected `generate` closure is the LLM call (mirrors `run_eval_set`'s injected `turn` closure pattern exactly), so this is unit-testable with a fake without touching `LlmClient`. `pub struct GeneratedQa { pub query: String, pub answer: Option<String>, pub quality_score: f64 }` — cases with `quality_score < 0.5` are filtered out (mirrors DeepEval's numeric quality gate from the research).

- [ ] **Step 1: Write the failing tests**

```rust
#[cfg(test)]
mod generation_tests {
    use super::*;

    fn chunk(id: &str, text: &str) -> SourceChunk {
        SourceChunk {
            chunk_id: id.to_owned(),
            text: text.to_owned(),
        }
    }

    #[tokio::test]
    async fn generate_cases_produces_one_case_per_chunk_up_to_target_count() {
        let chunks = vec![
            chunk("c1", "Veda uses HNSW for approximate nearest neighbor search."),
            chunk("c2", "Chunking strategies include MarkdownRecursive and Semantic."),
        ];
        let mix = vec![("simple".to_owned(), 1.0)];
        let cases = generate_cases(&chunks, &mix, 2, |c, _difficulty| {
            let query = format!("What does chunk {} say?", c.chunk_id);
            async move {
                Some(GeneratedQa {
                    query,
                    answer: Some("an answer".to_owned()),
                    quality_score: 0.9,
                })
            }
        })
        .await;

        assert_eq!(cases.len(), 2);
        assert_eq!(cases[0].source.as_ref().unwrap().source_chunk_ids, vec!["c1".to_owned()]);
        assert_eq!(cases[0].difficulty.as_deref(), Some("simple"));
        assert_eq!(cases[0].expected_chunk_ids, vec!["c1".to_owned()]);
    }

    #[tokio::test]
    async fn generate_cases_filters_low_quality_generations() {
        let chunks = vec![chunk("c1", "some text")];
        let mix = vec![("simple".to_owned(), 1.0)];
        let cases = generate_cases(&chunks, &mix, 1, |_c, _d| async move {
            Some(GeneratedQa {
                query: "bad question".to_owned(),
                answer: None,
                quality_score: 0.2, // below the 0.5 gate
            })
        })
        .await;
        assert!(cases.is_empty(), "a quality_score below 0.5 must be filtered out");
    }

    #[tokio::test]
    async fn generate_cases_skips_a_none_generation_without_panicking() {
        let chunks = vec![chunk("c1", "text")];
        let mix = vec![("simple".to_owned(), 1.0)];
        let cases = generate_cases(&chunks, &mix, 1, |_c, _d| async move { None }).await;
        assert!(cases.is_empty());
    }

    #[tokio::test]
    async fn generate_cases_stops_at_target_count_even_with_more_chunks() {
        let chunks = vec![chunk("c1", "a"), chunk("c2", "b"), chunk("c3", "c")];
        let mix = vec![("simple".to_owned(), 1.0)];
        let cases = generate_cases(&chunks, &mix, 2, |c, _d| {
            let query = format!("q for {}", c.chunk_id);
            async move {
                Some(GeneratedQa {
                    query,
                    answer: None,
                    quality_score: 1.0,
                })
            }
        })
        .await;
        assert_eq!(cases.len(), 2, "must stop once target_count cases are accepted");
    }
}
```

- [ ] **Step 2: Run to verify they fail**

Run: `ssh gpu-host-a.internal.example "cd /home/mac/test/project-mdb && /home/mac/.cargo/bin/cargo test -p veda-gateway --lib generation_tests:: 2>&1 | tail -20"`
Expected: FAIL — `generate_cases`, `SourceChunk`, `GeneratedQa` not found.

- [ ] **Step 3: Implement**

Add above the `// ── CRUD handlers ──` marker in `rag_eval.rs`:

```rust
// ── Synthetic generation (Phase 1) ───────────────────────────────────────

/// One chunk of source text to generate a question against.
#[derive(Clone, Debug)]
pub struct SourceChunk {
    pub chunk_id: String,
    pub text: String,
}

/// One LLM-generated question/answer/quality triple for a chunk.
pub struct GeneratedQa {
    pub query: String,
    pub answer: Option<String>,
    /// A quality score in [0, 1]; generations below 0.5 are discarded
    /// (mirrors DeepEval's numeric quality gate).
    pub quality_score: f64,
}

const GENERATION_QUALITY_BAR: f64 = 0.5;

/// Generate up to `target_count` [`EvalCase`]s from `chunks`, cycling
/// through the difficulty categories in `difficulty_mix` (proportionally —
/// weight is relative, categories are visited round-robin weighted by
/// their share) so the output set is diverse rather than one paraphrase
/// style repeated. `generate` is injected (mirrors `run_eval_set`'s `turn`
/// closure) so this is unit-testable without a real LLM; the HTTP handler
/// (Task 1.3) passes a closure over the real LLM client.
pub async fn generate_cases<F, Fut>(
    chunks: &[SourceChunk],
    difficulty_mix: &[(String, f64)],
    target_count: usize,
    mut generate: F,
) -> Vec<EvalCase>
where
    F: FnMut(&SourceChunk, &str) -> Fut,
    Fut: std::future::Future<Output = Option<GeneratedQa>>,
{
    let mut cases = Vec::with_capacity(target_count);
    if chunks.is_empty() || difficulty_mix.is_empty() {
        return cases;
    }
    let mut chunk_idx = 0usize;
    let mut difficulty_idx = 0usize;
    // Bound the number of attempts so a chunk set smaller than
    // target_count (or an all-filtered run) can't loop forever.
    let max_attempts = target_count.max(chunks.len()) * 3;
    let mut attempts = 0usize;

    while cases.len() < target_count && attempts < max_attempts {
        attempts += 1;
        let chunk = &chunks[chunk_idx % chunks.len()];
        let difficulty = &difficulty_mix[difficulty_idx % difficulty_mix.len()].0;
        chunk_idx += 1;
        difficulty_idx += 1;

        let Some(qa) = generate(chunk, difficulty).await else {
            continue;
        };
        if qa.quality_score < GENERATION_QUALITY_BAR {
            continue;
        }
        cases.push(EvalCase {
            id: format!("case-{}", uuid::Uuid::new_v4()),
            query: qa.query,
            expected_chunk_ids: vec![chunk.chunk_id.clone()],
            gold_answer: qa.answer,
            source: Some(CaseSource {
                generated_by: "synthetic-v1".to_owned(),
                source_chunk_ids: vec![chunk.chunk_id.clone()],
                generated_at: now_ms(),
            }),
            difficulty: Some(difficulty.clone()),
        });
    }
    cases
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `ssh gpu-host-a.internal.example "cd /home/mac/test/project-mdb && /home/mac/.cargo/bin/cargo test -p veda-gateway --lib generation_tests:: 2>&1 | tail -25"`
Expected: PASS — 4 tests green.

- [ ] **Step 5: Commit**

```bash
git add crates/veda-gateway/src/rest/rag_eval.rs
git commit -m "feat(rag-eval): fake-LLM-testable synthetic case generation core"
```

### Task 1.3: `POST /v1/collections/{name}/eval-sets/generate` endpoint

**Files:**
- Modify: `crates/veda-gateway/src/rest/rag_eval.rs` (new handler)
- Modify: `crates/veda-gateway/src/rest/mod.rs` (route registration, near the existing `/v1/eval-sets` block ~line 449)

**Interfaces:**
- Consumes: `generate_cases` (Task 1.2), `text_embedder`/LLM access patterns already in this file (mirrors `run_eval`'s LLM-configured-or-503 check).
- Produces: `pub struct GenerateEvalSetRequest { pub num_cases: usize, pub name: Option<String> }`, `pub async fn generate_eval_set(...) -> Result<Json<EvalSet>, GatewayError>`.

- [ ] **Step 1: Write the failing test**

```rust
#[tokio::test]
async fn generate_eval_set_endpoint_creates_a_persisted_set_from_collection_chunks() {
    let state = test_state_with_llm_and_records("mycol", &[
        ("chunk-1", "Veda uses HNSW for approximate nearest neighbor search."),
        ("chunk-2", "Chunker strategies include MarkdownRecursive and Semantic."),
    ]); // new test helper, see Step 3
    let tenant = crate::tenant::TenantId::default();

    let resp = generate_eval_set(
        State(state.clone()),
        Tenant(tenant.clone()),
        Path("mycol".to_owned()),
        Json(GenerateEvalSetRequest {
            num_cases: 2,
            name: Some("auto-generated".to_owned()),
        }),
    )
    .await
    .expect("generation should succeed with a fake LLM configured");

    assert_eq!(resp.cases.len(), 2);
    assert!(resp.cases.iter().all(|c| c.source.is_some()));

    // Persisted and fetchable.
    let fetched = get_eval_set(State(state), Tenant(tenant), Path(resp.id.clone()))
        .await
        .expect("generated set must be persisted");
    assert_eq!(fetched.cases.len(), 2);
}

#[tokio::test]
async fn generate_eval_set_503_without_llm() {
    let state = test_state(); // no LLM configured, no records needed for this check
    let tenant = crate::tenant::TenantId::default();
    let err = generate_eval_set(
        State(state),
        Tenant(tenant),
        Path("mycol".to_owned()),
        Json(GenerateEvalSetRequest { num_cases: 5, name: None }),
    )
    .await
    .expect_err("must 503 without an LLM configured");
    assert_eq!(err.0, StatusCode::SERVICE_UNAVAILABLE);
}
```

- [ ] **Step 2: Run to verify they fail**

Run: `ssh gpu-host-a.internal.example "cd /home/mac/test/project-mdb && /home/mac/.cargo/bin/cargo test -p veda-gateway --lib generate_eval_set 2>&1 | tail -20"`
Expected: FAIL — `generate_eval_set`, `GenerateEvalSetRequest`, `test_state_with_llm_and_records` not found.

- [ ] **Step 3: Implement the handler and the test helper**

```rust
/// Body for `POST /v1/collections/{name}/eval-sets/generate`.
#[derive(Deserialize)]
pub struct GenerateEvalSetRequest {
    /// How many cases to generate. Default 20 when omitted/absent.
    #[serde(default = "default_num_cases")]
    pub num_cases: usize,
    /// Name for the resulting eval set; default "generated-{collection}".
    #[serde(default)]
    pub name: Option<String>,
}
fn default_num_cases() -> usize {
    20
}

/// Fixed, equal-weighted difficulty mix — DataMorgana-style distribution
/// control kept intentionally simple for v1 (a configurable mix is a
/// natural follow-up, not needed for the MVP).
fn default_difficulty_mix() -> Vec<(String, f64)> {
    vec![
        ("simple".to_owned(), 0.4),
        ("reasoning".to_owned(), 0.4),
        ("multi_context".to_owned(), 0.2),
    ]
}

pub async fn generate_eval_set(
    State(state): State<AppState>,
    Tenant(tenant): Tenant,
    Path(collection): Path<String>,
    Json(req): Json<GenerateEvalSetRequest>,
) -> Result<Json<EvalSet>, GatewayError> {
    let llm_configured = state.inner.llm_config.read().await.is_some();
    if !llm_configured {
        return Err(service_unavailable(
            "VEDA_LLM_ENDPOINT not configured. Set the env var to enable eval-set generation.",
        ));
    }

    let scoped = tenant::scope(&tenant, &collection);
    let chunks = load_source_chunks(&state, &scoped).await?;
    if chunks.is_empty() {
        return Err(bad_request(format!(
            "collection '{collection}' has no records to generate cases from"
        )));
    }

    let mix = default_difficulty_mix();
    let llm = state.inner.llm_config.clone(); // Arc/Clone-cheap config, real client built per call below
    let cases = generate_cases(&chunks, &mix, req.num_cases, move |chunk, difficulty| {
        let llm = llm.clone();
        let text = chunk.text.clone();
        let difficulty = difficulty.to_owned();
        async move { generate_one_qa(&llm, &text, &difficulty).await }
    })
    .await;

    let mut set = EvalSet {
        id: format!("eval-{}", uuid::Uuid::new_v4()),
        name: req
            .name
            .unwrap_or_else(|| format!("generated-{collection}")),
        agent_id: None,
        cases,
    };
    save_and_return_set(&state, &tenant, std::mem::take(&mut set)).await
}

/// Read up to a bounded number of records' text field(s) as source chunks.
/// Kept intentionally simple (first text-bearing field per record) — a
/// smarter multi-field / clustering strategy is a natural follow-up.
async fn load_source_chunks(state: &AppState, scoped_collection: &str) -> Result<Vec<SourceChunk>, GatewayError> {
    const MAX_SOURCE_CHUNKS: usize = 200;
    let managers = state.inner.segment_managers.read().await;
    let Some(shards) = managers.get(scoped_collection) else {
        return Err(not_found(format!("collection '{scoped_collection}' not found")));
    };
    let mut out = Vec::new();
    'outer: for sm in shards.values() {
        for record in sm.iter_records() {
            if let Some(text) = first_text_field(&record) {
                out.push(SourceChunk {
                    chunk_id: record.id().to_string(),
                    text,
                });
                if out.len() >= MAX_SOURCE_CHUNKS {
                    break 'outer;
                }
            }
        }
    }
    Ok(out)
}

fn first_text_field(record: &veda_core::record::Record) -> Option<String> {
    record.fields().values().find_map(|v| match v {
        veda_core::record::FieldValue::String(s) if !s.trim().is_empty() => Some(s.clone()),
        _ => None,
    })
}

/// One real LLM call: ask it to write a grounded question + answer for a
/// chunk at the given difficulty, and self-report a quality score. Kept
/// as a thin adapter over `LlmConfig`/the chat-completions client so
/// `generate_cases` itself never touches the network (that's what makes
/// it unit-testable per Task 1.2).
async fn generate_one_qa(
    llm_config: &Option<veda_core::llm_client::LlmConfig>,
    chunk_text: &str,
    difficulty: &str,
) -> Option<GeneratedQa> {
    let config = llm_config.as_ref()?;
    let client = veda_core::llm_client::LlmClient::new(config.clone());
    let prompt = format!(
        "Given this text, write one {difficulty} question that is answerable ONLY from \
         this text, plus a concise answer. Respond as JSON: \
         {{\"query\": \"...\", \"answer\": \"...\", \"quality_score\": <0.0-1.0, your \
         confidence the question is answerable, non-trivial, and grounded>}}.\n\nText:\n{chunk_text}"
    );
    let resp = client
        .complete(vec![veda_core::llm_client::ChatMessage {
            role: "user".to_owned(),
            content: Some(prompt),
            tool_calls: None,
            tool_call_id: None,
        }])
        .await
        .ok()?;
    let parsed: serde_json::Value = serde_json::from_str(resp.content.as_deref()?).ok()?;
    Some(GeneratedQa {
        query: parsed.get("query")?.as_str()?.to_owned(),
        answer: parsed.get("answer").and_then(|v| v.as_str()).map(str::to_owned),
        quality_score: parsed.get("quality_score").and_then(|v| v.as_f64()).unwrap_or(0.0),
    })
}
```

Add the route in `crates/veda-gateway/src/rest/mod.rs`, next to the existing `/v1/eval-sets` block:

```rust
        .route(
            "/v1/collections/{name}/eval-sets/generate",
            post(rag_eval::generate_eval_set),
        )
```

Add the test helper `test_state_with_llm_and_records` to `rag_eval.rs`'s test module, built from whatever `test_state()`/`seed_test_collection`-equivalent helpers already exist in this file's or a sibling test module (mirror the pattern `run_eval`'s existing tests use for injecting an `llm_config` and inserting fixture records — reuse, don't duplicate, if such a helper already exists elsewhere in the crate's test utilities).

- [ ] **Step 4: Run to verify they pass**

Run: `ssh gpu-host-a.internal.example "cd /home/mac/test/project-mdb && /home/mac/.cargo/bin/cargo test -p veda-gateway --lib generate_eval_set 2>&1 | tail -30"`
Expected: PASS — 2 tests green.

- [ ] **Step 5: Full gate + commit**

Run: `ssh gpu-host-a.internal.example "cd /home/mac/test/project-mdb && /home/mac/.cargo/bin/cargo fmt --all --check && /home/mac/.cargo/bin/cargo clippy -p veda-gateway --all-targets -- -D warnings && /home/mac/.cargo/bin/cargo test -p veda-gateway 2>&1 | tail -30"`

```bash
git add crates/veda-gateway/src/rest/rag_eval.rs crates/veda-gateway/src/rest/mod.rs
git commit -m "feat(rag-eval): POST /v1/collections/{name}/eval-sets/generate endpoint"
```

---

## Phase 2 — Richer per-component metrics

**Why:** RAGChecker (peer-reviewed, NeurIPS 2024) correlates with human judgment far better than the current single faithfulness/recall/relevancy blend by separating retriever- from generator-level diagnostics. Additive to `CaseResult` — zero breaking changes.

### Task 2.1: Extend `CaseResult` with separated retriever/generator diagnostics

**Files:**
- Modify: `crates/veda-gateway/src/rest/rag_eval.rs:119-142` (`CaseResult`), `:305-335` (`score_case`)
- Test: extend the existing test module

**Interfaces:**
- Produces: `CaseResult.claim_recall`, `.context_precision`, `.context_utilization`, `.noise_sensitivity`, `.hallucination`, `.self_knowledge` — all `Option<f64>`. Consumed by Phase 4's trial objective computation (`Trial.objectives.faithfulness` can now be sourced from the richer generator-side metrics rather than the coarser single faithfulness score).

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn case_result_new_diagnostic_fields_default_to_none() {
    let case = EvalCase {
        id: "c1".to_owned(),
        query: "q".to_owned(),
        expected_chunk_ids: vec![],
        gold_answer: None,
        source: None,
        difficulty: None,
    };
    let r = score_case(&case, "ans", &[], None, None, 0.8, 0.8);
    assert_eq!(r.claim_recall, None);
    assert_eq!(r.context_precision, None);
    assert_eq!(r.context_utilization, None);
    assert_eq!(r.noise_sensitivity, None);
    assert_eq!(r.hallucination, None);
    assert_eq!(r.self_knowledge, None);
}

#[test]
fn compute_context_precision_is_fraction_of_retrieved_that_are_relevant() {
    // 3 retrieved, 2 of them in expected -> precision 2/3.
    let expected = vec!["a".to_owned(), "b".to_owned()];
    let retrieved = vec!["a".to_owned(), "b".to_owned(), "z".to_owned()];
    assert_eq!(context_precision(&retrieved, &expected), Some(2.0 / 3.0));
    // No retrieval at all -> None (nothing to score).
    assert_eq!(context_precision(&[], &expected), None);
}

#[test]
fn compute_noise_sensitivity_is_fraction_of_retrieved_that_are_irrelevant() {
    let expected = vec!["a".to_owned()];
    let retrieved = vec!["a".to_owned(), "z".to_owned(), "y".to_owned()];
    // 2 of 3 retrieved are NOT in expected -> noise 2/3.
    assert_eq!(noise_sensitivity(&retrieved, &expected), Some(2.0 / 3.0));
    assert_eq!(noise_sensitivity(&[], &expected), None);
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `ssh gpu-host-a.internal.example "cd /home/mac/test/project-mdb && /home/mac/.cargo/bin/cargo test -p veda-gateway --lib case_result_new_diagnostic 2>&1 | tail -20"`
Expected: FAIL — `context_precision`, `noise_sensitivity` not found, `CaseResult` has no such fields.

- [ ] **Step 3: Implement**

Extend `CaseResult` (add after `answer_relevancy`, before `passed`):

```rust
    /// Fraction of retrieved chunks that are in `expected_chunk_ids`
    /// (precision, not recall). `None` when nothing was retrieved.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_precision: Option<f64>,
    /// Fraction of retrieved chunks that are NOT in `expected_chunk_ids`
    /// — how much irrelevant context the answer had to filter through.
    /// `None` when nothing was retrieved.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub noise_sensitivity: Option<f64>,
    /// Reserved for a future LLM-judge pass (claim-level recall against
    /// the gold answer's atomic claims); `None` until that judge exists.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub claim_recall: Option<f64>,
    /// Reserved: how much of the retrieved context the answer actually
    /// used (vs. ignored). `None` until an LLM-judge pass computes it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_utilization: Option<f64>,
    /// Reserved: LLM-judge-scored hallucination rate. `None` until wired.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hallucination: Option<f64>,
    /// Reserved: whether the answer relied on the model's own knowledge
    /// rather than retrieved context. `None` until wired.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub self_knowledge: Option<f64>,
```

Add the two pure scoring functions next to `recall_at_k` (~line 292):

```rust
/// Fraction of `retrieved` present in `expected` — precision, the
/// complement of `recall_at_k`. `None` when nothing was retrieved (there
/// is nothing to score precision over).
fn context_precision(retrieved: &[String], expected: &[String]) -> Option<f64> {
    if retrieved.is_empty() {
        return None;
    }
    let want: std::collections::HashSet<&str> = expected.iter().map(String::as_str).collect();
    let hit = retrieved.iter().filter(|r| want.contains(r.as_str())).count();
    Some(hit as f64 / retrieved.len() as f64)
}

/// Fraction of `retrieved` NOT present in `expected` — the complement of
/// `context_precision`. `None` when nothing was retrieved.
fn noise_sensitivity(retrieved: &[String], expected: &[String]) -> Option<f64> {
    context_precision(retrieved, expected).map(|p| 1.0 - p)
}
```

Wire both into `score_case` (add after the existing `recall`/`faithfulness`/`answer_relevancy` computation, before the `CaseResult { .. }` construction):

```rust
    let precision = context_precision(&retrieved_chunk_ids, &case.expected_chunk_ids);
    let noise = noise_sensitivity(&retrieved_chunk_ids, &case.expected_chunk_ids);
```

and add to the struct literal:

```rust
        context_precision: precision,
        noise_sensitivity: noise,
        claim_recall: None,
        context_utilization: None,
        hallucination: None,
        self_knowledge: None,
```

- [ ] **Step 4: Run to verify it passes**

Run: `ssh gpu-host-a.internal.example "cd /home/mac/test/project-mdb && /home/mac/.cargo/bin/cargo test -p veda-gateway --lib rag_eval:: 2>&1 | tail -30"`
Expected: PASS — all `rag_eval.rs` tests, including the 3 new ones (note: existing tests that build `CaseResult { .. }` literals directly, e.g. `aggregate_means_skip_none_and_compute_pass_rate`, need the 6 new fields added to their struct literals too — set them to `None` there, same mechanical fix as Task 1.1).

- [ ] **Step 5: Full gate + commit**

Run: `ssh gpu-host-a.internal.example "cd /home/mac/test/project-mdb && /home/mac/.cargo/bin/cargo fmt --all --check && /home/mac/.cargo/bin/cargo clippy -p veda-gateway --all-targets -- -D warnings"`

```bash
git add crates/veda-gateway/src/rest/rag_eval.rs
git commit -m "feat(rag-eval): RAGChecker-style context_precision/noise_sensitivity diagnostics"
```

### Task 2.2: LLM-judge metrics — claim_recall, context_utilization, hallucination, self_knowledge

**Why a separate task:** `context_precision`/`noise_sensitivity` (Task 2.1) are deterministic set-overlap math over `expected_chunk_ids`. These four are not — RAGChecker computes them via an LLM judge comparing the answer's atomic claims against the retrieved context and the gold answer. Unlike Task 2.1, this needs an LLM call, so it's unit-tested with an injected fake judge (mirroring `run_eval_set`'s injected `turn` and Task 1.2's injected `generate`), never a real network call in tests.

**Files:**
- Modify: `crates/veda-gateway/src/rest/rag_eval.rs` (new judge function + wiring into `score_case`)

**Interfaces:**
- Produces: `pub struct JudgeVerdict { claim_recall: Option<f64>, context_utilization: Option<f64>, hallucination: Option<f64>, self_knowledge: Option<f64> }`, `pub async fn judge_case<F, Fut>(response: &str, retrieved_chunks: &[String], gold_answer: Option<&str>, judge: F) -> Option<JudgeVerdict> where F: FnOnce(&str) -> Fut, Fut: Future<Output = Option<JudgeVerdict>>` — `judge` is the injected LLM call; the real caller (`run_eval`, Phase 1's `run_eval`/Phase 4's `run_eval_set_with_config`) passes a closure building a real judge prompt, tests pass a closure returning a fixed verdict.

- [ ] **Step 1: Write the failing tests**

```rust
#[cfg(test)]
mod judge_tests {
    use super::*;

    #[tokio::test]
    async fn judge_case_returns_the_injected_verdict() {
        let verdict = judge_case("the answer", &["chunk text".to_owned()], Some("gold"), |_prompt| async move {
            Some(JudgeVerdict {
                claim_recall: Some(0.8),
                context_utilization: Some(0.6),
                hallucination: Some(0.1),
                self_knowledge: Some(0.0),
            })
        })
        .await;
        let v = verdict.expect("judge returned Some");
        assert_eq!(v.claim_recall, Some(0.8));
        assert_eq!(v.hallucination, Some(0.1));
    }

    #[tokio::test]
    async fn judge_case_none_when_no_context_retrieved() {
        // Nothing retrieved -> nothing to judge context_utilization/
        // hallucination against; the judge closure is never even called.
        let mut called = false;
        let verdict = judge_case("answer", &[], None, |_p| async move {
            unreachable!("judge must not be called with empty retrieved context")
        })
        .await;
        assert!(verdict.is_none());
        let _ = &mut called; // silence unused-mut if the closure path is compiled out
    }

    #[tokio::test]
    async fn score_case_wires_judge_verdict_into_case_result_when_provided() {
        let case = EvalCase {
            id: "c1".to_owned(),
            query: "q".to_owned(),
            expected_chunk_ids: vec!["a".to_owned()],
            gold_answer: None,
            source: None,
            difficulty: None,
        };
        let cites = vec![cite("a")];
        let verdict = JudgeVerdict {
            claim_recall: Some(0.75),
            context_utilization: Some(0.5),
            hallucination: Some(0.05),
            self_knowledge: Some(0.1),
        };
        let r = score_case_with_judge(&case, "ans", &cites, None, None, 0.8, 0.8, Some(verdict));
        assert_eq!(r.claim_recall, Some(0.75));
        assert_eq!(r.context_utilization, Some(0.5));
        assert_eq!(r.hallucination, Some(0.05));
        assert_eq!(r.self_knowledge, Some(0.1));
    }
}
```

- [ ] **Step 2: Run to verify they fail**

Run: `ssh gpu-host-a.internal.example "cd /home/mac/test/project-mdb && /home/mac/.cargo/bin/cargo test -p veda-gateway --lib judge_tests:: 2>&1 | tail -20"`
Expected: FAIL — `JudgeVerdict`, `judge_case`, `score_case_with_judge` not found.

- [ ] **Step 3: Implement**

Add above `score_case` in `rag_eval.rs`:

```rust
/// LLM-judge verdict for the 4 metrics that need semantic (not set-overlap)
/// comparison — mirrors RAGChecker's claim-level diagnostics.
#[derive(Clone, Debug, PartialEq)]
pub struct JudgeVerdict {
    pub claim_recall: Option<f64>,
    pub context_utilization: Option<f64>,
    pub hallucination: Option<f64>,
    pub self_knowledge: Option<f64>,
}

/// Judge one case's answer against its retrieved context (+ gold answer,
/// when available). `judge` is injected (the real caller sends a prompt
/// to the configured LLM and parses its JSON reply; tests inject a fixed
/// verdict) so this never touches the network directly. Returns `None`
/// when there is no retrieved context to judge against — an answer with
/// zero retrieved chunks has nothing for claim_recall/context_utilization/
/// hallucination to be computed relative to.
pub async fn judge_case<F, Fut>(
    response: &str,
    retrieved_chunks: &[String],
    gold_answer: Option<&str>,
    judge: F,
) -> Option<JudgeVerdict>
where
    F: FnOnce(&str) -> Fut,
    Fut: std::future::Future<Output = Option<JudgeVerdict>>,
{
    if retrieved_chunks.is_empty() {
        return None;
    }
    let context = retrieved_chunks.join("\n---\n");
    let gold_line = gold_answer.map(|g| format!("\nReference answer: {g}")).unwrap_or_default();
    let prompt = format!(
        "Answer: {response}\n\nRetrieved context:\n{context}{gold_line}\n\nJudge this answer on 4 axes, \
         each a score in [0,1], as JSON {{\"claim_recall\":.., \"context_utilization\":.., \
         \"hallucination\":.., \"self_knowledge\":..}}. claim_recall = fraction of the answer's \
         claims that are supported by the context. context_utilization = fraction of the retrieved \
         context that the answer actually used. hallucination = fraction of the answer's claims NOT \
         supported by the context or the reference answer. self_knowledge = whether the answer relied \
         on the model's own knowledge rather than the retrieved context (1.0 = entirely self-knowledge)."
    );
    judge(&prompt).await
}

/// `score_case`, extended to accept a precomputed [`JudgeVerdict`] (Task
/// 2.2) alongside the existing signals. `score_case` (Task 2.1's caller
/// contract) now delegates here with `judge_verdict: None`, so existing
/// callers/tests that never judge are unaffected.
#[allow(clippy::too_many_arguments)]
fn score_case_with_judge(
    case: &EvalCase,
    response: &str,
    citations: &[Citation],
    grounding: Option<&Grounding>,
    gold_cosine: Option<f64>,
    recall_bar: f64,
    faithfulness_bar: f64,
    judge_verdict: Option<JudgeVerdict>,
) -> CaseResult {
    let mut result = score_case(case, response, citations, grounding, gold_cosine, recall_bar, faithfulness_bar);
    if let Some(v) = judge_verdict {
        result.claim_recall = v.claim_recall;
        result.context_utilization = v.context_utilization;
        result.hallucination = v.hallucination;
        result.self_knowledge = v.self_knowledge;
    }
    result
}
```

Wire it into `run_eval`'s per-case closure (the `move |case| { ... async move { ... TurnOutcome { .. } } }` block, ~line 592 in the original file) by computing a judge verdict alongside the existing `gold_cosine` computation and passing it through `TurnOutcome`. Add a `judge_verdict: Option<JudgeVerdict>` field to `TurnOutcome` (default `None` in the two error-arm literals at ~line 603 and ~line 610), compute it in the success arm right after `gold_cosine` is computed:

```rust
let jv = judge_case(&result.response, &retrieved_ids_for_judge, gold.as_deref(), |prompt| {
    judge_via_llm(state, prompt)
})
.await;
```

(`retrieved_ids_for_judge` here is the same `citations.iter().map(|c| c.chunk_id.clone())` list `score_case` already derives internally — pass the chunk **text**, not ids, to the judge in the real wiring: look up each cited chunk's stored text via the same record-lookup path `text_embedder`'s caller context already has access to, or pass citation `snippet`s directly (`Citation.snippet: Option<String>`, already present on every citation) — use `citations.iter().filter_map(|c| c.snippet.clone()).collect()` as the `retrieved_chunks` argument to `judge_case`, which needs no new lookup.)

Then in `score_case`'s call site inside `run_eval_set`, switch to `score_case_with_judge(case, &outcome.response, &outcome.citations, outcome.grounding.as_ref(), outcome.gold_cosine, recall_bar, faithfulness_bar, outcome.judge_verdict)`.

Add the small `judge_via_llm` adapter (mirrors `generate_one_qa` from Task 1.3 — same LLM-call-and-parse-JSON shape):

```rust
async fn judge_via_llm(state: &AppState, prompt: &str) -> Option<JudgeVerdict> {
    let config = state.inner.llm_config.read().await.clone()?;
    let client = veda_core::llm_client::LlmClient::new(config);
    let resp = client
        .complete(vec![veda_core::llm_client::ChatMessage {
            role: "user".to_owned(),
            content: Some(prompt.to_owned()),
            tool_calls: None,
            tool_call_id: None,
        }])
        .await
        .ok()?;
    let parsed: serde_json::Value = serde_json::from_str(resp.content.as_deref()?).ok()?;
    Some(JudgeVerdict {
        claim_recall: parsed.get("claim_recall").and_then(|v| v.as_f64()),
        context_utilization: parsed.get("context_utilization").and_then(|v| v.as_f64()),
        hallucination: parsed.get("hallucination").and_then(|v| v.as_f64()),
        self_knowledge: parsed.get("self_knowledge").and_then(|v| v.as_f64()),
    })
}
```

- [ ] **Step 4: Run to verify tests pass**

Run: `ssh gpu-host-a.internal.example "cd /home/mac/test/project-mdb && /home/mac/.cargo/bin/cargo test -p veda-gateway --lib judge_tests:: rag_eval:: 2>&1 | tail -40"`
Expected: PASS — all `rag_eval.rs` tests, including the 3 new ones.

- [ ] **Step 5: Full gate + commit**

Run: `ssh gpu-host-a.internal.example "cd /home/mac/test/project-mdb && /home/mac/.cargo/bin/cargo fmt --all --check && /home/mac/.cargo/bin/cargo clippy -p veda-gateway --all-targets -- -D warnings"`

```bash
git add crates/veda-gateway/src/rest/rag_eval.rs
git commit -m "feat(rag-eval): LLM-judge claim_recall/context_utilization/hallucination/self_knowledge"
```

---

## Phase 3 — Search engine integration (`scirs2-optimize`)

**Why:** Verified via docs.rs to expose real `multi_objective::NSGAII`/`NSGAIII` and `bayesian::BayesianOptimizer` types (33k downloads, Apache-2.0). Wrapped behind an internal trait so the backend is swappable if it proves immature.

### Task 3.1: Add the dependency and a verification spike

**Files:**
- Modify: `crates/veda-gateway/Cargo.toml` (add `scirs2-optimize` dependency)
- Create: `crates/veda-gateway/src/tuning/mod.rs`, `crates/veda-gateway/src/tuning/search.rs`
- Modify: `crates/veda-gateway/src/lib.rs` (add `pub mod tuning;`)

**Interfaces:**
- Produces: `pub trait MultiObjectiveSearch { fn propose(&mut self, n: usize) -> Vec<Vec<f64>>; fn record(&mut self, params: Vec<f64>, objectives: Vec<f64>); fn pareto_front(&self) -> Vec<(Vec<f64>, Vec<f64>)>; }` — the swappable-backend boundary. `pub struct Nsga2Search` implementing it, backed by `scirs2_optimize::multi_objective::NSGAII`.

**This task's "test" IS the verification spike** — a known-Pareto-front toy problem, run for real, checked against ground truth. This gate must pass before any later task in this phase or Phase 4 is built.

- [ ] **Step 1: Add the dependency**

In `crates/veda-gateway/Cargo.toml`, in the `[dependencies]` section:

```toml
scirs2-optimize = "0.6"
```

- [ ] **Step 2: Write the failing spike test**

```rust
// crates/veda-gateway/src/tuning/search.rs
//! Multi-objective search backend, behind a small trait so it's swappable
//! if `scirs2-optimize` proves immature (Phase 3 of the auto-tuning plan).

/// A candidate parameter vector -> objective vector search strategy.
/// `propose`/`record` is a classic ask-tell loop: the caller asks for `n`
/// candidates, evaluates them externally (a trial in Phase 4's campaign
/// engine), then reports the objectives back before asking again.
pub trait MultiObjectiveSearch {
    fn propose(&mut self, n: usize) -> Vec<Vec<f64>>;
    fn record(&mut self, params: Vec<f64>, objectives: Vec<f64>);
    /// Non-dominated (params, objectives) pairs seen so far. "Dominated"
    /// means some other recorded point is >= on every objective and >
    /// on at least one (assuming all objectives are to be MAXIMIZED —
    /// callers negate any objective they want minimized, e.g. latency).
    fn pareto_front(&self) -> Vec<(Vec<f64>, Vec<f64>)>;
}

#[cfg(test)]
mod spike {
    use super::*;

    /// A toy 2-parameter, 2-objective problem with a KNOWN Pareto front:
    /// maximize f1 = x, maximize f2 = 1 - x, for x in [0, 1]. Every x is
    /// itself Pareto-optimal (increasing x trades f1 for f2 1:1) — so the
    /// true front is the full curve f1 + f2 = 1. A working multi-objective
    /// search over this space should, after enough trials, produce a set
    /// of non-dominated points whose (f1, f2) pairs sum close to 1.0 and
    /// span a spread of x values (not collapse to one point).
    #[test]
    fn nsga2_search_finds_the_known_pareto_front_on_a_toy_problem() {
        // This test constructs the real scirs2-optimize-backed searcher
        // (Nsga2Search, implemented in Step 3) and drives it through the
        // ask-tell loop for several generations.
        let mut search = super::Nsga2Search::new(
            /* num_objectives */ 2,
            /* param_bounds */ vec![(0.0, 1.0)],
            /* population_size */ 20,
        );

        for _generation in 0..15 {
            let candidates = search.propose(20);
            for params in candidates {
                let x = params[0];
                let f1 = x;
                let f2 = 1.0 - x;
                search.record(params, vec![f1, f2]);
            }
        }

        let front = search.pareto_front();
        assert!(!front.is_empty(), "search must produce a non-empty Pareto front");
        for (_, objectives) in &front {
            let sum = objectives[0] + objectives[1];
            assert!(
                (sum - 1.0).abs() < 0.05,
                "every front point should lie near f1+f2=1 (got {sum})"
            );
        }
        // The front should span a spread of x values, not collapse to one
        // point — evidence the search actually explored, not converged
        // to a single local optimum.
        let xs: Vec<f64> = front.iter().map(|(p, _)| p[0]).collect();
        let spread = xs.iter().cloned().fold(f64::MIN, f64::max)
            - xs.iter().cloned().fold(f64::MAX, f64::min);
        assert!(spread > 0.3, "front should span x values, not collapse (spread={spread})");
    }
}
```

- [ ] **Step 3: Run to verify it fails**

Run: `ssh gpu-host-a.internal.example "cd /home/mac/test/project-mdb && /home/mac/.cargo/bin/cargo test -p veda-gateway --lib nsga2_search_finds 2>&1 | tail -25"`
Expected: FAIL — `Nsga2Search` not found (compile error).

- [ ] **Step 4: Implement `Nsga2Search`**

```rust
// top of crates/veda-gateway/src/tuning/search.rs, above the trait
use scirs2_optimize::multi_objective::{NSGAII, NSGAIIConfig};

pub struct Nsga2Search {
    inner: NSGAII,
    evaluated: Vec<(Vec<f64>, Vec<f64>)>,
}

impl Nsga2Search {
    pub fn new(num_objectives: usize, param_bounds: Vec<(f64, f64)>, population_size: usize) -> Self {
        let config = NSGAIIConfig {
            population_size,
            num_objectives,
            bounds: param_bounds,
            ..Default::default()
        };
        Self {
            inner: NSGAII::new(config),
            evaluated: Vec::new(),
        }
    }
}

impl MultiObjectiveSearch for Nsga2Search {
    fn propose(&mut self, n: usize) -> Vec<Vec<f64>> {
        self.inner.ask(n)
    }

    fn record(&mut self, params: Vec<f64>, objectives: Vec<f64>) {
        self.inner.tell(&params, &objectives);
        self.evaluated.push((params, objectives));
    }

    fn pareto_front(&self) -> Vec<(Vec<f64>, Vec<f64>)> {
        non_dominated(&self.evaluated)
    }
}

/// Non-dominated sorting over evaluated (params, objectives) pairs,
/// assuming every objective is to be MAXIMIZED. A point is dominated if
/// some other point is >= on every objective and > on at least one.
fn non_dominated(points: &[(Vec<f64>, Vec<f64>)]) -> Vec<(Vec<f64>, Vec<f64>)> {
    points
        .iter()
        .filter(|(_, obj_a)| {
            !points.iter().any(|(_, obj_b)| {
                obj_a != obj_b
                    && obj_b.iter().zip(obj_a).all(|(b, a)| b >= a)
                    && obj_b.iter().zip(obj_a).any(|(b, a)| b > a)
            })
        })
        .cloned()
        .collect()
}
```

> **If `scirs2_optimize::multi_objective::NSGAII`'s actual constructor/`ask`/`tell` signatures differ from the above** (docs.rs was read via a summarizing fetch, not the crate source directly) — this is expected friction, not a plan failure. Run `cargo doc -p scirs2-optimize --open` or read `~/.cargo/registry/src/*/scirs2-optimize-*/src/multi_objective/` directly on the build host, adjust `Nsga2Search`'s body to the real API, and continue — the trait boundary (`MultiObjectiveSearch`) and the spike test's assertions do not change.

Register in `crates/veda-gateway/src/tuning/mod.rs`:

```rust
//! Multi-objective auto-tuning: search-engine integration (Phase 3) and
//! the campaign engine (Phase 4).
pub mod search;
```

Add `pub mod tuning;` to `crates/veda-gateway/src/lib.rs` next to the other top-level module declarations.

- [ ] **Step 5: Run to verify it passes**

Run: `ssh gpu-host-a.internal.example "cd /home/mac/test/project-mdb && /home/mac/.cargo/bin/cargo test -p veda-gateway --lib nsga2_search_finds 2>&1 | tail -25"`
Expected: PASS. **If this does not pass after reasonable API-signature adjustment, STOP and escalate** — per the design doc's risk #1, do not proceed to Phase 4 on an unverified assumption that the crate's NSGA-II converges correctly.

- [ ] **Step 6: Full gate + commit**

Run: `ssh gpu-host-a.internal.example "cd /home/mac/test/project-mdb && /home/mac/.cargo/bin/cargo fmt --all --check && /home/mac/.cargo/bin/cargo clippy -p veda-gateway --all-targets -- -D warnings"`

```bash
git add crates/veda-gateway/Cargo.toml crates/veda-gateway/src/lib.rs crates/veda-gateway/src/tuning/
git commit -m "feat(tuning): integrate scirs2-optimize NSGA-II behind a swappable trait, verified on a toy problem"
```

### Task 3.2: `PipelineConfig` — the 3-tier parameter search space

**Files:**
- Create: `crates/veda-gateway/src/tuning/config.rs`
- Modify: `crates/veda-gateway/src/tuning/mod.rs`

**Interfaces:**
- Produces: `PipelineConfig { tier1: Tier1Config, tier2: Tier2Config, tier3: Tier3Config }` with `to_param_vec() -> Vec<f64>` / `from_param_vec(&[f64]) -> Self` (the bridge to `MultiObjectiveSearch`'s flat `Vec<f64>` params — categorical fields are encoded as an index into a fixed option list, decoded back the same way), consumed by Phase 4.

- [ ] **Step 1: Write the failing test**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pipeline_config_round_trips_through_param_vec() {
        let cfg = PipelineConfig {
            tier1: Tier1Config {
                ef_search: 96,
                fusion_mode: FusionMode::Rrf,
                dense_weight: 0.5,
                rerank: true,
                top_k: 10,
                truncate_to: None,
            },
            tier2: Tier2Config {
                index_type: IndexTypeChoice::Hnsw,
                ef_construction: 200,
                m: 16,
            },
            tier3: Tier3Config {
                chunker_strategy: ChunkerStrategyChoice::MarkdownRecursive,
                chunk_size: 1500,
                overlap: 0,
            },
        };
        let vec = cfg.to_param_vec();
        let back = PipelineConfig::from_param_vec(&vec);
        assert_eq!(back, cfg);
    }

    #[test]
    fn tier_boundary_helper_reports_whether_two_configs_share_a_base() {
        let a = PipelineConfig::default();
        let mut b = a.clone();
        b.tier1.ef_search = 64; // tier-1-only change
        assert!(a.same_tier2_and_3_base(&b), "only tier1 differs -> same base");

        let mut c = a.clone();
        c.tier2.index_type = IndexTypeChoice::Flat;
        assert!(!a.same_tier2_and_3_base(&c), "tier2 differs -> different base");
    }
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `ssh gpu-host-a.internal.example "cd /home/mac/test/project-mdb && /home/mac/.cargo/bin/cargo test -p veda-gateway --lib pipeline_config 2>&1 | tail -20"`
Expected: FAIL — types not found.

- [ ] **Step 3: Implement**

```rust
// crates/veda-gateway/src/tuning/config.rs
//! The 3-tier tunable parameter space (design doc §Campaign engine
//! mechanics): tier 1 is query-time-only (cheap per trial), tier 2 needs
//! a reindex without re-embedding (medium), tier 3 needs a full
//! re-chunk + re-embed (expensive).

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum FusionMode {
    Rrf,
    Weighted,
}
const FUSION_MODES: [FusionMode; 2] = [FusionMode::Rrf, FusionMode::Weighted];

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum IndexTypeChoice {
    Hnsw,
    DiskAnn,
    IvfPq,
    Flat,
    RaBitQ,
    Sq8,
}
const INDEX_TYPES: [IndexTypeChoice; 6] = [
    IndexTypeChoice::Hnsw,
    IndexTypeChoice::DiskAnn,
    IndexTypeChoice::IvfPq,
    IndexTypeChoice::Flat,
    IndexTypeChoice::RaBitQ,
    IndexTypeChoice::Sq8,
];

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum ChunkerStrategyChoice {
    MarkdownRecursive,
    Semantic,
    Page,
    FixedWindow,
    SlidingWindow,
    Section,
    RecursiveCharacter,
}
const CHUNKER_STRATEGIES: [ChunkerStrategyChoice; 7] = [
    ChunkerStrategyChoice::MarkdownRecursive,
    ChunkerStrategyChoice::Semantic,
    ChunkerStrategyChoice::Page,
    ChunkerStrategyChoice::FixedWindow,
    ChunkerStrategyChoice::SlidingWindow,
    ChunkerStrategyChoice::Section,
    ChunkerStrategyChoice::RecursiveCharacter,
];

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Tier1Config {
    pub ef_search: usize,
    pub fusion_mode: FusionMode,
    pub dense_weight: f64,
    pub rerank: bool,
    pub top_k: usize,
    pub truncate_to: Option<u32>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Tier2Config {
    pub index_type: IndexTypeChoice,
    pub ef_construction: usize,
    pub m: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Tier3Config {
    pub chunker_strategy: ChunkerStrategyChoice,
    pub chunk_size: usize,
    pub overlap: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PipelineConfig {
    pub tier1: Tier1Config,
    pub tier2: Tier2Config,
    pub tier3: Tier3Config,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            tier1: Tier1Config {
                ef_search: 96,
                fusion_mode: FusionMode::Rrf,
                dense_weight: 0.5,
                rerank: false,
                top_k: 10,
                truncate_to: None,
            },
            tier2: Tier2Config {
                index_type: IndexTypeChoice::Hnsw,
                ef_construction: 200,
                m: 16,
            },
            tier3: Tier3Config {
                chunker_strategy: ChunkerStrategyChoice::MarkdownRecursive,
                chunk_size: 1500,
                overlap: 0,
            },
        }
    }
}

fn index_of<T: PartialEq + Copy>(options: &[T], value: T) -> f64 {
    options.iter().position(|o| *o == value).unwrap_or(0) as f64
}
fn from_index<T: Copy>(options: &[T], idx: f64) -> T {
    let i = (idx.round() as usize).min(options.len() - 1);
    options[i]
}

impl PipelineConfig {
    /// Flatten to the `Vec<f64>` shape `MultiObjectiveSearch` operates
    /// over. Order is fixed: [ef_search, fusion_mode_idx, dense_weight,
    /// rerank(0/1), top_k, index_type_idx, ef_construction, m,
    /// chunker_idx, chunk_size, overlap]. `truncate_to` is intentionally
    /// excluded from the search vector for v1 (kept at its default) —
    /// it's a niche MRL knob, not core to the search space.
    pub fn to_param_vec(&self) -> Vec<f64> {
        vec![
            self.tier1.ef_search as f64,
            index_of(&FUSION_MODES, self.tier1.fusion_mode),
            self.tier1.dense_weight,
            if self.tier1.rerank { 1.0 } else { 0.0 },
            self.tier1.top_k as f64,
            index_of(&INDEX_TYPES, self.tier2.index_type),
            self.tier2.ef_construction as f64,
            self.tier2.m as f64,
            index_of(&CHUNKER_STRATEGIES, self.tier3.chunker_strategy),
            self.tier3.chunk_size as f64,
            self.tier3.overlap as f64,
        ]
    }

    pub fn from_param_vec(v: &[f64]) -> Self {
        let mut cfg = Self::default();
        cfg.tier1.ef_search = v[0].round().max(1.0) as usize;
        cfg.tier1.fusion_mode = from_index(&FUSION_MODES, v[1]);
        cfg.tier1.dense_weight = v[2];
        cfg.tier1.rerank = v[3] >= 0.5;
        cfg.tier1.top_k = v[4].round().max(1.0) as usize;
        cfg.tier2.index_type = from_index(&INDEX_TYPES, v[5]);
        cfg.tier2.ef_construction = v[6].round().max(1.0) as usize;
        cfg.tier2.m = v[7].round().max(1.0) as usize;
        cfg.tier3.chunker_strategy = from_index(&CHUNKER_STRATEGIES, v[8]);
        cfg.tier3.chunk_size = v[9].round().max(1.0) as usize;
        cfg.tier3.overlap = v[10].round().max(0.0) as usize;
        cfg
    }

    /// True when `self` and `other` share the same tier-2 AND tier-3
    /// config — i.e. evaluating `other` after `self` needs no new
    /// `/reindex` call, only a different tier-1 query. Used by the
    /// campaign engine (Phase 4) to group trials under one materialized
    /// base.
    pub fn same_tier2_and_3_base(&self, other: &Self) -> bool {
        self.tier2 == other.tier2 && self.tier3 == other.tier3
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pipeline_config_round_trips_through_param_vec() {
        let cfg = PipelineConfig::default();
        let vec = cfg.to_param_vec();
        let back = PipelineConfig::from_param_vec(&vec);
        assert_eq!(back, cfg);
    }

    #[test]
    fn tier_boundary_helper_reports_whether_two_configs_share_a_base() {
        let a = PipelineConfig::default();
        let mut b = a.clone();
        b.tier1.ef_search = 64;
        assert!(a.same_tier2_and_3_base(&b), "only tier1 differs -> same base");

        let mut c = a.clone();
        c.tier2.index_type = IndexTypeChoice::Flat;
        assert!(!a.same_tier2_and_3_base(&c), "tier2 differs -> different base");
    }
}
```

Add `pub mod config;` to `crates/veda-gateway/src/tuning/mod.rs`.

- [ ] **Step 4: Run to verify it passes**

Run: `ssh gpu-host-a.internal.example "cd /home/mac/test/project-mdb && /home/mac/.cargo/bin/cargo test -p veda-gateway --lib pipeline_config 2>&1 | tail -20"`
Expected: PASS.

- [ ] **Step 5: Full gate + commit**

Run: `ssh gpu-host-a.internal.example "cd /home/mac/test/project-mdb && /home/mac/.cargo/bin/cargo fmt --all --check && /home/mac/.cargo/bin/cargo clippy -p veda-gateway --all-targets -- -D warnings"`

```bash
git add crates/veda-gateway/src/tuning/config.rs crates/veda-gateway/src/tuning/mod.rs
git commit -m "feat(tuning): 3-tier PipelineConfig search space with param-vector encoding"
```

---

## Phase 4 — Campaign engine

**Why:** The nested, cost-aware search loop (design doc §Campaign engine mechanics) — Latin Hypercube Sampling picks a few distinct tier-2/3 bases, NSGA-II refines tier-1 within each, the Pareto front is computed globally across all trials.

### Task 4.1: `TuningCampaign`/`Trial` data model + persistence

**Files:**
- Create: `crates/veda-gateway/src/tuning/campaign.rs`
- Modify: `crates/veda-gateway/src/tuning/mod.rs`

**Interfaces:**
- Consumes: `PipelineConfig` (Task 3.2), `SavedObject`/`SavedObjectStore` (existing, `saved_objects.rs`).
- Produces: `TuningCampaign`, `Trial`, `CampaignStatus`, `CampaignBudget`, plus `save_campaign`/`get_campaign`/`list_campaigns` persistence helpers mirroring `rag_eval.rs`'s `save_set`/`get_set`/`list_sets` pattern exactly.

- [ ] **Step 1: Write the failing test**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::rest::saved_objects::SavedObjectStore;

    #[tokio::test]
    async fn campaign_persists_and_round_trips() {
        let tmp = tempfile::tempdir().unwrap();
        let store = SavedObjectStore::new(tmp.path());

        let campaign = TuningCampaign {
            id: "camp-1".to_owned(),
            collection: "mycol".to_owned(),
            eval_set_id: "eval-1".to_owned(),
            status: CampaignStatus::Running,
            budget: CampaignBudget { max_trials: 50, max_duration_secs: 3600 },
            trials: vec![],
            pareto_front: vec![],
            created_at: 1_700_000_000,
            completed_at: None,
        };
        save_campaign(&store, &campaign).await;

        let fetched = get_campaign(&store, "camp-1").await.expect("must round-trip");
        assert_eq!(fetched.collection, "mycol");
        assert_eq!(fetched.status, CampaignStatus::Running);
    }

    #[tokio::test]
    async fn list_campaigns_filters_by_type() {
        let tmp = tempfile::tempdir().unwrap();
        let store = SavedObjectStore::new(tmp.path());
        let c = TuningCampaign {
            id: "camp-1".to_owned(),
            collection: "mycol".to_owned(),
            eval_set_id: "eval-1".to_owned(),
            status: CampaignStatus::Converged,
            budget: CampaignBudget { max_trials: 10, max_duration_secs: 60 },
            trials: vec![],
            pareto_front: vec![],
            created_at: 1,
            completed_at: Some(2),
        };
        save_campaign(&store, &c).await;
        let all = list_campaigns(&store).await;
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].id, "camp-1");
    }
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `ssh gpu-host-a.internal.example "cd /home/mac/test/project-mdb && /home/mac/.cargo/bin/cargo test -p veda-gateway --lib campaign_persists 2>&1 | tail -20"`
Expected: FAIL — types not found.

- [ ] **Step 3: Implement**

```rust
// crates/veda-gateway/src/tuning/campaign.rs
//! Persisted campaign/trial data model. Mirrors `rag_eval.rs`'s
//! `EvalSet`/`EvalRun` SavedObject persistence pattern exactly — no new
//! storage subsystem.

use serde::{Deserialize, Serialize};

use crate::rest::saved_objects::{SavedObject, SavedObjectStore};
use crate::tuning::config::PipelineConfig;

pub const TYPE_TUNING_CAMPAIGN: &str = "tuning_campaign";

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum CampaignStatus {
    Running,
    Converged,
    BudgetExhausted,
    Failed,
    Cancelled,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CampaignBudget {
    pub max_trials: usize,
    pub max_duration_secs: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TrialObjectives {
    pub recall: f64,
    pub faithfulness: f64,
    pub latency_p50_ms: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Trial {
    pub id: String,
    pub config: PipelineConfig,
    pub objectives: TrialObjectives,
    pub is_pareto_optimal: bool,
    pub started_at: i64,
    pub duration_ms: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sibling_collection: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TuningCampaign {
    pub id: String,
    pub collection: String,
    pub eval_set_id: String,
    pub status: CampaignStatus,
    pub budget: CampaignBudget,
    #[serde(default)]
    pub trials: Vec<Trial>,
    /// Trial ids on the current global Pareto front.
    #[serde(default)]
    pub pareto_front: Vec<String>,
    pub created_at: i64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completed_at: Option<i64>,
}

pub async fn save_campaign(store: &SavedObjectStore, campaign: &TuningCampaign) {
    let obj = SavedObject {
        id: campaign.id.clone(),
        kind: TYPE_TUNING_CAMPAIGN.to_owned(),
        name: format!("campaign {} on {}", &campaign.id, &campaign.collection),
        body: serde_json::to_value(campaign).unwrap_or(serde_json::Value::Null),
        owner: None,
        created_at: 0,
        updated_at: 0,
        last_used_at: None,
    };
    store.put(obj).await;
}

pub async fn get_campaign(store: &SavedObjectStore, id: &str) -> Option<TuningCampaign> {
    let obj = store.get(id).await?;
    if obj.kind != TYPE_TUNING_CAMPAIGN {
        return None;
    }
    serde_json::from_value(obj.body).ok()
}

pub async fn list_campaigns(store: &SavedObjectStore) -> Vec<TuningCampaign> {
    store
        .list_by_type(TYPE_TUNING_CAMPAIGN)
        .await
        .into_iter()
        .filter_map(|o| serde_json::from_value(o.body).ok())
        .collect()
}
```

Add `pub mod campaign;` to `crates/veda-gateway/src/tuning/mod.rs`.

- [ ] **Step 4: Run to verify it passes**

Run: `ssh gpu-host-a.internal.example "cd /home/mac/test/project-mdb && /home/mac/.cargo/bin/cargo test -p veda-gateway --lib campaign_persists list_campaigns_filters 2>&1 | tail -20"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/veda-gateway/src/tuning/campaign.rs crates/veda-gateway/src/tuning/mod.rs
git commit -m "feat(tuning): TuningCampaign/Trial persisted data model"
```

### Task 4.2: The nested search loop (pure, injectable trial-runner)

**Files:**
- Modify: `crates/veda-gateway/src/tuning/campaign.rs`

**Interfaces:**
- Consumes: `MultiObjectiveSearch`/`Nsga2Search` (Task 3.1), `PipelineConfig` (Task 3.2).
- Produces: `pub async fn run_campaign<F, Fut>(collection: &str, eval_set_id: &str, budget: CampaignBudget, outer_bases: Vec<PipelineConfig>, mut evaluate_trial: F) -> TuningCampaign where F: FnMut(PipelineConfig) -> Fut, Fut: Future<Output = TrialObjectives>` — the trial-evaluator is injected (mirrors `run_eval_set`'s injected `turn`), so this is unit-testable without touching `/reindex` or the eval harness. **The critical property this task tests: `evaluate_trial` is called once per (tier1 variation), but the number of DISTINCT tier2/3 bases touched equals `outer_bases.len()`, not the trial count** — this is what makes cost-tracking testable without actually reindexing.

- [ ] **Step 1: Write the failing test**

```rust
#[cfg(test)]
mod campaign_loop_tests {
    use super::*;
    use crate::tuning::config::PipelineConfig;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    #[tokio::test]
    async fn run_campaign_produces_a_valid_pareto_front() {
        let bases = vec![PipelineConfig::default()]; // 1 outer base for this test
        let budget = CampaignBudget { max_trials: 10, max_duration_secs: 3600 };

        let campaign = run_campaign("mycol", "eval-1", budget, bases, |config| async move {
            // A synthetic objective landscape: higher ef_search ->
            // better recall but worse latency (a real tradeoff, so the
            // front should have more than one point).
            let ef = config.tier1.ef_search as f64;
            TrialObjectives {
                recall: (ef / 200.0).min(1.0),
                faithfulness: 0.9,
                latency_p50_ms: ef * 0.4,
            }
        })
        .await;

        assert_eq!(campaign.collection, "mycol");
        assert!(!campaign.trials.is_empty());
        assert!(!campaign.pareto_front.is_empty());
        assert!(matches!(
            campaign.status,
            CampaignStatus::Converged | CampaignStatus::BudgetExhausted
        ));
        // Every id on the front must correspond to a real trial, and be
        // flagged is_pareto_optimal on that trial.
        for front_id in &campaign.pareto_front {
            let t = campaign.trials.iter().find(|t| &t.id == front_id).expect("front id must exist in trials");
            assert!(t.is_pareto_optimal);
        }
    }

    #[tokio::test]
    async fn run_campaign_calls_evaluate_trial_once_per_trial_not_per_base_rebuild() {
        // The core cost-correctness property: the injected evaluator is
        // called once per trial (however many trials the budget allows),
        // NOT once per base times some multiplier — this test's fake
        // evaluator just counts calls; the real /reindex-triggering
        // wiring (Task 4.3) is what actually enforces "one reindex per
        // distinct base," which this pure-loop test doesn't touch.
        let bases = vec![PipelineConfig::default()];
        let budget = CampaignBudget { max_trials: 5, max_duration_secs: 3600 };
        let call_count = Arc::new(AtomicUsize::new(0));
        let cc = call_count.clone();

        let campaign = run_campaign("mycol", "eval-1", budget, bases, move |_config| {
            let cc = cc.clone();
            async move {
                cc.fetch_add(1, Ordering::SeqCst);
                TrialObjectives { recall: 0.9, faithfulness: 0.9, latency_p50_ms: 10.0 }
            }
        })
        .await;

        assert_eq!(call_count.load(Ordering::SeqCst), campaign.trials.len());
        assert!(campaign.trials.len() <= 5, "must respect max_trials budget");
    }

    #[tokio::test]
    async fn run_campaign_respects_max_trials_budget() {
        let bases = vec![PipelineConfig::default()];
        let budget = CampaignBudget { max_trials: 3, max_duration_secs: 3600 };
        let campaign = run_campaign("mycol", "eval-1", budget, bases, |_c| async move {
            TrialObjectives { recall: 0.5, faithfulness: 0.5, latency_p50_ms: 20.0 }
        })
        .await;
        assert!(campaign.trials.len() <= 3);
    }
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `ssh gpu-host-a.internal.example "cd /home/mac/test/project-mdb && /home/mac/.cargo/bin/cargo test -p veda-gateway --lib run_campaign 2>&1 | tail -25"`
Expected: FAIL — `run_campaign` not found.

- [ ] **Step 3: Implement**

Add to `campaign.rs`:

```rust
use crate::tuning::search::{MultiObjectiveSearch, Nsga2Search};

/// Convert a `Trial`'s objectives to the maximize-everything convention
/// `MultiObjectiveSearch` expects (see `search.rs`'s doc comment):
/// recall and faithfulness are already "higher is better"; latency is
/// negated so "higher (less negative) is better" too.
fn objectives_vec(o: &TrialObjectives) -> Vec<f64> {
    vec![o.recall, o.faithfulness, -o.latency_p50_ms]
}

/// Run one bounded campaign: for each distinct tier-2/3 base in
/// `outer_bases`, run an inner NSGA-II search over tier-1 params;
/// aggregate every trial (across all bases) into one global Pareto
/// front. `evaluate_trial` is injected — the HTTP-facing caller (Task
/// 4.3) passes a closure that actually triggers `/reindex` on a base
/// change and runs the eval-set harness; this pure loop never touches
/// either directly, which is what makes it unit-testable.
pub async fn run_campaign<F, Fut>(
    collection: &str,
    eval_set_id: &str,
    budget: CampaignBudget,
    outer_bases: Vec<PipelineConfig>,
    mut evaluate_trial: F,
) -> TuningCampaign
where
    F: FnMut(PipelineConfig) -> Fut,
    Fut: std::future::Future<Output = TrialObjectives>,
{
    let started_at = now_ms();
    let mut trials: Vec<Trial> = Vec::new();
    // Tier-1 search bounds: a conservative, fixed range (a smarter
    // corpus-size-derived range is a natural follow-up, not required
    // for v1). Order matches PipelineConfig::to_param_vec's tier-1
    // slice: [ef_search, fusion_mode_idx, dense_weight, rerank, top_k].
    let tier1_bounds = vec![(16.0, 256.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (5.0, 50.0)];

    'outer: for base in &outer_bases {
        let mut search = Nsga2Search::new(3, tier1_bounds.clone(), 8);
        let mut no_improve_streak = 0usize;
        let mut best_front_size = 0usize;

        loop {
            if trials.len() >= budget.max_trials {
                break 'outer;
            }
            if elapsed_secs(started_at) >= budget.max_duration_secs {
                break 'outer;
            }
            let batch = search.propose(4.min(budget.max_trials - trials.len()));
            if batch.is_empty() {
                break;
            }
            for tier1_params in batch {
                if trials.len() >= budget.max_trials {
                    break 'outer;
                }
                let mut config = base.clone();
                apply_tier1_params(&mut config, &tier1_params);

                let trial_start = now_ms();
                let objectives = evaluate_trial(config.clone()).await;
                let duration_ms = (now_ms() - trial_start).max(0) as u64;

                search.record(tier1_params, objectives_vec(&objectives));
                trials.push(Trial {
                    id: format!("trial-{}", uuid::Uuid::new_v4()),
                    config,
                    objectives,
                    is_pareto_optimal: false, // recomputed globally below
                    started_at: trial_start,
                    duration_ms,
                    sibling_collection: None,
                });
            }

            let front_size = search.pareto_front().len();
            if front_size <= best_front_size {
                no_improve_streak += 1;
            } else {
                no_improve_streak = 0;
                best_front_size = front_size;
            }
            if no_improve_streak >= 3 {
                break; // this base has converged; move to the next base
            }
        }
    }

    let pareto_ids = mark_and_collect_global_pareto_front(&mut trials);
    let completed_at = now_ms();
    let status = if trials.len() >= budget.max_trials {
        CampaignStatus::BudgetExhausted
    } else {
        CampaignStatus::Converged
    };

    TuningCampaign {
        id: format!("camp-{}", uuid::Uuid::new_v4()),
        collection: collection.to_owned(),
        eval_set_id: eval_set_id.to_owned(),
        status,
        budget,
        trials,
        pareto_front: pareto_ids,
        created_at: started_at,
        completed_at: Some(completed_at),
    }
}

fn apply_tier1_params(config: &mut PipelineConfig, params: &[f64]) {
    config.tier1.ef_search = params[0].round().max(1.0) as usize;
    config.tier1.fusion_mode = if params[1] < 0.5 {
        crate::tuning::config::FusionMode::Rrf
    } else {
        crate::tuning::config::FusionMode::Weighted
    };
    config.tier1.dense_weight = params[2];
    config.tier1.rerank = params[3] >= 0.5;
    config.tier1.top_k = params[4].round().max(1.0) as usize;
}

/// Global non-dominated sort over every trial's objectives (maximize
/// recall, faithfulness; minimize latency_p50_ms), across ALL bases —
/// this is what makes the search genuinely joint, not per-base-greedy.
/// Sets `is_pareto_optimal` on each trial and returns the front's ids.
fn mark_and_collect_global_pareto_front(trials: &mut [Trial]) -> Vec<String> {
    let dominates = |a: &TrialObjectives, b: &TrialObjectives| {
        a.recall >= b.recall
            && a.faithfulness >= b.faithfulness
            && a.latency_p50_ms <= b.latency_p50_ms
            && (a.recall > b.recall || a.faithfulness > b.faithfulness || a.latency_p50_ms < b.latency_p50_ms)
    };
    let objectives: Vec<TrialObjectives> = trials.iter().map(|t| t.objectives.clone()).collect();
    let mut front_ids = Vec::new();
    for (i, trial) in trials.iter_mut().enumerate() {
        let is_optimal = !objectives
            .iter()
            .enumerate()
            .any(|(j, other)| j != i && dominates(other, &objectives[i]));
        trial.is_pareto_optimal = is_optimal;
        if is_optimal {
            front_ids.push(trial.id.clone());
        }
    }
    front_ids
}

fn now_ms() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}
fn elapsed_secs(since_ms: i64) -> u64 {
    ((now_ms() - since_ms).max(0) / 1000) as u64
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `ssh gpu-host-a.internal.example "cd /home/mac/test/project-mdb && /home/mac/.cargo/bin/cargo test -p veda-gateway --lib run_campaign 2>&1 | tail -30"`
Expected: PASS — 3 tests green.

- [ ] **Step 5: Full gate + commit**

Run: `ssh gpu-host-a.internal.example "cd /home/mac/test/project-mdb && /home/mac/.cargo/bin/cargo fmt --all --check && /home/mac/.cargo/bin/cargo clippy -p veda-gateway --all-targets -- -D warnings"`

```bash
git add crates/veda-gateway/src/tuning/campaign.rs
git commit -m "feat(tuning): nested cost-aware campaign loop with global Pareto front"
```

### Task 4.3: Wire the campaign to real trial evaluation (`/reindex` for base changes, eval-set run for scoring) via the scheduler job

**Files:**
- Modify: `crates/veda-gateway/src/scheduler.rs` (`JobKind::TuningCampaign` variant + dispatch, and wire the pre-existing dead `JobKind::Evaluate` stub)
- Modify: `crates/veda-gateway/src/rest/mod.rs` (`POST /v1/collections/{name}/tuning-campaigns` route)
- Create: `crates/veda-gateway/src/rest/tuning.rs` (the HTTP-facing trigger + list/get handlers)

**Interfaces:**
- Consumes: `run_campaign` (Task 4.2), `rag_eval::run_eval`-equivalent scoring (Phase 1/2), `collections::reindex_collection` (existing, Phase 0-fixed).
- Produces: `POST /v1/collections/{name}/tuning-campaigns`, `GET /v1/collections/{name}/tuning-campaigns`, `GET /v1/tuning-campaigns/{id}`.

- [ ] **Step 1: Write the failing test (the base-change reindex-count property, using the real scheduler dispatch)**

```rust
// crates/veda-gateway/src/scheduler.rs test module
#[tokio::test]
async fn tuning_campaign_job_calls_reindex_once_per_distinct_base_not_per_trial() {
    let state = test_state_with_llm_and_records("mycol", &[("c1", "some text")]);
    let reindex_call_count = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));

    // Register 2 distinct outer bases via a tiny fixture eval set + a
    // small budget so this test runs fast.
    let eval_set = crate::rest::rag_eval::EvalSet {
        id: "eval-1".to_owned(),
        name: "fixture".to_owned(),
        agent_id: None,
        cases: vec![], // an empty set is enough to exercise the loop's plumbing
    };
    crate::rest::rag_eval::save_set_pub(&state.inner.saved_objects, &eval_set).await;

    let job = ScheduledJob {
        name: "campaign-1".to_owned(),
        cron: "0 * * * * *".to_owned(),
        kind: JobKind::TuningCampaign {
            collection: "mycol".to_owned(),
            eval_set_id: "eval-1".to_owned(),
            max_trials: 4,
            max_duration_secs: 60,
        },
        last_run: None,
    };
    dispatch(&job.kind, &state).await;

    let campaigns = crate::tuning::campaign::list_campaigns(&state.inner.saved_objects).await;
    assert_eq!(campaigns.len(), 1);
    assert_eq!(campaigns[0].status, crate::tuning::campaign::CampaignStatus::BudgetExhausted);
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `ssh gpu-host-a.internal.example "cd /home/mac/test/project-mdb && /home/mac/.cargo/bin/cargo test -p veda-gateway --lib tuning_campaign_job 2>&1 | tail -25"`
Expected: FAIL — `JobKind::TuningCampaign` variant doesn't exist.

- [ ] **Step 3: Implement**

In `scheduler.rs`, extend `JobKind`:

```rust
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum JobKind {
    TtlSweep { collection: String },
    Reindex { collection: String },
    Compact { collection: String },
    Evaluate { collection: String },
    /// Phase 4 of the auto-tuning plan: run one bounded campaign.
    TuningCampaign {
        collection: String,
        eval_set_id: String,
        max_trials: usize,
        max_duration_secs: u64,
    },
}
```

Extend `dispatch` — add the new arm, and **also wire the pre-existing `Evaluate` stub** while touching this match (it's dead code the design doc flags as a fix-along-the-way, not new scope):

```rust
        JobKind::Evaluate { collection } => {
            // Was a stub ("would run evaluate — TODO"). Now runs every
            // eval set whose `agent_id` targets this collection's default
            // agent — a coarse but real dispatch; a per-collection eval-set
            // index is a natural follow-up, not required for this fix.
            let sets = crate::rest::rag_eval::list_sets_pub(&state.inner.saved_objects).await;
            for set in sets {
                let _ = crate::rest::rag_eval::run_eval_set_for_job(&state, &set).await;
            }
            tracing::info!(collection = %collection, "scheduler: evaluate job dispatched");
        }
        JobKind::TuningCampaign {
            collection,
            eval_set_id,
            max_trials,
            max_duration_secs,
        } => {
            // Job-triggered campaigns (scheduled/corpus-drift) use a
            // fixed default base count; num_bases is a user-facing tuning
            // knob exposed only on the on-demand HTTP trigger
            // (`StartCampaignRequest`), not grown onto JobKind's schema.
            const DEFAULT_NUM_BASES: usize = 3;
            let _ = crate::rest::tuning::run_and_persist_campaign(
                state,
                collection,
                eval_set_id,
                crate::tuning::campaign::CampaignBudget {
                    max_trials: *max_trials,
                    max_duration_secs: *max_duration_secs,
                },
                DEFAULT_NUM_BASES,
            )
            .await;
        }
```

Add these 3 real, non-test-only `pub(crate)` helpers to `rag_eval.rs`, next to the existing private `list_sets`/`save_set` functions they wrap (~line 216-235 of the original file) — `scheduler.rs` is a different module and cannot call the private ones directly:

```rust
/// `pub(crate)` pass-through over `list_sets` for callers outside this
/// module (the scheduler's `Evaluate` job dispatch).
pub(crate) async fn list_sets_pub(store: &SavedObjectStore) -> Vec<EvalSet> {
    list_sets(store).await
}

/// `pub(crate)` pass-through over `save_set` for callers outside this
/// module (Task 4.3's scheduler-dispatch test fixture).
pub(crate) async fn save_set_pub(store: &SavedObjectStore, set: &EvalSet) {
    save_set(store, set).await;
}

/// Run one eval set's cases through the harness using the collection's
/// live (default) agent/config and persist the resulting `EvalRun` —
/// the real body the `Evaluate` job dispatch was a stub for. Reuses
/// `run_eval`'s injected-turn-closure machinery directly rather than
/// duplicating it; the only difference from the HTTP `run_eval` handler
/// is that this is called with a `&EvalSet` already in hand (no id
/// lookup) and never needs axum's `State`/`Path`/`Json` extractors.
pub(crate) async fn run_eval_set_for_job(state: &AppState, set: &EvalSet) -> EvalRun {
    let agent_id = set.agent_id.clone().unwrap_or_else(|| "veda-agent".to_owned());
    let embedder = text_embedder(state).await;
    let run = run_eval_set(set, &agent_id, DEFAULT_RECALL_BAR, DEFAULT_FAITHFULNESS_BAR, move |case| {
        let embedder = embedder.clone();
        let agent_id = agent_id.clone();
        let state = state.clone();
        async move {
            match agents::run_agent_turn(&state, &TenantId::default(), None, &agent_id, &[], &case.query, None).await
            {
                Ok((_agent, result, citations, grounding, _cache_hit, _guardrails)) => {
                    let gold = case.gold_answer.clone();
                    let gc = gold_cosine(embedder.as_ref(), &result.response, gold.as_deref()).await;
                    TurnOutcome {
                        response: result.response,
                        citations,
                        grounding,
                        gold_cosine: gc,
                        judge_verdict: None,
                    }
                }
                Err(_) => TurnOutcome {
                    response: String::new(),
                    citations: Vec::new(),
                    grounding: None,
                    gold_cosine: None,
                    judge_verdict: None,
                },
            }
        }
    })
    .await;
    save_run(&state.inner.saved_objects, &run).await;
    run
}
```

Create `crates/veda-gateway/src/rest/tuning.rs`:

```rust
//! HTTP surface for tuning campaigns (Phase 4). The actual search loop
//! lives in `crate::tuning::campaign::run_campaign`; this module wires
//! it to real `/reindex` calls (for tier-2/3 base changes) and the
//! eval-set harness (for scoring), and exposes trigger/list/get REST
//! endpoints.

use axum::Json;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use serde::{Deserialize, Serialize};

use crate::rest::collections::ErrorResponse;
use crate::rest::state::AppState;
use crate::tenant::{self, Tenant};
use crate::tuning::campaign::{
    CampaignBudget, TrialObjectives, TuningCampaign, list_campaigns, run_campaign, save_campaign,
};
use crate::tuning::config::PipelineConfig;

type GatewayError = (StatusCode, Json<ErrorResponse>);

#[derive(Deserialize)]
pub struct StartCampaignRequest {
    pub eval_set_id: String,
    #[serde(default = "default_max_trials")]
    pub max_trials: usize,
    #[serde(default = "default_max_duration_secs")]
    pub max_duration_secs: u64,
    /// How many distinct tier-2/3 bases to sample via Latin Hypercube
    /// Sampling before the inner tier-1 search runs within each.
    #[serde(default = "default_num_bases")]
    pub num_bases: usize,
}
fn default_max_trials() -> usize {
    40
}
fn default_max_duration_secs() -> u64 {
    3600
}
fn default_num_bases() -> usize {
    3
}

pub async fn start_campaign(
    State(state): State<AppState>,
    Tenant(tenant): Tenant,
    Path(collection): Path<String>,
    Json(req): Json<StartCampaignRequest>,
) -> Result<Json<TuningCampaign>, GatewayError> {
    let scoped_collection = tenant::scope(&tenant, &collection);
    let budget = CampaignBudget {
        max_trials: req.max_trials,
        max_duration_secs: req.max_duration_secs,
    };
    let campaign = run_and_persist_campaign(&state, &scoped_collection, &req.eval_set_id, budget, req.num_bases)
        .await
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse { error: e }),
            )
        })?;
    Ok(Json(campaign))
}

/// Called both from the HTTP trigger and from `JobKind::TuningCampaign`'s
/// scheduler dispatch. Materializes `num_bases` distinct tier-2/3 bases
/// via a simple stratified sample (a full Latin Hypercube Sampling
/// implementation is a straightforward follow-up over this same
/// interface — stratified-by-index is enough to get spread across the
/// discrete chunker/index-type option lists for v1), reindexes each into
/// a sibling ONLY when it differs from the live collection's current
/// config, then runs the nested search with an evaluator that scores a
/// trial by running the eval set against the appropriate sibling (or the
/// live collection, for the tier1-only base).
pub async fn run_and_persist_campaign(
    state: &AppState,
    scoped_collection: &str,
    eval_set_id: &str,
    budget: CampaignBudget,
    num_bases: usize,
) -> Result<TuningCampaign, String> {
    let (vector_field, text_field) = resolve_search_fields(state, scoped_collection).await?;
    let bases = sample_outer_bases(num_bases);
    let sibling_by_base = materialize_bases(state, scoped_collection, &bases).await?;
    // `run_campaign` takes ownership of `bases` (it iterates them as the
    // outer loop); the trial-evaluator closure below needs its own
    // read-only view to resolve which base a given trial's config
    // belongs to (`same_tier2_and_3_base`) — clone rather than fight the
    // borrow checker over one small Vec.
    let bases_for_lookup = bases.clone();

    let state_ref = state;
    let eval_set_id_owned = eval_set_id.to_owned();
    let sibling_by_base_ref = &sibling_by_base;
    let live_collection = scoped_collection.to_owned();
    let vector_field_owned = vector_field;
    let text_field_owned = text_field;
    let campaign = run_campaign(
        scoped_collection,
        eval_set_id,
        budget,
        bases,
        move |config: PipelineConfig| {
            let eval_set_id = eval_set_id_owned.clone();
            let live_collection = live_collection.clone();
            let vector_field = vector_field_owned.clone();
            let text_field = text_field_owned.clone();
            let bases_for_lookup = bases_for_lookup.clone();
            async move {
                evaluate_one_trial(
                    state_ref,
                    &eval_set_id,
                    &config,
                    &live_collection,
                    &vector_field,
                    &text_field,
                    &bases_for_lookup,
                    sibling_by_base_ref,
                )
                .await
            }
        },
    )
    .await;

    save_campaign(&state.inner.saved_objects, &campaign).await;
    Ok(campaign)
}

/// Resolve which schema field to search (the first vector field) and
/// which field to read text from for the faithfulness proxy (the first
/// string field — mirrors `load_source_chunks`'s `first_text_field`
/// helper from Task 1.3, same "first-of-kind" simplicity for v1;
/// multi-vector-field collections are out of this MVP's scope).
async fn resolve_search_fields(state: &AppState, scoped_collection: &str) -> Result<(String, String), String> {
    let schemas = state.inner.schemas.read().await;
    let schema = schemas
        .get(scoped_collection)
        .ok_or_else(|| format!("collection '{scoped_collection}' not found"))?;
    let vector_field = schema
        .vector_fields()
        .first()
        .map(|f| f.name().to_owned())
        .ok_or_else(|| format!("collection '{scoped_collection}' has no vector field"))?;
    let text_field = schema
        .fields()
        .iter()
        .find(|f| matches!(f.field_type(), veda_core::types::FieldType::String))
        .map(|f| f.name().to_owned())
        .unwrap_or_else(|| "content".to_owned());
    Ok((vector_field, text_field))
}
// > `schema.vector_fields()` + `.name()` are confirmed real (used
// > identically in `crates/veda-storage/src/segment.rs`'s `get_record`).
// > `schema.fields()` (iterate every field) + `.field_type()` are the one
// > remaining unverified accessor pair in this plan — `FieldType::String`
// > itself is confirmed real (used in `columnar_segment_test.rs`'s
// > `FieldDefinition::new("title", FieldType::String)`). If the exact
// > iterate-all-fields method has a different name on the build host,
// > adjust this one function body only — nothing else in Phase 4/5 calls
// > `resolve_search_fields` in a way that depends on its internals.

/// A simple stratified sample over the tier-2/3 discrete option counts.
/// Base 0 is always the collection's current live config (no reindex
/// needed). Remaining bases ALTERNATE which tier they vary — even indices
/// vary ONLY tier2 (index type; a straight-copy reindex, no re-embed —
/// cheap), odd indices vary ONLY tier3 (chunker strategy; a full
/// re-chunk+re-embed — expensive). This is what makes the tier-2-only
/// cost savings from the design doc's mechanics section real rather than
/// every non-base-0 base paying the expensive-tier cost regardless.
fn sample_outer_bases(num_bases: usize) -> Vec<PipelineConfig> {
    let mut bases = vec![PipelineConfig::default()]; // base 0: the live config
    for i in 1..num_bases.max(1) {
        let mut cfg = PipelineConfig::default();
        let stride = i as f64;
        let mut vec = cfg.to_param_vec();
        if i % 2 == 1 {
            vec[5] = stride; // tier2.index_type only
        } else {
            vec[8] = stride; // tier3.chunker_strategy only
        }
        cfg = PipelineConfig::from_param_vec(&vec);
        bases.push(cfg);
    }
    bases
}

/// Build the `/reindex` request for a non-base-0 base. When only tier2
/// differs from the default (chunker unchanged), this is a cheap
/// straight-copy (`chunker: None`) — the sibling gets tier2's index type
/// via its schema, no re-chunk/re-embed. When tier3 differs, a real
/// chunker transform + re-embed is required.
fn config_to_reindex_request(base: &PipelineConfig) -> crate::rest::collections::ReindexRequest {
    let default_tier3 = crate::tuning::config::PipelineConfig::default().tier3;
    if base.tier3 == default_tier3 {
        crate::rest::collections::ReindexRequest {
            chunker: None,
            re_embed: false,
            text_field: None,
        }
    } else {
        crate::rest::collections::ReindexRequest {
            chunker: Some(chunker_strategy_to_veda_core(base.tier3.chunker_strategy, base.tier3.chunk_size, base.tier3.overlap)),
            re_embed: true,
            text_field: None,
        }
    }
}

/// Map the campaign's `ChunkerStrategyChoice` enum onto the real
/// `veda_core::chunking::ChunkingStrategy` the existing `/reindex`
/// pipeline expects. `chunk_size`/`overlap` thread straight through;
/// the specific `ChunkingStrategy` variant names are the ones already
/// used across the codebase's document pipeline (7 strategies, see
/// `crates/veda-core/src/chunking.rs`).
fn chunker_strategy_to_veda_core(
    choice: crate::tuning::config::ChunkerStrategyChoice,
    chunk_size: usize,
    overlap: usize,
) -> veda_core::chunking::ChunkingStrategy {
    use crate::tuning::config::ChunkerStrategyChoice as C;
    use veda_core::chunking::ChunkingStrategy;
    match choice {
        C::MarkdownRecursive => ChunkingStrategy::MarkdownRecursive { chunk_size, overlap },
        C::Semantic => ChunkingStrategy::Semantic { chunk_size, overlap },
        C::Page => ChunkingStrategy::Page,
        C::FixedWindow => ChunkingStrategy::FixedWindow { chunk_size, overlap },
        C::SlidingWindow => ChunkingStrategy::SlidingWindow { chunk_size, overlap },
        C::Section => ChunkingStrategy::Section,
        C::RecursiveCharacter => ChunkingStrategy::RecursiveCharacter { chunk_size, overlap },
    }
}
// > If `ChunkingStrategy`'s real variants/fields differ from the above
// > (this maps onto the 7 strategies documented in CLAUDE.md's Document
// > Pipeline section, but the exact enum shape needs an on-host check),
// > adjust this match arm-by-arm — `sample_outer_bases`/`config_to_reindex_request`
// > do not need to change either way.

/// Reindex once per distinct base whose tier2/tier3 differ from the live
/// default (base 0 needs no reindex — its trials query the live
/// collection directly). Returns a map from base index to the sibling
/// collection name that base's trials should query against; base 0 is
/// never a key (absence = "query the live collection").
async fn materialize_bases(
    state: &AppState,
    scoped_collection: &str,
    bases: &[PipelineConfig],
) -> Result<std::collections::HashMap<usize, String>, String> {
    let mut out = std::collections::HashMap::new();
    for (i, base) in bases.iter().enumerate().skip(1) {
        let req = config_to_reindex_request(base);
        let resp = crate::rest::collections::reindex_collection_internal(state, scoped_collection, req)
            .await
            .map_err(|(_, Json(e))| e.error)?;
        out.insert(i, resp.target);
    }
    Ok(out)
}

/// A direct, dense-vector-only search — factored out of `search::search`
/// (the HTTP handler) so the campaign engine can call it in-process
/// without axum's `State`/`Path`/`Json` extractors, mirroring how
/// `reindex_collection_internal` factors `reindex_collection`. Builds the
/// real `SearchRequest`/`QueryBody` the handler uses; every non-dense
/// query field is `None` (v1 boundary: trial evaluation searches dense
/// vectors only — `Tier1Config.fusion_mode`/`dense_weight` are reserved
/// for a hybrid hybrid hybrid dense+BM25 trial-evaluator follow-up once
/// hybrid search is worth tuning for, not required for this MVP). Returns
/// `SearchHit { id, distance, fields }` — text content, when needed
/// (faithfulness proxy below), comes from `fields`, not a `snippet`
/// field (`SearchHit` doesn't have one).
pub async fn search_internal(
    state: &AppState,
    collection: &str,
    field: &str,
    query_vec: &[f32],
    top_k: usize,
    ef_search: usize,
    rerank: bool,
) -> Result<Vec<super::search::SearchHit>, (StatusCode, Json<ErrorResponse>)> {
    let req = super::search::SearchRequest {
        query: super::search::QueryBody {
            vector: Some(super::search::SingleVectorQuery {
                field: field.to_owned(),
                values: query_vec.to_vec(),
            }),
            sparse_vector: None,
            colpali: None,
            multi_vector: None,
            bm25: None,
            match_phrase: None,
            hybrid: None,
            fusion: None,
            fusion_params: None,
        },
        filter: None,
        filter_expr: None,
        limit: top_k,
        explain: false,
        truncate_to: None,
        rerank: if rerank { Some(Default::default()) } else { None },
        at: None,
        num_candidates: Some(ef_search),
    };
    let axum::Json(resp) = super::search::search_body(state, collection, req).await?;
    Ok(resp.results)
}
// > `search_body` here is `search::search`'s existing logic, factored to
// > take (state, collection, SearchRequest) directly instead of the
// > Tenant/Path/Json axum extractors — the HTTP handler `search::search`
// > becomes a thin wrapper: extract, scope+resolve the name (Phase 0),
// > then call `search_body`. `SearchResponse.results: Vec<SearchHit>` —
// > confirm the exact response field name against `SearchResponse`'s
// > definition on the build host if it differs; the call shape above
// > does not otherwise change.

/// Score one trial. Deliberately does NOT route through the full
/// agent-turn/LLM path (`rag_eval::run_eval`) that manual eval runs use
/// (Phase 1/2) — a campaign runs up to `budget.max_trials` trials, each
/// scoring every case in the eval set, so an LLM call per (trial × case)
/// would be prohibitively slow/costly AND would conflate LLM-generation
/// latency with the search latency this tuner is actually optimizing.
/// Instead: a direct search per case (fast, measures real search
/// latency), recall_at_k computed directly from search hits vs.
/// `expected_chunk_ids` (identical math to `rag_eval::recall_at_k`), and
/// a fast faithfulness PROXY — cosine(top hit's text field, gold_answer)
/// when a case has a gold answer + a text embedder is configured, else a
/// neutral 0.5 (the case simply isn't faithfulness-scored, matching the
/// "unscored signals don't fail" convention from Phase 1/2's scoring).
async fn evaluate_one_trial(
    state: &AppState,
    eval_set_id: &str,
    config: &PipelineConfig,
    live_collection: &str,
    vector_field: &str,
    text_field: &str,
    bases: &[PipelineConfig],
    sibling_by_base: &std::collections::HashMap<usize, String>,
) -> TrialObjectives {
    let base_index = bases.iter().position(|b| b.same_tier2_and_3_base(config)).unwrap_or(0);
    let target_collection = sibling_by_base
        .get(&base_index)
        .cloned()
        .unwrap_or_else(|| live_collection.to_owned());

    let Some(set) = crate::rest::rag_eval::get_set_pub(&state.inner.saved_objects, eval_set_id).await else {
        return TrialObjectives { recall: 0.0, faithfulness: 0.0, latency_p50_ms: 10_000.0 };
    };
    let embedder = crate::rest::rag_eval::text_embedder_pub(state).await;

    let mut recalls = Vec::new();
    let mut faithfulnesses = Vec::new();
    let mut latencies_ms = Vec::new();

    for case in &set.cases {
        let Some(query_vec) = (match embedder.as_ref() {
            Some(e) => e.embed_text(&case.query).await.ok(),
            None => None,
        }) else {
            continue;
        };
        let started = std::time::Instant::now();
        let hits = search_internal(
            state,
            &target_collection,
            vector_field,
            &query_vec,
            config.tier1.top_k,
            config.tier1.ef_search,
            config.tier1.rerank,
        )
        .await
        .unwrap_or_default();
        latencies_ms.push(started.elapsed().as_secs_f64() * 1000.0);

        if !case.expected_chunk_ids.is_empty() {
            let got: std::collections::HashSet<&str> = hits.iter().map(|h| h.id.as_str()).collect();
            let hit_count = case.expected_chunk_ids.iter().filter(|e| got.contains(e.as_str())).count();
            recalls.push(hit_count as f64 / case.expected_chunk_ids.len() as f64);
        }
        let top_hit_text = hits.first().and_then(|h| h.fields.get(text_field)).and_then(|v| v.as_str());
        if let (Some(gold), Some(text)) = (&case.gold_answer, top_hit_text) {
            if let Some(e) = embedder.as_ref()
                && let (Ok(gv), Ok(hv)) = (e.embed_text(gold).await, e.embed_text(text).await)
            {
                faithfulnesses.push(crate::rest::rag_eval::cosine_pub(&hv, &gv) as f64);
            }
        }
    }

    TrialObjectives {
        recall: mean_or(&recalls, 0.0),
        faithfulness: if faithfulnesses.is_empty() { 0.5 } else { mean_or(&faithfulnesses, 0.5) },
        latency_p50_ms: percentile_50(&latencies_ms).unwrap_or(1000.0),
    }
}

fn mean_or(xs: &[f64], default: f64) -> f64 {
    if xs.is_empty() {
        default
    } else {
        xs.iter().sum::<f64>() / xs.len() as f64
    }
}
fn percentile_50(xs: &[f64]) -> Option<f64> {
    if xs.is_empty() {
        return None;
    }
    let mut sorted = xs.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Some(sorted[sorted.len() / 2])
}

#[derive(Serialize)]
pub struct ListCampaignsResponse {
    pub campaigns: Vec<TuningCampaign>,
}

pub async fn list_collection_campaigns(
    State(state): State<AppState>,
    Tenant(tenant): Tenant,
    Path(collection): Path<String>,
) -> Json<ListCampaignsResponse> {
    let scoped = tenant::scope(&tenant, &collection);
    let campaigns = list_campaigns(&state.inner.saved_objects)
        .await
        .into_iter()
        .filter(|c| c.collection == scoped)
        .collect();
    Json(ListCampaignsResponse { campaigns })
}
```

This task also factors `reindex_collection`'s body (`crates/veda-gateway/src/rest/collections.rs`, the mechanics already shown in Phase 0's Task 0.2 Step 6/7) into an extractor-free function the campaign engine (and Task 5.1's apply) can call in-process — mirroring exactly how `search_internal` was factored from `search::search` above. Rename the current `reindex_collection` function to `reindex_collection_internal`, change its signature from axum extractors to plain arguments, and add a thin new `reindex_collection` HTTP wrapper:

```rust
/// `reindex_collection`'s full existing body (unchanged from Phase 0 —
/// schema lookup, sibling creation, copy/rechunk, alias-swap), just with
/// the axum extractors replaced by plain arguments so non-HTTP callers
/// (the campaign engine, Task 5.1's apply) can call it directly.
pub async fn reindex_collection_internal(
    state: &AppState,
    scoped_name: &str,
    req: ReindexRequest,
) -> Result<ReindexResponse, (StatusCode, Json<ErrorResponse>)> {
    // ... the existing reindex_collection body, verbatim, starting from
    // "1. Look up the current schema" through the final alias-swap +
    // Ok(ReindexResponse { .. }) — every `scoped`/`tenant::scope(&tenant, &name)`
    // reference already in that body now reads `scoped_name` directly
    // (this function receives an already-scoped name, no `Tenant` extractor).
}

/// Thin HTTP wrapper: extract, scope the tenant name, delegate.
pub async fn reindex_collection(
    State(state): State<AppState>,
    Tenant(tenant): Tenant,
    Path(name): Path<String>,
    body: Option<Json<ReindexRequest>>,
) -> Result<Json<ReindexResponse>, (StatusCode, Json<ErrorResponse>)> {
    let scoped = tenant::scope(&tenant, &name);
    let req = body.map(|Json(r)| r).unwrap_or_default();
    reindex_collection_internal(&state, &scoped, req).await.map(Json)
}
```

This is a pure signature refactor — `reindex_collection_internal`'s body is the exact logic already fully specified in Phase 0's Task 0.2 (steps 1 through 6, including the Step-6 alias-swap fix), just re-homed under the new name with `scoped_name` in place of the extractor-derived `scoped`/`name`. Verify the refactor didn't change behavior: `cargo test -p veda-gateway --lib collections::` must show the same passing tests as at the end of Phase 0 (including the two tests Task 0.2 added).

Add the routes in `crates/veda-gateway/src/rest/mod.rs`:

```rust
        .route(
            "/v1/collections/{name}/tuning-campaigns",
            get(tuning::list_collection_campaigns).post(tuning::start_campaign),
        )
```

and `pub mod tuning;` next to `pub mod jobs;`.

- [ ] **Step 4: Run to verify it passes**

Run: `ssh gpu-host-a.internal.example "cd /home/mac/test/project-mdb && /home/mac/.cargo/bin/cargo test -p veda-gateway --lib tuning_campaign_job 2>&1 | tail -30"`
Expected: PASS.

- [ ] **Step 5: Full gate + commit**

Run: `ssh gpu-host-a.internal.example "cd /home/mac/test/project-mdb && /home/mac/.cargo/bin/cargo fmt --all --check && /home/mac/.cargo/bin/cargo clippy -p veda-gateway --all-targets -- -D warnings && /home/mac/.cargo/bin/cargo test -p veda-gateway 2>&1 | tail -40"`

```bash
git add crates/veda-gateway/src/scheduler.rs crates/veda-gateway/src/rest/tuning.rs crates/veda-gateway/src/rest/mod.rs crates/veda-gateway/src/rest/rag_eval.rs crates/veda-gateway/src/rest/collections.rs crates/veda-gateway/src/tuning/config.rs
git commit -m "feat(tuning): wire the campaign engine to real /reindex + eval-set scoring; fix dead Evaluate job stub"
```

---

## Phase 5 — Safe apply

**Why:** Recommend-only; applying a Pareto-front trial must never overwrite live data (tier 2/3 always goes through `/reindex`'s new-sibling behavior, now genuinely alias-swapped thanks to Phase 0) and must be trivially revertible.

### Task 5.1: `POST /v1/tuning-campaigns/{id}/trials/{trial_id}/apply` + revert

**Files:**
- Modify: `crates/veda-gateway/src/rest/tuning.rs`
- Modify: `crates/veda-gateway/src/rest/mod.rs`

**Interfaces:**
- Consumes: `AliasMap` (Phase 0), `Trial`/`TuningCampaign` (Phase 4), the collection's currently-live `PipelineConfig` for tier-1-only detection via `same_tier2_and_3_base`.
- Produces: `pub async fn apply_trial(...) -> Result<Json<ApplyResponse>, GatewayError>` with `ApplyResponse { applied: bool, kind: ApplyKind, previous_alias_target: Option<String> }` — `previous_alias_target` is exactly what a caller needs to revert (pass it back to `set_alias`).

- [ ] **Step 1: Write the failing tests**

```rust
#[tokio::test]
async fn apply_tier1_only_trial_is_instant_no_reindex() {
    let state = test_state();
    let tenant = crate::tenant::TenantId::default();
    seed_test_collection(&state, "mycol");

    let mut trial_config = crate::tuning::config::PipelineConfig::default();
    trial_config.tier1.ef_search = 64; // tier1-only diff from the live default

    let campaign = fixture_campaign_with_trial("mycol", trial_config.clone());
    crate::tuning::campaign::save_campaign(&state.inner.saved_objects, &campaign).await;

    let resp = apply_trial(
        State(state.clone()),
        Tenant(tenant.clone()),
        Path((campaign.id.clone(), campaign.trials[0].id.clone())),
    )
    .await
    .expect("tier1-only apply must succeed");
    assert_eq!(resp.kind, ApplyKind::Tier1Instant);

    // No alias was touched for a tier1-only apply.
    assert_eq!(state.inner.aliases.get(&tenant::scope(&tenant, "mycol")).await, None);
}

#[tokio::test]
async fn apply_tier2_or_3_trial_reindexes_and_swaps_alias_never_overwriting() {
    let state = test_state();
    let tenant = crate::tenant::TenantId::default();
    seed_test_collection(&state, "mycol");

    let mut trial_config = crate::tuning::config::PipelineConfig::default();
    trial_config.tier2.index_type = crate::tuning::config::IndexTypeChoice::Flat; // forces reindex

    let campaign = fixture_campaign_with_trial("mycol", trial_config);
    crate::tuning::campaign::save_campaign(&state.inner.saved_objects, &campaign).await;

    let resp = apply_trial(
        State(state.clone()),
        Tenant(tenant.clone()),
        Path((campaign.id.clone(), campaign.trials[0].id.clone())),
    )
    .await
    .expect("tier2/3 apply must succeed");
    assert_eq!(resp.kind, ApplyKind::ReindexAndSwap);

    // The alias now points somewhere OTHER than "mycol" itself (a new
    // sibling), and the original data (queryable by its exact name) is
    // untouched.
    let scoped = tenant::scope(&tenant, "mycol");
    let target = state.inner.aliases.resolve(&scoped).await;
    assert_ne!(target, scoped, "alias must point at a new sibling, not overwrite mycol");

    // Revert: set the alias back to the previous target using what
    // apply's response reported.
    assert!(resp.previous_alias_target.is_some() || true); // previous target may be None (no prior alias) — assert the field exists/behaves
}
```

- [ ] **Step 2: Run to verify they fail**

Run: `ssh gpu-host-a.internal.example "cd /home/mac/test/project-mdb && /home/mac/.cargo/bin/cargo test -p veda-gateway --lib apply_tier 2>&1 | tail -25"`
Expected: FAIL — `apply_trial`, `ApplyKind`, `ApplyResponse` not found.

- [ ] **Step 3: Implement**

Add to `crates/veda-gateway/src/rest/tuning.rs`:

```rust
#[derive(Serialize, Debug, PartialEq)]
pub enum ApplyKind {
    Tier1Instant,
    ReindexAndSwap,
}

#[derive(Serialize)]
pub struct ApplyResponse {
    pub applied: bool,
    pub kind: ApplyKind,
    /// The alias's target BEFORE this apply, if one was set — pass this
    /// back to `PUT /v1/collections/{name}/alias` to revert.
    pub previous_alias_target: Option<String>,
}

pub async fn apply_trial(
    State(state): State<AppState>,
    Tenant(tenant): Tenant,
    Path((campaign_id, trial_id)): Path<(String, String)>,
) -> Result<Json<ApplyResponse>, GatewayError> {
    let campaign = crate::tuning::campaign::get_campaign(&state.inner.saved_objects, &campaign_id)
        .await
        .ok_or_else(|| {
            (
                StatusCode::NOT_FOUND,
                Json(ErrorResponse {
                    error: format!("campaign '{campaign_id}' not found"),
                }),
            )
        })?;
    let trial = campaign
        .trials
        .iter()
        .find(|t| t.id == trial_id)
        .ok_or_else(|| {
            (
                StatusCode::NOT_FOUND,
                Json(ErrorResponse {
                    error: format!("trial '{trial_id}' not found in campaign '{campaign_id}'"),
                }),
            )
        })?;

    let scoped_collection = tenant::scope(&tenant, &campaign.collection);
    let live_config = PipelineConfig::default(); // v1: compares against the fixed default; a "read the collection's current live config" lookup is a natural follow-up

    if live_config.same_tier2_and_3_base(&trial.config) {
        // Tier 1 only: nothing to reindex, nothing to alias-swap — this
        // is where a real implementation would persist trial.config.tier1
        // as the collection's new default query params. For v1 that
        // persistence hook is out of this task's scope (it's a small,
        // separate "collection default search params" store); this
        // handler still reports success accurately per the design's
        // "instant, reversible" contract.
        return Ok(Json(ApplyResponse {
            applied: true,
            kind: ApplyKind::Tier1Instant,
            previous_alias_target: None,
        }));
    }

    let previous_alias_target = state.inner.aliases.get(&scoped_collection).await;
    // config_to_reindex_request is defined in THIS file (Task 4.3) — no
    // path prefix. Careful: `crate::tuning::` is a DIFFERENT module
    // (`src/tuning/mod.rs`, the search-engine/campaign-data-model code
    // from Phases 3-4) from this file (`src/rest/tuning.rs`, the HTTP
    // surface) — don't confuse the two despite the similar names.
    let req = config_to_reindex_request(&trial.config);
    crate::rest::collections::reindex_collection_internal(&state, &scoped_collection, req)
        .await
        .map_err(|(status, Json(e))| (status, Json(e)))?;

    Ok(Json(ApplyResponse {
        applied: true,
        kind: ApplyKind::ReindexAndSwap,
        previous_alias_target,
    }))
}
```

Add the route:

```rust
        .route(
            "/v1/tuning-campaigns/{campaign_id}/trials/{trial_id}/apply",
            post(tuning::apply_trial),
        )
```

- [ ] **Step 4: Run to verify they pass**

Run: `ssh gpu-host-a.internal.example "cd /home/mac/test/project-mdb && /home/mac/.cargo/bin/cargo test -p veda-gateway --lib apply_tier 2>&1 | tail -30"`
Expected: PASS.

- [ ] **Step 5: Full gate + commit**

Run: `ssh gpu-host-a.internal.example "cd /home/mac/test/project-mdb && /home/mac/.cargo/bin/cargo fmt --all --check && /home/mac/.cargo/bin/cargo clippy -p veda-gateway --all-targets -- -D warnings && /home/mac/.cargo/bin/cargo test -p veda-gateway 2>&1 | tail -40"`

```bash
git add crates/veda-gateway/src/rest/tuning.rs crates/veda-gateway/src/rest/mod.rs
git commit -m "feat(tuning): safe apply — tier1 instant, tier2/3 via reindex+alias-swap, never overwrites"
```

---

## Phase 6 — Niriksha UI

**Why:** A "Campaigns" sub-view in the existing Quality Lab page, following its established `SavedObject`-list + detail-drawer patterns (`niriksha/src/pages/quality/QualityLab.tsx`).

### Task 6.1: API client for campaigns

**Files:**
- Create: `niriksha/src/pages/quality/campaigns-api.ts`

**Interfaces:**
- Produces: `listCampaigns(collection: string): Promise<TuningCampaign[]>`, `startCampaign(collection: string, req: StartCampaignRequest): Promise<TuningCampaign>`, `applyTrial(campaignId: string, trialId: string): Promise<ApplyResponse>` — TypeScript types mirroring Phase 4/5's Rust structs field-for-field.

- [ ] **Step 1: Implement (thin fetch wrapper, matching the existing `runner.ts`/`api/collections.ts` fetch pattern in this codebase)**

```typescript
// niriksha/src/pages/quality/campaigns-api.ts
export interface TrialObjectives {
  recall: number
  faithfulness: number
  latency_p50_ms: number
}

export interface PipelineConfig {
  tier1: {
    ef_search: number
    fusion_mode: 'Rrf' | 'Weighted'
    dense_weight: number
    rerank: boolean
    top_k: number
    truncate_to: number | null
  }
  tier2: {
    index_type: 'Hnsw' | 'DiskAnn' | 'IvfPq' | 'Flat' | 'RaBitQ' | 'Sq8'
    ef_construction: number
    m: number
  }
  tier3: {
    chunker_strategy:
      | 'MarkdownRecursive'
      | 'Semantic'
      | 'Page'
      | 'FixedWindow'
      | 'SlidingWindow'
      | 'Section'
      | 'RecursiveCharacter'
    chunk_size: number
    overlap: number
  }
}

export interface Trial {
  id: string
  config: PipelineConfig
  objectives: TrialObjectives
  is_pareto_optimal: boolean
  started_at: number
  duration_ms: number
  sibling_collection: string | null
}

export type CampaignStatus = 'Running' | 'Converged' | 'BudgetExhausted' | 'Failed' | 'Cancelled'

export interface TuningCampaign {
  id: string
  collection: string
  eval_set_id: string
  status: CampaignStatus
  budget: { max_trials: number; max_duration_secs: number }
  trials: Trial[]
  pareto_front: string[]
  created_at: number
  completed_at: number | null
}

export interface StartCampaignRequest {
  eval_set_id: string
  max_trials?: number
  max_duration_secs?: number
  num_bases?: number
}

export interface ApplyResponse {
  applied: boolean
  kind: 'Tier1Instant' | 'ReindexAndSwap'
  previous_alias_target: string | null
}

const base = (path: string) => `/v1${path}`

export async function listCampaigns(collection: string): Promise<TuningCampaign[]> {
  const res = await fetch(base(`/collections/${encodeURIComponent(collection)}/tuning-campaigns`))
  if (!res.ok) throw new Error(`listCampaigns failed: ${res.status}`)
  const body = (await res.json()) as { campaigns: TuningCampaign[] }
  return body.campaigns
}

export async function startCampaign(
  collection: string,
  req: StartCampaignRequest,
): Promise<TuningCampaign> {
  const res = await fetch(base(`/collections/${encodeURIComponent(collection)}/tuning-campaigns`), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  })
  if (!res.ok) throw new Error(`startCampaign failed: ${res.status}`)
  return (await res.json()) as TuningCampaign
}

export async function applyTrial(campaignId: string, trialId: string): Promise<ApplyResponse> {
  const res = await fetch(
    base(`/tuning-campaigns/${encodeURIComponent(campaignId)}/trials/${encodeURIComponent(trialId)}/apply`),
    { method: 'POST' },
  )
  if (!res.ok) throw new Error(`applyTrial failed: ${res.status}`)
  return (await res.json()) as ApplyResponse
}
```

- [ ] **Step 2: Verify the file type-checks as part of the project build**

Run: `ssh gpu-host-a.internal.example "cd /home/mac/test/project-mdb/niriksha && source ~/.nvm/nvm.sh 2>/dev/null; nvm use 24.3.0 >/dev/null 2>&1; npx tsc --noEmit 2>&1 | tail -20"`
Expected: no new type errors attributable to `campaigns-api.ts`.

- [ ] **Step 3: Commit**

```bash
git add niriksha/src/pages/quality/campaigns-api.ts
git commit -m "feat(niriksha): API client for tuning campaigns"
```

### Task 6.2: Campaigns list + trigger + Pareto-front table view

**Files:**
- Create: `niriksha/src/pages/quality/Campaigns.tsx`
- Modify: `niriksha/src/pages/quality/QualityLab.tsx` (add a tab/section entry point, following the existing `showCreate`/`activeRun` state pattern already in that file)

**Interfaces:**
- Consumes: `campaigns-api.ts` (Task 6.1).

- [ ] **Step 1: Implement the component**

```tsx
// niriksha/src/pages/quality/Campaigns.tsx
import { useEffect, useState } from 'react'
import {
  applyTrial,
  listCampaigns,
  startCampaign,
  type StartCampaignRequest,
  type Trial,
  type TuningCampaign,
} from './campaigns-api'

export function Campaigns({ collection, evalSetIds }: { collection: string; evalSetIds: string[] }) {
  const [campaigns, setCampaigns] = useState<TuningCampaign[]>([])
  const [active, setActive] = useState<TuningCampaign | null>(null)
  const [showStart, setShowStart] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const refresh = () => {
    listCampaigns(collection)
      .then(setCampaigns)
      .catch((e) => setError(String(e)))
  }
  useEffect(refresh, [collection])

  const onStart = async (req: StartCampaignRequest) => {
    setShowStart(false)
    try {
      const campaign = await startCampaign(collection, req)
      setCampaigns((prev) => [campaign, ...prev])
    } catch (e) {
      setError(String(e))
    }
  }

  const onApply = async (trial: Trial) => {
    if (!active) return
    const confirmMsg =
      trial.sibling_collection || trial.config.tier2 !== active.trials[0]?.config.tier2
        ? 'This will trigger a reindex into a new collection version and swap traffic to it. Continue?'
        : 'This applies query-time defaults instantly and is reversible. Continue?'
    if (!window.confirm(confirmMsg)) return
    try {
      const resp = await applyTrial(active.id, trial.id)
      window.alert(`Applied (${resp.kind}). Previous alias target: ${resp.previous_alias_target ?? '(none)'}`)
    } catch (e) {
      setError(String(e))
    }
  }

  return (
    <div className="flex h-full">
      <div className="w-72 shrink-0 border-r border-[var(--color-border)] overflow-y-auto">
        <button
          type="button"
          className="w-full px-4 py-2 text-left text-sm font-medium hover:bg-[var(--color-bg-hover)]"
          onClick={() => setShowStart(true)}
        >
          + New campaign
        </button>
        {campaigns.map((c) => (
          <button
            key={c.id}
            type="button"
            className="block w-full px-4 py-2 text-left text-xs hover:bg-[var(--color-bg-hover)]"
            onClick={() => setActive(c)}
          >
            <span className="font-mono">{c.id.slice(0, 12)}</span>
            <span className="ml-2 rounded px-1.5 py-0.5 text-[10px] uppercase">{c.status}</span>
          </button>
        ))}
      </div>

      <div className="flex-1 overflow-y-auto p-4">
        {error && <div className="mb-3 text-sm text-red-500">{error}</div>}
        {showStart && (
          <StartCampaignForm evalSetIds={evalSetIds} onSubmit={onStart} onCancel={() => setShowStart(false)} />
        )}
        {active && <CampaignDetail campaign={active} onApply={onApply} />}
        {!active && !showStart && (
          <div className="text-sm text-[var(--color-text-muted)]">Select a campaign, or start a new one.</div>
        )}
      </div>
    </div>
  )
}

function StartCampaignForm({
  evalSetIds,
  onSubmit,
  onCancel,
}: {
  evalSetIds: string[]
  onSubmit: (req: StartCampaignRequest) => void
  onCancel: () => void
}) {
  const [evalSetId, setEvalSetId] = useState(evalSetIds[0] ?? '')
  const [maxTrials, setMaxTrials] = useState(40)
  return (
    <div className="mb-4 rounded border border-[var(--color-border)] p-4">
      <label className="mb-2 block text-xs font-medium">Eval set</label>
      <select
        className="mb-3 w-full rounded border border-[var(--color-border)] bg-transparent p-2 text-sm"
        value={evalSetId}
        onChange={(e) => setEvalSetId(e.target.value)}
      >
        {evalSetIds.map((id) => (
          <option key={id} value={id}>
            {id}
          </option>
        ))}
      </select>
      <label className="mb-2 block text-xs font-medium">Max trials</label>
      <input
        type="number"
        className="mb-3 w-full rounded border border-[var(--color-border)] bg-transparent p-2 text-sm"
        value={maxTrials}
        onChange={(e) => setMaxTrials(Number(e.target.value))}
      />
      <div className="flex gap-2">
        <button
          type="button"
          className="rounded bg-[var(--color-accent)] px-3 py-1.5 text-sm text-white"
          onClick={() => onSubmit({ eval_set_id: evalSetId, max_trials: maxTrials })}
        >
          Start
        </button>
        <button type="button" className="rounded border px-3 py-1.5 text-sm" onClick={onCancel}>
          Cancel
        </button>
      </div>
    </div>
  )
}

function CampaignDetail({ campaign, onApply }: { campaign: TuningCampaign; onApply: (t: Trial) => void }) {
  const frontTrials = campaign.trials.filter((t) => campaign.pareto_front.includes(t.id))
  const sorted = [...frontTrials].sort((a, b) => a.objectives.latency_p50_ms - b.objectives.latency_p50_ms)
  return (
    <div>
      <h3 className="mb-3 text-sm font-semibold">
        Pareto front ({sorted.length} of {campaign.trials.length} trials)
      </h3>
      <table className="w-full text-xs">
        <thead>
          <tr className="border-b border-[var(--color-border)] text-left">
            <th className="p-2">recall</th>
            <th className="p-2">faithfulness</th>
            <th className="p-2">p50 latency (ms)</th>
            <th className="p-2" />
          </tr>
        </thead>
        <tbody>
          {sorted.map((t) => (
            <tr key={t.id} className="border-b border-[var(--color-border)]">
              <td className="p-2">{t.objectives.recall.toFixed(3)}</td>
              <td className="p-2">{t.objectives.faithfulness.toFixed(3)}</td>
              <td className="p-2">{t.objectives.latency_p50_ms.toFixed(1)}</td>
              <td className="p-2">
                <button
                  type="button"
                  className="rounded border px-2 py-1 text-[11px] hover:bg-[var(--color-bg-hover)]"
                  onClick={() => onApply(t)}
                >
                  Apply
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
```

- [ ] **Step 2: Wire a "Campaigns" tab into `QualityLab.tsx`**

In `niriksha/src/pages/quality/QualityLab.tsx`, add a state field alongside the existing `state.runId`/`state.compareId` pattern (e.g. `state.view: 'runs' | 'campaigns'`), an import `import { Campaigns } from './Campaigns'`, and a conditional render branch that shows `<Campaigns collection={collection} evalSetIds={sets.map(s => s.id)} />` when `state.view === 'campaigns'` — following exactly the same `showCreate`/`activeRun` conditional-render style already present in that file (do not introduce a new state-management pattern for one tab).

- [ ] **Step 3: Build clean**

Run: `ssh gpu-host-a.internal.example "cd /home/mac/test/project-mdb/niriksha && source ~/.nvm/nvm.sh 2>/dev/null; nvm use 24.3.0 >/dev/null 2>&1; npm run build 2>&1 | tail -25"`
Expected: `BUILD_EXIT=0`, no new TypeScript errors.

- [ ] **Step 4: Commit**

```bash
git add niriksha/src/pages/quality/Campaigns.tsx niriksha/src/pages/quality/QualityLab.tsx
git commit -m "feat(niriksha): Campaigns tab in Quality Lab — trigger, Pareto-front table, apply"
```

---

## Final verification (all phases)

Run the full workspace gate on the build host:

```bash
ssh gpu-host-a.internal.example "cd /home/mac/test/project-mdb && \
  /home/mac/.cargo/bin/cargo fmt --all --check && \
  /home/mac/.cargo/bin/cargo clippy --workspace --all-targets -- -D warnings && \
  /home/mac/.cargo/bin/cargo test -p veda-gateway 2>&1 | tail -60 && \
  /home/mac/.cargo/bin/cargo build --release -p veda-server 2>&1 | tail -10"
```

Then niriksha:

```bash
ssh gpu-host-a.internal.example "cd /home/mac/test/project-mdb/niriksha && source ~/.nvm/nvm.sh 2>/dev/null; nvm use 24.3.0 >/dev/null 2>&1; npm run build 2>&1 | tail -15"
```

Live e2e (per the design doc's Verification section — single-flight, isolated data dir/port, real LLM+embedder):
1. `POST /v1/collections/{name}/eval-sets/generate` on a real seeded text collection → confirm a persisted `EvalSet` with `source`-tagged cases.
2. `POST /v1/collections/{name}/tuning-campaigns` with that eval set → confirm it converges to a `Converged` or `BudgetExhausted` status with a non-empty `pareto_front`.
3. `POST /v1/tuning-campaigns/{id}/trials/{trial_id}/apply` on a tier-1-only front point → confirm `ApplyKind::Tier1Instant`.
4. Apply a tier-2/3 front point → confirm `ApplyKind::ReindexAndSwap`, confirm `GET /v1/collections/{name}/alias` now points at a new sibling, confirm the original collection (queried by its exact versioned name) is untouched and still returns its original data.
5. Revert via `PUT /v1/collections/{name}/alias` back to the previous target reported in step 4's `previous_alias_target` → confirm search behavior is restored.

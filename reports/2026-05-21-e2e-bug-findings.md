# Veda E2E test campaign — bug & UX findings (2026-05-21)

## UX-1: `/embed-insert` requires explicit `auto_embed` map (no field-level opt-in)

**Severity:** P2 (UX, not functional bug)

**Repro:**
```bash
curl -X POST http://localhost:6341/v1/collections/smoke/embed-insert \
  -H "Content-Type: application/json" \
  -d "{\"records\":[{\"text\":\"hello world\",\"label\":\"greet\"}]}"
# 422 — missing field `auto_embed`
```

**Actual:** caller must supply `{"auto_embed": {"v": {"source_field": "text"}}}` — a per-target mapping.

**Expected:** An endpoint named *embed-insert* should be able to infer the mapping when:
- The collection has exactly one vector field, OR
- The schema declares `auto_embed: true` (with optional `source_field`) at field-create time.

**Suspected fix area:** `crates/veda-gateway/src/rest/embed_insert.rs:24` (EmbedInsertRequest); schema in `crates/veda-core/src/schema.rs` (add `auto_embed: Option<AutoEmbedSpec>` to FieldDef).

**Effort:** ~1 day. Add schema field, derive default mapping in handler when `auto_embed` map is empty.

**Impact:** Major DX win for the most common case (one text field → one vector field).

## SEC-1: Login compares passwords in plaintext

**Severity:** P0 (security)

**Repro:** `crates/veda-gateway/src/rest/admin.rs:312` performs `user.password != req.password` — direct string comparison of cleartext passwords.

**Why it matters:** stored credentials are in cleartext (no Argon2/bcrypt hash). Filesystem read = total compromise. Constant-time comparison is also missing — timing-attack viable.

**Suspected fix area:** `crates/veda-gateway/src/rest/admin.rs::login` + `state.inner.users` storage shape. Switch to `argon2` (already idiomatic in Rust) for hashing on user create; verify with `argon2::PasswordVerifier::verify_password` (constant-time). Migration: rewrite users.json (or however state.users persists) with hashed values; flag old plaintext entries for forced rotation.

**Effort:** ~1 day including a `veda-admin reset-password` migration command.

**Impact:** removes a P0 supply-chain risk for any production install.

## SEC-2: No default admin user seeded

**Severity:** P1 (UX + ops)

**Repro:** fresh `veda-data-e2e` start; `POST /v1/auth/login {admin,admin}` returns 401 `invalid credentials`. No CLI to bootstrap a first user.

**Note:** routes don't appear to require auth on first contact (anonymous calls to /collections and /admin/metrics succeed). So no actual lock-out — but the auth surface is broken in practice for any operator who turns it on.

**Suspected fix:** `veda-admin create-user --username admin --role admin --password <generated>` subcommand; or `VEDA_BOOTSTRAP_ADMIN_PASSWORD` env to seed on first run.

**Effort:** ~half day.

**Impact:** unblocks JWT/RBAC testing and any production deployment.

## BUG-1: MVCC `?at=<past_ts>` does not filter records on a `use_mvcc:true` collection

**Severity:** P0 (functional regression vs documented behaviour)

**Repro:**
```
POST /v1/collections {name: mvcc_test, use_mvcc: true, fields: [v vector(4), label string]}
T0 = ts_micros()       # before any insert
POST /insert {v=[1,0,0,0], label=v1}
T1 = ts_micros()
POST /insert {v=[0.9,0.1,0,0], label=v2}
T2 = ts_micros()

POST /search {at=T0}   # expect 0 hits — see 2
POST /search {at=T1}   # expect 1 hit  — see 2
POST /search {at=T2}   # expect 2 hits — see 2 ✓
POST /search {}         # expect 2 hits — see 2 ✓
```

All four searches return both records, regardless of `at=`. The past-snapshot semantics from `documentation/rearchitecture-overview.md` and the `SearchRequest::at` rustdoc ("hits whose record isn't visible at this `read_ts` are filtered out") do not apply.

**Hypothesis:** either (a) `use_mvcc:true` on the create-collection request isn't reaching the storage layer's commit_ts stamping, (b) records inserted into a growing segment don't carry a commit_ts until sealed, or (c) the executor's pre-filter bitmap is empty so nothing is excluded.

**Likely investigation start:**
1. `crates/veda-gateway/src/rest/collections.rs::create` — confirm `req.use_mvcc` reaches `schema.use_mvcc(...)`.
2. `crates/veda-storage/src/growing_segment.rs` — confirm `commit_ts: Some(...)` is stamped on inserted docs when `use_mvcc:true`.
3. `crates/veda-query/src/local_shard.rs::search_shard` — confirm `visible_doc_ids_at(read_ts)` is called and intersected into `allowed` when the collection has MVCC on.
4. Check whether the unit tests in `crates/veda-storage/tests/mvcc_isolation_gc_safety.rs` cover this end-to-end or only the storage internals (the storage unit tests work — see commit `0742654` — so the bug is in the wire-up between gateway and storage).

**Impact:** snapshot isolation reads do not work end-to-end. Cross-region reader gating (Phase H) depends on this, so it's likely affected too. Atomic reindex's "swap at commit_ts" promise also relies on it.

**Effort:** half-day to root-cause, half-day to fix + regression test (an existing fast test `crates/veda-gateway/tests/mvcc_test.rs` should exercise the same scenario).

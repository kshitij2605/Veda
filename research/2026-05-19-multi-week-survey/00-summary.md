# Multi-week-engineering survey — what's reinvent-the-wheel vs genuinely missing

Generated 2026-05-19 by a five-agent parallel research pass.
Files in this directory cover each topic in depth. This is the
TL;DR.

The user's question was: "are we reinventing the wheel?" Headline:
**partly yes, partly no.** Three of the five items can be cut from
multi-month down to days-or-weeks by adopting existing OSS. One
should be deferred entirely. One was misclassified as multi-month
and is actually session-scope.

| # | Item | Multi-month? | Best OSS path | Realistic effort |
|---|------|---|---|---|
| 1 | Cross-shard transactions | Yes | TiKV `txn_types` + `concurrency_manager` (Apache-2.0, Rust) | 28–43 weeks (vs 30–50 from scratch). Still long; but TiKV is a legit base. |
| 2 | Jepsen-class linearisability | **No** — wrong framing | madsim + elle-cli (deterministic simulation + Kingsbury's actual Elle checker) | **3–4 weeks** total. Veda only needs to record histories. |
| 3 | Pre-vote (consensus swap) | Borderline | tikv/raft-rs 0.7.0 is the only real candidate (and it's in maintenance) | **3–5 weeks** to swap, OR **hours-to-days** to upgrade openraft to 0.10 alpha which addresses the issue. **Recommendation: don't swap.** |
| 4 | Multi-tenancy / Spaces | Partial | No OSS to vendor — Pulsar-shape (tenant > namespace > collection) hierarchy is the converging design | **3–4 weeks** for logical-only isolation; ~2 months for cryptographic |
| 5 | Auth beyond bearer token | **No** — was misclassified | `jsonwebtoken` + JWKS cache, `cedar-policy` later | **1–3 days** for JWT-against-IdP. The "Phase ∞" framing was wrong. |

---

## Per-item one-paragraph summary (full detail in the numbered files)

### 1. Cross-shard transactions — `01-cross-shard-transactions.md`

**Vendoring candidate: TiKV.** CockroachDB became proprietary in Nov 2024 (Cockroach Software License); even pre-2024 commits are now Business Source not Apache, so a vendoring path was always tainted. FoundationDB is Apache-2.0 but C++ with deterministic-simulator coupling — only the design is borrowable. **TiKV is Apache-2.0 Rust** with a production-tested optimised-Percolator implementation (async-commit ≈ Cockroach's Parallel Commits, plus 1PC, pessimistic locks, parallel prewrite). Their `components/concurrency_manager` (~4k LoC) and `components/txn_types` (~3k LoC) are *relatively* self-contained but pull in `kvproto`, `pd_client`, `tikv_util`. Even with maximum vendoring there's 16–25 weeks of Veda-specific integration left: engine adapter onto our segment storage, openraft (not raftstore) integration, TSO-or-HLC decision, vector-aware transaction semantics, REST surface, observability hooks. **Honest cost: 28–43 engineer-weeks.** A scoped solution (single-region SI, no cross-region, no pessimistic locks) likely covers 95% of Veda's needs at 30–40% of full cost — worth scoping the user-visible semantics before committing to full TiKV parity.

### 2. Jepsen-class linearisability checker — `02-linearisability-checkers.md`

**The framing was wrong. The state of the art is no longer Jepsen-the-framework.** Kyle Kingsbury himself has moved to **Elle** — a cycle-detection consistency checker that's linear-in-history-length (vs Knossos's exponential), can verify 22M-transaction histories in ~2 minutes, and is exposed via standalone `elle-cli`. The newer pattern (TigerBeetle VOPR, RisingWave, Turso, Resonate, S2) is **deterministic simulation + cycle-detection checker on recorded histories** — and there's a Rust-native implementation: `madsim` (Apache-2.0, RisingWave-proven). The recommended stack is **madsim** (in-process cluster simulator that replaces tokio's network/IO surface behind `cfg(madsim)`, ~1–3 weeks to integrate) **+ elle-cli** (Java JAR that reads JSON history files, ~3–5 days to wire up). Other surveyed tools: Porcupine (Go, 1000–10000× faster than Knossos, used by etcd/TiKV — strong runner-up); Maelstrom (explicitly pedagogical, wrong shape for an existing DB); Turmoil (smaller scope than madsim); Stateright (model checker, useful for proving consensus crate but not whole-system); Loom/Shuttle/Kani (single-process, wrong layer); P/TLA+/FizzBee (design-time formal verification, orthogonal); Antithesis (closed-source commercial equivalent). **Honest cost: 3–4 engineer-weeks total** — Veda just needs to record histories against the existing REST API, write a sequential model (~200 LOC), and wire elle-cli for offline checking.

### 3. Pre-vote (consensus swap) — `03-consensus-crates.md`

**Recommendation: don't swap.** The disruptive-election failure mode pre-vote prevents is bounded (one election-timeout of unavailability on partition heal, no data loss), and openraft 0.9's `Vote` design (term+leader_id tuple + `effective_leader` heartbeat tracking) already prevents the worst of it. If forced to swap, the only realistic target is **tikv/raft-rs 0.7.0** — first-class `Config::pre_vote = true`, Apache-2.0, TiKV-proven — but it's in maintenance mode (last release 2023-03-07) and the API style is fundamentally different (pull-based `Ready` loop, sync core, BYO log/network/state machine). Swap cost: **3–5 weeks**, dominated by rewriting `raft_storage.rs` (954 lines) and the `raft_coordinator.rs` driver loop (638 lines, would grow ~300–400 lines for the Ready-processing state machine). Critically, **openraft 0.10-alpha.18 (2026-04-20) explicitly addresses simplifying vote/pre-vote in its roadmap** — a 0.9 → 0.10 point upgrade is hours-to-days vs weeks for a crate swap. Cheaper mitigations: enable `single-term-leader` feature flag, add application-level CheckQuorum (~100 lines watching `RaftMetrics`), tune election-timeout jitter, rate-limit term-bumps in `peer_health.rs`. Other candidates (riteraft, raftbare, little_raft, canonical/raft, hashicorp+FFI, etcd+FFI, paxakos) all ruled out.

### 4. Multi-tenancy / Spaces — `04-multi-tenancy.md`

**No OSS to vendor — port the design, not the code.** The closest systems are either wrong-language (Pulsar/Weaviate/Milvus on JVM/Go) or coupled to their own storage engine (Qdrant in Rust but tightly bound to its HNSW/shard manager). The converging design across the field — Pinecone, Mimir, Qdrant, Weaviate, Milvus — is **`tenant > namespace > collection` (Pulsar-shape)** with per-tenant shard policies at the collection level. **Three isolation tiers** per collection: `shared` (one HNSW with payload filter on tenant — Pinecone/Mimir/Qdrant default), `partitioned` (Curator-shape sub-indexes inside a shared structure — Qdrant `payload_m`), `dedicated` (one shard per tenant with ACTIVE/INACTIVE/OFFLOADED lifecycle — Weaviate). Three hard invariants for cross-tenant prevention: path-vs-claim consistency, shard-vs-tenant consistency, storage-key prefix discipline — all need property tests. The Curator paper (arXiv 2401.07119, Jan 2024) shows shared-structure-with-tenant-subtrees is the converging design. Qdrant 1.16 (Nov 2025) just shipped Tiered Multitenancy — strong external validation that "shared fallback + dedicated promotion" is the right shape. `axum-tenancy` (0.1.0) and `rust-rbac` are not production-grade. **Honest cost: 3–4 weeks for logical-only isolation** (shared + dedicated tiers, no partitioned); add 2–3 weeks for Curator-shape partitioned; cryptographic per-tenant isolation is a separate ~2–3 month effort and not recommended for v1.

### 5. Auth beyond bearer token — `05-auth-rust-crates.md`

**The "Phase ∞ — multi-month" framing was wrong. The Rust crate ecosystem covers everything; Veda just needs to wire them up.** Recommended stack:
- **Adopt now (1–3 days):** `jsonwebtoken` 10.4 (Keats, MIT, de-facto standard) + a ~150-LOC JWKS cache (or `async-oidc-jwt-validator`). Closes the gap from static-bearer to IdP-delegated JWT validation.
- **Adopt when needed:** `cedar-policy` 4.11 (Apache-2.0, sub-millisecond, formal-model-verified) when RBAC outgrows simple roles. `openidconnect` 4.0.1 if Niriksha hosts its own login flow. `rustls`+`axum-server` mTLS for intra-cluster Raft traffic.
- **Skip:** `oxide-auth` (Veda has no business issuing OAuth tokens — let the IdP do it), `samael` (still 0.0.x), `biscuit` (smaller community), `casbin-rs` (Cedar is stronger for greenfield).

The migration is *extending* the existing `auth::auth_middleware` (`rest/mod.rs:127`), not rewriting it. **~300 net new LOC for full IdP-delegated JWT validation. Honest cost: 2 engineer-days, 1 PR.** Keycloak/Auth0/Okta/Entra/Google Workspace integrate as three config fields each, not five separate integrations.

---

## What changes from the previous "won't ship in a session" framing

**Three items I should have called session-or-week-scope, not month-scope:**

- **Auth** — was "Phase ∞ multi-month." It's actually 1–3 days for the JWT-against-IdP first cut. The crate ecosystem already wrote the auth code; Veda just wires it.
- **Jepsen-class checker** — was "weeks of SMT-style search." It's actually 3–4 weeks total because the SMT-style search is the wrong approach in 2026 — cycle-detection (Elle) is.
- **Pre-vote** — was "2–4 weeks." It's actually "don't swap; wait for openraft 0.10."

**Two items where multi-week is honest:**

- **Cross-shard transactions** — still 28–43 weeks even with TiKV vendoring. The architectural integration tax is real.
- **Multi-tenancy** — 3–4 weeks logical-only; 2–3 months for cryptographic. The honest path is logical-only first, then revisit.

---

## Concrete next-engagement proposals

If you want me to push on any of these, **here's what's session-scope vs needs longer**:

1. **Auth (JWT against IdP)** — session-scope. ~1–3 days, ~300 LOC. Real work I can ship.
2. **Linearisability harness upgrade** — partial session-scope. Wiring elle-cli + writing the sequential model (~200 LOC) is doable; integrating madsim end-to-end is a week or two.
3. **Multi-tenancy logical-only** — partial session-scope. Threading `tenant_id` through `ClusterCommand` + `AppState` + per-tenant schema map is doable; the property tests for cross-tenant invariants are the bulk and might spill out.
4. **TiKV-based transactions** — not session-scope. Even the scoped version is 8–12 weeks.
5. **Consensus swap** — not session-scope. Don't do it; wait for openraft 0.10.

The user's prior question — "can we use existing code directly instead of starting from scratch?" — the answer is **yes for auth (immediately) and partially yes for linearisability/transactions/tenancy (with significant integration tax).** Pre-vote and the transactions architectural integration remain genuine multi-week engineering.

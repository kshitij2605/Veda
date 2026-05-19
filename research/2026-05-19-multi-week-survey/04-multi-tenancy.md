# Multi-Tenancy: OSS Survey and Recommended Adoption Path

**Date:** 2026-05-19
**Status:** Research findings -- not yet a design proposal
**Scope:** Per-tenant data isolation, per-tenant auth, resource quotas, and cross-tenant query prevention for Veda. Inventory of OSS systems we could vendor, adapt, or learn from.

---

## Executive Summary

Multi-tenancy is the single largest gap in Veda's current scope: it was explicitly deferred from the original plan and has no first-class primitives in either `veda-coordinator` or `veda-gateway`. After surveying ten OSS systems that solve closely related problems, **the recommended adoption path is a three-level hierarchy modelled on Apache Pulsar -- `tenant > namespace > collection` -- with logical isolation enforced at the gateway and shard router**, plus per-tenant HNSW shards modelled on Weaviate's bucketed architecture for the collections that need physical isolation.

This is a **hybrid strategy** because no single OSS system maps cleanly onto a vector DB:

- **Conceptual model: Pulsar** (Apache-2.0). Pulsar's tenant > namespace > topic hierarchy is the cleanest, best-documented model for a server-side tenancy primitive. We adapt the *concept*, not the code (Pulsar is JVM).
- **Physical isolation: Weaviate-shape** (BSD-3). One shard per tenant *only when explicitly opted-in at the collection level*, with ACTIVE/INACTIVE/OFFLOADED lifecycle. We adapt the *concept*; the bucketed LSM design is already present in `veda-storage`.
- **Query routing: Pinecone-shape namespaces** (closed source, learn from docs). Single `namespace` parameter on every data-plane request, served from a distinct logical partition; the simplest possible API shape and easy to map onto the existing `/v1/collections/{name}` routes.
- **Noisy-neighbour mitigation: Qdrant 1.16 tiered tenancy** (Apache-2.0). Shared "fallback shard" for the long tail of small tenants, promotion to dedicated shard when a tenant grows. Adapt the *concept*; the Rust implementation is available to read.
- **Resource quotas: Kafka-style per-principal quotas** plus ClickHouse-style `KEYED BY` quotas. Both are conceptual takeaways.
- **Auth model: Kibana Spaces RBAC plus Vault namespace delegation**. RBAC scoped to tenant; delegated admin per tenant.

**No OSS code base will be vendored wholesale.** The vendor cost is essentially zero because the systems closest to our problem (Pulsar, Weaviate, Qdrant, Mimir) are either in different languages or so tightly coupled to their existing storage layers that lifting the multi-tenancy module would be more expensive than re-implementing it natively in Rust. The recommended path is therefore **port the design, not the code**.

**Estimated work** (detailed at the end): logical-only isolation lands in ~3 to 4 weeks of focused engineering. Cryptographic per-tenant isolation (envelope encryption, key separation) is a separate ~2 to 3 month effort and is not recommended for v1.

---

## 1. Survey of Conceptual Models

The first question is not "how do we isolate" but "what is a tenant". Five distinct conceptual models appear in the OSS landscape:

### 1.1 Flat namespace (Pinecone, Qdrant payload-partition, Mimir, Loki)

A single global identifier (`namespace`, `group_id`, `X-Scope-OrgID`) on every request. No hierarchy; the tenant *is* the namespace.

- **Pros:** Trivially simple API; trivial routing; smallest implementation surface; works across millions of tenants if backed by physical partitioning.
- **Cons:** No room to model "team within a customer" or "environment within an app". Every grouping becomes naming convention (`acme-prod-foo`).
- **Best fit:** Single-tier SaaS where each tenant is a customer and there is no sub-structure.

### 1.2 Two-level: organisation > resource (Grafana, ClickHouse role-per-tenant)

An org wraps the user database and the resources users own. Quotas typically attach at the org level.

- **Pros:** Familiar from B2B SaaS; admin delegation is clean.
- **Cons:** Forces all resources of one customer into one org, which collapses dev/staging/prod separation unless the customer creates multiple orgs.

### 1.3 Three-level: tenant > namespace > topic/collection (Pulsar, Kafka prefixed-ACL convention)

Tenant for billing and authz, namespace for policy grouping (retention, quotas), topic/collection for data.

- **Pros:** Maps directly onto enterprise reality (org > project > resource). Policies attach at the natural level (quotas on tenant, retention on namespace, indexes on collection). Cluster-spanning by design.
- **Cons:** More objects to track; one extra hop in the URL/API. Worth the cost.

### 1.4 Hierarchical/recursive (Vault Enterprise namespaces)

Namespaces nest arbitrarily deep (`A/B/C`). Delegated admins can create child namespaces.

- **Pros:** Reflects org charts (org > BU > team > app).
- **Cons:** Significant complexity in policy inheritance, audit, and quota composition. Vault charges enterprise dollars for this and even Pulsar/Weaviate stop at two-to-three levels. Overkill for a vector DB.

### 1.5 Per-resource ACL with no tenancy primitive (Kibana Spaces plus Elasticsearch RBAC)

There is no "tenant" object; a "space" is just an ACL-tagged container for saved objects, and the underlying Elasticsearch indexes are independently access-controlled. The "tenancy" is whatever the operator builds out of role-to-index mappings.

- **Pros:** Maximally flexible; no abstraction to outgrow.
- **Cons:** Tenancy is a *convention*, not enforced by the system. The Elastic docs themselves warn the data layer remains accessible across spaces and isolation needs complementary controls. Easy to misconfigure.

### Verdict for Veda

The three-level Pulsar model is the right primitive. It is the only model that lets us put **quotas on the tenant** (billing-ready), **isolation/retention policies on the namespace** (operations-ready), and **schema/index on the collection** (developer-ready) without overloading any single object. Pinecone's flat model is simpler but limits future B2B features. Vault's recursive model is more flexible but invites complexity we will not pay back.

---

## 2. Survey of Candidates

Each candidate is rated on: **License**, **Conceptual model**, **Enforcement** (cryptographic vs logical), **Migration story**, **Vendor cost** (could we lift code).

### 2.1 Apache Pulsar -- the conceptual reference

- **License:** Apache-2.0 (vendorable; not relevant -- it is JVM).
- **Conceptual model:** Tenant > namespace > topic. Tenants own auth/authz schemes and cluster assignment; namespaces own policies (retention, storage quota, message TTL, isolation policy); topics inherit from namespace.
- **Enforcement:** Logical. Tenant identity is established at auth time and bound to the connection; every broker operation runs `canConsume(tenant, namespace, topic)` etc. against a policy cache. Policy changes propagate via a `__change_events` system topic that brokers subscribe to.
- **Migration story:** Pulsar always had tenancy; not relevant here, but the docs describe quota-then-isolation-policy escalation which is directly portable.
- **Vendor cost:** Concept only. Re-implementation in Rust is straightforward: three structs, a policy cache, and middleware in `veda-gateway` that extracts `(tenant, namespace, collection)` from request path.

**What we take:** the data model, the placement of quotas (tenant), the placement of retention/isolation (namespace), and the propagate-via-event-log pattern for policy changes (we already have a metadata log in `veda-coordinator` -- this is a natural fit for Raft once we land it).

### 2.2 Weaviate -- the per-tenant shard reference

- **License:** BSD-3-Clause (vendorable, but Go).
- **Conceptual model:** Collection with `multiTenancyConfig: { enabled: true }`. Each tenant is a *shard within the collection* -- not a separate object hierarchy.
- **Enforcement:** Physical at the shard level. One LSM bucket family per tenant. Cross-tenant queries are impossible by construction because the query layer takes the tenant as a routing parameter and never opens shards outside it.
- **Lifecycle:** Tenants have explicit states: ACTIVE (in-memory), INACTIVE (on local disk, fast warmup), OFFLOADED (in S3, slow warmup). Lazy shard loading: shards open on first access. Up to ~50,000 active shards per node and ~1M tenants per cluster.
- **Migration story:** Collection-level opt-in (`multiTenancyConfig`) means existing single-tenant collections stay unchanged. New collections opt in.
- **Vendor cost:** Concept and lifecycle model. The Go code is not portable but the design is documented well enough to port in ~1 week of work given our existing segment/bucket infrastructure in `veda-storage`.

**What we take:** the *collection-level opt-in*, the ACTIVE/INACTIVE/OFFLOADED lifecycle, and the lazy-loading pattern. Existing single-tenant collections in Veda are untouched; tenant-enabled collections get a sharding layer keyed by tenant ID.

### 2.3 Qdrant -- the tiered tenancy reference

- **License:** Apache-2.0 (Rust, readable, partially vendorable).
- **Conceptual model:** Two-layered. Default is **payload partitioning** on `group_id` within a shared collection (uses HNSW `payload_m` to build per-group sub-indexes inside one global index). Qdrant 1.16 (released Nov 2025) introduces **Tiered Multitenancy**: a shared "fallback shard" for the long tail of small tenants plus **dedicated shards** for promoted large tenants; routing is transparent via shard-key selectors.
- **Enforcement:** Logical for payload-partition (filter on `group_id`), physical-ish for the dedicated-shard tier.
- **Migration story:** Promotion from shared to dedicated shard is online and transparent to clients.
- **Vendor cost:** Highest of any candidate. Qdrant is Rust; the shard transfer plus payload-index code is in the same workspace shape as ours. We will not vendor wholesale (license fine, but their HNSW and shard code are coupled to their storage manager), but we **can read the code** as the closest reference implementation we have.

**What we take:** the *tiered* design itself -- a recognition that tenant-size distribution is bimodal and the right architecture serves both tails. We can adopt this in two phases: (a) payload-partition first, (b) promotion-to-dedicated-shard later, gated on demand.

### 2.4 Milvus -- the four-tier reference

- **License:** Apache-2.0.
- **Conceptual model:** Four nested options -- Database, Collection, Partition, Partition Key. Database-level isolation is heaviest; Partition Key (introduced 2.2.9) uses a designated payload column to bucket tenants inside one collection. Supports >10M tenants by Partition Key with strong search performance.
- **Enforcement:** Logical (partition routing on key).
- **Migration story:** Opt-in at collection creation (`partition_key_field`).
- **Vendor cost:** Concept only.

**What we take:** the recognition that there are *multiple isolation levels* and the customer should choose -- the Partition Key idea aligns exactly with Qdrant's payload partitioning.

### 2.5 Pinecone -- the namespace API reference

- **License:** Closed source. Learn from docs only.
- **Conceptual model:** Flat namespaces inside an index. Up to ~100k per index (more on request), each physically isolated in serverless.
- **Enforcement:** Physical via per-namespace segments; queries cannot cross namespaces by construction.
- **API shape:** Every data-plane call takes a `namespace` parameter. Simple and copy-able. This is what we should put on our REST surface.

**What we take:** the API ergonomics (`namespace` parameter pattern) and the "1 RU per 1 GB namespace vs. 100 RUs for filtered scan of 100 GB" performance argument -- which is the most concise framing of why physical isolation matters for vector search.

### 2.6 Pinecone-style versus Milvus-style isolation -- the noisy-neighbour question

Curator (Jin et al., arXiv 2401.07119, January 2024) is the academic statement of the tradeoff: per-tenant indexes give per-tenant query performance but blow up memory; shared indexes with metadata filtering give memory efficiency but slow queries. Curator's contribution is a *tenant-specific clustering tree compactly encoded as a subtree of a shared tree* -- which matches Qdrant's `payload_m` parameter conceptually. **This is the strongest signal that the future of vector tenancy is "shared structure, tenant-specific sub-indexes", not "one HNSW per tenant".** We should plan for it.

### 2.7 Apache Kafka -- the quota reference

- **License:** Apache-2.0.
- **Conceptual model:** No first-class tenant object; tenancy is convention (prefixed topic names plus prefixed ACLs). Quotas attach to user principals (`user@bytes-in/out, request-percentage, connections`).
- **What we take:** The four-requirement framework -- (1) authn, (2) authz, (3) quotas, (4) namespace -- as our acceptance criteria for "Veda is multi-tenant". And `KEYED BY` user quotas -- the right shape for our rate-limit plus token-bucket implementation.

### 2.8 ClickHouse -- the in-engine RBAC reference

- **License:** Apache-2.0.
- **Conceptual model:** Three options -- shared table with `tenant_id` column plus row policy; database per tenant; cluster per tenant. The recommended pattern is the first: shared MergeTree with `tenant_id` first in `ORDER BY` plus `KEYED BY` quotas.
- **What we take:** The ordering-key tactic (`tenant_id` first) maps directly to our HNSW sub-indexing strategy. The `KEYED BY` quotas pattern is portable.

### 2.9 Postgres Row-Level Security -- the discipline reference

- **License:** Postgres license (vendorable in concept).
- **Conceptual model:** Policy on each table; policies use `current_setting('app.current_tenant')` to filter. Not a library to vendor.
- **What we take:** Three durable lessons. (1) **Centralised filter discipline**: every read/write sees the filter, not just the ones developers remember to write. (2) **GUC pattern**: a per-request tenant context set at the start of the request and torn down at the end -- this is exactly our `tracing::Span` story plus an extractor on every handler. (3) **Index ordering**: every tenant-scoped table must have `tenant_id` as the leading index key, or RLS destroys performance. This is the single most-cited RLS gotcha and applies verbatim to our scalar indexes and our HNSW partitions.

### 2.10 Kibana Spaces plus Elasticsearch RBAC -- the cautionary tale

- **License:** Elastic 2.0 / SSPL (do not vendor).
- **Conceptual model:** Spaces are metadata containers; isolation is achieved by binding roles to index patterns separately. The Elastic docs explicitly warn that spaces alone do not isolate underlying data.
- **What we take:** Mostly *what not to do* -- never let the tenancy be a UI-only abstraction that pretends to enforce. If we have a tenant object, the data layer must respect it.

### 2.11 HashiCorp Vault namespaces -- the delegation reference

- **License:** Enterprise-only (do not vendor).
- **Conceptual model:** Recursive namespaces with delegated admin. `X-Vault-Namespace` header on every request.
- **What we take:** The *header convention* (`X-Veda-Tenant` / `X-Veda-Namespace`) plus the *delegated admin* idea -- a per-tenant root user who can create sub-users without bothering the platform admin. We do not adopt recursive nesting.

### 2.12 Grafana Mimir / Loki -- the header-routing reference

- **License:** AGPL-3.0 (Mimir) / AGPL-3.0 (Loki). Not vendorable into an Apache-2.0 project.
- **Conceptual model:** Single `X-Scope-OrgID` header on every API call; storage paths physically partitioned by org ID; no `tenant` object beyond the ID itself.
- **What we take:** Header-based routing, blob-storage path prefixing by tenant, and the cross-tenant query federation pattern (an explicit list of tenant IDs in the header, gated by admin permission) for the rare "give me a global view" case.

### 2.13 Rust crates

- **`axum-tenancy`** (0.1.0, MIT, ~undocumented): single contributor, 0% documented, builds on `axum-login`. Not production-grade. Useful only as a starting sketch.
- **`rust-rbac`**: a permissions library, not a tenancy library. Useful inside `veda-gateway` for the RBAC piece but does not solve isolation.
- **General assessment:** No production-grade Rust crate exists for the tenancy primitive. We will write our own; `veda-gateway` is the right home.

---

## 3. Recommendation: The Pulsar-shape Hierarchy

**Adopt a three-level model: `tenant > namespace > collection`**, with isolation policy chosen per-collection (Weaviate-style opt-in), payload-partitioning for the long tail (Qdrant/Curator-style), and dedicated shards for promoted tenants.

### 3.1 Objects

- **Tenant** -- billing/auth unit. Owns: API keys, RBAC roles, total-quota (storage GB, QPS, RU/month), default namespace. Maps onto the future "customer" object.
- **Namespace** -- policy unit inside a tenant. Owns: retention, default isolation policy, region affinity, replication factor, embedding-model defaults. Equivalent to "project" or "environment".
- **Collection** -- existing object, gains `tenant_id` and `namespace` fields. Inherits namespace policies. Existing single-tenant collections live in an implicit `__system/__default` tenant/namespace -- zero-migration default.

### 3.2 Isolation level per collection

Operator picks one of:

1. **`shared`** -- one HNSW, tenant filter on payload. Suits the long tail. Default for new collections under a namespace marked `tier: shared`.
2. **`partitioned`** -- payload-partition HNSW (`payload_m > 0`, `m = 0`), per-tenant sub-indexes inside one collection. Suits medium tenants. The Qdrant-1.16 / Curator design.
3. **`dedicated`** -- one shard per tenant (Weaviate-shape). Suits large tenants. Lifecycle states ACTIVE/INACTIVE/OFFLOADED. Promotion from `partitioned` to `dedicated` should be online.

### 3.3 Enforcement points

- **`veda-gateway` middleware**: extracts `(tenant, namespace)` from JWT claim or `X-Veda-Tenant` header; rejects requests where the URL path's tenant does not match the claim; sets a request-scoped `TenantContext`.
- **`veda-coordinator`**: shard router takes `TenantContext` and refuses to route to a shard whose `tenant_id` does not match. Mirror of the Pulsar broker pattern.
- **`veda-index`**: HNSW search functions take a `tenant_filter` parameter at the layer where Qdrant takes `group_id`. For `shared` collections, this is a payload filter. For `partitioned` collections, this restricts traversal to the tenant's sub-tree. For `dedicated`, it is a shard selection.
- **`veda-storage`**: blob paths prefixed `tenant=<id>/namespace=<ns>/collection=<col>/...` -- mirrors Mimir/Loki. Quota accounting attaches at the prefix level.

### 3.4 Cross-tenant query prevention -- the must-pass invariants

Three invariants, each enforceable with a unit test plus a property test:

1. **Path-vs-claim consistency**: the path `/v1/tenants/{t}/...` matches `TenantContext::id` or the request is 403. Property test: random `(claim, path)` pairs -- only equal pairs accept.
2. **Shard-vs-tenant consistency**: the shard router's `route(query)` never returns shards whose `tenant_id != query.tenant_id`. Property test: random query distributions over random tenant fixtures.
3. **Storage-key prefix**: every read/write to `veda-storage` carries a tenant prefix; a tenant prefix mismatch is a panic in debug builds and a 500 with audit-log entry in release. This is the moral equivalent of Postgres RLS being on by default.

---

## 4. Honest "What We Still Have to Build"

The OSS survey makes the design space clear, but enforcement is **not** something we can adopt -- it is the bulk of the work and is local to our codebase.

### What we get from the survey

- The conceptual model (Pulsar).
- The API shape (Pinecone namespace parameter, Vault header).
- The isolation taxonomy (Milvus four-level; we collapse to three).
- The tiered design rationale (Qdrant 1.16, Curator paper).
- The lifecycle vocabulary (Weaviate ACTIVE/INACTIVE/OFFLOADED).
- The quota shape (Kafka per-principal, ClickHouse `KEYED BY`).
- The RBAC discipline (RLS, Mimir header routing).

### What we have to write

- `Tenant`, `Namespace` types in `veda-core::schema`.
- A tenancy module in `veda-gateway` (middleware, JWT claim extractor, path validator).
- A tenant-aware shard router in `veda-coordinator`. The existing router routes by `(collection, shard_id)`; the new one routes by `(tenant, namespace, collection, shard_id)`.
- A tenant filter parameter in `veda-index`'s HNSW search entry points. For `partitioned`, this needs the Curator-style sub-index encoding -- non-trivial.
- A storage-prefix policy in `veda-storage`. Mostly mechanical.
- Per-tenant quota accounting -- a token bucket per `(tenant, resource)` and reservations in the gateway. We already have `metrics` in `veda-core`; this extends it.
- An auth/RBAC layer that maps API keys/JWTs to `(tenant, namespace, role)`. The current `auth` module in `veda-gateway` is single-tenant; this is a structural change.
- A migration path: an implicit default tenant/namespace so existing collections continue to work; an admin migration tool to relabel.
- A test surface: property tests for the three invariants above, plus a "fuzz the path" test that issues random cross-tenant requests and verifies 100% rejection.

### What stays deferred

- **Cryptographic per-tenant isolation** -- envelope encryption with per-tenant DEKs wrapped by a per-tenant KEK, rotation, KMS integration. This is a separate ~2-3 month effort and is not on the v1 path. Logical isolation plus storage prefixing is the right v1.
- **Recursive namespaces** (Vault-shape). Two levels is enough.
- **Cross-tenant federation** (Loki-shape). Useful eventually for admin/observability views; not v1.
- **Hard CPU/memory isolation between tenants** -- this is a kernel-level problem (cgroups, eBPF). v1 stays at logical limits plus admission control. Qdrant's tiered shards are a precondition before this becomes meaningful.

---

## 5. Estimated Work

Two cost bands -- the user asked for both explicitly.

### 5.1 Logical isolation -- ~3 to 4 weeks

Concrete breakdown (single engineer, with normal review overhead):

| Week | Scope | Verifiable artifact |
|------|-------|---------------------|
| 1 | `Tenant`, `Namespace`, `TenantContext` types; gateway extractor; path/claim validator; default-tenant migration. | Unit plus property tests for invariant #1; existing test suite green with default tenant. |
| 2 | Tenant-aware shard router in `veda-coordinator`; storage-key prefixing in `veda-storage`. | Property test for invariant #2; new integration test that cross-tenant requests are rejected. |
| 3 | RBAC overlay; per-tenant API keys; per-tenant quotas (token bucket). Document/OCR pipeline tenant-tagging. | New API surface documented; quota-exceeded test path. |
| 4 | Niriksha tenant switcher; admin endpoints (create/list/delete tenant, namespace); audit log; docs. | E2E test through Niriksha; CHANGELOG entry. |

This delivers `shared` and `dedicated` modes. `partitioned` (Curator-shape sub-indexes) is **not** in this estimate.

### 5.2 Cryptographic isolation -- ~2 to 3 months

| Phase | Scope | Note |
|-------|-------|------|
| 1 | Per-tenant DEK generation; envelope encryption of segment files; KMS abstraction (local plus AWS KMS). | ~3 weeks. |
| 2 | WAL encryption; in-memory zeroisation; key rotation. | ~3 weeks; touches hot paths. |
| 3 | Backup/restore with per-tenant DEK export; auditor APIs. | ~2 weeks. |
| 4 | Performance regression work -- AES-NI/AVX-VAES in `veda-core::simd`; benchmark gate. | ~2 weeks; this is where the budget overruns happen. |

Not recommended for v1. Logical isolation is sufficient for the use cases on the roadmap and gets to multi-tenant in a single quarter.

### 5.3 Curator-style partitioned indexes -- ~2 to 3 weeks of *research and prototyping* before any production work

Read the Curator paper, prototype in `veda-index`, benchmark against `shared` and `dedicated` on a 10k-tenant workload. Gate the decision on results. If we ship without this, we have `shared` (cheap, slow at scale) and `dedicated` (fast, expensive) -- the missing middle. Qdrant has it (`payload_m`); we will need it.

---

## 6. Closing Recommendation

Adopt the Pulsar-shape hierarchy with Weaviate-shape collection-level isolation opt-in. Vendor no code. Write ~3000 lines of Rust over a month. Defer cryptographic isolation and Curator-style partitioned indexes. Open issues now for the three remaining gaps: tenant migration tooling, partitioned indexes, and the federation query surface.

---

## Sources

- [Apache Pulsar -- Multi Tenancy](https://pulsar.apache.org/docs/4.0.x/concepts-multi-tenancy/)
- [Streaml.io: Multi-tenant messaging with Apache Pulsar](https://streaml.io/blog/multi-tenant-messaging-with-apache-pulsar)
- [Weaviate -- Rethinking Vector Search at Scale: Native Multi-Tenancy](https://weaviate.io/blog/weaviate-multi-tenancy-architecture-explained)
- [Weaviate Docs -- Multi-tenancy operations](https://docs.weaviate.io/weaviate/manage-collections/multi-tenancy)
- [Qdrant -- Multitenancy docs](https://qdrant.tech/documentation/manage-data/multitenancy/)
- [Qdrant 1.16 -- Tiered Multitenancy](https://qdrant.tech/blog/qdrant-1.16.x/)
- [Qdrant Tiered Multitenancy press release](https://www.businesswire.com/news/home/20251119343840/en/Qdrant-Introduces-Tiered-Multitenancy-to-Eliminate-Noisy-Neighbor-Problems-in-Vector-Search)
- [Pinecone -- Implement multitenancy using namespaces](https://docs.pinecone.io/guides/index-data/implement-multitenancy)
- [Pinecone Serverless architecture](https://www.pinecone.io/blog/serverless-architecture/)
- [Milvus -- Implement Multi-tenancy](https://milvus.io/docs/multi_tenancy.md)
- [Milvus -- Use Partition Key](https://milvus.io/docs/use-partition-key.md)
- [Milvus blog -- Multi-tenancy RAG best practices](https://milvus.io/blog/build-multi-tenancy-rag-with-milvus-best-practices-part-one.md)
- [Apache Kafka -- Multi-Tenancy](https://kafka.apache.org/42/operations/multi-tenancy/)
- [ClickHouse -- Multi tenancy](https://clickhouse.com/docs/cloud/bestpractices/multi-tenancy)
- [Postgres RLS for SaaS -- thenile.dev](https://www.thenile.dev/blog/multi-tenant-rls)
- [AWS Prescriptive Guidance -- Row-level security recommendations](https://docs.aws.amazon.com/prescriptive-guidance/latest/saas-multitenant-managed-postgresql/rls.html)
- [Kibana Spaces for multi-tenancy -- OneUptime](https://oneuptime.com/blog/post/2026-02-09-kibana-spaces-multi-tenant/view)
- [HashiCorp Vault -- Secure multi-tenancy with namespaces](https://developer.hashicorp.com/vault/tutorials/enterprise/namespaces)
- [Grafana Loki -- Manage tenant isolation](https://grafana.com/docs/loki/latest/operations/multi-tenancy/)
- [Grafana Mimir -- Authentication and authorization](https://grafana.com/docs/mimir/latest/manage/secure/authentication-and-authorization/)
- [Curator: Efficient Indexing for Multi-Tenant Vector Databases (Jin et al., 2024)](https://arxiv.org/abs/2401.07119)
- [Neon -- The Noisy Neighbor Problem in Multitenant Architectures](https://neon.com/blog/noisy-neighbor-multitenant)
- [Zilliz -- Multi-tenancy in scalability for vector DBs](https://zilliz.com/ai-faq/what-is-the-role-of-multitenancy-in-scalability-considerations-for-vector-databases-and-how-might-resource-isolation-be-handled-when-multiple-applications-share-the-same-infrastructure)
- [axum-tenancy on crates.io](https://crates.io/crates/axum-tenancy)
- [rust-rbac on crates.io](https://crates.io/crates/rust-rbac)

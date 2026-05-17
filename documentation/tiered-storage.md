# Tiered Storage Design

Veda supports three storage tiers per sealed segment. This document
describes what works today and what remains for follow-up work.

## Tiers

| Tier | Vector data | ANN index | Where it lives | Use case |
|------|-------------|-----------|----------------|----------|
| **Hot** | mmap-resident, in OS page cache | Built (HNSW / DiskANN / IVF-PQ / Flat) | Local disk | Active query traffic |
| **Warm** | mmap-resident | Not built; brute-force scan at search | Local disk | Cold segments that occasionally serve queries; trade query latency for ANN-index memory |
| **Cold** | Offloaded to S3 / MinIO | Not built | Object storage | Archival; needs copy-back to query |

## What works today (this commit)

- `SegmentMetadata.tier` field, default `Hot`, persisted in `metadata.json`.
- `SegmentManager::set_segment_tier(segment_id, tier)` — atomically updates
  the in-memory handle and rewrites `metadata.json` on disk.
- `LocalShardSearcher::build_indexes` skips ANN construction for `Warm`
  and `Cold` segments.
- `LocalShardSearcher::search_shard` falls back to brute-force scan over
  the segment reader's mmap'd vectors for segments without an ANN index.
  This gives correct results for Warm tier with a latency penalty
  proportional to `segment_row_count × dimension`.
- Backwards-compatible: pre-existing on-disk segments without a `tier`
  field deserialise as `Hot`.

## What does NOT yet work

### Cold-tier copy-back

`Cold` segments behave identically to `Warm` today — brute-force scan over
local mmap. The intended semantics are:

1. Mark segment Cold → mover task uploads segment dir to
   `s3://{bucket}/cold/{collection}/{segment_id}/`, then deletes the
   local copy.
2. Query against a Cold segment → returns a "tier=cold" sentinel in the
   search result so the caller can choose to promote.
3. `promote_segment(segment_id)` → downloads back to local, marks Warm
   (or Hot if `auto_warmup_on_promote` is set).

This needs a new background worker plus the segment-mover S3 dance, plus
a "needs-promotion" sentinel in `SearchResponse`. Estimated 1-2 days.

### Automatic tiering policy

Today, tier transitions are explicit: the operator calls
`set_segment_tier`. Real production wants an automated policy:

- LRU on segment access timestamps.
- Heat scores combining recency + frequency.
- Per-collection tier budgets ("at most 100GB hot").
- Demotion triggers: memory pressure, idle time.

The data the policy engine needs (access counts, last-access timestamp)
isn't tracked yet. Estimated 1-2 days for the policy engine plus access
instrumentation.

### Operator-facing API

There's currently no REST endpoint to set tiers — only the Rust API.
Plumbing this into the gateway as `POST /v1/admin/segments/{id}/tier`
would expose tier management to operators. Trivial follow-up.

## Operating notes

- Demoting a Hot segment to Warm frees the ANN index from memory at
  the next executor rebuild — typically the next insert into the
  collection. There is no current API to drop the index immediately.
- The OS page cache continues to manage the mmap'd vector data
  automatically for Warm segments. Under memory pressure, infrequently-
  accessed vectors page out; on access they page back in. This is
  usually fine but means Warm-tier query latency varies with system memory.
- ColPali / multi-vector fields and sparse fields are loaded the same
  way as dense fields and behave identically with respect to tier.
- Tier changes are persistent (metadata.json on disk) and survive restart.

# Tiered Storage Design

Veda supports three storage tiers per sealed segment. This document
describes what works today and what remains for follow-up work.

## Tiers

| Tier | Vector data | ANN index | Where it lives | Use case |
|------|-------------|-----------|----------------|----------|
| **Hot** | mmap-resident, in OS page cache | Built (HNSW / DiskANN / IVF-PQ / Flat) | Local disk | Active query traffic |
| **Warm** | mmap-resident | Not built; brute-force scan at search | Local disk | Cold segments that occasionally serve queries; trade query latency for ANN-index memory |
| **Cold** | Offloaded to S3 / MinIO | Not built | Object storage | Archival; needs copy-back to query |

## What works today

- `SegmentMetadata.tier` field, default `Hot`, persisted in `metadata.json`.
- `SegmentManager::set_segment_tier(segment_id, tier)` — atomically updates
  the in-memory handle and rewrites `metadata.json` on disk. **Hot/Warm
  only**; for Cold use the dedicated methods below.
- `LocalShardSearcher::build_indexes` skips ANN construction for `Warm`
  and `Cold` segments.
- `LocalShardSearcher::search_shard` falls back to brute-force scan over
  the segment reader's mmap'd vectors for segments without an ANN index.
  This gives correct results for Warm tier with a latency penalty
  proportional to `segment_row_count × dimension`.
- Backwards-compatible: pre-existing on-disk segments without a `tier`
  field deserialise as `Hot`.

### Cold-tier archival (S3 / MinIO copy-out and copy-back)

- `veda_storage::cold_storage::archive_segment_to_blob(blob_store, dir,
  collection, segment_id)` walks a sealed-segment directory and uploads
  every file as a separate blob under
  `cold/{collection}/{segment_id}/{relative_path}`.
- `restore_segment_from_blob(...)` is the inverse: it lists blobs under
  that prefix and writes each one back, preserving relative paths and
  subdirectories.
- `SegmentManager::demote_to_cold(segment_id, blob_store, collection)` is
  the high-level operator entry point. It:
  1. Uploads the segment's files via `archive_segment_to_blob`.
  2. Returns the metadata to the caller marked `tier = Cold` (the
     local handle is dropped — see "Cold segment registry" below).
  3. Removes the segment from `sealed_segments` and deletes the local
     directory. The shard no longer sees it on subsequent searches.
- `SegmentManager::promote_from_cold(segment_id, blob_store, collection)`
  reverses the operation: it downloads files back into
  `shard_dir/segment_{id}/`, re-opens a `SegmentReader`, marks the
  metadata as `Warm`, and reattaches it to `sealed_segments`. The next
  executor rebuild will brute-force scan it; an operator who wants Hot
  promotion can follow up with `set_segment_tier(id, Hot)`.
- Both helpers are storage-backend-agnostic: they work against any
  `BlobStore` implementation, so unit tests run against `LocalBlobStore`
  and production deployments use `ObjectBlobStore` (S3 or MinIO).

Per-file upload was chosen over a tarball to (1) avoid a tar dependency,
(2) let the blob store deduplicate at file granularity (multiple
segments share schema.json bytes), (3) let restore stream files
concurrently, and (4) allow a mid-archive failure to resume by listing
what's already uploaded.

## What does NOT yet work

### Cold segment registry

`demote_to_cold` returns the archived segment's metadata to the caller
but Veda does not yet maintain a persistent index of cold-but-archived
segments. The operator is responsible for tracking which segment IDs
have been archived so they can be promoted later. A future
`cold_registry.json` per collection would close this gap; the underlying
archive/restore code is already in place.

### "Search Cold without explicit promotion"

There is no transparent cold-tier search path. A search query will not
see archived segments at all. The intended user flow is:

1. Operator (or auto-policy) calls `demote_to_cold` — segment vanishes
   from query traffic.
2. To search it again, operator calls `promote_from_cold` first, then
   issues the query.

A "lazy promotion on filter match" path would need a cold-registry
lookup during query planning plus a streaming restore — a worthwhile
follow-up but not blocking for the archive functionality.

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

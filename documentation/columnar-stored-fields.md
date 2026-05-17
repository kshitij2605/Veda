# Columnar stored_fields format — design

Plan to replace the current `records.jsonl` + offset-table layout with
a Lucene-style columnar layout, where each stored field gets its own
file in the segment directory.

## Why

The current layout (`crates/veda-storage/src/segment.rs`):

- `records.jsonl` — one JSONL line per record, all scalar/blob fields
  serialized as `{"<field>": {"<TaggedFieldValue>": "..."}}`.
- Per-row byte-offset table built at `SegmentReader::open()` time.
- `mmap`'d for O(1) `get_record(doc_id)`.

This already wins on small records (4-8ms per search on 2.4K-row
rag_anything corpus). Where it loses:

1. **Reading one field forces parsing every field**. Search-side
   `populate_hit_fields` materializes a `Record`, then keeps only the
   non-vector fields it needs. With a few-KB record containing a
   `content` string plus other scalars, that's wasted JSON parsing per
   hit.
2. **No selective compression**. JSON values share one container so
   per-field codecs (zstd dict for text, fixed-size for int64) don't
   apply naturally.
3. **Cold-tier transfer cost**. The whole JSONL must round-trip even
   when only a subset of fields is read after promotion.

Columnar fixes all three for collections with wide records.

## Proposed layout

```
segment_{uuid}/
  metadata.json           # unchanged
  schema.json             # unchanged
  storage_format.txt      # "jsonl" | "columnar" — new
  records.jsonl           # legacy path; absent for new columnar segments
  columns/                # new
    <field>.col           # contiguous values for that field
    <field>.idx           # u64 offsets per row (variable-length only)
    <field>.meta          # codec, dictionary, null bitmap header
  id_to_uuid.idx          # u128 per row — global UUID column
  vectors/  scalars/  ...  # unchanged
```

Each `.col` file contains the concrete value bytes for a single field.
Encoding depends on `FieldType`:

| FieldType | Encoding |
|-----------|----------|
| `Int32` / `Int64` / `Float32` / `Float64` / `Bool` | Fixed-size, no separate `.idx` |
| `String` | Length-prefix varint + UTF-8 bytes; `.idx` holds row → byte-offset |
| `BlobRef` | Length-prefix varint + UTF-8 ref string; `.idx` for offsets |
| `Array<T>` | Recursive `.col` per element type, with run-length grouping |
| Nullable | Bitmap header in `.meta` |

The UUID column is special-cased because every record has exactly one,
and the existing `uuid_to_doc_id` index already exists.

## Migration

- `storage_format.txt` marks the segment's layout. SegmentReader picks
  at open time.
- Existing segments stay JSONL — no rewrite required. Cold-tier restore
  preserves the format.
- New segments default to columnar once the implementation is stable.
- A `segment-format` config knob (or schema flag) lets operators force
  one or the other while we gain confidence.

## Trait shape (proposed)

```rust
pub trait StoredFields {
    /// Read a single scalar/blob value for `(field, doc_id)` without
    /// touching other fields. The Lucene `Document` API analogue.
    fn get_field(&self, field: &str, doc_id: u32) -> Result<Option<FieldValue>>;

    /// Full `Record` reconstruction — for callers that genuinely need
    /// every field. Today's `SegmentReader::get_record` is this method.
    fn get_record(&self, doc_id: u32) -> Result<Record>;
}

impl StoredFields for JsonlStoredFields { /* current impl */ }
impl StoredFields for ColumnarStoredFields { /* new impl */ }
```

`SegmentReader::stored_fields` returns `&dyn StoredFields`. Search-side
hydrators (`populate_hit_fields` in `crates/veda-gateway/src/rest/search.rs`)
loop over the projection list and call `get_field` only for what's needed.

## Implementation steps

1. **Land the trait + JSONL impl** (~half a day, no behaviour change).
   Refactor existing `SegmentWriter.write_jsonl` and
   `SegmentReader::get_record` behind the trait.
2. **Write the columnar writer** (~1-2 days):
   - Encode each scalar type to its `.col` + `.idx` + `.meta`.
   - For fixed-width: direct slab of bytes.
   - For variable-width: prefix-length encoding + offset table.
   - Null bitmap in `.meta` if any null appears.
3. **Write the columnar reader** (~1 day): mmap each `.col`, read via
   the offset table, decode on demand.
4. **Wire format selection**: `storage_format.txt` + a `cargo test`
   that builds a segment in each format and confirms identical
   `get_record` output.
5. **Search-side projection** (~1 day): extend search response to
   accept `return_fields: ["content", "parent_doc_id"]` and have
   `populate_hit_fields` call `get_field` rather than `get_record`.
6. **Switch default** after a release cycle.

## What we get

Realistic targets, calibrated against Lucene's StoredFieldsFormat
benchmarks:

- 2-3× smaller per-record memory footprint on wide records (10+ fields).
- Hydration latency proportional to projection size, not total record
  size — for retrieve-3-of-20-fields queries this is the biggest win.
- Better cold-tier shipping: only relevant columns travel back from S3
  when promoted.

## What we don't get

- Faster search itself (HNSW + BM25 are already O(1) field-independent).
- Compression beyond per-column codec gains (these are real but small).
- Schema migration tooling (we still need to add/drop fields; not
  blocking, just a separate workstream).

## Status

This document is the design baseline; the implementation is the next
priority chunk of work after the Round-3+ items (BM25 hybrid, RaBitQ
opt-in, MRL, in-graph filter, recovery) all landed. Not yet started.

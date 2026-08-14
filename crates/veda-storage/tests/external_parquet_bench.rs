//! Benchmarks for read-in-place external Parquet collections.
//!
//! Measures what the design doc
//! (`documentation/2026-08-14-external-parquet-collections-design.md`)
//! only estimated: `load` and `stamp` wall time, the resident size of the
//! materialized vector column, on-disk Parquet size, and the extra cost a
//! filter predicate adds to `search` relative to an unfiltered query.
//!
//! Run with:
//! `cargo test -p veda-storage --test external_parquet_bench -- --ignored --nocapture`

use std::sync::Arc as StdArc;
use std::time::Instant;

use arrow::array::{ArrayRef, FixedSizeListArray, Float32Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;

use veda_core::record::{FieldValue, Record};
use veda_core::schema::ExternalSource;
use veda_core::types::DistanceMetric;
use veda_storage::blob_store::LocalBlobStore;
use veda_storage::external_parquet;

const DIM: usize = 768;

/// Write a Parquet file with `n` rows at `dim`, a string `id` column, a
/// `FixedSizeList<Float32>` `embedding` column, and a string `region`
/// column alternating between `"us"` and `"eu"` (used by the filtered
/// benchmark; `"us"` matches every other row, so the predicate is roughly
/// 50% selective rather than trivially near-0% or near-100%).
fn write_parquet(path: &std::path::Path, n: usize, dim: usize) {
    let ids: Vec<String> = (0..n).map(|i| format!("doc{i}")).collect();
    let regions: Vec<&str> = (0..n)
        .map(|i| if i % 2 == 0 { "us" } else { "eu" })
        .collect();

    let mut flat: Vec<f32> = Vec::with_capacity(n * dim);
    for i in 0..n {
        for d in 0..dim {
            flat.push(((i * dim + d) % 1000) as f32 / 1000.0);
        }
    }
    let values = Float32Array::from(flat);
    let item_field = StdArc::new(Field::new("item", DataType::Float32, false));
    let vectors =
        FixedSizeListArray::try_new(item_field.clone(), dim as i32, StdArc::new(values), None)
            .unwrap();

    let schema = StdArc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new(
            "embedding",
            DataType::FixedSizeList(item_field, dim as i32),
            false,
        ),
        Field::new("region", DataType::Utf8, false),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            StdArc::new(StringArray::from(ids)) as ArrayRef,
            StdArc::new(vectors) as ArrayRef,
            StdArc::new(StringArray::from(regions)) as ArrayRef,
        ],
    )
    .unwrap();
    let file = std::fs::File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

fn source_for(uri: String) -> ExternalSource {
    ExternalSource {
        format: "parquet".to_owned(),
        uri,
        id_field: "id".to_owned(),
        vector_field: "embedding".to_owned(),
        metric: DistanceMetric::Cosine,
    }
}

/// Run `load` + `stamp`, report timings, resident-bytes floor, and on-disk
/// size, for a dataset of `n` rows at `DIM` dimensions.
fn bench_size(n: usize) {
    let tmp = tempfile::tempdir().unwrap();
    let file_path = tmp.path().join("data.parquet");
    write_parquet(&file_path, n, DIM);
    let file_bytes = std::fs::metadata(&file_path).unwrap().len();

    let store = LocalBlobStore::new(tmp.path()).unwrap();
    let source = source_for(format!("file://{}", file_path.display()));

    let load_start = Instant::now();
    let dataset = external_parquet::load(&store, &source).unwrap();
    let load_elapsed = load_start.elapsed();

    let stamp_start = Instant::now();
    let _stamp = external_parquet::stamp(&store, &source).unwrap();
    let stamp_elapsed = stamp_start.elapsed();

    // The honest floor: row_count * dim * 4 bytes for the f32 vector
    // column alone. This is NOT measured RSS (no allocator/OS-level
    // accounting here) — it is what the vector buffer must occupy at
    // minimum, ignoring ids, scalar columns, and allocator overhead.
    let vector_floor_bytes = dataset.row_count() * DIM * 4;

    println!();
    println!("== {n} rows x {DIM} dim ==");
    println!("  load:               {load_elapsed:?}");
    println!("  stamp:              {stamp_elapsed:?}");
    println!(
        "  vector column floor: {vector_floor_bytes} bytes ({:.3} MB) [row_count * dim * 4, not measured RSS]",
        vector_floor_bytes as f64 / (1024.0 * 1024.0)
    );
    println!(
        "  parquet file size:   {file_bytes} bytes ({:.3} MB)",
        file_bytes as f64 / (1024.0 * 1024.0)
    );
    println!(
        "  compression ratio (floor / file_bytes): {:.3}x",
        vector_floor_bytes as f64 / file_bytes as f64
    );
}

/// Compare an unfiltered search against a filtered one over the same
/// 100,000-row dataset, reporting both timings.
///
/// `ExternalDataset::search` takes `allow: Option<&dyn Fn(&Record) -> bool>`
/// so it can share `evaluate_record` with the gateway without veda-storage
/// depending on veda-query. This benchmark exercises that closure boundary
/// directly with a plain predicate over `region`, without pulling in
/// veda-query.
fn bench_filtered_vs_unfiltered() {
    let n = 100_000;
    let tmp = tempfile::tempdir().unwrap();
    let file_path = tmp.path().join("data.parquet");
    write_parquet(&file_path, n, DIM);

    let store = LocalBlobStore::new(tmp.path()).unwrap();
    let source = source_for(format!("file://{}", file_path.display()));
    let dataset = external_parquet::load(&store, &source).unwrap();

    let query: Vec<f32> = (0..DIM).map(|d| (d % 1000) as f32 / 1000.0).collect();
    let k = 10;

    let unfiltered_start = Instant::now();
    let unfiltered = dataset
        .search(&query, k, DistanceMetric::Cosine, None)
        .unwrap();
    let unfiltered_elapsed = unfiltered_start.elapsed();

    // ~50% selective: every even row has region == "us" (see write_parquet).
    let allow =
        |rec: &Record| matches!(rec.get("region"), Some(FieldValue::String(s)) if s == "us");
    let allow_ref: &dyn Fn(&Record) -> bool = &allow;

    let filtered_start = Instant::now();
    let filtered = dataset
        .search(&query, k, DistanceMetric::Cosine, Some(allow_ref))
        .unwrap();
    let filtered_elapsed = filtered_start.elapsed();

    println!();
    println!("== filtered vs unfiltered search, {n} rows x {DIM} dim, k={k} ==");
    println!(
        "  unfiltered: {unfiltered_elapsed:?} ({} hits)",
        unfiltered.len()
    );
    println!(
        "  filtered:   {filtered_elapsed:?} ({} hits, ~50% selective on 'region')",
        filtered.len()
    );
    let ratio = filtered_elapsed.as_secs_f64() / unfiltered_elapsed.as_secs_f64().max(1e-12);
    println!("  filtered / unfiltered: {ratio:.3}x");
}

#[test]
#[ignore]
fn bench_10k_rows() {
    bench_size(10_000);
}

#[test]
#[ignore]
fn bench_100k_rows() {
    bench_size(100_000);
}

#[test]
#[ignore]
fn bench_filtered_search_100k_rows() {
    bench_filtered_vs_unfiltered();
}

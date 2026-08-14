//! Read-in-place Parquet datasets.
//!
//! A collection registered with an [`ExternalSource`] is queried directly
//! from the customer's files: nothing is ingested, nothing is copied into a
//! segment, and no index is built. Exact search only — see the design doc
//! at `documentation/2026-08-14-external-parquet-collections-design.md`.
//!
//! This module is the ONLY place in the workspace that knows about Arrow.

use std::collections::HashMap;

use bytes::Bytes;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

use veda_core::error::{Result, VedaError};
use veda_core::record::{FieldValue, Record};
use veda_core::schema::ExternalSource;
use veda_core::simd::{cosine_distance, dot_product, euclidean_distance};
use veda_core::types::DistanceMetric;

use crate::blob_store::BlobStore;

/// A Parquet dataset registered as a read-only collection.
#[derive(Debug)]
pub struct ExternalDataset {
    dim: u32,
    /// Flattened vector column, `dim` floats per row, row-aligned with `ids`.
    /// Read by `vector_at`/`search`.
    vectors: Vec<f32>,
    ids: Vec<String>,
    /// Name of the id column, so `record_at` can expose it alongside the
    /// other scalars — `read_batches` excludes it from `scalars` because it
    /// is already carried, row-aligned, in `ids`.
    id_field: String,
    /// Scalar columns, row-aligned, for predicates and for returning field
    /// values on a hit. Read by `record_at`.
    scalars: HashMap<String, Vec<FieldValue>>,
    /// Scalar columns dropped at load because their Arrow type has no
    /// `FieldValue` mapping: `(column name, Arrow type rendered with
    /// `{:?}`)`. A customer filtering on a dropped column would otherwise
    /// see silent zero hits with no diagnostic anywhere; Task 6 surfaces
    /// this list in the registration response.
    skipped: Vec<(String, String)>,
}

impl ExternalDataset {
    pub fn dim(&self) -> u32 {
        self.dim
    }

    pub fn row_count(&self) -> usize {
        self.ids.len()
    }

    pub fn ids(&self) -> &[String] {
        &self.ids
    }

    /// Scalar columns dropped at load because their Arrow type has no
    /// `FieldValue` mapping: `(column name, Arrow type)`.
    pub fn skipped_columns(&self) -> &[(String, String)] {
        &self.skipped
    }

    /// The vector for row `row`.
    ///
    /// `load` asserts `vectors.len() == ids.len() * dim`, so this slice is
    /// always in range for `row < row_count()`.
    pub fn vector_at(&self, row: usize) -> &[f32] {
        let d = self.dim as usize;
        &self.vectors[row * d..(row + 1) * d]
    }

    /// A `Record` view of row `row`.
    ///
    /// Exists so predicates can reuse the same evaluator the internal
    /// segment path uses, rather than a second filter implementation that
    /// could drift from it.
    pub fn record_at(&self, row: usize) -> Record {
        let mut rec = Record::new();
        rec.set(&self.id_field, FieldValue::String(self.ids[row].clone()));
        for (name, column) in &self.scalars {
            if let Some(v) = column.get(row) {
                rec.set(name, v.clone());
            }
        }
        rec
    }

    /// Exact top-k. Returns `(id, distance)` sorted ascending by distance,
    /// ties broken by id, truncated to `k`.
    ///
    /// Linear in `row_count()` by design: no index exists over external
    /// data, so recall is 1.0 and latency scales with dataset size.
    ///
    /// `allow` is an optional row predicate. It is a closure rather than a
    /// `FilterExpr` because `veda-query` depends on this crate, so naming
    /// its types here would be a dependency cycle. The gateway supplies
    /// `evaluate_record`, which is how external and internal filtering stay
    /// semantically identical.
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        metric: DistanceMetric,
        allow: Option<&dyn Fn(&Record) -> bool>,
    ) -> Result<Vec<(String, f32)>> {
        if query.len() != self.dim as usize {
            return Err(VedaError::Storage(format!(
                "query dimension {} does not match dataset dimension {}",
                query.len(),
                self.dim
            )));
        }
        if k == 0 {
            return Ok(Vec::new());
        }
        let mut scored: Vec<(String, f32)> = Vec::new();
        for row in 0..self.row_count() {
            if let Some(pred) = allow
                && !pred(&self.record_at(row))
            {
                continue;
            }
            let v = self.vector_at(row);
            let d = match metric {
                DistanceMetric::Cosine => cosine_distance(query, v),
                DistanceMetric::DotProduct => -dot_product(query, v),
                DistanceMetric::Euclidean => euclidean_distance(query, v),
                other => {
                    return Err(VedaError::Storage(format!(
                        "metric {other:?} is not supported for external collections: \
                         the vector column is float, so only Cosine, DotProduct and \
                         Euclidean apply"
                    )));
                }
            };
            scored.push((self.ids[row].clone(), d));
        }
        scored.sort_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        scored.truncate(k);
        Ok(scored)
    }
}

/// Strip a leading `<scheme>://<authority>/` from `uri`, leaving the object
/// key or key prefix passed to `BlobStore`.
///
/// `file://` URIs carry no authority — `file:///abs/path` is the scheme
/// followed directly by the absolute filesystem path `/abs/path` — so only
/// the literal `file://` prefix is stripped and the leading `/` is kept,
/// letting `LocalBlobStore` treat it as absolute and bypass its base
/// directory.
///
/// Any other scheme (`s3://`, ...) carries a bucket/authority next. The
/// underlying `BlobStore` already knows which bucket to talk to from server
/// configuration, so a bucket named in the URI is informational only and is
/// discarded here, leaving a bucket-relative key.
fn strip_scheme(uri: &str) -> String {
    if let Some(rest) = uri.strip_prefix("file://") {
        return rest.to_owned();
    }
    match uri.find("://") {
        Some(idx) => {
            let after_scheme = &uri[idx + 3..];
            match after_scheme.find('/') {
                Some(slash) => after_scheme[slash + 1..].to_owned(),
                None => String::new(),
            }
        }
        None => uri.to_owned(),
    }
}

/// Resolve `source.uri` to the object keys to read, in sorted order so a
/// dataset is assembled deterministically.
///
/// A resolved path ending in `.parquet` is a single object key; anything
/// else is a prefix and selects every `*.parquet` object beneath it. This
/// makes `s3://bucket/prefix` (no trailing slash) behave the same as
/// `s3://bucket/prefix/`, matching the documented URI form.
fn resolve_keys(store: &dyn BlobStore, uri: &str) -> Result<Vec<String>> {
    let cleaned = strip_scheme(uri);
    if cleaned.ends_with(".parquet") {
        Ok(vec![cleaned])
    } else {
        let mut keys: Vec<String> = store
            .list(&cleaned)?
            .into_iter()
            .filter(|k| k.ends_with(".parquet"))
            .collect();
        keys.sort();
        if keys.is_empty() {
            return Err(VedaError::Storage(format!(
                "no .parquet objects found under '{uri}'"
            )));
        }
        Ok(keys)
    }
}

/// Read and materialize the dataset described by `source`.
pub fn load(store: &dyn BlobStore, source: &ExternalSource) -> Result<ExternalDataset> {
    if source.format != "parquet" {
        return Err(VedaError::Storage(format!(
            "unsupported external format '{}': only 'parquet' is supported",
            source.format
        )));
    }
    let keys = resolve_keys(store, &source.uri)?;

    let mut dim: Option<u32> = None;
    let mut id_type: Option<(arrow::datatypes::DataType, String)> = None;
    let mut scalar_types: HashMap<String, (arrow::datatypes::DataType, String)> = HashMap::new();
    let mut vectors: Vec<f32> = Vec::new();
    let mut ids: Vec<String> = Vec::new();
    let mut scalars: HashMap<String, Vec<FieldValue>> = HashMap::new();
    let mut skipped: Vec<(String, String)> = Vec::new();

    for key in &keys {
        let raw = store.get(key)?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(Bytes::from(raw))
            .map_err(|e| VedaError::Storage(format!("parquet open '{key}': {e}")))?;
        let schema = builder.schema().clone();

        let available: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
        if !available.contains(&source.id_field.as_str()) {
            return Err(VedaError::Storage(format!(
                "id_field '{}' not found in '{key}'; available columns: {}",
                source.id_field,
                available.join(", ")
            )));
        }
        if !available.contains(&source.vector_field.as_str()) {
            return Err(VedaError::Storage(format!(
                "vector_field '{}' not found in '{key}'; available columns: {}",
                source.vector_field,
                available.join(", ")
            )));
        }

        // The id column's Arrow type must agree across every file behind a
        // prefix — otherwise e.g. an Int64 1 in one file and a Utf8 "1" in
        // another stringify to distinct ids for what may be the same row,
        // or distinct rows collide under a shared textual form.
        let this_id_type = id_arrow_type(&schema, &source.id_field, key)?;
        match &id_type {
            None => id_type = Some((this_id_type, key.clone())),
            Some((t, first_key)) if t != &this_id_type => {
                return Err(VedaError::Storage(format!(
                    "id_field '{}' type disagreement: '{first_key}' has {t:?}, '{key}' has {this_id_type:?}",
                    source.id_field
                )));
            }
            Some(_) => {}
        }

        // A scalar column's Arrow type must also agree across files: two
        // files disagreeing on a column's type is a schema disagreement
        // that should fail registration, not silently discard whichever
        // file's rows were already read for that column.
        for field in schema.fields() {
            let name = field.name().as_str();
            if name == source.id_field || name == source.vector_field {
                continue;
            }
            let ty = field.data_type().clone();
            match scalar_types.get(name) {
                None => {
                    scalar_types.insert(name.to_owned(), (ty, key.clone()));
                }
                Some((prev_ty, prev_key)) if prev_ty != &ty => {
                    return Err(VedaError::Storage(format!(
                        "scalar column '{name}' type disagreement: '{prev_key}' has {prev_ty:?}, '{key}' has {ty:?}"
                    )));
                }
                Some(_) => {}
            }
        }

        let file_dim = fixed_size_list_dim(&schema, &source.vector_field, key)?;
        match dim {
            None => dim = Some(file_dim),
            Some(d) if d != file_dim => {
                return Err(VedaError::Storage(format!(
                    "vector dimension disagreement: earlier files have {d}, '{key}' has {file_dim}"
                )));
            }
            Some(_) => {}
        }

        read_batches(
            builder,
            source,
            &mut ids,
            &mut vectors,
            &mut scalars,
            &mut skipped,
        )?;
    }

    // An empty result set is indistinguishable from a correct answer, so a
    // URI whose files exist but contain zero rows (a correctly-schema'd but
    // empty shard, an empty `_SUCCESS`-style file) must fail loudly rather
    // than register as a valid, permanently-empty collection.
    if ids.is_empty() {
        return Err(VedaError::Storage(format!(
            "no rows found behind '{}' ({} file(s) read)",
            source.uri,
            keys.len()
        )));
    }

    // Ambiguous ids make SearchHit.id meaningless, so duplicates — within
    // one file or across files behind a prefix — fail registration rather
    // than silently collapsing distinct rows under a shared id.
    {
        let mut seen: std::collections::HashSet<&str> = std::collections::HashSet::new();
        for id in &ids {
            if !seen.insert(id.as_str()) {
                return Err(VedaError::Storage(format!(
                    "duplicate id '{id}' behind '{}': id_field '{}' does not uniquely identify rows — register a different id_field",
                    source.uri, source.id_field
                )));
            }
        }
    }

    let dim = dim.ok_or_else(|| VedaError::Storage(format!("no data behind '{}'", source.uri)))?;

    // Defensive: `read_batches` maintains `vectors`/`scalars` row-alignment
    // by construction (verified by induction over the padding logic below),
    // but a future change to that logic breaching it should fail loudly at
    // load time rather than surface as a wrong vector or a panic when
    // `vector_at`/`record_at` index into these buffers by row.
    if vectors.len() != ids.len() * dim as usize {
        return Err(VedaError::Storage(format!(
            "internal error: vector buffer has {} floats, expected {} ({} rows * dim {dim})",
            vectors.len(),
            ids.len() * dim as usize,
            ids.len()
        )));
    }
    for (name, vals) in &scalars {
        if vals.len() != ids.len() {
            return Err(VedaError::Storage(format!(
                "internal error: scalar column '{name}' has {} entries, expected {} (row_count)",
                vals.len(),
                ids.len()
            )));
        }
    }

    Ok(ExternalDataset {
        dim,
        vectors,
        ids,
        id_field: source.id_field.clone(),
        scalars,
        skipped,
    })
}

/// Look up `field_name` in `schema` and confirm it is `FixedSizeList<Float32>`,
/// returning its declared length. Any other Arrow type is an error naming the
/// column, the file, and the actual type found.
fn fixed_size_list_dim(
    schema: &arrow::datatypes::Schema,
    field_name: &str,
    key: &str,
) -> Result<u32> {
    let field = schema.field_with_name(field_name).map_err(|e| {
        VedaError::Storage(format!(
            "vector_field '{field_name}' not found in '{key}': {e}"
        ))
    })?;
    match field.data_type() {
        arrow::datatypes::DataType::FixedSizeList(inner, len)
            if inner.data_type() == &arrow::datatypes::DataType::Float32 =>
        {
            Ok(*len as u32)
        }
        other => Err(VedaError::Storage(format!(
            "vector_field '{field_name}' in '{key}' must be FixedSizeList<Float32>, found {other:?}"
        ))),
    }
}

/// Look up `field_name` in `schema` and confirm its Arrow type is one of
/// the four supported id column types, returning it. Any other Arrow type
/// is an error naming the column, the file, and the actual type found.
fn id_arrow_type(
    schema: &arrow::datatypes::Schema,
    field_name: &str,
    key: &str,
) -> Result<arrow::datatypes::DataType> {
    let field = schema.field_with_name(field_name).map_err(|e| {
        VedaError::Storage(format!("id_field '{field_name}' not found in '{key}': {e}"))
    })?;
    match field.data_type() {
        arrow::datatypes::DataType::Utf8
        | arrow::datatypes::DataType::LargeUtf8
        | arrow::datatypes::DataType::Int32
        | arrow::datatypes::DataType::Int64 => Ok(field.data_type().clone()),
        other => Err(VedaError::Storage(format!(
            "id_field '{field_name}' in '{key}' has unsupported Arrow type {other:?}: must be Utf8, LargeUtf8, Int32 or Int64"
        ))),
    }
}

/// Append one file's batches to the accumulating dataset: ids, flattened
/// vectors, and every remaining scalar column of a supported type. Columns
/// of an unsupported Arrow type are skipped (recorded in `skipped` and
/// logged) rather than failing the load.
fn read_batches(
    builder: ParquetRecordBatchReaderBuilder<Bytes>,
    source: &ExternalSource,
    ids: &mut Vec<String>,
    vectors: &mut Vec<f32>,
    scalars: &mut HashMap<String, Vec<FieldValue>>,
    skipped: &mut Vec<(String, String)>,
) -> Result<()> {
    use arrow::array::{
        Array, BooleanArray, Date32Array, Date64Array, Float32Array, Float64Array, Int32Array,
        Int64Array, LargeStringArray, StringArray, TimestampMicrosecondArray,
        TimestampMillisecondArray, TimestampNanosecondArray, TimestampSecondArray,
    };
    use arrow::datatypes::{DataType, TimeUnit};

    let schema = builder.schema().clone();
    let reader = builder
        .build()
        .map_err(|e| VedaError::Storage(format!("parquet read: {e}")))?;

    for batch_result in reader {
        let batch = batch_result.map_err(|e| VedaError::Storage(format!("parquet read: {e}")))?;

        // ids: Utf8, LargeUtf8, Int32 and Int64 are accepted and stringified,
        // since external ids are commonly integers and SearchHit.id is a
        // String, so no synthetic identifier is needed. A null id would
        // otherwise stringify to "" (Utf8) or "0" (Int64) and silently
        // collide with other rows, so nulls are rejected outright.
        let id_col = batch
            .column_by_name(&source.id_field)
            .expect("id_field presence checked before read_batches is called");
        match id_col.data_type() {
            DataType::Utf8 => {
                let arr = id_col.as_any().downcast_ref::<StringArray>().unwrap();
                if arr.null_count() > 0 {
                    return Err(VedaError::Storage(format!(
                        "id_field '{}' has {} null row(s); ids must be non-null",
                        source.id_field,
                        arr.null_count()
                    )));
                }
                for i in 0..arr.len() {
                    ids.push(arr.value(i).to_owned());
                }
            }
            DataType::LargeUtf8 => {
                let arr = id_col.as_any().downcast_ref::<LargeStringArray>().unwrap();
                if arr.null_count() > 0 {
                    return Err(VedaError::Storage(format!(
                        "id_field '{}' has {} null row(s); ids must be non-null",
                        source.id_field,
                        arr.null_count()
                    )));
                }
                for i in 0..arr.len() {
                    ids.push(arr.value(i).to_owned());
                }
            }
            DataType::Int32 => {
                let arr = id_col.as_any().downcast_ref::<Int32Array>().unwrap();
                if arr.null_count() > 0 {
                    return Err(VedaError::Storage(format!(
                        "id_field '{}' has {} null row(s); ids must be non-null",
                        source.id_field,
                        arr.null_count()
                    )));
                }
                for i in 0..arr.len() {
                    ids.push(arr.value(i).to_string());
                }
            }
            DataType::Int64 => {
                let arr = id_col.as_any().downcast_ref::<Int64Array>().unwrap();
                if arr.null_count() > 0 {
                    return Err(VedaError::Storage(format!(
                        "id_field '{}' has {} null row(s); ids must be non-null",
                        source.id_field,
                        arr.null_count()
                    )));
                }
                for i in 0..arr.len() {
                    ids.push(arr.value(i).to_string());
                }
            }
            other => {
                return Err(VedaError::Storage(format!(
                    "id_field '{}' has unsupported Arrow type {other:?}: must be Utf8, LargeUtf8, Int32 or Int64",
                    source.id_field
                )));
            }
        }

        // vectors: flatten the FixedSizeList<Float32> column. A null row
        // would otherwise silently flatten to `dim` zeros and become a
        // first-class search candidate with a degenerate 0-norm vector.
        let vec_col = batch
            .column_by_name(&source.vector_field)
            .expect("vector_field presence checked before read_batches is called");
        let list = vec_col
            .as_any()
            .downcast_ref::<arrow::array::FixedSizeListArray>()
            .ok_or_else(|| {
                VedaError::Storage(format!(
                    "vector_field '{}' is not a FixedSizeList array",
                    source.vector_field
                ))
            })?;
        if list.null_count() > 0 {
            return Err(VedaError::Storage(format!(
                "vector_field '{}' has {} null row(s); vectors must be non-null",
                source.vector_field,
                list.null_count()
            )));
        }
        let values = list
            .values()
            .as_any()
            .downcast_ref::<Float32Array>()
            .ok_or_else(|| {
                VedaError::Storage(format!(
                    "vector_field '{}' values are not Float32",
                    source.vector_field
                ))
            })?;
        vectors.extend_from_slice(values.values());

        // scalars: every remaining column, mapped by supported Arrow type.
        // Nulls become FieldValue::Null; a predicate against a null scalar
        // is expected to simply not match, same as an absent field would.
        //
        // `scalars` accumulates across every file behind the URI, so a
        // column already collected before this batch must be padded to
        // `rows_before_batch` when it is first seen, and any column already
        // present that this batch's file does not have at all must be
        // padded forward by `batch.num_rows()` — otherwise a column present
        // in only some files would end up shorter than `row_count()` and
        // silently misaligned with `ids`. Cross-file *type* disagreement
        // for the same column name is rejected in `load` before this
        // function is ever called for the disagreeing file, so a column
        // keeps one Arrow type for its whole life in `scalars`.
        let rows_before_batch = ids.len() - batch.num_rows();
        let mut touched: std::collections::HashSet<&str> = std::collections::HashSet::new();
        for field in schema.fields() {
            let name = field.name().as_str();
            if name == source.id_field || name == source.vector_field {
                continue;
            }
            touched.insert(name);
            let col = batch
                .column_by_name(name)
                .expect("field from schema must be in batch");
            let out = scalars
                .entry(name.to_owned())
                .or_insert_with(|| vec![FieldValue::Null; rows_before_batch]);
            match col.data_type() {
                DataType::Utf8 => {
                    let arr = col.as_any().downcast_ref::<StringArray>().unwrap();
                    for i in 0..arr.len() {
                        out.push(if arr.is_null(i) {
                            FieldValue::Null
                        } else {
                            FieldValue::String(arr.value(i).to_owned())
                        });
                    }
                }
                DataType::LargeUtf8 => {
                    let arr = col.as_any().downcast_ref::<LargeStringArray>().unwrap();
                    for i in 0..arr.len() {
                        out.push(if arr.is_null(i) {
                            FieldValue::Null
                        } else {
                            FieldValue::String(arr.value(i).to_owned())
                        });
                    }
                }
                DataType::Int32 => {
                    let arr = col.as_any().downcast_ref::<Int32Array>().unwrap();
                    for i in 0..arr.len() {
                        out.push(if arr.is_null(i) {
                            FieldValue::Null
                        } else {
                            FieldValue::Int32(arr.value(i))
                        });
                    }
                }
                DataType::Int64 => {
                    let arr = col.as_any().downcast_ref::<Int64Array>().unwrap();
                    for i in 0..arr.len() {
                        out.push(if arr.is_null(i) {
                            FieldValue::Null
                        } else {
                            FieldValue::Int64(arr.value(i))
                        });
                    }
                }
                DataType::Float32 => {
                    let arr = col.as_any().downcast_ref::<Float32Array>().unwrap();
                    for i in 0..arr.len() {
                        out.push(if arr.is_null(i) {
                            FieldValue::Null
                        } else {
                            FieldValue::Float32(arr.value(i))
                        });
                    }
                }
                DataType::Float64 => {
                    let arr = col.as_any().downcast_ref::<Float64Array>().unwrap();
                    for i in 0..arr.len() {
                        out.push(if arr.is_null(i) {
                            FieldValue::Null
                        } else {
                            FieldValue::Float64(arr.value(i))
                        });
                    }
                }
                DataType::Boolean => {
                    let arr = col.as_any().downcast_ref::<BooleanArray>().unwrap();
                    for i in 0..arr.len() {
                        out.push(if arr.is_null(i) {
                            FieldValue::Null
                        } else {
                            FieldValue::Bool(arr.value(i))
                        });
                    }
                }
                DataType::Date32 => {
                    let arr = col.as_any().downcast_ref::<Date32Array>().unwrap();
                    for i in 0..arr.len() {
                        out.push(if arr.is_null(i) {
                            FieldValue::Null
                        } else {
                            // Date32 is days since the Unix epoch.
                            FieldValue::Timestamp(arr.value(i) as i64 * 86_400_000)
                        });
                    }
                }
                DataType::Date64 => {
                    let arr = col.as_any().downcast_ref::<Date64Array>().unwrap();
                    for i in 0..arr.len() {
                        out.push(if arr.is_null(i) {
                            FieldValue::Null
                        } else {
                            // Date64 is already milliseconds since the Unix epoch.
                            FieldValue::Timestamp(arr.value(i))
                        });
                    }
                }
                DataType::Timestamp(TimeUnit::Second, _) => {
                    let arr = col.as_any().downcast_ref::<TimestampSecondArray>().unwrap();
                    for i in 0..arr.len() {
                        out.push(if arr.is_null(i) {
                            FieldValue::Null
                        } else {
                            FieldValue::Timestamp(arr.value(i) * 1_000)
                        });
                    }
                }
                DataType::Timestamp(TimeUnit::Millisecond, _) => {
                    let arr = col
                        .as_any()
                        .downcast_ref::<TimestampMillisecondArray>()
                        .unwrap();
                    for i in 0..arr.len() {
                        out.push(if arr.is_null(i) {
                            FieldValue::Null
                        } else {
                            FieldValue::Timestamp(arr.value(i))
                        });
                    }
                }
                DataType::Timestamp(TimeUnit::Microsecond, _) => {
                    let arr = col
                        .as_any()
                        .downcast_ref::<TimestampMicrosecondArray>()
                        .unwrap();
                    for i in 0..arr.len() {
                        out.push(if arr.is_null(i) {
                            FieldValue::Null
                        } else {
                            FieldValue::Timestamp(arr.value(i) / 1_000)
                        });
                    }
                }
                DataType::Timestamp(TimeUnit::Nanosecond, _) => {
                    let arr = col
                        .as_any()
                        .downcast_ref::<TimestampNanosecondArray>()
                        .unwrap();
                    for i in 0..arr.len() {
                        out.push(if arr.is_null(i) {
                            FieldValue::Null
                        } else {
                            FieldValue::Timestamp(arr.value(i) / 1_000_000)
                        });
                    }
                }
                other => {
                    // Unsupported Arrow type: drop this column entirely
                    // rather than fail the whole load, so one unusual
                    // column elsewhere in a customer's file can't make an
                    // otherwise-valid dataset unusable. Surfaced via
                    // `skipped` (Task 6 returns it in the registration
                    // response) and a warning, so the drop isn't silent.
                    scalars.remove(name);
                    if !skipped.iter().any(|(n, _)| n == name) {
                        let type_desc = format!("{other:?}");
                        tracing::warn!(
                            column = name,
                            arrow_type = %type_desc,
                            "external parquet: dropping scalar column of unsupported Arrow type"
                        );
                        skipped.push((name.to_owned(), type_desc));
                    }
                }
            }
        }

        // A column collected from an earlier file that this batch's schema
        // doesn't mention at all: pad it with Null for this batch's rows so
        // it stays aligned with `ids`.
        let batch_len = batch.num_rows();
        for (name, out) in scalars.iter_mut() {
            if !touched.contains(name.as_str()) {
                out.resize(rows_before_batch + batch_len, FieldValue::Null);
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc as StdArc;

    use arrow::array::{
        ArrayRef, Date32Array, FixedSizeListArray, Float32Array, Int8Array, Int32Array, Int64Array,
        StringArray, TimestampMillisecondArray,
    };
    use arrow::buffer::NullBuffer;
    use arrow::datatypes::{DataType, Field, Schema, TimeUnit};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::ArrowWriter;

    use crate::blob_store::LocalBlobStore;
    use veda_core::types::DistanceMetric;

    /// Build a Parquet file with the given `ids` (row count comes from
    /// `ids.len()`) and a fixed-size-list vector column of dimension `dim`.
    fn write_parquet_with_ids(path: &std::path::Path, ids: &[String], dim: usize) {
        let n = ids.len();
        let mut flat: Vec<f32> = Vec::with_capacity(n * dim);
        for i in 0..n {
            for d in 0..dim {
                flat.push((i * dim + d) as f32 / 10.0);
            }
        }
        let values = Float32Array::from(flat);
        let vec_field = StdArc::new(Field::new("item", DataType::Float32, false));
        let vectors =
            FixedSizeListArray::try_new(vec_field.clone(), dim as i32, StdArc::new(values), None)
                .unwrap();
        let schema = StdArc::new(Schema::new(vec![
            Field::new("doc_id", DataType::Utf8, false),
            Field::new(
                "embedding",
                DataType::FixedSizeList(vec_field, dim as i32),
                false,
            ),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                StdArc::new(StringArray::from(ids.to_vec())) as ArrayRef,
                StdArc::new(vectors) as ArrayRef,
            ],
        )
        .unwrap();
        let file = std::fs::File::create(path).unwrap();
        let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
    }

    /// Build a Parquet file with `n` rows: a string id column `doc0..docN-1`
    /// and a fixed-size-list vector column of dimension `dim`.
    fn write_parquet(path: &std::path::Path, n: usize, dim: usize) {
        let ids: Vec<String> = (0..n).map(|i| format!("doc{i}")).collect();
        write_parquet_with_ids(path, &ids, dim);
    }

    /// Like `write_parquet` but with an `Int64` id column instead of `Utf8`,
    /// for cross-file id-type-disagreement tests.
    fn write_parquet_int_ids(path: &std::path::Path, n: usize, dim: usize) {
        let ids: Vec<i64> = (0..n as i64).collect();
        let mut flat: Vec<f32> = Vec::with_capacity(n * dim);
        for i in 0..n {
            for d in 0..dim {
                flat.push((i * dim + d) as f32 / 10.0);
            }
        }
        let values = Float32Array::from(flat);
        let vec_field = StdArc::new(Field::new("item", DataType::Float32, false));
        let vectors =
            FixedSizeListArray::try_new(vec_field.clone(), dim as i32, StdArc::new(values), None)
                .unwrap();
        let schema = StdArc::new(Schema::new(vec![
            Field::new("doc_id", DataType::Int64, false),
            Field::new(
                "embedding",
                DataType::FixedSizeList(vec_field, dim as i32),
                false,
            ),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                StdArc::new(Int64Array::from(ids)) as ArrayRef,
                StdArc::new(vectors) as ArrayRef,
            ],
        )
        .unwrap();
        let file = std::fs::File::create(path).unwrap();
        let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
    }

    /// Three rows, dimension `dim`, with the middle row's id null.
    fn write_parquet_with_null_id(path: &std::path::Path, dim: usize) {
        let n = 3;
        let mut flat: Vec<f32> = Vec::with_capacity(n * dim);
        for i in 0..n {
            for d in 0..dim {
                flat.push((i * dim + d) as f32 / 10.0);
            }
        }
        let values = Float32Array::from(flat);
        let vec_field = StdArc::new(Field::new("item", DataType::Float32, false));
        let vectors =
            FixedSizeListArray::try_new(vec_field.clone(), dim as i32, StdArc::new(values), None)
                .unwrap();
        let ids = StringArray::from(vec![Some("doc0"), None, Some("doc2")]);
        let schema = StdArc::new(Schema::new(vec![
            Field::new("doc_id", DataType::Utf8, true),
            Field::new(
                "embedding",
                DataType::FixedSizeList(vec_field, dim as i32),
                false,
            ),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                StdArc::new(ids) as ArrayRef,
                StdArc::new(vectors) as ArrayRef,
            ],
        )
        .unwrap();
        let file = std::fs::File::create(path).unwrap();
        let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
    }

    /// Two rows, dimension `dim`, with the second row's vector null.
    fn write_parquet_with_null_vector(path: &std::path::Path, dim: usize) {
        let n = 2;
        let mut flat: Vec<f32> = Vec::with_capacity(n * dim);
        for i in 0..n {
            for d in 0..dim {
                flat.push((i * dim + d) as f32 / 10.0);
            }
        }
        let values = Float32Array::from(flat);
        let vec_field = StdArc::new(Field::new("item", DataType::Float32, false));
        let nulls = NullBuffer::from(vec![true, false]);
        let vectors = FixedSizeListArray::try_new(
            vec_field.clone(),
            dim as i32,
            StdArc::new(values),
            Some(nulls),
        )
        .unwrap();
        let ids: Vec<String> = (0..n).map(|i| format!("doc{i}")).collect();
        let schema = StdArc::new(Schema::new(vec![
            Field::new("doc_id", DataType::Utf8, false),
            Field::new(
                "embedding",
                DataType::FixedSizeList(vec_field, dim as i32),
                true,
            ),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                StdArc::new(StringArray::from(ids)) as ArrayRef,
                StdArc::new(vectors) as ArrayRef,
            ],
        )
        .unwrap();
        let file = std::fs::File::create(path).unwrap();
        let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
    }

    /// `ids`-row dataset with an extra `region` column of the given Arrow
    /// type, for scalar-column tests.
    enum RegionColumn<'a> {
        Utf8(&'a [&'a str]),
        Int32(&'a [i32]),
    }

    fn write_parquet_with_region(
        path: &std::path::Path,
        ids: &[String],
        dim: usize,
        region: RegionColumn,
    ) {
        let n = ids.len();
        let mut flat: Vec<f32> = Vec::with_capacity(n * dim);
        for i in 0..n {
            for d in 0..dim {
                flat.push((i * dim + d) as f32 / 10.0);
            }
        }
        let values = Float32Array::from(flat);
        let vec_field = StdArc::new(Field::new("item", DataType::Float32, false));
        let vectors =
            FixedSizeListArray::try_new(vec_field.clone(), dim as i32, StdArc::new(values), None)
                .unwrap();
        let (region_type, region_array): (DataType, ArrayRef) = match region {
            RegionColumn::Utf8(vals) => (
                DataType::Utf8,
                StdArc::new(StringArray::from(vals.to_vec())) as ArrayRef,
            ),
            RegionColumn::Int32(vals) => (
                DataType::Int32,
                StdArc::new(Int32Array::from(vals.to_vec())) as ArrayRef,
            ),
        };
        let schema = StdArc::new(Schema::new(vec![
            Field::new("doc_id", DataType::Utf8, false),
            Field::new(
                "embedding",
                DataType::FixedSizeList(vec_field, dim as i32),
                false,
            ),
            Field::new("region", region_type, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                StdArc::new(StringArray::from(ids.to_vec())) as ArrayRef,
                StdArc::new(vectors) as ArrayRef,
                region_array,
            ],
        )
        .unwrap();
        let file = std::fs::File::create(path).unwrap();
        let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
    }

    /// Two rows, dimension `dim`, plus a `created_at`
    /// `Timestamp(Millisecond)` column with the given values.
    fn write_parquet_with_timestamp_millis(path: &std::path::Path, dim: usize, millis: &[i64]) {
        let n = millis.len();
        let mut flat: Vec<f32> = Vec::with_capacity(n * dim);
        for i in 0..n {
            for d in 0..dim {
                flat.push((i * dim + d) as f32 / 10.0);
            }
        }
        let values = Float32Array::from(flat);
        let vec_field = StdArc::new(Field::new("item", DataType::Float32, false));
        let vectors =
            FixedSizeListArray::try_new(vec_field.clone(), dim as i32, StdArc::new(values), None)
                .unwrap();
        let ids: Vec<String> = (0..n).map(|i| format!("doc{i}")).collect();
        let schema = StdArc::new(Schema::new(vec![
            Field::new("doc_id", DataType::Utf8, false),
            Field::new(
                "embedding",
                DataType::FixedSizeList(vec_field, dim as i32),
                false,
            ),
            Field::new(
                "created_at",
                DataType::Timestamp(TimeUnit::Millisecond, None),
                false,
            ),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                StdArc::new(StringArray::from(ids)) as ArrayRef,
                StdArc::new(vectors) as ArrayRef,
                StdArc::new(TimestampMillisecondArray::from(millis.to_vec())) as ArrayRef,
            ],
        )
        .unwrap();
        let file = std::fs::File::create(path).unwrap();
        let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
    }

    /// Rows, dimension `dim`, plus a `created_on` `Date32` column (days
    /// since the Unix epoch) with the given values.
    fn write_parquet_with_date32(path: &std::path::Path, dim: usize, days: &[i32]) {
        let n = days.len();
        let mut flat: Vec<f32> = Vec::with_capacity(n * dim);
        for i in 0..n {
            for d in 0..dim {
                flat.push((i * dim + d) as f32 / 10.0);
            }
        }
        let values = Float32Array::from(flat);
        let vec_field = StdArc::new(Field::new("item", DataType::Float32, false));
        let vectors =
            FixedSizeListArray::try_new(vec_field.clone(), dim as i32, StdArc::new(values), None)
                .unwrap();
        let ids: Vec<String> = (0..n).map(|i| format!("doc{i}")).collect();
        let schema = StdArc::new(Schema::new(vec![
            Field::new("doc_id", DataType::Utf8, false),
            Field::new(
                "embedding",
                DataType::FixedSizeList(vec_field, dim as i32),
                false,
            ),
            Field::new("created_on", DataType::Date32, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                StdArc::new(StringArray::from(ids)) as ArrayRef,
                StdArc::new(vectors) as ArrayRef,
                StdArc::new(Date32Array::from(days.to_vec())) as ArrayRef,
            ],
        )
        .unwrap();
        let file = std::fs::File::create(path).unwrap();
        let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
    }

    /// Rows, dimension `dim`, plus a `weird` `Int8` column — an Arrow type
    /// this module does not map to a `FieldValue`.
    fn write_parquet_with_unsupported_column(path: &std::path::Path, dim: usize, vals: &[i8]) {
        let n = vals.len();
        let mut flat: Vec<f32> = Vec::with_capacity(n * dim);
        for i in 0..n {
            for d in 0..dim {
                flat.push((i * dim + d) as f32 / 10.0);
            }
        }
        let values = Float32Array::from(flat);
        let vec_field = StdArc::new(Field::new("item", DataType::Float32, false));
        let vectors =
            FixedSizeListArray::try_new(vec_field.clone(), dim as i32, StdArc::new(values), None)
                .unwrap();
        let ids: Vec<String> = (0..n).map(|i| format!("doc{i}")).collect();
        let schema = StdArc::new(Schema::new(vec![
            Field::new("doc_id", DataType::Utf8, false),
            Field::new(
                "embedding",
                DataType::FixedSizeList(vec_field, dim as i32),
                false,
            ),
            Field::new("weird", DataType::Int8, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                StdArc::new(StringArray::from(ids)) as ArrayRef,
                StdArc::new(vectors) as ArrayRef,
                StdArc::new(Int8Array::from(vals.to_vec())) as ArrayRef,
            ],
        )
        .unwrap();
        let file = std::fs::File::create(path).unwrap();
        let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
    }

    fn source(uri: &str) -> ExternalSource {
        ExternalSource {
            format: "parquet".into(),
            uri: uri.into(),
            id_field: "doc_id".into(),
            vector_field: "embedding".into(),
            metric: DistanceMetric::Cosine,
        }
    }

    #[test]
    fn loads_ids_and_vectors() {
        let dir = tempfile::tempdir().unwrap();
        write_parquet(&dir.path().join("a.parquet"), 10, 4);
        let store = LocalBlobStore::new(dir.path()).unwrap();
        let ds = load(&store, &source("a.parquet")).unwrap();
        assert_eq!(ds.row_count(), 10);
        assert_eq!(ds.dim(), 4);
        assert_eq!(ds.ids()[0], "doc0");
        assert_eq!(ds.ids()[9], "doc9");
    }

    #[test]
    fn rejects_a_missing_vector_column() {
        let dir = tempfile::tempdir().unwrap();
        write_parquet(&dir.path().join("a.parquet"), 4, 4);
        let store = LocalBlobStore::new(dir.path()).unwrap();
        let mut src = source("a.parquet");
        src.vector_field = "not_there".into();
        let err = load(&store, &src).unwrap_err().to_string();
        assert!(
            err.contains("not_there") && err.contains("embedding"),
            "error must name the missing column AND what is available, got: {err}"
        );
    }

    #[test]
    fn rejects_a_vector_column_of_the_wrong_arrow_type() {
        let dir = tempfile::tempdir().unwrap();
        write_parquet(&dir.path().join("a.parquet"), 4, 4);
        let store = LocalBlobStore::new(dir.path()).unwrap();
        let mut src = source("a.parquet");
        // The id column is Utf8, not FixedSizeList<Float32>.
        src.vector_field = "doc_id".into();
        let err = load(&store, &src).unwrap_err().to_string();
        assert!(
            err.contains("doc_id"),
            "error must name the offending column, got: {err}"
        );
    }

    #[test]
    fn rejects_a_missing_id_column() {
        let dir = tempfile::tempdir().unwrap();
        write_parquet(&dir.path().join("a.parquet"), 4, 4);
        let store = LocalBlobStore::new(dir.path()).unwrap();
        let mut src = source("a.parquet");
        src.id_field = "nope".into();
        assert!(load(&store, &src).is_err());
    }

    #[test]
    fn rejects_a_non_parquet_format() {
        let dir = tempfile::tempdir().unwrap();
        write_parquet(&dir.path().join("a.parquet"), 4, 4);
        let store = LocalBlobStore::new(dir.path()).unwrap();
        let mut src = source("a.parquet");
        src.format = "lance".into();
        let err = load(&store, &src).unwrap_err().to_string();
        assert!(err.contains("lance"), "got: {err}");
    }

    #[test]
    fn reads_every_parquet_file_behind_a_prefix() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("part")).unwrap();
        // Disjoint id ranges: two files behind a prefix sharing ids would
        // now be rejected as duplicates (see `rejects_duplicate_ids_across_files`).
        let a_ids: Vec<String> = (0..5).map(|i| format!("a{i}")).collect();
        let b_ids: Vec<String> = (0..7).map(|i| format!("b{i}")).collect();
        write_parquet_with_ids(&dir.path().join("part/a.parquet"), &a_ids, 4);
        write_parquet_with_ids(&dir.path().join("part/b.parquet"), &b_ids, 4);
        let store = LocalBlobStore::new(dir.path()).unwrap();
        let ds = load(&store, &source("part/")).unwrap();
        assert_eq!(ds.row_count(), 12, "both files must be read as one dataset");
    }

    #[test]
    fn rejects_files_behind_a_prefix_that_disagree_on_dimension() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("part")).unwrap();
        write_parquet(&dir.path().join("part/a.parquet"), 5, 4);
        write_parquet(&dir.path().join("part/b.parquet"), 5, 8);
        let store = LocalBlobStore::new(dir.path()).unwrap();
        let err = load(&store, &source("part/")).unwrap_err().to_string();
        assert!(
            err.contains('4') && err.contains('8'),
            "error must name both dimensions, got: {err}"
        );
    }

    #[test]
    fn an_empty_prefix_is_an_error_not_an_empty_dataset() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("part")).unwrap();
        let store = LocalBlobStore::new(dir.path()).unwrap();
        // An empty result set is indistinguishable from a correct answer,
        // so a URI with no Parquet behind it must fail loudly.
        assert!(load(&store, &source("part/")).is_err());
    }

    #[test]
    fn a_zero_row_parquet_is_an_error_not_an_empty_dataset() {
        let dir = tempfile::tempdir().unwrap();
        write_parquet(&dir.path().join("a.parquet"), 0, 4);
        let store = LocalBlobStore::new(dir.path()).unwrap();
        let err = load(&store, &source("a.parquet")).unwrap_err().to_string();
        assert!(
            err.contains("a.parquet"),
            "error must name the URI with no rows behind it, got: {err}"
        );
    }

    #[test]
    fn rejects_a_null_id() {
        let dir = tempfile::tempdir().unwrap();
        write_parquet_with_null_id(&dir.path().join("a.parquet"), 4);
        let store = LocalBlobStore::new(dir.path()).unwrap();
        let err = load(&store, &source("a.parquet")).unwrap_err().to_string();
        assert!(
            err.contains("doc_id") && err.contains('1'),
            "error must name the id column and the null count, got: {err}"
        );
    }

    #[test]
    fn rejects_a_null_vector_row() {
        let dir = tempfile::tempdir().unwrap();
        write_parquet_with_null_vector(&dir.path().join("a.parquet"), 4);
        let store = LocalBlobStore::new(dir.path()).unwrap();
        let err = load(&store, &source("a.parquet")).unwrap_err().to_string();
        assert!(
            err.contains("embedding") && err.contains('1'),
            "error must name the vector column and the null count, got: {err}"
        );
    }

    #[test]
    fn rejects_duplicate_ids_across_files() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("part")).unwrap();
        write_parquet(&dir.path().join("part/a.parquet"), 5, 4); // doc0..doc4
        write_parquet(&dir.path().join("part/b.parquet"), 3, 4); // doc0..doc2, overlaps
        let store = LocalBlobStore::new(dir.path()).unwrap();
        let err = load(&store, &source("part/")).unwrap_err().to_string();
        assert!(
            err.contains("doc0"),
            "error must name the offending id, got: {err}"
        );
    }

    #[test]
    fn rejects_id_type_disagreement_across_files() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("part")).unwrap();
        write_parquet(&dir.path().join("part/a.parquet"), 3, 4); // Utf8 ids
        write_parquet_int_ids(&dir.path().join("part/b.parquet"), 3, 4); // Int64 ids
        let store = LocalBlobStore::new(dir.path()).unwrap();
        let err = load(&store, &source("part/")).unwrap_err().to_string();
        assert!(
            err.contains("doc_id") && err.contains("Utf8") && err.contains("Int64"),
            "error must name the column and both types, got: {err}"
        );
    }

    #[test]
    fn rejects_scalar_column_type_disagreement_across_files() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("part")).unwrap();
        let a_ids: Vec<String> = (0..3).map(|i| format!("a{i}")).collect();
        let b_ids: Vec<String> = (0..3).map(|i| format!("b{i}")).collect();
        write_parquet_with_region(
            &dir.path().join("part/a.parquet"),
            &a_ids,
            4,
            RegionColumn::Utf8(&["eu", "us", "eu"]),
        );
        write_parquet_with_region(
            &dir.path().join("part/b.parquet"),
            &b_ids,
            4,
            RegionColumn::Int32(&[1, 2, 1]),
        );
        let store = LocalBlobStore::new(dir.path()).unwrap();
        let err = load(&store, &source("part/")).unwrap_err().to_string();
        assert!(
            err.contains("region") && err.contains("Utf8") && err.contains("Int32"),
            "error must name the column and both types, got: {err}"
        );
    }

    #[test]
    fn maps_timestamp_millisecond_column_to_field_value_timestamp() {
        let dir = tempfile::tempdir().unwrap();
        write_parquet_with_timestamp_millis(
            &dir.path().join("a.parquet"),
            4,
            &[1_700_000_000_000, 1_700_000_100_000],
        );
        let store = LocalBlobStore::new(dir.path()).unwrap();
        let ds = load(&store, &source("a.parquet")).unwrap();
        let col = ds
            .scalars
            .get("created_at")
            .expect("created_at should be mapped, not skipped");
        assert_eq!(col[0], FieldValue::Timestamp(1_700_000_000_000));
        assert_eq!(col[1], FieldValue::Timestamp(1_700_000_100_000));
        assert!(
            ds.skipped_columns().is_empty(),
            "a temporal column must not be reported as skipped"
        );
    }

    #[test]
    fn maps_date32_column_to_field_value_timestamp() {
        let dir = tempfile::tempdir().unwrap();
        // 19000 days since epoch.
        write_parquet_with_date32(&dir.path().join("a.parquet"), 4, &[19000, 19001]);
        let store = LocalBlobStore::new(dir.path()).unwrap();
        let ds = load(&store, &source("a.parquet")).unwrap();
        let col = ds
            .scalars
            .get("created_on")
            .expect("created_on should be mapped, not skipped");
        assert_eq!(col[0], FieldValue::Timestamp(19000i64 * 86_400_000));
        assert_eq!(col[1], FieldValue::Timestamp(19001i64 * 86_400_000));
    }

    #[test]
    fn skips_and_records_an_unsupported_scalar_column() {
        let dir = tempfile::tempdir().unwrap();
        write_parquet_with_unsupported_column(&dir.path().join("a.parquet"), 4, &[1, 2, 3]);
        let store = LocalBlobStore::new(dir.path()).unwrap();
        let ds = load(&store, &source("a.parquet")).unwrap();
        assert!(
            !ds.scalars.contains_key("weird"),
            "unsupported column must not appear in scalars"
        );
        let skipped = ds.skipped_columns();
        assert_eq!(skipped.len(), 1);
        assert_eq!(skipped[0].0, "weird");
        assert!(
            skipped[0].1.contains("Int8"),
            "skipped entry must name the Arrow type, got: {:?}",
            skipped[0]
        );
    }

    #[test]
    fn resolves_an_s3_prefix_without_a_trailing_slash() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("prefix")).unwrap();
        let a_ids: Vec<String> = (0..3).map(|i| format!("a{i}")).collect();
        let b_ids: Vec<String> = (0..2).map(|i| format!("b{i}")).collect();
        write_parquet_with_ids(&dir.path().join("prefix/a.parquet"), &a_ids, 4);
        write_parquet_with_ids(&dir.path().join("prefix/b.parquet"), &b_ids, 4);
        let store = LocalBlobStore::new(dir.path()).unwrap();
        // The bucket name is informational only; LocalBlobStore has no
        // concept of a bucket, so the important thing is that "bucket" is
        // discarded and "prefix" is used as the key prefix.
        let ds = load(&store, &source("s3://bucket/prefix")).unwrap();
        assert_eq!(ds.row_count(), 5);
    }

    #[test]
    fn resolves_an_s3_prefix_with_a_trailing_slash() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("prefix")).unwrap();
        let a_ids: Vec<String> = (0..3).map(|i| format!("a{i}")).collect();
        let b_ids: Vec<String> = (0..2).map(|i| format!("b{i}")).collect();
        write_parquet_with_ids(&dir.path().join("prefix/a.parquet"), &a_ids, 4);
        write_parquet_with_ids(&dir.path().join("prefix/b.parquet"), &b_ids, 4);
        let store = LocalBlobStore::new(dir.path()).unwrap();
        let ds = load(&store, &source("s3://bucket/prefix/")).unwrap();
        assert_eq!(ds.row_count(), 5);
    }

    #[test]
    fn resolves_a_file_scheme_absolute_path() {
        // The file being read lives outside the store's own base directory
        // entirely, to prove the `file://` form bypasses it.
        let target_dir = tempfile::tempdir().unwrap();
        let abs_path = target_dir.path().join("a.parquet");
        write_parquet(&abs_path, 4, 4);

        let store_dir = tempfile::tempdir().unwrap();
        let store = LocalBlobStore::new(store_dir.path()).unwrap();

        let uri = format!("file://{}", abs_path.display());
        let ds = load(&store, &source(&uri)).unwrap();
        assert_eq!(ds.row_count(), 4);
    }

    #[test]
    fn resolves_a_bare_relative_key() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("sub")).unwrap();
        write_parquet(&dir.path().join("sub/a.parquet"), 6, 4);
        let store = LocalBlobStore::new(dir.path()).unwrap();
        let ds = load(&store, &source("sub/a.parquet")).unwrap();
        assert_eq!(ds.row_count(), 6);
    }

    #[test]
    fn search_is_exact_and_matches_direct_scoring() {
        let dir = tempfile::tempdir().unwrap();
        write_parquet(&dir.path().join("a.parquet"), 200, 8);
        let store = LocalBlobStore::new(dir.path()).unwrap();
        let ds = load(&store, &source("a.parquet")).unwrap();

        let query: Vec<f32> = (0..8).map(|d| d as f32 / 3.0).collect();
        let got = ds.search(&query, 10, DistanceMetric::Cosine, None).unwrap();

        // Oracle: score every row directly with the same kernel the
        // implementation uses, sort, truncate. If these disagree, the external
        // path is not exact, which is the single property this feature sells.
        let mut want: Vec<(String, f32)> = (0..ds.row_count())
            .map(|r| {
                (
                    ds.ids()[r].clone(),
                    veda_core::simd::cosine_distance(&query, ds.vector_at(r)),
                )
            })
            .collect();
        want.sort_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        want.truncate(10);
        assert_eq!(got, want);
    }

    #[test]
    fn search_honours_a_predicate() {
        let dir = tempfile::tempdir().unwrap();
        write_parquet(&dir.path().join("a.parquet"), 50, 4);
        let store = LocalBlobStore::new(dir.path()).unwrap();
        let ds = load(&store, &source("a.parquet")).unwrap();

        let query = vec![1.0f32, 0.0, 0.0, 0.0];
        let unfiltered = ds.search(&query, 5, DistanceMetric::Cosine, None).unwrap();
        let excluded = unfiltered[0].0.clone();

        // Exclude whatever the unfiltered search ranked FIRST, so the predicate
        // cannot be silently ignored — a filter that does nothing would still
        // return this id.
        let deny = |rec: &Record| rec.get("doc_id") != Some(&FieldValue::String(excluded.clone()));
        let filtered = ds
            .search(&query, 5, DistanceMetric::Cosine, Some(&deny))
            .unwrap();
        assert!(
            !filtered.iter().any(|(id, _)| *id == excluded),
            "predicate was ignored: {excluded} is still present"
        );
        assert_eq!(filtered.len(), 5);
    }

    #[test]
    fn search_with_k_larger_than_the_dataset_returns_everything() {
        let dir = tempfile::tempdir().unwrap();
        write_parquet(&dir.path().join("a.parquet"), 7, 4);
        let store = LocalBlobStore::new(dir.path()).unwrap();
        let ds = load(&store, &source("a.parquet")).unwrap();
        let hits = ds
            .search(&[1.0, 0.0, 0.0, 0.0], 100, DistanceMetric::Cosine, None)
            .unwrap();
        assert_eq!(hits.len(), 7);
    }

    #[test]
    fn search_with_k_zero_returns_nothing() {
        let dir = tempfile::tempdir().unwrap();
        write_parquet(&dir.path().join("a.parquet"), 7, 4);
        let store = LocalBlobStore::new(dir.path()).unwrap();
        let ds = load(&store, &source("a.parquet")).unwrap();
        let hits = ds
            .search(&[1.0, 0.0, 0.0, 0.0], 0, DistanceMetric::Cosine, None)
            .unwrap();
        assert!(hits.is_empty());
    }

    #[test]
    fn search_rejects_a_query_of_the_wrong_dimension() {
        let dir = tempfile::tempdir().unwrap();
        write_parquet(&dir.path().join("a.parquet"), 4, 8);
        let store = LocalBlobStore::new(dir.path()).unwrap();
        let ds = load(&store, &source("a.parquet")).unwrap();
        let err = ds
            .search(&[1.0, 0.0], 3, DistanceMetric::Cosine, None)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains('8') && err.contains('2'),
            "error must name both the dataset and query dimensions, got: {err}"
        );
    }

    #[test]
    fn search_rejects_an_unsupported_metric() {
        let dir = tempfile::tempdir().unwrap();
        write_parquet(&dir.path().join("a.parquet"), 4, 4);
        let store = LocalBlobStore::new(dir.path()).unwrap();
        let ds = load(&store, &source("a.parquet")).unwrap();
        // Hamming/Jaccard have no meaning for a float vector column.
        let err = ds
            .search(&[1.0, 0.0, 0.0, 0.0], 3, DistanceMetric::Hamming, None)
            .unwrap_err()
            .to_string();
        assert!(err.contains("Hamming"), "got: {err}");
    }

    #[test]
    fn record_at_exposes_scalar_columns() {
        let dir = tempfile::tempdir().unwrap();
        write_parquet(&dir.path().join("a.parquet"), 3, 4);
        let store = LocalBlobStore::new(dir.path()).unwrap();
        let ds = load(&store, &source("a.parquet")).unwrap();
        let rec = ds.record_at(1);
        assert_eq!(rec.get("doc_id"), Some(&FieldValue::String("doc1".into())));
    }

    #[test]
    fn vector_at_returns_the_row_aligned_slice() {
        let dir = tempfile::tempdir().unwrap();
        write_parquet(&dir.path().join("a.parquet"), 5, 4);
        let store = LocalBlobStore::new(dir.path()).unwrap();
        let ds = load(&store, &source("a.parquet")).unwrap();
        for r in 0..ds.row_count() {
            assert_eq!(ds.vector_at(r).len(), 4);
        }
        // write_parquet fills row i, dim d with (i*dim + d) / 10.0
        assert_eq!(ds.vector_at(2)[0], 8.0 / 10.0);
    }
}

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
use veda_core::record::FieldValue;
use veda_core::schema::ExternalSource;

use crate::blob_store::BlobStore;

/// A Parquet dataset registered as a read-only collection.
#[derive(Debug)]
pub struct ExternalDataset {
    dim: u32,
    /// Flattened vector column, `dim` floats per row, row-aligned with `ids`.
    /// Read by `vector_at`/`search`, added in the next task.
    #[allow(dead_code)]
    vectors: Vec<f32>,
    ids: Vec<String>,
    /// Scalar columns, row-aligned, for predicates and for returning field
    /// values on a hit. Read by `record_at`, added in the next task.
    #[allow(dead_code)]
    scalars: HashMap<String, Vec<FieldValue>>,
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
}

/// Resolve `source.uri` to the object keys to read, in sorted order so a
/// dataset is assembled deterministically.
///
/// A URI ending in `/` is a prefix and selects every `*.parquet` beneath it.
/// Anything else is a single object key.
fn resolve_keys(store: &dyn BlobStore, uri: &str) -> Result<Vec<String>> {
    let cleaned = uri.strip_prefix("file://").unwrap_or(uri);
    if cleaned.ends_with('/') {
        let mut keys: Vec<String> = store
            .list(cleaned)?
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
    } else {
        Ok(vec![cleaned.to_owned()])
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
    let mut vectors: Vec<f32> = Vec::new();
    let mut ids: Vec<String> = Vec::new();
    let mut scalars: HashMap<String, Vec<FieldValue>> = HashMap::new();

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

        read_batches(builder, source, &mut ids, &mut vectors, &mut scalars)?;
    }

    let dim = dim.ok_or_else(|| VedaError::Storage(format!("no data behind '{}'", source.uri)))?;
    Ok(ExternalDataset {
        dim,
        vectors,
        ids,
        scalars,
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

/// Append one file's batches to the accumulating dataset: ids, flattened
/// vectors, and every remaining scalar column of a supported type. Columns
/// of an unsupported Arrow type are skipped rather than failing the load.
fn read_batches(
    builder: ParquetRecordBatchReaderBuilder<Bytes>,
    source: &ExternalSource,
    ids: &mut Vec<String>,
    vectors: &mut Vec<f32>,
    scalars: &mut HashMap<String, Vec<FieldValue>>,
) -> Result<()> {
    use arrow::array::{
        Array, BooleanArray, Float32Array, Float64Array, Int32Array, Int64Array, LargeStringArray,
        StringArray,
    };
    use arrow::datatypes::DataType;

    let schema = builder.schema().clone();
    let reader = builder
        .build()
        .map_err(|e| VedaError::Storage(format!("parquet read: {e}")))?;

    for batch_result in reader {
        let batch = batch_result.map_err(|e| VedaError::Storage(format!("parquet read: {e}")))?;

        // ids: Utf8, LargeUtf8, Int32 and Int64 are accepted and stringified,
        // since external ids are commonly integers and SearchHit.id is a
        // String, so no synthetic identifier is needed.
        let id_col = batch
            .column_by_name(&source.id_field)
            .expect("id_field presence checked before read_batches is called");
        match id_col.data_type() {
            DataType::Utf8 => {
                let arr = id_col.as_any().downcast_ref::<StringArray>().unwrap();
                for i in 0..arr.len() {
                    ids.push(arr.value(i).to_owned());
                }
            }
            DataType::LargeUtf8 => {
                let arr = id_col.as_any().downcast_ref::<LargeStringArray>().unwrap();
                for i in 0..arr.len() {
                    ids.push(arr.value(i).to_owned());
                }
            }
            DataType::Int32 => {
                let arr = id_col.as_any().downcast_ref::<Int32Array>().unwrap();
                for i in 0..arr.len() {
                    ids.push(arr.value(i).to_string());
                }
            }
            DataType::Int64 => {
                let arr = id_col.as_any().downcast_ref::<Int64Array>().unwrap();
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

        // vectors: flatten the FixedSizeList<Float32> column.
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
        // silently misaligned with `ids`.
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
                _ => {
                    // Unsupported Arrow type: drop this column entirely
                    // rather than fail the whole load. Remove the entry we
                    // just created (or a same-named column supported in an
                    // earlier file) so `scalars` never holds a partially
                    // filled, misaligned column.
                    scalars.remove(name);
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

    use arrow::array::{ArrayRef, FixedSizeListArray, Float32Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::ArrowWriter;

    use crate::blob_store::LocalBlobStore;
    use veda_core::types::DistanceMetric;

    /// Build a Parquet file with `n` rows: a string id column and a
    /// fixed-size-list vector column of dimension `dim`.
    fn write_parquet(path: &std::path::Path, n: usize, dim: usize) {
        let ids: Vec<String> = (0..n).map(|i| format!("doc{i}")).collect();
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
        write_parquet(&dir.path().join("part/a.parquet"), 5, 4);
        write_parquet(&dir.path().join("part/b.parquet"), 7, 4);
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
}

use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use uuid::Uuid;

use crate::error::{Result, VedaError};
use crate::types::{CollectionId, DistanceMetric, FieldType, IndexType};

/// Definition of a single field within a collection schema.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FieldDefinition {
    name: String,
    field_type: FieldType,
    filterable: bool,
    index_type: IndexType,
    /// Build a full-text BM25 index over this field's string values.
    /// Only meaningful for `String` fields; ignored elsewhere. When
    /// `true`, the segment writer maintains a `Bm25Index` and the
    /// search API accepts a `bm25` query against this field.
    #[serde(default)]
    bm25: bool,
    /// Tokenizer name for the BM25 index. Recognized values:
    /// `"cjk_bigram"` (default), `"whitespace"`. Parsed by
    /// `TokenizerKind::from_name` in veda-storage; unknown names fall
    /// back to the default.
    #[serde(default)]
    bm25_tokenizer: Option<String>,
}

impl FieldDefinition {
    /// Create a new field definition with default settings (not filterable, no index).
    pub fn new(name: impl Into<String>, field_type: FieldType) -> Self {
        Self {
            name: name.into(),
            field_type,
            filterable: false,
            bm25: false,
            bm25_tokenizer: None,
            index_type: IndexType::None,
        }
    }

    /// Create a vector field definition with the given dimension and distance metric.
    /// Defaults to HNSW indexing and no model binding.
    pub fn vector(name: impl Into<String>, dimension: u32, metric: DistanceMetric) -> Self {
        Self {
            name: name.into(),
            field_type: FieldType::Vector {
                dimension,
                metric,
                model_id: None,
                mrl_dims: None,
                auto_embed: None,
            },
            filterable: false,
            bm25: false,
            bm25_tokenizer: None,
            index_type: IndexType::Hnsw,
        }
    }

    /// Create a vector field bound to a specific embedding model.
    /// Embed-insert calls using a different model will be rejected.
    pub fn vector_with_model(
        name: impl Into<String>,
        dimension: u32,
        metric: DistanceMetric,
        model_id: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            field_type: FieldType::Vector {
                dimension,
                metric,
                model_id: Some(model_id.into()),
                mrl_dims: None,
                auto_embed: None,
            },
            filterable: false,
            bm25: false,
            bm25_tokenizer: None,
            index_type: IndexType::Hnsw,
        }
    }

    /// Set the valid MRL truncation lengths for a vector field.
    /// Has no effect on non-vector fields.
    pub fn with_mrl_dims(mut self, dims: Vec<u32>) -> Self {
        if let FieldType::Vector { mrl_dims, .. } = &mut self.field_type {
            *mrl_dims = Some(dims);
        }
        self
    }

    /// Set per-field auto-embed config. The process-document handler
    /// looks at this to pick which source field on each chunk record
    /// (e.g. `"content"` or `"image_blob"`) feeds this vector field,
    /// and what modality of embedder to use. Has no effect on non-
    /// vector fields.
    pub fn with_auto_embed(mut self, cfg: crate::types::AutoEmbedConfig) -> Self {
        if let FieldType::Vector { auto_embed, .. } = &mut self.field_type {
            *auto_embed = Some(cfg);
        }
        self
    }

    /// Set per-field auto-embed config for a sparse-vector field. The
    /// process-document handler reads this and routes the source
    /// field through the configured sparse encoder (SPLADE) to fill
    /// the sparse column. No-op on non-sparse fields.
    pub fn with_sparse_auto_embed(mut self, cfg: crate::types::AutoEmbedConfig) -> Self {
        if let FieldType::SparseVector { auto_embed, .. } = &mut self.field_type {
            *auto_embed = Some(cfg);
        }
        self
    }

    /// Create a sparse vector field definition.
    /// `dimension` is the vocabulary / index upper bound.
    /// Sparse fields do not use any of the dense ANN indexes.
    pub fn sparse_vector(name: impl Into<String>, dimension: u32, metric: DistanceMetric) -> Self {
        Self {
            name: name.into(),
            field_type: FieldType::SparseVector {
                dimension,
                metric,
                auto_embed: None,
            },
            filterable: false,
            bm25: false,
            bm25_tokenizer: None,
            index_type: IndexType::None,
        }
    }

    /// Create a ColPali-style multi-vector field definition.
    /// Each record stores up to `max_vectors` vectors of `dimension`.
    pub fn multi_vector(
        name: impl Into<String>,
        dimension: u32,
        max_vectors: u32,
        metric: DistanceMetric,
    ) -> Self {
        Self {
            name: name.into(),
            field_type: FieldType::MultiVector {
                dimension,
                max_vectors,
                metric,
            },
            filterable: false,
            bm25: false,
            bm25_tokenizer: None,
            index_type: IndexType::None,
        }
    }

    /// Create a blob field definition.
    pub fn blob(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            field_type: FieldType::Blob,
            filterable: false,
            bm25: false,
            bm25_tokenizer: None,
            index_type: IndexType::None,
        }
    }

    /// Set whether this field is filterable.
    pub fn with_filterable(mut self, filterable: bool) -> Self {
        self.filterable = filterable;
        self
    }

    /// Set the index type for this field.
    pub fn with_index_type(mut self, index_type: IndexType) -> Self {
        self.index_type = index_type;
        self
    }

    /// Mark this string field for a full-text BM25 index. No-op on
    /// non-string fields — the segment writer just skips them.
    pub fn with_bm25(mut self, bm25: bool) -> Self {
        self.bm25 = bm25;
        self
    }

    /// Choose the BM25 tokenizer by name. Recognized: `cjk_bigram`
    /// (default), `whitespace`. Pass `None` to use the default.
    pub fn with_bm25_tokenizer(mut self, name: Option<impl Into<String>>) -> Self {
        self.bm25_tokenizer = name.map(Into::into);
        self
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn field_type(&self) -> &FieldType {
        &self.field_type
    }

    pub fn filterable(&self) -> bool {
        self.filterable
    }

    pub fn index_type(&self) -> IndexType {
        self.index_type
    }

    /// Whether this field should have a BM25 inverted index built.
    pub fn bm25(&self) -> bool {
        self.bm25
    }

    /// Optional BM25 tokenizer name (set via the `bm25_tokenizer` JSON
    /// field on collection create). When `None` the segment uses the
    /// default tokenizer.
    pub fn bm25_tokenizer(&self) -> Option<&str> {
        self.bm25_tokenizer.as_deref()
    }
}

/// Registration of a collection whose data Veda does not own and never
/// writes to. `None` for every ordinary collection.
///
/// The referenced files are queried in place: no ingestion, no copy, and
/// no index is built over them.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExternalSource {
    /// Only `"parquet"` is supported today. Present so a second reader
    /// can be added without changing the request shape.
    pub format: String,
    /// `file:///abs/path`, a bare local path, or `s3://bucket/prefix`.
    /// A prefix selects every `*.parquet` object beneath it.
    pub uri: String,
    /// Column holding the record identifier.
    pub id_field: String,
    /// Column holding the dense vector. Must be `FixedSizeList<Float32>`.
    pub vector_field: String,
    /// Distance metric. Not inferable from Parquet, so required.
    pub metric: DistanceMetric,
}

/// Schema describing the structure of a collection.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CollectionSchema {
    id: CollectionId,
    name: String,
    fields: Vec<FieldDefinition>,
    shard_count: u32,
    replication_factor: u32,
    /// Default stored-fields physical layout for new segments. Values:
    /// `"jsonl"` (default, all legacy + most current segments) or
    /// `"columnar"` (per-field column files). Per-segment metadata
    /// still records the format that was actually written, so a
    /// collection can mix formats across segments if the schema flag
    /// changed after some had already sealed.
    #[serde(default)]
    stored_format: Option<String>,
    /// Per-collection opt-in to the Phase C primary-write path.
    /// When `Some(true)`, writes for this collection go through
    /// `veda-log` regardless of the `VEDA_LOG_PRIMARY` env flag.
    /// When `None`, the env flag decides. Lets operators roll out
    /// the new path collection-by-collection.
    #[serde(default)]
    use_log_primary: Option<bool>,
    /// Per-collection opt-in to Phase F MVCC reads/writes. When
    /// `Some(true)`, the SegmentManager is constructed with
    /// `enable_mvcc()` called and writes mirror to the
    /// `MvccStore` overlay; reads at a given `commit_ts` use
    /// `read_at_ts`. When `None`, the `VEDA_MVCC` env flag
    /// decides.
    #[serde(default)]
    use_mvcc: Option<bool>,
    /// Per-collection Row-Level Security. When ,
    /// non-Admin users only read/write records where
    /// . Admin users bypass the
    /// filter. Existing collections without this field deserialise
    /// to  (backward-compatible).
    #[serde(default)]
    rls_owner_field: Option<String>,
    /// Set when this collection reads data Veda does not own. Absent for
    /// ordinary collections, so schemas written before this feature load
    /// unchanged.
    #[serde(default)]
    external: Option<ExternalSource>,
}

impl CollectionSchema {
    /// Start building a new collection schema with the given name.
    pub fn builder(name: impl Into<String>) -> CollectionSchemaBuilder {
        CollectionSchemaBuilder {
            name: name.into(),
            fields: Vec::new(),
            shard_count: 1,
            replication_factor: 1,
            stored_format: None,
            use_log_primary: None,
            use_mvcc: None,
            rls_owner_field: None,
            external: None,
        }
    }

    /// Per-collection log-primary override. Returns the
    /// effective decision, falling back to the env flag when
    /// the schema didn't specify.
    pub fn effective_use_log_primary(&self) -> bool {
        if let Some(v) = self.use_log_primary {
            return v;
        }
        matches!(
            std::env::var("VEDA_LOG_PRIMARY").as_deref(),
            Ok("1") | Ok("true") | Ok("TRUE")
        )
    }

    /// Per-collection MVCC override. Same env-fallback shape as
    /// `effective_use_log_primary`; env var is `VEDA_MVCC`.
    pub fn effective_use_mvcc(&self) -> bool {
        if let Some(v) = self.use_mvcc {
            return v;
        }
        matches!(
            std::env::var("VEDA_MVCC").as_deref(),
            Ok("1") | Ok("true") | Ok("TRUE")
        )
    }

    pub fn id(&self) -> CollectionId {
        self.id
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    /// Return the requested stored-fields format string, if the schema
    /// declared one. The veda-storage segment writer maps unrecognized
    /// values to the JSONL default.
    pub fn stored_format(&self) -> Option<&str> {
        self.stored_format.as_deref()
    }

    pub fn fields(&self) -> &[FieldDefinition] {
        &self.fields
    }

    /// Return only dense vector fields.
    pub fn vector_fields(&self) -> Vec<&FieldDefinition> {
        self.fields
            .iter()
            .filter(|f| matches!(f.field_type, FieldType::Vector { .. }))
            .collect()
    }

    /// Return only sparse vector fields.
    pub fn sparse_vector_fields(&self) -> Vec<&FieldDefinition> {
        self.fields
            .iter()
            .filter(|f| matches!(f.field_type, FieldType::SparseVector { .. }))
            .collect()
    }

    /// Return only multi-vector (ColPali-style) fields.
    pub fn multi_vector_fields(&self) -> Vec<&FieldDefinition> {
        self.fields
            .iter()
            .filter(|f| matches!(f.field_type, FieldType::MultiVector { .. }))
            .collect()
    }

    /// Return only scalar (non-vector, non-blob) fields.
    pub fn scalar_fields(&self) -> Vec<&FieldDefinition> {
        self.fields
            .iter()
            .filter(|f| {
                !matches!(
                    f.field_type,
                    FieldType::Vector { .. }
                        | FieldType::Blob
                        | FieldType::SparseVector { .. }
                        | FieldType::MultiVector { .. }
                )
            })
            .collect()
    }

    /// Return only blob fields.
    pub fn blob_fields(&self) -> Vec<&FieldDefinition> {
        self.fields
            .iter()
            .filter(|f| matches!(f.field_type, FieldType::Blob))
            .collect()
    }

    /// Look up a field by name.
    pub fn get_field(&self, name: &str) -> Option<&FieldDefinition> {
        self.fields.iter().find(|f| f.name == name)
    }

    pub fn shard_count(&self) -> u32 {
        self.shard_count
    }

    pub fn replication_factor(&self) -> u32 {
        self.replication_factor
    }

    /// Return the configured RLS owner field name, if any.
    pub fn rls_owner_field(&self) -> Option<&str> {
        self.rls_owner_field.as_deref()
    }

    /// The external source, if this collection reads data Veda does not own.
    pub fn external(&self) -> Option<&ExternalSource> {
        self.external.as_ref()
    }

    /// Whether this collection reads data Veda does not own. External
    /// collections are read-only and support only dense search and
    /// scalar filters.
    pub fn is_external(&self) -> bool {
        self.external.is_some()
    }
}

/// Builder for constructing a [`CollectionSchema`] with validation.
pub struct CollectionSchemaBuilder {
    name: String,
    fields: Vec<FieldDefinition>,
    shard_count: u32,
    replication_factor: u32,
    stored_format: Option<String>,
    use_log_primary: Option<bool>,
    use_mvcc: Option<bool>,
    rls_owner_field: Option<String>,
    external: Option<ExternalSource>,
}

impl CollectionSchemaBuilder {
    /// Add a field definition to the schema.
    pub fn add_field(mut self, field: FieldDefinition) -> Self {
        self.fields.push(field);
        self
    }

    /// Set the number of shards for this collection.
    pub fn shard_count(mut self, count: u32) -> Self {
        self.shard_count = count;
        self
    }

    /// Set the replication factor for this collection.
    pub fn replication_factor(mut self, factor: u32) -> Self {
        self.replication_factor = factor;
        self
    }

    /// Select the default physical layout for stored-fields data when
    /// new segments seal. `"jsonl"` (default) or `"columnar"`. Unknown
    /// values fall back to JSONL at segment-write time.
    pub fn stored_format(mut self, fmt: impl Into<String>) -> Self {
        self.stored_format = Some(fmt.into());
        self
    }

    /// Opt this collection into the Phase C primary-write path.
    /// `None` (the default) means "use the env flag";
    /// `Some(true)` / `Some(false)` is an explicit per-collection
    /// override.
    pub fn use_log_primary(mut self, v: Option<bool>) -> Self {
        self.use_log_primary = v;
        self
    }

    /// Opt this collection into the Phase F MVCC overlay. Same
    /// shape as `use_log_primary`. Env fallback: `VEDA_MVCC`.
    pub fn use_mvcc(mut self, v: Option<bool>) -> Self {
        self.use_mvcc = v;
        self
    }

    /// Enable Row-Level Security for this collection.
    pub fn rls_owner_field(mut self, field: Option<String>) -> Self {
        self.rls_owner_field = field;
        self
    }

    /// Register this collection as reading an external dataset in place.
    pub fn external(mut self, src: ExternalSource) -> Self {
        self.external = Some(src);
        self
    }

    /// Build and validate the schema.
    ///
    /// Validation rules:
    /// - Name must not be empty.
    /// - Field names must be unique.
    /// - Vector fields must have dimension > 0.
    pub fn build(self) -> Result<CollectionSchema> {
        // Reject empty name
        if self.name.trim().is_empty() {
            return Err(VedaError::InvalidSchema(
                "Collection name must not be empty".to_string(),
            ));
        }

        // Reject duplicate field names
        let mut seen_names = HashSet::new();
        for field in &self.fields {
            if !seen_names.insert(&field.name) {
                return Err(VedaError::InvalidSchema(format!(
                    "Duplicate field name: '{}'",
                    field.name
                )));
            }
        }

        // Reject vector / sparse-vector fields with dimension 0
        for field in &self.fields {
            match &field.field_type {
                FieldType::Vector { dimension, .. } if *dimension == 0 => {
                    return Err(VedaError::InvalidSchema(format!(
                        "Vector field '{}' must have dimension > 0",
                        field.name
                    )));
                }
                FieldType::SparseVector { dimension, .. } if *dimension == 0 => {
                    return Err(VedaError::InvalidSchema(format!(
                        "SparseVector field '{}' must have dimension > 0",
                        field.name
                    )));
                }
                _ => {}
            }
        }

        Ok(CollectionSchema {
            id: Uuid::new_v4(),
            name: self.name,
            fields: self.fields,
            shard_count: self.shard_count,
            replication_factor: self.replication_factor,
            stored_format: self.stored_format,
            use_log_primary: self.use_log_primary,
            use_mvcc: self.use_mvcc,
            rls_owner_field: self.rls_owner_field,
            external: self.external,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn external_source_round_trips_through_serde() {
        let src = ExternalSource {
            format: "parquet".into(),
            uri: "s3://lake/embeddings/".into(),
            id_field: "doc_id".into(),
            vector_field: "embedding".into(),
            metric: DistanceMetric::Cosine,
        };
        let schema = CollectionSchema::builder("lake_docs")
            .external(src.clone())
            .build()
            .unwrap();
        let json = serde_json::to_string(&schema).unwrap();
        let back: CollectionSchema = serde_json::from_str(&json).unwrap();
        assert_eq!(back.external(), Some(&src));
        assert!(back.is_external());
    }

    #[test]
    fn an_ordinary_collection_is_not_external() {
        let schema = CollectionSchema::builder("ordinary").build().unwrap();
        assert_eq!(schema.external(), None);
        assert!(!schema.is_external());
    }

    #[test]
    fn a_schema_saved_before_this_feature_still_loads() {
        // Older persisted schemas have no `external` key at all. `#[serde(default)]`
        // must let them load unchanged, so existing collections need no migration.
        let json = r#"{"id":"00000000-0000-0000-0000-000000000001","name":"legacy",
            "fields":[],"shard_count":1,"replication_factor":1}"#;
        let schema: CollectionSchema = serde_json::from_str(json).unwrap();
        assert!(!schema.is_external());
        assert_eq!(schema.name(), "legacy");
    }
}

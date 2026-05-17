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

    /// Create a sparse vector field definition.
    /// `dimension` is the vocabulary / index upper bound.
    /// Sparse fields do not use any of the dense ANN indexes.
    pub fn sparse_vector(name: impl Into<String>, dimension: u32, metric: DistanceMetric) -> Self {
        Self {
            name: name.into(),
            field_type: FieldType::SparseVector { dimension, metric },
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

/// Schema describing the structure of a collection.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CollectionSchema {
    id: CollectionId,
    name: String,
    fields: Vec<FieldDefinition>,
    shard_count: u32,
    replication_factor: u32,
}

impl CollectionSchema {
    /// Start building a new collection schema with the given name.
    pub fn builder(name: impl Into<String>) -> CollectionSchemaBuilder {
        CollectionSchemaBuilder {
            name: name.into(),
            fields: Vec::new(),
            shard_count: 1,
            replication_factor: 1,
        }
    }

    pub fn id(&self) -> CollectionId {
        self.id
    }

    pub fn name(&self) -> &str {
        &self.name
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
}

/// Builder for constructing a [`CollectionSchema`] with validation.
pub struct CollectionSchemaBuilder {
    name: String,
    fields: Vec<FieldDefinition>,
    shard_count: u32,
    replication_factor: u32,
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
        })
    }
}

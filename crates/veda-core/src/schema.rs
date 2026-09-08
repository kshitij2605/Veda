use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use uuid::Uuid;

use crate::error::{Result, VedaError};
use crate::types::{CollectionId, DistanceMetric, FieldType, IndexType};

/// How sensitive a field's values are.
///
/// Declared once on the schema by whoever designs the collection, rather
/// than re-discovered per record by the ingest-time PII detector, which
/// only ever guesses at something the operator already knows.
/// [`Sensitivity::Restricted`] (and the separate `pii` flag) drive
/// retrieval-time withholding for non-Admin callers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum Sensitivity {
    /// No restriction. The default, so an unannotated field behaves
    /// exactly as it did before annotations existed.
    #[default]
    Public,
    /// Advisory only — carried to agents and operators, never enforced.
    /// Present so a schema can record the distinction without changing
    /// what any caller sees.
    Internal,
    /// Withheld from non-Admin callers on every retrieval path.
    Restricted,
}

/// Meaning attached to a field, for agents and for enforcement.
///
/// Every member is optional and the struct has a `Default`, so a schema
/// written before annotations existed deserialises unchanged — see
/// `a_schema_saved_before_annotations_still_loads`.
///
/// The physical [`FieldType`] says how a value is stored; this says what
/// it *means*. A `String` field may be an ISO-3166 country code, an
/// email address or free prose, and nothing in the type distinguishes
/// them — which is what leaves an agent guessing from the field name.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct FieldAnnotations {
    /// Human-readable description, rendered to agents and MCP clients.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Semantic type beyond the physical one, e.g. `iso3166_alpha2`,
    /// `email`, `currency_code`. Free text by design: a closed
    /// vocabulary is enforceable but unadoptable, and which of the two
    /// matters is an open question the decision doc leaves to Gate A.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub semantic_type: Option<String>,
    /// The values this field actually takes, or a representative sample.
    /// Exists so an agent need not issue exploratory queries to learn
    /// that `"DE"` is stored rather than `"Germany"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value_domain: Option<Vec<String>>,
    /// Sensitivity class. `Public` by default.
    #[serde(default, skip_serializing_if = "Sensitivity::is_public")]
    pub sensitivity: Sensitivity,
    /// Whether this field holds personally-identifiable information.
    /// Enforced identically to [`Sensitivity::Restricted`]; kept
    /// separate because "restricted" and "is PII" are different facts
    /// and an operator may need to record one without the other.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub pii: bool,
}

impl Sensitivity {
    /// Serde skip helper — keeps the default out of persisted schemas.
    fn is_public(&self) -> bool {
        matches!(self, Sensitivity::Public)
    }
}

impl FieldAnnotations {
    /// Whether this field must be withheld from a non-Admin caller.
    ///
    /// The one place the rule lives, so every retrieval path enforces
    /// the same thing.
    pub fn is_withheld_from_non_admin(&self) -> bool {
        self.pii || matches!(self.sensitivity, Sensitivity::Restricted)
    }

    /// Whether any annotation was actually set. Used to skip rendering
    /// an empty object to agents.
    pub fn is_empty(&self) -> bool {
        self.description.is_none()
            && self.semantic_type.is_none()
            && self.value_domain.is_none()
            && self.sensitivity.is_public()
            && !self.pii
    }
}

/// Definition of a single field within a collection schema.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FieldDefinition {
    name: String,
    field_type: FieldType,
    filterable: bool,
    index_type: IndexType,
    /// Field meaning: description, semantic type, value domain,
    /// sensitivity. Defaults to empty, so schemas written before this
    /// existed load unchanged.
    #[serde(default)]
    annotations: FieldAnnotations,
    /// Declarative validation rules. Empty by default, so a field that
    /// declares none is unconstrained. See [`crate::constraints`].
    #[serde(default)]
    constraints: crate::constraints::FieldConstraints,
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
            annotations: FieldAnnotations::default(),
            constraints: crate::constraints::FieldConstraints::default(),
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
            annotations: FieldAnnotations::default(),
            constraints: crate::constraints::FieldConstraints::default(),
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
            annotations: FieldAnnotations::default(),
            constraints: crate::constraints::FieldConstraints::default(),
            index_type: IndexType::Hnsw,
        }
    }

    /// Set the valid MRL truncation lengths for a vector field.
    /// Has no effect on non-vector fields.
    /// Bind this vector field to the embedding model that produces its
    /// vectors.
    ///
    /// No-op on a non-vector field. Set at collection creation for fields
    /// that declare `auto_embed`: two models of the same dimension produce
    /// mutually orthogonal spaces, and a query embedded by the wrong one is
    /// accepted and returns confident nonsense. `model_id` is what lets
    /// `embed-insert` refuse to mix models and `select_query_model` refuse a
    /// model missing from the registry; neither can fire while it is `None`.
    pub fn with_model_id(mut self, model_id: impl Into<String>) -> Self {
        if let FieldType::Vector { model_id: slot, .. } = &mut self.field_type {
            *slot = Some(model_id.into());
        }
        self
    }

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
            annotations: FieldAnnotations::default(),
            constraints: crate::constraints::FieldConstraints::default(),
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
            annotations: FieldAnnotations::default(),
            constraints: crate::constraints::FieldConstraints::default(),
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
            annotations: FieldAnnotations::default(),
            constraints: crate::constraints::FieldConstraints::default(),
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

    /// Attach field meaning — description, semantic type, value domain,
    /// sensitivity.
    pub fn with_annotations(mut self, annotations: FieldAnnotations) -> Self {
        self.annotations = annotations;
        self
    }

    /// Attach declarative validation rules.
    pub fn with_constraints(mut self, constraints: crate::constraints::FieldConstraints) -> Self {
        self.constraints = constraints;
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

    /// Field meaning. Empty for a field that was never annotated.
    pub fn annotations(&self) -> &FieldAnnotations {
        &self.annotations
    }

    /// Validation rules. Empty for a field that declares none.
    pub fn constraints(&self) -> &crate::constraints::FieldConstraints {
        &self.constraints
    }

    /// Whether this field must be withheld from a non-Admin caller.
    /// Every retrieval path consults this rather than reading
    /// `sensitivity` and `pii` separately, so the rule cannot drift
    /// between paths.
    pub fn is_withheld_from_non_admin(&self) -> bool {
        self.annotations.is_withheld_from_non_admin()
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
    /// A key or key prefix **within the server's configured object store**,
    /// or `file:///abs/path` for a local file. Anything not ending in
    /// `.parquet` is treated as a prefix and selects every `*.parquet`
    /// object beneath it.
    ///
    /// A URI naming a bucket (for example `s3://bucket/prefix`) is
    /// rejected: Veda holds one object store against one configured
    /// bucket, so an authority here would be silently discarded and the
    /// read would go to the wrong bucket. Configure the bucket on the
    /// server instead.
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
    /// Check a record against every field's declared constraints.
    ///
    /// Returns all violations across all fields, so one pass tells an
    /// operator everything wrong with the record. An empty vec means the
    /// record satisfies the schema — which is every record on a collection
    /// that declares no constraints.
    pub fn validate_record(
        &self,
        record: &crate::record::Record,
    ) -> Vec<crate::constraints::Violation> {
        let mut out = Vec::new();
        for f in &self.fields {
            if f.constraints().is_empty() {
                continue;
            }
            out.extend(f.constraints().check(f.name(), record.get(f.name())));
        }
        out
    }

    /// Names of fields that must not be returned to a non-Admin caller.
    ///
    /// Empty for any collection whose fields carry no sensitivity
    /// annotation, which is every collection created before this
    /// feature — so enforcement is a no-op until someone opts in.
    pub fn withheld_fields_for_non_admin(&self) -> Vec<&str> {
        self.fields
            .iter()
            .filter(|f| f.is_withheld_from_non_admin())
            .map(|f| f.name())
            .collect()
    }

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
            uri: "embeddings/".into(),
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
    fn annotations_round_trip_through_serde() {
        let ann = FieldAnnotations {
            description: Some("ISO-3166 country of the supplier HQ".into()),
            semantic_type: Some("iso3166_alpha2".into()),
            value_domain: Some(vec!["DE".into(), "EE".into(), "CL".into()]),
            sensitivity: Sensitivity::Internal,
            pii: false,
        };
        let schema = CollectionSchema::builder("suppliers")
            .add_field(
                FieldDefinition::new("ctry_cd", FieldType::String).with_annotations(ann.clone()),
            )
            .build()
            .unwrap();
        let json = serde_json::to_string(&schema).unwrap();
        let back: CollectionSchema = serde_json::from_str(&json).unwrap();
        assert_eq!(back.get_field("ctry_cd").unwrap().annotations(), &ann);
    }

    #[test]
    fn an_unannotated_field_serialises_no_annotation_keys() {
        // The skip_serializing_if attributes keep persisted schemas
        // byte-identical for collections that never opt in. Without
        // them every existing schema would gain a noisy empty object.
        let schema = CollectionSchema::builder("plain")
            .add_field(FieldDefinition::new("body", FieldType::String))
            .build()
            .unwrap();
        let json = serde_json::to_string(&schema).unwrap();
        assert!(!json.contains("description"), "{json}");
        assert!(!json.contains("sensitivity"), "{json}");
        assert!(!json.contains("semantic_type"), "{json}");
        assert!(!json.contains("\"pii\""), "{json}");
    }

    #[test]
    fn restricted_and_pii_are_withheld_but_public_and_internal_are_not() {
        let field = |name: &str, s: Sensitivity, pii: bool| {
            FieldDefinition::new(name, FieldType::String).with_annotations(FieldAnnotations {
                sensitivity: s,
                pii,
                ..Default::default()
            })
        };
        let schema = CollectionSchema::builder("people")
            .add_field(field("nickname", Sensitivity::Public, false))
            .add_field(field("cost_centre", Sensitivity::Internal, false))
            .add_field(field("salary", Sensitivity::Restricted, false))
            .add_field(field("email", Sensitivity::Public, true))
            .build()
            .unwrap();

        // Internal is advisory: it is carried, never enforced. Asserting
        // that explicitly so a later change cannot quietly promote it.
        assert_eq!(
            schema.withheld_fields_for_non_admin(),
            vec!["salary", "email"]
        );
    }

    #[test]
    fn an_unannotated_collection_withholds_nothing() {
        // Enforcement must be a no-op for every collection that predates
        // annotations, or shipping this breaks existing deployments.
        let schema = CollectionSchema::builder("legacy")
            .add_field(FieldDefinition::new("body", FieldType::String))
            .add_field(FieldDefinition::vector("vec", 8, DistanceMetric::Cosine))
            .build()
            .unwrap();
        assert!(schema.withheld_fields_for_non_admin().is_empty());
    }

    #[test]
    fn a_schema_saved_before_annotations_still_loads() {
        // A persisted field object with no `annotations` key at all.
        // `#[serde(default)]` must let it load, so existing collections
        // need no migration.
        let json = r#"{"id":"00000000-0000-0000-0000-000000000002","name":"legacy",
            "fields":[{"name":"body","field_type":"String","filterable":true,
            "index_type":"None"}],"shard_count":1,"replication_factor":1}"#;
        let schema: CollectionSchema = serde_json::from_str(json).unwrap();
        let f = schema.get_field("body").unwrap();
        assert!(f.annotations().is_empty());
        assert!(!f.is_withheld_from_non_admin());
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

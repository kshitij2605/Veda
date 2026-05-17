use serde::{Deserialize, Serialize};
use uuid::Uuid;

// Type aliases
pub type RecordId = Uuid;
pub type CollectionId = Uuid;
pub type SegmentId = Uuid;
pub type ShardId = u32;

/// Distance metrics for vector fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DistanceMetric {
    Cosine,
    Euclidean,
    DotProduct,
    Hamming,
    Manhattan,
    Jaccard,
}

/// Field types supported by Veda.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FieldType {
    Vector {
        dimension: u32,
        metric: DistanceMetric,
        /// Optional identifier of the embedding model that produced vectors
        /// for this field. When set, embed-insert calls must use the same
        /// model — guards against silently mixing incompatible embedding
        /// spaces when an upstream model is swapped.
        #[serde(default)]
        model_id: Option<String>,
        /// Valid Matryoshka truncation lengths (e.g. `[256, 512, 1024, 2048]`).
        /// Stored vectors are always full-dimension; at search time the
        /// caller may pass `truncate_to=N` where N is in this list, and
        /// distances are computed on the first N components only.
        /// When `None`, any truncation up to `dimension` is permitted.
        #[serde(default)]
        mrl_dims: Option<Vec<u32>>,
    },
    /// High-dimensional, sparse f32 vector (e.g. SPLADE, BM25).
    /// `dimension` is the vocabulary / index upper bound. Indices in stored
    /// values must be `< dimension`.
    SparseVector {
        dimension: u32,
        metric: DistanceMetric,
    },
    String,
    Int32,
    Int64,
    Float32,
    Float64,
    Bool,
    Timestamp,
    Array(Box<FieldType>),
    Json,
    Blob,
    Uuid,
}

/// Index types for fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum IndexType {
    Hnsw,
    DiskAnn,
    IvfPq,
    Flat,
    SuffixArray,
    Inverted,
    Range,
    Bitmap,
    Hash,
    None,
}

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

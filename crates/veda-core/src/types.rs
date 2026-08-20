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
        /// Per-field auto-embed configuration. When set, the process-document
        /// pipeline reads `auto_embed.source` from each chunk record and
        /// runs it through an embedder selected for `auto_embed.modality`.
        /// When `None`, the gateway falls back to the legacy name-based
        /// heuristic (text bucket vs image bucket).
        #[serde(default)]
        auto_embed: Option<AutoEmbedConfig>,
    },
    /// High-dimensional, sparse f32 vector (e.g. SPLADE, BM25).
    /// `dimension` is the vocabulary / index upper bound. Indices in stored
    /// values must be `< dimension`.
    SparseVector {
        dimension: u32,
        metric: DistanceMetric,
        /// Per-field auto-embed config. When set, embed-insert and
        /// process-document route the named source field through a
        /// sparse encoder (SPLADE today) to populate this column.
        #[serde(default)]
        auto_embed: Option<AutoEmbedConfig>,
    },
    /// Multi-vector representation (ColBERT / ColPali / ColQwen style): each
    /// record contains a variable number of `dimension`-sized vectors and
    /// retrieval uses MaxSim — sum over query tokens of the maximum
    /// similarity to any document token.
    MultiVector {
        dimension: u32,
        /// Soft cap on the number of vectors per record (rejects inserts
        /// exceeding this — 0 means unbounded).
        max_vectors: u32,
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

/// Per-field auto-embed configuration used by the process-document /
/// embed-insert paths. Lets a schema declare exactly which source field
/// feeds which vector field and which modality to embed with — replaces
/// the global "any image in the doc → all vectors are multimodal"
/// heuristic that broke mixed text+image schemas.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AutoEmbedConfig {
    /// Name of the source field on each record (e.g. `"content"` for a
    /// text-chunk field or `"image_blob"` for an image-bytes field).
    pub source: String,
    /// Which modality of embedder to select from the registry. The first
    /// registered model whose `supported_modalities` contains this
    /// modality is used.
    pub modality: AutoEmbedModality,
}

/// Modality enum used by [`AutoEmbedConfig`]. Kept distinct from
/// `veda_core::embedding::Modality` so that `veda-core::types` doesn't
/// pull in the heavier embedding module just for two enum values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AutoEmbedModality {
    Text,
    Image,
    Video,
    Audio,
}

/// Index types for fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum IndexType {
    Hnsw,
    DiskAnn,
    IvfPq,
    Flat,
    /// 1-bit-per-dim asymmetric quantization (RaBitQ). Brute-force scan
    /// over compact binary codes — 32× smaller in memory than raw f32
    /// vectors, but no graph pruning. Wins above ~1M rows where the
    /// cheaper distance op + smaller working set beats HNSW's pruning.
    RaBitQ,
    /// Two-stage retrieval: RaBitQ for coarse candidate scan, then
    /// raw-f32 distance for the precise top-k pass. Trades the
    /// memory savings of pure RaBitQ for higher recall on the top
    /// of the result set; useful when quality matters more than
    /// memory footprint.
    RaBitQRerank,
    /// Deep-quantized HNSW: graph trained on raw vectors but
    /// persisted and searched with RaBitQ codes. ~32× smaller than
    /// raw HNSW; graph pruning preserves search latency. Use above
    /// ~10M rows where raw f32 vectors blow past L3 cache.
    HnswRaBitQ,
    /// CAGRA — NVIDIA cuVS GPU-native graph ANN. Build + search
    /// on GPU; serialises to a hnswlib-compatible format for cold
    /// storage. First-class as of 2026-05-19 per the SOTA audit
    /// (`research/2026-05-19-sota-audit/01-vector-indexes.md`).
    /// Requires the `gpu_cuvs` Cargo feature at compile time and
    /// an NVIDIA GPU + driver ≥535 at runtime; without the feature
    /// the index returns a clear error on build/search.
    Cagra,
    /// RaBitQ (1-bit coarse) + Extended-RaBitQ (4-bit rerank).
    /// Recommended default rerank stage per the 2026 SOTA audit
    /// (`research/2026-05-19-sota-audit/02-quantization.md`) —
    /// ~8× less rerank memory than `RaBitQRerank` (which keeps
    /// full f32) at the same recall envelope for typical
    /// embedding distributions.
    RaBitQExtRerank,
    /// 4-bit per-dimension scalar quantization. ~D/2 bytes/vec;
    /// turbovec-competitive recall. Brute-force scan.
    Sq4,
    /// 8-bit per-dimension scalar quantization. ~D bytes/vec;
    /// highest-recall quantized option (~0.99). Brute-force scan.
    Sq8,
    /// SPANN-style object-storage-native IVF. Centroids stay resident
    /// (`O(nlist x dim)`, independent of corpus size); posting lists live
    /// in the configured object store and are fetched by byte range, so a
    /// query reads only the lists it probes. Requires a registered object
    /// store (`veda_index::spann::register_store`), which the server
    /// installs at startup. See `crates/veda-index/src/spann.rs`.
    Spann,
    SuffixArray,
    Inverted,
    Range,
    Bitmap,
    Hash,
    None,
}

//! Attribute keys for **retrieval-quality** span attributes on the search
//! span, plus the `db.*` OTel database semantic-convention keys Veda uses.
//!
//! Centralized in one module — mirroring [`crate::genai_semconv`] — so a
//! span-attribute rename is a one-file change. The search handler declares
//! these as `tracing::field::Empty` in its `#[instrument]` `fields(...)` and
//! `record(...)`s the computed values, which the always-installed
//! `tracing_opentelemetry` layer maps to OTel span attributes verbatim.
//!
//! The `retrieval.*` namespace is Veda-specific (there is no stable OTel
//! semconv for vector retrieval yet); the `db.system` / `db.collection`
//! keys follow the OTel database conventions so a Veda RAG trace lines up
//! with how a backing datastore span would be tagged.

// ── Database (OTel semconv) ───────────────────────────────────────────────

/// `db.system` — the database system; always `"veda"` for search spans.
pub const DB_SYSTEM: &str = "db.system";
/// `db.collection` — the collection (user-facing name) being queried.
pub const DB_COLLECTION: &str = "db.collection";

/// Value of `db.system` for Veda search spans.
pub const DB_SYSTEM_VEDA: &str = "veda";

// ── Retrieval quality (Veda-specific) ─────────────────────────────────────

/// `retrieval.query_type` — coarse query classification
/// (`dense` / `sparse` / `hybrid` / `bm25` / …).
pub const RETRIEVAL_QUERY_TYPE: &str = "retrieval.query_type";
/// `retrieval.top_k` — the caller-requested result limit.
pub const RETRIEVAL_TOP_K: &str = "retrieval.top_k";
/// `retrieval.result_count` — number of hits actually returned.
pub const RETRIEVAL_RESULT_COUNT: &str = "retrieval.result_count";
/// `retrieval.reranked` — whether a cross-encoder rerank stage ran.
pub const RETRIEVAL_RERANKED: &str = "retrieval.reranked";
/// `retrieval.has_filter` — whether a scalar/RLS filter was applied.
pub const RETRIEVAL_HAS_FILTER: &str = "retrieval.has_filter";
/// `retrieval.index_type` — the backing vector index type of the queried
/// field (`hnsw` / `ivf_pq` / …); absent for lexical-only queries.
pub const RETRIEVAL_INDEX_TYPE: &str = "retrieval.index_type";

// ── Per-phase timing breakdown (mirrors `ExplainTiming`) ──────────────────

/// `retrieval.filter_ms` — time spent applying the scalar filter.
pub const RETRIEVAL_FILTER_MS: &str = "retrieval.filter_ms";
/// `retrieval.ann_search_ms` — time spent in the ANN index walk.
pub const RETRIEVAL_ANN_SEARCH_MS: &str = "retrieval.ann_search_ms";
/// `retrieval.scoring_ms` — time spent scoring / distance computation.
pub const RETRIEVAL_SCORING_MS: &str = "retrieval.scoring_ms";
/// `retrieval.fusion_ms` — time spent fusing multi-leg / multi-vector results.
pub const RETRIEVAL_FUSION_MS: &str = "retrieval.fusion_ms";
/// `retrieval.total_ms` — sum of the retrieval phases.
pub const RETRIEVAL_TOTAL_MS: &str = "retrieval.total_ms";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn attribute_keys_are_stable() {
        assert_eq!(DB_SYSTEM, "db.system");
        assert_eq!(DB_COLLECTION, "db.collection");
        assert_eq!(DB_SYSTEM_VEDA, "veda");
        assert_eq!(RETRIEVAL_QUERY_TYPE, "retrieval.query_type");
        assert_eq!(RETRIEVAL_TOP_K, "retrieval.top_k");
        assert_eq!(RETRIEVAL_RESULT_COUNT, "retrieval.result_count");
        assert_eq!(RETRIEVAL_RERANKED, "retrieval.reranked");
        assert_eq!(RETRIEVAL_HAS_FILTER, "retrieval.has_filter");
        assert_eq!(RETRIEVAL_INDEX_TYPE, "retrieval.index_type");
        assert_eq!(RETRIEVAL_FILTER_MS, "retrieval.filter_ms");
        assert_eq!(RETRIEVAL_ANN_SEARCH_MS, "retrieval.ann_search_ms");
        assert_eq!(RETRIEVAL_SCORING_MS, "retrieval.scoring_ms");
        assert_eq!(RETRIEVAL_FUSION_MS, "retrieval.fusion_ms");
        assert_eq!(RETRIEVAL_TOTAL_MS, "retrieval.total_ms");
    }
}

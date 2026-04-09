//! Veda metric name constants for the `metrics` crate.
//!
//! All metric recording uses `metrics::counter!()`, `metrics::histogram!()`,
//! and `metrics::gauge!()` macros directly in handlers, referencing these constants.

pub const REQUESTS_TOTAL: &str = "veda_requests_total";
pub const REQUEST_DURATION: &str = "veda_request_duration_seconds";
pub const SEARCH_DURATION: &str = "veda_search_duration_seconds";
pub const INSERT_TOTAL: &str = "veda_insert_total";
pub const INSERT_DURATION: &str = "veda_insert_duration_seconds";
pub const EMBEDDING_DURATION: &str = "veda_embedding_duration_seconds";
pub const BLOB_OP_DURATION: &str = "veda_blob_operation_duration_seconds";
pub const SEGMENTS_TOTAL: &str = "veda_segments_total";
pub const INDEX_BUILD_DURATION: &str = "veda_index_build_duration_seconds";
pub const COMPACTION_DURATION: &str = "veda_compaction_duration_seconds";
pub const ERRORS_TOTAL: &str = "veda_errors_total";
pub const ACTIVE_COLLECTIONS: &str = "veda_active_collections";
pub const RECORDS_TOTAL: &str = "veda_records_total";
pub const COMPACTION_IN_PROGRESS: &str = "veda_compaction_in_progress";
pub const INDEX_REBUILD_IN_PROGRESS: &str = "veda_index_rebuild_in_progress";
pub const SEAL_DURATION: &str = "veda_seal_duration_seconds";

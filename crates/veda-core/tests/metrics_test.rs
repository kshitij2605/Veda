use veda_core::metrics;

#[test]
fn test_metric_names_are_valid_prometheus_names() {
    let names = [
        metrics::REQUESTS_TOTAL,
        metrics::REQUEST_DURATION,
        metrics::SEARCH_DURATION,
        metrics::INSERT_TOTAL,
        metrics::INSERT_DURATION,
        metrics::EMBEDDING_DURATION,
        metrics::BLOB_OP_DURATION,
        metrics::SEGMENTS_TOTAL,
        metrics::INDEX_BUILD_DURATION,
        metrics::COMPACTION_DURATION,
        metrics::ERRORS_TOTAL,
        metrics::ACTIVE_COLLECTIONS,
        metrics::RECORDS_TOTAL,
        metrics::COMPACTION_IN_PROGRESS,
        metrics::INDEX_REBUILD_IN_PROGRESS,
        metrics::SEAL_DURATION,
    ];

    for name in &names {
        assert!(!name.is_empty(), "metric name must not be empty");
        assert!(
            name.starts_with(|c: char| c.is_ascii_alphabetic() || c == '_'),
            "metric '{name}' must start with letter or underscore"
        );
        assert!(
            name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_'),
            "metric '{name}' contains invalid characters"
        );
    }
}

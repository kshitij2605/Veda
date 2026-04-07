use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

/// Simple metrics collector using atomic counters.
pub struct Metrics {
    pub query_count: AtomicU64,
    pub query_latency_sum_us: AtomicU64,
    pub insert_count: AtomicU64,
    pub insert_latency_sum_us: AtomicU64,
    pub error_count: AtomicU64,
}

static GLOBAL_METRICS: Metrics = Metrics {
    query_count: AtomicU64::new(0),
    query_latency_sum_us: AtomicU64::new(0),
    insert_count: AtomicU64::new(0),
    insert_latency_sum_us: AtomicU64::new(0),
    error_count: AtomicU64::new(0),
};

impl Metrics {
    /// Return the global singleton metrics instance.
    pub fn global() -> &'static Metrics {
        &GLOBAL_METRICS
    }

    /// Record a completed query with its latency.
    pub fn record_query(&self, latency: Duration) {
        self.query_count.fetch_add(1, Ordering::Relaxed);
        self.query_latency_sum_us
            .fetch_add(latency.as_micros() as u64, Ordering::Relaxed);
    }

    /// Record a completed insert with its latency.
    pub fn record_insert(&self, latency: Duration) {
        self.insert_count.fetch_add(1, Ordering::Relaxed);
        self.insert_latency_sum_us
            .fetch_add(latency.as_micros() as u64, Ordering::Relaxed);
    }

    /// Record an error.
    pub fn record_error(&self) {
        self.error_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Export metrics in Prometheus text exposition format.
    pub fn export_prometheus(&self) -> String {
        let query_count = self.query_count.load(Ordering::Relaxed);
        let query_latency = self.query_latency_sum_us.load(Ordering::Relaxed);
        let insert_count = self.insert_count.load(Ordering::Relaxed);
        let insert_latency = self.insert_latency_sum_us.load(Ordering::Relaxed);
        let error_count = self.error_count.load(Ordering::Relaxed);

        format!(
            "# HELP veda_query_total Total number of queries.\n\
             # TYPE veda_query_total counter\n\
             veda_query_total {query_count}\n\
             # HELP veda_query_latency_microseconds_total Total query latency in microseconds.\n\
             # TYPE veda_query_latency_microseconds_total counter\n\
             veda_query_latency_microseconds_total {query_latency}\n\
             # HELP veda_insert_total Total number of inserts.\n\
             # TYPE veda_insert_total counter\n\
             veda_insert_total {insert_count}\n\
             # HELP veda_insert_latency_microseconds_total Total insert latency in microseconds.\n\
             # TYPE veda_insert_latency_microseconds_total counter\n\
             veda_insert_latency_microseconds_total {insert_latency}\n\
             # HELP veda_errors_total Total number of errors.\n\
             # TYPE veda_errors_total counter\n\
             veda_errors_total {error_count}\n"
        )
    }
}

use std::sync::atomic::Ordering;
use std::thread;
use std::time::Duration;

use veda_core::metrics::Metrics;

// Note: Since Metrics uses a global singleton with AtomicU64 counters that persist
// across tests, we test relative increments rather than absolute values.

#[test]
fn test_record_queries_increments_count() {
    let m = Metrics::global();
    let before = m.query_count.load(Ordering::Relaxed);

    m.record_query(Duration::from_micros(500));
    m.record_query(Duration::from_micros(300));

    let after = m.query_count.load(Ordering::Relaxed);
    assert_eq!(after - before, 2);
}

#[test]
fn test_record_inserts_increments_count() {
    let m = Metrics::global();
    let before = m.insert_count.load(Ordering::Relaxed);

    m.record_insert(Duration::from_micros(100));

    let after = m.insert_count.load(Ordering::Relaxed);
    assert_eq!(after - before, 1);
}

#[test]
fn test_record_errors() {
    let m = Metrics::global();
    let before = m.error_count.load(Ordering::Relaxed);

    m.record_error();
    m.record_error();
    m.record_error();

    let after = m.error_count.load(Ordering::Relaxed);
    assert_eq!(after - before, 3);
}

#[test]
fn test_export_prometheus_format() {
    let m = Metrics::global();
    // Record something so values are non-zero.
    m.record_query(Duration::from_micros(42));

    let output = m.export_prometheus();

    // Verify expected metric names and Prometheus format markers.
    assert!(output.contains("veda_query_total"));
    assert!(output.contains("veda_query_latency_microseconds_total"));
    assert!(output.contains("veda_insert_total"));
    assert!(output.contains("veda_insert_latency_microseconds_total"));
    assert!(output.contains("veda_errors_total"));
    assert!(output.contains("# HELP"));
    assert!(output.contains("# TYPE"));
    assert!(output.contains("counter"));
}

#[test]
fn test_concurrent_metric_recording() {
    let m = Metrics::global();
    let before_query = m.query_count.load(Ordering::SeqCst);
    let before_insert = m.insert_count.load(Ordering::SeqCst);

    let threads: Vec<_> = (0..10)
        .map(|_| {
            thread::spawn(|| {
                let m = Metrics::global();
                for _ in 0..100 {
                    m.record_query(Duration::from_micros(10));
                    m.record_insert(Duration::from_micros(5));
                }
            })
        })
        .collect();

    for t in threads {
        t.join().unwrap();
    }

    let after_query = m.query_count.load(Ordering::SeqCst);
    let after_insert = m.insert_count.load(Ordering::SeqCst);

    // Use >= because other tests in the same process may also be recording
    // metrics concurrently via the global singleton.
    assert!(after_query - before_query >= 1000);
    assert!(after_insert - before_insert >= 1000);
}

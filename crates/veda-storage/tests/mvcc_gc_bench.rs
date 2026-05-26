//! MVCC GC stress benchmark.
//!
//! Exercises GC correctness and throughput under high version counts.
//! Run with: `cargo test --test mvcc_gc_bench -- --ignored --nocapture`

use bytes::Bytes;
use std::time::Instant;
use veda_storage::mvcc::MvccStore;

fn gc_stress(n_keys: usize, versions_per_key: usize, safe_ts_frac: f64) {
    let total = n_keys * versions_per_key;
    let mut store = MvccStore::new().with_max_versions_per_key(versions_per_key + 1);

    let t0 = Instant::now();
    for k in 0..n_keys {
        let key = format!("key_{k:06}");
        for v in 0..versions_per_key {
            let ts = (k * versions_per_key + v + 1) as u128;
            let val = format!("v{v}");
            store.put_payload(key.as_bytes(), ts, Bytes::from(val.into_bytes())).unwrap();
        }
    }
    let insert_ms = t0.elapsed().as_secs_f64() * 1000.0;
    eprintln!("  inserted {total} versions ({n_keys} keys x {versions_per_key} ver) in {insert_ms:.1}ms");

    let max_ts = total as u128;
    let safe = (max_ts as f64 * safe_ts_frac) as u128;
    store.set_safe_ts(safe);

    let t1 = Instant::now();
    let dropped = store.gc();
    let gc_ms = t1.elapsed().as_secs_f64() * 1000.0;
    let gc_rate = if gc_ms > 0.0 { dropped as f64 / (gc_ms / 1000.0) } else { 0.0 };
    eprintln!("  GC: safe_ts={safe} dropped={dropped} in {gc_ms:.1}ms ({gc_rate:.0} ver/sec)");

    let t2 = Instant::now();
    for k in 0..n_keys {
        let key = format!("key_{k:06}");
        let key_max_ts = ((k + 1) * versions_per_key) as u128;
        let key_min_ts = (k * versions_per_key + 1) as u128;
        if key_min_ts <= safe {
            let result = store.get_at(key.as_bytes(), safe);
            assert!(result.is_some(), "floor must be kept for key {key}");
        }
        let latest = store.get_at(key.as_bytes(), max_ts);
        assert!(latest.is_some(), "latest version must survive GC for key {key}");
    }
    let verify_ms = t2.elapsed().as_secs_f64() * 1000.0;
    eprintln!("  verify: {n_keys} floor reads in {verify_ms:.1}ms — all correct");
}

#[test]
fn gc_stress_1k_keys_100_versions() {
    eprintln!("\n=== MVCC GC bench: 1k x 100 (100k versions) ===");
    gc_stress(1_000, 100, 0.5);
}

#[test]
#[ignore]
fn gc_stress_10k_keys_100_versions() {
    eprintln!("\n=== MVCC GC bench: 10k x 100 (1M versions) ===");
    gc_stress(10_000, 100, 0.5);
}

#[test]
#[ignore]
fn gc_stress_100k_keys_10_versions() {
    eprintln!("\n=== MVCC GC bench: 100k x 10 (1M versions) ===");
    gc_stress(100_000, 10, 0.5);
}

#[test]
#[ignore]
fn gc_stress_10k_keys_1000_versions() {
    eprintln!("\n=== MVCC GC bench: 10k x 1000 (10M versions) ===");
    gc_stress(10_000, 1_000, 0.5);
}

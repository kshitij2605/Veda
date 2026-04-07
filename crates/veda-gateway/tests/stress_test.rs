use std::net::SocketAddr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use reqwest::Client;
use serde_json::{json, Value};
use tokio::net::TcpListener;

use veda_gateway::rest::{build_router, state::AppState};

// ── Latency stats helper ────────────────────────────────────────────────────

struct LatencyStats {
    samples: Vec<Duration>,
}

impl LatencyStats {
    fn new() -> Self {
        Self {
            samples: Vec::new(),
        }
    }

    fn record(&mut self, d: Duration) {
        self.samples.push(d);
    }

    fn p50(&self) -> Duration {
        self.percentile(50)
    }

    fn p95(&self) -> Duration {
        self.percentile(95)
    }

    fn p99(&self) -> Duration {
        self.percentile(99)
    }

    fn avg(&self) -> Duration {
        if self.samples.is_empty() {
            return Duration::ZERO;
        }
        let total: Duration = self.samples.iter().sum();
        total / self.samples.len() as u32
    }

    fn percentile(&self, pct: usize) -> Duration {
        if self.samples.is_empty() {
            return Duration::ZERO;
        }
        let mut sorted = self.samples.clone();
        sorted.sort();
        let idx = (pct as f64 / 100.0 * (sorted.len() - 1) as f64).round() as usize;
        sorted[idx.min(sorted.len() - 1)]
    }

    fn count(&self) -> usize {
        self.samples.len()
    }
}

fn fmt_duration(d: Duration) -> String {
    let micros = d.as_micros();
    if micros < 1_000 {
        format!("{micros}us")
    } else {
        format!("{:.2}ms", micros as f64 / 1_000.0)
    }
}

fn print_report(title: &str, fields: &[(&str, String)]) {
    let width = 50;
    println!();
    println!("{}", "=".repeat(width + 4));
    println!("  {title}");
    println!("{}", "=".repeat(width + 4));
    for (label, value) in fields {
        println!("  {label:<16}{value}");
    }
    println!("{}", "=".repeat(width + 4));
    println!();
}

// ── Test infrastructure ─────────────────────────────────────────────────────

async fn start_test_server() -> (String, tempfile::TempDir) {
    let tmp = tempfile::tempdir().unwrap();
    let app_state = AppState::new(tmp.path());
    let router = build_router(app_state);

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr: SocketAddr = listener.local_addr().unwrap();
    let base_url = format!("http://{addr}");

    tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });

    (base_url, tmp)
}

fn shared_client() -> Client {
    Client::new()
}

fn seeded_vector(dim: usize, seed: u32) -> Vec<f32> {
    (0..dim)
        .map(|i| (seed as f32 * 0.1 + i as f32 * 0.03) % 1.0)
        .collect()
}

async fn create_collection(client: &Client, base: &str, name: &str, dim: usize) {
    let resp = client
        .post(format!("{base}/v1/collections"))
        .json(&json!({
            "name": name,
            "fields": [
                { "name": "embedding", "type": "vector", "dimension": dim, "metric": "cosine" },
                { "name": "tag", "type": "string", "filterable": true }
            ],
            "shard_count": 1
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 201, "Failed to create collection {name}");
}

async fn insert_records(
    client: &Client,
    base: &str,
    collection: &str,
    dim: usize,
    start_seed: u32,
    count: u32,
) -> Duration {
    let records: Vec<Value> = (start_seed..start_seed + count)
        .map(|i| {
            json!({
                "embedding": seeded_vector(dim, i),
                "tag": format!("item_{i}")
            })
        })
        .collect();

    let t = Instant::now();
    let resp = client
        .post(format!("{base}/v1/collections/{collection}/insert"))
        .json(&json!({ "records": records }))
        .send()
        .await
        .unwrap();
    let elapsed = t.elapsed();
    assert_eq!(
        resp.status(),
        201,
        "Insert failed: {}",
        resp.text().await.unwrap_or_default()
    );
    elapsed
}

async fn search(
    client: &Client,
    base: &str,
    collection: &str,
    dim: usize,
    seed: u32,
    limit: u32,
) -> (Duration, bool) {
    let t = Instant::now();
    let resp = client
        .post(format!("{base}/v1/collections/{collection}/search"))
        .json(&json!({
            "query": {
                "vector": { "field": "embedding", "values": seeded_vector(dim, seed) }
            },
            "limit": limit
        }))
        .send()
        .await;
    let elapsed = t.elapsed();

    match resp {
        Ok(r) if r.status().is_success() => (elapsed, true),
        _ => (elapsed, false),
    }
}

// ── 1. Concurrent writes stress test ────────────────────────────────────────

#[tokio::test]
#[ignore]
async fn test_concurrent_writes() {
    let (base, _tmp) = start_test_server().await;
    let client = shared_client();

    create_collection(&client, &base, "stress_writes", 32).await;

    let num_writers: u32 = 10;
    let records_per_writer: u32 = 100;
    let errors = Arc::new(AtomicU64::new(0));
    let wall_start = Instant::now();

    let mut handles = Vec::new();
    for writer_id in 0..num_writers {
        let c = client.clone();
        let b = base.clone();
        let errs = errors.clone();

        handles.push(tokio::spawn(async move {
            let mut latencies = LatencyStats::new();
            let start_seed = writer_id * records_per_writer;

            // Insert one record at a time to measure per-request latency
            for i in 0..records_per_writer {
                let seed = start_seed + i;
                let record = json!({
                    "embedding": seeded_vector(32, seed),
                    "tag": format!("w{writer_id}_r{i}")
                });

                let t = Instant::now();
                let resp = c
                    .post(format!("{b}/v1/collections/stress_writes/insert"))
                    .json(&json!({ "records": [record] }))
                    .send()
                    .await;
                let elapsed = t.elapsed();

                match resp {
                    Ok(r) if r.status() == 201 => latencies.record(elapsed),
                    _ => {
                        errs.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
            latencies
        }));
    }

    let mut combined = LatencyStats::new();
    for h in handles {
        let stats = h.await.unwrap();
        for s in &stats.samples {
            combined.record(*s);
        }
    }
    let wall_elapsed = wall_start.elapsed();
    let total_errors = errors.load(Ordering::Relaxed);
    let total_records = num_writers * records_per_writer;
    let throughput = total_records as f64 / wall_elapsed.as_secs_f64();

    print_report(
        "Stress Test: Concurrent Writes",
        &[
            ("Writers:", format!("{num_writers}")),
            ("Records:", format!("{total_records}")),
            ("Duration:", format!("{:.2}s", wall_elapsed.as_secs_f64())),
            ("Throughput:", format!("{throughput:.0} records/sec")),
            ("Latency avg:", fmt_duration(combined.avg())),
            ("Latency p50:", fmt_duration(combined.p50())),
            ("Latency p95:", fmt_duration(combined.p95())),
            ("Latency p99:", fmt_duration(combined.p99())),
            ("Errors:", format!("{total_errors}")),
        ],
    );

    // Verify: search with high limit to count records
    let resp = client
        .post(format!("{base}/v1/collections/stress_writes/search"))
        .json(&json!({
            "query": {
                "vector": { "field": "embedding", "values": seeded_vector(32, 0) }
            },
            "limit": total_records + 100
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    let result_count = body["count"].as_u64().unwrap();

    println!("  Verified records via search: {result_count}");
    assert_eq!(
        result_count,
        (total_records as u64) - total_errors,
        "Expected {} records, found {result_count}",
        total_records as u64 - total_errors
    );
    assert_eq!(total_errors, 0, "Expected zero errors");
}

// ── 2. Concurrent reads stress test ─────────────────────────────────────────

#[tokio::test]
#[ignore]
async fn test_concurrent_reads() {
    let (base, _tmp) = start_test_server().await;
    let client = shared_client();

    create_collection(&client, &base, "stress_reads", 32).await;

    // Seed 500 records in bulk
    let records: Vec<Value> = (0..500)
        .map(|i| {
            json!({
                "embedding": seeded_vector(32, i),
                "tag": format!("seed_{i}")
            })
        })
        .collect();
    let resp = client
        .post(format!("{base}/v1/collections/stress_reads/insert"))
        .json(&json!({ "records": records }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 201);

    let num_readers: u32 = 20;
    let queries_per_reader: u32 = 50;
    let errors = Arc::new(AtomicU64::new(0));
    let wall_start = Instant::now();

    let mut handles = Vec::new();
    for reader_id in 0..num_readers {
        let c = client.clone();
        let b = base.clone();
        let errs = errors.clone();

        handles.push(tokio::spawn(async move {
            let mut latencies = LatencyStats::new();
            for q in 0..queries_per_reader {
                let seed = reader_id * queries_per_reader + q;
                let (elapsed, ok) = search(&c, &b, "stress_reads", 32, seed, 10).await;
                if ok {
                    latencies.record(elapsed);
                } else {
                    errs.fetch_add(1, Ordering::Relaxed);
                }
            }
            latencies
        }));
    }

    let mut combined = LatencyStats::new();
    for h in handles {
        let stats = h.await.unwrap();
        for s in &stats.samples {
            combined.record(*s);
        }
    }
    let wall_elapsed = wall_start.elapsed();
    let total_errors = errors.load(Ordering::Relaxed);
    let total_queries = num_readers * queries_per_reader;
    let qps = total_queries as f64 / wall_elapsed.as_secs_f64();

    print_report(
        "Stress Test: Concurrent Reads",
        &[
            ("Readers:", format!("{num_readers}")),
            ("Queries:", format!("{total_queries}")),
            ("Duration:", format!("{:.2}s", wall_elapsed.as_secs_f64())),
            ("QPS:", format!("{qps:.0} queries/sec")),
            ("Latency avg:", fmt_duration(combined.avg())),
            ("Latency p50:", fmt_duration(combined.p50())),
            ("Latency p95:", fmt_duration(combined.p95())),
            ("Latency p99:", fmt_duration(combined.p99())),
            ("Errors:", format!("{total_errors}")),
        ],
    );

    assert_eq!(total_errors, 0, "Expected zero search errors");
    assert_eq!(
        combined.count(),
        total_queries as usize,
        "All queries should have succeeded"
    );
}

// ── 3. Mixed read/write workload ────────────────────────────────────────────

#[tokio::test]
#[ignore]
async fn test_mixed_read_write() {
    let (base, _tmp) = start_test_server().await;
    let client = shared_client();

    create_collection(&client, &base, "stress_mixed", 32).await;

    // Seed 200 records
    let records: Vec<Value> = (0..200)
        .map(|i| {
            json!({
                "embedding": seeded_vector(32, i),
                "tag": format!("seed_{i}")
            })
        })
        .collect();
    let resp = client
        .post(format!("{base}/v1/collections/stress_mixed/insert"))
        .json(&json!({ "records": records }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 201);

    let num_writers: u32 = 5;
    let inserts_per_writer: u32 = 100;
    let num_readers: u32 = 10;
    let queries_per_reader: u32 = 50;

    let write_errors = Arc::new(AtomicU64::new(0));
    let read_errors = Arc::new(AtomicU64::new(0));
    let wall_start = Instant::now();

    // Spawn writers
    let mut writer_handles = Vec::new();
    for writer_id in 0..num_writers {
        let c = client.clone();
        let b = base.clone();
        let errs = write_errors.clone();

        writer_handles.push(tokio::spawn(async move {
            let mut latencies = LatencyStats::new();
            let base_seed = 1000 + writer_id * inserts_per_writer;
            for i in 0..inserts_per_writer {
                let seed = base_seed + i;
                let record = json!({
                    "embedding": seeded_vector(32, seed),
                    "tag": format!("w{writer_id}_r{i}")
                });
                let t = Instant::now();
                let resp = c
                    .post(format!("{b}/v1/collections/stress_mixed/insert"))
                    .json(&json!({ "records": [record] }))
                    .send()
                    .await;
                let elapsed = t.elapsed();
                match resp {
                    Ok(r) if r.status() == 201 => latencies.record(elapsed),
                    _ => {
                        errs.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
            latencies
        }));
    }

    // Spawn readers
    let mut reader_handles = Vec::new();
    for reader_id in 0..num_readers {
        let c = client.clone();
        let b = base.clone();
        let errs = read_errors.clone();

        reader_handles.push(tokio::spawn(async move {
            let mut latencies = LatencyStats::new();
            for q in 0..queries_per_reader {
                let seed = reader_id * queries_per_reader + q;
                let (elapsed, ok) = search(&c, &b, "stress_mixed", 32, seed, 10).await;
                if ok {
                    latencies.record(elapsed);
                } else {
                    errs.fetch_add(1, Ordering::Relaxed);
                }
            }
            latencies
        }));
    }

    // Collect write stats
    let mut write_latencies = LatencyStats::new();
    for h in writer_handles {
        let stats = h.await.unwrap();
        for s in &stats.samples {
            write_latencies.record(*s);
        }
    }

    // Collect read stats
    let mut read_latencies = LatencyStats::new();
    for h in reader_handles {
        let stats = h.await.unwrap();
        for s in &stats.samples {
            read_latencies.record(*s);
        }
    }

    let wall_elapsed = wall_start.elapsed();
    let total_write_errors = write_errors.load(Ordering::Relaxed);
    let total_read_errors = read_errors.load(Ordering::Relaxed);
    let total_ops = (num_writers * inserts_per_writer + num_readers * queries_per_reader) as f64;
    let ops_per_sec = total_ops / wall_elapsed.as_secs_f64();

    print_report(
        "Stress Test: Mixed Read/Write",
        &[
            ("Writers:", format!("{num_writers}")),
            ("Readers:", format!("{num_readers}")),
            ("Duration:", format!("{:.2}s", wall_elapsed.as_secs_f64())),
            ("Throughput:", format!("{ops_per_sec:.0} ops/sec")),
            ("", String::new()),
            ("-- Writes --", String::new()),
            (
                "  Total:",
                format!("{}", num_writers * inserts_per_writer),
            ),
            ("  Avg:", fmt_duration(write_latencies.avg())),
            ("  p50:", fmt_duration(write_latencies.p50())),
            ("  p95:", fmt_duration(write_latencies.p95())),
            ("  p99:", fmt_duration(write_latencies.p99())),
            ("  Errors:", format!("{total_write_errors}")),
            ("", String::new()),
            ("-- Reads --", String::new()),
            (
                "  Total:",
                format!("{}", num_readers * queries_per_reader),
            ),
            ("  Avg:", fmt_duration(read_latencies.avg())),
            ("  p50:", fmt_duration(read_latencies.p50())),
            ("  p95:", fmt_duration(read_latencies.p95())),
            ("  p99:", fmt_duration(read_latencies.p99())),
            ("  Errors:", format!("{total_read_errors}")),
        ],
    );

    assert_eq!(total_write_errors, 0, "Expected zero write errors");
    assert_eq!(total_read_errors, 0, "Expected zero read errors");
}

// ── 4. Large collection test ────────────────────────────────────────────────

#[tokio::test]
#[ignore]
async fn test_large_collection() {
    let (base, _tmp) = start_test_server().await;
    let client = shared_client();

    create_collection(&client, &base, "stress_large", 32).await;

    let total_records: u32 = 10_000;
    let batch_size: u32 = 100;
    let num_batches = total_records / batch_size;

    let mut insert_latencies = LatencyStats::new();
    let wall_start = Instant::now();

    for batch in 0..num_batches {
        let start_seed = batch * batch_size;
        let elapsed =
            insert_records(&client, &base, "stress_large", 32, start_seed, batch_size).await;
        insert_latencies.record(elapsed);
    }

    let insert_elapsed = wall_start.elapsed();
    let insert_throughput = total_records as f64 / insert_elapsed.as_secs_f64();

    // Search with k=10
    let (search_time, search_ok) = search(&client, &base, "stress_large", 32, 0, 10).await;
    assert!(search_ok, "Search with k=10 should succeed");

    // Verify result count
    let resp = client
        .post(format!("{base}/v1/collections/stress_large/search"))
        .json(&json!({
            "query": {
                "vector": { "field": "embedding", "values": seeded_vector(32, 0) }
            },
            "limit": 10
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    let results = body["results"].as_array().unwrap();
    assert_eq!(results.len(), 10, "Expected 10 results for k=10 search");

    // Search with filter
    let t = Instant::now();
    let resp = client
        .post(format!("{base}/v1/collections/stress_large/search"))
        .json(&json!({
            "query": {
                "vector": { "field": "embedding", "values": seeded_vector(32, 42) }
            },
            "filter": "tag = 'item_42'",
            "limit": 10
        }))
        .send()
        .await
        .unwrap();
    let filter_search_time = t.elapsed();
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    let filter_count = body["count"].as_u64().unwrap();

    print_report(
        "Stress Test: Large Collection",
        &[
            ("Records:", format!("{total_records}")),
            ("Batch size:", format!("{batch_size}")),
            (
                "Insert time:",
                format!("{:.2}s", insert_elapsed.as_secs_f64()),
            ),
            (
                "Insert rate:",
                format!("{insert_throughput:.0} records/sec"),
            ),
            (
                "Batch avg:",
                fmt_duration(insert_latencies.avg()),
            ),
            (
                "Batch p95:",
                fmt_duration(insert_latencies.p95()),
            ),
            ("", String::new()),
            ("Search k=10:", fmt_duration(search_time)),
            (
                "Filtered:",
                format!(
                    "{} ({} results)",
                    fmt_duration(filter_search_time),
                    filter_count
                ),
            ),
        ],
    );
}

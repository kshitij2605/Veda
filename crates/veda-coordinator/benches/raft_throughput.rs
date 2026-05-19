//! Throughput / latency micro-benchmarks for the Raft path.
//!
//! Two scenarios:
//!   1. Single-node openraft with in-memory storage — establishes
//!      the baseline "best case" client_write latency (no network,
//!      no disk).
//!   2. Single-node openraft with durable WAL storage — measures
//!      the cost of fsync per append.
//!
//! Not Criterion — Criterion's setup hooks don't play well with
//! the async Raft bootstrap, and we'd be running benchmarks on
//! a tokio runtime anyway. Hand-rolled timing loop instead;
//! prints throughput + p50/p95 to stderr in the format
//! `name: ops/s p50_us p95_us` so a later grep across runs can
//! plot regression.
//!
//! Run as:
//!   cargo bench -p veda-coordinator --bench raft_throughput
//!
//! Tune via env:
//!   VEDA_BENCH_OPS  number of client_writes per scenario (default 1000)

use std::time::{Duration, Instant};

use veda_coordinator::raft_coordinator::RaftCoordinator;
use veda_coordinator::raft_types::{ClusterCommand, ClusterResponse};

fn percentile(samples: &mut Vec<Duration>, pct: f64) -> Duration {
    if samples.is_empty() {
        return Duration::ZERO;
    }
    samples.sort();
    let idx = ((samples.len() as f64 - 1.0) * pct).round() as usize;
    samples[idx.min(samples.len() - 1)]
}

async fn run_scenario(name: &str, coord: &RaftCoordinator, ops: usize) {
    // Wait briefly for the leader to settle.
    tokio::time::sleep(Duration::from_millis(50)).await;

    let mut latencies: Vec<Duration> = Vec::with_capacity(ops);
    let started = Instant::now();
    for _ in 0..ops {
        let t = Instant::now();
        let resp = coord
            .execute(ClusterCommand::Noop)
            .await
            .expect("Noop client_write should succeed");
        latencies.push(t.elapsed());
        debug_assert_eq!(resp, ClusterResponse::Ok);
    }
    let elapsed = started.elapsed();
    let p50 = percentile(&mut latencies, 0.50);
    let p95 = percentile(&mut latencies, 0.95);
    let p99 = percentile(&mut latencies, 0.99);
    let ops_per_sec = ops as f64 / elapsed.as_secs_f64();
    eprintln!(
        "{name}: ops={ops} elapsed={:.2}s rate={:.0} ops/s p50={}us p95={}us p99={}us",
        elapsed.as_secs_f64(),
        ops_per_sec,
        p50.as_micros(),
        p95.as_micros(),
        p99.as_micros(),
    );
    coord.shutdown().await.ok();
}

fn main() {
    let ops: usize = std::env::var("VEDA_BENCH_OPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1000);

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("build tokio runtime");

    rt.block_on(async {
        eprintln!("# Veda Raft throughput benchmark (ops={ops})");
        eprintln!("# scenario: ops elapsed rate p50 p95 p99");

        // Scenario A: in-memory only.
        let coord_mem = RaftCoordinator::start_single_node(1)
            .await
            .expect("bootstrap mem");
        run_scenario("single_node_inmem", &coord_mem, ops).await;

        // Scenario B: durable WAL + state-machine snapshot.
        let tmp = tempfile::tempdir().expect("tempdir");
        let coord_disk = RaftCoordinator::start_single_node_with_dir(2, tmp.path())
            .await
            .expect("bootstrap disk");
        run_scenario("single_node_durable", &coord_disk, ops).await;

        eprintln!("# done");
    });
}

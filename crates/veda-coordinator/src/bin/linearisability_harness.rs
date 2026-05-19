//! Linearisability harness: drives concurrent inserts + reads
//! against the 3-node Raft cluster, records every invocation,
//! then runs the `check_history` analyser.
//!
//! Usage:
//!   docker compose --profile raft-cluster up -d veda-1 veda-2 veda-3
//!   cargo run --release --bin veda-linearisability-harness
//!
//! Env knobs:
//!   VEDA_LIN_NODES        comma-separated node base URLs
//!                         (default: http://localhost:6340,
//!                          http://localhost:6342,http://localhost:6344)
//!   VEDA_LIN_DURATION_SECS run time (default 60)
//!   VEDA_LIN_CLIENTS       concurrent client tasks (default 4)
//!   VEDA_LIN_COLLECTION    collection name (default lin_smoke)
//!   VEDA_LIN_HISTORY_OUT   where to dump the JSONL history
//!                         (default ./linearisability-history.jsonl)
//!
//! The harness:
//!   1. Ensures the collection exists (create if missing).
//!   2. Spawns N client tasks that pick a random node, then
//!      either insert a new record or read one of the existing
//!      ids. Each op is timed (start/end_ns) and recorded.
//!   3. After DURATION_SECS, stops, dumps history to JSONL,
//!      runs check_history, prints the report.
//!
//! Exit code 0 iff the analyser found zero violations.

use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::Mutex;
use veda_coordinator::linearisability_checker::{
    check_history, History, OpKind, OpOutcome,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let nodes: Vec<String> = std::env::var("VEDA_LIN_NODES")
        .unwrap_or_else(|_| {
            "http://localhost:6340,http://localhost:6342,http://localhost:6344".into()
        })
        .split(',')
        .map(|s| s.trim().to_owned())
        .filter(|s| !s.is_empty())
        .collect();
    let duration_secs: u64 = std::env::var("VEDA_LIN_DURATION_SECS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(60);
    let clients: usize = std::env::var("VEDA_LIN_CLIENTS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);
    let collection = std::env::var("VEDA_LIN_COLLECTION")
        .unwrap_or_else(|_| "lin_smoke".into());
    let history_out = std::env::var("VEDA_LIN_HISTORY_OUT")
        .unwrap_or_else(|_| "linearisability-history.jsonl".into());

    eprintln!(
        "[lin] driving cluster: {nodes:?} duration={duration_secs}s clients={clients}"
    );

    // Use the leader's node to create the collection if needed.
    // For simplicity just try every node; whichever forwards to
    // the leader correctly will succeed.
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .no_proxy()
        .build()?;
    ensure_collection(&client, &nodes, &collection).await?;

    let history = History::new();
    let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let known_ids: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));

    let mut tasks = Vec::new();
    for client_id in 0..clients {
        let client = client.clone();
        let nodes = nodes.clone();
        let history = history.clone();
        let stop = stop.clone();
        let known_ids = known_ids.clone();
        let collection = collection.clone();
        tasks.push(tokio::spawn(async move {
            let mut rng_state: u64 = (client_id as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
            while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let node_idx = (rng_state >> 33) as usize % nodes.len();
                let node = &nodes[node_idx];
                let do_read = {
                    let known = known_ids.lock().await;
                    !known.is_empty() && (rng_state & 1) == 0
                };
                if do_read {
                    let id = {
                        let known = known_ids.lock().await;
                        let n = known.len();
                        known[(rng_state >> 1) as usize % n].clone()
                    };
                    let started = Instant::now();
                    let outcome = read_record(&client, node, &collection, &id).await;
                    history
                        .record(
                            client_id,
                            OpKind::Read {
                                collection: collection.clone(),
                                record_id: id,
                            },
                            outcome,
                            started,
                        )
                        .await;
                } else {
                    // Insert a fresh record with a deterministic payload.
                    let payload = format!("client{}-seq{}", client_id, rng_state);
                    let started = Instant::now();
                    let (id, outcome) =
                        insert_record(&client, node, &collection, &payload).await;
                    if let Some(real_id) = &id {
                        known_ids.lock().await.push(real_id.clone());
                    }
                    history
                        .record(
                            client_id,
                            OpKind::Insert {
                                collection: collection.clone(),
                                record_id: id.unwrap_or_default(),
                                payload,
                            },
                            outcome,
                            started,
                        )
                        .await;
                }
            }
        }));
    }

    tokio::time::sleep(Duration::from_secs(duration_secs)).await;
    stop.store(true, std::sync::atomic::Ordering::Relaxed);
    for t in tasks {
        let _ = t.await;
    }

    let inv = history.snapshot().await;
    eprintln!("[lin] history has {} invocations", inv.len());
    history.dump_jsonl(std::path::Path::new(&history_out)).await?;
    eprintln!("[lin] history dumped to {history_out}");

    let report = check_history(&inv);
    eprintln!("[lin] report:");
    eprintln!("  total: {}", report.total_invocations);
    eprintln!("  successful inserts: {}", report.successful_inserts);
    eprintln!("  successful reads: {}", report.successful_reads);
    eprintln!("  failed: {}", report.failed_invocations);
    eprintln!("  violations: {}", report.violations.len());
    for v in &report.violations {
        eprintln!("    [{}] seq={} {}", v.kind, v.seq, v.detail);
    }
    if report.violations.is_empty() {
        eprintln!("[lin] PASS");
        Ok(())
    } else {
        eprintln!("[lin] FAIL ({} violation(s))", report.violations.len());
        std::process::exit(1);
    }
}

async fn ensure_collection(
    client: &reqwest::Client,
    nodes: &[String],
    collection: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let body = serde_json::json!({
        "name": collection,
        "fields": [
            { "name": "payload", "type": "string" },
            { "name": "embedding", "type": "vector", "dimension": 4, "metric": "cosine" }
        ],
        "shard_count": 1,
        "replication_factor": 1,
    });
    for node in nodes {
        let url = format!("{}/v1/collections", node);
        let resp = client.post(&url).json(&body).send().await?;
        let s = resp.status();
        if s.is_success() || s.as_u16() == 409 {
            // 409 = already exists
            return Ok(());
        }
    }
    Err("failed to ensure collection".into())
}

async fn insert_record(
    client: &reqwest::Client,
    node: &str,
    collection: &str,
    payload: &str,
) -> (Option<String>, OpOutcome) {
    let body = serde_json::json!({
        "records": [{
            "payload": payload,
            "embedding": [0.1, 0.2, 0.3, 0.4]
        }]
    });
    let url = format!("{}/v1/collections/{}/insert", node, collection);
    match client.post(&url).json(&body).send().await {
        Ok(resp) => {
            let status = resp.status();
            if !status.is_success() {
                return (None, OpOutcome::Err(format!("HTTP {}", status)));
            }
            match resp.json::<serde_json::Value>().await {
                Ok(v) => {
                    // Raft-routed insert returns the assigned UUID
                    // in `ids[0]`. Non-Raft path leaves `ids`
                    // empty — fall back to None.
                    let id = v
                        .get("ids")
                        .and_then(|a| a.as_array())
                        .and_then(|arr| arr.first())
                        .and_then(|s| s.as_str())
                        .map(|s| s.to_owned());
                    (id, OpOutcome::Ok(None))
                }
                Err(e) => (None, OpOutcome::Err(format!("parse: {e}"))),
            }
        }
        Err(e) => (None, OpOutcome::Err(format!("post: {e}"))),
    }
}

async fn read_record(
    client: &reqwest::Client,
    node: &str,
    collection: &str,
    id: &str,
) -> OpOutcome {
    let url = format!(
        "{}/v1/collections/{}/records/{}",
        node, collection, id
    );
    match client.get(&url).send().await {
        Ok(resp) => {
            let s = resp.status();
            if s.as_u16() == 404 {
                OpOutcome::Ok(None)
            } else if s.is_success() {
                match resp.json::<serde_json::Value>().await {
                    Ok(v) => {
                        let payload = v
                            .get("fields")
                            .and_then(|f| f.get("payload"))
                            .and_then(|s| s.as_str())
                            .map(|s| s.to_owned());
                        OpOutcome::Ok(payload)
                    }
                    Err(e) => OpOutcome::Err(format!("parse: {e}")),
                }
            } else {
                OpOutcome::Err(format!("HTTP {}", s))
            }
        }
        Err(e) => OpOutcome::Err(format!("get: {e}")),
    }
}

//! In-process gRPC round-trip for the multi-node distributed search
//! transport. Starts a `ShardSearchService` on an ephemeral localhost
//! port, then drives it with the `RemoteShardSearcher` client + the
//! `fetch_records_remote` fetch helper. Proves the wire path end-to-end
//! without needing multiple machines.
//!
//! The live multi-node validation is a separate manual follow-up
//! documented in `documentation/multi-node-test-plan.md`.

use std::sync::Arc;
use std::time::Duration;

use uuid::Uuid;
use veda_core::record::{FieldValue, Record};
use veda_core::schema::{CollectionSchema, FieldDefinition};
use veda_core::types::{DistanceMetric, FieldType};
use veda_core::vector::DenseVector;
use veda_query::distributed::{DistributedQueryExecutor, ShardSearcher};
use veda_query::local_shard::LocalShardSearcher;
use veda_query::remote_shard::{
    RemoteShardSearcher, ShardSearchServer, ShardSearchService, fetch_records_remote,
};
use veda_query::search::VectorSearchRequest;
use veda_storage::segment_manager::SegmentManager;

const DIM: u32 = 16;

fn test_schema() -> CollectionSchema {
    CollectionSchema::builder("remote_coll")
        .add_field(FieldDefinition::vector(
            "embedding",
            DIM,
            DistanceMetric::Euclidean,
        ))
        .add_field(FieldDefinition::new("category", FieldType::String).with_filterable(true))
        .build()
        .unwrap()
}

/// Deterministic vector so local and remote searches operate on
/// identical data (no RNG drift between the two searchers).
fn seeded_vec(seed: u32) -> Vec<f32> {
    (0..DIM)
        .map(|i| ((seed.wrapping_mul(31).wrapping_add(i)) % 97) as f32 / 97.0)
        .collect()
}

fn build_segment_manager(tmp: &std::path::Path, n: u32) -> (SegmentManager, Vec<Uuid>) {
    let schema = test_schema();
    let mut sm = SegmentManager::new(schema, 0, tmp, 1024 * 1024 * 1024).unwrap();
    let mut ids = Vec::new();
    for i in 0..n {
        // Deterministic id so the server fixture and the oracle fixture
        // hold byte-identical records.
        let id = Uuid::from_u128(0x5eed_0000_0000_0000_0000_0000_0000_0000 | i as u128);
        let mut record = Record::with_id(id);
        record.set(
            "embedding",
            FieldValue::Vector(DenseVector::new(seeded_vec(i))),
        );
        record.set("category", FieldValue::String(format!("cat{}", i % 3)));
        sm.insert(record).unwrap();
        ids.push(id);
    }
    (sm, ids)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn remote_shard_search_and_fetch_round_trip() {
    // ── Build identical local data twice: one SegmentManager backs the
    //    gRPC server; an identical one backs a LocalShardSearcher we use
    //    as the ground-truth oracle. ──────────────────────────────────
    let tmp_server = tempfile::tempdir().unwrap();
    let tmp_oracle = tempfile::tempdir().unwrap();
    let (sm_server, ids) = build_segment_manager(tmp_server.path(), 40);
    let (sm_oracle, ids_oracle) = build_segment_manager(tmp_oracle.path(), 40);
    assert_eq!(
        ids, ids_oracle,
        "fixtures must be identical (deterministic ids)"
    );

    let oracle = LocalShardSearcher::new(sm_oracle);

    // Host the server-side searcher.
    let hosted = Arc::new(LocalShardSearcher::new(sm_server));
    let mut svc = ShardSearchService::new();
    svc.register("remote_coll", 0, hosted);

    // Bind an ephemeral port and spawn the server.
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let incoming = tokio_stream::wrappers::TcpListenerStream::new(listener);
    let server = tonic::transport::Server::builder()
        .add_service(ShardSearchServer::new(svc))
        .serve_with_incoming(incoming);
    let _handle = tokio::spawn(server);
    tokio::time::sleep(Duration::from_millis(50)).await;

    // ── Client: RemoteShardSearcher ──────────────────────────────────
    let remote = RemoteShardSearcher::connect("remote_coll", 0, format!("http://{addr}"))
        .await
        .unwrap();

    let request = VectorSearchRequest {
        field: "embedding".into(),
        query_vector: seeded_vec(7),
        k: 10,
        metric: DistanceMetric::Euclidean,
        filter: None,
        truncate_to: None,
        read_ts: None,
        filter_expr: None,
        ef_search: None,
    };

    // search_shard blocks on the runtime internally; run it off the
    // async worker via spawn_blocking so we don't block the reactor.
    let req_for_remote = request.clone();
    let (remote_results, _t) =
        tokio::task::spawn_blocking(move || remote.search_shard(0, &req_for_remote))
            .await
            .unwrap()
            .unwrap();

    // Ground truth from the in-process oracle.
    let (oracle_results, _t2) = oracle.search_shard(0, &request).unwrap();

    // (a) Lightweight Phase-1 payload matches the local result exactly.
    assert_eq!(remote_results.len(), oracle_results.len());
    assert_eq!(remote_results.len(), 10);
    for (r, o) in remote_results.iter().zip(oracle_results.iter()) {
        assert_eq!(r.id, o.id, "remote id mismatch");
        assert_eq!(r.doc_id, o.doc_id, "remote doc_id mismatch");
        // NOTE: segment_id is a per-SegmentManager-instance random UUID,
        // so it legitimately differs between the server fixture and the
        // independent oracle fixture. The id/doc_id equality above already
        // exercises the 16-byte UUID wire round-trip. We additionally
        // assert the server's segment_id survived the wire intact (16-byte
        // non-nil UUID) rather than equality with the oracle.
        assert_ne!(r.segment_id, Uuid::nil(), "segment_id must round-trip");
        assert!(
            (r.distance - o.distance).abs() < 1e-5,
            "distance mismatch: {} vs {}",
            r.distance,
            o.distance
        );
    }
    // Ascending by distance.
    for w in remote_results.windows(2) {
        assert!(w[0].distance <= w[1].distance);
    }

    // (b) FetchRecords for the top-k ids returns the correct full records.
    let top_ids: Vec<Uuid> = remote_results.iter().map(|r| r.id).collect();
    let channel = tonic::transport::Endpoint::from_shared(format!("http://{addr}"))
        .unwrap()
        .connect()
        .await
        .unwrap();
    let fetched = fetch_records_remote(channel, "remote_coll", &top_ids)
        .await
        .unwrap();
    assert_eq!(fetched.len(), top_ids.len(), "all top-k ids should resolve");

    // Each fetched record carries the full embedding vector + category —
    // proving full records (not just ids) come back over FetchRecords.
    for rec in &fetched {
        match rec.get("embedding") {
            Some(FieldValue::Vector(v)) => assert_eq!(v.as_slice().len(), DIM as usize),
            other => panic!("expected embedding vector, got {other:?}"),
        }
        assert!(matches!(rec.get("category"), Some(FieldValue::String(_))));
    }

    // Fetch an unknown id → silently skipped (not an error).
    let channel2 = tonic::transport::Endpoint::from_shared(format!("http://{addr}"))
        .unwrap()
        .connect()
        .await
        .unwrap();
    let none = fetch_records_remote(channel2, "remote_coll", &[Uuid::new_v4()])
        .await
        .unwrap();
    assert!(none.is_empty(), "unknown id must not resolve");
}

/// The `DistributedQueryExecutor` merges a MIX of local + remote
/// searchers and returns a correct global top-k.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn distributed_executor_mixes_local_and_remote() {
    // Remote shard 1 (served over gRPC).
    let tmp_remote = tempfile::tempdir().unwrap();
    let (sm_remote, _) = build_segment_manager(tmp_remote.path(), 30);
    let hosted = Arc::new(LocalShardSearcher::new(sm_remote));
    let mut svc = ShardSearchService::new();
    svc.register("remote_coll", 1, hosted);

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let incoming = tokio_stream::wrappers::TcpListenerStream::new(listener);
    let server = tonic::transport::Server::builder()
        .add_service(ShardSearchServer::new(svc))
        .serve_with_incoming(incoming);
    let _handle = tokio::spawn(server);
    tokio::time::sleep(Duration::from_millis(50)).await;

    let remote = RemoteShardSearcher::connect("remote_coll", 1, format!("http://{addr}"))
        .await
        .unwrap();

    // Local shard 0 (in-process).
    let tmp_local = tempfile::tempdir().unwrap();
    let (sm_local, _) = build_segment_manager(tmp_local.path(), 30);
    let local = LocalShardSearcher::new(sm_local);

    let request = VectorSearchRequest {
        field: "embedding".into(),
        query_vector: seeded_vec(3),
        k: 10,
        metric: DistanceMetric::Euclidean,
        filter: None,
        truncate_to: None,
        read_ts: None,
        filter_expr: None,
        ef_search: None,
    };

    // The executor fans out synchronously; the remote searcher bridges to
    // its runtime internally. Drive on a blocking thread.
    let (response, _t) = tokio::task::spawn_blocking(move || {
        let mut executor = DistributedQueryExecutor::new();
        executor.register_shard(0, Box::new(local));
        executor.register_shard(1, Box::new(remote));
        executor.search(&request)
    })
    .await
    .unwrap()
    .unwrap();

    assert_eq!(response.results.len(), 10, "merged global top-k");
    // total_candidates = 10 (local) + 10 (remote).
    assert_eq!(response.total_candidates, 20);
    for w in response.results.windows(2) {
        assert!(
            w[0].distance <= w[1].distance,
            "global merge must be sorted"
        );
    }
}

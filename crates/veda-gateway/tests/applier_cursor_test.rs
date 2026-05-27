//! #2 - durable applier cursor: resume-after-seal + replay-unsealed.

use bytes::Bytes;
use veda_core::record::{FieldValue, Record};
use veda_core::schema::{CollectionSchema, FieldDefinition};
use veda_core::types::DistanceMetric;
use veda_core::vector::DenseVector;
use veda_log::format::LogOp;
use veda_time::Hlc;

use veda_gateway::log_appliers::ShardApplier;
use veda_gateway::rest::state::AppState;

fn schema(name: &str) -> CollectionSchema {
    CollectionSchema::builder(name)
        .add_field(FieldDefinition::vector("v", 2, DistanceMetric::Cosine))
        .shard_count(1)
        .build()
        .unwrap()
}

fn record() -> Record {
    let mut r = Record::new();
    r.set("v", FieldValue::Vector(DenseVector::new(vec![1.0, 0.0])));
    r
}

async fn register_sm(state: &AppState, collection: &str, seal_bytes: usize) {
    let sm = veda_storage::segment_manager::SegmentManager::new(
        schema(collection),
        0,
        &state.inner.data_dir.join(collection),
        seal_bytes,
    )
    .unwrap();
    let mut managers = state.inner.segment_managers.write().await;
    managers.entry(collection.to_owned()).or_default().insert(0, sm);
}

async fn append_n(state: &AppState, collection: &str, n: u64) {
    let log = state.inner.shard_logs.get_or_open(collection, 0).unwrap();
    for i in 1..=n {
        let r = record();
        let payload = Bytes::from(serde_json::to_vec(&r).unwrap());
        let key = Bytes::copy_from_slice(r.id().as_bytes());
        log.append_one(Hlc::new(i, 0), 1, key, LogOp::Put, payload).unwrap();
    }
}

#[tokio::test]
async fn resumes_from_persisted_cursor_after_seal() {
    let tmp = tempfile::tempdir().unwrap();
    let state = AppState::new(tmp.path());
    register_sm(&state, "coll", 256).await; // tiny -> seals quickly
    append_n(&state, "coll", 20).await;

    let mut applier = ShardApplier::open(&state, "coll", 0).await.unwrap();
    for _ in 0..20 {
        applier.tick(&state, "coll", 0).await.unwrap();
    }

    let cursor_path = state.inner.shard_logs.shard_dir("coll", 0).join("applied.cursor");
    let persisted = veda_log::cursor::read_cursor(&cursor_path);
    assert!(persisted > 0, "a seal should have advanced the durable cursor");

    let applier2 = ShardApplier::open(&state, "coll", 0).await.unwrap();
    assert_eq!(applier2.next_seq(), persisted + 1, "resume from cursor+1, not 1");
}

#[tokio::test]
async fn replays_unsealed_entries_after_crash() {
    let tmp = tempfile::tempdir().unwrap();
    let state = AppState::new(tmp.path());
    register_sm(&state, "coll", 64 * 1024 * 1024).await; // huge -> never seals
    append_n(&state, "coll", 5).await;

    let mut applier = ShardApplier::open(&state, "coll", 0).await.unwrap();
    let applied = applier.tick(&state, "coll", 0).await.unwrap();
    assert_eq!(applied, 5);

    let cursor_path = state.inner.shard_logs.shard_dir("coll", 0).join("applied.cursor");
    assert_eq!(veda_log::cursor::read_cursor(&cursor_path), 0, "no seal -> cursor stays 0");

    let applier2 = ShardApplier::open(&state, "coll", 0).await.unwrap();
    assert_eq!(applier2.next_seq(), 1, "fresh applier replays unsealed entries from 1");
}

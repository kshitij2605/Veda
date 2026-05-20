//! Phase G snapshot-isolation anomaly tests (the plan's
//! `snapshot_isolation.rs`). Validates the three Adya invariants
//! Percolator SI claims:
//!
//! - **Write-skew allowed** — SI explicitly permits write-skew
//!   (Adya G2-item). Two txns can read overlapping sets and
//!   write disjoint members; both commit.
//! - **Lost-update prevented** — Adya P4. Two txns reading and
//!   modifying the same key must not both commit; the second
//!   gets a write-write conflict.
//! - **Read-only consistent** — Adya G1c (broken predicate
//!   reads). A read-only txn at snapshot start_ts sees a
//!   consistent view: never half-applied writes, never effects
//!   of txns that started later than the snapshot.

use std::sync::Arc;

use bytes::Bytes;
use veda_coordinator::percolator::{Mutation, ReadVisibility, TxnCoordinator, TxnError};
use veda_time::LocalTso;

fn cc() -> Arc<TxnCoordinator> {
    Arc::new(TxnCoordinator::new(Arc::new(LocalTso::new())))
}

#[tokio::test]
async fn write_skew_is_allowed_by_snapshot_isolation() {
    // Classic write-skew: two doctors on call, txn A demotes
    // doctor #1 to off-call iff doctor #2 is still on-call; txn
    // B does the inverse. Under SI both can commit (each saw
    // the other was on-call); the system ends up with both
    // off-call. Serialisable would prevent this; SI doesn't.
    let cc = cc();

    // Seed: both doctors on-call.
    {
        let mut seed = cc.begin().await.unwrap();
        seed.buffer(Mutation {
            key: b"doc1".to_vec(),
            value: Some(Bytes::from_static(b"on")),
        });
        seed.buffer(Mutation {
            key: b"doc2".to_vec(),
            value: Some(Bytes::from_static(b"on")),
        });
        cc.prewrite(&seed).await.unwrap();
        cc.commit(&seed).await.unwrap();
    }

    let mut a = cc.begin().await.unwrap();
    let mut b = cc.begin().await.unwrap();

    // Both txns read both doctors' state.
    let a_doc1 = cc.read(&a, b"doc1").await.unwrap();
    let a_doc2 = cc.read(&a, b"doc2").await.unwrap();
    let b_doc1 = cc.read(&b, b"doc1").await.unwrap();
    let b_doc2 = cc.read(&b, b"doc2").await.unwrap();
    assert!(matches!(a_doc1, ReadVisibility::Visible(Some(_))));
    assert!(matches!(a_doc2, ReadVisibility::Visible(Some(_))));
    assert!(matches!(b_doc1, ReadVisibility::Visible(Some(_))));
    assert!(matches!(b_doc2, ReadVisibility::Visible(Some(_))));

    // A flips doc1 off; B flips doc2 off — disjoint write sets.
    a.buffer(Mutation {
        key: b"doc1".to_vec(),
        value: Some(Bytes::from_static(b"off")),
    });
    b.buffer(Mutation {
        key: b"doc2".to_vec(),
        value: Some(Bytes::from_static(b"off")),
    });

    // SI: both commit successfully. Serialisability would fail
    // one of them; SI accepts this anomaly by design.
    cc.prewrite(&a).await.expect("write-skew: A prewrite ok");
    cc.prewrite(&b).await.expect("write-skew: B prewrite ok");
    cc.commit(&a).await.expect("write-skew: A commit ok");
    cc.commit(&b).await.expect("write-skew: B commit ok");

    // Final state: both off. SI explicitly allows this.
    let after = cc.begin().await.unwrap();
    match cc.read(&after, b"doc1").await.unwrap() {
        ReadVisibility::Visible(Some(v)) => assert_eq!(&v[..], b"off"),
        other => panic!("doc1 not off: {other:?}"),
    }
    match cc.read(&after, b"doc2").await.unwrap() {
        ReadVisibility::Visible(Some(v)) => assert_eq!(&v[..], b"off"),
        other => panic!("doc2 not off: {other:?}"),
    }
}

#[tokio::test]
async fn lost_update_is_prevented_by_write_write_conflict() {
    // Two txns both read counter, both write counter+1. Under
    // SI the second prewrite must detect that a newer
    // commit_ts exists since its start_ts and abort with
    // WriteWriteConflict. (Lost-update = both succeed and
    // increment goes from 1 to 2 instead of to 3.)
    let cc = cc();

    {
        let mut seed = cc.begin().await.unwrap();
        seed.buffer(Mutation {
            key: b"counter".to_vec(),
            value: Some(Bytes::from_static(b"1")),
        });
        cc.prewrite(&seed).await.unwrap();
        cc.commit(&seed).await.unwrap();
    }

    let mut a = cc.begin().await.unwrap();
    let mut b = cc.begin().await.unwrap();

    // Both read.
    let _ = cc.read(&a, b"counter").await.unwrap();
    let _ = cc.read(&b, b"counter").await.unwrap();

    a.buffer(Mutation {
        key: b"counter".to_vec(),
        value: Some(Bytes::from_static(b"2")),
    });
    b.buffer(Mutation {
        key: b"counter".to_vec(),
        value: Some(Bytes::from_static(b"2")),
    });

    // A commits first.
    cc.prewrite(&a).await.expect("A prewrite ok");
    cc.commit(&a).await.expect("A commit ok");

    // B's prewrite must detect the write-write conflict —
    // there's a newer committed version (A's) than B's
    // start_ts.
    let b_result = cc.prewrite(&b).await;
    match b_result {
        Err(TxnError::WriteWriteConflict { .. }) => {}
        other => panic!("expected WriteWriteConflict on B, got {other:?}"),
    }
    cc.abort(&b).await;
}

#[tokio::test]
async fn read_only_snapshot_is_consistent_across_concurrent_writes() {
    // Snapshot-isolation reader at T0 must NOT see a later
    // writer's effects even if the writer commits while the
    // reader is in flight. We open a read txn, then write +
    // commit a new value, then read again from the original
    // read txn. The read txn must still see the OLD value
    // because its snapshot is fixed at start_ts.
    let cc = cc();

    {
        let mut seed = cc.begin().await.unwrap();
        seed.buffer(Mutation {
            key: b"k".to_vec(),
            value: Some(Bytes::from_static(b"v1")),
        });
        cc.prewrite(&seed).await.unwrap();
        cc.commit(&seed).await.unwrap();
    }

    // Open the reader BEFORE the writer commits.
    let reader = cc.begin().await.unwrap();
    // Force time advance so a subsequent begin gets a strictly
    // newer start_ts.
    tokio::time::sleep(std::time::Duration::from_millis(1)).await;

    // Write v2.
    {
        let mut w = cc.begin().await.unwrap();
        w.buffer(Mutation {
            key: b"k".to_vec(),
            value: Some(Bytes::from_static(b"v2")),
        });
        cc.prewrite(&w).await.unwrap();
        cc.commit(&w).await.unwrap();
    }

    // The reader at the OLDER start_ts must still see v1.
    match cc.read(&reader, b"k").await.unwrap() {
        ReadVisibility::Visible(Some(v)) => {
            assert_eq!(&v[..], b"v1", "reader at older snapshot should see v1, not v2");
        }
        other => panic!("reader saw unexpected visibility: {other:?}"),
    }
}

#[tokio::test]
async fn abort_releases_locks_so_other_txns_can_proceed() {
    let cc = cc();

    let mut a = cc.begin().await.unwrap();
    a.buffer(Mutation {
        key: b"abort_key".to_vec(),
        value: Some(Bytes::from_static(b"a_value")),
    });
    cc.prewrite(&a).await.unwrap();
    cc.abort(&a).await;

    // Now a different txn can prewrite the same key.
    let mut b = cc.begin().await.unwrap();
    b.buffer(Mutation {
        key: b"abort_key".to_vec(),
        value: Some(Bytes::from_static(b"b_value")),
    });
    cc.prewrite(&b).await.expect("B prewrite should succeed after A aborts");
    cc.commit(&b).await.expect("B commit should succeed");
}

//! Phase F isolation + GC-safety linearisability tests against
//! the new MVCC + log architecture. Mirrors the plan's
//! `mvcc_isolation.rs` and `gc_safety.rs`.
//!
//! Two correctness invariants Veda's MVCC must hold:
//!
//! - **Snapshot isolation**: a reader at `read_ts = T` sees no
//!   write whose `commit_ts > T`, even if that write is
//!   disk-applied. Phrased linearisability-style: for every
//!   `read_at(k, T)`, the result reflects the unique most-
//!   recent committed write `w(k, v, ts)` with `ts ≤ T` (or
//!   `None` if no such write exists or it's a tombstone).
//! - **GC safety**: `gc()` never drops a version that any live
//!   read at `read_ts ≤ safe_ts` could observe. Concretely, the
//!   "floor" version — the newest version with `commit_ts ≤
//!   safe_ts` — is always kept, even if it's a tombstone (a
//!   delete-then-write history must not surface the older
//!   value after GC).

use bytes::Bytes;
use veda_storage::mvcc::MvccStore;

#[test]
fn snapshot_isolation_reader_never_sees_future_write() {
    let mut store = MvccStore::new();
    store
        .put_payload(b"k", 100, Bytes::from_static(b"v1"))
        .unwrap();
    // Snapshot at ts = 50 — earlier than v1's commit.
    assert!(store.get_at(b"k", 50).is_none());
    // Snapshot at ts = 99 — still earlier than commit 100.
    assert!(store.get_at(b"k", 99).is_none());
    // Snapshot at ts = 100 — exactly commit_ts.
    assert_eq!(store.get_at(b"k", 100).as_deref(), Some(&b"v1"[..]));
    // Snapshot at ts = 200 — well past.
    assert_eq!(store.get_at(b"k", 200).as_deref(), Some(&b"v1"[..]));

    // Future write at ts=300.
    store
        .put_payload(b"k", 300, Bytes::from_static(b"v2"))
        .unwrap();
    // Reader at 200 must still see v1, not v2.
    assert_eq!(
        store.get_at(b"k", 200).as_deref(),
        Some(&b"v1"[..]),
        "reader at past ts must not observe future write"
    );
    // Reader at 300 sees v2.
    assert_eq!(store.get_at(b"k", 300).as_deref(), Some(&b"v2"[..]));
}

#[test]
fn snapshot_isolation_holds_across_interleaved_writes() {
    // Interleaved writes on different keys must respect the
    // per-key snapshot order. (Cross-key SI is the same as
    // per-key SI under MVCC.)
    let mut store = MvccStore::new();
    store
        .put_payload(b"a", 100, Bytes::from_static(b"a1"))
        .unwrap();
    store
        .put_payload(b"b", 200, Bytes::from_static(b"b1"))
        .unwrap();
    store
        .put_payload(b"a", 300, Bytes::from_static(b"a2"))
        .unwrap();

    // Snapshot at ts=150 sees a1, no b.
    assert_eq!(store.get_at(b"a", 150).as_deref(), Some(&b"a1"[..]));
    assert!(store.get_at(b"b", 150).is_none());

    // Snapshot at ts=250 sees a1 + b1.
    assert_eq!(store.get_at(b"a", 250).as_deref(), Some(&b"a1"[..]));
    assert_eq!(store.get_at(b"b", 250).as_deref(), Some(&b"b1"[..]));

    // Snapshot at ts=400 sees a2 + b1.
    assert_eq!(store.get_at(b"a", 400).as_deref(), Some(&b"a2"[..]));
    assert_eq!(store.get_at(b"b", 400).as_deref(), Some(&b"b1"[..]));
}

#[test]
fn snapshot_isolation_handles_tombstones_correctly() {
    let mut store = MvccStore::new();
    store
        .put_payload(b"k", 100, Bytes::from_static(b"v1"))
        .unwrap();
    store.delete(b"k", 200).unwrap(); // tombstone at 200
    store
        .put_payload(b"k", 300, Bytes::from_static(b"v2"))
        .unwrap();

    // Snapshot before delete: see v1.
    assert_eq!(store.get_at(b"k", 150).as_deref(), Some(&b"v1"[..]));
    // Snapshot between delete and revive: see None (deleted).
    assert!(store.get_at(b"k", 250).is_none());
    // Snapshot after revive: see v2.
    assert_eq!(store.get_at(b"k", 400).as_deref(), Some(&b"v2"[..]));
}

#[test]
fn gc_safety_keeps_the_floor_version_even_after_safe_ts_advances() {
    // The "floor" version is the newest version with
    // commit_ts ≤ safe_ts. GC must keep it so reads at
    // read_ts = safe_ts return correctly.
    let mut store = MvccStore::new();
    store
        .put_payload(b"k", 100, Bytes::from_static(b"v1"))
        .unwrap();
    // Advance safe_ts to 200 — v1 is the floor (≤ 200, no
    // newer version exists). GC must NOT drop it.
    store.set_safe_ts(200);
    let dropped = store.gc();
    assert_eq!(
        dropped, 0,
        "floor version v1 must not be dropped (no newer version exists)"
    );
    // The floor is still observable.
    assert_eq!(store.get_at(b"k", 200).as_deref(), Some(&b"v1"[..]));
}

#[test]
fn gc_safety_drops_only_versions_strictly_below_floor() {
    let mut store = MvccStore::new();
    store
        .put_payload(b"k", 100, Bytes::from_static(b"v1"))
        .unwrap();
    store
        .put_payload(b"k", 200, Bytes::from_static(b"v2"))
        .unwrap();
    store
        .put_payload(b"k", 300, Bytes::from_static(b"v3"))
        .unwrap();
    store
        .put_payload(b"k", 400, Bytes::from_static(b"v4"))
        .unwrap();

    // Safe at 350: floor is v3 (300 ≤ 350; v2/v1 are
    // strictly below the floor). v4 is above safe_ts so must
    // be kept too (a reader at read_ts=400 needs it).
    store.set_safe_ts(350);
    let dropped = store.gc();
    assert_eq!(
        dropped, 2,
        "v1 + v2 strictly below the floor should be dropped"
    );

    // Surviving versions:
    //  - v3 (floor, kept regardless)
    //  - v4 (above safe_ts, never eligible)
    assert_eq!(store.get_at(b"k", 300).as_deref(), Some(&b"v3"[..]));
    assert_eq!(store.get_at(b"k", 400).as_deref(), Some(&b"v4"[..]));
    // Reading at the safe_ts itself must still see v3.
    assert_eq!(store.get_at(b"k", 350).as_deref(), Some(&b"v3"[..]));
}

#[test]
fn gc_safety_preserves_tombstone_floor() {
    // If the floor IS a tombstone, GC must keep it. Dropping
    // the tombstone would resurface the older value to readers
    // at safe_ts, violating the delete history.
    let mut store = MvccStore::new();
    store
        .put_payload(b"k", 100, Bytes::from_static(b"v1"))
        .unwrap();
    store.delete(b"k", 200).unwrap();
    // Advance safe_ts past the tombstone.
    store.set_safe_ts(250);
    let _ = store.gc();
    // Reads at read_ts ≥ 200 must STILL see None (tombstone).
    assert!(
        store.get_at(b"k", 250).is_none(),
        "tombstone floor must not be GC'd; key must stay deleted at read_ts=250"
    );
    assert!(store.get_at(b"k", 300).is_none());
}

#[test]
fn gc_safety_never_drops_versions_above_safe_ts() {
    // Versions above safe_ts may still be observed by readers
    // at higher read_ts; GC must leave them alone.
    let mut store = MvccStore::new();
    store
        .put_payload(b"k", 100, Bytes::from_static(b"v1"))
        .unwrap();
    store
        .put_payload(b"k", 500, Bytes::from_static(b"v2"))
        .unwrap();
    store.set_safe_ts(200);
    let dropped = store.gc();
    // v1 is the floor (kept); v2 is above safe_ts (kept).
    assert_eq!(dropped, 0);
    assert_eq!(store.get_at(b"k", 600).as_deref(), Some(&b"v2"[..]));
}

#[test]
fn snapshot_at_safe_ts_after_gc_returns_floor() {
    // Combined invariant: after GC, a snapshot read at the
    // safe_ts watermark returns exactly the floor — the
    // newest committed version ≤ safe_ts. This is what
    // protects long-running read-only transactions from
    // disappearing data.
    let mut store = MvccStore::new();
    for ts in [50u128, 100, 150, 200, 250] {
        store
            .put_payload(b"k", ts, Bytes::from(format!("v{ts}").into_bytes()))
            .unwrap();
    }
    store.set_safe_ts(180);
    let _ = store.gc();
    // Reader at read_ts=180 (exactly safe_ts) must see v150 —
    // the floor at this watermark.
    assert_eq!(
        store.get_at(b"k", 180).as_deref(),
        Some(&b"v150"[..]),
        "snapshot at safe_ts must return the floor version"
    );
    // Reader at read_ts=250 sees v250 (above safe_ts is always kept).
    assert_eq!(store.get_at(b"k", 250).as_deref(), Some(&b"v250"[..]));
}

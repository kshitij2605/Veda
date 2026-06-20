//! Tests for record TTL (_expires_at field) — RED phase first.
//!
//! `_expires_at` is an `Int64` field holding a Unix epoch SECONDS timestamp.
//! A record is expired at `now` iff `_expires_at <= now`. Records without
//! `_expires_at` are never expired (backward compat).
//!
//! The expiry filter lives in `SegmentManager::lookup_record` (read path) and
//! `SegmentManager::sweep_expired` (sweeper). The segment-level visibility
//! (is_visible_at / delete_bitmap) is reused for sweep: expired records are
//! marked deleted so compaction purges them. `now` is always injected
//! (deterministic, not SystemTime::now()).

use tempfile::TempDir;
use veda_core::record::{EXPIRES_AT_FIELD, FieldValue, Record};
use veda_core::schema::{CollectionSchema, FieldDefinition};
use veda_core::types::{DistanceMetric, FieldType};
use veda_core::vector::DenseVector;
use veda_storage::segment_manager::SegmentManager;

// ── Helpers ────────────────────────────────────────────────────────────────

fn test_schema() -> CollectionSchema {
    CollectionSchema::builder("ttl_test")
        .add_field(FieldDefinition::vector(
            "embedding",
            4,
            DistanceMetric::Euclidean,
        ))
        .add_field(FieldDefinition::new("tag", FieldType::String))
        .build()
        .unwrap()
}

fn make_vector(seed: u32) -> Vec<f32> {
    let mut v = vec![0.0f32; 4];
    let mut state = seed;
    for x in &mut v {
        state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
        *x = ((state >> 16) as f32 / 32768.0) - 1.0;
    }
    v
}

/// Build a record with a required embedding + optional TTL.
fn make_record_with_ttl(seed: u32, expires_at: Option<i64>) -> Record {
    let mut rec = Record::new();
    rec.set(
        "embedding",
        FieldValue::Vector(DenseVector::new(make_vector(seed))),
    );
    rec.set("tag", FieldValue::String(format!("item_{seed}")));
    if let Some(ts) = expires_at {
        rec.set(EXPIRES_AT_FIELD, FieldValue::Int64(ts));
    }
    rec
}

// ── Tests: read-time exclusion via lookup_record ───────────────────────────

/// A record without _expires_at is always visible (backward compat).
#[test]
fn no_expires_field_always_visible() {
    let dir = TempDir::new().unwrap();
    let mut mgr = SegmentManager::new(test_schema(), 0, dir.path(), 512 * 1024 * 1024).unwrap();

    let rec = make_record_with_ttl(1, None);
    let id = rec.id();
    mgr.insert(rec).unwrap();

    // Far future now — record still visible.
    let result = mgr.lookup_record_at(id, i64::MAX);
    assert!(
        result.is_some(),
        "record without TTL must always be visible"
    );
}

/// A record with _expires_at in the FUTURE is visible.
#[test]
fn future_expires_is_visible() {
    let dir = TempDir::new().unwrap();
    let mut mgr = SegmentManager::new(test_schema(), 0, dir.path(), 512 * 1024 * 1024).unwrap();

    let future_ts: i64 = 9_999_999_999; // year 2286
    let rec = make_record_with_ttl(2, Some(future_ts));
    let id = rec.id();
    mgr.insert(rec).unwrap();

    let now: i64 = 1_000_000_000; // year 2001 — well before expiry
    let result = mgr.lookup_record_at(id, now);
    assert!(result.is_some(), "record with future TTL must be visible");
}

/// A record with _expires_at == now is expired (boundary: <= means expired).
#[test]
fn expires_at_exactly_now_is_expired() {
    let dir = TempDir::new().unwrap();
    let mut mgr = SegmentManager::new(test_schema(), 0, dir.path(), 512 * 1024 * 1024).unwrap();

    let ts: i64 = 1_700_000_000;
    let rec = make_record_with_ttl(3, Some(ts));
    let id = rec.id();
    mgr.insert(rec).unwrap();

    let result = mgr.lookup_record_at(id, ts);
    assert!(
        result.is_none(),
        "record with _expires_at == now must be treated as expired"
    );
}

/// A record with _expires_at in the PAST is not visible.
#[test]
fn past_expires_is_not_visible() {
    let dir = TempDir::new().unwrap();
    let mut mgr = SegmentManager::new(test_schema(), 0, dir.path(), 512 * 1024 * 1024).unwrap();

    let past_ts: i64 = 1_000_000_000;
    let rec = make_record_with_ttl(4, Some(past_ts));
    let id = rec.id();
    mgr.insert(rec).unwrap();

    let now: i64 = 9_999_999_999;
    let result = mgr.lookup_record_at(id, now);
    assert!(result.is_none(), "record with past TTL must not be visible");
}

/// Same as above but after the record is in a SEALED segment.
#[test]
fn past_expires_in_sealed_segment_not_visible() {
    let dir = TempDir::new().unwrap();
    let mut mgr = SegmentManager::new(test_schema(), 0, dir.path(), 512 * 1024 * 1024).unwrap();

    let past_ts: i64 = 1_000_000_000;
    let rec = make_record_with_ttl(5, Some(past_ts));
    let id = rec.id();
    mgr.insert(rec).unwrap();
    mgr.seal_growing().unwrap();

    let now: i64 = 9_999_999_999;
    let result = mgr.lookup_record_at(id, now);
    assert!(
        result.is_none(),
        "sealed record with past TTL must not be visible"
    );
}

// ── Tests: sweep_expired ───────────────────────────────────────────────────

/// Sweeping marks expired records deleted; non-expired survive.
#[test]
fn sweep_marks_expired_deleted_in_growing() {
    let dir = TempDir::new().unwrap();
    let mut mgr = SegmentManager::new(test_schema(), 0, dir.path(), 512 * 1024 * 1024).unwrap();

    let past_ts: i64 = 1_000_000_000;
    let future_ts: i64 = 9_999_999_999;
    let now: i64 = 2_000_000_000;

    let expired_rec = make_record_with_ttl(10, Some(past_ts));
    let expired_id = expired_rec.id();
    mgr.insert(expired_rec).unwrap();

    let alive_rec = make_record_with_ttl(11, Some(future_ts));
    let alive_id = alive_rec.id();
    mgr.insert(alive_rec).unwrap();

    let no_ttl_rec = make_record_with_ttl(12, None);
    let no_ttl_id = no_ttl_rec.id();
    mgr.insert(no_ttl_rec).unwrap();

    let swept = mgr.sweep_expired(now);
    assert_eq!(swept, 1, "exactly 1 record should be swept");

    // Expired: deleted in delete_bitmap → lookup returns None.
    assert!(
        mgr.growing_segment().is_deleted(0),
        "expired record's doc_id must be marked deleted"
    );

    // Others untouched.
    assert!(
        mgr.lookup_record(alive_id).is_some(),
        "future-TTL record must survive sweep"
    );
    assert!(
        mgr.lookup_record(no_ttl_id).is_some(),
        "no-TTL record must survive sweep"
    );

    // lookup_record_at also excludes it.
    assert!(
        mgr.lookup_record_at(expired_id, now).is_none(),
        "swept expired record must be excluded from reads"
    );
}

/// Sweeping sealed segments also marks expired records deleted.
#[test]
fn sweep_marks_expired_deleted_in_sealed() {
    let dir = TempDir::new().unwrap();
    let mut mgr = SegmentManager::new(test_schema(), 0, dir.path(), 512 * 1024 * 1024).unwrap();

    let past_ts: i64 = 1_000_000_000;
    let now: i64 = 2_000_000_000;

    let expired_rec = make_record_with_ttl(20, Some(past_ts));
    let expired_id = expired_rec.id();
    mgr.insert(expired_rec).unwrap();

    let alive_rec = make_record_with_ttl(21, Some(9_999_999_999));
    let alive_id = alive_rec.id();
    mgr.insert(alive_rec).unwrap();

    mgr.seal_growing().unwrap();

    let swept = mgr.sweep_expired(now);
    assert_eq!(swept, 1, "exactly 1 sealed record should be swept");

    // After sweep, the sealed segment's delete_bitmap must be set.
    let handle = &mgr.sealed_segments()[0];
    assert!(
        handle.reader.delete_bitmap().is_deleted(0),
        "expired sealed record's doc_id must be marked deleted"
    );
    assert!(
        !handle.reader.delete_bitmap().is_deleted(1),
        "alive sealed record must not be marked deleted"
    );

    // lookup_record for expired returns None.
    assert!(
        mgr.lookup_record(expired_id).is_none(),
        "swept record must not be returned by lookup_record"
    );
    assert!(
        mgr.lookup_record(alive_id).is_some(),
        "alive sealed record must be found by lookup_record"
    );
}

/// Sweeping with a `now` in the far past sweeps nothing.
#[test]
fn sweep_nothing_when_now_is_far_past() {
    let dir = TempDir::new().unwrap();
    let mut mgr = SegmentManager::new(test_schema(), 0, dir.path(), 512 * 1024 * 1024).unwrap();

    let ts: i64 = 1_700_000_000;
    let rec = make_record_with_ttl(30, Some(ts));
    mgr.insert(rec).unwrap();

    // now is before ts → nothing should be swept.
    let swept = mgr.sweep_expired(ts - 1);
    assert_eq!(swept, 0, "no records should be swept when now < expires_at");
}

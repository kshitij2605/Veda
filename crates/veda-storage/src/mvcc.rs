//! MVCC primitive types for Phase F of the FoundationDB-shape
//! rearchitecture (`.claude/plans/structured-mixing-turing.md`).
//!
//! Veda's existing storage layer is keyed by `RecordId` alone —
//! one row per id, last-write-wins. Phase F adds a `commit_ts`
//! dimension so a read at `read_ts = T` sees the latest write
//! whose `commit_ts ≤ T`. That's the foundation for Phase G
//! snapshot-isolation transactions and for the WARP /
//! FoundationDB ordering invariants the data plane needs.
//!
//! ## What v0 ships here
//!
//! - [`MvccKey`] — `(user_key, commit_ts)` with a memcmp-comparable
//!   encoding (descending commit_ts so a forward iterator from
//!   `(user_key, READ_TS)` finds the newest visible version
//!   immediately).
//! - [`MvccStore`] — in-memory `BTreeMap<MvccKey, Bytes>` for the
//!   primitive semantics: put / get_at / latest / range_for_key /
//!   set_safe_ts / gc.
//! - Hard cap on versions per key (`max_versions_per_key`) as
//!   the circuit-breaker the plan calls for.
//! - Tombstones for deletes — kept until safe_ts moves past them.
//!
//! ## What v0 does NOT ship (documented v1 steps)
//!
//! - Integration of `MvccKey` into `SegmentManager::insert`. The
//!   live segment manager still stores by `RecordId`. The v1 plan
//!   has [`SegmentManager`] gaining a versioned tier; this file
//!   is its prerequisite.
//! - Versioned BM25 / vector / sparse indexes — those currently
//!   key on `(segment_id, doc_id)` and will gain MVCC-aware
//!   visibility filters when the SegmentManager change lands.
//! - On-disk persistence of the MVCC store. The in-memory
//!   `MvccStore` here is the algorithmic core; LSM-style
//!   segments come in v1.

use std::collections::BTreeMap;
use std::collections::HashMap;

use bytes::Bytes;
use serde::{Deserialize, Serialize};

/// MVCC timestamp. Same shape as `veda_time::Hlc` packed into a
/// u128 for memcmp-friendly storage. Reads compare timestamps,
/// not field-named structs.
pub type Ts = u128;

/// MVCC key = `(user_key, commit_ts)`. Sorted by `user_key`
/// ascending then `commit_ts` DESCENDING — so a forward iterator
/// from `(k, MAX_TS - read_ts)` finds the newest visible version
/// for `k` as its first hit.
///
/// The "descending commit_ts" trick is the RocksDB / TiKV
/// convention; saves one reverse-seek per read.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct MvccKey {
    pub user_key: Vec<u8>,
    /// Inverse of the real commit_ts — `INVERSE_TS - commit_ts`.
    /// Stored this way so smaller (sorted-earlier) keys are
    /// newer commits; forward iter from a probe finds the
    /// newest-visible-at-read_ts first.
    pub inverse_ts: Ts,
}

/// Maximum representable timestamp; used to compute the inverse.
pub const MAX_TS: Ts = Ts::MAX;

impl MvccKey {
    /// Construct from `(user_key, commit_ts)`.
    pub fn new(user_key: impl Into<Vec<u8>>, commit_ts: Ts) -> Self {
        Self {
            user_key: user_key.into(),
            inverse_ts: MAX_TS - commit_ts,
        }
    }

    /// Recover the real commit_ts.
    pub fn commit_ts(&self) -> Ts {
        MAX_TS - self.inverse_ts
    }

    /// Lowest key for `user_key` (== `(user_key, +∞)` in
    /// commit_ts terms). Forward iter from here yields newest
    /// version first.
    pub fn lo(user_key: &[u8]) -> Self {
        Self {
            user_key: user_key.to_vec(),
            inverse_ts: 0,
        }
    }

    /// Highest key for `user_key`. Forward iter from here is the
    /// "past the oldest" boundary.
    pub fn hi(user_key: &[u8]) -> Self {
        Self {
            user_key: user_key.to_vec(),
            inverse_ts: MAX_TS,
        }
    }
}

/// One stored version. `None` payload = tombstone.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MvccValue {
    pub payload: Option<Bytes>,
}

impl MvccValue {
    pub fn put(payload: Bytes) -> Self {
        Self {
            payload: Some(payload),
        }
    }
    pub fn tombstone() -> Self {
        Self { payload: None }
    }
    pub fn is_tombstone(&self) -> bool {
        self.payload.is_none()
    }
}

/// In-memory MVCC store. The primitive Phase F layer; the live
/// SegmentManager integration (v1) builds on top of this exact
/// algorithm.
#[derive(Debug, Default)]
pub struct MvccStore {
    /// All versions, sorted by `(user_key asc, commit_ts desc)`
    /// via the `MvccKey` encoding.
    store: BTreeMap<MvccKey, MvccValue>,
    /// Per-user-key version count — bounded by
    /// `max_versions_per_key`. Inserts past the cap drop the
    /// oldest version that's also ≤ safe_ts.
    counts: HashMap<Vec<u8>, usize>,
    /// All versions with `commit_ts ≤ safe_ts` are eligible for
    /// GC. Default 0 — operator must advance it explicitly.
    safe_ts: Ts,
    /// Operator-tunable cap. 100 matches the plan's default.
    max_versions_per_key: usize,
}

impl MvccStore {
    pub fn new() -> Self {
        Self {
            store: BTreeMap::new(),
            counts: HashMap::new(),
            safe_ts: 0,
            max_versions_per_key: 100,
        }
    }

    pub fn with_max_versions_per_key(mut self, n: usize) -> Self {
        self.max_versions_per_key = n.max(1);
        self
    }

    /// Insert a new (user_key, commit_ts, value). If the key
    /// already has `max_versions_per_key` versions stored, the
    /// oldest version ≤ safe_ts is dropped to make room. If
    /// the per-key cap is hit AND no version is past safe_ts,
    /// the insert fails — the caller (or operator) needs to
    /// advance safe_ts.
    pub fn put(
        &mut self,
        user_key: &[u8],
        commit_ts: Ts,
        value: MvccValue,
    ) -> Result<(), MvccError> {
        let k = MvccKey::new(user_key.to_vec(), commit_ts);
        let count = *self.counts.get(user_key).unwrap_or(&0);
        if count >= self.max_versions_per_key {
            self.evict_oldest_safe(user_key)?;
        }
        if self.store.insert(k.clone(), value).is_none() {
            *self.counts.entry(user_key.to_vec()).or_insert(0) += 1;
        }
        Ok(())
    }

    /// Convenience: put a payload as a non-tombstone version.
    pub fn put_payload(
        &mut self,
        user_key: &[u8],
        commit_ts: Ts,
        payload: Bytes,
    ) -> Result<(), MvccError> {
        self.put(user_key, commit_ts, MvccValue::put(payload))
    }

    /// Convenience: write a tombstone.
    pub fn delete(&mut self, user_key: &[u8], commit_ts: Ts) -> Result<(), MvccError> {
        self.put(user_key, commit_ts, MvccValue::tombstone())
    }

    /// MVCC point read: return the newest version of `user_key`
    /// whose `commit_ts ≤ read_ts`. Returns `None` if no such
    /// version exists OR the newest visible version is a
    /// tombstone.
    pub fn get_at(&self, user_key: &[u8], read_ts: Ts) -> Option<Bytes> {
        let probe = MvccKey::new(user_key.to_vec(), read_ts);
        // Forward iter from probe: first hit (if any) with the
        // same user_key has the largest commit_ts ≤ read_ts (due
        // to inverse_ts encoding).
        if let Some((k, v)) = self.store.range(probe..).next()
            && k.user_key == user_key
        {
            return v.payload.clone();
        }
        None
    }

    /// Latest version of `user_key` regardless of read_ts —
    /// useful for admin / debug.
    pub fn latest(&self, user_key: &[u8]) -> Option<(Ts, MvccValue)> {
        let lo = MvccKey::lo(user_key);
        let hi = MvccKey::hi(user_key);
        self.store
            .range(lo..=hi)
            .next()
            .map(|(k, v)| (k.commit_ts(), v.clone()))
    }

    /// All versions for `user_key`, newest first.
    pub fn versions(&self, user_key: &[u8]) -> Vec<(Ts, MvccValue)> {
        let lo = MvccKey::lo(user_key);
        let hi = MvccKey::hi(user_key);
        self.store
            .range(lo..=hi)
            .map(|(k, v)| (k.commit_ts(), v.clone()))
            .collect()
    }

    /// Number of stored versions across all keys. O(1).
    pub fn total_versions(&self) -> usize {
        self.store.len()
    }

    /// Phase F WAL checkpoint helper: clone the full BTreeMap +
    /// safe_ts into a `(Vec, Ts)` pair suitable for bincode
    /// serialisation. Caller (typically `MvccWal::checkpoint`)
    /// writes this atomically to a `mvcc.snap` file and then
    /// truncates the WAL.
    pub fn snapshot_entries(&self) -> (Vec<(MvccKey, MvccValue)>, Ts) {
        let entries: Vec<(MvccKey, MvccValue)> = self
            .store
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        (entries, self.safe_ts)
    }

    /// Phase F WAL checkpoint inverse: install the entries (and
    /// safe_ts) into a fresh `MvccStore`. Used by replay when a
    /// `mvcc.snap` file exists.
    pub fn install_snapshot_entries(&mut self, entries: Vec<(MvccKey, MvccValue)>, safe_ts: Ts) {
        self.store.clear();
        self.counts.clear();
        for (k, v) in entries {
            let user_key = k.user_key.clone();
            self.store.insert(k, v);
            *self.counts.entry(user_key).or_insert(0) += 1;
        }
        self.safe_ts = safe_ts;
    }

    /// Advance the GC safe-ts watermark. After this returns,
    /// versions with `commit_ts ≤ safe_ts` AND with a newer
    /// version present are eligible for `gc()`.
    pub fn set_safe_ts(&mut self, ts: Ts) {
        self.safe_ts = self.safe_ts.max(ts);
    }

    pub fn safe_ts(&self) -> Ts {
        self.safe_ts
    }

    /// Garbage-collect superseded versions.
    ///
    /// Per-user-key rule:
    ///   - Always keep every version with `commit_ts > safe_ts`
    ///     (these may be visible to readers at high read_ts).
    ///   - Keep the newest version with `commit_ts ≤ safe_ts` —
    ///     this is the "floor" that satisfies reads at
    ///     read_ts = safe_ts. Even if it's a tombstone, dropping
    ///     it would resurface an older visible value.
    ///   - Drop everything else (all versions ≤ safe_ts that are
    ///     OLDER than the floor).
    pub fn gc(&mut self) -> usize {
        let safe = self.safe_ts;
        let mut victims: Vec<MvccKey> = Vec::new();
        let user_keys: Vec<Vec<u8>> = self
            .store
            .keys()
            .map(|k| k.user_key.clone())
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
        for uk in user_keys {
            let lo = MvccKey::lo(&uk);
            let hi = MvccKey::hi(&uk);
            let versions: Vec<MvccKey> =
                self.store.range(lo..=hi).map(|(k, _)| k.clone()).collect();
            // Iterate newest-first. Keep `> safe`; keep the
            // first `≤ safe` as the floor; drop the rest.
            let mut floor_found = false;
            for k in versions {
                let commit = k.commit_ts();
                if commit > safe {
                    continue;
                }
                if !floor_found {
                    floor_found = true;
                } else {
                    victims.push(k);
                }
            }
        }
        let n = victims.len();
        for v in &victims {
            self.store.remove(v);
            if let Some(c) = self.counts.get_mut(&v.user_key) {
                *c = c.saturating_sub(1);
            }
        }
        self.counts.retain(|_, c| *c > 0);
        n
    }

    /// Drop the OLDEST version of `user_key` whose commit_ts ≤
    /// safe_ts. Returns Err if there's nothing eligible — the
    /// caller (or operator) needs to advance safe_ts.
    fn evict_oldest_safe(&mut self, user_key: &[u8]) -> Result<(), MvccError> {
        let lo = MvccKey::lo(user_key);
        let hi = MvccKey::hi(user_key);
        // Iterate newest-first; collect everything ≤ safe_ts.
        // Drop the last one (oldest such).
        let candidates: Vec<MvccKey> = self
            .store
            .range(lo..=hi)
            .filter(|(k, _)| k.commit_ts() <= self.safe_ts)
            .map(|(k, _)| k.clone())
            .collect();
        let victim = candidates
            .into_iter()
            .last()
            .ok_or(MvccError::TooManyVersions {
                user_key: user_key.to_vec(),
                cap: self.max_versions_per_key,
                safe_ts: self.safe_ts,
            })?;
        self.store.remove(&victim);
        if let Some(c) = self.counts.get_mut(user_key) {
            *c = c.saturating_sub(1);
        }
        Ok(())
    }
}

/// Errors that the MVCC store can surface.
#[derive(Debug, thiserror::Error)]
pub enum MvccError {
    #[error(
        "max_versions_per_key={cap} exceeded for key {user_key:?}; \
         no version ≤ safe_ts={safe_ts} available to evict — advance safe_ts"
    )]
    TooManyVersions {
        user_key: Vec<u8>,
        cap: usize,
        safe_ts: Ts,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mvcc_key_descending_order() {
        // For the same user_key, larger commit_ts maps to SMALLER
        // MvccKey (descending order).
        let early = MvccKey::new("k".as_bytes().to_vec(), 100);
        let late = MvccKey::new("k".as_bytes().to_vec(), 200);
        assert!(
            late < early,
            "late commit must sort before early: {late:?} >= {early:?}"
        );
    }

    #[test]
    fn put_get_at_returns_newest_visible() {
        let mut s = MvccStore::new();
        s.put_payload(b"k", 100, Bytes::from_static(b"v100"))
            .unwrap();
        s.put_payload(b"k", 200, Bytes::from_static(b"v200"))
            .unwrap();
        s.put_payload(b"k", 300, Bytes::from_static(b"v300"))
            .unwrap();
        assert_eq!(s.get_at(b"k", 150).as_deref(), Some(&b"v100"[..]));
        assert_eq!(s.get_at(b"k", 250).as_deref(), Some(&b"v200"[..]));
        assert_eq!(s.get_at(b"k", 999).as_deref(), Some(&b"v300"[..]));
        // read_ts before any commit → None.
        assert!(s.get_at(b"k", 50).is_none());
    }

    #[test]
    fn tombstone_hides_value_from_read_at_or_after() {
        let mut s = MvccStore::new();
        s.put_payload(b"k", 100, Bytes::from_static(b"v")).unwrap();
        s.delete(b"k", 200).unwrap();
        assert_eq!(s.get_at(b"k", 150).as_deref(), Some(&b"v"[..]));
        assert!(
            s.get_at(b"k", 250).is_none(),
            "after tombstone must be None"
        );
    }

    #[test]
    fn gc_drops_superseded_versions_under_safe_ts() {
        let mut s = MvccStore::new();
        for ts in [100, 200, 300, 400] {
            let body = format!("v{ts}");
            s.put_payload(b"k", ts, Bytes::from(body.into_bytes()))
                .unwrap();
        }
        assert_eq!(s.total_versions(), 4);
        s.set_safe_ts(250);
        let dropped = s.gc();
        // Floor at safe_ts=250 is ts=200 (newest ≤ 250). The
        // older ts=100 is dropped.
        assert_eq!(dropped, 1);
        // Reading at safe_ts itself hits the floor.
        assert_eq!(s.get_at(b"k", 250).as_deref(), Some(&b"v200"[..]));
        // Reads BELOW safe_ts have no guarantee — the safe_ts
        // contract says reads must be ≥ safe_ts. We don't assert
        // anything about ts=150.
        // Reading at ts=350 still sees 300.
        assert_eq!(s.get_at(b"k", 350).as_deref(), Some(&b"v300"[..]));
    }

    #[test]
    fn gc_keeps_tombstone_at_floor() {
        // Sequence: live at 100, tombstone at 200. With
        // safe_ts=300, the tombstone IS the floor — it must
        // survive GC so reads at ≥ 200 keep observing None.
        // The older live value at 100 IS dropped.
        let mut s = MvccStore::new();
        s.put_payload(b"k", 100, Bytes::from_static(b"v")).unwrap();
        s.delete(b"k", 200).unwrap();
        s.set_safe_ts(300);
        let dropped = s.gc();
        assert_eq!(dropped, 1, "only the older live value should GC");
        assert!(
            s.get_at(b"k", 999).is_none(),
            "tombstone floor must survive GC"
        );
    }

    #[test]
    fn max_versions_per_key_caps_storage() {
        let mut s = MvccStore::new().with_max_versions_per_key(3);
        // Three puts succeed.
        for ts in [100, 200, 300] {
            s.put_payload(b"k", ts, Bytes::from_static(b"v")).unwrap();
        }
        // Fourth must fail until safe_ts moves.
        let err = s
            .put_payload(b"k", 400, Bytes::from_static(b"v"))
            .unwrap_err();
        assert!(matches!(err, MvccError::TooManyVersions { .. }));
        // Advance safe_ts then retry.
        s.set_safe_ts(150);
        s.put_payload(b"k", 400, Bytes::from_static(b"v")).unwrap();
        // get_at(150) — original 100 was evicted, so we expect
        // None at read_ts=150.
        assert!(s.get_at(b"k", 150).is_none());
        assert_eq!(s.get_at(b"k", 250).as_deref(), Some(&b"v"[..]));
    }

    #[test]
    fn snapshot_isolation_property_holds_across_versions() {
        // Reader at snapshot ts=200 never sees writes at ts > 200,
        // even when those writes happen "after" (in wall-clock
        // sense) but before the reader queries.
        let mut s = MvccStore::new();
        s.put_payload(b"k", 100, Bytes::from_static(b"v100"))
            .unwrap();
        // Reader's snapshot opens at ts=200.
        let snap_ts: Ts = 200;
        // Concurrent writer races a new version at ts=300.
        s.put_payload(b"k", 300, Bytes::from_static(b"v300"))
            .unwrap();
        // Reader still observes v100 (the only version ≤ 200).
        assert_eq!(s.get_at(b"k", snap_ts).as_deref(), Some(&b"v100"[..]));
        // A reader at ts=350 sees v300.
        assert_eq!(s.get_at(b"k", 350).as_deref(), Some(&b"v300"[..]));
    }

    #[test]
    fn multiple_keys_isolated_in_store() {
        let mut s = MvccStore::new();
        s.put_payload(b"a", 100, Bytes::from_static(b"a100"))
            .unwrap();
        s.put_payload(b"b", 100, Bytes::from_static(b"b100"))
            .unwrap();
        s.put_payload(b"a", 200, Bytes::from_static(b"a200"))
            .unwrap();
        assert_eq!(s.get_at(b"a", 250).as_deref(), Some(&b"a200"[..]));
        assert_eq!(s.get_at(b"b", 250).as_deref(), Some(&b"b100"[..]));
    }

    #[test]
    fn versions_returns_newest_first() {
        let mut s = MvccStore::new();
        for ts in [100, 200, 300] {
            s.put_payload(b"k", ts, Bytes::from(format!("v{ts}").into_bytes()))
                .unwrap();
        }
        let v = s.versions(b"k");
        assert_eq!(v.len(), 3);
        assert!(v[0].0 > v[1].0 && v[1].0 > v[2].0);
    }
}

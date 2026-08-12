use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::sparse::SparseVec;
use crate::types::RecordId;
use crate::vector::DenseVector;

/// Reserved field name for per-record TTL (expiry).
///
/// The value must be a `FieldValue::Int64` holding a Unix epoch SECONDS
/// timestamp. A record is considered expired at read time when
/// `_expires_at <= now_secs`. Records without this field never expire.
pub const EXPIRES_AT_FIELD: &str = "_expires_at";

/// Reserved field: retrieval-guardrail screening flagged this record.
///
/// `FieldValue::Bool`. Absent means the record was never screened (indexed
/// before retrieval guardrails shipped, or screening failed open) — which is
/// reported as `unscreened`, NOT as clean.
pub const GUARDRAIL_FLAGGED_FIELD: &str = "_guardrail_flagged";

/// Reserved field: strongest pattern similarity observed while screening.
///
/// `FieldValue::Float64`. Stamped even when the record was NOT flagged, so
/// sub-threshold near-misses stay measurable.
pub const GUARDRAIL_SCORE_FIELD: &str = "_guardrail_score";

/// Reserved field: category of the strongest match, when flagged.
///
/// `FieldValue::String`, the snake_case `GuardrailCategory` name.
pub const GUARDRAIL_CATEGORY_FIELD: &str = "_guardrail_category";

/// Reserved field: ingest time as Unix epoch SECONDS.
///
/// `FieldValue::Int64`. Required for the temporal staleness check: MVCC
/// `commit_ts` is optional per record and absent on ordinary inserts, so it
/// cannot carry this signal alone.
pub const INDEXED_AT_FIELD: &str = "_indexed_at";

/// A dynamically-typed field value that can be stored in a [`Record`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FieldValue {
    String(String),
    Int32(i32),
    Int64(i64),
    Float32(f32),
    Float64(f64),
    Bool(bool),
    /// Unix milliseconds timestamp.
    Timestamp(i64),
    Vector(DenseVector),
    SparseVector(SparseVec),
    /// ColPali-style multi-vector: a list of dense vectors of equal dimension.
    MultiVector(Vec<Vec<f32>>),
    Json(serde_json::Value),
    /// URI pointing to object storage.
    BlobRef(String),
    Uuid(Uuid),
    Array(Vec<FieldValue>),
    Null,
}

/// A single record (row) in a collection, keyed by a UUID.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Record {
    id: RecordId,
    fields: HashMap<String, FieldValue>,
}

impl Record {
    /// Create a new record with an auto-generated UUID.
    pub fn new() -> Self {
        Self {
            id: Uuid::new_v4(),
            fields: HashMap::new(),
        }
    }

    /// Create a new record with an explicit ID.
    pub fn with_id(id: RecordId) -> Self {
        Self {
            id,
            fields: HashMap::new(),
        }
    }

    /// Return the record's ID.
    pub fn id(&self) -> RecordId {
        self.id
    }

    /// Set a field value, replacing any previous value for that field.
    pub fn set(&mut self, field: &str, value: FieldValue) {
        self.fields.insert(field.to_owned(), value);
    }

    /// Get a reference to a field value, or `None` if the field is absent.
    pub fn get(&self, field: &str) -> Option<&FieldValue> {
        self.fields.get(field)
    }

    /// Remove a field and return its value, or `None` if absent.
    pub fn remove(&mut self, field: &str) -> Option<FieldValue> {
        self.fields.remove(field)
    }

    /// Return a reference to the entire field map.
    pub fn fields(&self) -> &HashMap<String, FieldValue> {
        &self.fields
    }

    /// Return the names of all fields present in this record.
    pub fn field_names(&self) -> Vec<&str> {
        self.fields.keys().map(|k| k.as_str()).collect()
    }

    /// Check whether the record contains the named field.
    pub fn has_field(&self, field: &str) -> bool {
        self.fields.contains_key(field)
    }

    /// Return the TTL expiry as Unix epoch SECONDS, or `None` if not set.
    ///
    /// Reads `_expires_at` as an `Int64`. Any other variant (or absent field)
    /// is treated as "no expiry".
    pub fn expires_at_secs(&self) -> Option<i64> {
        match self.fields.get(EXPIRES_AT_FIELD) {
            Some(FieldValue::Int64(ts)) => Some(*ts),
            _ => None,
        }
    }

    /// Returns `true` if this record has expired at `now_secs`.
    ///
    /// `now_secs` is the caller-supplied Unix epoch SECONDS (injected for
    /// determinism in tests). Records without `_expires_at` never expire.
    pub fn is_expired_at(&self, now_secs: i64) -> bool {
        match self.expires_at_secs() {
            Some(ts) => ts <= now_secs,
            None => false,
        }
    }

    /// Whether retrieval-guardrail screening flagged this record.
    ///
    /// `None` = never screened. Callers MUST distinguish that from
    /// `Some(false)` (screened and clean).
    pub fn guardrail_flagged(&self) -> Option<bool> {
        match self.fields.get(GUARDRAIL_FLAGGED_FIELD) {
            Some(FieldValue::Bool(b)) => Some(*b),
            _ => None,
        }
    }

    /// Strongest pattern similarity observed at screening time.
    pub fn guardrail_score(&self) -> Option<f64> {
        match self.fields.get(GUARDRAIL_SCORE_FIELD) {
            Some(FieldValue::Float64(s)) => Some(*s),
            _ => None,
        }
    }

    /// Category of the strongest match, when flagged.
    pub fn guardrail_category(&self) -> Option<&str> {
        match self.fields.get(GUARDRAIL_CATEGORY_FIELD) {
            Some(FieldValue::String(s)) => Some(s.as_str()),
            _ => None,
        }
    }

    /// Ingest time as Unix epoch SECONDS, or `None` when unstamped.
    pub fn indexed_at_secs(&self) -> Option<i64> {
        match self.fields.get(INDEXED_AT_FIELD) {
            Some(FieldValue::Int64(ts)) => Some(*ts),
            _ => None,
        }
    }
}

impl Default for Record {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn guardrail_accessors_read_the_reserved_fields() {
        let mut r = Record::new();
        assert_eq!(r.guardrail_flagged(), None, "absent field reads as None");
        assert_eq!(r.guardrail_score(), None);
        assert_eq!(r.guardrail_category(), None);
        assert_eq!(r.indexed_at_secs(), None);

        r.set(GUARDRAIL_FLAGGED_FIELD, FieldValue::Bool(true));
        r.set(GUARDRAIL_SCORE_FIELD, FieldValue::Float64(0.71));
        r.set(
            GUARDRAIL_CATEGORY_FIELD,
            FieldValue::String("indirect_injection".to_owned()),
        );
        r.set(INDEXED_AT_FIELD, FieldValue::Int64(1_786_000_000));

        assert_eq!(r.guardrail_flagged(), Some(true));
        assert_eq!(r.guardrail_score(), Some(0.71));
        assert_eq!(r.guardrail_category(), Some("indirect_injection"));
        assert_eq!(r.indexed_at_secs(), Some(1_786_000_000));
    }

    #[test]
    fn guardrail_accessors_ignore_wrong_variants() {
        // A wrong FieldValue variant must read as absent, not panic — mirrors
        // how expires_at_secs treats a non-Int64.
        let mut r = Record::new();
        r.set(
            GUARDRAIL_FLAGGED_FIELD,
            FieldValue::String("yes".to_owned()),
        );
        r.set(GUARDRAIL_SCORE_FIELD, FieldValue::Int64(1));
        r.set(INDEXED_AT_FIELD, FieldValue::String("nope".to_owned()));
        assert_eq!(r.guardrail_flagged(), None);
        assert_eq!(r.guardrail_score(), None);
        assert_eq!(r.indexed_at_secs(), None);
    }
}

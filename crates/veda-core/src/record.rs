use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::sparse::SparseVec;
use crate::types::RecordId;
use crate::vector::DenseVector;

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
}

impl Default for Record {
    fn default() -> Self {
        Self::new()
    }
}

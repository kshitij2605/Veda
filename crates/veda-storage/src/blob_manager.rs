use base64::prelude::*;
use uuid::Uuid;

use veda_core::error::{Result, VedaError};
use veda_core::record::{FieldValue, Record};
use veda_core::schema::CollectionSchema;

use crate::blob_store::BlobStore;

/// Manages blob storage for a collection.
pub struct BlobManager {
    store: Box<dyn BlobStore>,
    collection_name: String,
}

impl BlobManager {
    pub fn new(store: Box<dyn BlobStore>, collection_name: &str) -> Self {
        Self {
            store,
            collection_name: collection_name.to_owned(),
        }
    }

    /// Process a record before insertion:
    /// - For each Blob field (per schema), decode the base64 string value,
    ///   upload to object storage, and replace with a BlobRef.
    pub fn process_insert(&self, record: &mut Record, schema: &CollectionSchema) -> Result<()> {
        let blob_fields: Vec<String> = schema
            .blob_fields()
            .iter()
            .map(|f| f.name().to_owned())
            .collect();

        for field_name in &blob_fields {
            let Some(value) = record.get(field_name) else {
                continue;
            };

            let raw_bytes = match value {
                FieldValue::String(s) => BASE64_STANDARD.decode(s).map_err(|e| {
                    VedaError::Storage(format!("Invalid base64 for field '{field_name}': {e}"))
                })?,
                _ => continue,
            };

            let key = format!("{}/{}/{}", self.collection_name, record.id(), field_name);
            self.store.put(&key, &raw_bytes)?;
            record.set(field_name, FieldValue::BlobRef(key));
        }

        Ok(())
    }

    /// Resolve blob refs in a record:
    /// - Fetch actual data from object storage and replace BlobRef with base64 string.
    pub fn resolve_blobs(&self, record: &mut Record, schema: &CollectionSchema) -> Result<()> {
        let blob_fields: Vec<String> = schema
            .blob_fields()
            .iter()
            .map(|f| f.name().to_owned())
            .collect();

        for field_name in &blob_fields {
            let Some(value) = record.get(field_name) else {
                continue;
            };

            let key = match value {
                FieldValue::BlobRef(k) => k.clone(),
                _ => continue,
            };

            let data = self.store.get(&key)?;
            let b64 = BASE64_STANDARD.encode(&data);
            record.set(field_name, FieldValue::String(b64));
        }

        Ok(())
    }

    /// Delete all blobs associated with a record.
    pub fn delete_blobs(&self, record_id: &Uuid) -> Result<()> {
        let prefix = format!("{}/{}/", self.collection_name, record_id);
        let keys = self.store.list(&prefix)?;
        for key in &keys {
            self.store.delete(key)?;
        }
        Ok(())
    }
}

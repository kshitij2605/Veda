use base64::prelude::*;
use tempfile::TempDir;

use veda_core::record::{FieldValue, Record};
use veda_core::schema::{CollectionSchema, FieldDefinition};
use veda_core::types::{DistanceMetric, FieldType};
use veda_storage::blob_manager::BlobManager;
use veda_storage::blob_store::{BlobStore, LocalBlobStore, ObjectStoreConfig};

fn make_store(tmp: &TempDir) -> LocalBlobStore {
    LocalBlobStore::new(tmp.path()).unwrap()
}

fn test_schema() -> CollectionSchema {
    CollectionSchema::builder("test_collection")
        .add_field(FieldDefinition::blob("image"))
        .add_field(FieldDefinition::new("title", FieldType::String))
        .add_field(FieldDefinition::vector("embedding", 4, DistanceMetric::Cosine))
        .build()
        .unwrap()
}

// --- LocalBlobStore tests ---

#[test]
fn local_put_get() {
    let tmp = TempDir::new().unwrap();
    let store = make_store(&tmp);

    let data = b"hello blob world";
    store.put("test_key", data).unwrap();

    let retrieved = store.get("test_key").unwrap();
    assert_eq!(retrieved, data);
}

#[test]
fn local_delete() {
    let tmp = TempDir::new().unwrap();
    let store = make_store(&tmp);

    store.put("del_key", b"data").unwrap();
    assert!(store.exists("del_key").unwrap());

    store.delete("del_key").unwrap();
    assert!(!store.exists("del_key").unwrap());
}

#[test]
fn local_list_prefix() {
    let tmp = TempDir::new().unwrap();
    let store = make_store(&tmp);

    for i in 0..5 {
        store.put(&format!("prefix/{i}.blob"), b"data").unwrap();
    }
    // A blob under a different prefix
    store.put("other/file.blob", b"data").unwrap();

    let listed = store.list("prefix/").unwrap();
    assert_eq!(listed.len(), 5);
    for key in &listed {
        assert!(key.starts_with("prefix/"));
    }
}

#[test]
fn local_nested_keys() {
    let tmp = TempDir::new().unwrap();
    let store = make_store(&tmp);

    let key = "col1/record1/image.blob";
    let data = b"nested blob data";
    store.put(key, data).unwrap();

    assert!(store.exists(key).unwrap());
    assert_eq!(store.get(key).unwrap(), data);
}

#[test]
fn get_nonexistent_blob() {
    let tmp = TempDir::new().unwrap();
    let store = make_store(&tmp);

    let result = store.get("does_not_exist");
    assert!(result.is_err());
}

// --- BlobManager tests ---

fn make_manager(tmp: &TempDir) -> BlobManager {
    let store = LocalBlobStore::new(tmp.path()).unwrap();
    BlobManager::new(Box::new(store), "test_collection")
}

#[test]
fn process_insert_replaces_with_blob_ref() {
    let tmp = TempDir::new().unwrap();
    let manager = make_manager(&tmp);
    let schema = test_schema();

    let raw_data = b"fake image bytes";
    let b64 = BASE64_STANDARD.encode(raw_data);

    let mut record = Record::new();
    record.set("image", FieldValue::String(b64));
    record.set("title", FieldValue::String("A photo".into()));

    manager.process_insert(&mut record, &schema).unwrap();

    // Image field should now be a BlobRef
    match record.get("image").unwrap() {
        FieldValue::BlobRef(key) => {
            assert!(key.starts_with("test_collection/"));
            assert!(key.ends_with("/image"));
        }
        other => panic!("Expected BlobRef, got {other:?}"),
    }

    // Verify the blob was stored
    let store = LocalBlobStore::new(tmp.path()).unwrap();
    if let FieldValue::BlobRef(key) = record.get("image").unwrap() {
        let stored = store.get(key).unwrap();
        assert_eq!(stored, raw_data);
    }
}

#[test]
fn resolve_blobs_recovers_data() {
    let tmp = TempDir::new().unwrap();
    let manager = make_manager(&tmp);
    let schema = test_schema();

    let raw_data = b"original image data";
    let b64_original = BASE64_STANDARD.encode(raw_data);

    let mut record = Record::new();
    record.set("image", FieldValue::String(b64_original.clone()));

    manager.process_insert(&mut record, &schema).unwrap();
    manager.resolve_blobs(&mut record, &schema).unwrap();

    match record.get("image").unwrap() {
        FieldValue::String(s) => {
            assert_eq!(s, &b64_original);
        }
        other => panic!("Expected String with base64, got {other:?}"),
    }
}

#[test]
fn delete_blobs_removes_from_storage() {
    let tmp = TempDir::new().unwrap();
    let manager = make_manager(&tmp);
    let schema = test_schema();

    let mut record = Record::new();
    record.set("image", FieldValue::String(BASE64_STANDARD.encode(b"img")));

    manager.process_insert(&mut record, &schema).unwrap();

    let record_id = record.id();
    manager.delete_blobs(&record_id).unwrap();

    // Verify the blob is gone
    let store = LocalBlobStore::new(tmp.path()).unwrap();
    let prefix = format!("test_collection/{}/", record_id);
    let remaining = store.list(&prefix).unwrap();
    assert!(remaining.is_empty());
}

#[test]
fn non_blob_fields_untouched() {
    let tmp = TempDir::new().unwrap();
    let manager = make_manager(&tmp);
    let schema = test_schema();

    let mut record = Record::new();
    record.set("image", FieldValue::String(BASE64_STANDARD.encode(b"img")));
    record.set("title", FieldValue::String("My Title".into()));

    manager.process_insert(&mut record, &schema).unwrap();

    // Title should be unchanged
    match record.get("title").unwrap() {
        FieldValue::String(s) => assert_eq!(s, "My Title"),
        other => panic!("Expected String, got {other:?}"),
    }

    // Image should be a BlobRef
    assert!(matches!(record.get("image").unwrap(), FieldValue::BlobRef(_)));
}

// --- ObjectStoreConfig tests ---

#[test]
fn object_store_config_minio_defaults() {
    let config = ObjectStoreConfig {
        backend: "minio".to_string(),
        endpoint: "http://minio:9000".to_string(),
        bucket: "veda-blobs".to_string(),
        access_key: "minioadmin".to_string(),
        secret_key: "minioadmin".to_string(),
        region: "us-east-1".to_string(),
    };
    assert_eq!(config.backend, "minio");
    assert_eq!(config.endpoint, "http://minio:9000");
    assert_eq!(config.bucket, "veda-blobs");
    assert_eq!(config.region, "us-east-1");
}

#[test]
fn object_store_config_s3_defaults() {
    let config = ObjectStoreConfig {
        backend: "s3".to_string(),
        endpoint: String::new(),
        bucket: "my-s3-bucket".to_string(),
        access_key: "AKIAIOSFODNN7EXAMPLE".to_string(),
        secret_key: "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY".to_string(),
        region: "us-west-2".to_string(),
    };
    assert_eq!(config.backend, "s3");
    assert_eq!(config.bucket, "my-s3-bucket");
    assert_eq!(config.region, "us-west-2");
}

#[test]
fn object_store_config_clone() {
    let config = ObjectStoreConfig {
        backend: "minio".to_string(),
        endpoint: "http://localhost:9000".to_string(),
        bucket: "test".to_string(),
        access_key: "key".to_string(),
        secret_key: "secret".to_string(),
        region: "us-east-1".to_string(),
    };
    let cloned = config.clone();
    assert_eq!(config.backend, cloned.backend);
    assert_eq!(config.endpoint, cloned.endpoint);
    assert_eq!(config.bucket, cloned.bucket);
    assert_eq!(config.access_key, cloned.access_key);
    assert_eq!(config.secret_key, cloned.secret_key);
    assert_eq!(config.region, cloned.region);
}

//! Integration test for ObjectBlobStore against a real MinIO instance.
//! Requires MinIO running at localhost:9200 (Docker Compose).
//! Run with: cargo test -p veda-storage --test object_blob_integration_test -- --ignored

use veda_storage::blob_store::{BlobStore, ObjectBlobStore, ObjectStoreConfig};

fn make_object_store() -> ObjectBlobStore {
    let config = ObjectStoreConfig {
        backend: "minio".to_string(),
        endpoint: "http://localhost:9200".to_string(),
        bucket: "veda-test-blobs".to_string(),
        access_key: "veda".to_string(),
        secret_key: "vedapassword".to_string(),
        region: "us-east-1".to_string(),
    };
    ObjectBlobStore::new(&config).expect("failed to connect to MinIO")
}

#[test]
#[ignore] // Requires running MinIO
fn test_object_store_put_get_delete() {
    let store = make_object_store();
    let key = "test-collection/test-id/test-field";
    let data = b"hello minio blob";

    store.put(key, data).unwrap();
    assert!(store.exists(key).unwrap());

    let retrieved = store.get(key).unwrap();
    assert_eq!(retrieved, data);

    store.delete(key).unwrap();
    assert!(!store.exists(key).unwrap());
}

#[test]
#[ignore]
fn test_object_store_list() {
    let store = make_object_store();
    store.put("list-test/a/1", b"one").unwrap();
    store.put("list-test/a/2", b"two").unwrap();
    store.put("list-test/b/3", b"three").unwrap();

    let keys = store.list("list-test/a/").unwrap();
    assert_eq!(keys.len(), 2);
    assert!(keys.contains(&"list-test/a/1".to_string()));
    assert!(keys.contains(&"list-test/a/2".to_string()));

    // Cleanup
    store.delete("list-test/a/1").unwrap();
    store.delete("list-test/a/2").unwrap();
    store.delete("list-test/b/3").unwrap();
}

#[test]
#[ignore]
fn test_object_store_get_nonexistent() {
    let store = make_object_store();
    let result = store.get("does-not-exist/nope/never");
    assert!(result.is_err());
}

#[test]
#[ignore]
fn test_object_store_overwrite() {
    let store = make_object_store();
    let key = "overwrite-test/record/field";

    store.put(key, b"version1").unwrap();
    assert_eq!(store.get(key).unwrap(), b"version1");

    store.put(key, b"version2").unwrap();
    assert_eq!(store.get(key).unwrap(), b"version2");

    // Cleanup
    store.delete(key).unwrap();
}

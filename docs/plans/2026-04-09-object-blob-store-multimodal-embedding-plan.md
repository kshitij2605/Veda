# Object Blob Store + Multimodal Embedding Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Wire MinIO as default blob storage, enable real multimodal embedding inference via Qwen3-VL-Embedding-2B, and add smart model auto-selection based on data modality.

**Architecture:** Replace `S3BlobStore` stub with `ObjectBlobStore` (MinIO/S3-compatible), make `EmbeddingClient::embed_bytes()` call real vLLM multimodal endpoints, and extend model registration + selection logic so users can bring their own models or let Veda auto-select.

**Tech Stack:** Rust, `rust-s3` crate, vLLM OpenAI-compatible API, axum, Docker Compose

---

### Task 1: Add `rust-s3` dependency to `veda-storage`

**Files:**
- Modify: `crates/veda-storage/Cargo.toml`

**Step 1: Add dependency**

Add `rust-s3` with blocking feature to `crates/veda-storage/Cargo.toml` under `[dependencies]`:

```toml
rust-s3 = { version = "0.35", features = ["sync", "tokio-rustls-tls"], default-features = false }
```

**Step 2: Verify it compiles**

Run: `cargo check -p veda-storage`
Expected: Compiles with no errors.

**Step 3: Commit**

```bash
git add crates/veda-storage/Cargo.toml Cargo.lock
git commit -m "chore: add rust-s3 dependency to veda-storage"
```

---

### Task 2: Implement `ObjectBlobStore`

**Files:**
- Modify: `crates/veda-storage/src/blob_store.rs` (replace `S3BlobStore` stub)

**Step 1: Write failing test**

Add to `crates/veda-storage/tests/blob_test.rs`:

```rust
// --- ObjectBlobStore config tests ---

#[test]
fn object_blob_store_config_minio_defaults() {
    let config = ObjectStoreConfig {
        backend: "minio".to_string(),
        endpoint: "http://localhost:9000".to_string(),
        bucket: "test-bucket".to_string(),
        access_key: "minioadmin".to_string(),
        secret_key: "minioadmin".to_string(),
        region: "us-east-1".to_string(),
    };
    assert_eq!(config.backend, "minio");
    assert!(!config.endpoint.is_empty());
}
```

Run: `cargo test -p veda-storage object_blob_store_config`
Expected: FAIL — `ObjectStoreConfig` not found.

**Step 2: Implement `ObjectStoreConfig` and `ObjectBlobStore`**

In `crates/veda-storage/src/blob_store.rs`, replace the `S3BlobStore` stub (lines 116-128) with:

```rust
/// Configuration for S3-compatible object storage (MinIO or AWS S3).
#[derive(Debug, Clone)]
pub struct ObjectStoreConfig {
    /// Backend type: "minio" (default) or "s3".
    pub backend: String,
    /// Endpoint URL (required for minio, e.g. "http://minio:9000").
    pub endpoint: String,
    /// Bucket name.
    pub bucket: String,
    /// Access key.
    pub access_key: String,
    /// Secret key.
    pub secret_key: String,
    /// Region (default "us-east-1").
    pub region: String,
}

/// Object storage blob store backed by MinIO or AWS S3.
pub struct ObjectBlobStore {
    bucket: s3::Bucket,
}

impl ObjectBlobStore {
    pub fn new(config: &ObjectStoreConfig) -> Result<Self> {
        let region = if config.backend == "s3" {
            s3::Region::from_str(&config.region)
                .map_err(|e| VedaError::Storage(format!("invalid S3 region: {e}")))?
        } else {
            s3::Region::Custom {
                region: config.region.clone(),
                endpoint: config.endpoint.clone(),
            }
        };

        let credentials = s3::creds::Credentials::new(
            Some(&config.access_key),
            Some(&config.secret_key),
            None, None, None,
        )
        .map_err(|e| VedaError::Storage(format!("invalid credentials: {e}")))?;

        let mut bucket = s3::Bucket::new(&config.bucket, region, credentials)
            .map_err(|e| VedaError::Storage(format!("failed to create bucket handle: {e}")))?;

        if config.backend == "minio" {
            bucket.set_path_style();
        }

        // Try to create bucket (ignore AlreadyExists).
        let _ = bucket.create_sync();

        Ok(Self { bucket })
    }
}

impl BlobStore for ObjectBlobStore {
    fn put(&self, key: &str, data: &[u8]) -> Result<()> {
        self.bucket
            .put_object_sync(key, data)
            .map_err(|e| VedaError::Storage(format!("object store put failed: {e}")))?;
        Ok(())
    }

    fn get(&self, key: &str) -> Result<Vec<u8>> {
        let response = self
            .bucket
            .get_object_sync(key)
            .map_err(|e| VedaError::Storage(format!("object store get failed: {e}")))?;
        if response.status_code() == 404 {
            return Err(VedaError::Storage(format!("Blob not found: {key}")));
        }
        Ok(response.to_vec())
    }

    fn delete(&self, key: &str) -> Result<()> {
        self.bucket
            .delete_object_sync(key)
            .map_err(|e| VedaError::Storage(format!("object store delete failed: {e}")))?;
        Ok(())
    }

    fn exists(&self, key: &str) -> Result<bool> {
        match self.bucket.head_object_sync(key) {
            Ok(_) => Ok(true),
            Err(_) => Ok(false),
        }
    }

    fn list(&self, prefix: &str) -> Result<Vec<String>> {
        let results = self
            .bucket
            .list_sync(prefix.to_string(), None)
            .map_err(|e| VedaError::Storage(format!("object store list failed: {e}")))?;
        let mut keys = Vec::new();
        for page in &results {
            for obj in &page.contents {
                keys.push(obj.key.clone());
            }
        }
        keys.sort();
        Ok(keys)
    }
}
```

Add `use std::str::FromStr;` and `use s3` to the imports at the top of the file.

**Step 3: Run test**

Run: `cargo test -p veda-storage object_blob_store_config`
Expected: PASS.

**Step 4: Commit**

```bash
git add crates/veda-storage/src/blob_store.rs crates/veda-storage/tests/blob_test.rs
git commit -m "feat: implement ObjectBlobStore backed by MinIO/S3"
```

---

### Task 3: Wire blob store into `AppState`

**Files:**
- Modify: `crates/veda-gateway/src/rest/state.rs`
- Modify: `crates/veda-gateway/src/server.rs`

**Step 1: Add `blob_store` to `AppStateInner`**

In `crates/veda-gateway/src/rest/state.rs`:

- Add import: `use veda_storage::blob_store::{BlobStore, LocalBlobStore, ObjectBlobStore, ObjectStoreConfig};`
- Add field to `AppStateInner`: `pub blob_store: Arc<dyn BlobStore>,`
- In all `AppState` constructors (`new`, `with_auth`, `with_auth_and_users`), initialize blob store with `LocalBlobStore` default:
  ```rust
  blob_store: Arc::new(
      LocalBlobStore::new(&data_dir.join("blobs")).expect("failed to create local blob store")
  ),
  ```
- Add a new method:
  ```rust
  pub fn with_blob_store(mut self, store: Arc<dyn BlobStore>) -> Self {
      // Re-create inner with new blob store.
      // Since inner is Arc, we need Arc::make_mut or reconstruct.
      // Simplest: add a setter that takes self before Arc is shared.
  }
  ```

Actually, simpler approach: add a `load_blob_store_from_env()` method similar to `load_embedding_from_env()` that replaces the blob store field. Use `RwLock<Arc<dyn BlobStore>>` instead of `Arc<dyn BlobStore>` so it can be swapped after construction.

**Step 2: Add `load_blob_store_from_env()` to `AppState`**

```rust
pub async fn load_blob_store_from_env(&self) {
    let backend = std::env::var("VEDA_BLOB_BACKEND").unwrap_or_default();
    if backend.is_empty() {
        tracing::info!("VEDA_BLOB_BACKEND not set, using local filesystem blob store");
        return;
    }

    let config = ObjectStoreConfig {
        backend: backend.clone(),
        endpoint: std::env::var("VEDA_BLOB_ENDPOINT").unwrap_or_else(|_| "http://minio:9000".into()),
        bucket: std::env::var("VEDA_BLOB_BUCKET").unwrap_or_else(|_| "veda-blobs".into()),
        access_key: std::env::var("VEDA_BLOB_ACCESS_KEY").unwrap_or_default(),
        secret_key: std::env::var("VEDA_BLOB_SECRET_KEY").unwrap_or_default(),
        region: std::env::var("VEDA_BLOB_REGION").unwrap_or_else(|_| "us-east-1".into()),
    };

    match ObjectBlobStore::new(&config) {
        Ok(store) => {
            let mut bs = self.inner.blob_store.write().await;
            *bs = Arc::new(store);
            tracing::info!(backend = %backend, endpoint = %config.endpoint, bucket = %config.bucket, "object blob store initialized");
        }
        Err(e) => {
            tracing::error!(error = %e, "failed to initialize object blob store, falling back to local");
        }
    }
}
```

**Step 3: Call from `server.rs`**

In `crates/veda-gateway/src/server.rs`, after `app_state.load_embedding_from_env().await`:

```rust
app_state.load_blob_store_from_env().await;
```

**Step 4: Verify it compiles**

Run: `cargo check -p veda-gateway`
Expected: Compiles.

**Step 5: Commit**

```bash
git add crates/veda-gateway/src/rest/state.rs crates/veda-gateway/src/server.rs
git commit -m "feat: wire ObjectBlobStore into AppState from env vars"
```

---

### Task 4: Wire blob store into insert and document handlers

**Files:**
- Modify: `crates/veda-gateway/src/rest/records.rs`
- Modify: `crates/veda-gateway/src/rest/document.rs`

**Step 1: Use `BlobManager` in insert handler**

In `records.rs` `insert()` handler, after parsing records but before inserting into segment manager, process blob fields:

```rust
let blob_store = state.inner.blob_store.read().await.clone();
let blob_manager = BlobManager::new(Box::new(ArcBlobStore(blob_store)), &name);

// For each record, process blob fields before insertion.
blob_manager.process_insert(&mut record, &schema).map_err(|e| {
    (StatusCode::INTERNAL_SERVER_ERROR, Json(ErrorResponse { error: format!("blob processing failed: {e}") }))
})?;
```

Need a thin wrapper `ArcBlobStore(Arc<dyn BlobStore>)` that implements `BlobStore` by delegating to the inner Arc. Add this to `blob_store.rs`.

**Step 2: Wire blob store in `document.rs`**

In the `process_document` handler, use the blob store from app state for storing extracted images instead of only setting `BlobRef` keys without actually storing data.

**Step 3: Verify it compiles**

Run: `cargo check -p veda-gateway`
Expected: Compiles.

**Step 4: Commit**

```bash
git add crates/veda-gateway/src/rest/records.rs crates/veda-gateway/src/rest/document.rs crates/veda-storage/src/blob_store.rs
git commit -m "feat: wire blob storage into insert and document handlers"
```

---

### Task 5: Implement `embed_bytes_http()` for multimodal embedding

**Files:**
- Modify: `crates/veda-core/src/embedding_client.rs`

**Step 1: Write failing test**

Add to `crates/veda-core/src/embedding_client.rs` (or a test file):

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedding::EmbeddingModelConfig;

    #[test]
    fn test_build_multimodal_payload_image() {
        let config = EmbeddingModelConfig {
            name: "test-model".into(),
            endpoint: "http://localhost:8000".into(),
            api_key: String::new(),
            supported_modalities: vec![Modality::Image],
            output_dimension: 2048,
            max_input_size_bytes: 10 * 1024 * 1024,
            timeout_seconds: 30,
            batch_size: 32,
        };
        let client = EmbeddingClient::new(config);
        let payload = client.build_multimodal_payload(b"fakeimagedata", Modality::Image);
        let messages = payload["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 1);
        let content = messages[0]["content"].as_array().unwrap();
        assert!(content.iter().any(|c| c["type"] == "image_url"));
    }
}
```

Run: `cargo test -p veda-core test_build_multimodal_payload`
Expected: FAIL — `build_multimodal_payload` not found.

**Step 2: Implement multimodal HTTP embedding**

In `crates/veda-core/src/embedding_client.rs`:

1. Add `use base64::prelude::*;` import.

2. Fix `is_real_endpoint()` — remove the `!self.config.endpoint.contains("localhost")` check:

```rust
fn is_real_endpoint(&self) -> bool {
    self.config.endpoint.starts_with("http") && !self.config.endpoint.contains("mock")
}
```

3. Update `embed_bytes()` to call HTTP when real endpoint:

```rust
pub fn embed_bytes(&self, data: &[u8], modality: Modality) -> Result<Vec<f32>> {
    self.validate_input(data, modality)?;
    if self.is_real_endpoint() {
        self.embed_bytes_http(data, modality)
    } else {
        Ok(deterministic_vector(data, self.config.output_dimension))
    }
}
```

4. Add `build_multimodal_payload()`:

```rust
fn build_multimodal_payload(&self, data: &[u8], modality: Modality) -> serde_json::Value {
    let b64 = BASE64_STANDARD.encode(data);
    let mime = match modality {
        Modality::Image => "image/jpeg",
        Modality::Video => "video/mp4",
        _ => "application/octet-stream",
    };
    let data_uri = format!("data:{mime};base64,{b64}");

    let content_part = match modality {
        Modality::Image => serde_json::json!({
            "type": "image_url",
            "image_url": {"url": data_uri}
        }),
        Modality::Video => serde_json::json!({
            "type": "video_url",
            "video_url": {"url": data_uri}
        }),
        _ => serde_json::json!({
            "type": "image_url",
            "image_url": {"url": data_uri}
        }),
    };

    serde_json::json!({
        "model": self.config.name,
        "messages": [{
            "role": "user",
            "content": [content_part]
        }],
        "encoding_format": "float"
    })
}
```

5. Add `embed_bytes_http()`:

```rust
fn embed_bytes_http(&self, data: &[u8], modality: Modality) -> Result<Vec<f32>> {
    let client = reqwest::blocking::Client::new();
    let body = self.build_multimodal_payload(data, modality);
    let url = format!("{}/v1/embeddings", self.config.endpoint.trim_end_matches('/'));

    let mut request = client
        .post(&url)
        .json(&body)
        .timeout(std::time::Duration::from_secs(self.config.timeout_seconds as u64));

    if !self.config.api_key.is_empty() {
        request = request.bearer_auth(&self.config.api_key);
    }

    let resp = request
        .send()
        .map_err(|e| VedaError::Storage(format!("multimodal embedding request failed: {e}")))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().unwrap_or_default();
        return Err(VedaError::Storage(format!(
            "multimodal embedding endpoint returned {status}: {body}"
        )));
    }

    let json: serde_json::Value = resp
        .json()
        .map_err(|e| VedaError::Storage(format!("embedding response parse failed: {e}")))?;

    let embedding = json["data"][0]["embedding"]
        .as_array()
        .ok_or_else(|| VedaError::Storage("invalid embedding response format".into()))?
        .iter()
        .map(|v| v.as_f64().unwrap_or(0.0) as f32)
        .collect();

    Ok(embedding)
}
```

6. Add `base64` to `crates/veda-core/Cargo.toml` dependencies:

```toml
base64 = "0.22"
```

**Step 3: Run tests**

Run: `cargo test -p veda-core test_build_multimodal_payload`
Expected: PASS.

**Step 4: Commit**

```bash
git add crates/veda-core/src/embedding_client.rs crates/veda-core/Cargo.toml
git commit -m "feat: implement real HTTP multimodal embedding for images and video"
```

---

### Task 6: Extend model registration for multimodal models

**Files:**
- Modify: `crates/veda-gateway/src/rest/state.rs`

**Step 1: Add multimodal model env var loading**

In `load_embedding_from_env()`, after registering the text model, add:

```rust
// Register multimodal embedding model if configured.
let mm_endpoint = match std::env::var("MULTIMODAL_EMBEDDING_ENDPOINT").ok() {
    Some(ep) if !ep.is_empty() => ep,
    _ => return,
};
let mm_model = match std::env::var("MULTIMODAL_EMBEDDING_MODEL").ok() {
    Some(m) if !m.is_empty() => m,
    _ => return,
};
let mm_api_key = std::env::var("MULTIMODAL_EMBEDDING_API_KEY").unwrap_or_default();
let mm_dim = std::env::var("MULTIMODAL_EMBEDDING_DIMENSION")
    .ok()
    .and_then(|s| s.parse().ok())
    .unwrap_or(2048);

let mm_config = EmbeddingModelConfig {
    name: mm_model.clone(),
    endpoint: mm_endpoint.clone(),
    api_key: mm_api_key,
    supported_modalities: vec![Modality::Text, Modality::Image, Modality::Video],
    output_dimension: mm_dim,
    max_input_size_bytes: 20 * 1024 * 1024,
    timeout_seconds: 60,
    batch_size: 8,
};

registry.register(mm_config);
tracing::info!(model = %mm_model, endpoint = %mm_endpoint, dimension = mm_dim, "registered multimodal embedding model from environment");
```

Note: The `registry` write lock is already held from the text model registration. Restructure the function so both registrations happen under the same lock, with the text model early-return changed to just skipping (not returning) so multimodal model can still be registered independently.

**Step 2: Verify it compiles**

Run: `cargo check -p veda-gateway`
Expected: Compiles.

**Step 3: Commit**

```bash
git add crates/veda-gateway/src/rest/state.rs
git commit -m "feat: register multimodal embedding model from env vars"
```

---

### Task 7: Auto-select model when user omits it in `embed-insert`

**Files:**
- Modify: `crates/veda-gateway/src/rest/embed_insert.rs`

**Step 1: Make `model` field optional in `AutoEmbedMapping`**

```rust
#[derive(Debug, Deserialize)]
pub struct AutoEmbedMapping {
    pub source_field: String,
    pub model: Option<String>,  // None = auto-select by modality
}
```

**Step 2: Add auto-selection logic in validation phase**

Replace the model lookup (lines 66-75) with:

```rust
let modality = Modality::from_field_name(&mapping.source_field).unwrap_or(Modality::Text);

let model_name = match &mapping.model {
    Some(name) => name.clone(),
    None => {
        // Auto-select: find first model supporting this modality.
        let candidates = registry.find_by_modality(modality);
        if candidates.is_empty() {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: format!(
                        "no model registered for modality '{modality}'. Register one via EMBEDDING_* or MULTIMODAL_EMBEDDING_* env vars"
                    ),
                }),
            ));
        }
        candidates[0].name.clone()
    }
};

let model = registry.get(&model_name).ok_or_else(|| {
    (StatusCode::BAD_REQUEST, Json(ErrorResponse {
        error: format!("model '{model_name}' not found in registry"),
    }))
})?;
```

Update all downstream references from `mapping.model` to `model_name`.

**Step 3: Run existing embed-insert tests**

Run: `cargo test -p veda-gateway embed_insert`
Expected: Existing tests still pass (they all specify `model` explicitly).

**Step 4: Commit**

```bash
git add crates/veda-gateway/src/rest/embed_insert.rs
git commit -m "feat: auto-select embedding model by modality when user omits model"
```

---

### Task 8: Add `multimodal-embedding` service to Docker Compose

**Files:**
- Modify: `docker-compose.yml`
- Modify: `.env`

**Step 1: Add service to `docker-compose.yml`**

Add after the `embedding` service:

```yaml
  # --- Multimodal Embedding Service (optional, enabled with --profile embedding) ---
  # Qwen3-VL-Embedding-2B served via vLLM for image/video embedding.
  # Default: Qwen/Qwen3-VL-Embedding-2B (~4GB GPU VRAM).
  # Users can substitute any vLLM-compatible multimodal embedding model.
  multimodal-embedding:
    profiles: ["embedding"]
    image: vllm/vllm-openai:nightly
    ports:
      - "8094:8000"
    environment:
      HF_TOKEN: ${HF_TOKEN:-}
      NVIDIA_VISIBLE_DEVICES: ${MULTIMODAL_EMBEDDING_GPU:-3}
      NO_PROXY: localhost,127.0.0.1,glm-ocr,vlm,minio,nats,veda,flashparse,embedding,multimodal-embedding
      no_proxy: localhost,127.0.0.1,glm-ocr,vlm,minio,nats,veda,flashparse,embedding,multimodal-embedding
    volumes:
      - multimodal-embedding-model-cache:/root/.cache/huggingface
    command:
      - "--model"
      - "${MULTIMODAL_EMBEDDING_MODEL:-Qwen/Qwen3-VL-Embedding-2B}"
      - "--runner"
      - "pooling"
      - "--port"
      - "8000"
      - "--served-model-name"
      - "${MULTIMODAL_EMBEDDING_MODEL:-Qwen/Qwen3-VL-Embedding-2B}"
      - "--gpu-memory-utilization"
      - "0.50"
      - "--trust-remote-code"
      - "--max-model-len"
      - "8192"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ["${MULTIMODAL_EMBEDDING_GPU:-3}"]
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "curl", "--noproxy", "*", "-f", "http://localhost:8000/health"]
      interval: 10s
      timeout: 5s
      retries: 60
      start_period: 10s
```

Add `multimodal-embedding-model-cache:` to the `volumes:` section.

Add blob store env vars to the `veda` service:

```yaml
      VEDA_BLOB_BACKEND: minio
      VEDA_BLOB_ENDPOINT: http://minio:9000
      VEDA_BLOB_BUCKET: veda-blobs
      VEDA_BLOB_ACCESS_KEY: veda
      VEDA_BLOB_SECRET_KEY: ${MINIO_PASSWORD:-vedapassword}
      MULTIMODAL_EMBEDDING_ENDPOINT: ${MULTIMODAL_EMBEDDING_ENDPOINT:-http://multimodal-embedding:8000}
      MULTIMODAL_EMBEDDING_MODEL: ${MULTIMODAL_EMBEDDING_MODEL:-Qwen/Qwen3-VL-Embedding-2B}
      MULTIMODAL_EMBEDDING_DIMENSION: ${MULTIMODAL_EMBEDDING_DIMENSION:-2048}
```

**Step 2: Add env vars to `.env`**

```
# Multimodal Embedding Model (vLLM serving Qwen3-VL-Embedding-2B)
MULTIMODAL_EMBEDDING_ENDPOINT=http://multimodal-embedding:8000
MULTIMODAL_EMBEDDING_MODEL=Qwen/Qwen3-VL-Embedding-2B
MULTIMODAL_EMBEDDING_DIMENSION=2048
MULTIMODAL_EMBEDDING_GPU=3
```

**Step 3: Validate compose config**

Run: `docker compose config --profiles embedding`
Expected: No errors, both `embedding` and `multimodal-embedding` services shown.

**Step 4: Commit**

```bash
git add docker-compose.yml .env
git commit -m "feat: add multimodal-embedding service and MinIO blob store config"
```

---

### Task 9: Integration test — object blob store with running MinIO

**Files:**
- Create: `crates/veda-storage/tests/object_blob_integration_test.rs`

**Step 1: Write integration test**

```rust
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
```

**Step 2: Run against live MinIO**

Run: `cargo test -p veda-storage --test object_blob_integration_test -- --ignored`
Expected: PASS (MinIO running on port 9200 from Docker Compose).

**Step 3: Commit**

```bash
git add crates/veda-storage/tests/object_blob_integration_test.rs
git commit -m "test: add integration tests for ObjectBlobStore against MinIO"
```

---

### Task 10: End-to-end test — multimodal embed + blob store

**Step 1: Deploy full stack with new multimodal-embedding service**

```bash
docker compose --profile ocr --profile embedding --profile ocr-vlm up -d
```

Wait for all services to be healthy.

**Step 2: Test blob storage via Veda API**

Create a collection with a blob field, insert a record with base64 image data, verify it's stored in MinIO.

**Step 3: Test multimodal embedding**

Create a collection with both `text_embedding` (1024-dim) and `image_embedding` (2048-dim) vector fields. Use `embed-insert` with auto-select (omit model) to embed text and image data. Verify:
- Text chunks go to BGE-M3 (1024-dim)
- Images go to Qwen3-VL (2048-dim)
- Search works on both fields
- Multi-vector fusion across text and image embeddings works

**Step 4: Commit any fixes**

```bash
git commit -m "fix: address issues found in e2e multimodal testing"
```

---

## Execution Order

Tasks 1-2 (blob store) and Task 5 (embedding client) are independent and can be parallelized.

```
Task 1 → Task 2 → Task 3 → Task 4 → Task 9
                                         ↘
Task 5 → Task 6 → Task 7 ──────────→ Task 8 → Task 10
```

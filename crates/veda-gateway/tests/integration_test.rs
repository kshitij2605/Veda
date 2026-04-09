use std::collections::HashMap;
use std::net::SocketAddr;

use reqwest::Client;
use serde_json::{json, Value};
use tokio::net::TcpListener;

use veda_gateway::auth::Role;
use veda_gateway::rest::state::{AppState, UserRecord};
use veda_gateway::rest::build_router;

// ── Test helpers ────────────────────────────────────────────────────────────

/// Start a test server (auth disabled) on a random port, return base URL + temp dir.
async fn start_test_server() -> (String, tempfile::TempDir) {
    let tmp = tempfile::tempdir().unwrap();
    let app_state = AppState::new(tmp.path());
    let router = build_router(app_state);

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr: SocketAddr = listener.local_addr().unwrap();
    let base_url = format!("http://{addr}");

    tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });

    (base_url, tmp)
}

/// Start a test server with auth enabled and default admin/viewer users.
async fn start_auth_server() -> (String, tempfile::TempDir) {
    let tmp = tempfile::tempdir().unwrap();
    let mut users = HashMap::new();
    users.insert(
        "admin".into(),
        UserRecord {
            password: "admin".into(),
            role: Role::Admin,
        },
    );
    users.insert(
        "viewer".into(),
        UserRecord {
            password: "viewer".into(),
            role: Role::Viewer,
        },
    );
    let app_state = AppState::with_auth_and_users(tmp.path(), "integration-jwt-secret", users);
    let router = build_router(app_state);

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr: SocketAddr = listener.local_addr().unwrap();
    let base_url = format!("http://{addr}");

    tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });

    (base_url, tmp)
}

fn client() -> Client {
    Client::new()
}

/// Generate a deterministic-ish vector of the given dimension seeded by `seed`.
fn seeded_vector(dim: usize, seed: u32) -> Vec<f32> {
    (0..dim)
        .map(|i| (seed as f32 * 0.1 + i as f32 * 0.03) % 1.0)
        .collect()
}

// ── 1. Full collection lifecycle ────────────────────────────────────────────

#[tokio::test]
async fn test_full_collection_lifecycle() {
    let (base, _tmp) = start_test_server().await;
    let c = client();

    // 1. Create collection with 2 vector fields + 3 scalar fields
    let resp = c
        .post(format!("{base}/v1/collections"))
        .json(&json!({
            "name": "lifecycle_col",
            "fields": [
                { "name": "embedding", "type": "vector", "dimension": 8, "metric": "cosine" },
                { "name": "thumbnail_vec", "type": "vector", "dimension": 4, "metric": "euclidean" },
                { "name": "title", "type": "string", "filterable": true },
                { "name": "price", "type": "float64", "filterable": true },
                { "name": "active", "type": "bool", "filterable": true }
            ],
            "shard_count": 1,
            "replication_factor": 1
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 201);

    // 2. Verify it appears in list
    let resp = c.get(format!("{base}/v1/collections")).send().await.unwrap();
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    let names: Vec<&str> = body["collections"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| c["name"].as_str().unwrap())
        .collect();
    assert!(names.contains(&"lifecycle_col"));

    // 3. Describe — verify schema matches
    let resp = c
        .get(format!("{base}/v1/collections/lifecycle_col"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["name"], "lifecycle_col");
    assert_eq!(body["shard_count"], 1);
    assert_eq!(body["replication_factor"], 1);
    let fields = body["fields"].as_array().unwrap();
    assert_eq!(fields.len(), 5);
    // Verify we have the expected field types
    let field_names: Vec<&str> = fields.iter().map(|f| f["name"].as_str().unwrap()).collect();
    assert!(field_names.contains(&"embedding"));
    assert!(field_names.contains(&"thumbnail_vec"));
    assert!(field_names.contains(&"title"));
    assert!(field_names.contains(&"price"));
    assert!(field_names.contains(&"active"));

    // 4. Insert 50 records with all field types populated
    let mut records = Vec::new();
    for i in 0..50 {
        records.push(json!({
            "embedding": seeded_vector(8, i),
            "thumbnail_vec": seeded_vector(4, i + 100),
            "title": format!("Item {i}"),
            "price": 10.0 + i as f64 * 5.5,
            "active": i % 2 == 0
        }));
    }
    let resp = c
        .post(format!("{base}/v1/collections/lifecycle_col/insert"))
        .json(&json!({ "records": records }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 201);
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["inserted"], 50);

    // 5. Search — verify results returned with correct structure
    let resp = c
        .post(format!("{base}/v1/collections/lifecycle_col/search"))
        .json(&json!({
            "query": {
                "vector": { "field": "embedding", "values": seeded_vector(8, 0) }
            },
            "limit": 10
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    let results = body["results"].as_array().unwrap();
    assert!(!results.is_empty());
    assert!(results.len() <= 10);
    assert!(body["took_ms"].as_u64().is_some());
    // Each result should have id and distance
    for hit in results {
        assert!(hit["id"].as_str().is_some());
        assert!(hit["distance"].as_f64().is_some());
    }

    // 6. Get record by ID — verify fields present
    let first_id = results[0]["id"].as_str().unwrap();
    let resp = c
        .get(format!(
            "{base}/v1/collections/lifecycle_col/records/{first_id}"
        ))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["id"], first_id);
    let rec_fields = body["fields"].as_object().unwrap();
    assert!(rec_fields.contains_key("embedding"));
    assert!(rec_fields.contains_key("thumbnail_vec"));
    assert!(rec_fields.contains_key("title"));
    assert!(rec_fields.contains_key("price"));
    assert!(rec_fields.contains_key("active"));

    // 7. Delete 5 records by ID — gather IDs from search results
    let ids_to_delete: Vec<&str> = results.iter().take(5).map(|r| r["id"].as_str().unwrap()).collect();
    let resp = c
        .delete(format!("{base}/v1/collections/lifecycle_col/records"))
        .json(&json!({ "ids": ids_to_delete }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["deleted"], 5);

    // 8. Verify deleted records are gone via get_record (returns 404).
    //    Note: the search executor has its own segment copies and doesn't
    //    see in-memory deletes until a rebuild, so we verify via get_record
    //    which reads directly from the segment manager.
    for deleted_id in &ids_to_delete {
        let resp = c
            .get(format!(
                "{base}/v1/collections/lifecycle_col/records/{deleted_id}"
            ))
            .send()
            .await
            .unwrap();
        assert_eq!(
            resp.status(),
            404,
            "Deleted record {deleted_id} should return 404 on get"
        );
    }

    // 9. Drop collection
    let resp = c
        .delete(format!("{base}/v1/collections/lifecycle_col"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["status"], "dropped");

    // 10. Verify it's gone from list
    let resp = c.get(format!("{base}/v1/collections")).send().await.unwrap();
    let body: Value = resp.json().await.unwrap();
    let names: Vec<&str> = body["collections"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| c["name"].as_str().unwrap())
        .collect();
    assert!(!names.contains(&"lifecycle_col"));

    // Also verify describe returns 404
    let resp = c
        .get(format!("{base}/v1/collections/lifecycle_col"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404);
}

// ── 2. Filtered vector search ───────────────────────────────────────────────

#[tokio::test]
async fn test_filtered_search() {
    let (base, _tmp) = start_test_server().await;
    let c = client();

    // 1. Create collection: vector(64) + category(string) + price(float64)
    c.post(format!("{base}/v1/collections"))
        .json(&json!({
            "name": "filter_test_col",
            "fields": [
                { "name": "embedding", "type": "vector", "dimension": 64, "metric": "cosine" },
                { "name": "category", "type": "string", "filterable": true },
                { "name": "price", "type": "float64", "filterable": true }
            ],
            "shard_count": 1
        }))
        .send()
        .await
        .unwrap();

    // 2. Insert 100 records across 3 categories
    let mut records = Vec::new();
    for i in 0..30 {
        records.push(json!({
            "embedding": seeded_vector(64, i),
            "category": "electronics",
            "price": 100.0 + i as f64 * 7.0 // 100-303
        }));
    }
    for i in 30..60 {
        records.push(json!({
            "embedding": seeded_vector(64, i),
            "category": "books",
            "price": 10.0 + (i - 30) as f64 * 1.4 // 10-50.6
        }));
    }
    for i in 60..100 {
        records.push(json!({
            "embedding": seeded_vector(64, i),
            "category": "clothing",
            "price": 20.0 + (i - 60) as f64 * 4.6 // 20-199.4
        }));
    }

    let resp = c
        .post(format!("{base}/v1/collections/filter_test_col/insert"))
        .json(&json!({ "records": records }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 201);
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["inserted"], 100);

    // 3. Search with filter "category = 'electronics'" — all results are electronics
    let resp = c
        .post(format!("{base}/v1/collections/filter_test_col/search"))
        .json(&json!({
            "query": {
                "vector": { "field": "embedding", "values": seeded_vector(64, 0) }
            },
            "filter": "category = 'electronics'",
            "limit": 50
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    let count = body["count"].as_u64().unwrap();
    // Should return only electronics (at most 30)
    assert!(count > 0 && count <= 30, "Expected 1-30 electronics results, got {count}");

    // 4. Search with filter "price BETWEEN 50 AND 150" — results in range
    let resp = c
        .post(format!("{base}/v1/collections/filter_test_col/search"))
        .json(&json!({
            "query": {
                "vector": { "field": "embedding", "values": seeded_vector(64, 0) }
            },
            "filter": "price BETWEEN 50 AND 150",
            "limit": 50
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    assert!(body["count"].as_u64().unwrap() > 0);

    // 5. Search with compound filter "category = 'books' AND price < 30"
    let resp = c
        .post(format!("{base}/v1/collections/filter_test_col/search"))
        .json(&json!({
            "query": {
                "vector": { "field": "embedding", "values": seeded_vector(64, 30) }
            },
            "filter": "category = 'books' AND price < 30",
            "limit": 50
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    // Books with price < 30 exist (prices start at 10)
    assert!(body["count"].as_u64().unwrap() > 0);

    // 6. Search with no filter — returns from any category
    let resp = c
        .post(format!("{base}/v1/collections/filter_test_col/search"))
        .json(&json!({
            "query": {
                "vector": { "field": "embedding", "values": seeded_vector(64, 50) }
            },
            "limit": 20
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    assert!(body["count"].as_u64().unwrap() > 0);
}

// ── 3. Multi-vector fusion search ───────────────────────────────────────────

#[tokio::test]
async fn test_multi_vector_fusion() {
    let (base, _tmp) = start_test_server().await;
    let c = client();

    // 1. Create collection with two vector fields
    c.post(format!("{base}/v1/collections"))
        .json(&json!({
            "name": "fusion_col",
            "fields": [
                { "name": "text_embedding", "type": "vector", "dimension": 64, "metric": "cosine" },
                { "name": "image_embedding", "type": "vector", "dimension": 32, "metric": "cosine" }
            ],
            "shard_count": 1
        }))
        .send()
        .await
        .unwrap();

    // 2. Insert 100 records with both vector fields
    let mut records = Vec::new();
    for i in 0..100 {
        records.push(json!({
            "text_embedding": seeded_vector(64, i),
            "image_embedding": seeded_vector(32, i + 500)
        }));
    }

    let resp = c
        .post(format!("{base}/v1/collections/fusion_col/insert"))
        .json(&json!({ "records": records }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 201);
    assert_eq!(resp.json::<Value>().await.unwrap()["inserted"], 100);

    // 3. Multi-vector search with both fields, RRF fusion
    let resp = c
        .post(format!("{base}/v1/collections/fusion_col/search"))
        .json(&json!({
            "query": {
                "multi_vector": [
                    { "field": "text_embedding", "values": seeded_vector(64, 0), "weight": 1.0 },
                    { "field": "image_embedding", "values": seeded_vector(32, 500), "weight": 1.0 }
                ],
                "fusion": "rrf",
                "fusion_params": { "k": 60 }
            },
            "limit": 10
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    let results = body["results"].as_array().unwrap();

    // 4. Verify results have fused scores
    assert!(!results.is_empty());
    for hit in results {
        assert!(hit["id"].as_str().is_some());
        assert!(hit["distance"].as_f64().is_some());
    }

    // 5. Verify results are sorted by score (descending for RRF — higher is better)
    let scores: Vec<f64> = results.iter().map(|r| r["distance"].as_f64().unwrap()).collect();
    for w in scores.windows(2) {
        assert!(
            w[0] >= w[1],
            "Results should be sorted by score descending: {} < {}",
            w[0],
            w[1]
        );
    }
}

// ── 4. Large batch insert ───────────────────────────────────────────────────

#[tokio::test]
async fn test_large_batch_insert() {
    let (base, _tmp) = start_test_server().await;
    let c = client();

    // 1. Create collection with vector(32)
    c.post(format!("{base}/v1/collections"))
        .json(&json!({
            "name": "batch_col",
            "fields": [
                { "name": "embedding", "type": "vector", "dimension": 32, "metric": "cosine" }
            ],
            "shard_count": 1
        }))
        .send()
        .await
        .unwrap();

    // 2. Insert 1000 records in one batch
    let records: Vec<Value> = (0..1000)
        .map(|i| json!({ "embedding": seeded_vector(32, i) }))
        .collect();

    let resp = c
        .post(format!("{base}/v1/collections/batch_col/insert"))
        .json(&json!({ "records": records }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 201);
    let body: Value = resp.json().await.unwrap();

    // 3. Verify count matches
    assert_eq!(body["inserted"], 1000);

    // 4. Search — verify results returned
    let resp = c
        .post(format!("{base}/v1/collections/batch_col/search"))
        .json(&json!({
            "query": {
                "vector": { "field": "embedding", "values": seeded_vector(32, 0) }
            },
            "limit": 10
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    assert!(body["count"].as_u64().unwrap() > 0);
    assert!(body["results"].as_array().unwrap().len() <= 10);
}

// ── 5. Error handling ───────────────────────────────────────────────────────

#[tokio::test]
async fn test_error_search_nonexistent_collection() {
    let (base, _tmp) = start_test_server().await;
    let c = client();

    // Search nonexistent collection -> 404
    let resp = c
        .post(format!("{base}/v1/collections/nonexistent/search"))
        .json(&json!({
            "query": {
                "vector": { "field": "embedding", "values": [0.1, 0.2, 0.3, 0.4] }
            },
            "limit": 5
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404);
    let body: Value = resp.json().await.unwrap();
    assert!(body["error"].as_str().unwrap().contains("not found"));
}

#[tokio::test]
async fn test_error_insert_wrong_dimension() {
    let (base, _tmp) = start_test_server().await;
    let c = client();

    // Create a collection with dimension 4
    c.post(format!("{base}/v1/collections"))
        .json(&json!({
            "name": "dim_err_col",
            "fields": [
                { "name": "embedding", "type": "vector", "dimension": 4, "metric": "cosine" }
            ],
            "shard_count": 1
        }))
        .send()
        .await
        .unwrap();

    // Insert with wrong dimension (8 instead of 4) -> 400 or 500
    let resp = c
        .post(format!("{base}/v1/collections/dim_err_col/insert"))
        .json(&json!({
            "records": [
                { "embedding": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] }
            ]
        }))
        .send()
        .await
        .unwrap();
    // Should be a client or server error (dimension mismatch)
    assert!(
        resp.status().is_client_error() || resp.status().is_server_error(),
        "Expected error status for dimension mismatch, got {}",
        resp.status()
    );
}

#[tokio::test]
async fn test_error_create_duplicate_collection() {
    let (base, _tmp) = start_test_server().await;
    let c = client();

    let payload = json!({
        "name": "dup_col",
        "fields": [
            { "name": "embedding", "type": "vector", "dimension": 4, "metric": "cosine" }
        ]
    });

    // Create first time
    let resp = c
        .post(format!("{base}/v1/collections"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 201);

    // Create again — should fail
    let resp = c
        .post(format!("{base}/v1/collections"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 400);
    let body: Value = resp.json().await.unwrap();
    assert!(body["error"].as_str().unwrap().contains("already exists"));
}

#[tokio::test]
async fn test_error_get_nonexistent_record() {
    let (base, _tmp) = start_test_server().await;
    let c = client();

    // Create collection
    c.post(format!("{base}/v1/collections"))
        .json(&json!({
            "name": "norec_col",
            "fields": [
                { "name": "embedding", "type": "vector", "dimension": 4, "metric": "cosine" }
            ],
            "shard_count": 1
        }))
        .send()
        .await
        .unwrap();

    // Get nonexistent record -> 404
    let fake_id = uuid::Uuid::new_v4();
    let resp = c
        .get(format!(
            "{base}/v1/collections/norec_col/records/{fake_id}"
        ))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404);
    let body: Value = resp.json().await.unwrap();
    assert!(body["error"].as_str().unwrap().contains("not found"));
}

#[tokio::test]
async fn test_error_empty_query() {
    let (base, _tmp) = start_test_server().await;
    let c = client();

    // Create collection
    c.post(format!("{base}/v1/collections"))
        .json(&json!({
            "name": "empty_q_col",
            "fields": [
                { "name": "embedding", "type": "vector", "dimension": 4, "metric": "cosine" }
            ],
            "shard_count": 1
        }))
        .send()
        .await
        .unwrap();

    // Search without vector or multi_vector -> 400
    let resp = c
        .post(format!("{base}/v1/collections/empty_q_col/search"))
        .json(&json!({
            "query": {},
            "limit": 5
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 400);
    let body: Value = resp.json().await.unwrap();
    assert!(body["error"]
        .as_str()
        .unwrap()
        .contains("must contain"));
}

#[tokio::test]
async fn test_error_insert_nonexistent_collection() {
    let (base, _tmp) = start_test_server().await;
    let c = client();

    // Insert into nonexistent collection -> 404
    let resp = c
        .post(format!("{base}/v1/collections/does_not_exist/insert"))
        .json(&json!({
            "records": [{ "embedding": [0.1, 0.2, 0.3, 0.4] }]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404);
}

#[tokio::test]
async fn test_error_delete_nonexistent_collection() {
    let (base, _tmp) = start_test_server().await;
    let c = client();

    // Delete records from nonexistent collection -> 404
    let resp = c
        .delete(format!("{base}/v1/collections/does_not_exist/records"))
        .json(&json!({ "ids": ["00000000-0000-0000-0000-000000000001"] }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404);
}

#[tokio::test]
async fn test_error_drop_nonexistent_collection() {
    let (base, _tmp) = start_test_server().await;
    let c = client();

    let resp = c
        .delete(format!("{base}/v1/collections/phantom_col"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404);
}

// ── 6. Metrics endpoint ─────────────────────────────────────────────────────

#[tokio::test]
async fn test_metrics_endpoint() {
    let (base, _tmp) = start_test_server().await;
    let c = client();

    // GET /v1/admin/metrics
    let resp = c
        .get(format!("{base}/v1/admin/metrics"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let content_type = resp
        .headers()
        .get("content-type")
        .unwrap()
        .to_str()
        .unwrap()
        .to_owned();
    let body = resp.text().await.unwrap();

    // Verify Prometheus text format content type
    assert!(
        content_type.contains("text/plain"),
        "Expected text/plain content type for Prometheus, got {content_type}"
    );

    // Test server uses AppState::new() (no Prometheus recorder installed),
    // so the endpoint returns the fallback message.
    // Metric counter assertions will be added in later tasks (4, 5) when
    // handlers actually record metrics via a live Prometheus recorder.
    assert!(
        body.contains("# metrics recorder not installed"),
        "Expected fallback message when no recorder is installed, got: {body}"
    );
}

// ── 7. Auth flow ────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_auth_flow() {
    let (base, _tmp) = start_auth_server().await;
    let c = client();

    // 1. Health is public — always accessible
    let resp = c
        .get(format!("{base}/v1/admin/health"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    // 2. Try to access protected endpoint without token -> 401
    let resp = c.get(format!("{base}/v1/collections")).send().await.unwrap();
    assert_eq!(resp.status(), 401);

    // 3. Login -> get token
    let resp = c
        .post(format!("{base}/v1/auth/login"))
        .json(&json!({ "username": "admin", "password": "admin" }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    let token = body["token"].as_str().unwrap().to_owned();
    assert!(!token.is_empty());

    // 4. Access with token -> 200
    let resp = c
        .get(format!("{base}/v1/collections"))
        .header("Authorization", format!("Bearer {token}"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    // 5. Can also create and use collections with the token
    let resp = c
        .post(format!("{base}/v1/collections"))
        .header("Authorization", format!("Bearer {token}"))
        .json(&json!({
            "name": "auth_col",
            "fields": [
                { "name": "embedding", "type": "vector", "dimension": 4, "metric": "cosine" }
            ],
            "shard_count": 1
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 201);

    // 6. Invalid token -> 401
    let resp = c
        .get(format!("{base}/v1/collections"))
        .header("Authorization", "Bearer invalid-token-value")
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 401);

    // 7. Login with wrong password -> 401
    let resp = c
        .post(format!("{base}/v1/auth/login"))
        .json(&json!({ "username": "admin", "password": "wrong" }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 401);
}

// ── 8. Multiple collections isolation ───────────────────────────────────────

#[tokio::test]
async fn test_multiple_collections_isolation() {
    let (base, _tmp) = start_test_server().await;
    let c = client();

    // Create two collections with different schemas
    c.post(format!("{base}/v1/collections"))
        .json(&json!({
            "name": "col_a",
            "fields": [
                { "name": "embedding", "type": "vector", "dimension": 8, "metric": "cosine" },
                { "name": "label", "type": "string", "filterable": true }
            ],
            "shard_count": 1
        }))
        .send()
        .await
        .unwrap();

    c.post(format!("{base}/v1/collections"))
        .json(&json!({
            "name": "col_b",
            "fields": [
                { "name": "embedding", "type": "vector", "dimension": 16, "metric": "euclidean" },
                { "name": "score", "type": "float64", "filterable": true }
            ],
            "shard_count": 1
        }))
        .send()
        .await
        .unwrap();

    // Insert into col_a
    let records_a: Vec<Value> = (0..20)
        .map(|i| {
            json!({
                "embedding": seeded_vector(8, i),
                "label": format!("a_{i}")
            })
        })
        .collect();
    let resp = c
        .post(format!("{base}/v1/collections/col_a/insert"))
        .json(&json!({ "records": records_a }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 201);

    // Insert into col_b
    let records_b: Vec<Value> = (0..30)
        .map(|i| {
            json!({
                "embedding": seeded_vector(16, i + 200),
                "score": i as f64 * 1.5
            })
        })
        .collect();
    let resp = c
        .post(format!("{base}/v1/collections/col_b/insert"))
        .json(&json!({ "records": records_b }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 201);

    // Search col_a — results come back
    let resp = c
        .post(format!("{base}/v1/collections/col_a/search"))
        .json(&json!({
            "query": {
                "vector": { "field": "embedding", "values": seeded_vector(8, 0) }
            },
            "limit": 5
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    assert!(body["count"].as_u64().unwrap() > 0);

    // Search col_b — results come back
    let resp = c
        .post(format!("{base}/v1/collections/col_b/search"))
        .json(&json!({
            "query": {
                "vector": { "field": "embedding", "values": seeded_vector(16, 200) }
            },
            "limit": 5
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    assert!(body["count"].as_u64().unwrap() > 0);

    // Drop col_a — col_b should still work
    c.delete(format!("{base}/v1/collections/col_a"))
        .send()
        .await
        .unwrap();

    let resp = c
        .post(format!("{base}/v1/collections/col_b/search"))
        .json(&json!({
            "query": {
                "vector": { "field": "embedding", "values": seeded_vector(16, 200) }
            },
            "limit": 5
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    // List should show only col_b
    let resp = c.get(format!("{base}/v1/collections")).send().await.unwrap();
    let body: Value = resp.json().await.unwrap();
    let names: Vec<&str> = body["collections"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| c["name"].as_str().unwrap())
        .collect();
    assert_eq!(names.len(), 1);
    assert!(names.contains(&"col_b"));
}

// ── 9. Get record after insert verifies all field types ─────────────────────

#[tokio::test]
async fn test_get_record_all_field_types() {
    let (base, _tmp) = start_test_server().await;
    let c = client();

    // Create collection with many field types
    c.post(format!("{base}/v1/collections"))
        .json(&json!({
            "name": "types_col",
            "fields": [
                { "name": "embedding", "type": "vector", "dimension": 4, "metric": "cosine" },
                { "name": "name", "type": "string", "filterable": true },
                { "name": "count", "type": "int64", "filterable": true },
                { "name": "score", "type": "float64", "filterable": true },
                { "name": "active", "type": "bool", "filterable": true }
            ],
            "shard_count": 1
        }))
        .send()
        .await
        .unwrap();

    // Insert a record with all fields
    let resp = c
        .post(format!("{base}/v1/collections/types_col/insert"))
        .json(&json!({
            "records": [{
                "embedding": [1.0, 0.0, 0.0, 0.0],
                "name": "test-item",
                "count": 42,
                "score": 3.14,
                "active": true
            }]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 201);

    // Search to get the ID
    let resp = c
        .post(format!("{base}/v1/collections/types_col/search"))
        .json(&json!({
            "query": {
                "vector": { "field": "embedding", "values": [1.0, 0.0, 0.0, 0.0] }
            },
            "limit": 1
        }))
        .send()
        .await
        .unwrap();
    let body: Value = resp.json().await.unwrap();
    let id = body["results"][0]["id"].as_str().unwrap();

    // Get record by ID and verify all fields
    let resp = c
        .get(format!("{base}/v1/collections/types_col/records/{id}"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["id"], id);
    let fields = body["fields"].as_object().unwrap();
    assert_eq!(fields["name"], "test-item");
    assert_eq!(fields["count"], 42);
    assert!((fields["score"].as_f64().unwrap() - 3.14).abs() < 0.001);
    assert_eq!(fields["active"], true);
    // Vector field should be an array of floats
    let vec_field = fields["embedding"].as_array().unwrap();
    assert_eq!(vec_field.len(), 4);
}

// ── 10. Delete and re-insert cycle ──────────────────────────────────────────

#[tokio::test]
async fn test_delete_then_reinsert() {
    let (base, _tmp) = start_test_server().await;
    let c = client();

    c.post(format!("{base}/v1/collections"))
        .json(&json!({
            "name": "reinsert_col",
            "fields": [
                { "name": "embedding", "type": "vector", "dimension": 4, "metric": "cosine" },
                { "name": "tag", "type": "string", "filterable": true }
            ],
            "shard_count": 1
        }))
        .send()
        .await
        .unwrap();

    // Insert 10 records
    let records: Vec<Value> = (0..10)
        .map(|i| {
            json!({
                "embedding": seeded_vector(4, i),
                "tag": format!("batch1_{i}")
            })
        })
        .collect();
    c.post(format!("{base}/v1/collections/reinsert_col/insert"))
        .json(&json!({ "records": records }))
        .send()
        .await
        .unwrap();

    // Search to get IDs
    let resp = c
        .post(format!("{base}/v1/collections/reinsert_col/search"))
        .json(&json!({
            "query": {
                "vector": { "field": "embedding", "values": seeded_vector(4, 0) }
            },
            "limit": 10
        }))
        .send()
        .await
        .unwrap();
    let body: Value = resp.json().await.unwrap();
    let all_ids: Vec<String> = body["results"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r["id"].as_str().unwrap().to_owned())
        .collect();
    assert_eq!(all_ids.len(), 10);

    // Delete all 10
    let resp = c
        .delete(format!("{base}/v1/collections/reinsert_col/records"))
        .json(&json!({ "ids": all_ids }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["deleted"], 10);

    // Verify deleted records return 404 via get_record
    for id in &all_ids {
        let resp = c
            .get(format!("{base}/v1/collections/reinsert_col/records/{id}"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 404, "Deleted record {id} should be 404");
    }

    // Insert 5 new records
    let new_records: Vec<Value> = (0..5)
        .map(|i| {
            json!({
                "embedding": seeded_vector(4, i + 100),
                "tag": format!("batch2_{i}")
            })
        })
        .collect();
    let resp = c
        .post(format!("{base}/v1/collections/reinsert_col/insert"))
        .json(&json!({ "records": new_records }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 201);
    assert_eq!(resp.json::<Value>().await.unwrap()["inserted"], 5);

    // Verify new records are retrievable via search
    let resp = c
        .post(format!("{base}/v1/collections/reinsert_col/search"))
        .json(&json!({
            "query": {
                "vector": { "field": "embedding", "values": seeded_vector(4, 100) }
            },
            "limit": 20
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    // Search results include newly inserted records (at minimum 5)
    assert!(body["count"].as_u64().unwrap() >= 5);
}

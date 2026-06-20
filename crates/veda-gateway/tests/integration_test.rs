use std::collections::HashMap;
use std::net::SocketAddr;

use reqwest::Client;
use serde_json::{Value, json};
use tokio::net::TcpListener;

use veda_gateway::auth::Role;
use veda_gateway::rest::build_router;
use veda_gateway::rest::state::{AppState, UserRecord};

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
    let resp = c
        .get(format!("{base}/v1/collections"))
        .send()
        .await
        .unwrap();
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
    let ids_to_delete: Vec<&str> = results
        .iter()
        .take(5)
        .map(|r| r["id"].as_str().unwrap())
        .collect();
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
    let resp = c
        .get(format!("{base}/v1/collections"))
        .send()
        .await
        .unwrap();
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
    assert!(
        count > 0 && count <= 30,
        "Expected 1-30 electronics results, got {count}"
    );

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
    let scores: Vec<f64> = results
        .iter()
        .map(|r| r["distance"].as_f64().unwrap())
        .collect();
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

    // Create again — should fail with 409 CONFLICT (duplicate).
    let resp = c
        .post(format!("{base}/v1/collections"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 409);
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
        .get(format!("{base}/v1/collections/norec_col/records/{fake_id}"))
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
    assert!(body["error"].as_str().unwrap().contains("must contain"));
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
    let resp = c
        .get(format!("{base}/v1/collections"))
        .send()
        .await
        .unwrap();
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
    let resp = c
        .get(format!("{base}/v1/collections"))
        .send()
        .await
        .unwrap();
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
#[allow(clippy::approx_constant)]
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

// ── list_records pagination ────────────────────────────────────────────────

#[tokio::test]
async fn test_list_records_pagination() {
    let (base, _tmp) = start_test_server().await;
    let c = client();

    c.post(format!("{base}/v1/collections"))
        .json(&json!({
            "name": "list_col",
            "fields": [
                { "name": "embedding", "type": "vector", "dimension": 4, "metric": "cosine" },
                { "name": "title", "type": "string", "filterable": true }
            ]
        }))
        .send()
        .await
        .unwrap();

    let mut records = Vec::new();
    for i in 0..23 {
        records.push(json!({
            "embedding": seeded_vector(4, i),
            "title": format!("Item {i}")
        }));
    }
    let resp = c
        .post(format!("{base}/v1/collections/list_col/insert"))
        .json(&json!({ "records": records }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 201);

    // Default limit (50): all 23 records returned, total=23.
    let resp = c
        .get(format!("{base}/v1/collections/list_col/records"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["total"], 23);
    assert_eq!(body["records"].as_array().unwrap().len(), 23);

    // limit=10: first 10 records, total still 23.
    let resp = c
        .get(format!("{base}/v1/collections/list_col/records?limit=10"))
        .send()
        .await
        .unwrap();
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["total"], 23);
    assert_eq!(body["records"].as_array().unwrap().len(), 10);

    // limit=10&offset=20: last 3 records.
    let resp = c
        .get(format!(
            "{base}/v1/collections/list_col/records?limit=10&offset=20"
        ))
        .send()
        .await
        .unwrap();
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["total"], 23);
    assert_eq!(body["records"].as_array().unwrap().len(), 3);

    // Unknown collection: 404.
    let resp = c
        .get(format!("{base}/v1/collections/does_not_exist/records"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404);
}

// ── Sparse vector end-to-end ───────────────────────────────────────────────

#[tokio::test]
async fn test_sparse_vector_end_to_end() {
    let (base, _tmp) = start_test_server().await;
    let c = client();

    // Create collection with a sparse vector field.
    let resp = c
        .post(format!("{base}/v1/collections"))
        .json(&json!({
            "name": "sparse_col",
            "fields": [
                { "name": "sparse_emb", "type": "sparse_vector", "dimension": 1000, "metric": "cosine" },
                { "name": "title", "type": "string", "filterable": true }
            ]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 201);

    // Insert three records with distinguishable sparse vectors.
    let resp = c
        .post(format!("{base}/v1/collections/sparse_col/insert"))
        .json(&json!({
            "records": [
                {
                    "sparse_emb": { "indices": [10, 20, 30], "values": [0.5, 0.5, 0.5] },
                    "title": "doc-a"
                },
                {
                    "sparse_emb": { "indices": [10, 20, 100], "values": [0.5, 0.5, 0.5] },
                    "title": "doc-b"
                },
                {
                    "sparse_emb": { "indices": [500, 600, 700], "values": [1.0, 1.0, 1.0] },
                    "title": "doc-c"
                }
            ]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 201);
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["inserted"], 3);

    // Sparse search: query matches doc-a most closely, then doc-b.
    let resp = c
        .post(format!("{base}/v1/collections/sparse_col/search"))
        .json(&json!({
            "query": {
                "sparse_vector": {
                    "field": "sparse_emb",
                    "indices": [10, 20, 30],
                    "values": [0.5, 0.5, 0.5]
                }
            },
            "limit": 3
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    let results = body["results"].as_array().unwrap();
    assert_eq!(
        results.len(),
        3,
        "expected 3 results, got {}",
        results.len()
    );
    // First result should be the most similar (cosine ~ 0 for identical).
    let top_distance = results[0]["distance"].as_f64().unwrap();
    assert!(
        top_distance < 0.01,
        "top result should have near-zero distance, got {top_distance}"
    );

    // Empty / invalid index out of range should 400.
    let resp = c
        .post(format!("{base}/v1/collections/sparse_col/insert"))
        .json(&json!({
            "records": [
                {
                    "sparse_emb": { "indices": [9999], "values": [0.5] },
                    "title": "doc-bad"
                }
            ]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 400);
}

// ── ColPali / multi-vector ────────────────────────────────────────────────

#[tokio::test]
async fn test_colpali_multi_vector_search() {
    let (base, _tmp) = start_test_server().await;
    let c = client();

    let resp = c
        .post(format!("{base}/v1/collections"))
        .json(&json!({
            "name": "colpali_col",
            "fields": [
                {
                    "name": "tokens",
                    "type": "multi_vector",
                    "dimension": 4,
                    "max_vectors": 8,
                    "metric": "cosine"
                },
                { "name": "title", "type": "string", "filterable": true }
            ]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 201);

    // Three records with different token sequences.
    let resp = c
        .post(format!("{base}/v1/collections/colpali_col/insert"))
        .json(&json!({
            "records": [
                {
                    "tokens": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
                    "title": "doc-a"
                },
                {
                    "tokens": [[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
                    "title": "doc-b"
                },
                {
                    "tokens": [[1.0, 1.0, 0.0, 0.0]],
                    "title": "doc-c"
                }
            ]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 201);

    // ColPali search — query exactly matches doc-a's first token.
    let resp = c
        .post(format!("{base}/v1/collections/colpali_col/search"))
        .json(&json!({
            "query": {
                "colpali": {
                    "field": "tokens",
                    "rows": [[1.0, 0.0, 0.0, 0.0]]
                }
            },
            "limit": 3
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    let results = body["results"].as_array().unwrap();
    assert!(!results.is_empty());
}

// ── MRL truncation ─────────────────────────────────────────────────────────

#[tokio::test]
async fn test_mrl_truncation_search() {
    let (base, _tmp) = start_test_server().await;
    let c = client();

    c.post(format!("{base}/v1/collections"))
        .json(&json!({
            "name": "mrl_col",
            "fields": [
                { "name": "embedding", "type": "vector", "dimension": 8, "metric": "cosine" }
            ]
        }))
        .send()
        .await
        .unwrap();

    // Three records with vectors that differ mostly in the trailing dims.
    c.post(format!("{base}/v1/collections/mrl_col/insert"))
        .json(&json!({
            "records": [
                { "embedding": [1.0, 0.0, 0.0, 0.0, 9.0, 9.0, 9.0, 9.0] },
                { "embedding": [0.0, 1.0, 0.0, 0.0, 9.0, 9.0, 9.0, 9.0] },
                { "embedding": [0.0, 0.0, 1.0, 0.0, 9.0, 9.0, 9.0, 9.0] }
            ]
        }))
        .send()
        .await
        .unwrap();

    // truncate_to=4: only the first 4 components matter; record 0 should match.
    let resp = c
        .post(format!("{base}/v1/collections/mrl_col/search"))
        .json(&json!({
            "query": { "vector": { "field": "embedding", "values": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] } },
            "limit": 3,
            "truncate_to": 4
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    let results = body["results"].as_array().unwrap();
    assert!(!results.is_empty());

    // truncate_to bigger than dimension → 400.
    let resp = c
        .post(format!("{base}/v1/collections/mrl_col/search"))
        .json(&json!({
            "query": { "vector": { "field": "embedding", "values": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] } },
            "limit": 1,
            "truncate_to": 16
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 400);
}

#[tokio::test]
async fn test_mrl_with_declared_dims_uses_index() {
    let (base, _tmp) = start_test_server().await;
    let c = client();

    // Declare a vector field with an MRL ladder.
    let resp = c
        .post(format!("{base}/v1/collections"))
        .json(&json!({
            "name": "mrl_ladder",
            "fields": [{
                "name": "embedding",
                "type": "vector",
                "dimension": 8,
                "metric": "cosine",
                "mrl_dims": [4]
            }]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 201);

    // Insert enough records that the search benefits from an index. After
    // insert, the gateway seals and rebuilds, which is when MRL indexes are
    // constructed.
    let mut records = Vec::new();
    for i in 0..50 {
        records.push(json!({
            "embedding": [
                (i as f32 % 7.0) * 0.1, (i as f32 % 5.0) * 0.1, (i as f32 % 3.0) * 0.1, i as f32 * 0.01,
                0.9, 0.8, 0.7, 0.6
            ]
        }));
    }
    let resp = c
        .post(format!("{base}/v1/collections/mrl_ladder/insert"))
        .json(&json!({ "records": records }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 201);

    // Search with truncate_to=4 (in mrl_dims) — must return results.
    let resp = c
        .post(format!("{base}/v1/collections/mrl_ladder/search"))
        .json(&json!({
            "query": { "vector": { "field": "embedding", "values": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] } },
            "limit": 5,
            "truncate_to": 4
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    assert!(!body["results"].as_array().unwrap().is_empty());

    // truncate_to=6 (NOT in mrl_dims) — still works via brute-force fallback.
    let resp = c
        .post(format!("{base}/v1/collections/mrl_ladder/search"))
        .json(&json!({
            "query": { "vector": { "field": "embedding", "values": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] } },
            "limit": 5,
            "truncate_to": 6
        }))
        .send()
        .await
        .unwrap();
    // With mrl_dims=[4], 6 isn't in the allow-list — server should reject.
    assert_eq!(resp.status(), 400);
}

// ── Embedding model versioning ─────────────────────────────────────────────

#[tokio::test]
async fn test_collection_records_declared_model() {
    let (base, _tmp) = start_test_server().await;
    let c = client();

    let resp = c
        .post(format!("{base}/v1/collections"))
        .json(&json!({
            "name": "model_bound",
            "fields": [
                {
                    "name": "embedding",
                    "type": "vector",
                    "dimension": 8,
                    "metric": "cosine",
                    "model": "bge-m3"
                },
                { "name": "title", "type": "string" }
            ]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 201);

    // Describe should surface the declared model.
    let resp = c
        .get(format!("{base}/v1/collections/model_bound"))
        .send()
        .await
        .unwrap();
    let body: Value = resp.json().await.unwrap();
    let emb = body["fields"]
        .as_array()
        .unwrap()
        .iter()
        .find(|f| f["name"] == "embedding")
        .unwrap();
    assert_eq!(emb["model"], "bge-m3");

    // Default (no model) — no model field on response.
    let resp = c
        .post(format!("{base}/v1/collections"))
        .json(&json!({
            "name": "no_model",
            "fields": [
                { "name": "embedding", "type": "vector", "dimension": 8, "metric": "cosine" }
            ]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 201);
    let resp = c
        .get(format!("{base}/v1/collections/no_model"))
        .send()
        .await
        .unwrap();
    let body: Value = resp.json().await.unwrap();
    let emb = body["fields"]
        .as_array()
        .unwrap()
        .iter()
        .find(|f| f["name"] == "embedding")
        .unwrap();
    assert!(emb.get("model").is_none() || emb["model"].is_null());
}

// ── MCP server ─────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_mcp_initialize_and_list_tools() {
    let (base, _tmp) = start_test_server().await;
    let c = client();

    // initialize
    let resp = c
        .post(format!("{base}/v1/mcp"))
        .json(&json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {}
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["jsonrpc"], "2.0");
    assert_eq!(body["id"], 1);
    assert!(body["result"]["serverInfo"]["name"] == "veda");
    assert!(body["result"]["protocolVersion"].as_str().is_some());

    // tools/list
    let resp = c
        .post(format!("{base}/v1/mcp"))
        .json(&json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list"
        }))
        .send()
        .await
        .unwrap();
    let body: Value = resp.json().await.unwrap();
    let tools = body["result"]["tools"].as_array().unwrap();
    let names: Vec<&str> = tools.iter().map(|t| t["name"].as_str().unwrap()).collect();
    // The full builtin catalog (8 tools) is now exposed over MCP.
    for expected in [
        "list_collections",
        "describe_collection",
        "search",
        "upsert_record",
        "get_record",
        "route",
        "aggregate",
        "filter_query",
    ] {
        assert!(names.contains(&expected), "missing MCP tool {expected}");
    }
    assert_eq!(tools.len(), 12, "expected the full 12-tool catalog");

    // Every entry has an inputSchema and an annotations object.
    for t in tools {
        assert!(t["inputSchema"].is_object(), "{} inputSchema", t["name"]);
        let ann = &t["annotations"];
        assert!(ann.is_object(), "{} annotations", t["name"]);
        assert!(ann["readOnlyHint"].is_boolean());
        assert!(ann["destructiveHint"].is_boolean());
        assert!(ann["title"].as_str().is_some());
    }

    // Read tools are read-only/non-destructive; upsert_record is the
    // sole mutating/destructive tool.
    let find = |n: &str| tools.iter().find(|t| t["name"] == n).unwrap();
    assert_eq!(find("search")["annotations"]["readOnlyHint"], true);
    assert_eq!(find("search")["annotations"]["destructiveHint"], false);
    assert_eq!(find("aggregate")["annotations"]["readOnlyHint"], true);
    assert_eq!(find("upsert_record")["annotations"]["readOnlyHint"], false);
    assert_eq!(
        find("upsert_record")["annotations"]["destructiveHint"],
        true
    );

    // The multimodal search schema flows through unchanged.
    let search_props = &find("search")["inputSchema"]["properties"];
    assert!(search_props["query_image_b64"].is_object());
    assert!(search_props["mode"].is_object());
}

#[tokio::test]
async fn test_mcp_aggregate_tool_call() {
    let (base, _tmp) = start_test_server().await;
    let c = client();

    // Collection with a numeric, filterable field.
    c.post(format!("{base}/v1/collections"))
        .json(&json!({
            "name": "mcp_agg",
            "fields": [
                { "name": "embedding", "type": "vector", "dimension": 4, "metric": "cosine" },
                { "name": "price", "type": "float64", "filterable": true }
            ]
        }))
        .send()
        .await
        .unwrap();

    let records: Vec<Value> = [10.0, 20.0, 30.0]
        .iter()
        .map(|p| json!({ "embedding": [1.0, 0.0, 0.0, 0.0], "price": p }))
        .collect();
    let resp = c
        .post(format!("{base}/v1/collections/mcp_agg/insert"))
        .json(&json!({ "records": records }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 201);

    // `aggregate` is a newly-exposed builtin — call it over MCP.
    let resp = c
        .post(format!("{base}/v1/mcp"))
        .json(&json!({
            "jsonrpc": "2.0",
            "id": 7,
            "method": "tools/call",
            "params": {
                "name": "aggregate",
                "arguments": { "collection": "mcp_agg", "field": "price", "metric": "sum" }
            }
        }))
        .send()
        .await
        .unwrap();
    let body: Value = resp.json().await.unwrap();
    assert!(body.get("error").is_none(), "body: {body}");
    let text = body["result"]["content"][0]["text"].as_str().unwrap();
    let parsed: Value = serde_json::from_str(text).unwrap();
    assert_eq!(parsed["groups"][0]["value"], 60.0);
}

#[tokio::test]
async fn test_mcp_list_collections_tool() {
    let (base, _tmp) = start_test_server().await;
    let c = client();

    // Create a collection first.
    c.post(format!("{base}/v1/collections"))
        .json(&json!({
            "name": "mcp_test",
            "fields": [{ "name": "embedding", "type": "vector", "dimension": 4, "metric": "cosine" }]
        }))
        .send()
        .await
        .unwrap();

    // Call the tool via MCP.
    let resp = c
        .post(format!("{base}/v1/mcp"))
        .json(&json!({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "list_collections",
                "arguments": {}
            }
        }))
        .send()
        .await
        .unwrap();
    let body: Value = resp.json().await.unwrap();
    let content = body["result"]["content"].as_array().unwrap();
    let text = content[0]["text"].as_str().unwrap();
    let parsed: Value = serde_json::from_str(text).unwrap();
    let names = parsed["collections"].as_array().unwrap();
    assert!(names.iter().any(|n| n.as_str() == Some("mcp_test")));
}

#[tokio::test]
async fn test_mcp_unknown_method_returns_error() {
    let (base, _tmp) = start_test_server().await;
    let c = client();
    let resp = c
        .post(format!("{base}/v1/mcp"))
        .json(&json!({
            "jsonrpc": "2.0",
            "id": 4,
            "method": "nonsense"
        }))
        .send()
        .await
        .unwrap();
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["error"]["code"], -32601);
}

#[tokio::test]
async fn test_mcp_tool_arg_error_returns_iserror_result() {
    let (base, _tmp) = start_test_server().await;
    let c = client();
    // `search` without the required `query` argument is a TOOL-execution
    // error: per MCP it must come back as a successful JSON-RPC result with
    // `isError: true` (a CallToolResult), NOT a JSON-RPC -32602 error, so the
    // calling model can read the message and self-correct.
    let resp = c
        .post(format!("{base}/v1/mcp"))
        .json(&json!({
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {
                "name": "search",
                "arguments": { "collection": "anything" }
            }
        }))
        .send()
        .await
        .unwrap();
    let body: Value = resp.json().await.unwrap();
    assert!(
        body.get("error").is_none(),
        "must not be a JSON-RPC error: {body}"
    );
    assert_eq!(body["result"]["isError"], true, "body: {body}");
    let text = body["result"]["content"][0]["text"].as_str().unwrap();
    assert!(!text.is_empty(), "error message must be non-empty: {body}");
}

#[tokio::test]
async fn test_mcp_tool_notfound_returns_iserror_result() {
    let (base, _tmp) = start_test_server().await;
    let c = client();
    // `aggregate` on a collection that does not exist is a NotFound
    // tool-execution error -> isError result, not a JSON-RPC error.
    let resp = c
        .post(format!("{base}/v1/mcp"))
        .json(&json!({
            "jsonrpc": "2.0",
            "id": 9,
            "method": "tools/call",
            "params": {
                "name": "aggregate",
                "arguments": { "collection": "does_not_exist", "field": "price", "metric": "sum" }
            }
        }))
        .send()
        .await
        .unwrap();
    let body: Value = resp.json().await.unwrap();
    assert!(
        body.get("error").is_none(),
        "must not be a JSON-RPC error: {body}"
    );
    assert_eq!(body["result"]["isError"], true, "body: {body}");
    let text = body["result"]["content"][0]["text"].as_str().unwrap();
    assert!(!text.is_empty(), "error message must be non-empty: {body}");
}

#[tokio::test]
async fn test_mcp_unknown_tool_name_returns_32601() {
    let (base, _tmp) = start_test_server().await;
    let c = client();
    // An unknown TOOL NAME is a protocol error (the tool doesn't exist),
    // so it stays a JSON-RPC -32601, distinct from a tool that ran and failed.
    let resp = c
        .post(format!("{base}/v1/mcp"))
        .json(&json!({
            "jsonrpc": "2.0",
            "id": 10,
            "method": "tools/call",
            "params": {
                "name": "no_such_tool",
                "arguments": {}
            }
        }))
        .send()
        .await
        .unwrap();
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["error"]["code"], -32601, "body: {body}");
}

#[tokio::test]
async fn test_mcp_success_has_structured_content() {
    let (base, _tmp) = start_test_server().await;
    let c = client();

    c.post(format!("{base}/v1/collections"))
        .json(&json!({
            "name": "mcp_struct",
            "fields": [{ "name": "embedding", "type": "vector", "dimension": 4, "metric": "cosine" }]
        }))
        .send()
        .await
        .unwrap();

    // A successful tools/call returns BOTH a content[0].text block AND a
    // structuredContent object (MCP 2025-06-18) equal to the tool's JSON.
    let resp = c
        .post(format!("{base}/v1/mcp"))
        .json(&json!({
            "jsonrpc": "2.0",
            "id": 11,
            "method": "tools/call",
            "params": {
                "name": "list_collections",
                "arguments": {}
            }
        }))
        .send()
        .await
        .unwrap();
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["result"]["isError"], false, "body: {body}");
    let text = body["result"]["content"][0]["text"].as_str().unwrap();
    let parsed_text: Value = serde_json::from_str(text).unwrap();
    let structured = &body["result"]["structuredContent"];
    assert!(
        structured.is_object(),
        "structuredContent must be present: {body}"
    );
    // structuredContent is the raw tool JSON, equal to the parsed text block.
    assert_eq!(
        *structured, parsed_text,
        "structuredContent must equal tool JSON"
    );
    assert!(
        structured["collections"]
            .as_array()
            .unwrap()
            .iter()
            .any(|n| n.as_str() == Some("mcp_struct"))
    );
}

#[tokio::test]
async fn test_mcp_initialize_protocol_negotiation() {
    let (base, _tmp) = start_test_server().await;
    let c = client();

    // Client requesting 2025-06-18 gets it echoed back.
    let resp = c
        .post(format!("{base}/v1/mcp"))
        .json(&json!({
            "jsonrpc": "2.0",
            "id": 20,
            "method": "initialize",
            "params": { "protocolVersion": "2025-06-18" }
        }))
        .send()
        .await
        .unwrap();
    let body: Value = resp.json().await.unwrap();
    assert_eq!(
        body["result"]["protocolVersion"], "2025-06-18",
        "body: {body}"
    );

    // Client requesting the older 2025-03-26 gets that echoed back.
    let resp = c
        .post(format!("{base}/v1/mcp"))
        .json(&json!({
            "jsonrpc": "2.0",
            "id": 21,
            "method": "initialize",
            "params": { "protocolVersion": "2025-03-26" }
        }))
        .send()
        .await
        .unwrap();
    let body: Value = resp.json().await.unwrap();
    assert_eq!(
        body["result"]["protocolVersion"], "2025-03-26",
        "body: {body}"
    );

    // No protocolVersion -> our preferred 2025-06-18.
    let resp = c
        .post(format!("{base}/v1/mcp"))
        .json(&json!({
            "jsonrpc": "2.0",
            "id": 22,
            "method": "initialize",
            "params": {}
        }))
        .send()
        .await
        .unwrap();
    let body: Value = resp.json().await.unwrap();
    assert_eq!(
        body["result"]["protocolVersion"], "2025-06-18",
        "body: {body}"
    );

    // An unsupported version -> our preferred 2025-06-18.
    let resp = c
        .post(format!("{base}/v1/mcp"))
        .json(&json!({
            "jsonrpc": "2.0",
            "id": 23,
            "method": "initialize",
            "params": { "protocolVersion": "1999-01-01" }
        }))
        .send()
        .await
        .unwrap();
    let body: Value = resp.json().await.unwrap();
    assert_eq!(
        body["result"]["protocolVersion"], "2025-06-18",
        "body: {body}"
    );
}

// ── MCP Streamable HTTP transport ──────────────────────────────────────────

#[tokio::test]
async fn test_mcp_post_with_supported_protocol_version_header() {
    let (base, _tmp) = start_test_server().await;
    let c = client();
    // A supported MCP-Protocol-Version header is accepted; response is the
    // unchanged JSON-RPC result with Content-Type: application/json.
    let resp = c
        .post(format!("{base}/v1/mcp"))
        .header("MCP-Protocol-Version", "2025-06-18")
        .json(&json!({
            "jsonrpc": "2.0",
            "id": 30,
            "method": "tools/list"
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let ct = resp
        .headers()
        .get("content-type")
        .unwrap()
        .to_str()
        .unwrap()
        .to_string();
    assert!(ct.starts_with("application/json"), "content-type: {ct}");
    let body: Value = resp.json().await.unwrap();
    assert!(body["result"]["tools"].as_array().unwrap().len() == 12);
}

#[tokio::test]
async fn test_mcp_post_without_protocol_version_header_is_lenient() {
    let (base, _tmp) = start_test_server().await;
    let c = client();
    // No MCP-Protocol-Version header -> still 200 (lenient).
    let resp = c
        .post(format!("{base}/v1/mcp"))
        .json(&json!({
            "jsonrpc": "2.0",
            "id": 31,
            "method": "tools/list"
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
}

#[tokio::test]
async fn test_mcp_post_with_unsupported_protocol_version_header_400() {
    let (base, _tmp) = start_test_server().await;
    let c = client();
    // An unsupported MCP-Protocol-Version header -> HTTP 400.
    let resp = c
        .post(format!("{base}/v1/mcp"))
        .header("MCP-Protocol-Version", "1999-01-01")
        .json(&json!({
            "jsonrpc": "2.0",
            "id": 32,
            "method": "tools/list"
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 400);
}

#[tokio::test]
async fn test_mcp_get_returns_405() {
    let (base, _tmp) = start_test_server().await;
    let c = client();
    // GET /v1/mcp (server->client SSE) is unsupported -> 405 with Allow: POST.
    let resp = c.get(format!("{base}/v1/mcp")).send().await.unwrap();
    assert_eq!(resp.status(), 405);
    let allow = resp
        .headers()
        .get("allow")
        .unwrap()
        .to_str()
        .unwrap()
        .to_string();
    assert!(allow.contains("POST"), "allow: {allow}");
}

// ── Array fields ───────────────────────────────────────────────────────────

#[tokio::test]
async fn test_array_field_indexing() {
    let (base, _tmp) = start_test_server().await;
    let c = client();

    // Collection with a tags array field (string[]) and a required vector.
    let resp = c
        .post(format!("{base}/v1/collections"))
        .json(&json!({
            "name": "tagged_col",
            "fields": [
                { "name": "embedding", "type": "vector", "dimension": 4, "metric": "cosine" },
                { "name": "tags", "type": "array_string", "filterable": true }
            ]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 201);

    let resp = c
        .post(format!("{base}/v1/collections/tagged_col/insert"))
        .json(&json!({
            "records": [
                { "embedding": [0.1, 0.2, 0.3, 0.4], "tags": ["red", "small"] },
                { "embedding": [0.2, 0.3, 0.4, 0.5], "tags": ["blue", "small"] },
                { "embedding": [0.3, 0.4, 0.5, 0.6], "tags": ["red", "large"] }
            ]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 201);

    // Filter on array membership: tags = 'red' should return 2 records.
    let resp = c
        .post(format!("{base}/v1/collections/tagged_col/search"))
        .json(&json!({
            "query": { "vector": { "field": "embedding", "values": [0.1, 0.2, 0.3, 0.4] } },
            "limit": 10,
            "filter": "tags = 'red'"
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    assert_eq!(
        body["count"].as_u64().unwrap(),
        2,
        "tags = 'red' should match 2 records, got: {body}"
    );

    // Filter on array membership: tags = 'small' should also return 2.
    let resp = c
        .post(format!("{base}/v1/collections/tagged_col/search"))
        .json(&json!({
            "query": { "vector": { "field": "embedding", "values": [0.1, 0.2, 0.3, 0.4] } },
            "limit": 10,
            "filter": "tags = 'small'"
        }))
        .send()
        .await
        .unwrap();
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["count"].as_u64().unwrap(), 2);

    // No match: tags = 'nonexistent' returns 0.
    let resp = c
        .post(format!("{base}/v1/collections/tagged_col/search"))
        .json(&json!({
            "query": { "vector": { "field": "embedding", "values": [0.1, 0.2, 0.3, 0.4] } },
            "limit": 10,
            "filter": "tags = 'nonexistent'"
        }))
        .send()
        .await
        .unwrap();
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["count"].as_u64().unwrap(), 0);
}

// ── Phase 5.5 — chunker A/B preview ─────────────────────────────────────────

#[tokio::test]
async fn test_chunk_preview_runs_each_strategy() {
    let (base, _tmp) = start_test_server().await;
    let c = client();

    let text = "# Section A\n\nFirst paragraph about cloud computing. Lots of \
                detail here that goes on for a while to give the chunker \
                something to work with.\n\nSecond paragraph here.\n\n# Section B\n\n\
                Third paragraph on Kubernetes.";

    let resp = c
        .post(format!("{base}/v1/chunk/preview"))
        .json(&json!({
            "text": text,
            "runs": [
                { "label": "Default", "strategy_name": "markdown_recursive" },
                { "label": "Fixed-80",  "strategy_name": "fixed_size", "chunk_size": 80, "chunk_overlap": 10 },
                { "strategy_name": "page" }
            ]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    assert_eq!(
        body["input_chars"].as_u64().unwrap(),
        text.chars().count() as u64
    );
    let runs = body["runs"].as_array().unwrap();
    assert_eq!(runs.len(), 3);

    // Default run keeps the user-provided label.
    assert_eq!(runs[0]["label"].as_str().unwrap(), "Default");
    assert_eq!(
        runs[0]["strategy_name"].as_str().unwrap(),
        "markdown_recursive"
    );
    // Each run reports nonzero elapsed and a chunks vec.
    for r in runs {
        assert!(r["elapsed_ms"].as_f64().is_some());
        assert!(r["chunks"].as_array().is_some());
        let total = r["total_chunks"].as_u64().unwrap() as usize;
        let returned = r["chunks"].as_array().unwrap().len();
        // returned <= total, with truncated flag when ineq.
        assert!(returned <= total);
        assert_eq!(r["truncated"].as_bool().unwrap(), returned < total);
    }
    // Fixed-80 should split a 150-ish-char doc into more chunks than the
    // markdown_recursive default (max 2000c per chunk).
    assert!(
        runs[1]["total_chunks"].as_u64().unwrap() > runs[0]["total_chunks"].as_u64().unwrap(),
        "fixed-80 ({}) should produce more chunks than markdown_recursive ({})",
        runs[1]["total_chunks"],
        runs[0]["total_chunks"],
    );
}

#[tokio::test]
async fn test_chunk_preview_rejects_invalid_inputs() {
    let (base, _tmp) = start_test_server().await;
    let c = client();

    // empty text -> 400
    let resp = c
        .post(format!("{base}/v1/chunk/preview"))
        .json(&json!({ "text": "", "runs": [{ "strategy_name": "fixed_size" }] }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 400);

    // empty runs -> 400
    let resp = c
        .post(format!("{base}/v1/chunk/preview"))
        .json(&json!({ "text": "hello", "runs": [] }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 400);

    // 7 runs (> 6 cap) -> 400
    let runs: Vec<_> = (0..7)
        .map(|i| json!({ "label": format!("r{i}"), "strategy_name": "fixed_size" }))
        .collect();
    let resp = c
        .post(format!("{base}/v1/chunk/preview"))
        .json(&json!({ "text": "hello world", "runs": runs }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 400);
}

// ── Phase 5.8 — LLM connector ───────────────────────────────────────────────

#[tokio::test]
async fn test_llm_complete_returns_503_when_not_configured() {
    // Tests run with VEDA_LLM_ENDPOINT unset; the endpoint should
    // refuse with 503 rather than panic or fall through.
    let (base, _tmp) = start_test_server().await;
    let c = client();
    let resp = c
        .post(format!("{base}/v1/llm/complete"))
        .json(&json!({
            "messages": [
                { "role": "user", "content": "hi" }
            ]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 503);
}

// ── Phase 5.7 — coverage rollup ────────────────────────────────────────────

#[tokio::test]
async fn test_coverage_empty_then_after_search() {
    let (base, _tmp) = start_test_server().await;
    let c = client();

    c.post(format!("{base}/v1/collections"))
        .json(&json!({
            "name": "cov_col",
            "fields": [
                { "name": "embedding", "type": "vector", "dimension": 4, "metric": "cosine" }
            ],
            "shard_count": 1,
            "replication_factor": 1
        }))
        .send()
        .await
        .unwrap();

    // Empty corpus + empty log.
    let resp = c
        .get(format!("{base}/v1/collections/cov_col/coverage?window=1h"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["total_queries"].as_u64().unwrap(), 0);
    assert_eq!(body["total_chunks"].as_u64().unwrap(), 0);
    assert_eq!(body["retrieved_chunks"].as_u64().unwrap(), 0);
    assert_eq!(body["orphan_count"].as_u64().unwrap(), 0);

    // Insert a handful of records so there's something to be orphaned.
    let records: Vec<_> = (0..5)
        .map(|i| json!({ "embedding": seeded_vector(4, i) }))
        .collect();
    let resp = c
        .post(format!("{base}/v1/collections/cov_col/insert"))
        .json(&json!({ "records": records }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 201);

    // Before any search: 5 chunks, all orphans.
    let resp = c
        .get(format!("{base}/v1/collections/cov_col/coverage?window=1h"))
        .send()
        .await
        .unwrap();
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["total_chunks"].as_u64().unwrap(), 5);
    assert_eq!(body["orphan_count"].as_u64().unwrap(), 5);
    assert_eq!(body["retrieved_chunks"].as_u64().unwrap(), 0);

    // Run a search so the JSONL append fires.
    let resp = c
        .post(format!("{base}/v1/collections/cov_col/search"))
        .json(&json!({
            "query": { "vector": { "field": "embedding", "values": seeded_vector(4, 0) } },
            "limit": 3
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    let hit_count = body["results"].as_array().unwrap().len();
    assert!(hit_count > 0);

    // The query log writer is fire-and-forget on a spawn_blocking — give
    // it a beat to land on disk before we read it back.
    tokio::time::sleep(std::time::Duration::from_millis(150)).await;

    let resp = c
        .get(format!("{base}/v1/collections/cov_col/coverage?window=1h"))
        .send()
        .await
        .unwrap();
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["total_queries"].as_u64().unwrap(), 1);
    assert_eq!(
        body["retrieved_chunks"].as_u64().unwrap(),
        hit_count as u64,
        "retrieved_chunks should equal the number of hits the search returned"
    );
    assert_eq!(body["orphan_count"].as_u64().unwrap(), 5 - hit_count as u64,);
    let top_hits = body["top_hits"].as_array().unwrap();
    assert_eq!(top_hits.len(), hit_count);
    for hit in top_hits {
        assert_eq!(hit["count"].as_u64().unwrap(), 1);
    }
}

#[tokio::test]
async fn test_coverage_window_parsing() {
    let (base, _tmp) = start_test_server().await;
    let c = client();

    c.post(format!("{base}/v1/collections"))
        .json(&json!({
            "name": "win_col",
            "fields": [
                { "name": "embedding", "type": "vector", "dimension": 4, "metric": "cosine" }
            ],
            "shard_count": 1,
            "replication_factor": 1
        }))
        .send()
        .await
        .unwrap();

    for (window, expected_secs) in [("1h", 3600u64), ("7d", 7 * 86400), ("30m", 1800)] {
        let resp = c
            .get(format!(
                "{base}/v1/collections/win_col/coverage?window={window}"
            ))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let body: Value = resp.json().await.unwrap();
        assert_eq!(
            body["window_secs"].as_u64().unwrap(),
            expected_secs,
            "window={window} should parse to {expected_secs}s"
        );
    }

    // Unknown collection -> 404.
    let resp = c
        .get(format!("{base}/v1/collections/no_such/coverage"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404);
}

//! Integration tests for Feature E: OLAP-style analytics aggregation endpoints.
//!
//! Covers:
//!   - GET /v1/collections/{name}/aggregate  (count/sum/avg/min/max, optional group_by, optional filter)
//!   - GET /v1/collections/{name}/histogram  (equal-width bins)
//!   - GET /v1/collections/{name}/field-stats (cardinality, null count, numeric min/max/avg)

use reqwest::Client;
use serde_json::{json, Value};
use std::net::SocketAddr;
use tokio::net::TcpListener;
use veda_gateway::rest::{build_router, state::AppState};

async fn start_test_server() -> (String, tempfile::TempDir) {
    let tmp = tempfile::tempdir().unwrap();
    let state = AppState::new(tmp.path());
    let router = build_router(state);
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr: SocketAddr = listener.local_addr().unwrap();
    let base = format!("http://{addr}");
    tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
    (base, tmp)
}

fn client() -> Client {
    Client::new()
}

/// Create a collection with category (string) + price (float64) fields and seed records.
///
/// Seed data:
///   electronics  price=100.0
///   electronics  price=200.0
///   books        price=10.0
///   books        price=20.0
///   books        price=30.0
///
/// Summary by category:
///   electronics: count=2  sum=300  avg=150  min=100  max=200
///   books:       count=3  sum=60   avg=20   min=10   max=30
///   total(no grp):count=5 sum=360  avg=72   min=10   max=200
async fn setup_collection(base: &str) -> String {
    let col = "agg_test";
    let r = client()
        .post(format!("{base}/v1/collections"))
        .json(&json!({
            "name": col,
            "fields": [
                { "name": "category", "type": "string",  "filterable": true },
                { "name": "price",    "type": "float64", "filterable": true }
            ]
        }))
        .send().await.unwrap();
    assert_eq!(r.status(), 201, "create failed: {}", r.text().await.unwrap());

    let records = json!({ "records": [
        { "category": "electronics", "price": 100.0 },
        { "category": "electronics", "price": 200.0 },
        { "category": "books",       "price": 10.0  },
        { "category": "books",       "price": 20.0  },
        { "category": "books",       "price": 30.0  }
    ]});
    let r = client()
        .post(format!("{base}/v1/collections/{col}/insert"))
        .json(&records)
        .send().await.unwrap();
    assert_eq!(r.status(), 201, "insert failed: {}", r.text().await.unwrap());

    col.to_owned()
}

// ── 1. aggregate with group_by ────────────────────────────────────────────────

#[tokio::test]
async fn test_aggregate_sum_group_by() {
    let (base, _tmp) = start_test_server().await;
    let col = setup_collection(&base).await;

    let resp = client()
        .get(format!("{base}/v1/collections/{col}/aggregate?field=price&metric=sum&group_by=category"))
        .send().await.unwrap();
    assert_eq!(resp.status(), 200, "{}", resp.text().await.unwrap());

    let body: Value = resp.json().await.unwrap();
    let groups = body["groups"].as_array().unwrap();
    assert_eq!(groups.len(), 2);

    // Build a map key -> value
    let map: std::collections::HashMap<&str, f64> = groups.iter()
        .map(|g| (g["key"].as_str().unwrap(), g["value"].as_f64().unwrap()))
        .collect();

    assert_eq!(map["electronics"], 300.0);
    assert_eq!(map["books"], 60.0);
}

#[tokio::test]
async fn test_aggregate_count_group_by() {
    let (base, _tmp) = start_test_server().await;
    let col = setup_collection(&base).await;

    let resp = client()
        .get(format!("{base}/v1/collections/{col}/aggregate?field=category&metric=count&group_by=category"))
        .send().await.unwrap();
    assert_eq!(resp.status(), 200, "{}", resp.text().await.unwrap());

    let body: Value = resp.json().await.unwrap();
    let groups = body["groups"].as_array().unwrap();
    let map: std::collections::HashMap<&str, f64> = groups.iter()
        .map(|g| (g["key"].as_str().unwrap(), g["value"].as_f64().unwrap()))
        .collect();

    assert_eq!(map["electronics"], 2.0);
    assert_eq!(map["books"], 3.0);
}

// ── 2. aggregate without group_by (scalar result) ─────────────────────────────

#[tokio::test]
async fn test_aggregate_no_group_by() {
    let (base, _tmp) = start_test_server().await;
    let col = setup_collection(&base).await;

    // count
    let resp = client()
        .get(format!("{base}/v1/collections/{col}/aggregate?field=price&metric=count"))
        .send().await.unwrap();
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    let groups = body["groups"].as_array().unwrap();
    assert_eq!(groups.len(), 1);
    assert_eq!(groups[0]["value"].as_f64().unwrap(), 5.0);

    // sum
    let resp = client()
        .get(format!("{base}/v1/collections/{col}/aggregate?field=price&metric=sum"))
        .send().await.unwrap();
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    let val = body["groups"][0]["value"].as_f64().unwrap();
    assert!((val - 360.0).abs() < 0.001, "sum expected 360, got {val}");

    // avg
    let resp = client()
        .get(format!("{base}/v1/collections/{col}/aggregate?field=price&metric=avg"))
        .send().await.unwrap();
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    let val = body["groups"][0]["value"].as_f64().unwrap();
    assert!((val - 72.0).abs() < 0.001, "avg expected 72, got {val}");

    // min
    let resp = client()
        .get(format!("{base}/v1/collections/{col}/aggregate?field=price&metric=min"))
        .send().await.unwrap();
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    let val = body["groups"][0]["value"].as_f64().unwrap();
    assert!((val - 10.0).abs() < 0.001, "min expected 10, got {val}");

    // max
    let resp = client()
        .get(format!("{base}/v1/collections/{col}/aggregate?field=price&metric=max"))
        .send().await.unwrap();
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    let val = body["groups"][0]["value"].as_f64().unwrap();
    assert!((val - 200.0).abs() < 0.001, "max expected 200, got {val}");
}

// ── 3. histogram ──────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_histogram() {
    let (base, _tmp) = start_test_server().await;
    let col = setup_collection(&base).await;

    // 4 bins over [10, 200] → width ≈ 47.5 each
    // [10,57.5)→books(10,20,30)=3, [57.5,105)→electronics(100)=1, [105,152.5)→0, [152.5,200]→electronics(200)=1
    let resp = client()
        .get(format!("{base}/v1/collections/{col}/histogram?field=price&bins=4"))
        .send().await.unwrap();
    assert_eq!(resp.status(), 200, "{}", resp.text().await.unwrap());

    let body: Value = resp.json().await.unwrap();
    let bins = body["bins"].as_array().unwrap();
    assert_eq!(bins.len(), 4, "expected 4 bins, got {:?}", body);

    // Total count across all bins must equal number of records (5)
    let total: u64 = bins.iter().map(|b| b["count"].as_u64().unwrap_or(0)).sum();
    assert_eq!(total, 5, "total bin count should be 5, got {total}");

    // Each bin has lo, hi, count keys
    for bin in bins {
        assert!(bin.get("lo").is_some());
        assert!(bin.get("hi").is_some());
        assert!(bin.get("count").is_some());
    }
}

#[tokio::test]
async fn test_histogram_default_bins() {
    let (base, _tmp) = start_test_server().await;
    let col = setup_collection(&base).await;

    // No bins param → default 10
    let resp = client()
        .get(format!("{base}/v1/collections/{col}/histogram?field=price"))
        .send().await.unwrap();
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["bins"].as_array().unwrap().len(), 10);
}

// ── 4. field-stats ────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_field_stats_numeric() {
    let (base, _tmp) = start_test_server().await;
    let col = setup_collection(&base).await;

    let resp = client()
        .get(format!("{base}/v1/collections/{col}/field-stats?field=price"))
        .send().await.unwrap();
    assert_eq!(resp.status(), 200, "{}", resp.text().await.unwrap());

    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["total_count"].as_u64().unwrap(), 5);
    assert_eq!(body["null_count"].as_u64().unwrap(), 0);
    assert_eq!(body["cardinality"].as_u64().unwrap(), 5); // all distinct prices

    let min = body["min"].as_f64().unwrap();
    let max = body["max"].as_f64().unwrap();
    let avg = body["avg"].as_f64().unwrap();
    assert!((min - 10.0).abs() < 0.001);
    assert!((max - 200.0).abs() < 0.001);
    assert!((avg - 72.0).abs() < 0.001);
}

#[tokio::test]
async fn test_field_stats_string() {
    let (base, _tmp) = start_test_server().await;
    let col = setup_collection(&base).await;

    let resp = client()
        .get(format!("{base}/v1/collections/{col}/field-stats?field=category"))
        .send().await.unwrap();
    assert_eq!(resp.status(), 200, "{}", resp.text().await.unwrap());

    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["total_count"].as_u64().unwrap(), 5);
    assert_eq!(body["null_count"].as_u64().unwrap(), 0);
    // 2 distinct categories
    assert_eq!(body["cardinality"].as_u64().unwrap(), 2);
    // For string field: min_length, max_length present
    assert!(body.get("min_length").is_some());
    assert!(body.get("max_length").is_some());
}

// ── 5. filter exclusion ───────────────────────────────────────────────────────

#[tokio::test]
async fn test_aggregate_with_filter() {
    let (base, _tmp) = start_test_server().await;
    let col = setup_collection(&base).await;

    // filter: only books (category = 'books')
    // expected sum for books = 60
    let resp = client()
        .get(format!("{base}/v1/collections/{col}/aggregate"))
        .query(&[("field", "price"), ("metric", "sum"), ("filter", "category = 'books'")])
        .send().await.unwrap();
    assert_eq!(resp.status(), 200, "{}", resp.text().await.unwrap());

    let body: Value = resp.json().await.unwrap();
    let val = body["groups"][0]["value"].as_f64().unwrap();
    assert!((val - 60.0).abs() < 0.001, "filtered sum expected 60, got {val}");
}

#[tokio::test]
async fn test_histogram_with_filter() {
    let (base, _tmp) = start_test_server().await;
    let col = setup_collection(&base).await;

    // filter: only electronics (2 records: 100, 200)
    let resp = client()
        .get(format!("{base}/v1/collections/{col}/histogram"))
        .query(&[("field", "price"), ("bins", "2"), ("filter", "category = 'electronics'")])
        .send().await.unwrap();
    assert_eq!(resp.status(), 200, "{}", resp.text().await.unwrap());

    let body: Value = resp.json().await.unwrap();
    let bins = body["bins"].as_array().unwrap();
    let total: u64 = bins.iter().map(|b| b["count"].as_u64().unwrap_or(0)).sum();
    assert_eq!(total, 2, "filtered histogram total should be 2, got {total}");
}

// ── 6. error cases ────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_aggregate_nonexistent_collection() {
    let (base, _tmp) = start_test_server().await;
    let resp = client()
        .get(format!("{base}/v1/collections/no_such_col/aggregate?field=price&metric=sum"))
        .send().await.unwrap();
    assert_eq!(resp.status(), 404);
}

#[tokio::test]
async fn test_aggregate_numeric_metric_on_string_field() {
    let (base, _tmp) = start_test_server().await;
    let col = setup_collection(&base).await;

    // sum on a string field → 400
    let resp = client()
        .get(format!("{base}/v1/collections/{col}/aggregate?field=category&metric=sum"))
        .send().await.unwrap();
    assert_eq!(resp.status(), 400, "expected 400 for sum on string field");
}

#[tokio::test]
async fn test_histogram_nonexistent_collection() {
    let (base, _tmp) = start_test_server().await;
    let resp = client()
        .get(format!("{base}/v1/collections/no_such_col/histogram?field=price"))
        .send().await.unwrap();
    assert_eq!(resp.status(), 404);
}

#[tokio::test]
async fn test_field_stats_nonexistent_collection() {
    let (base, _tmp) = start_test_server().await;
    let resp = client()
        .get(format!("{base}/v1/collections/no_such_col/field-stats?field=price"))
        .send().await.unwrap();
    assert_eq!(resp.status(), 404);
}

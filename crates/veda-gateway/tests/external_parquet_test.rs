//! Integration tests: registering an external Parquet collection via
//! `POST /v1/collections`. An external collection declares no Veda fields —
//! column names, types and the vector dimension come from the Parquet
//! schema itself, so `fields` is empty and an `external` block describes
//! the source instead.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::path::Path;
use std::sync::Arc as StdArc;

use arrow::array::{ArrayRef, FixedSizeListArray, Float32Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use reqwest::Client;
use serde_json::{Value, json};
use tokio::net::TcpListener;

use veda_gateway::auth::Role;
use veda_gateway::rest::build_router;
use veda_gateway::rest::state::{AppState, UserRecord};

// ── Harness ──────────────────────────────────────────────────────────────

async fn start_server() -> (String, tempfile::TempDir) {
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

/// Write a Parquet file with a string `id` column and, when `dim` is
/// `Some`, a `FixedSizeList<Float32>` `embedding` column of that width.
/// `ids.len()` rows.
fn write_parquet(path: &Path, ids: &[&str], dim: Option<usize>) {
    let mut fields = vec![Field::new("id", DataType::Utf8, false)];
    let mut columns: Vec<ArrayRef> = vec![StdArc::new(StringArray::from(ids.to_vec()))];

    if let Some(dim) = dim {
        let n = ids.len();
        let mut flat: Vec<f32> = Vec::with_capacity(n * dim);
        for i in 0..n {
            for d in 0..dim {
                flat.push((i * dim + d) as f32 / 10.0);
            }
        }
        let values = Float32Array::from(flat);
        let item_field = StdArc::new(Field::new("item", DataType::Float32, false));
        let vectors =
            FixedSizeListArray::try_new(item_field.clone(), dim as i32, StdArc::new(values), None)
                .unwrap();
        fields.push(Field::new(
            "embedding",
            DataType::FixedSizeList(item_field, dim as i32),
            false,
        ));
        columns.push(StdArc::new(vectors));
    }

    let schema = StdArc::new(Schema::new(fields));
    let batch = RecordBatch::try_new(schema.clone(), columns).unwrap();
    let file = std::fs::File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

fn file_uri(path: &Path) -> String {
    format!("file://{}", path.display())
}

/// Write a Parquet file with a string `id` column, a `FixedSizeList<Float32>`
/// `embedding` column of width `dim`, and a string `owner` column — for the
/// RLS test, which needs a scalar column `rls::inject_owner_filter`'s
/// predicate can match against. `rows` is `(id, owner)` per row; the vector
/// formula matches `write_parquet`'s (`(i * dim + d) / 10.0` for row `i`).
fn write_parquet_with_owner(path: &Path, rows: &[(&str, &str)], dim: usize) {
    let ids: Vec<&str> = rows.iter().map(|(id, _)| *id).collect();
    let owners: Vec<&str> = rows.iter().map(|(_, owner)| *owner).collect();
    let n = ids.len();
    let mut flat: Vec<f32> = Vec::with_capacity(n * dim);
    for i in 0..n {
        for d in 0..dim {
            flat.push((i * dim + d) as f32 / 10.0);
        }
    }
    let values = Float32Array::from(flat);
    let item_field = StdArc::new(Field::new("item", DataType::Float32, false));
    let vectors =
        FixedSizeListArray::try_new(item_field.clone(), dim as i32, StdArc::new(values), None)
            .unwrap();
    let schema = StdArc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new(
            "embedding",
            DataType::FixedSizeList(item_field, dim as i32),
            false,
        ),
        Field::new("owner", DataType::Utf8, false),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            StdArc::new(StringArray::from(ids)) as ArrayRef,
            StdArc::new(vectors) as ArrayRef,
            StdArc::new(StringArray::from(owners)) as ArrayRef,
        ],
    )
    .unwrap();
    let file = std::fs::File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

/// Log in against a server started with `AppState::with_auth_and_users`,
/// returning the bearer token. Mirrors `rls_test.rs`'s `login_user` — do not
/// invent a different auth shape.
async fn login_user(base: &str, user: &str, pass: &str) -> String {
    let resp = client()
        .post(format!("{base}/v1/auth/login"))
        .json(&json!({ "username": user, "password": pass }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200, "login {user} failed");
    resp.json::<Value>().await.unwrap()["token"]
        .as_str()
        .unwrap()
        .to_owned()
}

// ── Tests ────────────────────────────────────────────────────────────────

#[tokio::test]
async fn registering_a_valid_parquet_dataset_succeeds() {
    let (base, _tmp) = start_server().await;
    let data_dir = tempfile::tempdir().unwrap();
    let file_path = data_dir.path().join("data.parquet");
    write_parquet(&file_path, &["a", "b", "c"], Some(3));

    let r = client()
        .post(format!("{base}/v1/collections"))
        .json(&json!({
            "name": "ext_valid",
            "fields": [],
            "external": {
                "format": "parquet",
                "uri": file_uri(&file_path),
                "id_field": "id",
                "vector_field": "embedding",
                "metric": "Cosine"
            }
        }))
        .send()
        .await
        .unwrap();
    let status = r.status();
    let body: Value = r.json().await.unwrap();
    assert_eq!(status, 201, "create: {body}");
    assert_eq!(body["external_dimension"], 3, "body: {body}");
    assert_eq!(body["external_row_count"], 3, "body: {body}");
}

#[tokio::test]
async fn registering_with_a_missing_vector_column_is_rejected() {
    let (base, _tmp) = start_server().await;
    let data_dir = tempfile::tempdir().unwrap();
    let file_path = data_dir.path().join("data.parquet");
    // Only the id column — no `embedding` column at all.
    write_parquet(&file_path, &["a", "b"], None);

    let r = client()
        .post(format!("{base}/v1/collections"))
        .json(&json!({
            "name": "ext_missing_vec",
            "fields": [],
            "external": {
                "format": "parquet",
                "uri": file_uri(&file_path),
                "id_field": "id",
                "vector_field": "embedding",
                "metric": "Cosine"
            }
        }))
        .send()
        .await
        .unwrap();
    let status = r.status();
    let body = r.text().await.unwrap();
    assert_eq!(status, 400, "body: {body}");
    assert!(
        body.contains("embedding"),
        "missing column name absent from body: {body}"
    );
    assert!(
        body.contains("id"),
        "existing columns absent from body: {body}"
    );
}

#[tokio::test]
async fn registering_with_a_nonexistent_uri_is_rejected() {
    let (base, _tmp) = start_server().await;
    let data_dir = tempfile::tempdir().unwrap();
    // Never created — resolves to an existing tempdir with no such
    // subdirectory beneath it, so no .parquet objects can be found there.
    let missing = data_dir.path().join("does-not-exist");

    let r = client()
        .post(format!("{base}/v1/collections"))
        .json(&json!({
            "name": "ext_missing_uri",
            "fields": [],
            "external": {
                "format": "parquet",
                "uri": format!("file://{}/", missing.display()),
                "id_field": "id",
                "vector_field": "embedding",
                "metric": "Cosine"
            }
        }))
        .send()
        .await
        .unwrap();
    let status = r.status();
    let body = r.text().await.unwrap();
    assert_eq!(status, 400, "body: {body}");
}

#[tokio::test]
async fn registering_a_wrong_typed_vector_column_is_rejected() {
    let (base, _tmp) = start_server().await;
    let data_dir = tempfile::tempdir().unwrap();
    let file_path = data_dir.path().join("data.parquet");
    write_parquet(&file_path, &["a", "b"], Some(3));

    // vector_field points at the string `id` column instead of `embedding`.
    let r = client()
        .post(format!("{base}/v1/collections"))
        .json(&json!({
            "name": "ext_wrong_type",
            "fields": [],
            "external": {
                "format": "parquet",
                "uri": file_uri(&file_path),
                "id_field": "id",
                "vector_field": "id",
                "metric": "Cosine"
            }
        }))
        .send()
        .await
        .unwrap();
    let status = r.status();
    let body = r.text().await.unwrap();
    assert_eq!(status, 400, "body: {body}");
    assert!(body.contains("id"), "column name absent from body: {body}");
}

#[tokio::test]
async fn registering_with_a_bucket_in_the_uri_is_rejected() {
    let (base, _tmp) = start_server().await;

    let r = client()
        .post(format!("{base}/v1/collections"))
        .json(&json!({
            "name": "ext_bucket",
            "fields": [],
            "external": {
                "format": "parquet",
                "uri": "s3://bucket/prefix/",
                "id_field": "id",
                "vector_field": "embedding",
                "metric": "Cosine"
            }
        }))
        .send()
        .await
        .unwrap();
    let status = r.status();
    let body = r.text().await.unwrap();
    assert_eq!(status, 400, "body: {body}");
    assert!(
        body.contains("server configuration"),
        "bucket explanation absent from body: {body}"
    );
}

#[tokio::test]
async fn registering_an_external_collection_with_fields_is_rejected() {
    let (base, _tmp) = start_server().await;

    // A valid-shaped external block, but paired with a declared field —
    // an external collection infers its schema from the source, so a
    // caller-declared field can never be backed by real data and must be
    // rejected rather than silently dropped.
    let r = client()
        .post(format!("{base}/v1/collections"))
        .json(&json!({
            "name": "ext_with_fields",
            "fields": [
                { "name": "embedding", "type": "vector", "dimension": 768 }
            ],
            "external": {
                "format": "parquet",
                "uri": "file:///tmp/does-not-need-to-exist/",
                "id_field": "id",
                "vector_field": "embedding",
                "metric": "Cosine"
            }
        }))
        .send()
        .await
        .unwrap();
    let status = r.status();
    let body = r.text().await.unwrap();
    assert_eq!(status, 400, "body: {body}");
    assert!(body.contains("fields"), "'fields' absent from body: {body}");
}

// ── Search dispatch tests (Task 7) ──────────────────────────────────────

/// `write_parquet`'s vector formula is `(i * dim + d) / 10.0` for row `i`,
/// component `d`. Row 0's vector is always `[0.0, 0.1, ..., (dim-1)/10.0]`,
/// so a query built the same way is an exact match (distance 0) for
/// whichever id occupies row 0.
fn row0_query(dim: usize) -> Vec<f32> {
    (0..dim).map(|d| d as f32 / 10.0).collect()
}

async fn register_external(base: &str, name: &str, uri: &str) -> reqwest::Response {
    client()
        .post(format!("{base}/v1/collections"))
        .json(&json!({
            "name": name,
            "fields": [],
            "external": {
                "format": "parquet",
                "uri": uri,
                "id_field": "id",
                "vector_field": "embedding",
                "metric": "Cosine"
            }
        }))
        .send()
        .await
        .unwrap()
}

#[tokio::test]
async fn external_dense_search_returns_expected_ids() {
    let (base, _tmp) = start_server().await;
    let data_dir = tempfile::tempdir().unwrap();
    let file_path = data_dir.path().join("data.parquet");
    write_parquet(&file_path, &["a", "b", "c"], Some(3));

    let r = register_external(&base, "ext_search", &file_uri(&file_path)).await;
    assert_eq!(r.status(), 201, "create: {}", r.text().await.unwrap());

    let r = client()
        .post(format!("{base}/v1/collections/ext_search/search"))
        .json(&json!({
            "query": { "vector": { "field": "embedding", "values": row0_query(3) } },
            "limit": 10
        }))
        .send()
        .await
        .unwrap();
    let status = r.status();
    let body: Value = r.json().await.unwrap();
    assert_eq!(status, 200, "search: {body}");
    let results = body["results"].as_array().expect("results array");
    assert!(!results.is_empty(), "expected non-empty results: {body}");
    assert_eq!(
        results[0]["id"], "a",
        "nearest id should be 'a' (row 0 is an exact match for the query): {body}"
    );
}

#[tokio::test]
async fn external_search_honours_a_filter() {
    let (base, _tmp) = start_server().await;
    let data_dir = tempfile::tempdir().unwrap();
    let file_path = data_dir.path().join("data.parquet");
    write_parquet(&file_path, &["a", "b", "c"], Some(3));

    let r = register_external(&base, "ext_filter", &file_uri(&file_path)).await;
    assert_eq!(r.status(), 201, "create: {}", r.text().await.unwrap());

    // Sanity: unfiltered search returns "a" first (row 0 is an exact match).
    let r = client()
        .post(format!("{base}/v1/collections/ext_filter/search"))
        .json(&json!({
            "query": { "vector": { "field": "embedding", "values": row0_query(3) } },
            "limit": 10
        }))
        .send()
        .await
        .unwrap();
    let body: Value = r.json().await.unwrap();
    assert_eq!(body["results"][0]["id"], "a", "sanity check: {body}");

    // Filtered: exclude "a" via the id column (present on every hit's
    // record because `record_at` always sets the id_field).
    let r = client()
        .post(format!("{base}/v1/collections/ext_filter/search"))
        .json(&json!({
            "query": { "vector": { "field": "embedding", "values": row0_query(3) } },
            "filter": "id != 'a'",
            "limit": 10
        }))
        .send()
        .await
        .unwrap();
    let status = r.status();
    let body: Value = r.json().await.unwrap();
    assert_eq!(status, 200, "filtered search: {body}");
    let results = body["results"].as_array().expect("results array");
    assert!(
        !results.is_empty(),
        "filter should not exclude everything: {body}"
    );
    assert!(
        results.iter().all(|h| h["id"] != "a"),
        "'a' should have been excluded by the filter: {body}"
    );
}

#[tokio::test]
async fn a_rewritten_source_is_visible_to_the_next_query() {
    let (base, _tmp) = start_server().await;
    let data_dir = tempfile::tempdir().unwrap();
    let file_path = data_dir.path().join("data.parquet");
    write_parquet(&file_path, &["a", "b", "c"], Some(3));

    let r = register_external(&base, "ext_stale", &file_uri(&file_path)).await;
    assert_eq!(r.status(), 201, "create: {}", r.text().await.unwrap());

    let query = json!({
        "query": { "vector": { "field": "embedding", "values": row0_query(3) } },
        "limit": 10
    });

    let r = client()
        .post(format!("{base}/v1/collections/ext_stale/search"))
        .json(&query)
        .send()
        .await
        .unwrap();
    let status = r.status();
    let body: Value = r.json().await.unwrap();
    assert_eq!(status, 200, "first search: {body}");
    assert_eq!(body["results"][0]["id"], "a", "before rewrite: {body}");

    // Rewrite the same path: row 0's id changes from "a" to "q", and a 4th
    // row is added so the file's byte size changes too (mtime granularity
    // alone can't be trusted to distinguish a same-tick rewrite).
    write_parquet(&file_path, &["q", "a", "c", "extra"], Some(3));

    let r = client()
        .post(format!("{base}/v1/collections/ext_stale/search"))
        .json(&query)
        .send()
        .await
        .unwrap();
    let status = r.status();
    let body: Value = r.json().await.unwrap();
    assert_eq!(status, 200, "second search: {body}");
    assert_eq!(
        body["results"][0]["id"], "q",
        "after rewrite, row 0 (an exact match for the query) is 'q', not 'a' — a stale \
         cache would still return 'a': {body}"
    );
}

#[tokio::test]
async fn external_bm25_is_rejected() {
    let (base, _tmp) = start_server().await;
    let data_dir = tempfile::tempdir().unwrap();
    let file_path = data_dir.path().join("data.parquet");
    write_parquet(&file_path, &["a", "b", "c"], Some(3));

    let r = register_external(&base, "ext_bm25", &file_uri(&file_path)).await;
    assert_eq!(r.status(), 201, "create: {}", r.text().await.unwrap());

    let r = client()
        .post(format!("{base}/v1/collections/ext_bm25/search"))
        .json(&json!({
            "query": { "bm25": { "field": "content", "query": "hello" } },
            "limit": 10
        }))
        .send()
        .await
        .unwrap();
    let status = r.status();
    let body = r.text().await.unwrap();
    assert_eq!(status, 400, "body: {body}");
    assert!(
        body.contains("ext_bm25"),
        "collection name absent from body: {body}"
    );
    assert!(
        body.contains("inverted index"),
        "missing-index explanation absent from body: {body}"
    );
}

#[tokio::test]
async fn external_sparse_is_rejected() {
    let (base, _tmp) = start_server().await;
    let data_dir = tempfile::tempdir().unwrap();
    let file_path = data_dir.path().join("data.parquet");
    write_parquet(&file_path, &["a", "b", "c"], Some(3));

    let r = register_external(&base, "ext_sparse", &file_uri(&file_path)).await;
    assert_eq!(r.status(), 201, "create: {}", r.text().await.unwrap());

    let r = client()
        .post(format!("{base}/v1/collections/ext_sparse/search"))
        .json(&json!({
            "query": { "sparse_vector": { "field": "splade", "indices": [1, 2], "values": [0.5, 0.5] } },
            "limit": 10
        }))
        .send()
        .await
        .unwrap();
    let status = r.status();
    let body = r.text().await.unwrap();
    assert_eq!(status, 400, "body: {body}");
    assert!(
        body.contains("ext_sparse"),
        "collection name absent from body: {body}"
    );
    assert!(
        body.contains("sparse vectors") || body.contains("stored sparse"),
        "missing-index explanation absent from body: {body}"
    );
}

#[tokio::test]
async fn external_hybrid_is_rejected() {
    let (base, _tmp) = start_server().await;
    let data_dir = tempfile::tempdir().unwrap();
    let file_path = data_dir.path().join("data.parquet");
    write_parquet(&file_path, &["a", "b", "c"], Some(3));

    let r = register_external(&base, "ext_hybrid", &file_uri(&file_path)).await;
    assert_eq!(r.status(), 201, "create: {}", r.text().await.unwrap());

    let r = client()
        .post(format!("{base}/v1/collections/ext_hybrid/search"))
        .json(&json!({
            "query": {
                "hybrid": {
                    "dense": { "field": "embedding", "values": row0_query(3) },
                    "bm25": { "field": "content", "query": "hello" }
                }
            },
            "limit": 10
        }))
        .send()
        .await
        .unwrap();
    let status = r.status();
    let body = r.text().await.unwrap();
    assert_eq!(status, 400, "body: {body}");
    assert!(
        body.contains("ext_hybrid"),
        "collection name absent from body: {body}"
    );
    assert!(
        body.contains("bm25"),
        "missing-index explanation absent from body: {body}"
    );
}

#[tokio::test]
async fn external_match_phrase_is_rejected() {
    let (base, _tmp) = start_server().await;
    let data_dir = tempfile::tempdir().unwrap();
    let file_path = data_dir.path().join("data.parquet");
    write_parquet(&file_path, &["a", "b", "c"], Some(3));

    let r = register_external(&base, "ext_match_phrase", &file_uri(&file_path)).await;
    assert_eq!(r.status(), 201, "create: {}", r.text().await.unwrap());

    let r = client()
        .post(format!("{base}/v1/collections/ext_match_phrase/search"))
        .json(&json!({
            "query": { "match_phrase": { "field": "content", "query": "hello world" } },
            "limit": 10
        }))
        .send()
        .await
        .unwrap();
    let status = r.status();
    let body = r.text().await.unwrap();
    assert_eq!(status, 400, "body: {body}");
    assert!(
        body.contains("ext_match_phrase"),
        "collection name absent from body: {body}"
    );
    assert!(
        body.contains("inverted index"),
        "missing-index explanation absent from body: {body}"
    );
}

#[tokio::test]
async fn external_rerank_is_rejected() {
    let (base, _tmp) = start_server().await;
    let data_dir = tempfile::tempdir().unwrap();
    let file_path = data_dir.path().join("data.parquet");
    write_parquet(&file_path, &["a", "b", "c"], Some(3));

    let r = register_external(&base, "ext_rerank", &file_uri(&file_path)).await;
    assert_eq!(r.status(), 201, "create: {}", r.text().await.unwrap());

    // limit=3 with rerank: on an internal collection this oversamples to
    // limit*4=20 candidates internally and truncates back to 3 reranked
    // hits. External must reject outright rather than leak the oversampled,
    // untruncated, unreranked 20-hit candidate set.
    let r = client()
        .post(format!("{base}/v1/collections/ext_rerank/search"))
        .json(&json!({
            "query": { "vector": { "field": "embedding", "values": row0_query(3) } },
            "limit": 3,
            "rerank": { "query": "irrelevant", "field": "content" }
        }))
        .send()
        .await
        .unwrap();
    let status = r.status();
    let body = r.text().await.unwrap();
    assert_eq!(status, 400, "body: {body}");
    assert!(
        body.contains("ext_rerank"),
        "collection name absent from body: {body}"
    );
    assert!(
        body.contains("rerank"),
        "rerank explanation absent from body: {body}"
    );
}

#[tokio::test]
async fn external_malformed_filter_is_rejected() {
    let (base, _tmp) = start_server().await;
    let data_dir = tempfile::tempdir().unwrap();
    let file_path = data_dir.path().join("data.parquet");
    write_parquet(&file_path, &["a", "b", "c"], Some(3));

    let r = register_external(&base, "ext_bad_filter", &file_uri(&file_path)).await;
    assert_eq!(r.status(), 201, "create: {}", r.text().await.unwrap());

    // Unterminated quote. The old `effective_filter`'s `.ok()` swallowed
    // the parse error and ran an unfiltered search instead, returning
    // every row with a 200 — over-disclosure, not a diagnostic.
    let r = client()
        .post(format!("{base}/v1/collections/ext_bad_filter/search"))
        .json(&json!({
            "query": { "vector": { "field": "embedding", "values": row0_query(3) } },
            "filter": "id = 'a",
            "limit": 10
        }))
        .send()
        .await
        .unwrap();
    let status = r.status();
    let body = r.text().await.unwrap();
    assert_eq!(status, 400, "body: {body}");
    assert!(
        body.contains("filter"),
        "parse-failure explanation absent from body: {body}"
    );
}

#[tokio::test]
async fn external_search_through_an_alias_resolves() {
    let (base, _tmp) = start_server().await;
    let data_dir = tempfile::tempdir().unwrap();
    let file_path = data_dir.path().join("data.parquet");
    write_parquet(&file_path, &["a", "b", "c"], Some(3));

    let r = register_external(&base, "ext_alias_target", &file_uri(&file_path)).await;
    assert_eq!(r.status(), 201, "create: {}", r.text().await.unwrap());

    let r = client()
        .put(format!("{base}/v1/admin/collections/ext_alias_name/alias"))
        .json(&json!({ "target": "ext_alias_target" }))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200, "set_alias: {}", r.text().await.unwrap());

    let query = json!({
        "query": { "vector": { "field": "embedding", "values": row0_query(3) } },
        "limit": 10
    });

    let direct = client()
        .post(format!("{base}/v1/collections/ext_alias_target/search"))
        .json(&query)
        .send()
        .await
        .unwrap();
    let direct_status = direct.status();
    let direct_body: Value = direct.json().await.unwrap();
    assert_eq!(direct_status, 200, "direct search: {direct_body}");

    let via_alias = client()
        .post(format!("{base}/v1/collections/ext_alias_name/search"))
        .json(&query)
        .send()
        .await
        .unwrap();
    let alias_status = via_alias.status();
    let alias_body: Value = via_alias.json().await.unwrap();
    assert_eq!(alias_status, 200, "alias search: {alias_body}");

    assert_eq!(
        direct_body["results"], alias_body["results"],
        "search through the alias should return exactly the hits direct search does: \
         direct={direct_body}, alias={alias_body}"
    );
}

/// Security regression: an external collection registered with
/// `rls_owner_field` must apply the same owner-equality filter non-admin
/// searches get on internal collections — the external dispatch branch sits
/// after RLS injection precisely so this holds. Before that fix, the branch
/// returned before RLS ever ran, silently serving every row to every caller.
#[tokio::test]
async fn external_search_applies_row_level_security() {
    let mut users = HashMap::new();
    users.insert(
        "admin".to_owned(),
        UserRecord {
            password: "admin".into(),
            role: Role::Admin,
        },
    );
    users.insert(
        "alice".to_owned(),
        UserRecord {
            password: "alice".into(),
            role: Role::Editor,
        },
    );

    let tmp = tempfile::tempdir().unwrap();
    let state = AppState::with_auth_and_users(tmp.path(), "ext-rls-test-secret", users);
    let router = build_router(state);
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr: SocketAddr = listener.local_addr().unwrap();
    let base = format!("http://{addr}");
    tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });

    let admin = login_user(&base, "admin", "admin").await;
    let alice = login_user(&base, "alice", "alice").await;

    let data_dir = tempfile::tempdir().unwrap();
    let file_path = data_dir.path().join("data.parquet");
    // Row 0 and row 1 belong to alice, row 2 to bob (who never logs in —
    // his row must still never reach alice's results).
    write_parquet_with_owner(
        &file_path,
        &[("a1", "alice"), ("a2", "alice"), ("b1", "bob")],
        3,
    );

    let r = client()
        .post(format!("{base}/v1/collections"))
        .header("Authorization", format!("Bearer {admin}"))
        .json(&json!({
            "name": "ext_rls",
            "fields": [],
            "rls_owner_field": "owner",
            "external": {
                "format": "parquet",
                "uri": file_uri(&file_path),
                "id_field": "id",
                "vector_field": "embedding",
                "metric": "Cosine"
            }
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 201, "create: {}", r.text().await.unwrap());

    // Query built from row 1's vector (id "a2", owner "alice") so an
    // unfiltered search would surface rows across owners, not just
    // alice's — a passing result here is not a coincidence of geometry.
    let r = client()
        .post(format!("{base}/v1/collections/ext_rls/search"))
        .header("Authorization", format!("Bearer {alice}"))
        .json(&json!({
            "query": { "vector": { "field": "embedding", "values": [0.3, 0.4, 0.5] } },
            "limit": 10
        }))
        .send()
        .await
        .unwrap();
    let status = r.status();
    let body: Value = r.json().await.unwrap();
    assert_eq!(status, 200, "alice search: {body}");
    let results = body["results"].as_array().expect("results array");
    assert!(!results.is_empty(), "alice should see her own rows: {body}");
    assert!(
        results.iter().all(|h| h["fields"]["owner"] == "alice"),
        "RLS bypass: alice saw a row she does not own: {body}"
    );
    assert!(
        results.iter().all(|h| h["id"] != "b1"),
        "RLS bypass: alice saw bob's row 'b1': {body}"
    );
}

#[tokio::test]
async fn an_external_collection_reports_itself_as_external() {
    let (base, _tmp) = start_server().await;
    let data_dir = tempfile::tempdir().unwrap();
    let file_path = data_dir.path().join("data.parquet");
    write_parquet(&file_path, &["a", "b", "c"], Some(3));
    let uri = file_uri(&file_path);

    let r = client()
        .post(format!("{base}/v1/collections"))
        .json(&json!({
            "name": "ext_describe",
            "fields": [],
            "external": {
                "format": "parquet",
                "uri": uri.clone(),
                "id_field": "id",
                "vector_field": "embedding",
                "metric": "Cosine"
            }
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 201, "create: {}", r.text().await.unwrap());

    let r = client()
        .get(format!("{base}/v1/collections/ext_describe"))
        .send()
        .await
        .unwrap();
    let status = r.status();
    let body: Value = r.json().await.unwrap();
    assert_eq!(status, 200, "describe: {body}");
    assert_eq!(body["external"]["uri"], uri, "body: {body}");
    assert_eq!(body["external"]["id_field"], "id", "body: {body}");
    assert_eq!(
        body["external"]["vector_field"], "embedding",
        "body: {body}"
    );
}

// ── Read-only enforcement (Task 8) ──────────────────────────────────────
//
// External collections read a customer's Parquet file in place. Veda must
// never write records, derived data or an index for one — that state would
// go stale the moment the source changes and there is no local source of
// truth for it. Every route below is asserted 405 METHOD_NOT_ALLOWED with a
// body that names the collection and explains it is external/read-only.
// Registration-only routes (dropping the collection, repointing its alias)
// are asserted to keep working — an enforcement pass that accidentally
// bricks deregistration would be worse than no enforcement.

/// Register a throwaway external collection and return its name plus the
/// backing tempdir (kept alive so the Parquet file isn't deleted mid-test).
async fn setup_external(base: &str, name: &str) -> tempfile::TempDir {
    let data_dir = tempfile::tempdir().unwrap();
    let file_path = data_dir.path().join("data.parquet");
    write_parquet(&file_path, &["a", "b", "c"], Some(3));
    let r = register_external(base, name, &file_uri(&file_path)).await;
    assert_eq!(r.status(), 201, "create: {}", r.text().await.unwrap());
    data_dir
}

/// Common assertion for every REJECT route: 405, and a body that names the
/// collection and explains why. Always prints the body on failure so a
/// wrong-for-an-unrelated-reason failure is diagnosable, not just "not 405".
async fn assert_rejected_as_external(resp: reqwest::Response, route: &str, name: &str) {
    let status = resp.status();
    let body = resp.text().await.unwrap();
    assert_eq!(
        status, 405,
        "{route}: expected 405 METHOD_NOT_ALLOWED, got {status}. body: {body}"
    );
    assert!(
        body.contains(name),
        "{route}: collection name absent from body: {body}"
    );
    assert!(
        body.to_lowercase().contains("external") && body.to_lowercase().contains("read-only"),
        "{route}: body does not explain external/read-only: {body}"
    );
}

#[tokio::test]
async fn external_reindex_is_rejected() {
    let (base, _tmp) = start_server().await;
    let name = "ext_reindex";
    let _data = setup_external(&base, name).await;

    let r = client()
        .post(format!("{base}/v1/collections/{name}/reindex"))
        .json(&json!({}))
        .send()
        .await
        .unwrap();
    assert_rejected_as_external(r, "POST /reindex", name).await;
}

#[tokio::test]
async fn external_tuning_campaign_start_is_rejected() {
    let (base, _tmp) = start_server().await;
    let name = "ext_tuning";
    let _data = setup_external(&base, name).await;

    let r = client()
        .post(format!("{base}/v1/collections/{name}/tuning-campaigns"))
        .json(&json!({ "eval_set_id": "does-not-matter" }))
        .send()
        .await
        .unwrap();
    assert_rejected_as_external(r, "POST /tuning-campaigns", name).await;
}

#[tokio::test]
async fn external_seal_is_rejected() {
    let (base, _tmp) = start_server().await;
    let name = "ext_seal";
    let _data = setup_external(&base, name).await;

    let r = client()
        .post(format!("{base}/v1/collections/{name}/seal"))
        .send()
        .await
        .unwrap();
    assert_rejected_as_external(r, "POST /seal", name).await;
}

#[tokio::test]
async fn external_optimize_is_rejected() {
    let (base, _tmp) = start_server().await;
    let name = "ext_optimize";
    let _data = setup_external(&base, name).await;

    let r = client()
        .post(format!("{base}/v1/collections/{name}/optimize"))
        .send()
        .await
        .unwrap();
    assert_rejected_as_external(r, "POST /optimize", name).await;
}

#[tokio::test]
async fn external_insert_is_rejected() {
    let (base, _tmp) = start_server().await;
    let name = "ext_insert";
    let _data = setup_external(&base, name).await;

    let r = client()
        .post(format!("{base}/v1/collections/{name}/insert"))
        .json(&json!({ "records": [{ "id": "would-be-fabricated" }] }))
        .send()
        .await
        .unwrap();
    assert_rejected_as_external(r, "POST /insert", name).await;
}

#[tokio::test]
async fn external_embed_insert_is_rejected() {
    let (base, _tmp) = start_server().await;
    let name = "ext_embed_insert";
    let _data = setup_external(&base, name).await;

    let r = client()
        .post(format!("{base}/v1/collections/{name}/embed-insert"))
        .json(&json!({ "records": [] }))
        .send()
        .await
        .unwrap();
    assert_rejected_as_external(r, "POST /embed-insert", name).await;
}

#[tokio::test]
async fn external_ingest_logs_is_rejected() {
    let (base, _tmp) = start_server().await;
    let name = "ext_ingest_logs";
    let _data = setup_external(&base, name).await;

    let r = client()
        .post(format!("{base}/v1/collections/{name}/ingest-logs"))
        .json(&json!({ "lines": ["hello world"] }))
        .send()
        .await
        .unwrap();
    assert_rejected_as_external(r, "POST /ingest-logs", name).await;
}

#[tokio::test]
async fn external_delete_records_is_rejected() {
    let (base, _tmp) = start_server().await;
    let name = "ext_delete_records";
    let _data = setup_external(&base, name).await;

    let r = client()
        .delete(format!("{base}/v1/collections/{name}/records"))
        .json(&json!({ "ids": ["00000000-0000-0000-0000-000000000000"] }))
        .send()
        .await
        .unwrap();
    assert_rejected_as_external(r, "DELETE /records", name).await;
}

#[tokio::test]
async fn external_process_document_is_rejected() {
    let (base, _tmp) = start_server().await;
    let name = "ext_process_document";
    let _data = setup_external(&base, name).await;

    let form = reqwest::multipart::Form::new().part(
        "file",
        reqwest::multipart::Part::bytes(b"%PDF-1.4 fake".to_vec())
            .file_name("doc.pdf")
            .mime_str("application/pdf")
            .unwrap(),
    );
    let r = client()
        .post(format!("{base}/v1/collections/{name}/process-document"))
        .multipart(form)
        .send()
        .await
        .unwrap();
    assert_rejected_as_external(r, "POST /process-document", name).await;
}

#[tokio::test]
async fn external_process_documents_batch_is_rejected() {
    let (base, _tmp) = start_server().await;
    let name = "ext_process_documents";
    let _data = setup_external(&base, name).await;

    let form = reqwest::multipart::Form::new().part(
        "files",
        reqwest::multipart::Part::bytes(b"%PDF-1.4 fake".to_vec())
            .file_name("doc.pdf")
            .mime_str("application/pdf")
            .unwrap(),
    );
    let r = client()
        .post(format!("{base}/v1/collections/{name}/process-documents"))
        .multipart(form)
        .send()
        .await
        .unwrap();
    assert_rejected_as_external(r, "POST /process-documents", name).await;
}

#[tokio::test]
async fn external_ingest_url_is_rejected() {
    let (base, _tmp) = start_server().await;
    let name = "ext_ingest_url";
    let _data = setup_external(&base, name).await;

    let r = client()
        .post(format!("{base}/v1/collections/{name}/ingest-url"))
        .json(&json!({ "url": "http://example.com" }))
        .send()
        .await
        .unwrap();
    assert_rejected_as_external(r, "POST /ingest-url", name).await;
}

#[tokio::test]
async fn external_ingest_media_is_rejected() {
    let (base, _tmp) = start_server().await;
    let name = "ext_ingest_media";
    let _data = setup_external(&base, name).await;

    let form = reqwest::multipart::Form::new().part(
        "file",
        reqwest::multipart::Part::bytes(b"fake image bytes".to_vec())
            .file_name("pic.png")
            .mime_str("image/png")
            .unwrap(),
    );
    let r = client()
        .post(format!("{base}/v1/collections/{name}/ingest-media"))
        .multipart(form)
        .send()
        .await
        .unwrap();
    assert_rejected_as_external(r, "POST /ingest-media", name).await;
}

#[tokio::test]
async fn external_mvcc_checkpoint_is_rejected() {
    let (base, _tmp) = start_server().await;
    let name = "ext_mvcc_checkpoint";
    let _data = setup_external(&base, name).await;

    let r = client()
        .post(format!(
            "{base}/v1/admin/collections/{name}/mvcc_checkpoint"
        ))
        .send()
        .await
        .unwrap();
    assert_rejected_as_external(r, "POST /mvcc_checkpoint", name).await;
}

#[tokio::test]
async fn external_rescan_guardrails_is_rejected() {
    let (base, _tmp) = start_server().await;
    let name = "ext_rescan_guardrails";
    let _data = setup_external(&base, name).await;

    let r = client()
        .post(format!(
            "{base}/v1/admin/collections/{name}/rescan-guardrails"
        ))
        .send()
        .await
        .unwrap();
    assert_rejected_as_external(r, "POST /rescan-guardrails", name).await;
}

// ── Registration-only routes stay allowed ───────────────────────────────

#[tokio::test]
async fn dropping_an_external_collection_still_succeeds() {
    let (base, _tmp) = start_server().await;
    let name = "ext_drop_allowed";
    let _data = setup_external(&base, name).await;

    let r = client()
        .delete(format!("{base}/v1/collections/{name}"))
        .send()
        .await
        .unwrap();
    let status = r.status();
    let body = r.text().await.unwrap();
    assert_eq!(status, 200, "drop should succeed: {body}");

    // And it's really gone: describe now 404s.
    let r = client()
        .get(format!("{base}/v1/collections/{name}"))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 404, "collection should be gone after drop");
}

#[tokio::test]
async fn setting_the_alias_of_an_external_collection_still_succeeds() {
    let (base, _tmp) = start_server().await;
    let name = "ext_alias_allowed";
    let _data = setup_external(&base, name).await;

    let r = client()
        .put(format!("{base}/v1/admin/collections/{name}/alias"))
        .json(&json!({ "target": name }))
        .send()
        .await
        .unwrap();
    let status = r.status();
    let body = r.text().await.unwrap();
    assert_eq!(status, 200, "set_alias should succeed: {body}");
}

// ── Security: path traversal + admin-only registration (whole-branch
// review finding 1) ──────────────────────────────────────────────────────

#[tokio::test]
async fn registering_with_a_path_traversal_key_is_rejected() {
    let (base, _tmp) = start_server().await;

    // A bare key with a `..` component: `PathBuf::join` does not
    // canonicalise it away, so this would otherwise escape the configured
    // blob root.
    let r = client()
        .post(format!("{base}/v1/collections"))
        .json(&json!({
            "name": "ext_traversal",
            "fields": [],
            "external": {
                "format": "parquet",
                "uri": "../../etc/secret.parquet",
                "id_field": "id",
                "vector_field": "embedding",
                "metric": "Cosine"
            }
        }))
        .send()
        .await
        .unwrap();
    let status = r.status();
    let body = r.text().await.unwrap();
    assert_eq!(status, 400, "body: {body}");
    // Assert on the REJECTION's own wording, not on "..". A body containing
    // ".." proves nothing: with the traversal check removed the URI still
    // reaches the store and comes back as `Blob not found: ../../etc/...`,
    // which is also a 400 and also contains "..", so the weaker assertion
    // passed with the security fix deleted.
    assert!(
        body.contains("path component"),
        "traversal rejection absent from body: {body}"
    );
}

#[tokio::test]
async fn registering_with_a_path_traversal_in_a_file_uri_is_rejected() {
    let (base, _tmp) = start_server().await;

    // An absolute `file://` URI is a legitimate operator capability (see
    // the admin-gate tests below), but a `..` component inside one is
    // still rejected — nothing in the design ever intended it.
    let r = client()
        .post(format!("{base}/v1/collections"))
        .json(&json!({
            "name": "ext_traversal_file",
            "fields": [],
            "external": {
                "format": "parquet",
                "uri": "file:///srv/data/../../etc/secret.parquet",
                "id_field": "id",
                "vector_field": "embedding",
                "metric": "Cosine"
            }
        }))
        .send()
        .await
        .unwrap();
    let status = r.status();
    let body = r.text().await.unwrap();
    assert_eq!(status, 400, "body: {body}");
    // Assert on the REJECTION's own wording, not on "..". A body containing
    // ".." proves nothing: with the traversal check removed the URI still
    // reaches the store and comes back as `Blob not found: ../../etc/...`,
    // which is also a 400 and also contains "..", so the weaker assertion
    // passed with the security fix deleted.
    assert!(
        body.contains("path component"),
        "traversal rejection absent from body: {body}"
    );
}

#[tokio::test]
async fn registering_an_external_collection_as_non_admin_is_rejected() {
    let mut users = HashMap::new();
    users.insert(
        "admin".to_owned(),
        UserRecord {
            password: "admin".into(),
            role: Role::Admin,
        },
    );
    users.insert(
        "viewer".to_owned(),
        UserRecord {
            password: "viewer".into(),
            role: Role::Viewer,
        },
    );

    let tmp = tempfile::tempdir().unwrap();
    let state = AppState::with_auth_and_users(tmp.path(), "ext-admin-gate-secret", users);
    let router = build_router(state);
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr: SocketAddr = listener.local_addr().unwrap();
    let base = format!("http://{addr}");
    tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });

    let viewer = login_user(&base, "viewer", "viewer").await;

    let data_dir = tempfile::tempdir().unwrap();
    let file_path = data_dir.path().join("data.parquet");
    write_parquet(&file_path, &["a", "b", "c"], Some(3));

    let r = client()
        .post(format!("{base}/v1/collections"))
        .header("Authorization", format!("Bearer {viewer}"))
        .json(&json!({
            "name": "ext_viewer_denied",
            "fields": [],
            "external": {
                "format": "parquet",
                "uri": file_uri(&file_path),
                "id_field": "id",
                "vector_field": "embedding",
                "metric": "Cosine"
            }
        }))
        .send()
        .await
        .unwrap();
    let status = r.status();
    let body = r.text().await.unwrap();
    assert_eq!(status, 403, "body: {body}");
    assert!(
        body.to_lowercase().contains("admin"),
        "admin explanation absent from body: {body}"
    );
}

#[tokio::test]
async fn registering_an_external_collection_as_admin_succeeds() {
    let mut users = HashMap::new();
    users.insert(
        "admin".to_owned(),
        UserRecord {
            password: "admin".into(),
            role: Role::Admin,
        },
    );

    let tmp = tempfile::tempdir().unwrap();
    let state = AppState::with_auth_and_users(tmp.path(), "ext-admin-gate-ok-secret", users);
    let router = build_router(state);
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr: SocketAddr = listener.local_addr().unwrap();
    let base = format!("http://{addr}");
    tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });

    let admin = login_user(&base, "admin", "admin").await;

    let data_dir = tempfile::tempdir().unwrap();
    let file_path = data_dir.path().join("data.parquet");
    write_parquet(&file_path, &["a", "b", "c"], Some(3));

    let r = client()
        .post(format!("{base}/v1/collections"))
        .header("Authorization", format!("Bearer {admin}"))
        .json(&json!({
            "name": "ext_admin_allowed",
            "fields": [],
            "external": {
                "format": "parquet",
                "uri": file_uri(&file_path),
                "id_field": "id",
                "vector_field": "embedding",
                "metric": "Cosine"
            }
        }))
        .send()
        .await
        .unwrap();
    let status = r.status();
    let body = r.text().await.unwrap();
    assert_eq!(status, 201, "body: {body}");
}

// ── Read endpoints require locally stored records (whole-branch review
// finding 2) ──────────────────────────────────────────────────────────────
//
// `materialize_collection_locally` runs unconditionally, so an external
// collection gets an (empty) segment manager just like an internal one.
// Without a guard, these five read endpoints find that empty manager and
// answer 200 with zero rows/results on a fully populated external dataset —
// indistinguishable from a correct empty-collection answer. Each is
// asserted 400 (not 405: these are reads, not mutations) with a body
// naming the reason.

#[tokio::test]
async fn external_records_endpoint_is_rejected() {
    let (base, _tmp) = start_server().await;
    let name = "ext_records_read";
    let _data = setup_external(&base, name).await;

    let r = client()
        .get(format!("{base}/v1/collections/{name}/records"))
        .send()
        .await
        .unwrap();
    let status = r.status();
    let body = r.text().await.unwrap();
    assert_eq!(status, 400, "body: {body}");
    assert!(
        body.to_lowercase().contains("locally stored"),
        "explanation absent from body: {body}"
    );
}

#[tokio::test]
async fn external_aggregate_endpoint_is_rejected() {
    let (base, _tmp) = start_server().await;
    let name = "ext_aggregate_read";
    let _data = setup_external(&base, name).await;

    let r = client()
        .get(format!(
            "{base}/v1/collections/{name}/aggregate?field=id&metric=count"
        ))
        .send()
        .await
        .unwrap();
    let status = r.status();
    let body = r.text().await.unwrap();
    assert_eq!(status, 400, "body: {body}");
    assert!(
        body.to_lowercase().contains("locally stored"),
        "explanation absent from body: {body}"
    );
}

#[tokio::test]
async fn external_histogram_endpoint_is_rejected() {
    let (base, _tmp) = start_server().await;
    let name = "ext_histogram_read";
    let _data = setup_external(&base, name).await;

    let r = client()
        .get(format!("{base}/v1/collections/{name}/histogram?field=id"))
        .send()
        .await
        .unwrap();
    let status = r.status();
    let body = r.text().await.unwrap();
    assert_eq!(status, 400, "body: {body}");
    assert!(
        body.to_lowercase().contains("locally stored"),
        "explanation absent from body: {body}"
    );
}

#[tokio::test]
async fn external_field_stats_endpoint_is_rejected() {
    let (base, _tmp) = start_server().await;
    let name = "ext_field_stats_read";
    let _data = setup_external(&base, name).await;

    let r = client()
        .get(format!("{base}/v1/collections/{name}/field-stats?field=id"))
        .send()
        .await
        .unwrap();
    let status = r.status();
    let body = r.text().await.unwrap();
    assert_eq!(status, 400, "body: {body}");
    assert!(
        body.to_lowercase().contains("locally stored"),
        "explanation absent from body: {body}"
    );
}

#[tokio::test]
async fn external_evaluate_endpoint_is_rejected() {
    let (base, _tmp) = start_server().await;
    let name = "ext_evaluate_read";
    let _data = setup_external(&base, name).await;

    let r = client()
        .get(format!("{base}/v1/collections/{name}/evaluate"))
        .send()
        .await
        .unwrap();
    let status = r.status();
    let body = r.text().await.unwrap();
    assert_eq!(status, 400, "body: {body}");
    assert!(
        body.to_lowercase().contains("locally stored"),
        "explanation absent from body: {body}"
    );
}

// ── RLS + malformed filter (whole-branch review finding 5) ────────────────

/// Before the fix, `effective_filter_or_reject`'s malformed-filter rejection
/// was inert whenever RLS was enabled: the RLS-injection block parsed
/// `req.filter` with `.ok()`, silently discarding a parse error and always
/// setting `req.filter_expr` to (at least) the owner predicate, so
/// `effective_filter_or_reject`'s `filter_expr`-first check never re-parsed
/// the raw string and never saw the error — the malformed filter was
/// dropped instead of rejected, and a non-admin caller got every one of
/// their own rows back with a `200` instead of the `400` the same filter
/// gets without RLS.
#[tokio::test]
async fn external_rls_malformed_filter_is_rejected() {
    let mut users = HashMap::new();
    users.insert(
        "admin".to_owned(),
        UserRecord {
            password: "admin".into(),
            role: Role::Admin,
        },
    );
    users.insert(
        "alice".to_owned(),
        UserRecord {
            password: "alice".into(),
            role: Role::Editor,
        },
    );

    let tmp = tempfile::tempdir().unwrap();
    let state = AppState::with_auth_and_users(tmp.path(), "ext-rls-filter-test-secret", users);
    let router = build_router(state);
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr: SocketAddr = listener.local_addr().unwrap();
    let base = format!("http://{addr}");
    tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });

    let admin = login_user(&base, "admin", "admin").await;
    let alice = login_user(&base, "alice", "alice").await;

    let data_dir = tempfile::tempdir().unwrap();
    let file_path = data_dir.path().join("data.parquet");
    write_parquet_with_owner(&file_path, &[("a1", "alice"), ("b1", "bob")], 3);

    let r = client()
        .post(format!("{base}/v1/collections"))
        .header("Authorization", format!("Bearer {admin}"))
        .json(&json!({
            "name": "ext_rls_bad_filter",
            "fields": [],
            "rls_owner_field": "owner",
            "external": {
                "format": "parquet",
                "uri": file_uri(&file_path),
                "id_field": "id",
                "vector_field": "embedding",
                "metric": "Cosine"
            }
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 201, "create: {}", r.text().await.unwrap());

    // Unterminated quote, same malformed filter `external_malformed_filter_is_rejected`
    // uses on a non-RLS collection — must be rejected identically here.
    let r = client()
        .post(format!("{base}/v1/collections/ext_rls_bad_filter/search"))
        .header("Authorization", format!("Bearer {alice}"))
        .json(&json!({
            "query": { "vector": { "field": "embedding", "values": row0_query(3) } },
            "filter": "id = 'a",
            "limit": 10
        }))
        .send()
        .await
        .unwrap();
    let status = r.status();
    let body = r.text().await.unwrap();
    assert_eq!(status, 400, "body: {body}");
    assert!(
        body.contains("filter"),
        "parse-failure explanation absent from body: {body}"
    );
}

// ── Coverage gaps (whole-branch review finding 6): 405 on the three
// mutating endpoints missing it ─────────────────────────────────────────

#[tokio::test]
async fn external_connector_run_is_rejected() {
    let (base, _tmp) = start_server().await;
    let name = "ext_connector_run";
    let _data = setup_external(&base, name).await;

    let r = client()
        .post(format!("{base}/v1/admin/connectors"))
        .json(&json!({
            "kind": "web",
            "name": "test connector",
            "config": { "urls": ["http://example.com"], "collection": name }
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(
        r.status(),
        201,
        "create connector: {}",
        r.text().await.unwrap()
    );
    let created: Value = r.json().await.unwrap();
    let id = created["id"].as_str().expect("connector id");

    let r = client()
        .post(format!("{base}/v1/admin/connectors/{id}/run"))
        .send()
        .await
        .unwrap();
    assert_rejected_as_external(r, "POST /connectors/{id}/run", name).await;
}

#[tokio::test]
async fn external_eval_set_generation_is_rejected() {
    let (base, _tmp) = start_server().await;
    let name = "ext_eval_set_gen";
    let _data = setup_external(&base, name).await;

    let r = client()
        .post(format!("{base}/v1/collections/{name}/eval-sets/generate"))
        .json(&json!({}))
        .send()
        .await
        .unwrap();
    assert_rejected_as_external(r, "POST /eval-sets/generate", name).await;
}

#[tokio::test]
async fn external_set_auto_optimization_is_rejected() {
    let (base, _tmp) = start_server().await;
    let name = "ext_auto_optimization";
    let _data = setup_external(&base, name).await;

    let r = client()
        .put(format!("{base}/v1/collections/{name}/auto-optimization"))
        .json(&json!({ "enabled": true }))
        .send()
        .await
        .unwrap();
    assert_rejected_as_external(r, "PUT /auto-optimization", name).await;
}

/// The agent tools must not answer an external collection out of the empty
/// local segment manager it gets from `materialize_collection_locally`.
///
/// REST guards its own read paths with `reject_if_external_for_read`, whose
/// docstring notes this guard "has twice shipped applied at only some of
/// several entry points". The agent and MCP surfaces were a third: nothing in
/// `agent/tools/builtins.rs` ever asked `is_external()`, so `filter_query`
/// and `get_record` read the empty manager and returned a perfectly
/// well-formed **200 with zero rows** on a fully populated dataset — a silent
/// wrong answer handed to an autonomous agent, which is worse than an error.
#[tokio::test]
async fn agent_tools_refuse_an_external_collection_rather_than_answer_empty() {
    let (base, tmp) = start_server().await;
    let file_path = tmp.path().join("ext_agent.parquet");
    write_parquet(&file_path, &["a", "b", "c"], Some(4));
    let r = register_external(&base, "ext_agent", &file_uri(&file_path)).await;
    assert!(
        r.status().is_success(),
        "registration failed: {}",
        r.text().await.unwrap()
    );

    let cases: Vec<(&str, serde_json::Value)> = vec![
        (
            "filter_query",
            json!({ "collection": "ext_agent", "filter": "id = 'a'", "limit": 5 }),
        ),
        (
            // A syntactically valid UUID: `get_record` validates the id before
            // it ever looks at the schema, so an unparseable one would fail on
            // argument validation and never reach the path under test.
            "get_record",
            json!({
                "collection": "ext_agent",
                "id": "00000000-0000-4000-8000-000000000001"
            }),
        ),
        (
            "search",
            json!({ "collection": "ext_agent", "query": "anything", "limit": 5 }),
        ),
    ];

    for (tool, args) in cases {
        let resp = reqwest::Client::new()
            .post(format!("{base}/v1/agent-tools/{tool}/execute"))
            .json(&args)
            .send()
            .await
            .unwrap();
        let status = resp.status().as_u16();
        let body = resp.text().await.unwrap();

        assert_ne!(
            status, 200,
            "{tool} answered 200 on an external collection; a zero-row success here is              indistinguishable from a genuinely empty collection: {body}"
        );
        assert!(
            body.to_lowercase().contains("external"),
            "{tool} must say the collection is external so the agent can act on it,              got {status}: {body}"
        );
    }
}

/// `describe_collection` is the tool an agent uses to find out what it is
/// looking at, so it must not report an external collection as an ordinary
/// empty one.
///
/// `record_count` is summed from the local segment manager, which an external
/// collection has but never fills — so before this it answered `0` for a fully
/// populated dataset, with nothing in the payload marking it external. Unlike
/// the record-reading tools this one is not refused: describing is exactly how
/// the agent is supposed to learn the collection's nature.
#[tokio::test]
async fn describe_collection_marks_an_external_collection_instead_of_reporting_zero_rows() {
    let (base, tmp) = start_server().await;
    let file_path = tmp.path().join("ext_describe.parquet");
    write_parquet(&file_path, &["a", "b", "c"], Some(4));
    let r = register_external(&base, "ext_describe", &file_uri(&file_path)).await;
    assert!(r.status().is_success());

    let resp = reqwest::Client::new()
        .post(format!("{base}/v1/agent-tools/describe_collection/execute"))
        .json(&json!({ "name": "ext_describe" }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status().as_u16(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();

    let ext = body
        .get("external")
        .unwrap_or_else(|| panic!("must be marked external: {body}"));
    // The source descriptor, not just a flag: an external collection declares
    // no fields of its own (registering one with `fields` is rejected — it
    // infers its schema from the source), so the descriptor is the only thing
    // that actually tells an agent what it is looking at.
    assert_eq!(
        ext.get("format").and_then(|v| v.as_str()),
        Some("parquet"),
        "external source descriptor must be reported: {body}"
    );
    assert!(
        ext.get("uri")
            .and_then(|v| v.as_str())
            .is_some_and(|u| u.ends_with(".parquet")),
        "descriptor must name the source: {body}"
    );
    assert!(
        body.get("record_count").is_none(),
        "a local record count is meaningless here and must not be reported: {body}"
    );
    assert!(
        body.get("note")
            .and_then(|v| v.as_str())
            .is_some_and(|n| n.contains("search")),
        "the note must point at the route that does work: {body}"
    );
}

// ── Silently-ignored request knobs (2026-09-09) ──────────────────────────
//
// `explain` and `truncate_to` were accepted and then ignored on external
// collections. That is worse than refusing them: the caller who asked for a
// plan got results with no plan and no reason, and the caller who asked to
// truncate got untruncated vectors. Both read as the feature working.

async fn ext_search(base: &str, coll: &str, body: Value) -> (u16, String) {
    let r = client()
        .post(format!("{base}/v1/collections/{coll}/search"))
        .json(&body)
        .send()
        .await
        .unwrap();
    let status = r.status().as_u16();
    (status, r.text().await.unwrap())
}

fn dense_query() -> Value {
    json!({ "vector": { "field": "embedding", "values": [1.0, 0.0, 0.0] } })
}

#[tokio::test]
async fn external_explain_is_rejected_not_ignored() {
    let (base, _tmp) = start_server().await;
    let data_dir = tempfile::tempdir().unwrap();
    let file_path = data_dir.path().join("data.parquet");
    write_parquet(&file_path, &["a", "b", "c"], Some(3));
    let r = register_external(&base, "ext_explain", &file_uri(&file_path)).await;
    assert_eq!(r.status(), 201, "create: {}", r.text().await.unwrap());

    let (status, body) = ext_search(
        &base,
        "ext_explain",
        json!({ "query": dense_query(), "limit": 10, "explain": true }),
    )
    .await;
    assert_eq!(status, 400, "explain should be refused: {body}");
    assert!(body.contains("explain"), "{body}");
    assert!(body.contains("ext_explain"), "{body}");
}

#[tokio::test]
async fn external_truncate_to_is_rejected_not_ignored() {
    let (base, _tmp) = start_server().await;
    let data_dir = tempfile::tempdir().unwrap();
    let file_path = data_dir.path().join("data.parquet");
    write_parquet(&file_path, &["a", "b", "c"], Some(3));
    let r = register_external(&base, "ext_trunc", &file_uri(&file_path)).await;
    assert_eq!(r.status(), 201, "create: {}", r.text().await.unwrap());

    let (status, body) = ext_search(
        &base,
        "ext_trunc",
        json!({ "query": dense_query(), "limit": 10, "truncate_to": 2 }),
    )
    .await;
    assert_eq!(status, 400, "truncate_to should be refused: {body}");
    assert!(body.contains("truncate_to"), "{body}");
}

/// An external collection has exactly one vector column. A query naming a
/// different field was searched against that column anyway, so a typo — or a
/// query written for another collection — returned confident results from the
/// wrong field.
#[tokio::test]
async fn external_rejects_a_query_naming_the_wrong_vector_field() {
    let (base, _tmp) = start_server().await;
    let data_dir = tempfile::tempdir().unwrap();
    let file_path = data_dir.path().join("data.parquet");
    write_parquet(&file_path, &["a", "b", "c"], Some(3));
    let r = register_external(&base, "ext_field", &file_uri(&file_path)).await;
    assert_eq!(r.status(), 201, "create: {}", r.text().await.unwrap());

    let (status, body) = ext_search(
        &base,
        "ext_field",
        json!({
            "query": { "vector": { "field": "not_the_column", "values": [1.0, 0.0, 0.0] } },
            "limit": 10
        }),
    )
    .await;
    assert_eq!(status, 400, "wrong field should be refused: {body}");
    assert!(
        body.contains("not_the_column") && body.contains("embedding"),
        "the message must name both the asked-for and the real column: {body}"
    );

    // The correct field still works — the check must not reject everything.
    let (status, body) = ext_search(
        &base,
        "ext_field",
        json!({ "query": dense_query(), "limit": 10 }),
    )
    .await;
    assert_eq!(status, 200, "the declared field must still work: {body}");
}

/// The id column belongs in `SearchHit.id`, not also inside `fields`.
/// Internal collections skip the identifier during hydration, so emitting it
/// here made external hits a different shape — a client iterating `fields`
/// saw a phantom column on external collections only.
#[tokio::test]
async fn external_hit_does_not_repeat_the_id_inside_fields() {
    let (base, _tmp) = start_server().await;
    let data_dir = tempfile::tempdir().unwrap();
    let file_path = data_dir.path().join("data.parquet");
    write_parquet(&file_path, &["a", "b", "c"], Some(3));
    let r = register_external(&base, "ext_id", &file_uri(&file_path)).await;
    assert_eq!(r.status(), 201, "create: {}", r.text().await.unwrap());

    let (status, body) = ext_search(
        &base,
        "ext_id",
        json!({ "query": dense_query(), "limit": 3 }),
    )
    .await;
    assert_eq!(status, 200, "search failed: {body}");
    let out: Value = serde_json::from_str(&body).unwrap();
    let results = out["results"].as_array().expect("results array");
    assert!(!results.is_empty(), "no hits: {body}");
    for hit in results {
        assert!(
            hit["id"].as_str().is_some(),
            "hit must carry its id at the top level: {hit}"
        );
        assert!(
            hit["fields"].get("id").is_none(),
            "the id column must not be repeated inside fields: {hit}"
        );
    }
}

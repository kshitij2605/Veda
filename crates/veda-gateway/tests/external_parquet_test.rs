//! Integration tests: registering an external Parquet collection via
//! `POST /v1/collections`. An external collection declares no Veda fields —
//! column names, types and the vector dimension come from the Parquet
//! schema itself, so `fields` is empty and an `external` block describes
//! the source instead.

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

use veda_gateway::rest::build_router;
use veda_gateway::rest::state::AppState;

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

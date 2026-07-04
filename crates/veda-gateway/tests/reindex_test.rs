//! Phase G atomic re-chunk smoke. POST /reindex creates name@v2
//! and points the alias at it. The full re-chunking pipeline
//! isn't part of this commit — the test only exercises the
//! atomic-swap primitive.

use std::net::SocketAddr;

use reqwest::Client;
use serde_json::{Value, json};
use tokio::net::TcpListener;

use veda_gateway::rest::build_router;
use veda_gateway::rest::state::AppState;

async fn start_server() -> (String, tempfile::TempDir) {
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

#[tokio::test]
async fn reindex_in_non_raft_mode_returns_target_status_alias_swapped() {
    let (base, _tmp) = start_server().await;
    let c = Client::new();

    let resp = c
        .post(format!("{base}/v1/collections"))
        .json(&json!({
            "name": "docs",
            "shard_count": 1,
            "fields": [
                { "name": "v", "type": "vector", "dimension": 2, "metric": "cosine" }
            ]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 201, "create: {}", resp.text().await.unwrap());

    let resp = c
        .post(format!("{base}/v1/collections/docs/reindex"))
        .send()
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        200,
        "reindex: {}",
        resp.text().await.unwrap()
    );
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["target"], json!("docs@v2"));
    // Non-Raft mode: the local AliasMap (Task 0.2's "Phase 0 fix") makes
    // the swap genuinely work without a Raft coordinator too, so both
    // modes report the same "alias_swapped" status.
    // Status string reports copy count too: "alias_swapped (copied N records)".
    let status = body["status"].as_str().unwrap();
    assert!(
        status.starts_with("alias_swapped"),
        "status must start with 'alias_swapped', got {status}"
    );
}

#[tokio::test]
async fn reindex_with_existing_v2_bumps_to_v3() {
    let (base, _tmp) = start_server().await;
    let c = Client::new();

    let resp = c
        .post(format!("{base}/v1/collections"))
        .json(&json!({
            "name": "logs@v2",
            "shard_count": 1,
            "fields": [
                { "name": "v", "type": "vector", "dimension": 2, "metric": "cosine" }
            ]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        201,
        "create v2: {}",
        resp.text().await.unwrap()
    );

    let resp = c
        .post(format!("{base}/v1/collections/logs@v2/reindex"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["target"], json!("logs@v3"));
}

#[tokio::test]
async fn get_alias_returns_self_when_no_alias_set() {
    let (base, _tmp) = start_server().await;
    let c = Client::new();

    let resp = c
        .post(format!("{base}/v1/collections"))
        .json(&json!({
            "name": "plain",
            "shard_count": 1,
            "fields": [
                { "name": "v", "type": "vector", "dimension": 2, "metric": "cosine" }
            ]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 201);

    let resp = c
        .get(format!("{base}/v1/admin/collections/plain/alias"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200, "{}", resp.text().await.unwrap());
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["alias"], json!("plain"));
    assert_eq!(body["target"], json!("plain"));
}

#[tokio::test]
async fn reindex_unknown_collection_returns_404() {
    let (base, _tmp) = start_server().await;
    let c = Client::new();

    let resp = c
        .post(format!("{base}/v1/collections/ghost/reindex"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404);
}

#[tokio::test]
async fn reindex_copies_source_records_to_sibling() {
    // Phase G real pipeline: /reindex must end with the sibling
    // holding every (non-tombstoned) record from the source.
    // Today the copy is straight; chunker / embedder transforms
    // slot in later. The visible-from-tests invariant is record
    // count + payload identity.
    let (base, _tmp) = start_server().await;
    let c = Client::new();
    let resp = c
        .post(format!("{base}/v1/collections"))
        .json(&json!({
            "name": "reindex_copy",
            "shard_count": 1,
            "fields": [
                { "name": "v", "type": "vector", "dimension": 2, "metric": "cosine" },
                { "name": "label", "type": "string", "filterable": true }
            ],
            "seed_records": [
                { "v": [1.0, 0.0], "label": "alpha" },
                { "v": [0.0, 1.0], "label": "beta" },
                { "v": [1.0, 1.0], "label": "gamma" }
            ]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 201);

    let resp = c
        .post(format!("{base}/v1/collections/reindex_copy/reindex"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["target"], json!("reindex_copy@v2"));
    // Status reports the copy count.
    assert!(
        body["status"]
            .as_str()
            .unwrap()
            .contains("copied 3 records"),
        "status should report copy count; got {body}"
    );

    // List the sibling — must hold all 3 records.
    let resp = c
        .get(format!(
            "{base}/v1/collections/reindex_copy@v2/records?limit=10"
        ))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200, "{}", resp.text().await.unwrap());
    let body: Value = resp.json().await.unwrap();
    assert_eq!(
        body["total"],
        json!(3),
        "sibling should hold all 3 source records; got {body}"
    );
}

#[tokio::test]
async fn reindex_rechunks_with_supplied_chunker_strategy() {
    // Phase G real pipeline: supply a FixedSize chunker that
    // splits at 50 chars. Source has one parent document split
    // into 3 long chunks; after re-chunk the sibling holds the
    // doc re-split at the new boundaries — more chunks, smaller
    // each.
    let (base, _tmp) = start_server().await;
    let c = Client::new();
    let resp = c
        .post(format!("{base}/v1/collections"))
        .json(&json!({
            "name": "rechunk_pipe",
            "shard_count": 1,
            "fields": [
                { "name": "v", "type": "vector", "dimension": 4, "metric": "cosine" },
                { "name": "content", "type": "string", "filterable": true },
                { "name": "parent_doc_id", "type": "string", "filterable": true },
                { "name": "chunk_index", "type": "int64", "filterable": true }
            ],
            "seed_records": [
                { "v": [1.0, 0.0, 0.0, 0.0], "content": "alpha alpha alpha alpha", "parent_doc_id": "doc_a", "chunk_index": 0 },
                { "v": [0.0, 1.0, 0.0, 0.0], "content": "beta beta beta beta",     "parent_doc_id": "doc_a", "chunk_index": 1 },
                { "v": [0.0, 0.0, 1.0, 0.0], "content": "gamma gamma gamma gamma", "parent_doc_id": "doc_a", "chunk_index": 2 }
            ]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 201, "create: {}", resp.text().await.unwrap());

    // /reindex with a FixedSize chunker. Re-embedding is OFF
    // because the test environment has no embedding endpoint
    // configured — chunks will land with zero-vector placeholders.
    let resp = c
        .post(format!("{base}/v1/collections/rechunk_pipe/reindex"))
        .json(&json!({
            "chunker": {
                "FixedSize": { "chunk_size": 30, "chunk_overlap": 0 }
            },
            "re_embed": false
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        200,
        "reindex: {}",
        resp.text().await.unwrap()
    );

    // The reconstructed parent text is the three chunks
    // concatenated with newlines. FixedSize(30) splits it into
    // multiple new chunks; sibling holds the rechunked version.
    let resp = c
        .get(format!(
            "{base}/v1/collections/rechunk_pipe@v2/records?limit=50"
        ))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200, "{}", resp.text().await.unwrap());
    let body: Value = resp.json().await.unwrap();
    let total = body["total"].as_u64().unwrap();
    assert!(
        total >= 2,
        "rechunked sibling should have ≥2 chunks at chunk_size=30; got {total} body={body}"
    );

    // Every new chunk has parent_doc_id = "doc_a" and a
    // chunk_index. Pin that the parent/index metadata
    // survived the re-chunk.
    let records = body["records"].as_array().unwrap();
    for rec in records {
        assert_eq!(
            rec["fields"]["parent_doc_id"],
            json!("doc_a"),
            "parent_doc_id should be preserved on re-chunk; got {rec}"
        );
        assert!(
            rec["fields"]["chunk_index"].is_number(),
            "chunk_index should be present; got {rec}"
        );
    }
}

#[tokio::test]
async fn reindex_re_embed_without_embedding_endpoint_returns_424() {
    // re_embed=true with no embedding endpoint configured =
    // 424 Failed Dependency with a helpful message.
    let (base, _tmp) = start_server().await;
    let c = Client::new();
    let resp = c
        .post(format!("{base}/v1/collections"))
        .json(&json!({
            "name": "re_embed_no_client",
            "shard_count": 1,
            "fields": [
                { "name": "v", "type": "vector", "dimension": 4, "metric": "cosine" },
                { "name": "content", "type": "string", "filterable": true },
                { "name": "parent_doc_id", "type": "string", "filterable": true }
            ],
            "seed_records": [
                { "v": [1.0, 0.0, 0.0, 0.0], "content": "alpha", "parent_doc_id": "doc_a" }
            ]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 201);
    let resp = c
        .post(format!("{base}/v1/collections/re_embed_no_client/reindex"))
        .json(&json!({
            "chunker": { "FixedSize": { "chunk_size": 20, "chunk_overlap": 0 } },
            "re_embed": true
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        424,
        "expected 424 FAILED_DEPENDENCY, got {}",
        resp.status()
    );
    let body: Value = resp.json().await.unwrap();
    assert!(body["error"].as_str().unwrap().contains("embedding"));
}

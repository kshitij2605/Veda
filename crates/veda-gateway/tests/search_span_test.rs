//! APM Phase 5.1 — asserts the `search` span carries the retrieval-quality
//! and `db.*` attributes.
//!
//! We install a process-global `tracing_subscriber` with a custom capture
//! `Layer` (the task's documented fallback to a full in-memory OTel
//! exporter). The layer records each span's static name plus the
//! string-rendered fields set both at span creation and via
//! `Span::record(...)`. Because it's a global subscriber it applies on the
//! axum worker threads that actually run the handler, so a normal HTTP
//! search request exercises the real code path. In production the
//! `tracing_opentelemetry` layer maps these same names+fields to OTel span
//! attributes verbatim.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex, OnceLock};

use reqwest::Client;
use serde_json::json;
use tokio::net::TcpListener;
use tracing::field::{Field, Visit};
use tracing::span::{Attributes, Id};
use tracing_subscriber::layer::{Context, Layer, SubscriberExt as _};
use tracing_subscriber::util::SubscriberInitExt as _;

use veda_gateway::rest::build_router;
use veda_gateway::rest::state::AppState;

#[derive(Default, Clone)]
struct CapturedSpan {
    name: String,
    fields: HashMap<String, String>,
}

#[derive(Clone, Default)]
struct SpanCollector {
    spans: Arc<Mutex<HashMap<u64, CapturedSpan>>>,
}

struct FieldStringifier<'a>(&'a mut HashMap<String, String>);
impl Visit for FieldStringifier<'_> {
    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        self.0.insert(field.name().to_owned(), format!("{value:?}"));
    }
    fn record_str(&mut self, field: &Field, value: &str) {
        self.0.insert(field.name().to_owned(), value.to_owned());
    }
    fn record_i64(&mut self, field: &Field, value: i64) {
        self.0.insert(field.name().to_owned(), value.to_string());
    }
    fn record_u64(&mut self, field: &Field, value: u64) {
        self.0.insert(field.name().to_owned(), value.to_string());
    }
    fn record_f64(&mut self, field: &Field, value: f64) {
        self.0.insert(field.name().to_owned(), value.to_string());
    }
    fn record_bool(&mut self, field: &Field, value: bool) {
        self.0.insert(field.name().to_owned(), value.to_string());
    }
}

impl<S: tracing::Subscriber> Layer<S> for SpanCollector {
    fn on_new_span(&self, attrs: &Attributes<'_>, id: &Id, _ctx: Context<'_, S>) {
        let mut cap = CapturedSpan {
            name: attrs.metadata().name().to_owned(),
            fields: HashMap::new(),
        };
        attrs.record(&mut FieldStringifier(&mut cap.fields));
        self.spans.lock().unwrap().insert(id.into_u64(), cap);
    }
    fn on_record(&self, id: &Id, values: &tracing::span::Record<'_>, _ctx: Context<'_, S>) {
        if let Some(cap) = self.spans.lock().unwrap().get_mut(&id.into_u64()) {
            values.record(&mut FieldStringifier(&mut cap.fields));
        }
    }
}

/// Global capture buffer — installed once for this test binary.
static CAPTURED: OnceLock<Arc<Mutex<HashMap<u64, CapturedSpan>>>> = OnceLock::new();

fn captured() -> Arc<Mutex<HashMap<u64, CapturedSpan>>> {
    CAPTURED
        .get_or_init(|| {
            let collector = SpanCollector::default();
            let buf = collector.spans.clone();
            tracing_subscriber::registry().with(collector).init();
            buf
        })
        .clone()
}

async fn start_server() -> (String, tempfile::TempDir) {
    let tmp = tempfile::tempdir().unwrap();
    let app_state = AppState::new(tmp.path());
    let router = build_router(app_state);
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr: SocketAddr = listener.local_addr().unwrap();
    let base = format!("http://{addr}");
    tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    (base, tmp)
}

#[tokio::test]
async fn search_span_carries_retrieval_and_db_attributes() {
    let buf = captured();
    let (base, _tmp) = start_server().await;
    let c = Client::new();

    c.post(format!("{base}/v1/collections"))
        .json(&json!({
            "name": "spans",
            "fields": [
                { "name": "title", "type": "string", "filterable": true },
                { "name": "embedding", "type": "vector", "dimension": 4, "metric": "cosine" }
            ],
            "shard_count": 1
        }))
        .send()
        .await
        .unwrap();
    let records: Vec<_> = (0..5)
        .map(|i| json!({ "title": format!("Item {i}"), "embedding": [1.0, i as f64 * 0.1, 0.0, 0.0] }))
        .collect();
    c.post(format!("{base}/v1/collections/spans/insert"))
        .json(&json!({ "records": records }))
        .send()
        .await
        .unwrap();

    let r = c
        .post(format!("{base}/v1/collections/spans/search"))
        .json(&json!({
            "query": { "vector": { "field": "embedding", "values": [1.0, 0.0, 0.0, 0.0] } },
            "limit": 3
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200, "{:?}", r.text().await);

    // Find the most-recent `search` span and assert its attributes.
    let spans = buf.lock().unwrap();
    let search = spans
        .values()
        .filter(|s| s.name == "search")
        .max_by_key(|s| s.fields.len())
        .expect("a `search` span was captured");

    assert_eq!(
        search.fields.get("db.system").map(String::as_str),
        Some("veda")
    );
    assert_eq!(
        search.fields.get("db.collection").map(String::as_str),
        Some("spans")
    );
    assert_eq!(
        search.fields.get("retrieval.query_type").map(String::as_str),
        Some("dense")
    );
    assert_eq!(
        search.fields.get("retrieval.top_k").map(String::as_str),
        Some("3")
    );
    assert_eq!(
        search.fields.get("retrieval.index_type").map(String::as_str),
        Some("hnsw")
    );
    // result_count is recorded on the Ok path; 3 records match within top_k.
    let rc: usize = search
        .fields
        .get("retrieval.result_count")
        .expect("result_count recorded")
        .parse()
        .unwrap();
    assert!(rc <= 3, "result_count={rc} within top_k");
}

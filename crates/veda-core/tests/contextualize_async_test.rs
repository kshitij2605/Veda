//! `contextualize_chunks` against a real listener, from inside a Tokio runtime.
//!
//! The point of this test is the *context it runs in*, not the assertion. The
//! function used to build a `reqwest::blocking::Client`; dropping that client's
//! internal runtime inside a Tokio worker panics with "Cannot drop a runtime in
//! a context where blocking is not allowed". A test that called the old version
//! from a plain `#[test]` would have passed — the panic needs an ambient
//! runtime to happen at all.
//!
//! That is exactly how the identical bug shipped in `splade_client` (PR #22):
//! every test in that module drove the parser directly and none made a request,
//! so a full green suite proved nothing about the path the product used.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use veda_core::chunking::{ChunkingStrategy, chunk_markdown, contextualize_chunks};

/// Minimal OpenAI-shaped `/v1/chat/completions` stub. Raw TCP rather than a
/// framework so this test adds no dependency to reach the real client code.
async fn start_llm_stub(reply: &'static str) -> (String, Arc<AtomicUsize>) {
    let calls = Arc::new(AtomicUsize::new(0));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let counter = calls.clone();
    tokio::spawn(async move {
        loop {
            let Ok((mut sock, _)) = listener.accept().await else {
                return;
            };
            counter.fetch_add(1, Ordering::SeqCst);
            // Drain the request head; we do not need to parse it.
            let mut buf = [0u8; 4096];
            let _ = sock.read(&mut buf).await;
            let body = format!(r#"{{"choices":[{{"message":{{"content":"{reply}"}}}}]}}"#);
            let resp = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\n\
                 content-length: {}\r\nconnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            let _ = sock.write_all(resp.as_bytes()).await;
            let _ = sock.shutdown().await;
        }
    });
    (format!("http://{addr}"), calls)
}

#[tokio::test]
async fn contextualize_chunks_runs_inside_a_tokio_runtime() {
    let (endpoint, calls) = start_llm_stub("A note about valves.").await;

    let chunks = chunk_markdown(
        "Valves are inspected quarterly. Bolts are inspected annually.",
        &ChunkingStrategy::FixedSize {
            chunk_size: 30,
            chunk_overlap: 0,
        },
    );
    assert!(chunks.len() >= 2, "test needs multiple chunks");
    let n = chunks.len();

    let out =
        contextualize_chunks("Maintenance manual.", chunks, &endpoint, "test-model", None).await;

    assert_eq!(out.len(), n, "every chunk must survive");
    assert_eq!(
        calls.load(Ordering::SeqCst),
        n,
        "one LLM call per chunk, over real HTTP"
    );
    for c in &out {
        assert!(
            c.text.starts_with("A note about valves."),
            "context should be prepended: {:?}",
            c.text
        );
        assert!(
            c.content_types.iter().any(|t| t == "contextualised"),
            "chunk should be tagged: {:?}",
            c.content_types
        );
    }
}

/// A failing endpoint must lose the enrichment, never the chunk.
#[tokio::test]
async fn a_dead_endpoint_returns_the_chunks_unchanged() {
    // Port 1 on loopback: nothing listens, connection refused immediately.
    let chunks = chunk_markdown(
        "Valves are inspected quarterly. Bolts are inspected annually.",
        &ChunkingStrategy::FixedSize {
            chunk_size: 30,
            chunk_overlap: 0,
        },
    );
    let n = chunks.len();
    let original: Vec<String> = chunks.iter().map(|c| c.text.clone()).collect();

    let out = contextualize_chunks(
        "Maintenance manual.",
        chunks,
        "http://127.0.0.1:1",
        "test-model",
        None,
    )
    .await;

    assert_eq!(out.len(), n, "a failed call must not drop chunks");
    for (c, before) in out.iter().zip(original.iter()) {
        assert_eq!(&c.text, before, "text must be untouched on failure");
        assert!(
            !c.content_types.iter().any(|t| t == "contextualised"),
            "a failed chunk must not be tagged as contextualised"
        );
    }
}

/// An empty endpoint short-circuits without touching the network.
#[tokio::test]
async fn an_empty_endpoint_is_a_no_op() {
    let chunks = chunk_markdown(
        "Valves are inspected quarterly.",
        &ChunkingStrategy::FixedSize {
            chunk_size: 30,
            chunk_overlap: 0,
        },
    );
    let n = chunks.len();
    let out = contextualize_chunks("doc", chunks, "", "m", None).await;
    assert_eq!(out.len(), n);
}

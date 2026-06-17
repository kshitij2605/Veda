//! Self-service service / dependency map (`GET /v1/admin/depmap`).
//!
//! Returns a declarative graph of Veda's internal request topology —
//! gateway → query executor → shards / vector index → embedding / LLM /
//! rerank → blob store / NATS — so an operator can see, in Niriksha, what a
//! search or RAG request fans out to without reverse-engineering the code.
//!
//! Admin-only — registered under the protected router with an extra
//! `rbac::require_admin` route layer, mirroring [`crate::rest::correlations`].
//!
//! The graph is **static / declarative** (v1): it describes the
//! architecture, not live per-node health. Liveness for the external
//! dependencies is already surfaced by `/v1/admin/readiness` and the
//! `dependency_down` advisor rule; the dashboard can overlay those.

use axum::Json;
use serde::Serialize;

/// A node in the dependency graph.
#[derive(Debug, Clone, Serialize)]
pub struct Node {
    /// Stable identifier, referenced by edges.
    pub id: &'static str,
    /// Coarse kind, used by the UI to group / color nodes.
    /// One of: `gateway` | `query` | `storage` | `index` | `external`.
    pub kind: &'static str,
    /// Human-readable label.
    pub label: &'static str,
}

/// A directed dependency edge `from → to`.
#[derive(Debug, Clone, Serialize)]
pub struct Edge {
    pub from: &'static str,
    pub to: &'static str,
    /// Optional edge annotation (e.g. the operation that crosses it).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label: Option<&'static str>,
}

#[derive(Debug, Serialize)]
pub struct DepMapResponse {
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
}

const fn node(id: &'static str, kind: &'static str, label: &'static str) -> Node {
    Node { id, kind, label }
}

const fn edge(from: &'static str, to: &'static str, label: Option<&'static str>) -> Edge {
    Edge { from, to, label }
}

/// The declarative dependency graph. Kept as a function (not a `const`)
/// because `Vec` can't be `const`; the contents are all `'static`.
fn graph() -> DepMapResponse {
    let nodes = vec![
        node("gateway", "gateway", "REST Gateway"),
        node("query", "query", "Query Executor"),
        node("shards", "query", "Shards"),
        node("index", "index", "Vector Index"),
        node("embedding", "external", "Embedding Service"),
        node("llm", "external", "LLM Service"),
        node("rerank", "external", "Reranker"),
        node("blob", "storage", "Blob Store (MinIO/S3)"),
        node("nats", "storage", "NATS (log replication)"),
    ];
    let edges = vec![
        edge("gateway", "query", Some("search")),
        edge("gateway", "embedding", Some("embed")),
        edge("gateway", "llm", Some("complete")),
        edge("gateway", "rerank", Some("rerank")),
        edge("gateway", "blob", Some("blob get/put")),
        edge("gateway", "nats", Some("replicate")),
        edge("query", "shards", Some("fan-out")),
        edge("shards", "index", Some("ann search")),
    ];
    DepMapResponse { nodes, edges }
}

/// `GET /v1/admin/depmap` — declarative service / dependency graph.
pub async fn depmap() -> Json<DepMapResponse> {
    Json(graph())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn graph_is_well_formed() {
        let g = graph();
        // Every edge endpoint references a declared node id.
        let ids: std::collections::HashSet<&str> = g.nodes.iter().map(|n| n.id).collect();
        for e in &g.edges {
            assert!(ids.contains(e.from), "edge.from `{}` is a node", e.from);
            assert!(ids.contains(e.to), "edge.to `{}` is a node", e.to);
        }
        // The core topology nodes are present.
        for must in ["gateway", "query", "index", "embedding", "blob", "nats"] {
            assert!(ids.contains(must), "node `{must}` present");
        }
    }
}

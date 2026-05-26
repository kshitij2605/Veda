//! Logical multi-tenancy v1 — `X-Veda-Tenant` header scoping.
//!
//! Each request carries a tenant id (default `"default"`). Internal
//! storage keys (schemas, segment managers, query executors) are
//! prefixed with the tenant so two tenants writing to a collection
//! named `docs` get two independent collections.
//!
//! What this v1 covers:
//!   * REST-layer isolation: tenant A's collection list, record reads,
//!     and searches never see tenant B's data.
//!   * On-disk separation is implicit via the prefixed key — segments
//!     live at `data_dir/segments/{tenant}__{collection}/...`.
//!
//! What this v1 does NOT cover (deferred):
//!   * Per-tenant Raft groups (everything still goes through the
//!     cluster-wide group).
//!   * Cryptographic isolation (per-tenant KMS keys).
//!   * Tenant lifecycle API (provision, suspend, delete).
//!   * Hierarchical namespaces (Pulsar-shape `tenant/namespace/collection`).
//!
//! The "Curator-shape" partitioned and "Weaviate-shape" dedicated-shard
//! tiers from the research doc are also v2+.

use axum::extract::{FromRequestParts, Request, State};
use axum::http::request::Parts;
use axum::http::StatusCode;
use axum::middleware::Next;
use axum::response::Response;

use crate::rest::state::AppState;

pub const DEFAULT_TENANT: &str = "default";
pub const HEADER_NAME: &str = "x-veda-tenant";

/// Tenant id. Validated to `[a-zA-Z0-9_-]{1,64}` so it can safely
/// share filesystem paths and the `__` separator used in the
/// internal scoped key.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TenantId(String);

impl TenantId {
    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn default_tenant() -> Self {
        Self(DEFAULT_TENANT.to_owned())
    }

    /// Parse + validate. Returns `Err(error_string)` on invalid input.
    pub fn parse(raw: &str) -> Result<Self, String> {
        if raw.is_empty() {
            return Err("tenant id is empty".into());
        }
        if raw.len() > 64 {
            return Err(format!("tenant id too long ({}>64)", raw.len()));
        }
        for c in raw.chars() {
            if !(c.is_ascii_alphanumeric() || c == '_' || c == '-') {
                return Err(format!(
                    "tenant id contains invalid character {c:?}; only a-z A-Z 0-9 _ - allowed"
                ));
            }
        }
        Ok(Self(raw.to_owned()))
    }
}

/// Axum extractor: pulls TenantId from request extensions. Returns
/// the default tenant if the middleware didn't run (covers handler
/// unit tests that mount handlers without the full router).
pub struct Tenant(pub TenantId);

impl<S: Send + Sync> FromRequestParts<S> for Tenant {
    type Rejection = std::convert::Infallible;

    async fn from_request_parts(parts: &mut Parts, _state: &S) -> Result<Self, Self::Rejection> {
        Ok(Tenant(
            parts
                .extensions
                .get::<TenantId>()
                .cloned()
                .unwrap_or_else(TenantId::default_tenant),
        ))
    }
}

/// Middleware that reads `X-Veda-Tenant`, validates it, and stashes
/// a [`TenantId`] in the request extensions. Missing header →
/// default tenant. Invalid header → 400.
pub async fn tenant_middleware(
    State(_state): State<AppState>,
    mut req: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    let raw = req
        .headers()
        .get(HEADER_NAME)
        .and_then(|v| v.to_str().ok())
        .map(str::trim)
        .filter(|s| !s.is_empty());
    let tenant = match raw {
        None => TenantId::default_tenant(),
        Some(s) => match TenantId::parse(s) {
            Ok(t) => t,
            Err(_) => return Err(StatusCode::BAD_REQUEST),
        },
    };
    req.extensions_mut().insert(tenant);
    Ok(next.run(req).await)
}

/// Compose a tenant-scoped internal key. Used as the canonical key
/// in `schemas`, `segment_managers`, `query_executors`, and
/// `raft_materialized_records` maps. Tenant id is validated to a
/// charset that excludes `__`, so the prefix is unambiguous.
pub fn scope(tenant: &TenantId, collection: &str) -> String {
    format!("{}__{}", tenant.0, collection)
}

/// Strip the tenant prefix from a scoped collection name for
/// user-facing display (metrics, logs, API responses). Falls back
/// to the raw name if no `__` separator is found.
pub fn display_name(scoped: &str) -> &str {
    scoped.split_once("__").map(|(_, name)| name).unwrap_or(scoped)
}

/// Inverse of [`scope`]. Returns `Some(collection_name)` when
/// `key` is scoped under `tenant`, else `None`. Used by
/// `list_collections` to filter + strip the prefix.
pub fn strip_prefix<'a>(tenant: &TenantId, key: &'a str) -> Option<&'a str> {
    let prefix = format!("{}__", tenant.0);
    key.strip_prefix(&prefix)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_accepts_valid_ids() {
        TenantId::parse("default").unwrap();
        TenantId::parse("acme-corp").unwrap();
        TenantId::parse("tenant_123").unwrap();
        TenantId::parse("ABC").unwrap();
    }

    #[test]
    fn parse_rejects_invalid() {
        assert!(TenantId::parse("").is_err());
        assert!(TenantId::parse("has space").is_err());
        assert!(TenantId::parse("has/slash").is_err());
        assert!(TenantId::parse("has:colon").is_err());
        assert!(TenantId::parse(&"x".repeat(65)).is_err());
    }

    #[test]
    fn scope_and_strip_roundtrip() {
        let t = TenantId::parse("acme").unwrap();
        let k = scope(&t, "docs");
        assert_eq!(k, "acme__docs");
        assert_eq!(strip_prefix(&t, &k), Some("docs"));
        // wrong tenant → None
        let other = TenantId::parse("rival").unwrap();
        assert_eq!(strip_prefix(&other, &k), None);
        // unrelated key
        assert_eq!(strip_prefix(&t, "unrelated"), None);
    }

    #[test]
    fn scope_handles_collection_names_with_underscores() {
        // collection names can themselves contain `__` because the
        // first `__` is the separator and we only do prefix
        // matching — never round-trip parsing of the right half.
        let t = TenantId::parse("acme").unwrap();
        let k = scope(&t, "user__documents");
        assert_eq!(k, "acme__user__documents");
        assert_eq!(strip_prefix(&t, &k), Some("user__documents"));
    }
}

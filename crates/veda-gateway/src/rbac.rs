use axum::extract::Request;
use axum::http::StatusCode;
use axum::middleware::Next;
use axum::response::Response;

use crate::auth::{Claims, Role};

/// Check if a role has permission for a given action.
pub fn check_permission(role: Role, action: &str) -> bool {
    match role {
        Role::Admin => true,
        Role::Editor => matches!(
            action,
            "search"
                | "insert"
                | "upsert"
                | "delete_records"
                | "embed_insert"
                | "list_collections"
                | "describe_collection"
                | "get_record"
        ),
        Role::Viewer => matches!(
            action,
            "search"
                | "get_record"
                | "list_collections"
                | "describe_collection"
        ),
    }
}

/// Create an axum middleware layer that enforces the required role for an action.
///
/// If no Claims are present in extensions (auth disabled), the request is allowed.
pub async fn require_role(
    action: &str,
    req: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    if let Some(claims) = req.extensions().get::<Claims>()
        && !check_permission(claims.role, action)
    {
        return Err(StatusCode::FORBIDDEN);
    }
    Ok(next.run(req).await)
}

/// Helper to build a closure suitable for `axum::middleware::from_fn`.
///
/// Usage: `.route_layer(axum::middleware::from_fn(require_action("search")))`
///
/// Since axum's `from_fn` needs a function with a specific signature and we want
/// to parameterize by action, we return a boxed future via a macro-like pattern.
/// However, the simplest approach is individual functions per action.
pub async fn require_admin(req: Request, next: Next) -> Result<Response, StatusCode> {
    require_role("admin", req, next).await
}

pub async fn require_search(req: Request, next: Next) -> Result<Response, StatusCode> {
    require_role("search", req, next).await
}

pub async fn require_insert(req: Request, next: Next) -> Result<Response, StatusCode> {
    require_role("insert", req, next).await
}

pub async fn require_delete_records(req: Request, next: Next) -> Result<Response, StatusCode> {
    require_role("delete_records", req, next).await
}

pub async fn require_list_collections(req: Request, next: Next) -> Result<Response, StatusCode> {
    require_role("list_collections", req, next).await
}

pub async fn require_describe_collection(
    req: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    require_role("describe_collection", req, next).await
}

pub async fn require_get_record(req: Request, next: Next) -> Result<Response, StatusCode> {
    require_role("get_record", req, next).await
}

pub async fn require_embed_insert(req: Request, next: Next) -> Result<Response, StatusCode> {
    require_role("embed_insert", req, next).await
}

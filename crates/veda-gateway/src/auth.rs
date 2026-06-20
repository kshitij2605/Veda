use axum::extract::{Request, State};
use axum::http::StatusCode;
use axum::middleware::Next;
use axum::response::Response;
use jsonwebtoken::{DecodingKey, EncodingKey, Header, Validation};
use serde::{Deserialize, Serialize};

use crate::rest::state::AppState;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Claims {
    pub sub: String,
    pub role: Role,
    pub exp: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Role {
    Admin,
    Editor,
    Viewer,
}

/// Generate a JWT token for the given user and role.
pub fn generate_token(user_id: &str, role: Role, secret: &str) -> Result<String, StatusCode> {
    let exp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
        + 3600; // 1 hour

    let claims = Claims {
        sub: user_id.to_owned(),
        role,
        exp,
    };

    jsonwebtoken::encode(
        &Header::default(),
        &claims,
        &EncodingKey::from_secret(secret.as_bytes()),
    )
    .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
}

/// Verify and decode a JWT token, returning the claims.
pub fn verify_token(token: &str, secret: &str) -> Result<Claims, StatusCode> {
    let data = jsonwebtoken::decode::<Claims>(
        token,
        &DecodingKey::from_secret(secret.as_bytes()),
        &Validation::default(),
    )
    .map_err(|_| StatusCode::UNAUTHORIZED)?;

    Ok(data.claims)
}

/// Axum middleware that extracts and validates a bearer JWT.
///
/// Dispatches on `AppStateInner::auth_mode` (set from
/// `VEDA_AUTH_MODE` env):
///   - `Disabled` — no-op
///   - `Static`   — HS256 verify against `jwt_secret` (backwards-
///     compatible with the original path)
///   - `Oidc(..)` — verify against IdP-issued public key from JWKS
pub async fn auth_middleware(
    State(state): State<AppState>,
    mut req: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    let mode = state.inner.auth_mode.read().await.clone();
    let claims = match mode {
        crate::auth_oidc::AuthMode::Disabled => return Ok(next.run(req).await),
        crate::auth_oidc::AuthMode::Static => {
            // Legacy bool stays the source of truth here so tests
            // that pre-date AuthMode and set `auth_enabled=false`
            // directly still pass through.
            if !state.inner.auth_enabled {
                return Ok(next.run(req).await);
            }
            let token = extract_bearer(&req)?;
            verify_token(token, &state.inner.jwt_secret)?
        }
        crate::auth_oidc::AuthMode::Oidc(ctx) => {
            let token = extract_bearer(&req)?;
            crate::auth_oidc::verify_oidc(token, &ctx).await?
        }
    };
    req.extensions_mut().insert(claims);
    Ok(next.run(req).await)
}

fn extract_bearer(req: &Request) -> Result<&str, StatusCode> {
    let h = req
        .headers()
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .ok_or(StatusCode::UNAUTHORIZED)?;
    h.strip_prefix("Bearer ").ok_or(StatusCode::UNAUTHORIZED)
}

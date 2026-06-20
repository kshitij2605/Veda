//! IdP-delegated JWT validation via JWKS.
//!
//! Adds an OIDC mode on top of the existing static-secret HS256 JWT
//! path (auth.rs). When `VEDA_AUTH_MODE=oidc` is set, the gateway
//! discovers the IdP's JWKS endpoint, caches the public keys, and
//! validates incoming bearer tokens against them — no shared
//! secret with the IdP required.
//!
//! Configuration env vars:
//!   VEDA_AUTH_MODE         disabled | static | oidc       default: disabled
//!   VEDA_OIDC_ISSUER       https://idp.example.com         required for oidc
//!   VEDA_OIDC_AUDIENCE     veda                            required for oidc
//!   VEDA_OIDC_JWKS_URL     optional override; default is
//!                          discovered from
//!                          $issuer/.well-known/openid-configuration
//!   VEDA_OIDC_ROLES_CLAIM  name of the claim carrying the
//!                          user's roles                    default: roles
//!   VEDA_OIDC_JWKS_TTL_SECS    JWKS cache TTL              default: 600

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::http::StatusCode;
use jsonwebtoken::{DecodingKey, Validation, decode, decode_header, jwk::JwkSet};
use serde::Deserialize;
use tokio::sync::RwLock;

use crate::auth::{Claims, Role};

/// How the gateway authenticates incoming requests.
#[derive(Clone, Default)]
pub enum AuthMode {
    /// No authentication. All requests pass through.
    #[default]
    Disabled,
    /// HS256 tokens signed with a shared static secret. Useful for
    /// tests + early bootstrap; not the production target.
    Static,
    /// JWTs minted by an external IdP, validated against its JWKS.
    Oidc(Arc<OidcContext>),
}

impl std::fmt::Debug for AuthMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Disabled => f.write_str("Disabled"),
            Self::Static => f.write_str("Static"),
            Self::Oidc(_) => f.write_str("Oidc(..)"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct OidcConfig {
    pub issuer: String,
    pub audience: String,
    pub jwks_url: String,
    pub roles_claim: String,
    pub jwks_ttl: Duration,
}

impl OidcConfig {
    /// Build a config from `VEDA_OIDC_*` env vars. Returns
    /// `Ok(None)` when `VEDA_AUTH_MODE` is not `oidc`. Returns an
    /// error string when oidc is requested but a required var is
    /// missing or discovery fails.
    pub async fn from_env() -> Result<Option<Self>, String> {
        let mode = std::env::var("VEDA_AUTH_MODE").unwrap_or_default();
        if mode != "oidc" {
            return Ok(None);
        }
        let issuer = std::env::var("VEDA_OIDC_ISSUER")
            .map_err(|_| "VEDA_OIDC_ISSUER required when VEDA_AUTH_MODE=oidc".to_owned())?;
        let audience = std::env::var("VEDA_OIDC_AUDIENCE")
            .map_err(|_| "VEDA_OIDC_AUDIENCE required when VEDA_AUTH_MODE=oidc".to_owned())?;
        let jwks_url = match std::env::var("VEDA_OIDC_JWKS_URL") {
            Ok(u) => u,
            Err(_) => discover_jwks_url(&issuer).await?,
        };
        let roles_claim =
            std::env::var("VEDA_OIDC_ROLES_CLAIM").unwrap_or_else(|_| "roles".to_owned());
        let jwks_ttl = std::env::var("VEDA_OIDC_JWKS_TTL_SECS")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .map(Duration::from_secs)
            .unwrap_or_else(|| Duration::from_secs(600));
        Ok(Some(Self {
            issuer,
            audience,
            jwks_url,
            roles_claim,
            jwks_ttl,
        }))
    }
}

async fn discover_jwks_url(issuer: &str) -> Result<String, String> {
    let url = format!(
        "{}/.well-known/openid-configuration",
        issuer.trim_end_matches('/')
    );
    #[derive(Deserialize)]
    struct Discovery {
        jwks_uri: String,
    }
    let resp = reqwest::Client::new()
        .get(&url)
        .send()
        .await
        .map_err(|e| format!("OIDC discovery GET {url} failed: {e}"))?;
    if !resp.status().is_success() {
        return Err(format!(
            "OIDC discovery GET {url} returned {}",
            resp.status()
        ));
    }
    let body: Discovery = resp
        .json()
        .await
        .map_err(|e| format!("OIDC discovery body parse failed: {e}"))?;
    Ok(body.jwks_uri)
}

/// Live OIDC context owned by `AppState`. Holds the config + the
/// JWKS cache.
pub struct OidcContext {
    pub config: OidcConfig,
    pub jwks: JwksCache,
}

impl OidcContext {
    pub fn new(config: OidcConfig) -> Self {
        let cache = JwksCache::new(config.jwks_url.clone(), config.jwks_ttl);
        Self {
            config,
            jwks: cache,
        }
    }
}

/// Refreshable cache of JWKS keys. Refresh policy:
///   - on miss for unknown `kid`, refresh once before failing
///   - on TTL expiry, refresh on next read
///   - cached `DecodingKey` so the hot path doesn't re-parse JWK on
///     every request
pub struct JwksCache {
    jwks_url: String,
    ttl: Duration,
    state: RwLock<CacheState>,
}

struct CacheState {
    keys: HashMap<String, DecodingKey>,
    fetched_at: Option<Instant>,
}

impl JwksCache {
    pub fn new(jwks_url: String, ttl: Duration) -> Self {
        Self {
            jwks_url,
            ttl,
            state: RwLock::new(CacheState {
                keys: HashMap::new(),
                fetched_at: None,
            }),
        }
    }

    /// Look up a decoding key by `kid`. Refreshes the cache if (a)
    /// the kid is unknown, (b) the cache is empty, or (c) the cache
    /// is past its TTL.
    pub async fn get(&self, kid: &str) -> Option<DecodingKey> {
        if self.needs_refresh(kid).await
            && let Err(e) = self.refresh().await
        {
            tracing::warn!(error = %e, "JWKS refresh failed");
        }
        let state = self.state.read().await;
        state.keys.get(kid).cloned()
    }

    async fn needs_refresh(&self, kid: &str) -> bool {
        let state = self.state.read().await;
        match state.fetched_at {
            None => true,
            Some(_) if !state.keys.contains_key(kid) => true,
            Some(t) if t.elapsed() >= self.ttl => true,
            _ => false,
        }
    }

    pub async fn refresh(&self) -> Result<(), String> {
        let body: JwkSet = reqwest::Client::new()
            .get(&self.jwks_url)
            .send()
            .await
            .map_err(|e| format!("JWKS GET {} failed: {e}", self.jwks_url))?
            .json()
            .await
            .map_err(|e| format!("JWKS parse failed: {e}"))?;
        let mut keys = HashMap::new();
        for jwk in &body.keys {
            let Some(kid) = jwk.common.key_id.clone() else {
                continue;
            };
            match DecodingKey::from_jwk(jwk) {
                Ok(k) => {
                    keys.insert(kid, k);
                }
                Err(e) => {
                    tracing::warn!(error = %e, "skipping JWK with unsupported algorithm");
                }
            }
        }
        let n = keys.len();
        let mut state = self.state.write().await;
        state.keys = keys;
        state.fetched_at = Some(Instant::now());
        tracing::info!(keys = n, "JWKS refreshed");
        Ok(())
    }

    #[cfg(test)]
    pub async fn install_for_test(&self, keys: HashMap<String, DecodingKey>) {
        let mut state = self.state.write().await;
        state.keys = keys;
        state.fetched_at = Some(Instant::now());
    }
}

/// Token claims as received from the IdP. We deserialise the few
/// we need by name and stash the rest under `extra` for role
/// mapping.
#[derive(Debug, Deserialize)]
pub struct OidcClaims {
    pub sub: String,
    pub exp: u64,
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

/// Verify a bearer token against the OIDC config + JWKS cache.
/// Returns the existing Veda `Claims` shape so downstream handlers
/// continue to work without changes.
pub async fn verify_oidc(token: &str, ctx: &OidcContext) -> Result<Claims, StatusCode> {
    let header = decode_header(token).map_err(|e| {
        tracing::debug!(error = %e, "token header parse failed");
        StatusCode::UNAUTHORIZED
    })?;
    let kid = header.kid.ok_or_else(|| {
        tracing::debug!("token missing kid header");
        StatusCode::UNAUTHORIZED
    })?;
    let key = ctx.jwks.get(&kid).await.ok_or_else(|| {
        tracing::debug!(kid = %kid, "no JWK matching kid");
        StatusCode::UNAUTHORIZED
    })?;
    let mut validation = Validation::new(header.alg);
    validation.set_audience(&[&ctx.config.audience]);
    validation.set_issuer(&[&ctx.config.issuer]);
    validation.validate_exp = true;
    validation.leeway = 30;
    let data = decode::<OidcClaims>(token, &key, &validation).map_err(|e| {
        tracing::debug!(error = %e, "OIDC JWT verify failed");
        StatusCode::UNAUTHORIZED
    })?;
    let claims = data.claims;
    let role = map_role(&claims, &ctx.config.roles_claim);
    Ok(Claims {
        sub: claims.sub,
        role,
        exp: claims.exp,
    })
}

/// Map an IdP's claims to a Veda role. Looks at `roles_claim`,
/// expects either a string or an array of strings. First role
/// matching "admin"/"editor"/"viewer" (substring, case-insensitive)
/// wins. Defaults to Viewer when no claim matches.
fn map_role(claims: &OidcClaims, roles_claim: &str) -> Role {
    let Some(val) = claims.extra.get(roles_claim) else {
        return Role::Viewer;
    };
    let candidates: Vec<String> = match val {
        serde_json::Value::Array(arr) => arr
            .iter()
            .filter_map(|v| v.as_str().map(str::to_owned))
            .collect(),
        serde_json::Value::String(s) => vec![s.clone()],
        _ => vec![],
    };
    for c in candidates {
        let lower = c.to_ascii_lowercase();
        if lower.contains("admin") {
            return Role::Admin;
        }
        if lower.contains("editor") || lower.contains("writer") {
            return Role::Editor;
        }
        if lower.contains("viewer") || lower.contains("reader") {
            return Role::Viewer;
        }
    }
    Role::Viewer
}

#[cfg(test)]
mod tests {
    use super::*;
    use jsonwebtoken::{Algorithm, EncodingKey, Header, encode};

    fn rsa_keypair() -> (EncodingKey, DecodingKey, String) {
        let priv_pem = include_str!("testdata/jwt_test_priv.pem");
        let pub_pem = include_str!("testdata/jwt_test_pub.pem");
        let enc = EncodingKey::from_rsa_pem(priv_pem.as_bytes()).expect("priv");
        let dec = DecodingKey::from_rsa_pem(pub_pem.as_bytes()).expect("pub");
        (enc, dec, "test-kid".to_owned())
    }

    fn now() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
    }

    async fn install(ctx: &OidcContext, kid: &str, dec: DecodingKey) {
        let mut keys = HashMap::new();
        keys.insert(kid.to_owned(), dec);
        ctx.jwks.install_for_test(keys).await;
    }

    fn cfg() -> OidcConfig {
        OidcConfig {
            issuer: "https://idp.test".to_owned(),
            audience: "veda".to_owned(),
            jwks_url: "http://unused".to_owned(),
            roles_claim: "roles".to_owned(),
            jwks_ttl: Duration::from_secs(60),
        }
    }

    fn sign(enc: &EncodingKey, kid: &str, claim: serde_json::Value) -> String {
        let mut header = Header::new(Algorithm::RS256);
        header.kid = Some(kid.to_owned());
        encode(&header, &claim, enc).expect("encode")
    }

    #[tokio::test]
    async fn verify_oidc_happy_path_editor() {
        let (enc, dec, kid) = rsa_keypair();
        let ctx = OidcContext::new(cfg());
        install(&ctx, &kid, dec).await;
        let claim = serde_json::json!({
            "sub": "alice",
            "iss": "https://idp.test",
            "aud": "veda",
            "exp": now() + 60,
            "iat": now(),
            "roles": ["editor", "veda_user"],
        });
        let token = sign(&enc, &kid, claim);
        let claims = verify_oidc(&token, &ctx).await.expect("verify ok");
        assert_eq!(claims.sub, "alice");
        assert_eq!(claims.role, Role::Editor);
    }

    #[tokio::test]
    async fn verify_oidc_role_admin() {
        let (enc, dec, kid) = rsa_keypair();
        let ctx = OidcContext::new(cfg());
        install(&ctx, &kid, dec).await;
        let claim = serde_json::json!({
            "sub": "bob",
            "iss": "https://idp.test",
            "aud": "veda",
            "exp": now() + 60,
            "roles": ["VEDA_ADMIN"],
        });
        let token = sign(&enc, &kid, claim);
        let claims = verify_oidc(&token, &ctx).await.expect("ok");
        assert_eq!(claims.role, Role::Admin);
    }

    #[tokio::test]
    async fn verify_oidc_role_default_viewer_when_claim_missing() {
        let (enc, dec, kid) = rsa_keypair();
        let ctx = OidcContext::new(cfg());
        install(&ctx, &kid, dec).await;
        let claim = serde_json::json!({
            "sub": "carol",
            "iss": "https://idp.test",
            "aud": "veda",
            "exp": now() + 60,
        });
        let token = sign(&enc, &kid, claim);
        let claims = verify_oidc(&token, &ctx).await.expect("ok");
        assert_eq!(claims.role, Role::Viewer);
    }

    #[tokio::test]
    async fn verify_oidc_rejects_bad_audience() {
        let (enc, dec, kid) = rsa_keypair();
        let ctx = OidcContext::new(cfg());
        install(&ctx, &kid, dec).await;
        let claim = serde_json::json!({
            "sub": "x",
            "iss": "https://idp.test",
            "aud": "WRONG",
            "exp": now() + 60,
            "roles": ["admin"],
        });
        let token = sign(&enc, &kid, claim);
        assert_eq!(
            verify_oidc(&token, &ctx).await.unwrap_err(),
            StatusCode::UNAUTHORIZED
        );
    }

    #[tokio::test]
    async fn verify_oidc_rejects_bad_issuer() {
        let (enc, dec, kid) = rsa_keypair();
        let ctx = OidcContext::new(cfg());
        install(&ctx, &kid, dec).await;
        let claim = serde_json::json!({
            "sub": "x",
            "iss": "https://EVIL",
            "aud": "veda",
            "exp": now() + 60,
            "roles": ["admin"],
        });
        let token = sign(&enc, &kid, claim);
        assert_eq!(
            verify_oidc(&token, &ctx).await.unwrap_err(),
            StatusCode::UNAUTHORIZED
        );
    }

    #[tokio::test]
    async fn verify_oidc_rejects_expired() {
        let (enc, dec, kid) = rsa_keypair();
        let ctx = OidcContext::new(cfg());
        install(&ctx, &kid, dec).await;
        let claim = serde_json::json!({
            "sub": "x",
            "iss": "https://idp.test",
            "aud": "veda",
            "exp": now() - 600,
            "roles": ["admin"],
        });
        let token = sign(&enc, &kid, claim);
        assert_eq!(
            verify_oidc(&token, &ctx).await.unwrap_err(),
            StatusCode::UNAUTHORIZED
        );
    }

    #[tokio::test]
    async fn verify_oidc_rejects_unknown_kid() {
        let (enc, _dec, _kid) = rsa_keypair();
        let ctx = OidcContext::new(OidcConfig {
            jwks_url: "http://127.0.0.1:1".to_owned(),
            ..cfg()
        });
        let claim = serde_json::json!({
            "sub": "x",
            "iss": "https://idp.test",
            "aud": "veda",
            "exp": now() + 60,
        });
        let token = sign(&enc, "no-such-kid", claim);
        assert_eq!(
            verify_oidc(&token, &ctx).await.unwrap_err(),
            StatusCode::UNAUTHORIZED
        );
    }
}

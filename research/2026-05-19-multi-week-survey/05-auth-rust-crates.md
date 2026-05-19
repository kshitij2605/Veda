# OAuth2 / OIDC / JWT Rust Crates for Veda

**Status:** Research — 2026-05-19
**Scope:** Production-ready OSS Rust crates that let Veda add OIDC / OAuth2 / JWT / mTLS / WebAuthn / policy-based authorization without writing the underlying RFCs ourselves.
**Bottom line:** Veda's "Phase ∞" auth roadmap is over-sized. Everything we need is one well-maintained crate away. The only thing we need to *build* is the glue: a Tower/Axum middleware, a small `Principal` extractor, and a config surface.

---

## 1. Executive Summary — Recommended Stack

| Layer | Crate | What it does | Build effort |
|---|---|---|---|
| **JWT decode + signature verify** | `jsonwebtoken` (Keats) | Decode + verify RS256/ES256/EdDSA/HS256 tokens | ~0 LOC — used as-is |
| **JWKS fetch + cache + rotate** | `async-oidc-jwt-validator` (or roll our own ~150 LOC over `jsonwebtoken`) | OIDC discovery, JWKS caching, claims validation | small wrapper |
| **OIDC client (auth code flow, only if Veda ever fronts a UI directly)** | `openidconnect` (ramosbugs) | Full RP for OIDC — discovery, code flow, ID token validation | medium — only needed if we *issue* sessions |
| **OAuth2 client (M2M, optional)** | `oauth2` (ramosbugs) | RFC 6749 / 7636 / 8628 — PKCE, client credentials, device code | only if Veda *consumes* upstream OAuth-protected APIs |
| **Authorization / RBAC / ABAC** | `cedar-policy` (Amazon) — *or stay with the existing `rbac` module if simple* | Policy DSL, sub-millisecond evaluation | medium — schema design dominates |
| **mTLS (service-to-service trust)** | `rustls` + `axum-server` | Mutual TLS via `RustlsConfig::from_pem_file` + `ClientCertVerifier` | small |
| **WebAuthn / passkeys (only if Niriksha grows a user model)** | `webauthn-rs` (Kanidm) | FIDO2 RP — audited by SUSE | medium |
| **OAuth2 server (only if Veda issues its own tokens — strongly discouraged)** | `oxide-auth` | Pluggable OAuth2 AS | large — almost certainly out of scope |
| **SAML (only if a customer forces it)** | `samael` | SAML2 — early-stage, "work in progress" | high — avoid |

**Recommendation:** adopt the **first two rows** now. Treat OIDC client / Cedar / mTLS / WebAuthn as opt-in, each gated by a real customer request. Skip OAuth2 server and SAML entirely.

---

## 2. Layered Recommendation — Phase It

### Phase A (1–3 days of work): JWT validation against an external IdP
Customers run their own Keycloak / Auth0 / Okta / Entra. Veda accepts a `Bearer <JWT>`, validates it against the IdP's JWKS endpoint, and extracts a `Principal` (subject + scopes/roles). **No new identity store, no password hashing, no token issuance.**

- Add `jsonwebtoken = "10"` and `reqwest` (already in tree) to `veda-gateway`.
- Either embed `async-oidc-jwt-validator` *or* write a ~150-LOC `JwksCache` that does: GET `/.well-known/openid-configuration` once, GET the JWKS once, store `Arc<RwLock<HashMap<kid, DecodingKey>>>`, refresh on `kid` miss + a 1-hour TTL.
- Wire it into the existing `auth::auth_middleware` as a parallel path. Selection between "static bearer" and "JWT" is by config: `auth.mode = "static" | "jwt"`.

### Phase B (1 week, if requested): scopes / claims → RBAC
Map JWT `scopes` claim (or `roles`, or a custom claim like `veda.collection.read`) onto Veda's existing `rbac` module. The mapping table lives in YAML config. No code changes per tenant — just policy data.

If the existing `rbac` module is more than ~500 LOC and growing toward ABAC, replace it with **Cedar** (`cedar-policy` crate, Apache-2.0, AWS-backed, sub-millisecond). Cedar's policy language separates authorization from app code, which is exactly the right shape for "different customers want different policies on the same collections."

### Phase C (optional, customer-driven): OIDC client (login flow inside Veda)
If we ever decide Niriksha should host its own login (rather than be embedded behind a customer's reverse proxy), add `openidconnect = "4"`. This gives us the auth code + PKCE flow against any OIDC IdP. Store the resulting session via `axum-login` + `tower-sessions`.

### Phase D (optional): mTLS for service-to-service
For shard-to-shard / coordinator-to-gateway traffic inside a cluster, `rustls` + `axum-server::tls_rustls::RustlsConfig` plus a custom `ClientCertVerifier` against a tenant CA gives us cryptographic peer identity without involving an IdP. Pairs well with SPIFFE if customers already have one.

### Phase E (probably never): OAuth2 server, SAML, WebAuthn
Defer until a paying customer asks. None of these are needed to compete with Milvus / Qdrant / Weaviate today.

---

## 3. Crate-by-Crate Analysis

### `jsonwebtoken` — Keats/jsonwebtoken
- **Version:** 10.4.0 (active, regular releases, last commit recent)
- **License:** MIT
- **What it does:** Encode + decode JWT, verify signature, validate `exp` / `nbf` / `iss` / `aud` / `sub`. Algorithms: HS256/384/512, RS256/384/512, PS256/384/512, ES256/384, EdDSA.
- **JWKS:** Includes a `jwk` module with `Jwk` / `JwkSet` types (public keys only — verify-side, not issue-side). You still write the HTTP fetch + cache yourself, but the parsing is done.
- **Axum integration cost:** Trivial. Decode in a `tower::Layer` or a `FromRequestParts` extractor. ~30 LOC of glue.
- **Production usage:** Single most-downloaded JWT crate in the Rust ecosystem (millions/month). Used by countless production services. The de-facto default.
- **Verdict:** Adopt. No reason to look elsewhere.

### `oauth2` — ramosbugs/oauth2-rs
- **Version:** 5.x (5.0.0 released Jan 2025, dual MIT/Apache-2.0)
- **MSRV:** Rust 1.71
- **What it does:** OAuth2 *client* — RFC 6749 (auth code, client credentials, device code), RFC 7636 (PKCE), RFC 8414 (server metadata), RFC 8628 (device flow), token introspection, revocation. Pluggable HTTP backend (reqwest / curl / async-http-client).
- **What it does NOT do:** It is not an authorization server; for that, see `oxide-auth`.
- **Axum integration cost:** Self-contained client library. Used in an Axum handler the same way you would use `reqwest` — no Tower wiring needed. Axum's own examples include an OAuth one built on this crate.
- **Production usage:** The Rust OAuth2 client. Used by `openidconnect-rs`, `axum-login` backends, GitHub-CLI-style tools, and many SaaS Rust services.
- **Verdict:** Adopt *only* if Veda needs to consume upstream OAuth-protected services (e.g., calling Google Drive to ingest documents on behalf of a user). Not needed for "validate inbound tokens."

### `openidconnect` — ramosbugs/openidconnect-rs
- **Version:** 4.0.1 (released Jul 2025), built on `oauth2` 5.x
- **License:** MIT
- **What it does:** Full OIDC Relying Party. Discovery (`/.well-known/openid-configuration`), ID token validation (RS*/HS*/PS*/ES*/EdDSA), JWKS fetch, dynamic client registration, userinfo endpoint, claim verification, nonce checking. Does *not* support JWE, aggregated/distributed claims, or P-521 ECDSA.
- **Axum integration cost:** Medium. You hold a `CoreClient` in app state, redirect to `client.authorize_url(...)` for login, handle the callback, exchange code for tokens, validate ID token, materialise a session. Roughly 100–200 LOC for a complete login flow.
- **Production usage:** Standard Rust OIDC RP. 616 GitHub stars but high download volume; used by Kanidm, internal SaaS, and as the backbone of several Axum / Actix OIDC middleware crates.
- **Verdict:** Adopt only when Veda needs to *do the OIDC dance itself* (Phase C). For Phase A — receiving JWTs that were already issued by an IdP — `jsonwebtoken` + a tiny JWKS cache is enough and pulls 5× less dependency surface.

### `axum-login` — maxcountryman/axum-login
- **Version:** 0.18.0 (released Jul 2025)
- **License:** MIT
- **What it does:** Session-based auth middleware on top of `tower-sessions`. Defines `AuthnBackend` and `AuthzBackend` traits — you implement them however you want (DB, LDAP, OIDC). Provides `login_required!` / `permission_required!` macros for route protection.
- **Axum integration cost:** Low for the framework itself. The cost is *implementing your backend*. For Veda, the backend would be: "ID came from a JWT we validated; user_id is the `sub` claim; permissions are the `scope` claim."
- **Production usage:** Most popular session-auth library in the Axum ecosystem.
- **Verdict:** Use *if and only if* Niriksha hosts its own login UI (Phase C). For headless API access with JWT bearer tokens, this is over-kill — we don't need sessions at all.

### `biscuit` — lawliet89/biscuit
- **What it does:** Full JOSE (JWT + JWS + JWE) library. Supports encrypted JWTs (JWE).
- **Verdict:** Skip. Compared to `jsonwebtoken`: smaller community, fewer downloads, README explicitly admits "does not support all RFC features and probably never will." Only relevant if a customer mandates JWE (encrypted tokens), which is rare outside healthcare/finance.

### `aliri_oauth2` + `aliri_axum` — neoeinstein/aliri
- **Version:** aliri_axum 0.5.0 (Feb 2025), aliri_oauth2 maintained
- **License:** MIT/Apache-2.0
- **What it does:** Web API access control. JWT validation, OAuth2 scope enforcement, local + remote JWKS authority. The `aliri_axum` crate provides scope-guarded extractors and route layers.
- **Axum integration cost:** Low — drop-in extractors like `Scoped<("collections:write",)>`.
- **Production usage:** 128 stars, 274 commits, 18 releases. Smaller than ramosbugs's ecosystem but well-engineered. Good for cases where you want scope enforcement *baked into the type system* rather than checked imperatively.
- **Verdict:** Strong alternative to "write our own JWKS cache + scope check." If we don't adopt Cedar, `aliri_axum` is a more ergonomic landing spot than hand-rolled middleware. Worth a spike.

### `oxide-auth` — HeroicKatora/oxide-auth
- **What it does:** OAuth2 *authorization server* framework. Pluggable Registrar / Authorizer / Solicitor / Endpoint. Has actix/rocket/iron/rouille frontends — no first-class axum adapter today.
- **Verdict:** Skip. Veda has no reason to issue its own OAuth2 tokens — that's the IdP's job. The 80/20 win is "delegate to the customer's existing Okta/Auth0/Keycloak/Entra." Running our own AS doubles operational surface for no user value.

### `webauthn-rs` — kanidm/webauthn-rs
- **Version:** 0.5.2 (released Jul 2025)
- **License:** MPL-2.0
- **What it does:** WebAuthn / FIDO2 / passkeys. Relying Party only. Audited by SUSE Product Security.
- **Axum integration cost:** Medium — needs session storage to hold challenge state across register/login round-trips. Pairs naturally with `axum-login` + `tower-sessions`.
- **Production usage:** Powers Kanidm, the largest Rust IDP. 659 stars.
- **Verdict:** Best-in-class. Adopt *only* if Niriksha grows into a multi-user product with its own user accounts. For API-only / IdP-delegated auth, irrelevant.

### `cedar-policy` — cedar-policy/cedar
- **Version:** 4.11.0 (May 2026 — actively developed)
- **License:** Apache-2.0
- **What it does:** Policy-as-code language + evaluator. Sub-millisecond authorization decisions. RBAC + ABAC in one language. Differential-tested against a formal model — fewer footguns than hand-rolled access checks.
- **Axum integration cost:** Low for the engine itself (one call per request); medium for schema design (entities, actions, resources).
- **Production usage:** Powers AWS Verified Permissions. Adopted by MongoDB, Strongdm, and various SaaS authorization layers.
- **Verdict:** Strongly consider when Veda's `rbac` module grows beyond simple role checks. Policies like *"users in tenant A can read collections tagged 'public', but only owners can delete"* become declarative one-liners instead of bespoke Rust.

### `casbin-rs` — apache/casbin-rs
- **Version:** 2.8.0, requires tokio 1.x
- **License:** Apache-2.0
- **What it does:** RBAC + ABAC + ACL via the PERM model. CONF + CSV policy files.
- **Verdict:** Mature, multi-language (Casbin exists in Go, Java, Python…). If Veda's customers already run Casbin, adopting `casbin-rs` saves migration cost. Otherwise Cedar is a stronger pick: better tooling, formal model, AWS-backed momentum, modern policy syntax.

### `samael` — njaremko/samael
- **Version:** 0.0.20 (Mar 2026), still `0.0.x`, README explicitly "work in progress."
- **What it does:** SAML2.
- **Verdict:** Avoid unless a deal-breaker. SAML is a Java/Microsoft-shop reality, and the only credible path in Rust today is `samael`, which is not production-grade. If we must speak SAML, terminate it at an upstream proxy (Pomerium, oauth2-proxy, mod_auth_mellon) and let Veda see clean JWTs downstream.

### `rustls` + `axum-server` — for mTLS
- **What it does:** `axum-server::tls_rustls::RustlsConfig` accepts PEM cert + key. For mTLS, configure `ClientCertVerifier` on the underlying `rustls::ServerConfig` to require + validate client certs against a CA. Peer cert info available via `ConnectInfo` (an open Axum issue, but workable with hyper directly).
- **Axum integration cost:** Low — ~50 LOC including CA cert loading. The hardest part is cert lifecycle (rotation, revocation) which is operational, not code.
- **Verdict:** Adopt for service-to-service inside a cluster (coordinator <-> shards). Optional for north-south traffic.

---

## 4. Concrete Migration Path: Static Bearer -> IdP-Issued JWT

This is the work Veda actually needs.

### What we have today
- `Cargo.toml`: gateway depends on axum + tower.
- `crates/veda-gateway/src/lib.rs`: declares `pub mod auth;` and `pub mod rbac;`.
- `crates/veda-gateway/src/rest/mod.rs:127`: `auth::auth_middleware` is wired as a layer over all protected routes; public routes (`/health`, `/readiness`, `/auth/login`) are merged in afterwards.
- Auth secret presumably read from `VEDA_TOKEN` / `VEDA_AUTH` env var; if header matches, request continues.
- `rbac` module exists; permission model is in-tree.

### What we want
- The same middleware can run in **one of three modes** chosen by config:
  1. `static`: existing bearer-token behaviour (default; backward compatible).
  2. `jwt`: validate `Authorization: Bearer <JWT>` against an external IdP's JWKS, then map claims to `Principal`.
  3. `disabled`: dev mode only.

### Code-level diff (sketch — fits in a single PR)

**Add to `veda-gateway/Cargo.toml`:**
```toml
jsonwebtoken = "10"
# Either of:
async-oidc-jwt-validator = "0.x"   # ~zero glue
# ...or use jsonwebtoken directly and write a small JwksCache.
```

**New file `crates/veda-gateway/src/auth/jwt.rs` (~150 LOC):**
- `JwksCache { issuer: Url, http: reqwest::Client, keys: Arc<RwLock<HashMap<String, DecodingKey>>>, expires_at: Arc<AtomicU64> }`.
- `async fn ensure_fresh(&self)` — if expired or kid miss, GET `{issuer}/.well-known/openid-configuration`, follow `jwks_uri`, parse with `jsonwebtoken::jwk::JwkSet`, repopulate.
- `async fn validate(&self, token: &str) -> Result<Principal>` — `decode_header` -> look up kid -> `decode::<Claims>` with `Validation::new(alg).set_issuer(&[iss]).set_audience(&[aud])` -> map claims to `Principal { subject, scopes, roles, tenant }`.

**Modify `crates/veda-gateway/src/auth/mod.rs` (existing file):**
- Replace single-mode check with an `enum AuthMode { Static(String), Jwt(JwksCache), Disabled }` stored in `AppState`.
- `auth_middleware`: dispatch on mode, attach `Principal` to request extensions in every success path, return 401 with `WWW-Authenticate: Bearer realm="veda"` on every failure.

**Modify `Principal` (new or extended):**
```rust
struct Principal { subject: String, scopes: Vec<String>, tenant: Option<String>, raw_claims: serde_json::Value }
```
- The `rbac` module already exists — feed `Principal.scopes` into its `check(permission)` API. No changes needed to call sites.

**Config (extend whatever exists today):**
```yaml
auth:
  mode: jwt                          # static | jwt | disabled
  jwt:
    issuer: https://login.example.com/realms/veda
    audience: veda-api
    required_scopes: []              # optional belt-and-suspenders
    jwks_ttl_seconds: 3600
```

**That is the entire migration.** No new identity store. No password hashing. No token rotation logic. No SSO redirects. The IdP does all of that.

### Time estimate
- 2 days end-to-end for a senior Rust engineer, including tests against a local Keycloak in docker-compose.
- ~300 net new LOC, ~50 modified LOC.

---

## 5. Honest "What We Don't Need to Build"

Existing OSS does basically all of it. We do not need to write:

| Thing | Provided by |
|---|---|
| JWT parsing, signature verification, exp/nbf/iss/aud validation | `jsonwebtoken` |
| JWKS fetch + caching + rotation | `async-oidc-jwt-validator` *or* ~150 LOC over `jsonwebtoken` |
| OIDC discovery (`/.well-known/openid-configuration`) | same |
| OIDC auth code + PKCE flow | `openidconnect` |
| OAuth2 client credentials / device code | `oauth2` |
| OAuth2 authorization server | `oxide-auth` — don't run this, delegate to IdP |
| Sessions, cookies, login pages | `axum-login` + `tower-sessions` |
| Passkeys / FIDO2 | `webauthn-rs` |
| Policy DSL (RBAC + ABAC) | `cedar-policy` |
| mTLS | `rustls` + `axum-server` |
| SAML | Terminate at an upstream proxy. Don't write Rust SAML. |

**What we DO need to build (~300 LOC total):**
1. The `auth::AuthMode` enum and dispatch in middleware.
2. A `Principal` extractor (`FromRequestParts`) so handlers can ask for `Principal` instead of digging through extensions.
3. A claim-to-permission mapping in config (or, later, a Cedar schema).
4. Documentation: how to point Veda at Keycloak, at Auth0, at Okta, at Entra, at Google Workspace. These are config snippets, not code.

**What we should NOT build:**
- Our own user database. Customers already have one.
- Our own password hashing. Customers already do this.
- Our own MFA. Customers already do this.
- Our own OAuth2 server. Customers will not register apps in *us*.
- Our own SAML stack. Use an L7 proxy.

---

## 6. IdP Integration Notes

All of the major IdPs are interchangeable from Veda's point of view once we have JWT-via-JWKS. The only per-IdP knobs are:
- **Issuer URL** for OIDC discovery.
- **Audience** the IdP stamps into tokens.
- **Claim names** for scopes/roles/tenant (each IdP uses slightly different conventions).

| IdP | Issuer pattern | Notes |
|---|---|---|
| **Keycloak** | `https://<host>/realms/<realm>` | Self-hosted; the default test target. Uses `realm_access.roles` for role claims. |
| **Auth0** | `https://<tenant>.auth0.com/` | Custom rules can inject any claims (Veda config needs to name them). `permissions` claim if RBAC enabled. |
| **Okta** | `https://<tenant>.okta.com/oauth2/default` | Uses `scp` claim for scopes by default. |
| **Microsoft Entra (Azure AD)** | `https://login.microsoftonline.com/<tenant-id>/v2.0` | App roles surface as `roles` claim. Audience is the application ID URI. |
| **Google Workspace** | `https://accounts.google.com` | OIDC for human users; for M2M use Google service accounts which mint signed JWTs Veda can verify the same way. |

Treat them as **one IdP integration with three config fields**, not five separate integrations. This is the whole point of OIDC.

---

## 7. Final Recommendation

**Adopt now (Phase A):**
- `jsonwebtoken` for decode + verify.
- Either `async-oidc-jwt-validator` *or* ~150 LOC of in-tree JWKS cache.
- Extend the existing `auth` module with an `AuthMode` enum.

**Adopt when needed (Phase B–D):**
- `cedar-policy` when `rbac` outgrows simple role checks.
- `openidconnect` if Niriksha hosts its own login.
- `rustls` mTLS for intra-cluster traffic.

**Skip:**
- `oxide-auth` (don't be an AS).
- `samael` (not production-grade).
- `biscuit` (no JWE need).
- `casbin-rs` (Cedar is a stronger pick for new code).

The "Phase ∞ — multi-month" framing in the original plan is wrong. The work to add IdP-delegated JWT auth to Veda is roughly **one PR, ~300 LOC, two engineer-days, zero new identity infrastructure**. Everything beyond that is opt-in based on real customer requests, not a default roadmap commitment.

---

## Sources

- [ramosbugs/oauth2-rs (GitHub)](https://github.com/ramosbugs/oauth2-rs)
- [ramosbugs/openidconnect-rs (GitHub)](https://github.com/ramosbugs/openidconnect-rs)
- [Keats/jsonwebtoken (GitHub)](https://github.com/Keats/jsonwebtoken) / [docs.rs/jsonwebtoken](https://docs.rs/jsonwebtoken/)
- [maxcountryman/axum-login (GitHub)](https://github.com/maxcountryman/axum-login)
- [lawliet89/biscuit (GitHub)](https://github.com/lawliet89/biscuit)
- [neoeinstein/aliri (GitHub)](https://github.com/neoeinstein/aliri)
- [HeroicKatora/oxide-auth (GitHub)](https://github.com/HeroicKatora/oxide-auth)
- [kanidm/webauthn-rs (GitHub)](https://github.com/kanidm/webauthn-rs)
- [cedar-policy/cedar (GitHub)](https://github.com/cedar-policy/cedar)
- [apache/casbin-rs (GitHub)](https://github.com/apache/casbin-rs)
- [njaremko/samael (GitHub)](https://github.com/njaremko/samael)
- [async-oidc-jwt-validator (crates.io)](https://crates.io/crates/async-oidc-jwt-validator) / [oidc-jwt-validator (crates.io)](https://crates.io/crates/oidc-jwt-validator)
- [axum-server::tls_rustls (docs.rs)](https://docs.rs/axum-server/latest/axum_server/tls_rustls/index.html)
- [Axum OAuth example (tokio-rs/axum)](https://github.com/tokio-rs/axum/blob/main/examples/oauth/src/main.rs)
- [oauth-axum (crates.io)](https://crates.io/crates/oauth-axum)
- [Production-Ready JWT Validation in Axum (pipinghot.dev)](https://pipinghot.dev/production-ready-jwt-validation-in-axum-a-real-implementation/)
- [Cedar Policy Language overview (StrongDM)](https://www.strongdm.com/cedar-policy-language)
- [Auth0 Rust+JWT guide](https://auth0.com/blog/build-an-api-in-rust-with-jwt-authentication-using-actix-web/)
- [Implement OAuth2/OIDC with Axum (mojoauth.com)](https://mojoauth.com/oauth2-oidc/implement-oauth2-oidc-with-axum/)
- [Milvus auth quick reference](https://milvus.io/ai-quick-reference/how-do-i-implement-authentication-and-authorization-for-vector-databases)
- [API Authentication in 2026 (Elysiate)](https://www.elysiate.com/blog/api-authentication-oauth-jwt-api-keys-compared)
- [API key vs JWT (Scalekit)](https://www.scalekit.com/blog/apikey-jwt-comparison)

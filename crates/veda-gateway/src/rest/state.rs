use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use tokio::sync::RwLock;
use veda_core::embedding::{EmbeddingModelConfig, EmbeddingModelRegistry, Modality};
use veda_core::llm_client::LlmConfig;
use veda_core::splade_client::{SpladeClient, SpladeClientConfig};
use veda_core::schema::CollectionSchema;
use veda_query::distributed::DistributedQueryExecutor;
use metrics_exporter_prometheus::PrometheusHandle;
use veda_storage::blob_store::{BlobStore, LocalBlobStore, ObjectBlobStore, ObjectStoreConfig};
use veda_storage::segment_manager::SegmentManager;

use crate::auth::Role;
use crate::auth_oidc::AuthMode;
use crate::rest::saved_objects::SavedObjectStore;
use veda_coordinator::multi_group::MultiGroupCoordinator;
use veda_coordinator::raft_coordinator::RaftCoordinator;
use veda_log::ShardLog;
use veda_time::{LocalTso, Tso};

/// Represents a user in the in-memory user store.
#[derive(Clone)]
pub struct UserRecord {
    pub password: String,
    pub role: Role,
}

/// Shared application state accessible from all handlers.
#[derive(Clone)]
pub struct AppState {
    pub(crate) inner: Arc<AppStateInner>,
}

pub struct AppStateInner {
    /// Collection schemas keyed by collection name.
    pub schemas: RwLock<HashMap<String, CollectionSchema>>,
    /// Segment managers per collection, keyed by (collection_name, shard_id).
    pub segment_managers: RwLock<HashMap<String, HashMap<u32, SegmentManager>>>,
    /// Query executors per collection.
    pub query_executors: RwLock<HashMap<String, DistributedQueryExecutor>>,
    /// Embedding model registry.
    pub embedding_registry: RwLock<EmbeddingModelRegistry>,
    /// Optional SPLADE sparse-encoder client, configured from
    /// `VEDA_SPLADE_*` env vars at startup. None when no endpoint is
    /// set — sparse_vector fields with `auto_embed` then fall back to
    /// "leave empty" with a one-shot warn.
    pub splade_client: RwLock<Option<SpladeClient>>,
    /// Optional LLM endpoint config (OpenAI-compatible chat completions),
    /// from `VEDA_LLM_*` env vars at startup. When unset the
    /// `POST /v1/llm/complete` route returns 503 and LLM-backed UI
    /// affordances (cluster labels, NL→DSL, failure classifier) hide.
    pub llm_config: RwLock<Option<LlmConfig>>,
    /// Base data directory.
    pub data_dir: PathBuf,
    /// FlashParse OCR endpoint (None = document processing disabled).
    pub flashparse_endpoint: Option<String>,
    /// Whether JWT authentication is enabled. Legacy `Static`-mode
    /// flag; the active auth dispatcher reads `auth_mode` below.
    /// Kept so existing tests/callers that set this directly via
    /// `with_auth` continue to behave.
    pub auth_enabled: bool,
    /// Secret used for JWT signing and verification (Static mode).
    pub jwt_secret: String,
    /// Active authentication mode. Set from `VEDA_AUTH_MODE` via
    /// [`AppState::load_auth_from_env`]; defaults match `auth_enabled`
    /// (true → Static, false → Disabled) so constructors stay
    /// backwards compatible.
    pub auth_mode: RwLock<AuthMode>,
    /// In-memory user store: username -> UserRecord.
    pub users: HashMap<String, UserRecord>,
    /// Blob storage backend.
    pub blob_store: RwLock<Arc<dyn BlobStore>>,
    /// Prometheus metrics handle (None when recorder is not installed).
    pub prometheus_handle: Option<PrometheusHandle>,
    /// SavedObject store, persisted to `{data_dir}/_saved_objects.json`.
    pub saved_objects: SavedObjectStore,
    /// Process start time — used by `/v1/admin/nodes` to report uptime.
    pub started_at: Instant,
    /// When `VEDA_COORDINATOR=raft` is set, this holds a running
    /// `RaftCoordinator` (single-node bootstrapped). Otherwise `None`
    /// and the gateway runs without consensus. The admin nodes view
    /// reads this to flip `mode` between `"single"` and `"raft"`.
    pub raft: RwLock<Option<RaftCoordinator>>,
    /// Per-collection set of record IDs already materialised into
    /// the local SegmentManager from a Raft-replicated apply. The
    /// reconcile task uses this to skip records it's already seen.
    pub raft_materialized_records: RwLock<HashMap<String, std::collections::HashSet<String>>>,
    /// Map from Raft NodeId to a `host:port` REST address that
    /// peer accepts client traffic on. Used by the follower-to-
    /// leader 307 redirect path so mutating REST requests sent
    /// to a follower come back with `Location:` pointing at the
    /// current leader. Populated from
    /// `VEDA_RAFT_REST_PEERS="1@http://veda-1:6340,2@..."` at
    /// startup. Optional: if unset, the redirect is a no-op and
    /// the request proceeds (openraft forwards internally).
    pub raft_rest_peers: RwLock<HashMap<u64, String>>,
    /// MultiGroupCoordinator hosting per-(collection, shard_id)
    /// Raft groups. Auto-spawned by the reconcile loop when a
    /// CreateCollection commits. Groups use the in-process
    /// transport (start_single_group); cross-process replication
    /// for per-shard groups needs a multiplexing Tonic service
    /// that is its own architectural project, deferred from this
    /// session. Today this exists to exercise the routing code
    /// path and to surface where horizontal scaling will plug in.
    pub multi_group: RwLock<Option<MultiGroupCoordinator>>,
    /// Timestamp oracle. Default is `LocalTso` — Phase A
    /// (`.claude/plans/structured-mixing-turing.md`). The
    /// `MetaLeasedTso` impl that consults `_meta`'s lease ships
    /// in a Phase A follow-up.
    pub tso: Arc<dyn Tso>,
    /// Per-(collection, shard_id) append-only logs. Phase B
    /// scaffolding for the FoundationDB-shape data plane. Today
    /// these are dark-write targets: `records::insert` appends
    /// here too when `VEDA_LOG_DARK_WRITE=1`, but the existing
    /// Raft path remains the source of truth. Phase C will cut
    /// the Raft path and make these primary.
    pub shard_logs: RwLock<HashMap<String, Arc<ShardLog>>>,
}

impl AppState {
    /// Create a new AppState with auth **disabled** (default for tests).
    pub fn new(data_dir: &Path) -> Self {
        Self {
            inner: Arc::new(AppStateInner {
                schemas: RwLock::new(HashMap::new()),
                segment_managers: RwLock::new(HashMap::new()),
                query_executors: RwLock::new(HashMap::new()),
                embedding_registry: RwLock::new(EmbeddingModelRegistry::new()),
                splade_client: RwLock::new(None),
                llm_config: RwLock::new(None),
                data_dir: data_dir.to_owned(),
                flashparse_endpoint: std::env::var("VEDA_FLASHPARSE_ENDPOINT").ok().or_else(|| Some("http://flashparse:8000".to_owned())),
                auth_enabled: false,
                jwt_secret: "default-test-secret".to_owned(),
                auth_mode: RwLock::new(AuthMode::Disabled),
                users: HashMap::new(),
                blob_store: RwLock::new(Arc::new(
                    LocalBlobStore::new(&data_dir.join("blobs")).expect("failed to create local blob store")
                )),
                prometheus_handle: None,
                saved_objects: SavedObjectStore::new(data_dir),
                started_at: Instant::now(),
                raft: RwLock::new(None),
                raft_materialized_records: RwLock::new(HashMap::new()),
                raft_rest_peers: RwLock::new(HashMap::new()),
                multi_group: RwLock::new(None),
                tso: Arc::new(LocalTso::new()),
                shard_logs: RwLock::new(HashMap::new()),
            }),
        }
    }

    /// Create a new AppState with auth **enabled** and a default admin user.
    pub fn with_auth(data_dir: &Path, jwt_secret: &str) -> Self {
        let mut users = HashMap::new();
        users.insert(
            "admin".to_owned(),
            UserRecord {
                password: "admin".to_owned(),
                role: Role::Admin,
            },
        );
        Self {
            inner: Arc::new(AppStateInner {
                schemas: RwLock::new(HashMap::new()),
                segment_managers: RwLock::new(HashMap::new()),
                query_executors: RwLock::new(HashMap::new()),
                embedding_registry: RwLock::new(EmbeddingModelRegistry::new()),
                splade_client: RwLock::new(None),
                llm_config: RwLock::new(None),
                data_dir: data_dir.to_owned(),
                flashparse_endpoint: std::env::var("VEDA_FLASHPARSE_ENDPOINT").ok().or_else(|| Some("http://flashparse:8000".to_owned())),
                auth_enabled: true,
                jwt_secret: jwt_secret.to_owned(),
                auth_mode: RwLock::new(AuthMode::Static),
                users,
                blob_store: RwLock::new(Arc::new(
                    LocalBlobStore::new(&data_dir.join("blobs")).expect("failed to create local blob store")
                )),
                prometheus_handle: None,
                saved_objects: SavedObjectStore::new(data_dir),
                started_at: Instant::now(),
                raft: RwLock::new(None),
                raft_materialized_records: RwLock::new(HashMap::new()),
                raft_rest_peers: RwLock::new(HashMap::new()),
                multi_group: RwLock::new(None),
                tso: Arc::new(LocalTso::new()),
                shard_logs: RwLock::new(HashMap::new()),
            }),
        }
    }

    /// Create an AppState with auth enabled and custom users.
    pub fn with_auth_and_users(
        data_dir: &Path,
        jwt_secret: &str,
        users: HashMap<String, UserRecord>,
    ) -> Self {
        Self {
            inner: Arc::new(AppStateInner {
                schemas: RwLock::new(HashMap::new()),
                segment_managers: RwLock::new(HashMap::new()),
                query_executors: RwLock::new(HashMap::new()),
                embedding_registry: RwLock::new(EmbeddingModelRegistry::new()),
                splade_client: RwLock::new(None),
                llm_config: RwLock::new(None),
                data_dir: data_dir.to_owned(),
                flashparse_endpoint: std::env::var("VEDA_FLASHPARSE_ENDPOINT").ok().or_else(|| Some("http://flashparse:8000".to_owned())),
                auth_enabled: true,
                jwt_secret: jwt_secret.to_owned(),
                auth_mode: RwLock::new(AuthMode::Static),
                users,
                blob_store: RwLock::new(Arc::new(
                    LocalBlobStore::new(&data_dir.join("blobs")).expect("failed to create local blob store")
                )),
                prometheus_handle: None,
                saved_objects: SavedObjectStore::new(data_dir),
                started_at: Instant::now(),
                raft: RwLock::new(None),
                raft_materialized_records: RwLock::new(HashMap::new()),
                raft_rest_peers: RwLock::new(HashMap::new()),
                multi_group: RwLock::new(None),
                tso: Arc::new(LocalTso::new()),
                shard_logs: RwLock::new(HashMap::new()),
            }),
        }
    }

    /// Create a new AppState with a Prometheus handle (auth disabled, for backwards compat).
    pub fn with_prometheus(data_dir: &Path, prometheus_handle: PrometheusHandle) -> Self {
        let mut state = Self::new(data_dir);
        Arc::get_mut(&mut state.inner)
            .expect("no other references to AppStateInner")
            .prometheus_handle = Some(prometheus_handle);
        state
    }

    /// Create a new AppState with auth enabled and a Prometheus handle.
    pub fn with_auth_and_prometheus(
        data_dir: &Path,
        jwt_secret: &str,
        prometheus_handle: PrometheusHandle,
    ) -> Self {
        let mut state = Self::with_auth(data_dir, jwt_secret);
        Arc::get_mut(&mut state.inner)
            .expect("no other references to AppStateInner")
            .prometheus_handle = Some(prometheus_handle);
        state
    }

    /// Get a reference to the embedding model registry for configuration.
    pub fn embedding_registry(
        &self,
    ) -> &RwLock<EmbeddingModelRegistry> {
        &self.inner.embedding_registry
    }

    /// Load a SPLADE sparse-encoder client from `VEDA_SPLADE_*` env
    /// vars. No-op when the endpoint isn't configured; sparse_vector
    /// fields with `auto_embed.modality = text` will then leave their
    /// values empty at insert time.
    pub async fn load_splade_from_env(&self) {
        let Some(cfg) = SpladeClientConfig::from_env() else {
            return;
        };
        let endpoint = cfg.endpoint.clone();
        let model = cfg.model.clone();
        let mut guard = self.inner.splade_client.write().await;
        *guard = Some(SpladeClient::new(cfg));
        tracing::info!(model = %model, endpoint = %endpoint, "registered SPLADE encoder from environment");
    }

    /// Read `VEDA_AUTH_MODE` + `VEDA_OIDC_*` and install the
    /// matching `AuthMode`. Called once at server startup. If
    /// `VEDA_AUTH_MODE=oidc` is set but a required var is missing
    /// or JWKS discovery fails, logs an error and leaves the
    /// existing mode in place (i.e. fail closed in production
    /// because the server will then reject incoming traffic at the
    /// next middleware hop if Static was unset, or fall back to
    /// Static if it was set).
    pub async fn load_auth_from_env(&self) {
        let mode = std::env::var("VEDA_AUTH_MODE").unwrap_or_default();
        match mode.as_str() {
            "" | "disabled" => {
                // honor the legacy flag for backwards compat: if
                // auth_enabled is true (Static), leave the
                // constructor-set mode alone.
            }
            "static" => {
                let mut guard = self.inner.auth_mode.write().await;
                *guard = crate::auth_oidc::AuthMode::Static;
                tracing::info!("auth mode: static (HS256 shared-secret)");
            }
            "oidc" => match crate::auth_oidc::OidcConfig::from_env().await {
                Ok(Some(cfg)) => {
                    let issuer = cfg.issuer.clone();
                    let audience = cfg.audience.clone();
                    let jwks_url = cfg.jwks_url.clone();
                    let ctx = std::sync::Arc::new(crate::auth_oidc::OidcContext::new(cfg));
                    // Best-effort prime — failure here is logged
                    // but not fatal; the cache will retry on first
                    // request.
                    if let Err(e) = ctx.jwks.refresh().await {
                        tracing::warn!(
                            error = %e,
                            "initial JWKS refresh failed; cache will refresh on first auth"
                        );
                    }
                    let mut guard = self.inner.auth_mode.write().await;
                    *guard = crate::auth_oidc::AuthMode::Oidc(ctx);
                    tracing::info!(
                        issuer = %issuer,
                        audience = %audience,
                        jwks_url = %jwks_url,
                        "auth mode: oidc (IdP-delegated JWT)"
                    );
                }
                Ok(None) => unreachable!(),
                Err(e) => {
                    tracing::error!(
                        error = %e,
                        "VEDA_AUTH_MODE=oidc but config is incomplete; auth mode unchanged"
                    );
                }
            },
            other => {
                tracing::warn!(
                    mode = %other,
                    "unknown VEDA_AUTH_MODE; expected disabled|static|oidc"
                );
            }
        }
    }

    /// Spawn a background task that periodically reconciles the
    /// local Veda state (schemas + segment managers + executors +
    /// per-collection record stores) against the Raft state
    /// machine's view. Each tick:
    ///   - reads `coord.query_state()`
    ///   - materializes any collection present in Raft state but
    ///     missing locally (followers catching up after restart;
    ///     leader handles its own creates synchronously)
    ///   - drops any collection present locally but missing from
    ///     Raft state (covers DropCollection replicas)
    ///   - applies any record in Raft state but not yet
    ///     materialized in the node's SegmentManager (follower
    ///     catch-up of Raft-routed inserts)
    ///
    /// Runs only when raft mode is on. Idempotent — safe to call
    /// multiple times, but we only spawn once per start_server boot.
    pub async fn spawn_raft_reconcile(&self) {
        let guard = self.inner.raft.read().await;
        if guard.is_none() {
            return;
        }
        let state_clone = self.clone();
        let interval = std::env::var("VEDA_RAFT_RECONCILE_INTERVAL_SECS")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(2)
            .max(1);
        tokio::spawn(async move {
            let mut ticker =
                tokio::time::interval(std::time::Duration::from_secs(interval));
            ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
            loop {
                ticker.tick().await;
                if let Err(e) =
                    crate::rest::collections::reconcile_collections_from_raft(&state_clone).await
                {
                    tracing::warn!(error = ?e, "raft reconcile failed");
                }
            }
        });
    }

    /// Spawn a background task that subscribes to the Raft metrics
    /// watch channel and publishes the interesting numbers to the
    /// existing Prometheus recorder. Idempotent — only spawns when
    /// a coordinator is configured. Metrics:
    ///   - veda_raft_term: current Raft term
    ///   - veda_raft_leader_id: id of the current leader (0 if none)
    ///   - veda_raft_last_applied_index: last applied log index
    ///   - veda_raft_last_log_index: last log index (lag = this - last_applied)
    ///   - veda_raft_is_leader: 1 when self is leader, 0 otherwise
    pub async fn spawn_raft_metrics_publisher(&self) {
        let guard = self.inner.raft.read().await;
        let Some(coord) = guard.as_ref() else { return };
        let mut rx = coord.metrics();
        tokio::spawn(async move {
            loop {
                let m = rx.borrow_and_update().clone();
                metrics::gauge!("veda_raft_term").set(m.current_term as f64);
                metrics::gauge!("veda_raft_leader_id").set(m.current_leader.unwrap_or(0) as f64);
                metrics::gauge!("veda_raft_last_applied_index").set(
                    m.last_applied.map(|l| l.index).unwrap_or(0) as f64
                );
                metrics::gauge!("veda_raft_last_log_index").set(
                    m.last_log_index.unwrap_or(0) as f64
                );
                let is_leader = matches!(m.state, veda_coordinator::RaftServerState::Leader);
                metrics::gauge!("veda_raft_is_leader").set(if is_leader { 1.0 } else { 0.0 });
                if rx.changed().await.is_err() {
                    break;
                }
            }
            tracing::info!("raft metrics publisher exiting");
        });
    }

    /// Optionally bootstrap a single-node `RaftCoordinator` from the
    /// env. Activated when `VEDA_COORDINATOR=raft` is set. The node
    /// id is read from `VEDA_NODE_ID` (defaults to 1). No-op
    /// otherwise; the gateway keeps running without consensus.
    ///
    /// This is the Phase 4 + Phase 5 hook from
    /// `documentation/raft-migration.md`. The actual Raft loop and
    /// log/state-machine live inside `veda-coordinator`.
    /// Parse VEDA_RAFT_REST_PEERS into the raft_rest_peers map.
    /// Format: `"1@http://veda-1:6340,2@http://veda-2:6340"`.
    /// Used by the follower-to-leader 307 redirect path so a
    /// mutation sent to a follower replies with Location:
    /// pointing at the current leader's REST listener.
    pub async fn load_raft_rest_peers_from_env(&self) {
        let raw = std::env::var("VEDA_RAFT_REST_PEERS").unwrap_or_default();
        if raw.is_empty() {
            return;
        }
        let mut map = HashMap::new();
        for entry in raw.split(',') {
            let entry = entry.trim();
            if entry.is_empty() {
                continue;
            }
            let Some((id_s, addr)) = entry.split_once('@') else {
                tracing::warn!(entry = %entry, "VEDA_RAFT_REST_PEERS: missing '@'");
                continue;
            };
            let Ok(id) = id_s.trim().parse::<u64>() else {
                tracing::warn!(entry = %entry, "VEDA_RAFT_REST_PEERS: bad node id");
                continue;
            };
            map.insert(id, addr.trim().to_owned());
        }
        if !map.is_empty() {
            tracing::info!(peers = ?map, "raft REST peer addresses registered");
            *self.inner.raft_rest_peers.write().await = map;
        }
    }

    /// If this node is a Raft follower, return the
    /// `host:port` REST address of the current leader. Used by
    /// mutating handlers to issue a 307 redirect so clients
    /// hit the leader directly instead of waiting on openraft's
    /// internal forward-and-back.
    ///
    /// Returns None when:
    ///   - raft is not configured
    ///   - this node IS the leader
    ///   - the leader id isn't in raft_rest_peers (operator
    ///     hasn't populated VEDA_RAFT_REST_PEERS for it)
    pub async fn leader_rest_address(&self) -> Option<String> {
        let guard = self.inner.raft.read().await;
        let coord = guard.as_ref()?;
        let mem = coord.membership_snapshot();
        let self_id: u64 = std::env::var("VEDA_NODE_ID")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1);
        let leader_id = mem.current_leader?;
        if leader_id == self_id {
            return None;
        }
        self.inner
            .raft_rest_peers
            .read()
            .await
            .get(&leader_id)
            .cloned()
    }

    pub async fn load_raft_from_env(&self) {
        let want_raft = std::env::var("VEDA_COORDINATOR")
            .map(|v| v.eq_ignore_ascii_case("raft"))
            .unwrap_or(false);
        if !want_raft {
            return;
        }
        let node_id: u64 = std::env::var("VEDA_NODE_ID")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1);
        let raft_dir = self.inner.data_dir.join("raft");

        // Multi-node path: VEDA_RAFT_PEERS=1@host1:6341,2@host2:6341,...
        // VEDA_RAFT_LISTEN sets the gRPC bind address (default 0.0.0.0:6341).
        let peers_env = std::env::var("VEDA_RAFT_PEERS").unwrap_or_default();
        let peers = veda_coordinator::raft_tonic::parse_peers_env(&peers_env);
        if !peers.is_empty() {
            let listen = std::env::var("VEDA_RAFT_LISTEN")
                .unwrap_or_else(|_| "0.0.0.0:6341".to_owned());
            let listen_addr = match listen.parse::<std::net::SocketAddr>() {
                Ok(a) => a,
                Err(e) => {
                    tracing::error!(error = %e, listen = %listen, "VEDA_RAFT_LISTEN invalid");
                    return;
                }
            };
            // Bootstrap when explicitly requested via VEDA_RAFT_BOOTSTRAP=1.
            // Otherwise the node will recover from disk (if state exists)
            // or wait for a join RPC.
            let bootstrap = std::env::var("VEDA_RAFT_BOOTSTRAP")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false);
            match RaftCoordinator::start_multi_node(node_id, &raft_dir, listen_addr, peers.clone(), bootstrap).await {
                Ok(coord) => {
                    let mut guard = self.inner.raft.write().await;
                    *guard = Some(coord);
                    tracing::info!(
                        node_id,
                        listen = %listen_addr,
                        peers = peers.len(),
                        bootstrap,
                        raft_dir = %raft_dir.display(),
                        "raft coordinator started (multi-node, Tonic)",
                    );
                    return;
                }
                Err(e) => {
                    tracing::error!(error = %e, "failed to start multi-node raft");
                    return;
                }
            }
        }

        // Single-node path (no peers): durable storage, no Tonic
        // server. Useful for development and the simplest production
        // deployment.
        match RaftCoordinator::start_single_node_with_dir(node_id, &raft_dir).await {
            Ok(coord) => {
                let mut guard = self.inner.raft.write().await;
                *guard = Some(coord);
                tracing::info!(
                    node_id,
                    raft_dir = %raft_dir.display(),
                    "raft coordinator started (single-node, durable)",
                );
            }
            Err(e) => {
                tracing::error!(error = %e, "failed to start raft coordinator");
            }
        }

        // Always-on: a MultiGroupCoordinator under
        // {data_dir}/raft-groups/, ready to host per-shard
        // groups the reconcile loop auto-spawns. The cluster-wide
        // Raft handle above remains the source of truth for
        // replicated commands; per-shard groups are populated for
        // the routing code path and for the future multi-group
        // transport landing.
        let groups_dir = self.inner.data_dir.join("raft-groups");
        let mg = MultiGroupCoordinator::new(&groups_dir);
        *self.inner.multi_group.write().await = Some(mg);
    }

    /// Load an OpenAI-compatible LLM endpoint from `VEDA_LLM_*` env
    /// vars. No-op when unset — LLM-backed UI affordances then hide
    /// (cluster labels, failure classification, NL→DSL).
    pub async fn load_llm_from_env(&self) {
        let Some(cfg) = LlmConfig::from_env() else { return };
        let endpoint = cfg.endpoint.clone();
        let model = cfg.model.clone();
        let mut guard = self.inner.llm_config.write().await;
        *guard = Some(cfg);
        tracing::info!(model = %model, endpoint = %endpoint, "registered LLM endpoint from environment");
    }

    /// Load embedding model configuration from environment variables and register it.
    ///
    /// Registers two models when configured:
    /// 1. Text model from `EMBEDDING_*` env vars (registered first so `find_by_modality(Text)` returns it first)
    /// 2. Multimodal model from `MULTIMODAL_EMBEDDING_*` env vars
    pub async fn load_embedding_from_env(&self) {
        // --- Text embedding model (EMBEDDING_*) ---
        // Two paths:
        //   1. `EMBEDDING_PRESET=<name>` — pull a curated config
        //      from `EmbeddingModelConfig::text_presets()`. Endpoint
        //      and api_key still come from env vars.
        //   2. Manual: set EMBEDDING_MODEL + EMBEDDING_DIMENSION.
        // See research/2026-05-19-sota-audit/03-text-embeddings.md
        let text_endpoint = std::env::var("EMBEDDING_ENDPOINT")
            .ok()
            .filter(|s| !s.is_empty());
        let text_api_key = std::env::var("EMBEDDING_API_KEY").unwrap_or_default();
        let text_cfg = if let Some(endpoint) = text_endpoint.as_ref() {
            if let Ok(preset_name) = std::env::var("EMBEDDING_PRESET") {
                EmbeddingModelConfig::preset(&preset_name)
                    .map(|c| c.with_endpoint(endpoint.clone(), text_api_key.clone()))
                    .or_else(|| {
                        tracing::warn!(
                            preset = %preset_name,
                            "EMBEDDING_PRESET unknown; falling back to manual EMBEDDING_MODEL/DIMENSION"
                        );
                        None
                    })
            } else {
                None
            }
            .or_else(|| {
                std::env::var("EMBEDDING_MODEL")
                    .ok()
                    .filter(|s| !s.is_empty())
                    .map(|model| {
                        let dim = std::env::var("EMBEDDING_DIMENSION")
                            .ok()
                            .and_then(|s| s.parse().ok())
                            .unwrap_or(1024);
                        EmbeddingModelConfig {
                            name: model,
                            endpoint: endpoint.clone(),
                            api_key: text_api_key.clone(),
                            supported_modalities: vec![Modality::Text],
                            output_dimension: dim,
                            max_input_size_bytes: 32_768,
                            timeout_seconds: 30,
                            batch_size: 32,
                        }
                    })
            })
        } else {
            None
        };

        // --- Multimodal embedding model (MULTIMODAL_EMBEDDING_*) ---
        let mm_endpoint = std::env::var("MULTIMODAL_EMBEDDING_ENDPOINT")
            .ok()
            .filter(|s| !s.is_empty());
        let mm_api_key = std::env::var("MULTIMODAL_EMBEDDING_API_KEY").unwrap_or_default();
        let mm_cfg = if let Some(endpoint) = mm_endpoint.as_ref() {
            if let Ok(preset_name) = std::env::var("MULTIMODAL_EMBEDDING_PRESET") {
                EmbeddingModelConfig::preset(&preset_name)
                    .map(|c| c.with_endpoint(endpoint.clone(), mm_api_key.clone()))
                    .or_else(|| {
                        tracing::warn!(
                            preset = %preset_name,
                            "MULTIMODAL_EMBEDDING_PRESET unknown; falling back to manual settings"
                        );
                        None
                    })
            } else {
                None
            }
            .or_else(|| {
                std::env::var("MULTIMODAL_EMBEDDING_MODEL")
                    .ok()
                    .filter(|s| !s.is_empty())
                    .map(|model| {
                        let dim = std::env::var("MULTIMODAL_EMBEDDING_DIMENSION")
                            .ok()
                            .and_then(|s| s.parse().ok())
                            .unwrap_or(2048);
                        EmbeddingModelConfig {
                            name: model,
                            endpoint: endpoint.clone(),
                            api_key: mm_api_key.clone(),
                            supported_modalities: vec![Modality::Text, Modality::Image, Modality::Video],
                            output_dimension: dim,
                            max_input_size_bytes: 20 * 1024 * 1024,
                            timeout_seconds: 60,
                            batch_size: 8,
                        }
                    })
            })
        } else {
            None
        };

        if text_cfg.is_none() && mm_cfg.is_none() {
            return;
        }

        let mut registry = self.inner.embedding_registry.write().await;
        if let Some(cfg) = text_cfg {
            tracing::info!(
                model = %cfg.name,
                endpoint = %cfg.endpoint,
                dimension = cfg.output_dimension,
                "registered text embedding model from environment"
            );
            registry.register(cfg);
        }
        if let Some(cfg) = mm_cfg {
            tracing::info!(
                model = %cfg.name,
                endpoint = %cfg.endpoint,
                dimension = cfg.output_dimension,
                "registered multimodal embedding model from environment"
            );
            registry.register(cfg);
        }
    }

    /// Recover collections from `data_dir` on startup.
    ///
    /// For each subdirectory of `data_dir` that contains a `collection.json`,
    /// re-build the in-memory state (schema, segment managers, query
    /// executor) and load any sealed segments under each shard. This is
    /// what makes Veda survive restarts; without it the in-memory
    /// `schemas` map starts empty and any prior data is orphaned.
    ///
    /// Returns the number of collections recovered.
    pub async fn recover_collections_from_disk(&self) -> usize {
        use veda_storage::segment_manager::SegmentManager;
        use veda_query::distributed::DistributedQueryExecutor;
        use veda_query::local_shard::LocalShardSearcher;

        let data_dir = &self.inner.data_dir;
        let Ok(entries) = std::fs::read_dir(data_dir) else {
            return 0;
        };

        let mut recovered = 0usize;
        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }
            // Skip Veda-managed dirs that aren't collections.
            if let Some(name) = path.file_name().and_then(|s| s.to_str())
                && (name == "blobs" || name.starts_with('.'))
            {
                continue;
            }
            let marker = path.join("collection.json");
            if !marker.exists() {
                continue;
            }
            let Some(name) = path.file_name().and_then(|s| s.to_str()).map(String::from) else {
                continue;
            };

            let schema_bytes = match std::fs::read(&marker) {
                Ok(b) => b,
                Err(e) => {
                    tracing::warn!(collection = %name, error = %e, "recover: read schema failed");
                    continue;
                }
            };
            let schema: CollectionSchema = match serde_json::from_slice(&schema_bytes) {
                Ok(s) => s,
                Err(e) => {
                    tracing::warn!(collection = %name, error = %e, "recover: parse schema failed");
                    continue;
                }
            };

            let mut shard_managers: HashMap<u32, SegmentManager> = HashMap::new();
            let mut executor = DistributedQueryExecutor::new();
            let mut total_sealed = 0usize;

            for shard_id in 0..schema.shard_count() {
                let mut sm = match SegmentManager::new(
                    schema.clone(),
                    shard_id,
                    &path,
                    512 * 1024 * 1024,
                ) {
                    Ok(s) => s,
                    Err(e) => {
                        tracing::warn!(collection = %name, shard = shard_id, error = %e, "recover: build segment manager failed");
                        continue;
                    }
                };
                match sm.load_sealed_segments() {
                    Ok(n) => total_sealed += n,
                    Err(e) => {
                        tracing::warn!(collection = %name, shard = shard_id, error = %e, "recover: load_sealed_segments failed");
                    }
                }
                // Build a parallel manager for the searcher (same pattern as
                // create_collection — the executor wants its own owned
                // copy of the segments since it builds ANN indexes lazily).
                let mut searcher_sm = match SegmentManager::new(
                    schema.clone(),
                    shard_id,
                    &path,
                    512 * 1024 * 1024,
                ) {
                    Ok(s) => s,
                    Err(e) => {
                        tracing::warn!(collection = %name, shard = shard_id, error = %e, "recover: build searcher manager failed");
                        continue;
                    }
                };
                let _ = searcher_sm.load_sealed_segments();
                let searcher = LocalShardSearcher::new(searcher_sm);
                executor.register_shard(shard_id, Box::new(searcher));
                shard_managers.insert(shard_id, sm);
            }

            self.inner.schemas.write().await.insert(name.clone(), schema);
            self.inner
                .segment_managers
                .write()
                .await
                .insert(name.clone(), shard_managers);
            self.inner
                .query_executors
                .write()
                .await
                .insert(name.clone(), executor);
            recovered += 1;
            tracing::info!(collection = %name, sealed_segments = total_sealed, "recovered collection from disk");
        }
        recovered
    }

    /// Load blob store configuration from environment variables.
    ///
    /// If `VEDA_BLOB_BACKEND` is set, creates an `ObjectBlobStore` and swaps it
    /// into the RwLock (replacing the default `LocalBlobStore`).
    pub async fn load_blob_store_from_env(&self) {
        let backend = match std::env::var("VEDA_BLOB_BACKEND").ok().filter(|s| !s.is_empty()) {
            Some(b) => b,
            None => {
                tracing::info!("VEDA_BLOB_BACKEND not set, keeping local blob store");
                return;
            }
        };

        let config = ObjectStoreConfig {
            backend,
            endpoint: std::env::var("VEDA_BLOB_ENDPOINT")
                .unwrap_or_else(|_| "http://minio:9000".to_owned()),
            bucket: std::env::var("VEDA_BLOB_BUCKET")
                .unwrap_or_else(|_| "veda-blobs".to_owned()),
            access_key: std::env::var("VEDA_BLOB_ACCESS_KEY").unwrap_or_default(),
            secret_key: std::env::var("VEDA_BLOB_SECRET_KEY").unwrap_or_default(),
            region: std::env::var("VEDA_BLOB_REGION")
                .unwrap_or_else(|_| "us-east-1".to_owned()),
        };

        match ObjectBlobStore::new(&config) {
            Ok(store) => {
                let mut guard = self.inner.blob_store.write().await;
                *guard = Arc::new(store);
                tracing::info!(
                    backend = %config.backend,
                    endpoint = %config.endpoint,
                    bucket = %config.bucket,
                    "switched to object blob store"
                );
            }
            Err(e) => {
                tracing::error!(error = %e, "failed to create object blob store, keeping local");
            }
        }
    }
}

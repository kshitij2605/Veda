//! Built-in advisor rules.
//!
//! A [`Rule`] inspects an [`OpsSnapshot`] (plus the recent
//! [`SnapshotHistory`] for rate-based signals) and emits zero or more
//! [`RuleHit`]s — one per breaching target (collection / shard / node /
//! global). Each hit carries a *root cause* (`detail`), an actionable
//! `recommendation`, an optional `runbook` pointer, and a
//! [`RemediationAction`] marker (never executed here).
//!
//! Thresholds come from the rule's [`RuleConfig::thresholds`] JSON with
//! a compiled-in default, so operators can retune without a rebuild.
//! [`run_rules`] drives the catalog, skipping disabled/muted rules and
//! applying any `severity_override`.

use crate::advisor::config::{AdvisorConfig, RuleConfig};
use crate::advisor::event::{RemediationAction, Scope, Severity};
use crate::advisor::snapshot::{OpsSnapshot, SnapshotHistory};

/// A transient finding from a rule. Turned into a durable `OpsEvent`
/// by the [`EventStore`](crate::advisor::event::EventStore).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuleHit {
    pub rule_id: String,
    pub scope: Scope,
    pub severity: Severity,
    pub title: String,
    pub detail: String,
    pub recommendation: String,
    pub runbook: Option<String>,
    pub remediation: RemediationAction,
}

/// A snapshot-derived health rule.
pub trait Rule: Send + Sync {
    /// Stable id (matches the `RuleConfig` map key and the event id
    /// prefix). Must be unique within the catalog.
    fn id(&self) -> &'static str;
    /// Natural severity, used when the config sets no override and a
    /// hit doesn't pick its own.
    fn default_severity(&self) -> Severity;
    /// Evaluate against a snapshot. `history` supplies rates/trends;
    /// `cfg` supplies thresholds. May return multiple hits.
    fn check(&self, snap: &OpsSnapshot, history: &SnapshotHistory, cfg: &RuleConfig) -> Vec<RuleHit>;
}

/// Read a numeric threshold from the rule's `thresholds` JSON, falling
/// back to `default` when the key is absent or not a number.
fn threshold_f64(cfg: &RuleConfig, key: &str, default: f64) -> f64 {
    cfg.thresholds
        .get(key)
        .and_then(|v| v.as_f64())
        .unwrap_or(default)
}

fn threshold_u64(cfg: &RuleConfig, key: &str, default: u64) -> u64 {
    cfg.thresholds
        .get(key)
        .and_then(|v| v.as_u64())
        .unwrap_or(default)
}

// ---------------------------------------------------------------------
// Rules
// ---------------------------------------------------------------------

/// Too many sealed segments in a collection — compaction is behind.
struct CompactionBacklog;
impl Rule for CompactionBacklog {
    fn id(&self) -> &'static str {
        "compaction_backlog"
    }
    fn default_severity(&self) -> Severity {
        Severity::Medium
    }
    fn check(&self, snap: &OpsSnapshot, _h: &SnapshotHistory, cfg: &RuleConfig) -> Vec<RuleHit> {
        let limit = threshold_u64(cfg, "sealed_segments", 20);
        let mut hits = Vec::new();
        for c in &snap.collections {
            if c.sealed_segments as u64 > limit && !c.compaction_in_progress {
                hits.push(RuleHit {
                    rule_id: self.id().into(),
                    scope: Scope::Collection(c.name.clone()),
                    severity: self.default_severity(),
                    title: format!("Compaction backlog on '{}'", c.name),
                    detail: format!(
                        "{} sealed segments across {} shard(s) (threshold {}); no compaction running. Read amplification and search latency grow with segment count.",
                        c.sealed_segments, c.shard_count, limit
                    ),
                    recommendation: format!(
                        "Trigger compaction for '{}' to merge sealed segments.",
                        c.name
                    ),
                    runbook: Some(format!("POST /v1/collections/{}/compact", c.name)),
                    remediation: RemediationAction::Compact {
                        collection: c.name.clone(),
                    },
                });
            }
        }
        hits
    }
}

/// A growing segment that has accumulated many records but never seals.
struct SealStalled;
impl Rule for SealStalled {
    fn id(&self) -> &'static str {
        "seal_stalled"
    }
    fn default_severity(&self) -> Severity {
        Severity::Medium
    }
    fn check(&self, snap: &OpsSnapshot, _h: &SnapshotHistory, cfg: &RuleConfig) -> Vec<RuleHit> {
        let limit = threshold_u64(cfg, "growing_records", 100_000);
        let mut hits = Vec::new();
        for c in &snap.collections {
            if c.growing_records > limit && c.sealed_segments == 0 {
                hits.push(RuleHit {
                    rule_id: self.id().into(),
                    scope: Scope::Collection(c.name.clone()),
                    severity: self.default_severity(),
                    title: format!("Growing segment not sealing on '{}'", c.name),
                    detail: format!(
                        "{} records sit in the growing segment with 0 sealed segments (threshold {}). The growing segment is unindexed, so queries scan it linearly.",
                        c.growing_records, limit
                    ),
                    recommendation: format!(
                        "Seal '{}' to flush the growing segment and build its index; check the auto-seal threshold/worker.",
                        c.name
                    ),
                    runbook: Some(format!("POST /v1/collections/{}/seal", c.name)),
                    remediation: RemediationAction::None,
                });
            }
        }
        hits
    }
}

/// p95 search latency over the window exceeds the budget.
struct HighSearchLatency;
impl Rule for HighSearchLatency {
    fn id(&self) -> &'static str {
        "high_search_latency"
    }
    fn default_severity(&self) -> Severity {
        Severity::Medium
    }
    fn check(&self, snap: &OpsSnapshot, _h: &SnapshotHistory, cfg: &RuleConfig) -> Vec<RuleHit> {
        let limit = threshold_f64(cfg, "p95_ms", 500.0);
        if snap.query.count > 0 && snap.query.p95_ms > limit {
            return vec![RuleHit {
                rule_id: self.id().into(),
                scope: Scope::Global,
                severity: self.default_severity(),
                title: "High search latency".into(),
                detail: format!(
                    "p95 search latency is {:.0} ms over the last {} s ({} queries, threshold {:.0} ms).",
                    snap.query.p95_ms, snap.query.window_secs, snap.query.count, limit
                ),
                recommendation: "Investigate slow queries: check segment count/compaction, index parameters (efSearch/nprobe), and node load.".into(),
                runbook: Some("GET /v1/admin/metrics; use \"explain\": true on search".into()),
                remediation: RemediationAction::None,
            }];
        }
        Vec::new()
    }
}

/// Many individually-slow queries in the window.
struct SlowQuerySurge;
impl Rule for SlowQuerySurge {
    fn id(&self) -> &'static str {
        "slow_query_surge"
    }
    fn default_severity(&self) -> Severity {
        Severity::Low
    }
    fn check(&self, snap: &OpsSnapshot, _h: &SnapshotHistory, cfg: &RuleConfig) -> Vec<RuleHit> {
        let limit = threshold_u64(cfg, "slow_count", 10);
        if snap.query.slow_count > limit {
            return vec![RuleHit {
                rule_id: self.id().into(),
                scope: Scope::Global,
                severity: self.default_severity(),
                title: "Surge of slow queries".into(),
                detail: format!(
                    "{} slow queries in the last {} s (threshold {}).",
                    snap.query.slow_count, snap.query.window_secs, limit
                ),
                recommendation: "Review the query log for outliers; expensive filters or large top-k can dominate. Consider adding scalar indexes.".into(),
                runbook: Some("GET /v1/admin/metrics".into()),
                remediation: RemediationAction::None,
            }];
        }
        Vec::new()
    }
}

/// Error rate (errors/requests) over the window is too high.
struct HighErrorRate;
impl Rule for HighErrorRate {
    fn id(&self) -> &'static str {
        "high_error_rate"
    }
    fn default_severity(&self) -> Severity {
        Severity::High
    }
    fn check(&self, snap: &OpsSnapshot, history: &SnapshotHistory, cfg: &RuleConfig) -> Vec<RuleHit> {
        let limit = threshold_f64(cfg, "error_ratio", 0.05);
        let req_rate = history.rate("veda_requests_total", snap);
        let err_rate = history.rate("veda_errors_total", snap);
        // Cold start / no previous snapshot / no traffic → no signal.
        let (Some(req), Some(err)) = (req_rate, err_rate) else {
            return Vec::new();
        };
        if req <= 0.0 {
            return Vec::new();
        }
        let ratio = err / req;
        if ratio > limit {
            return vec![RuleHit {
                rule_id: self.id().into(),
                scope: Scope::Global,
                severity: self.default_severity(),
                title: "Elevated error rate".into(),
                detail: format!(
                    "{:.1}% of requests are failing ({:.2} err/s vs {:.2} req/s; threshold {:.1}%).",
                    ratio * 100.0, err, req, limit * 100.0
                ),
                recommendation: "Check recent errors in logs/traces; correlate with deploys, dependency health (MinIO/NATS), and embedding-service availability.".into(),
                runbook: Some("GET /v1/admin/metrics; GET /v1/admin/health".into()),
                remediation: RemediationAction::None,
            }];
        }
        Vec::new()
    }
}

/// Too many ERROR-level log records in the recent window.
struct LogErrorRate;
impl Rule for LogErrorRate {
    fn id(&self) -> &'static str {
        "log_error_rate"
    }
    fn default_severity(&self) -> Severity {
        Severity::High
    }
    fn check(&self, snap: &OpsSnapshot, _h: &SnapshotHistory, cfg: &RuleConfig) -> Vec<RuleHit> {
        let limit = threshold_u64(cfg, "error_logs", 20);
        if snap.error_log_count > limit {
            return vec![RuleHit {
                rule_id: self.id().into(),
                scope: Scope::Global,
                severity: self.default_severity(),
                title: "High error-log volume".into(),
                detail: format!(
                    "{} ERROR logs in last 5m (threshold {}).",
                    snap.error_log_count, limit
                ),
                recommendation: "Inspect recent errors (/v1/admin/logs?level=ERROR) and categories (/v1/admin/logs/categories) to find the dominant failure.".into(),
                runbook: Some("GET /v1/admin/logs?level=ERROR".into()),
                remediation: RemediationAction::None,
            }];
        }
        Vec::new()
    }
}

/// Text embedding service is unavailable.
struct EmbeddingUnhealthy;
impl Rule for EmbeddingUnhealthy {
    fn id(&self) -> &'static str {
        "embedding_unhealthy"
    }
    fn default_severity(&self) -> Severity {
        Severity::Medium
    }
    fn check(&self, snap: &OpsSnapshot, _h: &SnapshotHistory, _cfg: &RuleConfig) -> Vec<RuleHit> {
        let mut hits = Vec::new();
        for (which, status) in [
            ("embedding", &snap.health.embedding),
            ("multimodal_embedding", &snap.health.multimodal_embedding),
        ] {
            if status == "unavailable" {
                hits.push(RuleHit {
                    rule_id: self.id().into(),
                    scope: Scope::Global,
                    severity: self.default_severity(),
                    title: format!("Embedding service unavailable ({which})"),
                    detail: format!(
                        "Health probe for '{which}' reports 'unavailable'. Embedding-on-insert and embedded search will fail."
                    ),
                    recommendation: "Restart/check the embedding backend (TEI / vLLM) and its network reachability from the gateway.".into(),
                    runbook: Some("GET /v1/admin/health".into()),
                    remediation: RemediationAction::None,
                });
            }
        }
        hits
    }
}

/// A core dependency (object store / message bus) is down.
struct DependencyDown;
impl Rule for DependencyDown {
    fn id(&self) -> &'static str {
        "dependency_down"
    }
    fn default_severity(&self) -> Severity {
        Severity::High
    }
    fn check(&self, snap: &OpsSnapshot, _h: &SnapshotHistory, _cfg: &RuleConfig) -> Vec<RuleHit> {
        let mut hits = Vec::new();
        for (dep, status) in [("minio", &snap.health.minio), ("nats", &snap.health.nats)] {
            if status == "unavailable" {
                hits.push(RuleHit {
                    rule_id: self.id().into(),
                    scope: Scope::Global,
                    severity: self.default_severity(),
                    title: format!("Dependency down: {dep}"),
                    detail: format!(
                        "Health probe for '{dep}' reports 'unavailable'. Durability/replication and blob I/O are degraded."
                    ),
                    recommendation: format!(
                        "Restore '{dep}' connectivity; verify credentials/endpoint and that the service is running."
                    ),
                    runbook: Some("GET /v1/admin/health".into()),
                    remediation: RemediationAction::None,
                });
            }
        }
        hits
    }
}

/// Process RSS over the configured ceiling.
struct MemoryPressure;
impl Rule for MemoryPressure {
    fn id(&self) -> &'static str {
        "memory_pressure"
    }
    fn default_severity(&self) -> Severity {
        Severity::Medium
    }
    fn check(&self, snap: &OpsSnapshot, _h: &SnapshotHistory, cfg: &RuleConfig) -> Vec<RuleHit> {
        let limit = threshold_u64(cfg, "rss_bytes", 8 * 1024 * 1024 * 1024);
        let Some(rss) = snap.rss_bytes else {
            return Vec::new();
        };
        if rss > limit {
            return vec![RuleHit {
                rule_id: self.id().into(),
                scope: Scope::Global,
                severity: self.default_severity(),
                title: "Memory pressure".into(),
                detail: format!(
                    "Process RSS is {:.1} GiB (threshold {:.1} GiB).",
                    rss as f64 / 1.073_741_824e9,
                    limit as f64 / 1.073_741_824e9
                ),
                recommendation: "Reduce in-memory footprint: enable `use_log_primary` / quantization (SQ8/RaBitQ) for large collections, or scale out.".into(),
                runbook: Some("GET /v1/admin/metrics".into()),
                remediation: RemediationAction::None,
            }];
        }
        Vec::new()
    }
}

/// Replica safe timestamp lags wall clock by too much.
struct ReplicaLag;
impl Rule for ReplicaLag {
    fn id(&self) -> &'static str {
        "replica_lag"
    }
    fn default_severity(&self) -> Severity {
        Severity::High
    }
    fn check(&self, snap: &OpsSnapshot, _h: &SnapshotHistory, cfg: &RuleConfig) -> Vec<RuleHit> {
        let limit_ms = threshold_u64(cfg, "lag_ms", 60_000) as i64;
        let Some(ts) = snap.replica_safe_ts else {
            return Vec::new();
        };
        // HLC encoding: (physical_ns << 32) | logical. Extract the
        // physical-ns component and convert to ms. Best-effort: if the
        // encoding differs this yields a nonsensical (likely huge-lag)
        // value, so guard against a future-dated watermark too.
        let physical_ns = ts >> 32;
        let physical_ms = (physical_ns / 1_000_000) as i64;
        if physical_ms <= 0 {
            return Vec::new();
        }
        let lag = snap.taken_at_ms - physical_ms;
        if lag > limit_ms {
            return vec![RuleHit {
                rule_id: self.id().into(),
                scope: Scope::Global,
                severity: self.default_severity(),
                title: "Replica lag".into(),
                detail: format!(
                    "Replica safe timestamp lags wall clock by {} ms (threshold {} ms). Reads gated on the safe-ts may see stale data.",
                    lag, limit_ms
                ),
                recommendation: "Check replication/applier throughput and the message bus; a lagging or stuck follower holds back the safe-ts.".into(),
                runbook: Some("GET /v1/admin/nodes".into()),
                remediation: RemediationAction::None,
            }];
        }
        Vec::new()
    }
}

/// A node reporting an explicitly unhealthy status.
struct NodeUnhealthy;
impl Rule for NodeUnhealthy {
    fn id(&self) -> &'static str {
        "node_unhealthy"
    }
    fn default_severity(&self) -> Severity {
        Severity::High
    }
    fn check(&self, snap: &OpsSnapshot, _h: &SnapshotHistory, _cfg: &RuleConfig) -> Vec<RuleHit> {
        let mut hits = Vec::new();
        for n in &snap.nodes {
            // Only fire on an explicit bad status — peers commonly
            // report "unknown" (not probed) and that is NOT a fault.
            if n.status == "unhealthy" || n.status == "unavailable" {
                hits.push(RuleHit {
                    rule_id: self.id().into(),
                    scope: Scope::Node(n.id.clone()),
                    severity: self.default_severity(),
                    title: format!("Node {} {}", n.id, n.status),
                    detail: format!(
                        "Node '{}' (role {}) reports status '{}'. Shards it owns may be unavailable or under-replicated.",
                        n.id, n.role, n.status
                    ),
                    recommendation: format!(
                        "Investigate node '{}': process health, disk, and network; consider failover/replacement.",
                        n.id
                    ),
                    runbook: Some("GET /v1/admin/nodes".into()),
                    remediation: RemediationAction::None,
                });
            }
        }
        hits
    }
}

/// Records present but no segments ever sealed cluster-wide — the seal
/// pipeline may be globally stuck.
struct NoCompactionProgress;
impl Rule for NoCompactionProgress {
    fn id(&self) -> &'static str {
        "no_compaction_progress"
    }
    fn default_severity(&self) -> Severity {
        Severity::Low
    }
    fn check(&self, snap: &OpsSnapshot, _h: &SnapshotHistory, cfg: &RuleConfig) -> Vec<RuleHit> {
        let min_records = threshold_u64(cfg, "min_total_records", 1_000_000);
        let total_sealed: usize = snap.collections.iter().map(|c| c.sealed_segments).sum();
        if snap.global.total_records > min_records && total_sealed == 0 {
            return vec![RuleHit {
                rule_id: self.id().into(),
                scope: Scope::Global,
                severity: self.default_severity(),
                title: "No sealed segments cluster-wide".into(),
                detail: format!(
                    "{} records across {} collection(s) but 0 sealed segments anywhere (threshold {} records). The seal/flush pipeline may be stuck globally.",
                    snap.global.total_records, snap.global.active_collections, min_records
                ),
                recommendation: "Check the seal worker / background flush task and disk space; nothing has been persisted to sealed segments.".into(),
                runbook: Some("GET /v1/admin/metrics".into()),
                remediation: RemediationAction::None,
            }];
        }
        Vec::new()
    }
}

/// The built-in rule catalog.
pub fn catalog() -> Vec<Box<dyn Rule>> {
    vec![
        Box::new(CompactionBacklog),
        Box::new(SealStalled),
        Box::new(HighSearchLatency),
        Box::new(SlowQuerySurge),
        Box::new(HighErrorRate),
        Box::new(LogErrorRate),
        Box::new(EmbeddingUnhealthy),
        Box::new(DependencyDown),
        Box::new(MemoryPressure),
        Box::new(ReplicaLag),
        Box::new(NodeUnhealthy),
        Box::new(NoCompactionProgress),
    ]
}

/// Run every enabled, un-muted rule in `catalog` against the snapshot,
/// applying each rule's `severity_override` to the hits it produces.
pub fn run_rules(
    catalog: &[Box<dyn Rule>],
    snap: &OpsSnapshot,
    history: &SnapshotHistory,
    cfg: &AdvisorConfig,
) -> Vec<RuleHit> {
    let mut out = Vec::new();
    for rule in catalog {
        let rc = cfg.for_rule(rule.id());
        if !rc.enabled || rc.muted {
            continue;
        }
        let mut hits = rule.check(snap, history, &rc);
        if let Some(sev) = rc.severity_override {
            for h in &mut hits {
                h.severity = sev;
            }
        }
        out.append(&mut hits);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::advisor::snapshot::{
        CollectionSnapshot, GlobalSnapshot, HealthSnapshot, NodeSnapshot, OpsSnapshot,
        QuerySnapshot, SnapshotHistory, DEFAULT_HISTORY_CAP,
    };
    use std::collections::HashMap;

    fn healthy_health() -> HealthSnapshot {
        HealthSnapshot {
            minio: "healthy".into(),
            nats: "healthy".into(),
            embedding: "healthy".into(),
            multimodal_embedding: "healthy".into(),
        }
    }

    fn empty_query() -> QuerySnapshot {
        QuerySnapshot {
            window_secs: 300,
            count: 0,
            p50_ms: 0.0,
            p95_ms: 0.0,
            slow_count: 0,
        }
    }

    fn base_snapshot() -> OpsSnapshot {
        OpsSnapshot {
            taken_at_ms: 1_700_000_000_000,
            global: GlobalSnapshot {
                active_collections: 0,
                total_records: 0,
            },
            collections: Vec::new(),
            nodes: vec![NodeSnapshot {
                id: "node-0".into(),
                role: "leader".into(),
                status: "healthy".into(),
                shard_count: 1,
            }],
            replica_safe_ts: None,
            health: healthy_health(),
            query: empty_query(),
            rss_bytes: None,
            disk_total_bytes: None,
            disk_free_bytes: None,
            disk_used_pct: None,
            cpu_total_jiffies: None,
            cpu_busy_jiffies: None,
            num_cpus: 1,
            load1: None,
            load5: None,
            load15: None,
            open_fds: None,
            fd_limit: None,
            mem_total_bytes: None,
            mem_available_bytes: None,
            counters: HashMap::new(),
            error_log_count: 0,
            warn_log_count: 0,
        }
    }

    fn collection(name: &str, sealed: usize, growing: u64) -> CollectionSnapshot {
        CollectionSnapshot {
            name: name.into(),
            shard_count: 1,
            total_segments: sealed + 1,
            sealed_segments: sealed,
            growing_records: growing,
            total_records: growing,
            compaction_in_progress: false,
        }
    }

    fn empty_history() -> SnapshotHistory {
        SnapshotHistory::new(DEFAULT_HISTORY_CAP)
    }

    fn find<'a>(hits: &'a [RuleHit], rule_id: &str) -> Option<&'a RuleHit> {
        hits.iter().find(|h| h.rule_id == rule_id)
    }

    #[test]
    fn healthy_snapshot_trips_nothing() {
        let snap = base_snapshot();
        let hits = run_rules(&catalog(), &snap, &empty_history(), &AdvisorConfig::default());
        assert!(hits.is_empty(), "unexpected hits: {hits:?}");
    }

    #[test]
    fn compaction_backlog_fires_per_collection() {
        let mut snap = base_snapshot();
        snap.collections.push(collection("docs", 25, 10));
        snap.collections.push(collection("images", 5, 10)); // below threshold
        let cfg = RuleConfig::default();
        let hits = CompactionBacklog.check(&snap, &empty_history(), &cfg);
        assert_eq!(hits.len(), 1);
        let h = &hits[0];
        assert_eq!(h.rule_id, "compaction_backlog");
        assert_eq!(h.scope, Scope::Collection("docs".into()));
        assert_eq!(h.severity, Severity::Medium);
        assert!(!h.recommendation.is_empty());
        assert_eq!(
            h.remediation,
            RemediationAction::Compact {
                collection: "docs".into()
            }
        );
    }

    #[test]
    fn high_search_latency_fires() {
        let mut snap = base_snapshot();
        snap.query = QuerySnapshot {
            window_secs: 300,
            count: 100,
            p50_ms: 120.0,
            p95_ms: 800.0,
            slow_count: 0,
        };
        let hits = HighSearchLatency.check(&snap, &empty_history(), &RuleConfig::default());
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].scope, Scope::Global);
        assert!(!hits[0].recommendation.is_empty());
    }

    #[test]
    fn dependency_down_high_severity() {
        let mut snap = base_snapshot();
        snap.health.minio = "unavailable".into();
        let hits = DependencyDown.check(&snap, &empty_history(), &RuleConfig::default());
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].severity, Severity::High);
        assert_eq!(hits[0].rule_id, "dependency_down");
    }

    #[test]
    fn memory_pressure_fires_over_threshold() {
        let mut snap = base_snapshot();
        snap.rss_bytes = Some(10 * 1024 * 1024 * 1024); // 10 GiB
        let hits = MemoryPressure.check(&snap, &empty_history(), &RuleConfig::default());
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].scope, Scope::Global);
    }

    #[test]
    fn high_error_rate_via_two_snapshot_history() {
        // prev: 100 reqs, 0 errs @ t=1000ms; now: 200 reqs, 50 errs @ t=2000ms.
        // over 1s: req_rate=100/s, err_rate=50/s → ratio 0.5 > 0.05.
        let mut prev = base_snapshot();
        prev.taken_at_ms = 1000;
        prev.counters.insert("veda_requests_total".into(), 100.0);
        prev.counters.insert("veda_errors_total".into(), 0.0);

        let mut now = base_snapshot();
        now.taken_at_ms = 2000;
        now.counters.insert("veda_requests_total".into(), 200.0);
        now.counters.insert("veda_errors_total".into(), 50.0);

        let mut hist = SnapshotHistory::new(DEFAULT_HISTORY_CAP);
        hist.push(prev);

        let hits = HighErrorRate.check(&now, &hist, &RuleConfig::default());
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].severity, Severity::High);
    }

    #[test]
    fn high_error_rate_silent_on_cold_start() {
        // No history → no rate → no hit.
        let mut now = base_snapshot();
        now.counters.insert("veda_requests_total".into(), 200.0);
        now.counters.insert("veda_errors_total".into(), 50.0);
        let hits = HighErrorRate.check(&now, &empty_history(), &RuleConfig::default());
        assert!(hits.is_empty());
    }

    #[test]
    fn log_error_rate_fires_over_threshold() {
        let mut snap = base_snapshot();
        snap.error_log_count = 25; // > default 20
        let hits = LogErrorRate.check(&snap, &empty_history(), &RuleConfig::default());
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].rule_id, "log_error_rate");
        assert_eq!(hits[0].scope, Scope::Global);
        assert_eq!(hits[0].severity, Severity::High);
        assert!(!hits[0].recommendation.is_empty());
    }

    #[test]
    fn log_error_rate_silent_under_threshold() {
        let mut snap = base_snapshot();
        snap.error_log_count = 5; // <= default 20
        let hits = LogErrorRate.check(&snap, &empty_history(), &RuleConfig::default());
        assert!(hits.is_empty());
    }

    #[test]
    fn log_error_rate_healthy_zero_no_fire() {
        let snap = base_snapshot(); // error_log_count == 0
        let hits = LogErrorRate.check(&snap, &empty_history(), &RuleConfig::default());
        assert!(hits.is_empty());
    }

    #[test]
    fn node_unhealthy_ignores_unknown() {
        let mut snap = base_snapshot();
        snap.nodes = vec![
            NodeSnapshot {
                id: "node-1".into(),
                role: "follower".into(),
                status: "unknown".into(),
                shard_count: 0,
            },
            NodeSnapshot {
                id: "node-2".into(),
                role: "follower".into(),
                status: "unhealthy".into(),
                shard_count: 3,
            },
        ];
        let hits = NodeUnhealthy.check(&snap, &empty_history(), &RuleConfig::default());
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].scope, Scope::Node("node-2".into()));
    }

    #[test]
    fn run_rules_skips_disabled_and_muted() {
        let mut snap = base_snapshot();
        snap.collections.push(collection("docs", 25, 10));
        let mut cfg = AdvisorConfig::default();
        cfg.rules.insert(
            "compaction_backlog".into(),
            RuleConfig {
                enabled: false,
                ..Default::default()
            },
        );
        let hits = run_rules(&catalog(), &snap, &empty_history(), &cfg);
        assert!(find(&hits, "compaction_backlog").is_none());

        // muted also suppresses.
        let mut cfg2 = AdvisorConfig::default();
        cfg2.rules.insert(
            "compaction_backlog".into(),
            RuleConfig {
                muted: true,
                ..Default::default()
            },
        );
        let hits2 = run_rules(&catalog(), &snap, &empty_history(), &cfg2);
        assert!(find(&hits2, "compaction_backlog").is_none());
    }

    #[test]
    fn run_rules_applies_severity_override() {
        let mut snap = base_snapshot();
        snap.collections.push(collection("docs", 25, 10));
        let mut cfg = AdvisorConfig::default();
        cfg.rules.insert(
            "compaction_backlog".into(),
            RuleConfig {
                severity_override: Some(Severity::High),
                ..Default::default()
            },
        );
        let hits = run_rules(&catalog(), &snap, &empty_history(), &cfg);
        let h = find(&hits, "compaction_backlog").expect("rule fired");
        assert_eq!(h.severity, Severity::High);
    }

    #[test]
    fn threshold_override_from_config() {
        let mut snap = base_snapshot();
        snap.collections.push(collection("docs", 25, 10));
        // Raise threshold above 25 → no hit.
        let cfg = RuleConfig {
            thresholds: serde_json::json!({ "sealed_segments": 50 }),
            ..Default::default()
        };
        let hits = CompactionBacklog.check(&snap, &empty_history(), &cfg);
        assert!(hits.is_empty());
    }

    #[test]
    fn catalog_ids_are_unique() {
        let cat = catalog();
        let mut ids: Vec<&str> = cat.iter().map(|r| r.id()).collect();
        ids.sort_unstable();
        let n = ids.len();
        ids.dedup();
        assert_eq!(ids.len(), n, "duplicate rule id in catalog");
        assert!(n >= 10, "expected at least 10 rules, got {n}");
    }
}

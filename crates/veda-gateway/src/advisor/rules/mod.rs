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

/// Data-dir disk utilisation over the configured ceiling. Headline
/// resource rule. Linux-only signal; skips cleanly when unknown.
struct DiskSpaceLow;
impl Rule for DiskSpaceLow {
    fn id(&self) -> &'static str {
        "disk_space_low"
    }
    fn default_severity(&self) -> Severity {
        Severity::High
    }
    fn check(&self, snap: &OpsSnapshot, _h: &SnapshotHistory, cfg: &RuleConfig) -> Vec<RuleHit> {
        let limit = threshold_f64(cfg, "disk_used_pct", 90.0);
        let Some(pct) = snap.disk_used_pct else {
            return Vec::new();
        };
        if pct > limit {
            let free_gib = snap
                .disk_free_bytes
                .map(|b| b as f64 / 1.073_741_824e9)
                .unwrap_or(0.0);
            return vec![RuleHit {
                rule_id: self.id().into(),
                scope: Scope::Global,
                severity: self.default_severity(),
                title: "Disk space low".into(),
                detail: format!(
                    "data-dir disk {pct:.0}% used (free {free_gib:.1} GiB; threshold {limit:.0}%). Writes, seals, and compaction will fail when the disk fills."
                ),
                recommendation: "Free disk space (delete old data / logs), run compaction to reclaim space, or scale out / add storage.".into(),
                runbook: Some("GET /v1/admin/metrics".into()),
                remediation: RemediationAction::None,
            }];
        }
        Vec::new()
    }
}

/// CPU utilisation over the configured ceiling, computed from the
/// jiffy counters across the previous→current snapshot pair.
struct HighCpu;
impl Rule for HighCpu {
    fn id(&self) -> &'static str {
        "high_cpu"
    }
    fn default_severity(&self) -> Severity {
        Severity::Medium
    }
    fn check(&self, snap: &OpsSnapshot, history: &SnapshotHistory, cfg: &RuleConfig) -> Vec<RuleHit> {
        let limit = threshold_f64(cfg, "cpu_pct", 85.0);
        // Needs a previous snapshot to diff the jiffy counters.
        let Some(prev) = history.previous() else {
            return Vec::new();
        };
        let Some(pct) = crate::advisor::snapshot::cpu_pct(prev, snap) else {
            return Vec::new();
        };
        if pct > limit {
            return vec![RuleHit {
                rule_id: self.id().into(),
                scope: Scope::Global,
                severity: self.default_severity(),
                title: "High CPU utilisation".into(),
                detail: format!(
                    "CPU utilisation is {pct:.0}% across {} core(s) (threshold {limit:.0}%). Sustained high CPU degrades query/ingest throughput.",
                    snap.num_cpus
                ),
                recommendation: "Identify the hot path (heavy search/ingest, compaction, embedding); reduce load or scale out.".into(),
                runbook: Some("GET /v1/admin/metrics".into()),
                remediation: RemediationAction::None,
            }];
        }
        Vec::new()
    }
}

/// 1-minute load average per core over the configured ceiling.
struct HighLoad;
impl Rule for HighLoad {
    fn id(&self) -> &'static str {
        "high_load"
    }
    fn default_severity(&self) -> Severity {
        Severity::Medium
    }
    fn check(&self, snap: &OpsSnapshot, _h: &SnapshotHistory, cfg: &RuleConfig) -> Vec<RuleHit> {
        let limit = threshold_f64(cfg, "load_per_core", 2.0);
        let Some(load1) = snap.load1 else {
            return Vec::new();
        };
        let cores = snap.num_cpus.max(1) as f64;
        let per_core = load1 / cores;
        if per_core > limit {
            return vec![RuleHit {
                rule_id: self.id().into(),
                scope: Scope::Global,
                severity: self.default_severity(),
                title: "High system load".into(),
                detail: format!(
                    "1-minute load average is {load1:.2} across {} core(s) = {per_core:.2}/core (threshold {limit:.1}/core). Runnable work exceeds CPU capacity; latency suffers.",
                    snap.num_cpus
                ),
                recommendation: "Check for CPU/IO contention from other processes; reduce concurrent load or scale out.".into(),
                runbook: Some("GET /v1/admin/metrics".into()),
                remediation: RemediationAction::None,
            }];
        }
        Vec::new()
    }
}

/// Open file descriptors approaching the process limit.
struct FdExhaustion;
impl Rule for FdExhaustion {
    fn id(&self) -> &'static str {
        "fd_exhaustion"
    }
    fn default_severity(&self) -> Severity {
        Severity::High
    }
    fn check(&self, snap: &OpsSnapshot, _h: &SnapshotHistory, cfg: &RuleConfig) -> Vec<RuleHit> {
        let limit = threshold_f64(cfg, "fd_used_ratio", 0.8);
        let (Some(open), Some(max)) = (snap.open_fds, snap.fd_limit) else {
            return Vec::new();
        };
        if max == 0 {
            return Vec::new();
        }
        let ratio = open as f64 / max as f64;
        if ratio > limit {
            return vec![RuleHit {
                rule_id: self.id().into(),
                scope: Scope::Global,
                severity: self.default_severity(),
                title: "File descriptor exhaustion".into(),
                detail: format!(
                    "{open} of {max} file descriptors in use ({:.0}%, threshold {:.0}%). Exhausting FDs causes accept/open failures across the server.",
                    ratio * 100.0, limit * 100.0
                ),
                recommendation: "Raise the FD limit (ulimit -n / LimitNOFILE) or investigate an FD leak (sockets, segment files).".into(),
                runbook: Some("GET /v1/admin/metrics".into()),
                remediation: RemediationAction::None,
            }];
        }
        Vec::new()
    }
}

/// System-wide available memory has fallen to a small fraction of total.
struct SystemMemoryLow;
impl Rule for SystemMemoryLow {
    fn id(&self) -> &'static str {
        "system_memory_low"
    }
    fn default_severity(&self) -> Severity {
        Severity::Medium
    }
    fn check(&self, snap: &OpsSnapshot, _h: &SnapshotHistory, cfg: &RuleConfig) -> Vec<RuleHit> {
        let limit = threshold_f64(cfg, "mem_available_ratio", 0.1);
        let (Some(total), Some(avail)) = (snap.mem_total_bytes, snap.mem_available_bytes) else {
            return Vec::new();
        };
        if total == 0 {
            return Vec::new();
        }
        let ratio = avail as f64 / total as f64;
        if ratio < limit {
            return vec![RuleHit {
                rule_id: self.id().into(),
                scope: Scope::Global,
                severity: self.default_severity(),
                title: "System memory low".into(),
                detail: format!(
                    "Only {:.1} GiB of {:.1} GiB system memory available ({:.0}%, threshold {:.0}%). The OOM killer may terminate the process.",
                    avail as f64 / 1.073_741_824e9,
                    total as f64 / 1.073_741_824e9,
                    ratio * 100.0,
                    limit * 100.0
                ),
                recommendation: "Reduce in-memory footprint (quantization, use_log_primary), stop competing processes, or add RAM / scale out.".into(),
                runbook: Some("GET /v1/admin/metrics".into()),
                remediation: RemediationAction::None,
            }];
        }
        Vec::new()
    }
}

// ---------------------------------------------------------------------
// SLO rules — alert on burn-rate / error-budget over the per-SLO status
// the engine records onto `snap.slo_statuses`. Each emits one hit per
// matching SLO, scoped `Scope::Slo(slo_id)` so the EventStore dedups one
// event per (rule_id, slo). All skip cleanly when `slo_statuses` is empty.
// ---------------------------------------------------------------------

/// Fast burn-rate window over its alert threshold for an SLO.
struct SloBurnRateFast;
impl Rule for SloBurnRateFast {
    fn id(&self) -> &'static str {
        "slo_burn_rate_fast"
    }
    fn default_severity(&self) -> Severity {
        Severity::High
    }
    fn check(&self, snap: &OpsSnapshot, _h: &SnapshotHistory, _cfg: &RuleConfig) -> Vec<RuleHit> {
        let mut hits = Vec::new();
        for s in &snap.slo_statuses {
            if !s.breaching_fast {
                continue;
            }
            let burn = s.burn_rate_fast.unwrap_or(0.0);
            let budget = s
                .error_budget_remaining_pct
                .map(|p| format!("{p:.1}%"))
                .unwrap_or_else(|| "n/a".into());
            hits.push(RuleHit {
                rule_id: self.id().into(),
                scope: Scope::Slo(s.slo_id.clone()),
                severity: self.default_severity(),
                title: format!("SLO '{}' fast burn", s.slo_id),
                detail: format!(
                    "SLO '{}' is burning error budget at {:.1}x over the fast window (threshold {:.1}x); error budget remaining {}.",
                    s.slo_id, burn, s.fast_burn_threshold, budget
                ),
                recommendation: format!(
                    "Investigate the failing SLI for '{}' now (latency/errors) — at this rate the {}-window budget is exhausted imminently.",
                    s.slo_id, s.slo_id
                ),
                runbook: Some(format!("GET /v1/admin/slos/{}/status", s.slo_id)),
                remediation: RemediationAction::None,
            });
        }
        hits
    }
}

/// Slow burn-rate window over its alert threshold for an SLO.
struct SloBurnRateSlow;
impl Rule for SloBurnRateSlow {
    fn id(&self) -> &'static str {
        "slo_burn_rate_slow"
    }
    fn default_severity(&self) -> Severity {
        Severity::Medium
    }
    fn check(&self, snap: &OpsSnapshot, _h: &SnapshotHistory, _cfg: &RuleConfig) -> Vec<RuleHit> {
        let mut hits = Vec::new();
        for s in &snap.slo_statuses {
            if !s.breaching_slow {
                continue;
            }
            let burn = s.burn_rate_slow.unwrap_or(0.0);
            let budget = s
                .error_budget_remaining_pct
                .map(|p| format!("{p:.1}%"))
                .unwrap_or_else(|| "n/a".into());
            hits.push(RuleHit {
                rule_id: self.id().into(),
                scope: Scope::Slo(s.slo_id.clone()),
                severity: self.default_severity(),
                title: format!("SLO '{}' slow burn", s.slo_id),
                detail: format!(
                    "SLO '{}' is burning error budget at {:.1}x over the slow window (threshold {:.1}x); error budget remaining {}.",
                    s.slo_id, burn, s.slow_burn_threshold, budget
                ),
                recommendation: format!(
                    "Investigate the failing SLI for '{}' (latency/errors); a sustained slow burn erodes the compliance-window budget.",
                    s.slo_id
                ),
                runbook: Some(format!("GET /v1/admin/slos/{}/status", s.slo_id)),
                remediation: RemediationAction::None,
            });
        }
        hits
    }
}

/// SLO error budget exhausted over the compliance window.
struct SloBudgetExhausted;
impl Rule for SloBudgetExhausted {
    fn id(&self) -> &'static str {
        "slo_budget_exhausted"
    }
    fn default_severity(&self) -> Severity {
        Severity::High
    }
    fn check(&self, snap: &OpsSnapshot, _h: &SnapshotHistory, _cfg: &RuleConfig) -> Vec<RuleHit> {
        let mut hits = Vec::new();
        for s in &snap.slo_statuses {
            if !s.budget_exhausted {
                continue;
            }
            let remaining = s.error_budget_remaining_pct.unwrap_or(0.0);
            let sli = s
                .sli_pct
                .map(|p| format!("{p:.3}%"))
                .unwrap_or_else(|| "n/a".into());
            hits.push(RuleHit {
                rule_id: self.id().into(),
                scope: Scope::Slo(s.slo_id.clone()),
                severity: self.default_severity(),
                title: format!("SLO '{}' budget exhausted", s.slo_id),
                detail: format!(
                    "SLO '{}' has exhausted its error budget over the compliance window (remaining {:.1}%; SLI {} vs target {:.3}%).",
                    s.slo_id, remaining, sli, s.target_pct
                ),
                recommendation: format!(
                    "'{}' is out of error budget — freeze risky changes and prioritise reliability work on the failing SLI until the budget recovers.",
                    s.slo_id
                ),
                runbook: Some(format!("GET /v1/admin/slos/{}/status", s.slo_id)),
                remediation: RemediationAction::None,
            });
        }
        hits
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
        Box::new(DiskSpaceLow),
        Box::new(HighCpu),
        Box::new(HighLoad),
        Box::new(FdExhaustion),
        Box::new(SystemMemoryLow),
        Box::new(SloBurnRateFast),
        Box::new(SloBurnRateSlow),
        Box::new(SloBudgetExhausted),
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
            recent_latencies_ms: Vec::new(),
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
            slo_statuses: Vec::new(),
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
            recent_latencies_ms: Vec::new(),
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

    // --- resource rules ------------------------------------------------

    #[test]
    fn disk_space_low_fires_high() {
        let mut snap = base_snapshot();
        snap.disk_used_pct = Some(95.0);
        snap.disk_free_bytes = Some(5 * 1024 * 1024 * 1024);
        let hits = DiskSpaceLow.check(&snap, &empty_history(), &RuleConfig::default());
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].rule_id, "disk_space_low");
        assert_eq!(hits[0].scope, Scope::Global);
        assert_eq!(hits[0].severity, Severity::High);
        assert!(!hits[0].recommendation.is_empty());
    }

    #[test]
    fn disk_space_low_silent_under_threshold() {
        let mut snap = base_snapshot();
        snap.disk_used_pct = Some(50.0);
        let hits = DiskSpaceLow.check(&snap, &empty_history(), &RuleConfig::default());
        assert!(hits.is_empty());
    }

    #[test]
    fn disk_space_low_skips_on_none() {
        let snap = base_snapshot(); // disk_used_pct == None
        let hits = DiskSpaceLow.check(&snap, &empty_history(), &RuleConfig::default());
        assert!(hits.is_empty());
    }

    #[test]
    fn high_cpu_fires_over_threshold() {
        // prev busy 100/total 1000, now busy 1050/total 2000:
        // busy_delta 950 / total_delta 1000 = 95%.
        let mut prev = base_snapshot();
        prev.cpu_busy_jiffies = Some(100);
        prev.cpu_total_jiffies = Some(1000);
        let mut now = base_snapshot();
        now.cpu_busy_jiffies = Some(1050);
        now.cpu_total_jiffies = Some(2000);

        let mut hist = SnapshotHistory::new(DEFAULT_HISTORY_CAP);
        hist.push(prev);
        hist.push(now.clone());

        let hits = HighCpu.check(&now, &hist, &RuleConfig::default());
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].rule_id, "high_cpu");
        assert_eq!(hits[0].severity, Severity::Medium);
    }

    #[test]
    fn high_cpu_silent_when_low() {
        // busy_delta 100 / total_delta 1000 = 10%.
        let mut prev = base_snapshot();
        prev.cpu_busy_jiffies = Some(100);
        prev.cpu_total_jiffies = Some(1000);
        let mut now = base_snapshot();
        now.cpu_busy_jiffies = Some(200);
        now.cpu_total_jiffies = Some(2000);

        let mut hist = SnapshotHistory::new(DEFAULT_HISTORY_CAP);
        hist.push(prev);
        hist.push(now.clone());

        let hits = HighCpu.check(&now, &hist, &RuleConfig::default());
        assert!(hits.is_empty());
    }

    #[test]
    fn high_cpu_skips_without_previous() {
        let mut now = base_snapshot();
        now.cpu_busy_jiffies = Some(1050);
        now.cpu_total_jiffies = Some(2000);
        let hits = HighCpu.check(&now, &empty_history(), &RuleConfig::default());
        assert!(hits.is_empty());
    }

    #[test]
    fn high_load_fires_per_core() {
        let mut snap = base_snapshot();
        snap.load1 = Some(8.0);
        snap.num_cpus = 2; // 4.0/core > 2.0
        let hits = HighLoad.check(&snap, &empty_history(), &RuleConfig::default());
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].rule_id, "high_load");
        assert_eq!(hits[0].severity, Severity::Medium);
    }

    #[test]
    fn high_load_silent_when_low() {
        let mut snap = base_snapshot();
        snap.load1 = Some(1.0);
        snap.num_cpus = 2; // 0.5/core
        let hits = HighLoad.check(&snap, &empty_history(), &RuleConfig::default());
        assert!(hits.is_empty());
    }

    #[test]
    fn high_load_skips_on_none() {
        let snap = base_snapshot(); // load1 == None
        let hits = HighLoad.check(&snap, &empty_history(), &RuleConfig::default());
        assert!(hits.is_empty());
    }

    #[test]
    fn fd_exhaustion_fires() {
        let mut snap = base_snapshot();
        snap.open_fds = Some(900);
        snap.fd_limit = Some(1000); // 0.9 > 0.8
        let hits = FdExhaustion.check(&snap, &empty_history(), &RuleConfig::default());
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].rule_id, "fd_exhaustion");
        assert_eq!(hits[0].severity, Severity::High);
    }

    #[test]
    fn fd_exhaustion_silent_when_low() {
        let mut snap = base_snapshot();
        snap.open_fds = Some(100);
        snap.fd_limit = Some(1000); // 0.1
        let hits = FdExhaustion.check(&snap, &empty_history(), &RuleConfig::default());
        assert!(hits.is_empty());
    }

    #[test]
    fn fd_exhaustion_skips_on_none() {
        let mut snap = base_snapshot();
        snap.open_fds = Some(900);
        snap.fd_limit = None;
        assert!(FdExhaustion
            .check(&snap, &empty_history(), &RuleConfig::default())
            .is_empty());
        let mut snap2 = base_snapshot();
        snap2.open_fds = None;
        snap2.fd_limit = Some(1000);
        assert!(FdExhaustion
            .check(&snap2, &empty_history(), &RuleConfig::default())
            .is_empty());
    }

    #[test]
    fn resource_rules_skip_on_resourceless_snapshot() {
        // A snapshot with no resource signals must trip none of them.
        let snap = base_snapshot();
        let hits = run_rules(&catalog(), &snap, &empty_history(), &AdvisorConfig::default());
        for id in [
            "disk_space_low",
            "high_cpu",
            "high_load",
            "fd_exhaustion",
            "system_memory_low",
        ] {
            assert!(find(&hits, id).is_none(), "{id} fired on resourceless snapshot");
        }
    }
}

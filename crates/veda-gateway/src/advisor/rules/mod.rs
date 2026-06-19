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

// ---------------------------------------------------------------------
// Synthetic rule — alerts when an in-process self-check probe has failed
// `consecutive_failures` times in a row. Reads the per-probe summaries
// the engine records onto `snap.synthetic_summaries`; emits one hit per
// failing probe, scoped `Scope::Synthetic(probe_id)` so the EventStore
// dedups one event per probe. Skips cleanly when there are no summaries.
// ---------------------------------------------------------------------

/// A synthetic self-check probe failing repeatedly.
struct SyntheticFailing;
impl Rule for SyntheticFailing {
    fn id(&self) -> &'static str {
        "synthetic_failing"
    }
    fn default_severity(&self) -> Severity {
        Severity::High
    }
    fn check(&self, snap: &OpsSnapshot, _h: &SnapshotHistory, cfg: &RuleConfig) -> Vec<RuleHit> {
        let k = threshold_u64(cfg, "consecutive_failures", 3).max(1) as u32;
        let mut hits = Vec::new();
        for s in &snap.synthetic_summaries {
            if s.consecutive_failures < k {
                continue;
            }
            let uptime = s
                .uptime_pct
                .map(|p| format!("{p:.0}%"))
                .unwrap_or_else(|| "n/a".into());
            hits.push(RuleHit {
                rule_id: self.id().into(),
                scope: Scope::Synthetic(s.probe_id.clone()),
                severity: self.default_severity(),
                title: format!("Synthetic probe '{}' failing", s.name),
                detail: format!(
                    "Probe '{}' has failed {} time(s) in a row (threshold {}); uptime over the retained window {}.",
                    s.name, s.consecutive_failures, k, uptime
                ),
                recommendation: format!(
                    "Investigate the path probe '{}' exercises — the same in-process call is failing repeatedly.",
                    s.name
                ),
                runbook: Some(format!("GET /v1/admin/synthetics/{}/results", s.probe_id)),
                remediation: RemediationAction::None,
            });
        }
        hits
    }
}

// ---------------------------------------------------------------------
// Index-advisor rules — iterate `snap.index_health` (the recall-vs-latency
// pairing assembled in Phase 6a) and emit one hit per collection scoped
// `Scope::Index(collection)`. Both skip cleanly when `index_health` is
// empty or the needed signal (`recall` / `p95_ms`) is `None`. Recall is a
// **proxy** measurement (sampled queries + capped brute force), so the
// recommendations are directional and advisory only.
// ---------------------------------------------------------------------

/// A collection's proxy recall@k has fallen below the target floor — the
/// ANN index is dropping true neighbours. Advisory only.
struct IndexRecallDegraded;
impl Rule for IndexRecallDegraded {
    fn id(&self) -> &'static str {
        "index_recall_degraded"
    }
    fn default_severity(&self) -> Severity {
        Severity::High
    }
    fn check(&self, snap: &OpsSnapshot, _h: &SnapshotHistory, cfg: &RuleConfig) -> Vec<RuleHit> {
        let min = threshold_f64(cfg, "recall_min", 0.90);
        let mut hits = Vec::new();
        for ih in &snap.index_health {
            // Skip cleanly when recall is unmeasured.
            let Some(recall) = ih.recall else {
                continue;
            };
            if recall < min {
                hits.push(RuleHit {
                    rule_id: self.id().into(),
                    scope: Scope::Index(ih.collection.clone()),
                    severity: self.default_severity(),
                    title: format!("Index recall degraded on '{}'", ih.collection),
                    detail: format!(
                        "Proxy recall@k for '{}' ({} index) is {:.2}, below the {:.2} target. The ANN index is missing true nearest neighbours, so search quality is degraded. Recall is a proxy measurement (sampled queries + capped brute force).",
                        ih.collection, ih.index_type, recall, min
                    ),
                    recommendation: format!(
                        "Raise recall for '{}': increase efSearch (and efConstruction for a denser graph), or POST /v1/collections/{}/reindex with a higher-recall config. Current index params may be unknown, so tune directionally and re-probe.",
                        ih.collection, ih.collection
                    ),
                    runbook: Some(format!(
                        "GET /v1/admin/index-advisor; POST /v1/collections/{}/reindex",
                        ih.collection
                    )),
                    remediation: RemediationAction::None,
                });
            }
        }
        hits
    }
}

/// A collection is spending more latency than its (already adequate)
/// recall needs — the index is over-tuned for recall it already has.
/// Advisory only; only fires when recall is known to be good.
struct IndexLatencyHighForRecall;
impl Rule for IndexLatencyHighForRecall {
    fn id(&self) -> &'static str {
        "index_latency_high_for_recall"
    }
    fn default_severity(&self) -> Severity {
        Severity::Medium
    }
    fn check(&self, snap: &OpsSnapshot, _h: &SnapshotHistory, cfg: &RuleConfig) -> Vec<RuleHit> {
        let latency_limit = threshold_f64(cfg, "p95_ms", 50.0);
        let recall_ok = threshold_f64(cfg, "recall_ok", 0.95);
        let mut hits = Vec::new();
        for ih in &snap.index_health {
            // Need BOTH signals: a measured p95 over budget AND recall
            // already adequate (otherwise the recall rule owns it). Skip
            // cleanly when either is `None`.
            let (Some(p95), Some(recall)) = (ih.p95_ms, ih.recall) else {
                continue;
            };
            if p95 > latency_limit && recall >= recall_ok {
                hits.push(RuleHit {
                    rule_id: self.id().into(),
                    scope: Scope::Index(ih.collection.clone()),
                    severity: self.default_severity(),
                    title: format!("Index over-spending latency on '{}'", ih.collection),
                    detail: format!(
                        "'{}' ({} index) has p95 {:.0} ms (threshold {:.0} ms) while proxy recall is already {:.2} (>= {:.2}). It is paying latency for recall it does not need. Recall is a proxy measurement.",
                        ih.collection, ih.index_type, p95, latency_limit, recall, recall_ok
                    ),
                    recommendation: format!(
                        "Trade some recall headroom for latency on '{}': lower efSearch, or switch index type — HNSW→IvfPq/DiskAnn at scale, or RaBitQ to cut memory. POST /v1/collections/{}/reindex to apply, then re-probe.",
                        ih.collection, ih.collection
                    ),
                    runbook: Some(format!(
                        "GET /v1/admin/index-advisor; POST /v1/collections/{}/reindex",
                        ih.collection
                    )),
                    remediation: RemediationAction::None,
                });
            }
        }
        hits
    }
}

// ---------------------------------------------------------------------
// Capacity-forecast rules — read the predictive ETAs the engine attaches
// to `snap.forecasts` (least-squares trend over the snapshot history, with
// a warm-up + min-R² guard so they never fire on a flat/noisy series).
// Each fires only when a forecast's ETA falls inside the rule's window;
// all skip cleanly when there is no matching forecast (or its `eta_ms` is
// `None`). Forecasts are directional projections — the recommendations are
// advisory and the detail always surfaces the ETA + R² confidence.
// ---------------------------------------------------------------------

/// Convert an `eta_ms` epoch + `now` into "in ~N day(s)/hour(s)" plus the
/// fractional days remaining (for the threshold comparison).
fn eta_human_and_days(eta_ms: i64, now_ms: i64) -> (String, f64) {
    let dt_ms = (eta_ms - now_ms).max(0) as f64;
    let days = dt_ms / 86_400_000.0;
    let human = if days >= 1.0 {
        format!("~{days:.1} day(s)")
    } else {
        let hours = dt_ms / 3_600_000.0;
        format!("~{hours:.1} hour(s)")
    };
    (human, days)
}

/// Disk projected to reach the near-full mark within `days` (default 7).
struct DiskFullForecast;
impl Rule for DiskFullForecast {
    fn id(&self) -> &'static str {
        "disk_full_forecast"
    }
    fn default_severity(&self) -> Severity {
        Severity::High
    }
    fn check(&self, snap: &OpsSnapshot, _h: &SnapshotHistory, cfg: &RuleConfig) -> Vec<RuleHit> {
        let days_limit = threshold_f64(cfg, "days", 7.0);
        for f in &snap.forecasts {
            if f.metric != "disk_used_pct" {
                continue;
            }
            let Some(eta_ms) = f.eta_ms else { continue };
            let (human, days) = eta_human_and_days(eta_ms, snap.taken_at_ms);
            if days > days_limit {
                continue;
            }
            return vec![RuleHit {
                rule_id: self.id().into(),
                scope: Scope::Global,
                severity: self.default_severity(),
                title: "Disk projected to fill".into(),
                detail: format!(
                    "Disk usage is trending up ({:.4} %/s) and is forecast to reach {:.0}% in {} (currently {:.0}%; threshold {:.0} day(s); fit confidence R²={:.2}).",
                    f.slope_per_sec, f.threshold, human, f.current, days_limit, f.r2
                ),
                recommendation: "Free disk space (delete old data/logs), run compaction to reclaim space, or scale out / add storage before the disk fills.".into(),
                runbook: Some("GET /v1/admin/forecast; GET /v1/admin/resources".into()),
                remediation: RemediationAction::None,
            }];
        }
        Vec::new()
    }
}

/// Available memory projected to exhaust within `days` (default 7).
struct MemoryGrowthForecast;
impl Rule for MemoryGrowthForecast {
    fn id(&self) -> &'static str {
        "memory_growth_forecast"
    }
    fn default_severity(&self) -> Severity {
        Severity::Medium
    }
    fn check(&self, snap: &OpsSnapshot, _h: &SnapshotHistory, cfg: &RuleConfig) -> Vec<RuleHit> {
        let days_limit = threshold_f64(cfg, "days", 7.0);
        for f in &snap.forecasts {
            if f.metric != "mem_available_bytes" {
                continue;
            }
            let Some(eta_ms) = f.eta_ms else { continue };
            let (human, days) = eta_human_and_days(eta_ms, snap.taken_at_ms);
            if days > days_limit {
                continue;
            }
            return vec![RuleHit {
                rule_id: self.id().into(),
                scope: Scope::Global,
                severity: self.default_severity(),
                title: "Memory projected to exhaust".into(),
                detail: format!(
                    "Available system memory is trending down and is forecast to reach 0 in {} (currently {:.1} GiB; threshold {:.0} day(s); fit confidence R²={:.2}).",
                    human, f.current / 1.073_741_824e9, days_limit, f.r2
                ),
                recommendation: "Reduce in-memory footprint (quantization/use_log_primary), stop competing processes, or add RAM / scale out before memory is exhausted.".into(),
                runbook: Some("GET /v1/admin/forecast; GET /v1/admin/resources".into()),
                remediation: RemediationAction::None,
            }];
        }
        Vec::new()
    }
}

/// A collection's proxy recall projected to fall below the target within
/// `days` (default 7). Advisory — recall is a proxy measurement.
struct RecallDecayForecast;
impl Rule for RecallDecayForecast {
    fn id(&self) -> &'static str {
        "recall_decay_forecast"
    }
    fn default_severity(&self) -> Severity {
        Severity::Medium
    }
    fn check(&self, snap: &OpsSnapshot, _h: &SnapshotHistory, cfg: &RuleConfig) -> Vec<RuleHit> {
        let days_limit = threshold_f64(cfg, "days", 7.0);
        let mut hits = Vec::new();
        for f in &snap.forecasts {
            if f.metric != "recall" {
                continue;
            }
            let Some(eta_ms) = f.eta_ms else { continue };
            let (human, days) = eta_human_and_days(eta_ms, snap.taken_at_ms);
            if days > days_limit {
                continue;
            }
            let coll = f.scope.clone().unwrap_or_else(|| "?".into());
            hits.push(RuleHit {
                rule_id: self.id().into(),
                scope: Scope::Index(coll.clone()),
                severity: self.default_severity(),
                title: format!("Recall projected to decay on '{coll}'"),
                detail: format!(
                    "Proxy recall for '{}' is trending down and is forecast to fall below the {:.2} target in {} (currently {:.2}; threshold {:.0} day(s); fit confidence R²={:.2}). Recall is a proxy measurement.",
                    coll, f.threshold, human, f.current, days_limit, f.r2
                ),
                recommendation: format!(
                    "Pre-empt recall loss on '{coll}': raise efSearch or reindex with a higher-recall config, and re-probe. Investigate data drift / growing unindexed segments."
                ),
                runbook: Some(format!(
                    "GET /v1/admin/forecast; POST /v1/collections/{coll}/reindex"
                )),
                remediation: RemediationAction::None,
            });
        }
        hits
    }
}

// ---------------------------------------------------------------------
// Change-point rule — reads the sustained level/trend shifts the engine
// attaches to `snap.change_points` (mean/slope split-test over the
// snapshot history, warm-up + min-magnitude guarded so it never fires on a
// flat/noisy/short series). Skips cleanly when there are no change points.
// Advisory: a regime shift points at a recent deploy/config/load change.
// ---------------------------------------------------------------------

/// A sustained level or trend shift detected in a metric series.
struct ChangePointDetected;
impl Rule for ChangePointDetected {
    fn id(&self) -> &'static str {
        "change_point_detected"
    }
    fn default_severity(&self) -> Severity {
        Severity::Medium
    }
    fn check(&self, snap: &OpsSnapshot, _h: &SnapshotHistory, cfg: &RuleConfig) -> Vec<RuleHit> {
        use crate::advisor::changepoint::ChangeKind;
        // Optional extra floor on the normalized magnitude (the engine
        // already applied the detector's own guard).
        let min_magnitude = threshold_f64(cfg, "min_magnitude", 0.0);
        let mut hits = Vec::new();
        for cp in &snap.change_points {
            if cp.magnitude < min_magnitude {
                continue;
            }
            let kind = match cp.kind {
                ChangeKind::Level => "level",
                ChangeKind::Trend => "trend",
            };
            // Per-collection recall is index-scoped; everything else global.
            let scope = match (&cp.metric[..], &cp.scope) {
                ("recall", Some(coll)) => Scope::Index(coll.clone()),
                _ => Scope::Global,
            };
            let target = cp
                .scope
                .as_ref()
                .map(|s| format!("{} for '{}'", cp.metric, s))
                .unwrap_or_else(|| cp.metric.clone());
            hits.push(RuleHit {
                rule_id: self.id().into(),
                scope,
                severity: self.default_severity(),
                title: format!("Change point detected in {}", cp.metric),
                detail: format!(
                    "A sustained {kind} shift in {target}: {:.4} → {:.4} (normalized magnitude {:.2}, confidence {:.2}).",
                    cp.before, cp.after, cp.magnitude, cp.confidence
                ),
                recommendation: "A sustained level/trend shift, not a momentary spike — investigate a recent deploy, config, or load change around the shift time.".into(),
                runbook: Some("GET /v1/admin/change-points".into()),
                remediation: RemediationAction::None,
            });
        }
        hits
    }
}

/// Number of most-recent faithfulness samples that form one side of the
/// [`QualityDrift`] two-window comparison. The engine attaches twice this
/// many samples (baseline + recent) to `snap.faithfulness_recent`.
pub const QUALITY_DRIFT_WINDOW: usize = 10;

/// LLM spend over the snapshot window exceeds a configured USD budget.
///
/// Reads the cumulative `veda_llm_cost_usd_total` counter (micro-USD; see
/// [`veda_core::metrics::cost_micro_usd`]). Like [`HighErrorRate`], it uses
/// the history's per-second rate between the previous and current snapshot,
/// then scales it back to the window's wall-clock span to get the dollars
/// spent in this window. Fires when that exceeds `usd_per_window`.
struct LlmCostBudget;
impl Rule for LlmCostBudget {
    fn id(&self) -> &'static str {
        "llm_cost_budget"
    }
    fn default_severity(&self) -> Severity {
        Severity::Medium
    }
    fn check(&self, snap: &OpsSnapshot, history: &SnapshotHistory, cfg: &RuleConfig) -> Vec<RuleHit> {
        // Default budget: 1.0 USD per snapshot window. Operators retune via
        // RuleConfig thresholds `{"usd_per_window": x}`.
        let budget_usd = threshold_f64(cfg, "usd_per_window", 1.0);
        // Per-second micro-USD rate; `None` on cold start / missing counter /
        // counter reset → skip cleanly.
        let Some(micro_per_sec) = history.rate("veda_llm_cost_usd_total", snap) else {
            return Vec::new();
        };
        // Wall-clock span of this window (prev → now), in seconds.
        let window_secs = match history.latest() {
            Some(prev) => ((snap.taken_at_ms - prev.taken_at_ms) as f64 / 1000.0).max(0.0),
            None => return Vec::new(),
        };
        if window_secs <= 0.0 {
            return Vec::new();
        }
        // micro-USD over the window → USD.
        let window_usd = micro_per_sec * window_secs / 1_000_000.0;
        if window_usd > budget_usd {
            return vec![RuleHit {
                rule_id: self.id().into(),
                scope: Scope::Global,
                severity: self.default_severity(),
                title: "LLM cost over budget".into(),
                detail: format!(
                    "LLM spend was ${:.4} over the last {:.0}s (budget ${:.4} per window).",
                    window_usd, window_secs, budget_usd
                ),
                recommendation: "Review agent usage and model choice: enable the semantic response cache, cap max steps, or route to a cheaper model. Inspect per-model spend via veda_llm_cost_usd_total.".into(),
                runbook: Some("GET /v1/admin/metrics".into()),
                remediation: RemediationAction::None,
            }];
        }
        Vec::new()
    }
}

/// Sustained DOWNWARD shift in recent grounding faithfulness vs an earlier
/// baseline — an online quality-drift signal for the agent's answers.
///
/// Uses the `snap.faithfulness_recent` series (oldest-first) the engine
/// attaches from the `FaithfulnessResults` store. A simple two-window mean
/// comparison: the earlier half is the baseline, the latest
/// [`QUALITY_DRIFT_WINDOW`] samples are "recent". Fires when the recent mean
/// drops below the baseline mean by at least `min_drop` (absolute, on the
/// 0..1 faithfulness scale). Skips cleanly on too few samples (warm-up) or
/// when no faithfulness was produced (no embedder/LLM → empty series).
struct QualityDrift;
impl Rule for QualityDrift {
    fn id(&self) -> &'static str {
        "quality_drift"
    }
    fn default_severity(&self) -> Severity {
        Severity::Medium
    }
    fn check(&self, snap: &OpsSnapshot, _h: &SnapshotHistory, cfg: &RuleConfig) -> Vec<RuleHit> {
        // Minimum absolute drop in mean faithfulness to flag (default 0.15).
        let min_drop = threshold_f64(cfg, "min_drop", 0.15);
        let series = &snap.faithfulness_recent;
        let recent_n = QUALITY_DRIFT_WINDOW.min(series.len());
        // Warm-up: need a full recent window AND at least one baseline sample.
        if series.len() < QUALITY_DRIFT_WINDOW + 1 || recent_n == 0 {
            return Vec::new();
        }
        let split = series.len() - recent_n;
        let baseline = &series[..split];
        let recent = &series[split..];
        let mean = |xs: &[f64]| -> f64 { xs.iter().sum::<f64>() / xs.len() as f64 };
        let base_mean = mean(baseline);
        let recent_mean = mean(recent);
        let drop = base_mean - recent_mean;
        if drop >= min_drop {
            return vec![RuleHit {
                rule_id: self.id().into(),
                scope: Scope::Global,
                severity: self.default_severity(),
                title: "Answer quality drift".into(),
                detail: format!(
                    "Mean grounding faithfulness fell from {:.2} (baseline, {} samples) to {:.2} (recent {} samples), a drop of {:.2} (threshold {:.2}).",
                    base_mean, baseline.len(), recent_mean, recent.len(), drop, min_drop
                ),
                recommendation: "Investigate retrieval quality and prompt/model changes: check recent index/recall health, embedding-service status, and whether a model or instruction change preceded the drop.".into(),
                runbook: Some("GET /v1/admin/metrics; GET /v1/admin/change-points".into()),
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
        Box::new(DiskSpaceLow),
        Box::new(HighCpu),
        Box::new(HighLoad),
        Box::new(FdExhaustion),
        Box::new(SystemMemoryLow),
        Box::new(SloBurnRateFast),
        Box::new(SloBurnRateSlow),
        Box::new(SloBudgetExhausted),
        Box::new(SyntheticFailing),
        Box::new(IndexRecallDegraded),
        Box::new(IndexLatencyHighForRecall),
        Box::new(DiskFullForecast),
        Box::new(MemoryGrowthForecast),
        Box::new(RecallDecayForecast),
        Box::new(ChangePointDetected),
        Box::new(LlmCostBudget),
        Box::new(QualityDrift),
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
            synthetic_summaries: Vec::new(),
            index_health: Vec::new(),
            forecasts: Vec::new(),
            change_points: Vec::new(),
            faithfulness_recent: Vec::new(),
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
    fn llm_cost_budget_fires_over_budget() {
        // prev: 0 micro-USD @ t=0; now: 2_000_000 micro-USD (=$2) @ t=10s.
        // window spend = $2 > default $1 budget → fire.
        let mut prev = base_snapshot();
        prev.taken_at_ms = 0;
        prev.counters.insert("veda_llm_cost_usd_total".into(), 0.0);

        let mut now = base_snapshot();
        now.taken_at_ms = 10_000;
        now.counters.insert("veda_llm_cost_usd_total".into(), 2_000_000.0);

        let mut hist = SnapshotHistory::new(DEFAULT_HISTORY_CAP);
        hist.push(prev);

        let hits = LlmCostBudget.check(&now, &hist, &RuleConfig::default());
        assert_eq!(hits.len(), 1, "should fire over budget");
        assert_eq!(hits[0].rule_id, "llm_cost_budget");
        assert_eq!(hits[0].severity, Severity::Medium);
    }

    #[test]
    fn llm_cost_budget_silent_when_under_budget() {
        // $0.10 over the window < $1 budget → no fire.
        let mut prev = base_snapshot();
        prev.taken_at_ms = 0;
        prev.counters.insert("veda_llm_cost_usd_total".into(), 0.0);
        let mut now = base_snapshot();
        now.taken_at_ms = 10_000;
        now.counters.insert("veda_llm_cost_usd_total".into(), 100_000.0); // $0.10
        let mut hist = SnapshotHistory::new(DEFAULT_HISTORY_CAP);
        hist.push(prev);
        assert!(LlmCostBudget.check(&now, &hist, &RuleConfig::default()).is_empty());
    }

    #[test]
    fn llm_cost_budget_skips_without_counter_or_history() {
        // No cost counter at all → skip (even with a previous snapshot).
        let mut prev = base_snapshot();
        prev.taken_at_ms = 0;
        let mut now = base_snapshot();
        now.taken_at_ms = 10_000;
        let mut hist = SnapshotHistory::new(DEFAULT_HISTORY_CAP);
        hist.push(prev);
        assert!(LlmCostBudget.check(&now, &hist, &RuleConfig::default()).is_empty());

        // Counter present but no history (cold start) → skip.
        let mut now2 = base_snapshot();
        now2.counters.insert("veda_llm_cost_usd_total".into(), 5_000_000.0);
        assert!(LlmCostBudget.check(&now2, &empty_history(), &RuleConfig::default()).is_empty());
    }

    #[test]
    fn quality_drift_fires_on_sustained_drop() {
        // 11 baseline @ ~0.9 then 10 recent @ ~0.5 → drop ~0.4 > 0.15.
        let mut snap = base_snapshot();
        let mut series: Vec<f64> = vec![0.9; QUALITY_DRIFT_WINDOW + 1];
        series.extend(std::iter::repeat(0.5).take(QUALITY_DRIFT_WINDOW));
        snap.faithfulness_recent = series;
        let hits = QualityDrift.check(&snap, &empty_history(), &RuleConfig::default());
        assert_eq!(hits.len(), 1, "should fire on a sustained quality drop");
        assert_eq!(hits[0].rule_id, "quality_drift");
        assert_eq!(hits[0].severity, Severity::Medium);
    }

    #[test]
    fn quality_drift_silent_on_stable_series() {
        let mut snap = base_snapshot();
        snap.faithfulness_recent = vec![0.85; QUALITY_DRIFT_WINDOW * 2 + 1];
        assert!(QualityDrift.check(&snap, &empty_history(), &RuleConfig::default()).is_empty());
    }

    #[test]
    fn quality_drift_warmup_suppresses_short_series() {
        let mut snap = base_snapshot();
        // Fewer than QUALITY_DRIFT_WINDOW + 1 samples → warm-up, no fire,
        // even though the few present samples drop sharply.
        snap.faithfulness_recent = vec![0.9, 0.9, 0.1, 0.1];
        assert!(QualityDrift.check(&snap, &empty_history(), &RuleConfig::default()).is_empty());
        // Empty series (no faithfulness produced) → skip.
        let snap2 = base_snapshot();
        assert!(QualityDrift.check(&snap2, &empty_history(), &RuleConfig::default()).is_empty());
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

    // --- synthetic rule ------------------------------------------------

    fn summary(probe_id: &str, consecutive_failures: u32) -> crate::advisor::synthetic::SyntheticSummary {
        crate::advisor::synthetic::SyntheticSummary {
            probe_id: probe_id.into(),
            name: format!("{probe_id} probe"),
            consecutive_failures,
            last_ok: Some(consecutive_failures == 0),
            last_latency_ms: Some(5.0),
            uptime_pct: Some(50.0),
            last_ts_ms: Some(1000),
        }
    }

    #[test]
    fn synthetic_failing_fires_at_or_above_threshold() {
        let mut snap = base_snapshot();
        snap.synthetic_summaries = vec![summary("health", 3)]; // == default K=3
        let hits = SyntheticFailing.check(&snap, &empty_history(), &RuleConfig::default());
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].rule_id, "synthetic_failing");
        assert_eq!(hits[0].scope, Scope::Synthetic("health".into()));
        assert_eq!(hits[0].severity, Severity::High);
        assert!(!hits[0].recommendation.is_empty());
    }

    #[test]
    fn synthetic_failing_silent_below_threshold_and_when_healthy() {
        let mut snap = base_snapshot();
        snap.synthetic_summaries = vec![summary("health", 2), summary("search", 0)];
        let hits = SyntheticFailing.check(&snap, &empty_history(), &RuleConfig::default());
        assert!(hits.is_empty());
    }

    #[test]
    fn synthetic_failing_silent_when_no_summaries() {
        let snap = base_snapshot(); // synthetic_summaries empty
        let hits = SyntheticFailing.check(&snap, &empty_history(), &RuleConfig::default());
        assert!(hits.is_empty());
    }

    #[test]
    fn synthetic_failing_honors_configured_threshold() {
        let mut snap = base_snapshot();
        snap.synthetic_summaries = vec![summary("health", 5)];
        // Raise K to 10 → no hit.
        let cfg = RuleConfig {
            thresholds: serde_json::json!({ "consecutive_failures": 10 }),
            ..Default::default()
        };
        assert!(SyntheticFailing.check(&snap, &empty_history(), &cfg).is_empty());
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

    // --- index-advisor rules -------------------------------------------

    fn index_health(
        collection: &str,
        p95_ms: Option<f64>,
        recall: Option<f64>,
    ) -> crate::advisor::snapshot::IndexSnapshot {
        crate::advisor::snapshot::IndexSnapshot {
            collection: collection.into(),
            index_type: "hnsw".into(),
            p95_ms,
            recall,
            params: crate::advisor::recall_probe::IndexParams::default(),
        }
    }

    #[test]
    fn index_recall_degraded_fires_below_target() {
        let mut snap = base_snapshot();
        snap.index_health = vec![index_health("docs", Some(20.0), Some(0.7))];
        let hits = IndexRecallDegraded.check(&snap, &empty_history(), &RuleConfig::default());
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].rule_id, "index_recall_degraded");
        assert_eq!(hits[0].scope, Scope::Index("docs".into()));
        assert_eq!(hits[0].severity, Severity::High);
        assert!(hits[0].detail.to_lowercase().contains("proxy"));
        assert!(!hits[0].recommendation.is_empty());
        assert_eq!(hits[0].remediation, RemediationAction::None);
    }

    #[test]
    fn index_recall_degraded_silent_when_adequate() {
        let mut snap = base_snapshot();
        snap.index_health = vec![index_health("docs", Some(20.0), Some(0.95))];
        let hits = IndexRecallDegraded.check(&snap, &empty_history(), &RuleConfig::default());
        assert!(hits.is_empty());
    }

    #[test]
    fn index_recall_degraded_skips_on_empty_or_none() {
        // Empty index_health → nothing.
        let snap = base_snapshot();
        assert!(IndexRecallDegraded
            .check(&snap, &empty_history(), &RuleConfig::default())
            .is_empty());
        // recall == None → skip cleanly (cannot judge).
        let mut snap2 = base_snapshot();
        snap2.index_health = vec![index_health("docs", Some(20.0), None)];
        assert!(IndexRecallDegraded
            .check(&snap2, &empty_history(), &RuleConfig::default())
            .is_empty());
    }

    #[test]
    fn index_latency_high_for_recall_fires_when_recall_already_good() {
        let mut snap = base_snapshot();
        // p95 120ms > 50ms AND recall 0.98 >= 0.95 → over-spending.
        snap.index_health = vec![index_health("docs", Some(120.0), Some(0.98))];
        let hits =
            IndexLatencyHighForRecall.check(&snap, &empty_history(), &RuleConfig::default());
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].rule_id, "index_latency_high_for_recall");
        assert_eq!(hits[0].scope, Scope::Index("docs".into()));
        assert_eq!(hits[0].severity, Severity::Medium);
        assert!(!hits[0].recommendation.is_empty());
    }

    #[test]
    fn index_latency_high_for_recall_silent_when_recall_low() {
        // High p95 but recall NOT adequate → that's the recall rule's job.
        let mut snap = base_snapshot();
        snap.index_health = vec![index_health("docs", Some(120.0), Some(0.7))];
        let hits =
            IndexLatencyHighForRecall.check(&snap, &empty_history(), &RuleConfig::default());
        assert!(hits.is_empty());
    }

    #[test]
    fn index_latency_high_for_recall_silent_when_latency_low() {
        let mut snap = base_snapshot();
        snap.index_health = vec![index_health("docs", Some(10.0), Some(0.98))];
        let hits =
            IndexLatencyHighForRecall.check(&snap, &empty_history(), &RuleConfig::default());
        assert!(hits.is_empty());
    }

    #[test]
    fn index_latency_high_for_recall_skips_on_empty_or_none() {
        let snap = base_snapshot();
        assert!(IndexLatencyHighForRecall
            .check(&snap, &empty_history(), &RuleConfig::default())
            .is_empty());
        // p95 present but recall None → skip (cannot tell it's over-spending).
        let mut snap2 = base_snapshot();
        snap2.index_health = vec![index_health("docs", Some(120.0), None)];
        assert!(IndexLatencyHighForRecall
            .check(&snap2, &empty_history(), &RuleConfig::default())
            .is_empty());
        // recall present but p95 None → skip.
        let mut snap3 = base_snapshot();
        snap3.index_health = vec![index_health("docs", None, Some(0.98))];
        assert!(IndexLatencyHighForRecall
            .check(&snap3, &empty_history(), &RuleConfig::default())
            .is_empty());
    }

    // --- capacity-forecast rules ---------------------------------------

    /// A forecast for `metric`/`scope` with an ETA `days_out` days after
    /// the snapshot's `taken_at_ms`. `days_out = None` → no ETA (flat/away).
    fn forecast(
        snap: &OpsSnapshot,
        metric: &str,
        scope: Option<&str>,
        current: f64,
        threshold: f64,
        days_out: Option<f64>,
    ) -> crate::advisor::forecast::Forecast {
        crate::advisor::forecast::Forecast {
            metric: metric.into(),
            scope: scope.map(|s| s.to_owned()),
            current,
            slope_per_sec: if current < threshold { 1.0 } else { -1.0 },
            r2: 0.97,
            threshold,
            eta_ms: days_out.map(|d| snap.taken_at_ms + (d * 86_400_000.0) as i64),
        }
    }

    #[test]
    fn disk_full_forecast_fires_when_eta_within_window() {
        let mut snap = base_snapshot();
        snap.forecasts = vec![forecast(&snap, "disk_used_pct", None, 80.0, 95.0, Some(3.0))];
        let hits = DiskFullForecast.check(&snap, &empty_history(), &RuleConfig::default());
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].rule_id, "disk_full_forecast");
        assert_eq!(hits[0].scope, Scope::Global);
        assert_eq!(hits[0].severity, Severity::High);
        assert!(hits[0].detail.to_lowercase().contains("forecast"));
        assert!(hits[0].detail.contains("day"));
        assert!(!hits[0].recommendation.is_empty());
    }

    #[test]
    fn disk_full_forecast_silent_when_eta_beyond_window() {
        let mut snap = base_snapshot();
        // ETA 30 days out, default window 7 → no fire.
        snap.forecasts = vec![forecast(&snap, "disk_used_pct", None, 80.0, 95.0, Some(30.0))];
        let hits = DiskFullForecast.check(&snap, &empty_history(), &RuleConfig::default());
        assert!(hits.is_empty());
    }

    #[test]
    fn disk_full_forecast_silent_when_no_eta_or_no_forecast() {
        // No forecast at all.
        let snap = base_snapshot();
        assert!(DiskFullForecast
            .check(&snap, &empty_history(), &RuleConfig::default())
            .is_empty());
        // Forecast present but flat (eta None) → no fire.
        let mut snap2 = base_snapshot();
        snap2.forecasts = vec![forecast(&snap2, "disk_used_pct", None, 80.0, 95.0, None)];
        assert!(DiskFullForecast
            .check(&snap2, &empty_history(), &RuleConfig::default())
            .is_empty());
    }

    #[test]
    fn memory_growth_forecast_fires_within_window() {
        let mut snap = base_snapshot();
        snap.forecasts = vec![forecast(
            &snap,
            "mem_available_bytes",
            None,
            2.0 * 1.073_741_824e9,
            0.0,
            Some(2.0),
        )];
        let hits = MemoryGrowthForecast.check(&snap, &empty_history(), &RuleConfig::default());
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].rule_id, "memory_growth_forecast");
        assert_eq!(hits[0].severity, Severity::Medium);
        assert!(hits[0].detail.to_lowercase().contains("forecast"));
    }

    #[test]
    fn recall_decay_forecast_fires_for_downward_collection() {
        let mut snap = base_snapshot();
        snap.forecasts = vec![forecast(&snap, "recall", Some("docs"), 0.93, 0.90, Some(4.0))];
        let hits = RecallDecayForecast.check(&snap, &empty_history(), &RuleConfig::default());
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].rule_id, "recall_decay_forecast");
        assert_eq!(hits[0].scope, Scope::Index("docs".into()));
        assert_eq!(hits[0].severity, Severity::Medium);
        assert!(hits[0].detail.to_lowercase().contains("proxy"));
    }

    #[test]
    fn forecast_rules_silent_on_forecastless_snapshot() {
        let snap = base_snapshot();
        let hits = run_rules(&catalog(), &snap, &empty_history(), &AdvisorConfig::default());
        for id in [
            "disk_full_forecast",
            "memory_growth_forecast",
            "recall_decay_forecast",
        ] {
            assert!(find(&hits, id).is_none(), "{id} fired without a forecast");
        }
    }

    // --- change-point rule --------------------------------------------

    fn change_point(
        metric: &str,
        scope: Option<&str>,
        kind: crate::advisor::changepoint::ChangeKind,
    ) -> crate::advisor::changepoint::ChangePoint {
        crate::advisor::changepoint::ChangePoint {
            metric: metric.into(),
            scope: scope.map(|s| s.to_owned()),
            kind,
            at_ms: 1_700_000_000_000,
            before: 40.0,
            after: 80.0,
            magnitude: 3.0,
            confidence: 0.6,
        }
    }

    #[test]
    fn change_point_rule_fires_for_global_level_shift() {
        use crate::advisor::changepoint::ChangeKind;
        let mut snap = base_snapshot();
        snap.change_points = vec![change_point("disk_used_pct", None, ChangeKind::Level)];
        let hits = ChangePointDetected.check(&snap, &empty_history(), &RuleConfig::default());
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].rule_id, "change_point_detected");
        assert_eq!(hits[0].scope, Scope::Global);
        assert_eq!(hits[0].severity, Severity::Medium);
        assert!(hits[0].detail.to_lowercase().contains("level"));
        assert!(hits[0].detail.contains("40") && hits[0].detail.contains("80"));
    }

    #[test]
    fn change_point_rule_scopes_recall_to_index() {
        use crate::advisor::changepoint::ChangeKind;
        let mut snap = base_snapshot();
        snap.change_points = vec![change_point("recall", Some("docs"), ChangeKind::Trend)];
        let hits = ChangePointDetected.check(&snap, &empty_history(), &RuleConfig::default());
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].scope, Scope::Index("docs".into()));
        assert!(hits[0].detail.to_lowercase().contains("trend"));
    }

    #[test]
    fn change_point_rule_silent_when_empty() {
        let snap = base_snapshot();
        let hits = run_rules(&catalog(), &snap, &empty_history(), &AdvisorConfig::default());
        assert!(find(&hits, "change_point_detected").is_none());
    }
}

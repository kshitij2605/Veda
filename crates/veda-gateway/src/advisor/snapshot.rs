//! Point-in-time ops snapshot + a bounded history ring buffer.
//!
//! See module docs in `mod.rs`. The shapes here are consumed by
//! later advisor tasks (rules / anomaly detectors), so the public
//! field names and signatures are part of the advisor's internal
//! contract.

use std::collections::HashMap;
use std::sync::atomic::Ordering;

use serde::Serialize;

use crate::rest::state::AppState;
use crate::tenant;

/// How far back the query window looks, in seconds (5 minutes).
const QUERY_WINDOW_SECS: u64 = 300;
/// How far back the log-count window looks, in seconds (5 minutes).
const LOG_WINDOW_SECS: u64 = 300;
/// Cap on records pulled from the log buffer when counting by level.
const LOG_SCAN_CAP: usize = 5000;
/// A query slower than this many milliseconds is counted as "slow".
const SLOW_QUERY_MS: u64 = 1000;
/// Prometheus counters/gauges captured into every snapshot.
const TRACKED_COUNTERS: &[&str] = &[
    "veda_requests_total",
    "veda_errors_total",
    "veda_records_total",
    "veda_segments_total",
];
/// Default ring-buffer capacity (~240 snapshots; e.g. 20 minutes at
/// one snapshot / 5s, or 4 hours at one / minute).
pub const DEFAULT_HISTORY_CAP: usize = 240;

/// Global rollup across all collections.
#[derive(Debug, Clone, Serialize)]
pub struct GlobalSnapshot {
    pub active_collections: usize,
    pub total_records: u64,
}

/// Per-collection storage picture, gathered from `segment_managers`.
#[derive(Debug, Clone, Serialize)]
pub struct CollectionSnapshot {
    /// User-facing name (tenant prefix stripped, matching how
    /// metrics label collections).
    pub name: String,
    pub shard_count: usize,
    /// Sealed + growing segments across all shards.
    pub total_segments: usize,
    pub sealed_segments: usize,
    /// Records still in the (unsealed) growing segments.
    pub growing_records: u64,
    pub total_records: u64,
    pub compaction_in_progress: bool,
}

/// A node as reported by the same source `/v1/admin/nodes` uses.
#[derive(Debug, Clone, Serialize)]
pub struct NodeSnapshot {
    pub id: String,
    pub role: String,
    pub status: String,
    pub shard_count: u32,
}

/// External-dependency health, mirroring the `/v1/admin/health`
/// probe output strings ("healthy" | "unavailable" | "unconfigured").
#[derive(Debug, Clone, Serialize)]
pub struct HealthSnapshot {
    pub minio: String,
    pub nats: String,
    pub embedding: String,
    pub multimodal_embedding: String,
}

/// Recent query-latency rollup over a fixed window.
#[derive(Debug, Clone, Serialize)]
pub struct QuerySnapshot {
    pub window_secs: u64,
    pub count: u64,
    pub p50_ms: f64,
    pub p95_ms: f64,
    pub slow_count: u64,
}

/// A full point-in-time snapshot of cluster ops state.
#[derive(Debug, Clone, Serialize)]
pub struct OpsSnapshot {
    pub taken_at_ms: i64,
    pub global: GlobalSnapshot,
    pub collections: Vec<CollectionSnapshot>,
    pub nodes: Vec<NodeSnapshot>,
    pub replica_safe_ts: Option<u128>,
    pub health: HealthSnapshot,
    pub query: QuerySnapshot,
    /// Process resident set size in bytes (Linux); `None` elsewhere.
    pub rss_bytes: Option<u64>,
    /// Total bytes of the filesystem backing the data dir (`statvfs`);
    /// `None` on non-Linux or any error.
    pub disk_total_bytes: Option<u64>,
    /// Free bytes available on the data-dir filesystem; `None` on
    /// non-Linux or any error.
    pub disk_free_bytes: Option<u64>,
    /// Used percentage of the data-dir filesystem `(total-free)/total*100`;
    /// `None` when totals are unavailable.
    pub disk_used_pct: Option<f64>,
    /// Cumulative CPU jiffies across all fields of the aggregate `cpu`
    /// line in `/proc/stat`; `None` on non-Linux or parse error.
    pub cpu_total_jiffies: Option<u64>,
    /// Busy CPU jiffies = total minus (idle + iowait); pair with the
    /// previous snapshot via [`cpu_pct`]. `None` on non-Linux/error.
    pub cpu_busy_jiffies: Option<u64>,
    /// Number of logical CPUs (`available_parallelism`), default 1.
    pub num_cpus: usize,
    /// 1/5/15-minute load averages from `/proc/loadavg`; `None` on
    /// non-Linux or parse error.
    pub load1: Option<f64>,
    pub load5: Option<f64>,
    pub load15: Option<f64>,
    /// Count of entries in `/proc/self/fd` (open file descriptors);
    /// `None` on non-Linux or any error.
    pub open_fds: Option<u64>,
    /// Soft `RLIMIT_NOFILE` (max open files); `None` on non-Linux/error.
    pub fd_limit: Option<u64>,
    /// Total system memory in bytes from `/proc/meminfo`; `None` on
    /// non-Linux or parse error.
    pub mem_total_bytes: Option<u64>,
    /// Available system memory in bytes from `/proc/meminfo`; `None`
    /// on non-Linux or parse error.
    pub mem_available_bytes: Option<u64>,
    /// Totals for a small set of named Prometheus counters/gauges,
    /// summed across all label sets.
    pub counters: HashMap<String, f64>,
    /// Number of ERROR log records in the in-process buffer within the
    /// last `LOG_WINDOW_SECS`.
    pub error_log_count: u64,
    /// Number of WARN+ (WARN or ERROR) log records within the last
    /// `LOG_WINDOW_SECS`.
    pub warn_log_count: u64,
}

/// Collect a live snapshot from `state`. `now_ms` is injected so
/// callers/tests control the timestamp; the query window is measured
/// relative to wall-clock (the query log records wall-clock ms).
pub async fn collect(state: &AppState, now_ms: i64) -> OpsSnapshot {
    let inner = &state.inner;

    // --- collections (from segment_managers) -------------------------
    let compaction_flags: HashMap<String, bool> = {
        let locks = inner.compaction_locks.read().await;
        locks
            .iter()
            .map(|(k, v)| (k.clone(), v.load(Ordering::Acquire)))
            .collect()
    };

    let mut collections: Vec<CollectionSnapshot> = Vec::new();
    let mut total_records: u64 = 0;
    {
        let segs = inner.segment_managers.read().await;
        for (scoped_name, shard_map) in segs.iter() {
            let mut total_segments = 0usize;
            let mut sealed_segments = 0usize;
            let mut growing_records = 0u64;
            let mut coll_records = 0u64;
            for sm in shard_map.values() {
                let sealed = sm.sealed_segments().len();
                sealed_segments += sealed;
                // sealed segments + the single growing segment per shard.
                total_segments += sealed + 1;
                growing_records += sm.growing_segment().count() as u64;
                coll_records += sm.total_records();
            }
            total_records += coll_records;
            collections.push(CollectionSnapshot {
                name: tenant::display_name(scoped_name).to_owned(),
                shard_count: shard_map.len(),
                total_segments,
                sealed_segments,
                growing_records,
                total_records: coll_records,
                compaction_in_progress: compaction_flags
                    .get(scoped_name)
                    .copied()
                    .unwrap_or(false),
            });
        }
    }
    collections.sort_by(|a, b| a.name.cmp(&b.name));
    let active_collections = collections.len();

    // --- nodes (same source as /v1/admin/nodes) ----------------------
    let nodes = collect_nodes(state).await;

    // --- replica safe ts ---------------------------------------------
    let replica_safe_ts = {
        let ts = *inner.replica_safe_ts.read().await;
        // Sentinel 0 means "no gating active"; report None so later
        // lag math doesn't treat 0 as a real watermark.
        if ts == 0 {
            None
        } else {
            Some(ts)
        }
    };

    // --- health probes (reuse admin probe logic) ---------------------
    let checks = crate::rest::admin::run_health_checks().await;
    let health = HealthSnapshot {
        minio: checks
            .get("minio")
            .cloned()
            .unwrap_or_else(|| "unknown".to_owned()),
        nats: checks
            .get("nats")
            .cloned()
            .unwrap_or_else(|| "unknown".to_owned()),
        embedding: checks
            .get("embedding")
            .cloned()
            .unwrap_or_else(|| "unknown".to_owned()),
        multimodal_embedding: checks
            .get("multimodal_embedding")
            .cloned()
            .unwrap_or_else(|| "unknown".to_owned()),
    };

    // --- query window rollup (reuse query-log reader) ----------------
    let query = collect_query(state).await;

    // --- process RSS --------------------------------------------------
    let rss_bytes = read_rss_bytes();

    // --- host/process resource signals (best-effort, never panic) -----
    let data_dir_str = inner.data_dir.to_string_lossy().into_owned();
    let (disk_total_bytes, disk_free_bytes) = match statvfs_usage(&data_dir_str) {
        Some((t, f)) => (Some(t), Some(f)),
        None => (None, None),
    };
    let disk_used_pct = match (disk_total_bytes, disk_free_bytes) {
        (Some(t), Some(f)) if t > 0 => Some((t.saturating_sub(f)) as f64 / t as f64 * 100.0),
        _ => None,
    };
    let (cpu_total_jiffies, cpu_busy_jiffies) =
        match std::fs::read_to_string("/proc/stat").ok().and_then(|s| parse_proc_stat_cpu(&s)) {
            Some((total, busy)) => (Some(total), Some(busy)),
            None => (None, None),
        };
    let num_cpus = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let (load1, load5, load15) =
        match std::fs::read_to_string("/proc/loadavg").ok().and_then(|s| parse_loadavg(&s)) {
            Some((a, b, c)) => (Some(a), Some(b), Some(c)),
            None => (None, None, None),
        };
    let open_fds = read_open_fds();
    let fd_limit = read_fd_limit();
    let (mem_total_bytes, mem_available_bytes) =
        match std::fs::read_to_string("/proc/meminfo").ok().map(|s| parse_meminfo(&s)) {
            Some((t, a)) => (t, a),
            None => (None, None),
        };

    // --- prometheus counters -----------------------------------------
    let mut counters = HashMap::new();
    if let Some(handle) = inner.prometheus_handle.as_ref() {
        let text = handle.render();
        for metric in TRACKED_COUNTERS {
            counters.insert((*metric).to_owned(), parse_prom_counter(&text, metric));
        }
    }

    // --- log-content counts (cheap; in-process buffer, no I/O) --------
    let (error_log_count, warn_log_count) =
        count_recent_logs(state.log_buffer(), now_ms, LOG_WINDOW_SECS);

    OpsSnapshot {
        taken_at_ms: now_ms,
        global: GlobalSnapshot {
            active_collections,
            total_records,
        },
        collections,
        nodes,
        replica_safe_ts,
        health,
        query,
        rss_bytes,
        disk_total_bytes,
        disk_free_bytes,
        disk_used_pct,
        cpu_total_jiffies,
        cpu_busy_jiffies,
        num_cpus,
        load1,
        load5,
        load15,
        open_fds,
        fd_limit,
        mem_total_bytes,
        mem_available_bytes,
        counters,
        error_log_count,
        warn_log_count,
    }
}

/// Count ERROR and WARN+ log records in `buffer` whose `ts_ms` falls
/// within the last `window_secs` of `now_ms`. Returns
/// `(error_count, warn_plus_count)` where the WARN+ count includes
/// ERRORs (it's WARN-or-higher). Cheap: scans at most [`LOG_SCAN_CAP`]
/// recent WARN+ records under a brief buffer lock, no I/O.
pub fn count_recent_logs(
    buffer: &crate::logs::LogBuffer,
    now_ms: i64,
    window_secs: u64,
) -> (u64, u64) {
    let cutoff = now_ms - (window_secs as i64) * 1000;
    // A single WARN+ scan yields both counts (ERROR is a subset of WARN+).
    let recs = buffer.recent(LOG_SCAN_CAP, Some("WARN"), None);
    let mut errors = 0u64;
    let mut warn_plus = 0u64;
    for r in &recs {
        if r.ts_ms < cutoff {
            continue;
        }
        warn_plus += 1;
        if r.level.eq_ignore_ascii_case("ERROR") {
            errors += 1;
        }
    }
    (errors, warn_plus)
}

/// Gather the node list. In Raft mode, read the membership snapshot
/// (same source `/v1/admin/nodes` uses); otherwise report the single
/// local node. We compute only the fields the advisor needs
/// (`id`, `role`, `status`, `shard_count`).
async fn collect_nodes(state: &AppState) -> Vec<NodeSnapshot> {
    let inner = &state.inner;
    let shard_count: u32 = {
        let segs = inner.segment_managers.read().await;
        segs.values().map(|m| m.len() as u32).sum()
    };

    let raft_view = {
        let guard = inner.raft.read().await;
        guard.as_ref().map(|c| c.membership_snapshot())
    };

    if let Some(mem) = raft_view {
        let self_nid: u64 = std::env::var("VEDA_NODE_ID")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1);
        return mem
            .voters
            .iter()
            .copied()
            .map(|nid| {
                let is_self = nid == self_nid;
                let is_leader = mem.current_leader == Some(nid);
                NodeSnapshot {
                    id: format!("node-{nid}"),
                    role: if is_leader { "leader" } else { "follower" }.to_owned(),
                    status: if is_self { "healthy" } else { "unknown" }.to_owned(),
                    shard_count: if is_self { shard_count } else { 0 },
                }
            })
            .collect();
    }

    let self_id = std::env::var("VEDA_NODE_ID")
        .ok()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "node-0".to_owned());
    vec![NodeSnapshot {
        id: self_id,
        role: "leader".to_owned(),
        status: "healthy".to_owned(),
        shard_count,
    }]
}

/// Roll up recent query-log entries across all collections over a
/// fixed window. Reuses the per-collection `_query_log.jsonl` reader.
async fn collect_query(state: &AppState) -> QuerySnapshot {
    let inner = &state.inner;
    let scoped_names: Vec<String> = {
        let segs = inner.segment_managers.read().await;
        segs.keys().cloned().collect()
    };
    let data_dir = inner.data_dir.clone();

    let mut took: Vec<u64> = Vec::new();
    let mut slow_count = 0u64;
    for scoped in &scoped_names {
        let entries =
            crate::rest::query_log::read_window_entries(&data_dir, scoped, QUERY_WINDOW_SECS);
        for e in entries {
            if e.took_ms > SLOW_QUERY_MS {
                slow_count += 1;
            }
            took.push(e.took_ms);
        }
    }

    took.sort_unstable();
    let count = took.len() as u64;
    QuerySnapshot {
        window_secs: QUERY_WINDOW_SECS,
        count,
        p50_ms: percentile(&took, 0.50),
        p95_ms: percentile(&took, 0.95),
        slow_count,
    }
}

/// Nearest-rank percentile over a pre-sorted slice. Returns 0.0 for
/// an empty slice. `q` is in [0, 1].
fn percentile(sorted: &[u64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted[0] as f64;
    }
    let rank = (q * (sorted.len() as f64 - 1.0)).round() as usize;
    sorted[rank.min(sorted.len() - 1)] as f64
}

/// Read process resident set size from `/proc/self/status` (`VmRSS`).
/// Returns `None` on non-Linux platforms or any read/parse error.
fn read_rss_bytes() -> Option<u64> {
    let text = std::fs::read_to_string("/proc/self/status").ok()?;
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("VmRSS:") {
            // Format: "VmRSS:\t   12345 kB"
            let kb: u64 = rest.split_whitespace().next()?.parse().ok()?;
            return Some(kb * 1024);
        }
    }
    None
}

/// Filesystem usage for `path` via `statvfs(2)`. Returns
/// `(total_bytes, free_bytes)` using the available-to-caller block
/// count for free space (`f_bavail`). `None` on non-Linux or any
/// FFI error. Never panics; the FFI call is read-only.
#[cfg(target_os = "linux")]
fn statvfs_usage(path: &str) -> Option<(u64, u64)> {
    let c_path = std::ffi::CString::new(path).ok()?;
    // SAFETY: `stat` is a valid, sized-for-the-platform buffer; the
    // path is a valid NUL-terminated C string. `statvfs` only reads.
    let mut stat: libc::statvfs = unsafe { std::mem::zeroed() };
    let rc = unsafe { libc::statvfs(c_path.as_ptr(), &mut stat) };
    if rc != 0 {
        return None;
    }
    // Block size: prefer fragment size (`f_frsize`); fall back to
    // `f_bsize` when zero.
    let bsize = if stat.f_frsize != 0 {
        stat.f_frsize
    } else {
        stat.f_bsize
    } as u64;
    let total = (stat.f_blocks as u64).checked_mul(bsize)?;
    let free = (stat.f_bavail as u64).checked_mul(bsize)?;
    Some((total, free))
}

#[cfg(not(target_os = "linux"))]
fn statvfs_usage(_path: &str) -> Option<(u64, u64)> {
    None
}

/// Count open file descriptors via the entries in `/proc/self/fd`.
/// `None` on non-Linux or any read error. Never panics.
fn read_open_fds() -> Option<u64> {
    let dir = std::fs::read_dir("/proc/self/fd").ok()?;
    Some(dir.filter(|e| e.is_ok()).count() as u64)
}

/// Soft `RLIMIT_NOFILE` (max open files). Parses the
/// `/proc/self/limits` table. `None` on non-Linux or parse error.
fn read_fd_limit() -> Option<u64> {
    let text = std::fs::read_to_string("/proc/self/limits").ok()?;
    parse_limits_nofile(&text)
}

/// Parse the aggregate `cpu ` line of `/proc/stat` into
/// `(total_jiffies, busy_jiffies)`. `total` is the sum of all numeric
/// fields; `busy` is `total - (idle + iowait)` where idle is field 4
/// and iowait is field 5 (1-indexed after the `cpu` label). Returns
/// `None` if no `cpu ` line is found or it has fewer than 4 numeric
/// fields (idle is required). Pure — testable without `/proc`.
pub fn parse_proc_stat_cpu(text: &str) -> Option<(u64, u64)> {
    for line in text.lines() {
        // The aggregate line starts with "cpu " (a space, not a digit).
        let rest = match line.strip_prefix("cpu") {
            Some(r) if r.starts_with(' ') || r.starts_with('\t') => r,
            _ => continue,
        };
        let fields: Vec<u64> = rest
            .split_whitespace()
            .map(|t| t.parse::<u64>())
            .collect::<Result<Vec<_>, _>>()
            .ok()?;
        // Need at least user, nice, system, idle.
        if fields.len() < 4 {
            return None;
        }
        let total: u64 = fields.iter().copied().sum();
        let idle = fields[3];
        let iowait = fields.get(4).copied().unwrap_or(0);
        let busy = total.saturating_sub(idle + iowait);
        return Some((total, busy));
    }
    None
}

/// Percentage CPU busy between two snapshots, from their cumulative
/// jiffy counters. `(busy_now - busy_prev) / (total_now - total_prev)
/// * 100`. Returns `None` if either snapshot lacks the counters, the
/// total delta is non-positive, or the busy delta is negative
/// (counter reset). Mirrors [`SnapshotHistory::rate`].
pub fn cpu_pct(prev: &OpsSnapshot, now: &OpsSnapshot) -> Option<f64> {
    let total_prev = prev.cpu_total_jiffies?;
    let total_now = now.cpu_total_jiffies?;
    let busy_prev = prev.cpu_busy_jiffies?;
    let busy_now = now.cpu_busy_jiffies?;
    if total_now <= total_prev {
        return None;
    }
    if busy_now < busy_prev {
        return None;
    }
    let total_delta = (total_now - total_prev) as f64;
    let busy_delta = (busy_now - busy_prev) as f64;
    Some(busy_delta / total_delta * 100.0)
}

/// Parse `/proc/loadavg` into `(load1, load5, load15)`. The line is
/// `"<1m> <5m> <15m> <running>/<total> <lastpid>"`. Returns `None` if
/// the first three tokens can't be parsed as floats. Pure.
pub fn parse_loadavg(text: &str) -> Option<(f64, f64, f64)> {
    let mut it = text.split_whitespace();
    let a = it.next()?.parse::<f64>().ok()?;
    let b = it.next()?.parse::<f64>().ok()?;
    let c = it.next()?.parse::<f64>().ok()?;
    Some((a, b, c))
}

/// Parse `/proc/meminfo` into `(mem_total_bytes, mem_available_bytes)`.
/// Values in the file are in kB and are converted to bytes (×1024).
/// Each element is `None` if its key is absent or unparseable. Pure.
pub fn parse_meminfo(text: &str) -> (Option<u64>, Option<u64>) {
    let mut total = None;
    let mut avail = None;
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("MemTotal:") {
            total = rest.split_whitespace().next().and_then(|t| t.parse::<u64>().ok());
        } else if let Some(rest) = line.strip_prefix("MemAvailable:") {
            avail = rest.split_whitespace().next().and_then(|t| t.parse::<u64>().ok());
        }
    }
    (total.map(|kb| kb * 1024), avail.map(|kb| kb * 1024))
}

/// Parse the soft "Max open files" limit out of a `/proc/self/limits`
/// table. Each row is `Limit  Soft  Hard  Units`; the soft value is
/// the first numeric column after the limit name. An `"unlimited"`
/// soft value yields `None` (no finite cap). Returns `None` if the
/// row is missing. Pure.
pub fn parse_limits_nofile(text: &str) -> Option<u64> {
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("Max open files") {
            // First whitespace-delimited token after the name is the soft limit.
            let soft = rest.split_whitespace().next()?;
            return soft.parse::<u64>().ok();
        }
    }
    None
}

/// Sum the numeric value across all label sets for a single
/// Prometheus metric name from a `/metrics` text exposition.
///
/// Matches sample lines whose series name is exactly `metric`,
/// optionally followed by a `{labels}` block. Skips `# HELP` / `# TYPE`
/// comment lines and any metric with a different name (including the
/// `_bucket` / `_sum` / `_count` histogram companions — we never sum
/// histogram buckets). Unparseable values are ignored. Returns 0.0
/// when the metric is absent.
pub fn parse_prom_counter(text: &str, metric: &str) -> f64 {
    let mut total = 0.0f64;
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some(rest) = line.strip_prefix(metric) else {
            continue;
        };
        // The char immediately after the metric name must be a label
        // brace or whitespace — otherwise `metric` is only a prefix of
        // a longer series name (e.g. `veda_requests_total_bucket`).
        let next = rest.chars().next();
        let value_part = match next {
            Some('{') => match rest.split_once('}') {
                Some((_labels, after)) => after,
                None => continue,
            },
            Some(c) if c.is_whitespace() => rest,
            None => continue,
            _ => continue,
        };
        // value_part is " <value> [timestamp]"; take the first token.
        if let Some(tok) = value_part.split_whitespace().next() {
            if let Ok(v) = tok.parse::<f64>() {
                total += v;
            }
        }
    }
    total
}

/// A bounded ring buffer of [`OpsSnapshot`]s. Newest at the back.
/// When full, pushing drops the oldest. Designed to live behind a
/// lock on `AppState` later (not added in this task).
#[derive(Debug)]
pub struct SnapshotHistory {
    buf: std::collections::VecDeque<OpsSnapshot>,
    cap: usize,
}

impl SnapshotHistory {
    /// Create a history with the given capacity (clamped to >= 1).
    pub fn new(cap: usize) -> Self {
        Self {
            buf: std::collections::VecDeque::with_capacity(cap.max(1)),
            cap: cap.max(1),
        }
    }

    /// Append a snapshot, evicting the oldest if at capacity.
    pub fn push(&mut self, snap: OpsSnapshot) {
        if self.buf.len() == self.cap {
            self.buf.pop_front();
        }
        self.buf.push_back(snap);
    }

    /// Number of snapshots currently held.
    pub fn len(&self) -> usize {
        self.buf.len()
    }

    /// Whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    /// Capacity (max snapshots retained).
    pub fn capacity(&self) -> usize {
        self.cap
    }

    /// Most-recent snapshot.
    pub fn latest(&self) -> Option<&OpsSnapshot> {
        self.buf.back()
    }

    /// Second-most-recent snapshot.
    pub fn previous(&self) -> Option<&OpsSnapshot> {
        let n = self.buf.len();
        if n < 2 {
            None
        } else {
            self.buf.get(n - 2)
        }
    }

    /// Per-second delta of a named counter between the previous
    /// snapshot in the buffer and `snap_now`. Returns `None` when
    /// there is no previous snapshot, the metric is missing in either
    /// snapshot, the time gap is non-positive, or the counter went
    /// backwards (a reset — caller can't interpret a partial rate).
    pub fn rate(&self, metric_name: &str, snap_now: &OpsSnapshot) -> Option<f64> {
        let prev = self.latest()?;
        let prev_v = *prev.counters.get(metric_name)?;
        let now_v = *snap_now.counters.get(metric_name)?;
        let dt_ms = snap_now.taken_at_ms - prev.taken_at_ms;
        if dt_ms <= 0 {
            return None;
        }
        let delta = now_v - prev_v;
        if delta < 0.0 {
            return None;
        }
        Some(delta / (dt_ms as f64 / 1000.0))
    }

    /// Project a derived signal out of each retained snapshot, oldest
    /// first. Used by anomaly math in later tasks.
    pub fn series<F>(&self, f: F) -> Vec<f64>
    where
        F: Fn(&OpsSnapshot) -> f64,
    {
        self.buf.iter().map(|s| f(s)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use veda_core::record::{FieldValue, Record};
    use veda_core::schema::{CollectionSchema, FieldDefinition};
    use veda_core::types::{DistanceMetric, FieldType};
    use veda_core::vector::DenseVector;

    fn test_schema(scoped_name: &str) -> CollectionSchema {
        CollectionSchema::builder(scoped_name)
            .add_field(FieldDefinition::vector("v", 2, DistanceMetric::Cosine))
            .add_field(FieldDefinition::new("label", FieldType::String))
            .shard_count(1)
            .build()
            .unwrap()
    }

    fn test_record(label: &str) -> Record {
        let mut r = Record::new();
        r.set("v", FieldValue::Vector(DenseVector::new(vec![1.0, 0.0])));
        r.set("label", FieldValue::String(label.to_owned()));
        r
    }

    /// Seed a collection named `user_name` (default tenant) with
    /// `n` records inserted into shard 0.
    async fn seed_collection(state: &AppState, user_name: &str, n: usize) {
        let scoped = format!("{}__{}", tenant::DEFAULT_TENANT, user_name);
        let schema = test_schema(&scoped);
        let collection_dir = state.inner.data_dir.join(&scoped);
        std::fs::create_dir_all(&collection_dir).unwrap();
        crate::rest::collections::materialize_collection_locally(state, &schema, &collection_dir)
            .await
            .unwrap();
        let mut managers = state.inner.segment_managers.write().await;
        let shard_map = managers.get_mut(&scoped).unwrap();
        let sm = shard_map.get_mut(&0).unwrap();
        for i in 0..n {
            sm.insert(test_record(&format!("rec-{i}"))).unwrap();
        }
    }

    #[tokio::test]
    async fn collect_seeded_collection() {
        let tmp = tempfile::tempdir().unwrap();
        let state = AppState::new(tmp.path());
        seed_collection(&state, "docs", 5).await;

        let snap = collect(&state, 1_700_000_000_000).await;

        assert_eq!(snap.taken_at_ms, 1_700_000_000_000);
        assert!(snap.global.active_collections >= 1);
        assert_eq!(snap.global.total_records, 5);

        let c = snap
            .collections
            .iter()
            .find(|c| c.name == "docs")
            .expect("docs collection present in snapshot");
        // tenant prefix stripped to the user-facing name.
        assert_eq!(c.name, "docs");
        assert_eq!(c.total_records, 5);
        assert_eq!(c.growing_records, 5);
        assert_eq!(c.shard_count, 1);
        // 0 sealed + 1 growing segment per shard.
        assert_eq!(c.sealed_segments, 0);
        assert_eq!(c.total_segments, 1);
        assert!(!c.compaction_in_progress);

        // single-node defaults.
        assert_eq!(snap.nodes.len(), 1);
        assert_eq!(snap.nodes[0].role, "leader");
        // query window has the right shape with no logged queries.
        assert_eq!(snap.query.window_secs, QUERY_WINDOW_SECS);
        assert_eq!(snap.query.count, 0);
    }

    #[test]
    fn parse_counter_sums_multi_label_samples() {
        let text = "\
# HELP veda_requests_total Total requests.
# TYPE veda_requests_total counter
veda_requests_total{method=\"GET\",route=\"/search\"} 10
veda_requests_total{method=\"POST\",route=\"/insert\"} 32
veda_errors_total{route=\"/search\"} 3
veda_request_duration_seconds_bucket{le=\"0.1\"} 99
veda_records_total 7
";
        assert_eq!(parse_prom_counter(text, "veda_requests_total"), 42.0);
        assert_eq!(parse_prom_counter(text, "veda_errors_total"), 3.0);
        assert_eq!(parse_prom_counter(text, "veda_records_total"), 7.0);
        // histogram buckets must NOT be matched by the duration name.
        assert_eq!(
            parse_prom_counter(text, "veda_request_duration_seconds"),
            0.0
        );
        // absent metric returns 0.0.
        assert_eq!(parse_prom_counter(text, "veda_nope_total"), 0.0);
    }

    fn snap_with_counter(taken_at_ms: i64, metric: &str, value: f64) -> OpsSnapshot {
        let mut counters = HashMap::new();
        counters.insert(metric.to_owned(), value);
        OpsSnapshot {
            taken_at_ms,
            global: GlobalSnapshot {
                active_collections: 0,
                total_records: 0,
            },
            collections: Vec::new(),
            nodes: Vec::new(),
            replica_safe_ts: None,
            health: HealthSnapshot {
                minio: "unconfigured".into(),
                nats: "unconfigured".into(),
                embedding: "unconfigured".into(),
                multimodal_embedding: "unconfigured".into(),
            },
            query: QuerySnapshot {
                window_secs: QUERY_WINDOW_SECS,
                count: 0,
                p50_ms: 0.0,
                p95_ms: 0.0,
                slow_count: 0,
            },
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
            counters,
            error_log_count: 0,
            warn_log_count: 0,
        }
    }

    #[test]
    fn count_recent_logs_windows_by_level() {
        use crate::logs::{LogBuffer, LogRecord};
        let buf = LogBuffer::new(100);
        let push = |level: &str, ts: i64| {
            buf.push(LogRecord {
                ts_ms: ts,
                level: level.to_string(),
                target: "t".into(),
                message: "m".into(),
                fields: serde_json::Map::new(),
                trace_id: None,
                span_id: None,
            });
        };
        let now = 1_000_000i64;
        // window = 10s → cutoff = now - 10000.
        push("ERROR", now - 1_000); // in window
        push("ERROR", now - 9_000); // in window
        push("ERROR", now - 20_000); // OUT of window
        push("WARN", now - 2_000); // in window
        push("WARN", now - 50_000); // OUT of window
        push("INFO", now - 100); // excluded by level entirely
        let (errors, warn_plus) = count_recent_logs(&buf, now, 10);
        // 2 errors in window; WARN+ in window = 2 errors + 1 warn = 3.
        assert_eq!(errors, 2, "error count");
        assert_eq!(warn_plus, 3, "warn+ count (includes errors)");
    }

    #[test]
    fn history_push_latest_previous_and_rate() {
        let mut hist = SnapshotHistory::new(DEFAULT_HISTORY_CAP);
        assert!(hist.latest().is_none());
        assert!(hist.previous().is_none());

        let s1 = snap_with_counter(1000, "veda_requests_total", 100.0);
        hist.push(s1);
        assert_eq!(hist.latest().unwrap().taken_at_ms, 1000);
        assert!(hist.previous().is_none());

        // rate() compares the buffer's latest (1000 @ 100) to the new
        // snapshot (3000 @ 160): delta 60 over 2.0s = 30/s.
        let s2 = snap_with_counter(3000, "veda_requests_total", 160.0);
        let r = hist.rate("veda_requests_total", &s2).unwrap();
        assert!((r - 30.0).abs() < 1e-9, "rate was {r}");

        hist.push(s2);
        assert_eq!(hist.latest().unwrap().taken_at_ms, 3000);
        assert_eq!(hist.previous().unwrap().taken_at_ms, 1000);

        // missing metric -> None.
        let s3 = snap_with_counter(4000, "other_total", 1.0);
        assert!(hist.rate("missing_total", &s3).is_none());
    }

    #[test]
    fn history_series_in_order() {
        let mut hist = SnapshotHistory::new(DEFAULT_HISTORY_CAP);
        for (t, v) in [(10i64, 1.0), (20, 2.0), (30, 3.0)] {
            hist.push(snap_with_counter(t, "veda_requests_total", v));
        }
        let series = hist.series(|s| s.taken_at_ms as f64);
        assert_eq!(series, vec![10.0, 20.0, 30.0]);
        let vals = hist.series(|s| *s.counters.get("veda_requests_total").unwrap());
        assert_eq!(vals, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn history_is_bounded() {
        let mut hist = SnapshotHistory::new(3);
        for t in 0..10i64 {
            hist.push(snap_with_counter(t, "veda_requests_total", t as f64));
        }
        assert_eq!(hist.len(), 3);
        assert_eq!(hist.capacity(), 3);
        // oldest dropped: only 7, 8, 9 remain.
        let series = hist.series(|s| s.taken_at_ms as f64);
        assert_eq!(series, vec![7.0, 8.0, 9.0]);
        assert_eq!(hist.latest().unwrap().taken_at_ms, 9);
        assert_eq!(hist.previous().unwrap().taken_at_ms, 8);
    }

    #[test]
    fn rss_helper_does_not_panic() {
        // On the Linux build host this should be Some(>0); elsewhere
        // None is acceptable. Either way, it must not panic.
        if let Some(rss) = read_rss_bytes() {
            assert!(rss > 0);
        }
    }

    #[test]
    fn parse_proc_stat_cpu_sums_and_subtracts_idle() {
        // user=100 nice=0 system=200 idle=700 iowait=... etc.
        let text = "cpu  100 0 200 700 50 0 10 0 0 0\ncpu0 1 2 3 4\n";
        let (total, busy) = parse_proc_stat_cpu(text).expect("cpu line parsed");
        assert_eq!(total, 100 + 0 + 200 + 700 + 50 + 10);
        // busy = total - (idle 700 + iowait 50).
        assert_eq!(busy, total - (700 + 50));
    }

    #[test]
    fn parse_proc_stat_cpu_malformed_is_none() {
        // No aggregate cpu line at all.
        assert!(parse_proc_stat_cpu("intr 1 2 3\nctxt 99\n").is_none());
        // cpu line with non-numeric fields → None.
        assert!(parse_proc_stat_cpu("cpu  100 0 abc 700\n").is_none());
        // fewer than 4 fields (no idle) → None.
        assert!(parse_proc_stat_cpu("cpu  100 0 200\n").is_none());
    }

    #[test]
    fn parse_loadavg_three_floats() {
        let (a, b, c) = parse_loadavg("0.50 0.40 0.30 1/234 5678").expect("parsed");
        assert!((a - 0.5).abs() < 1e-9);
        assert!((b - 0.4).abs() < 1e-9);
        assert!((c - 0.3).abs() < 1e-9);
        // garbage → None.
        assert!(parse_loadavg("nope").is_none());
        assert!(parse_loadavg("").is_none());
    }

    #[test]
    fn parse_meminfo_converts_kb_to_bytes() {
        let text = "\
MemTotal:       16384000 kB
MemFree:         1000000 kB
MemAvailable:    8192000 kB
Buffers:          100000 kB
";
        let (total, avail) = parse_meminfo(text);
        assert_eq!(total, Some(16_384_000 * 1024));
        assert_eq!(avail, Some(8_192_000 * 1024));
        // missing keys → None each.
        let (t2, a2) = parse_meminfo("Buffers: 100 kB\n");
        assert!(t2.is_none());
        assert!(a2.is_none());
    }

    #[test]
    fn parse_limits_nofile_reads_soft_column() {
        let text = "\
Limit                     Soft Limit           Hard Limit           Units
Max cpu time              unlimited            unlimited            seconds
Max open files            1024                 1048576              files
Max locked memory         65536                65536                bytes
";
        assert_eq!(parse_limits_nofile(text), Some(1024));
        // absent row → None.
        assert!(parse_limits_nofile("Max cpu time unlimited unlimited\n").is_none());
        // unlimited soft → None (not a finite cap).
        assert!(parse_limits_nofile("Max open files            unlimited            unlimited            files\n").is_none());
    }

    fn snap_with_cpu(taken_at_ms: i64, total: Option<u64>, busy: Option<u64>) -> OpsSnapshot {
        let mut s = snap_with_counter(taken_at_ms, "veda_requests_total", 0.0);
        s.cpu_total_jiffies = total;
        s.cpu_busy_jiffies = busy;
        s
    }

    #[test]
    fn cpu_pct_computes_and_guards() {
        // prev total=1000 busy=200; now total=1100 busy=260.
        // delta busy 60 / delta total 100 * 100 = 60%.
        let prev = snap_with_cpu(1000, Some(1000), Some(200));
        let now = snap_with_cpu(2000, Some(1100), Some(260));
        let pct = cpu_pct(&prev, &now).expect("pct");
        assert!((pct - 60.0).abs() < 1e-9, "pct was {pct}");

        // zero total delta → None.
        let same = snap_with_cpu(2000, Some(1100), Some(260));
        assert!(cpu_pct(&now, &same).is_none());

        // negative total delta → None.
        assert!(cpu_pct(&now, &prev).is_none());

        // negative busy delta (busy reset) with positive total → None.
        let reset = snap_with_cpu(3000, Some(1200), Some(100));
        assert!(cpu_pct(&now, &reset).is_none());

        // missing fields → None.
        let missing = snap_with_cpu(4000, None, None);
        assert!(cpu_pct(&now, &missing).is_none());
        assert!(cpu_pct(&missing, &now).is_none());
    }

    #[tokio::test]
    async fn collect_populates_resource_signals() {
        let tmp = tempfile::tempdir().unwrap();
        let state = AppState::new(tmp.path());
        seed_collection(&state, "docs", 1).await;

        let snap = collect(&state, 1_700_000_000_000).await;

        // num_cpus is always >= 1.
        assert!(snap.num_cpus >= 1);

        // On the Linux build host the /proc + statvfs signals should be
        // Some; on other platforms None is tolerated (must not panic).
        if cfg!(target_os = "linux") {
            assert!(snap.disk_total_bytes.is_some(), "disk_total_bytes");
            assert!(snap.disk_free_bytes.is_some(), "disk_free_bytes");
            let pct = snap.disk_used_pct.expect("disk_used_pct");
            assert!((0.0..=100.0).contains(&pct), "disk_used_pct {pct}");
            assert!(snap.load1.is_some(), "load1");
            assert!(snap.open_fds.map(|n| n > 0).unwrap_or(false), "open_fds > 0");
            assert!(snap.mem_total_bytes.is_some(), "mem_total_bytes");
            assert!(snap.cpu_total_jiffies.is_some(), "cpu_total_jiffies");
        }
    }
}

# Veda Node Resource Signals

**Date:** 2026-06-16
**Status:** Shipped on `main` (v0.2.7). Validated end-to-end against a live server.

Node Resource Signals is Veda's in-lane answer to an infrastructure-monitoring platform:
give the Ops Advisor the **host/process resource signals it needs to operate Veda** — disk,
CPU, load, file descriptors, and system memory — so it can warn *before* the disk fills and
writes/seals/compaction start failing. It does NOT try to be Elastic's full infrastructure
platform (metric-collection agents, the inventory/waffle-map Hosts UI, k8s/cloud integrations,
a TSDS engine, cross-node host aggregation) — Veda already ships Prometheus/Mimir/Grafana/OTel
for that, and the rest is outside a vector DB's lane. This extends the existing Ops Advisor
engine (snapshot → rules → anomaly → events → channels → Niriksha); it is not a new product.

Scope is the **local node** (single-node focus). Cross-node host-metric aggregation is deferred.

---

## 1. Architecture

This reuses the Ops Advisor pipeline end-to-end; only the snapshot's inputs, the rule catalog,
and one REST surface grew.

```
  /proc/{stat,loadavg,meminfo,self/fd,self/limits}  +  statvfs(data_dir)   (Linux, best-effort)
        │  OpsSnapshot::collect()  (panic-free, non-blocking; None off-Linux/on error)
        ▼
  OpsSnapshot { disk_*, cpu_*_jiffies, num_cpus, load{1,5,15}, open_fds/fd_limit, mem_* , rss_bytes }
        │   bounded SnapshotHistory ring
        ├── resource rules + anomaly detectors → EventStore → alert channels (webhook/Slack/PD/email)
        └── GET /v1/admin/resources  → Niriksha "Resources" panel (System Health)
```

- All signals live on `OpsSnapshot` (`crates/veda-gateway/src/advisor/snapshot.rs`) alongside the
  pre-existing `rss_bytes`, and follow the same best-effort `Option` pattern: present on Linux,
  `None` off-Linux or on any read error — so a non-Linux build raises **no false alerts**.
- Reads are dependency-free `/proc` parses (like the existing RSS read) plus a single
  `statvfs` for the data dir; `libc` is added only under `cfg(target_os = "linux")`.
- CPU% is a **derived** signal: `/proc/stat` gives cumulative jiffy counters, and
  `cpu_pct(prev, now)` computes utilization from the busy/total delta between two snapshots.

---

## 2. Snapshot signals

| Signal | Source | Field(s) |
|---|---|---|
| Disk (data dir) | `statvfs` | `disk_total_bytes`, `disk_free_bytes`, `disk_used_pct` |
| CPU | `/proc/stat` | `cpu_total_jiffies`, `cpu_busy_jiffies` (+ `cpu_pct(prev, now)` helper) |
| Cores | `available_parallelism` | `num_cpus` |
| Load | `/proc/loadavg` | `load1`, `load5`, `load15` |
| File descriptors | `/proc/self/fd` (count) + `/proc/self/limits` | `open_fds`, `fd_limit` |
| System memory | `/proc/meminfo` | `mem_total_bytes`, `mem_available_bytes` |
| Process RSS | `/proc/self/status` (pre-existing) | `rss_bytes` |

Each parser is a small pure function (unit-tested against sample `/proc` text), and `statvfs`
is a checked FFI call returning `None` on a non-zero return code.

---

## 3. Resource rules

Threshold rules (defaults shown; all tunable per-rule via `/v1/admin/insights/rules`). **Each
skips cleanly when its required signal is `None`**, so they never fire on a resource-less snapshot:

- **`disk_space_low`** (High) — data-dir `disk_used_pct` > **90**. The headline: warns *before*
  the disk fills and writes, seals, and compaction begin to fail. Detail reports used % + free GiB;
  recommendation is free space / run compaction / scale out.
- **`high_cpu`** (Medium) — `cpu_pct(history.previous(), latest)` > **85**. Needs a previous
  snapshot; no previous → no fire.
- **`high_load`** (Medium) — `load1` / `num_cpus` > **2.0** (load per core).
- **`fd_exhaustion`** (High) — `open_fds` / `fd_limit` > **0.8**.
- **`system_memory_low`** (Medium) — `mem_available_bytes` / `mem_total_bytes` < **0.1**.

Anomaly detectors (reuse the existing EWMA/σ machinery; warm-up suppresses until enough samples):

- **`anomaly_cpu`** — CPU% series reconstructed from consecutive `(cpu_busy, cpu_total)` jiffy
  deltas across history (mirrors the error-rate rate-series approach), upper-tail.
- **`anomaly_disk_used`** — `disk_used_pct` trend, upper-tail (catches a rapid fill before it
  crosses the static threshold).

All resource events flow through the same `EventStore` (open / dedup / auto-close) and alert
channels (webhook / Slack / PagerDuty / email) as every other advisor rule, and obey the same
`enabled` / `muted` / `auto_remediate` config.

---

## 4. Resources REST API

`GET /v1/admin/resources` (admin-only) returns the local node's latest resource snapshot:

```json
{
  "disk":   { "total_bytes": 7937040248832, "free_bytes": 2111452635136, "used_pct": 73.4 },
  "cpu_pct": 11.4,
  "load":   { "load1": 8.85, "load5": 12.0, "load15": 15.74 },
  "num_cpus": 72,
  "fds":    { "open": 11, "limit": 1024 },
  "memory": { "total_bytes": 1079228002304, "available_bytes": 1053569273856 },
  "rss_bytes": 46637056
}
```

Values come from the advisor's latest `SnapshotHistory` entry; `cpu_pct` is computed from the
previous→latest pair (so it is `null` until two snapshots exist). When history is empty (the
advisor hasn't ticked yet) it falls back to a fresh `collect()` with `cpu_pct: null`. Every
signal field is nullable and serializes to `null` off-Linux/on error — the endpoint always
returns **200** with the full shape.

---

## 5. Niriksha Resources panel

A **Resources** section on the Niriksha **System Health** page renders the snapshot as small
rows/gauges: disk used % + free GiB (warning-colored past ~85%, danger past ~90%), CPU %,
load 1/5/15 + core count, open fds / limit, system memory available / total, and process RSS.
Null fields render as "—". Loaded on mount with a manual Refresh.

---

## 6. Configuration

- `VEDA_ADVISOR_INTERVAL_SECS` — advisor tick interval (default 30s); drives how often the
  resource signals are sampled and the rules evaluated. (`cpu_pct` and the CPU anomaly need ≥2
  ticks to produce a value.)
- Per-rule tuning via `PUT /v1/admin/insights/rules/{id}` — e.g. lower `disk_space_low`'s
  `disk_used_pct` threshold, mute `high_load`, or enable an anomaly detector.
- No new external dependency: signals are read from `/proc` + `statvfs` on Linux.

---

## 7. Validation (live)

Verified against a running server on the Linux host:
- **`GET /v1/admin/resources`** returned real signals: disk **73.4 % used** (2.1 TB free of
  7.9 TB), CPU **11.4 %** (computed from history), load **8.85 / 12.0 / 15.74** over **72** cores,
  fds **11 / 1024**, system memory ~1 TB available, RSS 46 MB.
- **`disk_space_low` tripped:** lowering its `disk_used_pct` threshold to 50 and forcing a tick
  (`POST /v1/admin/insights/run`) opened **1** High event — *"data-dir disk 73% used (free
  1966.4 GiB; threshold 50%). Writes, seals, and compaction will fail when the disk fills."* —
  with the free-space/compaction/scale-out recommendation.
- **Alerting:** the configured **webhook fired exactly once** (transition-only, no per-tick spam)
  carrying the full event payload (`disk_space_low:global`, severity high, detail, recommendation,
  runbook `GET /v1/admin/metrics`).
- **`high_load` correctly did not fire** — load1 8.85 / 72 cores = 0.12 per core, well under 2.0.

---

## 8. Limitations & follow-ups

- **Linux-only signals.** Off-Linux every field is `null` and the resource rules/anomalies skip —
  by design (no false alerts), but no resource monitoring there either.
- **Single node.** Reports the local node's resources only; cross-node host-metric aggregation is
  deferred (out of a vector DB's lane — use the Prometheus/Mimir/Grafana stack for fleet views).
- **Out of scope, deliberately:** metric-collection agents (node_exporter / Metricbeat / hostmetrics
  receiver), the inventory waffle-map / Hosts UI, k8s/cloud integrations, and a TSDS engine.
- `cpu_pct` and `anomaly_cpu` need ≥2 advisor ticks before they produce a value.

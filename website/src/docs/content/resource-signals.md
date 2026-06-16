# Node Resource Signals

**Node Resource Signals** gives the [Ops Advisor](/docs/ops-advisor) the host and process resource signals it needs to operate Veda — disk, CPU, load, file descriptors, and system memory — so it can warn *before* the disk fills and writes, seals, and compaction start failing.

It is not a full infrastructure-monitoring platform: there are no metric-collection agents, no Hosts inventory UI, no Kubernetes/cloud integrations, and no time-series engine — Veda already ships Prometheus / Mimir / Grafana / OTel for fleet metrics (see [Observability](/docs/observability)), and the rest is outside a vector database's lane. This extends the existing Ops Advisor engine rather than adding a new product, scoped to the **local node**.

## Architecture

Resource collection reuses the Ops Advisor pipeline end to end — only the snapshot's inputs, the rule catalog, and one REST surface grew:

```
  /proc/{stat,loadavg,meminfo,self/fd,self/limits} + statvfs(data_dir)   (Linux, best-effort)
        │  OpsSnapshot::collect()   (panic-free, non-blocking)
        ▼
  OpsSnapshot { disk_*, cpu_*_jiffies, num_cpus, load{1,5,15}, open_fds/fd_limit, mem_*, rss_bytes }
        │  bounded SnapshotHistory ring
        ├── resource rules + anomaly detectors → EventStore → alert channels
        └── GET /v1/admin/resources → Niriksha "Resources" panel
```

Reads are dependency-free `/proc` parses plus a single `statvfs` for the data directory. Every signal follows a best-effort `Option` pattern: present on Linux, `null` off-Linux or on any read error — so a non-Linux build raises **no false alerts**. `libc` is pulled in only under `cfg(target_os = "linux")`. CPU% is **derived**: `/proc/stat` gives cumulative jiffy counters, and utilization is computed from the busy/total delta between two consecutive snapshots.

## Snapshot signals

| Signal | Source | Fields |
|---|---|---|
| Disk (data dir) | `statvfs` | `disk_total_bytes`, `disk_free_bytes`, `disk_used_pct` |
| CPU | `/proc/stat` | `cpu_total_jiffies`, `cpu_busy_jiffies` (+ derived `cpu_pct`) |
| Cores | `available_parallelism` | `num_cpus` |
| Load | `/proc/loadavg` | `load1`, `load5`, `load15` |
| File descriptors | `/proc/self/fd` + `/proc/self/limits` | `open_fds`, `fd_limit` |
| System memory | `/proc/meminfo` | `mem_total_bytes`, `mem_available_bytes` |
| Process RSS | `/proc/self/status` | `rss_bytes` |

## Resource rules

Threshold rules — defaults shown, all tunable per-rule via `/v1/admin/insights/rules`. **Each skips cleanly when its required signal is `null`**, so they never fire on a resource-less snapshot:

| Rule | Severity | Default threshold | Recommendation |
|---|---|---|---|
| `disk_space_low` | High | data-dir `disk_used_pct` > 90 | free space / run compaction / scale out |
| `high_cpu` | Medium | derived `cpu_pct` > 85 (needs a previous tick) | check load / heavy queries |
| `high_load` | Medium | `load1 / num_cpus` > 2.0 (load per core) | check CPU saturation |
| `fd_exhaustion` | High | `open_fds / fd_limit` > 0.8 | raise the fd limit / investigate leaks |
| `system_memory_low` | Medium | `mem_available / mem_total` < 0.1 | scale out / reduce memory pressure |

Anomaly detectors reuse the EWMA/σ machinery (warm-up suppresses firing until enough samples):

| Rule | Severity | Fires when |
|---|---|---|
| `anomaly_cpu` | Med/High | CPU% (reconstructed from jiffy deltas across history) spikes above baseline |
| `anomaly_disk_used` | Med/High | `disk_used_pct` trend rises above baseline — catches a rapid fill *before* it crosses the static threshold |

`disk_space_low` is the headline rule: it warns before the disk fills, so you act before writes, seals, and compaction begin to fail. All resource events flow through the same [`EventStore`](/docs/ops-advisor#events-lifecycle) (open / dedup / auto-close) and [alert channels](/docs/ops-advisor#alerting-channels), and obey the same `enabled` / `muted` / `auto_remediate` config as every other rule.

## Resources REST API

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

Values come from the advisor's latest snapshot-history entry. `cpu_pct` is computed from the previous→latest pair, so it is `null` until two snapshots exist; if the advisor hasn't ticked yet, the endpoint falls back to a fresh collection with `cpu_pct: null`. Every field is nullable and serializes to `null` off-Linux or on a read error — the endpoint always returns **200** with the full shape.

## Niriksha Resources panel

A **Resources** section on the Niriksha **System Health** page renders the snapshot as compact rows and gauges: disk used % + free GiB (warning-colored past ~85%, danger past ~90%), CPU %, load 1/5/15 with core count, open fds / limit, system memory available / total, and process RSS. Null fields render as "—". It loads on mount with a manual refresh.

## Configuration

| Variable / location | Purpose |
|---|---|
| `VEDA_ADVISOR_INTERVAL_SECS` | Advisor tick interval (default 30s) — how often resources are sampled and rules evaluated. `cpu_pct` and `anomaly_cpu` need ≥2 ticks to produce a value |
| `PUT /v1/admin/insights/rules/{id}` | Tune a threshold (e.g. lower `disk_space_low`), mute `high_load`, or enable an anomaly detector |

No new external dependency — signals are read from `/proc` and `statvfs` on Linux.

## Live validation

Verified against a running server on a Linux host:

- `GET /v1/admin/resources` returned real signals: disk **73.4% used** (2.1 TB free of 7.9 TB), CPU **11.4%**, load **8.85 / 12.0 / 15.74** over **72** cores, fds **11 / 1024**, ~1 TB system memory available, RSS 46 MB.
- `disk_space_low` tripped on demand: lowering its threshold to 50% and forcing a tick (`POST /v1/admin/insights/run`) opened **1 High event** — *"data-dir disk 73% used (free 1966.4 GiB; threshold 50%). Writes, seals, and compaction will fail when the disk fills."* — with the free-space/compaction/scale-out recommendation.
- The configured **webhook fired exactly once** (transition-only, no per-tick spam) with the full event payload.
- `high_load` correctly did **not** fire — load1 8.85 / 72 cores = 0.12 per core, well under 2.0.

## Out of scope (deliberately)

- **Cross-node aggregation** — reports the local node only; use the Prometheus / Mimir / Grafana stack for fleet views.
- **Non-Linux signals** — every field is `null` off-Linux and the resource rules skip (no false alerts), but there's no resource monitoring there either.
- **Metric-collection agents** (node_exporter / Metricbeat / hostmetrics receiver), an **inventory / Hosts UI**, **Kubernetes / cloud integrations**, and a **time-series engine** — outside a vector database's lane.

# Observability

Veda is OpenTelemetry-first: metrics, traces, and logs ship through one pipeline, with a pre-provisioned Grafana stack one Compose profile away.

## Start the monitoring stack

```bash
docker compose --profile monitoring up
```

This starts six services alongside Veda:

| Service | Port | Purpose |
|---|---|---|
| OTel Collector | `:4317` (gRPC) | Receives traces, metrics, and logs via OTLP |
| Prometheus | `:9090` | Scrapes Veda's `/v1/admin/metrics` endpoint |
| Mimir | `:9009` | Long-term metrics storage (Prometheus remote-write) |
| Tempo | `:3200` | Distributed trace storage |
| Loki | `:3100` | Log aggregation |
| Grafana | `:3000` | Dashboards and exploration (pre-provisioned datasources and dashboard) |

## OpenTelemetry configuration

Veda exports traces and logs via OTLP when this is set:

| Variable | Default | Description |
|---|---|---|
| `OTEL_EXPORTER_OTLP_ENDPOINT` | *(unset)* | OTLP gRPC endpoint (e.g. `http://otel-collector:4317`). Export is disabled when unset. |
| `OTEL_SERVICE_NAME` | `veda` | Service name attached to all telemetry |

With `--profile monitoring`, these are pre-configured in `docker-compose.yml`.

## Prometheus metrics

Veda exposes Prometheus text-format metrics at `GET /v1/admin/metrics`: request counts and durations, search latency, insert counts, embedding and blob operation durations, segment and compaction gauges, and error totals.

## Health checks

```bash
# Liveness -- always returns 200
curl http://localhost:6340/v1/admin/health

# Readiness -- returns 503 if unhealthy
curl http://localhost:6340/v1/admin/readiness
```

Both endpoints return:

```json
{
  "status": "healthy",
  "version": "0.1.0",
  "checks": {
    "minio": "healthy",
    "nats": "healthy",
    "embedding": "unconfigured",
    "multimodal_embedding": "unconfigured"
  }
}
```

Status is one of `healthy`, `degraded` (non-critical service down), or `unhealthy` (MinIO or NATS down). The readiness endpoint returns HTTP 503 when unhealthy.

## Query explain

Add `"explain": true` to any search request for per-phase timing:

```bash
curl -X POST http://localhost:6340/v1/collections/products/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": {"vector": {"field": "embedding", "values": [0.15, 0.22, ...]}},
    "limit": 10,
    "explain": true
  }'
```

The response includes an `explain` object with `filter_ms`, `ann_search_ms`, `scoring_ms`, `fusion_ms`, and `total_ms`.

## Niriksha dashboard

**Niriksha** is Veda's built-in visualization and exploration dashboard: collection overviews with real-time statistics, interactive vector-space projections, point-and-click search with a filter builder, embedding inspection for any modality, cluster health, query performance analytics, and user/role management. Access it at `http://localhost:5173` when running via Docker Compose.

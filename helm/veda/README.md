# Veda Helm Chart

Deploys [Veda](https://github.com/kshitij2605/Veda) — a multimodal vector
database — to Kubernetes.

## Quick start

```bash
# Lint and render
helm lint helm/veda
helm template my-veda helm/veda

# Install (uses bundled MinIO + NATS by default)
helm install my-veda helm/veda \
    --namespace veda --create-namespace
```

The veda API is available at `Service/my-veda` on port `6340`. Port-forward:

```bash
kubectl -n veda port-forward svc/my-veda 6340:6340
```

## What gets deployed

By default, three workloads:

| Workload | Type | Purpose |
|----------|------|---------|
| `<release>-veda` | StatefulSet (1 replica) | The veda binary, persistent data volume |
| `<release>-minio` | StatefulSet (1 replica) | S3-compatible blob storage. Disable with `--set minio.enabled=false` |
| `<release>-nats` | StatefulSet (1 replica) | Message queue for WAL + streaming ingestion. Disable with `--set nats.enabled=false` |

Both MinIO and NATS are intentionally lightweight single-replica deployments
that mirror what `docker-compose up` provides. For production at scale,
disable them here and bring your own (e.g. AWS S3, Nats Operator HA cluster).

## Common settings

```bash
# Bigger data volume + custom storage class
helm install my-veda helm/veda \
    --set persistence.size=200Gi \
    --set persistence.storageClassName=fast-nvme

# Enable JWT auth
helm install my-veda helm/veda \
    --set auth.enabled=true \
    --set auth.jwtSecret=$(openssl rand -hex 32)

# Use an external S3 instead of the bundled MinIO
helm install my-veda helm/veda \
    --set minio.enabled=false \
    --set env.VEDA_BLOB_BACKEND=s3 \
    --set env.VEDA_BLOB_ENDPOINT=https://s3.amazonaws.com \
    --set env.VEDA_BLOB_BUCKET=my-veda-bucket

# Point at an external NATS cluster
helm install my-veda helm/veda \
    --set nats.enabled=false \
    --set env.VEDA_MQ_ENDPOINT=nats://my-nats.example:4222

# Enable OpenTelemetry export
helm install my-veda helm/veda \
    --set env.OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector.observability:4317
```

## Replicas

The current single-leader coordinator does not support multi-node consensus
(the openraft path is wired but the network transport is a placeholder).
Running `replicaCount > 1` only makes sense as a read-scale-out behind shared
blob storage — write coordination is not yet correct across replicas.

## Out of scope

This chart deliberately does NOT bundle the optional dependencies that
appear under docker-compose profiles:

- Text/multimodal embedding services (HuggingFace TEI, vLLM)
- OCR services (FlashParse, GLM-OCR, VLM)
- Observability stack (OTel Collector, Mimir, Tempo, Loki, Grafana)

Those are large, GPU-hungry services that should be deployed separately
(e.g. via their own Helm charts) and pointed at via the relevant
`env.*_ENDPOINT` values.

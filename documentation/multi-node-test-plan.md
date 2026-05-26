# Multi-Node Cluster Test Plan

Status: **Deferred** -- single-node validation is complete; multi-node is the next milestone.

## Prerequisites

1. **3+ physical or VM nodes** with SSH access between them
2. **Shared MinIO** -- all nodes must point at the same MinIO/S3 endpoint for blob storage
3. **Shared NATS** -- all nodes must connect to the same NATS cluster for messaging
4. **Veda binary** -- the same release build deployed to each node (or a shared NFS mount)
5. **Network** -- nodes must be able to reach each other on the configured Raft and gRPC ports

## Docker Compose Configuration

A `docker-compose.cluster.yml` is needed that starts **3 Veda instances** on different ports with shared infrastructure:

```yaml
# Sketch -- not production-ready
services:
  minio:
    image: minio/minio
    # ... (shared across all nodes)

  nats:
    image: nats:latest
    # ... (shared across all nodes)

  veda-node-0:
    build: .
    ports: ["6340:6340"]
    environment:
      VEDA_COORDINATOR: raft
      VEDA_NODE_ID: "0"
      VEDA_PEERS: "veda-node-0:6340,veda-node-1:6341,veda-node-2:6342"
      VEDA_BLOB_ENDPOINT: http://minio:9000

  veda-node-1:
    build: .
    ports: ["6341:6340"]
    environment:
      VEDA_COORDINATOR: raft
      VEDA_NODE_ID: "1"
      VEDA_PEERS: "veda-node-0:6340,veda-node-1:6341,veda-node-2:6342"
      VEDA_BLOB_ENDPOINT: http://minio:9000

  veda-node-2:
    build: .
    ports: ["6342:6340"]
    environment:
      VEDA_COORDINATOR: raft
      VEDA_NODE_ID: "2"
      VEDA_PEERS: "veda-node-0:6340,veda-node-1:6341,veda-node-2:6342"
      VEDA_BLOB_ENDPOINT: http://minio:9000
```

Each node runs with `VEDA_CLUSTER_MODE=raft` (via `VEDA_COORDINATOR`) and the full peer address list.

## What to Test

### 1. BookKeeper-style quorum writes

Verify `(E=3, Qw=3, Qa=2)` quorum replication across the 3 nodes:
- Insert records via any node and confirm they are durably replicated to all ensemble members.
- Kill one follower mid-write and confirm the write still succeeds (Qw satisfied by remaining nodes).
- Read from any node and confirm Qa=2 acknowledgement semantics.

### 2. FollowerHealthMonitor auto-orchestrator ensemble changes

- Kill one follower node.
- Verify `FollowerHealthMonitor` detects the failure via its edge-triggered probe loop.
- Confirm a single `rebuild_ensemble` call replaces the failed node (not one call per probe interval).
- Restart the killed node and verify it rejoins cleanly.

### 3. WAN log shipping with bounded staleness

- Configure cross-region replication between a "primary" set and a "replica" set.
- Insert records in the primary region.
- Verify the replica region's `safe_ts` advances and that reads ahead of `safe_ts` return `503` with lag information.
- Verify `check_read_ts` gating on the live `/records` and `/search` paths.

### 4. Promote-replica admin endpoint

- Set up a primary + replica topology via `PUT /v1/admin/region_topology`.
- Call `POST /v1/admin/promote-region` to promote the replica.
- Verify the topology epoch is bumped and old primaries are fenced.
- Confirm writes succeed on the newly promoted region and fail on the old primary.

### 5. Linearisability harness (long-duration)

Run the harness against the 3-node cluster with injected faults:

```bash
linearisability_harness --workload-txn-mix --duration 3600s \
  --nodes veda-node-0:6340,veda-node-1:6341,veda-node-2:6342 \
  --fault-mode partition
```

- Inject network partitions, node kills, and slow-link faults during the run.
- Verify the harness passes (no linearisability violations detected).

### 6. Split-brain resilience

- Start the 3-node cluster and insert baseline data.
- Kill one node (minority partition).
- Verify the remaining 2-node majority can still accept writes (write quorum degrades gracefully).
- Verify the isolated node rejects writes (no split-brain).
- Rejoin the killed node and verify it catches up from the log.

## Estimated Effort

**1-2 weeks** broken down as:

| Phase | Estimate | Scope |
|---|---|---|
| Infrastructure setup | 2-3 days | `docker-compose.cluster.yml`, CI runner with 3 containers, networking |
| Test harness integration | 3-4 days | Wire `linearisability_harness` into the cluster, fault injection scripts |
| Individual test scenarios | 2-3 days | Implement and validate scenarios 1-4 and 6 above |
| Long-duration soak + fix | 2-3 days | Run scenario 5 for 1h+, triage and fix any failures |

## Status

- **Single-node validation**: Complete. All Percolator transactions, MVCC, log replication, and reindex pipelines pass on a single node.
- **Multi-node**: Deferred. The Raft coordinator, BookKeeper replication, and WAN shipping code exist but have only been tested in simulated (in-process) multi-node configurations, not across real network boundaries.
- **Next step**: Stand up the 3-node Docker Compose cluster and run the quorum write test (scenario 1) as the first integration milestone.

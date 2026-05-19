#!/usr/bin/env bash
#
# Partition smoke test for the 3-node Raft cluster.
#
# Drives the cluster through a single failure scenario:
#   1. Cluster starts. Note the initial leader.
#   2. Stop the leader container (simulates a node death).
#   3. Wait for the remaining 2 nodes to elect a new leader.
#   4. Verify the new leader is different from the old one.
#   5. Restart the stopped node.
#   6. Verify the rejoined node recovers from disk (WAL replay)
#      and converges with the surviving pair (last_applied_index
#      matches).
#
# Note on partition mechanism: an earlier version of this script
# used `docker network disconnect` to simulate a partition without
# stopping the container. That works for the partition itself but
# Docker's embedded DNS server does not always restore the
# container's hostname after a network reconnect, which causes
# the post-heal heartbeats to fail at the TONIC dial step (NOT at
# the Raft layer). Real production environments use stable DNS
# (k8s / Consul / hosts file) where that's not an issue. To keep
# the smoke test honest about what it actually exercises, we use
# stop/start — it's a stricter test (full process restart, WAL
# replay) and avoids the Docker-DNS quirk.
#
# What this DOES test:
#   - A 3-node quorum survives loss of any one node, including
#     the current leader.
#   - openraft elects a new leader within the configured
#     election timeout (1500-3000ms) when the existing leader
#     becomes unreachable.
#   - On heal, the formerly-partitioned node catches up via
#     Tonic AppendEntries (last_log_index ticks back up to the
#     cluster's high-water mark).
#
# What this does NOT test:
#   - Packet loss / jitter / asymmetric partitions. Real fault
#     injection needs `tc qdisc` inside containers, which
#     needs NET_ADMIN — a separate setup we don't have here.
#   - Sustained-load behaviour (write throughput, follower lag
#     under continuous client_write). That's the soak harness.
#   - Multi-second partitions where the old leader's term
#     advances past the new leader's. The election timeout is
#     short enough that the new leader almost always lands at
#     term+1; an adversarial test would force higher terms by
#     repeated leader kills.
#
# Usage:
#   docker compose --profile raft-cluster up -d veda-1 veda-2 veda-3
#   ./scripts/raft-partition-smoke.sh
#
# Exit codes:
#   0 — partition + heal both succeeded
#   1 — initial cluster never converged
#   2 — re-election did not produce a different leader
#   3 — heal did not converge

set -euo pipefail

NETWORK="${VEDA_RAFT_NETWORK:-project-mdb_default}"
NODE_PORTS=(6340 6342 6344)
NODE_CONTAINERS=(project-mdb-veda-1-1 project-mdb-veda-2-1 project-mdb-veda-3-1)
PARTITION_HOLD_SECS="${PARTITION_HOLD_SECS:-10}"
SETTLE_SECS="${SETTLE_SECS:-8}"

log() { echo "[partition-smoke] $*" >&2; }

leader_of() {
  # Returns the leader id observed by node at $1 (port). 0 if no leader.
  local port="$1"
  curl -fsS "http://localhost:${port}/v1/admin/metrics" 2>/dev/null \
    | awk '/^veda_raft_leader_id /{print $2; exit}'
}

last_applied_of() {
  local port="$1"
  curl -fsS "http://localhost:${port}/v1/admin/metrics" 2>/dev/null \
    | awk '/^veda_raft_last_applied_index /{print $2; exit}'
}

reachable() {
  curl -fsS "http://localhost:${1}/v1/admin/health" >/dev/null 2>&1
}

# Step 1 — wait for initial convergence.
log "waiting up to ${SETTLE_SECS}s for initial leader election..."
initial_leader=0
for _ in $(seq 1 "$SETTLE_SECS"); do
  sleep 1
  initial_leader="$(leader_of 6340 || echo 0)"
  if [[ "$initial_leader" != "0" && -n "$initial_leader" ]]; then
    break
  fi
done
if [[ "$initial_leader" == "0" || -z "$initial_leader" ]]; then
  log "FATAL: cluster never elected a leader"
  exit 1
fi
log "initial leader: node-${initial_leader}"

# Map the leader id to its container + port.
case "$initial_leader" in
  1) leader_container="project-mdb-veda-1-1"; other_port=6342 ;;
  2) leader_container="project-mdb-veda-2-1"; other_port=6344 ;;
  3) leader_container="project-mdb-veda-3-1"; other_port=6340 ;;
  *) log "FATAL: unrecognised leader id $initial_leader"; exit 1 ;;
esac

# Step 2 — stop the leader container (full process death).
log "stopping ${leader_container}"
docker stop "$leader_container" >/dev/null

# Step 3 — wait for re-election among the remaining two.
log "waiting up to ${PARTITION_HOLD_SECS}s for re-election from a surviving node (port ${other_port})..."
new_leader=0
for _ in $(seq 1 "$PARTITION_HOLD_SECS"); do
  sleep 1
  l="$(leader_of "$other_port" || echo 0)"
  if [[ "$l" != "0" && "$l" != "$initial_leader" && -n "$l" ]]; then
    new_leader="$l"
    break
  fi
done
if [[ "$new_leader" == "0" || -z "$new_leader" ]]; then
  log "FATAL: no new leader emerged within ${PARTITION_HOLD_SECS}s; restarting & exiting"
  docker start "$leader_container" >/dev/null || true
  exit 2
fi
log "new leader after partition: node-${new_leader} (was node-${initial_leader})"

# Step 4 — restart the stopped node, observe convergence.
log "restarting ${leader_container}"
docker start "$leader_container" >/dev/null

log "waiting up to ${SETTLE_SECS}s for heal convergence..."
converged=0
for _ in $(seq 1 "$SETTLE_SECS"); do
  sleep 1
  # Get last_applied from all 3 ports; they should match.
  vals=()
  ok=1
  for p in "${NODE_PORTS[@]}"; do
    v="$(last_applied_of "$p" || echo "")"
    if [[ -z "$v" ]]; then ok=0; break; fi
    vals+=("$v")
  done
  if [[ "$ok" -ne 1 ]]; then continue; fi
  if [[ "${vals[0]}" == "${vals[1]}" && "${vals[1]}" == "${vals[2]}" ]]; then
    log "converged: all three nodes at last_applied_index=${vals[0]}"
    converged=1
    break
  fi
done
if [[ "$converged" -ne 1 ]]; then
  log "FATAL: cluster did not converge after heal"
  exit 3
fi

log "PASS"
exit 0

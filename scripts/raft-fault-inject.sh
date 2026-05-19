#!/usr/bin/env bash
#
# Network fault-injection harness for the 3-node Raft cluster.
#
# Uses `tc qdisc` inside each container's network namespace to
# introduce packet loss / latency / asymmetric drops between
# specific node pairs. Requires the cap_add: ["NET_ADMIN"] flag
# on each veda-N service in docker-compose.yml.
#
# Subcommands:
#
#   loss <container> <percent>
#     Drop <percent>% of packets leaving <container>. Use 0 to
#     clear. Example:
#       ./raft-fault-inject.sh loss project-mdb-veda-2-1 10
#
#   delay <container> <ms>
#     Add a fixed latency to all egress from <container>.
#       ./raft-fault-inject.sh delay project-mdb-veda-2-1 200
#
#   partition <container> <peer-ip>
#     Drop ALL traffic from <container> to <peer-ip>. Use
#     `unpartition` to clear. Asymmetric — the peer can still
#     reach this node. Use `docker inspect` to look up peer IPs.
#       PEER_IP=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' project-mdb-veda-1-1)
#       ./raft-fault-inject.sh partition project-mdb-veda-2-1 "$PEER_IP"
#
#   unpartition <container>
#     Clear all qdisc rules on the container. Restores normal
#     traffic.
#
#   smoke
#     Single-pass end-to-end test: introduce 10% packet loss
#     across all 3 nodes, run 50 Noops, clear loss, verify the
#     cluster still has a leader and all 3 nodes' last_applied
#     matches. Documents the floor — Raft survives a moderate
#     packet-loss regime.
#
# What this does NOT do:
#   - Run for hours. Each invocation is a single-pass smoke;
#     real hardening needs continuous mixed-fault runs.
#   - Bandwidth limits / re-ordering. `tc netem` supports both
#     (`rate Nmbit`, `reorder N% N%`); add them when actually
#     debugging a specific issue.
#   - Cross-node coordination. Each fault is applied to one
#     container; complex partition shapes (a-b split, c island)
#     need multiple invocations.

set -euo pipefail

VEDA_CONTAINERS=(project-mdb-veda-1-1 project-mdb-veda-2-1 project-mdb-veda-3-1)
NODE_PORTS=(6340 6342 6344)

log() { echo "[fault-inject] $*" >&2; }

run_tc() {
  # $1 = container, rest = tc args
  local container="$1"; shift
  docker exec "$container" tc "$@"
}

clear_qdisc() {
  local container="$1"
  # `del` may fail if there's no qdisc — that's fine.
  docker exec "$container" tc qdisc del dev eth0 root 2>/dev/null || true
}

cmd_loss() {
  local container="$1" pct="$2"
  clear_qdisc "$container"
  if [[ "$pct" -gt 0 ]]; then
    run_tc "$container" qdisc add dev eth0 root netem loss "${pct}%"
    log "${container}: ${pct}% packet loss on egress"
  else
    log "${container}: cleared loss"
  fi
}

cmd_delay() {
  local container="$1" ms="$2"
  clear_qdisc "$container"
  if [[ "$ms" -gt 0 ]]; then
    run_tc "$container" qdisc add dev eth0 root netem delay "${ms}ms"
    log "${container}: ${ms}ms delay on egress"
  else
    log "${container}: cleared delay"
  fi
}

cmd_partition() {
  # Drop all egress from $container to $peer_ip via a u32 filter.
  local container="$1" peer_ip="$2"
  clear_qdisc "$container"
  run_tc "$container" qdisc add dev eth0 root handle 1: prio
  run_tc "$container" qdisc add dev eth0 parent 1:1 handle 10: netem loss 100%
  run_tc "$container" filter add dev eth0 protocol ip parent 1: prio 1 \
    u32 match ip dst "${peer_ip}/32" flowid 1:1
  log "${container} -> ${peer_ip}: 100% drop (asymmetric partition)"
}

cmd_unpartition() {
  clear_qdisc "$1"
  log "$1: cleared all qdisc rules"
}

# Helper for the smoke test.
leader_of() {
  curl -fsS "http://localhost:${1}/v1/admin/metrics" 2>/dev/null \
    | awk '/^veda_raft_leader_id /{print $2; exit}'
}
last_applied_of() {
  curl -fsS "http://localhost:${1}/v1/admin/metrics" 2>/dev/null \
    | awk '/^veda_raft_last_applied_index /{print $2; exit}'
}

cmd_smoke() {
  log "smoke: introducing 10% packet loss on all 3 containers"
  for c in "${VEDA_CONTAINERS[@]}"; do cmd_loss "$c" 10; done

  log "smoke: finding leader..."
  local leader=0
  for _ in $(seq 1 20); do
    sleep 1
    leader="$(leader_of 6340 || echo 0)"
    [[ "$leader" != "0" && -n "$leader" ]] && break
  done
  if [[ "$leader" == "0" || -z "$leader" ]]; then
    log "FAIL: cluster lost leader under 10% loss"
    for c in "${VEDA_CONTAINERS[@]}"; do clear_qdisc "$c"; done
    exit 1
  fi
  case "$leader" in
    1) leader_port=6340 ;;
    2) leader_port=6342 ;;
    3) leader_port=6344 ;;
  esac
  log "smoke: leader is node-${leader} (port ${leader_port}) under 10% loss"

  log "smoke: writing 50 Noop entries through the leader..."
  local fails=0
  for i in $(seq 1 50); do
    code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 \
      -X POST "http://localhost:${leader_port}/v1/admin/cluster/noop")
    if [[ "$code" != "201" ]]; then
      fails=$((fails + 1))
    fi
  done
  log "smoke: ${fails} of 50 Noops failed under loss (expected non-zero)"

  log "smoke: clearing loss + waiting 10s for convergence..."
  for c in "${VEDA_CONTAINERS[@]}"; do clear_qdisc "$c"; done
  sleep 10

  # Cross-check.
  local vals=()
  for p in "${NODE_PORTS[@]}"; do
    vals+=("$(last_applied_of "$p" || echo 0)")
  done
  if [[ "${vals[0]}" == "${vals[1]}" && "${vals[1]}" == "${vals[2]}" ]]; then
    log "smoke PASS: all three nodes converged at last_applied=${vals[0]} after loss cleared"
    exit 0
  else
    log "smoke FAIL: divergence after clearing loss: ${vals[*]}"
    exit 2
  fi
}

case "${1:-}" in
  loss)        cmd_loss "$2" "$3" ;;
  delay)       cmd_delay "$2" "$3" ;;
  partition)   cmd_partition "$2" "$3" ;;
  unpartition) cmd_unpartition "$2" ;;
  smoke)       cmd_smoke ;;
  *)
    echo "Usage: $0 {loss|delay|partition|unpartition|smoke} ..." >&2
    echo "  loss <container> <pct>"
    echo "  delay <container> <ms>"
    echo "  partition <container> <peer-ip>"
    echo "  unpartition <container>"
    echo "  smoke"
    exit 2
    ;;
esac

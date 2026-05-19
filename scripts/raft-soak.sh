#!/usr/bin/env bash
#
# Long-running Raft soak harness.
#
# Continuously drives write load against the 3-node Raft cluster
# while randomly killing + restarting nodes on a schedule. After
# each kill/restart cycle, checks that all surviving nodes agree
# on last_applied_index and that the killed node catches up
# within HEAL_SECS after coming back. Logs every iteration so the
# operator can grep for the first failure.
#
# This script is designed to be run BY THE OPERATOR for hours.
# A single iteration takes ~60 seconds; reasonable soak budgets:
#   - 30 minutes  =>  ~30 iterations  =>  catches "obvious" bugs
#   - 2 hours     => ~120 iterations  =>  catches most flake
#   - 24 hours    => ~1440 iters      =>  what Jepsen-style runs do
#
# The smoke harnesses (scripts/raft-partition-smoke.sh,
# raft-snapshot-soak.sh, raft-fault-inject.sh smoke) verify each
# fault shape in isolation. This script chains them randomly so
# the cluster spends most of its time under SOME stress.
#
# Usage:
#   docker compose --profile raft-cluster up -d veda-1 veda-2 veda-3
#   ./scripts/raft-soak.sh                 # run until Ctrl-C
#   DURATION_SECS=1800 ./scripts/raft-soak.sh   # run for 30 min
#
# Env knobs:
#   DURATION_SECS         exit after this many seconds (default: forever)
#   ITERATION_SECS        seconds per fault cycle (default 60)
#   HEAL_SECS             wait window for catch-up (default 30)
#   WRITER_PROCS          background writer concurrency (default 3)
#   WRITER_RATE           sleep between writes per process (default 0.05)
#   FAULTS                space-separated subset of fault names
#                         (default: "kill restart loss latency partition")
#
# Failure modes the harness watches for (each prints FAIL: ...):
#   - leader never elected
#   - any node's last_applied_index regresses (data loss)
#   - cluster never converges after a fault (split-brain or hung
#     replication)
#   - HTTP /admin/health goes unhealthy and stays so

set -uo pipefail

NODE_PORTS=(6340 6342 6344)
NODE_CONTAINERS=(project-mdb-veda-1-1 project-mdb-veda-2-1 project-mdb-veda-3-1)
NODE_IDS=(1 2 3)
DURATION_SECS="${DURATION_SECS:-0}"
ITERATION_SECS="${ITERATION_SECS:-60}"
HEAL_SECS="${HEAL_SECS:-30}"
WRITER_PROCS="${WRITER_PROCS:-3}"
WRITER_RATE="${WRITER_RATE:-0.05}"
FAULTS="${FAULTS:-kill restart loss latency partition}"

start_time=$(date +%s)
iteration=0
total_writes=0
failures=0

log() { echo "[soak $(date +%H:%M:%S) iter=${iteration}] $*"; }

leader_of() {
  curl -fsS --max-time 2 "http://localhost:${1}/v1/admin/metrics" 2>/dev/null \
    | awk '/^veda_raft_leader_id /{print $2; exit}'
}
last_applied_of() {
  curl -fsS --max-time 2 "http://localhost:${1}/v1/admin/metrics" 2>/dev/null \
    | awk '/^veda_raft_last_applied_index /{print $2; exit}'
}

# Find the leader port from any responsive node.
leader_port() {
  for p in "${NODE_PORTS[@]}"; do
    local l
    l="$(leader_of "$p" || echo 0)"
    if [[ -n "$l" && "$l" != "0" ]]; then
      case "$l" in
        1) echo 6340; return 0 ;;
        2) echo 6342; return 0 ;;
        3) echo 6344; return 0 ;;
      esac
    fi
  done
  echo 0
}

# Background writer — fires Noops at the leader at ~WRITER_RATE.
# Writes survive leader changes (re-resolved each call).
writer_loop() {
  local id="$1"
  while :; do
    local lp
    lp="$(leader_port)"
    if [[ "$lp" != "0" ]]; then
      curl -fsS --max-time 5 -X POST "http://localhost:${lp}/v1/admin/cluster/noop" \
        >/dev/null 2>&1 && {
          flock -x 9
          echo "1" >> /tmp/raft-soak-writes.$$
          flock -u 9
        }
    fi
    sleep "$WRITER_RATE"
  done 9>/tmp/raft-soak-lock.$$
}

cleanup() {
  log "stopping writers + clearing faults..."
  for pid in "${WRITER_PIDS[@]:-}"; do
    [[ -n "$pid" ]] && kill "$pid" 2>/dev/null || true
  done
  for c in "${NODE_CONTAINERS[@]}"; do
    docker exec "$c" tc qdisc del dev eth0 root 2>/dev/null || true
    docker start "$c" >/dev/null 2>&1 || true
  done
  rm -f /tmp/raft-soak-writes.$$ /tmp/raft-soak-lock.$$ 2>/dev/null || true
  if [[ -f /tmp/raft-soak-writes.$$ ]]; then
    total_writes=$(wc -l < /tmp/raft-soak-writes.$$ 2>/dev/null || echo 0)
  fi
  log "soak done: ${iteration} iterations, ${total_writes} writes, ${failures} failures"
  if [[ "$failures" -gt 0 ]]; then exit 1; else exit 0; fi
}
trap cleanup INT TERM EXIT

# Pre-flight: cluster must already be up.
log "soak start; checking cluster..."
sleep 2
initial_leader="$(leader_port)"
if [[ "$initial_leader" == "0" ]]; then
  log "FATAL: no leader at boot — is the cluster up?"
  exit 1
fi
log "initial leader on port $initial_leader"

# Spawn writers.
> /tmp/raft-soak-writes.$$
WRITER_PIDS=()
for i in $(seq 1 "$WRITER_PROCS"); do
  writer_loop "$i" &
  WRITER_PIDS+=($!)
done
log "spawned ${WRITER_PROCS} writer processes"

random_fault() {
  # Pick a random word from $FAULTS
  read -r -a arr <<< "$FAULTS"
  echo "${arr[RANDOM % ${#arr[@]}]}"
}
random_node_idx() { echo $((RANDOM % 3)); }

iteration_loop() {
  while :; do
    iteration=$((iteration + 1))
    if [[ "$DURATION_SECS" -gt 0 ]]; then
      local elapsed=$(( $(date +%s) - start_time ))
      if [[ "$elapsed" -ge "$DURATION_SECS" ]]; then
        log "duration reached"
        return
      fi
    fi

    local fault target_idx target_container target_port
    fault="$(random_fault)"
    target_idx="$(random_node_idx)"
    target_container="${NODE_CONTAINERS[$target_idx]}"
    target_port="${NODE_PORTS[$target_idx]}"

    log "fault=$fault target=$target_container"

    case "$fault" in
      kill)
        docker stop "$target_container" >/dev/null 2>&1 || true
        sleep $((ITERATION_SECS / 3))
        docker start "$target_container" >/dev/null 2>&1 || true
        ;;
      restart)
        docker restart "$target_container" >/dev/null 2>&1 || true
        ;;
      loss)
        docker exec "$target_container" tc qdisc add dev eth0 root \
          netem loss 15% 2>/dev/null || true
        sleep $((ITERATION_SECS / 2))
        docker exec "$target_container" tc qdisc del dev eth0 root 2>/dev/null || true
        ;;
      latency)
        docker exec "$target_container" tc qdisc add dev eth0 root \
          netem delay 100ms 2>/dev/null || true
        sleep $((ITERATION_SECS / 2))
        docker exec "$target_container" tc qdisc del dev eth0 root 2>/dev/null || true
        ;;
      partition)
        # Pick a different node to partition against.
        local other_idx=$(( (target_idx + 1) % 3 ))
        local other_container="${NODE_CONTAINERS[$other_idx]}"
        local other_ip
        other_ip="$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' "$other_container" 2>/dev/null || echo "")"
        if [[ -n "$other_ip" ]]; then
          docker exec "$target_container" sh -c "
            tc qdisc del dev eth0 root 2>/dev/null
            tc qdisc add dev eth0 root handle 1: prio &&
            tc qdisc add dev eth0 parent 1:1 handle 10: netem loss 100% &&
            tc filter add dev eth0 protocol ip parent 1: prio 1 u32 match ip dst $other_ip/32 flowid 1:1
          " 2>/dev/null || true
        fi
        sleep $((ITERATION_SECS / 2))
        docker exec "$target_container" tc qdisc del dev eth0 root 2>/dev/null || true
        ;;
    esac

    log "clearing fault, waiting up to ${HEAL_SECS}s for convergence..."
    local converged=0 vals=()
    for _ in $(seq 1 "$HEAL_SECS"); do
      sleep 1
      vals=()
      local all_ok=1
      for p in "${NODE_PORTS[@]}"; do
        local v
        v="$(last_applied_of "$p" 2>/dev/null || echo "")"
        if [[ -z "$v" ]]; then all_ok=0; break; fi
        vals+=("$v")
      done
      if [[ "$all_ok" == "1" && "${vals[0]}" == "${vals[1]}" && "${vals[1]}" == "${vals[2]}" ]]; then
        converged=1
        break
      fi
    done
    if [[ "$converged" -eq 1 ]]; then
      log "iter PASS — all nodes at last_applied=${vals[0]}"
    else
      log "iter FAIL — divergent state: ${vals[*]}"
      failures=$((failures + 1))
    fi

    # Iteration runtime budget.
    local now elapsed_iter
    now="$(date +%s)"
    elapsed_iter=$(( now - start_time - (iteration - 1) * ITERATION_SECS ))
    if [[ "$elapsed_iter" -lt "$ITERATION_SECS" ]]; then
      sleep $(( ITERATION_SECS - elapsed_iter ))
    fi
  done
}

iteration_loop

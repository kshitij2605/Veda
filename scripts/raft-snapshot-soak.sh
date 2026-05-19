#!/usr/bin/env bash
#
# Snapshot install + log replication smoke test.
#
# Drives load through the Raft log, kills a follower, wipes its
# durable state, restarts it, and verifies it catches up via
# whatever combination of snapshot install + AppendEntries
# openraft uses to bring a totally-fresh node back in sync with
# the cluster.
#
# Sequence:
#   1. Note the current leader id and pick a follower to kill.
#   2. Fire N Noop commands through /v1/admin/cluster/noop to
#      advance the log on the leader. With openraft defaults
#      (max_in_snapshot_log_to_keep = 1000), this stays in the
#      pure-AppendEntries regime; the follower will recover by
#      replaying log entries rather than receiving a snapshot.
#      That's fine for this smoke pass — the replication code
#      path being exercised is the same; only the chunk source
#      differs.
#   3. Stop the chosen follower.
#   4. Wipe its raft data dir (raft_log.wal + raft_state.bin
#      under the named volume — this forces a from-zero
#      recovery, exactly what InstallSnapshot is designed for).
#   5. Restart the follower.
#   6. Verify within HEAL_SECS that the follower's
#      last_applied_index reaches the leader's high-water mark.
#
# What this does NOT cover:
#   - Continuous concurrent load during the kill. The script
#     drives load, then kills, then restarts; a real soak runs
#     all three in parallel for hours.
#   - Large-state snapshots (>chunk size). With Noop entries the
#     state machine snapshot stays tiny — under the default
#     snapshot_max_chunk_size every install is one chunk.
#     Exercising the chunked path needs entries that materially
#     grow the state, which means routing real
#     CreateCollection commands through Raft — that wiring
#     belongs to a future "Raft-routed schema mutations"
#     engagement.
#   - Snapshot corruption / partial install recovery.
#
# Usage:
#   docker compose --profile raft-cluster up -d veda-1 veda-2 veda-3
#   ./scripts/raft-snapshot-soak.sh
#
# Env knobs:
#   NOOP_COUNT       (default 200) — load entries to write
#   HEAL_SECS        (default 60)  — max wait for follower catch-up
#   SETTLE_SECS      (default 30)  — initial-election wait

set -euo pipefail

NODE_PORTS=(6340 6342 6344)
NODE_CONTAINERS=(project-mdb-veda-1-1 project-mdb-veda-2-1 project-mdb-veda-3-1)
NODE_VOLUMES=(project-mdb_veda-1-data project-mdb_veda-2-data project-mdb_veda-3-data)
NOOP_COUNT="${NOOP_COUNT:-200}"
HEAL_SECS="${HEAL_SECS:-60}"
SETTLE_SECS="${SETTLE_SECS:-30}"

log() { echo "[snapshot-soak] $*" >&2; }

leader_of() {
  curl -fsS "http://localhost:${1}/v1/admin/metrics" 2>/dev/null \
    | awk '/^veda_raft_leader_id /{print $2; exit}'
}
last_applied_of() {
  curl -fsS "http://localhost:${1}/v1/admin/metrics" 2>/dev/null \
    | awk '/^veda_raft_last_applied_index /{print $2; exit}'
}

# Step 1 — find the leader, pick a follower to kill.
log "waiting up to ${SETTLE_SECS}s for an initial leader..."
leader=0
for _ in $(seq 1 "$SETTLE_SECS"); do
  sleep 1
  leader="$(leader_of 6340 || echo 0)"
  [[ "$leader" != "0" && -n "$leader" ]] && break
done
if [[ "$leader" == "0" || -z "$leader" ]]; then
  log "FATAL: no leader"
  exit 1
fi
log "leader is node-${leader}"

# Pick a follower id != leader. Prefer node-3 then node-2.
follower=0
for cand in 3 2 1; do
  if [[ "$cand" != "$leader" ]]; then
    follower="$cand"
    break
  fi
done
case "$follower" in
  1) follower_port=6340; follower_container="project-mdb-veda-1-1"; follower_volume="project-mdb_veda-1-data" ;;
  2) follower_port=6342; follower_container="project-mdb-veda-2-1"; follower_volume="project-mdb_veda-2-data" ;;
  3) follower_port=6344; follower_container="project-mdb-veda-3-1"; follower_volume="project-mdb_veda-3-data" ;;
esac
case "$leader" in
  1) leader_port=6340 ;;
  2) leader_port=6342 ;;
  3) leader_port=6344 ;;
esac
log "will wipe + restart follower node-${follower} (container ${follower_container})"

# Step 2 — load N Noop commands through the leader.
log "writing ${NOOP_COUNT} Noop entries through node-${leader} (port ${leader_port})..."
for i in $(seq 1 "$NOOP_COUNT"); do
  code=$(curl -s -o /dev/null -w "%{http_code}" -X POST "http://localhost:${leader_port}/v1/admin/cluster/noop")
  if [[ "$code" != "201" ]]; then
    log "FATAL: noop ${i} returned ${code}"
    exit 2
  fi
  if [[ $((i % 50)) == 0 ]]; then log "  ${i} / ${NOOP_COUNT} ..."; fi
done
target_idx="$(last_applied_of "$leader_port")"
log "leader last_applied_index = ${target_idx}"

# Step 3 — stop the follower.
log "stopping ${follower_container}"
docker stop "$follower_container" >/dev/null

# Step 4 — wipe its raft state. The named volume mount is at
# /var/veda/data inside the container; raft files live under
# /var/veda/data/raft/. We blow them away from the host using
# a temporary helper container that mounts the same volume.
log "wiping ${follower_volume}/raft/*"
docker run --rm -v "${follower_volume}:/d" alpine sh -c "rm -rf /d/raft" >/dev/null

# Step 5 — restart the follower.
log "restarting ${follower_container}"
docker start "$follower_container" >/dev/null

# Step 6 — wait for catch-up. Newly-empty node has to receive
# every committed entry from the leader. With NOOP_COUNT=200
# entries this is fast over Tonic; the harness leaves plenty of
# margin in HEAL_SECS for slow runners.
log "waiting up to ${HEAL_SECS}s for follower to catch up to last_applied_index >= ${target_idx}..."
caught_up=0
for _ in $(seq 1 "$HEAL_SECS"); do
  sleep 1
  v="$(last_applied_of "$follower_port" 2>/dev/null || echo 0)"
  if [[ "$v" -ge "$target_idx" ]]; then
    log "follower at last_applied_index=${v} (target ${target_idx}) — caught up"
    caught_up=1
    break
  fi
done
if [[ "$caught_up" -ne 1 ]]; then
  cur="$(last_applied_of "$follower_port" 2>/dev/null || echo 0)"
  log "FATAL: follower stuck at ${cur}, target ${target_idx}"
  exit 3
fi

# Final cross-check: all three nodes report the same
# last_applied_index.
vals=()
for p in "${NODE_PORTS[@]}"; do
  vals+=("$(last_applied_of "$p")")
done
if [[ "${vals[0]}" != "${vals[1]}" || "${vals[1]}" != "${vals[2]}" ]]; then
  log "FATAL: last_applied diverged across nodes: ${vals[*]}"
  exit 4
fi
log "all three nodes at last_applied_index=${vals[0]} — PASS"
exit 0

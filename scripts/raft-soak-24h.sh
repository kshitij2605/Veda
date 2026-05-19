#!/usr/bin/env bash
#
# Wrapper for the standard soak harness with the parameters
# tuned for a 24-hour Jepsen-style operator-confidence run.
#
# This script does NOT execute the soak in the foreground —
# 24h shells are fragile (SSH disconnect, terminal close).
# Instead it:
#   1. Validates the cluster is up and healthy.
#   2. Writes a structured run header to LOG_DIR.
#   3. nohup's the soak loop into the background.
#   4. Prints the PID + log path for the operator to monitor.
#
# Usage:
#   docker compose --profile raft-cluster up -d veda-1 veda-2 veda-3
#   LOG_DIR=/tmp/veda-soak ./scripts/raft-soak-24h.sh
#
# Then to check progress at any point:
#   tail -f /tmp/veda-soak/soak.log
#   grep FAIL /tmp/veda-soak/soak.log     # any divergence events
#   grep PASS /tmp/veda-soak/soak.log | wc -l   # successful iterations
#
# To stop early:
#   kill $(cat /tmp/veda-soak/soak.pid)
#
# Recommended schedule for a real production-confidence run:
#   - 24h soak with ITERATION_SECS=60 (1440 fault iterations)
#   - Watch for first FAIL (script keeps running so subsequent
#     faults can surface compounding failure modes)
#   - At the end, the report enumerates failure types, divergent
#     states, and pinpoints the iteration that broke things.
#
# What this DOES NOT do:
#   - Drive linearisability load alongside the faults. The
#     soak harness only issues Noop writes (small, fixed-size).
#     For a real Jepsen-style run, also start the
#     linearisability harness in parallel:
#       VEDA_LIN_DURATION_SECS=86400 VEDA_LIN_CLIENTS=8 \
#         ./target/release/veda-linearisability-harness &
#   - Detect deadlock vs. liveness vs. progress. We assert
#     convergence after each fault clears; a hung leader that
#     never recovers shows up as "FAIL: divergent state" but
#     not as a clean diagnosis.

set -uo pipefail

LOG_DIR="${LOG_DIR:-/tmp/veda-soak}"
ITERATION_SECS="${ITERATION_SECS:-60}"
HEAL_SECS="${HEAL_SECS:-30}"
WRITER_PROCS="${WRITER_PROCS:-3}"
WRITER_RATE="${WRITER_RATE:-0.05}"
# Default duration: 24h. Override with DURATION_SECS=N.
DURATION_SECS="${DURATION_SECS:-86400}"

mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/soak.log"
PID_FILE="$LOG_DIR/soak.pid"
HEADER="$LOG_DIR/soak.header"

# Pre-flight: assert cluster is up.
for p in 6340 6342 6344; do
  if ! curl -fsS --max-time 3 "http://localhost:${p}/v1/admin/health" >/dev/null; then
    echo "FATAL: node on port ${p} not reachable; cluster not running?" >&2
    exit 1
  fi
done
LEADER="$(curl -fsS http://localhost:6340/v1/admin/metrics | awk '/^veda_raft_leader_id /{print $2; exit}' || echo 0)"
if [[ "$LEADER" == "0" || -z "$LEADER" ]]; then
  echo "FATAL: cluster has no leader yet" >&2
  exit 1
fi
echo "cluster healthy; current leader = node-${LEADER}"

# Header (gets included in any post-mortem).
{
  echo "=== Veda Raft soak run ==="
  echo "started_at:    $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "host:          $(hostname)"
  echo "git_commit:    $(git -C "$(dirname "$0")/.." rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "duration_secs: ${DURATION_SECS}"
  echo "iteration_secs:${ITERATION_SECS}"
  echo "heal_secs:     ${HEAL_SECS}"
  echo "writer_procs:  ${WRITER_PROCS}"
  echo "writer_rate:   ${WRITER_RATE}"
  echo "initial_leader:${LEADER}"
  echo ""
} > "$HEADER"
cat "$HEADER" > "$LOG"

# Detach the soak.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
nohup env \
  DURATION_SECS="$DURATION_SECS" \
  ITERATION_SECS="$ITERATION_SECS" \
  HEAL_SECS="$HEAL_SECS" \
  WRITER_PROCS="$WRITER_PROCS" \
  WRITER_RATE="$WRITER_RATE" \
  "$SCRIPT_DIR/raft-soak.sh" >> "$LOG" 2>&1 &
echo $! > "$PID_FILE"

echo ""
echo "soak running in background. PID: $(cat "$PID_FILE")"
echo "log:    $LOG"
echo "header: $HEADER"
echo ""
echo "monitor with:  tail -f $LOG"
echo "stop with:     kill \$(cat $PID_FILE)"
echo "count PASSes:  grep -c 'iter PASS' $LOG"
echo "count FAILs:   grep -c 'iter FAIL' $LOG"

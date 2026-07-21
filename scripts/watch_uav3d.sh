#!/usr/bin/env bash
# Watch a UAVTurret3D smoke/collect run and report terminal state. Runs ON THE
# HOST (the GPU box), polling the host-side log. Unlike the run script's
# `timeout` backstop (which does NOT reliably kill the isaac-lab container —
# host SIGTERM doesn't propagate through bash→isaaclab.sh→Kit), this watcher's
# only kill mechanism is `docker kill` (SIGKILL, kernel-enforced), which IS
# reliable. It kills the container on DISK_LOW and on TIMEOUT.
#
# Usage:  bash watch_uav3d.sh <log_name> <max_seconds> [success_marker] [image]
#   log_name       basename under ~/docker/isaac-sim/logs/ (e.g. smoke_uav3d_run1.log)
#   max_seconds    hard cap before TIMEOUT-killing the container
#   success_marker grep pattern for success (default: "UAVTURRET3D OK")
#   image          container image to kill (default: nvcr.io/nvidia/isaac-lab:2.3.2)
#
# Emits exactly one terminal line: SUCCESS: / FAILED: / DISK_LOW: / TIMEOUT:
# Each terminal branch also kills any lingering isaac-lab container first (except
# SUCCESS, where the container already exited via --rm).
set -u
LOG_NAME="${1:?log_name required}"
MAX="${2:?max_seconds required}"
MARKER="${3:-UAVTURRET3D OK}"
IMAGE="${4:-nvcr.io/nvidia/isaac-lab:2.3.2}"
LOG="$HOME/docker/isaac-sim/logs/$LOG_NAME"
POLL=15
DISK_LOW_BYTES=15000000000   # 15 G — standing rule: SSD must never fill

kill_container() {
  local cid
  cid=$(docker ps --filter ancestor="$IMAGE" -q | head -1)
  if [ -n "$cid" ]; then docker kill "$cid" >/dev/null 2>&1 && echo "(killed container $cid)"; fi
}

elapsed=0
while [ "$elapsed" -lt "$MAX" ]; do
  if grep -qE "$MARKER" "$LOG" 2>/dev/null; then
    echo "SUCCESS: run completed ($MARKER found)"
    grep -E "\[SMOKE\]|\[COLLECT\]" "$LOG" 2>/dev/null | tail -12
    exit 0
  fi
  if grep -qE "SMOKE3D DONE|COLLECT.* DONE" "$LOG" 2>/dev/null && ! grep -qE "$MARKER" "$LOG" 2>/dev/null; then
    echo "FAILED: DONE without success marker"
    echo "--- error context (Traceback/Error/Exception lines, if any) ---"
    grep -iE "error|traceback|exception|raise |assert|cuda|abort" "$LOG" 2>/dev/null | grep -viE "warning|deprecated|no error|0 error" | tail -25
    echo "--- last 15 non-empty log lines ---"
    grep -vE "^\s*$" "$LOG" 2>/dev/null | tail -15
    exit 0
  fi
  FREE=$(df --output=avail -B1 / | tail -1 | tr -d " ")
  if [ -n "$FREE" ] && [ "$FREE" -lt "$DISK_LOW_BYTES" ]; then
    echo "DISK_LOW: SSD free ${FREE} bytes < 15G — killing container"
    kill_container
    exit 0
  fi
  sleep "$POLL"
  elapsed=$((elapsed + POLL))
done

echo "TIMEOUT: run did not finish in ${MAX}s — killing container"
kill_container
echo "--- last 20 non-empty log lines ---"
grep -vE "^\s*$" "$LOG" 2>/dev/null | tail -20

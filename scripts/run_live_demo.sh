#!/usr/bin/env bash
# Phase 7 live demo: fly the CEM planner (or WaypointPolicy baseline) in the
# Isaac UAVTurret3D env. Two-process: the host planner_server.py (swm venv) must
# already be listening on --planner_port (start it with the SAME --num_envs).
# This wrapper runs the container side. Mirrors run_collect_uav3d_train.sh:
# same bind-mounts + LOAD-BEARING thruster passthrough + disk-safety guard.
# --network=host so the container reaches the host planner on 127.0.0.1:PORT.
#
# Usage:  bash run_live_demo.sh [mode] [num_envs] [episodes] [port] [mcap_dir]
#   mode: nav | planner | detector | waypoint | showcase
#     nav       = oblivious A->B (goal+building only; rounds the corner into T, dies)
#     planner   = WM A->B (goal+building + imagined danger; detours around the block)
#     detector  = detector-reactive A->B (goal+building nav, hard flee on det_logit)
#     waypoint  = WaypointPolicy wander baseline (no planner server needed)
#     showcase  = phantom (drone flies oblivious head-on; server returns signals only)
#   nav/planner/detector/showcase need the host planner_server.py listening on --port
#   (start it with the SAME --num_envs and the matching --mode). waypoint needs none.
#   mcap_dir: container path under /workspace/output (already bind-mounted) to log
#             one Foxglove MCAP per episode; empty = off. e.g. /workspace/output/showcase
#   defaults: mode=planner num_envs=16 episodes=40 port=5557 mcap_dir=""
set -u
MODE="${1:-planner}"
NUM_ENVS="${2:-16}"
EPISODES="${3:-40}"
PORT="${4:-5557}"
MCAP_DIR="${5:-}"
LOG="$HOME/docker/isaac-sim/logs/live_demo_${MODE}.log"
mkdir -p "$(dirname "$LOG")"

# --- disk-safety guard (standing rule: SSD must never fill) ---
FREE=$(df --output=avail -B1 / | tail -1 | tr -d " ")
if [ "$FREE" -lt 20000000000 ]; then
  echo "ABORT: SSD < 20G free ($FREE bytes) — refusing to launch live demo." | tee "$LOG"
  exit 2
fi

echo "=== LIVE-DEMO START $(date -u +%Y-%m-%dT%H:%M:%SZ) mode=${MODE} free=${FREE} num_envs=${NUM_ENVS} episodes=${EPISODES} port=${PORT} mcap_dir=${MCAP_DIR:-off} ===" | tee "$LOG"

MCAP_ARG=""
if [ -n "${MCAP_DIR}" ]; then
  MCAP_ARG="--mcap_dir ${MCAP_DIR}"
fi

timeout 5400 docker run --rm --gpus all --network=host \
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
  -e PYTHONPATH=/workspace -e PYTHONUNBUFFERED=1 \
  --entrypoint bash \
  -v "$HOME/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw" \
  -v "$HOME/docker/isaac-sim/cache/ov:/root/.cache/ov:rw" \
  -v "$HOME/docker/isaac-sim/cache/pip:/root/.cache/pip:rw" \
  -v "$HOME/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw" \
  -v "$HOME/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw" \
  -v "$HOME/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw" \
  -v "$HOME/docker/isaac-sim/data:/root/.local/share/ov/data:rw" \
  -v "$HOME/docker/isaac-sim/scripts:/workspace/scripts:ro" \
  -v "$HOME/docker/isaac-sim/uav_wm:/workspace/uav_wm:ro" \
  -v "$HOME/docker/isaac-sim/libs:/workspace/libs:ro" \
  -v "$HOME/docker/isaac-sim/patches/thruster.py:/workspace/isaaclab/source/isaaclab_contrib/isaaclab_contrib/actuators/thruster.py:ro" \
  -v "$HOME/docker/isaac-sim/output:/workspace/output:rw" \
  nvcr.io/nvidia/isaac-lab:2.3.2 \
  -c "export PYTHONPATH=/workspace PYTHONUNBUFFERED=1 && /workspace/isaaclab/isaaclab.sh -p /workspace/scripts/live/live_demo.py --headless --enable_cameras --num_envs ${NUM_ENVS} --episodes ${EPISODES} --mode ${MODE} --planner_port ${PORT} ${MCAP_ARG}" \
  2>&1 | tee -a "$LOG"
RC=${PIPESTATUS[0]}

echo "=== LIVE-DEMO DONE exit=$RC $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$LOG"

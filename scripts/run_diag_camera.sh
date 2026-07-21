#!/usr/bin/env bash
# Wrapper: run scripts/diag_camera.py inside the isaac-lab container to
# diagnose whether the drone POV camera follows the drone's pose (Evan's
# critique: POV renders a FIXED perspective, not the drone's). Captures all
# stdout/stderr to a host-side log (SSD). Bounded by `timeout`. Disk-safety
# guard: abort if SSD < 20G free. Success marker to grep: "[DIAG] UAVTURRET3D DIAG OK".
#
# Same mounts as run_smoke_uav3d.sh (incl. the LOAD-BEARING thruster-patch
# bind-mount so the drone actually flies during the trace). num_envs=1.
set -u
LOG="$HOME/docker/isaac-sim/logs/diag_camera_run1.log"
mkdir -p "$(dirname "$LOG")"
mkdir -p "$HOME/docker/isaac-sim/output/uav3d_diag"

# --- disk-safety guard (standing rule: SSD must never fill) ---
FREE=$(df --output=avail -B1 / | tail -1 | tr -d " ")
if [ "$FREE" -lt 20000000000 ]; then
  echo "ABORT: SSD < 20G free ($FREE bytes) — refusing to launch diag." | tee "$LOG"
  exit 2
fi

echo "=== DIAGCAM START $(date -u +%Y-%m-%dT%H:%M:%SZ) free=${FREE} ===" | tee "$LOG"

timeout 600 docker run --rm --gpus all --network=host \
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
  -v "$HOME/docker/isaac-sim/patches/thruster.py:/workspace/isaaclab/source/isaaclab_contrib/isaaclab_contrib/actuators/thruster.py:ro" \
  -v "$HOME/docker/isaac-sim/output:/workspace/output:rw" \
  nvcr.io/nvidia/isaac-lab:2.3.2 \
  -c "export PYTHONPATH=/workspace PYTHONUNBUFFERED=1 && /workspace/isaaclab/isaaclab.sh -p /workspace/scripts/diag_camera.py --headless --enable_cameras --num_envs 1 --output_dir /workspace/output/uav3d_diag" \
  2>&1 | tee -a "$LOG"
RC=${PIPESTATUS[0]}

echo "=== DIAGCAM DONE exit=$RC $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$LOG"

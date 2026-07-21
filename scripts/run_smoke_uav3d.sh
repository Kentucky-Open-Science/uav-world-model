#!/usr/bin/env bash
# Wrapper: run the UAVTurret3D env smoke inside the isaac-lab container.
# Captures ALL stdout/stderr to a host-side log (SSD, bind-mounted). Bounded by
# `timeout`. Disk-safety guard: abort if SSD < 20G free at start. Emits
# start/done sentinels for the watcher. The success marker to grep is
# "[SMOKE] UAVTURRET3D OK".
#
# Mounts the uav_wm package (read-only) at /workspace/uav_wm and sets
# PYTHONPATH=/workspace so `import uav_wm.envs` works in the container. Output
# PNGs go to a bind-mounted host dir (NEVER in-container — standing SSD rule).
set -u
LOG="$HOME/docker/isaac-sim/logs/smoke_uav3d_run1.log"
mkdir -p "$(dirname "$LOG")"
mkdir -p "$HOME/docker/isaac-sim/output/uav3d_smoke"

# --- disk-safety guard (standing rule: SSD must never fill) ---
FREE=$(df --output=avail -B1 / | tail -1 | tr -d " ")
if [ "$FREE" -lt 20000000000 ]; then
  echo "ABORT: SSD < 20G free ($FREE bytes) — refusing to launch smoke." | tee "$LOG"
  exit 2
fi

echo "=== SMOKE3D START $(date -u +%Y-%m-%dT%H:%M:%SZ) free=${FREE} ===" | tee "$LOG"

timeout 600 docker run --rm --gpus all --network=host \
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
  -e PYTHONPATH=/workspace -e PYTHONUNBUFFERED=1 \
  -e UAV_CTRL_DEBUG=1 \
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
  -c "export PYTHONPATH=/workspace PYTHONUNBUFFERED=1 && /workspace/isaaclab/isaaclab.sh -p /workspace/scripts/smoke_uav3d.py --headless --enable_cameras --num_envs 4 --output_dir /workspace/output/uav3d_smoke" \
  2>&1 | tee -a "$LOG"
RC=${PIPESTATUS[0]}

echo "=== SMOKE3D DONE exit=$RC $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$LOG"

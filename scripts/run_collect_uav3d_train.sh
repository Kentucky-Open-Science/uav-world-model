#!/usr/bin/env bash
# Wrapper: run the UAVTurret3D TRAINING collector (Stage 1) in the isaac-lab container.
# Produces per-episode pickles; lance_from_episodes.py (host venv) is Stage 2.
# Mirrors run_collect_uav3d.sh: same bind-mounts + LOAD-BEARING thruster passthrough +
# disk-safety guard (SSD never fills). Success marker to grep:
#   "[COLLECT] UAVTURRET3D TRAIN COLLECT OK".
#
# Usage:  bash run_collect_uav3d_train.sh [num_envs] [episodes] [q]
#   defaults: num_envs=16 episodes=50 q=0.3   (50 = the density GATE; scale to ~2000 after)
# q = policy WANDER fraction (q=0.3 -> 30% wander / 70% provoke, max turret encounters;
#   see collect_uav_3d.py header). Mid-city spawn (env P_INTERIOR_SPAWN) + provoke
#   manufacture the 5-10% danger density (Modification 1). The drone flies BODY-FRAME
#   4D (vx_fwd,vy_strafe,vz,yaw_rate), no auto-yaw; shot drones fall+tumble.
# Output pickles + manifest -> ~/docker/isaac-sim/output/uav3d_episodes (bind-mounted,
#   NEVER in-container -- standing SSD rule). Stage 2 reads them and writes the lance
#   to the SSD dataset dir (~/.stable_worldmodel/datasets/uav_isaac_train.lance).
#
# LOAD-BEARING: the patches/thruster.py bind-mount overlays the passthrough fix over
# the container's thruster. Without it the rpm-domain motor filter overshoots commanded
# thrust ~2.2x -> drone climbs uncontrollably and episodes are garbage. Do NOT drop.
set -u
NUM_ENVS="${1:-16}"
EPISODES="${2:-50}"
Q="${3:-0.3}"
LOG="$HOME/docker/isaac-sim/logs/collect_uav3d_train.log"
mkdir -p "$(dirname "$LOG")"
mkdir -p "$HOME/docker/isaac-sim/output/uav3d_episodes"

# Start each run fresh: a smaller re-run would leave stale higher-index pickles
# from the last run, and lance_from_episodes.py reads EVERY pickle in the dir --
# so stale pickles would silently mix old+new policy data into the lance. Wipe
# the dir's products before collecting (data-integrity + storage-mindful).
rm -f "$HOME/docker/isaac-sim/output/uav3d_episodes"/*.pkl \
      "$HOME/docker/isaac-sim/output/uav3d_episodes"/manifest.txt

# --- disk-safety guard (standing rule: SSD must never fill) ---
FREE=$(df --output=avail -B1 / | tail -1 | tr -d " ")
if [ "$FREE" -lt 20000000000 ]; then
  echo "ABORT: SSD < 20G free ($FREE bytes) — refusing to launch collector." | tee "$LOG"
  exit 2
fi

echo "=== COLLECT3D-TRAIN START $(date -u +%Y-%m-%dT%H:%M:%SZ) free=${FREE} num_envs=${NUM_ENVS} episodes=${EPISODES} q=${Q} ===" | tee "$LOG"

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
  -c "export PYTHONPATH=/workspace PYTHONUNBUFFERED=1 && /workspace/isaaclab/isaaclab.sh -p /workspace/scripts/collect_uav_3d.py --headless --enable_cameras --num_envs ${NUM_ENVS} --episodes ${EPISODES} --q ${Q} --output_dir /workspace/output/uav3d_episodes" \
  2>&1 | tee -a "$LOG"
RC=${PIPESTATUS[0]}

echo "=== COLLECT3D-TRAIN DONE exit=$RC $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$LOG"

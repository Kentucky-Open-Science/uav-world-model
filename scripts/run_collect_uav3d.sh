#!/usr/bin/env bash
# Wrapper: run the UAVTurret3D critique collector inside the isaac-lab container.
# Produces Foxglove MCAP episodes for Evan's pre-training critique gate (Task #14).
# Mirrors run_smoke_uav3d.sh: same bind-mounts, disk-safety guard (SSD never fills),
# start/done sentinels for the watcher. Success marker to grep:
#   "[COLLECT] UAVTURRET3D CRITIQUE OK".
#
# Usage:  bash run_collect_uav3d.sh [num_envs] [episodes] [q_provoke]
#   defaults: num_envs=6 episodes=12 q_provoke=0.4
# q_provoke = drunken-explorer provoke (vs wander) fraction. ~0.3 is the spec
#   (70% wander / 30% provoke); 0.4 leans slightly provoke so the critique reliably
#   has approach->danger->kill+fall episodes to review (the gate's whole point).
#   Wander itself also crosses the turret's street-intersection aim cone, so danger
#   is denser than the old A->B policy at the same q. r (commit vs evade in danger)
#   is left at the policy default 0.3. The drone flies BODY-FRAME 4D
#   (vx_fwd,vy_strafe,vz,yaw_rate), no auto-yaw, and shot drones fall+tumble,
#   ending FALL_RECORD_S after ground impact.
# Output MCAPs + manifest -> ~/docker/isaac-sim/output/uav3d_critique (bind-mounted,
#   NEVER in-container — standing SSD rule). rsync to Mac via scripts/replay_foxglove.sh.
#
# LOAD-BEARING: the patches/thruster.py bind-mount overlays the passthrough fix
# over the container's thruster. Without it the rpm-domain motor filter overshoots
# commanded thrust ~2.2x -> drone climbs uncontrollably and episodes are garbage.
# Do NOT drop this mount. (See run_smoke_uav3d.sh for the same mount.)
set -u
NUM_ENVS="${1:-6}"
EPISODES="${2:-12}"
Q="${3:-0.4}"
LOG="$HOME/docker/isaac-sim/logs/collect_uav3d.log"
mkdir -p "$(dirname "$LOG")"
mkdir -p "$HOME/docker/isaac-sim/output/uav3d_critique"

# Start each run fresh: short episodes leave stale step-NN PNGs and a smaller run
# leaves stale higher-index MCAPs/PNGs from the last run. Wipe the dir's products
# (standing storage rule: don't let old examples pile up). rsync --delete on the
# Mac side then mirrors this clean state.
rm -f "$HOME/docker/isaac-sim/output/uav3d_critique"/*.mcap \
      "$HOME/docker/isaac-sim/output/uav3d_critique"/*.png \
      "$HOME/docker/isaac-sim/output/uav3d_critique"/manifest.txt

# --- disk-safety guard (standing rule: SSD must never fill) ---
FREE=$(df --output=avail -B1 / | tail -1 | tr -d " ")
if [ "$FREE" -lt 20000000000 ]; then
  echo "ABORT: SSD < 20G free ($FREE bytes) — refusing to launch collector." | tee "$LOG"
  exit 2
fi

echo "=== COLLECT3D START $(date -u +%Y-%m-%dT%H:%M:%SZ) free=${FREE} num_envs=${NUM_ENVS} episodes=${EPISODES} q=${Q} ===" | tee "$LOG"

timeout 2400 docker run --rm --gpus all --network=host \
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
  -c "export PYTHONPATH=/workspace PYTHONUNBUFFERED=1 && /workspace/isaaclab/isaaclab.sh -p /workspace/scripts/collect_uav3d_critique.py --headless --enable_cameras --num_envs ${NUM_ENVS} --episodes ${EPISODES} --q ${Q} --output_dir /workspace/output/uav3d_critique" \
  2>&1 | tee -a "$LOG"
RC=${PIPESTATUS[0]}

echo "=== COLLECT3D DONE exit=$RC $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$LOG"

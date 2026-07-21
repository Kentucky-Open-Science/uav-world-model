#!/usr/bin/env bash
# Wrapper: run the native-multirotor smoke inside the isaac-lab container,
# capturing ALL stdout/stderr to a host-side log file (SSD, bind-mounted).
# Bounded by `timeout`. Disk-safety guard: abort if SSD < 20G free at start.
# Emits explicit start/done sentinels so a watcher can detect completion.
set -u
LOG="$HOME/docker/isaac-sim/logs/smoke_run4.log"
mkdir -p "$(dirname "$LOG")"

# --- disk-safety guard (standing rule: SSD must never fill) ---
FREE=$(df --output=avail -B1 / | tail -1 | tr -d " ")
if [ "$FREE" -lt 20000000000 ]; then
  echo "ABORT: SSD < 20G free ($FREE bytes) — refusing to launch smoke." | tee "$LOG"
  exit 2
fi

echo "=== SMOKE START $(date -u +%Y-%m-%dT%H:%M:%SZ) free=${FREE} ===" | tee "$LOG"

timeout 600 docker run --rm --gpus all --network=host \
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y --entrypoint bash \
  -v "$HOME/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw" \
  -v "$HOME/docker/isaac-sim/cache/ov:/root/.cache/ov:rw" \
  -v "$HOME/docker/isaac-sim/cache/pip:/root/.cache/pip:rw" \
  -v "$HOME/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw" \
  -v "$HOME/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw" \
  -v "$HOME/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw" \
  -v "$HOME/docker/isaac-sim/data:/root/.local/share/ov/data:rw" \
  -v "$HOME/docker/isaac-sim/scripts:/workspace/scripts:ro" \
  nvcr.io/nvidia/isaac-lab:2.3.2 \
  -c "/workspace/isaaclab/isaaclab.sh -p /workspace/scripts/smoke_native_multirotor.py --headless --num_envs 4" \
  2>&1 | tee -a "$LOG"
RC=${PIPESTATUS[0]}

echo "=== SMOKE DONE exit=$RC $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$LOG"

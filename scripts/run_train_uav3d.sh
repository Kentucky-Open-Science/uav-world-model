#!/usr/bin/env bash
# Run LeWM Phase-3 training on the uav3d POV lance (HOST venv, NOT the Isaac
# container -- swm + torch live in the host swm venv, which cannot run in-container
# because lance.py imports the swm data-stack and the container has no swm).
#
# Disk-safe (standing SSD rule): disk guard at start + a background watcher that
# kills training if SSD free < 15G. Checkpoints are pruned to last-3 by
# SaveCkptCallback._prune_old_epochs (lewm.py, task #26), so the run cannot fill
# the SSD via checkpoints; the watcher is the belt-and-suspenders backstop for any
# unforeseen growth (wandb is disabled, so the only writes are 3 .pt + config.yaml
# + the resume .ckpt). 10h `timeout` backstop guards against a hang.
#
# trainer.devices=1: a single A6000 is plenty for the tiny ViT (encoder_scale=tiny,
# embed_dim=192, batch 128) and avoids DDP quirks for an unattended PoC run.
#
# trainer.max_epochs=25: measured throughput ~3.2 it/s (300-batch probe, steady
# state) -> ~3074 train batches/epoch (393k windows / batch 128) + ~769 val
# batches/epoch = ~20 min/epoch. 100 epochs (lewm.yaml default) = ~33h, which
# would hit the 10h `timeout` at ~epoch 30 with an INCOMPLETE cosine LR schedule
# (LR not annealed -> suboptimal final weights). 25 epochs = ~8.3h, fits the 10h
# timeout with margin AND lets the cosine schedule (total_steps = max_epochs *
# len(train), auto-rescaled) complete cleanly. 25 ep x 530k frames ~= 13.3M
# frame-views, comparable to the 2D PoC's full run. PoC-length: Phase 4's danger
# probe validates the latent; if weak, resume/extend (spt.Manager resumes from the
# per-run .ckpt + last weights_epoch_*.pt).
#
# Launch DETACHED so it survives the SSH session (UAVWM = your repo root on the box):
#   ssh "$GPU_BOX" 'nohup bash "$UAVWM/scripts/run_train_uav3d.sh" </dev/null >>~/docker/isaac-sim/logs/train_uav3d.nohup 2>&1 &'
# Then tail:  ssh "$GPU_BOX" 'tail -f ~/docker/isaac-sim/logs/train_uav3d.log'
# Success marker:  "=== TRAIN DONE ... exit=0 ==="
set -u
# Host swm training venv (editable-installs swm). Override with SWM_VENV if yours differs.
VENV="${SWM_VENV:-$HOME/venvs/swm-train}"
# swm submodule root (lewm.py lives at scripts/train/lewm.py inside it).
REPO="$(cd "$(dirname "$0")/../repos/stable-worldmodel" && pwd)"
LOG="${UAVWM_LOG_DIR:-$HOME/docker/isaac-sim/logs}/train_uav3d.log"
mkdir -p "$(dirname "$LOG")"

# --- disk-safety guard (standing rule: SSD must never fill) ---
FREE=$(df --output=avail -B1 / | tail -1 | tr -d " ")
if [ "$FREE" -lt 20000000000 ]; then
  echo "ABORT: SSD < 20G free ($FREE bytes) — refusing to launch training." | tee "$LOG"
  exit 2
fi

echo "=== TRAIN START $(date -u +%Y-%m-%dT%H:%M:%SZ) free=${FREE} venv=${VENV} ===" | tee "$LOG"

# --- background disk watcher: kill training if SSD free drops below 15G ---
( while true; do
    sleep 60
    f=$(df --output=avail -B1 / | tail -1 | tr -d " ")
    if [ "$f" -lt 15000000000 ]; then
      echo "=== DISK WATCHER: SSD < 15G ($f bytes) — killing lewm.py ===" | tee -a "$LOG"
      pkill -f "scripts/train/lewm.py" 2>/dev/null
      break
    fi
  done ) &
WATCHER=$!

# wandb is disabled by default (launcher/local.yaml wandb.enabled=false -> logger=None);
# WANDB_MODE=disabled is belt-and-suspenders against any stray init.
cd "$REPO" || { echo "ABORT: repo not found: $REPO" | tee -a "$LOG"; exit 2; }
WANDB_MODE=disabled WANDB_DISABLED=true \
  timeout 36000 "$VENV/bin/python" scripts/train/lewm.py data=uav3d trainer.devices=1 trainer.max_epochs=25 2>&1 | tee -a "$LOG"
RC=${PIPESTATUS[0]}

kill "$WATCHER" 2>/dev/null
echo "=== TRAIN DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) exit=$RC ===" | tee -a "$LOG"
exit $RC

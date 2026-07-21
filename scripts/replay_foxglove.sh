#!/usr/bin/env bash
# rsync UAVTurret3D critique MCAPs from the GPU box to the Mac and open in Foxglove.
#
# Part of the pre-training critique gate (Task #13/#14): Evan reviews a few
# episodes in Foxglove BEFORE any LeWM training. The agent does NOT inspect the
# images — it only reports the manifest text + paths.
#
# Usage:  bash scripts/replay_foxglove.sh [remote_output_dir]
#   remote_output_dir defaults to ~/docker/isaac-sim/output/uav3d_critique
set -euo pipefail

REMOTE="${GPU_BOX:-gpu-box}"   # set GPU_BOX to your GPU box's SSH alias
REMOTE_DIR="${1:-~/docker/isaac-sim/output/uav3d_critique}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
LOCAL_DIR="$REPO/data/uav3d_critique"

mkdir -p "$LOCAL_DIR"
echo "=== rsync $REMOTE:$REMOTE_DIR -> $LOCAL_DIR ==="
# shellcheck disable=SC2088  (tilde expansion must happen on the remote)
rsync -az --delete -e ssh "$REMOTE:$REMOTE_DIR/" "$LOCAL_DIR/"

echo
echo "=== local contents ==="
ls -la "$LOCAL_DIR"

echo
echo "=== manifest (episode stats) ==="
cat "$LOCAL_DIR/manifest.txt" 2>/dev/null || echo "(no manifest.txt — did the collector finish?)"

echo
# open the highest-danger episode first (the one where the drone spent the most
# steps near the turret -- best single episode to verify BOTH the POV fix [the
# view moves with the drone] AND the turret-approach/danger behavior). Fall back
# to the first MCAP if no manifest or no danger rows.
shopt -s nullglob
MCAPS=( "$LOCAL_DIR"/*.mcap )
if [ "${#MCAPS[@]}" -eq 0 ]; then
  echo "NO MCAPs found in $LOCAL_DIR — did the collector run on the GPU box?"
  exit 1
fi
OPEN_MCAP="${MCAPS[0]}"
if [ -f "$LOCAL_DIR/manifest.txt" ]; then
  BEST=$(awk '!/^#/ && NF>=4 {d=$4+0; if(d>max){max=d; name=$1}} END{print name}' "$LOCAL_DIR/manifest.txt")
  [ -n "$BEST" ] && [ -f "$LOCAL_DIR/$BEST" ] && OPEN_MCAP="$LOCAL_DIR/$BEST"
fi
echo "=== ${#MCAPS[@]} MCAP file(s). Opening highest-danger episode in Foxglove: ==="
echo "  $OPEN_MCAP"
open -a Foxglove "$OPEN_MCAP" 2>/dev/null || {
  echo "(auto-open failed — open Foxglove and drag the .mcap onto it, or: open -a Foxglove <file.mcap>)"
}

cat <<'GUIDE'

=== Foxglove setup (do once) ===
Add panels and pick these topics (one MCAP has all four):
  /drone/pov/image  -> Image panel  (foxglove.RawImage)  [the WM's training pixels — 224x224 drone POV]
  /scene            -> 3D panel     (foxglove.SceneUpdate)
  /state            -> Log panel    (foxglove.Log)       [danger flags per step, syncs to timeline]
  /drone/tf         -> (3D panel uses it to offer a "drone" camera frame)

3D panel legend:  drone=blue cube  turret=red cube  obstacles=grey  goal=green sphere
                   aim-line = orange (safe) / RED (danger active).
Scrub the timeline; the aim-line goes red and the Log panel warns when danger fires.
Open the other episodes:  open -a Foxglove data/uav3d_critique/episode_001.mcap

See foxglove/uav3d_layout.md for the full critique checklist.
GUIDE

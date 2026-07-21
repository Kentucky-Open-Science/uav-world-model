#!/usr/bin/env bash
# rsync the live-demo showcase MCAPs from the GPU box to the Mac, text-rank them
# by how far the WM danger signal leads the detector logit (pick_showcase.py),
# and open the top-ranked episode in Foxglove. The agent does NOT inspect images
# -- it reports the ranking text + path; Evan does the visual review.
#
# Showcase = the "world model predicts ahead of detection" MCAPs. Each planner-mode
# episode carries a /signals channel (PoseInFrame: position.x=wm_danger, y=det_logit,
# z=danger) so Foxglove's Plot panel graphs imagination-vs-detection on ONE timeline.
#
# Usage:  bash scripts/replay_showcase.sh [mode]
#   mode = phantom (THE lead showcase: WM-leads-detector on /signals)
#        | planner | waypoint | both (planner+waypoint)   -- Result 1 / wander story
#        | nav                                          -- Result 2: A->B head-on hold-back
#   nav rsyncs the THREE nav controllers (nav_planner/nav_oblivious/nav_detector)
#   and ranks them by the ideal trio (planner holds back/survives, oblivious+detector die).
set -euo pipefail

REMOTE="${GPU_BOX:-gpu-box}"   # set GPU_BOX to your GPU box's SSH alias
MODE="${1:-planner}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
SHOWCASE_VENV="${SHOWCASE_VENV:-$HOME/venvs/uav-showcase/bin/python}"

rsync_one() {
  local sub="$1"   # showcase_planner | nav_planner | ...
  local remote_dir="~/docker/isaac-sim/output/${sub}"
  local local_dir="$REPO/data/${sub}"
  mkdir -p "$local_dir"
  echo "=== rsync $REMOTE:$remote_dir -> $local_dir ==="
  # shellcheck disable=SC2088  (tilde expansion on the remote)
  rsync -az --delete -e ssh "$REMOTE:$remote_dir/" "$local_dir/" || echo "  (none on remote yet)"
  echo "=== local ($sub): $(ls "$local_dir"/*.mcap 2>/dev/null | wc -l | tr -d ' ') MCAPs ==="
}

# ---- nav: the A->B corner-ambush story (Result 2) ----
if [ "$MODE" = "nav" ]; then
  rsync_one nav_planner
  rsync_one nav_oblivious
  rsync_one nav_detector
  echo
  echo "=== nav A->B ranking (text-only; ideal trio) ==="
  python3 "$REPO/scripts/pick_showcase.py" --mode nav 10 \
    --nav_planner "$REPO/data/nav_planner" \
    --nav_oblivious "$REPO/data/nav_oblivious" \
    --nav_detector "$REPO/data/nav_detector" || true
  echo
  echo "=== build the 3-panel A->B gif from the top-ranked trio (text-only verify) ==="
  TRIO="$(python3 "$REPO/scripts/pick_showcase.py" --mode nav 1 \
    --nav_planner "$REPO/data/nav_planner" \
    --nav_oblivious "$REPO/data/nav_oblivious" \
    --nav_detector "$REPO/data/nav_detector" 2>/dev/null \
    | grep '^# top pick' -A4 | grep '\.mcap' | sed 's/^#.*-> //')"
  PL="$(echo "$TRIO" | grep nav_planner || true)"
  OB="$(echo "$TRIO" | grep nav_oblivious || true)"
  DT="$(echo "$TRIO" | grep nav_detector || true)"
  if [ -n "$PL" ] && [ -n "$OB" ] && [ -n "$DT" ]; then
    "$SHOWCASE_VENV" "$REPO/scripts/make_showcase_figures.py" \
      --only nav-gif --nav_planner "$PL" --nav_oblivious "$OB" --nav_detector "$DT" || true
    echo "=== nav_a_to_b.gif written to $REPO/docs/assets/ -- Evan: open + visual review ==="
  else
    echo "no complete trio found -- run the 3 nav controllers on the GPU box first"
  fi
  exit 0
fi

if [ "$MODE" = "both" ]; then
  rsync_one showcase_planner
  rsync_one showcase_waypoint
  PICK_DIR="$REPO/data/showcase_planner"
else
  rsync_one "showcase_${MODE}"
  PICK_DIR="$REPO/data/showcase_${MODE}"
fi

echo
echo "=== showcase ranking (text-only; WM-leads-detector) ==="
python3 "$REPO/scripts/pick_showcase.py" "$PICK_DIR" 10 || true

echo
# pick the top .mcap from the ranking (re-run the picker, grab the 'top pick' path)
TOP_MCAP="$(python3 "$REPO/scripts/pick_showcase.py" "$PICK_DIR" 1 2>/dev/null \
  | grep '^# top pick:' | sed 's/^# top pick: //')"
if [ -z "$TOP_MCAP" ] || [ ! -f "$TOP_MCAP" ]; then
  # fall back to the first .mcap present
  TOP_MCAP="$(ls "$PICK_DIR"/*.mcap 2>/dev/null | head -1 || true)"
fi

if [ -z "$TOP_MCAP" ]; then
  echo "NO MCAPs found in $PICK_DIR -- did the showcase run finish on the GPU box?"
  exit 1
fi
echo "=== opening top-ranked episode in Foxglove: ==="
echo "  $TOP_MCAP"
open -a Foxglove "$TOP_MCAP" 2>/dev/null || {
  echo "(auto-open failed -- open Foxglove and drag the .mcap onto it)"
}

cat <<'GUIDE'

=== Foxglove showcase setup (do once, save as a layout) ===
Panels + topics (a phantom- or planner-mode MCAP carries all five; waypoint carries none):
  /drone/pov/image -> Image panel  (foxglove.RawImage)  [224x224 drone POV]
  /scene           -> 3D panel     (foxglove.SceneUpdate) [drone/turret/obstacles/aim/trail]
  /signals         -> Plot panel   (foxglove.PoseInFrame) [THE showcase panel]
  /state           -> Log panel    (foxglove.Log)        [danger flags per step]
  /drone/tf        -> (3D panel uses it for a drone-cam frame)

THE showcase panel = Plot on /signals (phantom-mode only -- under the planner the
drone evades so both /signals stay low; waypoint carries no /signals at all).
Graph these vs the header timestamp:
  pose.position.x  = wm_danger    (the world model's imagined danger -- LEADS)
  pose.position.y  = det_logit    (the single-frame detector -- LAGS)
  pose.position.z  = danger       (0/1 ground truth -- when the threat is real)
Overlay the horizontal line y=1.53 (the detector's fire threshold). The phantom story:
wm_danger (x) rises while det_logit (y) is still flat-low, then det_logit catches
up, then danger (z) goes 1. The WM's lead = the horizontal gap between the x-rise
and the y-rise. (wm_danger is noisy/jagged, not a clean ramp -- the lead TIMING is
the claim.) For the survival story instead, run planner vs waypoint (3D trails).
See foxglove/showcase_layout.md for the full checklist.

3D legend: drone=blue, turret=red, obstacles=grey, aim-line=orange(safe)/RED(danger),
green line = the drone's trail this episode (shows the evade).
GUIDE

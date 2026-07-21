# Phase 8 — ROS 2 bridge for live showcase (feasibility + options, 2026-07-17)

**Status: DEFERRED per Evan (chose C — MCAP replay is sufficient), 2026-07-17.** No work to be done unless a live-audience need or a real-robot target appears. The feasibility findings + options are kept below for when it's revisited.

The Phase-7 MCAP showcase (`briefs/phase7-live-demo.md`) already proves the claim with *recorded* episodes opened in Foxglove. Phase 8 was the optional upgrade to *live* streaming (Isaac sim → Foxglove in real time), and — if done via ROS 2 — the bridge layer that would carry forward to a real-robot demo. Evan reviewed the three options (A/B/C below) and chose **C**: the recorded showcase already shows the WM leading the detector and the planner surviving, so ship that and revisit live/ROS 2 only when there's a concrete need. The feasibility findings are recorded here so the choice is informed when it's revisited.

## Container reality (verified on the GPU box, `nvcr.io/nvidia/isaac-lab:2.3.2`)
- **ROS 2 is NOT installed** in the NGC container: `ROS_DISTRO=unset`, no `ros2` CLI, no `/opt/ros`, no `rclpy`.
- **The bridge *extensions* are present** under `/workspace/isaaclab/_isaac_sim/exts/`: `isaacsim.ros2.bridge`, `isaacsim.ros2.sim_control`, `isaacsim.ros2.tf_viewer`, `isaacsim.ros2.urdf`.
- **`isaacsim.ros2.core` is MISSING** — the extension that provides the ROS 2 libraries (rclpy + Cyclone DDS). Without it the bridge extension can't load. (Isaac Sim docs say on Ubuntu 24.04 it can auto-load *internal* Jazzy libs from `isaacsim.ros2.core/jazzy/lib`, but that extension isn't in this container.)
- **Kit Python is 3.11.13**; Isaac Sim's internal ROS 2 libs are built against **Python 3.12** (Cyclone DDS). A version mismatch to resolve whichever route is chosen.
- **Ubuntu 24.04.2 (Noble)** → ROS 2 **Jazzy** is the native distro match (Humble also supported on 22.04).
- Isaac Sim ROS 2 docs: the bridge extension is `isaacsim.ros2.bridge`; the Docker pattern is a sidecar `osrf/ros:jazzy-desktop` container with `--net=host` + a `fastdds.xml` (UDP transport) for cross-container DDS.

## The scope decision (Evan's call)
The MCAP replay showcase is done and validated. Phase 8 is only worth doing if *live* streaming (or the ROS 2 bridge-for-real-robots angle) is wanted. Three options, ordered by effort:

- **A — Full ROS 2 bridge (the planned Phase 8).** Install `isaacsim.ros2.core` (extension manager / pip) or system ROS 2 Jazzy, resolve the 3.11/3.12 mismatch, configure `isaacsim.ros2.bridge` to publish the drone POV (`sensor_msgs/Image`), scene (`visualization_msgs/MarkerArray`), and signals (`std_msgs/Float32`) from `live_demo.py`, run `foxglove_bridge` (ROS 2 → WebSocket) in a sidecar `osrf/ros:jazzy-desktop` container (`--net=host`), open Foxglove on the Mac live. **Pro:** robotics-standard bridge, carries to a real-robot deployment (planner ↔ real drone via ROS 2). **Con:** multi-hour setup, integration risk (missing core ext, python version, DDS config).
- **B — Lightweight live Foxglove WebSocket (no ROS 2).** `live_demo.py` already streams the POV batch + state to the host planner server over a socket; add a `foxglove-sdk` WebSocket server on the host that publishes the same MCAP channels (`/drone/pov/image`, `/scene`, `/signals`, `/state`) live. Foxglove on the Mac connects via SSH tunnel. **Pro:** fast (reuses the existing two-process socket stream + the `LiveEpisodeWriter` channel builders), no ROS 2 / image-build pain. **Con:** not ROS 2 — doesn't advance the real-robot bridge goal; "live" but not a ROS 2 showcase.
- **C — MCAP replay is sufficient; defer Phase 8.** The recorded showcase already shows the WM leading the detector and the planner surviving. Ship that; revisit live/ROS 2 when there's a real-robot target. **Pro:** zero new work; the claim is already shown. **Con:** no live demo, no ROS 2 layer.

## If A is chosen — first steps (de-risk before committing)
1. Get `isaacsim.ros2.core` into the container: try `~/isaaclab/_isaac_sim/python.sh -m pip install isaacsim-ros2-core` (matching the container's Isaac Sim version), else enable via the Isaac Sim extension manager UI (`--enable isaacsim.ros2.bridge` + core). Verify `import rclpy` works in the kit python.
2. Resolve the 3.11/3.12 question: if the internal Jazzy libs require 3.12, either install system ROS 2 Jazzy (apt, /opt/ros/jazzy) sourced so the bridge uses it, or confirm the 3.11 kit python works with the core ext's bundled libs.
3. Smallest bridge smoke: publish one topic (`std_msgs/Float32` = a step counter) from a throwaway Isaac script, echo it from a sidecar `osrf/ros:jazzy-desktop` container on `--net=host`. Confirms DDS cross-container before wiring the full channel set.
4. Then map the `LiveEpisodeWriter` channels (already built for MCAP) to ROS 2 message types and add a `--ros2` mode to `live_demo.py`.

## Open question that blocks starting
- Which approach (A / B / C)? A is the planned phase but the largest bet; B is the fast live-viz path; C ships what's done. Phase 7 (UKy campus map) is separately blocked on the LiDAR-map data location.

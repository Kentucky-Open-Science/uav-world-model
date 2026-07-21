# UAVTurret3D — Foxglove critique guide (pre-training gate)

This is the **critique gate**: before any LeWM training, Evan reviews a few
collected episodes in Foxglove and signs off (or asks for changes). The agent
does **not** inspect the images — it reports only the `manifest.txt` text +
file paths, and Evan does the visual review.

## Open an episode

```bash
bash scripts/replay_foxglove.sh        # rsyncs all episodes from the GPU box, opens the first
# or open a specific one:
open -a Foxglove data/uav3d_critique/episode_001.mcap
```

## Panels + topics (set up once, save as a layout)

| Panel | Topic | Type | What it shows |
|-------|-------|------|---------------|
| Image | `/drone/pov/image` | `foxglove.RawImage` | The 224×224 drone POV — **the pixels LeWM trains on** |
| 3D | `/scene` | `foxglove.SceneUpdate` | Drone (blue) · turret (red) · obstacles (grey) · goal (green sphere) · aim-line |
| Log | `/state` | `foxglove.Log` | Per-step danger flags, syncs to the timeline |
| — | `/drone/tf` | `foxglove.FrameTransform` | Lets the 3D panel mount a camera on the drone |

3D legend: **aim-line = orange (safe) → RED (danger active)**. Scrub the timeline;
the Log panel warns and the aim-line goes red exactly when danger fires.

## What to judge (the actual gate)

This re-run addresses Evan's "still very jittery" critique and adds the
fall-after-shot recording. Three structural changes to verify:
**(a) body-frame 4D action** `(vx_fwd, vy_strafe, vz, yaw_rate)` — yaw is now an
EXPLICIT action (no auto-yaw), so the camera only turns when commanded;
**(b) drunken explorer** — the drone wanders (~70%) and provokes the turret
(~30%) instead of flying A→B, holding each maneuver 10–30 steps so the command
is smooth within a maneuver (no 10 Hz decision flips);
**(c) fall recording** — a shot drone is NOT terminated at the kill; thrust cuts
to a small random per-motor value (it falls + tumbles), and the episode ends
~1 s after it hits the ground.

**1. POV steadiness (most important — this was Evan's open critique)**
- Is the view **smooth** now — no left-right oscillation, no 10 Hz jitter? (The
  fix: explicit yaw + multi-step maneuver holding. Direction changes ~every 1–3 s
  are expected and fine; high-frequency shake is the failure mode.)
- Does the drone **strafe sideways without turning its heading**? (Body-frame
  strafe with `yaw_rate=0` — the camera should pan laterally, not rotate. The old
  auto-yaw would have yawed into the strafe; it must NOT now.)
- Does it look like a real drone first-person camera (forward-facing, 224²)?
- Is the **turret visible in frame when the drone approaches**, out of frame when
  fleeing/behind? (Core demo claim — the WM must imagine an occluded/behind threat.)

**2. Danger behavior (drunken explorer)**
- Does the drone **wander** the streets (cruise / strafe / spin / bank / hover)
  AND occasionally **provoke** — fly toward the turret, then **evade** (strafe +
  climb + spin) or **commit** (sustain LOS → killed)?
- Does the aim-line go red **only** when in-range AND aimed AND line-of-sight?
  (No false danger behind obstacles.)
- Do kills happen after ~0.4 s of sustained danger (not instant)?

**3. The fall (Evan's explicit request)**
- After a kill (4 danger frames, aim-line red), does the drone **fall and tumble**
  — the POV should spin/blur as it drops — then **come to rest** and hold still
  for ~1 s before the episode ends? (Tumble = small random per-motor thrust;
  rest = zero thrust on ground impact. The episode must NOT cut off at the kill.)

**4. Domain-randomization diversity (across episodes)**
- Do obstacle layouts, turret position, and drone spawn differ per episode? (v1 is
  geometric DR; light/texture DR is deferred — flag if textures look memorizable.)

**5. Outcome balance (read `manifest.txt`)**
- The drunken explorer does NOT pursue the goal, so `reached_goal` ≈ 0 is expected.
  Want a usable mix of `killed` (each with a recorded fall) / `timeout` (long
  wanders), and enough **danger frames** to later train the Phase-4 danger head.

**6. 3D plausibility**
- Does the drone actually fly in 3D (climb/descend), the turret track in yaw +
  pitch, and do obstacles physically block line-of-sight?

## Sign-off → next step

If the POV + danger behavior + DR look right: the lance training collector
(Task #12) runs on the GPU box to produce `uav_isaac_train.lance`, then Phase 3
training (`python scripts/train/lewm.py data=uav3d`).

If something's off (POV framing, danger logic, DR): tell the agent what to change
**before** any training — a re-collect is cheap; re-training is not.

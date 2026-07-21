# Live-demo showcase — WM predicts ahead of detection (Foxglove guide)

Two showcase modes, each telling one half of the claim — open the one that
matches the story you want to see:

- **`phantom` — "the WM predicts ahead of detection" (the core claim, validated).**
  The drone flies **oblivious and head-on** at the turret (no evasion), so the
  danger the WM imagines actually materializes. Both signals rise, and the WM's
  imagined danger *leads* the single-frame detector. **This is the mode to open
  for the `/signals` Plot panel story.** (Under the `planner`, the drone evades,
  the turret leaves the POV, and both signals stay low — so a planner MCAP does
  *not* show the lead; use `planner` for the survival story, not the lead story.)
- **`planner` — "acting on imagination saves you."** The danger-aware CEM planner
  flies the same forced-turret scenario; its green trail shows the evade and the
  episode `timeout`s (survives). Pair with `waypoint` (below) for the contrast.
- **`waypoint` — the oblivious baseline (flies into the turret, `killed`).** No
  planner, no detector — carries no `/signals`. The "what happens with no
  imagination" foil to `planner`.

Each episode is one Foxglove MCAP with the **world-model imagination signal and
the single-frame detector signal on the same timeline** (`phantom`/`planner`
only) — so the showcase claim ("the WM predicts danger the detector can't yet
see") is visible as the WM signal *leading* the detector signal.

The agent does **not** inspect the images. It text-ranks episodes with
`pick_showcase.py` (by how far `wm_danger` leads `det_logit`) and hands Evan the
top-ranked MCAP path for visual review.

## Open the showcase

```bash
bash scripts/replay_showcase.sh phantom   # THE core showcase: WM-leads-detector on /signals
bash scripts/replay_showcase.sh planner   # planner evades + survives (3D trail story)
bash scripts/replay_showcase.sh waypoint  # oblivious baseline (flies into the turret, killed)
bash scripts/replay_showcase.sh both      # planner + waypoint (the survival contrast)
```

## Panels + topics (set up once, save as a layout)

| Panel | Topic | Type | What it shows |
|-------|-------|------|---------------|
| **Plot** | `/signals` | `foxglove.PoseInFrame` | **THE showcase.** `position.x`=wm_danger, `position.y`=det_logit, `position.z`=danger vs time |
| Image | `/drone/pov/image` | `foxglove.RawImage` | 224×224 drone POV — what each signal was computed from |
| 3D | `/scene` | `foxglove.SceneUpdate` | drone (blue) · turret (red) · obstacles (grey) · aim-line · green trail |
| Log | `/state` | `foxglove.Log` | Per-step danger/in-range/los/aimed/fire flags, syncs to the timeline |
| — | `/drone/tf` | `foxglove.FrameTransform` | Lets the 3D panel mount a camera on the drone |

## The showcase panel: Plot on `/signals`

Foxglove's Plot panel graphs `pose.position.{x,y,z}` vs the message timestamp.
Add all three series, then overlay a horizontal line at `y = 1.53` — the
detector's calibrated best-F1 fire threshold (the level at which the
detector-reactive baseline would flee).

- **`position.x` = `wm_danger`** — the world model's *imagined* danger logit for a
  **forward-reference plan** (fly straight ahead along the drone's heading), rolled
  t+1..t+4 through predicted dynamics. This is the planner's forward-looking signal,
  and it is computed on a "keep flying ahead" plan *independent of* the evasive plan
  the CEM solver actually chose — so it rises as the drone closes on the turret
  whether or not the planner then evades. (The selected plan's own imagined danger
  stays low when the planner evades — it's the safe plan by construction — so it
  can't show "predicts ahead." The forward reference can.)
- **`position.y` = `det_logit`** — the Phase-4 present-frame danger probe logit
  (a single-frame detector on the *current* POV). This is what a standard
  detector sees — no prediction.
- **`position.z` = `danger`** — 0/1 ground truth (in-range & in-FOV & LOS & aimed).
  When this goes 1, the threat is real *right now*.

**The story the best episode tells:** `wm_danger` (x) rises while `det_logit`
(y) is still flat and well below 1.53 — the WM imagines the turret becoming
dangerous before it is visible as a present-frame threat. Then `det_logit`
catches up, and `danger` (z) goes 1. The **lead time** = the horizontal gap
between the x-rise and the y-rise (this is what `pick_showcase.py` ranks by).
In the 3D + Image panels you can see *why*: the drone is closing on the turret,
and the WM's rolled-forward latent anticipates the aim/LOS that the current
frame doesn't yet show.

### Sidecar (text preview, no Foxglove needed)
Each episode also has a `.signals.txt` (one line/step: `step wm_danger det_logit
danger fire dist_t`) and a `.outcome` (`killed`/`timeout`/`crash`). `pick_showcase.py`
reads these to rank — scan a dir without opening any MCAP:
```bash
python3 scripts/pick_showcase.py data/showcase_planner 10
```

## What to judge

1. **WM leads detector (the core claim — open a `phantom` MCAP).** In the Plot
   panel, does `wm_danger` (x) rise *before* `det_logit` (y) crosses 1.53, with
   `danger` (z) going 1 afterward? The bigger the lead, the stronger the
   showcase. (Use `phantom`, not `planner` — the planner evades so its `/signals`
   stay low; the phantom flies head-on so the lead is visible.)
2. **Planner evades (WM vs baseline — open `planner` + `waypoint`).** In a
   planner episode vs a waypoint episode on the same scenario: the planner's
   green trail should show a clear evade (strafe/climb away from the aim-line)
   and the episode should `timeout` (survive); the waypoint baseline flies
   straight through, the aim-line goes red, and it is `killed`. The planner MCAP
   carries `/signals`; the waypoint MCAP carries none (oblivious).
3. **Honesty check.** Some planner episodes will be `killed` anyway (the planner
   isn't perfect, n is small). That's fine — the *signal lead* is the claim, not
   a perfect survival rate. The matched survival numbers live in
   `briefs/phase7-live-demo.md`.

## Notes / limitations
- `wm_danger` and `det_logit` are logits on different heads (danger-head vs
  Phase-4 probe), so their *absolute* levels aren't directly comparable — the
  showcase is the **relative timing** of their rises, not their numeric overlap.
- **`wm_danger` is noisy, not a clean ramp.** The ViT-tiny's imagined embeddings
  aren't monotonically tied to closing distance, and the forward reference from
  ~20 m only reaches ~14 m in the imagined future — outside the turret's 8 m
  range — so the elevated early values are partly the danger head's noise on
  imagined futures, not a clean monotonic trend. The detector is cleaner
  (flat-negative then a monotonic rise at the end). The story holds — the WM is
  the only signal elevated *during the approach* — but expect a jagged `x` trace.
  A smoothed overlay (rolling mean) in the Plot panel would clarify it; the raw
  signal is logged as-is.
- The detector signal here is the *probe logit*, not the detector-mode flee
  decision; the flee threshold (1.53) is drawn as a reference line.
- v0.3.0 of `foxglove-schemas-protobuf` has no numeric-scalar schema, so the
  three signals ride in a `PoseInFrame` position (the Plot panel graphs
  `position.{x,y,z}`). This is why `/signals` is a Pose, not a scalar series.

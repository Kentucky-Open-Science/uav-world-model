# UAV World Model

**PLAN** — Predictive Latent Autonomous Navigation.

Imagination beats detection for drone danger. A world model that rolls its latent
forward sees a threat a single-frame detector can't, and acting on that imagination
keeps a drone alive.

Sim-only PoC. Synthetic urban environment, NVIDIA Isaac Sim 2.3.2, 2× RTX A6000.
Trained weights are not redistributed; produce them from source. Apache-2.0.

## Situation

A turret that is visible but not yet aimed is not yet dangerous. A turret about to
aim is dangerous before any single frame shows it. A detector reads one frame and
asks whether there is a threat now based on the latent representation of that frame. 
A world model thinks ahead in latent space, and asks whether it will be in danger in
the future if it follows it's current planned course.

Two results follow.

## Result 1 — the world model predicts ahead of detection

The drone flies oblivious and head-on at the turret so the danger it
imagines actually materializes. Both methods detect danger, but the WM leads.

![Drone POV with the live wm_danger vs det_logit trace — the WM signal leads](docs/assets/phantom_lead.gif)

Blue is imagined danger. Orange is the danger of the single-frame detector.
The WM signal stays elevated through the closing approach while the detector 
sits negative until the turret enters its ~8 m range.

Across 55 episodes, 42/55 (76%) show clean WM-lead separation before
danger; max lead ~17 s; best annotated episode (`env5_ep001`) leads by ~11.4 s.
However, the WM trace is noisy. The lead timing is the claim, not a clean ramp.

## Result 2 — acting on imagination saves you

A drone flies down a street toward a goal. A turret sits midway on the route.
Three controllers fly the same head-on approach, matched seeds, closed-loop in Isaac Sim.

![Top-down trails: oblivious (killed) vs planner (holds back, survives) vs detector (fires late, killed)](docs/assets/nav_a_to_b.gif)

Left, oblivious A→B: no imagination, flies the greedy route into the 8 m kill
zone, killed. Middle, WM planner A→B: imagines the turret from ~9 m and holds
back, never enters the zone, survives. Right, detector-reactive A→B: its probe
fires only once the turret is inside the kill zone, the flee can't break the lock,
killed.

Kill rates, matched n=12:

| Controller | Kill | Survival |
|---|---|---|
| Imagination planner | 0.000 | 1.000 |
| Oblivious A→B | 1.000 | 0.000 |
| Detector-reactive A→B | 0.917 | 0.083 |

The planner is the only controller that survives. It imagines the turret from ~9 m,
before the 8 m kill zone and ~1.6 s before the detector fires, so it holds rather
than presses in. Every planner episode sits at 8.5–10.2 m with zero danger frames.
The oblivious baseline flies to ~5.7 m and dies every time. The detector-reactive
baseline fires inside the kill zone (5.8–7.8 m), too late to escape, and dies 11/12.

Scope: the planner survives the approach; it does not reach the goal. It holds back
rather than detour, because a goal-reaching route around the block is one the
untrained WM cannot find by imagination alone. The claim is the contrast:
imagination avoids danger entirely where both reacting to detection and ignoring
detection die. Future work will focus on increasing the planning horizon and improving
the WM representation to be able to reason to actively pursue the goal while still 
prioritizing safety.

## How it works

The Isaac container (Python 3.11, the sim) and the host venv (Python 3.12, the
model) can't share a process, so the live demo runs as two processes on one box
over a localhost socket, with MCAP logged for Foxglove replay.

- Env: Isaac Lab `ManagerBasedRLEnv`. Native multirotor + 224×224 POV camera +
  domain randomization. Static turret yaws toward the drone at 110°/s, 8 m range,
  ±30° FOV.
- World model: LeWM, a JEPA-style latent predictor. A ViT-tiny encoder reads
  pixels only (state never feeds encode or predict); a Transformer predictor rolls
  the latent forward from context frames and an action sequence. A danger head
  scores imagined future latents, val AUROC 0.821.
- Planner: CEM over body-frame actions, 4 steps ahead. For each candidate, LeWM
  imagines the future latents, the danger head scores them, CEM minimizes max
  imagined danger. Imagination compounds the action through predicted physics; the
  detector is stuck on the present frame.
- Detector-reactive baseline: a single linear layer on the present-frame latent,
  val AUROC 0.776. When its logit crosses threshold the controller overrides the
  goal action with a vision-only flee. It sees the current frame only, so it
  reacts only once the turret is in-frame.

## How to run

Clone with the swm submodule (LeWM, SIGreg, stock CEM planner):

```bash
git clone --recursive https://github.com/Kentucky-Open-Science/uav-world-model.git
cd uav-world-model
git submodule update --init --recursive
```

Host training venv (Python 3.12 + torch). Editable-install swm, apply the local
overlays, train LeWM, then the danger readouts:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e repos/stable-worldmodel
cp patches/lewm.py    repos/stable-worldmodel/scripts/train/lewm.py
cp patches/uav.yaml   repos/stable-worldmodel/scripts/train/config/data/uav.yaml
cp patches/uav3d.yaml repos/stable-worldmodel/scripts/train/config/data/uav3d.yaml
bash scripts/run_train_uav3d.sh
```

Open the showcase in Foxglove (Mac, no GPU):

```bash
bash scripts/replay_showcase.sh phantom   # WM leads detector on /signals
bash scripts/replay_showcase.sh both      # planner vs oblivious
```

Regenerate the README figures:

```bash
python scripts/make_showcase_figures.py
```

Re-run the live demo (GPU box):

```bash
python scripts/live/planner_server.py --mode planner --num_envs 16 &
bash scripts/run_live_demo.sh planner 16 24 5557 /workspace/output/showcase_planner
```

## Limitations

- Sim-only, synthetic. A cuboid-grid city silhouette, not photoreal, no sim-to-real
  claim. Domain randomization keeps the WM from memorizing one scene.
- Small n, contrived. The A→B head-on is n=12; it places the turret in the drone's
  path to isolate the hold-back mechanism, not to measure general navigation. The
  planner survives by holding back, not by reaching the goal.
- The WM signal is noisy. Lead timing is robust; the raw trace is not a clean ramp.
- The detector-reactive baseline's best operating point is degenerate: at threshold
  0 it fires ~57% of frames and survives by climbing out of reach. At every honest
  operating point the detector is no better than oblivious; only the planner stays
  safe without degenerating.

## Stack

NVIDIA Isaac Sim / Isaac Lab 2.3.2, native multirotor, Ubuntu 24.04, 2× RTX A6000.
World model: vendored `stable_worldmodel` (LeWM, JEPA + SIGreg), ViT-tiny encoder.
Planner: swm stock CEM with a custom danger cost. Viz: Foxglove MCAP replay;
matplotlib and Pillow for the figures.

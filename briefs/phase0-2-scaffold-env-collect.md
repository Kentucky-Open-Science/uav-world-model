# Phases 0–2 — Scaffold, 2D synthetic env, 2D collect (2026-07)

> Retrospective combined brief. Phases 0–2 are the **2D top-down pygame foundation**; they were **superseded by the 3D Isaac Sim work** (Phases 3+) but remain the fast Mac smoke test (`scripts/smoke_env.py` against `swm/UAVTurret-v0`). Recorded here so the 2D dataset/env facts have a home.

## Phase 0 — Scaffold + swm smoke
- Vendored `stable_worldmodel` (LeWM — JEPA latent prediction + SIGreg) into `repos/stable-worldmodel/`; `uav_wm/` package + `scripts/`.
- Got the stock swm smoke running on the Mac to confirm the training stack loads end-to-end before any custom env.

## Phase 1 — 2D synthetic `UAVTurretEnv`
- `uav_wm/envs/uav_turret_env.py`: top-down pygame `gymnasium` env. A drone navigates a 2D arena with a turret that yaws toward it with finite yaw rate; `danger = in_range & in_fov & los & aimed`; kill after sustained LOS+aim. Registered `swm/UAVTurret-v0`.
- **State (16-dim)** with `danger` at a fixed index; **action (2-dim)** = 2D velocity. This is where the `state[DANGER_IDX]=danger` convention was established (carried forward to 3D as `state[13]`).

## Phase 2 — 2D collect
- `scripts/collect_uav.py` + `ExplorationPolicy` (3-mode reactive: safe→approach/goal, near→evade, danger→flee-perpendicular) via `swm.World.collect` (which `LanceWriter` writes behind).
- Dataset: **2000 episodes / 134,584 frames / 13.8% danger**, at `$STABLEWM_HOME/datasets/uav_turret_train.lance` (`~/.stable_worldmodel/datasets/`).
- 2D training entry (for reference, now superseded by 3D `data=uav3d`): `python scripts/train/lewm.py data=uav` from `repos/stable-worldmodel`; loads `num_steps=4, frameskip=5, keys_to_load=[pixels,action,state]`; `get_dim('action')=2`→`action_encoder.input_dim=10`; batch shapes pixels `(4,3,224,224)`, action `(4,10)`, state `(4,16)`.

## Supersession note
The 2D env/collect were the proving ground for the swm integration (LanceWriter schema, `load_dataset`, training loop). The **3D Isaac Sim** env (`UAVTurret3D`, `uav_wm/envs/uav_turret_3d.py`) replaced both: native multirotor physics, real drone-POV camera, domain randomization, 4-dim body-frame action (`action_encoder.input_dim=20`), 21-dim state. See `phase3-training.md` onward. The 2D env is retained **only** as a Mac-side smoke test (no Isaac needed).

"""Phase 0 smoke: validate UAVTurretEnv mechanics + the swm data pipeline.

Run from the project root:
    .venv/bin/python scripts/smoke_env.py

Checks:
  1. Standalone env steps + renders; a "fly-straight-to-goal" policy usually
     gets shot (validates turret tracking + LOS + kill).
  2. swm World collects a tiny lance dataset from swm/UAVTurret-v0.
  3. The dataset loads and exposes pixels + action (+ state).
Saves sample frames to outputs/smoke/.
"""
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # so `import uav_wm` works without install

import uav_wm  # noqa: F401  (registers swm/UAVTurret-v0)
import stable_worldmodel as swm

OUT = ROOT / "outputs" / "smoke"
OUT.mkdir(parents=True, exist_ok=True)


def save_frame(img, name):
    from PIL import Image
    Image.fromarray(img).save(OUT / name)
    print(f"  saved {name}  {img.shape} {img.dtype}")


def _run_policy(env, policy_fn, name, max_steps=200):
    obs, info = env.reset(seed=0)
    danger_seen = 0
    first_danger = None
    for t in range(max_steps):
        act = policy_fn(obs, info)
        obs, reward, term, trunc, info = env.step(act)
        if info["danger"] > 0:
            danger_seen += 1
            if first_danger is None:
                first_danger = t
                save_frame(env.render(), f"frame_{name}_danger.png")
        if term or trunc:
            break
    save_frame(env.render(), f"frame_{name}_end.png")
    killed = bool(info["killed"])
    reached = bool(info["reached_goal"])
    print(f"  [{name}] steps={t+1} killed={killed} reached={reached} "
          f"danger_frames={danger_seen} first_danger={first_danger}")
    return killed, reached, danger_seen


def test_standalone():
    print("\n=== 1. standalone env mechanics ===")
    import gymnasium as gym
    env = gym.make("swm/UAVTurret-v0", max_episode_steps=200)
    obs, info = env.reset(seed=0)
    save_frame(env.render(), "frame_reset.png")
    print(f"  obs.shape={obs.shape}  danger={info['danger']:.0f}  "
          f"drone={info['drone_pos']} turret={info['turret_pos']}")

    # Fly straight AT the turret -> must get shot (validates tracking+LOS+kill).
    def fly_at_turret(obs, info):
        rel = info["turret_pos"] - info["drone_pos"]
        return np.clip(rel / 3.0, -1, 1).astype(np.float32)

    killed, _, danger = _run_policy(env, fly_at_turret, "atturret")
    assert danger > 0 and killed, "flying at the turret did not get shot -- kill mechanic broken"

    # Fly straight at the goal (correct action) -> should make progress.
    def fly_at_goal(obs, info):
        rel = info["goal_state"] * 10.0 - info["drone_pos"]
        return np.clip(rel / 3.0, -1, 1).astype(np.float32)

    killed, reached, danger = _run_policy(env, fly_at_goal, "atgoal")
    assert reached or killed, "drone neither reached goal nor got shot -- action/stuck bug"
    env.close()


def test_collect():
    print("\n=== 2. swm World.collect (tiny lance dataset) ===")
    world = swm.World(
        "swm/UAVTurret-v0",
        num_envs=4,
        image_shape=(224, 224),
        max_episode_steps=100,
    )
    world.set_policy(swm.policy.RandomPolicy(seed=0))
    ds_path = OUT / "uav_smoke.lance"
    if ds_path.exists():
        import shutil
        shutil.rmtree(ds_path)
    world.collect(ds_path, episodes=12, seed=0)
    print(f"  collected -> {ds_path}")
    print(f"  infos['pixels'].shape = {world.infos['pixels'].shape}")
    world.close()


def test_load():
    print("\n=== 3. load dataset ===")
    ds = swm.data.load_dataset(
        str(OUT / "uav_smoke.lance"),
        num_steps=4,
        keys_to_load=["pixels", "action", "state"],
    )
    print(f"  type={type(ds).__name__}")
    # datasets yield dict batches; grab one
    try:
        batch = ds[0]
    except Exception:
        batch = next(iter(ds))
    if not isinstance(batch, dict):
        batch = batch[0] if isinstance(batch, (list, tuple)) else batch
    for k, v in (batch.items() if isinstance(batch, dict) else [("batch", batch)]):
        try:
            print(f"  {k}: shape={getattr(v,'shape',v)}")
        except Exception:
            print(f"  {k}: {type(v)}")


if __name__ == "__main__":
    test_standalone()
    test_collect()
    test_load()
    print("\nSMOKE OK")

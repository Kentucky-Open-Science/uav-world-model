"""Collect synthetic UAV+turret training data into a swm lance dataset.

Default target is the swm datasets cache dir (bare-name resolution), so the
training config can reference it portably as `name: uav_turret_train.lance`.

Examples (from project root):
    .venv/bin/python scripts/collect_uav.py --episodes 40          # quick yield check
    .venv/bin/python scripts/collect_uav.py --episodes 1000        # full collect
    .venv/bin/python scripts/collect_uav.py --inspect-only uav_turret_train.lance
"""
import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # so `import uav_wm` works without install

import uav_wm  # noqa: F401  (registers swm/UAVTurret-v0)
import stable_worldmodel as swm
from stable_worldmodel.data.utils import get_cache_dir

from uav_wm.data.policies import ExplorationPolicy


def _dataset_path(name):
    return get_cache_dir(sub_folder="datasets") / name


def collect(episodes, num_envs, seed, name, max_steps, q, r, noise,
            image_shape=(224, 224)):
    ds_path = _dataset_path(name)
    if ds_path.exists():
        shutil.rmtree(ds_path)
    print(f"collecting {episodes} episodes (num_envs={num_envs}) -> {ds_path}")
    world = swm.World(
        "swm/UAVTurret-v0",
        num_envs=num_envs,
        image_shape=image_shape,
        max_episode_steps=max_steps,
    )
    world.set_policy(ExplorationPolicy(seed=seed, q=q, r=r, noise=noise))
    world.collect(ds_path, episodes=episodes, seed=seed)
    world.close()
    print("collect done.")
    inspect(name)


def inspect(name):
    import lance

    ds = lance.dataset(str(_dataset_path(name)))
    want = ["episode_idx", "danger", "killed", "reached_goal",
            "terminated", "truncated", "state"]
    have = [c for c in want if c in ds.schema.names]
    tbl = ds.to_table(columns=have)

    ep = np.asarray(tbl.column("episode_idx").to_pylist())
    n_frames = ds.count_rows()
    n_ep = int(ep.max()) + 1 if len(ep) else 0
    counts = np.bincount(ep) if len(ep) else np.array([0])

    danger = (np.asarray(tbl.column("danger").to_pylist()) > 0) \
        if "danger" in have else None
    killed = (np.asarray(tbl.column("killed").to_pylist()) > 0) \
        if "killed" in have else None
    reached = (np.asarray(tbl.column("reached_goal").to_pylist()) > 0) \
        if "reached_goal" in have else None

    print(f"\n=== inspect {name} ===")
    print(f"  episodes : {n_ep}")
    print(f"  frames   : {n_frames}  (avg len {n_frames / max(n_ep, 1):.1f}, "
          f"min {int(counts.min())} max {int(counts.max())} "
          f"median {int(np.median(counts))})")
    if danger is not None:
        print(f"  danger frames : {int(danger.sum()):6d} "
              f"({100 * danger.mean():.1f}% of frames)")
    if killed is not None:
        print(f"  killed eps    : {int(killed.sum()):6d} "
              f"({100 * killed.sum() / max(n_ep, 1):.1f}% of eps)")
    if reached is not None:
        print(f"  reached eps   : {int(reached.sum()):6d} "
              f"({100 * reached.sum() / max(n_ep, 1):.1f}% of eps)")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--num-envs", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--name", default="uav_turret_train.lance")
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--q", type=float, default=0.3, help="P(approach|safe)")
    p.add_argument("--r", type=float, default=0.3, help="P(commit|danger)")
    p.add_argument("--noise", type=float, default=0.1)
    p.add_argument("--inspect-only", default=None,
                   help="skip collect; just inspect an existing lance name")
    args = p.parse_args()

    if args.inspect_only:
        inspect(args.inspect_only)
        return

    collect(args.episodes, args.num_envs, args.seed, args.name, args.max_steps,
            args.q, args.r, args.noise)


if __name__ == "__main__":
    main()

"""Smoke: confirm native multirotor + the ARL drone task run headless in the container.

This is the native-multirotor gate (Task #10). It boots the Kit app via
AppLauncher, instantiates the stock ARL ``TrackPositionNoObstacles`` task (which
uses ``isaaclab_contrib.assets.Multirotor`` + ``ThrustActionCfg``), inspects the
multirotor API we depend on, and steps the env a few times.

Run inside the isaac-lab container (script is mounted from the host):
    /workspace/isaaclab/isaaclab.sh -p /workspace/scripts/smoke_native_multirotor.py \
        --headless --num_envs 4
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Native multirotor + ARL task smoke.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of envs.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest of imports must come after AppLauncher (pxr/omni only exist post-startup)."""
import gymnasium as gym
import isaaclab_tasks  # noqa: F401  (registers gym ids)
import torch
from isaaclab_tasks.utils import parse_env_cfg

TASK = "Isaac-TrackPositionNoObstacles-ARL-Robot-1-v0"


def main() -> None:
    env_cfg = parse_env_cfg(TASK, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(TASK, cfg=env_cfg)

    robot = env.unwrapped.scene["robot"]
    print(f"[SMOKE] robot class: {type(robot).__name__}")
    print(f"[SMOKE] num_thrusters: {robot.num_thrusters}")
    print(f"[SMOKE] has set_thrust_target: {hasattr(robot, 'set_thrust_target')}")
    # allocation_matrix is a property on the Multirotor class (not .data).
    alloc = robot.allocation_matrix
    print(f"[SMOKE] allocation_matrix shape: {tuple(alloc.shape)}")
    assert type(robot).__name__ == "Multirotor", f"expected Multirotor, got {type(robot)}"
    assert robot.num_thrusters == 4, f"expected 4 thrusters, got {robot.num_thrusters}"
    assert hasattr(robot, "set_thrust_target"), "Multirotor missing set_thrust_target"
    assert tuple(alloc.shape) == (6, 4), f"expected (6,4) allocation, got {alloc.shape}"

    obs, _ = env.reset()
    print(f"[SMOKE] obs keys: {list(obs.keys())}")
    for k, v in obs.items():
        print(f"[SMOKE]   {k}: {tuple(v.shape)}")

    total_rew = torch.zeros(env.unwrapped.num_envs, device=env.unwrapped.device)
    for i in range(10):
        # Isaac Lab vec-env step expects a torch tensor; action_space.sample()
        # returns numpy, so convert (else `action.to(device)` raises AttributeError).
        act = env.action_space.sample()
        act = torch.as_tensor(act, dtype=torch.float32, device=env.unwrapped.device)
        obs, rew, term, trunc, info = env.step(act)
        total_rew += rew.float()
    print(f"[SMOKE] stepped 10x | mean reward: {float(total_rew.mean()):.4f} | "
          f"term sum: {int(term.sum())} | trunc sum: {int(trunc.sum())}")

    # Optional controller inputs (resolved after the step-loop so a wrong guess
    # can't mask the step-loop result). default_mass: (num_instances, num_bodies).
    try:
        mass = float(robot.data.default_mass[0].sum())
        print(f"[SMOKE] total mass (env 0): {mass:.4f} kg")
    except Exception as e:
        print(f"[SMOKE] mass lookup skipped: {e}")

    env.close()
    print("[SMOKE] NATIVE MULTIROTOR + ARL TASK OK")


if __name__ == "__main__":
    main()
    simulation_app.close()

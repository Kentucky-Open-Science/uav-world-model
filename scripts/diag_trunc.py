"""Diagnose whether uav_time_out truncation fires at the 300-step cap.

Boots UAVTurret3D, hovers all envs with a zero command (no turret approach ->
no kill), and prints episode_length_buf / terminated / truncated / drone_z /
danger from step 295..310. If truncation works: at the cap, trunc=1, the env
auto-resets (ep_len_buf jumps back to 300, z back to ~spawn). If broken:
ep_len_buf goes <= 0 (negative) and trunc stays 0, and the episode runs on.

Run in the isaac-lab container (mirrors run_smoke_uav3d.sh mounts):
    /workspace/isaaclab/isaaclab.sh -p /workspace/scripts/diag_trunc.py \
        --headless --enable_cameras --num_envs 4
"""
import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=4)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import uav_wm.envs  # noqa: E402,F401  (registers Isaac-UAVTurret3D-v0)
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

TASK = "Isaac-UAVTurret3D-v0"
env = gym.make(TASK, cfg=parse_env_cfg(TASK, device=args_cli.device, num_envs=args_cli.num_envs))
env = env.unwrapped
n = env.num_envs
dev = env.device

obs, _ = env.reset()
print(f"[DIAG] num_envs={n} step_dt={env.step_dt} max_episode_length={env.max_episode_length}")
hover = torch.zeros(n, 4, device=dev)
for g in range(312):
    obs, rew, term, trunc, info = env.step(hover)
    if g >= 294:
        elb = int(env.episode_length_buf[0].item())
        z = float(env.scene["robot"].data.root_pos_w[0, 2].item() - env.scene.env_origins[0, 2].item())
        d = int(obs["state"][0, 13].item())
        fire = float(obs["state"][0, 17].item())
        print(f"[DIAG] step {g+1:3d} ep_len_buf={elb:5d} term={int(term[0])} "
              f"trunc={int(trunc[0])} z={z:.2f} danger={d} fire={fire:.2f}")
print("[DIAG] DONE")
env.close()
simulation_app.close()

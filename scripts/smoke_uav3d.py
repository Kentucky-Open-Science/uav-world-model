"""Smoke: UAVTurret3D env constructs, steps, renders POV, danger fires.

Gate for Task #11. Boots AppLauncher (--headless --enable_cameras), registers +
makes ``Isaac-UAVTurret3D-v0``, resets, steps ~60 times under a scripted
velocity command, prints pixel/state shapes and the danger/fire progression,
and saves a few drone-POV PNGs to the bind-mounted output dir for Evan to
inspect (the agent does not inspect raw images — text-only rule).

Run inside the isaac-lab container (script mounted from host):
    /workspace/isaaclab/isaaclab.sh -p /workspace/scripts/smoke_uav3d.py \
        --headless --enable_cameras --num_envs 4 --output_dir /workspace/output/uav3d_smoke
"""
import argparse
import os
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="UAVTurret3D env smoke.")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--output_dir", type=str, default="/workspace/output/uav3d_smoke")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Imports after AppLauncher (pxr/omni only exist post-startup)."""
import gymnasium as gym
import math
import torch

import uav_wm.envs  # noqa: F401  (registers Isaac-UAVTurret3D-v0)
from isaaclab_tasks.utils import parse_env_cfg

TASK = "Isaac-UAVTurret3D-v0"
DANGER_IDX = 13


def main() -> None:
    out = Path(args_cli.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    env_cfg = parse_env_cfg(TASK, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(TASK, cfg=env_cfg)
    print(f"[SMOKE] env: {type(env.unwrapped).__name__} num_envs={env.unwrapped.num_envs}")

    obs, _ = env.reset()
    print(f"[SMOKE] obs keys: {list(obs.keys())}")
    for k, v in obs.items():
        print(f"[SMOKE]   {k}: shape={tuple(v.shape)} dtype={v.dtype}")

    # --- flight diagnostics: what thrust is actually delivered, and is cfg.dt set? ---
    robot = env.unwrapped.scene["robot"]
    thr = robot.actuators["thrusters"]
    try:
        rpv = robot.root_physx_view
        masses = rpv.get_masses()
        rmass = float(masses[0].sum()) if masses is not None else float("nan")
    except Exception as e:
        rmass = float("nan")
    print(f"[DIAG] thrusters.cfg.dt={thr.cfg.dt}")
    print(f"[DIAG] root mass={rmass:.4f} kg  hover_thrust/motor={rmass*9.81/4:.4f} N  thrust_range={tuple(float(x) for x in thr.thrust_r)}")
    print(f"[DIAG] init curr_thrust[0]={thr.curr_thrust[0].cpu().tolist()}  sum={float(thr.curr_thrust[0].sum()):.3f} N (hover needs {rmass*9.81:.3f} N total)")

    # post-reset pose: was the drone kicked/spun by the init thrust?
    pos0 = robot.data.root_pos_w[0].cpu().tolist()
    vel0 = robot.data.root_lin_vel_w[0].cpu().tolist()
    w0 = robot.data.root_ang_vel_b[0].cpu().tolist()
    q0 = robot.data.root_quat_w[0].cpu().tolist()
    yaw0 = math.atan2(2.0 * (q0[0] * q0[3] + q0[1] * q0[2]), 1.0 - 2.0 * (q0[2] ** 2 + q0[3] ** 2))
    print(f"[DIAG] post-reset pos0=[{pos0[0]:+.3f},{pos0[1]:+.3f},{pos0[2]:+.3f}] "
          f"vel0=[{vel0[0]:+.3f},{vel0[1]:+.3f},{vel0[2]:+.3f}] yaw0={yaw0:+.3f} "
          f"angvel_b=[{w0[0]:+.3f},{w0[1]:+.3f},{w0[2]:+.3f}]")

    # confirm pixels + state shapes
    pix = obs["policy"]
    st = obs["state"]
    print(f"[SMOKE] pixels {tuple(pix.shape)} {pix.dtype} | state {tuple(st.shape)} {st.dtype}")
    assert st.shape[-1] == 21, f"expected state dim 21, got {st.shape[-1]}"
    assert int(st[0, DANGER_IDX]) in (0, 1), "danger should be 0/1"

    # scripted BODY-frame 4D command (vx_fwd, vy_strafe, vz_climb, yaw_rate):
    # seg1 forward+climb, seg2 strafe+climb with yaw_rate=0 (heading must HOLD --
    # the old auto-yaw would have yawed into the strafe; explicit yaw keeps it
    # steady), seg3 reverse + yaw spin (exercises the explicit yaw axis).
    n = env.unwrapped.num_envs
    dev = env.unwrapped.device
    saved = 0
    for i in range(60):
        if i < 20:
            cmd = torch.tensor([0.6, 0.0, 0.2, 0.0], device=dev).expand(n, 4)
        elif i < 40:
            cmd = torch.tensor([0.0, 0.5, 0.3, 0.0], device=dev).expand(n, 4)
        else:
            cmd = torch.tensor([-0.4, 0.0, 0.0, 0.6], device=dev).expand(n, 4)
        cmd = cmd + 0.05 * torch.randn(n, 4, device=dev)  # a little jitter
        obs, rew, term, trunc, info = env.step(cmd.float())
        if i % 10 == 0 or i < 5:
            s = obs["state"][0]
            dp = s[0:3].cpu().tolist()
            ap = thr.applied_thrust[0].cpu().tolist()  # actually-delivered per-motor thrust
            ct = float(thr.curr_thrust[0].sum())  # pre-clip total
            # thrust_target = the des_thrust the actuator's compute() received (should
            # equal the controller's output). If it differs from `applied`, the actuator
            # model is the culprit; if it differs from the CTRLDIAG command, something
            # overwrites _data.thrust_target between apply_actions and write_data_to_sim.
            tt = robot.data.thrust_target[0].cpu().tolist() if hasattr(robot.data, "thrust_target") else None
            qq = robot.data.root_quat_w[0].cpu().tolist()
            wz = float(robot.data.root_ang_vel_b[0, 2])
            yaw = math.atan2(2.0 * (qq[0] * qq[3] + qq[1] * qq[2]), 1.0 - 2.0 * (qq[2] ** 2 + qq[3] ** 2))
            vel = robot.data.root_lin_vel_w[0].cpu().tolist()
            print(f"[SMOKE] step {i:2d} drone_pos=[{dp[0]:+.2f},{dp[1]:+.2f},{dp[2]:+.2f}] "
                  f"dist_t={float(s[12]):.2f} danger={int(s[13])} in_range={int(s[14])} "
                  f"los={int(s[15])} aimed={int(s[16])} fire={float(s[17]):.2f}")
            print(f"[DIAG]  vel=[{vel[0]:+.2f},{vel[1]:+.2f},{vel[2]:+.2f}] yaw={yaw:+.2f} wz={wz:+.2f} "
                  f"target=[{','.join(f'{x:.2f}' for x in tt) if tt else 'NA'}] "
                  f"applied=[{','.join(f'{x:.2f}' for x in ap)}] sum={sum(ap):.2f}N  (hover~{rmass*9.81:.2f}N)")
        # save 3 POV frames from env 0 (early/mid/late) for Evan to inspect
        if i in (2, 30, 58) and saved < 3:
            _save_png(pix_from_obs(obs["policy"], 0), out / f"pov_env0_step{i:02d}.png")
            saved += 1

    print(f"[SMOKE] saved {saved} POV PNGs to {out} (for Evan to inspect)")
    term_sum = int(term.sum())
    print(f"[SMOKE] terminations: {term_sum}/{n} | (kills/crashes/goals expected)")
    print("[SMOKE] UAVTURRET3D OK")
    env.close()


def pix_from_obs(pix_obs: torch.Tensor, env_idx: int) -> "torch.Tensor":
    """Return env_idx's frame as (H, W, 3) uint8, handling (N,C,H,W) or (N,H,W,C)."""
    import torch as _t

    fr = pix_obs[env_idx]
    if fr.dim() == 3 and fr.shape[0] <= 4:  # (C, H, W)
        fr = fr.permute(1, 2, 0)
    if fr.dtype != _t.uint8:
        fr = (fr.clamp(0.0, 1.0) * 255.0).to(_t.uint8)
    return fr.contiguous()


def _save_png(frame: torch.Tensor, path: Path) -> None:
    # Best-effort: PIL may be absent in the container; the env working is the
    # real signal. Never let a missing preview crash the smoke.
    try:
        import numpy as np
        from PIL import Image

        arr = frame.cpu().numpy().astype("uint8")
        Image.fromarray(arr).save(str(path))
        print(f"[SMOKE]   wrote {path} ({arr.shape})")
    except Exception as e:
        print(f"[SMOKE]   (preview PNG skipped: {e})")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # Print the traceback BEFORE simulation_app.close() — Kit's shutdown
        # tears down carb logging and can swallow a pending Python traceback,
        # leaving a silent exit=0 (isaaclab.sh -p doesn't propagate exit codes).
        import traceback
        traceback.print_exc()
        print("[SMOKE] CRASHED (see traceback above)", flush=True)
    finally:
        simulation_app.close()

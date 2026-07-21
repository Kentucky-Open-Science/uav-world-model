"""Diagnose whether the drone POV camera actually follows the drone's pose.

Evan's critique: the POV renders from a FIXED perspective, not the drone's.
This script boots UAVTurret3D (1 env), steps it under a forward+climb command,
and prints -- per step -- the drone world pose vs the CAMERA world pose, plus
the camera's parent prim path, the robot's root link name, and the SHA256 of
each rendered POV frame.

The trace is decisive and text-only (we compare hashes/poses, never inspect the
images, per Evan's rule). NOTE: CameraCfg.update_latest_camera_pose defaults to
False, so cam.data.pos_w is STALE unless we force it on (done below). With it on:
  * cam.data.pos_w TRACKS robot.data.root_pos_w (small constant delta ~offset)
    AND POV SHA256s differ across steps
        => camera IS body-mounted correctly (first-person). Fix confirmed.
  * cam.data.pos_w stays at offset.pos (origin) while the drone moves
        => camera prim's world transform is NOT composing with base_link --
           prim_path parenting alone is insufficient; need to drive the pose
           manually each step (cam.set_world_pose) or reparent differently.
  * pose tracks BUT all POV SHA256s identical
        => stale render (sensor not refreshed each step).

Run inside the isaac-lab container (script mounted from host):
    /workspace/isaaclab/isaaclab.sh -p /workspace/scripts/diag_camera.py \
        --headless --enable_cameras --num_envs 1 --output_dir /workspace/output/uav3d_diag
"""
import argparse
import hashlib
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="UAVTurret3D POV camera pose diagnostic.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--output_dir", type=str, default="/workspace/output/uav3d_diag")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Imports after AppLauncher (pxr/omni only exist post-startup)."""
import gymnasium as gym  # noqa: E402
import math  # noqa: E402
import torch  # noqa: E402

import uav_wm.envs  # noqa: F401,E402  (registers Isaac-UAVTurret3D-v0)
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

TASK = "Isaac-UAVTurret3D-v0"


def _yaw(q):
    """q = (w, x, y, z)."""
    return math.atan2(2.0 * (q[0] * q[3] + q[1] * q[2]), 1.0 - 2.0 * (q[2] ** 2 + q[3] ** 2))


def _sha(arr) -> str:
    return hashlib.sha256(arr.tobytes()).hexdigest()[:12]


def _frame_uint8_hwc(pix_obs, env_idx):
    fr = pix_obs[env_idx]
    if fr.dim() == 3 and fr.shape[0] <= 4:  # (C, H, W)
        fr = fr.permute(1, 2, 0)
    if fr.dtype != torch.uint8:
        fr = (fr.clamp(0.0, 1.0) * 255.0).to(torch.uint8)
    return fr.cpu().numpy().astype("uint8")


def main() -> None:
    out = Path(args_cli.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    env_cfg = parse_env_cfg(TASK, device=args_cli.device, num_envs=args_cli.num_envs)
    # CRITICAL for the diagnostic: CameraCfg.update_latest_camera_pose defaults to
    # False, so cam.data.pos_w is only set at reset and goes STALE -- useless for
    # checking whether the camera follows the drone. Force it on so pos_w is live.
    env_cfg.scene.tiled_camera.update_latest_camera_pose = True
    env = gym.make(TASK, cfg=env_cfg)
    base = env.unwrapped
    robot = base.scene["robot"]
    cam = base.scene["tiled_camera"]

    print(f"[DIAG] camera.cfg.prim_path        = {cam.cfg.prim_path}")
    print(f"[DIAG] camera.cfg.offset.convention= {cam.cfg.offset.convention}")
    print(f"[DIAG] camera.cfg.offset.pos       = {cam.cfg.offset.pos}")
    print(f"[DIAG] camera.cfg.offset.rot       = {cam.cfg.offset.rot}")
    # authoritative body-link list (index 0 = root body the camera should mount to)
    for attr in ("body_names",):
        v = getattr(robot.data, attr, None) or getattr(robot, attr, None)
        if v is not None:
            print(f"[DIAG] robot body_names = {v}  (index 0 = root body)")
            break
    else:
        print("[DIAG] robot body_names: not found on .data or robot")
    # any camera-internal prim/path bookkeeping the version exposes
    for attr in ("_data", "data"):
        d = getattr(cam, attr, None)
        if d is None:
            continue
        for k in ("frame_prim_paths", "target_frame_prim_paths", "parent_prim_paths"):
            v = getattr(d, k, None)
            if v is not None:
                try:
                    print(f"[DIAG] camera.{attr}.{k} = {v}")
                except Exception:
                    print(f"[DIAG] camera.{attr}.{k} = <unprintable>")

    obs, _ = env.reset()
    n = base.num_envs
    dev = base.device

    print("\n[DIAG] step | drone_pos(xyz) yaw | cam_pos(xyz) | cam-drone delta | pov_sha mean_rgb | turret_pos")
    saved = set()
    for i in range(45):
        if i < 15:
            cmd = torch.tensor([0.6, 0.0, 0.2], device=dev).expand(n, 3)
        elif i < 30:
            cmd = torch.tensor([0.0, 0.5, 0.3], device=dev).expand(n, 3)
        else:
            cmd = torch.tensor([-0.4, 0.0, 0.0], device=dev).expand(n, 3)
        obs, rew, term, trunc, info = env.step(cmd.float())

        if i in (2, 12, 22, 32, 42):
            dp = robot.data.root_pos_w[0].cpu().tolist()
            dq = robot.data.root_quat_w[0].cpu().tolist()
            cp = cam.data.pos_w[0].cpu().tolist()
            delta = [cp[k] - dp[k] for k in range(3)]
            arr = _frame_uint8_hwc(obs["policy"], 0)
            sha = _sha(arr)
            m = arr.reshape(-1, 3).mean(axis=0)
            tp = base.turret_pos[0].cpu().tolist() if hasattr(base, "turret_pos") else [0, 0, 0]
            print(f"[DIAG] {i:3d} | d=[{dp[0]:+7.2f},{dp[1]:+7.2f},{dp[2]:+6.2f}] {_yaw(dq):+5.2f} "
                  f"| c=[{cp[0]:+7.2f},{cp[1]:+7.2f},{cp[2]:+6.2f}] "
                  f"| dc=[{delta[0]:+5.2f},{delta[1]:+5.2f},{delta[2]:+5.2f}] "
                  f"| {sha} ({m[0]:.0f},{m[1]:.0f},{m[2]:.0f}) "
                  f"| t=[{tp[0]:+5.1f},{tp[1]:+5.1f},{tp[2]:+.1f}]", flush=True)

        if i in (5, 25, 44) and i not in saved:
            arr = _frame_uint8_hwc(obs["policy"], 0)
            try:
                from PIL import Image
                Image.fromarray(arr).save(str(out / f"pov_step{i:02d}.png"))
                print(f"[DIAG]   saved pov_step{i:02d}.png sha={_sha(arr)}", flush=True)
            except Exception as e:
                print(f"[DIAG]   (png skipped: {e})")
            saved.add(i)

    print("[DIAG] UAVTURRET3D DIAG OK")
    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # Print the traceback BEFORE simulation_app.close() -- Kit shutdown can
        # swallow a pending Python traceback (isaaclab.sh -p doesn't propagate).
        import traceback
        traceback.print_exc()
        print("[DIAG] CRASHED (see traceback above)", flush=True)
    finally:
        simulation_app.close()

"""Diagnostic: import uav_turret_3d with the real traceback (no swallow).

The env __init__ guards registration in try/except, so an import error vanishes
and gym just says "env doesn't exist". This boots the app and imports the module
directly so the real error surfaces.
"""
import sys
import traceback

from isaaclab.app import AppLauncher

parser = AppLauncher.add_app_launcher_args.__self__ if False else None
import argparse

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

steps = [
    ("isaaclab_contrib.assets.MultirotorCfg", "from isaaclab_contrib.assets import MultirotorCfg"),
    ("isaaclab_assets.arl_robot_1", "from isaaclab_assets.robots.arl_robot_1 import ARL_ROBOT_1_CFG"),
    ("uav_wm.envs.uav_turret_3d", "from uav_wm.envs.uav_turret_3d import UAVTurret3DEnvCfg"),
    ("gym registration", "import uav_wm.envs; import gymnasium as gym; print(gym.spec('Isaac-UAVTurret3D-v0'))"),
]
for label, stmt in steps:
    try:
        exec(stmt)
        print(f"[DIAG] OK  {label}")
    except Exception:
        print(f"[DIAG] FAIL {label}:")
        traceback.print_exc()

app.close()
print("[DIAG] DONE")

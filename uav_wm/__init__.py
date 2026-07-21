"""uav_wm — UAV world-model danger-reasoning PoC built on stable-worldmodel.

Importing this package registers the 2D swm environment, so do
`import uav_wm` before `swm.World("swm/UAVTurret-v0", ...)` or the `swm` CLI.
The 3D Isaac env is registered by `uav_wm.envs` when Isaac Lab is importable.

Both registrations are guarded: on the Isaac Lab container (no
stable_worldmodel/pygame) the 2D path skips and only the 3D env registers;
on the Mac (no Isaac) the 2D env registers and the 3D path skips.
"""
try:
    from stable_worldmodel.envs import register

    from uav_wm.envs import UAVTurretEnv

    register(id="swm/UAVTurret-v0", entry_point=UAVTurretEnv)

    __all__ = ["UAVTurretEnv"]
except Exception:
    # stable_worldmodel / 2D pygame env not available (Isaac Lab container) —
    # the 3D Isaac env still registers via uav_wm.envs when Isaac is importable.
    __all__ = []

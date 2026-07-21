"""uav_wm environments.

The 2D ``UAVTurretEnv`` imports anywhere there's pygame/numpy (Mac). The 3D
``Isaac-UAVTurret3D-v0`` is an Isaac Lab env that only registers when Isaac is
importable (the GPU box, post-AppLauncher). Both paths are guarded so the
Isaac container (no pygame) still gets the 3D env, and the Mac (no Isaac)
still gets the 2D env.
"""
__all__ = []

try:
    from uav_wm.envs.uav_turret import UAVTurretEnv

    __all__.append("UAVTurretEnv")
except Exception:
    # pygame/2D deps not available (Isaac Lab container) — 3D env only.
    pass

try:
    import gymnasium as gym

    from uav_wm.envs.uav_turret_3d import UAVTurret3DEnvCfg

    try:
        gym.spec("Isaac-UAVTurret3D-v0")
    except gym.error.Error:  # NameNotFound (unregistered) — gymnasium has no NoSuchRegisteredEnv
        gym.register(
            id="Isaac-UAVTurret3D-v0",
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            kwargs={"env_cfg_entry_point": UAVTurret3DEnvCfg},
        )
    __all__.append("UAVTurret3DEnvCfg")
except Exception:
    # Isaac Lab not available (Mac, or pre-AppLauncher) — 2D env only.
    pass

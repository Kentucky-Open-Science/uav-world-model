"""uav_wm data utilities: collection policies.

The 2D ``ExplorationPolicy`` needs swm + pygame (Mac). The 3D
``ExplorationPolicy3D`` is pure torch and imports anywhere (incl. the Isaac
Lab container). Both guarded so the container (no swm/pygame) still gets the
3D policy and the Mac (no torch-GPU needed) still gets the 2D policy.
"""
__all__ = []

try:
    from uav_wm.data.policies import ExplorationPolicy

    __all__.append("ExplorationPolicy")
except Exception:
    # swm / 2D deps not available (Isaac Lab container) — 3D policy only.
    pass

try:
    from uav_wm.data.policies_3d import ExplorationPolicy3D

    __all__.append("ExplorationPolicy3D")
except Exception:
    pass

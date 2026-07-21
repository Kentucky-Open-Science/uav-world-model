"""Unit tests for MultirotorVelocityController (pure torch, Mac-runnable).

Verifies the velocity->thrust cascade against the ARL_ROBOT_1 allocation matrix
before any Isaac runtime is involved. Run: ``.venv/bin/python scripts/test_multirotor_controller.py``
"""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # so `import uav_wm` works without install

from uav_wm.envs.multirotor_velocity_controller import MultirotorVelocityController

# ARL_ROBOT_1_CFG allocation matrix (6x4): rows = [Fx, Fy, Fz, Tx, Ty, Tz],
# cols = [back_left, back_right, front_left, front_right].
A = torch.tensor(
    [
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0],
        [-0.13, -0.13, 0.13, 0.13],
        [-0.13, 0.13, 0.13, -0.13],
        [-0.07, 0.07, -0.07, 0.07],
    ],
    dtype=torch.float32,
)
MASS = 1.5  # placeholder; the env resolves the real mass from the USD at runtime.
G = 9.81


def wrench_from_thrusts(thrusts: torch.Tensor) -> torch.Tensor:
    """Forward-map motor thrusts back to the 6D wrench via the allocation matrix."""
    return thrusts @ A.t()  # (N,4) @ (4,6) -> (N,6)


def level_quat(n: int) -> torch.Tensor:
    q = torch.zeros(n, 4)
    q[:, 0] = 1.0
    return q


def main() -> None:
    ctrl = MultirotorVelocityController(mass=MASS, allocation_matrix=A, thrust_range=(0.1, 10.0))
    n = 5

    # --- 1. Hover: level, at rest, zero desired velocity -> mg/4 per motor ---
    t = ctrl.compute(
        level_quat(n),
        torch.zeros(n, 3),
        torch.zeros(n, 3),
        torch.zeros(n, 3),
        torch.zeros(n),
    )
    hover_thrust = MASS * G / 4.0
    assert t.shape == (n, 4), f"shape {t.shape}"
    assert torch.allclose(t, t[:, :1].expand_as(t), atol=1e-4), "hover thrusts not symmetric"
    assert torch.allclose(t[0], torch.full((4,), hover_thrust), atol=1e-3), (
        f"hover thrust {t[0]} != mg/4={hover_thrust}"
    )
    assert (t >= 0.1).all() and (t <= 10.0).all(), "hover thrust out of range"
    w = wrench_from_thrusts(t)
    assert torch.allclose(w[:, 2], torch.full((n,), MASS * G), atol=1e-3), "hover Fz != mg"
    assert w[:, [0, 1, 3, 4, 5]].abs().max() < 1e-3, "hover has spurious lateral force/torque"
    print(f"[ok] hover: thrusts={t[0].tolist()} (mg/4={hover_thrust:.4f})")

    # --- 2. Forward +x velocity command -> nose-down pitch toward +x (positive Ty torque) ---
    # Tilting body-up b3 from +Z toward +X (to thrust the drone +x) is a +Y rotation
    # (Ry(+90) maps +Z->+X), so a correct controller yields Ty > 0.
    vel_des = torch.zeros(n, 3)
    vel_des[:, 0] = 3.0
    t = ctrl.compute(level_quat(n), torch.zeros(n, 3), torch.zeros(n, 3), vel_des, torch.zeros(n))
    w = wrench_from_thrusts(t)
    assert t.std() > 1e-3, "forward command produced symmetric (no-tilt) thrusts"
    assert (w[:, 4] > 1e-3).all(), f"forward command should pitch nose-down toward +x (Ty>0), got {w[:, 4]}"
    assert w[:, [0, 1, 5]].abs().max() < 1e-2, "forward command leaked into Fx/Fy/Tz"
    print(f"[ok] forward vx: Ty={w[0,4].item():.3f} (>0 = nose-down pitch toward +x), thrusts={t[0].tolist()}")

    # --- 3. Yaw-rate command -> nonzero yaw torque (Tz) ---
    t = ctrl.compute(
        level_quat(n), torch.zeros(n, 3), torch.zeros(n, 3), torch.zeros(n, 3), torch.full((n,), 2.0)
    )
    w = wrench_from_thrusts(t)
    assert (w[:, 5].abs() > 1e-3).all(), f"yaw command should produce Tz, got {w[:, 5]}"
    print(f"[ok] yaw rate: Tz={w[0,5].item():.3f}, thrusts={t[0].tolist()}")

    # --- 4. Climb command (+z) -> more collective thrust than hover ---
    vel_des = torch.zeros(n, 3)
    vel_des[:, 2] = 2.0
    t = ctrl.compute(level_quat(n), torch.zeros(n, 3), torch.zeros(n, 3), vel_des, torch.zeros(n))
    assert (t.sum(dim=-1) > MASS * G).all(), "climb should demand more collective thrust than hover"
    print(f"[ok] climb vz: collective={t[0].sum().item():.3f} > mg={MASS*G:.3f}")

    print("\nALL CONTROLLER TESTS PASSED")


if __name__ == "__main__":
    main()

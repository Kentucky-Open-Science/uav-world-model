"""Velocity -> thrust controller for the native Isaac Lab ``Multirotor``.

The native multirotor (``isaaclab_contrib.assets.Multirotor``) is *thrust
controlled*: its only command interface is :meth:`set_thrust_target`, which
takes a per-rotor thrust. There is no shipped high-level velocity/position
controller -- the manager-based ARL task pairs ``ThrustActionCfg`` (4 raw motor
thrusts) with RL. Our reactive ``ExplorationPolicy`` wants to command *velocity
directions*, so this module fills that gap with a standard cascade:

    desired world velocity  ->  velocity PD  ->  desired body thrust direction
        ->  attitude PD  ->  body wrench [Fx,Fy,Fz,Tx,Ty,Tz]
        ->  per-motor thrusts via the Moore-Penrose inverse of the allocation matrix

The controller is pure ``torch`` (no Isaac imports) so it can be unit-tested on
the Mac. The env simply passes the multirotor's ``data`` tensors in and feeds
the returned thrusts to ``robot.set_thrust_target`` then ``robot.write_data_to_sim``.

Quaternion convention matches Isaac Lab: ``(w, x, y, z)``.
"""

from __future__ import annotations

import os

import torch


def quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vector ``v`` (..., 3) by quaternion ``q`` (..., 4) in (w,x,y,z).

    Mirrors ``isaaclab.utils.math.quat_apply``.
    """
    q_w = q[..., 0:1]
    q_xyz = q[..., 1:4]
    t = 2.0 * torch.linalg.cross(q_xyz, v, dim=-1)
    return v + q_w * t + torch.linalg.cross(q_xyz, t, dim=-1)


def quat_apply_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate ``v`` by the inverse of ``q`` (w,x,y,z) — i.e. world->body frame.

    For a unit quaternion the inverse equals the conjugate (w, -x, -y, -z).
    """
    q_inv = torch.empty_like(q)
    q_inv[..., 0] = q[..., 0]
    q_inv[..., 1:4] = -q[..., 1:4]
    return quat_apply(q_inv, v)


def yaw_from_quat(q: torch.Tensor) -> torch.Tensor:
    """Yaw (rad) from quaternion ``(w,x,y,z)``."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class MultirotorVelocityController:
    """Map desired world-frame velocity + yaw rate to per-motor thrust targets.

    Parameters:
        mass: total multirotor mass (kg). Resolve at runtime from the asset
            (e.g. ``float(robot.data.default_mass[0].sum())`` — `default_mass`
            is `(num_instances, num_bodies)` on ArticulationData); same for all envs.
        allocation_matrix: ``(6, num_thrusters)`` tensor. Rows are the 6D wrench
            ``[Fx, Fy, Fz, Tx, Ty, Tz]`` produced per unit rotor thrust. For a
            standard quadrotor the Fx/Fy rows are zero (lateral motion comes from
            tilting). Fetched at runtime from the ``Multirotor.allocation_matrix``
            property (a class property, NOT ``robot.data.allocation_matrix``).
        thrust_range: ``(min, max)`` per-motor thrust (N). From ``ThrusterCfg.thrust_range``.
        gravity: gravitational acceleration (m/s^2), Z-up world.
        kp_v, kd_v: velocity PD gains.
        kp_R, kd_R: attitude (tilt) PD gains.
        kp_yaw: yaw-rate gain.
        device: torch device for the cached allocation inverse.
    """

    def __init__(
        self,
        mass: float,
        allocation_matrix: torch.Tensor,
        thrust_range: tuple[float, float] = (0.1, 10.0),
        gravity: float = 9.81,
        kp_v: float = 1.0,
        kd_v: float = 1.0,  # velocity derivative damping — WITHOUT it the velocity
        # loop is pure-P, overshoots, swings thrust_dir, and the attitude PD chases
        # it into motor saturation (oscillating [lo,hi,lo,hi] <-> [hi,lo,hi,lo]).
        kp_R: float = 3.0,  # low enough that a 30 deg tilt error (0.52 rad*e_R) ->>
        # 1.6 N·m torque, under the allocator's ~2.6 N·m max, so the PD stays LINEAR
        # and can actually damp. kp_R=8 commanded near-max torque for ~15 deg errors
        # -> saturation -> saturation breaks damping -> sustained oscillation.
        kd_R: float = 0.5,  # ~critical damping (zeta~1) for a fast attitude pole.
        kp_yaw: float = 1.0,  # low: yaw_rate_des can reach ~pi rad/s, and kp_yaw*that
        # must stay under the allocator's ~1.4 N·m yaw max or it saturates the motors.
        device: str | torch.device = "cpu",
    ) -> None:
        self.mass = float(mass)
        self.gravity = float(gravity)
        self.kp_v = kp_v
        self.kd_v = kd_v
        self.kp_R = kp_R
        self.kd_R = kd_R
        self.kp_yaw = kp_yaw
        self.thrust_min, self.thrust_max = thrust_range

        # allocation A is (6, Nt): wrench = A @ thrusts  =>  thrusts = A^+ @ wrench.
        # Cache A^+ transposed to (6, Nt) so a batched wrench (N, 6) maps via matmul.
        A = allocation_matrix.to(device=device, dtype=torch.float32)
        self.alloc_pinv_t = torch.linalg.pinv(A).t().contiguous()  # (6, Nt)
        self.num_thrusters = A.shape[1]
        self._eps = 1e-6
        # TEMP DIAG: dump the allocation matrix rows so we can verify the yaw-row
        # sign (suspected inverted -> positive-feedback yaw spin -> climb). Off by
        # default; on only when UAV_CTRL_DEBUG=1 (set in the smoke container).
        self._debug = os.environ.get("UAV_CTRL_DEBUG") == "1"
        if self._debug:
            print(f"[CTRLDIAG] allocation A (6x{A.shape[1]}), rows=[Fx,Fy,Fz,Tx,Ty,Tz]:")
            for i, name in enumerate(("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")):
                print(f"[CTRLDIAG]   {name}: {[float(x) for x in A[i].tolist()]}")
            self._dbg_calls = 0

        z = torch.zeros(1, 3, device=device)
        z[:, 2] = 1.0
        self._world_up = z  # (1, 3) world +Z

    def compute(
        self,
        root_quat_w: torch.Tensor,
        lin_vel_w: torch.Tensor,
        ang_vel_b: torch.Tensor,
        vel_des: torch.Tensor,
        yaw_rate_des: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-motor thrust targets for a batch of environments.

        Args:
            root_quat_w: ``(N, 4)`` base orientation in world, ``(w, x, y, z)``.
            lin_vel_w: ``(N, 3)`` base linear velocity in world.
            ang_vel_b: ``(N, 3)`` base angular velocity in body frame.
            vel_des: ``(N, 3)`` desired world-frame linear velocity.
            yaw_rate_des: ``(N,)`` desired yaw rate (rad/s).

        Returns:
            ``(N, num_thrusters)`` per-motor thrust targets, clamped to ``thrust_range``.
        """
        dev = self.alloc_pinv_t.device
        root_quat_w = root_quat_w.to(dev)
        lin_vel_w = lin_vel_w.to(dev)
        ang_vel_b = ang_vel_b.to(dev)
        vel_des = vel_des.to(dev)
        yaw_rate_des = yaw_rate_des.to(dev).reshape(-1, 1)

        n = root_quat_w.shape[0]
        up = self._world_up.expand(n, -1)

        # --- velocity PD -> desired world-frame force (incl. gravity compensation) ---
        a_des = self.kp_v * (vel_des - lin_vel_w) - self.kd_v * lin_vel_w
        g_vec = torch.zeros_like(a_des)
        g_vec[:, 2] = self.mass * self.gravity
        f_des = self.mass * a_des + g_vec  # (N, 3)

        # current body up (+Z) expressed in world
        b3 = quat_apply(root_quat_w, up)  # (N, 3)

        # collective thrust = projection of desired force onto current body up
        fz = (f_des * b3).sum(dim=-1, keepdim=True).clamp_min(0.0)  # (N, 1)

        # desired body-up direction = direction of desired force
        f_norm = torch.linalg.vector_norm(f_des, dim=-1, keepdim=True)
        thrust_dir = f_des / (f_norm + self._eps)

        # attitude error: rotation vector aligning b3 -> thrust_dir.
        # cross(b3, d) is the angular-velocity direction that rotates b3 INTO d
        # (right-hand rule: Ry(+90) maps +Z->+X), so the proportional torque must
        # be +kp_R*e_R; a leading minus would tilt the body AWAY from the command.
        e_R_w = torch.linalg.cross(b3, thrust_dir, dim=-1)  # (N, 3) world frame

        # e_R is in the WORLD frame, but body torques/wrench are applied in the
        # BODY frame and ang_vel_b is body-frame. Express the error in the body
        # frame (rotate by q^-1) so the PD is consistent — and so tilting is
        # correct when the drone has yawed (auto-yaw faces the travel heading).
        e_R = quat_apply_inverse(root_quat_w, e_R_w)  # (N, 3) body frame

        # body torques: PD on tilt error + angular-rate damping
        torques = self.kp_R * e_R - self.kd_R * ang_vel_b  # (N, 3)
        # decouple yaw: replace body-z torque with a yaw-rate controller
        yaw_err = yaw_rate_des - ang_vel_b[:, 2:3]
        torques[:, 2:3] = self.kp_yaw * yaw_err
        # TEMP DIAG: per-call sign check. If commanded Tz has the SAME sign as the
        # wz the controller read, the yaw loop is positive feedback (spins up, not
        # damps) -> saturates {0,2} or {1,3} -> net over-thrust -> climb.
        if self._debug and self._dbg_calls < 12:
            self._dbg_calls += 1
            wzb0 = float(ang_vel_b[0, 2])
            yr0 = float(yaw_rate_des[0])
            tz0 = float(torques[0, 2])
            er0 = float(e_R[0, 2])
            print(f"[CTRLDIAG] call#{self._dbg_calls} read_wz={wzb0:+.3f} yaw_rate_des={yr0:+.3f} "
                  f"e_R_z={er0:+.3f} -> commanded_Tz={tz0:+.3f} "
                  f"{'<< SAME SIGN AS wz (positive feedback!)' if (wzb0 != 0 and tz0 * wzb0 > 0 and abs(yr0) < 0.3) else ''}")

        # wrench [Fx, Fy, Fz, Tx, Ty, Tz]; Fx=Fy=0 (quadrotor has no lateral force)
        fx_fy = torch.zeros(n, 2, device=dev)
        wrench = torch.cat([fx_fy, fz, torques], dim=-1)  # (N, 6)

        # motor thrusts = wrench @ A^+^T
        thrusts = wrench @ self.alloc_pinv_t  # (N, Nt)
        return thrusts.clamp(self.thrust_min, self.thrust_max)

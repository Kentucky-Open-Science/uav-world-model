"""Isaac Lab 3D drone+turret env: native multirotor + drone POV camera + DR.

This is the 3D successor to the 2D ``UAVTurretEnv``. It runs only on the GPU box
(inside the ``nvcr.io/nvidia/isaac-lab:2.3.2`` container) — the Mac has no Isaac
Sim. It is a config-only ``ManagerBasedRLEnv`` (no env subclass):

* **Drone**: Isaac Lab's *native* multirotor (``ARL_ROBOT_1_CFG`` — 4 thrusters,
  no high-level controller). The policy commands a 4D **body-frame** velocity
  ``(vx_fwd, vy_strafe, vz_climb, yaw_rate)``; a custom ``VelocityToThrustAction``
  rotates the body xy-velocity by the drone yaw into the world frame the
  :class:`~uav_wm.envs.multirotor_velocity_controller.MultirotorVelocityController`
  expects (velocity PD -> attitude PD -> wrench -> pinv(allocation) -> motor
  thrusts) and passes ``yaw_rate`` straight through — NO auto-yaw (auto-yaw
  amplified step-to-step velocity-heading jitter into POV shake; an explicit yaw
  rate keeps the camera steady and lets the policy strafe-hold-heading /
  spin-in-place). Mirrors the shipped ``ThrustAction`` (``set_thrust_target``).
* **POV camera**: a ``TiledCameraCfg`` parented under the drone's MOVING root
  body (``{ENV_REGEX_NS}/Robot/base_link/Camera``), ``data_types=["rgb"]``,
  224x224, convention ``"ros"`` (body-frame — the camera FOLLOWS the drone's
  pose). Its RGB IS the ``pixels`` the world model trains on. NOTE: the parent
  MUST be a body link, NOT the articulation-root Xform ``{ENV_REGEX_NS}/Robot``
  — that Xform is a STATIC container; the moving rigid body is its child
  ``base_link`` (confirmed via fabric path ``Robot/base_link/collisions/...``).
  Parenting to ``/Robot`` yields a FIXED-perspective camera (Evan's 2026-07-16
  critique); parenting to ``/Robot/base_link`` makes it true first-person.
* **Turret** (NPC): a kinematic box, repositioned each episode to a random street
  intersection, that rate-limits its yaw/pitch to track the drone (50 deg/s yaw).
  Aim + LOS + range define danger.
* **Danger model** (3D analog of the 2D env): ``danger = in_range & aimed & los``
  where ``aimed`` = drone inside the turret's +/-30 deg aim cone, ``los`` =
  turret->drone segment clears all obstacle AABBs (analytical, torch), and
  ``in_range`` = dist < 8 m. Sustained danger for ``FIRE_INTERVAL`` s kills the
  drone. The turret writes its pose each step via an ``interval`` EventTerm so
  the POV camera sees it tracking.
* **Urban environment** (v2): a FIXED 4x4 textured building grid on a dark
  asphalt ground = the navigable terrain + the turret's LOS blockers. The grid is
  the SWAP POINT for the UK LiDAR campus map (replace with a USD terrain).
* **Domain Randomization** (v2: geometric): per-episode the reset event
  randomizes the turret position (street intersection), goal (far edge), and
  drone spawn (near edge, A->B). Light/texture DR is deferred. Buildings are fixed
  (a city doesn't move) — the WM learns dynamics/danger, not one building layout.

State vector is 21-dim with ``danger`` at index 13 (matching the 2D convention so
the Phase-4 danger head reads the same index) and drone yaw at index 20 (the
policy needs it for body-frame geometry). Action is 4D body-frame
``(vx_fwd, vy_strafe, vz_climb, yaw_rate)`` in [-1, 1], scaled to +/-
``MAX_SPEED`` / ``MAX_YAW_RATE``. A killed drone is NOT terminated immediately —
it falls (thrust cut + tumble) and the episode ends ``FALL_RECORD_S`` after
ground impact, so the WM learns "death = camera tumbles then stops."

Gym id: ``Isaac-UAVTurret3D-v0`` (registered in ``uav_wm/envs/__init__.py``).
"""

from __future__ import annotations

import math
from dataclasses import MISSING

import torch

from isaaclab.assets import AssetBaseCfg, RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp import (
    image as mdp_image,
    reset_root_state_uniform,
)
from isaaclab.managers import (
    ActionTerm,
    ActionTermCfg,
    EventTermCfg as EventTerm,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    TerminationTermCfg as DoneTerm,
    SceneEntityCfg,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from isaaclab_contrib.assets import MultirotorCfg
from isaaclab_assets.robots.arl_robot_1 import ARL_ROBOT_1_CFG

from .multirotor_velocity_controller import MultirotorVelocityController

# ----------------------------------------------------------------------------
# Task constants (3D analog of the 2D UAVTurretEnv).
# ----------------------------------------------------------------------------
TURRET_RANGE = 8.0  # m; drone within this of the turret is "in range"
FOV_HALF = math.radians(30.0)  # turret aim cone half-angle (rad)
# Sustained danger for FIRE_INTERVAL s kills the drone. The MultirotorVelocity
# controller is conservative (kp_v=1.0 -> ~0.5 m/s^2 horizontal accel, attitude
# cascade), so an EVADE drone commands full-reverse but sheds its ~1 m/s
# approach momentum slowly. 0.7 s killed the drone mid-reversal (it had barely
# opened the range) -> 1.87% danger, all lethal. 2.0 s (20 steps) gives the
# sluggish EVADE time to reverse + exit the 8 m range from a 7.7 m stand-off
# (0.3 m exit at ~0.5 m/s^2 from ~0 velocity ~= 1.5 s < 2.0 s) -> survives ->
# approach->danger->evade->survive->re-provoke cycles. COMMIT (suicide dive)
# still sustains danger > 2.0 s -> dies (the lethal tail). See policies_3d EVADE.
FIRE_INTERVAL = 2.0  # s of sustained danger before the drone is killed
# 50 deg/s (was 110): slow enough that `aimed` lags an off-axis drone by ~10-18
# steps, giving the EVADE drone a pre-danger window to reach cover. At 110 deg/s
# the turret aimed almost instantly -> no time to reach cover -> no survival.
TURRET_YAW_RATE = math.radians(50.0)  # rad/s
TURRET_PITCH_RATE = math.radians(60.0)  # rad/s
PITCH_LIMIT = math.radians(70.0)  # turret pitch clamp
MAX_SPEED = 2.0  # m/s; action[0:3] in [-1,1] maps to +/- MAX_SPEED (body-frame vx/vy/vz)
MAX_YAW_RATE = 2.0  # rad/s; action[3] in [-1,1] maps to +/- this (~115 deg/s, well above the
                    # 50 deg/s turret so the drone yaws faster than it tracks -> break aim up close)
GOAL_RADIUS = 0.6  # m; drone within this of the goal succeeds
DRONE_MASS = 1.24  # kg; resolved at runtime via root_physx_view.get_masses() (5 bodies).
DRONE_SPAWN_Z = 1.5  # m; spawn height
CRASH_Z = 0.2  # m; below this a (non-shot) drone has crashed
IMPACT_Z = 0.4  # m; a SHOT drone at/below this has hit the ground (start the fall-record window).
# MUST exceed the asset's resting height (~0.25 m): the drone center never gets
# lower than that on the ground, so IMPACT_Z <= 0.25 would never latch a ground
# fall and every kill would burn the full SHOT_HARD_CAP_S while bouncing. 0.4
# latches the instant a falling shot drone descends past ~knee height.
FALL_RECORD_S = 1.0  # s recorded after a shot drone's ground impact before the episode ends
# Hard cap on the fall: a shot drone that lands on a building ROOF (z > IMPACT_Z,
# e.g. a 2 m cuboid) never reaches the ground, so the z<=IMPACT_Z impact test would
# never latch and uav_time_out would suppress truncation forever -> the episode
# hangs. After SHOT_HARD_CAP_S of being shot, force _impacted (start the 1 s rest,
# then end). A real street fall impacts in ~0.8 s, well under this cap, so the cap
# only catches the roof-landing minority.
SHOT_HARD_CAP_S = 3.0
TURRET_LOCAL = (0.0, 0.0, 0.5)  # turret base, env-local (init pose; overwritten at reset)
TURRET_BASE_Z = 0.5  # m; turret center height (pedestal)
ARENA_HALF = 22.0  # m; urban arena half-extent (44x44 m) — big enough for long A->B paths

# --- Urban city grid (the navigable terrain + the turret's LOS blockers) ---
# Buildings are FIXED per env (a city doesn't move). This layout is the SWAP POINT
# for the UK LiDAR campus map: replace BUILDING_LAYOUT + the ground cuboid with a
# TerrainImporterCfg(terrain_type="usd", usd_path=<campus.usd>) and keep the
# turret/spawn/goal/danger logic unchanged.
BUILDING_GRID = (-9.0, -3.0, 3.0, 9.0)  # building center x,y coords (6 m spacing -> 3 m streets)
BUILDING_HALF = (1.5, 1.5)  # footprint half-extents (3x3 m buildings)
BUILDING_HEIGHTS = (5.0, 8.0, 6.0, 10.0, 4.0, 7.0, 9.0, 5.0,
                    8.0, 4.0, 6.0, 9.0, 5.0, 7.0, 10.0, 6.0)  # varied facades
_BUILDING_COLORS = [
    (0.50, 0.50, 0.52),  # concrete grey
    (0.60, 0.35, 0.30),  # brick red
    (0.30, 0.45, 0.60),  # glass blue
    (0.75, 0.70, 0.60),  # beige
    (0.30, 0.30, 0.32),  # dark
]
STREET_GRID = (-12.0, -6.0, 0.0, 6.0, 12.0)  # street intersections (clear of buildings; spawn + turret placement)

# Episode-type mix (per-env; the collector rolls base._next_episode_type and the
# env commits it to base.episode_type on reset). Models the "fixed flight + WM as
# a safety monitor" use case: most episodes are a predetermined route (with or
# without a turret) and a minority are the drunk-provoke danger manufacturer.
#   EP_NO_TURRET       stows the turret (danger masked to 0) -> pure normal flight
#   EP_WAYPOINT_TURRET biases the turret onto the route -> fixed flight crosses FOV
#   EP_DRUNK           survivable-danger provoker (ExplorationPolicy3D)
EP_NO_TURRET, EP_WAYPOINT_TURRET, EP_DRUNK = 0, 1, 2
# Where a no-turret env's turret is stowed: far outside the arena + below ground,
# so it is invisible to the POV and never in_range (dist >> TURRET_RANGE).
TURRET_STOW = (999.0, 999.0, -50.0)
# Where the goal marker is stowed (no goal now -- Evan: "remove the target
# location"; the drone follows fixed paths, not a goal-seeking A->B). Far outside
# the arena + below ground: invisible to the POV, and uav_goal_reached never fires
# (norm(GOAL_STOW - drone) >> GOAL_RADIUS).
GOAL_STOW = (999.0, 999.0, -50.0)


def roll_episode_types(n: int, device, generator=None) -> torch.Tensor:
    """Per-env 40% no-turret / 40% waypoint+turret / 20% drunk+turret (long tensor)."""
    r = torch.rand(n, device=device, generator=generator)
    t = torch.full((n,), EP_DRUNK, dtype=torch.long, device=device)
    t[r < 0.4] = EP_NO_TURRET
    t[(r >= 0.4) & (r < 0.8)] = EP_WAYPOINT_TURRET
    return t


def building_layout():
    """Fixed 4x4 urban building grid: list of (center_xyz, full_size_xyz), env-local."""
    layout = []
    for i, x in enumerate(BUILDING_GRID):
        for j, y in enumerate(BUILDING_GRID):
            h = BUILDING_HEIGHTS[(i * len(BUILDING_GRID) + j) % len(BUILDING_HEIGHTS)]
            layout.append(((x, y, h / 2.0), (2 * BUILDING_HALF[0], 2 * BUILDING_HALF[1], h)))
    return layout


BUILDING_LAYOUT = building_layout()
NUM_OBSTACLES = len(BUILDING_LAYOUT)  # = 16 buildings (drive the turret LOS test)

# State layout (21-dim). DANGER at index 13 (matches 2D convention so the Phase-4
# danger head reads the same index). Index 20 = drone yaw (world) — the policy
# needs it to compute body-frame turret bearing + building avoidance. The WM does
# NOT use `state` in its loss (pixels+action only), so the extra dim is free.
STATE_DIM = 21
DANGER_IDX = 13


# ----------------------------------------------------------------------------
# Custom action: 3D world velocity -> per-motor thrusts via the PD cascade.
# ----------------------------------------------------------------------------
class VelocityToThrustAction(ActionTerm):
    """Command a 4D BODY-frame velocity; internally convert to motor thrusts.

    Action is ``(vx_fwd, vy_strafe, vz_climb, yaw_rate)`` in [-1, 1]:
      * vx_fwd / vy_strafe / vz_climb are BODY-frame (vx = nose-forward, vy =
        left-strafe, vz = up) — the WM sees only the POV, so a world-frame action
        would be many-to-one (same action, different pixel effect per heading).
        Body-frame grounds the action to the camera: "vx=1 always means the
        center pixels get bigger."
      * yaw_rate is an EXPLICIT yaw rate (rad/s), NOT auto-derived from the
        velocity heading. Auto-yaw was a sim crutch that amplified the policy's
        step-to-step velocity-heading jitter into POV shake; an explicit rate
        lets the policy command strafe-with-zero-yaw and spin-in-place — and
        keeps the camera steady when yaw_rate=0.

    ``apply_actions`` rotates the body xy-velocity by the drone's current yaw into
    the world frame the controller expects, passes ``yaw_rate`` straight through,
    and zeros/tumbles shot drones so they fall. Mirrors the shipped ``ThrustAction``
    (calls ``robot.set_thrust_target``).
    """

    def __init__(self, cfg: "VelocityToThrustActionCfg", env) -> None:
        # base sets _cfg/_env/_asset/_IO_descriptor; device & num_envs are inherited
        # read-only properties (must NOT be assigned here). Mirrors shipped ThrustAction.
        super().__init__(cfg, env)
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._controller = None  # lazy: needs mass, which is None until sim runs
        self._world_up = torch.tensor([0.0, 0.0, 1.0], device=self.device)

    # -- ActionTerm API --
    @property
    def action_dim(self) -> int:
        return 4

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions
        scaled = torch.empty_like(self._raw_actions)
        scaled[:, 0:3] = self._raw_actions[:, 0:3] * self.cfg.max_speed      # body vx/vy/vz (m/s)
        scaled[:, 3] = self._raw_actions[:, 3] * self.cfg.max_yaw_rate        # yaw rate (rad/s)
        self._processed_actions = scaled

    def apply_actions(self) -> None:
        self._ensure_controller()
        _init_buffers(self._env)  # ensures _shot/_impacted exist before the first world_step
        robot = self._asset
        a = self._processed_actions  # (N, 4) [vx_b, vy_b, vz, yaw_rate]
        # current drone yaw from root quat (w,x,y,z)
        q = robot.data.root_quat_w
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        cy, sy = torch.cos(yaw), torch.sin(yaw)
        vx_b, vy_b, vz = a[:, 0], a[:, 1], a[:, 2]
        # body -> world (Rz(yaw) on body xy); vz is body up == world up for a level drone
        wx = vx_b * cy - vy_b * sy
        wy = vx_b * sy + vy_b * cy
        vel_des = torch.stack([wx, wy, vz], dim=-1)  # (N, 3) world-frame, as the controller expects
        yaw_rate_des = a[:, 3]
        ang_vel_b = robot.data.root_ang_vel_b if hasattr(robot.data, "root_ang_vel_b") \
            else robot.data.root_ang_vel_w
        thrusts = self._controller.compute(
            root_quat_w=robot.data.root_quat_w,
            lin_vel_w=robot.data.root_lin_vel_w,
            ang_vel_b=ang_vel_b,
            vel_des=vel_des,
            yaw_rate_des=yaw_rate_des,
        )
        # Shot drones fall + tumble (then rest): replace the controller's hover thrust
        # with a small RANDOM per-motor thrust while falling (<< hover => falls;
        # asymmetric => tumbles the camera) and ZERO thrust once impacted (rests still,
        # so the recorded 1 s post-impact shows a stopped camera). Net thrust ~0.4-6 N
        # vs ~12 N hover weight => always falls.
        shot = self._env._shot
        if shot.any():
            ns = int(shot.sum().item())
            impacted = self._env._impacted[shot].unsqueeze(-1)
            tumble = torch.rand(ns, thrusts.shape[1], device=self.device) * 1.5
            rest = torch.zeros(ns, thrusts.shape[1], device=self.device)
            thrusts[shot] = torch.where(impacted, rest, tumble)
        robot.set_thrust_target(thrusts, thruster_ids=slice(None))

    def _ensure_controller(self) -> None:
        if self._controller is not None:
            return
        mass = None
        src = "none"
        # default_mass is None on the native Multirotor (its _initialize_impl
        # doesn't set it); read the PhysX masses directly from the root view.
        try:
            rpv = getattr(self._asset, "root_physx_view", None)
            if rpv is not None:
                masses = rpv.get_masses()  # (num_envs, num_bodies)
                if masses is not None and masses.numel() > 0:
                    mass = float(masses[0].sum())
                    src = f"root_physx_view.get_masses (n_bodies={masses.shape[1]})"
        except Exception:
            mass = None
        if mass is None or mass <= 0.0:
            try:
                dm = self._asset.data.default_mass
                if dm is not None and dm.numel() > 0:
                    mass = float(dm[0].sum())
                    src = "data.default_mass"
            except Exception:
                mass = None
        if mass is None or mass <= 0.0:
            mass = self.cfg.mass  # fallback (1.5 kg)
            src = "FALLBACK cfg.mass"
        print(f"[CTRL] mass={mass:.4f} kg (source: {src})", flush=True)
        alloc = self._asset.allocation_matrix
        self._controller = MultirotorVelocityController(
            mass=mass,
            allocation_matrix=alloc,
            thrust_range=self.cfg.thrust_range,
            device=self.device,
        )


@configclass
class VelocityToThrustActionCfg(ActionTermCfg):
    """Config for :class:`VelocityToThrustAction`."""

    class_type: type = VelocityToThrustAction
    asset_name: str = "robot"
    max_speed: float = MAX_SPEED
    max_yaw_rate: float = MAX_YAW_RATE
    mass: float = DRONE_MASS  # used only if runtime mass lookup fails
    thrust_range: tuple[float, float] = (0.1, 10.0)


# ----------------------------------------------------------------------------
# Custom MDP functions: per-step world update, reset/spawn+DR, obs, terms.
# All buffers live on the env object (created lazily on first reset).
# ----------------------------------------------------------------------------
def _init_buffers(env) -> None:
    if getattr(env, "_uav3d_bufs", False):
        return
    n = env.num_envs
    dev = env.device
    env._uav3d_bufs = True
    env.turret_yaw = torch.zeros(n, device=dev)
    env.turret_pitch = torch.zeros(n, device=dev)
    env.fire_timer = torch.zeros(n, device=dev)
    env.danger = torch.zeros(n, device=dev, dtype=torch.bool)
    env.in_range = torch.zeros(n, device=dev, dtype=torch.bool)
    env.los = torch.ones(n, device=dev, dtype=torch.bool)
    env.aimed = torch.zeros(n, device=dev, dtype=torch.bool)
    # shot / fall state: a killed drone is NOT terminated immediately — it falls
    # (tumble) and the episode ends FALL_RECORD_S after ground impact, so the WM
    # learns "death = camera tumbles then stops". See uav_killed / apply_actions.
    env._shot = torch.zeros(n, device=dev, dtype=torch.bool)
    env._impacted = torch.zeros(n, device=dev, dtype=torch.bool)
    env._post_impact_timer = torch.zeros(n, device=dev)
    env._shot_timer = torch.zeros(n, device=dev)  # s since shot (hard-caps roof-landed falls)
    # episode-type mix (collector rolls _next_episode_type when an env's CURRENT
    # episode starts; reset commits it to episode_type). Initialized to a random
    # 40/40/20 mix so episode 0 is mixed (collector rolls _next for ep >= 1).
    env.episode_type = roll_episode_types(n, dev)
    env._next_episode_type = env.episode_type.clone()
    env.goal_pos = torch.zeros(n, 3, device=dev)
    env.turret_pos = torch.zeros(n, 3, device=dev)  # randomized per episode (street intersection)
    # buildings: FIXED grid (same across envs + episodes). AABBs drive the LOS test.
    centers = torch.tensor([c for c, _ in BUILDING_LAYOUT], device=dev, dtype=torch.float32)
    halves = torch.tensor([(s[0] / 2, s[1] / 2, s[2] / 2) for _, s in BUILDING_LAYOUT],
                          device=dev, dtype=torch.float32)
    env.obstacle_centers = centers.unsqueeze(0).expand(n, -1, -1).contiguous()  # (N,16,3)
    env.obstacle_half = halves.unsqueeze(0).expand(n, -1, -1).contiguous()  # (N,16,3)


def _drone_pos_env(env) -> torch.Tensor:
    """Drone position in env-local frame (N, 3)."""
    return env.scene["robot"].data.root_pos_w - env.scene.env_origins


def _turret_pos_w(env) -> torch.Tensor:
    """Turret world position (N, 3) = env origin + per-episode randomized turret_pos."""
    return env.scene.env_origins + env.turret_pos


def uav_world_step(env, env_ids) -> None:
    """Interval event (fires each env step after the sim step, before terms/obs):
    rate-limit the turret aim toward the drone, write the turret pose, compute
    LOS / danger / fire-timer. Stores results on ``env`` for the term/obs funcs.
    """
    _init_buffers(env)
    n = env.num_envs
    dt = env.step_dt
    robot = env.scene["robot"]
    turret = env.scene["turret"]

    drone_w = robot.data.root_pos_w  # (N, 3) world
    turret_w = _turret_pos_w(env)  # (N, 3) world
    rel = drone_w - turret_w  # (N, 3) turret -> drone, world
    dist = torch.linalg.vector_norm(rel, dim=-1)  # (N,)
    dir_to_drone = rel / (dist.unsqueeze(-1) + 1e-6)

    # desired yaw/pitch to point at the drone
    des_yaw = torch.atan2(dir_to_drone[:, 1], dir_to_drone[:, 0])
    des_pitch = torch.asin(dir_to_drone[:, 2].clamp(-1.0, 1.0))

    # rate-limit current aim toward desired
    dyaw = torch.atan2(torch.sin(des_yaw - env.turret_yaw), torch.cos(des_yaw - env.turret_yaw))
    dyaw = torch.clamp(dyaw, -TURRET_YAW_RATE * dt, TURRET_YAW_RATE * dt)
    env.turret_yaw = env.turret_yaw + dyaw
    dpitch = torch.clamp(des_pitch - env.turret_pitch, -TURRET_PITCH_RATE * dt, TURRET_PITCH_RATE * dt)
    env.turret_pitch = torch.clamp(env.turret_pitch + dpitch, -PITCH_LIMIT, PITCH_LIMIT)

    # current aim direction (from yaw/pitch)
    cp = torch.cos(env.turret_pitch)
    aim_dir = torch.stack([cp * torch.cos(env.turret_yaw), cp * torch.sin(env.turret_yaw),
                           torch.sin(env.turret_pitch)], dim=-1)  # (N, 3)
    cos_aim = (aim_dir * dir_to_drone).sum(dim=-1).clamp(-1.0, 1.0)
    env.aimed = cos_aim > math.cos(FOV_HALF)
    env.in_range = dist < TURRET_RANGE
    # no-turret envs: never in_range/aimed (turret stowed far; mask so fire_timer
    # can't accumulate and danger stays 0 for the whole episode).
    has_turret = env.episode_type != EP_NO_TURRET
    env.in_range = env.in_range & has_turret
    env.aimed = env.aimed & has_turret

    # LOS: segment turret->drone vs each obstacle AABB (analytical slab test)
    los = torch.ones(n, dtype=torch.bool, device=env.device)
    seg_dir = rel / (dist.unsqueeze(-1) + 1e-6)
    t1 = torch.zeros(n, device=env.device)
    t2 = dist.clone()
    for k in range(NUM_OBSTACLES):
        cen = env.obstacle_centers[:, k, :] + env.scene.env_origins  # world
        half = env.obstacle_half[:, k, :]  # (N, 3)
        d = cen - turret_w  # turret -> obstacle center
        # slab method on each axis; seg_dir may have zeros -> handle
        inv = torch.where(seg_dir.abs() < 1e-6, torch.full_like(seg_dir, 1e9), 1.0 / seg_dir)
        ta = (d - half) * inv
        tb = (d + half) * inv
        tmin = torch.minimum(ta, tb).max(dim=-1).values
        tmax = torch.maximum(ta, tb).min(dim=-1).values
        # obstacle blocks if [tmin, tmax] overlaps [0, dist] with tmin < tmax
        blocked = (tmin < t2) & (tmax > t1) & (tmin < tmax) & (tmin < dist) & (tmax > 0.0)
        los = los & ~blocked
    env.los = los

    new_danger = env.in_range & env.aimed & env.los
    env.fire_timer = torch.where(new_danger, env.fire_timer + dt, torch.zeros_like(env.fire_timer))
    # kill -> shot (once): sustained danger for FIRE_INTERVAL. NOT terminated here;
    # the drone falls and uav_killed ends the episode FALL_RECORD_S after impact.
    env._shot = env._shot | (env.fire_timer >= FIRE_INTERVAL)
    # hold fire_timer at FIRE_INTERVAL once shot (fire_prog stays 1.0 through the fall
    # so the collector's outcome inference still sees "killed" on the last frame).
    env.fire_timer = torch.where(env._shot, torch.full_like(env.fire_timer, FIRE_INTERVAL),
                                 env.fire_timer)
    # danger excludes already-shot drones (shot = killed, not "about to be killed");
    # keeps the Phase-4 danger label meaning "imminent kill" and the aim-line from
    # staying red through the fall.
    env.danger = new_danger & ~env._shot
    # ground-impact detection for the falling (shot) drone -> start the post-impact
    # record window. env-local z = world z - env origin z.
    drone_z = robot.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    env._shot_timer = torch.where(env._shot, env._shot_timer + dt, torch.zeros_like(env._shot_timer))
    # A real street fall reaches z<=IMPACT_Z in ~0.8 s; the hard cap catches shot
    # drones that landed on a building roof (z>IMPACT_Z) and would otherwise hang.
    roof_capped = env._shot & (env._shot_timer >= SHOT_HARD_CAP_S)
    env._impacted = env._impacted | (env._shot & ~env._impacted & (drone_z <= IMPACT_Z)) | roof_capped
    env._post_impact_timer = torch.where(env._impacted, env._post_impact_timer + dt,
                                         env._post_impact_timer)

    # write the turret kinematic pose so the POV camera sees it tracking.
    # Orient the box's +X to aim_dir (quat from +X to aim_dir).
    quat = _quat_from_x_to(aim_dir, env.device)
    pose = torch.cat([turret_w, quat], dim=-1)  # (N, 7)
    turret.write_root_pose_to_sim(pose)
    turret.write_root_velocity_to_sim(torch.zeros(n, 6, device=env.device))


def _quat_from_x_to(target_dir: torch.Tensor, device) -> torch.Tensor:
    """Quaternion (w,x,y,z) rotating the body +X axis to ``target_dir`` (N,3)."""
    x_axis = torch.tensor([1.0, 0.0, 0.0], device=device).expand_as(target_dir)
    d = (x_axis * target_dir).sum(dim=-1).clamp(-1.0, 1.0)
    axis = torch.linalg.cross(x_axis, target_dir, dim=-1)
    axis_norm = torch.linalg.vector_norm(axis, dim=-1, keepdim=True)
    # near-parallel: identity; near-antiparallel: 180 about +Z
    w = torch.sqrt((1.0 + d).clamp_min(0.0)) * 0.5
    xyz = axis / (2.0 * w.unsqueeze(-1) + 1e-6)
    # antiparallel fallback
    anti = d < -0.999
    w = torch.where(anti, torch.zeros_like(w), w)
    xyz = torch.where(anti.unsqueeze(-1), torch.tensor([0.0, 0.0, 1.0], device=device).expand_as(xyz), xyz)
    q = torch.cat([w.unsqueeze(-1), xyz], dim=-1)
    return q / (torch.linalg.vector_norm(q, dim=-1, keepdim=True) + 1e-6)


def uav_reset_spawn(env, env_ids) -> None:
    """Reset event: randomize drone spawn (delegated to reset_root_state_uniform
    separately), obstacle layouts, goal, and reset per-env turret/fire buffers.
    This is the v1 geometric Domain Randomization.
    """
    _init_buffers(env)
    n_ids = len(env_ids)
    dev = env.device
    quat_id = torch.tensor([1.0, 0.0, 0.0, 0.0], device=dev).expand(n_ids, 4)
    zeros6 = torch.zeros(n_ids, 6, device=dev)

    # buildings are a FIXED city grid — never repositioned. Only turret/goal reset.

    # commit this episode's type from the one-ahead config the collector rolled
    # when the env's CURRENT episode started (_init_buffers seeds ep 0; the
    # collector rolls _next for ep >= 1 so reset places the right turret here).
    env.episode_type[env_ids] = env._next_episode_type[env_ids]
    etype = env.episode_type[env_ids]

    # No goal (Evan: "remove the target location; the drone follows fixed paths
    # throughout the map"). Stow the marker far outside the arena so it is
    # invisible to the POV and uav_goal_reached never fires.
    env.goal_pos[env_ids] = torch.tensor(GOAL_STOW, device=dev, dtype=torch.float32)
    goal = env.scene["goal"]
    gpos_w = env.goal_pos[env_ids] + env.scene.env_origins[env_ids]
    goal.write_root_pose_to_sim(torch.cat([gpos_w, quat_id], dim=-1), env_ids=env_ids)
    goal.write_root_velocity_to_sim(zeros6, env_ids=env_ids)

    # turret placement depends on episode type:
    #  - EP_DRUNK: random street intersection (clear LOS down the streets).
    #  - EP_WAYPOINT_TURRET: random street intersection here as a frame-0
    #    fallback; the COLLECTOR relocates it onto a mid-route waypoint after
    #    generating the route (the env can't predict the random-walk route).
    #  - EP_NO_TURRET: stow far outside the arena (invisible, never in_range).
    street = torch.tensor(STREET_GRID, device=dev)
    S = len(STREET_GRID)
    tx = street[torch.randint(0, S, (n_ids,), device=dev)]
    ty = street[torch.randint(0, S, (n_ids,), device=dev)]
    tz = torch.full((n_ids,), TURRET_BASE_Z, device=dev)
    has_turret = etype != EP_NO_TURRET
    stow = ~has_turret
    stow_x, stow_y, stow_z = TURRET_STOW
    tx = torch.where(stow, torch.full_like(tx, stow_x), tx)
    ty = torch.where(stow, torch.full_like(ty, stow_y), ty)
    tz = torch.where(stow, torch.full_like(tz, stow_z), tz)
    env.turret_pos[env_ids] = torch.stack([tx, ty, tz], dim=-1)
    turret = env.scene["turret"]
    tpos_w = env.turret_pos[env_ids] + env.scene.env_origins[env_ids]
    turret.write_root_pose_to_sim(torch.cat([tpos_w, quat_id], dim=-1), env_ids=env_ids)
    turret.write_root_velocity_to_sim(zeros6, env_ids=env_ids)

    # Random street-intersection spawn for ALL episodes (Evan: "randomize the
    # starting location of the drone"). reset_drone (reset_root_state_uniform,
    # declared above) already spawned every env on the near edge; OVERWRITE at a
    # random STREET_GRID intersection (building-free -> no spawn-in-wall; covers
    # the whole map) with a random heading. DRONE_SPAWN_Z (1.5 m) < roofline ->
    # canyon flight, not a roof. Declaration order (reset_drone before
    # reset_world) means this write wins. Drunk drones avoid the turret's own
    # intersection (0-distance -> instant in_range); path episodes don't care
    # (the waypoint turret is relocated onto the route, away from the spawn).
    flat = torch.randint(0, S * S, (n_ids,), device=dev)
    sx = street[flat % S]
    sy = street[flat // S]
    sz = torch.full((n_ids,), DRONE_SPAWN_Z, device=dev)
    drunk_e = etype == EP_DRUNK
    tcell = drunk_e & (sx == tx) & (sy == ty)  # drunk on the turret's cell -> bump x
    if tcell.any():
        sx = torch.where(tcell, street[(flat % S + 1) % S], sx)
    yaw_r = (torch.rand(n_ids, device=dev) * 2.0 - 1.0) * torch.pi  # random heading
    qz = torch.sin(yaw_r / 2.0)
    qw = torch.cos(yaw_r / 2.0)
    quat_sp = torch.stack([qw, torch.zeros_like(qz), torch.zeros_like(qz), qz], dim=-1)
    pos = env.scene.env_origins[env_ids] + torch.stack([sx, sy, sz], dim=-1)
    robot = env.scene["robot"]
    robot.write_root_pose_to_sim(torch.cat([pos, quat_sp], dim=-1), env_ids=env_ids)
    robot.write_root_velocity_to_sim(torch.zeros(n_ids, 6, device=dev), env_ids=env_ids)

    # reset turret aim + fire timer for these envs (point forward initially)
    env.turret_yaw[env_ids] = 0.0
    env.turret_pitch[env_ids] = 0.0
    env.fire_timer[env_ids] = 0.0
    env.danger[env_ids] = False
    env.in_range[env_ids] = False
    env.los[env_ids] = True
    env.aimed[env_ids] = False
    env._shot[env_ids] = False
    env._impacted[env_ids] = False
    env._post_impact_timer[env_ids] = 0.0
    env._shot_timer[env_ids] = 0.0


def uav_state(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """21-dim state vector (env-local). ``danger`` is at index 13; drone yaw at 20."""
    _init_buffers(env)
    dp = _drone_pos_env(env)  # (N, 3)
    dv = env.scene["robot"].data.root_lin_vel_w  # (N, 3) world ~= env (no rotation)
    # No goal (Evan: "remove the target location"). goal_pos is stowed far outside
    # the arena; emitting that vector would inject a near-constant far-field offset
    # into the state. Zero goal_rel so the (placeholder) slots stay neutral.
    goal_rel = torch.zeros_like(dp)  # (N, 3)
    turret_rel = env.turret_pos - dp  # (N, 3) turret is randomized per episode
    dist_t = torch.linalg.vector_norm(turret_rel, dim=-1, keepdim=True)  # (N, 1)
    danger = env.danger.float().unsqueeze(-1)
    in_range = env.in_range.float().unsqueeze(-1)
    los = env.los.float().unsqueeze(-1)
    aimed = env.aimed.float().unsqueeze(-1)
    fire_prog = (env.fire_timer / FIRE_INTERVAL).clamp(0.0, 1.0).unsqueeze(-1)
    tyaw = env.turret_yaw.unsqueeze(-1)
    tpit = env.turret_pitch.unsqueeze(-1)
    # drone yaw (world) — the policy needs it for body-frame turret bearing + avoidance
    q = env.scene["robot"].data.root_quat_w
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    drone_yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)).unsqueeze(-1)
    return torch.cat([dp, dv, goal_rel, turret_rel, dist_t, danger, in_range, los, aimed,
                      fire_prog, tyaw, tpit, drone_yaw], dim=-1)  # (N, 21)


def uav_killed(env) -> torch.Tensor:
    """Termination: a SHOT drone that has impacted the ground AND been recorded
    for ``FALL_RECORD_S`` seconds. The kill itself (sustained danger) only sets
    ``_shot`` in ``uav_world_step`` — the drone then falls (thrust cut in
    ``apply_actions``) and the episode ends here, ``FALL_RECORD_S`` after impact,
    so the WM captures the tumble-to-rest tail.

    Isaac Lab 2.3.2 termination terms take ONLY ``env`` (the manager resolves
    with min_argc=1, so a second non-default arg like ``env_ids`` is rejected).
    Returns a ``(num_envs,)`` bool tensor; the manager ANDs it with active envs.
    """
    _init_buffers(env)
    return env._shot & env._impacted & (env._post_impact_timer >= FALL_RECORD_S)


def uav_crash(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Termination: a NON-shot drone below CRASH_Z (controller-failure safety net;
    should not fire under the thruster-passthrough fix). Shot drones are excluded
    so their ground contact is handled by the fall path (uav_killed), not cut short."""
    _init_buffers(env)
    drone_z = env.scene["robot"].data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return ~env._shot & (drone_z < CRASH_Z)


def uav_time_out(env) -> torch.Tensor:
    """Truncation: the 300-step episode cap, but NOT while a shot drone is mid-fall.

    A shot drone must complete its tumble -> ground impact -> FALL_RECORD_S record
    (Evan's requirement: "capture the drone actually falling... ending one second
    after it hits the ground"). Suppressing the cap while ``_shot & ~killed`` lets
    ``uav_killed`` end the episode instead of truncating the fall short. Non-shot
    drones time out normally at the cap. A shot drone always impacts within ~0.5 s
    of thrust cut, so this never hangs; the collector's outer step cap is a backstop.
    """
    _init_buffers(env)
    killed = env._shot & env._impacted & (env._post_impact_timer >= FALL_RECORD_S)
    falling = env._shot & ~killed
    # episode_length_buf INCREMENTS (0,1,2,...) in Isaac Lab 2.3.2; the cap fires
    # at >= max_episode_length. Suppress it only while a shot drone is mid-fall.
    return (env.episode_length_buf >= env.max_episode_length) & ~falling


def uav_goal_reached(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Termination: drone within GOAL_RADIUS of the goal."""
    _init_buffers(env)
    dp = _drone_pos_env(env)
    return torch.linalg.vector_norm(env.goal_pos - dp, dim=-1) < GOAL_RADIUS


def goal_distance(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward helper (not used by the WM; the env just requires a reward manager).

    The goal is stowed (Evan: "remove the target location"), so this returns zeros
    -- the term contributes nothing and avoids a near-constant far-field distance.
    Kept only so the RewTerm in the config resolves."""
    _init_buffers(env)
    dp = _drone_pos_env(env)
    return torch.zeros(dp.shape[0], device=dp.device)


# ----------------------------------------------------------------------------
# Scene
# ----------------------------------------------------------------------------
def _building_cfg(idx, center, size, color):
    """One urban building (kinematic cuboid, collision enabled, varied facade)."""
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Building{idx}",
        spawn=sim_utils.CuboidCfg(
            size=size,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=center),
    )


@configclass
class UAVTurret3DSceneCfg(InteractiveSceneCfg):
    """Drone + urban buildings + ground + turret + goal + POV camera + light."""

    # robots
    robot: MultirotorCfg = ARL_ROBOT_1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # turret (kinematic box at env origin); rotated each step to track the drone.
    # Kinematic so collisions never move it; collision disabled (NPC marker).
    turret: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Turret",
        spawn=sim_utils.CuboidCfg(
            size=(0.4, 0.4, 0.6),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.2)),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=TURRET_LOCAL),
    )

    # urban ground (dark asphalt). Collision enabled so the drone lands on / crashes
    # onto it (CRASH_Z termination). Kinematic — never moves.
    ground: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Ground",
        spawn=sim_utils.CuboidCfg(
            size=(2 * ARENA_HALF, 2 * ARENA_HALF, 0.2),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.12, 0.13, 0.14)),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -0.1)),
    )

    # urban buildings: FIXED 4x4 grid (varied facades + heights) = the navigable
    # terrain AND the turret's LOS blockers. Kinematic + collision enabled so the
    # drone must fly around them. One RigidObjectCollectionCfg holds all 16.
    buildings: RigidObjectCollectionCfg = RigidObjectCollectionCfg(
        rigid_objects={
            f"building_{i}": _building_cfg(i, center, size, _BUILDING_COLORS[i % len(_BUILDING_COLORS)])
            for i, (center, size) in enumerate(BUILDING_LAYOUT)
        }
    )

    # goal marker (sphere); position randomized per episode to the far edge (A->B).
    # Kinematic, no collision so the drone can reach within GOAL_RADIUS.
    goal: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Goal",
        spawn=sim_utils.SphereCfg(
            radius=0.3,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.8, 0.2)),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(ARENA_HALF * 0.9, 0.0, DRONE_SPAWN_Z)),
    )

    # drone POV camera: parented under the drone's MOVING root body link
    # (base_link), 224x224. convention="ros" applies the offset in the BODY frame
    # so the camera FOLLOWS the drone's pose (position AND heading — yaw is an
    # explicit action now, so the camera only turns when the policy commands it;
    # body +X = forward).
    # rot=(0.5,-0.5,0.5,-0.5) maps the ROS optical axes (forward +Z, up -Y) onto
    # body forward (+X = travel) and body up (+Z = sky) — an UPRIGHT forward view.
    # CRITICAL: prim_path MUST be under /Robot/base_link (the moving body), NOT
    # /Robot (the static articulation-root Xform). /Robot/Camera renders a FIXED
    # perspective (the camera never moves with the drone) — Evan's 2026-07-16
    # critique. /Robot/base_link/Camera is true first-person. See Isaac Lab 2.3.2:
    # ArticulationCfg spawns the rigid body as a CHILD of the root Xform.
    # NOTE: DLSS is disabled at the SIM level (sim.antialiasing_mode="FXAA" in
    # __post_init__), NOT here — TiledCameraCfg has no dlss_cfg field in 2.3.2.
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link/Camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.15, 0.0, 0.05), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            # focal 12 mm + 20.955 mm aperture -> ~82 deg HFOV (was 24 mm / ~47 deg).
            # Wider lens so a close building doesn't fill the frame (Evan 2026-07-16);
            # some prop may clip the frame bottom -- review in the critique PNGs.
            focal_length=12.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 80.0)
        ),
        width=224,
        height=224,
    )

    # lights: neutral dome fills the urban canyon (geometry, not a white void, is
    # what makes the POV meaningful). DLSS/FXAA set at the sim level.
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=900.0),
    )


# ----------------------------------------------------------------------------
# MDP config groups
# ----------------------------------------------------------------------------
@configclass
class ActionsCfg:
    velocity_command = VelocityToThrustActionCfg(asset_name="robot")


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # the drone POV — the pixels the WM trains on. normalize=False keeps raw
        # uint8 (the collector JPEG-encodes this; the WM dataloader normalizes).
        pixels = ObsTerm(
            func=mdp_image,
            params={"sensor_cfg": SceneEntityCfg("tiled_camera"), "data_type": "rgb", "normalize": False},
        )

        def __post_init__(self):
            self.enable_corruption = False
            # single image term; concatenate so obs["policy"] is a tensor, not a dict
            self.concatenate_terms = True

    @configclass
    class StateCfg(ObsGroup):
        state = ObsTerm(func=uav_state)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    state: StateCfg = StateCfg()


@configclass
class EventCfg:
    # reset: drone spawn randomization (built-in) + obstacle/goal/buffer reset (custom DR)
    reset_drone = EventTerm(
        func=reset_root_state_uniform,
        mode="reset",
        params={
            # Fallback spawn pose; uav_reset_spawn (reset_world, declared below)
            # OVERWRITES this for every env with a random STREET_GRID intersection
            # + random heading (Evan: "randomize the starting location"). Kept so
            # the root state is initialized cleanly before the override write.
            "pose_range": {"x": (-ARENA_HALF * 0.9, -ARENA_HALF * 0.9), "y": (-8.0, 8.0),
                           "z": (DRONE_SPAWN_Z, DRONE_SPAWN_Z),
                           "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)},
            "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                               "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)},
        },
    )
    reset_world = EventTerm(func=uav_reset_spawn, mode="reset")
    # interval: turret tracking + danger computation each env step
    world_step = EventTerm(func=uav_world_step, mode="interval", interval_range_s=(0.1, 0.1))


@configclass
class RewardsCfg:
    # Minimal — the WM does not use rewards, but the env requires a reward manager.
    goal_dist = RewTerm(func=goal_distance, weight=-0.1)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=uav_time_out, time_out=True)
    crash = DoneTerm(func=uav_crash)
    killed = DoneTerm(func=uav_killed)
    reached_goal = DoneTerm(func=uav_goal_reached)


# ----------------------------------------------------------------------------
# Env config
# ----------------------------------------------------------------------------
@configclass
class UAVTurret3DEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the 3D UAV+turret environment."""

    scene: UAVTurret3DSceneCfg = UAVTurret3DSceneCfg(num_envs=64, env_spacing=50.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        self.decimation = 10
        self.episode_length_s = 30.0  # 300 steps — long enough for A->B through the city grid
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation
        # The thruster spool (tau_inc/dec) advances in actuator.compute(), which
        # the action manager calls once per ENV step (decimation sim steps), not
        # per sim step. So the mixing factor 1-exp(-dt/tau) must use the env-step
        # dt = decimation*sim.dt — using sim.dt makes the spool 10x too slow, so
        # from the init rps (~2 N total vs ~12 N hover) the drone sinks to the
        # ground before thrust builds and never recovers. (The shipped ARL task
        # uses sim.dt, but an RL policy learns to pre-compensate; our reactive
        # controller can't, so we use the env-step dt for a fast, correct spool.)
        self.scene.robot.actuators["thrusters"].dt = self.decimation * self.sim.dt
        # Make the thruster spool near-instant (tau_inc/dec -> 1 ms). The shipped
        # tau_inc=0.05-0.08 is a slow rise: with the asymmetric init thrust it lagged
        # the controller's tilt command, the drone over-tilted, the PD reversed, and
        # the drone rolled back and forth (thrust alternating [lo,hi,lo,hi] <->>
        # [hi,lo,hi,lo]) delivering net <hover thrust -> it sank to the ground while
        # thrashing. A near-instant spool makes delivered thrust == command within one
        # step, so the init rps and spool-up lag stop matter and the attitude PD sees
        # a clean plant. (Realism cost: idealized motors. The rigid-body + thrust +
        # gravity + collision dynamics are still fully simulated; only the motor
        # first-order lag is removed, which is negligible for POV world-model data.)
        self.scene.robot.actuators["thrusters"].tau_inc_range = (0.001, 0.001)
        self.scene.robot.actuators["thrusters"].tau_dec_range = (0.001, 0.001)
        # Zero the init rotor rps so the drone spawns with ZERO thrust (not the
        # ~113 N / 9.3x hover that the default init rps produces via thrust_const).
        # That huge init thrust is applied at reset; with per-motor thrust_const
        # differences it spins the drone, the yaw PD saturates trying to catch the
        # spin, and the saturation corrupts the collective -> the drone climbs
        # uncontrollably. With rps=0 there is no reset kick; the near-instant spool
        # (tau=1 ms above) brings thrust to the controller's command on step 0, so
        # the drone starts in equilibrium. (rps=0 -> curr_thrust = thrust_const*0 = 0
        # regardless of the sampled thrust_const, unlike a nonzero rps whose thrust
        # varies wildly with the randomized thrust_const.)
        self.scene.robot.init_state.rps = {name: 0.0 for name in self.scene.robot.init_state.rps}
        self.sim.physics_material = sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        )
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        # DLSS (the sim default antialiasing_mode) deadlocks the first render at
        # 224x224: it upscales internally to 260x260, below DLSS's 300px minimum
        # (verified: smoke hung 25 min at the DLSS "below minimal input resolution"
        # warning). FXAA is spatial — no resolution floor, no temporal ghosting
        # (good for a frame-predicting WM). The 224x224 res is fixed by the WM.
        self.sim.antialiasing_mode = "FXAA"

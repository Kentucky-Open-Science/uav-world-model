"""Collect a few UAVTurret3D episodes as Foxglove MCAP for Evan to critique.

This is the **pre-training critique gate** (Task #14): a small collect that Evan
reviews in Foxglove BEFORE any LeWM training. It writes MCAP only — no lance
(the training lance collector is Task #12, run after sign-off, since Evan may
ask for POV/scene changes that would invalidate a lance dataset).

Per episode, one MCAP with:
  /drone/pov/image   foxglove.RawImage     224x224 RGB = the WM's training pixels
  /scene             foxglove.SceneUpdate  drone+turret+obstacles+aim-line+state-text
  /drone/tf          foxglove.FrameTransform  drone world pose (attach Foxglove cam to drone)
  /state             foxglove.Log          numeric state + danger flags per step (Log panel)

Episode ends on terminated|truncated (Isaac Lab ``step`` auto-resets). A per-env
MCAP writer rotates on each end. Coordinates are env-local (subtract env_origin)
so every episode is centered at the Foxglove origin. First/mid/last POV PNG per
episode are also saved for a quick non-Foxglove preview (agent does NOT inspect).

MCAP libs (mcap, mcap-protobuf-support, foxglove-schemas-protobuf) are pip-
installed to /workspace/libs (bind-mounted host dir); added to sys.path here.

Run inside the isaac-lab container (on the GPU box):
    /workspace/isaaclab/isaaclab.sh -p /workspace/scripts/collect_uav3d_critique.py \
        --headless --enable_cameras --num_envs 6 --episodes 6 \
        --output_dir /workspace/output/uav3d_critique
"""
import argparse
import math
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="UAVTurret3D critique collector (Foxglove MCAP).")
parser.add_argument("--num_envs", type=int, default=6)
parser.add_argument("--episodes", type=int, default=6)
parser.add_argument("--output_dir", type=str, default="/workspace/output/uav3d_critique")
parser.add_argument("--q", type=float, default=0.3,
                    help="policy: provoke (vs wander) fraction. ~0.3 = 70%% wander / 30%% provoke "
                         "(drunken explorer: wander teaches flight dynamics, provoke manufactures danger).")
parser.add_argument("--r", type=float, default=0.3,
                    help="policy: commit (vs evade) fraction when in danger. Commit flies into the "
                         "turret -> kill + recorded fall; evade strafes/climbs/spins to (maybe) escape.")
parser.add_argument("--noise", type=float, default=0.0,
                    help="policy: per-step action noise. 0 by default -- the drunken explorer holds "
                         "each maneuver 10-30 steps (no per-step re-sampling to jitter) and yaw is "
                         "explicit (no auto-yaw to amplify noise). Re-enable only to test.")
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Imports after AppLauncher (pxr/omni only exist post-startup)."""
import gymnasium as gym
import torch

# MCAP stack lives in the bind-mounted /workspace/libs (not in the container image).
sys.path.insert(0, "/workspace/libs")
from google.protobuf.timestamp_pb2 import Timestamp  # noqa: E402
from mcap_protobuf.writer import Writer as McapWriter  # noqa: E402

import uav_wm.envs  # noqa: F401,E402  (registers Isaac-UAVTurret3D-v0)
from uav_wm.data.policies_3d import ExplorationPolicy3D, WaypointPolicy  # noqa: E402
from uav_wm.envs.uav_turret_3d import (  # noqa: E402
    CRASH_Z,
    EP_DRUNK,
    EP_NO_TURRET,
    EP_WAYPOINT_TURRET,
    FOV_HALF,
    NUM_OBSTACLES,
    TURRET_RANGE,
    roll_episode_types,
)
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

TASK = "Isaac-UAVTurret3D-v0"
STEP_DT = 0.1  # env step dt (sim.dt 0.01 * decimation 10) -> 0.1 s; drives MCAP timeline


# ----------------------------------------------------------------------------
# Foxglove message builders (field names verified against foxglove-schemas-protobuf 0.3.0)
# ----------------------------------------------------------------------------
def _ts(ns: int) -> Timestamp:
    return Timestamp(seconds=ns // 1_000_000_000, nanos=ns % 1_000_000_000)


def _vec3(x, y, z):
    from foxglove_schemas_protobuf.Vector3_pb2 import Vector3
    return Vector3(x=float(x), y=float(y), z=float(z))


def _quat(w, x, y, z):
    from foxglove_schemas_protobuf.Quaternion_pb2 import Quaternion
    return Quaternion(w=float(w), x=float(x), y=float(y), z=float(z))


def _pose(px, py, pz, qw=1.0, qx=0.0, qy=0.0, qz=0.0):
    from foxglove_schemas_protobuf.Pose_pb2 import Pose
    return Pose(position=_vec3(px, py, pz), orientation=_quat(qw, qx, qy, qz))


def _color(r, g, b, a=1.0):
    from foxglove_schemas_protobuf.Color_pb2 import Color
    return Color(r=float(r), g=float(g), b=float(b), a=float(a))


def _point(x, y, z):
    from foxglove_schemas_protobuf.Point3_pb2 import Point3
    return Point3(x=float(x), y=float(y), z=float(z))


class EpisodeWriter:
    """One Foxglove MCAP per episode. All coordinates env-local (origin-centered)."""

    def __init__(self, path: str, ep_label: str):
        self.path = path
        self.ep_label = ep_label
        self.w = McapWriter(path)
        self.t0_ns = time.time_ns()
        self.steps = 0
        self.danger_frames = 0
        self.outcome = "open"

    def _tns(self, step: int) -> int:
        return self.t0_ns + step * int(STEP_DT * 1e9)

    def write_step(self, step, pov_hwc_rgb, drone_xyz, drone_quat_wxyz,
                   turret_xyz, aim_dir, obstacles, st, has_turret=True):
        """st = 21-dim state (torch or list). obstacles = list of (center, full_extents)."""
        tns = self._tns(step)
        ts = _ts(tns)
        danger = int(st[13]); in_range = int(st[14]); los = int(st[15])
        aimed = int(st[16]); fire = float(st[17]); dist_t = float(st[12])
        self.steps = step + 1
        if danger:
            self.danger_frames += 1

        # --- /drone/pov/image : RawImage ---
        h, w, _ = pov_hwc_rgb.shape
        from foxglove_schemas_protobuf.RawImage_pb2 import RawImage
        img = RawImage(timestamp=ts, frame_id="drone", width=w, height=h,
                       encoding="rgb8", step=w * 3, data=bytes(pov_hwc_rgb.tobytes()))
        self.w.write_message("/drone/pov/image", img, log_time=tns, publish_time=tns)

        # --- /scene : SceneUpdate (single entity "world", primitives in world coords) ---
        from foxglove_schemas_protobuf.SceneUpdate_pb2 import SceneUpdate
        from foxglove_schemas_protobuf.SceneEntity_pb2 import SceneEntity
        from foxglove_schemas_protobuf.CubePrimitive_pb2 import CubePrimitive
        from foxglove_schemas_protobuf.LinePrimitive_pb2 import LinePrimitive
        from foxglove_schemas_protobuf.TextPrimitive_pb2 import TextPrimitive
        cubes = []
        # drone (blue)
        cubes.append(CubePrimitive(pose=_pose(*drone_xyz), size=_vec3(0.3, 0.3, 0.1),
                                   color=_color(0.2, 0.4, 1.0)))
        # turret body (red) -- omitted for no-turret episodes (turret is stowed far)
        if has_turret:
            cubes.append(CubePrimitive(pose=_pose(*turret_xyz), size=_vec3(0.4, 0.4, 0.6),
                                       color=_color(0.9, 0.2, 0.2)))
        # obstacles (grey)
        for cen, ext in obstacles:
            cubes.append(CubePrimitive(pose=_pose(cen[0], cen[1], cen[2]),
                                       size=_vec3(ext[0], ext[1], ext[2]),
                                       color=_color(0.5, 0.5, 0.5)))
        # no goal marker (Evan: "remove the target location") -- empty spheres list
        spheres = []
        # aim line: turret -> turret + aim_dir * TURRET_RANGE (red if danger else
        # orange). Omitted for no-turret episodes (no turret to aim).
        lines = []
        if has_turret:
            ax, ay, az = aim_dir
            tx, ty, tz = turret_xyz
            line_color = _color(1.0, 0.1, 0.1) if danger else _color(1.0, 0.5, 0.1)
            lines.append(LinePrimitive(
                type=LinePrimitive.LINE_LIST, pose=_pose(0, 0, 0), thickness=0.04,
                scale_invariant=False, points=[
                    _point(tx, ty, tz),
                    _point(tx + ax * TURRET_RANGE, ty + ay * TURRET_RANGE, tz + az * TURRET_RANGE),
                ], color=line_color))
        # state text above the drone
        txt = (f"{self.ep_label} step{step} d={danger} r={in_range} l={los} a={aimed} "
               f"fire={fire:.2f} dist={dist_t:.1f}")
        texts = [TextPrimitive(pose=_pose(drone_xyz[0], drone_xyz[1], drone_xyz[2] + 0.6),
                               billboard=False, font_size=0.18, scale_invariant=False,
                               color=_color(1.0, 1.0, 1.0) if not danger else _color(1.0, 0.2, 0.2),
                               text=txt)]
        ent = SceneEntity(timestamp=ts, frame_id="world", id="world",
                          frame_locked=False, cubes=cubes, spheres=spheres, lines=lines, texts=texts)
        su = SceneUpdate(entities=[ent])
        self.w.write_message("/scene", su, log_time=tns, publish_time=tns)

        # --- /drone/tf : FrameTransform (world -> drone) ---
        from foxglove_schemas_protobuf.FrameTransform_pb2 import FrameTransform
        ft = FrameTransform(timestamp=ts, parent_frame_id="world", child_frame_id="drone",
                            translation=_vec3(*drone_xyz),
                            rotation=_quat(drone_quat_wxyz[0], drone_quat_wxyz[1],
                                           drone_quat_wxyz[2], drone_quat_wxyz[3]))
        self.w.write_message("/drone/tf", ft, log_time=tns, publish_time=tns)

        # --- /state : Log (numeric state, for the Log panel) ---
        from foxglove_schemas_protobuf.Log_pb2 import Log
        dp = [float(st[0]), float(st[1]), float(st[2])]
        msg = (f"step{step} drone=[{dp[0]:+.2f},{dp[1]:+.2f},{dp[2]:+.2f}] "
               f"dist_t={dist_t:.2f} danger={danger} in_range={in_range} los={los} "
               f"aimed={aimed} fire={fire:.2f}")
        lvl = Log.WARNING if danger else Log.INFO
        log = Log(timestamp=ts, level=lvl, message=msg, name="uav3d")
        self.w.write_message("/state", log, log_time=tns, publish_time=tns)

    def close(self):
        self.w.finish()


def pov_rgb(pix_obs, i):
    """env i's POV as (H, W, 3) uint8 numpy."""
    fr = pix_obs[i]
    if fr.dim() == 3 and fr.shape[0] <= 4:          # (C, H, W)
        if fr.shape[0] == 4:
            fr = fr[:3]
        fr = fr.permute(1, 2, 0)
    if fr.dtype != torch.uint8:
        fr = (fr.clamp(0.0, 1.0) * 255.0).to(torch.uint8)
    return fr.cpu().numpy().astype("uint8")[:, :, :3]


def infer_outcome(st, term, trunc):
    """Infer episode outcome from the PRE-step state (the last frame of the episode).

    No 'reached_goal': the goal is stowed (Evan: "remove the target location") and
    goal_rel is zeroed in the state."""
    drone_z = float(st[2]); fire = float(st[17])
    if fire >= 1.0:
        return "killed"
    if drone_z < CRASH_Z:
        return "crash"
    if trunc:
        return "timeout"
    if term:
        return "terminated"
    return "ended"


def main() -> None:
    out = Path(args_cli.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    env_cfg = parse_env_cfg(TASK, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(TASK, cfg=env_cfg)
    base = env.unwrapped
    n = base.num_envs
    dev = base.device
    print(f"[COLLECT] env={type(base).__name__} num_envs={n} device={dev}")

    torch.manual_seed(args_cli.seed)  # reproducible spawns + episode-type rolls
    drunk = ExplorationPolicy3D(q=args_cli.q, r=args_cli.r, noise=args_cli.noise,
                                seed=args_cli.seed, device=str(dev))
    wp = WaypointPolicy(seed=args_cli.seed + 1, device=str(dev))
    drunk._ensure_buffers(n)
    wp._ensure_buffers(n)

    obs, _ = env.reset()
    state = obs["state"]  # (N, 21) -- 20-dim + drone_yaw at index 20
    assert state.shape[-1] == 21

    def begin_episode(env_ids):
        """Set up policies for envs whose episode just (re)started, from the
        episode_type the env committed during reset; then roll _next_episode_type
        for the FOLLOWING episode (one-ahead). Waypoint envs get a fresh route;
        drunk envs get a maneuver reset."""
        env_ids = list(env_ids)
        if not env_ids:
            return
        et = base.episode_type[env_ids].tolist()
        wp_ids = [e for e, t in zip(env_ids, et) if t != EP_DRUNK]
        drunk_ids = [e for e, t in zip(env_ids, et) if t == EP_DRUNK]
        if wp_ids:
            spawn = (base.scene["robot"].data.root_pos_w - base.scene.env_origins)[wp_ids]
            wp.reset_envs(wp_ids, spawn)   # goal-less fixed route + per-ep speed
        wp._active[drunk_ids] = False
        # relocate the waypoint+turret turret onto a mid-route waypoint (xy only;
        # keep the reset ground z). uav_world_step re-writes the turret pose next step.
        wpt = [e for e in wp_ids if base.episode_type[e].item() == EP_WAYPOINT_TURRET]
        if wpt:
            wpt_t = torch.as_tensor(wpt, device=dev, dtype=torch.long)
            mid = torch.clamp(wp._wp_len[wpt_t] // 2, min=1)
            mid_wp = wp._waypoints[wpt_t, mid]            # (m, 3) env-local
            base.turret_pos[wpt_t, 0] = mid_wp[:, 0]
            base.turret_pos[wpt_t, 1] = mid_wp[:, 1]
        if drunk_ids:
            drunk.reset_envs(drunk_ids)
        base._next_episode_type[env_ids] = roll_episode_types(len(env_ids), dev)

    begin_episode(list(range(n)))

    writers = [None] * n
    ep_step = [0] * n
    ep_id = [0]
    manifest = []
    episodes_done = [0]

    def start_ep(i):
        eid = ep_id[0]; ep_id[0] += 1
        et = int(base.episode_type[i].item())
        tag = {EP_NO_TURRET: "noturret", EP_WAYPOINT_TURRET: "wpturret", EP_DRUNK: "drunk"}.get(et, "?")
        path = str(out / f"episode_{eid:03d}.mcap")
        writers[i] = EpisodeWriter(path, f"ep{eid}_{tag}")
        writers[i].episode_type = et
        ep_step[i] = 0
        return eid, path

    # open one writer per env, up to --episodes
    for i in range(min(n, args_cli.episodes)):
        start_ep(i)

    def geometry(i):
        """Pull env i's geometry as env-local python tuples/lists."""
        drone_w = base.scene["robot"].data.root_pos_w[i]
        q = base.scene["robot"].data.root_quat_w[i]          # (w,x,y,z)
        origin = base.scene.env_origins[i]
        drone = (drone_w - origin).cpu()
        tyaw = float(base.turret_yaw[i].item()); tpit = float(base.turret_pitch[i].item())
        aim = (math.cos(tpit) * math.cos(tyaw), math.cos(tpit) * math.sin(tyaw), math.sin(tpit))
        obs_c = base.obstacle_centers[i].cpu()               # (NUM_OBSTACLES,3) env-local
        obs_h = base.obstacle_half[i].cpu()                  # half-extents
        obs_list = [((float(obs_c[k, 0]), float(obs_c[k, 1]), float(obs_c[k, 2])),
                     (2.0 * float(obs_h[k, 0]), 2.0 * float(obs_h[k, 1]), 2.0 * float(obs_h[k, 2])))
                    for k in range(NUM_OBSTACLES)]
        turret = (float(base.turret_pos[i, 0]), float(base.turret_pos[i, 1]), float(base.turret_pos[i, 2]))
        has_turret = bool(base.episode_type[i].item() != EP_NO_TURRET)
        return (drone.tolist(), (float(q[0]), float(q[1]), float(q[2]), float(q[3])),
                turret, aim, obs_list, has_turret)

    # generous cap: each env can run several episodes before --episodes is met
    max_steps = args_cli.episodes * 350 + 600  # 300-step episodes (fixed path through the city)
    for gstep in range(max_steps):
        if gstep % 50 == 0:
            # lightweight progress so a long camera run is observable (no per-step spam).
            nd = episodes_done[0]
            dz = sum(1 for w in writers if w is not None and w.danger_frames > 0)
            na = sum(1 for w in writers if w is not None)
            print(f"[COLLECT] gstep={gstep:4d} done={nd}/{args_cli.episodes} "
                  f"active={na} envs_with_danger={dz}", flush=True)
        pix = obs["policy"]
        # write current (pre-step) frame for each env with an active writer
        for i in range(n):
            if writers[i] is None:
                continue
            drone, dq, turret, aim, obs_list, has_turret = geometry(i)
            writers[i].write_step(ep_step[i], pov_rgb(pix, i), drone, dq,
                                  turret, aim, obs_list, state[i], has_turret)
            ep_step[i] += 1
        # also save a PNG preview at first / mid / (near)last step per episode.
        # Best-effort: PIL may be absent in the container; the MCAP (the actual
        # gate deliverable) only needs mcap/protobuf. Never let a preview failure
        # crash the collection.
        for i in range(n):
            if writers[i] is None:
                continue
            s = ep_step[i]
            target = {1, 40, 78}
            if s in target:
                try:
                    from PIL import Image
                    Image.fromarray(pov_rgb(pix, i)).save(
                        str(out / f"{Path(writers[i].path).stem}_step{s:02d}.png"))
                except Exception as e:
                    print(f"[COLLECT] (preview PNG skipped for env {i} step {s}: {e})")

        act_drunk = drunk.get_action(state, base.obstacle_centers, base.obstacle_half)
        act_wp = wp.get_action(state, base.obstacle_centers, base.obstacle_half)
        mode_wp = base.episode_type != EP_DRUNK
        action = torch.where(mode_wp.unsqueeze(-1), act_wp, act_drunk)
        obs, rew, term, trunc, info = env.step(action)
        new_state = obs["state"]

        for i in range(n):
            if writers[i] is None:
                continue
            t = bool(term[i].item()); tr = bool(trunc[i].item())
            if t or tr:
                writers[i].outcome = infer_outcome(state[i], t, tr)
                writers[i].close()
                manifest.append((Path(writers[i].path).name, writers[i].steps,
                                 writers[i].outcome, writers[i].danger_frames,
                                 getattr(writers[i], "episode_type", -1)))
                episodes_done[0] += 1
                writers[i] = None
                if episodes_done[0] < args_cli.episodes:
                    start_ep(i)
        # Every auto-reset env (recorded or not) starts a fresh episode -> clear
        # its maneuver state so a fresh spawn doesn't inherit a stale evade-spin.
        reset_mask = term | trunc
        if reset_mask.any():
            begin_episode(reset_mask.nonzero(as_tuple=False).squeeze(-1).tolist())
        state = new_state
        if episodes_done[0] >= args_cli.episodes and all(w is None for w in writers):
            break

    # close any still-open writers (shouldn't happen, but be safe)
    for i in range(n):
        if writers[i] is not None:
            writers[i].outcome = "open"
            writers[i].close()
            manifest.append((Path(writers[i].path).name, writers[i].steps,
                             writers[i].outcome, writers[i].danger_frames,
                             getattr(writers[i], "episode_type", -1)))

    env.close()

    # text manifest (agent reports this; does NOT inspect images)
    man_path = out / "manifest.txt"
    type_names = {EP_NO_TURRET: "no_turret", EP_WAYPOINT_TURRET: "wpturret", EP_DRUNK: "drunk"}
    type_dang = {EP_NO_TURRET: 0, EP_WAYPOINT_TURRET: 0, EP_DRUNK: 0}
    with open(man_path, "w") as f:
        f.write(f"# UAVTurret3D critique collect — {len(manifest)} episodes\n")
        f.write(f"# num_envs={n} q={args_cli.q} r={args_cli.r} noise={args_cli.noise} seed={args_cli.seed}\n")
        f.write("# episode                steps outcome       danger_frames type\n")
        tot_d = 0
        for name, steps, outcome, df, et in manifest:
            f.write(f"{name:24s} {steps:5d} {outcome:13s} {df:4d} {type_names.get(et, '?')}\n")
            tot_d += df
            if et in type_dang:
                type_dang[et] += df
        f.write(f"# total danger frames: {tot_d}\n")
        for et in (EP_NO_TURRET, EP_WAYPOINT_TURRET, EP_DRUNK):
            f.write(f"# {type_names[et]:9s} danger_frames: {type_dang[et]}\n")
    outcomes = {}
    for _, _, o, _, _ in manifest:
        outcomes[o] = outcomes.get(o, 0) + 1
    print(f"[COLLECT] DONE episodes={len(manifest)} outcomes={outcomes} danger_frames={tot_d}")
    print(f"[COLLECT] manifest -> {man_path}")
    print(f"[COLLECT] MCAPs + PNGs -> {out}")
    print("[COLLECT] UAVTURRET3D CRITIQUE OK")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        print("[COLLECT] CRASHED (see traceback above)", flush=True)
    finally:
        simulation_app.close()

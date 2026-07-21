#!/usr/bin/env python
"""Phase 7 live demo: fly the danger-aware CEM planner (or the WaypointPolicy
baseline) in the Isaac UAVTurret3D env and measure survival / kill rate.

Two-process (see planner_server.py): this script runs in the Isaac Lab container
(swm-free; py3.11). The three A->B modes (nav / planner / detector) share IDENTICAL
client logic: send the POV batch + state + goal + obstacles to the host NavServer and
step the env with the returned 4-dim action. The modes differ ONLY in one host-side
knob each (danger_weight / reactive_flee), so kill/reach differences attribute cleanly
to imagination vs reaction. nav=oblivious A->B (rounds the corner, dies); planner=WM
A->B (imagines the turret around the corner, detours to B); detector=reactive A->B
(probe flee, fires too late). All three fly the corner-ambush scenario (begin_episode):
spawn at A, goal B, turret T just around a building corner -- the oblivious route rounds
the corner into T. waypoint uses the local WaypointPolicy alone (wander baseline);
showcase flies oblivious head-on at the turret for the Result-1 phantom signal.

Mirrors collect_uav_3d.py's step/reset pattern: Isaac auto-resets on term/trunc,
so the obs returned by env.step is the NEW episode's first frame; we track a
just_reset mask so the host clears that env's 3-frame context.

  docker run --gpus all --network=host ... \
    /workspace/isaaclab/isaaclab.sh -p /workspace/scripts/live/live_demo.py \
      --headless --enable_cameras --num_envs 16 --episodes 40 --mode planner \
      --planner_port 5557
"""
import argparse
import math
import os
import pickle
import random
import socket
import struct
import time
from collections import Counter

import numpy as np
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Phase 7 live demo (planner vs detector vs waypoint vs nav).")
parser.add_argument("--mode", choices=["nav", "planner", "detector", "waypoint", "showcase"],
                    default="planner",
                    help="nav=oblivious A->B (corner-ambush: rounds the corner, dies); "
                         "planner=WM A->B (imagines the turret around the corner, detours to B); "
                         "detector=reactive A->B (probe flee, fires too late); "
                         "waypoint=wander baseline; showcase=oblivious head-on (Result-1 phantom).")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--episodes", type=int, default=40)
parser.add_argument("--planner_host", default="127.0.0.1")
parser.add_argument("--planner_port", type=int, default=5557)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--force_type", choices=["wpturret", "mix"], default="wpturret",
                    help="wpturret: all episodes EP_WAYPOINT_TURRET (turret on route). "
                         "mix: the env's natural 40/40/20 type mix.")
parser.add_argument("--mcap_dir", default=None,
                    help="If set, log one Foxglove MCAP per episode here (the showcase). "
                         "Container path, bind-mounted to host. None = off (use for the "
                         "matched n=40 comparison runs; set for the showcase run).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import uav_wm.envs  # noqa: E402,F401  (registers Isaac-UAVTurret3D-v0)
from uav_wm.data.policies_3d import WaypointPolicy  # noqa: E402
from uav_wm.envs.uav_turret_3d import (  # noqa: E402
    CRASH_Z,
    DRONE_SPAWN_Z,
    EP_NO_TURRET,
    EP_WAYPOINT_TURRET,
    NUM_OBSTACLES,
    STREET_GRID,
    TURRET_BASE_Z,
)
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from mcap_viz import LiveEpisodeWriter  # noqa: E402  (scripts/live is sys.path[0])

TASK = "Isaac-UAVTurret3D-v0"
FIRE_IDX = 17  # fire_prog in state (infer_outcome reads st[17])
SHOWCASE_CRUISE = 0.7  # oblivious forward vx in showcase mode; MUST match
                       # PhantomServer.cruise in planner_server.py so the WM's
                       # recorded context action == the action actually flown.

# Collinear A->T->B scenario templates (Seed 0 + 90deg rotations). Each tuple is
# (A, B, T, yaw): spawn at A with heading yaw, goal B, turret T dead-ahead on the
# beeline. Seed 0: A=(-12,0) heading +x, B=(12,0), T=(0,0). The beeline runs along
# the y=0 STREET (a clear corridor -- the fixed 4x4 building grid sits at y=+-3,+-9,
# never on the axes), so the OBLIVIOUS drone (dw=0) is pulled straight +x by the
# goal term, straight into T (dist 0, in range + LOS clear along the axis -> killed).
# B is 12 m from T (outside the 8 m kill zone). T is dead-ahead (0 deg bearing) the
# whole approach -> in the WM's POV from the start, so the WM imagines the approach
# and fires BEFORE the drone is in range: the planner (dw>0) turns / holds back
# (cautious) and survives; the detector (present-frame probe) fires only at dist<8
# (its training label) -> too late -> dies. This is the "imagination > detection for
# survival" story; whether the planner then navigates AROUND T to reach B is a
# stretch (the 2 s / ~1 m imagined horizon is short for a building-threading detour)
# -- reported honestly from the run, not assumed. T=(0,0) is the rotation center, so
# all four beelines run along an axis (always a clear street); A/B rotate around T.
_NAV_TEMPLATES = []
_A, _B, _T, _YAW = (-12.0, 0.0), (12.0, 0.0), (0.0, 0.0), 0.0
for _k in range(4):
    _NAV_TEMPLATES.append((_A, _B, _T, _YAW + _k * math.pi / 2))
    _A = (-_A[1], _A[0]); _B = (-_B[1], _B[0])  # T=(0,0) fixed under rotation (no-op)


def pov_hwc(pix_obs, i):
    """env i's POV as (H, W, 3) uint8 numpy (matches collect_uav_3d.pov_hwc)."""
    fr = pix_obs[i]
    if fr.dim() == 3 and fr.shape[0] <= 4:  # (C, H, W)
        if fr.shape[0] == 4:
            fr = fr[:3]
        fr = fr.permute(1, 2, 0)
    if fr.dtype != torch.uint8:
        fr = (fr.clamp(0.0, 1.0) * 255.0).to(torch.uint8)
    return fr.detach().cpu().numpy().astype("uint8")[:, :, :3]


def infer_outcome(st, term, trunc):
    drone_z = float(st[2])
    fire = float(st[FIRE_IDX])
    if fire >= 1.0:
        return "killed"
    if drone_z < CRASH_Z:
        return "crash"
    if trunc:
        return "timeout"
    if term:
        # Done terms are exactly time_out(trunc)/crash/killed/reached_goal. Having
        # ruled out killed+crash+timeout, a terminating step is reached_goal -- the
        # drone reached B (uav_goal_reached fires once goal_pos is un-stowed in
        # begin_episode). For goal-stowed modes (showcase/waypoint) reached_goal can
        # never fire, so this branch is inert there.
        return "reached_B"
    return "ended"


def recv_exact(s, n):
    b = bytearray()
    while len(b) < n:
        c = s.recv(n - len(b))
        if not c:
            raise ConnectionError("closed")
        b.extend(c)
    return bytes(b)


def send_msg(s, o):
    d = pickle.dumps(o, protocol=pickle.HIGHEST_PROTOCOL)
    s.sendall(struct.pack(">I", len(d)) + d)


def recv_msg(s):
    (n,) = struct.unpack(">I", recv_exact(s, 4))
    return pickle.loads(recv_exact(s, n))


def main():
    torch.manual_seed(args_cli.seed)
    env_cfg = parse_env_cfg(TASK, device=args_cli.device, num_envs=args_cli.num_envs)
    if args_cli.mode in ("nav", "planner", "detector"):
        # Collinear A->T->B: the oblivious drone dies early (flown into T); the
        # planner may hold back / nudge (survives=timeout) or, in the stretch case,
        # thread a detour to B. 60 s gives the planner room to maneuver at the
        # action_std-throttled cruise (~0.6 m/s). max_episode_length is derived as
        # episode_length_s/step_dt (step_dt=0.1) in the env __init__; the time-out
        # term reads it (uav_turret_3d.py:653). Env file is read-only, so override
        # via cfg (no env edit).
        env_cfg.episode_length_s = 60.0
    env = gym.make(TASK, cfg=env_cfg)
    base = env.unwrapped
    n = base.num_envs
    dev = base.device
    print(f"[LIVE] mode={args_cli.mode} num_envs={n} device={dev} "
          f"force_type={args_cli.force_type} episodes={args_cli.episodes}", flush=True)

    wp = WaypointPolicy(seed=args_cli.seed + 1, device=str(dev))
    wp._ensure_buffers(n)
    if args_cli.force_type == "wpturret":
        base._next_episode_type[:] = EP_WAYPOINT_TURRET

    sock = None
    if args_cli.mode in ("nav", "planner", "detector", "showcase"):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((args_cli.planner_host, args_cli.planner_port))
        print(f"[LIVE] connected to {args_cli.mode} server {args_cli.planner_host}:{args_cli.planner_port}",
              flush=True)

    # --- MCAP showcase logging (one Foxglove file per episode) -----------------
    mcap_on = bool(args_cli.mcap_dir)
    writers = [None] * n          # LiveEpisodeWriter per env, or None
    ep_step = [0] * n            # step within the current episode
    ep_id = [0] * n              # episode index per env (for the filename)
    nav_ep = [0] * n             # nav-family episode index per env (scenario cycle seed)
    trails = [[] for _ in range(n)]  # drone positions this episode (LINE_STRIP)
    if mcap_on:
        os.makedirs(args_cli.mcap_dir, exist_ok=True)
        print(f"[LIVE] MCAP showcase logging -> {args_cli.mcap_dir}", flush=True)

    def env_geometry(i):
        """Per-env env-local geometry for MCAP. Mirrors collect_uav3d_critique.geometry."""
        robot = base.scene["robot"].data
        origin = base.scene.env_origins[i]
        drone_xyz = (robot.root_pos_w[i] - origin).cpu().tolist()
        q = robot.root_quat_w[i].cpu()              # (w, x, y, z)
        drone_quat = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
        tyaw = float(base.turret_yaw[i]); tpit = float(base.turret_pitch[i])
        aim = (math.cos(tpit) * math.cos(tyaw), math.cos(tpit) * math.sin(tyaw), math.sin(tpit))
        turret_xyz = (float(base.turret_pos[i, 0]), float(base.turret_pos[i, 1]),
                      float(base.turret_pos[i, 2]))
        oc = base.obstacle_centers[i].cpu(); oh = base.obstacle_half[i].cpu()
        obstacles = [((float(oc[k, 0]), float(oc[k, 1]), float(oc[k, 2])),
                      (2.0 * float(oh[k, 0]), 2.0 * float(oh[k, 1]), 2.0 * float(oh[k, 2])))
                     for k in range(NUM_OBSTACLES)]
        has_turret = bool(base.episode_type[i].item() != EP_NO_TURRET)
        # goal B (env-local). STOW (999,999,-50) for showcase/waypoint -- the viz
        # marker is suppressed there by GOAL_STOW's huge norm (see mcap_viz.write_step).
        goal_xyz = (float(base.goal_pos[i, 0]), float(base.goal_pos[i, 1]),
                    float(base.goal_pos[i, 2]))
        return drone_xyz, drone_quat, turret_xyz, aim, obstacles, has_turret, goal_xyz

    def mcap_open(env_ids):
        if not mcap_on:
            return
        for i in env_ids:
            ep_id[i] += 1
            ep_step[i] = 0
            trails[i] = []
            path = os.path.join(args_cli.mcap_dir,
                                f"{args_cli.mode}_env{i}_ep{ep_id[i]:03d}.mcap")
            writers[i] = LiveEpisodeWriter(path, ep_label=f"{args_cli.mode}_e{ep_id[i]}")

    def mcap_close(i, outcome):
        if writers[i] is None:
            return
        p = writers[i].path
        writers[i].close()
        writers[i] = None
        with open(os.path.splitext(p)[0] + ".outcome", "w") as f:  # easy text selection
            f.write(outcome)

    def begin_episode(env_ids):
        env_ids = list(env_ids)
        if not env_ids:
            return
        if args_cli.mode == "showcase":
            # Head-on approach: spawn the drone at a street intersection with an
            # axis-aligned heading down a long street, turret at the far end of
            # that street. The oblivious forward flight (main loop) then closes
            # head-on with the turret in the POV the whole way -- the geometry the
            # 'WM predicts ahead of detection' showcase NEEDS. (The random
            # Manhattan route + mid-waypoint turret usually enters the kill zone
            # laterally, turret off-camera, so neither signal can rise -- verified
            # on the planner run: even killed approach eps had det_logit negative
            # throughout because the turret was never in the POV.)
            ids = env_ids
            n_ids = len(ids)
            street = torch.tensor(STREET_GRID, device=dev)  # (-12,-6,0,6,12)
            S = len(STREET_GRID)
            flat = torch.randint(0, S * S, (n_ids,), device=dev)
            sx = street[flat % S]
            sy = street[flat // S]
            axis_x = torch.randint(0, 2, (n_ids,), device=dev).bool()  # True=approach along x
            # far street end along the chosen axis (>=12 m away; spawn at -12 -> +12 = 24 m)
            far_x = torch.where(sx < 0, 12.0 * torch.ones_like(sx), -12.0 * torch.ones_like(sx))
            far_y = torch.where(sy < 0, 12.0 * torch.ones_like(sy), -12.0 * torch.ones_like(sy))
            tx = torch.where(axis_x, far_x, sx)
            ty = torch.where(axis_x, sy, far_y)
            yaw = torch.where(
                axis_x,
                torch.where(far_x > sx, torch.zeros_like(sx), math.pi * torch.ones_like(sx)),
                torch.where(far_y > sy, (math.pi / 2) * torch.ones_like(sy),
                            (-math.pi / 2) * torch.ones_like(sy)),
            )
            sz = torch.full((n_ids,), DRONE_SPAWN_Z, device=dev)
            qz = torch.sin(yaw / 2.0)
            qw = torch.cos(yaw / 2.0)
            quat = torch.stack([qw, torch.zeros_like(qz), torch.zeros_like(qz), qz], dim=-1)
            robot = base.scene["robot"]
            origin = base.scene.env_origins[ids]
            pos = origin + torch.stack([sx, sy, sz], dim=-1)
            ids_t = torch.tensor(ids, dtype=torch.int32, device=dev)
            robot.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids=ids_t)
            robot.write_root_velocity_to_sim(torch.zeros(n_ids, 6, device=dev), env_ids=ids_t)
            # turret at the far street end (buffer; uav_world_step syncs the scene
            # pose next step -- same pattern as the wpturret relocation below).
            tz = torch.full((n_ids,), TURRET_BASE_Z, device=dev)
            base.turret_pos[ids] = torch.stack([tx, ty, tz], dim=-1)
            base.episode_type[ids] = EP_WAYPOINT_TURRET  # has_turret=True
            base._next_episode_type[ids] = EP_WAYPOINT_TURRET
        elif args_cli.mode in ("nav", "planner", "detector"):
            # Corner-ambush A->B (see _NAV_TEMPLATES). Spawn at A with the approach
            # heading, goal B, turret T just around a building corner. The oblivious/
            # detector route rounds the corner into T (killed); the WM planner imagines
            # the danger (CEM yaw-turn candidates reveal the turret) and detours to B.
            # Rigid translation jitter (shared by A/B/T so the occlusion holds), seeded
            # by (env_id, nav_ep) -> matched-n: the same env+episode sees the SAME
            # scenario across nav/planner/detector. Un-stowing goal_pos=B reactivates
            # uav_goal_reached (the reached_B outcome) + the MCAP goal marker.
            ids = env_ids
            n_ids = len(ids)
            ax = torch.zeros(n_ids, device=dev); ay = torch.zeros(n_ids, device=dev)
            bx = torch.zeros(n_ids, device=dev); by = torch.zeros(n_ids, device=dev)
            tx = torch.zeros(n_ids, device=dev); ty = torch.zeros(n_ids, device=dev)
            yaw = torch.zeros(n_ids, device=dev)
            for jj, i in enumerate(ids):
                (a_x, a_y), (b_x, b_y), (t_x, t_y), y = _NAV_TEMPLATES[nav_ep[i] % 4]
                rng = random.Random((int(i) * 1000003) ^ (nav_ep[i] * 1009) ^ 0x5EED)
                dx = (rng.random() - 0.5) * 0.8   # +-0.4 m rigid shift (shared by A/B/T)
                dy = (rng.random() - 0.5) * 0.8
                ax[jj] = a_x + dx; ay[jj] = a_y + dy
                bx[jj] = b_x + dx; by[jj] = b_y + dy
                tx[jj] = t_x + dx; ty[jj] = t_y + dy
                yaw[jj] = y
                nav_ep[i] += 1
            sz = torch.full((n_ids,), DRONE_SPAWN_Z, device=dev)
            qz = torch.sin(yaw / 2.0); qw = torch.cos(yaw / 2.0)
            quat = torch.stack([qw, torch.zeros_like(qz), torch.zeros_like(qz), qz], dim=-1)
            robot = base.scene["robot"]
            origin = base.scene.env_origins[ids]
            pos = origin + torch.stack([ax, ay, sz], dim=-1)
            ids_t = torch.tensor(ids, dtype=torch.int32, device=dev)
            robot.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids=ids_t)
            robot.write_root_velocity_to_sim(torch.zeros(n_ids, 6, device=dev), env_ids=ids_t)
            tz = torch.full((n_ids,), TURRET_BASE_Z, device=dev)
            base.turret_pos[ids] = torch.stack([tx, ty, tz], dim=-1)
            base.goal_pos[ids] = torch.stack([bx, by, sz], dim=-1)   # un-stow -> uav_goal_reached
            base.episode_type[ids] = EP_WAYPOINT_TURRET
            base._next_episode_type[ids] = EP_WAYPOINT_TURRET
        elif args_cli.force_type == "wpturret":
            base.episode_type[env_ids] = EP_WAYPOINT_TURRET
            base._next_episode_type[env_ids] = EP_WAYPOINT_TURRET
            spawn = (base.scene["robot"].data.root_pos_w - base.scene.env_origins)[env_ids]
            wp.reset_envs(env_ids, spawn)        # fresh route + per-ep speed
            wp._active[env_ids] = True
            # relocate turret onto the mid-route waypoint (the env's reset placed it
            # at a random intersection; move it onto the route so fixed flight crosses
            # its FOV). xy only; keep the reset ground z. Mirrors collect_uav_3d.
            mid = torch.clamp(wp._wp_len[env_ids] // 2, min=1)
            mid_wp = wp._waypoints[env_ids, mid]
            base.turret_pos[env_ids, 0] = mid_wp[:, 0]
            base.turret_pos[env_ids, 1] = mid_wp[:, 1]
        mcap_open(env_ids)

    obs, _ = env.reset()
    state = obs["state"]
    assert state.shape[-1] == 21
    begin_episode(list(range(n)))
    # Refresh obs/state after begin_episode's write_root_pose_to_sim: env.reset()'s
    # observation still holds the DEFAULT spawn (a random street intersection), so
    # without this the first logged step's state + scene pose are stale (off-template).
    # A no-op step (zero action = hold hover) propagates begin_episode's nav-template
    # pose into obs + the robot's cached root_pos_w before step 0 is recorded. The
    # reset_ids path (line ~431) needs the same cure -- handled by the step that
    # triggered the reset already having refreshed obs for the NEXT episode's step 0.
    obs, _, _, _, _ = env.step(torch.zeros(n, 4, dtype=torch.float32, device=dev))
    state = obs["state"]

    just_reset = np.ones(n, dtype=bool)
    outcomes = Counter()
    done = [0]
    target = args_cli.episodes

    def plan_action(pix, jr):
        pix_hwc = np.stack([pov_hwc(pix, i) for i in range(n)])  # (N,H,W,3) uint8
        # Thread state/goal/obstacles for the NavServer (nav/planner/detector). goal =
        # base.goal_pos = B (set by begin_episode; STOW for showcase, which the phantom
        # server ignores). obstacles = [centers, half] env-local, matching the NavPlanner
        # building term + the drone's env-local state[0:3].
        state_np = state.detach().cpu().numpy().astype(np.float32)             # (N,21)
        goal_np = base.goal_pos.detach().cpu().numpy().astype(np.float32)      # (N,3)
        obs_np = torch.cat([base.obstacle_centers, base.obstacle_half], dim=-1). \
            detach().cpu().numpy().astype(np.float32)                         # (N,16,6)
        send_msg(sock, {"just_reset": jr.astype(bool), "pix": pix_hwc,
                        "state": state_np, "goal": goal_np, "obstacles": obs_np})
        # nav/planner/detector: {action, wm_danger, det_logit, flee}
        # showcase (phantom server): {wm_danger, det_logit}  (no action)
        return recv_msg(sock)

    max_steps = target * 400 + 800
    t0 = time.time()
    for gstep in range(max_steps):
        if done[0] >= target:
            break
        pix = obs["policy"]
        wm_d = det_l = None  # nav/planner/showcase: both signals; waypoint: none
        if args_cli.mode in ("nav", "planner", "detector"):
            # Unified A->B client: the NavServer returns the action (already flee-
            # overridden in detector mode) + the two signals. nav/planner/detector
            # share IDENTICAL client logic; only the host-side knob differs
            # (danger_weight / reactive_flee), so kill/reach differences attribute
            # cleanly to imagination vs reaction.
            resp = plan_action(pix, just_reset)
            action = torch.tensor(resp["action"], dtype=torch.float32, device=dev)
            wm_d = resp["wm_danger"]; det_l = resp["det_logit"]
        elif args_cli.mode == "showcase":
            # Phantom server: returns ONLY the two signals (no action) -- the drone
            # flies oblivious forward (head-on at the turret) so the danger the WM
            # imagines actually materializes. cruise must match PhantomServer.cruise.
            resp = plan_action(pix, just_reset)
            action = torch.zeros(n, 4, dtype=torch.float32, device=dev)
            action[:, 0] = SHOWCASE_CRUISE
            wm_d = resp["wm_danger"]; det_l = resp["det_logit"]
        else:
            action = wp.get_action(state, base.obstacle_centers, base.obstacle_half)

        state_cpu = state.detach().cpu()
        # MCAP showcase log (PRE-step frame = what the planner/detector saw this step).
        if mcap_on:
            for i in range(n):
                if writers[i] is None:
                    continue
                gx = env_geometry(i)
                trails[i].append(gx[0])
                writers[i].write_step(
                    ep_step[i], pov_hwc(pix, i), gx[0], gx[1], gx[2], gx[3], gx[4],
                    state_cpu[i], has_turret=gx[5], trail=trails[i],
                    goal_xyz=gx[6],
                    wm_danger=wm_d[i] if wm_d is not None else None,
                    det_logit=det_l[i] if det_l is not None else None,
                )
                ep_step[i] += 1
        obs, rew, term, trunc, info = env.step(action)
        term_cpu = term.detach().cpu()
        trunc_cpu = trunc.detach().cpu()

        reset_mask = term | trunc
        reset_ids = reset_mask.nonzero(as_tuple=False).squeeze(-1).tolist()
        for i in reset_ids:
            outcome = infer_outcome(state_cpu[i], bool(term_cpu[i]), bool(trunc_cpu[i]))
            outcomes[outcome] += 1
            mcap_close(i, outcome)
            done[0] += 1
            if done[0] >= target:
                break
        if reset_ids:
            begin_episode([i for i in reset_ids])
        state = obs["state"]
        jr_next = np.zeros(n, dtype=bool)
        jr_next[reset_ids] = True
        just_reset = jr_next

        if gstep % 25 == 0 or done[0] >= target:
            print(f"[LIVE] gstep={gstep:5d} done={done[0]}/{target} "
                  f"outcomes={dict(outcomes)} elapsed={time.time()-t0:.0f}s", flush=True)

    if mcap_on:  # close any writers still open when the run ended mid-episode
        for i in range(n):
            mcap_close(i, "open")

    total = sum(outcomes.values())
    killed = outcomes.get("killed", 0)
    crash = outcomes.get("crash", 0)
    timeout = outcomes.get("timeout", 0)
    reached = outcomes.get("reached_B", 0)
    ended = outcomes.get("ended", 0)
    surv = timeout + reached + ended   # not killed, not crashed
    print(f"[LIVE] ============ RESULT mode={args_cli.mode} "
          f"force_type={args_cli.force_type} ============", flush=True)
    print(f"[LIVE] total={total} killed={killed} crash={crash} timeout={timeout} "
          f"reached_B={reached} ended={ended} | kill_rate={killed/max(total,1):.3f} "
          f"reach_rate={reached/max(total,1):.3f} survival={surv/max(total,1):.3f}", flush=True)
    print(f"[LIVE] outcomes={dict(outcomes)}", flush=True)

    if sock:
        sock.close()
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()

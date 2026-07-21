"""Collect UAVTurret3D episodes for LeWM training -- Stage 1 (container rollout).

Two-stage collection decouples the Isaac container (has Isaac + numpy + PIL, NOT
swm/lancedb/pyarrow) from the host training venv (has swm, NOT Isaac):

  Stage 1  THIS script, isaac-lab container: roll out the drunken-explorer policy,
           record aligned PRE-step frames per env, and on each episode end pickle
           one file per episode to the bind-mounted output dir. Pixels are JPEG-
           encoded in-container (PIL) to keep the intermediate small (~5 GB for
           2000 eps vs ~45 GB raw). Stage 2 decodes + writes lance.
  Stage 2  lance_from_episodes.py, host venv: read the pickles -> swm.LanceWriter
           -> uav_isaac_train.lance.

Per-episode pickle = dict of equal-length lists (one entry per frame):
  pixels     list[bytes]          JPEG 224x224 RGB (Stage 2 decodes -> HWC uint8)
  action     list[np.float32(4)]  body-frame (vx_fwd,vy_strafe,vz,yaw_rate) in [-1,1]
  state      list[np.float32(21)] env state (danger@13, drone_yaw@20)
  shot       list[int]            0/1 thrust-override mask (Modification 2: LeWM masks)
  danger     list[int]            0/1 (= state[13]; convenience for density counting)
  drone_pos  list[np.float32(3)]  env-local
  turret_pos list[np.float32(3)]  env-local
  barrel     list[np.float32(2)]  [turret_yaw, turret_pitch]
  outcome    str (episode-level)  'killed'|'timeout'|'crash'|'ended'

PRE-step recording (NO action-shift): frame t records (state_t, pixels_t, action_t)
where pixels_t/state_t are the CURRENT obs and action_t=policy(state_t); env.step
then yields state_{t+1}. So (state_t, action_t) is aligned to transition t->t+1 and
the reader's frameskip=5 windows are consistent. (2D World.collect left-rotates
action to compensate for POST-step recording; that does NOT apply here.)

Mid-city spawn (env P_INTERIOR_SPAWN) + drunken-explorer provoke manufacture the
5-10% danger density (Modification 1). The 50-ep validation run is the density gate.

Run in the isaac-lab container (mirrors run_collect_uav3d.sh mounts):
    /workspace/isaaclab/isaaclab.sh -p /workspace/scripts/collect_uav_3d.py \
        --headless --enable_cameras --num_envs 16 --episodes 50 \
        --output_dir /workspace/output/uav3d_episodes
"""
import argparse
import io
import pickle
import time
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="UAVTurret3D training collector (Stage 1 -> episode pickles).")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--episodes", type=int, default=50)
parser.add_argument("--output_dir", type=str, default="/workspace/output/uav3d_episodes")
parser.add_argument("--q", type=float, default=0.3,
                    help="policy WANDER fraction. q=0.3 -> 30%% wander / 70%% provoke (max encounters). "
                         "NOTE: policies_3d mislabels q 'provoke' but it is the WANDER fraction.")
parser.add_argument("--r", type=float, default=0.3, help="policy commit (vs evade) fraction in danger.")
parser.add_argument("--noise", type=float, default=0.0)
parser.add_argument("--jpeg_quality", type=int, default=95)
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

import uav_wm.envs  # noqa: F401,E402  (registers Isaac-UAVTurret3D-v0)
from uav_wm.data.policies_3d import ExplorationPolicy3D, WaypointPolicy  # noqa: E402
from uav_wm.envs.uav_turret_3d import (  # noqa: E402
    CRASH_Z,
    EP_DRUNK,
    EP_NO_TURRET,
    EP_WAYPOINT_TURRET,
    roll_episode_types,
)
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

TASK = "Isaac-UAVTurret3D-v0"


def pov_hwc(pix_obs, i):
    """env i's POV as (H, W, 3) uint8 numpy (JPEG-encoding input). Works on CPU or GPU."""
    fr = pix_obs[i]
    if fr.dim() == 3 and fr.shape[0] <= 4:  # (C, H, W)
        if fr.shape[0] == 4:
            fr = fr[:3]
        fr = fr.permute(1, 2, 0)
    if fr.dtype != torch.uint8:
        fr = (fr.clamp(0.0, 1.0) * 255.0).to(torch.uint8)
    return fr.detach().cpu().numpy().astype("uint8")[:, :, :3]


def jpeg_bytes(hwc_uint8, quality):
    buf = io.BytesIO()
    Image.fromarray(hwc_uint8).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def infer_outcome(st, term, trunc):
    """Episode outcome from the last recorded (pre-step) state + term/trunc flags.

    No 'reached_goal' outcome: the goal is stowed (Evan: "remove the target
    location") and goal_rel is zeroed in the state, so there is no goal to reach."""
    drone_z = float(st[2])
    fire = float(st[17])
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
    print(f"[COLLECT] env={type(base).__name__} num_envs={n} device={dev} "
          f"q={args_cli.q} r={args_cli.r}", flush=True)

    torch.manual_seed(args_cli.seed)  # reproducible spawns + episode-type rolls
    drunk = ExplorationPolicy3D(q=args_cli.q, r=args_cli.r, noise=args_cli.noise,
                                seed=args_cli.seed, device=str(dev))
    wp = WaypointPolicy(seed=args_cli.seed + 1, device=str(dev))
    drunk._ensure_buffers(n)   # so reset_envs is non-no-op from episode 0
    wp._ensure_buffers(n)

    obs, _ = env.reset()
    state = obs["state"]
    assert state.shape[-1] == 21

    def begin_episode(env_ids):
        """Set up policies for envs whose episode just (re)started, from the
        episode_type the env committed during reset; then roll _next_episode_type
        for the FOLLOWING episode (one-ahead so the next reset places the right
        turret). Waypoint envs (no-turret + waypoint+turret) get a fresh route;
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
        wp._active[drunk_ids] = False      # drunk envs don't follow waypoints
        # relocate the waypoint+turret turret onto a mid-route waypoint (the env
        # placed it at a random intersection as a frame-0 fallback; move it onto the
        # actual route so the oblivious fixed flight flies into its FOV -> the
        # danger/shot/fall transition the WM must learn). xy only; keep the reset
        # ground z. uav_world_step re-writes the turret scene pose next step.
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

    begin_episode(list(range(n)))   # ep 0: env.reset() already committed types

    bufs = [None] * n          # per-env episode dict-of-lists, or None
    ep_id = [0]
    episodes_done = [0]
    manifest = []

    def start_ep(i):
        bufs[i] = {"pixels": [], "action": [], "state": [], "shot": [], "danger": [],
                   "drone_pos": [], "turret_pos": [], "barrel": [],
                   "episode_type": int(base.episode_type[i].item())}

    for i in range(min(n, args_cli.episodes)):
        start_ep(i)

    max_steps = args_cli.episodes * 350 + 600
    t0 = time.time()
    for gstep in range(max_steps):
        if gstep % 50 == 0:
            nd = episodes_done[0]
            print(f"[COLLECT] gstep={gstep:5d} done={nd}/{args_cli.episodes} "
                  f"active={sum(1 for b in bufs if b is not None)} "
                  f"elapsed={time.time()-t0:.0f}s", flush=True)

        pix = obs["policy"]            # (N, C, H, W)
        act_drunk = drunk.get_action(state, base.obstacle_centers, base.obstacle_half)
        act_wp = wp.get_action(state, base.obstacle_centers, base.obstacle_half)
        mode_wp = base.episode_type != EP_DRUNK            # True -> follow waypoints
        action = torch.where(mode_wp.unsqueeze(-1), act_wp, act_drunk)  # (N, 4)

        # Vectorized CPU pull (one transfer per field, NOT per env) -- avoids N GPU
        # syncs per step. Frame t = current obs (pre-step); action_t = policy(state_t).
        pix_cpu = pix.detach().cpu()
        state_cpu = state.detach().cpu()
        action_cpu = action.detach().cpu().to(torch.float32)
        shot_cpu = base._shot.detach().cpu()
        danger_cpu = base.danger.detach().cpu()
        drone_pos_cpu = (base.scene["robot"].data.root_pos_w - base.scene.env_origins).detach().cpu()
        turret_pos_cpu = base.turret_pos.detach().cpu()
        barrel_cpu = torch.stack([base.turret_yaw, base.turret_pitch], dim=-1).detach().cpu()

        for i in range(n):
            b = bufs[i]
            if b is None:
                continue
            b["pixels"].append(jpeg_bytes(pov_hwc(pix_cpu, i), args_cli.jpeg_quality))
            b["action"].append(action_cpu[i].numpy().astype("float32"))
            b["state"].append(state_cpu[i].numpy().astype("float32"))
            b["shot"].append(int(shot_cpu[i].item()))
            b["danger"].append(int(danger_cpu[i].item()))
            b["drone_pos"].append(drone_pos_cpu[i].numpy().astype("float32"))
            b["turret_pos"].append(turret_pos_cpu[i].numpy().astype("float32"))
            b["barrel"].append(barrel_cpu[i].numpy().astype("float32"))

        obs, rew, term, trunc, info = env.step(action)
        new_state = obs["state"]

        for i in range(n):
            b = bufs[i]
            if b is None:
                continue
            t = bool(term[i].item()); tr = bool(trunc[i].item())
            if t or tr:
                outcome = infer_outcome(state_cpu[i], t, tr)
                nframes = len(b["pixels"])
                ndang = int(sum(b["danger"]))
                eid = ep_id[0]; ep_id[0] += 1
                b["outcome"] = outcome
                with open(out / f"episode_{eid:05d}.pkl", "wb") as f:
                    pickle.dump(b, f, protocol=pickle.HIGHEST_PROTOCOL)
                manifest.append((f"episode_{eid:05d}.pkl", nframes, outcome, ndang, b["episode_type"]))
                episodes_done[0] += 1
                bufs[i] = None
                if episodes_done[0] < args_cli.episodes:
                    start_ep(i)

        reset_mask = term | trunc
        if reset_mask.any():
            begin_episode(reset_mask.nonzero(as_tuple=False).squeeze(-1).tolist())
        state = new_state
        if episodes_done[0] >= args_cli.episodes and all(b is None for b in bufs):
            break

    # flush any still-open episodes (shouldn't happen -- bounded loop ended early)
    for i in range(n):
        b = bufs[i]
        if b is not None:
            nframes = len(b["pixels"]); ndang = int(sum(b["danger"]))
            eid = ep_id[0]; ep_id[0] += 1
            b["outcome"] = "open"
            with open(out / f"episode_{eid:05d}.pkl", "wb") as f:
                pickle.dump(b, f, protocol=pickle.HIGHEST_PROTOCOL)
            manifest.append((f"episode_{eid:05d}.pkl", nframes, "open", ndang, b["episode_type"]))

    env.close()

    tot = 0; tot_d = 0
    outcomes = {}
    type_names = {EP_NO_TURRET: "no_turret", EP_WAYPOINT_TURRET: "wpturret", EP_DRUNK: "drunk"}
    type_frames = {EP_NO_TURRET: [0, 0], EP_WAYPOINT_TURRET: [0, 0], EP_DRUNK: [0, 0]}  # [frames, danger]
    man_path = out / "manifest.txt"
    with open(man_path, "w") as f:
        f.write(f"# UAVTurret3D training collect (Stage 1) -- {len(manifest)} episodes\n")
        f.write(f"# num_envs={n} q={args_cli.q} r={args_cli.r} noise={args_cli.noise} seed={args_cli.seed}\n")
        f.write("# episode                steps outcome       danger_frames type\n")
        for name, steps, outcome, df, et in manifest:
            f.write(f"{name:24s} {steps:5d} {outcome:13s} {df:4d} {type_names.get(et, '?')}\n")
            tot += steps; tot_d += df
            outcomes[outcome] = outcomes.get(outcome, 0) + 1
            if et in type_frames:
                type_frames[et][0] += steps
                type_frames[et][1] += df
        dens = (100.0 * tot_d / tot) if tot else 0.0
        f.write(f"# total frames: {tot}  danger frames: {tot_d}  danger density: {dens:.2f}%\n")
        for et in (EP_NO_TURRET, EP_WAYPOINT_TURRET, EP_DRUNK):
            fr, dng = type_frames[et]
            d = (100.0 * dng / fr) if fr else 0.0
            f.write(f"# {type_names[et]:9s} frames: {fr}  danger: {dng}  density: {d:.2f}%\n")
    print(f"[COLLECT] DONE episodes={len(manifest)} outcomes={outcomes} "
          f"frames={tot} danger={tot_d} density={dens:.2f}%")
    print(f"[COLLECT] manifest -> {man_path}")
    print(f"[COLLECT] pickles -> {out}")
    print("[COLLECT] UAVTURRET3D TRAIN COLLECT OK")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        print("[COLLECT] CRASHED (see traceback above)", flush=True)
    finally:
        simulation_app.close()

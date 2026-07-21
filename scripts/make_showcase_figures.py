#!/usr/bin/env python3
"""Generate showcase visuals for the README from the live-demo MCAPs + .signals.txt
sidecars. Text-only safe: it WRITES png/gif assets but never inspects them -- the
caller verifies outputs by metadata (size, frame count) + numeric stats only.

Outputs (under docs/assets/):
  phantom_signals.png       static /signals trace, best-lead phantom episode
                            (wm_danger vs det_logit vs danger, rise steps annotated)
  phantom_lead.gif          drone POV (top) + growing /signals trace (bottom),
                            best-lead phantom episode -- the "WM predicts ahead" replay
  planner_vs_waypoint.gif   top-down x-y trails, planner(timeout,evade) vs
                            waypoint(killed,fly-into-turret), step-synced
  nav_a_to_b.gif            top-down x-y, 3 step-synced panels for the head-on
                            A->B task: oblivious(flies into kill zone, killed) vs WM
                            planner (imagines + holds back, survives) vs detector(fires late);
                            A/B/turret/buildings marked + outcome annotated

MCAP schema (from scripts/live/mcap_viz.py):
  /drone/pov/image  RawImage     rgb8, 224x224, data=H*W*3 bytes
  /scene            SceneUpdate  entities[0].cubes=[drone,turret,obstacles...],
                                 .lines=[aim, fov, fov, trail]
  /signals          PoseInFrame  pose.position.{x=wm_danger, y=det_logit, z=danger}
  /drone/tf         FrameTransform  translation = drone world xyz
  /state            Log          (unused here; .signals.txt sidecar is easier)

Deps (the showcase venv): mcap, mcap-protobuf-support,
  foxglove-schemas-protobuf==0.3.0, numpy, pillow, matplotlib

Usage:
  python scripts/make_showcase_figures.py
  python scripts/make_showcase_figures.py --phantom data/showcase_phantom/showcase_env5_ep001.mcap
"""
import os
import sys
import glob
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

from mcap.reader import make_reader
from foxglove_schemas_protobuf.RawImage_pb2 import RawImage
from foxglove_schemas_protobuf.SceneUpdate_pb2 import SceneUpdate
from foxglove_schemas_protobuf.PoseInFrame_pb2 import PoseInFrame
from foxglove_schemas_protobuf.FrameTransform_pb2 import FrameTransform

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(REPO, "docs", "assets")
DET_THRESHOLD = 1.53  # detector's calibrated best-F1 fire threshold (ref line)
STEP_DT = 0.1         # env step dt -> seconds


# ----------------------------- MCAP reading ----------------------------------
def read_mcap(path):
    """Return dict topic -> list of (log_time_ns, proto_msg), sorted by time."""
    out = {"/drone/pov/image": [], "/scene": [], "/signals": [], "/drone/tf": []}
    cls = {"/drone/pov/image": RawImage, "/scene": SceneUpdate,
           "/signals": PoseInFrame, "/drone/tf": FrameTransform}
    with open(path, "rb") as f:
        r = make_reader(f)
        for schema, channel, msg in r.iter_messages():
            t = channel.topic
            if t not in out:
                continue
            m = cls[t]()
            m.ParseFromString(msg.data)
            out[t].append((msg.log_time, m))
    for k in out:
        out[k].sort(key=lambda p: p[0])
    return out


def pov_frames(mcap):
    """List of (H,W,3) uint8 arrays from /drone/pov/image."""
    frames = []
    for _, m in mcap["/drone/pov/image"]:
        arr = np.frombuffer(m.data, dtype=np.uint8)
        c = arr.size // (m.height * m.width) or 3
        frames.append(arr.reshape(m.height, m.width, c)[:, :, :3].copy())
    return frames


def signals(mcap):
    """Arrays (wm, det, danger) from /signals PoseInFrame."""
    wm, det, dang = [], [], []
    for _, m in mcap["/signals"]:
        wm.append(m.pose.position.x)
        det.append(m.pose.position.y)
        dang.append(int(round(m.pose.position.z)))
    return np.array(wm), np.array(det), np.array(dang)


def scene_geometry(mcap):
    """Static-ish geometry from the first /scene message:
    turret (x,y) or None, obstacles [(cx,cy,dx,dy,col)...], drone (x,y) at step0,
    goal (x,y) or None (nav-family B marker; None for showcase/waypoint where the
    goal is STOW and suppressed). Cubes are classified by COLOR (not index) so the
    green goal cube isn't mistaken for a building: drone=cube[0], turret=red,
    goal=green, obstacles=grey."""
    if not mcap["/scene"]:
        return None, [], None, None
    _, s = mcap["/scene"][0]
    ent = s.entities[0]
    turret = None
    obstacles = []
    drone = None
    goal = None
    for i, c in enumerate(ent.cubes):
        x, y = c.pose.position.x, c.pose.position.y
        dx, dy = c.size.x, c.size.y
        col = (c.color.r, c.color.g, c.color.b, c.color.a)
        if i == 0:
            drone = (x, y)
            continue
        r, g, b = col[0], col[1], col[2]
        if r > 0.5 and g < 0.5 and b < 0.5:        # red = turret
            turret = (x, y)
        elif r < 0.3 and g > 0.5 and b < 0.5:      # green = goal B
            goal = (x, y)
        else:                                       # grey = building
            obstacles.append((x, y, dx, dy, col))
    return turret, obstacles, drone, goal


def drone_trail(mcap):
    """List of (x,y) from /drone/tf, one per step."""
    return [(m.translation.x, m.translation.y) for _, m in mcap["/drone/tf"]]


def read_sidecar(mcap_path):
    """Fallback signals from the .signals.txt sidecar (if /signals missing)."""
    s = mcap_path[:-5] + ".signals.txt"
    if not os.path.exists(s):
        return None
    wm, det, dang = [], [], []
    for ln in open(s):
        if ln.startswith("#") or not ln.strip():
            continue
        p = ln.split()
        wm.append(float(p[1])); det.append(float(p[2])); dang.append(int(p[3]))
    return np.array(wm), np.array(det), np.array(dang)


# ----------------------------- helpers ---------------------------------------
def rise_step(v, frac=0.30, min_amp=0.20):
    base = float(np.median(v[:5])) if len(v) >= 5 else float(np.median(v))
    peak = float(np.max(v))
    if peak - base < min_amp:
        return None
    thr = base + frac * (peak - base)
    for i, val in enumerate(v):
        if val > thr:
            return i
    return None


def _font(size):
    try:
        return ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", size)
    except Exception:
        return ImageFont.load_default()


# ----------------------------- asset 1: phantom_signals.png -------------------
def make_phantom_signals(mcap_path, out):
    mcap = read_mcap(mcap_path)
    sig = signals(mcap)
    if sig is None or len(sig[0]) == 0:
        sig = read_sidecar(mcap_path)
    wm, det, dang = sig
    t = np.arange(len(wm)) * STEP_DT
    wm_r = rise_step(wm); det_r = rise_step(det)
    dstep = int(np.argmax(dang)) if dang.max() > 0 else None
    lead = (det_r - wm_r) if (wm_r is not None and det_r is not None) else None

    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=130)
    # danger shading
    ax.axhline(0, color="0.75", lw=0.8)
    ax.axhline(DET_THRESHOLD, color="C1", ls="--", lw=1.0, alpha=0.7,
               label=f"detector fire thr = {DET_THRESHOLD}")
    if dstep is not None:
        ax.axvspan(dstep * STEP_DT, (dstep + 4) * STEP_DT, color="red", alpha=0.10,
                   label="danger=1 (threat real)")
    ax.plot(t, wm, color="C0", lw=1.6, label="wm_danger  (world model imagination)")
    ax.plot(t, det, color="C1", lw=1.6, label="det_logit  (single-frame detector)")
    ax.plot(t, dang * (max(wm.max(), det.max()) * 0.9), color="red", lw=1.2, alpha=0.5,
            label="danger (gt, scaled)")
    # rise markers
    if wm_r is not None:
        ax.axvline(wm_r * STEP_DT, color="C0", ls=":", lw=1.0)
        ax.annotate("WM rises", xy=(wm_r * STEP_DT, wm[wm_r]),
                    xytext=(wm_r * STEP_DT + 1.0, wm[wm_r] + 0.4), color="C0",
                    arrowprops=dict(arrowstyle="->", color="C0", lw=0.8))
    if det_r is not None:
        ax.axvline(det_r * STEP_DT, color="C1", ls=":", lw=1.0)
        ax.annotate("detector rises", xy=(det_r * STEP_DT, det[det_r]),
                    xytext=(det_r * STEP_DT + 0.4, det[det_r] - 0.7), color="C1",
                    arrowprops=dict(arrowstyle="->", color="C1", lw=0.8))
    ax.set_xlabel("time (s)  [1 step = 0.1 s]")
    ax.set_ylabel("danger logit (per head; not cross-comparable)")
    stem = os.path.basename(mcap_path)[:-5]
    lead_s = f"{lead*STEP_DT:.1f}s" if lead is not None else "n/a"
    ax.set_title(f"{stem}   |   WM predicts ahead by {lead} steps ≈ {lead_s}   "
                 f"(wm_rise={wm_r} det_rise={det_r})")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return {"steps": len(wm), "wm_rise": wm_r, "det_rise": det_r,
            "lead_steps": lead, "danger_step": dstep}


# ----------------------------- asset 2: phantom_lead.gif ----------------------
def make_phantom_gif(mcap_path, out, stride=2, frame_ms=120):
    mcap = read_mcap(mcap_path)
    pov = pov_frames(mcap)
    sig = signals(mcap)
    if sig is None or len(sig[0]) == 0:
        sig = read_sidecar(mcap_path)
    wm, det, dang = sig
    n = min(len(pov), len(wm))
    wm, det, dang = wm[:n], det[:n], dang[:n]
    pov = pov[:n]

    W = 460
    pov_w = W
    pov_h = W  # square
    plot_h = 200
    font = _font(18); small = _font(14)
    t_all = np.arange(n) * STEP_DT
    ylim = (min(wm.min(), det.min(), -2.0) - 0.3, max(wm.max(), det.max(), 2.0) + 0.3)

    frames = []
    idxs = list(range(0, n, stride))
    for k, i in enumerate(idxs):
        # --- top: POV with live readout ---
        img = Image.fromarray(pov[i]).resize((pov_w, pov_h), Image.NEAREST)
        d = ImageDraw.Draw(img)
        # header bar
        d.rectangle([0, 0, pov_w, 30], fill=(0, 0, 0, 180))
        d.text((8, 5), f"drone POV  step {i}/{n-1}  t={i*STEP_DT:.1f}s",
               fill="white", font=small)
        # live signal readout (bottom bar)
        d.rectangle([0, pov_h - 34, pov_w, pov_h], fill=(0, 0, 0, 180))
        col_w = "deepskyblue" if wm[i] > 0.5 else "white"
        col_d = "orange" if det[i] > 0 else "white"
        d.text((8, pov_h - 30),
               f"wm_danger={wm[i]:+.2f}   det_logit={det[i]:+.2f}   danger={dang[i]}",
               fill="white", font=small)
        if dang[i]:
            d.rectangle([0, pov_h - 34, pov_w, pov_h], fill=(180, 0, 0, 200))

        # --- bottom: signals-so-far ---
        fig, ax = plt.subplots(figsize=(W / 100, plot_h / 100), dpi=100)
        ax.axhline(0, color="0.75", lw=0.6)
        ax.axhline(DET_THRESHOLD, color="C1", ls="--", lw=0.8)
        ax.plot(t_all[:i + 1], wm[:i + 1], color="C0", lw=1.4)
        ax.plot(t_all[:i + 1], det[:i + 1], color="C1", lw=1.4)
        ax.plot(t_all[:i + 1], dang[:i + 1] * (ylim[1] * 0.85), color="red", lw=1.0, alpha=0.5)
        ax.scatter([t_all[i]], [wm[i]], color="C0", s=18, zorder=5)
        ax.scatter([t_all[i]], [det[i]], color="C1", s=18, zorder=5)
        ax.set_xlim(0, t_all[-1])
        ax.set_ylim(*ylim)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.tick_params(left=False, bottom=False)
        for sp in ax.spines.values():
            sp.set_visible(False)
        fig.canvas.draw()
        plot = Image.frombytes("RGBA", fig.canvas.get_width_height(),
                               fig.canvas.buffer_rgba()).convert("RGB")
        plt.close(fig)

        composite = Image.new("RGB", (W, pov_h + plot_h), "white")
        composite.paste(img, (0, 0))
        composite.paste(plot, (0, pov_h))
        frames.append(composite)

    frames[0].save(out, save_all=True, append_images=frames[1:],
                   duration=frame_ms, loop=0, optimize=True, disposal=2)
    return {"frames": len(frames), "size": (W, pov_h + plot_h), "n_steps": n}


# ----------------------------- asset 3: planner_vs_waypoint.gif ---------------
def _topdown_frame(ax, trail, turret, obstacles, upto, outcome, title,
                   goal=None, spawn=None):
    ax.clear()
    if trail:
        xs = [p[0] for p in trail[:upto + 1]]
        ys = [p[1] for p in trail[:upto + 1]]
        ax.plot(ys, xs, color="C0", lw=1.4, alpha=0.8)  # y on x-axis for top-down North-up
        ax.scatter([ys[-1]], [xs[-1]], color="C0", s=30, zorder=5)
    if turret is not None:
        ax.scatter([turret[1]], [turret[0]], color="red", marker="s", s=80, zorder=6)
        ax.add_patch(plt.Circle((turret[1], turret[0]), 8.0, color="red", fill=False,
                                ls="--", lw=1.0, alpha=0.5))  # TURRET_RANGE=8m
    for (cx, cy, dx, dy, col) in obstacles:
        ax.add_patch(plt.Rectangle((cy - dy / 2, cx - dx / 2), dy, dx,
                                   color="0.6", alpha=0.5))
    if spawn is not None:   # A = episode spawn (first trail point)
        ax.scatter([spawn[1]], [spawn[0]], color="0.15", marker="*", s=160, zorder=7)
        ax.text(spawn[1], spawn[0], " A", color="0.15", fontsize=9, fontweight="bold")
    if goal is not None:    # B = goal
        ax.scatter([goal[1]], [goal[0]], color="green", marker="*", s=160, zorder=7)
        ax.text(goal[1], goal[0], " B", color="green", fontsize=9, fontweight="bold")
    ax.set_aspect("equal")
    ax.set_title(f"{title}\n{outcome}  step {upto}", fontsize=9)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.2)


def make_survival_gif(planner_path, waypoint_path, out, stride=3, frame_ms=140):
    pm = read_mcap(planner_path)
    wm_ = read_mcap(waypoint_path)
    p_trail = drone_trail(pm); p_tur, p_obs, _, _ = scene_geometry(pm)
    w_trail = drone_trail(wm_); w_tur, w_obs, _, _ = scene_geometry(wm_)

    def oc(path):
        o = path[:-5] + ".outcome"
        return open(o).read().strip() if os.path.exists(o) else "?"
    p_out = f"PLANNER: {oc(planner_path)}"
    w_out = f"WAYPOINT (no imagination): {oc(waypoint_path)}"

    # common axis bounds across both episodes for a fair side-by-side
    allx = [p[0] for p in p_trail] + [p[0] for p in w_trail] + \
           ([p_tur[0]] if p_tur else []) + ([w_tur[0]] if w_tur else [])
    ally = [p[1] for p in p_trail] + [p[1] for p in w_trail] + \
           ([p_tur[1]] if p_tur else []) + ([w_tur[1]] if w_tur else [])
    pad = 6.0
    xmin, xmax = min(allx) - pad, max(allx) + pad
    ymin, ymax = min(ally) - pad, max(ally) + pad

    n = max(len(p_trail), len(w_trail))
    idxs = list(range(0, n, stride))
    frames = []
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8, 4.2), dpi=110)
    for i in idxs:
        _topdown_frame(a1, p_trail, p_tur, p_obs, min(i, len(p_trail) - 1), p_out, "planner (imagines + evades)")
        _topdown_frame(a2, w_trail, w_tur, w_obs, min(i, len(w_trail) - 1), w_out, "waypoint (oblivious)")
        for a in (a1, a2):
            a.set_xlim(ymin, ymax); a.set_ylim(xmin, xmax)
        fig.suptitle("Acting on imagination saves you — same scenario, matched n=24",
                     fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.canvas.draw()
        im = Image.frombytes("RGBA", fig.canvas.get_width_height(),
                             fig.canvas.buffer_rgba()).convert("RGB")
        frames.append(im)
    plt.close(fig)
    frames[0].save(out, save_all=True, append_images=frames[1:],
                   duration=frame_ms, loop=0, optimize=True, disposal=2)
    return {"frames": len(frames), "planner_len": len(p_trail),
            "waypoint_len": len(w_trail),
            "size": frames[0].size}


# ----------------------------- asset 4: nav_a_to_b.gif ------------------------
def make_nav_ab_gif(planner_path, oblivious_path, detector_path, out,
                    stride=3, frame_ms=140):
    """Three step-synced top-down panels for the head-on A->B scenario
    (matched n: same seed -> same A/B/T across the three MCAPs):
      left  = oblivious A->B (no imagination)  -- flies straight into T's kill zone, killed
      mid   = WM planner A->B                  -- imagines the turret ahead, holds back at ~9 m, survives
      right = detector-reactive A->B           -- fires only when T is in-frame, too late, killed
    A (spawn), B (goal), turret (+8 m range ring), and buildings are marked; the
    .outcome sidecar annotates each panel. Dead drones freeze at their last position
    (trail clamped), so the GIF shows oblivious/detector dying in the kill zone while
    the planner holds back and survives."""
    om = read_mcap(oblivious_path)
    pm = read_mcap(planner_path)
    dm = read_mcap(detector_path)
    o_trail = drone_trail(om); o_tur, o_obs, _, o_goal = scene_geometry(om)
    p_trail = drone_trail(pm); p_tur, p_obs, _, p_goal = scene_geometry(pm)
    d_trail = drone_trail(dm); d_tur, d_obs, _, d_goal = scene_geometry(dm)
    o_spawn = o_trail[0] if o_trail else None
    p_spawn = p_trail[0] if p_trail else None
    d_spawn = d_trail[0] if d_trail else None

    def oc(path):
        o = path[:-5] + ".outcome"
        return open(o).read().strip() if os.path.exists(o) else "?"

    def display_outcome(raw):
        # timeout/reached_B/ended/open all mean the drone was still alive when the
        # run ended (open = censored mid-episode) => "survives" for the viewer.
        return {"killed": "killed", "crash": "crashed"}.get(raw, "survives")

    o_lab = f"oblivious · {display_outcome(oc(oblivious_path))}"
    p_lab = f"WM planner · {display_outcome(oc(planner_path))}"
    d_lab = f"detector · {display_outcome(oc(detector_path))}"

    # Bounds must include BUILDINGS (+spawn/goal/turret), not just trails: the
    # head-on street is long but the trails hug the street centerline, so
    # trail-only bounds give an extreme aspect (~0.37) -> equal-aspect panels
    # collapse to tall slivers that tight_layout packs into the figure middle
    # (the "plots move to the middle" bug) and the flanking buildings get clipped.
    trails = [o_trail, p_trail, d_trail]
    turs = [o_tur, p_tur, d_tur]
    allx, ally = [], []
    for tr in trails:
        allx += [p[0] for p in tr]; ally += [p[1] for p in tr]
    for t in turs:
        if t is not None:
            allx.append(t[0]); ally.append(t[1])
    for g in (o_goal, p_goal, d_goal):
        if g is not None:
            allx.append(g[0]); ally.append(g[1])
    for s in (o_spawn, p_spawn, d_spawn):
        if s is not None:
            allx.append(s[0]); ally.append(s[1])
    for ob in (o_obs, p_obs, d_obs):
        for (cx, cy, dx, dy, _c) in ob:
            allx += [cx - dx / 2, cx + dx / 2]
            ally += [cy - dy / 2, cy + dy / 2]
    pad = 2.0
    xmin, xmax = min(allx) - pad, max(allx) + pad
    ymin, ymax = min(ally) - pad, max(ally) + pad

    n = max(len(o_trail), len(p_trail), len(d_trail))
    idxs = list(range(0, n, stride))
    frames = []
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(12, 4.8), dpi=110)
    # Lay out ONCE. Per-frame tight_layout x set_aspect('equal', adjustable='box')
    # shrinks the boxes a sliver each frame => inward drift. With fixed limits,
    # the box is a deterministic function of the slot, so a one-time layout is
    # stable across frames. Set limits + a 2-line placeholder title + aspect
    # before tight_layout so it reserves the right title/headroom.
    for a in (a1, a2, a3):
        a.set_xlim(ymin, ymax); a.set_ylim(xmin, xmax); a.set_aspect("equal")
        a.set_title(" \n ", fontsize=9)
    fig.suptitle("Imagination > detection for the task — head-on A→B (matched n)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    for i in idxs:
        _topdown_frame(a1, o_trail, o_tur, o_obs, min(i, len(o_trail) - 1) if o_trail else 0,
                       o_lab, "no imagination", goal=o_goal, spawn=o_spawn)
        _topdown_frame(a2, p_trail, p_tur, p_obs, min(i, len(p_trail) - 1) if p_trail else 0,
                       p_lab, "imagines + holds back", goal=p_goal, spawn=p_spawn)
        _topdown_frame(a3, d_trail, d_tur, d_obs, min(i, len(d_trail) - 1) if d_trail else 0,
                       d_lab, "fires too late", goal=d_goal, spawn=d_spawn)
        for a in (a1, a2, a3):
            a.set_xlim(ymin, ymax); a.set_ylim(xmin, xmax)
        fig.canvas.draw()
        im = Image.frombytes("RGBA", fig.canvas.get_width_height(),
                             fig.canvas.buffer_rgba()).convert("RGB")
        frames.append(im)
    plt.close(fig)
    if not frames:
        return {"frames": 0}
    frames[0].save(out, save_all=True, append_images=frames[1:],
                   duration=frame_ms, loop=0, optimize=True, disposal=2)
    return {"frames": len(frames), "oblivious_len": len(o_trail),
            "planner_len": len(p_trail), "detector_len": len(d_trail),
            "size": frames[0].size}


# ----------------------------- main ------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phantom", default="data/showcase_phantom/showcase_env5_ep001.mcap",
                    help="phantom episode MCAP (best-lead default)")
    ap.add_argument("--planner", default="data/showcase_planner/planner_env12_ep001.mcap",
                    help="planner episode MCAP (timeout/evade default; largest trail + close skirt)")
    ap.add_argument("--waypoint", default="data/showcase_waypoint/waypoint_env8_ep001.mcap",
                    help="waypoint episode MCAP (killed default)")
    ap.add_argument("--nav_planner", default="data/nav_planner/planner_env1_ep003.mcap",
                    help="WM A->B planner episode MCAP (holds back at ~9 m, survives)")
    ap.add_argument("--nav_oblivious", default="data/nav_oblivious/nav_env1_ep003.mcap",
                    help="oblivious A->B episode MCAP (same seed as --nav_planner; flies into kill zone, killed)")
    ap.add_argument("--nav_detector", default="data/nav_detector/detector_env1_ep003.mcap",
                    help="detector-reactive A->B episode MCAP (same seed; fires late)")
    ap.add_argument("--only", choices=["signals", "phantom-gif", "survival-gif",
                                       "nav-gif", "all"],
                    default="all")
    args = ap.parse_args()
    os.makedirs(ASSETS, exist_ok=True)
    os.chdir(REPO)

    report = {}
    if args.only in ("all", "signals"):
        o = os.path.join(ASSETS, "phantom_signals.png")
        report["phantom_signals.png"] = make_phantom_signals(args.phantom, o)
        print(f"[ok] {o}  -> {report['phantom_signals.png']}")
    if args.only in ("all", "phantom-gif"):
        o = os.path.join(ASSETS, "phantom_lead.gif")
        report["phantom_lead.gif"] = make_phantom_gif(args.phantom, o)
        print(f"[ok] {o}  -> {report['phantom_lead.gif']}")
    if args.only in ("all", "survival-gif"):
        o = os.path.join(ASSETS, "planner_vs_waypoint.gif")
        report["planner_vs_waypoint.gif"] = make_survival_gif(args.planner, args.waypoint, o)
        print(f"[ok] {o}  -> {report['planner_vs_waypoint.gif']}")
    if args.only in ("all", "nav-gif"):
        o = os.path.join(ASSETS, "nav_a_to_b.gif")
        report["nav_a_to_b.gif"] = make_nav_ab_gif(
            args.nav_planner, args.nav_oblivious, args.nav_detector, o)
        print(f"[ok] {o}  -> {report['nav_a_to_b.gif']}")

    # ---- metadata-only verification (no image inspection) ----
    print("\n=== verify (metadata + numeric; no pixel inspection) ===")
    for fn in ["phantom_signals.png", "phantom_lead.gif", "planner_vs_waypoint.gif",
               "nav_a_to_b.gif"]:
        p = os.path.join(ASSETS, fn)
        if not os.path.exists(p):
            print(f"  MISSING {fn}"); continue
        im = Image.open(p)
        sz = os.path.getsize(p) / 1024
        extra = f", frames={im.n_frames}" if getattr(im, "is_animated", False) else ""
        print(f"  {fn}: {im.size[0]}x{im.size[1]}{extra}, {sz:.0f} KB")


if __name__ == "__main__":
    main()

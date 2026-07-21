#!/usr/bin/env python
"""Phase 7 host-side planner server (swm training venv on the GPU box).

Reuses DangerPlanner + CEMSolver + danger_head from cem_planner.py. Socket server:
the Isaac env driver sends a batch of N POV images (uint8 RGB) + states + a
just_reset mask each control step; this server keeps per-env 3-frame pixel context
+ 2-action history, plans all N envs in ONE batched CEM solve, and returns N
clipped 4-dim body-frame actions in [-1,1].

Why two-process: the Isaac container (py3.11) has no swm/lancedb/stable_pretraining
and a different torch (2.7.0+cu128); the host venv (py3.12) ran training + the
offline eval, so it has every dep. The container is swm-free by design (see
collect_uav_3d.py header). The spt_stub/ vendor dir is NOT needed on the host --
the host imports the real stable_pretraining.

Protocol (stdlib socket + struct + pickle, 4-byte big-endian length prefix):
  req  = {'just_reset': np.bool_ (N,), 'pix': np.uint8 (N,H,W,3) RGB,
          'state': np.float32 (N,21), 'goal': np.float32 (N,3),
          'obstacles': np.float32 (N,16,6)=[cx,cy,cz,hx,hy,hz]}   # nav/planner/detector
  req  = {'just_reset', 'pix'}                                     # phantom (signals only)
  resp = {'action': (N,4), 'wm_danger': (N,), 'det_logit': (N,), 'flee': (N,) bool}  # nav/planner/detector
  resp = {'wm_danger': (N,), 'det_logit': (N,)}                    # phantom (no action)
  # wm_danger  = forward-reference imagined danger logit (fly straight ahead) -- the
  #              'WM predicts ahead' signal (PURE danger, not the combined nav cost).
  # det_logit  = present-frame danger probe logit (the single-frame detector signal).
  # Both are logged to MCAP by the client so Foxglove can plot imagination vs detection
  # on one timeline -- the showcase for "WM predicts ahead of detection".

  --mode nav       : oblivious A->B (danger_weight=0) -- rounds the corner, dies.
  --mode planner   (default): WM A->B (danger_weight=W) -- imagines the turret around
    the corner and detours around the buildings to reach B (proactive).
  --mode detector  : reactive A->B -- nav (danger_weight=0) + present-frame probe that
    overrides the action with a hard evade (reverse + per-ep-random strafe + climb) on
    fire. Reacts to PRESENT danger (turret in frame), not imagined future danger -- so
    the comparison isolates imagination's lead time vs detection's reaction time.
  --mode phantom   : Result-1 showcase signal server (no CEM, no goal).

  python scripts/live/planner_server.py --mode nav --port 5555 --num_envs 8
  python scripts/live/planner_server.py --mode planner --danger_weight 1.0 --port 5555 --num_envs 8
  python scripts/live/planner_server.py --mode detector --port 5555 --num_envs 8
"""
import argparse
import os
import pickle
import random
import socket
import struct
import sys
from collections import deque

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces as gym_spaces

# reuse the proven planner pieces from this repo's uav_wm/planning/cem_planner.py
# (NavPlanner/DangerPlanner/imagined_danger live here; swm itself is an
# editable-installed submodule providing CEMSolver/load_pretrained/PlanConfig).
_HERE = os.path.dirname(os.path.abspath(__file__))
_UAVWM_ROOT = os.path.dirname(os.path.dirname(_HERE))  # .../UAV-World-Model
sys.path.insert(0, _UAVWM_ROOT)  # so `import uav_wm` works without install
from uav_wm.planning.cem_planner import (  # noqa: E402
    ACT_DIM,
    CTX_ACT_LEN,
    DANGER_IDX,
    FRAMESKIP,
    HISTORY,
    K_MAX,
    RAW_ACTION_DIM,
    DangerPlanner,
    NavPlanner,
    compute_action_std,
    img_preprocessor,
    latest_ckpt,
    load_danger_head,
)
from stable_worldmodel.planning.solver.cem import CEMSolver  # noqa: E402
from stable_worldmodel.policy import PlanConfig  # noqa: E402
from stable_worldmodel.wm.utils import load_pretrained  # noqa: E402

# ImageNet norm -- matches spt_data.transforms.ToImage(**ImageNet) used at training
# (mean/std canonical; rgb=False so RGB channel order is preserved, which is what
# the env POV and the JPEG'd training pixels both are). Shape (1,3,1,1) broadcasts
# against (N,3,H,W).
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ---- length-prefixed pickle messaging ----------------------------------------
def recv_exact(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("socket closed by peer")
        buf.extend(chunk)
    return bytes(buf)


def send_msg(sock, obj):
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(struct.pack(">I", len(data)) + data)


def recv_msg(sock):
    (n,) = struct.unpack(">I", recv_exact(sock, 4))
    return pickle.loads(recv_exact(sock, n))


# ---- preprocessing -----------------------------------------------------------
def preprocess(pix_uint8, dev):
    """(N,H,W,3) uint8 RGB -> (N,3,H,W) float, ImageNet-normalized on dev."""
    x = torch.from_numpy(pix_uint8).permute(0, 3, 1, 2).float().to(dev) / 255.0
    return (x - IMAGENET_MEAN.to(dev)) / IMAGENET_STD.to(dev)


class PlannerServer:
    """Holds the model + solver + per-env context buffers; answers plan requests."""

    def __init__(self, model, head, action_std, dev, num_envs,
                 num_samples=128, n_steps=8, topk=16, seed=0, probe=None):
        self.dev = dev
        self.action_std = action_std.to(dev)  # (4,)
        self.N = num_envs
        self.planner = DangerPlanner(model, head, action_std).to(dev)
        self.probe = probe  # present-frame danger probe (Phase-4) -- for the detector signal
        action_space = gym_spaces.Box(
            low=-1.0, high=1.0, shape=(1, RAW_ACTION_DIM), dtype=np.float32
        )
        cfg = PlanConfig(horizon=K_MAX, receding_horizon=1, action_block=1)
        # batch_size=num_envs -> plan all envs in one inner batch (vectorized CEM)
        self.solver = CEMSolver(
            cost=self.planner, num_samples=num_samples, n_steps=n_steps,
            topk=topk, device=dev, seed=seed, batch_size=num_envs,
        )
        self.solver.configure(action_space=action_space, n_envs=num_envs, config=cfg)
        self.pix_buf = [deque(maxlen=HISTORY) for _ in range(num_envs)]
        self.act_buf = [deque(maxlen=CTX_ACT_LEN) for _ in range(num_envs)]
        self._n_req = 0

    def plan(self, just_reset, pix_uint8):
        # N from the incoming batch, not self.N: the Isaac driver sends exactly
        # num_envs, but deriving from the request keeps client/server decoupled
        # (e.g. the synthetic smoke client sends 2 against a num_envs=8 server).
        N = len(pix_uint8)
        assert N <= self.N, f"batch {N} > server num_envs {self.N} (buffers undersized)"
        just_reset = np.asarray(just_reset).astype(bool)
        for i in range(N):
            if just_reset[i]:
                self.pix_buf[i].clear()
                self.act_buf[i].clear()
        frames = preprocess(pix_uint8, self.dev)  # (N,3,H,W)
        for i in range(N):
            self.pix_buf[i].append(frames[i])
        pix_ctx = torch.stack([self._pad_pix(i) for i in range(N)])  # (N,3,C,H,W)
        ctx_act4 = torch.stack([self._pad_act(i) for i in range(N)])  # (N,2,4)
        ctx_act = ctx_act4.repeat(1, 1, FRAMESKIP)  # (N,2,20)
        info = {"pixels": pix_ctx, "context_action": ctx_act}
        with torch.inference_mode():
            out = self.solver.solve(info)
        plan_norm = out["actions"].to(self.dev)  # (N,K_MAX,4) normalized
        first = plan_norm[:, 0] * self.action_std  # (N,4) dataset units
        action = first.clamp(-1.0, 1.0)  # env body-frame bounds
        for i in range(N):
            self.act_buf[i].append(action[i].detach())
        self._n_req += 1
        # Showcase signals (exposed for MCAP viz; the 'WM predicts ahead of
        # detection' story):
        #   wm_danger  = imagined danger logit of a FORWARD-REFERENCE plan (fly
        #                straight ahead along the drone's heading). Independent of
        #                the selected (evading) plan, so it RISES as the drone heads
        #                toward the turret and LEADS the present-frame detector --
        #                the WM imagines the turret becoming dangerous before it is
        #                aimed/in-range in the current frame. (The selected plan's
        #                own imagined danger stays LOW when the planner evades, so
        #                it is the wrong signal for a 'predicts ahead' showcase.)
        #   det_logit  = present-frame danger probe logit (sees ONLY the current POV --
        #                the single-frame detector baseline's signal).
        # Both are ~free: get_cost on one candidate is 1/num_samples of the CEM solve;
        # the probe is one linear pass on the already-encoded present frame.
        with torch.inference_mode():
            # get_cost expects (B,S,...); plan()'s info is per-env (N,...). solver.solve
            # adds the B+S dims internally; this standalone call must add S=1 itself.
            info_sel = {"pixels": pix_ctx.unsqueeze(1),            # (N,1,3,C,H,W)
                        "context_action": ctx_act.unsqueeze(1)}    # (N,1,2,20)
            fwd = torch.zeros(N, K_MAX, RAW_ACTION_DIM, device=self.dev)
            fwd[:, :, 0] = 1.0  # full forward (vx=1), no strafe/climb/yaw
            wm_danger = self.planner.get_cost(info_sel, fwd.unsqueeze(1)).squeeze(1)  # (N,)
            present = pix_ctx[:, -1:, :, :, :]  # (N,1,C,H,W) = current frame
            emb_now = self.planner.model.encode({"pixels": present})["emb"][:, 0, :]  # (N,192)
            det_logit = self.probe(emb_now).squeeze(-1)  # (N,)
        if self._n_req % 25 == 0:
            print(f"[PLANNER] req={self._n_req} planned {N} envs "
                  f"wm_danger[med={float(wm_danger.median()):.2f} "
                  f"max={float(wm_danger.max()):.2f}] "
                  f"det_logit[med={float(det_logit.median()):.2f} "
                  f"max={float(det_logit.max()):.2f}]", flush=True)
        return {
            "action": action.detach().cpu().numpy().astype(np.float32),
            "wm_danger": wm_danger.detach().cpu().numpy().astype(np.float32),
            "det_logit": det_logit.detach().cpu().numpy().astype(np.float32),
        }

    def _pad_pix(self, i):
        b = list(self.pix_buf[i])  # 1..HISTORY frames
        while len(b) < HISTORY:
            b.insert(0, b[0])  # repeat first at the front (live approx until filled)
        return torch.stack(b)  # (3,C,H,W)

    def _pad_act(self, i):
        b = list(self.act_buf[i])  # 0..CTX_ACT_LEN actions
        pad = torch.zeros(RAW_ACTION_DIM, device=self.dev)
        while len(b) < CTX_ACT_LEN:
            b.insert(0, pad)
        return torch.stack(b)  # (2,4)


# ---- detector mode (Phase-4 present-frame probe + reactive flee) -------------
def load_danger_probe(dev):
    """nn.Linear(192,1) present-frame probe (danger_probe.pt). Mirrors
    load_danger_head but the probe artifact saves only state_dict/auroc/ap
    (no in_dim, no threshold). in_dim=192 = LeWM hidden_size (see startup log)."""
    import stable_worldmodel as swm
    p = swm.data.utils.get_cache_dir(sub_folder="checkpoints") / "lewm" / "danger_probe.pt"
    blob = torch.load(p, map_location="cpu")
    head = nn.Linear(192, 1)
    head.load_state_dict(blob["state_dict"])
    head.eval().to(dev)
    for q in head.parameters():
        q.requires_grad_(False)
    print(f"[DETECTOR] probe loaded (val_auroc={blob.get('val_auroc', float('nan')):.4f})",
          flush=True)
    return head


def derive_threshold(model, probe, dev, dataset, n_val=4000, seed=0):
    """Re-derive the probe's best-F1 threshold on a val subset. danger_probe.pt
    saves only auroc/ap, not the threshold, so the operating point must be
    re-derived. Mirrors danger_probe.py exactly: per-frame load (num_steps=1),
    the same ImageNet-norm transform the probe trained on, encode present frame,
    probe logit, best-F1 operating point. ~30 s one-time at startup."""
    import stable_worldmodel as swm
    from torch.utils.data import DataLoader, Subset
    from sklearn.metrics import roc_auc_score

    ds = swm.data.load_dataset(
        dataset, num_steps=1, frameskip=1, keys_to_load=["pixels", "state"]
    )
    ds.transform = img_preprocessor()  # same ImageNet norm the probe trained on
    rng = random.Random(seed)
    n = len(ds)
    split = int(n * 0.8)
    idx = rng.sample(range(split, n), min(n_val, n - split))
    # num_workers=0: each worker would re-import swm (slow spinup, ~10s/worker);
    # 4000 frames / batch 256 = 16 batches, single-process encode is ~seconds.
    loader = DataLoader(Subset(ds, idx), batch_size=256, num_workers=0, shuffle=False)
    lats, labs = [], []
    with torch.inference_mode():
        for bi, batch in enumerate(loader):
            px = batch["pixels"].to(dev)            # (B,1,C,H,W)
            st = batch["state"]                      # (B,1,21)
            emb = model.encode({"pixels": px})["emb"][:, 0, :]  # (B,192)
            lats.append(emb.float().cpu())
            labs.append(st[:, 0, DANGER_IDX].cpu())
            if bi % 10 == 0:
                print(f"[DETECTOR]   thr-val batch {bi} emb={tuple(emb.shape)}", flush=True)
    X = torch.cat(lats).to(dev)
    y = torch.cat(labs).float().cpu().numpy()
    with torch.inference_mode():
        sc = probe(X).squeeze(-1).cpu().numpy()
    auroc = roc_auc_score(y, sc)
    order = np.argsort(-sc)
    ys = y[order]
    tp = np.cumsum(ys); fp = np.cumsum(1 - ys); P = ys.sum()
    prec = tp / np.maximum(tp + fp, 1); rec = tp / max(P, 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    bi = int(f1.argmax())
    thr = float(sc[order[bi]])
    print(f"[DETECTOR] best-F1 threshold thr={thr:.4f} f1={f1[bi]:.4f} "
          f"prec={prec[bi]:.4f} rec={rec[bi]:.4f} val_auroc={auroc:.4f} "
          f"(n_val={len(y)} prevalence={y.mean():.4f})", flush=True)
    return thr


class DetectorServer:
    """Present-frame danger probe + reactive flee. Vision-only (same info the
    planner has): encode the current POV -> probe logit -> fire if > threshold.
    On fire: hard evade [reverse, per-ep-random strafe, climb, no yaw] -- pure
    translation away from the in-frame threat (no ground-truth turret bearing).
    On no-fire: zeros + flee=False; the client follows its waypoint. So the
    detector-reactive baseline == the oblivious waypoint baseline + detection-
    triggered evasion -- the ONLY difference from oblivious is the evasion,
    making kill differences cleanly attributable to reaction vs imagination."""

    FLEE_BASIS = torch.tensor([-1.0, 0.0, 1.0, 0.0])  # vx reverse, vz climb, no yaw

    def __init__(self, model, probe, threshold, dev, num_envs, seed=0):
        self.dev = dev
        self.model = model
        self.probe = probe
        self.threshold = float(threshold)
        self.N = num_envs
        self.rng = random.Random(seed + 7)
        self.strafe_sign = torch.tensor(
            [1.0 if self.rng.random() < 0.5 else -1.0 for _ in range(num_envs)],
            device=dev,
        )
        self.flee_basis = self.FLEE_BASIS.to(dev)  # (4,)
        self._n_req = 0
        self._n_fire = 0

    def plan(self, just_reset, pix_uint8):
        N = len(pix_uint8)
        assert N <= self.N, f"batch {N} > server num_envs {self.N}"
        just_reset = np.asarray(just_reset).astype(bool)
        for i in range(N):
            if just_reset[i]:  # re-roll strafe sign per episode (no single-frame L/R cue)
                self.strafe_sign[i] = 1.0 if self.rng.random() < 0.5 else -1.0
        x = preprocess(pix_uint8, self.dev)           # (N,3,H,W)
        px = x.unsqueeze(1)                           # (N,1,3,H,W) -- encode present frame
        with torch.inference_mode():
            emb = self.model.encode({"pixels": px})["emb"][:, 0, :]  # (N,192)
            logit = self.probe(emb).squeeze(-1)      # (N,)
        flee = logit > self.threshold                 # (N,) bool
        action = torch.zeros(N, RAW_ACTION_DIM, device=self.dev)
        if flee.any():
            basis = self.flee_basis.expand(N, RAW_ACTION_DIM).clone()  # (N,4)
            basis[:, 1] = self.strafe_sign           # per-env strafe sign
            action[flee] = basis[flee]               # reverse + strafe + climb, no yaw
        self._n_req += 1
        self._n_fire += int(flee.sum().item())
        if self._n_req % 25 == 0:
            print(f"[DETECTOR] req={self._n_req} fired {int(flee.sum().item())}/{N} "
                  f"(cum fire-rate {self._n_fire / max(self._n_req * N, 1):.3f}) "
                  f"logit[min={float(logit.min()):.2f} med={float(logit.median()):.2f} "
                  f"max={float(logit.max()):.2f}] thr={self.threshold:.2f}", flush=True)
        return (
            action.detach().cpu().numpy().astype(np.float32),
            flee.detach().cpu().numpy(),
            logit.detach().cpu().numpy().astype(np.float32),  # det_logit for viz
        )


class PhantomServer:
    """Showcase signal server: NO CEM. The drone flies a fixed oblivious plan
    (the CLIENT commands it -- forward in body frame, CRUISE vx). This server
    only computes the two 'WM predicts ahead of detection' signals for MCAP
    logging; it returns NO action.

      wm_danger = imagined danger logit of a forward-reference plan (fly straight
                  ahead, [1,0,0,0] x action_std), rolled t+1..t+4 by the danger
                  head. RISES as the drone closes on the turret and LEADS the
                  present-frame detector (the WM imagines the looming before the
                  probe confirms aim/in-range).
      det_logit = present-frame danger probe logit (the single-frame detector).

    The act_buf records the forward action the drone ACTUALLY flies (CRUISE), so
    the WM's action context matches reality. Reusing PlannerServer here would
    record the evasive CEM action the drone never flew -- a wrong context. The
    showcase drone is oblivious (flies into the turret head-on), so the forward
    reference is the right 'what happens if I keep going' question.
    """

    def __init__(self, model, head, action_std, dev, num_envs, probe, cruise=0.7):
        self.dev = dev
        self.action_std = action_std.to(dev)  # (4,)
        self.N = num_envs
        self.planner = DangerPlanner(model, head, action_std).to(dev)
        self.probe = probe
        self.cruise = float(cruise)  # vx the client flies (must match live_demo showcase)
        self.pix_buf = [deque(maxlen=HISTORY) for _ in range(num_envs)]
        self.act_buf = [deque(maxlen=CTX_ACT_LEN) for _ in range(num_envs)]
        self._n_req = 0

    def _pad_pix(self, i):
        b = list(self.pix_buf[i])
        while len(b) < HISTORY:
            b.insert(0, b[0])
        return torch.stack(b)

    def _pad_act(self, i):
        b = list(self.act_buf[i])
        pad = torch.zeros(RAW_ACTION_DIM, device=self.dev)
        while len(b) < CTX_ACT_LEN:
            b.insert(0, pad)
        return torch.stack(b)

    def plan(self, just_reset, pix_uint8):
        N = len(pix_uint8)
        assert N <= self.N, f"batch {N} > server num_envs {self.N}"
        just_reset = np.asarray(just_reset).astype(bool)
        for i in range(N):
            if just_reset[i]:
                self.pix_buf[i].clear()
                self.act_buf[i].clear()
        frames = preprocess(pix_uint8, self.dev)  # (N,3,H,W)
        for i in range(N):
            self.pix_buf[i].append(frames[i])
        pix_ctx = torch.stack([self._pad_pix(i) for i in range(N)])  # (N,3,C,H,W)
        ctx_act4 = torch.stack([self._pad_act(i) for i in range(N)])  # (N,2,4)
        ctx_act = ctx_act4.repeat(1, 1, FRAMESKIP)  # (N,2,20)
        # record the forward action the drone actually flies (matches the client's
        # oblivious CRUISE forward) so the WM action context == reality.
        flown = torch.zeros(N, RAW_ACTION_DIM, device=self.dev)
        flown[:, 0] = self.cruise
        for i in range(N):
            self.act_buf[i].append(flown[i].detach())
        with torch.inference_mode():
            info_sel = {"pixels": pix_ctx.unsqueeze(1),            # (N,1,3,C,H,W)
                        "context_action": ctx_act.unsqueeze(1)}    # (N,1,2,20)
            fwd = torch.zeros(N, K_MAX, RAW_ACTION_DIM, device=self.dev)
            fwd[:, :, 0] = 1.0  # full forward (vx=1), no strafe/climb/yaw
            wm_danger = self.planner.get_cost(info_sel, fwd.unsqueeze(1)).squeeze(1)
            present = pix_ctx[:, -1:, :, :, :]  # (N,1,C,H,W) = current frame
            emb_now = self.planner.model.encode({"pixels": present})["emb"][:, 0, :]
            det_logit = self.probe(emb_now).squeeze(-1)
        self._n_req += 1
        if self._n_req % 25 == 0:
            print(f"[PHANTOM] req={self._n_req} N={N} "
                  f"wm_danger[med={float(wm_danger.median()):.2f} "
                  f"max={float(wm_danger.max()):.2f}] "
                  f"det_logit[med={float(det_logit.median()):.2f} "
                  f"max={float(det_logit.max()):.2f}]", flush=True)
        return {
            "wm_danger": wm_danger.detach().cpu().numpy().astype(np.float32),
            "det_logit": det_logit.detach().cpu().numpy().astype(np.float32),
        }


class NavServer:
    """Unified A->B navigation engine (nav / planner / detector modes).

    NavPlanner (CEM: goal + building + alt + imagined-danger) produces the action;
    the present-frame probe produces det_logit and (in detector mode) a reactive
    flee override. The three A->B modes differ in exactly one knob each:
      nav      : danger_weight=0, reactive_flee=False  (oblivious A->B)
      planner  : danger_weight=W, reactive_flee=False  (WM, proactive)
      detector : danger_weight=0, reactive_flee=True   (probe overrides action)
    so kill/reach differences are cleanly attributable to imagination vs reaction.

    Returns {action, wm_danger, det_logit, flee} every step. wm_danger is the
    forward-reference imagined danger (fly straight ahead) -- the 'WM predicts ahead'
    signal -- computed via NavPlanner._danger_term (PURE danger, NOT the combined
    get_cost) so it is comparable across modes and to the phantom showcase.
    """

    FLEE_BASIS = torch.tensor([-1.0, 0.0, 1.0, 0.0])  # vx reverse, vz climb, no yaw

    def __init__(self, model, head, probe, action_std, dev, num_envs, *,
                 danger_weight, reactive_flee, flee_threshold,
                 w_goal=1.0, w_build=2.0, w_alt=0.1, w_yaw=1.0, build_margin=0.5,
                 num_samples=128, n_steps=8, topk=16, seed=0):
        self.dev = dev
        self.action_std = action_std.to(dev)
        self.N = num_envs
        self.probe = probe
        self.reactive_flee = bool(reactive_flee)
        self.flee_threshold = float(flee_threshold)
        self.planner = NavPlanner(
            model, head, action_std,
            w_danger=danger_weight, w_goal=w_goal, w_build=w_build, w_alt=w_alt,
            w_yaw=w_yaw, build_margin=build_margin,
        ).to(dev)
        action_space = gym_spaces.Box(
            low=-1.0, high=1.0, shape=(1, RAW_ACTION_DIM), dtype=np.float32
        )
        cfg = PlanConfig(horizon=K_MAX, receding_horizon=1, action_block=1)
        # batch_size=num_envs -> plan all envs in one inner batch (vectorized CEM)
        self.solver = CEMSolver(
            cost=self.planner, num_samples=num_samples, n_steps=n_steps,
            topk=topk, device=dev, seed=seed, batch_size=num_envs,
        )
        self.solver.configure(action_space=action_space, n_envs=num_envs, config=cfg)
        self.pix_buf = [deque(maxlen=HISTORY) for _ in range(num_envs)]
        self.act_buf = [deque(maxlen=CTX_ACT_LEN) for _ in range(num_envs)]
        self.rng = random.Random(seed + 7)
        self.strafe_sign = torch.tensor(
            [1.0 if self.rng.random() < 0.5 else -1.0 for _ in range(num_envs)],
            device=dev,
        )
        self.flee_basis = self.FLEE_BASIS.to(dev)
        self._n_req = 0

    def _pad_pix(self, i):
        b = list(self.pix_buf[i])
        while len(b) < HISTORY:
            b.insert(0, b[0])
        return torch.stack(b)

    def _pad_act(self, i):
        b = list(self.act_buf[i])
        pad = torch.zeros(RAW_ACTION_DIM, device=self.dev)
        while len(b) < CTX_ACT_LEN:
            b.insert(0, pad)
        return torch.stack(b)

    def plan(self, just_reset, pix_uint8, state_np, goal_np, obstacles_np):
        N = len(pix_uint8)
        assert N <= self.N, f"batch {N} > server num_envs {self.N} (buffers undersized)"
        just_reset = np.asarray(just_reset).astype(bool)
        for i in range(N):
            if just_reset[i]:
                self.pix_buf[i].clear()
                self.act_buf[i].clear()
                self.strafe_sign[i] = 1.0 if self.rng.random() < 0.5 else -1.0
        frames = preprocess(pix_uint8, self.dev)                       # (N,3,H,W)
        for i in range(N):
            self.pix_buf[i].append(frames[i])
        pix_ctx = torch.stack([self._pad_pix(i) for i in range(N)])    # (N,3,C,H,W)
        ctx_act4 = torch.stack([self._pad_act(i) for i in range(N)])   # (N,2,4)
        ctx_act = ctx_act4.repeat(1, 1, FRAMESKIP)                     # (N,2,20)
        state = torch.from_numpy(np.asarray(state_np, dtype=np.float32)).to(self.dev)        # (N,21)
        goal = torch.from_numpy(np.asarray(goal_np, dtype=np.float32)).to(self.dev)          # (N,3)
        obstacles = torch.from_numpy(np.asarray(obstacles_np, dtype=np.float32)).to(self.dev)  # (N,16,6)
        info = {"pixels": pix_ctx, "context_action": ctx_act,
                "state": state, "goal": goal, "obstacles": obstacles}
        with torch.inference_mode():
            out = self.solver.solve(info)
        plan_norm = out["actions"].to(self.dev)        # (N,K_MAX,4) normalized
        first = plan_norm[:, 0] * self.action_std      # (N,4) dataset units
        action = first.clamp(-1.0, 1.0)                # (N,4) env body-frame bounds
        for i in range(N):
            self.act_buf[i].append(action[i].detach())
        # Showcase signals + reactive flee (cheap: one danger pass + one probe pass).
        # wm_danger uses _danger_term (PURE danger), NOT get_cost -- get_cost would
        # return the COMBINED cost (wrong signal) and is only right for DangerPlanner.
        with torch.inference_mode():
            info_sel = {"pixels": pix_ctx.unsqueeze(1),            # (N,1,3,C,H,W)
                        "context_action": ctx_act.unsqueeze(1),    # (N,1,2,20)
                        "state": state.unsqueeze(1),               # (N,1,21)
                        "goal": goal.unsqueeze(1),                 # (N,1,3)
                        "obstacles": obstacles.unsqueeze(1)}       # (N,1,16,6)
            fwd = torch.zeros(N, K_MAX, RAW_ACTION_DIM, device=self.dev)
            fwd[:, :, 0] = 1.0  # forward-reference plan (vx=1, no strafe/climb/yaw)
            wm_danger = self.planner._danger_term(info_sel, fwd.unsqueeze(1)).squeeze(1)  # (N,)
            present = pix_ctx[:, -1:, :, :, :]                     # (N,1,C,H,W) current frame
            emb_now = self.planner.model.encode({"pixels": present})["emb"][:, 0, :]      # (N,192)
            det_logit = self.probe(emb_now).squeeze(-1)            # (N,)
            flee = torch.zeros(N, dtype=torch.bool, device=self.dev)
            if self.reactive_flee:
                flee = det_logit > self.flee_threshold
                if flee.any():
                    basis = self.flee_basis.expand(N, RAW_ACTION_DIM).clone()  # (N,4)
                    basis[:, 1] = self.strafe_sign               # per-env strafe sign
                    action = action.clone()
                    action[flee] = basis[flee]                   # reverse + strafe + climb
        self._n_req += 1
        if self._n_req % 25 == 0:
            print(f"[NAV] req={self._n_req} N={N} dw={self.planner.w_danger} "
                  f"flee={self.reactive_flee} wm_danger[med={float(wm_danger.median()):.2f} "
                  f"max={float(wm_danger.max()):.2f}] det_logit[med={float(det_logit.median()):.2f} "
                  f"max={float(det_logit.max()):.2f}] flee_n={int(flee.sum())}", flush=True)
        return {
            "action": action.detach().cpu().numpy().astype(np.float32),
            "wm_danger": wm_danger.detach().cpu().numpy().astype(np.float32),
            "det_logit": det_logit.detach().cpu().numpy().astype(np.float32),
            "flee": flee.detach().cpu().numpy(),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["nav", "planner", "detector", "phantom"],
                    default="planner",
                    help="nav=oblivious A->B; planner=WM A->B (proactive); "
                         "detector=reactive A->B (probe flee); phantom=Result-1 "
                         "showcase signal server (no CEM, no goal)")
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--num_envs", type=int, default=16)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--dataset", default="uav_isaac_train.lance")
    ap.add_argument("--num_samples", type=int, default=128)
    ap.add_argument("--n_steps", type=int, default=8)
    ap.add_argument("--topk", type=int, default=16)
    ap.add_argument("--threshold", type=float, default=None,
                    help="detector fire threshold (default: re-derive best-F1)")
    ap.add_argument("--danger_weight", type=float, default=1.0,
                    help="planner mode: imagined-danger weight -- the proactive/"
                         "anticipatory knob (0=oblivious). nav/detector force 0.")
    ap.add_argument("--goal_weight", type=float, default=2.0,
                    help="goal-progress weight (strong vs building so the oblivious "
                         "drone is pulled into the corridor, not stalled by a build hill)")
    ap.add_argument("--building_weight", type=float, default=1.0,
                    help="building-repulsion weight (kept moderate: strong enough to "
                         "avoid crashes, weak enough that goal wins along the corridor)")
    ap.add_argument("--alt_weight", type=float, default=0.1)
    ap.add_argument("--yaw_weight", type=float, default=1.0,
                    help="yaw-rate penalty -- pins the drone to its approach heading so "
                         "it keeps the turret in the POV (the WM can only imagine a "
                         "turret it can see); damps the yaw drift/gaming that otherwise "
                         "rotates the turret out of frame before firing range")
    ap.add_argument("--build_margin", type=float, default=0.5,
                    help="building repulsion softplus margin (m)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tag = args.mode.upper()  # NAV | PLANNER | DETECTOR | PHANTOM

    ckpt = args.ckpt or latest_ckpt()
    print(f"[{tag}] loading LeWM {ckpt} on {dev}", flush=True)
    model = load_pretrained(ckpt).eval().to(dev)
    for p in model.parameters():
        p.requires_grad_(False)

    # all modes need head (danger) + probe (det_logit) + action_std (CEM/exec scaling).
    # phantom uses DangerPlanner for the forward-ref signal; nav/planner/detector use
    # the unified NavServer (NavPlanner). The detector mode also needs the probe's
    # best-F1 fire threshold for the reactive flee override.
    head = load_danger_head(dev)
    probe = load_danger_probe(dev)
    import stable_worldmodel as swm
    ds = swm.data.load_dataset(
        args.dataset, num_steps=HISTORY + K_MAX, frameskip=FRAMESKIP,
        keys_to_load=["pixels", "action", "state", "shot"],
    )
    ds.transform = None
    action_std = compute_action_std(ds).to(dev)
    print(f"[{tag}] action_std(4)={[round(x, 4) for x in action_std.cpu().tolist()]}",
          flush=True)

    threshold = None
    if args.mode == "phantom":
        # Result-1 showcase signal server: no CEM, no goal. Untouched.
        server = PhantomServer(model, head, action_std, dev, args.num_envs, probe=probe)
    else:
        # nav / planner / detector -> unified NavServer. mode -> (dw, flee): one knob
        # differs per mode so kill/reach differences attribute to imagination vs reaction.
        if args.mode == "planner":
            danger_weight, reactive_flee = args.danger_weight, False
        elif args.mode == "nav":
            danger_weight, reactive_flee = 0.0, False
        else:  # detector
            danger_weight, reactive_flee = 0.0, True
            threshold = (args.threshold if args.threshold is not None
                         else derive_threshold(model, probe, dev, args.dataset, seed=args.seed))
        server = NavServer(
            model, head, probe, action_std, dev, args.num_envs,
            danger_weight=danger_weight, reactive_flee=reactive_flee,
            flee_threshold=threshold if threshold is not None else 0.0,
            w_goal=args.goal_weight, w_build=args.building_weight,
            w_alt=args.alt_weight, w_yaw=args.yaw_weight, build_margin=args.build_margin,
            num_samples=args.num_samples, n_steps=args.n_steps,
            topk=args.topk, seed=args.seed,
        )

    extra = ""
    if args.mode == "planner":
        extra = f", dw={args.danger_weight}"
    elif args.mode == "detector" and threshold is not None:
        extra = f", thr={threshold:.4f}"
    print(f"[{tag}] ready, listening on {args.host}:{args.port} "
          f"(num_envs={args.num_envs}{extra})", flush=True)

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((args.host, args.port))
    srv.listen(1)
    conn, addr = srv.accept()
    print(f"[{tag}] connection from {addr}", flush=True)
    try:
        while True:
            req = recv_msg(conn)
            if args.mode == "phantom":
                resp = server.plan(req["just_reset"], req["pix"])  # {wm_danger, det_logit}
            else:  # nav / planner / detector -> NavServer threads state/goal/obstacles
                resp = server.plan(req["just_reset"], req["pix"],
                                   req["state"], req["goal"], req["obstacles"])
            # .tolist() on all: version-proof pickle (host numpy 2.x -> container numpy 1.x)
            send_msg(conn, {k: v.tolist() for k, v in resp.items()})
    except (ConnectionError, EOFError, KeyboardInterrupt) as e:
        print(f"[{tag}] session ended ({type(e).__name__})", flush=True)
    finally:
        conn.close()
        srv.close()


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Phase 6: danger-aware CEM planner (offline eval).

The planner: CEM searches 4-dim RAW actions (vx, vy, vz, yaw_rate) over a 4-step
(~2 s) horizon -- these are the t->t+1 ... t+3->t+4 transitions. For each
candidate sequence the DangerPlanner Costable:
  1. scales the normalized 4-dim action to dataset units, repeats each 5x -> 20-dim
     (the frameskip-stack LeWM.action_encoder expects),
  2. prepends the H-1=2 RECORDED context actions (act[t-2], act[t-1]) ->
     (B,S,6,20). The first SAMPLED action lands in act_0[2] = act[t], so it drives
     the t->t+1 prediction -> ALL of t+1..t+4 are controllable (fixes the
     action-invariant-cost bug from prepending H recorded actions),
  3. rolls LeWM.rollout (H=3 context frames, T=6) -> predicted_emb (B,S,7,192),
  4. applies the frozen danger head to imagined frames t+1..t+4 (indices 3..6),
  5. cost = max danger logit over the horizon (avoid the worst imagined danger).
CEM keeps the lowest-cost elites; the planner returns the first 4-dim action
(the t->t+1 transition = the next action to execute).

OFFLINE EVAL -- circular (the WM is both imaginer and evaluator; the live gate is
Phase 7). On held-out val windows we check:
  - mechanical: valid 4-dim actions returned;
  - danger-min: the planner's chosen plan has lower imagined danger than random /
    zero / the RECORDED future action (which actually flew on these windows);
  - qualitative: on approach-to-danger windows (present safe, future danger) the
    planner diverges from the recorded action; on safe windows it ~follows.

  python -m uav_wm.planning.cem_planner
"""
import argparse
import random

import numpy as np
import stable_worldmodel as swm
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces as gym_spaces
from torch.utils.data import DataLoader, Subset

from stable_pretraining import data as spt_data
from stable_worldmodel.planning.solver.cem import CEMSolver
from stable_worldmodel.policy import PlanConfig
from stable_worldmodel.wm.utils import load_pretrained

DANGER_IDX = 13
HISTORY = 3  # wm.history_size (context frames t-2,t-1,t)
CTX_ACT_LEN = HISTORY - 1  # 2 recorded context actions (act[t-2], act[t-1])
K_MAX = 4  # plan horizon: control t+1..t+4 (~2s)
FRAMESKIP = 5
RAW_ACTION_DIM = 4
ACT_DIM = RAW_ACTION_DIM * FRAMESKIP  # 20
T_PLAN = CTX_ACT_LEN + K_MAX  # 6: action_sequence length passed to rollout


def img_preprocessor(img_size: int = 224):
    tr = spt_data.transforms
    return tr.Compose(
        tr.ToImage(**spt_data.dataset_stats.ImageNet, source='pixels', target='pixels'),
        tr.Resize(img_size, source='pixels', target='pixels'),
    )


def latest_ckpt(run_name: str = 'lewm'):
    d = swm.data.utils.get_cache_dir(sub_folder='checkpoints') / run_name
    cands = sorted(
        d.glob('weights_epoch_*.pt'), key=lambda p: int(p.stem.split('_')[-1])
    )
    if not cands:
        raise FileNotFoundError(f"No weights_epoch_*.pt in {d}")
    return f'{run_name}/{cands[-1].name}'


def load_danger_head(dev):
    p = swm.data.utils.get_cache_dir(sub_folder='checkpoints') / 'lewm' / 'danger_head.pt'
    blob = torch.load(p, map_location='cpu')
    head = nn.Linear(blob['in_dim'], 1)
    head.load_state_dict(blob['state_dict'])
    head.eval().to(dev)
    for q in head.parameters():
        q.requires_grad_(False)
    print(f'[CEM] danger_head loaded (val_auroc={blob.get("val_auroc", float("nan")):.4f})')
    return head


def compute_action_std(ds, n=4000, seed=0):
    """Per-dim std of the RAW 4-dim action, from frameskip-stacked 20-dim samples."""
    rng = random.Random(seed)
    idxs = rng.sample(range(len(ds)), min(n, len(ds)))
    acts = []
    for i in idxs:
        a = ds[i]['action']  # (7,20)
        acts.append(a.reshape(-1, FRAMESKIP, RAW_ACTION_DIM))  # (7,5,4)
    A = torch.cat([x.reshape(-1, RAW_ACTION_DIM) for x in acts])  # (7*n*5, 4)
    return A.std(dim=0)


class DangerPlanner(nn.Module):
    """Costable for CEM: roll out [ctx_act(2); scaled cand(4)] via LeWM, score max imagined danger."""

    def __init__(self, model, danger_head, action_std, history=HISTORY,
                 n_future=K_MAX, frameskip=FRAMESKIP):
        super().__init__()
        self.model = model
        self.danger_head = danger_head  # frozen, registered (so parameters()->dtype)
        self.register_buffer('action_std', action_std)  # (4,)
        self.H = history
        self.n_future = n_future
        self.frameskip = frameskip

    def _danger_term(self, info_dict, action_candidates):
        # action_candidates: (B, S, K_MAX, 4) normalized. Returns the worst imagined
        # danger logit over the horizon (B, S). Factored out so NavPlanner can reuse
        # the danger path AND so the forward-reference showcase signal can read PURE
        # danger (not a combined cost). Extra info keys (state/goal/obstacles) are
        # carried into rollout->encode but ignored by encode (pixels-only).
        ctx_act = info_dict['context_action']  # (B, S, 2, 20) recorded
        cand = action_candidates * self.action_std  # -> dataset units, (B,S,K,4)
        cand20 = cand.repeat(1, 1, 1, self.frameskip)  # (B,S,K,20)
        full_act = torch.cat([ctx_act, cand20], dim=2)  # (B,S,6,20); act_0[2]=cand[0]
        info = self.model.rollout(info_dict, full_act, history_size=self.H)
        pred = info['predicted_emb']  # (B,S,7,192); indices 3..6 = t+1..t+4
        future = pred[:, :, self.H : self.H + self.n_future, :]  # (B,S,4,192)
        danger = self.danger_head(future).squeeze(-1)  # (B,S,4)
        return danger.max(dim=-1).values  # (B,S) worst imagined danger

    def get_cost(self, info_dict, action_candidates):
        return self._danger_term(info_dict, action_candidates)


class NavPlanner(DangerPlanner):
    """Goal-directed A->B Costable: imagined danger (WM) + goal progress + building
    repulsion + altitude hold.

        cost = w_danger*(danger/scale_danger) + w_goal*(goal/scale_goal)
             + w_build*(build/scale_build) + w_alt*alt

    ``w_danger=0`` => oblivious A->B (the WM rollout is SKIPPED in the CEM cost; the
    WM is still used for the forward-reference showcase signal). ``w_danger>0`` =>
    the WM planner (proactive). The detector-reactive baseline is w_danger=0 nav with
    a present-frame probe that overrides the action (handled in NavServer, not here).

    The goal/building/alt terms integrate a KINEMATIC forward model over the SAME
    candidate the WM rollout consumes. The two terms scale the candidate differently
    because they model different consumers -- do NOT cross them:
      * danger term: ``c * action_std`` (dataset units, UNCLAMPED -- what the WM was
        trained on; matches DangerPlanner).
      * kinematic terms: ``clamp(c * action_std, -1, 1) * [MAX_SPEED, ...]`` -- the
        FULL env execution chain (candidate -> *action_std -> clamp -> *MAX_SPEED in
        process_actions -> velocity). The clamp mirrors NavServer's
        ``action = (plan_norm * action_std).clamp(-1, 1)``; it is a no-op when
        action_std < 1 (typical) but keeps the model honest for action_std > 1.

    Kinematic model (must match uav_turret_3d.apply_actions; the host planner_server
    cannot import that Isaac env, so the scalars are hardcoded here and validated by
    the forward-model unit test): one 4-dim candidate is held for FRAMESKIP=5 env
    sub-steps (0.5 s = one WM window); position moves under the START-of-sub-step
    heading and yaw updates AFTER (matches apply_actions reading the start-of-step
    quat for vel_des). K_MAX windows x FRAMESKIP sub-steps = 2.0 s. This is an
    IDEAL-velocity model -- the real drone tracks velocity via a PD cascade
    (~0.5 m/s^2), so predicted positions LEAD real ones; benign for relative candidate
    ranking, and the danger term (WM-based) is dynamics-honest.
    """

    # uav_turret_3d.py scalars (must match; validated by the forward-model unit test)
    MAX_SPEED = 2.0
    MAX_YAW_RATE = 2.0
    STEP_DT = 0.1
    DRONE_Z = 1.5

    def __init__(self, model, danger_head, action_std, *,
                 w_danger=1.0, w_goal=1.0, w_build=2.0, w_alt=0.1, w_yaw=1.0,
                 scale_danger=3.0, scale_goal=30.0, scale_build=3.0,
                 build_margin=0.5,
                 history=HISTORY, n_future=K_MAX, frameskip=FRAMESKIP):
        super().__init__(model, danger_head, action_std, history, n_future, frameskip)
        self.w_danger = float(w_danger)
        self.w_goal = float(w_goal)
        self.w_build = float(w_build)
        self.w_alt = float(w_alt)
        self.w_yaw = float(w_yaw)
        self.scale_danger = float(scale_danger)
        self.scale_goal = float(scale_goal)
        self.scale_build = float(scale_build)
        self.build_margin = float(build_margin)
        self.register_buffer(
            "kin_scale",
            torch.tensor([self.MAX_SPEED, self.MAX_SPEED, self.MAX_SPEED, self.MAX_YAW_RATE]),
        )

    def _danger_term(self, info_dict, action_candidates):
        # Imagine danger with the candidate's YAW ZEROED. Real danger is heading-
        # independent (in_range & aimed & los are all turret-side/geometric), but the
        # WM's imagined danger is POV-dependent (it can only imagine a turret in
        # frame). Without this override the CEM 'closes its eyes' -- yaws the turret
        # out of frame to lower IMAGINED danger without lowering REAL danger. Zeroing
        # yaw in the imagination means only strafe/climb/forward -- which actually
        # move the drone and so also lower real danger -- can lower imagined danger;
        # the planner must detour, not look away. (The showcase signal already passes
        # fwd with yaw=0, so this is a no-op there; DangerPlanner/phantom untouched.)
        zero_yaw = action_candidates.clone()
        zero_yaw[..., 3] = 0.0
        return super()._danger_term(info_dict, zero_yaw)

    def _kinematic_terms(self, info_dict, action_candidates):
        # state (B,S,21): xyz=state[...,:3], yaw=state[...,20]
        # goal (B,S,3); obstacles (B,S,16,6) = [cx,cy,cz,hx,hy,hz]
        state = info_dict["state"]
        goal = info_dict["goal"]
        obstacles = info_dict["obstacles"]
        pos = state[..., :3].clone()               # (B,S,3)
        yaw = state[..., 20].clone()               # (B,S)
        # env execution chain: candidate(normalized) -> *action_std -> clamp[-1,1]
        # (the action the env receives) -> *MAX_SPEED (process_actions) -> velocity.
        a_exec = (action_candidates * self.action_std).clamp(-1.0, 1.0)  # (B,S,K,4)
        a = a_exec * self.kin_scale                # (B,S,K,4) physical velocity
        K = self.n_future
        poss = []
        for k in range(K):
            vx_b, vy_b, vz, yr = a[..., k, 0], a[..., k, 1], a[..., k, 2], a[..., k, 3]
            for _ in range(self.frameskip):        # 5 sub-steps per window (0.5 s)
                cy, sy = torch.cos(yaw), torch.sin(yaw)
                wx = vx_b * cy - vy_b * sy         # Rz(yaw) body->world (matches apply_actions)
                wy = vx_b * sy + vy_b * cy
                pos = pos + torch.stack([wx, wy, vz], dim=-1) * self.STEP_DT
                yaw = yaw + yr * self.STEP_DT      # yaw updates AFTER the position step
                poss.append(pos)
        poss = torch.stack(poss, dim=-2)           # (B,S,K*FS,3)
        # goal: closest 2D-xy approach over the horizon (z held by the alt term)
        gxy = goal[..., :2]                         # (B,S,2)
        d_goal = torch.norm(poss[..., :2] - gxy.unsqueeze(-2), dim=-1)  # (B,S,T)
        goal_cost = d_goal.amin(dim=-1)             # (B,S)
        # building: max over sub-steps & boxes of softplus(margin - outside_dist)
        # outside_dist = ||max(|pos-center| - half, 0)||  (0 inside the AABB, +outside)
        centers = obstacles[..., :3]                # (B,S,16,3)
        halves = obstacles[..., 3:6]                # (B,S,16,3)
        diff = poss.unsqueeze(-2) - centers.unsqueeze(-3)            # (B,S,T,16,3)
        outside = torch.clamp(diff.abs() - halves.unsqueeze(-3), min=0.0)  # (B,S,T,16,3)
        d_out = torch.norm(outside, dim=-1)         # (B,S,T,16)
        pen = F.softplus(self.build_margin - d_out) # (B,S,T,16) ~0 far, rises inside/near
        build_cost = pen.amax(dim=(-1, -2))         # (B,S) worst moment / worst box
        # altitude hold: mean (z - DRONE_Z)^2 over the horizon (blocks the climb-over escape)
        alt_cost = ((poss[..., 2] - self.DRONE_Z) ** 2).mean(dim=-1)  # (B,S)
        # yaw-rate penalty: mean |executed yaw_rate| over the horizon. Pins the drone to
        # its approach heading so it keeps the turret in the POV (the WM can only imagine
        # a turret it can see). Damps the early-approach yaw drift that, left unchecked,
        # rotates the turret out of frame before the drone closes to firing range. The
        # zero-yaw imagination (_danger_term above) already removes the close-range
        # 'close your eyes' gaming incentive; this just keeps the heading honest so the
        # WM actually gets a dead-ahead view to fire on (as the phantom showcase does).
        yaw_cost = a_exec[..., :, 3].abs().mean(dim=-1)              # (B,S)
        return goal_cost, build_cost, alt_cost, yaw_cost

    def get_cost(self, info_dict, action_candidates):
        danger_term = 0.0
        if self.w_danger > 0:                       # skip the WM rollout when oblivious
            danger = self._danger_term(info_dict, action_candidates)  # (B,S)
            danger_term = self.w_danger * (danger / self.scale_danger)
        goal_cost, build_cost, alt_cost, yaw_cost = self._kinematic_terms(info_dict, action_candidates)
        return (danger_term
                + self.w_goal * (goal_cost / self.scale_goal)
                + self.w_build * (build_cost / self.scale_build)
                + self.w_alt * alt_cost
                + self.w_yaw * yaw_cost)


def imagined_danger(model, head, px_ctx, ctx_act2, future_act20_4, dev, history=HISTORY,
                    n_future=K_MAX):
    """Imagined max-danger of a dataset-unit 4-action plan (for baselines).

    px_ctx: (3,C,H,W); ctx_act2: (2,20) recorded; future_act20_4: (4,20) the plan.
    """
    with torch.inference_mode():
        px_ctx = px_ctx.unsqueeze(0).unsqueeze(0).to(dev)  # (1,1,3,C,H,W)
        act_seq = torch.cat([ctx_act2.to(dev), future_act20_4.to(dev)], dim=0).unsqueeze(0).unsqueeze(0)  # (1,1,6,20)
        pred = model.rollout({'pixels': px_ctx}, act_seq, history_size=history)['predicted_emb']
        future = pred[0, 0, history : history + n_future, :]  # (4,192)
        return head(future).squeeze(-1).max().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=None)
    ap.add_argument('--dataset', default='uav_isaac_train.lance')
    ap.add_argument('--n_approach', type=int, default=50)
    ap.add_argument('--n_safe', type=int, default=50)
    ap.add_argument('--num_samples', type=int, default=128)
    ap.add_argument('--n_steps', type=int, default=8)
    ap.add_argument('--topk', type=int, default=16)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    ckpt = args.ckpt or latest_ckpt()
    print(f'[CEM] ckpt={ckpt}  plan: ctx={CTX_ACT_LEN} recorded + {K_MAX} sampled -> T={T_PLAN}')
    model = load_pretrained(ckpt).eval().to(dev)
    for p in model.parameters():
        p.requires_grad_(False)
    head = load_danger_head(dev)

    num_steps = HISTORY + K_MAX  # 7 (load 7 frames; use first 3 as context)
    ds = swm.data.load_dataset(
        args.dataset, num_steps=num_steps, frameskip=5,
        keys_to_load=['pixels', 'action', 'state', 'shot'],
    )
    ds.transform = img_preprocessor()
    action_std = compute_action_std(ds).to(dev)
    print(f'[CEM] action_std (4-dim)={[round(x, 4) for x in action_std.cpu().tolist()]}')

    # split val (last 20%), stratify into approach / safe
    n = len(ds)
    split = int(n * 0.8)
    rng = random.Random(args.seed)
    val_idx = list(range(split, n))
    rng.shuffle(val_idx)

    approach_idx, safe_idx = [], []
    for i in val_idx:
        st = ds[i]['state']  # (7,21)
        present = float(st[HISTORY - 1, DANGER_IDX])
        future = float(st[HISTORY : HISTORY + K_MAX, DANGER_IDX].max())
        if present < 0.5 and future > 0.5 and len(approach_idx) < args.n_approach:
            approach_idx.append(i)
        elif future < 0.5 and len(safe_idx) < args.n_safe:
            safe_idx.append(i)
        if len(approach_idx) >= args.n_approach and len(safe_idx) >= args.n_safe:
            break
    print(f'[CEM] eval windows: approach={len(approach_idx)} safe={len(safe_idx)}')

    planner = DangerPlanner(model, head, action_std).to(dev)
    action_space = gym_spaces.Box(low=-1.0, high=1.0, shape=(1, RAW_ACTION_DIM), dtype=np.float32)
    cfg = PlanConfig(horizon=K_MAX, receding_horizon=1, action_block=1)
    solver = CEMSolver(
        cost=planner, num_samples=args.num_samples, n_steps=args.n_steps,
        topk=args.topk, device=dev, seed=args.seed,
    )
    solver.configure(action_space=action_space, n_envs=1, config=cfg)

    def eval_window(i):
        s = ds[i]
        px = s['pixels']  # (7,C,H,W)
        act = s['action']  # (7,20)
        ctx_act = act[:CTX_ACT_LEN]  # (2,20) recorded act[t-2],act[t-1]
        recorded_future = act[CTX_ACT_LEN : CTX_ACT_LEN + K_MAX]  # (4,20) act[t..t+3]
        px_ctx = px[:HISTORY]  # (3,C,H,W)
        info = {
            'pixels': px_ctx.unsqueeze(0).to(dev),          # (1,3,C,H,W)
            'context_action': ctx_act.unsqueeze(0).to(dev),  # (1,2,20)
        }
        out = solver.solve(info)
        plan_norm = out['actions'][0].to(dev)  # (K_MAX,4) normalized (solver returns CPU)
        plan20 = (plan_norm * action_std).repeat(1, FRAMESKIP)  # (K_MAX,20) dataset units
        d_plan = imagined_danger(model, head, px_ctx, ctx_act, plan20, dev)
        d_rec = imagined_danger(model, head, px_ctx, ctx_act, recorded_future, dev)
        d_zero = imagined_danger(model, head, px_ctx, ctx_act, torch.zeros(K_MAX, ACT_DIM), dev)
        rnd = torch.randn(K_MAX, RAW_ACTION_DIM) * action_std.cpu()
        d_rand = imagined_danger(model, head, px_ctx, ctx_act, rnd.repeat(1, FRAMESKIP), dev)
        return {
            'plan': d_plan, 'recorded': d_rec, 'zero': d_zero, 'random': d_rand,
            'first_action_norm': plan_norm[0].cpu().tolist(),
        }

    def aggregate(idxs, tag):
        if not idxs:
            return
        R = [eval_window(i) for i in idxs]
        mean = lambda k: float(np.mean([r[k] for r in R]))
        frac = lambda a, b: float(np.mean([1.0 if r[a] < r[b] else 0.0 for r in R]))
        print(f'[CEM] ---- {tag} (n={len(R)}) ----')
        print(f'[CEM]   mean imagined-danger:  planner={mean("plan"):.3f}  '
              f'recorded={mean("recorded"):.3f}  zero={mean("zero"):.3f}  random={mean("random"):.3f}')
        print(f'[CEM]   planner < recorded: {frac("plan", "recorded"):.2f} | '
              f'planner < zero: {frac("plan", "zero"):.2f} | planner < random: {frac("plan", "random"):.2f}')
        print(f'[CEM]   |first action| L2 mean={float(np.mean([np.linalg.norm(r["first_action_norm"]) for r in R])):.3f}')
        for r in R[:3]:
            print(f'[CEM]   ex: plan={r["plan"]:.3f} recorded={r["recorded"]:.3f} '
                  f'zero={r["zero"]:.3f} rand={r["random"]:.3f} | act0={[round(x, 2) for x in r["first_action_norm"]]}')

    print('[CEM] ============ RESULT (offline, circular) ============')
    aggregate(approach_idx, 'APPROACH (present safe, future danger)')
    aggregate(safe_idx, 'SAFE (no future danger)')
    print('[CEM] GATE: on APPROACH, planner imagined-danger < recorded (the action that '
          'actually flew into danger) AND < random => planner avoids imagined danger => Phase 7 live demo.')


if __name__ == '__main__':
    main()

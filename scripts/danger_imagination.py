#!/usr/bin/env python
"""Phase 5: imagination vs detection for FUTURE danger (the core demo claim).

CLAIM: a single-frame detector sees a turret but can't tell it will become
dangerous; a world model that imagines the future can.

For each anchor frame t (currently SAFE: danger[t]=0, no shot in window), predict
whether danger appears within k steps (k=1..4, ~0.5-2s at frameskip=5). Compare:
  - DETECTOR: linear probe on the PRESENT latent emb_actual[:,t,:] -> future danger.
    (What a detector can do: it only sees the current frame.)
  - IMAGINATION: linear probe on the IMAGINED future latent rollout->emb[t+k] -> future danger.
    (What the WM adds: it rolls the dynamics forward via the action-conditioned predictor.)
  - ACTION control: linear probe on the present action -> future danger.
    (Rules out "imagination wins only via action-correlation": if the WM's edge is just
    the action leaking imminent-danger info, action-alone matches it.)

Dedicated linear probes per (method, k) -- standard protocol, each method gets its
best readout. GATE: imagination AUROC > detector AUROC (consistently across k) =>
the WM encodes danger-relevant DYNAMICS, not just appearance => Phase 6 (CEM
planner) proceeds. Imagination ~= detector => the latent encodes danger but the
predictor isn't forecasting it (investigate: more epochs / longer horizon / the
predictor collapsed to identity).

Runs in the HOST swm training venv on the GPU box, from the
repo root (``stable_worldmodel`` is installed editable). NOT
the Isaac container (no swm there).

  python scripts/danger_imagination.py
  python scripts/danger_imagination.py --ckpt lewm/weights_epoch_25.pt --n_windows 40000

API (verified, see memory uav-wm-project-state.md):
  - load_pretrained(name) -> LeWM (.eval().to('cuda') yourself; freeze params)
  - model.encode({'pixels':x})['emb']                                 -> (B,T,192) actual per-frame latents
  - model.rollout({'pixels':x_ctx}, act_seq, history_size=3)['predicted_emb']
                                                                      -> (B,1,T+1,192) imagined trajectory
  - danger = state[:, t, 13]; frameskip=5 -> 1 step = 0.5s; episode-contiguous in lance
"""
import argparse
import random

import numpy as np
import stable_worldmodel as swm
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Subset

from stable_pretraining import data as spt_data
from stable_worldmodel.wm.utils import load_pretrained

DANGER_IDX = 13
HISTORY = 3  # wm.history_size (lewm.yaml); predictor num_frames
K_MAX = 4  # imagine up to 4 steps (~2s) ahead


def img_preprocessor(img_size: int = 224):
    tr = spt_data.transforms
    return tr.Compose(
        tr.ToImage(
            **spt_data.dataset_stats.ImageNet, source='pixels', target='pixels'
        ),
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


def train_probe(Xtr, ytr, Xva, yva, dev, epochs=20, lr=1e-3, batch_size=256):
    """Linear probe + BCE(pos_weight) for the 2.4% imbalance; returns val logits."""
    probe = nn.Linear(Xtr.shape[1], 1).to(dev)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    pos_weight = torch.tensor([(1 - ytr.mean()) / max(ytr.mean(), 1e-6)], device=dev)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    for _ in range(epochs):
        probe.train()
        perm = torch.randperm(Xtr.shape[0])
        for i in range(0, Xtr.shape[0], batch_size):
            b = perm[i : i + batch_size]
            xb, yb = Xtr[b].to(dev), ytr[b].to(dev)
            opt.zero_grad()
            loss = lossf(probe(xb).squeeze(-1), yb)
            loss.backward()
            opt.step()
    probe.eval()
    with torch.no_grad():
        return probe(Xva.to(dev)).squeeze(-1).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=None, help='lewm/weights_epoch_N.pt (default: latest)')
    ap.add_argument('--dataset', default='uav_isaac_train.lance')
    ap.add_argument('--n_windows', type=int, default=30000)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    num_steps = HISTORY + K_MAX  # 7: context(3) + future(K_MAX)
    ckpt = args.ckpt or latest_ckpt()
    print(f'[IMAG] ckpt={ckpt} num_steps={num_steps} K_MAX={K_MAX} (1 step = 0.5s)')
    model = load_pretrained(ckpt).eval().to(dev)
    for p in model.parameters():
        p.requires_grad_(False)

    ds = swm.data.load_dataset(
        args.dataset,
        num_steps=num_steps,
        frameskip=5,
        keys_to_load=['pixels', 'action', 'state', 'shot'],
    )
    ds.transform = img_preprocessor()
    n = len(ds)
    print(f'[IMAG] dataset={args.dataset} windows={n}')

    # contiguous-region split: train = first 80%, val = last 20% (near-episode-clean)
    rng = random.Random(args.seed)
    split = int(n * 0.8)
    n_tr = min(int(args.n_windows * 0.8), split)
    n_va = min(args.n_windows - n_tr, n - split)
    tr_idx = rng.sample(range(0, split), n_tr)
    va_idx = rng.sample(range(split, n), n_va)

    def collect(indices, tag):
        loader = DataLoader(
            Subset(ds, indices), batch_size=128, num_workers=8, shuffle=False
        )
        present_lats, act_now = [], []
        imagined = [[] for _ in range(K_MAX)]
        danger_all, shot_all = [], []
        with torch.inference_mode():
            for bi, batch in enumerate(loader):
                px = batch['pixels'].to(dev)  # (B,7,C,H,W)
                act = batch['action'].to(dev)  # (B,7,20)
                st = batch['state']  # (B,7,21)
                shot = batch['shot']  # (B,7,1)
                emb = model.encode({'pixels': px})['emb']  # (B,7,192)
                # imagination: rollout context[:3] + all actions forward
                px_ctx = px[:, :HISTORY].unsqueeze(1)  # (B,1,3,C,H,W)
                act_seq = act.unsqueeze(1)  # (B,1,7,20)
                pred = model.rollout(
                    {'pixels': px_ctx}, act_seq, history_size=HISTORY
                )['predicted_emb']  # (B,1,8,192); imagined[t+k] at index 2+k
                present_lats.append(emb[:, HISTORY - 1, :].float().cpu())  # frame t (idx 2)
                act_now.append(act[:, HISTORY - 1, :].float().cpu())  # present action
                for k in range(1, K_MAX + 1):
                    imagined[k - 1].append(
                        pred[:, 0, HISTORY - 1 + k, :].float().cpu()
                    )  # index 2+k
                danger_all.append(st[:, :, DANGER_IDX].float().cpu())  # (B,7)
                shot_all.append(shot[..., 0].float().cpu())  # (B,7)
                if bi % 20 == 0:
                    print(
                        f'[IMAG]   {tag} batch {bi} emb={tuple(emb.shape)} '
                        f'pred={tuple(pred.shape)}'
                    )
        P = torch.cat(present_lats)
        A = torch.cat(act_now)
        I = [torch.cat(imagined[k]) for k in range(K_MAX)]
        D = torch.cat(danger_all)  # (N,7)
        Sh = torch.cat(shot_all)  # (N,7)
        return P, A, I, D, Sh

    print(f'[IMAG] extracting train ({n_tr})...')
    Ptr, Atr, Itr, Dtr, Shtr = collect(tr_idx, 'train')
    print(f'[IMAG] extracting val ({n_va})...')
    Pva, Ava, Iva, Dva, Shva = collect(va_idx, 'val')

    # present danger = frame t (idx 2); future label(k) = max danger over frames 3..2+k
    present_danger = lambda D: D[:, HISTORY - 1]
    future_label = lambda D, k: D[:, HISTORY : HISTORY + k].max(dim=1).values
    # early-warning subset: present safe + no shot anywhere in the window (clean physics)
    clean_mask = lambda D, Sh: (present_danger(D) < 0.5) & (Sh.sum(dim=1) < 0.5)

    cm_tr = clean_mask(Dtr, Shtr)
    cm_va = clean_mask(Dva, Shva)
    print(
        f'[IMAG] early-warning subset: train={int(cm_tr.sum())}/{len(cm_tr)} | '
        f'val={int(cm_va.sum())}/{len(cm_va)}'
    )

    def auroc_ap(y, s):
        return roc_auc_score(y, s), average_precision_score(y, s)

    print(
        '[IMAG] ============ RESULT (early-warning: danger[t]=0, no-shot window) ============'
    )
    print(f'[IMAG] {"k":>2} {"method":<12} {"AUROC":>7} {"AP":>7} {"prev":>7}')
    rows = []
    for k in range(1, K_MAX + 1):
        ytr = future_label(Dtr, k)[cm_tr]
        yva = future_label(Dva, k)[cm_va]
        prev = float(yva.mean())
        for name, Xtr, Xva in [
            ('detector', Ptr, Pva),
            ('imagination', Itr[k - 1], Iva[k - 1]),
            ('action', Atr, Ava),
        ]:
            s = train_probe(Xtr[cm_tr], ytr, Xva[cm_va], yva, dev, args.epochs)
            au, aps = auroc_ap(yva.numpy(), s)
            rows.append((k, name, au, aps, prev))
            print(f'[IMAG] {k:>2} {name:<12} {au:7.4f} {aps:7.4f} {prev:7.4f}')

    det = {k: au for k, m, au, ap, p in rows if m == 'detector'}
    ima = {k: au for k, m, au, ap, p in rows if m == 'imagination'}
    act = {k: au for k, m, au, ap, p in rows if m == 'action'}
    wins_det = sum(1 for k in range(1, K_MAX + 1) if ima[k] > det[k])
    wins_act = sum(1 for k in range(1, K_MAX + 1) if ima[k] > act[k])
    print(
        f'[IMAG] imagination > detector at {wins_det}/{K_MAX} horizons; '
        f'imagination > action at {wins_act}/{K_MAX} horizons'
    )
    print(
        f'[IMAG] GATE: {"PASS" if wins_det >= 3 else "WEAK" if wins_det >= 2 else "FAIL"} '
        f'(imagination forecasts danger beyond the current frame -> Phase 6 CEM planner)'
    )

    out = (
        swm.data.utils.get_cache_dir(sub_folder='checkpoints')
        / 'lewm'
        / 'danger_imagination.pt'
    )
    torch.save({'rows': rows, 'ckpt': ckpt, 'det': det, 'ima': ima, 'act': act}, out)
    print(f'[IMAG] saved -> {out}')


if __name__ == '__main__':
    main()

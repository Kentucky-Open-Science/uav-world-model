#!/usr/bin/env python
"""Phase 6: train the danger readout the CEM planner uses as its cost.

A single linear head nn.Linear(192,1) that reads danger from an IMAGINED future
latent (LeWM.rollout's predicted_emb), pooled over horizons k=1..4. This is a
per-frame danger detector on imagined latents -- distinct from Phase 5's
per-(method,k) early-warning probes (which predicted future-danger-from-a-safe-
anchor). Here the label is danger AT the imagined frame (state[:,13] at t+k).

Trained on imagined latents from RECORDED actions; the planner applies it to
imagined latents from SAMPLED actions. Both live in the same pred_proj space, so
the readout transfers (Phase 5 already showed imagined latents are probe-readable).

  python scripts/danger_head.py

API: model.rollout({'pixels':px_ctx}, act_seq, history_size=3)['predicted_emb']
  px_ctx (B,1,3,C,H,W), act_seq (B,1,7,20) -> (B,1,8,192); imagined t+k at idx 2+k.
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
HISTORY = 3  # wm.history_size
K_MAX = 4  # imagine up to 4 steps (~2s) ahead


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=None)
    ap.add_argument('--dataset', default='uav_isaac_train.lance')
    ap.add_argument('--n_windows', type=int, default=30000)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    num_steps = HISTORY + K_MAX  # 7
    ckpt = args.ckpt or latest_ckpt()
    print(f'[HEAD] ckpt={ckpt} num_steps={num_steps} K_MAX={K_MAX} (1 step = 0.5s)')
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
    print(f'[HEAD] dataset={args.dataset} windows={n}')

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
        imagined, danger, keep = [], [], []
        with torch.inference_mode():
            for bi, batch in enumerate(loader):
                px = batch['pixels'].to(dev)  # (B,7,C,H,W)
                act = batch['action'].to(dev)  # (B,7,20)
                st = batch['state']  # (B,7,21)
                shot = batch['shot']  # (B,7,1)
                px_ctx = px[:, :HISTORY].unsqueeze(1)  # (B,1,3,C,H,W)
                act_seq = act.unsqueeze(1)  # (B,1,7,20)
                pred = model.rollout(
                    {'pixels': px_ctx}, act_seq, history_size=HISTORY
                )['predicted_emb']  # (B,1,8,192)
                # clean physics: drop windows with any shot (fall) frame
                clean = shot[..., 0].sum(dim=1) < 0.5  # (B,)
                for k in range(1, K_MAX + 1):
                    imagined.append(pred[:, 0, HISTORY - 1 + k, :].float().cpu())
                    danger.append(st[:, HISTORY - 1 + k, DANGER_IDX].float().cpu())
                    keep.append(clean)
                if bi % 20 == 0:
                    print(f'[HEAD]   {tag} batch {bi} pred={tuple(pred.shape)}')
        X = torch.cat(imagined)  # (N*K, 192)
        y = torch.cat(danger)  # (N*K,)
        m = torch.cat(keep)  # (N*K,)
        return X[m], y[m]

    print(f'[HEAD] extracting train ({n_tr})...')
    Xtr, ytr = collect(tr_idx, 'train')
    print(f'[HEAD] extracting val ({n_va})...')
    Xva, yva = collect(va_idx, 'val')
    print(
        f'[HEAD] train={Xtr.shape[0]} danger={int(ytr.sum())} ({ytr.mean():.4f}) | '
        f'val={Xva.shape[0]} danger={int(yva.sum())} ({yva.mean():.4f})'
    )

    head = nn.Linear(Xtr.shape[1], 1).to(dev)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    pos_weight = torch.tensor([(1 - ytr.mean()) / max(ytr.mean(), 1e-6)], device=dev)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    bs = 512
    for ep in range(args.epochs):
        head.train()
        perm = torch.randperm(Xtr.shape[0])
        for i in range(0, Xtr.shape[0], bs):
            b = perm[i : i + bs]
            xb, yb = Xtr[b].to(dev), ytr[b].to(dev)
            opt.zero_grad()
            loss = lossf(head(xb).squeeze(-1), yb)
            loss.backward()
            opt.step()
        head.eval()
        with torch.no_grad():
            s = head(Xva.to(dev)).squeeze(-1).cpu().numpy()
        if (ep + 1) % 5 == 0 or ep == 0:
            print(
                f'[HEAD] epoch {ep + 1}/{args.epochs} val_auroc='
                f'{roc_auc_score(yva.numpy(), s):.4f} val_ap='
                f'{average_precision_score(yva.numpy(), s):.4f}'
            )
    with torch.no_grad():
        s = head(Xva.to(dev)).squeeze(-1).cpu().numpy()
    au = roc_auc_score(yva.numpy(), s)
    aps = average_precision_score(yva.numpy(), s)
    print('[HEAD] ============ RESULT ============')
    print(f'[HEAD] val_auroc={au:.4f} val_ap={aps:.4f} prev={yva.mean():.4f}')
    print(
        f'[HEAD] GATE: {"PASS" if au > 0.65 else "WEAK" if au > 0.55 else "FAIL"} '
        f'(imagined-latent danger readout usable by the planner)'
    )

    out = swm.data.utils.get_cache_dir(sub_folder='checkpoints') / 'lewm' / 'danger_head.pt'
    torch.save(
        {'state_dict': head.state_dict(), 'in_dim': Xtr.shape[1],
         'val_auroc': au, 'val_ap': aps, 'ckpt': ckpt},
        out,
    )
    print(f'[HEAD] saved -> {out}')


if __name__ == '__main__':
    main()

#!/usr/bin/env python
"""Phase 4: danger head on frozen LeWM latents (uav3d).

GATE: does the trained latent ENCODE danger? A linear probe on frozen per-frame
latents (pixels-only ``emb[:,0,:]``, D=192) predicts current danger (state[13]).
AUROC >> 0.5 => latent is danger-aware => Phase 5 (imagination-vs-detection demo)
proceeds. AUROC ~= 0.5 => latent failed to encode danger (investigate: more
epochs / sparse 2.4% signal / collapse).

Runs in the HOST swm training venv on the GPU box, from the
repo root (``stable_worldmodel`` is installed editable). NOT
the Isaac container (no swm there).

  python scripts/danger_probe.py
  python scripts/danger_probe.py --ckpt lewm/weights_epoch_25.pt --n_frames 40000

Loads the latest surviving ``weights_epoch_*.pt`` by default (pruning keeps
last-3, so the final epoch is always available).

Split: contiguous frame-index regions (train = first 80% of the lance, val =
last 20%). Episodes are contiguous in the lance, so this is near-episode-clean
(<=1 boundary episode straddles) WITHOUT needing an episode_idx column (the
Stage-2 writer does not expose one). Class imbalance (2.4% danger): report AUROC
(imbalance-robust) + average-precision + prevalence + accuracy + best-F1 point.

API map (verified, see memory uav-wm-project-state.md):
  - load_pretrained(name)           -> LeWM, must .eval().to('cuda') yourself
  - model.encode({'pixels':x[:,None]})['emb'][:,0,:]  -> (N,192), PIXELS-ONLY
  - load_dataset(name, num_steps=1, frameskip=1)      -> per-frame (T=1)
  - danger = state[0,13]   (uav3d.yaml: state[DANGER_IDX]==danger)
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


def img_preprocessor(img_size: int = 224):
    # mirror lewm.py: `transforms` is a submodule of `data`, not top-level
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=None, help='lewm/weights_epoch_N.pt (default: latest)')
    ap.add_argument('--dataset', default='uav_isaac_train.lance')
    ap.add_argument('--n_frames', type=int, default=30000)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch_size', type=int, default=256)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    # --- load frozen model ---
    ckpt = args.ckpt or latest_ckpt()
    print(f'[PROBE] loading {ckpt}')
    model = load_pretrained(ckpt).eval().to(dev)
    for p in model.parameters():
        p.requires_grad_(False)

    # --- dataset (per-frame) ---
    ds = swm.data.load_dataset(
        args.dataset,
        num_steps=1,
        frameskip=1,
        keys_to_load=['pixels', 'state'],
    )
    ds.transform = img_preprocessor()
    n = len(ds)
    print(f'[PROBE] dataset={args.dataset} frames={n}')

    # --- contiguous-region split: train from first 80%, val from last 20% ---
    rng = random.Random(args.seed)
    split = int(n * 0.8)
    n_tr = min(int(args.n_frames * 0.8), split)
    n_va = min(args.n_frames - n_tr, n - split)
    tr_idx = rng.sample(range(0, split), n_tr)
    va_idx = rng.sample(range(split, n), n_va)

    def collect(indices, tag):
        loader = DataLoader(
            Subset(ds, indices), batch_size=args.batch_size, num_workers=8, shuffle=False
        )
        lats, labs = [], []
        with torch.inference_mode():
            for bi, batch in enumerate(loader):
                px = batch['pixels'].to(dev)            # (B,1,C,H,W)
                st = batch['state']                      # (B,1,21)
                emb = model.encode({'pixels': px})['emb'][:, 0, :]  # (B,192)
                lats.append(emb.float().cpu())
                labs.append(st[:, 0, DANGER_IDX].cpu())
                if bi % 20 == 0:
                    print(f'[PROBE]   {tag} batch {bi} emb={tuple(emb.shape)}')
        return torch.cat(lats), torch.cat(labs).float()

    print(f'[PROBE] extracting train latents ({n_tr} frames)...')
    Xtr, ytr = collect(tr_idx, 'train')
    print(f'[PROBE] extracting val latents ({n_va} frames)...')
    Xva, yva = collect(va_idx, 'val')
    print(
        f'[PROBE] train={Xtr.shape[0]} danger={int(ytr.sum())} ({100*ytr.mean():.2f}%) | '
        f'val={Xva.shape[0]} danger={int(yva.sum())} ({100*yva.mean():.2f}%)'
    )

    # --- linear probe (pos_weight for the 2.4% imbalance) ---
    probe = nn.Linear(Xtr.shape[1], 1).to(dev)
    opt = torch.optim.Adam(probe.parameters(), lr=args.lr)
    pos_weight = torch.tensor([(1 - ytr.mean()) / max(ytr.mean(), 1e-6)], device=dev)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    for ep in range(args.epochs):
        probe.train()
        perm = torch.randperm(Xtr.shape[0])
        for i in range(0, Xtr.shape[0], args.batch_size):
            b = perm[i : i + args.batch_size]
            xb, yb = Xtr[b].to(dev), ytr[b].to(dev)
            opt.zero_grad()
            loss = lossf(probe(xb).squeeze(-1), yb)
            loss.backward()
            opt.step()
        probe.eval()
        with torch.no_grad():
            sc = probe(Xva.to(dev)).squeeze(-1).cpu().numpy()
        if (ep + 1) % 5 == 0 or ep == 0:
            print(
                f'[PROBE] epoch {ep+1}/{args.epochs} val_auroc='
                f'{roc_auc_score(yva.numpy(), sc):.4f}'
            )

    # --- final report ---
    with torch.no_grad():
        sc = probe(Xva.to(dev)).squeeze(-1).cpu().numpy()
    yv = yva.numpy()
    auroc = roc_auc_score(yv, sc)
    ap = average_precision_score(yv, sc)
    acc = float(((sc > 0).astype(np.float32) == yv).mean())
    # best-F1 operating point
    order = np.argsort(-sc)
    ys = yv[order]
    tp = np.cumsum(ys); fp = np.cumsum(1 - ys); P = ys.sum()
    prec = tp / np.maximum(tp + fp, 1); rec = tp / max(P, 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    bi = int(f1.argmax())
    print('[PROBE] ============ RESULT ============')
    print(
        f'[PROBE] val_auroc={auroc:.4f}  val_ap={ap:.4f}  '
        f'val_acc={acc:.4f}  val_prevalence={yv.mean():.4f}'
    )
    print(
        f'[PROBE] best-F1: f1={f1[bi]:.4f} prec={prec[bi]:.4f} rec={rec[bi]:.4f} '
        f'(thr={sc[order[bi]]:.4f})'
    )
    print(
        f'[PROBE] GATE: {"PASS" if auroc > 0.7 else "WEAK" if auroc > 0.6 else "FAIL"} '
        f'(>0.7 latent encodes danger -> Phase 5; 0.6-0.7 borderline; <0.6 investigate)'
    )

    out = (
        swm.data.utils.get_cache_dir(sub_folder='checkpoints')
        / 'lewm'
        / 'danger_probe.pt'
    )
    torch.save(
        {'state_dict': probe.state_dict(), 'val_auroc': auroc, 'val_ap': ap},
        out,
    )
    print(f'[PROBE] saved probe -> {out}')


if __name__ == '__main__':
    main()

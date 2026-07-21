# Phase 4 — Danger probe (does the latent encode danger?)

**Status: PASS.** Linear probe on frozen per-frame latents → current-danger AUROC **0.776** (>0.7 gate). The LeWM encoder, trained with no danger label, represents danger-relevant appearance in its latent.

## Setup
- Checkpoint: `lewm/weights_epoch_25.pt` (final; val `pred_loss` 0.0058, sigreg 0.79).
- Per-frame latent: `model.encode({'pixels': x[:,None]})['emb'][:,0,:]` → `(N,192)`. Pixels-only (state never enters `encode`).
- Probe: `nn.Linear(192,1)` + `BCEWithLogitsLoss(pos_weight)` for the 2.4% imbalance, Adam lr 1e-3, 20 epochs, batch 256.
- Split: contiguous-region (train = first 80% / val = last 20%), near-episode-clean.
- Label: `state[:,13]` (`danger = in_range & in_fov & los & aimed`).
- 24,000 train frames (577 danger, 2.40%) / 6,000 val (161 danger, 2.68%).

## Result
| metric | value |
|---|---|
| val AUROC | 0.7758 |
| val AP | 0.1385 (prev 2.68% → ~5× random) |
| val accuracy | 0.7330 |
| best-F1 | 0.257 (prec 0.213, rec 0.323, thr 1.47) |

AUROC climbed 0.703 (ep 1) → 0.776 (ep 20), monotonic, no overfit.

## Interpretation
The latent encodes danger despite never being told the label — the JEPA objective (predict-next-latent + SIGReg anti-collapse) yields a representation where "is the drone about to be shot" is linearly decodable. AUROC 0.776 is solid for a 2.4%-prevalence signal from an unsupervised representation. Best-F1 (0.26) is low purely because of the prevalence; AUROC (threshold-free) is the gate metric.

## Gate
**PASS** (>0.7). → Phase 5: does the WM *imagine future* danger better than a single-frame detector can see it?

## Artifacts
- `scripts/danger_probe.py`, `scripts/danger_probe_smoke.py`
- `checkpoints/lewm/danger_probe.pt` (probe logits + metrics)

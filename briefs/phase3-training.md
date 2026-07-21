# Phase 3 — LeWM training on 3D drone POV

**Status:** training running (epoch 11/25 at last check); collect + lance + sanity + smoke all done. This brief documents the setup and the convergence trend; final-loss + Phase-4 gate filled in on completion.

## What was trained
LeWM (vendored `stable_worldmodel`) — a JEPA world model: a ViT encoder maps each drone-POV frame to a 192-d latent, and an action-conditioned transformer predictor learns to predict the *next frame's latent* from 3 context frames + their actions. Loss is latent-MSE + SIGReg (anti-collapse). Pixels-only encoding (state never enters encode/predict); `state` is cached only for Phase-4 danger labeling.

## Dataset (uav_isaac_train.lance)
Collected in NVIDIA Isaac Sim 2.3.2 on the GPU box (2× RTX A6000), drone first-person POV, domain-randomized urban scene.

| metric | value |
|---|---|
| episodes | 2015 |
| frames | 530,191 |
| danger frames | 12,654 (2.39%) |
| killed | 474 (23.5%) |
| timeout | 1541 (76.5%) |
| crash / reached_goal | 0 / 0 |
| lance size | 3.1 GB, 491,906 windows |
| `get_dim('action')` | 4 → `action_encoder.input_dim = 20` (frameskip 5) |
| `get_dim('state')` | 21, `shot` mask present (0.75%) |

Episode mix: 40% no-turret, 40% waypoint+turret, 20% drunk (survivable-danger provoker). Evan's 3 changes applied: random per-episode cruise speed, random spawn, goal removed (fixed street-grid paths).

## Config
- 25 epochs, single A6000 (`trainer.devices=1`), batch 128, bf16, lr 5e-5 cosine-annealed (auto-rescaled to 25 epochs).
- `wm.history_size=3, num_preds=1, num_steps=4, frameskip=5` (1 step = 0.5 s).
- ViT-tiny (192-d, 12 layers), 6-layer predictor. ~5 M params.
- Checkpoint pruning `keep_last=3` (final survivors epochs 23/24/25; `weights_epoch_25.pt` always available for Phase 4).
- Fall-frame masking: `pred_loss` zeroed on windows containing a `shot`/falling frame (actions decouple from physics there).
- Disk-safe: <20 GB abort guard + <15 GB kill watcher + 10 h `timeout`. wandb off (local log only).

**Epoch budget rationale:** measured ~3.2 it/s → ~18.4 min/epoch → 25 epochs ≈ 7.6 h, fits the 10 h timeout with margin AND lets the cosine schedule complete cleanly (100-epoch default would hit the timeout at ~epoch 30 with an unannealed LR).

## Convergence trend (val `pred_loss`)
| epoch | val pred_loss | sigreg |
|---|---|---|
| 0 | 0.0424 | 1.53 |
| 1 | 0.0245 | 1.34 |
| 2 | 0.0201 | 1.17 |
| 3 | 0.0175 | 1.09 |
| 4 | 0.0141 | 1.06 |
| 5 | 0.0134 | 0.95 |
| … | (descending, decelerating toward floor) | … |

−68% over the first 5 epochs, monotonic, val≈train (no overfit), all finite. SIGReg settled ~0.9 (latent not collapsing). Throughput locked at ~18.4 min/epoch.

## Interpretation
The latent is converging on a useful representation: prediction loss is descending cleanly and the anti-collapse term has settled, so the encoder isn't taking the trivial shortcut. The fall-frame masking keeps the predictor from learning bogus dynamics during crashes. This is the expected JEPA convergence trajectory.

## What's next
- **Phase 4** (danger probe): linear probe on frozen per-frame latents `emb[:,0,:]` → current-danger AUROC. GATE: >0.7 ⇒ latent encodes danger ⇒ proceed; <0.6 ⇒ investigate (more epochs / sparse 2.4% signal).
- On `TRAIN DONE exit=0`: confirm `weights_epoch_25.pt` + `config.json`, then run `danger_probe.py`.

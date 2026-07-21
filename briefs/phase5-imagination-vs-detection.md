# Phase 5 — Imagination vs detection (the core demo claim)

**Status: PASS.** The world model's *imagined-future* latent forecasts danger better than the present-frame latent, at all 4 horizons (0.5–2 s). The edge is dynamics, not action-leak: imagination beats action-alone at 3/4 horizons, losing only at the shortest where action→danger is nearly direct.

## The claim
A single-frame detector sees a turret but can't tell it will become dangerous; a world model that imagines the future can. Test: for each anchor frame t that is currently SAFE (`danger[t]=0`, no shot in window — the "early-warning" clean-physics subset), predict whether danger appears within k steps (k=1..4, ~0.5–2 s at `frameskip=5`). Three methods, each its own linear probe:
- **DETECTOR** — present latent `emb[:,t,:]` (what a detector sees: current frame only).
- **IMAGINATION** — imagined future latent `rollout→emb[t+k]` (what the WM adds: action-conditioned dynamics rolled forward).
- **ACTION** — present action (control: rules out "imagination wins only via action-correlation").

## Result (val, early-warning subset: 5889/6000 windows)
| k (s) | detector | imagination | action | prev |
|---|---|---|---|---|
| 1 (0.5) | 0.7240 | **0.7400** | 0.7730 | 0.006 |
| 2 (1.0) | 0.7795 | **0.8083** | 0.7341 | 0.014 |
| 3 (1.5) | 0.7844 | **0.8096** | 0.7209 | 0.020 |
| 4 (2.0) | 0.7787 | **0.8041** | 0.6978 | 0.025 |

- **Imagination > detector: 4/4 horizons** (gate ≥3/4 → PASS). Consistent +0.016 to +0.029 AUROC.
- **Imagination > action: 3/4 horizons.** Loss only at k=1.

## Interpretation
1. **The core claim holds.** Rolling the latent forward via the action-conditioned predictor yields a representation that forecasts future danger better than the present frame alone — at every horizon. The WM isn't just recognizing a turret; it's modeling that the turret *will become* dangerous.
2. **The edge is dynamics, not action-leak.** Action-alone wins at k=1 (0.5 s) — expected, since the imminent action (e.g. flying toward the turret) is nearly directly tied to imminent danger. But action *decays* with horizon (0.773→0.734→0.721→0.698) while imagination *holds* (0.740→0.808→0.810→0.804). The imagination latent compounds the action through predicted dynamics, sustaining forecast power where raw action can't. So the WM's edge is genuinely the rolled-forward dynamics, not the action encoder leaking a shortcut.
3. **Absolute imagination AUROC 0.74–0.81 across 0.5–2 s** — strong, usable signal for a planner.

## Gate
**PASS** → Phase 6 CEM planner: use the imagined-future latent (or a danger readout on it) as the cost in CEM action search, so the drone avoids trajectories the WM forecasts as dangerous.

## Artifacts
- `scripts/danger_imagination.py`, `scripts/danger_imagination_smoke.py`
- `checkpoints/lewm/danger_imagination.pt` (per-(k, method) AUROC/AP)

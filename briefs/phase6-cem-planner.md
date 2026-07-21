# Phase 6 — Danger-aware CEM planner (offline)

**Status: DONE (offline).** A CEM planner that rolls LeWM's imagination forward over candidate action sequences and minimizes the danger readout on the imagined trajectory. On held-out windows it finds actions the WM rates as less dangerous than the recorded action that actually flew — on 100% of approach-to-danger windows. The live gate is Phase 7 (the offline eval is circular: the WM is both imaginer and evaluator).

## Design
CEM searches **4-dim raw actions** (vx, vy, vz, yaw_rate) over a 4-step (~2 s) horizon — these are the t→t+1 … t+3→t+4 transitions. For each candidate sequence the `DangerPlanner` Costable:
1. scales the normalized 4-dim action to dataset units (× per-dim std), repeats each 5× → 20-dim (the frameskip-stack `action_encoder` expects);
2. prepends the **2 recorded context actions** (act[t-2], act[t-1]) → `(B,S,6,20)`. The first **sampled** action lands in `act_0[2]` = act[t], so it drives the t→t+1 prediction → **all** of t+1..t+4 are controllable;
3. rolls `LeWM.rollout` (H=3 context frames, T=6) → `predicted_emb (B,S,7,192)`;
4. applies the frozen danger head (Phase 6 readout) to imagined frames t+1..t+4 (indices 3..6);
5. **cost = max danger logit** over the horizon (avoid the worst imagined danger).

CEM (128 samples, 8 steps, top-16 elites) keeps the lowest-cost sequences; the planner returns the first 4-dim action (the t→t+1 transition = the next action to execute). Wired through swm's existing `CEMSolver` + `PlanConfig` — no solver code written, only the `DangerPlanner` Costable (a `nn.Module` with `get_cost`).

## Danger readout (Phase 6 head)
A single `nn.Linear(192,1)` trained on (imagined[t+k] latent, danger-at-t+k) pooled over k=1..4, on clean (no-shot) windows. **val AUROC 0.821** — stronger than the actual-latent probe (0.776), because the imagined latent is the model's belief about the future frame and the label is danger *at* that frame. Prevalence 1.6% (shot/fall windows filtered → pure approach-danger, what the planner should avoid). Saved to `checkpoints/lewm/danger_head.pt`.

## Result (50 approach + 50 safe held-out windows)
| subset | planner | recorded | zero | random | planner<recorded | planner<random |
|---|---|---|---|---|---|---|
| APPROACH (present safe, future danger) | **−0.068** | 0.948 | 0.768 | 0.862 | 50/50 | 50/50 |
| SAFE (no future danger) | **−1.648** | −0.696 | −0.885 | −0.733 | 50/50 | 50/50 |

Mean imagined danger (logits). On approach windows the planner drives danger from 0.948 (the recorded action, which actually flew into danger) to −0.068 — a ~1.0-logit reduction. Examples show the expected behavior: hard brake + turn on inescapable approach windows (act0=[−3.7, 0.7, 2.4, −3.5]).

## Controllability bug (found + fixed)
First version prepended **3** recorded context actions (including act[t], the t→t+1 transition), so imagined[t+1] was action-invariant and dominated the `max` → cost couldn't be optimized (plan == recorded == zero == random on many windows). Fix: prepend only **2** recorded context actions; the first sampled action becomes act[t], making all of t+1..t+4 controllable. After the fix, the cost is action-sensitive and the planner beats all baselines.

## Circularity + why it's still meaningful
The offline eval is circular: the WM imagines the danger *and* scores the planner's action. Beating random/zero is partly tautological (CEM minimizes that same signal by construction). The **less-circular** signal is `planner < recorded`: the recorded action *actually flew* (into danger, on approach windows), and the planner finds an action the WM rates as less dangerous. This is meaningful because the WM's imagined danger was **independently validated** in Phase 5 (AUROC 0.74–0.81 for predicting *real* future danger) — so the planner is optimizing a signal that genuinely tracks real danger, not an arbitrary cost. The **live demo (Phase 7)** is the real gate: fly the planner in Isaac and measure survival vs a baseline.

## Open items for Phase 7 (live)
- **Container/swm co-location**: the closed loop needs the Isaac env (container) + LeWM/planner (swm). Investigate importing `stable_worldmodel` into the container's Python (LeWM + planning stack are torch-only; `load_pretrained` needs hydra + the ViT backbone — check availability in-container).
- **Action clipping**: the planner's normalized actions reach L2≈5 (CEM mean drifts past the ±1 Box since plain CEM doesn't clip). Clip to the env's action bounds for execution.
- **Collision awareness**: the cost is turret-danger only — the planner may fly the drone into a building while evading. May need a control penalty or a soft obstacle cost; or accept crashes and report survival + crash split.
- **Baseline**: compare survival/kill-rate vs `WaypointPolicy` (flies fixed routes, no danger reaction) and vs a detector-based reactive policy.

## Artifacts
- `scripts/danger_head.py` (readout), `uav_wm/planning/cem_planner.py` (planner + offline eval)
- `checkpoints/lewm/danger_head.pt`, `checkpoints/lewm/danger_imagination.pt`

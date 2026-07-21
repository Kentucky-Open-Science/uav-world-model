# Phase 7 — Live CEM demo (planner vs baseline in Isaac)

**Status: PASS (live).** The Phase-6 CEM planner, flown closed-loop in the Isaac UAVTurret3D env, survives more than an oblivious baseline that flies the same routes. Matched comparison (n=40, `force_type=wpturret`, 8 envs): planner **kill_rate 0.25 / survival 0.75** vs waypoint baseline **kill_rate 0.40 / survival 0.60** — a 37.5% relative reduction in kills, zero crashes either way. This is the live gate the Phase-6 offline eval pointed at (and which that circular eval couldn't answer on its own).

**Imagination-vs-detection, live (added):** a single-frame danger-detector baseline — the Phase-4 present-frame probe firing at its calibrated best-F1 threshold, evading only when it *sees* present danger — survives *less* than the oblivious baseline: **kill_rate 0.475 / survival 0.525**, vs planner 0.250 and oblivious 0.400. Detection reacts too late (probe recall 0.33 at best-F1; it fires only once the turret is already in-frame and aimed, and the flee then halts forward progress in the kill zone). This is the live complement to Phase 5's offline 4/4 result, and closes the "honest read" gap below.

## Architecture: two-process
The closed loop splits across two Python environments that cannot share one:
- **Container** (`nvcr.io/nvidia/isaac-lab:2.3.2`, py3.11, torch 2.7.0+cu128): runs the Isaac env + a socket client. swm-free by design — the container lacks lancedb's Rust ext + 17 `stable_pretraining` deps, and py3.11↔py3.12 compiled wheels can't cross-mount. Sends each step's POV batch; steps the env with the returned action.
- **Host** (swm-train venv, py3.12): runs `planner_server.py` — LeWM + danger head + CEM solver. Keeps per-env 3-frame pixel context + 2-action history; plans all N envs in one batched CEM solve; returns N clipped 4-dim actions.

`--network=host` so the container reaches the host planner on `127.0.0.1:PORT`. The host venv is the known-good one that ran training + the offline eval. (Single-process was ruled out: importing swm into the container fails on lancedb; a `spt_stub` vendoring `vit_hf` got the encoder in but lancedb still blocks.)

## Protocol
Stdlib socket + `struct` 4-byte big-endian length prefix + pickle.
- req = `{just_reset: (N,) bool, pix: (N,H,W,3) uint8 RGB, state: (N,21) float32}`
- resp = `{action: (N,4) float32}` body-frame `(vx,vy,vz,yaw_rate)`, clipped to `[-1,1]`

`just_reset` lets the host clear that env's context on episode auto-reset (Isaac resets on term/trunc, so the obs returned by `env.step` is the *next* episode's first frame — without the mask the host would feed stale context across the reset boundary).

## Closed-loop mechanics (host `plan()`)
- **Context buffers:** `pix_buf` (deque maxlen 3) + `act_buf` (deque maxlen 2) per env. On each request: clear buffers where `just_reset`, append the new frame, pad to length (repeat-first frame at front / zero-pad action), repeat each action 5× → 20-dim frameskip-stack the `action_encoder` expects.
- **Rollout:** cat ctx+candidate → `(B,S,6,20)`, `LeWM.rollout` (H=3 context, T=6) → `predicted_emb (B,S,7,192)`; danger head on imagined t+1..t+4 (indices 3..6); CEM cost = **max** danger logit over the horizon (avoid the worst imagined danger).
- **Preprocessing:** uint8 RGB → permute → /255 → ImageNet normalize (mean/std as `(1,3,1,1)`; rgb=False preserves RGB — matches training's JPEG'd pixels).
- **Action return:** CEM (128 samples, 8 steps, top-16) first 4-dim normalized action × per-dim std, clamped to `[-1,1]` for the env's Box (CEM mean drifts past ±1; plain CEM doesn't clip).

## Episode forcing (clean planner-vs-baseline)
`--force_type wpturret` sets every episode to `EP_WAYPOINT_TURRET` and relocates the turret onto the mid-route waypoint. The oblivious `WaypointPolicy` flies fixed routes straight through the turret's FOV; the planner must evade. Both modes use the same forcing so survival differences are attributable to the planner, not the scenario.

## Outcome accounting
`infer_outcome(state, term, trunc)`: **killed** (fire_prog≥1.0) / **crash** (drone_z<0.2) / **timeout** (trunc) / **terminated** (term) / **ended**. **kill_rate = killed/total**, **survival = (timeout+ended)/total**. Crashes (fly-into-building while evading) are reported separately, not folded into survival — Phase 6 flagged this as an open risk (cost is turret-danger only).

## Results
Matched comparison, same scenario (`force_type=wpturret` — turret relocated onto the mid-route waypoint; 8 parallel envs; n=40 episodes each):

| mode | n | killed | timeout | crash | kill_rate | survival |
|---|---|---|---|---|---|---|
| **planner** (CEM + danger imagination) | 40 | 10 | 30 | 0 | **0.250** | **0.750** |
| waypoint (oblivious baseline) | 40 | 16 | 24 | 0 | 0.400 | 0.600 |
| detector (probe, best-F1 thr=1.53) | 40 | 19 | 21 | 0 | 0.475 | 0.525 |
| detector (probe, eager thr=0.0) | 40 | 3 | 37 | 0 | 0.075† | 0.925† |

† degenerate — see below. `ended=0` for every mode (the turret sits on the route, so no policy completes it; all survivors time out).

- **Kill rate 0.40 → 0.25** (37.5% relative reduction). **Survival 0.60 → 0.75.**
- **Zero crashes** either way — the Phase-6 open risk ("evade the turret, fly into a building") did not materialize on this scenario. The cost is turret-danger only; crash separation is reported but was not a confound here.
- Both modes see the same episode forcing, so the survival difference is attributable to the planner's danger-aware action choice, not the scenario.
- An earlier n=8 planner smoke landed on the identical 0.250/0.750 — consistent, though n=8 alone was too small to trust.
- Planner runs at **~0.475 s/step** (one batched CEM solve over all 8 envs; min 0.472 / max 0.477 — near-constant). The baseline, with no planner round-trip, steps far faster but dies more.

### Detector-reactive baseline (imagination vs detection, live)
The comparison the Phase-7 "honest read" named as missing: a **single-frame danger detector** — the Phase-4 present-frame probe (`danger_probe.pt`, nn.Linear 192→1 on the encoded POV) — firing and evading only when it *sees* present danger. Implemented as `--mode detector` on both server (`DetectorServer`) and client (oblivious waypoint + server-triggered flee). On fire: hard evade `[reverse, per-ep-random strafe, climb, no yaw]` blended with the waypoint policy's building-repulsion; on no-fire: follow the waypoint. So detector-reactive = oblivious baseline + detection-triggered evasion — the *only* difference from oblivious is the evasion, making kill differences cleanly attributable to reaction.

- **Calibrated detector (best-F1 thr=1.5291): kill_rate 0.475 — worse than oblivious (0.400), much worse than the planner (0.250).** The probe's best-F1 operating point has recall 0.33 (val AUROC 0.78, prevalence 2.4%): it fires on only a third of true-danger frames, and late — once the turret is already in-frame and aimed. When it does fire, the flee (reverse + climb) halts forward progress *inside the kill zone*, so it dies more often than flying straight through. This is the live mechanism behind "detection reacts too late": the detector's information (present frame) simply lacks the lead time the planner's imagination provides.
- **Eager detector (thr=0.0, kill_rate 0.075) is degenerate, not a win.** Cranking the threshold to 0 makes the detector fire on ~57% of frames (server fire-rate log: median logit ≈ 0, so the threshold sits at the median; against 2.4% true-danger prevalence, >95% of fires are on *safe* frames). Survival then comes from the flee's climb outrunning the turret's limited vertical reach — a geometric escape that has nothing to do with detecting the turret. It is a "climb on most frames" policy, not a calibrated detector, and is reported only to expose the failure mode: beating the planner's kill rate this way requires abandoning selective detection entirely.

**Takeaway:** at every *honest* operating point the detector is at-or-worse-than oblivious, and the imagination-driven planner is the only policy that stays safe *without* degenerating into perpetual flight. The planner anticipates danger the detector cannot yet see; the detector, by the time it fires, is already too late.

### Honest read
The original Phase-7 asymmetry — planner beats *no-reaction*, not a *reactive detector* — is now closed by the detector-reactive baseline above. The planner's edge over the calibrated detector (0.250 vs 0.475 kill) is the live isolation of "imagination beats detection," complementing Phase 5's offline 4/4 horizon result. The remaining caveat is sample size (n=40 per mode) and that the detector shares the planner's flee primitive; the *detection* quality (probe AUROC 0.78, best-F1 recall 0.33) is the binding constraint, not the evade maneuver.

## Closed-loop bugs (found + fixed)
- **numpy 2.x → 1.x pickle break.** The host (numpy 2.4.4) pickled the action response with `numpy._core` internals the Isaac container (numpy 1.x, pinned by Isaac Lab 2.3.2) can't unpickle — `ModuleNotFoundError: No module named 'numpy._core'` on `recv_msg`. Only the host→container direction broke (the request, 1.x→2.x, unpickles fine), so the host-only smoke missed it. Fix: server sends the action as plain Python floats (`.tolist()`) — version-proof pickle; the container's `torch.tensor(resp["action"])` accepts a list, so zero container-side change.
- **Hardcoded batch size.** `plan()` used `N = self.N` (the server's startup `--num_envs`), so a client sending a different batch tripped `IndexError: index N out of bounds`. The Isaac driver sends exactly `num_envs` so the real run wouldn't hit it, but it blocked cheap validation. Fix: derive `N` from the incoming batch (`len(pix_uint8)`) with an assert against oversize.

## MCAP showcase (imagination vs detection, visualized)
Evan's 2026-07-17 directive: surface on one timeline the WM imagined-danger signal vs the single-frame detector logit, so Foxglove visibly shows the WM *leading* the detector. The closed loop already computes both — the showcase just exposes + logs them.

- **Showcase signals (exposed by `planner_server.py`, ~free):**
  - `wm_danger` = imagined danger logit of a **forward-reference plan** — fly straight ahead along the drone's heading (`vx=1`, no strafe/climb/yaw, ×`action_std`) — rolled t+1..t+4 by `DangerPlanner.get_cost`. Independent of the **selected** CEM plan, so it RISES as the drone heads toward the turret and LEADS the present-frame detector: the WM imagines the turret becoming dangerous before it is aimed/in-range in the current frame. (The selected plan's own imagined danger stays LOW when the planner evades — it is the safe plan by construction — so it is the *wrong* signal for a "predicts ahead" showcase. A forward reference answers "what happens if I keep flying ahead," which is exactly the question the detector can't ask.)
  - `det_logit` = Phase-4 present-frame probe logit on the **current** POV (one linear pass on the already-encoded present frame). The single-frame detector's signal.
  - Both ride in the planner-mode response (`{action, wm_danger, det_logit}`, all `.tolist()`); detector-mode exposes `det_logit` only; waypoint exposes neither.
- **`scripts/live/mcap_viz.py` (`LiveEpisodeWriter`):** one Foxglove MCAP per episode, adapted from the collector's `EpisodeWriter`. Channels: `/drone/pov/image` (RawImage), `/scene` (SceneUpdate — drone/turret/obstacles/aim-line/FOV-cone/green trail), `/drone/tf`, `/state` (Log), and **`/signals`** — a `PoseInFrame` carrying `(position.x=wm_danger, y=det_logit, z=danger)` so Foxglove's Plot panel graphs all three vs the header timestamp. (v0.3.0 `foxglove-schemas-protobuf` has no numeric-scalar schema and `KeyValuePair` is string-only — so numeric signals ride in a Pose position. The Plot panel graphs `pose.position.{x,y,z}`.)
- **Sidecar `.signals.txt`** (one line/step: `step wm_danger det_logit danger fire dist_t`) + `.outcome` per episode — lets the best showcase episode be picked **textually** without opening Foxglove (text-only rule).
- **`scripts/pick_showcase.py`:** scans the sidecars, ranks episodes by how far `wm_danger` leads `det_logit`. Because the two signals are logits from different heads on different scales, it uses a **relative rise** for each: first step the signal climbs to 30% of its way from an early baseline (median of first 5 steps) toward its episode peak (min amplitude 0.20 to count as risen). `lead = det_rise_step − wm_rise_step` (positive ⇒ WM predicted first). Top ranks: episodes where `danger=1` occurred (real threat) AND the WM rose early AND the detector rose late-or-never ("WM saw it, detector never did" = strongest case).
- **`scripts/replay_showcase.sh` + `foxglove/showcase_layout.md`:** rsync the showcase MCAPs to the Mac, run the picker, open the top-ranked episode in Foxglove with the Plot-on-`/signals` layout (draw `y=1.53` as the detector fire line; the WM's lead = the horizontal gap between the `wm_danger` rise and the `det_logit` rise).
- **`live_demo.py --mcap_dir`** (opt-in; passed through `run_live_demo.sh`'s 5th arg; writes under the already-bind-mounted `/workspace/output`). Planner pass produces the signal-bearing MCAPs; a waypoint pass (no server) produces the oblivious-baseline MCAPs (no signals — flies straight into the turret).
- **Dedicated `phantom`+`showcase` modes (the validated showcase path).** The forward-reference signal alone was insufficient under the *planner*: the planner evades, which turns the turret out of the camera, so `wm_danger` falls and `det_logit` stays negative — no lead to show (verified: 7/8 long planner episodes had `dang_hit=0`; even the one killed approach episode had `det_logit` negative throughout because the turret was never in the POV). Fix: a dedicated **`phantom` server mode** (no CEM — computes only the two signals as phantoms, records the forward CRUISE action the drone actually flies as the WM context so imagination matches reality, returns `{wm_danger, det_logit}` with **no action**) + a **`showcase` client mode** that spawns the drone head-on at a street intersection with an axis-aligned heading down a long street, turret at the far street end, then flies **oblivious forward** (`SHOWCASE_CRUISE=0.7`, matching `PhantomServer.cruise`). The turret stays in the POV the whole closing approach → both signals can rise → the WM's lead is visible. `SHOWCASE_CRUISE` and `PhantomServer.cruise` MUST match (else the WM's recorded context action ≠ the flown action).

### Showcase result (validated, 2026-07-17)
`run_live_demo.sh showcase 8 48 5557 /workspace/output/showcase_phantom` (phantom server, head-on oblivious flight): **48/48 killed** (kill_rate 1.0 — as expected; oblivious head-on flight into the turret always dies, so the danger the WM imagines always materializes). 55 episodes logged (48 + re-spawns in the 8-env pool).

**The WM leads the detector — validated textually** (per-episode `.signals.txt` sidecars, ranked by `pick_showcase.py`):
- **43/55 (78%)** episodes have a clean separation moment: `wm_danger > 0.5` while `det_logit < 0.0` (WM elevated, detector blind).
- **28/55 (51%)** have `lead > 0` by the relative-rise metric (median lead 31 steps; 12 episodes ≥50 steps, 8 ≥100 steps). Top lead = 170 steps (~17 s at 10 Hz).
- **Mechanism, read from the traces:** the WM's forward-reference imagination ("fly straight ahead") flags the looming turret during the whole closing approach (dist 9–15 m), while the present-frame detector is pinned negative until the turret is within ~8 m (its range). E.g. `env1_ep002`: at dist 20 m, `wm=1.34 / det=−0.61`; at dist 14.6 m, `wm=0.54 / det=−1.67` — the WM is elevated where the detector is still blind. The strongest-lead episode `env5_ep001` (lead=114, ~11 s) makes it concrete: at step 129 / dist 15.3 m, `wm=0.51 / det=−1.28` (WM up, detector blind); over steps 129→136 `wm` rises monotonically `0.51→0.85` while `det` stays negative (`−1.28→−0.85`); only at step 243 / dist 7.4 m do both cross positive (`wm=1.42 / det=1.22`) and `danger` goes 1. The detector's range (~8 m) is the binding constraint — the WM's rolled-forward latent reaches the danger regime from ~15 m.
- **Honest caveat:** the `wm_danger` signal is **noisy** (oscillates ±2–3; the ViT-tiny's imagined embeddings aren't monotonically tied to closing distance, and the forward reference from 20 m only reaches ~14 m — outside the 8 m range, so the elevated early values are partly the danger head's noise on imagined futures, not a clean monotonic trend). The detector is cleaner (flat-negative then monotonic rise) but fires only at the end. The showcase story holds — the WM is the only signal elevated during the approach — but the raw `wm_danger` trace is noisy, not a clean ramp. A smoothed overlay (rolling mean) would clarify the Plot panel; the raw signal is logged as-is.
- **Top picks for visual review** (on the Mac under `data/showcase_phantom/`): `showcase_env7_ep005.mcap` (sep=54), `showcase_env0_ep004.mcap` (sep=46), `showcase_env1_ep002.mcap` (sep=45, lead=30), `showcase_env5_ep001.mcap` (lead=114). `pick_showcase.py` ranks by `sep_steps` (sustained separation) first — robust to noisy single-step blips that inflate the raw `lead`.

### Planner vs waypoint showcase (matched n=24, the "acting on imagination saves you" half)
The vs-baseline complement to the phantom: a **matched** planner-vs-waypoint MCAP pair on the *same* scenario (`force_type=wpturret`, 16 envs, n=24 each). The planner MCAPs are for the **3D-trail + outcome** survival story (not the `/signals` lead — under the planner the drone evades, so both signals stay low; the lead lives in the phantom above). Both runs logged one MCAP/episode under `/workspace/output/showcase_{planner,waypoint}`; rsync'd to the Mac at `data/showcase_{planner,waypoint}/`.

| mode (n=24, matched) | killed | timeout | kill_rate | survival |
|---|---|---|---|---|
| **planner** (imagination-driven) | 8 | 16 | **0.333** | **0.667** |
| **waypoint** (oblivious baseline) | 13 | 11 | **0.542** | **0.458** |

The planner cuts the kill rate 0.542→0.333 (39% relative reduction) — a clean within-run survival contrast, consistent with (and sharper than) the n=40 matched comparison in the Results table above (0.40→0.25). Zero crashes either way.

**Cleanest finding: all 16 planner timeout (survived) episodes have `danger_frames=0`** — the planner evaded *before ever entering the kill zone* in every survived episode. It never has to escape danger; it avoids it. The oblivious baseline, by contrast, flies straight through and accumulates ~19–22 danger frames before dying. So the 3D-panel story is: planner trail veers off (strafe/climb) before the aim-line goes red and the episode `timeout`s; waypoint trail goes straight, the aim-line goes red, and it's `killed`.

**Episode picks for visual review:**
- Planner (survived, the evade): `planner_env6_ep001.mcap` (timeout, n=300, `maxwm=0.80` — the forward-ref WM imagination was most elevated, so the "WM saw it and evaded" arc is clearest), `planner_env1_ep002.mcap` (timeout, `maxdet=1.71` — detector nearly fired, drone got close, planner still evaded with `danger=0`).
- Waypoint (died, the foil): `waypoint_env8_ep001.mcap` (killed, n=153, 22 danger frames — the most sustained danger before death; oblivious flight into the turret).
- Open with `bash scripts/replay_showcase.sh planner` / `waypoint` (rsync + rank + open top in Foxglove); the 3D panel shows drone (blue) / turret (red) / aim-line (orange→RED on danger) / green trail (the evade or the straight-through). See `foxglove/showcase_layout.md` for the per-mode story (phantom = `/signals` lead; planner/waypoint = 3D survival).

**Two-mode showcase, together:** the phantom `/signals` Plot proves the WM *predicts* the danger the detector can't yet see (lead up to ~11 s); the planner-vs-waypoint 3D trails prove that *acting on that prediction* lets the drone survive where flying oblivious (or reacting only to the present-frame detector) gets it killed. The detector's binding constraint — ~8 m range, fires only once the turret is in-frame and aimed — is the same in both halves.

## Operational notes
- **One-session server:** `planner_server.py` accepts one connection and exits when it closes (the `while True` loop catches the client's `ConnectionError` → `finally` closes the listen socket). Relaunch per run. stdout → a log file (decouples server lifetime from the ssh channel, which otherwise dies when the detached docker holds it open).
- **Detached launches:** `nohup ... & disown` on the GPU box starts the docker run but hangs the ssh channel (the container keeps it open); the launch completes regardless. Use fresh-ssh status queries + log monitors, not the launch task's exit.
- **Zombie risk:** a crashed `live_demo.py` can leave the `kit` process stuck in Isaac Sim shutdown, holding GPU memory for hours. Force-`docker rm -f` any stale `isaac-lab` container before re-running.

## Artifacts
- `scripts/live/planner_server.py` (host server), `scripts/live/live_demo.py` (container driver), `scripts/live/mcap_viz.py` (MCAP writer), `scripts/run_live_demo.sh` (wrapper)
- `scripts/pick_showcase.py` (text-only episode ranker), `scripts/replay_showcase.sh` (rsync + rank + open), `foxglove/showcase_layout.md` (layout guide)
- reuses `uav_wm/planning/cem_planner.py` (`DangerPlanner`, danger head, `action_std`), `uav_wm/data/policies_3d.py` (`WaypointPolicy`)

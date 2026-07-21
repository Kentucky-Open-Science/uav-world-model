#!/usr/bin/env python3
"""Forward-model unit test (Validation step 1 of the nav plan).

Replays RECORDED dataset actions through the NavPlanner kinematic forward model
(the 20-sub-step ideal-velocity integrator) and compares predicted vs recorded
drone xyz at t+1..t+4. Validates three things the host planner_server cannot
import the env to check:
  * integration sign + Rz(yaw) body->world rotation (matches apply_actions)
  * yaw-update ORDER (position uses start-of-sub-step heading; yaw updates after)
  * frameskip (4 windows x 5 sub-steps = 20 sub-steps = 2.0 s; NOT 4)

Expected (per the plan's Risk 1): predicted LEADS real -- the real drone tracks
vel_des via a PD cascade (~0.5 m/s^2), so recorded positions lag the ideal-
velocity model. Same direction + same OoM + lead fraction > 0.5 => the model is
honest. A sign flip or 10x error => integration/yaw/frameskip bug.

Also reports whether the 5 sub-actions inside each 20-dim dataset action are
HELD (identical, std~0) or VARIED -- tells us whether NavPlanner's held-action
assumption (cand.repeat(frameskip)) matches the collector.

Runs on the box (swm-train venv: torch + the lance dataset). No GPU, no sim.
Usage: python scripts/test_forward_model.py [n_windows]
"""
import sys
import numpy as np
import torch

import stable_worldmodel as swm

# uav_turret_3d.py scalars (must match NavPlanner + the env)
MAX_SPEED = 2.0
MAX_YAW_RATE = 2.0
STEP_DT = 0.1
FRAMESKIP = 5
K_MAX = 4
HISTORY = 3                      # context frames t-2,t-1,t ; present = frame index 2

KIN_SCALE = np.array([MAX_SPEED, MAX_SPEED, MAX_SPEED, MAX_YAW_RATE], dtype=np.float64)


def integrate(state, action):
    """state (7,21), action (7,20) -> predicted xyz (4,3) at t+1..t+4, starting
    from the present frame (index HISTORY-1 = 2). Uses the ACTUAL 5 env sub-actions
    per window (action[k].reshape(5,4)), not the held approximation."""
    pos = state[HISTORY - 1, :3].astype(np.float64).copy()
    yaw = float(state[HISTORY - 1, 20])
    pred = np.zeros((K_MAX, 3), dtype=np.float64)
    for j in range(K_MAX):
        sub = action[HISTORY - 1 + j].reshape(FRAMESKIP, 4).astype(np.float64)  # (5,4) env units
        for s in range(FRAMESKIP):
            a = sub[s] * KIN_SCALE                        # actual per-sub-step env action
            cy, sy = np.cos(yaw), np.sin(yaw)
            wx = a[0] * cy - a[1] * sy
            wy = a[0] * sy + a[1] * cy
            pos = pos + np.array([wx, wy, a[2]]) * STEP_DT
            yaw = yaw + a[3] * STEP_DT
        pred[j] = pos
    return pred


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    ds = swm.data.load_dataset(
        "uav_isaac_train.lance", num_steps=HISTORY + K_MAX, frameskip=FRAMESKIP,
        keys_to_load=["pixels", "action", "state", "shot"],
    )
    rng = np.random.default_rng(0)
    idxs = rng.choice(len(ds), size=min(n, len(ds)), replace=False)

    errs = np.zeros((len(idxs), K_MAX, 3))      # predicted - recorded, per step
    leads = np.zeros((len(idxs), K_MAX))        # 1 if err in motion direction (lead)
    held_std = []                               # std of the 5 sub-actions (0 = held)
    for wi, i in enumerate(idxs):
        d = ds[int(i)]
        st = d["state"]                          # (7,21) numpy or tensor
        ac = d["action"]                         # (7,20)
        st = st.numpy() if hasattr(st, "numpy") else np.asarray(st)
        ac = ac.numpy() if hasattr(ac, "numpy") else np.asarray(ac)
        pred = integrate(st, ac)
        for j in range(K_MAX):
            rec = st[HISTORY + j, :3].astype(np.float64)     # recorded xyz at t+1+j
            err = pred[j] - rec
            errs[wi, j] = err
            motion = rec - st[HISTORY + j - 1, :3].astype(np.float64)  # real displacement
            nrm = np.linalg.norm(motion)
            if nrm > 1e-6:
                leads[wi, j] = 1.0 if np.dot(err, motion) > 0 else 0.0
        held_std.append(float(ac.reshape(-1, FRAMESKIP, 4).std(axis=1).mean()))

    print(f"=== forward-model unit test ({len(idxs)} windows) ===")
    print(f"sub-actions: mean std across the 5 repeats = {np.mean(held_std):.4f} "
          f"(0 = HELD/identical -> NavPlanner cand.repeat matches; >0 = VARIED per env-step)")
    print(f"{'step':4} {'mean|err| (m)':14} {'mean|dx|':9} {'mean|dy|':9} {'mean|dz|':9} "
          f"{'lead%':6}")
    for j in range(K_MAX):
        ae = np.abs(errs[:, j, :])
        lead = leads[:, j].mean() * 100
        print(f"t+{j+1:<2} {ae.mean():14.3f} {ae[:,0].mean():9.3f} {ae[:,1].mean():9.3f} "
              f"{ae[:,2].mean():9.3f} {lead:6.0f}")
    # overall direction check: is the mean error vector small + lead fraction > 50%?
    mean_err = errs.mean(axis=(0, 1))
    overall_lead = leads.mean() * 100
    print(f"\nmean err vector (m): [{mean_err[0]:+.3f}, {mean_err[1]:+.3f}, {mean_err[2]:+.3f}]")
    print(f"overall lead fraction: {overall_lead:.0f}%  (>50% => predicted leads real = PD lag, as expected)")
    ok = overall_lead > 50 and np.abs(errs[:, :, :2]).mean() < 2.0
    print(f"\nVERDICT: {'PASS (integration/yaw/frameskip honest; predicted leads real)' if ok else 'REVIEW (sign flip, 10x error, or no lead)'}")


if __name__ == "__main__":
    main()

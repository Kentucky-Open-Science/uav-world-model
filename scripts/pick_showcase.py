#!/usr/bin/env python
"""Text-only showcase-episode selector for the live-demo MCAPs.

Scans the .signals.txt sidecars written by mcap_viz.LiveEpisodeWriter and ranks
episodes by how far the world-model danger signal LEADS the single-frame detector
logit -- the quantity the showcase is built to show. No image inspection (per the
text-only rule): Evan opens the top-ranked .mcap in Foxglove for the visual.

Sidecar format (one line/step, '#' header):
  # step wm_danger det_logit danger fire dist_t
  0 -1.59 -1.15 0 0.00 8.50
  ...

The wm_danger and det_logit are logits from different heads on different scales, so
absolute thresholds aren't comparable. Instead we use a RELATIVE rise for each: the
first step the signal climbs to 30% of its way from an early baseline (median of the
first 5 steps -- drone is far/safe) toward its episode peak. The LEAD = det_rise_step
- wm_rise_step (positive => the WM's signal rose first => it predicted ahead).

A good showcase episode needs BOTH signals to rise meaningfully (else there's no
"predicted ahead of" to show) AND a real threat to materialize (danger=1). Episodes
where the WM rose but the detector never rose ("WM saw it, detector never did") are
the strongest case and ranked just behind positive-lead episodes.

Usage:  python scripts/pick_showcase.py [signals_dir] [n]
  signals_dir defaults to data/showcase_planner (after replay_showcase.sh rsyncs here)
  n defaults to 10 (print top-n)
"""
import os
import sys

RISE_FRAC = 0.30     # signal "rose" once it's 30% of the way baseline->peak
MIN_AMP = 0.20       # min (peak-baseline) for a signal to count as having risen


def parse(path):
    steps, wm, det, dang = [], [], [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = line.split()
            if len(p) < 5:
                continue
            steps.append(int(p[0]))
            wm.append(float(p[1]))
            det.append(float(p[2]))
            dang.append(int(p[3]))
    return steps, wm, det, dang


def median(x):
    s = sorted(x)
    return s[len(s) // 2] if s else 0.0


def rise_step(vals, baseline, peak):
    """First step where vals exceed baseline + RISE_FRAC*(peak-baseline)."""
    if peak - baseline < MIN_AMP:
        return None  # flat signal -- no rise to speak of
    thr = baseline + RISE_FRAC * (peak - baseline)
    for i, v in enumerate(vals):
        if v > thr:
            return i
    return None


def score(path, stem):
    steps, wm, det, dang = parse(path)
    if not steps:
        return None
    outcome = "?"
    oc = os.path.join(os.path.dirname(path), stem + ".outcome")
    if os.path.exists(oc):
        outcome = open(oc).read().strip()
    n = len(steps)
    bl_wm = median(wm[:5])
    bl_det = median(det[:5])
    pk_wm = max(wm)
    pk_det = max(det)
    wm_rise = rise_step(wm, bl_wm, pk_wm)
    det_rise = rise_step(det, bl_det, pk_det)
    wm_amp = pk_wm - bl_wm
    det_amp = pk_det - bl_det
    danger_hit = 1 in dang
    danger_step = steps[dang.index(1)] if danger_hit else None
    if wm_rise is None or det_rise is None:
        lead = None
        tag = "wm-flat" if wm_rise is None else "det-flat"
    else:
        lead = det_rise - wm_rise
        if lead > 0:
            tag = "lead"
        elif lead == 0:
            tag = "tie"
        else:
            tag = "det-first"
    # peak-step lead (secondary, threshold-free): did the WM peak before the det?
    wm_peak_step = wm.index(pk_wm)
    det_peak_step = det.index(pk_det)
    peak_lead = det_peak_step - wm_peak_step
    # SUSTAINED separation (the cleanest showcase metric): steps where the WM is
    # clearly elevated (wm>0.5) while the detector is still blind (det<0), counted
    # only up to the danger step (after that the threat is real, not "predicted").
    # A noisy step-0 blip can fake a big `lead`; sep_steps rewards episodes where
    # the WM stays up while the detector stays flat -- the visual Evan wants.
    sep_thr_wm = 0.5
    sep_thr_det = 0.0
    cutoff = danger_step if danger_step is not None else n
    sep_steps = sum(
        1 for i in range(min(cutoff, n))
        if wm[i] > sep_thr_wm and det[i] < sep_thr_det
    )
    return {
        "stem": stem, "outcome": outcome, "n": n,
        "wm_rise": wm_rise, "det_rise": det_rise, "lead": lead,
        "wm_amp": wm_amp, "det_amp": det_amp, "tag": tag,
        "danger_hit": danger_hit, "danger_step": danger_step,
        "wm_peak": pk_wm, "det_peak": pk_det,
        "wm_peak_step": wm_peak_step, "det_peak_step": det_peak_step,
        "peak_lead": peak_lead,
        "sep_steps": sep_steps,
    }


def _suffix(stem):
    """'planner_env0_ep000' -> 'env0_ep000' (strip the mode prefix so the three
    nav controllers -- planner_/nav_/detector_ -- match on the shared env+ep)."""
    for pre in ("planner_", "nav_", "detector_", "waypoint_", "showcase_", "phantom_"):
        if stem.startswith(pre):
            return stem[len(pre):]
    return stem


def rank_phantom(d, n):
    rows = []
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".signals.txt"):
            continue
        r = score(os.path.join(d, fn), fn[:-len(".signals.txt")])
        if r:
            rows.append(r)

    # rank: (1) danger hit (threat materialized -- a real "predicted ahead" case);
    # (2) sustained separation steps (WM elevated while detector blind, before
    # danger) -- the cleanest visual, robust to noisy single-step blips; (3) lead>0
    # and its magnitude; (4) peak lead; flat-signal episodes last.
    def key(r):
        if r["lead"] is None:
            return (0, r["sep_steps"], 0, 0, 0, 0)
        return (1 if r["danger_hit"] else 0,
                r["sep_steps"],
                1 if r["lead"] > 0 else 0,
                r["lead"],
                1 if r["peak_lead"] > 0 else 0,
                r["wm_amp"])
    rows.sort(key=key, reverse=True)

    print(f"# showcase ranking ({len(rows)} episodes) dir={d}")
    print(f"# rise-frac={RISE_FRAC} min-amp={MIN_AMP}  "
          f"lead = det_rise_step - wm_rise_step (positive => WM predicted first)")
    print(f"# sep = steps WM>0.5 while det<0 (before danger) -- the sustained-lead visual")
    print(f"# {'rk':3} {'episode':28} {'out':9} {'n':4} {'sep':4} {'wmR':4} {'detR':5} "
          f"{'lead':5} {'pkL':4} {'dng':4} {'wmAmp':6} {'detAmp':7} tag")
    for i, r in enumerate(rows[:n]):
        wr = "-" if r["wm_rise"] is None else str(r["wm_rise"])
        dr = "-" if r["det_rise"] is None else str(r["det_rise"])
        ld = "-" if r["lead"] is None else str(r["lead"])
        pl = str(r["peak_lead"])
        dn = "-" if r["danger_step"] is None else str(r["danger_step"])
        print(f"{i+1:3d} {r['stem']:28} {r['outcome']:9} {r['n']:4d} {r['sep_steps']:4d} {wr:>4} "
              f"{dr:>5} {ld:>5} {pl:>4} {dn:>4} {r['wm_amp']:6.2f} "
              f"{r['det_amp']:7.2f} {r['tag']}")
    if rows:
        top = rows[0]
        mcap = os.path.join(d, top["stem"] + ".mcap")
        print(f"\n# top pick: {mcap}")
        print(f"#   {top['stem']} outcome={top['outcome']} "
              f"sep_steps={top['sep_steps']} wm_rise={top['wm_rise']} "
              f"det_rise={top['det_rise']} lead={top['lead']} "
              f"peak_lead={top['peak_lead']} danger_step={top['danger_step']} "
              f"tag={top['tag']}")


def rank_nav(planner_dir, oblivious_dir, detector_dir, n):
    """Rank episode indices for the A->B head-on hold-back story (option C). Matched-n:
    the same env+episode index is run under all three controllers, so we join the three
    .signals.txt/.outcome sidecars on the env_ep suffix and rank by the ideal trio --
    planner SURVIVES (holds back at ~9 m, never entering the 8 m kill zone), oblivious +
    detector fly in and die, planner outlasts both (outlast = planner_steps -
    oblivious_steps). Text-only: Evan confirms the hold-back geometry visually on the
    top-ranked trio."""
    def outcomes(d):
        out = {}
        if not os.path.isdir(d):
            return out
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".signals.txt"):
                continue
            stem = fn[:-len(".signals.txt")]
            r = score(os.path.join(d, fn), stem)
            if r:
                out[_suffix(stem)] = r
        return out
    P = outcomes(planner_dir); O = outcomes(oblivious_dir); D = outcomes(detector_dir)
    common = sorted(set(P) & set(O) & set(D))
    rows = []
    for suf in common:
        p, o, dd = P[suf], O[suf], D[suf]
        rows.append({
            "suf": suf,
            "p_out": p["outcome"], "o_out": o["outcome"], "d_out": dd["outcome"],
            "p_n": p["n"], "o_n": o["n"], "d_n": dd["n"],
            # "open" = run ended mid-episode (censored); for the planner that is still
            # surviving (it was holding back when cut off), so count it as survived.
            "survived": p["outcome"] in ("timeout", "reached_B", "ended", "open"),
            "obliv_killed": o["outcome"] == "killed",
            "det_killed": dd["outcome"] == "killed",
            "outlast": p["n"] - o["n"],            # planner survives longer => held back (C) / went around (B)
            "p_wm_rise": p["wm_rise"], "p_det_rise": p["det_rise"], "p_lead": p["lead"],
        })
    # rank: ideal trio first (planner survived + both killed), then longest outlast.
    def key(r):
        return (1 if r["survived"] else 0,
                1 if r["obliv_killed"] else 0,
                1 if r["det_killed"] else 0,
                r["outlast"], r["p_n"])
    rows.sort(key=key, reverse=True)

    print(f"# nav A->B ranking ({len(rows)} matched episodes) "
          f"planner={planner_dir} oblivious={oblivious_dir} detector={detector_dir}")
    print(f"# ideal trio: planner=survived(holds back), oblivious=killed, detector=killed; "
          f"outlast = planner_steps - oblivious_steps (planner held back / went around)")
    print(f"# pWMr/pDETr = planner's WM/detector rise steps; pLead = det_rise-wm_rise "
          f"(positive => WM imagined the turret first)")
    print(f"# {'rk':3} {'episode':14} {'planner_out':12} {'obliv_out':10} {'det_out':10} "
          f"{'pN':4} {'oN':4} {'dN':4} {'outlast':6} pWMr pDETr pLead")
    for i, r in enumerate(rows[:n]):
        wr = "-" if r["p_wm_rise"] is None else str(r["p_wm_rise"])
        dr = "-" if r["p_det_rise"] is None else str(r["p_det_rise"])
        ld = "-" if r["p_lead"] is None else str(r["p_lead"])
        print(f"{i+1:3d} {r['suf']:14} {r['p_out']:12} {r['o_out']:10} {r['d_out']:10} "
              f"{r['p_n']:4d} {r['o_n']:4d} {r['d_n']:4d} {r['outlast']:6d} {wr:>4} {dr:>4} {ld:>5}")
    if rows:
        top = rows[0]
        print(f"\n# top pick (episode index {top['suf']}):")
        print(f"#   planner   {top['p_out']:12} ({top['p_n']} steps) "
              f"-> {planner_dir}/planner_{top['suf']}.mcap")
        print(f"#   oblivious {top['o_out']:10} ({top['o_n']} steps) "
              f"-> {oblivious_dir}/nav_{top['suf']}.mcap")
        print(f"#   detector  {top['d_out']:10} ({top['d_n']} steps) "
              f"-> {detector_dir}/detector_{top['suf']}.mcap")
        print(f"#   outlast={top['outlast']} steps  pWM_rise={top['p_wm_rise']} "
              f"pDET_rise={top['p_det_rise']} pLead={top['p_lead']}")


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dir", nargs="?", default="data/showcase_planner",
                    help="signals dir (phantom mode); ignored in --mode nav")
    ap.add_argument("n", nargs="?", type=int, default=10, help="print top-n")
    ap.add_argument("--mode", choices=["phantom", "nav"], default="phantom",
                    help="phantom = WM-leads-detector ranking (Result 1); "
                         "nav = A->B ideal-trio ranking across 3 controllers (Result 2)")
    ap.add_argument("--nav_planner", default="data/nav_planner")
    ap.add_argument("--nav_oblivious", default="data/nav_oblivious")
    ap.add_argument("--nav_detector", default="data/nav_detector")
    args = ap.parse_args()
    if args.mode == "nav":
        rank_nav(args.nav_planner, args.nav_oblivious, args.nav_detector, args.n)
    else:
        rank_phantom(args.dir, args.n)


if __name__ == "__main__":
    main()

"""Tune and score the orientation-only contact detector against force-plate flags.  [needs opensim]

    conda run -n jcf python -m imu.evaluate_contact --root jcf/full_duration/training/walking \
        --per-lab 10 --out results/imu/contact_tuning [--noise-deg 1 --noise-corr-s 0.1]

For each sampled trial, the foot (calcn) orientations come from the trial's mocap IK through the scaled
model (what a foot IMU would report, without calibration or IK error), optionally with band-limited
noise. detect_contact_orientation runs over a grid of (threshold, extension) settings; each setting is
scored against the force-plate flags the surrogates were trained with (|vertical GRF| > 5 % BW).

Tune on training trials, then score the chosen setting once on the test split (--thresh and --extend
with single values).

Outputs in --out: trials.csv (one row per trial x foot x setting), by_lab.csv, best.json.
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from imu import contact                                             # noqa: E402
from imu.synthetic import band_limited_noise, body_rotations, read_mot, set_pose   # noqa: E402


def foot_rotations(trial_dir, fs):
    """Ground orientation of calcn_r and calcn_l, resampled to fs. -> (time, {side: Rotation})."""
    import opensim as osim
    t, names, v = read_mot(os.path.join(trial_dir, "ik_results.mot"))
    tn = np.arange(t[0], t[-1], 1.0 / fs)
    v = np.column_stack([np.interp(tn, t, v[:, j]) for j in range(v.shape[1])])
    model = osim.Model(os.path.join(trial_dir, "scaled_model.osim"))
    state = model.initSystem()
    cs = model.getCoordinateSet()
    coords = {cs.get(i).getName(): cs.get(i) for i in range(cs.getSize())}
    R = {"r": [], "l": []}
    for row in v:
        set_pose(model, state, coords, names, row)
        m = body_rotations(model, state, ["calcn_r", "calcn_l"])
        R["r"].append(m["calcn_r"]); R["l"].append(m["calcn_l"])
    return tn, {s: Rotation.from_matrix(np.array(R[s])) for s in R}


def lab_of(trial_dir):
    return os.path.basename(trial_dir.rstrip("/")).split("_")[0]


def sample_trials(root, labs, per_lab, seed):
    dirs = sorted(d for d in glob.glob(os.path.join(root, "*"))
                  if os.path.exists(os.path.join(d, "grf_data.mot")) and os.path.exists(os.path.join(d, "ik_results.mot")))
    if labs:
        dirs = [d for d in dirs if lab_of(d) in labs]
    rng = np.random.default_rng(seed)
    out = []
    for lab in sorted(set(map(lab_of, dirs))):
        ds = [d for d in dirs if lab_of(d) == lab]
        # one trial per person first, so a lab with many trials per person still spans people
        people = {}
        for d in ds:
            people.setdefault(re.sub(r"_t\d+(_r\d+)?$", "", os.path.basename(d)), []).append(d)
        pick = [p[rng.integers(len(p))] for p in people.values()]
        rng.shuffle(pick)
        out += pick[:per_lab] if per_lab else pick
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="jcf/full_duration/training/walking")
    ap.add_argument("--dirs", nargs="+", default=None, help="explicit trial folders instead of sampling --root")
    ap.add_argument("--labs", nargs="+", default=None)
    ap.add_argument("--per-lab", type=int, default=10, help="trials per lab, one per person (0 = all people)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fs", type=float, default=100.0)
    ap.add_argument("--thresh", type=float, nargs="+", default=[60, 100, 150, 200])
    ap.add_argument("--extend", type=float, nargs="+", default=[0.0, 0.08, 0.12, 0.16])
    ap.add_argument("--noise-deg", type=float, default=0.0)
    ap.add_argument("--noise-corr-s", type=float, default=0.1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    trials = a.dirs or sample_trials(a.root, a.labs, a.per_lab, a.seed)
    rng = np.random.default_rng(a.seed)
    rows = []
    for k, d in enumerate(trials):
        try:
            mass = float(json.load(open(os.path.join(d, "metadata.json")))["mass_kg"])
            t, rot = foot_rotations(d, a.fs)
            ref = dict(zip(("r", "l"), contact.grf_contact(d, t, mass)))
        except Exception as e:                               # missing files or channels
            print(f"skip {os.path.basename(d)}: {type(e).__name__}: {e}")
            continue
        for side in ("r", "l"):
            if ref[side].all() or not ref[side].any():
                continue                                     # no stance/swing transitions to score
            r = rot[side] * band_limited_noise(len(t), 1 / a.fs, a.noise_deg, a.noise_corr_s, rng) if a.noise_deg else rot[side]
            for th in a.thresh:
                for ex in a.extend:
                    m = contact.compare(contact.detect_contact_orientation(r, a.fs, th, ex), ref[side], a.fs)
                    rows.append(dict(trial=os.path.basename(d), lab=lab_of(d), foot=side, thresh=th, extend=ex,
                                     duration_s=float(t[-1] - t[0]), **m))
        print(f"[{k + 1}/{len(trials)}] {os.path.basename(d)}")
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("no trials scored")
    df.to_csv(os.path.join(a.out, "trials.csv"), index=False)
    cols = ["frame_agreement", "recall", "precision", "onset_error_mean_s", "offset_error_mean_s",
            "onset_within_tol_frac", "offset_within_tol_frac"]
    by_lab = df.groupby(["thresh", "extend", "lab"])[cols].mean()
    by_lab.to_csv(os.path.join(a.out, "by_lab.csv"))
    # rank by the lab-mean of frame agreement, so the lab with the most trials does not decide it
    lab_mean = by_lab["frame_agreement"].groupby(level=["thresh", "extend"]).mean().sort_values(ascending=False)
    th, ex = lab_mean.index[0]
    best = dict(thresh_deg_s=float(th), extend_s=float(ex), lab_mean_frame_agreement=float(lab_mean.iloc[0]),
                per_lab=by_lab.loc[(th, ex)].round(4).to_dict(orient="index"), n_trials=int(df.trial.nunique()),
                noise_deg=a.noise_deg, noise_corr_s=a.noise_corr_s, trials=sorted(df.trial.unique()))
    with open(os.path.join(a.out, "best.json"), "w") as f:
        json.dump(best, f, indent=1)
    print("\nlab-mean frame agreement by setting (top 8):")
    print(lab_mean.head(8).round(3).to_string())
    print(f"\nbest: thresh {th:g} deg/s, extend {ex:g} s")
    print(by_lab.loc[(th, ex)].round(3).to_string())
    print("written to", a.out)


if __name__ == "__main__":
    main()

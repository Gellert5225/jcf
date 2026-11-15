#!/usr/bin/env python3
"""Score an exported optimizer trajectory (robot_q_log CSV) with OpenSim: right-knee F_medial per stance from the label
pipeline with exact static optimization, next to what the surrogate predicts for the same frames.

  1. labels the export if its folder has no exact-SO labels yet: pipeline/label_trajectory_csv.py (IK from the export,
     OpenSim SO, exact SO re-solve, JointReaction with the exact SO muscle forces, label_mask.npz; ~5 min);
  2. runs the William-style figure (analysis/thesis/fig_export_for_william.py): the plain surrogates s1-s3 and the
     current fpopt model through infer_trajectory_csv.py, OpenSim exact SO as the reference;
  3. writes results/thesis/exports/score_<name>.csv: one row per right stance (right foot in contact, double support
     included; only contact periods containing right single stance, so standing on both feet is left out) and per right
     single stance, plus an "all" row per phase, with
       osim_*   OpenSim exact SO, F_medial = 0.5|Fy| - Mx/0.045 (-Mx, BW): mean, peak, % of frames above --limit
       valid_pct, osim_valid_*  frames passing the six SO rules, and the same statistics on them plus the 95th
                percentile (p95: single-frame spikes at contact chatter can pass the SO rules, so p95 is the robust peak)
       surr_*   the plain surrogates (mean of s1-s3): mean, peak, p95, % above --limit
       osim_so_mean  OpenSim's own SO solution (stops short of its optimum on many frames), for comparison

OpenSim is the label pipeline, not measured ground truth (on the Grand Challenge implants it reads ~2x high at push-off).
Usage (repo root, OpenSim env):
  python -I tools/score_export.py --csv "robot_q_log (3).csv" --name "Nov 14 export" --folder jcf/external/william_2026-11-14
"""
import argparse
import os
import site
import subprocess
import sys

sys.path.append(site.getusersitepackages())
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

D_M = 0.045


def run(cmd):
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=REPO, check=True)


def tag_of(name):
    return "".join(ch if ch.isalnum() else "_" for ch in name).strip("_").lower()


def runs(mask, min_len=5):
    e = np.flatnonzero(np.diff(np.r_[0, mask.astype(int), 0]))
    return [(a, b) for a, b in zip(e[::2], e[1::2]) if b - a >= min_len]


def stats(x, limit):
    """mean, peak, 95th percentile, % of frames above limit."""
    x = x[np.isfinite(x)]
    if not len(x):
        return np.nan, np.nan, np.nan, np.nan
    return float(x.mean()), float(x.max()), float(np.percentile(x, 95)), float(100 * np.mean(x > limit))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--csv", required=True)
    ap.add_argument("--name", required=True, help="label used in file names and the figure title")
    ap.add_argument("--folder", required=True, help="labelled trial folder (created if missing)")
    ap.add_argument("--mass", type=float, default=75.0)
    ap.add_argument("--height", type=float, default=1.7)
    ap.add_argument("--limit", type=float, default=1.4, help="F_medial limit (BW) for the %% above columns")
    ap.add_argument("--out", default=os.path.join(REPO, "results", "thesis", "exports"))
    a = ap.parse_args()
    py = sys.executable
    folder = os.path.abspath(a.folder)
    csv = os.path.abspath(a.csv)
    if not os.path.exists(os.path.join(folder, "jcf_output_exact_sojr", "label_mask.npz")):
        relabel = ["--relabel-only"] if os.path.exists(os.path.join(folder, "jcf_output", "_analyze_setup.xml")) else []
        run([py, "pipeline/label_trajectory_csv.py", "--csv", csv, "--out", folder, "--mass", str(a.mass),
             "--height", str(a.height), "--plot"] + relabel)
    elif "so_valid" not in pd.read_csv(os.path.join(folder, "labels.csv"), nrows=1).columns:
        run([py, "pipeline/label_trajectory_csv.py", "--csv", csv, "--out", folder, "--mass", str(a.mass),
             "--height", str(a.height), "--plot", "--relabel-only"])
    run([py, "-I", "analysis/thesis/fig_export_for_william.py", "--csv", csv, "--name", a.name, "--mass", str(a.mass),
         "--height", str(a.height), "--opensim-folder", folder, "--opensim-subdir", "jcf_output_exact_sojr",
         "--suffix", "_exactso", "--out", a.out])

    tag = tag_of(a.name)
    lab = pd.read_csv(os.path.join(folder, "labels.csv"))
    t = lab.time.to_numpy()
    plain = [pd.read_csv(os.path.join(a.out, f"{tag}_plain_s{s}.csv")) for s in (1, 2, 3)]
    surr = np.mean([np.interp(t, p.time, 0.5 * np.abs(p.fy) - p.mx * a.height / D_M) for p in plain], axis=0)
    osim = lab.f_medial_bw.to_numpy()
    valid = lab.so_valid.astype(float).to_numpy() > 0.5
    rows = []
    for phase, mask in (("right stance", lab.phase.isin(["right", "dual"]).to_numpy()),
                        ("right single stance", (lab.phase == "right").to_numpy())):
        single = (lab.phase == "right").to_numpy()
        spans = [(x, y) for x, y in runs(mask) if single[x:y].any()]   # steps only: not standing on both feet
        masks = []
        for k, (s0, s1) in enumerate(spans, 1):
            m = np.zeros(len(t), bool)
            m[s0:s1] = True
            masks.append((k, m))
        if spans:
            masks.append(("all", np.any([m for _, m in masks], axis=0)))
        for k, m in masks:
            o, ov, sv = stats(osim[m], a.limit), stats(osim[m & valid], a.limit), stats(surr[m], a.limit)
            rows.append(dict(export=a.name, phase=phase, stance=k, t_start=float(t[m].min()), t_end=float(t[m].max()),
                             frames=int(m.sum()), valid_pct=float(100 * valid[m].mean()),
                             osim_mean=o[0], osim_peak=o[1], osim_above_pct=o[3],
                             osim_valid_mean=ov[0], osim_valid_peak=ov[1], osim_valid_p95=ov[2], osim_valid_above_pct=ov[3],
                             surr_mean=sv[0], surr_peak=sv[1], surr_p95=sv[2], surr_above_pct=sv[3],
                             osim_so_mean=float(lab.f_medial_opensim_so_bw.to_numpy()[m].mean())))
    tab = pd.DataFrame(rows)
    out = os.path.join(a.out, f"score_{tag}.csv")
    tab.to_csv(out, index=False)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    print(tab.drop(columns=["export"]).round(2).to_string(index=False))
    print("wrote", out)


if __name__ == "__main__":
    main()

"""SC (Grand Challenge subject 3GC, left knee): SO+JR over-read against the implant on stance frames that pass the
label QC (keep_l of <segment>/jcf_output_sojr/label_mask.npz), for Sec. IV-C / Discussion of the paper.

Reads files only (no OpenSim):
  results/thesis/gc/pilot_AB_sojr/frames.csv.gz          instrumented-limb frames, SO+JR labels, measured forces
  jcf/full_duration/testing/walking/<seg>/jcf_output_sojr/label_mask.npz   keep_l, fixed_cop_l on the JR time grid
Each frame takes the mask value of the nearest JR row (SC's JR rows are at most 4.2 ms from the frame times).
Total force = |F_y| of the SO+JR label; medial force = 0.5|F_y| + M_x H / 0.045 (physical sign: +M_x at the left knee).
The 'all' row reproduces the pilot README (bias +0.50 BW, r 0.56; total 3.30 vs 2.12 BW).

    python -I thesis_paper/checks/sc_qc_overread.py      (from the repo root; writes thesis_paper/checks/sc_qc_overread.csv)
"""
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FRAMES = os.path.join(REPO, "results/thesis/gc/pilot_AB_sojr/frames.csv.gz")
SEG_ROOT = os.path.join(REPO, "jcf/full_duration/testing/walking")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sc_qc_overread.csv")


def main():
    f = pd.read_csv(FRAMES)
    sc = f[(f.subject == "SC") & f.stance.astype(bool) & f.fy_osim_sojr.notna()].copy()
    keep, fixed, gap = [], [], 0.0
    for fold, g in sc.groupby("folder"):
        m = np.load(os.path.join(SEG_ROOT, fold, "jcf_output_sojr", "label_mask.npz"))
        t = np.asarray(m["time"], float)
        i = np.clip(np.searchsorted(t, g.t_seg.values), 1, len(t) - 1)
        i = np.where(np.abs(t[i - 1] - g.t_seg.values) < np.abs(t[i] - g.t_seg.values), i - 1, i)
        gap = max(gap, float(np.abs(t[i] - g.t_seg.values).max()))
        keep.append(pd.Series(np.asarray(m["keep_l"], bool)[i], index=g.index))
        fixed.append(pd.Series(bool(np.asarray(m["fixed_cop_l"]).any()), index=g.index))
    sc["keep_l"] = pd.concat(keep)
    sc["fixed_cop_l"] = pd.concat(fixed)
    sc["tot"] = sc.fy_osim_sojr.abs()
    sc["fmed"] = 0.5 * sc.tot + sc.mx_osim_sojr * sc.H / 0.045
    rows = []
    for name, g in (("all", sc), ("moving_cop", sc[~sc.fixed_cop_l]), ("qc_pass", sc[sc.keep_l]),
                    ("qc_fail", sc[~sc.keep_l])):
        rows.append(dict(subset=name, frames=len(g), segments=g.folder.nunique(),
                         share_of_stance_frames=len(g) / len(sc),
                         total_pred_mean=g.tot.mean(), total_meas_mean=g.meas_tot.mean(),
                         total_bias=(g.tot - g.meas_tot).mean(), total_ratio=g.tot.mean() / g.meas_tot.mean(),
                         total_r=np.corrcoef(g.tot, g.meas_tot)[0, 1],
                         medial_bias=(g.fmed - g.meas_med).mean(), medial_mae=(g.fmed - g.meas_med).abs().mean(),
                         medial_r=np.corrcoef(g.fmed, g.meas_med)[0, 1]))
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False, float_format="%.4f")
    print(out.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"largest frame-to-JR-row gap: {gap * 1e3:.1f} ms; wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

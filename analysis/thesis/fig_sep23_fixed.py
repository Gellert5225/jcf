#!/usr/bin/env python3
"""F_medial on the optimizer's Sep 23 reference motion with the audit fixes applied (D1-D4 plus the pelvis mapping):
the corrected models (SO+JR labels, QC mask, physical sign, *_t100 inputs; three training seeds) against the old
walking model and the ISRR paper's checkpoint, both read with the physical sign. Every model gets the pelvis angles
converted to OpenSim's convention and the optimizer's exported COM / foot point (sep23_seeds.py 'export' variant).

There is no ground truth on this motion (sep23_seeds.py docstring): the SO+JR rerun fails the label QC on every
right-knee frame, and the earlier 'ground truth' (labels.csv) is the stored label (no muscle forces) with the pipeline
sign on mis-mapped pelvis angles. That earlier curve is drawn faintly for orientation only.

Data: results/thesis/sep23/per_frame.csv (sep23_seeds.py), results/thesis/sep23/paper_model.csv
(sep23_paper_model.py). Plots only.
Usage (repo root): python -I analysis/thesis/fig_sep23_fixed.py [out.png]
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SEP = os.path.join(REPO, "results", "thesis", "sep23")
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(SEP, "fig_sep23_fixed.png")
LIMIT = 1.4
WINDOWS = {"S1": (1.98, 2.77), "S2": (4.19, 4.83)}     # right single stance (sep23_seeds.WINDOWS)

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, SECOND, MUTED, BAND, GRID = "#0b0b0b", "#52514e", "#8a8986", "#f1f0ec", "#e4e3df"
BLUE_BAND = "#cfe0f6"

d = pd.read_csv(os.path.join(SEP, "per_frame.csv"))
p = pd.read_csv(os.path.join(SEP, "paper_model.csv"))
assert np.allclose(d.t.to_numpy(), p.t.to_numpy())
t = d.t.to_numpy()
seeds = d[["qc_s1_fmed", "qc_s2_fmed", "qc_s3_fmed"]].to_numpy()
s_mean, s_lo, s_hi = seeds.mean(axis=1), seeds.min(axis=1), seeds.max(axis=1)
old = d.sept_legacy_fmed.to_numpy()
paper = p.paper_trial_physical_fmed.to_numpy()
old_gt = d.labels_csv_fmed_pipeline.to_numpy()

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": SECOND, "xtick.color": SECOND,
                     "ytick.color": SECOND, "legend.frameon": True, "savefig.dpi": 200, "savefig.bbox": "tight"})
fig, ax = plt.subplots(figsize=(13, 4.6))
for a, b in WINDOWS.values():
    ax.axvspan(a, b, color=BAND, lw=0, zorder=0)
ax.axhline(LIMIT, color=SECOND, lw=1, ls=(0, (4, 3)), zorder=1)
ax.axhline(0, color=GRID, lw=0.8, zorder=1)
ax.grid(axis="both", color=GRID, lw=0.6, zorder=0)
ax.plot(t, old_gt, color=MUTED, lw=1.0, ls=(0, (1, 1.5)), zorder=2,
        label="old 'ground truth' from the Sep 23 figure (not a reference: no muscles, flipped sign)")
ax.fill_between(t, s_lo, s_hi, color=BLUE_BAND, lw=0, zorder=3, label="corrected model: range of 3 training seeds")
ax.plot(t, s_mean, color=BLUE, lw=2.4, zorder=6, label="corrected model: mean of 3 seeds (all fixes)")
ax.plot(t, old, color=ORANGE, lw=1.8, zorder=5,
        label="old walking fmW0.5 model, no pelvis translation (read with the correct sign)")
ax.plot(t, paper, color=ORANGE, lw=1.8, ls=(0, (4, 2)), zorder=5,
        label="paper model: base, global-mean pelvis (read with the correct sign)")
ax.set_xlim(-0.2, 6.7)
ax.set_ylim(-1.5, 3.5)
ax.set_xlabel("time (s)")
ax.set_ylabel("F_medial (BW)")
ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.13), ncol=2, fontsize=8.6, frameon=False, columnspacing=2.0)

rows = []
for w, (a, b) in WINDOWS.items():
    m = (t >= a - 1e-6) & (t <= b + 1e-6)
    sm = seeds[m].mean(axis=0)
    rows.append(f"{w}: corrected {s_mean[m].mean():.2f} (seeds {sm.min():.2f}-{sm.max():.2f}), "
                f"old {old[m].mean():.2f}, paper {paper[m].mean():.2f}")
    print(f"[{w}] {m.sum()} frames: corrected mean {s_mean[m].mean():.2f} (seeds {np.round(sm, 2)}), "
          f"peak {s_mean[m].max():.2f}; old mean {old[m].mean():.2f}; paper mean {paper[m].mean():.2f}, "
          f"peak {paper[m].max():.2f}; corrected >1.4 in {np.mean(s_mean[m] > LIMIT):.0%}")
ax.text(0.0, -0.36, "Mean F_medial in right single stance (BW).  " + ".   ".join(rows) + ".\n"
        "No ground truth on this motion: OpenSim SO+JR here fails the label quality check on every right-stance "
        "frame (pressure point pinned mid-foot, labelled with mis-mapped pelvis angles).",
        transform=ax.transAxes, ha="left", va="top", fontsize=8.6, color=SECOND, linespacing=1.5)
ax.set_title("Sep 23 reference motion with D1-D4 fixed: correct moment sign, pelvis angles converted, "
             "corrected model vs the old ones (grey = right stance, dashed = 1.4 BW)", fontsize=10.5)
fig.savefig(OUT)
print("wrote", OUT)

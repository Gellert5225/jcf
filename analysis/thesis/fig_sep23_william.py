#!/usr/bin/env python3
"""For William: F_medial on the Sep 23 reference motion from the three corrected checkpoints
(..._fpopt_w20_t100_sojrqc_fmW0p5_e100_s{1,2,3}: SO+JR labels with muscle forces, QC-filtered, physical sign -mx)
against the optimizer's current checkpoint (..._fpopt_w20_fmW0p5_e100) read with +mx (as in ISRR) and with -mx.
Same inputs for every curve (sep23_seeds.py 'export' variant): pelvis angles converted from the URDF's axes to
OpenSim's convention, the optimizer's exported COM and foot point, contact flags from the exported foot wrenches.
No ground truth exists on this motion (sep23_seeds.py docstring).

Data: results/thesis/sep23/per_frame.csv (sep23_seeds.py), results/thesis/sep23/paper_model.csv
(sep23_paper_model.py, william_* columns). Plots only.
Usage (repo root): python -I analysis/thesis/fig_sep23_william.py [out.png]
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
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(SEP, "fig_sep23_corrected_for_william.png")
LIMIT = 1.4
WINDOWS = {"S1": (1.98, 2.77), "S2": (4.19, 4.83)}     # right single stance (sep23_seeds.WINDOWS)

BLUE, ORANGE = "#2a78d6", "#eb6834"
SEED_BLUES = ("#9dc1ee", "#8fb6ea", "#3f84d9")
INK, SECOND, BAND, GRID = "#0b0b0b", "#52514e", "#ecebe6", "#e4e3df"

d = pd.read_csv(os.path.join(SEP, "per_frame.csv"))
w = pd.read_csv(os.path.join(SEP, "paper_model.csv"))
assert np.allclose(d.t.to_numpy(), w.t.to_numpy())
t = d.t.to_numpy()
# Every curve is |F_medial|, as in the ISRR figures and its constraint |F_medial| <= F_limit.
seeds = {f"s{i}": np.abs(d[f"qc_s{i}_fmed"].to_numpy()) for i in (1, 2, 3)}
s_mean = np.abs(np.mean([d[f"qc_s{i}_fmed"].to_numpy() for i in (1, 2, 3)], axis=0))   # |mean of the three|
cur_plus = np.abs(w.william_pipeline_fmed.to_numpy())         # current checkpoint, +mx (ISRR convention)
cur_minus = np.abs(d.fpopt_legacy_fmed.to_numpy())            # current checkpoint, -mx
assert np.allclose(cur_minus, np.abs(w.william_physical_fmed.to_numpy()), atol=1e-4)
# The 'reference' curve of ISRR Fig. 5a, reproduced: mlp-grffree-base, +mx, pelvis angles as written in ik_results.mot
# (copied URDF values), pelvis translations centred on the trial mean (0.07 BW mean deviation from the published curve
# at 12 read-off points; the published curve is zero in left single stance, this one is not masked).
isrr_ref = np.abs(w.paper_trial_copied_pipeline_fmed.to_numpy())
# OpenSim on this motion WITH the muscle forces (JointReaction rerun with the static-optimization forces,
# jcf_output_sojr/), physical sign -mx: sep23_seeds' sojr_unusable column. Computed on the pelvis angles as written
# (copied) and the optimizer's exported foot wrenches; label_mask.npz keeps none of its right-stance frames.
gt_sep23 = np.abs(d.sojr_unusable_fmed.to_numpy())
YMAX = 4.2

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": SECOND, "xtick.color": SECOND,
                     "ytick.color": SECOND, "savefig.dpi": 200, "savefig.bbox": "tight"})
fig, ax = plt.subplots(figsize=(13, 4.6))
for a, b in WINDOWS.values():
    ax.axvspan(a, b, color=BAND, lw=0, zorder=0)
ax.axhline(LIMIT, color=SECOND, lw=1, ls=(0, (4, 3)), zorder=1)
ax.axhline(0, color=GRID, lw=0.8, zorder=1)
ax.grid(axis="both", color=GRID, lw=0.6, zorder=0)
ax.plot(t, gt_sep23, color="#b9b7b0", lw=3.2, zorder=2, solid_capstyle="round",
        label="OpenSim with muscle forces, -Mx (fails the label quality filter on every right-stance frame)")
off = gt_sep23 > YMAX
ax.plot(t[off], np.full(off.sum(), YMAX), ls="none", marker="v", ms=6, color="#9a988f", clip_on=False, zorder=7,
        label=f"OpenSim frames above {YMAX:g} BW ({off.sum()}, max {gt_sep23.max():.1f} BW at {t[gt_sep23.argmax()]:.2f} s)")
for k, (name, f) in enumerate(seeds.items()):
    ax.plot(t, f, color=SEED_BLUES[1], lw=1.1, zorder=3,
            label="corrected s1, s2, s3: three training runs (muscle forces, -Mx)" if k == 0 else None)
m1 = (t >= WINDOWS["S1"][0]) & (t <= WINDOWS["S1"][1])
i_top = np.flatnonzero(m1)[np.argmax(seeds["s1"][m1])]
ax.annotate("s1, s3", (t[i_top], seeds["s1"][i_top]), xytext=(4, 4), textcoords="offset points", fontsize=8.5,
            color=SECOND, ha="left", va="bottom")
ax.annotate("s2 (best checkpoint at epoch 8)", (t[i_top] + 0.05, seeds["s2"][i_top]), xytext=(3.05, 2.3),
            textcoords="data", fontsize=8.5, color=SECOND, ha="left", va="center",
            arrowprops=dict(arrowstyle="-", color=SECOND, lw=0.7, shrinkA=2, shrinkB=2))
ax.plot(t, s_mean, color=BLUE, lw=2.6, zorder=6, label="corrected, mean of s1-s3")
ax.plot(t, isrr_ref, color=INK, lw=1.8, ls=(0, (5, 2)), zorder=5,
        label="ISRR Fig. 5a reference curve (reproduced, see note)")
ax.plot(t, cur_plus, color=SECOND, lw=1.5, ls=(0, (1.2, 1.6)), zorder=5,
        label="your current model (fpopt_w20_fmW0p5), +Mx as in ISRR")
ax.plot(t, cur_minus, color=ORANGE, lw=1.8, zorder=4, label="your current model, correct sign (-Mx)")
ax.set_xlim(-0.2, 6.7)
ax.set_ylim(0, YMAX)
ax.set_xlabel("time (s)")
ax.set_ylabel("|F_medial| (BW)")
ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.13), ncol=2, fontsize=8.6, frameon=False, columnspacing=1.6)

rows = []
for w_name, (a, b) in WINDOWS.items():
    m = (t >= a - 1e-6) & (t <= b + 1e-6)
    sm = [f[m].mean() for f in seeds.values()]
    rows.append(f"{dict(S1='1st right stance', S2='2nd right stance')[w_name]}: corrected mean {s_mean[m].mean():.2f} (s1-s3 {min(sm):.2f}-{max(sm):.2f}), "
                f"OpenSim with muscle forces {gt_sep23[m].mean():.2f}, ISRR reference {isrr_ref[m].mean():.2f}, "
                f"current +Mx {cur_plus[m].mean():.2f}, "
                f"current -Mx {cur_minus[m].mean():.2f}")
    print(f"[{w_name}] {m.sum()} frames: corrected mean {s_mean[m].mean():.2f} peak {s_mean[m].max():.2f}, "
          f"seeds {np.round(sm, 2)}, >1.4 {np.mean(s_mean[m] > LIMIT):.0%}; current +Mx mean {cur_plus[m].mean():.2f} "
          f"peak {cur_plus[m].max():.2f}; GT mean {gt_sep23[m].mean():.2f} peak {gt_sep23[m].max():.2f}; ISRR ref mean {isrr_ref[m].mean():.2f} peak {isrr_ref[m].max():.2f}; current -Mx mean {cur_minus[m].mean():.2f} peak {cur_minus[m].max():.2f}")
ax.text(0.0, -0.40,
        "Mean |F_medial| in right single stance (BW).\n  " + "\n  ".join(rows) + "\n"
        "Inputs: pelvis angles converted from the URDF to OpenSim's convention, your exported COM and foot points, "
        "contact from the exported foot wrenches.\n"
        "ISRR reference curve: mlp-grffree-base with +Mx and the pelvis angles as written, which reproduces Fig. 5a "
        "(within 0.07 BW; left stance not zeroed here).\n"
        "OpenSim with muscle forces is not a validated reference here: 3+ leg muscles saturated in 76 % of right-stance "
        "frames, foot moments up to 684 N m (a pressure point up to 81 cm to the side of the foot)\n"
        "and pulling contact forces at the contact switches (the spikes), and pelvis angles copied from the URDF. "
        "The quality filter rejects every right-stance frame.\n"
        "On the Grand Challenge instrumented knees the corrected mean peaks at 2.7-2.9 BW "
        "medial force vs 1.5-2.2 BW measured, so 1.4 BW on this scale is stricter than 1.4 BW of real load.",
        transform=ax.transAxes, ha="left", va="top", fontsize=8.6, color=SECOND, linespacing=1.5)
ax.set_title("Sep 23 reference motion: OpenSim and corrected surrogates (muscle forces, correct sign) vs your current model "
             "(grey = right stance, dashed = 1.4 BW)", fontsize=10.5)
fig.savefig(OUT)
print("wrote", OUT)

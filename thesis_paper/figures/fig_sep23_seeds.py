"""Fig. 3 (S4) of the audit paper: F_medial of the corrected seeds, the calcn-origin run and the legacy model on the
optimizer's locked-knee reference motion (Sep 23 trajectory). No ground-truth line: the motion has none.

Data: results/thesis/sep23/per_frame.csv (written by analysis/thesis/sep23_seeds.py; the *_fmed columns are its primary
variant: the optimizer's pelvis angles converted from the URDF's axes to OpenSim's, its exported COM and foot point,
identical for all models; physical M_x sign; phi 0.5, d 45 mm). Plots only, no new numbers.
Usage: python -I fig_sep23_seeds.py [<per_frame.csv> [<out.pdf> [<preview.png>]]]
"""
import os
import sys

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "..", "results", "thesis", "sep23", "per_frame.csv")
out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "fig_sep23_seeds.pdf")
png = sys.argv[3] if len(sys.argv) > 3 else None
d = pd.read_csv(src, keep_default_na=False)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Liberation Serif", "STIXGeneral", "DejaVu Serif"],
    "font.size": 7, "axes.labelsize": 7, "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2, "ytick.major.size": 2, "pdf.fonttype": 42,
})
INK, MUTED, BAND = "#0b0b0b", "#8a8986", "#efeeea"
# column -> (legend label, colour, line width, dash, z). Seeds: one blue family, told apart by dash (blue = corrected
# baseline, as in fig2_invivo.py); legacy = orange, as there. The calcn-origin run is dark grey ink, not aqua, because
# aqua marks the SO+JR labels in fig2_invivo.py (blue/orange/aqua are the palette's all-pairs-safe slots). It is
# dotted because its grey has the same CIELAB lightness as seed 1's blue (L* 34), so grayscale print needs the dash.
style = {
    "qc_s1_fmed":        ("Seed 1", "#184f95", 1.0, "-", 6),
    "qc_s2_fmed":        ("Seed 2", "#2a78d6", 1.0, (0, (3.2, 1.3)), 5),
    "qc_s3_fmed":        ("Seed 3", "#6da7ec", 1.0, (0, (4, 1.2, 1, 1.2)), 5),
    "qc_origin_fmed":    ("Calcaneus origin", "#52514e", 1.0, (0, (1, 1)), 4),   # dotted: same lightness as seed 1
    "fpopt_legacy_fmed": ("Legacy", "#eb6834", 1.0, "-", 3),
}

fig, ax = plt.subplots(figsize=(3.5, 1.45))   # 1.45 in tall: page budget (was 1.6)
for w in ("S1", "S2"):
    t = d.loc[d.window == w, "t"]
    x0, x1 = t.min() - 0.005, t.max() + 0.005
    ax.axvspan(x0, x1, color=BAND, lw=0, zorder=0)   # right single stance; no 'S1'/'S2' text (clashes with S1-S4)
ax.axhline(1.4, color=MUTED, lw=0.7, ls=(0, (2.2, 1.6)), zorder=2)
ax.text(0.06, 1.4 - 0.07, "1.4 BW", ha="left", va="top", color=MUTED, fontsize=7)
ax.axhline(0, color="#c3c2b7", lw=0.4, zorder=1)
for col, (lab, c, lw, ls, z) in style.items():
    ax.plot(d.t, d[col], color=c, lw=lw, ls=ls, zorder=z, label=lab, solid_capstyle="round")
ax.set_xlim(0, 6.45)
ax.set_ylim(-0.5, 3.4)
ax.set_yticks([0, 1, 2, 3])
ax.set_xticks([0, 1, 2, 3, 4, 5, 6])
ax.set_xlabel("Time (s)", labelpad=1)
ax.set_ylabel("Medial force (BW)", labelpad=1)   # as in fig2_invivo.py; no sub-7-pt subscript
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
fig.legend(loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.545, 1.025), handlelength=1.7,
           columnspacing=0.8, handletextpad=0.35, borderaxespad=0.1)
fig.subplots_adjust(left=0.1, right=0.985, bottom=0.221, top=0.845)   # same absolute margins as at 1.6 in
fig.savefig(out)
if png:
    fig.savefig(png, dpi=300)

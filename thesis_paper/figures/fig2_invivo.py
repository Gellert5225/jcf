"""Fig. 2 of the audit paper: medial force over stance against the Grand Challenge implants.

Mean stance per subject: every instrumented-limb stance frame of the accepted segments, averaged in bins of
BIN_PCT % of stance (a segment covers 38-51 % of a stance on average; no SC stance is covered beyond 71 %, so a
single full stance cannot represent every subject). Curves: measured implant medial force; our SO+JR labels;
the corrected baseline as the 3-seed ensemble (mean F_y and M_x of qc_s1..s3, as qc_keep/README.md defines
qc_ens); the legacy fpopt model; on SC, the released SO+JR of Kaneda et al. (StatOp_Val).
Every curve: F_medial = 0.5|F_y| + M_x H / 0.045 m with the physical (medial-positive) M_x stored in the frames
file; this reproduces audit_gc/summary.csv and qc_keep/stance_shape.csv exactly. DM is at lag 0.
SC is drawn on the stance frames inside Kaneda et al.'s windows (audit_gc 'statop_overlap' subset, 1,021 frames;
1,015 of them have a stance %), so every SC curve, the implant included, uses identical frames; JW (889) and DM
(878) use all stance frames. The 6 SC stance frames without a stance % (stance_id -1) are left out, as in
qc_keep/stance_shape.csv.

Data: results/thesis/gc/audit_gc/frames.csv.gz (analysis/thesis/grand_challenge.py --name audit_gc).
Plots only, no new numbers.
Usage: python -I fig2_invivo.py [<frames.csv.gz> [<out.pdf> [<bins.csv> [<preview.png>]]]]
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "..", "results", "thesis", "gc", "audit_gc",
                                                         "frames.csv.gz")
out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "fig2_invivo.pdf")
bins_out = sys.argv[3] if len(sys.argv) > 3 else None
png = sys.argv[4] if len(sys.argv) > 4 else None
BIN_PCT = 5.0
PHI, D_M = 0.5, 0.045

d = pd.read_csv(src)
s = d[d.stance & d.pct.notna()].copy()


def fmed(fy, mx, h):
    return PHI * np.abs(fy) + mx * h / D_M


s["f_meas"] = s.meas_med
s["f_sojr"] = fmed(s.fy_osim_sojr, s.mx_osim_sojr, s.H)
s["f_ens"] = fmed(s[["fy_qc_s1", "fy_qc_s2", "fy_qc_s3"]].mean(axis=1),
                  s[["mx_qc_s1", "mx_qc_s2", "mx_qc_s3"]].mean(axis=1), s.H)
s["f_legacy"] = fmed(s.fy_fpopt_legacy, s.mx_fpopt_legacy, s.H)
s["f_statop"] = s.fmed_statop
s = s[(s.subject != "SC") | s.f_statop.notna()]          # SC: Kaneda's windows only (identical frames)
assert s.groupby("subject").size().to_dict() == {"JW": 889, "DM": 878, "SC": 1015}   # 6 SC frames lack a stance %
s["bin"] = np.minimum(np.floor(s.pct / BIN_PCT), 100 / BIN_PCT - 1) * BIN_PCT + BIN_PCT / 2
cols = ["f_meas", "f_sojr", "f_ens", "f_legacy", "f_statop"]
b = s.groupby(["subject", "bin"])[cols].mean()
b["n_frames"] = s.groupby(["subject", "bin"]).size()
b = b.reset_index()
if bins_out:
    b.round(4).to_csv(bins_out, index=False)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Liberation Serif", "STIXGeneral", "DejaVu Serif"],
    "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.5,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2, "ytick.major.size": 2,
    "pdf.fonttype": 42,
})
INK, MUTED, BAND, ZERO = "#0b0b0b", "#8a8986", "#f1f0ec", "#c3c2b7"
# column -> (legend label, colour, line width, dash, z). Colours: the reference palette's first three slots
# (blue / orange / aqua validate all-pairs); blue = corrected baseline and orange = legacy, as in fig_sep23_seeds.
# Legacy is dashed so that SO+JR, legacy and corrected (similar lightness) stay apart in grayscale and under
# protanopia; the only solid curves are the implant (thick black), SO+JR and the corrected baseline.
style = {
    "f_meas":   ("Implant (measured)", INK, 1.5, "-", 6),
    "f_sojr":   ("SO+JR (ours)", "#1baf7a", 1.0, "-", 4),
    "f_ens":    ("Corrected baseline", "#2a78d6", 1.0, "-", 5),
    "f_legacy": ("Legacy", "#eb6834", 1.0, (0, (4, 1.5)), 3),   # dashed: SO+JR / legacy are close in lightness
    "f_statop": ("Kaneda et al.", MUTED, 1.0, (0, (2.4, 1.2)), 5),
}
panels = [("JW", "JW, right knee"), ("DM", "DM, right knee"), ("SC", "SC, left knee")]
fig, axes = plt.subplots(1, 3, figsize=(3.5, 1.62), sharey=True)
for ax, (subj, title) in zip(axes, panels):
    g = b[b.subject == subj]
    for x0, x1 in ((5, 45), (55, 95)):                    # peak windows of the analysis
        ax.axvspan(x0, x1, color=BAND, lw=0, zorder=0)
    ax.axhline(0, color=ZERO, lw=0.5, zorder=1)
    for col, (lab, c, lw, ls, z) in style.items():
        if g[col].isna().all():
            continue
        ax.plot(g["bin"], g[col], color=c, lw=lw, ls=ls, zorder=z, label=lab, solid_capstyle="round",
                dash_capstyle="butt")
    ax.set_title(title, pad=2)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 50, 100])
    ax.set_xticklabels(["0", "50", "100"])
    ax.get_xticklabels()[0].set_horizontalalignment("left")
    ax.get_xticklabels()[-1].set_horizontalalignment("right")
    ax.set_xlabel("% of stance", labelpad=1)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
axes[0].set_ylabel("Medial force (BW)", labelpad=1)
axes[0].set_ylim(-0.6, 3.85)
axes[0].set_yticks([0, 1, 2, 3])
h, lab = [], []
for ax in axes:                                           # Kaneda's line exists on SC only
    for hh, ll in zip(*ax.get_legend_handles_labels()):
        if ll not in lab:
            h.append(hh)
            lab.append(ll)
order = [lab.index(k[0]) for k in style.values()]
fig.legend([h[i] for i in order], [lab[i] for i in order], loc="upper center", ncol=3, frameon=False,
           bbox_to_anchor=(0.545, 1.025), handlelength=1.9, columnspacing=1.2, handletextpad=0.4,
           labelspacing=0.25, borderaxespad=0.1)
fig.subplots_adjust(left=0.105, right=0.99, bottom=0.195, top=0.745, wspace=0.13)
fig.savefig(out)
if png:
    fig.savefig(png, dpi=300)

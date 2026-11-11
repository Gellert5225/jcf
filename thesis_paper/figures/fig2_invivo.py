"""Fig. 2 of the audit paper: stance-binned medial force, Grand Challenge implants vs label sources.

Data: results/thesis/labels_qc/gc_sojr_shape/ensemble.csv (variant 'common'; M_x sign physical,
phi 0.5, d 45 mm; written by analysis/thesis/gc_shape.py --run pilot_AB_sojr). Plots only, no new numbers.
Usage: python -I fig2_invivo.py <ensemble.csv> <out.pdf>
"""
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

src, out = sys.argv[1], sys.argv[2]
d = pd.read_csv(src)
d = d[d.variant == "common"]

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Liberation Serif", "STIXGeneral", "DejaVu Serif"],
    "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "legend.fontsize": 6.5,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "axes.linewidth": 0.5,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2, "ytick.major.size": 2,
    "pdf.fonttype": 42,
})
INK, MUTED = "#0b0b0b", "#8a8986"
style = {  # source -> (label, color, lw, ls, z)
    "meas":       ("Implant (measured)", INK, 1.6, "-", 5),
    "osim_state": ("Stored labels", MUTED, 1.0, (0, (3, 1.5)), 3),
    "osim_sojr":  ("SO+JR labels", "#2a78d6", 1.1, "-", 4),
    "fpopt":      ("Legacy surrogates", "#eb6834", 0.9, "-", 2),
    "sept":       (None, "#eb6834", 0.9, (0, (1, 1)), 2),
}
subjects = [("JW", "JW, right knee"), ("DM", "DM, right knee"), ("SC", "SC, left knee")]
fig, axes = plt.subplots(1, 3, figsize=(3.45, 1.65), sharey=True)
for ax, (s, title) in zip(axes, subjects):
    for x0, x1 in ((5, 45), (55, 95)):
        ax.axvspan(x0, x1, color="#efeeea", lw=0, zorder=0)
    ax.axhline(0, color=MUTED, lw=0.4, zorder=1)
    for source, (lab, col, lw, ls, z) in style.items():
        g = d[(d.subject == s) & (d.source == source)].sort_values("bin_mid")
        if g.empty:
            continue
        ax.plot(g.bin_mid, g["mean"], color=col, lw=lw, ls=ls, zorder=z, label=lab)
    ax.set_title(title, pad=2)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 50, 100]); ax.set_xticklabels(["0", "50", "100"])
    ax.get_xticklabels()[0].set_horizontalalignment("left"); ax.get_xticklabels()[-1].set_horizontalalignment("right")
    ax.set_xlabel("% of stance", labelpad=1)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
axes[0].set_ylabel("Medial force (BW)", labelpad=1)
axes[0].set_ylim(-0.6, 3.8); axes[0].set_yticks([0, 1, 2, 3])
h, l = axes[0].get_legend_handles_labels()
keep = [(hh, ll) for hh, ll in zip(h, l) if ll and not ll.startswith("_")]
fig.legend([k[0] for k in keep], [k[1] for k in keep], loc="upper center", ncol=2, frameon=False,
           bbox_to_anchor=(0.53, 1.01), handlelength=1.8, columnspacing=1.0, handletextpad=0.4)
fig.subplots_adjust(left=0.115, right=0.99, bottom=0.2, top=0.74, wspace=0.2)
fig.savefig(out)
fig.savefig(out.replace(".pdf", ".png"), dpi=300)

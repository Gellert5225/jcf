#!/usr/bin/env python3
"""Grand Challenge implants: measured medial force vs OpenSim (SO+JR, SO forces applied) and the surrogates before and
after heading removal (plain ..._sojrqc_e100_s{1,2,3} vs heading-free ..._sojrqc_hd_e100_s{1,2,3}, mean of 3 seeds).

Reads results/thesis/gc/<run>/frames.csv.gz (analysis/thesis/grand_challenge.py; M_x there is medial-positive for every
source, SC's left knee fed mirrored), F_medial = 0.5|F_y| + M_x*H/0.045. Each instrumented stance (stance_id, pct) is
pooled into 2 % bins of stance (the segments cover parts of stances) and each source's median per bin is drawn per
subject, with the measured interquartile band; r and MAE are over all stance frames.
Usage (repo root): python -I analysis/thesis/fig_gc_heading.py [--run heading_hd_gc]
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, SECOND, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8986", "#e4e3df"
PCT = np.linspace(0, 100, 51)
SUBJECTS = (("JW", "JW (right knee)"), ("DM", "DM (right knee)"), ("SC", "SC (left knee)"))

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
                     "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": SECOND, "xtick.color": SECOND, "ytick.color": SECOND, "legend.frameon": False,
                     "savefig.dpi": 200, "savefig.bbox": "tight"})


def fmed(df, src):
    return 0.5 * np.abs(df[f"fy_{src}"].to_numpy(float)) + df[f"mx_{src}"].to_numpy(float) * df["H"].to_numpy(float) / 0.045


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", default="heading_hd_gc")
    a = ap.parse_args()
    run = os.path.join(REPO, "results", "thesis", "gc", a.run)
    f = pd.read_csv(os.path.join(run, "frames.csv.gz"))
    f = f[f.stance.astype(bool) & f.pct.notna()].copy()
    f["meas"] = f.meas_med
    f["osim"] = fmed(f, "osim_sojr")
    f["plain"] = np.mean([fmed(f, f"plain_s{s}") for s in (1, 2, 3)], axis=0)
    f["hd"] = np.mean([fmed(f, f"hd_s{s}") for s in (1, 2, 3)], axis=0)
    sources = (("meas", "measured (implant)", INK, 2.6), ("osim", "OpenSim SO+JR", AQUA, 1.8),
               ("plain", "surrogate, plain (mean of 3)", ORANGE, 1.8), ("hd", "surrogate, heading-free (mean of 3)", BLUE, 2.4))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True, gridspec_kw=dict(wspace=0.06))
    for ax, (subj, title) in zip(axes, SUBJECTS):
        g = f[f.subject == subj]
        bins = np.clip(np.round(g.pct.to_numpy() / 2.0).astype(int), 0, 50)      # 2 % bins, every stance frame
        curves = {key: np.array([np.median(g[key].to_numpy()[bins == k]) if (bins == k).any() else np.nan
                                 for k in range(51)]) for key, *_ in sources}
        q25 = np.array([np.percentile(g.meas.to_numpy()[bins == k], 25) if (bins == k).any() else np.nan for k in range(51)])
        q75 = np.array([np.percentile(g.meas.to_numpy()[bins == k], 75) if (bins == k).any() else np.nan for k in range(51)])
        n = g.groupby(["gc_key", "stance_id"]).ngroups
        ax.fill_between(PCT, q25, q75, color=MUTED, alpha=0.25, lw=0)
        lines = []
        for key, name, c, lw in sources:
            ok = np.isfinite(g[key]) & np.isfinite(g.meas)
            r = np.corrcoef(g.meas[ok], g[key][ok])[0, 1] if key != "meas" else np.nan
            mae = np.abs(g.meas[ok] - g[key][ok]).mean() if key != "meas" else np.nan
            ax.plot(PCT, curves[key], color=c, lw=lw, zorder=5 if key in ("meas", "hd") else 4,
                    label=name)
            if key != "meas":
                lines.append(f"{name.split(' (')[0]}: r {r:.2f}, MAE {mae:.2f} BW")
        ax.text(0.0, -0.2, "\n".join(lines), transform=ax.transAxes, ha="left", va="top", fontsize=8.2, color=SECOND,
                linespacing=1.5)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 3.6)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_xlabel("% of instrumented-knee stance")
        ax.set_title(f"{title}: {n} stances")
    axes[0].set_ylabel("medial force (BW)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper left", bbox_to_anchor=(0.12, 1.005), ncol=4, columnspacing=1.8, fontsize=9)
    fig.text(0.125, -0.2, "Median of every accepted stance frame (normal and modified gaits; segments cover parts of "
             "stance) in 2 % bins of stance; grey band: measured interquartile range. "
             "OpenSim's late-stance spikes are clipped at 3.6 BW.\nr and MAE over every stance frame. DM's implant timing is offset (see the grand_challenge README), so its "
             "frame metrics are conservative.", fontsize=8, color=SECOND)
    fig.suptitle("Grand Challenge implants: removing the heading from the surrogate's inputs", x=0.125, ha="left",
                 fontsize=12, fontweight="bold", y=1.08)
    out = os.path.join(run, "fig_gc_heading.png")
    fig.savefig(out)
    print("wrote", out)


if __name__ == "__main__":
    main()

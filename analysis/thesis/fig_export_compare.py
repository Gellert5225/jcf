#!/usr/bin/env python3
"""Compare William's exports over right single stance: knee flexion, external knee moment, and right-knee F_medial from
OpenSim (exact SO, frames passing the SO rules) and the plain surrogates (mean of s1-s3).

Each export is averaged over its right single stances, each normalised to 0-100 %. Normal walking (labelled Moore
training trials) is drawn for the knee angle and moment. Needs each export scored first (tools/score_export.py: labels.csv
with so_valid, results/thesis/exports/<tag>_plain_s{1,2,3}.csv).

Usage (repo root):
  python -I analysis/thesis/fig_export_compare.py \\
      --export "robot_q_log (3).csv" "Nov 14 export" jcf/external/william_2026-11-14 \\
      --export "robot_q_log (4).csv" "Nov 15 export" jcf/external/william_2026-11-15 --out-name fig_nov14_vs_nov15.png
"""
import argparse
import os
import site
import sys

sys.path.append(site.getusersitepackages())
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p_ in (REPO, os.path.join(REPO, "analysis", "thesis"), os.path.join(REPO, "tools")):
    if p_ not in sys.path:
        sys.path.insert(0, p_)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from fig_so_diagnosis import (BLUE, GRID, INK, MUTED, ORANGE, PCT, SECOND, export_series,  # noqa: E402
                              normal_walking, normalise, stances)

EXPORTS = os.path.join(REPO, "results", "thesis", "exports")
COLORS = (ORANGE, BLUE)
D_M, LIMIT = 0.045, 1.4


def tag_of(name):
    return "".join(ch if ch.isalnum() else "_" for ch in name).strip("_").lower()


def stance_mean(x, spans):
    return np.nanmean([normalise(x, a, b) for a, b in spans], axis=0)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--export", nargs=3, action="append", metavar=("CSV", "NAME", "FOLDER"), required=True)
    ap.add_argument("--mass", type=float, default=75.0)
    ap.add_argument("--height", type=float, default=1.7)
    ap.add_argument("--out-name", default="fig_export_compare.png")
    a = ap.parse_args()
    if len(a.export) > len(COLORS):
        raise SystemExit(f"at most {len(COLORS)} exports")
    nm, na = normal_walking()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), gridspec_kw=dict(wspace=0.28))
    ax_k, ax_m, ax_f = axes
    for ax, data, lab in ((ax_k, na, "normal walking (median, IQR)"), (ax_m, nm, "normal walking (median, IQR)")):
        ax.fill_between(PCT, np.percentile(data, 25, 0), np.percentile(data, 75, 0), color=MUTED, alpha=0.3, lw=0)
        ax.plot(PCT, np.median(data, 0), color=INK, lw=2.0, label=lab)
    ax_m.axhline(0, color=SECOND, lw=0.8)
    ax_f.axhline(LIMIT, color=SECOND, lw=0.9, ls=(0, (4, 3)))
    summary = []
    for (csv, name, folder), c in zip(a.export, COLORS):
        t, mom, knee, single = export_series(os.path.join(REPO, csv), a.mass, a.height)
        spans = stances(single)
        ax_k.plot(PCT, stance_mean(knee, spans), color=c, lw=2.4, label=name)
        ax_m.plot(PCT, stance_mean(mom, spans), color=c, lw=2.4, label=name)
        lab = pd.read_csv(os.path.join(REPO, folder, "labels.csv"))
        valid = np.interp(t, lab.time, lab.so_valid.astype(float)) > 0.5
        osim = np.where(valid, np.interp(t, lab.time, lab.f_medial_bw), np.nan)
        tag = tag_of(name)
        pl = [pd.read_csv(os.path.join(EXPORTS, f"{tag}_plain_s{s}.csv")) for s in (1, 2, 3)]
        surr = np.mean([np.interp(t, p.time, 0.5 * np.abs(p.fy) - p.mx * a.height / D_M) for p in pl], axis=0)
        ax_f.plot(PCT, stance_mean(osim, spans), color=c, lw=2.6, label=f"{name}: OpenSim, exact SO (valid frames)")
        ax_f.plot(PCT, stance_mean(surr, spans), color=c, lw=1.7, ls=(0, (4, 2)), label=f"{name}: surrogate (plain, mean of 3)")
        m = np.zeros(len(t), bool)
        for s0, s1 in spans:
            m[s0:s1] = True
        summary.append(f"{name}: knee {np.nanmin(knee[m]):.0f}-{np.nanmax(knee[m]):.0f} deg, moment up to "
                       f"{np.nanmax(mom[m]):+.1f} % BW·H; OpenSim {np.nanmean(osim[m & valid]):.2f} / "
                       f"{np.nanpercentile(osim[m & valid], 95):.2f} BW (mean / p95, {valid[m].mean():.0%} valid), "
                       f"surrogate {surr[m].mean():.2f} / {np.percentile(surr[m], 95):.2f}")
    ax_k.set_ylabel("right knee flexion (deg)")
    ax_k.set_ylim(-2, 32)
    ax_k.set_title("A. Knee flexion")
    ax_m.set_ylabel("external knee moment (% BW·H), + = extending")
    ax_m.set_title("B. Ground force about the knee\n    (+: in front of the knee, pushing it straight)")
    ax_f.set_ylabel("F_medial, -Mx (BW)")
    ax_f.set_ylim(0, 5)
    ax_f.set_title("C. Right-knee F_medial\n    (solid: OpenSim exact SO; dashed: surrogate; dotted: 1.4 BW)")
    for ax in axes:
        ax.set_xlim(0, 100)
        ax.set_xlabel("% of right single stance (mean of the export's stances)")
        ax.grid(axis="y", color=GRID, lw=0.6)
    ax_k.legend(loc="upper right")
    ax_m.legend(loc="upper left")
    ax_f.legend(loc="upper left", bbox_to_anchor=(-2.55, -0.16), ncol=4, columnspacing=1.4, fontsize=8)
    fig.text(0.125, -0.1, "\n".join(summary), fontsize=8.6, color=SECOND, va="top", linespacing=1.6)
    fig.suptitle("William's exports over right single stance", x=0.125, ha="left", fontsize=12, fontweight="bold", y=1.02)
    out = os.path.join(EXPORTS, a.out_name)
    fig.savefig(out)
    print("\n".join(summary))
    print("wrote", out)


if __name__ == "__main__":
    main()

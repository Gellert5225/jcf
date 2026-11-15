#!/usr/bin/env python3
"""Sign ablation of the F_medial loss term: muscle-force (SO+JR, QC-masked) models trained with -Mx (the corrected
s1-s3, plus a seed-1 rerun) and with +Mx (mxplus s1-s3), identical otherwise. Every model is SCORED with the physical
sign (-Mx at the right knee). Plots only, from:
  training log            <scratchpad>/sign_ablation_train.log (+Mx s1-s3, -Mx s1 rerun; --log)
  held-out test           results/fmedial_eval/sign_ablation_test/summary.csv
  Grand Challenge         results/thesis/gc/sign_ablation_gc/{frames.csv.gz, gait_mods.csv}
  Sep 23 reference        results/thesis/sep23/sign_ablation.csv, per_frame.csv (OpenSim with muscle forces)
Usage (repo root): python -I analysis/thesis/fig_sign_ablation.py --log <train log> [--out results/thesis/sign_ablation]
"""
import argparse
import os
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(REPO, "results")
D_M, PHI = 0.045, 0.5
# palette slots 1-3 (validated all pairs, normal and CVD); black = measured; grey = reference only
MINUS, PLUS, OSIM = "#2a78d6", "#eb6834", "#1baf7a"
INK, SECOND, MUTED, BAND, GRID = "#0b0b0b", "#52514e", "#8a8986", "#ecebe6", "#e4e3df"
MINUS_SEEDS, PLUS_SEEDS = ["minus_s1", "minus_s2", "minus_s3"], ["plus_s1", "plus_s2", "plus_s3"]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "legend.fontsize": 8.3,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "axes.linewidth": 0.6, "axes.edgecolor": SECOND,
    "xtick.color": SECOND, "ytick.color": SECOND, "text.color": INK, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 200, "savefig.bbox": "tight"})


def grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)


def read_log(path):
    runs, cur = {}, None
    for line in open(path):
        if line.startswith("==="):
            m = re.match(r"=== ([+-])Mx seed (\d)( rerun)?", line)
            cur = (f"{'plus' if m.group(1) == '+' else 'minus'}_s{m.group(2)}{'_rerun' if m.group(3) else ''}"
                   if m else None)
            if cur:
                runs[cur] = []
            continue
        m = re.match(r"Epoch\s+(\d+)/\d+ \| train: ([\d.]+) \| val: ([\d.]+)", line)
        if cur and m:
            runs[cur].append((int(m.group(1)), float(m.group(2)), float(m.group(3))))
    return {k: np.array(v) for k, v in runs.items() if v}


def fmed(fy, mx_medial, h):
    return PHI * np.abs(fy) + mx_medial * h / D_M          # frames store the medial-positive moment


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--log", required=True)
    ap.add_argument("--out", default=os.path.join(RES, "thesis", "sign_ablation"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fig = plt.figure(figsize=(15, 9.6))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1, 1.05], hspace=0.5, wspace=0.3, width_ratios=[1, 0.9, 1])

    # A: validation loss per epoch
    ax = fig.add_subplot(gs[0, 0])
    runs = read_log(a.log)
    for k, v in runs.items():
        c = PLUS if k.startswith("plus") else MINUS
        ax.plot(v[:, 0], v[:, 2], color=c, lw=1.4, alpha=0.9, zorder=3)
        i = int(np.argmin(v[:, 2]))
        ax.plot(v[i, 0], v[i, 2], "o", ms=5, color=c, mec="white", mew=0.8, zorder=4)
    ax.plot([], [], color=MINUS, lw=1.6, label="-Mx loss (seed 1 rerun)")
    ax.plot([], [], color=PLUS, lw=1.6, label="+Mx loss (seeds 1-3)")
    ax.plot([], [], "o", ms=5, color=SECOND, label="best epoch (kept)")
    ax.set_yscale("log")
    ax.set_ylim(0.2, 1.0)
    ax.set_yticks([0.2, 0.3, 0.4, 0.6, 1.0])
    ax.set_yticklabels(["0.2", "0.3", "0.4", "0.6", "1.0"])
    ax.minorticks_off()
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation loss (log)")
    grid(ax)
    ax.legend(loc="upper right")
    ax.set_title("A. Training converges for every run\n    (losses differ between signs, so not comparable)")
    print("[A]", {k: (int(v[np.argmin(v[:, 2]), 0]), round(float(v[:, 2].min()), 4)) for k, v in runs.items()})

    # B: held-out test, stance F_medial error (physical sign), per seed
    ax = fig.add_subplot(gs[0, 1])
    s = pd.read_csv(os.path.join(RES, "fmedial_eval", "sign_ablation_test", "summary.csv"))
    s = s[(s.activity == "walking") & (s.group == "ALL")]
    metrics = (("fmed_mae", "stance", "F_medial error\nin stance (BW)"), ("fy_mae", "stance", "F_y error\nin stance (BW)"),
               ("mx_mae", "stance", "M_x error x H/d\nin stance (BW)"))
    for j, (col, sub, lab) in enumerate(metrics):
        for k, (seeds, c, dx) in enumerate(((MINUS_SEEDS, MINUS, -0.14), (PLUS_SEEDS, PLUS, 0.14))):
            v = s[(s.subset == sub) & s.model.isin(seeds)][col].to_numpy(float)
            if col == "mx_mae":
                v = v * 1.712 / D_M                   # in the units it enters F_medial (median pool height)
            ax.scatter(np.full(len(v), j + dx), v, s=34, color=c, edgecolor="white", lw=0.6, zorder=3)
            ax.plot([j + dx - 0.09, j + dx + 0.09], [v.mean()] * 2, color=INK, lw=1.4, zorder=4)
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([m[2] for m in metrics])
    ax.set_xlim(-0.5, len(metrics) - 0.5)
    ax.set_ylim(0, 0.8)
    ax.set_ylabel("mean absolute error (BW)")
    grid(ax)
    ax.scatter([], [], s=34, color=MINUS, label="-Mx loss, 3 seeds")
    ax.scatter([], [], s=34, color=PLUS, label="+Mx loss, 3 seeds")
    ax.plot([], [], color=INK, lw=1.4, label="mean")
    ax.legend(loc="upper left", ncol=3, columnspacing=1.0, handletextpad=0.3)
    ax.set_title("B. Held-out test people (scored with -Mx):\n    +Mx-trained models are a bit more accurate")
    print("[B]", s[s.subset == "stance"].set_index("model")[["fmed_mae", "fy_mae", "mx_mae"]].round(4).to_dict())

    # C: gait modifications, predicted vs measured change of the stance-peak medial force
    ax = fig.add_subplot(gs[0, 2])
    g = pd.read_csv(os.path.join(RES, "thesis", "gc", "sign_ablation_gc", "gait_mods.csv"))
    m = g.meas_change.to_numpy()
    lim = (-1.0, 1.75)
    ax.plot(lim, lim, color=MUTED, lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax.axhline(0, color=GRID, lw=0.8, zorder=1)
    ax.axvline(0, color=GRID, lw=0.8, zorder=1)
    series = (("OpenSim with muscle forces", g.osim_sojr_change.to_numpy(), OSIM, "D"),
              ("-Mx loss (mean of 3)", g[[f"{k}_change" for k in MINUS_SEEDS]].mean(axis=1).to_numpy(), MINUS, "o"),
              ("+Mx loss (mean of 3)", g[[f"{k}_change" for k in PLUS_SEEDS]].mean(axis=1).to_numpy(), PLUS, "s"))
    for lab, p, c, mk in series:
        clip = np.clip(p, *lim)
        r = np.corrcoef(p, m)[0, 1]
        n_ok = int((np.sign(p) == np.sign(m)).sum())
        ax.scatter(m, clip, s=36, color=c, marker=mk, edgecolor="white", lw=0.6, zorder=3,
                   label=f"{lab}: {n_ok}/10, r {r:+.2f}")
        print(f"[C] {lab}: direction {n_ok}/10, r {r:+.2f}, clipped {(p != clip).sum()}")
    ax.set_xlim(-0.75, 0.6)
    ax.set_ylim(*lim)
    ax.set_xlabel("measured change from normal walking (BW)")
    ax.set_ylabel("predicted change (BW)")
    grid(ax, "both")
    ax.legend(loc="upper left", title="direction right / r with measured", title_fontsize=8.3, alignment="left")
    ax.text(0.98, 0.02, "dashed: prediction = measured\n(points beyond the axis are clipped)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color=SECOND)
    ax.set_title("C. Gait modifications (implants): neither\n    sign learns how the load changes")

    # D: implants, mean stance curves (JW, DM, SC)
    sub = GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[1, 0:2], wspace=0.12)
    fr = pd.read_csv(os.path.join(RES, "thesis", "gc", "sign_ablation_gc", "frames.csv.gz"))
    st = fr[fr.stance & fr.pct.notna()].copy()
    st["bin"] = np.minimum(np.floor(st.pct / 5.0), 19) * 5 + 2.5
    st["meas"] = st.meas_med
    st["osim"] = fmed(st.fy_osim_sojr, st.mx_osim_sojr, st.H)
    for name, seeds in (("minus", MINUS_SEEDS), ("plus", PLUS_SEEDS)):
        st[name] = fmed(st[[f"fy_{k}" for k in seeds]].mean(axis=1), st[[f"mx_{k}" for k in seeds]].mean(axis=1), st.H)
    axes = []
    for k, (subj, title) in enumerate((("JW", "JW, right knee"), ("DM", "DM, right knee"), ("SC", "SC, left knee"))):
        ax = fig.add_subplot(sub[k], sharey=axes[0] if axes else None)
        axes.append(ax)
        b = st[st.subject == subj].groupby("bin")[["meas", "osim", "minus", "plus"]].mean()
        for x0, x1 in ((5, 45), (55, 95)):
            ax.axvspan(x0, x1, color=BAND, lw=0, zorder=0)
        ax.plot(b.index, b.meas, color=INK, lw=2.4, zorder=6, label="implant (measured)")
        ax.plot(b.index, b.osim, color=OSIM, lw=1.6, zorder=4, label="OpenSim with muscle forces")
        ax.plot(b.index, b.minus, color=MINUS, lw=2.0, zorder=5, label="-Mx loss (mean of 3)")
        ax.plot(b.index, b.plus, color=PLUS, lw=2.0, ls=(0, (4, 2)), zorder=5, label="+Mx loss (mean of 3)")
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 50, 100])
        ax.set_xlabel("% of stance")
        ax.text(0.03, 0.97, title, transform=ax.transAxes, ha="left", va="top", fontsize=8.5)
        grid(ax)
        if k:
            ax.tick_params(labelleft=False)
        g_ = st[st.subject == subj]
        print(f"[D] {subj}: r with implant  -Mx {np.corrcoef(g_.minus, g_.meas)[0, 1]:+.2f}  +Mx "
              f"{np.corrcoef(g_.plus, g_.meas)[0, 1]:+.2f}  OpenSim {np.corrcoef(g_.osim, g_.meas)[0, 1]:+.2f}; "
              f"mean bias -Mx {(g_.minus - g_.meas).mean():+.2f} +Mx {(g_.plus - g_.meas).mean():+.2f}")
    axes[0].set_ylabel("medial force (BW)")
    axes[0].set_ylim(-0.5, 5.4)
    axes[0].legend(loc="upper left", bbox_to_anchor=(0, 0.93), fontsize=8)
    axes[0].set_title("D. Real knees (Grand Challenge implants), mean stance: both signs give the same shape and\n"
                      "    read high (shaded: peak windows)")

    # E: Sep 23 reference motion
    ax = fig.add_subplot(gs[1, 2])
    p = pd.read_csv(os.path.join(RES, "thesis", "sep23", "sign_ablation.csv"))
    pf = pd.read_csv(os.path.join(RES, "thesis", "sep23", "per_frame.csv"))
    t = p.t.to_numpy()
    for lo, hi in ((1.98, 2.77), (4.19, 4.83)):
        ax.axvspan(lo, hi, color=BAND, lw=0, zorder=0)
    ax.axhline(1.4, color=SECOND, lw=0.9, ls=(0, (4, 3)), zorder=1)
    ax.plot(t, np.clip(pf.sojr_unusable_fmed.abs(), 0, 4.0), color=OSIM, lw=1.0, alpha=0.7, zorder=2,
            label="OpenSim with muscle forces (fails QC; clipped at 4)")
    ax.plot(t, p[[f"{k}_fmed" for k in MINUS_SEEDS]].mean(axis=1), color=MINUS, lw=2.0, zorder=4,
            label="-Mx loss (mean of 3)")
    ax.plot(t, p[[f"{k}_fmed" for k in PLUS_SEEDS]].mean(axis=1), color=PLUS, lw=2.0, ls=(0, (4, 2)), zorder=5,
            label="+Mx loss (mean of 3)")
    ax.plot(t, p.fpopt_old_labels_fmed, color=MUTED, lw=1.4, ls=(0, (1.5, 1.5)), zorder=3,
            label="old fpopt model (no muscle forces)")
    ax.set_xlim(0, t[-1])
    ax.set_ylim(-0.3, 4.2)
    ax.set_xlabel("time (s)   [shaded: right single stance; dashed: 1.4 BW]")
    ax.set_ylabel("F_medial, -Mx (BW)")
    grid(ax)
    ax.legend(loc="upper left", fontsize=7.8)
    ax.set_title("E. Sep 23 reference motion: muscle forces set the\n    stance level; the loss sign matters outside stance")

    fig.suptitle("Sign ablation: same muscle-force labels, F_medial loss with -Mx vs +Mx (all scored with -Mx)",
                 x=0.07, ha="left", fontsize=12, fontweight="bold", y=0.995)
    out = os.path.join(a.out, "fig_sign_ablation.png")
    fig.savefig(out)
    print("wrote", out)


if __name__ == "__main__":
    main()

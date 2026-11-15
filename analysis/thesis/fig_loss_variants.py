#!/usr/bin/env python3
"""Loss variants on the muscle-force (SO+JR, QC-masked) labels, 3 seeds each, all scored with -Mx:
  -Mx combined   F_medial MSE term with the physical sign (the corrected s1-s3)
  +Mx combined   the same term with the ISRR sign (--mx-sign pipeline)
  sign-free      0.25*(d|Fy|)^2 + (dMx*H/d)^2, the F_medial term without its sign-carrying cross term
  plain          per-output MSE only, no F_medial term (--fmedial-weight 0)
One dot per seed; references as labelled lines. Plots only, from results/fmedial_eval/loss_variants_test,
results/thesis/gc/loss_variants_gc, results/thesis/gc/audit_gc/kam_frames.csv.gz (KAM, analysis/thesis/gc_kam.py) and
results/thesis/sep23/{loss_variants.csv, per_frame.csv}. The quasi-static KAM reference (7/10, r +0.51) is from the
scratchpad test gc_kam_qs.py and is passed in as a constant.
Usage (repo root): python -I analysis/thesis/fig_loss_variants.py [--out results/thesis/sign_ablation]
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(REPO, "results")
VARIANTS = (("minus", "-Mx\ncombined"), ("plus", "+Mx\ncombined"), ("sf", "sign-free"), ("plain", "plain\nper-output"))
DOT, INK, SECOND, MUTED, GRID, OSIM = "#2a78d6", "#0b0b0b", "#52514e", "#8a8986", "#e4e3df", "#1baf7a"
KAM_QS = (7, 0.51)          # quasi-static KAM on the same 10 comparisons (gc_kam_qs.py)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "legend.fontsize": 8.3,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "axes.linewidth": 0.6, "axes.edgecolor": SECOND,
    "xtick.color": SECOND, "ytick.color": SECOND, "text.color": INK, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 200, "savefig.bbox": "tight"})


def strip(ax, values, ylab, title, refs=(), ylim=None, fmt="{:.2f}"):
    """values: {variant key: [seed values]}; one dot per seed, black bar = mean, mean printed above."""
    for i, (k, _lab) in enumerate(VARIANTS):
        v = np.asarray(values[k], float)
        ax.scatter(np.full(len(v), i) + np.array([-0.08, 0.0, 0.08])[:len(v)], v, s=36, color=DOT,
                   edgecolor="white", lw=0.6, zorder=3)
        ax.plot([i - 0.2, i + 0.2], [v.mean()] * 2, color=INK, lw=1.5, zorder=4)
    for ref in refs:
        y, lab, c = ref[:3]
        side, va = (ref[3] if len(ref) > 3 else "right"), (ref[4] if len(ref) > 4 else "bottom")
        ax.axhline(y, color=c, lw=1.0, ls=(0, (4, 3)), zorder=1)
        x = len(VARIANTS) - 0.45 if side == "right" else -0.45
        ax.text(x, ref[5] if len(ref) > 5 else y, lab, ha=side, va=va, fontsize=8, color=SECOND)
    ax.set_xticks(range(len(VARIANTS)))
    ax.set_xticklabels([lab for _k, lab in VARIANTS])
    ax.set_xlim(-0.5, len(VARIANTS) - 0.5)
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_ylabel(ylab)
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title(title)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=os.path.join(RES, "thesis", "sign_ablation"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    seeds = {k: [f"{k}_s{i}" for i in (1, 2, 3)] for k, _ in VARIANTS}
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.6), gridspec_kw=dict(hspace=0.55, wspace=0.3))

    # held-out test people
    s = pd.read_csv(os.path.join(RES, "fmedial_eval", "loss_variants_test", "summary.csv"))
    s = s[(s.activity == "walking") & (s.group == "ALL") & (s.subset == "stance")].set_index("model")
    strip(axes[0, 0], {k: s.loc[v, "fmed_mae"] for k, v in seeds.items()}, "F_medial error in stance (BW)",
          "A. Held-out test people: F_medial error\n    (lower is better; scored with -Mx)", ylim=(0, 0.6))
    strip(axes[0, 1], {k: s.loc[v, "fy_mae"] for k, v in seeds.items()}, "F_y error in stance (BW)",
          "B. Held-out test people: compression error\n    (lower is better)", ylim=(0, 0.85))

    # gait modifications
    g = pd.read_csv(os.path.join(RES, "thesis", "gc", "loss_variants_gc", "gait_mods.csv"))
    m = g.meas_change.to_numpy()
    r = {k: [np.corrcoef(g[f"{x}_change"], m)[0, 1] for x in v] for k, v in seeds.items()}
    r_osim = float(np.corrcoef(g.osim_sojr_change, m)[0, 1])
    kam = pd.read_csv(os.path.join(RES, "thesis", "gc", "audit_gc", "kam_gait_mods.csv"))
    r_kam = float(np.corrcoef(kam.kam, kam.meas)[0, 1])
    strip(axes[0, 2], r, "r(predicted change, measured change)",
          "C. Gait modifications on the implants:\n    does the model follow the load change?",
          refs=((r_osim, f"OpenSim with muscles {r_osim:+.2f}", OSIM, "left"),
                (r_kam, f"KAM, measured GRF {r_kam:+.2f}", MUTED, "left"),
                (KAM_QS[1], f"KAM, quasi-static {KAM_QS[1]:+.2f}", MUTED, "left"), (0, "no relation", GRID)),
          ylim=(-0.4, 0.9))
    print("[C] r per seed", {k: np.round(v, 2).tolist() for k, v in r.items()}, "OpenSim", round(r_osim, 2), "KAM", round(r_kam, 2))
    n_ok = {k: [int((np.sign(g[f"{x}_change"]) == np.sign(m)).sum()) for x in v] for k, v in seeds.items()}
    strip(axes[1, 0], n_ok, "gait modifications with the right direction (of 10)",
          "D. Gait modifications: direction of the\n    change right (5 = chance)",
          refs=((10, "OpenSim with muscles 10", OSIM, "left"), (KAM_QS[0], "KAM (either) 7", MUTED, "left"),
                (5, "chance 5", GRID)),
          ylim=(0, 10.8), fmt="{:.0f}")
    print("[D] direction per seed", n_ok)

    # implants: correlation with the measured medial force, mean over the 3 subjects
    gs = pd.read_csv(os.path.join(RES, "thesis", "gc", "loss_variants_gc", "summary.csv"))
    gs = gs[gs.subset == "all"]
    rim = {k: [gs[gs.source == x].r.mean() for x in v] for k, v in seeds.items()}
    r_osim_im = float(gs[gs.source == "osim_sojr"].r.mean())
    kf = pd.read_csv(os.path.join(RES, "thesis", "gc", "audit_gc", "kam_frames.csv.gz"))
    fr = pd.read_csv(os.path.join(RES, "thesis", "gc", "audit_gc", "frames.csv.gz"), usecols=["subject", "stance", "meas_med"])
    kf["meas_med"], kf["stance"] = fr.meas_med.values, fr.stance.values
    r_kam_im = float(np.mean([np.corrcoef(v.kam, v.meas_med)[0, 1] for _s, v in kf[kf.stance].groupby("subject")]))
    strip(axes[1, 1], rim, "r with measured medial force (mean of 3 subjects)",
          "E. Real knees (implants): tracking the\n    measured medial force over stance",
          refs=((r_osim_im, f"OpenSim with muscles {r_osim_im:.2f}", OSIM), (r_kam_im, f"KAM, measured GRF {r_kam_im:.2f}", MUTED)),
          ylim=(0, 0.85))
    print("[E] implant r per seed", {k: np.round(v, 2).tolist() for k, v in rim.items()}, "OpenSim", round(r_osim_im, 2),
          "KAM", round(r_kam_im, 2))

    # Sep 23 reference motion
    p = pd.read_csv(os.path.join(RES, "thesis", "sep23", "loss_variants.csv"))
    pf = pd.read_csv(os.path.join(RES, "thesis", "sep23", "per_frame.csv"))
    t = p.t.to_numpy()
    w = ((t >= 1.98 - 1e-6) & (t <= 2.77 + 1e-6)) | ((t >= 4.19 - 1e-6) & (t <= 4.83 + 1e-6))
    sep = {k: [p.loc[w, f"{x}_fmed"].mean() for x in v] for k, v in seeds.items()}
    osim_sep = float(pf.loc[w, "sojr_unusable_fmed"].mean())
    old = float(p.loc[w, "fpopt_old_labels_fmed"].mean())
    strip(axes[1, 2], sep, "mean F_medial, right single stance (BW)",
          "F. Sep 23 reference motion: stance level\n    (every variant trained with muscle forces)",
          refs=((osim_sep, f"green dashed: OpenSim with muscles\n{osim_sep:.2f} BW (fails QC)", OSIM, "right", "center", 2.0), (1.4, "1.4 BW", GRID),
                (old, f"old fpopt model {old:.2f}", MUTED)), ylim=(0, 3.2))
    print("[F] Sep 23 stance means", {k: np.round(v, 2).tolist() for k, v in sep.items()}, "OpenSim", round(osim_sep, 2),
          "old", round(old, 2))

    fig.suptitle("Loss variants on the muscle-force labels (3 seeds each, one dot per seed, bar = mean; all scored with -Mx)",
                 x=0.07, ha="left", fontsize=12, fontweight="bold", y=0.985)
    out = os.path.join(a.out, "fig_loss_variants.png")
    fig.savefig(out)
    print("wrote", out)


if __name__ == "__main__":
    main()

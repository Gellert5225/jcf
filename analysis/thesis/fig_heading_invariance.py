#!/usr/bin/env python3
"""William's exports rotated about the vertical to six headings: right-knee F_medial from the plain models
(..._sojrqc_e100_s{1,2,3}, heading-dependent inputs) and the heading-free models (..._sojrqc_hd_e100_s{1,2,3},
inference_posture_surrogate.remove_heading), mean of 3 seeds per heading, against OpenSim with exact SO (labels.csv of
the scored folder; OpenSim does not depend on the heading).

A rotation by psi (robot frame: x forward, y left, z up) adds psi to the URDF pelvis yaw (q3, the first pelvis joint,
about world z) and rotates every x/y pair: pelvis translation, COM, pelvis and foot positions, foot forces and torques,
and their velocities. Predictions are cached in results/thesis/exports/heading/<model>_<export>_psi<psi>_s<seed>.csv.
Usage (repo root): python -I analysis/thesis/fig_heading_invariance.py
"""
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
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

EXPORTS = os.path.join(REPO, "results", "thesis", "exports")
CACHE = os.path.join(EXPORTS, "heading")
CK = "models/runs/mlp/best_model_mlp_walking_qinst_notrans_pelvisrel_winpos_contact_fpopt_w20_t100_sojrqc{}_e100_s{}.pt"
MODELS = (("plain", "", "Plain models"), ("hd", "_hd", "Heading-free models"))
EXPORTS_IN = (("Nov14", "robot_q_log (3).csv", "jcf/external/william_2026-11-14", "Nov 14 (locked knee)"),
              ("Nov15", "robot_q_log (4).csv", "jcf/external/william_2026-11-15", "Nov 15 (knee ~5 deg)"))
HEADINGS = (0, 45, 90, -90, -45, 180)
MASS, H, D_M = 75.0, 1.7, 0.045
BLUE, ORANGE, AQUA, OSIM_FAINT = "#2a78d6", "#eb6834", "#1baf7a", "#a9d9c4"
INK, SECOND, MUTED, BAND, GRID = "#0b0b0b", "#52514e", "#b9b7b0", "#ecebe6", "#e4e3df"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
                     "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": SECOND, "xtick.color": SECOND, "ytick.color": SECOND, "legend.frameon": False,
                     "savefig.dpi": 200, "savefig.bbox": "tight"})


def rotate(df, psi_deg):
    df = df.copy()
    c, s = np.cos(np.radians(psi_deg)), np.sin(np.radians(psi_deg))

    def rot(ax, ay):
        x, y = df[ax].to_numpy(float), df[ay].to_numpy(float)
        df[ax], df[ay] = c * x - s * y, s * x + c * y
    for pre in ("q", "dq", "ddq"):
        rot(f"{pre}0", f"{pre}1")
    df["q3"] = df["q3"] + np.radians(psi_deg)
    for p in ("com_position", "pelvis_position", "right_foot_position", "left_foot_position"):
        rot(f"{p}_0", f"{p}_1")
    for w in ("right_foot_wrench", "left_foot_wrench"):
        rot(f"{w}_0", f"{w}_1")
        rot(f"{w}_3", f"{w}_4")
    return df


def predictions(model, suffix, key, csv, psi):
    """Mean F_medial (-Mx, BW) of the 3 seeds for the export rotated by psi; (time, f_medial)."""
    out = []
    for s in (1, 2, 3):
        f = os.path.join(CACHE, f"{model}_{key}_psi{psi}_s{s}.csv")
        if not os.path.exists(f):
            from fig_plain_trajectories import infer
            rc = os.path.join(CACHE, f"_{key}_psi{psi}.csv")
            if not os.path.exists(rc):
                rotate(pd.read_csv(os.path.join(REPO, csv)), psi).to_csv(rc, index=False)
            infer(rc, os.path.join(REPO, CK.format(suffix, s)), MASS, H, f)
        r = pd.read_csv(f)
        out.append((r.time.to_numpy(), (0.5 * np.abs(r.fy) - r.mx * H / D_M).to_numpy()))
    return out[0][0], np.mean([x for _, x in out], axis=0)


def main():
    os.makedirs(CACHE, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 7.6), sharey=True, gridspec_kw=dict(hspace=0.42, wspace=0.06))
    for j, (key, csv, folder, ename) in enumerate(EXPORTS_IN):
        lab = pd.read_csv(os.path.join(REPO, folder, "labels.csv"))
        tl, osim = lab.time.to_numpy(), lab.f_medial_bw.to_numpy()
        valid = lab.so_valid.astype(float).to_numpy() > 0.5
        single = (lab.phase == "right").to_numpy()
        e = np.flatnonzero(np.diff(np.r_[0, single.astype(int), 0]))
        spans = [(tl[a], tl[b - 1]) for a, b in zip(e[::2], e[1::2]) if b - a >= 10]
        m = single & valid
        for i, (model, suffix, mname) in enumerate(MODELS):
            ax = axes[i, j]
            for lo, hi in spans:
                ax.axvspan(lo, hi, color=BAND, lw=0, zorder=0)
            ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
            ax.plot(tl, np.where(valid, np.clip(osim, -0.5, 5), np.nan), color=AQUA, lw=2.6, zorder=2)
            ax.plot(tl, np.where(~valid, np.clip(osim, -0.5, 5), np.nan), color=OSIM_FAINT, lw=0.9, ls=(0, (1.2, 1.8)),
                    zorder=1)
            means = []
            for psi in HEADINGS:
                t, f = predictions(model, suffix, key, csv, psi)
                means.append(np.interp(tl, t, f)[single].mean())
                style = dict(color=BLUE, lw=2.2, zorder=6) if psi == 0 else \
                    dict(color=ORANGE, lw=1.8, zorder=5) if psi == 90 else dict(color=SECOND, lw=0.9, zorder=4)
                ax.plot(t, f, **style)
            ax.text(0.01, 0.97, f"right single stance mean: {min(means):.2f}-{max(means):.2f} BW across the six "
                    f"headings\nOpenSim exact SO: {osim[m].mean():.2f} BW (valid frames)", transform=ax.transAxes,
                    va="top", fontsize=8.4, color=SECOND)
            ax.set_xlim(tl[0], tl[-1])
            ax.set_ylim(-0.3, 5.0)
            if j == 0:
                ax.set_ylabel("F_medial, -Mx (BW)")
            if i == 1:
                ax.set_xlabel("time (s)   [shaded: right single stance]")
            ax.set_title(f"{mname} — {ename}")
    handles = [Line2D([], [], color=BLUE, lw=2.2, label="model, as exported (heading 0 deg)"),
               Line2D([], [], color=ORANGE, lw=1.8, label="model, rotated +90 deg (the Moore lab frame)"),
               Line2D([], [], color=SECOND, lw=0.9, label="model, rotated -90, +-45, 180 deg"),
               Line2D([], [], color=AQUA, lw=2.6, label="OpenSim, exact SO, valid frames (same at any heading)"),
               Line2D([], [], color=OSIM_FAINT, lw=0.9, ls=(0, (1.2, 1.8)), label="OpenSim, rejected frames"),
               Patch(facecolor=BAND, label="right single stance")]
    axes[1, 0].legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, -0.2), ncol=3, columnspacing=1.6)
    fig.suptitle("The same motion rotated about the vertical: the plain models change their answer, the heading-free "
                 "models do not (mean of 3 training runs per line)", x=0.125, ha="left", fontsize=11.5,
                 fontweight="bold", y=0.99)
    out = os.path.join(EXPORTS, "fig_heading_invariance.png")
    fig.savefig(out)
    print("wrote", out)


if __name__ == "__main__":
    main()

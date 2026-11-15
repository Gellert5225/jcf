#!/usr/bin/env python3
"""Why OpenSim's static optimization rejects frames on William's Nov 14 export, and what tuning SO changes.

A  external sagittal moment of the ground force about the right knee (tibia z axis; + = extending, the force line in
   front of the knee), % BW*H, over right single stance: the export's two stances vs normal walking (labelled Moore
   training trials, measured GRF, OpenSim forward kinematics; median and IQR)
B  right knee flexion over the same stance
C  SO activations on the export (labeller default; gastrocnemius = max of med/lat, tibialis anterior, soleus); right
   single stance shaded, frames rejected by the SO rules (label_mask so_any_r) hatched, Ipopt failures marked
D  gastrocnemius and tibialis anterior / EDL activations in the two right single stances: OpenSim's stored solution vs
   the exact optimum of the same frame problem (so_exact.py, jcf_output_exact/)
Second figure (fig_so_tuning_nov14.png): A  OpenSim F_medial (SO muscle forces, -Mx) over time for OpenSim's SO and the
exact optima (plain model, hyperextension capsule); B  per SO setting: share of right-single-stance frames passing the
six SO rules and share solved to the optimum (cost within 10 % of the exact optimum), Ipopt failures, F_medial mean / peak

Reads results/thesis/exports/so_tuning_nov14.csv, so_tuning_nov14_frames.csv.gz, so_failures_nov14.csv and
jcf/external/william_2026-11-14/jcf_output_exact/ (so_exact.py)
(analysis/thesis/so_tuning_nov14.py; runs from analysis/thesis/so_variant.py).
Usage (repo root): python -I analysis/thesis/fig_so_diagnosis.py
"""
import argparse
import glob
import json
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

from fig_defects import read_sto  # noqa: E402
from kam_trajectory_csv import to_opensim  # noqa: E402
from inference_posture_surrogate import urdf_pelvis_to_opensim  # noqa: E402
from pn3_redis.model import COORDS, LowerBody  # noqa: E402
from pn3_redis.publisher import URDF_Q_NAMES  # noqa: E402
from analysis.thesis.osim_fk import OsimChain  # noqa: E402

FOLDER = os.path.join(REPO, "jcf", "external", "william_2026-11-14")
CSV = os.path.join(REPO, "robot_q_log (3).csv")
EXPORTS = os.path.join(REPO, "results", "thesis", "exports")
MASS, HEIGHT = 75.0, 1.7
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, SECOND, MUTED, BAND, GRID = "#0b0b0b", "#52514e", "#8a8986", "#ecebe6", "#e4e3df"
PCT = np.linspace(0, 100, 51)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "legend.fontsize": 8,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "axes.linewidth": 0.6, "axes.edgecolor": SECOND,
    "xtick.color": SECOND, "ytick.color": SECOND, "text.color": INK, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 200, "savefig.bbox": "tight"})


def ext_moment(F, P, T, knee, R):
    """External moment of the ground force (force F at P plus free torque T, ground axes) about the knee, on the tibia's
    z (flexion) axis; + = extending."""
    return np.einsum("tji,tj->ti", R, np.cross(P - knee, F) + T)[:, 2]


def stances(single, min_len=20):
    e = np.flatnonzero(np.diff(np.r_[0, single.astype(int), 0]))
    return [(a, b) for a, b in zip(e[::2], e[1::2]) if b - a >= min_len]


def normalise(x, a, b):
    return np.interp(PCT, np.linspace(0, 100, b - a), x[a:b])


def export_series(csv=CSV, mass=MASS, height=HEIGHT):
    """time, external knee moment (% BW*H, + = extending), right knee flexion (deg), right single stance of an export."""
    df = pd.read_csv(csv).dropna(subset=["com_position_0"]).reset_index(drop=True)
    idx = {n: i for i, n in enumerate(URDF_Q_NAMES)}
    qu = df[[f"q{i}" for i in range(37)]].to_numpy(float)
    q = np.zeros((len(df), len(COORDS)))
    for j, n in enumerate(COORDS):
        q[:, j] = qu[:, idx[n]]
    q[:, [COORDS.index(c) for c in ("pelvis_tx", "pelvis_ty", "pelvis_tz")]] = to_opensim(
        qu[:, [idx["pelvis_tx"], idx["pelvis_ty"], idx["pelvis_tz"]]])
    q[:, [COORDS.index(c) for c in ("pelvis_tilt", "pelvis_list", "pelvis_rotation")]] = urdf_pelvis_to_opensim(
        qu[:, [idx["pelvis_tilt"], idx["pelvis_list"], idx["pelvis_rotation"]]])
    R, knee = LowerBody(height_m=height).fk(q)["tibia_r"]
    W = lambda s, a: to_opensim(df[[f"right_foot_{s}_{k}" for k in a]].to_numpy(float))   # noqa: E731
    m = ext_moment(W("wrench", range(3)), W("position", range(3)), W("wrench", range(3, 6)), knee, R)
    bw = mass * 9.81
    single = (df.right_foot_wrench_2.to_numpy() > 0.05 * bw) & (df.left_foot_wrench_2.to_numpy() < 0.05 * bw)
    return df.time.to_numpy(), m / (bw * height) * 100, np.degrees(qu[:, idx["knee_angle_r"]]), single


def normal_walking(n=12):
    mom, ang = [], []
    folders = [d for d in sorted(glob.glob(os.path.join(REPO, "jcf", "full_duration", "training", "walking",
                                                        "moore_subject*_t1*_r00"))) if os.path.exists(f"{d}/ik_results.mot")]
    for d in folders[:n]:
        ik = pd.read_csv(f"{d}/ik_results.mot", sep=r"\s+", skiprows=6)
        co = {c: ik[c].to_numpy(float) for c in ik.columns if c != "time"}
        meta = json.load(open(f"{d}/metadata.json"))
        bw, H = meta["mass_kg"] * 9.81, meta["height_m"]
        g = read_sto(f"{d}/grf_data.mot")
        Rt, pt = OsimChain(f"{d}/scaled_model.osim").body_pose("tibia_r", co)
        I = lambda c: np.interp(ik.time, g.time, g[c])     # noqa: E731
        F = np.column_stack([I(f"calcn_r_force_v{a}") for a in "xyz"])
        P = np.column_stack([I(f"calcn_r_force_p{a}") for a in "xyz"])
        T = np.column_stack([I(f"calcn_r_torque_{a}") for a in "xyz"])
        m = ext_moment(F, P, T, pt, Rt) / (bw * H) * 100
        single = (F[:, 1] > 0.05 * bw) & (I("calcn_l_force_vy") < 0.05 * bw)
        kn = np.degrees(ik.knee_angle_r.to_numpy())
        for a, b in stances(single):
            mom.append(normalise(m, a, b))
            ang.append(normalise(kn, a, b))
    return np.array(mom), np.array(ang)


def shade_single(ax, t, single):
    for s0, s1 in stances(single, 10):
        ax.axvspan(t[s0], t[s1 - 1], color=BAND, lw=0, zorder=0)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=EXPORTS)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tab = pd.read_csv(os.path.join(EXPORTS, "so_tuning_nov14.csv"))
    fails = pd.read_csv(os.path.join(EXPORTS, "so_failures_nov14.csv"))

    t, mom, knee, single = export_series()
    nm, na = normal_walking()
    fig = plt.figure(figsize=(14, 14.2))
    gs = fig.add_gridspec(3, 2, hspace=0.6, wspace=0.2, height_ratios=[1, 1, 0.95], top=0.94)

    # A: external knee moment over right single stance
    ax = fig.add_subplot(gs[0, 0])
    ax.axhspan(0, 7, color=BAND, lw=0, zorder=0)
    ax.text(97, 6.6, "force line in front of the knee:\npushes it straight (hyperextension)", ha="right", va="top",
            fontsize=8, color=SECOND)
    ax.text(97, -4.6, "force line behind the knee: bends it", ha="right", va="bottom", fontsize=8, color=SECOND)
    ax.axhline(0, color=SECOND, lw=0.8, zorder=1)
    ax.fill_between(PCT, np.percentile(nm, 25, 0), np.percentile(nm, 75, 0), color=MUTED, alpha=0.3, lw=0, zorder=2)
    ax.plot(PCT, np.median(nm, 0), color=INK, lw=2.2, zorder=4, label=f"normal walking (median, IQR; {len(nm)} stances)")
    for k, (s0, s1) in enumerate(stances(single)):
        ax.plot(PCT, normalise(mom, s0, s1), color=BLUE, lw=2.2, ls="-" if k == 0 else (0, (4, 2)), zorder=5,
                label=f"Nov 14 export, right stance {k + 1} ({t[s0]:.2f}-{t[s1 - 1]:.2f} s)")
    ax.set_xlim(0, 100)
    ax.set_ylim(-5, 7)
    ax.set_xlabel("% of right single stance")
    ax.set_ylabel("external knee moment (% BW·H)")
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.93))
    ax.set_title("A. The ground force moves in front of the knee in late stance\n    (normal walking: always behind it)")

    # B: knee angle
    ax = fig.add_subplot(gs[0, 1])
    ax.fill_between(PCT, np.percentile(na, 25, 0), np.percentile(na, 75, 0), color=MUTED, alpha=0.3, lw=0)
    ax.plot(PCT, np.median(na, 0), color=INK, lw=2.2, label="normal walking (median, IQR)")
    for k, (s0, s1) in enumerate(stances(single)):
        ax.plot(PCT, normalise(knee, s0, s1), color=BLUE, lw=2.2, ls="-" if k == 0 else (0, (4, 2)),
                label=f"Nov 14 export, right stance {k + 1}")
    ax.set_xlim(0, 100)
    ax.set_ylim(-3, 35)
    ax.set_xlabel("% of right single stance")
    ax.set_ylabel("right knee flexion (deg)")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.legend(loc="upper right")
    ax.set_title("B. ...while the knee stays fully straight (0 deg). The model's knee has no\n"
                 "    hyperextension stop, so only muscles can hold it")

    # C: activations (default run)
    ax = fig.add_subplot(gs[1, :])
    act = read_sto(os.path.join(FOLDER, "jcf_output", "BatchJCF_StaticOptimization_activation.sto"))
    z = np.load(os.path.join(FOLDER, "jcf_output_sojr", "label_mask.npz"))
    rej = np.interp(act.time, z["time"], z["so_any_r"].astype(float)) > 0.5
    shade_single(ax, t, single)
    ax.fill_between(act.time, 0, 1.06, where=rej, color="none", hatch="////", edgecolor="#cfccc2", lw=0, zorder=1)
    ax.plot(act.time, np.maximum(act.gasmed_r, act.gaslat_r), color=BLUE, lw=2.0, zorder=4)
    ax.plot(act.time, act.tibant_r, color=ORANGE, lw=1.6, zorder=3)
    ax.plot(act.time, act.soleus_r, color=AQUA, lw=1.6, zorder=3)
    ax.axhline(1.0, color=SECOND, lw=0.8, ls=(0, (4, 3)))
    f0 = fails.query("tag == 'default'").time.to_numpy()
    ax.plot(f0, np.full(len(f0), 1.1), ls="none", marker="v", ms=5, color=INK, clip_on=False, zorder=6)
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(0, 1.13)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("activation (right leg)")
    handles = [Patch(facecolor=BAND, edgecolor="none", label="right single stance"),
               Patch(facecolor="none", edgecolor="#cfccc2", hatch="////", label="frame rejected by the SO rules"),
               Line2D([], [], ls="none", marker="v", ms=5, color=INK, label=f"Ipopt failure ({len(f0)}, all in left stance)"),
               Line2D([], [], color=BLUE, lw=2.0, label="gastrocnemius (knee flexor + plantarflexor)"),
               Line2D([], [], color=ORANGE, lw=1.6, label="tibialis anterior (dorsiflexor)"),
               Line2D([], [], color=AQUA, lw=1.6, label="soleus (plantarflexor)")]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, -0.16), ncol=3, columnspacing=1.6)
    ax.set_title("C. OpenSim's SO solution: in late right stance the gastrocnemius saturates (1.0) and the tibialis anterior "
                 "cancels its ankle pull (soleus never used).\n    Saturation and dorsi/plantarflexor co-activation are "
                 "what reject the frames. Ipopt reports no failure there, but the solution is not optimal (D)")

    # D: stored vs exact-optimum activations in the two right single stances
    o = read_sto(os.path.join(FOLDER, "jcf_output_exact", "FullDuration_StaticOptimization_activation.sto"))
    sub = gs[2, :].subgridspec(1, 2, wspace=0.06)
    for k, (s0, s1) in enumerate(stances(single)):
        ax = fig.add_subplot(sub[0, k])
        lo, hi = t[s0], t[s1 - 1]
        m = (act.time >= lo - 1e-9) & (act.time <= hi + 1e-9)
        mo = (o.time >= lo - 1e-9) & (o.time <= hi + 1e-9)
        ax.plot(act.time[m], np.maximum(act.gasmed_r, act.gaslat_r)[m], color=BLUE, lw=2.0, label="gastrocnemius, OpenSim")
        ax.plot(o.time[mo], np.maximum(o.gasmed_r, o.gaslat_r)[mo], color=BLUE, lw=1.6, ls=(0, (1.2, 1.4)),
                label="gastrocnemius, exact optimum")
        ax.plot(act.time[m], np.maximum(act.tibant_r, act.edl_r)[m], color=ORANGE, lw=2.0,
                label="tibialis anterior / EDL, OpenSim")
        ax.plot(o.time[mo], np.maximum(o.tibant_r, o.edl_r)[mo], color=ORANGE, lw=1.6, ls=(0, (1.2, 1.4)),
                label="tibialis anterior / EDL, exact optimum")
        ax.axhline(1.0, color=SECOND, lw=0.8, ls=(0, (4, 3)))
        ax.set_xlim(lo, hi)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel(f"time (s), right stance {k + 1}")
        ax.grid(axis="y", color=GRID, lw=0.6)
        if k == 0:
            ax.set_ylabel("activation (right leg)")
            ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.17), ncol=4, columnspacing=1.6)
            ax.set_title("D. The co-activation is the solver stopping early: the exact optimum of the same SO problem needs little dorsiflexor.\n    The gastrocnemius still saturates at the end of stance (holding the straight knee against the ground force)")
        else:
            ax.tick_params(axis="y", labelleft=False)

    fig.suptitle("Nov 14 export: most rejected frames come from OpenSim's SO stopping short of its optimum, not from the "
                 "motion", x=0.07, ha="left", fontsize=12, fontweight="bold", y=0.985)
    out = os.path.join(a.out, "fig_so_diagnosis_nov14.png")
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)
    tuning_figure(tab, t, single, a.out)
    print(tab[["tag", "ipopt_failures", "pass_pct", "nonopt_pct", "fmed_mean", "fmed_peak"]].round(2).to_string(index=False))


def tuning_figure(tab, t, single, out_dir):
    """F_medial over time for OpenSim's SO and the exact optima; per setting: frames passing the SO rules and solved to
    the optimum."""
    frames = pd.read_csv(os.path.join(EXPORTS, "so_tuning_nov14_frames.csv.gz"))
    by = tab.set_index("tag")
    fig = plt.figure(figsize=(14, 10.6))
    gs = fig.add_gridspec(2, 1, hspace=0.42, height_ratios=[1, 1.05], top=0.93)

    # A: F_medial over time
    ax = fig.add_subplot(gs[0])
    shade_single(ax, t, single)
    for tag, c, lw, zo, name in (("default", MUTED, 3.2, 3, "OpenSim SO as labelled (tolerance 1e-4)"),
                                 ("optimum_plain", BLUE, 1.9, 5, "exact optimum of the same SO problem"),
                                 ("optimum_capsule", ORANGE, 1.9, 4, "exact optimum + hyperextension capsule (model change)")):
        f = frames[frames.tag == tag]
        r = by.loc[tag]
        ax.plot(f.time, np.clip(f.fmed, -0.5, 5.2), color=c, lw=lw, zorder=zo,
                label=f"{name}: R single stance mean {r.fmed_mean:.2f}, peak {r.fmed_peak:.2f} BW; "
                      f"{r.pass_pct:.0f}% of frames pass the SO rules")
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(-0.5, 5.2)
    ax.set_xlabel("time (s)   [shaded: right single stance]")
    ax.set_ylabel("OpenSim F_medial, -Mx (BW)")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.13), ncol=1)
    ax.set_title("A. OpenSim F_medial (JointReaction with the SO muscle forces) for OpenSim's SO solution and the exact "
                 "optimum. Solving to the optimum\n    removes the 2.0-2.2 s and 4.0-4.2 s plateaus; the end-of-stance "
                 "peak stays unless something other than muscles holds the straight knee")

    # B: per setting, frames passing the SO rules and frames at their optimum
    sub = gs[1].subgridspec(1, 3, width_ratios=[0.36, 1.0, 0.42], wspace=0.0)
    ax = fig.add_subplot(sub[0, 1])
    y = np.arange(len(tab))[::-1]
    for yi, (_, r) in zip(y, tab.iterrows()):
        ax.plot([0, 100], [yi, yi], color=GRID, lw=0.8, zorder=0)
        if np.isfinite(r.get("nonopt_pct", np.nan)):
            ax.plot(100 - r.nonopt_pct, yi, "o", ms=8.5, mfc="white", mec=MUTED, mew=1.6, zorder=2)
        ax.plot(r.pass_pct, yi, "o", ms=7, color=INK, mec="white", mew=1.2, zorder=3, clip_on=False)
        fails = "  -" if not np.isfinite(r.ipopt_failures) else f"{int(r.ipopt_failures):>3d}"
        ax.text(104, yi, f"{fails}      {r.fmed_mean:4.2f} / {r.fmed_peak:4.2f}", va="center", fontsize=8, color=SECOND,
                family="DejaVu Sans Mono")
    ax.text(104, len(tab) - 0.45, "Ipopt    F_medial R stance\nfailures mean / peak (BW)", fontsize=7.8, color=SECOND,
            va="bottom", family="DejaVu Sans Mono")
    ax.set_yticks(y)
    ax.set_yticklabels(tab.setting, fontsize=8)
    ax.set_xlim(0, 101)
    ax.set_ylim(-0.6, len(tab) - 0.45)
    ax.set_xlabel("% of right-single-stance frames")
    ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(handles=[Line2D([], [], ls="none", marker="o", ms=7, color=INK, label="pass the six SO rules"),
                       Line2D([], [], ls="none", marker="o", ms=8.5, mfc="white", mec=MUTED, mew=1.6,
                              label="solved to the optimum (cost within 10 % of it)")],
              loc="upper left", bbox_to_anchor=(0.0, -0.13), ncol=2)
    ax.set_title("B. Per SO setting: loosening the tolerance never changes which frames pass; solving to the optimum does",
                 pad=10, x=-0.36)
    ax.text(-0.36, -0.27, "Tighter tolerances (1e-6, 1e-8) were also tried: Ipopt failed on 53 of the first 55 frames "
            "(\"Restoration failed\", ~4 s per frame); runs stopped.\nExact optimum: every SO frame re-solved exactly "
            "(SLSQP, same cost, bounds and joint-moment targets as OpenSim's SO), then OpenSim's JointReaction.\n"
            "19 frames (left stance, 16 of them OpenSim's Ipopt failures) could not be solved and keep OpenSim's values. "
            "Capsule: a torque at the right knee that can only resist hyperextension.",
            transform=ax.transAxes, fontsize=8, color=SECOND, va="top", linespacing=1.5)
    fig.suptitle("Nov 14 export: what tuning static optimization changes", x=0.07, ha="left", fontsize=12,
                 fontweight="bold", y=0.985)
    out = os.path.join(out_dir, "fig_so_tuning_nov14.png")
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""William-style |F_medial| plot for one exported optimizer trajectory (robot_q_log CSV format): the plain
per-output-loss models (muscle-force labels; ..._t100_sojrqc_e100_s{1,2,3}) and his current checkpoint
(..._fpopt_w20_fmW0p5_e100, labels without muscle forces) read with +Mx (as in ISRR) and with -Mx. Every model goes
through infer_trajectory_csv.py (pelvis angles converted, contact from the foot wrenches, centred 20-frame windows).
For the Sep 23 export, OpenSim with muscle forces (results/thesis/sep23/per_frame.csv) is added as a reference.

Usage (repo root):
  python -I analysis/thesis/fig_export_for_william.py --csv "robot_q_log (3).csv" --name "Nov 14 export" [--mass 75 --height 1.7]
"""
import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "analysis", "thesis"))
sys.path.insert(0, os.path.join(REPO, "tools"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from fig_plain_trajectories import infer  # noqa: E402   (runs infer_trajectory_csv.py through its CLI)
from kam_trajectory_csv import kam_series, runs  # noqa: E402

PLAIN = "models/runs/mlp/best_model_mlp_walking_qinst_notrans_pelvisrel_winpos_contact_fpopt_w20_t100_sojrqc_e100_s{}.pt"
CURRENT = "models/runs/mlp/best_model_mlp_walking_qinst_notrans_pelvisrel_winpos_contact_fpopt_w20_fmW0p5_e100.pt"
D_M, LIMIT = 0.045, 1.4
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, SECOND, BAND, GRID, SEED = "#0b0b0b", "#52514e", "#ecebe6", "#e4e3df", "#8fb6ea"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": SECOND, "xtick.color": SECOND,
                     "ytick.color": SECOND, "savefig.dpi": 200, "savefig.bbox": "tight"})


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--csv", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--mass", type=float, default=75.0)
    ap.add_argument("--height", type=float, default=1.7)
    ap.add_argument("--opensim-folder", default=None, help="labelled trial folder of this export (label_trajectory_csv.py); "
                    "draws its JointReaction with the SO muscle forces (jcf_output_sojr/), -Mx")
    ap.add_argument("--out", default=os.path.join(REPO, "results", "thesis", "exports"))
    a = ap.parse_args()
    os.chdir(REPO)
    os.makedirs(a.out, exist_ok=True)
    H = a.height
    tag = "".join(ch if ch.isalnum() else "_" for ch in a.name).strip("_").lower()
    plain = [infer(a.csv, PLAIN.format(s), a.mass, H, os.path.join(a.out, f"{tag}_plain_s{s}.csv")) for s in (1, 2, 3)]
    cur = infer(a.csv, CURRENT, a.mass, H, os.path.join(a.out, f"{tag}_current_fpopt.csv"))
    t = plain[0].time.to_numpy()
    seeds = [np.abs(0.5 * np.abs(r.fy) - r.mx * H / D_M).to_numpy() for r in plain]       # -Mx, |F_medial|
    mean = np.abs(np.mean([0.5 * np.abs(r.fy) - r.mx * H / D_M for r in plain], axis=0))
    cur_plus = np.abs(0.5 * np.abs(cur.fy) + cur.mx * H / D_M).to_numpy()
    cur_minus = np.abs(0.5 * np.abs(cur.fy) - cur.mx * H / D_M).to_numpy()

    src = pd.read_csv(a.csv).dropna(subset=["com_position_0"]).reset_index(drop=True)
    k = kam_series(src, a.mass, H)
    single = k.contact_r.to_numpy().astype(bool) & ~k.contact_l.to_numpy().astype(bool)
    spans = [(k.time.iloc[x], k.time.iloc[y - 1]) for x, y in runs(single) if y - x >= 10]

    fig, ax = plt.subplots(figsize=(13, 4.6))
    for lo, hi in spans:
        ax.axvspan(lo, hi, color=BAND, lw=0, zorder=0)
    ax.axhline(LIMIT, color=SECOND, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.grid(axis="both", color=GRID, lw=0.6, zorder=0)
    ref = os.path.join(REPO, "results", "thesis", "sep23", "per_frame.csv")
    osim_rows = None
    if a.opensim_folder:
        so = os.path.join(a.opensim_folder, "jcf_output_sojr")
        f = next(os.path.join(so, x) for x in sorted(os.listdir(so)) if x.endswith("JointReaction_ReactionLoads.sto"))
        txt = open(f).read().split("\n"); n = next(i for i, l in enumerate(txt) if l.strip().lower() == "endheader")
        jr = pd.read_csv(f, sep=r"\s+", skiprows=n + 1)
        pre, bw = "walker_knee_r_on_tibia_r_in_tibia_r_", a.mass * 9.81
        g = np.abs(0.5 * np.abs(jr[pre + "fy"]) / bw - jr[pre + "mx"] / (bw * H) * H / D_M).to_numpy()
        keep = None
        mk = os.path.join(so, "label_mask.npz")
        if os.path.exists(mk):
            z = np.load(mk)
            # solver checks only: the fixed-COP rule reads the force's application point, and an optimizer export
            # applies its force at a fixed foot point with the heel-to-toe shift carried by the foot moment
            ok = ~(z["so_any_r"].astype(bool) | z["nonfinite_r"].astype(bool))
            keep = np.interp(jr.time, z["time"], ok.astype(float)) > 0.5
        gc = np.clip(g, 0, 3.8)
        if keep is None:
            ax.plot(jr.time, gc, color=AQUA, lw=1.4, zorder=2, label="OpenSim with muscle forces, -Mx (clipped at 3.8)")
        else:
            ax.plot(jr.time, np.where(keep, gc, np.nan), color=AQUA, lw=1.6, zorder=2,
                    label="OpenSim with muscle forces, -Mx (frames passing the solver checks)")
            ax.plot(jr.time, np.where(~keep, gc, np.nan), color=AQUA, lw=1.0, ls=(0, (1.5, 1.5)), zorder=2,
                    label="OpenSim, frames failing the solver checks (clipped at 3.8)")
        osim_rows = (jr.time.to_numpy(), g, keep)
    elif os.path.abspath(a.csv) == os.path.join(REPO, "robot_q_log.csv") and os.path.exists(ref):
        pf = pd.read_csv(ref)
        g = np.abs(pf.sojr_unusable_fmed.to_numpy())
        ax.plot(pf.t, np.clip(g, 0, 3.8), color="#b9b7b0", lw=3.0, zorder=2,
                label="OpenSim with muscle forces, -Mx (fails the label QC; clipped at 3.8)")
    for i, f in enumerate(seeds):
        ax.plot(t, f, color=SEED, lw=1.0, zorder=3, label="new model, 3 training runs (muscle forces, -Mx)" if i == 0 else None)
    ax.plot(t, mean, color=BLUE, lw=2.6, zorder=6, label="new model, mean of 3")
    ax.plot(t, cur_plus, color=INK, lw=1.8, ls=(0, (5, 2)), zorder=5, label="your current model (fpopt_w20_fmW0p5), +Mx as in ISRR")
    ax.plot(t, cur_minus, color=ORANGE, lw=1.8, zorder=4, label="your current model, correct sign (-Mx)")
    ax.set_xlim(t[0] - 0.1, t[-1] + 0.1)
    ax.set_ylim(0, 3.8)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("|F_medial| (BW)")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.13), ncol=2, fontsize=8.6, frameon=False, columnspacing=1.6)
    rows = []
    for n, (lo, hi) in enumerate(spans, 1):
        m = (t >= lo - 1e-9) & (t <= hi + 1e-9)
        osim_txt = ""
        if osim_rows is not None:
            mo = (osim_rows[0] >= lo - 1e-9) & (osim_rows[0] <= hi + 1e-9)
            osim_txt = f", OpenSim with muscles {osim_rows[1][mo].mean():.2f} (peak {osim_rows[1][mo].max():.2f})"
            if osim_rows[2] is not None:
                osim_txt += f" ({np.mean(osim_rows[2][mo]):.0%} pass the solver checks)"
        rows.append(f"stance {n} ({lo:.2f}-{hi:.2f} s): new {mean[m].mean():.2f} (runs "
                    f"{min(f[m].mean() for f in seeds):.2f}-{max(f[m].mean() for f in seeds):.2f}), current +Mx "
                    f"{cur_plus[m].mean():.2f}, current -Mx {cur_minus[m].mean():.2f}" + osim_txt)
        print(f"[{a.name}] {rows[-1]}; new peak {mean[m].max():.2f}, current +Mx peak {cur_plus[m].max():.2f}, "
              f">1.4 BW: new {np.mean(mean[m] > LIMIT):.0%}, current +Mx {np.mean(cur_plus[m] > LIMIT):.0%}")
    ax.text(0.0, -0.36, "Mean |F_medial| in right single stance (BW).\n  " + "\n  ".join(rows) + "\n"
            f"Inputs: pelvis angles converted from the URDF to OpenSim's convention, your exported COM and foot points, "
            f"contact from the exported foot wrenches; {a.mass:g} kg, {H:g} m.\n" +
            ("OpenSim with muscle forces is the label pipeline the new model was trained on, not measured ground truth: on "
             "the Grand Challenge implants it reads ~2x high at push-off.\n" if osim_rows is not None else
             "No ground truth on this motion. ") +
            "On the Grand Challenge implants the new model reads ~2x high at push-off and its loading peak comes ~80 ms late.",
            transform=ax.transAxes, ha="left", va="top", fontsize=8.6, color=SECOND, linespacing=1.5)
    ax.set_title(f"{a.name}: new model (muscle forces, correct sign) vs your current model "
                 f"(grey = right single stance, dashed = 1.4 BW)", fontsize=10.5)
    out = os.path.join(a.out, f"fig_{tag}_for_william.png")
    fig.savefig(out)
    print("wrote", out)


if __name__ == "__main__":
    main()

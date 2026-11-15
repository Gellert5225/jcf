#!/usr/bin/env python3
"""The plain per-output-loss models (muscle-force labels, no F_medial loss term; ..._t100_sojrqc_e100_s{1,2,3}) on
exported optimizer trajectories, through the deployment path (infer_trajectory_csv.py: robot -> OpenSim axes, pelvis
angles converted, contact from the foot wrenches, 20-frame windows), F_medial with the physical sign (-Mx). Each motion
also gets the quasi-static KAM (tools/kam_trajectory_csv.py), the old fpopt model (-Mx), and OpenSim with muscle forces
from the export's labelled folder under jcf/external (pipeline/label_trajectory_csv.py), when there is one.

Usage (repo root):
  python -I analysis/thesis/fig_plain_trajectories.py --csv robot_q_log.csv "robot_q_log (3).csv" \
      --names "Sep 23 reference" "Nov 14 export" --mass 75 --height 1.7
"""
import argparse
import io
import os
import runpy
import site
import sys
from contextlib import redirect_stdout

sys.path.append(site.getusersitepackages())
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "tools"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from kam_trajectory_csv import kam_series, runs  # noqa: E402

CKPT = "models/runs/mlp/best_model_mlp_walking_qinst_notrans_pelvisrel_winpos_contact_fpopt_w20_t100_sojrqc_e100_s{}.pt"
OLD = "models/runs/mlp/best_model_mlp_walking_qinst_notrans_pelvisrel_winpos_contact_fpopt_w20_fmW0p5_e100.pt"
D_M = 0.045
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, SECOND, MUTED, BAND, GRID = "#0b0b0b", "#52514e", "#8a8986", "#ecebe6", "#e4e3df"
SEED_BLUE = "#9dc1ee"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "legend.fontsize": 8,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "axes.linewidth": 0.6, "axes.edgecolor": SECOND,
    "xtick.color": SECOND, "ytick.color": SECOND, "text.color": INK, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 200, "savefig.bbox": "tight"})


def infer(csv, ckpt, mass, height, out_csv):
    """infer_trajectory_csv.py through its own CLI (the deployment path); returns its per-frame table."""
    argv = sys.argv
    sys.argv = ["infer_trajectory_csv.py", "--csv", csv, "--ckpt", ckpt, "--mass", str(mass), "--height", str(height),
                "--out", out_csv.replace(".csv", ".png"), "--out-csv", out_csv]
    try:
        with redirect_stdout(io.StringIO()):
            runpy.run_path(os.path.join(REPO, "infer_trajectory_csv.py"), run_name="__main__")
    finally:
        sys.argv = argv
    return pd.read_csv(out_csv)


def opensim_for(csv, mass, H):
    """OpenSim with the SO muscle forces for this export: the labelled folder under jcf/external whose metadata names
    this CSV (pipeline/label_trajectory_csv.py). Returns time, F_medial / compression / moment parts (-Mx, BW), the frames
    passing the SO checks of label_mask.npz (the fixed-COP rule is left out: optimizer exports apply the force at a
    fixed foot point and carry the heel-to-toe shift in the foot moment), and whether the pelvis angles were converted."""
    import glob
    import json
    for m in sorted(glob.glob(os.path.join(REPO, "jcf", "external", "*", "metadata.json"))):
        meta = json.load(open(m))
        d = os.path.dirname(m)
        if os.path.abspath(meta.get("source_csv", "")) != os.path.abspath(csv) or not os.path.isdir(f"{d}/jcf_output_sojr"):
            continue
        f = next((os.path.join(d, "jcf_output_sojr", x) for x in sorted(os.listdir(f"{d}/jcf_output_sojr"))
                  if x.endswith("JointReaction_ReactionLoads.sto")), None)
        if f is None:
            continue
        txt = open(f).read().split("\n")
        n = next(i for i, line in enumerate(txt) if line.strip().lower() == "endheader")
        jr = pd.read_csv(f, sep=r"\s+", skiprows=n + 1)
        pre, bw = "walker_knee_r_on_tibia_r_in_tibia_r_", mass * 9.81
        comp = 0.5 * np.abs(jr[pre + "fy"].to_numpy()) / bw
        mom = -jr[pre + "mx"].to_numpy() / (bw * D_M)
        keep = None
        if os.path.exists(f"{d}/jcf_output_sojr/label_mask.npz"):
            z = np.load(f"{d}/jcf_output_sojr/label_mask.npz")
            ok = ~(z["so_any_r"].astype(bool) | z["nonfinite_r"].astype(bool))
            keep = np.interp(jr.time, z["time"], ok.astype(float)) > 0.5
        return dict(t=jr.time.to_numpy(), fmed=comp + mom, comp=comp, mom=mom, keep=keep,
                    converted="pelvis_angles" in meta and "converted" in meta["pelvis_angles"], folder=d)
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--csv", nargs="+", required=True)
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--mass", type=float, default=75.0)
    ap.add_argument("--height", type=float, default=1.7)
    ap.add_argument("--out", default=os.path.join(REPO, "results", "thesis", "sign_ablation"))
    a = ap.parse_args()
    os.chdir(REPO)
    os.makedirs(a.out, exist_ok=True)
    H = a.height
    fig, axes = plt.subplots(3, len(a.csv), figsize=(7.2 * len(a.csv), 9.4), squeeze=False,
                             gridspec_kw=dict(hspace=0.45, wspace=0.16, height_ratios=[1.25, 1, 0.8]))
    for j, (csv, name) in enumerate(zip(a.csv, a.names)):
        tag = "".join(ch if ch.isalnum() else "_" for ch in name).strip("_").lower()
        seeds = []
        for s in (1, 2, 3):
            r = infer(csv, CKPT.format(s), a.mass, a.height, os.path.join(a.out, f"plain_s{s}_{tag}.csv"))
            r["fmed"] = 0.5 * r.fy.abs() - r.mx * H / D_M          # physical sign, whatever the checkpoint stores
            r["comp"] = 0.5 * r.fy.abs()
            r["mom"] = -r.mx * H / D_M
            seeds.append(r)
        t = seeds[0].time.to_numpy()
        mean = {c: np.mean([r[c].to_numpy() for r in seeds], axis=0) for c in ("fmed", "comp", "mom")}
        old = infer(csv, OLD, a.mass, a.height, os.path.join(a.out, f"old_fpopt_{tag}.csv"))
        old_f = (0.5 * old.fy.abs() - old.mx * H / D_M).to_numpy()
        osim = opensim_for(csv, a.mass, H)
        src = pd.read_csv(csv).dropna(subset=["com_position_0"]).reset_index(drop=True)
        k = kam_series(src, a.mass, a.height)
        cr, cl = k.contact_r.to_numpy().astype(bool), k.contact_l.to_numpy().astype(bool)
        single = cr & ~cl
        spans = [(k.time.iloc[x], k.time.iloc[y - 1]) for x, y in runs(single) if y - x >= 10]
        for ax in axes[:, j]:
            for lo, hi in spans:
                ax.axvspan(lo, hi, color=BAND, lw=0, zorder=0)
            ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
            ax.set_axisbelow(True)
            ax.set_xlim(k.time.iloc[0], k.time.iloc[-1])

        # row 1: F_medial
        ax = axes[0, j]
        ax.axhline(1.4, color=SECOND, lw=0.9, ls=(0, (4, 3)), zorder=1)
        for i, r in enumerate(seeds):
            ax.plot(t, r.fmed, color=SEED_BLUE, lw=1.0, zorder=3, label="seeds 1-3" if i == 0 else None)
        ax.plot(t, mean["fmed"], color=BLUE, lw=2.4, zorder=5, label="plain model, mean of 3")
        if osim is not None:
            pel = "pelvis converted" if osim["converted"] else "pelvis copied"
            g = np.clip(osim["fmed"], -0.3, 4.4)
            if osim["keep"] is None:
                ax.plot(osim["t"], g, color=AQUA, lw=1.3, zorder=2, label=f"OpenSim with muscle forces ({pel})")
            else:
                ax.plot(osim["t"], np.where(osim["keep"], g, np.nan), color=AQUA, lw=1.6, zorder=2,
                        label=f"OpenSim + muscles ({pel}), passes SO checks")
                ax.plot(osim["t"], np.where(~osim["keep"], g, np.nan), color=AQUA, lw=0.9, ls=(0, (1.5, 1.5)),
                        zorder=2, label="OpenSim, fails SO checks (clipped)")
        ax.plot(t, old_f, color=MUTED, lw=1.3, ls=(0, (1.5, 1.5)), zorder=2, label="old fpopt model (no muscles)")
        ax.set_ylim(-0.3, 6.2)
        ax.set_ylabel("F_medial, -Mx (BW)")
        ax.legend(loc="upper left", ncol=2, columnspacing=1.2, fontsize=7.6)
        ax.set_title(f"{name}: F_medial from the plain-loss models\n(shaded: right single stance; dashed: 1.4 BW)")

        # row 2: the two parts of F_medial (mean of seeds)
        ax = axes[1, j]
        ax.axhline(0, color=GRID, lw=0.8, zorder=1)
        ax.plot(t, mean["comp"], color=BLUE, lw=2.0, zorder=4, label="compression part 0.5|Fy|")
        ax.plot(t, mean["mom"], color=ORANGE, lw=2.0, zorder=4, label="moment part -Mx H/d")
        if osim is not None:
            kk = osim["keep"] if osim["keep"] is not None else np.ones(len(osim["t"]), bool)
            ax.plot(osim["t"], np.where(kk, osim["comp"], np.nan), color=BLUE, lw=1.0, ls=(0, (1.5, 1.5)), zorder=3,
                    label="OpenSim compression (passing frames)")
            ax.plot(osim["t"], np.where(kk, osim["mom"], np.nan), color=ORANGE, lw=1.0, ls=(0, (1.5, 1.5)), zorder=3,
                    label="OpenSim moment (passing frames)")
        ax.set_ylim(-1.2, 3.6)
        ax.set_ylabel("BW")
        ax.legend(loc="upper left", ncol=2)
        ax.set_title("Its two parts: model (solid, mean of 3) and OpenSim (dotted)")

        # row 3: quasi-static KAM, right knee
        ax = axes[2, j]
        ax.axhline(0, color=GRID, lw=0.8, zorder=1)
        ax.plot(k.time, 100 * k.kam_qs_r, color=INK, lw=1.6, zorder=4)
        ax.set_ylim(-1.5, 6.0)
        ax.set_ylabel("KAM (% BW·H)")
        ax.set_xlabel("time (s)")
        ax.set_title("Quasi-static KAM, right knee (from kinematics + contact; for comparison)")

        # per right single stance: plain-model mean / peak, and the KAM peak
        print(f"== {name} ({csv}): {len(t)} frames scored, {len(spans)} right single stances")
        for lo, hi in spans:
            m = (t >= lo - 1e-9) & (t <= hi + 1e-9)
            mk = (k.time >= lo - 1e-9) & (k.time <= hi + 1e-9)
            per = [r.fmed.to_numpy()[m].mean() for r in seeds]
            print(f"   {lo:.2f}-{hi:.2f} s: F_medial mean {mean['fmed'][m].mean():.2f} (seeds {np.round(per, 2)}), "
                  f"peak {mean['fmed'][m].max():.2f}, >1.4 BW {np.mean(mean['fmed'][m] > 1.4):.0%} | compression "
                  f"{mean['comp'][m].mean():.2f}, moment {mean['mom'][m].mean():+.2f} | KAM peak "
                  f"{100 * k.kam_qs_r[mk].max():.2f} %BW*H | old model {old_f[m].mean():.2f}"
                  + ("" if osim is None else
                     (lambda mo: f" | OpenSim {osim['fmed'][mo].mean():.2f} (compression {osim['comp'][mo].mean():.2f}, "
                                 f"moment {osim['mom'][mo].mean():+.2f}"
                                 + ("" if osim["keep"] is None else f"; {np.mean(osim['keep'][mo]):.0%} pass SO checks")
                                 + ")")((osim["t"] >= lo - 1e-9) & (osim["t"] <= hi + 1e-9))))
    fig.suptitle("Plain per-output-loss models (muscle-force labels) on William's exported trajectories",
                 x=0.07, ha="left", fontsize=12, fontweight="bold", y=0.995)
    out = os.path.join(a.out, "fig_plain_trajectories.png")
    fig.savefig(out)
    print("wrote", out)


if __name__ == "__main__":
    main()

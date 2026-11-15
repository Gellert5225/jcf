#!/usr/bin/env python3
"""Validation plots for the plain per-output-loss models (..._t100_sojrqc_e100_s{1,2,3}, mean of the three):
held-out test people (jcf/full_duration/testing/walking, one person per lab, never trained on) against their labels
(OpenSim JointReaction with the static-optimization muscle forces, jcf_output_sojr/), and the Grand Challenge implants
against the measured medial force. F_medial = 0.5|Fy| - Mx*H/0.045 m (physical sign) for labels and predictions alike.

Top row: a typical (median-error) and a worse (75th-percentile) Moore test trial and a typical trial of Han, a lab never
seen in training, label vs prediction over time; frames the label QC rejects are drawn dotted. Bottom row: predicted vs label
F_medial and |Fy| over every QC-kept right-stance frame of the test split; the implants' mean stance curves (JW, SC).

Usage (repo root): python -I analysis/thesis/fig_plain_validation.py [--out results/thesis/sign_ablation]
"""
import argparse
import glob
import json
import os
import site
import sys
from types import SimpleNamespace

sys.path.append(site.getusersitepackages())
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec  # noqa: E402

CKPT = "models/runs/mlp/best_model_mlp_walking_qinst_notrans_pelvisrel_winpos_contact_fpopt_w20_t100_sojrqc_e100_s{}.pt"
D_M = 0.045
BLUE, AQUA, INK, SECOND, MUTED, BAND, GRID = "#2a78d6", "#1baf7a", "#0b0b0b", "#52514e", "#8a8986", "#ecebe6", "#e4e3df"
LAB_NAMES = {"moore": "Moore", "carter": "Carter", "falisse": "Falisse", "han": "Han (lab not in training)",
             "fregly": "Fregly (lab not in training)", "tiziana": "Lencioni"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "legend.fontsize": 8,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "axes.linewidth": 0.6, "axes.edgecolor": SECOND,
    "xtick.color": SECOND, "ytick.color": SECOND, "text.color": INK, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 200, "savefig.bbox": "tight"})


def predict_test_split():
    from tools.evaluate_fmedial import features_for, load_model
    from train_mlp import INPUT_SETS
    models = [load_model(os.path.join(REPO, CKPT.format(s)), "cpu") for s in (1, 2, 3)]
    iset = models[0][3]["input_set"]
    cfg = dict(INPUT_SETS[iset], label_subdir="jcf_output_sojr")
    ns = SimpleNamespace(deploy_features=False, pelvis="trial", max_peak=1000.0, max_moment=1000.0, label_mask=True,
                         label_mask_subdir="jcf_output_sojr")
    out = []
    for d in sorted(glob.glob(os.path.join(REPO, "jcf", "full_duration", "testing", "walking", "*"))):
        try:
            built = features_for(d, cfg, ns)
        except (FileNotFoundError, ValueError):
            continue
        if built is None:
            continue
        feat, lab, meta = built
        H = float(json.load(open(os.path.join(d, "metadata.json")))["height_m"])
        x = torch.from_numpy(feat).float()
        with torch.no_grad():
            pr = np.mean([m((x - mu) / sd).numpy() for m, mu, sd, _i in models], axis=0)
        out.append(dict(trial=os.path.basename(d), lab=os.path.basename(d).split("_")[0], H=H,
                        t=np.asarray(meta["t"], float), stance=np.asarray(meta["r_contact"], bool),
                        keep=np.asarray(meta["label_keep"], bool), lab_fy=lab[:, 1], lab_mx=lab[:, 3],
                        pr_fy=pr[:, 1], pr_mx=pr[:, 3]))
    return out


def fmed(fy, mx, H):
    return 0.5 * np.abs(fy) - mx * H / D_M


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=os.path.join(REPO, "results", "thesis", "sign_ablation"))
    a = ap.parse_args()
    os.chdir(REPO)
    torch.set_num_threads(4)
    trials = predict_test_split()
    for r in trials:
        r["lab_f"], r["pr_f"] = fmed(r["lab_fy"], r["lab_mx"], r["H"]), fmed(r["pr_fy"], r["pr_mx"], r["H"])
        m = r["stance"] & r["keep"]
        r["mae"] = float(np.mean(np.abs(r["pr_f"][m] - r["lab_f"][m]))) if m.sum() >= 20 else np.nan
    df = pd.DataFrame([{k: r[k] for k in ("trial", "lab", "mae")} for r in trials])
    print(f"{len(trials)} test trials; with >= 20 kept stance frames: {df.mae.notna().sum()}")
    print("per lab: trials / median stance F_medial error (BW)\n",
          df.dropna().groupby("lab").mae.agg(["size", "median"]).round(3).to_string())

    fig = plt.figure(figsize=(15, 9.4))
    gs = GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.28, height_ratios=[1, 1.05])

    # top: representative trial per lab
    for j, (lab, q) in enumerate((("moore", 0.5), ("moore", 0.75), ("han", 0.5))):
        g = df[(df.lab == lab) & df.mae.notna()]
        if g.empty:
            continue
        name = g.iloc[(g.mae - g.mae.quantile(q)).abs().argmin()].trial
        r = next(x for x in trials if x["trial"] == name)
        t, keep = r["t"], r["keep"]
        span = min(3.0, t[-1] - t[0])
        cnt = [((t >= s) & (t <= s + span) & keep & r["stance"]).sum() for s in t[t <= t[-1] - span + 1e-9]]
        t0 = t[int(np.argmax(cnt))] if cnt else t[0]
        w = (t >= t0) & (t <= t0 + span)
        ax = fig.add_subplot(gs[0, j])
        on = r["stance"][w]
        e = np.flatnonzero(np.diff(np.r_[0, on.astype(int), 0]))
        for s0, s1 in zip(e[::2], e[1::2]):
            ax.axvspan(t[w][s0], t[w][min(s1, len(on)) - 1], color=BAND, lw=0, zorder=0)
        lab_f = r["lab_f"][w]
        ax.plot(t[w], np.where(keep[w], lab_f, np.nan), color=AQUA, lw=2.0, zorder=3,
                label="label: OpenSim with muscle forces")
        ax.plot(t[w], np.where(~keep[w], lab_f, np.nan), color=AQUA, lw=1.0, ls=(0, (1.5, 1.5)), zorder=2,
                label="label, rejected by QC (not trained on)")
        ax.plot(t[w], r["pr_f"][w], color=BLUE, lw=2.0, zorder=4, label="plain model (mean of 3)")
        ax.set_xlim(t[w][0], t[w][-1])
        ax.set_ylim(-0.5, max(5.0, float(np.nanmax(lab_f[keep[w]])) * 1.15 if keep[w].any() else 5.0))
        ax.set_xlabel("time (s)   [shaded: right foot on the ground]")
        ax.set_ylabel("F_medial (BW)")
        ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        if j == 0:
            ax.legend(loc="upper left", fontsize=7.8)
        which = "typical trial" if q == 0.5 else f"worse trial ({int(q * 100)}th pct)"
        ax.set_title(f"{'ABC'[j]}. Held-out {LAB_NAMES[lab].replace(' (lab not in training)', '')}, {which}\n"
                     f"    stance F_medial error {r['mae']:.2f} BW (lab median {g.mae.median():.2f})")
        ax.text(0.98, 0.97 if j else 0.62, name + ("\nlab not in training" if "not in training" in LAB_NAMES[lab] else ""),
                transform=ax.transAxes, ha="right", va="top", fontsize=7.6, color=SECOND)
        print(f"[{'ABC'[j]}] {lab}: {name}, error {r['mae']:.3f}, window {t0:.2f}-{t0 + span:.2f} s")

    # bottom left / middle: predicted vs label over all kept stance frames
    allf = {k: np.concatenate([x[k][x["stance"] & x["keep"]] for x in trials]) for k in ("lab_f", "pr_f", "lab_fy", "pr_fy")}
    for j, (lk, pk, lab, lim) in enumerate((("lab_f", "pr_f", "F_medial (BW)", 7), ("lab_fy", "pr_fy", "compression |Fy| (BW)", 9))):
        ax = fig.add_subplot(gs[1, j])
        xl, yp = np.abs(allf[lk]) if "fy" in lk else allf[lk], np.abs(allf[pk]) if "fy" in pk else allf[pk]
        hb = ax.hexbin(xl, yp, gridsize=45, extent=(0, lim, 0, lim), cmap="Blues", mincnt=1, bins="log", lw=0)
        ax.plot([0, lim], [0, lim], color=INK, lw=0.9, ls=(0, (4, 3)))
        r_ = np.corrcoef(xl, yp)[0, 1]
        mae = np.mean(np.abs(yp - xl))
        ax.text(0.03, 0.97, f"{len(xl):,} kept right-stance frames, {len(trials)} trials\n"
                            f"r = {r_:.2f}, mean error {mae:.2f} BW, bias {np.mean(yp - xl):+.2f} BW",
                transform=ax.transAxes, ha="left", va="top", fontsize=8.3, color=SECOND)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_aspect("equal")
        ax.set_xlabel(f"label {lab}")
        ax.set_ylabel(f"predicted {lab}")
        cb = fig.colorbar(hb, ax=ax, fraction=0.04, pad=0.02)
        cb.set_label("frames (log)", fontsize=8)
        ax.set_title(f"{'DE'[j]}. All held-out test people: predicted vs label\n    {lab} (dashed: perfect)")
        print(f"[{'DE'[j]}] {lab}: n {len(xl)}, r {r_:.3f}, MAE {mae:.3f}, bias {np.mean(yp - xl):+.3f}")

    # bottom right: implants, mean stance curves
    sub = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1, 2], wspace=0.12)
    fr = pd.read_csv(os.path.join(REPO, "results", "thesis", "gc", "loss_variants_gc", "frames.csv.gz"))
    st = fr[fr.stance & fr.pct.notna()].copy()
    st["bin"] = np.minimum(np.floor(st.pct / 5.0), 19) * 5 + 2.5
    st["plain"] = 0.5 * st[[f"fy_plain_s{i}" for i in (1, 2, 3)]].mean(axis=1).abs() + \
        st[[f"mx_plain_s{i}" for i in (1, 2, 3)]].mean(axis=1) * st.H / D_M      # frames store the medial-positive Mx
    st["osim"] = 0.5 * st.fy_osim_sojr.abs() + st.mx_osim_sojr * st.H / D_M
    axes = []
    for k, (subj, title) in enumerate((("JW", "JW, right knee"), ("SC", "SC, left knee"))):
        ax = fig.add_subplot(sub[k], sharey=axes[0] if axes else None)
        axes.append(ax)
        b = st[st.subject == subj].groupby("bin")[["meas_med", "plain", "osim"]].mean()
        ax.plot(b.index, b.meas_med, color=INK, lw=2.4, zorder=5, label="implant (measured)")
        ax.plot(b.index, b.osim, color=AQUA, lw=1.4, zorder=3, label="OpenSim with muscles")
        ax.plot(b.index, b.plain, color=BLUE, lw=2.0, zorder=4, label="plain model")
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 50, 100])
        ax.set_xlabel("% of stance")
        ax.text(0.04, 0.97, title, transform=ax.transAxes, ha="left", va="top", fontsize=8.3)
        ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        if k:
            ax.tick_params(labelleft=False)
    axes[0].set_ylabel("medial force (BW)")
    axes[0].set_ylim(-0.3, 5.0)
    axes[0].legend(loc="upper left", bbox_to_anchor=(0, 0.9), fontsize=7.6)
    axes[0].set_title("F. Real knees (implants), mean stance:\n    early peak late, late peak ~2x high")

    fig.suptitle("Validation of the plain per-output-loss models (mean of 3 seeds): held-out people vs OpenSim labels, "
                 "and real knees", x=0.07, ha="left", fontsize=12, fontweight="bold", y=0.995)
    out = os.path.join(a.out, "fig_plain_validation.png")
    os.makedirs(a.out, exist_ok=True)
    fig.savefig(out)
    print("wrote", out)


if __name__ == "__main__":
    main()

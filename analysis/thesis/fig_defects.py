#!/usr/bin/env python3
"""Explanatory figures for the four audit defects D1-D4 (one PNG per defect, plain-language titles).

Plots only: every number drawn or printed here comes from files the audit already wrote (results/thesis/...) or from
the trial folders under jcf/full_duration/. Where a panel restates a paper number, the script recomputes it and
prints it so it can be checked against the paper.

    D1  fig_d1_no_muscle.png   stored labels have no muscle force (vasti bound, all trials, relabel by lab)
        fig_d1_fixed.png       the same after the SO+JR rerun (one trial, JW/DM implants)
    D2  fig_d2_sign.png        medial-positive moment is -mx on the right knee (geometry, Kaneda et al.'s condyle
                               forces, legacy model vs implant under both signs, Carter's lateral early moment)
        fig_d2_fixed.png       physical sign: labels, retrained and old model against the implants
    D3  fig_d3_inflated.png    SO+JR labels too high (fixed COP, unconverged SO, label peaks vs implant peaks)
        fig_d3_fixed.png       the QC mask, what it leaves, retrained model vs implant peaks
    D4  fig_d4_timing.png      legacy windows time-warped and of lab-dependent length
        fig_d4_fixed.png       *_t100 inputs: no drift, 200 ms windows

Usage (repo root): python -I analysis/thesis/fig_defects.py [--out results/thesis/defects] [--only d1 d2fix ...]
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(REPO, "results", "thesis")
TRIALS = os.path.join(REPO, "jcf", "full_duration")
GC_ROOT = os.path.join(REPO, "jcf", "external", "grand_challenge")
G = 9.81
PHI, D_M = 0.5, 0.045

# Reference palette, first three slots (validated all-pairs): blue = corrected, orange = old (stored labels / legacy
# model), aqua = SO+JR labels. Black = measured truth; grey = reference lines.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, SECOND, MUTED, BAND, GRID = "#0b0b0b", "#52514e", "#8a8986", "#f1f0ec", "#e4e3df"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "legend.fontsize": 8.5, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "axes.linewidth": 0.6,
    "axes.edgecolor": SECOND, "xtick.color": SECOND, "ytick.color": SECOND, "axes.labelcolor": INK,
    "text.color": INK, "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 200, "savefig.bbox": "tight",
})


def read_sto(path):
    with open(path) as f:
        for n, line in enumerate(f):
            if line.strip().lower() == "endheader":
                break
    return pd.read_csv(path, sep=r"\s+", skiprows=n + 1)


def trial_dir(name):
    hits = glob.glob(os.path.join(TRIALS, "*", "*", name))
    if not hits:
        raise FileNotFoundError(name)
    return hits[0]


def mass_of(folder):
    return float(json.load(open(os.path.join(folder, "metadata.json")))["mass_kg"])


def grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)


def note(ax, text, xy=(0.02, 0.97), ha="left", va="top", size=8.3, color=SECOND):
    ax.text(*xy, text, transform=ax.transAxes, ha=ha, va=va, fontsize=size, color=color, linespacing=1.35)


def stance_spans(t, on):
    """(start, end) times of the runs where on is True."""
    on = np.asarray(on, bool)
    edges = np.flatnonzero(np.diff(np.r_[0, on.astype(int), 0]))
    return [(t[a], t[min(b, len(t)) - 1]) for a, b in zip(edges[::2], edges[1::2])]


def knee_fy_bw(folder, sub, mass):
    j = read_sto(os.path.join(folder, sub, "BatchJCF_JointReaction_ReactionLoads.sto"))
    return j["time"].to_numpy(), np.abs(j["walker_knee_r_on_tibia_r_in_tibia_r_fy"].to_numpy()) / (mass * G)


# ─── D1 ──────────────────────────────────────────────────────────────────────

def fig_d1(out):
    fig = plt.figure(figsize=(14, 4.3))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.25, 1, 1.1], wspace=0.34)

    # A: one Moore training trial
    ax = fig.add_subplot(gs[0])
    name, t_lo, t_hi = "carter_P003_split0_t04", 2.0, 4.5
    f = trial_dir(name)
    m = mass_of(f)
    so = read_sto(os.path.join(f, "jcf_output", "BatchJCF_StaticOptimization_force.sto"))
    vas = so[["vasint_r", "vaslat_r", "vasmed_r"]].sum(axis=1).to_numpy() / (m * G)
    t_st, fy_st = knee_fy_bw(f, "jcf_output", m)
    grf = read_sto(os.path.join(f, "grf_data.mot"))
    for a, b in stance_spans(grf["time"].to_numpy(), grf["calcn_r_force_vy"].to_numpy() > 0.05 * m * G):
        ax.axvspan(a, b, color=BAND, lw=0, zorder=0)
    sel = (t_st >= t_lo) & (t_st <= t_hi)
    vsel = (so["time"] >= t_lo) & (so["time"] <= t_hi)
    so, vas, t_st, fy_st = so[vsel], vas[vsel.to_numpy()], t_st[sel], fy_st[sel]
    ax.plot(so["time"], vas, color=MUTED, lw=1.6, ls=(0, (4, 2)), label="vasti pull alone (from step 1)", zorder=3)
    ax.plot(t_st, fy_st, color=ORANGE, lw=2.2, label="knee force as stored (your training labels)", zorder=5)
    ax.set_xlim(t_st[0], t_st[-1])
    ax.set_ylim(0, 5.3)
    ax.set_xlabel("time (s)   [shaded = right foot on the ground]")
    ax.set_ylabel("force (body weights)")
    grid(ax)
    ax.legend(loc="upper left", bbox_to_anchor=(0, 1.0), handlelength=2.2)
    ax.set_title("A. One Carter training trial: the stored knee force is far\n    below the pull of one muscle group (the vasti) alone")
    print(f"[D1-A] {name}: stored max {fy_st.max():.2f} BW, vasti max {vas.max():.2f} BW")

    # B: every walking trial with strong vasti frames
    ax = fig.add_subplot(gs[1])
    a = pd.read_csv(os.path.join(RES, "gc", "knee_jr", "stored_check_all_walking.csv"))
    v = a[a.vas_r_frames > 0]
    below_peak = float((v.stored_r_F_max_bw < v.so_vas_r_max_bw).mean())
    below_frames = float((v.stored_below_vas_r_frac >= 0.5).mean())
    lim = 10
    inside = (v.so_vas_r_max_bw <= lim) & (v.stored_r_F_max_bw <= lim)
    ax.scatter(v.so_vas_r_max_bw[inside], v.stored_r_F_max_bw[inside], s=5, color=ORANGE,
               alpha=0.25, lw=0, zorder=3, rasterized=True)
    ax.plot([0, lim], [0, lim], color=INK, lw=1, zorder=4)
    ax.text(lim * 0.55, lim * 0.57, "knee force = vasti pull", rotation=45, color=INK, fontsize=8.3,
            ha="center", va="bottom", rotation_mode="anchor")
    ax.fill_between([0, lim], [0, lim], 0, color=BAND, zorder=0, lw=0)
    ax.text(lim * 0.70, lim * 0.28, "physically impossible\nif the muscles were included", ha="center",
            fontsize=8.3, color=SECOND)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("peak vasti pull in the trial (BW)")
    ax.set_ylabel("peak stored knee force in the trial (BW)")
    grid(ax, "both")
    note(ax, f"one dot = one trial\nstored peak < vasti peak: {below_peak:.0%}\n"
             f"({int((~inside).sum())} trials beyond 10 BW\n not drawn)", xy=(0.03, 0.97))
    ax.set_title("B. All 8,037 walking trials with strong\n    vasti frames: the same, almost everywhere")
    print(f"[D1-B] {len(v)} trials; stored peak < vasti peak {below_peak:.3f}; "
          f"stored below vasti in >=50% of strong-vasti frames {below_frames:.3f}")

    # C: training-pool stance peaks, stored vs SO+JR, by lab
    ax = fig.add_subplot(gs[2])
    p = pd.read_csv(os.path.join(RES, "labels_qc", "per_trial.csv"), low_memory=False)
    w = p[(p.split == "training") & (p.activity == "walking") & p.lab.isin(["carter", "moore", "tiziana", "falisse"])]
    labs = [("carter", "Carter"), ("moore", "Moore"), ("tiziana", "Lencioni"), ("falisse", "Falisse")]
    ax.axhspan(2.2, 3.0, color=BAND, lw=0, zorder=0)
    ax.text(4.45, 2.6, "real knees\n(implants)", va="center", ha="left", fontsize=8, color=SECOND)
    for i, (lab, title) in enumerate(labs):
        g = w[w.lab == lab]
        for k, (col, c, dx) in enumerate((("stored_r_fy_stance_peak", ORANGE, -0.17),
                                         ("sojr_r_fy_stance_peak", AQUA, 0.17))):
            x = g[col].dropna().to_numpy()
            q5, q25, q50, q75, q95 = np.percentile(x, [5, 25, 50, 75, 95])
            ax.add_patch(Rectangle((i + dx - 0.13, q25), 0.26, q75 - q25, color=c, lw=0, zorder=3))
            ax.plot([i + dx, i + dx], [q5, q25], color=c, lw=1, zorder=2)
            ax.plot([i + dx, i + dx], [q75, q95], color=c, lw=1, zorder=2)
            ax.plot([i + dx - 0.13, i + dx + 0.13], [q50, q50], color=INK, lw=1.6, zorder=4)
            ax.text(i + dx, q95 + 0.25, f"{q50:.1f}", ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(range(len(labs)))
    ax.set_xticklabels([t for _, t in labs])
    ax.set_xlim(-0.55, 4.4)
    ax.set_ylim(0, 17.5)
    ax.set_ylabel("peak knee force in stance (BW)")
    grid(ax)
    ax.plot([], [], color=ORANGE, lw=6, label="stored labels (no muscles)")
    ax.plot([], [], color=AQUA, lw=6, label="rerun with the muscle file")
    ax.legend(loc="upper right", handlelength=1.2)
    ax.set_title("C. Rerun WITH the muscles: too low becomes\n    too high (number = median; box = middle half)")
    s_med = w.stored_r_fy_stance_peak.median()
    n_med = w.sojr_r_fy_stance_peak.median()
    print(f"[D1-C] training walking {len(w)} trials: median stance peak {s_med:.2f} -> {n_med:.2f} BW")

    fig.suptitle("D1 - The training labels were computed without the muscle forces", x=0.06, ha="left",
                 fontsize=12, fontweight="bold", y=1.04)
    fig.savefig(os.path.join(out, "fig_d1_no_muscle.png"))
    plt.close(fig)


def fig_d1_fixed(out):
    """Companion to fig_d1: the same views with the knee force recomputed from the SO muscle forces (SO+JR)."""
    fig = plt.figure(figsize=(10.5, 4.3))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.25, 1.1], wspace=0.24)

    # A: the D1-A trial and window, now with SO+JR
    ax = fig.add_subplot(gs[0])
    name, t_lo, t_hi = "carter_P003_split0_t04", 2.0, 4.5
    f = trial_dir(name)
    m = mass_of(f)
    so = read_sto(os.path.join(f, "jcf_output", "BatchJCF_StaticOptimization_force.sto"))
    vas = so[["vasint_r", "vaslat_r", "vasmed_r"]].sum(axis=1).to_numpy() / (m * G)
    t_st, fy_st = knee_fy_bw(f, "jcf_output", m)
    t_nw, fy_nw = knee_fy_bw(f, "jcf_output_sojr", m)
    grf = read_sto(os.path.join(f, "grf_data.mot"))
    for a0, b0 in stance_spans(grf["time"].to_numpy(), grf["calcn_r_force_vy"].to_numpy() > 0.05 * m * G):
        ax.axvspan(a0, b0, color=BAND, lw=0, zorder=0)
    vsel = ((so["time"] >= t_lo) & (so["time"] <= t_hi)).to_numpy()
    s_sel = (t_st >= t_lo) & (t_st <= t_hi)
    n_sel = (t_nw >= t_lo) & (t_nw <= t_hi)
    ax.plot(so["time"][vsel], vas[vsel], color=MUTED, lw=1.6, ls=(0, (4, 2)), label="vasti pull alone", zorder=3)
    ax.plot(t_st[s_sel], fy_st[s_sel], color=ORANGE, lw=1.6, label="before: stored label (no muscles)", zorder=4)
    ax.plot(t_nw[n_sel], fy_nw[n_sel], color=AQUA, lw=2.2, label="after: rerun with the muscle forces", zorder=5)
    ax.set_xlim(t_lo, t_hi)
    ax.set_ylim(0, 11)
    ax.set_xlabel("time (s)   [shaded = right foot on the ground]")
    ax.set_ylabel("knee force (body weights)")
    grid(ax)
    ax.legend(loc="upper left", bbox_to_anchor=(0, 1.0), handlelength=2.2)
    vi = np.interp(t_nw[n_sel], so["time"], vas)
    big = vi > 1.0
    ax.set_title("A. The same Carter trial, rerun with the muscle\n    forces: the knee force now exceeds the vasti pull")
    print(f"[D1fix-A] {name}: stored max {fy_st[s_sel].max():.2f}, SO+JR max {fy_nw[n_sel].max():.2f}, "
          f"vasti max {vas[vsel].max():.2f} BW; SO+JR below vasti in {np.mean(fy_nw[n_sel][big] < vi[big]):.0%} "
          f"of {int(big.sum())} frames with vasti > 1 BW")

    # B: implant subjects with stored labels (JW, DM; SC has none), mean stance in 5 % bins
    sub = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1], wspace=0.14)
    d = pd.read_csv(os.path.join(RES, "gc", "audit_gc", "frames.csv.gz"))
    s = d[d.stance & d.pct.notna() & d.subject.isin(["JW", "DM"])].copy()
    s["bin"] = np.minimum(np.floor(s.pct / 5.0), 19) * 5.0 + 2.5
    s["stored"] = s.fy_osim_state.abs()
    s["sojr"] = s.fy_osim_sojr.abs()
    b = s.groupby(["subject", "bin"])[["meas_tot", "stored", "sojr"]].mean().reset_index()
    axes = []
    for k, (subj, title) in enumerate((("JW", "JW, right knee"), ("DM", "DM (timing uncertain)"))):
        ax = fig.add_subplot(sub[k], sharey=axes[0] if axes else None)
        axes.append(ax)
        g = b[b.subject == subj]
        ax.plot(g.bin, g.meas_tot, color=INK, lw=2.2, label="real knee (implant)", zorder=5)
        ax.plot(g.bin, g.stored, color=ORANGE, lw=1.6, label="before (no muscles)", zorder=4)
        ax.plot(g.bin, g.sojr, color=AQUA, lw=2.0, label="after (with muscles)", zorder=4)
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 50, 100])
        ax.get_xticklabels()[0].set_horizontalalignment("left")
        ax.get_xticklabels()[-1].set_horizontalalignment("right")
        ax.set_xlabel("% of stance")
        ax.text(0.03, 0.97, title, transform=ax.transAxes, ha="left", va="top", fontsize=8.3, color=INK)
        grid(ax)
        if k:
            ax.tick_params(labelleft=False)
        gs_ = s[s.subject == subj]
        print(f"[D1fix-B] {subj}: {len(gs_)} stance frames; mean of binned peak: implant {g.meas_tot.max():.2f}, "
              f"stored {g.stored.max():.2f}, SO+JR {g.sojr.max():.2f} BW")
    axes[0].set_ylabel("knee force (BW)")
    axes[0].set_ylim(0, 8.2)
    axes[0].legend(loc="upper left", bbox_to_anchor=(0, 0.92), handlelength=1.6, fontsize=8)
    axes[0].set_title("B. Against real knee implants: with the muscles\n    the force is too high instead of too low (D3)")

    fig.suptitle("D1 fixed - the knee force recomputed with the muscle forces", x=0.06,
                 ha="left", fontsize=12, fontweight="bold", y=1.04)
    fig.savefig(os.path.join(out, "fig_d1_fixed.png"))
    plt.close(fig)


# ─── D2 ──────────────────────────────────────────────────────────────────────

def _schematic(ax):
    """Right knee seen from behind: tibial z axis to the subject's right (= lateral)."""
    ax.set_xlim(-0.075, 0.075)
    ax.set_ylim(-0.078, 0.082)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(Rectangle((-0.045, -0.050), 0.09, 0.040, color="#dcdad3", lw=0))          # tibia
    ax.add_patch(Rectangle((-0.050, -0.012), 0.10, 0.008, color="#bdbbb3", lw=0))          # plateau
    ax.text(0, -0.031, "shin bone (tibia)\nright leg, seen from behind", ha="center", va="center", fontsize=8,
            color=SECOND)
    z_med, z_lat = -0.022, 0.031                       # medial / lateral knee-muscle points, scaled models
    ax.add_patch(FancyArrowPatch((z_med, 0.040), (z_med, -0.003), arrowstyle="-|>", mutation_scale=16, color=INK,
                                 lw=3.2))
    ax.add_patch(FancyArrowPatch((z_lat, 0.020), (z_lat, -0.003), arrowstyle="-|>", mutation_scale=11, color=INK,
                                 lw=1.6))
    ax.text(z_med - 0.006, 0.022, "inner\n(medial)", ha="right", va="center", fontsize=8.3)
    ax.text(z_lat + 0.006, 0.011, "outer\n(lateral)", ha="left", va="center", fontsize=8.3)
    ax.text(0, 0.080, "more push on the inner side\n= force at negative z\n= NEGATIVE $M_x$", ha="center",
            va="top", fontsize=8.3, fontweight="bold")
    ax.add_patch(FancyArrowPatch((-0.040, -0.060), (0.068, -0.060), arrowstyle="-|>", mutation_scale=12,
                                 color=BLUE, lw=1.4))
    ax.text(0.068, -0.065, "+z  (subject's right)", ha="right", va="top", fontsize=8, color=BLUE)
    ax.set_title("A. Geometry: on the right knee, 'inner'\n    is the minus side of OpenSim's z axis")


def _kaneda(ax, frames, subj, side, title):
    g = frames[(frames.subject == subj) & ((frames.f_med + frames.f_lat) > 200.0)]
    x = (g.f_med - g.f_lat).to_numpy() / 1000.0
    y = g.mx.to_numpy()
    r = float(np.corrcoef(x, y)[0, 1])
    ax.scatter(x, y, s=3, color=MUTED, alpha=0.25, lw=0, rasterized=True, zorder=2)
    k, c = np.polyfit(x, y, 1)
    xx = np.array([x.min(), x.max()])
    ax.plot(xx, k * xx + c, color=INK, lw=1.6, zorder=3)
    ax.axhline(0, color=GRID, lw=0.8, zorder=1)
    ax.set_xlabel("inner minus outer contact force (kN)\n[measured by their knee model]")
    ax.set_ylabel("OpenSim $M_x$ (N·m)")
    grid(ax, "both")
    note(ax, f"r = {r:+.2f}", xy=(0.97, 0.97), ha="right", size=10, color=INK)
    ax.set_title(title)
    return r


def _bin_stance(s, cols, bin_pct=5.0):
    s = s.copy()
    s["bin"] = np.minimum(np.floor(s.pct / bin_pct), 100 / bin_pct - 1) * bin_pct + bin_pct / 2
    return s.groupby("bin")[cols].mean().reset_index()


def fig_d2(out):
    import sys
    sys.path.insert(0, os.path.join(REPO, "analysis", "thesis"))
    from knee_mx_sign import kaneda_frames                       # noqa: E402  (our own reader; data are files)

    fig = plt.figure(figsize=(14, 9.2))
    gs = GridSpec(2, 6, figure=fig, hspace=0.55, wspace=0.9, height_ratios=[1, 1])
    _schematic(fig.add_subplot(gs[0, 0:2]))

    kf = kaneda_frames(GC_ROOT)
    r_jw = _kaneda(fig.add_subplot(gs[0, 2:4]), kf, "JW", "r", "B. Real data, RIGHT knee (JW): more inner\n"
                                                               "    push -> more negative $M_x$")
    r_sc = _kaneda(fig.add_subplot(gs[0, 4:6]), kf, "SC", "l", "C. LEFT knee (SC): the opposite slope,\n"
                                                               "    as the geometry predicts")
    r_dm = float(np.corrcoef(*(lambda g: ((g.f_med - g.f_lat), g.mx))(
        kf[(kf.subject == "DM") & ((kf.f_med + kf.f_lat) > 200.0)]))[0, 1])
    print(f"[D2-BC] Kaneda r(M_x, F_med - F_lat): JW {r_jw:+.2f}, DM {r_dm:+.2f}, SC {r_sc:+.2f}")

    # D: legacy model vs implant, JW, both readings
    ax = fig.add_subplot(gs[1, 0:3])
    d = pd.read_csv(os.path.join(RES, "gc", "audit_gc", "frames.csv.gz"))
    s = d[d.stance & d.pct.notna()].copy()
    s["meas"] = s.meas_med
    s["phys"] = PHI * np.abs(s.fy_fpopt_legacy) + s.mx_fpopt_legacy * s.H / D_M          # mx stored medial-positive
    s["inv"] = PHI * np.abs(s.fy_fpopt_legacy) - s.mx_fpopt_legacy * s.H / D_M
    rr = {k: (float(np.corrcoef(g.phys, g.meas)[0, 1]), float(np.corrcoef(g.inv, g.meas)[0, 1]))
          for k, g in d[d.stance].assign(
              meas=lambda x: x.meas_med,
              phys=lambda x: PHI * np.abs(x.fy_fpopt_legacy) + x.mx_fpopt_legacy * x.H / D_M,
              inv=lambda x: PHI * np.abs(x.fy_fpopt_legacy) - x.mx_fpopt_legacy * x.H / D_M).groupby("subject")}
    b = _bin_stance(s[s.subject == "JW"], ["meas", "phys", "inv"])
    ax.axhline(0, color=GRID, lw=0.8)
    ax.plot(b.bin, b.meas, color=INK, lw=2.6, label="real implant (measured)", zorder=5)
    ax.plot(b.bin, b.inv, color=ORANGE, lw=1.8, ls=(0, (4, 2)), label="old model, sign as in the code (+$M_x$)",
            zorder=4)
    ax.plot(b.bin, b.phys, color=ORANGE, lw=2.0, label="old model, correct sign (-$M_x$)", zorder=4)
    ax.set_xlim(0, 100)
    ax.set_ylim(-1.4, 2.6)
    ax.set_xlabel("% of stance (right foot on the ground)")
    ax.set_ylabel("inner-knee force (BW)")
    grid(ax)
    ax.legend(loc="upper right")
    note(ax, "correlation with the implant (JW / DM / SC)\n"
             f"  sign as in the code:  {rr['JW'][1]:+.2f} / {rr['DM'][1]:+.2f} / {rr['SC'][1]:+.2f}\n"
             f"  correct sign:          {rr['JW'][0]:+.2f} / {rr['DM'][0]:+.2f} / {rr['SC'][0]:+.2f}",
         xy=(0.02, 0.03), va="bottom")
    ax.set_title("D. The wrong sign made the old model LOOK right. With the\n"
                 "    correct sign it moves opposite to the real knee (JW shown)")
    print("[D2-D] legacy (optimizer pt) r physical / inverted: "
          + ", ".join(f"{k} {v[0]:+.2f}/{v[1]:+.2f}" for k, v in rr.items()))

    # E: Carter's early lateral moment
    ax = fig.add_subplot(gs[1, 3:6])
    prof = pd.read_csv(os.path.join(RES, "labels_qc", "mx_sign", "population_profile.csv"))
    pct = np.arange(5, 100, 10)
    cols = [f"pct{p}" for p in pct]
    h_med = 1.712                                     # median pool height (paper); moment term = M_x H / d
    to_bw = h_med / D_M / 1000.0                      # profile is medial-positive M_x in BW*H x 1000

    def row(lab):
        return prof[(prof.lab == lab) & (prof.source == "sojr") & (prof.side == "r")][cols].to_numpy()[0] * to_bw
    car, moo = row("carter"), row("moore")
    ax.axhline(0, color=GRID, lw=0.8)
    ax.axvspan(0, 40, color=BAND, lw=0, zorder=0)
    ax.plot(pct, moo, color=MUTED, lw=2.0, marker="o", ms=4, label="Moore labels, correct sign", zorder=3)
    ax.plot(pct, car, color=INK, lw=2.2, marker="o", ms=4, label="Carter labels, correct sign", zorder=4)
    ax.plot(pct, -car, color=INK, lw=1.6, ls=(0, (4, 2)), marker="o", ms=3.5, mfc="white",
            label="Carter labels, sign as in the code", zorder=4)
    ax.set_xlim(0, 100)
    ax.set_ylim(-2.4, 3.7)
    ax.set_xlabel("% of stance")
    ax.set_ylabel("moment part of inner-knee force (BW)")
    grid(ax)
    ax.legend(loc="upper right")
    ax.text(20, ax.get_ylim()[1] * 0.93, "Carter: pushes OUTWARD here\n(not physiological)", ha="center", va="top",
            fontsize=8.3, color=SECOND)
    ax.set_title("E. A second problem the sign was hiding: Carter's labels (80.6% of\n"
                 "    old training frames) push outward in early stance")
    print(f"[D2-E] Carter SO+JR moment term (BW), 5..95 %: {np.round(car, 2)}")

    fig.suptitle("D2 - The inner/outer direction of the knee moment was flipped for the right knee",
                 x=0.06, ha="left", fontsize=12, fontweight="bold", y=0.985)
    fig.savefig(os.path.join(out, "fig_d2_sign.png"))
    plt.close(fig)


# ─── D3 ──────────────────────────────────────────────────────────────────────

def fig_d3(out):
    fig = plt.figure(figsize=(14, 4.6))
    gs = GridSpec(1, 3, figure=fig, wspace=0.3, width_ratios=[1.15, 1, 1])

    # A: Carter P003 t05, fixed COP; top = knee load, bottom = positions
    sub = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0], height_ratios=[1, 1.6], hspace=0.12)
    c = pd.read_csv(os.path.join(RES, "labels_qc", "carter_P003_t05_cop_vs_foot.csv"))
    c = c[c.side == "r"]
    f = trial_dir("carter_P003_split0_t05")
    m = mass_of(f)
    t, fy = knee_fy_bw(f, "jcf_output_sojr", m)
    keep = (t >= c.time.min()) & (t <= c.time.max())
    ax0 = fig.add_subplot(sub[0])
    ax0.plot(t[keep], fy[keep], color=AQUA, lw=2)
    i = int(np.argmax(np.where(keep, fy, -1)))
    ax0.annotate(f"{fy[i]:.1f} BW", (t[i], fy[i]), xytext=(8, 4), textcoords="offset points", fontsize=8.3,
                 ha="left", va="bottom")
    ax0.set_ylabel("knee force\n(BW)")
    ax0.set_ylim(0, 13)
    ax0.set_xlim(c.time.min(), c.time.max())
    ax0.tick_params(labelbottom=False)
    grid(ax0)
    ax0.set_title("A. Stuck pressure point (Carter P003, one right step)")
    ax = fig.add_subplot(sub[1], sharex=ax0)
    ax.fill_between(c.time, c.heel_x, c.toes_x, color="#dcdad3", lw=0, label="the foot (heel to toes)")
    ax.plot(c.time, c.cop_x, color=INK, lw=2.4, label="recorded pressure point (COP)")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("position along\nwalkway (m)")
    grid(ax)
    ax.legend(loc="upper right", fontsize=8)
    ax.text(c.time.iloc[1], c.cop_x.iloc[0] - 0.04, "behind\nthe heel", fontsize=8, va="top", color=SECOND)
    ax.text(c.time.iloc[-6], c.cop_x.iloc[0] + 0.04, "ahead of\nthe toes", fontsize=8, va="bottom", ha="center",
            color=SECOND)
    print(f"[D3-A] P003 t05 right stance {c.time.min():.3f}-{c.time.max():.3f} s: COP x "
          f"{c.cop_x.min():.4f}-{c.cop_x.max():.4f} m; foot moves heel {c.heel_x.iloc[0]:.3f}->{c.heel_x.iloc[-1]:.3f}"
          f"; SO+JR peak {fy[i]:.2f} BW at {t[i]:.3f} s")

    # B: unconverged SO, 149 random stance frames with a moving COP
    ax = fig.add_subplot(gs[1])
    r = pd.read_csv(os.path.join(RES, "labels_qc", "resolve_random_stance.csv"))
    r = r[~r.fixed_cop.astype(bool)]
    no, ok = r[r.nonopt.astype(bool)], r[~r.nonopt.astype(bool)]
    lim = 8
    ax.plot([0, lim], [0, lim], color=INK, lw=1, zorder=2, label="line: no change")
    ax.scatter(ok.fy_r, ok.fy_resolved_est, s=22, facecolor="white", edgecolor=MUTED, lw=1.1, zorder=3,
               label=f"solver had finished ({len(ok)})")
    ax.scatter(no.fy_r, no.fy_resolved_est, s=22, color=AQUA, edgecolor="white", lw=0.6, zorder=4,
               label=f"solver had NOT finished ({len(no)}, {len(no) / len(r):.1%})")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("knee force in the SO+JR label (BW)")
    ax.set_ylabel("knee force after re-solving properly (BW)")
    grid(ax, "both")
    ax.legend(loc="upper left")
    note(ax, f"medians: unfinished {no.fy_r.median():.2f} -> {no.fy_resolved_est.median():.2f} BW\n"
             f"when re-solved; finished: {ok.fy_r.median():.2f} BW", xy=(0.03, 0.76), ha="left", va="top")
    ax.set_title("B. Muscle solver stopped early: finishing it\n    shrinks the force")
    print(f"[D3-B] {len(r)} frames, non-optimal {len(no) / len(r):.3f}; fy {no.fy_r.median():.2f} (non-opt) "
          f"{ok.fy_r.median():.2f} (opt) -> re-solved {no.fy_resolved_est.median():.2f}")

    # C: label peaks vs implant peaks per covered stance window
    ax = fig.add_subplot(gs[2])
    d = pd.read_csv(os.path.join(RES, "gc", "audit_gc", "frames.csv.gz"))
    h = pd.read_csv(os.path.join(RES, "gc", "audit_gc", "halves.csv"))
    h = h[h.cover >= 0.9]
    rows = []
    for _, w in h.iterrows():
        g = d[(d.gc_key == w.gc_key) & (d.stance_id == w.stance) & (d.part == w.part)]
        rows.append(dict(meas=g.meas_tot.max(), sojr=np.abs(g.fy_osim_sojr).max(),
                         stored=np.abs(g.fy_osim_state).max() if g.osim_state_ok.all() else np.nan))
    W = pd.DataFrame(rows)
    lim = 7
    ax.plot([0, lim], [0, lim], color=INK, lw=1, zorder=2, label="line: label = implant")
    ax.scatter(W.meas, W.sojr, s=34, color=AQUA, edgecolor="white", lw=0.6, zorder=4,
               label=f"SO+JR labels ({len(W)} windows)")
    ws = W.dropna(subset=["stored"])
    ax.scatter(ws.meas, ws.stored, s=34, color=ORANGE, edgecolor="white", lw=0.6, zorder=4, marker="s",
               label=f"stored labels ({len(ws)} windows, right knees)")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, 9)
    ax.set_xlabel("peak force measured by the implant (BW)")
    ax.set_ylabel("peak force in the label (BW)")
    grid(ax, "both")
    ax.legend(loc="upper left")
    note(ax, f"SO+JR > implant: {int((W.sojr > W.meas).sum())}/{len(W)} ({np.median(W.sojr / W.meas):.2f}x)\n"
             f"stored < implant: {int((ws.stored < ws.meas).sum())}/{len(ws)} "
             f"({np.median(ws.stored / ws.meas):.2f}x)", xy=(0.99, 0.235), ha="right", va="center")
    ax.set_title("C. Against real knees: SO+JR about 2x too high,\n    stored labels about 0.4x (too low)")
    print(f"[D3-C] windows {len(W)}: SO+JR > implant {(W.sojr > W.meas).sum()}, median ratio "
          f"{np.median(W.sojr / W.meas):.2f}; stored < implant {(ws.stored < ws.meas).sum()}/{len(ws)}, "
          f"median {np.median(ws.stored / ws.meas):.2f}")

    fig.suptitle("D3 - Even with the muscles, the OpenSim labels are too high, for three data/solver reasons",
                 x=0.06, ha="left", fontsize=12, fontweight="bold", y=1.03)
    fig.savefig(os.path.join(out, "fig_d3_inflated.png"))
    plt.close(fig)


# ─── D4 ──────────────────────────────────────────────────────────────────────

def _grids(folder):
    """(exact JR-frame grid, legacy full-segment grid) as train_mlp builds them (_common_time_grid / linspace)."""
    ik = read_sto(os.path.join(folder, "ik_results.mot"))["time"].to_numpy()
    grf = read_sto(os.path.join(folder, "grf_data.mot"))["time"].to_numpy()
    jr = read_sto(os.path.join(folder, "jcf_output", "BatchJCF_JointReaction_ReactionLoads.sto"))["time"].to_numpy()
    t0, t1 = max(ik[0], grf[0], jr[0]), min(ik[-1], grf[-1], jr[-1])
    exact = ik[(ik >= t0) & (ik <= t1)]
    return exact, np.linspace(ik[0], ik[-1], len(exact)), ik


def fig_d4(out):
    fig = plt.figure(figsize=(14, 4.3))
    gs = GridSpec(1, 3, figure=fig, wspace=0.36, width_ratios=[1, 1.35, 0.9])

    # A: offset across a Carter trial and a Moore trial
    ax = fig.add_subplot(gs[0])
    car = trial_dir("carter_P003_split0_t04")
    moo = trial_dir("moore_subject13_t35_r09")         # typical Moore shift (median 0.17 s)
    for f, lab, col, lw in ((moo, "Moore trial", MUTED, 1.8), (car, "Carter trial", ORANGE, 2.2)):
        ex, leg, _ = _grids(f)
        ax.plot(ex, leg - ex, color=col, lw=lw, label=f"{lab} ({ex[-1] - ex[0]:.1f} s)")
        print(f"[D4-A] {os.path.basename(f)}: offset {np.min(leg - ex):+.3f} .. {np.max(leg - ex):+.3f} s, "
              f"mean |.| {np.mean(np.abs(leg - ex)):.3f} s")
    ax.axhline(0, color=INK, lw=1)
    ax.text(7.9, -0.012, "in sync", fontsize=8, color=INK, va="top", ha="right")
    ax.set_xlabel("time of the label (s)")
    ax.set_ylabel("time of the position input minus\ntime of the label (s)")
    ax.set_ylim(-0.36, 0.36)
    grid(ax)
    ax.legend(loc="upper left")
    ax.set_title("A. Inputs drift out of sync: up to 0.3 s\n    behind at the start, ahead at the end")

    # B: what the model saw - right heel height, legacy vs exact grid, with true stance
    ax = fig.add_subplot(gs[1])
    ex, leg, _ = _grids(car)
    z = np.load(os.path.join(car, "com_features.npz"))
    heel_ex = np.interp(ex, z["time"], z["calcn_r_pos"][:, 1])
    heel_leg = np.interp(leg, z["time"], z["calcn_r_pos"][:, 1])
    m = mass_of(car)
    grf = read_sto(os.path.join(car, "grf_data.mot"))
    on = np.interp(ex, grf["time"], grf["calcn_r_force_vy"]) > 0.05 * m * G
    win = (ex >= 0.35) & (ex <= 2.6)
    for a, b in stance_spans(ex[win], on[win]):
        ax.axvspan(a, b, color=BAND, lw=0, zorder=0)
    ax.plot(ex[win], heel_ex[win], color=BLUE, lw=2.0, label="what it should see (in sync, new)", zorder=4)
    ax.plot(ex[win], heel_leg[win], color=ORANGE, lw=2.0, ls=(0, (4, 2)), label="what the old model saw", zorder=5)
    ax.set_xlim(ex[win][0], ex[win][-1])
    ax.set_xlabel("time of the label (s)   [shaded = right foot really on the ground]")
    ax.set_ylabel("right heel height (m)")
    grid(ax)
    ax.set_ylim(0.03, 0.33)
    ax.legend(loc="upper center", ncol=2)
    ax.set_title("B. Example (start of the Carter trial): the old heel signal\n"
                 "    lags 0.1-0.3 s, so 'heel down' no longer matches stance")

    # C: window length per lab
    ax = fig.add_subplot(gs[2])
    labs = [("carter", "Carter"), ("tiziana", "Lencioni"), ("moore", "Moore"), ("falisse", "Falisse")]
    spans = []
    for lab, title in labs:
        rates = []
        for f in sorted(glob.glob(os.path.join(TRIALS, "training", "walking", f"{lab}*")))[:60]:
            p = os.path.join(f, "ik_results.mot")
            if os.path.exists(p):
                rates.append(1.0 / np.median(np.diff(read_sto(p)["time"].to_numpy())))
        hz = float(np.median(rates))
        spans.append((title, 20 / hz * 1000, hz))
        print(f"[D4-C] {title}: {hz:.0f} Hz -> 20 frames = {20 / hz * 1000:.0f} ms ({len(rates)} trials)")
    spans.append(("in use\n(suit, optimizer)", 200.0, 100.0))
    for i, (title, ms, hz) in enumerate(spans):
        last = i == len(spans) - 1
        ax.bar(i, ms, width=0.62, color=BLUE if last else MUTED, zorder=3)
        ax.text(i, ms + 4, f"{ms:.0f} ms\n({hz:.0f} Hz)", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(spans)))
    ax.set_xticklabels([s[0] + ("\n(80.6% of old\ntraining frames)" if s[0] == "Carter" else "") for s in spans])
    ax.set_ylim(0, 260)
    ax.set_ylabel("time covered by the 20-frame window (ms)")
    grid(ax)
    ax.set_title("C. 20 frames, different\n    time spans")

    fig.suptitle("D4 - The model's inputs were out of sync with its labels, and of varying length "
                 "(smallest effect of the four)", x=0.06, ha="left", fontsize=12, fontweight="bold", y=1.04)
    fig.savefig(os.path.join(out, "fig_d4_timing.png"))
    plt.close(fig)


# ─── "fixed" companions for D2-D4 ────────────────────────────────────────────

def fig_d2_fixed(out):
    """Medial force with the physical sign (frames.csv stores M_x medial-positive) against the implant on JW, DM, SC:
    SO+JR labels, the corrected model (mean F_y and M_x of qc_s1-3, trained with the physical sign) and the old fpopt
    model (trained with the inverted sign, read here with the physical one). r over all stance frames, as
    qc_keep/summary.csv; curves are 5 % stance bins of the frames with a stance %."""
    d = pd.read_csv(os.path.join(RES, "gc", "audit_gc", "frames.csv.gz"))

    def fm(fy, mx, h):
        return PHI * np.abs(fy) + mx * h / D_M
    d["meas"] = d.meas_med
    d["sojr"] = fm(d.fy_osim_sojr, d.mx_osim_sojr, d.H)
    d["new"] = fm(d[["fy_qc_s1", "fy_qc_s2", "fy_qc_s3"]].mean(axis=1),
                  d[["mx_qc_s1", "mx_qc_s2", "mx_qc_s3"]].mean(axis=1), d.H)
    d["old"] = fm(d.fy_fpopt_legacy, d.mx_fpopt_legacy, d.H)
    st = d[d.stance]
    cols = ["meas", "sojr", "new", "old"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), sharey=True, gridspec_kw=dict(wspace=0.07))
    panels = (("JW", "A. JW, right knee"), ("DM", "B. DM, right knee (implant timing uncertain)"),
              ("SC", "C. SC, left knee"))
    for ax, (subj, title) in zip(axes, panels):
        g = st[st.subject == subj]
        r = {c: float(np.corrcoef(g[c], g.meas)[0, 1]) for c in ("sojr", "new", "old")}
        bias = {c: float((g[c] - g.meas).mean()) for c in ("new", "old")}
        b = _bin_stance(g[g.pct.notna()], cols)
        ax.axhline(0, color=GRID, lw=0.8)
        ax.plot(b.bin, b.meas, color=INK, lw=2.6, label="real knee (implant)", zorder=6)
        ax.plot(b.bin, b.sojr, color=AQUA, lw=1.8, label="SO+JR labels, correct sign", zorder=4)
        ax.plot(b.bin, b.new, color=BLUE, lw=2.4, label="retrained model (learned the correct sign)", zorder=5)
        ax.plot(b.bin, b.old, color=ORANGE, lw=1.8, ls=(0, (4, 2)),
                label="old model (learned the flipped sign)", zorder=3)
        ax.set_xlim(0, 100)
        ax.set_xlabel("% of stance")
        grid(ax)
        ax.text(0.97, 0.03, f"correlation with the implant\nSO+JR labels:  {r['sojr']:+.2f}\n"
                f"retrained model:  {r['new']:+.2f}\nold model:  {r['old']:+.2f}", transform=ax.transAxes,
                ha="right", va="bottom", multialignment="right", fontsize=8.3, color=SECOND, linespacing=1.35)
        ax.set_title(title)
        print(f"[D2fix] {subj}: {len(g)} stance frames; r SO+JR {r['sojr']:+.3f}, retrained {r['new']:+.3f}, "
              f"old {r['old']:+.3f}; mean error retrained {bias['new']:+.2f}, old {bias['old']:+.2f} BW")
    axes[0].set_ylabel("inner-knee force (BW)")
    axes[0].set_ylim(-1.35, 3.75)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.06), handlelength=2.4)
    fig.suptitle("D2 fixed - with the correct sign, the labels and the retrained model rise and fall with the real "
                 "knee (still too high: D3)", x=0.06, ha="left", fontsize=12,
                 fontweight="bold", y=1.03)
    fig.savefig(os.path.join(out, "fig_d2_fixed.png"))
    plt.close(fig)


def fig_d3_fixed(out):
    """The D3 mitigation: the frame-level QC mask (label_mask.npz keep_r), what it leaves for training, and the
    model retrained on the kept frames against the implant peaks (same 30 windows as fig_d3 C)."""
    fig = plt.figure(figsize=(14, 4.5))
    gs = GridSpec(1, 3, figure=fig, wspace=0.3, width_ratios=[1.35, 0.85, 1])

    # A: one Moore trial (moving COP) with the frames the mask drops
    ax = fig.add_subplot(gs[0])
    name, t_lo, t_hi = "moore_subject5_t02_r03", 0.7, 3.65
    f = trial_dir(name)
    m = mass_of(f)
    t, fy = knee_fy_bw(f, "jcf_output_sojr", m)
    z = np.load(os.path.join(f, "jcf_output_sojr", "label_mask.npz"))
    assert len(z["time"]) == len(t) and np.allclose(z["time"], t)
    keep, stance = z["keep_r"].astype(bool), z["stance_r"].astype(bool)
    w = (t >= t_lo) & (t <= t_hi)
    for a0, b0 in stance_spans(t[w], stance[w]):
        ax.axvspan(a0, b0, color=BAND, lw=0, zorder=0)
    for k, y in enumerate((2.2, 3.0)):
        ax.axhline(y, color=INK, lw=0.8, ls=(0, (1, 2)), zorder=2,
                   label="real knees: peaks of 2.2-3 BW" if k else None)
    ax.plot(t[w], fy[w], color=MUTED, lw=1.1, label="dropped by the filter", zorder=3)
    ax.plot(t[w], np.where(keep[w], fy[w], np.nan), color=AQUA, lw=2.3, label="kept for training", zorder=4)
    ax.set_xlim(t_lo, t_hi)
    ax.set_ylim(0, 20)
    ax.set_xlabel("time (s)   [shaded = right foot on the ground]")
    ax.set_ylabel("SO+JR knee force (BW)")
    grid(ax)
    ax.legend(loc="upper left", ncol=2, columnspacing=1.2)
    ws = w & stance
    ax.set_title("A. The filter drops the frames where the solver failed\n"
                 "    (a Moore trial; the pressure point moves normally)")
    print(f"[D3fix-A] {name} {t_lo}-{t_hi} s: kept {keep[ws].mean():.0%} of stance frames; dropped max "
          f"{fy[w & ~keep].max():.1f} BW, kept max {fy[w & keep].max():.1f} BW; whole trial kept "
          f"{keep[stance].mean():.0%}")

    # B: share of training stance time left per lab
    ax = fig.add_subplot(gs[1])
    bl = pd.read_csv(os.path.join(RES, "labels_qc", "label_masks", "by_lab.csv"))
    bl = bl[(bl.split == "training") & (bl.activity == "walking")]
    rows = [("carter", "Carter"), ("tiziana", "Lencioni"), ("falisse", "Falisse"), ("moore", "Moore"),
            ("pool", "all four")]
    for i, (lab, title) in enumerate(rows):
        r = bl[bl.lab == lab].iloc[0]
        frac = float(r.kept_stance_frac_r)
        ax.bar(i, frac * 100, width=0.62, color=BLUE if lab == "pool" else AQUA, zorder=3)
        ax.text(i, frac * 100 + 1.2, f"{frac:.0%}" if frac >= 0.1 else f"{frac:.1%}", ha="center", va="bottom",
                fontsize=8.5)
        print(f"[D3fix-B] {title}: {int(r.trials)} trials, frozen COP {int(r.trials_fixed_cop_r)}, kept stance "
              f"{frac:.3f}")
    fc = {lab: bl[bl.lab == lab].iloc[0] for lab, _ in rows}
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([t for _, t in rows])
    note(ax, "trials with a frozen pressure point\n(all their frames dropped):\n"
             + "\n".join(f"  {t}: {int(fc[k].trials_fixed_cop_r):,} of {int(fc[k].trials):,}" for k, t in rows[:4]),
         xy=(0.03, 0.97))
    ax.set_ylim(0, 80)
    ax.set_ylabel("share of stance time kept (%)")
    grid(ax)
    ax.set_title("B. What is left to train on:\n    a quarter of stance, mostly Moore")

    # C: implant peak vs label / retrained-model peak, the fig_d3 C windows
    ax = fig.add_subplot(gs[2])
    d = pd.read_csv(os.path.join(RES, "gc", "audit_gc", "frames.csv.gz"))
    h = pd.read_csv(os.path.join(RES, "gc", "audit_gc", "halves.csv"))
    h = h[h.cover >= 0.9]
    rows = []
    for _, wnd in h.iterrows():
        g = d[(d.gc_key == wnd.gc_key) & (d.stance_id == wnd.stance) & (d.part == wnd.part)]
        new = np.abs(g[["fy_qc_s1", "fy_qc_s2", "fy_qc_s3"]].mean(axis=1))
        rows.append(dict(meas=g.meas_tot.max(), sojr=np.abs(g.fy_osim_sojr).max(), new=new.max()))
    W = pd.DataFrame(rows)
    lim = 7
    ax.plot([0, lim], [0, lim], color=INK, lw=1, zorder=2, label="line: = implant")
    ax.scatter(W.meas, W.sojr, s=26, facecolor="white", edgecolor=AQUA, lw=1.1, zorder=3,
               label=f"SO+JR labels ({np.median(W.sojr / W.meas):.1f}x)")
    ax.scatter(W.meas, W.new, s=34, color=BLUE, edgecolor="white", lw=0.6, zorder=4,
               label=f"retrained model ({np.median(W.new / W.meas):.1f}x)")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, 9)
    ax.set_xlabel("peak force measured by the implant (BW)")
    ax.set_ylabel("peak force predicted (BW)")
    grid(ax, "both")
    ax.legend(loc="upper left")
    note(ax, f"{len(W)} stance windows (JW, DM, SC)\nabove the implant: SO+JR {int((W.sojr > W.meas).sum())}, "
             f"retrained {int((W.new > W.meas).sum())}", xy=(0.97, 0.03), ha="right", va="bottom")
    ax.set_title("C. Against real knees the retrained model\n    is lower than its labels, still too high")
    print(f"[D3fix-C] {len(W)} windows: median ratio SO+JR {np.median(W.sojr / W.meas):.2f}, retrained "
          f"{np.median(W.new / W.meas):.2f}; above implant {(W.sojr > W.meas).sum()} / {(W.new > W.meas).sum()}")

    fig.suptitle("D3 mitigated, not solved - the bad frames are filtered out, but the knee force still reads high",
                 x=0.06, ha="left", fontsize=12, fontweight="bold", y=1.03)
    fig.savefig(os.path.join(out, "fig_d3_fixed.png"))
    plt.close(fig)


def _t100_grid(exact):
    """The *_t100 output grid (train_mlp._build_com_features with resample_hz=100 on the exact JR-frame grid)."""
    return exact[0] + np.arange(int(np.floor((exact[-1] - exact[0]) * 100.0 + 1e-9)) + 1) / 100.0


def fig_d4_fixed(out):
    """The *_t100 input sets: features and labels on one 100 Hz grid inside the JR window. Checked once on
    carter_P003_split0_t04 with train_mlp._build_com_features (right-calcaneus height channel against the calcaneus
    height at the label times): 0.0 m error for ..._w20_t100, up to 0.20 m for the legacy ..._w20."""
    fig = plt.figure(figsize=(14, 4.3))
    gs = GridSpec(1, 3, figure=fig, wspace=0.36, width_ratios=[1, 1.35, 0.9])
    car = trial_dir("carter_P003_split0_t04")
    moo = trial_dir("moore_subject13_t35_r09")

    # A: input-time minus label-time, old vs new
    ax = fig.add_subplot(gs[0])
    for f, lab, ls in ((moo, "Moore", (0, (1.5, 1.5))), (car, "Carter", (0, (4, 2)))):
        ex, leg, _ = _grids(f)
        ax.plot(ex, leg - ex, color=ORANGE, lw=1.5, ls=ls, label=f"before, {lab} trial", zorder=3)
    ax.axhline(0, color=BLUE, lw=2.6, label="after, every trial: 0 s", zorder=4)
    ax.set_xlabel("time of the label (s)")
    ax.set_ylabel("time of the position input minus\ntime of the label (s)")
    ax.set_ylim(-0.36, 0.36)
    grid(ax)
    ax.legend(loc="upper left")
    ax.set_title("A. Inputs and labels now share one\n    clock: no drift")

    # B: right heel height on the new 100 Hz grid against stance
    ax = fig.add_subplot(gs[1])
    ex, leg, _ = _grids(car)
    tn = _t100_grid(ex)
    zc = np.load(os.path.join(car, "com_features.npz"))
    heel_new = np.interp(tn, zc["time"], zc["calcn_r_pos"][:, 1])
    heel_old_err = np.abs(np.interp(leg, zc["time"], zc["calcn_r_pos"][:, 1])
                          - np.interp(ex, zc["time"], zc["calcn_r_pos"][:, 1]))
    m = mass_of(car)
    grf = read_sto(os.path.join(car, "grf_data.mot"))
    on = np.interp(tn, grf["time"], grf["calcn_r_force_vy"]) > 0.05 * m * G
    win = (tn >= 0.35) & (tn <= 2.6)
    for a0, b0 in stance_spans(tn[win], on[win]):
        ax.axvspan(a0, b0, color=BAND, lw=0, zorder=0)
    ax.plot(tn[win], heel_new[win], color=BLUE, lw=2.0, marker="o", ms=2.2, label="what the model sees now (100 Hz)",
            zorder=4)
    ax.set_xlim(tn[win][0], tn[win][-1])
    ax.set_ylim(0.03, 0.33)
    ax.set_xlabel("time of the label (s)   [shaded = right foot really on the ground]")
    ax.set_ylabel("right heel height (m)")
    grid(ax)
    ax.legend(loc="upper center")
    ax.set_title("B. Same example as before: the heel signal now\n    lines up with the foot's ground contact")
    print(f"[D4fix-B] Carter t04: legacy heel-height input off by up to {heel_old_err.max() * 100:.1f} cm "
          f"(mean {heel_old_err.mean() * 100:.1f} cm); t100 grid {len(tn)} frames at 100 Hz")

    # C: window span per lab, before and after
    ax = fig.add_subplot(gs[2])
    spans = _lab_spans()
    for i, (title, ms_old, hz) in enumerate(spans):
        ax.bar(i - 0.19, ms_old, width=0.36, color=MUTED, zorder=3, label="before" if i == 0 else None)
        ax.bar(i + 0.19, 200.0, width=0.36, color=BLUE, zorder=3, label="after" if i == 0 else None)
        ax.text(i - 0.19, ms_old + 4, f"{ms_old:.0f}", ha="center", va="bottom", fontsize=7.8, color=SECOND)
        ax.text(i + 0.19, 204, "200", ha="center", va="bottom", fontsize=7.8, color=INK)
    ax.axhline(200, color=INK, lw=0.8, ls=(0, (1, 2)), zorder=2)
    ax.set_xticks(range(len(spans)))
    ax.set_xticklabels([s[0] for s in spans])
    ax.set_ylim(0, 270)
    ax.set_ylabel("time covered by the 20-frame window (ms)")
    grid(ax)
    ax.legend(loc="upper left", ncol=2)
    ax.set_title("C. Every window now spans 200 ms,\n    as in the suit and the optimizer")

    fig.suptitle("D4 fixed - inputs rebuilt on the label clock at 100 Hz (changed the in-vivo error by at most "
                 "0.03 BW)", x=0.06, ha="left", fontsize=12, fontweight="bold", y=1.04)
    fig.savefig(os.path.join(out, "fig_d4_fixed.png"))
    plt.close(fig)


def _lab_spans(n_per_lab=60):
    """(lab title, ms spanned by 20 native-rate frames, native Hz) from the median IK step of up to n_per_lab trials."""
    out = []
    for lab, title in (("carter", "Carter"), ("tiziana", "Lencioni"), ("moore", "Moore"), ("falisse", "Falisse")):
        rates = []
        for f in sorted(glob.glob(os.path.join(TRIALS, "training", "walking", f"{lab}*")))[:n_per_lab]:
            p = os.path.join(f, "ik_results.mot")
            if os.path.exists(p):
                rates.append(1.0 / np.median(np.diff(read_sto(p)["time"].to_numpy())))
        hz = float(np.median(rates))
        out.append((title, 20 / hz * 1000, hz))
    return out


FIGS = (("d1", "fig_d1"), ("d1fix", "fig_d1_fixed"), ("d2", "fig_d2"), ("d2fix", "fig_d2_fixed"), ("d3", "fig_d3"),
        ("d3fix", "fig_d3_fixed"), ("d4", "fig_d4"), ("d4fix", "fig_d4_fixed"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=os.path.join(RES, "defects"))
    ap.add_argument("--only", nargs="*", choices=[k for k, _ in FIGS])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for k, fn in ((k, globals()[n]) for k, n in FIGS):
        if not a.only or k in a.only:
            fn(a.out)
    print("wrote", a.out)


if __name__ == "__main__":
    main()

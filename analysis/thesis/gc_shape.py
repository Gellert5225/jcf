#!/usr/bin/env python3
"""Stance shape on the Knee Grand Challenge: does each source reproduce the measured double hump of medial force?

Reads a finished analysis/thesis/grand_challenge.py run (results/thesis/gc/<run>/frames.csv.gz, halves.csv,
stances.csv); runs no model and no OpenSim. Every number is on identical frames for every source of a subject:

  frames      instrumented-limb stance frames inside a complete kinematic stance (pct defined) where the measured force
              and every source the subject has are finite. SC's frames are restricted further to Kaneda et al.'s
              windows in the `statop_overlap` variant, which adds their published SO+JR force.
  ensemble    mean F_medial per --bin-pct bin of stance, pooled over all segments. Segments are mostly half stances, so
              different bins come from different stances; the comparison between sources stays fair because they share
              the frames.
  humps       from each ensemble curve: maximum in the early (5-45 %) and the late (55-95 %) window, whether it is
              interior (not the window's first or last bin), the trough between the two maxima and its depth.
              double_hump = both maxima interior and the trough at least --min-depth (fraction of the smaller peak)
              below both.
  windows     per covered stance window (halves.csv, cover >= the run's --min-cover): within-window Pearson r of the
              source against the measured force (shape only, insensitive to bias), and from stances.csv the share of
              scored windows (interior measured maximum) where the source has an interior maximum, with its timing
              error in % of stance.

M_x is taken as the run stored it: medial-positive under the run's --mx-convention (run.json; runs without the key
used the inverted `pipeline` sign). The convention is written to run_info.json and the figure title.

Usage (any env with pandas + matplotlib):
    python -I analysis/thesis/gc_shape.py --run pilot_AB_sojr --out results/thesis/labels_qc/gc_sojr_shape
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from analysis.thesis.grand_challenge import OSIM_SOURCES, PEAK_WINDOWS, SUBJECTS, fmed_of  # noqa: E402

GC_OUT = os.path.join(REPO, "results", "thesis", "gc")
FRAME_COLS = {"subject", "folder", "gc_key", "trial", "gait", "k", "t_seg", "row", "gc_time", "H", "mass", "meas_med",
              "meas_lat", "meas_tot", "stance", "contact_input", "stance_id", "pct", "part"}


def frame_sources(fr):
    """Sources in a frames table, in display order: OpenSim labels, surrogates, then StatOp."""
    fy = [c[3:] for c in fr.columns if c.startswith("fy_")]
    osim = [s for s in OSIM_SOURCES if s in fy and fr[f"fy_{s}"].notna().any()]
    sur = [s for s in fy if s not in OSIM_SOURCES and fr[f"fy_{s}"].notna().any()]
    return osim + sur + (["statop"] if "fmed_statop" in fr and fr["fmed_statop"].notna().any() else [])


def shape_frames(fr, sources, phi=0.5, d_mm=45.0, with_statop=False):
    """Stance frames with pct, F_medial per source as f_<src>, restricted per subject to the frames where the measured
    force and every source that subject has are finite. StatOp joins only with with_statop (then only its frames)."""
    st = fr[fr["stance"].astype(bool) & fr["pct"].notna()].copy()
    use = [s for s in sources if with_statop or s != "statop"]
    for s in use:
        st[f"f_{s}"] = fmed_of(st, s, phi, d_mm)
    out = []
    for subj, g in st.groupby("subject", sort=False):
        have = [s for s in use if np.isfinite(g[f"f_{s}"]).any()]
        if with_statop and "statop" not in have:
            continue
        ok = np.isfinite(g["meas_med"].to_numpy(float))
        for s in have:
            ok &= np.isfinite(g[f"f_{s}"].to_numpy(float))
        gg = g[ok].copy()
        for s in set(use) - set(have):
            gg[f"f_{s}"] = np.nan
        out.append(gg)
    return pd.concat(out, ignore_index=True) if out else st.iloc[:0]


def ensemble(sf, sources, bin_pct=5.0, min_frames=10):
    """Mean F_medial per stance bin, per subject x source ('meas' = implant). Bins with < min_frames are NaN."""
    rows = []
    edges = np.arange(0.0, 100.0 + bin_pct / 2, bin_pct)
    for subj, g in sf.groupby("subject", sort=False):
        b = np.clip(np.digitize(g["pct"].to_numpy(float), edges) - 1, 0, len(edges) - 2)
        win = (g["folder"].astype(str) + "#" + g["stance_id"].astype(str)).to_numpy()
        for src in ["meas"] + list(sources):
            col = "meas_med" if src == "meas" else f"f_{src}"
            if col not in g or not np.isfinite(g[col]).any():
                continue
            v = g[col].to_numpy(float)
            for i in range(len(edges) - 1):
                m = b == i
                n = int(m.sum())
                rows.append(dict(subject=subj, source=src, bin_lo=edges[i], bin_mid=edges[i] + bin_pct / 2, n_frames=n,
                                 n_stances=len(set(win[m])),
                                 mean=float(v[m].mean()) if n >= min_frames else np.nan,
                                 sd=float(v[m].std()) if n >= min_frames else np.nan))
    return pd.DataFrame(rows)


def hump_features(mid, y, windows=PEAK_WINDOWS, min_depth=0.05):
    """Early / late maxima of a binned stance curve, whether each is interior to its window (not the first or last
    finite bin inside it), the trough between them and the double-hump verdict (both interior, trough at least
    min_depth x the smaller peak below both)."""
    mid, y = np.asarray(mid, float), np.asarray(y, float)
    out = {}
    idx = {}
    for name, lo, hi in windows:
        m = np.flatnonzero((mid >= lo) & (mid <= hi) & np.isfinite(y))
        if len(m) < 3:
            out.update({f"{name}_peak": np.nan, f"{name}_pct": np.nan, f"{name}_interior": False})
            idx[name] = None
            continue
        k = m[int(np.argmax(y[m]))]
        idx[name] = k
        out.update({f"{name}_peak": float(y[k]), f"{name}_pct": float(mid[k]),
                    f"{name}_interior": bool(m[0] < k < m[-1])})
    ke, kl = idx.get("early"), idx.get("late")
    trough = np.nan
    if ke is not None and kl is not None and kl - ke > 1:
        seg = y[ke + 1:kl]
        if np.isfinite(seg).any():
            j = ke + 1 + int(np.nanargmin(seg))
            trough = float(y[j])
            out["trough_pct"] = float(mid[j])
    out.setdefault("trough_pct", np.nan)
    out["trough"] = trough
    small = min(out.get("early_peak", np.nan), out.get("late_peak", np.nan))
    out["depth"] = small - trough if np.isfinite(trough) else np.nan
    out["depth_rel"] = out["depth"] / small if np.isfinite(out["depth"]) and small > 0 else np.nan
    out["late_early_ratio"] = out["late_peak"] / out["early_peak"] if out.get("early_peak", 0) else np.nan
    out["double_hump"] = bool(out["early_interior"] and out["late_interior"] and np.isfinite(out["depth_rel"])
                              and out["depth_rel"] >= min_depth)
    return out


def hump_table(ens, min_depth=0.05, variant="common"):
    """hump_features per subject x source plus the binned curve's r / MAE against the measured curve."""
    rows = []
    for subj, g in ens.groupby("subject", sort=False):
        meas = g[g.source == "meas"].sort_values("bin_mid")
        for src, gs in g.groupby("source", sort=False):
            gs = gs.sort_values("bin_mid")
            f = hump_features(gs["bin_mid"], gs["mean"], min_depth=min_depth)
            a, b = meas["mean"].to_numpy(float), gs["mean"].to_numpy(float)
            ok = np.isfinite(a) & np.isfinite(b)
            r = float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 2 and np.std(b[ok]) > 0 else np.nan
            rows.append(dict(variant=variant, subject=subj, source=src, n_bins=int(ok.sum()),
                             n_frames=int(gs["n_frames"].sum()), curve_r=r,
                             curve_mae=float(np.abs(a[ok] - b[ok]).mean()) if ok.any() else np.nan, **f))
    return pd.DataFrame(rows)


def window_shape(sf, halves, sources, min_frames=5):
    """Within-window Pearson r (source vs measured) on every covered stance window, one row per window x source."""
    cov = halves[halves["scored"].astype(bool)][["subject", "folder", "stance", "part"]].drop_duplicates()
    groups = {k: g for k, g in sf[sf["part"].isin([w[0] for w in PEAK_WINDOWS])].groupby(
        ["subject", "folder", "stance_id", "part"], sort=False)}
    rows = []
    for w in cov.itertuples(index=False):
        g = groups.get((w.subject, w.folder, int(w.stance), w.part))
        if g is None or len(g) < min_frames:
            continue
        a = g["meas_med"].to_numpy(float)
        for src in sources:
            b = g[f"f_{src}"].to_numpy(float) if f"f_{src}" in g else np.full(len(g), np.nan)
            if not np.isfinite(b).all():
                continue
            r = float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else np.nan
            rows.append(dict(subject=w.subject, folder=w.folder, stance=w.stance, part=w.part, source=src,
                             n_frames=len(g), r_within=r, bias=float((b - a).mean())))
    return pd.DataFrame(rows)


def window_summary(ws, st, sources):
    """Per subject x window x source: covered windows with a within-window r (median, share > 0), and on the scored
    windows (stances.csv, interior measured maximum) the share with an interior predicted maximum and the median
    timing error of that maximum (predicted - measured, % of stance; only where both are interior)."""
    rows = []
    parts = [w[0] for w in PEAK_WINDOWS]
    for subj in ws["subject"].unique() if not ws.empty else []:
        for part in parts:
            for src in sources:
                g = ws[(ws.subject == subj) & (ws.part == part) & (ws.source == src)]
                row = dict(subject=subj, part=part, source=src, n_covered=len(g),
                           r_within_median=float(g.r_within.median()) if len(g) else np.nan,
                           r_within_pos_frac=float((g.r_within > 0).mean()) if len(g) else np.nan)
                s = st[(st.subject == subj) & (st.part == part) & st["meas_interior"].astype(bool)] \
                    if not st.empty else st
                if f"peak_{src}" in s:
                    s = s[s[f"peak_{src}"].notna()]
                    both = s[s[f"interior_{src}"].astype(bool)]
                    row.update(n_scored=len(s),
                               interior_frac=float(s[f"interior_{src}"].astype(bool).mean()) if len(s) else np.nan,
                               timing_err_median_pct=float((both[f"peak_pct_{src}"] - both["meas_peak_pct"]).median())
                               if len(both) else np.nan,
                               timing_abs_err_median_pct=float((both[f"peak_pct_{src}"] - both["meas_peak_pct"]).abs()
                                                               .median()) if len(both) else np.nan)
                if len(g) or row.get("n_scored"):
                    rows.append(row)
    return pd.DataFrame(rows)


def make_figure(path, ens, sources, title_note=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # same entity colours as grand_challenge.make_figures (validated categorical slots 1-3, 7); measured = ink
    pal = ["#eb6834", "#1baf7a", "#e87ba4", "#eda100"]
    sur = [s for s in sources if s not in OSIM_SOURCES and s != "statop"]
    color = {"osim_sojr": "#2a78d6", "osim_state": "#86b6ef", "statop": "#6250d6", "meas": "#0b0b0b"}
    color.update({s: pal[i % len(pal)] for i, s in enumerate(sur)})
    dash = {"osim_state": (0, (4, 2)), "statop": (0, (1, 1.5))}
    name = {k: v[1] for k, v in OSIM_SOURCES.items()}
    name.update({"osim_sojr": "OpenSim SO+JR", "osim_state": "OpenSim stored labels",
                 "statop": "SO+JR (Kaneda et al.)", "meas": "measured (implant)"})
    name.update({s: f"surrogate {s}" for s in sur})
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6})
    subjects = [s for s in SUBJECTS if s in set(ens.subject)]
    fig, axs = plt.subplots(1, len(subjects), figsize=(3.3 * len(subjects), 2.7), squeeze=False, sharey=True)
    for ax, subj in zip(axs[0], subjects):
        g = ens[ens.subject == subj]
        for _, lo, hi in PEAK_WINDOWS:
            ax.axvspan(lo, hi, color="#f0efec", lw=0, zorder=0)
        for src in ["meas"] + list(sources):
            gs = g[g.source == src].sort_values("bin_mid")
            if gs.empty or not gs["mean"].notna().any():
                continue
            ax.plot(gs.bin_mid, gs["mean"], color=color.get(src, "#8a8985"), lw=2.2 if src == "meas" else 1.6,
                    ls=dash.get(src, "-"), label=name.get(src, src), zorder=3 if src == "meas" else 2)
        n = int(g[g.source == "meas"]["n_frames"].sum())
        ax.set_title(f"{subj} ({SUBJECTS[subj]['side'].upper()} knee), {n} frames", fontsize=8)
        ax.set_xlabel("% of stance")
        ax.set_xlim(0, 100)
    axs[0][0].set_ylabel("medial force (BW), bin mean")
    h, l_ = [], []
    for ax in axs[0]:
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in l_:
                h.append(hh)
                l_.append(ll)
    fig.legend(h, l_, loc="lower center", ncol=min(len(l_), 6), frameon=False, fontsize=7)
    if title_note:
        fig.suptitle(title_note, fontsize=8)
    fig.tight_layout(rect=(0, 0.1, 1, 0.95 if title_note else 1))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def run_mx_convention(run):
    """--mx-convention of a grand_challenge.py run; runs from before the option used the pipeline sign."""
    try:
        with open(os.path.join(run, "run.json")) as f:
            return json.load(f).get("args", {}).get("mx_convention", "pipeline")
    except (OSError, ValueError):
        return "unknown"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="results/thesis/gc/<run> directory or its name")
    ap.add_argument("--out", required=True)
    ap.add_argument("--phi", type=float, default=0.5)
    ap.add_argument("--d-mm", type=float, default=45.0)
    ap.add_argument("--bin-pct", type=float, default=5.0)
    ap.add_argument("--min-frames", type=int, default=10, help="frames a stance bin needs to get a mean")
    ap.add_argument("--min-depth", type=float, default=0.05, help="trough depth (fraction of the smaller peak)")
    args = ap.parse_args(argv)
    run = args.run if os.path.isdir(args.run) else os.path.join(GC_OUT, args.run)
    fr = pd.read_csv(os.path.join(run, "frames.csv.gz"), low_memory=False)
    halves = pd.read_csv(os.path.join(run, "halves.csv"))
    st = pd.read_csv(os.path.join(run, "stances.csv"))
    sources = frame_sources(fr)
    conv = run_mx_convention(run)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "run_info.json"), "w") as f:
        json.dump(dict(run=os.path.relpath(run, REPO), mx_convention=conv, phi=args.phi, d_mm=args.d_mm,
                       bin_pct=args.bin_pct, min_frames=args.min_frames, min_depth=args.min_depth), f, indent=2)
    tabs, ens_all, win_all, wsum_all = [], [], [], []
    for variant, with_statop in (("common", False), ("statop_overlap", True)):
        srcs = [s for s in sources if with_statop or s != "statop"]
        sf = shape_frames(fr, srcs, args.phi, args.d_mm, with_statop=with_statop)
        if sf.empty:
            continue
        ens = ensemble(sf, srcs, args.bin_pct, args.min_frames).assign(variant=variant)
        ens_all.append(ens)
        tabs.append(hump_table(ens, args.min_depth, variant))
        ws = window_shape(sf, halves, srcs).assign(variant=variant)
        win_all.append(ws)
        wsum_all.append(window_summary(ws, st, srcs).assign(variant=variant))
        if variant == "common":
            make_figure(os.path.join(args.out, "fig_gc_stance_ensemble.png"), ens, srcs,
                        f"{os.path.basename(run)}: stance-binned medial force on identical frames "
                        f"(phi {args.phi:g}, d {args.d_mm:g} mm, M_x sign {conv})")
    ens = pd.concat(ens_all, ignore_index=True)
    humps = pd.concat(tabs, ignore_index=True)
    ws = pd.concat(win_all, ignore_index=True)
    wsum = pd.concat(wsum_all, ignore_index=True)
    ens.to_csv(os.path.join(args.out, "ensemble.csv"), index=False)
    humps.to_csv(os.path.join(args.out, "humps.csv"), index=False)
    ws.to_csv(os.path.join(args.out, "windows.csv"), index=False)
    wsum.to_csv(os.path.join(args.out, "windows_summary.csv"), index=False)
    pd.set_option("display.width", 220)
    cols = ["variant", "subject", "source", "n_frames", "curve_r", "curve_mae", "early_peak", "early_pct",
            "early_interior", "late_peak", "late_pct", "late_interior", "trough", "trough_pct", "depth_rel",
            "late_early_ratio", "double_hump"]
    print(humps[cols].round(3).to_string(index=False))
    print(wsum.round(3).to_string(index=False))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

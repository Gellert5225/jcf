"""Grand Challenge validation on QC-passing frames, the init-seed spread and the seed ensemble (post-processing of a
grand_challenge.py run; no model, no OpenSim).

grand_challenge.py scores every instrumented-limb stance frame. The *_sojrqc surrogates were trained only on frames
whose SO+JR label passes the label QC (<trial>/jcf_output_sojr/label_mask.npz, analysis/thesis/so_label_qc.py masks),
so this script adds the same frame mask to the in vivo comparison:

  keep        frames whose instrumented-knee SO+JR label passes the QC (keep_r for JW and DM, keep_l for SC, the
              knee the surrogate reads after mirroring), carried onto the 100 Hz eval grid as train_mlp.carry_label_mask
              does (summed linear-interpolation weight of masked JR rows <= 0.05; one interpolation JR -> eval grid
              here, two in training: JR -> IK -> 100 Hz)
  fail        the other stance frames
  keep_nearest  sensitivity: mask value of the nearest JR row (as thesis_paper/checks/sc_qc_overread.py)

The measured force is the same on every subset, so 'keep' asks whether the surrogates track the implant better where
their training labels would have been trusted. Every source (stored labels, SO+JR, StatOp, every model) is scored on
identical frames within a subset.

Also written: the seed ensemble (--ensemble-name, default qc_ens: mean F_y and M_x of the --seeds checkpoints), the seed spread of every
frame and peak metric, stance peaks on windows whose frames are all kept, and DM at the fixed sensitivity lag (default
+95 ms, the run's model-free heel-strike-to-first-peak estimate) on both subsets.

    python -I analysis/thesis/gc_keep_subset.py --run audit_gc --seeds qc_s1 qc_s2 qc_s3

Outputs in results/thesis/gc/<run>/qc_keep/: frame_keep.csv.gz, keep_share.csv, summary.csv, peaks.csv,
peak_windows.csv, seed_spread.csv, stance_shape.csv,
label_agreement.csv, dm_lag.csv, README.md.
"""
import argparse
import datetime
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from analysis.thesis.grand_challenge import (GC_ROOT, OUT_ROOT, SUBJECTS, frame_summary, fmed_of,  # noqa: E402
                                             shift_measured, window_peak, _metrics)

MASK_TOL = 0.05                      # train_mlp.LABEL_MASK_TOL
RULES = ("sat_knee", "sat_leg3", "res_leg", "pelvis_m", "jump", "coact")
STO = "BatchJCF_JointReaction_ReactionLoads.sto"


def sto_time(path):
    """Time column of an OpenSim .sto file."""
    with open(path) as f:
        for line in f:
            if line.strip().lower() == "endheader":
                break
        f.readline()                                     # column names
        return np.array([float(l.split()[0]) for l in f if l.strip()])


def segment_dir(folder, repo=REPO):
    hits = glob.glob(os.path.join(repo, "jcf", "full_duration", "*", "walking", folder))
    if len(hits) != 1:
        raise FileNotFoundError(f"{folder}: {len(hits)} segment folders")
    return hits[0]


def carry_keep(t_mask, keep, t, tol=MASK_TOL):
    """keep on the frames t: summed linear-interpolation weight of masked JR rows <= tol."""
    bad = np.interp(np.asarray(t, float), t_mask, (~np.asarray(keep, bool)).astype(float))
    return bad <= tol + 1e-12


def nearest_index(t_mask, t):
    t = np.asarray(t, float)
    i = np.clip(np.searchsorted(t_mask, t), 1, len(t_mask) - 1)
    return np.where(np.abs(t_mask[i - 1] - t) <= np.abs(t_mask[i] - t), i - 1, i)


def add_keep(fr):
    """keep / keep_nearest / fixed_cop / rule flags of the instrumented knee per frame, from each segment's mask."""
    out = {c: np.zeros(len(fr), bool) for c in ("keep", "keep_nearest", "fixed_cop", *RULES)}
    out["mask_gap_ms"] = np.full(len(fr), np.nan)
    info = []
    for (subj, folder), g in fr.groupby(["subject", "folder"]):
        side = SUBJECTS[subj]["side"]
        d = os.path.join(segment_dir(folder), "jcf_output_sojr")
        with np.load(os.path.join(d, "label_mask.npz")) as z:
            t_mask = np.asarray(z["time"], float)
            keep = np.asarray(z[f"keep_{side}"], bool)
            flags = {r: np.asarray(z[f"{r}_{side}"], bool) for r in RULES}
            fixed = bool(np.asarray(z[f"fixed_cop_{side}"]).any())
        t_sto = sto_time(os.path.join(d, STO))
        if len(t_sto) != len(t_mask) or np.abs(t_sto - t_mask).max() > 1e-6:
            raise ValueError(f"{folder}: label_mask.npz is stale against {STO}")
        pos = fr.index.get_indexer(g.index)
        t = g["t_seg"].to_numpy(float)
        j = nearest_index(t_mask, t)
        out["keep"][pos] = carry_keep(t_mask, keep, t)
        out["keep_nearest"][pos] = keep[j]
        out["fixed_cop"][pos] = fixed
        for r in RULES:
            out[r][pos] = flags[r][j]
        out["mask_gap_ms"][pos] = 1000.0 * np.abs(t_mask[j] - t)
        info.append(dict(subject=subj, folder=folder, side=side, fixed_cop=fixed, jr_rows=len(t_mask),
                         jr_keep_frac=float(keep.mean())))
    for c, v in out.items():
        fr[c] = v
    return fr, pd.DataFrame(info)


def add_ensemble(fr, seeds, name):
    for q in ("fy", "mx"):
        fr[f"{q}_{name}"] = fr[[f"{q}_{s}" for s in seeds]].mean(axis=1, skipna=False)
    return fr


def window_table(fr, st, sources, phi, d_mm, margin):
    """Stance peaks per covered window of stances.csv, recomputed from the frame table (every source incl. the
    ensemble), with each window's share of kept frames."""
    rows = []
    for _, w in st.iterrows():
        g = fr[(fr.folder == w.folder) & (fr.stance_id == w.stance) & (fr.part == w.part)]
        pct = g["pct"].to_numpy(float)
        _, pk, pk_pct, interior = window_peak(g["meas_med"].to_numpy(float), pct, margin)
        row = dict(subject=w.subject, folder=w.folder, gc_key=w.gc_key, gait=w.gait, stance=w.stance, part=w.part,
                   n_frames=len(g), keep_frac=float(g["keep"].mean()), meas_peak=pk, meas_peak_pct=pk_pct,
                   meas_interior=interior)
        for src in sources:
            v = fmed_of(g, src, phi, d_mm)
            if len(v) and np.isfinite(v).all():
                _, ppk, ppct, pint = window_peak(v, pct, margin)
                row.update({f"peak_{src}": ppk, f"err_{src}": ppk - pk, f"interior_{src}": pint})
        rows.append(row)
    return pd.DataFrame(rows)


def peak_summary(wt, sources, subset):
    rows = []
    sc = wt[wt["meas_interior"].astype(bool)]
    for (subj, part), g in sc.groupby(["subject", "part"]):
        for src in sources:
            if f"peak_{src}" not in g or not g[f"peak_{src}"].notna().any():
                continue
            ok = g[f"peak_{src}"].notna()
            m = _metrics(g.loc[ok, "meas_peak"], g.loc[ok, f"peak_{src}"])
            rows.append(dict(subset=subset, subject=subj, part=part, source=src, n_windows=m["n"], mae=m["mae"],
                             bias=m["bias"], r=m["r"], meas_mean=float(g.loc[ok, "meas_peak"].mean()),
                             pred_mean=float(g.loc[ok, f"peak_{src}"].mean()),
                             pred_interior_frac=float(g.loc[ok, f"interior_{src}"].astype(bool).mean())))
    return pd.DataFrame(rows)


def seed_spread(summ, peaks, seeds, ens):
    rows = []
    fm = ["mae", "rmse", "bias", "r", "total_mae", "total_bias", "total_r"]
    for (subset, subj), g in summ.groupby(["subset", "subject"]):
        g = g.set_index("source")
        if not set(seeds) <= set(g.index):
            continue
        for m in fm:
            v = g.loc[list(seeds), m].to_numpy(float)
            rows.append(dict(kind="frames", subset=subset, subject=subj, part="", metric=m,
                             **{s: x for s, x in zip(seeds, v)}, mean=v.mean(), sd=v.std(ddof=1), min=v.min(),
                             max=v.max(), range=v.max() - v.min(), ensemble=g.loc[ens, m] if ens in g.index else np.nan))
    for (subset, subj, part), g in peaks.groupby(["subset", "subject", "part"]):
        g = g.set_index("source")
        if not set(seeds) <= set(g.index):
            continue
        for m in ("mae", "bias"):
            v = g.loc[list(seeds), m].to_numpy(float)
            rows.append(dict(kind="peaks", subset=subset, subject=subj, part=part, metric=m,
                             **{s: x for s, x in zip(seeds, v)}, mean=v.mean(), sd=v.std(ddof=1), min=v.min(),
                             max=v.max(), range=v.max() - v.min(), ensemble=g.loc[ens, m] if ens in g.index else np.nan))
    return pd.DataFrame(rows)


def stance_shape(fr, sources, phi, d_mm, subset, step=10.0):
    """Mean F_medial per subject x %-of-stance bin (frames inside a complete stance), measured and every source, on
    the bin's frames where every source the subject has exists (SC has no stored labels)."""
    out = []
    for subj, g in fr[fr.stance & (fr.stance_id >= 0) & fr.pct.notna()].groupby("subject"):
        g = g.copy()
        cols = {src: fmed_of(g, src, phi, d_mm) for src in sources if src != "statop"}
        cols = {src: v for src, v in cols.items() if np.isfinite(v).any()}
        for src, v in cols.items():
            g[f"fm_{src}"] = v
        g = g.dropna(subset=[f"fm_{s}" for s in cols])
        g["bin"] = (np.floor(g.pct / step) * step).clip(0, 100 - step)
        agg = {"n_frames": ("meas_med", "size"), "meas": ("meas_med", "mean")}
        agg.update({src: (f"fm_{src}", "mean") for src in cols})
        out.append(g.groupby(["subject", "bin"]).agg(**agg).reset_index())
    return pd.concat(out, ignore_index=True).assign(subset=subset)


def label_agreement(fr, models, osim, phi, d_mm, subset):
    """Pearson r between each model's output and each OpenSim label on the same stance frames (raw M_x, F_y and
    F_medial; out of distribution: the Fregly patients were excluded from training)."""
    rows = []
    for subj, g in fr[fr.stance].groupby("subject"):
        for m in models:
            for lab in osim:
                ok = g[f"mx_{m}"].notna() & g[f"mx_{lab}"].notna()
                if ok.sum() < 3:
                    continue
                gg = g[ok]
                rows.append(dict(subset=subset, subject=subj, model=m, label=lab, n_frames=int(ok.sum()),
                                 r_mx=np.corrcoef(gg[f"mx_{m}"], gg[f"mx_{lab}"])[0, 1],
                                 r_fy=np.corrcoef(gg[f"fy_{m}"], gg[f"fy_{lab}"])[0, 1],
                                 r_fmed=np.corrcoef(fmed_of(gg, m, phi, d_mm), fmed_of(gg, lab, phi, d_mm))[0, 1]))
    return pd.DataFrame(rows)


def _fmt(df, cols=None):
    df = df if cols is None else df[[c for c in cols if c in df]]
    return df.to_string(index=False, float_format=lambda v: f"{v:.3f}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="grand_challenge.py run name under results/thesis/gc/")
    ap.add_argument("--seeds", nargs="+", default=["qc_s1", "qc_s2", "qc_s3"], help="labels of the seed checkpoints")
    ap.add_argument("--ensemble-name", default="qc_ens")
    ap.add_argument("--dm-lag-ms", type=float, default=None, help="default: the run's dm_lag_sensitivity.csv lag")
    ap.add_argument("--out-root", default=OUT_ROOT)
    ap.add_argument("--gc-root", default=GC_ROOT)
    args = ap.parse_args(argv)
    run = os.path.join(args.out_root, args.run)
    out = os.path.join(run, "qc_keep")
    os.makedirs(out, exist_ok=True)
    meta = json.load(open(os.path.join(run, "run.json")))
    ra = meta["args"]
    if ra.get("mx_convention", "physical") != "physical":
        raise SystemExit("run is not in the physical M_x convention")
    phi, d_mm, margin = 0.5, 45.0, float(ra.get("edge_margin_pct", 3.0))
    fr = pd.read_csv(os.path.join(run, "frames.csv.gz"))
    fr["stance"] = fr["stance"].astype(bool)
    st = pd.read_csv(os.path.join(run, "stances.csv"))
    labels = list(ra["labels"])
    missing = [s for s in args.seeds if s not in labels]
    if missing:
        raise SystemExit(f"--seeds {missing} not in the run's labels {labels}")
    fr = add_ensemble(fr, args.seeds, args.ensemble_name)
    fr, seg_info = add_keep(fr)
    osim = [s for s in ("osim_state", "osim_sojr") if f"fy_{s}" in fr and fr[f"fy_{s}"].notna().any()]
    cut = max(labels.index(s) for s in args.seeds) + 1          # the ensemble right after the last seed
    models = labels[:cut] + [args.ensemble_name] + labels[cut:]
    sources = osim + models + (["statop"] if "fmed_statop" in fr else [])

    keepcols = ["subject", "folder", "gc_key", "k", "t_seg", "stance", "keep", "keep_nearest", "fixed_cop",
                *RULES, "mask_gap_ms"]
    fr[keepcols].to_csv(os.path.join(out, "frame_keep.csv.gz"), index=False, float_format="%.6g")

    # share of stance frames kept, and why the others fail
    s_ = fr[fr.stance]
    ks = s_.groupby("subject").agg(stance_frames=("keep", "size"), keep=("keep", "sum"),
                                   keep_nearest=("keep_nearest", "sum"), fixed_cop_frames=("fixed_cop", "sum"),
                                   **{f"{r}_frames": (r, "sum") for r in RULES},
                                   segments=("folder", "nunique"),
                                   segments_with_keep=("keep", lambda v: int(s_.loc[v.index][v].folder.nunique())),
                                   max_mask_gap_ms=("mask_gap_ms", "max")).reset_index()
    ks["keep_frac"] = ks["keep"] / ks["stance_frames"]
    ks["keep_nearest_frac"] = ks["keep_nearest"] / ks["stance_frames"]
    nfix = seg_info.groupby("subject").agg(segments_total=("folder", "size"),
                                           segments_fixed_cop=("fixed_cop", "sum")).reset_index()
    ks = ks.merge(nfix, on="subject")
    ks.to_csv(os.path.join(out, "keep_share.csv"), index=False)

    # frame metrics
    subsets = {"all": fr, "keep": fr[fr.keep], "fail": fr[~fr.keep], "keep_nearest": fr[fr.keep_nearest]}
    summ = pd.concat([frame_summary(f, sources, phi, d_mm).assign(subset=k) for k, f in subsets.items()
                      if f["stance"].any()], ignore_index=True)
    summ = summ[["subset"] + [c for c in summ.columns if c != "subset"]]
    summ.to_csv(os.path.join(out, "summary.csv"), index=False)

    # stance peaks: all covered windows (reproduces the run) and windows whose frames are all kept
    wt = window_table(fr, st, sources, phi, d_mm, margin)
    wt.to_csv(os.path.join(out, "peak_windows.csv"), index=False)
    peaks = pd.concat([peak_summary(wt, sources, "all"), peak_summary(wt[wt.keep_frac >= 1.0], sources, "keep")],
                      ignore_index=True)
    peaks.to_csv(os.path.join(out, "peaks.csv"), index=False)
    wcount = wt.assign(scored=wt.meas_interior.astype(bool), kept=wt.keep_frac >= 1.0).groupby(
        ["subject", "part"]).agg(covered=("scored", "size"), scored=("scored", "sum"),
                                 scored_and_kept=("kept", lambda v: int((v & wt.loc[v.index, "meas_interior"]
                                                                         .astype(bool)).sum()))).reset_index()

    spread = seed_spread(summ, peaks, args.seeds, args.ensemble_name)
    spread.to_csv(os.path.join(out, "seed_spread.csv"), index=False)
    shape = pd.concat([stance_shape(fr, sources, phi, d_mm, "all"), stance_shape(fr[fr.keep], sources, phi, d_mm,
                                                                                 "keep")], ignore_index=True)
    shape.to_csv(os.path.join(out, "stance_shape.csv"), index=False)
    agree = pd.concat([label_agreement(fr, models, osim, phi, d_mm, "all"),
                       label_agreement(fr[fr.keep], models, osim, phi, d_mm, "keep")], ignore_index=True)
    agree.to_csv(os.path.join(out, "label_agreement.csv"), index=False)

    # DM at the fixed sensitivity lag, all and keep frames
    lag = args.dm_lag_ms
    if lag is None:
        lag = float(meta["dm_lag_ms_used"])                 # the run's model-free DM sensitivity lag
    dm = fr[fr.subject == "DM"]
    dml = []
    for L in (0.0, lag):
        f = shift_measured(dm, L, args.gc_root) if L else dm
        for k, ff in (("all", f), ("keep", f[f.keep])):
            dml.append(frame_summary(ff, sources, phi, d_mm).assign(kind="frames", lag_ms=L, subset=k))
        wtl = window_table(f, st[st.subject == "DM"], sources, phi, d_mm, margin)
        for k, ww in (("all", wtl), ("keep", wtl[wtl.keep_frac >= 1.0])):
            p = peak_summary(ww, sources, k)
            if not p.empty:
                dml.append(p.assign(kind="peaks", lag_ms=L).rename(columns={"n_windows": "n_frames"}))
    dml = pd.concat(dml, ignore_index=True)
    dml.to_csv(os.path.join(out, "dm_lag.csv"), index=False)

    # README
    fcols = ["subset", "subject", "source", "n_frames", "n_segments", "mae", "bias", "r", "total_bias", "total_r"]
    L = [f"# Grand Challenge on QC-passing frames: {args.run}",
         "",
         f"Written {datetime.datetime.now():%Y-%m-%d %H:%M} by `analysis/thesis/gc_keep_subset.py --run {args.run} "
         f"--seeds {' '.join(args.seeds)}` from `results/thesis/gc/{args.run}/frames.csv.gz` (physical M_x sign, "
         f"phi = {phi}, d = {d_mm} mm) and each segment's `jcf_output_sojr/label_mask.npz`. No model was rerun.",
         f"`{args.ensemble_name}` = mean F_y and M_x of {', '.join(args.seeds)}.",
         "",
         "keep = the instrumented knee's SO+JR label passes the QC (keep_r for JW/DM, keep_l for SC), carried to the "
         f"eval grid as in training (masked interpolation weight <= {MASK_TOL}); fail = the other stance frames; "
         "keep_nearest = nearest JR row (sensitivity). All sources are scored on the same frames within a subset; "
         "the measured force is unchanged. The `all` rows reproduce the run's README (frames.csv.gz is stored at 5 "
         "significant digits).",
         "",
         "Notes on the run's checkpoints (from run.json; the run README's 'PROVISIONAL ... rerun after the *_t100 "
         "retraining' banner is printed for every run and is stale when the *_t100 checkpoints are in it):",
         "",
         *[f"- `{lab}`: input set `{i.get('input_set')}`, exclude {i.get('exclude')}" +
           ("  **(training exclusion not recorded; whether the fregly folders (JW, DM) were in its training set is "
            "unknown)**" if not i.get("exclude") else "")
           for lab, i in zip(labels, meta.get("checkpoints", [{}] * len(labels)))],
         "",
         "## Kept share of stance frames",
         "", "```", _fmt(ks), "```", "",
         "## Frame metrics (stance, BW)", "", "```", _fmt(summ, fcols), "```", "",
         "## Stance peaks (scored = interior measured maximum; keep = every frame of the window kept)", "",
         "```", _fmt(wcount), "```", "",
         "```", _fmt(peaks, ["subset", "subject", "part", "source", "n_windows", "mae", "bias", "meas_mean",
                             "pred_mean", "pred_interior_frac"]), "```", "",
         f"## Seed spread ({', '.join(args.seeds)}; sd with ddof = 1)", "",
         "```", _fmt(spread[spread.kind == "frames"].drop(columns=["kind", "part"])), "```", "",
         "```", _fmt(spread[spread.kind == "peaks"].drop(columns=["kind"])), "```", "",
         "## Stance shape (mean F_medial per 10 % of stance, frames where every source exists; StatOp left out)",
         "", "```", _fmt(shape[shape.subset == "all"].drop(columns=["subset"])), "```", "",
         "## Model output vs OpenSim labels on the same stance frames (Pearson r; M_x in the run's medial-positive "
         "convention, F_y raw)", "", "```", _fmt(agree[agree.subset == "all"]), "```", "",
         f"## DM at lag 0 and +{lag:.0f} ms (measured force taken later), all and kept frames", "",
         "```", _fmt(dml[dml.kind == "frames"], ["lag_ms", "subset", "source", "n_frames", "mae", "bias", "r",
                                                  "total_bias"]), "```", "",
         "```", _fmt(dml[dml.kind == "peaks"], ["lag_ms", "subset", "part", "source", "n_frames", "mae", "bias",
                                                 "pred_interior_frac"]), "```", ""]
    with open(os.path.join(out, "README.md"), "w") as f:
        f.write("\n".join(L))
    print(_fmt(ks))
    print(_fmt(summ[summ.subset.isin(["all", "keep"])], fcols))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Is the SO+JR relabel complete enough to train on, and what do the label filters do on the same trials?

The SO+JR labels live in <trial>/jcf_output_sojr/BatchJCF_JointReaction_ReactionLoads.sto (analysis/thesis/gc_knee_jr.py);
the stored labels in <trial>/jcf_output/. train_mlp.py and tools/evaluate_fmedial.py discover a trial by its label
file, so a relabel with failed folders silently changes the trial set unless it is checked.

--gate --require-mask (the *_sojrqc input sets): additionally every complete SO+JR file needs its frame mask
jcf_output_sojr/label_mask.npz (analysis/thesis/so_label_qc.py masks) with the same time column; a missing or
mismatched mask fails the gate.

--gate (coverage only, fast): every folder in scope that has the stored labels must have a complete SO+JR file (as
many data rows as its nRows header). Exit status 0 when that holds, 1 otherwise, with the missing folders per split,
activity and lab. Files with non-finite values are listed but do not fail the gate: the loaders drop those trials and
say so. The relabel summary (results/thesis/gc/knee_jr/relabel_all.csv) is read for its status counts per lab and
failure cause. Its rows are keyed by folder name, which is not unique across the roots (moore_subject5_t00 exists in
training/walking and training/static), so the per-folder test is made on disk: a folder has a complete file exactly
when the relabel wrote it (status ok) or found it (status exists), and the test stays valid when failed folders are
redone in a later run with a different summary file.

Default (coverage + filters): for every folder, the label-filter verdicts of the training code under both label
sets: non-finite rows (train_cnn.label_nonfinite_rows), the flat filter (train_cnn.filter_flat_subjects, with its
CV and range), and load_subject's trial-level peak tests (right-knee force resultant > --max-peak BW, |M_x| >
--max-moment BW*H in any label frame). Per-lab tables compare the two label sets on the same trials (folders with
both, finite).

Outputs (default results/thesis/labels_qc/train_plumbing/):
    coverage.csv          split, activity, lab: stored-label folders, SO+JR complete / incomplete / missing, non-finite
    relabel_status.csv    lab x status from the relabel summary (failures grouped by cause)
    trials.csv            one row per stored-label folder (filters mode)
    flat_same_trials.csv  walking, per split and lab, on the trials with both label sets: flat under each, both, only one
    peak_same_trials.csv  the same for the peak and moment tests
    README.md, run_info.json

Usage (repo root; reads files only):
    python analysis/thesis/sojr_trainset.py --gate --activity walking --exclude han fregly
    python analysis/thesis/sojr_trainset.py --gate --require-mask --activity walking --exclude han fregly
    python analysis/thesis/sojr_trainset.py --workers 4
"""
import os as _os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")   # one thread per worker
import argparse
import datetime
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

JR_FILE = "BatchJCF_JointReaction_ReactionLoads.sto"
MASK_FILE = "label_mask.npz"              # train_mlp.LABEL_MASK_FILE
STORED, SOJR = "jcf_output", "jcf_output_sojr"
SOURCES = (("stored", STORED), ("sojr", SOJR))
PRE = "walker_knee_r_on_tibia_r_in_tibia_r_"
DATA = os.path.join(REPO, "jcf", "full_duration")
SPLITS, ACTIVITIES = ("training", "testing"), ("walking", "static")
DEFAULT_OUT = os.path.join(REPO, "results", "thesis", "labels_qc", "train_plumbing")
RELABEL_SUMMARY = os.path.join(REPO, "results", "thesis", "gc", "knee_jr", "relabel_all.csv")
FLAT_CV, FLAT_DR = 0.25, 0.3            # train_mlp.FLAT_CV / FLAT_DR
MAX_PEAK_BW, MAX_MOMENT_BWH = 6.0, 0.15  # train_mlp.MAX_PEAK_BW / MAX_MOMENT_BWH
LAB_ORDER = ("carter", "moore", "tiziana", "falisse", "fregly", "han")


def lab_of(name):
    return os.path.basename(name.rstrip("/")).split("_")[0]


def header_nrows(path):
    """nRows= from an OpenSim header (None when absent)."""
    with open(path) as f:
        for line in f:
            if line.startswith("nRows="):
                return int(line.split("=", 1)[1])
            if line.strip().lower() == "endheader":
                return None
    return None


def read_labels(path):
    """(table, state): state 'complete' when the file parses and has nRows data rows, else 'incomplete'."""
    from train_cnn import load_sto
    try:
        n = header_nrows(path)
        j = load_sto(path)
    except Exception:
        return None, "incomplete"
    return j, ("complete" if n is not None and len(j) == n and n > 0 else "incomplete")


def folders(splits=SPLITS, activities=ACTIVITIES, exclude=(), labs=None, data=DATA):
    """(split, activity, folder) for every folder with the stored labels."""
    out = []
    for split in splits:
        for act in activities:
            root = os.path.join(data, split, act)
            for name in sorted(os.listdir(root)) if os.path.isdir(root) else []:
                lab = lab_of(name)
                if lab in (exclude or ()) or (labs and lab not in labs):
                    continue
                d = os.path.join(root, name)
                if os.path.exists(os.path.join(d, STORED, JR_FILE)):
                    out.append((split, act, d))
    return out


def label_stats(j, mass, height):
    """Peak resultant (BW), peak |M_x| (BW*H), CV and range of the right-knee resultant, as the training code does."""
    bw = mass * 9.81
    res = np.sqrt(sum(j[PRE + c].to_numpy() ** 2 for c in ("fx", "fy", "fz"))) / bw
    mx = np.abs(j[PRE + "mx"].to_numpy()) / (bw * height)
    return dict(peak_bw=float(res.max()), mx_peak_bwh=float(mx.max()),
                cv=float(res.std() / (res.mean() + 1e-12)), dr=float(res.max() - res.min()))


def trial_row(job):
    """Coverage (and, unless gate_only, filter verdicts) for one folder."""
    split, act, d, gate_only, flat_cv, flat_dr = job[:6]
    require_mask = len(job) > 6 and job[6]
    from train_cnn import nonfinite_rows, filter_flat_subjects
    row = dict(split=split, activity=act, lab=lab_of(d), trial=os.path.basename(d))
    sojr_path = os.path.join(d, SOJR, JR_FILE)
    tables = {}
    if os.path.exists(sojr_path):
        j, state = read_labels(sojr_path)
        row["sojr_state"] = state
        if state == "complete":
            row["sojr_nonfinite_rows"] = nonfinite_rows(j)
            tables["sojr"] = j
            if require_mask:
                row["mask_state"] = mask_state(d, j)
    else:
        row["sojr_state"] = "missing"
    if gate_only:
        return row
    j, state = read_labels(os.path.join(d, STORED, JR_FILE))
    row["stored_state"] = state
    if j is not None:
        row["stored_nonfinite_rows"] = nonfinite_rows(j)
        tables["stored"] = j
    try:
        meta = json.load(open(os.path.join(d, "metadata.json")))
        mass, height = float(meta["mass_kg"]), float(meta.get("height_m", 1.7))
    except Exception:
        return row
    for src, sub in SOURCES:
        j = tables.get(src)
        if j is None or row.get(f"{src}_nonfinite_rows", 1):
            continue
        for k, v in label_stats(j, mass, height).items():
            row[f"{src}_{k}"] = v
        kept, _ = filter_flat_subjects([d], cv_threshold=flat_cv, dr_threshold=flat_dr, jcf_subdir=sub)
        row[f"{src}_flat"] = not kept
    return row


def mask_state(d, j):
    """'ok', 'missing' or 'mismatch' (time column differs from the SO+JR file j, or unreadable) for the frame mask."""
    path = os.path.join(d, SOJR, MASK_FILE)
    if not os.path.exists(path):
        return "missing"
    try:
        with np.load(path) as z:
            t = np.asarray(z["time"], float)
            ok = len(t) == len(j) and len(z["keep_r"]) == len(t) and np.abs(t - j["time"].to_numpy(float)).max() <= 1e-6
    except Exception:
        return "mismatch"
    return "ok" if ok else "mismatch"


def relabel_status(paths):
    """lab x status counts of the relabel summaries; failures grouped by cause."""
    frames = []
    for p in paths:
        if not os.path.exists(p):
            continue
        s = pd.read_csv(p, usecols=["folder", "status"])
        s["summary"] = os.path.relpath(p, REPO)
        frames.append(s)
    if not frames:
        return pd.DataFrame(columns=["summary", "lab", "status", "n"])
    s = pd.concat(frames, ignore_index=True)
    st = s["status"].astype(str)
    cause = np.where(st.str.startswith("failed") & st.str.contains("Cannot open file /Users/", regex=False),
                     "failed: setup points to a /Users/ (macOS) model path",
                     np.where(st.str.startswith("failed"), "failed: other", st))
    s = s.assign(lab=s["folder"].map(lab_of), status=cause)
    return s.groupby(["summary", "lab", "status"]).size().rename("n").reset_index()


def coverage_table(df):
    """split, activity, lab (and ALL): stored-label folders and the SO+JR file state."""
    rows = []
    for keys, g in list(df.groupby(["split", "activity", "lab"])) + [(("ALL", "ALL", "ALL"), df)]:
        st = g["sojr_state"].value_counts()
        nf = g.get("sojr_nonfinite_rows", pd.Series(dtype=float)).fillna(0)
        rows.append(dict(split=keys[0], activity=keys[1], lab=keys[2], n_stored=len(g),
                         sojr_complete=int(st.get("complete", 0)), sojr_incomplete=int(st.get("incomplete", 0)),
                         sojr_missing=int(st.get("missing", 0)), sojr_nonfinite=int((nf > 0).sum())))
    return pd.DataFrame(rows)


def _same_trials(df):
    """Walking folders with finite labels in both sets."""
    need = ["stored_flat", "sojr_flat"]
    if not all(c in df.columns for c in need):
        return df.iloc[0:0]
    return df[(df["activity"] == "walking") & df["stored_flat"].notna() & df["sojr_flat"].notna()]


def _groups(df):
    out = []
    for split, g in df.groupby("split"):
        for lab, gl in g.groupby("lab"):
            out.append((split, lab, gl))
        out.append((split, "ALL", g))
    return out


def flat_table(df):
    """Flat-filter verdicts under both label sets on the same walking trials."""
    rows = []
    for split, lab, g in _groups(_same_trials(df)):
        fs, fj = g["stored_flat"].astype(bool), g["sojr_flat"].astype(bool)
        rows.append(dict(split=split, lab=lab, n_same=len(g), flat_stored=int(fs.sum()), flat_sojr=int(fj.sum()),
                         flat_both=int((fs & fj).sum()), flat_only_stored=int((fs & ~fj).sum()),
                         flat_only_sojr=int((~fs & fj).sum()), kept_stored=int((~fs).sum()),
                         kept_sojr=int((~fj).sum())))
    return pd.DataFrame(rows)


def peak_table(df, max_peak, max_moment):
    """load_subject's peak and moment tests under both label sets on the same walking trials."""
    rows = []
    for split, lab, g in _groups(_same_trials(df)):
        r = dict(split=split, lab=lab, n_same=len(g))
        for src in ("stored", "sojr"):
            pk, mx = g[f"{src}_peak_bw"] > max_peak, g[f"{src}_mx_peak_bwh"] > max_moment
            r.update({f"{src}_peak_gt": int(pk.sum()), f"{src}_mx_gt": int(mx.sum()),
                      f"{src}_rejected": int((pk | mx).sum()),
                      f"{src}_peak_median_bw": float(g[f"{src}_peak_bw"].median())})
        rows.append(r)
    return pd.DataFrame(rows)


def _md(df, cols=None):
    df = df if cols is None else df[cols]
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(f"{v:.2f}" if isinstance(v, float) else str(v) for v in r.values) + " |")
    return "\n".join(lines)


def _git():
    try:
        return subprocess.run(["git", "-C", REPO, "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True).stdout.strip() or None
    except Exception:
        return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gate", action="store_true", help="coverage only; exit status 1 when SO+JR labels are missing")
    ap.add_argument("--require-mask", action="store_true",
                    help=f"also require {SOJR}/{MASK_FILE} matching each SO+JR file (the *_sojrqc input sets)")
    ap.add_argument("--split", nargs="+", default=list(SPLITS), choices=SPLITS)
    ap.add_argument("--activity", nargs="+", default=list(ACTIVITIES), choices=ACTIVITIES)
    ap.add_argument("--exclude", nargs="+", default=[], help="labs to leave out, e.g. han fregly")
    ap.add_argument("--labs", nargs="+", default=None, help="only these labs")
    ap.add_argument("--summary", nargs="+", default=[RELABEL_SUMMARY], help="relabel summary CSV(s) (status counts)")
    ap.add_argument("--flat-cv", type=float, default=FLAT_CV)
    ap.add_argument("--flat-dr", type=float, default=FLAT_DR)
    ap.add_argument("--max-peak", type=float, default=MAX_PEAK_BW)
    ap.add_argument("--max-moment", type=float, default=MAX_MOMENT_BWH)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--data", default=DATA, help=argparse.SUPPRESS)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args(argv)

    t0 = time.time()
    found = folders(args.split, args.activity, args.exclude, args.labs, data=args.data)
    jobs = [(s, a, d, args.gate, args.flat_cv, args.flat_dr, args.require_mask) for s, a, d in found]
    if args.workers > 1 and len(jobs) > 50:
        with mp.Pool(args.workers) as pool:
            rows = pool.map(trial_row, jobs, chunksize=32)
    else:
        rows = [trial_row(j) for j in jobs]
    df = pd.DataFrame(rows, columns=None if rows else ["split", "activity", "lab", "trial", "sojr_state"])
    cov = coverage_table(df) if len(df) else pd.DataFrame()
    status = relabel_status(args.summary)
    scope = dict(split=args.split, activity=args.activity, exclude=args.exclude, labs=args.labs)
    not_ready = df[df["sojr_state"] != "complete"] if len(df) else df
    nonfinite = df[df.get("sojr_nonfinite_rows", pd.Series(0, index=df.index)).fillna(0) > 0] if len(df) else df
    ready = len(df) > 0 and len(not_ready) == 0
    if args.require_mask and len(df):
        complete = df[df["sojr_state"] == "complete"]
        bad_mask = complete[complete["mask_state"] != "ok"] if "mask_state" in complete else complete
        ready = ready and len(bad_mask) == 0

    print(f"scope {scope}: {len(df)} folders with {STORED}/ labels; SO+JR complete {len(df) - len(not_ready)}, "
          f"missing or incomplete {len(not_ready)}; non-finite SO+JR files {len(nonfinite)}  ({time.time() - t0:.0f} s)")
    if len(not_ready):
        miss = not_ready.groupby(["split", "activity", "lab"]).size()
        for (s, a, lab), n in miss.items():
            tot = int(((df["split"] == s) & (df["activity"] == a) & (df["lab"] == lab)).sum())
            print(f"  {s}/{a}/{lab}: {n} of {tot} lack a complete {SOJR}/ file")
    for _, r in nonfinite.iterrows():
        print(f"  non-finite {SOJR}/ labels ({int(r['sojr_nonfinite_rows'])} rows; the loaders drop it): "
              f"{r['split']}/{r['activity']}/{r['trial']}")
    if len(status):
        bad = status[~status["status"].isin(["ok", "exists", "no SO output"])]
        if len(bad):
            print("relabel summary, unfinished statuses: " + "; ".join(
                f"{r['lab']} {r['n']} x {r['status']}" for _, r in bad.iterrows()))
    if args.require_mask and len(df):
        print(f"frame masks ({SOJR}/{MASK_FILE}): {len(complete) - len(bad_mask)} of {len(complete)} complete SO+JR "
              f"folders have a matching mask")
        for (s, a, lab, st_), g in bad_mask.groupby(["split", "activity", "lab", "mask_state"]):
            print(f"  {s}/{a}/{lab}: {len(g)} mask {st_} (e.g. {g['trial'].iloc[0]}); write them with: "
                  "python -I analysis/thesis/so_label_qc.py masks --workers 20")
    print(f"READY: every folder in scope has complete {SOJR}/ labels" + (" and frame masks" if args.require_mask else "")
          if ready else f"NOT READY: {len(not_ready)} folders in scope lack complete {SOJR}/ labels"
          + (f", {len(bad_mask)} lack a matching frame mask" if args.require_mask and len(df) else ""))
    if args.gate:
        return 0 if ready else 1

    os.makedirs(args.out, exist_ok=True)
    df.to_csv(os.path.join(args.out, "trials.csv"), index=False)
    cov.to_csv(os.path.join(args.out, "coverage.csv"), index=False)
    status.to_csv(os.path.join(args.out, "relabel_status.csv"), index=False)
    flat = flat_table(df)
    peak = peak_table(df, args.max_peak, args.max_moment) if len(_same_trials(df)) else pd.DataFrame()
    flat.to_csv(os.path.join(args.out, "flat_same_trials.csv"), index=False)
    peak.to_csv(os.path.join(args.out, "peak_same_trials.csv"), index=False)
    walk = df[df["activity"] == "walking"]
    flat_all = []     # flat verdicts on all stored-label walking trials (the stored-label run's view)
    for split, lab, g in _groups(walk):
        if "stored_flat" in g:
            flat_all.append(dict(split=split, lab=lab, n=len(g), flat_stored=int(g["stored_flat"].eq(True).sum()),
                                 nonfinite_stored=int((g["stored_nonfinite_rows"].fillna(0) > 0).sum())))
    info = dict(created=datetime.datetime.now().isoformat(timespec="seconds"), git=_git(), scope=scope,
                n_folders=len(df), ready=bool(ready), n_not_ready=int(len(not_ready)),
                nonfinite_sojr=[f"{r['split']}/{r['activity']}/{r['trial']}" for _, r in nonfinite.iterrows()],
                flat_cv=args.flat_cv, flat_dr=args.flat_dr, max_peak=args.max_peak, max_moment=args.max_moment,
                summaries=[os.path.relpath(p, REPO) for p in args.summary], seconds=round(time.time() - t0, 1))
    json.dump(info, open(os.path.join(args.out, "run_info.json"), "w"), indent=1)
    lines = ["# SO+JR training set: coverage and label filters on the same trials", "",
             f"Created {info['created']} (git {info['git']}) by analysis/thesis/sojr_trainset.py; reads files only.",
             f"Scope: {scope}; {len(df)} folders with {STORED}/ labels.", "",
             f"**{'READY' if ready else 'NOT READY'}**: {len(df) - len(not_ready)} of {len(df)} have a complete "
             f"{SOJR}/ file; {len(nonfinite)} of those hold non-finite values (dropped by the loaders).", "",
             "## Coverage (split x activity x lab)", "", _md(cov), "",
             "## Relabel summary statuses (lab x status)", "",
             "Rows are keyed by folder name, which repeats across roots, so these are counts, not a per-folder test.",
             "", _md(status) if len(status) else "(no summary)", "",
             f"## Flat filter on the same walking trials (CV < {args.flat_cv:g} or range < {args.flat_dr:g} BW)", "",
             "Trials with finite labels in both sets. flat_only_stored = removed under the stored labels but kept "
             "under SO+JR.", "", _md(flat) if len(flat) else "(no trials with both label sets)", "",
             "## Flat filter on all stored-label walking trials", "", _md(pd.DataFrame(flat_all)) if flat_all else "",
             "", f"## Peak tests on the same walking trials (resultant > {args.max_peak:g} BW, |M_x| > "
             f"{args.max_moment:g} BW*H in any label frame)", "", _md(peak) if len(peak) else "(none)", "",
             "Files: coverage.csv, relabel_status.csv, trials.csv, flat_same_trials.csv, peak_same_trials.csv, "
             "run_info.json"]
    open(os.path.join(args.out, "README.md"), "w").write("\n".join(lines) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

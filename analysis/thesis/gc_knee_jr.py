#!/usr/bin/env python3
"""OpenSim SO+JR knee labels for the Grand Challenge segment folders: rerun JointReaction only, applying the static
optimization forces through forces_file (no static optimization is solved again).

Why the stored labels are not SO+JR: pipeline/batch_process.py ran StaticOptimization and JointReaction in one
AnalyzeTool with an empty <forces_file />. OpenSim 4.5.2 documents that property as: "If a file name is provided, the
forces for all actuators will be applied according to values specified in the forces_file instead of being computed
from the states. This option should be used to calculate joint reactions from static optimization results."
StaticOptimization solves on its own working copy of the model, so with an empty forces_file the stored jcf_output/
knee loads carry muscle forces computed from the model states (default activation), not the SO solution. The stored
files show it: the stored knee |F| stays below the SO vasti force alone in most frames where that force exceeds 1 BW
(fregly_2GC_t00_r01: vasti up to 5.0 BW, stored knee |F| at most 0.98 BW). The `stored_*` columns of the summary
measure this for every folder, and stored_right_max_abs_diff is the decisive check (< 1e-3 BW would mean the stored
labels were SO+JR after all).

What it does, per folder: copy jcf_output/_analyze_setup.xml (model with reserves, padded IK, external loads, time
range, 6 Hz coordinate filter), switch StaticOptimization off, point JointReaction's forces_file at the folder's
jcf_output/BatchJCF_StaticOptimization_force.sto, and report both knees (walker_knee_r and walker_knee_l, on and in the
child body): JW and DM are right implants, SC a left one, and every subject gets labels from the same run.

Output per folder, jcf_output_sojr/: _jr_setup.xml, BatchJCF_JointReaction_ReactionLoads.sto
Summary CSV, one row per folder:
    stored_right_max_abs_diff   max |new - stored| of the walker_knee_r F (BW) and M (BW*H) columns
    vas_r_frames, stored_below_vas_r_frac, new_below_vas_r_frac (and _l)
                                frames whose SO vasti force (vaslat + vasmed + vasint of that side) exceeds 1 BW, and the
                                fraction of them in which that knee's |F| is below the vasti force, for the stored and the
                                new labels. The knee reaction has to balance the vasti pull on the tibia (the other loads
                                on the shank mostly add to it), so consistent SO+JR labels give a fraction near 0
    {r,l}_stance_frames, {r,l}_fy_compressive_frac, {r,l}_mx_medial_pos_frac, {r,l}_fy_peak_bw
                                sign checks on each knee's stance frames (|GRF_y| > 5 % BW); medial-positive M_x is +mx
                                for the right knee and -mx for the left

Usage (repo root, jcf env; JR only):
    # decisive one-folder probe (seconds)
    conda run -n jcf python analysis/thesis/gc_knee_jr.py --pattern 'fregly_3GC_t18_r00' --workers 1 \\
        --summary results/thesis/gc/knee_jr/probe_t18.csv
    # every GC folder (JW, DM, SC; 139 folders, about a minute on 20 workers)
    conda run -n jcf python analysis/thesis/gc_knee_jr.py --workers 20
    # setup files only, or the stored-label check only (no OpenSim):
    conda run -n jcf python analysis/thesis/gc_knee_jr.py --dry-run
    python -I analysis/thesis/gc_knee_jr.py --stored-only --summary results/thesis/gc/knee_jr/stored_check.csv
Then rerun analysis/thesis/grand_challenge.py; it reads jcf_output_sojr/ as the source `osim_sojr` automatically.
"""
import os as _os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")   # one thread per worker: 20 workers x 48 default threads stalled the machine
import argparse
import glob
import json
import multiprocessing as mp
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

OUT_SUBDIR = "jcf_output_sojr"
JOINTS = ("walker_knee_r", "walker_knee_l")
SO_FORCES = "BatchJCF_StaticOptimization_force.sto"
JR_FILE = "BatchJCF_JointReaction_ReactionLoads.sto"
DEFAULT_ROOTS = (os.path.join(REPO, "jcf", "full_duration", "training", "walking"),
                 os.path.join(REPO, "jcf", "full_duration", "testing", "walking"))
DEFAULT_PATTERN = "fregly_[123]GC_*"
VASTI = ("vaslat", "vasmed", "vasint")
G = 9.81


def jr_only_setup(src_xml, forces_file, results_dir, joints=JOINTS):
    """The folder's SO+JR AnalyzeTool setup turned into a JR-only setup (ElementTree; no OpenSim needed).

    StaticOptimization is switched off; JointReaction applies forces_file and reports `joints` on and in the child
    body; the results go to results_dir. Everything else (model, coordinates, loads, times, filter) is unchanged."""
    tree = ET.parse(src_xml)
    tool = tree.getroot().find("AnalyzeTool")
    if tool is None:
        raise ValueError(f"{src_xml}: no AnalyzeTool")
    tool.find("results_directory").text = os.path.abspath(results_dir)
    # Most setups were written on the labelling Mac (/Users/gellertli/...); point every input file that does not exist here
    # at the same file inside this trial folder (jcf_output/<name> or <folder>/<name>).
    folder = os.path.dirname(os.path.dirname(os.path.abspath(src_xml)))
    for tag in ("model_file", "coordinates_file", "external_loads_file", "states_file", "speeds_file"):
        el = tool.find(tag)
        if el is None or not (el.text or "").strip() or os.path.exists(el.text.strip()):
            continue
        name = os.path.basename(el.text.strip().replace("\\", "/"))
        local = [c for c in (os.path.join(folder, "jcf_output", name), os.path.join(folder, name)) if os.path.exists(c)]
        if not local:
            raise FileNotFoundError(f"{src_xml}: {tag} {el.text.strip()} not found, and no local {name}")
        el.text = local[0]
    objs = tool.find("AnalysisSet").find("objects")
    so = objs.find("StaticOptimization")
    if so is not None:
        so.find("on").text = "false"
    jr = objs.find("JointReaction")
    if jr is None:
        raise ValueError(f"{src_xml}: no JointReaction analysis")
    jr.find("on").text = "true"
    jr.find("forces_file").text = os.path.abspath(forces_file)
    jr.find("joint_names").text = " " + " ".join(joints)
    jr.find("apply_on_bodies").text = " " + " ".join("child" for _ in joints)
    jr.find("express_in_frame").text = " " + " ".join("child" for _ in joints)
    return tree


def read_sto(path):
    with open(path) as f:
        for i, line in enumerate(f):
            if line.strip().lower() == "endheader":
                break
    return pd.read_csv(path, sep=r"\s+", skiprows=i + 1)


def knee_prefix(side):
    return f"walker_knee_{side}_on_tibia_{side}_in_tibia_{side}_"


def knee_force_bw(jr, side, bw, t=None):
    """|F| of one knee (BW) from a JR table, on its own rows or interpolated onto times t; None without the columns."""
    pre = knee_prefix(side)
    if pre + "fy" not in jr:
        return None
    t = jr["time"].to_numpy(float) if t is None else np.asarray(t, float)
    f = [np.interp(t, jr["time"].to_numpy(float), jr[pre + c].to_numpy(float)) for c in ("fx", "fy", "fz")]
    return np.sqrt(sum(v ** 2 for v in f)) / bw


def vasti_bw(so, side, bw):
    """Summed SO vasti force (BW) of one side on the SO rows; None if the model has no vasti of that side."""
    cols = [f"{m}_{side}" for m in VASTI if f"{m}_{side}" in so]
    return so[cols].sum(axis=1).to_numpy(float) / bw if cols else None


def below_vasti(so, jr, side, bw, min_bw=1.0):
    """(frames with vasti > min_bw, fraction of them in which the knee |F| is below the vasti force)."""
    vas = vasti_bw(so, side, bw)
    kf = knee_force_bw(jr, side, bw, so["time"]) if jr is not None else None
    if vas is None or kf is None:
        return 0, np.nan
    big = vas > min_bw
    return int(big.sum()), (float(np.mean(kf[big] < vas[big])) if big.any() else np.nan)


def check_folder(d, out_subdir=OUT_SUBDIR, stored_only=False):
    """Stored-vs-new difference, vasti consistency and knee sign checks for one folder."""
    meta = json.load(open(os.path.join(d, "metadata.json")))
    bw, H = float(meta["mass_kg"]) * G, float(meta.get("height_m", 1.7))
    so = read_sto(os.path.join(d, "jcf_output", SO_FORCES))
    old = read_sto(os.path.join(d, "jcf_output", JR_FILE))
    new = None if stored_only else read_sto(os.path.join(d, out_subdir, JR_FILE))
    row = dict(folder=os.path.basename(d), n_so=len(so), n_new=len(new) if new is not None else 0)
    if new is not None:
        pre, t = knee_prefix("r"), old["time"].to_numpy()
        diffs = [np.max(np.abs(np.interp(t, new["time"], new[pre + c]) - old[pre + c])) / scale
                 for c, scale in (("fx", bw), ("fy", bw), ("fz", bw), ("mx", bw * H), ("my", bw * H), ("mz", bw * H))
                 if pre + c in new and pre + c in old]
        row["stored_right_max_abs_diff"] = float(max(diffs)) if diffs else np.nan
    for side in "rl":
        n, frac = below_vasti(so, old, side, bw)
        row[f"vas_{side}_frames"] = n
        row[f"stored_below_vas_{side}_frac"] = frac
        if new is not None:
            row[f"new_below_vas_{side}_frac"] = below_vasti(so, new, side, bw)[1]
        kf = knee_force_bw(old, side, bw)
        row[f"stored_{side}_F_max_bw"] = float(kf.max()) if kf is not None else np.nan
        vas = vasti_bw(so, side, bw)
        row[f"so_vas_{side}_max_bw"] = float(vas.max()) if vas is not None else np.nan
    if new is None:
        return row
    g = read_sto(os.path.join(d, "grf_data.mot"))
    for side, sign in (("r", 1.0), ("l", -1.0)):
        pre = knee_prefix(side)
        if pre + "fy" not in new:
            continue
        stance = np.interp(new["time"], g["time"], np.abs(g[f"calcn_{side}_force_vy"])) > 0.05 * bw
        row[f"{side}_stance_frames"] = int(stance.sum())
        kf = knee_force_bw(new, side, bw)
        row[f"new_{side}_F_max_bw"] = float(kf.max())
        if stance.any():
            row[f"{side}_fy_compressive_frac"] = float(np.mean(new.loc[stance, pre + "fy"] < 0))
            row[f"{side}_mx_medial_pos_frac"] = float(np.mean(sign * new.loc[stance, pre + "mx"] > 0))
            row[f"{side}_fy_peak_bw"] = float(np.abs(new.loc[stance, pre + "fy"]).max() / bw)
    return row


def run_folder(job):
    d, out_subdir, dry_run, force, stored_only = job
    out = os.path.join(d, out_subdir)
    src = os.path.join(d, "jcf_output", "_analyze_setup.xml")
    forces = os.path.join(d, "jcf_output", SO_FORCES)
    name = os.path.basename(d)
    if not (os.path.exists(src) and os.path.exists(forces)):
        return dict(folder=name, status="no SO output")
    if stored_only:
        return dict(check_folder(d, out_subdir, stored_only=True), status="stored check")
    if os.path.exists(os.path.join(out, JR_FILE)) and not force:
        return dict(check_folder(d, out_subdir), status="exists")
    os.makedirs(out, exist_ok=True)
    setup = os.path.join(out, "_jr_setup.xml")
    jr_only_setup(src, forces, out).write(setup, encoding="UTF-8", xml_declaration=True)
    if dry_run:
        return dict(folder=name, status="setup written")
    try:
        import opensim as osm
        try:
            osm.Logger.setLevelString("warn")
        except Exception:
            pass
        tool = osm.AnalyzeTool(setup)
        tool.run()
        if not os.path.exists(os.path.join(out, JR_FILE)):
            return dict(folder=name, status="no JR output")
        return dict(check_folder(d, out_subdir), status="ok")
    except Exception as e:                       # one failing folder must not stop the pool
        return dict(folder=name, status=f"failed: {type(e).__name__}: {e}")


def report(df):
    """Console verdict on the stored labels and on the new ones."""
    print(df["status"].value_counts().to_string())
    if "stored_right_max_abs_diff" in df and df.stored_right_max_abs_diff.notna().any():
        dmax = df.stored_right_max_abs_diff.max()
        dmed = df.stored_right_max_abs_diff.median()
        verdict = ("the stored jcf_output/ labels ARE SO+JR (no relabelling needed)" if dmax < 1e-3 else
                   "the stored jcf_output/ labels are NOT SO+JR (muscle forces from the states)")
        print(f"stored vs SO+JR, walker_knee_r: max |diff| {dmax:.3g}, median {dmed:.3g} (BW, BW*H) over "
              f"{df.stored_right_max_abs_diff.notna().sum()} folders -> {verdict}")
    for side in "rl":
        c = f"stored_below_vas_{side}_frac"
        if c in df and df[c].notna().any():
            line = (f"walker_knee_{side}: {int(df[f'vas_{side}_frames'].sum())} frames with SO vasti > 1 BW; knee |F| below "
                    f"the vasti force in a median {df[c].median():.2f} of them (stored)")
            if f"new_below_vas_{side}_frac" in df and df[f"new_below_vas_{side}_frac"].notna().any():
                line += f", {df[f'new_below_vas_{side}_frac'].median():.2f} (SO+JR)"
            print(line)
    for side in "rl":
        if f"{side}_mx_medial_pos_frac" in df and df[f"{side}_mx_medial_pos_frac"].notna().any():
            print(f"walker_knee_{side} stance: median fraction with medial-positive M_x "
                  f"{df[f'{side}_mx_medial_pos_frac'].median():.2f}, with F_y < 0 "
                  f"{df[f'{side}_fy_compressive_frac'].median():.2f}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", nargs="+", default=list(DEFAULT_ROOTS))
    ap.add_argument("--pattern", default=DEFAULT_PATTERN)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--out-subdir", default=OUT_SUBDIR)
    ap.add_argument("--dry-run", action="store_true", help="write the JR-only setup files, do not run OpenSim")
    ap.add_argument("--stored-only", action="store_true",
                    help="only the vasti consistency check of the stored labels (no OpenSim, writes nothing in the folders)")
    ap.add_argument("--force", action="store_true", help="rerun folders that already have output")
    ap.add_argument("--summary", default=os.path.join(REPO, "results", "thesis", "gc", "knee_jr", "summary.csv"))
    args = ap.parse_args(argv)
    dirs = [d for r in args.root for d in sorted(glob.glob(os.path.join(r, args.pattern))) if os.path.isdir(d)]
    jobs = [(d, args.out_subdir, args.dry_run, args.force, args.stored_only) for d in dirs]
    print(f"{len(jobs)} folders under {', '.join(args.root)} matching {args.pattern}")
    if args.workers > 1 and len(jobs) > 1:
        # a fresh process per OpenSim run (memory); the file-only check reuses its workers
        per_child = None if args.stored_only else 1
        with mp.get_context("spawn").Pool(min(args.workers, len(jobs)), maxtasksperchild=per_child) as pool:
            rows = []
            for r in pool.imap_unordered(run_folder, jobs, chunksize=1 if per_child else 16):
                rows.append(r)
                if not args.stored_only or len(jobs) <= 200:
                    print(f"  {r['folder']}: {r['status']}", flush=True)
    else:
        rows = [run_folder(j) for j in jobs]
    if not rows:
        raise SystemExit("no folders matched")
    df = pd.DataFrame(rows).sort_values("folder")
    os.makedirs(os.path.dirname(os.path.abspath(args.summary)), exist_ok=True)
    df.to_csv(args.summary, index=False)
    pd.set_option("display.width", 200)
    report(df)
    print(f"wrote {args.summary}")


if __name__ == "__main__":
    main()

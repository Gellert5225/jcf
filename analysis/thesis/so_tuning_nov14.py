#!/usr/bin/env python3
"""Score StaticOptimization settings on William's Nov 14 export with the label-QC frame rules, and build the exact SO
optimum as a reference run.

Runs: the labeller default (jcf_output/, jcf_output_sojr/), analysis/thesis/so_variant.py runs (tolerance, iterations,
muscle physiology, a hyperextension capsule at the right knee), and two exact-optimum runs (analysis/thesis/so_exact.py,
built here if missing): jcf_output_exact/ (re-solve of the default run) and so_variants/optimum_capsule/ (re-solve of the
capsule run), each with OpenSim's JointReaction in <dir>_sojr/, so they are scored like any run.

Each run's SO and SO+JR outputs go through so_label_qc.frame_table / mask_arrays (temporary folder of symlinks),
restricted to right single stance (right foot > 5 % BW, left foot below). Per run: Ipopt failures (opensim.log) and how
many fall in right single stance, share of right-single-stance frames passing the six SO rules (so_any_r false), each
rule's frame count, F_medial = 0.5|Fy| - Mx/d (-Mx, BW) mean and peak, peak gastrocnemius activation, largest |knee
reserve|, capsule torque, and how far the stored solution is from the exact optimum of its problem (cost ratio median /
max, % of frames above so_label_qc.COST_RATIO_NONOPT).

Writes results/thesis/exports/so_tuning_nov14.csv (summary), so_tuning_nov14_frames.csv.gz (per frame and run) and
so_failures_nov14.csv (Ipopt failure times).
Usage (repo root): python -I analysis/thesis/so_tuning_nov14.py [--rebuild-optimum]
"""
import argparse
import json
import os
import re
import site
import sys
import tempfile

sys.path.append(site.getusersitepackages())
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from analysis.thesis import so_label_qc as q  # noqa: E402
from analysis.thesis import so_exact  # noqa: E402
from analysis.thesis.gc_knee_jr import read_sto  # noqa: E402

FOLDER = os.path.join(REPO, "jcf", "external", "william_2026-11-14")
OUT = os.path.join(REPO, "results", "thesis", "exports")
PRE = "walker_knee_r_on_tibia_r_in_tibia_r_"
D_M = 0.045
CAPSULE_N_M = 1000.0      # so_variant.py --capsule of the capsule run
# tag, label, SO directory (relative to FOLDER), SO+JR directory, kind
RUNS = (
    ("default", "tolerance 1e-4, 200 iterations (labeller default)", "jcf_output", "jcf_output_sojr", "solver"),
    ("tol1e-3", "tolerance 1e-3, 200 iterations", "so_tol1e-3", "so_tol1e-3/sojr", "solver"),
    ("tol1e-2", "tolerance 1e-2, 200 iterations", "so_variants/tol1e-2", "so_variants/tol1e-2/sojr", "solver"),
    ("tol1e-1", "tolerance 1e-1, 200 iterations", "so_variants/tol1e-1", "so_variants/tol1e-1/sojr", "solver"),
    ("it1000", "tolerance 1e-4, 1000 iterations", "so_iter1000", "so_iter1000/sojr", "solver"),
    ("tol1e-3_it1000", "tolerance 1e-3, 1000 iterations", "so_variants/tol1e-3_it1000", "so_variants/tol1e-3_it1000/sojr",
     "solver"),
    ("nophys", "no muscle force-length-velocity", "so_variants/nophys", "so_variants/nophys/sojr", "muscle"),
    ("capsule", "hyperextension capsule at the right knee", "so_variants/capsule", "so_variants/capsule/sojr", "model"),
    ("optimum_plain", "exact optimum, plain model", "jcf_output_exact", "jcf_output_exact_sojr", "optimum"),
    ("optimum_capsule", "exact optimum, capsule", "so_variants/optimum_capsule", "so_variants/optimum_capsule_sojr",
     "optimum"),
)
# exact-optimum run (so_exact.build) -> (run whose stored solution defines the problem, capsule optimal force)
EXACT = {"optimum_plain": ("jcf_output", 0.0), "optimum_capsule": ("so_variants/capsule", CAPSULE_N_M)}
# the problem each run is compared with for the cost ratio (None: different muscle model, not comparable)
PROBLEM = {"capsule": "optimum_capsule", "optimum_capsule": "optimum_capsule", "nophys": None}


def stored_cost(so_dir, times, mus, acts):
    """sum a^2 + sum u^2 of a run's stored solution over the muscles and the frame-problem actuators, at each time."""
    so = read_sto(q.results_file(os.path.join(FOLDER, so_dir), ".", q.SO_FORCES))
    act = read_sto(q.results_file(os.path.join(FOLDER, so_dir), ".", q.SO_ACT))
    st = so["time"].to_numpy(float)
    out = []
    for time in times:
        j = int(np.argmin(np.abs(st - time)))
        out.append(float((act.loc[j, mus].to_numpy(float) ** 2).sum()
                         + sum((so.loc[j, n] / fo) ** 2 for n, fo in acts if n in so)))
    return np.array(out)


def failures(log):
    if not os.path.exists(log):
        return None
    return sorted(float(x.rstrip(".")) for x in re.findall(r"could not find a solution at time = ([0-9.]+)", open(log).read()))


def view(tmp, tag, so_dir, jr_dir):
    """A folder laid out like a labelled trial (jcf_output/, sojr/) whose SO and JR files are this run's."""
    d = os.path.join(tmp, tag)
    os.makedirs(d)
    for name, target in (("metadata.json", "metadata.json"), ("grf_data.mot", "grf_data.mot"),
                         ("ik_results.mot", "ik_results.mot"), ("jcf_output", so_dir), ("sojr", jr_dir)):
        os.symlink(os.path.join(FOLDER, target), os.path.join(d, name))
    return d


def score(tmp, tag, label, so_dir, jr_dir, kind):
    d = view(tmp, tag, so_dir, jr_dir)
    ft = q.frame_table(d, "sojr")
    m = q.mask_arrays(d, "sojr", ft)
    t = m["time"]
    single = m["stance_r"] & ~m["stance_l"]
    ok = single & ~m["so_any_r"]
    bw = ft.attrs["bw"]
    jr = read_sto(q.results_file(d, "sojr", q.JR_FILE))
    fm = (0.5 * np.abs(jr[PRE + "fy"]) / bw - jr[PRE + "mx"] / (bw * D_M)).to_numpy()
    so = read_sto(q.results_file(d, "jcf_output", q.SO_FORCES))
    act = read_sto(q.results_file(d, "jcf_output", q.SO_ACT))
    s_so = np.interp(so.time.to_numpy(), t, single.astype(float)) > 0.5
    fail = failures(os.path.join(FOLDER, so_dir, "opensim.log"))
    in_single = None if fail is None else int(sum(np.interp(x, t, single.astype(float)) > 0.5 for x in fail))
    row = dict(tag=tag, setting=label, kind=kind, ipopt_failures=None if fail is None else len(fail),
               failures_in_right_single=in_single, single_frames=int(single.sum()),
               pass_pct=100.0 * ok.sum() / single.sum(), fmed_mean=fm[single].mean(), fmed_peak=fm[single].max(),
               fmed_mean_pass=fm[ok].mean() if ok.any() else np.nan,
               gastroc_max=float(np.maximum(act.gasmed_r, act.gaslat_r).to_numpy()[s_so].max()),
               knee_reserve_max_nm=float(np.abs(so.reserve_knee_angle_r).to_numpy()[s_so].max()))
    for r in q.MASK_SO_RULES:
        row[f"n_{r}"] = int((m[f"{r}_r"] & single).sum())
    if "capsule_knee_angle_r" in so:
        row["capsule_max_nm"] = float(so.capsule_knee_angle_r.to_numpy()[s_so].max())
    frames = pd.DataFrame(dict(tag=tag, time=t, fmed=fm, single=single, so_any=m["so_any_r"]))
    fails = pd.DataFrame(dict(tag=tag, setting=label, time=fail or []))
    return row, frames, fails


def done(jr_dir):
    jd = os.path.join(FOLDER, jr_dir)
    return os.path.isdir(jd) and any(f.endswith("ReactionLoads.sto") for f in os.listdir(jd))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rebuild-optimum", action="store_true", help="rebuild the exact-optimum runs even if present")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    jr_dirs = {tag: jr for tag, _, _, jr, _ in RUNS}
    so_dirs = {tag: so_dir for tag, _, so_dir, _, _ in RUNS}
    todo = {tag: v for tag, v in EXACT.items() if (a.rebuild_optimum or not done(jr_dirs[tag])) and done(
        "jcf_output_sojr" if v[0] == "jcf_output" else v[0] + "/sojr")}
    if todo:
        src = os.path.join(FOLDER, "jcf_output")
        times = read_sto(q.results_file(src, ".", q.SO_FORCES)).time.to_numpy(float)
        geometry = so_exact.frame_geometry(FOLDER, times, cutoff=so_exact.lowpass_hz(os.path.join(src, "_analyze_setup.xml")))
        for tag, (src_dir, cap) in todo.items():
            so_exact.build(FOLDER, src_dir, so_dirs[tag], cap, geometry)

    rows, frames, fails = [], [], []
    with tempfile.TemporaryDirectory() as tmp:
        for tag, label, so_dir, jr_dir, kind in RUNS:
            if not done(jr_dir):
                print("missing, skipped:", tag)
                continue
            r, f, x = score(tmp, tag, label, so_dir, jr_dir, kind)
            rows.append(r)
            frames.append(f)
            fails.append(x)
    frames = pd.concat(frames)
    f0 = frames[frames.tag == "default"]
    times = f0.time[f0.single].to_numpy()
    for r in rows:
        ref = PROBLEM.get(r["tag"], "optimum_plain")
        if ref is None or not os.path.exists(os.path.join(FOLDER, so_dirs[ref], "problem.json")):
            continue
        p = json.load(open(os.path.join(FOLDER, so_dirs[ref], "problem.json")))
        o = pd.read_csv(os.path.join(FOLDER, so_dirs[ref], "per_frame.csv.gz"))
        best = np.interp(times, o.time, o.cost_opt)
        ratio = stored_cost(so_dirs[r["tag"]], times, p["muscles"], p["actuators"]) / best
        r.update(cost_ratio_median=float(np.median(ratio)), cost_ratio_max=float(ratio.max()),
                 nonopt_pct=float(100 * np.mean(ratio > q.COST_RATIO_NONOPT)))
    tab = pd.DataFrame(rows)
    ref = f0.fmed.to_numpy()
    tab["max_diff_vs_default_single"] = tab.tag.map(
        {g: np.abs(f.fmed.to_numpy() - ref)[f0.single.to_numpy()].max() for g, f in frames.groupby("tag")})
    tab.to_csv(os.path.join(OUT, "so_tuning_nov14.csv"), index=False)
    frames.to_csv(os.path.join(OUT, "so_tuning_nov14_frames.csv.gz"), index=False)
    pd.concat(fails).to_csv(os.path.join(OUT, "so_failures_nov14.csv"), index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print(tab.drop(columns=["setting"]).round(2).to_string(index=False))


if __name__ == "__main__":
    main()

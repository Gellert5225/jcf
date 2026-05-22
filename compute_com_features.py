"""
compute_com_features.py

For each AddBiomechanics trial, compute and cache:
  - COM position (3D, world frame)
  - COM velocity (3D, numerical derivative)
  - calcn_r world-frame position (3D)
  - calcn_l world-frame position (3D)

…using OpenSim's Python API and the per-trial scaled_model.osim + ik_results.mot.
Caches results as <trial_dir>/com_features.npz so subsequent training/test
runs read pre-computed features instead of invoking OpenSim per frame.

Usage:
    conda run -n jcf python compute_com_features.py
    conda run -n jcf python compute_com_features.py --limit 200      # quick test on first 200 trials
    conda run -n jcf python compute_com_features.py --workers 8      # parallel
    conda run -n jcf python compute_com_features.py --overwrite      # recompute even if cached
"""
import os
import sys
import argparse
import traceback
import multiprocessing as mp
from functools import partial

import numpy as np
import pandas as pd


def _opensim():
    try:
        import opensim
        return opensim
    except ImportError:
        raise SystemExit(
            "OpenSim Python not found. Install:\n"
            "    conda install -c opensim-org opensim\n"
        )


def load_mot(path, skiprows=6):
    return pd.read_csv(path, sep=r"\s+", skiprows=skiprows)


def compute_for_trial(trial_dir, overwrite=False):
    """Compute COM and per-foot positions for one trial. Returns status string."""
    opensim = _opensim()

    model_path = os.path.join(trial_dir, "scaled_model.osim")
    ik_path = os.path.join(trial_dir, "ik_results.mot")
    out_path = os.path.join(trial_dir, "com_features.npz")

    if os.path.exists(out_path) and not overwrite:
        return "skip"
    if not (os.path.exists(model_path) and os.path.exists(ik_path)):
        return "missing"

    model = opensim.Model(model_path)
    state = model.initSystem()

    ik = load_mot(ik_path)
    time = ik["time"].values.astype(np.float64)
    T = len(time)

    # Build coordinate index → IK column name mapping
    coord_set = model.getCoordinateSet()
    n_coords = coord_set.getSize()
    coord_to_ik_col = {}
    for i in range(n_coords):
        name = coord_set.get(i).getName()
        if name in ik.columns:
            coord_to_ik_col[i] = name

    # Foot bodies
    body_set = model.getBodySet()
    try:
        calcn_r = body_set.get("calcn_r")
        calcn_l = body_set.get("calcn_l")
    except Exception as e:
        return f"no_feet_bodies:{e}"

    com_pos = np.zeros((T, 3), dtype=np.float64)
    calcn_r_pos = np.zeros((T, 3), dtype=np.float64)
    calcn_l_pos = np.zeros((T, 3), dtype=np.float64)

    zero_vec = opensim.Vec3(0.0, 0.0, 0.0)

    # Read column data once into numpy for speed
    ik_arr = {name: ik[name].values for _, name in coord_to_ik_col.items()}

    for t in range(T):
        for i, name in coord_to_ik_col.items():
            coord_set.get(i).setValue(state, float(ik_arr[name][t]), False)
        model.realizePosition(state)

        com_vec = model.calcMassCenterPosition(state)
        com_pos[t, 0] = com_vec.get(0)
        com_pos[t, 1] = com_vec.get(1)
        com_pos[t, 2] = com_vec.get(2)

        r_vec = calcn_r.findStationLocationInGround(state, zero_vec)
        calcn_r_pos[t, 0] = r_vec.get(0)
        calcn_r_pos[t, 1] = r_vec.get(1)
        calcn_r_pos[t, 2] = r_vec.get(2)

        l_vec = calcn_l.findStationLocationInGround(state, zero_vec)
        calcn_l_pos[t, 0] = l_vec.get(0)
        calcn_l_pos[t, 1] = l_vec.get(1)
        calcn_l_pos[t, 2] = l_vec.get(2)

    # Numerical COM velocity (central diff via np.gradient)
    if T >= 2:
        com_vel = np.gradient(com_pos, time, axis=0)
    else:
        com_vel = np.zeros_like(com_pos)

    np.savez(
        out_path,
        time=time.astype(np.float32),
        com_pos=com_pos.astype(np.float32),
        com_vel=com_vel.astype(np.float32),
        calcn_r_pos=calcn_r_pos.astype(np.float32),
        calcn_l_pos=calcn_l_pos.astype(np.float32),
    )
    return "ok"


def _safe_compute(trial_dir, overwrite=False):
    try:
        return (trial_dir, compute_for_trial(trial_dir, overwrite=overwrite))
    except Exception as e:
        return (trial_dir, f"error:{e}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="./jcf/full_duration")
    p.add_argument("--limit", type=int, default=None,
                   help="Limit number of trials (for testing).")
    p.add_argument("--overwrite", action="store_true",
                   help="Recompute even if com_features.npz exists.")
    p.add_argument("--workers", type=int, default=1,
                   help="Parallel workers (default 1). OpenSim model loading is "
                        "heavy so 4-8 workers usually peaks throughput.")
    p.add_argument("--exclude", type=str, nargs="+", default=None,
                   help="Exclude dataset prefixes (e.g. han fregly).")
    args = p.parse_args()

    excluded = set(args.exclude) if args.exclude else set()

    trial_dirs = []
    for split in ["training", "testing"]:
        for act in ["walking", "static"]:
            base = os.path.join(args.root, split, act)
            if not os.path.isdir(base):
                continue
            for name in sorted(os.listdir(base)):
                if any(name.startswith(f"{ex}_") for ex in excluded):
                    continue
                d = os.path.join(base, name)
                if os.path.isdir(d):
                    trial_dirs.append(d)

    if args.limit:
        trial_dirs = trial_dirs[: args.limit]

    print(f"Processing {len(trial_dirs)} trials with {args.workers} worker(s)...")

    counts = {"ok": 0, "skip": 0, "missing": 0, "error": 0}

    if args.workers <= 1:
        for i, td in enumerate(trial_dirs):
            _, status = _safe_compute(td, overwrite=args.overwrite)
            key = "error" if status.startswith("error") or status.startswith("no_feet") else status
            counts[key] = counts.get(key, 0) + 1
            if (i + 1) % 100 == 0:
                print(f"  [{i+1}/{len(trial_dirs)}] {counts}", flush=True)
            if status.startswith("error") or status.startswith("no_feet"):
                print(f"  ERROR {td}: {status}", flush=True)
    else:
        worker = partial(_safe_compute, overwrite=args.overwrite)
        with mp.Pool(processes=args.workers) as pool:
            for i, (td, status) in enumerate(pool.imap_unordered(worker, trial_dirs, chunksize=4)):
                key = "error" if status.startswith("error") or status.startswith("no_feet") else status
                counts[key] = counts.get(key, 0) + 1
                if (i + 1) % 100 == 0:
                    print(f"  [{i+1}/{len(trial_dirs)}] {counts}", flush=True)
                if status.startswith("error") or status.startswith("no_feet"):
                    print(f"  ERROR {td}: {status}", flush=True)

    print(f"\nFinal counts: {counts}")


if __name__ == "__main__":
    main()

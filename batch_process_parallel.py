"""
Parallel batch processing wrapper for batch_process.py
=======================================================
Runs the same pipeline as batch_process.py but uses multiprocessing.Pool
with thread-count throttling to avoid oversubscription.

OpenSim/Simbody + numpy/MKL each spawn internal threads per process.
With N workers × K threads, you get N*K threads fighting for K cores.
Fix: set OMP/MKL/OPENBLAS threads to 1 BEFORE importing anything,
then let mp.Pool handle the parallelism at the process level.

Usage:
    conda run -n jcf python batch_process_parallel.py              # auto-detect cores
    conda run -n jcf python batch_process_parallel.py --workers 4  # 4 workers
"""

# ── Thread throttling (MUST come before any other imports) ────────────────────
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import sys
import json
import glob
import argparse
import multiprocessing as mp
import time

# Import everything from batch_process (uses throttled threads)
import batch_process
from batch_process import (
    DATASET_PREFIX, process_one_subject,
)

# Override input/output paths — set at import time, updated in main() per split
B3D_ROOTS = {
    "training": "./with_arm/training",
    "testing": "./with_arm/testing",
}


def _configure_paths(split):
    """Set batch_process module globals for the given split (training/testing)."""
    batch_process.B3D_ROOT = B3D_ROOTS[split]
    out = f"./jcf/full_duration/{split}"
    batch_process.OUTPUT_ROOT = out
    batch_process.OUTPUT_ROOT_RUNNING = os.path.join(out, "running")
    batch_process.OUTPUT_ROOT_WALKING = os.path.join(out, "walking")
    batch_process.SCAN_LOG = os.path.join(out, "scan_failures.txt")
    return out


def main():
    parser = argparse.ArgumentParser(description="Parallel batch processing")
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of parallel workers (default: physical cores - 1)')
    parser.add_argument('--split', type=str, default='training',
                        choices=['training', 'testing', 'both'],
                        help='Which data split to process (default: training)')
    args = parser.parse_args()

    n_workers = args.workers
    if n_workers is None:
        try:
            n_workers = max(1, len(os.sched_getaffinity(0)) - 1)
        except AttributeError:
            n_workers = max(1, mp.cpu_count() - 1)

    splits = ['training', 'testing'] if args.split == 'both' else [args.split]

    for split in splits:
        out = _configure_paths(split)
        os.makedirs(out, exist_ok=True)
        os.makedirs(batch_process.OUTPUT_ROOT_RUNNING, exist_ok=True)
        os.makedirs(batch_process.OUTPUT_ROOT_WALKING, exist_ok=True)

        # Find all b3d files
        b3d_files = []
        for dataset in sorted(os.listdir(batch_process.B3D_ROOT)):
            dataset_path = os.path.join(batch_process.B3D_ROOT, dataset)
            if not os.path.isdir(dataset_path):
                continue
            for subject in sorted(os.listdir(dataset_path)):
                subj_path = os.path.join(dataset_path, subject)
                b3d_list = glob.glob(os.path.join(subj_path, "*.b3d"))
                if b3d_list:
                    b3d_files.append((dataset, subject, b3d_list[0]))

        total = len(b3d_files)
        print(f"\n{'=' * 60}")
        print(f"Split: {split}")
        print(f"Found {total} subjects to process")
        print(f"Input:  {batch_process.B3D_ROOT}")
        print(f"Output: {out}")
        print(f"Workers: {n_workers}")
        print("=" * 60)

        worker_args = [
            (ds, subj, path, idx, total)
            for idx, (ds, subj, path) in enumerate(b3d_files)
        ]

        results = {"success": [], "skip": [], "scan_fail": [], "convert_fail": [],
                   "jcf_fail": [], "error": []}

        t_start = time.time()

        with mp.Pool(processes=n_workers) as pool:
            for output_name, status in pool.imap_unordered(process_one_subject, worker_args):
                results[status].append(output_name)
                done = sum(len(v) for v in results.values())
                elapsed = time.time() - t_start
                rate = done / elapsed if elapsed > 0 else 0
                eta = (total - done) / rate if rate > 0 else 0
                print(f"  Progress: {done}/{total} ({rate:.1f}/min) ETA: {eta:.0f}s "
                      f"[{status}: {output_name}]", flush=True)

        elapsed = time.time() - t_start

        # Summary
        print("\n" + "=" * 60)
        print(f"BATCH PROCESSING SUMMARY — {split}")
        print(f"  Time:           {elapsed:.0f}s ({elapsed/60:.1f}min)")
        print(f"  Workers:        {n_workers}")
        print(f"  Success:        {len(results['success'])}")
        print(f"  Skipped:        {len(results['skip'])}")
        print(f"  Scan fail:      {len(results['scan_fail'])}")
        print(f"  Convert fail:   {len(results['convert_fail'])}")
        print(f"  JCF fail:       {len(results['jcf_fail'])}")
        print(f"  Errors:         {len(results['error'])}")

        # Save results
        with open(os.path.join(out, 'batch_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {out}/batch_results.json")


if __name__ == '__main__':
    main()

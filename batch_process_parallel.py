"""
Parallel batch processing wrapper for batch_process.py
=======================================================
Two-phase pipeline:
  Phase 1 (serial): Scan all b3d files to discover individual segments
  Phase 2 (parallel): Dispatch each segment as an independent work item

This keeps all N workers busy at all times, instead of one worker grinding
through 10 segments while others sit idle.

Thread throttling: OMP/MKL/OPENBLAS set to 1 thread per process to avoid
oversubscription when running N workers on N cores.

Usage:
    conda run -n jcf python batch_process_parallel.py              # auto-detect cores
    conda run -n jcf python batch_process_parallel.py --workers 4  # 4 workers
    conda run -n jcf python batch_process_parallel.py --split both # training + testing
"""

# ── Thread throttling (MUST come before any other imports) ────────────────────
# 1 thread per worker: SO matrices too small for multi-threaded BLAS to help.
# More workers × 1 thread gives better throughput than fewer workers × N threads.
import os
_THREADS_PER_WORKER = "1"
os.environ["OMP_NUM_THREADS"] = _THREADS_PER_WORKER
os.environ["MKL_NUM_THREADS"] = _THREADS_PER_WORKER
os.environ["OPENBLAS_NUM_THREADS"] = _THREADS_PER_WORKER
os.environ["NUMEXPR_NUM_THREADS"] = _THREADS_PER_WORKER
os.environ["VECLIB_MAXIMUM_THREADS"] = _THREADS_PER_WORKER

import sys
import json
import glob
import argparse
import traceback
import multiprocessing as mp
import time

import batch_process
from batch_process import (
    DATASET_PREFIX, GRF_CAP_WALKING,
    scan_b3d_all_runs, _process_activity,
)

# ── Per-split output config ───────────────────────────────────────────────────

B3D_ROOTS = {
    "training": "./with_arm/training",
    "testing": "./with_arm/testing",
}


def _configure_paths(split):
    """Set batch_process module globals for the given split."""
    batch_process.B3D_ROOT = B3D_ROOTS[split]
    out = f"./jcf/full_duration/{split}"
    batch_process.OUTPUT_ROOT = out
    batch_process.OUTPUT_ROOT_RUNNING = os.path.join(out, "running")
    batch_process.OUTPUT_ROOT_WALKING = os.path.join(out, "walking")
    batch_process.SCAN_LOG = os.path.join(out, "scan_failures.txt")
    return out


# ── Segment-level worker ─────────────────────────────────────────────────────

def process_one_segment(args):
    """
    Process a single segment (one b3d trial slice → convert + SO + JR).
    Returns (seg_name, status).
    """
    seg_name, b3d_path, out_root, trial, start_frame, num_frames, idx, total = args

    subj_output = os.path.join(out_root, seg_name)
    done = os.path.exists(os.path.join(
        subj_output, 'jcf_output',
        "BatchJCF_JointReaction_ReactionLoads.sto"))
    if done:
        return (seg_name, 'skip')

    try:
        print(f"  [{idx+1}/{total}] {seg_name}: processing...", flush=True)
        ok = _process_activity(seg_name, b3d_path, subj_output, out_root,
                               trial, start_frame, num_frames)
        return (seg_name, 'success' if ok else 'jcf_fail')
    except Exception as e:
        print(f"  [{idx+1}/{total}] {seg_name}: ERROR {e}", flush=True)
        traceback.print_exc()
        return (seg_name, 'error')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Parallel batch processing (segment-level)")
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of parallel workers (default: physical cores - 1)')
    parser.add_argument('--split', type=str, default='training',
                        choices=['training', 'testing', 'both'],
                        help='Which data split to process (default: training)')
    parser.add_argument('--activity', type=str, default='running',
                        choices=['running', 'walking'],
                        help='Activity type to process (default: running)')
    args = parser.parse_args()

    n_workers = args.workers
    if n_workers is None:
        try:
            n_cores = len(os.sched_getaffinity(0))
        except AttributeError:
            n_cores = mp.cpu_count()
        threads = int(os.environ.get("OMP_NUM_THREADS", "6"))
        n_workers = max(1, n_cores // threads)

    splits = ['training', 'testing'] if args.split == 'both' else [args.split]
    activity = args.activity
    is_running = activity == 'running'

    for split in splits:
        out = _configure_paths(split)
        out_activity = batch_process.OUTPUT_ROOT_RUNNING if is_running else batch_process.OUTPUT_ROOT_WALKING
        os.makedirs(out, exist_ok=True)
        os.makedirs(out_activity, exist_ok=True)

        # ── Phase 1: Scan all b3d files to build segment list ────────────
        print(f"\n{'=' * 60}")
        print(f"Split: {split}  |  Activity: {activity}")
        print(f"Input:  {batch_process.B3D_ROOT}")
        print(f"Output: {out_activity}")
        print(f"Workers: {n_workers}")
        print(f"\nPhase 1: Scanning b3d files for segments...")
        print("=" * 60)

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

        # Scan each b3d to discover segments
        segment_args = []
        scan_fails = []
        t_scan = time.time()

        for i, (dataset, subject, b3d_path) in enumerate(b3d_files):
            prefix = DATASET_PREFIX.get(dataset, dataset.split('_')[0].lower())
            output_name = f"{prefix}_{subject}"

            print(f"  [{i+1}/{len(b3d_files)}] Scanning {dataset}/{subject}...",
                  end='', flush=True)
            try:
                segments = scan_b3d_all_runs(b3d_path)
            except Exception as e:
                scan_fails.append(output_name)
                print(f" ERROR: {e}", flush=True)
                continue

            if not segments:
                scan_fails.append(output_name)
                print(" no valid segments", flush=True)
                continue

            # Build naming: t{trial}, _r{run} only if multiple runs per trial
            trial_run_counts = {}
            for seg in segments:
                t = seg['trial']
                trial_run_counts[t] = trial_run_counts.get(t, 0) + 1

            trial_run_seen = {}
            n_matched = 0
            for seg in segments:
                t = seg['trial']
                run_idx = trial_run_seen.get(t, 0)
                trial_run_seen[t] = run_idx + 1

                seg_is_running = seg['peak_foot_grf_bw'] > GRF_CAP_WALKING
                if seg_is_running != is_running:
                    continue

                if trial_run_counts[t] == 1:
                    seg_name = f"{output_name}_t{t:02d}"
                else:
                    seg_name = f"{output_name}_t{t:02d}_r{run_idx:02d}"

                segment_args.append((
                    seg_name, b3d_path, out_activity,
                    seg['trial'], seg['start_frame'], seg['num_frames'],
                ))
                n_matched += 1

            print(f" {n_matched} {activity} segments ({len(segments)} total)", flush=True)

        scan_time = time.time() - t_scan
        total_segs = len(segment_args)

        # Check how many are already done
        n_already_done = sum(
            1 for s in segment_args
            if os.path.exists(os.path.join(s[2], s[0], 'jcf_output',
                              'BatchJCF_JointReaction_ReactionLoads.sto'))
        )

        print(f"\nScan complete in {scan_time:.0f}s")
        print(f"  Total {activity} segments: {total_segs}")
        print(f"  Already done:           {n_already_done}")
        print(f"  To process:             {total_segs - n_already_done}")
        print(f"  Scan failures:          {len(scan_fails)}")

        # ── Phase 2: Process segments in parallel ────────────────────────
        print(f"\nPhase 2: Processing segments with {n_workers} workers...")
        print("=" * 60)

        # Add index/total for progress display
        worker_args = [
            (*s, idx, total_segs)
            for idx, s in enumerate(segment_args)
        ]

        results = {"success": [], "skip": [], "scan_fail": scan_fails,
                   "convert_fail": [], "jcf_fail": [], "error": []}

        t_start = time.time()

        with mp.Pool(processes=n_workers) as pool:
            for seg_name, status in pool.imap_unordered(process_one_segment, worker_args):
                results[status].append(seg_name)
                done = sum(len(v) for v in results.values())
                elapsed = time.time() - t_start
                rate = (done / elapsed * 60) if elapsed > 0 else 0
                remaining = total_segs - done
                eta = (remaining / (rate / 60)) if rate > 0 else 0
                print(f"  Progress: {done}/{total_segs} ({rate:.1f}/min) "
                      f"ETA: {eta/60:.1f}min [{status}: {seg_name}]", flush=True)

        elapsed = time.time() - t_start

        # Summary
        print("\n" + "=" * 60)
        print(f"BATCH PROCESSING SUMMARY — {split}")
        print(f"  Scan time:      {scan_time:.0f}s")
        print(f"  Process time:   {elapsed:.0f}s ({elapsed/60:.1f}min)")
        print(f"  Workers:        {n_workers}")
        print(f"  Success:        {len(results['success'])}")
        print(f"  Skipped:        {len(results['skip'])}")
        print(f"  Scan fail:      {len(results['scan_fail'])}")
        print(f"  Convert fail:   {len(results['convert_fail'])}")
        print(f"  JCF fail:       {len(results['jcf_fail'])}")
        print(f"  Errors:         {len(results['error'])}")

        with open(os.path.join(out, 'batch_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {out}/batch_results.json")


if __name__ == '__main__':
    main()

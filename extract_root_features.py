"""
Extract root-frame features from .b3d files for subjects that already
have JCF labels from the SO+JR pipeline.

For each processed subject, reads the source .b3d and extracts:
  - jointCentersInRootFrame  (n_joints * 3)
  - rootLinearVelInRootFrame  (3)
  - rootAngularVelInRootFrame (3)
  - rootLinearAccInRootFrame  (3)
  - rootAngularAccInRootFrame (3)
  - groundContactForceInRootFrame (6)  -- GRF in pelvis frame
  - comAccInRootFrame (3)

Saves as root_features.npy alongside existing data.

Usage:
    conda run -n jcf python extract_root_features.py
    conda run -n jcf python extract_root_features.py --testing
"""

import os
import re
import json
import glob
import argparse
import numpy as np
import nimblephysics as nimble

B3D_ROOTS = ["./with_arm/training", "./with_arm/testing"]
DATASET_MAP = {
    'carter': 'Carter2023_Formatted_With_Arm',
    'hammer': 'Hammer2013_Formatted_With_Arm',
    'moore': 'Moore2015_Formatted_With_Arm',
    'tiziana': 'Tiziana2019_Formatted_With_Arm',
    'falisse': 'Falisse2017_Formatted_With_Arm',
    'fregly': 'Fregly2012_Formatted_With_Arm',
    'han': 'Han2023_Formatted_With_Arm',
}


def parse_subject_name(name):
    """Parse subject directory name into dataset, subject, trial, run.

    Examples:
        carter_P002_split1_t00     -> ('carter', 'P002_split1', 0, None)
        carter_P002_split1_t00_r01 -> ('carter', 'P002_split1', 0, 1)
        hammer_subject10_t00       -> ('hammer', 'subject10', 0, None)
        hammer_subject10           -> ('hammer', 'subject10', 0, None)  # no trial suffix
        carter_P010_split0         -> ('carter', 'P010_split0', 0, None)
    """
    # Try with trial suffix first
    match = re.match(r'^(\w+?)_(.+)_t(\d+)(?:_r(\d+))?$', name)
    if match:
        dataset = match.group(1)
        subject = match.group(2)
        trial = int(match.group(3))
        run = int(match.group(4)) if match.group(4) else None
        return dataset, subject, trial, run
    # No trial suffix — assume trial 0
    match = re.match(r'^(\w+?)_(.+)$', name)
    if match:
        dataset = match.group(1)
        subject = match.group(2)
        if dataset in DATASET_MAP:
            return dataset, subject, 0, None
    return None


def find_b3d_path(dataset, subject):
    """Find the .b3d file for a given dataset and subject."""
    dataset_dir = DATASET_MAP.get(dataset)
    if not dataset_dir:
        return None
    for b3d_root in B3D_ROOTS:
        b3d_path = os.path.join(b3d_root, dataset_dir, subject, f"{subject}.b3d")
        if os.path.exists(b3d_path):
            return b3d_path
        b3d_files = glob.glob(os.path.join(b3d_root, dataset_dir, subject, "*.b3d"))
        if b3d_files:
            return b3d_files[0]
    return None


def find_matching_segment(subject_obj, ik_pos, n_passes, trial_hint=None):
    """Find the trial and start frame in the .b3d that matches the IK data
    by correlating joint positions. Uses dynamics pass (same as b3d_to_opensim).

    If trial_hint is given, searches that trial first (fast path for _tXX subjects).
    Returns (trial, start_frame, num_frames) or (None, None, None).
    """
    n_target = len(ik_pos)
    ik_frame0 = ik_pos[0]
    dyn_pass = n_passes - 1

    def search_trial(trial):
        trial_len = subject_obj.getTrialLength(trial)
        if trial_len < n_target:
            return None, float('inf')
        trial_passes = subject_obj.getTrialNumProcessingPasses(trial)
        if trial_passes < 2:
            return None, float('inf')
        frames = subject_obj.readFrames(
            trial=trial, startFrame=0, numFramesToRead=trial_len,
            includeSensorData=False, includeProcessingPasses=True
        )
        pass_idx = min(dyn_pass, len(frames[0].processingPasses) - 1)
        b3d_pos = np.array([np.array(f.processingPasses[pass_idx].pos) for f in frames])
        best_start = None
        best_err = float('inf')
        for offset in range(trial_len - n_target + 1):
            err = np.sum((b3d_pos[offset] - ik_frame0) ** 2)
            if err < best_err:
                best_err = err
                best_start = offset
        return best_start, best_err

    # Fast path: search hint trial first
    if trial_hint is not None and trial_hint < subject_obj.getNumTrials():
        start, err = search_trial(trial_hint)
        if err < 0.01:
            return trial_hint, start, n_target

    # Slow path: search all trials
    best_trial = None
    best_start = None
    best_err = float('inf')
    for trial in range(subject_obj.getNumTrials()):
        start, err = search_trial(trial)
        if err < best_err:
            best_err = err
            best_trial = trial
            best_start = start

    if best_err > 1.0:
        return None, None, None
    return best_trial, best_start, n_target


def extract_root_features(subject_obj, trial, start_frame, num_frames):
    """Extract root-frame features from a .b3d segment."""
    frames = subject_obj.readFrames(
        trial=trial, startFrame=start_frame,
        numFramesToRead=num_frames,
        includeSensorData=False, includeProcessingPasses=True
    )

    all_jc = []
    all_root_lv = []
    all_root_av = []
    all_root_la = []
    all_root_aa = []
    all_grf_root = []
    all_com_acc = []

    for frame in frames:
        kin = frame.processingPasses[0]
        all_jc.append(np.array(kin.jointCentersInRootFrame))
        all_root_lv.append(np.array(kin.rootLinearVelInRootFrame))
        all_root_av.append(np.array(kin.rootAngularVelInRootFrame))
        all_root_la.append(np.array(kin.rootLinearAccInRootFrame))
        all_root_aa.append(np.array(kin.rootAngularAccInRootFrame))
        all_grf_root.append(np.array(kin.groundContactForceInRootFrame))
        all_com_acc.append(np.array(kin.comAccInRootFrame))

    features = np.hstack([
        np.array(all_jc),        # [T, n_joints*3]
        np.array(all_root_lv),   # [T, 3]
        np.array(all_root_av),   # [T, 3]
        np.array(all_root_la),   # [T, 3]
        np.array(all_root_aa),   # [T, 3]
        np.array(all_grf_root),  # [T, 6]
        np.array(all_com_acc),   # [T, 3]
    ])
    return features


def process_directory(data_root, force=False):
    """Process all subjects in a directory."""
    import pandas as pd
    subjects = sorted(os.listdir(data_root))
    n_ok = 0
    n_skip = 0
    n_fail = 0
    b3d_cache = {}

    for name in subjects:
        subj_dir = os.path.join(data_root, name)
        if not os.path.isdir(subj_dir):
            continue

        out_path = os.path.join(subj_dir, 'root_features.npy')
        if os.path.exists(out_path) and not force:
            n_skip += 1
            continue

        parsed = parse_subject_name(name)
        if not parsed:
            n_fail += 1
            continue

        dataset, subject, trial_hint, run = parsed
        b3d_path = find_b3d_path(dataset, subject)
        if not b3d_path:
            print(f"  SKIP {name}: .b3d not found")
            n_fail += 1
            continue

        ik_path = os.path.join(subj_dir, 'ik_results.mot')
        if not os.path.exists(ik_path):
            n_fail += 1
            continue
        ik = pd.read_csv(ik_path, sep='\t', skiprows=6)
        ik_pos = ik.drop(columns=['time']).values

        try:
            if b3d_path not in b3d_cache:
                b3d_cache[b3d_path] = nimble.biomechanics.SubjectOnDisk(b3d_path)
            subj_obj = b3d_cache[b3d_path]

            trial, start_frame, num_frames = find_matching_segment(
                subj_obj, ik_pos, subj_obj.getNumProcessingPasses(),
                trial_hint=trial_hint
            )
            if trial is None:
                print(f"  SKIP {name}: no matching segment found in .b3d")
                n_fail += 1
                continue

            features = extract_root_features(subj_obj, trial, start_frame, num_frames)
            np.save(out_path, features)
            n_ok += 1
            if n_ok % 50 == 0:
                print(f"  {n_ok} extracted...", flush=True)
        except Exception as e:
            print(f"  FAIL {name}: {e}")
            n_fail += 1
            continue

    print(f"\nDone: {n_ok} extracted, {n_skip} already exist, {n_fail} failed")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--testing', action='store_true',
                        help='Process testing subjects instead of training')
    parser.add_argument('--force', action='store_true',
                        help='Re-extract even if root_features.npy exists')
    args = parser.parse_args()

    if args.testing:
        root = "./jcf/testing/running"
    else:
        root = "./jcf/full_duration/training/running"

    print(f"Extracting root-frame features from: {root}")
    process_directory(root, force=args.force)

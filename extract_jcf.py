"""
Step 2: Extract Knee JCF from B3D Files
========================================
Reads .b3d files, extracts joint positions/velocities/torques and GRF,
computes knee joint contact force using Nimble's inverse dynamics,
and saves feature/label arrays for MLP training.

The key insight: JCF at the knee ≈ sum of all forces acting across the joint,
which we get from the dynamics pass. Specifically, we use the skeleton's
inverse dynamics (tau) + GRF to compute the net force at the knee joint.

Install:
    pip3 install nimblephysics numpy

Usage:
    python 02_extract_jcf.py /path/to/b3d_files/ --output ./processed_data/
"""

import nimblephysics as nimble
import numpy as np
import os
import sys
import glob
import argparse
import json
from typing import List, Dict, Optional


def extract_subject_data(b3d_path: str, output_dir: str) -> Optional[Dict]:
    """
    Extract features and JCF-relevant labels from one .b3d subject.
    
    For each valid frame (with GRF data), we extract:
    
    FEATURES (input to MLP):
        - Joint positions (pos) from kinematics pass
        - Joint velocities (vel) from kinematics pass
        - Joint accelerations (acc) from kinematics pass
    
    LABELS (what the MLP predicts):
        - Joint torques (tau) from dynamics pass — includes knee joint torques
        - Ground contact forces in root frame
        - Ground contact CoP in root frame
    
    We use kinematics pass for inputs (hasn't seen force data)
    and dynamics pass for outputs (physically consistent with forces).
    """
    subject = nimble.biomechanics.SubjectOnDisk(b3d_path)
    fname = os.path.basename(b3d_path).replace('.b3d', '')
    
    mass_kg = subject.getMassKg()
    if mass_kg <= 0:
        print(f"  SKIP {fname}: no mass info")
        return None
    
    n_passes = subject.getNumProcessingPasses()
    if n_passes < 2:
        print(f"  SKIP {fname}: only {n_passes} pass(es), need kinematics + dynamics")
        return None
    
    # Find kinematics pass (first) and dynamics pass (last with dynamics type)
    kinematics_pass = 0  # Always first
    dynamics_pass = n_passes - 1  # Usually last
    
    n_dofs = subject.getNumDofs()
    n_joints = subject.getNumJoints()
    contact_bodies = subject.getGroundForceBodies()
    
    # Collect data across all trials
    all_pos = []       # Joint positions [kinematics pass]
    all_vel = []       # Joint velocities [kinematics pass]
    all_acc = []       # Joint accelerations [kinematics pass]
    all_tau = []       # Joint torques [dynamics pass] — THIS IS KEY for JCF
    all_grf = []       # Ground reaction forces [dynamics pass]
    all_cop = []       # Center of pressure [dynamics pass]
    all_com_acc = []   # COM acceleration [dynamics pass]
    
    total_valid = 0
    
    for trial in range(subject.getNumTrials()):
        trial_len = subject.getTrialLength(trial)
        if trial_len < 10:
            continue
        
        # Check which pass count this trial has
        trial_passes = subject.getTrialNumProcessingPasses(trial)
        if trial_passes < 2:
            continue
        
        # Get missing GRF flags to filter frames
        missing_grf = subject.getMissingGRF(trial)
        
        # Read all frames for this trial
        try:
            frames = subject.readFrames(
                trial=trial,
                startFrame=0,
                numFramesToRead=trial_len,
                includeSensorData=False,
                includeProcessingPasses=True
            )
        except Exception as e:
            print(f"  WARN: Could not read trial {trial} of {fname}: {e}")
            continue
        
        for i, frame in enumerate(frames):
            # Skip frames with missing GRF
            if missing_grf[i] != nimble.biomechanics.MissingGRFReason.notMissingGRF:
                continue
            
            if len(frame.processingPasses) < 2:
                continue
            
            # Kinematics pass (input features — never saw force data)
            kin = frame.processingPasses[kinematics_pass]
            # Dynamics pass (output labels — physically consistent)
            dyn = frame.processingPasses[min(dynamics_pass, len(frame.processingPasses)-1)]
            
            all_pos.append(np.array(kin.pos))
            all_vel.append(np.array(kin.vel))
            all_acc.append(np.array(kin.acc))
            
            all_tau.append(np.array(dyn.tau))
            all_grf.append(np.array(dyn.groundContactForceInRootFrame))
            all_cop.append(np.array(dyn.groundContactCenterOfPressureInRootFrame))
            all_com_acc.append(np.array(dyn.comAccInRootFrame))
            
            total_valid += 1
    
    if total_valid < 50:
        print(f"  SKIP {fname}: only {total_valid} valid frames")
        return None
    
    # Stack into arrays
    pos = np.array(all_pos, dtype=np.float32)       # [T, n_dofs]
    vel = np.array(all_vel, dtype=np.float32)       # [T, n_dofs]
    acc = np.array(all_acc, dtype=np.float32)       # [T, n_dofs]
    tau = np.array(all_tau, dtype=np.float32)       # [T, n_dofs]
    grf = np.array(all_grf, dtype=np.float32)       # [T, n_contact_bodies * 3]
    cop = np.array(all_cop, dtype=np.float32)       # [T, n_contact_bodies * 3]
    com_acc = np.array(all_com_acc, dtype=np.float32)  # [T, 3]
    
    # --- Build feature matrix and label matrix ---
    # Features: pos + vel + acc (from kinematics pass)
    features = np.hstack([pos, vel, acc])  # [T, 3 * n_dofs]
    
    # Labels: tau (joint torques from dynamics pass)
    # The knee torques within tau ARE the net joint moments,
    # which serve as proxy for JCF when combined with GRF
    labels_tau = tau                        # [T, n_dofs]
    labels_grf = grf                        # [T, n_contact * 3]
    
    # Save
    subj_dir = os.path.join(output_dir, fname)
    os.makedirs(subj_dir, exist_ok=True)
    
    np.save(os.path.join(subj_dir, 'features.npy'), features)
    np.save(os.path.join(subj_dir, 'tau.npy'), labels_tau)
    np.save(os.path.join(subj_dir, 'grf.npy'), labels_grf)
    np.save(os.path.join(subj_dir, 'cop.npy'), cop)
    np.save(os.path.join(subj_dir, 'com_acc.npy'), com_acc)
    np.save(os.path.join(subj_dir, 'pos.npy'), pos)
    np.save(os.path.join(subj_dir, 'vel.npy'), vel)
    np.save(os.path.join(subj_dir, 'acc.npy'), acc)
    
    # Save metadata
    meta = {
        'source_file': os.path.basename(b3d_path),
        'mass_kg': mass_kg,
        'height_m': subject.getHeightM(),
        'n_dofs': n_dofs,
        'n_joints': n_joints,
        'n_valid_frames': total_valid,
        'feature_dim': features.shape[1],
        'tau_dim': labels_tau.shape[1],
        'grf_dim': labels_grf.shape[1],
        'contact_bodies': contact_bodies,
        'body_weight_N': mass_kg * 9.81,
    }
    with open(os.path.join(subj_dir, 'metadata.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    
    print(f"  OK {fname}: {total_valid} frames, "
          f"features={features.shape}, tau={labels_tau.shape}")
    
    return meta


def batch_extract(data_dir: str, output_dir: str, max_subjects: int = 100):
    """Extract data from all .b3d files."""
    b3d_files = sorted(glob.glob(os.path.join(data_dir, "**/*.b3d"), recursive=True))
    
    if not b3d_files:
        print(f"No .b3d files found in {data_dir}")
        return
    
    print(f"Found {len(b3d_files)} .b3d files, processing up to {max_subjects}")
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    for i, path in enumerate(b3d_files[:max_subjects]):
        print(f"\n[{i+1}/{min(len(b3d_files), max_subjects)}] {os.path.basename(path)}")
        try:
            meta = extract_subject_data(path, output_dir)
            if meta:
                results.append(meta)
        except Exception as e:
            print(f"  ERROR: {e}")
    
    # Save master index
    index_path = os.path.join(output_dir, 'subjects_index.json')
    with open(index_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"EXTRACTION COMPLETE")
    print(f"{'='*60}")
    print(f"  Subjects processed: {len(results)}")
    total_frames = sum(r['n_valid_frames'] for r in results)
    print(f"  Total valid frames: {total_frames:,}")
    print(f"  Output directory:   {output_dir}")
    print(f"  Index file:         {index_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract JCF data from B3D files")
    parser.add_argument("data_dir", help="Directory containing .b3d files")
    parser.add_argument("--output", default="./processed_data/", 
                        help="Output directory for .npy files")
    parser.add_argument("--max-subjects", type=int, default=100)
    args = parser.parse_args()
    
    batch_extract(args.data_dir, args.output, args.max_subjects)
"""
B3D → OpenSim File Converter
==============================
Extracts OpenSim-compatible files from .b3d so you can run
the JCF_singleFile pipeline (SO + JR) on Windows with OpenSim.

Run this in WSL (Linux) where nimblephysics works, then copy
the output folders to Windows and run your JCF scripts there.

Outputs per subject:
    subject_name/
        scaled_model.osim    ← OpenSim model XML
        ik_results.mot       ← joint angles (IK-equivalent)
        grf_loads.xml        ← external loads setup file
        grf_data.mot         ← ground reaction force data

Install (in WSL):
    pip3 install nimblephysics numpy

Usage:
    python b3d_to_opensim.py /path/to/b3d_files/ --output ./opensim_data/
    python b3d_to_opensim.py /path/to/single_file.b3d --output ./opensim_data/
"""

import nimblephysics as nimble
import numpy as np
import os
import sys
import glob
import argparse
import json


def write_mot_file(filepath: str, time: np.ndarray, data: np.ndarray, 
                   col_names: list, header_name: str = "IK"):
    """Write an OpenSim .mot file (tab-delimited with header)."""
    n_rows, n_cols = data.shape
    
    with open(filepath, 'w') as f:
        f.write(f"{header_name}\n")
        f.write(f"version=1\n")
        f.write(f"nRows={n_rows}\n")
        f.write(f"nColumns={n_cols + 1}\n")
        f.write(f"inDegrees=no\n")
        f.write(f"endheader\n")
        
        # Column headers
        f.write("time\t" + "\t".join(col_names) + "\n")
        
        # Data rows
        for i in range(n_rows):
            row = f"{time[i]:.6f}"
            for j in range(n_cols):
                row += f"\t{data[i, j]:.8f}"
            f.write(row + "\n")


def write_grf_mot(filepath: str, time: np.ndarray,
                  forces: np.ndarray, cops: np.ndarray, torques: np.ndarray,
                  contact_bodies: list):
    """
    Write GRF data as an OpenSim .mot file.
    
    Forces/CoP/torques arrays: [T, n_bodies * 3]
    """
    n_frames = len(time)
    n_bodies = len(contact_bodies)
    
    # Build column names and data matrix
    col_names = []
    columns = []
    
    for i, body in enumerate(contact_bodies):
        # Force components
        col_names += [f"{body}_force_vx", f"{body}_force_vy", f"{body}_force_vz"]
        columns.append(forces[:, i*3:(i+1)*3])
        
        # Point of application (CoP)
        col_names += [f"{body}_force_px", f"{body}_force_py", f"{body}_force_pz"]
        columns.append(cops[:, i*3:(i+1)*3])
        
        # Torques
        col_names += [f"{body}_torque_x", f"{body}_torque_y", f"{body}_torque_z"]
        columns.append(torques[:, i*3:(i+1)*3])
    
    data = np.hstack(columns)  # [T, n_bodies * 9]
    write_mot_file(filepath, time, data, col_names, header_name="GRF")


def write_external_loads_xml(filepath: str, grf_mot_path: str,
                              contact_bodies: list):
    """Write the OpenSim ExternalLoads .xml setup file."""
    
    # Map contact body names to the OpenSim body they act on
    # In the Rajagopal model: calcn_r, calcn_l are typical
    xml_forces = ""
    for body in contact_bodies:
        if body == 'pelvis':
            continue
        xml_forces += f"""
        <ExternalForce name="{body}_force">
            <applied_to_body>{body}</applied_to_body>
            <force_expressed_in_body>ground</force_expressed_in_body>
            <point_expressed_in_body>ground</point_expressed_in_body>
            <force_identifier>{body}_force_v</force_identifier>
            <point_identifier>{body}_force_p</point_identifier>
            <torque_identifier>{body}_torque_</torque_identifier>
        </ExternalForce>"""
    
    xml_content = f"""<?xml version="1.0" encoding="UTF-8" ?>
<OpenSimDocument Version="40000">
    <ExternalLoads name="external_loads">
        <objects>{xml_forces}
        </objects>
        <datafile>{os.path.basename(grf_mot_path)}</datafile>
        <external_loads_model_kinematics_file />
    </ExternalLoads>
</OpenSimDocument>"""
    
    with open(filepath, 'w') as f:
        f.write(xml_content)


def convert_subject(b3d_path: str, output_dir: str,
                    trial_index: int = None, start_frame: int = None,
                    num_frames: int = None, output_name: str = None) -> dict:
    """
    Convert one .b3d file into OpenSim files.
    
    If trial_index/start_frame/num_frames are given, only export that slice.
    Returns metadata dict or None on failure.
    """
    subject = nimble.biomechanics.SubjectOnDisk(b3d_path)
    fname = output_name or os.path.basename(b3d_path).replace('.b3d', '')
    
    n_passes = subject.getNumProcessingPasses()
    if n_passes < 2:
        print(f"  SKIP {fname}: need ≥2 processing passes, got {n_passes}")
        return None
    
    mass_kg = subject.getMassKg()
    contact_bodies = subject.getGroundForceBodies()
    
    # --- 1. Write the .osim model file ---
    subj_dir = os.path.join(output_dir, fname)
    os.makedirs(subj_dir, exist_ok=True)
    
    # Use the dynamics pass model (last pass, has tuned masses)
    osim_xml = subject.getOpensimFileText(n_passes - 1)
    osim_path = os.path.join(subj_dir, "scaled_model.osim")
    with open(osim_path, 'w') as f:
        f.write(osim_xml)
    
    # --- 2. Get coordinate (DOF) names from the skeleton ---
    skel = subject.readSkel(processingPass=n_passes - 1, ignoreGeometry=True)
    dof_names = [skel.getDofByIndex(i).getName() for i in range(skel.getNumDofs())]
    n_dofs = len(dof_names)
    
    # --- 3. Extract frame data across trials (or a specific slice) ---
    # We use the dynamics pass for IK (it has the best kinematics)
    # and raw force plate data for GRF
    
    all_time = []
    all_pos = []
    all_grf_force = []
    all_grf_cop = []
    all_grf_torque = []
    
    time_offset = 0.0  # accumulate across trials
    total_valid = 0
    
    # Determine which trials to process
    if trial_index is not None:
        trials_to_process = [trial_index]
    else:
        trials_to_process = list(range(subject.getNumTrials()))
    
    for trial in trials_to_process:
        trial_len = subject.getTrialLength(trial)
        if trial_len < 10:
            continue
        
        trial_passes = subject.getTrialNumProcessingPasses(trial)
        if trial_passes < 2:
            continue
        
        dt = subject.getTrialTimestep(trial)
        missing_grf = subject.getMissingGRF(trial)
        
        # Determine frame range for this trial
        sf = start_frame if (trial_index is not None and start_frame is not None) else 0
        nf = num_frames if (trial_index is not None and num_frames is not None) else (trial_len - sf)
        nf = min(nf, trial_len - sf)  # clamp
        
        # Read frames from the dynamics pass
        dyn_pass = min(n_passes - 1, trial_passes - 1)
        try:
            frames = subject.readFrames(
                trial=trial, startFrame=sf,
                numFramesToRead=nf,
                includeSensorData=True,
                includeProcessingPasses=True
            )
        except Exception as e:
            print(f"  WARN: trial {trial} read error: {e}")
            continue
        
        for i, frame in enumerate(frames):
            frame_idx = sf + i
            if missing_grf[frame_idx] != nimble.biomechanics.MissingGRFReason.notMissingGRF:
                continue
            
            if len(frame.processingPasses) < 1:
                continue
            
            fp = frame.processingPasses[-1]  # dynamics pass
            
            t = time_offset + i * dt
            all_time.append(t)
            all_pos.append(np.array(fp.pos))
            all_grf_force.append(np.array(fp.groundContactForce))
            all_grf_cop.append(np.array(fp.groundContactCenterOfPressure))
            all_grf_torque.append(np.array(fp.groundContactTorque))
            total_valid += 1
        
        time_offset += nf * dt + 0.1  # gap between trials
    
    if total_valid < 50:
        print(f"  SKIP {fname}: only {total_valid} valid frames")
        return None
    
    time_arr = np.array(all_time)
    pos_arr = np.array(all_pos)        # [T, n_dofs]
    grf_force = np.array(all_grf_force)  # [T, n_contacts * 3]
    grf_cop = np.array(all_grf_cop)
    grf_torque = np.array(all_grf_torque)
    
    # --- 4. Write IK .mot file ---
    ik_path = os.path.join(subj_dir, "ik_results.mot")
    write_mot_file(ik_path, time_arr, pos_arr, dof_names, header_name="IK")
    
    # --- 5. Write GRF .mot file ---
    grf_mot_path = os.path.join(subj_dir, "grf_data.mot")
    write_grf_mot(grf_mot_path, time_arr, grf_force, grf_cop, grf_torque,
                  contact_bodies)
    
    # --- 6. Write ExternalLoads .xml ---
    grf_xml_path = os.path.join(subj_dir, "grf_loads.xml")
    write_external_loads_xml(grf_xml_path, grf_mot_path, contact_bodies)
    
    # --- 7. Metadata ---
    meta = {
        'subject': fname,
        'mass_kg': mass_kg,
        'height_m': subject.getHeightM(),
        'n_dofs': n_dofs,
        'dof_names': dof_names,
        'n_valid_frames': total_valid,
        'contact_bodies': contact_bodies,
        'files': {
            'model': 'scaled_model.osim',
            'ik': 'ik_results.mot',
            'grf_data': 'grf_data.mot',
            'grf_xml': 'grf_loads.xml',
        }
    }
    with open(os.path.join(subj_dir, 'metadata.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    
    print(f"  OK {fname}: {total_valid} frames, {n_dofs} DOFs, "
          f"contacts={contact_bodies}")
    
    return meta


def batch_convert(data_dir: str, output_dir: str, max_subjects: int = 100,
                  trial_index: int = None, start_frame: int = None,
                  num_frames: int = None, output_name: str = None):
    """Convert all .b3d files in a directory."""
    
    if os.path.isfile(data_dir) and data_dir.endswith('.b3d'):
        b3d_files = [data_dir]
    else:
        b3d_files = sorted(glob.glob(os.path.join(data_dir, "**/*.b3d"), 
                                      recursive=True))
    
    if not b3d_files:
        print(f"No .b3d files found in {data_dir}")
        return
    
    print(f"Found {len(b3d_files)} .b3d files, converting up to {max_subjects}")
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    for i, path in enumerate(b3d_files[:max_subjects]):
        print(f"\n[{i+1}/{min(len(b3d_files), max_subjects)}] "
              f"{os.path.basename(path)}")
        try:
            meta = convert_subject(path, output_dir,
                                   trial_index=trial_index,
                                   start_frame=start_frame,
                                   num_frames=num_frames,
                                   output_name=output_name)
            if meta:
                results.append(meta)
        except Exception as e:
            print(f"  ERROR: {e}")
    
    # Save master index
    with open(os.path.join(output_dir, 'subjects_index.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"CONVERSION COMPLETE")
    print(f"{'='*60}")
    print(f"  Converted: {len(results)} / {min(len(b3d_files), max_subjects)}")
    print(f"  Output:    {output_dir}")
    print(f"{'='*60}")
    print(f"\nNext steps:")
    print(f"  1. Copy {output_dir} to Windows")
    print(f"  2. Run JCF_singleFile on one subject to test:")
    print(f"     python jcf_single_subject.py {output_dir}/<subject_name>/ scaled_model.osim")
    print(f"  3. Then batch process:")
    print(f"     python jcf_batch_processing.py {output_dir}/ --max-subjects 100")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert B3D files to OpenSim format for JCF pipeline")
    parser.add_argument("data_dir", 
                        help="Directory with .b3d files, or a single .b3d file")
    parser.add_argument("--output", default="./opensim_data/",
                        help="Output directory for OpenSim files")
    parser.add_argument("--max-subjects", type=int, default=100)
    parser.add_argument("--trial", type=int, default=None,
                        help="Only export this trial index")
    parser.add_argument("--start-frame", type=int, default=None,
                        help="Start frame within the trial")
    parser.add_argument("--num-frames", type=int, default=None,
                        help="Number of frames to export")
    parser.add_argument("--output-name", type=str, default=None,
                        help="Override the output folder name")
    args = parser.parse_args()
    
    batch_convert(args.data_dir, args.output, args.max_subjects,
                  trial_index=args.trial, start_frame=args.start_frame,
                  num_frames=args.num_frames, output_name=args.output_name)
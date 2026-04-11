"""
Batch processing pipeline: b3d -> OpenSim files -> SO + JR (JCF)
================================================================
Processes all .b3d files in with_arm/training/With_Arm/,
converts to OpenSim format, then runs Static Optimization + JointReaction
to produce knee JCF labels for NN training.

Usage:
    conda run -n jcf python batch_process.py

Steps per subject:
    1. b3d_to_opensim: .b3d -> .osim, .mot, grf
    2. Auto-detect a 10s walking window from GRF
    3. Add reserve actuators, run SO + JR
    4. Save JCF .sto results
"""

import opensim as osm
import os
import sys
import json
import subprocess
import pandas as pd
import numpy as np
import glob
import traceback

# ─── Config ──────────────────────────────────────────────────────────────────

B3D_ROOT = "./with_arm/training/With_Arm"
OUTPUT_ROOT = "./jcf/training"
WINDOW_DURATION = 10.0  # seconds of walking to analyze
BUFFER = 0.5            # padding for filter edge effects

# ─── XML Template (from JCF_singleFile.py) ───────────────────────────────────

ANALYZE_XML_TEMPLATE = """<?xml version="1.0" encoding="UTF-8" ?>
<OpenSimDocument Version="40600">
  <AnalyzeTool name="{name}">
    <model_file>{model_file}</model_file>
    <replace_force_set>false</replace_force_set>
    <force_set_files />
    <results_directory>{results_dir}</results_directory>
    <output_precision>8</output_precision>
    <initial_time>{t0}</initial_time>
    <final_time>{t1}</final_time>
    <solve_for_equilibrium_for_auxiliary_states>true</solve_for_equilibrium_for_auxiliary_states>
    <maximum_number_of_integrator_steps>20000</maximum_number_of_integrator_steps>
    <maximum_integrator_step_size>1</maximum_integrator_step_size>
    <minimum_integrator_step_size>1e-08</minimum_integrator_step_size>
    <integrator_error_tolerance>1e-05</integrator_error_tolerance>
    <AnalysisSet name="Analyses">
      <objects>
        <StaticOptimization name="StaticOptimization">
          <on>true</on>
          <start_time>-Inf</start_time>
          <end_time>Inf</end_time>
          <step_interval>1</step_interval>
          <in_degrees>true</in_degrees>
          <use_model_force_set>true</use_model_force_set>
          <activation_exponent>2</activation_exponent>
          <use_muscle_physiology>true</use_muscle_physiology>
          <optimizer_convergence_criterion>0.0001</optimizer_convergence_criterion>
          <optimizer_max_iterations>200</optimizer_max_iterations>
        </StaticOptimization>
        <JointReaction name="JointReaction">
          <on>true</on>
          <start_time>-Inf</start_time>
          <end_time>Inf</end_time>
          <step_interval>1</step_interval>
          <in_degrees>true</in_degrees>
          <forces_file />
          <joint_names> walker_knee_r</joint_names>
          <apply_on_bodies> child</apply_on_bodies>
          <express_in_frame> child</express_in_frame>
        </JointReaction>
      </objects>
      <groups />
    </AnalysisSet>
    <ControllerSet name="Controllers">
      <objects />
      <groups />
    </ControllerSet>
    <external_loads_file>{grf_file}</external_loads_file>
    <states_file />
    <coordinates_file>{coordinates_file}</coordinates_file>
    <speeds_file />
    <lowpass_cutoff_frequency_for_coordinates>6</lowpass_cutoff_frequency_for_coordinates>
  </AnalyzeTool>
</OpenSimDocument>
"""


# ─── Step 1: Scan b3d for best walking window ────────────────────────────────

def scan_b3d_for_walking(b3d_path, window_duration=10.0):
    """
    Scan a b3d file directly with nimblephysics to find the best contiguous
    walking window. Returns (trial, start_frame, num_frames, mass_kg) or None.
    
    This avoids converting the entire b3d file — we only need to read
    lightweight header + GRF from candidate trials.
    """
    import nimblephysics as nimble
    
    subject = nimble.biomechanics.SubjectOnDisk(b3d_path)
    mass_kg = subject.getMassKg()
    if mass_kg <= 0:
        return None
    BW = mass_kg * 9.81
    
    best = None  # (score, trial, start_frame, num_frames)
    
    for trial in range(subject.getNumTrials()):
        trial_len = subject.getTrialLength(trial)
        trial_passes = subject.getTrialNumProcessingPasses(trial)
        if trial_passes < 2 or trial_len < 50:
            continue
        
        dt = subject.getTrialTimestep(trial)
        window_frames = int(window_duration / dt)
        if window_frames > trial_len:
            continue
        
        # Get missing GRF flags (cheap — header only)
        missing = subject.getMissingGRF(trial)
        valid = [m == nimble.biomechanics.MissingGRFReason.notMissingGRF for m in missing]
        
        # Find longest contiguous valid run
        runs = []
        run_start = None
        for i, v in enumerate(valid):
            if v and run_start is None:
                run_start = i
            elif not v and run_start is not None:
                runs.append((run_start, i - run_start))
                run_start = None
        if run_start is not None:
            runs.append((run_start, len(valid) - run_start))
        
        # Only consider runs long enough for our window
        for run_start, run_len in runs:
            if run_len < window_frames:
                continue
            
            # Read a small sample from middle of the run to check GRF magnitude
            mid = run_start + run_len // 2
            sample_start = max(run_start, mid - 25)
            sample_n = min(50, run_len)
            try:
                frames = subject.readFrames(trial, sample_start, sample_n,
                                            includeSensorData=False,
                                            includeProcessingPasses=True)
            except Exception:
                continue
            
            # Check vertical GRF magnitude and variation
            vy_vals = []
            for f in frames:
                if len(f.processingPasses) < 1:
                    continue
                fp = f.processingPasses[-1]
                grf = np.array(fp.groundContactForce)
                # Sum vertical component from all contact bodies (index 1 of each 3-vec)
                total_vy = sum(grf[j*3 + 1] for j in range(len(grf) // 3))
                vy_vals.append(total_vy)
            
            if len(vy_vals) < 10:
                continue
            vy_arr = np.array(vy_vals)
            mean_vy = np.mean(vy_arr)
            std_vy = np.std(vy_arr)
            
            # Walking criteria: mean GRF near BW, decent variation
            if mean_vy < 0.3 * BW:
                continue
            score = std_vy / BW + 0.01 * (run_len / trial_len)  # prefer longer runs
            
            # Pick start of the contiguous valid run (use as much as we can)
            sf = run_start
            nf = min(run_len, window_frames)
            
            if best is None or score > best[0]:
                best = (score, trial, sf, nf)
    
    if best is None:
        return None
    
    _, trial, sf, nf = best
    return (trial, sf, nf, mass_kg)


# ─── Step 2: Convert only the needed slice ────────────────────────────────────

def convert_b3d_slice(b3d_path, output_dir, trial, start_frame, num_frames,
                      output_name):
    """Run b3d_to_opensim.py on a specific trial/frame slice."""
    result = subprocess.run(
        [sys.executable, "./b3d_to_opensim.py", b3d_path,
         "--output", output_dir,
         "--trial", str(trial),
         "--start-frame", str(start_frame),
         "--num-frames", str(num_frames),
         "--output-name", output_name],
        capture_output=True, text=True, timeout=120
    )
    if result.returncode != 0:
        print(f"    b3d_to_opensim FAILED: {result.stderr[-300:]}")
        return False
    return True


# ─── Step 3: Run SO + JR ─────────────────────────────────────────────────────

def run_jcf(subject_dir, t0, t1):
    """Run Static Optimization + JointReaction on one converted subject."""
    output_dir = os.path.join(subject_dir, 'jcf_output')
    os.makedirs(output_dir, exist_ok=True)

    ik_file = os.path.join(subject_dir, 'ik_results.mot')
    grf_file = os.path.join(subject_dir, 'grf_loads.xml')
    model_path = os.path.join(subject_dir, 'scaled_model.osim')

    if not all(os.path.exists(f) for f in [ik_file, grf_file, model_path]):
        print(f"    Missing files in {subject_dir}")
        return False

    # Add reserve actuators
    model = osm.Model(model_path)
    model_coord_names = [model.getCoordinateSet().get(i).getName()
                         for i in range(model.getCoordinateSet().getSize())]

    for i in range(model.getCoordinateSet().getSize()):
        coord = model.getCoordinateSet().get(i)
        if coord.isConstrained(model.initSystem()):
            continue
        reserve = osm.CoordinateActuator(coord.getName())
        reserve.setName(f"reserve_{coord.getName()}")
        if coord.getMotionType() == osm.Coordinate.Rotational:
            reserve.setOptimalForce(1.0)
        else:
            reserve.setOptimalForce(10.0)
        reserve.setMinControl(-float('inf'))
        reserve.setMaxControl(float('inf'))
        model.addForce(reserve)

    augmented_model = os.path.join(output_dir, '_model_with_reserves.osim')
    model.printToXML(augmented_model)
    del model

    # Pad missing IK columns & trim
    ik_df = pd.read_csv(ik_file, sep='\t', skiprows=6)
    for name in model_coord_names:
        if name not in ik_df.columns:
            ik_df[name] = 0.0

    mask = (ik_df['time'] >= t0 - BUFFER) & (ik_df['time'] <= t1 + BUFFER)
    ik_trimmed = ik_df[mask].reset_index(drop=True)
    if len(ik_trimmed) < 10:
        print(f"    Too few frames in window [{t0:.1f}, {t1:.1f}]")
        return False

    padded_ik = os.path.join(output_dir, '_padded_ik.mot')
    with open(padded_ik, 'w') as f:
        f.write(f"Coordinates\nversion=1\nnRows={len(ik_trimmed)}\n"
                f"nColumns={len(ik_trimmed.columns)}\ninDegrees=no\nendheader\n")
        ik_trimmed.to_csv(f, sep='\t', index=False)

    # Write XML setup
    setup_xml = os.path.join(output_dir, '_analyze_setup.xml')
    with open(setup_xml, 'w') as f:
        f.write(ANALYZE_XML_TEMPLATE.format(
            name="BatchJCF",
            model_file=os.path.abspath(augmented_model),
            results_dir=os.path.abspath(output_dir),
            t0=t0, t1=t1,
            grf_file=os.path.abspath(os.path.join(subject_dir, 'grf_loads.xml')),
            coordinates_file=os.path.abspath(padded_ik),
        ))

    # Run AnalyzeTool
    analyze = osm.AnalyzeTool(setup_xml)
    analyze.run()
    analyze.printResults(analyze.getName(), os.path.abspath(output_dir))

    # Check output exists
    jcf_file = os.path.join(output_dir, "BatchJCF_JointReaction_ReactionLoads.sto")
    if os.path.exists(jcf_file):
        return True
    # Try to find it
    for fname in os.listdir(output_dir):
        if 'ReactionLoads' in fname:
            return True
    return False


# ─── Main batch loop ─────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    # Filter to specific datasets (set to None to process all)
    DATASETS_TO_PROCESS = ["Moore2015_Formatted_With_Arm"]

    # Find all b3d files
    b3d_files = []
    for dataset in sorted(os.listdir(B3D_ROOT)):
        dataset_path = os.path.join(B3D_ROOT, dataset)
        if not os.path.isdir(dataset_path):
            continue
        if DATASETS_TO_PROCESS is not None and dataset not in DATASETS_TO_PROCESS:
            continue
        for subject in sorted(os.listdir(dataset_path)):
            subj_path = os.path.join(dataset_path, subject)
            b3d_list = glob.glob(os.path.join(subj_path, "*.b3d"))
            if b3d_list:
                b3d_files.append((dataset, subject, b3d_list[0]))

    print(f"Found {len(b3d_files)} subjects to process")
    print(f"Output: {OUTPUT_ROOT}")
    print("=" * 60)

    results = {"success": [], "scan_fail": [], "convert_fail": [], "jcf_fail": [], "error": []}

    for idx, (dataset_name, subject_name, b3d_path) in enumerate(b3d_files):
        tag = f"[{idx+1}/{len(b3d_files)}] {dataset_name}/{subject_name}"

        # Check if already processed
        subj_output = os.path.join(OUTPUT_ROOT, subject_name)
        jcf_output = os.path.join(subj_output, 'jcf_output')
        jcf_sto = os.path.join(jcf_output, "BatchJCF_JointReaction_ReactionLoads.sto")
        if os.path.exists(jcf_sto):
            print(f"{tag}: SKIP (already done)")
            results["success"].append(subject_name)
            continue

        print(f"\n{tag}: Processing...")

        try:
            # Step 1: Scan b3d for best walking window (fast, no conversion)
            print(f"  Scanning b3d for walking window...")
            scan = scan_b3d_for_walking(b3d_path, WINDOW_DURATION)
            if scan is None:
                print(f"  No valid walking window found in b3d")
                results["scan_fail"].append(subject_name)
                continue
            trial, start_frame, num_frames, mass_kg = scan
            dt_est = WINDOW_DURATION / num_frames if num_frames > 0 else 0.01
            print(f"  Found: trial {trial}, frames {start_frame}-{start_frame+num_frames} "
                  f"({num_frames * dt_est:.1f}s), mass={mass_kg:.1f}kg")

            # Step 2: Convert only the needed slice to OpenSim
            if not os.path.exists(os.path.join(subj_output, 'scaled_model.osim')):
                print(f"  Converting b3d slice...")
                ok = convert_b3d_slice(b3d_path, OUTPUT_ROOT, trial,
                                       start_frame, num_frames, subject_name)
                if not ok:
                    results["convert_fail"].append(subject_name)
                    continue
            else:
                print(f"  OpenSim files exist, skipping conversion")

            # Step 3: Run SO + JR on the converted slice
            # The converted .mot starts at time 0, so t0/t1 are relative
            meta_path = os.path.join(subj_output, 'metadata.json')
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
                n_valid = meta.get('n_valid_frames', 0)
            else:
                n_valid = num_frames
            
            # Read the IK file to get actual time range
            ik_path = os.path.join(subj_output, 'ik_results.mot')
            ik_df = pd.read_csv(ik_path, sep='\t', skiprows=6)
            t0 = ik_df['time'].iloc[0] + BUFFER
            t1 = ik_df['time'].iloc[-1] - BUFFER
            if t1 - t0 < 1.0:
                print(f"  Converted slice too short ({t1-t0:.1f}s)")
                results["convert_fail"].append(subject_name)
                continue
            print(f"  Running SO + JointReaction on [{t0:.2f}, {t1:.2f}]s...")
            sys.stdout.flush()

            ok = run_jcf(subj_output, t0, t1)
            if ok:
                print(f"  SUCCESS")
                results["success"].append(subject_name)
            else:
                print(f"  JCF output missing")
                results["jcf_fail"].append(subject_name)

        except Exception as e:
            print(f"  ERROR: {e}")
            traceback.print_exc()
            results["error"].append(subject_name)

        sys.stdout.flush()

    # Summary
    print("\n" + "=" * 60)
    print("BATCH PROCESSING SUMMARY")
    print(f"  Success:        {len(results['success'])}")
    print(f"  Scan fail:      {len(results['scan_fail'])}")
    print(f"  Convert fail:   {len(results['convert_fail'])}")
    print(f"  JCF fail:       {len(results['jcf_fail'])}")
    print(f"  Errors:         {len(results['error'])}")

    # Save results
    with open(os.path.join(OUTPUT_ROOT, 'batch_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_ROOT}/batch_results.json")


if __name__ == '__main__':
    main()

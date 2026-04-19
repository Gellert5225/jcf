"""
Batch re-run SO + JR with 2.0× muscle strength scaling
========================================================
Re-processes all training subjects with max_isometric_force × 2.0
to reduce activation saturation and produce more accurate peak JCF labels.

Saves output to jcf_output_2x/ within each subject directory, preserving
the original jcf_output/ for comparison.

Usage:
    conda activate jcf
    python batch_rescale.py
    python batch_rescale.py --scale 1.5   # alternative scale factor
    python batch_rescale.py --root ./jcf/testing/running   # different root
"""

import opensim as osm
import os
import json
import argparse
import time
import numpy as np
import pandas as pd

# ─── Config ──────────────────────────────────────────────────────────────────

BUFFER = 0.3

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


def run_so_jr_scaled(subject_dir, scale_factor, output_subdir="jcf_output_2x"):
    """
    Run SO + JointReaction with scaled muscle max_isometric_force.
    Saves to subject_dir/<output_subdir>/.
    Returns True on success.
    """
    output_dir = os.path.join(subject_dir, output_subdir)
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(subject_dir, 'scaled_model.osim')
    ik_file = os.path.join(subject_dir, 'ik_results.mot')
    grf_file = os.path.join(subject_dir, 'grf_loads.xml')

    if not all(os.path.exists(f) for f in [model_path, ik_file, grf_file]):
        return False, "missing files"

    # Get time window from existing JCF output
    existing_jcf = os.path.join(subject_dir, 'jcf_output',
                                'BatchJCF_JointReaction_ReactionLoads.sto')
    if not os.path.exists(existing_jcf):
        return False, "no existing JCF output"

    with open(existing_jcf) as f:
        hdr = 0
        for line in f:
            hdr += 1
            if line.strip() == 'endheader':
                break
    jcf_df = pd.read_csv(existing_jcf, sep='\t', skiprows=hdr)
    t0 = jcf_df['time'].iloc[0]
    t1 = jcf_df['time'].iloc[-1]

    # Load and modify model
    model = osm.Model(model_path)

    # Scale muscles
    muscles = model.getMuscles()
    for i in range(muscles.getSize()):
        m = muscles.get(i)
        m.setMaxIsometricForce(m.getMaxIsometricForce() * scale_factor)

    # Add reserve actuators
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
        return False, f"too few frames ({len(ik_trimmed)})"

    padded_ik = os.path.join(output_dir, '_padded_ik.mot')
    with open(padded_ik, 'w') as f:
        f.write(f"Coordinates\nversion=1\nnRows={len(ik_trimmed)}\n"
                f"nColumns={len(ik_trimmed.columns)}\ninDegrees=no\nendheader\n")
        ik_trimmed.to_csv(f, sep='\t', index=False)

    # Write XML setup — use "BatchJCF" as name for consistent output filenames
    setup_xml = os.path.join(output_dir, '_analyze_setup.xml')
    with open(setup_xml, 'w') as f:
        f.write(ANALYZE_XML_TEMPLATE.format(
            name="BatchJCF",
            model_file=os.path.abspath(augmented_model),
            results_dir=os.path.abspath(output_dir),
            t0=t0, t1=t1,
            grf_file=os.path.abspath(grf_file),
            coordinates_file=os.path.abspath(padded_ik),
        ))

    # Run AnalyzeTool
    try:
        analyze = osm.AnalyzeTool(setup_xml)
        analyze.run()
        analyze.printResults(analyze.getName(), os.path.abspath(output_dir))
    except Exception as e:
        return False, str(e)

    # Check output
    jcf_out = os.path.join(output_dir, "BatchJCF_JointReaction_ReactionLoads.sto")
    if os.path.exists(jcf_out):
        return True, "ok"

    for fname in os.listdir(output_dir):
        if 'ReactionLoads' in fname and fname.endswith('.sto'):
            return True, "ok"

    return False, "no JCF output"


def main():
    parser = argparse.ArgumentParser(description="Re-run SO+JR with scaled muscle strength")
    parser.add_argument('--scale', type=float, default=2.0, help='Muscle strength scale factor')
    parser.add_argument('--root', type=str, default='./jcf/training/running',
                        help='Root directory containing subject folders')
    parser.add_argument('--output-subdir', type=str, default=None,
                        help='Output subdirectory name (default: jcf_output_{scale}x)')
    args = parser.parse_args()

    scale = args.scale
    root = args.root
    output_subdir = args.output_subdir or f"jcf_output_{scale:.0f}x"

    # Find all subjects with existing JCF output
    subject_dirs = []
    for name in sorted(os.listdir(root)):
        subj_dir = os.path.join(root, name)
        jcf_sto = os.path.join(subj_dir, 'jcf_output',
                                'BatchJCF_JointReaction_ReactionLoads.sto')
        if os.path.isdir(subj_dir) and os.path.exists(jcf_sto):
            subject_dirs.append(subj_dir)

    print(f"Found {len(subject_dirs)} subjects in {root}")
    print(f"Scale factor: {scale}x")
    print(f"Output: <subject>/{output_subdir}/")

    # Check what's already done
    done = 0
    todo = []
    for subj_dir in subject_dirs:
        out_dir = os.path.join(subj_dir, output_subdir)
        jcf_out = os.path.join(out_dir, "BatchJCF_JointReaction_ReactionLoads.sto")
        if os.path.exists(jcf_out):
            done += 1
        else:
            todo.append(subj_dir)

    print(f"Already done: {done}, remaining: {len(todo)}")
    if not todo:
        print("All subjects already processed!")
        return

    # Process
    t_start = time.time()
    successes = 0
    failures = []

    for i, subj_dir in enumerate(todo):
        name = os.path.basename(subj_dir)
        t0 = time.time()
        ok, msg = run_so_jr_scaled(subj_dir, scale, output_subdir)
        elapsed = time.time() - t0

        total_done = done + i + 1
        total = len(subject_dirs)
        if ok:
            successes += 1
            print(f"[{total_done}/{total}] {name}: OK ({elapsed:.1f}s)")
        else:
            failures.append((name, msg))
            print(f"[{total_done}/{total}] {name}: FAILED - {msg} ({elapsed:.1f}s)")

        # ETA
        if (i + 1) % 10 == 0:
            avg_time = (time.time() - t_start) / (i + 1)
            remaining = len(todo) - (i + 1)
            eta_min = avg_time * remaining / 60
            print(f"  ... avg {avg_time:.1f}s/subject, ETA: {eta_min:.0f} min")

    total_time = time.time() - t_start
    print(f"\nDone! {successes}/{len(todo)} succeeded in {total_time/60:.1f} min")
    if failures:
        print(f"Failures ({len(failures)}):")
        for name, msg in failures:
            print(f"  {name}: {msg}")


if __name__ == "__main__":
    main()

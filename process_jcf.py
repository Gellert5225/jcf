"""
Process JCF: Convert full b3d → SO + JointReaction → Plot resultant.
=====================================================================
Converts the ENTIRE b3d file (all trials, all valid GRF frames) to OpenSim,
then runs Static Optimization + Joint Reaction across the full duration.
No time-window selection — you see every frame.

Usage:
    # Single b3d file
    conda run -n jcf python process_jcf.py path/to/subject.b3d

    # All b3d files under a folder (recursive)
    conda run -n jcf python process_jcf.py --all with_arm/testing

    # Custom output directory
    conda run -n jcf python process_jcf.py path/to/subject.b3d --output ./my_output
"""

import opensim as osm
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from b3d_to_opensim import convert_subject

OUTPUT_DIR = "./jcf/full_duration"
BUFFER = 0.3  # seconds of padding for filter edge effects

# ─── XML Template ─────────────────────────────────────────────────────────────

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
    <lowpass_cutoff_frequency_for_coordinates>6</lowpass_cutoff_frequency_for_coordinates>
  </AnalyzeTool>
</OpenSimDocument>
"""


# ─── SO + JR ──────────────────────────────────────────────────────────────────

def run_jcf(subject_dir, t0, t1):
    """Run Static Optimization + JointReaction on the full time range."""
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

    # Pad missing IK columns (no trimming — use full range)
    ik_df = pd.read_csv(ik_file, sep='\t', skiprows=6)
    for name in model_coord_names:
        if name not in ik_df.columns:
            ik_df[name] = 0.0

    padded_ik = os.path.join(output_dir, '_padded_ik.mot')
    with open(padded_ik, 'w') as f:
        f.write(f"Coordinates\nversion=1\nnRows={len(ik_df)}\n"
                f"nColumns={len(ik_df.columns)}\ninDegrees=no\nendheader\n")
        ik_df.to_csv(f, sep='\t', index=False)

    # Write XML setup
    setup_xml = os.path.join(output_dir, '_analyze_setup.xml')
    with open(setup_xml, 'w') as f:
        f.write(ANALYZE_XML_TEMPLATE.format(
            name="FullDuration",
            model_file=os.path.abspath(augmented_model),
            results_dir=os.path.abspath(output_dir),
            t0=t0, t1=t1,
            grf_file=os.path.abspath(os.path.join(subject_dir, 'grf_loads.xml')),
            coordinates_file=os.path.abspath(padded_ik),
        ))

    # Run
    print(f"    Running SO + JR on [{t0:.2f}, {t1:.2f}]s...")
    sys.stdout.flush()
    analyze = osm.AnalyzeTool(setup_xml)
    analyze.run()
    analyze.printResults(analyze.getName(), os.path.abspath(output_dir))

    jcf_file = os.path.join(output_dir, "FullDuration_JointReaction_ReactionLoads.sto")
    if os.path.exists(jcf_file):
        return True
    for fname in os.listdir(output_dir):
        if 'ReactionLoads' in fname:
            return True
    return False


# ─── b3d finding ──────────────────────────────────────────────────────────────

def find_b3d_files(path):
    """Find all .b3d files under a path (recursive)."""
    if path.endswith('.b3d'):
        return [path] if os.path.exists(path) else []
    b3d_files = []
    for root, dirs, files in os.walk(path):
        for f in sorted(files):
            if f.endswith('.b3d'):
                b3d_files.append(os.path.join(root, f))
    return b3d_files


# ─── Process one subject ─────────────────────────────────────────────────────

def process_b3d(b3d_path, output_dir):
    """Convert full b3d → run SO+JR on entire duration → return results."""
    name = os.path.basename(b3d_path).replace('.b3d', '')
    subj_dir = os.path.join(output_dir, name)

    # Step 1: Convert full b3d (all trials, no windowing)
    ik_path = os.path.join(subj_dir, 'ik_results.mot')
    if not os.path.exists(ik_path):
        print(f"  Converting full b3d: {b3d_path}")
        meta = convert_subject(b3d_path, output_dir)
        if meta is None:
            print(f"  SKIP {name}: conversion failed")
            return None
    else:
        print(f"  {name}: OpenSim files already exist, skipping conversion")

    # Step 2: Read IK to get full time range
    ik_df = pd.read_csv(ik_path, sep='\t', skiprows=6)
    t0 = ik_df['time'].iloc[0] + BUFFER
    t1 = ik_df['time'].iloc[-1] - BUFFER
    duration = t1 - t0
    n_frames = len(ik_df)

    with open(os.path.join(subj_dir, 'metadata.json')) as f:
        meta = json.load(f)
    mass = meta['mass_kg']
    BW = mass * 9.81

    print(f"  {name}: {n_frames} frames, {duration:.1f}s, mass={mass:.1f}kg")

    # Step 3: Check if already done
    jcf_sto = None
    jcf_output = os.path.join(subj_dir, 'jcf_output')
    for candidate in ['FullDuration_JointReaction_ReactionLoads.sto']:
        p = os.path.join(jcf_output, candidate)
        if os.path.exists(p):
            jcf_sto = p
            break
    if jcf_sto is None and os.path.isdir(jcf_output):
        for fname in os.listdir(jcf_output):
            if 'ReactionLoads' in fname:
                jcf_sto = os.path.join(jcf_output, fname)
                break

    if jcf_sto is None:
        # Run SO + JR
        ok = run_jcf(subj_dir, t0, t1)
        if not ok:
            print(f"  FAIL {name}: SO+JR failed")
            return None
        jcf_sto = os.path.join(jcf_output,
                               "FullDuration_JointReaction_ReactionLoads.sto")
        if not os.path.exists(jcf_sto):
            for fname in os.listdir(jcf_output):
                if 'ReactionLoads' in fname:
                    jcf_sto = os.path.join(jcf_output, fname)
                    break
    else:
        print(f"  {name}: JCF output exists, skipping SO+JR")

    # Step 4: Load JCF results
    with open(jcf_sto) as f:
        skip = 0
        for i, line in enumerate(f):
            if 'endheader' in line:
                skip = i + 1
                break
    jcf_df = pd.read_csv(jcf_sto, sep='\t', skiprows=skip)
    time_arr = jcf_df['time'].values

    fx_col = [c for c in jcf_df.columns if 'fx' in c.lower() and 'tibia' in c.lower()]
    fy_col = [c for c in jcf_df.columns if 'fy' in c.lower() and 'tibia' in c.lower()]
    fz_col = [c for c in jcf_df.columns if 'fz' in c.lower() and 'tibia' in c.lower()]

    fx = jcf_df[fx_col[0]].values / BW
    fy = jcf_df[fy_col[0]].values / BW
    fz = jcf_df[fz_col[0]].values / BW
    resultant = np.sqrt(fx**2 + fy**2 + fz**2)
    peak = np.max(resultant)

    print(f"    JCF: {len(time_arr)} frames, peak resultant = {peak:.2f} BW")

    return {
        'name': name,
        'time': time_arr,
        'fx': fx, 'fy': fy, 'fz': fz,
        'resultant': resultant,
        'mass': mass,
        'peak': peak,
        'duration': duration,
        'n_frames': len(time_arr),
    }


# ─── Plot ─────────────────────────────────────────────────────────────────────

def plot_results(results, output_path):
    """Plot resultant JCF for all subjects, full duration."""
    n = len(results)
    fig, axes = plt.subplots(n, 1, figsize=(14, 4 * n), squeeze=False)

    for i, res in enumerate(results):
        ax = axes[i, 0]
        ax.plot(res['time'], res['resultant'], 'b-', linewidth=0.8,
                label='JCF (SO + JR)')
        ax.axhline(y=2.5, color='green', linestyle='--', linewidth=1.5,
                    alpha=0.8, label='2.5 BW')
        ax.axhline(y=3.5, color='red', linestyle='--', linewidth=1.5,
                    alpha=0.8, label='3.5 BW')
        ax.axhspan(2.5, 3.5, alpha=0.08, color='green')

        ax.set_ylabel('Resultant JCF (BW)')
        ax.set_xlabel('Time (s)')
        ax.set_title(f'{res["name"]} — Full Duration SO+JR '
                     f'({res["n_frames"]} frames, {res["duration"]:.1f}s, '
                     f'peak: {res["peak"]:.2f} BW)')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\nPlot saved to {output_path}")
    plt.close()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Convert full b3d → SO+JR → plot knee JCF (entire duration)')
    parser.add_argument('path',
                        help='Path to .b3d file or directory (with --all)')
    parser.add_argument('--all', action='store_true',
                        help='Recursively find and process all b3d files')
    parser.add_argument('--output', default=OUTPUT_DIR,
                        help=f'Output directory (default: {OUTPUT_DIR})')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Find b3d files
    if args.all:
        b3d_files = find_b3d_files(args.path)
    else:
        b3d_files = [args.path]

    print(f"Found {len(b3d_files)} b3d file(s)\n")

    # Process each
    results = []
    for b3d_path in b3d_files:
        print(f"\n{'='*60}")
        print(f"Processing: {b3d_path}")
        res = process_b3d(b3d_path, args.output)
        if res is not None:
            results.append(res)

    if not results:
        print("\nNo subjects processed.")
        return

    # Plot
    plot_path = os.path.join(args.output, 'full_duration_jcf.png')
    plot_results(results, plot_path)

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY: {len(results)} subjects")
    for r in results:
        print(f"  {r['name']:30s}  {r['n_frames']:5d} frames  "
              f"{r['duration']:7.1f}s  peak={r['peak']:.2f} BW")


if __name__ == '__main__':
    main()

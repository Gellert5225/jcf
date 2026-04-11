import opensim as osm
import os
import pandas as pd
import numpy as np
import sys

# Template for AnalyzeTool XML setup
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


def test_single_subject(subject_folder, model_file):
    """
    Runs Static Optimization and Joint Reaction on one subject to verify 
    the pipeline for your ISRR paper's surrogate model.
    """
    # 1. Setup Paths
    output_dir = os.path.join(subject_folder, 'test_output')
    os.makedirs(output_dir, exist_ok=True)

    ik_file = os.path.join(subject_folder, 'ik_results.mot')
    grf_file = os.path.join(subject_folder, 'grf_loads.xml')
    model_path = os.path.join(subject_folder, model_file)

    # 2. Identify model coordinates, add reserve actuators, pad missing IK coords
    model = osm.Model(model_path)
    model_coord_names = [model.getCoordinateSet().get(i).getName()
                         for i in range(model.getCoordinateSet().getSize())]

    # Add reserve actuators (low-strength backup for when muscles saturate)
    for i in range(model.getCoordinateSet().getSize()):
        coord = model.getCoordinateSet().get(i)
        if coord.isConstrained(model.initSystem()):
            continue
        reserve = osm.CoordinateActuator(coord.getName())
        reserve.setName(f"reserve_{coord.getName()}")
        # 1 Nm for rotational, 10 N for translational — just enough to
        # prevent SO failure without dominating muscle forces
        if coord.getMotionType() == osm.Coordinate.Rotational:
            reserve.setOptimalForce(1.0)
        else:
            reserve.setOptimalForce(10.0)
        reserve.setMinControl(-float('inf'))
        reserve.setMaxControl(float('inf'))
        model.addForce(reserve)

    # Save augmented model
    augmented_model = os.path.join(output_dir, '_model_with_reserves.osim')
    model.printToXML(augmented_model)
    print(f"  Added reserve actuators to model")
    del model  # Release before analysis

    ik_df = pd.read_csv(ik_file, sep='\t', skiprows=6)
    for name in model_coord_names:
        if name not in ik_df.columns:
            ik_df[name] = 0.0

    # 3. Trim to analysis window with buffer for filter edge effects
    t0, t1 = 139.0, 149.0
    buffer = 0.5
    mask = (ik_df['time'] >= t0 - buffer) & (ik_df['time'] <= t1 + buffer)
    ik_trimmed = ik_df[mask].reset_index(drop=True)
    print(f"  Trimmed IK: {len(ik_trimmed)} frames "
          f"({ik_trimmed['time'].iloc[0]:.3f} to {ik_trimmed['time'].iloc[-1]:.3f}s)")

    padded_ik = os.path.join(output_dir, '_padded_ik.mot')
    with open(padded_ik, 'w') as f:
        f.write(f"Coordinates\nversion=1\nnRows={len(ik_trimmed)}\n"
                f"nColumns={len(ik_trimmed.columns)}\ninDegrees=no\nendheader\n")
        ik_trimmed.to_csv(f, sep='\t', index=False)

    # 4. Write AnalyzeTool setup XML
    setup_xml = os.path.join(output_dir, '_analyze_setup.xml')
    with open(setup_xml, 'w') as f:
        f.write(ANALYZE_XML_TEMPLATE.format(
            name="SingleSubjTest",
            model_file=os.path.abspath(augmented_model),
            results_dir=os.path.abspath(output_dir),
            t0=t0, t1=t1,
            grf_file=os.path.abspath(grf_file),
            coordinates_file=os.path.abspath(padded_ik),
        ))

    # 5. Load AnalyzeTool from XML and run
    #    This avoids SWIG memory issues by letting C++ own all objects
    print("--- Starting Physics Solve (SO + JCF) ---")
    sys.stdout.flush()
    analyze = osm.AnalyzeTool(setup_xml)
    analyze.run()
    analyze.printResults(analyze.getName(), os.path.abspath(output_dir))
    print(f"--- Process Complete. Files saved to: {output_dir} ---")
    sys.stdout.flush()

    # 6. Quick Validation Check
    jcf_results = os.path.join(output_dir,
                               "SingleSubjTest_JointReaction_ReactionLoads.sto")
    if not os.path.exists(jcf_results):
        for fname in os.listdir(output_dir):
            if 'ReactionLoads' in fname:
                jcf_results = os.path.join(output_dir, fname)
                break

    if os.path.exists(jcf_results):
        # Find endheader to know how many lines to skip
        with open(jcf_results) as f:
            skip = 0
            for i, line in enumerate(f):
                if 'endheader' in line:
                    skip = i + 1
                    break
        df = pd.read_csv(jcf_results, sep='\t', skiprows=skip)
        fy_col = [c for c in df.columns
                  if 'fy' in c.lower() and 'tibia' in c.lower()]
        if fy_col:
            peak_fy = df[fy_col[0]].abs().max()
            print(f"Validation: Peak Knee Contact Fy = {peak_fy:.1f} N")
            print("  Divide by (mass * 9.81) to get BW; "
                  "expect 2.0-4.0 BW during stance.")
    else:
        print(f"  WARNING: JCF output file not found")
        found = [f for f in os.listdir(output_dir) if not f.startswith('_')]
        print(f"  Output files: {found}")

    sys.stdout.flush()
    os._exit(0)


# --- Run the Test ---
test_single_subject(
    subject_folder='./jcf/subject10',
    model_file='scaled_model.osim'
)


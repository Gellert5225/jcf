"""
Muscle Strength Scaling Sensitivity Test
==========================================
Re-runs Static Optimization + JointReaction on a stratified sample of 10
subjects with max_isometric_force scaled by 1.0× (baseline), 1.5×, and 2.0×.

Measures: activation saturation, peak JCF, reserve usage, activation patterns.

Rationale: Rajagopal 2016 model may be under-strength for running; scaling
factors of 1.5-2× are common in running biomech literature (Miller et al.,
Pandy group). This tests whether reduced saturation produces biomechanically
plausible JCF estimates closer to instrumented-knee values (~2.5-3× BW walking,
higher for running).

Usage:
    conda activate jcf
    python scale_muscle_test.py
"""

import opensim as osm
import os
import json
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ─── Config ──────────────────────────────────────────────────────────────────

DATA_ROOT = "./jcf/training/running"
OUTPUT_ROOT = "./jcf/training/scaling_test"
BUFFER = 0.3

SCALE_FACTORS = [1.0, 1.5, 2.0]

# Stratified sample: 10 subjects spanning datasets, mass, saturation
SAMPLE_SUBJECTS = [
    "carter_P035_split0",    # carter, 63.4kg, sat=0.70
    "carter_P052_split5",    # carter, 53.2kg, sat=1.00, peak=3.92BW
    "falisse_subject_5",     # falisse, 60.1kg, sat=0.41
    "falisse_subject_2",     # falisse, 71.7kg, sat=1.00
    "fregly_2GC",            # fregly, 67.0kg, sat=0.86
    "fregly_5GC",            # fregly, 75.0kg, sat=1.00
    "hammer_subject01",      # hammer, 72.8kg, sat=0.92
    "hammer_subject11",      # hammer, 69.3kg, sat=1.00
    "han_s006_split1",       # han, 49.0kg, sat=0.31
    "han_s007_split2",       # han, 64.0kg, sat=1.00
]

# XML template (same as batch_process.py)
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


def run_so_jr(subject_dir, scale_factor, output_dir):
    """
    Run SO + JointReaction with muscle max_isometric_force scaled by scale_factor.
    Returns True on success.
    """
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(subject_dir, 'scaled_model.osim')
    ik_file = os.path.join(subject_dir, 'ik_results.mot')
    grf_file = os.path.join(subject_dir, 'grf_loads.xml')

    if not all(os.path.exists(f) for f in [model_path, ik_file, grf_file]):
        print(f"    Missing files in {subject_dir}")
        return False

    # Load and modify model
    model = osm.Model(model_path)

    # Scale muscle max_isometric_force
    muscles = model.getMuscles()
    n_muscles = muscles.getSize()
    muscle_info = []
    for i in range(n_muscles):
        m = muscles.get(i)
        orig_force = m.getMaxIsometricForce()
        new_force = orig_force * scale_factor
        m.setMaxIsometricForce(new_force)
        muscle_info.append({
            'name': m.getName(),
            'orig_max_force': orig_force,
            'scaled_max_force': new_force,
        })

    # Add reserve actuators (same as batch_process.py)
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

    # Pad missing IK columns & get time range from existing JCF output
    ik_df = pd.read_csv(ik_file, sep='\t', skiprows=6)
    for name in model_coord_names:
        if name not in ik_df.columns:
            ik_df[name] = 0.0

    # Get time window from existing JCF output
    existing_jcf = os.path.join(subject_dir, 'jcf_output',
                                'BatchJCF_JointReaction_ReactionLoads.sto')
    with open(existing_jcf) as f:
        hdr = 0
        for line in f:
            hdr += 1
            if line.strip() == 'endheader':
                break
    jcf_df = pd.read_csv(existing_jcf, sep='\t', skiprows=hdr)
    t0 = jcf_df['time'].iloc[0]
    t1 = jcf_df['time'].iloc[-1]

    # Trim IK with buffer
    mask = (ik_df['time'] >= t0 - BUFFER) & (ik_df['time'] <= t1 + BUFFER)
    ik_trimmed = ik_df[mask].reset_index(drop=True)
    if len(ik_trimmed) < 10:
        print(f"    Too few frames in window [{t0:.3f}, {t1:.3f}]")
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
            name="ScaleTest",
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
        print(f"    SO/JR failed: {e}")
        return False

    # Check output exists
    jcf_file = os.path.join(output_dir, "ScaleTest_JointReaction_ReactionLoads.sto")
    if not os.path.exists(jcf_file):
        # Try alternate name
        for fname in os.listdir(output_dir):
            if 'ReactionLoads' in fname and fname.endswith('.sto'):
                jcf_file = os.path.join(output_dir, fname)
                break
        else:
            print(f"    No JCF output found")
            return False

    return True


def analyze_results(output_dir, mass_kg):
    """
    Analyze SO + JR output: saturation, peak JCF, reserve usage.
    """
    # Find output files
    act_file = None
    jcf_file = None
    force_file = None
    for fname in os.listdir(output_dir):
        if 'activation' in fname and fname.endswith('.sto'):
            act_file = os.path.join(output_dir, fname)
        elif 'ReactionLoads' in fname and fname.endswith('.sto'):
            jcf_file = os.path.join(output_dir, fname)
        elif 'StaticOptimization_force' in fname and fname.endswith('.sto'):
            force_file = os.path.join(output_dir, fname)

    results = {}
    BW = mass_kg * 9.81

    # Activation analysis
    if act_file and os.path.exists(act_file):
        with open(act_file) as f:
            hdr = 0
            for line in f:
                hdr += 1
                if line.strip() == 'endheader':
                    break
        df = pd.read_csv(act_file, sep=r'\s+', skiprows=hdr)
        muscle_cols = [c for c in df.columns
                       if c != 'time' and not c.startswith('reserve_')
                       and not c.startswith('calcn_')]
        if muscle_cols:
            vals = df[muscle_cols].values
            n_frames = len(vals)
            n_muscles = len(muscle_cols)

            # Saturation metrics
            any_sat = np.any(vals >= 0.999, axis=1)  # frames with any muscle at bound
            multi_sat = np.sum(vals >= 0.999, axis=1) > 5  # frames with >5 at bound
            sat_fraction_per_frame = np.sum(vals >= 0.999, axis=1) / n_muscles

            results['n_frames'] = n_frames
            results['n_muscles'] = n_muscles
            results['frac_frames_any_sat'] = float(any_sat.mean())
            results['frac_frames_multi_sat'] = float(multi_sat.mean())
            results['mean_sat_fraction'] = float(sat_fraction_per_frame.mean())
            results['max_sat_fraction'] = float(sat_fraction_per_frame.max())
            results['mean_activation'] = float(vals.mean())
            results['max_activation'] = float(vals.max())

            # Per-muscle saturation frequency
            per_muscle_sat = (vals >= 0.999).mean(axis=0)
            top_saturators = sorted(zip(muscle_cols, per_muscle_sat),
                                    key=lambda x: -x[1])[:5]
            results['top_saturating_muscles'] = [
                {'name': name, 'sat_frac': float(frac)}
                for name, frac in top_saturators
            ]

    # JCF analysis
    if jcf_file and os.path.exists(jcf_file):
        with open(jcf_file) as f:
            hdr = 0
            for line in f:
                hdr += 1
                if line.strip() == 'endheader':
                    break
        df = pd.read_csv(jcf_file, sep='\t', skiprows=hdr)
        fx = df['walker_knee_r_on_tibia_r_in_tibia_r_fx'].values / BW
        fy = df['walker_knee_r_on_tibia_r_in_tibia_r_fy'].values / BW
        fz = df['walker_knee_r_on_tibia_r_in_tibia_r_fz'].values / BW
        resultant = np.sqrt(fx**2 + fy**2 + fz**2)

        results['peak_fy_bw'] = float(np.abs(fy).max())
        results['peak_resultant_bw'] = float(resultant.max())
        results['mean_fy_bw'] = float(np.abs(fy).mean())
        results['fy_timeseries'] = fy.tolist()
        results['resultant_timeseries'] = resultant.tolist()
        results['time'] = df['time'].values.tolist()

    # Reserve analysis
    if force_file and os.path.exists(force_file):
        with open(force_file) as f:
            hdr = 0
            for line in f:
                hdr += 1
                if line.strip() == 'endheader':
                    break
        df = pd.read_csv(force_file, sep=r'\s+', skiprows=hdr)
        reserve_cols = [c for c in df.columns if c.startswith('reserve_')]
        muscle_force_cols = [c for c in df.columns
                             if c != 'time' and c not in reserve_cols
                             and not c.startswith('calcn_')]
        if reserve_cols and muscle_force_cols:
            reserve_total = np.abs(df[reserve_cols].values).sum()
            muscle_total = np.abs(df[muscle_force_cols].values).sum()
            results['reserve_muscle_ratio'] = float(reserve_total / max(muscle_total, 1e-10))
            results['peak_reserve'] = float(np.abs(df[reserve_cols].values).max())

    return results


def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    all_results = {}

    for subj_name in SAMPLE_SUBJECTS:
        subject_dir = os.path.join(DATA_ROOT, subj_name)
        meta_path = os.path.join(subject_dir, 'metadata.json')
        if not os.path.exists(meta_path):
            print(f"SKIP {subj_name}: no metadata")
            continue

        with open(meta_path) as f:
            meta = json.load(f)
        mass = meta['mass_kg']

        print(f"\n{'='*60}")
        print(f"Subject: {subj_name}  (mass={mass:.1f}kg)")
        print(f"{'='*60}")

        subj_results = {}

        for sf in SCALE_FACTORS:
            label = f"x{sf:.1f}"
            out_dir = os.path.join(OUTPUT_ROOT, subj_name, label)

            # Check if already done
            done_files = [f for f in os.listdir(out_dir)
                          if 'ReactionLoads' in f] if os.path.exists(out_dir) else []
            if done_files:
                print(f"  [{label}] Already done, analyzing...")
            else:
                print(f"  [{label}] Running SO+JR with scale={sf}...")
                ok = run_so_jr(subject_dir, sf, out_dir)
                if not ok:
                    print(f"  [{label}] FAILED")
                    continue

            results = analyze_results(out_dir, mass)
            subj_results[label] = results

            sat_str = f"sat={results.get('frac_frames_any_sat', '?'):.2f}" if 'frac_frames_any_sat' in results else "sat=?"
            multi_str = f"multi={results.get('frac_frames_multi_sat', '?'):.2f}" if 'frac_frames_multi_sat' in results else ""
            peak_str = f"peak_Fy={results.get('peak_fy_bw', '?'):.2f}BW" if 'peak_fy_bw' in results else ""
            res_str = f"peak_R={results.get('peak_resultant_bw', '?'):.2f}BW" if 'peak_resultant_bw' in results else ""
            reserve_str = f"res/mus={results.get('reserve_muscle_ratio', '?'):.4f}" if 'reserve_muscle_ratio' in results else ""
            print(f"  [{label}] {sat_str}  {multi_str}  {peak_str}  {res_str}  {reserve_str}")

        all_results[subj_name] = subj_results

    # ─── Summary table ────────────────────────────────────────────────────
    print(f"\n\n{'='*100}")
    print("SUMMARY: Muscle Strength Scaling Sensitivity Test")
    print(f"{'='*100}")
    print(f"{'Subject':40s} {'Scale':6s} {'Sat%':6s} {'Multi%':7s} {'PeakFy':8s} {'PeakR':8s} {'Res/Mus':8s}")
    print("-" * 100)

    for subj_name in SAMPLE_SUBJECTS:
        if subj_name not in all_results:
            continue
        for sf in SCALE_FACTORS:
            label = f"x{sf:.1f}"
            if label not in all_results[subj_name]:
                continue
            r = all_results[subj_name][label]
            print(f"{subj_name:40s} {label:6s} "
                  f"{r.get('frac_frames_any_sat', 0)*100:5.1f}% "
                  f"{r.get('frac_frames_multi_sat', 0)*100:6.1f}% "
                  f"{r.get('peak_fy_bw', 0):7.2f}BW "
                  f"{r.get('peak_resultant_bw', 0):7.2f}BW "
                  f"{r.get('reserve_muscle_ratio', 0):7.4f}")

    # ─── Aggregate statistics ─────────────────────────────────────────────
    print(f"\n{'='*100}")
    print("AGGREGATE (mean across subjects)")
    print(f"{'='*100}")
    for sf in SCALE_FACTORS:
        label = f"x{sf:.1f}"
        sat_vals, multi_vals, peak_fy_vals, peak_r_vals, reserve_vals = [], [], [], [], []
        for subj_name in SAMPLE_SUBJECTS:
            if subj_name in all_results and label in all_results[subj_name]:
                r = all_results[subj_name][label]
                if 'frac_frames_any_sat' in r:
                    sat_vals.append(r['frac_frames_any_sat'])
                if 'frac_frames_multi_sat' in r:
                    multi_vals.append(r['frac_frames_multi_sat'])
                if 'peak_fy_bw' in r:
                    peak_fy_vals.append(r['peak_fy_bw'])
                if 'peak_resultant_bw' in r:
                    peak_r_vals.append(r['peak_resultant_bw'])
                if 'reserve_muscle_ratio' in r:
                    reserve_vals.append(r['reserve_muscle_ratio'])

        if sat_vals:
            print(f"  {label}: sat={np.mean(sat_vals)*100:.1f}%  "
                  f"multi={np.mean(multi_vals)*100:.1f}%  "
                  f"peak_Fy={np.mean(peak_fy_vals):.2f}±{np.std(peak_fy_vals):.2f}BW  "
                  f"peak_R={np.mean(peak_r_vals):.2f}±{np.std(peak_r_vals):.2f}BW  "
                  f"res/mus={np.mean(reserve_vals):.4f}")

    # ─── Plot: Fy timeseries comparison ───────────────────────────────────
    n_subj = len([s for s in SAMPLE_SUBJECTS if s in all_results
                  and any('fy_timeseries' in all_results[s].get(f'x{sf:.1f}', {})
                          for sf in SCALE_FACTORS)])
    if n_subj > 0:
        fig, axes = plt.subplots(n_subj, 1, figsize=(12, 3 * n_subj), squeeze=False)
        row = 0
        for subj_name in SAMPLE_SUBJECTS:
            if subj_name not in all_results:
                continue
            has_data = False
            ax = axes[row, 0]
            colors = {1.0: '#999999', 1.5: '#2196F3', 2.0: '#F44336'}
            for sf in SCALE_FACTORS:
                label = f"x{sf:.1f}"
                if label in all_results[subj_name] and 'fy_timeseries' in all_results[subj_name][label]:
                    r = all_results[subj_name][label]
                    t = np.array(r['time'])
                    fy = np.array(r['fy_timeseries'])
                    ax.plot(t, fy, color=colors[sf], label=f'{label}', linewidth=1.5 if sf > 1 else 1)
                    has_data = True
            if has_data:
                ax.set_ylabel('Fy (BW)')
                ax.set_title(subj_name, fontsize=10)
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
                row += 1

        axes[-1, 0].set_xlabel('Time (s)')
        plt.suptitle('Knee JCF Fy: Effect of Muscle Strength Scaling', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_ROOT, 'scaling_fy_comparison.png'), dpi=150)
        print(f"\nFy comparison plot saved to {OUTPUT_ROOT}/scaling_fy_comparison.png")

    # ─── Plot: Peak JCF bar chart ─────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    subj_labels = []
    peak_data = {f"x{sf:.1f}": [] for sf in SCALE_FACTORS}
    sat_data = {f"x{sf:.1f}": [] for sf in SCALE_FACTORS}

    for subj_name in SAMPLE_SUBJECTS:
        if subj_name not in all_results:
            continue
        has_all = all(f"x{sf:.1f}" in all_results[subj_name] for sf in SCALE_FACTORS)
        if not has_all:
            continue
        subj_labels.append(subj_name.split('_')[0] + '_' + subj_name.split('_')[-1])
        for sf in SCALE_FACTORS:
            label = f"x{sf:.1f}"
            r = all_results[subj_name][label]
            peak_data[label].append(r.get('peak_fy_bw', 0))
            sat_data[label].append(r.get('frac_frames_any_sat', 0) * 100)

    x = np.arange(len(subj_labels))
    width = 0.25
    colors = {'x1.0': '#999999', 'x1.5': '#2196F3', 'x2.0': '#F44336'}

    for i, (label, vals) in enumerate(peak_data.items()):
        ax1.bar(x + (i - 1) * width, vals, width, label=label, color=colors[label])
    ax1.set_ylabel('Peak |Fy| (BW)')
    ax1.set_title('Peak Knee JCF by Scale Factor')
    ax1.set_xticks(x)
    ax1.set_xticklabels(subj_labels, rotation=45, ha='right', fontsize=8)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    for i, (label, vals) in enumerate(sat_data.items()):
        ax2.bar(x + (i - 1) * width, vals, width, label=label, color=colors[label])
    ax2.set_ylabel('Frames with any saturation (%)')
    ax2.set_title('Activation Saturation by Scale Factor')
    ax2.set_xticks(x)
    ax2.set_xticklabels(subj_labels, rotation=45, ha='right', fontsize=8)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_ROOT, 'scaling_summary.png'), dpi=150)
    print(f"Summary plot saved to {OUTPUT_ROOT}/scaling_summary.png")

    # Save results JSON
    # Strip timeseries for JSON (too large)
    json_results = {}
    for subj_name, subj_data in all_results.items():
        json_results[subj_name] = {}
        for label, r in subj_data.items():
            r_clean = {k: v for k, v in r.items()
                       if k not in ('fy_timeseries', 'resultant_timeseries', 'time')}
            json_results[subj_name][label] = r_clean

    with open(os.path.join(OUTPUT_ROOT, 'scaling_results.json'), 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"Results saved to {OUTPUT_ROOT}/scaling_results.json")


if __name__ == "__main__":
    main()

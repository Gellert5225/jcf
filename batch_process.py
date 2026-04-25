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
import multiprocessing as mp

# ─── Config ──────────────────────────────────────────────────────────────────

B3D_ROOT = "./with_arm/testing"
OUTPUT_ROOT = "./jcf/testing"
OUTPUT_ROOT_WALKING = "./jcf/testing/walking"
OUTPUT_ROOT_RUNNING = "./jcf/testing/running"
WINDOW_DURATION = 2.0   # seconds — used when FULL_DURATION=False
FULL_DURATION = True    # Extract all valid GRF segments instead of 2s windows
BUFFER = 0.3            # padding for filter edge effects
GRF_CAP_WALKING = 1.7   # BW — per-foot peak cap for walking
GRF_CAP_RUNNING = 3.5   # BW — per-foot peak cap for running
SCAN_LOG = os.path.join(OUTPUT_ROOT, 'scan_failures.txt')
_scan_log_file = None

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

def _find_heel_strikes(vy, bw):
    """Detect heel strikes: frames where per-foot GRF rises from ~0
    (< 2% BW, i.e. foot in swing) to loading (>= 10% BW).
    Returns list of frame indices at the onset of loading."""
    off_thresh = 0.02 * bw   # must be essentially 0 during swing
    on_thresh = 0.10 * bw    # beginning of loading response
    strikes = []
    was_off = vy[0] < off_thresh
    for i in range(1, len(vy)):
        if was_off and vy[i] >= on_thresh:
            strikes.append(i)
            was_off = False
        elif vy[i] < off_thresh:
            was_off = True
    return strikes


MIN_WINDOW_DURATION = 0.3  # shortest usable window (seconds)


def _scan_log(msg):
    if _scan_log_file is not None:
        _scan_log_file.write(msg + '\n')
        _scan_log_file.flush()


def scan_b3d_for_walking(b3d_path, window_duration=2.0, grf_cap_bw=GRF_CAP_WALKING):
    """
    Scan a b3d file to find the best walking window.
    
    Strategy: detect heel strikes (per-foot GRF rising from ~0 to loading)
    and anchor windows at those onsets. This captures the loading response
    phase where knee JCF is highest.
    
    Adaptive window sizing: tries window_duration first, then falls back
    to shorter durations (down to MIN_WINDOW_DURATION) if no runs are
    long enough (e.g. single-step force plate trials).
    
    Window constraints:
      - Per-foot GRF must not exceed grf_cap_bw * BW
      - Heel strikes start from ~0 (< 2% BW)
    
    Returns (trial, start_frame, num_frames, mass_kg) or None.
    """
    import nimblephysics as nimble
    
    subject = nimble.biomechanics.SubjectOnDisk(b3d_path)
    mass_kg = subject.getMassKg()
    if mass_kg <= 0:
        _scan_log(f"{b3d_path}: invalid mass ({mass_kg})")
        return None
    BW = mass_kg * 9.81
    
    # Identify which contact body indices are feet (calcn_r, calcn_l)
    contact_bodies = subject.getGroundForceBodies()
    foot_indices = [i for i, b in enumerate(contact_bodies) if 'calcn' in b]
    if len(foot_indices) < 2:
        _scan_log(f"{b3d_path}: <2 foot contacts (bodies: {list(contact_bodies)})")
        return None
    
    # Collect all valid GRF runs across all trials (with per-foot data)
    all_runs = []  # (trial, run_start, run_len, dt, vy_r, vy_l)
    
    for trial in range(subject.getNumTrials()):
        trial_len = subject.getTrialLength(trial)
        trial_passes = subject.getTrialNumProcessingPasses(trial)
        if trial_passes < 2 or trial_len < 10:
            continue
        
        dt = subject.getTrialTimestep(trial)
        min_frames = max(10, int(MIN_WINDOW_DURATION / dt))
        
        # Get missing GRF flags (cheap — header only)
        missing = subject.getMissingGRF(trial)
        valid = [m == nimble.biomechanics.MissingGRFReason.notMissingGRF for m in missing]
        
        # Find contiguous valid GRF runs
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
        
        for rs, rl in runs:
            if rl < min_frames:
                continue
            
            # Read GRF for the entire valid run
            try:
                frames = subject.readFrames(trial, rs, rl,
                                            includeSensorData=False,
                                            includeProcessingPasses=True)
            except Exception:
                continue
            
            # Extract per-foot vertical GRF (only calcn_r, calcn_l)
            vy_r = np.zeros(rl)
            vy_l = np.zeros(rl)
            for fi, f in enumerate(frames):
                if len(f.processingPasses) < 1:
                    continue
                fp = f.processingPasses[-1]
                grf = np.array(fp.groundContactForce)
                n_contacts = len(grf) // 3
                for idx in foot_indices:
                    if idx < n_contacts:
                        vy = grf[idx * 3 + 1]
                        body_name = contact_bodies[idx]
                        if '_r' in body_name:
                            vy_r[fi] = vy
                        else:
                            vy_l[fi] = vy
            
            all_runs.append((trial, rs, rl, dt, vy_r, vy_l))
    
    if not all_runs:
        n_trials = subject.getNumTrials()
        _scan_log(f"{b3d_path}: no valid GRF runs ({n_trials} trials examined)")
        return None
    
    # Determine effective window size: use requested duration if runs
    # are long enough, otherwise use the longest available run
    max_run_len = max(rl for _, _, rl, _, _, _ in all_runs)
    dt0 = all_runs[0][3]  # use first trial's dt as reference
    desired_frames = int(window_duration / dt0)
    
    if max_run_len >= desired_frames:
        window_frames = desired_frames
    else:
        # Adaptive: use the longest available run (but at least MIN_WINDOW)
        window_frames = max_run_len
        if window_frames * dt0 < MIN_WINDOW_DURATION:
            _scan_log(f"{b3d_path}: longest run too short ({window_frames * dt0:.2f}s < {MIN_WINDOW_DURATION}s)")
            return None
    
    # Score windows across all collected runs
    best_strike = None
    best_fallback = None
    
    for trial, rs, rl, dt, vy_r, vy_l in all_runs:
        wf = min(window_frames, rl)  # can't exceed this run's length
        if wf * dt < MIN_WINDOW_DURATION:
            continue
        
        # Detect heel strikes for each foot
        strikes_r = _find_heel_strikes(vy_r, BW)
        strikes_l = _find_heel_strikes(vy_l, BW)
        all_strikes = sorted(strikes_r + strikes_l)
        
        def _score_window(w_start, w_end):
            """Score a window by walking quality.
            Returns None for invalid windows (too low GRF, or
            per-foot GRF exceeds 1.7 BW — not normal walking)."""
            r_win = vy_r[w_start:w_end]
            l_win = vy_l[w_start:w_end]
            foot_total = r_win + l_win
            mean_total = np.mean(foot_total)
            if mean_total < 0.3 * BW:
                return None
            # Reject if any single foot GRF exceeds cap
            peak_r = np.max(r_win)
            peak_l = np.max(l_win)
            if peak_r > grf_cap_bw * BW or peak_l > grf_cap_bw * BW:
                return None
            peak_single = max(peak_r, peak_l)
            peak_score = peak_single / BW
            var_score = (np.std(r_win) + np.std(l_win)) / (2 * BW)
            n_strikes = sum(1 for s in all_strikes if w_start <= s < w_end)
            strike_score = min(n_strikes / 4.0, 1.0)
            return 1.0 * peak_score + 1.0 * var_score + 0.5 * strike_score
        
        # --- Heel-strike-anchored windows (primary) ---
        for hs in all_strikes:
            w_start = max(0, hs - max(1, int(0.02 / dt)))
            if w_start + wf > rl:
                continue
            score = _score_window(w_start, w_start + wf)
            if score is None:
                continue
            abs_start = rs + w_start
            if best_strike is None or score > best_strike[0]:
                best_strike = (score, trial, abs_start, wf)
        
        # --- Sliding windows (fallback) ---
        step = max(1, wf // 4)
        for w_start in range(0, rl - wf + 1, step):
            score = _score_window(w_start, w_start + wf)
            if score is None:
                continue
            abs_start = rs + w_start
            if best_fallback is None or score > best_fallback[0]:
                best_fallback = (score, trial, abs_start, wf)
    
    # Prefer heel-strike-anchored window; use fallback only if none found
    best = best_strike if best_strike is not None else best_fallback
    if best is None:
        _scan_log(f"{b3d_path}: no window passed quality filter (GRF too low or too high)")
        return None
    
    _, trial, sf, nf = best
    return (trial, sf, nf, mass_kg)


def scan_b3d_all_runs(b3d_path, min_duration=MIN_WINDOW_DURATION):
    """
    Scan a b3d file and return ALL valid contiguous GRF segments
    where the subject is bearing weight.

    Returns list of dicts:
        {'trial': int, 'start_frame': int, 'num_frames': int,
         'mass_kg': float, 'dt': float, 'duration': float,
         'peak_foot_grf_bw': float}
    or empty list on failure.
    """
    import nimblephysics as nimble

    subject = nimble.biomechanics.SubjectOnDisk(b3d_path)
    mass_kg = subject.getMassKg()
    if mass_kg <= 0:
        _scan_log(f"{b3d_path}: invalid mass ({mass_kg})")
        return []
    BW = mass_kg * 9.81

    contact_bodies = subject.getGroundForceBodies()
    foot_indices = [i for i, b in enumerate(contact_bodies) if 'calcn' in b]
    if len(foot_indices) < 2:
        _scan_log(f"{b3d_path}: <2 foot contacts (bodies: {list(contact_bodies)})")
        return []

    segments = []

    for trial in range(subject.getNumTrials()):
        trial_name = subject.getTrialName(trial)
        if 'static' in trial_name.lower():
            continue  # skip static posture trials (Carter: Static_1, Han: YYYYMMDD_static_1)

        trial_len = subject.getTrialLength(trial)
        trial_passes = subject.getTrialNumProcessingPasses(trial)
        if trial_passes < 2 or trial_len < 10:
            continue

        dt = subject.getTrialTimestep(trial)
        min_frames = max(10, int(min_duration / dt))

        missing = subject.getMissingGRF(trial)
        valid = [m == nimble.biomechanics.MissingGRFReason.notMissingGRF
                 for m in missing]

        # Find contiguous valid GRF runs
        runs = []
        run_start = None
        for i, v in enumerate(valid):
            if v and run_start is None:
                run_start = i
            elif not v and run_start is not None:
                run_len = i - run_start
                if run_len >= min_frames:
                    runs.append((run_start, run_len))
                run_start = None
        if run_start is not None:
            run_len = len(valid) - run_start
            if run_len >= min_frames:
                runs.append((run_start, run_len))

        for rs, rl in runs:
            try:
                frames = subject.readFrames(trial, rs, rl,
                                            includeSensorData=False,
                                            includeProcessingPasses=True)
            except Exception:
                continue

            # Extract per-foot vertical GRF
            vy_r = np.zeros(rl)
            vy_l = np.zeros(rl)
            for fi, f in enumerate(frames):
                if len(f.processingPasses) < 1:
                    continue
                fp = f.processingPasses[-1]
                grf = np.array(fp.groundContactForce)
                n_contacts = len(grf) // 3
                for idx in foot_indices:
                    if idx < n_contacts:
                        vy = grf[idx * 3 + 1]
                        body_name = contact_bodies[idx]
                        if '_r' in body_name:
                            vy_r[fi] = vy
                        else:
                            vy_l[fi] = vy

            total_vy = vy_r + vy_l
            mean_grf = np.mean(total_vy)
            if mean_grf < 0.3 * BW:
                continue  # not enough loading — swing

            # Static filter: gait alternates foot loading, so per-foot GRF
            # should vary substantially. Static standing gives near-zero std.
            # Threshold 0.1*BW cleanly separates static (~0) from walking (>0.3*BW).
            foot_std = max(np.std(vy_r), np.std(vy_l))
            if foot_std < 0.1 * BW:
                continue  # static — no alternating foot loading

            peak_foot = max(np.max(vy_r), np.max(vy_l))
            duration = rl * dt

            segments.append({
                'trial': trial,
                'start_frame': rs,
                'num_frames': rl,
                'mass_kg': mass_kg,
                'dt': dt,
                'duration': duration,
                'peak_foot_grf_bw': peak_foot / BW,
            })

    return segments


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

    state = model.initSystem()
    # Collect constrained flags before addForce invalidates the system
    constrained = [model.getCoordinateSet().get(i).isConstrained(state)
                   for i in range(model.getCoordinateSet().getSize())]
    for i in range(model.getCoordinateSet().getSize()):
        if constrained[i]:
            continue
        coord = model.getCoordinateSet().get(i)
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


# ─── Per-subject worker (runs in child process) ─────────────────────────────

DATASET_PREFIX = {
    "Moore2015_Formatted_With_Arm": "moore",
    "Tiziana2019_Formatted_With_Arm": "tiziana",
    "Carter2023_Formatted_With_Arm": "carter",
    "Falisse2017_Formatted_With_Arm": "falisse",
    "Fregly2012_Formatted_With_Arm": "fregly",
    "Hammer2013_Formatted_With_Arm": "hammer",
    "Han2023_Formatted_With_Arm": "han",
}


def _process_activity(output_name, b3d_path, subj_output, out_root,
                      trial, start_frame, num_frames):
    """Convert b3d slice and run SO+JR for one activity."""
    ik_path = os.path.join(subj_output, 'ik_results.mot')
    if not os.path.exists(ik_path):
        print(f"  [{output_name}] Converting b3d slice...", flush=True)
        ok = convert_b3d_slice(b3d_path, out_root, trial,
                               start_frame, num_frames, output_name)
        if not ok or not os.path.exists(ik_path):
            print(f"  [{output_name}] Conversion failed", flush=True)
            return False
    else:
        print(f"  [{output_name}] OpenSim files exist, skipping conversion", flush=True)

    ik_df = pd.read_csv(ik_path, sep='\t', skiprows=6)
    ik_duration = ik_df['time'].iloc[-1] - ik_df['time'].iloc[0]
    buf = min(BUFFER, ik_duration * 0.15)
    t0 = ik_df['time'].iloc[0] + buf
    t1 = ik_df['time'].iloc[-1] - buf
    if t1 - t0 < 0.2:
        print(f"  [{output_name}] Slice too short ({t1-t0:.1f}s)", flush=True)
        return False

    print(f"  [{output_name}] SO+JR [{t0:.2f}, {t1:.2f}]s...", flush=True)
    ok = run_jcf(subj_output, t0, t1)
    if ok:
        print(f"  [{output_name}] SUCCESS", flush=True)
    else:
        print(f"  [{output_name}] JCF output missing", flush=True)
    return ok


def process_one_subject(args):
    """
    Process a single subject. Designed to run in a worker process.
    Returns (output_name, status_key) where status_key is one of:
        'success', 'skip', 'scan_fail', 'convert_fail', 'jcf_fail', 'error'
    """
    dataset_name, subject_name, b3d_path, idx, total = args
    prefix = DATASET_PREFIX.get(dataset_name, dataset_name.split('_')[0].lower())
    output_name = f"{prefix}_{subject_name}"
    tag = f"[{idx+1}/{total}] {dataset_name}/{subject_name}"

    print(f"\n{tag}: Processing...", flush=True)

    try:
        if FULL_DURATION:
            # --- Full duration: extract ALL valid GRF segments ---
            print(f"  [{output_name}] Scanning for all valid segments...", flush=True)
            segments = scan_b3d_all_runs(b3d_path)
            if not segments:
                print(f"  [{output_name}] No valid segments", flush=True)
                return (output_name, 'scan_fail')

            # Naming: t{trial}, add _r{run} if multiple runs per trial
            trial_run_counts = {}
            for seg in segments:
                t = seg['trial']
                trial_run_counts[t] = trial_run_counts.get(t, 0) + 1

            trial_run_seen = {}
            n_ok = 0
            n_skip = 0
            for seg in segments:
                t = seg['trial']
                run_idx = trial_run_seen.get(t, 0)
                trial_run_seen[t] = run_idx + 1

                if trial_run_counts[t] == 1:
                    seg_name = f"{output_name}_t{t:02d}"
                else:
                    seg_name = f"{output_name}_t{t:02d}_r{run_idx:02d}"

                # Classify activity by per-foot peak GRF — skip walking
                is_running = seg['peak_foot_grf_bw'] > GRF_CAP_WALKING
                if not is_running:
                    n_skip += 1
                    continue
                out_root = OUTPUT_ROOT_RUNNING

                subj_output = os.path.join(out_root, seg_name)
                done = os.path.exists(os.path.join(
                    subj_output, 'jcf_output',
                    "BatchJCF_JointReaction_ReactionLoads.sto"))
                if done:
                    n_skip += 1
                    continue

                print(f"  [{seg_name}] running — trial {t}, "
                      f"frames {seg['start_frame']}-{seg['start_frame']+seg['num_frames']}, "
                      f"{seg['duration']:.1f}s, peak={seg['peak_foot_grf_bw']:.1f}BW",
                      flush=True)
                ok = _process_activity(seg_name, b3d_path, subj_output, out_root,
                                       seg['trial'], seg['start_frame'],
                                       seg['num_frames'])
                if ok:
                    n_ok += 1

            total_segs = len(segments)
            print(f"  [{output_name}] {n_ok} new + {n_skip} existing / {total_segs} segments",
                  flush=True)
            return (output_name, 'success' if (n_ok + n_skip) > 0 else 'jcf_fail')

        else:
            # --- Original: single best 2s window per activity ---
            subj_walking = os.path.join(OUTPUT_ROOT_WALKING, output_name)
            subj_running = os.path.join(OUTPUT_ROOT_RUNNING, output_name)
            walking_done = os.path.exists(os.path.join(
                subj_walking, 'jcf_output', "BatchJCF_JointReaction_ReactionLoads.sto"))
            running_done = os.path.exists(os.path.join(
                subj_running, 'jcf_output', "BatchJCF_JointReaction_ReactionLoads.sto"))

            print(f"  [{output_name}] Scanning b3d...", flush=True)
            scan_walk = scan_b3d_for_walking(b3d_path, WINDOW_DURATION, GRF_CAP_WALKING) if not walking_done else None
            scan_run = scan_b3d_for_walking(b3d_path, WINDOW_DURATION, GRF_CAP_RUNNING) if not running_done else None

            if walking_done and running_done:
                print(f"  [{output_name}] SKIP (both done)", flush=True)
                return (output_name, 'skip')

            if scan_walk is not None and not walking_done:
                trial, start_frame, num_frames, mass_kg = scan_walk
                print(f"  [{output_name}] walking — trial {trial}, frames {start_frame}-{start_frame+num_frames}, "
                      f"mass={mass_kg:.1f}kg", flush=True)
                _process_activity(output_name, b3d_path, subj_walking, OUTPUT_ROOT_WALKING,
                                  trial, start_frame, num_frames)

            if scan_run is not None and not running_done:
                trial, start_frame, num_frames, mass_kg = scan_run
                print(f"  [{output_name}] running — trial {trial}, frames {start_frame}-{start_frame+num_frames}, "
                      f"mass={mass_kg:.1f}kg", flush=True)
                _process_activity(output_name, b3d_path, subj_running, OUTPUT_ROOT_RUNNING,
                                  trial, start_frame, num_frames)

            if scan_walk is None and scan_run is None and not walking_done and not running_done:
                print(f"  [{output_name}] No valid window", flush=True)
                return (output_name, 'scan_fail')

            return (output_name, 'success')

    except Exception as e:
        print(f"  [{output_name}] ERROR: {e}", flush=True)
        traceback.print_exc()
        return (output_name, 'error')


# ─── Main batch loop ─────────────────────────────────────────────────────────

def main():
    global _scan_log_file
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    os.makedirs(OUTPUT_ROOT_WALKING, exist_ok=True)
    os.makedirs(OUTPUT_ROOT_RUNNING, exist_ok=True)
    _scan_log_file = open(SCAN_LOG, 'w')

    # Filter to specific datasets (set to None to process all)
    DATASETS_TO_PROCESS = None
    # Filter to specific subjects (set to None to process all in dataset)
    SUBJECTS_TO_PROCESS = None

    # Find all b3d files
    b3d_files = []
    for dataset in sorted(os.listdir(B3D_ROOT)):
        dataset_path = os.path.join(B3D_ROOT, dataset)
        if not os.path.isdir(dataset_path):
            continue
        if DATASETS_TO_PROCESS is not None and dataset not in DATASETS_TO_PROCESS:
            continue
        for subject in sorted(os.listdir(dataset_path)):
            if SUBJECTS_TO_PROCESS is not None and subject not in SUBJECTS_TO_PROCESS:
                continue
            subj_path = os.path.join(dataset_path, subject)
            b3d_list = glob.glob(os.path.join(subj_path, "*.b3d"))
            if b3d_list:
                b3d_files.append((dataset, subject, b3d_list[0]))

    total = len(b3d_files)
    print(f"Found {total} subjects to process")
    print(f"Output: {OUTPUT_ROOT}")
    print("=" * 60)

    # Build worker args
    worker_args = [
        (ds, subj, path, idx, total)
        for idx, (ds, subj, path) in enumerate(b3d_files)
    ]

    results = {"success": [], "skip": [], "scan_fail": [], "convert_fail": [],
               "jcf_fail": [], "error": []}

    for a in worker_args:
        output_name, status = process_one_subject(a)
        results[status].append(output_name)

    # Summary
    print("\n" + "=" * 60)
    print("BATCH PROCESSING SUMMARY")
    print(f"  Success:        {len(results['success'])}")
    print(f"  Skipped:        {len(results['skip'])}")
    print(f"  Scan fail:      {len(results['scan_fail'])}")
    print(f"  Convert fail:   {len(results['convert_fail'])}")
    print(f"  JCF fail:       {len(results['jcf_fail'])}")
    print(f"  Errors:         {len(results['error'])}")

    # Save results
    with open(os.path.join(OUTPUT_ROOT, 'batch_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_ROOT}/batch_results.json")

    _scan_log_file.close()
    print(f"Scan failure details: {SCAN_LOG}")


if __name__ == '__main__':
    main()

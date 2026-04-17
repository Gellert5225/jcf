"""
Validate training data quality for CNN training.
=================================================
Checks each subject's OpenSim SO + JR output for:

1. SO convergence  – muscle activations saturated at upper bound (≥0.99)
2. Reserve actuator usage – reserve forces too large relative to muscles
3. JCF magnitude sanity – peak resultant knee JCF outside physiological range

Usage:
    conda run -n jcf python validate_training_data.py [--activity running|walking|both]
                                                       [--threshold-activations 0.10]
                                                       [--threshold-reserve 5.0]
                                                       [--jcf-range 0.5 8.0]
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# ─── Defaults ────────────────────────────────────────────────────────────────

TRAINING_ROOT = "./jcf/training"
TESTING_ROOT = "./jcf/testing"

# Columns in the SO force file that are reserve actuators
RESERVE_PREFIX = "reserve_"

# Muscle activation columns (everything that is NOT reserve_ and NOT time and NOT calcn_*)
# We'll detect them programmatically.

# Reserve optimal forces from batch_process.py
ROTATIONAL_COORDS = {
    "pelvis_tilt", "pelvis_list", "pelvis_rotation",
    "hip_flexion_r", "hip_adduction_r", "hip_rotation_r",
    "knee_angle_r", "ankle_angle_r",
    "hip_flexion_l", "hip_adduction_l", "hip_rotation_l",
    "knee_angle_l", "ankle_angle_l",
    "lumbar_extension", "lumbar_bending", "lumbar_rotation",
    "arm_flex_r", "arm_add_r", "arm_rot_r",
    "elbow_flex_r", "pro_sup_r",
    "arm_flex_l", "arm_add_l", "arm_rot_l",
    "elbow_flex_l", "pro_sup_l",
}
TRANSLATIONAL_COORDS = {"pelvis_tx", "pelvis_ty", "pelvis_tz"}


def parse_sto(filepath):
    """Read an OpenSim .sto file into a pandas DataFrame."""
    with open(filepath) as f:
        header_lines = 0
        for line in f:
            header_lines += 1
            if line.strip() == "endheader":
                break
    return pd.read_csv(filepath, sep=r'\s+', skiprows=header_lines)


def get_reserve_cols(columns):
    """Return list of column names that are reserve actuators."""
    return [c for c in columns if c.startswith(RESERVE_PREFIX)]


def get_muscle_cols(columns):
    """Return list of column names that are muscle forces (not reserve, not time, not external)."""
    return [c for c in columns if c != "time"
            and not c.startswith(RESERVE_PREFIX)
            and not c.startswith("calcn_")]


def validate_subject(subject_dir, jcf_range, reserve_threshold_nm,
                     activation_sat_fraction):
    """
    Validate a single subject. Returns a dict of issues found, or None if clean.
    """
    output_dir = os.path.join(subject_dir, "jcf_output")
    metadata_file = os.path.join(subject_dir, "metadata.json")

    # Required files
    so_force_file = os.path.join(output_dir, "BatchJCF_StaticOptimization_force.sto")
    so_act_file = os.path.join(output_dir, "BatchJCF_StaticOptimization_activation.sto")
    jcf_file = os.path.join(output_dir, "BatchJCF_JointReaction_ReactionLoads.sto")

    if not os.path.isdir(output_dir):
        return {"missing": "jcf_output directory not found"}

    issues = {}

    # ── Load metadata ────────────────────────────────────────────────────
    if not os.path.exists(metadata_file):
        return {"missing": "metadata.json not found"}
    with open(metadata_file) as f:
        meta = json.load(f)
    mass_kg = meta.get("mass_kg")
    if mass_kg is None or mass_kg <= 0:
        return {"missing": f"invalid mass_kg: {mass_kg}"}
    bw = mass_kg * 9.81

    # ── 1. SO activation saturation ─────────────────────────────────────
    if os.path.exists(so_act_file):
        act_df = parse_sto(so_act_file)
        muscle_act_cols = get_muscle_cols(act_df.columns)
        if muscle_act_cols:
            act_vals = act_df[muscle_act_cols].values
            n_total = act_vals.size
            n_saturated = np.sum(act_vals >= 0.99)
            sat_fraction = n_saturated / n_total if n_total > 0 else 0
            if sat_fraction > activation_sat_fraction:
                # Find which muscles saturate the most
                per_muscle_sat = (act_vals >= 0.99).sum(axis=0)
                worst_idx = np.argsort(per_muscle_sat)[-3:][::-1]
                worst_muscles = [(muscle_act_cols[i], int(per_muscle_sat[i]))
                                 for i in worst_idx if per_muscle_sat[i] > 0]
                issues["activation_saturation"] = {
                    "fraction": round(float(sat_fraction), 4),
                    "n_saturated_elements": int(n_saturated),
                    "n_total_elements": int(n_total),
                    "worst_muscles": worst_muscles,
                }
    else:
        issues["missing_so_activation"] = so_act_file

    # ── 2. Reserve actuator magnitude ───────────────────────────────────
    if os.path.exists(so_force_file):
        force_df = parse_sto(so_force_file)
        reserve_cols = get_reserve_cols(force_df.columns)
        muscle_force_cols = get_muscle_cols(force_df.columns)

        if reserve_cols:
            reserve_data = force_df[reserve_cols].values  # N or Nm
            # Normalize reserves by their optimal force to get "equivalent activations"
            # rotational reserves have opt force = 1 Nm, translational = 10 N
            reserve_opt = np.ones(len(reserve_cols))
            for i, col in enumerate(reserve_cols):
                coord_name = col[len(RESERVE_PREFIX):]
                if coord_name in TRANSLATIONAL_COORDS:
                    reserve_opt[i] = 10.0

            # Peak absolute reserve force (normalized by opt force)
            abs_reserve = np.abs(reserve_data)
            peak_reserve_per_coord = abs_reserve.max(axis=0)
            mean_reserve_per_coord = abs_reserve.mean(axis=0)

            # Flag if any reserve contributes more than threshold
            flagged_reserves = []
            for i, col in enumerate(reserve_cols):
                peak_val = peak_reserve_per_coord[i]
                # For rotational (opt=1Nm), actual force in Nm
                # For translational (opt=10N), actual force in N
                # Compare raw force to threshold (in Nm or N)
                if peak_val > reserve_threshold_nm:
                    flagged_reserves.append({
                        "name": col,
                        "peak_abs": round(float(peak_val), 3),
                        "mean_abs": round(float(mean_reserve_per_coord[i]), 3),
                        "optimal_force": float(reserve_opt[i]),
                    })

            if flagged_reserves:
                # Total reserve RMS vs total muscle force RMS
                total_muscle_rms = np.sqrt(np.mean(
                    force_df[muscle_force_cols].values ** 2)) if muscle_force_cols else 0
                total_reserve_rms = np.sqrt(np.mean(reserve_data ** 2))
                issues["reserve_actuators"] = {
                    "n_flagged": len(flagged_reserves),
                    "flagged": flagged_reserves,
                    "total_reserve_rms": round(float(total_reserve_rms), 3),
                    "total_muscle_rms": round(float(total_muscle_rms), 3),
                }
    else:
        issues["missing_so_force"] = so_force_file

    # ── 3. JCF magnitude sanity ─────────────────────────────────────────
    if os.path.exists(jcf_file):
        jcf_df = parse_sto(jcf_file)
        fx_col = "walker_knee_r_on_tibia_r_in_tibia_r_fx"
        fy_col = "walker_knee_r_on_tibia_r_in_tibia_r_fy"
        fz_col = "walker_knee_r_on_tibia_r_in_tibia_r_fz"

        if all(c in jcf_df.columns for c in [fx_col, fy_col, fz_col]):
            fx = jcf_df[fx_col].values
            fy = jcf_df[fy_col].values
            fz = jcf_df[fz_col].values
            resultant = np.sqrt(fx**2 + fy**2 + fz**2)

            # Normalize to BW
            resultant_bw = resultant / bw
            peak_bw = float(np.max(resultant_bw))
            mean_bw = float(np.mean(resultant_bw))

            # Also check individual components in BW
            peak_fy_bw = float(np.max(np.abs(fy))) / bw  # axial (compressive)

            if peak_bw < jcf_range[0] or peak_bw > jcf_range[1]:
                issues["jcf_magnitude"] = {
                    "peak_resultant_bw": round(peak_bw, 3),
                    "mean_resultant_bw": round(mean_bw, 3),
                    "peak_fy_bw": round(peak_fy_bw, 3),
                    "range": jcf_range,
                }
            # Also flag if JCF is essentially zero (SO failed silently)
            if peak_bw < 0.1:
                issues["jcf_near_zero"] = {
                    "peak_resultant_bw": round(peak_bw, 3),
                }
        else:
            issues["missing_jcf_columns"] = list(jcf_df.columns)
    else:
        issues["missing_jcf_file"] = jcf_file

    return issues if issues else None


def scan_subjects(root, activity):
    """Find all subject directories under root/activity/."""
    subjects = []
    if activity == "both":
        activities = ["running", "walking"]
    else:
        activities = [activity]

    for act in activities:
        act_dir = os.path.join(root, act)
        if not os.path.isdir(act_dir):
            continue
        for name in sorted(os.listdir(act_dir)):
            subj_path = os.path.join(act_dir, name)
            if os.path.isdir(subj_path) and not name.startswith('.'):
                # Skip non-subject dirs (e.g. best_model.pt files show as dirs?)
                if os.path.exists(os.path.join(subj_path, "metadata.json")):
                    subjects.append((act, name, subj_path))
    return subjects


def main():
    parser = argparse.ArgumentParser(description="Validate SO/JCF training data quality")
    parser.add_argument("--activity", choices=["running", "walking", "both"],
                        default="both", help="Which activity to validate")
    parser.add_argument("--split", choices=["training", "testing", "both"],
                        default="training", help="Training or testing split")
    parser.add_argument("--threshold-activations", type=float, default=0.05,
                        help="Flag if >X fraction of muscle activation elements are ≥0.99 (default: 0.05)")
    parser.add_argument("--threshold-reserve", type=float, default=10.0,
                        help="Flag if any reserve peak abs force > X Nm/N (default: 10.0)")
    parser.add_argument("--jcf-min", type=float, default=0.5,
                        help="Min expected peak resultant JCF in BW (default: 0.5)")
    parser.add_argument("--jcf-max", type=float, default=8.0,
                        help="Max expected peak resultant JCF in BW (default: 8.0)")
    parser.add_argument("--output", type=str, default=None,
                        help="Save detailed report as JSON")
    args = parser.parse_args()

    jcf_range = [args.jcf_min, args.jcf_max]

    roots = []
    if args.split in ("training", "both"):
        roots.append(("training", TRAINING_ROOT))
    if args.split in ("testing", "both"):
        roots.append(("testing", TESTING_ROOT))

    all_subjects = []
    for split_name, root in roots:
        subjects = scan_subjects(root, args.activity)
        for act, name, path in subjects:
            all_subjects.append((split_name, act, name, path))

    print(f"Scanning {len(all_subjects)} subjects "
          f"(activity={args.activity}, split={args.split})")
    print(f"Thresholds: activation_sat={args.threshold_activations}, "
          f"reserve={args.threshold_reserve} Nm/N, "
          f"JCF range={jcf_range} BW")
    print("=" * 80)

    flagged = {}
    clean_count = 0
    missing_count = 0
    category_counts = {"activation_saturation": 0, "reserve_actuators": 0,
                       "jcf_magnitude": 0, "jcf_near_zero": 0, "missing": 0}

    for split_name, act, name, path in all_subjects:
        issues = validate_subject(path, jcf_range, args.threshold_reserve,
                                  args.threshold_activations)
        if issues is None:
            clean_count += 1
        else:
            key = f"{split_name}/{act}/{name}"
            flagged[key] = issues
            for cat in category_counts:
                if cat in issues or f"missing_{cat.split('_')[0]}" in issues:
                    category_counts[cat] += 1
            # Count missing files
            for k in issues:
                if k.startswith("missing"):
                    category_counts["missing"] += 1
                    break

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\nResults: {clean_count} clean, {len(flagged)} flagged "
          f"out of {len(all_subjects)} subjects\n")

    # Category breakdown
    print("Issue breakdown:")
    for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        if count > 0:
            print(f"  {cat}: {count} subjects")

    # ── Detailed per-subject report ──────────────────────────────────────
    if flagged:
        print(f"\n{'─' * 80}")
        print("FLAGGED SUBJECTS:")
        print(f"{'─' * 80}")

        # Group by issue type
        for key in sorted(flagged.keys()):
            issues = flagged[key]
            print(f"\n  {key}:")
            for issue_type, detail in issues.items():
                if issue_type == "activation_saturation":
                    print(f"    Activation saturation: {detail['fraction']*100:.1f}% "
                          f"of elements ≥0.99")
                    for muscle, count in detail["worst_muscles"]:
                        print(f"      {muscle}: {count} frames")
                elif issue_type == "reserve_actuators":
                    print(f"    Reserve actuators: {detail['n_flagged']} flagged "
                          f"(reserve RMS={detail['total_reserve_rms']:.1f}, "
                          f"muscle RMS={detail['total_muscle_rms']:.1f})")
                    for r in detail["flagged"]:
                        print(f"      {r['name']}: peak={r['peak_abs']:.1f}, "
                              f"mean={r['mean_abs']:.1f}")
                elif issue_type == "jcf_magnitude":
                    print(f"    JCF out of range: peak={detail['peak_resultant_bw']:.2f} BW "
                          f"(expected {detail['range'][0]}-{detail['range'][1]} BW), "
                          f"peak Fy={detail['peak_fy_bw']:.2f} BW")
                elif issue_type == "jcf_near_zero":
                    print(f"    JCF near zero: peak={detail['peak_resultant_bw']:.3f} BW")
                elif issue_type.startswith("missing"):
                    print(f"    {issue_type}: {detail}")

    # ── Save JSON report ─────────────────────────────────────────────────
    output_file = args.output
    if output_file is None:
        output_file = os.path.join(TRAINING_ROOT, "validation_report.json")

    report = {
        "settings": {
            "activity": args.activity,
            "split": args.split,
            "threshold_activations": args.threshold_activations,
            "threshold_reserve": args.threshold_reserve,
            "jcf_range": jcf_range,
        },
        "summary": {
            "total": len(all_subjects),
            "clean": clean_count,
            "flagged": len(flagged),
            "category_counts": category_counts,
        },
        "flagged_subjects": flagged,
    }
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nDetailed report saved to: {output_file}")


if __name__ == "__main__":
    main()

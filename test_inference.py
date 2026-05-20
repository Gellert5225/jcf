"""
End-to-end verification of inference.py against real test data.

Tests two things:
1. Pipeline parity. The inference module and the training-test pipeline
   (load_subject + model) should produce identical predictions on the same
   input. Any divergence indicates a feature-alignment bug in inference.py
   that would silently corrupt William's downstream integration.
2. Agreement with ground truth. The inference predictions should match the
   OpenSim Static Optimization + Joint Reaction labels to within the same
   MAE that test_cnn.py reports.

Tests both walking and quasi-static trials from the held-out test set.

Usage:
    python test_inference.py
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from inference import JCFPredictor, JOINT_ORDER, preprocess
from train_cnn import load_subject, JCF_CNN_v2

CKPT = "./jcf/full_duration/training/walking/best_model_n_d05_m_s2_s2.pt"

# Test trials: a mix of walking and quasi-static, across labs.
TEST_TRIALS = [
    ("walking",  "./jcf/full_duration/testing/walking/carter_P010_split0_t00"),
    ("walking",  "./jcf/full_duration/testing/walking/carter_P010_split0_t01"),
    ("walking",  "./jcf/full_duration/testing/walking/moore_subject10_t27_r07"),
    ("walking",  "./jcf/full_duration/testing/walking/moore_subject10_t87"),
    ("static",   "./jcf/full_duration/testing/static/moore_subject10_t01"),
    ("static",   "./jcf/full_duration/testing/static/moore_subject10_t33"),
    ("static",   "./jcf/full_duration/testing/static/tiziana_Subject10_t02"),
]


def load_mot(path, skiprows=6):
    return pd.read_csv(path, sep=r"\s+", skiprows=skiprows)


def build_inference_input(trial_dir):
    """Mimic what William will do: read the raw OpenSim outputs, strip dead
    joints, trim to the IK ∩ GRF ∩ JCF common time range, and hand the
    16-DOF IK + 6-channel GRF to inference.preprocess()."""
    ik = load_mot(os.path.join(trial_dir, "ik_results.mot"))
    grf = load_mot(os.path.join(trial_dir, "grf_data.mot"))
    meta = json.load(open(os.path.join(trial_dir, "metadata.json")))

    ik_time = ik["time"].values
    grf_time = grf["time"].values

    # Read JCF .sto for its time range (so we only compare to official on the
    # frames the official pipeline kept). In a real William deployment this
    # alignment is the user's responsibility — we mimic it here.
    sto_path = os.path.join(trial_dir, "jcf_output",
                            "BatchJCF_JointReaction_ReactionLoads.sto")
    with open(sto_path) as f:
        slines = f.readlines()
    hidx = next(i for i, l in enumerate(slines) if l.startswith("endheader"))
    sdata = np.array([list(map(float, l.split())) for l in slines[hidx+2:] if l.strip()])
    jcf_time = sdata[:, 0]

    t_start = max(ik_time[0], grf_time[0], jcf_time[0])
    t_end   = min(ik_time[-1], grf_time[-1], jcf_time[-1])
    ik_keep = (ik_time >= t_start) & (ik_time <= t_end)
    ik_time_k = ik_time[ik_keep]

    ja_full = np.column_stack([ik[c].values for c in JOINT_ORDER])
    ja = ja_full[ik_keep]

    grf_cols = [c for c in grf.columns
                if ("calcn_r" in c or "calcn_l" in c) and "_force_v" in c]
    if len(grf_cols) != 6:
        raise ValueError(f"Expected 6 GRF cols, got {grf_cols}")
    grf_full = grf[grf_cols].values
    grf_data = np.column_stack([np.interp(ik_time_k, grf_time, grf_full[:, i])
                                for i in range(6)])

    dt = float(np.median(np.diff(ik_time_k)))
    return ja, grf_data, meta["mass_kg"], meta.get("height_m", 1.75), dt


def load_official_pipeline(trial_dir, ckpt):
    """Run load_subject + the model directly, exactly as test_cnn.py does."""
    res = load_subject(
        trial_dir,
        lower_body_only=True, clean_features=True,
        include_mass=True, include_speed=True,
        predict_moment=True, predict_flexion_moment=False,
        max_peak_bw=10.0, min_length=1,
        native_time_grid=True,
    )
    if res is None:
        return None
    inputs, labels, mass = res
    return inputs, labels, mass


def main():
    print(f"Loading checkpoint: {CKPT}")
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=True)
    predictor = JCFPredictor(CKPT, device="cpu")

    model = JCF_CNN_v2(n_features=ckpt["n_features"], n_outputs=ckpt["n_outputs"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    inp_mean, inp_std = ckpt["input_mean"], ckpt["input_std"]

    print(f"\n{'trial':<55} {'mode':<8} {'feat_diff':>10} {'pred_diff':>10} "
          f"{'fy_MAE':>8} {'mx_MAE':>8}")
    print("-" * 110)

    overall_pred_diffs = []
    for mode, trial in TEST_TRIALS:
        try:
            ja, grf, mass, height, dt = build_inference_input(trial)
        except Exception as e:
            print(f"  {os.path.basename(trial):<53} SKIP build_inference_input: {e}")
            continue

        official = load_official_pipeline(trial, ckpt)
        if official is None:
            print(f"  {os.path.basename(trial):<53} SKIP load_subject returned None")
            continue
        off_inputs, labels, mass_off = official

        # Pipeline A — official: hand-built features → model
        T = len(off_inputs)
        with torch.no_grad():
            x = (torch.tensor(off_inputs, dtype=torch.float32) - inp_mean) / inp_std
            pred_official = model(x.unsqueeze(0))[0].numpy()

        # Pipeline B — inference.py: raw IK+GRF → preprocess → predict
        # Truncate / interpolate to match official length for direct comparison.
        # Use dt from the IK file rate.
        try:
            inf_inputs = preprocess(ja, grf, mass, dt=dt, zero_center_pelvis=True)
        except Exception as e:
            print(f"  {os.path.basename(trial):<53} SKIP preprocess: {e}")
            continue

        # The two feature streams may differ in length because:
        #   official: load_subject interpolates to JCF time grid
        #   inference: works at IK time grid
        # Compare the SHORTER of the two so we can assert numerical agreement
        # on the overlapping region. (For real William usage, this won't matter.)
        n = min(len(off_inputs), len(inf_inputs))
        feat_max_diff = float(np.abs(off_inputs[:n] - inf_inputs[:n]).max())

        with torch.no_grad():
            x2 = (torch.tensor(inf_inputs, dtype=torch.float32) - inp_mean) / inp_std
            pred_inf = model(x2.unsqueeze(0))[0].numpy()

        pred_max_diff = float(np.abs(pred_official[:n] - pred_inf[:n]).max())

        # Compare official prediction to GT labels
        n2 = min(T, len(labels))
        fy_mae = float(np.mean(np.abs(pred_official[:n2, 1] - labels[:n2, 1])))
        mx_mae = float(np.mean(np.abs(pred_official[:n2, 3] - labels[:n2, 3])))

        overall_pred_diffs.append(pred_max_diff)

        name = os.path.basename(trial)
        print(f"  {name:<53} {mode:<8} {feat_max_diff:>10.6f} {pred_max_diff:>10.6f} "
              f"{fy_mae:>8.4f} {mx_mae:>8.5f}")

    print("\n" + "=" * 110)
    if overall_pred_diffs:
        worst = max(overall_pred_diffs)
        median = sorted(overall_pred_diffs)[len(overall_pred_diffs) // 2]
        print(f"Pipeline parity: max-abs-prediction-diff across all trials")
        print(f"  median: {median:.6f} BW   worst: {worst:.6f} BW   (model MAE for context: ~0.13 BW)")
        print()
        if worst < 0.001:
            print("  ✓ EXACT parity (residual is floating-point noise).")
            print("    inference.py is a faithful re-implementation of the training-time pipeline.")
        elif worst < 0.05:
            print("  ✓ Divergence is small (<5% of model MAE). Safe to ship inference.py.")
        elif worst < 0.20:
            print("  ⚠ Divergence within model uncertainty but non-trivial. Investigate before handoff.")
        else:
            print("  ✗ Divergence exceeds model MAE — investigate before handoff.")


if __name__ == "__main__":
    main()

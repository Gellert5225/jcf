"""
Feature Importance Analysis for Exp A (full-body, 123 features)
================================================================
Uses gradient × input attribution to understand which features drive
knee JCF predictions, and whether upper-body joints contribute signal
or just noise.
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from train_cnn import JCF_CNN, load_subject

# ─── Joint DOF names (37 total) ──────────────────────────────────────────────
JOINT_NAMES = [
    "pelvis_tilt", "pelvis_list", "pelvis_rotation",
    "pelvis_tx", "pelvis_ty", "pelvis_tz",
    "hip_flexion_r", "hip_adduction_r", "hip_rotation_r",
    "knee_angle_r", "ankle_angle_r", "subtalar_angle_r", "mtp_angle_r",
    "hip_flexion_l", "hip_adduction_l", "hip_rotation_l",
    "knee_angle_l", "ankle_angle_l", "subtalar_angle_l", "mtp_angle_l",
    "lumbar_extension", "lumbar_bending", "lumbar_rotation",
    "arm_flex_r", "arm_add_r", "arm_rot_r",
    "elbow_flex_r", "pro_sup_r", "wrist_flex_r", "wrist_dev_r",
    "arm_flex_l", "arm_add_l", "arm_rot_l",
    "elbow_flex_l", "pro_sup_l", "wrist_flex_l", "wrist_dev_l",
]

N_JOINTS = 37
N_LOWER = 20  # joints 0-19

# Feature groups in 123-feature layout:
# [0:37]   = positions
# [37:74]  = velocities
# [74:111] = accelerations
# [111:117] = GRF forces (6)
# [117:123] = GRF velocities (6)

FEATURE_NAMES = (
    [f"pos_{j}" for j in JOINT_NAMES] +
    [f"vel_{j}" for j in JOINT_NAMES] +
    [f"acc_{j}" for j in JOINT_NAMES] +
    ["grf_calcn_r_x", "grf_calcn_r_y", "grf_calcn_r_z",
     "grf_calcn_l_x", "grf_calcn_l_y", "grf_calcn_l_z"] +
    ["grf_vel_calcn_r_x", "grf_vel_calcn_r_y", "grf_vel_calcn_r_z",
     "grf_vel_calcn_l_x", "grf_vel_calcn_l_y", "grf_vel_calcn_l_z"]
)

GROUP_LABELS = {
    "lower_body_pos": list(range(0, N_LOWER)),
    "upper_body_pos": list(range(N_LOWER, N_JOINTS)),
    "lower_body_vel": list(range(N_JOINTS, N_JOINTS + N_LOWER)),
    "upper_body_vel": list(range(N_JOINTS + N_LOWER, 2 * N_JOINTS)),
    "lower_body_acc": list(range(2 * N_JOINTS, 2 * N_JOINTS + N_LOWER)),
    "upper_body_acc": list(range(2 * N_JOINTS + N_LOWER, 3 * N_JOINTS)),
    "grf_forces": list(range(3 * N_JOINTS, 3 * N_JOINTS + 6)),
    "grf_velocities": list(range(3 * N_JOINTS + 6, 3 * N_JOINTS + 12)),
}


def compute_gradient_x_input(model, inputs, norm_mean, norm_std):
    """
    Compute gradient × input attribution for each feature.
    
    Args:
        model: trained JCF_CNN
        inputs: [T, 123] raw (unnormalized) numpy array
        norm_mean, norm_std: [123] tensors
    
    Returns:
        attributions: [T, 123] numpy array (absolute gradient × input)
    """
    # Normalize
    x = torch.tensor(inputs, dtype=torch.float32)
    x = (x - norm_mean) / norm_std
    x = x.unsqueeze(0)  # [1, T, 123]
    x.requires_grad_(True)

    # Forward pass
    model.eval()
    preds = model(x)  # [1, T, 3]

    # Backprop from Fy (the dominant axial component, index 1)
    fy_sum = preds[0, :, 1].sum()
    fy_sum.backward()

    # gradient × input
    attr = (x.grad[0] * x[0]).detach().numpy()  # [T, 123]
    return np.abs(attr)


def main():
    # Load exp A checkpoint
    ckpt_path = "./jcf/training/running/best_model_a.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    n_features = ckpt["n_features"]
    print(f"Loaded exp A model: {n_features} features, epoch {ckpt['epoch']}")

    model = JCF_CNN(n_features=n_features)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    norm_mean = ckpt["input_mean"]
    norm_std = ckpt["input_std"]
    if isinstance(norm_mean, np.ndarray):
        norm_mean = torch.tensor(norm_mean, dtype=torch.float32)
        norm_std = torch.tensor(norm_std, dtype=torch.float32)

    # Load test subjects
    test_root = "./jcf/testing/running"
    test_dirs = sorted([
        os.path.join(test_root, d)
        for d in os.listdir(test_root)
        if os.path.isdir(os.path.join(test_root, d))
    ])

    all_attr = []
    for subj_dir in test_dirs:
        result = load_subject(subj_dir, lower_body_only=False)
        if result is None:
            continue
        inputs, labels, mass = result
        name = os.path.basename(subj_dir)
        attr = compute_gradient_x_input(model, inputs, norm_mean, norm_std)
        # Weighted by |Fy| magnitude — importance at high-force frames
        fy_mag = np.abs(labels[:, 1])
        weighted_attr = attr * fy_mag[:, None]
        all_attr.append(weighted_attr)
        print(f"  {name}: {inputs.shape[0]} frames")

    # Aggregate across all test subjects
    concat_attr = np.concatenate(all_attr, axis=0)  # [N_total, 123]
    mean_attr = concat_attr.mean(axis=0)  # [123]

    # ─── Per-feature ranking ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Top 20 features by gradient×input attribution (Fy-weighted)")
    print("=" * 70)
    ranking = np.argsort(mean_attr)[::-1]
    for rank, idx in enumerate(ranking[:20]):
        print(f"  {rank+1:2d}. {FEATURE_NAMES[idx]:30s}  attr={mean_attr[idx]:.6f}")

    # ─── Group-level summary ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Feature group attribution (mean across features in group)")
    print("=" * 70)
    group_totals = {}
    for group_name, indices in GROUP_LABELS.items():
        group_mean = mean_attr[indices].mean()
        group_sum = mean_attr[indices].sum()
        group_totals[group_name] = group_sum
        print(f"  {group_name:20s}  mean={group_mean:.6f}  sum={group_sum:.6f}  (n={len(indices)})")

    total_attr = mean_attr.sum()
    lower_attr = sum(group_totals[g] for g in group_totals if 'lower' in g)
    upper_attr = sum(group_totals[g] for g in group_totals if 'upper' in g)
    grf_attr = sum(group_totals[g] for g in group_totals if 'grf' in g)

    print(f"\n  Lower body total: {lower_attr:.6f} ({100*lower_attr/total_attr:.1f}%)")
    print(f"  Upper body total: {upper_attr:.6f} ({100*upper_attr/total_attr:.1f}%)")
    print(f"  GRF total:        {grf_attr:.6f} ({100*grf_attr/total_attr:.1f}%)")

    # ─── Plot ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Bar chart: per-feature importance
    ax = axes[0]
    colors = []
    for i in range(123):
        if i < N_LOWER or (N_JOINTS <= i < N_JOINTS + N_LOWER) or (2*N_JOINTS <= i < 2*N_JOINTS + N_LOWER):
            colors.append('#2196F3')  # blue = lower body
        elif i >= 3 * N_JOINTS:
            colors.append('#4CAF50')  # green = GRF
        else:
            colors.append('#FF9800')  # orange = upper body
    ax.bar(range(123), mean_attr, color=colors, width=1.0)
    ax.set_xlabel("Feature index")
    ax.set_ylabel("Mean |grad × input| (Fy-weighted)")
    ax.set_title("Per-Feature Attribution — Exp A (full body, 123 features)")
    # Legend
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor='#2196F3', label='Lower body'),
        Patch(facecolor='#FF9800', label='Upper body'),
        Patch(facecolor='#4CAF50', label='GRF'),
    ])

    # Bar chart: group-level
    ax = axes[1]
    groups = list(GROUP_LABELS.keys())
    group_sums = [mean_attr[GROUP_LABELS[g]].sum() for g in groups]
    group_colors = ['#2196F3' if 'lower' in g else '#FF9800' if 'upper' in g else '#4CAF50' for g in groups]
    ax.barh(groups, group_sums, color=group_colors)
    ax.set_xlabel("Total attribution (sum)")
    ax.set_title("Feature Group Attribution")
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig("./jcf/testing/running/feature_importance_a.png", dpi=150)
    print(f"\nPlot saved to ./jcf/testing/running/feature_importance_a.png")


if __name__ == "__main__":
    main()

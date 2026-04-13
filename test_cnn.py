"""
Test the trained 1D CNN on held-out subjects in jcf/testing/.
Produces per-subject metrics and a plot comparing predicted vs ground truth JCF.

Usage:
    conda run -n jcf python test_cnn.py
"""

import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt

# Reuse data loading and model from train_cnn
from train_cnn import load_subject, JCF_CNN

TEST_ROOT = "./jcf/testing"
MODEL_PATH = "./jcf/training/best_model.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def test():
    # Load checkpoint
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    n_features = checkpoint['n_features']
    window_size = checkpoint['window_size']

    model = JCF_CNN(n_features=n_features, n_outputs=3).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Loaded model from epoch {checkpoint['epoch']+1} "
          f"(val_loss={checkpoint['val_loss']:.6f})")
    print(f"Window size: {window_size}, Features: {n_features}, Device: {DEVICE}")

    # Find test subjects
    subject_dirs = []
    for name in sorted(os.listdir(TEST_ROOT)):
        subj_dir = os.path.join(TEST_ROOT, name)
        jcf_sto = os.path.join(subj_dir, 'jcf_output',
                               'BatchJCF_JointReaction_ReactionLoads.sto')
        if os.path.isdir(subj_dir) and os.path.exists(jcf_sto):
            subject_dirs.append((name, subj_dir))

    print(f"\nFound {len(subject_dirs)} test subjects")

    if not subject_dirs:
        print("No test data found.")
        return

    # Predict on each subject using sliding window, then average overlaps
    all_results = []
    for subj_name, subj_dir in subject_dirs:
        result = load_subject(subj_dir)
        if result is None:
            print(f"  {subj_name}: SKIP (missing files)")
            continue

        inputs, labels, mass = result
        T = len(labels)
        BW = mass * 9.81

        # Predict full sequence using overlapping windows, average predictions
        pred_sum = np.zeros((T, 3))
        pred_count = np.zeros(T)

        with torch.no_grad():
            for start in range(0, T - window_size + 1, 1):
                end = start + window_size
                inp = torch.tensor(inputs[start:end], dtype=torch.float32)
                inp = inp.unsqueeze(0).to(DEVICE)  # [1, W, F]
                out = model(inp)                     # [1, W, 3]
                pred_sum[start:end] += out[0].cpu().numpy()
                pred_count[start:end] += 1

        # Handle edges (frames with fewer overlapping windows)
        valid = pred_count > 0
        preds = np.zeros_like(pred_sum)
        preds[valid] = pred_sum[valid] / pred_count[valid, None]

        # Metrics (in BW)
        errors = preds[valid] - labels[valid]
        mae = np.mean(np.abs(errors), axis=0)
        rmse = np.sqrt(np.mean(errors**2, axis=0))

        # Resultant force
        gt_resultant = np.sqrt(np.sum(labels[valid]**2, axis=1))
        pred_resultant = np.sqrt(np.sum(preds[valid]**2, axis=1))
        res_mae = np.mean(np.abs(pred_resultant - gt_resultant))
        res_rmse = np.sqrt(np.mean((pred_resultant - gt_resultant)**2))

        # Correlation
        corr_fy = np.corrcoef(preds[valid, 1], labels[valid, 1])[0, 1]
        corr_res = np.corrcoef(pred_resultant, gt_resultant)[0, 1]

        peak_gt = np.max(np.abs(gt_resultant))
        peak_pred = np.max(np.abs(pred_resultant))

        print(f"\n  {subj_name} ({T} frames, mass={mass:.1f}kg)")
        print(f"    Component MAE  (BW):  Fx={mae[0]:.4f}  Fy={mae[1]:.4f}  Fz={mae[2]:.4f}")
        print(f"    Component RMSE (BW):  Fx={rmse[0]:.4f}  Fy={rmse[1]:.4f}  Fz={rmse[2]:.4f}")
        print(f"    Resultant MAE:  {res_mae:.4f} BW   RMSE: {res_rmse:.4f} BW")
        print(f"    Fy correlation: {corr_fy:.4f}   Resultant correlation: {corr_res:.4f}")
        print(f"    Peak resultant:  GT={peak_gt:.3f} BW  Pred={peak_pred:.3f} BW")

        all_results.append({
            'name': subj_name,
            'T': T,
            'mass': mass,
            'labels': labels,
            'preds': preds,
            'valid': valid,
        })

    # ── Plot ──────────────────────────────────────────────────────────────────
    n_subjects = len(all_results)
    fig, axes = plt.subplots(n_subjects, 2, figsize=(14, 5 * n_subjects),
                             squeeze=False)

    for i, res in enumerate(all_results):
        labels = res['labels']
        preds = res['preds']
        valid = res['valid']
        T = res['T']
        time = np.arange(T) * 0.01  # 100 Hz

        # Left: Axial component (Fy)
        ax = axes[i, 0]
        ax.plot(time[valid], labels[valid, 1], 'b-', linewidth=1.5, label='Ground Truth')
        ax.plot(time[valid], preds[valid, 1], 'r--', linewidth=1.5, label='Predicted')
        ax.set_ylabel('Fy (BW)')
        ax.set_xlabel('Time (s)')
        ax.set_title(f'{res["name"]} — Axial Force (Fy)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Right: Resultant
        ax = axes[i, 1]
        gt_res = np.sqrt(np.sum(labels[valid]**2, axis=1))
        pred_res = np.sqrt(np.sum(preds[valid]**2, axis=1))
        ax.plot(time[valid], gt_res, 'b-', linewidth=1.5, label='Ground Truth')
        ax.plot(time[valid], pred_res, 'r--', linewidth=1.5, label='Predicted')
        ax.axhspan(2.5, 3.5, alpha=0.1, color='green', label='Expected peak range')
        ax.set_ylabel('Resultant JCF (BW)')
        ax.set_xlabel('Time (s)')
        ax.set_title(f'{res["name"]} — Resultant JCF')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(TEST_ROOT, 'test_results.png')
    plt.savefig(out_path, dpi=150)
    print(f"\nPlot saved to {out_path}")
    plt.close()


if __name__ == '__main__':
    test()

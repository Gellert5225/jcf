"""
Test the trained 1D CNN on held-out subjects in jcf/testing/.
Produces per-subject metrics and a plot comparing predicted vs ground truth JCF.

Usage:
    conda run -n jcf python test_cnn.py
    conda run -n jcf python test_cnn.py --exp a
    conda run -n jcf python test_cnn.py --exp b
"""

import os
import json
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

# Reuse data loading and model from train_cnn
from train_cnn import load_subject, JCF_CNN, JCF_CNN_v2, JCF_Transformer, JCF_TCN, JCF_FFT_MLP

TEST_ROOT = "./jcf/testing/running"
DEVICE = "cpu"


def test(exp=None):
    suffix = f"_{exp}" if exp else ""
    model_name = f"best_model_{exp}.pt" if exp else "best_model.pt"
    model_path = f"./jcf/full_duration/training/running/{model_name}"

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=True)
    log_targets = checkpoint.get('log_targets', False)
    lower_body_only = checkpoint.get('lower_body_only', False)
    model_class = checkpoint.get('model_class', 'v1')
    jcf_subdir = checkpoint.get('jcf_subdir', 'jcf_output')
    clean_features = checkpoint.get('clean_features', False)
    n_features = checkpoint['n_features']
    input_mean = checkpoint['input_mean']
    input_std = checkpoint['input_std']

    if model_class == 'tcn':
        ModelClass = JCF_TCN
    elif model_class == 'fft_mlp':
        ModelClass = JCF_FFT_MLP
    elif model_class == 'transformer':
        ModelClass = JCF_Transformer
    elif model_class == 'v2':
        ModelClass = JCF_CNN_v2
    else:
        ModelClass = JCF_CNN
    model = ModelClass(n_features=n_features, n_outputs=3).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Loaded model from epoch {checkpoint['epoch']+1} "
          f"(val_loss={checkpoint['val_loss']:.6f})")
    print(f"Features: {n_features}, Device: {DEVICE}")
    if lower_body_only:
        print("Using lower-body joints only (0-19)")
    if clean_features:
        print("Clean features: pelvis translations zero-centered, dead joints removed")
    if jcf_subdir != 'jcf_output':
        print(f"Using JCF labels from {jcf_subdir}/")

    # Find test subjects
    subject_dirs = []
    for name in sorted(os.listdir(TEST_ROOT)):
        subj_dir = os.path.join(TEST_ROOT, name)
        jcf_sto = os.path.join(subj_dir, jcf_subdir,
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
        result = load_subject(subj_dir, lower_body_only=lower_body_only,
                             jcf_subdir=jcf_subdir, clean_features=clean_features)
        if result is None:
            print(f"  {subj_name}: SKIP (missing files)")
            continue

        inputs, labels, mass = result
        T = len(labels)
        BW = mass * 9.81

        # Single forward pass on the full sequence (with normalization)
        with torch.no_grad():
            inp = torch.tensor(inputs, dtype=torch.float32)
            inp = (inp - input_mean) / input_std
            inp = inp.unsqueeze(0).to(DEVICE)
            out = model(inp)  # [1, T, 3]
            preds = out[0].cpu().numpy()

        # Inverse log-space transform if model was trained with log targets
        if log_targets:
            preds = np.sign(preds) * (np.exp(np.abs(preds)) - 1)

        valid = np.ones(T, dtype=bool)

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
    fig, axes = plt.subplots(n_subjects, 4, figsize=(28, 5 * n_subjects),
                             squeeze=False)

    for i, res in enumerate(all_results):
        labels = res['labels']
        preds = res['preds']
        valid = res['valid']
        T = res['T']
        time = np.arange(T) * 0.01  # 100 Hz

        # Col 0: Fx
        ax = axes[i, 0]
        ax.plot(time[valid], labels[valid, 0], 'b-', linewidth=1.5, label='Ground Truth')
        ax.plot(time[valid], preds[valid, 0], 'r--', linewidth=1.5, label='Predicted')
        ax.set_ylabel('Fx (BW)')
        ax.set_xlabel('Time (s)')
        ax.set_title(f'{res["name"]} — Fx')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Col 1: Fy (axial)
        ax = axes[i, 1]
        ax.plot(time[valid], labels[valid, 1], 'b-', linewidth=1.5, label='Ground Truth')
        ax.plot(time[valid], preds[valid, 1], 'r--', linewidth=1.5, label='Predicted')
        ax.set_ylabel('Fy (BW)')
        ax.set_xlabel('Time (s)')
        ax.set_title(f'{res["name"]} — Fy')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Col 2: Fz
        ax = axes[i, 2]
        ax.plot(time[valid], labels[valid, 2], 'b-', linewidth=1.5, label='Ground Truth')
        ax.plot(time[valid], preds[valid, 2], 'r--', linewidth=1.5, label='Predicted')
        ax.set_ylabel('Fz (BW)')
        ax.set_xlabel('Time (s)')
        ax.set_title(f'{res["name"]} — Fz')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Col 3: Resultant
        ax = axes[i, 3]
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
    out_path = os.path.join(TEST_ROOT, f'test_results{suffix}.png')
    plt.savefig(out_path, dpi=150)
    print(f"\nPlot saved to {out_path}")
    plt.close()

    # ── Write GT vs Predicted to text file ────────────────────────────────────
    txt_path = os.path.join(TEST_ROOT, f'test_results{suffix}.txt')
    with open(txt_path, 'w') as f:
        for res in all_results:
            labels = res['labels']
            preds = res['preds']
            valid = res['valid']
            T = res['T']
            time = np.arange(T) * 0.01

            f.write(f"Subject: {res['name']}  (T={T}, mass={res['mass']:.1f}kg)\n")
            f.write(f"{'Frame':>6}  {'Time':>8}  "
                    f"{'GT_Fx':>10}  {'GT_Fy':>10}  {'GT_Fz':>10}  {'GT_Res':>10}  "
                    f"{'Pred_Fx':>10}  {'Pred_Fy':>10}  {'Pred_Fz':>10}  {'Pred_Res':>10}\n")
            f.write("-" * 114 + "\n")

            for t in range(T):
                if not valid[t]:
                    continue
                gt_res = np.sqrt(np.sum(labels[t]**2))
                pred_res = np.sqrt(np.sum(preds[t]**2))
                f.write(f"{t:>6}  {time[t]:>8.3f}  "
                        f"{labels[t,0]:>10.5f}  {labels[t,1]:>10.5f}  {labels[t,2]:>10.5f}  {gt_res:>10.5f}  "
                        f"{preds[t,0]:>10.5f}  {preds[t,1]:>10.5f}  {preds[t,2]:>10.5f}  {pred_res:>10.5f}\n")

            f.write("\n")

    print(f"GT vs Predicted written to {txt_path}")

    # ── Scatter plot: predicted peak vs GT peak (diagnostic) ──────────────────
    from scipy.signal import find_peaks

    all_gt_peaks = []
    all_pred_peaks = []
    peak_labels = []
    for res in all_results:
        labels = res['labels']
        preds = res['preds']
        valid = res['valid']
        gt_res = np.sqrt(np.sum(labels[valid]**2, axis=1))
        pred_res = np.sqrt(np.sum(preds[valid]**2, axis=1))
        peaks, _ = find_peaks(gt_res, height=0.5, distance=15)
        for p in peaks:
            all_gt_peaks.append(gt_res[p])
            all_pred_peaks.append(pred_res[p])
            peak_labels.append(res['name'])

    if all_gt_peaks:
        all_gt_peaks = np.array(all_gt_peaks)
        all_pred_peaks = np.array(all_pred_peaks)

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(all_gt_peaks, all_pred_peaks, alpha=0.7, edgecolors='k', linewidth=0.5)
        lims = [0, max(all_gt_peaks.max(), all_pred_peaks.max()) * 1.1]
        ax.plot(lims, lims, 'k--', linewidth=1, label='y = x')

        # Linear fit
        slope, intercept = np.polyfit(all_gt_peaks, all_pred_peaks, 1)
        x_fit = np.linspace(lims[0], lims[1], 100)
        ax.plot(x_fit, slope * x_fit + intercept, 'r-', linewidth=1.5,
                label=f'Fit: y = {slope:.3f}x + {intercept:.3f}')

        ax.set_xlabel('GT Peak Resultant (BW)')
        ax.set_ylabel('Predicted Peak Resultant (BW)')
        ax.set_title('Peak Prediction Scatter (all subjects)')
        ax.legend()
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        scatter_path = os.path.join(TEST_ROOT, f'peak_scatter{suffix}.png')
        plt.savefig(scatter_path, dpi=150, bbox_inches='tight')
        print(f"Peak scatter saved to {scatter_path}")
        plt.close()

        print(f"  Fit: slope={slope:.3f}, intercept={intercept:.3f}")
        print(f"  Mean ratio: {np.mean(all_pred_peaks/all_gt_peaks):.3f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default=None, choices=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm'])
    args = parser.parse_args()
    test(exp=args.exp)

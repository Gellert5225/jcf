"""
Evaluate trained JCF surrogate model on held-out test subjects.
Loads model + norm params, runs inference on jcf/testing/*,
compares predictions vs ground truth (SO+JR).
"""
import torch
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from train_jcf import JCFNet, build_dataset_from_subject


def evaluate():
    test_root = './jcf/testing'
    model_path = './jcf_model.pt'
    norm_path = './jcf_norm_params.npz'

    # Load normalization params
    norm = np.load(norm_path)
    X_mean, X_std = norm['mean'], norm['std']

    # Discover test subjects
    subject_dirs = sorted(
        glob.glob(os.path.join(test_root, 'moore_*'))
        + glob.glob(os.path.join(test_root, 'tiziana_*')),
        key=lambda p: os.path.basename(p).lower()
    )
    print(f"Test subjects: {len(subject_dirs)}")

    # Load test data per subject
    subjects = {}
    for sd in subject_dirs:
        name = os.path.basename(sd)
        result = build_dataset_from_subject(sd)
        if result is None:
            print(f"  {name}: SKIP (missing files)")
            continue
        inputs, targets = result
        subjects[name] = (inputs, targets)
        print(f"  {name}: {len(targets)} frames")

    if not subjects:
        print("No test data found!")
        return

    # Load model
    input_dim = X_mean.shape[0]
    model = JCFNet(input_dim=input_dim)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    # Run inference per subject
    axis_names = ['Fx (shear)', 'Fy (compressive)', 'Fz (shear)']
    n_subj = len(subjects)
    fig, axes = plt.subplots(n_subj, 4, figsize=(18, 4.5 * n_subj))
    if n_subj == 1:
        axes = axes[np.newaxis, :]

    print(f"\n{'Subject':<25} {'RMSE_Fx':>8} {'RMSE_Fy':>8} {'RMSE_Fz':>8} {'RMSE_|F|':>9} {'R2_|F|':>7}")
    print('-' * 75)

    for row, (name, (inputs, targets)) in enumerate(subjects.items()):
        # Normalize
        X_norm = (inputs - X_mean) / X_std
        X_t = torch.from_numpy(X_norm.astype(np.float32))

        with torch.no_grad():
            preds = model(X_t).numpy()

        # Per-axis RMSE
        rmses = [np.sqrt(np.mean((preds[:, i] - targets[:, i])**2)) for i in range(3)]
        # Resultant
        pred_res = np.sqrt(np.sum(preds**2, axis=1))
        true_res = np.sqrt(np.sum(targets**2, axis=1))
        rmse_res = np.sqrt(np.mean((pred_res - true_res)**2))
        ss_res = np.sum((pred_res - true_res)**2)
        ss_tot = np.sum((true_res - true_res.mean())**2)
        r2_res = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        print(f"{name:<25} {rmses[0]:8.3f} {rmses[1]:8.3f} {rmses[2]:8.3f} {rmse_res:9.3f} {r2_res:7.3f}")

        # Time axis (frame index)
        t = np.arange(len(targets))

        # Plot per-axis time series
        for i in range(3):
            ax = axes[row, i]
            ax.plot(t, targets[:, i], 'b-', label='Ground Truth', linewidth=1.5)
            ax.plot(t, preds[:, i], 'r--', label='Predicted', linewidth=1.5)
            ax.set_ylabel(f'{axis_names[i]} (BW)')
            ax.set_title(f'{name} — {axis_names[i]}  RMSE={rmses[i]:.3f}')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            if row == n_subj - 1:
                ax.set_xlabel('Frame')

        # Plot resultant
        ax = axes[row, 3]
        ax.plot(t, true_res, 'b-', label='Ground Truth', linewidth=1.5)
        ax.plot(t, pred_res, 'r--', label='Predicted', linewidth=1.5)
        ax.set_ylabel('|F| (BW)')
        ax.set_title(f'{name} — Resultant  RMSE={rmse_res:.3f}  R²={r2_res:.3f}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        if row == n_subj - 1:
            ax.set_xlabel('Frame')

    plt.tight_layout()
    plt.savefig('test_results.png', dpi=150)
    plt.show()
    print("\nSaved test_results.png")


if __name__ == '__main__':
    evaluate()

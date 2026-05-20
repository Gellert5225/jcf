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
from train_cnn import load_subject, JCF_CNN, JCF_CNN_v2, JCF_CNN_v2_causal, JCF_CNN_v3, JCF_Transformer, JCF_TCN, JCF_FFT_MLP

TEST_ROOT = "./jcf/full_duration/testing/walking"
TEST_ROOT_STATIC = "./jcf/full_duration/testing/static"
DEVICE = "cpu"


def test(exp=None, calibrate_peaks=0, calibrate_mode='trial', dataset=None, exclude=None, max_peak_bw=10.0, seed=None):
    seed_suffix = f"_s{seed}" if seed is not None else ""
    suffix = f"_{exp}{seed_suffix}" if exp else seed_suffix
    if calibrate_peaks > 0:
        cal_tag = f"_cal{calibrate_peaks}"
        if calibrate_mode == 'subject':
            cal_tag += 's'
        suffix += cal_tag
    model_name = f"best_model_{exp}{seed_suffix}.pt" if exp else f"best_model{seed_suffix}.pt"
    model_path = f"./jcf/full_duration/training/walking/{model_name}"

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=True)
    log_targets = checkpoint.get('log_targets', False)
    lower_body_only = checkpoint.get('lower_body_only', False)
    model_class = checkpoint.get('model_class', 'v1')
    jcf_subdir = checkpoint.get('jcf_subdir', 'jcf_output')
    clean_features = checkpoint.get('clean_features', False)
    include_mass = checkpoint.get('include_mass', False)
    use_root_features = checkpoint.get('use_root_features', False)
    combine_root_features = checkpoint.get('combine_root_features', False)
    lookahead = checkpoint.get('lookahead', 0)
    include_speed = checkpoint.get('include_speed', False)
    include_stance = checkpoint.get('include_stance', False)
    predict_moment = checkpoint.get('predict_moment', False)
    predict_flexion_moment = checkpoint.get('predict_flexion_moment', False)
    n_outputs = checkpoint.get('n_outputs', 3)
    n_features = checkpoint['n_features']
    input_mean = checkpoint['input_mean']
    input_std = checkpoint['input_std']

    if model_class == 'tcn':
        ModelClass = JCF_TCN
    elif model_class == 'fft_mlp':
        ModelClass = JCF_FFT_MLP
    elif model_class == 'transformer':
        ModelClass = JCF_Transformer
    elif model_class == 'v3':
        ModelClass = JCF_CNN_v3
    elif model_class == 'v2_causal':
        ModelClass = JCF_CNN_v2_causal
    elif model_class == 'v2':
        ModelClass = JCF_CNN_v2
    else:
        ModelClass = JCF_CNN
    model = ModelClass(n_features=n_features, n_outputs=n_outputs).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Loaded model from epoch {checkpoint['epoch']+1} "
          f"(val_loss={checkpoint['val_loss']:.6f})")
    print(f"Features: {n_features}, Device: {DEVICE}")
    if lower_body_only:
        print("Using lower-body joints only (0-19)")
    if clean_features:
        print("Clean features: pelvis translations zero-centered, dead joints removed")
    if include_mass:
        print("Body mass included as input feature")
    if use_root_features:
        print("Using root-frame features from .b3d (joint centers, root dynamics, GRF in root frame)")
    if combine_root_features:
        print("Combining IK features with root-frame dynamics from .b3d")
    if lookahead > 0:
        print(f"Lookahead: {lookahead} frames ({lookahead*10}ms latency at 100Hz)")
    if jcf_subdir != 'jcf_output':
        print(f"Using JCF labels from {jcf_subdir}/")

    excluded_set = set(exclude) if exclude else set()
    test_roots = [TEST_ROOT]
    if exp in ('n_d05_m_s', 'n_d05_m_s2') and os.path.isdir(TEST_ROOT_STATIC):
        test_roots.append(TEST_ROOT_STATIC)
    subject_dirs = []
    for root in test_roots:
        if not os.path.isdir(root):
            continue
        for name in sorted(os.listdir(root)):
            if dataset and not name.startswith(f"{dataset}_"):
                continue
            if any(name.startswith(f"{ex}_") for ex in excluded_set):
                continue
            subj_dir = os.path.join(root, name)
            jcf_sto = os.path.join(subj_dir, jcf_subdir,
                                   'BatchJCF_JointReaction_ReactionLoads.sto')
            if os.path.isdir(subj_dir) and os.path.exists(jcf_sto):
                tag = 'static' if '/static/' in subj_dir else 'walking'
                subject_dirs.append((f"{tag}/{name}", subj_dir))
    if dataset:
        print(f"Dataset filter: {dataset} only")
    if excluded_set:
        print(f"Excluded datasets: {sorted(excluded_set)}")

    print(f"\nFound {len(subject_dirs)} test subjects")

    if not subject_dirs:
        print("No test data found.")
        return

    # Cache of subject_key -> calibration scale (used in subject mode only)
    import re
    def _subject_key(name):
        # Strip trailing _t## and optional _r## so all trials of one subject share a key
        return re.sub(r'_t\d+(_r\d+)?$', '', name)

    calibration_scales = {}

    # When testing the static-aware exp, allow very short static trials through
    # load_subject's length filter so we can boundary-pad them at inference.
    # The CNN's bidirectional receptive field needs ~120 frames per side to fill
    # the conv windows; for trials shorter than that we replicate the first/last
    # frame to extend the input, run the forward pass, then slice predictions
    # back to the original frames. This matches the "pad with held posture"
    # inference pattern the static-aware model is designed for.
    RF_PAD = 120

    # Predict on each subject using sliding window, then average overlaps
    all_results = []
    for subj_name, subj_dir in subject_dirs:
        is_static_dir = '/static/' in subj_dir
        load_min_length = 1 if (is_static_dir and exp in ('n_d05_m_s', 'n_d05_m_s2')) else 100
        use_native_grid = (exp == 'n_d05_m_s2')
        result = load_subject(subj_dir, lower_body_only=lower_body_only,
                             jcf_subdir=jcf_subdir, clean_features=clean_features,
                             include_mass=include_mass,
                             use_root_features=use_root_features,
                             combine_root_features=combine_root_features,
                             max_peak_bw=max_peak_bw,
                             include_speed=include_speed,
                             include_stance=include_stance,
                             predict_moment=predict_moment,
                             predict_flexion_moment=predict_flexion_moment,
                             min_length=load_min_length,
                             native_time_grid=use_native_grid)
        if result is None:
            print(f"  {subj_name}: SKIP (load_subject returned None — missing files, "
                  f"peak > {max_peak_bw} BW, trial < {load_min_length} frames, or wrong GRF columns)")
            continue

        inputs, labels, mass = result
        T = len(labels)
        BW = mass * 9.81

        # Boundary-pad short trials so the conv receptive field is filled.
        pad_left = pad_right = 0
        if T < 2 * RF_PAD:
            pad_left = pad_right = RF_PAD
            inputs_for_model = np.concatenate([
                np.repeat(inputs[0:1], pad_left, axis=0),
                inputs,
                np.repeat(inputs[-1:], pad_right, axis=0),
            ], axis=0)
            print(f"  {subj_name}: T={T} → padded to {len(inputs_for_model)} "
                  f"(+{pad_left} edge frames each side)")
        else:
            inputs_for_model = inputs

        # Single forward pass on the full sequence (with normalization)
        with torch.no_grad():
            inp = torch.tensor(inputs_for_model, dtype=torch.float32)
            inp = (inp - input_mean) / input_std
            inp = inp.unsqueeze(0).to(DEVICE)
            out = model(inp)  # [1, T_padded, n_out]
            preds = out[0].cpu().numpy()
        if pad_left or pad_right:
            preds = preds[pad_left:pad_left + T]  # slice back to original frames

        # Inverse log-space transform if model was trained with log targets
        if log_targets:
            preds = np.sign(preds) * (np.exp(np.abs(preds)) - 1)

        # Apply lookahead shift: model's output[t] predicts label[t-N], so align
        # preds[N:] with labels[:T-N]. The last N label frames have no prediction.
        if lookahead > 0:
            if T <= lookahead + 30:
                print(f"  {subj_name}: SKIP (too short for lookahead={lookahead})")
                continue
            preds = preds[lookahead:]
            labels = labels[:-lookahead]
            T = T - lookahead

        # Subject-specific calibration: fit a scalar using the first N GT peaks.
        # Simulates measuring a few reference cycles per subject before deployment.
        # Mode 'trial': compute scale per trial. Mode 'subject': compute scale once
        # from the first trial of each subject and reuse for all later trials.
        scale = 1.0
        cal_end = 0
        if calibrate_peaks > 0:
            from scipy.signal import find_peaks as _find_peaks
            # Force resultant uses first 3 channels only (Fx, Fy, Fz). If Mx is
            # present (n_outputs=4), it's normalized differently and shouldn't
            # be mixed into the magnitude.
            gt_res = np.sqrt((labels[:, :3] ** 2).sum(axis=1))
            pred_res = np.sqrt((preds[:, :3] ** 2).sum(axis=1))

            need_compute = True
            if calibrate_mode == 'subject':
                subj_key = _subject_key(subj_name)
                if subj_key in calibration_scales:
                    scale = calibration_scales[subj_key]
                    need_compute = False  # reuse cached scale, don't mask any frames

            if need_compute:
                gt_peak_idx, _ = _find_peaks(gt_res, height=0.5, distance=15)
                if len(gt_peak_idx) >= calibrate_peaks:
                    cal_idx = gt_peak_idx[:calibrate_peaks]
                    ratios = gt_res[cal_idx] / (pred_res[cal_idx] + 1e-8)
                    scale = float(np.median(ratios))  # median: robust to single bad peak
                    cal_end = cal_idx[-1] + 15
                else:
                    scale = 1.0
                    cal_end = 0
                if calibrate_mode == 'subject':
                    calibration_scales[_subject_key(subj_name)] = scale

            preds = preds * scale

        valid = np.ones(T, dtype=bool)
        if calibrate_peaks > 0 and cal_end > 0:
            valid[:cal_end] = False  # don't score on calibration frames
        if valid.sum() < 30:
            print(f"  {subj_name}: SKIP (only {valid.sum()} frames after calibration)")
            continue

        # Metrics (in BW for forces, BW·m for moment Mx if present)
        errors = preds[valid] - labels[valid]
        mae = np.mean(np.abs(errors), axis=0)
        rmse = np.sqrt(np.mean(errors**2, axis=0))

        # Resultant force from first 3 channels only (Mx is on a different scale)
        gt_resultant = np.sqrt(np.sum(labels[valid, :3]**2, axis=1))
        pred_resultant = np.sqrt(np.sum(preds[valid, :3]**2, axis=1))
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
        if predict_moment and len(mae) >= 4:
            corr_mx = np.corrcoef(preds[valid, 3], labels[valid, 3])[0, 1]
            print(f"    Mx (norm by BW·H):    MAE={mae[3]:.4f}  RMSE={rmse[3]:.4f}  corr={corr_mx:.4f}")
        if predict_flexion_moment and len(mae) >= 5:
            corr_mz = np.corrcoef(preds[valid, 4], labels[valid, 4])[0, 1]
            print(f"    Mz (norm by BW·H):    MAE={mae[4]:.4f}  RMSE={rmse[4]:.4f}  corr={corr_mz:.4f}")
        print(f"    Resultant MAE:  {res_mae:.4f} BW   RMSE: {res_rmse:.4f} BW")
        print(f"    Fy correlation: {corr_fy:.4f}   Resultant correlation: {corr_res:.4f}")
        print(f"    Peak resultant:  GT={peak_gt:.3f} BW  Pred={peak_pred:.3f} BW")

        all_results.append({
            'name': subj_name,
            'dir': subj_dir,
            'T': T,
            'mass': mass,
            'labels': labels,
            'preds': preds,
            'valid': valid,
        })

    # ── Per-subject plots (saved in each subject's folder) ─────────────────
    for res in all_results:
        labels = res['labels']
        preds = res['preds']
        valid = res['valid']
        T = res['T']
        time = np.arange(T) * 0.01

        gt_res = np.sqrt(np.sum(labels[valid]**2, axis=1))
        pred_res = np.sqrt(np.sum(preds[valid]**2, axis=1))

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(time[valid], gt_res, 'b-', lw=1.5, label='GT')
        ax.plot(time[valid], pred_res, 'r--', lw=1.5, label='Pred')
        ax.set_ylabel('Resultant JCF (BW)')
        ax.set_xlabel('Time (s)')
        ax.set_title(f'{res["name"]} — Resultant')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig_path = os.path.join(res['dir'], f'prediction{suffix}.png')
        plt.savefig(fig_path, dpi=120)
        plt.close()

    print(f"\nPer-subject plots saved ({len(all_results)} subjects)")

    # ── Group results by activity (walking vs static) ─────────────────────────
    # When testing the static-aware exp, dump separate aggregates per activity
    # into their respective test roots. Otherwise everything is walking.
    groups = {TEST_ROOT: [r for r in all_results if '/static/' not in r['dir']]}
    static_results = [r for r in all_results if '/static/' in r['dir']]
    if static_results:
        groups[TEST_ROOT_STATIC] = static_results

    from scipy.signal import find_peaks

    for group_root, group_results in groups.items():
        if not group_results:
            continue
        is_static = (group_root == TEST_ROOT_STATIC)
        tag = 'static' if is_static else 'walking'

        # ── GT vs Predicted text dump ─────────────────────────────────────────
        txt_path = os.path.join(group_root, f'test_results{suffix}.txt')
        with open(txt_path, 'w') as f:
            for res in group_results:
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
                    gt_r = np.sqrt(np.sum(labels[t, :3]**2))
                    pred_r = np.sqrt(np.sum(preds[t, :3]**2))
                    f.write(f"{t:>6}  {time[t]:>8.3f}  "
                            f"{labels[t,0]:>10.5f}  {labels[t,1]:>10.5f}  {labels[t,2]:>10.5f}  {gt_r:>10.5f}  "
                            f"{preds[t,0]:>10.5f}  {preds[t,1]:>10.5f}  {preds[t,2]:>10.5f}  {pred_r:>10.5f}\n")
                f.write("\n")
        print(f"[{tag}] GT vs Predicted written to {txt_path}")

        # ── Scatter: peaks for walking, mean resultant for static ─────────────
        gt_pts, pred_pts = [], []
        for res in group_results:
            labels = res['labels']
            preds = res['preds']
            valid = res['valid']
            gt_res = np.sqrt(np.sum(labels[valid, :3]**2, axis=1))
            pred_res = np.sqrt(np.sum(preds[valid, :3]**2, axis=1))
            if is_static:
                # Static signals are nearly flat — use the mean per-trial.
                gt_pts.append(float(np.mean(gt_res)))
                pred_pts.append(float(np.mean(pred_res)))
            else:
                peaks, _ = find_peaks(gt_res, height=0.5, distance=15)
                for p in peaks:
                    gt_pts.append(gt_res[p])
                    pred_pts.append(pred_res[p])

        if not gt_pts:
            continue
        gt_pts = np.array(gt_pts)
        pred_pts = np.array(pred_pts)

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(gt_pts, pred_pts, alpha=0.7, edgecolors='k', linewidth=0.5)
        lims = [0, max(gt_pts.max(), pred_pts.max()) * 1.1]
        ax.plot(lims, lims, 'k--', linewidth=1, label='y = x')

        if len(gt_pts) >= 2:
            slope, intercept = np.polyfit(gt_pts, pred_pts, 1)
            x_fit = np.linspace(lims[0], lims[1], 100)
            ax.plot(x_fit, slope * x_fit + intercept, 'r-', linewidth=1.5,
                    label=f'Fit: y = {slope:.3f}x + {intercept:.3f}')
            print(f"[{tag}] Fit: slope={slope:.3f}, intercept={intercept:.3f}")

        metric_name = 'Mean Resultant (BW, per trial)' if is_static else 'Peak Resultant (BW)'
        ax.set_xlabel(f'GT {metric_name}')
        ax.set_ylabel(f'Predicted {metric_name}')
        ax.set_title(f'[{tag}] Prediction Scatter — {len(gt_pts)} points, {len(group_results)} subjects')
        ax.legend()
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        scatter_path = os.path.join(group_root, f'peak_scatter{suffix}.png')
        plt.savefig(scatter_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"[{tag}] Scatter saved to {scatter_path}")
        print(f"[{tag}] Mean ratio pred/GT: {np.mean(pred_pts/np.maximum(gt_pts, 1e-6)):.3f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default=None, choices=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'n3', 'n3s', 'n3_bin', 'n3_w15', 'n3_d05', 'n3_d05_w15', 'n3_d05_stance', 'n_d05', 'n_d05_m', 'n_d05_m2', 'n_d05_m_s', 'n_d05_m_s2'])
    parser.add_argument('--calibrate', type=int, default=0,
                        help='Use first N GT peaks to compute a per-subject scale factor. 0=disabled.')
    parser.add_argument('--calibrate-mode', type=str, default='trial',
                        choices=['trial', 'subject'],
                        help='trial: compute scale per trial. subject: compute once per subject from first trial.')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Restrict testing to one dataset prefix (e.g. carter). Default: all.')
    parser.add_argument('--exclude', type=str, nargs='+', default=None,
                        help='Dataset prefixes to exclude (e.g. --exclude han fregly).')
    parser.add_argument('--max-peak', type=float, default=10.0,
                        help='Reject test trials with JCF resultant peak > this many BW. Walking: 4.0 recommended.')
    parser.add_argument('--seed', type=int, default=None,
                        help='Seed of model to load (matches the --seed used during training).')
    args = parser.parse_args()
    test(exp=args.exp, calibrate_peaks=args.calibrate,
         calibrate_mode=args.calibrate_mode, dataset=args.dataset,
         exclude=args.exclude, max_peak_bw=args.max_peak, seed=args.seed)

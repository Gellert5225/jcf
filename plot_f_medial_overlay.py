"""
plot_f_medial_overlay.py

Diagnostic: take one walking trial, run the GRF-free MLP surrogate
(qinst_pelvisrel_winpos_w20) on it, compute F_medial from both the
predicted (F_y, M_x) and the OpenSim SO+JR ground-truth (F_y, M_x),
and overlay the two traces on one plot.

Goal: tell us whether the windowed MLP actually reproduces the
walking F_medial spike amplitude on training-distribution data, or
whether the regression-to-mean attenuation we documented for the
BiCNN also applies (and probably more strongly) to the GRF-free MLP.

Usage:
    conda run -n jcf python plot_f_medial_overlay.py \
        --trial ./jcf/full_duration/training/walking/carter_P003_split0_t04 \
        --ckpt ./jcf/full_duration/training/static/best_model_mlp_both_qinst_pelvisrel_winpos_w20.pt \
        --out /tmp/f_medial_overlay.png
"""
import argparse
import json
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from train_mlp import INPUT_SETS, JCF_MLP, _build_com_features, _windowize_positions_only


ALPHA = 0.5
D_M = 0.045


def f_medial(fy, mx, height_m):
    """F_medial = alpha * |F_y| + (M_x * H) / d_m  (all in BW)."""
    return ALPHA * np.abs(fy) + (mx * height_m) / D_M


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trial", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", default="/tmp/f_medial_overlay.png")
    args = p.parse_args()

    cfg_key = "qinst_pelvisrel_winpos_w20"
    cfg = INPUT_SETS[cfg_key]

    # Build inputs + GT labels for this trial.
    built = _build_com_features(args.trial, cfg,
                                max_peak_bw=6.0,
                                min_length=1,
                                predict_moment=True)
    if built is None:
        raise SystemExit(f"Failed to build features for {args.trial}")
    feat_per_frame, labels, mass = built     # feat: [T, 27], labels: [T, 4] in BW / BW*H
    feat_windowed = _windowize_positions_only(feat_per_frame, cfg["window"])
    T = feat_windowed.shape[0]

    # Subject height (needed for F_medial denormalization).
    meta = json.load(open(os.path.join(args.trial, "metadata.json")))
    height_m = float(meta.get("height_m", 1.7))

    # Time axis from raw IK.
    import pandas as pd
    ik = pd.read_csv(os.path.join(args.trial, "ik_results.mot"),
                     sep=r"\s+", skiprows=6)
    time_full = ik["time"].values
    # _build_com_features built features on a linspace over the full ik time range, length T.
    time = np.linspace(time_full[0], time_full[-1], T)

    # Load MLP checkpoint.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)
    assert ckpt["input_set"] == cfg_key, \
        f"checkpoint input_set={ckpt['input_set']}, expected {cfg_key}"

    model = JCF_MLP(
        n_features=ckpt["n_features"],
        n_outputs=ckpt.get("n_outputs", 4),
        hidden=ckpt.get("hidden", 128),
        dropout=0.0,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    input_mean = ckpt["input_mean"].to(device).float()
    input_std = ckpt["input_std"].to(device).float()

    x = torch.tensor(feat_windowed, dtype=torch.float32, device=device)
    x = (x - input_mean) / input_std
    with torch.no_grad():
        pred = model(x).cpu().numpy()        # [T, 4] in BW / BW*H

    fy_pred, mx_pred = pred[:, 1], pred[:, 3]
    fy_gt,   mx_gt   = labels[:, 1], labels[:, 3]

    fm_pred = f_medial(fy_pred, mx_pred, height_m)
    fm_gt   = f_medial(fy_gt,   mx_gt,   height_m)

    # Summary numbers.
    mae_fy = np.mean(np.abs(fy_pred - fy_gt))
    mae_mx = np.mean(np.abs(mx_pred - mx_gt))
    mae_fm = np.mean(np.abs(fm_pred - fm_gt))
    peak_pred = np.max(np.abs(fm_pred))
    peak_gt   = np.max(np.abs(fm_gt))

    print(f"Trial: {os.path.basename(args.trial)}")
    print(f"  T={T}, mass={mass:.1f} kg, height={height_m:.2f} m")
    print(f"  MAE F_y     = {mae_fy:.3f} BW")
    print(f"  MAE M_x     = {mae_mx:.4f} BW*H")
    print(f"  MAE F_medial= {mae_fm:.3f} BW")
    print(f"  peak |F_medial| GT  = {peak_gt:.3f} BW")
    print(f"  peak |F_medial| Pred= {peak_pred:.3f} BW")
    print(f"  peak attenuation Pred/GT = {peak_pred / peak_gt:.2f}")

    # Plot: 3 panels — F_y, M_x, F_medial.
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)

    axes[0].plot(time, fy_gt, label="GT  $F_y$ (SO+JR)", color="k", lw=1.5)
    axes[0].plot(time, fy_pred, label="MLP $\\hat F_y$", color="C3", lw=1.2)
    axes[0].set_ylabel("$F_y$ [BW]")
    axes[0].legend(loc="upper right")
    axes[0].grid(alpha=0.3)

    axes[1].plot(time, mx_gt, label="GT  $M_x$ (SO+JR)", color="k", lw=1.5)
    axes[1].plot(time, mx_pred, label="MLP $\\hat M_x$", color="C3", lw=1.2)
    axes[1].set_ylabel("$M_{add}$ [BW$\\cdot$H]")
    axes[1].legend(loc="upper right")
    axes[1].grid(alpha=0.3)

    axes[2].plot(time, fm_gt, label="GT  $F_{medial}$", color="k", lw=1.5)
    axes[2].plot(time, fm_pred, label="MLP $\\hat F_{medial}$", color="C3", lw=1.2)
    axes[2].set_ylabel("$F_{medial}$ [BW]")
    axes[2].set_xlabel("time [s]")
    axes[2].legend(loc="upper right")
    axes[2].grid(alpha=0.3)
    axes[2].axhline(1.8, color="gray", ls="--", lw=0.7, alpha=0.7,
                    label="$F_{pain}$ = 1.8 BW")

    fig.suptitle(
        f"{os.path.basename(args.trial)}  |  "
        f"mass={mass:.1f} kg, H={height_m:.2f} m  |  "
        f"MAE $F_{{medial}}$ = {mae_fm:.3f} BW  |  "
        f"peak Pred/GT = {peak_pred / peak_gt:.2f}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f"\nSaved overlay to {args.out}")


if __name__ == "__main__":
    main()

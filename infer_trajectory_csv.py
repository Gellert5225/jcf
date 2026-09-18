"""Run a qinst_pelvisrel_winpos* surrogate on an exported trajectory CSV.

Expected columns (100 Hz): time, q0..q36 (Rajagopal URDF order), com_position_{0,1,2},
pelvis_position_{0,1,2}, right/left_foot_position_{0,1,2}, right/left_foot_wrench_{0..5}.

The CSV is assumed to be in robot frame (x-forward, y-left, z-up) with q ordered
[tx, ty, tz, tilt, list, rot, ...]. The model expects OpenSim ground frame
(x-forward, y-up, z-right) and q ordered [tilt, list, rot, tx, ty, tz, ...].
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from inference_posture_surrogate import JCFPostureSurrogate

# Rajagopal2015 URDF q indices (revolute+prismatic only, in URDF order)
URDF_Q = {
    "pelvis_tx": 0, "pelvis_ty": 1, "pelvis_tz": 2,
    "pelvis_tilt": 3, "pelvis_list": 4, "pelvis_rotation": 5,
    "hip_flexion_r": 6, "hip_adduction_r": 7, "hip_rotation_r": 8,
    "knee_angle_r": 9, "ankle_angle_r": 10,
    "hip_flexion_l": 13, "hip_adduction_l": 14, "hip_rotation_l": 15,
    "knee_angle_l": 16, "ankle_angle_l": 17,
}
MODEL_JOINT_ORDER = [
    "pelvis_tilt", "pelvis_list", "pelvis_rotation",
    "pelvis_tx", "pelvis_ty", "pelvis_tz",
    "hip_flexion_r", "hip_adduction_r", "hip_rotation_r",
    "knee_angle_r", "ankle_angle_r",
    "hip_flexion_l", "hip_adduction_l", "hip_rotation_l",
    "knee_angle_l", "ankle_angle_l",
]
MODEL_Q_IDX = np.array([URDF_Q[n] for n in MODEL_JOINT_ORDER])

WINDOW = 20


def to_opensim(v):
    """(x, y, z) robot (y-left, z-up) → (x, y, z) OpenSim (y-up, z-right)."""
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    return np.stack([x, z, -y], axis=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="exported trajectory CSV")
    ap.add_argument(
        "--ckpt",
        default="./jcf/full_duration/training/static/best_model_mlp_both_qinst_pelvisrel_winpos_w20.pt",
    )
    ap.add_argument("--mass", type=float, default=70.0, help="kg")
    ap.add_argument("--height", type=float, default=1.75, help="m")
    ap.add_argument("--out", default="./f_medial_trajectory.png")
    args = ap.parse_args()

    df = pd.read_csv(args.csv).dropna(subset=["com_position_0"]).reset_index(drop=True)
    N = len(df)
    t = df["time"].to_numpy()

    q_full = df[[f"q{i}" for i in range(37)]].to_numpy()      # (N, 37) URDF order
    q16_raw = q_full[:, MODEL_Q_IDX].copy()                   # before rotation
    q16_rot = q_full[:, MODEL_Q_IDX].copy()
    q16_rot[:, 3:6] = to_opensim(q16_rot[:, 3:6])

    # Peek at the checkpoint to see which variant it is.
    # (Cheaper than constructing the model just to check the flags.)
    import torch as _torch
    _ckpt_peek = _torch.load(args.ckpt, map_location="cpu", weights_only=True)
    _input_set = str(_ckpt_peek.get("input_set", ""))
    need_window_pelvis_centering = _input_set.endswith("_winctr_w20")
    need_contact = _input_set == "qinst_pelvisrel_winpos_contact_w20"
    del _ckpt_peek

    # Contact flags from the foot wrenches. The 6-dim wrench is
    # [Mx, My, Mz, Fx, Fy, Fz] in the robot frame; vertical force is channel 2
    # (peaks ~760 N ≈ body weight). Contact = |Fz| above 5% BW.
    r_contact = l_contact = None
    if need_contact:
        bw_n = args.mass * 9.81
        thr = 0.05 * bw_n
        rfz = df["right_foot_wrench_2"].to_numpy().astype(float)
        lfz = df["left_foot_wrench_2"].to_numpy().astype(float)
        r_contact = (np.abs(rfz) > thr).astype(np.float32)
        l_contact = (np.abs(lfz) > thr).astype(np.float32)
        print(f"Contact variant: R in contact {r_contact.mean()*100:.0f}% of frames, "
              f"L {l_contact.mean()*100:.0f}% (threshold {thr:.0f} N)")

    if "pelvis_position_0" not in df.columns:
        raise SystemExit(
            f"{args.csv} does not have pelvis/foot position columns — use "
            "a CSV that includes pelvis_position_* and *_foot_position_* columns."
        )

    com_raw = df[[f"com_position_{i}"        for i in range(3)]].to_numpy()
    pel_raw = df[[f"pelvis_position_{i}"     for i in range(3)]].to_numpy()
    rf_raw  = df[[f"right_foot_position_{i}" for i in range(3)]].to_numpy()
    lf_raw  = df[[f"left_foot_position_{i}"  for i in range(3)]].to_numpy()

    com_rot, pel_rot = to_opensim(com_raw), to_opensim(pel_raw)
    rf_rot,  lf_rot  = to_opensim(rf_raw),  to_opensim(lf_raw)

    def rels(com, pel, rf, lf):
        return com - pel, rf - pel, lf - pel

    com_rel_raw, rf_rel_raw, lf_rel_raw = rels(com_raw, pel_raw, rf_raw, lf_raw)
    com_rel_rot, rf_rel_rot, lf_rel_rot = rels(com_rot, pel_rot, rf_rot, lf_rot)

    print(f"Loaded {N} frames, duration {t[-1] - t[0]:.2f}s\n")
    print(f"t=0 rfoot_rel (RAW  robot frame) = {rf_rel_raw[0]}")
    print(f"t=0 rfoot_rel (OpenSim, rotated)   = {rf_rel_rot[0]}  (expect ~(0, -0.95, +0.08))")

    n_win = N - WINDOW + 1
    def windowed(x):
        return np.stack([x[i:i+WINDOW] for i in range(n_win)])

    center = WINDOW // 2
    t_center = t[center : center + n_win]

    model = JCFPostureSurrogate(args.ckpt)
    print(f"\nLoaded checkpoint (epoch={model.epoch+1}, val_loss={model.val_loss:.4f}, "
          f"activation={model.activation})")

    def run(q16, com_rel, rf_rel, lf_rel):
        q_center = q16[center : center + n_win].copy()
        if need_window_pelvis_centering:
            # Subtract per-20-frame-window mean of pelvis_tx/ty/tz from the
            # center-frame value — matches the training-time _windowize_positions_only
            # subtraction for the *_winctr_* input_set.
            pelvis_win = np.stack([q16[i:i+WINDOW, 3:6] for i in range(n_win)])  # (n_win, 20, 3)
            q_center[:, 3:6] = q_center[:, 3:6] - pelvis_win.mean(axis=1)
        kw = {}
        if need_contact:
            kw["r_contact_window"] = windowed(r_contact)
            kw["l_contact_window"] = windowed(l_contact)
        return model.predict_windowed(
            joint_angles      = q_center,
            com_rel_window    = windowed(com_rel),
            calcn_r_rel_window= windowed(rf_rel),
            calcn_l_rel_window= windowed(lf_rel),
            mass_kg=args.mass, height_m=args.height,
            **kw,
        )

    pred_raw = run(q16_raw, com_rel_raw, rf_rel_raw, lf_rel_raw)
    pred_rot = run(q16_rot, com_rel_rot, rf_rel_rot, lf_rel_rot)

    def summarize(tag, p):
        print(f"  {tag:18s} |F_y| peak={np.abs(p['fy']).max():.3f}  "
              f"M_x peak={np.abs(p['mx']).max():.4f}  "
              f"F_med peak={np.abs(p['f_medial']).max():.3f} BW  "
              f"mean={p['f_medial'].mean():.3f}")

    print(f"\nSummary (mass={args.mass} kg, height={args.height} m):")
    summarize("RAW (no rotate)", pred_raw)
    summarize("OpenSim rotated", pred_rot)

    fig, ax = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
    ax[0].plot(t_center, pred_raw["f_medial"], label="F_medial — RAW robot frame", color="tab:orange", alpha=0.85)
    ax[0].plot(t_center, pred_rot["f_medial"], label="F_medial — OpenSim rotated", color="tab:blue")
    ax[0].set_ylabel("F_medial (BW)")
    ax[0].set_title(f"Surrogate F_medial — before vs after OpenSim conversion  "
                    f"(mass={args.mass} kg, h={args.height} m)")
    ax[0].grid(True); ax[0].legend()

    ax[1].plot(t_center, pred_raw["fy"], "--", color="tab:orange", alpha=0.6, label="F_y (raw)")
    ax[1].plot(t_center, pred_rot["fy"],        color="tab:blue", label="F_y (rotated)")
    ax[1].plot(t_center, pred_raw["mx"], ":", color="tab:orange", alpha=0.6, label="M_x (raw)")
    ax[1].plot(t_center, pred_rot["mx"],        color="tab:cyan", label="M_x (rotated)")
    ax[1].set_xlabel("time (s)")
    ax[1].set_ylabel("BW / BW*H")
    ax[1].grid(True); ax[1].legend(ncol=2)

    plt.tight_layout()
    plt.savefig(args.out, dpi=120)
    print(f"\nSaved comparison plot to {args.out}")


if __name__ == "__main__":
    main()

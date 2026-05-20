"""
Knee JCF Inference — n_d05_m_s static-aware model
=================================================

Self-contained inference module. Loads the trained bidirectional CNN
surrogate and predicts right-knee joint contact forces (Fx, Fy, Fz) and
external adduction moment (M_add) from joint angles + ground reaction
forces. The same predictor handles both walking and quasi-static postures
(see "Static posture inference" below).

Usage:
    from inference import JCFPredictor

    predictor = JCFPredictor("best_model_n_d05_m_s2_s2.pt")
    out = predictor.predict(joint_angles, grf, mass_kg=70.0, height_m=1.75)
    # out["fy"]: axial force in BW (compressive → negative)
    # out["mx"]: external KAM (M_add) in BW*Height units (medial-loading-positive)
    # out["f_medial"]: medial-compartment proxy in BW (Schipplein-Andriacchi)

Sign convention:
    out["mx"] is the EXTERNAL knee adduction moment under the convention
    that M_add > 0 corresponds to medial-compartment loading during
    right-leg stance. The Schipplein-Andriacchi formula

        F_medial = alpha * F_comp + M_add / d

    can therefore be evaluated by the controller without any sign flip
    on the surrogate's output. See predict() for the implementation.

Static posture inference:
    For a single held posture (no temporal context), replicate the
    feature vector across at least 240 frames (the bidirectional
    receptive field) and read out the prediction at the middle frame.
    See predict_static() helper below.

See INFERENCE_GUIDE.md for full documentation.
"""

import numpy as np
import torch
import torch.nn as nn

# ─── Model architecture (must match training) ────────────────────────────────

class _ResBlock1d(nn.Module):
    """Residual block with dilated conv + GroupNorm + dropout.

    Note: Dropout is always present in the structure (even at 0.0) so the saved
    state_dict layer indices match. At inference dropout is set to 0 anyway.
    """
    def __init__(self, channels, kernel_size=5, dilation=1, dropout=0.0):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            nn.GroupNorm(8, channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            nn.GroupNorm(8, channels),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.block(x))


class JCF_CNN_v2(nn.Module):
    """
    Bidirectional 1D CNN with residual blocks (dilations 1, 2, 4, 8).
    ~720K parameters. Outputs per-frame predictions.
    """
    def __init__(self, n_features=62, n_outputs=4, dropout=0.0):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Conv1d(n_features, 128, kernel_size=7, padding=3),
            nn.GroupNorm(8, 128),
            nn.ReLU(),
        )
        self.res_blocks = nn.Sequential(
            _ResBlock1d(128, kernel_size=5, dilation=1, dropout=dropout),
            _ResBlock1d(128, kernel_size=5, dilation=2, dropout=dropout),
            _ResBlock1d(128, kernel_size=5, dilation=4, dropout=dropout),
            _ResBlock1d(128, kernel_size=5, dilation=8, dropout=dropout),
        )
        self.head = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(64, n_outputs, kernel_size=1),
        )

    def forward(self, x):
        # x: [B, T, F]
        x = x.permute(0, 2, 1)        # → [B, F, T]
        x = self.input_proj(x)
        x = self.res_blocks(x)
        x = self.head(x)              # → [B, n_outputs, T]
        x = x.permute(0, 2, 1)        # → [B, T, n_outputs]
        return x


# ─── Preprocessing ────────────────────────────────────────────────────────────

# 16 lower-body DOFs after dropping dead joints (subtalar_r/l, mtp_r/l).
# Order MUST match training. pelvis_tx/ty/tz at indices 3, 4, 5.
JOINT_ORDER = [
    "pelvis_tilt", "pelvis_list", "pelvis_rotation",
    "pelvis_tx", "pelvis_ty", "pelvis_tz",
    "hip_flexion_r", "hip_adduction_r", "hip_rotation_r",
    "knee_angle_r", "ankle_angle_r",
    "hip_flexion_l", "hip_adduction_l", "hip_rotation_l",
    "knee_angle_l", "ankle_angle_l",
]
PELVIS_TX_IDX = [3, 4, 5]   # pelvis translation channels (will be zero-centered)
N_FEATURES = 62
N_OUTPUTS = 4
SAMPLING_HZ = 100


def preprocess(joint_angles, grf, mass_kg, dt=1.0 / SAMPLING_HZ,
               zero_center_pelvis=True, presmooth=False):
    """
    Build the 62-channel input tensor from raw IK and GRF.

    Args:
        joint_angles: [T, 16] np.ndarray, radians, in JOINT_ORDER above.
        grf:          [T, 6] np.ndarray, Newtons.
                      Columns: r_vx, r_vy, r_vz, l_vx, l_vy, l_vz.
        mass_kg:      scalar subject mass in kg.
        dt:           sampling period in seconds (default 0.01 = 100 Hz).
        zero_center_pelvis: subtract per-trial mean from pelvis_tx/ty/tz.
                            Set False only if your input is already zero-centered.
        presmooth:    apply a 2-tap moving average to IK and GRF before computing
                      derivatives. The training pipeline interpolates IK/GRF onto
                      a sub-sample-shifted JCF time grid, which acts as a mild
                      low-pass filter; presmooth=True replicates that filter so
                      the network sees inputs in the same statistical regime as
                      at training time. Set False to use raw inputs (slightly
                      higher noise in joint accelerations and GRF derivatives).

    Returns:
        inputs: [T, 62] np.float32
    """
    joint_angles = np.asarray(joint_angles, dtype=np.float64)
    grf = np.asarray(grf, dtype=np.float64)
    if joint_angles.shape[1] != 16:
        raise ValueError(f"joint_angles must have 16 columns, got {joint_angles.shape[1]}")
    if grf.shape[1] != 6:
        raise ValueError(f"grf must have 6 columns, got {grf.shape[1]}")
    if len(joint_angles) != len(grf):
        raise ValueError(f"Time mismatch: joint_angles {len(joint_angles)} vs grf {len(grf)}")
    if presmooth and len(joint_angles) >= 2:
        # 2-tap moving average: matches the smoothing introduced when the
        # training pipeline linearly interpolates IK onto a half-sample-shifted
        # JCF time grid. Same kernel applied symmetrically with edge padding.
        joint_angles = 0.5 * (joint_angles + np.concatenate(
            [joint_angles[:1], joint_angles[:-1]], axis=0))
        grf = 0.5 * (grf + np.concatenate([grf[:1], grf[:-1]], axis=0))

    BW = mass_kg * 9.81
    T = len(joint_angles)
    t = np.arange(T) * dt

    if zero_center_pelvis:
        joint_angles = joint_angles.copy()
        joint_angles[:, PELVIS_TX_IDX] -= joint_angles[:, PELVIS_TX_IDX].mean(axis=0)

    angle_vel = np.gradient(joint_angles, t, axis=0)
    angle_acc = np.gradient(angle_vel, t, axis=0)

    grf_norm = grf / BW
    grf_vel = np.gradient(grf_norm, t, axis=0)

    mass_chan = np.full((T, 1), mass_kg)

    pelvis_tx = joint_angles[:, 3]
    speed_raw = np.abs(np.gradient(pelvis_tx, t))
    win = min(100, T)
    kernel = np.ones(win) / win
    pelvis_speed = np.convolve(speed_raw, kernel, mode="same").reshape(-1, 1)

    inputs = np.hstack([
        joint_angles,       # 0:16
        angle_vel,          # 16:32
        angle_acc,          # 32:48
        grf_norm,           # 48:54
        grf_vel,            # 54:60
        mass_chan,          # 60
        pelvis_speed,       # 61
    ]).astype(np.float32)

    assert inputs.shape == (T, N_FEATURES), \
        f"Expected [T, {N_FEATURES}] but got {inputs.shape}"
    return inputs


# ─── Predictor wrapper ────────────────────────────────────────────────────────

class JCFPredictor:
    """
    Loads the trained static-aware JCF model and predicts forces + M_add
    from raw inputs. Handles both gait sequences and single static postures.

    Default constants for the medial compartment estimate (Schipplein-Andriacchi):
        alpha = 0.5  (baseline medial load fraction; calibrate per subject if available)
        d_m   = 0.045 m  (mediolateral compartment center spacing)
    """

    DEFAULT_ALPHA = 0.5
    DEFAULT_D_M = 0.045   # meters

    def __init__(self, checkpoint_path, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
        n_features = ckpt.get("n_features", N_FEATURES)
        n_outputs = ckpt.get("n_outputs", N_OUTPUTS)
        if n_features != N_FEATURES:
            raise ValueError(f"Checkpoint expects {n_features} features but inference.py expects {N_FEATURES}")
        if n_outputs != N_OUTPUTS:
            raise ValueError(f"Checkpoint has {n_outputs} outputs but inference.py expects {N_OUTPUTS}")

        self.model = JCF_CNN_v2(n_features=n_features, n_outputs=n_outputs, dropout=0.0)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(device).eval()

        self.input_mean = ckpt["input_mean"].to(device)
        self.input_std = ckpt["input_std"].to(device)
        self.epoch = ckpt.get("epoch", -1)
        self.val_loss = ckpt.get("val_loss", -1)

    @torch.no_grad()
    def predict(self, joint_angles, grf, mass_kg, height_m=None,
                alpha=None, d_m=None, dt=1.0 / SAMPLING_HZ):
        """
        Predict knee JCF + Mx from raw inputs.

        Args:
            joint_angles: [T, 16] in JOINT_ORDER
            grf:          [T, 6] in Newtons
            mass_kg:      scalar
            height_m:     optional. If given, also returns f_medial estimate.
            alpha, d_m:   optional medial compartment params (defaults: 0.5, 0.045).

        Returns:
            dict with keys:
                "fx", "fy", "fz":  [T] np.ndarray in BW
                "mx":              [T] np.ndarray, M_add in BW × Height (dimensionless),
                                   medial-loading-positive convention
                "fy_n":            [T] in Newtons (= fy * BW)
                "mx_nm":           [T] in N·m (= mx * BW * height_m), only if height_m given
                "f_medial":        [T] medial compartment force in BW, only if height_m given
                                   = alpha * |fy| + (mx * height_m) / d_m
        """
        raw = preprocess(joint_angles, grf, mass_kg, dt=dt)
        inputs = torch.tensor(raw, device=self.device)
        inputs = (inputs - self.input_mean) / self.input_std
        out = self.model(inputs.unsqueeze(0))[0].cpu().numpy()  # [T, 4]

        BW = mass_kg * 9.81
        result = {
            "fx": out[:, 0],
            "fy": out[:, 1],
            "fz": out[:, 2],
            "mx": out[:, 3],
            "fy_n": out[:, 1] * BW,
        }
        if height_m is not None:
            a = self.DEFAULT_ALPHA if alpha is None else alpha
            d = self.DEFAULT_D_M if d_m is None else d_m
            result["mx_nm"] = out[:, 3] * BW * height_m
            result["f_medial"] = a * np.abs(out[:, 1]) + (out[:, 3] * height_m) / d
        return result

    @torch.no_grad()
    def predict_static(self, joint_angles_static, grf_static, mass_kg,
                       height_m=None, alpha=None, d_m=None,
                       n_pad_frames=240, dt=1.0 / SAMPLING_HZ):
        """
        Predict knee JCF + M_add for a SINGLE held posture (no temporal context).

        Internally replicates the input across `n_pad_frames` frames so the
        bidirectional CNN's receptive field is filled, then returns the
        prediction at the middle frame.

        Args:
            joint_angles_static: [16] np.ndarray (single posture, JOINT_ORDER)
            grf_static:          [6]  np.ndarray (single GRF vector, Newtons)
            mass_kg, height_m, alpha, d_m: as in predict()
            n_pad_frames: total padded length. Default 240 covers the full
                          ~240-frame symmetric receptive field.

        Returns:
            dict with the SAME keys as predict() but each value is a SCALAR
            (not a [T] array) — the prediction at the central frame.
        """
        joint_angles_static = np.asarray(joint_angles_static, dtype=np.float64)
        grf_static = np.asarray(grf_static, dtype=np.float64)
        if joint_angles_static.shape != (16,):
            raise ValueError(f"joint_angles_static must be shape (16,), got {joint_angles_static.shape}")
        if grf_static.shape != (6,):
            raise ValueError(f"grf_static must be shape (6,), got {grf_static.shape}")

        ja_padded = np.tile(joint_angles_static[np.newaxis, :], (n_pad_frames, 1))
        grf_padded = np.tile(grf_static[np.newaxis, :], (n_pad_frames, 1))

        seq = self.predict(ja_padded, grf_padded, mass_kg, height_m=height_m,
                           alpha=alpha, d_m=d_m, dt=dt)
        mid = n_pad_frames // 2
        return {k: v[mid] for k, v in seq.items()}


# ─── Standalone test / example ────────────────────────────────────────────────

if __name__ == "__main__":
    """Smoke test: load model, run on synthetic input, print output shapes."""
    import sys
    if len(sys.argv) < 2:
        print("Usage: python inference.py <path_to_checkpoint.pt>")
        sys.exit(1)

    predictor = JCFPredictor(sys.argv[1])
    print(f"Loaded model (epoch={predictor.epoch+1}, val_loss={predictor.val_loss:.6f})")
    print(f"Device: {predictor.device}")

    # Synthetic example: 5 seconds of fake data at 100 Hz
    T = 500
    rng = np.random.default_rng(42)
    joint_angles = rng.normal(0, 0.5, (T, 16))
    grf = np.abs(rng.normal(400, 200, (T, 6)))   # Newtons
    mass_kg = 70.0
    height_m = 1.75

    out = predictor.predict(joint_angles, grf, mass_kg, height_m=height_m)
    print(f"Sequence outputs (T={T}):")
    print(f"  fy (BW):       range=[{out['fy'].min():.3f}, {out['fy'].max():.3f}]")
    print(f"  fy_N:          range=[{out['fy_n'].min():.0f}, {out['fy_n'].max():.0f}]")
    print(f"  mx (BW*H):     range=[{out['mx'].min():.4f}, {out['mx'].max():.4f}]")
    print(f"  mx_Nm:         range=[{out['mx_nm'].min():.1f}, {out['mx_nm'].max():.1f}]")
    print(f"  f_medial (BW): range=[{out['f_medial'].min():.3f}, {out['f_medial'].max():.3f}]")

    # Static posture inference example
    static_pose = rng.normal(0, 0.5, 16)
    static_grf = np.abs(rng.normal(400, 200, 6))
    static_out = predictor.predict_static(static_pose, static_grf,
                                          mass_kg, height_m=height_m)
    print(f"\nStatic posture (single frame, replicated to 240):")
    print(f"  fy (BW):       {static_out['fy']:.3f}")
    print(f"  mx (BW*H):     {static_out['mx']:.4f}")
    print(f"  f_medial (BW): {static_out['f_medial']:.3f}")

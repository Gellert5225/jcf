"""
Single-frame MLP Inference — JCF + Mx surrogate for posture optimization
==========================================================================

Self-contained. Predicts knee JCF (Fx, Fy, Fz) and adduction moment (Mx) from
a single frame of (joint angles, joint velocities, joint accelerations, GRF, mass).

Unlike the CNN model (inference.py), this one:
  - Has NO temporal context — single-frame input → single-frame output
  - Handles arbitrary postures including static (zero velocity / acceleration)
  - Suitable for posture optimization where you evaluate hypothetical configurations

Usage:
    from inference_mlp import JCFMLPPredictor

    predictor = JCFMLPPredictor("best_model_mlp.pt")

    # Single frame
    out = predictor.predict_frame(
        joint_angles_q,    # [16] in radians
        joint_vel_qdot,    # [16] in rad/s (or 0 for static)
        joint_acc_qddot,   # [16] in rad/s² (or 0 for static)
        grf,               # [6] in Newtons
        mass_kg=70.0,
        height_m=1.75,
    )
    # out["fy"], out["mx"], out["f_medial"]
"""

import numpy as np
import torch
import torch.nn as nn

# Same as the CNN inference module
JOINT_ORDER = [
    "pelvis_tilt", "pelvis_list", "pelvis_rotation",
    "pelvis_tx", "pelvis_ty", "pelvis_tz",
    "hip_flexion_r", "hip_adduction_r", "hip_rotation_r",
    "knee_angle_r", "ankle_angle_r",
    "hip_flexion_l", "hip_adduction_l", "hip_rotation_l",
    "knee_angle_l", "ankle_angle_l",
]
N_FEATURES = 62
N_OUTPUTS = 4


class JCF_MLP(nn.Module):
    """5-layer MLP, 256 hidden units, ~270K params. Identical to train_mlp.py."""
    def __init__(self, n_features=62, n_outputs=4, hidden=256, dropout=0.0):
        super().__init__()
        layers = []
        in_dim = n_features
        for _ in range(5):
            layers += [
                nn.Linear(in_dim, hidden),
                nn.LayerNorm(hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_dim = hidden
        layers.append(nn.Linear(hidden, n_outputs))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        in_shape = x.shape
        if x.ndim == 3:
            return self.net(x.reshape(-1, x.shape[-1])).reshape(*in_shape[:-1], -1)
        return self.net(x)


def build_feature_vector(joint_angles, joint_vel, joint_acc, grf, mass_kg,
                         pelvis_speed=0.0):
    """
    Construct the 62-channel input vector from raw single-frame measurements.

    Args:
        joint_angles: [16] in radians, in JOINT_ORDER. pelvis_tx/ty/tz must be
                      zero-centered (relative to a reference) — the static MLP
                      doesn't care about absolute lab position.
        joint_vel:    [16] in rad/s. Pass zeros for static posture.
        joint_acc:    [16] in rad/s². Pass zeros for static posture.
        grf:          [6] in Newtons (calcn_r xyz, calcn_l xyz).
        mass_kg:      scalar.
        pelvis_speed: smoothed |pelvis horizontal velocity| in m/s. Default 0.

    Returns:
        [62] np.float32 feature vector ready for predict_frame_normalized().
    """
    joint_angles = np.asarray(joint_angles, dtype=np.float32).flatten()
    joint_vel = np.asarray(joint_vel, dtype=np.float32).flatten()
    joint_acc = np.asarray(joint_acc, dtype=np.float32).flatten()
    grf = np.asarray(grf, dtype=np.float32).flatten()
    if joint_angles.shape != (16,) or joint_vel.shape != (16,) or joint_acc.shape != (16,):
        raise ValueError(f"Expected 16-element joint vectors, got "
                         f"{joint_angles.shape}, {joint_vel.shape}, {joint_acc.shape}")
    if grf.shape != (6,):
        raise ValueError(f"Expected 6-element GRF, got {grf.shape}")
    BW = mass_kg * 9.81
    grf_norm = grf / BW
    grf_vel = np.zeros(6, dtype=np.float32)   # static: zero GRF derivative
    return np.concatenate([
        joint_angles,        # 16
        joint_vel,           # 16
        joint_acc,           # 16
        grf_norm,            # 6
        grf_vel,             # 6
        [mass_kg],           # 1
        [pelvis_speed],      # 1
    ]).astype(np.float32)


class JCFMLPPredictor:
    """
    Loads the trained MLP and predicts knee JCF + Mx for individual frames or
    batches of frames.

    Defaults for the medial compartment estimate (Schipplein-Andriacchi):
        alpha = 0.5
        d_m   = 0.045 m
    """
    DEFAULT_ALPHA = 0.5
    DEFAULT_D_M = 0.045

    def __init__(self, checkpoint_path, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
        self.model = JCF_MLP(n_features=ckpt.get("n_features", N_FEATURES),
                             n_outputs=ckpt.get("n_outputs", N_OUTPUTS),
                             dropout=0.0)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(device).eval()

        self.input_mean = ckpt["input_mean"].to(device)
        self.input_std = ckpt["input_std"].to(device)
        self.epoch = ckpt.get("epoch", -1)
        self.val_loss = ckpt.get("val_loss", -1)

    @torch.no_grad()
    def predict_normalized(self, raw_features):
        """
        Run inference on raw (un-normalized) features.

        Args:
            raw_features: np.ndarray of shape [62] (single frame) or [B, 62] (batch).

        Returns:
            np.ndarray of shape [4] or [B, 4]. Channels: Fx, Fy, Fz (BW),
            Mx (BW × Height units).
        """
        x = torch.tensor(raw_features, dtype=torch.float32, device=self.device)
        single = (x.ndim == 1)
        if single:
            x = x.unsqueeze(0)
        x = (x - self.input_mean) / self.input_std
        out = self.model(x).cpu().numpy()
        return out[0] if single else out

    def predict_frame(self, joint_angles, joint_vel, joint_acc, grf, mass_kg,
                      height_m=None, alpha=None, d_m=None, pelvis_speed=0.0):
        """
        Predict knee JCF + Mx for a single posture.

        Args:
            joint_angles: [16] radians
            joint_vel:    [16] rad/s (zeros for static)
            joint_acc:    [16] rad/s² (zeros for static)
            grf:          [6] N (calcn_r xyz, calcn_l xyz)
            mass_kg:      scalar
            height_m:     scalar; if given, also returns f_medial
            alpha, d_m:   medial compartment params (defaults: 0.5, 0.045)
            pelvis_speed: smoothed |dpelvis_x/dt| in m/s; default 0

        Returns:
            dict with fx, fy, fz, mx (scalars). With height_m: also fy_n, mx_nm,
            f_medial.
        """
        feats = build_feature_vector(joint_angles, joint_vel, joint_acc, grf,
                                     mass_kg, pelvis_speed=pelvis_speed)
        out = self.predict_normalized(feats)   # [4]
        BW = mass_kg * 9.81
        result = {
            "fx": float(out[0]),
            "fy": float(out[1]),
            "fz": float(out[2]),
            "mx": float(out[3]),
        }
        if height_m is not None:
            a = self.DEFAULT_ALPHA if alpha is None else alpha
            d = self.DEFAULT_D_M if d_m is None else d_m
            result["fy_n"] = result["fy"] * BW
            result["mx_nm"] = result["mx"] * BW * height_m
            result["f_medial"] = a * abs(result["fy"]) + (result["mx"] * height_m) / d
        return result

    def predict_batch(self, joint_angles_b, joint_vel_b, joint_acc_b, grf_b,
                      mass_kg_b, height_m_b=None, alpha=None, d_m=None):
        """
        Batched prediction across multiple postures.

        All inputs prefixed _b: shape [B, ...] where B is batch size.
        Returns dict with arrays of shape [B] per channel.
        """
        feats = np.stack([
            build_feature_vector(joint_angles_b[i], joint_vel_b[i],
                                 joint_acc_b[i], grf_b[i], mass_kg_b[i])
            for i in range(len(joint_angles_b))
        ])
        out = self.predict_normalized(feats)   # [B, 4]
        BW = np.asarray(mass_kg_b) * 9.81
        result = {
            "fx": out[:, 0],
            "fy": out[:, 1],
            "fz": out[:, 2],
            "mx": out[:, 3],
        }
        if height_m_b is not None:
            a = self.DEFAULT_ALPHA if alpha is None else alpha
            d = self.DEFAULT_D_M if d_m is None else d_m
            H = np.asarray(height_m_b)
            result["fy_n"] = out[:, 1] * BW
            result["mx_nm"] = out[:, 3] * BW * H
            result["f_medial"] = a * np.abs(out[:, 1]) + (out[:, 3] * H) / d
        return result


# ─── Smoke test ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python inference_mlp.py <best_model_mlp.pt>")
        sys.exit(1)

    p = JCFMLPPredictor(sys.argv[1])
    print(f"Loaded MLP (epoch={p.epoch+1}, val_loss={p.val_loss:.6f})")
    print(f"Device: {p.device}")

    # Static neutral standing posture (joint angles ~0, GRF half BW each foot)
    q = np.zeros(16)
    qdot = np.zeros(16)        # static
    qddot = np.zeros(16)       # static
    mass = 70.0
    height = 1.75
    grf = np.array([0, mass * 9.81 / 2, 0,    # right foot vertical = half BW
                    0, mass * 9.81 / 2, 0])   # left foot vertical = half BW

    out = p.predict_frame(q, qdot, qddot, grf, mass, height_m=height)
    print(f"\nStatic neutral-posture prediction:")
    for k, v in out.items():
        print(f"  {k:<10} = {v:.4f}")

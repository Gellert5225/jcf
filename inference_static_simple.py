"""
Single-Posture Knee JCF Inference for Posture Optimization
==========================================================

Self-contained inference module for the (q, F, mass, height) MLP surrogate.
Designed for the multi-objective posture optimization use case: query one
candidate configuration at a time, get scalar predictions back. No temporal
buffering, no window replication, no torch DataLoader.

Usage:
    from inference_static_simple import JCFStaticPredictor

    predictor = JCFStaticPredictor("best_model_mlp_both_qfmh_forward.pt")
    out = predictor.predict(
        joint_angles=q,          # shape (16,) in JOINT_ORDER, radians
        grf=F,                   # shape (6,) in Newtons (r_vx,r_vy,r_vz,l_vx,l_vy,l_vz)
        mass_kg=70.0,
        height_m=1.75,
    )
    # out["fy"]:        axial knee force in BW (scalar)
    # out["mx"]:        external knee adduction moment in BW * H (scalar)
    # out["f_medial"]:  Schipplein-Andriacchi medial proxy in BW (scalar)

Batched usage (multiple postures at once — useful inside an optimizer loop):
    out = predictor.predict(
        joint_angles=q_batch,    # shape (N, 16)
        grf=F_batch,             # shape (N, 6)
        mass_kg=70.0,
        height_m=1.75,
    )
    # All outputs become shape (N,) instead of scalars.

Sign convention for `mx`: matches the external knee adduction moment used in
the Schipplein-Andriacchi formula. mx > 0 corresponds to medial compartment
loading during right-leg stance. No post-hoc negation needed at the controller.

Only depends on numpy + torch.
"""

import numpy as np
import torch
import torch.nn as nn


# ─── Model architecture (must match training) ────────────────────────────────

class _JCF_MLP(nn.Module):
    """Per-frame MLP. 3 hidden layers x 128 units, LayerNorm + ReLU + dropout.
    Identical structure to train_mlp.JCF_MLP so saved weights load directly.
    """
    def __init__(self, n_features, n_outputs=4, hidden=128, dropout=0.0):
        super().__init__()
        layers = []
        in_dim = n_features
        for _ in range(3):
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
        return self.net(x)


# ─── Constants matching training ──────────────────────────────────────────────

# The 16 lower-body DOFs in the order the model expects them. The user must
# supply joint_angles in this exact column order.
JOINT_ORDER = [
    "pelvis_tilt", "pelvis_list", "pelvis_rotation",
    "pelvis_tx", "pelvis_ty", "pelvis_tz",
    "hip_flexion_r", "hip_adduction_r", "hip_rotation_r",
    "knee_angle_r", "ankle_angle_r",
    "hip_flexion_l", "hip_adduction_l", "hip_rotation_l",
    "knee_angle_l", "ankle_angle_l",
]
PELVIS_TX_COLS = [3, 4, 5]   # pelvis translation channels — zero-center per-call

N_JOINTS = 16
N_GRF = 6
N_OUTPUTS = 4   # Fx, Fy, Fz, Mx (M_add)


# ─── Predictor ────────────────────────────────────────────────────────────────

class JCFStaticPredictor:
    """
    Loads the (q, F, mass, height) MLP surrogate and predicts knee JCF + M_add
    for single static postures. Stateless — safe to share across optimizer
    threads.

    Default Schipplein-Andriacchi constants for the medial compartment proxy:
        alpha = 0.5    (baseline medial fraction; raise for varus-aligned subjects)
        d_m   = 0.045  (mediolateral compartment-center spacing, meters)
    """

    DEFAULT_ALPHA = 0.5
    DEFAULT_D_M = 0.045   # meters

    def __init__(self, checkpoint_path, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)

        # Validate that this checkpoint is the (q, F, mass, height) variant.
        input_set = ckpt.get("input_set", None)
        if input_set != "qfmh":
            raise ValueError(
                f"Expected input_set='qfmh' (q, F, mass, height) but checkpoint "
                f"has input_set='{input_set}'. This predictor only supports the "
                f"qfmh variant. For other variants use inference.py."
            )

        self.n_features = ckpt["n_features"]   # 24 for qfmh
        if self.n_features != N_JOINTS + N_GRF + 2:
            raise ValueError(
                f"Expected n_features={N_JOINTS + N_GRF + 2} for qfmh, "
                f"got {self.n_features}"
            )

        self.model = _JCF_MLP(
            n_features=self.n_features,
            n_outputs=ckpt.get("n_outputs", N_OUTPUTS),
            hidden=ckpt.get("hidden", 128),
            dropout=0.0,
        )
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(device).eval()

        self.input_mean = ckpt["input_mean"].to(device).float()
        self.input_std = ckpt["input_std"].to(device).float()
        self.epoch = ckpt.get("epoch", -1)
        self.val_loss = ckpt.get("val_loss", -1.0)

    def _build_feature(self, joint_angles, grf, mass_kg, height_m):
        """Construct the 24-channel input tensor from raw inputs.

        Returns a torch tensor of shape (N, 24).
        """
        ja = np.asarray(joint_angles, dtype=np.float64)
        gr = np.asarray(grf, dtype=np.float64)

        # Promote 1-D single-posture inputs to a (1, .) batch
        if ja.ndim == 1:
            ja = ja[None, :]
        if gr.ndim == 1:
            gr = gr[None, :]
        if ja.shape[1] != N_JOINTS:
            raise ValueError(
                f"joint_angles must have {N_JOINTS} columns (one per JOINT_ORDER), "
                f"got {ja.shape[1]}"
            )
        if gr.shape[1] != N_GRF:
            raise ValueError(f"grf must have {N_GRF} columns, got {gr.shape[1]}")
        if len(ja) != len(gr):
            raise ValueError(
                f"Posture count mismatch: joint_angles {len(ja)} vs grf {len(gr)}"
            )

        # Zero-center pelvis translations (matches training-time clean_features).
        # For a single-frame query, this drives pelvis_tx/ty/tz to exactly zero,
        # which is the correct interpretation: the model never sees absolute
        # lab position, so we strip it at inference time too.
        ja = ja.copy()
        ja[:, PELVIS_TX_COLS] -= ja[:, PELVIS_TX_COLS].mean(axis=0, keepdims=True)

        # Normalize GRF by body weight (matches training).
        BW = mass_kg * 9.81
        grf_norm = gr / BW

        mass_col = np.full((len(ja), 1), mass_kg, dtype=np.float64)
        height_col = np.full((len(ja), 1), height_m, dtype=np.float64)

        feat = np.hstack([ja, grf_norm, mass_col, height_col]).astype(np.float32)
        assert feat.shape[1] == self.n_features, \
            f"Feature shape mismatch: got {feat.shape[1]}, expected {self.n_features}"
        return torch.tensor(feat, device=self.device)

    @torch.no_grad()
    def predict(self, joint_angles, grf, mass_kg, height_m,
                alpha=None, d_m=None):
        """
        Predict knee JCF + M_add for one (or N) static posture(s).

        Args:
            joint_angles: shape (16,) for a single posture, or (N, 16) for a batch.
                          Radians. Order = JOINT_ORDER (constant in this module).
            grf:          shape (6,) or (N, 6). Newtons.
                          Columns: r_vx, r_vy, r_vz, l_vx, l_vy, l_vz.
            mass_kg:      subject mass, scalar (single value for all postures).
            height_m:     subject height, scalar.
            alpha, d_m:   optional Schipplein-Andriacchi parameters. Defaults to
                          alpha=0.5, d_m=0.045 m.

        Returns:
            dict with keys (each scalar for single input, shape (N,) for batched):
                "fx", "fy", "fz": forces in BW
                "mx":             external KAM in BW * H (medial-loading-positive)
                "fy_n":           axial force in Newtons (= fy * BW)
                "mx_nm":          KAM in N*m (= mx * BW * height_m)
                "f_medial":       medial-compartment proxy in BW
                                  = alpha * |fy| + (mx * height_m) / d_m
        """
        single = np.asarray(joint_angles).ndim == 1

        x = self._build_feature(joint_angles, grf, mass_kg, height_m)
        x = (x - self.input_mean) / self.input_std
        out = self.model(x).cpu().numpy()   # shape (N, 4)

        BW = mass_kg * 9.81
        a = self.DEFAULT_ALPHA if alpha is None else alpha
        d = self.DEFAULT_D_M if d_m is None else d_m

        result = {
            "fx": out[:, 0],
            "fy": out[:, 1],
            "fz": out[:, 2],
            "mx": out[:, 3],
            "fy_n": out[:, 1] * BW,
            "mx_nm": out[:, 3] * BW * height_m,
            "f_medial": a * np.abs(out[:, 1]) + (out[:, 3] * height_m) / d,
        }

        if single:
            result = {k: float(v[0]) for k, v in result.items()}
        return result


# ─── Smoke test ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """Verify the module loads and runs on a synthetic posture."""
    import sys
    if len(sys.argv) < 2:
        print("Usage: python inference_static_simple.py <path_to_checkpoint.pt>")
        sys.exit(1)

    predictor = JCFStaticPredictor(sys.argv[1])
    print(f"Loaded MLP qfmh checkpoint (epoch={predictor.epoch+1}, "
          f"val_loss={predictor.val_loss:.6f})")
    print(f"Device: {predictor.device}, n_features={predictor.n_features}")

    # Single-posture prediction
    rng = np.random.default_rng(42)
    q = rng.normal(0, 0.3, N_JOINTS)
    F = np.abs(rng.normal(400, 200, N_GRF))   # Newtons
    mass_kg = 70.0
    height_m = 1.75

    out = predictor.predict(q, F, mass_kg, height_m)
    print(f"\nSingle-posture prediction:")
    print(f"  fy (BW):         {out['fy']:+.4f}")
    print(f"  fy_N:            {out['fy_n']:+.1f}")
    print(f"  mx (BW*H):       {out['mx']:+.5f}")
    print(f"  mx_Nm:           {out['mx_nm']:+.2f}")
    print(f"  f_medial (BW):   {out['f_medial']:+.4f}")

    # Batched prediction (e.g., many candidate postures inside an optimizer)
    N = 100
    q_batch = rng.normal(0, 0.3, (N, N_JOINTS))
    F_batch = np.abs(rng.normal(400, 200, (N, N_GRF)))
    out_batch = predictor.predict(q_batch, F_batch, mass_kg, height_m)
    print(f"\nBatched prediction (N={N}):")
    print(f"  fy (BW) range:     [{out_batch['fy'].min():+.3f}, {out_batch['fy'].max():+.3f}]")
    print(f"  f_medial (BW):     [{out_batch['f_medial'].min():+.3f}, {out_batch['f_medial'].max():+.3f}]")
    print(f"  output shape:       {out_batch['fy'].shape}")

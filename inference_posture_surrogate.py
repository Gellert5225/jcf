"""
Posture-Optimizer Knee JCF Inference (GRF-free, q-instantaneous variant)
========================================================================

Self-contained inference module for the
    (q_inst, [x_com - x_pelvis]_w, [x_r - x_pelvis]_w, [x_l - x_pelvis]_w,
     mass, height)
MLP surrogate (`qinst_pelvisrel_winpos_w20`).

This is the variant designed for William's posture-optimizer use case:
    - No GRF input (over-actuated sim has no real ground contact).
    - q is a single instantaneous snapshot (one posture).
    - Position features (COM and both feet, expressed as offsets from the
      pelvis in world frame) are taken over a 20-frame window centered on
      the current sample.

For a pure static query the position window is just the current snapshot
replicated 20 times — matches how the model saw quiet-stance trials during
training. For walking, pass the actual 20-frame trajectory ending at (or
centered on) the current frame.

Usage — static posture (one query):
    from inference_posture_surrogate import JCFPostureSurrogate

    predictor = JCFPostureSurrogate(
        "best_model_mlp_both_qinst_pelvisrel_winpos_w20.pt"
    )
    out = predictor.predict_static(
        joint_angles=q,             # (16,) in JOINT_ORDER, radians
        com_rel_pelvis=p_com,       # (3,) com - pelvis, meters, world frame
        calcn_r_rel_pelvis=p_r,     # (3,) right foot - pelvis, meters
        calcn_l_rel_pelvis=p_l,     # (3,) left foot - pelvis, meters
        mass_kg=70.0,
        height_m=1.75,
    )

Usage — walking with a position history (one query per call):
    out = predictor.predict_windowed(
        joint_angles=q,             # (16,)  current posture
        com_rel_window=W_com,       # (20, 3) com - pelvis over 20 frames
        calcn_r_rel_window=W_r,     # (20, 3)
        calcn_l_rel_window=W_l,     # (20, 3)
        mass_kg=70.0,
        height_m=1.75,
    )

Sign convention for mx: external knee adduction moment (KAM), normalized by
BW * H. mx > 0 corresponds to medial loading during right-stance.

Only depends on numpy + torch.
"""

import numpy as np
import torch
import torch.nn as nn


# ─── Model architecture (must match training) ────────────────────────────────

class _JCF_MLP(nn.Module):
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

JOINT_ORDER = [
    "pelvis_tilt", "pelvis_list", "pelvis_rotation",
    "pelvis_tx", "pelvis_ty", "pelvis_tz",
    "hip_flexion_r", "hip_adduction_r", "hip_rotation_r",
    "knee_angle_r", "ankle_angle_r",
    "hip_flexion_l", "hip_adduction_l", "hip_rotation_l",
    "knee_angle_l", "ankle_angle_l",
]
PELVIS_TX_COLS = [3, 4, 5]   # pelvis translation channels — zero-center per call

N_JOINTS = 16
N_POS_CH = 9       # rel_com (3) + rel_r (3) + rel_l (3) per frame
WINDOW = 20
N_OUTPUTS = 4      # Fx, Fy, Fz, Mx

# Feature layout per sample (198 channels):
#   [0    : 16 ]  q (instantaneous, center-frame)
#   [16   : 196]  windowed positions, shape (20, 9) flattened
#                 -> frame-major: f0_relcom(3), f0_relr(3), f0_rell(3),
#                                 f1_relcom(3), ... f19_rell(3)
#   [196  : 198]  mass, height (instantaneous)
N_FEATURES = N_JOINTS + WINDOW * N_POS_CH + 2   # 198


class JCFPostureSurrogate:
    """
    Loads the (q_inst, windowed positions, mass, height) MLP surrogate.
    Stateless; safe to share across optimizer threads.

    Default Schipplein-Andriacchi constants for the medial proxy:
        alpha = 0.5
        d_m   = 0.045 m
    """

    DEFAULT_ALPHA = 0.5
    DEFAULT_D_M = 0.045

    EXPECTED_INPUT_SET = "qinst_pelvisrel_winpos_w20"

    def __init__(self, checkpoint_path, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)

        input_set = ckpt.get("input_set", None)
        if input_set != self.EXPECTED_INPUT_SET:
            raise ValueError(
                f"Expected input_set='{self.EXPECTED_INPUT_SET}' but checkpoint "
                f"has input_set='{input_set}'. Use a different inference module "
                f"for that variant."
            )

        self.n_features = ckpt["n_features"]
        if self.n_features != N_FEATURES:
            raise ValueError(
                f"Expected n_features={N_FEATURES}, got {self.n_features}"
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

    # ─── Feature construction ────────────────────────────────────────────────

    def _build_feature_windowed(self, joint_angles, com_rel_w, r_rel_w, l_rel_w,
                                mass_kg, height_m):
        """
        Build the (N, 198) feature tensor.

        Shapes accepted (single-posture form, then batched form):
            joint_angles : (16,)        or (N, 16)
            com_rel_w    : (20, 3)      or (N, 20, 3)
            r_rel_w      : (20, 3)      or (N, 20, 3)
            l_rel_w      : (20, 3)      or (N, 20, 3)
        """
        ja = np.asarray(joint_angles, dtype=np.float64)
        cw = np.asarray(com_rel_w, dtype=np.float64)
        rw = np.asarray(r_rel_w, dtype=np.float64)
        lw = np.asarray(l_rel_w, dtype=np.float64)

        if ja.ndim == 1:
            ja = ja[None, :]
            cw = cw[None, ...]
            rw = rw[None, ...]
            lw = lw[None, ...]

        N = ja.shape[0]
        if ja.shape != (N, N_JOINTS):
            raise ValueError(f"joint_angles must be (16,) or (N, 16); got {ja.shape}")
        for name, arr in [("com_rel_w", cw), ("calcn_r_rel_w", rw), ("calcn_l_rel_w", lw)]:
            if arr.shape != (N, WINDOW, 3):
                raise ValueError(
                    f"{name} must be (20, 3) or (N, 20, 3); got {arr.shape}"
                )

        # Zero-center pelvis translations (matches training-time clean_features).
        # For a single posture this drives pelvis_tx/ty/tz to 0 — correct, since
        # the model never sees absolute lab position.
        ja = ja.copy()
        ja[:, PELVIS_TX_COLS] -= ja[:, PELVIS_TX_COLS].mean(axis=0, keepdims=True)

        # Stack positions per frame as (rel_com, rel_r, rel_l) → (N, 20, 9),
        # then flatten frame-major to (N, 180). Order matches _windowize_positions_only
        # in train_mlp.py: pos_windowed.reshape(T, window * n_pos).
        per_frame = np.concatenate([cw, rw, lw], axis=-1)         # (N, 20, 9)
        pos_flat = per_frame.reshape(N, WINDOW * N_POS_CH)        # (N, 180)

        mass_col = np.full((N, 1), mass_kg, dtype=np.float64)
        height_col = np.full((N, 1), height_m, dtype=np.float64)

        feat = np.hstack([ja, pos_flat, mass_col, height_col]).astype(np.float32)
        assert feat.shape[1] == self.n_features, \
            f"Feature width {feat.shape[1]} != expected {self.n_features}"
        return torch.tensor(feat, device=self.device)

    # ─── Prediction entry points ─────────────────────────────────────────────

    @torch.no_grad()
    def predict_windowed(self, joint_angles, com_rel_window,
                         calcn_r_rel_window, calcn_l_rel_window,
                         mass_kg, height_m, alpha=None, d_m=None):
        """
        Predict knee JCF + KAM for a posture with a 20-frame position history.

        joint_angles      : (16,) or (N, 16). Center-frame snapshot.
        com_rel_window    : (20, 3) or (N, 20, 3). com - pelvis, meters.
        calcn_r_rel_window: (20, 3) or (N, 20, 3). right foot - pelvis.
        calcn_l_rel_window: (20, 3) or (N, 20, 3). left foot - pelvis.
        mass_kg, height_m : scalars (one subject per call).

        Returns dict (scalars for single input, shape (N,) for batched):
            fx, fy, fz : forces in BW
            mx         : KAM in BW * H (medial-loading-positive)
            fy_n       : axial force in N
            mx_nm      : KAM in N*m
            f_medial   : Schipplein-Andriacchi medial proxy in BW
                         = alpha * |fy| + (mx * height_m) / d_m
        """
        single = np.asarray(joint_angles).ndim == 1
        x = self._build_feature_windowed(
            joint_angles, com_rel_window, calcn_r_rel_window, calcn_l_rel_window,
            mass_kg, height_m,
        )
        x = (x - self.input_mean) / self.input_std
        out = self.model(x).cpu().numpy()

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

    def predict_static(self, joint_angles, com_rel_pelvis,
                       calcn_r_rel_pelvis, calcn_l_rel_pelvis,
                       mass_kg, height_m, alpha=None, d_m=None):
        """
        Predict for a single static posture (no history). The position snapshot
        is replicated across the 20-frame window internally — matches how the
        model saw quiet-stance trials during training.

        joint_angles       : (16,) or (N, 16).
        com_rel_pelvis     : (3,)  or (N, 3). com - pelvis, world frame, meters.
        calcn_r_rel_pelvis : (3,)  or (N, 3).
        calcn_l_rel_pelvis : (3,)  or (N, 3).
        mass_kg, height_m  : scalars.

        Returns same dict as predict_windowed().
        """
        ja = np.asarray(joint_angles)
        pc = np.asarray(com_rel_pelvis, dtype=np.float64)
        pr = np.asarray(calcn_r_rel_pelvis, dtype=np.float64)
        pl = np.asarray(calcn_l_rel_pelvis, dtype=np.float64)

        if ja.ndim == 1:
            pc_w = np.broadcast_to(pc[None, :], (WINDOW, 3)).copy()
            pr_w = np.broadcast_to(pr[None, :], (WINDOW, 3)).copy()
            pl_w = np.broadcast_to(pl[None, :], (WINDOW, 3)).copy()
        else:
            N = ja.shape[0]
            pc_w = np.broadcast_to(pc[:, None, :], (N, WINDOW, 3)).copy()
            pr_w = np.broadcast_to(pr[:, None, :], (N, WINDOW, 3)).copy()
            pl_w = np.broadcast_to(pl[:, None, :], (N, WINDOW, 3)).copy()

        return self.predict_windowed(
            joint_angles, pc_w, pr_w, pl_w, mass_kg, height_m,
            alpha=alpha, d_m=d_m,
        )


# ─── Smoke test ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python inference_posture_surrogate.py <path_to_checkpoint.pt>")
        sys.exit(1)

    predictor = JCFPostureSurrogate(sys.argv[1])
    print(f"Loaded qinst_pelvisrel_winpos_w20 checkpoint "
          f"(epoch={predictor.epoch+1}, val_loss={predictor.val_loss:.6f})")
    print(f"Device: {predictor.device}, n_features={predictor.n_features}")

    rng = np.random.default_rng(42)
    q = rng.normal(0, 0.3, N_JOINTS)
    # Plausible relative positions (meters): COM ~0 above pelvis (well, com is
    # slightly above-ish, ~0.05 m for upright stance), feet ~1 m below pelvis.
    p_com = np.array([0.0, 0.05, 0.0])
    p_r = np.array([0.1, -0.95, 0.0])
    p_l = np.array([-0.1, -0.95, 0.0])
    mass_kg = 70.0
    height_m = 1.75

    out = predictor.predict_static(q, p_com, p_r, p_l, mass_kg, height_m)
    print(f"\nSingle static posture:")
    print(f"  fy (BW):         {out['fy']:+.4f}")
    print(f"  fy_N:            {out['fy_n']:+.1f}")
    print(f"  mx (BW*H):       {out['mx']:+.5f}")
    print(f"  mx_Nm:           {out['mx_nm']:+.2f}")
    print(f"  f_medial (BW):   {out['f_medial']:+.4f}")

    # Batched static (e.g. inside an optimizer loop)
    N = 100
    q_batch = rng.normal(0, 0.3, (N, N_JOINTS))
    pc_batch = rng.normal(0, 0.02, (N, 3)) + p_com
    pr_batch = rng.normal(0, 0.02, (N, 3)) + p_r
    pl_batch = rng.normal(0, 0.02, (N, 3)) + p_l
    out_b = predictor.predict_static(q_batch, pc_batch, pr_batch, pl_batch,
                                     mass_kg, height_m)
    print(f"\nBatched static (N={N}):")
    print(f"  fy (BW) range:     [{out_b['fy'].min():+.3f}, {out_b['fy'].max():+.3f}]")
    print(f"  f_medial (BW):     [{out_b['f_medial'].min():+.3f}, {out_b['f_medial'].max():+.3f}]")

    # Windowed prediction (walking with position history)
    com_w = np.broadcast_to(p_com[None, :], (WINDOW, 3)).copy()
    com_w += rng.normal(0, 0.01, com_w.shape)   # tiny variation
    r_w = np.broadcast_to(p_r[None, :], (WINDOW, 3)).copy()
    r_w += rng.normal(0, 0.05, r_w.shape)
    l_w = np.broadcast_to(p_l[None, :], (WINDOW, 3)).copy()
    l_w += rng.normal(0, 0.05, l_w.shape)
    out_w = predictor.predict_windowed(q, com_w, r_w, l_w, mass_kg, height_m)
    print(f"\nWalking (with 20-frame position window):")
    print(f"  fy (BW):         {out_w['fy']:+.4f}")
    print(f"  f_medial (BW):   {out_w['f_medial']:+.4f}")

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
    def __init__(self, n_features, n_outputs=4, hidden=128, dropout=0.0,
                 activation='relu'):
        super().__init__()
        act_map = {'relu': nn.ReLU, 'silu': nn.SiLU}
        if activation not in act_map:
            raise ValueError(f"activation must be one of {list(act_map)}; got {activation!r}")
        act_layer = act_map[activation]
        layers = []
        in_dim = n_features
        for _ in range(3):
            layers += [
                nn.Linear(in_dim, hidden),
                nn.LayerNorm(hidden),
                act_layer(),
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
N_POS_CH_CONTACT = 11   # + right-contact (1) + left-contact (1) for the contact variant
WINDOW = 20
N_OUTPUTS = 4      # Fx, Fy, Fz, Mx

# Feature layout per sample (198 channels, base variant):
#   [0    : 16 ]  q (instantaneous, center-frame)
#   [16   : 196]  windowed positions, shape (20, 9) flattened
#                 -> frame-major: f0_relcom(3), f0_relr(3), f0_rell(3),
#                                 f1_relcom(3), ... f19_rell(3)
#   [196  : 198]  mass, height (instantaneous)
# The contact variant windows 11 channels per frame (adds R/L contact flags),
# giving 16 + 20*11 + 2 = 238.
N_FEATURES = N_JOINTS + WINDOW * N_POS_CH + 2            # 198
N_FEATURES_CONTACT = N_JOINTS + WINDOW * N_POS_CH_CONTACT + 2   # 238


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

    # Base variants share a 198-channel feature shape; the contact variant adds
    # two windowed R/L contact flags for 238 channels.
    SUPPORTED_INPUT_SETS = (
        "qinst_pelvisrel_winpos_w20",          # per-trial centering
        "qinst_pelvisrel_winpos_winctr_w20",   # per 20-frame-window centering
        "qinst_pelvisrel_winpos_contact_w20",  # + windowed R/L contact flags
    )

    def __init__(self, checkpoint_path, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)

        input_set = ckpt.get("input_set", None)
        if input_set not in self.SUPPORTED_INPUT_SETS:
            raise ValueError(
                f"Expected input_set in {self.SUPPORTED_INPUT_SETS} but checkpoint "
                f"has input_set='{input_set}'. Use a different inference module "
                f"for that variant."
            )
        self.input_set = input_set
        self.pelvis_per_window_centering = input_set.endswith("_winctr_w20")
        self.include_contact = input_set == "qinst_pelvisrel_winpos_contact_w20"
        self.n_pos_ch = N_POS_CH_CONTACT if self.include_contact else N_POS_CH

        self.n_features = ckpt["n_features"]
        expected_features = N_FEATURES_CONTACT if self.include_contact else N_FEATURES
        if self.n_features != expected_features:
            raise ValueError(
                f"Expected n_features={expected_features} for input_set "
                f"'{input_set}', got {self.n_features}"
            )

        # Older checkpoints don't store 'activation'; assume ReLU for back-compat.
        self.activation = ckpt.get("activation", "relu")
        self.model = _JCF_MLP(
            n_features=self.n_features,
            n_outputs=ckpt.get("n_outputs", N_OUTPUTS),
            hidden=ckpt.get("hidden", 128),
            dropout=0.0,
            activation=self.activation,
        )
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(device).eval()

        self.input_mean = ckpt["input_mean"].to(device).float()
        self.input_std = ckpt["input_std"].to(device).float()
        self.epoch = ckpt.get("epoch", -1)
        self.val_loss = ckpt.get("val_loss", -1.0)

    # ─── Feature construction ────────────────────────────────────────────────

    def _build_feature_windowed(self, joint_angles, com_rel_w, r_rel_w, l_rel_w,
                                mass_kg, height_m, skip_pelvis_centering=False,
                                r_contact_w=None, l_contact_w=None):
        """
        Build the (N, 198) or (N, 238) feature tensor.

        Shapes accepted (single-posture form, then batched form):
            joint_angles : (16,)        or (N, 16)
            com_rel_w    : (20, 3)      or (N, 20, 3)
            r_rel_w      : (20, 3)      or (N, 20, 3)
            l_rel_w      : (20, 3)      or (N, 20, 3)
            r_contact_w  : (20,)        or (N, 20)   [contact variant only]
            l_contact_w  : (20,)        or (N, 20)   [contact variant only]

        skip_pelvis_centering: if True, the caller has already centered
            pelvis_tx/ty/tz in `joint_angles` and the internal batch-mean
            subtraction is skipped. Use this when you want per-window (or
            any other externally-defined) centering instead of per-batch.
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

        if self.include_contact:
            if r_contact_w is None or l_contact_w is None:
                raise ValueError(
                    "This is a contact-flag checkpoint; r_contact_window and "
                    "l_contact_window are required."
                )
            rc = np.asarray(r_contact_w, dtype=np.float64)
            lc = np.asarray(l_contact_w, dtype=np.float64)
            if rc.ndim == 1:
                rc = rc[None, :]
                lc = lc[None, :]
            for name, arr in [("r_contact_window", rc), ("l_contact_window", lc)]:
                if arr.shape != (N, WINDOW):
                    raise ValueError(
                        f"{name} must be (20,) or (N, 20); got {arr.shape}"
                    )

        # Zero-center pelvis translations (matches training-time clean_features).
        # For a single posture this drives pelvis_tx/ty/tz to 0 — correct, since
        # the model never sees absolute lab position. Caller can opt out via
        # skip_pelvis_centering=True if they've already centered externally
        # (e.g. per 20-frame window for sparse-Jacobian optimizers).
        #
        # For *_winctr_* checkpoints, the model was trained with per-window
        # centering applied at sample-construction time. Internal batch-mean
        # centering would double-center; the caller is responsible for
        # supplying per-window-centered pelvis_tx/ty/tz, so we always skip.
        ja = ja.copy()
        do_center = (not skip_pelvis_centering) and (not self.pelvis_per_window_centering)
        if do_center:
            ja[:, PELVIS_TX_COLS] -= ja[:, PELVIS_TX_COLS].mean(axis=0, keepdims=True)

        # Stack positions per frame as (rel_com, rel_r, rel_l[, Rc, Lc]) →
        # (N, 20, n_pos), then flatten frame-major. Order matches
        # _windowize_positions_only in train_mlp.py: reshape(T, window * n_pos),
        # and _build_com_features appends contact flags after rel_l.
        blocks = [cw, rw, lw]
        if self.include_contact:
            blocks += [rc[..., None], lc[..., None]]              # (N, 20, 1) each
        per_frame = np.concatenate(blocks, axis=-1)               # (N, 20, n_pos)
        pos_flat = per_frame.reshape(N, WINDOW * self.n_pos_ch)

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
                         mass_kg, height_m, alpha=None, d_m=None,
                         skip_pelvis_centering=False,
                         r_contact_window=None, l_contact_window=None):
        """
        Predict knee JCF + KAM for a posture with a 20-frame position history.

        joint_angles      : (16,) or (N, 16). Center-frame snapshot.
        com_rel_window    : (20, 3) or (N, 20, 3). com - pelvis, meters.
        calcn_r_rel_window: (20, 3) or (N, 20, 3). right foot - pelvis.
        calcn_l_rel_window: (20, 3) or (N, 20, 3). left foot - pelvis.
        mass_kg, height_m : scalars (one subject per call).
        skip_pelvis_centering: pass True if you've already centered
            joint_angles[..., 3:6] externally (e.g. per 20-frame window).
        r_contact_window, l_contact_window: (20,) or (N, 20) binary
            in-contact flags per frame. Required for the contact variant;
            ignored otherwise.

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
            skip_pelvis_centering=skip_pelvis_centering,
            r_contact_w=r_contact_window, l_contact_w=l_contact_window,
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
                       mass_kg, height_m, alpha=None, d_m=None,
                       r_contact=None, l_contact=None):
        """
        Predict for a single static posture (no history). The position snapshot
        is replicated across the 20-frame window internally — matches how the
        model saw quiet-stance trials during training.

        joint_angles       : (16,) or (N, 16).
        com_rel_pelvis     : (3,)  or (N, 3). com - pelvis, world frame, meters.
        calcn_r_rel_pelvis : (3,)  or (N, 3).
        calcn_l_rel_pelvis : (3,)  or (N, 3).
        mass_kg, height_m  : scalars.
        r_contact, l_contact: scalar (or (N,)) binary flags, replicated across
            the window. Required for the contact variant; a static posture is
            assumed to hold the same stance state across the window. Defaults to
            both feet in contact (1.0) if not given.

        Returns same dict as predict_windowed().
        """
        ja = np.asarray(joint_angles)
        pc = np.asarray(com_rel_pelvis, dtype=np.float64)
        pr = np.asarray(calcn_r_rel_pelvis, dtype=np.float64)
        pl = np.asarray(calcn_l_rel_pelvis, dtype=np.float64)

        rc_w = lc_w = None
        if ja.ndim == 1:
            pc_w = np.broadcast_to(pc[None, :], (WINDOW, 3)).copy()
            pr_w = np.broadcast_to(pr[None, :], (WINDOW, 3)).copy()
            pl_w = np.broadcast_to(pl[None, :], (WINDOW, 3)).copy()
            if self.include_contact:
                rc = 1.0 if r_contact is None else float(r_contact)
                lc = 1.0 if l_contact is None else float(l_contact)
                rc_w = np.full(WINDOW, rc, dtype=np.float64)
                lc_w = np.full(WINDOW, lc, dtype=np.float64)
        else:
            N = ja.shape[0]
            pc_w = np.broadcast_to(pc[:, None, :], (N, WINDOW, 3)).copy()
            pr_w = np.broadcast_to(pr[:, None, :], (N, WINDOW, 3)).copy()
            pl_w = np.broadcast_to(pl[:, None, :], (N, WINDOW, 3)).copy()
            if self.include_contact:
                rc = np.ones(N) if r_contact is None else np.asarray(r_contact, float)
                lc = np.ones(N) if l_contact is None else np.asarray(l_contact, float)
                rc_w = np.broadcast_to(rc[:, None], (N, WINDOW)).copy()
                lc_w = np.broadcast_to(lc[:, None], (N, WINDOW)).copy()

        return self.predict_windowed(
            joint_angles, pc_w, pr_w, pl_w, mass_kg, height_m,
            alpha=alpha, d_m=d_m,
            r_contact_window=rc_w, l_contact_window=lc_w,
        )


# ─── Smoke test ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python inference_posture_surrogate.py <path_to_checkpoint.pt>")
        sys.exit(1)

    predictor = JCFPostureSurrogate(sys.argv[1])
    print(f"Loaded {predictor.input_set} checkpoint "
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
    kw_c = {}
    if predictor.include_contact:
        kw_c = dict(r_contact_window=np.ones(WINDOW), l_contact_window=np.ones(WINDOW))
    out_w = predictor.predict_windowed(q, com_w, r_w, l_w, mass_kg, height_m, **kw_c)
    print(f"\nWalking (with 20-frame position window):")
    print(f"  fy (BW):         {out_w['fy']:+.4f}")
    print(f"  f_medial (BW):   {out_w['f_medial']:+.4f}")

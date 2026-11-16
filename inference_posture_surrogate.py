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

Foot point: x_r / x_l are a fixed point on each calcn body, `predictor.foot_point` in the calcn frame
(OpenSim axes, x forward, y up, z right; metres). It is the calcn origin, (0, 0, 0), for every checkpoint
except *_fp13_* (0.13, 0, -0.01), and *_fpopt_* (0.13, -0.01, 0), the optimizer's foot point, which is
(0.13, 0, -0.01) in the Rajagopal URDF's Z-up link axes. Pass positions of that point (see
docs/INFERENCE_GUIDE_MLP.md, "Foot point").

For a pure static query the position window is just the current snapshot
replicated 20 times — matches how the model saw quiet-stance trials during
training. For walking, pass the actual 20-frame trajectory ending at (or
centered on) the current frame.

Usage — static posture (one query):
    from inference_posture_surrogate import JCFPostureSurrogate

    predictor = JCFPostureSurrogate(
        "models/mlp-grffree-contact/mlp-grffree-contact.pt"
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

Sign convention for mx (Nov 2026): `mx` is the raw network output, the JointReaction moment of walker_knee_r about the
tibia x axis (on and in tibia_r) / (BW * H), as in the training labels. Physically the medial-positive (adduction)
moment is -mx at the right knee (analysis/thesis/knee_mx_sign.py, tests/test_knee_mx_sign.py). f_medial is
    alpha * |fy| + mx_medial_sign * mx * H / d,      mx_medial = mx_medial_sign * mx
with mx_medial_sign read from the checkpoint: -1 (physical) for the *_sojrqc checkpoints, which store it; +1 for
every older checkpoint, which does not, so f_medial of those is unchanged (the ISRR 2026 / pipeline formula, kept so
the optimizer results made with them stay reproducible). `predictor.mx_medial_sign` holds the value in use.

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



def remove_pelvis_drift(pelvis_xyz, fs_hz=100.0, window_s=1.2, fit_s=0.5):
    """Subtract a centred running mean from the pelvis translation along a trajectory.

    Removes the walking-distance drift of an overground trajectory while keeping the
    within-stride pelvis motion the models were trained on (treadmill data, per-trial
    centred, range ≈ ±0.1 m). The path is extended at constant velocity (fitted over the
    first/last `fit_s` seconds) by half a window at each end, so edge frames carry no
    residual drift. Use the result as q[:, 3:6] with skip_pelvis_centering=True.

    pelvis_xyz : (N, 3) pelvis_tx/ty/tz in the OpenSim frame, metres. Returns (N, 3).
    """
    x = np.asarray(pelvis_xyz, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 3:
        raise ValueError(f"pelvis_xyz must be (N, 3); got {x.shape}")
    N = len(x); L = max(2, int(round(window_s * fs_hz))); h = L // 2; k = max(2, min(N, int(round(fit_s * fs_hz))))
    i = np.arange(k, dtype=np.float64)
    s0 = np.polyfit(i, x[:k], 1)[0] if k >= 2 else np.zeros(3)
    s1 = np.polyfit(i, x[-k:], 1)[0] if k >= 2 else np.zeros(3)
    pre = x[0] + s0[None, :] * np.arange(-h, 0, dtype=np.float64)[:, None]
    post = x[-1] + s1[None, :] * np.arange(1, h + 1, dtype=np.float64)[:, None]
    ext = np.vstack([pre, x, post])
    cs = np.cumsum(np.vstack([np.zeros((1, 3)), ext]), axis=0)
    idx = np.arange(h, h + N)
    lo = idx - h; hi = np.minimum(idx + h + 1, len(ext))
    mean = (cs[hi] - cs[lo]) / (hi - lo)[:, None]
    return x - mean


# URDF link axes <-> OpenSim body axes: v_urdf = C_URDF @ v_opensim (x forward; URDF z up / y left, OpenSim y up / z right)
C_URDF = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])


def _rot(axis, ang):
    """(..., 3, 3) rotation matrices about coordinate axis 0/1/2 by angles (...)."""
    c, s = np.cos(ang), np.sin(ang)
    R = np.zeros(np.shape(ang) + (3, 3))
    i, j = [(1, 2), (2, 0), (0, 1)][axis]
    R[..., axis, axis] = 1.0
    R[..., i, i] = c; R[..., j, j] = c
    R[..., i, j] = -s; R[..., j, i] = s
    return R


def urdf_pelvis_to_opensim(pelvis_rot_urdf):
    """OpenSim pelvis_tilt / pelvis_list / pelvis_rotation from the Rajagopal2015.urdf pelvis rotation coordinates.

    assets/Rajagopal2015.urdf (and OpenSai, which reads it literally) rotates the pelvis about the URDF's own z (up),
    x (forward) and y (left) axes, in that order, for the joints it names pelvis_tilt, pelvis_list, pelvis_rotation.
    OpenSim's pelvis_tilt / list / rotation rotate about ground z (right), x, y. Copying the three values across, as
    the repo's converters used to, mis-orients the pelvis by up to ~13 deg on an optimizer trajectory (the copied
    'pelvis_tilt' turns about the vertical). This re-expresses the same orientation, R_opensim = C^T Rz(a) Rx(b) Ry(c) C,
    as OpenSim's body-fixed Z-X-Y angles. To first order: tilt = -c, list = b, rotation = a.

    pelvis_rot_urdf : (..., 3) the URDF coordinates [pelvis_tilt, pelvis_list, pelvis_rotation] (URDF q3, q4, q5), rad.
    Returns (..., 3) OpenSim [pelvis_tilt, pelvis_list, pelvis_rotation], rad. Pelvis translations are separate
    (rotate the 3-vector to OpenSim ground axes; see docs/INFERENCE_GUIDE_MLP.md, *Coordinate frame*).
    """
    a = np.asarray(pelvis_rot_urdf, dtype=np.float64)
    if a.shape[-1] != 3:
        raise ValueError(f"pelvis_rot_urdf must be (..., 3); got {a.shape}")
    Ru = _rot(2, a[..., 0]) @ _rot(0, a[..., 1]) @ _rot(1, a[..., 2])
    R = C_URDF.T @ Ru @ C_URDF
    # R = Rz(tilt) Rx(list) Ry(rotation): R[2,1] = sin(list), R[0,1] = -sin(tilt)cos(list), R[1,1] = cos(tilt)cos(list),
    # R[2,0] = -cos(list)sin(rotation), R[2,2] = cos(list)cos(rotation)
    lst = np.arcsin(np.clip(R[..., 2, 1], -1.0, 1.0))
    tilt = np.arctan2(-R[..., 0, 1], R[..., 1, 1])
    rot = np.arctan2(-R[..., 2, 0], R[..., 2, 2])
    return np.stack([tilt, lst, rot], axis=-1)


def opensim_pelvis_matrix(pelvis_ang):
    """(..., 3, 3) pelvis orientation in ground, R = Rz(tilt) Rx(list) Ry(rotation), from OpenSim
    [pelvis_tilt, pelvis_list, pelvis_rotation] (..., 3) in rad."""
    a = np.asarray(pelvis_ang, dtype=np.float64)
    return _rot(2, a[..., 0]) @ _rot(0, a[..., 1]) @ _rot(1, a[..., 2])


def opensim_pelvis_angles(R):
    """Inverse of opensim_pelvis_matrix: OpenSim [pelvis_tilt, pelvis_list, pelvis_rotation] (..., 3), rad."""
    lst = np.arcsin(np.clip(R[..., 2, 1], -1.0, 1.0))
    tilt = np.arctan2(-R[..., 0, 1], R[..., 1, 1])
    rot = np.arctan2(-R[..., 2, 0], R[..., 2, 2])
    return np.stack([tilt, lst, rot], axis=-1)


def pelvis_heading(pelvis_ang):
    """Heading of the pelvis (rad): yaw about ground y (up) of its forward (x) axis projected on the ground plane;
    0 = facing ground +x, positive = turned from +x toward -z (OpenSim's positive rotation about y)."""
    x = opensim_pelvis_matrix(pelvis_ang)[..., :, 0]
    return np.arctan2(-x[..., 2], x[..., 0])


def remove_heading(pelvis_ang, *vectors):
    """Rotate each sample about the vertical so its pelvis faces ground +x (the "_hd" input sets).

    The knee load does not depend on which way the subject faces in the lab, but the surrogate's inputs did: 75 % of
    the walking training trials (Moore, treadmill) face +90 deg in OpenSim ground and the rest ~0 deg, so the heading
    acted as a lab label and a 0-deg optimizer export was read as a different lab. The rotation uses the centre frame's
    pelvis heading (pelvis_heading), so a sample needs nothing outside its own inputs and turning is handled.

    pelvis_ang : (N, 3) OpenSim [pelvis_tilt, pelvis_list, pelvis_rotation] of the centre frame, rad.
    vectors    : arrays (N, ..., 3) in OpenSim ground axes (e.g. the (N, 20, 3) COM / foot windows relative to the
                 pelvis), each rotated by the sample's own heading.
    Returns (pelvis angles (N, 3) with heading 0, [rotated vectors]).
    """
    a = np.asarray(pelvis_ang, dtype=np.float64)
    Rm = _rot(1, -pelvis_heading(a))                                     # (N, 3, 3)
    ang = opensim_pelvis_angles(Rm @ opensim_pelvis_matrix(a))
    out = []
    for v in vectors:
        v = np.asarray(v, dtype=np.float64)
        Rb = Rm.reshape(Rm.shape[:1] + (1,) * (v.ndim - 2) + (3, 3))
        out.append((Rb @ v[..., None])[..., 0])
    return ang, out


def _rot_torch(axis, ang):
    """torch version of _rot: (..., 3, 3) rotations about coordinate axis 0/1/2, differentiable."""
    c, s = torch.cos(ang), torch.sin(ang)
    o, z = torch.ones_like(ang), torch.zeros_like(ang)
    rows = {0: [[o, z, z], [z, c, -s], [z, s, c]],
            1: [[c, z, s], [z, o, z], [-s, z, c]],
            2: [[c, -s, z], [s, c, z], [z, z, o]]}[axis]
    return torch.stack([torch.stack(r, dim=-1) for r in rows], dim=-2)


def remove_heading_torch(pelvis_ang, *vectors):
    """Differentiable torch version of remove_heading, for building the *_hd features inside an autodiff graph
    (*Getting gradients* in docs/INFERENCE_GUIDE_MLP.md). pelvis_ang (..., 3) OpenSim [tilt, list, rotation] (rad) of
    the centre frame; vectors (..., W, 3) or (..., 3) in OpenSim ground axes with the same leading shape.
    Returns (pelvis angles with heading 0, [rotated vectors]). Smooth unless the pelvis x axis points straight up."""
    a = pelvis_ang
    R = _rot_torch(2, a[..., 0]) @ _rot_torch(0, a[..., 1]) @ _rot_torch(1, a[..., 2])
    Rm = _rot_torch(1, -torch.atan2(-R[..., 2, 0], R[..., 0, 0]))       # undo the heading of the pelvis x axis
    R2 = Rm @ R
    ang = torch.stack([torch.atan2(-R2[..., 0, 1], R2[..., 1, 1]),
                       torch.asin(torch.clamp(R2[..., 2, 1], -1.0, 1.0)),
                       torch.atan2(-R2[..., 2, 0], R2[..., 2, 2])], dim=-1)
    out = []
    for v in vectors:
        extra = v.dim() - (Rm.dim() - 2) - 1                            # window axes between the batch and xyz
        Rb = Rm.reshape(Rm.shape[:-2] + (1,) * extra + (3, 3))
        out.append((Rb @ v.unsqueeze(-1)).squeeze(-1))
    return ang, out


class JCFPostureSurrogate:
    """
    Loads the (q_inst, windowed positions, mass, height) MLP surrogate.
    Stateless; safe to share across optimizer threads.

    Default Schipplein-Andriacchi constants for the medial proxy:
        alpha = 0.5
        d_m   = 0.045 m
    f_medial = alpha*|fy| + mx_medial_sign*mx*H/d_m, mx_medial_sign from the checkpoint (+1 when absent; see the
    module docstring).
    """

    DEFAULT_ALPHA = 0.5
    DEFAULT_D_M = 0.045

    # Base variants share a 198-channel feature shape; the contact variant adds
    # two windowed R/L contact flags for 238 channels.
    SUPPORTED_INPUT_SETS = (
        "qinst_pelvisrel_winpos_w20",          # per-trial centering
        "qinst_pelvisrel_winpos_winctr_w20",   # per 20-frame-window centering
        "qinst_pelvisrel_winpos_contact_w20",  # + windowed R/L contact flags
        "qinst_notrans_pelvisrel_winpos_contact_w20",  # contact, without pelvis_tx/ty/tz (235 ch)
        "qinst_notrans_pelvisrel_winpos_contact_fp13_w20",  # as above, feet at calcn (0.13, 0, -0.01)
        "qinst_notrans_pelvisrel_winpos_contact_fpopt_w20",  # as above, feet at calcn (0.13, -0.01, 0) = URDF (0.13, 0, -0.01)
        # Nov 2026 "_t100" sets: trained on exactly aligned features at 100 Hz, which is what this module builds
        "qinst_notrans_pelvisrel_winpos_contact_w20_t100",
        "qinst_notrans_pelvisrel_winpos_contact_fpopt_w20_t100",
        # same inputs as the two above, trained on static optimization + joint reaction labels (jcf_output_sojr/)
        "qinst_notrans_pelvisrel_winpos_contact_w20_t100_sojr",
        "qinst_notrans_pelvisrel_winpos_contact_fpopt_w20_t100_sojr",
        # as the two above, trained only on QC-passing SO+JR frames with the physical M_x sign (mx_medial_sign -1)
        "qinst_notrans_pelvisrel_winpos_contact_w20_t100_sojrqc",
        "qinst_notrans_pelvisrel_winpos_contact_fpopt_w20_t100_sojrqc",
        # as the two above with the heading removed per sample (remove_heading; done here, callers pass lab axes)
        "qinst_notrans_pelvisrel_winpos_contact_w20_t100_sojrqc_hd",
        "qinst_notrans_pelvisrel_winpos_contact_fpopt_w20_t100_sojrqc_hd",
    )
    LEGACY_MX_MEDIAL_SIGN = 1.0   # checkpoints without "mx_medial_sign" (all before Nov 2026)

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
        self.include_contact = "_contact_" in input_set
        # notrans: pelvis_tx/ty/tz are not model inputs. The caller still passes all 16
        # joint angles; the three translations are dropped here, so they have no effect
        # on the output and receive no gradient.
        self.drop_pelvis_translation = "_notrans_" in input_set
        # *_hd: each sample is rotated about the vertical so its centre-frame pelvis faces +x (remove_heading), as in
        # training; the caller keeps passing lab-axis inputs, whichever way the subject faces
        self.heading_norm = bool(ckpt.get("heading_norm", input_set.endswith("_hd")))
        # Point on each calcn body whose position the foot channels expect (calcn frame, m).
        # Checkpoints from before foot-point variants used the calcn origin.
        self.foot_point = np.asarray(ckpt.get("foot_point", (0.0, 0.0, 0.0)), dtype=np.float64)
        # Factor on the raw right-knee mx in f_medial: what the checkpoint's F_medial loss used. Older checkpoints do
        # not store it and keep the legacy +1, so their f_medial is unchanged.
        self.mx_medial_sign = float(ckpt.get("mx_medial_sign", self.LEGACY_MX_MEDIAL_SIGN))
        if self.mx_medial_sign not in (1.0, -1.0):
            raise ValueError(f"mx_medial_sign must be +1 or -1, checkpoint has {self.mx_medial_sign}")
        self.n_pos_ch = N_POS_CH_CONTACT if self.include_contact else N_POS_CH

        self.n_features = ckpt["n_features"]
        expected_features = N_FEATURES_CONTACT if self.include_contact else N_FEATURES
        if self.drop_pelvis_translation:
            expected_features -= len(PELVIS_TX_COLS)
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

        if self.drop_pelvis_translation:
            ja = np.delete(ja, PELVIS_TX_COLS, axis=1)             # (N, 13)
        feat = np.hstack([ja, pos_flat, mass_col, height_col]).astype(np.float32)
        if self.heading_norm:
            feat = self._remove_heading_features(feat)
        assert feat.shape[1] == self.n_features, \
            f"Feature width {feat.shape[1]} != expected {self.n_features}"
        return torch.tensor(feat, device=self.device)

    def _remove_heading_features(self, feat):
        """Rotate each feature row so its centre-frame pelvis faces +x: pelvis angles (columns 0:3) and the COM / foot
        3-vectors of all 20 window frames (the same transform as train_mlp._windowize_positions_only(heading_norm))."""
        q_dim = feat.shape[1] - WINDOW * self.n_pos_ch - 2
        win = feat[:, q_dim:q_dim + WINDOW * self.n_pos_ch].reshape(len(feat), WINDOW, self.n_pos_ch)
        ang, vecs = remove_heading(feat[:, 0:3], *(win[:, :, k:k + 3] for k in (0, 3, 6)))
        out = feat.copy()
        out[:, 0:3] = ang
        win = win.copy()
        for k, v in zip((0, 3, 6), vecs):
            win[:, :, k:k + 3] = v
        out[:, q_dim:q_dim + WINDOW * self.n_pos_ch] = win.reshape(len(feat), -1)
        return out.astype(np.float32)

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
            mx         : raw right-knee JR moment about the tibia x axis, BW * H (network output;
                         physically medial-positive is -mx)
            mx_medial  : mx_medial_sign * mx, the moment term f_medial uses, BW * H
            fy_n       : axial force in N
            mx_nm      : mx in N*m
            f_medial   : Schipplein-Andriacchi medial proxy in BW
                         = alpha * |fy| + (mx_medial_sign * mx * height_m) / d_m
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
            "mx_medial": self.mx_medial_sign * out[:, 3],
            "f_medial": a * np.abs(out[:, 1]) + (self.mx_medial_sign * out[:, 3] * height_m) / d,
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
    print(f"Device: {predictor.device}, n_features={predictor.n_features}, mx_medial_sign={predictor.mx_medial_sign:+g}")

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

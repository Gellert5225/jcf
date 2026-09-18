"""
Train a single-frame MLP surrogate for knee JCF prediction.

Designed for static / quasi-static posture optimization where the bidirectional
CNN can't be used (no temporal context, single-frame query).

Uses the same input features as the CNN (62 channels) but maps each frame
independently → single-frame output. No temporal convolution, no receptive-field
warm-up.

Usage:
    # Static surrogate (default — for posture optimization)
    conda run -n jcf python train_mlp.py
    conda run -n jcf python train_mlp.py --activity static --exclude han fregly

    # Walking-frame surrogate (single-frame variant of the CNN walking model)
    conda run -n jcf python train_mlp.py --activity walking --max-peak 6.0 --exclude han fregly

    # Combined static + walking (uses oversampling so static gets equal weight)
    conda run -n jcf python train_mlp.py --activity both
"""
import os
import re
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split

from train_cnn import load_subject, filter_flat_subjects, SEED, TRAIN_SPLIT

DATA_ROOTS = {
    "static":  "./jcf/full_duration/training/static",
    "walking": "./jcf/full_duration/training/walking",
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4096
EPOCHS = 50
LEARNING_RATE = 1e-3
N_OUTPUTS = 4      # Fx, Fy, Fz, Mx
HIDDEN = 128       # smaller model to combat overfitting on the small static set
DROPOUT = 0.3      # stronger regularization
WEIGHT_DECAY = 1e-3

# Channel layout of load_subject output (lower_body_only=True, clean_features=True,
# include_mass=True, include_speed=True): 62 channels in order
#   0-15   joint angles (q)
#   16-31  joint velocities (dq)        — central-difference by default
#   32-47  joint accelerations (ddq)
#   48-53  GRF per foot (Fx_r, Fy_r, Fz_r, Fx_l, Fy_l, Fz_l) in BW
#   54-59  GRF derivative (dF)
#   60     body mass (kg)
#   61     smoothed pelvis horizontal speed (m/s)
# FrameDataset optionally appends:
#   62     subject height (m)
#   63     BMI (kg/m^2)

INPUT_SETS = {
    "full": {
        "base_indices": list(range(62)),
        "include_height": True,
        "include_bmi": True,
        "uses_com": False,
        "window": 1,
        "desc": "full (q, dq, ddq, F, dF, mass, speed, height, BMI) — 64 ch",
    },
    "qdqf": {
        "base_indices": list(range(16)) + list(range(16, 32)) + list(range(48, 54)),
        "include_height": False,
        "include_bmi": False,
        "uses_com": False,
        "window": 1,
        "desc": "(q, dq, F) — 38 ch",
    },
    "qdqfmh": {
        "base_indices": list(range(16)) + list(range(16, 32)) + list(range(48, 54)) + [60],
        "include_height": True,
        "include_bmi": False,
        "uses_com": False,
        "window": 1,
        "desc": "(q, dq, F, mass, height) — 40 ch",
    },
    "qf": {
        "base_indices": list(range(16)) + list(range(48, 54)),
        "include_height": False,
        "include_bmi": False,
        "uses_com": False,
        "window": 1,
        "desc": "(q, F) — 22 ch",
    },
    "qfmh": {
        "base_indices": list(range(16)) + list(range(48, 54)) + [60],
        "include_height": True,
        "include_bmi": False,
        "uses_com": False,
        "window": 1,
        "desc": "(q, F, mass, height) — 24 ch",
    },
    # William's GRF-free variants. Features are constructed differently
    # (load_subject's GRF channels are NOT used). Per-frame layout:
    #    0-15   q (16 lower-body joint angles, clean_features)
    #    16-18  v_com (COM velocity x, y, z in world frame)
    #    19-21  x_com - x_calcn_r (right foot relative position)
    #    22-24  x_com - x_calcn_l (left foot relative position)
    #    [25   mass (kg), if include_height]
    #    [26   height (m), if include_height]
    # _w20 variant additionally flattens a 20-frame temporal window around
    # each frame (center frame is the prediction target).
    "qvcomrel": {
        "base_indices": None,           # custom assembly, not slicing
        "include_height": True,         # adds mass and height (treated as a pair)
        "include_bmi": False,
        "uses_com": True,
        "window": 1,
        "per_frame_ch": 27,
        "desc": "(q, v_com, x_com - x_r, x_com - x_l, mass, height) — 27 ch, single frame",
    },
    "qvcomrel_w20": {
        "base_indices": None,
        "include_height": True,
        "include_bmi": False,
        "uses_com": True,
        "window": 20,
        "per_frame_ch": 27,
        "desc": "(q, v_com, x_com - x_r, x_com - x_l, mass, height) — 27 ch × 20-frame window = 540 ch",
    },
    # William's variant: q is the CURRENT (center-frame) posture replicated across
    # the entire 20-frame window. Position features are pelvis-relative: COM and
    # both feet expressed as offsets from the pelvis world position. Pelvis world
    # position is read from raw IK (pelvis_tx/ty/tz BEFORE clean_features
    # zero-centering). Closer to a posture-optimizer setup where the optimizer
    # picks one q and queries the surrogate.
    "qinst_pelvisrel_w20": {
        "base_indices": None,
        "include_height": True,
        "include_bmi": False,
        "uses_com": True,
        "uses_pelvis_rel": True,
        "replicate_q": True,
        "window": 20,
        "per_frame_ch": 27,
        "desc": "(q_inst, x_com - x_pelvis, x_r - x_pelvis, x_l - x_pelvis, mass, height) — 27 ch × 20-frame window = 540 ch; q replicated across window",
    },
    # Variant where q is a TRUE single-frame snapshot (not in any window),
    # and only the position-relative features are windowed. q + (mass, height)
    # appear once per sample; the 9 relative-position channels are windowed
    # into 60 channels each. Smaller per-sample footprint (198 ch) and matches
    # the posture-optimizer setting more cleanly: optimizer picks one q,
    # surrogate uses the trajectory of COM and feet up to that posture.
    "qinst_pelvisrel_winpos_w20": {
        "base_indices": None,
        "include_height": True,
        "include_bmi": False,
        "uses_com": True,
        "uses_pelvis_rel": True,
        "replicate_q": False,
        "window_only_positions": True,
        "window": 20,
        "per_frame_ch": 27,
        "n_features_total": 16 + 20 * 9 + 2,    # 198: q + window*9 positions + mass + height
        "desc": "(q_inst (1×), [x_com - x_pelvis windowed], [x_r - x_pelvis windowed], [x_l - x_pelvis windowed], mass, height) — q and (mass,height) instantaneous, positions in 20-frame window = 198 ch total",
    },
    # Same as qinst_pelvisrel_winpos_w20, but pelvis_tx/ty/tz in q are centered
    # per 20-frame window (mean over those 20 frames subtracted from the center
    # frame's value) instead of per-trial. Designed for trajectory optimizers
    # (e.g. William's ipopt loop) where per-trial centering creates a dense
    # Jacobian — per-window centering keeps it block-diagonal across windows.
    "qinst_pelvisrel_winpos_winctr_w20": {
        "base_indices": None,
        "include_height": True,
        "include_bmi": False,
        "uses_com": True,
        "uses_pelvis_rel": True,
        "replicate_q": False,
        "window_only_positions": True,
        "pelvis_per_window_centering": True,
        "window": 20,
        "per_frame_ch": 27,
        "n_features_total": 16 + 20 * 9 + 2,
        "desc": "Same as qinst_pelvisrel_winpos_w20 but pelvis_tx/ty/tz centered per 20-frame window (sparse-Jacobian friendly)",
    },
    # Same as qinst_pelvisrel_winpos_w20, plus two binary contact flags
    # (right-foot, left-foot) appended to the windowed per-frame block. The
    # flags are derived from the vertical GRF at train time and let the model
    # distinguish single- vs dual-stance — the regime where kinematics alone
    # cannot resolve inter-limb load sharing (e.g. the late-stance medial-force
    # spike entering double support). Consistent with the paper's windowing of
    # per-foot geometric features.
    "qinst_pelvisrel_winpos_contact_w20": {
        "base_indices": None,
        "include_height": True,
        "include_bmi": False,
        "uses_com": True,
        "uses_pelvis_rel": True,
        "replicate_q": False,
        "window_only_positions": True,
        "include_contact": True,
        "window": 20,
        "per_frame_ch": 29,                     # q(16) + pos(9) + contact(2) + mass,height(2)
        "n_pos": 11,                            # rel_com(3)+rel_r(3)+rel_l(3)+Rc(1)+Lc(1)
        "n_features_total": 16 + 20 * 11 + 2,   # 238
        "desc": "qinst_pelvisrel_winpos_w20 + windowed binary R/L contact flags (single/dual stance) = 238 ch",
    },
}

CONTACT_GRF_BW_THRESHOLD = 0.05   # vertical GRF (BW) above which a foot counts as in contact


def feature_count(input_set):
    cfg = INPUT_SETS[input_set]
    if "n_features_total" in cfg:
        return cfg["n_features_total"]
    if cfg.get("uses_com"):
        return cfg["per_frame_ch"] * cfg.get("window", 1)
    return len(cfg["base_indices"]) + int(cfg["include_height"]) + int(cfg["include_bmi"])


# ─── Single-frame MLP architecture ───────────────────────────────────────────

class JCF_MLP(nn.Module):
    """
    Per-frame MLP. Maps one frame of (joint q/qdot/qddot, GRF, mass, speed,
    height, BMI) to one frame of (Fx, Fy, Fz, Mx).

    Architecture: 3 hidden layers of 128 units, LayerNorm + ReLU + dropout 0.3.
    ~25K parameters. Smaller capacity + heavier regularization help generalize
    from the limited static training set.
    """
    def __init__(self, n_features=64, n_outputs=4, hidden=128, dropout=0.3,
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
        # Accept [B, F] or [B, T, F]; return same shape with last dim = n_outputs
        in_shape = x.shape
        if x.ndim == 3:
            flat = x.reshape(-1, x.shape[-1])
            return self.net(flat).reshape(*in_shape[:-1], -1)
        return self.net(x)


# ─── Dataset: individual frames sampled across all subjects ──────────────────

def _replace_dq_with_forward_diff(inputs, dt=0.01):
    """Replace channels 16-31 (dq) with William's forward finite difference:
        dq_k = (q_k - q_{k-1}) / dt
    Frame 0 has no previous frame; we duplicate dq[1] there to avoid a NaN edge.
    """
    q = inputs[:, :16]
    dq = np.zeros_like(q)
    dq[1:] = (q[1:] - q[:-1]) / dt
    if len(dq) >= 2:
        dq[0] = dq[1]
    inputs[:, 16:32] = dq
    return inputs


def _load_com_features_aligned(subj_dir, target_time):
    """Load com_features.npz for one trial and interpolate every channel onto
    target_time (in seconds). Returns dict of arrays each [T, 3] aligned to
    target_time, or None if features aren't cached.
    """
    path = os.path.join(subj_dir, "com_features.npz")
    if not os.path.exists(path):
        return None
    f = np.load(path)
    src_t = f["time"]
    out = {}
    for key in ("com_pos", "com_vel", "calcn_r_pos", "calcn_l_pos"):
        arr = f[key]
        out[key] = np.column_stack([
            np.interp(target_time, src_t, arr[:, j]) for j in range(3)
        ]).astype(np.float32)
    return out


def _build_com_features(subj_dir, cfg, max_peak_bw, min_length, predict_moment):
    """Construct the (q, v_com, x_com - x_r, x_com - x_l, [mass, height]) feature
    matrix for one trial. Returns (features [T, per_frame_ch], labels [T, 4], mass).

    Uses load_subject with native_time_grid=True so the IK time grid is preserved
    end-to-end, making alignment with the cached COM features straightforward.
    """
    import json as _json
    r = load_subject(subj_dir,
                     lower_body_only=True,
                     clean_features=True,
                     include_mass=True,
                     include_speed=False,           # not needed for this variant
                     predict_moment=predict_moment,
                     max_peak_bw=max_peak_bw,
                     min_length=min_length,
                     native_time_grid=True)
    if r is None:
        return None
    inputs, labels, mass = r

    # q is channels 0:16 of the load_subject output
    q = inputs[:, :16]
    T = q.shape[0]

    # Read trial metadata + subject height
    try:
        meta = _json.load(open(os.path.join(subj_dir, "metadata.json")))
        height_m = float(meta.get("height_m", 1.7))
    except Exception:
        return None
    if height_m <= 0:
        return None

    # We need IK time to align COM features. native_time_grid=True trims to
    # the common range, so reconstruct the time grid from the trial's IK file.
    ik_path = os.path.join(subj_dir, "ik_results.mot")
    if not os.path.exists(ik_path):
        return None
    ik_df = pd.read_csv(ik_path, sep=r"\s+", skiprows=6)
    ik_time = ik_df["time"].values
    # Trim to match q (load_subject's trim mask used max/min on ik/grf/jcf times).
    # We don't have the exact mask, so interpolate onto a uniform t-grid matching q's length.
    # Build a uniform grid spanning the full range of ik_time, length T.
    if T < 1 or len(ik_time) < 1:
        return None
    t_target = np.linspace(ik_time[0], ik_time[-1], T).astype(np.float64)
    # Better alternative: pick t_target so its spacing equals median ik_dt,
    # then slice. But for COM interpolation, near-uniform spacing matters more
    # than exact alignment.

    com = _load_com_features_aligned(subj_dir, t_target)
    if com is None:
        return None

    if cfg.get("uses_pelvis_rel"):
        # Pelvis world position from raw IK (not zero-centered).
        # ik_df is already loaded above. pelvis_tx/ty/tz are at the trial's
        # native ik_time grid; interpolate onto t_target.
        pelvis_raw = np.column_stack([
            ik_df["pelvis_tx"].values,
            ik_df["pelvis_ty"].values,
            ik_df["pelvis_tz"].values,
        ])
        pelvis_pos = np.column_stack([
            np.interp(t_target, ik_time, pelvis_raw[:, j]) for j in range(3)
        ]).astype(np.float32)                               # [T, 3]
        rel_com = com["com_pos"] - pelvis_pos               # COM relative to pelvis
        rel_r = com["calcn_r_pos"] - pelvis_pos             # right foot relative to pelvis
        rel_l = com["calcn_l_pos"] - pelvis_pos             # left foot relative to pelvis
        # For the per-window centering variant, swap the trial-centered q[3:6]
        # back to the raw (uncentered) pelvis world position so that
        # _windowize_positions_only can apply per-window centering at the
        # right granularity. Other variants keep the per-trial-centered q.
        if cfg.get("pelvis_per_window_centering"):
            q = q.copy()
            q[:, 3:6] = pelvis_pos
        parts = [q, rel_com, rel_r, rel_l]                  # [T, 16+3+3+3=25]
    else:
        # qvcomrel: COM velocity + foot positions relative to COM
        v_com = com["com_vel"]                              # [T, 3]
        rel_r = com["com_pos"] - com["calcn_r_pos"]         # [T, 3]
        rel_l = com["com_pos"] - com["calcn_l_pos"]         # [T, 3]
        parts = [q, v_com, rel_r, rel_l]                    # [T, 16+3+3+3=25]

    # Binary R/L contact flags from vertical GRF, appended to the windowed
    # position block so they carry stance context (single vs dual) across the
    # 20-frame window. GRF per foot is channels 48-53 of load_subject output:
    # (Fx_r, Fy_r, Fz_r, Fx_l, Fy_l, Fz_l) in BW; Fy is vertical (OpenSim y-up).
    if cfg.get("include_contact"):
        if inputs.shape[1] < 53:
            return None   # GRF channels unavailable; can't build contact flags
        fy_r = inputs[:, 49]
        fy_l = inputs[:, 52]
        r_contact = (np.abs(fy_r) > CONTACT_GRF_BW_THRESHOLD).astype(np.float32)[:, None]
        l_contact = (np.abs(fy_l) > CONTACT_GRF_BW_THRESHOLD).astype(np.float32)[:, None]
        parts += [r_contact, l_contact]                    # [T, ..., +2]

    if cfg["include_height"]:
        parts.append(np.full((T, 1), mass, dtype=np.float32))
        parts.append(np.full((T, 1), height_m, dtype=np.float32))

    feat = np.hstack(parts).astype(np.float32)
    assert feat.shape[1] == cfg["per_frame_ch"], \
        f"Feature width {feat.shape[1]} != expected {cfg['per_frame_ch']}"
    return feat, labels.astype(np.float32), mass


def _windowize(feat, window, replicate_q=False, q_dim=16):
    """Build a [T, window * F] array where each row k contains the flattened
    window of `window` frames centered on k, with edge replication at boundaries.
    `feat` is [T, F]; the prediction target for each row is still frame k's label.

    If replicate_q=True, the first q_dim channels (joint angles) at every frame
    within the window are overwritten by the center frame's q values, so the
    window contains only one snapshot of joint posture — the position/velocity
    channels still vary across the window normally.
    """
    if window <= 1:
        return feat
    T, F = feat.shape
    half = window // 2     # if window=20, half=10 (10 past + 10 future, with current at index 10)
    out = np.zeros((T, window, F), dtype=feat.dtype)
    for offset in range(window):
        # Relative position within the window: index k of the window samples frame (k - half + offset)
        rel = offset - half
        src_idx = np.clip(np.arange(T) + rel, 0, T - 1)
        out[:, offset, :] = feat[src_idx]
    if replicate_q:
        # Replace q (channels 0:q_dim) at every window position with the
        # center-frame q. Broadcasts feat[:, :q_dim] (shape [T, q_dim]) across
        # the window axis.
        out[:, :, :q_dim] = feat[:, np.newaxis, :q_dim]
    return out.reshape(T, window * F)


def _windowize_positions_only(feat, window, q_dim=16, n_pos=9,
                              pelvis_t_cols=None):
    """For variants where only the position channels are windowed; q and any
    extras (mass, height) are taken from the center frame only.

    Layout of `feat` per frame: [q (q_dim), positions (n_pos), extras (rest)].
    Output per sample: [q (q_dim), windowed_positions (window*n_pos), extras (rest)].

    For window=20, q_dim=16, n_pos=9, extras=2 → 16 + 180 + 2 = 198 channels.

    pelvis_t_cols: optional iterable of column indices in q (e.g. (3, 4, 5))
        whose values get **per-window** zero-centered: each output row's
        q[cols] becomes q[k, cols] − mean(q[k-half : k-half+window, cols]).
        Caller must supply raw (un-trial-centered) values in those columns of
        `feat`. Used by the `*_winctr_*` input_sets.
    """
    if window <= 1:
        return feat
    T, F = feat.shape
    extras_size = F - q_dim - n_pos
    half = window // 2
    pos_start = q_dim
    pos_end = q_dim + n_pos
    pos_windowed = np.zeros((T, window, n_pos), dtype=feat.dtype)
    for offset in range(window):
        rel = offset - half
        src_idx = np.clip(np.arange(T) + rel, 0, T - 1)
        pos_windowed[:, offset, :] = feat[src_idx, pos_start:pos_end]
    pos_flat = pos_windowed.reshape(T, window * n_pos)
    q_center = feat[:, :q_dim].copy()
    if pelvis_t_cols is not None:
        cols = list(pelvis_t_cols)
        # Per-window mean of the pelvis_t channels using the same edge-replicating
        # window slicing the positions use, so train and inference agree even at
        # trial boundaries.
        pelvis_win = np.zeros((T, window, len(cols)), dtype=feat.dtype)
        for offset in range(window):
            rel = offset - half
            src_idx = np.clip(np.arange(T) + rel, 0, T - 1)
            pelvis_win[:, offset, :] = feat[src_idx][:, cols]
        q_center[:, cols] = q_center[:, cols] - pelvis_win.mean(axis=1)
    extras_center = feat[:, pos_end:] if extras_size > 0 else np.zeros((T, 0), dtype=feat.dtype)
    return np.hstack([q_center, pos_flat, extras_center])


class FrameDataset(Dataset):
    """
    Flattens each subject's full sequence into individual frames.
    All frames from all subjects are pooled into one big sample list.
    Memory-resident for fast random access during training.
    """
    def __init__(self, subject_dirs, max_peak_bw=6.0, include_speed=True,
                 predict_moment=True, min_length=1,
                 input_set="full", dq_method="central"):
        import json as _json
        cfg = INPUT_SETS[input_set]
        n_features = feature_count(input_set)
        is_com = cfg.get("uses_com", False)
        window = cfg.get("window", 1)
        chunks_in, chunks_out = [], []

        for subj_dir in subject_dirs:
            # ─── COM-based variants take a separate code path ───
            if is_com:
                built = _build_com_features(subj_dir, cfg,
                                            max_peak_bw=max_peak_bw,
                                            min_length=min_length,
                                            predict_moment=predict_moment)
                if built is None:
                    continue
                feat, labels, mass = built
                if window > 1:
                    if cfg.get("window_only_positions", False):
                        pelvis_cols = (3, 4, 5) if cfg.get("pelvis_per_window_centering") else None
                        feat = _windowize_positions_only(
                            feat, window, n_pos=cfg.get("n_pos", 9),
                            pelvis_t_cols=pelvis_cols,
                        )
                    else:
                        feat = _windowize(feat, window,
                                          replicate_q=cfg.get("replicate_q", False))
                chunks_in.append(feat)
                chunks_out.append(labels)
                continue

            # ─── Existing (q, F, ...) variants ───
            r = load_subject(subj_dir,
                             lower_body_only=True,
                             clean_features=True,
                             include_mass=True,
                             include_speed=include_speed,
                             predict_moment=predict_moment,
                             max_peak_bw=max_peak_bw,
                             min_length=min_length)
            if r is None:
                continue
            inputs, labels, mass = r
            # Optionally swap central-difference dq for William's forward diff.
            if dq_method == "forward":
                inputs = _replace_dq_with_forward_diff(inputs)
            elif dq_method != "central":
                raise ValueError(f"Unknown dq_method: {dq_method}")
            try:
                meta = _json.load(open(os.path.join(subj_dir, "metadata.json")))
                height_m = float(meta.get("height_m", 1.7))
            except Exception:
                continue
            if height_m <= 0:
                continue
            T = inputs.shape[0]
            # Select the requested base channels for this input_set.
            inputs = inputs[:, cfg["base_indices"]]
            # Conditionally append subject-level features. Height (m) helps the
            # model normalize moments (M_add is reported per BW·H). BMI helps
            # cross-subject generalization for full feature sets.
            extras = []
            if cfg["include_height"]:
                extras.append(np.full(T, height_m, dtype=np.float32))
            if cfg["include_bmi"]:
                bmi = mass / (height_m * height_m)
                extras.append(np.full(T, bmi, dtype=np.float32))
            if extras:
                inputs = np.hstack([inputs, np.column_stack(extras)])
            chunks_in.append(inputs.astype(np.float32))
            chunks_out.append(labels.astype(np.float32))

        if chunks_in:
            self.frames_in = np.concatenate(chunks_in, axis=0)
            self.frames_out = np.concatenate(chunks_out, axis=0)
        else:
            self.frames_in = np.zeros((0, n_features), dtype=np.float32)
            self.frames_out = np.zeros((0, N_OUTPUTS), dtype=np.float32)
        self.n_features = n_features
        self.input_set = input_set
        self.dq_method = dq_method

    def __len__(self):
        return len(self.frames_in)

    def __getitem__(self, idx):
        return self.frames_in[idx], self.frames_out[idx]


# ─── Training loop ────────────────────────────────────────────────────────────

def _scan_dirs(data_root, dataset=None, exclude=None):
    excluded = set(exclude) if exclude else set()
    out = []
    for name in sorted(os.listdir(data_root)):
        if dataset and not name.startswith(f"{dataset}_"):
            continue
        if any(name.startswith(f"{ex}_") for ex in excluded):
            continue
        subj_dir = os.path.join(data_root, name)
        jcf_sto = os.path.join(subj_dir, "jcf_output",
                               "BatchJCF_JointReaction_ReactionLoads.sto")
        if os.path.isdir(subj_dir) and os.path.exists(jcf_sto):
            out.append(subj_dir)
    return out


def _subject_id(path):
    return re.sub(r"_t\d+(_r\d+)?$", "", os.path.basename(path))


def _split_dirs(subject_dirs, train_size, seed):
    unique = sorted(set(_subject_id(d) for d in subject_dirs))
    train_subj, val_subj = train_test_split(unique, train_size=train_size, random_state=seed)
    s = set(train_subj)
    train_dirs = [d for d in subject_dirs if _subject_id(d) in s]
    val_dirs = [d for d in subject_dirs if _subject_id(d) not in s]
    return train_dirs, val_dirs, len(train_subj), len(val_subj)


def train(activity='static', filter_flat=True, dataset=None, exclude=None,
          max_peak_bw=6.0, seed=SEED, input_set='full', dq_method='central',
          epochs=None, lr_patience=None, fmedial_weight=0.0, activation='relu',
          dropout=None):
    """
    Train MLP on per-frame data from one or both activities.
      activity='static':  ./jcf/full_duration/training/static/  (for posture optimization)
      activity='walking': ./jcf/full_duration/training/walking/ (for gait analysis)
      activity='both':    combined; uses WeightedRandomSampler to give static and
                          walking equal mass per epoch despite size imbalance.

      input_set: which subset of features the model receives. See INPUT_SETS.
      dq_method: 'central' (np.gradient, default) or 'forward' (William's
                 dq_k = (q_k - q_{k-1})/dt).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    n_features = feature_count(input_set)
    print(f"Input set: {input_set} — {INPUT_SETS[input_set]['desc']}  ({n_features} ch)")
    print(f"dq method: {dq_method}")

    # Static is essentially flat by construction — skip the flat-waveform filter.
    apply_flat_filter = filter_flat and activity == 'walking'

    # Collect train/val dirs per activity
    train_dirs_by = {}
    val_dirs_by = {}
    output_root = None
    for act in (['static', 'walking'] if activity == 'both' else [activity]):
        root = DATA_ROOTS[act]
        if not os.path.isdir(root):
            print(f"[{act}] root does not exist: {root}")
            continue
        dirs = _scan_dirs(root, dataset=dataset, exclude=exclude)
        print(f"[{act}] found {len(dirs)} subject directories")
        if apply_flat_filter and act == 'walking':
            n0 = len(dirs)
            dirs, n_flat = filter_flat_subjects(dirs)
            print(f"[{act}] flat filter: {n0} → {len(dirs)} (removed {n_flat})")
        td, vd, ns_train, ns_val = _split_dirs(dirs, TRAIN_SPLIT, seed)
        train_dirs_by[act] = td
        val_dirs_by[act] = vd
        print(f"[{act}] train: {ns_train} subjects ({len(td)} trials), "
              f"val: {ns_val} subjects ({len(vd)} trials)")
        output_root = root if output_root is None else output_root

    if not train_dirs_by:
        print("No training data found.")
        return

    print("\nLoading training frames...")
    train_ds_per = {}
    for act, td in train_dirs_by.items():
        ds = FrameDataset(td, max_peak_bw=max_peak_bw,
                          input_set=input_set, dq_method=dq_method)
        train_ds_per[act] = ds
        print(f"  [{act}] {len(ds):,} training frames")

    # Combine if both
    if activity == 'both':
        all_in = np.concatenate([d.frames_in for d in train_ds_per.values()], axis=0)
        all_out = np.concatenate([d.frames_out for d in train_ds_per.values()], axis=0)
        sample_weights = np.concatenate([
            np.full(len(d), 1.0 / max(len(d), 1)) for d in train_ds_per.values()
        ])
        train_ds = FrameDataset.__new__(FrameDataset)
        train_ds.frames_in = all_in
        train_ds.frames_out = all_out
    else:
        train_ds = list(train_ds_per.values())[0]
        sample_weights = None

    if len(train_ds) == 0:
        print("No training frames. Check data.")
        return

    # Compute and apply normalization (always from the combined train set)
    inp_mean = torch.from_numpy(train_ds.frames_in.mean(axis=0))
    inp_std = torch.from_numpy(train_ds.frames_in.std(axis=0)).clamp(min=1e-8)
    print(f"\n  input_mean range: [{inp_mean.min():.2f}, {inp_mean.max():.2f}]")
    train_ds.frames_in = (train_ds.frames_in - inp_mean.numpy()) / inp_std.numpy()

    print("\nLoading validation frames...")
    val_pieces_in, val_pieces_out = [], []
    for act, vd in val_dirs_by.items():
        ds = FrameDataset(vd, max_peak_bw=max_peak_bw,
                          input_set=input_set, dq_method=dq_method)
        ds.frames_in = (ds.frames_in - inp_mean.numpy()) / inp_std.numpy()
        val_pieces_in.append(ds.frames_in)
        val_pieces_out.append(ds.frames_out)
        print(f"  [{act}] {len(ds):,} validation frames")
    val_ds = FrameDataset.__new__(FrameDataset)
    val_ds.frames_in = np.concatenate(val_pieces_in, axis=0)
    val_ds.frames_out = np.concatenate(val_pieces_out, axis=0)

    if sample_weights is not None:
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights),
                                        replacement=True)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                                  num_workers=0, pin_memory=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, pin_memory=True)

    effective_dropout = DROPOUT if dropout is None else dropout
    model = JCF_MLP(n_features=n_features, n_outputs=N_OUTPUTS,
                    hidden=HIDDEN, dropout=effective_dropout,
                    activation=activation).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: JCF_MLP {n_params:,} params (hidden={HIDDEN}, dropout={effective_dropout}, activation={activation})")

    n_epochs = epochs if epochs is not None else EPOCHS
    sched_patience = lr_patience if lr_patience is not None else 3
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE,
                                 weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=sched_patience, factor=0.5)

    # Per-sample subject height (m) recovery for the F_medial loss term.
    # Requires include_height=True with height stored at the last input channel
    # (true for all COM variants and for qfmh/qdqfmh).
    cfg_loss = INPUT_SETS[input_set]
    if fmedial_weight > 0 and not cfg_loss.get("include_height", False):
        raise ValueError(
            f"--fmedial-weight requires input_set with include_height=True; "
            f"got input_set='{input_set}'."
        )
    height_mean_t = inp_mean[-1].to(DEVICE).float() if fmedial_weight > 0 else None
    height_std_t = inp_std[-1].to(DEVICE).float() if fmedial_weight > 0 else None
    ALPHA_FMEDIAL = 0.5
    D_M_FMEDIAL = 0.045

    def loss_fn(pred, target, inp=None):
        # Plain MSE + magnitude-weighted force MSE.
        # No gradient term — single-frame model has no temporal axis.
        mse = ((pred - target) ** 2).mean()
        force_mag = torch.sqrt((target[:, :3] ** 2).sum(dim=-1, keepdim=True)).detach()
        weighted_num = ((pred[:, :3] - target[:, :3]) ** 2 * force_mag).sum()
        weighted_den = (force_mag.sum() + 1e-8) * 3
        weighted = weighted_num / weighted_den
        total = mse + 0.5 * weighted
        if fmedial_weight > 0 and inp is not None:
            # De-normalize height per sample (last input channel) and add an
            # explicit F_medial = alpha*|F_y| + (M_x*H)/d_m MSE term so the
            # optimizer sees the lever-arm-amplified M_x error directly.
            H = inp[:, -1] * height_std_t + height_mean_t                  # [B]
            fy_p, mx_p = pred[:, 1], pred[:, 3]
            fy_t, mx_t = target[:, 1], target[:, 3]
            fm_p = ALPHA_FMEDIAL * torch.abs(fy_p) + (mx_p * H) / D_M_FMEDIAL
            fm_t = ALPHA_FMEDIAL * torch.abs(fy_t) + (mx_t * H) / D_M_FMEDIAL
            fm_loss = ((fm_p - fm_t) ** 2).mean()
            total = total + fmedial_weight * fm_loss
        return total

    best_val = float("inf")
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0
        n_train = 0
        for inp, lbl in train_loader:
            inp = inp.to(DEVICE); lbl = lbl.to(DEVICE)
            optimizer.zero_grad()
            pred = model(inp)
            loss = loss_fn(pred, lbl, inp)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(inp)
            n_train += len(inp)
        train_loss /= n_train

        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for inp, lbl in val_loader:
                inp = inp.to(DEVICE); lbl = lbl.to(DEVICE)
                pred = model(inp)
                # Validation uses the same composite objective so that
                # checkpointing tracks the F_medial-aware optimum.
                v = loss_fn(pred, lbl, inp)
                val_loss += v.item() * len(inp)
                n_val += len(inp)
        val_loss /= n_val
        scheduler.step(val_loss)

        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1:3d}/{n_epochs} | train: {train_loss:.6f} | "
              f"val: {val_loss:.6f} | lr: {lr:.1e}")

        if val_loss < best_val:
            best_val = val_loss
            # Default input_set keeps original filename; add suffix for variants
            suffix = "" if input_set == "full" else f"_{input_set}"
            if dq_method != "central":
                suffix += f"_{dq_method}"
            if fmedial_weight > 0:
                # e.g. fmW2 for fmedial_weight=2.0
                w = ("%g" % fmedial_weight).replace(".", "p")
                suffix += f"_fmW{w}"
            if activation != 'relu':
                suffix += f"_{activation}"
            ckpt_name = f"best_model_mlp_{activity}{suffix}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
                "n_features": n_features,
                "n_outputs": N_OUTPUTS,
                "input_mean": inp_mean,
                "input_std": inp_std,
                "model_type": "mlp",
                "activity": activity,
                "hidden": HIDDEN,
                "dropout": effective_dropout,
                "input_set": input_set,
                "dq_method": dq_method,
                "fmedial_weight": fmedial_weight,
                "activation": activation,
                "base_indices": INPUT_SETS[input_set]["base_indices"],
                "include_height": INPUT_SETS[input_set]["include_height"],
                "include_bmi": INPUT_SETS[input_set]["include_bmi"],
            }, os.path.join(output_root, ckpt_name))
            print(f"  → saved (val={val_loss:.6f}) → {output_root}/{ckpt_name}")

    suffix = "" if input_set == "full" else f"_{input_set}"
    if dq_method != "central":
        suffix += f"_{dq_method}"
    print(f"\nBest val loss: {best_val:.6f}")
    print(f"Saved to {output_root}/best_model_mlp_{activity}{suffix}.pt")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--activity", type=str, default="static",
                   choices=["static", "walking", "both"],
                   help="Which dataset(s) to train on. 'static' (default) for "
                        "posture optimization. 'walking' for gait. 'both' uses "
                        "weighted sampling so static and walking get equal mass.")
    p.add_argument("--no-flat-filter", action="store_true",
                   help="Disable flat-waveform filter (only relevant for walking).")
    p.add_argument("--dataset", type=str, default=None,
                   help="Restrict to one source dataset prefix (e.g. carter).")
    p.add_argument("--exclude", type=str, nargs="+", default=None,
                   help="Exclude source dataset prefixes (e.g. han fregly).")
    p.add_argument("--max-peak", type=float, default=6.0,
                   help="Reject subjects with JCF peak > this many BW.")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--input-set", type=str, default="full",
                   choices=list(INPUT_SETS.keys()),
                   help="Which feature subset the MLP sees. "
                        "'full' (default) keeps the original 64-channel input. "
                        "'qdqf' = (q, dq, F) only — strict 38 ch. "
                        "'qdqfmh' = above + mass + height — 40 ch. "
                        "'qf' = (q, F) only — 22 ch. "
                        "'qfmh' = (q, F, mass, height) — 24 ch.")
    p.add_argument("--dq-method", type=str, default="central",
                   choices=["central", "forward"],
                   help="How to compute dq channels. 'central' uses np.gradient "
                        "(default). 'forward' uses William's "
                        "(q_k - q_{k-1})/dt — matches a real-time forward "
                        "finite-difference deployment.")
    p.add_argument("--epochs", type=int, default=None,
                   help=f"Number of training epochs (default {EPOCHS}). "
                        f"Use more if val loss is still dropping at the end.")
    p.add_argument("--lr-patience", type=int, default=None,
                   help="ReduceLROnPlateau patience (default 3). Higher = "
                        "scheduler waits longer before halving LR, so the model "
                        "trains longer at each LR rung. Useful when increasing "
                        "--epochs.")
    p.add_argument("--fmedial-weight", type=float, default=0.0,
                   help="If >0, add lambda * MSE(F_medial_pred, F_medial_target) "
                        "to the training loss, where "
                        "F_medial = 0.5*|F_y| + (M_x*H)/0.045 (BW). This gives "
                        "the optimizer direct pressure on the lever-arm-amplified "
                        "M_x error, addressing the observed peak attenuation in "
                        "the medial proxy. Requires include_height=True. Try "
                        "1.0-5.0 as a starting range.")
    p.add_argument("--activation", type=str, default="relu",
                   choices=["relu", "silu"],
                   help="Hidden-layer activation. 'relu' (default) is the "
                        "original; 'silu' (Swish, x*sigmoid(x)) is smooth "
                        "everywhere, needed when the surrogate is wrapped in "
                        "a derivative-based optimizer like Ipopt that fails on "
                        "ReLU's kink at zero.")
    p.add_argument("--dropout", type=float, default=None,
                   help=f"Override hidden-layer dropout (default {DROPOUT}). "
                        f"Raise (e.g. 0.4-0.5) when overfitting — useful for "
                        f"SiLU runs which fit train more aggressively than ReLU.")
    args = p.parse_args()
    train(activity=args.activity,
          filter_flat=not args.no_flat_filter,
          dataset=args.dataset, exclude=args.exclude,
          max_peak_bw=args.max_peak, seed=args.seed,
          input_set=args.input_set, dq_method=args.dq_method,
          epochs=args.epochs, lr_patience=args.lr_patience,
          fmedial_weight=args.fmedial_weight,
          activation=args.activation,
          dropout=args.dropout)

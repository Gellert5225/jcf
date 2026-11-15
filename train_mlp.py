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

    # SO+JR labels (jcf_output_sojr/): the *_sojr input sets; label filters set with --max-peak (BW),
    # --max-moment (BW*H) and --flat-cv / --flat-dr / --no-flat-filter (walking only)
    python train_mlp.py --activity walking --exclude han fregly \\
        --input-set qinst_notrans_pelvisrel_winpos_contact_fpopt_w20_t100_sojr --max-peak <BW> --max-moment <BW*H> ...

    # QC-passing SO+JR frames only, physical M_x sign in the F_medial loss (the *_sojrqc sets; needs
    # <trial>/jcf_output_sojr/label_mask.npz from `python -I analysis/thesis/so_label_qc.py masks`). The trial-level
    # --max-peak / --max-moment filters are off for these sets unless given (1000 = off), and a given value is judged
    # on the QC-kept frames only (see MASKED_MAX_PEAK_BW); state them explicitly so the command is self-describing:
    python train_mlp.py --activity walking --exclude han fregly \\
        --input-set qinst_notrans_pelvisrel_winpos_contact_fpopt_w20_t100_sojrqc --max-peak 1000 --max-moment 1000 \\
        --fmedial-weight 0.5 ...
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

from train_cnn import load_subject, filter_flat_subjects, label_nonfinite_rows, jr_label_path, SEED, TRAIN_SPLIT
import provenance as _prov
from pipeline.compute_com_features import foot_point_filename

DATA_ROOTS = {
    "static":  "./jcf/full_duration/training/static",
    "walking": "./jcf/full_duration/training/walking",
}
CKPT_ROOT = "./models/runs/mlp"   # where training writes checkpoints; curated models live in ./models/<name>/

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
    # Same as qinst_pelvisrel_winpos_contact_w20 but WITHOUT pelvis_tx/ty/tz in q.
    # The models were very sensitive to pelvis height relative to its local mean
    # (~0.3–0.5 BW per cm), a gait-phase cue an optimizer can exploit; this variant
    # removes the channel. Pelvis tilt/list/rotation and all COM/foot-relative-to-pelvis
    # windows are kept, so posture information is unchanged.
    "qinst_notrans_pelvisrel_winpos_contact_w20": {
        "base_indices": None,
        "include_height": True,
        "include_bmi": False,
        "uses_com": True,
        "uses_pelvis_rel": True,
        "replicate_q": False,
        "window_only_positions": True,
        "include_contact": True,
        "drop_pelvis_translation": True,
        "q_dim": 13,                            # q16 minus pelvis_tx/ty/tz
        "window": 20,
        "per_frame_ch": 26,                     # q(13) + pos(9) + contact(2) + mass,height(2)
        "n_pos": 11,
        "n_features_total": 13 + 20 * 11 + 2,   # 235
        "desc": "qinst_pelvisrel_winpos_contact_w20 without pelvis_tx/ty/tz = 235 ch",
    },
    # Same as qinst_notrans_pelvisrel_winpos_contact_w20, but each foot is the point
    # (0.13, 0, -0.01) m in its calcn frame (x forward, y up, z right; same point for
    # both feet) instead of the calcn origin, to match the optimizer's foot point.
    # Needs the foot_point_0.130_0.000_-0.010.npz cache from
    # pipeline/compute_com_features.py --foot-point 0.13 0 -0.01.
    "qinst_notrans_pelvisrel_winpos_contact_fp13_w20": {
        "base_indices": None,
        "include_height": True,
        "include_bmi": False,
        "uses_com": True,
        "uses_pelvis_rel": True,
        "replicate_q": False,
        "window_only_positions": True,
        "include_contact": True,
        "drop_pelvis_translation": True,
        "foot_point": (0.13, 0.0, -0.01),
        "q_dim": 13,
        "window": 20,
        "per_frame_ch": 26,
        "n_pos": 11,
        "n_features_total": 13 + 20 * 11 + 2,   # 235
        "desc": "qinst_notrans_pelvisrel_winpos_contact_w20 with feet at calcn (0.13, 0, -0.01) = 235 ch",
    },
    # The optimizer's foot point: (0.13, 0, -0.01) in the Rajagopal URDF's Z-up calcn link axes, which is
    # (0.13, -0.01, 0) in OpenSim calcn axes (13 cm forward, 1 cm below the origin, centred; same on both feet).
    # The fp13 set above took the URDF numbers as OpenSim axes. Needs the foot_point_0.130_-0.010_0.000.npz
    # cache from pipeline/compute_com_features.py --foot-point 0.13 -0.01 0.
    "qinst_notrans_pelvisrel_winpos_contact_fpopt_w20": {
        "base_indices": None,
        "include_height": True,
        "include_bmi": False,
        "uses_com": True,
        "uses_pelvis_rel": True,
        "replicate_q": False,
        "window_only_positions": True,
        "include_contact": True,
        "drop_pelvis_translation": True,
        "foot_point": (0.13, -0.01, 0.0),
        "q_dim": 13,
        "window": 20,
        "per_frame_ch": 26,
        "n_pos": 11,
        "n_features_total": 13 + 20 * 11 + 2,   # 235
        "desc": "qinst_notrans_pelvisrel_winpos_contact_w20 with feet at the optimizer's point, calcn (0.13, -0.01, 0) = 235 ch",
    },
}

# "_t100" variants (thesis, Nov 2026): same inputs as their base set, built on the exact time grid and resampled to
# 100 Hz. The legacy builder sampled the COM/foot/pelvis windows on linspace(ik_time[0], ik_time[-1], T), i.e. over the
# whole IK segment including the 0.3 s SO buffers, while q, contact flags and labels sit on the IK frames inside the
# JR window: positions were time-warped by up to +-0.3 s (foot channels ~16-22 cm off on Carter trials). Legacy sets
# also kept each lab's native rate, so the 20-frame window spanned 80 ms (250 Hz) to 200 ms (100 Hz); the optimizer and
# the PN3 driver feed 20 frames at 100 Hz (200 ms). Legacy sets are left unchanged so existing checkpoints evaluate as
# they were trained.
for _base in ("qinst_notrans_pelvisrel_winpos_contact_w20", "qinst_notrans_pelvisrel_winpos_contact_fpopt_w20"):
    INPUT_SETS[_base + "_t100"] = dict(INPUT_SETS[_base], time_exact=True, resample_hz=100.0,
                                       desc=INPUT_SETS[_base]["desc"] + ", exact time alignment, 100 Hz windows (200 ms)")

# Knee labels: every set reads <trial>/<label_subdir>/BatchJCF_JointReaction_ReactionLoads.sto, "jcf_output" unless
# the set says otherwise. The stored jcf_output/ labels are not static optimization + joint reaction: batch_process.py
# ran SO and JR in one AnalyzeTool with an empty <forces_file/>, so JR used muscle forces computed from the model
# states. jcf_output_sojr/ (analysis/thesis/gc_knee_jr.py) is JR rerun with forces_file = the trial's SO force file.
# The "_sojr" sets below have the same inputs as their _t100 base and take those labels.
DEFAULT_LABEL_SUBDIR = "jcf_output"
SOJR_LABEL_SUBDIR = "jcf_output_sojr"
for _base in ("qinst_notrans_pelvisrel_winpos_contact_w20_t100", "qinst_notrans_pelvisrel_winpos_contact_fpopt_w20_t100"):
    INPUT_SETS[_base + "_sojr"] = dict(INPUT_SETS[_base], label_subdir=SOJR_LABEL_SUBDIR,
                                       desc=INPUT_SETS[_base]["desc"] + f", SO+JR labels ({SOJR_LABEL_SUBDIR}/)")


def label_subdir_of(cfg):
    """Folder (inside each trial) whose BatchJCF_JointReaction_ReactionLoads.sto holds an input set's labels."""
    return cfg.get("label_subdir", DEFAULT_LABEL_SUBDIR)


# M_x sign convention (Nov 2026). Labels and network outputs are the raw JointReaction mx of walker_knee_r (on and in
# tibia_r) / (BW*H). The medial-positive moment that F_medial needs is -mx at walker_knee_r (+mx at walker_knee_l):
# analysis/thesis/knee_mx_sign.py, tests/test_knee_mx_sign.py. The pipeline and the ISRR 2026 formula used +mx.
# cfg["mx_medial_sign"] is the factor on the raw right-knee mx in F_medial = 0.5*|F_y| + sign*mx*H/0.045: +1
# (MX_SIGN_PIPELINE, the default) for every legacy set, so they train and evaluate as before; -1 (MX_SIGN_PHYSICAL) for
# the *_sojrqc sets. Every checkpoint stores the sign its F_medial loss used, and inference_posture_surrogate.py forms
# f_medial with it (old checkpoints, which do not store it, keep +1).
MX_SIGN_PIPELINE = 1.0
MX_SIGN_PHYSICAL = -1.0
MX_CONVENTIONS = {"physical": MX_SIGN_PHYSICAL, "pipeline": MX_SIGN_PIPELINE}


def mx_medial_sign_of(cfg):
    """Factor on the raw right-knee mx that gives the medial moment in an input set's (or checkpoint's) F_medial."""
    return float(cfg.get("mx_medial_sign", MX_SIGN_PIPELINE))


# Frame-level label QC (Nov 2026): "*_sojrqc" = the *_t100_sojr set, physical M_x sign, and only frames that pass the
# SO+JR label QC as training targets. <trial>/<label_subdir>/label_mask.npz (written by
# `analysis/thesis/so_label_qc.py masks`) holds keep_r on the JR time grid: False on every frame of a trial whose right
# foot has a fixed centre of pressure and on frames any SO rule flags (knee muscle or >= 3 leg muscles saturated, leg
# reserve > 0.1 N*m/kg, pelvis residual moment > 5 %, activation jump, ankle co-activation). The |F_y| > 5 BW flag is
# stored but not part of keep_r. Inputs still use full windows; only rows whose centre frame is masked are dropped.
# Carrying the mask onto the frame grid (carry_label_mask): the labels reach the 100 Hz grid by two linear
# interpolations (JR -> IK frames in load_subject, IK -> 100 Hz here). The same two interpolations are applied to the
# masked-frame indicator, which gives at each output frame the summed interpolation weight of masked JR frames in its
# label; a frame is kept when that weight is at most LABEL_MASK_TOL = 0.05, i.e. at most 5 % of a kept label comes
# from rejected JR frames. Why not 0, "nearest JR frame kept" or "all JR frames within +-5 ms kept" (426 training
# walking trials, kept 100 Hz frames relative to nearest-kept): 0 keeps only 62 % on Moore, whose JR rows sit a
# fraction of a millisecond off the IK frames, so every masked run grows by one frame at ~2 % weight; nearest and
# +-5 ms keep frames whose label is up to 50 % (nearest) and 44 % (+-5 ms, Fregly; 27 % Moore) masked JR frames;
# 0.05 keeps 86 % (Moore) to 98 % (Han, Tiziana) with that bound.
LABEL_MASK_FILE = "label_mask.npz"
LABEL_MASK_KEY = "keep_r"      # the right knee: what the network predicts
LABEL_MASK_TOL = 0.05
for _base in ("qinst_notrans_pelvisrel_winpos_contact_w20_t100_sojr",
              "qinst_notrans_pelvisrel_winpos_contact_fpopt_w20_t100_sojr"):
    INPUT_SETS[_base + "qc"] = dict(INPUT_SETS[_base], mx_medial_sign=MX_SIGN_PHYSICAL, label_mask=LABEL_MASK_FILE,
                                    desc=INPUT_SETS[_base]["desc"] + ", physical M_x sign, QC frame mask "
                                         f"({LABEL_MASK_FILE}: {LABEL_MASK_KEY})")


def fmedial_term(fy, mx, height_m, mx_sign, alpha=0.5, d_m=0.045):
    """F_medial = alpha*|F_y| + mx_sign*M_x*H/d_m (BW) from F_y (BW) and the raw right-knee M_x (BW*H); torch tensors
    or numpy arrays. The F_medial loss term of train() and inference_posture_surrogate's f_medial."""
    return alpha * abs(fy) + (mx_sign * mx * height_m) / d_m


def label_mask_path(subj_dir, cfg):
    """<trial>/<label folder>/<mask file> of an input set with a label mask (cfg["label_mask_subdir"] overrides the
    folder: the evaluator scores stored labels on the frames of the SO+JR mask); None without one."""
    name = cfg.get("label_mask")
    if not name:
        return None
    return os.path.join(subj_dir, cfg.get("label_mask_subdir") or label_subdir_of(cfg), name)


def load_label_mask(subj_dir, cfg, label_time=None, key=None):
    """(time, keep) of an input set's label mask; key defaults to LABEL_MASK_KEY. FileNotFoundError when the file is
    missing, ValueError when its time column is not label_time (the label file's rows, given)."""
    path = label_mask_path(subj_dir, cfg)
    if path is None:
        raise ValueError("input set has no label_mask")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} is missing; write it with: python -I analysis/thesis/so_label_qc.py masks")
    with np.load(path) as z:
        t = np.asarray(z["time"], dtype=np.float64)
        keep = np.asarray(z[key or cfg.get("label_mask_key", LABEL_MASK_KEY)], dtype=bool)
    if len(t) != len(keep):
        raise ValueError(f"{path}: {len(t)} times but {len(keep)} mask frames")
    if label_time is not None:
        label_time = np.asarray(label_time, dtype=np.float64)
        if len(label_time) != len(t) or np.abs(label_time - t).max() > 1e-6:
            raise ValueError(f"{path} does not match the label file's time column ({len(t)} vs {len(label_time)} "
                             "rows); rewrite it with: python -I analysis/thesis/so_label_qc.py masks")
    return t, keep


def carry_label_mask(t_mask, keep, t_frames, t_grid=None, tol=LABEL_MASK_TOL):
    """Keep flags on the loader's frame grid from keep on the JR rows t_mask.

    t_frames: the frames load_subject interpolates the labels onto (IK frames inside the JR window); t_grid: the grid
    _build_com_features then resamples to (None: no resampling). The masked-frame indicator goes through the same two
    linear interpolations as the labels, so its value at a frame is the summed weight of masked JR frames in that
    frame's label; the frame is kept when it is <= tol."""
    bad = np.interp(np.asarray(t_frames, dtype=np.float64), t_mask, (~np.asarray(keep, bool)).astype(np.float64))
    if t_grid is not None:
        bad = np.interp(np.asarray(t_grid, dtype=np.float64), t_frames, bad)
    return bad <= tol + 1e-12


CONTACT_GRF_BW_THRESHOLD = 0.05   # vertical GRF (BW) above which a foot counts as in contact

# Trial-level label filters (train_mlp CLI: --max-peak, --max-moment, --flat-cv / --flat-dr, --no-flat-filter).
MAX_PEAK_BW = 6.0        # drop a trial whose right-knee force resultant exceeds this (BW) in any frame
MAX_MOMENT_BWH = 0.15    # drop a trial whose |M_x| exceeds this (BW*H) in any frame; healthy walking peaks < ~0.05
FLAT_CV = 0.25           # walking only: drop a trial whose resultant has coefficient of variation below this
FLAT_DR = 0.3            # ... or range (max - min, BW) below this (train_cnn.filter_flat_subjects)
# label_mask sets (*_sojrqc): the peak and moment filters are a trial-level proxy for SO-corrupted labels, which the
# frame mask already removes. Judged on every JR row they drop a trial for rows the mask has thrown away (dry build,
# 6 BW / 0.15 BW*H: 86 % of the kept frames lost, 4,147 of 4,938 dropped trials with kept frames failing on masked
# rows only; kept rows never exceed 8.5 BW or 0.065 BW*H). So for these sets (a) the filters are judged on the
# training targets only: the labels of the kept frames on the output grid, after the mask is carried; and (b) their
# default is off: MASKED_MAX_PEAK_BW / MASKED_MAX_MOMENT_BWH, far above any kept label. Judged on kept frames, 6 BW /
# 0.15 BW*H would still drop 696 trials and 17.1 % of the kept frames (Moore 676 trials: kept resultants reach 8.5 BW),
# a cut on the label value the QC deliberately leaves out (results/thesis/labels_qc/label_masks/dry_build_*/).
MASKED_MAX_PEAK_BW = 1000.0
MASKED_MAX_MOMENT_BWH = 1000.0


def label_filter_defaults(cfg, max_peak_bw=None, max_moment_bwh=None):
    """(max_peak_bw, max_moment_bwh) a run uses: the given values, else the input set's defaults (MAX_PEAK_BW /
    MAX_MOMENT_BWH; MASKED_MAX_PEAK_BW / MASKED_MAX_MOMENT_BWH, i.e. off, for a set with a label_mask)."""
    masked = bool(cfg.get("label_mask"))
    peak = (MASKED_MAX_PEAK_BW if masked else MAX_PEAK_BW) if max_peak_bw is None else float(max_peak_bw)
    moment = (MASKED_MAX_MOMENT_BWH if masked else MAX_MOMENT_BWH) if max_moment_bwh is None else float(max_moment_bwh)
    return peak, moment


def kept_labels_pass(labels, keep, max_peak_bw, max_moment_bwh=None):
    """The trial-level label filters of load_subject, judged on the rows of labels [T, >=3 (+M_x)] where keep is True:
    force resultant (cols 0-2, BW) <= max_peak_bw and, when max_moment_bwh is given, |M_x| (col 3, BW*H) <=
    max_moment_bwh. True when no row is kept (a fully masked trial is counted as such, not as a filter drop)."""
    lk = np.asarray(labels, dtype=np.float64)[np.asarray(keep, bool)]
    if not len(lk):
        return True
    if np.sqrt((lk[:, :3] ** 2).sum(axis=1)).max() > max_peak_bw:
        return False
    return not (max_moment_bwh is not None and lk.shape[1] > 3 and np.abs(lk[:, 3]).max() > max_moment_bwh)


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


def _load_com_features_aligned(subj_dir, target_time, foot_point=None):
    """Load com_features.npz for one trial and interpolate every channel onto
    target_time (in seconds). Returns dict of arrays each [T, 3] aligned to
    target_time, or None if features aren't cached.

    With foot_point, calcn_r_pos / calcn_l_pos come from that point's cache
    (foot_point_<x>_<y>_<z>.npz) instead of the calcn origins.
    """
    path = os.path.join(subj_dir, "com_features.npz")
    if not os.path.exists(path):
        return None
    sources = {key: path for key in ("com_pos", "com_vel", "calcn_r_pos", "calcn_l_pos")}
    if foot_point is not None:
        fp_path = os.path.join(subj_dir, foot_point_filename(foot_point))
        if not os.path.exists(fp_path):
            return None
        sources["calcn_r_pos"] = sources["calcn_l_pos"] = fp_path
    out, files = {}, {}
    for key, src in sources.items():
        f = files.setdefault(src, np.load(src))
        src_t, arr = f["time"], f[key]
        out[key] = np.column_stack([
            np.interp(target_time, src_t, arr[:, j]) for j in range(3)
        ]).astype(np.float32)
    return out


def _sto_time(path):
    """Time column of an OpenSim .mot/.sto file (header ends at 'endheader')."""
    with open(path) as f:
        for n, line in enumerate(f):
            if line.strip().lower() == "endheader":
                break
    return pd.read_csv(path, sep=r"\s+", skiprows=n + 1, usecols=["time"])["time"].to_numpy(dtype=np.float64)


def _common_time_grid(subj_dir, ik_time, jcf_subdir="jcf_output"):
    """The IK frames load_subject(native_time_grid=True) keeps: inside the overlap of the IK, GRF and JR time ranges."""
    grf_t = _sto_time(os.path.join(subj_dir, "grf_data.mot"))
    jcf_t = _sto_time(jr_label_path(subj_dir, jcf_subdir))
    t0, t1 = max(ik_time[0], grf_t[0], jcf_t[0]), min(ik_time[-1], grf_t[-1], jcf_t[-1])
    return ik_time[(ik_time >= t0) & (ik_time <= t1)]


def _resample(t_from, x, t_to):
    """Linear interpolation of the columns of x [T, C] from t_from onto t_to."""
    return np.column_stack([np.interp(t_to, t_from, x[:, j]) for j in range(x.shape[1])]).astype(np.float32)


def _build_com_features(subj_dir, cfg, max_peak_bw, min_length, predict_moment, return_meta=False,
                        max_moment_bwh=MAX_MOMENT_BWH):
    """Construct the (q, v_com, x_com - x_r, x_com - x_l, [mass, height]) feature
    matrix for one trial. Returns (features [T, per_frame_ch], labels [T, 4], mass).

    Uses load_subject with native_time_grid=True so the IK time grid is preserved
    end-to-end. Input sets with time_exact sample the COM/foot/pelvis windows on that same grid; legacy sets use a
    full-segment linspace (see the "_t100" note at INPUT_SETS). resample_hz puts everything on a fixed-rate grid.
    Labels come from cfg's label_subdir (default jcf_output/); the trial is dropped (None) when they exceed max_peak_bw
    (force resultant, BW) or max_moment_bwh (|M_x|, BW*H, only with predict_moment) in any frame. With a cfg label_mask
    "any frame" means any kept frame: the test runs on the labels of the frames label_keep keeps (kept_labels_pass), not
    on every JR row as load_subject does.
    return_meta=True adds a dict with the frame times, right-foot contact and rate, and label_keep: with a cfg
    label_mask (the *_sojrqc sets), which frames' labels pass the QC (carry_label_mask), else None. A missing or stale
    mask raises (FileNotFoundError / ValueError) instead of dropping the trial quietly.
    """
    import json as _json
    jcf_subdir = label_subdir_of(cfg)
    masked_set = bool(cfg.get("label_mask"))
    r = load_subject(subj_dir,
                     lower_body_only=True,
                     clean_features=True,
                     include_mass=True,
                     include_speed=False,           # not needed for this variant
                     jcf_subdir=jcf_subdir,
                     predict_moment=predict_moment,
                     # label_mask sets: the peak/moment tests run below, on the kept frames only
                     max_peak_bw=np.inf if masked_set else max_peak_bw,
                     max_moment_bwh=np.inf if masked_set else max_moment_bwh,
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
    if cfg.get("time_exact"):
        t_target = _common_time_grid(subj_dir, ik_time, jcf_subdir)   # the grid q, contact and labels are on
        if len(t_target) != T:
            return None
    else:   # legacy (pre-Nov 2026 checkpoints): full-segment grid, time-warped against q by up to the SO buffer
        t_target = np.linspace(ik_time[0], ik_time[-1], T).astype(np.float64)
    # Better alternative: pick t_target so its spacing equals median ik_dt,
    # then slice. But for COM interpolation, near-uniform spacing matters more
    # than exact alignment.

    com = _load_com_features_aligned(subj_dir, t_target, foot_point=cfg.get("foot_point"))
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
    fy_grf = inputs[:, [49, 52]] if inputs.shape[1] >= 53 else None   # right/left vertical GRF (BW)
    hz = cfg.get("resample_hz")
    label_keep = None
    if cfg.get("label_mask"):
        # the labels sit on the IK frames inside the JR window (load_subject native grid), whatever t_target is
        label_time = _sto_time(jr_label_path(subj_dir, jcf_subdir))
        t_lab = t_target if cfg.get("time_exact") else _common_time_grid(subj_dir, ik_time, jcf_subdir)
        if len(t_lab) != T:
            return None
        t_mask, keep_jr = load_label_mask(subj_dir, cfg, label_time=label_time)   # raises when missing or stale
        t_out = (t_target[0] + np.arange(int(np.floor((t_target[-1] - t_target[0]) * hz + 1e-9)) + 1) / hz
                 if hz else None)
        label_keep = carry_label_mask(t_mask, keep_jr, t_lab, t_out, tol=cfg.get("label_mask_tol", LABEL_MASK_TOL))
    if hz:
        # fixed-rate grid, so the 20-frame window always spans 20/hz seconds
        t_new = t_target[0] + np.arange(int(np.floor((t_target[-1] - t_target[0]) * hz + 1e-9)) + 1) / hz
        parts = [_resample(t_target, np.asarray(x, dtype=np.float64), t_new) for x in parts]
        labels = _resample(t_target, np.asarray(labels, dtype=np.float64), t_new)
        if fy_grf is not None:
            fy_grf = _resample(t_target, fy_grf.astype(np.float64), t_new)
        t_target, T = t_new, len(t_new)
    if label_keep is not None and len(label_keep) != T:
        raise AssertionError(f"{subj_dir}: label mask on {len(label_keep)} frames, features on {T}")
    if label_keep is not None and not kept_labels_pass(labels, label_keep, max_peak_bw,
                                                       max_moment_bwh if predict_moment else None):
        return None
    if cfg.get("include_contact"):
        if fy_grf is None:
            return None   # GRF channels unavailable; can't build contact flags
        fy_r = fy_grf[:, 0]
        fy_l = fy_grf[:, 1]
        r_contact = (np.abs(fy_r) > CONTACT_GRF_BW_THRESHOLD).astype(np.float32)[:, None]
        l_contact = (np.abs(fy_l) > CONTACT_GRF_BW_THRESHOLD).astype(np.float32)[:, None]
        parts += [r_contact, l_contact]                    # [T, ..., +2]

    if cfg["include_height"]:
        parts.append(np.full((T, 1), mass, dtype=np.float32))
        parts.append(np.full((T, 1), height_m, dtype=np.float32))

    if cfg.get("drop_pelvis_translation"):
        parts[0] = np.delete(parts[0], [3, 4, 5], axis=1)   # q16 → q13 (no pelvis_tx/ty/tz)
    feat = np.hstack(parts).astype(np.float32)
    assert feat.shape[1] == cfg["per_frame_ch"], \
        f"Feature width {feat.shape[1]} != expected {cfg['per_frame_ch']}"
    if return_meta:
        meta = dict(t=np.asarray(t_target, dtype=np.float64),
                    r_contact=None if fy_grf is None else np.abs(fy_grf[:, 0]) > CONTACT_GRF_BW_THRESHOLD,
                    fs=float(hz) if hz else None,
                    label_keep=label_keep)      # bool per frame with a label_mask (see carry_label_mask), else None
        return feat, labels.astype(np.float32), mass, meta
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
                 input_set="full", dq_method="central", max_moment_bwh=MAX_MOMENT_BWH):
        import json as _json
        cfg = INPUT_SETS[input_set]
        n_features = feature_count(input_set)
        is_com = cfg.get("uses_com", False)
        window = cfg.get("window", 1)
        chunks_in, chunks_out = [], []
        self.n_trials = len(subject_dirs)
        self.rejected_dirs = []    # trials the loaders dropped (label filters, non-finite labels, length, channels)
        # label_mask sets (*_sojrqc): rows whose centre frame fails the label QC are dropped after windowing
        masked = bool(cfg.get("label_mask"))
        self.label_masked = masked
        self.n_frames_unmasked = 0     # frames of the loaded trials before the mask
        self.fully_masked_dirs = []    # loaded trials whose mask keeps no frame (e.g. a fixed right-foot COP)

        for subj_dir in subject_dirs:
            # ─── COM-based variants take a separate code path ───
            if is_com:
                built = _build_com_features(subj_dir, cfg,
                                            max_peak_bw=max_peak_bw,
                                            min_length=min_length,
                                            predict_moment=predict_moment,
                                            max_moment_bwh=max_moment_bwh,
                                            return_meta=masked)
                if built is None:
                    self.rejected_dirs.append(subj_dir)
                    continue
                feat, labels, mass = built[:3]
                keep = built[3]["label_keep"] if masked else None
                self.n_frames_unmasked += len(labels)
                if window > 1:
                    if cfg.get("window_only_positions", False):
                        pelvis_cols = (3, 4, 5) if cfg.get("pelvis_per_window_centering") else None
                        feat = _windowize_positions_only(
                            feat, window, q_dim=cfg.get("q_dim", 16),
                            n_pos=cfg.get("n_pos", 9),
                            pelvis_t_cols=pelvis_cols,
                        )
                    else:
                        feat = _windowize(feat, window,
                                          replicate_q=cfg.get("replicate_q", False))
                if keep is not None:          # windows were built on the full trial; drop masked centre frames
                    if not keep.any():
                        self.fully_masked_dirs.append(subj_dir)
                        continue
                    feat, labels = feat[keep], labels[keep]
                chunks_in.append(feat)
                chunks_out.append(labels)
                continue

            # ─── Existing (q, F, ...) variants ───
            r = load_subject(subj_dir,
                             lower_body_only=True,
                             clean_features=True,
                             include_mass=True,
                             include_speed=include_speed,
                             jcf_subdir=label_subdir_of(cfg),
                             predict_moment=predict_moment,
                             max_peak_bw=max_peak_bw,
                             max_moment_bwh=max_moment_bwh,
                             min_length=min_length)
            if r is None:
                self.rejected_dirs.append(subj_dir)
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
                self.rejected_dirs.append(subj_dir)
                continue
            if height_m <= 0:
                self.rejected_dirs.append(subj_dir)
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

def _scan_dirs(data_root, dataset=None, exclude=None, label_subdir=DEFAULT_LABEL_SUBDIR):
    """Trial folders under data_root that have <label_subdir>/BatchJCF_JointReaction_ReactionLoads.sto."""
    excluded = set(exclude) if exclude else set()
    out = []
    for name in sorted(os.listdir(data_root)):
        if dataset and not name.startswith(f"{dataset}_"):
            continue
        if any(name.startswith(f"{ex}_") for ex in excluded):
            continue
        subj_dir = os.path.join(data_root, name)
        jcf_sto = os.path.join(subj_dir, label_subdir,
                               "BatchJCF_JointReaction_ReactionLoads.sto")
        if os.path.isdir(subj_dir) and os.path.exists(jcf_sto):
            out.append(subj_dir)
    return out


def _by_lab(dirs):
    """'carter 3, moore 12' for a list of trial folders (lab = folder-name prefix)."""
    counts = {}
    for d in dirs:
        lab = os.path.basename(d.rstrip("/")).split("_")[0]
        counts[lab] = counts.get(lab, 0) + 1
    return ", ".join(f"{lab} {n}" for lab, n in sorted(counts.items()))


def _partial_labels_error(where, missing, n_stored, label_subdir):
    """SystemExit for folders that have the stored jcf_output/ labels but none in label_subdir/."""
    return SystemExit(
        f"{where}: {len(missing)} of {n_stored} folders with {DEFAULT_LABEL_SUBDIR}/ labels have no {label_subdir}/ "
        f"labels ({_by_lab(missing)}; e.g. {os.path.basename(missing[0].rstrip('/'))}). Training on the rest would "
        f"change the trial set and the person split. Finish the relabel first (check: python "
        f"analysis/thesis/sojr_trainset.py --gate), or pass --allow-partial-labels to train on the folders that have "
        f"{label_subdir}/ labels.")


def _missing_mask_error(where, missing, n, label_subdir, label_mask):
    """SystemExit for labelled folders without the label-QC mask of a *_sojrqc set."""
    return SystemExit(
        f"{where}: {len(missing)} of {n} folders with {label_subdir}/ labels have no {label_subdir}/{label_mask} "
        f"({_by_lab(missing)}; e.g. {os.path.basename(missing[0].rstrip('/'))}). This input set trains only on frames "
        f"that pass the label QC; write the masks first: python -I analysis/thesis/so_label_qc.py masks --workers 20")


def _find_trials(data_root, dataset=None, exclude=None, label_subdir=DEFAULT_LABEL_SUBDIR,
                 flat_filter=False, flat_cv=FLAT_CV, flat_dr=FLAT_DR, flat_label_subdir=None,
                 allow_partial_labels=True, label_mask=None):
    """Labelled trial folders, without those whose labels hold a non-finite value, then (flat_filter) without those
    whose label waveform is flat.

    label_mask: mask file name inside label_subdir/ (the *_sojrqc sets): SystemExit when a labelled folder has none,
    before the flat filter (the mask is the label QC; training without it would silently train on rejected frames).

    flat_label_subdir: the label folder the flat filter judges (default: label_subdir). The stored jcf_output keeps
    the stored-label flat selection in an SO+JR run, so both runs start from the same trials and person split.
    allow_partial_labels=False: SystemExit (before the flat filter) when some folders have the stored jcf_output/
    labels but none in label_subdir/, as while a relabel is unfinished or has failed folders.

    Returns (dirs, scan); scan = dict(n_found, n_stored, missing, nonfinite, n_flat, flat_label_subdir):
    n_found / n_stored = folders with label_subdir/ / jcf_output/ labels, missing = folders with jcf_output/ labels
    but none in label_subdir/ ([] for the stored labels), nonfinite = folders dropped because their label_subdir/
    file has a NaN/inf (train_cnn.label_nonfinite_rows), n_flat = flat folders removed."""
    dirs = _scan_dirs(data_root, dataset=dataset, exclude=exclude, label_subdir=label_subdir)
    have = set(dirs)
    stored = dirs if label_subdir == DEFAULT_LABEL_SUBDIR else _scan_dirs(data_root, dataset=dataset, exclude=exclude)
    missing = [d for d in stored if d not in have]
    if missing and not allow_partial_labels:
        raise _partial_labels_error(data_root, missing, len(stored), label_subdir)
    nonfinite = [d for d in dirs if label_nonfinite_rows(d, label_subdir)]
    if nonfinite:
        bad = set(nonfinite)
        dirs = [d for d in dirs if d not in bad]
    if label_mask:
        no_mask = [d for d in dirs if not os.path.exists(os.path.join(d, label_subdir, label_mask))]
        if no_mask:
            raise _missing_mask_error(data_root, no_mask, len(dirs), label_subdir, label_mask)
    flat_sub = flat_label_subdir or label_subdir
    n_flat = 0
    if flat_filter:
        flat_bad = []      # non-finite in flat_sub/ (only possible when it differs from label_subdir)
        dirs, n_flat = filter_flat_subjects(dirs, cv_threshold=flat_cv, dr_threshold=flat_dr, jcf_subdir=flat_sub,
                                            nonfinite=flat_bad)
        n_flat -= len(flat_bad)
        nonfinite += flat_bad
    return dirs, dict(n_found=len(have), n_stored=len(stored), missing=missing, nonfinite=nonfinite, n_flat=n_flat,
                      flat_label_subdir=flat_sub if flat_filter else None)


def _label_config(input_set, max_peak_bw, max_moment_bwh, flat_filter, flat_cv, flat_dr, flat_label_subdir=None,
                  allow_partial_labels=False):
    """Which labels a run trained on and how they were filtered (stored in the checkpoint as plain data)."""
    cfg = INPUT_SETS[input_set]
    sub = label_subdir_of(cfg)
    mask = cfg.get("label_mask")
    return dict(label_subdir=sub,
                max_peak_bw=float(max_peak_bw), max_moment_bwh=float(max_moment_bwh),
                flat_filter=bool(flat_filter),
                flat_cv=float(flat_cv) if flat_filter else None, flat_dr=float(flat_dr) if flat_filter else None,
                flat_label_subdir=(flat_label_subdir or sub) if flat_filter else None,
                nonfinite_rejected=True, allow_partial_labels=bool(allow_partial_labels),
                label_mask=mask,                  # frame mask file in label_subdir/ (None: every frame is a target)
                label_mask_key=cfg.get("label_mask_key", LABEL_MASK_KEY) if mask else None,
                label_mask_tol=float(cfg.get("label_mask_tol", LABEL_MASK_TOL)) if mask else None)


def _subject_id(path):
    return re.sub(r"_t\d+(_r\d+)?$", "", os.path.basename(path))


def _split_dirs(subject_dirs, train_size, seed):
    unique = sorted(set(_subject_id(d) for d in subject_dirs))
    train_subj, val_subj = train_test_split(unique, train_size=train_size, random_state=seed)
    s = set(train_subj)
    train_dirs = [d for d in subject_dirs if _subject_id(d) in s]
    val_dirs = [d for d in subject_dirs if _subject_id(d) not in s]
    return train_dirs, val_dirs, len(train_subj), len(val_subj)


def _mask_stats(ds):
    """Label-mask frame counts of a FrameDataset ({} without a mask)."""
    if not getattr(ds, "label_masked", False):
        return {}
    return dict(frames_before_mask=int(ds.n_frames_unmasked), frames_kept=int(len(ds)),
                fully_masked_trials=sorted(os.path.basename(d) for d in ds.fully_masked_dirs))


def _mask_msg(ds):
    st = _mask_stats(ds)
    if not st:
        return ""
    fm = ds.fully_masked_dirs
    return (f"; label mask kept {st['frames_kept']:,} of {st['frames_before_mask']:,} frames "
            f"({st['frames_kept'] / max(st['frames_before_mask'], 1):.1%})"
            + (f", {len(fm)} trials fully masked ({_by_lab(fm)})" if fm else ""))


def train(activity='static', filter_flat=True, dataset=None, exclude=None,
          init_from=None, extra_dirs=None, extra_weight=1.0, extra_val_dirs=None,
          lr=None, tag=None,
          max_peak_bw=None, seed=SEED, input_set='full', dq_method='central',
          epochs=None, lr_patience=None, fmedial_weight=0.0, activation='relu',
          dropout=None, init_seed=None, max_moment_bwh=None, flat_cv=FLAT_CV, flat_dr=FLAT_DR,
          flat_label_subdir=None, allow_partial_labels=False, mx_sign_override=None, fmedial_loss="combined"):
    """
    Train MLP on per-frame data from one or both activities.
      activity='static':  ./jcf/full_duration/training/static/  (for posture optimization)
      activity='walking': ./jcf/full_duration/training/walking/ (for gait analysis)
      activity='both':    combined; uses WeightedRandomSampler to give static and
                          walking equal mass per epoch despite size imbalance.

      input_set: which subset of features the model receives. See INPUT_SETS. Its label_subdir (default
                 jcf_output) picks the label files, for trial discovery, the flat filter and the loaders alike.
      dq_method: 'central' (np.gradient, default) or 'forward' (William's
                 dq_k = (q_k - q_{k-1})/dt).
      seed: train/val split by person (and weight init when init_seed is None).
      init_seed: weight init and batch order only, with the split fixed by seed (run-to-run spread).
      Label filters (per trial, any frame): max_peak_bw (right-knee force resultant, BW), max_moment_bwh (|M_x|,
      BW*H); None = the input set's default (label_filter_defaults: 6 BW / 0.15 BW*H; off, 1000, for a label_mask
      set, where a given value is judged on the QC-kept frames only); filter_flat with activity='walking' only (not
      'both'): drop trials whose resultant has CV < flat_cv or range < flat_dr BW, judged on flat_label_subdir's labels (default: the run's own). A trial whose labels
      hold a non-finite value is always dropped (before the split).
      allow_partial_labels: by default a run stops when some folders have the stored jcf_output/ labels but none in
      the input set's label folder (unfinished or partly failed relabel); True trains on the folders that have them.
      mx_sign_override: 'physical' (-mx) or 'pipeline' (+mx) for the F_medial loss term instead of the input set's
      own sign (sign ablations); stored in the checkpoint like any sign, and a run whose sign differs from the set's
      default gets _mxplus / _mxminus in its file name.
      fmedial_loss: 'combined' (default) = MSE of F_medial = 0.5|F_y| + s*M_x*H/d; 'signfree' = the same expansion
      without the sign-carrying cross term, 0.25*(|F_y| - |F_y_true|)^2 + ((M_x - M_x_true)*H/d)^2, i.e. per-output
      errors with M_x weighted at the scale it enters F_medial (weight fmedial_weight; file name gets _sf).
    """
    _call_args = dict(locals())          # every train() argument, defaults included (provenance)
    filters_given = (max_peak_bw is not None, max_moment_bwh is not None)
    max_peak_bw, max_moment_bwh = label_filter_defaults(INPUT_SETS[input_set], max_peak_bw, max_moment_bwh)
    _call_args.update(max_peak_bw=max_peak_bw, max_moment_bwh=max_moment_bwh)   # as resolved
    torch.manual_seed(seed if init_seed is None else init_seed)
    np.random.seed(seed if init_seed is None else init_seed)
    n_features = feature_count(input_set)
    print(f"Input set: {input_set} — {INPUT_SETS[input_set]['desc']}  ({n_features} ch)")
    print(f"dq method: {dq_method}")

    # Static is essentially flat by construction — skip the flat-waveform filter.
    apply_flat_filter = filter_flat and activity == 'walking'
    label_cfg = _label_config(input_set, max_peak_bw, max_moment_bwh, apply_flat_filter, flat_cv, flat_dr,
                              flat_label_subdir, allow_partial_labels)
    label_subdir = label_cfg["label_subdir"]
    label_mask = label_cfg["label_mask"]
    mx_sign = mx_medial_sign_of(INPUT_SETS[input_set])
    if fmedial_loss not in ("combined", "signfree"):
        raise ValueError("fmedial_loss must be 'combined' or 'signfree'")
    mx_sign_suffix = ""
    if mx_sign_override is not None:
        if mx_sign_override not in MX_CONVENTIONS:
            raise ValueError(f"mx_sign_override must be one of {sorted(MX_CONVENTIONS)}")
        if MX_CONVENTIONS[mx_sign_override] != mx_sign:
            mx_sign_suffix = "_mxplus" if MX_CONVENTIONS[mx_sign_override] > 0 else "_mxminus"
        mx_sign = MX_CONVENTIONS[mx_sign_override]
        if fmedial_weight <= 0:
            print("WARNING: --mx-sign has no effect without --fmedial-weight > 0 (the sign only enters the F_medial "
                  "loss term)")
    print(f"Labels: {label_subdir}/BatchJCF_JointReaction_ReactionLoads.sto; trial filters: resultant peak <= "
          f"{max_peak_bw:g} BW, |M_x| <= {max_moment_bwh:g} BW*H, non-finite labels dropped, "
          + (f"flat filter on walking (drop CV < {flat_cv:g} or range < {flat_dr:g} BW of the "
             f"{label_cfg['flat_label_subdir']}/ labels)" if apply_flat_filter else "no flat filter")
          + (f"; frame mask {label_subdir}/{label_mask} ({label_cfg['label_mask_key']}, tol "
             f"{label_cfg['label_mask_tol']:g})" if label_mask else ""))
    if label_mask:
        off = (max_peak_bw >= MASKED_MAX_PEAK_BW, max_moment_bwh >= MASKED_MAX_MOMENT_BWH)
        given = [f"--{n}" for n, g in zip(("max-peak", "max-moment"), filters_given) if g]
        print("  label-mask set: the peak/moment filters are judged on the QC-kept frames only; "
              + ("both off" if all(off) else
                 f"peak {'off' if off[0] else f'{max_peak_bw:g} BW'}, moment "
                 f"{'off' if off[1] else f'{max_moment_bwh:g} BW*H'}")
              + (f" ({', '.join(given)} given)" if given else
                 f" (default {MASKED_MAX_PEAK_BW:g} / {MASKED_MAX_MOMENT_BWH:g} for label-mask sets: the frame mask "
                 "replaces them)"))
    if fmedial_weight > 0 and fmedial_loss == "signfree":
        print(f"F_medial loss (sign-free, weight {fmedial_weight:g}): 0.25*(|F_y|-|F_y_true|)^2 + "
              f"((M_x-M_x_true)*H/0.045)^2; checkpoint sign for inference {mx_sign:+g}")
    elif fmedial_weight > 0:
        print(f"F_medial loss: 0.5*|F_y| + ({mx_sign:+g})*M_x*H/0.045, M_x = raw walker_knee_r mx / (BW*H) ("
              + ("physical medial sign)" if mx_sign == MX_SIGN_PHYSICAL else
                 "legacy pipeline sign; physically the medial moment is -mx at walker_knee_r)"))

    # Explicitly passed trial folders (fine-tuning) obey the same rule as the split: stop when some have the stored
    # labels but not the run's (checked first, so a partial relabel fails before the flat filter runs).
    for kind, ds_dirs in (("external", extra_dirs), ("external-val", extra_val_dirs)):
        nolab = [d for d in (ds_dirs or []) if os.path.isdir(d) and not os.path.exists(
            jr_label_path(d, label_subdir))]
        partial = [d for d in nolab if label_subdir != DEFAULT_LABEL_SUBDIR and os.path.exists(
            os.path.join(d, DEFAULT_LABEL_SUBDIR, "BatchJCF_JointReaction_ReactionLoads.sto"))]
        if partial and not allow_partial_labels:
            n_stored = sum(os.path.exists(os.path.join(d, DEFAULT_LABEL_SUBDIR, "BatchJCF_JointReaction_ReactionLoads.sto"))
                           for d in ds_dirs if os.path.isdir(d))
            raise _partial_labels_error(f"--{'extra-dirs' if kind == 'external' else 'extra-val-dirs'}", partial,
                                        n_stored, label_subdir)
        if nolab:
            print(f"[{kind}] WARNING: {len(nolab)} dirs have no {label_subdir}/ labels and will be skipped "
                  f"(e.g. {nolab[0]})")
        if label_mask:
            lab_dirs = [d for d in (ds_dirs or []) if os.path.isdir(d) and d not in nolab]
            no_mask = [d for d in lab_dirs if not os.path.exists(os.path.join(d, label_subdir, label_mask))]
            if no_mask:
                raise _missing_mask_error(f"--{'extra-dirs' if kind == 'external' else 'extra-val-dirs'}", no_mask,
                                          len(lab_dirs), label_subdir, label_mask)

    # Collect train/val dirs per activity
    train_dirs_by = {}
    val_dirs_by = {}
    discovery = {}
    output_root = CKPT_ROOT
    os.makedirs(CKPT_ROOT, exist_ok=True)
    for act in (['static', 'walking'] if activity == 'both' else [activity]):
        root = DATA_ROOTS[act]
        if not os.path.isdir(root):
            print(f"[{act}] root does not exist: {root}")
            continue
        flat_here = apply_flat_filter and act == 'walking'
        dirs, scan = _find_trials(root, dataset=dataset, exclude=exclude, label_subdir=label_subdir,
                                  flat_filter=flat_here, flat_cv=flat_cv, flat_dr=flat_dr,
                                  flat_label_subdir=flat_label_subdir, allow_partial_labels=allow_partial_labels,
                                  label_mask=label_mask)
        n0 = scan["n_found"]
        print(f"[{act}] found {n0} subject directories with {label_subdir}/ labels")
        if scan["missing"]:
            print(f"[{act}] WARNING (--allow-partial-labels): {len(scan['missing'])} of {scan['n_stored']} folders "
                  f"with {DEFAULT_LABEL_SUBDIR}/ labels have no {label_subdir}/ labels ({_by_lab(scan['missing'])}); "
                  f"skipped, so the trial set and the person split differ from a complete-label run")
        if scan["nonfinite"]:
            print(f"[{act}] {len(scan['nonfinite'])} folders have non-finite label values; dropped "
                  f"({_by_lab(scan['nonfinite'])}; e.g. {os.path.basename(scan['nonfinite'][0])})")
        if flat_here:
            n_in = len(dirs) + scan["n_flat"]
            print(f"[{act}] flat filter ({scan['flat_label_subdir']}/ labels): {n_in} → {len(dirs)} "
                  f"(removed {scan['n_flat']})")
        discovery[act] = dict(n_found=n0, n_stored=scan["n_stored"], n_missing_labels=len(scan["missing"]),
                              missing_by_lab=_by_lab(scan["missing"]),
                              nonfinite=sorted(os.path.basename(d) for d in scan["nonfinite"]),
                              n_flat_removed=scan["n_flat"], flat_label_subdir=scan["flat_label_subdir"],
                              n_trials=len(dirs))
        td, vd, ns_train, ns_val = _split_dirs(dirs, TRAIN_SPLIT, seed)
        train_dirs_by[act] = td
        val_dirs_by[act] = vd
        print(f"[{act}] train: {ns_train} subjects ({len(td)} trials), "
              f"val: {ns_val} subjects ({len(vd)} trials)")

    # Foot-point variants: every trial with com_features.npz must also have the foot-point
    # cache, or the trial set would silently differ from the calcn-origin model's.
    fp = INPUT_SETS[input_set].get("foot_point")
    if fp is not None:
        fp_name = foot_point_filename(fp)
        missing = [d for dirs in list(train_dirs_by.values()) + list(val_dirs_by.values()) for d in dirs
                   if os.path.exists(os.path.join(d, "com_features.npz"))
                   and not os.path.exists(os.path.join(d, fp_name))]
        if missing:
            raise SystemExit(f"{len(missing)} trials lack {fp_name} (e.g. {missing[0]}). Cache it first:\n"
                             f"    conda run -n jcf python pipeline/compute_com_features.py --foot-point "
                             f"{' '.join('%g' % v for v in fp)} --workers 8")
        print(f"Foot point: {fp} in the calcn frames ({fp_name})")

    # Fine-tuning: external labelled trials (e.g. jcf/external/<traj>_seg1, made by
    # pipeline/label_trajectory_csv.py) join training as their own sampling bucket,
    # bypassing the flat filter and the subject split.
    if extra_dirs:
        extra_dirs = [d.rstrip("/") for d in extra_dirs if os.path.isdir(d)]
        train_dirs_by["external"] = extra_dirs
        print(f"[external] {len(extra_dirs)} extra trial dirs (bucket weight {extra_weight:g})")

    if not train_dirs_by:
        print("No training data found.")
        return

    print("\nLoading training frames...")
    train_ds_per = {}
    loaded = {}    # (split, bucket) -> trials FrameDataset dropped
    masked_stats = {}   # (split, bucket) -> label-mask frame counts (label_mask sets)
    for act, td in train_dirs_by.items():
        ds = FrameDataset(td, max_peak_bw=max_peak_bw,
                          input_set=input_set, dq_method=dq_method, max_moment_bwh=max_moment_bwh)
        train_ds_per[act] = ds
        loaded[("train", act)] = ds.rejected_dirs
        masked_stats[("train", act)] = _mask_stats(ds)
        print(f"  [{act}] {len(ds):,} training frames from {ds.n_trials - len(ds.rejected_dirs)} of {ds.n_trials} "
              f"trials" + (f"; {len(ds.rejected_dirs)} dropped by the loaders (peak/moment filters, length, "
                           f"channels: {_by_lab(ds.rejected_dirs)})" if ds.rejected_dirs else "")
              + _mask_msg(ds))

    # Combine if more than one bucket (both activities and/or external trials).
    # Each main bucket has total sampling weight 1; the external bucket has extra_weight.
    if len(train_ds_per) > 1:
        all_in = np.concatenate([d.frames_in for d in train_ds_per.values()], axis=0)
        all_out = np.concatenate([d.frames_out for d in train_ds_per.values()], axis=0)
        sample_weights = np.concatenate([
            np.full(len(d), (extra_weight if act == "external" else 1.0) / max(len(d), 1))
            for act, d in train_ds_per.items()
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

    # Compute and apply normalization (from the combined train set, or, when
    # fine-tuning, reuse the checkpoint's so the feature space stays identical).
    init_ckpt = None
    if init_from:
        init_ckpt = torch.load(init_from, map_location="cpu", weights_only=False)
        if init_ckpt.get("input_set") != input_set:
            raise ValueError(f"--init-from checkpoint has input_set={init_ckpt.get('input_set')!r}, "
                             f"but --input-set is {input_set!r}")
        if int(init_ckpt["n_features"]) != n_features:
            raise ValueError(f"--init-from checkpoint has {init_ckpt['n_features']} features, expected {n_features}")
        inp_mean = init_ckpt["input_mean"].float()
        inp_std = init_ckpt["input_std"].float()
        print(f"Fine-tuning from {init_from} (epoch {init_ckpt.get('epoch')}, val {init_ckpt.get('val_loss', float('nan')):.6f}); "
              f"normalization taken from the checkpoint")
    else:
        inp_mean = torch.from_numpy(train_ds.frames_in.mean(axis=0))
        inp_std = torch.from_numpy(train_ds.frames_in.std(axis=0)).clamp(min=1e-8)
    print(f"\n  input_mean range: [{inp_mean.min():.2f}, {inp_mean.max():.2f}]")
    train_ds.frames_in = (train_ds.frames_in - inp_mean.numpy()) / inp_std.numpy()

    print("\nLoading validation frames...")
    val_pieces_in, val_pieces_out = [], []
    for act, vd in val_dirs_by.items():
        ds = FrameDataset(vd, max_peak_bw=max_peak_bw,
                          input_set=input_set, dq_method=dq_method, max_moment_bwh=max_moment_bwh)
        ds.frames_in = (ds.frames_in - inp_mean.numpy()) / inp_std.numpy()
        val_pieces_in.append(ds.frames_in)
        val_pieces_out.append(ds.frames_out)
        loaded[("val", act)] = ds.rejected_dirs
        masked_stats[("val", act)] = _mask_stats(ds)
        print(f"  [{act}] {len(ds):,} validation frames from {ds.n_trials - len(ds.rejected_dirs)} of {ds.n_trials} "
              f"trials" + (f"; {len(ds.rejected_dirs)} dropped by the loaders ({_by_lab(ds.rejected_dirs)})"
                           if ds.rejected_dirs else "") + _mask_msg(ds))
    val_ds = FrameDataset.__new__(FrameDataset)
    val_ds.frames_in = np.concatenate(val_pieces_in, axis=0)
    val_ds.frames_out = np.concatenate(val_pieces_out, axis=0)

    ext_val_loader = None
    if extra_val_dirs:
        ext_val_dirs = [d.rstrip("/") for d in extra_val_dirs if os.path.isdir(d)]
        ext_ds = FrameDataset(ext_val_dirs, max_peak_bw=max_peak_bw,
                              input_set=input_set, dq_method=dq_method, max_moment_bwh=max_moment_bwh)
        ext_ds.frames_in = (ext_ds.frames_in - inp_mean.numpy()) / inp_std.numpy()
        print(f"  [external-val] {len(ext_ds):,} frames from {len(ext_val_dirs)} dirs "
              f"(reported each epoch; checkpoint selection still uses the main val loss)")
        ext_val_loader = DataLoader(ext_ds, batch_size=BATCH_SIZE, shuffle=False,
                                    num_workers=0, pin_memory=True)

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
    if init_ckpt is not None:
        if init_ckpt.get("activation", "relu") != activation:
            raise ValueError(f"--init-from checkpoint uses activation={init_ckpt.get('activation')!r}; pass --activation to match")
        model.load_state_dict(init_ckpt["model_state_dict"])
        print("  weights loaded from --init-from")

    n_epochs = epochs if epochs is not None else EPOCHS
    sched_patience = lr_patience if lr_patience is not None else 3
    effective_lr = LEARNING_RATE if lr is None else lr
    optimizer = torch.optim.Adam(model.parameters(), lr=effective_lr,
                                 weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=sched_patience, factor=0.5)

    provenance = _prov.build(
        call_args=_call_args,
        hyperparameters=dict(
            learning_rate=effective_lr, weight_decay=WEIGHT_DECAY, batch_size=BATCH_SIZE,
            hidden=HIDDEN, dropout=effective_dropout, epochs=n_epochs, lr_patience=sched_patience,
            train_split=TRAIN_SPLIT, contact_grf_bw_threshold=CONTACT_GRF_BW_THRESHOLD,
            n_features=n_features, data_roots=DATA_ROOTS, labels=label_cfg, mx_medial_sign=mx_sign),
        splits={act: dict(train_subjects=_prov.subject_ids(train_dirs_by.get(act, []), _subject_id),
                          val_subjects=_prov.subject_ids(val_dirs_by.get(act, []), _subject_id),
                          n_train_trials=len(train_dirs_by.get(act, [])),
                          n_val_trials=len(val_dirs_by.get(act, [])),
                          train_trials_dropped_by_loaders=sorted(
                              os.path.basename(d) for d in loaded.get(("train", act), [])),
                          val_trials_dropped_by_loaders=sorted(
                              os.path.basename(d) for d in loaded.get(("val", act), [])))
                for act in sorted(set(train_dirs_by) | set(val_dirs_by))},
        extra=dict(n_train_frames=len(train_ds.frames_in), n_val_frames=len(val_ds.frames_in),
                   n_params=n_params, discovery=discovery,
                   label_mask={f"{sp}/{act}": v for (sp, act), v in masked_stats.items() if v}))
    _prov.announce(provenance)

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
            # explicit F_medial = alpha*|F_y| + (s*M_x*H)/d_m MSE term so the
            # optimizer sees the lever-arm-amplified M_x error directly.
            # s = the input set's mx_medial_sign (-1 physical, +1 legacy pipeline; see MX_SIGN_PHYSICAL).
            H = inp[:, -1] * height_std_t + height_mean_t                  # [B]
            fy_p, mx_p = pred[:, 1], pred[:, 3]
            fy_t, mx_t = target[:, 1], target[:, 3]
            if fmedial_loss == "signfree":
                # per-output terms of the F_medial error without its sign-carrying cross term
                fm_loss = ((ALPHA_FMEDIAL * (torch.abs(fy_p) - torch.abs(fy_t))) ** 2
                           + ((mx_p - mx_t) * H / D_M_FMEDIAL) ** 2).mean()
            else:
                fm_p = fmedial_term(fy_p, mx_p, H, mx_sign, ALPHA_FMEDIAL, D_M_FMEDIAL)
                fm_t = fmedial_term(fy_t, mx_t, H, mx_sign, ALPHA_FMEDIAL, D_M_FMEDIAL)
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

        ext_msg = ""
        if ext_val_loader is not None:
            e_loss, n_e = 0.0, 0
            with torch.no_grad():
                for inp, lbl in ext_val_loader:
                    inp = inp.to(DEVICE); lbl = lbl.to(DEVICE)
                    e_loss += loss_fn(model(inp), lbl, inp).item() * len(inp); n_e += len(inp)
            ext_msg = f" | ext-val: {e_loss / max(n_e, 1):.6f}"

        cur_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1:3d}/{n_epochs} | train: {train_loss:.6f} | "
              f"val: {val_loss:.6f}{ext_msg} | lr: {cur_lr:.1e}")

        if val_loss < best_val:
            best_val = val_loss
            # Default input_set keeps original filename; add suffix for variants
            suffix = "" if input_set == "full" else f"_{input_set}"
            if dq_method != "central":
                suffix += f"_{dq_method}"
            if fmedial_weight > 0:
                # e.g. fmW2 for fmedial_weight=2.0
                w = ("%g" % fmedial_weight).replace(".", "p")
                suffix += f"_fmW{w}" + ("_sf" if fmedial_loss == "signfree" else "")
            if activation != 'relu':
                suffix += f"_{activation}"
            suffix += mx_sign_suffix
            if tag:
                suffix += f"_{tag}"
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
                "fmedial_loss": fmedial_loss,
                "activation": activation,
                "base_indices": INPUT_SETS[input_set]["base_indices"],
                "include_height": INPUT_SETS[input_set]["include_height"],
                "include_bmi": INPUT_SETS[input_set]["include_bmi"],
                "foot_point": list(INPUT_SETS[input_set].get("foot_point", (0.0, 0.0, 0.0))),
                "label_subdir": label_subdir,      # labels the run trained on (tools/evaluate_fmedial.py reads it)
                "label_filters": {k: v for k, v in label_cfg.items() if k != "label_subdir"},
                # factor on the raw right-knee mx in F_medial (loss term and inference_posture_surrogate.f_medial);
                # checkpoints without the key used the legacy +1
                "mx_medial_sign": mx_sign,
                "label_mask": label_mask,
                "init_from": init_from,
                "extra_dirs": extra_dirs,
                "extra_weight": extra_weight,
                "tag": tag,
                "lr": effective_lr,
                "provenance": provenance,
            }, os.path.join(output_root, ckpt_name))
            print(f"  → saved (val={val_loss:.6f}) → {output_root}/{ckpt_name}")

    print(f"\nBest val loss: {best_val:.6f}")
    print(f"Saved to {output_root}/{ckpt_name if best_val < float('inf') else '(nothing saved)'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--activity", type=str, default="static",
                   choices=["static", "walking", "both"],
                   help="Which dataset(s) to train on. 'static' (default) for "
                        "posture optimization. 'walking' for gait. 'both' uses "
                        "weighted sampling so static and walking get equal mass.")
    p.add_argument("--no-flat-filter", action="store_true",
                   help="Disable the flat-waveform filter. It only runs with --activity walking (not static, not "
                        "both) and drops a walking trial whose right-knee force resultant has CV < --flat-cv or "
                        "range < --flat-dr BW.")
    p.add_argument("--flat-cv", type=float, default=FLAT_CV,
                   help=f"Flat filter: coefficient-of-variation threshold (default {FLAT_CV}).")
    p.add_argument("--flat-dr", type=float, default=FLAT_DR,
                   help=f"Flat filter: range (max - min) threshold in BW (default {FLAT_DR}).")
    p.add_argument("--flat-labels", type=str, default=None, metavar="SUBDIR",
                   help="Label folder the flat filter judges (default: the input set's own labels). "
                        f"--flat-labels {DEFAULT_LABEL_SUBDIR} in an SO+JR run keeps the stored-label flat selection, "
                        "so the run starts from the same trials and person split as a stored-label run.")
    p.add_argument("--allow-partial-labels", action="store_true",
                   help="Train on the folders that have the input set's labels even when some folders have only "
                        f"the stored {DEFAULT_LABEL_SUBDIR}/ labels (default: stop with the per-lab count).")
    p.add_argument("--dataset", type=str, default=None,
                   help="Restrict to one source dataset prefix (e.g. carter).")
    p.add_argument("--exclude", type=str, nargs="+", default=None,
                   help="Exclude source dataset prefixes (e.g. han fregly).")
    p.add_argument("--max-peak", type=float, default=None,
                   help="Reject trials whose right-knee force resultant exceeds this many BW in any frame "
                        f"(default {MAX_PEAK_BW}; for the *_sojrqc label-mask sets the default is off "
                        f"({MASKED_MAX_PEAK_BW:g}) and a given value is judged on the QC-kept frames only).")
    p.add_argument("--max-moment", type=float, default=None,
                   help="Reject trials whose |M_x| (knee adduction moment) exceeds this many BW*H in any frame "
                        f"(default {MAX_MOMENT_BWH}; healthy walking peaks below ~0.05; *_sojrqc sets: off "
                        f"({MASKED_MAX_MOMENT_BWH:g}) by default, judged on the QC-kept frames when given).")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--init-seed", type=int, default=None,
                   help="seed for weight init and batch order only; the person split stays fixed by --seed")
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
                        "F_medial = 0.5*|F_y| + (s*M_x*H)/0.045 (BW), s = the input set's mx_medial_sign "
                        "(+1 legacy pipeline sign; -1, the physical medial sign, for the *_sojrqc sets). This gives "
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
    p.add_argument("--init-from", type=str, default=None,
                   help="Fine-tune: start from this checkpoint (weights + input normalization).")
    p.add_argument("--extra-dirs", type=str, nargs="+", default=None,
                   help="Extra labelled trial dirs (e.g. jcf/external/*_seg1) added to training "
                        "as their own sampling bucket; not subject-split, not flat-filtered.")
    p.add_argument("--extra-weight", type=float, default=1.0,
                   help="Sampling weight of the external bucket relative to each main bucket "
                        "(each main bucket = 1). 1.0 with --activity both ⇒ 1/3 of samples.")
    p.add_argument("--extra-val-dirs", type=str, nargs="+", default=None,
                   help="Extra labelled trial dirs reported as a separate 'ext-val' loss each epoch.")
    p.add_argument("--lr", type=float, default=None,
                   help=f"Learning rate (default {LEARNING_RATE}); use ~1e-4 when fine-tuning.")
    p.add_argument("--fmedial-loss", choices=["combined", "signfree"], default="combined",
                   help="combined: MSE of F_medial = 0.5|F_y| + s*M_x*H/d (default). signfree: "
                        "0.25*(|F_y|-|F_y_true|)^2 + ((M_x-M_x_true)*H/d)^2, the same terms without the sign-carrying "
                        "cross term (file name gets _sf).")
    p.add_argument("--mx-sign", choices=sorted(MX_CONVENTIONS), default=None,
                   help="Moment sign in the F_medial loss term, overriding the input set's: physical = -mx (the "
                        "medial moment at walker_knee_r), pipeline = +mx (ISRR 2026). For sign ablations; a run whose "
                        "sign differs from the set's default gets _mxplus / _mxminus in its file name.")
    p.add_argument("--tag", type=str, default=None,
                   help="Suffix appended to the checkpoint name (e.g. ft_sep23) so the base run is not overwritten.")
    args = p.parse_args()
    train(activity=args.activity,
          filter_flat=not args.no_flat_filter,
          dataset=args.dataset, exclude=args.exclude,
          max_peak_bw=args.max_peak, max_moment_bwh=args.max_moment,
          flat_cv=args.flat_cv, flat_dr=args.flat_dr,
          flat_label_subdir=args.flat_labels.strip("/") if args.flat_labels else None,
          allow_partial_labels=args.allow_partial_labels,
          seed=args.seed, init_seed=args.init_seed,
          input_set=args.input_set, dq_method=args.dq_method,
          epochs=args.epochs, lr_patience=args.lr_patience,
          fmedial_weight=args.fmedial_weight,
          activation=args.activation,
          dropout=args.dropout,
          init_from=args.init_from, extra_dirs=args.extra_dirs,
          extra_weight=args.extra_weight, extra_val_dirs=args.extra_val_dirs,
          lr=args.lr, tag=args.tag, mx_sign_override=args.mx_sign, fmedial_loss=args.fmedial_loss)

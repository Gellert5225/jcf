"""
1D CNN Training Script for Knee JCF Prediction
================================================
Trains a 1D CNN to predict knee joint contact forces (Fx, Fy, Fz)
from joint kinematics + GRF.

Inputs (per frame):
    - Joint positions (37 DOFs)
    - GRF forces (6: calcn_r xyz + calcn_l xyz)
    Total: 43 features per frame

Labels (per frame):
    - Knee JCF Fx, Fy, Fz in tibia frame (Newtons, normalized by BW)

Data source:
    - jcf/training/<subject>/ik_results.mot       (positions)
    - jcf/training/<subject>/grf_data.mot          (GRF)
    - jcf/training/<subject>/jcf_output/BatchJCF_JointReaction_ReactionLoads.sto (labels)
    - jcf/training/<subject>/metadata.json         (mass)

Usage:
    conda run -n jcf python train_cnn.py
    conda run -n jcf python train_cnn.py --exp a   # symmetric loss + new arch
    conda run -n jcf python train_cnn.py --exp b   # log-space targets + MSE
"""

import os
import json
import glob
import argparse
import numpy as np
import pandas as pd
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# ─── Config ──────────────────────────────────────────────────────────────────

SEED = 42
DATA_ROOT = "./jcf/full_duration/training/running"
BATCH_SIZE = 8          # full sequences per batch (pad to max length)
EPOCHS = 300
LEARNING_RATE = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_SPLIT = 0.8       # 80% train, 20% val (by subject)
N_LOWER_BODY = 20       # joints 0-19: pelvis, hips, knees, ankles, feet

# Pelvis translations (absolute lab position) — non-stationary, must be zero-centered per trial
PELVIS_TX_COLS = [3, 4, 5]  # pelvis_tx, pelvis_ty, pelvis_tz
# Locked/dead joints — near-zero variance across training data
DEAD_JOINT_COLS = [11, 12, 18, 19]  # subtalar_angle_r, mtp_angle_r, subtalar_angle_l, mtp_angle_l


# ─── Data Quality Filter ──────────────────────────────────────────────────────

def filter_flat_subjects(subject_dirs, cv_threshold=0.25, dr_threshold=0.3,
                         jcf_subdir='jcf_output'):
    """
    Remove subjects whose JCF resultant waveform is flat (no cyclic peaks).
    A subject is 'flat' if its coefficient of variation < cv_threshold
    OR its dynamic range < dr_threshold (in BW).

    Returns (kept_dirs, n_removed).
    """
    kept = []
    removed = 0
    for subj_dir in subject_dirs:
        jcf_path = os.path.join(subj_dir, jcf_subdir,
                                'BatchJCF_JointReaction_ReactionLoads.sto')
        meta_path = os.path.join(subj_dir, 'metadata.json')
        if not os.path.exists(jcf_path) or not os.path.exists(meta_path):
            kept.append(subj_dir)
            continue
        with open(meta_path) as f:
            mass = json.load(f)['mass_kg']
        BW = mass * 9.81
        jcf = load_sto(jcf_path)
        fx = jcf['walker_knee_r_on_tibia_r_in_tibia_r_fx'].values
        fy = jcf['walker_knee_r_on_tibia_r_in_tibia_r_fy'].values
        fz = jcf['walker_knee_r_on_tibia_r_in_tibia_r_fz'].values
        resultant = np.sqrt(fx**2 + fy**2 + fz**2) / BW
        cv = resultant.std() / (resultant.mean() + 1e-12)
        dr = resultant.max() - resultant.min()
        if cv < cv_threshold or dr < dr_threshold:
            removed += 1
        else:
            kept.append(subj_dir)
    return kept, removed


def get_clean_subjects(subject_dirs, max_multi_sat_frames=0):
    """
    Filter subjects to those with clean SO convergence.
    A subject is 'clean' if it has <= max_multi_sat_frames frames where
    >5 muscles simultaneously hit the activation upper bound (>=0.999).

    Returns filtered list of subject_dirs.
    """
    clean = []
    for subj_dir in subject_dirs:
        act_file = os.path.join(subj_dir, 'jcf_output',
                                'BatchJCF_StaticOptimization_activation.sto')
        if not os.path.exists(act_file):
            continue
        # Parse .sto header
        with open(act_file) as f:
            header_lines = 0
            for line in f:
                header_lines += 1
                if line.strip() == 'endheader':
                    break
        df = pd.read_csv(act_file, sep=r'\s+', skiprows=header_lines)
        muscle_cols = [c for c in df.columns
                       if c != 'time' and not c.startswith('reserve_')
                       and not c.startswith('calcn_')]
        if not muscle_cols:
            continue
        vals = df[muscle_cols].values
        # Count frames where >5 muscles are simultaneously at bound
        n_multi_sat = int((np.sum(vals >= 0.999, axis=1) > 5).sum())
        if n_multi_sat <= max_multi_sat_frames:
            clean.append(subj_dir)
    return clean


def load_confidence_weights(subject_dir):
    """
    Compute per-frame confidence weights from SO activation saturation.
    Frames where many muscles are at their activation bound (>=0.999)
    get lower confidence. Returns array of shape [T] with values in (0, 1].
    """
    act_file = os.path.join(subject_dir, 'jcf_output',
                            'BatchJCF_StaticOptimization_activation.sto')
    if not os.path.exists(act_file):
        return None
    with open(act_file) as f:
        header_lines = 0
        for line in f:
            header_lines += 1
            if line.strip() == 'endheader':
                break
    df = pd.read_csv(act_file, sep=r'\s+', skiprows=header_lines)
    muscle_cols = [c for c in df.columns
                   if c != 'time' and not c.startswith('reserve_')
                   and not c.startswith('calcn_')]
    if not muscle_cols:
        return None
    vals = df[muscle_cols].values
    n_muscles = len(muscle_cols)
    # Fraction of muscles at bound per frame
    sat_fraction = np.sum(vals >= 0.999, axis=1) / n_muscles  # [T], range [0, 1]
    # Confidence: 1.0 for clean frames, decays toward 0.2 for fully saturated
    confidence = 1.0 - 0.8 * sat_fraction
    return confidence.astype(np.float32)


# ─── Data Loading ─────────────────────────────────────────────────────────────

def load_mot(path, skiprows=6):
    """Load an OpenSim .mot file."""
    return pd.read_csv(path, sep='\t', skiprows=skiprows)


def load_sto(path, skiprows=11):
    """Load an OpenSim .sto file."""
    return pd.read_csv(path, sep='\t', skiprows=skiprows)


def load_subject(subject_dir, lower_body_only=False, with_confidence=False,
                 jcf_subdir='jcf_output', clean_features=False, include_mass=False,
                 use_root_features=False, combine_root_features=False):
    """
    Load one subject's data. Returns (inputs, labels, mass) or None.

    inputs: [T, n_features]  (positions + GRF)
    labels: [T, 3]           (Fx, Fy, Fz in BW)

    clean_features: if True, zero-center pelvis translations per trial
                    and remove dead joints (subtalar, mtp).
    use_root_features: if True, use root-frame features from .b3d instead
                       of IK-derived features. Requires root_features.npy.
    """
    ik_path = os.path.join(subject_dir, 'ik_results.mot')
    grf_path = os.path.join(subject_dir, 'grf_data.mot')
    jcf_path = os.path.join(subject_dir, jcf_subdir,
                            'BatchJCF_JointReaction_ReactionLoads.sto')
    meta_path = os.path.join(subject_dir, 'metadata.json')

    if not all(os.path.exists(p) for p in [ik_path, grf_path, jcf_path, meta_path]):
        return None

    with open(meta_path) as f:
        meta = json.load(f)
    mass = meta['mass_kg']
    BW = mass * 9.81

    # Load kinematics (joint positions)
    ik = load_mot(ik_path)
    ik_time = ik['time'].values
    ik_data = ik.drop(columns=['time']).values  # [T, 37]

    # Load GRF (forces only — 6 columns: calcn_r xyz + calcn_l xyz)
    grf = load_mot(grf_path)
    force_cols = [c for c in grf.columns
                  if ('calcn_r' in c or 'calcn_l' in c) and '_force_v' in c]
    if len(force_cols) != 6:
        return None
    grf_time = grf['time'].values
    grf_data = grf[force_cols].values / BW  # normalize by BW

    # Load JCF labels
    jcf = load_sto(jcf_path)
    jcf_time = jcf['time'].values
    fx = jcf['walker_knee_r_on_tibia_r_in_tibia_r_fx'].values
    fy = jcf['walker_knee_r_on_tibia_r_in_tibia_r_fy'].values
    fz = jcf['walker_knee_r_on_tibia_r_in_tibia_r_fz'].values
    jcf_data = np.column_stack([fx, fy, fz]) / BW  # normalize by BW

    # Filter out subjects with unreasonable JCF (bad SO output)
    peak_resultant = np.sqrt((jcf_data ** 2).sum(axis=1)).max()
    if peak_resultant > 10.0:
        return None

    # Align by time: find overlapping time range
    t_start = max(ik_time[0], grf_time[0], jcf_time[0])
    t_end = min(ik_time[-1], grf_time[-1], jcf_time[-1])

    # Interpolate everything to JCF timestamps (they're the most sparse)
    jcf_mask = (jcf_time >= t_start) & (jcf_time <= t_end)
    t_common = jcf_time[jcf_mask]
    labels = jcf_data[jcf_mask]

    # Interpolate IK and GRF to jcf timestamps
    ik_interp = np.column_stack([
        np.interp(t_common, ik_time, ik_data[:, j])
        for j in range(ik_data.shape[1])
    ])
    grf_interp = np.column_stack([
        np.interp(t_common, grf_time, grf_data[:, j])
        for j in range(grf_data.shape[1])
    ])

    # Compute velocity (1st derivative) and acceleration (2nd derivative)
    # Pass t_common directly to np.gradient for correct time scaling
    ik_vel = np.gradient(ik_interp, t_common, axis=0)
    grf_vel = np.gradient(grf_interp, t_common, axis=0)
    ik_acc = np.gradient(ik_vel, t_common, axis=0)

    # Optionally select only lower-body joints (0-19)
    if lower_body_only:
        ik_interp = ik_interp[:, :N_LOWER_BODY]
        ik_vel = ik_vel[:, :N_LOWER_BODY]
        ik_acc = ik_acc[:, :N_LOWER_BODY]

    if clean_features:
        # Zero-center pelvis translations per trial (remove absolute lab position)
        ik_interp[:, PELVIS_TX_COLS] -= ik_interp[:, PELVIS_TX_COLS].mean(axis=0)
        # Remove dead joints (subtalar, mtp) from positions, velocities, accelerations
        n_joints = ik_interp.shape[1]
        keep = [j for j in range(n_joints) if j not in DEAD_JOINT_COLS]
        ik_interp = ik_interp[:, keep]
        ik_vel = ik_vel[:, keep]
        ik_acc = ik_acc[:, keep]

    # Build input features
    if use_root_features:
        root_path = os.path.join(subject_dir, 'root_features.npy')
        if not os.path.exists(root_path):
            return None
        root_data = np.load(root_path)  # [T_b3d, 81]
        # root_data has same frame count as IK; interpolate to JCF timestamps
        root_time = ik_time[:len(root_data)]
        root_interp = np.column_stack([
            np.interp(t_common, root_time, root_data[:, j])
            for j in range(root_data.shape[1])
        ])
        # Normalize GRF-in-root-frame columns (last 6 before COM acc) by BW
        # Layout: joint_centers(60) + root_dynamics(12) + grf_root(6) + com_acc(3) = 81
        root_interp[:, 72:78] /= BW
        parts = [root_interp]
    else:
        parts = [ik_interp, ik_vel, ik_acc, grf_interp, grf_vel]
    if combine_root_features:
        root_path = os.path.join(subject_dir, 'root_features.npy')
        if not os.path.exists(root_path):
            return None
        root_data = np.load(root_path)
        root_time = ik_time[:len(root_data)]
        # Only take root dynamics (60:81): root vel/acc (12) + GRF in root frame (6) + COM acc (3)
        root_dyn = root_data[:, 60:]
        root_dyn_interp = np.column_stack([
            np.interp(t_common, root_time, root_dyn[:, j])
            for j in range(root_dyn.shape[1])
        ])
        root_dyn_interp[:, 12:18] /= BW  # GRF-in-root-frame columns
        parts.append(root_dyn_interp)
    if include_mass:
        parts.append(np.full((len(t_common), 1), mass))
    inputs = np.hstack(parts)

    if with_confidence:
        conf = load_confidence_weights(subject_dir)
        if conf is not None:
            # Align confidence to jcf timestamps (same length)
            conf = conf[jcf_mask] if len(conf) == len(jcf_mask) else np.ones(len(labels), dtype=np.float32)
        else:
            conf = np.ones(len(labels), dtype=np.float32)
        return inputs, labels, mass, conf

    return inputs, labels, mass


# ─── Dataset ──────────────────────────────────────────────────────────────────

class JCFDataset(Dataset):
    """Full-sequence dataset. Each item is one subject's entire trial."""

    def __init__(self, subject_dirs, lower_body_only=False, with_confidence=False,
                 jcf_subdir='jcf_output', clean_features=False, include_mass=False,
                 use_root_features=False, combine_root_features=False):
        self.sequences = []  # list of (inputs, labels, length) or (inputs, labels, length, confidence)
        self.has_confidence = with_confidence

        for subj_dir in subject_dirs:
            result = load_subject(subj_dir, lower_body_only=lower_body_only,
                                 with_confidence=with_confidence,
                                 jcf_subdir=jcf_subdir, clean_features=clean_features,
                                 include_mass=include_mass,
                                 use_root_features=use_root_features,
                                 combine_root_features=combine_root_features)
            if result is None:
                continue
            if with_confidence:
                inputs, labels, mass, conf = result
                T = len(labels)
                self.sequences.append((
                    torch.tensor(inputs, dtype=torch.float32),
                    torch.tensor(labels, dtype=torch.float32),
                    T,
                    torch.tensor(conf, dtype=torch.float32),
                ))
            else:
                inputs, labels, mass = result
                T = len(labels)
                self.sequences.append((
                    torch.tensor(inputs, dtype=torch.float32),
                    torch.tensor(labels, dtype=torch.float32),
                    T,
                ))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]

    def normalize(self, mean, std):
        """Apply z-score normalization to all input sequences."""
        if self.has_confidence:
            self.sequences = [
                ((inp - mean) / std, lbl, length, conf)
                for inp, lbl, length, conf in self.sequences
            ]
        else:
            self.sequences = [
                ((inp - mean) / std, lbl, length)
                for inp, lbl, length in self.sequences
            ]

    def transform_labels_log(self):
        """Transform labels to signed log-space: sign(y)*log(1+|y|)."""
        if self.has_confidence:
            self.sequences = [
                (inp, torch.sign(lbl) * torch.log1p(torch.abs(lbl)), length, conf)
                for inp, lbl, length, conf in self.sequences
            ]
        else:
            self.sequences = [
                (inp, torch.sign(lbl) * torch.log1p(torch.abs(lbl)), length)
                for inp, lbl, length in self.sequences
            ]


def collate_fn(batch):
    """Pad sequences to the max length in the batch + return masks."""
    has_conf = len(batch[0]) == 4
    if has_conf:
        inputs_list, labels_list, lengths, conf_list = zip(*batch)
    else:
        inputs_list, labels_list, lengths = zip(*batch)
    max_len = max(lengths)
    n_feat = inputs_list[0].shape[1]

    padded_inputs = torch.zeros(len(batch), max_len, n_feat)
    padded_labels = torch.zeros(len(batch), max_len, 3)
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    padded_conf = torch.ones(len(batch), max_len)  # default confidence = 1.0

    for i in range(len(batch)):
        L = lengths[i]
        padded_inputs[i, :L] = inputs_list[i]
        padded_labels[i, :L] = labels_list[i]
        mask[i, :L] = True
        if has_conf:
            padded_conf[i, :L] = conf_list[i]

    return padded_inputs, padded_labels, mask, padded_conf


# ─── Model ────────────────────────────────────────────────────────────────────

class ResBlock1d(nn.Module):
    """Residual block with dilated conv + GroupNorm."""
    def __init__(self, channels, kernel_size=5, dilation=1, dropout=0.0):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        layers = [
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            nn.GroupNorm(8, channels),
            nn.ReLU(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers += [
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            nn.GroupNorm(8, channels),
        ]
        self.block = nn.Sequential(*layers)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.block(x))


class CausalConv1d(nn.Module):
    """Conv1d with left-only (causal) padding — output[t] only sees input[<=t]."""
    def __init__(self, in_ch, out_ch, kernel_size, dilation=1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=0, dilation=dilation)

    def forward(self, x):
        x = nn.functional.pad(x, (self.pad, 0))
        return self.conv(x)


class ChannelLayerNorm(nn.Module):
    """LayerNorm over channels at each timestep. Preserves causality
    (GroupNorm would normalize across time and leak future into past)."""
    def __init__(self, num_channels):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels)

    def forward(self, x):  # x: [N, C, T]
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class CausalResBlock1d(nn.Module):
    """Residual block with causal dilated conv + per-timestep channel norm."""
    def __init__(self, channels, kernel_size=5, dilation=1, dropout=0.0):
        super().__init__()
        layers = [
            CausalConv1d(channels, channels, kernel_size, dilation=dilation),
            ChannelLayerNorm(channels),
            nn.ReLU(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers += [
            CausalConv1d(channels, channels, kernel_size, dilation=dilation),
            ChannelLayerNorm(channels),
        ]
        self.block = nn.Sequential(*layers)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.block(x))


class JCF_CNN(nn.Module):
    """
    1D CNN for predicting knee JCF from kinematics + GRF.
    
    Input:  [batch, window_size, n_features]
    Output: [batch, window_size, 3]  (Fx, Fy, Fz per frame)
    """

    def __init__(self, n_features=43, n_outputs=3):
        super().__init__()

        # Conv1d expects [batch, channels, length]
        # Dilated convolutions for large receptive field (~200+ frames)
        self.encoder = nn.Sequential(
            nn.Conv1d(n_features, 64, kernel_size=7, padding=3),
            nn.GroupNorm(8, 64),
            nn.ReLU(),

            nn.Conv1d(64, 128, kernel_size=5, padding=4, dilation=2),
            nn.GroupNorm(8, 128),
            nn.ReLU(),

            nn.Conv1d(128, 128, kernel_size=5, padding=8, dilation=4),
            nn.GroupNorm(8, 128),
            nn.ReLU(),

            nn.Conv1d(128, 64, kernel_size=3, padding=4, dilation=4),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
        )

        # Per-frame output head
        self.head = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(32, n_outputs, kernel_size=1),
        )

    def forward(self, x):
        # x: [batch, window_size, n_features]
        x = x.permute(0, 2, 1)  # → [batch, n_features, window_size]
        x = self.encoder(x)      # → [batch, 64, window_size]
        x = self.head(x)         # → [batch, 3, window_size]
        x = x.permute(0, 2, 1)  # → [batch, window_size, 3]
        return x


class JCF_CNN_v2(nn.Module):
    """
    Wider 1D CNN with residual blocks for better peak tracking.
    
    Input:  [batch, seq_len, n_features]
    Output: [batch, seq_len, 3]
    """

    def __init__(self, n_features=43, n_outputs=3):
        super().__init__()

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Conv1d(n_features, 128, kernel_size=7, padding=3),
            nn.GroupNorm(8, 128),
            nn.ReLU(),
        )

        # Residual blocks with increasing dilation
        self.res_blocks = nn.Sequential(
            ResBlock1d(128, kernel_size=5, dilation=1),
            ResBlock1d(128, kernel_size=5, dilation=2),
            ResBlock1d(128, kernel_size=5, dilation=4),
            ResBlock1d(128, kernel_size=5, dilation=8),
        )

        # Per-frame output head
        self.head = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(64, n_outputs, kernel_size=1),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)    # → [batch, n_features, seq_len]
        x = self.input_proj(x)     # → [batch, 128, seq_len]
        x = self.res_blocks(x)     # → [batch, 128, seq_len]
        x = self.head(x)           # → [batch, 3, seq_len]
        x = x.permute(0, 2, 1)    # → [batch, seq_len, 3]
        return x


class JCF_CNN_v2_causal(nn.Module):
    """
    Causal variant of v2: each frame only sees past frames.
    For online/streaming inference at 1000Hz.
    Receptive field ≈ 4*(5-1)*(1+2+4+8) = 240 frames lookback.
    """

    def __init__(self, n_features=43, n_outputs=3):
        super().__init__()
        self.input_proj = nn.Sequential(
            CausalConv1d(n_features, 128, kernel_size=7),
            ChannelLayerNorm(128),
            nn.ReLU(),
        )
        self.res_blocks = nn.Sequential(
            CausalResBlock1d(128, kernel_size=5, dilation=1),
            CausalResBlock1d(128, kernel_size=5, dilation=2),
            CausalResBlock1d(128, kernel_size=5, dilation=4),
            CausalResBlock1d(128, kernel_size=5, dilation=8),
        )
        self.head = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(64, n_outputs, kernel_size=1),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.input_proj(x)
        x = self.res_blocks(x)
        x = self.head(x)
        x = x.permute(0, 2, 1)
        return x


class JCF_CNN_v3(nn.Module):
    """Wider CNN with more residual blocks and dropout for higher-dimensional input."""

    def __init__(self, n_features=43, n_outputs=3, dropout=0.15):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Conv1d(n_features, 256, kernel_size=7, padding=3),
            nn.GroupNorm(8, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.res_blocks = nn.Sequential(
            ResBlock1d(256, kernel_size=5, dilation=1, dropout=dropout),
            ResBlock1d(256, kernel_size=5, dilation=2, dropout=dropout),
            ResBlock1d(256, kernel_size=5, dilation=4, dropout=dropout),
            ResBlock1d(256, kernel_size=5, dilation=8, dropout=dropout),
            ResBlock1d(256, kernel_size=5, dilation=16, dropout=dropout),
            ResBlock1d(256, kernel_size=5, dilation=1, dropout=dropout),
        )
        self.head = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(128, n_outputs, kernel_size=1),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.input_proj(x)
        x = self.res_blocks(x)
        x = self.head(x)
        x = x.permute(0, 2, 1)
        return x


class TCNBlock(nn.Module):
    """Single TCN block: dilated causal conv + residual."""
    def __init__(self, channels, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)
        self.dropout = nn.Dropout(dropout)
        self.padding = padding
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: [B, C, T]
        residual = x
        out = self.conv1(x)
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        out = self.norm1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        out = self.norm2(out)
        out = self.relu(out)
        out = self.dropout(out)
        return self.relu(out + residual)


class JCF_TCN(nn.Module):
    """
    Temporal Convolutional Network with skip connections from each dilation level.
    Uses causal convolutions with dilation factors [1,2,4,8,16,32,64] for ~384-frame RF.

    Input:  [batch, seq_len, n_features]
    Output: [batch, seq_len, 3]
    """

    def __init__(self, n_features=60, n_outputs=3, hidden=128, kernel_size=3,
                 dilations=(1, 2, 4, 8, 16, 32, 64), dropout=0.1):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Conv1d(n_features, hidden, kernel_size=1),
            nn.GroupNorm(8, hidden),
            nn.ReLU(),
        )
        self.tcn_blocks = nn.ModuleList([
            TCNBlock(hidden, kernel_size=kernel_size, dilation=d, dropout=dropout)
            for d in dilations
        ])
        self.skip_proj = nn.ModuleList([
            nn.Conv1d(hidden, hidden // 2, kernel_size=1)
            for _ in dilations
        ])
        self.head = nn.Sequential(
            nn.Conv1d(hidden // 2 * len(dilations) + hidden, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(64, n_outputs, kernel_size=1),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)  # [B, F, T]
        x = self.input_proj(x)   # [B, hidden, T]
        skips = []
        for block, skip in zip(self.tcn_blocks, self.skip_proj):
            x = block(x)
            skips.append(skip(x))
        out = torch.cat([x] + skips, dim=1)  # [B, hidden + hidden//2 * n_levels, T]
        out = self.head(out)
        return out.permute(0, 2, 1)  # [B, T, 3]


class JCF_FFT_MLP(nn.Module):
    """
    Per-frame MLP with windowed FFT features.
    For each frame, extracts a local window, computes FFT magnitude spectrum,
    and concatenates with the raw features as input to an MLP.

    Input:  [batch, seq_len, n_features]
    Output: [batch, seq_len, 3]
    """

    def __init__(self, n_features=60, n_outputs=3, window_size=64,
                 hidden=256, dropout=0.1):
        super().__init__()
        self.window_size = window_size
        self.n_features = n_features
        n_fft_bins = window_size // 2 + 1
        self.n_fft_bins = n_fft_bins
        fft_features = n_features * n_fft_bins
        total_input = n_features + fft_features
        self.mlp = nn.Sequential(
            nn.Linear(total_input, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, n_outputs),
        )

    def forward(self, x):
        # x: [B, T, F]
        B, T, F = x.shape
        half_w = self.window_size // 2
        x_padded = torch.nn.functional.pad(x, (0, 0, half_w, half_w - 1), mode='reflect')
        windows = x_padded.unfold(1, self.window_size, 1)  # [B, T, F, W]
        fft_out = torch.fft.rfft(windows, dim=-1)  # [B, T, F, W//2+1]
        fft_mag = torch.log1p(fft_out.abs())  # [B, T, F, n_fft_bins]
        fft_flat = fft_mag.reshape(B, T, F * self.n_fft_bins)
        mlp_input = torch.cat([x, fft_flat], dim=-1)  # [B, T, F + F*n_fft_bins]
        return self.mlp(mlp_input)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, d_model, max_len=1000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class JCF_Transformer(nn.Module):
    """
    Conv stem + Transformer encoder for JCF prediction.
    Self-attention provides global trial context for better peak tracking.

    Input:  [batch, seq_len, n_features]
    Output: [batch, seq_len, 3]
    """

    def __init__(self, n_features=72, n_outputs=3, d_model=128, nhead=8,
                 num_layers=4, dim_feedforward=256, dropout=0.1):
        super().__init__()

        # Conv stem: extract local temporal features
        self.conv_stem = nn.Sequential(
            nn.Conv1d(n_features, d_model, kernel_size=7, padding=3),
            nn.GroupNorm(8, d_model),
            nn.ReLU(),
            nn.Conv1d(d_model, d_model, kernel_size=5, padding=2),
            nn.GroupNorm(8, d_model),
            nn.ReLU(),
        )

        self.pos_enc = PositionalEncoding(d_model, max_len=1000, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, n_outputs),
        )

    def forward(self, x, mask=None):
        # x: [B, T, F]
        x = x.permute(0, 2, 1)      # [B, F, T]
        x = self.conv_stem(x)        # [B, d_model, T]
        x = x.permute(0, 2, 1)      # [B, T, d_model]
        x = self.pos_enc(x)

        src_key_padding_mask = ~mask if mask is not None else None
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)

        x = self.head(x)             # [B, T, n_outputs]
        return x


# ─── Training ─────────────────────────────────────────────────────────────────

def train(exp=None, filter_flat=True):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    EXP = exp
    lower_body = EXP in ('c', 'd', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't')
    use_v2 = EXP in ('d', 'g', 'i', 'j', 'k', 'n', 'o', 'p', 'q')
    use_v2_causal = EXP in ('r', 's', 't')
    use_v3 = False
    use_transformer = EXP == 'h'
    use_tcn = EXP == 'l'
    use_fft_mlp = EXP == 'm'
    rebalance = EXP in ('d', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't')
    clean_only = EXP == 'e'
    use_confidence = EXP == 'f'
    clean_feats = EXP in ('i', 'j', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 't')
    mass_input = EXP in ('l', 'n', 'o', 'p', 'q', 'r', 's', 't')
    root_feats = EXP == 'o'
    combine_root = EXP == 'p'
    lookahead = 10 if EXP in ('s', 't') else 0
    scaled_labels = EXP in ('g', 'h')  # use 2x muscle-scaled SO labels
    # Find all subjects with JCF output
    jcf_subdir = 'jcf_output_2x' if scaled_labels else 'jcf_output'
    subject_dirs = []
    for name in sorted(os.listdir(DATA_ROOT)):
        subj_dir = os.path.join(DATA_ROOT, name)
        jcf_sto = os.path.join(subj_dir, jcf_subdir,
                               'BatchJCF_JointReaction_ReactionLoads.sto')
        if os.path.isdir(subj_dir) and os.path.exists(jcf_sto):
            subject_dirs.append(subj_dir)
    if scaled_labels:
        print(f"Using scaled JCF labels from {jcf_subdir}/")
    if clean_feats:
        print("Clean features: pelvis translations zero-centered, dead joints removed")

    print(f"Found {len(subject_dirs)} subjects with JCF data")
    if len(subject_dirs) == 0:
        print("No data! Run batch_process.py first.")
        return

    # Filter out flat (non-cyclic) waveforms
    if filter_flat:
        n_before = len(subject_dirs)
        subject_dirs, n_flat = filter_flat_subjects(subject_dirs, jcf_subdir=jcf_subdir)
        print(f"Flat-waveform filter: {n_before} → {len(subject_dirs)} subjects ({n_flat} flat removed)")

    # Exp E: filter to clean SO convergence subjects only
    if clean_only:
        n_before = len(subject_dirs)
        subject_dirs = get_clean_subjects(subject_dirs, max_multi_sat_frames=0)
        print(f"Clean-subjects filter: {n_before} → {len(subject_dirs)} subjects")
        if len(subject_dirs) == 0:
            print("No clean subjects found!")
            return

    # Split by subject (not by window) to avoid data leakage
    train_dirs, val_dirs = train_test_split(
        subject_dirs, train_size=TRAIN_SPLIT, random_state=42
    )
    print(f"Train: {len(train_dirs)} subjects, Val: {len(val_dirs)} subjects")

    # Create datasets
    print("Loading training data...")
    train_ds = JCFDataset(train_dirs, lower_body_only=lower_body,
                          with_confidence=use_confidence,
                          jcf_subdir=jcf_subdir, clean_features=clean_feats,
                          include_mass=mass_input, use_root_features=root_feats,
                          combine_root_features=combine_root)
    print(f"  {len(train_ds)} training sequences")

    # Compute global normalization stats from training data
    all_frames = torch.cat([s[0] for s in train_ds.sequences], dim=0)  # [N, F]
    input_mean = all_frames.mean(dim=0)  # [F]
    input_std = all_frames.std(dim=0).clamp(min=1e-8)  # [F]
    del all_frames
    print(f"  Input normalization computed (mean range: [{input_mean.min():.2f}, {input_mean.max():.2f}])")

    # Normalize training data
    train_ds.normalize(input_mean, input_std)

    print("Loading validation data...")
    val_ds = JCFDataset(val_dirs, lower_body_only=lower_body,
                        with_confidence=use_confidence,
                        jcf_subdir=jcf_subdir, clean_features=clean_feats,
                        include_mass=mass_input, use_root_features=root_feats,
                        combine_root_features=combine_root)
    val_ds.normalize(input_mean, input_std)
    print(f"  {len(val_ds)} validation sequences")

    # Exp B: transform labels to log-space
    if EXP == 'b':
        train_ds.transform_labels_log()
        val_ds.transform_labels_log()
        print("  Labels transformed to signed log-space")

    if len(train_ds) == 0:
        print("No training sequences. Check data.")
        return

    # Dataset rebalancing: oversample minority datasets
    if rebalance:
        from torch.utils.data import WeightedRandomSampler
        # Compute per-sample weight based on dataset source
        dataset_counts = {}
        sample_datasets = []
        for d in train_dirs:
            name = os.path.basename(d)
            # Extract dataset prefix (e.g. 'carter', 'hammer', 'moore')
            prefix = name.split('_')[0]
            dataset_counts[prefix] = dataset_counts.get(prefix, 0) + 1
            sample_datasets.append(prefix)
        # Only count samples that made it into the dataset (load_subject can return None)
        # Re-derive from train_dirs that successfully loaded
        loaded_prefixes = []
        for d in train_dirs:
            result = load_subject(d, lower_body_only=lower_body, jcf_subdir=jcf_subdir,
                                 clean_features=clean_feats, include_mass=mass_input,
                                 use_root_features=root_feats,
                                 combine_root_features=combine_root)
            if result is not None:
                loaded_prefixes.append(os.path.basename(d).split('_')[0])
        dataset_counts_loaded = {}
        for p in loaded_prefixes:
            dataset_counts_loaded[p] = dataset_counts_loaded.get(p, 0) + 1
        weights = [1.0 / dataset_counts_loaded[p] for p in loaded_prefixes]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        print(f"  Rebalancing: {dataset_counts_loaded}")
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                                  num_workers=0, pin_memory=True,
                                  collate_fn=collate_fn)
    else:
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=0, pin_memory=True,
                                  collate_fn=collate_fn)

    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, pin_memory=True,
                            collate_fn=collate_fn)

    # Compute input feature count from first sample
    n_features = train_ds[0][0].shape[1]
    print(f"Input features per frame: {n_features}")

    # Create model
    if use_tcn:
        model = JCF_TCN(n_features=n_features, n_outputs=3).to(DEVICE)
        print("Using JCF_TCN (temporal convolutional network, dilations 1-64)")
    elif use_fft_mlp:
        model = JCF_FFT_MLP(n_features=n_features, n_outputs=3).to(DEVICE)
        print("Using JCF_FFT_MLP (windowed FFT features + MLP)")
    elif use_transformer:
        model = JCF_Transformer(n_features=n_features, n_outputs=3).to(DEVICE)
        print("Using JCF_Transformer (conv stem + transformer encoder)")
    elif use_v3:
        model = JCF_CNN_v3(n_features=n_features, n_outputs=3).to(DEVICE)
        print("Using JCF_CNN_v3 (256ch, 6 res blocks, dropout 0.15)")
    elif use_v2_causal:
        model = JCF_CNN_v2_causal(n_features=n_features, n_outputs=3).to(DEVICE)
        print("Using JCF_CNN_v2_causal (causal residual blocks — online/streaming)")
    elif use_v2:
        model = JCF_CNN_v2(n_features=n_features, n_outputs=3).to(DEVICE)
        print("Using JCF_CNN_v2 (residual blocks, wider)")
    else:
        model = JCF_CNN(n_features=n_features, n_outputs=3).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    print(f"Device: {DEVICE}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5
    )

    def masked_mse(preds, labels, mask, confidence=None):
        mask_3d = mask.unsqueeze(-1)
        return ((preds - labels) ** 2 * mask_3d).sum() / (mask.sum().float() * 3)

    def symmetric_loss(preds, labels, mask, confidence=None):
        """Magnitude-weighted MSE + gradient matching."""
        mask_3d = mask.unsqueeze(-1)
        n_valid = mask.sum().float() * 3
        # Base MSE
        mse = ((preds - labels) ** 2 * mask_3d).sum() / n_valid
        # Magnitude-weighted MSE (emphasize high-force frames)
        mag = torch.sqrt((labels ** 2).sum(dim=-1, keepdim=True)).detach()
        weighted = ((preds - labels) ** 2 * mag * mask_3d).sum() / ((mag * mask_3d).sum() + 1e-8)
        # Gradient loss (temporal shape matching)
        pred_grad = preds[:, 1:] - preds[:, :-1]
        label_grad = labels[:, 1:] - labels[:, :-1]
        grad_mask = (mask[:, 1:] & mask[:, :-1]).unsqueeze(-1)
        grad_loss = ((pred_grad - label_grad) ** 2 * grad_mask).sum() / (grad_mask.sum().float() * 3 + 1e-8)
        return mse + 0.5 * weighted + 0.3 * grad_loss

    def confidence_weighted_loss(preds, labels, mask, confidence=None):
        """Symmetric loss weighted by per-frame SO confidence."""
        mask_3d = mask.unsqueeze(-1)
        conf_3d = confidence.unsqueeze(-1) if confidence is not None else torch.ones_like(mask_3d, dtype=torch.float32)
        w = mask_3d * conf_3d  # [B, T, 1]
        n_valid = w.sum() * 3 + 1e-8
        # Confidence-weighted MSE
        mse = ((preds - labels) ** 2 * w).sum() / n_valid
        # Confidence-weighted magnitude term
        mag = torch.sqrt((labels ** 2).sum(dim=-1, keepdim=True)).detach()
        weighted = ((preds - labels) ** 2 * mag * w).sum() / ((mag * w).sum() + 1e-8)
        # Gradient loss (shape matching, also confidence-weighted)
        pred_grad = preds[:, 1:] - preds[:, :-1]
        label_grad = labels[:, 1:] - labels[:, :-1]
        grad_w = (mask[:, 1:] & mask[:, :-1]).unsqueeze(-1).float()
        if confidence is not None:
            grad_conf = torch.min(confidence[:, 1:], confidence[:, :-1]).unsqueeze(-1)
            grad_w = grad_w * grad_conf
        grad_loss = ((pred_grad - label_grad) ** 2 * grad_w).sum() / (grad_w.sum() * 3 + 1e-8)
        return mse + 0.5 * weighted + 0.3 * grad_loss

    def expectile_loss(preds, labels, mask, tau=0.7, confidence=None):
        """Expectile loss — τ>0.5 biases predictions toward upper tail."""
        err = labels - preds
        weight = torch.where(err > 0, tau, 1.0 - tau)
        return (weight * err ** 2 * mask.unsqueeze(-1)).sum() / (mask.sum().float() * 3)

    def log_magnitude_loss(preds, labels, mask, confidence=None):
        """Like symmetric_loss but with flattened magnitude weighting:
        1 + log(1 + mag) instead of raw mag. Gives intermediate peaks
        meaningful weight instead of deprioritizing them."""
        mask_3d = mask.unsqueeze(-1)
        n_valid = mask.sum().float() * 3
        mse = ((preds - labels) ** 2 * mask_3d).sum() / n_valid
        mag = torch.sqrt((labels ** 2).sum(dim=-1, keepdim=True)).detach()
        log_mag = 1.0 + torch.log1p(mag)
        weighted = ((preds - labels) ** 2 * log_mag * mask_3d).sum() / ((log_mag * mask_3d).sum() + 1e-8)
        pred_grad = preds[:, 1:] - preds[:, :-1]
        label_grad = labels[:, 1:] - labels[:, :-1]
        grad_mask = (mask[:, 1:] & mask[:, :-1]).unsqueeze(-1)
        grad_loss = ((pred_grad - label_grad) ** 2 * grad_mask).sum() / (grad_mask.sum().float() * 3 + 1e-8)
        return mse + 0.5 * weighted + 0.3 * grad_loss

    def log_magnitude_peak_loss(preds, labels, mask, confidence=None):
        """Log-magnitude loss + peak matching penalty.
        Detects local peaks in GT resultant and adds extra penalty
        for underestimating them, regardless of peak magnitude."""
        mask_3d = mask.unsqueeze(-1)
        n_valid = mask.sum().float() * 3
        mse = ((preds - labels) ** 2 * mask_3d).sum() / n_valid
        mag = torch.sqrt((labels ** 2).sum(dim=-1, keepdim=True)).detach()
        log_mag = 1.0 + torch.log1p(mag)
        weighted = ((preds - labels) ** 2 * log_mag * mask_3d).sum() / ((log_mag * mask_3d).sum() + 1e-8)
        pred_grad = preds[:, 1:] - preds[:, :-1]
        label_grad = labels[:, 1:] - labels[:, :-1]
        grad_mask = (mask[:, 1:] & mask[:, :-1]).unsqueeze(-1)
        grad_loss = ((pred_grad - label_grad) ** 2 * grad_mask).sum() / (grad_mask.sum().float() * 3 + 1e-8)
        # Peak loss: find frames where GT resultant has a local max (higher than both neighbors)
        gt_res = mag.squeeze(-1)  # [B, T]
        left = gt_res[:, :-2]
        center = gt_res[:, 1:-1]
        right = gt_res[:, 2:]
        is_peak = (center > left) & (center > right) & mask[:, 1:-1]
        if is_peak.any():
            peak_preds = preds[:, 1:-1][is_peak]   # [N_peaks, 3]
            peak_labels = labels[:, 1:-1][is_peak]  # [N_peaks, 3]
            peak_loss = ((peak_preds - peak_labels) ** 2).mean()
        else:
            peak_loss = torch.tensor(0.0, device=preds.device)
        return mse + 0.5 * weighted + 0.3 * grad_loss + 0.5 * peak_loss

    def asymmetric_magnitude_loss(preds, labels, mask, confidence=None):
        """MSE + magnitude-weighted MSE with asymmetric underprediction penalty.
        At high GT magnitudes, underpredicting costs ~3x more than overpredicting."""
        mask_3d = mask.unsqueeze(-1)
        n_valid = mask.sum().float() * 3
        mse = ((preds - labels) ** 2 * mask_3d).sum() / n_valid
        mag = torch.sqrt((labels ** 2).sum(dim=-1, keepdim=True)).detach()
        pred_mag = torch.sqrt((preds ** 2).sum(dim=-1, keepdim=True)).detach()
        underpred = (pred_mag < mag).float()
        asym_weight = mag * (1.0 + 2.0 * underpred * mag)
        weighted = ((preds - labels) ** 2 * asym_weight * mask_3d).sum() / ((asym_weight * mask_3d).sum() + 1e-8)
        pred_grad = preds[:, 1:] - preds[:, :-1]
        label_grad = labels[:, 1:] - labels[:, :-1]
        grad_mask = (mask[:, 1:] & mask[:, :-1]).unsqueeze(-1)
        grad_loss = ((pred_grad - label_grad) ** 2 * grad_mask).sum() / (grad_mask.sum().float() * 3 + 1e-8)
        return mse + 0.5 * weighted + 0.3 * grad_loss

    def asymmetric_linear_loss(preds, labels, mask, confidence=None):
        """Asymmetric magnitude-aware loss with linear scaling (not quadratic like Q).
        Selectively penalizes underprediction at high peaks without blanket upward bias."""
        mask_3d = mask.unsqueeze(-1)
        n_valid = mask.sum().float() * 3
        mse = ((preds - labels) ** 2 * mask_3d).sum() / n_valid
        mag = torch.sqrt((labels ** 2).sum(dim=-1, keepdim=True)).detach()
        pred_mag = torch.sqrt((preds ** 2).sum(dim=-1, keepdim=True)).detach()
        underpred = (pred_mag < mag).float()
        # Linear: 1 at low mag (no bias), grows with mag only for underpredictions
        # mag=4, underpred: weight=5.  mag=4, overpred: weight=1.  mag=0.5: weight≈1.
        asym_weight = 1.0 + 1.0 * underpred * mag
        weighted = ((preds - labels) ** 2 * asym_weight * mask_3d).sum() / ((asym_weight * mask_3d).sum() + 1e-8)
        pred_grad = preds[:, 1:] - preds[:, :-1]
        label_grad = labels[:, 1:] - labels[:, :-1]
        grad_mask = (mask[:, 1:] & mask[:, :-1]).unsqueeze(-1)
        grad_loss = ((pred_grad - label_grad) ** 2 * grad_mask).sum() / (grad_mask.sum().float() * 3 + 1e-8)
        return mse + 0.5 * weighted + 0.3 * grad_loss

    # Select loss based on experiment
    if EXP == 'f':
        criterion = confidence_weighted_loss
        print("Loss: confidence-weighted symmetric (MSE + magnitude + gradient)")
    elif EXP == 'q':
        criterion = asymmetric_magnitude_loss
        print("Loss: asymmetric magnitude (underprediction penalized ~3x at high forces)")
    elif EXP in ('r', 's'):
        criterion = asymmetric_linear_loss
        print("Loss: asymmetric linear (underprediction weight grows linearly with mag)")
    elif EXP in ('a', 'c', 'd', 'e', 'h', 'i', 'l', 'm', 'n', 'o', 'p', 't'):
        criterion = symmetric_loss
        print("Loss: symmetric (MSE + magnitude-weighted + gradient)")
    elif EXP == 'j':
        criterion = log_magnitude_loss
        print("Loss: log-magnitude (flattened weighting + gradient)")
    elif EXP == 'k':
        criterion = log_magnitude_peak_loss
        print("Loss: log-magnitude + peak matching")
    elif EXP == 'b':
        criterion = masked_mse
        print("Loss: MSE (log-space targets)")
    else:
        criterion = lambda p, l, m, confidence=None: expectile_loss(p, l, m, tau=0.7)
        print("Loss: expectile (tau=0.7)")

    if lookahead > 0:
        # Delayed-causal: output[t] predicts label[t-lookahead], so model uses
        # inputs up to time t to predict the target at time t-N.
        # At deployment this means N-frame latency with N-frame lookahead context.
        _inner = criterion
        def _lookahead_wrap(preds, labels, mask, confidence=None):
            p = preds[:, lookahead:]
            l = labels[:, :-lookahead]
            m = mask[:, :-lookahead]
            c = confidence[:, :-lookahead] if confidence is not None else None
            return _inner(p, l, m, confidence=c)
        criterion = _lookahead_wrap
        print(f"Lookahead: {lookahead} frames ({lookahead*10}ms at 100Hz)")

    best_val_loss = float('inf')

    n_train_frames = sum(s[2] for s in train_ds.sequences)
    n_val_frames = sum(s[2] for s in val_ds.sequences)

    for epoch in range(EPOCHS):
        # ── Train ──
        model.train()
        train_loss = 0.0
        train_frames = 0
        for inputs, labels, mask, confidence in train_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            mask = mask.to(DEVICE)
            confidence = confidence.to(DEVICE)

            optimizer.zero_grad()
            if use_transformer:
                preds = model(inputs, mask=mask)
            else:
                preds = model(inputs)
            loss = criterion(preds, labels, mask, confidence=confidence)
            loss.backward()
            optimizer.step()
            batch_frames = mask.sum().item()
            train_loss += loss.item() * batch_frames
            train_frames += batch_frames

        train_loss /= train_frames

        # ── Validate ──
        model.eval()
        val_loss = 0.0
        val_frames = 0
        val_fy_errors = []
        with torch.no_grad():
            for inputs, labels, mask, confidence in val_loader:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)
                mask = mask.to(DEVICE)
                mask_3d = mask.unsqueeze(-1)

                if use_transformer:
                    preds = model(inputs, mask=mask)
                else:
                    preds = model(inputs)
                # Apply lookahead shift for val too (same semantics as training loss)
                if lookahead > 0:
                    preds_v = preds[:, lookahead:]
                    labels_v = labels[:, :-lookahead]
                    mask_v = mask[:, :-lookahead]
                else:
                    preds_v, labels_v, mask_v = preds, labels, mask
                mask_v_3d = mask_v.unsqueeze(-1)
                sq_err = (preds_v - labels_v) ** 2 * mask_v_3d
                n_valid = mask_v.sum().float()
                loss = sq_err.sum() / (n_valid * 3)
                batch_frames = n_valid.item()
                val_loss += loss.item() * batch_frames
                val_frames += batch_frames

                # Fy (axial) MAE in BW (masked)
                fy_err = (torch.abs(preds_v[:, :, 1] - labels_v[:, :, 1]) * mask_v).sum() / n_valid
                val_fy_errors.append(fy_err.item())

        val_loss /= val_frames
        val_fy_mae = np.mean(val_fy_errors)

        scheduler.step(val_loss)

        # ── Report ──
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:3d}/{EPOCHS} | "
              f"Train MSE: {train_loss:.6f} | "
              f"Val MSE: {val_loss:.6f} | "
              f"Val Fy MAE: {val_fy_mae:.4f} BW | "
              f"LR: {lr:.1e}")

        # ── Save best ──
        model_name = f'best_model_{EXP}.pt' if EXP else 'best_model.pt'
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'n_features': n_features,
                'input_mean': input_mean,
                'input_std': input_std,
                'log_targets': (EXP == 'b'),
                'lower_body_only': lower_body,
                'model_class': 'tcn' if use_tcn else ('fft_mlp' if use_fft_mlp else ('transformer' if use_transformer else ('v3' if use_v3 else ('v2_causal' if use_v2_causal else ('v2' if use_v2 else 'v1'))))),
                'jcf_subdir': jcf_subdir,
                'clean_features': clean_feats,
                'include_mass': mass_input,
                'use_root_features': root_feats,
                'combine_root_features': combine_root,
                'lookahead': lookahead,
            }, os.path.join(DATA_ROOT, model_name))
            print(f"  → Saved best model (val_loss={val_loss:.6f})")

    print(f"\nTraining complete. Best val MSE: {best_val_loss:.6f}")
    print(f"Model saved to {DATA_ROOT}/{model_name}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default=None, choices=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't'],
                        help='a=symmetric loss, b=log-space, c=lower-body, d=lower-body+resnet+rebalance, e=clean-subjects-only, f=confidence-weighted, g=2x-muscle-scaled labels, h=transformer+2x-scaled, i=d+clean features, j=i+log-mag loss, k=i+log-mag+peak loss, l=TCN, m=FFT-MLP, n=i+body mass, o=root-frame features')
    parser.add_argument('--no-flat-filter', action='store_true',
                        help='Disable flat-waveform quality filter (keep all subjects)')
    args = parser.parse_args()
    train(exp=args.exp, filter_flat=not args.no_flat_filter)

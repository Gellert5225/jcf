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

SEED = 1
DATA_ROOT = "./jcf/full_duration/training/walking"
BATCH_SIZE = 8          # full sequences per batch (pad to max length)
EPOCHS = 300
LEARNING_RATE = 5e-4
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
                 use_root_features=False, combine_root_features=False,
                 max_peak_bw=10.0, include_speed=False, include_stance=False,
                 predict_moment=False, predict_flexion_moment=False,
                 max_moment_bwh=0.15, min_length=100,
                 native_time_grid=False):
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
    force_data = np.column_stack([fx, fy, fz]) / BW  # forces normalized by BW

    if predict_moment:
        # Mx = knee adduction/abduction (varus-valgus) moment in N·m
        # Mz = knee flexion-extension moment (clinically relevant for patellofemoral loading)
        # Normalize both by BW × height (dimensionless), per Schipplein/Andriacchi
        # convention. height_m must be in metadata.
        height_m = meta.get('height_m', None)
        if height_m is None or height_m <= 0:
            return None
        # OpenSim's JointReaction (apply_on_bodies=child, express_in_frame=child)
        # for the Rajagopal walker_knee_r returns Mx in the tibia frame with the
        # convention that Mx > 0 corresponds to medial-compartment loading during
        # right-leg stance — i.e., it already matches the Schipplein-Andriacchi
        # M_add. No sign flip is needed.
        mx = jcf['walker_knee_r_on_tibia_r_in_tibia_r_mx'].values
        mx_norm = mx / (BW * height_m)
        # Reject SO-corrupted moments (healthy walking |Mx_norm| < ~0.05; 0.15 = 3× max)
        if np.abs(mx_norm).max() > max_moment_bwh:
            return None
        moment_cols = [mx_norm.reshape(-1, 1)]
        if predict_flexion_moment:
            mz = jcf['walker_knee_r_on_tibia_r_in_tibia_r_mz'].values
            mz_norm = mz / (BW * height_m)
            if np.abs(mz_norm).max() > max_moment_bwh:
                return None
            moment_cols.append(mz_norm.reshape(-1, 1))
        jcf_data = np.column_stack([force_data, *moment_cols])  # [T, 4 or 5]
    else:
        jcf_data = force_data  # [T, 3]

    # Filter out subjects with unreasonable JCF (bad SO output).
    # Use force-only resultant for the peak-magnitude filter — moment scale differs.
    peak_resultant = np.sqrt((force_data ** 2).sum(axis=1)).max()
    if peak_resultant > max_peak_bw:
        return None

    # Reject trials too short to provide context for a bidirectional CNN
    # (Falisse, downhill running, and other fragments often have <100 frames).
    # MLP path passes min_length=1 to keep short static segments.
    if len(jcf_data) < min_length:
        return None

    # Align by time: find overlapping time range
    t_start = max(ik_time[0], grf_time[0], jcf_time[0])
    t_end = min(ik_time[-1], grf_time[-1], jcf_time[-1])

    if native_time_grid:
        # Anchor to the IK time grid: no smoothing of IK inputs from interpolation.
        # Labels and GRF get interpolated onto IK time. This matches what an
        # inference deployment sees (raw IK with no resampling), so the same
        # network at runtime gets exactly the inputs it was trained on.
        ik_mask = (ik_time >= t_start) & (ik_time <= t_end)
        t_common = ik_time[ik_mask]
        ik_interp = ik_data[ik_mask]
        grf_interp = np.column_stack([
            np.interp(t_common, grf_time, grf_data[:, j])
            for j in range(grf_data.shape[1])
        ])
        labels = np.column_stack([
            np.interp(t_common, jcf_time, jcf_data[:, j])
            for j in range(jcf_data.shape[1])
        ])
    else:
        # Default: interpolate everything to JCF timestamps (they're the most sparse).
        # NOTE: this slightly low-pass filters the IK/GRF inputs because JCF
        # timestamps are typically sub-sample-shifted from IK timestamps. The
        # native_time_grid path above avoids this.
        jcf_mask = (jcf_time >= t_start) & (jcf_time <= t_end)
        t_common = jcf_time[jcf_mask]
        labels = jcf_data[jcf_mask]
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
        # Normalize GRF-in-root-frame columns (last 6 before COM acc) by BW.
        # Layout: joint_centers(n_joints*3) + root_dynamics(12) + grf_root(6) + com_acc(3)
        # Use negative indices since n_joints varies by dataset.
        root_interp[:, -9:-3] /= BW
        parts = [root_interp]
    else:
        parts = [ik_interp, ik_vel, ik_acc, grf_interp, grf_vel]
    if combine_root_features:
        root_path = os.path.join(subject_dir, 'root_features.npy')
        if not os.path.exists(root_path):
            return None
        root_data = np.load(root_path)
        root_time = ik_time[:len(root_data)]
        # Skip leading joint_centers block (varies by dataset: 60/63/66 cols).
        # Take last 21 cols: root vel/acc (12) + GRF in root frame (6) + COM acc (3)
        root_dyn = root_data[:, -21:]
        root_dyn_interp = np.column_stack([
            np.interp(t_common, root_time, root_dyn[:, j])
            for j in range(root_dyn.shape[1])
        ])
        root_dyn_interp[:, 12:18] /= BW  # GRF-in-root-frame columns
        parts.append(root_dyn_interp)
    if include_mass:
        parts.append(np.full((len(t_common), 1), mass))
    if include_speed:
        # Smoothed horizontal pelvis speed (proxy for gait speed). 1s window.
        # Uses raw IK pelvis_tx (index 3) before any joint slicing.
        speed_raw = np.abs(np.gradient(ik_data[:, 3], ik_time))
        speed_interp = np.interp(t_common, ik_time, speed_raw)
        win = min(100, len(speed_interp))
        kernel = np.ones(win) / win
        speed_smooth = np.convolve(speed_interp, kernel, mode='same')
        parts.append(speed_smooth.reshape(-1, 1))
    if include_stance:
        # Smoothed per-foot vertical GRF over 1-second window.
        # Captures gait tempo: slower gait → longer stance phase → higher mean.
        # force_cols order: r_vx, r_vy, r_vz, l_vx, l_vy, l_vz so vy is at 1, 4.
        win = min(100, len(t_common))
        kernel = np.ones(win) / win
        stance_r = np.convolve(grf_interp[:, 1], kernel, mode='same')
        stance_l = np.convolve(grf_interp[:, 4], kernel, mode='same')
        parts.append(stance_r.reshape(-1, 1))
        parts.append(stance_l.reshape(-1, 1))
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
                 use_root_features=False, combine_root_features=False,
                 max_peak_bw=10.0, include_speed=False, include_stance=False,
                 predict_moment=False, predict_flexion_moment=False,
                 native_time_grid=False):
        self.sequences = []  # list of (inputs, labels, length) or (inputs, labels, length, confidence)
        self.loaded_dirs = []  # subj_dirs that successfully loaded (1:1 with sequences)
        self.has_confidence = with_confidence

        for subj_dir in subject_dirs:
            result = load_subject(subj_dir, lower_body_only=lower_body_only,
                                 native_time_grid=native_time_grid,
                                 with_confidence=with_confidence,
                                 jcf_subdir=jcf_subdir, clean_features=clean_features,
                                 include_mass=include_mass,
                                 use_root_features=use_root_features,
                                 combine_root_features=combine_root_features,
                                 max_peak_bw=max_peak_bw,
                                 include_speed=include_speed,
                                 include_stance=include_stance,
                                 predict_moment=predict_moment,
                                 predict_flexion_moment=predict_flexion_moment)
            if result is None:
                continue
            self.loaded_dirs.append(subj_dir)
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
    n_out = labels_list[0].shape[1]

    padded_inputs = torch.zeros(len(batch), max_len, n_feat)
    padded_labels = torch.zeros(len(batch), max_len, n_out)
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

    def __init__(self, n_features=43, n_outputs=3, dropout=0.1):
        super().__init__()

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Conv1d(n_features, 128, kernel_size=7, padding=3),
            nn.GroupNorm(8, 128),
            nn.ReLU(),
        )

        # Residual blocks with increasing dilation
        self.res_blocks = nn.Sequential(
            ResBlock1d(128, kernel_size=5, dilation=1, dropout=dropout),
            ResBlock1d(128, kernel_size=5, dilation=2, dropout=dropout),
            ResBlock1d(128, kernel_size=5, dilation=4, dropout=dropout),
            ResBlock1d(128, kernel_size=5, dilation=8, dropout=dropout),
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

def train(exp=None, filter_flat=True, dataset=None, exclude=None, max_peak_bw=10.0, seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    seed_suffix = f"_s{seed}" if seed != SEED else ""
    EXP = exp
    n3_variants = ('n3', 'n3s', 'n3_bin', 'n3_w15', 'n3_d05', 'n3_d05_w15', 'n3_d05_stance')
    n_d05_variants = ('n_d05', 'n_d05_m', 'n_d05_m2', 'n_d05_m_s', 'n_d05_m_s2')  # V2 + speed + low dropout. _m=+Mx, _m2=+Mx+Mz, _m_s=+Mx+static, _m_s2=_m_s + native time grid (no IK/GRF resampling)
    lower_body = EXP in ('c', 'd', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't') + n3_variants + n_d05_variants
    use_v2 = EXP in ('d', 'g', 'i', 'j', 'k', 'n', 'o', 'p', 'q') + n_d05_variants
    use_v2_causal = EXP in ('r', 's', 't')
    use_v3 = EXP in n3_variants
    use_transformer = EXP == 'h'
    use_tcn = EXP == 'l'
    use_fft_mlp = EXP == 'm'
    rebalance = EXP in ('d', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't') + n3_variants + n_d05_variants
    bin_rebalance = EXP == 'n3_bin'  # rebalance by peak magnitude bin instead of dataset
    clean_only = EXP == 'e'
    use_confidence = EXP == 'f'
    clean_feats = EXP in ('i', 'j', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 't') + n3_variants + n_d05_variants
    mass_input = EXP in ('l', 'n', 'o', 'p', 'q', 'r', 's', 't') + n3_variants + n_d05_variants
    speed_input = EXP in ('n3s', 'n3_bin', 'n3_w15', 'n3_d05', 'n3_d05_w15', 'n3_d05_stance') + n_d05_variants
    stance_input = EXP == 'n3_d05_stance'
    predict_moment = EXP in ('n_d05_m', 'n_d05_m2', 'n_d05_m_s', 'n_d05_m_s2')
    include_static = EXP in ('n_d05_m_s', 'n_d05_m_s2')
    native_time_grid = EXP == 'n_d05_m_s2'
    predict_flexion_moment = EXP == 'n_d05_m2'
    root_feats = EXP == 'o'
    combine_root = EXP == 'p'
    # Per-experiment overrides
    v3_dropout = 0.05 if EXP in ('n3_d05', 'n3_d05_w15', 'n3_d05_stance') else 0.15
    v2_dropout = 0.05 if EXP in n_d05_variants else 0.1
    sym_loss_mag_weight = 1.5 if EXP in ('n3_w15', 'n3_d05_w15') else 0.5
    n_outputs = 3 + (1 if predict_moment else 0) + (1 if predict_flexion_moment else 0)
    lookahead = 10 if EXP in ('s', 't') else 0
    scaled_labels = EXP in ('g', 'h')  # use 2x muscle-scaled SO labels
    # Find all subjects with JCF output
    jcf_subdir = 'jcf_output_2x' if scaled_labels else 'jcf_output'
    excluded_set = set(exclude) if exclude else set()
    roots_to_scan = [DATA_ROOT]
    if include_static:
        static_root = "./jcf/full_duration/training/static"
        if os.path.isdir(static_root):
            roots_to_scan.append(static_root)
        else:
            print(f"WARNING: include_static set but {static_root} missing")
    subject_dirs = []
    for root in roots_to_scan:
        for name in sorted(os.listdir(root)):
            if dataset and not name.startswith(f"{dataset}_"):
                continue
            if any(name.startswith(f"{ex}_") for ex in excluded_set):
                continue
            subj_dir = os.path.join(root, name)
            jcf_sto = os.path.join(subj_dir, jcf_subdir,
                                   'BatchJCF_JointReaction_ReactionLoads.sto')
            if os.path.isdir(subj_dir) and os.path.exists(jcf_sto):
                subject_dirs.append(subj_dir)
    if include_static:
        n_static = sum(1 for d in subject_dirs if '/static/' in d)
        n_walking = len(subject_dirs) - n_static
        print(f"  Loaded {n_walking} walking + {n_static} static subject dirs")
    if dataset:
        print(f"Dataset filter: {dataset} only ({len(subject_dirs)} subjects)")
    if excluded_set:
        print(f"Excluded datasets: {sorted(excluded_set)}")
    if max_peak_bw < 10.0:
        print(f"Max peak filter: {max_peak_bw} BW (subjects with higher peaks rejected)")
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

    # Split by SUBJECT (not by trial) to avoid data leakage. Different trials
    # of the same subject are biomechanically near-identical, so keeping all
    # trials of a subject in the same split gives an honest val signal.
    import re
    def _subject_id(path):
        # Strip trailing _t## and optional _r## so all trials of one subject share an ID.
        # e.g. "carter_P003_split0_t04" → "carter_P003_split0"
        return re.sub(r'_t\d+(_r\d+)?$', '', os.path.basename(path))

    unique_subjects = sorted(set(_subject_id(d) for d in subject_dirs))
    train_subj, val_subj = train_test_split(
        unique_subjects, train_size=TRAIN_SPLIT, random_state=seed
    )
    train_subj_set = set(train_subj)
    train_dirs = [d for d in subject_dirs if _subject_id(d) in train_subj_set]
    val_dirs   = [d for d in subject_dirs if _subject_id(d) not in train_subj_set]
    print(f"Train: {len(train_subj)} subjects ({len(train_dirs)} trials), "
          f"Val: {len(val_subj)} subjects ({len(val_dirs)} trials)")

    # Create datasets
    print("Loading training data...")
    train_ds = JCFDataset(train_dirs, lower_body_only=lower_body,
                          with_confidence=use_confidence,
                          jcf_subdir=jcf_subdir, clean_features=clean_feats,
                          include_mass=mass_input, use_root_features=root_feats,
                          combine_root_features=combine_root,
                          max_peak_bw=max_peak_bw,
                          include_speed=speed_input,
                          include_stance=stance_input,
                          predict_moment=predict_moment,
                          predict_flexion_moment=predict_flexion_moment,
                          native_time_grid=native_time_grid)
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
                        combine_root_features=combine_root,
                        max_peak_bw=max_peak_bw,
                        include_speed=speed_input,
                        include_stance=stance_input,
                        predict_moment=predict_moment,
                        predict_flexion_moment=predict_flexion_moment,
                        native_time_grid=native_time_grid)
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

    # Dataset/magnitude rebalancing: oversample minority groups
    if rebalance:
        from torch.utils.data import WeightedRandomSampler
        if bin_rebalance:
            # Per-subject peak magnitude binning. Force equal exposure across peak
            # ranges to fight the model's tendency to predict near the median.
            bin_edges = [0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0]
            peaks = []
            for seq in train_ds.sequences:
                labels = seq[1].numpy() if hasattr(seq[1], 'numpy') else seq[1]
                peaks.append(float(np.sqrt((labels ** 2).sum(axis=-1)).max()))
            peaks = np.array(peaks)
            bin_idx = np.digitize(peaks, bin_edges[1:-1])  # 0..len-2
            bin_counts = np.bincount(bin_idx, minlength=len(bin_edges)-1)
            weights = [1.0 / max(bin_counts[bi], 1) for bi in bin_idx]
            sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
            label_pairs = [(f'{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}', int(bin_counts[i]))
                           for i in range(len(bin_counts))]
            print(f"  Magnitude-bin rebalancing: {label_pairs}")
        else:
            # Use the dataset's record of which subjects actually loaded (1:1 with
            # train_ds.sequences) so the sampler is sized correctly.
            # When mixing activities (include_static), tag dataset prefix with
            # activity so e.g. carter_walking and carter_static get separate
            # buckets — otherwise walking (vastly more trials) drowns out static.
            def _bucket(d):
                prefix = os.path.basename(d).split('_')[0]
                if include_static:
                    activity = 'static' if '/static/' in d else 'walking'
                    return f"{prefix}_{activity}"
                return prefix
            loaded_buckets = [_bucket(d) for d in train_ds.loaded_dirs]
            dataset_counts_loaded = {}
            for p in loaded_buckets:
                dataset_counts_loaded[p] = dataset_counts_loaded.get(p, 0) + 1
            weights = [1.0 / dataset_counts_loaded[p] for p in loaded_buckets]
            sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
            print(f"  Dataset rebalancing: {dataset_counts_loaded}")
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
        model = JCF_TCN(n_features=n_features, n_outputs=n_outputs).to(DEVICE)
        print("Using JCF_TCN (temporal convolutional network, dilations 1-64)")
    elif use_fft_mlp:
        model = JCF_FFT_MLP(n_features=n_features, n_outputs=n_outputs).to(DEVICE)
        print("Using JCF_FFT_MLP (windowed FFT features + MLP)")
    elif use_transformer:
        model = JCF_Transformer(n_features=n_features, n_outputs=n_outputs).to(DEVICE)
        print("Using JCF_Transformer (conv stem + transformer encoder)")
    elif use_v3:
        model = JCF_CNN_v3(n_features=n_features, n_outputs=n_outputs, dropout=v3_dropout).to(DEVICE)
        print(f"Using JCF_CNN_v3 (256ch, 6 res blocks, dropout {v3_dropout}, outputs {n_outputs})")
    elif use_v2_causal:
        model = JCF_CNN_v2_causal(n_features=n_features, n_outputs=n_outputs).to(DEVICE)
        print(f"Using JCF_CNN_v2_causal (causal residual blocks — online/streaming, outputs {n_outputs})")
    elif use_v2:
        model = JCF_CNN_v2(n_features=n_features, n_outputs=n_outputs, dropout=v2_dropout).to(DEVICE)
        print(f"Using JCF_CNN_v2 (residual blocks, wider, dropout {v2_dropout}, outputs {n_outputs})")
    else:
        model = JCF_CNN(n_features=n_features, n_outputs=n_outputs).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    print(f"Device: {DEVICE}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )

    def masked_mse(preds, labels, mask, confidence=None):
        mask_3d = mask.unsqueeze(-1)
        return ((preds - labels) ** 2 * mask_3d).sum() / (mask.sum().float() * 3)

    def symmetric_loss(preds, labels, mask, confidence=None, mag_weight=0.5):
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
        return mse + mag_weight * weighted + 0.3 * grad_loss

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
    elif EXP in ('a', 'c', 'd', 'e', 'h', 'i', 'l', 'm', 'n', 'o', 'p', 't', 'n3', 'n3s', 'n3_bin', 'n3_w15', 'n3_d05', 'n3_d05_w15', 'n3_d05_stance', 'n_d05', 'n_d05_m', 'n_d05_m2'):
        # Inject the per-experiment magnitude weight (default 0.5; n3_w15 uses 1.5)
        _w = sym_loss_mag_weight
        criterion = lambda p, l, m, confidence=None: symmetric_loss(p, l, m, confidence, mag_weight=_w)
        print(f"Loss: symmetric (MSE + {_w}×magnitude-weighted + 0.3×gradient)")
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
        model_name = f'best_model_{EXP}{seed_suffix}.pt' if EXP else f'best_model{seed_suffix}.pt'
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
                'include_speed': speed_input,
                'include_stance': stance_input,
                'predict_moment': predict_moment,
                'predict_flexion_moment': predict_flexion_moment,
                'n_outputs': n_outputs,
            }, os.path.join(DATA_ROOT, model_name))
            print(f"  → Saved best model (val_loss={val_loss:.6f})")

    print(f"\nTraining complete. Best val MSE: {best_val_loss:.6f}")
    print(f"Model saved to {DATA_ROOT}/{model_name}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default=None, choices=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'n3', 'n3s', 'n3_bin', 'n3_w15', 'n3_d05', 'n3_d05_w15', 'n3_d05_stance', 'n_d05', 'n_d05_m', 'n_d05_m2', 'n_d05_m_s', 'n_d05_m_s2'],
                        help='a=symmetric loss, b=log-space, c=lower-body, d=lower-body+resnet+rebalance, e=clean-subjects-only, f=confidence-weighted, g=2x-muscle-scaled labels, h=transformer+2x-scaled, i=d+clean features, j=i+log-mag loss, k=i+log-mag+peak loss, l=TCN, m=FFT-MLP, n=i+body mass, o=root-frame features')
    parser.add_argument('--no-flat-filter', action='store_true',
                        help='Disable flat-waveform quality filter (keep all subjects)')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Restrict training to one dataset prefix (e.g. carter, hammer). Default: all.')
    parser.add_argument('--exclude', type=str, nargs='+', default=None,
                        help='Dataset prefixes to exclude (e.g. --exclude han fregly).')
    parser.add_argument('--max-peak', type=float, default=10.0,
                        help='Reject subjects with JCF resultant peak > this many BW. Walking: 4.0 recommended.')
    parser.add_argument('--seed', type=int, default=SEED,
                        help=f'Random seed (default: {SEED}). Non-default seeds save to best_model_{{exp}}_s{{N}}.pt')
    args = parser.parse_args()
    train(exp=args.exp, filter_flat=not args.no_flat_filter,
          dataset=args.dataset, exclude=args.exclude, max_peak_bw=args.max_peak,
          seed=args.seed)

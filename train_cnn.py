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
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# ─── Config ──────────────────────────────────────────────────────────────────

DATA_ROOT = "./jcf/training/running"
BATCH_SIZE = 8          # full sequences per batch (pad to max length)
EPOCHS = 200
LEARNING_RATE = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_SPLIT = 0.8       # 80% train, 20% val (by subject)
N_LOWER_BODY = 20       # joints 0-19: pelvis, hips, knees, ankles, feet


# ─── Data Quality Filter ──────────────────────────────────────────────────────

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


def load_subject(subject_dir, lower_body_only=False, with_confidence=False):
    """
    Load one subject's data. Returns (inputs, labels, mass) or None.
    
    inputs: [T, n_features]  (positions + GRF)
    labels: [T, 3]           (Fx, Fy, Fz in BW)
    """
    ik_path = os.path.join(subject_dir, 'ik_results.mot')
    grf_path = os.path.join(subject_dir, 'grf_data.mot')
    jcf_path = os.path.join(subject_dir, 'jcf_output',
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

    # Concatenate inputs: [positions, velocities, accelerations, GRF, GRF_vel]
    inputs = np.hstack([ik_interp, ik_vel, ik_acc, grf_interp, grf_vel])
    # [T, 37+37+37+6+6 = 123]

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

    def __init__(self, subject_dirs, lower_body_only=False, with_confidence=False):
        self.sequences = []  # list of (inputs, labels, length) or (inputs, labels, length, confidence)
        self.has_confidence = with_confidence

        for subj_dir in subject_dirs:
            result = load_subject(subj_dir, lower_body_only=lower_body_only,
                                 with_confidence=with_confidence)
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
    def __init__(self, channels, kernel_size=5, dilation=1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            nn.GroupNorm(8, channels),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            nn.GroupNorm(8, channels),
        )
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


# ─── Training ─────────────────────────────────────────────────────────────────

def train(exp=None):
    EXP = exp
    lower_body = EXP in ('c', 'd')
    use_v2 = EXP == 'd'
    rebalance = EXP == 'd'
    clean_only = EXP == 'e'
    use_confidence = EXP == 'f'
    # Find all subjects with JCF output
    subject_dirs = []
    for name in sorted(os.listdir(DATA_ROOT)):
        subj_dir = os.path.join(DATA_ROOT, name)
        jcf_sto = os.path.join(subj_dir, 'jcf_output',
                               'BatchJCF_JointReaction_ReactionLoads.sto')
        if os.path.isdir(subj_dir) and os.path.exists(jcf_sto):
            subject_dirs.append(subj_dir)

    print(f"Found {len(subject_dirs)} subjects with JCF data")
    if len(subject_dirs) == 0:
        print("No data! Run batch_process.py first.")
        return

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
                          with_confidence=use_confidence)
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
                        with_confidence=use_confidence)
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
            result = load_subject(d, lower_body_only=lower_body)
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
    if use_v2:
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

    # Select loss based on experiment
    if EXP == 'f':
        criterion = confidence_weighted_loss
        print("Loss: confidence-weighted symmetric (MSE + magnitude + gradient)")
    elif EXP in ('a', 'c', 'd', 'e'):
        criterion = symmetric_loss
        print("Loss: symmetric (MSE + magnitude-weighted + gradient)")
    elif EXP == 'b':
        criterion = masked_mse
        print("Loss: MSE (log-space targets)")
    else:
        criterion = lambda p, l, m, confidence=None: expectile_loss(p, l, m, tau=0.7)
        print("Loss: expectile (tau=0.7)")

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

                preds = model(inputs)
                # Masked MSE for val
                sq_err = (preds - labels) ** 2 * mask_3d
                n_valid = mask.sum().float()
                loss = sq_err.sum() / (n_valid * 3)
                batch_frames = n_valid.item()
                val_loss += loss.item() * batch_frames
                val_frames += batch_frames

                # Fy (axial) MAE in BW (masked)
                fy_err = (torch.abs(preds[:, :, 1] - labels[:, :, 1]) * mask).sum() / n_valid
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
                'model_class': 'v2' if use_v2 else 'v1',
            }, os.path.join(DATA_ROOT, model_name))
            print(f"  → Saved best model (val_loss={val_loss:.6f})")

    print(f"\nTraining complete. Best val MSE: {best_val_loss:.6f}")
    print(f"Model saved to {DATA_ROOT}/{model_name}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default=None, choices=['a', 'b', 'c', 'd', 'e', 'f'],
                        help='a=symmetric loss, b=log-space, c=lower-body, d=lower-body+resnet+rebalance, e=clean-subjects-only, f=confidence-weighted')
    args = parser.parse_args()
    train(exp=args.exp)

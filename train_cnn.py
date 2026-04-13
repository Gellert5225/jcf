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
"""

import os
import json
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# ─── Config ──────────────────────────────────────────────────────────────────

DATA_ROOT = "./jcf/training"
WINDOW_SIZE = 50        # frames per window (0.5s at 100Hz)
STRIDE = 10             # sliding window stride (0.1s)
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_SPLIT = 0.8       # 80% train, 20% val (by subject)


# ─── Data Loading ─────────────────────────────────────────────────────────────

def load_mot(path, skiprows=6):
    """Load an OpenSim .mot file."""
    return pd.read_csv(path, sep='\t', skiprows=skiprows)


def load_sto(path, skiprows=11):
    """Load an OpenSim .sto file."""
    return pd.read_csv(path, sep='\t', skiprows=skiprows)


def load_subject(subject_dir):
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
    force_cols = [c for c in grf.columns if '_force_v' in c]
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

    # Concatenate inputs: [positions, GRF]
    inputs = np.hstack([ik_interp, grf_interp])  # [T, 37+6=43]

    return inputs, labels, mass


# ─── Dataset ──────────────────────────────────────────────────────────────────

class JCFDataset(Dataset):
    """Sliding window dataset over multiple subjects."""

    def __init__(self, subject_dirs, window_size=50, stride=10):
        self.windows = []  # list of (inputs_window, labels_window)

        for subj_dir in subject_dirs:
            result = load_subject(subj_dir)
            if result is None:
                continue
            inputs, labels, mass = result
            T = len(labels)

            # Create sliding windows
            for start in range(0, T - window_size, stride):
                end = start + window_size
                inp_window = inputs[start:end]   # [W, 43]
                lbl_window = labels[start:end]   # [W, 3]
                self.windows.append((
                    torch.tensor(inp_window, dtype=torch.float32),
                    torch.tensor(lbl_window, dtype=torch.float32),
                ))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return self.windows[idx]


# ─── Model ────────────────────────────────────────────────────────────────────

class JCF_CNN(nn.Module):
    """
    1D CNN for predicting knee JCF from kinematics + GRF.
    
    Input:  [batch, window_size, n_features]
    Output: [batch, window_size, 3]  (Fx, Fy, Fz per frame)
    """

    def __init__(self, n_features=43, n_outputs=3):
        super().__init__()

        # Conv1d expects [batch, channels, length]
        self.encoder = nn.Sequential(
            nn.Conv1d(n_features, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Conv1d(128, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
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


# ─── Training ─────────────────────────────────────────────────────────────────

def train():
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

    # Split by subject (not by window) to avoid data leakage
    train_dirs, val_dirs = train_test_split(
        subject_dirs, train_size=TRAIN_SPLIT, random_state=42
    )
    print(f"Train: {len(train_dirs)} subjects, Val: {len(val_dirs)} subjects")

    # Create datasets
    print("Loading training data...")
    train_ds = JCFDataset(train_dirs, WINDOW_SIZE, STRIDE)
    print(f"  {len(train_ds)} training windows")

    print("Loading validation data...")
    val_ds = JCFDataset(val_dirs, WINDOW_SIZE, STRIDE)
    print(f"  {len(val_ds)} validation windows")

    if len(train_ds) == 0:
        print("No training windows created. Check data.")
        return

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, pin_memory=True)

    # Compute input feature count from first sample
    n_features = train_ds[0][0].shape[1]
    print(f"Input features per frame: {n_features}")

    # Create model
    model = JCF_CNN(n_features=n_features, n_outputs=3).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    print(f"Device: {DEVICE}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5
    )
    criterion = nn.MSELoss()

    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        # ── Train ──
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            preds = model(inputs)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)

        train_loss /= len(train_ds)

        # ── Validate ──
        model.eval()
        val_loss = 0.0
        val_fy_errors = []  # track axial component error specifically
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                preds = model(inputs)
                loss = criterion(preds, labels)
                val_loss += loss.item() * inputs.size(0)

                # Fy (axial) MAE in BW
                fy_err = torch.abs(preds[:, :, 1] - labels[:, :, 1]).mean()
                val_fy_errors.append(fy_err.item())

        val_loss /= len(val_ds)
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
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'n_features': n_features,
                'window_size': WINDOW_SIZE,
            }, os.path.join(DATA_ROOT, 'best_model.pt'))
            print(f"  → Saved best model (val_loss={val_loss:.6f})")

    print(f"\nTraining complete. Best val MSE: {best_val_loss:.6f}")
    print(f"Model saved to {DATA_ROOT}/best_model.pt")


if __name__ == '__main__':
    train()

"""
MLP Training Script for Knee JCF Prediction
=============================================
Trains an MLP to predict knee joint contact forces (Fx, Fy, Fz)
from joint kinematics + GRF. Uses the same data pipeline as train_cnn.py.

The MLP operates per-frame: it flattens a window of frames into a single
vector and predicts the JCF for **all frames in the window** at once.

Usage:
    conda run -n jcf python train_mlp.py
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# Reuse data loading and dataset from train_cnn
from train_cnn import load_subject, JCFDataset

# ─── Config ──────────────────────────────────────────────────────────────────

DATA_ROOT = "./jcf/training/walking"
WINDOW_SIZE = 50        # frames per window (0.5s at 100Hz)
STRIDE = 10             # sliding window stride (0.1s)
BATCH_SIZE = 64
EPOCHS = 200
LEARNING_RATE = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_SPLIT = 0.8


# ─── Model ────────────────────────────────────────────────────────────────────

class JCF_MLP(nn.Module):
    """
    MLP for predicting knee JCF from kinematics + GRF.

    Input:  [batch, window_size, n_features]
    Output: [batch, window_size, 3]  (Fx, Fy, Fz per frame)
    """

    def __init__(self, n_features=43, window_size=50, n_outputs=3):
        super().__init__()
        input_dim = n_features * window_size
        output_dim = n_outputs * window_size
        self.window_size = window_size
        self.n_outputs = n_outputs

        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, output_dim),
        )

    def forward(self, x):
        # x: [batch, window_size, n_features]
        B = x.size(0)
        x = x.reshape(B, -1)                          # [batch, W*F]
        x = self.net(x)                                # [batch, W*3]
        x = x.reshape(B, self.window_size, self.n_outputs)  # [batch, W, 3]
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
    model = JCF_MLP(n_features=n_features, window_size=WINDOW_SIZE,
                    n_outputs=3).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    print(f"Device: {DEVICE}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5
    )
    mse_loss = nn.MSELoss()

    def criterion(preds, labels):
        loss_mse = mse_loss(preds, labels)

        pred_diff = preds[:, 1:, :] - preds[:, :-1, :]
        label_diff = labels[:, 1:, :] - labels[:, :-1, :]
        loss_grad = mse_loss(pred_diff, label_diff)

        pred_peak, _ = torch.abs(preds).max(dim=1)
        label_peak, _ = torch.abs(labels).max(dim=1)
        loss_peak = mse_loss(pred_peak, label_peak)

        return loss_mse + 0.5 * loss_grad + 0.5 * loss_peak

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
        val_fy_errors = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                preds = model(inputs)
                loss = mse_loss(preds, labels)
                val_loss += loss.item() * inputs.size(0)

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
                'model_type': 'mlp',
            }, os.path.join(DATA_ROOT, 'best_model_mlp.pt'))
            print(f"  → Saved best model (val_loss={val_loss:.6f})")

    print(f"\nTraining complete. Best val MSE: {best_val_loss:.6f}")
    print(f"Model saved to {DATA_ROOT}/best_model_mlp.pt")


if __name__ == '__main__':
    train()

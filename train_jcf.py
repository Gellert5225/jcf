import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt


# ─── Data Loading ───────────────────────────────────────────────────────────

def load_sto(path):
    """Load an OpenSim .sto file, skipping the variable-length header."""
    with open(path) as f:
        for i, line in enumerate(f):
            if 'endheader' in line:
                skip = i + 1
                break
    return pd.read_csv(path, sep='\t', skiprows=skip)


def load_mot(path):
    """Load an OpenSim .mot file (6-line header)."""
    return pd.read_csv(path, sep='\t', skiprows=6)


def build_dataset_from_subject(subject_dir, mass_kg):
    """
    Align GRF inputs with JCF targets for one subject.
    Returns (inputs, targets) as numpy arrays, or None if files missing.
    """
    grf_path = os.path.join(subject_dir, 'grf_data.mot')
    jcf_path = os.path.join(subject_dir, 'test_output',
                            'SingleSubjTest_JointReaction_ReactionLoads.sto')

    if not os.path.exists(grf_path) or not os.path.exists(jcf_path):
        print(f"  Skipping {subject_dir}: missing files")
        return None

    grf = load_mot(grf_path)
    jcf = load_sto(jcf_path)

    BW = mass_kg * 9.81

    # Extract GRF force columns (right + left foot)
    grf_cols = [c for c in grf.columns if '_force_v' in c]
    # e.g. calcn_r_force_vx, calcn_r_force_vy, calcn_r_force_vz,
    #      calcn_l_force_vx, calcn_l_force_vy, calcn_l_force_vz

    # Compute JCF resultant (target) in BW
    fx_col = [c for c in jcf.columns if c.endswith('_fx')][0]
    fy_col = [c for c in jcf.columns if c.endswith('_fy')][0]
    fz_col = [c for c in jcf.columns if c.endswith('_fz')][0]
    jcf['resultant_BW'] = np.sqrt(
        jcf[fx_col]**2 + jcf[fy_col]**2 + jcf[fz_col]**2
    ) / BW

    # Interpolate GRF onto JCF timestamps (they may differ in sampling rate)
    jcf_times = jcf['time'].values
    inputs = np.zeros((len(jcf_times), len(grf_cols)))
    for i, col in enumerate(grf_cols):
        inputs[:, i] = np.interp(jcf_times, grf['time'].values, grf[col].values)

    # Normalize GRF by body weight too
    inputs /= BW

    targets = jcf['resultant_BW'].values

    return inputs.astype(np.float32), targets.astype(np.float32)


# ─── Dataset ────────────────────────────────────────────────────────────────

class JCFDataset(Dataset):
    def __init__(self, inputs, targets):
        self.X = torch.from_numpy(inputs)
        self.y = torch.from_numpy(targets).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ─── Model ──────────────────────────────────────────────────────────────────

class JCFNet(nn.Module):
    def __init__(self, input_dim, hidden_dims=[64, 64, 32]):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ─── Training ───────────────────────────────────────────────────────────────

def train():
    # --- Config ---
    subject_dirs = ['./jcf/P010_split0']  # Add more subjects here
    mass_kg = 55.3  # TODO: per-subject mass lookup
    epochs = 200
    batch_size = 64
    lr = 1e-3
    val_split = 0.2

    # --- Load data from all subjects ---
    all_inputs, all_targets = [], []
    for sd in subject_dirs:
        result = build_dataset_from_subject(sd, mass_kg)
        if result is not None:
            inputs, targets = result
            all_inputs.append(inputs)
            all_targets.append(targets)
            print(f"  Loaded {sd}: {len(targets)} frames")

    X = np.concatenate(all_inputs)
    y = np.concatenate(all_targets)
    print(f"Total: {len(y)} samples, input dim: {X.shape[1]}")

    # --- Normalize inputs (z-score) ---
    X_mean, X_std = X.mean(axis=0), X.std(axis=0)
    X_std[X_std == 0] = 1.0  # avoid div by zero
    X_norm = (X - X_mean) / X_std

    # Save normalization params for inference
    np.savez('jcf_norm_params.npz', mean=X_mean, std=X_std)

    # --- Split ---
    dataset = JCFDataset(X_norm, y)
    n_val = int(len(dataset) * val_split)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size)

    # --- Model ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = JCFNet(input_dim=X.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    print(f"Model: {sum(p.numel() for p in model.parameters())} params, device={device}")

    # --- Train loop ---
    train_losses, val_losses = [], []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        train_losses.append(epoch_loss / n_train)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += loss_fn(model(xb), yb).item() * len(xb)
        val_losses.append(val_loss / n_val)

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs}  "
                  f"train={train_losses[-1]:.6f}  val={val_losses[-1]:.6f}")

    # --- Save model ---
    torch.save(model.state_dict(), 'jcf_model.pt')
    print("Saved model to jcf_model.pt")

    # --- Plot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss curves
    ax1.plot(train_losses, label='Train')
    ax1.plot(val_losses, label='Val')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MSE Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Predictions vs ground truth (on validation set)
    model.eval()
    preds, truths = [], []
    with torch.no_grad():
        for xb, yb in val_dl:
            preds.append(model(xb.to(device)).cpu())
            truths.append(yb)
    preds = torch.cat(preds).numpy().flatten()
    truths = torch.cat(truths).numpy().flatten()

    ax2.scatter(truths, preds, s=2, alpha=0.5)
    lims = [0, max(truths.max(), preds.max()) * 1.1]
    ax2.plot(lims, lims, 'r--', linewidth=1)
    ax2.set_xlabel('Ground Truth (BW)')
    ax2.set_ylabel('Predicted (BW)')
    ax2.set_title(f'Validation (R²={1 - np.sum((preds-truths)**2)/np.sum((truths-truths.mean())**2):.3f})')
    ax2.set_xlim(lims)
    ax2.set_ylim(lims)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_results.png', dpi=150)
    plt.show()
    print("Saved training_results.png")


if __name__ == '__main__':
    train()

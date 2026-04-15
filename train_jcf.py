import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
import os
import json
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
    return pd.read_csv(path, sep=r'\s+', skiprows=skip)


def load_mot(path):
    """Load an OpenSim .mot file, auto-detecting header."""
    with open(path) as f:
        for i, line in enumerate(f):
            if 'endheader' in line:
                skip = i + 1
                break
    return pd.read_csv(path, sep=r'\s+', skiprows=skip)


def build_dataset_from_subject(subject_dir):
    """
    Build (inputs, targets) for one subject from batch pipeline output.
    
    Inputs:  6 GRF channels (calcn_r/l vx,vy,vz) normalised by BW
    Targets: 3 JCF components (Fx, Fy, Fz in tibial child frame) in BW
    
    Returns (inputs, targets) as numpy arrays, or None if files missing.
    """
    meta_path = os.path.join(subject_dir, 'metadata.json')
    grf_path = os.path.join(subject_dir, 'grf_data.mot')
    jcf_path = os.path.join(subject_dir, 'jcf_output',
                            'BatchJCF_JointReaction_ReactionLoads.sto')

    if not all(os.path.exists(p) for p in [meta_path, grf_path, jcf_path]):
        return None

    meta = json.load(open(meta_path))
    mass_kg = meta['mass_kg']
    BW = mass_kg * 9.81

    grf = load_mot(grf_path)
    jcf = load_sto(jcf_path)

    # Extract per-foot GRF force columns (calcn_r and calcn_l, vx/vy/vz)
    grf_cols = sorted([c for c in grf.columns
                       if ('calcn_r' in c or 'calcn_l' in c) and '_force_v' in c])
    if len(grf_cols) < 6:
        # Fallback: any force columns
        grf_cols = [c for c in grf.columns if '_force_v' in c][:6]
    if len(grf_cols) < 6:
        return None

    # Extract JCF 3-axis components (Fx, Fy, Fz in child/tibial frame)
    jcf_cols = jcf.columns.tolist()
    # Columns are: time, <joint>_on_<body>_in_<frame>_fx, _fy, _fz, _mx, _my, _mz
    fx_col = jcf_cols[1]  # force x
    fy_col = jcf_cols[2]  # force y (compressive axis in tibial frame)
    fz_col = jcf_cols[3]  # force z

    # Interpolate GRF onto JCF timestamps
    jcf_times = jcf['time'].values
    inputs = np.zeros((len(jcf_times), len(grf_cols)))
    for i, col in enumerate(grf_cols):
        inputs[:, i] = np.interp(jcf_times, grf['time'].values, grf[col].values)
    inputs /= BW  # normalise by body weight

    # 3-axis JCF targets in BW
    targets = np.column_stack([
        jcf[fx_col].values / BW,
        jcf[fy_col].values / BW,
        jcf[fz_col].values / BW,
    ])

    return inputs.astype(np.float32), targets.astype(np.float32)


# ─── Dataset ────────────────────────────────────────────────────────────────

class JCFDataset(Dataset):
    def __init__(self, inputs, targets):
        self.X = torch.from_numpy(inputs)
        self.y = torch.from_numpy(targets)  # shape (N, 3)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ─── Model ──────────────────────────────────────────────────────────────────

class JCFNet(nn.Module):
    """MLP with 3 output nodes: Fx, Fy, Fz in tibial child frame.
    
    This is superior for operational space control because the gradient
    tells exactly which direction of movement reduces each specific
    force component (e.g. Fy = compressive axis).
    """
    def __init__(self, input_dim, hidden_dims=[64, 64, 32]):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, 3))  # 3 outputs: Fx, Fy, Fz
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ─── Training ───────────────────────────────────────────────────────────────

def train():
    # --- Config ---
    data_root = './jcf/training'
    epochs = 200
    batch_size = 64
    lr = 1e-3
    val_split = 0.2

    # --- Discover subjects (prefixed: moore_*, tiziana_*) ---
    subject_dirs = sorted(
        glob.glob(os.path.join(data_root, 'moore_*'))
        + glob.glob(os.path.join(data_root, 'tiziana_*')),
        key=lambda p: os.path.basename(p).lower()
    )
    print(f"Found {len(subject_dirs)} subject directories")

    # --- Load data from all subjects ---
    all_inputs, all_targets = [], []
    for sd in subject_dirs:
        name = os.path.basename(sd)
        result = build_dataset_from_subject(sd)
        if result is not None:
            inputs, targets = result
            # Skip subjects with anomalous JCF (> 10 BW resultant)
            resultant = np.sqrt(np.sum(targets**2, axis=1))
            if resultant.max() > 10.0:
                print(f"  SKIP {name}: peak JCF {resultant.max():.1f} BW (anomalous)")
                continue
            # Skip subjects with suspiciously low JCF (failed SO)
            if resultant.max() < 1.5:
                print(f"  SKIP {name}: peak JCF {resultant.max():.2f} BW (SO likely failed)")
                continue
            all_inputs.append(inputs)
            all_targets.append(targets)
            print(f"  {name}: {len(targets)} frames, "
                  f"peak Fy={np.min(targets[:,1]):.2f} BW, "
                  f"peak |F|={resultant.max():.2f} BW")

    if not all_inputs:
        print("No valid data found!")
        return

    X = np.concatenate(all_inputs)
    y = np.concatenate(all_targets)
    print(f"\nTotal: {len(y)} samples, input dim: {X.shape[1]}, output dim: 3")

    # --- Normalize inputs (z-score) ---
    X_mean, X_std = X.mean(axis=0), X.std(axis=0)
    X_std[X_std == 0] = 1.0
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

    # --- Collect predictions ---
    model.eval()
    preds, truths = [], []
    with torch.no_grad():
        for xb, yb in val_dl:
            preds.append(model(xb.to(device)).cpu())
            truths.append(yb)
    preds = torch.cat(preds).numpy()   # (N, 3)
    truths = torch.cat(truths).numpy() # (N, 3)

    # --- Plot ---
    axis_names = ['Fx (shear)', 'Fy (compressive)', 'Fz (shear)']
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # Top row: per-axis scatter plots (pred vs truth)
    for i, ax in enumerate(axes[0]):
        p, t = preds[:, i], truths[:, i]
        ss_res = np.sum((p - t) ** 2)
        ss_tot = np.sum((t - t.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        ax.scatter(t, p, s=2, alpha=0.5)
        lims = [min(t.min(), p.min()) * 1.1, max(t.max(), p.max()) * 1.1]
        ax.plot(lims, lims, 'r--', linewidth=1)
        ax.set_xlabel('Ground Truth (BW)')
        ax.set_ylabel('Predicted (BW)')
        ax.set_title(f'{axis_names[i]}  R²={r2:.3f}')
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.grid(True, alpha=0.3)

    # Bottom left: resultant scatter
    ax = axes[1][0]
    pred_res = np.sqrt(np.sum(preds**2, axis=1))
    true_res = np.sqrt(np.sum(truths**2, axis=1))
    ss_res = np.sum((pred_res - true_res) ** 2)
    ss_tot = np.sum((true_res - true_res.mean()) ** 2)
    r2_res = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    ax.scatter(true_res, pred_res, s=2, alpha=0.5)
    lims = [0, max(true_res.max(), pred_res.max()) * 1.1]
    ax.plot(lims, lims, 'r--', linewidth=1)
    ax.set_xlabel('Ground Truth (BW)')
    ax.set_ylabel('Predicted (BW)')
    ax.set_title(f'Resultant |F|  R²={r2_res:.3f}')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.grid(True, alpha=0.3)

    # Bottom middle: loss curves
    ax = axes[1][1]
    ax.plot(train_losses, label='Train')
    ax.plot(val_losses, label='Val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title('Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Bottom right: per-axis RMSE bar chart
    ax = axes[1][2]
    rmses = [np.sqrt(np.mean((preds[:, i] - truths[:, i])**2)) for i in range(3)]
    rmses.append(np.sqrt(np.mean((pred_res - true_res)**2)))
    bars = ax.bar(['Fx', 'Fy', 'Fz', '|F|'], rmses, color=['tab:blue', 'tab:orange', 'tab:green', 'tab:red'])
    ax.set_ylabel('RMSE (BW)')
    ax.set_title('Per-axis RMSE')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, rmses):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig('training_results.png', dpi=150)
    plt.show()
    print("Saved training_results.png")


if __name__ == '__main__':
    train()

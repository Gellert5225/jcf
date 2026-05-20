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
        "desc": "full (q, dq, ddq, F, dF, mass, speed, height, BMI) — 64 ch",
    },
    "qdqf": {
        "base_indices": list(range(16)) + list(range(16, 32)) + list(range(48, 54)),
        "include_height": False,
        "include_bmi": False,
        "desc": "(q, dq, F) — 38 ch",
    },
    "qdqfmh": {
        "base_indices": list(range(16)) + list(range(16, 32)) + list(range(48, 54)) + [60],
        "include_height": True,
        "include_bmi": False,
        "desc": "(q, dq, F, mass, height) — 40 ch",
    },
    "qf": {
        "base_indices": list(range(16)) + list(range(48, 54)),
        "include_height": False,
        "include_bmi": False,
        "desc": "(q, F) — 22 ch",
    },
    "qfmh": {
        "base_indices": list(range(16)) + list(range(48, 54)) + [60],
        "include_height": True,
        "include_bmi": False,
        "desc": "(q, F, mass, height) — 24 ch",
    },
}


def feature_count(input_set):
    cfg = INPUT_SETS[input_set]
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
    def __init__(self, n_features=64, n_outputs=4, hidden=128, dropout=0.3):
        super().__init__()
        layers = []
        in_dim = n_features
        for _ in range(3):
            layers += [
                nn.Linear(in_dim, hidden),
                nn.LayerNorm(hidden),
                nn.ReLU(),
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
        chunks_in, chunks_out = [], []
        for subj_dir in subject_dirs:
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
          max_peak_bw=6.0, seed=SEED, input_set='full', dq_method='central'):
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

    model = JCF_MLP(n_features=n_features, n_outputs=N_OUTPUTS,
                    hidden=HIDDEN, dropout=DROPOUT).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: JCF_MLP {n_params:,} params (hidden={HIDDEN}, dropout={DROPOUT})")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE,
                                 weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5)

    def loss_fn(pred, target):
        # Plain MSE + magnitude-weighted force MSE.
        # No gradient term — single-frame model has no temporal axis.
        mse = ((pred - target) ** 2).mean()
        force_mag = torch.sqrt((target[:, :3] ** 2).sum(dim=-1, keepdim=True)).detach()
        weighted_num = ((pred[:, :3] - target[:, :3]) ** 2 * force_mag).sum()
        weighted_den = (force_mag.sum() + 1e-8) * 3
        weighted = weighted_num / weighted_den
        return mse + 0.5 * weighted

    best_val = float("inf")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        n_train = 0
        for inp, lbl in train_loader:
            inp = inp.to(DEVICE); lbl = lbl.to(DEVICE)
            optimizer.zero_grad()
            pred = model(inp)
            loss = loss_fn(pred, lbl)
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
                v = ((pred - lbl) ** 2).mean()
                val_loss += v.item() * len(inp)
                n_val += len(inp)
        val_loss /= n_val
        scheduler.step(val_loss)

        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1:3d}/{EPOCHS} | train: {train_loss:.6f} | "
              f"val: {val_loss:.6f} | lr: {lr:.1e}")

        if val_loss < best_val:
            best_val = val_loss
            # Default input_set keeps original filename; add suffix for variants
            suffix = "" if input_set == "full" else f"_{input_set}"
            if dq_method != "central":
                suffix += f"_{dq_method}"
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
                "dropout": DROPOUT,
                "input_set": input_set,
                "dq_method": dq_method,
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
    args = p.parse_args()
    train(activity=args.activity,
          filter_flat=not args.no_flat_filter,
          dataset=args.dataset, exclude=args.exclude,
          max_peak_bw=args.max_peak, seed=args.seed,
          input_set=args.input_set, dq_method=args.dq_method)

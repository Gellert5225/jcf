"""
Test the trained MLP surrogate (`best_model_mlp_<activity>.pt`).

Loads each test subject, runs the per-frame MLP on the same input pipeline used
in training (no temporal context — the MLP sees one frame at a time), and
reports R²/MAE/correlation per dataset and per channel.

Per-subject plots (GT vs Predicted, Fy and Mx) are saved into each test
subject's folder as `inference_mlp_<activity>.png`.

Usage:
    python test_mlp.py --activity static
    python test_mlp.py --activity walking --exclude han fregly
    python test_mlp.py --activity both --max-peak 6.0
"""
import os
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

from train_mlp import JCF_MLP, FrameDataset, DATA_ROOTS

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TEST_ROOTS = {
    "static":  "./jcf/full_duration/testing/static",
    "walking": "./jcf/full_duration/testing/walking",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--activity", type=str, default="static",
                   choices=["static", "walking", "both"])
    p.add_argument("--exclude", type=str, nargs="+", default=None)
    p.add_argument("--dataset", type=str, default=None,
                   help="Restrict tests to one prefix (e.g. carter)")
    p.add_argument("--max-peak", type=float, default=6.0)
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Override checkpoint path. Default looks at "
                        "DATA_ROOTS[activity]/best_model_mlp_<activity>.pt")
    return p.parse_args()


def main():
    args = parse_args()

    # Resolve checkpoint
    if args.checkpoint:
        ckpt_path = args.checkpoint
    else:
        # 'both' is saved into the static root by convention (first key in DATA_ROOTS)
        root_for_ckpt = DATA_ROOTS["static"] if args.activity in ("static", "both") \
            else DATA_ROOTS["walking"]
        ckpt_path = os.path.join(root_for_ckpt, f"best_model_mlp_{args.activity}.pt")
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}")
        print("Run train_mlp.py first.")
        return

    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    # Default hidden=128 (current). For older 256-hidden checkpoints, override
    # with `--hidden 256` if you ever need to load them.
    model = JCF_MLP(n_features=ckpt["n_features"], n_outputs=ckpt["n_outputs"],
                    hidden=ckpt.get("hidden", 128), dropout=0.0).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    inp_mean = ckpt["input_mean"].cpu().numpy()
    inp_std = ckpt["input_std"].cpu().numpy()
    # Variant checkpoints record their input_set / dq_method; default to "full"/"central"
    # for backwards compatibility with older checkpoints.
    input_set = ckpt.get("input_set", "full")
    dq_method = ckpt.get("dq_method", "central")
    print(f"Loaded {ckpt_path}")
    print(f"  epoch={ckpt['epoch']+1}, val_loss={ckpt['val_loss']:.6f}, "
          f"trained on activity={ckpt.get('activity', '?')}")
    print(f"  input_set={input_set}, dq_method={dq_method}, n_features={ckpt['n_features']}")

    # Pick test directories matching the requested activity
    excluded = set(args.exclude) if args.exclude else set()
    test_dirs = []
    for act in (["static", "walking"] if args.activity == "both" else [args.activity]):
        root = TEST_ROOTS[act]
        if not os.path.isdir(root):
            print(f"  [{act}] test root missing: {root}")
            continue
        for name in sorted(os.listdir(root)):
            if args.dataset and not name.startswith(f"{args.dataset}_"):
                continue
            if any(name.startswith(f"{ex}_") for ex in excluded):
                continue
            d = os.path.join(root, name)
            jcf = os.path.join(d, "jcf_output", "BatchJCF_JointReaction_ReactionLoads.sto")
            if os.path.isdir(d) and os.path.exists(jcf):
                test_dirs.append((act, name, d))
    print(f"\nFound {len(test_dirs)} test subjects")
    if not test_dirs:
        return

    channel_names = ["Fx", "Fy", "Fz", "Mx"][:ckpt["n_outputs"]]
    stats = defaultdict(lambda: defaultdict(list))
    n_processed = 0
    activity_tag = ckpt.get('activity', 'mlp')

    for act, name, d in test_dirs:
        single_ds = FrameDataset([d], max_peak_bw=args.max_peak,
                                 input_set=input_set, dq_method=dq_method)
        if len(single_ds) == 0:
            continue
        n_processed += 1

        x = (single_ds.frames_in - inp_mean) / inp_std
        with torch.no_grad():
            pred = model(torch.tensor(x, dtype=torch.float32, device=DEVICE)).cpu().numpy()
        gt = single_ds.frames_out

        ds_key = name.split("_")[0]
        for ds in (ds_key, "all"):
            for ch_idx, ch in enumerate(channel_names):
                stats[ds][f"{ch}_g"].extend(gt[:, ch_idx])
                stats[ds][f"{ch}_p"].extend(pred[:, ch_idx])

        # Per-subject plot
        T = len(gt)
        t = np.arange(T) * 0.01
        n_panels = 2 if pred.shape[1] >= 4 else 1
        fig, axes = plt.subplots(n_panels, 1, figsize=(13, 3.5 * n_panels), sharex=True)
        if n_panels == 1:
            axes = [axes]
        axes[0].plot(t, gt[:, 1], "b-", lw=1.4, label="GT")
        axes[0].plot(t, pred[:, 1], "r--", lw=1.4, label="Pred")
        axes[0].set_ylabel("Fy (BW)")
        axes[0].set_title(f"{name}  T={T}f  (MLP, activity={activity_tag})")
        axes[0].grid(True, alpha=0.3); axes[0].legend(loc="upper right")
        if pred.shape[1] >= 4:
            axes[1].plot(t, gt[:, 3], "b-", lw=1.4, label="GT")
            axes[1].plot(t, pred[:, 3], "r--", lw=1.4, label="Pred")
            axes[1].set_ylabel("Mx (BW × H)")
            axes[1].grid(True, alpha=0.3); axes[1].legend(loc="upper right")
        axes[-1].set_xlabel("Frame index × 0.01 s")
        plt.tight_layout()
        plt.savefig(os.path.join(d, f"inference_mlp_{activity_tag}.png"),
                    dpi=110, bbox_inches="tight")
        plt.close(fig)

    print(f"\nProcessed {n_processed} subjects (plots saved per folder)")
    print(f"\n{'split':<10} {'channel':<5} {'N':>10} {'corr':>7} {'MAE':>10} {'RMSE':>10} {'R²':>8}")
    print("-" * 64)
    for ds in sorted(stats.keys()):
        for ch in channel_names:
            g = np.array(stats[ds][f"{ch}_g"])
            p = np.array(stats[ds][f"{ch}_p"])
            if len(g) == 0:
                continue
            corr = np.corrcoef(g, p)[0, 1]
            mae = np.mean(np.abs(p - g))
            rmse = np.sqrt(np.mean((p - g) ** 2))
            r2 = 1 - np.sum((p - g) ** 2) / max(np.sum((g - g.mean()) ** 2), 1e-12)
            print(f"{ds:<10} {ch:<5} {len(g):>10} {corr:>7.3f} {mae:>10.5f} "
                  f"{rmse:>10.5f} {r2:>8.3f}")
        print()


if __name__ == "__main__":
    main()

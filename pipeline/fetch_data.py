#!/usr/bin/env python3
"""Fetch the processed knee-JCF training data (OpenSim IK/GRF/SO/JR outputs) from Hugging Face
and reconstruct the ``jcf/full_duration/training/`` layout the training scripts expect.

    python pipeline/fetch_data.py                      # everything (~15 GB download, ~45 GB on disk)
    python pipeline/fetch_data.py --labs carter moore  # subset of laboratories
    python pipeline/fetch_data.py --activity static    # one activity only

Requires ``huggingface_hub`` and read access to the (private) dataset repo: run ``hf auth login`` once.
Shards are plain tar.gz (extracted with the standard library). The per-trial OpenSim model files are
stored once per unique content hash and restored into every trial folder after extraction.
"""
import argparse, csv, os, shutil, tarfile
from huggingface_hub import hf_hub_download, snapshot_download

REPO = "gellert5225/jcf-knee-labels"
REVISION = "v1-isrr2026"
MODEL_FILES = (("scaled_model_sha256", "scaled_model.osim"),
               ("reserves_model_sha256", os.path.join("jcf_output", "_model_with_reserves.osim")))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", default="jcf/full_duration/training", help="target folder (default: repo layout)")
    ap.add_argument("--activity", nargs="+", default=["walking", "static"], choices=["walking", "static"])
    ap.add_argument("--labs", nargs="+", default=None, help="e.g. carter moore tiziana han fregly falisse")
    ap.add_argument("--revision", default=REVISION)
    ap.add_argument("--link", action="store_true", help="hard-link model files instead of copying them")
    args = ap.parse_args()

    local = snapshot_download(REPO, repo_type="dataset", revision=args.revision,
                              allow_patterns=["trials.csv", "models/*"])
    with open(os.path.join(local, "trials.csv")) as f:
        rows = list(csv.DictReader(f))
    wanted = [r for r in rows if r["activity"] in args.activity and (args.labs is None or r["lab"] in args.labs)]
    shards = sorted({(r["activity"], r["lab"]) for r in wanted})
    print(f"{len(wanted)} trials in {len(shards)} shard(s): {', '.join(f'{a}/{l}' for a, l in shards)}")

    for act, lab in shards:
        path = hf_hub_download(REPO, f"{act}/{lab}.tar.gz", repo_type="dataset", revision=args.revision)
        out = os.path.join(args.dest, act); os.makedirs(out, exist_ok=True)
        print(f"  extracting {act}/{lab}.tar.gz -> {out}", flush=True)
        with tarfile.open(path, "r:gz") as tf:
            tf.extractall(out, filter="data")

    put = os.link if args.link else shutil.copy2
    restored = 0
    for r in wanted:
        tdir = os.path.join(args.dest, r["activity"], r["trial"])
        for key, rel in MODEL_FILES:
            if not r.get(key):
                continue
            dst = os.path.join(tdir, rel)
            if os.path.isdir(os.path.dirname(dst)) and not os.path.exists(dst):
                put(os.path.join(local, "models", f"{r[key]}.osim"), dst); restored += 1
    print(f"done: {len(wanted)} trials under {args.dest}, {restored} model files restored")


if __name__ == "__main__":
    main()

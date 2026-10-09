"""Training provenance, stored in every checkpoint as ckpt["provenance"].

Records what produced a checkpoint, so it can be reproduced or audited later:
the exact command line, every train() argument including defaults, the effective
hyperparameters, the git commit and whether tracked files had uncommitted
changes, and the train/validation subject IDs of the split.

Show it for one or more checkpoints:

    python provenance.py models/runs/mlp/<checkpoint>.pt

Everything stored is plain Python data (str, int, float, bool, None, list, dict),
so torch.load(..., weights_only=True) keeps working on these checkpoints.
Checkpoints saved before this module existed have no "provenance" key.
"""
import datetime
import os
import platform
import shlex
import socket
import subprocess
import sys

SCHEMA = 1
_REPO = os.path.dirname(os.path.abspath(__file__))


def _git(*args):
    try:
        r = subprocess.run(["git", *args], cwd=_REPO, capture_output=True, text=True, timeout=10)
        return r.stdout.rstrip("\n") if r.returncode == 0 else None
    except Exception:
        return None


def git_info():
    """Commit, branch and any uncommitted changes to tracked files."""
    status = _git("status", "--porcelain", "--untracked-files=no")
    changed = [line[3:] for line in status.splitlines()] if status else []
    return {
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(changed),
        "uncommitted_files": changed[:50],
    }


def plain(x):
    """Convert to plain Python data so the checkpoint loads with weights_only=True."""
    if x is None:
        return None
    if isinstance(x, bool):
        return bool(x)
    for base in (str, int, float):                       # subclasses -> base type
        if isinstance(x, base):
            return base(x)
    if isinstance(x, dict):
        return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set, frozenset)):
        items = list(x)
        if isinstance(x, (set, frozenset)):
            try:
                items = sorted(items)
            except TypeError:
                items = sorted(items, key=str)
        return [plain(v) for v in items]
    if hasattr(x, "item") and callable(x.item):          # numpy / torch scalars
        try:
            return plain(x.item())
        except Exception:
            pass
    if hasattr(x, "tolist") and callable(x.tolist):      # small arrays
        try:
            return plain(x.tolist())
        except Exception:
            pass
    return str(x)


def subject_ids(dirs, id_fn):
    """Sorted unique subject IDs for a list of trial directories."""
    return sorted({id_fn(d) for d in dirs})


def build(call_args, hyperparameters, splits, extra=None):
    """Assemble the provenance record for one training run."""
    try:
        import torch
        torch_version = str(torch.__version__)
    except Exception:
        torch_version = None
    record = {
        "schema": SCHEMA,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "command": " ".join(shlex.quote(a) for a in [os.path.basename(sys.executable)] + sys.argv),
        "cwd": os.getcwd(),
        "host": socket.gethostname(),
        "python": platform.python_version(),
        "torch": torch_version,
        "git": git_info(),
        "call_args": call_args,
        "hyperparameters": hyperparameters,
        "splits": splits,
    }
    if extra:
        record["extra"] = extra
    return plain(record)


def announce(record):
    """Print a one-line summary at the start of training, and warn on uncommitted changes."""
    g = record.get("git") or {}
    commit = (g.get("commit") or "unknown")[:10]
    print(f"provenance: git {commit} on {g.get('branch')}"
          + ("  (UNCOMMITTED CHANGES)" if g.get("dirty") else ""))
    if g.get("dirty"):
        print("  WARNING: tracked files differ from the commit, so this run cannot be "
              "reproduced from git alone. Changed files:")
        for f in g.get("uncommitted_files", [])[:10]:
            print(f"    {f}")


def describe(path):
    """Print the provenance of one checkpoint. Returns 0, or 1 if it has none."""
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=True)
    p = ck.get("provenance")
    if not p:
        print(f"{path}: no provenance (saved before provenance tracking was added)\n")
        return 1
    g = p.get("git") or {}
    print(f"checkpoint  : {path}")
    print(f"created     : {p.get('created_utc')} on {p.get('host')} "
          f"(python {p.get('python')}, torch {p.get('torch')})")
    print(f"command     : {p.get('command')}")
    state = ("DIRTY: " + ", ".join(g.get("uncommitted_files", []))) if g.get("dirty") else "clean"
    print(f"git         : {g.get('commit')} ({g.get('branch')}), {state}")
    print("arguments   : " + ", ".join(f"{k}={v}" for k, v in (p.get("call_args") or {}).items()))
    print("hyperparams : " + ", ".join(f"{k}={v}" for k, v in (p.get("hyperparameters") or {}).items()
                                      if k != "data_roots"))
    for act, s in (p.get("splits") or {}).items():
        tr, va = s.get("train_subjects", []), s.get("val_subjects", [])
        print(f"split {act:<7}: {len(tr)} train / {len(va)} val subjects; val = {', '.join(va) or '-'}")
    if p.get("extra"):
        print("extra       : " + ", ".join(f"{k}={v}" for k, v in p["extra"].items()))
    print()
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(max(describe(a) for a in sys.argv[1:]))

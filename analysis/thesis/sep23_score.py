#!/usr/bin/env python3
"""F_medial on the optimizer's Sep 23 reference motion for any GRF-free MLP checkpoints, with sep23_seeds.py's primary
inputs (pelvis angles converted from the URDF's axes, the optimizer's exported COM and foot point, deployment-style
windows) and the physical sign (-mx at the right knee) for every checkpoint, whatever sign it was trained with.

Usage (repo root): python -I analysis/thesis/sep23_score.py --ckpt A.pt B.pt --labels a b [--out results/thesis/sep23/x]
Prints stance means / peaks / share over 1.4 BW per checkpoint (right single stance S1, S2) and writes <out>.csv
(per frame) when --out is given.
"""
import argparse
import os
import site
import sys
import tempfile
from types import SimpleNamespace

sys.path.append(site.getusersitepackages())       # torch's dependencies live in the user site (-I drops it)
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from analysis.thesis import sep23_seeds as S  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if len(a.ckpt) != len(a.labels):
        raise SystemExit("--ckpt and --labels must have the same length")
    os.chdir(REPO)
    torch.set_num_threads(1)
    src = os.path.join(REPO, S.TRIAL)
    H = float(pd.read_json(os.path.join(src, "metadata.json"), typ="series")["height_m"])
    from tools.evaluate_fmedial import load_model
    models = {lab: load_model(os.path.join(REPO, p), "cpu") for lab, p in zip(a.labels, a.ckpt)}
    _, coords_w = S.ik_coords(src)
    coords_c, _ = S.urdf_pelvis_to_opensim(coords_w)
    ns = SimpleNamespace(deploy_features=True, pelvis="running", max_peak=1000.0, max_moment=1000.0,
                         label_mask=False, label_mask_subdir="jcf_output_sojr")
    with tempfile.TemporaryDirectory(prefix="sep23_score_") as tmp:
        d = S.shadow(src, tmp, "export", ik_coords_override=coords_c)
        preds, _feats, fm = S.predict_all(d, models, ns)
    t = np.asarray(fm["t"], float)
    wm = S.window_masks(t)
    cols = {"t": t}
    print(f"{'model':22s} {'trained sign':>12s} | S1 mean / peak / >1.4 | S2 mean / peak / >1.4")
    for lab in a.labels:
        p = preds[lab]
        f, ax, mo = S.fmedial(p[:, 1], p[:, 3], H, S.MX_PHYSICAL)
        cols[f"{lab}_fmed"], cols[f"{lab}_axial"], cols[f"{lab}_moment"] = f, ax, mo
        sign = models[lab][3].get("mx_medial_sign")
        st = [S.stats(f, ax, mo, t, wm[w]) for w in ("S1", "S2")]
        print(f"{lab:22s} {('+1 (old)' if sign is None else f'{float(sign):+g}'):>12s} | "
              + " | ".join(f"{x['mean']:.2f} / {x['peak']:.2f} / {x['frac_over_limit']:.0%}" for x in st))
    if a.out:
        pd.DataFrame(cols).round(5).to_csv(a.out + ".csv", index=False)
        print("wrote", a.out + ".csv")
    return cols


if __name__ == "__main__":
    main()

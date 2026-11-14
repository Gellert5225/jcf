#!/usr/bin/env python3
"""The ISRR checkpoints on the optimizer's Sep 23 reference motion: the one the optimizer used (William, Nov 2026:
best_model_mlp_walking_qinst_notrans_pelvisrel_winpos_contact_fpopt_w20_fmW0p5_e100 = sep23_seeds' fpopt_legacy;
columns william_*) and models/mlp-grffree-base (columns paper_*; the model card's claim), read the way
sep23_seeds.py reads the other six checkpoints: pelvis angles converted from the URDF's axes to OpenSim's convention,
the optimizer's exported COM and foot point, deployment-style windows, physical M_x sign (-mx at the right knee).
The base model takes pelvis translations; they are centred on the trial mean (--pelvis trial, as in the paper) and,
for comparison, with the running-mean drift removal (--pelvis running).

Check: sept_legacy is run through the same path and must reproduce sep23_seeds' per_frame.csv column.

Usage (repo root): python -I analysis/thesis/sep23_paper_model.py
Output: results/thesis/sep23/paper_model.csv (t, fmed / axial / moment per pelvis mode, physical and pipeline sign;
    *_copied_* = pelvis angles as written in ik_results.mot, the repo converters' copy convention)
"""
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

PAPER = "models/mlp-grffree-base/mlp-grffree-base.pt"


def main():
    os.chdir(REPO)
    torch.set_num_threads(1)
    src = os.path.join(REPO, S.TRIAL)
    meta = pd.read_json(os.path.join(src, "metadata.json"), typ="series")
    H = float(meta["height_m"])
    from tools.evaluate_fmedial import load_model
    models = {"paper": load_model(os.path.join(REPO, PAPER), "cpu"),
              "william": load_model(os.path.join(REPO, S.CKPTS["fpopt_legacy"]), "cpu"),   # the optimizer's model
              "sept_legacy": load_model(os.path.join(REPO, S.CKPTS["sept_legacy"]), "cpu")}
    _, coords_w = S.ik_coords(src)
    coords_c, _ = S.urdf_pelvis_to_opensim(coords_w)
    out, out_copied = {}, {}
    with tempfile.TemporaryDirectory(prefix="sep23_paper_") as tmp:
        d = S.shadow(src, tmp, "export", ik_coords_override=coords_c)
        dc = S.shadow(src, tmp, "export_copied")
        for mode in ("trial", "running"):
            ns = SimpleNamespace(deploy_features=True, pelvis=mode, max_peak=1000.0, max_moment=1000.0,
                                 label_mask=False, label_mask_subdir="jcf_output_sojr")
            preds, _feats, fm = S.predict_all(d, models, ns)
            out[mode] = preds
            t = np.asarray(fm["t"], float)
            out_copied[mode] = S.predict_all(dc, models, ns)[0]
    ref = pd.read_csv(os.path.join(REPO, "results", "thesis", "sep23", "per_frame.csv"))
    assert np.allclose(ref.t.to_numpy(), t)
    p = out["running"]["sept_legacy"]
    chk = S.fmedial(p[:, 1], p[:, 3], H, S.MX_PHYSICAL)[0]
    err = float(np.abs(chk - ref.sept_legacy_fmed.to_numpy()).max())
    print(f"check: sept_legacy vs per_frame.csv max |diff| {err:.2e} BW")
    if err > 1e-3:
        raise SystemExit("the input path does not reproduce sep23_seeds")
    cols = dict(t=t)
    for label in ("william",):          # notrans: no pelvis-translation input, so the pelvis mode does not matter
        for tag, src_out in (("", out), ("copied_", out_copied)):
            pw = src_out["trial"][label]
            for sign_name, sign in (("physical", S.MX_PHYSICAL), ("pipeline", S.MX_PIPELINE)):
                cols[f"{label}_{tag}{sign_name}_fmed"] = S.fmedial(pw[:, 1], pw[:, 3], H, sign)[0]
    pw = out["trial"]["william"]
    err_w = float(np.abs(S.fmedial(pw[:, 1], pw[:, 3], H, S.MX_PHYSICAL)[0] - ref.fpopt_legacy_fmed.to_numpy()).max())
    print(f"check: william (fpopt_legacy) vs per_frame.csv max |diff| {err_w:.2e} BW")
    if err_w > 1e-3:
        raise SystemExit("the input path does not reproduce sep23_seeds for fpopt_legacy")
    for mode in ("trial", "running"):
        p = out[mode]["paper"]
        for sign_name, sign in (("physical", S.MX_PHYSICAL), ("pipeline", S.MX_PIPELINE)):
            f, ax, mo = S.fmedial(p[:, 1], p[:, 3], H, sign)
            cols[f"paper_{mode}_{sign_name}_fmed"] = f
            cols[f"paper_{mode}_{sign_name}_moment"] = mo
        cols[f"paper_{mode}_axial"] = ax
        pc = out_copied[mode]["paper"]          # pelvis angles as written by label_trajectory_csv.py (copied)
        for sign_name, sign in (("physical", S.MX_PHYSICAL), ("pipeline", S.MX_PIPELINE)):
            cols[f"paper_{mode}_copied_{sign_name}_fmed"] = S.fmedial(pc[:, 1], pc[:, 3], H, sign)[0]
    df = pd.DataFrame(cols)
    dst = os.path.join(REPO, "results", "thesis", "sep23", "paper_model.csv")
    df.round(5).to_csv(dst, index=False)
    wm = S.window_masks(t)
    for c in [c for c in df.columns if c.endswith("_fmed")]:
        f = df[c].to_numpy()
        print(f"{c}: " + ", ".join(f"{w} mean {f[m].mean():.2f} peak {f[m].max():.2f}" for w, m in wm.items())
              + f"; whole motion max {f.max():.2f} at {t[np.argmax(f)]:.2f} s, frames > 1.4: {(f > 1.4).sum()}")
    print("wrote", dst)


if __name__ == "__main__":
    main()

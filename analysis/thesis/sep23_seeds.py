#!/usr/bin/env python3
"""S4: training-run (seed) spread on the optimizer's reference motion (William's Sep 23 trajectory).

There is no ground truth on this motion. Its SO+JR labels (jcf_output_sojr/) are unusable (right-knee F_medial peak
27.5 BW; label_mask.npz keeps no right-knee frame) and its stored labels (jcf_output/) were made without muscle forces
(_analyze_setup.xml has an empty <forces_file/>). Both were computed by OpenSim on this folder's ik_results.mot, whose
pelvis angles are mis-mapped (below), so OpenSim's feet sit away from the applied COP. Both are reported for context
only, never as a reference.

Pelvis angles. The folder was written by pipeline/label_trajectory_csv.py, which copies the optimizer's URDF pelvis
angles q3..q5 straight into OpenSim's pelvis_tilt / pelvis_list / pelvis_rotation (line 81, ik[:, 0:3] = q[:, 3:6];
infer_trajectory_csv.py and the PN3 driver's ::q_urdf use the same copy). assets/Rajagopal2015.urdf rotates its pelvis
about the URDF's own z (up), x (forward) and y (left) axes, so the copied 'pelvis_tilt' is a rotation about the
vertical (pn3_redis/model.py docstring; docs/PN3_REDIS_DRIVER.md 'URDF pelvis axes'). urdf_pelvis_to_opensim()
re-expresses the optimizer's pelvis orientation R_u = Rz(q3) Rx(q4) Ry(q5) in OpenSim's ground, R = C^T R_u C, and
returns OpenSim's body-fixed Z-X-Y angles (to first order tilt = -q5, list = q4, rotation = q3). With the converted
angles the generic URDF reproduces the optimizer's exported foot points and COM to millimetres (foot_geometry.csv,
pelvis_conversion.csv); with the copied angles they are off by about 13 cm and 1.2 cm.

Inputs (all six checkpoints see the same matrix in each variant):
  * q, contact flags, mass/height: the trial folder (ik_results.mot = the optimizer's q, grf_data.mot = its foot
    wrenches), built by tools/evaluate_fmedial.features_for with --deploy-features semantics (exact time alignment,
    100 Hz, 20-frame windows; the trajectory is already 100 Hz on one 0.01-6.45 s grid, so legacy and _t100 sets
    get identical windows). The primary variants replace the three pelvis angles with the converted ones (a shadow
    ik_results.mot; every other column is copied unchanged, asserted).
  * COM and foot positions: com_features.npz of the folder. For this folder it is NOT OpenSim output:
    pipeline/label_trajectory_csv.py wrote the optimizer's own exported com_position / *_foot_position into it.
    The exported foot position is the optimizer's planted contact point, which is the fpopt point (0.13, -0.01, 0) m
    of the URDF calcn frame (mid-foot, 13 cm ahead of the calcn origin), and grf_data.mot's COP equals it.
Variants:
  export         (primary) converted pelvis angles; every model gets the exported COM and foot point, through a
                 shadow folder whose foot_point_0.130_-0.010_0.000.npz holds the exported positions, so fpopt and
                 calcn-origin input sets read the same numbers (asserted).
  export_copied  as export, but with the pelvis angles as written in ik_results.mot (the repo converters' copy
                 convention; the previous primary). Whether the optimizer's own surrogate call copies them the same
                 way is not checked here (its code is not in this repository).
  own            converted angles; calcn-origin checkpoints get the exported point moved back to the calcn origin
                 along the foot, x0 = x_export - R_calcn(q) (0.13, -0.01, 0), R_calcn from the generic URDF
                 (pn3_redis.model.LowerBody, the optimizer's body model); fpopt checkpoints as in export. Measures the
                 foot-point convention at deployment.
  scaled         converted angles; every model gets its own foot point placed on the trial's borrowed scaled OpenSim
                 model by forward kinematics (what pipeline/compute_com_features.py --foot-point would cache from the
                 converted q; ~2.5 cm from the exported point, segment scaling). COM stays the exported one.
                 Sensitivity only.

F_medial = 0.5 |F_y| + M_med H / 0.045 m, physical sign (M_med = -mx at the right knee) for every checkpoint, labels and
raw outputs alike (the legacy checkpoints' own +1 sign is not used). Windows: right single stance S1 1.98-2.77 s
(80 frames) and S2 4.19-4.83 s (65 frames), the phase=='right' runs of labels.csv.

Usage (repo root):
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -I analysis/thesis/sep23_seeds.py
Outputs: results/thesis/sep23/{per_frame.csv, summary.csv, spread.csv, pairs.csv, variants.csv, labels_context.csv,
    fk_validation.csv, foot_geometry.csv, pelvis_conversion.csv, cop_reference.csv, run_info.json, README.md}
"""
import argparse
import datetime
import glob
import json
import os
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import torch  # noqa: E402

from analysis.thesis.osim_fk import OsimChain  # noqa: E402

TRIAL = "jcf/external/william_2026-09-23"
CKPTS = {   # label -> checkpoint (relative to the repo root); order = table order
    "qc_s1": "models/runs/mlp/best_model_mlp_walking_qinst_notrans_pelvisrel_winpos_contact_fpopt_w20_t100_sojrqc_fmW0p5_e100_s1.pt",
    "qc_s2": "models/runs/mlp/best_model_mlp_walking_qinst_notrans_pelvisrel_winpos_contact_fpopt_w20_t100_sojrqc_fmW0p5_e100_s2.pt",
    "qc_s3": "models/runs/mlp/best_model_mlp_walking_qinst_notrans_pelvisrel_winpos_contact_fpopt_w20_t100_sojrqc_fmW0p5_e100_s3.pt",
    "qc_origin": "models/runs/mlp/best_model_mlp_walking_qinst_notrans_pelvisrel_winpos_contact_w20_t100_sojrqc_fmW0p5_e100_s1.pt",
    "fpopt_legacy": "models/runs/mlp/best_model_mlp_walking_qinst_notrans_pelvisrel_winpos_contact_fpopt_w20_fmW0p5_e100.pt",
    "sept_legacy": "models/runs/mlp/best_model_mlp_walking_qinst_notrans_pelvisrel_winpos_contact_w20_fmW0p5_e100_2.pt",
}
SEEDS = ["qc_s1", "qc_s2", "qc_s3"]
FPOPT = (0.13, -0.01, 0.0)
WINDOWS = {"S1": (1.98, 2.77), "S2": (4.19, 4.83)}
PHI, D_M, LIMIT = 0.5, 0.045, 1.4
MX_PHYSICAL, MX_PIPELINE = -1.0, 1.0
FEATURE_LABELS = "jcf_output"      # label folder whose time grid the features are built on (all three are identical)
FK_CHECK_N_PER_LAB = 2
COP_REF_N = 3                      # moving-COP Moore training trials for the heel-to-toe reference
PELVIS_ROT = ("pelvis_tilt", "pelvis_list", "pelvis_rotation")
C_URDF = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])     # v_urdf = C v_opensim (pn3_redis/model.py)
VARIANTS = ("export", "export_copied", "own", "scaled")


def fmedial(fy, mx_raw, H, sign):
    axial = PHI * np.abs(fy)
    moment = sign * mx_raw * H / D_M
    return axial + moment, axial, moment


def read_sto(path):
    with open(path) as f:
        for n, line in enumerate(f):
            if line.strip().lower() == "endheader":
                break
    return pd.read_csv(path, sep=r"\s+", skiprows=n + 1)


def ik_coords(d):
    q = pd.read_csv(os.path.join(d, "ik_results.mot"), sep=r"\s+", skiprows=6)
    return q["time"].to_numpy(float), {c: q[c].to_numpy(float) for c in q.columns if c != "time"}


# ─── pelvis angles: URDF axes -> OpenSim convention ──────────────────────────────────────────────────────────────

def urdf_pelvis_to_opensim(coords):
    """coords with pelvis_tilt / list / rotation replaced by OpenSim-convention angles.

    The input values are the optimizer's q3..q5 as label_trajectory_csv.py copied them: rotations about the URDF's
    z (up), x (forward), y (left) axes in that order (assets/Rajagopal2015.urdf joints pelvis_tilt / pelvis_list /
    pelvis_rotation). R_u = Rz(q3) Rx(q4) Ry(q5) in the URDF's z-up world is R = C^T R_u C in OpenSim's y-up ground,
    and OpenSim's ground_pelvis angles are its body-fixed Z-X-Y Euler angles. Returns (coords, round-trip error)."""
    from scipy.spatial.transform import Rotation
    from pn3_redis.model import axis_angle
    q3, q4, q5 = (np.asarray(coords[k], float) for k in PELVIS_ROT)
    Ru = axis_angle([0, 0, 1], q3) @ axis_angle([1, 0, 0], q4) @ axis_angle([0, 1, 0], q5)
    R = np.einsum("ij,sjk,kl->sil", C_URDF.T, Ru, C_URDF)
    e = Rotation.from_matrix(R).as_euler("ZXY")              # intrinsic z-x-y: R = Rz(tilt) Rx(list) Ry(rotation)
    back = axis_angle([0, 0, 1], e[:, 0]) @ axis_angle([1, 0, 0], e[:, 1]) @ axis_angle([0, 1, 0], e[:, 2])
    err = float(np.abs(back - R).max())
    if err > 1e-9:
        raise SystemExit(f"pelvis conversion round trip failed ({err:.1e})")
    out = dict(coords)
    out.update({k: e[:, i] for i, k in enumerate(PELVIS_ROT)})
    return out, err


def write_ik(src_path, dst_path, coords):
    """Copy of src_path with the pelvis-angle columns taken from coords (header kept verbatim; the training loaders
    skip a fixed 6 header lines). Every other column must read back unchanged."""
    txt = open(src_path).read()
    head = txt[:txt.index("endheader\n") + len("endheader\n")]
    df = pd.read_csv(src_path, sep=r"\s+", skiprows=6)
    new = df.copy()
    for k in PELVIS_ROT:
        new[k] = coords[k]
    with open(dst_path, "w") as f:
        f.write(head)
        new.to_csv(f, sep="\t", index=False, float_format="%.8f")
    chk = pd.read_csv(dst_path, sep=r"\s+", skiprows=6)
    other = [c for c in df.columns if c not in PELVIS_ROT]
    if list(chk.columns) != list(df.columns) or not np.array_equal(chk[other].to_numpy(), df[other].to_numpy()) \
            or np.abs(chk[list(PELVIS_ROT)].to_numpy() - new[list(PELVIS_ROT)].to_numpy()).max() > 1e-8:
        raise SystemExit(f"{dst_path}: rewritten IK does not read back as intended")


# ─── forward-kinematics check against OpenSim-written caches ─────────────────────────────────────────────────────

def fk_validation(seed=0):
    """Max |numpy FK - OpenSim cache| (m) for the calcn origin (com_features.npz) and the fpopt point
    (foot_point_0.130_-0.010_0.000.npz) on training trials of every lab, plus carter_P022_split0_t05 (whose scaled
    model the Sep 23 folder borrows)."""
    files = sorted(glob.glob(os.path.join(REPO, "jcf/full_duration/training/walking/*/foot_point_0.130_-0.010_0.000.npz")))
    by_lab = {}
    for f in files:
        by_lab.setdefault(os.path.basename(os.path.dirname(f)).split("_")[0], []).append(f)
    rng = np.random.default_rng(seed)
    pick = [os.path.join(REPO, "jcf/full_duration/training/walking/carter_P022_split0_t05/foot_point_0.130_-0.010_0.000.npz")]
    for lab in sorted(by_lab):
        pick += list(rng.choice(by_lab[lab], size=min(FK_CHECK_N_PER_LAB, len(by_lab[lab])), replace=False))
    rows = []
    for f in pick:
        d = os.path.dirname(f)
        _t, coords = ik_coords(d)
        fk = OsimChain(os.path.join(d, "scaled_model.osim"))
        z, zf = np.load(os.path.join(d, "com_features.npz")), np.load(f)
        row = dict(trial=os.path.basename(d), lab=os.path.basename(d).split("_")[0], frames=len(_t))
        for s in "rl":
            row[f"origin_{s}_max_err_m"] = float(np.abs(fk.station(f"calcn_{s}", (0, 0, 0), coords) - z[f"calcn_{s}_pos"]).max())
            row[f"fpopt_{s}_max_err_m"] = float(np.abs(fk.station(f"calcn_{s}", FPOPT, coords) - zf[f"calcn_{s}_pos"]).max())
        rows.append(row)
    return pd.DataFrame(rows), len(files)


# ─── foot geometry ───────────────────────────────────────────────────────────────────────────────────────────────

def foot_geometry(src, coords_by_q, z, H, ph, wm):
    """Exported foot point / COP against the fpopt point and calcn frame of the generic URDF (the optimizer's body
    model) and of the borrowed scaled model, with the pelvis angles as written and converted."""
    from pn3_redis.model import LowerBody, COORDS as URDF_COORDS
    lb = LowerBody(height_m=H)
    fk = OsimChain(os.path.join(src, "scaled_model.osim"))
    g = read_sto(os.path.join(src, "grf_data.mot"))
    fp = np.asarray(FPOPT)
    rows, com_err, urdf_frames = [], {}, {}
    for qn, co in coords_by_q.items():
        fr = lb.fk(np.column_stack([co[c] for c in URDF_COORDS]))
        urdf_frames[qn] = fr
        com_err[qn] = np.linalg.norm(lb.com(fr) - z["com_pos"], axis=1)
        for mn in ("urdf", "scaled"):
            for s in "rl":
                R, p = fr[f"calcn_{s}"] if mn == "urdf" else fk.body_pose(f"calcn_{s}", co)
                pt = p + np.einsum("sij,j->si", R, fp)
                x = z[f"calcn_{s}_pos"].astype(float)
                cop = g[[f"calcn_{s}_force_p{c}" for c in "xyz"]].to_numpy(float)
                loc = np.einsum("tji,tj->ti", R, cop - p)     # COP in this model's calcn frame
                wins = [("S1", wm["S1"]), ("S2", wm["S2"]), ("right_single", ph == "right")] if s == "r" \
                    else [("left_single", ph == "left")]
                for w, m in wins:
                    dist = np.linalg.norm(pt - x, axis=1)[m]
                    rows.append(dict(q_pelvis=qn, model=mn, foot=s, window=w, n=int(m.sum()),
                                     fpopt_vs_export_mean_m=float(dist.mean()), fpopt_vs_export_max_m=float(dist.max()),
                                     fpopt_world_range_x_m=float(np.ptp(pt[m, 0])),
                                     export_world_range_x_m=float(np.ptp(x[m, 0])),
                                     cop_in_calcn_x_min_m=float(loc[m, 0].min()), cop_in_calcn_x_max_m=float(loc[m, 0].max()),
                                     cop_in_calcn_y_min_m=float(loc[m, 1].min()), cop_in_calcn_y_max_m=float(loc[m, 1].max()),
                                     cop_equals_export_max_m=float(np.abs(cop - x).max())))
    return pd.DataFrame(rows), com_err, urdf_frames


def pelvis_conversion_table(coords_w, coords_c, com_err, wm):
    rows = []
    T = len(coords_w["pelvis_tilt"])
    for w in ("S1", "S2", "S1+S2", "all"):
        m = np.ones(T, bool) if w == "all" else wm[w]
        r = dict(window=w, n=int(m.sum()))
        for k in PELVIS_ROT:
            d = np.degrees(np.abs(coords_c[k] - coords_w[k]))[m]
            r[f"d_{k}_max_deg"], r[f"d_{k}_mean_abs_deg"] = float(d.max()), float(d.mean())
        for qn in ("as_written", "converted"):
            r[f"urdf_com_vs_export_{qn}_mean_m"] = float(com_err[qn][m].mean())
            r[f"urdf_com_vs_export_{qn}_max_m"] = float(com_err[qn][m].max())
        rows.append(r)
    return pd.DataFrame(rows)


def cop_reference(n=COP_REF_N, seed=0):
    """Heel-to-toe progression on moving-COP Moore training trials (the lab that supplies most kept frames): the
    right-foot COP of grf_data.mot in the trial's own scaled calcn_r frame over right single stance."""
    files = sorted(glob.glob(os.path.join(REPO, "jcf/full_duration/training/walking/moore_*/jcf_output_sojr/label_mask.npz")))
    rng = np.random.default_rng(seed)
    rows = []
    for f in rng.permutation(files):
        if len(rows) >= n:
            break
        zm = np.load(f)
        if bool(zm["fixed_cop_r"]) or zm["keep_r"].mean() < 0.2:
            continue
        d = os.path.dirname(os.path.dirname(f))
        mass = float(json.load(open(os.path.join(d, "metadata.json"))).get("mass_kg") or 0)
        if mass <= 0:
            continue
        t, coords = ik_coords(d)
        g = read_sto(os.path.join(d, "grf_data.mot"))
        BW = mass * 9.81
        gt = g["time"].to_numpy(float)
        ry = np.interp(t, gt, g["calcn_r_force_vy"].to_numpy(float))
        ly = np.interp(t, gt, g["calcn_l_force_vy"].to_numpy(float))
        ss = (np.abs(ry) > 0.05 * BW) & (np.abs(ly) <= 0.05 * BW)
        if ss.sum() < 30:
            continue
        cop = np.column_stack([np.interp(t, gt, g[f"calcn_r_force_p{c}"].to_numpy(float)) for c in "xyz"])
        R, p = OsimChain(os.path.join(d, "scaled_model.osim")).body_pose("calcn_r", coords)
        x = np.einsum("tji,tj->ti", R, cop - p)[ss, 0]
        rows.append(dict(trial=os.path.basename(d), single_stance_frames=int(ss.sum()), cop_speed_r_mps=float(zm["cop_speed_r"]),
                         keep_r_share=float(zm["keep_r"].mean()), cop_in_calcn_x_min_m=float(x.min()),
                         cop_in_calcn_x_p05_m=float(np.percentile(x, 5)), cop_in_calcn_x_p95_m=float(np.percentile(x, 95)),
                         cop_in_calcn_x_max_m=float(x.max())))
    return pd.DataFrame(rows)


# ─── shadow trial folders for the input variants ─────────────────────────────────────────────────────────────────

def shadow(src, root, name, com_override=None, fp_override=None, ik_coords_override=None):
    """Folder `name` under root that links every entry of src, with com_features.npz, the fpopt cache and/or the
    pelvis angles of ik_results.mot replaced. Position overrides: dict(calcn_r_pos=..., calcn_l_pos=...) on the
    com_features time grid."""
    d = os.path.join(root, name)
    os.makedirs(d)
    for e in os.listdir(src):
        os.symlink(os.path.join(src, e), os.path.join(d, e))
    z = np.load(os.path.join(src, "com_features.npz"))
    if com_override is not None:
        os.remove(os.path.join(d, "com_features.npz"))
        np.savez(os.path.join(d, "com_features.npz"), **{k: z[k] for k in z.files if k not in com_override},
                 **{k: np.asarray(v, np.float32) for k, v in com_override.items()})
    if ik_coords_override is not None:
        os.remove(os.path.join(d, "ik_results.mot"))
        write_ik(os.path.join(src, "ik_results.mot"), os.path.join(d, "ik_results.mot"), ik_coords_override)
    from pipeline.compute_com_features import foot_point_filename
    fp_name = foot_point_filename(FPOPT)
    if os.path.lexists(os.path.join(d, fp_name)):
        os.remove(os.path.join(d, fp_name))
    pos = fp_override if fp_override is not None else dict(calcn_r_pos=z["calcn_r_pos"], calcn_l_pos=z["calcn_l_pos"])
    np.savez(os.path.join(d, fp_name), time=z["time"],      # same time array as com_features.npz -> identical interpolation
             calcn_r_pos=np.asarray(pos["calcn_r_pos"], np.float32), calcn_l_pos=np.asarray(pos["calcn_l_pos"], np.float32),
             foot_point=np.asarray(FPOPT, float))
    return d


def inference_crosscheck(src, H, mass, preds, coords):
    """Max |F_medial| difference between the features_for path and inference_posture_surrogate.predict_windowed fed
    q from `coords`, the folder's com_features.npz positions and GRF contact flags (the deployment path)."""
    from inference_posture_surrogate import JCFPostureSurrogate, JOINT_ORDER
    ik = pd.read_csv(os.path.join(src, "ik_results.mot"), sep=r"\s+", skiprows=6)
    z = np.load(os.path.join(src, "com_features.npz"))
    g = read_sto(os.path.join(src, "grf_data.mot"))
    T = len(ik)
    q = np.column_stack([coords[c] for c in JOINT_ORDER])
    pel = ik[["pelvis_tx", "pelvis_ty", "pelvis_tz"]].to_numpy(float)
    rel = [z[k] - pel for k in ("com_pos", "calcn_r_pos", "calcn_l_pos")]
    BW = mass * 9.81
    rc = (np.abs(g["calcn_r_force_vy"].to_numpy(float)) > 0.05 * BW).astype(float)
    lc = (np.abs(g["calcn_l_force_vy"].to_numpy(float)) > 0.05 * BW).astype(float)
    idx = np.clip(np.arange(T)[:, None] + np.arange(-10, 10)[None], 0, T - 1)   # train_mlp._windowize, window 20
    out = {}
    for lab, path in CKPTS.items():
        sur = JCFPostureSurrogate(os.path.join(REPO, path), device="cpu")
        o = sur.predict_windowed(q, rel[0][idx], rel[1][idx], rel[2][idx], mass, H,
                                 r_contact_window=rc[idx], l_contact_window=lc[idx])
        f_inf = fmedial(o["fy"], o["mx"], H, MX_PHYSICAL)[0]
        f_ev = fmedial(preds[lab][:, 1], preds[lab][:, 3], H, MX_PHYSICAL)[0]
        out[lab] = float(np.abs(f_inf - f_ev).max())
    return out


def predict_all(d, models, args_ns):
    from tools.evaluate_fmedial import features_for
    from train_mlp import INPUT_SETS
    feats, out, meta = {}, {}, None
    for lab, (m, mu, sd, info) in models.items():
        iset = info["input_set"]
        if iset not in feats:
            built = features_for(d, dict(INPUT_SETS[iset], label_subdir=FEATURE_LABELS), args_ns)
            if built is None:
                raise SystemExit(f"{d}: features_for returned None for {iset}")
            feats[iset] = built
        feat, _labels, fm = feats[iset]
        meta = meta or fm
        with torch.no_grad():
            out[lab] = m((torch.from_numpy(feat).float() - mu) / sd).numpy().astype(np.float64)
    return out, feats, meta


# ─── metrics ─────────────────────────────────────────────────────────────────────────────────────────────────────

def window_masks(t):
    m = {k: (t >= a - 1e-6) & (t <= b + 1e-6) for k, (a, b) in WINDOWS.items()}
    m["S1+S2"] = m["S1"] | m["S2"]
    return m


def stats(f, ax, mo, t, mask):
    i = np.argmax(np.where(mask, f, -np.inf))
    return dict(n=int(mask.sum()), mean=float(f[mask].mean()), peak=float(f[mask].max()), t_peak=float(t[i]),
                min=float(f[mask].min()), frac_over_limit=float((f[mask] > LIMIT).mean()),
                axial_mean=float(ax[mask].mean()), moment_mean=float(mo[mask].mean()))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trial", default=TRIAL)
    ap.add_argument("--out", default="results/thesis/sep23")
    ap.add_argument("--skip-fk-check", action="store_true")
    a = ap.parse_args(argv)
    os.chdir(REPO)
    torch.set_num_threads(1)
    out_dir = os.path.join(REPO, a.out)
    os.makedirs(out_dir, exist_ok=True)
    src = os.path.join(REPO, a.trial)
    meta_j = json.load(open(os.path.join(src, "metadata.json")))
    H, mass = float(meta_j["height_m"]), float(meta_j["mass_kg"])

    from tools.evaluate_fmedial import load_model
    models = {lab: load_model(os.path.join(REPO, p), "cpu") for lab, p in CKPTS.items()}
    ck_rows = []
    for lab, p in CKPTS.items():
        ck = torch.load(os.path.join(REPO, p), map_location="cpu", weights_only=True)
        ca = (ck.get("provenance") or {}).get("call_args") or {}
        ck_rows.append(dict(model=lab, path=p, input_set=ck["input_set"], best_epoch=int(ck["epoch"]) + 1,
                            val_loss=float(ck["val_loss"]), init_seed=ca.get("init_seed"), split_seed=ca.get("seed"),
                            foot_point=str(ck.get("foot_point")), label_subdir=ck.get("label_subdir") or "jcf_output",
                            stored_mx_medial_sign=ck.get("mx_medial_sign")))
    ck_df = pd.DataFrame(ck_rows)

    args_ns = SimpleNamespace(deploy_features=True, pelvis="running", max_peak=1000.0, max_moment=1000.0,
                              label_mask=False, label_mask_subdir="jcf_output_sojr")

    # pelvis angles as written (copied URDF values) and converted to OpenSim's convention
    t_ik, coords_w = ik_coords(src)
    coords_c, rt_err = urdf_pelvis_to_opensim(coords_w)
    z = np.load(os.path.join(src, "com_features.npz"))
    if not np.allclose(z["time"], t_ik):
        raise SystemExit("com_features.npz and ik_results.mot are on different time grids")
    wm_ik = window_masks(t_ik)
    lab_csv = pd.read_csv(os.path.join(src, "labels.csv"))
    ph = lab_csv["phase"].to_numpy()
    if len(ph) != len(t_ik) or not np.allclose(lab_csv["time"].to_numpy(float), t_ik):
        raise SystemExit("labels.csv is not on the IK grid")

    geo_df, com_err, urdf_frames = foot_geometry(src, {"as_written": coords_w, "converted": coords_c}, z, H, ph, wm_ik)
    pconv = pelvis_conversion_table(coords_w, coords_c, com_err, wm_ik)
    cop_ref = cop_reference()

    # foot points for the variants (converted angles)
    fk = OsimChain(os.path.join(src, "scaled_model.osim"))
    own_origin, scaled_pts = {}, {"origin": {}, "fpopt": {}}
    own_shift_vs_scaled = 0.0
    for s in "rl":
        Ru_c, _pu = urdf_frames["converted"][f"calcn_{s}"]
        x = z[f"calcn_{s}_pos"].astype(float)
        own_origin[f"calcn_{s}_pos"] = x - Ru_c @ np.asarray(FPOPT)
        R, p0 = fk.body_pose(f"calcn_{s}", coords_c)
        own_shift_vs_scaled = max(own_shift_vs_scaled, float(np.abs((x - R @ np.asarray(FPOPT)) - own_origin[f"calcn_{s}_pos"]).max()))
        scaled_pts["origin"][f"calcn_{s}_pos"] = p0
        scaled_pts["fpopt"][f"calcn_{s}_pos"] = p0 + R @ np.asarray(FPOPT)

    variants = {}
    with tempfile.TemporaryDirectory(prefix="sep23_") as tmp:
        dirs = {"export": shadow(src, tmp, "export", ik_coords_override=coords_c),
                "export_copied": shadow(src, tmp, "export_copied"),
                "own": shadow(src, tmp, "own", com_override=own_origin, ik_coords_override=coords_c),
                "scaled": shadow(src, tmp, "scaled", com_override=scaled_pts["origin"], fp_override=scaled_pts["fpopt"],
                                 ik_coords_override=coords_c)}
        for name in VARIANTS:
            variants[name] = predict_all(dirs[name], models, args_ns)
    preds, feats, fmeta = variants["export"]
    isets = list(feats)
    same = all(np.array_equal(feats[isets[0]][0], feats[k][0]) for k in isets[1:]) and \
        all(np.array_equal(variants["export_copied"][1][isets[0]][0], variants["export_copied"][1][k][0]) for k in isets[1:])
    if not same:
        raise SystemExit("export variants: input sets did not receive identical feature matrices")
    # the two export variants may differ only in the three pelvis-angle columns (q's first three channels)
    diff_cols = sorted(set(np.where((feats[isets[0]][0] != variants["export_copied"][1][isets[0]][0]).any(0))[0].tolist()))
    if not set(diff_cols) <= {0, 1, 2}:
        raise SystemExit(f"export vs export_copied differ outside the pelvis angles: columns {diff_cols}")
    t = np.asarray(fmeta["t"], float)
    T = len(t)
    if T != len(t_ik) or not np.allclose(t, t_ik):
        raise SystemExit(f"feature grid ({T} frames) differs from the 100 Hz trajectory grid ({len(t_ik)})")
    wm = window_masks(t)
    for w, (a0, a1) in WINDOWS.items():
        m = wm[w]
        if not (ph[m] == "right").all() or (ph[np.where(m)[0][0] - 1] == "right") or (ph[np.where(m)[0][-1] + 1] == "right"):
            raise SystemExit(f"window {w} is not a whole phase=='right' run of labels.csv")

    # labels (context only)
    pre = "walker_knee_r_on_tibia_r_in_tibia_r_"
    BW = mass * 9.81
    lab_src = {"stored_no_muscle": os.path.join(src, "jcf_output", "BatchJCF_JointReaction_ReactionLoads.sto"),
               "sojr_unusable": os.path.join(src, "jcf_output_sojr", "FullDuration_JointReaction_ReactionLoads.sto")}
    labels = {}
    for k, pth in lab_src.items():
        j = read_sto(pth)
        if len(j) != T or not np.allclose(j["time"].to_numpy(float), t):
            raise SystemExit(f"{pth}: not on the 645-frame grid")
        labels[k] = (j[pre + "fy"].to_numpy(float) / BW, j[pre + "mx"].to_numpy(float) / (BW * H))
    keep_r = np.load(os.path.join(src, "jcf_output_sojr", "label_mask.npz"))["keep_r"]

    # per-frame table (export variant = primary; *_copied = pelvis angles as written)
    pf = pd.DataFrame(dict(t=t, phase=ph, r_contact=np.asarray(fmeta["r_contact"], bool),
                           window=np.select([wm["S1"], wm["S2"]], ["S1", "S2"], "")))
    fm, fm_cp = {}, {}
    for lab, p in preds.items():
        f, ax, mo = fmedial(p[:, 1], p[:, 3], H, MX_PHYSICAL)
        fm[lab] = (f, ax, mo)
        pf[f"{lab}_fmed"], pf[f"{lab}_axial"], pf[f"{lab}_moment"] = f, ax, mo
    for lab, p in variants["export_copied"][0].items():
        fm_cp[lab] = fmedial(p[:, 1], p[:, 3], H, MX_PHYSICAL)
        pf[f"{lab}_fmed_copied"] = fm_cp[lab][0]
    for k, (fy, mx) in labels.items():
        f, ax, mo = fmedial(fy, mx, H, MX_PHYSICAL)
        fm[k] = (f, ax, mo)
        pf[f"{k}_fmed"] = f
    pf["labels_csv_fmed_pipeline"] = lab_csv["f_medial_bw"].to_numpy(float)
    pf["sojr_keep_r"] = keep_r.astype(bool)
    pf["seed_range"] = np.ptp(np.stack([fm[s][0] for s in SEEDS]), 0)
    pf["seed_mean"] = np.mean(np.stack([fm[s][0] for s in SEEDS]), 0)
    pf["seed_range_copied"] = np.ptp(np.stack([fm_cp[s][0] for s in SEEDS]), 0)
    for k in PELVIS_ROT:
        pf[f"{k}_as_written"], pf[f"{k}_converted"] = coords_w[k], coords_c[k]
    pf.to_csv(os.path.join(out_dir, "per_frame.csv"), index=False, float_format="%.5f")

    # summary per model x window (primary)
    rows = []
    for lab in list(CKPTS) + ["seed_mean"] + list(labels):
        if lab == "seed_mean":
            f, ax, mo = [np.stack([fm[s][i] for s in SEEDS]).mean(0) for i in range(3)]
        else:
            f, ax, mo = fm[lab]
        for w in ("S1", "S2", "S1+S2"):
            rows.append(dict(model=lab, window=w, **stats(f, ax, mo, t, wm[w])))
        rows.append(dict(model=lab, window="all", **stats(f, ax, mo, t, np.ones(T, bool))))
    summ = pd.DataFrame(rows)
    summ.to_csv(os.path.join(out_dir, "summary.csv"), index=False, float_format="%.4f")

    # seed spread (primary)
    srows = []
    seed_stack = np.stack([fm[s][0] for s in SEEDS])
    rng_f = np.ptp(seed_stack, 0)
    sd_f = seed_stack.std(0, ddof=1)
    for w in ("S1", "S2", "S1+S2", "all"):
        m = np.ones(T, bool) if w == "all" else wm[w]
        means = [float(fm[s][0][m].mean()) for s in SEEDS]
        peaks = [float(fm[s][0][m].max()) for s in SEEDS]
        fr = [float((fm[s][0][m] > LIMIT).mean()) for s in SEEDS]
        i = np.argmax(np.where(m, rng_f, -np.inf))
        srows.append(dict(window=w, n=int(m.sum()), frame_range_mean=float(rng_f[m].mean()),
                          frame_range_median=float(np.median(rng_f[m])), frame_range_max=float(rng_f[m].max()),
                          t_frame_range_max=float(t[i]), frame_sd_mean=float(sd_f[m].mean()),
                          stance_mean_min=min(means), stance_mean_max=max(means), stance_mean_range=max(means) - min(means),
                          peak_min=min(peaks), peak_max=max(peaks), peak_range=max(peaks) - min(peaks),
                          frac_over_min=min(fr), frac_over_max=max(fr),
                          frac_frames_seeds_disagree_on_limit=float(
                              ((seed_stack[:, m] > LIMIT).any(0) & ~(seed_stack[:, m] > LIMIT).all(0)).mean()),
                          # supplementary, not a replacement: the two seeds whose early stopping ran past epoch 50
                          s1_s3_mean_abs=float(np.abs(fm["qc_s1"][0] - fm["qc_s3"][0])[m].mean()),
                          s1_s3_max_abs=float(np.abs(fm["qc_s1"][0] - fm["qc_s3"][0])[m].max())))
    spread = pd.DataFrame(srows)
    spread.to_csv(os.path.join(out_dir, "spread.csv"), index=False, float_format="%.4f")

    # pairwise differences (primary)
    pairs = [("qc_origin", "qc_s1", "foot point under corrected training (export inputs)"),
             ("fpopt_legacy", "qc_s1", "legacy vs corrected, optimizer foot point"),
             ("sept_legacy", "qc_origin", "legacy vs corrected, calcn-origin foot point"),
             ("sept_legacy", "fpopt_legacy", "legacy pair (foot point and run)")]
    prow = []
    for x, y, what in pairs:
        dlt = fm[x][0] - fm[y][0]
        for w in ("S1", "S2", "S1+S2", "all"):
            m = np.ones(T, bool) if w == "all" else wm[w]
            prow.append(dict(a=x, b=y, what=what, window=w, mean_diff=float(dlt[m].mean()),
                             mean_abs_diff=float(np.abs(dlt[m]).mean()), max_abs_diff=float(np.abs(dlt[m]).max()),
                             r=float(np.corrcoef(fm[x][0][m], fm[y][0][m])[0, 1])))
    seed_mean_f = seed_stack.mean(0)
    for x in ("qc_origin", "fpopt_legacy", "sept_legacy"):
        dlt = fm[x][0] - seed_mean_f
        m = wm["S1+S2"]
        out_band = (fm[x][0] < seed_stack.min(0)) | (fm[x][0] > seed_stack.max(0))
        prow.append(dict(a=x, b="seed_mean", what="vs mean of the three seeds", window="S1+S2",
                         mean_diff=float(dlt[m].mean()), mean_abs_diff=float(np.abs(dlt[m]).mean()),
                         max_abs_diff=float(np.abs(dlt[m]).max()), r=float(np.corrcoef(fm[x][0][m], seed_mean_f[m])[0, 1]),
                         frac_outside_seed_band=float(out_band[m].mean())))
    for lab in CKPTS:     # converted minus copied pelvis angles, same model
        dlt = fm[lab][0] - fm_cp[lab][0]
        for w in ("S1", "S2", "S1+S2", "all"):
            m = np.ones(T, bool) if w == "all" else wm[w]
            prow.append(dict(a=lab, b=f"{lab}_copied", what="converted minus copied pelvis angles", window=w,
                             mean_diff=float(dlt[m].mean()), mean_abs_diff=float(np.abs(dlt[m]).mean()),
                             max_abs_diff=float(np.abs(dlt[m]).max()), r=float(np.corrcoef(fm[lab][0][m], fm_cp[lab][0][m])[0, 1])))
    pair_df = pd.DataFrame(prow)
    pair_df.to_csv(os.path.join(out_dir, "pairs.csv"), index=False, float_format="%.4f")

    # input variants
    vrows = []
    for vname in VARIANTS:
        vp = variants[vname][0]
        for lab, p in vp.items():
            f, ax, mo = fmedial(p[:, 1], p[:, 3], H, MX_PHYSICAL)
            for w in ("S1", "S2", "S1+S2"):
                vrows.append(dict(variant=vname, model=lab, window=w, **stats(f, ax, mo, t, wm[w])))
        vstack = np.stack([fmedial(vp[s][:, 1], vp[s][:, 3], H, MX_PHYSICAL)[0] for s in SEEDS])
        vr = np.ptp(vstack, 0)
        for w in ("S1", "S2", "S1+S2"):
            m = wm[w]
            sm = vstack[:, m].mean(1)
            vrows.append(dict(variant=vname, model="seed_range", window=w, n=int(m.sum()), mean=float(vr[m].mean()),
                              peak=float(vr[m].max()), stance_mean_range=float(np.ptp(sm)),
                              frac_over_limit=float((vstack[:, m] > LIMIT).mean())))
    # pipeline-sign readings of the legacy models: export_copied reproduces SEP3 / SEP4 (facts_verified.md), which
    # were taken with the copied angles; export gives the same with the converted angles
    for vname in ("export_copied", "export"):
        for lab in ("sept_legacy", "fpopt_legacy"):
            p = variants[vname][0][lab]
            f, ax, mo = fmedial(p[:, 1], p[:, 3], H, MX_PIPELINE)
            for w in ("S1", "S2", "S1+S2"):
                vrows.append(dict(variant=f"{vname}_pipeline_sign", model=lab, window=w, **stats(f, ax, mo, t, wm[w])))
    var_df = pd.DataFrame(vrows)
    var_df.to_csv(os.path.join(out_dir, "variants.csv"), index=False, float_format="%.4f")

    # labels context (pipeline sign too)
    crow = []
    for k, (fy, mx) in labels.items():
        for sname, sign in (("physical", MX_PHYSICAL), ("pipeline", MX_PIPELINE)):
            f, ax, mo = fmedial(fy, mx, H, sign)
            for w in ("S1", "S2", "S1+S2"):
                crow.append(dict(labels=k, sign=sname, window=w, **stats(f, ax, mo, t, wm[w]),
                                 sojr_keep_r_frames=int(keep_r[wm[w]].sum())))
    ctx = pd.DataFrame(crow)
    ctx.to_csv(os.path.join(out_dir, "labels_context.csv"), index=False, float_format="%.4f")
    geo_df.to_csv(os.path.join(out_dir, "foot_geometry.csv"), index=False, float_format="%.4f")
    pconv.to_csv(os.path.join(out_dir, "pelvis_conversion.csv"), index=False, float_format="%.4f")
    cop_ref.to_csv(os.path.join(out_dir, "cop_reference.csv"), index=False, float_format="%.4f")

    if not a.skip_fk_check:
        fkv, n_cache = fk_validation()
        fkv.to_csv(os.path.join(out_dir, "fk_validation.csv"), index=False, float_format="%.3e")
    else:
        fkv, n_cache = None, None

    try:
        git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    except Exception:
        git = None
    xcheck = inference_crosscheck(src, H, mass, preds, coords_c)
    xcheck_cp = inference_crosscheck(src, H, mass, variants["export_copied"][0], coords_w)
    json.dump(dict(created=datetime.datetime.now().isoformat(timespec="seconds"), git=git, trial=a.trial, H=H, mass=mass,
                   phi=PHI, d_m=D_M, limit=LIMIT, mx_sign="physical (-mx, right knee)", windows=WINDOWS,
                   feature_label_grid=FEATURE_LABELS, deploy_features=True, export_inputs_identical=bool(same),
                   pelvis_angles_primary="converted from the URDF's axes (urdf_pelvis_to_opensim)",
                   pelvis_conversion_roundtrip_max_err=rt_err,
                   export_vs_copied_feature_columns_differing=diff_cols,
                   own_origin_urdf_vs_scaled_rotation_max_m=own_shift_vs_scaled,
                   inference_crosscheck_max_abs_bw=xcheck, inference_crosscheck_copied_max_abs_bw=xcheck_cp,
                   checkpoints=ck_rows), open(os.path.join(out_dir, "run_info.json"), "w"), indent=1)
    write_readme(out_dir, ck_df, summ, spread, pair_df, var_df, ctx, geo_df, pconv, cop_ref, fkv, n_cache,
                 xcheck, xcheck_cp, own_shift_vs_scaled)
    print(open(os.path.join(out_dir, "README.md")).read())


def _f(x, n=2):
    return f"{x:.{n}f}"


def write_readme(out_dir, ck_df, summ, spread, pair_df, var_df, ctx, geo_df, pconv, cop_ref, fkv, n_cache,
                 xcheck, xcheck_cp, own_shift):
    S = summ.set_index(["model", "window"])
    V = var_df.set_index(["variant", "model", "window"])
    G = geo_df.set_index(["q_pelvis", "model", "foot", "window"])
    PC = pconv.set_index("window")
    L = []
    L.append("# S4: seed spread on the optimizer's reference motion (Sep 23 trajectory)\n")
    L.append("Written by `analysis/thesis/sep23_seeds.py` (no ground truth on this motion; see the script docstring). "
             "F_medial = 0.5|F_y| + M_med H/0.045 m with the physical sign for every checkpoint; 1.4 BW limit. "
             "Primary inputs: the trial folder's q with the optimizer's pelvis angles converted from the URDF's axes to "
             "OpenSim's convention (section 'Pelvis angles'), its contact flags, and the optimizer's exported COM and foot "
             "point (com_features.npz); identical for all six checkpoints (asserted), deployment-style windows (exact time, "
             "100 Hz, 20 frames). The previous run fed the pelvis angles as written in ik_results.mot; that is now the "
             "`export_copied` variant. Windows: right single stance S1 1.98-2.77 s (80 frames), S2 4.19-4.83 s (65 frames).\n")
    L.append("## Checkpoints\n")
    L.append("| model | best epoch | val loss | init seed | foot point | trained labels |")
    L.append("|---|---|---|---|---|---|")
    for r in ck_df.itertuples():
        seed = "-" if pd.isna(r.init_seed) else int(r.init_seed)
        L.append(f"| {r.model} | {r.best_epoch} | {r.val_loss:.3f} | {seed} | {r.foot_point} | {r.label_subdir} |")
    L.append("\n## Stance readings (primary: converted pelvis angles, exported positions, physical sign)\n")
    L.append("| model | S1 mean | S1 peak | S1 >1.4 | S2 mean | S2 peak | S2 >1.4 | S1+S2 mean | S1+S2 >1.4 | axial / moment (S1+S2) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for m in list(CKPTS) + ["seed_mean"]:
        a, b, c = S.loc[(m, "S1")], S.loc[(m, "S2")], S.loc[(m, "S1+S2")]
        L.append(f"| {m} | {_f(a['mean'])} | {_f(a.peak)} | {a.frac_over_limit:.0%} | {_f(b['mean'])} | {_f(b.peak)} | "
                 f"{b.frac_over_limit:.0%} | {_f(c['mean'])} | {c.frac_over_limit:.0%} | {_f(c.axial_mean)} / {_f(c.moment_mean)} |")
    L.append("\n## Across-seed spread (qc_s1..s3; per-frame max - min, BW; primary inputs)\n")
    L.append("| window | frames | mean | median | max (at t) | range of stance means | range of peaks | share >1.4 (min-max) | frames where seeds disagree on the limit | s1 vs s3 only: mean / max abs |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in spread.itertuples():
        L.append(f"| {r.window} | {r.n} | {_f(r.frame_range_mean)} | {_f(r.frame_range_median)} | {_f(r.frame_range_max)} ({r.t_frame_range_max:.2f} s) | "
                 f"{_f(r.stance_mean_min)}-{_f(r.stance_mean_max)} ({_f(r.stance_mean_range)}) | {_f(r.peak_min)}-{_f(r.peak_max)} ({_f(r.peak_range)}) | "
                 f"{r.frac_over_min:.0%}-{r.frac_over_max:.0%} | {r.frac_frames_seeds_disagree_on_limit:.0%} | "
                 f"{_f(r.s1_s3_mean_abs)} / {_f(r.s1_s3_max_abs)} |")
    L.append("\nqc_s2 stopped at epoch 8 (val 0.363 vs 0.300 / 0.305 for s1 / s3): real run-to-run variance of the "
             "recipe, kept in every spread figure; the s1-vs-s3 column is supplementary.")
    L.append("\n## Pairwise differences (a - b, BW; primary inputs)\n")
    L.append("| a | b | window | mean diff | mean abs diff | max abs diff | r | outside seed band |")
    L.append("|---|---|---|---|---|---|---|---|")
    for r in pair_df[~pair_df.b.str.endswith("_copied")].itertuples():
        ob = "" if pd.isna(getattr(r, "frac_outside_seed_band", np.nan)) else f"{r.frac_outside_seed_band:.0%}"
        L.append(f"| {r.a} | {r.b} | {r.window} | {r.mean_diff:+.2f} | {_f(r.mean_abs_diff)} | {_f(r.max_abs_diff)} | {r.r:.2f} | {ob} |")

    L.append("\n## Converted vs copied pelvis angles (same exported positions)\n")
    L.append("| model | S1 mean copied -> converted | S2 mean copied -> converted | S1 / S2 peak converted | S1+S2 >1.4 copied -> converted | converted - copied, S1+S2 mean (max abs) |")
    L.append("|---|---|---|---|---|---|")
    PD = pair_df[pair_df.b.str.endswith("_copied")].set_index(["a", "window"])
    for m in CKPTS:
        c1, c2 = V.loc[("export_copied", m, "S1")], V.loc[("export_copied", m, "S2")]
        e1, e2 = V.loc[("export", m, "S1")], V.loc[("export", m, "S2")]
        cb, eb = V.loc[("export_copied", m, "S1+S2")], V.loc[("export", m, "S1+S2")]
        dd = PD.loc[(m, "S1+S2")]
        L.append(f"| {m} | {_f(c1['mean'])} -> {_f(e1['mean'])} | {_f(c2['mean'])} -> {_f(e2['mean'])} | {_f(e1.peak)} / {_f(e2.peak)} | "
                 f"{cb.frac_over_limit:.0%} -> {eb.frac_over_limit:.0%} | {dd.mean_diff:+.2f} ({_f(dd.max_abs_diff)}) |")
    cells = []
    for v in ("export_copied", "export"):
        r1, r2, rb = V.loc[(v, "seed_range", "S1")], V.loc[(v, "seed_range", "S2")], V.loc[(v, "seed_range", "S1+S2")]
        cells.append(f"{v}: frame range mean {_f(rb['mean'])}, max {_f(rb.peak)}; stance-mean range S1 {_f(r1.stance_mean_range)}, S2 {_f(r2.stance_mean_range)}")
    L.append("\nSeed spread, " + "; ".join(cells) + ".\n")

    L.append("## Input variants (S1+S2 mean / peak / share >1.4)\n")
    Vs = var_df[var_df.window == "S1+S2"]
    L.append("| model | export (primary) | export_copied | own foot point | scaled-model FK |")
    L.append("|---|---|---|---|---|")
    for m in list(CKPTS):
        cells = []
        for v in VARIANTS:
            r = Vs[(Vs.variant == v) & (Vs.model == m)].iloc[0]
            cells.append(f"{_f(r['mean'])} / {_f(r.peak)} / {r.frac_over_limit:.0%}")
        L.append(f"| {m} | " + " | ".join(cells) + " |")
    cells = []
    for v in VARIANTS:
        r = Vs[(Vs.variant == v) & (Vs.model == "seed_range")].iloc[0]
        cells.append(f"seed range mean {_f(r['mean'])}, max {_f(r.peak)}")
    L.append("| seeds | " + " | ".join(cells) + " |")
    L.append("\n`export_copied`: pelvis angles as written (the repo converters' copy convention: label_trajectory_csv.py, "
             "infer_trajectory_csv.py, PN3 ::q_urdf); not checked against the optimizer's own code. `own`: only the "
             "calcn-origin checkpoints change (exported point moved 13 cm back along the generic URDF's calcn axes; the "
             f"scaled model's axes give the same point to {own_shift*100:.1f} cm). `scaled`: every foot point from the borrowed "
             "scaled OpenSim model's forward kinematics with the converted angles (about 2.5 cm from the exported point, "
             "foot_geometry.csv); sensitivity only.\n")
    P = var_df[var_df.variant.str.endswith("_pipeline_sign")].set_index(["variant", "model", "window"])

    def _pp(v, m):
        return (f"{_f(P.loc[(v, m, 'S1')]['mean'])} / {_f(P.loc[(v, m, 'S2')]['mean'])}, "
                f"peaks {_f(P.loc[(v, m, 'S1')].peak)} / {_f(P.loc[(v, m, 'S2')].peak)}")
    L.append("Legacy readings in the pipeline sign (+mx), S1 / S2 means: with the copied angles (the convention of the "
             f"earlier facts) sept_legacy {_pp('export_copied_pipeline_sign', 'sept_legacy')} (facts SEP3: 1.46 / 1.04, "
             f"peaks 2.42 / 2.16); fpopt_legacy {_pp('export_copied_pipeline_sign', 'fpopt_legacy')} (SEP4: 1.27 / 0.94, "
             f"peaks 1.99 / 1.79). With the converted angles: sept_legacy {_pp('export_pipeline_sign', 'sept_legacy')}; "
             f"fpopt_legacy {_pp('export_pipeline_sign', 'fpopt_legacy')}.\n")
    L.append("## Labels on this motion: context only, NOT ground truth\n")
    L.append("| labels | sign | S1 mean | S1 peak | S2 mean | S2 peak | S1+S2 >1.4 | SO+JR keep_r frames |")
    L.append("|---|---|---|---|---|---|---|---|")
    C = ctx.set_index(["labels", "sign", "window"])
    for k in ("stored_no_muscle", "sojr_unusable"):
        for s in ("physical", "pipeline"):
            a, b, c = C.loc[(k, s, "S1")], C.loc[(k, s, "S2")], C.loc[(k, s, "S1+S2")]
            L.append(f"| {k} | {s} | {_f(a['mean'])} | {_f(a.peak)} | {_f(b['mean'])} | {_f(b.peak)} | {c.frac_over_limit:.0%} | {int(c.sojr_keep_r_frames)} |")
    osr = G.loc[("as_written", "scaled", "r", "right_single")]      # OpenSim's own foot in the labelling run
    L.append("\nstored_no_muscle = jcf_output/BatchJCF_JointReaction_ReactionLoads.sto (JointReaction without muscle forces; "
             "labels.csv f_medial_bw is its pipeline-sign column). sojr_unusable = jcf_output_sojr/FullDuration_... (SO+JR; "
             "label_mask.npz keeps none of its right-knee frames). Both were computed by OpenSim on this folder's "
             "ik_results.mot, i.e. with the copied pelvis angles, where the scaled model's foot point sits "
             f"{_f(osr.fpopt_vs_export_mean_m*100,1)} cm (mean over right single stance) from the applied COP (next section).\n")

    L.append("## Pelvis angles and foot geometry\n")
    pb = PC.loc["S1+S2"]
    pa = PC.loc["all"]
    L.append("pipeline/label_trajectory_csv.py (line 81, `ik[:, 0:3] = q[:, 3:6]`) copies the optimizer's URDF pelvis angles "
             "into OpenSim's pelvis_tilt / list / rotation. assets/Rajagopal2015.urdf rotates its pelvis about the URDF's "
             "own z (up), x and y axes (joints pelvis_tilt / pelvis_list / pelvis_rotation; pn3_redis/model.py docstring), "
             "so the copied 'pelvis_tilt' is a rotation about the vertical. Converted (R = C^T Rz(q3) Rx(q4) Ry(q5) C, "
             "OpenSim Z-X-Y angles; to first order tilt = -q5, list = q4, rotation = q3), the angles differ from the written "
             f"ones by up to {_f(pb.d_pelvis_tilt_max_deg,1)} deg (tilt), {_f(pb.d_pelvis_list_max_deg,1)} deg (list) and "
             f"{_f(pb.d_pelvis_rotation_max_deg,1)} deg (rotation) in S1+S2 ({_f(pa.d_pelvis_tilt_max_deg,1)} / "
             f"{_f(pa.d_pelvis_list_max_deg,1)} / {_f(pa.d_pelvis_rotation_max_deg,1)} deg over the whole motion; "
             "pelvis_conversion.csv).\n")
    L.append("| pelvis angles | body model | right foot, right single stance: fpopt point to exported point, mean / max (cm) | left foot, left single stance (cm) | fpopt point x range in S1 / S2 (cm) | COP in calcn frame, right single stance: x (m) |")
    L.append("|---|---|---|---|---|---|")
    for qn in ("as_written", "converted"):
        for mn in ("urdf", "scaled"):
            r, l_ = G.loc[(qn, mn, "r", "right_single")], G.loc[(qn, mn, "l", "left_single")]
            s1, s2 = G.loc[(qn, mn, "r", "S1")], G.loc[(qn, mn, "r", "S2")]
            L.append(f"| {qn} | {'generic URDF (1.70 m)' if mn == 'urdf' else 'borrowed scaled model'} | "
                     f"{_f(r.fpopt_vs_export_mean_m*100)} / {_f(r.fpopt_vs_export_max_m*100)} | "
                     f"{_f(l_.fpopt_vs_export_mean_m*100)} / {_f(l_.fpopt_vs_export_max_m*100)} | "
                     f"{_f(s1.fpopt_world_range_x_m*100)} / {_f(s2.fpopt_world_range_x_m*100)} | "
                     f"{_f(r.cop_in_calcn_x_min_m,3)} to {_f(r.cop_in_calcn_x_max_m,3)} |")
    e1, e2 = G.loc[("converted", "urdf", "r", "S1")], G.loc[("converted", "urdf", "r", "S2")]
    rc = G.loc[("converted", "urdf", "r", "right_single")]
    lc = G.loc[("converted", "urdf", "l", "left_single")]
    L.append(f"\nThe exported point itself moves {_f(e1.export_world_range_x_m*100)} / {_f(e2.export_world_range_x_m*100)} cm in x "
             f"over S1 / S2. COP = exported point to {rc.cop_equals_export_max_m:.1e} m (grf_data.mot). Generic-URDF COM vs "
             f"the exported COM: {_f(pa.urdf_com_vs_export_as_written_mean_m*100)} cm mean ({_f(pa.urdf_com_vs_export_as_written_max_m*100)} max) "
             f"as written, {_f(pa.urdf_com_vs_export_converted_mean_m*100)} cm ({_f(pa.urdf_com_vs_export_converted_max_m*100)}) converted.\n")
    if len(cop_ref):
        ref = "; ".join(f"{r.trial} {_f(r.cop_in_calcn_x_p05_m,3)} to {_f(r.cop_in_calcn_x_p95_m,3)} m" for r in cop_ref.itertuples())
    else:
        ref = "none found"
    L.append("Reading. With the converted angles the optimizer's own body model reproduces its export to millimetres, so "
             "the conversion is right and the copied angles are not OpenSim angles. The optimizer's feet are planted, and "
             "the COP is pinned at its mid-foot contact point: in the URDF calcn frame it stays at x = "
             f"{_f(rc.cop_in_calcn_x_min_m,3)} to {_f(rc.cop_in_calcn_x_max_m,3)} m, y = {_f(rc.cop_in_calcn_y_min_m,3)} to "
             f"{_f(rc.cop_in_calcn_y_max_m,3)} m over every right single stance (left: x = {_f(lc.cop_in_calcn_x_min_m,3)} to "
             f"{_f(lc.cop_in_calcn_x_max_m,3)} m), i.e. on the fpopt point (0.13, -0.01, 0), 13 cm ahead of the calcn origin, "
             "with no heel-to-toe progression. For comparison, the right-foot COP of moving-COP Moore training trials runs "
             f"over x (p5-p95, own scaled calcn frame; cop_reference.csv): {ref}. So the COP is not pinned to the heel, and "
             "the foot does not slide under it: the 12-15 cm slide reported in the previous version of this file came from "
             "the copied pelvis angles. The borrowed scaled model keeps a near-constant offset of about 2.5 cm after the "
             "conversion (segment scaling; COP at x = 0.15 m in its calcn frame).\n")
    L.append("Consequences. (1) This folder's OpenSim kinematics (ik_results.mot) misplace the pelvis orientation, which puts "
             f"the scaled model's foot point {_f(osr.fpopt_vs_export_mean_m*100,1)} cm (right single stance mean; up to "
             f"{_f(osr.fpopt_vs_export_max_m*100,1)} cm) away from the COP that SO and JR apply, against "
             f"{_f(G.loc[('converted', 'scaled', 'r', 'right_single')].fpopt_vs_export_mean_m*100,1)} cm with the converted "
             "angles; both label "
             "sets above were computed that way, on top of the pinned COP (which alone makes label_mask's fixed-COP rule "
             "drop the trial). Whether the mis-mapping causes the 27.5 BW SO+JR peak is not tested here (no OpenSim run). "
             "(2) Re-labelling optimizer trajectories with SO+JR needs the pelvis conversion in "
             "pipeline/label_trajectory_csv.py first. (3) infer_trajectory_csv.py copies the angles the same way, so a "
             "model fed an optimizer export through it sees a pelvis up to ~13 deg off (the PN3 driver feeds its surrogate "
             "OpenSim-correct ::q; only its ::q_urdf output key follows the copy convention).\n")
    L.append("Cross-check: inference_posture_surrogate.predict_windowed on the same q, positions and contact flags "
             f"reproduces every model's F_medial to {max(xcheck.values()):.1e} BW with the converted angles and "
             f"{max(xcheck_cp.values()):.1e} BW with the copied ones (run_info.json).\n")
    if fkv is not None:
        L.append(f"Forward kinematics check (`analysis/thesis/osim_fk.py` vs OpenSim caches, {len(fkv)} training trials from "
                 f"{fkv.lab.nunique()} labs, {n_cache} fpopt caches available): max error "
                 f"{fkv.filter(like='_max_err_m').to_numpy().max():.1e} m (fk_validation.csv).\n")
    open(os.path.join(out_dir, "README.md"), "w").write("\n".join(L))


if __name__ == "__main__":
    main()

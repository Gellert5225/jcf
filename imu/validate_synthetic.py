"""End-to-end check of the IMU path on a recorded trial.  [needs opensim and torch: run in the jcf env]

    conda run -n jcf python -m imu.validate_synthetic \
        --trial jcf/full_duration/testing/walking/carter_P010_split0_t10 \
        --out results/imu/val_P010_t10 --t0 2 --t1 6 --fs 100 --noise-deg 1 --drift-deg-per-s 0.5

Steps:
1. virtual IMUs from the trial's mocap IK (synthetic.py), with optional noise and heading drift
2. OpenSense calibration and IMU IK (opensense_ik.py)
3. joint-angle error of the IMU path against the mocap IK
4. foot contact without force plates: world-frame detector on mocap heel/toe (what mocap alone can do) and the
   orientation-only detector on the foot IMUs (what the suit can do), both scored against force-plate contact
5. surrogate F_medial three ways, each compared with OpenSim's labels:
      mocap angles + force-plate contact     the reference: the model with perfect inputs
      IMU angles   + force-plate contact     isolates the effect of IMU kinematics
      IMU angles   + IMU contact             the fully portable pipeline

Only checkpoints without pelvis-translation inputs are valid for the IMU path, because orientation sensors do
not observe where the pelvis is; others are scored on the mocap path only.

Outputs in --out: summary.json, angles_rmse.csv, fmedial.csv, fmedial.png, plus the synthetic and IK files.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from imu import contact                                         # noqa: E402
from imu.opensense_io import read_orientations, to_scipy        # noqa: E402
from imu.opensense_ik import calibrate, run_ik                  # noqa: E402
from imu.synthetic import read_mot, set_pose, synthesize        # noqa: E402

DEFAULT_CKPTS = [
    "models/runs/mlp/best_model_mlp_both_qinst_notrans_pelvisrel_winpos_contact_w20_fmW0p5_e100.pt",
    "models/runs/mlp/best_model_mlp_walking_qinst_notrans_pelvisrel_winpos_contact_w20_fmW0p5_e100_2.pt",
]
PRE = "walker_knee_r_on_tibia_r_in_tibia_r_"
D_M, PHI = 0.045, 0.5


def fk_positions(model_path, names, values):
    """Ground positions of the COM, pelvis origin, heels (calcn) and toes for each row of coordinates."""
    import opensim as osim
    model = osim.Model(model_path)
    state = model.initSystem()
    cs = model.getCoordinateSet()
    coords = {cs.get(i).getName(): cs.get(i) for i in range(cs.getSize())}
    bs = model.getBodySet()
    pts = {k: bs.get(k) for k in ("pelvis", "calcn_r", "calcn_l", "toes_r", "toes_l")}
    zero = osim.Vec3(0, 0, 0)
    out = {k: np.zeros((len(values), 3)) for k in ["com"] + list(pts)}
    for i, row in enumerate(values):
        set_pose(model, state, coords, names, row)
        model.realizePosition(state)
        c = model.calcMassCenterPosition(state)
        out["com"][i] = [c.get(0), c.get(1), c.get(2)]
        for k, b in pts.items():
            v = b.findStationLocationInGround(state, zero)
            out[k][i] = [v.get(0), v.get(1), v.get(2)]
    return out


def surrogate_fmedial(model, q16, pos, rc, lc, mass, height):
    """Windowed surrogate prediction; returns (centre-frame indices, F_medial, Fy, Mx)."""
    W = 20
    n = len(q16) - W + 1
    c = W // 2
    win = lambda x: np.stack([x[i:i + W] for i in range(n)])
    pel = pos["pelvis"]
    kw = dict(r_contact_window=win(rc.astype(np.float32)), l_contact_window=win(lc.astype(np.float32))) if model.include_contact else {}
    p = model.predict_windowed(joint_angles=q16[c:c + n], com_rel_window=win(pos["com"] - pel),
                               calcn_r_rel_window=win(pos["calcn_r"] - pel), calcn_l_rel_window=win(pos["calcn_l"] - pel),
                               mass_kg=mass, height_m=height, **kw)     # pelvis_tx/ty/tz centred over the clip
    return np.arange(c, c + n), np.asarray(p["f_medial"]), np.asarray(p["fy"]), np.asarray(p["mx"])


def err(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)                 # labels are NaN outside the OpenSim output's time range
    a, b = a[ok], b[ok]
    return dict(n=int(ok.sum()), mae=float(np.mean(np.abs(a - b))), rmse=float(np.sqrt(np.mean((a - b) ** 2))),
                bias=float(np.mean(a - b)), r=float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else float("nan"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trial", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--t0", type=float, default=None)
    ap.add_argument("--t1", type=float, default=None)
    ap.add_argument("--fs", type=float, default=100.0, help="suit sample rate to simulate")
    ap.add_argument("--calib-s", type=float, default=1.0)
    ap.add_argument("--noise-deg", type=float, default=0.0)
    ap.add_argument("--noise-corr-s", type=float, default=0.1, help="noise correlation time; 0 = white")
    ap.add_argument("--drift-deg-per-s", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt", nargs="+", default=None)
    ap.add_argument("--contact-thresh-deg-s", type=float, default=150.0)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    trial = a.trial.rstrip("/")
    meta_trial = json.load(open(os.path.join(trial, "metadata.json")))
    mass, height = float(meta_trial["mass_kg"]), float(meta_trial.get("height_m", 1.7))
    model_path = os.path.join(trial, "scaled_model.osim")

    # 1-2: virtual IMUs, calibration, IMU IK
    syn = synthesize(trial, a.out, t0=a.t0, t1=a.t1, calib_s=a.calib_s, fs=a.fs, noise_deg=a.noise_deg,
                     noise_corr_s=a.noise_corr_s, drift_deg_per_s=a.drift_deg_per_s, seed=a.seed)
    orient = os.path.join(a.out, "orientations.sto")
    cal = calibrate(model_path, orient, os.path.join(a.out, "calibrated_model.osim"))
    ik_path = run_ik(cal, orient, os.path.join(a.out, "ik"), t0=syn["calib_s"])

    t_ik, names, v_imu = read_mot(ik_path)
    t_tr, n_tr, v_tr = read_mot(os.path.join(a.out, "truth_ik.mot"))
    v_true = np.column_stack([np.interp(t_ik, t_tr, v_tr[:, j]) for j in range(len(n_tr))])
    t_orig = t_ik + syn["time_offset"]
    # IMU IK also reports the knee's coupled coordinates (knee_angle_*_beta), which the mocap IK file lacks
    shared = [n for n in n_tr if n in names]
    imu_col = lambda n: v_imu[:, names.index(n)]
    true_col = lambda n: v_true[:, n_tr.index(n)]

    # 3: joint-angle error
    from inference_posture_surrogate import JOINT_ORDER, JCFPostureSurrogate
    rows = []
    for n in shared:
        d = imu_col(n) - true_col(n)
        trans = n.endswith(("_tx", "_ty", "_tz"))
        k = 1.0 if trans else 180 / np.pi
        rows.append(dict(coordinate=n, unit="m" if trans else "deg", surrogate_input=n in JOINT_ORDER,
                         rmse=float(np.sqrt(np.mean(d ** 2)) * k), truth_range=float(np.ptp(true_col(n)) * k),
                         observed=not trans))
    ang = pd.DataFrame(rows)
    ang.to_csv(os.path.join(a.out, "angles_rmse.csv"), index=False)

    # 4: contact
    pos_true = fk_positions(model_path, n_tr, v_true)
    pos_imu = fk_positions(model_path, names, v_imu)
    rc_grf, lc_grf = contact.grf_contact(trial, t_orig, mass)
    BW = mass * 9.81
    rc_world = contact.detect_contact([pos_true["calcn_r"], pos_true["toes_r"]], a.fs)
    lc_world = contact.detect_contact([pos_true["calcn_l"], pos_true["toes_l"]], a.fs)
    t_o, quats, _ = read_orientations(orient)
    foot = {s: to_scipy(np.column_stack([np.interp(t_ik, t_o, quats[f"calcn_{s}_imu"][:, j]) for j in range(4)]))
            for s in ("r", "l")}
    rc_imu = contact.detect_contact_orientation(foot["r"], a.fs, a.contact_thresh_deg_s)
    lc_imu = contact.detect_contact_orientation(foot["l"], a.fs, a.contact_thresh_deg_s)
    contact_scores = {
        # world-frame heel/toe detector on the mocap path; it fails on treadmill trials, where the stance foot
        # moves with the belt, so read it only for overground data
        "mocap_world_detector": {"right": contact.compare(rc_world, rc_grf, a.fs), "left": contact.compare(lc_world, lc_grf, a.fs)},
        "imu_orientation_detector": {"right": contact.compare(rc_imu, rc_grf, a.fs), "left": contact.compare(lc_imu, lc_grf, a.fs)},
    }

    # 5: surrogate
    jt, jn, jv = read_mot(os.path.join(trial, "jcf_output", "BatchJCF_JointReaction_ReactionLoads.sto"), angles=False)
    fy_gt = np.interp(t_orig, jt, jv[:, jn.index(PRE + "fy")]) / BW
    mx_gt = np.interp(t_orig, jt, jv[:, jn.index(PRE + "mx")]) / (BW * height)
    fm_gt = PHI * np.abs(fy_gt) + mx_gt * height / D_M
    fm_gt[(t_orig < jt[0]) | (t_orig > jt[-1])] = np.nan
    q_true = np.column_stack([true_col(n) for n in JOINT_ORDER])
    q_imu = np.column_stack([imu_col(n) for n in JOINT_ORDER])
    ckpts = a.ckpt or [p for p in (os.path.join(REPO, c) for c in DEFAULT_CKPTS) if os.path.exists(p)]
    curves = pd.DataFrame({"time_original": t_orig, "opensim": fm_gt, "right_contact_grf": rc_grf})
    model_scores = {}
    stance = rc_grf
    for ck in ckpts:
        m = JCFPostureSurrogate(ck)
        name = os.path.splitext(os.path.basename(ck))[0]
        idx, f_ref, _, _ = surrogate_fmedial(m, q_true, pos_true, rc_grf, lc_grf, mass, height)
        res = {"pelvis_translation_input": not m.drop_pelvis_translation,
               "mocap+grf_vs_opensim": err(f_ref, fm_gt[idx]), "mocap+grf_vs_opensim_stance": err(f_ref[stance[idx]], fm_gt[idx][stance[idx]])}
        curves.loc[idx, f"{name}|mocap+grf"] = f_ref
        if m.drop_pelvis_translation:
            for tag, (rc, lc) in {"imu+grf": (rc_grf, lc_grf), "imu+imu_contact": (rc_imu, lc_imu)}.items():
                _, f_imu, _, _ = surrogate_fmedial(m, q_imu, pos_imu, rc, lc, mass, height)
                curves.loc[idx, f"{name}|{tag}"] = f_imu
                res[f"{tag}_vs_mocap+grf"] = err(f_imu, f_ref)
                res[f"{tag}_vs_opensim"] = err(f_imu, fm_gt[idx])
                res[f"{tag}_vs_opensim_stance"] = err(f_imu[stance[idx]], fm_gt[idx][stance[idx]])
        else:
            res["note"] = "uses pelvis translation, which IMUs do not observe; scored on the mocap path only"
        model_scores[name] = res
    curves.to_csv(os.path.join(a.out, "fmedial.csv"), index=False)

    used = ang[ang.observed & ang.surrogate_input]
    summary = dict(trial=trial, synthetic=syn, angles_rmse_deg=dict(zip(used.coordinate, used.rmse.round(3))),
                   angles_rmse_deg_max=float(used.rmse.max()), contact=contact_scores, surrogate=model_scores)
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.fill_between(t_orig, -0.5, 3.0, where=rc_grf, color="#EEEEEE", lw=0, step="mid", label="right stance (force plate)")
    ax.plot(t_orig, fm_gt, color="#000000", lw=2, label="OpenSim")
    cols = [c for c in curves.columns if "|" in c]
    models = list(dict.fromkeys(c.split("|")[0] for c in cols))
    style = {"mocap+grf": "-", "imu+grf": "--", "imu+imu_contact": ":"}
    for col in cols:
        name, path = col.split("|")
        short = name.replace("best_model_mlp_", "").split("_qinst")[0] + ("" if len(models) == len(set(
            m.replace("best_model_mlp_", "").split("_qinst")[0] for m in models)) else f" #{models.index(name)}")
        ax.plot(curves.time_original, curves[col], lw=1.4, ls=style[path], color=f"C{models.index(name)}",
                label=f"{short}: {path}")
    ax.axhline(1.4, color="#888888", lw=1, ls="--")
    ax.set_xlabel("time in trial (s)"); ax.set_ylabel("F_medial (BW)")
    ax.set_title(f"{os.path.basename(trial)} · noise {a.noise_deg}° · drift {a.drift_deg_per_s}°/s", fontsize=10)
    ax.legend(fontsize=7, loc="upper right"); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(os.path.join(a.out, "fmedial.png"), dpi=110)

    print(f"joint angles, IMU vs mocap: max RMSE {summary['angles_rmse_deg_max']:.2f} deg over the 13 observed coordinates")
    for det, s in contact_scores.items():
        r = s["right"]
        print(f"contact {det:26s}: agreement right {r['frame_agreement']:.2f}, left {s['left']['frame_agreement']:.2f}; "
              f"right onset {r['onset_error_mean_s'] * 1000:+.0f} ms, offset {r['offset_error_mean_s'] * 1000:+.0f} ms")
    for name, r in model_scores.items():
        line = f"{name[:60]}: vs OpenSim, mocap+grf MAE {r['mocap+grf_vs_opensim']['mae']:.3f}"
        if "imu+grf_vs_opensim" in r:
            line += (f" | imu+grf {r['imu+grf_vs_opensim']['mae']:.3f} | imu+imu_contact {r['imu+imu_contact_vs_opensim']['mae']:.3f}"
                     f" | IMU path vs mocap path MAE {r['imu+grf_vs_mocap+grf']['mae']:.3f} (grf contact), "
                     f"{r['imu+imu_contact_vs_mocap+grf']['mae']:.3f} (IMU contact)")
        print(line)
    print("written to", a.out)
    return summary


if __name__ == "__main__":
    main()

"""Simulated-suit check of the PN3 wearable path with the corrected baseline (thesis paper, Sec. Wearable Path).

Every run is the deployed pipeline, called through its own command lines:

    python -m pn3_redis.driver --source sto <take>/orientations.sto --resample-hz 60 --contact gyro \
        --backdate-s {0.2,0} --height H --mass M --surrogate <ckpt> --log <csv> --quiet
    python -m pn3_redis.evaluate --log <csv> --synthetic <take> --trial <trial> --out <dir> \
        --label-subdir jcf_output_sojr --mx-convention physical

plus one force-plate-contact run per (take, checkpoint) (--contact source --contact-trial <trial>, which also anchors
the UKF on the force plates) that separates contact error from kinematic error. Takes come from imu/synthetic.py
(results/thesis/pn3_sim/takes; Carter P010 t10 with 1 deg noise is results/imu/driver_demo/take_P010_t10_noise1deg).

The surrogate's F_medial is then compared, on the frames it was published for (surrogate_time, original trial clock,
from calib_s + 0.5 s on, inside the label window), with:
  L       lab path: the same checkpoint on its own training features of the trial (tools/evaluate_fmedial.features_for
          with --deploy-features: train_mlp._build_com_features, scaled OpenSim model, exact time alignment, 100 Hz,
          force-plate contact), the trial's own height. What the model predicts from motion capture.
  O       optical, generic body: the trial's IK through the driver's height-scaled LowerBody model (the driver's
          height), checkpoint foot point, centred 20-frame windows, force-plate contact
          (analysis/thesis/twente_imu.reference_surrogate, the Twente O path). U - O is what the suit, the UKF and the
          online contact add; O - L what the generic body model and the entered height add.
  labels  SO+JR (jcf_output_sojr), physical sign, on the frames the label QC keeps (label_mask.npz keep_r carried to
          the frame times with train_mlp.LABEL_MASK_TOL), and, flagged as not a valid reference, on every frame.
F_medial = 0.5|F_y| + s*M_x*H/0.045 with s = -1 (physical: medial moment = -mx of walker_knee_r) for labels and every
raw output alike, H the height each path ran with. Right stance = force-plate |F_y| > 5 % BW.

Stance peaks: right-stance intervals of at least 0.2 s that lie wholly inside the scored span; against labels, only
stances whose frames are all QC-kept.

Usage (per-frame logs go to --work, outside the repo; metrics, tables and figures to --out):
    python analysis/thesis/pn3_sim.py --workers 20
    python analysis/thesis/pn3_sim.py --replot          # tables and figures from the saved runs.csv / series
"""
import argparse
import copy
import json
import os
import subprocess
import sys
import time
from multiprocessing import Pool
from types import SimpleNamespace

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np                                                        # noqa: E402
import pandas as pd                                                       # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

OUT_DEFAULT = os.path.join(REPO, "results", "thesis", "pn3_sim")
WORK_DEFAULT = "/tmp/claude-1000/-home-gellert-Developer-medical-dt/187d5d35-6c3c-4c5e-8d5a-f69463aa63f9/scratchpad/pn3sim/work"
TAKES_DIR = os.path.join(OUT_DEFAULT, "takes")
LABEL_SUBDIR = "jcf_output_sojr"
SIGN = -1.0                                   # physical: medial moment = -mx of walker_knee_r
PHI, D_M, CONTACT_BW = 0.5, 0.045, 0.05

CKPTS = {
    "qc_s1": "models/runs/mlp/best_model_mlp_walking_qinst_notrans_pelvisrel_winpos_contact_fpopt_w20_t100_sojrqc_fmW0p5_e100_s1.pt",
    "qc_s2": "models/runs/mlp/best_model_mlp_walking_qinst_notrans_pelvisrel_winpos_contact_fpopt_w20_t100_sojrqc_fmW0p5_e100_s2.pt",
    "qc_s3": "models/runs/mlp/best_model_mlp_walking_qinst_notrans_pelvisrel_winpos_contact_fpopt_w20_t100_sojrqc_fmW0p5_e100_s3.pt",
    "qc_origin": "models/runs/mlp/best_model_mlp_walking_qinst_notrans_pelvisrel_winpos_contact_w20_t100_sojrqc_fmW0p5_e100_s1.pt",
    "fpopt_legacy": "models/runs/mlp/best_model_mlp_walking_qinst_notrans_pelvisrel_winpos_contact_fpopt_w20_fmW0p5_e100.pt",
    "sept_legacy": "models/runs/mlp/best_model_mlp_walking_qinst_notrans_pelvisrel_winpos_contact_w20_fmW0p5_e100_2.pt",
}
QC = ["qc_s1", "qc_s2", "qc_s3"]
LEGACY = ["fpopt_legacy", "sept_legacy"]

# height / mass as entered in the driver. Carter P010: 1.7 m (the demo's entered height; the trial's is 1.656 m), and
# once more with the true height (config gyro_bd0.2_trueH) to show what the entered height costs.
TRIALS = {
    "carter_P010_t10": dict(trial="jcf/full_duration/testing/walking/carter_P010_split0_t10", height=1.7, mass=54.0),
    "moore_s10_t55r02": dict(trial="jcf/full_duration/testing/walking/moore_subject10_t55_r02", height=1.77, mass=90.7),
}
TAKES = ["clean", "noise1deg", "noise2deg", "noise1deg_drift0p5"]
TAKE_LABEL = {"clean": "clean", "noise1deg": "1° noise", "noise2deg": "2° noise", "noise1deg_drift0p5": "1° + 0.5°/s drift"}


def take_dir(trial_key, take):
    if (trial_key, take) == ("carter_P010_t10", "noise1deg"):
        return os.path.join(REPO, "results", "imu", "driver_demo", "take_P010_t10_noise1deg")
    return os.path.join(TAKES_DIR, f"{trial_key}_{take}")


def configs(trial_key):
    tr = TRIALS[trial_key]
    c = {"gyro_bd0.2": dict(args=["--contact", "gyro", "--backdate-s", "0.2"], height=tr["height"]),
         "gyro_bd0": dict(args=["--contact", "gyro", "--backdate-s", "0"], height=tr["height"]),
         "fp": dict(args=["--contact", "source", "--contact-trial", os.path.join(REPO, tr["trial"]), "--backdate-s", "0"],
                    height=tr["height"])}
    h_true = float(json.load(open(os.path.join(REPO, tr["trial"], "metadata.json")))["height_m"])
    if abs(h_true - tr["height"]) > 1e-6:
        c["gyro_bd0.2_trueH"] = dict(args=["--contact", "gyro", "--backdate-s", "0.2"], height=round(h_true, 3))
    return c


def fmed(fy, mx, height):
    return PHI * np.abs(np.asarray(fy, float)) + SIGN * np.asarray(mx, float) * height / D_M


# ─── References: lab path (L) and optical through the generic body (O), one per (trial, checkpoint, height) ──────

def ref_path(work, trial_key, ck, height):
    return os.path.join(work, "refs", f"{trial_key}__{ck}__h{height:.3f}.npz")


def build_refs(job):
    trial_key, ck, height, work = job
    import importlib.util
    import torch
    torch.set_num_threads(1)
    from train_mlp import INPUT_SETS, load_label_mask
    from imu.contact import grf_contact
    from imu.synthetic import read_mot
    from pn3_redis.model import COORDS, LowerBody
    from inference_posture_surrogate import JCFPostureSurrogate
    from analysis.thesis.twente_imu import reference_surrogate
    spec = importlib.util.spec_from_file_location("evaluate_fmedial", os.path.join(REPO, "tools", "evaluate_fmedial.py"))
    ef = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ef)
    tr = TRIALS[trial_key]
    d = os.path.join(REPO, tr["trial"])
    meta = json.load(open(os.path.join(d, "metadata.json")))
    h_trial, m_trial = float(meta["height_m"]), float(meta["mass_kg"])
    path = os.path.join(REPO, CKPTS[ck])
    # L: the checkpoint's own feature builder (deployment alignment: exact time, 100 Hz; legacy sets included)
    m, mu, sd, info = ef.load_model(path, "cpu")
    args = SimpleNamespace(max_peak=np.inf, max_moment=np.inf, pelvis="trial", deploy_features=True, label_mask=True,
                           label_mask_subdir=LABEL_SUBDIR)
    built = ef.features_for(d, dict(INPUT_SETS[info["input_set"]], label_subdir=LABEL_SUBDIR), args)
    if built is None:
        raise RuntimeError(f"{trial_key}: no lab features for {ck}")
    feat, labels, fm = built
    with torch.no_grad():
        pL = m((torch.from_numpy(feat).float() - mu) / sd).numpy()
    tL = np.asarray(fm["t"], float)
    # O: trial IK -> generic LowerBody at the driver's height -> checkpoint foot point -> force-plate contact
    ti, ni, vi = read_mot(os.path.join(d, "ik_results.mot"))
    tg = np.arange(ti[0], ti[-1] + 1e-9, 0.01)
    q = np.column_stack([np.interp(tg, ti, vi[:, ni.index(c)]) for c in COORDS])
    cr, cl = grf_contact(d, tg, m_trial)
    sur = JCFPostureSurrogate(path, device="cpu")
    k, out = reference_surrogate(sur, LowerBody(height_m=height), q, cr, cl, tr["mass"], height)
    t_mask, keep_jr = load_label_mask(d, dict(label_mask="label_mask.npz", label_mask_subdir=LABEL_SUBDIR))
    os.makedirs(os.path.dirname(ref_path(work, trial_key, ck, height)), exist_ok=True)
    np.savez(ref_path(work, trial_key, ck, height),
             tL=tL, fL=fmed(pL[:, 1], pL[:, 3], h_trial), fyL=pL[:, 1], mxL=pL[:, 3],
             fLab=fmed(labels[:, 1], labels[:, 3], h_trial), keepL=np.asarray(fm["label_keep"], bool),
             stL=np.asarray(fm["r_contact"], bool),
             tO=tg[k], fO=fmed(np.ravel(out["fy"]), np.ravel(out["mx"]), height),
             t_mask=t_mask, keep_jr=np.asarray(keep_jr, bool), h_trial=h_trial, m_trial=m_trial,
             foot_point=sur.foot_point, input_set=info["input_set"])
    return trial_key, ck, height


# ─── One run: driver -> evaluate -> comparison ────────────────────────────────────────────────────────────────────

def stance_runs(t, st, lo, hi, min_s=0.2):
    """(i0, i1) index ranges of contiguous right-stance frames lasting >= min_s and wholly inside (lo, hi)."""
    out, i, n = [], 0, len(st)
    while i < n:
        if st[i]:
            j = i
            while j + 1 < n and st[j + 1]:
                j += 1
            if i > 0 and j < n - 1 and t[j] - t[i] >= min_s and t[i] > lo and t[j] < hi:
                out.append((i, j + 1))
            i = j + 1
        else:
            i += 1
    return out


def pair(a, b, m):
    m = np.asarray(m, bool) & np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return dict(n=int(m.sum()), mae=np.nan, bias=np.nan, rmse=np.nan, r=np.nan)
    e = a[m] - b[m]
    r = float(np.corrcoef(a[m], b[m])[0, 1]) if np.std(a[m]) > 0 and np.std(b[m]) > 0 else np.nan
    return dict(n=int(m.sum()), mae=float(np.mean(np.abs(e))), bias=float(np.mean(e)),
                rmse=float(np.sqrt(np.mean(e ** 2))), r=r)


def peaks(a, b, runs, ok_run=None, min_kept=0.5):
    """Mean and mean-absolute stance-peak error a - b. With ok_run (label comparisons), the peaks are taken over the
    QC-kept frames of each stance, for stances with at least min_kept of their frames kept."""
    e = []
    for i0, i1 in runs:
        sel = np.ones(i1 - i0, bool) if ok_run is None else ok_run[i0:i1]
        if sel.mean() < (min_kept if ok_run is not None else 0) or not sel.any():
            continue
        e.append(float(a[i0:i1][sel].max() - b[i0:i1][sel].max()))
    return dict(n_stances=len(e), peak_err=float(np.mean(e)) if e else np.nan,
                peak_mae=float(np.mean(np.abs(e))) if e else np.nan)


def run_job(job):
    from imu.contact import grf_contact
    from pn3_redis.evaluate import label_file, label_fmedial
    from imu.synthetic import read_mot
    from train_mlp import carry_label_mask, LABEL_MASK_TOL
    trial_key, take, ck, cfg, work = job
    tr = TRIALS[trial_key]
    c = configs(trial_key)[cfg]
    tdir = take_dir(trial_key, take)
    name = f"{trial_key}__{take}__{ck}__{cfg}"
    rdir = os.path.join(work, "runs", name)
    os.makedirs(rdir, exist_ok=True)
    log = os.path.join(rdir, "log.csv")
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    drv = [sys.executable, "-m", "pn3_redis.driver", "--source", "sto", os.path.join(tdir, "orientations.sto"),
           "--resample-hz", "60", *c["args"], "--height", f"{c['height']:g}", "--mass", f"{tr['mass']:g}",
           "--surrogate", os.path.join(REPO, CKPTS[ck]), "--log", log, "--quiet"]
    t0 = time.time()
    p = subprocess.run(drv, cwd=REPO, env=env, capture_output=True, text=True)
    t_drv = time.time() - t0
    if p.returncode:
        raise RuntimeError(f"{name}: driver failed\n{p.stderr[-2000:]}")
    ev = [sys.executable, "-m", "pn3_redis.evaluate", "--log", log, "--synthetic", tdir,
          "--trial", os.path.join(REPO, tr["trial"]), "--out", rdir, "--label-subdir", LABEL_SUBDIR,
          "--mx-convention", "physical", "--title", f"{trial_key} {TAKE_LABEL[take]} · {ck} · {cfg}"]
    p = subprocess.run(ev, cwd=REPO, env=env, capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError(f"{name}: evaluate failed\n{p.stderr[-2000:]}")
    M = json.load(open(os.path.join(rdir, "metrics.json")))

    R = np.load(ref_path(work, trial_key, ck, c["height"]))
    L = pd.read_csv(log)
    syn = json.load(open(os.path.join(tdir, "synthetic.json")))
    t = L.t.to_numpy()
    walk = t >= syn["calib_s"] + 0.5
    # joint angles, every rotational coordinate (pn3_redis.evaluate's angle_rmse_deg leaves out pelvis_tilt: its
    # filter c.startswith("pelvis_t") meant for the translations also matches it); same frames and truth as there
    from pn3_redis.model import COORDS, TRANSLATIONS
    tt, tn, tv = read_mot(os.path.join(tdir, "truth_ik.mot"))
    rot = [c_ for c_ in COORDS if c_ not in TRANSLATIONS]
    ang = {c_: float(np.degrees(np.sqrt(np.mean((L[c_].to_numpy()[walk] - np.interp(t[walk], tt, tv[:, tn.index(c_)])) ** 2))))
           for c_ in rot}
    surr13 = [c_ for c_ in COORDS[:16] if c_ not in TRANSLATIONS]
    # pelvis progression and side-to-side error along the walking direction (pn3_redis.evaluate assumes +x; the Moore
    # subject faces +-z, pelvis_rotation ~93 deg): direction = the stance heel's ground-plane velocity (the belt)
    from pn3_redis.model import LowerBody
    d = os.path.join(REPO, tr["trial"])
    truth = np.column_stack([np.interp(t, tt, tv[:, tn.index(c_)]) for c_ in COORDS])
    rc_t, lc_t = grf_contact(d, t + syn["time_offset"], float(R["m_trial"]))
    heel = LowerBody(height_m=float(R["h_trial"])).fk(truth)["calcn_r"][1][:, [0, 2]]
    vh = np.gradient(heel, t, axis=0)
    belt_vec = -np.median(vh[walk & rc_t], axis=0)
    fwd = belt_vec / np.linalg.norm(belt_vec)
    lat = np.array([-fwd[1], fwd[0]])
    est_xz = L[["pelvis_x", "pelvis_z"]].to_numpy()
    tru_xz = truth[:, [COORDS.index("pelvis_tx"), COORDS.index("pelvis_tz")]]
    v_fwd = float(np.polyfit(t[walk], est_xz[walk] @ fwd, 1)[0])
    e_lat = (est_xz[walk] - tru_xz[walk]) @ lat
    lat_rmse = float(np.sqrt(np.mean((e_lat - e_lat.mean()) ** 2)))
    walk_heading_deg = float(np.degrees(np.arctan2(-fwd[1], fwd[0])))   # about +y, OpenSim convention
    # heel-strike / toe-off timing of the flags the surrogate used vs the force plates (s, used - force plate)
    ev = {}
    if "surrogate_contact_r" in L and L.surrogate_contact_r.notna().any():
        u = L.surrogate_contact_r.notna().to_numpy() & walk
        tu = L.surrogate_time.to_numpy()[u] + syn["time_offset"]
        for s_, ref in zip("rl", grf_contact(d, tu, float(R["m_trial"]))):
            fl = L[f"surrogate_contact_{s_}"].to_numpy()[u] > 0.5
            for kind, sgn in (("on", 1), ("off", -1)):
                er = np.flatnonzero(np.diff(ref.astype(int)) == sgn) + 1
                eu = np.flatnonzero(np.diff(fl.astype(int)) == sgn) + 1
                errs = [tu[eu[np.argmin(np.abs(tu[eu] - tu[i]))]] - tu[i] for i in er if len(eu)
                        and np.min(np.abs(tu[eu] - tu[i])) <= 0.25]
                ev[f"{kind}_err_med_s_{s_}"] = float(np.median(errs)) if errs else np.nan
                ev[f"{kind}_err_mad_s_{s_}"] = float(np.median(np.abs(errs))) if errs else np.nan
                ev[f"{kind}_n_{s_}"] = len(errs)
                ev[f"{kind}_n_ref_{s_}"] = len(er)
            ev[f"false_contact_in_swing_{s_}"] = float(np.mean(fl[~ref])) if (~ref).any() else np.nan
            ev[f"missed_contact_in_stance_{s_}"] = float(np.mean(~fl[ref])) if ref.any() else np.nan
    ok = L.f_medial.notna().to_numpy() & walk
    ts = L.surrogate_time.to_numpy()[ok] + syn["time_offset"]
    fU = fmed(L.fy.to_numpy()[ok], L.mx.to_numpy()[ok], c["height"])
    tL, tO = R["tL"], R["tO"]
    lo, hi = max(tL[0], tO[0]), min(tL[-1], tO[-1])
    v = (ts >= lo) & (ts <= hi)
    ts, fU = ts[v], fU[v]
    fL, fO = np.interp(ts, tL, R["fL"]), np.interp(ts, tO, R["fO"])
    jt, jn, jv = read_mot(label_file(d, LABEL_SUBDIR), angles=False)
    BW = float(R["m_trial"]) * 9.81
    fLab = label_fmedial(jt, jn, jv, ts, BW, SIGN)
    keep = carry_label_mask(R["t_mask"], R["keep_jr"], ts, tol=LABEL_MASK_TOL)
    st = grf_contact(d, ts, float(R["m_trial"]))[0]
    runs = stance_runs(ts, st, ts[0], ts[-1])
    allf = np.ones(len(ts), bool)
    row = dict(trial=trial_key, take=take, ckpt=ck, config=cfg, height=c["height"], mass=tr["mass"],
               n_frames=int(len(ts)), n_stance=int(st.sum()), n_kept=int(keep.sum()), n_kept_stance=int((keep & st).sum()),
               t_scored_s=float(ts[-1] - ts[0]), driver_wall_s=t_drv, driver_frames=int(len(t)),
               angle_rmse_max_deg=max(ang[c_] for c_ in surr13),
               angle_rmse_max_coord=max(surr13, key=lambda c_: ang[c_]),
               angle_rmse_max_deg_evaluate=M["angle_rmse_deg_max"],
               angle_rmse_max_all16_deg=max(ang.values()),
               angle_rmse_median_deg=float(np.median([ang[c_] for c_ in surr13])),
               knee_r_rmse_deg=ang["knee_angle_r"], knee_l_rmse_deg=ang["knee_angle_l"],
               hip_flex_r_rmse_deg=ang["hip_flexion_r"], ankle_r_rmse_deg=ang["ankle_angle_r"],
               pelvis_tilt_rmse_deg=ang["pelvis_tilt"], pelvis_list_rmse_deg=ang["pelvis_list"],
               pelvis_rotation_rmse_deg=ang["pelvis_rotation"], lumbar_ext_rmse_deg=ang["lumbar_extension"],
               **{f"rmse_deg__{c_}": ang[c_] for c_ in rot},
               pelvis_height_rmse_cm=100 * M["pelvis_height_rmse_m"], pelvis_lateral_rmse_cm=100 * lat_rmse,
               forward_speed_mps=v_fwd, belt_speed_mps=float(np.linalg.norm(belt_vec)), walk_heading_deg=walk_heading_deg,
               pelvis_lateral_rmse_cm_evaluate=100 * M["pelvis_lateral_rmse_m"],
               forward_speed_mps_evaluate=M["forward_speed_mps"], belt_speed_mps_evaluate=M["belt_speed_mps"], **ev,
               contact_online_r=M["contact_agreement"]["r"], contact_online_l=M["contact_agreement"]["l"],
               contact_used_r=(M.get("contact_agreement_surrogate") or {}).get("r", np.nan),
               contact_used_l=(M.get("contact_agreement_surrogate") or {}).get("l", np.nan),
               latency_s=M["f_medial"].get("latency_s"), logged_mx_sign=M["f_medial"].get("logged_mx_sign"),
               logged_height_m=M["f_medial"].get("logged_height_m"),
               eval_mae_alllabels=M["f_medial"].get("mae"), eval_mae_stance_alllabels=M["f_medial"].get("mae_stance"),
               eval_r_alllabels=M["f_medial"].get("r"))
    comps = {"U_L": (fU, fL), "U_O": (fU, fO), "O_L": (fO, fL), "U_lab": (fU, fLab), "L_lab": (fL, fLab),
             "O_lab": (fO, fLab)}
    for key, (a, b) in comps.items():
        masks = {"all": allf, "stance": st}
        if key.endswith("lab"):
            masks = {"kept": keep, "kept_stance": keep & st, "unmasked_all": allf, "unmasked_stance": st}
        for mk, mm in masks.items():
            for k_, v_ in pair(a, b, mm).items():
                row[f"{key}_{mk}_{k_}"] = v_
        pk = peaks(a, b, runs, keep if key.endswith("lab") else None)
        for k_, v_ in pk.items():
            row[f"{key}_{k_}"] = v_
    row["mean_fU_stance"] = float(np.mean(fU[st])) if st.any() else np.nan
    row["mean_fL_stance"] = float(np.mean(fL[st])) if st.any() else np.nan
    row["mean_fLab_kept_stance"] = float(np.mean(fLab[keep & st])) if (keep & st).any() else np.nan
    cu = L.surrogate_contact_r.to_numpy()[ok][v] if "surrogate_contact_r" in L else np.full(len(ts), np.nan)
    np.savez(os.path.join(rdir, "series.npz"), t=ts, fU=fU, fL=fL, fO=fO, fLab=fLab, keep=keep, stance=st,
             contact_used_r=cu)
    return row


# ─── Trial-level label coverage ───────────────────────────────────────────────────────────────────────────────────

def coverage_rows():
    rows = []
    for key, tr in TRIALS.items():
        d = os.path.join(REPO, tr["trial"])
        with np.load(os.path.join(d, LABEL_SUBDIR, "label_mask.npz")) as z:
            st, k = z["stance_r"], z["keep_r"]
            t = z["time"]
            row = dict(trial=key, folder=tr["trial"], jr_rows=int(len(k)), jr_span_s=float(t[-1] - t[0]),
                       jr_rate_hz=float((len(t) - 1) / (t[-1] - t[0])), kept_frac=float(k.mean()),
                       stance_rows=int(st.sum()), kept_stance_frac=float((k & st).sum() / max(st.sum(), 1)),
                       fixed_cop_r=bool(z["fixed_cop_r"]), cop_speed_r_mps=float(z["cop_speed_r"]))
            for r_ in [str(x) for x in z["rules"]] + ["so_any", "fy5"]:
                if f"{r_}_r" in z.files:
                    row[f"flag_{r_}_frac"] = float(z[f"{r_}_r"].mean())
                    row[f"flag_{r_}_stance_frac"] = float((z[f"{r_}_r"] & st).sum() / max(st.sum(), 1))
            row["thresholds"] = str(z["thresholds"])
        meta = json.load(open(os.path.join(d, "metadata.json")))
        row.update(mass_kg=meta["mass_kg"], height_m=meta.get("height_m"), driver_height_m=tr["height"],
                   driver_mass_kg=tr["mass"])
        rows.append(row)
    return pd.DataFrame(rows)


# ─── Tables, figures, README ──────────────────────────────────────────────────────────────────────────────────────

def summarise(df, out):
    # what the suit adds to the model's own error against the QC-kept labels: U vs labels minus L vs labels, on the
    # same kept right-stance frames (NaN where no frame is kept, e.g. Carter P010)
    df = df.copy()
    df["suit_added_kept_stance_mae"] = df["U_lab_kept_stance_mae"] - df["L_lab_kept_stance_mae"]
    df.to_csv(os.path.join(out, "runs.csv"), index=False)
    cols = ["angle_rmse_max_deg", "angle_rmse_median_deg", "knee_r_rmse_deg", "pelvis_height_rmse_cm",
            "pelvis_lateral_rmse_cm", "forward_speed_mps", "belt_speed_mps", "contact_online_r", "contact_online_l",
            "contact_used_r", "contact_used_l", "U_L_all_mae", "U_L_stance_mae", "U_L_stance_bias", "U_L_all_r",
            "U_L_peak_err", "U_L_peak_mae", "U_O_stance_mae", "U_O_all_r", "O_L_stance_mae", "O_L_stance_bias",
            "U_lab_kept_mae", "U_lab_kept_stance_mae", "U_lab_kept_stance_bias", "U_lab_kept_r", "U_lab_peak_err",
            "U_lab_n_stances", "L_lab_kept_mae", "L_lab_kept_stance_mae", "L_lab_kept_stance_bias", "L_lab_kept_r",
            "L_lab_peak_err", "suit_added_kept_stance_mae", "U_lab_unmasked_stance_mae", "L_lab_unmasked_stance_mae",
            "n_frames", "n_stance",
            "n_kept", "n_kept_stance", "latency_s"]
    # seed spread of the corrected baseline (optimizer foot point): mean, min, max over s1..s3
    q = df[df.ckpt.isin(QC)]
    g = q.groupby(["trial", "take", "config"])[cols]
    spread = pd.concat({"mean": g.mean(), "min": g.min(), "max": g.max()}, axis=1)
    spread.columns = [f"{c}__{s}" for s, c in spread.columns]
    spread.reset_index().to_csv(os.path.join(out, "seed_spread.csv"), index=False)
    head = df[df.config.isin(["gyro_bd0.2", "gyro_bd0", "fp"])][["trial", "take", "ckpt", "config"] + cols]
    head.sort_values(["trial", "take", "ckpt", "config"]).to_csv(os.path.join(out, "table_all.csv"), index=False)
    # back-dating before/after, per trial x take x checkpoint
    bd = []
    for (tk, tk2, ck), gr in df.groupby(["trial", "take", "ckpt"]):
        a = gr[gr.config == "gyro_bd0"]
        b = gr[gr.config == "gyro_bd0.2"]
        f = gr[gr.config == "fp"]
        if a.empty or b.empty:
            continue
        a, b = a.iloc[0], b.iloc[0]
        r_ = dict(trial=tk, take=tk2, ckpt=ck)
        for c in ["contact_used_r", "contact_used_l", "U_L_stance_mae", "U_L_all_mae", "U_L_peak_err",
                  "U_lab_kept_stance_mae", "U_lab_kept_mae", "U_lab_peak_err", "L_lab_kept_stance_mae",
                  "suit_added_kept_stance_mae"]:
            r_[f"{c}__bd0"], r_[f"{c}__bd0.2"] = a[c], b[c]
            r_[f"{c}__fp"] = f.iloc[0][c] if not f.empty else np.nan
        bd.append(r_)
    pd.DataFrame(bd).to_csv(os.path.join(out, "backdating.csv"), index=False)
    # headline: per trial x take x config, seed 1 and "mean [min, max]" over seeds 1-3, for the numbers the paper uses
    hcols = ["angle_rmse_max_deg", "pelvis_height_rmse_cm", "pelvis_lateral_rmse_cm", "forward_speed_mps",
             "belt_speed_mps", "contact_used_r", "contact_used_l", "U_L_all_mae", "U_L_stance_mae", "U_L_stance_bias",
             "U_L_all_r", "U_L_peak_err", "U_O_stance_mae", "O_L_stance_mae", "U_lab_kept_mae", "U_lab_kept_stance_mae",
             "U_lab_kept_stance_bias", "U_lab_kept_r", "U_lab_peak_err", "L_lab_kept_mae", "L_lab_kept_stance_mae",
             "L_lab_kept_stance_bias", "L_lab_kept_r", "L_lab_peak_err", "suit_added_kept_stance_mae", "U_lab_n_stances"]
    hrows = []
    for (tk, tk2, cfg), gr in q.groupby(["trial", "take", "config"]):
        r_ = dict(trial=tk, take=tk2, config=cfg)
        s1 = gr[gr.ckpt == "qc_s1"]
        for c in hcols:
            v_ = gr[c].to_numpy(float)
            r_[f"{c}__s1"] = float(s1[c].iloc[0]) if len(s1) else np.nan
            r_[f"{c}__seeds"] = (f"{np.mean(v_):.3f} [{np.min(v_):.3f}, {np.max(v_):.3f}]"
                                 if np.isfinite(v_).all() else "n/a")
        hrows.append(r_)
    pd.DataFrame(hrows).to_csv(os.path.join(out, "headline.csv"), index=False)
    # per-seed values of the F_medial numbers the text quotes, so no headline rests on one seed
    pcols = ["U_L_stance_mae", "U_lab_kept_stance_mae", "L_lab_kept_stance_mae", "suit_added_kept_stance_mae"]
    pr = []
    for (tk, tk2, cfg), gr in q.groupby(["trial", "take", "config"]):
        for c in pcols:
            v_ = {ck: float(gr.loc[gr.ckpt == ck, c].iloc[0]) for ck in QC if (gr.ckpt == ck).any()}
            a_ = np.array(list(v_.values()), float)
            pr.append(dict(trial=tk, take=tk2, config=cfg, metric=c, **v_,
                           mean=np.mean(a_) if np.isfinite(a_).all() else np.nan,
                           min=np.min(a_) if np.isfinite(a_).all() else np.nan,
                           max=np.max(a_) if np.isfinite(a_).all() else np.nan))
    pd.DataFrame(pr).to_csv(os.path.join(out, "seed_detail.csv"), index=False, float_format="%.6f")
    return spread


def fig_series(df, work, out, ck="qc_s1", take="noise1deg"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK, MUTED, GRID = "#0b0b0b", "#8a8984", "#e6e5e0"
    S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": MUTED, "xtick.color": "#52514e", "ytick.color": "#52514e"})
    fig, axs = plt.subplots(2, 1, figsize=(10, 6.4))
    for ax, tk in zip(axs, TRIALS):
        z = {c: np.load(os.path.join(work, "runs", f"{tk}__{take}__{ck}__{c}", "series.npz")) for c in ("gyro_bd0.2", "gyro_bd0")}
        s = z["gyro_bd0.2"]
        t = s["t"]
        m = t <= t[0] + 4.0
        lab = np.where(s["keep"], s["fLab"], np.nan)
        lab_masked = np.where(~s["keep"], s["fLab"], np.nan)
        ax.fill_between(t[m], 0, 1, where=s["stance"][m], transform=ax.get_xaxis_transform(), color="#f1f0ec",
                        lw=0, step="mid")
        ax.plot(t[m], np.clip(lab_masked[m], None, 8), color=MUTED, lw=1, ls=":", label="SO+JR label, QC-masked")
        ax.plot(t[m], lab[m], color=INK, lw=2, label="SO+JR label, QC-kept")
        ax.plot(t[m], s["fL"][m], color=S1, lw=2, label="lab path (mocap features)")
        ax.plot(t[m], s["fU"][m], color=S2, lw=2, label="suit, back-dated 0.2 s")
        zz = z["gyro_bd0"]
        mm = zz["t"] <= t[0] + 4.0
        ax.plot(zz["t"][mm], zz["fU"][mm], color=S3, lw=1.5, ls="--", label="suit, no back-dating")
        r = df[(df.trial == tk) & (df["take"] == take) & (df.ckpt == ck) & (df.config == "gyro_bd0.2")].iloc[0]
        r0 = df[(df.trial == tk) & (df["take"] == take) & (df.ckpt == ck) & (df.config == "gyro_bd0")].iloc[0]
        cov = r.n_kept / r.n_frames
        ttl = (f"{tk}: suit vs lab stance MAE {r.U_L_stance_mae:.2f} BW (no back-dating {r0.U_L_stance_mae:.2f}); "
               f"labels kept on {100 * cov:.0f} % of frames")
        ax.set_title(ttl, fontsize=10, loc="left", color=INK)
        ax.set_ylabel("F_medial (BW)")
        top = np.nanmax(np.r_[s["fL"][m], s["fU"][m], lab[m] if np.isfinite(lab[m]).any() else 0]) * 1.25
        ax.set_ylim(-0.3, max(top, 2.5))
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
    axs[1].set_xlabel("trial time (s); shaded = force-plate right stance")
    axs[0].legend(fontsize=8, loc="upper right", ncol=3, frameon=False)
    fig.suptitle(f"Simulated suit (1° noise, 60 Hz) · {ck} · physical M_x sign", fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"fig_series_{ck}_{take}.png"), dpi=160)
    plt.close(fig)


def fig_noise(df, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK, MUTED, GRID = "#0b0b0b", "#8a8984", "#e6e5e0"
    S1, S2 = "#2a78d6", "#eb6834"
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": MUTED, "xtick.color": "#52514e", "ytick.color": "#52514e"})
    metrics = [("angle_rmse_max_deg", "worst joint-angle RMSE (deg)"), ("U_L_stance_mae", "F_medial suit vs lab,\nstance MAE (BW)"),
               ("U_lab_kept_stance_mae", "F_medial suit vs kept labels,\nstance MAE (BW)")]
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.8))
    x = np.arange(len(TAKES))
    for ax, (col, lab) in zip(axs, metrics):
        for j, (tk, colr) in enumerate(zip(TRIALS, (S1, S2))):
            q = df[(df.trial == tk) & df.ckpt.isin(QC) & (df.config == "gyro_bd0.2")]
            if col.startswith("U_lab") and q[col].isna().all():
                continue
            for i, tk2 in enumerate(TAKES):
                v = q[q["take"] == tk2][col].to_numpy()
                xx = x[i] + (j - 0.5) * 0.22
                ax.plot([xx] * len(v), v, "o", ms=6, color=colr, mec="white", mew=1.2, alpha=0.9,
                        label=tk if i == 0 else None)
                s1 = q[(q["take"] == tk2) & (q.ckpt == "qc_s1")][col]
                if len(s1):
                    ax.plot([xx - 0.08, xx + 0.08], [s1.iloc[0]] * 2, color=INK, lw=1.5)
            if col.startswith("U_lab"):
                ref = df[(df.trial == tk) & df.ckpt.isin(QC) & (df.config == "gyro_bd0.2")]["L_lab_kept_stance_mae"]
                ax.axhspan(ref.min(), ref.max(), color=colr, alpha=0.12, lw=0)
                ax.text(x[-1] + 0.35, ref.mean(), "lab path\nvs labels", fontsize=8, color="#52514e", va="center")
        ax.set_xticks(x, [TAKE_LABEL[t].replace(" + ", "\n+ ") for t in TAKES], fontsize=8)
        ax.set_title(lab, fontsize=10, loc="left", color=INK)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.set_ylim(bottom=0)
        ax.set_xlim(-0.5, len(TAKES) - 0.1 + (0.6 if col.startswith("U_lab") else 0))
    axs[0].legend(fontsize=8, frameon=False, loc="upper left")
    fig.suptitle("Corrected baseline, seeds 1-3 (dots; bar = seed 1), gyro contact back-dated 0.2 s", fontsize=11,
                 x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_noise_sweep.png"), dpi=160)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--work", default=WORK_DEFAULT)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--replot", action="store_true")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    if not a.replot:
        jobs = []
        for tk in TRIALS:
            for take in TAKES:
                for ck in CKPTS:
                    if ck in LEGACY and take != "noise1deg":
                        continue
                    for cfg in configs(tk):
                        if cfg.endswith("trueH") and ck not in QC:
                            continue
                        jobs.append((tk, take, ck, cfg, a.work))
        refs = sorted({(tk, ck, configs(tk)[cfg]["height"], a.work) for tk, _t, ck, cfg, _w in jobs})
        t0 = time.time()
        with Pool(min(a.workers, len(refs))) as p:
            for r in p.imap_unordered(build_refs, refs):
                print("refs", r, f"{time.time() - t0:.0f} s", flush=True)
        rows = []
        with Pool(min(a.workers, len(jobs))) as p:
            for i, r in enumerate(p.imap_unordered(run_job, jobs)):
                rows.append(r)
                if (i + 1) % 10 == 0:
                    print(f"{i + 1}/{len(jobs)} runs, {time.time() - t0:.0f} s", flush=True)
        df = pd.DataFrame(rows).sort_values(["trial", "take", "ckpt", "config"])
        cov = coverage_rows()
        cov.to_csv(os.path.join(a.out, "label_coverage.csv"), index=False)
        json.dump(dict(date=time.strftime("%Y-%m-%d %H:%M"), n_runs=len(df), wall_s=time.time() - t0,
                       ckpts=CKPTS, trials=TRIALS, takes={f"{tk}_{t}": os.path.relpath(take_dir(tk, t), REPO)
                                                         for tk in TRIALS for t in TAKES},
                       label_subdir=LABEL_SUBDIR, mx_sign=SIGN, work=a.work),
                  open(os.path.join(a.out, "run_info.json"), "w"), indent=1)
    else:
        df = pd.read_csv(os.path.join(a.out, "runs.csv"))
    summarise(df, a.out)
    fig_series(df, a.work, a.out)
    fig_noise(df, a.out)
    print("written to", a.out)


if __name__ == "__main__":
    main()

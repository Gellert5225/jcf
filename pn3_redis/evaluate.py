"""Score a driver run on virtual-suit data against the trial's ground truth.

    python -m pn3_redis.evaluate --log run.csv --synthetic results/imu/synth --trial jcf/.../carter_P010_split0_t10 \
        --out results/imu/driver_eval

The run must replay an orientation file made by imu/synthetic.py from --trial (driver --source sto ... --log run.csv).
Reports joint-angle RMSE against the mocap IK, pelvis height and side-to-side error, forward speed against the
treadmill belt (the planted-foot anchoring turns treadmill walking into overground-equivalent progress), contact
agreement with the force plates, and the live surrogate's F_medial against OpenSim's labels.
Writes metrics.json and driver_eval.png in --out.

F_medial = 0.5*|F_y| + s*M_x*H/0.045 (BW) with one sign s for the labels and the surrogate (--mx-convention):
physical (default) takes the medial moment as -mx of walker_knee_r (analysis/thesis/knee_mx_sign.py), pipeline as +mx
(the pre-Nov-2026 numbers). The surrogate's side is recomputed from its logged raw fy and mx; the logged f_medial is
what the checkpoint published (inference_posture_surrogate: the checkpoint's mx_medial_sign, +1 for old checkpoints),
and the sign and height it implies are reported as f_medial.logged_mx_sign / logged_height_m. Labels come from
<trial>/<--label-subdir>/ (jcf_output = stored, jcf_output_sojr = SO+JR).
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

from imu.contact import grf_contact                    # noqa: E402
from imu.synthetic import read_mot                     # noqa: E402
from pn3_redis.model import COORDS, LowerBody          # noqa: E402

PRE = "walker_knee_r_on_tibia_r_in_tibia_r_"
MX_SIGN = {"physical": -1.0, "pipeline": 1.0}   # factor on the raw walker_knee_r mx (= train_mlp.MX_CONVENTIONS)
PHI, D_M = 0.5, 0.045


def fmedial_bw(fy_bw, mx_bwh, height_m, sign):
    """F_medial (BW) from F_y (BW) and the raw walker_knee_r mx (BW*H) with medial sign `sign`."""
    return PHI * np.abs(fy_bw) + sign * np.asarray(mx_bwh, float) * height_m / D_M


def label_fmedial(jt, jn, jv, t, bw, sign):
    """F_medial of the right-knee JR labels (read_mot table, N and N*m) at times t, BW."""
    fy = np.interp(t, jt, jv[:, jn.index(PRE + "fy")]) / bw
    mx = np.interp(t, jt, jv[:, jn.index(PRE + "mx")]) / bw          # N*m / BW = (mx / (BW*H)) * H
    return fmedial_bw(fy, mx, 1.0, sign)


def logged_sign_and_height(f, fy, mx):
    """(sign, height) the logged f_medial implies: f - 0.5|fy| = sign*H/0.045 * mx, least squares (NaN, NaN when the
    log has too little moment to tell)."""
    f, fy, mx = (np.asarray(v, float) for v in (f, fy, mx))
    y = f - PHI * np.abs(fy)
    ok = np.isfinite(y) & np.isfinite(mx) & (np.abs(mx) > 1e-4)
    if ok.sum() < 5:
        return np.nan, np.nan
    k = float(np.sum(mx[ok] * y[ok]) / np.sum(mx[ok] ** 2)) * D_M
    return float(np.sign(k)), abs(k)


def label_file(trial, subdir):
    """<trial>/<subdir>/BatchJCF_JointReaction_ReactionLoads.sto, or the FullDuration_ name when only that exists."""
    for name in ("BatchJCF_JointReaction_ReactionLoads.sto", "FullDuration_JointReaction_ReactionLoads.sto"):
        p = os.path.join(trial, subdir, name)
        if os.path.exists(p):
            return p
    raise SystemExit(f"no JointReaction labels in {os.path.join(trial, subdir)}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", required=True)
    ap.add_argument("--synthetic", required=True, help="folder with orientations.sto, truth_ik.mot, synthetic.json")
    ap.add_argument("--trial", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default=None)
    ap.add_argument("--label-subdir", default="jcf_output",
                    help="label folder inside --trial: jcf_output (stored) or jcf_output_sojr (SO+JR)")
    ap.add_argument("--mx-convention", choices=sorted(MX_SIGN), default="physical",
                    help="medial moment for labels and surrogate alike: physical = -mx (default), pipeline = +mx")
    a = ap.parse_args(argv)
    sign = MX_SIGN[a.mx_convention]
    os.makedirs(a.out, exist_ok=True)
    L = pd.read_csv(a.log)
    syn = json.load(open(os.path.join(a.synthetic, "synthetic.json")))
    meta = json.load(open(os.path.join(a.trial, "metadata.json")))
    mass, H = float(meta["mass_kg"]), float(meta.get("height_m", 1.7))
    t = L.t.to_numpy()
    walk = t >= syn["calib_s"] + 0.5                       # skip the first half second after calibration
    tt, names, vt = read_mot(os.path.join(a.synthetic, "truth_ik.mot"))
    truth = np.column_stack([np.interp(t, tt, vt[:, names.index(c)]) for c in COORDS])
    est = L[COORDS].to_numpy()
    deg = {c: float(np.degrees(np.sqrt(np.mean((est[walk, i] - truth[walk, i]) ** 2))))
           for i, c in enumerate(COORDS) if not c.startswith("pelvis_t")}

    # pelvis: height and side-to-side after removing the mean; forward speed against the belt
    def rms_dm(e):
        return float(np.sqrt(np.mean((e - e.mean()) ** 2)))
    py = rms_dm(L.pelvis_y.to_numpy()[walk] - truth[walk, COORDS.index("pelvis_ty")])
    pz = rms_dm(L.pelvis_z.to_numpy()[walk] - truth[walk, COORDS.index("pelvis_tz")])
    t_orig = t + syn["time_offset"]
    rc, lc = grf_contact(a.trial, t_orig, mass)
    lb = LowerBody(height_m=H)
    heel = lb.fk(truth)["calcn_r"][1][:, 0]
    vheel = np.gradient(heel, t)
    belt = float(-np.median(vheel[walk & rc]))             # stance heel moves backwards at belt speed
    v_est = float(np.polyfit(t[walk], L.pelvis_x.to_numpy()[walk], 1)[0])
    agree = {s: float(np.mean(L[f"contact_{s}"].to_numpy()[walk] == ref[walk])) for s, ref in (("r", rc), ("l", lc))}
    # flags the surrogate saw at each query frame (published flags, with heel strikes back-dated), on the frame grid
    used, agree_used = None, {}
    if "surrogate_contact_r" in L and L.surrogate_contact_r.notna().any():
        u = L.surrogate_contact_r.notna().to_numpy()
        tu = L.surrogate_time.to_numpy()[u]
        inside = (t >= tu[0]) & (t <= tu[-1])
        used = {s: inside & (np.interp(t, tu, L[f"surrogate_contact_{s}"].to_numpy()[u]) > 0.5) for s in ("r", "l")}
        agree_used = {s: float(np.mean(used[s][walk & inside] == ref[walk & inside]))
                      for s, ref in (("r", rc), ("l", lc))}

    # live F_medial vs OpenSim labels, at the time each prediction refers to
    fm = {}
    pred_all = None
    if L.f_medial.notna().any():
        jt, jn, jv = read_mot(label_file(a.trial, a.label_subdir), angles=False)
        BW = mass * 9.81
        ok = L.f_medial.notna().to_numpy() & walk
        ts = L.surrogate_time.to_numpy()[ok] + syn["time_offset"]
        gt = label_fmedial(jt, jn, jv, ts, BW, sign)
        # the surrogate's F_medial under the same convention, from its raw outputs (height as the driver used it)
        f_log = L.f_medial.to_numpy()
        have_raw = "fy" in L and "mx" in L and L.fy.notna().any() and L.mx.notna().any()
        s_log, h_log = logged_sign_and_height(f_log, L.fy, L.mx) if have_raw else (np.nan, np.nan)
        if have_raw and np.isfinite(h_log):
            pred_all = fmedial_bw(L.fy.to_numpy(), L.mx.to_numpy(), h_log, sign)
        else:
            print("WARNING: no raw fy/mx in the log; scoring the logged f_medial (checkpoint convention)")
            pred_all = f_log
        pr = pred_all[ok]
        st = np.interp(ts - syn["time_offset"], t, rc.astype(float)) > 0.5
        fm = dict(mae=float(np.mean(np.abs(pr - gt))), mae_stance=float(np.mean(np.abs(pr[st] - gt[st]))),
                  r=float(np.corrcoef(pr, gt)[0, 1]), latency_s=float(np.median(t[ok] - L.surrogate_time.to_numpy()[ok])),
                  mx_convention=a.mx_convention, label_subdir=a.label_subdir, logged_mx_sign=s_log,
                  logged_height_m=h_log, recomputed_from_raw=bool(have_raw and np.isfinite(h_log)))
    dt = np.diff(t)
    metrics = dict(log=a.log, frames=int(len(t)), rate_hz=float(1 / np.median(dt)), angle_rmse_deg=deg,
                   angle_rmse_deg_max=max(deg[c] for c in COORDS[:16] if c in deg),
                   pelvis_height_rmse_m=py, pelvis_lateral_rmse_m=pz, forward_speed_mps=v_est, belt_speed_mps=belt,
                   contact_agreement=agree, contact_agreement_surrogate=agree_used, f_medial=fm,
                   mx_convention=a.mx_convention, label_subdir=a.label_subdir,
                   noise_deg=syn.get("noise_deg"))
    json.dump(metrics, open(os.path.join(a.out, "metrics.json"), "w"), indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False})
    BLUE, ORANGE, GREY = "#0072B2", "#E69F00", "#8A8A8A"
    fig, ax = plt.subplots(2, 2, figsize=(13, 7.2))
    m = walk & (t <= t[walk][0] + 4.0)
    for c, lab, col in (("knee_angle_r", "knee", BLUE), ("hip_flexion_r", "hip", ORANGE)):
        i = COORDS.index(c)
        ax[0, 0].plot(t[m], np.degrees(truth[m, i]), color="#111", lw=2.2, alpha=0.8)
        ax[0, 0].plot(t[m], np.degrees(est[m, i]), color=col, lw=1.6, ls="--", label=f"{lab} R: UKF (RMSE {deg[c]:.1f}°)")
    ax[0, 0].plot([], [], color="#111", lw=2.2, label="mocap (truth)")
    ax[0, 0].set_ylabel("deg"); ax[0, 0].set_title("joint angles from suit orientations"); ax[0, 0].legend(fontsize=9, loc="upper right")
    yi = COORDS.index("pelvis_ty")
    ax[0, 1].plot(t[m], (truth[m, yi] - truth[walk, yi].mean()) * 100, color="#111", lw=2.2, label="mocap (truth)")
    ax[0, 1].plot(t[m], (L.pelvis_y.to_numpy()[m] - L.pelvis_y.to_numpy()[walk].mean()) * 100, color=BLUE, lw=1.6, ls="--",
                  label=f"UKF, contact-anchored (RMSE {py * 100:.1f} cm)")
    ax[0, 1].set_ylabel("cm"); ax[0, 1].set_title(f"pelvis height · forward speed {v_est:.2f} m/s vs belt {belt:.2f} m/s")
    ax[0, 1].legend(fontsize=9, loc="upper right")
    # with back-dating, show the flags the network saw; otherwise the published online flags
    shown, ag, lab = ((used, agree_used, "online, heel strike back-dated") if used is not None and fm else
                      ({"r": L.contact_r.to_numpy() > 0, "l": L.contact_l.to_numpy() > 0}, agree, "online detector"))
    ax[1, 0].fill_between(t[m], 1.05, 1.95, where=lc[m], color="#DDDDDD", step="mid", lw=0, label="force plate")      # left on top
    ax[1, 0].fill_between(t[m], 1.25, 1.75, where=shown["l"][m], color=BLUE, step="mid", lw=0, label=lab)
    ax[1, 0].fill_between(t[m], 0.05, 0.95, where=rc[m], color="#DDDDDD", step="mid", lw=0)
    ax[1, 0].fill_between(t[m], 0.25, 0.75, where=shown["r"][m], color=BLUE, step="mid", lw=0)
    ax[1, 0].set_yticks([0.5, 1.5], ["right", "left"]); ax[1, 0].set_xlabel("time (s)")
    ax[1, 0].set_title(f"foot contact (agreement L {ag['l']:.2f}, R {ag['r']:.2f})"); ax[1, 0].legend(fontsize=9, loc="upper right", ncol=2)
    if fm:
        ok = L.f_medial.notna().to_numpy() & m
        ts = L.surrogate_time.to_numpy()[ok]
        jt, jn, jv = read_mot(label_file(a.trial, a.label_subdir), angles=False)
        BW = mass * 9.81
        tg = np.linspace(t[m][0], t[m][-1], 400)
        g = label_fmedial(jt, jn, jv, tg + syn["time_offset"], BW, sign)
        ax[1, 1].plot(tg, g, color="#111", lw=2.2, label="OpenSim (physics-based)")
        ax[1, 1].plot(ts, pred_all[ok], color=ORANGE, lw=1.8, label=f"live surrogate (MAE {fm['mae']:.2f} BW)")
        ax[1, 1].axhline(1.4, color=GREY, ls="--", lw=1)
        ax[1, 1].set_ylabel("F_medial (BW)"); ax[1, 1].set_xlabel("time (s)"); ax[1, 1].legend(fontsize=9, loc="upper right")
        ax[1, 1].set_title(f"medial knee force, live (latency {fm['latency_s'] * 1000:.0f} ms; M_x {a.mx_convention})")
    fig.suptitle(a.title or f"PN3 driver on virtual-suit data · {metrics['rate_hz']:.0f} Hz", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "driver_eval.png"), dpi=150)
    print(json.dumps({k: v for k, v in metrics.items() if k != "angle_rmse_deg"}, indent=1))
    print("written to", a.out)


if __name__ == "__main__":
    main()

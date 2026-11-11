#!/usr/bin/env python3
"""Quality diagnostics for the OpenSim SO+JR knee labels (jcf_output_sojr/). Reads files only; it never runs OpenSim
tools. Only the `masks` command writes into the trial folders (<trial>/jcf_output_sojr/label_mask.npz).

Per frame (SO time grid) of one trial folder, from files that already exist:
    jcf_output/BatchJCF_StaticOptimization_force.sto        muscle forces (N), reserve and pelvis residual
                                                             generalized forces (N, N*m; reserves have optimal force
                                                             1 N*m, pelvis translations 10 N, controls unbounded)
    jcf_output/BatchJCF_StaticOptimization_activation.sto   activations: saturation is activation >= SAT_ACT
    jcf_output_sojr/BatchJCF_JointReaction_ReactionLoads.sto SO+JR loads of both knees (optional)
    grf_data.mot, ik_results.mot, metadata.json
and per trial the centre-of-pressure travel of each foot: the median, over stance runs with vertical GRF above
COP_BW, of the larger of the x and z ranges of the COP in ground, and that travel divided by the run duration (COP
speed). A physical COP travels about a foot length heel to toe (more on a treadmill, where the foot also rides the
belt), so a few millimetres per stance means the GRF is applied at a point fixed in the lab, which puts the line of
action tens of centimetres away from the foot.

fixed_cop is a per-foot flag, set on the row of the knee on the same side: that foot has gait-like stance runs
(median run <= GAIT_STANCE_S) and a median COP speed below FIXED_COP_MPS, in a walking (non-static) folder. The knee
moment, and therefore the knee-crossing muscle forces and the knee label, depend on the GRF of that leg only, so a
moving left COP does not clean a right-knee label. fixed_cop_trial (either foot fixed) is reported separately.
FIXED_COP_MPS = 0.2 m/s sits in the empty gap of the Carter speeds (none between 0.15 and 0.5 m/s); the other labs
have no such gap, so the threshold matters there (rules.csv reports the counts at 0.1 m/s as a sensitivity check).

Frame rules (counted on all frames and on that knee's stance frames, for each knee):
    sat_knee   a muscle crossing that knee (KNEE_MUSCLES) is saturated
    sat_leg3   at least 3 muscles of that leg are saturated
    res_leg    a reserve of that leg (hip x3, knee, ankle) exceeds RES_NMKG N*m/kg
    pelvis_m   the pelvis residual moment exceeds PELVIS_M_PCT % of (peak total GRF x pelvis height),
               a Hicks et al. 2015 style residual-moment criterion (their guideline is about 1 %)
    jump       a leg muscle's activation changes by more than JUMP_PER_10MS per 10 ms between SO frames
    coact      a dorsiflexor (tibant, edl, ehl) and a plantarflexor (soleus, gasmed, gaslat) both above COACT
    fy4, fy5   knee |F_y| above 4 or 5 BW
fmed_<side> = 0.5*|F_y| + M_x*H/0.045 with the medial-positive M_x = -mx (right) / +mx (left), MX_MEDIAL.
Trial flags: fixed_cop (see above; never set in the static folders), the training-loader filters
(resultant peak > 6 BW, |M_x| > 0.15 BW*H, flat waveform: CV < 0.25 or range < 0.3 BW) on the right knee.

Shares: the SO grid differs between labs (Carter and Han 250 Hz, Tiziana 200 Hz, Moore and Falisse 100 Hz, Fregly
120-200 Hz), and the thesis input sets resample to 100 Hz, so rule_table reports time-weighted shares (frames x SO
time step: the share on a common time grid such as the 100 Hz training grid) next to the native frame shares.

`classify_peak` names the most likely cause of a stance |F_y| peak from the peak-frame metrics of the scan;
`classify_sample_peak` does the same with the moment-arm split (`moment_arm_peak`, OpenSim API read-only: it loads the
model and evaluates moment arms, nothing is solved or written) and the re-solve cost ratio. Both attribute a peak to a
fixed COP only when the peak frame shows the mechanism (an extension-dominated knee with that foot's COP fixed).

`resolve_so_frame` (OpenSim API read-only + scipy) re-solves the SO problem of one frame with the same moment arms,
rigid-tendon capacities and net generalized forces; a cost well above the re-solved optimum means the stored SO
solution did not converge. `annotate` attaches the final SO+JR |F_y| and the COP flags to a resolve CSV (file reads).

Frame masks for training (`masks`): <trial>/jcf_output_sojr/label_mask.npz, on the JR time grid (the `time` column
of the SO+JR JointReaction file; the SO grid is the same in every folder, checked per folder as so_jr_dt_max):
    keep_r, keep_l            bool per JR frame: not fixed_cop_<side> and none of MASK_SO_RULES on that frame (and the
                              label and SO rows finite). train_mlp's *_sojrqc input sets train on keep_r frames only.
    <rule>_r, <rule>_l        the per-frame flags of MASK_SO_RULES (definitions above, frame_flags), so_any_<side>
                              (their union), nonfinite_<side>, stance_<side>, and fy5_<side> (|F_y| > 5 BW): a
                              label-value rule that is stored but not part of keep_<side>.
    fixed_cop_r, fixed_cop_l  trial-level per-foot flags at FIXED_COP_MPS (0.2 m/s), with cop_speed_<side>
    rules, version, thresholds (JSON), label_file, so_jr_dt_max
The union in keep_<side> is the six SO frame rules, sat_leg3 included; the any_so union of rules.csv leaves sat_leg3
out (`masks` reports what that difference keeps). Summary: results/thesis/labels_qc/label_masks/ (per_trial.csv,
by_lab.csv with the frames the 5 BW rule would remove in addition, README.md).

Usage (repo root):
    python -I analysis/thesis/so_label_qc.py masks --workers 20        # -> <trial>/jcf_output_sojr/label_mask.npz
    python -I analysis/thesis/so_label_qc.py scan --workers 4          # -> results/thesis/labels_qc/trial_qc.csv (~2 min)
    python -I analysis/thesis/so_label_qc.py rules                     # -> rules.csv, rules_label_stats.csv
    python -I analysis/thesis/so_label_qc.py sample --folders <names> --moments-from <old sample_peaks.csv>  # no OpenSim
    python -I analysis/thesis/so_label_qc.py annotate --resolve results/thesis/labels_qc/resolve_random_stance.csv
    python -I analysis/thesis/so_label_qc.py classify                  # -> sample_classification.csv
    python -I analysis/thesis/so_label_qc.py peak jcf/full_duration/training/walking/carter_P003_split0_t05 --side r
    conda run -n jcf python analysis/thesis/so_label_qc.py sample --folders <names> \
        --root jcf/full_duration/training/walking jcf/full_duration/testing/walking   # -> sample_peaks.csv
    conda run -n jcf python -W ignore analysis/thesis/so_label_qc.py resolve --sample results/thesis/labels_qc/sample_peaks.csv \
        --root jcf/full_duration/training/walking jcf/full_duration/testing/walking --out results/thesis/labels_qc/resolve_sample_peaks.csv
    conda run -n jcf python -W ignore analysis/thesis/so_label_qc.py resolve --random 25 --seed 1 \
        --root jcf/full_duration/training/walking jcf/full_duration/testing/walking --out results/thesis/labels_qc/resolve_random_stance.csv
"""
import os as _os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")
import argparse
import glob
import json
import multiprocessing as mp
import os
import re
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from analysis.thesis.gc_knee_jr import JR_FILE, OUT_SUBDIR, SO_FORCES, knee_prefix, read_sto  # noqa: E402

G = 9.81
SO_ACT = "BatchJCF_StaticOptimization_activation.sto"
SAT_ACT = 0.99
STANCE_BW = 0.05          # contact threshold used by train_mlp and gc_knee_jr
COP_BW = 0.2              # COP is noisy at low force; travel is measured above this vertical GRF
MIN_RUN = 10              # frames in a stance run before its COP travel counts
FIXED_COP_MPS = 0.2       # COP speed below which a walking stance has a COP fixed in the lab (heel to toe is ~0.3 m/s;
                          # Carter speeds have an empty gap between 0.15 and 0.5 m/s)
FIXED_COP_SENS = 0.1      # alternative threshold reported as a sensitivity check
GAIT_STANCE_S = 1.0       # longest median stance run (s) for which the fixed-COP test applies
RES_NMKG = 0.1            # about 7.5 N*m at 75 kg; a 1 N*m-optimal-force reserve costs as much as 56 saturated muscles
PELVIS_M_PCT = 5.0
JUMP_PER_10MS = 0.5       # largest activation change of a leg muscle between SO frames, scaled to 10 ms
COACT = 0.5               # dorsiflexor and plantarflexor activations both above this
PHI, D_MEDIAL = 0.5, 0.045
MX_MEDIAL = {"r": -1.0, "l": 1.0}   # medial-positive M_x = MX_MEDIAL * raw JR mx (knee_mx_sign.py, physical;
                                   # trial_qc.csv files written before 2026-11-10 used the inverted +mx right)
EXT_NMKG = 1.5            # net knee extension moment (N*m/kg) about twice the loading-response peak of normal walking
CCI_LOW = 0.3             # knee co-contraction index below which the knee is extension (or flexion) dominated
EXT_BW = 1.0              # summed vasti + rectus femoris force (BW) above which a peak is extensor driven (scan level)
COST_RATIO_NONOPT = 1.1   # stored SO cost / re-solved optimum above which the stored solution did not converge
NON_MUSCLE = ("reserve_", "lumbar_", "shoulder_", "elbow_", "pro_sup_", "wrist_")
KNEE_MUSCLES = ("vaslat", "vasmed", "vasint", "recfem", "bflh", "bfsh", "semimem", "semiten", "gaslat", "gasmed",
                "grac", "sart", "tfl")
KNEE_EXT = ("vaslat", "vasmed", "vasint", "recfem")
KNEE_FLEX = ("bflh", "bfsh", "semimem", "semiten", "gaslat", "gasmed")
ANKLE_DF = ("tibant", "edl", "ehl")
ANKLE_PF = ("soleus", "gasmed", "gaslat")
LEG_RESERVES = ("hip_flexion", "hip_adduction", "hip_rotation", "knee_angle", "ankle_angle")
PELVIS_F = ("pelvis_tx", "pelvis_ty", "pelvis_tz")
PELVIS_M = ("pelvis_tilt", "pelvis_list", "pelvis_rotation")
FRAME_RULES = ("sat_knee", "sat_leg3", "res_leg", "pelvis_m", "jump", "coact", "fy4", "fy5")
LABS = ("carter", "moore", "tiziana", "falisse", "fregly", "han")
DEFAULT_ROOTS = tuple(os.path.join(REPO, "jcf", "full_duration", s, a)
                      for s in ("training", "testing") for a in ("walking", "static"))
MASK_ROOTS = DEFAULT_ROOTS + (os.path.join(REPO, "jcf", "external"),)
MASK_FILE = "label_mask.npz"
MASK_SO_RULES = ("sat_knee", "sat_leg3", "res_leg", "pelvis_m", "jump", "coact")   # keep_<side> = none of these
MASK_VALUE_RULES = ("fy5",)                                                          # stored, not in keep_<side>
ANY_SO_RULES = ("sat_knee", "res_leg", "pelvis_m", "jump", "coact")                  # rules.csv any_so (no sat_leg3)
MASK_VERSION = 1
FULL_DURATION = "FullDuration_"   # file prefix of an AnalyzeTool named FullDuration (jcf/external/william_2026-09-23)


def results_file(d, subdir, name):
    """<d>/<subdir>/<name>, or the same file with the FullDuration_ prefix instead of BatchJCF_ when only that exists
    (None when neither exists)."""
    p = os.path.join(d, subdir, name)
    if os.path.exists(p):
        return p
    alt = os.path.join(d, subdir, name.replace("BatchJCF_", FULL_DURATION, 1))
    return alt if os.path.exists(alt) else None


def muscles_of(columns, side=None):
    """Muscle columns of an SO table (not reserves or the torque actuators), optionally of one leg."""
    m = [c for c in columns if c != "time" and not c.startswith(NON_MUSCLE) and c[-2:] in ("_r", "_l")]
    return [c for c in m if c.endswith("_" + side)] if side else m


def lab_of(folder):
    name = os.path.basename(os.path.normpath(folder)).lower()
    return next((lab for lab in LABS if name.startswith(lab)), name.split("_")[0])


def person_of(folder):
    """Person ID of a trial folder: the name without _tNN(_rNN) and without a _splitN suffix (Carter, Han)."""
    name = re.sub(r"_t\d+(_r\d+)?$", "", os.path.basename(os.path.normpath(folder)))
    return re.sub(r"_split\d+$", "", name)


def fixed_cop_foot(speed, dur, static=False, threshold=FIXED_COP_MPS):
    """True when one foot's COP is fixed in the lab: gait-like stance runs (median <= GAIT_STANCE_S) whose median COP
    speed is below threshold; False for static folders and for a foot without measurable stance runs (NaN)."""
    if static or not (np.isfinite(speed) and np.isfinite(dur)):
        return False
    return bool(dur <= GAIT_STANCE_S and speed < threshold)


def extension_dominated(ext, flex):
    """Scan-level proxy of the fixed-COP mechanism: a large extensor force with little flexor force (force CCI)."""
    return bool(ext > EXT_BW and 2.0 * min(ext, flex) / (ext + flex) < CCI_LOW)


def _meta(d):
    meta = json.load(open(os.path.join(d, "metadata.json")))
    mass = float(meta["mass_kg"])
    return mass, mass * G, float(meta.get("height_m", 1.7))


def _stance_runs(mask, min_len=1):
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(int), 0]))
    return [(a, b) for a, b in zip(edges[::2], edges[1::2]) if b - a >= min_len]


def cop_travel(grf, side, bw):
    """(median COP travel in m, number of runs, median COP speed in m/s, median run duration in s) of one foot over
    its stance runs with vertical GRF > COP_BW and at least MIN_RUN frames; NaNs without runs. Speed = travel / run
    duration, so a short (half-stance) window is not mistaken for a fixed COP."""
    fy = grf[f"calcn_{side}_force_vy"].to_numpy(float) / bw
    px = grf[f"calcn_{side}_force_px"].to_numpy(float)
    pz = grf[f"calcn_{side}_force_pz"].to_numpy(float)
    t = grf["time"].to_numpy(float)
    dt = float(np.median(np.diff(t))) if len(t) > 1 else np.nan
    runs = _stance_runs(fy > COP_BW, MIN_RUN)
    tr = [max(np.ptp(px[a:b]), np.ptp(pz[a:b])) for a, b in runs]
    sp = [x / ((b - a) * dt) for x, (a, b) in zip(tr, runs)]
    if not tr:
        return np.nan, 0, np.nan, np.nan
    return float(np.median(tr)), len(tr), float(np.median(sp)), float(np.median([(b - a) * dt for a, b in runs]))


def frame_table(d, sojr_subdir=OUT_SUBDIR):
    """Per-frame label-quality metrics of one folder on the SO time grid (see module docstring)."""
    mass, bw, H = _meta(d)
    so = read_sto(results_file(d, "jcf_output", SO_FORCES) or os.path.join(d, "jcf_output", SO_FORCES))
    act = read_sto(results_file(d, "jcf_output", SO_ACT) or os.path.join(d, "jcf_output", SO_ACT))
    g = read_sto(os.path.join(d, "grf_data.mot"))
    ik = read_sto(os.path.join(d, "ik_results.mot"))
    jr_path = results_file(d, sojr_subdir, JR_FILE)
    jr = read_sto(jr_path) if jr_path else None
    t = so["time"].to_numpy(float)
    gt = g["time"].to_numpy(float)
    out = {"time": t}
    ftot = np.sqrt(sum((g[f"calcn_r_force_v{a}"] + g[f"calcn_l_force_v{a}"]).to_numpy(float) ** 2 for a in "xyz"))
    max_ext = max(float(ftot.max()), 1e-9)
    h_com = float(ik["pelvis_ty"].mean()) if "pelvis_ty" in ik else 0.55 * H
    amus = muscles_of(act.columns)
    out["nsat_all"] = (act[amus].to_numpy(float) >= SAT_ACT).sum(1)
    for s, sign in (("r", MX_MEDIAL["r"]), ("l", MX_MEDIAL["l"])):
        gy = np.interp(t, gt, g[f"calcn_{s}_force_vy"].to_numpy(float)) / bw
        out[f"grf_{s}"] = gy
        out[f"stance_{s}"] = np.abs(gy) > STANCE_BW
        if f"knee_angle_{s}" in ik:
            out[f"knee_deg_{s}"] = np.degrees(np.interp(t, ik["time"].to_numpy(float), ik[f"knee_angle_{s}"]))
        A = act[muscles_of(act.columns, s)].to_numpy(float)
        out[f"nsat_leg_{s}"] = (A >= SAT_ACT).sum(1)
        step = np.diff(t)
        scale = 0.01 / np.where(step > 0, step, np.inf)
        out[f"jump_{s}"] = np.r_[0.0, np.abs(np.diff(A, axis=0)).max(1) * scale] if len(t) > 1 else np.zeros(len(t))
        df_ = [f"{m}_{s}" for m in ANKLE_DF if f"{m}_{s}" in act]
        pf_ = [f"{m}_{s}" for m in ANKLE_PF if f"{m}_{s}" in act]
        out[f"coact_{s}"] = (np.minimum(act[df_].max(1), act[pf_].max(1)).to_numpy(float) if df_ and pf_
                             else np.zeros(len(t)))
        km = [f"{m}_{s}" for m in KNEE_MUSCLES if f"{m}_{s}" in act]
        out[f"nsat_knee_{s}"] = (act[km].to_numpy(float) >= SAT_ACT).sum(1)
        res = {c: np.abs(so[f"reserve_{c}_{s}"].to_numpy(float)) / mass for c in LEG_RESERVES
               if f"reserve_{c}_{s}" in so}
        z = np.zeros(len(t))
        out[f"res_knee_{s}"] = res.get("knee_angle", z)
        out[f"res_ankle_{s}"] = res.get("ankle_angle", z)
        out[f"res_hip_{s}"] = np.max([res[c] for c in res if c.startswith("hip")] or [z], axis=0)
        out[f"res_leg_{s}"] = np.max(list(res.values()) or [z], axis=0)
        out[f"ext_{s}"] = so[[f"{m}_{s}" for m in KNEE_EXT if f"{m}_{s}" in so]].sum(1).to_numpy(float) / bw
        out[f"flex_{s}"] = so[[f"{m}_{s}" for m in KNEE_FLEX if f"{m}_{s}" in so]].sum(1).to_numpy(float) / bw
        pre = knee_prefix(s)
        if jr is not None and pre + "fy" in jr:
            jt = jr["time"].to_numpy(float)
            f = {c: np.interp(t, jt, jr[pre + c].to_numpy(float)) for c in ("fx", "fy", "fz", "mx")}
            out[f"fy_{s}"] = f["fy"] / bw
            out[f"fres_{s}"] = np.sqrt(f["fx"] ** 2 + f["fy"] ** 2 + f["fz"] ** 2) / bw
            out[f"mx_{s}"] = f["mx"] / (bw * H)
            out[f"fmed_{s}"] = PHI * np.abs(f["fy"]) / bw + sign * f["mx"] / (bw * D_MEDIAL)
    pf = np.sqrt(sum(so[f"reserve_{c}"].to_numpy(float) ** 2 for c in PELVIS_F if f"reserve_{c}" in so))
    pm = np.sqrt(sum(so[f"reserve_{c}"].to_numpy(float) ** 2 for c in PELVIS_M if f"reserve_{c}" in so))
    out["pelvis_f_pct"] = 100.0 * pf / max_ext
    out["pelvis_m_pct"] = 100.0 * pm / (max_ext * h_com)
    ft = pd.DataFrame(out)
    ft.attrs.update(mass=mass, bw=bw, height=H, max_ext_n=max_ext, h_com=h_com)
    for s in "rl":
        (ft.attrs[f"cop_travel_{s}"], ft.attrs[f"cop_runs_{s}"], ft.attrs[f"cop_speed_{s}"],
         ft.attrs[f"cop_dur_{s}"]) = cop_travel(g, s, bw)
    return ft


def frame_flags(ft, side):
    """Boolean per-frame rule flags (FRAME_RULES) for one knee; fy rules are False without SO+JR labels."""
    fy = ft[f"fy_{side}"].abs() if f"fy_{side}" in ft else pd.Series(0.0, index=ft.index)
    return pd.DataFrame({
        "sat_knee": ft[f"nsat_knee_{side}"] >= 1,
        "sat_leg3": ft[f"nsat_leg_{side}"] >= 3,
        "res_leg": ft[f"res_leg_{side}"] > RES_NMKG,
        "pelvis_m": ft["pelvis_m_pct"] > PELVIS_M_PCT,
        "jump": ft[f"jump_{side}"] > JUMP_PER_10MS,
        "coact": ft[f"coact_{side}"] > COACT,
        "fy4": fy > 4.0,
        "fy5": fy > 5.0,
    })


def time_since_strike(stance, i, dt):
    """Seconds from the onset of the stance run containing frame i (NaN if i is not in stance)."""
    if not stance[i]:
        return np.nan
    j = i
    while j > 0 and stance[j - 1]:
        j -= 1
    return float((i - j) * dt)


def classify_peak(p):
    """Most likely cause of a stance |F_y| peak from a dict of peak-frame metrics (keys as in trial_summary).

    Order: a fixed COP when that foot's COP is fixed and the peak frame is extension dominated (the mechanism: the
    misplaced GRF demands a large knee extension moment that the quadriceps supply without antagonists); then dynamic
    inconsistency, then a non-smooth SO solution (activation jump within +-2 frames of the peak or ankle
    dorsi/plantarflexor co-activation: a convex SO problem on 6 Hz kinematics has neither), then SO at capacity, then a
    contact transient, then antagonist co-contraction, then deep knee flexion (a large but possibly genuine moment).
    A fixed-COP trial whose peak lacks the mechanism falls through to the other causes; fixed_cop stays a column."""
    if p.get("fixed_cop") and extension_dominated(p.get("pk_ext", 0.0), p.get("pk_flex", 0.0)):
        return "fixed_cop"
    if p.get("pk_pelvis_m_pct", 0) > PELVIS_M_PCT:
        return "dyn_inconsistent"
    if p.get("pk_jump", 0) > JUMP_PER_10MS or p.get("pk_coact", 0) > COACT:
        return "so_nonsmooth"
    if p.get("pk_nsat_knee", 0) >= 1 or p.get("pk_res_leg", 0) > RES_NMKG:
        return "saturation"
    if p.get("pk_t_strike", np.inf) < 0.03:
        return "strike_transient"
    if p.get("pk_ext", 0) > 1.0 and p.get("pk_flex", 0) > 0.5 * p.get("pk_ext", 0):
        return "cocontraction"
    if p.get("pk_knee_deg", 0) > 40:
        return "deep_flexion"
    return "unexplained"


SAMPLE_CLASSES = {
    "A": "fixed COP (GRF artefact)",
    "B": "SO not converged",
    "C": "dynamic inconsistency",
    "D": "deep knee flexion",
    "E": "consistent",
    "F": "high load at the SO optimum, unexplained",
}


def classify_sample_peak(r):
    """Class (SAMPLE_CLASSES key) of a sampled peak from its peak-frame mechanism (dict or row with the moment-arm split
    of moment_arm_peak, the re-solve cost ratio and the scan columns). The trial flags are not used on their own.

    B  stored SO cost > COST_RATIO_NONOPT x the re-solved optimum (the label is not the SO answer)
    A  at the optimum, knee net extension moment > EXT_NMKG N*m/kg with knee CCI < CCI_LOW, and that foot's COP fixed
    C  at the optimum, pelvis residual moment > PELVIS_M_PCT % at the peak
    D  at the optimum, residual small, knee flexed > 40 deg and extension dominated (CCI < CCI_LOW)
    F  at the optimum, residual small, |F_y| > 4 BW without A, C or D (load demanded by the solution, cause unknown)
    E  the rest: optimal, residual small, peak <= 4 BW or explained by a normal moment pattern"""
    g = r.get if isinstance(r, dict) else (lambda k, d=None: r[k] if k in r and pd.notna(r[k]) else d)
    if g("cost_ratio", 1.0) > COST_RATIO_NONOPT:
        return "B"
    ext_dom = g("knee_angle_net_nmkg", 0.0) < -EXT_NMKG and g("knee_angle_cci", 1.0) < CCI_LOW
    if ext_dom and g("fixed_cop", False):
        return "A"
    if g("pk_pelvis_m_pct", 0.0) > PELVIS_M_PCT:
        return "C"
    if g("pk_knee_deg", 0.0) > 40 and g("knee_angle_cci", 1.0) < CCI_LOW:
        return "D"
    if g("fy_peak", 0.0) > 4.0:
        return "F"
    return "E"


def trial_summary(ft, folder):
    """One row per knee: label stats, peak-frame metrics, rule counts, training-filter flags, cause of the peak."""
    rows = []
    dt = float(np.median(np.diff(ft["time"]))) if len(ft) > 1 else np.nan
    # a still COP is expected while standing, so the fixed-COP flag only applies outside the static folders, and only
    # to gait-like stance runs: a foot planted for seconds (quasi-static tasks, standing) has a still COP
    static = os.path.basename(os.path.dirname(os.path.normpath(folder))) == "static"
    foot = {s: fixed_cop_foot(ft.attrs.get(f"cop_speed_{s}", np.nan), ft.attrs.get(f"cop_dur_{s}", np.nan), static)
            for s in "rl"}
    sens = {s: fixed_cop_foot(ft.attrs.get(f"cop_speed_{s}", np.nan), ft.attrs.get(f"cop_dur_{s}", np.nan), static,
                              FIXED_COP_SENS) for s in "rl"}
    for s in "rl":
        st = ft[f"stance_{s}"].to_numpy(bool)
        row = dict(folder=os.path.basename(os.path.normpath(folder)), lab=lab_of(folder), person=person_of(folder),
                   side=s, n=len(ft), n_stance=int(st.sum()), dt=dt, mass=ft.attrs["mass"], height=ft.attrs["height"],
                   cop_travel_r=ft.attrs.get("cop_travel_r"), cop_travel_l=ft.attrs.get("cop_travel_l"),
                   cop_speed_r=ft.attrs.get("cop_speed_r"), cop_speed_l=ft.attrs.get("cop_speed_l"),
                   cop_dur_r=ft.attrs.get("cop_dur_r"), cop_dur_l=ft.attrs.get("cop_dur_l"),
                   fixed_cop=foot[s], fixed_cop_trial=foot["r"] or foot["l"], fixed_cop_sens=sens[s],
                   has_sojr=f"fy_{s}" in ft)
        for name, k in (("pelvis_m_pct", "pelvis_m_pct"), ("pelvis_f_pct", "pelvis_f_pct"),
                        ("res_leg", f"res_leg_{s}"), ("nsat_leg", f"nsat_leg_{s}")):
            if st.any():
                row[f"stance_med_{name}"] = float(np.median(ft.loc[st, k]))
                row[f"stance_p95_{name}"] = float(np.percentile(ft.loc[st, k], 95))
        fl = frame_flags(ft, s)
        for r in FRAME_RULES:
            row[f"n_{r}"] = int(fl[r].sum())
            row[f"n_{r}_stance"] = int((fl[r] & st).sum())
        bad = fl.sat_knee | fl.res_leg | fl.pelvis_m | fl.jump | fl.coact
        row["n_any_so"] = int(bad.sum())
        row["n_any_so_stance"] = int((bad & st).sum())
        if f"fy_{s}" in ft:
            fy = ft[f"fy_{s}"].abs().to_numpy(float)
            res = ft[f"fres_{s}"].to_numpy(float)
            row.update(fres_max=float(res.max()), mx_absmax=float(ft[f"mx_{s}"].abs().max()),
                       flat=bool(res.std() / (res.mean() + 1e-12) < 0.25 or np.ptp(res) < 0.3))
            if st.any():
                i = int(np.flatnonzero(st)[np.argmax(fy[st])])
                row.update(fy_peak=float(fy[i]), fy_mean=float(fy[st].mean()), fy_p95=float(np.percentile(fy[st], 95)),
                           fmed_peak=float(ft.loc[st, f"fmed_{s}"].max()), fmed_mean=float(ft.loc[st, f"fmed_{s}"].mean()),
                           pk_time=float(ft.time[i]), pk_grf=float(ft[f"grf_{s}"][i]),
                           pk_knee_deg=float(ft[f"knee_deg_{s}"][i]) if f"knee_deg_{s}" in ft else np.nan,
                           pk_t_strike=time_since_strike(st, i, dt), pk_nsat_leg=int(ft[f"nsat_leg_{s}"][i]),
                           pk_nsat_knee=int(ft[f"nsat_knee_{s}"][i]), pk_res_leg=float(ft[f"res_leg_{s}"][i]),
                           pk_res_knee=float(ft[f"res_knee_{s}"][i]), pk_res_ankle=float(ft[f"res_ankle_{s}"][i]),
                           pk_res_hip=float(ft[f"res_hip_{s}"][i]), pk_pelvis_m_pct=float(ft.pelvis_m_pct[i]),
                           pk_ext=float(ft[f"ext_{s}"][i]), pk_flex=float(ft[f"flex_{s}"][i]),
                           pk_jump=float(ft[f"jump_{s}"][max(0, i - 2):i + 3].max()),
                           pk_coact=float(ft[f"coact_{s}"][i]),
                           pk_mx_medial=float(MX_MEDIAL[s] * ft[f"mx_{s}"][i]),
                           pk_fmed=float(ft[f"fmed_{s}"][i]))
                row["cause"] = classify_peak(row)
                # per-step peaks over full stance runs (not cut by the trial ends), comparable with in vivo step peaks;
                # fy_peak is the maximum over all steps of a trial (about 6 steps in a 7 s Carter trial)
                steps = [float(fy[a:b].max()) for a, b in _stance_runs(st, MIN_RUN) if a > 0 and b < len(st)]
                row.update(n_full_steps=len(steps), fy_step_peak_med=float(np.median(steps)) if steps else np.nan)
                for tag, keep in (("clean_so", st & ~bad.to_numpy()),
                                  ("clean_core", st & ~(fl.sat_knee | fl.res_leg).to_numpy())):
                    row[f"n_{tag}"] = int(keep.sum())
                    if keep.any():
                        row[f"fy_peak_{tag}"] = float(fy[keep].max())
                        row[f"fy_mean_{tag}"] = float(fy[keep].mean())
                        row[f"fmed_mean_{tag}"] = float(ft.loc[keep, f"fmed_{s}"].mean())
        rows.append(row)
    return rows


def scan_folder(job):
    d, sojr_subdir = job
    try:
        if not results_file(d, "jcf_output", SO_FORCES):
            return []
        rows = trial_summary(frame_table(d, sojr_subdir), d)
        for r in rows:
            r["split"] = os.path.basename(os.path.dirname(os.path.dirname(os.path.normpath(d))))
            r["activity"] = os.path.basename(os.path.dirname(os.path.normpath(d)))
        return rows
    except Exception as e:                  # one unreadable folder must not stop the scan
        return [dict(folder=os.path.basename(d), lab=lab_of(d), error=f"{type(e).__name__}: {e}")]


def _split_activity(d):
    """(split, activity) of a trial folder: (training|testing, walking|static), or ("external", "external")."""
    parent = os.path.basename(os.path.dirname(os.path.normpath(d)))
    if parent == "external":
        return "external", "external"
    return os.path.basename(os.path.dirname(os.path.dirname(os.path.normpath(d)))), parent


def mask_arrays(d, sojr_subdir=OUT_SUBDIR, ft=None):
    """Contents of <d>/<sojr_subdir>/label_mask.npz (see the module docstring) as a dict of numpy arrays.

    The rule flags are frame_flags() of frame_table() (SO time grid) taken at the SO frame nearest each JR frame; the
    two grids are identical in every folder (so_jr_dt_max records the largest offset). fy5 and the finite check use
    the JR file itself. keep_<side> = not fixed_cop_<side> (fixed_cop_foot, as in trial_summary) and no MASK_SO_RULES
    flag and finite label and SO rows."""
    jr_path = results_file(d, sojr_subdir, JR_FILE)
    if jr_path is None:
        raise FileNotFoundError(f"no {sojr_subdir}/ JointReaction file in {d}")
    ft = frame_table(d, sojr_subdir) if ft is None else ft
    jr = read_sto(jr_path)
    t_jr = jr["time"].to_numpy(float)
    t_so = ft["time"].to_numpy(float)
    j = np.clip(np.searchsorted(t_so, t_jr), 1, max(len(t_so) - 1, 1))
    idx = np.where(np.abs(t_so[j - 1] - t_jr) <= np.abs(t_so[np.minimum(j, len(t_so) - 1)] - t_jr), j - 1,
                   np.minimum(j, len(t_so) - 1)) if len(t_so) > 1 else np.zeros(len(t_jr), int)
    so_dt_max = float(np.abs(t_so[idx] - t_jr).max()) if len(t_jr) else 0.0
    so = read_sto(results_file(d, "jcf_output", SO_FORCES) or os.path.join(d, "jcf_output", SO_FORCES))
    act = read_sto(results_file(d, "jcf_output", SO_ACT) or os.path.join(d, "jcf_output", SO_ACT))
    so_bad = (~np.isfinite(so.to_numpy(float)).all(1)) | (~np.isfinite(act.to_numpy(float)).all(1))
    static = _split_activity(d)[1] == "static"
    bw = ft.attrs["bw"]
    out = dict(time=t_jr, rules=np.array(MASK_SO_RULES), value_rules=np.array(MASK_VALUE_RULES),
               version=np.int64(MASK_VERSION), label_file=np.array(os.path.relpath(jr_path, d)),
               so_jr_dt_max=np.float64(so_dt_max),
               thresholds=np.array(json.dumps(dict(SAT_ACT=SAT_ACT, RES_NMKG=RES_NMKG, PELVIS_M_PCT=PELVIS_M_PCT,
                                                   JUMP_PER_10MS=JUMP_PER_10MS, COACT=COACT,
                                                   FIXED_COP_MPS=FIXED_COP_MPS, GAIT_STANCE_S=GAIT_STANCE_S,
                                                   STANCE_BW=STANCE_BW, FY_BW=5.0))))
    for side in "rl":
        fl = frame_flags(ft, side)
        speed, dur = ft.attrs.get(f"cop_speed_{side}", np.nan), ft.attrs.get(f"cop_dur_{side}", np.nan)
        fixed = fixed_cop_foot(speed, dur, static)
        pre = knee_prefix(side)
        lab = jr[[pre + c for c in ("fx", "fy", "fz", "mx")]].to_numpy(float) if pre + "fy" in jr else \
            np.full((len(t_jr), 4), np.nan)
        nonfinite = ~np.isfinite(lab).all(1) | so_bad[idx]
        flags = {r: fl[r].to_numpy(bool)[idx] for r in MASK_SO_RULES}
        so_any = np.any([flags[r] for r in MASK_SO_RULES], axis=0)
        with np.errstate(invalid="ignore"):
            fy5 = np.abs(lab[:, 1]) / bw > 5.0
        for r in MASK_SO_RULES:
            out[f"{r}_{side}"] = flags[r]
        out[f"so_any_{side}"] = so_any
        out[f"any_so5_{side}"] = np.any([flags[r] for r in ANY_SO_RULES], axis=0)   # rules.csv any_so, for reports
        out[f"fy5_{side}"] = fy5
        out[f"nonfinite_{side}"] = nonfinite
        out[f"stance_{side}"] = ft[f"stance_{side}"].to_numpy(bool)[idx]
        out[f"fixed_cop_{side}"] = np.bool_(fixed)
        out[f"cop_speed_{side}"] = np.float64(speed)
        out[f"keep_{side}"] = ~so_any & ~nonfinite & (not fixed)
    return out


def mask_summary(d, m):
    """One summary row of a mask dict (frames and time-weighted counts per side)."""
    split, activity = _split_activity(d)
    t = m["time"]
    dt = float(np.median(np.diff(t))) if len(t) > 1 else np.nan
    row = dict(folder=os.path.basename(os.path.normpath(d)), split=split, activity=activity, lab=lab_of(d),
               person=person_of(d), n=len(t), dt=dt, so_jr_dt_max=float(m["so_jr_dt_max"]),
               label_file=str(m["label_file"]))
    for s in "rl":
        st, keep = m[f"stance_{s}"], m[f"keep_{s}"]
        fixed = bool(m[f"fixed_cop_{s}"])
        keep5 = ~m[f"any_so5_{s}"] & ~m[f"nonfinite_{s}"] & (not fixed)
        row.update({f"fixed_cop_{s}": fixed, f"cop_speed_{s}": float(m[f"cop_speed_{s}"]),
                    f"n_stance_{s}": int(st.sum()), f"n_keep_{s}": int(keep.sum()),
                    f"n_keep_stance_{s}": int((keep & st).sum()),
                    f"n_so_any_{s}": int(m[f"so_any_{s}"].sum()), f"n_so_any_stance_{s}": int((m[f"so_any_{s}"] & st).sum()),
                    f"n_nonfinite_{s}": int(m[f"nonfinite_{s}"].sum()),
                    f"n_fy5_{s}": int(m[f"fy5_{s}"].sum()), f"n_fy5_stance_{s}": int((m[f"fy5_{s}"] & st).sum()),
                    f"n_fy5_kept_{s}": int((m[f"fy5_{s}"] & keep).sum()),
                    f"n_fy5_kept_stance_{s}": int((m[f"fy5_{s}"] & keep & st).sum()),
                    f"n_keep_any_so5_{s}": int(keep5.sum()), f"n_keep_any_so5_stance_{s}": int((keep5 & st).sum())})
        for r in MASK_SO_RULES:
            row[f"n_{r}_stance_{s}"] = int((m[f"{r}_{s}"] & st).sum())
    return row


def write_mask(d, m, sojr_subdir=OUT_SUBDIR, name=MASK_FILE):
    """Write a mask dict to <d>/<sojr_subdir>/<name> (temporary file, then rename: never half-written)."""
    path = os.path.join(d, sojr_subdir, name)
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **m)
    os.replace(tmp, path)
    return path


def mask_folder(job):
    """masks worker: build, write (unless dry) and summarise one folder; errors are returned as a row."""
    d, sojr_subdir, write = job
    try:
        m = mask_arrays(d, sojr_subdir)
        if write:
            write_mask(d, m, sojr_subdir)
        return mask_summary(d, m)
    except Exception as e:                  # one unreadable folder must not stop the run
        split, activity = _split_activity(d)
        return dict(folder=os.path.basename(os.path.normpath(d)), split=split, activity=activity, lab=lab_of(d),
                    error=f"{type(e).__name__}: {e}")


def mask_table(rows):
    """Per (split, activity, lab) and the thesis pool: right-knee trials, fixed-COP trials, stance time and what the
    mask keeps, and what the |F_y| > 5 BW rule would remove in addition among the kept frames (frames and
    time-weighted shares; the time weight is the JR step, i.e. shares on a common grid such as 100 Hz)."""
    df = pd.DataFrame([r for r in rows if "error" not in r])
    if not len(df):
        return pd.DataFrame()
    groups = list(df.groupby(["split", "activity", "lab"]))
    walk = df[(df.split == "training") & (df.activity == "walking")]
    groups.append((("training", "walking", "pool"), walk[walk.lab.isin(TRAIN_POOL)]))
    out = []
    for key, g in groups:
        if not len(g):
            continue
        w = g.dt.to_numpy(float)
        tw = lambda c: float((g[c].to_numpy(float) * w).sum())      # noqa: E731 - seconds
        st, kst = tw("n_stance_r"), tw("n_keep_stance_r")
        out.append(dict(zip(("split", "activity", "lab"), key), trials=len(g), people=int(g.person.nunique()),
                        trials_fixed_cop_r=int(g.fixed_cop_r.sum()),
                        trials_with_kept_frames_r=int((g.n_keep_r > 0).sum()),
                        people_with_kept_frames_r=int(g[g.n_keep_r > 0].person.nunique()),
                        frames=int(g.n.sum()), kept_frames_r=int(g.n_keep_r.sum()),
                        seconds=tw("n"), kept_seconds_r=tw("n_keep_r"),
                        stance_seconds_r=st, kept_stance_seconds_r=kst,
                        kept_frac_r=tw("n_keep_r") / max(tw("n"), 1e-12),
                        kept_stance_frac_r=kst / max(st, 1e-12),
                        kept_stance_frac_any_so5_r=tw("n_keep_any_so5_stance_r") / max(st, 1e-12),
                        fy5_stance_frames_r=int(g.n_fy5_stance_r.sum()),
                        fy5_after_so_frames_r=int(g.n_fy5_kept_r.sum()),
                        fy5_after_so_stance_frames_r=int(g.n_fy5_kept_stance_r.sum()),
                        fy5_after_so_stance_seconds_r=tw("n_fy5_kept_stance_r"),
                        fy5_after_so_frac_of_kept_stance_r=tw("n_fy5_kept_stance_r") / max(kst, 1e-12),
                        fy5_after_so_frac_of_kept_r=tw("n_fy5_kept_r") / max(tw("n_keep_r"), 1e-12),
                        kept_stance_frac_with_fy5_r=(kst - tw("n_fy5_kept_stance_r")) / max(st, 1e-12),
                        nonfinite_frames_r=int(g.n_nonfinite_r.sum()),
                        so_jr_dt_max=float(g.so_jr_dt_max.max())))
    return pd.DataFrame(out)


def cmd_masks(args):
    import time as _time
    t0 = _time.time()
    dirs = [d for d in list_folders(args.root, args.pattern) if results_file(d, args.sojr_subdir, JR_FILE)]
    print(f"{len(dirs)} folders with {args.sojr_subdir}/ JointReaction labels", flush=True)
    jobs = [(d, args.sojr_subdir, not args.dry_run) for d in dirs]
    rows = []
    if args.workers > 1:
        with mp.get_context("spawn").Pool(args.workers) as pool:
            for k, r in enumerate(pool.imap_unordered(mask_folder, jobs, chunksize=8)):
                rows.append(r)
                if (k + 1) % 1000 == 0:
                    print(f"  {k + 1}/{len(jobs)}  {_time.time() - t0:.0f} s", flush=True)
    else:
        rows = [mask_folder(j) for j in jobs]
    runtime = _time.time() - t0
    os.makedirs(args.out, exist_ok=True)
    per = pd.DataFrame(rows)
    per.to_csv(os.path.join(args.out, "per_trial.csv"), index=False)
    tab = mask_table(rows)
    tab.to_csv(os.path.join(args.out, "by_lab.csv"), index=False)
    errs = [r for r in rows if "error" in r]
    info = dict(command="python -I " + " ".join(sys.argv), n_folders=len(dirs), n_written=0 if args.dry_run else
                len(dirs) - len(errs), n_errors=len(errs), runtime_s=round(runtime, 1), workers=args.workers,
                mask_file=MASK_FILE, rules=list(MASK_SO_RULES), value_rules=list(MASK_VALUE_RULES),
                version=MASK_VERSION, errors=[(r["folder"], r["error"]) for r in errs][:50])
    json.dump(info, open(os.path.join(args.out, "run_info.json"), "w"), indent=1)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    cols = ["split", "activity", "lab", "trials", "people", "trials_fixed_cop_r", "trials_with_kept_frames_r",
            "kept_frac_r", "kept_stance_frac_r", "kept_stance_frac_any_so5_r", "fy5_after_so_stance_frames_r",
            "fy5_after_so_frac_of_kept_stance_r", "kept_stance_frac_with_fy5_r"]
    if len(tab):
        print(tab[cols].round(3).to_string(index=False))
    readme = [f"# SO+JR label masks ({MASK_FILE})", "",
              f"- command: `{info['command']}`; {len(dirs)} folders, {len(errs)} errors, {runtime:.0f} s on "
              f"{args.workers} workers" + (" (dry run: nothing written)" if args.dry_run else ""),
              f"- keep_r = right foot COP not fixed (< {FIXED_COP_MPS} m/s) and no frame flag of {', '.join(MASK_SO_RULES)}"
              " (frame_flags in analysis/thesis/so_label_qc.py) and finite label/SO rows. fy5 (|F_y| > 5 BW) is stored, "
              "not in keep_r.",
              "- by_lab.csv (right knee; *_seconds and *_frac are weighted by the JR time step): kept_stance_frac_r = "
              "share of stance time keep_r keeps; kept_stance_frac_any_so5_r = the same with the rules.csv any_so union "
              "(no sat_leg3); fy5_after_so_* = kept frames the 5 BW rule would remove in addition; "
              "kept_stance_frac_with_fy5_r = stance share left if it were added.", ""]
    if len(tab):
        readme += ["| split | activity | lab | trials | fixed COP (r) | trials with kept frames | kept stance | "
                   "kept stance (any_so, no sat_leg3) | +5 BW removes (stance frames) | of kept stance | stance left with 5 BW |",
                   "|---|---|---|---|---|---|---|---|---|---|---|"]
        for _, r in tab.iterrows():
            readme.append(f"| {r.split} | {r.activity} | {r.lab} | {r.trials} | {r.trials_fixed_cop_r} | "
                          f"{r.trials_with_kept_frames_r} | {r.kept_stance_frac_r:.1%} | {r.kept_stance_frac_any_so5_r:.1%} | "
                          f"{r.fy5_after_so_stance_frames_r} | {r.fy5_after_so_frac_of_kept_stance_r:.1%} | "
                          f"{r.kept_stance_frac_with_fy5_r:.1%} |")
    open(os.path.join(args.out, "README.md"), "w").write("\n".join(readme) + "\n")
    print(f"{len(dirs) - len(errs)} masks {'checked' if args.dry_run else 'written'}, {len(errs)} errors, "
          f"{runtime:.0f} s; summary in {args.out}")


def peak_table(d, side=None, half_window=11, sojr_subdir=OUT_SUBDIR):
    """Frames around the stance |F_y| peak of one knee with the top muscles at the peak (for the write-up)."""
    ft = frame_table(d, sojr_subdir)
    so = read_sto(os.path.join(d, "jcf_output", SO_FORCES))
    act = read_sto(os.path.join(d, "jcf_output", SO_ACT))
    bw = ft.attrs["bw"]
    if side is None:
        side = max("rl", key=lambda s: (ft[f"fy_{s}"].abs() * ft[f"stance_{s}"]).max())
    st = ft[f"stance_{side}"].to_numpy(bool)
    i = int(np.flatnonzero(st)[np.argmax(ft[f"fy_{side}"].abs().to_numpy()[st])])
    w = ft.iloc[max(0, i - half_window * 4):i + half_window * 4 + 1:4]
    mus = muscles_of(so.columns)
    top = (so.loc[i, mus] / bw).sort_values(ascending=False).head(10)
    top = pd.DataFrame({"force_bw": top, "activation": act.loc[i, top.index].to_numpy(float)})
    reserves = {c: float(so.loc[i, c]) for c in so.columns if c.startswith("reserve_")}
    return side, i, w, top, reserves, ft


def moment_arm_peak(d, side, time, coords=None):
    """Muscle, reserve and net generalized forces at one frame (OpenSim API, read-only).

    Returns {coord: dict(pos, neg, reserve, net, cci)} in N*m: pos/neg are the summed positive and negative muscle
    moments (for knee_angle, positive = flexion), cci = 2*min(pos, |neg|)/(pos + |neg|) the moment co-contraction
    index. Coordinates are set from ik_results.mot (unfiltered; SO used them low-passed at 6 Hz)."""
    import opensim as osm
    try:
        osm.Logger.setLevelString("error")
    except Exception:
        pass
    coords = coords or (f"hip_flexion_{side}", f"knee_angle_{side}", f"ankle_angle_{side}")
    so = read_sto(os.path.join(d, "jcf_output", SO_FORCES))
    ik = read_sto(os.path.join(d, "ik_results.mot"))
    model = osm.Model(os.path.join(d, "jcf_output", "_model_with_reserves.osim"))
    state = model.initSystem()
    cs = model.getCoordinateSet()
    i = int(np.argmin(np.abs(ik["time"].to_numpy(float) - time)))
    j = int(np.argmin(np.abs(so["time"].to_numpy(float) - time)))
    for c in ik.columns[1:]:
        if cs.contains(c):
            cs.get(c).setValue(state, float(ik.loc[i, c]), False)
    model.assemble(state)
    model.realizePosition(state)
    ms = model.getMuscles()
    out = {}
    for c in coords:
        co = cs.get(c)
        pos = neg = 0.0
        for k in range(ms.getSize()):
            mu = ms.get(k)
            if mu.getName() not in so:
                continue
            m = float(so.loc[j, mu.getName()]) * mu.computeMomentArm(state, co)
            pos, neg = (pos + m, neg) if m > 0 else (pos, neg + m)
        res = float(so.loc[j, f"reserve_{c}"]) if f"reserve_{c}" in so else 0.0
        tot = pos - neg
        out[c] = dict(pos=pos, neg=neg, reserve=res, net=pos + neg + res,
                      cci=2.0 * min(pos, -neg) / tot if tot > 0 else 0.0)
    return out


def resolve_so_frame(d, time, lowpass_hz=6.0):
    """Re-solve the static optimization problem of one frame and compare it with the stored SO solution.

    OpenSim SO (use_muscle_physiology, exponent 2) minimises sum(a_i^2) + sum(u_k^2) subject to, for every
    unconstrained coordinate j, sum_i (a_i C_i + P_i) r_ij + sum_k u_k F_k [k acts on j] = tau_j, with C_i the
    rigid-tendon active force at full activation, a_i in [0.01, 1] and u_k the control of a CoordinateActuator of
    optimal force F_k. Here tau_j is what the stored solution produces (muscle forces x moment arms + actuators), so the
    stored solution is feasible by construction and P_i = F_i - a_i C_i. Coordinates: hips, knees, ankles, lumbar
    (the pelvis and arm coordinates have no muscles). Kinematics are ik_results.mot low-passed at lowpass_hz like the
    AnalyzeTool. OpenSim API read-only; nothing is solved by OpenSim or written.

    Returns a dict: cost_so, cost_qp, max equality residual (N*m), and per knee the summed force (BW) of the muscles
    crossing it in both solutions, plus the per-muscle forces (N) of both solutions."""
    import opensim as osm
    from scipy.optimize import minimize
    from scipy.signal import butter, filtfilt
    try:
        osm.Logger.setLevelString("error")
    except Exception:
        pass
    mass, bw, _ = _meta(d)
    so = read_sto(os.path.join(d, "jcf_output", SO_FORCES))
    act = read_sto(os.path.join(d, "jcf_output", SO_ACT))
    ik = read_sto(os.path.join(d, "ik_results.mot"))
    model = osm.Model(os.path.join(d, "jcf_output", "_model_with_reserves.osim"))
    ms = model.getMuscles()
    for k in range(ms.getSize()):
        ms.get(k).set_ignore_tendon_compliance(True)
    state = model.initSystem()
    cs = model.getCoordinateSet()
    t = ik["time"].to_numpy(float)
    dt = float(np.median(np.diff(t)))
    i = int(np.argmin(np.abs(t - time)))
    j = int(np.argmin(np.abs(so["time"].to_numpy(float) - time)))
    filt = len(t) > 15 and lowpass_hz < 0.5 / dt
    if filt:
        bb, aa = butter(4, lowpass_hz / (0.5 / dt))
    for c in ik.columns[1:]:
        if cs.contains(c):
            q = ik[c].to_numpy(float)
            q = filtfilt(bb, aa, q) if filt else q
            cs.get(c).setValue(state, float(q[i]), False)
            cs.get(c).setSpeedValue(state, float(np.gradient(q, t)[i]))
    model.assemble(state)
    mus = [ms.get(k).getName() for k in range(ms.getSize()) if ms.get(k).getName() in so]
    for m in mus:
        ms.get(m).setActivation(state, 1.0)
    model.realizeVelocity(state)
    dofs = [cs.get(k).getName() for k in range(cs.getSize())
            if not cs.get(k).isConstrained(state)
            and cs.get(k).getName().startswith(("hip_", "knee_angle", "ankle_angle", "lumbar_"))]
    R = np.array([[ms.get(m).computeMomentArm(state, cs.get(c)) for c in dofs] for m in mus])
    cap = np.array([ms.get(m).calcInextensibleTendonActiveFiberForce(state, 1.0) for m in mus])
    f_so = so.loc[j, mus].to_numpy(float)
    a_so = act.loc[j, mus].to_numpy(float)
    passive = f_so - a_so * cap
    acts = []
    fs = model.getForceSet()
    for k in range(fs.getSize()):
        ca = osm.CoordinateActuator.safeDownCast(fs.get(k))
        if ca is not None and ca.get_coordinate() in dofs and ca.getName() in so:
            acts.append((ca.getName(), dofs.index(ca.get_coordinate()), float(ca.getOptimalForce())))
    B = np.zeros((len(dofs), len(acts)))
    u_so = np.zeros(len(acts))
    for n, (name, jj, opt) in enumerate(acts):
        B[jj, n] = opt
        u_so[n] = float(so.loc[j, name]) / opt
    A = np.hstack([R.T * cap[None, :], B])
    tau = R.T @ f_so + B @ u_so
    rhs = tau - R.T @ passive
    x0 = np.r_[a_so, u_so]
    bounds = [(0.01, 1.0)] * len(mus) + [(None, None)] * len(acts)
    sol = minimize(lambda x: float(x @ x), x0, jac=lambda x: 2 * x, method="SLSQP", bounds=bounds,
                   constraints=[dict(type="eq", fun=lambda x: A @ x - rhs, jac=lambda x: A)],
                   options=dict(maxiter=500, ftol=1e-10))
    a_qp = sol.x[:len(mus)]
    f_qp = a_qp * cap + passive
    out = dict(folder=os.path.basename(os.path.normpath(d)), time=float(so.time[j]), success=bool(sol.success),
               cost_so=float(x0 @ x0), cost_qp=float(sol.x @ sol.x),
               residual_nm=float(np.abs(A @ sol.x - rhs).max()), min_passive_n=float(passive.min()))
    for s in "rl":
        kn = dofs.index(f"knee_angle_{s}") if f"knee_angle_{s}" in dofs else None
        if kn is None:
            continue
        cross = np.abs(R[:, kn]) > 1e-3
        out[f"knee_{s}_muscle_sum_so_bw"] = float(f_so[cross].sum() / bw)
        out[f"knee_{s}_muscle_sum_qp_bw"] = float(f_qp[cross].sum() / bw)
    out["forces"] = pd.DataFrame({"so_n": f_so, "qp_n": f_qp, "a_so": a_so, "a_qp": a_qp, "cap_n": cap}, index=mus)
    return out


def list_folders(roots, pattern="*"):
    return [d for r in roots for d in sorted(glob.glob(os.path.join(r, pattern))) if os.path.isdir(d)]


def cmd_scan(args):
    dirs = list_folders(args.root, args.pattern)
    print(f"{len(dirs)} folders", flush=True)
    jobs = [(d, args.sojr_subdir) for d in dirs]
    rows = []
    if args.workers > 1:
        with mp.get_context("spawn").Pool(args.workers) as pool:
            for k, r in enumerate(pool.imap_unordered(scan_folder, jobs, chunksize=8)):
                rows += r
                if (k + 1) % 500 == 0:
                    print(f"  {k + 1}/{len(jobs)}", flush=True)
    else:
        for j in jobs:
            rows += scan_folder(j)
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"wrote {args.out}: {len(df)} rows")


def cmd_peak(args):
    side, i, w, top, reserves, ft = peak_table(args.folder, args.side)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 60)
    print(f"{args.folder} knee {side}: peak frame {i}, t = {ft.time[i]:.4f} s, |F_y| = {abs(ft[f'fy_{side}'][i]):.2f} BW")
    print(top.round(3).T.to_string())
    print({k: round(v, 2) for k, v in reserves.items() if abs(v) > 0.5})
    cols = ["time", f"grf_{side}", f"knee_deg_{side}", f"fy_{side}", f"fmed_{side}", f"ext_{side}", f"flex_{side}",
            f"nsat_leg_{side}", f"nsat_knee_{side}", f"res_knee_{side}", f"res_ankle_{side}", f"res_hip_{side}",
            f"jump_{side}", f"coact_{side}", "pelvis_m_pct", "pelvis_f_pct"]
    print(w[cols].round(3).to_string(index=False))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        w[cols].to_csv(args.out, index=False)
        top.to_csv(args.out.replace(".csv", "_top_muscles.csv"))
        print(f"wrote {args.out}")


def _sample_job(job):
    d, side, t, key = job
    try:
        m = moment_arm_peak(d, side, t)
        row = dict(zip(("split", "activity", "folder"), key), side=side, time=t)
        for c, v in m.items():
            name = c.rsplit("_", 1)[0]
            for k, x in v.items():
                row[f"{name}_{k}"] = x
        return row
    except Exception as e:
        return dict(zip(("split", "activity", "folder"), key), side=side, time=t, error=f"{type(e).__name__}: {e}")


def _folder_key(d):
    p = os.path.normpath(d)
    return (os.path.basename(os.path.dirname(os.path.dirname(p))), os.path.basename(os.path.dirname(p)),
            os.path.basename(p))


MOMENT_PREFIXES = ("hip_flexion_", "knee_angle_", "ankle_angle_")
SAMPLE_KEY = ["split", "activity", "folder", "side"]


def cmd_sample(args):
    """Moment-arm split at the stance peak of the given trials (rows of a scan CSV with SO+JR labels).

    --moments-from reuses the moment-arm columns of an earlier sample CSV for every trial whose peak time is unchanged
    (|time - pk_time| < 1e-6 s), so a rescan only needs OpenSim for new or moved peaks; the trial columns always come
    from --trials."""
    tq = pd.read_csv(args.trials, low_memory=False)
    tq = tq[tq.has_sojr.astype(bool) & tq.fy_peak.notna() & (tq.side == args.side) & (tq.activity == args.activity)]
    if args.folders:
        sel = tq[tq.folder.isin(args.folders)]
    else:
        sel = tq.groupby(["split", "lab"], group_keys=False).apply(
            lambda g: g.sample(min(len(g), args.per_lab), random_state=0))
    reused = pd.DataFrame(columns=SAMPLE_KEY + ["time"])
    if args.moments_from:
        old = pd.read_csv(args.moments_from, low_memory=False)
        cols = SAMPLE_KEY + ["time"] + [c for c in old if c.startswith(MOMENT_PREFIXES) and not c.endswith("_nmkg")]
        m = sel[SAMPLE_KEY + ["pk_time"]].merge(old[cols], on=SAMPLE_KEY, how="inner")
        reused = m[np.abs(m.time - m.pk_time) < 1e-6].drop(columns="pk_time")
        print(f"reusing moment arms of {len(reused)} of {len(sel)} trials from {args.moments_from}")
    todo = sel.merge(reused[SAMPLE_KEY], on=SAMPLE_KEY, how="left", indicator=True)
    todo = todo[todo["_merge"] == "left_only"]
    rows = reused.to_dict("records")
    if len(todo):
        dirs = {_folder_key(d): d for d in list_folders(args.root)}
        jobs = [(dirs[k], r.side, r.pk_time, k) for r in todo.itertuples()
                if (k := (r.split, r.activity, r.folder)) in dirs]
        with mp.get_context("spawn").Pool(max(1, args.workers)) as pool:
            rows += pool.map(_sample_job, jobs, chunksize=1)
    df = pd.DataFrame(rows).merge(sel, on=SAMPLE_KEY, how="left")
    for k in ("hip_flexion", "knee_angle", "ankle_angle"):
        if f"{k}_net" in df:
            df[f"{k}_net_nmkg"] = df[f"{k}_net"] / df["mass"]
    df.to_csv(args.out, index=False)
    print(f"wrote {args.out}: {len(df)} rows")


def _resolve_job(job):
    d, t, seed, key = job
    try:
        ft = frame_table(d)
        if t is None:                                  # a random right-stance frame
            st = np.flatnonzero(ft["stance_r"].to_numpy(bool))
            if not len(st):
                return dict(zip(("split", "activity", "folder"), key), error="no right stance")
            t = float(ft.time[np.random.default_rng(seed).choice(st)])
        k = int(np.argmin(np.abs(ft.time.to_numpy(float) - t)))
        o = resolve_so_frame(d, t)
        f = o.pop("forces")
        row = dict(zip(("split", "activity", "folder"), key), lab=lab_of(d), **o, cost_ratio=o["cost_so"] / o["cost_qp"])
        for c in ("fy_r", "grf_r", "nsat_leg_r", "nsat_knee_r", "jump_r", "coact_r", "res_leg_r", "pelvis_m_pct",
                  "ext_r", "flex_r", "knee_deg_r"):
            if c in ft:
                row[c] = float(ft[c][k])
        sat = f.a_so >= SAT_ACT
        row["n_sat_so"] = int(sat.sum())
        row["n_sat_qp"] = int((f.a_qp >= SAT_ACT).sum())
        return row
    except Exception as e:
        return dict(zip(("split", "activity", "folder"), key), error=f"{type(e).__name__}: {e}")


def cmd_resolve(args):
    """Re-solve SO at stance peaks (--sample CSV with pk_time) or at random right-stance frames (--random N per lab)."""
    dirs = {_folder_key(d): d for d in list_folders(args.root)}
    jobs = []
    if args.sample:
        sp = pd.read_csv(args.sample, low_memory=False)
        jobs += [(dirs[k], float(r.pk_time), 0, k) for r in sp.itertuples()
                 if (k := (r.split, r.activity, r.folder)) in dirs]
    if args.random:
        tq = pd.read_csv(args.trials, low_memory=False)
        tq = tq[(tq.side == "r") & (tq.activity == args.activity) & (tq.n_stance > 0)]
        for (_, _), g in tq.groupby(["split", "lab"]):
            for n, r in enumerate(g.sample(min(len(g), args.random), random_state=args.seed).itertuples()):
                k = (r.split, r.activity, r.folder)
                if k in dirs:
                    jobs.append((dirs[k], None, args.seed + n, k))
    with mp.get_context("spawn").Pool(max(1, args.workers)) as pool:
        rows = pool.map(_resolve_job, jobs, chunksize=1)
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"wrote {args.out}: {len(df)} rows")


def frame_label(d, time, side="r", sojr_subdir=OUT_SUBDIR):
    """SO+JR |F_y| (BW) of one knee at the SO frame nearest `time` (NaN without labels); file reads only."""
    p = os.path.join(d, sojr_subdir, JR_FILE)
    if not os.path.exists(p):
        return np.nan
    jr = read_sto(p)
    col = knee_prefix(side) + "fy"
    if col not in jr:
        return np.nan
    k = int(np.argmin(np.abs(jr["time"].to_numpy(float) - time)))
    if abs(float(jr["time"][k]) - time) > 0.011:
        return np.nan
    return float(abs(jr[col][k]) / _meta(d)[1])


def annotate_resolve(rs, tq, dirs, side="r"):
    """Attach the final SO+JR |F_y| at each re-solved frame (fy_<side>, re-read from the labels), that trial's COP
    flags from a scan table, and the re-solve estimate fy_resolved_est = |F_y| - (summed knee-crossing muscle force,
    stored - re-solved): a first-order estimate that assumes |F_y| moves one to one with that sum."""
    rs = rs.copy()
    fy = []
    for r in rs.itertuples():
        d = dirs.get((r.split, r.activity, r.folder))
        fy.append(frame_label(d, float(r.time), side) if d and pd.notna(r.time) else np.nan)
    if f"fy_{side}_at_resolve" not in rs:      # the value the resolve run saw (NaN where labels were missing then)
        rs[f"fy_{side}_at_resolve"] = rs[f"fy_{side}"].abs() if f"fy_{side}" in rs else np.nan
    rs[f"fy_{side}"] = fy
    flags = tq[tq.side == side][["split", "activity", "folder", "fixed_cop", "fixed_cop_trial", "fixed_cop_sens",
                                 "person"] + [c for c in ("cop_speed_r", "cop_speed_l") if c in tq]]
    rs = rs.drop(columns=[c for c in flags.columns[3:] if c in rs]).merge(
        flags, on=["split", "activity", "folder"], how="left")
    for c in ("fixed_cop", "fixed_cop_trial", "fixed_cop_sens"):
        rs[c] = rs[c].fillna(False).astype(bool)
    rs["nonopt"] = rs.cost_ratio > COST_RATIO_NONOPT
    rs["fy_resolved_est"] = rs[f"fy_{side}"] - (rs[f"knee_{side}_muscle_sum_so_bw"] - rs[f"knee_{side}_muscle_sum_qp_bw"])
    return rs


def random_frame_stats(rs, side="r"):
    """Non-optimal rate and |F_y| medians of re-solved random stance frames: all frames, labelled frames, and with
    fixed-COP trials excluded; per (split, lab) and pooled. The |F_y| statistics use labelled frames only."""
    rs = rs[rs.cost_ratio.notna()]
    out = []
    for subset, sub in (("all", rs), ("moving_cop", rs[~rs.fixed_cop.astype(bool)])):
        for key, g in list(sub.groupby(["split", "lab"])) + [(("all", "all"), sub)]:
            lab_ = g[g[f"fy_{side}"].notna()]
            row = dict(subset=subset, split=key[0], lab=key[1], frames=len(g), labelled=len(lab_),
                       from_fixed_cop_trials=int(g.fixed_cop.astype(bool).sum()),
                       nonopt_frac=float(g.nonopt.mean()) if len(g) else np.nan)
            for tag, m in (("opt", ~lab_.nonopt), ("nonopt", lab_.nonopt)):
                row[f"n_{tag}"] = int(m.sum())
                row[f"fy_med_{tag}"] = float(lab_.loc[m, f"fy_{side}"].median()) if m.any() else np.nan
                row[f"fy_est_med_{tag}"] = float(lab_.loc[m, "fy_resolved_est"].median()) if m.any() else np.nan
            out.append(row)
    return pd.DataFrame(out)


def cmd_annotate(args):
    rs = pd.read_csv(args.resolve, low_memory=False)
    tq = pd.read_csv(args.trials, low_memory=False)
    dirs = {_folder_key(d): d for d in list_folders(args.root)}
    rs = annotate_resolve(rs, tq, dirs)
    out = args.out or args.resolve
    rs.to_csv(out, index=False)
    st = random_frame_stats(rs)
    st.to_csv(out.replace(".csv", "_stats.csv"), index=False)
    pd.set_option("display.width", 250)
    print(st.round(3).to_string(index=False))
    print(f"wrote {out} and {out.replace('.csv', '_stats.csv')}")


def classify_sample(sp, rsp):
    """sample_classification table: one row per sampled peak with the peak-frame mechanism, the re-solve result, the
    trial flags (reported, not used alone) and the class of classify_sample_peak."""
    r = rsp[["split", "activity", "folder", "time", "cost_ratio", "success", "knee_r_muscle_sum_so_bw",
             "knee_r_muscle_sum_qp_bw"]].rename(columns={"time": "resolve_time", "success": "resolve_success"})
    df = sp.merge(r, on=["split", "activity", "folder"], how="left")
    df["resolve_success"] = df.resolve_success.astype(object)
    bad = (df.resolve_time - df.pk_time).abs() > 1e-3     # re-solved at another frame than this peak
    df.loc[bad, ["cost_ratio", "resolve_success"]] = np.nan
    df["cop_speed_foot"] = np.where(df.side == "r", df.cop_speed_r, df.cop_speed_l)
    df["class"] = [classify_sample_peak(row) for row in df.to_dict("records")]
    df["class_name"] = df["class"].map(SAMPLE_CLASSES)
    cols = ["split", "lab", "folder", "side", "fy_peak", "fy_mean", "pk_knee_deg", "knee_angle_net_nmkg",
            "knee_angle_cci", "hip_flexion_net_nmkg", "ankle_angle_net_nmkg", "ankle_angle_cci", "pk_ext", "pk_flex",
            "cost_ratio", "resolve_success", "pk_pelvis_m_pct", "cop_speed_foot", "fixed_cop", "fixed_cop_trial",
            "cause", "class", "class_name"]
    return df[[c for c in cols if c in df]]


def cmd_classify(args):
    sp = pd.read_csv(args.sample, low_memory=False)
    rsp = pd.read_csv(args.resolve, low_memory=False)
    df = classify_sample(sp, rsp)
    df.to_csv(args.out, index=False)
    pd.set_option("display.width", 250)
    print(df.round(3).to_string(index=False))
    for tag, g in (("all", df), ("peak > 4 BW", df[df.fy_peak > 4])):
        print(tag, len(g), g["class"].value_counts().sort_index().to_dict(),
              "| fixed_cop flag by class:", g.groupby("class").fixed_cop.sum().astype(int).to_dict())
    print(f"wrote {args.out}")


RULES = {   # name: (level, description)
    "fixed_cop": ("trial", "walking trial whose COP on that knee's foot moves < 0.2 m/s during gait-like stance"),
    "fixed_cop_sens": ("trial", "fixed_cop at 0.1 m/s (sensitivity of the threshold)"),
    "fixed_cop_trial": ("trial", "fixed_cop on either foot (reported, not used for a knee label)"),
    "loader_filters": ("trial", "current train_cnn.load_subject filters on SO+JR labels: resultant > 6 BW or "
                                "|M_x| > 0.15 BW*H anywhere, or flat waveform"),
    "sat_knee": ("frame", "a muscle crossing that knee at activation >= 0.99"),
    "sat_leg3": ("frame", ">= 3 muscles of that leg saturated"),
    "res_leg": ("frame", "a hip/knee/ankle reserve of that leg > 0.1 N*m/kg"),
    "pelvis_m": ("frame", "pelvis residual moment > 5 % of peak GRF x pelvis height"),
    "jump": ("frame", "an activation of that leg changes > 0.5 per 10 ms"),
    "coact": ("frame", "dorsiflexor and plantarflexor of that leg both above activation 0.5"),
    "any_so": ("frame", "sat_knee | res_leg | pelvis_m | jump | coact"),
    "fy5": ("frame", "knee |F_y| > 5 BW"),
}
TRAIN_POOL = ("carter", "moore", "tiziana", "falisse")   # thesis model: walking, --exclude han fregly


def _shares(g, drop=None, col=None):
    """Dropped shares of a group of trial rows: by trials, native frames, native stance frames and the same two
    weighted by the SO time step (time shares, i.e. shares on a common time grid). drop: boolean trial mask;
    col: a per-trial count of dropped frames (n_<rule>, with n_<rule>_stance for stance)."""
    dt = g.dt.to_numpy(float) if "dt" in g else np.ones(len(g))
    n, ns = g.n.to_numpy(float), g.n_stance.to_numpy(float)
    if drop is not None:
        m = np.asarray(drop, bool)
        dn, dns = n * m, ns * m
        out = dict(trial_frac=float(m.mean()) if len(m) else np.nan)
    else:
        dn, dns = g[col].to_numpy(float), g[col + "_stance"].to_numpy(float)
        out = {}
    out.update(frame_frac=float(dn.sum() / max(n.sum(), 1)), stance_frac=float(dns.sum() / max(ns.sum(), 1)),
               time_frac=float((dn * dt).sum() / max((n * dt).sum(), 1e-12)),
               stance_time_frac=float((dns * dt).sum() / max((ns * dt).sum(), 1e-12)))
    return out


def _people(g):
    """People and the share of trials from the most frequent person of a group of trial rows."""
    if not len(g):
        return dict(n_people=0, top_person="", top_person_share=np.nan)
    pid = g.person if "person" in g else g.folder.map(person_of)
    vc = pid.value_counts()
    return dict(n_people=int(len(vc)), top_person=str(vc.index[0]), top_person_share=float(vc.iloc[0] / len(g)))


def _mask_group(masks, key):
    """Rows of the `masks` per-trial table (label_masks/per_trial.csv) in the rule_table group key."""
    act, split, lab = key
    if (split, lab) == ("all", "all"):
        return masks[masks.activity == act]
    if lab == "pool":
        return masks[(masks.activity == act) & (masks.split == split) & masks.lab.isin(TRAIN_POOL)]
    return masks[(masks.activity == act) & (masks.split == split) & (masks.lab == lab)]


def rule_table(tq, side="r", masks=None):
    """Fraction of trials, frames and stance frames each rule drops, per (activity, split, lab), pooled over walking
    and over the thesis training pool (TRAIN_POOL, training split). Every share comes as native frames (frame_frac,
    stance_frac) and time weighted (time_frac, stance_time_frac); the time-weighted ones are what a 100 Hz resample
    sees.

    Frame rules are counted on all trials; fixed_cop on walking trials; loader_filters and fy5 only on trials with
    SO+JR labels and stance on that knee (n_labelled, with the number of those people and the top person's share of
    those trials; n_sojr counts every trial with SO+JR labels). The
    `recommended_approx` row estimates the combination drop fixed_cop trials, then any_so frames, then fy5 frames
    from the per-trial counts alone: it treats the fy5 and any_so frames as independent, which they are not (most
    fy5 frames are SO-flagged), so it overstates the drop (pool: 78.6 % dropped against 75.3 % exact).

    masks (the per-trial table of `masks`, label_masks/per_trial.csv) adds the exact frame-level rows, from the
    per-frame flags; every share is of the group's stance (all its trials; a trial without SO+JR labels counts as
    dropped):
        mask_any_so     frames any of MASK_SO_RULES flags (sat_leg3 included, unlike any_so)
        training_mask   dropped by keep_<side>: fixed-COP trials, then MASK_SO_RULES frames (and non-finite rows);
                        what the *_sojrqc training sets train on (no 5 BW rule)
        recommended     training_mask plus the fy5 frames left after it (the exact recommended combination)"""
    t = tq[tq.side == side].copy()
    t["has_sojr"] = t["has_sojr"].astype(bool)
    if "dt" not in t:
        t["dt"] = 1.0
    rows = []
    walk = t[t.activity == "walking"]
    groups = (list(t.groupby(["activity", "split", "lab"])) + [(("walking", "all", "all"), walk)]
              + [(("walking", "training", "pool"), walk[(walk.split == "training") & walk.lab.isin(TRAIN_POOL)])])
    for key, g in groups:
        lb = g[g.has_sojr & g.fy_peak.notna()] if "fy_peak" in g else g.iloc[:0]
        lab_rows = dict(zip(("activity", "split", "lab"), key), trials=len(g), n_sojr=int(g.has_sojr.sum()),
                        frames=int(g.n.sum()),
                        stance=int(g.n_stance.sum()), seconds=float((g.n * g.dt).sum()),
                        stance_seconds=float((g.n_stance * g.dt).sum()), dt_median=float(g.dt.median()),
                        n_labelled=len(lb), **{k + "_labelled": v for k, v in _people(lb).items()})
        for r, (level, _) in RULES.items():
            row = dict(lab_rows, rule=r, level=level)
            if r.startswith("fixed_cop"):
                col = r if r in g else "fixed_cop"
                row.update(_shares(g, drop=g[col].astype(bool)))
            elif r == "loader_filters":
                if len(lb):
                    row.update(_shares(lb, drop=(lb.fres_max > 6.0) | (lb.mx_absmax > 0.15) | lb.flat.astype(bool)))
            else:
                src = lb if r.startswith("fy") else g
                if len(src):
                    row.update(_shares(src, col=f"n_{r}"))
            rows.append(row)
        keep = ~g.fixed_cop.astype(bool).to_numpy()
        kept = g[keep]
        dt = kept.dt.to_numpy(float)
        st_left = (kept.n_stance - kept.n_any_so_stance).to_numpy(float)
        lk = kept[kept.has_sojr & kept.fy_peak.notna()] if ("fy_peak" in kept and len(kept)) else kept.iloc[:0]
        fy5 = (lk.n_fy5_stance.sum() / max(lk.n_stance.sum(), 1)) if len(lk) else 0.0
        fy5_t = ((lk.n_fy5_stance * lk.dt).sum() / max((lk.n_stance * lk.dt).sum(), 1e-12)) if len(lk) else 0.0
        rows.append(dict(lab_rows, rule="recommended_approx", level="trial+frame", trial_frac=float(1 - keep.mean()),
                         stance_frac=float(1 - st_left.sum() * (1 - fy5) / max(g.n_stance.sum(), 1)),
                         stance_time_frac=float(1 - (st_left * dt).sum() * (1 - fy5_t)
                                                / max((g.n_stance * g.dt).sum(), 1e-12))))
        if masks is not None:
            rows += _mask_rule_rows(lab_rows, g, _mask_group(masks, key), side)
    return pd.DataFrame(rows)


def _mask_rule_rows(lab_rows, g, mg, side):
    """Exact mask_any_so / training_mask / recommended rows of one rule_table group (see rule_table)."""
    n, ns = float(g.n.sum()), float(g.n_stance.sum())
    tot, tot_st = float((g.n * g.dt).sum()), float((g.n_stance * g.dt).sum())
    dt = mg.dt.to_numpy(float)
    c = lambda k: mg[f"{k}_{side}"].to_numpy(float)          # noqa: E731
    kept = {"mask_any_so": (mg.n.to_numpy(float) - c("n_so_any"), c("n_stance") - c("n_so_any_stance")),
            "training_mask": (c("n_keep"), c("n_keep_stance")),
            "recommended": (c("n_keep") - c("n_fy5_kept"), c("n_keep_stance") - c("n_fy5_kept_stance"))}
    out = []
    for rule, (k, ks) in kept.items():
        row = dict(lab_rows, rule=rule, level="frame" if rule == "mask_any_so" else "trial+frame",
                   frame_frac=float(1 - k.sum() / max(n, 1)), stance_frac=float(1 - ks.sum() / max(ns, 1)),
                   time_frac=float(1 - (k * dt).sum() / max(tot, 1e-12)),
                   stance_time_frac=float(1 - (ks * dt).sum() / max(tot_st, 1e-12)))
        if rule != "mask_any_so":   # trials left without a kept frame, of the group's trials
            row["trial_frac"] = float(1 - (k > 0).sum() / max(len(g), 1))
        out.append(row)
    return out


def label_stats(tq, side="r"):
    """Stance |F_y| and F_medial of the SO+JR labels per (activity, split, lab) and the thesis training pool: all
    labelled trials, the trials kept by fixed_cop (moving COP on that foot) and their frames clean of the SO rules.
    *step_peak_* are medians of per-step peaks (median over a trial's full stance runs; compare these with in vivo
    step peaks), fy_peak_* medians of trial peaks (maximum over all steps), fy_mean_* medians of stance means."""
    t = tq[(tq.side == side) & tq.has_sojr.astype(bool) & tq.fy_peak.notna()]
    walk = t[t.activity == "walking"]
    groups = list(t.groupby(["activity", "split", "lab"])) + [
        (("walking", "training", "pool"), walk[(walk.split == "training") & walk.lab.isin(TRAIN_POOL)])]
    out = []
    for key, g in groups:
        kept = g[~g.fixed_cop.astype(bool)]
        out.append(dict(zip(("activity", "split", "lab"), key), n_labelled=len(g),
                        **{k + "_labelled": v for k, v in _people(g).items()},
                        fy_peak_med=g.fy_peak.median(), fy_mean_med=g.fy_mean.median(),
                        fmed_mean_med=g.fmed_mean.median(), frac_peak_gt4=float((g.fy_peak > 4).mean()),
                        n_kept_trials=len(kept), **{k + "_kept": v for k, v in _people(kept).items()},
                        step_peak_med=g.fy_step_peak_med.median() if "fy_step_peak_med" in g else np.nan,
                        n_kept_with_steps=int(kept.fy_step_peak_med.notna().sum()) if "fy_step_peak_med" in g else 0,
                        kept_step_peak_med=kept.fy_step_peak_med.median() if "fy_step_peak_med" in g else np.nan,
                        kept_step_peak_q25=kept.fy_step_peak_med.quantile(0.25) if "fy_step_peak_med" in g else np.nan,
                        kept_step_peak_q75=kept.fy_step_peak_med.quantile(0.75) if "fy_step_peak_med" in g else np.nan,
                        kept_fy_peak_med=kept.fy_peak.median(), kept_fy_peak_q25=kept.fy_peak.quantile(0.25),
                        kept_fy_peak_q75=kept.fy_peak.quantile(0.75), kept_fy_mean_med=kept.fy_mean.median(),
                        kept_frac_peak_gt4=float((kept.fy_peak > 4).mean()) if len(kept) else np.nan,
                        n_kept_with_clean=int(kept.fy_peak_clean_so.notna().sum()),
                        clean_fy_peak_med=kept.fy_peak_clean_so.median(),
                        clean_fy_mean_med=kept.fy_mean_clean_so.median(),
                        clean_fmed_mean_med=kept.fmed_mean_clean_so.median(),
                        clean_frac_peak_gt4=(float((kept.fy_peak_clean_so.dropna() > 4).mean())
                                             if kept.fy_peak_clean_so.notna().any() else np.nan)))
    return pd.DataFrame(out)


def cmd_rules(args):
    tq = pd.read_csv(args.trials, low_memory=False)
    masks = pd.read_csv(args.masks, low_memory=False) if args.masks and os.path.exists(args.masks) else None
    if masks is None:
        print(f"no mask table ({args.masks}); the exact training_mask / recommended rows are left out "
              "(write it with: python -I analysis/thesis/so_label_qc.py masks)")
    rt = rule_table(tq, args.side, masks)
    ls = label_stats(tq, args.side)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    rt.to_csv(args.out, index=False)
    ls.to_csv(args.out.replace(".csv", "_label_stats.csv"), index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 400)
    print(rt.pivot_table(index=["activity", "split", "lab"], columns="rule", values="stance_frac").round(3).to_string())
    print(ls.round(3).to_string(index=False))
    print(f"wrote {args.out}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("scan", help="per-trial QC table over folders (file reads only)")
    a.add_argument("--root", nargs="+", default=list(DEFAULT_ROOTS))
    a.add_argument("--pattern", default="*")
    a.add_argument("--workers", type=int, default=4)
    a.add_argument("--sojr-subdir", default=OUT_SUBDIR)
    a.add_argument("--out", default=os.path.join(REPO, "results", "thesis", "labels_qc", "trial_qc.csv"))
    a.set_defaults(func=cmd_scan)
    b = sub.add_parser("peak", help="frames and muscles around one trial's stance |F_y| peak")
    b.add_argument("folder")
    b.add_argument("--side", choices=("r", "l"))
    b.add_argument("--out")
    b.set_defaults(func=cmd_peak)
    c = sub.add_parser("sample", help="moment-arm split at the peak of sampled trials (needs opensim)")
    c.add_argument("--trials", default=os.path.join(REPO, "results", "thesis", "labels_qc", "trial_qc.csv"))
    c.add_argument("--root", nargs="+", default=list(DEFAULT_ROOTS))
    c.add_argument("--folders", nargs="*")
    c.add_argument("--per-lab", type=int, default=4)
    c.add_argument("--side", choices=("r", "l"), default="r")
    c.add_argument("--activity", default="walking")
    c.add_argument("--workers", type=int, default=4)
    c.add_argument("--moments-from", help="earlier sample CSV whose moment-arm columns are reused at unchanged peaks")
    c.add_argument("--out", default=os.path.join(REPO, "results", "thesis", "labels_qc", "sample_peaks.csv"))
    c.set_defaults(func=cmd_sample)
    e = sub.add_parser("resolve", help="re-solve SO at given or random frames and compare costs (needs opensim)")
    e.add_argument("--sample", help="CSV with split, activity, folder, pk_time (e.g. sample_peaks.csv)")
    e.add_argument("--trials", default=os.path.join(REPO, "results", "thesis", "labels_qc", "trial_qc.csv"))
    e.add_argument("--random", type=int, default=0, help="random right-stance frames per (split, lab)")
    e.add_argument("--activity", default="walking")
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--root", nargs="+", default=list(DEFAULT_ROOTS))
    e.add_argument("--workers", type=int, default=4)
    e.add_argument("--out", default=os.path.join(REPO, "results", "thesis", "labels_qc", "resolve.csv"))
    e.set_defaults(func=cmd_resolve)
    f = sub.add_parser("rules", help="drop fractions of the label-quality rules from a scan CSV")
    f.add_argument("--trials", default=os.path.join(REPO, "results", "thesis", "labels_qc", "trial_qc.csv"))
    f.add_argument("--side", choices=("r", "l"), default="r")
    f.add_argument("--out", default=os.path.join(REPO, "results", "thesis", "labels_qc", "rules.csv"))
    f.add_argument("--masks", default=os.path.join(REPO, "results", "thesis", "labels_qc", "label_masks", "per_trial.csv"),
                   help="per-trial table of `masks`, for the exact training_mask / recommended rows")
    f.set_defaults(func=cmd_rules)
    QC = os.path.join(REPO, "results", "thesis", "labels_qc")
    h = sub.add_parser("annotate", help="attach final SO+JR |F_y| and COP flags to a resolve CSV (file reads only)")
    h.add_argument("--resolve", default=os.path.join(QC, "resolve_random_stance.csv"))
    h.add_argument("--trials", default=os.path.join(QC, "trial_qc.csv"))
    h.add_argument("--root", nargs="+", default=list(DEFAULT_ROOTS))
    h.add_argument("--out", help="default: overwrite --resolve")
    h.set_defaults(func=cmd_annotate)
    mk = sub.add_parser("masks", help="write <trial>/jcf_output_sojr/label_mask.npz (training frame masks)")
    mk.add_argument("--root", nargs="+", default=list(MASK_ROOTS))
    mk.add_argument("--pattern", default="*")
    mk.add_argument("--workers", type=int, default=4)
    mk.add_argument("--sojr-subdir", default=OUT_SUBDIR)
    mk.add_argument("--dry-run", action="store_true", help="summaries only, write no mask")
    mk.add_argument("--out", default=os.path.join(QC, "label_masks"))
    mk.set_defaults(func=cmd_masks)
    k = sub.add_parser("classify", help="class of each sampled peak from its peak-frame mechanism")
    k.add_argument("--sample", default=os.path.join(QC, "sample_peaks.csv"))
    k.add_argument("--resolve", default=os.path.join(QC, "resolve_sample_peaks.csv"))
    k.add_argument("--out", default=os.path.join(QC, "sample_classification.csv"))
    k.set_defaults(func=cmd_classify)
    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Knee label quality control: the stored labels (jcf_output/) against the SO+JR labels (jcf_output_sojr/).

The stored jcf_output/BatchJCF_JointReaction_ReactionLoads.sto were not computed from the static optimization
solution (pipeline/batch_process.py ran SO and JR in one AnalyzeTool with an empty forces_file; see
analysis/thesis/gc_knee_jr.py). gc_knee_jr.py writes the SO+JR labels to jcf_output_sojr/. This script reads both,
together with the SO activations and forces, and reports per trial and per lab x activity x split how the labels and
the SO solutions behave. It only reads files (no OpenSim) and writes nothing inside the trial folders.

Per trial and knee (walker_knee_r and walker_knee_l, on and in the child tibia; the stored files hold the right knee
only), for each label source `stored` and `sojr` (columns <source>_<side>_<metric>):
    stance                      frames with |vertical GRF| of that side > 5 % BW (grf_data.mot, interpolated)
    fy_stance_mean/peak         |F_y| over stance (BW)
    fres_peak, fres_stance_peak peak resultant |F| over all frames / stance (BW)
    mx_min, mx_max, mx_range    M_x over all frames (BW*H), medial-positive (-mx right knee, +mx left knee; the
                                physical sign of analysis/thesis/knee_mx_sign.py; runs before 2026-11-10 used the
                                inverted pipeline sign +mx right, -mx left)
    fmed_stance_mean/peak       F_medial = 0.5*|F_y| + M_x*H/0.045 over stance (BW)
    fmed_gt14_frac, fy_gt14_frac fraction of stance frames with F_medial (|F_y|) above 1.4 BW
    fy_compressive_frac         fraction of stance frames with F_y < 0 (sign check)
    n_res_gt6, n_mx_gt015       frames failing the training label filters (resultant > 6 BW; |M_x| > 0.15 BW*H)
    reject_peak, reject_mx, reject_any
                                the trial-level filters of train_cnn.load_subject (any frame over the limit)
    cv, dr, flat                train_cnn.filter_flat_subjects (CV < 0.25 or range < 0.3 BW of the resultant; the
                                training code applies it to walking only)
    n_nonfinite                 frames with a NaN/inf (or non-numeric text, e.g. OpenSim's padded '-nan') in F or M_x;
                                every other statistic of that knee uses the finite frames only, and the folder gets
                                status 'nonfinite labels' (left out of every table; listed in run_info.json)
    n_full_stance, fy_full_peak, fmed_full_peak
                                complete stance phases (a non-stance frame before and after inside the trial, at least
                                0.25 s) and the median over them of each one's peak |F_y| / F_medial; fragments that hold
                                only part of a stance (most Tiziana and Fregly folders) have none
    spike_n                     frames with |F_res - 5-frame running median| > 1 BW (isolated spikes)
    mx_spike_n                  the same for M_x with 0.05 BW*H
    max_step, n_step_gt05, max_rate
                                largest frame-to-frame change of the resultant (BW), steps above 0.5 BW, and the largest
                                rate (BW/s); at 250 Hz a 0.5 BW step is 125 BW/s, far above walking (about 10-20 BW/s)
SO quality (columns so_*, res_*, vas_*, gas_*):
    so_sat_any_frac             fraction of SO frames with any muscle activation >= 0.999
    so_sat_gt5_frac             fraction with more than 5 saturated muscles (the William-trajectory criterion)
    so_n_sat_muscles            muscles saturated in at least one frame (of so_n_muscles); so_sat_max_per_frame
    so_sat_top                  the three muscles saturated in the most frames (name:frames)
    so_sat_any_stance_r_frac    saturation fraction over right-stance frames
    res_<coord>_peak_nmkg       peak |reserve torque| of the hip flexion/adduction, knee and ankle reserves (N*m/kg);
                                a typical peak net joint moment in walking is about 1.0 (hip flexion), 0.5-0.7 (knee),
                                1.4-1.6 (ankle) N*m/kg, so a reserve near those values carries the whole joint
    res_knee_<side>_stance_mean_nmkg, res_knee_<side>_peak_bwh
    res_force_share             sum |reserve| / (sum |reserve| + sum muscle force) over all frames, the same ratio as
                                the early-pipeline '~1.7 % of total force' note (docs/project_summary.md; mixes N and N*m)
    resid_force_pct             peak |pelvis translational reserves| / peak |total GRF| * 100 (Hicks et al. 2015
                                suggest < 5 %)
    resid_moment_pct            peak |pelvis rotational reserves| / (mean COM height * peak |GRF|) * 100 (Hicks: < 1 %)
    vas_<side>_max_bw, gas_<side>_max_bw   peak summed vasti (vaslat+vasmed+vasint) and gastrocnemius force (BW)
Centre of pressure (grf_data.mot with com_features.npz and foot_point_0.130_0.000_-0.010.npz), per foot, on frames
with |GRF_y| > 20 % BW: cop_<side>_fwd_p5 / fwd_p95 / lat_med (m along the foot axes from the calcn origin), dy_med
(CoP height minus calcn origin height, m) and off_frac (fraction of those frames with the CoP outside -6..+30 cm
forward or more than 8 cm sideways: the GRF is applied off the foot).
Speed (com_features.npz): speed_ms = median horizontal speed of the COM relative to the single-stance calcaneus
(walking speed on a treadmill and overground alike); com_speed_ms = median horizontal COM speed in the lab frame.
ds_frac = fraction of SO frames with both feet loaded (> 5 % BW): near 1 for standing and for the quasi-static Han
'walking' trials, about 0.2-0.3 in walking.
Fregly: instrumented_side = the implant knee (fregly_1GC JW right, fregly_2GC DM right, fregly_3GC SC left;
results/thesis/gc/EXPLORE_NOTES.md section 2); fregly_5GC is unknown.

Relabel coverage: every folder with SO output under the roots gets a relabel state (relabel_coverage.csv):
    labelled                    jcf_output_sojr/ holds the JR file
    failed                      gc_knee_jr wrote jcf_output_sojr/_jr_setup.xml before the last summary
                                (results/thesis/gc/knee_jr/relabel_all.csv) was written, but no JR file
    pending                     _jr_setup.xml newer than that summary and no JR file: a run that has not written its
                                summary yet (in progress or interrupted)
    not run / no setup          no _jr_setup.xml / no jcf_output/_analyze_setup.xml
and setup_resolves = whether gc_knee_jr.jr_only_setup now finds every input file of the folder's setup here (it
rewrites the labelling Mac's /Users/... paths to the trial folder). run_info.json reports these counts per lab and the
status counts of relabel_all.csv per lab. That CSV is counted, not joined: it has folder names without paths, and 365
names occur under two roots.

Fregly measured check (needs results/thesis/gc/pilot_AB_sojr/frames.csv.gz from analysis/thesis/grand_challenge.py,
which aligns the GC measured implant force to each segment frame by an exact index): SO+JR |F_y| of the instrumented
knee against the measured total axial force on the instrumented-limb stance frames, per segment and per stance window
(early 5-45 %, late 55-95 % of stance, covered >= 90 %).

Outputs (default results/thesis/labels_qc/):
    per_trial.csv               one row per labelled folder
    relabel_coverage.csv        one row per folder with SO output: relabel state, setup_resolves, SO frames
    by_group.csv                long table: split, activity, lab (and lab = ALL), metric, n, median, q25, q75, p95, mean
    headline.csv                per knee: stance-peak |F_y|, F_medial peak and mean: median [IQR] and p95, stored vs SO+JR;
                                with n_people, the largest person's share and <metric>_by_person = median of the
                                per-person medians [smallest, largest person median] (each person weighted equally)
    by_person.csv               per person: trial count, medians of the main metrics, filter rejections
    correlates.csv              Spearman rho of the SO+JR right-knee stance-peak |F_y| with speed, body size and the
                                SO quality indicators, per group with >= 20 trials
    filter_rejections.csv       share of trials the training label filters reject, stored vs SO+JR, per group (pooled
                                over trials, and *_pct_pw = mean over people)
    fregly_instrumented.csv     the instrumented knee of each Fregly folder
    fregly_vs_measured.csv      the measured check (kind = segment / window)
    fig_peaks_by_lab.png, fig_peak_fy_vs_speed.png, fig_filter_rejections.png
    run_info.json               folder counts, relabel coverage and summary status, non-finite folders, git commit

Usage (repo root; reads files only):
    python -I analysis/thesis/label_qc.py --workers 4                  # every folder that has jcf_output_sojr/
    python -I analysis/thesis/label_qc.py --workers 20 --root jcf/full_duration/testing/walking
    python -I analysis/thesis/label_qc.py --from-csv results/thesis/labels_qc/per_trial.csv   # tables and figures only
"""
import os as _os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")   # one thread per worker
import argparse
import glob
import json
import multiprocessing as mp
import os
import re
import subprocess
import sys
import time

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

JR_FILE = "BatchJCF_JointReaction_ReactionLoads.sto"
SO_FORCE = "BatchJCF_StaticOptimization_force.sto"
SO_ACT = "BatchJCF_StaticOptimization_activation.sto"
SOURCES = (("stored", "jcf_output"), ("sojr", "jcf_output_sojr"))
SIDES = ("r", "l")
MX_SIGN = {"r": -1.0, "l": 1.0}       # medial-positive M_x = MX_SIGN * raw mx (knee_mx_sign.medial_mx_sign, physical)
G = 9.81
CONTACT_BW = 0.05                      # stance: |GRF_y| > 5 % BW
PHI, D_M, LIMIT_BW = 0.5, 0.045, 1.4   # F_medial = PHI*|F_y| + M_x*H/D_M; the optimizer limit
MAX_PEAK_BW, MAX_MX_BWH = 6.0, 0.15    # train_cnn.load_subject (train_mlp --max-peak 6.0, max_moment_bwh 0.15)
FLAT_CV, FLAT_DR = 0.25, 0.3           # train_cnn.filter_flat_subjects
SAT_ACT = 0.999                        # pipeline/label_trajectory_csv.py saturation threshold
SPIKE_BW, SPIKE_MX_BWH, STEP_BW = 1.0, 0.05, 0.5
MEDIAN_WIN = 5
FOOT_POINT = (0.13, 0.0, -0.01)        # pipeline/compute_com_features.py --foot-point 0.13 0 -0.01 (calcn frame, m)
COP_MIN_BW = 0.2                       # CoP statistics on frames with |GRF_y| > 20 % BW (CoP is noisy at light load)
COP_FWD, COP_LAT = (-0.06, 0.30), 0.08  # 'on the foot': -6 to +30 cm along the foot from the calcn origin, |lateral| <= 8 cm
# The 40 Millard muscles per leg of the AddBiomechanics Rajagopal model; the other actuators in the SO files are
# coordinate actuators (lumbar, arms) and the reserve_* actuators added before SO.
MUSCLES = ("addbrev", "addlong", "addmagDist", "addmagIsch", "addmagMid", "addmagProx", "bflh", "bfsh", "edl", "ehl",
           "fdl", "fhl", "gaslat", "gasmed", "glmax1", "glmax2", "glmax3", "glmed1", "glmed2", "glmed3", "glmin1",
           "glmin2", "glmin3", "grac", "iliacus", "perbrev", "perlong", "piri", "psoas", "recfem", "sart", "semimem",
           "semiten", "soleus", "tfl", "tibant", "tibpost", "vasint", "vaslat", "vasmed")
VASTI = ("vaslat", "vasmed", "vasint")
GASTROC = ("gasmed", "gaslat")
RESERVE_JOINTS = (("hipflex", "hip_flexion"), ("hipadd", "hip_adduction"), ("knee", "knee_angle"),
                  ("ankle", "ankle_angle"))
PELVIS_T = ("pelvis_tx", "pelvis_ty", "pelvis_tz")
PELVIS_R = ("pelvis_tilt", "pelvis_list", "pelvis_rotation")
DEFAULT_ROOTS = tuple(os.path.join(REPO, "jcf", *p) for p in (
    ("full_duration", "training", "walking"), ("full_duration", "training", "static"),
    ("full_duration", "testing", "walking"), ("full_duration", "testing", "static"), ("external",)))
DEFAULT_OUT = os.path.join(REPO, "results", "thesis", "labels_qc")
RELABEL_SUMMARY = os.path.join(REPO, "results", "thesis", "gc", "knee_jr", "relabel_all.csv")
SO_SETUP = "_analyze_setup.xml"        # jcf_output/: the labelling AnalyzeTool setup
SOJR_SETUP = "_jr_setup.xml"           # jcf_output_sojr/: written by gc_knee_jr just before it runs OpenSim
GC_FRAMES = os.path.join(REPO, "results", "thesis", "gc", "pilot_AB_sojr", "frames.csv.gz")
FREGLY_INSTRUMENTED = {"fregly_1GC": "r", "fregly_2GC": "r", "fregly_3GC": "l"}   # JW, DM, SC (EXPLORE_NOTES section 2)
MIN_FULL_STANCE_S = 0.25
STANCE_WINDOWS = (("early", 5.0, 45.0), ("late", 55.0, 95.0))   # % of stance, as grand_challenge.py
GROUP = ["split", "activity", "lab"]
LAB_ORDER = ("carter", "moore", "tiziana", "falisse", "fregly", "han", "william")


# ─── Readers ───────────────────────────────────────────────────────────────────

def read_sto(path):
    """OpenSim .sto/.mot as a DataFrame (header up to 'endheader'); tab-separated, any whitespace as a fallback.

    Every column is numeric: OpenSim pads values to a fixed width, so a failed frame is written as '            -nan',
    which pandas leaves as text; text is stripped and parsed, and what is still not a number becomes NaN."""
    with open(path) as f:
        for i, line in enumerate(f):
            if line.strip().lower() == "endheader":
                break
        else:
            raise ValueError(f"{path}: no endheader")
    df = pd.read_csv(path, sep="\t", skiprows=i + 1)
    if df.shape[1] == 1:
        df = pd.read_csv(path, sep=r"\s+", skiprows=i + 1)
    df.columns = [c.strip() for c in df.columns]
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = pd.to_numeric(df[c].astype(str).str.strip(), errors="coerce")
    return df


def sto_nrows(path):
    """nRows from an OpenSim header (NaN without one)."""
    try:
        with open(path) as f:
            for line in f:
                if line.lower().startswith("nrows="):
                    return int(line.split("=", 1)[1])
                if line.strip().lower() == "endheader":
                    break
    except (OSError, ValueError):
        pass
    return np.nan


def person_id(path):
    """Person behind a trial folder: strips _tNN, _rNN and _splitN (as tools/evaluate_fmedial.person_id, which
    imports torch)."""
    s = re.sub(r"_t\d+(_r\d+)?$", "", os.path.basename(os.path.normpath(path)))
    return re.sub(r"_split\d+$", "", s)


def folder_info(d):
    """split / activity / lab / person of a trial folder from its path."""
    parent, grand = os.path.basename(os.path.dirname(d)), os.path.basename(os.path.dirname(os.path.dirname(d)))
    if parent in ("walking", "static") and grand in ("training", "testing"):
        split, activity = grand, parent
    else:
        split, activity = "external", "walking"
    name = os.path.basename(os.path.normpath(d))
    person = person_id(d)
    out = dict(folder=name, split=split, activity=activity, lab=name.split("_")[0], person=person)
    if person in FREGLY_INSTRUMENTED:
        out["instrumented_side"] = FREGLY_INSTRUMENTED[person]
    return out


def find_folders(roots, pattern="*", need_sojr=True):
    """Trial folders under the roots that have SO output (and, with need_sojr, SO+JR labels), sorted per root."""
    out = []
    for r in roots:
        for d in sorted(glob.glob(os.path.join(r, pattern))):
            if not os.path.isdir(d) or not os.path.exists(os.path.join(d, "jcf_output", SO_FORCE)):
                continue
            if need_sojr and not os.path.exists(os.path.join(d, "jcf_output_sojr", JR_FILE)):
                continue
            out.append(d)
    return out


# ─── Per-signal statistics (pure functions) ─────────────────────────────────────

def knee_loads(jr, side, bw, H):
    """Knee loads of one side from a JR table: t, fx, fy, fz (BW), mx (BW*H, medial-positive); None without them."""
    pre = f"walker_knee_{side}_on_tibia_{side}_in_tibia_{side}_"
    if jr is None or pre + "fy" not in jr:
        return None
    k = {c: jr[pre + c].to_numpy(float) / bw for c in ("fx", "fy", "fz")}
    k["mx"] = MX_SIGN[side] * jr[pre + "mx"].to_numpy(float) / (bw * H)
    k["t"] = jr["time"].to_numpy(float)
    return k


def stance_mask(t, grf, side, bw, thr=CONTACT_BW):
    """|vertical GRF| of one foot above thr*BW, interpolated onto times t."""
    col = f"calcn_{side}_force_vy"
    if col not in grf:
        return np.zeros(len(t), bool)
    return np.interp(t, grf["time"].to_numpy(float), np.abs(grf[col].to_numpy(float))) > thr * bw


def running_median(x, w=MEDIAN_WIN):
    """Centred running median with edge replication (same length as x)."""
    x = np.asarray(x, float)
    if len(x) < w:
        return np.full_like(x, np.median(x)) if len(x) else x
    h = w // 2
    return np.median(np.lib.stride_tricks.sliding_window_view(np.pad(x, h, mode="edge"), w), axis=1)


def spike_stats(res, mx, t):
    """Spike and discontinuity indicators of one knee's resultant force (BW) and M_x (BW*H)."""
    out = dict(spike_n=int(np.sum(np.abs(res - running_median(res)) > SPIKE_BW)),
               mx_spike_n=int(np.sum(np.abs(mx - running_median(mx)) > SPIKE_MX_BWH)))
    if len(res) < 2:
        return dict(out, max_step=np.nan, n_step_gt05=0, max_rate=np.nan)
    step, dt = np.abs(np.diff(res)), np.diff(t)
    ok = dt > 0
    return dict(out, max_step=float(step.max()), n_step_gt05=int(np.sum(step > STEP_BW)),
                max_rate=float((step[ok] / dt[ok]).max()) if ok.any() else np.nan)


def _smean(x, m):
    return float(np.mean(x[m])) if m.any() else np.nan


def _smax(x, m):
    return float(np.max(x[m])) if m.any() else np.nan


def full_stances(stance, t, min_s=MIN_FULL_STANCE_S):
    """(start, stop) index pairs (stop exclusive) of the complete stance phases: runs of stance frames with a
    non-stance frame before and after them inside the trial, lasting at least min_s seconds."""
    s = np.asarray(stance, bool).astype(int)
    if not len(s):
        return []
    edge = np.diff(np.r_[0, s, 0])
    starts, stops = np.flatnonzero(edge == 1), np.flatnonzero(edge == -1)
    t = np.asarray(t, float)
    return [(a, b) for a, b in zip(starts, stops) if a > 0 and b < len(s) and t[b - 1] - t[a] >= min_s]


def knee_stats(k, stance, H, phi=PHI, d_m=D_M, limit=LIMIT_BW):
    """Statistics of one knee's labels (dict from knee_loads) with its stance mask; see the module docstring.

    Frames with a non-finite F or M_x are counted (n_nonfinite) and left out of every other statistic."""
    stance = np.asarray(stance, bool)
    fin = np.isfinite(k["fx"]) & np.isfinite(k["fy"]) & np.isfinite(k["fz"]) & np.isfinite(k["mx"])
    n_nonfinite = int((~fin).sum())
    if not fin.any():
        return dict(n=len(fin), n_nonfinite=n_nonfinite, stance_n=int(stance.sum()))
    fx, fy, fz, mx, t = (np.asarray(k[c], float)[fin] for c in ("fx", "fy", "fz", "mx", "t"))
    stance = stance[fin]
    res = np.sqrt(fx ** 2 + fy ** 2 + fz ** 2)
    afy = np.abs(fy)
    fmed = phi * afy + mx * H / d_m
    cv = float(res.std() / (res.mean() + 1e-12))
    dr = float(res.max() - res.min())
    n_res, n_mx = int(np.sum(res > MAX_PEAK_BW)), int(np.sum(np.abs(mx) > MAX_MX_BWH))
    full = full_stances(stance, t)
    out = dict(n=len(fin), n_nonfinite=n_nonfinite, stance_n=int(stance.sum()),
               fy_stance_mean=_smean(afy, stance), fy_stance_peak=_smax(afy, stance),
               n_full_stance=len(full),
               fy_full_peak=float(np.median([afy[a:b].max() for a, b in full])) if full else np.nan,
               fmed_full_peak=float(np.median([fmed[a:b].max() for a, b in full])) if full else np.nan,
               fres_peak=float(res.max()), fres_stance_peak=_smax(res, stance),
               mx_min=float(mx.min()), mx_max=float(mx.max()), mx_range=float(mx.max() - mx.min()),
               fmed_stance_mean=_smean(fmed, stance), fmed_stance_peak=_smax(fmed, stance),
               fmed_gt14_frac=_smean(fmed > limit, stance), fy_gt14_frac=_smean(afy > limit, stance),
               fy_compressive_frac=_smean(fy < 0, stance),
               n_res_gt6=n_res, n_mx_gt015=n_mx,
               reject_peak=bool(n_res > 0), reject_mx=bool(n_mx > 0), reject_any=bool(n_res > 0 or n_mx > 0),
               cv=cv, dr=dr, flat=bool(cv < FLAT_CV or dr < FLAT_DR))
    out.update(spike_stats(res, mx, t))
    return out


def muscle_columns(columns):
    """Muscle columns of an SO table: the Rajagopal lower-limb muscles of both legs."""
    names = set(MUSCLES)
    return [c for c in columns if c[-2:] in ("_r", "_l") and c[:-2] in names]


def saturation_stats(act, stance_r=None, thr=SAT_ACT):
    """Activation saturation (any muscle >= thr) of an SO activation table."""
    mus = muscle_columns(act.columns)
    if not mus:
        return dict(so_n_muscles=0)
    sat = act[mus].to_numpy(float) >= thr
    per_frame = sat.sum(axis=1)
    per_muscle = sat.sum(axis=0)
    top = np.argsort(-per_muscle, kind="stable")[:3]
    out = dict(so_n=len(act), so_n_muscles=len(mus), so_sat_any_frac=float(np.mean(per_frame > 0)),
               so_sat_gt5_frac=float(np.mean(per_frame > 5)), so_n_sat_muscles=int(np.sum(per_muscle > 0)),
               so_sat_max_per_frame=int(per_frame.max()) if len(per_frame) else 0,
               so_sat_top=";".join(f"{mus[i]}:{int(per_muscle[i])}" for i in top if per_muscle[i] > 0))
    if stance_r is not None:
        out["so_sat_any_stance_r_frac"] = _smean(per_frame > 0, np.asarray(stance_r, bool))
    return out


def reserve_stats(frc, mass, H, stances=None, grf=None, com_height=np.nan):
    """Reserve and residual actuators, and vasti / gastrocnemius forces, of an SO force table."""
    bw = mass * G
    out = {}
    stances = stances or {}
    for key, coord in RESERVE_JOINTS:
        for side in SIDES:
            c = f"reserve_{coord}_{side}"
            if c not in frc:
                continue
            a = np.abs(frc[c].to_numpy(float))
            out[f"res_{key}_{side}_peak_nmkg"] = float(a.max() / mass)
            if key == "knee":
                out[f"res_knee_{side}_peak_bwh"] = float(a.max() / (bw * H))
                if side in stances:
                    out[f"res_knee_{side}_stance_mean_nmkg"] = _smean(a, np.asarray(stances[side], bool)) / mass
    res_cols = [c for c in frc.columns if c.startswith("reserve_")]
    mus = muscle_columns(frc.columns)
    if res_cols and mus:
        r_sum = float(np.abs(frc[res_cols].to_numpy(float)).sum())
        m_sum = float(np.abs(frc[mus].to_numpy(float)).sum())
        out["res_force_share"] = r_sum / (r_sum + m_sum) if r_sum + m_sum > 0 else np.nan
    if grf is not None:
        g = np.zeros((len(frc), 3))
        t = frc["time"].to_numpy(float)
        for side in SIDES:
            cols = [f"calcn_{side}_force_v{a}" for a in "xyz"]
            if all(c in grf for c in cols):
                g += np.column_stack([np.interp(t, grf["time"].to_numpy(float), grf[c].to_numpy(float)) for c in cols])
        gmax = float(np.sqrt((g ** 2).sum(axis=1)).max()) if len(g) else np.nan
        for name, coords, scale in (("resid_force_pct", PELVIS_T, gmax), ("resid_moment_pct", PELVIS_R, gmax * com_height)):
            cols = [f"reserve_{c}" for c in coords]
            if all(c in frc for c in cols) and scale and np.isfinite(scale) and scale > 0:
                v = np.sqrt((frc[cols].to_numpy(float) ** 2).sum(axis=1)).max()
                out[name] = float(100.0 * v / scale)
    for name, group in (("vas", VASTI), ("gas", GASTROC)):
        for side in SIDES:
            cols = [f"{m}_{side}" for m in group if f"{m}_{side}" in frc]
            if cols:
                out[f"{name}_{side}_max_bw"] = float(frc[cols].to_numpy(float).sum(axis=1).max() / bw)
    return out


def foot_axes(calcn, toe, point=FOOT_POINT):
    """Horizontal forward (calcn x) and lateral unit vectors of a foot from the calcn origin and the cached position of
    `point` (in the calcn frame, x forward, y up, z to the right). Assumes a level foot: the horizontal projection of
    point = (px, 0, pz) is px*X + pz*Z with Z = X x up, which is inverted for X."""
    v = np.asarray(toe, float) - np.asarray(calcn, float)
    px, pz = point[0], point[2]
    a = px * v[:, 0] + pz * v[:, 2]                           # (vx, vz) = M (Xx, Xz), M = [[px, -pz], [pz, px]];
    b = -pz * v[:, 0] + px * v[:, 2]                          # M^-1 up to a positive scale
    fwd = np.column_stack([a, np.zeros_like(a), b])
    fwd /= np.maximum(np.linalg.norm(fwd, axis=1, keepdims=True), 1e-12)
    right = np.cross(fwd, [0.0, 1.0, 0.0])                    # calcn z: to the right of the body
    return fwd, right


def cop_stats(grf, com, toe, side, bw, min_bw=COP_MIN_BW):
    """Centre of pressure of one foot relative to its calcaneus origin on loaded frames (|GRF_y| > min_bw BW), on the
    GRF rows: forward p5 / p95 and lateral median along the foot axes (m), the median height of the CoP above the calcn
    origin (m), and the fraction of loaded frames with the CoP off the foot (forward outside COP_FWD or |lateral| >
    COP_LAT). A CoP off the foot puts the external load's moment arm, and with it the joint moments SO has to
    balance, in the wrong place."""
    cols = [f"calcn_{side}_force_p{a}" for a in "xyz"]
    if not all(c in grf for c in cols) or f"calcn_{side}_force_vy" not in grf:
        return {}
    t = grf["time"].to_numpy(float)
    on = np.abs(grf[f"calcn_{side}_force_vy"].to_numpy(float)) > min_bw * bw
    if not on.any():
        return {}
    tc = np.asarray(com["time"], float)
    cal = np.column_stack([np.interp(t[on], tc, com[f"calcn_{side}_pos"][:, k]) for k in range(3)])
    tt = np.asarray(toe["time"], float)
    tp = np.column_stack([np.interp(t[on], tt, toe[f"calcn_{side}_pos"][:, k]) for k in range(3)])
    fwd, right = foot_axes(cal, tp)
    d = grf[cols].to_numpy(float)[on] - cal
    f = np.sum(d * fwd, axis=1)
    lat = np.sum(d * right, axis=1) * (1.0 if side == "r" else -1.0)
    off = (f < COP_FWD[0]) | (f > COP_FWD[1]) | (np.abs(lat) > COP_LAT)
    p = f"cop_{side}_"
    return {p + "fwd_p5": float(np.percentile(f, 5)), p + "fwd_p95": float(np.percentile(f, 95)),
            p + "lat_med": float(np.median(lat)), p + "dy_med": float(np.median(d[:, 1])),
            p + "off_frac": float(np.mean(off))}


def walking_speed(com, grf, bw):
    """(speed relative to the single-stance calcaneus, horizontal COM speed), medians in m/s, from com_features.

    On a treadmill the COM stays put and the stance foot moves backward at belt speed; overground the stance foot
    stays put and the COM moves; |v_COM - v_stance foot| is the walking speed in both cases."""
    t = np.asarray(com["time"], float)
    v = np.asarray(com["com_vel"], float)[:, [0, 2]]           # ground frame, y up
    com_speed = float(np.median(np.linalg.norm(v, axis=1))) if len(v) else np.nan
    if len(t) < 3:
        return np.nan, com_speed
    st = {s: stance_mask(t, grf, s, bw) for s in SIDES}
    rel = []
    for s, o in (("r", "l"), ("l", "r")):
        single = st[s] & ~st[o]
        if single.any():
            vf = np.gradient(np.asarray(com[f"calcn_{s}_pos"], float)[:, [0, 2]], t, axis=0)
            rel.append(np.linalg.norm(v[single] - vf[single], axis=1))
    speed = float(np.median(np.concatenate(rel))) if rel else np.nan
    return speed, com_speed


# ─── One trial ──────────────────────────────────────────────────────────────────

def trial_row(d):
    """All per-trial statistics of one folder (a row of per_trial.csv); never raises."""
    row = folder_info(d)
    try:
        meta = json.load(open(os.path.join(d, "metadata.json")))
        mass, H = float(meta["mass_kg"]), float(meta.get("height_m") or 1.7)
        bw = mass * G
        row.update(mass_kg=mass, height_m=H)
        grf = read_sto(os.path.join(d, "grf_data.mot"))
        for src, sub in SOURCES:
            p = os.path.join(d, sub, JR_FILE)
            jr = read_sto(p) if os.path.exists(p) else None
            for side in SIDES:
                k = knee_loads(jr, side, bw, H)
                if k is None:
                    continue
                st = knee_stats(k, stance_mask(k["t"], grf, side, bw), H)
                row.update({f"{src}_{side}_{m}": v for m, v in st.items()})
            if jr is not None:
                t = jr["time"].to_numpy(float)
                row[f"{src}_t0"], row[f"{src}_t1"] = float(t[0]), float(t[-1])
                row[f"{src}_fs_hz"] = float((len(t) - 1) / (t[-1] - t[0])) if len(t) > 1 and t[-1] > t[0] else np.nan
        com, com_h = None, np.nan
        cp = os.path.join(d, "com_features.npz")
        if os.path.exists(cp):
            with np.load(cp) as z:
                com = {k: z[k] for k in z.files}
            com_h = float(np.mean(com["com_pos"][:, 1]))
            row["speed_ms"], row["com_speed_ms"] = walking_speed(com, grf, bw)
            row["com_height_m"] = com_h
            from pipeline.compute_com_features import foot_point_filename
            tp = os.path.join(d, foot_point_filename(FOOT_POINT))
            if os.path.exists(tp):
                with np.load(tp) as z:
                    toe = {k: z[k] for k in z.files}
                for side in SIDES:
                    row.update(cop_stats(grf, com, toe, side, bw))
        ap = os.path.join(d, "jcf_output", SO_ACT)
        if os.path.exists(ap):
            act = read_sto(ap)
            row.update(saturation_stats(act, stance_mask(act["time"].to_numpy(float), grf, "r", bw)))
        fp = os.path.join(d, "jcf_output", SO_FORCE)
        if os.path.exists(fp):
            frc = read_sto(fp)
            tf = frc["time"].to_numpy(float)
            st = {s: stance_mask(tf, grf, s, bw) for s in SIDES}
            row.update(reserve_stats(frc, mass, H, st, grf, com_h))
            row["ds_frac"] = float(np.mean(st["r"] & st["l"])) if len(tf) else np.nan
            row["so_n_nonfinite"] = int((~np.isfinite(frc.to_numpy(float))).any(axis=1).sum())
        bad = {src: sum(int(row.get(f"{src}_{s}_n_nonfinite") or 0) for s in SIDES) for src, _ in SOURCES}
        row["status"] = "nonfinite labels" if any(bad.values()) else "ok"
    except Exception as e:                      # one bad folder must not stop the run
        row["status"] = f"failed: {type(e).__name__}: {e}"
    return row


def collect(dirs, workers=4, progress=True):
    if workers > 1 and len(dirs) > 1:
        rows = []
        with mp.get_context("fork").Pool(min(workers, len(dirs))) as pool:
            for i, r in enumerate(pool.imap_unordered(trial_row, dirs, chunksize=8), 1):
                rows.append(r)
                if progress and (i % 500 == 0 or i == len(dirs)):
                    print(f"  {i}/{len(dirs)} folders", flush=True)
    else:
        rows = [trial_row(d) for d in dirs]
    return pd.DataFrame(rows).sort_values(["split", "activity", "lab", "folder"]).reset_index(drop=True)


# ─── Relabel coverage ───────────────────────────────────────────────────────────

def _mtime(path):
    try:
        return os.path.getmtime(path)
    except OSError:
        return np.nan


def relabel_state(d, summary_mtime=np.nan):
    """SO+JR relabel state of one folder with SO output (a row of relabel_coverage.csv; see the module docstring).

    summary_mtime = modification time of gc_knee_jr's last summary CSV: a _jr_setup.xml without labels written before it
    is a failure of that run, one written after it belongs to a run that has not finished (or was interrupted)."""
    row = folder_info(d)
    row["path"] = os.path.relpath(d, REPO)
    src = os.path.join(d, "jcf_output", SO_SETUP)
    forces = os.path.join(d, "jcf_output", SO_FORCE)
    row["so_frames"] = sto_nrows(forces)
    jr = os.path.join(d, "jcf_output_sojr", JR_FILE)
    setup_t = _mtime(os.path.join(d, "jcf_output_sojr", SOJR_SETUP))
    if os.path.exists(jr):
        row["relabel"] = "labelled"
        row["labelled_after_summary"] = bool(_mtime(jr) > summary_mtime)
    elif not os.path.exists(src):
        row["relabel"] = "no setup"
    elif np.isnan(setup_t):
        row["relabel"] = "not run"
    else:
        row["relabel"] = "pending" if np.isnan(summary_mtime) or setup_t > summary_mtime else "failed"
    if row["relabel"] in ("failed", "pending", "not run"):
        try:                                     # the setup gc_knee_jr would write now (nothing is written here)
            from analysis.thesis.gc_knee_jr import jr_only_setup
            jr_only_setup(src, forces, os.path.join(d, "jcf_output_sojr"))
            row["setup_resolves"] = True
        except Exception as e:
            row["setup_resolves"] = False
            row["setup_error"] = f"{type(e).__name__}: {e}"[:300]
    return row


def relabel_coverage(dirs, workers=4, summary=RELABEL_SUMMARY):
    """relabel_state of every folder (dirs = folders with SO output)."""
    from functools import partial
    fn = partial(relabel_state, summary_mtime=_mtime(summary))
    if workers > 1 and len(dirs) > 1:
        with mp.get_context("fork").Pool(min(workers, len(dirs))) as pool:
            rows = pool.map(fn, dirs, chunksize=64)
    else:
        rows = [fn(d) for d in dirs]
    cols = ["path", "folder", "split", "activity", "lab", "person", "relabel", "setup_resolves", "so_frames"]
    df = pd.DataFrame(rows)
    df = df.reindex(columns=cols + [c for c in df.columns if c not in cols])
    return df.sort_values(["split", "activity", "lab", "folder"]).reset_index(drop=True)


def failure_reason(status):
    """Short category of a gc_knee_jr 'failed: ...' status."""
    s = str(status)
    m = re.search(r"Cannot open file (\S+)", s)
    if m:
        return ("input file not found (labelling-Mac /Users/ path)" if m.group(1).startswith("/Users/")
                else "input file not found")
    return re.sub(r"^failed:\s*", "", s)[:80]


def relabel_summary(path=RELABEL_SUMMARY):
    """Status counts of gc_knee_jr's summary CSV, per lab (lab = folder name prefix), and the failure reasons; None
    without the file. The CSV holds one row per job but no paths (365 folder names occur under two roots), so it is
    counted, not joined to folders."""
    if not os.path.exists(path):
        return None
    s = pd.read_csv(path, usecols=["folder", "status"])
    st = s["status"].astype(str)
    kind = st.where(~st.str.startswith("failed"), "failed")
    lab = s["folder"].astype(str).str.split("_").str[0]
    by_lab = {l: {k: int(v) for k, v in g.value_counts().items()} for l, g in kind.groupby(lab)}
    return dict(path=os.path.relpath(path, REPO), modified=time.strftime("%Y-%m-%d %H:%M:%S",
                                                                         time.localtime(_mtime(path))),
                n_rows=len(s), status_counts={k: int(v) for k, v in kind.value_counts().items()}, by_lab=by_lab,
                failure_reasons={k: int(v) for k, v in st[kind == "failed"].map(failure_reason).value_counts().items()})


def is_relabel_cmdline(cmdline):
    """A /proc cmdline (NUL-separated argv) of an interpreter running gc_knee_jr.py: an argument after argv[0] is the
    script itself. A shell whose command string merely mentions the script does not count."""
    args = [a for a in cmdline.split(b"\0") if a]
    return len(args) > 1 and any(os.path.basename(a) == b"gc_knee_jr.py" for a in args[1:])


def relabel_running():
    """True when a gc_knee_jr.py process is running on this machine (Linux /proc; None when it cannot be read)."""
    try:
        for p in os.listdir("/proc"):
            if p.isdigit():
                try:
                    with open(f"/proc/{p}/cmdline", "rb") as f:
                        if is_relabel_cmdline(f.read()):
                            return True
                except OSError:
                    continue
        return False
    except OSError:
        return None


def coverage_summary(cov):
    """Counts of relabel states, overall, per lab and per split/activity/lab, for run_info.json."""
    def counts(g):
        c = {k: int(v) for k, v in g["relabel"].value_counts().items()}
        n = len(g)
        out = dict(n_with_so=n, **c, labelled_frac=round(c.get("labelled", 0) / n, 4) if n else np.nan)
        un = g[g["relabel"] != "labelled"]
        if "setup_resolves" in g and len(un):
            out["unlabelled_setup_resolves"] = int(un["setup_resolves"].eq(True).sum())
            out["unlabelled_so_frames"] = int(un["so_frames"].fillna(0).sum())
        return out
    return dict(total=counts(cov), complete=bool((cov["relabel"] == "labelled").all()),
                by_lab={l: counts(g) for l, g in cov.groupby("lab")},
                by_group={"/".join(k): counts(g) for k, g in cov.groupby(GROUP)})


# ─── Aggregation ────────────────────────────────────────────────────────────────

HEADLINE = ("fy_stance_peak", "fy_stance_mean", "fy_full_peak", "fmed_stance_peak", "fmed_stance_mean",
            "fmed_gt14_frac", "fres_peak", "mx_range")


def numeric_metrics(df):
    skip = {"mass_kg", "height_m"}
    return [c for c in df.columns if c not in skip and c not in GROUP + ["folder", "person", "status"]
            and pd.api.types.is_numeric_dtype(df[c])]


def _with_all(df):
    """The groups plus a pooled lab = ALL row per split and activity."""
    return pd.concat([df, df.assign(lab="ALL")], ignore_index=True)


def aggregate(df, metrics=None):
    """Long table: per split x activity x lab (and lab ALL), per metric: n, median, q25, q75, p95, mean."""
    metrics = metrics or numeric_metrics(df)
    rows = []
    for keys, g in _with_all(df).groupby(GROUP, sort=False):
        for m in metrics:
            x = pd.to_numeric(g[m], errors="coerce").astype(float).dropna().to_numpy()
            if not len(x):
                continue
            q = np.percentile(x, [25, 50, 75, 95])
            rows.append(dict(zip(GROUP, keys), metric=m, n=len(x), median=q[1], q25=q[0], q75=q[2], p95=q[3],
                             mean=float(x.mean())))
    return pd.DataFrame(rows)


def headline(df, sides=SIDES):
    """Per group and knee: median [q25, q75] and p95 of the headline metrics, stored vs SO+JR (the stored files have
    no left knee)."""
    rows = []
    for side in sides:
        rows += _headline_side(df, side)
    return pd.DataFrame(rows)


def _people(g):
    """n_people, the person with the most trials and that person's share of the trials."""
    c = g["person"].value_counts()
    return dict(n_people=len(c), top_person=c.index[0] if len(c) else "",
                top_person_share=round(float(c.iloc[0] / len(g)), 3) if len(c) else np.nan)


def _headline_side(df, side):
    rows = []
    for keys, g in _with_all(df).groupby(GROUP, sort=False):
        r = dict(zip(GROUP, keys), knee=side, n_trials=len(g), **_people(g))
        for m in HEADLINE:
            for src, _ in SOURCES:
                c = f"{src}_{side}_{m}"
                x = g[c].dropna().to_numpy(float) if c in g else np.array([])
                if len(x):
                    q = np.percentile(x, [25, 50, 75, 95])
                    r[f"{src}_{m}"] = f"{q[1]:.2f} [{q[0]:.2f}, {q[2]:.2f}]"
                    r[f"{src}_{m}_p95"] = round(float(q[3]), 2)
                    pm = g.groupby("person")[c].median().dropna()
                    r[f"{src}_{m}_by_person"] = f"{np.median(pm):.2f} [{pm.min():.2f}, {pm.max():.2f}]"
        rows.append(r)
    return rows


CORRELATES = ("speed_ms", "mass_kg", "height_m", "so_sat_any_frac", "so_sat_gt5_frac", "so_n_sat_muscles",
              "vas_{side}_max_bw", "gas_{side}_max_bw", "res_knee_{side}_peak_nmkg", "res_ankle_{side}_peak_nmkg",
              "res_hipadd_{side}_peak_nmkg", "res_force_share", "resid_force_pct", "resid_moment_pct",
              "cop_{side}_off_frac", "cop_{side}_dy_med",
              "stored_{side}_fy_stance_peak")


def correlates(df, target="sojr_{side}_fy_stance_peak", side="r", min_n=20):
    """Spearman rank correlation of a per-trial label statistic with speed, body size and the SO quality indicators,
    per split x activity x lab with at least min_n trials (and lab = ALL)."""
    tgt = target.format(side=side)
    rows = []
    for keys, g in _with_all(df).groupby(GROUP, sort=False):
        if tgt not in g or g[tgt].notna().sum() < min_n:
            continue
        r = dict(zip(GROUP, keys), target=tgt, n=int(g[tgt].notna().sum()))
        y = g[tgt].astype(float).rank()
        for c in CORRELATES:
            c = c.format(side=side)
            if c in g and g[c].notna().sum() >= min_n and g[c].nunique() > 1:
                ok = g[c].notna() & g[tgt].notna()
                r[c] = float(np.corrcoef(y[ok].rank(), g.loc[ok, c].astype(float).rank())[0, 1])
        rows.append(r)
    return pd.DataFrame(rows)


def _filter_flags(g, p, walking):
    """Per-trial booleans of the training label filters for one source/side prefix (rows with a verdict only)."""
    gg = g[g[p + "reject_any"].notna()]
    flat = gg[p + "flat"].astype(bool)
    rej = gg[p + "reject_any"].astype(bool)
    return pd.DataFrame({"person": gg["person"], "reject_peak": gg[p + "reject_peak"].astype(bool),
                         "reject_mx": gg[p + "reject_mx"].astype(bool), "reject_any": rej, "flat": flat,
                         "dropped": rej | flat if walking else rej}, index=gg.index), gg


def filter_rejections(df):
    """Share of trials each training label filter rejects (right knee, as the training loader; left for reference).

    keep = passes the resultant and |M_x| filters and, for walking, the flat filter (train_mlp applies the flat
    filter only when activity == 'walking'). *_pct pools the trials; *_pct_pw is the mean over people of each
    person's share (every person weighted equally)."""
    rows = []
    for keys, g in _with_all(df).groupby(GROUP, sort=False):
        r = dict(zip(GROUP, keys), n_trials=len(g), **_people(g))
        for src, _ in SOURCES:
            for side in SIDES:
                p = f"{src}_{side}_"
                if p + "reject_any" not in g or g[p + "reject_any"].isna().all():
                    continue
                fl, gg = _filter_flags(g, p, keys[1] == "walking")
                pw = fl.groupby("person").mean(numeric_only=True).mean()
                r[f"{p}n"] = len(fl)
                for k in ("reject_peak", "reject_mx", "reject_any", "flat", "dropped"):
                    r[f"{p}{k}_pct"] = 100 * fl[k].mean()
                    r[f"{p}{k}_pct_pw"] = 100 * pw[k]
                r.update({f"{p}frames_res_gt6_pct": 100 * gg[p + "n_res_gt6"].sum() / max(gg[p + "n"].sum(), 1),
                          f"{p}frames_mx_gt015_pct": 100 * gg[p + "n_mx_gt015"].sum() / max(gg[p + "n"].sum(), 1)})
        rows.append(r)
    return pd.DataFrame(rows)


PERSON_METRICS = ("stored_r_fy_stance_peak", "sojr_r_fy_stance_peak", "sojr_l_fy_stance_peak", "sojr_r_fy_full_peak",
                  "sojr_l_fy_full_peak", "stored_r_fmed_stance_peak", "sojr_r_fmed_stance_peak", "sojr_r_fres_peak",
                  "sojr_r_mx_range", "res_knee_r_peak_nmkg", "res_ankle_r_peak_nmkg", "so_sat_any_frac",
                  "vas_r_max_bw", "resid_force_pct", "cop_r_off_frac", "speed_ms", "ds_frac", "mass_kg")


def by_person(df):
    """One row per split x activity x lab x person: trial count, medians of PERSON_METRICS and the share of trials
    the training label filters drop (right knee)."""
    rows = []
    for keys, g in df.groupby(GROUP + ["person"], sort=True):
        r = dict(zip(GROUP + ["person"], keys), n_trials=len(g))
        for c in PERSON_METRICS:
            if c in g and g[c].notna().any():
                r[c + "_med"] = float(g[c].median())
        for src, _ in SOURCES:
            p = f"{src}_r_"
            if p + "reject_any" in g and g[p + "reject_any"].notna().any():
                fl, _ = _filter_flags(g, p, keys[1] == "walking")
                for k in ("reject_peak", "reject_mx", "flat", "dropped"):
                    r[f"{p}{k}_pct"] = 100 * fl[k].mean()
        rows.append(r)
    return pd.DataFrame(rows)


FREGLY_COLS = ("n", "stance_n", "n_full_stance", "fy_stance_peak", "fy_stance_mean", "fy_full_peak",
               "fmed_stance_peak", "reject_any", "flat", "n_nonfinite")


def fregly_instrumented(df):
    """The instrumented knee of every Fregly folder with a known implant side (SO+JR and, for the right knee, the
    stored labels), with speed and double support."""
    if "instrumented_side" not in df:
        return pd.DataFrame()
    rows = []
    for _, t in df[df["instrumented_side"].notna()].iterrows():
        s = t["instrumented_side"]
        r = {c: t.get(c) for c in ("folder", "person", "split", "activity", "status", "speed_ms", "ds_frac")}
        r["side"] = s
        for src, _ in SOURCES:
            for m in FREGLY_COLS:
                r[f"{src}_{m}"] = t.get(f"{src}_{s}_{m}", np.nan)
        rows.append(r)
    return pd.DataFrame(rows)


GC_LABEL_ROOTS = tuple(os.path.join(REPO, "jcf", "full_duration", s, "walking") for s in ("training", "testing"))


def labels_at_frames(f, sub="jcf_output_sojr", roots=GC_LABEL_ROOTS):
    """F_y (BW, signed) of the instrumented knee from each segment folder's own <sub>/ JR file at the frame times t_seg
    of a grand_challenge frames table (BW = the table's mass * g, as grand_challenge.py). Folders are looked up in
    <split>/walking like grand_challenge.segment_dirs; NaN where the folder, its labels or the time is missing."""
    out = pd.Series(np.nan, index=f.index)
    for folder, g in f.groupby("folder"):
        d = next((os.path.join(r, folder) for r in roots if os.path.isdir(os.path.join(r, folder))), None)
        side = FREGLY_INSTRUMENTED.get(person_id(folder))
        p = os.path.join(d, sub, JR_FILE) if d else None
        if side is None or p is None or not os.path.exists(p):
            continue
        jr = read_sto(p)
        c = f"walker_knee_{side}_on_tibia_{side}_in_tibia_{side}_fy"
        if c not in jr:
            continue
        t, y = jr["time"].to_numpy(float), jr[c].to_numpy(float)
        ts = g["t_seg"].to_numpy(float)
        v = np.interp(ts, t, y) / (g["mass"].to_numpy(float) * G)
        v[(ts < t[0] - 1e-6) | (ts > t[-1] + 1e-6)] = np.nan
        out.loc[g.index] = v
    return out


def measured_check(frames_path=GC_FRAMES, source="osim_sojr", fill=True, roots=GC_LABEL_ROOTS):
    """SO+JR |F_y| of the instrumented knee against the measured total axial force (both BW), from grand_challenge.py's
    frames.csv.gz (exact index alignment). Returns (table, summary): table rows of kind 'segment' (instrumented-limb
    stance frames of one segment: means, peaks, the stance % covered) and 'window' (stance windows covered >= 90 %:
    peaks); summary per subject. (None, None) without the file or the source.

    fill: frames without a fy_<source> value (segments whose SO+JR labels were written after grand_challenge.py ran)
    get it from the segment folder's labels (labels_at_frames); summary n_filled counts them and fill_check_max_diff is
    the largest |difference| from grand_challenge's own value on the frames that have both (0 = same reading).
    DM (GC2) has a likely implant timing lag (results/thesis/gc/EXPLORE_NOTES.md): its frame-level numbers are at lag 0."""
    if not frames_path or not os.path.exists(frames_path):
        return None, None
    f = pd.read_csv(frames_path)
    fy = f"fy_{source}"
    if fy not in f:
        return None, None
    f["filled"] = False
    check = np.nan
    if fill and {"folder", "t_seg", "mass"} <= set(f.columns):
        own = labels_at_frames(f, roots=roots)
        both = f[fy].notna() & own.notna()
        check = float((f.loc[both, fy] - own[both]).abs().max()) if both.any() else np.nan
        miss = f[fy].isna() & own.notna()
        f.loc[miss, fy] = own[miss]
        f.loc[miss, "filled"] = True
    f = f[f["stance"].astype(bool) & f[fy].notna() & f["meas_tot"].notna()].copy()
    if not len(f):
        return None, None
    f["pred"] = f[fy].abs()
    rows = []
    for (subj, folder), g in f.groupby(["subject", "folder"]):
        rows.append(dict(kind="segment", subject=subj, folder=folder, n_frames=len(g), filled=bool(g["filled"].any()),
                         pct_min=float(g["pct"].min()), pct_max=float(g["pct"].max()),
                         full_stance=bool(g["pct"].min() <= 5 and g["pct"].max() >= 95),
                         pred_mean=float(g["pred"].mean()), meas_mean=float(g["meas_tot"].mean()),
                         pred_peak=float(g["pred"].max()), meas_peak=float(g["meas_tot"].max())))
        for sid, gs in g[g["stance_id"] >= 0].groupby("stance_id"):
            for part, lo, hi in STANCE_WINDOWS:
                w = gs[(gs["pct"] >= lo) & (gs["pct"] <= hi)]
                if len(w) >= 3 and (w["pct"].max() - w["pct"].min()) >= 0.9 * (hi - lo):
                    rows.append(dict(kind="window", subject=subj, folder=folder, stance_id=int(sid), part=part,
                                     n_frames=len(w), pred_peak=float(w["pred"].max()),
                                     meas_peak=float(w["meas_tot"].max())))
    tab = pd.DataFrame(rows)
    tab["peak_diff"] = tab["pred_peak"] - tab["meas_peak"]
    summ = {}
    for subj, g in f.groupby("subject"):
        seg = tab[(tab["kind"] == "segment") & (tab["subject"] == subj)]
        win = tab[(tab["kind"] == "window") & (tab["subject"] == subj)]
        s = dict(n_segments=len(seg), n_frames=len(g), n_filled=int(g["filled"].sum()),
                 fill_check_max_diff=None if np.isnan(check) else round(check, 6),
                 n_full_stance_segments=int(seg["full_stance"].sum()),
                 frame_pred_mean=round(float(g["pred"].mean()), 3), frame_meas_mean=round(float(g["meas_tot"].mean()), 3),
                 frame_bias=round(float((g["pred"] - g["meas_tot"]).mean()), 3),
                 frame_r=(round(float(np.corrcoef(g["pred"], g["meas_tot"])[0, 1]), 3)
                          if len(g) > 2 and g["pred"].std() > 0 and g["meas_tot"].std() > 0 else None),
                 segment_peak_pred_median=round(float(seg["pred_peak"].median()), 3),
                 segment_peak_meas_median=round(float(seg["meas_peak"].median()), 3))
        for part, _, _ in STANCE_WINDOWS:
            w = win[win["part"] == part]
            if len(w):
                s[f"{part}_windows"] = len(w)
                s[f"{part}_peak_pred_median"] = round(float(w["pred_peak"].median()), 3)
                s[f"{part}_peak_meas_median"] = round(float(w["meas_peak"].median()), 3)
        summ[subj] = s
    return tab, summ


# ─── Figures ────────────────────────────────────────────────────────────────────

SRC_STYLE = {"stored": dict(label="stored (jcf_output)", color="#2a78d6"),
             "sojr": dict(label="SO+JR (jcf_output_sojr)", color="#eb6834")}
INK, MUTED, GRID, REF = "#0b0b0b", "#52514e", "#e4e3df", "#8a8984"


def _labs(df):
    present = list(dict.fromkeys(df["lab"]))
    return [l for l in LAB_ORDER if l in present] + sorted(l for l in present if l not in LAB_ORDER)


def _style(ax):
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)


def fig_peaks_by_lab(df, path, side="r"):
    """Per-lab distributions of stance-peak |F_y| and F_medial, stored vs SO+JR, walking and static (one knee)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    acts = [a for a in ("walking", "static") if (df["activity"] == a).any()]
    metrics = (("fy_stance_peak", "stance-peak |F$_y$| (BW)"), ("fmed_stance_peak", "stance-peak F$_{medial}$ (BW)"))
    fig, axes = plt.subplots(len(metrics), len(acts), figsize=(5.2 * len(acts), 7.2), squeeze=False)
    for j, act in enumerate(acts):
        sub = df[df["activity"] == act]
        labs = _labs(sub)
        for i, (m, ylabel) in enumerate(metrics):
            ax = axes[i, j]
            _style(ax)
            vals_all = []
            for k, (src, _) in enumerate(SOURCES):
                c = f"{src}_{side}_{m}"
                data = [sub.loc[sub["lab"] == l, c].dropna().to_numpy(float) if c in sub else np.array([])
                        for l in labs]
                pos = np.arange(len(labs)) + (k - 0.5) * 0.36
                keep = [n for n, x in enumerate(data) if len(x)]
                if keep:
                    bp = ax.boxplot([data[n] for n in keep], positions=pos[keep], widths=0.3, whis=(5, 95),
                                    showfliers=False, patch_artist=True, medianprops=dict(color=INK, lw=1.5),
                                    whiskerprops=dict(color=MUTED), capprops=dict(color=MUTED),
                                    boxprops=dict(edgecolor="white", lw=1))
                    for b in bp["boxes"]:
                        b.set_facecolor(SRC_STYLE[src]["color"])
                vals_all += [np.percentile(x, 95) for x in data if len(x)]
            if m == "fy_stance_peak":
                if act == "walking":                  # in vivo tibiofemoral walking peaks (Kutzner 2010; Fregly 2012)
                    ax.axhspan(2.2, 3.0, color=REF, alpha=0.18, lw=0)
                    ax.text(len(labs) - 0.5, 2.6, "in vivo walking\npeak 2.2-3 BW", ha="right", va="center",
                            fontsize=7.5, color=MUTED)
                ax.axhline(MAX_PEAK_BW, color=MUTED, ls="--", lw=1)
                ax.text(-0.45, MAX_PEAK_BW, " 6 BW resultant filter", va="bottom", fontsize=7.5, color=MUTED)
            else:
                ax.axhline(LIMIT_BW, color=MUTED, ls="--", lw=1)
                ax.text(-0.45, LIMIT_BW, " 1.4 BW limit", va="bottom", fontsize=7.5, color=MUTED)
            top = max(vals_all + [MAX_PEAK_BW if m == "fy_stance_peak" else LIMIT_BW]) * 1.1
            ax.set_ylim(0, top)
            ns = [int(sub.loc[sub["lab"] == l, f"sojr_{side}_{m}"].notna().sum()) if f"sojr_{side}_{m}" in sub else 0
                  for l in labs]
            npp = [sub.loc[sub["lab"] == l, "person"].nunique() for l in labs]
            ax.set_xticks(np.arange(len(labs)), [f"{l}\nn={n}\n{k} people" for l, n, k in zip(labs, ns, npp)],
                          fontsize=7.5)
            ax.set_ylabel(ylabel, color=INK)
            if i == 0:
                ax.set_title(f"{act}", color=INK, fontsize=11)
    handles = [plt.Rectangle((0, 0), 1, 1, color=s["color"]) for s in SRC_STYLE.values()]
    fig.legend(handles, [s["label"] for s in SRC_STYLE.values()], loc="upper center", ncol=2, frameon=False,
               fontsize=9)
    fig.text(0.5, 0.005, f"{'right' if side == 'r' else 'left'} knee; boxes IQR, line median, whiskers 5-95 %; "
             "stance = |GRF$_y$| > 5 % BW", ha="center", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def fig_peak_vs_speed(df, path, side="r"):
    """Stance-peak |F_y| against walking speed, one panel per lab, stored vs SO+JR (walking trials)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sub = df[(df["activity"] == "walking") & df.get("speed_ms", pd.Series(np.nan, index=df.index)).notna()]
    labs = _labs(sub)
    if not labs:
        return False
    nc = min(3, len(labs))
    nr = int(np.ceil(len(labs) / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(3.6 * nc, 3.0 * nr), squeeze=False, sharey=True, sharex=True)
    ymax = 0.0
    for n, lab in enumerate(labs):
        ax = axes.flat[n]
        _style(ax)
        g = sub[sub["lab"] == lab]
        for src, _ in SOURCES:
            c = f"{src}_{side}_fy_stance_peak"
            if c not in g:
                continue
            ok = g[c].notna()
            ax.scatter(g.loc[ok, "speed_ms"], g.loc[ok, c], s=9, alpha=0.45, lw=0, color=SRC_STYLE[src]["color"],
                       label=SRC_STYLE[src]["label"])
            if ok.any():
                ymax = max(ymax, float(np.percentile(g.loc[ok, c], 99)))
        ax.axhspan(2.2, 3.0, color=REF, alpha=0.18, lw=0)
        ax.set_title(f"{lab} (n={len(g)}, {g['person'].nunique()} people)", fontsize=9.5, color=INK)
        if n % nc == 0:
            ax.set_ylabel("stance-peak |F$_y$| (BW)", color=INK)
        if n // nc == nr - 1:
            ax.set_xlabel("walking speed (m/s)", color=INK)
    for ax in axes.flat[len(labs):]:
        ax.set_visible(False)
    axes.flat[0].set_ylim(0, max(ymax * 1.05, 3.5))
    axes.flat[0].set_xlim(0, max(float(np.nanpercentile(sub["speed_ms"], 99.5)) * 1.05, 1.0))
    for ax in axes.flat[:len(labs)]:
        ax.tick_params(labelbottom=True)
    h, l_ = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, l_, loc="upper center", ncol=2, frameon=False, fontsize=8.5, markerscale=2)
    fig.text(0.5, 0.005, f"{'right' if side == 'r' else 'left'} knee; speed = median |v$_{{COM}}$ - "
             "v$_{stance\\ calcaneus}$| in single stance; grey band = in vivo peak 2.2-3 BW", ha="center",
             fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return True


def fig_filter_rejections(rej, path):
    """Share of trials dropped by the training label filters (right knee), stored vs SO+JR, per lab x activity."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    g = rej[rej["lab"] != "ALL"].copy()
    g = g.groupby(["activity", "lab"], as_index=False).apply(
        lambda x: pd.Series({"n_trials": x["n_trials"].sum(),
                             **{f"{s}_r_dropped_pct": np.average(x[f"{s}_r_dropped_pct"].fillna(0),
                                                                 weights=x["n_trials"])
                                for s, _ in SOURCES if f"{s}_r_dropped_pct" in x}}), include_groups=False)
    acts = [a for a in ("walking", "static") if (g["activity"] == a).any()]
    fig, axes = plt.subplots(1, len(acts), figsize=(5.0 * len(acts), 3.4), squeeze=False)
    for j, act in enumerate(acts):
        ax = axes[0, j]
        _style(ax)
        sub = g[g["activity"] == act].set_index("lab")
        labs = [l for l in _labs(sub.reset_index()) if l in sub.index]
        for k, (src, _) in enumerate(SOURCES):
            c = f"{src}_r_dropped_pct"
            if c not in sub:
                continue
            v = sub.loc[labs, c].to_numpy(float)
            x = np.arange(len(labs)) + (k - 0.5) * 0.38
            ax.bar(x, v, width=0.36, color=SRC_STYLE[src]["color"], label=SRC_STYLE[src]["label"],
                   edgecolor="white", lw=1)
            for xi, vi in zip(x, v):
                ax.text(xi, vi + 1, f"{vi:.0f}", ha="center", va="bottom", fontsize=7, color=MUTED)
        ax.set_xticks(np.arange(len(labs)), [f"{l}\nn={int(sub.loc[l, 'n_trials'])}" for l in labs], fontsize=8.5)
        ax.set_ylim(0, 110)
        ax.set_ylabel("trials dropped (%)", color=INK)
        ax.set_title(act + (" (6 BW, |M$_x$| 0.15, flat)" if act == "walking" else " (6 BW, |M$_x$| 0.15)"),
                     fontsize=10, color=INK)
    h, l_ = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l_, loc="upper center", ncol=2, frameon=False, fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, dpi=180)
    plt.close(fig)


# ─── Main ───────────────────────────────────────────────────────────────────────

def git_commit():
    try:
        return subprocess.run(["git", "-C", REPO, "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              timeout=10).stdout.strip()
    except Exception:
        return ""


def _json_default(o):
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    return str(o)


def nonfinite_folders(df):
    """Folders with non-finite label frames: {folder: {split, activity, '<source>_<side>': frames}}."""
    out = {}
    for _, r in df.iterrows():
        n = {f"{src}_{s}": int(r[f"{src}_{s}_n_nonfinite"]) for src, _ in SOURCES for s in SIDES
             if f"{src}_{s}_n_nonfinite" in r and pd.notna(r[f"{src}_{s}_n_nonfinite"]) and r[f"{src}_{s}_n_nonfinite"] > 0}
        if n:
            out[f"{r['split']}/{r['activity']}/{r['folder']}"] = n
    return out


def _coverage_by_group(cov):
    """n_with_so and labelled share per split x activity x lab (and lab ALL), to merge into the tables."""
    c = _with_all(cov.assign(labelled=cov["relabel"] == "labelled"))
    return c.groupby(GROUP, sort=False).agg(n_with_so=("labelled", "size"),
                                            sojr_coverage=("labelled", "mean")).round(3).reset_index()


def write_outputs(df, out, figures=True, info=None, cov=None, gc_frames=GC_FRAMES):
    os.makedirs(out, exist_ok=True)
    ok = df[df["status"] == "ok"]
    df.to_csv(os.path.join(out, "per_trial.csv"), index=False)
    aggregate(ok).to_csv(os.path.join(out, "by_group.csv"), index=False)
    head = headline(ok)
    rej = filter_rejections(ok)
    if cov is not None and len(cov):
        cg = _coverage_by_group(cov)
        head = head.merge(cg, on=GROUP, how="left")
        rej = rej.merge(cg, on=GROUP, how="left")
    head.to_csv(os.path.join(out, "headline.csv"), index=False)
    rej.to_csv(os.path.join(out, "filter_rejections.csv"), index=False)
    by_person(ok).to_csv(os.path.join(out, "by_person.csv"), index=False)
    correlates(ok).to_csv(os.path.join(out, "correlates.csv"), index=False)
    fregly_instrumented(df).to_csv(os.path.join(out, "fregly_instrumented.csv"), index=False)
    meas, meas_summary = measured_check(gc_frames)
    if meas is not None:
        meas.to_csv(os.path.join(out, "fregly_vs_measured.csv"), index=False)
    if figures and len(ok):
        fig_peaks_by_lab(ok, os.path.join(out, "fig_peaks_by_lab.png"))
        fig_peak_vs_speed(ok, os.path.join(out, "fig_peak_fy_vs_speed.png"))
        fig_filter_rejections(rej, os.path.join(out, "fig_filter_rejections.png"))
    if info is not None:
        info = dict(info, n_rows=len(df), n_ok=len(ok), status_counts=df["status"].value_counts().to_dict(),
                    nonfinite_folders=nonfinite_folders(df),
                    per_group={"/".join(k): int(v) for k, v in ok.groupby(GROUP).size().items()},
                    fregly_vs_measured=dict(source=os.path.relpath(gc_frames, REPO) if gc_frames else None,
                                            by_subject=meas_summary))
        with open(os.path.join(out, "run_info.json"), "w") as f:
            json.dump(info, f, indent=1, default=_json_default)
    return head, rej


def report(head, rej):
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 40)
    cols = [c for c in ("split", "activity", "lab", "knee", "n_trials", "n_people", "top_person_share",
                        "stored_fy_stance_peak", "sojr_fy_stance_peak", "sojr_fy_stance_peak_by_person",
                        "sojr_fy_stance_peak_p95", "stored_fmed_stance_peak", "sojr_fmed_stance_peak", "sojr_coverage")
            if c in head]
    print(head.loc[head["knee"] == "r", cols].to_string(index=False))
    cols = [c for c in ("split", "activity", "lab", "n_trials", "n_people", "stored_r_dropped_pct",
                        "sojr_r_reject_peak_pct", "sojr_r_reject_mx_pct", "sojr_r_flat_pct", "sojr_r_dropped_pct",
                        "sojr_r_dropped_pct_pw") if c in rej]
    print(rej[cols].round(1).to_string(index=False))


def coverage_warning(cs, summary, running):
    """Console lines on how much of the folder set has SO+JR labels."""
    t = cs["total"]
    lines = [f"SO+JR labels: {t.get('labelled', 0)} of {t['n_with_so']} folders with SO output "
             f"({100 * t['labelled_frac']:.1f} %); " + ", ".join(f"{k} {v}" for k, v in t.items()
                                                                if k in ("failed", "pending", "not run", "no setup"))]
    if not cs["complete"]:
        lines.append("  per lab (labelled / with SO): " + ", ".join(
            f"{l} {c.get('labelled', 0)}/{c['n_with_so']}" for l, c in cs["by_lab"].items()))
        lines.append("  WARNING: the relabel is incomplete; every table below describes the labelled folders only.")
    if running:
        lines.append("  a gc_knee_jr.py process is running: labels are still being written.")
    if summary is not None:
        lines.append(f"  {summary['path']} ({summary['modified']}): " +
                     ", ".join(f"{k} {v}" for k, v in summary["status_counts"].items()))
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", nargs="+", default=list(DEFAULT_ROOTS))
    ap.add_argument("--pattern", default="*")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="first N folders only (smoke test)")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--from-csv", default=None, help="rebuild the tables and figures from an existing per_trial.csv "
                    "(and the relabel_coverage.csv next to it, when there is one); run_info.json is not rewritten")
    ap.add_argument("--relabel-summary", default=RELABEL_SUMMARY, help="gc_knee_jr's summary CSV")
    ap.add_argument("--gc-frames", default=GC_FRAMES, help="grand_challenge.py frames.csv.gz for the measured check "
                    "('' to skip)")
    args = ap.parse_args(argv)
    if args.from_csv:
        df = pd.read_csv(args.from_csv)
        cp = os.path.join(os.path.dirname(os.path.abspath(args.from_csv)), "relabel_coverage.csv")
        cov = pd.read_csv(cp) if os.path.exists(cp) else None
        head, rej = write_outputs(df, args.out, figures=not args.no_figures, cov=cov, gc_frames=args.gc_frames)
        report(head, rej)
        return df
    t0 = time.time()
    roots = [os.path.abspath(r) for r in args.root]
    cov = relabel_coverage(find_folders(roots, args.pattern, need_sojr=False), args.workers, args.relabel_summary)
    summary, running = relabel_summary(args.relabel_summary), relabel_running()
    cs = coverage_summary(cov) if len(cov) else None
    if cs is not None:
        print(coverage_warning(cs, summary, running), flush=True)
    dirs = [os.path.join(REPO, p) for p in cov.loc[cov["relabel"] == "labelled", "path"]] if len(cov) else []
    if args.limit:
        dirs = dirs[:args.limit]
    if not dirs:
        raise SystemExit("no folders with jcf_output_sojr/ labels under the roots")
    print(f"{len(dirs)} folders with SO+JR labels", flush=True)
    df = collect(dirs, args.workers)
    os.makedirs(args.out, exist_ok=True)
    cov.to_csv(os.path.join(args.out, "relabel_coverage.csv"), index=False)
    info = dict(roots=args.root, pattern=args.pattern, n_folders=len(dirs), relabel_coverage=cs,
                relabel_complete=bool(cs and cs["complete"]), relabel_process_running=running,
                relabel_summary=summary, git=git_commit(),
                created=time.strftime("%Y-%m-%d %H:%M:%S"), seconds=round(time.time() - t0, 1))
    head, rej = write_outputs(df, args.out, figures=not args.no_figures, info=info, cov=cov, gc_frames=args.gc_frames)
    report(head, rej)
    nf = nonfinite_folders(df)
    if nf:
        print(f"{len(nf)} folders with non-finite labels (status 'nonfinite labels', left out of the tables): "
              + ", ".join(sorted(nf)[:5]))
    if cs is not None and not cs["complete"]:
        print("WARNING: SO+JR relabel incomplete (" + f"{cs['total'].get('labelled', 0)}/{cs['total']['n_with_so']}"
              + " folders); see run_info.json relabel_coverage")
    print(f"{len(dirs)} folders in {time.time() - t0:.0f} s; wrote {args.out}")
    return df


if __name__ == "__main__":
    main()

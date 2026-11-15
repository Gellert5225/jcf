#!/usr/bin/env python3
"""In vivo validation on the Knee Grand Challenge (thesis Aim 2): measured medial contact force from instrumented
knee implants against the GRF-free surrogate and against OpenSim joint reaction labels.

Three implant patients (Fregly et al. 2012) are in our AddBiomechanics Fregly2012 b3d files and were cut into
labelled segment folders, jcf/full_duration/<split>/walking/fregly_<code>_*:

    JW  1st competition  RIGHT knee  eKnee, 4 uniaxial load cells  200 Hz  fregly_1GC  training folder, excluded
    DM  2nd competition  RIGHT knee  eTibia, 6 channels            200 Hz  fregly_2GC  training folder, excluded
    SC  3rd competition  LEFT knee   eTibia, 6 channels            120 Hz  fregly_3GC  test split

Folders under jcf/external/grand_challenge/ are not named after the competitions, so every subject is identified by
its knee-force file prefix (jw_, dm_, SC_). The package files are only parsed as data; nothing in them is executed.

Pipeline, per segment folder:
  1. Measured medial / lateral / total force from the implant file (lbf -> N, in*lbf -> N*mm; the package README
     regressions for DM and SC; JW medial = posterior-medial + anterior-medial cell), in BW of the AddBiomechanics
     metadata mass (the mass the surrogate is given). The published mass is recorded alongside.
  2. Alignment: the segment's vertical GRF (calcn_r + calcn_l) is cross-correlated with the GC force plates
     (sum of |Fz|, sampled at the marker rows) over every gait trial of the subject. This finds the trial and the GC
     row of segment frame 0. It is checked against the exact map from marker / IK identity (--build-maps), when that
     map exists. Poor correlations, ambiguous matches and disagreements are rejected and reported.
     AddBiomechanics stores SC's GRF one frame (8.3 ms) late against the kinematics (the organisers synced the implant
     to the markers), so kinematic row = GRF row + 1 for SC and + 0 for JW and DM.
  3. Surrogate inputs from the same files the training builder reads (label-free copy of
     train_mlp._build_com_features; --features deploy = exact time alignment and 100 Hz windows). For SC's left knee
     the inputs are mirrored about the sagittal plane (mirror_per_frame), so the network's "right knee" is SC's left.
  4. OpenSim labels of the instrumented knee (walker_knee_r for JW/DM, walker_knee_l for SC), two sources:
       osim_state  the stored jcf_output/ labels (right knee only). pipeline/batch_process.py ran SO and JR in one
                   AnalyzeTool with an empty forces_file, so JR computed the muscle forces from the model states, not
                   from SO (analysis/thesis/gc_knee_jr.py documents the evidence). These are also the training labels.
       osim_sojr   SO+JR proper: JR rerun with forces_file = the folder's SO force file, both knees
                   (jcf_output_sojr/, written by analysis/thesis/gc_knee_jr.py; user-run).
     Kaneda et al.'s published StatOp_Val SO+JR medial force for SC (62 half-stance windows) is added on its frames.
  5. Metrics on instrumented-limb stance frames (MAE, RMSE, bias, r); stance peaks in an early (5-45 %) and a late
     (55-95 %) window of stance, scored only where the segment covers the window and the measured maximum is an
     interior local maximum (not within --edge-margin-pct of the covered window's edge); gait-modification effects,
     a phi/d sweep, the implied phi/d, a lag sweep of the measured force, and a DM sensitivity run at one fixed,
     model-independent lag (DM's implant sync is suspect; lag 0 stays the primary result).
  6. Figures: representative stances and a stance-peak scatter.

F_medial = phi*|F_y| + M_x*H/d (BW), phi = 0.5, d = 45 mm by default, with M_x medial-positive. --mx-convention
physical (default, settled by analysis/thesis/knee_mx_sign.py and tests/test_knee_mx_sign.py): medial M_x = -mx at
walker_knee_r and +mx at walker_knee_l. The surrogates predict the raw right-knee mx they were trained on (inputs
mirrored for a left knee), so their medial M_x is -prediction. --mx-convention pipeline reproduces the training-code
sign (+mx right, -mx left), which is inverted; every run before 2026-11-10 used it. frames.csv.gz stores the
medial-positive M_x of the run's convention (run.json args.mx_convention); mx_sign_check.csv tests it against the
implant's medial - lateral force.

Usage (repo root; both commands take a few minutes at most):

    # once, jcf env (reads the b3d files with nimblephysics): exact b3d -> GC offsets and the segment map
    conda run -n jcf python -I analysis/thesis/grand_challenge.py --build-maps

    # validation (any env with torch)
    python -I analysis/thesis/grand_challenge.py \\
        --ckpt models/runs/mlp/<A>.pt models/runs/mlp/<B>.pt --labels fpopt sept \\
        --phi 0.4 0.5 0.6 --d-mm 40 45 50 --lag-sweep-ms 100 --name pilot_AB

Outputs in results/thesis/gc/<name>/:
    segments.csv          one row per segment folder: alignment (trial, GC row, GRF lag, r), status, coverage
    alignment_trials.csv  the alignment quality aggregated per GC trial
    frames.csv.gz         per-frame measured force and every source's F_y, M_x (100 Hz eval grid)
    summary.csv           frame metrics per subject x source x subset (all / common / statop_overlap)
    summary_by_gait.csv   the same per gait type
    halves.csv            every stance window a segment touches, with its coverage (covered if >= --min-cover)
    stances.csv           peaks per covered window, measured and predicted, with meas_interior (scored if True)
    peak_counts.csv       per subject x window: covered, rejected (measured maximum at the window edge), scored
    peaks_summary.csv     peak errors per subject x source x window, interior measured peaks only
    gait_mods.csv         change of the mean peak from normal gait, measured vs predicted, with sign agreement
    sweep.csv             phi / d sweep;  implied_phi_d.csv  least-squares phi and d per subject x source
    lag_sweep.csv         metrics with the measured force shifted by -L..L ms
    hs_peak_delay.csv     time from a force-plate heel strike to the measured first peak, per stance (no model)
    dm_lag_sensitivity.csv  DM frame and peak metrics with the measured force shifted by that fixed lag
    mx_sign_check.csv     per subject x source: r of medial-positive M_x with the implant's medial - lateral force,
                          medial-moment share of frames, mean moment term and the medial share of the axial force
    statop_windows.csv    Kaneda's SO+JR medial force vs measured, per window (statop_halves.csv: their peaks)
    stance_events.csv     kinematic stance-event bias against the segments' force-plate events
    fig_gc_examples.png, fig_gc_peaks.png, README.md, run.json

Surrogate inputs are built on the whole segment (--context ik) and scored on the JR-window frames (--grid jr), so
every source is compared on identical frames without the windows' edge replication inside them. --features native
--context jr reproduces the training-time inputs of a legacy checkpoint exactly.
"""
import argparse
import datetime
import functools
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from analysis.thesis.knee_mx_sign import MX_CONVENTIONS, medial_mx_sign  # noqa: E402

GC_ROOT = os.path.join(REPO, "jcf", "external", "grand_challenge")
OUT_ROOT = os.path.join(REPO, "results", "thesis", "gc")
LBF_N = 4.4482216152605        # N per lbf
INLBF_NMM = 112.98482933       # N*mm per in*lbf
G = 9.81
CONTACT_BW = 0.05              # same contact rule as training (train_mlp.CONTACT_GRF_BW_THRESHOLD)

SUBJECTS = {
    "JW": dict(code="1GC", competition=1, side="r", fs_gc=200.0, prefix="jw_", implant="eKnee (4 uniaxial load cells)",
               split="training", b3d="with_arm/training/Fregly2012_Formatted_With_Arm/1GC/1GC.b3d",
               mass_published=68.0, height_published=1.66, grf_delay_frames=0,
               lab_axes={"x": ("X", 1.0), "z": ("Y", -1.0)}),
    "DM": dict(code="2GC", competition=2, side="r", fs_gc=200.0, prefix="dm_", implant="eTibia (6 channels)",
               split="training", b3d="with_arm/training/Fregly2012_Formatted_With_Arm/2GC/2GC.b3d",
               mass_published=70.0, height_published=1.70, grf_delay_frames=0,
               lab_axes={"x": ("X", 1.0), "z": ("Y", -1.0)}),
    "SC": dict(code="3GC", competition=3, side="l", fs_gc=120.0, prefix="sc_", implant="eTibia (6 channels)",
               split="testing", b3d="with_arm/testing/Fregly2012_Formatted_With_Arm/3GC/3GC.b3d",
               mass_published=78.4, height_published=1.67, grf_delay_frames=1,
               lab_axes={"x": ("X", -1.0), "z": ("Y", 1.0)}),
}
# Gait codes in the trial names. SC's "mtpgait" is forefoot strike by elimination against Kaneda et al.'s trial counts
# (baseline 5, bouncy 4, crouch 4, forefoot 5, medial thrust 5, smooth 5, trunk sway 4, poles 2 long + 3 short).
GAIT_LABELS = {"ngait": "normal", "mtgait": "medial thrust", "medialthrust": "medial thrust",
               "tsgait": "trunk sway", "trunksway": "trunk sway", "wpgait": "walking poles",
               "wpgaitl": "walking poles (long)", "wpgaits": "walking poles (short)", "bouncy": "bouncy",
               "crouch": "crouch", "smooth": "smooth", "mtpgait": "forefoot strike"}
STATOP_COLS = ("med_cond_joint_l_on_med_cond_l_in_med_cond_l_fy", "lat_cond_joint_l_on_lat_cond_l_in_lat_cond_l_fy")
PROVISIONAL = "PROVISIONAL (surrogate output; rerun after the *_t100 retraining)"
# OpenSim label sources: key -> (sub-folder of the segment, display name). osim_state = the stored labels (SO and JR in
# one AnalyzeTool, empty forces_file: JR muscle forces from the model states); osim_sojr = JR applying the SO forces.
OSIM_SOURCES = {
    "osim_state": ("jcf_output", "OpenSim JR, state-derived muscle forces (stored labels)"),
    "osim_sojr": ("jcf_output_sojr", "OpenSim SO+JR (ours, SO forces applied)"),
}
MX_README = {
    "physical": "M_x convention `physical` (medial-positive M_x = -mx at walker_knee_r, +mx at walker_knee_l; the "
                "surrogates' M_x = -output). Settled by `analysis/thesis/knee_mx_sign.py` "
                "(results/thesis/labels_qc/mx_sign/) and checked against the implant in the M_x sign section below.",
    "pipeline": "**M_x convention `pipeline` (+mx right, -mx left, the training code's sign). It is inverted "
                "(`analysis/thesis/knee_mx_sign.py`); this run only reproduces the old numbers.**",
}
RESERVED_LABELS = set(OSIM_SOURCES) | {"osim", "statop", "statop_own_windows", "meas"}
PEAK_WINDOWS = (("early", 5.0, 45.0), ("late", 55.0, 95.0))     # % of stance searched for the two peaks


# ─── Names, files and measured forces (pure, unit-tested) ─────────────────────

def trial_key(name):
    """Match key for a GC trial: lower-case alphanumerics, file suffix dropped ('dm_ngait4' == 'DM_ngait_4')."""
    base = os.path.basename(str(name))
    for suf in ("_knee_forces.csv", "_trajectories.csv", "_grf.csv", "_emg.csv", ".csv"):
        if base.lower().endswith(suf):
            base = base[:-len(suf)]
            break
    return re.sub(r"[^a-z0-9]", "", base.lower())


def gait_code(name):
    """'SC_ngait_og5' -> 'ngait', 'jw_wpgait_10' -> 'wpgait', 'SC_wpgait_l11' -> 'wpgaitl'."""
    s = re.sub(r"^(jw|dm|sc)_", "", os.path.basename(str(name)).lower())
    s = re.sub(r"(_og)?_?\d+$", "", s)
    return re.sub(r"[^a-z]", "", s)


def read_gc_csv(path):
    """A GC csv: rows end with a comma the header lacks, so read with index_col=False and drop 'Unnamed' columns."""
    df = pd.read_csv(path, index_col=False)
    return df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]


def time_column(df):
    """First column called time*, else column 0 (the JW eKnee header names it '17:00')."""
    for c in df.columns:
        if str(c).lower().startswith("time"):
            return df[c].to_numpy(dtype=float)
    return df.iloc[:, 0].to_numpy(dtype=float)


def measured_forces(subj, df):
    """Medial, lateral and total (axial) tibial contact force in N from one knee-force file.

    JW (eKnee): Sensor1..4 = posterior-medial, anterior-medial, anterior-lateral, posterior-lateral load cells (lbf).
        No regression is supplied; medial = S1 + S2, lateral = S3 + S4 (direct sum; check against Zhao et al. 2007).
    DM (eTibia, right): Sensor1..6 = Fx, Fy, Fz, Mx, My, Mz (lbf, in*lbf). README (Meyer et al. 2011):
        F_med = -0.876 F_ML + 0.494 F_SI - 0.0189 M_VV, F_lat = 0.876 F_ML + 0.505 F_SI + 0.0189 M_VV,
        F_ML = Sensor1, F_SI = Sensor3, M_VV = Sensor5 (moment about the anterior axis), N and N*mm.
    SC (eTibia, left): Fx..Tz (lbf, in*lbf). README: F_med = 0.942 Fx + 0.497 Fz + 0.0184 Ty,
        F_lat = -0.942 Fx + 0.503 Fz - 0.0184 Ty. Total axial force = Fz (compression positive).
    """
    t = time_column(df)
    if subj == "JW":
        s = df[["Sensor1", "Sensor2", "Sensor3", "Sensor4"]].to_numpy(dtype=float) * LBF_N
        med, lat = s[:, 0] + s[:, 1], s[:, 2] + s[:, 3]
        tot = med + lat
    elif subj == "DM":
        fml, fsi = df["Sensor1"].to_numpy(float) * LBF_N, df["Sensor3"].to_numpy(float) * LBF_N
        mvv = df["Sensor5"].to_numpy(float) * INLBF_NMM
        med = -0.876 * fml + 0.494 * fsi - 0.0189 * mvv
        lat = 0.876 * fml + 0.505 * fsi + 0.0189 * mvv
        tot = fsi
    elif subj == "SC":
        fx, fz = df["Fx"].to_numpy(float) * LBF_N, df["Fz"].to_numpy(float) * LBF_N
        ty = df["Ty"].to_numpy(float) * INLBF_NMM
        med = 0.942 * fx + 0.497 * fz + 0.0184 * ty
        lat = -0.942 * fx + 0.503 * fz - 0.0184 * ty
        tot = fz
    else:
        raise ValueError(f"unknown subject {subj!r}")
    return pd.DataFrame({"t": t, "medial_N": med, "lateral_N": lat, "total_N": tot})


@functools.lru_cache(maxsize=None)
def find_subject_dirs(root=GC_ROOT):
    """{subject: GC package folder}, found from the knee-force file prefixes (the folder names are not reliable)."""
    out = {}
    for d in sorted(glob.glob(os.path.join(root, "*", ""))):
        names = [os.path.basename(p).lower() for p in glob.glob(os.path.join(d, "Gait Trials", "*", "*_knee_forces.csv"))]
        hits = {s for s, c in SUBJECTS.items() for n in names if n.startswith(c["prefix"])}
        if len(hits) > 1:
            raise ValueError(f"{d}: knee-force files of several subjects {sorted(hits)}")
        if hits:
            s = hits.pop()
            if s in out:
                raise ValueError(f"subject {s} found in {out[s]} and {d}")
            out[s] = os.path.normpath(d)
    return out


@functools.lru_cache(maxsize=None)
def gc_trial_files(subj, root=GC_ROOT):
    """{trial key: {name, traj, grf, knee}} for one subject's gait trials that have both motion and knee files."""
    dirs = find_subject_dirs(root)
    if subj not in dirs:
        return {}
    d = dirs[subj]
    knee = {trial_key(p): p for p in glob.glob(os.path.join(d, "Gait Trials", "*", "*_knee_forces.csv"))}
    traj = {trial_key(p): p for p in glob.glob(os.path.join(d, "Gait Trials", "Video Motion Data", "*_trajectories.csv"))}
    out = {}
    for k in sorted(knee.keys() & traj.keys()):
        tp = traj[k]
        out[k] = dict(name=os.path.basename(tp)[:-len("_trajectories.csv")], traj=tp,
                      grf=tp[:-len("_trajectories.csv")] + "_grf.csv", knee=knee[k])
    return out


@functools.lru_cache(maxsize=None)
def load_gc_trial(subj, key, root=GC_ROOT):
    """One GC gait trial on its marker rows: time, measured forces (N), plate vertical GRF (N), foot markers (m, y up).

    The knee-force file shares the trajectories' time column row for row (the organisers synced them); if a file
    did not, it is interpolated onto the marker times and knee_time_equal is False.
    """
    f = gc_trial_files(subj, root)[key]
    traj = read_gc_csv(f["traj"])
    t = time_column(traj)
    meas = measured_forces(subj, read_gc_csv(f["knee"]))
    same = len(meas) == len(t) and bool(np.allclose(meas["t"].to_numpy(), t, atol=1e-3))
    cols = {}
    for c in ("medial_N", "lateral_N", "total_N"):
        v = meas[c].to_numpy(dtype=float)
        cols[c] = v if same else np.interp(t, meas["t"].to_numpy(), v)
    vgrf = np.zeros(len(t))
    if os.path.exists(f["grf"]):
        g = read_gc_csv(f["grf"])
        fz = [c for c in g.columns if re.fullmatch(r"Fz\d+", str(c))]
        vgrf = np.interp(t, time_column(g), np.abs(g[fz].to_numpy(dtype=float)).sum(axis=1))
    feet = {}
    for s in "RL":
        pts = []
        for m in ("Heel", "Toe"):
            xyz = traj[[f"{s}.{m}X", f"{s}.{m}Y", f"{s}.{m}Z"]].interpolate(limit_direction="both").to_numpy(float)
            pts.append(xyz[:, [0, 2, 1]] / 1000.0)        # lab Z is up -> y up, metres
        feet[s] = pts
    fs = 1.0 / float(np.median(np.diff(t)))
    return dict(name=f["name"], key=key, t=t, fs=fs, medial=cols["medial_N"], lateral=cols["lateral_N"],
                total=cols["total_N"], vgrf=vgrf, feet=feet, knee_time_equal=same)


@functools.lru_cache(maxsize=None)
def kinematic_stances(subj, key, root=GC_ROOT):
    """Instrumented-foot contact runs [(on_row, off_row)) over the whole GC trial, from heel/toe markers."""
    from imu.contact import detect_contact, _runs
    tr = load_gc_trial(subj, key, root)
    side = SUBJECTS[subj]["side"].upper()
    c = detect_contact(tr["feet"][side], tr["fs"])
    return tuple((int(a), int(b)) for a, b in _runs(c))


# ─── Time mapping and GRF cross-correlation alignment ────────────────────────

def seg_time_to_gc_row(t, row0, dt_b3d, t0=0.0):
    """GC marker row (fractional) of segment time t: frame = (t - t0) / dt_b3d, row = row0 + frame.

    dt_b3d is the AddBiomechanics timestep the segment's IK file was written with (0.2-0.3 % shorter than the true
    marker period), so segment time must go through the frame index, never be compared with GC time directly."""
    return row0 + (np.asarray(t, dtype=float) - t0) / dt_b3d


def gc_row_to_time(rows, t_gc):
    return np.interp(rows, np.arange(len(t_gc)), t_gc)


def sliding_corr(seg, ref):
    """Pearson r of seg (n) against every length-n window of ref; r[k] pairs seg[i] with ref[k + i]."""
    seg, ref = np.asarray(seg, float), np.asarray(ref, float)
    n, m = len(seg), len(ref)
    if n < 3 or m < n:
        return np.full(max(m - n + 1, 0), np.nan)
    w = np.lib.stride_tricks.sliding_window_view(ref, n)
    a = seg - seg.mean()
    wc = w - w.mean(axis=1, keepdims=True)
    den = np.sqrt((wc ** 2).sum(axis=1) * (a ** 2).sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (wc @ a) / den
    r[~(den > 1e-9)] = np.nan
    return r


def align_by_grf(seg_vgrf, trials):
    """Best GC trial and row for a segment's vertical GRF, by cross-correlation over all trials of the subject.

    trials: {key: per-row vertical GRF}. Returns dict(key, grf_row0, r_best, r_second, sharpness, ratio, rmse_N):
    grf_row0 is the GC row that segment frame 0 matches on the GRF clock; r_second is the best r of any other trial, or
    of the same trial more than half a segment away; sharpness = (1 - r_second) / (1 - r_best), the factor by which the
    runner-up leaves more GRF variance unexplained (smooth gait GRFs give r_second ~0.99, so r alone cannot tell)."""
    n = len(seg_vgrf)
    best, scores = None, {}
    for key, ref in trials.items():
        r = sliding_corr(seg_vgrf, ref)
        if len(r) == 0 or not np.isfinite(r).any():
            continue
        k = int(np.nanargmax(r))
        scores[key] = (k, r)
        if best is None or r[k] > best[2]:
            best = (key, k, float(r[k]))
    if best is None:
        return dict(key=None, grf_row0=np.nan, r_best=np.nan, r_second=np.nan, sharpness=np.nan, ratio=np.nan,
                    rmse_N=np.nan)
    key, k, rb = best
    second = -np.inf
    for kk, (kb, r) in scores.items():
        if kk != key:
            second = max(second, float(np.nanmax(r)))
        else:
            far = np.abs(np.arange(len(r)) - k) > max(5, n // 2)
            if far.any() and np.isfinite(r[far]).any():
                second = max(second, float(np.nanmax(r[far])))
    ref = trials[key][k:k + n]
    big = ref > 200.0
    ratio = float(np.median(seg_vgrf[big] / ref[big])) if big.any() else np.nan
    second = second if np.isfinite(second) else np.nan
    sharp = (1.0 - second) / max(1.0 - rb, 1e-12) if np.isfinite(second) else np.inf
    return dict(key=key, grf_row0=k, r_best=rb, r_second=second, sharpness=float(sharp), ratio=ratio,
                rmse_N=float(np.sqrt(np.mean((seg_vgrf - ref) ** 2))))


# ─── Segment folders ──────────────────────────────────────────────────────────

def segment_dirs(subj, repo=REPO):
    s = SUBJECTS[subj]
    pat = os.path.join(repo, "jcf", "full_duration", s["split"], "walking", f"fregly_{s['code']}_*")
    return [d for d in sorted(glob.glob(pat))
            if os.path.exists(os.path.join(d, "ik_results.mot")) and os.path.exists(os.path.join(d, "grf_data.mot"))]


def read_segment(d):
    """IK time grid, per-foot vertical GRF (N) on the IK frames, mass and height of one segment folder."""
    from train_cnn import load_mot
    meta = json.load(open(os.path.join(d, "metadata.json")))
    ik_t = load_mot(os.path.join(d, "ik_results.mot"))["time"].to_numpy(dtype=float)
    g = load_mot(os.path.join(d, "grf_data.mot"))
    gt = g["time"].to_numpy(dtype=float)
    vy = {s: np.interp(ik_t, gt, g[f"calcn_{s}_force_vy"].to_numpy(dtype=float)) for s in "rl"}
    dt = (ik_t[-1] - ik_t[0]) / max(len(ik_t) - 1, 1)
    return dict(folder=os.path.basename(d), dir=d, t=ik_t, dt=dt, n=len(ik_t), vy=vy,
                mass=float(meta["mass_kg"]), height=float(meta.get("height_m", 1.7)))


# ─── Exact maps from the b3d files (jcf env, nimblephysics) ──────────────────

def _norm_marker(n):
    return re.sub(r"[^a-z0-9]", "", str(n).lower())


def build_maps(out_root=OUT_ROOT, root=GC_ROOT, repo=REPO):
    """alignment.csv (b3d frame k -> GC row k + offset, from marker identity) and segment_map.csv (segment folder ->
    b3d trial and start frame, from IK identity). Needs nimblephysics (jcf env)."""
    sys.path.append(os.path.expanduser("~/.local/lib/python3.10/site-packages"))
    import nimblephysics as nimble
    from train_cnn import load_mot
    os.makedirs(out_root, exist_ok=True)
    al_rows, seg_rows = [], []
    for subj, s in SUBJECTS.items():
        b3d = os.path.join(repo, s["b3d"])
        if not os.path.exists(b3d):
            print(f"{subj}: no b3d at {b3d}")
            continue
        sub = nimble.biomechanics.SubjectOnDisk(b3d)
        files = gc_trial_files(subj, root)
        q_by_trial, names = {}, {}
        for t in range(sub.getNumTrials()):
            orig, L, dt = sub.getTrialOriginalName(t), sub.getTrialLength(t), sub.getTrialTimestep(t)
            fr = sub.readFrames(t, 0, L, includeSensorData=True, includeProcessingPasses=True)
            q_by_trial[t] = np.array([np.asarray(f.processingPasses[-1].pos) for f in fr])
            names[t] = orig
            row = dict(subject=subj, code=s["code"], trial=t, orig=orig, gc_key=trial_key(orig), n_b3d=L, dt_b3d=dt)
            if trial_key(orig) not in files:
                al_rows.append(dict(row, status="no_gc_trial"))
                continue
            traj = read_gc_csv(files[trial_key(orig)]["traj"])
            tg = time_column(traj)
            mk = {}
            for c in traj.columns:
                c = str(c)
                if c.endswith("X") and c[:-1] + "Y" in traj.columns and c[:-1] + "Z" in traj.columns:
                    mk[_norm_marker(c[:-1])] = traj[[c, c[:-1] + "Y", c[:-1] + "Z"]].to_numpy(float) / 1000.0
            common = sorted({_norm_marker(m[0]) for f in fr for m in f.markerObservations} & set(mk))
            B = np.full((L, len(common), 3), np.nan)
            for i, f in enumerate(fr):
                for nm, p in f.markerObservations:
                    k = _norm_marker(nm)
                    if k in mk:
                        B[i, common.index(k)] = np.asarray(p)
            Gm = np.stack([mk[n] for n in common], axis=1)
            best = (None, np.inf, 0)
            for off in range(-(L - 20), len(tg) - 20):                 # vertical: b3d y vs lab Z
                i0, i1 = max(0, -off), min(L, len(tg) - off)
                if i1 - i0 < 20:
                    continue
                e = np.nanmedian(np.abs(B[i0:i1, :, 1] - Gm[i0 + off:i1 + off, :, 2]))
                if e < best[1]:
                    best = (off, e, i1 - i0)
            off, e, nov = best
            i0, i1 = max(0, -off), min(L, len(tg) - off)
            ax = s["lab_axes"]
            lab = {"X": 0, "Y": 1}
            ex = np.nanmedian(np.abs(B[i0:i1, :, 0] - ax["x"][1] * Gm[i0 + off:i1 + off, :, lab[ax["x"][0]]]))
            ez = np.nanmedian(np.abs(B[i0:i1, :, 2] - ax["z"][1] * Gm[i0 + off:i1 + off, :, lab[ax["z"][0]]]))
            kt = time_column(read_gc_csv(files[trial_key(orig)]["knee"]))
            al_rows.append(dict(row, status="ok", n_gc=len(tg), gc_t0=tg[0], offset=int(off), n_overlap=int(nov),
                                vert_err_mm=1000 * e, x_err_mm=1000 * ex, z_err_mm=1000 * ez, n_markers=len(common),
                                knee_time_eq_traj=bool(len(kt) == len(tg) and np.allclose(kt, tg, atol=1e-3))))
            print(f"{subj} {orig}: offset {off}, residual {1000 * e:.3f} mm ({len(common)} markers)", flush=True)
        offsets = {r["trial"]: r.get("offset") for r in al_rows if r["subject"] == subj and r["status"] == "ok"}
        seen = {}
        for d in segment_dirs(subj, repo):
            m = load_mot(os.path.join(d, "ik_results.mot")).drop(columns=["time"]).to_numpy(float)
            hits = []
            for t, q in q_by_trial.items():
                if q.shape[1] != m.shape[1] or len(q) < len(m):
                    continue
                first = np.abs(q - m[0]).max(axis=1)
                for sf in np.flatnonzero(first < 1e-6):
                    if sf + len(m) <= len(q) and np.abs(q[sf:sf + len(m)] - m).max() < 1e-6:
                        hits.append((t, int(sf)))
            row = dict(subject=subj, folder=os.path.basename(d), n_ik=len(m), n_hits=len(hits))
            if len(hits) == 1:
                t, sf = hits[0]
                off = offsets.get(t)
                ident = (t, sf, len(m))
                row.update(trial=t, orig=names[t], gc_key=trial_key(names[t]), start_frame=sf,
                           gc_row_start=None if off is None else sf + off,
                           gc_row_end=None if off is None else sf + off + len(m) - 1,
                           duplicate_of=seen.get(ident))
                seen.setdefault(ident, row["folder"])
            seg_rows.append(row)
    al, sm = pd.DataFrame(al_rows), pd.DataFrame(seg_rows)
    al.to_csv(os.path.join(out_root, "alignment.csv"), index=False)
    sm.to_csv(os.path.join(out_root, "segment_map.csv"), index=False)
    print(f"wrote {os.path.join(out_root, 'alignment.csv')} ({len(al)} trials) and segment_map.csv ({len(sm)} folders)")
    return al, sm


def read_maps(out_root=OUT_ROOT):
    """(alignment, segment map) from build_maps, or (None, None) if they have not been built."""
    pa, ps = os.path.join(out_root, "alignment.csv"), os.path.join(out_root, "segment_map.csv")
    if not (os.path.exists(pa) and os.path.exists(ps)):
        return None, None
    return pd.read_csv(pa), pd.read_csv(ps)


# ─── Surrogate inputs (label-free copy of train_mlp._build_com_features) ──────

def build_frame_features(d, cfg, grid="jr"):
    """Per-frame surrogate inputs for one segment folder, built exactly as train_mlp._build_com_features builds them,
    without its label filters (it drops trials whose labels fail the |M_x| / peak filters, which says nothing about the
    inputs).

    grid="jr": the IK frames inside the IK, GRF and JR time ranges (the frames training used, and the frames our OpenSim
    labels exist on). grid="ik": every IK frame inside the GRF range (no JR file needed).
    Returns (feat [T, per_frame_ch] float32, meta) or None; meta["t"] is the time of each row (the q / contact / label
    instants), meta["t_common"] the native IK frames before any resampling."""
    from train_cnn import load_mot, N_LOWER_BODY, PELVIS_TX_COLS, DEAD_JOINT_COLS
    from train_mlp import _load_com_features_aligned, _sto_time, _resample, CONTACT_GRF_BW_THRESHOLD
    try:
        meta = json.load(open(os.path.join(d, "metadata.json")))
        mass, height = float(meta["mass_kg"]), float(meta.get("height_m", 1.7))
    except Exception:
        return None
    if height <= 0 or not cfg.get("uses_com"):
        return None
    bw = mass * G
    ik = load_mot(os.path.join(d, "ik_results.mot"))
    ik_time = ik["time"].to_numpy()
    ik_data = ik.drop(columns=["time"]).to_numpy()
    grf = load_mot(os.path.join(d, "grf_data.mot"))
    force_cols = [c for c in grf.columns if ("calcn_r" in c or "calcn_l" in c) and "_force_v" in c]
    if len(force_cols) != 6:
        return None
    grf_time, grf_data = grf["time"].to_numpy(), grf[force_cols].to_numpy() / bw
    t0, t1 = max(ik_time[0], grf_time[0]), min(ik_time[-1], grf_time[-1])
    if grid == "jr":
        jr = os.path.join(d, "jcf_output", "BatchJCF_JointReaction_ReactionLoads.sto")
        if not os.path.exists(jr):
            return None
        jr_t = _sto_time(jr)
        t0, t1 = max(t0, jr_t[0]), min(t1, jr_t[-1])
    elif grid != "ik":
        raise ValueError(f"grid must be 'jr' or 'ik', got {grid!r}")
    mask = (ik_time >= t0) & (ik_time <= t1)
    t_common = ik_time[mask]
    T = len(t_common)
    if T < 1:
        return None
    grf_i = np.column_stack([np.interp(t_common, grf_time, grf_data[:, j]) for j in range(grf_data.shape[1])])
    q = ik_data[mask][:, :N_LOWER_BODY].copy()
    q[:, PELVIS_TX_COLS] -= q[:, PELVIS_TX_COLS].mean(axis=0)            # per-trial centering, as load_subject
    q = q[:, [j for j in range(q.shape[1]) if j not in DEAD_JOINT_COLS]]
    if cfg.get("time_exact"):
        t_target = t_common
    else:                                     # legacy sets: positions on a full-segment grid (see train_mlp INPUT_SETS)
        t_target = np.linspace(ik_time[0], ik_time[-1], T).astype(np.float64)
    com = _load_com_features_aligned(d, t_target, foot_point=cfg.get("foot_point"))
    if com is None:
        return None
    if cfg.get("uses_pelvis_rel"):
        pelvis = np.column_stack([np.interp(t_target, ik_time, ik[c].to_numpy())
                                  for c in ("pelvis_tx", "pelvis_ty", "pelvis_tz")]).astype(np.float32)
        if cfg.get("pelvis_per_window_centering"):
            q[:, 3:6] = pelvis
        parts = [q, com["com_pos"] - pelvis, com["calcn_r_pos"] - pelvis, com["calcn_l_pos"] - pelvis]
    else:
        parts = [q, com["com_vel"], com["com_pos"] - com["calcn_r_pos"], com["com_pos"] - com["calcn_l_pos"]]
    fy = grf_i[:, [1, 4]]
    t_rows = t_common
    hz = cfg.get("resample_hz")
    if hz:
        t_new = t_target[0] + np.arange(int(np.floor((t_target[-1] - t_target[0]) * hz + 1e-9)) + 1) / hz
        parts = [_resample(t_target, np.asarray(x, dtype=np.float64), t_new) for x in parts]
        fy = _resample(t_target, fy.astype(np.float64), t_new)
        t_rows, T = t_new, len(t_new)
    r_c = np.abs(fy[:, 0]) > CONTACT_GRF_BW_THRESHOLD
    l_c = np.abs(fy[:, 1]) > CONTACT_GRF_BW_THRESHOLD
    if cfg.get("include_contact"):
        parts += [r_c.astype(np.float32)[:, None], l_c.astype(np.float32)[:, None]]
    if cfg["include_height"]:
        parts += [np.full((T, 1), mass, dtype=np.float32), np.full((T, 1), height, dtype=np.float32)]
    if cfg.get("drop_pelvis_translation"):
        parts[0] = np.delete(parts[0], [3, 4, 5], axis=1)
    feat = np.hstack(parts).astype(np.float32)
    if feat.shape[1] != cfg["per_frame_ch"]:
        raise ValueError(f"feature width {feat.shape[1]} != {cfg['per_frame_ch']}")
    return feat, dict(t=np.asarray(t_rows, dtype=np.float64), t_common=t_common, r_contact=r_c, l_contact=l_c,
                      fs=float(hz) if hz else None, mass=mass, height=height, ik_t0=float(ik_time[0]))


def _q_layout(cfg):
    """Index of each named coordinate in the per-frame q block of an input set."""
    from inference_posture_surrogate import JOINT_ORDER
    names = list(JOINT_ORDER)
    if cfg.get("drop_pelvis_translation"):
        names = [n for n in names if n not in ("pelvis_tx", "pelvis_ty", "pelvis_tz")]
    if len(names) != cfg.get("q_dim", 16):
        raise ValueError(f"q layout {len(names)} != q_dim {cfg.get('q_dim', 16)}")
    return {n: i for i, n in enumerate(names)}


def mirror_per_frame(feat, cfg):
    """Reflect per-frame surrogate inputs about the sagittal (x-y) plane: the person's left side becomes the right.

    Derivation (OpenSim ground x forward, y up, z right; reflection S = diag(1, 1, -1); tests/test_grand_challenge.py):
    - positions: z -> -z for every point; the right and left foot exchange places.
    - pelvis (ground_pelvis = Rz(tilt) Rx(list) Ry(rotation)): S Rz(a) S = Rz(a), S Rx(b) S = Rx(-b), S Ry(c) S = Ry(-c),
      so tilt stays and list and rotation change sign; pelvis_tz changes sign, tx and ty stay.
    - hips: Rajagopal's hip_l uses the hip_r axes with coefficient -1 on adduction and rotation, i.e.
      H_l(f, a, r) = Rz(f) Rx(-a) Ry(-r) = S H_r(f, a, r) S. Knee and ankle are mirrored in the model too, so a
      mirrored right-side coordinate is the left-side coordinate with the same value: swap r and l, keep the signs.
    - contact flags swap; mass and height stay.
    The output's knee is the input's left knee, with F_y (axial) unchanged and M_x in its medial-positive sense.
    """
    f = np.array(feat, dtype=np.float32, copy=True)
    src = np.asarray(feat)
    qi = _q_layout(cfg)
    for n in ("hip_flexion", "hip_adduction", "hip_rotation", "knee_angle", "ankle_angle"):
        f[:, qi[n + "_r"]], f[:, qi[n + "_l"]] = src[:, qi[n + "_l"]], src[:, qi[n + "_r"]]
    for n in ("pelvis_list", "pelvis_rotation", "pelvis_tz"):
        if n in qi:
            f[:, qi[n]] = -src[:, qi[n]]
    p = cfg.get("q_dim", 16)                     # three xyz triplets: COM (or v_com), right foot, left foot
    for k in range(3):
        f[:, p + 3 * k + 2] = -src[:, p + 3 * k + 2]
    f[:, p + 3:p + 6], f[:, p + 6:p + 9] = f[:, p + 6:p + 9].copy(), f[:, p + 3:p + 6].copy()
    if cfg.get("include_contact"):
        f[:, p + 9], f[:, p + 10] = src[:, p + 10], src[:, p + 9]
    return f


def windowize(feat, cfg):
    from train_mlp import _windowize_positions_only, _windowize
    w = cfg.get("window", 1)
    if w <= 1:
        return feat
    if cfg.get("window_only_positions"):
        return _windowize_positions_only(feat, w, q_dim=cfg.get("q_dim", 16), n_pos=cfg.get("n_pos", 9),
                                         pelvis_t_cols=(3, 4, 5) if cfg.get("pelvis_per_window_centering") else None,
                                         heading_norm=cfg.get("heading_norm", False))
    return _windowize(feat, w, replicate_q=cfg.get("replicate_q", False))


def jr_labels(d, meta, joint="walker_knee_r", subdir="jcf_output", mx_sign=1.0, max_peak=6.0, max_mx=0.15):
    """OpenSim JR knee loads (F_x, F_y, F_z in BW, mx_sign * M_x in BW*H) from
    <d>/<subdir>/BatchJCF_JointReaction_ReactionLoads.sto. mx_sign = 1 gives the raw moment (the training labels);
    medial_mx_sign(joint, convention) makes it medial-positive. Read on the rows of meta, interpolated as load_subject
    does (onto the IK frames, then resampled). Returns (labels or None, info): info = status, label_ok (the training label filters
    evaluated on the whole JR file) and filter (the reasons it fails them)."""
    from train_mlp import _resample
    p = os.path.join(d, subdir, "BatchJCF_JointReaction_ReactionLoads.sto")
    if not os.path.exists(p):
        return None, dict(status="missing")
    from tools.evaluate_fmedial import _read_sto
    j = _read_sto(p)
    pre = f"{joint}_on_{joint.replace('walker_knee', 'tibia')}_in_{joint.replace('walker_knee', 'tibia')}_"
    if pre + "fy" not in j.columns:
        return None, dict(status=f"no {joint} columns")
    bw, H = meta["mass"] * G, meta["height"]
    jt = j["time"].to_numpy(dtype=float)
    vals = np.column_stack([j[pre + "fx"] / bw, j[pre + "fy"] / bw, j[pre + "fz"] / bw,
                            mx_sign * j[pre + "mx"] / (bw * H)]).astype(np.float64)
    res = np.sqrt((vals[:, :3] ** 2).sum(axis=1))
    reasons = []
    if np.abs(vals[:, 3]).max() > max_mx:
        reasons.append(f"|Mx| peak {np.abs(vals[:, 3]).max():.2f} BW*H > {max_mx:g}")
    if res.max() > max_peak:
        reasons.append(f"resultant peak {res.max():.1f} BW > {max_peak:g}")
    lab = np.column_stack([np.interp(meta["t_common"], jt, vals[:, k]) for k in range(4)])
    if not np.array_equal(meta["t"], meta["t_common"]):
        lab = _resample(meta["t_common"], lab, meta["t"]).astype(np.float64)
    lab[(meta["t"] < jt[0] - 1e-9) | (meta["t"] > jt[-1] + 1e-9)] = np.nan
    return lab, dict(status="ok", label_ok=not reasons, filter="; ".join(reasons))


# ─── Stance events and peaks ──────────────────────────────────────────────────

def grf_events(rows, on):
    """Contact runs of a segment on GC rows: [(first contact row, first row after contact, hs seen, to seen)]."""
    from imu.contact import _runs
    out, n = [], len(on)
    for a, b in _runs(np.asarray(on, bool)):
        out.append((float(rows[a]), float(rows[b]) if b < n else float(rows[b - 1] + 1), a > 0, b < n))
    return out


def event_bias(events, kin_runs, fs, max_s=0.2):
    """Kinematic minus force-plate event time (rows) for every observed heel strike and toe-off."""
    on_b, off_b = [], []
    kon = np.array([a for a, _ in kin_runs], float)
    koff = np.array([b for _, b in kin_runs], float)
    for hs, to, hs_seen, to_seen in events:
        if hs_seen and len(kon):
            d = kon - hs
            k = int(np.argmin(np.abs(d)))
            if abs(d[k]) <= max_s * fs:
                on_b.append(d[k])
        if to_seen and len(koff):
            d = koff - to
            k = int(np.argmin(np.abs(d)))
            if abs(d[k]) <= max_s * fs:
                off_b.append(d[k])
    return on_b, off_b


def complete_stances(events, kin_runs, bias_on, bias_off, fs, dur_s=(0.3, 1.6)):
    """Full stance [hs, to) for each contact run of a segment: an event seen in the segment's GRF is used as is; an
    event outside the segment comes from the overlapping kinematic stance, bias-corrected. Runs without a usable
    kinematic stance, or with an implausible duration, are dropped."""
    out = []
    for hs, to, hs_seen, to_seen in events:
        src_h, src_t = "grf", "grf"
        if not (hs_seen and to_seen):
            ov = [(min(b, to) - max(a, hs), a, b) for a, b in kin_runs if min(b, to) > max(a, hs)]
            if not ov:
                continue
            _, a, b = max(ov)
            if not hs_seen:
                hs, src_h = min(a - bias_on, hs), "kin"
            if not to_seen:
                to, src_t = max(b - bias_off, to), "kin"
        if dur_s[0] <= (to - hs) / fs <= dur_s[1]:
            out.append(dict(hs=hs, to=to, hs_src=src_h, to_src=src_t))
    return out


def stance_halves(rows, gate, stances, min_cover=0.9, windows=PEAK_WINDOWS):
    """Assign eval-grid frames to the stance peak windows (default early = 5-45 %, late = 55-95 % of stance).

    Returns (stance_id, pct, part) per frame (part = the window a frame falls in, '' outside both) and one dict per
    stance x window with its coverage (fraction of the window's duration spanned by gated frames) and `scored`
    (cover >= min_cover). Whether its peak can be scored also needs window_peak (interior maximum)."""
    rows = np.asarray(rows, float)
    sid = np.full(len(rows), -1)
    pct = np.full(len(rows), np.nan)
    part = np.full(len(rows), "", dtype=object)
    step = float(np.median(np.diff(rows))) if len(rows) > 1 else 1.0
    halves = []
    for i, s in enumerate(stances):
        dur = s["to"] - s["hs"]
        p = 100.0 * (rows - s["hs"]) / dur
        inside = (p >= 0) & (p < 100) & np.asarray(gate, bool)
        sid[inside], pct[inside] = i, p[inside]
        for name, lo, hi in windows:
            m = inside & (p >= lo) & (p <= hi)
            part[m] = name
            cover = 0.0
            if m.any():
                cover = min(1.0, (rows[m].max() - rows[m].min() + step) / ((hi - lo) / 100.0 * dur))
            halves.append(dict(stance=i, part=name, win_lo=lo, win_hi=hi, cover=cover, scored=cover >= min_cover,
                               mask=m, **s))
    return sid, pct, part, halves


def window_peak(values, pct, margin=3.0):
    """Maximum of `values` over one stance window (frames in time order, pct = their % of stance) and whether it is an
    interior local maximum: at least `margin` % of stance away from the first and the last frame of the window.

    A maximum at the window's edge is the tail of a peak outside the window (e.g. the first peak's descent at the start
    of the late window), not a peak of its own. Returns (index, value, pct of the maximum, interior)."""
    v, p = np.asarray(values, float), np.asarray(pct, float)
    if not len(v) or not np.isfinite(v).any():
        return None, np.nan, np.nan, False
    k = int(np.nanargmax(v))
    interior = bool(p[k] - p[0] >= margin and p[-1] - p[k] >= margin)
    return k, float(v[k]), float(p[k]), interior


# ─── StatOp_Val (Kaneda et al. 2023) ──────────────────────────────────────────

def _read_ws_sto(path):
    with open(path) as f:
        for i, line in enumerate(f):
            if line.strip().lower() == "endheader":
                break
    return pd.read_csv(path, sep=r"\s+", skiprows=i + 1)


def statop_windows(root=GC_ROOT, fs=120.0):
    """Kaneda et al.'s SO+JR medial / lateral compartment force for SC's left knee: one dict per half-stance window.

    StatOp kinematic time tau maps to GC marker row 1 + round(120 tau) (verified against the GC markers on all 62
    IK runs; results/thesis/gc/EXPLORE_NOTES.md section 6). Forces are negative for compression; returned as |.| in N."""
    out = []
    for d in sorted(glob.glob(os.path.join(root, "StatOp_Val_Data", "3GC", "SO_JRA", "SC_*_stance*"))):
        p = os.path.join(d, "instKnee", "results_JointReaction_JointRxn_ReactionLoads.sto")
        if not os.path.exists(p):
            continue
        j = _read_ws_sto(p)
        if not all(c in j.columns for c in STATOP_COLS):
            continue
        tag = os.path.basename(d)
        tau = j["time"].to_numpy(dtype=float)
        out.append(dict(tag=tag, gc_key=trial_key(tag.rsplit("_stance", 1)[0]), tau=tau,
                        rows=1 + np.rint(tau * fs).astype(int),
                        medial_N=np.abs(j[STATOP_COLS[0]].to_numpy(float)),
                        lateral_N=np.abs(j[STATOP_COLS[1]].to_numpy(float))))
    return out


# ─── Metrics ──────────────────────────────────────────────────────────────────

def fmed_of(df, src, phi=0.5, d_mm=45.0):
    """F_medial (BW) of a source from the frame table; StatOp carries F_medial directly."""
    if f"fy_{src}" not in df:
        return df[f"fmed_{src}"].to_numpy(dtype=float)
    return phi * np.abs(df[f"fy_{src}"].to_numpy(float)) + df[f"mx_{src}"].to_numpy(float) * df["H"].to_numpy(float) / (d_mm / 1000.0)


def _metrics(g, p):
    from tools.evaluate_fmedial import basic_metrics
    g, p = np.asarray(g, float), np.asarray(p, float)
    ok = np.isfinite(g) & np.isfinite(p)
    return basic_metrics(g[ok], p[ok])


def frame_summary(fr, sources, phi, d_mm, by=("subject",)):
    rows = []
    for key, g in fr[fr["stance"]].groupby(list(by)):
        key = key if isinstance(key, tuple) else (key,)
        for src in sources:
            if f"fmed_{src}" not in g and f"fy_{src}" not in g:
                continue
            fm = fmed_of(g, src, phi, d_mm)
            ok = np.isfinite(fm)
            if not ok.any():
                continue
            m = _metrics(g["meas_med"].to_numpy()[ok], fm[ok])
            row = dict(zip(by, key), source=src, n_frames=m["n"], n_segments=g.loc[ok, "folder"].nunique(),
                       n_trials=g.loc[ok, "gc_key"].nunique(), mae=m["mae"], rmse=m["rmse"], bias=m["bias"], r=m["r"],
                       meas_mean=float(g["meas_med"].to_numpy()[ok].mean()), pred_mean=float(fm[ok].mean()))
            if f"fy_{src}" in g:
                t = _metrics(g["meas_tot"].to_numpy()[ok], np.abs(g[f"fy_{src}"].to_numpy())[ok])
                row.update(total_mae=t["mae"], total_bias=t["bias"], total_r=t["r"])
            rows.append(row)
    return pd.DataFrame(rows)


def implied_phi_d(fr, sources):
    """Least squares meas_med = phi |F_y| + k M_x (stance frames, no intercept); d = H / k."""
    rows = []
    for subj, g in fr[fr["stance"]].groupby("subject"):
        H = float(g["H"].iloc[0])
        for src in sources:
            if f"fy_{src}" not in g:
                continue
            ok = np.isfinite(g[f"fy_{src}"]) & np.isfinite(g[f"mx_{src}"])
            if ok.sum() < 10:
                continue
            X = np.column_stack([np.abs(g.loc[ok, f"fy_{src}"]), g.loc[ok, f"mx_{src}"]])
            y = g.loc[ok, "meas_med"].to_numpy()
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
            res = y - X @ coef
            rows.append(dict(subject=subj, source=src, n_frames=int(ok.sum()), phi=coef[0],
                             d_mm=1000 * H / coef[1] if coef[1] > 0 else np.nan, k_moment=coef[1],
                             rmse_fit=float(np.sqrt(np.mean(res ** 2))),
                             r2=float(1 - res.var() / y.var()) if y.var() > 0 else np.nan))
    return pd.DataFrame(rows)


def osim_joint_sign(inst, convention="physical"):
    """(JR joint, factor making its raw mx medial-positive) for the instrumented side."""
    joint = "walker_knee_r" if inst == "r" else "walker_knee_l"
    return joint, medial_mx_sign(joint, convention)


def surrogate_mx_sign(convention="physical"):
    """Factor making a surrogate's M_x output medial-positive. The network predicts the raw walker_knee_r mx of its
    training labels; a left knee is fed mirrored, so its output is a right knee too."""
    return medial_mx_sign("walker_knee_r", convention)


def mx_sign_check(fr, sources, d_mm=45.0):
    """Does each source's medial-positive M_x follow the implant? Stance frames, per subject x source: r of M_x with the
    measured medial - lateral force, the share of frames with a medial moment (M_x > 0), the mean moment term
    M_x*H/d (BW), and the medial share mean(F_medial) / mean(|F_y|) with phi = 0.5, against the measured
    mean(medial) / mean(total)."""
    rows = []
    for subj, g in fr[fr["stance"].astype(bool)].groupby("subject"):
        dl = (g["meas_med"] - g["meas_lat"]).to_numpy(float)
        for src in sources:
            if f"mx_{src}" not in g:
                continue
            mx, fy = g[f"mx_{src}"].to_numpy(float), g[f"fy_{src}"].to_numpy(float)
            ok = np.isfinite(mx) & np.isfinite(fy) & np.isfinite(dl)
            if ok.sum() < 10:
                continue
            term = mx[ok] * g["H"].to_numpy(float)[ok] / (d_mm / 1000.0)
            afy = np.abs(fy[ok])
            rows.append(dict(subject=subj, source=src, n_frames=int(ok.sum()),
                             r_mx_vs_meas_med_minus_lat=float(np.corrcoef(mx[ok], dl[ok])[0, 1]),
                             medial_moment_frac=float(np.mean(mx[ok] > 0)), moment_term_mean_bw=float(term.mean()),
                             medial_share=float((0.5 * afy + term).mean() / afy.mean()),
                             meas_medial_share=float(g["meas_med"].to_numpy(float)[ok].mean() /
                                                     g["meas_tot"].to_numpy(float)[ok].mean())))
    return pd.DataFrame(rows)


# ─── Main ─────────────────────────────────────────────────────────────────────

def process_subject(subj, models, cfgs, args, seg_map, root):
    """Frame table, segment table, stance halves and StatOp rows for one subject."""
    import torch
    s = SUBJECTS[subj]
    files = gc_trial_files(subj, root)
    if not files:
        print(f"{subj}: no GC files under {root}")
        return None
    trials = {k: load_gc_trial(subj, k, root) for k in files}
    vgrf = {k: tr["vgrf"] for k, tr in trials.items()}
    ident = {}
    if seg_map is not None:
        for _, r in seg_map[(seg_map.subject == subj) & seg_map.gc_row_start.notna()].iterrows():
            ident[r.folder] = (r.gc_key, int(r.gc_row_start), r.get("duplicate_of"))
    dirs = segment_dirs(subj)[:args.max_segments] if args.max_segments else segment_dirs(subj)
    delay = s["grf_delay_frames"]
    seg_rows, frames, events_all, seg_data = [], [], [], []
    seen = {}
    for d in dirs:
        seg = read_segment(d)
        bw = seg["mass"] * G
        v = seg["vy"]["r"] + seg["vy"]["l"]
        a = align_by_grf(v, vgrf)
        row = dict(subject=subj, folder=seg["folder"], n_ik=seg["n"], dt_b3d=seg["dt"], gc_key=a["key"],
                   trial=trials[a["key"]]["name"] if a["key"] else None, grf_row0=a["grf_row0"],
                   r_best=a["r_best"], r_second=a["r_second"], sharpness=a["sharpness"], grf_ratio=a["ratio"],
                   grf_rmse_N=a["rmse_N"], grf_rmse_bw=a["rmse_N"] / bw,
                   row0_grf_xcorr=a["grf_row0"] + delay if a["key"] else np.nan)
        idn = ident.get(seg["folder"])
        status = "ok"
        if a["key"] is None or not (a["r_best"] >= args.min_r):
            status = "poor_r"
        elif not (0.9 <= a["ratio"] <= 1.1):
            status = "grf_scale"
        if idn is not None:
            row.update(identity_key=idn[0], row0_identity=idn[1],
                       grf_lag_frames=a["grf_row0"] - idn[1] if idn[0] == a["key"] else np.nan)
            if status == "ok" and (idn[0] != a["key"] or abs(a["grf_row0"] + delay - idn[1]) > 1):
                status = "identity_mismatch"
            if isinstance(idn[2], str) and idn[2]:
                status = "duplicate" if status == "ok" else status
                row["duplicate_of"] = idn[2]
        elif status == "ok" and not (a["sharpness"] >= args.min_sharpness):
            status = "ambiguous"
        row0 = idn[1] if (idn is not None and status in ("ok", "duplicate")) else row["row0_grf_xcorr"]
        sig = (a["key"], row0, seg["n"])
        if status == "ok" and sig in seen:
            status, row["duplicate_of"] = "duplicate", seen[sig]
        seen.setdefault(sig, seg["folder"])
        row.update(row0=row0, row0_source="identity" if (idn is not None and status in ("ok", "duplicate")) else "grf_xcorr",
                   status=status, gait=gait_code(row["trial"]) if row["trial"] else None)
        seg_rows.append(row)
        if status != "ok":
            continue
        tr = trials[a["key"]]
        inst = s["side"]
        # instrumented-foot contact on the GC GRF clock (AddB's GRF is `delay` frames late against the kinematics)
        rows_native = row0 + np.arange(seg["n"]) - delay
        on_native = np.abs(seg["vy"][inst]) > CONTACT_BW * bw
        ev = grf_events(rows_native, on_native)
        events_all.append((a["key"], ev))
        # eval grid: deploy-style 100 Hz rows inside the JR window
        ecfg = dict(cfgs[0], time_exact=True, resample_hz=100.0)
        built = build_frame_features(d, ecfg, grid=args.grid)
        if built is None:
            row["status"] = "no_features"
            continue
        _, emeta = built
        rows = seg_time_to_gc_row(emeta["t"], row0, seg["dt"], emeta["ik_t0"])
        n_gc = len(tr["t"])
        inb = (rows >= 0) & (rows <= n_gc - 1)
        idx = np.arange(n_gc)
        fr = pd.DataFrame(dict(subject=subj, folder=seg["folder"], gc_key=a["key"], trial=tr["name"], gait=row["gait"],
                               k=np.arange(len(rows)), t_seg=emeta["t"], row=rows, gc_time=gc_row_to_time(rows, tr["t"]),
                               H=emeta["height"], mass=emeta["mass"],
                               meas_med=np.interp(rows, idx, tr["medial"]) / bw,
                               meas_lat=np.interp(rows, idx, tr["lateral"]) / bw,
                               meas_tot=np.interp(rows, idx, tr["total"]) / bw))
        gate = np.interp(rows, rows_native, np.abs(seg["vy"][inst])) > CONTACT_BW * bw
        fr["stance"] = gate & inb
        fr["contact_input"] = emeta["l_contact" if inst == "l" else "r_contact"]
        for (m, mu, sd, info), cfg, lab in zip(models, cfgs, args.labels):
            mb = build_frame_features(d, cfg, grid=args.context)
            if mb is None:
                fr[f"fy_{lab}"] = fr[f"mx_{lab}"] = np.nan
                continue
            feat, mmeta = mb
            if inst == "l":
                feat = mirror_per_frame(feat, cfg)
            x = torch.from_numpy(windowize(feat, cfg)).float()
            with torch.no_grad():
                pr = m((x - mu) / sd).numpy().astype(np.float64)
            if not (len(mmeta["t"]) == len(emeta["t"]) and np.allclose(mmeta["t"], emeta["t"])):
                pr = np.column_stack([np.interp(emeta["t"], mmeta["t"], pr[:, j], left=np.nan, right=np.nan)
                                      for j in range(pr.shape[1])])
            fr[f"fy_{lab}"], fr[f"mx_{lab}"] = pr[:, 1], surrogate_mx_sign(args.mx_convention) * pr[:, 3]
        joint, sign = osim_joint_sign(inst, args.mx_convention)
        st = fr["stance"].to_numpy()
        for src, (sub, _) in OSIM_SOURCES.items():
            lab_o, oinfo = jr_labels(d, emeta, joint, sub, sign, max_peak=args.max_peak)
            row.update({f"{src}_{k}": v for k, v in oinfo.items()})
            fr[f"fy_{src}"] = lab_o[:, 1] if lab_o is not None else np.nan
            fr[f"mx_{src}"] = lab_o[:, 3] if lab_o is not None else np.nan
            fr[f"{src}_ok"] = bool(oinfo.get("label_ok", False))
            if lab_o is not None:                    # medial-positive M_x (run's convention) and compressive F_y
                row[f"{src}_mx_pos_frac_stance"] = float(np.mean(lab_o[st, 3] > 0)) if st.any() else np.nan
                row[f"{src}_fy_neg_frac_stance"] = float(np.mean(lab_o[st, 1] < 0)) if st.any() else np.nan
        row.update(n_eval=len(fr), n_stance=int(fr["stance"].sum()))
        frames.append(fr)
        seg_data.append((row, fr, ev))
    seg_df = pd.DataFrame(seg_rows)
    if not frames:
        return dict(segments=seg_df, frames=pd.DataFrame(), halves=[], bias=None, statop=None)
    # kinematic stance-event bias from every observed force-plate edge of this subject
    on_b, off_b = [], []
    for key, ev in events_all:
        o, f_ = event_bias(ev, kinematic_stances(subj, key, root), s["fs_gc"])
        on_b += o
        off_b += f_
    bias_on = float(np.median(on_b)) if on_b else 0.0
    bias_off = float(np.median(off_b)) if off_b else 0.0
    bias = dict(subject=subj, n_onsets=len(on_b), onset_bias_ms=1000 * bias_on / s["fs_gc"],
                onset_iqr_ms=1000 * float(np.subtract(*np.percentile(on_b, [75, 25]))) / s["fs_gc"] if on_b else np.nan,
                n_offsets=len(off_b), offset_bias_ms=1000 * bias_off / s["fs_gc"],
                offset_iqr_ms=1000 * float(np.subtract(*np.percentile(off_b, [75, 25]))) / s["fs_gc"] if off_b else np.nan)
    halves, offset = [], 0
    for row, fr, ev in seg_data:                    # same order as `frames`, so offset indexes the concatenation
        stances = complete_stances(ev, kinematic_stances(subj, row["gc_key"], root), bias_on, bias_off, s["fs_gc"])
        sid, pct, part, hv = stance_halves(fr["row"].to_numpy(), fr["stance"].to_numpy(), stances, args.min_cover)
        fr["stance_id"], fr["pct"], fr["part"] = sid, pct, part
        for h in hv:
            halves.append(dict(h, subject=subj, folder=row["folder"], gc_key=row["gc_key"], trial=row["trial"],
                               gait=row["gait"], frame_index=offset + np.flatnonzero(h["mask"])))
        row["n_windows_covered"] = int(sum(h["scored"] for h in hv))
        offset += len(fr)
    seg_df = pd.DataFrame(seg_rows)
    fr_all = pd.concat(frames, ignore_index=True)
    statop = None
    if subj == "SC" and not args.no_statop:
        statop = statop_rows(subj, trials, fr_all, bias_on, bias_off, args, root)
    return dict(segments=seg_df, frames=fr_all, halves=halves, bias=bias, statop=statop)


def statop_rows(subj, trials, fr_all, bias_on, bias_off, args, root):
    """Kaneda's windows vs measured (own frames), their stance halves, and StatOp on our segments' frames."""
    s = SUBJECTS[subj]
    wins = statop_windows(root, s["fs_gc"])
    if not wins:
        return None
    rows, halves = [], []
    fr_all["fmed_statop"] = np.nan
    for w in wins:
        if w["gc_key"] not in trials:
            continue
        tr = trials[w["gc_key"]]
        bw = float(fr_all["mass"].iloc[0]) * G       # AddBiomechanics mass, as for every other source
        ok = (w["rows"] >= 0) & (w["rows"] < len(tr["t"]))
        r_ = w["rows"][ok]
        meas = tr["medial"][r_] / bw
        so = w["medial_N"][ok] / bw
        m = _metrics(meas, so)
        rows.append(dict(tag=w["tag"], gc_key=w["gc_key"], trial=tr["name"], gait=gait_code(tr["name"]),
                         row_start=int(r_.min()), row_end=int(r_.max()), n=m["n"], mae=m["mae"], rmse=m["rmse"],
                         bias=m["bias"], r=m["r"], meas_peak=float(meas.max()), statop_peak=float(so.max())))
        kin = kinematic_stances(subj, w["gc_key"], root)
        st = [dict(hs=a - bias_on, to=b - bias_off, hs_src="kin", to_src="kin") for a, b in kin
              if b > r_.min() and a < r_.max()]
        _, pct, _, hv = stance_halves(r_.astype(float), np.ones(len(r_), bool), st, args.min_cover)
        for h in hv:
            if h["scored"]:
                m_ = h["mask"]
                _, pk, pk_pct, interior = window_peak(meas[m_], pct[m_], args.edge_margin_pct)
                _, spk, spk_pct, s_int = window_peak(so[m_], pct[m_], args.edge_margin_pct)
                halves.append(dict(subject=subj, tag=w["tag"], gc_key=w["gc_key"], trial=tr["name"],
                                   gait=gait_code(tr["name"]), part=h["part"], cover=h["cover"],
                                   meas_peak=pk, meas_peak_pct=pk_pct, meas_interior=interior,
                                   statop_peak=spk, statop_peak_pct=spk_pct, statop_interior=s_int))
        sel = (fr_all["gc_key"] == w["gc_key"]).to_numpy() & (fr_all["row"] >= r_.min()).to_numpy() & \
              (fr_all["row"] <= r_.max()).to_numpy()
        if sel.any():
            fr_all.loc[sel, "fmed_statop"] = np.interp(fr_all.loc[sel, "row"], r_, so)
    return dict(windows=pd.DataFrame(rows), halves=pd.DataFrame(halves))


def peak_table(halves, fr, sources, phi, d_mm, margin=3.0):
    """One row per covered stance window: the measured maximum (meas_peak, its % of stance, meas_interior) and every
    source's maximum over the same frames (peak_*, err_* = predicted - measured, peak_pct_*, interior_*). Only rows
    with meas_interior are scored (peaks_summary); the others are counted as rejected (peak_counts)."""
    rows = []
    fm = {src: fmed_of(fr, src, phi, d_mm) for src in sources if f"fy_{src}" in fr or f"fmed_{src}" in fr}
    meas, pct = fr["meas_med"].to_numpy(), fr["pct"].to_numpy(dtype=float)
    for h in halves:
        if not h["scored"]:
            continue
        idx = h["frame_index"]
        _, pk, pk_pct, interior = window_peak(meas[idx], pct[idx], margin)
        row = dict(subject=h["subject"], folder=h["folder"], gc_key=h["gc_key"], trial=h["trial"], gait=h["gait"],
                   stance=h["stance"], part=h["part"], win_lo=h["win_lo"], win_hi=h["win_hi"], cover=h["cover"],
                   hs=h["hs"], to=h["to"], hs_src=h["hs_src"], to_src=h["to_src"], win_first_pct=float(pct[idx[0]]),
                   win_last_pct=float(pct[idx[-1]]), meas_peak=pk, meas_peak_pct=pk_pct, meas_interior=interior)
        for src, v in fm.items():
            vv = v[idx]
            if np.isfinite(vv).all():
                _, ppk, ppct, pint = window_peak(vv, pct[idx], margin)
                row.update({f"peak_{src}": ppk, f"err_{src}": ppk - pk, f"peak_pct_{src}": ppct,
                            f"interior_{src}": pint})
        rows.append(row)
    return pd.DataFrame(rows)


def scored_peaks(st):
    """The stance windows whose measured maximum is an interior local maximum."""
    return st[st["meas_interior"].astype(bool)] if not st.empty else st


def peak_counts(st):
    """Per subject x window: covered windows, rejected ones (measured maximum at the window edge) and scored ones."""
    if st.empty:
        return pd.DataFrame()
    g = st.assign(edge=~st["meas_interior"].astype(bool)).groupby(["subject", "part"])
    return g.agg(n_covered=("meas_peak", "size"), n_edge_rejected=("edge", "sum")).reset_index().assign(
        n_scored=lambda d: d.n_covered - d.n_edge_rejected)


def peaks_summary(st, sources):
    """Peak errors per subject x window x source on the scored (interior) windows. pred_interior_frac = share of them
    in which the source has an interior maximum of its own (a peak, not an edge value)."""
    rows = []
    for (subj, part), g in scored_peaks(st).groupby(["subject", "part"]):
        for src in sources:
            if f"peak_{src}" not in g:
                continue
            ok = g[f"peak_{src}"].notna()
            if not ok.any():
                continue
            m = _metrics(g.loc[ok, "meas_peak"], g.loc[ok, f"peak_{src}"])
            rows.append(dict(subject=subj, part=part, source=src, n_halves=m["n"], mae=m["mae"], rmse=m["rmse"],
                             bias=m["bias"], r=m["r"], meas_mean=float(g.loc[ok, "meas_peak"].mean()),
                             pred_mean=float(g.loc[ok, f"peak_{src}"].mean()),
                             pred_interior_frac=float(g.loc[ok, f"interior_{src}"].astype(bool).mean())))
    return pd.DataFrame(rows)


def hs_peak_delays(res, root, margin=3.0, windows=PEAK_WINDOWS):
    """Model-free timing check: per stance whose heel strike the segment's force plate saw, the time from that heel
    strike to the measured first peak (maximum of the full GC trial's measured force in the early window, interior
    maxima only), for the medial and the total force. A late implant clock shows as a longer delay."""
    lo, hi = next((w[1], w[2]) for w in windows if w[0] == "early")
    rows, seen = [], set()
    for subj, r in res.items():
        fs = SUBJECTS[subj]["fs_gc"]
        for h in r["halves"]:
            if h["part"] != "early" or h["hs_src"] != "grf":
                continue
            key = (subj, h["gc_key"], round(float(h["hs"]), 1))
            if key in seen:
                continue
            seen.add(key)
            tr = load_gc_trial(subj, h["gc_key"], root)
            dur = h["to"] - h["hs"]
            rr = np.arange(int(np.ceil(h["hs"] + lo / 100 * dur)), int(np.floor(h["hs"] + hi / 100 * dur)) + 1)
            rr = rr[(rr >= 0) & (rr < len(tr["t"]))]
            if len(rr) < 3:
                continue
            pct = 100 * (rr - h["hs"]) / dur
            row = dict(subject=subj, gc_key=h["gc_key"], trial=h["trial"], gait=h["gait"],
                       gait_label=GAIT_LABELS.get(h["gait"], h["gait"]), hs_row=float(h["hs"]),
                       stance_ms=1000 * dur / fs, to_src=h["to_src"])
            for name, sig in (("med", tr["medial"]), ("tot", tr["total"])):
                k, _, p, interior = window_peak(sig[rr], pct, margin)
                row.update({f"delay_{name}_ms": 1000 * (rr[k] - h["hs"]) / fs, f"pct_{name}": p,
                            f"interior_{name}": interior})
            rows.append(row)
    return pd.DataFrame(rows)


def fixed_lag_ms(delays, subj="DM", ref=("JW", "SC"), signal="med", step_ms=5.0, match_gait=True):
    """Model-independent sync estimate from hs_peak_delays (interior first peaks only), rounded to step_ms.

    match_gait=True: the median over `subj`'s stances of (its delay - the median delay of the reference subjects'
    stances of the same gait type, GAIT_LABELS), which keeps SC's crouch / bouncy / forefoot trials out of DM's normal
    and trunk-sway comparison; falls back to the pooled estimate when no stance has a gait match.
    match_gait=False: median `subj` delay minus the pooled reference median. NaN if either side is empty."""
    if delays is None or delays.empty:
        return np.nan
    ok = delays[delays[f"interior_{signal}"].astype(bool)]
    col = f"delay_{signal}_ms"
    a, b = ok[ok.subject == subj], ok[ok.subject.isin(ref)]
    if a.empty or b.empty:
        return np.nan
    est = a[col].median() - b[col].median()
    if match_gait:
        lab = lambda g: g.map(lambda x: GAIT_LABELS.get(x, x))
        ref_med = b.groupby(lab(b["gait"]))[col].median()
        diffs = [d - ref_med[g] for d, g in zip(a[col], lab(a["gait"])) if g in ref_med.index]
        if diffs:
            est = float(np.median(diffs))
    return float(step_ms * np.round(est / step_ms))


def shift_measured(fr, lag_ms, root):
    """Copy of a frame table with the measured forces taken lag_ms later on the GC clock (positive = the implant is
    assumed late), interpolated from the full GC trial as in lag_sweep."""
    out = fr.copy()
    for (subj, key), g in fr.groupby(["subject", "gc_key"]):
        tr = load_gc_trial(subj, key, root)
        rows = g["row"].to_numpy() + lag_ms * SUBJECTS[subj]["fs_gc"] / 1000.0
        bw = g["mass"].to_numpy() * G
        idx = np.arange(len(tr["t"]))
        for col, sig in (("meas_med", "medial"), ("meas_lat", "lateral"), ("meas_tot", "total")):
            out.loc[g.index, col] = np.interp(rows, idx, tr[sig]) / bw
    return out


def gait_mods(st, sources):
    """Mean peak per subject x gait x window and its change from normal gait, measured vs predicted (scored windows)."""
    rows = []
    for (subj, part), g in scored_peaks(st).groupby(["subject", "part"]):
        base = g[g.gait == "ngait"]
        if base.empty:
            continue
        for gait, gg in g.groupby("gait"):
            if gait == "ngait":
                continue
            row = dict(subject=subj, part=part, gait=gait, gait_label=GAIT_LABELS.get(gait, gait), n_halves=len(gg),
                       n_normal=len(base), meas_change=gg.meas_peak.mean() - base.meas_peak.mean())
            for src in sources:
                if f"peak_{src}" in g and gg[f"peak_{src}"].notna().any() and base[f"peak_{src}"].notna().any():
                    row[f"{src}_change"] = gg[f"peak_{src}"].mean() - base[f"peak_{src}"].mean()
                    row[f"{src}_sign_agree"] = bool(np.sign(row[f"{src}_change"]) == np.sign(row["meas_change"]))
            rows.append(row)
    return pd.DataFrame(rows)


def lag_sweep(fr, sources, lag_ms, step_ms, root, phi, d_mm):
    rows = []
    for subj, g in fr.groupby("subject"):
        s = SUBJECTS[subj]
        st = g[g["stance"]]
        fms = {src: fmed_of(st, src, phi, d_mm) for src in sources if f"fy_{src}" in st or f"fmed_{src}" in st}
        fms = {src: v for src, v in fms.items() if np.isfinite(v).any()}
        for lag in np.arange(-lag_ms, lag_ms + 1e-9, step_ms):
            meas = np.full(len(st), np.nan)
            for key, gg in st.groupby("gc_key"):
                tr = load_gc_trial(subj, key, root)
                pos = st.index.get_indexer(gg.index)
                shifted = gg["row"].to_numpy() + lag * s["fs_gc"] / 1000.0
                meas[pos] = np.interp(shifted, np.arange(len(tr["t"])), tr["medial"]) / (gg["mass"].to_numpy() * G)
            for src, fm in fms.items():
                m = _metrics(meas, fm)
                rows.append(dict(subject=subj, source=src, lag_ms=float(lag), n=m["n"], mae=m["mae"], rmse=m["rmse"],
                                 bias=m["bias"], r=m["r"]))
    return pd.DataFrame(rows)


def make_figures(out, fr, st, sources, labels, phi, d_mm):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # fixed colour per entity (validated categorical slots 1-3, slot 7 for the published baseline); measured = ink.
    # The stored (state-derived) OpenSim labels share the OpenSim blue at a lighter step and are dashed.
    pal = ["#eb6834", "#1baf7a", "#e87ba4", "#eda100"]
    color = {"osim_sojr": "#2a78d6", "osim_state": "#86b6ef", "statop": "#6250d6"}
    color.update({lab: pal[i % len(pal)] for i, lab in enumerate(labels)})
    dash = {"osim_state": (0, (4, 2))}
    name = {k: v[1] for k, v in OSIM_SOURCES.items()}
    name.update({"statop": "SO+JR (Kaneda et al.)", "meas": "measured (implant)"})
    name.update({lab: f"surrogate {lab}" for lab in labels})
    st = scored_peaks(st)
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6})
    subjects = [s for s in SUBJECTS if s in set(fr["subject"])]
    # representative stance per subject: normal gait if possible, most scored halves, then most stance frames
    fig, axs = plt.subplots(1, len(subjects), figsize=(3.3 * len(subjects), 2.6), squeeze=False)
    for ax, subj in zip(axs[0], subjects):
        g = fr[fr.subject == subj]
        rank = g.groupby("folder").agg(normal=("gait", lambda x: x.iloc[0] == "ngait"), n_stance=("stance", "sum"))
        rank["halves"] = (st[st.subject == subj].groupby("folder").size() if not st.empty else 0)
        rank["halves"] = rank["halves"].fillna(0)
        folder = rank.sort_values(["halves", "normal", "n_stance"], ascending=False).index[0]
        if rank.loc[rank.normal & (rank.halves > 0)].shape[0]:
            folder = rank[rank.normal & (rank.halves > 0)].sort_values(["halves", "n_stance"], ascending=False).index[0]
        gg = g[g.folder == folder]
        tt = gg["gc_time"].to_numpy()
        stc = gg["stance"].to_numpy()
        ax.fill_between(tt, 0, 1, where=stc, transform=ax.get_xaxis_transform(), color="#f0efec", lw=0)
        ax.plot(tt, gg["meas_med"], color="#0b0b0b", lw=2, label=name["meas"])
        for src in list(OSIM_SOURCES) + ["statop"] + list(labels):
            if (f"fy_{src}" in gg or f"fmed_{src}" in gg):
                v = fmed_of(gg, src, phi, d_mm)
                if np.isfinite(v).any():
                    ax.plot(tt, v, color=color[src], lw=1.5, ls=dash.get(src, "-"), label=name[src])
        ax.set_title(f"{subj} ({SUBJECTS[subj]['side'].upper()} knee) {gg['trial'].iloc[0]}", fontsize=8)
        ax.set_xlabel("time in GC trial (s)")
        ax.set_ylabel("medial force (BW)")
    h, l_ = [], []
    for ax in axs[0]:
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in l_:
                h.append(hh)
                l_.append(ll)
    fig.legend(h, l_, loc="lower center", ncol=min(len(l_), 5), frameon=False, fontsize=7)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(os.path.join(out, "fig_gc_examples.png"), dpi=200)
    plt.close(fig)
    # stance-peak scatter
    if st.empty:
        return
    fig, axs = plt.subplots(1, len(subjects), figsize=(3.0 * len(subjects), 3.0), squeeze=False)
    hi = 0.0
    for ax, subj in zip(axs[0], subjects):
        g = st[st.subject == subj]
        for src in list(OSIM_SOURCES) + ["statop"] + list(labels):
            if f"peak_{src}" not in g:
                continue
            for part, mk in (("early", "o"), ("late", "^")):
                gg = g[(g.part == part) & g[f"peak_{src}"].notna()]
                if gg.empty:
                    continue
                ax.scatter(gg.meas_peak, gg[f"peak_{src}"], s=22, marker=mk, color=color[src], edgecolors="white",
                           linewidths=0.8, label=f"{name[src]}, {part}")
                hi = max(hi, gg.meas_peak.max(), gg[f"peak_{src}"].max())
        ax.set_title(f"{subj}: stance peaks (interior measured maxima)", fontsize=8)
        ax.set_xlabel("measured peak (BW)")
        ax.set_ylabel("predicted peak (BW)")
    for ax in axs[0]:
        ax.plot([0, hi * 1.05], [0, hi * 1.05], color="#8a8985", lw=1)
        ax.set_xlim(0, hi * 1.05)
        ax.set_ylim(0, hi * 1.05)
        ax.set_aspect("equal")
    h, l_ = [], []
    for ax in axs[0]:
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in l_:
                h.append(hh)
                l_.append(ll)
    fig.legend(h, l_, loc="lower center", ncol=4, frameon=False, fontsize=6.5)
    fig.tight_layout(rect=(0, 0.16, 1, 0.97))
    fig.savefig(os.path.join(out, "fig_gc_peaks.png"), dpi=200)
    plt.close(fig)


def _fmt(df, cols):
    return df[[c for c in cols if c in df]].round(3).to_string(index=False) if not df.empty else "(none)"


def osim_label_notes(seg, repo=REPO):
    """README lines on the two OpenSim sources: what each is, how many accepted segments have it, and the stored-label
    check of analysis/thesis/gc_knee_jr.py when its summary exists."""
    ok = seg[seg.status == "ok"]
    lines = ["## OpenSim label sources", ""]
    for src, (sub, name) in OSIM_SOURCES.items():
        col = f"{src}_status"
        have = (ok[col] == "ok") if col in ok else pd.Series(False, index=ok.index)
        per = ", ".join(f"{sj} {int(v)}" for sj, v in have.groupby(ok["subject"]).sum().items())
        lines.append(f"- `{src}` = {name}, from `<segment>/{sub}/`: {int(have.sum())} of {len(ok)} accepted "
                     f"segments ({per}).")
    lines += ["",
              "`osim_state` are the stored labels, which are also the surrogates' training labels. pipeline/batch_process.py "
              "ran StaticOptimization and JointReaction in one AnalyzeTool with an empty `<forces_file />`; OpenSim 4.5.2 "
              "documents that JR then computes the actuator forces from the states instead of applying the SO result "
              "(SO solves on its own copy of the model). They exist for walker_knee_r only, so SC (left knee) has none.",
              ]
    chk = os.path.join(repo, "results", "thesis", "gc", "knee_jr", "stored_check.csv")
    if os.path.exists(chk):
        c = pd.read_csv(chk)
        c = c[c.vas_r_frames.notna()] if "vas_r_frames" in c else c.iloc[0:0]
        if not c.empty:
            lines += ["",
                      f"Stored-label check (`{os.path.relpath(chk, repo)}`, {len(c)} GC folders, no OpenSim): in the "
                      f"{int(c.vas_r_frames.sum())} frames where the SO vasti force (vaslat + vasmed + vasint) exceeds 1 BW, "
                      f"the stored right-knee |F| is below that vasti force alone in a median "
                      f"{c.stored_below_vas_r_frac.median():.2f} of each folder's frames (minimum "
                      f"{c.stored_below_vas_r_frac.min():.2f}); stored knee |F| peaks at {c.stored_r_F_max_bw.median():.2f} BW "
                      f"(median; max {c.stored_r_F_max_bw.max():.2f}) while the SO vasti reach "
                      f"{c.so_vas_r_max_bw.max():.1f} BW. A knee label that applied the SO forces would carry the vasti pull."]
    chk_all = os.path.join(repo, "results", "thesis", "gc", "knee_jr", "stored_check_all_walking.csv")
    if os.path.exists(chk_all):
        c = pd.read_csv(chk_all)
        c = c[c.vas_r_frames > 0] if "vas_r_frames" in c else c.iloc[0:0]
        if not c.empty:
            lines += ["",
                      f"The same check on every walking folder with SO output (`{os.path.relpath(chk_all, repo)}`, the "
                      f"training and test labels): in {(c.stored_below_vas_r_frac > 0.5).mean():.0%} of the {len(c)} "
                      "folders with vasti > 1 BW frames, the stored knee |F| is below the vasti force in most of them."]
    if not (ok.get("osim_sojr_status", pd.Series(dtype=object)) == "ok").any():
        lines += ["",
                  "`osim_sojr` is not generated yet (JR-only OpenSim rerun, user-run). Decisive one-folder probe, then all "
                  "GC folders (both knees):", "", "```",
                  "conda run -n jcf python analysis/thesis/gc_knee_jr.py --pattern 'fregly_3GC_t18_r00' --workers 1 "
                  "--summary results/thesis/gc/knee_jr/probe_t18.csv",
                  "conda run -n jcf python analysis/thesis/gc_knee_jr.py --workers 20", "```"]
    sojr = (ok["osim_sojr_status"] == "ok") if "osim_sojr_status" in ok else pd.Series(False, index=ok.index)
    if sojr.any() and "osim_sojr_label_ok" in ok:
        bad = ok[sojr & ~ok["osim_sojr_label_ok"].map(lambda v: v is True or v == 1.0 or v == "True")]
        per = ", ".join(f"{sj} {int((bad.subject == sj).sum())} of {int((sojr & (ok.subject == sj)).sum())}"
                        for sj in ok.loc[sojr, "subject"].unique())
        lines += ["", f"`osim_sojr` segments failing the training label filters ({per}; reasons in segments.csv "
                  "`osim_sojr_filter`) are left out of the `common` subset, so `common` is not the same frames as `all` "
                  "for a subject with SO+JR labels."]
    lacking = [sj for sj in ok["subject"].unique() if not sojr[ok["subject"] == sj].any()]
    partial = [f"{sj} has them on only {int(sojr[ok.subject == sj].sum())} of {int((ok.subject == sj).sum())} "
               "accepted segments" for sj in ok["subject"].unique()
               if 0 < sojr[ok["subject"] == sj].sum() < (ok["subject"] == sj).sum()]
    if lacking or partial:
        what = ([f"no `osim_sojr` labels for {', '.join(lacking)}"] if lacking else []) + partial
        lines += ["", "**SO+JR coverage is incomplete: " + "; ".join(what) + ". Where they are missing, no OpenSim row "
                  "below is SO+JR, and the surrogates' force bias and missing late peak cannot be attributed to the "
                  "surrogate: they learnt `osim_state`-type labels.**"]
    lines += [""]
    return lines


def write_readme(out, args, infos, seg, summ, peaks, bias_df, statop_win, sweep_best, lag_best, implied, extra=None):
    extra = extra or {}
    ok = seg[seg.status == "ok"]
    lines = [
        f"# Grand Challenge in vivo validation: {args.name}",
        "",
        f"Run {datetime.datetime.now():%Y-%m-%d %H:%M} by `analysis/thesis/grand_challenge.py`.",
        f"Features: `{args.features}` (deploy = exact time alignment, 20 frames at 100 Hz). Scored frames: "
        f"`{args.grid}` window; surrogate input context: `{args.context}`.",
        "F_medial = phi*|F_y| + M_x*H/d with phi = {}, d = {} mm unless stated (sweep: phi {}, d {} mm).".format(
            *main_constants(args), args.phi, args.d_mm),
        MX_README[getattr(args, "mx_convention", "pipeline")],
        "BW = AddBiomechanics metadata mass (JW 64.6 kg, DM 67.0 kg, SC 78.4 kg); the published masses "
        "(JW 68, DM 70, SC 78.4 kg) would scale every BW number of JW and DM by about 0.95.",
        "",
        f"**Every number from a surrogate below is {PROVISIONAL}.**",
        "",
        "## Checkpoints",
        "",
    ]
    for lab, info in zip(args.labels, infos):
        lines.append(f"- `{lab}`: `{os.path.relpath(info['path'], REPO)}` (input set `{info['input_set']}`, "
                     f"exclude {info.get('exclude')}, git {info.get('git')})")
    lines += ["", "## Alignment", "",
              "Segment vertical GRF vs GC plates (sum |Fz| at marker rows), searched over all trials of the subject. "
              f"Accepted: r >= {args.min_r}, amplitude ratio 0.9-1.1, and agreement with the marker/IK identity map "
              f"(or, without it, sharpness (1 - r_second)/(1 - r_best) >= {args.min_sharpness}).", "",
              "```", seg.groupby(["subject", "status"]).size().rename("segments").reset_index().to_string(index=False),
              "```", "",
              "```", _fmt(ok.groupby("subject").agg(segments=("folder", "size"), trials=("gc_key", "nunique"),
                                                     r_min=("r_best", "min"), r_median=("r_best", "median"),
                                                     r_second_max=("r_second", "max"),
                                                     sharpness_min=("sharpness", "min"),
                                                     lag_frames_median=("grf_lag_frames", "median"),
                                                     ).reset_index(),
                                ["subject", "segments", "trials", "r_min", "r_median", "r_second_max",
                                 "sharpness_min", "lag_frames_median"]), "```",
              "", "grf_lag_frames = GRF-match row minus the identity (kinematic) row: expected 0 for JW and DM and -1 "
              "for SC (AddBiomechanics stores SC's GRF one frame late).", ""]
    lines += osim_label_notes(seg)
    lines += ["## Frame metrics (instrumented-limb stance frames, BW; lag 0 = primary result)", "",
              "Subsets: `all` = every accepted segment, each source on the frames it has; `common` = frames where every "
              "OpenSim source the subject has exists and passes the training label filters, identical frames for every "
              "source; `statop_overlap` = SC frames inside Kaneda's windows. total_* = |F_y| vs measured axial force.",
              "DM rows at lag 0 are confounded with DM's implant timing (see the DM timing section); state them "
              "conditionally and do not pool DM into headline statements on lag 0 alone.", "",
              "```", _fmt(summ, ["subset", "subject", "source", "n_frames", "n_segments", "mae", "rmse", "bias", "r",
                                 "total_mae", "total_bias", "total_r"]), "```", ""]
    windows = ", ".join(f"{n} {lo:g}-{hi:g} %" for n, lo, hi in PEAK_WINDOWS)
    lines += ["## Stance peaks", "",
              f"Windows of stance: {windows}. A window is covered when the segment spans >= {int(100 * args.min_cover)} % "
              f"of it, and scored only when the measured maximum is an interior local maximum, at least "
              f"{args.edge_margin_pct:g} % of stance from the first and last covered frame of the window (a maximum at "
              "the edge is the tail of a peak outside the window, e.g. the first peak's descent). pred_interior_frac = "
              "share of the scored windows in which the source has an interior maximum of its own.", ""]
    counts = extra.get("peak_counts")
    if counts is not None and not counts.empty:
        lines += ["```", counts.to_string(index=False), "```", ""]
        late = counts[counts.part == "late"].set_index("subject")
        thin = [f"{sj} ({int(r.n_scored)} of {int(r.n_covered)} late windows scored)" for sj, r in late.iterrows()
                if r.n_scored < max(3, 0.5 * r.n_covered)]
        if thin:
            lines += [f"Late-peak statements cannot include {', '.join(thin)}: most of their late-window maxima sit on "
                      "the window edge.", ""]
    lines += ["```", _fmt(peaks, ["subject", "part", "source", "n_halves", "mae", "bias", "r", "meas_mean",
                                  "pred_mean", "pred_interior_frac"]), "```", ""]
    delays, dm_lag, sens = extra.get("hs_delays"), extra.get("dm_lag_ms"), extra.get("dm_sens")
    if delays is not None and not delays.empty:
        agg = []
        for sj, g in delays.groupby("subject", sort=False):
            for sig in ("med", "tot"):
                v = g.loc[g[f"interior_{sig}"].astype(bool), f"delay_{sig}_ms"]
                agg.append(dict(subject=sj, signal={"med": "medial", "tot": "total"}[sig], n_stances=len(v),
                                median_ms=v.median() if len(v) else np.nan,
                                min_ms=v.min() if len(v) else np.nan, max_ms=v.max() if len(v) else np.nan,
                                values_ms=" ".join(f"{x:.0f}" for x in v)))
        lines += ["## DM timing (model-independent) and DM sensitivity at a fixed lag", "",
                  "Time from a heel strike seen on the segment's force plate to the measured first peak (interior "
                  "maximum of the full GC trial's force in the early window):", "",
                  "```", pd.DataFrame(agg).round(1).to_string(index=False), "```", "",
                  "DM minus JW + SC, rounded to 5 ms (gait-matched = median over DM stances of DM minus the reference "
                  "stances of the same gait type; pooled = median minus median over all stances): " + "; ".join(
                      f"{'medial' if sig == 'med' else 'total'} force {'gait-matched' if mg else 'pooled'} "
                      f"{fixed_lag_ms(delays, signal=sig, match_gait=mg):+.0f} ms"
                      for sig in ("med", "tot") for mg in (True, False)) + ".", "",
                  f"Lag used for the DM sensitivity rows: **{dm_lag:+.0f} ms**" + (
                      " (set by --dm-lag-ms)" if args.dm_lag_ms is not None else
                      f" ({'medial' if args.dm_lag_signal == 'med' else 'total'} force, "
                      f"{'pooled' if args.dm_lag_pooled else 'gait-matched'})") + ". "
                  "It uses no model, unlike the lag sweep's best-r lags, which also contain each model's own phase error. "
                  "The GRF-edge timing points the same way (EXPLORE_NOTES section 4.4, implant minus plate edge at heel "
                  "strike / toe-off: DM +40 / +55 ms, JW -100 / -37 ms, SC +21 / -8 ms). Gait differences between the "
                  "patients also move the first peak, so this is a sensitivity check, not a correction.", ""]
        if sens is not None and not sens.empty:
            fcols = ["lag_ms", "subset", "source", "n", "mae", "bias", "r", "total_bias"]
            pcols = ["lag_ms", "part", "source", "n", "n_scored", "n_edge_rejected", "mae", "bias", "pred_interior_frac"]
            lines += ["DM frame metrics (stance frames), lag 0 vs the fixed lag, every source:", "", "```",
                      _fmt(sens[sens.kind == "frames"], fcols), "```", "",
                      "DM stance peaks under the same shift (windows and interior rule as above):", "", "```",
                      _fmt(sens[sens.kind == "peaks"], pcols), "```", ""]
    if statop_win is not None and not statop_win.empty:
        lines += ["## StatOp_Val (Kaneda et al. 2023) SO+JR medial vs measured, SC, own windows", "",
                  f"{len(statop_win)} windows; pooled over windows: mean RMSE {statop_win.rmse.mean():.3f} BW, "
                  f"mean r {statop_win.r.mean():.3f}. Our recomputation; Kaneda's own metric definition is not "
                  "verified, so do not quote this as the published value.", ""]
    if bias_df is not None and not bias_df.empty:
        lines += ["## Kinematic stance events vs segment force-plate events (used for events outside a segment)", "",
                  "```", bias_df.round(1).to_string(index=False), "```", ""]
    if sweep_best is not None and not sweep_best.empty:
        lines += ["## phi / d sweep: best MAE per subject x source", "", "```",
                  _fmt(sweep_best, ["subject", "source", "phi", "d_mm", "mae", "bias", "r"]), "```", ""]
    if implied is not None and not implied.empty:
        lines += ["## Implied phi, d (least squares, no intercept)", "", "```",
                  _fmt(implied, ["subject", "source", "n_frames", "phi", "k_moment", "d_mm", "rmse_fit", "r2"]),
                  "```", "d = H / k_moment is undefined (NaN) when the fitted moment coefficient is not positive.", ""]
    if lag_best is not None and not lag_best.empty:
        lines += ["## Lag sweep (positive lag = measured force taken later than the prediction)", "",
                  "lag_ms = lag of minimum MAE, lag_ms_best_r = lag of maximum r. A model's systematic phase error "
                  "moves both, so a sync offset shows only if every source (incl. SO+JR) agrees.", "", "```",
                  _fmt(lag_best, ["subject", "source", "lag_ms", "mae", "mae_at_0", "lag_ms_best_r", "best_r",
                                  "r_at_0"]), "```", ""]
    sc = extra.get("mx_sign")
    if sc is not None and not sc.empty:
        lines += ["## M_x sign against the implant (stance frames)", "",
                  "r_mx_vs_meas_med_minus_lat > 0 means the source's medial-positive M_x rises when the implant's "
                  "medial - lateral force rises. medial_moment_frac = share of frames with a medial moment; "
                  "medial_share = mean(F_medial) / mean(|F_y|) against the implant's mean(medial) / mean(total).", "",
                  "```", _fmt(sc, ["subject", "source", "n_frames", "r_mx_vs_meas_med_minus_lat", "medial_moment_frac",
                                   "moment_term_mean_bw", "medial_share", "meas_medial_share"]), "```", ""]
    lines += ["## Caveats", "",
              "- Segments are mostly half stances (SO windows), so frame metrics weight the covered phases.",
              "- JW medial = posterior-medial + anterior-medial cell (no regression supplied); unverified split.",
              "- DM's implant timing is suspect (DM timing section); lag 0 is the primary result, DM statements are "
              "conditional.",
              "- SC is a LEFT knee: surrogate inputs are mirrored; its only OpenSim labels are `osim_sojr` "
              "(walker_knee_l, `analysis/thesis/gc_knee_jr.py`, user-run).",
              "- JW and DM sit in the training folders but every checkpoint here was trained with "
              "`--exclude han fregly`; SC is in the test split.",
              "- The README regressions were fitted on gait and filtered data; bouncy/crouch exceed that range.",
              ""]
    with open(os.path.join(out, "README.md"), "w") as f:
        f.write("\n".join(lines))


def common_mask(fr):
    """Frames where every OpenSim source the subject has at all exists and passes the training label filters (a
    subject without a source, e.g. SC without osim_state, is not restricted by it)."""
    m = pd.Series(True, index=fr.index)
    for src in OSIM_SOURCES:
        if f"fy_{src}" not in fr:
            continue
        has = fr.groupby("subject")[f"fy_{src}"].transform(lambda v: v.notna().any())
        m &= (fr[f"fy_{src}"].notna() & fr[f"{src}_ok"].astype(bool)) | ~has
    return m.to_numpy()


def dm_lag_sensitivity(r, sources, lag_ms, args, phi, d_mm):
    """DM frame metrics (subsets all / common) and stance peaks at lag 0 and with the measured force shifted by lag_ms,
    for every source. Long format: kind = frames | peaks."""
    rows = []
    for lag in sorted({0.0, float(lag_ms)}):
        frd = shift_measured(r["frames"], lag, args.gc_root) if lag else r["frames"]
        for subset, f in (("all", frd), ("common", frd[common_mask(frd)])):
            fs = frame_summary(f, sources, phi, d_mm)
            for _, x in fs.iterrows():
                rows.append(dict(kind="frames", lag_ms=lag, subset=subset, source=x.source, n=x.n_frames, mae=x.mae,
                                 rmse=x.rmse, bias=x.bias, r=x.r, total_bias=x.get("total_bias", np.nan)))
        st = peak_table(r["halves"], frd, sources, phi, d_mm, args.edge_margin_pct)
        if st.empty:
            continue
        cnt = peak_counts(st).set_index("part")
        ps = peaks_summary(st, sources)
        for part, c in cnt.iterrows():
            base = dict(kind="peaks", lag_ms=lag, part=part, n_scored=int(c.n_scored),
                        n_edge_rejected=int(c.n_edge_rejected))
            got = ps[ps.part == part] if not ps.empty else ps
            if got.empty:
                rows.append(dict(base, source="(none scored)", n=0))
            for _, x in got.iterrows():
                rows.append(dict(base, source=x.source, n=x.n_halves, mae=x.mae, rmse=x.rmse, bias=x.bias, r=x.r,
                                 pred_interior_frac=x.pred_interior_frac))
    return pd.DataFrame(rows)


def main_constants(args):
    """Headline phi and d: 0.5 and 45 mm when they are in the sweep lists, else the first values."""
    return (0.5 if 0.5 in args.phi else args.phi[0]), (45.0 if 45.0 in args.d_mm else args.d_mm[0])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build-maps", action="store_true", help="jcf env: build alignment.csv and segment_map.csv, then exit")
    ap.add_argument("--ckpt", nargs="+", help="GRF-free MLP checkpoint(s)")
    ap.add_argument("--labels", nargs="+", default=None, help="short names, one per --ckpt")
    ap.add_argument("--subjects", nargs="+", default=list(SUBJECTS), choices=list(SUBJECTS))
    ap.add_argument("--features", choices=["deploy", "native"], default="deploy",
                    help="deploy = exact time alignment + 100 Hz windows; native = the checkpoint's own input set")
    ap.add_argument("--grid", choices=["jr", "ik"], default="jr",
                    help="frames scored: jr = inside the JR window (identical frames for every source); ik = whole segment")
    ap.add_argument("--context", choices=["jr", "ik"], default="ik",
                    help="frames the surrogate's inputs are built on: ik = the whole segment (more context, no edge "
                         "replication inside the scored frames); jr = the JR window, as training built them")
    ap.add_argument("--mx-convention", choices=MX_CONVENTIONS, default="physical",
                    help="sign making M_x medial-positive: physical = -mx right / +mx left (settled, "
                         "analysis/thesis/knee_mx_sign.py); pipeline = the training code's +mx right / -mx left")
    ap.add_argument("--phi", type=float, nargs="+", default=[0.5])
    ap.add_argument("--d-mm", type=float, nargs="+", default=[45.0])
    ap.add_argument("--min-r", type=float, default=0.98, help="minimum GRF cross-correlation to accept a segment")
    ap.add_argument("--min-sharpness", type=float, default=5.0,
                    help="without the identity map: (1 - r_second) / (1 - r_best) needed to accept a match")
    ap.add_argument("--min-cover", type=float, default=0.9, help="fraction of a stance peak window a segment must cover")
    ap.add_argument("--edge-margin-pct", type=float, default=3.0,
                    help="a window's measured maximum within this %% of stance of the window edge is not a peak")
    ap.add_argument("--dm-lag-ms", type=float, default=None,
                    help="fixed DM sensitivity lag; default = model-free heel-strike-to-first-peak estimate")
    ap.add_argument("--dm-lag-signal", choices=["med", "tot"], default="med",
                    help="measured force for the heel-strike-to-first-peak estimate")
    ap.add_argument("--dm-lag-pooled", action="store_true",
                    help="compare DM with all JW + SC stances pooled instead of gait-matched stances")
    ap.add_argument("--max-peak", type=float, default=6.0, help="label filter (resultant BW) for the OpenSim flag")
    ap.add_argument("--lag-sweep-ms", type=float, default=100.0, help="0 disables the lag sweep")
    ap.add_argument("--lag-step-ms", type=float, default=10.0)
    ap.add_argument("--no-statop", action="store_true")
    ap.add_argument("--max-segments", type=int, default=None, help="cap segments per subject (smoke tests)")
    ap.add_argument("--gc-root", default=GC_ROOT)
    ap.add_argument("--out-root", default=OUT_ROOT)
    ap.add_argument("--name", default=None)
    args = ap.parse_args(argv)
    if args.build_maps:
        build_maps(args.out_root, args.gc_root)
        return
    if not args.ckpt:
        ap.error("--ckpt is required (unless --build-maps)")
    from tools.evaluate_fmedial import load_model
    from train_mlp import INPUT_SETS
    models = [load_model(p, "cpu") for p in args.ckpt]
    infos = [m[3] for m in models]
    if args.labels is None:
        args.labels = [f"m{i}" for i in range(len(models))]
    if len(args.labels) != len(models) or len(set(args.labels)) != len(args.labels) or \
            RESERVED_LABELS & set(args.labels):
        ap.error(f"--labels: one unique name per --ckpt, none of {sorted(RESERVED_LABELS)}")
    cfgs = []
    for info in infos:
        cfg = dict(INPUT_SETS[info["input_set"]])
        if args.features == "deploy":
            cfg.update(time_exact=True, resample_hz=100.0)
        cfgs.append(cfg)
    args.name = args.name or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(args.out_root, args.name)
    os.makedirs(out, exist_ok=True)
    al, seg_map = read_maps(args.out_root)
    if seg_map is None:
        print("no segment_map.csv: alignment uses GRF cross-correlation only (run --build-maps in the jcf env)")
    res = {}
    for subj in args.subjects:
        r = process_subject(subj, models, cfgs, args, seg_map, args.gc_root)
        if r is not None:
            res[subj] = r
            n_ok = int((r["segments"].status == "ok").sum()) if not r["segments"].empty else 0
            print(f"{subj}: {len(r['segments'])} segments, {n_ok} accepted, {len(r['frames'])} eval frames", flush=True)
    if not res:
        raise SystemExit("no subject could be processed (are the GC files under --gc-root?)")
    seg = pd.concat([r["segments"] for r in res.values()], ignore_index=True)
    for c in ["grf_lag_frames", "duplicate_of"] + [f"{k}_{x}" for k in OSIM_SOURCES for x in ("status", "filter")]:
        if c not in seg:
            seg[c] = np.nan
    fr = pd.concat([r["frames"] for r in res.values() if not r["frames"].empty], ignore_index=True)
    osim = [k for k in OSIM_SOURCES if f"fy_{k}" in fr and fr[f"fy_{k}"].notna().any()]
    sources = osim + list(args.labels) + (["statop"] if "fmed_statop" in fr else [])
    phi0, d0 = main_constants(args)
    seg.to_csv(os.path.join(out, "segments.csv"), index=False)
    seg[seg.gc_key.notna()].groupby(["subject", "gc_key", "trial"]).agg(
        segments=("folder", "size"), accepted=("status", lambda s: int((s == "ok").sum())),
        r_min=("r_best", "min"), r_median=("r_best", "median"), r_second_max=("r_second", "max"),
        sharpness_min=("sharpness", "min"),
        lag_frames=("grf_lag_frames", lambda s: ";".join(str(int(v)) for v in sorted(set(s.dropna())))),
        statuses=("status", lambda s: ";".join(sorted(set(s))))).reset_index().to_csv(
        os.path.join(out, "alignment_trials.csv"), index=False)
    fr.to_csv(os.path.join(out, "frames.csv.gz"), index=False, float_format="%.5g")
    # frame metrics
    subsets = {"all": fr, "common": fr[common_mask(fr)]}
    if "fmed_statop" in fr:
        subsets["statop_overlap"] = fr[fr["fmed_statop"].notna()]
    summ = pd.concat([frame_summary(f, sources, phi0, d0).assign(subset=k) for k, f in subsets.items() if not f.empty],
                     ignore_index=True)
    summ.to_csv(os.path.join(out, "summary.csv"), index=False)
    frame_summary(fr, sources, phi0, d0, by=("subject", "gait")).to_csv(os.path.join(out, "summary_by_gait.csv"),
                                                                      index=False)
    # stance peaks
    st = pd.concat([peak_table(r["halves"], r["frames"], sources, phi0, d0, args.edge_margin_pct)
                    for r in res.values() if r["halves"]],
                   ignore_index=True) if any(r["halves"] for r in res.values()) else pd.DataFrame()
    st.to_csv(os.path.join(out, "stances.csv"), index=False)
    counts = peak_counts(st)
    counts.to_csv(os.path.join(out, "peak_counts.csv"), index=False)
    hv_all = pd.DataFrame([{k: v for k, v in h.items() if k not in ("mask", "frame_index")}
                           for r in res.values() for h in r["halves"]])
    hv_all.to_csv(os.path.join(out, "halves.csv"), index=False)
    peaks = peaks_summary(st, sources) if not st.empty else pd.DataFrame()
    statop_win = None
    if "SC" in res and res["SC"]["statop"] is not None:
        so = res["SC"]["statop"]
        statop_win = so["windows"]
        statop_win.to_csv(os.path.join(out, "statop_windows.csv"), index=False)
        so["halves"].to_csv(os.path.join(out, "statop_halves.csv"), index=False)
        if not so["halves"].empty:
            h = so["halves"][so["halves"]["meas_interior"].astype(bool)]
            own = [dict(subject="SC", part=part, source="statop_own_windows", n_halves=len(g),
                        **{k: v for k, v in _metrics(g.meas_peak, g.statop_peak).items() if k != "n"},
                        meas_mean=g.meas_peak.mean(), pred_mean=g.statop_peak.mean(),
                        pred_interior_frac=g.statop_interior.astype(bool).mean()) for part, g in h.groupby("part")]
            peaks = pd.concat([peaks, pd.DataFrame(own)], ignore_index=True)
    peaks.to_csv(os.path.join(out, "peaks_summary.csv"), index=False)
    gm = gait_mods(st, sources) if not st.empty else pd.DataFrame()
    gm.to_csv(os.path.join(out, "gait_mods.csv"), index=False)
    bias_df = pd.DataFrame([r["bias"] for r in res.values() if r["bias"]])
    bias_df.to_csv(os.path.join(out, "stance_events.csv"), index=False)
    # phi / d sweep and implied constants
    sw = pd.concat([frame_summary(fr, [s_ for s_ in sources if s_ != "statop"], p, d).assign(phi=p, d_mm=d)
                    for p in args.phi for d in args.d_mm], ignore_index=True)
    sw.to_csv(os.path.join(out, "sweep.csv"), index=False)
    sweep_best = sw.loc[sw.dropna(subset=["mae"]).groupby(["subject", "source"])["mae"].idxmin()] if not sw.empty else None
    implied = implied_phi_d(fr, sources)
    implied.to_csv(os.path.join(out, "implied_phi_d.csv"), index=False)
    sign_chk = mx_sign_check(fr, [s_ for s_ in sources if s_ != "statop"], d0)
    sign_chk.to_csv(os.path.join(out, "mx_sign_check.csv"), index=False)
    lag_best = None
    if args.lag_sweep_ms > 0:
        lg = lag_sweep(fr, sources, args.lag_sweep_ms, args.lag_step_ms, args.gc_root, phi0, d0)
        lg.to_csv(os.path.join(out, "lag_sweep.csv"), index=False)
        if not lg.empty:
            lag_best = lg.loc[lg.dropna(subset=["mae"]).groupby(["subject", "source"])["mae"].idxmin()].copy()
            at0 = lg[lg.lag_ms == 0].set_index(["subject", "source"])
            lag_best["mae_at_0"] = [at0.loc[(a, b), "mae"] for a, b in zip(lag_best.subject, lag_best.source)]
            lag_best["r_at_0"] = [at0.loc[(a, b), "r"] for a, b in zip(lag_best.subject, lag_best.source)]
            rbest = lg.loc[lg.dropna(subset=["r"]).groupby(["subject", "source"])["r"].idxmax()].set_index(
                ["subject", "source"])
            lag_best["lag_ms_best_r"] = [rbest.loc[(a, b), "lag_ms"] for a, b in zip(lag_best.subject, lag_best.source)]
            lag_best["best_r"] = [rbest.loc[(a, b), "r"] for a, b in zip(lag_best.subject, lag_best.source)]
    # DM sync: model-free heel-strike-to-first-peak delays and every source at one fixed lag (lag 0 stays primary)
    delays = hs_peak_delays(res, args.gc_root, args.edge_margin_pct)
    delays.to_csv(os.path.join(out, "hs_peak_delay.csv"), index=False)
    dm_lag = args.dm_lag_ms if args.dm_lag_ms is not None else fixed_lag_ms(delays, signal=args.dm_lag_signal,
                                                                           match_gait=not args.dm_lag_pooled)
    sens = dm_lag_sensitivity(res["DM"], sources, dm_lag, args, phi0, d0) if "DM" in res and np.isfinite(dm_lag) \
        else pd.DataFrame()
    sens.to_csv(os.path.join(out, "dm_lag_sensitivity.csv"), index=False)
    make_figures(out, fr, st, sources, args.labels, phi0, d0)
    write_readme(out, args, infos, seg, summ, peaks, bias_df, statop_win, sweep_best, lag_best, implied,
                 extra=dict(peak_counts=counts, hs_delays=delays, dm_lag_ms=dm_lag, dm_sens=sens, mx_sign=sign_chk))
    with open(os.path.join(out, "run.json"), "w") as f:
        json.dump(dict(argv=sys.argv, args={k: v for k, v in vars(args).items()}, dm_lag_ms_used=dm_lag,
                       peak_windows=PEAK_WINDOWS,
                       checkpoints=[dict(label=l_, **{k: v for k, v in i.items()}) for l_, i in zip(args.labels, infos)]),
                  f, indent=2, default=str)
    pd.set_option("display.width", 200)
    print(summ[summ.subset == "all"][["subject", "source", "n_frames", "mae", "rmse", "bias", "r"]].round(3).to_string(index=False))
    if not peaks.empty:
        print(peaks[["subject", "part", "source", "n_halves", "mae", "bias", "r"]].round(3).to_string(index=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

"""Real-IMU check on the Twente treadmill dataset (Wang et al. 2023): Xsens MVN -> PN3Estimator (UKF) -> surrogate.

Each walking trial (9 subjects x 6 belt speeds) goes through the live PN3 pipeline of pn3_redis/driver.py and is
compared with the optical recording made at the same time. Three paths are built on one time base (the IK clock):

  O  reference   marker IK (OS/DataFiles/*IK.mot, gait2392) -> Rajagopal conventions -> LowerBody FK scaled by height
                 -> checkpoint foot point -> centred 20-frame windows at 100 Hz -> surrogate, treadmill contact
  U  IMU         MVN segment (or raw sensor) orientations resampled to 60 Hz (the PN3 rate) -> standing calibration
                 -> UKF over the 19 coordinates -> LiveSurrogate, contact from the driver's gyro detectors with
                 heel-strike back-dating. The same UKF run also feeds LiveSurrogates with treadmill or insole flags,
                 which splits the error into kinematics (U-grf vs O) and contact (U-gyro vs U-grf).
  V  vendor      MVN's own joint angles (IK_IMU.mot) through the same FK, windows and contact as O

The dataset has no knee-load labels: every F_medial number is U (or V) against O on the same frames. That difference
is not only what the sensors, the calibration and the filter add. It also holds the trunk and pelvis definition
differences between gait2392 marker IK and the MVN / UKF model (see Conventions): the lumbar angles are not surrogate
inputs, but they move the whole-body COM through FK and the COM-minus-pelvis window carries that. Decomposition rows
(fmedial.csv, decomposition.csv) separate the two:

  O/V:<coords>          O with V's lumbar_*, pelvis_tilt or both swapped in, against O: the definition difference
                        alone, with no IMU in the loop
  O+Vbias:lumbar+tilt   O shifted by the trial's mean V - O on those four coordinates, against O: its constant part
  Omatch:<path>         offset-matched optical: O shifted by the subject's constant (path - O) offset on pelvis_tilt
                        and lumbar_*, estimated on the subject's OTHER speeds (leave-one-speed-out), so the scored
                        trial never fits its own correction. U and V against it give what is left once a constant
                        per-subject trunk / tilt offset is removed. That offset also absorbs any constant per-subject
                        calibration error in those coordinates. The parts do not add: F_medial is non-linear in its
                        inputs, and where the trunk / tilt offset was cancelling another error, matching raises the
                        difference (it does for checkpoint B on V and seg-mvnmodel in the pilot).

Conventions (how each was checked: results/thesis/twente/EXPLORE_NOTES.md)
  * MVNX quaternions are (w, x, y, z) and rotate segment (sensor) axes into MVN global. MVN global is x forward,
    y left, z up; M_MVN maps it to OpenSim ground (x forward, y up, z right). M_MVN is a proper rotation.
  * MVN segment axes: x forward, y left, z up in the N-pose. The raw Link pelvis sensor points its -z axis forward.
  * gait2392 -> Rajagopal (the surrogate's and the estimator's convention):
      knee_angle_r/l   sign flipped: gait2392 flexion is negative (range -120..10 deg), Rajagopal positive (0..120)
      pelvis_tilt/list/rotation   same ground_pelvis sequence Rz(tilt) Rx(list) Ry(rotation), same signs
      hip_flexion/adduction/rotation   same axes (Z, X, Y; left side mirrored in both models), same signs
      ankle_angle   same Delp oblique axis, same sign; lumbar_*   same axes and signs
      degrees -> radians for every rotation; pelvis_tx/ty/tz stay in metres (exact names, see GAIT2392_SIGN)
    Anatomical frame offsets between the two models are not corrected; they show up as bias in angles.csv. They are
    not small for the trunk and pelvis. IK_IMU's lumbar_extension sits a median 11 deg above the optical one (8-17 deg
    across subjects; error SD about 1 deg and median r 0.75, so a near-constant offset, not a sign error). Its pelvis
    tilt sits 2.5 deg below (1-4 deg). The UKF inherits MVN's definition with mvnmodel calibration (+11 / -2.5 deg)
    and the standing pose's with static calibration (+8 / +1.2 deg). lumbar_* are therefore scored (SCORED) next to
    the 13 surrogate angles, and the decomposition rows above separate their effect on F_medial. In F_medial the two
    push in opposite directions (MVN's lumbar raises it, MVN's tilt lowers it), so together they move it less than
    the lumbar swap alone.
  * Clock: t_IK = index / 240 - L with L = 0.5 s + Datastr.Resample.Sych.DeltaT.Motion (Subj<XX>_<trial>.mat). IK,
    IK_IMU, XLD and XLD_INSOLE share one 100 Hz clock from 0. Every trial is checked by cross-correlating a knee angle
    from the MVN segments against the optical knee (sync_lag_ms, expected within a few ms of 0). If DeltaT is missing
    the lag is estimated from that cross-correlation instead.
  * Contact: |vertical GRF| > 5 % BW, the training definition. XLD ground_force_vy is the right foot (calcn_r),
    l_ground_force_vy the left. XLD_INSOLE (Moticon) gives the same columns from the insoles.
  * IK_IMU pelvis_rotation carries MVN's heading (about 185 deg); its tilt and list are already heading-free (as given
    they match the optical tilt/list with RMSE 1.1-4.6 deg over 18 trials, 2-7 deg after a proper re-decomposition
    about the vertical). V therefore subtracts the static-pose MVN pelvis heading from pelvis_rotation only: the same
    heading reference the UKF calibration uses.

Calibration variants (--variants, "<field>-<calib>"; field: seg = MVN segment orientations, sensor = raw sensors):
  static     the estimator's standing calibration (driver-faithful, assumes the Rajagopal default pose q = 0). It uses
             the first still second at the start of the walking trial if there is one (there is none in this
             dataset), else the first still second of static_pose.mvnx from t = 1 s (same MVN calibration and sensor
             mounting as the walks).
  mvnmodel   MVN's model pose as q = 0: one second of frames in which every segment is MVN's identity orientation
             turned to the static-pose pelvis heading (segment field only).

F_medial sign (--mx-convention): every F_medial here is formed from the surrogate's raw fy and mx (right knee, BW and
BW*H) as 0.5*|fy| + s*mx*H/0.045, with s = -1 (physical: the medial moment is -mx at walker_knee_r,
analysis/thesis/knee_mx_sign.py; default) or +1 (pipeline: what pre-Nov-2026 checkpoints publish as f_medial, and
what the Nov 2026 all_AB / all_AB_anchorgrf runs used). The checkpoint's own f_medial is not used, so one convention
holds for every checkpoint and every path.

Scoring: from 3 s after the first MVN frame (the UKF starts from the standing pose while the subject walks) to the
end of the IK record. Aggregation: trial -> subject mean over speeds -> median and IQR over subjects; per speed:
median and IQR over subjects. Subj04 was reprocessed offline by MVN ("HD"); 05-12 are MVN "Live" (summary "live").
The Omatch rows come from a second pass after all jobs, because each trial's offset is fitted on the other trials
(offsets.csv lists them; --no-offset-match skips the pass). angles_table.csv, fmedial_table.csv and
decomposition.csv hold median [q25, q75] over subjects of every (test, ref) pair, overall and per speed.

Privacy: the MVNX subject label and originalFilename hold participants' first names. read_mvnx keeps only the
header attributes in MVNX_META, and nothing else from the headers reaches the outputs. The dataset states no
licence: per-frame outputs (frames/*.npz) stay local (*.npz is git-ignored); only aggregates go into the paper.

Usage (python -I: the dataset is untrusted data; nothing from it is imported or run):
    python -I analysis/thesis/twente_imu.py --ckpt A.pt B.pt --labels A B --subjects 05 --name pilot_s05
    python -I analysis/thesis/twente_imu.py --ckpt A.pt B.pt --labels A B --name all_AB --workers 23
    python -I analysis/thesis/twente_imu.py --replot --name all_AB          # figures from the saved outputs
"""
import argparse
import copy
import datetime
import json
import os
import sys
import time
import xml.etree.ElementTree as ET
from collections import deque

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):   # one process per job: no BLAS threads
    os.environ.setdefault(_v, "1")

import numpy as np                                                    # noqa: E402
from scipy.spatial.transform import Rotation, Slerp                   # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from pn3_redis.estimator import CausalContact, PN3Estimator          # noqa: E402
from pn3_redis.model import COORDS, TRANSLATIONS, LowerBody           # noqa: E402

DATA_DEFAULT = "/home/gellert/Developer/datasets/wang2023_twente/extracted"
OUT_DEFAULT = os.path.join(REPO, "results", "thesis", "twente")
SUBJECTS = ["04", "05", "06", "07", "08", "09", "10", "11", "12"]
WALK_SPEED_MPS = {"walk_09": 0.25, "walk_18": 0.50, "walk_27": 0.75, "walk_36": 1.00, "walk_45": 1.25, "walk_54": 1.50}
HD_SUBJECTS = {"04"}                       # MVN offline reprocessing; the rest are MVN Live (causal filter)

NS = "{http://www.xsens.com/mvn/mvnx}"
MVNX_META = ("frameRate", "segmentCount", "configuration", "userScenario", "processingQuality")   # whitelist
M_MVN = np.array([[1.0, 0, 0], [0, 0, 1], [0, -1, 0]])     # MVN global (x fwd, y left, z up) -> OpenSim ground
MVN_SEGMENTS = {"pelvis_imu": "Pelvis", "torso_imu": "T8", "femur_r_imu": "RightUpperLeg",
                "tibia_r_imu": "RightLowerLeg", "calcn_r_imu": "RightFoot", "femur_l_imu": "LeftUpperLeg",
                "tibia_l_imu": "LeftLowerLeg", "calcn_l_imu": "LeftFoot"}        # estimator sensor -> MVN label
FIELDS = {"seg": "orientation", "sensor": "sensorOrientation"}
HEADING_AXIS = {"seg": (1.0, 0, 0), "sensor": (0, 0, -1.0)}   # pelvis axis that points forward when standing
MX_SIGN = {"physical": -1.0, "pipeline": 1.0}   # factor on the raw walker_knee_r mx (= train_mlp.MX_CONVENTIONS)


def fmedial_from(fy, mx, height_m, convention="physical"):
    """F_medial (BW) from the surrogate's raw outputs: 0.5*|fy| + s*mx*H/0.045 (fy in BW, mx in BW*H)."""
    if convention not in MX_SIGN:
        raise ValueError(f"convention must be one of {sorted(MX_SIGN)}, got {convention!r}")
    return 0.5 * np.abs(np.asarray(fy, float)) + MX_SIGN[convention] * np.asarray(mx, float) * height_m / 0.045
FEET = {"r": "calcn_r_imu", "l": "calcn_l_imu"}

GAIT2392_SIGN = {"knee_angle_r": -1.0, "knee_angle_l": -1.0}  # gait2392 -> Rajagopal; every other coordinate: +1
C13 = ["pelvis_tilt", "pelvis_list", "pelvis_rotation", "hip_flexion_r", "hip_adduction_r", "hip_rotation_r",
       "knee_angle_r", "ankle_angle_r", "hip_flexion_l", "hip_adduction_l", "hip_rotation_l", "knee_angle_l",
       "ankle_angle_l"]                                       # the surrogate's joint-angle inputs
LUMBAR = ["lumbar_extension", "lumbar_bending", "lumbar_rotation"]
SCORED = C13 + LUMBAR                                         # lumbar: not an input, but it moves the COM input
DEF_COORDS = ["pelvis_tilt"] + LUMBAR                         # trunk / pelvis definition offsets, gait2392 vs MVN
DEF_SWAPS = {"lumbar": LUMBAR, "tilt": ["pelvis_tilt"], "lumbar+tilt": DEF_COORDS}
WRAP = {"pelvis_rotation"}
CONTACT_BW, G = 0.05, 9.81
WINDOW, FS_SUR = 20, 100.0                                    # surrogate window: frames k-10 .. k+9 at 100 Hz
SCORE_AFTER_S = 3.0
CONTACT_SOURCES = ("gyro", "grf", "insole")


# ─── Reading the data (no code from the dataset is run; files are parsed as data) ────────────────────────────────

def read_mvnx(path, fields=("orientation",), max_frames=None):
    """Stream an MVNX file (MVN 2021, version 4).

    Returns meta (only the MVNX_META header attributes and the MVN version), the segment / sensor / joint labels in
    file order, index and time_ms of the normal frames, special frames ({type: {field: values}}) and one
    (n_frames, k) array per requested field present. max_frames stops after that many normal frames."""
    meta, segs, sensors, joints, special = {}, [], [], [], {}
    rows, idx, tms = {f: [] for f in fields}, [], []
    with open(path, "rb") as fh:
        for ev, el in ET.iterparse(fh, events=("start", "end")):
            tag = el.tag.rsplit("}", 1)[-1]
            if ev == "start":
                if tag == "subject":                      # attributes are complete at the start tag
                    meta.update({k: el.get(k) for k in MVNX_META if el.get(k) is not None})
                elif tag == "mvn":
                    meta["mvn_version"] = el.get("version")
                continue
            if tag == "frame":
                rec = {}
                for f in fields:
                    c = el.find(NS + f)
                    if c is not None and c.text and c.text.strip():
                        rec[f] = np.array(c.text.split(), float)
                if el.get("type") == "normal":
                    for f in fields:
                        rows[f].append(rec.get(f))
                    idx.append(int(el.get("index")))
                    tms.append(float(el.get("time")))
                else:
                    special[el.get("type")] = rec
                el.clear()
                if max_frames is not None and len(idx) >= max_frames:
                    break
            elif tag == "segment" and el.get("id") is not None:
                segs.append(el.get("label"))
            elif tag == "sensor" and el.get("label"):
                sensors.append(el.get("label"))
            elif tag == "joint" and el.find(NS + "connector1") is not None:
                joints.append(el.get("label"))
    out = dict(meta=meta, segments=segs, sensors=sensors, joints=joints, special=special,
               index=np.array(idx, int), time_ms=np.array(tms))
    for f in fields:
        if rows[f] and all(r is not None for r in rows[f]):
            out[f] = np.vstack(rows[f])
    return out


def read_mot(path):
    """OpenSim .mot/.sto table: whitespace separated, trailing tabs allowed. -> dict(names, data, in_degrees)."""
    with open(path) as f:
        lines = f.read().splitlines()
    end = next(i for i, line in enumerate(lines) if line.strip().lower() == "endheader")
    head = {k.strip().lower(): v.strip().lower() for k, v in (line.split("=", 1) for line in lines[:end] if "=" in line)}
    names = lines[end + 1].split()
    data = np.array([line.split() for line in lines[end + 2:] if line.strip()], float)
    if data.shape[1] != len(names):
        raise ValueError(f"{path}: {data.shape[1]} columns for {len(names)} names")
    return dict(names=names, data=data, in_degrees=head.get("indegrees") == "yes")


def col(tab, name):
    return tab["data"][:, tab["names"].index(name)]


def read_trial_info(mat_path):
    """Mass (kg), height (m) and the MVN -> IK clock offset L (s, None if absent) from Subj<XX>_<trial>.mat."""
    import scipy.io as sio
    ds = sio.loadmat(mat_path, variable_names=["Datastr"])["Datastr"][0, 0]
    info = ds["Info"][0, 0]
    out = dict(mass=float(info["subjMass"].ravel()[0]), height=float(info["subjHeight"].ravel()[0]), L=None)
    try:
        out["L"] = 0.5 + float(ds["Resample"][0, 0]["Sych"][0, 0]["DeltaT"][0, 0]["Motion"].ravel()[0])
    except (KeyError, ValueError, IndexError):
        pass
    return out


def trial_paths(root, subj, trial):
    r = os.path.join(root, f"Subj{subj}", f"Subj{subj}")
    dd = os.path.join(r, "OS", "DataFiles", f"Subj{subj}{trial}")
    return dict(mvnx=os.path.join(r, "Xsens", f"{trial}.mvnx"), static=os.path.join(r, "Xsens", "static_pose.mvnx"),
                mat=os.path.join(r, f"Subj{subj}_{trial}.mat"), ik=dd + "IK.mot", ik_imu=dd + "IK_IMU.mot",
                xld=dd + "XLD.mot", insole=dd + "XLD_INSOLE.mot")


# ─── Conventions ──────────────────────────────────────────────────────────────────────────────────────────────────

def to_rajagopal(tab):
    """gait2392 IK table (degrees) -> (N, 19) in pn3_redis COORDS order: radians, metres, knee flexion positive."""
    if not tab["in_degrees"]:
        raise ValueError("expected an IK table in degrees")
    q = np.zeros((len(tab["data"]), len(COORDS)))
    for j, c in enumerate(COORDS):
        v = col(tab, c)
        q[:, j] = v if c in TRANSLATIONS else np.radians(v) * GAIT2392_SIGN.get(c, 1.0)
    return q


def mvn_rotations(d, field):
    """Rotations of the 8 estimator bodies from one MVNX field, as {sensor key: Rotation}, plus {key: dropout mask}.

    Dropouts (all-zero quaternions, raw sensors only) hold the last good orientation for the contact detector; the
    mask lets the UKF skip that measurement."""
    labels = d["segments"] if field == "orientation" else d["sensors"]
    rots, bad = {}, {}
    for key, lab in MVN_SEGMENTS.items():
        i = labels.index(lab)
        w = d[field][:, 4 * i:4 * i + 4].copy()
        b = np.linalg.norm(w, axis=1) < 0.5
        if b.all():
            raise ValueError(f"{field} {lab}: no valid quaternion")
        if b.any():
            good = np.flatnonzero(~b)
            prev = good[np.clip(np.searchsorted(good, np.flatnonzero(b)) - 1, 0, len(good) - 1)]
            w[b] = w[prev]
        rots[key], bad[key] = Rotation.from_quat(w[:, [1, 2, 3, 0]]), b       # MVNX (w, x, y, z) -> scipy (x, y, z, w)
    return rots, bad


def resample(t, rots, fs, bad=None):
    """Slerp every rotation track onto a uniform grid t[0] + k / fs; dropout masks spread to any touched sample."""
    tn = t[0] + np.arange(int(np.floor((t[-1] - t[0]) * fs + 1e-9)) + 1) / fs
    out = {k: Slerp(t, r)(tn) for k, r in rots.items()}
    if bad is None:
        return tn, out
    return tn, out, {k: np.interp(tn, t, b.astype(float)) > 0 for k, b in bad.items()}


def mvn_heading(R):
    """Heading (rad) of a segment's forward (x) axis about MVN's vertical; equals the OpenSim pelvis_rotation sense."""
    x = R.apply([1.0, 0, 0])
    return np.arctan2(x[..., 1], x[..., 0])


def vendor_heading_removed(q, psi):
    """IK_IMU (Rajagopal order) with the MVN heading psi (rad) taken out of pelvis_rotation, wrapped to +-pi."""
    out = q.copy()
    j = COORDS.index("pelvis_rotation")
    out[:, j] = (out[:, j] - psi + np.pi) % (2 * np.pi) - np.pi
    return out


def swap_coords(q, src, coords):
    """q (N, 19) with the columns `coords` taken from src (same frames)."""
    out = q.copy()
    j = [COORDS.index(c) for c in coords]
    out[:, j] = src[:, j]
    return out


def shift_coords(q, delta):
    """q (N, 19) plus a constant {coord: offset (rad)}."""
    out = q.copy()
    for c, d in delta.items():
        out[:, COORDS.index(c)] += d
    return out


def definition_paths(q_o, q_v, ok):
    """O with the vendor's trunk / pelvis definition: V's coordinates swapped in (DEF_SWAPS), and O shifted by the
    trial's mean V - O on DEF_COORDS over the frames `ok` (the constant part only). -> {path name: (N, 19)}."""
    out = {f"O/V:{name}": swap_coords(q_o, q_v, cs) for name, cs in DEF_SWAPS.items()}
    d = {c: float(np.mean(q_v[ok, COORDS.index(c)] - q_o[ok, COORDS.index(c)])) for c in DEF_COORDS}
    out["O+Vbias:lumbar+tilt"] = shift_coords(q_o, d)
    return out


def angular_speed_deg(t, R):
    """Angular speed (deg/s) between consecutive samples, first sample repeated."""
    w = np.degrees(np.linalg.norm((R[:-1].inv() * R[1:]).as_rotvec(), axis=1)) / np.diff(t)
    return np.r_[w[:1], w]


def standing_window(t, rots, dur_s=1.0, max_deg_s=10.0, t_from=0.0, t_to=None):
    """Index range of the first dur_s window (start in [t_from, t_to]) in which every body turns slower than
    max_deg_s, or None. Quiet standing in static_pose.mvnx stays below 3.1 deg/s (p95)."""
    w = np.max([angular_speed_deg(t, r) for r in rots.values()], axis=0)
    n = int(round(dur_s / np.median(np.diff(t))))
    quiet = np.convolve((w < max_deg_s).astype(float), np.ones(n), "valid") >= n - 0.5
    ok = (t[:len(quiet)] >= t_from) & ((t[:len(quiet)] <= t_to) if t_to is not None else True)
    hit = np.flatnonzero(quiet & ok)
    return None if len(hit) == 0 else (int(hit[0]), int(hit[0]) + n)


def calibration_frames(kind, rots, window):
    """Calibration orientations {key: Rotation (n,)} for the estimator, from the standing window of `rots`.

    static:   the measured standing orientations (the estimator then assumes the default pose q = 0)
    mvnmodel: MVN's identity orientation turned to the window's mean pelvis heading, i.e. MVN's model pose = q0"""
    a, b = window
    if kind == "static":
        return {k: r[a:b] for k, r in rots.items()}
    if kind == "mvnmodel":
        psi = float(np.angle(np.mean(np.exp(1j * mvn_heading(rots["pelvis_imu"][a:b])))))
        return {k: Rotation.from_euler("z", np.full((b - a, 1), psi)) for k in rots}
    raise ValueError(f"unknown calibration {kind!r}")


def wxyz(R):
    q = R.as_quat()
    return np.c_[q[:, 3], q[:, :3]]


# ─── Metrics (pure functions, unit-tested) ────────────────────────────────────────────────────────────────────────

def wrap_deg(e):
    return (np.asarray(e) + 180.0) % 360.0 - 180.0


def angle_stats(est, ref, wrap=False):
    """RMSE, bias (est - ref), SD of the error and Pearson r for one coordinate (degrees)."""
    est, ref = np.asarray(est, float), np.asarray(ref, float)
    e = wrap_deg(est - ref) if wrap else est - ref
    if wrap:                                                       # correlate on the reference's branch
        est = ref + e
    r = float(np.corrcoef(est, ref)[0, 1]) if len(e) > 2 and est.std() > 1e-9 and ref.std() > 1e-9 else np.nan
    return dict(rmse=float(np.sqrt(np.mean(e ** 2))), bias=float(e.mean()), sd=float(e.std()), r=r, n=int(len(e)))


def transitions(t, flags):
    """Onset and offset times of a boolean series: the first sample of each on / off run."""
    f = np.asarray(flags, bool).astype(int)
    d = np.diff(f)
    return np.asarray(t)[1:][d == 1], np.asarray(t)[1:][d == -1]


def event_errors(t_ref, t_test, max_s=0.25):
    """Signed timing error (s) of the nearest test event for each reference event within max_s; unmatched -> NaN."""
    t_test = np.sort(np.asarray(t_test, float))
    out = np.full(len(t_ref), np.nan)
    if len(t_test) == 0:
        return out
    for i, tr in enumerate(t_ref):
        j = np.searchsorted(t_test, tr)
        cand = t_test[max(j - 1, 0):j + 1]
        k = np.argmin(np.abs(cand - tr))
        if abs(cand[k] - tr) <= max_s:
            out[i] = cand[k] - tr
    return out


def stance_intervals(t, flags, t_min=-np.inf, t_max=np.inf, min_s=0.2):
    """(onset, offset) of every complete stance (on run) inside [t_min, t_max], at least min_s long."""
    on, off = transitions(t, flags)
    out = []
    for a in on:
        later = off[off > a]
        if len(later) and a >= t_min and later[0] <= t_max and later[0] - a >= min_s:
            out.append((float(a), float(later[0])))
    return out


def stride_peaks(t, f, intervals):
    """(n, 2) maxima of f in the first and second half of each stance interval (the two F_medial humps)."""
    t, f = np.asarray(t), np.asarray(f)
    out = []
    for a, b in intervals:
        m = 0.5 * (a + b)
        h1, h2 = (t >= a) & (t < m), (t >= m) & (t < b)
        out.append((f[h1].max() if h1.any() else np.nan, f[h2].max() if h2.any() else np.nan))
    return np.array(out).reshape(-1, 2)


def fmed_stats(test, ref, stance, peaks_test, peaks_ref):
    """F_medial agreement of a test path with a reference on the same frames (BW)."""
    test, ref, stance = np.asarray(test, float), np.asarray(ref, float), np.asarray(stance, bool)
    e = test - ref
    r = float(np.corrcoef(test, ref)[0, 1]) if len(e) > 2 and test.std() > 1e-9 and ref.std() > 1e-9 else np.nan
    dp = peaks_test - peaks_ref
    ok = np.all(np.isfinite(dp), axis=1)
    out = dict(n=int(len(e)), mae=float(np.mean(np.abs(e))), rmse=float(np.sqrt(np.mean(e ** 2))),
               bias=float(e.mean()), r=r, ref_mean=float(ref.mean()), test_mean=float(test.mean()),
               mae_rstance=float(np.mean(np.abs(e[stance]))) if stance.any() else np.nan,
               bias_rstance=float(np.mean(e[stance])) if stance.any() else np.nan, n_strides=int(ok.sum()))
    for j, name in enumerate(("peak1", "peak2")):
        d = dp[ok, j]
        out.update({f"{name}_mae": float(np.mean(np.abs(d))) if len(d) else np.nan,
                    f"{name}_bias": float(np.mean(d)) if len(d) else np.nan,
                    f"{name}_ref": float(np.mean(peaks_ref[ok, j])) if len(d) else np.nan})
    return out


def loso_offsets(rows, coords=DEF_COORDS):
    """Leave-one-speed-out constant offsets of each kinematic path against O.

    rows: angle rows (dicts or a DataFrame with subject, trial, test, ref, coord, bias in deg). For every (subject,
    trial, test) with ref O, the offset on each of `coords` is the mean per-trial bias over the subject's OTHER trials
    (speeds). -> {(subject, trial, test): {coord: offset (rad)}}; a trial is left out when a coordinate has no other
    trial to take it from."""
    import pandas as pd
    df = pd.DataFrame(rows) if not isinstance(rows, pd.DataFrame) else rows
    df = df[(df["ref"] == "O") & df["coord"].isin(coords)]
    out = {}
    for (subj, test), g in df.groupby(["subject", "test"]):
        piv = g.pivot_table(index="trial", columns="coord", values="bias", aggfunc="mean")
        for trial in piv.index:
            other = piv.drop(index=trial)
            d = {c: float(other[c].mean()) if c in other else np.nan for c in coords}
            if all(np.isfinite(v) for v in d.values()):
                out[(subj, trial, test)] = {c: float(np.radians(v)) for c, v in d.items()}
    return out


def xcorr_lag(t_ref, ref, t_sig, sig, max_lag=0.2, step=0.001):
    """Lag (s) maximising corr(sig(t_ref + lag), ref(t_ref)): > 0 means the signal is late. -> (lag, r)."""
    best = (np.nan, -2.0)
    for lag in np.arange(-max_lag, max_lag + step / 2, step):
        tt = t_ref + lag
        ok = (tt >= t_sig[0]) & (tt <= t_sig[-1])
        if ok.sum() < 100:
            continue
        r = np.corrcoef(np.interp(tt[ok], t_sig, sig), ref[ok])[0, 1]
        if r > best[1]:
            best = (round(float(lag), 6), float(r))
    return best


# ─── Pipeline pieces ──────────────────────────────────────────────────────────────────────────────────────────────

def load_opensim(root, subj, trial, max_s=None):
    """The OpenSim side of one trial (no MVNX): IK clock, optical and vendor angles, contact flags, mass, height."""
    p = trial_paths(root, subj, trial)
    info = read_trial_info(p["mat"])
    ik, ik_imu, xld = read_mot(p["ik"]), read_mot(p["ik_imu"]), read_mot(p["xld"])
    ins = read_mot(p["insole"]) if os.path.exists(p["insole"]) else None
    t_ik = col(ik, "time")
    keep = t_ik <= (max_s if max_s is not None else np.inf)
    for tab in (ik, ik_imu, xld) + ((ins,) if ins else ()):
        if len(tab["data"]) != len(t_ik) or not np.allclose(col(tab, "time"), t_ik):
            raise ValueError(f"Subj{subj} {trial}: OpenSim tables are not on one clock")
        tab["data"] = tab["data"][keep]
    t_ik = t_ik[keep]
    q_o = to_rajagopal(ik)
    BW = info["mass"] * G
    flags = {"grf": {"r": col(xld, "ground_force_vy") > CONTACT_BW * BW, "l": col(xld, "l_ground_force_vy") > CONTACT_BW * BW}}
    if ins is not None:
        flags["insole"] = {"r": col(ins, "ground_force_vy") > CONTACT_BW * BW,
                           "l": col(ins, "l_ground_force_vy") > CONTACT_BW * BW}
    grf_bw = float(np.mean(col(xld, "ground_force_vy") + col(xld, "l_ground_force_vy")) / BW)
    return dict(subj=subj, trial=trial, speed=WALK_SPEED_MPS.get(trial), mass=info["mass"], height=info["height"],
                L=info["L"], grf_bw_ratio=grf_bw, t_ik=t_ik, q_o=q_o, q_v_raw=to_rajagopal(ik_imu), flags=flags,
                paths=p)


def load_trial(root, subj, trial, max_s=None):
    """Everything one trial needs, on the IK clock. max_s truncates (quick pilots and tests)."""
    tr = load_opensim(root, subj, trial, max_s)
    p, L, t_ik, q_o = tr.pop("paths"), tr["L"], tr["t_ik"], tr["q_o"]
    max_frames = None if max_s is None else int((max_s + (L or 1.5) + 1.5) * 240)
    d = read_mvnx(p["mvnx"], tuple(FIELDS.values()), max_frames=max_frames)
    ds = read_mvnx(p["static"], tuple(FIELDS.values()))
    fs_mvn, fs_static = float(d["meta"]["frameRate"]), float(ds["meta"]["frameRate"])
    seg, _ = mvn_rotations(d, "orientation")
    t_raw = d["index"] / fs_mvn
    # knee angle from the MVN segments (relative rotation magnitude) for the sync check / fallback
    knee_mvn = np.degrees((seg["femur_r_imu"].inv() * seg["tibia_r_imu"]).magnitude())
    knee_o = np.degrees(q_o[:, COORDS.index("knee_angle_r")])
    lag_src = "DeltaT"
    if L is None:                                     # no DeltaT: align the MVN knee to the optical knee (0-3 s)
        L, _ = xcorr_lag(t_ik, knee_o, t_raw - 1.5, knee_mvn, max_lag=1.5, step=0.002)
        L, lag_src = L + 1.5, "xcorr"
    t_mvn = t_raw - L
    lag, lag_r = xcorr_lag(t_ik, knee_o, t_mvn, knee_mvn)
    static_t = ds["index"] / fs_static
    tr.update(L=float(L), lag_src=lag_src, sync_lag_ms=1000 * lag, sync_r=lag_r,
              quality=d["meta"].get("processingQuality"), fs_mvn=fs_mvn, t_mvn=t_mvn, mvn=d, static=ds,
              static_t=static_t)
    return tr


def reference_surrogate(m, lb, q, cr, cl, mass, height):
    """Surrogate on a 100 Hz trajectory: LowerBody FK, checkpoint foot point, centred windows k-10..k+9.

    Returns (frame indices k, output dict of (n,) arrays). Same features LiveSurrogate builds online."""
    F = lb.fk(q)
    pel = F["pelvis"][1]
    com = lb.com(F) - pel
    feet = {s: F[f"calcn_{s}"][1] + np.einsum("sij,j->si", F[f"calcn_{s}"][0], m.foot_point) - pel for s in "rl"}
    h = WINDOW // 2
    k = np.arange(h, len(q) - h + 1)
    win = k[:, None] + np.arange(-h, WINDOW - h)[None, :]
    kw = dict(r_contact_window=cr[win].astype(float), l_contact_window=cl[win].astype(float)) if m.include_contact else {}
    out = m.predict_windowed(q[k, :16], com[win], feet["r"][win], feet["l"][win], mass_kg=mass, height_m=height, **kw)
    return k, out


def run_ukf(tr, variant, surrogates, sources=CONTACT_SOURCES, fs=60.0, anchor="gyro", backdate_s=0.2,
            sigma_orient_deg=2.0, calib_s=1.0, mx_convention="physical"):
    """The driver loop (pn3_redis/driver.run) on one trial: calibration, both gyro detectors, UKF, LiveSurrogates.

    surrogates: {label: LiveSurrogate}; each is copied once per contact source (one shared network).
    anchor: contact for the UKF's foot anchoring, "gyro" (the driver's planted detector) or "grf"."""
    field, calib = variant.split("-", 1)
    if calib == "mvnmodel" and field != "seg":
        raise ValueError("mvnmodel calibration needs MVN segment orientations")
    rots, bad = mvn_rotations(tr["mvn"], FIELDS[field])
    t, R, miss = resample(tr["t_mvn"], rots, fs, bad)
    # calibration: still standing at the start of the walking trial, else the static pose (same MVN calibration)
    win = standing_window(t, R, dur_s=calib_s, t_to=t[0] + 2.0)
    cal_src = "trial"
    if win is not None:
        cal_rots = R
    else:
        srots, sbad = mvn_rotations(tr["static"], FIELDS[field])
        st, cal_rots, sbad = resample(tr["static_t"], srots, fs, sbad)
        win = standing_window(st, cal_rots, dur_s=calib_s, t_from=1.0)
        cal_src = "static_pose"
        if win is None:
            raise RuntimeError(f"Subj{tr['subj']}: no still second in static_pose.mvnx")
    cal = calibration_frames(calib, cal_rots, win)
    if calib == "static":                                        # skip raw-sensor dropouts while calibrating
        cal_bad = sbad if cal_src == "static_pose" else miss
    else:
        cal_bad = {k: np.zeros(win[1], bool) for k in cal}
    n_cal = win[1] - win[0]
    est = PN3Estimator(height_m=tr["height"], sensor_frame=M_MVN, heading_axis=HEADING_AXIS[field],
                       sigma_orient_deg=sigma_orient_deg, calib_s=(n_cal - 1.5) / fs)
    qc = {k: wxyz(r) for k, r in cal.items()}
    for i in range(n_cal):                                       # calibration frames just before the first walk frame
        tc = t[0] - (n_cal - i) / fs
        if est.add_calibration_frame(tc, {k: qc[k][i] for k in qc if not cal_bad[k][win[0] + i]}):
            break
    if not est.calibrated:
        raise RuntimeError("calibration did not complete")
    q_w = {k: wxyz(r) for k, r in R.items()}
    Rf = {s: R[k].as_matrix() for s, k in FEET.items()}
    planted = CausalContact(on_deg_s=100, off_deg_s=200)
    stance = CausalContact(on_deg_s=150, off_deg_s=400, backdate_s=backdate_s)
    for det in (planted, stance):                                # reset on the first walking frame (no splice spike)
        det.reset(est.t, {s: Rf[s][0] for s in FEET})
    ext = {src: {s: np.interp(t, tr["t_ik"], tr["flags"][src][s].astype(float)) > 0.5 for s in "rl"}
           for src in sources if src in tr["flags"]}
    surs = {}
    for lab, base in surrogates.items():
        for src in sources:
            if src == "gyro" or src in ext:
                s = copy.copy(base)
                s.buf = deque()
                surs[(lab, src)] = s
    n = len(t)
    Q = np.zeros((n, len(COORDS)))
    C = {k: np.zeros((n, 2), bool) for k in ("stance", "planted")}
    onsets = {"r": [], "l": []}
    sur_out = {key: [] for key in surs}
    t_prev, t0 = est.t, time.time()
    for i in range(n):
        dt = t[i] - t_prev
        rf = {s: Rf[s][i] for s in FEET}
        g_planted, g_stance = planted.update(t[i], rf, dt), stance.update(t[i], rf, dt)
        on = dict(stance.onsets)
        for s, ton in on.items():
            onsets[s].append(ton)
        a_flags = g_planted if anchor == "gyro" else {s: bool(ext[anchor][s][i]) for s in "rl"}
        st = est.step(t[i], {k: q_w[k][i] for k in q_w if not miss[k][i]}, a_flags)
        Q[i] = st["q"]
        C["stance"][i] = g_stance["r"], g_stance["l"]
        C["planted"][i] = a_flags["r"], a_flags["l"]
        for (lab, src), sur in surs.items():
            s = dict(st)
            if src == "gyro":
                s["contact"], s["contact_onset"] = dict(g_stance), on
            else:
                s["contact"], s["contact_onset"] = {sd: bool(ext[src][sd][i]) for sd in "rl"}, {}
            o = sur.update(s)
            if o:
                sur_out[(lab, src)].append((o["surrogate_time"], float(fmedial_from(o["fy"], o["mx"], sur.height,
                                                                                    mx_convention)), o["fy"], o["mx"],
                                            o["contact_used"]["r"], o["contact_used"]["l"]))
        t_prev = t[i]
    return dict(t=t, q=Q, stance=C["stance"], planted=C["planted"], onsets={s: np.array(v) for s, v in onsets.items()},
                sur={k: np.array(v) for k, v in sur_out.items()}, runtime_s=time.time() - t0, cal_src=cal_src,
                cal_window_s=[float(win[0] / fs), float(win[1] / fs)],
                heading_deg=float(np.degrees(np.arctan2(est.A[0, 2], est.A[0, 0]))),
                dropout_frac={k: float(v.mean()) for k, v in miss.items()})


# ─── One job: (trial, variant) ────────────────────────────────────────────────────────────────────────────────────

def _labels(tr):
    return dict(subject=tr["subj"], trial=tr["trial"], speed=tr["speed"], quality=tr["quality"])


def _contact_rows(tr, variant, u, t_eval_ok):
    """Detector flags vs treadmill flags on the UKF frames inside the scoring window."""
    t, rows = u["t"], []
    ref = {s: np.interp(t, tr["t_ik"], tr["flags"]["grf"][s].astype(float)) > 0.5 for s in "rl"}
    sources = {"gyro_stance": {s: u["stance"][:, j] for j, s in enumerate("rl")},
               "planted": {s: u["planted"][:, j] for j, s in enumerate("rl")}}
    if "insole" in tr["flags"]:
        sources["insole"] = {s: np.interp(t, tr["t_ik"], tr["flags"]["insole"][s].astype(float)) > 0.5 for s in "rl"}
    lo, hi = t[t_eval_ok][0], t[t_eval_ok][-1]
    for name, fl in sources.items():
        for s in "rl":
            on_r, off_r = transitions(tr["t_ik"], tr["flags"]["grf"][s])
            on_r, off_r = on_r[(on_r > lo) & (on_r < hi)], off_r[(off_r > lo) & (off_r < hi)]
            on_t, off_t = transitions(t, fl[s])
            e_on, e_off = event_errors(on_r, on_t), event_errors(off_r, off_t)
            row = dict(_labels(tr), variant=variant, source=name, side=s,
                       agreement=float(np.mean(fl[s][t_eval_ok] == ref[s][t_eval_ok])),
                       stance_frac=float(np.mean(fl[s][t_eval_ok])), stance_frac_ref=float(np.mean(ref[s][t_eval_ok])),
                       n_ref_on=int(len(on_r)), on_matched=float(np.mean(np.isfinite(e_on))) if len(e_on) else np.nan,
                       off_matched=float(np.mean(np.isfinite(e_off))) if len(e_off) else np.nan,
                       on_err_med_ms=1000 * float(np.nanmedian(e_on)) if np.isfinite(e_on).any() else np.nan,
                       off_err_med_ms=1000 * float(np.nanmedian(e_off)) if np.isfinite(e_off).any() else np.nan)
            if name == "gyro_stance":                            # the back-dated heel strikes the surrogate gets
                e_bd = event_errors(on_r, u["onsets"][s])
                row["backdated_on_err_med_ms"] = 1000 * float(np.nanmedian(e_bd)) if np.isfinite(e_bd).any() else np.nan
                row["backdated_on_matched"] = float(np.mean(np.isfinite(e_bd))) if len(e_bd) else np.nan
            rows.append(row)
    return rows


def run_job(job):
    """All metrics for one (subject, trial, variant). Returns plain rows, picklable; errors are returned, not raised."""
    a = job["args"]
    t_start = time.time()
    try:
        import torch
        torch.set_num_threads(1)
        from pn3_redis.surrogate import LiveSurrogate
        tr = load_trial(a["data"], job["subj"], job["trial"], max_s=a.get("max_s"))
        conv = a.get("mx_convention", "physical")
        base = {lab: LiveSurrogate(p, tr["mass"], tr["height"]) for lab, p in zip(a["labels"], a["ckpt"])}
        lb = LowerBody(height_m=tr["height"])
        t_ik, gr = tr["t_ik"], tr["flags"]["grf"]
        # vendor path: MVN joint angles with the static-pose MVN heading removed (the UKF calibration's reference)
        srots, _ = mvn_rotations(tr["static"], "orientation")
        sw = standing_window(tr["static_t"], srots, t_from=1.0) or (0, len(tr["static_t"]))
        psi_static = float(np.angle(np.mean(np.exp(1j * mvn_heading(srots["pelvis_imu"][sw[0]:sw[1]])))))
        q_v = vendor_heading_removed(tr["q_v_raw"], psi_static)
        t0s = max(tr["t_mvn"][0] + SCORE_AFTER_S, t_ik[0])
        t1s = t_ik[-1] - WINDOW / FS_SUR
        ok_i = (t_ik >= t0s) & (t_ik <= t1s)
        paths_100 = {"O": tr["q_o"], "V": q_v}
        if job["first_variant"]:                       # trunk / pelvis definition alone (variant independent)
            paths_100.update(definition_paths(tr["q_o"], q_v, ok_i))
        ref_out = {}
        for lab, s in base.items():
            for pth, q in paths_100.items():
                k, out = reference_surrogate(s.m, lb, q, gr["r"], gr["l"], tr["mass"], tr["height"])
                ref_out[(lab, pth)] = (t_ik[k], fmedial_from(out["fy"], out["mx"], tr["height"], conv), out["fy"],
                                       out["mx"])
        u = run_ukf(tr, job["variant"], base, sources=a["contact"], fs=a["fs"], anchor=a["anchor"],
                    backdate_s=a["backdate_s"], sigma_orient_deg=a["sigma_orient_deg"], mx_convention=conv)
        lab0 = _labels(tr)
        rows_ang, rows_fm = [], []
        # ── joint angles: U vs O and U vs V on the UKF frames, V vs O on the IK frames
        ok_u = (u["t"] >= t0s) & (u["t"] <= t1s)
        deg = {"O": np.degrees(tr["q_o"]), "V": np.degrees(q_v)}
        U = np.degrees(u["q"])
        for c in SCORED:
            j = COORDS.index(c)
            for ref in ("O", "V"):
                rv = np.interp(u["t"][ok_u], t_ik, np.unwrap(np.radians(deg[ref][:, j])) * 180 / np.pi)
                rows_ang.append(dict(lab0, test=f"U:{job['variant']}", ref=ref, coord=c,
                                     **angle_stats(U[ok_u, j], rv, wrap=c in WRAP)))
            if job["first_variant"]:
                rows_ang.append(dict(lab0, test="V", ref="O", coord=c,
                                     **angle_stats(deg["V"][ok_i, j], deg["O"][ok_i, j], wrap=c in WRAP)))
        lag_ukf, r_ukf = xcorr_lag(t_ik[ok_i], deg["O"][ok_i, COORDS.index("knee_angle_r")], u["t"],
                                   U[:, COORDS.index("knee_angle_r")])
        # ── F_medial: every path on the 100 Hz reference frames inside the scoring window
        intervals = stance_intervals(t_ik, gr["r"], t0s, t1s)
        match = {}
        for lab in a["labels"]:
            to, fo = ref_out[(lab, "O")][:2]
            ok = (to >= t0s) & (to <= t1s)
            tt, fref = to[ok], fo[ok]
            stn = gr["r"][np.searchsorted(t_ik, tt)]
            series = {"O": fref, "V": np.interp(tt, *ref_out[(lab, "V")][:2])}
            for src in a["contact"]:
                arr = u["sur"].get((lab, src))
                if arr is not None and len(arr):
                    series[f"U:{job['variant']}:{src}"] = np.interp(tt, arr[:, 0], arr[:, 1])
            dser = {p: np.interp(tt, *ref_out[(lab, p)][:2]) for p in paths_100 if p not in ("O", "V")}
            peaks = {k: stride_peaks(tt, v, intervals) for k, v in {**series, **dser}.items()}
            pairs = [(k, "O") for k in series if k.startswith("U:")]
            if job["first_variant"]:
                pairs += [("V", "O")] + [(p, "O") for p in dser]
            grf_key = f"U:{job['variant']}:grf"
            pairs += [(k, grf_key) for k in series if k.startswith("U:") and k != grf_key and grf_key in series]
            for test, ref in pairs:
                ser = series if test in series else dser
                rows_fm.append(dict(lab0, ckpt=lab, test=test, ref=ref,
                                    **fmed_stats(ser[test], series[ref], stn, peaks[test], peaks[ref])))
            # what the offset-matched pass needs: the test series on the scored frames (V only once per trial)
            match[lab] = dict(t=tt, stance=stn, intervals=intervals,
                              series={k: v for k, v in series.items()
                                      if k.startswith("U:") or (k == "V" and job["first_variant"])})
            if job.get("save_frames"):
                job.setdefault("_frames", {})[lab] = dict(t=tt, stance_r=stn, **series)
        rows_ct = _contact_rows(tr, job["variant"], u, ok_u)
        trial_row = dict(lab0, variant=job["variant"], mass=tr["mass"], height=tr["height"], L=tr["L"],
                         lag_src=tr["lag_src"], sync_lag_ms=tr["sync_lag_ms"], sync_r=tr["sync_r"],
                         ukf_knee_lag_ms=1000 * lag_ukf, ukf_knee_r=r_ukf, grf_bw_ratio=tr["grf_bw_ratio"],
                         static_heading_deg=float(np.degrees(psi_static)), ukf_heading_deg=u["heading_deg"],
                         cal_src=u["cal_src"], cal_t0_s=u["cal_window_s"][0], cal_t1_s=u["cal_window_s"][1],
                         ik_dur_s=float(t_ik[-1] - t_ik[0]), scored_s=float(t1s - t0s), n_strides=len(intervals),
                         ukf_frames=len(u["t"]), ukf_runtime_s=u["runtime_s"],
                         ms_per_frame=1000 * u["runtime_s"] / len(u["t"]),
                         max_dropout_frac=max(u["dropout_frac"].values()),
                         knee_at_zero_frac_r=float(np.mean(u["q"][ok_u, COORDS.index("knee_angle_r")] < np.radians(0.5))),
                         job_s=time.time() - t_start)
        res = dict(job=f"Subj{job['subj']}_{job['trial']}_{job['variant']}", angles=rows_ang, fmedial=rows_fm,
                   contact=rows_ct, trial=trial_row, match=dict(labels=lab0, series=match))
        if job.get("save_frames"):
            res["frames"] = dict(t_ik=t_ik, q_o=tr["q_o"].astype(np.float32), q_v=q_v.astype(np.float32),
                                 grf_r=gr["r"], grf_l=gr["l"], t_u=u["t"], q_u=u["q"].astype(np.float32),
                                 stance_u=u["stance"], fmed=job["_frames"])
        return res
    except Exception as e:                                             # noqa: BLE001 - reported per job
        import traceback
        return dict(job=f"Subj{job['subj']}_{job['trial']}_{job['variant']}", error=f"{type(e).__name__}: {e}",
                    tb=traceback.format_exc(limit=4))


# ─── Second pass: offset-matched optical reference (needs every trial's angle rows first) ────────────────────────

def match_jobs(results, args):
    """One job per trial: each kinematic path's leave-one-speed-out offset (loso_offsets) and its F_medial series."""
    ok = [r for r in results if "error" not in r]
    offs = loso_offsets([row for r in ok for row in r["angles"]])
    by_trial = {}
    for r in ok:
        lab0 = r["match"]["labels"]
        key = (lab0["subject"], lab0["trial"])
        j = by_trial.setdefault(key, dict(args=args, subj=key[0], trial=key[1], labels=lab0, series={}))
        for lab, m in r["match"]["series"].items():
            s = j["series"].setdefault(lab, dict(t=m["t"], stance=m["stance"], intervals=m["intervals"], series={}))
            s["series"].update(m["series"])
    jobs = []
    for (subj, trial), j in sorted(by_trial.items()):
        paths = {k.rsplit(":", 1)[0] if k.startswith("U:") else k for s in j["series"].values() for k in s["series"]}
        j["offsets"] = {p: offs[(subj, trial, p)] for p in sorted(paths) if (subj, trial, p) in offs}
        if j["offsets"]:
            jobs.append(j)
    return jobs


def run_match_job(job):
    """F_medial of every test path against Omatch:<path> (O plus that path's offset) on one trial."""
    a = job["args"]
    name = f"Subj{job['subj']}_{job['trial']}_match"
    try:
        import torch
        torch.set_num_threads(1)
        from inference_posture_surrogate import JCFPostureSurrogate
        tr = load_opensim(a["data"], job["subj"], job["trial"], max_s=a.get("max_s"))
        lb = LowerBody(height_m=tr["height"])
        t_ik, gr = tr["t_ik"], tr["flags"]["grf"]
        rows = []
        for lab, p in zip(a["labels"], a["ckpt"]):
            if lab not in job["series"]:
                continue
            m = JCFPostureSurrogate(p, device="cpu")
            sv = job["series"][lab]
            tt, stn, iv = sv["t"], sv["stance"], sv["intervals"]
            for path, delta in job["offsets"].items():
                k, out = reference_surrogate(m, lb, shift_coords(tr["q_o"], delta), gr["r"], gr["l"], tr["mass"],
                                             tr["height"])
                fref = np.interp(tt, t_ik[k], fmedial_from(out["fy"], out["mx"], tr["height"],
                                                           a.get("mx_convention", "physical")))
                pk_ref = stride_peaks(tt, fref, iv)
                for test, f in sv["series"].items():
                    if test == path or test.startswith(path + ":"):
                        rows.append(dict(job["labels"], ckpt=lab, test=test, ref=f"Omatch:{path}",
                                         **fmed_stats(f, fref, stn, stride_peaks(tt, f, iv), pk_ref)))
        offsets = [dict(job["labels"], test=path, **{f"{c}_deg": float(np.degrees(v)) for c, v in d.items()})
                   for path, d in job["offsets"].items()]
        return dict(job=name, fmedial=rows, offsets=offsets)
    except Exception as e:                                             # noqa: BLE001 - reported per job
        import traceback
        return dict(job=name, error=f"{type(e).__name__}: {e}", tb=traceback.format_exc(limit=4))


def run_pool(fn, jobs, workers, t0, tag=""):
    results = []
    if workers > 1 and len(jobs) > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(min(workers, len(jobs)), maxtasksperchild=4) as pool:
            it = pool.imap_unordered(fn, jobs)
            for i, r in enumerate(it, 1):
                results.append(r)
                print(f"[{tag}{i}/{len(jobs)} {time.time() - t0:5.0f} s] {r['job']}"
                      + (f"  FAILED {r['error']}" if "error" in r else ""), flush=True)
    else:
        for i, j in enumerate(jobs, 1):
            r = fn(j)
            results.append(r)
            print(f"[{tag}{i}/{len(jobs)} {time.time() - t0:5.0f} s] {r['job']}"
                  + (f"  FAILED {r['error']}" if "error" in r else ""), flush=True)
    return results


# ─── Aggregation ──────────────────────────────────────────────────────────────────────────────────────────────────

def _iqr(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if not len(x):
        return dict(median=None, q25=None, q75=None, n=0)
    return dict(median=float(np.median(x)), q25=float(np.percentile(x, 25)), q75=float(np.percentile(x, 75)),
                n=int(len(x)))


def aggregate(df, keys, metrics, subjects=None):
    """{key tuple: {metric: {"all": IQR of subject means, "by_speed": {speed: IQR over subjects}}}}."""
    if subjects is not None:
        df = df[df["subject"].isin(subjects)]
    out = {}
    for key, g in df.groupby(keys, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        subj = g.groupby("subject")[metrics].mean()
        res = {m: dict(all=_iqr(subj[m])) for m in metrics}
        for sp, gs in g.groupby("speed"):
            per = gs.groupby("subject")[metrics].mean()
            for m in metrics:
                res[m].setdefault("by_speed", {})[f"{sp:.2f}"] = _iqr(per[m])
        out["|".join(str(k) for k in key)] = res
    return out


def checkpoint_info(paths, labels):
    import torch
    out = {}
    for lab, p in zip(labels, paths):
        ck = torch.load(p, map_location="cpu", weights_only=True)
        prov = ck.get("provenance") or {}
        out[lab] = dict(path=os.path.relpath(p, REPO) if p.startswith(REPO) else p, input_set=ck.get("input_set"),
                        foot_point=[float(x) for x in ck.get("foot_point", (0.0, 0.0, 0.0))],
                        epoch=int(ck.get("epoch", -1)), val_loss=float(ck.get("val_loss", np.nan)),
                        train_commit=(prov.get("git") or {}).get("commit"))
    return out


def flat_table(df, keys, metrics):
    """Long table, one row per key and speed ("all": subject means over speeds): median, q25, q75 over subjects."""
    import pandas as pd
    rows = []
    for key, g in df.groupby(keys, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        for sp, gs in [("all", g)] + [(f"{s:.2f}", x) for s, x in g.groupby("speed")]:
            per = gs.groupby("subject")[metrics].mean()
            row = dict(zip(keys, key), speed=sp, n_subjects=len(per))
            for m in metrics:
                q = _iqr(per[m])
                row.update({f"{m}_median": q["median"], f"{m}_q25": q["q25"], f"{m}_q75": q["q75"]})
            rows.append(row)
    return pd.DataFrame(rows)


def decomposition_spec(variants):
    """(component, test, ref) rows of decomposition.csv in reading order: the definition difference alone, then each
    kinematic path against the optical path as recorded and against its offset-matched optical path."""
    spec = [("definition: O with MVN lumbar + pelvis tilt", "O/V:lumbar+tilt", "O"),
            ("definition: O with MVN lumbar", "O/V:lumbar", "O"),
            ("definition: O with MVN pelvis tilt", "O/V:tilt", "O"),
            ("definition: O + constant trial-mean MVN offset (lumbar, tilt)", "O+Vbias:lumbar+tilt", "O"),
            ("vendor angles vs O", "V", "O"),
            ("vendor angles vs offset-matched O", "V", "Omatch:V")]
    for v in variants:
        for src, what in (("grf", "kinematics only (treadmill contact)"), ("gyro", "deployment (gyro contact)")):
            spec += [(f"IMU {v}, {what} vs O", f"U:{v}:{src}", "O"),
                     (f"IMU {v}, {what} vs offset-matched O", f"U:{v}:{src}", f"Omatch:U:{v}")]
    return spec


def write_outputs(out_dir, results, args, matched=()):
    import pandas as pd
    ok = [r for r in results if "error" not in r]
    errs = [r for r in list(results) + list(matched) if "error" in r]
    m_ok = [r for r in matched if "error" not in r]
    dfs = {k: pd.DataFrame([row for r in ok for row in r[k]]) for k in ("angles", "fmedial", "contact")}
    if m_ok:
        dfs["fmedial"] = pd.concat([dfs["fmedial"], pd.DataFrame([row for r in m_ok for row in r["fmedial"]])],
                                   ignore_index=True)
        dfs["offsets"] = pd.DataFrame([row for r in m_ok for row in r["offsets"]])
    dfs["trials"] = pd.DataFrame([r["trial"] for r in ok])
    for k, df in dfs.items():
        if len(df):
            df.sort_values([c for c in ("subject", "trial", "variant", "test", "ckpt", "coord", "source", "side")
                            if c in df.columns]).to_csv(os.path.join(out_dir, f"{k}.csv"), index=False,
                                                        float_format="%.5g")
    if not m_ok and os.path.exists(os.path.join(out_dir, "offsets.csv")):
        os.remove(os.path.join(out_dir, "offsets.csv"))                 # stale from a run with the second pass
    # median [IQR] over subjects, overall and per speed, of every (test, ref) pair: the numbers the text quotes
    if len(dfs["angles"]):
        flat_table(dfs["angles"], ["test", "ref", "coord"], ["rmse", "bias", "sd", "r"]).to_csv(
            os.path.join(out_dir, "angles_table.csv"), index=False, float_format="%.4g")
    if len(dfs["fmedial"]):
        fmt = flat_table(dfs["fmedial"], ["ckpt", "test", "ref"],
                         ["mae", "rmse", "bias", "r", "mae_rstance", "peak1_mae", "peak2_mae"])
        fmt.to_csv(os.path.join(out_dir, "fmedial_table.csv"), index=False, float_format="%.4g")
        spec = pd.DataFrame(decomposition_spec(args.variants), columns=["component", "test", "ref"])
        spec["order"] = np.arange(len(spec))
        dec = spec.merge(fmt, on=["test", "ref"]).sort_values(["ckpt", "order", "speed"]).drop(columns="order")
        dec.to_csv(os.path.join(out_dir, "decomposition.csv"), index=False, float_format="%.4g")
    os.makedirs(os.path.join(out_dir, "frames"), exist_ok=True)
    for r in ok:
        if "frames" in r:
            fr = r["frames"]
            flat = {k: v for k, v in fr.items() if k != "fmed"}
            for lab, ser in fr["fmed"].items():
                flat.update({f"fmed|{lab}|{k}": v for k, v in ser.items()})
            np.savez_compressed(os.path.join(out_dir, "frames", f"{r['job']}.npz"), **flat)
    groups = {"all": None, "live": [s for s in SUBJECTS if s not in HD_SUBJECTS], "hd": sorted(HD_SUBJECTS)}
    summary = dict(created=datetime.datetime.now().isoformat(timespec="seconds"),
                   command="python -I " + " ".join(sys.argv), args={k: v for k, v in vars(args).items()},
                   n_jobs=len(results), n_match_jobs=len(matched), n_failed=len(errs),
                   failures=[(r["job"], r["error"]) for r in errs])
    try:
        from provenance import git_info
        summary["git"] = git_info()
    except Exception:                                                  # noqa: BLE001
        pass
    summary["checkpoints"] = checkpoint_info(args.ckpt, args.labels)
    for name, sub in groups.items():
        s = {}
        if len(dfs["angles"]):
            s["angles"] = aggregate(dfs["angles"], ["test", "ref", "coord"], ["rmse", "bias", "sd", "r"], sub)
        if len(dfs["fmedial"]):
            s["fmedial"] = aggregate(dfs["fmedial"], ["ckpt", "test", "ref"],
                                     ["mae", "rmse", "bias", "r", "mae_rstance", "peak1_mae", "peak2_mae",
                                      "peak1_bias", "peak2_bias", "ref_mean", "peak1_ref", "peak2_ref"], sub)
        if len(dfs["contact"]):
            s["contact"] = aggregate(dfs["contact"], ["variant", "source", "side"],
                                     ["agreement", "on_err_med_ms", "off_err_med_ms", "on_matched"]
                                     + (["backdated_on_err_med_ms"] if "backdated_on_err_med_ms" in dfs["contact"] else []),
                                     sub)
        summary[name] = s
    if len(dfs["trials"]):
        tdf = dfs["trials"]
        summary["checks"] = dict(sync_lag_ms=_iqr(tdf["sync_lag_ms"]), sync_lag_abs_max_ms=float(tdf["sync_lag_ms"].abs().max()),
                                 ukf_knee_lag_ms=_iqr(tdf["ukf_knee_lag_ms"]), ms_per_frame=_iqr(tdf["ms_per_frame"]),
                                 grf_bw_ratio_by_subject=tdf.groupby("subject")["grf_bw_ratio"].mean().round(3).to_dict(),
                                 cal_src=tdf["cal_src"].value_counts().to_dict())
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    if os.path.exists(os.path.join(out_dir, "failures.txt")):
        os.remove(os.path.join(out_dir, "failures.txt"))
    if errs:
        with open(os.path.join(out_dir, "failures.txt"), "w") as f:
            for r in errs:
                f.write(f"{r['job']}: {r['error']}\n{r.get('tb', '')}\n")
    headline(out_dir, dfs, args)
    return dfs, summary


def headline(out_dir, dfs, args):
    """Per speed (and over all speeds) median over subjects of the main numbers; printed and saved as CSV."""
    import pandas as pd
    v0, rows = args.variants[0], []
    an, fm, ct = dfs["angles"], dfs["fmedial"], dfs["contact"]
    if not len(an):
        return

    def med(df, mask, m, sp=None):
        d = df[mask] if sp is None else df[mask & (df["speed"] == sp)]
        return float(np.median(d.groupby("subject")[m].mean())) if len(d) else np.nan

    for sp in [None] + sorted(an["speed"].unique()):
        row = dict(speed="all" if sp is None else f"{sp:.2f}")
        for c in ("knee_angle_r", "hip_flexion_r", "ankle_angle_r"):
            for test in (f"U:{v0}", "V"):
                row[f"{c}_rmse_{test.split(':')[0]}"] = med(an, (an["test"] == test) & (an["ref"] == "O") & (an["coord"] == c), "rmse", sp)
        for c in ("lumbar_extension", "pelvis_tilt"):                  # trunk / pelvis definition offsets
            for test in (f"U:{v0}", "V"):
                row[f"{c}_bias_{test.split(':')[0]}"] = med(an, (an["test"] == test) & (an["ref"] == "O") & (an["coord"] == c), "bias", sp)
        for lab in args.labels:
            f = fm[fm["ckpt"] == lab]
            for test, ref, name in ((f"U:{v0}:gyro", "O", "U_gyro"), (f"U:{v0}:grf", "O", "U_grf"), ("V", "O", "V"),
                                    (f"U:{v0}:gyro", f"U:{v0}:grf", "contact_only"),
                                    ("O/V:lumbar+tilt", "O", "definition"), (f"U:{v0}:grf", f"Omatch:U:{v0}", "U_grf_matched"),
                                    ("V", "Omatch:V", "V_matched")):
                row[f"fmed_mae_{name}|{lab}"] = med(f, (f["test"] == test) & (f["ref"] == ref), "mae", sp)
            row[f"fmed_r_U_gyro|{lab}"] = med(f, (f["test"] == f"U:{v0}:gyro") & (f["ref"] == "O"), "r", sp)
            row[f"peak1_mae_U_gyro|{lab}"] = med(f, (f["test"] == f"U:{v0}:gyro") & (f["ref"] == "O"), "peak1_mae", sp)
        if len(ct):
            row["contact_agree_gyro_r"] = med(ct, (ct["variant"] == v0) & (ct["source"] == "gyro_stance") & (ct["side"] == "r"), "agreement", sp)
        rows.append(row)
    h = pd.DataFrame(rows)
    h.to_csv(os.path.join(out_dir, "headline.csv"), index=False, float_format="%.4g")
    with pd.option_context("display.width", 250, "display.max_columns", 50):
        print(h.round(3).to_string(index=False))


# ─── Figures ──────────────────────────────────────────────────────────────────────────────────────────────────────
# Series identity, the same in every figure (validated default categorical order): optical reference = ink,
# IMU deployment (UKF + gyro contact) = slot 1, vendor = slot 2, IMU with treadmill contact = slot 3,
# IMU mvnmodel calibration = slot 4, insole contact = slot 5. fig_decomposition colours the reference instead:
# optical as recorded = slot 6, offset-matched optical = slot 7 (adjacent in the order) plus a hatch.
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COL = dict(U="#2a78d6", V="#eb6834", U_grf="#1baf7a", U_mvn="#eda100", insole="#e87ba4", ref_O="#008300",
           ref_Omatch="#6250d6")


def _style(ax):
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def make_figures(out_dir, variants, labels, example=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    an = pd.read_csv(os.path.join(out_dir, "angles.csv"), dtype={"subject": str})
    fm = pd.read_csv(os.path.join(out_dir, "fmedial.csv"), dtype={"subject": str})
    ct = pd.read_csv(os.path.join(out_dir, "contact.csv"), dtype={"subject": str})
    v0, lab = variants[0], labels[0]
    speeds = sorted(an["speed"].unique())

    def per_speed(df, mask, m):
        med, lo, hi = [], [], []
        for sp in speeds:
            x = df[mask & (df["speed"] == sp)].groupby("subject")[m].mean()
            med.append(np.median(x) if len(x) else np.nan)
            lo.append(np.percentile(x, 25) if len(x) else np.nan)
            hi.append(np.percentile(x, 75) if len(x) else np.nan)
        return np.array(med), np.array(lo), np.array(hi)

    def line(ax, df, mask, m, color, label, ls="-", dx=0.0):
        y, lo, hi = per_speed(df, mask, m)
        x = np.array(speeds) + dx
        ax.errorbar(x, y, yerr=[y - lo, hi - y], color=color, lw=2, ls=ls, marker="o", ms=5, capsize=0,
                    elinewidth=1, label=label)

    # ── summary vs belt speed: (a) knee RMSE, (b) F_medial difference, (c) contact agreement
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.3), constrained_layout=True)
    a0 = an[(an["ref"] == "O") & (an["coord"] == "knee_angle_r")]
    line(axs[0], a0, a0["test"] == f"U:{v0}", "rmse", COL["U"], f"IMU + UKF ({v0})", dx=-0.012)
    if f"U:seg-mvnmodel" in set(a0["test"]) and v0 != "seg-mvnmodel":
        line(axs[0], a0, a0["test"] == "U:seg-mvnmodel", "rmse", COL["U_mvn"], "IMU + UKF (seg-mvnmodel)")
    line(axs[0], a0, a0["test"] == "V", "rmse", COL["V"], "Xsens MVN angles", dx=0.012)
    axs[0].set_ylabel("(a) right knee RMSE vs optical (deg)", fontsize=9)
    f0 = fm[(fm["ckpt"] == lab) & (fm["ref"] == "O")]
    line(axs[1], f0, f0["test"] == f"U:{v0}:gyro", "mae", COL["U"], "IMU, gyro contact", dx=-0.012)
    line(axs[1], f0, f0["test"] == f"U:{v0}:grf", "mae", COL["U_grf"], "IMU, treadmill contact")
    line(axs[1], f0, f0["test"] == "V", "mae", COL["V"], "Xsens MVN angles, treadmill contact", dx=0.012)
    axs[1].set_ylabel(f"(b) F_medial vs optical path, MAE (BW)", fontsize=9)
    c0 = ct[(ct["variant"] == v0) & (ct["side"] == "r")]
    line(axs[2], c0, c0["source"] == "gyro_stance", "agreement", COL["U"], "gyro detector (driver)", dx=-0.012)
    if "insole" in set(c0["source"]):
        line(axs[2], c0, c0["source"] == "insole", "agreement", COL["insole"], "insoles (5 % BW)", dx=0.012)
    axs[2].set_ylabel("(c) right contact agreement with treadmill", fontsize=9)
    axs[2].set_ylim(0.5, 1.0)
    for ax, t in zip(axs, "abc"):
        _style(ax)
        ax.set_xlabel("belt speed (m/s)", fontsize=9)
        ax.set_xticks(speeds)
        ax.set_ylim(bottom=0 if t != "c" else 0.5)
        ax.legend(fontsize=7, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0))
    fig.suptitle(f"median and IQR over subjects; surrogate {lab}; IMU calibration {v0}", fontsize=8, color=MUTED)
    fig.savefig(os.path.join(out_dir, "fig_vs_speed.png"), dpi=200)
    plt.close(fig)

    # ── joint-angle RMSE per coordinate (median over subjects, IQR whiskers)
    tests = [(f"U:{v0}", COL["U"], f"IMU + UKF ({v0})")]
    if "U:seg-mvnmodel" in set(an["test"]) and v0 != "seg-mvnmodel":
        tests.append(("U:seg-mvnmodel", COL["U_mvn"], "IMU + UKF (seg-mvnmodel)"))
    tests.append(("V", COL["V"], "Xsens MVN angles"))
    coords = [c for c in SCORED if c in set(an["coord"])]
    fig, ax = plt.subplots(figsize=(6.5, 4.2 + 0.25 * (len(coords) - len(C13))), constrained_layout=True)
    h = 0.8 / len(tests)
    for j, (test, color, label) in enumerate(tests):
        g = an[(an["test"] == test) & (an["ref"] == "O")]
        subj = g.groupby(["coord", "subject"])["rmse"].mean().unstack()
        y = np.arange(len(coords)) + (j - (len(tests) - 1) / 2) * h
        med = [np.median(subj.loc[c]) for c in coords]
        lo = [np.percentile(subj.loc[c], 25) for c in coords]
        hi = [np.percentile(subj.loc[c], 75) for c in coords]
        ax.barh(y, med, height=h * 0.9, color=color, label=label, edgecolor="white", linewidth=1)
        ax.errorbar(med, y, xerr=[np.subtract(med, lo), np.subtract(hi, med)], fmt="none", ecolor=INK, elinewidth=0.8)
    ax.set_yticks(np.arange(len(coords)), [c.replace("_", " ") for c in coords], fontsize=8)
    xlabel = "RMSE vs optical marker IK (deg), median over subjects"
    if len(coords) > len(C13):                                         # lumbar: scored, not a surrogate input
        ax.axhline(len(C13) - 0.5, color=MUTED, lw=0.8, ls=":")
        xlabel += "\nbelow the dotted line: lumbar, not a surrogate input but it moves the COM input"
    ax.invert_yaxis()
    ax.set_xlabel(xlabel, fontsize=9)
    _style(ax)
    ax.legend(fontsize=8, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=len(tests))
    fig.savefig(os.path.join(out_dir, "fig_angles_rmse.png"), dpi=200)
    plt.close(fig)

    # ── F_medial decomposition: the definition difference alone, then each path vs O and vs offset-matched O
    rows = [("optical, MVN lumbar + pelvis tilt", "O/V:lumbar+tilt", None),
            ("optical, MVN lumbar", "O/V:lumbar", None), ("optical, MVN pelvis tilt", "O/V:tilt", None),
            ("Xsens MVN angles", "V", "V")]                          # path None: definition only, vs O alone
    rows += [(f"IMU {v}, treadmill contact", f"U:{v}:grf", f"U:{v}") for v in variants]
    rows += [(f"IMU {v0}, gyro contact", f"U:{v0}:gyro", f"U:{v0}")]
    tests_fm = set(fm["test"])
    rows = [r for r in rows if r[1] in tests_fm]
    labs = [lb_ for lb_ in labels if lb_ in set(fm["ckpt"])]
    if rows and labs:
        refs = [("O", COL["ref_O"], "against optical as recorded", None),
                ("Omatch", COL["ref_Omatch"], "against offset-matched optical (lumbar + tilt, leave-one-speed-out)",
                 "////")]
        fig, axs = plt.subplots(1, len(labs), figsize=(4.6 * len(labs) + 1.6, 0.42 * len(rows) + 1.3),
                                sharey=True, sharex=True, constrained_layout=True, squeeze=False)
        h = 0.38
        for ax, lb_ in zip(axs[0], labs):
            f = fm[fm["ckpt"] == lb_]
            for j, (rname, color, label, hatch) in enumerate(refs):
                y, med, lo, hi = [], [], [], []
                for i, (_, test, path) in enumerate(rows):
                    if path is None and rname != "O":
                        continue
                    ref = "O" if rname == "O" else f"Omatch:{path}"
                    x = f[(f["test"] == test) & (f["ref"] == ref)].groupby("subject")["mae"].mean()
                    if len(x):
                        y.append(i if path is None else i + (j - 0.5) * h)
                        med.append(np.median(x))
                        lo.append(np.percentile(x, 25))
                        hi.append(np.percentile(x, 75))
                if not y:
                    continue
                ax.barh(y, med, height=h * 0.92, color=color, hatch=hatch, label=label, edgecolor="white",
                        linewidth=1)
                ax.errorbar(med, y, xerr=[np.subtract(med, lo), np.subtract(hi, med)], fmt="none", ecolor=INK,
                            elinewidth=0.8)
            n_def = sum(r[1].startswith("O/V:") for r in rows)
            if n_def:
                ax.axhline(n_def - 0.5, color=MUTED, lw=0.8, ls=":")     # above: no IMU in the loop
            ax.set_title(f"surrogate {lb_}", fontsize=9, color=INK)
            ax.set_xlabel("F_medial difference, MAE (BW), median over subjects", fontsize=9)
            _style(ax)
        axs[0][0].set_yticks(np.arange(len(rows)), [r[0] for r in rows], fontsize=8)
        axs[0][0].invert_yaxis()
        axs[0][0].legend(fontsize=7, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.08), ncol=1)
        fig.savefig(os.path.join(out_dir, "fig_decomposition.png"), dpi=200)
        plt.close(fig)

    # ── one trial: knee and hip angles, F_medial
    files = sorted(os.listdir(os.path.join(out_dir, "frames"))) if os.path.isdir(os.path.join(out_dir, "frames")) else []
    pick = [f for f in files if f.endswith(f"_{v0}.npz") and (example is None or f.startswith(f"Subj{example[0]}_{example[1]}_"))]
    if not pick:
        return
    z = np.load(os.path.join(out_dir, "frames", pick[0]))
    t_ik, t_u = z["t_ik"], z["t_u"]
    a, b = t_ik[0] + 20.0, t_ik[0] + 24.0
    if b > t_ik[-1]:
        a, b = max(t_ik[0], t_ik[-1] - 4.0), t_ik[-1]
    fig, axs = plt.subplots(3, 1, figsize=(7, 6.2), sharex=True, constrained_layout=True)
    mi, mu = (t_ik >= a) & (t_ik <= b), (t_u >= a) & (t_u <= b)
    for ax, c in zip(axs[:2], ("knee_angle_r", "hip_flexion_r")):
        j = COORDS.index(c)
        ax.plot(t_ik[mi], np.degrees(z["q_o"][mi, j]), color=INK, lw=2, label="optical marker IK")
        ax.plot(t_u[mu], np.degrees(z["q_u"][mu, j]), color=COL["U"], lw=2, label=f"IMU + UKF ({v0})")
        ax.plot(t_ik[mi], np.degrees(z["q_v"][mi, j]), color=COL["V"], lw=1.5, ls="--", label="Xsens MVN angles")
        ax.set_ylabel(c.replace("_", " ") + " (deg)", fontsize=9)
    key = f"fmed|{lab}|"
    tt = z[key + "t"]
    mf = (tt >= a) & (tt <= b)
    axs[2].plot(tt[mf], z[key + "O"][mf], color=INK, lw=2, label="optical, treadmill contact")
    axs[2].plot(tt[mf], z[key + f"U:{v0}:gyro"][mf], color=COL["U"], lw=2, label="IMU, gyro contact")
    if key + f"U:{v0}:grf" in z.files:
        axs[2].plot(tt[mf], z[key + f"U:{v0}:grf"][mf], color=COL["U_grf"], lw=1.5, label="IMU, treadmill contact")
    axs[2].plot(tt[mf], z[key + "V"][mf], color=COL["V"], lw=1.5, ls="--", label="Xsens MVN angles")
    axs[2].set_ylabel("F_medial (BW)", fontsize=9)
    st = z["grf_r"][mi]
    for ax in axs:
        ymin, ymax = ax.get_ylim()
        ax.fill_between(t_ik[mi], ymin, ymax, where=st, color=GRID, alpha=0.6, lw=0, label=None)
        ax.set_ylim(ymin, ymax)
        _style(ax)
    for ax in (axs[0], axs[2]):
        ax.legend(fontsize=7, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=4)
    axs[-1].set_xlabel("time (s); grey = right stance (treadmill)", fontsize=9)
    fig.suptitle(f"{pick[0].replace('.npz', '').replace('_', ' ')}; surrogate {lab}", fontsize=9, color=INK)
    fig.savefig(os.path.join(out_dir, "fig_example.png"), dpi=200)
    plt.close(fig)


# ─── CLI ──────────────────────────────────────────────────────────────────────────────────────────────────────────

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=DATA_DEFAULT, help="Twente 'extracted' folder (SubjXX/SubjXX/...)")
    ap.add_argument("--ckpt", nargs="+", help="surrogate checkpoints (any input set inference_posture_surrogate accepts)")
    ap.add_argument("--labels", nargs="+", default=None, help="short names for --ckpt (default: file stems)")
    ap.add_argument("--subjects", nargs="+", default=SUBJECTS)
    ap.add_argument("--trials", nargs="+", default=list(WALK_SPEED_MPS))
    ap.add_argument("--variants", nargs="+", default=["seg-static", "seg-mvnmodel", "sensor-static"],
                    help="<seg|sensor>-<static|mvnmodel>; the first is the primary one in headline and figures")
    ap.add_argument("--contact", nargs="+", default=list(CONTACT_SOURCES), choices=CONTACT_SOURCES,
                    help="contact sources fed to the surrogate on the same UKF run")
    ap.add_argument("--anchor", default="gyro", choices=["gyro", "grf"], help="contact for the UKF foot anchoring")
    ap.add_argument("--fs", type=float, default=60.0, help="IMU stream rate (PN3: 60 Hz; MVN records 240 Hz)")
    ap.add_argument("--backdate-s", type=float, default=0.2)
    ap.add_argument("--sigma-orient-deg", type=float, default=2.0)
    ap.add_argument("--max-s", type=float, default=None, help="use only the first max_s seconds of each trial")
    ap.add_argument("--mx-convention", choices=sorted(MX_SIGN), default="physical",
                    help="F_medial from the raw fy, mx: physical = -mx medial (default), pipeline = +mx (pre-Nov-2026)")
    ap.add_argument("--no-offset-match", action="store_true",
                    help="skip the second pass (Omatch:<path> rows; needs at least 2 trials per subject)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--example", nargs=2, default=["05", "walk_36"], metavar=("SUBJ", "TRIAL"),
                    help="trial for fig_example (its per-frame series are saved)")
    ap.add_argument("--save-frames", choices=["example", "all"], default="example",
                    help="per-frame npz in <out>/frames (local only: *.npz is git-ignored)")
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--replot", action="store_true", help="only redraw the figures from <out>/<name>")
    a = ap.parse_args(argv)
    out_dir = os.path.join(a.out, a.name)
    if a.replot:
        with open(os.path.join(out_dir, "summary.json")) as f:
            s = json.load(f)["args"]
        make_figures(out_dir, s["variants"], s["labels"], example=a.example)
        print("figures redrawn in", out_dir)
        return
    if not a.ckpt:
        ap.error("--ckpt is required")
    a.ckpt = [os.path.abspath(p) for p in a.ckpt]
    a.labels = a.labels or [os.path.basename(p).replace("best_model_mlp_", "").replace(".pt", "") for p in a.ckpt]
    if len(a.labels) != len(a.ckpt):
        ap.error("--labels needs one name per --ckpt")
    for v in a.variants:
        f, _, c = v.partition("-")
        if f not in FIELDS or c not in ("static", "mvnmodel") or (c == "mvnmodel" and f != "seg"):
            ap.error(f"bad variant {v!r}")
    os.makedirs(out_dir, exist_ok=True)
    shared = dict(data=a.data, ckpt=a.ckpt, labels=a.labels, contact=a.contact, anchor=a.anchor, fs=a.fs,
                  backdate_s=a.backdate_s, sigma_orient_deg=a.sigma_orient_deg, max_s=a.max_s,
                  mx_convention=a.mx_convention)
    jobs = [dict(args=shared, subj=s, trial=t, variant=v, first_variant=(i == 0),
                 save_frames=a.save_frames == "all" or (s, t) == tuple(a.example))
            for s in a.subjects for t in a.trials for i, v in enumerate(a.variants)]
    print(f"{len(jobs)} jobs ({len(a.subjects)} subjects x {len(a.trials)} trials x {len(a.variants)} variants) "
          f"on {a.workers} workers -> {out_dir}", flush=True)
    t0 = time.time()
    results = run_pool(run_job, jobs, a.workers, t0)
    matched = []
    if not a.no_offset_match:
        mjobs = match_jobs(results, shared)
        n_trials = len({(r["match"]["labels"]["subject"], r["match"]["labels"]["trial"]) for r in results if "error" not in r})
        print(f"offset-matched optical reference (leave-one-speed-out): {len(mjobs)} of {n_trials} trials", flush=True)
        matched = run_pool(run_match_job, mjobs, a.workers, t0, tag="match ")
    write_outputs(out_dir, results, a, matched)
    make_figures(out_dir, a.variants, a.labels, example=a.example)
    n_bad = sum("error" in r for r in results + matched)
    print(f"done in {time.time() - t0:.0f} s; {n_bad} failed; outputs in {out_dir}")


if __name__ == "__main__":
    main()

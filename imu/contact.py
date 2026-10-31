"""Foot contact from kinematics.

The contact-flag surrogates were trained with flags from force plates (vertical GRF > 5 % BW). A portable
IMU setup has no force plates, so the flags must come from foot motion: a foot is in contact when one of its
tracked points (heel, toe) is close to the ground and nearly still.

    flags = detect_contact([heel_xyz, toe_xyz], fs=100)          # positions in metres, y up

Without positions (orientation-only suits), use detect_contact_orientation on the foot sensors instead.
Tune either against force-plate flags on recorded trials with imu/evaluate_contact.py.
"""
import numpy as np


def _smooth(x, n):
    if n <= 1:
        return x
    k = np.ones(n) / n
    pad = n // 2
    xp = np.pad(x, ((pad, n - 1 - pad),) + ((0, 0),) * (x.ndim - 1), mode="edge")
    return np.apply_along_axis(lambda c: np.convolve(c, k, mode="valid"), 0, xp)


def _runs(mask):
    """(start, stop) index pairs of True runs."""
    d = np.diff(np.r_[0, mask.astype(int), 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def clean_runs(mask, min_on, min_off):
    """Fill False gaps shorter than min_off frames, then drop True runs shorter than min_on frames."""
    m = mask.copy()
    for a, b in _runs(~m):
        if a > 0 and b < len(m) and b - a < min_off:
            m[a:b] = True
    for a, b in _runs(m):
        if b - a < min_on:
            m[a:b] = False
    return m


def detect_contact(points, fs, up_axis=1, height_margin=0.04, speed_thresh=0.35, ground=None,
                   ground_percentile=5.0, smooth_s=0.03, min_on_s=0.08, min_off_s=0.05):
    """Contact flag per frame for one foot.

    points: list of (n, 3) trajectories of points on the foot (e.g. heel and toe), metres.
    A point is "down" when its height is within height_margin of the ground and its speed is below
    speed_thresh (m/s); the foot is in contact when any of its points is down.
    ground: ground height; default = the ground_percentile of the lowest point's height over the trial.
    """
    pts = [np.asarray(p, dtype=float) for p in points]
    n = len(pts[0])
    if any(len(p) != n for p in pts):
        raise ValueError("all foot points need the same number of frames")
    w = max(1, int(round(smooth_s * fs)))
    heights = [_smooth(p[:, up_axis], w) for p in pts]
    if ground is None:
        ground = np.percentile(np.min(np.column_stack(heights), axis=1), ground_percentile)
    down = np.zeros(n, bool)
    for p, h in zip(pts, heights):
        v = np.gradient(_smooth(p, w), 1.0 / fs, axis=0)
        speed = np.linalg.norm(v, axis=1)
        down |= (h <= ground + height_margin) & (speed <= speed_thresh)
    return clean_runs(down, max(1, int(round(min_on_s * fs))), max(1, int(round(min_off_s * fs))))


def compare(pred, ref, fs, tol_s=0.05):
    """Agreement of predicted contact flags with reference flags (e.g. force-plate based)."""
    pred, ref = np.asarray(pred, bool), np.asarray(ref, bool)
    out = dict(frame_agreement=float(np.mean(pred == ref)),
               recall=float((pred & ref).sum() / max(ref.sum(), 1)),
               precision=float((pred & ref).sum() / max(pred.sum(), 1)))
    for kind, edge in (("onset", 1), ("offset", -1)):
        e_ref = np.flatnonzero(np.diff(ref.astype(int)) == edge)
        e_pred = np.flatnonzero(np.diff(pred.astype(int)) == edge)
        errs = []
        for e in e_ref:
            if len(e_pred):
                j = np.argmin(np.abs(e_pred - e))
                if abs(e_pred[j] - e) <= 0.3 * fs:
                    errs.append((e_pred[j] - e) / fs)
        out[f"{kind}_error_mean_s"] = float(np.mean(errs)) if errs else np.nan
        out[f"{kind}_error_abs_mean_s"] = float(np.mean(np.abs(errs))) if errs else np.nan
        out[f"{kind}_matched_frac"] = float(len(errs) / max(len(e_ref), 1))
        out[f"{kind}_within_tol_frac"] = float(np.mean(np.abs(errs) <= tol_s)) if errs else np.nan
    return out


def angular_speed(rot, fs):
    """Angular speed (rad/s) per frame from a scipy Rotation series sampled at fs."""
    rel = rot[:-1].inv() * rot[1:]
    w = np.linalg.norm(rel.as_rotvec(), axis=1) * fs
    return np.r_[w[0], 0.5 * (w[:-1] + w[1:]), w[-1]]


def detect_contact_orientation(foot_rot, fs, thresh_deg_s=150.0, extend_s=0.12, smooth_s=0.04, min_on_s=0.10,
                               min_off_s=0.06):
    """Contact from a foot IMU's orientation alone, the zero-velocity idea used with foot-mounted IMUs.

    It needs no position, so it works when the pelvis translation is unknown (orientation-only suits).
    Mid-stance is quiet (foot angular speed ~20-80 deg/s) but the foot rotates fast just after heel strike
    (foot flat, ~300 deg/s) and before toe-off (push-off, ~500 deg/s), both inside force-plate stance. So the
    detector finds the quiet runs (speed <= thresh_deg_s) and extends each one back and forward to the
    largest speed within extend_s, which lands on those two peaks.

    Defaults were set on one treadmill trial (Carter P010 t10: frame agreement 0.96 with force-plate flags,
    edge errors ~10 ms); tune them across trials with evaluate_contact.py.
    """
    w = _smooth(np.degrees(angular_speed(foot_rot, fs)), max(1, int(round(smooth_s * fs))))
    still = clean_runs(w <= thresh_deg_s, max(1, int(round(min_on_s * fs))), max(1, int(round(min_off_s * fs))))
    e = int(round(extend_s * fs))
    out = still.copy()
    for a, b in _runs(still) if e else []:
        lo, hi = max(0, a - e), min(len(w), b + e)
        if a > lo:
            out[lo + int(np.argmax(w[lo:a])):a] = True
        if hi > b:
            out[b:b + int(np.argmax(w[b:hi])) + 1] = True
    return out


GRF_CONTACT_BW = 0.05      # same rule as training (train_mlp.CONTACT_GRF_BW_THRESHOLD)


def grf_contact(trial_dir, t, mass_kg):
    """Force-plate contact flags (|vertical GRF| > 5 % BW) for each foot at times t (trial time, s)."""
    import os
    from .synthetic import read_mot
    tg, names, v = read_mot(os.path.join(trial_dir, "grf_data.mot"), angles=False)
    bw = mass_kg * 9.81
    return tuple(np.abs(np.interp(t, tg, v[:, names.index(f"calcn_{s}_force_vy")])) > GRF_CONTACT_BW * bw
                 for s in ("r", "l"))

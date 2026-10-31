"""Virtual IMUs from a mocap trial, to validate the IMU path before real suit data exists.  [needs opensim]

    conda run -n jcf python -m imu.synthetic --trial jcf/full_duration/testing/walking/carter_P010_split0_t10 \
        --out results/imu/synth_P010_t10 --t0 2 --t1 6 --noise-deg 1 --drift-deg-per-s 0.5

For each chosen body, the "IMU" is a sensor fixed to that body: its orientation in ground is the body's
orientation, computed by OpenSim from the trial's IK. The file starts with a calibration segment of --calib-s
seconds in the model's default pose, the standing posture OpenSense assumes for calibration, then the motion.

Outputs in --out:
    orientations.sto   OpenSense orientation file (labels <body>_imu)
    truth_ik.mot       the coordinates behind every frame (radians), on the same time axis
    synthetic.json     settings and the time offset back to the original trial: t_original = t + time_offset

Errors: --noise-deg adds rotation noise (RMS per axis) to each sensor, band-limited with correlation time
--noise-corr-s (AR(1); 0 = independent per frame). Sensor-fusion output errs slowly, not frame to frame, and
white noise would put fake angular speeds of hundreds of deg/s into anything that differentiates orientation.
--drift-deg-per-s adds a per-sensor heading drift about the vertical axis that grows linearly in time, the
dominant IMU error.
"""
import argparse
import json
import os

import numpy as np
from scipy.spatial.transform import Rotation

from .opensense_io import from_scipy, write_orientations
from .segments import LOWER_BODY, body_of


def read_mot(path, angles=True):
    """-> (time (n,), column names, values (n, k)). With angles=True, columns of a file marked inDegrees=yes
    are converted to radians except translations; pass angles=False for force and moment tables."""
    import pandas as pd
    with open(path) as f:
        lines = f.read().splitlines()
    end = next(i for i, l in enumerate(lines) if l.strip() == "endheader")
    in_deg = any(l.strip().lower() == "indegrees=yes" for l in lines[:end])
    df = pd.read_csv(path, sep="\t", skiprows=end + 1)
    names = [c for c in df.columns if c != "time"]
    vals = df[names].to_numpy(float)
    if in_deg and angles:
        rot = [i for i, n in enumerate(names) if not any(n.endswith(s) for s in ("_tx", "_ty", "_tz"))]
        vals[:, rot] = np.radians(vals[:, rot])
    return df["time"].to_numpy(float), names, vals


def write_mot(path, time, names, values, name="coordinates"):
    with open(path, "w") as f:
        f.write(f"{name}\nversion=1\nnRows={len(time)}\nnColumns={len(names) + 1}\ninDegrees=no\nendheader\n")
        f.write("time\t" + "\t".join(names) + "\n")
        for t, row in zip(time, values):
            f.write(f"{t:.6f}\t" + "\t".join(f"{v:.8f}" for v in row) + "\n")


def band_limited_noise(n, dt, noise_deg, corr_s, rng):
    """(n,) Rotation series of small random rotations, RMS noise_deg per axis, AR(1) with correlation time
    corr_s (0 = independent per frame)."""
    e = rng.standard_normal((n, 3))
    if corr_s > 0:                                       # unit stationary variance
        c = np.exp(-dt / corr_s)
        for k in range(1, n):
            e[k] = c * e[k - 1] + np.sqrt(1 - c * c) * e[k]
    return Rotation.from_rotvec(np.radians(noise_deg) * e)


def _matrix(rot):
    return np.array([[rot.get(i, j) for j in range(3)] for i in range(3)])


def set_pose(model, state, coords, names, values):
    """Set coordinate values (radians/metres) and satisfy the model's constraints."""
    for n, v in zip(names, values):
        if n in coords:
            coords[n].setValue(state, float(v), False)
    model.assemble(state)


def body_rotations(model, state, bodies):
    model.realizePosition(state)
    bs = model.getBodySet()
    return {b: _matrix(bs.get(b).getTransformInGround(state).R()) for b in bodies}


def synthesize(trial_dir, out_dir, labels=LOWER_BODY, t0=None, t1=None, calib_s=1.0, fs=None,
               noise_deg=0.0, noise_corr_s=0.1, drift_deg_per_s=0.0, seed=0):
    import opensim as osim
    os.makedirs(out_dir, exist_ok=True)
    time, names, vals = read_mot(os.path.join(trial_dir, "ik_results.mot"))
    keep = np.ones(len(time), bool)
    if t0 is not None:
        keep &= time >= t0
    if t1 is not None:
        keep &= time <= t1
    time, vals = time[keep], vals[keep]
    if fs:                                               # resample to the suit's rate
        t_new = np.arange(time[0], time[-1], 1.0 / fs)
        vals = np.column_stack([np.interp(t_new, time, vals[:, j]) for j in range(vals.shape[1])])
        time = t_new
    dt = float(np.median(np.diff(time)))
    model = osim.Model(os.path.join(trial_dir, "scaled_model.osim"))
    state = model.initSystem()
    cs = model.getCoordinateSet()
    coords = {cs.get(i).getName(): cs.get(i) for i in range(cs.getSize())}
    bodies = [body_of(l) for l in labels]

    default_vals = np.array([coords[n].getDefaultValue() if n in coords else 0.0 for n in names])
    n_cal = int(round(calib_s / dt))
    set_pose(model, state, coords, names, default_vals)
    cal_R = body_rotations(model, state, bodies)
    R_true = {b: [cal_R[b]] * n_cal for b in bodies}
    for row in vals:
        set_pose(model, state, coords, names, row)
        for b, m in body_rotations(model, state, bodies).items():
            R_true[b].append(m)

    out_time = np.arange(n_cal + len(time)) * dt
    time_offset = float(time[0] - n_cal * dt)            # t_original = t + time_offset (motion part)
    rng = np.random.default_rng(seed)
    quats = {}
    for lab, b in zip(labels, bodies):
        R = Rotation.from_matrix(np.array(R_true[b]))
        if drift_deg_per_s:
            rate = drift_deg_per_s * rng.choice([-1.0, 1.0])
            R = Rotation.from_euler("y", rate * out_time, degrees=True) * R      # heading drift in ground
        if noise_deg:
            R = R * band_limited_noise(len(out_time), dt, noise_deg, noise_corr_s, rng)
        quats[lab] = from_scipy(R)
    write_orientations(os.path.join(out_dir, "orientations.sto"), out_time, quats, data_rate=1.0 / dt)
    write_mot(os.path.join(out_dir, "truth_ik.mot"), out_time, names,
              np.vstack([np.tile(default_vals, (n_cal, 1)), vals]), name="truth_ik")
    meta = dict(trial_dir=os.path.abspath(trial_dir), labels=list(labels), calib_s=n_cal * dt, n_calib_frames=n_cal,
                fs=1.0 / dt, time_offset=time_offset, t0=float(time[0]), t1=float(time[-1]),
                noise_deg=noise_deg, noise_corr_s=noise_corr_s, drift_deg_per_s=drift_deg_per_s, seed=seed)
    with open(os.path.join(out_dir, "synthetic.json"), "w") as f:
        json.dump(meta, f, indent=1)
    return meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trial", required=True, help="trial folder with ik_results.mot and scaled_model.osim")
    ap.add_argument("--out", required=True)
    ap.add_argument("--t0", type=float, default=None)
    ap.add_argument("--t1", type=float, default=None)
    ap.add_argument("--calib-s", type=float, default=1.0)
    ap.add_argument("--fs", type=float, default=None, help="resample to this rate (Hz)")
    ap.add_argument("--noise-deg", type=float, default=0.0)
    ap.add_argument("--noise-corr-s", type=float, default=0.1)
    ap.add_argument("--drift-deg-per-s", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    print(json.dumps(synthesize(a.trial, a.out, t0=a.t0, t1=a.t1, calib_s=a.calib_s, fs=a.fs, noise_deg=a.noise_deg,
                                noise_corr_s=a.noise_corr_s, drift_deg_per_s=a.drift_deg_per_s, seed=a.seed), indent=1))


if __name__ == "__main__":
    main()

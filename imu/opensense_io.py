"""OpenSense orientation files: a .sto table whose cells are quaternions "w,x,y,z".

    DataRate=100.000000
    DataType=Quaternion
    version=3
    OpenSimVersion=4.5
    endheader
    time    pelvis_imu    femur_r_imu ...
    0       1,0,0,0       0.99,0.01,-0.02,0.03 ...

Quaternions here are scalar-first (w, x, y, z), as OpenSim writes them. scipy's Rotation uses
scalar-last; use to_scipy / from_scipy to convert.
"""
import numpy as np
from scipy.spatial.transform import Rotation


def from_scipy(rot):
    """scipy Rotation (n) -> (n, 4) array w, x, y, z with w >= 0."""
    q = np.atleast_2d(rot.as_quat())[:, [3, 0, 1, 2]]
    q[q[:, 0] < 0] *= -1
    return q


def to_scipy(q_wxyz):
    q = np.atleast_2d(np.asarray(q_wxyz, dtype=float))
    return Rotation.from_quat(q[:, [1, 2, 3, 0]])


def write_orientations(path, time, quats, data_rate=None, opensim_version="4.5"):
    """time: (n,), quats: dict label -> (n, 4) w,x,y,z. Labels must end in '_imu' for OpenSense."""
    time = np.asarray(time, dtype=float)
    labels = list(quats)
    for lab in labels:
        if quats[lab].shape != (len(time), 4):
            raise ValueError(f"{lab}: expected ({len(time)}, 4), got {quats[lab].shape}")
    if data_rate is None:
        data_rate = 1.0 / np.median(np.diff(time)) if len(time) > 1 else 100.0
    with open(path, "w") as f:
        f.write(f"DataRate={data_rate:.6f}\nDataType=Quaternion\nversion=3\nOpenSimVersion={opensim_version}\nendheader\n")
        f.write("time\t" + "\t".join(labels) + "\n")
        for k, t in enumerate(time):
            cells = [",".join(f"{v:.9f}" for v in quats[lab][k]) for lab in labels]
            f.write(f"{t:.6f}\t" + "\t".join(cells) + "\n")


def read_orientations(path):
    """-> (time (n,), dict label -> (n, 4) w,x,y,z, data_rate)."""
    with open(path) as f:
        lines = f.read().splitlines()
    end = next(i for i, l in enumerate(lines) if l.strip() == "endheader")
    rate = next((float(l.split("=")[1]) for l in lines[:end] if l.startswith("DataRate=")), None)
    header = lines[end + 1].split("\t")
    labels = header[1:]
    rows = [l.split("\t") for l in lines[end + 2:] if l.strip()]
    time = np.array([float(r[0]) for r in rows])
    quats = {lab: np.array([[float(v) for v in r[j + 1].split(",")] for r in rows]) for j, lab in enumerate(labels)}
    return time, quats, rate


def bvh_to_orientations(bvh, mapping, scale=1.0):
    """Global segment orientations from a BVH for each OpenSense label. mapping: label -> BVH joint."""
    from .bvh import forward_kinematics
    fk = forward_kinematics(bvh, scale=scale, joints=list(mapping.values()))
    time = np.arange(bvh.n_frames) * bvh.frame_time
    return time, {lab: from_scipy(fk[j][1]) for lab, j in mapping.items()}

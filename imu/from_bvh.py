"""Axis Studio BVH export -> OpenSense orientation file.

    python -m imu.from_bvh --bvh take01.bvh --out results/imu/take01/orientations.sto [--start 2.0] [--end 30]

Each OpenSense label gets the global orientation of its mapped suit segment (segments.PERCEPTION_NEURON).
OpenSense calibrates on the FIRST row, assuming the subject stands in the model's default pose (upright,
feet straight ahead, knees straight). Record a few seconds of quiet standing at the start of every take and
pass --start at a time inside it, or trim the take in Axis Studio.

The segment frames of the BVH skeleton do not need to match OpenSim's: calibration measures each sensor's
offset on its body in that first frame. Only the vertical must agree (both y-up), and the forward direction
of the base sensor is chosen in opensense_ik.py with --heading-axis (check it on the first real take).
"""
import argparse
import os

import numpy as np

from . import bvh, segments
from .opensense_io import bvh_to_orientations, write_orientations


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bvh", required=True)
    ap.add_argument("--out", required=True, help="orientation .sto to write")
    ap.add_argument("--start", type=float, default=0.0, help="time (s) of the calibration frame, in quiet standing")
    ap.add_argument("--end", type=float, default=None)
    ap.add_argument("--labels", nargs="+", default=segments.LOWER_BODY)
    a = ap.parse_args(argv)
    b = bvh.read(a.bvh)
    mapping = segments.resolve(b.order, a.labels)
    t, q = bvh_to_orientations(b, mapping)
    keep = (t >= a.start - 1e-9) & ((t <= a.end + 1e-9) if a.end is not None else True)
    t, q = t[keep] - t[keep][0], {k: v[keep] for k, v in q.items()}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    write_orientations(a.out, t, q, data_rate=b.fs)
    print(f"{len(t)} frames at {b.fs:g} Hz ({t[-1]:.2f} s) from {a.start:g} s; segments: "
          + ", ".join(f"{k}<-{v}" for k, v in mapping.items()))
    print("written to", a.out)


if __name__ == "__main__":
    main()

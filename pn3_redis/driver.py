"""Perception Neuron 3 -> Redis driver for OpenSai.

    # live, from Axis Studio (BVH broadcasting over UDP to this machine)
    python -m pn3_redis.driver --source mocapapi --udp-port 7012 --sensor-frame axis --heading-axis z \
        --height 1.75 --mass 70 --redis redis://localhost:6379/0

    # replay a suit recording (Axis Studio BVH export)
    python -m pn3_redis.driver --source bvh take01.bvh --sensor-frame axis --heading-axis z --realtime --redis ...

    # replay virtual-suit data made from a mocap trial (imu/synthetic.py), no Redis needed
    python -m pn3_redis.driver --source sto results/imu/synth/orientations.sto --resample-hz 60 --log run.csv

Pipeline per frame: calibration (first --calib-s seconds standing still, upright, feet straight) -> foot contact
(the source's own flags if it has them, else the online gyro detector) -> UKF over 19 joint coordinates with
contact-anchored pelvis position -> optional knee-load surrogate -> Redis / CSV.
See docs/PN3_REDIS_DRIVER.md for the Redis keys.
"""
import argparse
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):   # small matrices: threads only add latency
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from pn3_redis.estimator import CausalContact, PN3Estimator, quat_to_matrix   # noqa: E402
from pn3_redis.publisher import CSVLogger, MemorySink, RedisPublisher          # noqa: E402

HEADING = {"x": (1, 0, 0), "-x": (-1, 0, 0), "y": (0, 1, 0), "-y": (0, -1, 0), "z": (0, 0, 1), "-z": (0, 0, -1)}


def make_source(a):
    from pn3_redis import sources
    if a.source == "mocapapi":
        return sources.MocapApiSource(udp_port=a.udp_port)
    if a.source == "bvh":
        return sources.BVHSource(a.path, realtime=a.realtime, speed=a.speed)
    if a.source == "sto":
        return sources.StoSource(a.path, realtime=a.realtime, speed=a.speed, resample_hz=a.resample_hz,
                                 contact_trial=a.contact_trial, loop=a.loop)
    raise SystemExit(f"unknown source {a.source}")


def run(a, sinks):
    est = PN3Estimator(height_m=a.height, sensor_frame=a.sensor_frame, heading_axis=HEADING[a.heading_axis],
                       sigma_orient_deg=a.sigma_orient_deg, accel_rot=a.accel_rot, calib_s=a.calib_s)
    feet = {"r": "calcn_r_imu", "l": "calcn_l_imu"}
    # two online detectors from the foot sensors: "planted" (foot flat, for anchoring the pelvis) and "stance"
    # (closer to force-plate stance, published and fed to the surrogate, which was trained on force-plate flags).
    # The stance detector also back-dates each heel strike, which the surrogate applies to its buffered window.
    planted = CausalContact(on_deg_s=100, off_deg_s=200)
    stance = CausalContact(on_deg_s=150, off_deg_s=400, backdate_s=a.backdate_s)
    sur = None
    if a.surrogate:
        from pn3_redis.surrogate import LiveSurrogate
        sur = LiveSurrogate(a.surrogate, a.mass, a.height)
    n, last_print, t_prev, rate = 0, time.monotonic(), None, 0.0
    for f in make_source(a):
        if not est.calibrated:
            if est.add_calibration_frame(f.t, f.quats):
                for det in (planted, stance):
                    det.reset(f.t, {s: quat_to_matrix(f.quats[k]) for s, k in feet.items()})
                print(f"calibrated at t = {f.t:.2f} s (heading {np.degrees(np.arctan2(est.A[0, 2], est.A[0, 0])):.1f} deg)")
            t_prev = f.t
            continue
        dt = f.t - t_prev
        if dt <= 0:
            continue
        rots = {s: quat_to_matrix(f.quats[k]) for s, k in feet.items()}
        g_planted, g_stance = planted.update(f.t, rots, dt), stance.update(f.t, rots, dt)
        onsets = dict(stance.onsets)
        if a.contact == "source" or (a.contact == "auto" and f.contact):
            g_planted = g_stance = f.contact
            onsets = {}
        s = est.step(f.t, f.quats, g_planted)
        s["contact"] = dict(g_stance)
        s["contact_onset"] = onsets
        s["quats"] = f.quats if a.publish_sensors else None
        if sur is not None:
            out = sur.update(s)
            if out:
                s.update(out)
        status = dict(calibrated=True, frame=n, rate_hz=round(rate, 1))
        for k in sinks:
            k.publish(s, status)
        n += 1
        rate = 0.9 * rate + 0.1 / dt if rate else 1.0 / dt
        t_prev = f.t
        if not a.quiet and time.monotonic() - last_print > 1.0:
            last_print = time.monotonic()
            q = dict(zip(est.model.index, np.degrees(s["q"])))
            fm = f"  F_medial {s['f_medial']:.2f} BW" if s.get("f_medial") is not None else ""
            print(f"t {f.t:7.2f}  {rate:5.1f} Hz  knee R/L {q['knee_angle_r']:5.1f}/{q['knee_angle_l']:5.1f} deg  "
                  f"hip R {q['hip_flexion_r']:5.1f}  contact R/L {int(s['contact']['r'])}/{int(s['contact']['l'])}"
                  f"  pelvis x {s['pelvis'][0]:6.2f} m{fm}", flush=True)
    return n


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["mocapapi", "bvh", "sto"], required=True)
    ap.add_argument("path", nargs="?", help="BVH or orientation .sto file for replay sources")
    ap.add_argument("--udp-port", type=int, default=7012)
    ap.add_argument("--realtime", action="store_true", help="replay at the recorded rate")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--loop", action="store_true", help="replay an .sto file in a loop (demos)")
    ap.add_argument("--resample-hz", type=float, default=None, help=".sto replay: resample (PN3 runs at 60 Hz)")
    ap.add_argument("--contact-trial", default=None, help=".sto replay of a synthetic file: trial folder for force-plate contact")
    ap.add_argument("--contact", choices=["auto", "source", "gyro"], default="auto",
                    help="auto = the source's flags when present (Axis grounding state, force plates), else gyro detector")
    ap.add_argument("--backdate-s", type=float, default=0.2,
                    help="gyro contact: look back this far to date each heel strike for the surrogate's window (0 = off)")
    ap.add_argument("--sensor-frame", choices=["opensim", "axis"], default="opensim",
                    help="world frame of the sensor orientations: opensim (synthetic files) or axis (Axis Studio)")
    ap.add_argument("--heading-axis", choices=list(HEADING), default="x",
                    help="pelvis sensor axis that points forward during calibration (x for synthetic, z for Axis bones)")
    ap.add_argument("--calib-s", type=float, default=1.0)
    ap.add_argument("--height", type=float, default=1.70)
    ap.add_argument("--mass", type=float, default=70.0)
    ap.add_argument("--sigma-orient-deg", type=float, default=2.0)
    ap.add_argument("--accel-rot", type=float, default=300.0)
    ap.add_argument("--surrogate", default=None, help="knee-load surrogate checkpoint (publishes f_medial)")
    ap.add_argument("--redis", default=None, help="redis://host:port/db")
    ap.add_argument("--namespace", default="sai::sensors::pn3")
    ap.add_argument("--stream", action="store_true",
                    help="also write a per-frame Redis stream (::stream) for pn3_redis.monitor; breaks the sai-interfaces web UI")
    ap.add_argument("--viz-wrap-m", type=float, default=None,
                    help="wrap pelvis_tx in ::joint_positions into [-L/2, L/2) so OpenSai's camera keeps the body in view")
    ap.add_argument("--publish-sensors", action="store_true", help="also publish the raw sensor quaternions")
    ap.add_argument("--log", default=None, help="CSV log of every published frame")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    if a.source in ("bvh", "sto") and not a.path:
        ap.error(f"--source {a.source} needs a file path")
    sinks = []
    if a.redis:
        from pn3_redis.model import GENERIC_HEIGHT_M
        sinks.append(RedisPublisher(a.redis, a.namespace, stream_maxlen=2000 if a.stream else 0,
                                    scale=a.height / GENERIC_HEIGHT_M, wrap_m=a.viz_wrap_m))
    if a.log:
        sinks.append(CSVLogger(a.log))
    if not sinks:
        sinks.append(MemorySink())
        print("no --redis or --log given: running without output (dry run)")
    try:
        n = run(a, sinks)
        print(f"done: {n} frames published")
    except KeyboardInterrupt:
        print("stopped")
    finally:
        for s in sinks:
            s.close()


if __name__ == "__main__":
    main()

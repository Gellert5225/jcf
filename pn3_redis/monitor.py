"""Live view of what the driver publishes, read back from Redis only (as an OpenSai app would).

    python -m pn3_redis.monitor --redis redis://localhost:6379/0 [--window-s 6] [--save snapshot.png --duration 10]

Panels: hip and knee flexion (both legs), foot contact, pelvis path (top view), and the knee-load surrogate's medial
force with the 1.4 BW limit when the driver runs with --surrogate. Reads every frame from ::stream when the driver
runs with --stream, otherwise samples the latest-value keys (coarser, but compatible with the sai-interfaces web UI).
"""
import argparse
import json
import time
from collections import deque

import numpy as np


def read_vec(r, key):
    v = r.get(key)
    if v is None:
        return None
    s = v.decode()
    return np.array(json.loads(s), float) if s.startswith("[") else np.array([float(x) for x in s.split()])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--redis", default="redis://localhost:6379/0")
    ap.add_argument("--namespace", default="sai::sensors::pn3")
    ap.add_argument("--window-s", type=float, default=6.0)
    ap.add_argument("--hz", type=float, default=30.0, help="polling rate")
    ap.add_argument("--save", default=None, help="headless: save a PNG after --duration seconds and exit")
    ap.add_argument("--duration", type=float, default=10.0)
    a = ap.parse_args(argv)
    import redis
    import matplotlib
    if a.save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    r = redis.Redis.from_url(a.redis)
    ns = a.namespace
    names = json.loads(r.get(f"{ns}::joint_names") or "[]")
    if not names:
        raise SystemExit(f"no {ns}::joint_names in Redis: is the driver running?")
    ix = {n: i for i, n in enumerate(names)}
    buf = deque()
    fig, ax = plt.subplots(2, 2, figsize=(11, 6.5))
    fig.canvas.manager.set_window_title("PN3 driver monitor") if hasattr(fig.canvas, "manager") and fig.canvas.manager else None
    start, last_id, last_t = time.monotonic(), "0-0", None
    use_stream = r.type(f"{ns}::stream") == b"stream"        # driver run with --stream: every frame; else poll keys
    while True:
        if use_stream:
            res = r.xread({f"{ns}::stream": last_id}, count=5000, block=int(1000 / a.hz))
            for _key, entries in res or []:
                for eid, e in entries:
                    last_id = eid
                    fm = e.get(b"f_medial")
                    buf.append((float(e[b"t"]), np.array(json.loads(e[b"q"])), np.array(json.loads(e[b"pelvis_position"])),
                                np.array(json.loads(e[b"contact"])), None if fm is None else float(fm)))
        else:
            t = read_vec(r, f"{ns}::timestamp")
            if t is not None and t[0] != last_t:
                last_t = t[0]
                fm = read_vec(r, f"{ns}::f_medial")
                buf.append((t[0], read_vec(r, f"{ns}::q"), read_vec(r, f"{ns}::pelvis_position"),
                            read_vec(r, f"{ns}::contact"), None if fm is None else fm[0]))
            time.sleep(0.5 / a.hz)
        while buf and buf[0][0] < buf[-1][0] - a.window_s:
            buf.popleft()
        if len(buf) > 2:
            T = np.array([b[0] for b in buf]); Q = np.degrees(np.array([b[1] for b in buf]))
            P = np.array([b[2] for b in buf]); Cc = np.array([b[3] for b in buf]); F = [b[4] for b in buf]
            for x in ax.flat:
                x.cla()
            for side, col in (("r", "#D55E00"), ("l", "#0072B2")):
                ax[0, 0].plot(T, Q[:, ix[f"knee_angle_{side}"]], color=col, lw=2, label=f"knee {side.upper()}")
                ax[0, 0].plot(T, Q[:, ix[f"hip_flexion_{side}"]], color=col, lw=1.2, ls="--", label=f"hip {side.upper()}")
            ax[0, 0].set_ylabel("deg"); ax[0, 0].set_title("joint angles (UKF)"); ax[0, 0].legend(fontsize=8, ncol=2, loc="upper left")
            ax[1, 0].fill_between(T, 1.05, 1.95, where=Cc[:, 1] > 0, color="#0072B2", step="mid", lw=0, label="left")   # left on top
            ax[1, 0].fill_between(T, 0.05, 0.95, where=Cc[:, 0] > 0, color="#D55E00", step="mid", lw=0, label="right")
            ax[1, 0].set_yticks([0.5, 1.5], ["right", "left"]); ax[1, 0].set_title("foot contact"); ax[1, 0].set_xlabel("time (s)")
            ax[0, 1].plot(P[:, 0], P[:, 2], color="#444", lw=1.5); ax[0, 1].plot(P[-1, 0], P[-1, 2], "o", color="#E69F00")
            ax[0, 1].set_xlabel("forward x (m)"); ax[0, 1].set_ylabel("right z (m)"); ax[0, 1].set_title("pelvis path, top view")
            ax[0, 1].set_aspect("equal", adjustable="datalim")
            if any(f is not None for f in F):
                Fv = np.array([np.nan if f is None else f for f in F])
                ax[1, 1].plot(T, Fv, color="#E69F00", lw=2)
                ax[1, 1].axhline(1.4, color="#888", ls="--", lw=1)
                ax[1, 1].set_title(f"medial knee force (surrogate): {Fv[np.isfinite(Fv)][-1]:.2f} BW")
                ax[1, 1].set_ylabel("BW"); ax[1, 1].set_xlabel("time (s)")
            else:
                ax[1, 1].text(0.5, 0.5, "run the driver with --surrogate\nto see the medial knee force", ha="center", va="center")
                ax[1, 1].set_axis_off()
            st = r.get(f"{ns}::status")
            fig.suptitle(f"PN3 -> Redis  {json.loads(st)['rate_hz'] if st else '?'} Hz", fontsize=10)
            fig.tight_layout()
        if a.save:
            if time.monotonic() - start > a.duration:
                fig.savefig(a.save, dpi=130)
                print("saved", a.save)
                return
        else:
            plt.pause(0.001)


if __name__ == "__main__":
    main()

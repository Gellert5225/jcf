"""Outputs: Redis keys for OpenSai, plus CSV and in-memory sinks.

Redis keys, under a namespace (default sai::sensors::pn3):
    ::timestamp                 seconds (source clock)
    ::joint_names               JSON list, the 19 coordinates of ::q (written once)
    ::q, ::dq                   joint positions/velocities, OpenSim convention (radians, metres)
    ::q_std                     filter standard deviation of each coordinate
    ::q_urdf                    37-vector in Rajagopal2015.urdf joint order; pelvis translation in the robot frame
                                (x forward, y left, z up), rotations copied, as infer_trajectory_csv.py and
                                pipeline/label_trajectory_csv.py assume
    ::joint_positions           the 37-vector OpenSai's SimViz reads for a robot named "pn3" (prefix "sai"), for the
    ::joint_velocities          URDF exactly as OpenSai interprets it: sai-model's joint order (= URDF file order);
                                pelvis rotations re-expressed for the URDF's literal pelvis axes (unlike ::q_urdf);
                                pelvis translations divided by the height scale so the unscaled URDF keeps its feet
                                on the floor; optionally pelvis_tx wrapped into [-L/2, L/2) to stay in camera view
    ::pelvis_position           OpenSim ground frame (x forward, y up, z right)
    ::com_position              whole-body COM, same frame
    ::contact                   [right, left] 0/1
    ::sensor::<label>::orientation    raw sensor quaternion w, x, y, z
    ::f_medial, ::fy, ::mx, ::surrogate_time    knee-load surrogate output, when enabled
    ::status                    JSON {calibrated, frame, rate_hz}
    ::stream                    optional (stream_maxlen > 0): Redis stream, one entry per frame (t, q, pelvis_position,
                                contact, f_medial), for consumers that need every frame. Off by default: a non-string
                                key breaks the sai-interfaces web UI's key cache for the whole Redis database.
Vectors are JSON arrays ("[1.0, 2.0]"), the representation sai-common's RedisClient decodes (it splits on commas;
a space-separated vector would decode as its first element only). Non-finite values are never published.
"""
import csv
import json
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from .model import COORDS, URDF, C

URDF_Q_NAMES = [j.get("name") for j in ET.parse(URDF).getroot().findall("joint") if j.get("type") != "fixed"]


def to_robot(v):
    """OpenSim ground (x fwd, y up, z right) -> robot frame (x fwd, y left, z up)."""
    v = np.asarray(v, float)
    return np.array([v[0], -v[2], v[1]])


def q_urdf(q):
    out = np.zeros(len(URDF_Q_NAMES))
    for i, n in enumerate(URDF_Q_NAMES):
        if n in COORDS:
            out[i] = q[COORDS.index(n)]
    t = to_robot([q[COORDS.index(c)] for c in ("pelvis_tx", "pelvis_ty", "pelvis_tz")])
    for c, v in zip(("pelvis_tx", "pelvis_ty", "pelvis_tz"), t):
        out[URDF_Q_NAMES.index(c)] = v
    return out


def urdf_pelvis_angles(q):
    """Pelvis angles under Rajagopal2015.urdf's literal joint axes (tilt about z, list about x, rotation about y of
    its z-up frame) that reproduce the OpenSim pelvis orientation Rz(tilt) Rx(list) Ry(rotation) of q. The URDF
    links are OpenSim frames relabelled by C, so the pelvis link must have orientation C R C^T."""
    i = [COORDS.index(c) for c in ("pelvis_tilt", "pelvis_list", "pelvis_rotation")]
    R = Rotation.from_euler("ZXY", q[i]).as_matrix()                  # intrinsic z, x, y = OpenSim ground_pelvis
    return Rotation.from_matrix(C @ R @ C.T).as_euler("ZXY")           # same sequence about the URDF's axes


def sai_joint_vectors(q, dq, scale=1.0, wrap_m=None, eps=1e-4):
    """OpenSai joint_positions / joint_velocities for the URDF as OpenSai loads it: q_urdf order, pelvis rotations
    re-expressed for the URDF's literal pelvis axes, translations in the unscaled URDF's units."""
    qp, qv = q_urdf(q), q_urdf(dq)
    ir = [URDF_Q_NAMES.index(c) for c in ("pelvis_tilt", "pelvis_list", "pelvis_rotation")]
    a0 = urdf_pelvis_angles(q)
    qp[ir] = a0
    d = urdf_pelvis_angles(q + eps * dq) - a0                          # rates by a small finite difference,
    qv[ir] = ((d + np.pi) % (2 * np.pi) - np.pi) / eps                 # wrapped (Euler angles jump at +/-pi)
    it = [URDF_Q_NAMES.index(c) for c in ("pelvis_tx", "pelvis_ty", "pelvis_tz")]
    qp[it] /= scale
    qv[it] /= scale
    if wrap_m:
        qp[it[0]] = (qp[it[0]] + wrap_m / 2) % wrap_m - wrap_m / 2
    assert qp.size == qv.size == 37, "OpenSai copies vectors into fixed 37-element buffers without a size check"
    return qp, qv


def flatten(state, scale=1.0, wrap_m=None):
    """Estimator state (+ optional surrogate output) -> flat dict of named values for the sinks."""
    q = np.asarray(state["q"])
    jp, jv = sai_joint_vectors(q, np.asarray(state["dq"]), scale, wrap_m)
    out = {"timestamp": float(state["t"]), "q": q, "dq": np.asarray(state["dq"]), "q_std": np.asarray(state["q_std"]),
           "q_urdf": q_urdf(q), "joint_positions": jp, "joint_velocities": jv,
           "pelvis_position": np.asarray(state["pelvis"]), "com_position": np.asarray(state["com"]),
           "contact": np.array([int(state["contact"]["r"]), int(state["contact"]["l"])])}
    for k in ("f_medial", "fy", "mx", "surrogate_time"):
        if state.get(k) is not None:
            out[k] = float(state[k])
    for lab, qq in (state.get("quats") or {}).items():
        out[f"sensor::{lab}::orientation"] = np.asarray(qq, float)
    return out


class RedisPublisher:
    def __init__(self, url="redis://localhost:6379/0", namespace="sai::sensors::pn3", stream_maxlen=0, scale=1.0,
                 wrap_m=None):
        try:
            import redis
        except ImportError as e:
            raise SystemExit("redis-py not installed: pip install redis") from e
        self.r = redis.Redis.from_url(url)
        try:
            self.r.ping()
        except redis.exceptions.ConnectionError as e:
            raise SystemExit(f"No Redis server at {url} ({e}).\nStart one first, in another terminal: redis-server "
                             "(install: sudo apt install redis-server, or conda install -c conda-forge redis-server)") from None
        self.ns, self.maxlen, self.scale, self.wrap_m = namespace, stream_maxlen, scale, wrap_m
        if not self.maxlen:
            self.r.delete(f"{self.ns}::stream")          # a leftover stream would keep breaking the web UI
        self.r.set(f"{self.ns}::joint_names", json.dumps(COORDS))
        self.r.set(f"{self.ns}::urdf_joint_names", json.dumps(URDF_Q_NAMES))

    def _enc(self, v):
        if np.ndim(v) == 0:
            return repr(float(v))
        return json.dumps([float(x) for x in np.asarray(v, float).ravel()])

    def publish(self, state, status=None):
        p = self.r.pipeline(transaction=False)
        fl = flatten(state, self.scale, self.wrap_m)
        for k, v in fl.items():
            if np.all(np.isfinite(v)):                     # sai-common's stod accepts "nan", which would reach the render
                p.set(f"{self.ns}::{k}", self._enc(v))
        if status is not None:
            p.set(f"{self.ns}::status", json.dumps(status))
        if self.maxlen:
            entry = {"t": fl["timestamp"], "q": json.dumps(np.round(fl["q"], 6).tolist()),
                     "pelvis_position": json.dumps(np.round(fl["pelvis_position"], 5).tolist()),
                     "contact": json.dumps(fl["contact"].tolist())}
            if "f_medial" in fl and np.isfinite(fl["f_medial"]):
                entry["f_medial"] = fl["f_medial"]
            p.xadd(f"{self.ns}::stream", entry, maxlen=self.maxlen, approximate=True)
        p.execute()

    def close(self):
        self.r.close()


class CSVLogger:
    """One row per frame: time, q (19), dq (19), pelvis and COM position, contact, surrogate output."""

    def __init__(self, path):
        self.f = open(path, "w", newline="")
        self.w = csv.writer(self.f)
        self.w.writerow(["t"] + COORDS + [f"d_{c}" for c in COORDS] + ["pelvis_x", "pelvis_y", "pelvis_z",
                        "com_x", "com_y", "com_z", "contact_r", "contact_l", "f_medial", "fy", "mx", "surrogate_time",
                        "surrogate_contact_r", "surrogate_contact_l"])

    def publish(self, state, status=None):
        g = lambda k: "" if state.get(k) is None else f"{state[k]:.6f}"
        used = state.get("contact_used")
        self.w.writerow([f"{state['t']:.6f}"] + [f"{v:.6f}" for v in state["q"]] + [f"{v:.6f}" for v in state["dq"]]
                        + [f"{v:.6f}" for v in state["pelvis"]] + [f"{v:.6f}" for v in state["com"]]
                        + [int(state["contact"]["r"]), int(state["contact"]["l"])]
                        + [g("f_medial"), g("fy"), g("mx"), g("surrogate_time")]
                        + ([int(used["r"]), int(used["l"])] if used else ["", ""]))

    def close(self):
        self.f.close()


class MemorySink:
    def __init__(self):
        self.rows = []

    def publish(self, state, status=None):
        self.rows.append(flatten(state))

    def close(self):
        pass

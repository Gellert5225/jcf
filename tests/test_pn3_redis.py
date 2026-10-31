"""Tests for the PN3 -> Redis driver (numpy/scipy only; Redis and MocapApi are faked). Run: python -m pytest tests -q"""
import json
import os
import sys
import types

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from imu.opensense_io import from_scipy                                     # noqa: E402
from pn3_redis import publisher                                              # noqa: E402
from pn3_redis.estimator import DEFAULT_SENSORS, CausalContact, PN3Estimator  # noqa: E402
from pn3_redis.model import COORDS, LowerBody                                # noqa: E402


def test_model_default_pose_and_symmetry():
    lb = LowerBody()
    q0 = lb.default_q()
    F = lb.fk(q0)
    assert abs(min(F["calcn_r"][1][0, 1], F["calcn_l"][1][0, 1])) < 1e-9          # heels on the floor
    np.testing.assert_allclose(F["calcn_r"][1][0, [0, 1]], F["calcn_l"][1][0, [0, 1]], atol=1e-3)
    assert F["calcn_r"][1][0, 2] > 0 > F["calcn_l"][1][0, 2]                      # right foot at +z (OpenSim)
    for b in ("pelvis", "femur_r", "torso"):
        np.testing.assert_allclose(F[b][0][0], np.eye(3), atol=1e-9)               # default pose: aligned frames
    q = q0.copy(); q[COORDS.index("knee_angle_r")] = np.radians(60)
    shank = lb.fk(q)["tibia_r"][0][0]
    ang = np.degrees(np.linalg.norm(Rotation.from_matrix(shank).as_rotvec()))
    assert 55 < ang < 70                                                          # walker knee: ~60 deg + coupling


def _quats(lb, q, A=np.eye(3)):
    F = lb.fk(q)
    return [{k: from_scipy(Rotation.from_matrix(A @ F[b][0][i]))[0] for k, b in DEFAULT_SENSORS.items()} for i in range(len(q))]


def test_ukf_tracks_joint_angles_and_holds_base_when_standing():
    est = PN3Estimator(calib_s=0.5)
    lb = est.model
    fs, n = 100, 300
    t = np.arange(n) / fs
    q = np.tile(lb.default_q(), (n, 1))
    walk = t >= 0.6
    for c, amp in (("hip_flexion_r", 25), ("knee_angle_r", 40), ("ankle_angle_r", 10), ("lumbar_extension", 5)):
        q[walk, COORDS.index(c)] += np.radians(amp) * np.sin(2 * np.pi * 1.0 * (t[walk] - 0.6)) ** 2
    A = Rotation.from_euler("y", 30, degrees=True).as_matrix()                    # sensor world turned 30 deg
    frames = _quats(lb, q, A)
    est.heading_axis = np.array([1.0, 0, 0])
    errs = []
    for k in range(n):
        if not est.calibrated:
            est.add_calibration_frame(t[k], frames[k])
            continue
        s = est.step(t[k], frames[k], {"r": False, "l": True})                  # left foot planted
        errs.append(s["q"][:16] - q[k, :16])
    e = np.degrees(np.sqrt(np.mean(np.array(errs)[50:] ** 2, axis=0)))
    rot_idx = [i for i, c in enumerate(COORDS[:16]) if not c.startswith("pelvis_t")]
    assert e[rot_idx].max() < 1.5, dict(zip(np.array(COORDS)[rot_idx], e[rot_idx].round(2)))
    assert abs(s["pelvis"][1] - lb.default_q()[COORDS.index("pelvis_ty")]) < 0.02   # left leg static: pelvis stays put


def test_causal_contact_hysteresis():
    fs, cc = 100, CausalContact(on_deg_s=100, off_deg_s=200, smooth_s=0.0, min_dwell_s=0.0)
    ang = np.r_[np.zeros(30), np.cumsum(np.full(30, np.radians(5))), np.full(30, 30 * np.radians(5))]   # still, 500 deg/s, still
    rots = Rotation.from_euler("z", ang[:, None]).as_matrix()
    cc.reset(0.0, {"r": rots[0]})
    states = [cc.update(k / fs, {"r": rots[k]}, 1 / fs)["r"] for k in range(1, len(ang))]
    assert states[20] and not states[45] and states[-1]


def test_causal_contact_backdates_heel_strike():
    # stance, swing at 500 deg/s, speed dip at initial contact (frames 60-61), foot slap (62-64), quiet stance
    fs = 100
    speed = np.r_[np.zeros(30), np.full(30, 500.0), [100.0, 100.0], np.full(3, 250.0), np.full(30, 20.0)]
    rots = Rotation.from_euler("z", np.cumsum(np.radians(speed) / fs)[:, None]).as_matrix()
    cc = CausalContact(on_deg_s=150, off_deg_s=400, backdate_s=0.2)
    cc.reset(0.0, {"r": rots[0]})
    onsets = []
    for k in range(1, len(speed)):
        state = cc.update(k / fs, {"r": rots[k]}, 1 / fs)["r"]
        if "r" in cc.onsets:
            onsets.append((k, cc.onsets["r"]))
            assert state                                          # published flag switches on now, causally
    assert len(onsets) == 1
    k_on, t_on = onsets[0]
    assert k_on >= 66                                             # detector waits for foot-flat
    assert abs(t_on - 62 / fs) <= 1 / fs + 1e-9                   # heel strike dated just after the dip


class _FakeRedis:
    def __init__(self):
        self.kv, self.stream = {}, []

    @classmethod
    def from_url(cls, url):
        return cls()

    def ping(self):
        return True

    def set(self, k, v):
        self.kv[k] = v

    def delete(self, k):
        self.deleted = getattr(self, "deleted", []) + [k]

    def xadd(self, k, entry, maxlen=None, approximate=True):
        self.stream.append((k, entry))

    def pipeline(self, transaction=False):
        parent = self

        class P:
            def set(self, k, v): parent.set(k, v)
            def xadd(self, *a, **kw): parent.xadd(*a, **kw)
            def execute(self): pass
        return P()

    def close(self):
        pass


def _state():
    q = np.zeros(len(COORDS)); q[COORDS.index("pelvis_tx")], q[COORDS.index("pelvis_ty")], q[COORDS.index("pelvis_tz")] = 1.0, 0.9, 0.2
    return dict(t=1.5, q=q, dq=np.zeros_like(q), q_std=np.zeros_like(q), pelvis=np.array([1.0, 0.9, 0.2]),
                com=np.array([1.0, 0.95, 0.2]), calcn_r=np.zeros(3), calcn_l=np.zeros(3), contact={"r": True, "l": False},
                f_medial=1.2, fy=-2.0, mx=0.01, surrogate_time=1.4, quats={"pelvis_imu": [1, 0, 0, 0]})


def test_redis_keys_and_urdf_vector(monkeypatch):
    monkeypatch.setitem(sys.modules, "redis", types.SimpleNamespace(Redis=_FakeRedis))
    pub = publisher.RedisPublisher(namespace="ns", stream_maxlen=2000, scale=1.1, wrap_m=4.0)
    st = _state(); st["q"][COORDS.index("pelvis_tx")] = 9.9
    pub.publish(st, {"calibrated": True})
    kv = pub.r.kv
    assert json.loads(kv["ns::joint_names"]) == COORDS
    assert json.loads(kv["ns::contact"]) == [1.0, 0.0] and float(kv["ns::f_medial"]) == 1.2
    qu = json.loads(kv["ns::q_urdf"])
    names = publisher.URDF_Q_NAMES
    assert len(qu) == 37 and names[:6] == ["pelvis_tx", "pelvis_ty", "pelvis_tz", "pelvis_tilt", "pelvis_list", "pelvis_rotation"]
    assert qu[:3] == pytest.approx([9.9, -0.2, 0.9])                              # robot frame: x fwd, y left, z up
    jp = json.loads(kv["ns::joint_positions"])                                     # OpenSai SimViz key
    assert len(jp) == 37 and len(json.loads(kv["ns::joint_velocities"])) == 37
    assert jp[0] == pytest.approx(((9.9 / 1.1) + 2.0) % 4.0 - 2.0)                # unscaled URDF units, wrapped
    assert jp[1:3] == pytest.approx([-0.2 / 1.1, 0.9 / 1.1]) and jp[3:] == qu[3:]
    assert "ns::sensor::pelvis_imu::orientation" in kv and pub.r.stream[0][0] == "ns::stream"
    st["f_medial"] = float("nan")
    pub2 = publisher.RedisPublisher(namespace="ns2")                                # stream off by default
    pub2.publish(st)
    assert "ns2::f_medial" not in pub2.r.kv and not pub2.r.stream and "ns2::stream" in pub2.r.deleted


class _Joint:
    def __init__(self, name, R, children=(), grounded=False, pos=None):
        self.name, self.R, self.children, self.grounded, self.pos = name, R, list(children), grounded, pos

    def get_name(self): return self.name
    def get_local_rotation(self):
        x, y, z, w = self.R.as_quat()
        return w, x, y, z
    def get_children(self): return self.children
    def get_grounding_state(self): return 0 if self.grounded else 1
    def get_local_position(self): return self.pos if self.pos is not None else (None, None, None)


def test_mocapapi_source_rebuilds_global_orientations(monkeypatch):
    r = lambda ax, deg: Rotation.from_euler(ax, deg, degrees=True)
    leg = lambda side, g: _Joint(f"{side}UpLeg", r("x", 20), [_Joint(f"{side}Leg", r("x", -40),
                                  [_Joint(f"{side}Foot", r("x", 10), grounded=g)])])
    hips = _Joint("Hips", r("y", 90), [leg("Right", True), leg("Left", False),
                                       _Joint("Spine", r("z", 0), [_Joint("Spine1", r("z", 0), [_Joint("Spine2", r("z", 5))])])],
                  pos=(10.0, 90.0, -5.0))
    fake = types.SimpleNamespace(MCPApplication=object, MCPSettings=object, MCPEventType=types.SimpleNamespace(AvatarUpdated=256),
                                 MCPAvatar=lambda h: types.SimpleNamespace(get_root_joint=lambda: hips))
    monkeypatch.setitem(sys.modules, "mocap_api", fake)
    from pn3_redis.sources import MocapApiSource
    src = MocapApiSource.__new__(MocapApiSource)                                  # skip opening a socket
    src.labels, src.scale, src.map = list(DEFAULT_SENSORS), 0.01, None
    f = src._frame(fake.MCPAvatar(0))
    expected = r("y", 90) * r("x", 20) * r("x", -40)                             # Hips * UpLeg * Leg
    got = Rotation.from_quat(np.r_[f.quats["tibia_r_imu"][1:], f.quats["tibia_r_imu"][0]])
    assert np.degrees((got.inv() * expected).magnitude()) < 1e-6
    assert f.contact == {"r": True, "l": False}
    np.testing.assert_allclose(f.hip_position, [0.1, 0.9, -0.05])
    assert set(f.quats) == set(DEFAULT_SENSORS)


def test_sai_pelvis_angles_reproduce_orientation_under_literal_urdf_axes():
    import xml.etree.ElementTree as ET
    from pn3_redis.model import C, URDF, axis_angle
    root = ET.parse(URDF).getroot()
    axis = {j.get("name"): np.array([float(x) for x in j.find("axis").get("xyz").split()])
            for j in root.findall("joint") if j.find("axis") is not None}
    rng = np.random.default_rng(1)
    for _ in range(20):
        q = np.zeros(len(COORDS))
        q[:3] = rng.uniform(-0.4, 0.4, 3)
        jp, _jv = publisher.sai_joint_vectors(q, np.zeros_like(q))
        R_urdf = np.eye(3)
        for k, name in enumerate(("pelvis_tilt", "pelvis_list", "pelvis_rotation")):      # chain order in the URDF
            R_urdf = R_urdf @ axis_angle(axis[name], np.array([jp[3 + k]]))[0]
        R_osim = LowerBody().fk(q)["pelvis"][0][0]
        np.testing.assert_allclose(R_urdf, C @ R_osim @ C.T, atol=1e-9)


def test_sai_pelvis_rate_continuous_across_heading_wrap():
    q, dq = np.zeros(len(COORDS)), np.zeros(len(COORDS))
    q[COORDS.index("pelvis_rotation")] = np.pi - 2e-5                 # facing -x: URDF yaw angle near +/-pi
    dq[COORDS.index("pelvis_rotation")] = 1.0
    _jp, jv = publisher.sai_joint_vectors(q, dq)
    assert abs(abs(jv[3]) - 1.0) < 1e-3                                # ~1 rad/s, not ~6e4

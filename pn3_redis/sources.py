"""Frame sources: live Axis Studio (MocapApi), BVH replay, and OpenSense orientation-file replay.

Each source yields Frame objects: a timestamp, one world-frame orientation (w, x, y, z) per sensor label
(pelvis_imu, femur_r_imu, ...), and optionally foot contact and a hip position when the source provides them.
"""
import os
import time
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from imu import bvh, segments
from imu.opensense_io import from_scipy, read_orientations, to_scipy


@dataclass
class Frame:
    t: float
    quats: dict                                   # label -> (4,) w, x, y, z in the sensor world frame
    contact: dict = None                          # {"r": bool, "l": bool} when the source knows it
    hip_position: np.ndarray = None               # metres, sensor world frame
    extra: dict = field(default_factory=dict)


def _pace(frames, realtime, speed=1.0):
    """Yield frames, sleeping so replays run at their recorded rate (times speed)."""
    start, t0 = None, None
    for f in frames:
        if realtime:
            if start is None:
                start, t0 = time.monotonic(), f.t
            lag = (f.t - t0) / speed - (time.monotonic() - start)
            if lag > 0:
                time.sleep(lag)
        yield f


class StoSource:
    """Replay an OpenSense orientation file (imu/synthetic.py or imu/from_bvh.py output).

    resample_hz emulates the suit's rate (PN3: 60 Hz). With a synthetic file, contact_trial (the source trial folder)
    supplies force-plate contact as a stand-in for Axis Studio's own contact flags.
    """

    def __init__(self, path, realtime=False, speed=1.0, resample_hz=None, contact_trial=None, loop=False):
        self.t, self.q, self.rate = read_orientations(path)
        if resample_hz:
            tn = np.arange(self.t[0], self.t[-1], 1.0 / resample_hz)
            self.q = {k: from_scipy(Slerp(self.t, to_scipy(v))(tn)) for k, v in self.q.items()}
            self.t, self.rate = tn, resample_hz
        self.realtime, self.speed, self.loop = realtime, speed, loop
        self.contact = None
        syn_path = os.path.join(os.path.dirname(os.path.abspath(path)), "synthetic.json")
        self.calib_s = 0.0
        if os.path.exists(syn_path):
            import json
            self.calib_s = float(json.load(open(syn_path))["calib_s"])
        if contact_trial:
            import json
            from imu.contact import grf_contact
            meta = json.load(open(os.path.join(contact_trial, "metadata.json")))
            syn = json.load(open(os.path.join(os.path.dirname(os.path.abspath(path)), "synthetic.json")))
            rc, lc = grf_contact(contact_trial, self.t + syn["time_offset"], float(meta["mass_kg"]))
            standing = self.t < syn["calib_s"]
            self.contact = (rc | standing, lc | standing)

    def __iter__(self):
        def frames():
            offset, k0 = 0.0, 0
            while True:
                for k in range(k0, len(self.t)):
                    c = None if self.contact is None else {"r": bool(self.contact[0][k]), "l": bool(self.contact[1][k])}
                    yield Frame(self.t[k] + offset, {lab: v[k] for lab, v in self.q.items()}, c)
                if not self.loop:
                    return
                k1 = int(np.searchsorted(self.t, self.calib_s))         # later laps skip the standing segment
                offset = self.t[-1] + offset + 1.0 / self.rate - self.t[k1]
                k0 = k1
        return _pace(frames(), self.realtime, self.speed)


class BVHSource:
    """Replay an Axis Studio BVH export: global bone orientations for the mapped segments, plus the hip position."""

    def __init__(self, path, labels=segments.LOWER_BODY, scale=0.01, realtime=False, speed=1.0):
        self.b = bvh.read(path)
        self.map = segments.resolve(self.b.order, labels)
        self.fk = bvh.forward_kinematics(self.b, scale=scale, joints=list(self.map.values()) + [self.b.root])
        self.realtime, self.speed = realtime, speed

    def __iter__(self):
        q = {lab: from_scipy(self.fk[j][1]) for lab, j in self.map.items()}
        hip = self.fk[self.b.root][0]
        frames = (Frame(k * self.b.frame_time, {lab: v[k] for lab, v in q.items()}, None, hip[k])
                  for k in range(self.b.n_frames))
        return _pace(frames, self.realtime, self.speed)


class MocapApiSource:
    """Live frames from Axis Studio through Noitom's MocapApi Python wrapper (pnmocap/MocapApi, demo/demo-py).

    In Axis Studio: Settings > BVH Broadcasting, enable BVH-Capture (live) or BVH-Edit (playback of a recorded take),
    UDP, destination = this machine, port = udp_port, displacement on. Global bone orientations are rebuilt here from
    the local rotations; foot contact is Axis Studio's grounding state of the foot joints.
    """

    def __init__(self, udp_port=7012, labels=segments.LOWER_BODY, position_scale=0.01, timeout_s=10.0):
        try:
            from mocap_api import MCPApplication, MCPAvatar, MCPEventType, MCPSettings
        except ImportError as e:
            raise SystemExit("MocapApi Python wrapper not found: pip install it from "
                             "https://github.com/pnmocap/MocapApi (demo/demo-py/MocapApi)") from e
        self._MCPAvatar, self._AvatarUpdated = MCPAvatar, MCPEventType.AvatarUpdated
        self.app = MCPApplication()
        settings = MCPSettings()
        settings.set_udp(udp_port)
        self.app.set_settings(settings)
        self.app.open()
        self.labels, self.scale, self.timeout = labels, position_scale, timeout_s
        self.map = None

    def _walk(self, joint, R_parent, out):
        w, x, y, z = joint.get_local_rotation()
        R = R_parent * Rotation.from_quat([x, y, z, w])
        out[joint.get_name()] = (R, joint)
        for c in joint.get_children():
            self._walk(c, R, out)

    def _frame(self, avatar):
        joints = {}
        root = avatar.get_root_joint()
        self._walk(root, Rotation.identity(), joints)
        if self.map is None:
            self.map = segments.resolve(list(joints), self.labels)
        quats = {lab: from_scipy(joints[j][0])[0] for lab, j in self.map.items()}
        contact = {}
        for side, foot in (("r", "Right"), ("l", "Left")):
            names = [n for n in joints if n.startswith(foot) and ("Foot" in n or "Toe" in n)]
            contact[side] = any(joints[n][1].get_grounding_state() == 0 for n in names) if names else None
        pos = root.get_local_position()
        hip = None if pos[0] is None else np.array(pos) * self.scale
        return Frame(time.monotonic(), quats, contact if None not in contact.values() else None, hip)

    def __iter__(self):
        last = time.monotonic()
        try:
            while True:
                got = False
                for evt in self.app.poll_next_event():
                    if evt.event_type == self._AvatarUpdated:
                        got = True
                        yield self._frame(self._MCPAvatar(evt.event_data.avatar_handle))
                if got:
                    last = time.monotonic()
                elif time.monotonic() - last > self.timeout:
                    raise SystemExit(f"no data from Axis Studio for {self.timeout:.0f} s: check BVH broadcasting (UDP, port)")
                else:
                    time.sleep(0.001)
        finally:
            self.app.close()

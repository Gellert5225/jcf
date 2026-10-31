"""Lower-body kinematics of the Rajagopal model, in OpenSim's convention, for the estimator.

The geometry comes from assets/Rajagopal2015.urdf (the model OpenSai simulates). That file re-expresses OpenSim's
y-up frames in z-up coordinates: a vector v in an OpenSim body frame appears as C v in the matching URDF link frame,
C = [[1,0,0],[0,0,-1],[0,1,0]]. Every joint below the pelvis is mapped back with C^T, so link frames, joint axes and
coordinate values here are OpenSim's: the joint vector q can go straight into OpenSim or the knee-load surrogate.

The pelvis is OpenSim's ground_pelvis joint, written out explicitly: translation (tx, ty, tz) along ground axes,
then the body-fixed rotation sequence Rz(tilt) Rx(list) Ry(rotation). (The URDF's pelvis chain keeps OpenSim's raw
axes inside its z-up frame, so it is not used here.)

Free coordinates (19): the surrogate's 16 (pelvis 6, hips 3+3, knees, ankles) plus the 3 lumbar joints, so the
torso sensor has something to move. Subtalar, MTP and the arms are locked at 0. The URDF freezes the walker knee's
coupled motions at the default value (up to ~14 deg of tibia error at deep flexion), so the femur -> tibia transform
is instead interpolated from knee_table.npz, sampled from the OpenSim walker knee by make_knee_table.py.
"""
import os
import xml.etree.ElementTree as ET

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URDF = os.path.join(REPO, "assets", "Rajagopal2015.urdf")
KNEE_TABLE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "knee_table.npz")
C = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])          # URDF = C @ OpenSim
GENERIC_HEIGHT_M = 1.70                                         # translations are scaled by height / this

COORDS = ["pelvis_tilt", "pelvis_list", "pelvis_rotation", "pelvis_tx", "pelvis_ty", "pelvis_tz",
          "hip_flexion_r", "hip_adduction_r", "hip_rotation_r", "knee_angle_r", "ankle_angle_r",
          "hip_flexion_l", "hip_adduction_l", "hip_rotation_l", "knee_angle_l", "ankle_angle_l",
          "lumbar_extension", "lumbar_bending", "lumbar_rotation"]
TRANSLATIONS = {"pelvis_tx", "pelvis_ty", "pelvis_tz"}
SENSOR_BODIES = ["pelvis", "femur_r", "tibia_r", "calcn_r", "femur_l", "tibia_l", "calcn_l", "torso"]


def rpy_matrix(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def rotvec_matrix(rv):
    """Batched rotation matrices (S, 3, 3) from rotation vectors (S, 3)."""
    th = np.linalg.norm(rv, axis=1)
    a = rv / np.where(th > 1e-12, th, 1.0)[:, None]
    K = np.zeros((len(rv), 3, 3))
    K[:, 0, 1], K[:, 0, 2], K[:, 1, 2] = -a[:, 2], a[:, 1], -a[:, 0]
    K = K - K.transpose(0, 2, 1)
    s, c = np.sin(th)[:, None, None], np.cos(th)[:, None, None]
    return np.eye(3) + s * K + (1 - c) * (K @ K)


def axis_angle(axis, ang):
    """Batched rotation matrices (S, 3, 3) about a fixed unit axis by angles (S,)."""
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    s, c = np.sin(ang)[:, None, None], np.cos(ang)[:, None, None]
    return np.eye(3) + s * K + (1 - c) * (K @ K)


class LowerBody:
    """Batched forward kinematics. q: (S, 19) in COORDS order (radians, metres)."""

    def __init__(self, urdf=URDF, height_m=GENERIC_HEIGHT_M, knee_table=KNEE_TABLE):
        self.scale = height_m / GENERIC_HEIGHT_M
        self.knee = dict(np.load(knee_table)) if knee_table and os.path.exists(knee_table) else None
        root = ET.parse(urdf).getroot()
        self.joints = {}           # child link -> (parent link, joint name, type, p, R, axis)
        for j in root.findall("joint"):
            o, a = j.find("origin"), j.find("axis")
            xyz = np.array([float(v) for v in (o.get("xyz") if o is not None else "0 0 0").split()])
            rpy = [float(v) for v in (o.get("rpy") if o is not None else "0 0 0").split()]
            ax = np.array([float(v) for v in a.get("xyz").split()]) if a is not None else np.zeros(3)
            child, parent = j.find("child").get("link"), j.find("parent").get("link")
            self.joints[child] = (parent, j.get("name"), j.get("type"), C.T @ xyz * self.scale,
                                  C.T @ rpy_matrix(*rpy) @ C, C.T @ ax)
        self.mass, self.com_local = {}, {}
        for L in root.findall("link"):
            i = L.find("inertial")
            if i is not None and float(i.find("mass").get("value")) > 0.01:
                o = i.find("origin")
                xyz = np.array([float(v) for v in (o.get("xyz") if o is not None else "0 0 0").split()])
                self.mass[L.get("name")] = float(i.find("mass").get("value"))
                self.com_local[L.get("name")] = C.T @ xyz * self.scale
        self.total_mass = sum(self.mass.values())
        # every link reachable from the pelvis, parents before children
        kids = {}
        for child, (parent, *_rest) in self.joints.items():
            kids.setdefault(parent, []).append(child)
        self.order, stack = [], list(kids.get("pelvis", []))
        while stack:
            c = stack.pop(0)
            self.order.append(c)
            stack += kids.get(c, [])
        self.index = {n: i for i, n in enumerate(COORDS)}
        self.lower = np.array([-np.pi] * 3 + [-1e4, -1e4, -1e4] + [0] * 13, float)
        self.upper = np.array([np.pi] * 3 + [1e4, 1e4, 1e4] + [0] * 13, float)
        for child, (_p, name, typ, *_r) in self.joints.items():
            if name in self.index and name not in TRANSLATIONS and not name.startswith("pelvis"):
                lim = [j for j in root.findall("joint") if j.get("name") == name][0].find("limit")
                self.lower[self.index[name]], self.upper[self.index[name]] = float(lim.get("lower")), float(lim.get("upper"))

    def fk(self, q):
        """-> dict link -> (R (S,3,3), p (S,3)) in the OpenSim ground frame, for the pelvis and every link below it."""
        q = np.atleast_2d(np.asarray(q, float))
        S = len(q)
        I = self.index
        Rp = axis_angle([0, 0, 1], q[:, I["pelvis_tilt"]]) @ axis_angle([1, 0, 0], q[:, I["pelvis_list"]]) \
            @ axis_angle([0, 1, 0], q[:, I["pelvis_rotation"]])
        pp = q[:, [I["pelvis_tx"], I["pelvis_ty"], I["pelvis_tz"]]].copy()
        out = {"pelvis": (Rp, pp)}
        for child in self.order:
            parent, name, typ, p0, R0, ax = self.joints[child]
            if self.knee is not None and child in ("tibia_r", "tibia_l"):
                out[child] = self._tibia(out, child[-1], q[:, I[f"knee_angle_{child[-1]}"]])
                continue
            Rpar, ppar = out[parent]
            R = Rpar @ R0
            p = ppar + np.einsum("sij,j->si", Rpar, p0)
            if typ in ("revolute", "continuous") and name in I:
                R = R @ axis_angle(ax, q[:, I[name]])
            elif typ == "prismatic" and name in I:
                p = p + np.einsum("sij,j->si", R, ax) * q[:, I[name]][:, None]
            out[child] = (R, p)
        return out

    def _tibia(self, out, side, th):
        """Tibia frame from the femur frame and the tabulated walker knee."""
        tb = self.knee["theta"]
        rv = np.column_stack([np.interp(th, tb, self.knee[f"rotvec_{side}"][:, k]) for k in range(3)])
        pu = np.column_stack([np.interp(th, tb, self.knee[f"p_{side}"][:, k]) for k in range(3)])
        _p, _n, _t, pk, Rk, _a = self.joints[f"knee_angle_{side}_link"]
        femur_len = np.linalg.norm(pk + Rk @ self.joints[f"tibia_{side}"][3])     # URDF tibia origin at 0 deg (scaled),
        # the same femur-to-tibia distance make_knee_table.py normalised by, so the thigh matches the URDF OpenSai draws
        Rf, pf = out[f"femur_{side}"]
        return Rf @ rotvec_matrix(rv), pf + np.einsum("sij,sj->si", Rf, pu * femur_len)

    def com(self, frames):
        """Whole-body centre of mass (S, 3) from fk() output (arms in the default pose)."""
        acc = 0.0
        for link, m in self.mass.items():
            if link in frames:
                R, p = frames[link]
                acc = acc + m * (p + np.einsum("sij,j->si", R, self.com_local[link]))
        return acc / sum(m for l, m in self.mass.items() if l in frames)

    def default_q(self):
        """Upright default pose (all zero) with the pelvis raised so the heels (calcn origins) sit at y = 0."""
        q = np.zeros(len(COORDS))
        f = self.fk(q)
        q[self.index["pelvis_ty"]] = -min(f["calcn_r"][1][0, 1], f["calcn_l"][1][0, 1])
        return q

    def clip(self, q):
        return np.clip(q, self.lower, self.upper)

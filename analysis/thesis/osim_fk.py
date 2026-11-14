"""Numpy forward kinematics of an OpenSim 4 model's joint chain (no OpenSim needed).

Used by analysis/thesis/sep23_seeds.py to place the optimizer's foot point (0.13, -0.01, 0) m in each calcn frame
when the OpenSim-written cache (pipeline/compute_com_features.py --foot-point ...) is missing from a trial folder.
It reproduces what compute_for_trial does with the OpenSim API: every IK coordinate is set with
Coordinate.setValue(..., enforceConstraints=False), then Frame.findStationLocationInGround.

Supported: CustomJoint (SpatialTransform with LinearFunction, SimmSpline, Constant and MultiplierFunction),
PinJoint (rotation about the joint frame's z axis), WeldJoint; PhysicalOffsetFrame parent/child frames
(translation, body-fixed x-y-z orientation). Defaults match what the OpenSim-written caches contain (checked against
foot_point_0.130_-0.010_0.000.npz / com_features.npz of training trials from all six labs, see
results/thesis/sep23/fk_validation.csv): locked coordinates keep their default value (locked="default"; the IK
value is ignored, as Coordinate.setValue does for a locked coordinate) and values outside a clamped range are
used as given (clamp=False; clamping reproduces OpenSim only where IK stays in range).

    fk = OsimChain("scaled_model.osim")
    R, p = fk.body_pose("calcn_r", coords)     # coords: {name: (T,) array}; R (T,3,3), p (T,3) in ground
    x = fk.station("calcn_r", (0.13, -0.01, 0.0), coords)
"""
import xml.etree.ElementTree as ET

import numpy as np


def _vec(text, n=3):
    v = np.array([float(x) for x in (text or "").split()], dtype=float)
    return v if len(v) == n else np.zeros(n)


def _axis_rot(axis, ang):
    """(T, 3, 3) rotation matrices about a fixed unit axis by angles (T,)."""
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    ang = np.asarray(ang, float)
    s, c = np.sin(ang)[:, None, None], np.cos(ang)[:, None, None]
    return np.eye(3)[None] + s * K[None] + (1 - c) * (K @ K)[None]


def _xyz_body_fixed(o):
    """Rotation of a PhysicalOffsetFrame: body-fixed x-y-z sequence Rx(o0) Ry(o1) Rz(o2)."""
    R = np.eye(3)
    for k, ang in enumerate(o):
        ax = np.zeros(3); ax[k] = 1.0
        R = R @ _axis_rot(ax, [ang])[0]
    return R


class SimmSpline:
    """OpenSim SimmSpline (Forsythe-Malcolm-Moler cubic spline; linear extrapolation outside the knots)."""

    def __init__(self, x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        n = len(x)
        self.x, self.y = x, y
        b, c, d = np.zeros(n), np.zeros(n), np.zeros(n)
        if n < 2:
            raise ValueError("SimmSpline needs at least 2 points")
        if n < 4:   # OpenSim: fewer than 4 points -> linear segments (only the n == 2 case occurs in practice)
            if n == 2:
                t = (y[1] - y[0]) / (x[1] - x[0])
                b[:] = t
            else:
                b[0] = (y[1] - y[0]) / (x[1] - x[0])
                b[1] = (y[2] - y[1]) / (x[2] - x[1])
                b[2] = b[1]
            self.b, self.c, self.d = b, c, d
            return
        nm1, nm2 = n - 1, n - 2
        d[0] = x[1] - x[0]
        c[1] = (y[1] - y[0]) / d[0]
        for i in range(1, nm1):
            d[i] = x[i + 1] - x[i]
            b[i] = 2.0 * (d[i - 1] + d[i])
            c[i + 1] = (y[i + 1] - y[i]) / d[i]
            c[i] = c[i + 1] - c[i]
        b[0] = -d[0]
        b[nm1] = -d[n - 2]
        c[0] = 0.0
        c[nm1] = 0.0
        d31 = x[3] - x[1]; d20 = x[2] - x[0]; d1 = x[nm1] - x[n - 3]; d2 = x[nm2] - x[n - 4]
        d30 = x[3] - x[0]; d3 = x[nm1] - x[n - 4]
        c[0] = c[2] / d31 - c[1] / d20
        c[nm1] = c[nm2] / d1 - c[n - 3] / d2
        c[0] = c[0] * d[0] * d[0] / d30
        c[nm1] = -c[nm1] * d[n - 2] * d[n - 2] / d3
        for i in range(1, n):
            t = d[i - 1] / b[i - 1]
            b[i] -= t * d[i - 1]
            c[i] -= t * c[i - 1]
        c[nm1] /= b[nm1]
        for j in range(nm1):
            i = nm2 - j
            c[i] = (c[i] - d[i] * c[i + 1]) / b[i]
        b[nm1] = (y[nm1] - y[n - 2]) / d[n - 2] + d[n - 2] * (c[n - 2] + 2.0 * c[nm1])
        for i in range(nm1):
            b[i] = (y[i + 1] - y[i]) / d[i] - d[i] * (c[i + 1] + 2.0 * c[i])
            d[i] = (c[i + 1] - c[i]) / d[i]
            c[i] *= 3.0
        c[nm1] *= 3.0
        d[nm1] = d[n - 2]
        self.b, self.c, self.d = b, c, d

    def __call__(self, xq):
        xq = np.asarray(xq, float)
        x, y, b, c, d = self.x, self.y, self.b, self.c, self.d
        i = np.clip(np.searchsorted(x, xq, side="right") - 1, 0, len(x) - 2)
        dx = xq - x[i]
        out = y[i] + dx * (b[i] + dx * (c[i] + dx * d[i]))
        lo, hi = xq < x[0], xq > x[-1]
        out = np.where(lo, y[0] + (xq - x[0]) * b[0], out)
        out = np.where(hi, y[-1] + (xq - x[-1]) * b[-1], out)
        return out


def _function(el):
    """Callable f(*coord_arrays) for an OpenSim Function element."""
    tag = el.tag
    if tag == "LinearFunction":
        a, c0 = _vec(el.findtext("coefficients"), 2)
        return lambda *q: a * q[0] + c0
    if tag == "Constant":
        v = float(el.findtext("value"))
        return lambda *q: v
    if tag == "SimmSpline":
        sp = SimmSpline(_vec(el.findtext("x"), len(el.findtext("x").split())),
                        _vec(el.findtext("y"), len(el.findtext("y").split())))
        return lambda *q: sp(q[0])
    if tag == "MultiplierFunction":
        inner = _function(next(iter(el.find("function"))))
        s = float(el.findtext("scale"))
        return lambda *q: s * inner(*q)
    raise NotImplementedError(f"OpenSim function {tag!r} is not supported")


class OsimChain:
    JOINT_TAGS = ("CustomJoint", "PinJoint", "WeldJoint")

    def __init__(self, osim_path, locked="default", clamp=False):
        if locked not in ("value", "default"):
            raise ValueError("locked must be 'value' or 'default'")
        self.locked, self.clamp = locked, clamp
        root = ET.parse(osim_path).getroot()
        model = root.find("Model")
        self.coords = {}      # name -> dict(default, range, clamped, locked)
        self.joints = {}      # child body -> joint dict
        for tag in ("CustomJoint", "PinJoint", "WeldJoint", "UniversalJoint", "BallJoint", "FreeJoint"):
            for j in model.iter(tag):
                frames = {}
                for f in j.iter("PhysicalOffsetFrame"):
                    frames[f.get("name")] = dict(body=f.findtext("socket_parent").strip().split("/")[-1],
                                                 p=_vec(f.findtext("translation")),
                                                 R=_xyz_body_fixed(_vec(f.findtext("orientation"))))
                cnames = []
                for c in j.iter("Coordinate"):
                    rng = _vec(c.findtext("range"), 2)
                    self.coords[c.get("name")] = dict(
                        default=float(c.findtext("default_value") or 0.0), range=rng,
                        clamped=(c.findtext("clamped") or "").strip() == "true",
                        locked=(c.findtext("locked") or "").strip() == "true")
                    cnames.append(c.get("name"))

                def frame(sock):
                    name = j.findtext(sock).strip()
                    if name in frames:
                        return frames[name]
                    return dict(body=name.split("/")[-1], p=np.zeros(3), R=np.eye(3))

                par, chi = frame("socket_parent_frame"), frame("socket_child_frame")
                jd = dict(name=j.get("name"), type=tag, parent=par, child=chi, coords=cnames)
                if tag == "CustomJoint":
                    axes = []
                    for ta in j.find("SpatialTransform").findall("TransformAxis"):
                        fn_el = next(e for e in ta if e.tag not in ("coordinates", "axis"))
                        axes.append(dict(kind=ta.get("name")[:-1], coords=(ta.findtext("coordinates") or "").split(),
                                         axis=_vec(ta.findtext("axis")), f=_function(fn_el)))
                    jd["axes"] = axes
                elif tag not in ("PinJoint", "WeldJoint"):
                    jd["unsupported"] = True
                self.joints[chi["body"]] = jd

    def _value(self, name, coords, T):
        c = self.coords[name]
        if c["locked"] and self.locked == "default":
            v = np.full(T, c["default"])
        else:
            v = np.asarray(coords[name], float) if name in coords else np.full(T, c["default"])
        if self.clamp and c["clamped"]:
            v = np.clip(v, c["range"][0], c["range"][1])
        return v

    def _x_fm(self, jd, coords, T):
        R = np.repeat(np.eye(3)[None], T, 0)
        p = np.zeros((T, 3))
        if jd.get("unsupported"):
            raise NotImplementedError(f"joint {jd['name']} ({jd['type']}) is not supported")
        if jd["type"] == "PinJoint":
            R = _axis_rot([0, 0, 1], self._value(jd["coords"][0], coords, T))
        elif jd["type"] == "CustomJoint":
            for ax in jd["axes"]:
                args = [self._value(n, coords, T) for n in ax["coords"]]
                val = np.broadcast_to(np.asarray(ax["f"](*args) if args else ax["f"](), float), (T,))
                if ax["kind"] == "rotation":
                    R = R @ _axis_rot(ax["axis"], val)        # successive body-fixed rotations
                else:
                    p = p + val[:, None] * (ax["axis"] / np.linalg.norm(ax["axis"]))[None]   # along F axes
        return R, p

    def body_pose(self, body, coords):
        """(R_GB (T,3,3), p_GB (T,3)) of a body in ground for coordinate arrays coords {name: (T,)}."""
        T = len(next(iter(coords.values())))
        chain = []
        b = body
        while b != "ground":
            jd = self.joints[b]
            chain.append(jd)
            b = jd["parent"]["body"]
        R = np.repeat(np.eye(3)[None], T, 0)
        p = np.zeros((T, 3))
        for jd in reversed(chain):
            Pf, Cf = jd["parent"], jd["child"]
            # parent body -> F
            p = p + R @ Pf["p"]
            R = R @ Pf["R"]
            # F -> M
            Rfm, pfm = self._x_fm(jd, coords, T)
            p = p + np.einsum("tij,tj->ti", R, pfm)
            R = R @ Rfm
            # M -> child body: X_BM = (Cf.R, Cf.p) so X_MB = (Cf.R^T, -Cf.R^T Cf.p)
            p = p + R @ (-(Cf["R"].T @ Cf["p"]))
            R = R @ Cf["R"].T
        return R, p

    def station(self, body, point, coords):
        R, p = self.body_pose(body, coords)
        return p + R @ np.asarray(point, float)

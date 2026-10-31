"""Joint-state estimation from segment orientations: calibration, a causal contact detector, and a UKF.

State x = [q, dq] over the 19 coordinates of model.COORDS (OpenSim convention), constant-velocity process model.
Measurements, at each suit frame:
  * one orientation per sensor (pelvis, thighs, shanks, feet, torso): residual log(R_meas^T R_pred(q)), 3 per sensor
  * for each foot in contact: its stance point (heel or toe, whichever is lower) stays where it touched down, at
    floor height. This is what makes the pelvis position observable: the suit measures orientations only, so the
    base is carried forward through the planted leg (legged odometry).

Calibration (first calib_s seconds, subject standing upright, feet straight, like OpenSim's default pose):
  R_meas_i = A R_link_i(q) B_i, with A a rotation about the vertical that aligns the sensor world's heading with the
  model's (from the pelvis sensor's forward axis) and B_i each sensor's mounting on its segment.
"""
import numpy as np
from scipy.spatial.transform import Rotation

from .model import COORDS, TRANSLATIONS, LowerBody

DEFAULT_SENSORS = {"pelvis_imu": "pelvis", "femur_r_imu": "femur_r", "tibia_r_imu": "tibia_r", "calcn_r_imu": "calcn_r",
                   "femur_l_imu": "femur_l", "tibia_l_imu": "tibia_l", "calcn_l_imu": "calcn_l", "torso_imu": "torso"}
# sensor world -> OpenSim ground (x forward, y up, z right). Axis Studio's world is left-up-forward.
SENSOR_FRAMES = {"opensim": np.eye(3), "axis": np.array([[0.0, 0, 1], [0, 1, 0], [-1, 0, 0]])}


def quat_to_matrix(q_wxyz):
    q = np.asarray(q_wxyz, float)
    return Rotation.from_quat(np.r_[q[1:], q[0]]).as_matrix()


class CausalContact:
    """Foot contact from foot-sensor angular speed, online (no look-ahead): on below on_deg_s, off above off_deg_s.

    The offline detector in imu/contact.py extends each quiet run to the heel-strike and push-off peaks, which needs
    future frames. Online, stance is reported from foot-flat to push-off, a little shorter than force-plate stance;
    that is the part of stance where the foot is planted, which is what the base anchoring needs.

    With backdate_s > 0, each switch-on also dates the heel strike after the fact: walking back from the switch over
    the foot-slap peak to the speed dip where the foot's rotation reverses at initial contact, the frame after the
    dip (≈ force-plate onset, ~90 ms before foot-flat). The time is reported in `onsets` for that update only; the
    returned flags stay causal. Consumers that buffer frames (the surrogate) can relabel the frames since then.
    """

    def __init__(self, on_deg_s=150.0, off_deg_s=400.0, smooth_s=0.03, min_dwell_s=0.06, backdate_s=0.0):
        """Defaults approximate force-plate stance (tuned on a training trial); use ~100/200 for foot-flat only."""
        self.on, self.off, self.tau, self.dwell = np.radians(on_deg_s), np.radians(off_deg_s), smooth_s, min_dwell_s
        self.backdate = backdate_s
        self.prev, self.speed, self.state, self.since, self.hist = {}, {}, {}, {}, {}
        self.onsets = {}

    def reset(self, t, rots, state=True):
        for k, R in rots.items():
            self.prev[k], self.speed[k], self.state[k], self.since[k], self.hist[k] = R, 0.0, state, t, []

    def _heel_strike(self, k):
        """Time of the frame after the speed dip that precedes the foot-slap peak (within backdate_s)."""
        h = self.hist[k]
        w = np.array([x[1] for x in h])
        if len(w) >= 3:                               # 3-frame average: noise ripples would stop the walk early
            w = np.convolve(np.pad(w, 1, mode="edge"), np.ones(3) / 3, mode="valid")
        j, tol = len(w) - 1, np.radians(10.0)
        while j > 0 and w[j - 1] >= w[j] - tol:       # back through quiet stance, up the foot-slap peak
            j -= 1
        while j > 0 and w[j - 1] < w[j]:              # down to the dip at initial contact
            j -= 1
        return h[min(j + 1, len(h) - 1)][0]

    def update(self, t, rots, dt):
        out, self.onsets = {}, {}
        for k, R in rots.items():
            w = np.linalg.norm(Rotation.from_matrix(self.prev[k].T @ R).as_rotvec()) / max(dt, 1e-6)
            a = dt / (self.tau + dt)
            self.speed[k] = (1 - a) * self.speed[k] + a * w
            self.prev[k] = R
            if self.backdate:
                self.hist[k].append((t, w))
                while self.hist[k][0][0] < t - self.backdate:
                    self.hist[k].pop(0)
            if t - self.since[k] >= self.dwell:
                new = self.speed[k] < self.on if not self.state[k] else self.speed[k] < self.off
                if new != self.state[k]:
                    self.state[k], self.since[k] = new, t
                    if new and self.backdate:
                        self.onsets[k] = self._heel_strike(k)
            out[k] = self.state[k]
        return out


class PN3Estimator:
    def __init__(self, height_m=1.70, sensors=None, sensor_frame="opensim", heading_axis=(1.0, 0, 0),
                 sigma_orient_deg=2.0, sigma_contact_m=0.015, accel_rot=300.0, accel_trans=6.0, calib_s=1.0):
        self.model = LowerBody(height_m=height_m)
        self.toe_tip = np.array([0.24, -0.002, 0.0]) * self.model.scale       # calcn frame, metres
        self.sensors = dict(sensors or DEFAULT_SENSORS)
        self.M = SENSOR_FRAMES[sensor_frame] if isinstance(sensor_frame, str) else np.asarray(sensor_frame)
        self.heading_axis = np.asarray(heading_axis, float)
        self.r_orient = np.radians(sigma_orient_deg) ** 2
        self.r_contact = sigma_contact_m ** 2
        self.sig_a = np.array([accel_trans if c in TRANSLATIONS else accel_rot for c in COORDS])
        self.calib_s = calib_s
        self.n = len(COORDS)
        self._cal, self._t0, self.calibrated = {k: [] for k in self.sensors}, None, False
        self.t, self.x, self.P = None, None, None
        self.anchor = {"r": None, "l": None}        # (point name, anchor position)
        self.contact = {"r": True, "l": True}

    # ─── calibration ──────────────────────────────────────────────────────────────────────────
    def add_calibration_frame(self, t, quats):
        """Accumulate standing frames; returns True once calib_s seconds are collected and the estimator is ready."""
        self._t0 = t if self._t0 is None else self._t0
        for k in self.sensors:
            if k in quats:
                self._cal[k].append(np.r_[quats[k][1:], quats[k][0]])
        if t - self._t0 >= self.calib_s:
            self.calibrate(t)
        return self.calibrated

    def calibrate(self, t):
        R_cal = {k: self.M @ Rotation.from_quat(np.array(v)).mean().as_matrix() for k, v in self._cal.items() if v}
        missing = set(self.sensors) - set(R_cal)
        if missing:
            raise RuntimeError(f"no calibration data for {sorted(missing)}")
        f = R_cal["pelvis_imu"] @ self.heading_axis
        psi = np.arctan2(-f[2], f[0])                       # heading of the pelvis sensor's forward axis
        self.A = Rotation.from_euler("y", psi).as_matrix()
        self.q0 = self.model.default_q()
        F = self.model.fk(self.q0)
        self.B = {k: F[b][0][0].T @ self.A.T @ R_cal[k] for k, b in self.sensors.items()}
        self.floor = {(side, kind): self._point(F, side, kind)[0, 1] for side in "rl" for kind in ("heel", "toe")}
        self.x = np.r_[self.q0, np.zeros(self.n)]
        sd = np.r_[[np.radians(2.0) if c not in TRANSLATIONS else 0.01 for c in COORDS], [0.1] * self.n]
        self.P = np.diag(sd ** 2)
        self.t, self.calibrated = t, True
        self.anchor = {"r": None, "l": None}

    # ─── filter ───────────────────────────────────────────────────────────────────────────────
    def predict(self, dt):
        n = self.n
        F = np.eye(2 * n)
        F[:n, n:] = np.eye(n) * dt
        s2 = self.sig_a ** 2
        Q = np.zeros((2 * n, 2 * n))
        Q[:n, :n] = np.diag(s2 * dt ** 4 / 4)
        Q[:n, n:] = Q[n:, :n] = np.diag(s2 * dt ** 3 / 2)
        Q[n:, n:] = np.diag(s2 * dt ** 2)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def _point(self, F, side, kind):
        """Contact points on the (rigid) foot: the heel (calcn origin) and the toe tip, 24 cm forward of it."""
        R, p = F[f"calcn_{side}"]
        return p if kind == "heel" else p + np.einsum("sij,j->si", R, self.toe_tip)

    def _stance_point(self, side, F):
        """Heel early in stance, toe tip once the heel lifts: the lower of the two, re-anchored on a switch."""
        h = self._point(F, side, "heel")[0, 1] - self.floor[(side, "heel")]
        tz = self._point(F, side, "toe")[0, 1] - self.floor[(side, "toe")]
        return (side, "heel") if h <= tz else (side, "toe")

    def update(self, quats, contact):
        n2 = 2 * self.n
        mean_frames = self.model.fk(self.x[:self.n])
        for side in ("r", "l"):
            if contact.get(side):
                pt = self._stance_point(side, mean_frames)
                if self.anchor[side] is None or self.anchor[side][0] != pt:
                    pos = self._point(mean_frames, *pt)[0].copy()
                    pos[1] = self.floor[pt]
                    self.anchor[side] = (pt, pos)
            else:
                self.anchor[side] = None
        self.contact = {s: bool(contact.get(s)) for s in ("r", "l")}
        meas = [(k, self.M @ quat_to_matrix(q)) for k, q in quats.items() if k in self.sensors]
        anchors = [a for a in self.anchor.values() if a is not None]
        m = 3 * len(meas) + 3 * len(anchors)
        if m == 0:
            return
        # cubature sigma points (unscented transform with alpha=1, beta=0, kappa=0)
        L = np.linalg.cholesky(n2 * (self.P + 1e-12 * np.eye(n2)))
        X = np.vstack([self.x + L.T, self.x - L.T])                    # (2*n2, n2)
        Fr = self.model.fk(X[:, :self.n])
        Z = np.zeros((len(X), m))
        col = 0
        for k, Rm in meas:
            Rp = self.A @ Fr[self.sensors[k]][0] @ self.B[k]
            Z[:, col:col + 3] = Rotation.from_matrix(np.einsum("ji,sjk->sik", Rm, Rp)).as_rotvec()
            col += 3
        for pt, pos in anchors:
            Z[:, col:col + 3] = self._point(Fr, *pt) - pos
            col += 3
        w = 1.0 / len(X)
        zbar = Z.mean(axis=0)
        dZ, dX = Z - zbar, X - self.x
        Rn = np.diag(np.r_[np.full(3 * len(meas), self.r_orient), np.full(3 * len(anchors), self.r_contact)])
        S = w * dZ.T @ dZ + Rn
        Pxz = w * dX.T @ dZ
        K = np.linalg.solve(S, Pxz.T).T
        self.x = self.x + K @ (0.0 - zbar)
        self.P = self.P - K @ S @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        self.x[:self.n] = self.model.clip(self.x[:self.n])

    def step(self, t, quats, contact):
        dt = max(t - self.t, 1e-4)
        self.predict(dt)
        self.update(quats, contact)
        self.t = t
        return self.state()

    def state(self):
        q, dq = self.x[:self.n], self.x[self.n:]
        F = self.model.fk(q)
        return dict(t=self.t, q=q.copy(), dq=dq.copy(), pelvis=F["pelvis"][1][0].copy(), com=self.model.com(F)[0],
                    calcn_r=F["calcn_r"][1][0].copy(), calcn_l=F["calcn_l"][1][0].copy(),
                    calcn_r_R=F["calcn_r"][0][0].copy(), calcn_l_R=F["calcn_l"][0][0].copy(), contact=dict(self.contact),
                    q_std=np.sqrt(np.clip(np.diag(self.P)[:self.n], 0, None)))

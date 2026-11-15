#!/usr/bin/env python3
"""Knee adduction moment (KAM) along an exported optimizer trajectory CSV, for the antalgic-gait constraint.

KAM = external moment of a foot's ground force about the knee joint centre, on the tibia's anterior axis, in BW*H;
positive = medial loading on either knee (right: +x component, left: -x component of the moment in tibia axes).

Two versions per knee:
  kam_qs      quasi-static, from kinematics and the contact schedule only (what a kinematic optimization has):
              single stance -> BW straight down through the COM's ground projection; double support -> the foot
              carries s * BW at its foot point, s ramping linearly 0 -> 1 after its heel strike and 1 -> 0 before its
              toe-off. Tested on the Knee Grand Challenge implants (analysis/thesis/gc_kam.py, the quasi-static
              variant): r 0.50-0.68 with the measured medial force in stance and 7/10 gait-modification directions
              right (the learned surrogates: r 0.22-0.35, 4/10). It reads higher than the true KAM (peaks ~5-6 % BW*H),
              so use it relative to the reference gait, not against an absolute value.
  kam_wrench  from the exported foot wrench (force at the foot point plus the wrench moment), i.e. the full KAM when the
              wrench is physical (dynamic stage: no pulling, centre of pressure inside the foot).

Kinematics: the optimizer's own body model (pn3_redis.model.LowerBody, assets/Rajagopal2015.urdf with the OpenSim
walker knee), pelvis angles converted from the URDF's axes (inference_posture_surrogate.urdf_pelvis_to_opensim). The
knee centre is the tibia link origin (within ~1 cm of the OpenSim walker-knee joint centre).

CSV columns (robot frame x fwd / y left / z up, 100 Hz): time, q0..q36 (Rajagopal2015.urdf joint order),
com_position_{0,1,2}, {right,left}_foot_position_{0,1,2}, {right,left}_foot_wrench_{0..5}.

Usage (repo root):
    python -I tools/kam_trajectory_csv.py --csv robot_q_log.csv --mass 75 --height 1.7 --out kam_sep23
    -> kam_sep23.csv (per frame), kam_sep23.png, and a per-stance peak table on stdout
"""
import argparse
import os
import site
import sys

sys.path.append(site.getusersitepackages())
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from inference_posture_surrogate import urdf_pelvis_to_opensim  # noqa: E402
from pn3_redis.model import COORDS, LowerBody  # noqa: E402
from pn3_redis.publisher import URDF_Q_NAMES  # noqa: E402

G = 9.81
FOOT_POINT = np.array([0.13, -0.01, 0.0])     # the optimizer's foot point on calcn (OpenSim axes)
CONTACT_BW = 0.05


def to_opensim(v):
    """robot (x fwd, y left, z up) -> OpenSim ground (x fwd, y up, z right)."""
    v = np.asarray(v, float)
    return np.stack([v[..., 0], v[..., 2], -v[..., 1]], axis=-1)


def runs(on):
    e = np.flatnonzero(np.diff(np.r_[0, np.asarray(on, int), 0]))
    return list(zip(e[::2], e[1::2]))


def load_share(on_s, on_o):
    """Share of body weight on foot s under the quasi-static rule (1 in its single stance, linear ramps in double
    support, 0 in swing)."""
    share = np.where(on_s & ~on_o, 1.0, 0.0)
    n = len(on_s)
    for a, b in runs(on_s & on_o):
        ramp = (np.arange(b - a) + 0.5) / (b - a)
        loading = a > 0 and not on_s[a - 1]            # foot s just landed
        unloading = b < n and not on_s[b]              # foot s about to lift
        share[a:b] = ramp if loading else (1 - ramp if unloading else 0.5)
    return share


def kam_series(df, mass, height):
    bw, bwh = mass * G, mass * G * height
    q_urdf = df[[f"q{i}" for i in range(37)]].to_numpy(float)
    idx = {n: i for i, n in enumerate(URDF_Q_NAMES)}
    q = np.zeros((len(df), len(COORDS)))
    for j, n in enumerate(COORDS):
        q[:, j] = q_urdf[:, idx[n]]
    q[:, [COORDS.index(c) for c in ("pelvis_tx", "pelvis_ty", "pelvis_tz")]] = \
        to_opensim(q_urdf[:, [idx["pelvis_tx"], idx["pelvis_ty"], idx["pelvis_tz"]]])
    q[:, [COORDS.index(c) for c in ("pelvis_tilt", "pelvis_list", "pelvis_rotation")]] = \
        urdf_pelvis_to_opensim(q_urdf[:, [idx["pelvis_tilt"], idx["pelvis_list"], idx["pelvis_rotation"]]])
    fr = LowerBody(height_m=height).fk(q)
    com = to_opensim(df[[f"com_position_{k}" for k in range(3)]].to_numpy(float))
    out = {"time": df["time"].to_numpy(float)}
    W = {s: df[[f"{s}_foot_wrench_{k}" for k in range(6)]].to_numpy(float) for s in ("right", "left")}
    on = {s: np.abs(W[s][:, 2]) > CONTACT_BW * bw for s in ("right", "left")}
    for s, o, k in (("right", "left", "r"), ("left", "right", "l")):
        R, knee = fr[f"tibia_{k}"]
        Rc, pc = fr[f"calcn_{k}"]
        foot_fk = pc + Rc @ FOOT_POINT
        sign = 1.0 if k == "r" else -1.0
        # quasi-static: vertical BW share, at the COM projection in single stance, at the foot point in double support
        share = load_share(on[s], on[o])
        single = on[s] & ~on[o]
        cop = np.where(single[:, None], com, foot_fk)
        F = np.zeros_like(cop)
        F[:, 1] = share * bw
        M = np.cross(cop - knee, F)
        out[f"kam_qs_{k}"] = sign * np.einsum("tji,tj->ti", R, M)[:, 0] / bwh
        # from the exported wrench: force at the exported foot point plus the wrench moment
        Fw, Tw = to_opensim(W[s][:, 0:3]), to_opensim(W[s][:, 3:6])
        pw = to_opensim(df[[f"{s}_foot_position_{j}" for j in range(3)]].to_numpy(float))
        M = np.cross(pw - knee, Fw) + Tw
        out[f"kam_wrench_{k}"] = sign * np.einsum("tji,tj->ti", R, M)[:, 0] / bwh
        out[f"contact_{k}"] = on[s].astype(int)
        out[f"share_{k}"] = share
        out[f"foot_fk_vs_export_cm_{k}"] = np.linalg.norm(foot_fk - pw, axis=1) * 100
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--csv", required=True)
    ap.add_argument("--mass", type=float, required=True, help="kg")
    ap.add_argument("--height", type=float, required=True, help="m")
    ap.add_argument("--out", default="kam_trajectory", help="output prefix (.csv, .png)")
    a = ap.parse_args()
    df = pd.read_csv(a.csv).dropna(subset=["com_position_0"]).reset_index(drop=True)
    k = kam_series(df, a.mass, a.height)
    k.to_csv(a.out + ".csv", index=False)
    print(f"{len(k)} frames; foot point by FK vs exported, right/left median "
          f"{k.foot_fk_vs_export_cm_r.median():.2f} / {k.foot_fk_vs_export_cm_l.median():.2f} cm (pelvis conversion check)")
    print("\nper stance, peak KAM (% BW*H):   knee  start-end (s)   quasi-static   from wrench")
    for side in ("r", "l"):
        for s0, s1 in runs(k[f"contact_{side}"].to_numpy().astype(bool)):
            if s1 - s0 < 10:
                continue
            seg = k.iloc[s0:s1]
            print(f"{'':32s}{side:>4s}  {seg.time.iloc[0]:.2f}-{seg.time.iloc[-1]:.2f}"
                  f"   {100 * seg[f'kam_qs_{side}'].max():10.2f}   {100 * seg[f'kam_wrench_{side}'].max():11.2f}")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    for ax, side, name in ((axes[0], "r", "right"), (axes[1], "l", "left")):
        for s0, s1 in runs(k[f"contact_{side}"].to_numpy().astype(bool)):
            ax.axvspan(k.time.iloc[s0], k.time.iloc[s1 - 1], color="#ecebe6", lw=0)
        ax.plot(k.time, 100 * k[f"kam_qs_{side}"], color="#2a78d6", lw=2, label="quasi-static (COM + contact schedule)")
        ax.plot(k.time, 100 * k[f"kam_wrench_{side}"], color="#8a8986", lw=1, label="from the exported foot wrench")
        ax.axhline(0, color="#c3c2b7", lw=0.8)
        top = max(8.0, 1.3 * 100 * float(k[f"kam_qs_{side}"].max()))
        ax.set_ylim(-3.0, top)                         # wrench spikes (non-physical exports) may run off this scale
        ax.set_ylabel(f"{name} KAM (% BW·H)")
        ax.legend(loc="upper right", fontsize=8, frameon=False)
    axes[1].set_xlabel("time (s)   [shaded = foot in contact]")
    axes[0].set_title("Knee adduction moment, positive = medial loading")
    fig.tight_layout()
    fig.savefig(a.out + ".png", dpi=150)
    print(f"\nwrote {a.out}.csv, {a.out}.png")


if __name__ == "__main__":
    main()

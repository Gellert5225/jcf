"""Exploration: kinematic contact (imu.contact.detect_contact on GC heel/toe markers) vs plate stance edges."""
import sys, os, glob, re
sys.path.insert(0, "/home/gellert/Developer/medical_dt")
import numpy as np
import pandas as pd
from imu.contact import detect_contact
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from measured_explore import SUBJ, GC, rcsv, plate_feet

rows = []
for code, (fold, kdir, mass, side) in SUBJ.items():
    bw = mass * 9.81
    for tp in sorted(glob.glob(f"{GC}/{fold}/Gait Trials/Video Motion Data/*_trajectories.csv")):
        traj = pd.read_csv(tp).interpolate(limit_direction="both")
        tt = traj["time(sec)"].to_numpy(float); fs = 1 / np.median(np.diff(tt))
        pts = [np.c_[traj[f"{side}.{m}X"], traj[f"{side}.{m}Z"], traj[f"{side}.{m}Y"]] / 1000 for m in ("Heel", "Toe")]
        kc = detect_contact(pts, fs)
        tg, vg = plate_feet(rcsv(tp.replace("_trajectories.csv", "_grf.csv")), traj, side)
        on = vg > 0.05 * bw
        idx = np.flatnonzero(on)
        if not len(idx):
            continue
        for gi in np.flatnonzero(np.diff(idx) > 1):
            if (idx[gi + 1] - idx[gi]) * (tg[1] - tg[0]) < 0.08:
                on[idx[gi]:idx[gi + 1]] = True
        d = np.diff(on.astype(int)); ups, downs = np.flatnonzero(d == 1) + 1, np.flatnonzero(d == -1) + 1
        ke = np.diff(kc.astype(int)); kup, kdn = tt[np.flatnonzero(ke == 1) + 1], tt[np.flatnonzero(ke == -1) + 1]
        for u in ups:
            dn = downs[downs > u]
            if not len(dn) or not (0.45 < tg[dn[0]] - tg[u] < 1.2):
                continue
            ta, tb = tg[u], tg[dn[0]]
            e_on = kup[np.argmin(np.abs(kup - ta))] - ta if len(kup) else np.nan
            e_off = kdn[np.argmin(np.abs(kdn - tb))] - tb if len(kdn) else np.nan
            rows.append(dict(subj=code, trial=os.path.basename(tp), on_err_ms=1000 * e_on, off_err_ms=1000 * e_off))
df = pd.DataFrame(rows)
print(df.groupby("subj")[["on_err_ms", "off_err_ms"]].describe(percentiles=[.5]).round(1).T)

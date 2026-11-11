"""Exploration: complete instrumented-limb stance phases (kinematic contact, GC heel/toe markers) inside the b3d frame range."""
import sys, os
sys.path.insert(0, "/home/gellert/Developer/medical_dt")
import numpy as np
import pandas as pd
from imu.contact import detect_contact, _runs
GC = "/home/gellert/Developer/medical_dt/jcf/external/grand_challenge"
FOLD = {"1GC": ("GC3", "R"), "2GC": ("GC2", "R"), "3GC": ("GC1", "L")}
al = pd.read_csv(sys.argv[1])
rows = []
for _, a in al.iterrows():
    fold, side = FOLD[a.subj]
    tr = pd.read_csv(f"{GC}/{fold}/Gait Trials/Video Motion Data/{a.orig}_trajectories.csv").interpolate(limit_direction="both")
    r0, r1 = int(a.offset), int(a.offset) + int(a.n_b3d)
    tr = tr.iloc[r0:r1]
    fs = 1 / a.gc_dt
    out = {}
    for s in "RL":
        pts = [np.c_[tr[f"{s}.{m}X"], tr[f"{s}.{m}Z"], tr[f"{s}.{m}Y"]] / 1000 for m in ("Heel", "Toe")]
        c = detect_contact(pts, fs)
        runs = [(x, y) for x, y in _runs(c) if x > 0 and y < len(c) and 0.4 < (y - x) / fs < 1.5]
        out[s] = runs
    rows.append(dict(subj=a.subj, orig=a.orig, inst_stances=len(out[side]), contra_stances=len(out["L" if side == "R" else "R"]),
                     inst_frames=sum(y - x for x, y in out[side])))
df = pd.DataFrame(rows)
print(df.groupby("subj")[["inst_stances", "contra_stances", "inst_frames"]].sum())

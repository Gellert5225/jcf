"""Exploration: find the StatOp_Val time origin relative to GC synchronized time (via GRF)."""
import sys, os, glob
import numpy as np
import pandas as pd
GC = "/home/gellert/Developer/medical_dt/jcf/external/grand_challenge"
SO = f"{GC}/StatOp_Val_Data/3GC"

def read_mot(p):
    with open(p) as f:
        for i, l in enumerate(f):
            if l.strip().lower() == "endheader":
                break
    return pd.read_csv(p, sep=r"\s+", skiprows=i + 1)

out = []
for p in sorted(glob.glob(f"{SO}/ID/ExternalLoads/*_grf.mot")):
    trial = os.path.basename(p).replace("_grf.mot", "")
    m = read_mot(p)
    ts = m.iloc[:, 0].to_numpy(float)
    vcols = [c for c in m.columns if c.endswith("vy") and "force" in c]
    fs = np.abs(m[vcols].to_numpy(float)).sum(1)
    g = pd.read_csv(f"{GC}/GC1/Gait Trials/Video Motion Data/{trial}_grf.csv", index_col=False)
    g = g.loc[:, ~g.columns.str.startswith("Unnamed")]
    tg = g.iloc[:, 0].to_numpy(float)
    fg = np.abs(g[[c for c in g.columns if c.startswith("Fz")]].to_numpy(float)).sum(1)
    tr = pd.read_csv(f"{GC}/GC1/Gait Trials/Video Motion Data/{trial}_trajectories.csv")
    tt = tr["time(sec)"].to_numpy(float)
    # candidate offsets: t_gc = t_so + c
    best = None
    for c in np.arange(tg[0] - 0.05, tg[0] + 0.05, 0.0002):
        x = np.interp(ts + c, tg, fg, left=np.nan, right=np.nan)
        ok = np.isfinite(x)
        if ok.sum() < 100:
            continue
        e = np.mean(np.abs(x[ok] - fs[ok]))
        if best is None or e < best[1]:
            best = (c, e)
    out.append(dict(trial=trial, so_t0=ts[0], so_dt=np.median(np.diff(ts)), n_so=len(ts), vcols=len(vcols),
                    c=best[0], err_N=best[1], gc_t_row0=tt[0], gc_t_row1=tt[1], c_minus_row1=best[0] - tt[1],
                    c_minus_row0=best[0] - tt[0]))
    print(out[-1], flush=True)
pd.DataFrame(out).to_csv(sys.argv[1], index=False)

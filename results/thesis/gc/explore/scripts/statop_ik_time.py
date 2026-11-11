"""Exploration: StatOp IK time -> GC row, via IK model marker locations vs GC experimental markers."""
import sys, os, glob, re
import numpy as np
import pandas as pd
GC = "/home/gellert/Developer/medical_dt/jcf/external/grand_challenge"
SO = f"{GC}/StatOp_Val_Data/3GC"
norm = lambda n: re.sub(r"[^a-z0-9]", "", n.lower())

def read_sto(p):
    with open(p) as f:
        for i, l in enumerate(f):
            if l.strip().lower() == "endheader":
                break
    return pd.read_csv(p, sep=r"\s+", skiprows=i + 1)

out = []
for p in sorted(glob.glob(f"{SO}/IK/Results/*/instKnee/*_ik_model_marker_locations.sto")):
    tag = os.path.basename(p).replace("_ik_model_marker_locations.sto", "")
    trial = tag.rsplit("_stance", 1)[0]
    m = read_sto(p)
    tau = m.time.to_numpy(float)
    tr = pd.read_csv(f"{GC}/GC1/Gait Trials/Video Motion Data/{trial}_trajectories.csv")
    names = [c[:-3] for c in m.columns if c.endswith("_tx")]
    gcols = {norm(c[:-1]): c[:-1] for c in tr.columns if c.endswith("X")}
    common = [n for n in names if norm(n) in gcols]
    A = np.stack([m[[f"{n}_tx", f"{n}_ty"]].to_numpy(float) for n in common], 1)  # x, y(up)
    G = np.stack([np.c_[-tr[gcols[norm(n)] + "X"], tr[gcols[norm(n)] + "Z"]] / 1000 for n in common], 1)
    k = np.rint(tau * 120).astype(int)
    best = None
    for off in range(-3, 4):
        r = k + off
        ok = (r >= 0) & (r < len(G))
        e = np.nanmedian(np.abs(A[ok] - G[r[ok]]))
        if best is None or e < best[1]:
            best = (off, e)
    out.append(dict(tag=tag, tau0=tau[0], n=len(tau), dtau=np.median(np.diff(tau)), best_row_offset=best[0], err_mm=1000 * best[1]))
print(pd.DataFrame(out).to_string())
pd.DataFrame(out).to_csv(sys.argv[1], index=False)

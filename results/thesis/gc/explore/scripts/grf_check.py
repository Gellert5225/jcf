"""Exploration: check GRF sync (b3d raw plates vs GC grf.csv) and knee-force time base.

Run: conda run -n jcf python -I grf_check.py ALIGN_CSV OUT_CSV
"""
import sys, os, glob, re
sys.path.append(os.path.expanduser("~/.local/lib/python3.10/site-packages"))
import numpy as np
import pandas as pd
import nimblephysics as nimble

REPO = "/home/gellert/Developer/medical_dt"
GC = f"{REPO}/jcf/external/grand_challenge"
SUBJ = {
    "1GC": (f"{REPO}/with_arm/training/Fregly2012_Formatted_With_Arm/1GC/1GC.b3d", f"{GC}/GC3", "EKnee Data"),
    "2GC": (f"{REPO}/with_arm/training/Fregly2012_Formatted_With_Arm/2GC/2GC.b3d", f"{GC}/GC2", "eTibia Data"),
    "3GC": (f"{REPO}/with_arm/testing/Fregly2012_Formatted_With_Arm/3GC/3GC.b3d", f"{GC}/GC1", "eTibia Data"),
}


def norm(n):
    return re.sub(r"[^a-z0-9]", "", n.lower())


def rcsv(p):
    d = pd.read_csv(p, index_col=False)
    return d.loc[:, ~d.columns.str.startswith("Unnamed")]


def main(align_csv, out_csv):
    al = pd.read_csv(align_csv)
    rows = []
    subs = {}
    for _, r in al.iterrows():
        b3d, gcdir, kdir = SUBJ[r.subj]
        if r.subj not in subs:
            subs[r.subj] = nimble.biomechanics.SubjectOnDisk(b3d)
            s = subs[r.subj]
            print(r.subj, "plates trial0", s.getNumForcePlates(0))
        s = subs[r.subj]
        t, off = int(r.trial), int(r.offset)
        L = s.getTrialLength(t)
        fr = s.readFrames(t, 0, L, includeSensorData=True, includeProcessingPasses=True)
        raw = np.array([np.sum([np.asarray(x)[1] for x in f.rawForcePlateForces]) for f in fr])
        pp = np.array([np.asarray(f.processingPasses[-1].groundContactForce) for f in fr])  # [L, 6] calcn_r, calcn_l
        g = rcsv(os.path.join(gcdir, "Gait Trials", "Video Motion Data", f"{r.orig}_grf.csv"))
        tg = g.iloc[:, 0].to_numpy(float)
        fz = g[[c for c in g.columns if re.fullmatch(r"Fz\d", c)]].to_numpy(float)
        tr = rcsv(os.path.join(gcdir, "Gait Trials", "Video Motion Data", f"{r.orig}_trajectories.csv"))
        tt = tr[[c for c in tr.columns if c.lower().startswith("time")][0]].to_numpy(float)
        # GC vertical GRF at marker times of the b3d frames
        tm = tt[off:off + L]
        gz = np.interp(tm, tg, np.abs(fz).sum(1))
        n = min(len(gz), L)
        # best lag (in marker frames) between b3d raw vertical and GC vertical
        lags = range(-10, 11)
        cc = []
        for lg in lags:
            a = raw[max(0, lg):n + min(0, lg)]
            b = gz[max(0, -lg):n - max(0, lg)]
            cc.append(np.corrcoef(a, b)[0, 1] if a.std() > 0 and b.std() > 0 else np.nan)
        best = list(lags)[int(np.nanargmax(cc))]
        ratio = np.nanmedian(raw[:n][gz[:n] > 200] / gz[:n][gz[:n] > 200]) if (gz[:n] > 200).any() else np.nan
        # knee file
        cands = {norm(os.path.basename(p).replace("_knee_forces.csv", "")): p
                 for p in glob.glob(os.path.join(gcdir, "Gait Trials", kdir, "*_knee_forces.csv"))}
        kp = cands.get(norm(r.orig))
        kd = rcsv(kp)
        kt = kd.iloc[:, 0].to_numpy(float)
        k_eq = bool(len(kt) == len(tt) and np.allclose(kt, tt, atol=2e-3))
        k_nan = int(kd.isna().sum().sum())
        rows.append(dict(subj=r.subj, trial=t, orig=r.orig, grf_rate=1 / np.median(np.diff(tg)),
                         n_plates_gc=fz.shape[1], grf_t0=tg[0], traj_t0=tt[0], grf_tN=tg[-1], traj_tN=tt[-1],
                         raw_vs_gc_best_lag=best, raw_vs_gc_r0=cc[10], raw_vs_gc_rbest=np.nanmax(cc),
                         raw_over_gc=ratio, knee_cols="|".join(kd.columns), knee_n=len(kt), knee_t0=kt[0],
                         knee_time_eq_traj=k_eq, knee_nan=k_nan))
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

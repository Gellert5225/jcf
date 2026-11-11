"""Exploration: align b3d trials with Grand Challenge synchronized files by marker matching.

Run: conda run -n jcf python -I align_explore.py OUT_CSV
Reads b3d (nimblephysics) and GC csv files as data only.
"""
import sys, os, glob, re
sys.path.append(os.path.expanduser("~/.local/lib/python3.10/site-packages"))
import numpy as np
import pandas as pd
import nimblephysics as nimble

REPO = "/home/gellert/Developer/medical_dt"
GC = f"{REPO}/jcf/external/grand_challenge"
SUBJ = {
    # code: (b3d, gc folder, knee-force folder)
    "1GC": (f"{REPO}/with_arm/training/Fregly2012_Formatted_With_Arm/1GC/1GC.b3d", f"{GC}/GC3", "EKnee Data"),
    "2GC": (f"{REPO}/with_arm/training/Fregly2012_Formatted_With_Arm/2GC/2GC.b3d", f"{GC}/GC2", "eTibia Data"),
    "3GC": (f"{REPO}/with_arm/testing/Fregly2012_Formatted_With_Arm/3GC/3GC.b3d", f"{GC}/GC1", "eTibia Data"),
}


def norm(n):
    return re.sub(r"[^a-z0-9]", "", n.lower())


def read_traj(path):
    df = pd.read_csv(path)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    tcol = [c for c in df.columns if c.lower().startswith("time")][0]
    t = df[tcol].to_numpy(float)
    mk = {}
    for c in df.columns:
        if c.endswith("X") and c[:-1] + "Y" in df.columns and c[:-1] + "Z" in df.columns:
            mk[norm(c[:-1])] = df[[c, c[:-1] + "Y", c[:-1] + "Z"]].to_numpy(float) / 1000.0
    return t, mk


def knee_file(folder, orig):
    cands = {norm(os.path.basename(p).replace("_knee_forces.csv", "")): p
             for p in glob.glob(os.path.join(folder, "*_knee_forces.csv"))}
    return cands.get(norm(orig))


def main(out_csv):
    rows = []
    for code, (b3d, gcdir, kdir) in SUBJ.items():
        s = nimble.biomechanics.SubjectOnDisk(b3d)
        for t in range(s.getNumTrials()):
            orig = s.getTrialOriginalName(t)
            L = s.getTrialLength(t)
            tp = os.path.join(gcdir, "Gait Trials", "Video Motion Data", f"{orig}_trajectories.csv")
            if not os.path.exists(tp):
                rows.append(dict(subj=code, trial=t, orig=orig, status="no_traj"))
                continue
            tg, mk = read_traj(tp)
            fr = s.readFrames(t, 0, L, includeSensorData=True, includeProcessingPasses=False)
            names = sorted({norm(m[0]) for f in fr for m in f.markerObservations} & set(mk))
            B = np.full((L, len(names), 3), np.nan)
            for i, f in enumerate(fr):
                for n, p in f.markerObservations:
                    k = norm(n)
                    if k in names:
                        B[i, names.index(k)] = np.asarray(p)
            G = np.stack([mk[n] for n in names], axis=1)  # [Tg, M, 3]
            # vertical: b3d y vs GC Z
            best = None
            for off in range(-(L - 20), len(tg) - 20):
                i0, i1 = max(0, -off), min(L, len(tg) - off)
                if i1 - i0 < 20:
                    continue
                d = np.abs(B[i0:i1, :, 1] - G[i0 + off:i1 + off, :, 2])
                e = np.nanmedian(d)
                if best is None or e < best[1]:
                    best = (off, e, i1 - i0)
            off, e, nov = best
            i0, i1 = max(0, -off), min(L, len(tg) - off)
            # horizontal check with x_b3d = -X, z_b3d = Y and also +X,-Y
            bx = B[i0:i1, :, 0]; bz = B[i0:i1, :, 2]
            gx = G[i0 + off:i1 + off, :, 0]; gy = G[i0 + off:i1 + off, :, 1]
            hx = min(np.nanmedian(np.abs(bx - sx * gx)) for sx in (1, -1))
            hz = min(np.nanmedian(np.abs(bz - sy * gy)) for sy in (1, -1))
            sx = 1 if np.nanmedian(np.abs(bx - gx)) < np.nanmedian(np.abs(bx + gx)) else -1
            sy = 1 if np.nanmedian(np.abs(bz - gy)) < np.nanmedian(np.abs(bz + gy)) else -1
            kp = knee_file(os.path.join(gcdir, "Gait Trials", kdir), orig)
            kn = kt0 = ktmatch = None
            if kp:
                kd = pd.read_csv(kp)
                kd = kd.loc[:, ~kd.columns.str.startswith("Unnamed")]
                kt = kd.iloc[:, 0].to_numpy(float)
                kn, kt0 = len(kt), kt[0]
                ktmatch = bool(len(kt) == len(tg) and np.allclose(kt, tg, atol=1e-3))
            rows.append(dict(subj=code, trial=t, orig=orig, split=s.getTrialSplitIndex(t), status="ok",
                             n_b3d=L, dt_b3d=s.getTrialTimestep(t), n_gc=len(tg),
                             gc_t0=tg[0], gc_dt=float(np.median(np.diff(tg))), offset=off,
                             b3d_t0_gc=tg[off] if 0 <= off < len(tg) else np.nan,
                             n_overlap=nov, vert_err_mm=1000 * e, x_err_mm=1000 * hx, z_err_mm=1000 * hz,
                             sign_x=sx, sign_z=sy, n_markers=len(names),
                             knee_file=os.path.basename(kp) if kp else None, knee_n=kn, knee_t0=kt0,
                             knee_time_eq_traj=ktmatch))
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)


if __name__ == "__main__":
    main(sys.argv[1])

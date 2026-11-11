"""Exploration: map every fregly_* segment folder to (b3d trial, start frame) by matching IK content.

Run: conda run -n jcf python -I contentmap.py OUT_CSV
"""
import sys, os, glob
sys.path.append(os.path.expanduser("~/.local/lib/python3.10/site-packages"))
import numpy as np
import pandas as pd
import nimblephysics as nimble

REPO = "/home/gellert/Developer/medical_dt"
B3D = {
    "1GC": f"{REPO}/with_arm/training/Fregly2012_Formatted_With_Arm/1GC/1GC.b3d",
    "2GC": f"{REPO}/with_arm/training/Fregly2012_Formatted_With_Arm/2GC/2GC.b3d",
    "3GC": f"{REPO}/with_arm/testing/Fregly2012_Formatted_With_Arm/3GC/3GC.b3d",
}


def read_mot(p):
    with open(p) as f:
        for i, l in enumerate(f):
            if l.strip() == "endheader":
                break
    return pd.read_csv(p, sep="\t", skiprows=i + 1)


def main(out_csv):
    rows = []
    for code, b3d in B3D.items():
        s = nimble.biomechanics.SubjectOnDisk(b3d)
        Q = {}
        for t in range(s.getNumTrials()):
            L = s.getTrialLength(t)
            fr = s.readFrames(t, 0, L, includeSensorData=False, includeProcessingPasses=True)
            Q[t] = np.array([np.asarray(f.processingPasses[-1].pos) for f in fr])
        dirs = sorted(glob.glob(f"{REPO}/jcf/full_duration/*/walking/fregly_{code}_*"))
        for d in dirs:
            ik = os.path.join(d, "ik_results.mot")
            if not os.path.exists(ik):
                continue
            m = read_mot(ik).iloc[:, 1:].to_numpy(float)
            hits = []
            for t, q in Q.items():
                if q.shape[1] != m.shape[1]:
                    continue
                err = np.abs(q[:, None, :] - m[None, :1, :]).max(-1)[:, 0]
                for sf in np.flatnonzero(err < 1e-6):
                    if sf + len(m) <= len(q) and np.abs(q[sf:sf + len(m)] - m).max() < 1e-6:
                        hits.append((t, int(sf)))
            rows.append(dict(subj=code, folder=os.path.basename(d), n=len(m),
                             hits=";".join(f"{t}:{sf}" for t, sf in hits),
                             trial=hits[0][0] if len(hits) == 1 else None,
                             start_frame=hits[0][1] if len(hits) == 1 else None))
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)


if __name__ == "__main__":
    main(sys.argv[1])

import sys, os
sys.path.append(os.path.expanduser("~/.local/lib/python3.10/site-packages"))
import sys
import numpy as np
import nimblephysics as nimble
for p in sys.argv[1:]:
    s = nimble.biomechanics.SubjectOnDisk(p)
    print("==", p, "mass", s.getMassKg(), "height", s.getHeightM(), "sex", s.getBiologicalSex(), "ntrials", s.getNumTrials())
    print("contact bodies", list(s.getGroundForceBodies()))
    for t in range(s.getNumTrials()):
        L = s.getTrialLength(t); dt = s.getTrialTimestep(t)
        on = s.getTrialOriginalName(t); sp = s.getTrialSplitIndex(t)
        fr = s.readFrames(t, 0, 1, includeSensorData=True, includeProcessingPasses=False)
        fr2 = s.readFrames(t, L-1, 1, includeSensorData=False, includeProcessingPasses=False)
        nfp = s.getNumForcePlates(t)
        print(f"t{t:02d} {s.getTrialName(t):24s} orig={on:22s} split={sp} n={L} dt={dt:.6f} t0={fr[0].t:.4f} tN={fr2[0].t:.4f} nFP={nfp} nmk={len(fr[0].markerObservations)} passes={s.getTrialNumProcessingPasses(t)}")

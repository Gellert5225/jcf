"""Exploration: dynamics vs kinematics pass pos on full b3d trials, split by missing-GRF frames; count missing GRF."""
import sys, os
sys.path.append(os.path.expanduser("~/.local/lib/python3.10/site-packages"))
import numpy as np
import nimblephysics as nimble
REPO = "/home/gellert/Developer/medical_dt"
B3D = {"1GC": f"{REPO}/with_arm/training/Fregly2012_Formatted_With_Arm/1GC/1GC.b3d",
       "2GC": f"{REPO}/with_arm/training/Fregly2012_Formatted_With_Arm/2GC/2GC.b3d",
       "3GC": f"{REPO}/with_arm/testing/Fregly2012_Formatted_With_Arm/3GC/3GC.b3d"}
NM = nimble.biomechanics.MissingGRFReason.notMissingGRF
for code, p in B3D.items():
    s = nimble.biomechanics.SubjectOnDisk(p)
    skel = s.readSkel(processingPass=s.getNumProcessingPasses() - 1, ignoreGeometry=True)
    names = [skel.getDofByIndex(i).getName() for i in range(skel.getNumDofs())]
    ki = [names.index(n) for n in ("knee_angle_r", "knee_angle_l", "hip_flexion_r", "hip_adduction_r", "pelvis_list")]
    dv, dm, nmiss, ntot, jumps = [], [], 0, 0, []
    for t in range(s.getNumTrials()):
        L = s.getTrialLength(t)
        fr = s.readFrames(t, 0, L, includeSensorData=False, includeProcessingPasses=True)
        k = np.array([np.asarray(f.processingPasses[0].pos)[ki] for f in fr])
        d = np.array([np.asarray(f.processingPasses[-1].pos)[ki] for f in fr])
        miss = np.array([m != NM for m in s.getMissingGRF(t)])
        nmiss += miss.sum(); ntot += L
        diff = np.degrees(np.abs(k - d)).max(1)
        dv += list(diff[~miss]); dm += list(diff[miss])
        jumps.append(np.degrees(np.abs(np.diff(d, axis=0))).max())
    print(code, "frames", ntot, "missingGRF", nmiss, f"({nmiss/ntot:.0%})",
          "max|dyn-kin| deg (median/95%) valid:", np.round(np.percentile(dv, [50, 95]), 2),
          "missing:", np.round(np.percentile(dm, [50, 95]), 2) if dm else None,
          "max frame-to-frame jump deg:", round(max(jumps), 2))

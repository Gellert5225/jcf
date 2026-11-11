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
    sel = ["pelvis_tilt", "pelvis_list", "pelvis_rotation", "hip_flexion_r", "hip_adduction_r", "hip_rotation_r", "knee_angle_r", "ankle_angle_r",
           "hip_flexion_l", "hip_adduction_l", "hip_rotation_l", "knee_angle_l", "ankle_angle_l"]
    ki = [names.index(n) for n in sel]
    big = []; nb = 0; nfr = 0
    for t in range(s.getNumTrials()):
        L = s.getTrialLength(t); dt = s.getTrialTimestep(t)
        fr = s.readFrames(t, 0, L, includeSensorData=False, includeProcessingPasses=True)
        d = np.array([np.asarray(f.processingPasses[-1].pos)[ki] for f in fr])
        miss = np.array([m != NM for m in s.getMissingGRF(t)])
        bnd = np.flatnonzero(np.diff(miss.astype(int))) 
        vel = np.degrees(np.abs(np.diff(d, axis=0))) / dt   # deg/s
        for i, j in zip(*np.nonzero(vel > 1500)):
            near = np.min(np.abs(bnd - i)) if len(bnd) else 999
            big.append((t, i, sel[j], round(vel[i, j]), L, near))
        nfr += L
    print(code, "frames", nfr, "n(|dq/dt|>1500 deg/s)", len(big), "examples (trial, frame, dof, deg/s, L, dist_to_missingGRF_boundary):", big[:8])

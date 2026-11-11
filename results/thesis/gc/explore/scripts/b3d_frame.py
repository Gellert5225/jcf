import sys, os
sys.path.append(os.path.expanduser("~/.local/lib/python3.10/site-packages"))
import numpy as np
import nimblephysics as nimble
p, t = sys.argv[1], int(sys.argv[2])
s = nimble.biomechanics.SubjectOnDisk(p)
fr = s.readFrames(t, 0, 3, includeSensorData=True, includeProcessingPasses=True)
f = fr[0]
print([a for a in dir(f) if not a.startswith('_')])
print("t", [x.t for x in fr])
print("markers", [(m[0], np.round(np.array(m[1]), 4).tolist()) for m in f.markerObservations][:8])
print("nplates", len(f.rawForcePlateForces), [np.round(np.array(x),2).tolist() for x in f.rawForcePlateForces])
print("cops", [np.round(np.array(x),3).tolist() for x in f.rawForcePlateCenterOfPressures])
pp = f.processingPasses[-1]
print("pp attrs", [a for a in dir(pp) if not a.startswith('_')])
print("pass types", [x.type for x in f.processingPasses])
print("subj attrs", [a for a in dir(s) if not a.startswith('_')])

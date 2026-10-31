"""Tabulate the OpenSim walker knee (femur -> tibia transform versus knee angle) for model.py.  [needs opensim]

    conda run -n jcf python -m pn3_redis.make_knee_table --model jcf/full_duration/testing/walking/carter_P010_split0_t10/scaled_model.osim

The URDF freezes the knee's coupled rotations and translations at the default angle, which leaves up to ~14 deg of
tibia orientation error at deep flexion. This samples the real joint from an OpenSim Rajagopal model, every 1 deg
over the URDF range, and writes pn3_redis/knee_table.npz: per side, the tibia frame's rotation (as a rotation
vector) and origin in the femur frame. Translations are divided by that model's femur length so model.py can rescale
them to any height.
"""
import argparse
import os

import numpy as np
from scipy.spatial.transform import Rotation


def main(argv=None):
    import opensim as osim
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "knee_table.npz"))
    a = ap.parse_args(argv)
    m = osim.Model(a.model)
    s = m.initSystem()
    cs = m.getCoordinateSet()
    theta = np.radians(np.arange(0, 121, 1.0))
    out = {"theta": theta}
    for side in ("r", "l"):
        femur, tibia = m.getBodySet().get(f"femur_{side}"), m.getBodySet().get(f"tibia_{side}")
        rv, p = [], []
        for th in theta:
            for i in range(cs.getSize()):
                cs.get(i).setValue(s, 0.0, False)
            cs.get(f"knee_angle_{side}").setValue(s, float(th), False)
            m.assemble(s)
            m.realizePosition(s)
            X = tibia.findTransformBetween(s, femur)            # tibia frame expressed in the femur frame
            R = np.array([[X.R().get(i, j) for j in range(3)] for i in range(3)])
            rv.append(Rotation.from_matrix(R).as_rotvec())
            p.append([X.p().get(k) for k in range(3)])
        p = np.array(p)
        out[f"rotvec_{side}"], out[f"p_{side}"] = np.array(rv), p / np.linalg.norm(p[0])
    np.savez(a.out, **out)
    print("wrote", a.out)


if __name__ == "__main__":
    main()

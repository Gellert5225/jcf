"""inference_posture_surrogate.urdf_pelvis_to_opensim: URDF pelvis rotation coordinates -> OpenSim angles.

Run: python -I tests/test_urdf_pelvis.py   (or pytest, where installed)
"""
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from inference_posture_surrogate import C_URDF, _rot, urdf_pelvis_to_opensim  # noqa: E402


def _opensim_R(e):
    return _rot(2, e[..., 0]) @ _rot(0, e[..., 1]) @ _rot(1, e[..., 2])     # Rz(tilt) Rx(list) Ry(rotation)


def test_same_orientation():
    rng = np.random.default_rng(0)
    a = rng.uniform(-0.6, 0.6, size=(500, 3))
    e = urdf_pelvis_to_opensim(a)
    Ru = _rot(2, a[:, 0]) @ _rot(0, a[:, 1]) @ _rot(1, a[:, 2])
    assert np.abs(_opensim_R(e) - C_URDF.T @ Ru @ C_URDF).max() < 1e-12


def test_first_order_and_identity():
    assert np.allclose(urdf_pelvis_to_opensim(np.zeros(3)), 0.0)
    for k, (i_out, sign) in enumerate(((2, 1.0), (1, 1.0), (0, -1.0))):   # tilt=-c, list=b, rotation=a
        a = np.zeros(3); a[k] = 1e-4
        out = urdf_pelvis_to_opensim(a)
        assert abs(out[i_out] - sign * 1e-4) < 1e-10 and np.abs(np.delete(out, i_out)).max() < 1e-10


def test_matches_scipy():
    from scipy.spatial.transform import Rotation
    rng = np.random.default_rng(1)
    a = rng.uniform(-0.6, 0.6, size=(200, 3))
    Ru = Rotation.from_euler("ZXY", a).as_matrix()                          # intrinsic z, x, y about URDF axes
    ref = Rotation.from_matrix(C_URDF.T @ Ru @ C_URDF).as_euler("ZXY")
    assert np.abs(urdf_pelvis_to_opensim(a) - ref).max() < 1e-10


def test_sep23_trajectory():
    """Agrees with the conversion the thesis audit used on the Sep 23 export (analysis/thesis/sep23_seeds.py)."""
    trial = os.path.join(REPO, "jcf", "external", "william_2026-09-23")
    if not os.path.isdir(trial):
        import pytest
        pytest.skip("Sep 23 trial folder not present")
    from analysis.thesis.sep23_seeds import ik_coords, urdf_pelvis_to_opensim as audit_convert, PELVIS_ROT
    _, coords = ik_coords(trial)
    ref, _ = audit_convert(coords)
    a = np.column_stack([coords[k] for k in PELVIS_ROT])
    out = urdf_pelvis_to_opensim(a)
    assert np.abs(out - np.column_stack([ref[k] for k in PELVIS_ROT])).max() < 1e-10


if __name__ == "__main__":
    for _n, _f in list(globals().items()):
        if _n.startswith("test_") and callable(_f):
            _f()
            print("ok", _n)

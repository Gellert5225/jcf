"""Tests for the numpy-only parts of the imu package. Run from the repo root:  python -m pytest tests -q"""
import os
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from imu import bvh, contact, opensense_io, segments  # noqa: E402

# Two frames. Frame 0: rest pose. Frame 1: root yawed +90 deg about Y, hip flexed 30 deg about X.
TINY = """HIERARCHY
ROOT Hips
{
  OFFSET 0 0 0
  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
  JOINT RightUpLeg
  {
    OFFSET -10 0 0
    CHANNELS 3 Zrotation Xrotation Yrotation
    JOINT RightLeg
    {
      OFFSET 0 -40 0
      CHANNELS 3 Zrotation Xrotation Yrotation
      End Site
      {
        OFFSET 0 -40 0
      }
    }
  }
}
MOTION
Frames: 2
Frame Time: 0.01
0 90 0 0 0 0   0 0 0   0 0 0
0 90 0 0 0 90  0 30 0  0 0 0
"""


def test_parse_structure():
    b = bvh.parse(TINY)
    assert b.order == ["Hips", "RightUpLeg", "RightLeg"]
    assert b.n_frames == 2 and b.fs == pytest.approx(100.0)
    assert b.columns == {"Hips": (0, 6), "RightUpLeg": (6, 9), "RightLeg": (9, 12)}
    assert b.joints["RightLeg"].parent == "RightUpLeg"
    np.testing.assert_allclose(b.joints["RightLeg"].end_site, [0, -40, 0])
    assert b.channel("RightUpLeg", "Xrotation")[1] == 30


def test_forward_kinematics_hand_checked():
    fk = bvh.forward_kinematics(bvh.parse(TINY))
    np.testing.assert_allclose(fk["Hips"][0], [[0, 90, 0], [0, 90, 0]], atol=1e-9)
    np.testing.assert_allclose(fk["RightUpLeg"][0][0], [-10, 90, 0], atol=1e-9)
    np.testing.assert_allclose(fk["RightLeg"][0][0], [-10, 50, 0], atol=1e-9)
    # yaw 90 about Y sends the hip offset (-10, 0, 0) to (0, 0, 10)
    np.testing.assert_allclose(fk["RightUpLeg"][0][1], [0, 90, 10], atol=1e-9)
    # thigh (0,-40,0) rotated by Rx(30) then Ry(90): (-40 sin30, -40 cos30, 0)
    np.testing.assert_allclose(fk["RightLeg"][0][1], [0 - 20, 90 - 40 * np.cos(np.radians(30)), 10], atol=1e-9)
    # scale applies to positions only
    fk2 = bvh.forward_kinematics(bvh.parse(TINY), scale=0.01, joints=["RightLeg"])
    assert list(fk2) == ["RightLeg"]
    np.testing.assert_allclose(fk2["RightLeg"][0][0], [-0.10, 0.50, 0], atol=1e-12)


def test_rotation_channels_are_intrinsic_in_listed_order():
    text = TINY.replace("0 90 0 0 0 0   0 0 0   0 0 0", "0 90 0 90 90 0   0 0 0   0 0 0")   # Z=90 then X=90
    r = bvh.local_rotations(bvh.parse(text), "Hips")[0].as_matrix()
    expected = Rotation.from_euler("z", 90, degrees=True).as_matrix() @ Rotation.from_euler("x", 90, degrees=True).as_matrix()
    np.testing.assert_allclose(r, expected, atol=1e-12)


def test_parse_errors():
    with pytest.raises(ValueError):
        bvh.parse("MOTION\nFrames: 0\n")
    with pytest.raises(ValueError):
        bvh.parse(TINY.replace("Frames: 2", "Frames: 3"))           # too few values


def test_segment_mapping():
    joints = ["Hips", "RightUpLeg", "RightLeg", "RightFoot", "LeftUpLeg", "LeftLeg", "LeftFoot", "Spine", "Spine1", "Spine2"]
    m = segments.resolve(joints, segments.LOWER_BODY)
    assert m["torso_imu"] == "Spine2" and m["tibia_r_imu"] == "RightLeg"
    assert segments.resolve(["Hips", "Spine"], ["pelvis_imu", "torso_imu"])["torso_imu"] == "Spine"
    with pytest.raises(KeyError):
        segments.resolve(["Hips"], ["pelvis_imu", "femur_r_imu"])
    assert segments.body_of("calcn_l_imu") == "calcn_l"
    with pytest.raises(ValueError):
        segments.body_of("pelvis")


def test_orientation_file_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    rots = Rotation.random(50, random_state=1)
    q = opensense_io.from_scipy(rots)
    assert np.all(q[:, 0] >= 0)
    np.testing.assert_allclose(opensense_io.to_scipy(q).as_matrix(), rots.as_matrix(), atol=1e-12)
    t = np.arange(50) / 100
    path = tmp_path / "o.sto"
    opensense_io.write_orientations(path, t, {"pelvis_imu": q, "femur_r_imu": q[::-1].copy()})
    t2, q2, rate = opensense_io.read_orientations(path)
    assert rate == pytest.approx(100.0) and list(q2) == ["pelvis_imu", "femur_r_imu"]
    np.testing.assert_allclose(t2, t, atol=1e-6)
    np.testing.assert_allclose(q2["pelvis_imu"], q, atol=1e-8)
    with pytest.raises(ValueError):
        opensense_io.write_orientations(path, t, {"pelvis_imu": q[:10]})


def test_bvh_to_orientations():
    b = bvh.parse(TINY)
    t, q = opensense_io.bvh_to_orientations(b, {"pelvis_imu": "Hips", "tibia_r_imu": "RightLeg"})
    np.testing.assert_allclose(t, [0, 0.01])
    np.testing.assert_allclose(q["pelvis_imu"][0], [1, 0, 0, 0], atol=1e-12)
    expected = Rotation.from_euler("y", 90, degrees=True) * Rotation.from_euler("x", 30, degrees=True)
    np.testing.assert_allclose(opensense_io.to_scipy(q["tibia_r_imu"][1]).as_matrix()[0], expected.as_matrix(), atol=1e-12)


def _synthetic_gait(fs=100, steps=6, stance=0.6, swing=0.4):
    """Heel and toe of one foot: on the ground and still during stance, lifted and moving during swing."""
    heel, toe, truth = [], [], []
    x = 0.0
    for _ in range(steps):
        ns, nw = int(stance * fs), int(swing * fs)
        heel += [[x, 0.02, 0]] * ns; toe += [[x + 0.2, 0.02, 0]] * ns; truth += [True] * ns
        s = np.linspace(0, 1, nw, endpoint=False)
        lift = 0.12 * np.sin(np.pi * s)
        heel += [[x + 0.8 * si, 0.02 + li, 0] for si, li in zip(s, lift)]
        toe += [[x + 0.2 + 0.8 * si, 0.02 + li, 0] for si, li in zip(s, lift)]
        truth += [False] * nw
        x += 0.8
    return np.array(heel), np.array(toe), np.array(truth)


def test_contact_detection_on_synthetic_gait():
    heel, toe, truth = _synthetic_gait()
    pred = contact.detect_contact([heel, toe], fs=100)
    m = contact.compare(pred, truth, fs=100)
    assert m["frame_agreement"] > 0.9 and m["recall"] > 0.9 and m["precision"] > 0.9
    assert m["onset_matched_frac"] == 1.0 and m["onset_error_abs_mean_s"] < 0.05


def test_clean_runs():
    m = np.array([1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0], bool)
    out = contact.clean_runs(m, min_on=2, min_off=2)
    np.testing.assert_array_equal(out, [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0])


def test_orientation_contact_detector():
    fs, truth, angles = 100, [], []
    a = 0.0
    for _ in range(5):                         # stance: foot still; swing: foot pitches 60 deg and back
        truth += [True] * 60; angles += [a] * 60
        sw = 60 * np.sin(np.linspace(0, np.pi, 40, endpoint=False)); truth += [False] * 40; angles += list(sw)
    rot = Rotation.from_euler("z", np.array(angles)[:, None], degrees=True)
    w = contact.angular_speed(rot, fs)
    assert w[:55].max() < 1e-9 and np.degrees(w).max() > 200
    pred = contact.detect_contact_orientation(rot, fs)
    m = contact.compare(pred, np.array(truth), fs)
    assert m["frame_agreement"] > 0.9 and m["onset_matched_frac"] == 1.0


def test_band_limited_noise():
    from imu.synthetic import band_limited_noise
    rng = np.random.default_rng(0)
    for corr in (0.0, 0.1):
        e = np.degrees(band_limited_noise(20000, 0.01, 2.0, corr, rng).as_rotvec())
        assert np.std(e, axis=0) == pytest.approx([2.0] * 3, rel=0.1)
        lag1 = np.corrcoef(e[:-1, 0], e[1:, 0])[0, 1]
        assert (abs(lag1) < 0.05) if corr == 0 else (lag1 == pytest.approx(np.exp(-0.1), abs=0.02))


def test_from_bvh_cli(tmp_path):
    from imu import from_bvh
    src = tmp_path / "t.bvh"
    src.write_text(TINY)
    out = tmp_path / "o" / "orientations.sto"
    from_bvh.main(["--bvh", str(src), "--out", str(out), "--start", "0.01", "--labels", "pelvis_imu", "femur_r_imu"])
    t, q, rate = opensense_io.read_orientations(out)
    assert rate == pytest.approx(100.0) and list(q) == ["pelvis_imu", "femur_r_imu"]
    np.testing.assert_allclose(t, [0.0])                    # starts at the requested calibration frame
    expected = Rotation.from_euler("y", 90, degrees=True) * Rotation.from_euler("x", 30, degrees=True)
    np.testing.assert_allclose(opensense_io.to_scipy(q["femur_r_imu"][0]).as_matrix()[0], expected.as_matrix(), atol=1e-9)

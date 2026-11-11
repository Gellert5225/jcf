"""Tests for the Twente real-IMU check (analysis/thesis/twente_imu.py). Run: python -m pytest tests -q

The synthetic tests need no data and no torch. The data tests read a few seconds of one trial and skip when the
Twente folder is missing."""
import json
import os
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from analysis.thesis import twente_imu as tw          # noqa: E402
from pn3_redis.model import COORDS, LowerBody          # noqa: E402

HAVE_DATA = os.path.isdir(os.path.join(tw.DATA_DEFAULT, "Subj04", "Subj04"))
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="Twente dataset not available")
LABELS = list(tw.MVN_SEGMENTS.values())


def _quat_text(qs):
    return " ".join(f"{v:.6f}" for q in qs for v in q)


def _write_mvnx(path, orient, sensor, name="Alice Example"):
    """Minimal MVNX 4 file: 8 segments / sensors, identity + tpose special frames, then normal frames."""
    seg = "".join(f'<segment label="{lab}" id="{i + 1}"><points><point label="p{i}"><pos_b>0 0 0</pos_b></point>'
                  f'</points></segment>' for i, lab in enumerate(LABELS))
    sen = "".join(f'<sensor label="{lab}"/>' for lab in LABELS)
    ident = _quat_text([(1, 0, 0, 0)] * 8)
    frames = (f'<frame time="0" type="identity"><orientation>{ident}</orientation></frame>'
              f'<frame time="0" type="tpose"><orientation>{ident}</orientation></frame>')
    for i, (o, s) in enumerate(zip(orient, sensor)):
        frames += (f'<frame time="{round(i / 0.24)}" index="{i}" tc="x" ms="0" type="normal">'
                   f'<orientation>{_quat_text(o)}</orientation><sensorOrientation>{_quat_text(s)}</sensorOrientation>'
                   f'</frame>')
    xml = ('<?xml version="1.0" encoding="UTF-8"?>\n<mvnx xmlns="http://www.xsens.com/mvn/mvnx" version="4">'
           '<mvn version="2021.0.0" build="b"/><comment></comment>'
           f'<subject label="{name}" frameRate="240" segmentCount="8" recDate="Mon" originalFilename="C:/{name}/w.mvnx" '
           'configuration="LowerBodyPlus" userScenario="singleLevel" processingQuality="Live">'
           f'<segments>{seg}</segments><sensors>{sen}</sensors>'
           '<joints><joint label="jL5S1"><connector1>Pelvis/jL5S1</connector1><connector2>T8/jL5S1</connector2>'
           f'</joint></joints><frames segmentCount="8">{frames}</frames></subject></mvnx>')
    with open(path, "w") as f:
        f.write(xml)


def test_read_mvnx_quaternion_order_special_frames_dropouts_and_privacy(tmp_path):
    th = np.radians(30.0)
    zq = (np.cos(th / 2), 0.0, 0.0, np.sin(th / 2))                # w, x, y, z: 30 deg about MVN z (up)
    orient = [[(1, 0, 0, 0)] * 8, [zq] + [(1, 0, 0, 0)] * 7, [zq] * 8]
    sensor = [[(1, 0, 0, 0)] * 8, [zq] * 8, [zq] * 4 + [(0, 0, 0, 0)] + [zq] * 3]   # RightFoot drops out at frame 2
    p = str(tmp_path / "t.mvnx")
    _write_mvnx(p, orient, sensor)
    d = tw.read_mvnx(p, ("orientation", "sensorOrientation"))
    assert list(d["index"]) == [0, 1, 2]                               # special frames are not data frames
    assert set(d["special"]) == {"identity", "tpose"}
    assert d["orientation"].shape == (3, 32) and d["segments"] == LABELS and d["sensors"] == LABELS
    assert d["joints"] == ["jL5S1"]
    assert set(d["meta"]) <= set(tw.MVNX_META) | {"mvn_version"}       # whitelist only
    assert "Alice" not in json.dumps(d["meta"]) and d["meta"]["frameRate"] == "240"
    rots, bad = tw.mvn_rotations(d, "orientation")
    np.testing.assert_allclose(rots["pelvis_imu"][1].apply([1.0, 0, 0]), [np.cos(th), np.sin(th), 0], atol=1e-5)
    np.testing.assert_allclose(np.degrees(tw.mvn_heading(rots["pelvis_imu"])), [0, 30, 30], atol=1e-3)
    assert not any(b.any() for b in bad.values())
    rs, bs = tw.mvn_rotations(d, "sensorOrientation")
    assert list(bs["calcn_r_imu"]) == [False, False, True] and not bs["calcn_l_imu"].any()
    np.testing.assert_allclose(rs["calcn_r_imu"][2].as_quat(), rs["calcn_r_imu"][1].as_quat(), atol=1e-9)  # held
    d2 = tw.read_mvnx(p, ("orientation",), max_frames=2)
    assert len(d2["index"]) == 2 and d2["meta"]["processingQuality"] == "Live"


def test_mvn_to_opensim_axes():
    M = tw.M_MVN
    np.testing.assert_allclose(M @ M.T, np.eye(3))
    assert np.isclose(np.linalg.det(M), 1.0)                           # a rotation, not a reflection
    np.testing.assert_allclose(M @ [1, 0, 0], [1, 0, 0])               # forward stays forward
    np.testing.assert_allclose(M @ [0, 0, 1], [0, 1, 0])               # MVN up (z) -> OpenSim up (y)
    np.testing.assert_allclose(M @ [0, 1, 0], [0, 0, -1])              # MVN left (y) -> OpenSim -z (z is right)
    # a turn about MVN's vertical is the same turn about OpenSim's vertical, the sense pelvis_rotation uses
    Rz = Rotation.from_euler("z", 0.4).as_matrix()
    np.testing.assert_allclose(M @ Rz @ M.T, Rotation.from_euler("y", 0.4).as_matrix(), atol=1e-12)


def test_gait2392_to_rajagopal_conversion():
    names = ["time"] + COORDS + ["subtalar_angle_r"]
    vals = np.tile(np.arange(len(names), dtype=float), (4, 1))
    q = tw.to_rajagopal(dict(names=names, data=vals, in_degrees=True))
    for c in COORDS:
        v = float(names.index(c))
        if c in ("pelvis_tx", "pelvis_ty", "pelvis_tz"):
            assert q[0, COORDS.index(c)] == v                         # metres untouched (pelvis_tilt is NOT a translation)
        elif c.startswith("knee_angle"):
            assert np.isclose(q[0, COORDS.index(c)], -np.radians(v))  # gait2392 flexion is negative
        else:
            assert np.isclose(q[0, COORDS.index(c)], np.radians(v))
    with pytest.raises(ValueError):
        tw.to_rajagopal(dict(names=names, data=vals, in_degrees=False))


def test_vendor_heading_removed_wraps():
    q = np.zeros((2, len(COORDS)))
    j = COORDS.index("pelvis_rotation")
    q[:, j] = np.radians([185.0, -175.0])
    out = tw.vendor_heading_removed(q, np.radians(-175.0))
    np.testing.assert_allclose(np.degrees(out[:, j]), [0.0, 0.0], atol=1e-9)
    assert np.all(out[:, :j] == q[:, :j])


def test_standing_window_and_calibration_frames():
    fs = 60.0
    t = np.arange(240) / fs
    ang = np.where(t < 1.5, 0.0, np.radians(100.0) * (t - 1.5))      # still for 1.5 s, then 100 deg/s
    rots = {k: Rotation.from_euler("z", (ang + 0.3)[:, None]) for k in tw.MVN_SEGMENTS}
    assert tw.standing_window(t, rots, dur_s=1.0) == (0, 60)
    assert tw.standing_window(t, rots, dur_s=1.0, t_from=1.0) is None
    moving = {k: Rotation.from_euler("z", (np.radians(100.0) * t)[:, None]) for k in tw.MVN_SEGMENTS}
    assert tw.standing_window(t, moving) is None
    cal = tw.calibration_frames("mvnmodel", rots, (0, 60))
    assert len(cal["tibia_r_imu"]) == 60
    np.testing.assert_allclose(tw.mvn_heading(cal["femur_l_imu"]), 0.3, atol=1e-9)
    st = tw.calibration_frames("static", rots, (5, 20))
    assert len(st["pelvis_imu"]) == 15
    with pytest.raises(ValueError):
        tw.calibration_frames("tpose", rots, (0, 60))


def test_event_and_stride_metrics():
    k = np.arange(1000)
    t = k / 100.0
    ref = (k % 100) < 60                                               # stance 0.6 s every second
    late = ((k - 3) % 100) < 60                                        # the same, 30 ms late
    on_r, off_r = tw.transitions(t, ref)
    on_t, _ = tw.transitions(t, late)
    np.testing.assert_allclose(on_r, np.arange(1, 10), atol=1e-9)
    np.testing.assert_allclose(off_r, np.arange(0, 10) + 0.6, atol=1e-9)
    np.testing.assert_allclose(tw.event_errors(on_r, on_t), 0.03, atol=1e-9)
    assert np.isnan(tw.event_errors([5.0], [5.5])[0])                 # nothing within 0.25 s
    iv = tw.stance_intervals(t, ref, 0.5, 9.0)
    assert len(iv) == 8 and np.allclose(iv[0], (1.0, 1.6))
    f = np.where(ref, np.sin(np.pi * (k % 100) / 60) + 0.5 * ((k % 100) > 30), 0.0)
    pk = tw.stride_peaks(t, f, iv)
    assert pk.shape == (8, 2) and np.all(pk[:, 1] > pk[:, 0])
    s = tw.fmed_stats(f + 0.1, f, ref, pk + 0.1, pk)
    assert np.isclose(s["mae"], 0.1) and np.isclose(s["bias"], 0.1) and np.isclose(s["peak1_bias"], 0.1)
    assert s["n_strides"] == 8 and np.isclose(s["r"], 1.0)


def test_angle_stats_wraps_rotation():
    s = tw.angle_stats([179.0, 178.0, -179.0], [-179.0, -180.0, 179.0], wrap=True)   # errors -2, -2, +2 deg
    np.testing.assert_allclose([s["rmse"], s["bias"]], [2.0, -2.0 / 3.0])
    assert np.isclose(tw.xcorr_lag(np.arange(0, 5, 0.01), np.sin(np.arange(0, 5, 0.01) * 3),
                                   np.arange(-1, 6, 0.004), np.sin((np.arange(-1, 6, 0.004) - 0.02) * 3))[0], 0.02)


def test_scored_angles_include_lumbar():
    assert set(tw.C13) < set(tw.SCORED) and set(tw.LUMBAR) <= set(tw.SCORED)
    assert all(c in COORDS for c in tw.SCORED) and set(tw.DEF_COORDS) == {"pelvis_tilt", *tw.LUMBAR}


def test_swap_shift_and_definition_paths():
    rng = np.random.default_rng(0)
    q_o, q_v = rng.normal(size=(50, len(COORDS))), rng.normal(size=(50, len(COORDS)))
    jt, jl = COORDS.index("pelvis_tilt"), [COORDS.index(c) for c in tw.LUMBAR]
    s = tw.swap_coords(q_o, q_v, ["pelvis_tilt"])
    assert np.array_equal(s[:, jt], q_v[:, jt]) and np.array_equal(np.delete(s, jt, 1), np.delete(q_o, jt, 1))
    sh = tw.shift_coords(q_o, {"lumbar_extension": 0.2})
    np.testing.assert_allclose(sh[:, jl[0]] - q_o[:, jl[0]], 0.2)
    assert np.array_equal(np.delete(sh, jl[0], 1), np.delete(q_o, jl[0], 1))
    ok = np.arange(50) >= 10
    p = tw.definition_paths(q_o, q_v, ok)
    assert set(p) == {"O/V:lumbar", "O/V:tilt", "O/V:lumbar+tilt", "O+Vbias:lumbar+tilt"}
    assert np.array_equal(p["O/V:lumbar+tilt"][:, [jt] + jl], q_v[:, [jt] + jl])
    assert np.array_equal(p["O/V:lumbar"][:, jt], q_o[:, jt])
    d = p["O+Vbias:lumbar+tilt"] - q_o                                 # constant: the mean V - O over ok only
    np.testing.assert_allclose(d[:, jt], np.mean(q_v[ok, jt] - q_o[ok, jt]))
    assert np.allclose(np.delete(d, [jt] + jl, 1), 0.0)


def _angle_rows(subj, trial, test, bias_deg, coords=None, ref="O"):
    return [dict(subject=subj, trial=trial, speed=tw.WALK_SPEED_MPS[trial], test=test, ref=ref, coord=c, bias=bias_deg)
            for c in (coords or tw.DEF_COORDS)]


def test_loso_offsets_never_use_the_scored_trial():
    rows = (_angle_rows("01", "walk_09", "V", 1.0) + _angle_rows("01", "walk_18", "V", 2.0)
            + _angle_rows("01", "walk_27", "V", 6.0) + _angle_rows("01", "walk_27", "V", 50.0, ref="V")
            + _angle_rows("01", "walk_09", "U:seg-static", -3.0) + _angle_rows("01", "walk_18", "U:seg-static", -5.0)
            + _angle_rows("02", "walk_09", "V", 9.0)                        # one trial: nothing to estimate from
            + _angle_rows("03", "walk_09", "V", 1.0) + _angle_rows("03", "walk_18", "V", 2.0, coords=["pelvis_tilt"]))
    off = tw.loso_offsets(rows)
    deg = {k: {c: np.degrees(v) for c, v in d.items()} for k, d in off.items()}
    np.testing.assert_allclose(deg[("01", "walk_09", "V")]["lumbar_extension"], 4.0)      # mean of 2 and 6
    np.testing.assert_allclose(deg[("01", "walk_27", "V")]["pelvis_tilt"], 1.5)           # ref V rows ignored
    np.testing.assert_allclose(deg[("01", "walk_09", "U:seg-static")]["lumbar_rotation"], -5.0)
    assert ("02", "walk_09", "V") not in off
    assert ("03", "walk_18", "V") in off and ("03", "walk_09", "V") not in off   # lumbar missing on walk_18


def test_match_jobs_pair_series_with_their_path_offsets():
    t = np.arange(10) / 100.0

    def res(trial, variant, first, bias_u, bias_v):
        ser = {f"U:{variant}:grf": t * 0, f"U:{variant}:gyro": t * 0}
        if first:
            ser["V"] = t * 0
        rows = _angle_rows("01", trial, f"U:{variant}", bias_u) + (_angle_rows("01", trial, "V", bias_v) if first else [])
        lab0 = dict(subject="01", trial=trial, speed=tw.WALK_SPEED_MPS[trial], quality="Live")
        return dict(angles=rows, match=dict(labels=lab0, series={"A": dict(t=t, stance=t > 0, intervals=[],
                                                                            series=ser)}))
    results = [res(tr, v, i == 0, bu, bv) for tr, bu, bv in (("walk_09", 1.0, 10.0), ("walk_18", 3.0, 12.0))
               for i, v in enumerate(("seg-static", "seg-mvnmodel"))] + [dict(job="x", error="boom")]
    jobs = tw.match_jobs(results, {})
    assert [(j["subj"], j["trial"]) for j in jobs] == [("01", "walk_09"), ("01", "walk_18")]
    j = jobs[0]
    assert set(j["offsets"]) == {"V", "U:seg-static", "U:seg-mvnmodel"}
    np.testing.assert_allclose(np.degrees(j["offsets"]["V"]["lumbar_extension"]), 12.0)   # from walk_18 only
    np.testing.assert_allclose(np.degrees(j["offsets"]["U:seg-static"]["pelvis_tilt"]), 3.0)
    assert set(j["series"]["A"]["series"]) == {"V", "U:seg-static:grf", "U:seg-static:gyro",
                                               "U:seg-mvnmodel:grf", "U:seg-mvnmodel:gyro"}


def test_flat_table_and_decomposition_spec():
    import pandas as pd
    df = pd.DataFrame(dict(subject=["01", "01", "02", "02"], speed=[0.25, 0.5, 0.25, 0.5], test="V", ref="O",
                           mae=[1.0, 3.0, 5.0, 7.0]))
    ft = tw.flat_table(df, ["test", "ref"], ["mae"]).set_index("speed")
    assert ft.loc["all", "mae_median"] == 4.0 and ft.loc["all", "n_subjects"] == 2      # subject means 2 and 6
    assert ft.loc["0.25", "mae_median"] == 3.0 and ft.loc["0.50", "mae_q75"] == 6.0
    spec = tw.decomposition_spec(["seg-static"])
    pairs = {(t_, r) for _, t_, r in spec}
    assert ("O/V:lumbar+tilt", "O") in pairs and ("V", "Omatch:V") in pairs
    assert ("U:seg-static:grf", "Omatch:U:seg-static") in pairs and ("U:seg-static:gyro", "O") in pairs


def _mvn_stream(lb, q, psi):
    """MVN-style segment orientations for a Rajagopal trajectory: MVN identity = q0, heading psi about MVN up."""
    F = lb.fk(q)
    M = tw.M_MVN
    rz = Rotation.from_euler("z", psi).as_matrix()
    cols = []
    for key, lab in tw.MVN_SEGMENTS.items():
        body = key.replace("_imu", "")
        R = rz @ M.T @ F[body][0] @ M
        cols.append(Rotation.from_matrix(R).as_quat()[:, [3, 0, 1, 2]])
    return np.hstack(cols)


@pytest.mark.parametrize("calib", ["mvnmodel", "static"])
def test_ukf_recovers_angles_from_mvn_format_orientations(calib):
    """End to end on synthetic MVN data: frame map, both calibrations, the driver loop and the UKF."""
    lb = LowerBody(height_m=1.75)
    q0 = lb.default_q()
    t = np.arange(0, 3.0, 1 / 240.0)
    q = np.tile(q0, (len(t), 1))
    for c, amp, ph in (("hip_flexion_r", 25, 0.0), ("knee_angle_r", 30, 0.8), ("ankle_angle_r", 10, 1.6),
                       ("hip_flexion_l", 25, np.pi), ("knee_angle_l", 30, np.pi + 0.8), ("hip_adduction_r", 5, 0.3)):
        q[:, COORDS.index(c)] += np.radians(amp) * (0.5 + 0.5 * np.sin(2 * np.pi * t + ph))
    psi = np.radians(150.0)
    static = np.tile(q0, (480, 1))
    ts = np.arange(480) / 240.0
    tr = dict(subj="xx", height=1.75, t_mvn=t, t_ik=t, flags={"grf": {"r": np.zeros(len(t), bool), "l": np.zeros(len(t), bool)}},
              mvn=dict(segments=LABELS, orientation=_mvn_stream(lb, q, psi)),
              static=dict(segments=LABELS, orientation=_mvn_stream(lb, static, psi)), static_t=ts)
    u = tw.run_ukf(tr, f"seg-{calib}", {}, sources=("grf",), fs=60.0, anchor="grf")
    assert u["cal_src"] == "static_pose" and np.isclose(u["heading_deg"], 150.0, atol=0.1)
    ok = u["t"] > 1.0
    for c in ("hip_flexion_r", "knee_angle_r", "ankle_angle_r", "knee_angle_l", "hip_adduction_r", "pelvis_tilt"):
        j = COORDS.index(c)
        ref = np.interp(u["t"][ok], t, q[:, j])
        rmse = np.degrees(np.sqrt(np.mean((u["q"][ok, j] - ref) ** 2)))
        assert rmse < 1.5, (c, rmse)


# ─── Real data (skipped without the dataset) ──────────────────────────────────────────────────────────────────────

@needs_data
def test_twente_trial_sync_contact_and_privacy():
    tr = tw.load_trial(tw.DATA_DEFAULT, "04", "walk_27", max_s=20.0)
    assert np.isclose(tr["mass"], 75.2) and np.isclose(tr["height"], 1.80) and tr["quality"] == "HD"
    assert tr["lag_src"] == "DeltaT" and 0.5 < tr["L"] < 1.0
    assert abs(tr["sync_lag_ms"]) <= 15 and tr["sync_r"] > 0.99      # MVN knee vs optical knee after the clock rule
    for s in "rl":
        assert 0.55 < tr["flags"]["grf"][s].mean() < 0.85
    assert set(tr["mvn"]["meta"]) <= set(tw.MVNX_META) | {"mvn_version"}
    knee = np.degrees(tr["q_o"][:, COORDS.index("knee_angle_r")])
    assert np.percentile(knee, 95) > 40 and np.percentile(knee, 5) > -10   # flexion positive after the sign flip


@needs_data
def test_twente_trunk_definition_offset():
    """IK_IMU (MVN) vs optical gait2392 IK: a constant lumbar_extension offset of about +9..+13 deg and a negative
    pelvis tilt offset. Not a sign error (the two still correlate), and far from 'a few degrees'."""
    tr = tw.load_opensim(tw.DATA_DEFAULT, "05", "walk_36", max_s=30.0)
    assert set(tr) >= {"t_ik", "q_o", "q_v_raw", "flags", "mass", "height"}
    o, v = (np.degrees(tr[k]) for k in ("q_o", "q_v_raw"))
    j, jt = COORDS.index("lumbar_extension"), COORDS.index("pelvis_tilt")
    off = np.mean(v[:, j] - o[:, j])
    assert 5.0 < off < 20.0 and np.corrcoef(v[:, j], o[:, j])[0, 1] > 0, off
    assert np.std(v[:, j] - o[:, j]) < 0.5 * abs(off)                  # mostly constant
    assert -6.0 < np.mean(v[:, jt] - o[:, jt]) < -1.0


@needs_data
def test_twente_ukf_tracks_optical_knee():
    tr = tw.load_trial(tw.DATA_DEFAULT, "05", "walk_36", max_s=12.0)
    u = tw.run_ukf(tr, "seg-mvnmodel", {}, sources=("grf",), fs=60.0)
    ok = (u["t"] > tr["t_mvn"][0] + tw.SCORE_AFTER_S) & (u["t"] < tr["t_ik"][-1])
    j = COORDS.index("knee_angle_r")
    ref = np.interp(u["t"][ok], tr["t_ik"], tr["q_o"][:, j])
    s = tw.angle_stats(np.degrees(u["q"][ok, j]), np.degrees(ref))
    assert s["rmse"] < 5.0 and s["r"] > 0.98, s

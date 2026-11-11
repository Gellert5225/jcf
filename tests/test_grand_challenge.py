"""Grand Challenge in vivo validation (analysis/thesis/grand_challenge.py, gc_knee_jr.py).

Covers the parts that decide whether a number is right: implant-file parsing and units, subject detection, time
alignment, the left-knee mirroring (derived from the kinematics and checked on a synthetic symmetric gait), the
label-free feature builder against the training builder, stance peak windows and the interior-maximum rule, the
model-free DM timing estimate and the measured-force shift, and the JR-only OpenSim setup with SO forces applied.
Tests that need the Grand Challenge package or the fregly segment folders skip when those are missing.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from analysis.thesis import grand_challenge as gc  # noqa: E402
from analysis.thesis import gc_knee_jr  # noqa: E402
from analysis.thesis.gc_knee_jr import jr_only_setup  # noqa: E402
from train_mlp import INPUT_SETS, _build_com_features  # noqa: E402

WALK = os.path.join(REPO, "jcf", "full_duration")
SEG_SC = os.path.join(WALK, "testing", "walking", "fregly_3GC_t18_r00")
SEG_JW = os.path.join(WALK, "training", "walking", "fregly_1GC_t00_r00")
NOTRANS = "qinst_notrans_pelvisrel_winpos_contact_w20"
FPOPT = "qinst_notrans_pelvisrel_winpos_contact_fpopt_w20"
Q16 = "qinst_pelvisrel_winpos_contact_w20"


def _gc_ok():
    try:
        return set(gc.find_subject_dirs()) == {"JW", "DM", "SC"}
    except Exception:
        return False


needs_gc = pytest.mark.skipif(not _gc_ok(), reason="needs jcf/external/grand_challenge")


# ─── Names, files, units ──────────────────────────────────────────────────────

def test_trial_key_and_gait_code():
    assert gc.trial_key("dm_ngait4_knee_forces.csv") == gc.trial_key("DM_ngait_4") == "dmngait4"
    assert gc.trial_key("/x/SC_ngait_og5_trajectories.csv") == "scngaitog5"
    assert gc.gait_code("SC_ngait_og5") == "ngait"
    assert gc.gait_code("jw_wpgait_10") == "wpgait"
    assert gc.gait_code("SC_wpgait_l11") == "wpgaitl"
    assert gc.gait_code("SC_medialthrust3") == "medialthrust"
    assert gc.gait_code("DM_mtgait_10") == "mtgait"


def test_read_gc_csv_trailing_comma(tmp_path):
    p = tmp_path / "x_knee_forces.csv"
    p.write_text("Time(sec),Fx,Fz,Ty\n0.0,1.0,2.0,3.0,\n0.5,4.0,5.0,6.0,\n")
    df = gc.read_gc_csv(p)
    assert list(df.columns) == ["Time(sec)", "Fx", "Fz", "Ty"]
    np.testing.assert_allclose(df["Fx"], [1.0, 4.0])          # no column shift from the trailing comma
    np.testing.assert_allclose(gc.time_column(df), [0.0, 0.5])
    j = tmp_path / "jw.csv"                                     # eKnee header: time column named '17:00'
    j.write_text("17:00,Sensor1\n0,1\n0.005,2\n")
    np.testing.assert_allclose(gc.time_column(gc.read_gc_csv(j)), [0.0, 0.005])


def test_measured_forces_units_and_regressions():
    lbf, inlbf = gc.LBF_N, gc.INLBF_NMM
    assert lbf == pytest.approx(4.448222, abs=1e-6) and inlbf == pytest.approx(112.98483, abs=1e-5)
    sc = gc.measured_forces("SC", pd.DataFrame({"Time(sec)": [0.0, 1.0], "Fx": [0.0, 10.0], "Fz": [100.0, 0.0],
                                                 "Ty": [0.0, 1.0]}))
    np.testing.assert_allclose(sc.total_N, [100 * lbf, 0.0])
    np.testing.assert_allclose(sc.medial_N, [0.497 * 100 * lbf, 0.942 * 10 * lbf + 0.0184 * inlbf])
    np.testing.assert_allclose(sc.lateral_N, [0.503 * 100 * lbf, -0.942 * 10 * lbf - 0.0184 * inlbf])
    np.testing.assert_allclose(sc.medial_N + sc.lateral_N, sc.total_N, atol=1e-9)   # the SC pair sums to Fz
    dm = gc.measured_forces("DM", pd.DataFrame({"time(sec)": [0.0], "Sensor1": [10.0], "Sensor2": [99.0],
                                                 "Sensor3": [100.0], "Sensor4": [99.0], "Sensor5": [2.0],
                                                 "Sensor6": [99.0]}))
    assert dm.total_N[0] == pytest.approx(100 * lbf)
    assert dm.medial_N[0] == pytest.approx(-0.876 * 10 * lbf + 0.494 * 100 * lbf - 0.0189 * 2 * inlbf)
    assert dm.lateral_N[0] == pytest.approx(0.876 * 10 * lbf + 0.505 * 100 * lbf + 0.0189 * 2 * inlbf)
    jw = gc.measured_forces("JW", pd.DataFrame({"17:00": [0.0], "Sensor1": [1.0], "Sensor2": [2.0],
                                                 "Sensor3": [3.0], "Sensor4": [4.0]}))
    assert jw.medial_N[0] == pytest.approx(3 * lbf) and jw.lateral_N[0] == pytest.approx(7 * lbf)
    assert jw.total_N[0] == pytest.approx(10 * lbf)
    with pytest.raises(ValueError):
        gc.measured_forces("XX", sc)


def test_find_subject_dirs_uses_file_prefixes_not_folder_names(tmp_path):
    for folder, sub, name in (("GC1", "eTibia Data", "SC_ngait_og5"), ("GC2", "eTibia Data", "dm_ngait4"),
                              ("GC3", "EKnee Data", "jw_ngait_2")):
        d = tmp_path / folder / "Gait Trials" / sub
        d.mkdir(parents=True)
        (d / f"{name}_knee_forces.csv").write_text("t\n0\n")
    found = gc.find_subject_dirs(str(tmp_path))
    assert {k: os.path.basename(v) for k, v in found.items()} == {"SC": "GC1", "DM": "GC2", "JW": "GC3"}
    mixed = tmp_path / "GC2" / "Gait Trials" / "eTibia Data" / "jw_x_knee_forces.csv"
    mixed.write_text("t\n0\n")
    gc.find_subject_dirs.cache_clear()
    with pytest.raises(ValueError):
        gc.find_subject_dirs(str(tmp_path))
    gc.find_subject_dirs.cache_clear()


# ─── Alignment ────────────────────────────────────────────────────────────────

def test_sliding_corr_and_grf_alignment_on_a_synthetic_trial():
    rng = np.random.default_rng(0)
    t = np.arange(800) / 200.0
    walk = lambda ph: np.clip(800 * np.sin(2 * np.pi * (t / 1.1 + ph)), 0, None) + 5 * rng.standard_normal(len(t))
    trials = {"a": walk(0.0), "b": walk(0.37), "c": walk(0.71)}
    seg = trials["b"][237:337].copy()
    r = gc.sliding_corr(seg, trials["b"])
    assert int(np.nanargmax(r)) == 237 and r[237] == pytest.approx(1.0)
    a = gc.align_by_grf(seg, trials)
    assert a["key"] == "b" and a["grf_row0"] == 237
    assert a["ratio"] == pytest.approx(1.0) and a["rmse_N"] < 1e-9
    assert a["sharpness"] > 5                     # other steps of the same trial correlate, but far less exactly
    assert np.isnan(gc.align_by_grf(np.zeros(50), trials)["grf_row0"])


def test_segment_time_maps_through_the_frame_index():
    # AddB writes segment time as frame * dt_b3d (0.2-0.3 % short); row = row0 + frame, whatever the grid
    dt_b3d = 0.0083093
    rows = gc.seg_time_to_gc_row(np.array([0.0, 64 * dt_b3d, 0.25]), 113, dt_b3d)
    np.testing.assert_allclose(rows, [113, 177, 113 + 0.25 / dt_b3d])
    t_gc = 2.59 + np.arange(400) / 120.0
    assert gc.gc_row_to_time(113.0, t_gc) == pytest.approx(2.59 + 113 / 120.0)


@needs_gc
@pytest.mark.skipif(not os.path.exists(os.path.join(SEG_SC, "grf_data.mot")), reason="needs fregly_3GC_t18_r00")
def test_sc_segment_aligns_to_its_trial_with_the_known_grf_delay():
    keys = ["scngaitog5", "scngaitog6", "scbouncyog3"]
    trials = {k: gc.load_gc_trial("SC", k)["vgrf"] for k in keys}
    seg = gc.read_segment(SEG_SC)
    a = gc.align_by_grf(seg["vy"]["r"] + seg["vy"]["l"], trials)
    assert a["key"] == "scngaitog5" and a["r_best"] > 0.998
    # b3d frames 112-176 are GC marker rows 113-177; AddB's SC GRF sits one frame earlier on the GC clock
    assert a["grf_row0"] + gc.SUBJECTS["SC"]["grf_delay_frames"] == 113
    _, sm = gc.read_maps()
    if sm is not None:
        r = sm.set_index("folder").loc["fregly_3GC_t18_r00"]
        assert (int(r.trial), int(r.start_frame), int(r.gc_row_start), int(r.gc_row_end)) == (18, 112, 113, 177)


@pytest.mark.skipif(gc.read_maps()[0] is None, reason="needs results/thesis/gc/alignment.csv (--build-maps)")
def test_alignment_map_offsets():
    al = gc.read_maps()[0].set_index("orig")
    assert int(al.loc["DM_mtgait_4", "offset"]) == 206
    assert int(al.loc["SC_crouch_og3", "offset"]) == 130
    assert int(al.loc["SC_ngait_og5", "offset"]) == 1
    assert (al["vert_err_mm"] < 0.1).all() and al["knee_time_eq_traj"].all()


@needs_gc
def test_knee_force_file_shares_the_marker_time_base():
    for subj, key, fs, mass in (("JW", "jwngait2", 200.0, 64.6), ("DM", "dmngait4", 200.0, 67.0),
                                ("SC", "scngaitog5", 120.0, 78.4)):
        tr = gc.load_gc_trial(subj, key)
        assert tr["knee_time_equal"] and tr["fs"] == pytest.approx(fs, rel=1e-3)
        assert 1.5 < tr["total"].max() / (gc.G * mass) < 6.0          # plausible peak axial load in BW
        assert tr["medial"].max() < tr["total"].max()


# ─── Left-knee mirroring ──────────────────────────────────────────────────────

def _rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def test_mirror_derivation_from_the_rajagopal_joint_conventions():
    """S = diag(1, 1, -1) reflects the ground z axis. The mirrored pelvis is (tilt, -list, -rotation) and the mirrored
    right hip is the left hip with the same coordinate values (hip_l uses coefficient -1 on adduction / rotation)."""
    S = np.diag([1.0, 1.0, -1.0])
    rng = np.random.default_rng(1)
    for _ in range(20):
        tilt, lst, rot, f, a, r = rng.uniform(-0.6, 0.6, 6)
        pelvis = _rz(tilt) @ _rx(lst) @ _ry(rot)
        np.testing.assert_allclose(S @ pelvis @ S, _rz(tilt) @ _rx(-lst) @ _ry(-rot), atol=1e-12)
        hip_r = _rz(f) @ _rx(a) @ _ry(r)
        hip_l = _rz(f) @ _rx(-a) @ _ry(-r)
        np.testing.assert_allclose(S @ (pelvis @ hip_r) @ S, (_rz(tilt) @ _rx(-lst) @ _ry(-rot)) @ hip_l, atol=1e-12)
        # a femur point at the right hip (pelvis offset +z) lands on the mirrored person's left side
        p_r = pelvis @ (np.array([-0.07, -0.08, 0.08]) + hip_r @ np.array([0.0, -0.4, 0.0]))
        p_l_mirror = (_rz(tilt) @ _rx(-lst) @ _ry(-rot)) @ (np.array([-0.07, -0.08, -0.08]) + hip_l @ np.array([0.0, -0.4, 0.0]))
        np.testing.assert_allclose(S @ p_r, p_l_mirror, atol=1e-12)
    # moments are pseudo-vectors: S M -> -det(S) S M flips M_x, so the two knees' medial signs are opposite
    # (medial-positive M_x = -mx_r = +mx_l; tests/test_knee_mx_sign.py settles which one is which)
    M = np.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(np.linalg.det(S) * (S @ M), [-1.0, -2.0, 3.0])


@pytest.mark.skipif(not os.path.exists(os.path.join(SEG_SC, "scaled_model.osim")), reason="needs a scaled model")
def test_scaled_model_uses_the_mirrored_conventions():
    import xml.etree.ElementTree as ET
    root = ET.parse(os.path.join(SEG_SC, "scaled_model.osim")).getroot()
    coef = {}
    for j in root.iter("CustomJoint"):
        for ta in j.iter("TransformAxis"):
            name = (ta.find("coordinates").text or "").strip()
            lin = ta.find(".//LinearFunction/coefficients")
            if name and lin is not None:
                coef[(j.get("name"), ta.get("name"))] = (float(lin.text.split()[0]),
                                                        tuple(float(v) for v in ta.find("axis").text.split()))
    for rot in ("rotation1", "rotation2", "rotation3"):
        assert coef[("ground_pelvis", rot)][0] == 1.0
    assert coef[("hip_r", "rotation2")] == (1.0, (1.0, 0.0, 0.0)) and coef[("hip_l", "rotation2")] == (-1.0, (1.0, 0.0, 0.0))
    assert coef[("hip_r", "rotation3")] == (1.0, (0.0, 1.0, 0.0)) and coef[("hip_l", "rotation3")] == (-1.0, (0.0, 1.0, 0.0))
    assert coef[("hip_l", "rotation1")] == coef[("hip_r", "rotation1")]
    assert coef[("walker_knee_l", "rotation1")][1] == (-1.0, 0.0, 0.0)   # the knee is mirrored in the model too


@pytest.mark.parametrize("iset", [NOTRANS, Q16])
def test_mirror_channels_and_involution(iset):
    cfg = INPUT_SETS[iset]
    p = cfg.get("q_dim", 16)
    x = np.arange(4 * cfg["per_frame_ch"], dtype=np.float32).reshape(4, -1) + 1
    m = gc.mirror_per_frame(x, cfg)
    np.testing.assert_array_equal(gc.mirror_per_frame(m, cfg), x)
    qi = gc._q_layout(cfg)
    assert m[0, qi["pelvis_tilt"]] == x[0, qi["pelvis_tilt"]]
    assert m[0, qi["pelvis_list"]] == -x[0, qi["pelvis_list"]]
    assert m[0, qi["pelvis_rotation"]] == -x[0, qi["pelvis_rotation"]]
    if "pelvis_tz" in qi:
        assert m[0, qi["pelvis_tz"]] == -x[0, qi["pelvis_tz"]] and m[0, qi["pelvis_tx"]] == x[0, qi["pelvis_tx"]]
    assert m[0, qi["knee_angle_r"]] == x[0, qi["knee_angle_l"]] and m[0, qi["hip_adduction_l"]] == x[0, qi["hip_adduction_r"]]
    np.testing.assert_array_equal(m[:, p:p + 3], x[:, p:p + 3] * [1, 1, -1])             # COM
    np.testing.assert_array_equal(m[:, p + 3:p + 6], x[:, p + 6:p + 9] * [1, 1, -1])     # right <- mirrored left foot
    np.testing.assert_array_equal(m[:, p + 6:p + 9], x[:, p + 3:p + 6] * [1, 1, -1])
    np.testing.assert_array_equal(m[:, p + 9], x[:, p + 10])                             # contact flags swap
    np.testing.assert_array_equal(m[:, -2:], x[:, -2:])                                  # mass, height


def _symmetric_gait(cfg, n=300, period=100):
    """Per-frame inputs of a perfectly left-right symmetric gait: the left side is the mirrored right side half a
    cycle later, and out-of-plane signals (list, rotation, tz, every z) are antiperiodic over a half cycle."""
    t = np.arange(n, dtype=float)
    w = 2 * np.pi / period
    per = lambda k, ph: np.cos(2 * w * t + ph) + 0.3 * np.sin(4 * w * t + k)       # period / 2 periodic
    anti = lambda k, ph: np.sin(w * t + ph) + 0.2 * np.cos(3 * w * t + k)          # changes sign every half cycle
    side = lambda k, shift: np.sin(w * (t + shift) + k) + 0.4 * np.cos(2 * w * (t + shift) + 2 * k)
    h = period / 2
    q = {"pelvis_tilt": per(0, 0.1), "pelvis_list": anti(1, 0.2), "pelvis_rotation": anti(2, 0.3),
         "pelvis_tx": per(3, 0.4), "pelvis_ty": per(4, 0.5), "pelvis_tz": anti(5, 0.6)}
    for k, n_ in enumerate(("hip_flexion", "hip_adduction", "hip_rotation", "knee_angle", "ankle_angle")):
        q[n_ + "_r"], q[n_ + "_l"] = side(k, 0.0), side(k, h)
    qi = gc._q_layout(cfg)
    cols = [None] * len(qi)
    for name, i in qi.items():
        cols[i] = q[name]
    foot = lambda shift: np.column_stack([side(7, shift), side(8, shift), side(9, shift)])
    fr, fl = foot(0.0), foot(h) * [1, 1, -1]
    com = np.column_stack([per(10, 0.0), per(11, 0.0), anti(12, 0.0)])
    c = lambda shift: (np.sin(w * (t + shift) + 0.1) > 0).astype(float)
    x = np.column_stack(cols + [com, fr, fl, c(0.0), c(h), np.full(n, 70.0), np.full(n, 1.7)])
    return x.astype(np.float32), int(h)


@pytest.mark.parametrize("iset", [NOTRANS, Q16])
def test_mirror_of_a_symmetric_gait_is_the_gait_half_a_cycle_later(iset):
    cfg = INPUT_SETS[iset]
    x, h = _symmetric_gait(cfg)
    m = gc.mirror_per_frame(x, cfg)
    np.testing.assert_allclose(m[:-h], x[h:], atol=1e-5)
    # also after windowing (away from the replicated edges)
    wm, wx = gc.windowize(m, cfg), gc.windowize(x, cfg)
    np.testing.assert_allclose(wm[20:-h - 20], wx[h + 20:-20], atol=1e-5)


# ─── Feature builder, labels, stance halves ───────────────────────────────────

@pytest.mark.skipif(not os.path.exists(os.path.join(SEG_JW, "jcf_output", "BatchJCF_JointReaction_ReactionLoads.sto")),
                    reason="needs the labelled fregly_1GC_t00_r00 segment")
@pytest.mark.parametrize("iset", [FPOPT, FPOPT + "_t100", NOTRANS])
def test_label_free_builder_equals_the_training_builder(iset):
    cfg = INPUT_SETS[iset]
    ref = _build_com_features(SEG_JW, cfg, max_peak_bw=6.0, min_length=1, predict_moment=True, return_meta=True)
    assert ref is not None, "segment should pass the training label filters"
    feat_ref, labels_ref, _, meta_ref = ref
    built = gc.build_frame_features(SEG_JW, cfg, grid="jr")
    feat, meta = built
    np.testing.assert_allclose(feat, feat_ref, atol=1e-5)
    if cfg.get("time_exact"):
        np.testing.assert_allclose(meta["t"], meta_ref["t"], atol=1e-12)
    lab, info = gc.jr_labels(SEG_JW, meta)
    assert info["status"] == "ok" and info["label_ok"]
    np.testing.assert_allclose(lab, labels_ref, atol=1e-5)


@pytest.mark.skipif(not os.path.exists(os.path.join(SEG_SC, "com_features.npz")), reason="needs fregly_3GC_t18_r00")
def test_builder_context_grid_covers_the_jr_window():
    cfg = dict(INPUT_SETS[FPOPT], time_exact=True, resample_hz=100.0)
    f_jr, m_jr = gc.build_frame_features(SEG_SC, cfg, grid="jr")
    f_ik, m_ik = gc.build_frame_features(SEG_SC, cfg, grid="ik")
    assert m_ik["t"][0] <= m_jr["t"][0] and m_ik["t"][-1] >= m_jr["t"][-1] and len(f_ik) > len(f_jr)
    np.testing.assert_allclose(np.diff(m_jr["t"]), 0.01, atol=1e-9)


def test_jr_labels_left_knee_sign(tmp_path):
    d = tmp_path / "seg"
    (d / "jcf_output_sojr").mkdir(parents=True)
    pre = "walker_knee_l_on_tibia_l_in_tibia_l_"
    t = np.linspace(0, 0.1, 11)
    cols = {"time": t, pre + "fx": np.zeros(11), pre + "fy": -700.0 * np.ones(11), pre + "fz": np.zeros(11),
            pre + "mx": 20.0 * np.ones(11)}
    with open(d / "jcf_output_sojr" / "BatchJCF_JointReaction_ReactionLoads.sto", "w") as f:
        f.write("JR\nversion=1\nendheader\n")
        pd.DataFrame(cols).to_csv(f, sep="\t", index=False)
    meta = dict(mass=70.0, height=1.75, t=t, t_common=t)
    joint, sign = gc.osim_joint_sign("l")
    assert (joint, sign) == ("walker_knee_l", 1.0)                                      # medial = +mx on the left
    lab, info = gc.jr_labels(str(d), meta, joint, gc.OSIM_SOURCES["osim_sojr"][0], sign)
    np.testing.assert_allclose(lab[:, 1], -700.0 / (70 * gc.G))
    np.testing.assert_allclose(lab[:, 3], 20.0 / (70 * gc.G * 1.75))                    # medial-positive
    jp, sp = gc.osim_joint_sign("l", "pipeline")
    lab_p, _ = gc.jr_labels(str(d), meta, jp, gc.OSIM_SOURCES["osim_sojr"][0], sp)
    np.testing.assert_allclose(lab_p[:, 3], -lab[:, 3])                                 # the old sign: inverted
    assert info["label_ok"]
    assert gc.jr_labels(str(d), meta)[0] is None                                        # no stored-label file
    lab_r, info_r = gc.jr_labels(str(d), meta, "walker_knee_r", "jcf_output_sojr")
    assert lab_r is None and info_r["status"] == "no walker_knee_r columns"


def test_mx_convention_default_and_signs():
    assert gc.osim_joint_sign("r") == ("walker_knee_r", -1.0)
    assert gc.osim_joint_sign("r", "pipeline") == ("walker_knee_r", 1.0)
    assert gc.surrogate_mx_sign() == -1.0 and gc.surrogate_mx_sign("pipeline") == 1.0
    src = open(gc.__file__).read()
    assert 'ap.add_argument("--mx-convention", choices=MX_CONVENTIONS, default="physical"' in src


def test_mx_sign_check_on_synthetic_frames():
    n = 50
    dl = np.linspace(0.2, 1.0, n)                                 # measured medial - lateral (BW)
    fr = pd.DataFrame(dict(subject="X", stance=True, H=1.7, meas_med=1.0 + dl / 2, meas_lat=1.0 - dl / 2,
                           meas_tot=2.0 * np.ones(n), fy_good=-2.0, mx_good=dl * 0.045 / 1.7 / 2,
                           fy_bad=-2.0, mx_bad=-dl * 0.045 / 1.7 / 2))
    c = gc.mx_sign_check(fr, ["good", "bad"]).set_index("source")
    assert c.loc["good", "r_mx_vs_meas_med_minus_lat"] > 0.99 and c.loc["bad", "r_mx_vs_meas_med_minus_lat"] < -0.99
    np.testing.assert_allclose(c.loc["good", "medial_share"], c.loc["good", "meas_medial_share"])
    assert c.loc["good", "medial_moment_frac"] == 1.0 and c.loc["bad", "medial_moment_frac"] == 0.0


def test_stance_windows_coverage_and_kinematic_completion():
    rows = np.arange(30, 100, dtype=float)                        # segment starts 30 % into a 0-100 stance
    gate = np.ones(len(rows), bool)
    stances = [dict(hs=0.0, to=100.0, hs_src="kin", to_src="grf")]
    sid, pct, part, halves = gc.stance_halves(rows, gate, stances, min_cover=0.9)
    early, late = (h for h in halves)
    assert (early["win_lo"], early["win_hi"], late["win_lo"], late["win_hi"]) == (5.0, 45.0, 55.0, 95.0)
    assert not early["scored"] and early["cover"] == pytest.approx(16 / 40)        # rows 30-45 of the 5-45 window
    assert late["scored"] and late["cover"] == pytest.approx(1.0)
    assert set(part[(rows >= 30) & (rows <= 45)]) == {"early"} and set(part[(rows >= 55) & (rows <= 95)]) == {"late"}
    assert set(part[(rows > 45) & (rows < 55)]) == {""} and set(part[rows > 95]) == {""}
    np.testing.assert_allclose(pct, rows)
    # contact seen from row 30 to the segment end: heel strike from the kinematics (bias-corrected), toe-off unseen
    ev = gc.grf_events(np.arange(30, 90, dtype=float), np.r_[np.ones(50, bool), np.zeros(10, bool)])
    assert ev == [(30.0, 80.0, False, True)]
    st = gc.complete_stances(ev, [(5, 83)], bias_on=6.0, bias_off=3.0, fs=100.0)
    assert st == [dict(hs=-1.0, to=80.0, hs_src="kin", to_src="grf")]
    assert gc.complete_stances(ev, [(200, 300)], 0.0, 0.0, 100.0) == []                 # no overlapping stance
    on_b, off_b = gc.event_bias([(30.0, 80.0, True, True)], [(34, 77)], fs=100.0)
    assert on_b == [4.0] and off_b == [-3.0]


def test_window_peak_rejects_a_maximum_on_the_window_edge():
    p = np.arange(55.0, 96.0)                                     # late window, 1 % steps
    tail = np.exp(-(p - 30.0) ** 2 / 400.0)                       # descent of a first peak at 30 %: max at 55 %
    k, v, at, interior = gc.window_peak(tail, p)
    assert k == 0 and at == 55.0 and not interior
    hump = np.exp(-(p - 75.0) ** 2 / 50.0)                        # a genuine second peak
    k, v, at, interior = gc.window_peak(hump, p)
    assert at == 75.0 and v == pytest.approx(1.0) and interior
    assert not gc.window_peak(np.exp(-(p - 57.0) ** 2 / 50.0), p, margin=3.0)[3]        # 2 % from the edge
    assert gc.window_peak(np.exp(-(p - 58.0) ** 2 / 50.0), p, margin=3.0)[3]            # 3 % from the edge
    # the edge is the first covered frame, not the nominal window start
    assert not gc.window_peak(hump[18:], p[18:])[3]
    assert gc.window_peak(np.full(3, np.nan), p[:3])[0] is None


def _one_stance_frames(meas, pred, n=101):
    pct = np.linspace(0, 100, n, endpoint=False)
    rows = pct.copy()
    fr = pd.DataFrame(dict(subject="DM", folder="f", gc_key="k", trial="t", gait="ngait", row=rows, stance=True,
                           H=1.7, mass=70.0, meas_med=meas(pct), fy_m=-2 * pred(pct), mx_m=0.0))
    _, fr["pct"], fr["part"], hv = gc.stance_halves(rows, np.ones(n, bool), [dict(hs=0.0, to=100.0, hs_src="grf",
                                                                                to_src="grf")])
    halves = [dict(h, subject="DM", folder="f", gc_key="k", trial="t", gait="ngait",
                   frame_index=np.flatnonzero(h["mask"])) for h in hv]
    return fr, halves


def test_peak_table_scores_only_interior_measured_peaks():
    one = lambda p: 2.0 * np.exp(-(p - 30) ** 2 / 200)                          # single early peak (DM-like)
    two = lambda p: 2.0 * np.exp(-(p - 25) ** 2 / 100) + 1.6 * np.exp(-(p - 75) ** 2 / 100)
    fr, halves = _one_stance_frames(one, one)
    st = gc.peak_table(halves, fr, ["m"], 0.5, 45.0)
    assert list(st.part) == ["early", "late"] and list(st.meas_interior) == [True, False]
    cnt = gc.peak_counts(st).set_index("part")
    assert (cnt.loc["late", "n_covered"], cnt.loc["late", "n_edge_rejected"], cnt.loc["late", "n_scored"]) == (1, 1, 0)
    ps = gc.peaks_summary(st, ["m"])
    assert list(ps.part) == ["early"] and ps.mae.iloc[0] == pytest.approx(0.0, abs=1e-9)
    # a source without the second peak: scored on the interior measured late peak, flagged as not interior itself
    fr, halves = _one_stance_frames(two, one)
    st = gc.peak_table(halves, fr, ["m"], 0.5, 45.0)
    late = st.set_index("part").loc["late"]
    assert late.meas_interior and not late.interior_m and late.err_m < -1.0
    assert gc.peaks_summary(st, ["m"]).set_index("part").loc["late", "pred_interior_frac"] == 0.0
    assert gc.gait_mods(st, ["m"]).empty                                          # no non-normal gait


def test_fixed_lag_from_heel_strike_to_first_peak_delays():
    d = pd.DataFrame(dict(subject=["DM"] * 3 + ["JW"] * 5 + ["SC"] * 4,
                          delay_med_ms=[236, 226, 221, 151, 161, 141, 131, 116, 149, 116, 130, 197],
                          gait=["ngait", "tsgait", "tsgait", "tsgait", "tsgait", "mtgait", "mtgait", "wpgait",
                                "ngait", "trunksway", "crouch", "crouch"], interior_med=True))
    assert gc.fixed_lag_ms(d, match_gait=False) == 85.0     # 226 - median(116 ... 197) = 226 - 141
    # gait-matched: normal 236 - 149 = 87; trunk sway (JW tsgait 151, 161, SC trunksway 116 -> 151): 75, 70
    assert gc.fixed_lag_ms(d) == 75.0
    d.loc[0, "interior_med"] = False                        # an edge maximum does not count
    assert gc.fixed_lag_ms(d, match_gait=False) == 80.0     # 223.5 - 141 = 82.5 -> 80 (5 ms steps)
    assert gc.fixed_lag_ms(d) == 70.0                       # median(75, 70) = 72.5 -> 70
    no_match = d.assign(gait=np.where(d.subject == "DM", "bouncy", d.gait))
    assert gc.fixed_lag_ms(no_match) == gc.fixed_lag_ms(no_match, match_gait=False)       # falls back to pooled
    assert np.isnan(gc.fixed_lag_ms(d[d.subject != "DM"])) and np.isnan(gc.fixed_lag_ms(pd.DataFrame()))


def _fake_trial(monkeypatch, n=400, fs=200.0, peak_row=130):
    r = np.arange(n, dtype=float)
    sig = 1000.0 * np.exp(-(r - peak_row) ** 2 / 50.0)
    tr = dict(t=r / fs, medial=sig, lateral=0.5 * sig, total=1.5 * sig)
    monkeypatch.setattr(gc, "load_gc_trial", lambda subj, key, root=None: tr)
    return tr


def test_shift_measured_takes_the_implant_force_later(monkeypatch):
    tr = _fake_trial(monkeypatch)
    fr = pd.DataFrame(dict(subject="DM", gc_key="k", row=np.arange(100, 160, dtype=float), mass=70.0,
                           meas_med=0.0, meas_lat=0.0, meas_tot=0.0))
    out = gc.shift_measured(fr, 50.0, None)                 # 50 ms = 10 rows at 200 Hz
    np.testing.assert_allclose(out.meas_med, tr["medial"][110:170] / (70 * gc.G))
    np.testing.assert_allclose(out.meas_tot, tr["total"][110:170] / (70 * gc.G))
    assert (fr.meas_med == 0).all()                         # the input is not modified


def test_hs_peak_delays_time_the_first_peak_after_a_plate_heel_strike(monkeypatch):
    _fake_trial(monkeypatch, peak_row=130)
    h = dict(part="early", hs=100.0, to=230.0, hs_src="grf", to_src="grf", gc_key="k", trial="t", gait="ngait")
    res = {"DM": dict(halves=[h, dict(h), dict(h, part="late"), dict(h, hs_src="kin", hs=90.0)])}
    d = gc.hs_peak_delays(res, None)
    assert len(d) == 1                                      # duplicates, late windows and kinematic heel strikes skipped
    assert d.delay_med_ms.iloc[0] == pytest.approx(150.0) and d.interior_med.iloc[0]
    assert d.pct_med.iloc[0] == pytest.approx(100 * 30 / 130)


def test_common_mask_restricts_each_subject_only_by_the_sources_it_has():
    fr = pd.DataFrame(dict(subject=["JW", "JW", "JW", "SC", "SC"],
                           fy_osim_state=[1.0, np.nan, 1.0, np.nan, np.nan], osim_state_ok=[True, True, False, False, False],
                           fy_osim_sojr=[1.0, 1.0, 1.0, 1.0, np.nan], osim_sojr_ok=[True, True, True, True, True]))
    np.testing.assert_array_equal(gc.common_mask(fr), [True, False, False, True, False])


@needs_gc
def test_statop_windows_map_to_gc_rows():
    w = gc.statop_windows()
    if not w:
        pytest.skip("no StatOp_Val_Data")
    assert len(w) == 62
    for x in w[:5]:
        np.testing.assert_array_equal(x["rows"], 1 + np.rint(120 * x["tau"]).astype(int))
        assert (x["medial_N"] >= 0).all() and x["gc_key"] in gc.gc_trial_files("SC")


# ─── JR-only OpenSim setup ────────────────────────────────────────────────────

@pytest.mark.skipif(not os.path.exists(os.path.join(SEG_SC, "jcf_output", "_analyze_setup.xml")),
                    reason="needs the SO+JR setup of fregly_3GC_t18_r00")
def test_jr_only_setup_turns_off_so_and_reads_the_force_file(tmp_path):
    src = os.path.join(SEG_SC, "jcf_output", "_analyze_setup.xml")
    forces = os.path.join(SEG_SC, "jcf_output", "BatchJCF_StaticOptimization_force.sto")
    tool = jr_only_setup(src, forces, str(tmp_path)).getroot().find("AnalyzeTool")
    orig = __import__("xml.etree.ElementTree", fromlist=["parse"]).parse(src).getroot().find("AnalyzeTool")
    objs = tool.find("AnalysisSet").find("objects")
    assert objs.find("StaticOptimization").find("on").text == "false"
    jr = objs.find("JointReaction")
    assert jr.find("on").text == "true" and jr.find("forces_file").text == os.path.abspath(forces)
    assert not (orig.find("AnalysisSet").find("objects").find("JointReaction").find("forces_file").text or "").strip()
    assert jr.find("joint_names").text.split() == ["walker_knee_r", "walker_knee_l"]
    assert jr.find("apply_on_bodies").text.split() == ["child", "child"]
    assert jr.find("express_in_frame").text.split() == ["child", "child"]
    assert tool.find("results_directory").text == str(tmp_path)
    for tag in ("model_file", "initial_time", "final_time", "coordinates_file", "external_loads_file",
                "lowpass_cutoff_frequency_for_coordinates"):
        assert tool.find(tag).text == orig.find(tag).text


def _sto(path, cols):
    with open(path, "w") as f:
        f.write("x\nversion=1\nendheader\n")
        pd.DataFrame(cols).to_csv(f, sep="\t", index=False)


def test_vasti_consistency_check(tmp_path):
    t = np.linspace(0, 0.5, 6)
    bw = 70 * gc_knee_jr.G
    so = pd.DataFrame({"time": t, "vaslat_r": np.array([0.2, 1.0, 1.5, 1.5, 0.5, 0.1]) * bw, "vasmed_r": 0.0,
                       "vasint_r": 0.0, "gasmed_r": 5 * bw})
    pre = gc_knee_jr.knee_prefix("r")
    low = pd.DataFrame({"time": t, pre + "fx": 0.0, pre + "fy": -1.2 * bw, pre + "fz": 0.0})       # below 1.5 BW
    high = pd.DataFrame({"time": t, pre + "fx": 0.3 * bw, pre + "fy": -2.0 * bw, pre + "fz": 0.0})
    assert gc_knee_jr.below_vasti(so, low, "r", bw) == (2, 1.0)                # frames with vasti > 1 BW: 2
    assert gc_knee_jr.below_vasti(so, high, "r", bw) == (2, 0.0)
    n_l, frac_l = gc_knee_jr.below_vasti(so, high, "l", bw)                    # no left vasti, no left knee columns
    assert n_l == 0 and np.isnan(frac_l)
    np.testing.assert_allclose(gc_knee_jr.knee_force_bw(high, "r", bw), np.hypot(0.3, 2.0))


@pytest.mark.skipif(not os.path.exists(os.path.join(SEG_JW, "jcf_output", "BatchJCF_StaticOptimization_force.sto")),
                    reason="needs the labelled fregly_1GC_t00_r00 segment")
def test_stored_label_check_runs_on_a_real_folder():
    row = gc_knee_jr.check_folder(SEG_JW, stored_only=True)
    assert row["n_so"] > 10 and "stored_right_max_abs_diff" not in row
    for side in "rl":
        f = row[f"stored_below_vas_{side}_frac"]
        assert np.isnan(f) or 0.0 <= f <= 1.0
    assert row["stored_r_F_max_bw"] > 0.3 and np.isnan(row["stored_l_F_max_bw"])     # stored labels: right knee only


def test_osim_label_notes_sojr_coverage(tmp_path):
    seg = pd.DataFrame(dict(subject=["JW", "DM", "DM", "SC", "SC"], status="ok",
                            osim_state_status=["ok", "ok", "ok", "missing", "missing"],
                            osim_sojr_status=["missing", "ok", "missing", "ok", "ok"],
                            osim_sojr_label_ok=[np.nan, True, np.nan, False, True]))
    txt = "\n".join(gc.osim_label_notes(seg, repo=str(tmp_path)))
    assert "SC 1 of 2" in txt and "DM 0 of 1" in txt                  # label-filter failures per subject
    assert "no `osim_sojr` labels for JW" in txt and "DM has them on only 1 of 2" in txt
    full = seg.assign(osim_sojr_status="ok", osim_sojr_label_ok=True)
    txt = "\n".join(gc.osim_label_notes(full, repo=str(tmp_path)))
    assert "coverage is incomplete" not in txt and "not generated yet" not in txt

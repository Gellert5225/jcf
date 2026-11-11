"""Which sign of the OpenSim knee moment M_x is medial (analysis/thesis/knee_mx_sign.py).

Settles the convention behind F_medial = 0.5|F_y| + s*M_x*H/d: s = -1 at walker_knee_r and +1 at walker_knee_l.
Three independent checks: the two-point contact algebra, the medial / lateral geometry of our own models, and
Kaneda et al.'s JR output, which reports the total tibia load and the condyle split in one table. The last two skip
when the model files or the Grand Challenge package are missing.
"""
import glob
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from analysis.thesis import knee_mx_sign as ks  # noqa: E402

D = 0.045


def test_convention_table():
    assert ks.medial_mx_sign("walker_knee_r") == -1.0 and ks.medial_mx_sign("walker_knee_l") == 1.0
    assert ks.medial_mx_sign("r", "pipeline") == 1.0 and ks.medial_mx_sign("tibia_l", "pipeline") == -1.0
    with pytest.raises(ValueError):
        ks.medial_mx_sign("walker_knee_r", "other")
    with pytest.raises(ValueError):
        ks.knee_side("knee")


@pytest.mark.parametrize("side", ["r", "l"])
def test_two_point_contact_recovers_the_medial_force_only_with_the_physical_sign(side):
    """Medial condyle at z = -d/2 on the right tibia, +d/2 on the left (z points right on both legs)."""
    rng = np.random.default_rng(0)
    f_med, f_lat = rng.uniform(0, 2000, 100), rng.uniform(0, 2000, 100)
    z_med = -D / 2 if side == "r" else D / 2
    fy, mx = ks.two_point_reaction(f_med, f_lat, z_med, -z_med)
    np.testing.assert_allclose(fy, -(f_med + f_lat))
    np.testing.assert_allclose(mx, z_med * f_med - z_med * f_lat)              # M_x = z_c * f per contact
    np.testing.assert_allclose(ks.fmedial_n(fy, mx, ks.medial_mx_sign(side), d_m=D), f_med, atol=1e-9)
    np.testing.assert_allclose(ks.fmedial_n(fy, mx, ks.medial_mx_sign(side, "pipeline"), d_m=D), f_lat, atol=1e-9)
    # a pure medial load: compressive F_y, M_x < 0 on the right tibia and > 0 on the left
    fy1, mx1 = ks.two_point_reaction(1000.0, 0.0, z_med, -z_med)
    assert fy1 < 0 and np.sign(mx1) == ks.medial_mx_sign(side)


def test_two_point_moment_ignores_the_anterior_offset_of_a_vertical_force():
    _, mx = ks.two_point_reaction(800.0, 300.0, -0.02, 0.02, x_med=0.01, x_lat=-0.015)
    np.testing.assert_allclose(mx, -0.02 * 800 + 0.02 * 300)


MODELS = ks.one_model_per_lab()


@pytest.mark.skipif(not MODELS, reason="needs jcf/full_duration/*/walking/*/jcf_output/_model_with_reserves.osim")
def test_model_geometry_puts_the_medial_condyle_at_minus_z_on_the_right():
    geo = ks.geometry_check(MODELS)
    assert set(geo.side) == {"r", "l"} and len(geo) == 2 * len(MODELS)
    assert geo.medial_all_one_side.all() and geo.lateral_all_one_side.all()
    assert (geo.n_medial >= 4).all() and (geo.n_lateral >= 2).all()
    for _, r in geo.iterrows():
        assert r.implied_medial_sign == r.physical_sign == ks.medial_mx_sign(r.side)
        assert abs(r.z_medial_m) > 0.01 and abs(r.z_lateral_m) > 0.01


def test_knee_geometry_reads_muscle_points_relative_to_the_knee_centre(tmp_path):
    p = tmp_path / "m.osim"
    p.write_text("""<OpenSimDocument><Model>
      <CustomJoint name="walker_knee_r"><frames>
        <PhysicalOffsetFrame name="tibia_r_offset"><socket_parent>/bodyset/tibia_r</socket_parent>
          <translation>0 0 -0.002</translation></PhysicalOffsetFrame></frames></CustomJoint>
      <Millard2012EquilibriumMuscle name="grac_r"><GeometryPath name="path"><PathPointSet><objects>
        <PathPoint name="grac_r-P1"><socket_parent_frame>/bodyset/pelvis</socket_parent_frame>
          <location>0 0 0.03</location></PathPoint>
        <PathPoint name="grac_r-P2"><socket_parent_frame>/bodyset/tibia_r</socket_parent_frame>
          <location>0 -0.05 -0.032</location></PathPoint>
      </objects></PathPointSet></GeometryPath></Millard2012EquilibriumMuscle>
      <Millard2012EquilibriumMuscle name="bfsh_r"><GeometryPath name="path"><PathPointSet><objects>
        <PathPoint name="bfsh_r-P2"><socket_parent_frame>/bodyset/tibia_r</socket_parent_frame>
          <location>0 -0.05 0.034</location></PathPoint>
      </objects></PathPointSet></GeometryPath></Millard2012EquilibriumMuscle>
    </Model></OpenSimDocument>""")
    g = ks.knee_geometry(str(p))
    assert list(g.muscle) == ["grac_r", "bfsh_r"]                 # the pelvis point is not a knee point
    np.testing.assert_allclose(g.z_rel, [-0.030, 0.036])
    chk = ks.geometry_check({"x": str(p)})
    assert chk.implied_medial_sign.tolist() == [-1.0]


def _kaneda_ok():
    return bool(glob.glob(os.path.join(ks.GC_ROOT, "StatOp_Val_Data", "*", "SO_JRA", "*", "instKnee", "*.sto")))


@pytest.mark.skipif(not _kaneda_ok(), reason="needs jcf/external/grand_challenge/StatOp_Val_Data")
def test_kaneda_condyle_split_identity():
    """In Kaneda et al.'s own JR output the tibia moment is the condyle forces times their lever arms:
    medial lever arm > 0 at SC's left knee and < 0 at JW's and DM's right knees."""
    kan = ks.kaneda_check(ks.kaneda_frames()).set_index("subject")
    assert set(kan.index) == {"SC", "JW", "DM"}
    for subj, r in kan.iterrows():
        assert r.fy_compressive_frac > 0.99
        assert r.implied_medial_sign == ks.medial_mx_sign(r.side), subj
        assert abs(r.r_mx_vs_med_minus_lat) > 0.9 and r.resid_rms_nm < 0.25 * r.mx_sd_nm, subj
        assert 0.01 < abs(r.a_med_m) < 0.07 and 0.01 < abs(r.b_lat_m) < 0.07, subj


def test_stance_profile_uses_complete_contact_runs_only():
    t = np.arange(0, 3.0, 0.01)
    on = ((t > 0.2) & (t < 0.9)) | ((t > 1.5) & (t < 2.2)) | (t > 2.8)        # two complete runs, one cut off
    x = np.where(on, 1.0, 0.0)
    x[(t > 1.5) & (t < 2.2)] = 3.0
    prof = ks.stance_profile(t, on, x)
    np.testing.assert_allclose(prof, 2.0)                                     # mean of the two complete runs
    assert np.isnan(ks.stance_profile(t, t > 2.8, x)).all()                   # no complete run

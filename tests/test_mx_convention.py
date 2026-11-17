"""The explicit M_x sign convention (Nov 2026): new checkpoints store mx_medial_sign and use it in their F_medial loss
and in inference_posture_surrogate's f_medial; old checkpoints keep the legacy +1; the evaluators form F_medial with
the physical sign (-mx at walker_knee_r) by default and reproduce the old numbers with --mx-convention pipeline.
"""
import inspect
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "tools"))

import train_mlp as tm  # noqa: E402
import evaluate_fmedial as ev  # noqa: E402
from analysis.thesis import knee_mx_sign as ks  # noqa: E402

FPOPT_CKPT = os.path.join(REPO, "models", "runs", "mlp",
                          "best_model_mlp_walking_qinst_notrans_pelvisrel_winpos_contact_fpopt_w20_fmW0p5_e100.pt")
PAPER_CKPT = os.path.join(REPO, "models", "mlp-grffree-base", "mlp-grffree-base.pt")
TRIAL = os.path.join(REPO, "jcf", "full_duration", "testing", "walking", "carter_P010_split0_t10")
QC_SETS = ("qinst_notrans_pelvisrel_winpos_contact_w20_t100_sojrqc",
           "qinst_notrans_pelvisrel_winpos_contact_fpopt_w20_t100_sojrqc")
D = 0.045


# ─── One convention table everywhere ──────────────────────────────────────────

def test_sign_constants_agree_with_the_settled_convention():
    assert tm.MX_SIGN_PHYSICAL == ks.medial_mx_sign("walker_knee_r") == -1.0
    assert tm.MX_SIGN_PIPELINE == ks.medial_mx_sign("walker_knee_r", "pipeline") == 1.0
    from pn3_redis import evaluate as pe
    from analysis.thesis import twente_imu as tw
    for table in (ev.MX_CONVENTION_SIGN, pe.MX_SIGN, tw.MX_SIGN):
        assert table == tm.MX_CONVENTIONS


def test_only_the_qc_sets_use_the_physical_sign():
    for name, cfg in tm.INPUT_SETS.items():
        if name.startswith(QC_SETS) or name == tm.NO_MUSCLE_HD_SET:   # QC sets, their *_hd variants, no-muscle QC set
            assert tm.mx_medial_sign_of(cfg) == -1.0, name
        else:
            assert "mx_medial_sign" not in cfg and tm.mx_medial_sign_of(cfg) == 1.0, name
    assert tm.mx_medial_sign_of({}) == 1.0           # old checkpoints / configs without the key


def test_fmedial_term_gives_the_medial_condyle_force_only_with_the_physical_sign():
    """Two-point contact on the right tibia (medial condyle at z = -d/2): F_medial = the medial contact force."""
    rng = np.random.default_rng(3)
    f_med, f_lat = rng.uniform(0, 2000, 200), rng.uniform(0, 2000, 200)
    fy, mx = ks.two_point_reaction(f_med, f_lat, -D / 2, D / 2)
    bw, h = 800.0, 1.75
    fy_bw, mx_bwh = fy / bw, mx / (bw * h)
    np.testing.assert_allclose(tm.fmedial_term(fy_bw, mx_bwh, h, tm.MX_SIGN_PHYSICAL) * bw, f_med, atol=1e-9)
    np.testing.assert_allclose(tm.fmedial_term(fy_bw, mx_bwh, h, tm.MX_SIGN_PIPELINE) * bw, f_lat, atol=1e-9)
    import torch
    t = tm.fmedial_term(torch.tensor(fy_bw), torch.tensor(mx_bwh), torch.tensor(h), -1.0)
    np.testing.assert_allclose(t.numpy() * bw, f_med, atol=1e-6)


def test_train_uses_and_stores_the_input_set_sign():
    """No training run: the loss term and the checkpoint dict read mx_sign = the input set's mx_medial_sign."""
    src = inspect.getsource(tm.train)
    assert "mx_sign = mx_medial_sign_of(INPUT_SETS[input_set])" in src
    assert "fmedial_term(fy_p, mx_p, H, mx_sign" in src and "fmedial_term(fy_t, mx_t, H, mx_sign" in src
    assert '"mx_medial_sign": mx_sign' in src and '"label_mask": label_mask' in src
    assert "mx_p * H) / D_M_FMEDIAL" not in src       # the old unsigned formula is gone


# ─── Inference module ─────────────────────────────────────────────────────────

def _fake(tmp_path, name, **kw):
    import torch
    ck = torch.load(FPOPT_CKPT, map_location="cpu", weights_only=True)
    ck.update(kw)
    p = str(tmp_path / f"{name}.pt")
    torch.save(ck, p)
    return p


def _query(m, rng):
    q = rng.normal(0, 0.2, (8, 16))
    w = lambda c: np.broadcast_to(np.array(c)[None, None], (8, 20, 3)) + rng.normal(0, 0.02, (8, 20, 3))  # noqa: E731
    return dict(joint_angles=q, com_rel_window=w([0, 0.05, 0]), calcn_r_rel_window=w([0.1, -0.9, 0.1]),
                calcn_l_rel_window=w([-0.1, -0.9, -0.1]), mass_kg=72.0, height_m=1.78,
                r_contact_window=np.ones((8, 20)), l_contact_window=np.zeros((8, 20)))


@pytest.mark.skipif(not os.path.exists(FPOPT_CKPT), reason="needs the fpopt checkpoint")
def test_inference_f_medial_follows_the_checkpoint_sign(tmp_path):
    from inference_posture_surrogate import JCFPostureSurrogate as S
    old = S(FPOPT_CKPT, device="cpu")                                   # stores no sign: legacy +1
    new = S(_fake(tmp_path, "qc", input_set=QC_SETS[1], mx_medial_sign=-1.0), device="cpu")
    assert old.mx_medial_sign == 1.0 and new.mx_medial_sign == -1.0
    kw = _query(old, np.random.default_rng(0))
    a, b = old.predict_windowed(**kw), new.predict_windowed(**kw)
    np.testing.assert_allclose(a["mx"], b["mx"])                       # same weights: same raw outputs
    np.testing.assert_allclose(a["f_medial"], 0.5 * np.abs(a["fy"]) + a["mx"] * 1.78 / D, rtol=1e-6)   # unchanged
    np.testing.assert_allclose(b["f_medial"], 0.5 * np.abs(b["fy"]) - b["mx"] * 1.78 / D, rtol=1e-6)
    np.testing.assert_allclose(b["mx_medial"], -b["mx"])
    np.testing.assert_allclose(a["mx_medial"], a["mx"])
    with pytest.raises(ValueError):
        S(_fake(tmp_path, "bad", mx_medial_sign=0.5), device="cpu")
    for name in QC_SETS:
        assert name in S.SUPPORTED_INPUT_SETS


# ─── tools/evaluate_fmedial.py ────────────────────────────────────────────────

def test_medial_moment_and_to_convention():
    mx = np.array([0.02, -0.01])
    np.testing.assert_allclose(ev.medial_moment(mx), -mx)
    np.testing.assert_allclose(ev.medial_moment(mx, "pipeline"), mx)
    a = np.array([[0.1, -2.0, 0.3, 0.02]])
    b = ev.to_convention(a)
    assert b[0, 3] == -0.02 and a[0, 3] == 0.02 and np.all(b[0, :3] == a[0, :3])   # a copy; forces untouched
    with pytest.raises(ValueError):
        ev.medial_moment(mx, "lateral")


@pytest.mark.skipif(not (os.path.exists(PAPER_CKPT) and os.path.isdir(TRIAL)), reason="needs the paper checkpoint and a test trial")
def test_evaluator_applies_one_convention_to_labels_and_outputs(tmp_path):
    common = ["--ckpt", PAPER_CKPT, "--dirs", TRIAL, "--out-root", str(tmp_path)]
    out_p = ev.main(common + ["--name", "pipe", "--mx-convention", "pipeline"])
    out_f = ev.main(common + ["--name", "phys"])
    row = lambda out, s: pd.read_csv(os.path.join(out, "trials.csv")).query("subset == @s").iloc[0]  # noqa: E731
    p, f = row(out_p, "all"), row(out_f, "all")
    assert p.n_frames == f.n_frames and p.fy_mae == pytest.approx(f.fy_mae) and p.mx_mae == pytest.approx(f.mx_mae)
    assert p.mx_bias == pytest.approx(-f.mx_bias) and p.moment_part_bias == pytest.approx(-f.moment_part_bias)
    assert p.fmed_mae != pytest.approx(f.fmed_mae, abs=1e-3)
    # ground truth: the stance F_medial peak from the label file under each sign
    meta = json.load(open(os.path.join(TRIAL, "metadata.json")))
    bw, h = meta["mass_kg"] * 9.81, meta["height_m"]
    j = ev._read_sto(os.path.join(TRIAL, "jcf_output", ev.LABEL_FILE))
    ik_t = pd.read_csv(os.path.join(TRIAL, "ik_results.mot"), sep=r"\s+", skiprows=6)["time"].to_numpy()
    t = tm._common_time_grid(TRIAL, ik_t)
    fy = np.interp(t, j["time"], j["walker_knee_r_on_tibia_r_in_tibia_r_fy"]) / bw
    mx = np.interp(t, j["time"], j["walker_knee_r_on_tibia_r_in_tibia_r_mx"]) / (bw * h)
    grf = ev._read_sto(os.path.join(TRIAL, "grf_data.mot"))
    st = np.abs(np.interp(t, grf["time"], grf["calcn_r_force_vy"])) / bw > ev.CONTACT_BW
    assert row(out_p, "stance").peak_gt == pytest.approx((0.5 * np.abs(fy) + mx * h / D)[st].max(), abs=1e-3)
    assert row(out_f, "stance").peak_gt == pytest.approx((0.5 * np.abs(fy) - mx * h / D)[st].max(), abs=1e-3)
    assert "M_x convention: physical" in open(os.path.join(out_f, "README.md")).read()
    assert "M_x convention: pipeline" in open(os.path.join(out_p, "README.md")).read()
    assert "+1 (legacy, not stored)" in open(os.path.join(out_f, "README.md")).read()


# ─── pn3_redis/evaluate.py and analysis/thesis/twente_imu.py ──────────────────

def test_pn3_evaluate_recovers_the_logged_sign_and_height():
    from pn3_redis import evaluate as pe
    rng = np.random.default_rng(1)
    fy, mx = rng.uniform(-3, 0, 300), rng.uniform(-0.05, 0.05, 300)
    for sign in (-1.0, 1.0):
        f = 0.5 * np.abs(fy) + sign * mx * 1.83 / D
        s, h = pe.logged_sign_and_height(np.round(f, 6), np.round(fy, 6), np.round(mx, 6))
        assert s == sign and h == pytest.approx(1.83, abs=1e-3)
        np.testing.assert_allclose(pe.fmedial_bw(fy, mx, 1.83, sign), f)
    assert np.isnan(pe.logged_sign_and_height(fy, fy, np.zeros(300))[0])


def test_pn3_label_fmedial_uses_the_requested_sign():
    from pn3_redis import evaluate as pe
    bw, h = 700.0, 1.7
    jt = np.linspace(0, 1, 11)
    names = [pe.PRE + "fy", pe.PRE + "mx"]
    vals = np.column_stack([-2.0 * bw * np.ones(11), 0.03 * bw * h * np.ones(11)])
    phys = pe.label_fmedial(jt, names, vals, [0.25, 0.5], bw, pe.MX_SIGN["physical"])
    pipe = pe.label_fmedial(jt, names, vals, [0.25, 0.5], bw, pe.MX_SIGN["pipeline"])
    np.testing.assert_allclose(phys, 1.0 - 0.03 * h / D)
    np.testing.assert_allclose(pipe, 1.0 + 0.03 * h / D)


def test_twente_fmedial_from_raw_outputs():
    from analysis.thesis import twente_imu as tw
    fy, mx = np.array([-2.0, -1.0]), np.array([0.02, -0.01])
    np.testing.assert_allclose(tw.fmedial_from(fy, mx, 1.7), 0.5 * np.abs(fy) - mx * 1.7 / D)
    np.testing.assert_allclose(tw.fmedial_from(fy, mx, 1.7, "pipeline"), 0.5 * np.abs(fy) + mx * 1.7 / D)
    with pytest.raises(ValueError):
        tw.fmedial_from(fy, mx, 1.7, "checkpoint")
    src = inspect.getsource(tw.run_job) + inspect.getsource(tw.run_match_job) + inspect.getsource(tw.run_ukf)
    assert 'out["f_medial"]' not in src and 'o["f_medial"]' not in src     # never the checkpoint's own f_medial


def test_validate_synthetic_scores_labels_and_surrogates_with_one_sign(tmp_path):
    """imu/validate_synthetic.py forms the label and the prediction F_medial with the --mx-convention sign, from raw
    fy/mx, so a *_sojrqc checkpoint (mx_medial_sign -1) and a legacy one (+1) are scored alike."""
    from imu import validate_synthetic as vs
    assert vs.MX_SIGN == tm.MX_CONVENTIONS
    bw, h = 700.0, 1.7
    jt = np.linspace(0, 1, 11)
    names = [vs.PRE + "fy", vs.PRE + "mx"]
    vals = np.column_stack([-2.0 * bw * np.ones(11), 0.03 * bw * h * np.ones(11)])
    fm, fy, mx = vs.label_fmedial(jt, names, vals, np.array([-0.1, 0.25, 0.5]), bw, h, vs.MX_SIGN["physical"])
    np.testing.assert_allclose(fm[1:], 1.0 - 0.03 * h / D)
    assert np.isnan(fm[0])                                   # outside the label file's time range
    np.testing.assert_allclose(mx, 0.03)
    pipe, _, _ = vs.label_fmedial(jt, names, vals, np.array([0.5]), bw, h, vs.MX_SIGN["pipeline"])
    np.testing.assert_allclose(pipe, 1.0 + 0.03 * h / D)

    class Fake:                                              # predict_windowed's raw outputs; f_medial deliberately off
        include_contact = False

        def __init__(self, s):
            self.mx_medial_sign = s

        def predict_windowed(self, **kw):
            n = len(kw["joint_angles"])
            fy_, mx_ = -2.0 * np.ones(n), 0.03 * np.ones(n)
            return dict(fy=fy_, mx=mx_, f_medial=0.5 * np.abs(fy_) + self.mx_medial_sign * mx_ * h / D)

    T = 30
    pos = {k: np.zeros((T, 3)) for k in ("com", "pelvis", "calcn_r", "calcn_l")}
    q = np.zeros((T, 16))
    c = np.zeros(T, bool)
    for sign in (-1.0, 1.0):
        out = [vs.surrogate_fmedial(Fake(s), q, pos, c, c, 70.0, h, sign)[1] for s in (-1.0, 1.0)]
        np.testing.assert_allclose(out[0], out[1])           # the checkpoint's own sign does not enter
        np.testing.assert_allclose(out[0], 1.0 + sign * 0.03 * h / D)
    (tmp_path / "jcf_output_sojr").mkdir()
    (tmp_path / "jcf_output_sojr" / "FullDuration_JointReaction_ReactionLoads.sto").write_text("")
    assert vs.label_file(str(tmp_path), "jcf_output_sojr").endswith("FullDuration_JointReaction_ReactionLoads.sto")
    with pytest.raises(SystemExit):
        vs.label_file(str(tmp_path), "jcf_output")
    assert 'p["f_medial"]' not in inspect.getsource(vs)

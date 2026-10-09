"""Tests for tools/evaluate_fmedial.py. Run from the repo root:  python -m pytest tests -q"""
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "tools"))

import evaluate_fmedial as ev  # noqa: E402


def test_normalized_and_physical_units_agree():
    rng = np.random.default_rng(0)
    for _ in range(200):
        Fjcf, Madd = rng.uniform(0, 4, 64), rng.uniform(-0.05, 0.05, 64)
        phi, d_mm = rng.uniform(0.3, 0.7), rng.uniform(35, 55)
        H, m = rng.uniform(1.5, 2.0), rng.uniform(45, 110)
        BW = m * 9.81
        norm = ev.fmedial(Fjcf, Madd, phi, d_mm, H, m, units="normalized")
        phys = ev.fmedial(Fjcf * BW, Madd * BW * H, phi, d_mm, H, m, units="physical")
        np.testing.assert_allclose(norm * BW, phys, rtol=1e-12, atol=1e-9)


def test_matches_deployed_inference_formula():
    from inference_posture_surrogate import JCFPostureSurrogate as S
    rng = np.random.default_rng(1)
    fy, mx, H = rng.uniform(-3, 0.2, 100), rng.uniform(-0.03, 0.05, 100), 1.73
    deployed = S.DEFAULT_ALPHA * np.abs(fy) + mx * H / S.DEFAULT_D_M
    np.testing.assert_allclose(ev.fmedial(np.abs(fy), mx, 0.5, 45.0, H), deployed, rtol=1e-12, atol=1e-12)


def test_unknown_units_rejected():
    with pytest.raises(ValueError):
        ev.fmedial(1.0, 0.01, units="SI")


def test_phi_and_d_move_fmedial_the_right_way():
    Fjcf, Madd, H = np.array([2.0]), np.array([0.02]), 1.7
    assert ev.fmedial(Fjcf, Madd, 0.6, 45, H) > ev.fmedial(Fjcf, Madd, 0.5, 45, H) > ev.fmedial(Fjcf, Madd, 0.4, 45, H)
    assert ev.fmedial(Fjcf, Madd, 0.5, 40, H) > ev.fmedial(Fjcf, Madd, 0.5, 45, H) > ev.fmedial(Fjcf, Madd, 0.5, 50, H)
    # the worked example from the reference gait: 2.11 BW axial, 0.0186 BW*H moment -> about 1.76 BW
    assert abs(float(ev.fmedial(2.11, 0.0186, 0.5, 45, 1.70)) - 1.76) < 0.01


@pytest.mark.parametrize("name,expected", [
    ("carter_P019_split0_t19", "carter_P019"),
    ("han_s001_split1_t03_r02", "han_s001"),
    ("moore_subject10_t23_r08", "moore_subject10"),
    ("fregly_3GC_t00_r01", "fregly_3GC"),
    ("jcf/external/william_2026-09-23/", "william_2026-09-23"),
])
def test_person_id_merges_trials_and_splits(name, expected):
    assert ev.person_id(name) == expected


def test_basic_metrics():
    m = ev.basic_metrics([0, 1, 2, 3], [1, 1, 3, 3])
    assert m["n"] == 4 and m["mae"] == pytest.approx(0.5) and m["bias"] == pytest.approx(0.5)
    assert m["rmse"] == pytest.approx(np.sqrt(0.5))
    assert np.isnan(ev.basic_metrics([1, 1, 1], [1, 2, 3])["r"])          # constant ground truth
    assert ev.basic_metrics([], [])["n"] == 0


def test_sign_check():
    stance = np.array([1, 1, 1, 1, 0], bool)
    assert ev.sign_check(np.array([-1, -2, -1, -0.5, 3.0]), stance) == (1.0, True)
    frac, ok = ev.sign_check(np.array([1, 2, -1, 0.5, 0.0]), stance)
    assert frac == 0.25 and not ok
    frac, ok = ev.sign_check(np.array([1.0, 2.0]), np.array([0, 0], bool))
    assert np.isnan(frac) and ok


def test_block_metrics_parts_and_validity():
    T, H = 6, 1.7
    gt = np.zeros((T, 4)); pred = np.zeros((T, 4))
    gt[:, 1] = -2.0; gt[:, 3] = 0.01          # force part 1.0, moment part 0.378
    pred[:, 1] = -1.4; pred[:, 3] = 0.02      # force part 0.7, moment part 0.756
    stance = np.ones(T, bool)
    b = ev.frame_block(gt, pred, H, stance, 0.5, 45.0)
    m = ev.block_metrics(b, stance)
    assert m["force_part_bias"] == pytest.approx(-0.3)
    assert m["moment_part_bias"] == pytest.approx(0.01 * H / 0.045)
    assert m["fmed_bias"] == pytest.approx(m["force_part_bias"] + m["moment_part_bias"])
    assert m["invalid_frac_gt"] == 0.0
    pred[:, 3] = 0.08                          # moment part 3.0 > axial 1.4: physically impossible
    m2 = ev.block_metrics(ev.frame_block(gt, pred, H, stance, 0.5, 45.0), stance)
    assert m2["invalid_frac_pred"] == 1.0


PAPER_CKPT = os.path.join(REPO, "models", "mlp-grffree-base", "mlp-grffree-base.pt")
TEST_WALK = os.path.join(REPO, "jcf", "full_duration", "testing", "walking")


@pytest.mark.skipif(not (os.path.exists(PAPER_CKPT) and os.path.isdir(TEST_WALK)), reason="needs the paper checkpoint and test data")
def test_end_to_end_one_trial(tmp_path):
    out = ev.main(["--ckpt", PAPER_CKPT, "--activity", "walking", "--labs", "carter", "--max-trials", "1",
                   "--name", "pytest_smoke", "--out-root", str(tmp_path)])
    import pandas as pd
    summ = pd.read_csv(os.path.join(out, "summary.csv"))
    row = summ[(summ["group"] == "ALL") & (summ["subset"] == "all")].iloc[0]
    assert row["n_trials"] == 1 and np.isfinite(row["fmed_mae"]) and np.isfinite(row["fy_mae"])
    for f in ("persons.csv", "trials.csv", "sweep.csv", "rejected.csv", "README.md"):
        assert os.path.exists(os.path.join(out, f))

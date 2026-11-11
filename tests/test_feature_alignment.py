"""Time alignment and 100 Hz resampling of the GRF-free MLP features (train_mlp._build_com_features).

The "_t100" input sets must put the COM/foot/pelvis channels on the same instants as q, contact and labels, on a fixed
100 Hz grid; legacy sets must stay exactly as before so existing checkpoints evaluate as they were trained.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from train_mlp import (INPUT_SETS, _build_com_features, _common_time_grid, _load_com_features_aligned,  # noqa: E402
                       _sto_time, CONTACT_GRF_BW_THRESHOLD)

TRIAL = os.path.join(REPO, "jcf", "full_duration", "testing", "walking", "carter_P010_split0_t10")
LEGACY = "qinst_notrans_pelvisrel_winpos_contact_w20"
T100 = LEGACY + "_t100"
pytestmark = pytest.mark.skipif(not os.path.exists(os.path.join(TRIAL, "com_features.npz")),
                                reason="needs the labelled test trial on disk")


def _build(iset):
    return _build_com_features(TRIAL, INPUT_SETS[iset], max_peak_bw=6.0, min_length=1, predict_moment=True,
                               return_meta=True)


def test_t100_sets_registered_with_base_inputs():
    for base in ("qinst_notrans_pelvisrel_winpos_contact_w20", "qinst_notrans_pelvisrel_winpos_contact_fpopt_w20"):
        cfg = INPUT_SETS[base + "_t100"]
        assert cfg["time_exact"] and cfg["resample_hz"] == 100.0
        assert cfg["n_features_total"] == INPUT_SETS[base]["n_features_total"] == 235
        assert cfg.get("foot_point") == INPUT_SETS[base].get("foot_point")


def test_t100_grid_is_100hz_inside_the_jr_window():
    feat, labels, mass, meta = _build(T100)
    t = meta["t"]
    np.testing.assert_allclose(np.diff(t), 0.01, atol=1e-9)
    jr_t = _sto_time(os.path.join(TRIAL, "jcf_output", "BatchJCF_JointReaction_ReactionLoads.sto"))
    assert t[0] >= jr_t[0] - 1e-9 and t[-1] <= jr_t[-1] + 1e-9
    assert len(feat) == len(labels) == len(t)


def test_t100_positions_and_labels_sit_on_the_same_instants():
    feat, labels, mass, meta = _build(T100)
    t = meta["t"]
    ik = pd.read_csv(os.path.join(TRIAL, "ik_results.mot"), sep=r"\s+", skiprows=6)
    pel = np.column_stack([np.interp(t, ik["time"], ik[c]) for c in ("pelvis_tx", "pelvis_ty", "pelvis_tz")])
    pos = _load_com_features_aligned(TRIAL, t)
    q_dim = INPUT_SETS[T100]["q_dim"]
    np.testing.assert_allclose(feat[:, q_dim:q_dim + 3], pos["com_pos"] - pel, atol=2e-4)        # COM - pelvis
    np.testing.assert_allclose(feat[:, q_dim + 3:q_dim + 6], pos["calcn_r_pos"] - pel, atol=2e-4)  # right foot
    np.testing.assert_allclose(feat[:, q_dim + 6:q_dim + 9], pos["calcn_l_pos"] - pel, atol=2e-4)  # left foot
    # knee angle (q index 9 in the 16-coordinate order, 6 once the pelvis translations are dropped) at the same instants
    np.testing.assert_allclose(feat[:, 6], np.interp(t, ik["time"], ik["knee_angle_r"]), atol=1e-4)


def test_t100_contact_matches_resampled_grf():
    feat, labels, mass, meta = _build(T100)
    grf = pd.read_csv(os.path.join(TRIAL, "grf_data.mot"), sep=r"\s+", skiprows=_header_rows(os.path.join(TRIAL, "grf_data.mot")))
    col = [c for c in grf.columns if c.endswith("force_vy") and "r" in c.split("_")[1]][0]   # e.g. calcn_r_force_vy
    bw = mass * 9.81
    expect = np.abs(np.interp(meta["t"], grf["time"], grf[col])) / bw > CONTACT_GRF_BW_THRESHOLD
    q_dim = INPUT_SETS[T100]["q_dim"]
    rc = feat[:, q_dim + 9] > 0.5
    assert np.mean(rc == expect) > 0.99
    assert np.array_equal(meta["r_contact"], rc)


def test_legacy_set_unchanged():
    feat, labels, mass, meta = _build(LEGACY)
    ik_t = pd.read_csv(os.path.join(TRIAL, "ik_results.mot"), sep=r"\s+", skiprows=6)["time"].to_numpy()
    t_common = _common_time_grid(TRIAL, ik_t)
    assert len(feat) == len(t_common)                                       # native rate, JR window
    t_legacy = np.linspace(ik_t[0], ik_t[-1], len(feat))                    # the historical full-segment grid
    np.testing.assert_allclose(meta["t"], t_legacy)
    assert meta["fs"] is None


def test_legacy_misalignment_is_what_t100_removes():
    """Documents the bug: on this 8 s Carter trial the legacy grid is off by up to ~0.3 s at the ends."""
    ik_t = pd.read_csv(os.path.join(TRIAL, "ik_results.mot"), sep=r"\s+", skiprows=6)["time"].to_numpy()
    t_common = _common_time_grid(TRIAL, ik_t)
    off = np.linspace(ik_t[0], ik_t[-1], len(t_common)) - t_common
    assert off.min() < -0.25 and off.max() > 0.25


def _header_rows(path):
    with open(path) as f:
        for n, line in enumerate(f):
            if line.strip().lower() == "endheader":
                return n + 1
    return 0

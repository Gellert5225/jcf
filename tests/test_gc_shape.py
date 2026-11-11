"""Stance-shape check of the Grand Challenge run (analysis/thesis/gc_shape.py), on synthetic stances."""
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from analysis.thesis import gc_shape  # noqa: E402

H = 1.7


def _double(p):
    return 2.0 * np.exp(-((p - 25) / 10) ** 2) + 1.6 * np.exp(-((p - 75) / 10) ** 2) + 0.8


def _single(p):
    return 2.2 * np.exp(-((p - 20) / 12) ** 2) + 0.1


def _fy(fmed, phi=0.5):
    return -fmed / phi                                # M_x = 0, so F_medial = phi * |F_y|


def _frames(n_stances=6, n=101):
    rows = []
    for s in range(n_stances):
        p = np.linspace(0.5, 99.5, n)
        part = np.where((p >= 5) & (p <= 45), "early", np.where((p >= 55) & (p <= 95), "late", ""))
        rows.append(pd.DataFrame(dict(
            subject="JW", folder=f"seg{s}", stance=True, stance_id=0, pct=p, part=part, H=H,
            meas_med=_double(p), fy_good=_fy(0.8 * _double(p)), mx_good=0.0,
            fy_single=_fy(_single(p)), mx_single=0.0)))
    fr = pd.concat(rows, ignore_index=True)
    fr["fy_osim_sojr"] = _fy(_double(fr["pct"].to_numpy()) + 0.3)
    fr["mx_osim_sojr"] = 0.0
    fr.loc[fr.folder == "seg0", ["fy_osim_sojr", "mx_osim_sojr"]] = np.nan       # one segment without SO+JR
    fr["osim_sojr_ok"] = True
    return fr


def test_frame_sources_order():
    fr = _frames()
    assert gc_shape.frame_sources(fr) == ["osim_sojr", "good", "single"]
    fr["fmed_statop"] = np.nan
    assert "statop" not in gc_shape.frame_sources(fr)          # all-NaN StatOp is not a source


def test_shape_frames_identical_frames():
    fr = _frames()
    sf = gc_shape.shape_frames(fr, gc_shape.frame_sources(fr))
    assert "seg0" not in set(sf.folder)                      # dropped: a source of this subject is missing there
    assert len(sf) == 5 * 101
    for s in ("osim_sojr", "good", "single"):
        assert np.isfinite(sf[f"f_{s}"]).all()
    np.testing.assert_allclose(sf["f_good"], 0.8 * sf["meas_med"], rtol=1e-12)


def test_hump_features_double_and_single():
    mid = np.arange(2.5, 100, 5.0)
    d = gc_shape.hump_features(mid, _double(mid))
    assert d["double_hump"] and d["early_interior"] and d["late_interior"]
    assert d["early_pct"] == 22.5 or d["early_pct"] == 27.5
    assert 72.5 <= d["late_pct"] <= 77.5
    assert 0 < d["depth_rel"] < 1 and d["late_early_ratio"] < 1
    s = gc_shape.hump_features(mid, _single(mid))
    assert not s["double_hump"]
    assert not s["late_interior"]                              # monotone decline: maximum on the window's first bin
    flat = gc_shape.hump_features(mid, np.ones_like(mid))
    assert not flat["double_hump"]


def test_ensemble_and_humps():
    fr = _frames()
    srcs = gc_shape.frame_sources(fr)
    sf = gc_shape.shape_frames(fr, srcs)
    ens = gc_shape.ensemble(sf, srcs, bin_pct=5.0, min_frames=10)
    assert set(ens.source) == {"meas", "osim_sojr", "good", "single"}
    assert (ens.groupby("source").n_frames.sum() == len(sf)).all()
    hum = gc_shape.hump_table(ens).set_index("source")
    assert hum.loc["meas", "double_hump"] and hum.loc["good", "double_hump"] and hum.loc["osim_sojr", "double_hump"]
    assert not hum.loc["single", "double_hump"]
    assert hum.loc["good", "curve_r"] > 0.999 and abs(hum.loc["osim_sojr", "curve_mae"] - 0.3) < 1e-9


def test_window_shape_and_summary():
    fr = _frames()
    srcs = gc_shape.frame_sources(fr)
    sf = gc_shape.shape_frames(fr, srcs)
    halves = pd.DataFrame([dict(subject="JW", folder=f"seg{s}", stance=0, part=p, scored=True)
                           for s in range(6) for p in ("early", "late")])
    ws = gc_shape.window_shape(sf, halves, srcs)
    assert set(ws.folder) == {f"seg{s}" for s in range(1, 6)}
    assert np.allclose(ws[ws.source == "good"].r_within, 1.0)
    st = pd.DataFrame([dict(subject="JW", part="late", meas_interior=True, meas_peak_pct=75.0, peak_good=1.3,
                            interior_good=True, peak_pct_good=77.0)])
    wsum = gc_shape.window_summary(ws, st, ["good"]).set_index("part")
    assert wsum.loc["late", "n_covered"] == 5 and wsum.loc["late", "n_scored"] == 1
    assert wsum.loc["late", "timing_err_median_pct"] == 2.0


def test_run_mx_convention(tmp_path):
    import json
    assert gc_shape.run_mx_convention(str(tmp_path)) == "unknown"                   # no run.json
    (tmp_path / "run.json").write_text(json.dumps(dict(args=dict(name="x"))))
    assert gc_shape.run_mx_convention(str(tmp_path)) == "pipeline"                  # runs before the option
    (tmp_path / "run.json").write_text(json.dumps(dict(args=dict(mx_convention="physical"))))
    assert gc_shape.run_mx_convention(str(tmp_path)) == "physical"

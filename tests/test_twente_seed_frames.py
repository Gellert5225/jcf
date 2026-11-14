"""Seed-disagreement tables of the Twente check (analysis/thesis/twente_seed_frames.py) on synthetic per-frame npz.

Each checkpoint's series is a shared waveform plus a constant offset, so every MAE, bias and ratio is known."""
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from analysis.thesis import twente_seed_frames as sf  # noqa: E402

OFFSET = {"qc_s1": 0.0, "qc_s2": 0.3, "qc_s3": 0.1, "qc_origin": 0.2}     # optical path, per checkpoint
PATH_SHIFT = {"O": 0.0, "V": 0.05, "U:seg-static:grf": 0.2, "U:seg-static:gyro": 0.4}


def _write_run(root):
    os.makedirs(os.path.join(root, "frames"))
    t = np.arange(60) * 0.01
    base = 1.0 + 0.5 * np.sin(2 * np.pi * t)
    stance = (np.arange(60) % 20) < 12
    for subj in ("04", "05"):
        for trial in ("walk_18", "walk_36"):
            z = {}
            for ck, off in OFFSET.items():
                z[f"fmed|{ck}|t"] = t
                z[f"fmed|{ck}|stance_r"] = stance
                for p, shift in PATH_SHIFT.items():
                    z[f"fmed|{ck}|{p}"] = base + off + shift
            np.savez(os.path.join(root, "frames", f"Subj{subj}_{trial}_seg-static.npz"), **z)


def test_tables_from_synthetic_frames(tmp_path):
    run = str(tmp_path / "run")
    _write_run(run)
    assert sf.main([run, "--quiet"]) == 0
    trials = pd.read_csv(os.path.join(run, sf.TRIALS_CSV))
    assert len(trials) == 4 and set(trials["speed"]) == {0.5, 1.0}
    row = trials.iloc[0]
    assert np.isclose(row["seedpair_mae_s1s3|O"], 0.1)
    assert np.isclose(row["seedpair_mae_s1s2|O"], 0.3)
    assert np.isclose(row["seedpair_mae_s2x|O"], (0.3 + 0.2) / 2)
    assert np.isclose(row["foot_s1_vs_origin|O"], 0.2)
    assert np.isclose(row["qc_s1_vs_O|U_grf"], 0.2) and np.isclose(row["qc_s3_vs_O_stance|U_gyro"], 0.4)
    assert np.isclose(row["ens13_vs_O|U_gyro"], 0.4) and np.isclose(row["ens13_bias_vs_O|U_gyro"], 0.4)
    assert np.isclose(row["ens13_r_vs_O|V"], 1.0)

    table = pd.read_csv(os.path.join(run, sf.TABLE_CSV), dtype=str).set_index("metric")
    assert table.loc["seedpair_mae_s1s3|O", "all"] == "0.100 [0.100, 0.100]"
    assert table.loc["ens13_vs_O|U_grf", "1.00"] == "0.200"

    ratio = pd.read_csv(os.path.join(run, sf.RATIO_CSV)).set_index("numerator")
    assert np.isclose(ratio.loc["qc_s1_vs_O|U_grf", "ratio_of_medians"], 2.0)
    assert np.isclose(ratio.loc["qc_s3_vs_O_stance|U_gyro", "median_subject_ratio"], 4.0)
    assert int(ratio.loc["qc_s1_vs_O|V", "n_subjects"]) == 2

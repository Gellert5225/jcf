"""Non-finite labels and partial label sets (Nov 2026).

OpenSim writes a failed JointReaction frame as '            -nan' (fixed-width padded, tab-separated). pandas leaves
that as text, so the column became strings and training crashed (or, read as NaN, every trial filter compared False
and the trial passed). The loaders now read every column as float and drop a trial with any non-finite label value
before the peak, moment and flat tests. A run on a label folder that some trials lack (an unfinished or partly failed
relabel) stops unless --allow-partial-labels is given. analysis/thesis/sojr_trainset.py --gate checks coverage.
"""
import glob
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "tools"))

import train_mlp as tm  # noqa: E402
from train_cnn import (load_sto, nonfinite_rows, label_nonfinite_rows, filter_flat_subjects,  # noqa: E402
                       load_subject)
import evaluate_fmedial as ev  # noqa: E402
sys.path.insert(0, os.path.join(REPO, "analysis", "thesis"))
import sojr_trainset as st  # noqa: E402

LABEL = "BatchJCF_JointReaction_ReactionLoads.sto"
PRE = "walker_knee_r_on_tibia_r_in_tibia_r_"
FPOPT_SOJR = "qinst_notrans_pelvisrel_winpos_contact_fpopt_w20_t100_sojr"
FPOPT_CACHE = "foot_point_0.130_-0.010_0.000.npz"
PAPER_CKPT = os.path.join(REPO, "models", "mlp-grffree-base", "mlp-grffree-base.pt")
TIZIANA = os.path.join(REPO, "jcf", "full_duration", "training", "walking", "tiziana_Subject12_t03")
NAN_FIELD = " " * 12 + "-nan"          # as OpenSim writes it
BW = 70 * 9.81
CYCLIC = -BW * (1.5 + np.sin(np.linspace(0, 6 * np.pi, 200)))      # 0.5-2.5 BW, three peaks
FLAT = -BW * np.ones(200)


def write_jr(path, fy, nan_rows=(), n_rows_header=None):
    """A JointReaction .sto as OpenSim writes it: 11 header lines, values padded to 16 characters, tabs between.
    nan_rows get '-nan' in the six right-knee columns; n_rows_header overrides nRows (an incomplete file)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    n = len(fy)
    cols = ["time"] + [PRE + c for c in ("fx", "fy", "fz", "mx", "my", "mz")]
    with open(path, "w") as f:
        f.write(f"Joint Reaction Loads\nversion=1\nnRows={n if n_rows_header is None else n_rows_header}\n"
                f"nColumns={len(cols)}\ninDegrees=no\n\nThis file contains the reaction forces\n"
                "applied to the specified body\n\nUnits are S.I. units\nendheader\n")
        f.write("\t".join(cols) + "\n")
        for i in range(n):
            vals = [i * 0.01, 0.0, fy[i], 0.0, 0.0, 0.0, 0.0]
            fields = [f"{v:16.8f}" for v in vals]
            if i in nan_rows:
                fields[1:] = [NAN_FIELD] * 6
            f.write("\t".join(fields) + "\n")


def make_trial(root, name, labels):
    """Trial folder with metadata.json and one JR file per {subdir: (fy, nan_rows[, n_rows_header])}."""
    d = os.path.join(root, name)
    os.makedirs(d, exist_ok=True)
    json.dump(dict(mass_kg=70.0, height_m=1.75), open(os.path.join(d, "metadata.json"), "w"))
    for sub, spec in labels.items():
        write_jr(os.path.join(d, sub, LABEL), *spec)
    return d


# ─── Reading ──────────────────────────────────────────────────────────────────

def test_padded_nan_row_is_read_as_nan(tmp_path):
    d = make_trial(str(tmp_path), "carter_A_t01", {"jcf_output_sojr": (CYCLIC, (3,))})
    path = os.path.join(d, "jcf_output_sojr", LABEL)
    raw = pd.read_csv(path, sep="\t", skiprows=11)
    assert raw[PRE + "fy"].dtype == object                    # the bug: plain pandas leaves the column as text
    j = load_sto(path)
    assert all(pd.api.types.is_float_dtype(t) for t in j.dtypes)
    assert np.isnan(j[PRE + "fy"].iloc[3]) and np.isfinite(j[PRE + "fy"].drop(index=3)).all()
    assert j[PRE + "fy"].iloc[4] == pytest.approx(CYCLIC[4], abs=1e-6)
    assert nonfinite_rows(j) == 1 and label_nonfinite_rows(d, "jcf_output_sojr") == 1
    assert label_nonfinite_rows(d, "jcf_output") == 0         # no file: nothing to report
    assert nonfinite_rows(ev._read_sto(path)) == 1            # the evaluator reads it the same way


def test_numeric_reading_leaves_finite_files_unchanged():
    files = sorted(glob.glob(os.path.join(REPO, "jcf", "full_duration", "testing", "walking", "carter_*",
                                          "jcf_output*", LABEL)))[:4]
    if not files:
        pytest.skip("needs label files on disk")
    for p in files:
        old, new = pd.read_csv(p, sep="\t", skiprows=11), load_sto(p)
        assert list(new.columns) == [c.strip() for c in old.columns]
        np.testing.assert_array_equal(new.to_numpy(), old.to_numpy(dtype=np.float64))


def test_text_that_is_not_a_number_counts_as_nonfinite(tmp_path):
    d = make_trial(str(tmp_path), "carter_A_t01", {"jcf_output": (CYCLIC, ())})
    p = os.path.join(d, "jcf_output", LABEL)
    txt = open(p).read().split("\n")
    f = txt[12 + 5].split("\t")
    f[2] = "    nan(ind)"                                         # MSVC spelling, not in pandas' NA list
    txt[12 + 5] = "\t".join(f)
    open(p, "w").write("\n".join(txt))
    assert nonfinite_rows(load_sto(p)) == 1


# ─── Filters and discovery (synthetic) ───────────────────────────────────────

def test_flat_filter_drops_and_reports_a_nonfinite_trial(tmp_path):
    bad = make_trial(str(tmp_path), "carter_A_t01", {"jcf_output_sojr": (CYCLIC, (3,))})
    good = make_trial(str(tmp_path), "carter_B_t01", {"jcf_output_sojr": (CYCLIC, ())})
    nf = []
    kept, removed = filter_flat_subjects([bad, good], jcf_subdir="jcf_output_sojr", nonfinite=nf)
    assert kept == [good] and removed == 1 and nf == [bad]      # NaN comparisons would have kept it
    kept, removed = filter_flat_subjects([bad], jcf_subdir="jcf_output_sojr")
    assert kept == [] and removed == 1


def test_discovery_drops_nonfinite_trials_before_the_flat_filter(tmp_path):
    root = str(tmp_path)
    bad = make_trial(root, "moore_C_t01", {"jcf_output": (CYCLIC, ()), "jcf_output_sojr": (CYCLIC, (0,))})
    make_trial(root, "carter_B_t01", {"jcf_output": (CYCLIC, ()), "jcf_output_sojr": (CYCLIC, ())})
    for flat in (False, True):
        dirs, scan = tm._find_trials(root, label_subdir="jcf_output_sojr", flat_filter=flat)
        assert [os.path.basename(d) for d in dirs] == ["carter_B_t01"]
        assert scan["nonfinite"] == [bad] and scan["n_flat"] == 0 and scan["missing"] == []
    dirs, scan = tm._find_trials(root, flat_filter=True)          # the stored labels of the same folder are fine
    assert len(dirs) == 2 and scan["nonfinite"] == []


def test_rejection_reason_names_nonfinite_labels(tmp_path):
    d = make_trial(str(tmp_path), "carter_A_t01", {"jcf_output_sojr": (CYCLIC, (3, 4))})
    np.savez(os.path.join(d, "com_features.npz"), time=np.zeros(1))
    assert ev.rejection_reason(d, 6.0, (), "jcf_output_sojr", 0.15) == \
        "non-finite labels (2 of 200 rows in jcf_output_sojr/)"


# ─── Loaders on a real trial with one padded '-nan' row ───────────────────────

def _relabelled_carter_trial():
    cands = [os.path.join(REPO, "jcf", "full_duration", "testing", "walking", "carter_P010_split0_t10")] + \
        sorted(glob.glob(os.path.join(REPO, "jcf", "full_duration", "testing", "walking", "carter_*")))
    for d in cands:
        files = ("com_features.npz", FPOPT_CACHE, "metadata.json", "ik_results.mot", "grf_data.mot",
                 os.path.join("jcf_output", LABEL), os.path.join("jcf_output_sojr", LABEL))
        if all(os.path.exists(os.path.join(d, f)) for f in files) and st.read_labels(
                os.path.join(d, "jcf_output_sojr", LABEL))[1] == "complete":
            return d
    return None


TRIAL = _relabelled_carter_trial()
needs_trial = pytest.mark.skipif(TRIAL is None, reason="needs a Carter trial with jcf_output_sojr/ labels")
OPEN = dict(max_peak_bw=1e9, max_moment_bwh=1e9)


def _copy_with_nan_row(src, dst, row=10):
    """src copied to dst with data row `row` of its SO+JR file written as OpenSim writes a failed frame."""
    os.makedirs(dst)
    for f in ("ik_results.mot", "grf_data.mot", "metadata.json", "com_features.npz", FPOPT_CACHE):
        shutil.copy(os.path.join(src, f), dst)
    for sub in ("jcf_output", "jcf_output_sojr"):
        os.makedirs(os.path.join(dst, sub))
        shutil.copy(os.path.join(src, sub, LABEL), os.path.join(dst, sub, LABEL))
    path = os.path.join(dst, "jcf_output_sojr", LABEL)
    lines = open(path).read().split("\n")
    k = next(i for i, line in enumerate(lines) if line.strip() == "endheader") + 2 + row
    fields = lines[k].split("\t")
    fields[1:7] = [NAN_FIELD] * 6                               # the right knee, as in tiziana_Subject12_t03
    lines[k] = "\t".join(fields)
    open(path, "w").write("\n".join(lines))
    return dst


@needs_trial
def test_loaders_drop_a_trial_with_a_nan_label_row(tmp_path):
    bad = _copy_with_nan_row(TRIAL, str(tmp_path / os.path.basename(TRIAL)))
    kw = dict(lower_body_only=True, clean_features=True, include_mass=True, include_speed=False, predict_moment=True,
              min_length=1, native_time_grid=True, max_peak_bw=1e9, max_moment_bwh=1e9)
    assert load_subject(bad, jcf_subdir="jcf_output_sojr", **kw) is None
    assert load_subject(bad, jcf_subdir="jcf_output", **kw) is not None       # its stored labels are finite
    cfg = tm.INPUT_SETS[FPOPT_SOJR]
    assert tm._build_com_features(bad, cfg, min_length=1, predict_moment=True, **OPEN) is None
    assert tm._build_com_features(TRIAL, cfg, min_length=1, predict_moment=True, **OPEN) is not None
    ds = tm.FrameDataset([bad, TRIAL], input_set=FPOPT_SOJR, **OPEN)
    ref = tm.FrameDataset([TRIAL], input_set=FPOPT_SOJR, **OPEN)
    assert ds.n_trials == 2 and ds.rejected_dirs == [bad] and ref.rejected_dirs == []
    assert len(ds) == len(ref) > 0 and np.isfinite(ds.frames_out).all()
    np.testing.assert_array_equal(ds.frames_out, ref.frames_out)
    reason = ev.rejection_reason(bad, 1e9, [cfg["foot_point"]], "jcf_output_sojr", 1e9)
    assert reason == "non-finite labels (1 of %d rows in jcf_output_sojr/)" % len(
        load_sto(os.path.join(bad, "jcf_output_sojr", LABEL)))


@needs_trial
@pytest.mark.skipif(not os.path.exists(PAPER_CKPT), reason="needs the paper checkpoint")
def test_evaluator_lists_the_nonfinite_trial_as_rejected(tmp_path):
    bad = _copy_with_nan_row(TRIAL, str(tmp_path / "trials" / os.path.basename(TRIAL)))
    out = ev.main(["--ckpt", PAPER_CKPT, "--dirs", bad, TRIAL, "--label-subdir", "jcf_output_sojr",
                   "--max-peak", "1e9", "--max-moment", "1e9", "--out-root", str(tmp_path), "--name", "nf"])
    rej = pd.read_csv(os.path.join(out, "rejected.csv"))
    assert len(rej) == 1 and rej["reason"].iloc[0].startswith("non-finite labels (1 of ")
    assert pd.read_csv(os.path.join(out, "trials.csv"))["trial"].nunique() == 1


@pytest.mark.skipif(not os.path.exists(os.path.join(TIZIANA, "jcf_output_sojr", LABEL)),
                    reason="needs tiziana_Subject12_t03 SO+JR labels")
def test_the_real_nan_file_is_dropped():
    j = load_sto(os.path.join(TIZIANA, "jcf_output_sojr", LABEL))
    if not nonfinite_rows(j):
        pytest.skip("tiziana_Subject12_t03 has been relabelled without NaN rows")
    assert all(pd.api.types.is_float_dtype(t) for t in j.dtypes)
    assert nonfinite_rows(load_sto(os.path.join(TIZIANA, "jcf_output", LABEL))) == 0
    kw = dict(lower_body_only=True, clean_features=True, include_mass=True, predict_moment=True, min_length=1,
              native_time_grid=True, max_peak_bw=1e9, max_moment_bwh=1e9)
    assert load_subject(TIZIANA, jcf_subdir="jcf_output_sojr", **kw) is None
    nf = []
    kept, _ = filter_flat_subjects([TIZIANA], jcf_subdir="jcf_output_sojr", nonfinite=nf)
    assert kept == [] and nf == [TIZIANA]


# ─── Partial label sets stop train() and the evaluator ────────────────────────

class _NoFrames:
    """FrameDataset stand-in: records the trials train() would load, then makes train() stop (no frames)."""
    seen = []

    def __init__(self, dirs, **kw):
        _NoFrames.seen.append(list(dirs))
        self.n_trials, self.rejected_dirs = len(dirs), []
        self.frames_in, self.frames_out = np.zeros((0, 1), np.float32), np.zeros((0, 4), np.float32)

    def __len__(self):
        return 0


@pytest.fixture
def walking_root(tmp_path, monkeypatch):
    root = str(tmp_path / "walking")
    make_trial(root, "carter_A_t01", {"jcf_output": (CYCLIC, ())})                              # not relabelled
    for name in ("carter_B_t01", "carter_D_t01", "carter_E_t01", "moore_F_t01"):
        make_trial(root, name, {"jcf_output": (CYCLIC, ()), "jcf_output_sojr": (CYCLIC, ())})
    make_trial(root, "moore_C_t01", {"jcf_output": (CYCLIC, ()), "jcf_output_sojr": (CYCLIC, (5,))})
    monkeypatch.setitem(tm.DATA_ROOTS, "walking", root)
    monkeypatch.setattr(tm, "CKPT_ROOT", str(tmp_path / "ckpt"))
    monkeypatch.setattr(tm, "FrameDataset", _NoFrames)
    _NoFrames.seen = []
    return root


def test_train_stops_on_a_partial_label_set(walking_root, capsys):
    with pytest.raises(SystemExit, match=r"1 of 6 folders with jcf_output/ labels have no jcf_output_sojr/ labels "
                                         r"\(carter 1; e\.g\. carter_A_t01\).*--allow-partial-labels"):
        tm.train(activity="walking", input_set=FPOPT_SOJR)
    assert _NoFrames.seen == []                                   # stopped before loading anything
    tm.train(activity="walking", input_set=FPOPT_SOJR, allow_partial_labels=True)
    loaded = [os.path.basename(d) for d in _NoFrames.seen[0]]
    assert "carter_A_t01" not in loaded and "moore_C_t01" not in loaded
    out = capsys.readouterr().out
    assert "WARNING (--allow-partial-labels): 1 of 6 folders" in out
    assert "1 folders have non-finite label values; dropped (moore 1; e.g. moore_C_t01)" in out


def test_train_stops_on_extra_dirs_without_the_labels(walking_root):
    a = os.path.join(walking_root, "carter_A_t01")
    with pytest.raises(SystemExit, match=r"--extra-dirs: 1 of 1 folders"):
        tm.train(activity="walking", input_set=FPOPT_SOJR, extra_dirs=[a], allow_partial_labels=False)
    tm.train(activity="walking", input_set=FPOPT_SOJR.replace("_sojr", ""), extra_dirs=[a])  # stored labels: fine


def test_train_flat_filter_can_judge_the_stored_labels(walking_root, capsys):
    # moore_F is flat only in its SO+JR labels
    make_trial(walking_root, "moore_F_t01", {"jcf_output": (CYCLIC, ()), "jcf_output_sojr": (FLAT, ())})
    tm.train(activity="walking", input_set=FPOPT_SOJR, allow_partial_labels=True)
    assert "moore_F_t01" not in [os.path.basename(d) for d in _NoFrames.seen[-1]]
    tm.train(activity="walking", input_set=FPOPT_SOJR, allow_partial_labels=True, flat_label_subdir="jcf_output")
    out = capsys.readouterr().out
    assert "flat filter (jcf_output/ labels): 4 → 4 (removed 0)" in out
    assert "flat filter (jcf_output_sojr/ labels): 4 → 3 (removed 1)" in out


@pytest.mark.skipif(not os.path.exists(PAPER_CKPT), reason="needs the paper checkpoint")
def test_evaluator_split_run_stops_on_a_partial_label_set(tmp_path, monkeypatch, walking_root):
    import torch
    ck = torch.load(PAPER_CKPT, map_location="cpu", weights_only=True)
    fake = str(tmp_path / "sojr.pt")
    torch.save(dict(ck, label_subdir="jcf_output_sojr"), fake)
    monkeypatch.setitem(ev.SPLIT_ROOTS, "test", os.path.dirname(walking_root))
    args = ["--ckpt", fake, "--activity", "walking", "--out-root", str(tmp_path / "out"), "--name", "x"]
    with pytest.raises(SystemExit, match=r"1 folders have jcf_output/ labels but no jcf_output_sojr/ labels "
                                         r"\(carter 1; e\.g\. carter_A_t01\).*--allow-partial-labels"):
        ev.main(args)
    assert not os.path.exists(tmp_path / "out" / "x")
    with pytest.raises(SystemExit, match="no trials could be evaluated"):     # past the check; no features here
        ev.main(args + ["--allow-partial-labels"])


# ─── analysis/thesis/sojr_trainset.py ─────────────────────────────────────────

@pytest.fixture
def data_root(tmp_path):
    data = str(tmp_path / "full")
    tw = os.path.join(data, "training", "walking")
    make_trial(tw, "carter_A_t01", {"jcf_output": (CYCLIC, ())})                                  # missing
    make_trial(tw, "carter_B_t01", {"jcf_output": (CYCLIC, ()), "jcf_output_sojr": (CYCLIC, ())})
    make_trial(tw, "moore_C_t01", {"jcf_output": (CYCLIC, ()), "jcf_output_sojr": (CYCLIC, (2,))})  # non-finite
    make_trial(tw, "han_D_t01", {"jcf_output": (CYCLIC, ()), "jcf_output_sojr": (CYCLIC, (), 300)})  # incomplete
    make_trial(tw, "tiziana_E_t01", {"jcf_output": (FLAT, ()), "jcf_output_sojr": (CYCLIC * 3, ())})  # flat stored
    make_trial(os.path.join(data, "testing", "walking"), "tiziana_F_t01",
               {"jcf_output": (CYCLIC, ()), "jcf_output_sojr": (CYCLIC, ())})
    os.makedirs(os.path.join(tw, "moore_G_t01"))                                                 # unlabelled: ignored
    return data


def test_gate_passes_only_when_every_labelled_folder_has_complete_labels(data_root, tmp_path, capsys):
    common = ["--data", data_root, "--summary", str(tmp_path / "none.csv"), "--workers", "1"]
    assert st.main(["--gate"] + common) == 1
    out = capsys.readouterr().out
    assert "training/walking/carter: 1 of 2 lack" in out and "training/walking/han: 1 of 1 lack" in out
    assert "non-finite jcf_output_sojr/ labels (1 rows" in out and "NOT READY: 2 folders" in out
    assert st.main(["--gate", "--labs", "moore", "tiziana"] + common) == 0    # non-finite alone does not fail
    assert st.main(["--gate", "--exclude", "carter", "--split", "testing"] + common) == 0
    assert st.main(["--gate", "--exclude", "carter", "--out", str(tmp_path / "gate")] + common) == 1   # han_D
    assert not os.path.exists(tmp_path / "gate")                              # the gate writes nothing


def test_filter_tables_compare_the_same_trials(data_root, tmp_path):
    summ = tmp_path / "relabel.csv"
    pd.DataFrame(dict(folder=["moore_x_t00", "moore_y_t00", "carter_z_t00"],
                      status=["ok", "failed: RuntimeError: ... Cannot open file /Users/someone/x.osim It may", "exists"])
                 ).to_csv(summ, index=False)
    out = str(tmp_path / "out")
    assert st.main(["--data", data_root, "--summary", str(summ), "--workers", "1", "--out", out]) == 0
    trials = pd.read_csv(os.path.join(out, "trials.csv")).set_index("trial")
    assert trials.loc["carter_A_t01", "sojr_state"] == "missing" and trials.loc["han_D_t01", "sojr_state"] == "incomplete"
    assert trials.loc["moore_C_t01", "sojr_nonfinite_rows"] == 1 and np.isnan(trials.loc["moore_C_t01", "sojr_flat"])
    assert bool(trials.loc["tiziana_E_t01", "stored_flat"]) and not bool(trials.loc["tiziana_E_t01", "sojr_flat"])
    # the verdicts are the training code's
    d = os.path.join(data_root, "training", "walking", "tiziana_E_t01")
    assert filter_flat_subjects([d], jcf_subdir="jcf_output")[0] == [] and \
        filter_flat_subjects([d], jcf_subdir="jcf_output_sojr")[0] == [d]
    assert trials.loc["tiziana_E_t01", "sojr_peak_bw"] == pytest.approx(7.5, abs=1e-3)
    flat = pd.read_csv(os.path.join(out, "flat_same_trials.csv")).set_index(["split", "lab"])
    assert flat.loc[("training", "tiziana"), ["n_same", "flat_stored", "flat_sojr", "flat_only_stored"]].tolist() == \
        [1, 1, 0, 1]
    assert flat.loc[("training", "ALL"), "n_same"] == 2            # carter_B and tiziana_E (moore_C non-finite)
    assert ("training", "moore") not in flat.index
    peak = pd.read_csv(os.path.join(out, "peak_same_trials.csv")).set_index(["split", "lab"])
    assert peak.loc[("training", "tiziana"), ["stored_peak_gt", "sojr_peak_gt"]].tolist() == [0, 1]
    cov = pd.read_csv(os.path.join(out, "coverage.csv"))
    tot = cov[cov["lab"] == "ALL"].iloc[0]
    assert (tot["n_stored"], tot["sojr_complete"], tot["sojr_missing"], tot["sojr_incomplete"], tot["sojr_nonfinite"]) \
        == (6, 4, 1, 1, 1)
    status = pd.read_csv(os.path.join(out, "relabel_status.csv"))
    assert set(status["status"]) == {"ok", "exists", "failed: setup points to a /Users/ (macOS) model path"}
    assert "NOT READY" in open(os.path.join(out, "README.md")).read()

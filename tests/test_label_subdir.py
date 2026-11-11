"""Label-folder plumbing (Nov 2026): the *_sojr input sets train and evaluate on jcf_output_sojr/ (static optimization +
joint reaction, analysis/thesis/gc_knee_jr.py), every other set on the stored jcf_output/ labels, and
tools/evaluate_fmedial.py --label-subdir scores any checkpoint against either. No label-quality rule is chosen here:
the data tests switch the trial filters off.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "tools"))

from train_mlp import (INPUT_SETS, DEFAULT_LABEL_SUBDIR, SOJR_LABEL_SUBDIR, _build_com_features,  # noqa: E402
                       _common_time_grid, _find_trials, _label_config, _scan_dirs, label_subdir_of)
from train_cnn import load_subject  # noqa: E402
import evaluate_fmedial as ev  # noqa: E402

LABEL = "BatchJCF_JointReaction_ReactionLoads.sto"
PRE = "walker_knee_r_on_tibia_r_in_tibia_r_"
LEGACY = "qinst_notrans_pelvisrel_winpos_contact_w20"
T100 = LEGACY + "_t100"
FPOPT_T100 = "qinst_notrans_pelvisrel_winpos_contact_fpopt_w20_t100"
FPOPT_CACHE = "foot_point_0.130_-0.010_0.000.npz"
NO_FILTER = dict(max_peak_bw=1e9, max_moment_bwh=1e9)       # plumbing only: no label-quality rule
PAPER_CKPT = os.path.join(REPO, "models", "mlp-grffree-base", "mlp-grffree-base.pt")
PREFERRED = os.path.join(REPO, "jcf", "full_duration", "testing", "walking", "carter_P010_split0_t10")


def _complete(path):
    """The .sto exists and has all nRows rows (gc_knee_jr.py may be writing it while the tests run)."""
    try:
        with open(path) as f:
            lines = f.read().splitlines()
        n = int(next(line.split("=", 1)[1] for line in lines if line.startswith("nRows=")))
        hdr = next(i for i, line in enumerate(lines) if line.strip() == "endheader")
        return n > 0 and len([line for line in lines[hdr + 2:] if line.strip()]) == n
    except (OSError, StopIteration, ValueError):
        return False


def _relabelled_trial():
    """carter_P010_split0_t10 once it has SO+JR labels, else the first relabelled Carter training trial."""
    cands = [PREFERRED] + sorted(glob.glob(os.path.join(REPO, "jcf", "full_duration", "training", "walking", "carter_*")))
    for d in cands:
        if all(os.path.exists(os.path.join(d, f)) for f in ("com_features.npz", FPOPT_CACHE, "metadata.json")) \
                and _complete(os.path.join(d, SOJR_LABEL_SUBDIR, LABEL)):
            return d
    return None


TRIAL = _relabelled_trial()
needs_trial = pytest.mark.skipif(TRIAL is None, reason="needs a Carter trial with jcf_output_sojr/ labels on disk")


def _labels_from_file(d, subdir, t, via_ik=False):
    """[Fx, Fy, Fz] / BW and M_x / (BW*H) of the right knee, straight from the .sto, on the times t.

    via_ik: first onto the IK frames inside the JR window, then onto t, which is how the loaders reach a 100 Hz grid
    (load_subject(native_time_grid=True), then the resampling in _build_com_features)."""
    meta = json.load(open(os.path.join(d, "metadata.json")))
    bw, h = meta["mass_kg"] * 9.81, meta["height_m"]
    j = ev._read_sto(os.path.join(d, subdir, LABEL))
    cols = [j[PRE + c] / bw for c in ("fx", "fy", "fz")] + [j[PRE + "mx"] / (bw * h)]
    if via_ik:
        ik_t = pd.read_csv(os.path.join(d, "ik_results.mot"), sep=r"\s+", skiprows=6)["time"].to_numpy()
        t_ik = _common_time_grid(d, ik_t, subdir)
        cols = [np.interp(t_ik, j["time"], c) for c in cols]
        return np.column_stack([np.interp(t, t_ik, c) for c in cols]), bw, h
    return np.column_stack([np.interp(t, j["time"], c) for c in cols]), bw, h


# ─── Input sets ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("base", [T100, FPOPT_T100])
def test_sojr_sets_are_their_t100_base_with_sojr_labels(base):
    cfg, ref = INPUT_SETS[base + "_sojr"], INPUT_SETS[base]
    assert label_subdir_of(cfg) == SOJR_LABEL_SUBDIR == "jcf_output_sojr"
    strip = lambda c: {k: v for k, v in c.items() if k not in ("label_subdir", "desc")}  # noqa: E731
    assert strip(cfg) == strip(ref)
    from inference_posture_surrogate import JCFPostureSurrogate
    assert base + "_sojr" in JCFPostureSurrogate.SUPPORTED_INPUT_SETS


FPOPT_CKPT = os.path.join(REPO, "models", "runs", "mlp",
                          "best_model_mlp_walking_qinst_notrans_pelvisrel_winpos_contact_fpopt_w20_fmW0p5_e100.pt")


@pytest.mark.skipif(not os.path.exists(FPOPT_CKPT), reason="needs the fpopt checkpoint")
def test_inference_module_loads_a_sojr_checkpoint(tmp_path):
    """A *_sojr checkpoint has the fpopt network shape (235 ch, no pelvis translation, contact flags)."""
    import torch
    from inference_posture_surrogate import JCFPostureSurrogate
    ck = torch.load(FPOPT_CKPT, map_location="cpu", weights_only=True)
    fake = str(tmp_path / "fpopt_t100_sojr.pt")
    torch.save(dict(ck, input_set=FPOPT_T100 + "_sojr", label_subdir="jcf_output_sojr"), fake)
    m = JCFPostureSurrogate(fake, device="cpu")
    assert m.n_features == 235 and m.drop_pelvis_translation and m.include_contact
    np.testing.assert_allclose(m.foot_point, (0.13, -0.01, 0.0))


def test_every_other_set_reads_the_stored_labels():
    for name, cfg in INPUT_SETS.items():
        if not name.endswith(("_sojr", "_sojrqc")):
            assert "label_subdir" not in cfg and label_subdir_of(cfg) == DEFAULT_LABEL_SUBDIR == "jcf_output", name


def test_label_config_is_plain_and_complete():
    import provenance
    c = _label_config(FPOPT_T100 + "_sojr", 8.0, 0.2, True, 0.25, 0.3)
    assert c == dict(label_subdir="jcf_output_sojr", max_peak_bw=8.0, max_moment_bwh=0.2, flat_filter=True,
                     flat_cv=0.25, flat_dr=0.3, flat_label_subdir="jcf_output_sojr", nonfinite_rejected=True,
                     allow_partial_labels=False, label_mask=None, label_mask_key=None, label_mask_tol=None)
    assert provenance.plain(c) == c                       # loads with torch.load(weights_only=True)
    off = _label_config(LEGACY, 6.0, 0.15, False, 0.25, 0.3)
    assert off["label_subdir"] == "jcf_output" and off["flat_filter"] is False and off["flat_cv"] is None
    assert off["flat_label_subdir"] is None
    stored_flat = _label_config(FPOPT_T100 + "_sojr", 8.0, 0.2, True, 0.25, 0.3, "jcf_output", True)
    assert stored_flat["flat_label_subdir"] == "jcf_output" and stored_flat["allow_partial_labels"] is True


# ─── Trial discovery and the flat filter (synthetic folders) ──────────────────

def _write_jr(path, fy, n=200):
    """A JointReaction .sto in OpenSim's layout (11 header lines, as train_cnn.load_sto expects)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    t = np.arange(n) * 0.01
    df = pd.DataFrame({"time": t, PRE + "fx": np.zeros(n), PRE + "fy": fy, PRE + "fz": np.zeros(n),
                       PRE + "mx": np.zeros(n), PRE + "my": np.zeros(n), PRE + "mz": np.zeros(n)})
    with open(path, "w") as f:
        f.write(f"Joint Reaction Loads\nversion=1\nnRows={n}\nnColumns=7\ninDegrees=no\n\n"
                "This file contains the reaction forces\napplied to the specified body\n\nUnits are S.I. units\nendheader\n")
        df.to_csv(f, sep="\t", index=False)


def test_discovery_and_flat_filter_follow_the_label_folder(tmp_path):
    bw = 70 * 9.81
    cyclic = -bw * (1.5 + np.sin(np.linspace(0, 6 * np.pi, 200)))     # 0.5-2.5 BW, three peaks
    flat = -bw * np.ones(200)
    folders = {"carter_A_t01": dict(jcf_output=cyclic),                              # stored labels only
               "carter_B_t01": dict(jcf_output=cyclic, jcf_output_sojr=cyclic),
               "moore_C_t01": dict(jcf_output=cyclic, jcf_output_sojr=flat)}         # flat only in SO+JR
    for name, subdirs in folders.items():
        os.makedirs(tmp_path / name)
        json.dump(dict(mass_kg=70.0, height_m=1.75), open(tmp_path / name / "metadata.json", "w"))
        for sub, fy in subdirs.items():
            _write_jr(str(tmp_path / name / sub / LABEL), fy)
    names = lambda ds: [os.path.basename(d) for d in ds]  # noqa: E731
    assert names(_scan_dirs(str(tmp_path))) == ["carter_A_t01", "carter_B_t01", "moore_C_t01"]
    assert names(_scan_dirs(str(tmp_path), label_subdir="jcf_output_sojr")) == ["carter_B_t01", "moore_C_t01"]
    dirs, scan = _find_trials(str(tmp_path), label_subdir="jcf_output_sojr", flat_filter=True)
    assert names(dirs) == ["carter_B_t01"]
    assert (scan["n_found"], scan["n_stored"], scan["n_flat"], names(scan["missing"]), scan["nonfinite"]) == \
        (2, 3, 1, ["carter_A_t01"], [])
    assert scan["flat_label_subdir"] == "jcf_output_sojr"
    dirs, scan = _find_trials(str(tmp_path), flat_filter=True)                            # stored: nothing flat
    assert len(dirs) == 3 and (scan["n_found"], scan["n_flat"], scan["missing"]) == (3, 0, [])
    dirs, _ = _find_trials(str(tmp_path), label_subdir="jcf_output_sojr", flat_filter=True, flat_dr=0.0, flat_cv=0.0)
    assert len(dirs) == 2                                                                 # thresholds reach the filter
    dirs, _ = _find_trials(str(tmp_path), exclude=["moore"], label_subdir="jcf_output_sojr")
    assert names(dirs) == ["carter_B_t01"]
    # the flat filter can judge the stored labels instead: moore_C (flat only in SO+JR) stays
    dirs, scan = _find_trials(str(tmp_path), label_subdir="jcf_output_sojr", flat_filter=True,
                              flat_label_subdir="jcf_output")
    assert names(dirs) == ["carter_B_t01", "moore_C_t01"] and scan["n_flat"] == 0
    assert scan["flat_label_subdir"] == "jcf_output"
    # a partial label set stops discovery unless allowed, naming the labs and the flag
    with pytest.raises(SystemExit, match=r"1 of 3 folders .*carter 1.*--allow-partial-labels"):
        _find_trials(str(tmp_path), label_subdir="jcf_output_sojr", flat_filter=True, allow_partial_labels=False)
    _find_trials(str(tmp_path), label_subdir="jcf_output", allow_partial_labels=False)    # stored: never partial


# ─── Features and labels on a relabelled trial ────────────────────────────────

@needs_trial
@pytest.mark.parametrize("base", [T100, FPOPT_T100])
def test_sojr_set_gives_sojr_labels_and_the_same_inputs(base):
    ref = _build_com_features(TRIAL, INPUT_SETS[base], min_length=1, predict_moment=True, return_meta=True, **NO_FILTER)
    new = _build_com_features(TRIAL, INPUT_SETS[base + "_sojr"], min_length=1, predict_moment=True, return_meta=True,
                              **NO_FILTER)
    assert ref is not None and new is not None
    (f0, y0, m0, meta0), (f1, y1, m1, meta1) = ref, new
    np.testing.assert_array_equal(f1, f0)                         # inputs do not depend on the label folder
    np.testing.assert_array_equal(meta1["t"], meta0["t"])
    exp_new, _, _ = _labels_from_file(TRIAL, "jcf_output_sojr", meta1["t"], via_ik=True)
    exp_old, _, _ = _labels_from_file(TRIAL, "jcf_output", meta0["t"], via_ik=True)
    np.testing.assert_allclose(y1, exp_new, atol=1e-5)            # the SO+JR file, on the 100 Hz grid
    np.testing.assert_allclose(y0, exp_old, atol=1e-5)            # the stored file
    assert np.mean(np.abs(y1[:, 1] - y0[:, 1])) > 0.05            # the two label sets really differ here


@needs_trial
def test_legacy_set_still_reads_the_stored_labels():
    built = _build_com_features(TRIAL, INPUT_SETS[LEGACY], max_peak_bw=1e9, min_length=1, predict_moment=True,
                                max_moment_bwh=1e9)
    old = load_subject(TRIAL, lower_body_only=True, clean_features=True, include_mass=True, include_speed=False,
                       predict_moment=True, max_peak_bw=1e9, max_moment_bwh=1e9, min_length=1, native_time_grid=True)
    np.testing.assert_array_equal(built[1], old[1].astype(np.float32))
    explicit = _build_com_features(TRIAL, dict(INPUT_SETS[LEGACY], label_subdir="jcf_output"), max_peak_bw=1e9,
                                   min_length=1, predict_moment=True, max_moment_bwh=1e9)
    np.testing.assert_array_equal(explicit[0], built[0])
    np.testing.assert_array_equal(explicit[1], built[1])


@needs_trial
def test_moment_limit_reaches_the_loader():
    cfg = INPUT_SETS[T100 + "_sojr"]
    _, y, _ = _build_com_features(TRIAL, cfg, min_length=1, predict_moment=True, **NO_FILTER)
    _, bw, h = _labels_from_file(TRIAL, "jcf_output_sojr", np.array([0.0]))
    j = ev._read_sto(os.path.join(TRIAL, "jcf_output_sojr", LABEL))
    mx_peak = float(np.abs(j[PRE + "mx"]).max() / (bw * h))
    assert _build_com_features(TRIAL, cfg, 1e9, 1, True, max_moment_bwh=mx_peak * 0.99) is None
    assert _build_com_features(TRIAL, cfg, 1e9, 1, True, max_moment_bwh=mx_peak * 1.01) is not None


# ─── Evaluator ────────────────────────────────────────────────────────────────

def test_resolve_label_subdir():
    assert ev.resolve_label_subdir(["jcf_output", "jcf_output"]) == "jcf_output"
    assert ev.resolve_label_subdir(["jcf_output_sojr"]) == "jcf_output_sojr"
    assert ev.resolve_label_subdir(["jcf_output", "jcf_output_sojr"], "jcf_output_sojr/") == "jcf_output_sojr"
    with pytest.raises(SystemExit):
        ev.resolve_label_subdir(["jcf_output", "jcf_output_sojr"])


@pytest.mark.skipif(not os.path.exists(PAPER_CKPT), reason="needs the paper checkpoint")
def test_checkpoints_trained_on_different_labels_need_label_subdir(tmp_path):
    import torch
    ck = torch.load(PAPER_CKPT, map_location="cpu", weights_only=True)
    fake = str(tmp_path / "sojr_trained.pt")
    torch.save(dict(ck, label_subdir="jcf_output_sojr"), fake)
    assert ev.load_model(PAPER_CKPT, "cpu")[3]["label_subdir"] == "jcf_output"     # pre-Nov 2026: stored labels
    assert ev.load_model(fake, "cpu")[3]["label_subdir"] == "jcf_output_sojr"
    with pytest.raises(SystemExit, match="--label-subdir"):
        ev.main(["--ckpt", PAPER_CKPT, fake, "--dirs", str(tmp_path), "--name", "x", "--out-root", str(tmp_path)])


@needs_trial
@pytest.mark.skipif(not os.path.exists(PAPER_CKPT), reason="needs the paper checkpoint")
def test_evaluate_label_subdir_scores_a_stored_label_checkpoint_against_sojr(tmp_path):
    common = ["--ckpt", PAPER_CKPT, "--dirs", TRIAL, "--max-peak", "1e9", "--max-moment", "1e9",
              "--out-root", str(tmp_path)]
    out_old = ev.main(common + ["--name", "stored"])
    out_new = ev.main(common + ["--name", "sojr", "--label-subdir", "jcf_output_sojr"])
    readme = open(os.path.join(out_new, "README.md")).read()
    assert "labels: jcf_output_sojr/" in readme and "(--label-subdir)" in readme
    assert "labels: jcf_output/" in open(os.path.join(out_old, "README.md")).read()
    row = lambda out, s: pd.read_csv(os.path.join(out, "trials.csv")).query("subset == @s").iloc[0]  # noqa: E731
    # Ground truth = the SO+JR file: the right-stance F_medial peak recomputed from it on the trial's frames
    ik_t = pd.read_csv(os.path.join(TRIAL, "ik_results.mot"), sep=r"\s+", skiprows=6)["time"].to_numpy()
    t = _common_time_grid(TRIAL, ik_t, "jcf_output_sojr")
    lab, bw, h = _labels_from_file(TRIAL, "jcf_output_sojr", t)
    grf = ev._read_sto(os.path.join(TRIAL, "grf_data.mot"))
    stance = np.abs(np.interp(t, grf["time"], grf["calcn_r_force_vy"])) / bw > ev.CONTACT_BW
    # the evaluator's default --mx-convention physical: medial moment = -mx of walker_knee_r
    peak = (0.5 * np.abs(lab[:, 1]) - lab[:, 3] * h / 0.045)[stance].max()
    assert row(out_new, "stance")["peak_gt"] == pytest.approx(peak, abs=1e-3)
    assert row(out_old, "stance")["peak_gt"] != pytest.approx(peak, abs=1e-2)
    # same checkpoint, same inputs: only the labels moved
    assert row(out_new, "all")["n_frames"] == row(out_old, "all")["n_frames"]
    assert row(out_new, "all")["fy_mae"] != pytest.approx(row(out_old, "all")["fy_mae"], abs=1e-3)

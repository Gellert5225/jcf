"""SO+JR label-QC frame masks (Nov 2026).

B: `analysis/thesis/so_label_qc.py masks` writes <trial>/jcf_output_sojr/label_mask.npz on the JR time grid with keep_r /
keep_l = no fixed COP on that foot and none of the six SO frame rules, the rule flags, and fy5 as a separate flag.
C: the *_sojrqc input sets carry keep_r onto the 100 Hz frame grid (train_mlp.carry_label_mask), FrameDataset drops
the rows whose centre frame is masked (inputs keep their full windows), folders without a mask stop the run, and
tools/evaluate_fmedial.py --label-mask scores the kept frames only.
"""
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS = os.path.dirname(os.path.abspath(__file__))
for p in (REPO, os.path.join(REPO, "tools"), TESTS):
    if p not in sys.path:
        sys.path.insert(0, p)

import train_mlp as tm  # noqa: E402
import evaluate_fmedial as ev  # noqa: E402
from analysis.thesis import so_label_qc as q  # noqa: E402
import test_so_label_qc as tq  # noqa: E402  (synthetic trial folder builder)
sys.path.insert(0, os.path.join(REPO, "analysis", "thesis"))

QC_FPOPT = "qinst_notrans_pelvisrel_winpos_contact_fpopt_w20_t100_sojrqc"
SOJR_FPOPT = "qinst_notrans_pelvisrel_winpos_contact_fpopt_w20_t100_sojr"
REAL = os.path.join(REPO, "jcf", "full_duration", "testing", "walking", "carter_P010_split0_t10")
REAL_FILES = ("ik_results.mot", "grf_data.mot", "metadata.json", "com_features.npz", "foot_point_0.130_-0.010_0.000.npz")
PAPER_CKPT = os.path.join(REPO, "models", "mlp-grffree-base", "mlp-grffree-base.pt")


def _set_fy(d, frame, fy_bw):
    """Put the right-knee F_y of one JR row of a synthetic folder at fy_bw (BW)."""
    path = os.path.join(d, q.OUT_SUBDIR, q.JR_FILE)
    jr = q.read_sto(path)
    jr.loc[frame, q.knee_prefix("r") + "fy"] = fy_bw * tq.BW
    tq._write_sto(path, jr, "JR")


# ─── B: mask contents ─────────────────────────────────────────────────────────

def test_mask_rules_keep_and_fy5_flag(tmp_path):
    d = tq._folder(tmp_path)
    _set_fy(d, 2, -5.5)                       # an otherwise clean frame above 5 BW
    m = q.mask_arrays(d)
    jr = q.read_sto(os.path.join(d, q.OUT_SUBDIR, q.JR_FILE))
    np.testing.assert_array_equal(m["time"], jr["time"].to_numpy())
    assert list(m["rules"]) == ["sat_knee", "sat_leg3", "res_leg", "pelvis_m", "jump", "coact"]
    # synthetic folder: vaslat saturated on 7, three leg muscles on 8, ankle reserve on 9, co-activation on 10,
    # pelvis residual on 11, activation jump into 12 and out of 13 (and into / out of the other single-frame
    # activations 7-11); frame 9 also has |F_y| = 6 BW
    expect = {"sat_knee": {7, 8}, "sat_leg3": {8}, "res_leg": {9}, "pelvis_m": {11}, "jump": set(range(7, 14)),
              "coact": {10}}
    for r, frames in expect.items():
        assert set(np.flatnonzero(m[f"{r}_r"])) == frames, r
    assert set(np.flatnonzero(~m["keep_r"])) == {7, 8, 9, 10, 11, 12, 13}
    np.testing.assert_array_equal(m["so_any_r"], ~m["keep_r"])
    assert set(np.flatnonzero(m["fy5_r"])) == {2, 9}
    assert m["keep_r"][2]                     # the 5 BW rule is stored, not applied
    assert set(np.flatnonzero(~m["keep_l"])) == {11}    # only the pelvis residual reaches the left knee
    assert not m["fixed_cop_r"] and not m["fixed_cop_l"] and m["stance_r"].sum() == 10
    th = json.loads(str(m["thresholds"]))
    assert th["FIXED_COP_MPS"] == 0.2 and th["SAT_ACT"] == q.SAT_ACT and th["FY_BW"] == 5.0
    # summary: the 5 BW rule would remove frame 2 after the SO rules; frame 9 is already masked
    row = q.mask_summary(d, m)
    assert row["n_fy5_r"] == 2 and row["n_fy5_kept_r"] == 1 and row["n_keep_r"] == 13
    assert row["n_keep_stance_r"] == 10 - 7 and row["n_fy5_kept_stance_r"] == 0   # frame 2 is right swing


def test_fixed_cop_masks_that_knee_only(tmp_path):
    d = tq._folder(tmp_path, fixed_cop=True, fixed_side="r", long_left=True)
    m = q.mask_arrays(d)
    assert bool(m["fixed_cop_r"]) and not bool(m["fixed_cop_l"])
    assert not m["keep_r"].any()
    assert m["keep_l"].sum() == len(m["time"]) - 1         # left: only the pelvis-residual frame
    assert m["cop_speed_r"] < q.FIXED_COP_MPS <= m["cop_speed_l"]


def test_static_folders_never_get_a_fixed_cop(tmp_path):
    d = tq._folder(tmp_path / "static", fixed_cop=True)
    assert os.path.basename(os.path.dirname(d)) == "static"
    m = q.mask_arrays(d)
    assert not m["fixed_cop_r"] and m["keep_r"].sum() == len(m["time"]) - 7


def test_full_duration_file_names(tmp_path):
    """William's folder: jcf_output_sojr/FullDuration_JointReaction_ReactionLoads.sto (and FullDuration_ SO files)."""
    d = tq._folder(tmp_path)
    ref = q.mask_arrays(d)
    for sub, name in ((q.OUT_SUBDIR, q.JR_FILE), ("jcf_output", q.SO_FORCES), ("jcf_output", q.SO_ACT)):
        os.rename(os.path.join(d, sub, name), os.path.join(d, sub, name.replace("BatchJCF_", "FullDuration_")))
    m = q.mask_arrays(d)
    assert str(m["label_file"]) == os.path.join(q.OUT_SUBDIR, "FullDuration_JointReaction_ReactionLoads.sto")
    for k in ("time", "keep_r", "keep_l", "fy5_r"):
        np.testing.assert_array_equal(m[k], ref[k])
    from train_cnn import jr_label_path
    assert jr_label_path(d, q.OUT_SUBDIR).endswith("FullDuration_JointReaction_ReactionLoads.sto")


def test_masks_command_writes_loadable_files(tmp_path):
    root = tmp_path / "training" / "walking"
    d1 = tq._folder(root / "a")
    d1 = shutil.move(d1, str(root / "carter_P001_split0_t00"))
    d2 = tq._folder(root / "b", fixed_cop=True, fixed_side="r")
    d2 = shutil.move(d2, str(root / "moore_subject1_t00"))
    shutil.rmtree(root / "a"); shutil.rmtree(root / "b")
    out = tmp_path / "summary"
    q.main(["masks", "--root", str(root), "--workers", "1", "--out", str(out), "--dry-run"])
    assert not os.path.exists(os.path.join(d1, q.OUT_SUBDIR, q.MASK_FILE))
    q.main(["masks", "--root", str(root), "--workers", "1", "--out", str(out)])
    per = pd.read_csv(out / "per_trial.csv")
    assert set(per.folder) == {"carter_P001_split0_t00", "moore_subject1_t00"} and "error" not in per
    tab = pd.read_csv(out / "by_lab.csv")
    assert set(tab.lab) >= {"carter", "moore", "pool"}
    assert tab.set_index("lab").loc["moore", "trials_fixed_cop_r"] == 1
    assert (out / "README.md").exists() and json.load(open(out / "run_info.json"))["n_written"] == 2
    cfg = dict(label_subdir=q.OUT_SUBDIR, label_mask=q.MASK_FILE)
    t, keep = tm.load_label_mask(d1, cfg)
    assert keep.sum() == len(t) - 7 and tm.load_label_mask(d2, cfg)[1].sum() == 0
    with pytest.raises(ValueError):                         # stale: does not match the label rows
        tm.load_label_mask(d1, cfg, label_time=t + 0.001)
    os.remove(os.path.join(d1, q.OUT_SUBDIR, q.MASK_FILE))
    with pytest.raises(FileNotFoundError):
        tm.load_label_mask(d1, cfg)


# ─── C: carrying the mask onto the frame grid ─────────────────────────────────

def _chain(t_jr, x, t_ik, t_out):
    """The loaders' label path: JR rows -> IK frames (load_subject) -> 100 Hz (_build_com_features)."""
    return np.interp(t_out, t_ik, np.interp(t_ik, t_jr, x))


@pytest.mark.parametrize("dt_jr,offset", [(0.004, 0.0009), (0.01, 0.00017), (0.005, 0.0)])
def test_carry_label_mask_bounds_the_masked_share_of_every_kept_label(dt_jr, offset):
    rng = np.random.default_rng(7)
    t_jr = 0.3 + np.arange(600) * dt_jr
    t_ik = t_jr[3:-3] + offset
    t_out = t_ik[0] + np.arange(int((t_ik[-1] - t_ik[0]) * 100) + 1) / 100.0
    keep = rng.random(len(t_jr)) > 0.3
    lab = rng.normal(2.0, 0.3, len(t_jr))
    spoiled = np.where(keep, lab, lab + 10.0)              # 10 BW of garbage on every masked JR row
    for tol in (0.0, 0.05, 0.2):
        k = tm.carry_label_mask(t_jr, keep, t_ik, t_out, tol=tol)
        diff = np.abs(_chain(t_jr, spoiled, t_ik, t_out) - _chain(t_jr, lab, t_ik, t_out))[k]
        assert diff.max() <= 10.0 * tol + 1e-9            # a kept label draws <= tol of its weight from masked rows
    k0 = tm.carry_label_mask(t_jr, keep, t_ik, t_out, tol=0.0)
    np.testing.assert_allclose(_chain(t_jr, spoiled, t_ik, t_out)[k0], _chain(t_jr, lab, t_ik, t_out)[k0])
    # the masked fraction is the interpolated indicator; with no masked rows everything is kept
    assert tm.carry_label_mask(t_jr, np.ones(len(t_jr), bool), t_ik, t_out).all()
    # identical grids: the mask itself
    np.testing.assert_array_equal(tm.carry_label_mask(t_jr, keep, t_jr), keep)


def test_carry_label_mask_tolerance_spares_sub_sample_offsets():
    """Moore-like: JR rows 0.17 ms off 100 Hz IK frames. tol 0 would also drop the neighbour of every masked row."""
    t_jr = np.arange(200) * 0.01
    t_ik = t_jr[1:-1] + 0.00017
    keep = np.ones(200, bool)
    keep[100] = False
    k0 = tm.carry_label_mask(t_jr, keep, t_ik, t_ik, tol=0.0)
    k5 = tm.carry_label_mask(t_jr, keep, t_ik, t_ik)        # default LABEL_MASK_TOL = 0.05
    assert (~k0).sum() == 2 and (~k5).sum() == 1
    assert tm.LABEL_MASK_TOL == 0.05


def _real_copy(tmp_path, keep_fn, name="carter_P010_split0_t10"):
    """A trial folder of symlinks to REAL with its own jcf_output_sojr/label_mask.npz (keep_r = keep_fn(time))."""
    d = tmp_path / name
    (d / tm.SOJR_LABEL_SUBDIR).mkdir(parents=True)
    for f in REAL_FILES:
        os.symlink(os.path.join(REAL, f), d / f)
    lab = os.path.join(REAL, tm.SOJR_LABEL_SUBDIR, "BatchJCF_JointReaction_ReactionLoads.sto")
    os.symlink(lab, d / tm.SOJR_LABEL_SUBDIR / "BatchJCF_JointReaction_ReactionLoads.sto")
    t = tm._sto_time(lab)
    keep = np.asarray(keep_fn(t), bool)
    np.savez(d / tm.SOJR_LABEL_SUBDIR / tm.LABEL_MASK_FILE, time=t, keep_r=keep, keep_l=keep)
    return str(d), t, keep


real_ok = all(os.path.exists(os.path.join(REAL, f)) for f in REAL_FILES) and os.path.exists(
    os.path.join(REAL, tm.SOJR_LABEL_SUBDIR, "BatchJCF_JointReaction_ReactionLoads.sto"))
needs_real = pytest.mark.skipif(not real_ok, reason="needs carter_P010_split0_t10 with SO+JR labels and caches")


def test_qc_sets_are_their_sojr_base_plus_sign_and_mask():
    for base in ("qinst_notrans_pelvisrel_winpos_contact_w20_t100_sojr", SOJR_FPOPT):
        qc, ref = tm.INPUT_SETS[base + "qc"], tm.INPUT_SETS[base]
        strip = lambda c: {k: v for k, v in c.items() if k not in ("desc", "mx_medial_sign", "label_mask")}  # noqa: E731
        assert strip(qc) == strip(ref)
        assert qc["mx_medial_sign"] == -1.0 and qc["label_mask"] == "label_mask.npz"
        assert tm.label_subdir_of(qc) == "jcf_output_sojr"
    c = tm._label_config(QC_FPOPT, 1e9, 1e9, True, 0.25, 0.3)
    assert c["label_mask"] == "label_mask.npz" and c["label_mask_key"] == "keep_r" and c["label_mask_tol"] == 0.05


@needs_real
def test_build_com_features_carries_the_mask_and_leaves_inputs_alone(tmp_path):
    rng = np.random.default_rng(2)
    d, t_jr, keep = _real_copy(tmp_path, lambda t: rng.random(len(t)) > 0.2)
    kw = dict(max_peak_bw=1e9, min_length=1, predict_moment=True, max_moment_bwh=1e9, return_meta=True)
    fq, lq, _, mq = tm._build_com_features(d, tm.INPUT_SETS[QC_FPOPT], **kw)
    fs, ls, _, ms = tm._build_com_features(d, tm.INPUT_SETS[SOJR_FPOPT], **kw)
    np.testing.assert_array_equal(fq, fs)
    np.testing.assert_array_equal(lq, ls)
    assert ms["label_keep"] is None and mq["label_keep"].dtype == bool and len(mq["label_keep"]) == len(lq)
    ik_t = pd.read_csv(os.path.join(d, "ik_results.mot"), sep=r"\s+", skiprows=6)["time"].to_numpy()
    t_lab = tm._common_time_grid(d, ik_t, tm.SOJR_LABEL_SUBDIR)
    np.testing.assert_array_equal(mq["label_keep"], tm.carry_label_mask(t_jr, keep, t_lab, mq["t"]))
    assert 0 < mq["label_keep"].sum() < len(lq)


@needs_real
def test_frame_dataset_drops_masked_centre_frames_only(tmp_path):
    rng = np.random.default_rng(4)
    d, _, _ = _real_copy(tmp_path / "a", lambda t: rng.random(len(t)) > 0.3)
    dz, _, _ = _real_copy(tmp_path / "z", lambda t: np.zeros(len(t), bool))
    full = tm.FrameDataset([d], max_peak_bw=1e9, max_moment_bwh=1e9, input_set=SOJR_FPOPT)
    qc = tm.FrameDataset([d, dz], max_peak_bw=1e9, max_moment_bwh=1e9, input_set=QC_FPOPT)
    keep = tm._build_com_features(d, tm.INPUT_SETS[QC_FPOPT], 1e9, 1, True, return_meta=True,
                                  max_moment_bwh=1e9)[3]["label_keep"]
    np.testing.assert_array_equal(qc.frames_in, full.frames_in[keep])     # windows from the full trial
    np.testing.assert_array_equal(qc.frames_out, full.frames_out[keep])
    assert qc.fully_masked_dirs == [dz] and qc.rejected_dirs == []
    assert qc.n_frames_unmasked == 2 * len(full) and qc.label_masked
    st = tm._mask_stats(qc)
    assert st["frames_kept"] == keep.sum() and st["fully_masked_trials"] == [os.path.basename(dz)]
    assert tm._mask_stats(full) == {}


def test_label_filter_defaults_are_off_for_mask_sets_only():
    qc, sojr = tm.INPUT_SETS[QC_FPOPT], tm.INPUT_SETS[SOJR_FPOPT]
    assert tm.label_filter_defaults(sojr) == (tm.MAX_PEAK_BW, tm.MAX_MOMENT_BWH) == (6.0, 0.15)
    assert tm.label_filter_defaults(qc) == (tm.MASKED_MAX_PEAK_BW, tm.MASKED_MAX_MOMENT_BWH) == (1000.0, 1000.0)
    assert tm.label_filter_defaults(qc, 8.0, None) == (8.0, 1000.0)       # a given value is used as given
    assert tm.label_filter_defaults(sojr, None, 0.3) == (6.0, 0.3)
    import inspect
    src = inspect.getsource(tm.train)
    assert "label_filter_defaults(INPUT_SETS[input_set], max_peak_bw, max_moment_bwh)" in src
    assert inspect.signature(tm.train).parameters["max_peak_bw"].default is None
    assert inspect.signature(tm.train).parameters["max_moment_bwh"].default is None
    # the documented *_sojrqc command states the filters
    assert "_sojrqc --max-peak 1000 --max-moment 1000" in tm.__doc__


def test_kept_labels_pass_ignores_masked_rows():
    lab = np.zeros((6, 4))
    lab[:, 1] = -2.0
    lab[2, 1] = -30.0                          # a corrupted row
    lab[4, 3] = 0.5                            # a corrupted moment
    keep = np.array([1, 1, 0, 1, 0, 1], bool)
    assert tm.kept_labels_pass(lab, keep, 6.0, 0.15)
    assert not tm.kept_labels_pass(lab, np.ones(6, bool), 6.0, None)
    assert not tm.kept_labels_pass(lab, keep | (np.arange(6) == 4), 6.0, 0.15)
    assert tm.kept_labels_pass(lab, keep | (np.arange(6) == 4), 6.0, None)     # no moment test without predict_moment
    assert not tm.kept_labels_pass(lab, keep, 1.9, 0.15)
    assert tm.kept_labels_pass(lab, np.zeros(6, bool), 0.0, 0.0)             # fully masked: left to the mask count


@needs_real
def test_trial_filters_are_judged_on_kept_frames_for_mask_sets(tmp_path):
    """A trial whose large labels are all masked loads under a QC set with any peak/moment limit above its kept frames,
    while the unmasked set (every JR row) drops it; a limit below a kept frame drops it under both."""
    from train_cnn import load_sto, jr_label_path
    pre = "walker_knee_r_on_tibia_r_in_tibia_r_"
    meta = json.load(open(os.path.join(REAL, "metadata.json")))
    bw, h = meta["mass_kg"] * 9.81, float(meta["height_m"])
    jr = load_sto(jr_label_path(REAL, tm.SOJR_LABEL_SUBDIR))
    res = np.sqrt(sum(jr[pre + c].to_numpy(float) ** 2 for c in ("fx", "fy", "fz"))) / bw
    mx = np.abs(jr[pre + "mx"].to_numpy(float)) / (bw * h)
    lim_r, lim_m = np.percentile(res, 60), np.percentile(mx, 80)
    d, _, keep_jr = _real_copy(tmp_path, lambda t: (res < lim_r) & (mx < lim_m))
    assert 0 < keep_jr.sum() < len(keep_jr)
    qc, sojr = tm.INPUT_SETS[QC_FPOPT], tm.INPUT_SETS[SOJR_FPOPT]
    _, lab, _, m = tm._build_com_features(d, qc, 1e9, 1, True, return_meta=True, max_moment_bwh=1e9)
    lk = lab[m["label_keep"]]
    kept_r, kept_m = np.sqrt((lk[:, :3] ** 2).sum(1)).max(), np.abs(lk[:, 3]).max()
    assert kept_r < res.max() and kept_m < mx.max()
    peak, mom = (kept_r + res.max()) / 2, (kept_m + mx.max()) / 2       # above every kept frame, below the worst row
    assert tm._build_com_features(d, sojr, peak, 1, True, max_moment_bwh=1e9) is None
    assert tm._build_com_features(d, sojr, 1e9, 1, True, max_moment_bwh=mom) is None
    built = tm._build_com_features(d, qc, peak, 1, True, return_meta=True, max_moment_bwh=mom)
    np.testing.assert_array_equal(built[1], lab)
    assert tm._build_com_features(d, qc, kept_r * 0.99, 1, True, max_moment_bwh=mom) is None
    assert tm._build_com_features(d, qc, peak, 1, True, max_moment_bwh=kept_m * 0.99) is None
    ds = tm.FrameDataset([d], max_peak_bw=peak, max_moment_bwh=mom, input_set=QC_FPOPT)
    assert ds.rejected_dirs == [] and len(ds) == m["label_keep"].sum()
    ds6 = tm.FrameDataset([d], max_peak_bw=peak, max_moment_bwh=mom, input_set=SOJR_FPOPT)
    assert ds6.rejected_dirs == [d]
    reason = ev.rejection_reason(d, kept_r * 0.5, (), tm.SOJR_LABEL_SUBDIR, 1e9, tm.SOJR_LABEL_SUBDIR)
    assert "on QC-kept rows" in reason


@needs_real
def test_missing_mask_is_loud(tmp_path):
    d, _, _ = _real_copy(tmp_path, lambda t: np.ones(len(t), bool))
    os.remove(os.path.join(d, tm.SOJR_LABEL_SUBDIR, tm.LABEL_MASK_FILE))
    with pytest.raises(FileNotFoundError):
        tm.FrameDataset([d], max_peak_bw=1e9, max_moment_bwh=1e9, input_set=QC_FPOPT)
    with pytest.raises(SystemExit, match="label_mask.npz"):
        tm._find_trials(str(tmp_path), label_subdir=tm.SOJR_LABEL_SUBDIR, label_mask=tm.LABEL_MASK_FILE)
    dirs, _ = tm._find_trials(str(tmp_path), label_subdir=tm.SOJR_LABEL_SUBDIR)     # no mask asked: found
    assert dirs == [d]


@needs_real
@pytest.mark.skipif(not os.path.exists(PAPER_CKPT), reason="needs the paper checkpoint")
def test_evaluator_label_mask_scores_kept_frames_only(tmp_path):
    rng = np.random.default_rng(5)
    d, _, _ = _real_copy(tmp_path / "a", lambda t: rng.random(len(t)) > 0.4)
    dz, _, _ = _real_copy(tmp_path / "z", lambda t: np.zeros(len(t), bool), name="carter_P010_split0_t11")
    common = ["--ckpt", PAPER_CKPT, "--max-peak", "1e9", "--max-moment", "1e9", "--label-subdir", "jcf_output_sojr",
              "--deploy-features", "--out-root", str(tmp_path / "out")]
    out_all = ev.main(common + ["--dirs", d, "--name", "all"])
    out_m = ev.main(common + ["--dirs", d, dz, "--name", "masked", "--label-mask"])
    cfg = dict(tm.INPUT_SETS["qinst_pelvisrel_winpos_w20"], time_exact=True, resample_hz=100.0,
               label_subdir="jcf_output_sojr", label_mask="label_mask.npz")
    keep = tm._build_com_features(d, cfg, 1e9, 1, True, return_meta=True, max_moment_bwh=1e9)[3]["label_keep"]
    tr = lambda out: pd.read_csv(os.path.join(out, "trials.csv")).query("subset == 'all'")  # noqa: E731
    assert tr(out_all).n_frames.iloc[0] == len(keep) and tr(out_m).n_frames.iloc[0] == keep.sum() < len(keep)
    rej = pd.read_csv(os.path.join(out_m, "rejected.csv"))
    assert list(rej.trial) == ["carter_P010_split0_t11"] and "keeps no frame" in rej.reason.iloc[0]
    assert "label mask jcf_output_sojr/label_mask.npz" in open(os.path.join(out_m, "README.md")).read()
    os.remove(os.path.join(dz, "jcf_output_sojr", "label_mask.npz"))
    with pytest.raises(SystemExit, match="label_mask.npz"):
        ev.main(common + ["--dirs", d, dz, "--name", "nomask", "--label-mask"])


# ─── Masks on disk (after `so_label_qc.py masks`) ─────────────────────────────

ON_DISK = [os.path.join(REPO, "jcf", "full_duration", s, a, n) for s, a, n in (
    ("training", "walking", "carter_P040_split0_t30"), ("training", "walking", "moore_subject13_t00"),
    ("testing", "walking", "carter_P010_split0_t10"), ("training", "static", "carter_P003_split0_t00"))] + [
    os.path.join(REPO, "jcf", "external", "william_2026-09-23")]


@pytest.mark.parametrize("d", ON_DISK, ids=os.path.basename)
def test_masks_on_disk_match_their_label_files(d):
    path = os.path.join(d, "jcf_output_sojr", "label_mask.npz")
    if not os.path.exists(path):
        pytest.skip("mask not written yet (python -I analysis/thesis/so_label_qc.py masks)")
    from train_cnn import jr_label_path
    with np.load(path) as z:
        t = z["time"]
        np.testing.assert_array_equal(t, tm._sto_time(jr_label_path(d, "jcf_output_sojr")))
        for s in "rl":
            np.testing.assert_array_equal(z[f"keep_{s}"], ~z[f"so_any_{s}"] & ~z[f"nonfinite_{s}"]
                                          & (not bool(z[f"fixed_cop_{s}"])))
            np.testing.assert_array_equal(z[f"so_any_{s}"], np.any([z[f"{r}_{s}"] for r in q.MASK_SO_RULES], axis=0))
        assert float(z["so_jr_dt_max"]) < 1e-6


def test_gate_require_mask(tmp_path, capsys):
    """sojr_trainset.py --gate --require-mask fails on a missing or mismatched frame mask."""
    import sojr_trainset as st
    from test_label_nonfinite import CYCLIC, make_trial
    data = str(tmp_path / "full")
    tw = os.path.join(data, "training", "walking")
    for n in ("carter_A_t01", "moore_B_t01"):
        make_trial(tw, n, {"jcf_output": (CYCLIC, ()), "jcf_output_sojr": (CYCLIC, ())})
    common = ["--gate", "--data", data, "--summary", str(tmp_path / "none.csv"), "--workers", "1"]
    assert st.main(common) == 0
    assert st.main(common + ["--require-mask"]) == 1
    assert "2 mask missing" not in capsys.readouterr().out          # reported per lab
    for n in ("carter_A_t01", "moore_B_t01"):
        t = tm._sto_time(os.path.join(tw, n, "jcf_output_sojr", "BatchJCF_JointReaction_ReactionLoads.sto"))
        np.savez(os.path.join(tw, n, "jcf_output_sojr", "label_mask.npz"), time=t + (0.01 if n.startswith("moore") else 0),
                 keep_r=np.ones(len(t), bool))
    assert st.main(common + ["--require-mask"]) == 1
    assert "training/walking/moore: 1 mask mismatch" in capsys.readouterr().out
    assert st.main(common + ["--require-mask", "--labs", "carter"]) == 0

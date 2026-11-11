"""Knee label quality control (analysis/thesis/label_qc.py).

Synthetic trial folders with known answers cover the stance statistics, F_medial, the medial sign of the left knee,
the training label filters (checked against train_cnn.filter_flat_subjects and tools/evaluate_fmedial's
rejection_reason), spikes, SO saturation, reserves and residuals, the walking-speed estimate (treadmill and
overground), the aggregation and an end-to-end run. One test checks a real Carter folder against the gc_knee_jr probe
and skips when the folder or its SO+JR labels are missing.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from analysis.thesis import label_qc as q  # noqa: E402
from pipeline.compute_com_features import foot_point_filename  # noqa: E402

MASS, H = 70.0, 1.8
BW = MASS * q.G
PRE = {s: f"walker_knee_{s}_on_tibia_{s}_in_tibia_{s}_" for s in "rl"}


JR_NOTE = ("\nThis file contains the reaction forces and moments\napplied to the specified body of a joint pair and "
           "expressed  in the specified reference frame.\n\nUnits are S.I. units (seconds, meters, Newtons, ...)\n")


def write_sto(path, df, name="table"):
    """OpenSim-style table; JR files get OpenSim's 11-line header (train_cnn.load_sto skips exactly 11 rows)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    note = JR_NOTE if os.path.basename(path) == q.JR_FILE else ""
    with open(path, "w") as f:
        f.write(f"{name}\nversion=1\nnRows={len(df)}\nnColumns={df.shape[1]}\ninDegrees=no\n{note}endheader\n")
        df.to_csv(f, sep="\t", index=False, float_format="%.8f")


def jr_table(t, fy_bw, mx_bwh, sides="r", fx_bw=0.1):
    """JR table with the given F_y (BW) and medial-positive M_x (BW*H) on each side (raw mx = MX_SIGN * M_x:
    -M_x on the right knee, +M_x on the left)."""
    cols = {"time": t}
    for s in sides:
        sign = q.MX_SIGN[s]
        cols.update({PRE[s] + "fx": np.full_like(t, fx_bw * BW), PRE[s] + "fy": np.asarray(fy_bw) * BW,
                     PRE[s] + "fz": np.zeros_like(t), PRE[s] + "mx": sign * np.asarray(mx_bwh) * BW * H,
                     PRE[s] + "my": np.zeros_like(t), PRE[s] + "mz": np.zeros_like(t)})
    return pd.DataFrame(cols)


def grf_table(t, on_r, on_l, fy_bw=1.0, cop=None):
    """Feet GRF (vertical fy_bw BW while on); cop = {side: [T, 3] centre of pressure} adds the CoP columns."""
    cols = {"time": t}
    for s, on in (("r", on_r), ("l", on_l)):
        cols.update({f"calcn_{s}_force_vx": np.zeros_like(t), f"calcn_{s}_force_vy": np.where(on, fy_bw * BW, 0.0),
                     f"calcn_{s}_force_vz": np.zeros_like(t)})
        if cop is not None:
            cols.update({f"calcn_{s}_force_p{a}": cop[s][:, k] for k, a in enumerate("xyz")})
    return pd.DataFrame(cols)


def so_tables(t, act_value=0.1, force_n=100.0, reserves=None):
    mus = [f"{m}_{s}" for s in "rl" for m in q.MUSCLES]
    act = pd.DataFrame({"time": t, **{m: np.full_like(t, act_value) for m in mus}, "lumbar_ext": np.ones_like(t)})
    frc = pd.DataFrame({"time": t, **{m: np.full_like(t, force_n) for m in mus}, "lumbar_ext": np.ones_like(t)})
    for c, v in (reserves or {}).items():
        act["reserve_" + c] = v
        frc["reserve_" + c] = v
    return act, frc


def make_folder(root, name, fy_r, mx_r, fy_l=None, mx_l=None, n=200, fs=100.0, stance_r=None, stance_l=None,
                belt=0.0, overground=0.0):
    """A trial folder with stored (right knee) and SO+JR (both knees) labels, SO files, GRF and com_features."""
    d = os.path.join(root, name)
    os.makedirs(d, exist_ok=True)
    t = np.arange(n) / fs
    ph = (t % 1.0)
    on_r = ph < 0.6 if stance_r is None else stance_r
    on_l = (ph >= 0.5) | (ph < 0.1) if stance_l is None else stance_l
    json.dump({"mass_kg": MASS, "height_m": H}, open(os.path.join(d, "metadata.json"), "w"))
    # walking: COM moves at `overground`, the stance foot moves backward at `belt`; level feet along +x, CoP 10 cm
    # ahead of and 3 cm below the calcn origin (cop_off_frac = 0)
    x_foot = -belt * t
    calc = {s: np.column_stack([x_foot, np.zeros_like(t), np.full_like(t, z)]) for s, z in (("r", 0.1), ("l", -0.1))}
    cop = {s: calc[s] + [0.10, -0.03, 0.0] for s in "rl"}
    write_sto(os.path.join(d, "grf_data.mot"), grf_table(t, on_r, on_l, cop=cop), "GRF")
    write_sto(os.path.join(d, "jcf_output", q.JR_FILE), jr_table(t, np.asarray(fy_r) * 0.5, mx_r, "r"))
    fy_l = fy_r if fy_l is None else fy_l
    mx_l = mx_r if mx_l is None else mx_l
    sojr = jr_table(t, fy_r, mx_r, "r").merge(jr_table(t, fy_l, mx_l, "l"), on="time")
    write_sto(os.path.join(d, "jcf_output_sojr", q.JR_FILE), sojr)
    act, frc = so_tables(t, reserves={"knee_angle_r": 7.0, "ankle_angle_r": 0.0, "pelvis_tx": 0.0,
                                      "pelvis_ty": 0.0, "pelvis_tz": 0.0})
    write_sto(os.path.join(d, "jcf_output", q.SO_ACT), act)
    write_sto(os.path.join(d, "jcf_output", q.SO_FORCE), frc)
    np.savez(os.path.join(d, "com_features.npz"), time=t.astype(np.float32),
             com_pos=np.column_stack([overground * t, np.full_like(t, 1.0), np.zeros_like(t)]).astype(np.float32),
             com_vel=np.column_stack([np.full_like(t, overground), np.zeros_like(t), np.zeros_like(t)]).astype(np.float32),
             **{f"calcn_{s}_pos": calc[s].astype(np.float32) for s in "rl"})
    fp = np.asarray(q.FOOT_POINT)
    np.savez(os.path.join(d, foot_point_filename(q.FOOT_POINT)), time=t.astype(np.float32),
             **{f"calcn_{s}_pos": (calc[s] + fp).astype(np.float32) for s in "rl"}, foot_point=fp)
    return d


def gait(n=200, fs=100.0, peak=2.0, mx=0.02):
    t = np.arange(n) / fs
    ph = t % 1.0
    fy = -np.where(ph < 0.6, 0.3 + (peak - 0.3) * np.sin(np.pi * ph / 0.6), 0.3)
    return fy, np.full(n, mx)


# ─── Readers and folder info ────────────────────────────────────────────────────

def test_read_sto_tab_and_whitespace(tmp_path):
    df = pd.DataFrame({"time": [0.0, 0.01], "a": [1.0, 2.0]})
    write_sto(str(tmp_path / "x.sto"), df)
    assert q.read_sto(str(tmp_path / "x.sto"))["a"].tolist() == [1.0, 2.0]
    with open(tmp_path / "y.sto", "w") as f:
        f.write("hdr\nendheader\ntime   a\n0.0   1.0\n0.01   2.0\n")
    assert q.read_sto(str(tmp_path / "y.sto"))["a"].tolist() == [1.0, 2.0]
    assert q.sto_nrows(str(tmp_path / "x.sto")) == 2 and np.isnan(q.sto_nrows(str(tmp_path / "y.sto")))


def nan_row(path, row=1):
    """Overwrite data row `row` of an OpenSim table with OpenSim's padded '-nan' (time kept), as JointReaction writes
    a failed frame."""
    lines = open(path).read().split("\n")
    i = lines.index("endheader") + 2 + row
    cells = lines[i].split("\t")
    lines[i] = "\t".join([cells[0]] + ["            -nan"] * (len(cells) - 1))
    open(path, "w").write("\n".join(lines))


def test_read_sto_padded_nan_is_numeric(tmp_path):
    p = str(tmp_path / q.JR_FILE)
    write_sto(p, jr_table(np.arange(4) / 100.0, np.full(4, -2.0), np.full(4, 0.01), "rl"))
    nan_row(p, 1)
    df = q.read_sto(p)
    assert all(pd.api.types.is_numeric_dtype(df[c]) for c in df.columns)
    assert df[PRE["r"] + "fy"].isna().tolist() == [False, True, False, False]
    assert df["time"].notna().all()


def test_folder_info():
    i = q.folder_info("/x/jcf/full_duration/testing/static/moore_subject10_split3_t07")
    assert (i["split"], i["activity"], i["lab"], i["person"]) == ("testing", "static", "moore", "moore_subject10")
    i = q.folder_info("/x/jcf/external/william_2026-09-23")
    assert (i["split"], i["activity"], i["lab"]) == ("external", "walking", "william")
    assert q.person_id("fregly_3GC_t18_r00") == "fregly_3GC"


def test_find_folders_needs_sojr(tmp_path):
    fy, mx = gait()
    a = make_folder(str(tmp_path), "carter_P001_split0_t00", fy, mx)
    b = make_folder(str(tmp_path), "carter_P001_split0_t01", fy, mx)
    os.remove(os.path.join(b, "jcf_output_sojr", q.JR_FILE))
    assert q.find_folders([str(tmp_path)]) == [a]
    assert q.find_folders([str(tmp_path)], need_sojr=False) == [a, b]


# ─── Knee statistics ────────────────────────────────────────────────────────────

def test_knee_stats_stance_fmedial_filters_and_spikes():
    n = 101
    t = np.arange(n) / 100.0
    stance = t < 0.5
    fy = np.where(stance, -2.0, -0.2)
    fy[20] = -7.0                                      # one-frame spike above the 6 BW filter, inside stance
    mx = np.where(stance, 0.02, -0.01)
    mx[80] = 0.2                                       # |M_x| above 0.15 BW*H, in swing
    k = dict(t=t, fx=np.zeros(n), fy=fy, fz=np.zeros(n), mx=mx)
    s = q.knee_stats(k, stance, H)
    ns = stance.sum()
    assert s["stance_n"] == ns
    assert s["fy_stance_peak"] == pytest.approx(7.0)
    assert s["fy_stance_mean"] == pytest.approx((2.0 * (ns - 1) + 7.0) / ns)
    fmed = 0.5 * 2.0 + 0.02 * H / 0.045                # 1.8 BW in every normal stance frame
    assert s["fmed_stance_peak"] == pytest.approx(0.5 * 7.0 + 0.02 * H / 0.045)
    assert s["fmed_stance_mean"] == pytest.approx((fmed * (ns - 1) + 0.5 * 7 + 0.8) / ns)
    assert s["fmed_gt14_frac"] == 1.0 and s["fy_gt14_frac"] == 1.0
    assert s["fy_compressive_frac"] == 1.0
    assert (s["n_res_gt6"], s["n_mx_gt015"]) == (1, 1)
    assert s["reject_peak"] and s["reject_mx"] and s["reject_any"]
    assert (s["mx_min"], s["mx_max"]) == (pytest.approx(-0.01), pytest.approx(0.2))
    assert s["spike_n"] == 1 and s["mx_spike_n"] == 1
    assert s["max_step"] == pytest.approx(5.0)
    assert s["n_step_gt05"] == 3                       # into and out of the spike, and the stance-to-swing edge
    assert s["max_rate"] == pytest.approx(500.0)


def test_knee_stats_nonfinite_frames_are_counted_and_skipped():
    n = 101
    t = np.arange(n) / 100.0
    stance = t < 0.5
    fy = np.where(stance, -2.0, -0.2)
    mx = np.full(n, 0.01)
    k = dict(t=t, fx=np.zeros(n), fy=fy.copy(), fz=np.zeros(n), mx=mx.copy())
    ref = q.knee_stats(dict(k, t=np.delete(t, 3), fx=np.zeros(n - 1), fy=np.delete(fy, 3), fz=np.zeros(n - 1),
                            mx=np.delete(mx, 3)), np.delete(stance, 3), H)
    k["fy"][3] = np.nan
    s = q.knee_stats(k, stance, H)
    assert s["n_nonfinite"] == 1 and ref["n_nonfinite"] == 0
    assert s["n"] == n and s["stance_n"] == stance.sum() - 1
    for m in ("fy_stance_peak", "fres_peak", "cv", "dr", "fmed_stance_mean", "n_res_gt6"):
        assert s[m] == pytest.approx(ref[m]) and np.isfinite(s[m]), m
    allnan = q.knee_stats(dict(k, fy=np.full(n, np.nan)), stance, H)
    assert allnan["n_nonfinite"] == n and "fy_stance_peak" not in allnan


def test_full_stances_need_both_edges_and_min_duration():
    t = np.arange(100) / 100.0
    s = np.zeros(100, bool)
    s[:20] = True                                        # starts at the trial start: incomplete
    s[30:60] = True                                      # complete, 0.29 s
    s[70:80] = True                                      # complete but 0.09 s: too short
    s[90:] = True                                        # runs to the trial end: incomplete
    assert q.full_stances(s, t) == [(30, 60)]
    assert q.full_stances(s, t, min_s=0.05) == [(30, 60), (70, 80)]
    fy = -np.where(s, 1.0, 0.2)
    fy[45] = -3.0
    st = q.knee_stats(dict(t=t, fx=np.zeros(100), fy=fy, fz=np.zeros(100), mx=np.zeros(100)), s, H)
    assert st["n_full_stance"] == 1 and st["fy_full_peak"] == pytest.approx(3.0)
    st = q.knee_stats(dict(t=t, fx=np.zeros(100), fy=fy, fz=np.zeros(100), mx=np.zeros(100)), np.ones(100, bool), H)
    assert st["n_full_stance"] == 0 and np.isnan(st["fy_full_peak"])


def test_smooth_waveform_has_no_spikes():
    t = np.arange(400) / 250.0
    res = 1.0 + 1.5 * np.sin(2 * np.pi * t) ** 2
    s = q.spike_stats(res, np.zeros_like(t), t)
    assert s["spike_n"] == 0 and s["n_step_gt05"] == 0 and s["max_rate"] < 2 * np.pi * 1.5 + 1e-6


def test_running_median_edges():
    x = np.array([0.0, 0.0, 9.0, 0.0, 0.0, 0.0])
    assert q.running_median(x).tolist() == [0.0] * 6
    assert q.running_median(np.array([1.0, 3.0])).tolist() == [2.0, 2.0]


def test_left_knee_mx_is_medial_positive():
    t = np.arange(10) / 100.0
    jr = jr_table(t, np.full(10, -2.0), np.full(10, 0.03), "rl")
    # stored OpenSim sign of a medial moment (physical convention, analysis/thesis/knee_mx_sign.py): -mx on the
    # right tibia, +mx on the left
    assert jr[PRE["r"] + "mx"].iloc[0] < 0 < jr[PRE["l"] + "mx"].iloc[0]
    for s in "rl":
        k = q.knee_loads(jr, s, BW, H)
        assert k["mx"] == pytest.approx(np.full(10, 0.03))
        assert k["fy"] == pytest.approx(np.full(10, -2.0))
    assert q.knee_loads(jr_table(t, np.ones(10), np.zeros(10), "r"), "l", BW, H) is None


def test_stance_mask_threshold():
    t = np.arange(5) / 100.0
    g = pd.DataFrame({"time": t, "calcn_r_force_vy": [0.0, 0.04, 0.06, 1.0, 0.0]}) * [1, BW]
    assert q.stance_mask(t, g, "r", BW).tolist() == [False, False, True, True, False]
    assert not q.stance_mask(t, g, "l", BW).any()


def test_filters_match_training_code(tmp_path):
    """reject_* agree with tools/evaluate_fmedial.rejection_reason and flat with train_cnn.filter_flat_subjects."""
    from tools.evaluate_fmedial import rejection_reason
    from train_cnn import filter_flat_subjects
    fy, mx = gait(peak=2.0)
    cases = {"ok": (fy, mx), "mx": (fy, np.where(np.arange(len(fy)) == 50, 0.3, mx)),
             "peak": (np.where(np.arange(len(fy)) == 30, -14.0, fy), mx),      # stored = half: 7 BW
             "flat": (np.full(len(fy), -1.0), mx)}
    for i, (name, (f, m)) in enumerate(cases.items()):
        d = make_folder(str(tmp_path), f"carter_P00{i}_{name}_t00", f, m)
        row = q.trial_row(d)
        reason = rejection_reason(d, 6.0)
        assert row["stored_r_reject_mx"] == ("Mx" in reason), name
        assert row["stored_r_reject_peak"] == ("resultant" in reason), name
        for src, sub in q.SOURCES:
            kept, _ = filter_flat_subjects([d], jcf_subdir=sub)
            assert row[f"{src}_r_flat"] == (not kept), (name, src)
    assert q.trial_row(os.path.join(str(tmp_path), "carter_P003_flat_t00"))["sojr_r_flat"]


def test_nonfinite_label_folder_status(tmp_path):
    """A '-nan' frame in the SO+JR file: counted per side, status 'nonfinite labels', finite statistics; the training
    loader's own check (train_cnn.label_nonfinite_rows, when present) sees the same file as non-finite."""
    fy, mx = gait()
    d = make_folder(str(tmp_path), "tiziana_Subject12_t03", fy, mx)
    nan_row(os.path.join(d, "jcf_output_sojr", q.JR_FILE), 1)
    r = q.trial_row(d)
    assert r["status"] == "nonfinite labels"
    assert r["sojr_r_n_nonfinite"] == 1 and r["sojr_l_n_nonfinite"] == 1 and r["stored_r_n_nonfinite"] == 0
    assert np.isfinite(r["sojr_r_fres_peak"]) and np.isfinite(r["sojr_r_cv"])
    assert q.nonfinite_folders(pd.DataFrame([r])) == {"external/walking/tiziana_Subject12_t03": {"sojr_r": 1,
                                                                                                "sojr_l": 1}}
    import train_cnn
    if hasattr(train_cnn, "label_nonfinite_rows"):
        assert train_cnn.label_nonfinite_rows(d, "jcf_output_sojr") == 1
        assert train_cnn.label_nonfinite_rows(d, "jcf_output") == 0


def test_double_support_fraction_and_instrumented_side(tmp_path):
    fy, mx = gait()
    d = make_folder(str(tmp_path), "fregly_3GC_t18_r00", fy, mx)
    r = q.trial_row(d)
    assert r["ds_frac"] == pytest.approx(0.2)             # right stance 0-60 %, left 50-110 %: 2 x 10 % overlap
    assert r["instrumented_side"] == "l" and r["so_n_nonfinite"] == 0
    fi = q.fregly_instrumented(pd.DataFrame([r]))
    assert fi.loc[0, "side"] == "l"
    assert fi.loc[0, "sojr_fy_stance_peak"] == pytest.approx(r["sojr_l_fy_stance_peak"])
    assert np.isnan(fi.loc[0, "stored_fy_stance_peak"])   # the stored files have no left knee
    assert "instrumented_side" not in q.folder_info("/x/fregly_5GC_t00")


# ─── Relabel coverage ───────────────────────────────────────────────────────────

SETUP_XML = """<?xml version="1.0" encoding="UTF-8" ?>
<OpenSimDocument Version="40000">
  <AnalyzeTool name="BatchJCF">
    <model_file>/Users/someone/jcf/{name}/jcf_output/_model_with_reserves.osim</model_file>
    <results_directory>/Users/someone/jcf/{name}/jcf_output</results_directory>
    <AnalysisSet name="Analyses">
      <objects>
        <StaticOptimization name="StaticOptimization"><on>true</on></StaticOptimization>
        <JointReaction name="JointReaction"><on>true</on><forces_file /><joint_names> all</joint_names>
          <apply_on_bodies> parent</apply_on_bodies><express_in_frame> parent</express_in_frame></JointReaction>
      </objects>
    </AnalysisSet>
    <external_loads_file>/Users/someone/jcf/{name}/grf_loads.xml</external_loads_file>
    <states_file />
    <coordinates_file>/Users/someone/jcf/{name}/jcf_output/_padded_ik.mot</coordinates_file>
    <speeds_file />
  </AnalyzeTool>
</OpenSimDocument>
"""


def with_setup(d, local=("jcf_output/_model_with_reserves.osim", "jcf_output/_padded_ik.mot", "grf_loads.xml")):
    """The labelling setup with the Mac's /Users/ paths, and the local copies of its inputs that exist here."""
    name = os.path.basename(d)
    open(os.path.join(d, "jcf_output", q.SO_SETUP), "w").write(SETUP_XML.format(name=name))
    for f in local:
        open(os.path.join(d, f), "w").write("x")
    return d


def test_relabel_state(tmp_path):
    fy, mx = gait()
    mk = lambda n: with_setup(make_folder(str(tmp_path), n, fy, mx))   # noqa: E731
    done, failed, pending, fresh = (mk(f"moore_subject0{i}_t00") for i in range(1, 5))
    nosetup = make_folder(str(tmp_path), "moore_subject05_t00", fy, mx)
    for d in (failed, pending, fresh, nosetup):
        os.remove(os.path.join(d, "jcf_output_sojr", q.JR_FILE))
    for d in (failed, pending):
        open(os.path.join(d, "jcf_output_sojr", q.SOJR_SETUP), "w").write("<x/>")
    os.remove(os.path.join(fresh, "jcf_output", "_padded_ik.mot"))     # an input with no local copy
    summary_t = 1_000_000.0
    os.utime(os.path.join(failed, "jcf_output_sojr", q.SOJR_SETUP), (summary_t - 60, summary_t - 60))
    os.utime(os.path.join(pending, "jcf_output_sojr", q.SOJR_SETUP), (summary_t + 60, summary_t + 60))
    st = {os.path.basename(d): q.relabel_state(d, summary_t) for d in (done, failed, pending, fresh, nosetup)}
    assert [st[k]["relabel"] for k in sorted(st)] == ["labelled", "failed", "pending", "not run", "no setup"]
    assert st["moore_subject01_t00"]["labelled_after_summary"]             # the JR file is newer than 1e6 s
    assert st["moore_subject02_t00"]["setup_resolves"] and st["moore_subject03_t00"]["setup_resolves"]
    assert not st["moore_subject04_t00"]["setup_resolves"]
    assert "_padded_ik.mot" in st["moore_subject04_t00"]["setup_error"]
    assert st["moore_subject01_t00"]["so_frames"] == 200
    assert not os.path.exists(os.path.join(fresh, "jcf_output_sojr", q.SOJR_SETUP))   # nothing written
    cov = q.relabel_coverage([done, failed, pending, fresh, nosetup], workers=1, summary=str(tmp_path / "none.csv"))
    cs = q.coverage_summary(cov)
    assert cs["total"]["n_with_so"] == 5 and cs["total"]["labelled"] == 1 and not cs["complete"]
    assert cs["total"]["pending"] == 2                                     # no summary: every setup is a pending run
    assert cs["by_lab"]["moore"]["unlabelled_setup_resolves"] == 2
    assert cs["by_lab"]["moore"]["unlabelled_so_frames"] == 800
    assert q.coverage_summary(cov[cov["relabel"] == "labelled"])["complete"]


def test_is_relabel_cmdline():
    assert q.is_relabel_cmdline(b"python\0analysis/thesis/gc_knee_jr.py\0--workers\x0020\0")
    assert q.is_relabel_cmdline(b"/x/conda\0run\0-n\0jcf\0python\0analysis/thesis/gc_knee_jr.py\0")
    assert not q.is_relabel_cmdline(b"/bin/bash\0-c\0until ! pgrep -f gc_knee_jr.py; do sleep 5; done\0")
    assert not q.is_relabel_cmdline(b"python\0-m\0pytest\0tests/test_gc_knee_jr.py\0")
    assert not q.is_relabel_cmdline(b"")


def test_relabel_summary_counts_per_lab(tmp_path):
    p = tmp_path / "relabel_all.csv"
    mac = ("failed: RuntimeError: std::exception in 'OpenSim::AnalyzeTool::AnalyzeTool(std::string const &)': "
           "Object: Cannot open file /Users/someone/jcf/x/jcf_output/_model_with_reserves.osim. It may not exist")
    pd.DataFrame({"folder": ["moore_s1_t00", "moore_s1_t01", "moore_s1_t02", "han_s001_t00", "carter_P1_t00",
                             "carter_P1_t00", "fregly_3GC_t01"],
                  "status": [mac, mac, "ok", "failed: ValueError: no JointReaction analysis", "exists", "ok",
                             "no SO output"]}).to_csv(p, index=False)
    s = q.relabel_summary(str(p))
    assert s["n_rows"] == 7 and s["status_counts"] == {"failed": 3, "ok": 2, "exists": 1, "no SO output": 1}
    assert s["by_lab"]["moore"] == {"failed": 2, "ok": 1} and s["by_lab"]["carter"] == {"exists": 1, "ok": 1}
    assert s["failure_reasons"] == {"input file not found (labelling-Mac /Users/ path)": 2,
                                    "ValueError: no JointReaction analysis": 1}
    assert q.relabel_summary(str(tmp_path / "missing.csv")) is None


# ─── People ─────────────────────────────────────────────────────────────────────

def person_table():
    """Lab 'moore': person A 9 trials, all over the 6 BW filter; person B 1 clean trial."""
    rows = []
    for i in range(10):
        a = i < 9
        rows.append(dict(split="training", activity="walking", lab="moore", folder=f"t{i}",
                         person="moore_A" if a else "moore_B", status="ok",
                         sojr_r_fy_stance_peak=9.0 + 0.1 * i if a else 2.0, sojr_r_reject_peak=a, sojr_r_reject_mx=False,
                         sojr_r_reject_any=a, sojr_r_flat=False, sojr_r_n=100, sojr_r_n_res_gt6=10 if a else 0,
                         sojr_r_n_mx_gt015=0))
    return pd.DataFrame(rows)


def test_person_weighted_rejections_and_headline():
    df = person_table()
    rej = q.filter_rejections(df).set_index("lab")
    assert rej.loc["moore", "sojr_r_dropped_pct"] == pytest.approx(90.0)
    assert rej.loc["moore", "sojr_r_dropped_pct_pw"] == pytest.approx(50.0)
    assert rej.loc["moore", "n_people"] == 2 and rej.loc["moore", "top_person"] == "moore_A"
    assert rej.loc["moore", "top_person_share"] == pytest.approx(0.9)
    head = q.headline(df, sides="r").set_index("lab")
    assert head.loc["moore", "sojr_fy_stance_peak"].startswith("9.35")
    pm = np.median([9.4, 2.0])                                             # A's median 9.4, B's 2.0
    assert head.loc["moore", "sojr_fy_stance_peak_by_person"] == f"{pm:.2f} [2.00, 9.40]"
    bp = q.by_person(df).set_index("person")
    assert bp.loc["moore_A", "n_trials"] == 9 and bp.loc["moore_A", "sojr_r_dropped_pct"] == 100.0
    assert bp.loc["moore_B", "sojr_r_fy_stance_peak_med"] == 2.0 and bp.loc["moore_B", "sojr_r_dropped_pct"] == 0.0


# ─── Measured check ─────────────────────────────────────────────────────────────

def test_measured_check(tmp_path):
    """Synthetic grand_challenge frames: segment means and peaks, window coverage, the stance flag."""
    rows = []
    pct = np.linspace(0, 100, 41)                       # one full stance, 2.5 % steps
    for k, p in enumerate(pct):
        rows.append(dict(subject="SC", folder="fregly_3GC_t00", stance=True, stance_id=0, pct=p,
                         meas_tot=2.0 if p < 50 else 1.0, fy_osim_sojr=-(3.0 if p < 50 else 4.0)))
    for k, p in enumerate(np.linspace(60, 70, 5)):      # a fragment: no window covered
        rows.append(dict(subject="SC", folder="fregly_3GC_t01", stance=True, stance_id=0, pct=p, meas_tot=1.0,
                         fy_osim_sojr=-1.5))
    rows.append(dict(subject="SC", folder="fregly_3GC_t01", stance=False, stance_id=-1, pct=np.nan, meas_tot=0.1,
                     fy_osim_sojr=-9.0))                # swing: ignored
    p = tmp_path / "frames.csv.gz"
    pd.DataFrame(rows).to_csv(p, index=False)
    tab, summ = q.measured_check(str(p))
    seg = tab[tab["kind"] == "segment"].set_index("folder")
    assert seg.loc["fregly_3GC_t00", "full_stance"] and not seg.loc["fregly_3GC_t01", "full_stance"]
    assert seg.loc["fregly_3GC_t01", "pred_peak"] == 1.5                   # the swing frame is not counted
    win = tab[tab["kind"] == "window"].set_index("part")
    assert win.loc["early", "pred_peak"] == 3.0 and win.loc["early", "meas_peak"] == 2.0
    assert win.loc["late", "pred_peak"] == 4.0 and win.loc["late", "peak_diff"] == 3.0
    s = summ["SC"]
    assert s["n_segments"] == 2 and s["n_full_stance_segments"] == 1 and s["early_windows"] == 1
    n0, n1 = 41, 5
    pred = (20 * 3.0 + 21 * 4.0 + n1 * 1.5) / (n0 + n1)
    meas = (20 * 2.0 + 21 * 1.0 + n1 * 1.0) / (n0 + n1)
    assert s["frame_bias"] == pytest.approx(pred - meas, abs=1e-3)
    assert q.measured_check(str(tmp_path / "missing.csv.gz")) == (None, None)
    assert q.measured_check("") == (None, None)


def test_measured_check_fills_from_segment_labels(tmp_path):
    """Frames without a grand_challenge SO+JR value take the instrumented knee's F_y from the folder's own labels;
    frames that have one are left alone and serve as the consistency check."""
    root = tmp_path / "walking"
    fy, mx = gait()
    fy_l = fy * 1.5
    make_folder(str(root), "fregly_1GC_t00", fy, mx, fy_l=fy_l)          # JW: instrumented right knee
    make_folder(str(root), "fregly_3GC_t00", fy, mx, fy_l=fy_l)          # SC: instrumented left knee
    t = np.arange(200) / 100.0
    ks = np.arange(20, 60)
    rows = [dict(subject=s, folder=fo, k=k, t_seg=t[k], mass=MASS, stance=True, stance_id=0, pct=100 * (k - 20) / 39,
                 meas_tot=1.0, fy_osim_sojr=v) for s, fo, ref in (("JW", "fregly_1GC_t00", fy),
                                                                  ("SC", "fregly_3GC_t00", fy_l))
            for k in ks for v in [np.nan if s == "JW" else ref[k]]]
    p = tmp_path / "frames.csv.gz"
    frames = pd.DataFrame(rows)
    frames.to_csv(p, index=False)
    own = q.labels_at_frames(frames, roots=(str(root),))
    assert own[frames["subject"] == "JW"].to_numpy() == pytest.approx(fy[ks], abs=1e-6)
    assert own[frames["subject"] == "SC"].to_numpy() == pytest.approx(fy_l[ks], abs=1e-6)
    tab, summ = q.measured_check(str(p), roots=(str(root),))
    assert summ["JW"]["n_filled"] == len(ks) and summ["SC"]["n_filled"] == 0
    assert summ["JW"]["fill_check_max_diff"] == pytest.approx(0.0, abs=1e-6)
    seg = tab[tab["kind"] == "segment"].set_index("subject")
    assert seg.loc["JW", "pred_peak"] == pytest.approx(np.abs(fy[ks]).max(), abs=1e-6) and seg.loc["JW", "filled"]
    _, nofill = q.measured_check(str(p), fill=False, roots=(str(root),))
    assert "JW" not in nofill and nofill["SC"]["fill_check_max_diff"] is None


# ─── SO quality ─────────────────────────────────────────────────────────────────

def test_saturation_stats():
    t = np.arange(10) / 100.0
    act, _ = so_tables(t, act_value=0.2, reserves={"knee_angle_r": 5.0})
    act.loc[[0, 1, 2], "vaslat_r"] = 1.0
    act.loc[[1], "gasmed_l"] = 0.9995
    for m in ("soleus_r", "tibant_r", "edl_r", "fdl_r", "fhl_r", "ehl_r"):
        act.loc[[5], m] = 1.0
    act["lumbar_ext"] = 1.0                           # coordinate actuator, not a muscle
    s = q.saturation_stats(act, stance_r=t < 0.03)
    assert s["so_n_muscles"] == 80
    assert s["so_sat_any_frac"] == pytest.approx(0.4)
    assert s["so_sat_gt5_frac"] == pytest.approx(0.1)
    assert s["so_n_sat_muscles"] == 8 and s["so_sat_max_per_frame"] == 6
    assert s["so_sat_top"].startswith("vaslat_r:3")
    assert s["so_sat_any_stance_r_frac"] == 1.0


def test_reserve_stats():
    t = np.arange(5) / 100.0
    _, frc = so_tables(t, force_n=100.0, reserves={"knee_angle_r": np.array([0, -35.0, 10, 0, 0]),
                                                   "knee_angle_l": 7.0, "ankle_angle_r": 0.0,
                                                   "pelvis_tx": np.array([0, 30.0, 0, 0, 0]),
                                                   "pelvis_ty": np.array([0, 40.0, 0, 0, 0]), "pelvis_tz": 0.0,
                                                   "pelvis_tilt": 7.0, "pelvis_list": 0.0, "pelvis_rotation": 0.0})
    frc["vaslat_r"], frc["vasmed_r"], frc["vasint_r"] = 2 * BW, BW, 0.0
    grf = grf_table(t, np.ones(5, bool), np.zeros(5, bool), fy_bw=1.0)
    stances = {"r": np.array([0, 1, 1, 0, 0], bool), "l": np.zeros(5, bool)}
    s = q.reserve_stats(frc, MASS, H, stances, grf, com_height=1.0)
    assert s["res_knee_r_peak_nmkg"] == pytest.approx(35.0 / MASS)
    assert s["res_knee_r_peak_bwh"] == pytest.approx(35.0 / (BW * H))
    assert s["res_knee_r_stance_mean_nmkg"] == pytest.approx(22.5 / MASS)
    assert np.isnan(s["res_knee_l_stance_mean_nmkg"])
    assert s["res_ankle_r_peak_nmkg"] == 0.0
    assert s["resid_force_pct"] == pytest.approx(100 * 50.0 / BW)
    assert s["resid_moment_pct"] == pytest.approx(100 * 7.0 / (BW * 1.0))
    assert s["vas_r_max_bw"] == pytest.approx(3.0)
    assert s["gas_l_max_bw"] == pytest.approx(200.0 / BW)
    res_sum = 45 + 5 * 7 + (30 + 40) + 5 * 7           # knee_r, knee_l, pelvis tx + ty, pelvis tilt
    mus_sum = 100.0 * 5 * 77 + 5 * 3 * BW               # 77 muscles at 100 N, vasti of the right side
    assert s["res_force_share"] == pytest.approx(res_sum / (res_sum + mus_sum))


# ─── Centre of pressure ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("heading_deg", [0.0, 35.0, -120.0])
def test_foot_axes_recovers_heading(heading_deg):
    h = np.radians(heading_deg)
    X = np.array([np.cos(h), 0.0, -np.sin(h)])          # rotation about +y (up)
    Z = np.cross(X, [0.0, 1.0, 0.0])
    calcn = np.array([[0.3, 0.05, -0.2]])
    toe = calcn + q.FOOT_POINT[0] * X + q.FOOT_POINT[2] * Z
    fwd, right = q.foot_axes(calcn, toe)
    assert fwd[0] == pytest.approx(X) and right[0] == pytest.approx(Z)


def test_cop_stats_on_and_off_the_foot():
    n = 10
    t = np.arange(n) / 100.0
    cal = np.tile([1.0, 0.05, 0.1], (n, 1))
    com = {"time": t, "calcn_r_pos": cal, "calcn_l_pos": cal}
    toe = {"time": t, "calcn_r_pos": cal + np.asarray(q.FOOT_POINT), "calcn_l_pos": cal + np.asarray(q.FOOT_POINT)}
    fwd = np.array([0.10] * 8 + [0.40, -0.10])           # two frames beyond the toes / behind the heel
    cop = cal + np.column_stack([fwd, np.full(n, -0.04), np.full(n, 0.02)])
    grf = pd.DataFrame({"time": t, "calcn_r_force_vy": np.full(n, BW), "calcn_r_force_px": cop[:, 0],
                        "calcn_r_force_py": cop[:, 1], "calcn_r_force_pz": cop[:, 2],
                        "calcn_l_force_vy": np.full(n, BW), "calcn_l_force_px": cop[:, 0],
                        "calcn_l_force_py": cop[:, 1], "calcn_l_force_pz": cop[:, 2]})
    r = q.cop_stats(grf, com, toe, "r", BW)
    assert r["cop_r_off_frac"] == pytest.approx(0.2)
    assert r["cop_r_lat_med"] == pytest.approx(0.02) and r["cop_r_dy_med"] == pytest.approx(-0.04)
    assert r["cop_r_fwd_p95"] == pytest.approx(np.percentile(fwd, 95))
    assert r["cop_r_fwd_p5"] == pytest.approx(np.percentile(fwd, 5))
    assert q.cop_stats(grf, com, toe, "l", BW)["cop_l_lat_med"] == pytest.approx(-0.02)   # +z is medial for the left
    grf["calcn_r_force_vy"] = 0.1 * BW
    assert q.cop_stats(grf, com, toe, "r", BW) == {}


# ─── Speed ──────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("belt,overground", [(1.25, 0.0), (0.0, 1.4)])
def test_walking_speed_treadmill_and_overground(tmp_path, belt, overground):
    fy, mx = gait()
    d = make_folder(str(tmp_path), "moore_subject01_t00", fy, mx, belt=belt, overground=overground)
    r = q.trial_row(d)
    assert r["speed_ms"] == pytest.approx(belt + overground, abs=1e-3)
    assert r["com_speed_ms"] == pytest.approx(overground, abs=1e-3)


# ─── Aggregation and end to end ────────────────────────────────────────────────

def test_aggregate_quantiles():
    df = pd.DataFrame({"split": "training", "activity": "walking", "lab": ["a"] * 5 + ["b"] * 5, "folder": "x",
                       "person": "p", "status": "ok", "m": np.arange(10.0)})
    agg = q.aggregate(df, ["m"]).set_index("lab")
    assert agg.loc["a", "median"] == 2.0 and agg.loc["a", "q25"] == 1.0 and agg.loc["a", "q75"] == 3.0
    assert agg.loc["a", "p95"] == pytest.approx(3.8)
    assert agg.loc["ALL", "n"] == 10 and agg.loc["ALL", "median"] == 4.5


def test_filter_rejections_flat_counts_only_for_walking():
    base = dict(folder="x", person="p", status="ok", sojr_r_reject_peak=False, sojr_r_reject_mx=False,
                sojr_r_reject_any=False, sojr_r_flat=True, sojr_r_n=100, sojr_r_n_res_gt6=0, sojr_r_n_mx_gt015=0)
    df = pd.DataFrame([dict(base, split="training", activity="walking", lab="a"),
                       dict(base, split="training", activity="static", lab="a"),
                       dict(base, split="training", activity="static", lab="a", sojr_r_reject_any=True,
                            sojr_r_reject_peak=True, sojr_r_n_res_gt6=10)])
    r = q.filter_rejections(df).set_index(["activity", "lab"])
    assert r.loc[("walking", "a"), "sojr_r_dropped_pct"] == 100.0
    assert r.loc[("static", "a"), "sojr_r_dropped_pct"] == 50.0
    assert r.loc[("static", "a"), "sojr_r_frames_res_gt6_pct"] == pytest.approx(5.0)


def test_correlates_spearman():
    n = 30
    x = np.arange(n, dtype=float)
    df = pd.DataFrame({"split": "training", "activity": "walking", "lab": "a", "folder": "x", "person": "p",
                       "status": "ok", "sojr_r_fy_stance_peak": x ** 2, "speed_ms": x, "mass_kg": -x,
                       "so_sat_any_frac": 1.0})
    c = q.correlates(df).set_index("lab")
    assert c.loc["a", "speed_ms"] == pytest.approx(1.0) and c.loc["a", "mass_kg"] == pytest.approx(-1.0)
    assert "so_sat_any_frac" not in c or np.isnan(c.loc["a", "so_sat_any_frac"])   # constant: no rank correlation
    assert q.correlates(df.iloc[:10]).empty                                          # below min_n


def test_main_end_to_end(tmp_path):
    root = tmp_path / "jcf" / "full_duration" / "training"
    fy, mx = gait(peak=2.0)
    fy_hi, _ = gait(peak=9.0)
    make_folder(str(root / "walking"), "carter_P001_split0_t00", fy, mx, belt=1.2)
    make_folder(str(root / "walking"), "carter_P001_split0_t01", fy_hi, mx, belt=1.2)
    make_folder(str(root / "walking"), "moore_subject02_t00", fy, mx, belt=0.8)
    make_folder(str(root / "static"), "moore_subject02_t01", np.full(200, -1.0), mx,
                stance_r=np.ones(200, bool), stance_l=np.ones(200, bool))
    bad = make_folder(str(root / "walking"), "tiziana_Subject12_t03", fy, mx)
    nan_row(os.path.join(bad, "jcf_output_sojr", q.JR_FILE), 1)
    todo = with_setup(make_folder(str(root / "walking"), "moore_subject03_t00", fy, mx))
    os.remove(os.path.join(todo, "jcf_output_sojr", q.JR_FILE))            # SO output, no SO+JR labels yet
    summ = tmp_path / "relabel_all.csv"
    pd.DataFrame({"folder": ["moore_subject03_t00", "carter_P001_split0_t00"],
                  "status": ["failed: RuntimeError: Cannot open file /Users/x/m.osim.", "ok"]}).to_csv(summ, index=False)
    out = tmp_path / "out"
    df = q.main(["--root", str(root / "walking"), str(root / "static"), "--out", str(out), "--workers", "1",
                 "--relabel-summary", str(summ), "--gc-frames", ""])
    assert len(df) == 5 and (df["status"] == "ok").sum() == 4
    assert df.set_index("folder").loc["tiziana_Subject12_t03", "status"] == "nonfinite labels"
    for f in ("per_trial.csv", "by_group.csv", "headline.csv", "filter_rejections.csv", "correlates.csv", "run_info.json",
              "by_person.csv", "relabel_coverage.csv", "fregly_instrumented.csv",
              "fig_peaks_by_lab.png", "fig_peak_fy_vs_speed.png", "fig_filter_rejections.png"):
        assert (out / f).exists(), f
    assert not (out / "fregly_vs_measured.csv").exists()
    rej = pd.read_csv(out / "filter_rejections.csv").set_index(["activity", "lab"])
    assert rej.loc[("walking", "carter"), "sojr_r_reject_peak_pct"] == 50.0
    assert rej.loc[("walking", "carter"), "stored_r_reject_peak_pct"] == 0.0
    head = pd.read_csv(out / "headline.csv")
    assert set(head["knee"]) == {"r", "l"} and head.loc[head["knee"] == "l", "stored_fy_stance_peak"].isna().all()
    head = head[head["knee"] == "r"].set_index(["activity", "lab"])
    assert head.loc[("walking", "moore"), "sojr_fy_stance_peak"].startswith("2.00")
    assert head.loc[("walking", "moore"), "n_with_so"] == 2 and head.loc[("walking", "moore"), "sojr_coverage"] == 0.5
    info = json.load(open(out / "run_info.json"))
    assert info["n_folders"] == 5 and info["n_ok"] == 4
    assert "n_labelled_expected" not in info and "relabel_finished" not in info
    assert not info["relabel_complete"]
    assert info["relabel_coverage"]["total"] == dict(n_with_so=6, labelled=5, labelled_frac=0.8333, **{"not run": 1},
                                                     unlabelled_setup_resolves=1, unlabelled_so_frames=200)
    assert info["relabel_summary"]["by_lab"] == {"carter": {"ok": 1}, "moore": {"failed": 1}}
    assert info["nonfinite_folders"] == {"training/walking/tiziana_Subject12_t03": {"sojr_r": 1, "sojr_l": 1}}
    assert info["status_counts"] == {"ok": 4, "nonfinite labels": 1}
    cov = pd.read_csv(out / "relabel_coverage.csv").set_index("folder")
    assert cov.loc["moore_subject03_t00", "relabel"] == "not run" and cov.loc["moore_subject03_t00", "setup_resolves"]
    per = pd.read_csv(out / "per_trial.csv")
    walk = per[(per["activity"] == "walking") & (per["status"] == "ok")]
    assert (walk["cop_r_off_frac"] == 0).all() and walk["cop_r_dy_med"].round(3).eq(-0.03).all()
    assert walk["cop_r_fwd_p95"].round(3).eq(0.10).all()
    again = q.main(["--from-csv", str(out / "per_trial.csv"), "--out", str(tmp_path / "out2"), "--no-figures"])
    assert len(again) == 5 and (tmp_path / "out2" / "headline.csv").exists()


CARTER = os.path.join(REPO, "jcf", "full_duration", "training", "walking", "carter_P003_split0_t05")


@pytest.mark.skipif(not os.path.exists(os.path.join(CARTER, "jcf_output_sojr", q.JR_FILE)),
                    reason="carter_P003_split0_t05 SO+JR labels not present")
def test_real_carter_matches_gc_knee_jr_probe():
    """Same folder as results/thesis/gc/knee_jr/probe_carter.csv (gc_knee_jr.check_folder, an independent reader)."""
    from analysis.thesis.gc_knee_jr import check_folder
    r = q.trial_row(CARTER)
    p = check_folder(CARTER)
    assert r["status"] == "ok"
    assert r["sojr_r_stance_n"] == p["r_stance_frames"] and r["sojr_l_stance_n"] == p["l_stance_frames"]
    assert r["sojr_r_fy_stance_peak"] == pytest.approx(p["r_fy_peak_bw"])
    assert r["sojr_l_fy_stance_peak"] == pytest.approx(p["l_fy_peak_bw"])
    assert r["sojr_r_fres_peak"] == pytest.approx(p["new_r_F_max_bw"])
    assert r["stored_r_fres_peak"] == pytest.approx(p["stored_r_F_max_bw"])
    assert r["vas_r_max_bw"] == pytest.approx(p["so_vas_r_max_bw"])
    assert r["sojr_r_fy_compressive_frac"] == pytest.approx(p["r_fy_compressive_frac"])
    assert r["so_n_muscles"] == 80

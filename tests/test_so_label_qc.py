"""SO+JR label-quality diagnostics (analysis/thesis/so_label_qc.py) on synthetic trial folders.

Checks the parts that decide a rule fraction: muscle/actuator separation, saturation and reserve counts with their
normalisation, the pelvis residual scaling, the knee loads in BW (left-knee medial sign), the activation-jump scaling
with the frame rate, the per-foot COP test (one fixed and one moving foot), the time-weighted shares, the per-trial
counts, the order in which a peak is attributed to a cause (scan level and sampled peaks with moment arms and
re-solve cost), and the re-attachment of final labels to re-solved frames. No OpenSim needed.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from analysis.thesis import so_label_qc as q  # noqa: E402

MASS, H = 80.0, 1.8
BW = MASS * 9.81
MUSCLES = [f"{m}_{s}" for s in "rl" for m in ("vaslat", "gasmed", "soleus", "tibant", "glmax1")]
RESERVES = [f"reserve_{c}_{s}" for s in "rl" for c in ("hip_flexion", "knee_angle", "ankle_angle")]
PELVIS = [f"reserve_{c}" for c in ("pelvis_tilt", "pelvis_list", "pelvis_rotation", "pelvis_tx", "pelvis_ty",
                                   "pelvis_tz")]


def _write_sto(path, df, name="x"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(f"{name}\nversion=1\nnRows={len(df)}\nnColumns={df.shape[1]}\ninDegrees=no\nendheader\n")
        df.to_csv(f, sep="\t", index=False, float_format="%.8f")


def _folder(tmp_path, n=20, dt=0.01, fixed_cop=False, with_jr=True, fixed_side="rl", long_left=False):
    """A 20-frame trial: right stance on frames 5..14 (1 BW), left stance on 0..4 and 15..19 (or on 0..14 with
    long_left, so the left foot also has a measurable stance run). fixed_cop fixes the COP of the feet in fixed_side."""
    d = tmp_path / "carter_P999_split0_t00"
    d.mkdir(parents=True)
    json.dump({"mass_kg": MASS, "height_m": H}, open(d / "metadata.json", "w"))
    t = np.arange(n) * dt
    act = pd.DataFrame({"time": t, **{m: 0.01 for m in MUSCLES}, "lumbar_ext": 0.5,
                        **{r: 0.0 for r in RESERVES + PELVIS}})
    act.loc[7, "vaslat_r"] = 1.0                       # knee muscle saturated (frame 7)
    act.loc[8, ["soleus_r", "gasmed_r", "glmax1_r"]] = 1.0  # 3 leg muscles saturated, one of them (gasmed) crosses the knee
    act.loc[10, "tibant_r"] = 0.8                      # ankle co-activation on frame 10
    act.loc[10, "soleus_r"] = 0.7
    act.loc[12, "glmax1_r"] = 0.61                     # jump of 0.6 into frame 12 and back out at 13
    act.loc[3, "lumbar_ext"] = 1.0                     # torque actuator: never counted as a muscle
    force = act.copy()
    for m in MUSCLES:
        force[m] = act[m] * 1000.0
    force["reserve_ankle_angle_r"] = 0.0
    force.loc[9, "reserve_ankle_angle_r"] = 0.2 * MASS  # 0.2 N*m/kg > RES_NMKG
    force.loc[11, "reserve_pelvis_tilt"] = 200.0       # residual moment
    _write_sto(str(d / "jcf_output" / q.SO_ACT), act)
    _write_sto(str(d / "jcf_output" / q.SO_FORCES), force)
    right = (np.arange(n) >= 5) & (np.arange(n) < 15)
    left = (np.arange(n) < 15) if long_left else ~right
    gt = np.arange(n) * dt
    grf = {"time": gt}
    for s, on in (("r", right), ("l", left)):
        grf[f"calcn_{s}_force_vx"] = 0.0
        grf[f"calcn_{s}_force_vy"] = np.where(on, BW, 0.0)
        grf[f"calcn_{s}_force_vz"] = 0.0
        grf[f"calcn_{s}_force_px"] = 0.5 if (fixed_cop and s in fixed_side) else np.linspace(0.0, 0.4, n)
        grf[f"calcn_{s}_force_py"] = 0.0
        grf[f"calcn_{s}_force_pz"] = 0.1
        for a in "xyz":
            grf[f"calcn_{s}_torque_{a}"] = 0.0
    _write_sto(str(d / "grf_data.mot"), pd.DataFrame(grf), "GRF")
    ik = pd.DataFrame({"time": t, "pelvis_ty": 1.0, "knee_angle_r": np.radians(20.0), "knee_angle_l": np.radians(5.0)})
    _write_sto(str(d / "ik_results.mot"), ik, "IK")
    if with_jr:
        jr = {"time": t}
        for s in "rl":
            pre = q.knee_prefix(s)
            jr[pre + "fx"] = 0.0
            jr[pre + "fy"] = -2.0 * BW
            jr[pre + "fz"] = 0.0
            jr[pre + "mx"] = 0.02 * BW * H
            for c in ("my", "mz", "px", "py", "pz"):
                jr[pre + c] = 0.0
        jr[q.knee_prefix("r") + "fy"] = np.where(np.arange(n) == 9, -6.0, -2.0) * BW
        _write_sto(str(d / q.OUT_SUBDIR / q.JR_FILE), pd.DataFrame(jr), "JR")
    return str(d)


def test_muscles_of_excludes_reserves_and_torque_actuators():
    cols = ["time", "vaslat_r", "soleus_l", "reserve_knee_angle_r", "lumbar_ext", "shoulder_flex_r", "elbow_flex_l",
            "pro_sup_r", "wrist_dev_l", "reserve_pelvis_tx"]
    assert q.muscles_of(cols) == ["vaslat_r", "soleus_l"]
    assert q.muscles_of(cols, "r") == ["vaslat_r"]
    assert q.lab_of("/a/b/moore_subject5_t42_r03/") == "moore"


def test_frame_table_counts_and_units(tmp_path):
    ft = q.frame_table(_folder(tmp_path))
    assert ft.attrs["bw"] == pytest.approx(BW)
    assert ft.nsat_knee_r[7] == 1 and ft.nsat_knee_r[8] == 1      # vaslat, then gasmed
    assert ft.nsat_leg_r[8] == 3 and ft.nsat_leg_r[3] == 0          # lumbar_ext is not a muscle
    assert ft.nsat_all[8] == 3
    assert ft.res_ankle_r[9] == pytest.approx(0.2) and ft.res_leg_r[9] == pytest.approx(0.2)
    assert ft.res_knee_r.max() == 0
    # pelvis moment: 200 N*m / (peak total GRF 1 BW x pelvis height 1 m)
    assert ft.pelvis_m_pct[11] == pytest.approx(100 * 200 / BW)
    assert ft.fy_r[9] == pytest.approx(-6.0) and ft.fres_r[0] == pytest.approx(2.0)
    assert ft.mx_r[0] == pytest.approx(0.02)
    # F_medial = 0.5|F_y| + M_x/(BW*d) with the physical medial M_x: -mx on the right knee, +mx on the left
    assert ft.fmed_r[0] == pytest.approx(1.0 - 0.02 * H / 0.045)
    assert ft.fmed_l[0] == pytest.approx(1.0 + 0.02 * H / 0.045)
    assert ft.stance_r.sum() == 10 and ft.knee_deg_r[0] == pytest.approx(20.0)
    assert ft.coact_r[10] == pytest.approx(0.7)
    assert ft.jump_r[7] == pytest.approx(0.99)                       # vaslat 0.01 -> 1.0
    assert ft.jump_r[12] == pytest.approx(0.6) and ft.jump_r[13] == pytest.approx(0.6)
    assert ft.jump_r[0] == 0.0


def test_jump_is_scaled_to_10_ms(tmp_path):
    a = q.frame_table(_folder(tmp_path / "a", dt=0.01))
    b = q.frame_table(_folder(tmp_path / "b", dt=0.005))
    np.testing.assert_allclose(b.jump_r.to_numpy(), 2.0 * a.jump_r.to_numpy())


def test_cop_travel_separates_fixed_and_moving(tmp_path):
    fixed = q.frame_table(_folder(tmp_path / "f", fixed_cop=True))
    moving = q.frame_table(_folder(tmp_path / "m"))
    assert fixed.attrs["cop_travel_r"] == pytest.approx(0.0)
    assert moving.attrs["cop_travel_r"] == pytest.approx(0.4 * 9 / 19)   # 10 stance frames of a 0..0.4 m ramp
    assert fixed.attrs["cop_runs_r"] == 1
    assert moving.attrs["cop_speed_r"] == pytest.approx(0.4 * 9 / 19 / 0.1)  # travel over 10 frames of 10 ms
    assert np.isnan(moving.attrs["cop_speed_l"])                            # left runs are shorter than MIN_RUN
    assert fixed.attrs["cop_dur_r"] == pytest.approx(0.1)
    rows = {r["side"]: r for r in q.trial_summary(fixed, "carter_P999_split0_t00")}
    assert rows["r"]["fixed_cop"] and rows["r"]["fixed_cop_trial"]
    # the flag alone does not attribute the peak: frame 9 has no extensor force, so it is not the fixed-COP mechanism
    assert rows["r"]["cause"] != "fixed_cop"
    assert not rows["l"]["fixed_cop"]                    # left speed is NaN (no run of MIN_RUN frames)
    rows = {r["side"]: r for r in q.trial_summary(moving, "carter_P999_split0_t00")}
    assert not rows["r"]["fixed_cop"] and not rows["r"]["fixed_cop_trial"]


def test_fixed_cop_is_per_foot(tmp_path):
    """Right COP fixed, left COP moving: the right-knee row is flagged, the left-knee row is not, and the trial flag is
    set on both rows (the old all-feet rule passed this trial as moving)."""
    ft = q.frame_table(_folder(tmp_path / "a", fixed_cop=True, fixed_side="r", long_left=True))
    assert ft.attrs["cop_speed_r"] == pytest.approx(0.0)
    assert ft.attrs["cop_speed_l"] > 0.5                  # 15 frames of the 0..0.4 m ramp in 0.15 s
    rows = {r["side"]: r for r in q.trial_summary(ft, "carter_P999_split0_t00")}
    assert rows["r"]["fixed_cop"] and not rows["l"]["fixed_cop"]
    assert rows["r"]["fixed_cop_trial"] and rows["l"]["fixed_cop_trial"]
    ft = q.frame_table(_folder(tmp_path / "b", fixed_cop=True, fixed_side="l", long_left=True))
    rows = {r["side"]: r for r in q.trial_summary(ft, "carter_P999_split0_t00")}
    assert not rows["r"]["fixed_cop"] and rows["l"]["fixed_cop"]


def test_fixed_cop_threshold_sits_in_the_gap():
    """0.006 m/s (Carter P014 t22) and 0.06 m/s (just above the old 0.05 cut) are fixed; a heel-to-toe 0.6 m/s is not;
    the sensitivity threshold (0.1 m/s) keeps 0.15 m/s as moving; long planted stances and static folders never flag."""
    f = q.fixed_cop_foot
    assert f(0.006, 0.6) and f(0.06, 0.6) and f(0.15, 0.6) and not f(0.6, 0.6)
    assert not f(0.15, 0.6, threshold=q.FIXED_COP_SENS)
    assert not f(0.006, 3.0) and not f(0.006, 0.6, static=True) and not f(np.nan, 0.6)


def test_trial_summary_peak_and_rule_counts(tmp_path):
    rows = {r["side"]: r for r in q.trial_summary(q.frame_table(_folder(tmp_path)), "carter_P999_split0_t00")}
    r = rows["r"]
    assert r["lab"] == "carter" and r["n"] == 20 and r["n_stance"] == 10
    assert r["fy_peak"] == pytest.approx(6.0) and r["pk_time"] == pytest.approx(0.09)
    assert r["pk_t_strike"] == pytest.approx(0.04)
    assert r["n_sat_knee"] == 2 and r["n_sat_leg3"] == 1 and r["n_res_leg"] == 1 and r["n_pelvis_m"] == 1
    assert r["n_coact"] == 1 and r["n_jump"] >= 2
    assert r["n_fy4"] == 1 and r["n_fy4_stance"] == 1 and r["n_fy5"] == 1
    assert r["fres_max"] == pytest.approx(6.0) and r["mx_absmax"] == pytest.approx(0.02)
    assert r["pk_res_leg"] == pytest.approx(0.2)
    assert r["cause"] == "so_nonsmooth"           # jump within 2 frames of the peak (frame 8 -> 9)
    assert r["n_full_steps"] == 1 and r["fy_step_peak_med"] == pytest.approx(6.0)   # one full run, frames 5..14
    assert rows["l"]["n_full_steps"] == 0          # both left runs touch a trial end
    assert rows["l"]["fy_peak"] == pytest.approx(2.0)


def test_summary_without_sojr_labels(tmp_path):
    rows = q.trial_summary(q.frame_table(_folder(tmp_path, with_jr=False)), "x")
    assert not rows[0]["has_sojr"] and "fy_peak" not in rows[0] and rows[0]["n_fy4"] == 0
    assert rows[0]["n_sat_knee"] == 2


def test_time_since_strike():
    st = np.array([0, 0, 1, 1, 1, 1, 0, 1], bool)
    assert q.time_since_strike(st, 5, 0.01) == pytest.approx(0.03)
    assert q.time_since_strike(st, 2, 0.01) == 0.0
    assert np.isnan(q.time_since_strike(st, 6, 0.01))
    assert q.time_since_strike(st, 7, 0.01) == 0.0


def test_classify_peak_order():
    c = q.classify_peak
    # fixed COP needs the mechanism at the peak frame (extensor driven, few antagonists), not only the trial flag
    assert c(dict(fixed_cop=True, pk_ext=8.5, pk_flex=0.1, pk_pelvis_m_pct=50, pk_nsat_knee=3)) == "fixed_cop"
    assert c(dict(fixed_cop=True, pk_ext=2.4, pk_flex=5.1, pk_pelvis_m_pct=1, pk_coact=0.8)) == "so_nonsmooth"
    assert c(dict(fixed_cop=True, pk_pelvis_m_pct=50, pk_nsat_knee=3)) == "dyn_inconsistent"
    assert c(dict(fixed_cop=False, pk_ext=8.5, pk_flex=0.1, pk_pelvis_m_pct=50)) == "dyn_inconsistent"
    assert c(dict(pk_pelvis_m_pct=q.PELVIS_M_PCT + 1, pk_nsat_knee=3)) == "dyn_inconsistent"
    assert c(dict(pk_jump=0.9, pk_nsat_knee=3)) == "so_nonsmooth"
    assert c(dict(pk_coact=0.8)) == "so_nonsmooth"
    assert c(dict(pk_nsat_knee=1)) == "saturation"
    assert c(dict(pk_res_leg=q.RES_NMKG * 2)) == "saturation"
    assert c(dict(pk_t_strike=0.01)) == "strike_transient"
    assert c(dict(pk_t_strike=0.1, pk_ext=3.0, pk_flex=2.0)) == "cocontraction"
    assert c(dict(pk_t_strike=0.1, pk_ext=3.0, pk_flex=0.5, pk_knee_deg=50)) == "deep_flexion"
    assert c(dict(pk_t_strike=0.1, pk_ext=3.0, pk_flex=0.5, pk_knee_deg=20)) == "unexplained"


def test_scan_folder_reports_errors_instead_of_raising(tmp_path):
    d = tmp_path / "moore_subject1_t00"
    (d / "jcf_output").mkdir(parents=True)
    (d / "jcf_output" / q.SO_FORCES).write_text("garbage")
    rows = q.scan_folder((str(d), q.OUT_SUBDIR))
    assert len(rows) == 1 and "error" in rows[0] and rows[0]["lab"] == "moore"
    assert q.scan_folder((str(tmp_path / "missing"), q.OUT_SUBDIR)) == []


def test_clean_stance_stats_and_static_cop(tmp_path):
    rows = {r["side"]: r for r in q.trial_summary(q.frame_table(_folder(tmp_path)), "carter_P999_split0_t00")}
    r = rows["r"]
    # stance frames 5..14; SO flags on 7 (sat), 8 (sat, jump), 9 (reserve, jump: the 6 BW frame), 10 (coact, jump),
    # 11 (pelvis, jump), 12 and 13 (jump) -> clean frames 5, 6, 14 at 2 BW
    assert r["n_clean_so"] == 3 and r["fy_peak_clean_so"] == pytest.approx(2.0)
    assert r["n_clean_core"] == 7 and r["fy_peak_clean_core"] == pytest.approx(2.0)
    d = tmp_path / "s" / "static"
    ft = q.frame_table(_folder(d, fixed_cop=True))
    assert not q.trial_summary(ft, os.path.join(str(d), "carter_P999_split0_t00"))[0]["fixed_cop"]


def test_fixed_cop_ignores_long_planted_stances(tmp_path):
    """A foot planted for 3 s (quasi-static task) with a still COP is not a fixed-COP artefact; a gait stance is."""
    ft = q.frame_table(_folder(tmp_path / "a", n=400, fixed_cop=True))     # right stance frames 5..14 only
    assert q.trial_summary(ft, "x")[0]["fixed_cop"]
    ft.attrs["cop_dur_r"] = 3.0
    assert not q.trial_summary(ft, "x")[0]["fixed_cop"]


def test_classify_sample_peak_by_mechanism():
    """Rows modelled on the sample: P003 t05 (fixed COP), Tiziana S10 t18 (fixed-COP trial, non-optimal SO), P040 t30,
    Moore subject9 t53 (moving COP, residual), Fregly 3GC t03 (deep flexion), Tiziana S12 t05, Han s004 t00."""
    c = q.classify_sample_peak
    p003 = dict(cost_ratio=1.0, knee_angle_net_nmkg=-3.6, knee_angle_cci=0.04, fixed_cop=True, pk_pelvis_m_pct=19,
                pk_knee_deg=36, fy_peak=10.0)
    assert c(p003) == "A"
    assert c(dict(p003, fixed_cop=False)) == "C"          # same moments, moving COP: the residual is the evidence
    assert c(dict(p003, cost_ratio=1.5)) == "B"
    assert c(dict(cost_ratio=1.84, knee_angle_net_nmkg=-0.21, knee_angle_cci=0.90, fixed_cop=True, fy_peak=7.5)) == "B"
    assert c(dict(cost_ratio=1.0, knee_angle_net_nmkg=-1.0, knee_angle_cci=0.6, fixed_cop=True, pk_pelvis_m_pct=2,
                  fy_peak=7.7)) == "F"                    # flagged trial, peak without the mechanism
    assert c(dict(cost_ratio=1.0, knee_angle_net_nmkg=-12.8, knee_angle_cci=0.02, fixed_cop=False,
                  pk_pelvis_m_pct=62, fy_peak=25.7)) == "C"
    assert c(dict(cost_ratio=1.0, knee_angle_net_nmkg=-2.5, knee_angle_cci=0.03, pk_pelvis_m_pct=2.1, pk_knee_deg=69,
                  fy_peak=9.7)) == "D"
    assert c(dict(cost_ratio=1.0, knee_angle_net_nmkg=-0.86, knee_angle_cci=0.68, pk_pelvis_m_pct=2.3,
                  pk_knee_deg=51, fy_peak=20.6)) == "F"
    assert c(dict(cost_ratio=1.0, knee_angle_net_nmkg=-0.79, knee_angle_cci=0.38, pk_pelvis_m_pct=1.1,
                  pk_knee_deg=29, fy_peak=3.9)) == "E"
    assert c(pd.Series(dict(p003, cost_ratio=np.nan))) == "A"   # a missing re-solve does not make it non-optimal


def _trow(lab, dt, n, n_stance, fixed, n_coact_stance=0, folder=None, labelled=True):
    return dict(folder=folder or f"{lab}_x_t00", lab=lab, person=f"{lab}_x", side="r", activity="walking",
                split="training", n=n, n_stance=n_stance, dt=dt, fixed_cop=fixed, fixed_cop_trial=fixed,
                fixed_cop_sens=fixed, has_sojr=labelled, fy_peak=3.0 if labelled else np.nan, fres_max=3.0,
                mx_absmax=0.05, flat=False, n_fy5=0, n_fy5_stance=0, n_any_so=n_coact_stance,
                n_any_so_stance=n_coact_stance,
                fy_peak_clean_so=3.0, fy_mean_clean_so=2.0, fmed_mean_clean_so=1.0, fy_mean=2.0, fmed_mean=1.0,
                **{f"n_{r}": 0 for r in ("sat_knee", "sat_leg3", "res_leg", "pelvis_m", "jump")},
                **{f"n_{r}_stance": 0 for r in ("sat_knee", "sat_leg3", "res_leg", "pelvis_m", "jump")},
                n_coact=n_coact_stance, n_coact_stance=n_coact_stance)


def test_rule_table_time_weighted_shares():
    """A fixed-COP 250 Hz trial of 1,000 frames (4 s) and a moving 100 Hz trial of 400 frames (4 s): 71 % of the
    frames but 50 % of the time; a coact rule on 100 of the 100 Hz stance frames is 50 % of that trial's stance."""
    tq = pd.DataFrame([_trow("carter", 0.004, 1000, 500, True), _trow("moore", 0.01, 400, 200, False, 100)])
    rt = q.rule_table(tq)
    r = rt[(rt.lab == "all") & (rt.rule == "fixed_cop")].iloc[0]
    assert r.trial_frac == pytest.approx(0.5)
    assert r.frame_frac == pytest.approx(1000 / 1400) and r.time_frac == pytest.approx(0.5)
    assert r.stance_frac == pytest.approx(500 / 700) and r.stance_time_frac == pytest.approx(0.5)
    c = rt[(rt.lab == "all") & (rt.rule == "coact")].iloc[0]
    assert c.stance_frac == pytest.approx(100 / 700) and c.stance_time_frac == pytest.approx(1.0 / 4.0)
    rec = rt[(rt.lab == "all") & (rt.rule == "recommended_approx")].iloc[0]
    assert rec.stance_time_frac == pytest.approx(1 - 1.0 / 4.0)     # only the moving trial's 1 s of clean stance left
    assert not rt.rule.isin(["recommended", "training_mask"]).any()  # exact rows need the mask table
    pool = rt[(rt.lab == "pool") & (rt.rule == "fixed_cop")].iloc[0]
    assert pool.trials == 2 and pool.n_labelled == 2 and pool.n_people_labelled == 2
    assert pool.top_person_share_labelled == pytest.approx(0.5)


def test_label_stats_people_and_kept_peaks():
    rows = [_trow("moore", 0.01, 100, 50, False, folder=f"moore_subject9_t{k:02d}") for k in range(3)]
    rows += [_trow("moore", 0.01, 100, 50, True, folder="moore_subject5_t00")]
    tq = pd.DataFrame(rows).drop(columns="person")
    ls = q.label_stats(tq)
    r = ls[ls.lab == "moore"].iloc[0]
    assert r.n_labelled == 4 and r.n_people_labelled == 2 and r.top_person_labelled == "moore_subject9"
    assert r.top_person_share_labelled == pytest.approx(0.75) and r.n_kept_trials == 3
    assert r.kept_fy_peak_med == pytest.approx(3.0)


def test_annotate_resolve_reads_final_labels(tmp_path):
    d = _folder(tmp_path)
    key = ("training", "walking", os.path.basename(d))
    rs = pd.DataFrame([dict(split=key[0], activity=key[1], folder=key[2], lab="carter", time=0.09, cost_ratio=2.0, fy_r=np.nan,
                            knee_r_muscle_sum_so_bw=5.0, knee_r_muscle_sum_qp_bw=2.0)])
    tq = pd.DataFrame([dict(split=key[0], activity=key[1], folder=key[2], side="r", fixed_cop=True,
                            fixed_cop_trial=True, fixed_cop_sens=True, person="carter_P999")])
    out = q.annotate_resolve(rs, tq, {key: d})
    assert out.fy_r[0] == pytest.approx(6.0) and np.isnan(out.fy_r_at_resolve[0])
    assert out.fy_resolved_est[0] == pytest.approx(3.0) and out.nonopt[0] and out.fixed_cop[0]
    st = q.random_frame_stats(out)
    a = st[(st.subset == "all") & (st.lab == "all")].iloc[0]
    assert a.labelled == 1 and a.from_fixed_cop_trials == 1 and a.fy_med_nonopt == pytest.approx(6.0)
    assert st[(st.subset == "moving_cop")].frames.sum() == 0


def test_person_of():
    assert q.person_of("carter_P014_split0_t22") == "carter_P014"
    assert q.person_of("/x/han_s001_split4_t09/") == "han_s001"
    assert q.person_of("moore_subject9_t53_r00") == "moore_subject9"
    assert q.person_of("falisse_subject_3_t00") == "falisse_subject_3"


def test_rule_table_exact_rows_from_the_mask_table():
    """With the mask table, training_mask = 1 - keep_r stance share and recommended = that plus the fy5 frames left,
    both exact; the independence estimate (recommended_approx) overstates the drop when fy5 sits in SO-flagged frames."""
    t1 = _trow("moore", 0.01, 400, 200, False, 100, folder="moore_a_t00")
    t1.update(n_fy5_stance=100, n_fy5=100)                   # every fy5 frame is also an SO (coact) frame
    t2 = _trow("carter", 0.004, 1000, 500, True, folder="carter_b_t00")
    tq = pd.DataFrame([t1, t2])
    mrow = lambda f, lab, n, ns, dt, keep, keep_st, so, so_st, fy5k, fy5k_st: dict(  # noqa: E731
        folder=f, split="training", activity="walking", lab=lab, dt=dt, n=n, n_stance_r=ns, n_keep_r=keep,
        n_keep_stance_r=keep_st, n_so_any_r=so, n_so_any_stance_r=so_st, n_fy5_kept_r=fy5k, n_fy5_kept_stance_r=fy5k_st)
    masks = pd.DataFrame([mrow("moore_a_t00", "moore", 400, 200, 0.01, 300, 100, 100, 100, 10, 10),
                          mrow("carter_b_t00", "carter", 1000, 500, 0.004, 0, 0, 0, 0, 0, 0)])
    rt = q.rule_table(tq, masks=masks)
    get = lambda lab, rule: rt[(rt.lab == lab) & (rt.rule == rule)].iloc[0]  # noqa: E731
    # stance time: moore 2 s, carter 2 s; keep_r keeps 1 s of moore's, the 5 BW rule 0.1 s more
    assert get("all", "training_mask").stance_time_frac == pytest.approx(1 - 1.0 / 4.0)
    assert get("all", "recommended").stance_time_frac == pytest.approx(1 - 0.9 / 4.0)
    assert get("all", "mask_any_so").stance_time_frac == pytest.approx(1.0 / 4.0)
    assert get("all", "training_mask").trial_frac == pytest.approx(0.5)
    assert get("moore", "recommended").stance_time_frac == pytest.approx(1 - 0.9 / 2.0)
    # the estimate applies moore's fy5 share (50 % of stance) to the clean second: 0.5 s left instead of 0.9 s
    assert get("all", "recommended_approx").stance_time_frac == pytest.approx(1 - 0.5 / 4.0)
    assert get("pool", "recommended").stance_time_frac == pytest.approx(1 - 0.9 / 4.0)

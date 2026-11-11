# Aim 2: in vivo validation with the Knee Grand Challenge (exploration notes)

Written 2026-11-10. All numbers come from the CSVs in `results/thesis/gc/explore/`. The scripts that produced them
are in `results/thesis/gc/explore/scripts/` (exploration code, run with `python -I`, or with
`conda run -n jcf python -I` for anything that reads b3d files). Nothing under `jcf/external/grand_challenge/`
was executed or imported; it was only parsed as data.

## 1. Bottom line

- The AddBiomechanics Fregly2012 b3d files hold the same marker data as the Grand Challenge (GC) synchronized files.
  On all 75 b3d trials the markers match the GC `*_trajectories.csv` rows exactly (median |difference| 0.006-0.037 mm).
  b3d frame k is GC row k+1 (AddB dropped the first frame). The two `segment_1` trials are the exceptions:
  DM_mtgait_4 (row = k + 206) and SC_crouch_og3 (row = k + 130).
- On every trial the knee-force CSV has the same time column as `*_trajectories.csv`, row for row.
  So each b3d frame, each frame of our segment folders, and each measured-force sample are linked by an exact
  integer index. No cross-correlation is needed. GRF cross-correlation only serves as a check, and it agrees
  (r = 1.000 for JW/DM; r = 0.9999 for SC at a 1-frame lag, see section 4.3).
- All 139 labelled fregly folders that have IK map to a unique (b3d trial, start frame) by IK content.
- **Folder names are swapped relative to the task text.** `GC1/` holds the 3rd competition (SC, left knee, eTibia, 120 Hz).
  `GC3/` holds the 1st competition (JW, right knee, eKnee, 200 Hz). `GC2/` holds DM, as expected. Code must
  detect the subject from file prefixes (`SC_`, `DM_`/`dm_`, `jw_`), not from folder names.
- StatOp_Val_Data has **no measured forces**. It does have Kaneda et al.'s SO+JR medial and lateral compartment forces for
  SC's left knee on 62 half-stance windows (37 trials). I aligned them to GC rows, and against measured force they give a
  medial RMSE of 0.38-0.39 BW pooled (r = 0.87). This works as a published-method baseline and as an end-to-end check
  of our measured-force pipeline.
- Possible sync problem in DM (GC2): the measured knee-force edges lag the instrumented-foot GRF edges by
  +40 ms (heel strike) and +55 ms (toe-off), median. For JW and SC the lags are near zero or negative. Treat DM with a lag
  sensitivity analysis; do not shift it by default.

## 2. What is on disk (`jcf/external/grand_challenge/`)

| folder | competition | subject | knee | implant file | rate | gait-trial knee files | b3d trials (ours) |
|---|---|---|---|---|---|---|---|
| `GC3/` | 1st (2010) | JW | right | `Gait Trials/EKnee Data/jw_<type>_<n>_knee_forces.csv` | 200 Hz | 20 | 18 (`1GC.b3d`; no jw_tsgait_11, jw_wpgait_12) |
| `GC2/` | 2nd (2011) | DM | right | `Gait Trials/eTibia Data/dm_<type><n>_knee_forces.csv` (lower case, no `_` before the number) | 200 Hz | 20 | 20 (`2GC.b3d`) |
| `GC1/` | 3rd (2012) | SC | **left** | `Gait Trials/eTibia Data/SC_<trial>_knee_forces.csv` | 120 Hz | 42 | 37 (`3GC.b3d`; no SC_bouncy_og7, SC_crouch_og4, SC_mtpgait5, SC_trunksway1, SC_wpgait_l11) |

Every b3d trial has a knee-force file. The match is by name after lower-casing and stripping non-alphanumerics
(`dm_ngait4` = `DM_ngait_4`). `Competition Data Description.pdf` is the 3rd-competition description (subject SC).

### 2.1 Knee-force files

- **SC (eTibia, left implant)**: header `Time(sec),Fx,Fy,Fz,Tx,Ty,Tz,GON/FP,EMG`. Forces are in lbf and moments in in-lbf.
  GON/FP and EMG are raw sync signals. Rows carry a trailing comma the header lacks, so read with `index_col=False`.
  The regression comes from `GC1/Gait Trials/eTibia Data/README.txt`, with F in N and T in N·mm:
  `F_medial = 0.942*Fx + 0.497*Fz + 0.0184*Ty`, `F_lateral = -0.942*Fx + 0.503*Fz - 0.0184*Ty`.
  The README says it was fitted on data filtered with a 6 Hz Woltring filter and validated for gait only.
  Axes (`eTibia Coordinate System.pdf`, left implant): +Fx lateral, +Fy anterior, +Fz inferior (compression),
  +Ty medial. Total axial load is Fz.
- **DM (eTibia, right implant)**: header `time(sec),Sensor1..Sensor6`. Sensors 1-6 are Fx, Fy, Fz, Mx, My, Mz in
  lbf and in-lbf. The x axis points medially for a right implant. From `GC2/Gait Trials/eTibia Data/README.txt`
  (Meyer et al. 2011): `F_medial = -0.876*F_ML + 0.494*F_SI - 0.0189*M_VV` and
  `F_lateral = 0.876*F_ML + 0.505*F_SI + 0.0189*M_VV`, with F_ML = Sensor1, F_SI = Sensor3 and M_VV = Sensor5
  (moment about the anterior axis). Total axial load is Sensor3.
- **JW (eKnee, 1st generation)**: header `17:00,Sensor1..Sensor4`. The first cell is an Excel artefact for the time
  column. There are four uniaxial load cells in lbf: 1 posterior-medial, 2 anterior-medial, 3 anterior-lateral,
  4 posterior-lateral. `EKnee Data/README.txt` says 200 Hz, which the time column confirms
  (`ORIGINAL README.txt` says 120 Hz and is stale). No regression is provided. **Assumption to verify:** medial = S1 + S2,
  lateral = S3 + S4, total = sum of all four. This is the direct channel sum; check the method in Zhao et al. 2007
  (`zhao2007` in refs.bib), which used this implant.
- Conversions: 1 lbf = 4.4482216 N; 1 in·lbf = 112.98483 N·mm. No NaNs were found.

### 2.2 Motion and plate files

- `*_trajectories.csv`: markers in mm, lab Z up. SC has 62 markers, including virtual LHJC/RHJC/LKJC/RKJC/LAJC/RAJC;
  JW and DM have 32. SC's `_trajectories.csv` has a `Frame` column (e.g. SC_ngait_og5 starts at original frame 312,
  t = 2.590 s); the others start at t = 0 (some DM trials at 0.008-0.125 s).
- `*_grf.csv`: SC uses 3 plates at 1200 Hz; JW/DM use 4 plates at 1000 Hz. Columns are Fx, Fy, Fz, COPx/y/z and Tz (or Mz)
  per plate, in N and mm. These files also have trailing commas.
- Lab to OpenSim/b3d axes, from the marker match: JW/DM x = X, y = Z, z = -Y. SC x = -X, y = Z, z = Y.
  Both are proper rotations.

## 3. Our data for the same people

- b3d: `with_arm/training/Fregly2012_Formatted_With_Arm/{1GC,2GC}/` (JW 64.6 kg / 1.66 m; DM 67.0 kg / 1.72 m) and
  `with_arm/testing/Fregly2012_Formatted_With_Arm/3GC/` (SC 78.4 kg / 1.67 m). `getTrialOriginalName` gives the GC
  name exactly. The frame `.t` field is 0 on every frame, so use frame indices.
- AddB timestep quirk: `getTrialTimestep` is 0.004990-0.004995 s (JW/DM) and 0.008300-0.008312 s (SC), against true
  values of 0.005 s and 0.008333 s (about 0.2-0.3 % short). Our `ik_results.mot` time is `i * dt_b3d`, so convert
  segment time to a frame index with `dt_b3d`, then to GC time with the true GC row times. Never mix the two clocks.
- Segment folders: `jcf/full_duration/training/walking/fregly_{1GC,2GC}_*` and
  `jcf/full_duration/testing/walking/fregly_3GC_*`.
  - Re-running `pipeline/batch_process.scan_b3d_all_runs` plus the naming logic of `batch_process_parallel.py` gives
    every 3GC folder, 44 of 45 for 1GC and 47 of 51 for 2GC. Mapping by IK content
    (`segment_map_by_content.csv`) places **all 139 folders with IK**, uniquely and exactly:
    1GC 45, 2GC 51, 3GC 43.
  - The scan logic does not produce five older folders: `fregly_1GC_t02_r02` (frame 424), `fregly_2GC_t11_r00/_r01/_r02`
    (frames 164/303/433) and `fregly_2GC_t16_r02` (frame 483). **`fregly_2GC_t11_r00` duplicates `fregly_2GC_t11`**
    (same trial, start and length), so dedupe it.
  - Why the 22 SC folders have no IK: every one has 36-49 frames, and `pipeline/b3d_to_opensim.py` skips a slice with fewer
    than 50 valid frames ("SKIP ... only N valid frames"). They are not failures. data_scout.md is wrong on this.
  - Example: `fregly_3GC_t18_r00` is SC_ngait_og5 b3d frames 112-176, which are GC rows 113-177 (t = 2.590 + row/120).
  - JR labels cover only part of each segment (e.g. t18_r00 IK covers 0-0.532 s, JR 0.075-0.449 s, because of SO
    padding). `_common_time_grid` handles this.
- Stance coverage of the instrumented limb in the labelled segments (folders with IK; GRF of the instrumented foot):

  | subject | segments | frames | full stance | contains heel strike | contains toe-off |
  |---|---|---|---|---|---|
  | JW (R) | 44 | 5602 | 2 | 5 | 15 |
  | DM (R) | 47 | 5570 | 3 | 3 | 15 |
  | SC (L) | 43 | 3197 | 0 | 9 | 15 |

- Complete instrumented-limb stance phases inside the b3d frame range of the full trials (kinematic contact from GC
  heel/toe markers): **JW 33, DM 42, SC 67**. Force plates catch only about 12-27 clean ones per subject. The eTibia
  records the whole trial, and the surrogate needs no GRF, so full-trial evaluation gets 2-5 times more stances.
- b3d dynamics-pass IK is usable on full trials. Its maximum difference from the kinematics pass over 5 key DOFs has a
  median of about 3 degrees whether or not the frame has GRF (no break at missing-GRF boundaries). There are 27 isolated
  velocity spikes above 1500 deg/s in 32,804 frames, mostly knee angle in swing.

## 4. Synchronisation

### 4.1 Knee force and motion (organiser sync)
The 3rd-competition description (section III.C) says the video motion and GRF, EMG and eTibia data were synchronised
with a common goniometer or EMG signal, then resampled and filtered. On all 75 b3d trials the knee-force time column
equals the trajectories time column (`grf_and_knee_timebase_check.csv`). **Measured row r is therefore at marker row r.**

### 4.2 b3d and GC (exact)
`alignment_b3d_to_gc.csv` was built by searching integer offsets on marker height (b3d y against GC Z) and then checking
horizontally. Every trial has a sub-0.04 mm residual on 32-69 markers. The offset is 1, except DM_mtgait_4 (206) and
SC_crouch_og3 (130).

Mapping for a segment folder with b3d trial T and start frame s, at segment time t (any grid, including the 100 Hz
deploy grid):
`frame = s + t / dt_b3d(T)`; `gc_row = frame + offset(T)`; `gc_time = t_gc[0] + gc_row / fs_gc`,
where fs_gc = 120 Hz for SC and 200 Hz for JW/DM, and t_gc[0] is the first time in the trajectories file.
Interpolate the measured force at gc_time.

### 4.3 GRF one-frame lag in SC's b3d (and StatOp)
b3d raw plate Fz equals GC plate Fz resampled at the marker times (ratio 1.00). For JW/DM the best lag is 0 frames.
For SC it is 1 frame: the b3d GRF at frame k is the GC GRF of row k, not row k+1. StatOp_Val shows the same thing.
Its `.trc` frame 0 is GC row 1, but its GRF time 0 is GC row 0 (`statop_grf_time_origin.csv`).

So for SC, AddB and StatOp put the GRF 8.3 ms late relative to the kinematics, compared with the organisers' sync.
This affects SC's SO+JR labels and the GRF-based contact flags by one frame. Align on kinematics (marker rows).

### 4.4 DM sync check (open issue)
These checks time the knee-force edges against the plate GRF edges of the instrumented foot, as median (knee minus GRF)
in ms. Files: `edge_timing.csv` and `stance_lag.csv`.

| subject | heel strike | toe-off | shape cross-correlation lag |
|---|---|---|---|
| JW | -100 | -37 | 0 |
| DM | **+40** | **+55** | **+56** |
| SC | +21 | -8 | +10 |

DM's knee force trails its GRF on both edges, unlike the others. This could be a sync offset of roughly 50-90 ms in the GC2
package, or a waveform difference specific to DM. The plan: report DM at lag 0 (the organisers' data) and add a lag
sweep from -100 to +100 ms as a sensitivity analysis. The 2024 forum question on the time base (p=49262, data_scout.md)
is unanswered.

## 5. Plausibility of the measured forces (`measured_summary.csv`, BW from b3d mass)

| subject | medial peak (median over trials) | lateral peak | total peak | medial share in plate stance |
|---|---|---|---|---|
| JW | 2.11 | 1.04 | 2.3-3.1 | 0.55-0.68 |
| DM (valgus) | 1.67 | 1.69 | 2.7-3.8 | 0.38-0.52 (lateral dominant, as the forum notes) |
| SC normal gait | 1.91-2.04 | 0.67-0.79 | 2.57-2.65 | 0.75-0.78 |

SC bouncy and crouch gait reach 4.7-5.6 BW total, and medial thrust 4.3-4.7 BW. The regression gives slightly negative medial values in swing
(down to -0.47 BW), so evaluate on stance frames only. `measured_examples.png` shows one trial per subject.

## 6. StatOp_Val_Data (Kaneda et al. 2023)

- The README describes trc, IK, ID, external loads, scaled models and SO_JRA for 3GC, 4GC and 6GC. **No measured forces**
  are included. 3GC covers the same 37 SC trials as our b3d. There are 62 SO windows (`SO_JRA/SC_<trial>_stance{1,2}/instKnee/`),
  each a half stance; `stance1`/`stance2` are different steps, combined because the left foot missed the plate.
  `Gait/StanceTimes_*.mat` are MATLAB tables (MCOS), which scipy cannot read easily. The JR file time ranges give
  the same windows.
- Medial and lateral force come from `results_JointReaction_JointRxn_ReactionLoads.sto`, columns
  `med_cond_joint_l_on_med_cond_l_in_med_cond_l_fy` and `lat_cond_joint_l_on_lat_cond_l_in_lat_cond_l_fy`.
  They are negative for compression, so use the absolute value. Units are N.
- Time base: the StatOp kinematic time tau maps to GC row `1 + round(120*tau)`. This was checked on all 62 IK runs
  with model marker locations against GC markers (best offset +1 every time, error 3-5 mm). The StatOp GRF is one row
  earlier (section 4.3).
- Against measured medial (BW 78.4 kg; `statop_vs_measured.csv`, `statop_vs_measured.png`): the mean RMSE per window
  is 0.36 BW (r 0.87); pooled it is 0.39 BW (0.38 at -1 row, 0.385 with a 6 Hz low-pass on the eTibia). Lateral RMSE is 0.40-0.47 BW.
  data_scout.md quotes Kaneda's published value as 0.32 BW RMSE (and peak MAE 0.16 / 0.31 BW). The metric definition is
  unverified, so before citing it, find Kaneda's exact metric and reproduce it.

## 7. Pilot smoke test (labelled segments, right knee, JW/DM; not for the paper)

`pilot_segments_jw_dm.csv` covers checkpoints A (fpopt) and B (notrans _2) with deployment features
(`tools/evaluate_fmedial.features_for` with deploy) on frames in right stance. Only segments that pass the training label
filters are included (JW 27 segments, 889 frames; DM 23, 874).

| subject | model | surrogate F_med vs measured medial: MAE (bias), BW | OpenSim SO+JR F_med vs measured | OpenSim abs(F_y) vs measured total |
|---|---|---|---|---|
| JW | A | 0.52 (-0.21) | 0.68 (-0.52) | 0.83 (-0.80) |
| JW | B | 0.47 (-0.19) | same | same |
| DM | A | 0.73 (-0.40) | 0.68 (-0.67) | 1.37 (-1.37) |
| DM | B | 0.61 (-0.29) | same | same |

These are partial stances on half-stance SO windows with padding effects, plus DM's possible offset, so they are
indicative only. They do show the whole chain working: name mapping, offsets, units, BW and stance gating.
The OpenSim labels under-predict measured total force by about 0.8 BW (JW) and 1.4 BW (DM).

## 8. Decisions

1. **Time alignment** uses exact frame-index mapping from marker identity (section 4.2), not GRF cross-correlation.
   GRF cross-correlation stays as an automatic check: flag a trial if the best lag is not 0 (JW/DM) or 1 (SC), or if
   r < 0.99. Store the per-trial offsets in `results/thesis/gc/alignment.csv` and recompute them in code; do not
   hard-code them.
2. **Body weight**: use the b3d mass (64.6, 67.0 and 78.4 kg), the same mass the surrogate gets as an input and that
   normalises its outputs. Report N as well. The published masses (JW 68 kg, DM 70 kg) would scale the BW numbers by
   about 0.95.
3. **Quantity compared**: measured medial force against surrogate F_medial = phi*abs(F_y) + M_x*H/d (phi 0.5, d 45 mm,
   with the sweep). Also compare measured total against abs(F_y), so the force and moment parts stay separable.
4. **SC is a left knee.** Mirror the surrogate inputs at the feature level:
   - swap the r and l DOFs (hip flexion/adduction/rotation, knee, ankle; signs unchanged);
   - negate pelvis_list, pelvis_rotation and pelvis_tz (pelvis_tilt and the other translations unchanged);
   - negate the z component of rel_com, rel_r and rel_l, and swap rel_r with rel_l;
   - swap the R and L contact flags.

   The prediction is then the left knee, medial-positive. For our own left-knee OpenSim labels, use F_y as is and
   M_x_left = -mx(walker_knee_l), and check that it is positive in left stance.
5. **Two evaluation sets**:
   - **Set A, labelled segments**: identical frames for measured, our SO+JR and the surrogate. JW/DM work now. SC needs a
     left-knee JR rerun, which the user runs.
   - **Set B, full b3d trials**: all instrumented stances (33/42/67) for measured against the surrogate. The SC StatOp
     baseline is added on its own windows. Contact flags come from kinematics (`imu.contact.detect_contact`), checked
     against plate flags. Defaults are about 40 ms late at onset and 20-30 ms early at offset for JW/DM; for SC they are about 0 at onset and
     18 ms early at offset, with outliers in crouch and bouncy gait,
     so tune `height_margin` and `speed_thresh` per subject on plate stances and report the edge errors. In a sensitivity
     run, use plate flags where the foot is on a plate.
6. **Primary metrics** (stance frames of the instrumented limb, pooled per subject and per gait type):
   - MAE, RMSE, bias and r;
   - first and second peak errors per stance (peaks in 0-50 % and 50-100 % of stance);
   - **gait-modification effect**: for each subject and gait type, the change in mean peak medial force relative to
     normal gait, measured against predicted, with sign agreement and correlation. This tests whether the surrogate gets
     the direction of a pain reduction right;
   - implied (phi, d) per subject by least squares of measured medial on abs(F_y) and M_x*H. Do this for both the surrogate
     and the OpenSim outputs (thesis_software_plan Task 2).

## 9. Implementation plan

New code goes in `analysis/thesis/` and follows the repo style: module docstring with usage, argparse, a
`sys.path.insert` of the repo root, and matplotlib Agg. Every script takes `--ckpt` (one or more) and `--labels`, so it
can be rerun when the `_t100` models arrive. Outputs go to `results/thesis/gc/<name>/`.

1. **`analysis/thesis/gc_data.py`** (library; no OpenSim, no torch):
   - `SUBJECTS`: code → side, b3d path, b3d mass/height, fs_gc, knee reader, and the subject's file prefix.
     `find_subject_dirs(root)` detects the GC folder for each subject from its file prefixes.
   - `read_gc_csv(path)`: `index_col=False`, drops `Unnamed:*`, treats column 0 as time.
     `trial_key(name)`: lower-case alphanumeric.
   - `measured_forces(code, path)` → DataFrame with t, medial_N, lateral_N, total_N, using the regressions and unit
     conversions in section 2.1.
   - `gc_trial_files(code)` → {key: (trajectories, grf, knee)}.
   - `align_b3d(code)` (jcf env, nimblephysics): the marker-identity offset per b3d trial, plus a GRF lag and
     correlation check. Writes or reads `results/thesis/gc/alignment.csv`.
   - `segment_map(code)` (jcf env): maps fregly folders to (trial, start) by IK content and drops duplicates.
     Writes or reads `results/thesis/gc/segment_map.csv`.
   - `seg_time_to_gc_time(...)` (section 4.2).
   - `statop_windows(root)`: reads StatOp JR medial/lateral and maps tau to a GC row.
2. **`analysis/thesis/gc_features.py`** (library):
   - `deploy_features(ik_df, com_npz, foot_npz, r_contact, l_contact, mass, height, cfg)`: a label-free copy of
     `train_mlp._build_com_features` with time_exact and resampling to 100 Hz, followed by `_windowize_positions_only`.
     It returns the per-frame matrix, the windowed features and the 100 Hz time grid. It is needed because
     `_build_com_features` drops trials whose labels fail the filters and requires JR files.
   - `mirror_per_frame(feat, cfg)` (section 8.4).
   - `kinematic_contact(foot_points, fs, **kw)`, a thin wrapper of `imu.contact.detect_contact`.
3. **`analysis/thesis/gc_export_fulltrials.py`** (jcf env, `python -I`; about 1 min):
   - For each of the 75 b3d trials, write `results/thesis/gc/fulltrial/<code>/<orig>/ik_results.mot` from the
     dynamics-pass `pos` for all frames, on the true GC time `t_gc[row]`. Copy `scaled_model.osim` (identical across a
     subject's segments) and write `metadata.json`.
   - Then call `pipeline/compute_com_features.compute_for_trial` for `com_features.npz`, for the calcn origin and for
     the fpopt point (0.13, -0.01, 0). This is FK only, not SO.
   - Also store the calcn and toes station tracks for kinematic contact.
4. **`analysis/thesis/gc_validate.py`** (main; `--ckpt ... --labels ... --sets A B --phi --d-mm --lag-sweep-ms --name`):
   - Set A: for each fregly folder, compute deploy features (mirrored for SC), predict, and load the OpenSim labels
     (`jcf_output/` right knee for JW/DM; `jcf_output_left/` for SC when present). Map to GC time, interpolate the
     measured force, gate on the instrumented-limb GRF stance, and compute metrics.
   - Set B: build features from the full-trial exports with kinematic contact, predict, segment the stances, and
     compute per-stance curves (101 points) and peaks. Add the StatOp windows (SC) as the baseline.
   - Outputs:
     - `summary.csv` (subject x model x set),
     - `stances.csv`,
     - `gait_mods.csv`,
     - `sweep.csv` (phi, d),
     - `implied_phi_d.csv`,
     - `lag_sweep.csv` (DM and all),
     - `contact_check.csv`,
     - `README.md`,
     - `fig_gc_curves.png` (mean ± SD stance curves, measured against model, per subject; candidate for paper Fig. 3),
     - `fig_gc_gaitmods.png` (predicted against measured change from normal gait, one point per gait type and subject).
   - Every surrogate-derived number in the README is marked as provisional, to be replaced after retraining.
5. **`analysis/thesis/gc_left_knee_jr.py`** (**the user runs this**; OpenSim JR rerun):
   - For the 43 SC folders with `jcf_output/BatchJCF_StaticOptimization_force.sto`, write a JR-only AnalyzeTool setup
     from `_analyze_setup.xml`: StaticOptimization off, `forces_file` set to the SO force file, and `joint_names` =
     `walker_knee_l walker_knee_r`, on child and in child. Output goes to `jcf_output_left/`.
   - Run it with a multiprocessing pool. Also re-check that walker_knee_r reproduces the existing labels
     (max abs difference < 1e-3 BW) as a regression test.
   - Command:
     `conda run -n jcf python analysis/thesis/gc_left_knee_jr.py --root jcf/full_duration/testing/walking --pattern 'fregly_3GC_*' --workers 20`.
   - Expected time is under a minute of wall time (JR only), but the compute rules give OpenSim reruns to the user.
6. Optional, phase 2 (user-run): full-stance SO+JR windows from the full-trial exports, so OpenSim is also compared on
   whole stances for all three subjects, both knees. data_scout.md estimates 0.3-11 min of wall time on 23 workers.
7. **Tests** (`tests/test_gc_validation.py`; fast; `pytest.skip` when the data are missing):
   - CSV reader with a trailing comma (synthetic temp file);
   - unit conversions and regressions on synthetic rows (e.g. Fz = 100 lbf gives 444.8 N);
   - `trial_key` for `dm_ngait4` and `DM_ngait_4`;
   - mirroring twice is the identity, and the swaps and sign flips land on the right channels;
   - `deploy_features` equals `train_mlp._build_com_features` (deploy cfg) on one fregly segment;
   - alignment offsets (DM_mtgait_4 = 206, SC_crouch_og3 = 130, SC_ngait_og5 = 1) with a residual under 0.1 mm;
   - segment map (`fregly_3GC_t18_r00` gives trial 18, start 112, GC rows 113-177);
   - measured time equals trajectory time for one trial per subject.
8. **Run commands** for the implementer (each under 10 min):
   - `conda run -n jcf python -I analysis/thesis/gc_export_fulltrials.py`
   - `python -I analysis/thesis/gc_validate.py --ckpt <A> <B> --labels fpopt sept --sets A B --phi 0.4 0.5 0.6 --d-mm 40 45 50 --lag-sweep-ms 100 --name pilot_AB`
   - `pytest -q tests/test_gc_validation.py`

## 10. Open questions and risks

- The JW medial/lateral split (S1+S2 and S3+S4) is unverified. Check Zhao et al. 2007 before citing JW medial numbers.
  JW total force is safe.
- DM sync (section 4.4). If the lag sweep shows a clear optimum near +50 ms for both the surrogate and OpenSim, report it
  as a data limitation. Do not tune per model.
- The SC regression and the DM regression were fitted on gait only. SC bouncy and crouch gait (total 4-5.6 BW) lie outside that fit.
- All three subjects are older TKA patients and outside the healthy training population. None of them was in training
  for A or B (provenance: `--exclude han fregly`).
- 3GC is the paper's test person (`fregly_3GC` is in the test split). JW and DM sit in the *training* folders but were
  excluded from training. Say so explicitly in the paper.
- The gap between Kaneda's published RMSE and our recomputation (section 6) must be closed before StatOp is quoted as
  "published".

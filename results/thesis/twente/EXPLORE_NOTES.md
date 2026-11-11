# Twente (Wang et al. 2023) real-IMU check: exploration notes

Date: 2026-11-10. Aim 3, real Xsens data with an optical reference.
Data, read-only and untrusted: `/home/gellert/Developer/datasets/wang2023_twente/extracted/SubjXX/SubjXX/`.
Everything was parsed as data with `python -I`; nothing from the dataset folders was run or imported.

Raw probe outputs (no personal fields):
- `results/thesis/twente/explore/probe_trials.json`: 63 trials (54 walking, 9 static_pose)
- `results/thesis/twente/explore/pilot_ukf_runs.jsonl`: 72 pilot UKF runs

Prototype scripts, verified, in the session scratchpad `/tmp/claude-1000/-home-gellert-Developer-medical-dt/187d5d35-6c3c-4c5e-8d5a-f69463aa63f9/scratchpad/twente/`:
- `mvnx.py`: streaming MVNX reader. It drops the subject `label` and `originalFilename` attributes, which contain a participant's first name.
- `probe_all.py`: the per-trial probe.
- `pilot_ukf.py`: UKF plus surrogate pilot.
- `mini_batch.sh`, `map_check.py`, `footspeed.py`, `grf_check.py`: helper checks.

Extraction is complete (30 GB). No unrar or wget processes are still running.

## 1. What is there

- **Subjects.** 9 subjects, Subj04-Subj12: 5 men and 4 women, aged 23-30 years.
- **Walking trials.** Each subject has 6 treadmill walking trials, `walk_09 … walk_54`, at 0.9-5.4 km/h, i.e. 0.25, 0.50, 0.75, 1.00, 1.25 and 1.50 m/s. That is 54 trials.
  - Each trial has `Xsens/<trial>.mvnx`, `OS/DataFiles/Subj<XX><trial>{IK,IK_IMU,XLD,XLD_INSOLE,ID,IDf,EMG}.mot/.sto` and `Subj<XX>_<trial>.mat`.
  - IK overlap per trial is 57.5-81.3 s; 0.92 h in total.
  - Subj05 also has a `walk_99.mvnx` with no OpenSim files. It is ignored.
- **Static pose.** All 9 subjects have `Xsens/static_pose.mvnx`, 11-18 s of quiet standing. There is no OpenSim IK of the static pose.
  - `OS/SubjXXstatic_pose.trc` exists for 06-12 only.
  - `Qualisys/static_pose.mat` exists for all 9.
- **Mass and height.** Taken from `Subj<XX>_<trial>.mat → Datastr.Info.subjMass / subjHeight`. The `.osim` scaling XMLs have the same masses, but their height is -1.

| subj | 04 | 05 | 06 | 07 | 08 | 09 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|
| mass kg | 75.2 | 65.1 | 56.0 | 80.8 | 61.5 | 81.0 | 61.6 | 62.4 | 69.0 |
| height m | 1.80 | 1.71 | 1.685 | 1.82 | 1.62 | 1.82 | 1.66 | 1.68 | 1.79 |

## 2. MVNX structure (MVN 2021.0, mvnx version 4)

- **Header.**
  - `subject` has `frameRate=240`, `configuration=LowerBodyPlus` and `userScenario=singleLevel`.
  - `processingQuality` is `HD` (offline reprocessing) for Subj04 only and `Live` (the real-time, causal filter) for Subj05-12. Live is the closer analogue of a PN3 live stream.
- **Sensors and segments.** There are 8 sensors: Pelvis, T8, RightUpperLeg, RightLowerLeg, RightFoot, LeftUpperLeg, LeftLowerLeg, LeftFoot. MVN outputs 23 body-model segments, arms included, and 22 joints.
- **Frames.**
  - There are 3 special frames (`identity`, `tpose`, `tpose-isb`), then `normal` frames.
  - The normal frames' `index` runs 0…N-1 with no gaps.
  - The `time` attribute is in integer ms, rounded (steps of 3, 4 or 5 ms). Use `index / 240` as the clock; it differs from `time` by at most 1.3 ms.
  - The `tpose` frame is identical for all subjects: it is MVN's generic model pose, not a measurement.
- **Per normal frame:**
  - `orientation`: segment quaternions, 23 × 4
  - `position`: 23 × 3
  - `sensorFreeAcceleration`, `sensorMagneticField`: 8 × 3 each
  - `sensorOrientation`: 8 × 4
  - `jointAngle` and `jointAngleXZY`: 22 × 3
  - `jointAngleErgo*`
  - `centerOfMass`: position, velocity and acceleration, 9 values
  - There is **no** angular velocity and **no** `footContacts` element.
- **Quaternions.** They are scalar-first (w, x, y, z); the `identity` frame is `1 0 0 0`. They map segment (or sensor) coordinates to global (an active rotation). This was checked: the pelvis segment's x axis in global matches the MVN direction of progression to within 1-8°.
- **MVN global frame.** x is forward (the heading reference), y is left, z is up: pelvis z is about 1.05 m, and `jRightHip` is at -y in the pelvis frame.
  - The map to OpenSim ground (x forward, y up, z right) is `M_MVN = [[1,0,0],[0,0,1],[0,-1,0]]`. Pass it as `PN3Estimator(sensor_frame=M_MVN)`.
  - MVN segment frames are aligned with global in the N-pose, so the segment heading axis is +x.
- **Heading.** The MVN heading is not aligned with the lab. The trial-mean pelvis yaw is about ±180° for most subjects; Subj08 is -80° and Subj09 -157°. The calibration's heading rotation A absorbs this.
- **Treadmill progression.** `singleLevel` integrates the treadmill walk as overground walking. MVN position moves about 50 m per trial at 1.00-1.07 × belt speed. In MVN global, the stance foot is nearly still: its speed is 0.02-0.23 m/s in stance against 0.9-3.6 m/s in swing.
- **Segment vs sensor orientation.**
  - The offset R_sensor⁻¹ R_segment is constant within a trial: median deviation 0.2-1.1°, p99 0.5-2.7°, worst 8°.
  - It is the same across all trials of a subject, static pose included, to within ≤0.9°. That means one MVN calibration and one sensor mounting per subject session.
  - **Segment orientations have no gaps.** **Sensor orientations have dropouts**, stored as all-zero quaternions, in 52 of 54 walking trials: up to 10 % of frames for one sensor (1,432 frames).
  - The Link pelvis sensor's forward axis is **sensor -z**. Its "up" axis is -x for 8 subjects and +x for Subj05.
- **What `IK_IMU.mot` is.** It is *not* OpenSense IK. It is MVN's own `jointAngle` output renamed to gait2392 coordinates, resampled to 100 Hz and shifted onto the IK clock (r = 1.0000, RMSE ≤ 0.13°):
  - knee_angle = -jKnee.z
  - hip_flexion = +jHip.z
  - hip_adduction = -jHip.x
  - hip_rotation = +jHip.y
  - ankle_angle = +jAnkle.z
  - subtalar = +jAnkle.y
  - Pelvis rotations come from the segment orientations and keep the MVN heading (pelvis_rotation about 185°).
  - pelvis_tx and pelvis_tz are 0.
  - This makes it the **vendor-fusion baseline**.
- **What the `.mat` holds.** It contains the same MVN data parsed (with one extra garbage trailing row), Qualisys markers at 128 Hz, treadmill force at 2048 Hz, Moticon insoles, EMG, and `Resample.Sych.*` (synchronised 100 Hz tables equal to the `.mot` files).

## 3. Calibration: no standing period in the walking trials

- **No standing in walking trials.** Every walking trial starts mid-walk: pelvis speed over the first second is at belt speed, and no 1-s quiet window exists anywhere in any trial. The driver's 1-s standing calibration cannot use the trial itself.
- **The static pose works for calibration.**
  - `static_pose.mvnx` is truly still: segment angular speed p95 < 3.1 °/s and pelvis drift p95 ≤ 1.5 cm/s. It was recorded 5-32 min before the walks with the same MVN calibration (§2).
  - Its pelvis heading is within ±4° of the walking heading, except Subj11 (-8.6°).
  - MVN's own static joint angles are close to zero but not at zero:
    - knee flexion 0-5°, except Subj10: R 10°, L 6°
    - ankle 5-10°
    - hip within ±3°
- **Two calibrations, both piloted.** Neither needs any code change in `pn3_redis`.
  - **`static`** (faithful to the driver): `add_calibration_frame` on static_pose frames at t = 1-2 s, assuming the Rajagopal q = 0.
  - **`mvnmodel`**: one second of synthetic frames in which every segment is `Rz(ψ_static)`. This is the MVN identity pose turned to the static pelvis heading; it treats MVN's N-pose as q = 0.
- **Known failure mode of `static`.** Subj10 stands with about 10° of knee flexion. Calibrated as 0°, extension during stance maps below 0 and is clipped at the URDF limit: the knee sits at 0° in 74 % of frames and its RMSE is 12-15°. `mvnmodel` does not have this problem; there the knee is at 0° in 6 % of frames.
- **Start-up.** After calibration the UKF starts from the standing q0 while the subject is mid-stride. Score from 3 s after MVN start.

## 4. Synchronisation (exact)

- **The rule:** `t_MVN = t_IK + L`, with `L = 0.5 + Datastr.Resample.Sych.DeltaT.Motion` (from `Subj<XX>_<trial>.mat`).
  - This holds to the millisecond for all 54 trials. The check was a cross-correlation of MVN knee flexion (240 Hz) with `IK_IMU.mot` knee, r = 1.0000.
  - L is 0.530-0.894 s; Subj04 walk_54 is the outlier.
  - The 0.5 s is the authors' start crop.
- **Against the optical IK.** The knee cross-correlation lag agrees with L to within -3…+10 ms (median +2 ms R, 0 ms L), with r 0.989-1.000.
  - There is no drift: the lags over the first and last 20 s differ by ≤ 2 ms.
- **Shared clock.** `IK.mot`, `IK_IMU.mot`, `XLD.mot`, `XLD_INSOLE.mot` and `ID.sto` share one 100 Hz clock starting at 0, with identical time columns.
- **Coverage.** The MVN record starts L before IK t = 0 and ends 0.94-1.30 s after the IK end. So the UKF can run on the full MVN record and be scored on the IK overlap.
- **Fallback.** Use the cross-correlation against IK_IMU knee if DeltaT is unavailable; the static_pose `.mat` has no DeltaT, but it is not needed.

## 5. OpenSim files and gait2392 vs Rajagopal conventions

- **Model.** `OS/UTmodel/gait2392_simbody_subjXX_scaled_1.osim` is the model used by IK and ID.
- **IK files.** `IK.mot` is inDegrees=yes, 23 coordinates (no arms), whitespace or tab separated; `IK_IMU.mot` has a trailing tab. Optical IK marker RMS is about 1.6 cm (`OS/Logs`).
- **Ground frame.** x is forward, y up, z right:
  - the right-foot COP has z > 0 and the left z < 0
  - vx < 0 early in stance (braking)
  - the optical pelvis_rotation trial mean is -5…+6.6°, so subjects face +x
  - pelvis_tx moves within 9-36 cm (treadmill)

  This matches the surrogate's convention, so no rotation is needed.
- **Coordinate mapping to Rajagopal (the surrogate and estimator convention):**
  - **knee_angle: flip the sign.** gait2392 range is -120…+10° with flexion negative; Rajagopal is 0…120° with flexion positive.
  - **Same names, axes and signs for the rest:**
    - pelvis: ground_pelvis Rz(tilt) Rx(list) Ry(rotation), plus translations
    - hip: Z flexion, X adduction, Y rotation
    - ankle: the Delp oblique axis (-0.105, -0.174, 0.979)
    - subtalar
    - lumbar
  - Anatomical frame offsets between the two models: mostly small for the legs (median |bias| of IK_IMU vs optical at most
    1 deg; the ankle reaches +4 to +7 deg in 3 subjects), but NOT for the trunk and pelvis (checked in the
    full run, `all_AB/angles_table.csv`). IK_IMU's lumbar_extension is a median +11 deg above the optical one (8-17 deg
    across subjects, error SD about 1 deg, so near-constant), and its pelvis tilt -2.5 deg below. Lumbar is not a surrogate
    input, but it moves the COM input. `twente_imu.py` now scores lumbar_* and writes decomposition rows (`O/V:*`,
    `Omatch:*`, `decomposition.csv`) that separate this definition difference from the IMU error.
  - **Gotcha, already hit:** when converting degrees to radians, test for the translations by exact name. A `startswith("pelvis_t")` test also matches `pelvis_tilt`.

## 6. GRF and contact flags

- **Which file.** `XLD.mot` holds the treadmill force (Motek ForceLink split-belt, resampled to 100 Hz on the IK clock), in OpenSim ground axes.
  - `ground_force_v{x,y,z}` is the **right** foot (ExternalLoads → calcn_r).
  - `l_ground_force_v*` is the **left** foot (calcn_l).
  - **Vertical is vy, in N.**
  - Contact = `vy > 0.05 · mass · g`, the training definition.
- **Signal properties.**
  - Values below about 20 N are set to exactly 0 (35 % of samples, i.e. swing).
  - There are no negative values.
  - Stance fraction is 0.61-0.79 and double support 0.22-0.57, both falling with speed. There is no flight phase.
- **BW check.** Mean (R + L)/BW is 1.00-1.04, but **Subj10 is 1.15 at all six speeds**. Either the recorded mass (61.6 kg) is low or the plate scale is off. Keep 61.6 kg and flag it.
- **Split-belt risk.** A few trials have asymmetric stance fractions (Subj06 walk_54 R 0.70 / L 0.62; Subj11 walk_36 L 0.69), which may be cross-over steps.
- **`XLD_INSOLE.mot`.** This is the Moticon insole estimate: vertical force and COP only, same clock. It is a *portable* contact source. Its contact agreement with the treadmill at 5 % BW is median 0.94 R / 0.97 L, minimum 0.62 (Subj08 walk_45 R).

## 7. Pilot results (feasibility only; these are not the thesis numbers)

**Setup:**
- PN3Estimator as in the driver: σ_orient 2°, 19 coordinates, torso from the T8 segment, height-scaled generic URDF.
- MVN data resampled to 60 Hz (the PN3 rate) with Slerp, on the IK clock.
- LiveSurrogate in the loop for both pilot checkpoints:
  - A = `*_contact_fpopt_w20_fmW0p5_e100`
  - B = `*_contact_w20_fmW0p5_e100_2`
- **Optical path:** gait2392 optical IK, knee sign flipped → the same LowerBody FK → checkpoint foot point → centred 20-frame windows at 100 Hz → `predict_windowed`, with treadmill contact.
- **What the numbers measure:** "F_med Δ" below is the surrogate on IMU inputs against the surrogate on optical inputs. It is **not** error against measured knee load; the dataset has no knee-load labels.

Joint angles were scored against the optical IK from 3 s after MVN start, over 18 runs (9 subjects × walk_18 and walk_45).
Cost is about 4.5 ms/frame including two surrogates, about 15 s per 60-s trial on one core.

| variant | knee RMSE °, median (max) | hip flex ° | ankle ° | F_med Δ MAE BW, B / A | r, B / A |
|---|---|---|---|---|---|
| `mvnmodel` + treadmill contact | 2.8 (4.4) | 2.5 | 2.4 (9.2) | 0.121 / 0.146 | 0.95 / 0.92 |
| `mvnmodel` + driver gyro contact | 2.8 | 3.3 | 2.8 | 0.201 / 0.235 | 0.85 / 0.83 |
| `static` + treadmill contact | 5.4 (14.9, Subj10) | 3.5 | 5.0 (bias about -5) | 0.144 / 0.161 | 0.93 / 0.89 |
| raw `sensorOrientation` + `static` + treadmill | 5.8 (15.6, Subj10) | 3.7 | 5.5 | 0.123 / 0.154 | 0.94 / 0.88 |

**Vendor baseline:** MVN's own angles (IK_IMU) against optical IK, all 54 trials, median RMSE:

| joint | RMSE ° | r |
|---|---|---|
| knee | 2.3 / 2.7 | 0.997 |
| hip flexion | 2.5-2.6 | |
| ankle | 2.5 / 3.0 | |
| hip adduction | 3.6 / 3.8 | |
| hip rotation | 5.3 / 6.1 | 0.37-0.41 |
| pelvis tilt | 3.1 (bias -2.9) | |

The optical hip rotation is itself poorly determined. Subj04's hip-rotation bias is +11.8° even for MVN.

**Findings:**
1. **The driver's gyro contact defaults fail below about 1.2 m/s.** The swing-foot smoothed angular speed peaks at 150-320 °/s at 0.25-0.75 m/s, which never reaches the 400 °/s off-threshold. The stance flag stays on: at 0.5 m/s, agreement equals the stance fraction (0.64-0.69). At 1.25 m/s it is 0.61-0.94 across subjects, and for Subj04 at 1.5 m/s it is 0.80 R / 0.87 L. This is the main source of IMU-side error at slow speeds.
2. **The raw sensor path needs `heading_axis = -z`.** With +x the heading is wrong and 16 of 18 runs diverge. It is otherwise mathematically equivalent to the segment path with `static` calibration, since the sensor-to-segment offset is constant.
3. **Optical-path features are in distribution (checked on Subj04 walk_27, centre frame).** Their z-scores against the checkpoint's training normalisation are |z| < 1.1 for q and |z| < 2.6 for COM/feet.

## 8. Plan for the real-IMU check

### 8.1 Code (new, in `analysis/thesis/`; repo style; `sys.path.insert` of the repo root; matplotlib Agg)

**1. `analysis/thesis/twente_io.py`**, data only, numpy/scipy:
- `read_mvnx(path, fields)`: promote the verified scratch reader; it must keep dropping `label` and `originalFilename`.
- `read_mot(path)`: whitespace split; handles the trailing tab.
- `load_trial(subj, trial)`. It returns:
  - mass and height from the `.mat` Info
  - L = 0.5 + DeltaT.Motion, cross-checked by knee cross-correlation against IK_IMU; warn if the two differ by more than 5 ms
  - MVN `index/240 - L` time
  - segment and sensor rotations for the 8 bodies, plus dropout masks (zero quaternions)
  - optical IK converted to Rajagopal convention (radians, knees negated, translations untouched)
  - IK_IMU converted the same way, with pelvis_rotation minus the static-pose MVN heading
  - treadmill and insole contact flags at 5 % BW
- Constants:
  - `M_MVN`
  - `MAP = {pelvis_imu: Pelvis, torso_imu: T8, femur_r_imu: RightUpperLeg, tibia_r_imu: RightLowerLeg, calcn_r_imu: RightFoot, …_l}`
  - `HEADING_AXIS = {"orientation": +x, "sensorOrientation": -z}`
  - `TRIAL_SPEED_MPS`

**2. `analysis/thesis/twente_imu_check.py`**, the CLI. One process per (trial, variant), via multiprocessing with 23 workers.

*IMU path (U).* The same loop as `pn3_redis/driver.run`, called programmatically:
- calibration (`static` or `mvnmodel`)
- `CausalContact` planted (100/200) and stance (150/400, backdate 0.2 s)
- `est.step`
- `LiveSurrogate` for every `--ckpt`

Calibration × field variants:
- `seg-mvnmodel`: primary
- `seg-static`: faithful to the driver
- `sensor-static`: the most PN3-like raw input; dropouts are omitted from that frame's quaternion dict, which the UKF supports

Contact sources (`--contact`): `grf` (oracle), `gyro` (driver defaults), `insole` (portable). Optionally `mvnpos`: MVN foot-segment speed < 0.3 m/s, an analogue of the vendor's own flags.

Rate: `--fs` 60 is the default (PN3); 100 and 240 are sensitivity runs.

*Optical path (O, reference) and vendor path (V).*
- O uses optical IK; V uses IK_IMU angles.
- Both go through the same `LowerBody(height)` FK and checkpoint `foot_point`, centred 20-frame windows at 100 Hz and treadmill flags, batched `predict_windowed`.
- Optional O′: FK with the subject's scaled gait2392 OpenSim model (jcf env, FK only, quick). It tests the generic-URDF COM/feet approximation.

*Metrics*, on the IK overlap from 3 s after MVN start:
- **Angles:** RMSE, bias, de-biased SD and r for the 13 surrogate angles, for U and V against O. Wrap pelvis_rotation.
- **Contact:** agreement, and onset/offset error in ms against treadmill flags.
- **F_medial:**
  - MAE over the whole trial and over right stance
  - bias and r
  - per-stride peak-1 and peak-2 errors (maximum in the first and second half of right stance)
  - for U(contact) and V against O

  Decomposition:
  - kinematics = U-grf against O
  - contact = U-gyro against U-grf
  - vendor fusion = V against O

*Aggregation:* trial → subject mean over the 6 speeds → median and IQR over the 9 subjects. Also report per speed. Report Subj04 (HD) separately from 05-12 (Live) as a check.

*Outputs* go to `results/thesis/twente/<run_name>/`:
- `per_trial.csv`
- `angles_long.csv`
- `summary.json` (with checkpoint provenance)
- `fig_angles_rmse.png`: U variants against vendor, per coordinate
- `fig_fmedial_example.png`: O vs U vs V for one trial and stride
- `fig_vs_speed.png`: F_med Δ MAE and contact agreement against belt speed
- `fig_contact.png`

**3. `tests/test_twente.py`** (pytest, fast):
- MVNX reader on a tiny synthetic MVNX written to tmp_path: w-first order, special frames excluded, zero-quaternion mask, label and originalFilename dropped
- `M_MVN` maps MVN z-up to OpenSim y-up and MVN y-left to OpenSim -z
- gait2392 → Rajagopal conversion: only knees negated; pelvis_tilt converted to radians, translations not
- skipped if the data is missing:
  - Subj04 walk_27: L within 2 ms of the cross-correlation lag and r > 0.999
  - treadmill stance fraction between 0.55 and 0.85

### 8.2 Run (under 10 min wall, so I can run it myself)
- **Size.** 54 trials × 3 calibration/field variants × 3 contacts = 486 runs at about 15 s each, about 5-6 min on 23 workers. The O and V paths take seconds.
- **Checkpoints.** For now, pilot checkpoints A and B. When the user's `*_t100` retrains land, rerun with `--ckpt` set to the new paths. Nothing else changes; LiveSurrogate and `predict_windowed` already accept `*_t100` sets.
- **Example command:**
  ```
  python -I analysis/thesis/twente_imu_check.py --data /home/gellert/Developer/datasets/wang2023_twente/extracted \
      --ckpt <A> <B> --variants seg-mvnmodel seg-static sensor-static --contact grf gyro insole --fs 60 \
      --workers 23 --name pilot_AB
  ```

### 8.3 Optional, for the user to run (OpenSim SO+JR; over 10 min)
The dataset has no knee-load labels, so absolute error against physics-based F_medial needs SO+JR. There are two routes:

- **(a) Preferred: AddBiomechanics.** Use the AddBiomechanics "Wang et al. 2023" b3d files (Rajagopal, the same family as the training labels) through the existing `pipeline/batch_process_parallel.py`.
  - The user would have to fetch them; they are not local.
- **(b) Local fallback: gait2392.** A script that writes SO+JR setups with the provided `gait2392_*_scaled_1.osim`, `IK.mot` and `XLD.mot` (local paths swapped into copies of the XLDset XML), with JR on `knee_r` in `tibia_r`.
  - Run one 8-s window per trial: 54 windows, about 1.6-2 h wall on 23 workers by the 40-50 min per 8-s rule.
  - Caveat: the knee model and muscle set differ from walker_knee_r, so these are approximate labels.
  - I would write the script and print its command; the user runs it.

### 8.4 Decisions for the user
1. **Primary calibration.** `mvnmodel` is robust and vendor-anatomical. `static` is what the live driver does, but it fails for people who stand with flexed knees (Subj10).
   - Suggestion: report both, with `static` as the deployment-faithful number.
   - A third option is a `static` calibration that keeps MVN's static joint angles as q_cal instead of 0. It needs a small estimator option.
2. **Contact detector at slow speeds.** The finding stands as is. Retuning (speed-adaptive thresholds) must use a separate tuning set, e.g. leave-one-subject-out, or the driver's tuning trials only. The insole contact is a portable alternative.
3. **Subj10 BW anomaly (1.15).** Keep the recorded mass, or exclude Subj10 from the BW-normalised metrics.

## 9. Open points and risks
- The gait2392 vs Rajagopal anatomical frame offsets are only partly quantified: the knee sign, and the gait2392-vs-MVN trunk/pelvis offsets above (lumbar_extension +11 deg, pelvis tilt -2.5 deg). Re-running IK with a scaled Rajagopal model would quantify the rest; it is IK only, about 15 s per trial, but needs a Rajagopal marker set for the 33 Twente markers and static `.trc` files (missing for 04/05; they would have to be rebuilt from `Qualisys/static_pose.mat`).
- Treadmill gait only; no overground validation of the legged odometry. The surrogate inputs are pelvis-relative, so they are unaffected.
- The optical hip-rotation reference is weak (MVN and UKF both r ≈ 0.4). Don't over-read hip-rotation errors.
- The Zenodo record states no license (the paper is CC BY 4.0). Do not redistribute derived per-frame data.
- **Privacy.** The MVNX `subject label`, `originalFilename` and some setup-XML paths contain participants' first names. Never copy those fields into results or the paper.

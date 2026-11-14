# S4: seed spread on the optimizer's reference motion (Sep 23 trajectory)

Written by `analysis/thesis/sep23_seeds.py` (no ground truth on this motion; see the script docstring). F_medial = 0.5|F_y| + M_med H/0.045 m with the physical sign for every checkpoint; 1.4 BW limit. Primary inputs: the trial folder's q with the optimizer's pelvis angles converted from the URDF's axes to OpenSim's convention (section 'Pelvis angles'), its contact flags, and the optimizer's exported COM and foot point (com_features.npz); identical for all six checkpoints (asserted), deployment-style windows (exact time, 100 Hz, 20 frames). The previous run fed the pelvis angles as written in ik_results.mot; that is now the `export_copied` variant. Windows: right single stance S1 1.98-2.77 s (80 frames), S2 4.19-4.83 s (65 frames).

## Checkpoints

| model | best epoch | val loss | init seed | foot point | trained labels |
|---|---|---|---|---|---|
| qc_s1 | 56 | 0.300 | 1 | [0.13, -0.01, 0.0] | jcf_output_sojr |
| qc_s2 | 8 | 0.363 | 2 | [0.13, -0.01, 0.0] | jcf_output_sojr |
| qc_s3 | 90 | 0.305 | 3 | [0.13, -0.01, 0.0] | jcf_output_sojr |
| qc_origin | 68 | 0.300 | 1 | [0.0, 0.0, 0.0] | jcf_output_sojr |
| fpopt_legacy | 30 | 0.105 | - | [0.13, -0.01, 0.0] | jcf_output |
| sept_legacy | 32 | 0.105 | - | None | jcf_output |

## Stance readings (primary: converted pelvis angles, exported positions, physical sign)

| model | S1 mean | S1 peak | S1 >1.4 | S2 mean | S2 peak | S2 >1.4 | S1+S2 mean | S1+S2 >1.4 | axial / moment (S1+S2) |
|---|---|---|---|---|---|---|---|---|---|
| qc_s1 | 2.40 | 3.04 | 100% | 2.54 | 3.01 | 100% | 2.46 | 100% | 1.92 / 0.54 |
| qc_s2 | 1.80 | 2.04 | 100% | 1.81 | 2.02 | 100% | 1.80 | 100% | 1.59 / 0.21 |
| qc_s3 | 2.45 | 3.02 | 100% | 2.61 | 3.00 | 100% | 2.52 | 100% | 1.97 / 0.55 |
| qc_origin | 2.73 | 2.96 | 100% | 2.79 | 2.95 | 100% | 2.76 | 100% | 2.08 / 0.68 |
| fpopt_legacy | 0.26 | 0.42 | 0% | 0.37 | 0.48 | 0% | 0.31 | 0% | 0.50 / -0.18 |
| sept_legacy | 0.17 | 0.37 | 0% | 0.35 | 0.48 | 0% | 0.25 | 0% | 0.55 / -0.29 |
| seed_mean | 2.22 | 2.69 | 100% | 2.32 | 2.67 | 100% | 2.26 | 100% | 1.83 / 0.44 |

## Across-seed spread (qc_s1..s3; per-frame max - min, BW; primary inputs)

| window | frames | mean | median | max (at t) | range of stance means | range of peaks | share >1.4 (min-max) | frames where seeds disagree on the limit | s1 vs s3 only: mean / max abs |
|---|---|---|---|---|---|---|---|---|---|
| S1 | 80 | 0.66 | 0.66 | 1.02 (2.58 s) | 1.80-2.45 (0.65) | 2.04-3.04 (1.00) | 100%-100% | 0% | 0.07 / 0.23 |
| S2 | 65 | 0.81 | 0.82 | 1.02 (4.62 s) | 1.81-2.61 (0.80) | 2.02-3.01 (0.99) | 100%-100% | 0% | 0.09 / 0.23 |
| S1+S2 | 145 | 0.73 | 0.77 | 1.02 (2.58 s) | 1.80-2.52 (0.72) | 2.04-3.04 (1.00) | 100%-100% | 0% | 0.08 / 0.23 |
| all | 645 | 0.56 | 0.65 | 1.02 (2.58 s) | 0.99-1.48 (0.49) | 2.04-3.04 (1.00) | 29%-54% | 26% | 0.13 / 0.34 |

qc_s2 stopped at epoch 8 (val 0.363 vs 0.300 / 0.305 for s1 / s3): real run-to-run variance of the recipe, kept in every spread figure; the s1-vs-s3 column is supplementary.

## Pairwise differences (a - b, BW; primary inputs)

| a | b | window | mean diff | mean abs diff | max abs diff | r | outside seed band |
|---|---|---|---|---|---|---|---|
| qc_origin | qc_s1 | S1 | +0.33 | 0.36 | 0.66 | 0.75 |  |
| qc_origin | qc_s1 | S2 | +0.26 | 0.28 | 0.65 | 0.66 |  |
| qc_origin | qc_s1 | S1+S2 | +0.30 | 0.32 | 0.66 | 0.73 |  |
| qc_origin | qc_s1 | all | +0.30 | 0.31 | 0.76 | 0.94 |  |
| fpopt_legacy | qc_s1 | S1 | -2.13 | 2.13 | 2.63 | 0.83 |  |
| fpopt_legacy | qc_s1 | S2 | -2.16 | 2.16 | 2.57 | 0.81 |  |
| fpopt_legacy | qc_s1 | S1+S2 | -2.15 | 2.15 | 2.63 | 0.81 |  |
| fpopt_legacy | qc_s1 | all | -1.23 | 1.23 | 2.63 | 0.56 |  |
| sept_legacy | qc_origin | S1 | -2.55 | 2.55 | 2.77 | 0.55 |  |
| sept_legacy | qc_origin | S2 | -2.45 | 2.45 | 2.64 | 0.27 |  |
| sept_legacy | qc_origin | S1+S2 | -2.51 | 2.51 | 2.77 | 0.50 |  |
| sept_legacy | qc_origin | all | -1.58 | 1.58 | 2.77 | 0.32 |  |
| sept_legacy | fpopt_legacy | S1 | -0.09 | 0.09 | 0.16 | 0.99 |  |
| sept_legacy | fpopt_legacy | S2 | -0.03 | 0.05 | 0.14 | 0.80 |  |
| sept_legacy | fpopt_legacy | S1+S2 | -0.06 | 0.07 | 0.16 | 0.95 |  |
| sept_legacy | fpopt_legacy | all | -0.04 | 0.04 | 0.27 | 0.94 |  |
| qc_origin | seed_mean | S1+S2 | +0.50 | 0.50 | 0.80 | 0.68 | 77% |
| fpopt_legacy | seed_mean | S1+S2 | -1.95 | 1.95 | 2.29 | 0.83 | 100% |
| sept_legacy | seed_mean | S1+S2 | -2.01 | 2.01 | 2.33 | 0.78 | 100% |

## Converted vs copied pelvis angles (same exported positions)

| model | S1 mean copied -> converted | S2 mean copied -> converted | S1 / S2 peak converted | S1+S2 >1.4 copied -> converted | converted - copied, S1+S2 mean (max abs) |
|---|---|---|---|---|---|
| qc_s1 | 2.44 -> 2.40 | 2.57 -> 2.54 | 3.04 / 3.01 | 100% -> 100% | -0.04 (0.10) |
| qc_s2 | 1.80 -> 1.80 | 1.81 -> 1.81 | 2.04 / 2.02 | 100% -> 100% | -0.01 (0.02) |
| qc_s3 | 2.52 -> 2.45 | 2.65 -> 2.61 | 3.02 / 3.00 | 100% -> 100% | -0.05 (0.11) |
| qc_origin | 2.77 -> 2.73 | 2.82 -> 2.79 | 2.96 / 2.95 | 100% -> 100% | -0.04 (0.08) |
| fpopt_legacy | 0.13 -> 0.26 | 0.29 -> 0.37 | 0.42 / 0.48 | 0% -> 0% | +0.11 (0.33) |
| sept_legacy | -0.02 -> 0.17 | 0.16 -> 0.35 | 0.37 / 0.48 | 0% -> 0% | +0.19 (0.54) |

Seed spread, export_copied: frame range mean 0.77, max 1.05; stance-mean range S1 0.71, S2 0.83; export: frame range mean 0.73, max 1.02; stance-mean range S1 0.65, S2 0.80.

## Input variants (S1+S2 mean / peak / share >1.4)

| model | export (primary) | export_copied | own foot point | scaled-model FK |
|---|---|---|---|---|
| qc_s1 | 2.46 / 3.04 / 100% | 2.50 / 3.08 / 100% | 2.46 / 3.04 / 100% | 2.44 / 3.05 / 100% |
| qc_s2 | 1.80 / 2.04 / 100% | 1.81 / 2.05 / 100% | 1.80 / 2.04 / 100% | 1.81 / 2.04 / 100% |
| qc_s3 | 2.52 / 3.02 / 100% | 2.57 / 3.07 / 100% | 2.52 / 3.02 / 100% | 2.49 / 3.01 / 100% |
| qc_origin | 2.76 / 2.96 / 100% | 2.79 / 3.00 / 100% | 2.70 / 3.03 / 100% | 2.69 / 3.05 / 100% |
| fpopt_legacy | 0.31 / 0.48 / 0% | 0.20 / 0.46 / 0% | 0.31 / 0.48 / 0% | 0.32 / 0.49 / 0% |
| sept_legacy | 0.25 / 0.48 / 0% | 0.06 / 0.44 / 0% | 0.33 / 0.62 / 0% | 0.35 / 0.65 / 0% |
| seeds | seed range mean 0.73, max 1.02 | seed range mean 0.77, max 1.05 | seed range mean 0.73, max 1.02 | seed range mean 0.69, max 1.03 |

`export_copied`: pelvis angles as written (the repo converters' copy convention: label_trajectory_csv.py, infer_trajectory_csv.py, PN3 ::q_urdf); not checked against the optimizer's own code. `own`: only the calcn-origin checkpoints change (exported point moved 13 cm back along the generic URDF's calcn axes; the scaled model's axes give the same point to 0.2 cm). `scaled`: every foot point from the borrowed scaled OpenSim model's forward kinematics with the converted angles (about 2.5 cm from the exported point, foot_geometry.csv); sensitivity only.

Legacy readings in the pipeline sign (+mx), S1 / S2 means: with the copied angles (the convention of the earlier facts) sept_legacy 1.46 / 1.04, peaks 2.42 / 2.16 (facts SEP3: 1.46 / 1.04, peaks 2.42 / 2.16); fpopt_legacy 1.18 / 0.82, peaks 2.07 / 1.80 (SEP4: 1.27 / 0.94, peaks 1.99 / 1.79). With the converted angles: sept_legacy 1.04 / 0.60, peaks 2.04 / 1.17; fpopt_legacy 0.79 / 0.54, peaks 1.66 / 1.35.

## Labels on this motion: context only, NOT ground truth

| labels | sign | S1 mean | S1 peak | S2 mean | S2 peak | S1+S2 >1.4 | SO+JR keep_r frames |
|---|---|---|---|---|---|---|---|
| stored_no_muscle | physical | 0.35 | 0.81 | 0.77 | 3.08 | 4% | 0 |
| stored_no_muscle | pipeline | 1.76 | 2.15 | 1.17 | 1.77 | 72% | 0 |
| sojr_unusable | physical | 2.17 | 3.25 | 2.62 | 6.13 | 98% | 0 |
| sojr_unusable | pipeline | 1.35 | 2.96 | 0.78 | 1.66 | 37% | 0 |

stored_no_muscle = jcf_output/BatchJCF_JointReaction_ReactionLoads.sto (JointReaction without muscle forces; labels.csv f_medial_bw is its pipeline-sign column). sojr_unusable = jcf_output_sojr/FullDuration_... (SO+JR; label_mask.npz keeps none of its right-knee frames). Both were computed by OpenSim on this folder's ik_results.mot, i.e. with the copied pelvis angles, where the scaled model's foot point sits 11.0 cm (mean over right single stance) from the applied COP (next section).

## Pelvis angles and foot geometry

pipeline/label_trajectory_csv.py (line 81, `ik[:, 0:3] = q[:, 3:6]`) copies the optimizer's URDF pelvis angles into OpenSim's pelvis_tilt / list / rotation. assets/Rajagopal2015.urdf rotates its pelvis about the URDF's own z (up), x and y axes (joints pelvis_tilt / pelvis_list / pelvis_rotation; pn3_redis/model.py docstring), so the copied 'pelvis_tilt' is a rotation about the vertical. Converted (R = C^T Rz(q3) Rx(q4) Ry(q5) C, OpenSim Z-X-Y angles; to first order tilt = -q5, list = q4, rotation = q3), the angles differ from the written ones by up to 12.9 deg (tilt), 0.7 deg (list) and 9.2 deg (rotation) in S1+S2 (12.9 / 0.7 / 11.3 deg over the whole motion; pelvis_conversion.csv).

| pelvis angles | body model | right foot, right single stance: fpopt point to exported point, mean / max (cm) | left foot, left single stance (cm) | fpopt point x range in S1 / S2 (cm) | COP in calcn frame, right single stance: x (m) |
|---|---|---|---|---|---|
| as_written | generic URDF (1.70 m) | 12.76 / 20.61 | 6.60 / 16.84 | 11.63 / 14.93 | -0.074 to 0.103 |
| as_written | borrowed scaled model | 11.01 / 18.80 | 6.33 / 15.06 | 11.73 / 15.15 | -0.054 to 0.123 |
| converted | generic URDF (1.70 m) | 0.28 / 0.80 | 0.38 / 1.57 | 1.67 / 2.59 | 0.131 to 0.134 |
| converted | borrowed scaled model | 2.47 / 2.76 | 2.48 / 3.16 | 1.62 / 2.63 | 0.151 to 0.153 |

The exported point itself moves 1.61 / 2.51 cm in x over S1 / S2. COP = exported point to 6.0e-08 m (grf_data.mot). Generic-URDF COM vs the exported COM: 1.24 cm mean (1.91 max) as written, 0.21 cm (0.49) converted.

Reading. With the converted angles the optimizer's own body model reproduces its export to millimetres, so the conversion is right and the copied angles are not OpenSim angles. The optimizer's feet are planted, and the COP is pinned at its mid-foot contact point: in the URDF calcn frame it stays at x = 0.131 to 0.134 m, y = -0.011 to -0.009 m over every right single stance (left: x = 0.131 to 0.132 m), i.e. on the fpopt point (0.13, -0.01, 0), 13 cm ahead of the calcn origin, with no heel-to-toe progression. For comparison, the right-foot COP of moving-COP Moore training trials runs over x (p5-p95, own scaled calcn frame; cop_reference.csv): moore_subject15_t10_r00 0.104 to 0.162 m; moore_subject5_t35_r01 0.099 to 0.207 m; moore_subject15_t53_r00 0.105 to 0.218 m. So the COP is not pinned to the heel, and the foot does not slide under it: the 12-15 cm slide reported in the previous version of this file came from the copied pelvis angles. The borrowed scaled model keeps a near-constant offset of about 2.5 cm after the conversion (segment scaling; COP at x = 0.15 m in its calcn frame).

Consequences. (1) This folder's OpenSim kinematics (ik_results.mot) misplace the pelvis orientation, which puts the scaled model's foot point 11.0 cm (right single stance mean; up to 18.8 cm) away from the COP that SO and JR apply, against 2.5 cm with the converted angles; both label sets above were computed that way, on top of the pinned COP (which alone makes label_mask's fixed-COP rule drop the trial). Whether the mis-mapping causes the 27.5 BW SO+JR peak is not tested here (no OpenSim run). (2) Re-labelling optimizer trajectories with SO+JR needs the pelvis conversion in pipeline/label_trajectory_csv.py first. (3) infer_trajectory_csv.py copies the angles the same way, so a model fed an optimizer export through it sees a pelvis up to ~13 deg off (the PN3 driver feeds its surrogate OpenSim-correct ::q; only its ::q_urdf output key follows the copy convention).

Cross-check: inference_posture_surrogate.predict_windowed on the same q, positions and contact flags reproduces every model's F_medial to 1.2e-06 BW with the converted angles and 1.1e-06 BW with the copied ones (run_info.json).

Forward kinematics check (`analysis/thesis/osim_fk.py` vs OpenSim caches, 13 training trials from 6 labs, 8343 fpopt caches available): max error 6.0e-08 m (fk_validation.csv).

# Simulated suit (PN3 driver) with the corrected baseline

Script: `analysis/thesis/pn3_sim.py` (120 driver runs, 50 s on 20 workers; `--replot` rebuilds the tables and figures).
Tables last rebuilt with `--replot` after adding `suit_added_kept_stance_mae` and `seed_detail.csv`. The driver runs
were not repeated, and every earlier column and both figures are unchanged.
Date: 2026-11-10. Physical M_x sign throughout (medial moment = -mx of walker_knee_r, for labels and for every raw
output, legacy checkpoints included). Labels: SO+JR (`jcf_output_sojr`). Nothing was trained, and no OpenSim tool was
run. `imu/synthetic.py` (OpenSim forward kinematics only, `jcf` env) made the takes.

## What was run

Each run calls the deployed command lines:

    python -m pn3_redis.driver --source sto <take>/orientations.sto --resample-hz 60 --contact gyro \
        --backdate-s {0.2 | 0} --height H --mass M --surrogate <ckpt> --log log.csv --quiet
    python -m pn3_redis.evaluate --log log.csv --synthetic <take> --trial <trial> --out <dir> \
        --label-subdir jcf_output_sojr --mx-convention physical

There is also one force-plate-contact run (`--contact source --contact-trial <trial>`, config `fp`) that separates
contact error from kinematic error. Per-frame logs: `/tmp/claude-1000/-home-gellert-Developer-medical-dt/187d5d35-6c3c-4c5e-8d5a-f69463aa63f9/scratchpad/pn3sim/work/runs/` (not kept).

| Trial | Folder | H / M entered | Scored span |
|---|---|---|---|
| Carter P010 t10 (held-out) | `jcf/full_duration/testing/walking/carter_P010_split0_t10` | 1.70 m / 54 kg (true H 1.656 m; config `gyro_bd0.2_trueH` uses it) | 7.27 s, 437 frames, 257 right stance |
| Moore subject10 t55_r02 (held-out) | `jcf/full_duration/testing/walking/moore_subject10_t55_r02` | 1.77 m / 90.7 kg | 12.77 s, 767 frames, 477 right stance |

Takes are made with `python -m imu.synthetic --trial <trial> --out <dir> --fs 100 [--noise-deg 1|2] [--drift-deg-per-s 0.5]`,
seed 0, AR(1) noise with 0.1 s correlation, and a 1 s default-pose calibration. There are four per trial: clean, 1°,
2°, and 1° + 0.5°/s heading drift. They are in `takes/`, except that the P010 1° take is
`results/imu/driver_demo/take_P010_t10_noise1deg`. Regenerating that take with this command reproduces it
byte for byte. The extra takes named in the task did not exist on disk, so they were made here.

Checkpoints: qc_s1/s2/s3 (optimizer foot point, seeds 1-3), qc_origin, fpopt_legacy and sept_legacy (legacy ones only
on the 1° takes).

The live F_medial (U) is compared, on the frames it was published for, with three references:
- **L, lab path.** The same checkpoint on its own training features: `tools/evaluate_fmedial.features_for --deploy-features`,
  the scaled OpenSim model, force-plate contact and the trial's true height.
- **O, optical through the driver's generic body.** The trial's IK through `LowerBody` at the entered height, the
  checkpoint foot point and force-plate contact, i.e. the Twente O path. U − O is what the suit, the UKF and the
  online contact add; O − L is what the generic body model adds.
- **SO+JR labels** on QC-kept frames (`label_mask.npz` keep_r, `LABEL_MASK_TOL` 0.05).

Stance means force-plate right stance. Stance peaks are taken over intervals of at least 0.2 s inside the scored span.
Against labels, the maximum is taken over the kept frames of stances with at least 50 % of their frames kept.

## Label coverage (`label_coverage.csv`)

| | JR rows kept | right-stance rows kept | scored frames kept | why masked |
|---|---|---|---|---|
| Carter P010 t10 | **0 / 1850 (0 %)** | 0 / 1107 | 0 / 437 | fixed COP: right COP moves 0.030 m/s (< 0.2 m/s threshold), so every row is masked. SO flags are also on 71.6 % of rows and F_y > 5 BW on 56.3 % of stance rows |
| Moore s10 t55_r02 | 753 / 1292 (58.3 %) | 528 / 816 (64.7 %) | 361 / 767 (47.1 %); stance 261 / 477 (54.7 %) | the first 15-24 % of **every** right stance (loading response) is masked (SO flags 35.3 % of stance rows) |

**Carter P010 is unusable as a label reference.** Its unmasked SO+JR labels average 2.98 BW in stance, against 1.85 BW
from the lab path, and the unmasked suit-vs-label stance MAE is 1.40-1.50 BW for qc_s1 in every take and config. Those numbers
measure the label artefact, not the suit. Moore t55_r02 is therefore the second take. Of the long Moore test trials
(≥ 8 s), it has the most kept stance rows among those with ≥ 60 % stance coverage.

Because the loading response is masked in every Moore stance, label "peaks" are late-stance maxima. Seven of 10 stances
qualify.

## Results, seed 1 (qc_s1) with seeds 1-3 as mean [min, max] (`headline.csv`, `runs.csv`, `seed_spread.csv`)

### Joint angles and pelvis (label-free, against the IK truth, gyro contact)

| take | worst of 13 surrogate angles, RMSE (°) P010 / Moore | pelvis height RMSE (cm) | pelvis side-to-side RMSE (cm) | forward speed vs belt (m/s) |
|---|---|---|---|---|
| clean | 0.93 / 1.52 | 0.87 / 1.33 | 2.08 / 5.56 | 1.34 vs 1.30 / 1.15 vs 1.07 |
| 1° | 2.06 (knee R) / 1.80 (hip flex L) | 1.12 / 1.56 | 2.10 / 5.36 | 1.33 vs 1.30 / 1.14 vs 1.07 |
| 2° | 3.63 / 3.54 | 2.13 / 3.78 | 3.15 / 11.04 | 1.34 / 1.11 |
| 1° + 0.5°/s drift | 3.11 (hip rot L) / 5.72 (hip rot L) | 1.69 / 1.59 | 7.07 / 13.83 | 1.33 / 1.14 |

Angles do not depend on the checkpoint. The 2.06° P010 number reproduces the paper's 2.1° (A3-44). Pelvis tilt was
never the worst angle: 1.00° (P010 1°) and 1.23° (Moore 1°).

### Contact, flags the surrogate saw: agreement with force plates, R / L (`backdating.csv`)

| take | P010 no back-dating → 0.2 s | Moore no back-dating → 0.2 s |
|---|---|---|
| clean | 0.908/0.851 → 0.959/0.939 | 0.895/0.875 → 0.928/**0.842** |
| 1° | 0.887/0.872 → **0.962/0.962** | 0.916/0.871 → 0.942/**0.850** |
| 2° | 0.876/0.860 → 0.955/0.950 | 0.864/0.888 → 0.912/0.895 |
| 1° + drift | 0.910/0.883 → 0.944/0.950 | 0.885/0.867 → 0.937/0.880 |

P010 1° reproduces the paper's 0.88/0.87 → 0.96/0.96; the online flags are 0.880/0.873.

**On Moore, back-dating makes the left foot worse.** The stance detector re-triggers about 0.09 s after left toe-off on
3 of 11 left strides (at 4.92, 7.32 and 9.72 s; toe-off at 4.83, 7.23 and 9.63 s), so the swing reads as stance.
Back-dating then joins that false onset to the preceding stance and dates the real left heel strike a median 0.10 s
early, in terminal swing. Left-swing frames flagged as contact rise from 25.8 % to 36.4 % (1°). With only 7 of 11 left
heel strikes matched within 0.25 s, the right-knee prediction dips in late right stance (`fig_series_qc_s1_noise1deg.png`).

### F_medial (BW), stance MAE

Each cell is seed 1; seeds 1-3 as mean [min, max]. "Suit added" is U vs kept labels minus L vs kept labels on the same
261 kept right-stance frames (`suit_added_kept_stance_mae`; per seed in `seed_detail.csv`). Seed 1 is the worst seed for
most of these numbers, so quote the seed mean and range, not seed 1 alone.

| take | config | P010: U vs L | Moore: U vs L | Moore: U vs kept labels | Moore: L vs kept labels | Moore: suit added |
|---|---|---|---|---|---|---|
| 1° | gyro, no back-dating | 0.243; 0.178 [0.110, 0.243] | 0.232; 0.199 [0.157, 0.232] | 0.433; 0.414 [0.365, 0.443] | 0.316; 0.337 [0.316, 0.363] | +0.117; +0.076 [+0.002, +0.117] |
| 1° | gyro, back-dated 0.2 s | **0.165; 0.123 [0.060, 0.165]** | **0.218; 0.183 [0.149, 0.218]** | **0.434; 0.404 [0.347, 0.434]** | same | **+0.118; +0.067 [−0.017, +0.118]** |
| 1° | force-plate contact | 0.073; 0.055 [0.015, 0.077] | 0.065; 0.050 [0.031, 0.065] | 0.305; 0.331 [0.305, 0.363] | same | −0.011; −0.006 [−0.011, −0.000] |
| clean | back-dated | 0.177; 0.139 [0.068, 0.177] | 0.204; 0.173 [0.150, 0.204] | 0.435; 0.400 [0.342, 0.435] | same | +0.118; +0.063 [−0.021, +0.118] |
| 2° | back-dated | 0.199; 0.139 [0.062, 0.199] | 0.221; 0.190 [0.153, 0.221] | 0.402; 0.398 [0.384, 0.409] | same | +0.086; +0.061 [+0.021, +0.086] |
| 1° + drift | back-dated | 0.203; 0.157 [0.080, 0.203] | 0.201; 0.165 [0.132, 0.201] | 0.411; 0.390 [0.352, 0.411] | same | +0.095; +0.053 [−0.011, +0.095] |

Per seed (1°), suit added, back-dated / no back-dating / force-plate contact: s1 +0.118 / +0.117 / −0.011,
s2 −0.017 / +0.002 / −0.000, s3 +0.101 / +0.111 / −0.007. With back-dating, seed 2 is negative in every take except
2° (−0.021 clean, −0.017 1°, +0.021 2°, −0.011 drift); seed 1 is +0.086 to +0.118.

Why the seeds differ: the suit shifts stance F_medial down for every seed (U − L stance bias, 1° back-dated: −0.119,
−0.117, −0.102). Seeds 1 and 3 already read below the kept labels (L stance bias −0.198, −0.246), so the shift adds
error; seed 2 does not (L bias +0.031), so the same shift costs no MAE. Every seed already under-reads the late-stance
maximum, and the suit makes that error worse for every seed: −0.734 → −0.766, −0.942 → −1.000, −0.748 → −0.803 (L → U).

Other measures (seed 1, 1°, back-dated):
- **Correlation and bias.** U vs L r = 0.972 (P010) and 0.927 (Moore). Stance bias of U vs L is +0.084 on P010
  and −0.119 on Moore.
- **All-frame MAE (U vs L).** 0.163 → 0.115 on P010; 0.156 → 0.148 on Moore.
- **Stance-peak error (U − L).** +0.100 BW on P010 (7 stances) and −0.036 on Moore (10 stances).
- **Moore against kept labels.** Kept-frame r is 0.895 for U and 0.925 for L. Stance bias is −0.374 for U and −0.198
  for L. The late-stance maximum error is −0.766 for U and −0.734 for L, so the model already under-reads the SO+JR
  late peak before the suit adds anything.
- **Decomposition.** U vs O is 0.139 (P010) and 0.203 (Moore). O vs L, the generic body alone, is 0.049 and 0.026
  (seeds 0.040 [0.010, 0.062] and 0.025 [0.021, 0.027]).
- **Entered height.** Entering P010's true 1.656 m instead of 1.70 m changes U vs L from 0.165 to 0.174 (seed 1), so
  the height entry is negligible.

Main findings:
1. **Online contact, not the IMU kinematics, causes most of the suit's F_medial error.** With force-plate contact, the
   same suit kinematics stay within 0.04-0.09 BW of the lab path from clean to 1° + drift, and 0.10 BW at 2° (seed 1;
   all seeds, takes and both trials 0.014-0.107). With the gyro detector, the 1° gap is 0.17-0.24 BW for seed 1 and
   0.06-0.24 over seeds and back-dating settings. With back-dating, the gyro gap is 1.9-4.7 times the force-plate gap
   at 1° for every seed and both trials, and 1.5-6.7 times over all takes.
2. **Back-dating clearly cuts the gap to the lab path only on P010, and does not cut the error against labels.**
   Stance MAE vs the lab path, 1°, no back-dating → 0.2 s: P010 0.243 → 0.165 for seed 1 (seed mean 0.178 → 0.123; s2 0.110 → 0.060,
   s3 0.179 → 0.143); Moore 0.232 → 0.218 (seed mean 0.199 → 0.183). Against the QC-kept labels, which exist only on
   Moore, it changes nothing for seed 1 (0.433 → 0.434; seed mean 0.414 → 0.404). On Moore the left-contact agreement
   gets worse (0.871 → 0.850), because of the false re-onsets described above. P010 carries no usable labels, so its
   gain is a gain toward the mocap-fed prediction, not a measured accuracy gain.
3. **On the only trial with kept labels, the suit adds 0.07 BW on average** to the model's own stance error against
   the labels (1°, back-dated; seeds −0.02 to +0.12; 0.434 vs 0.316 for seed 1, 0.404 vs 0.337 as the seed mean).
   The range spans zero: for seed 2 the suit is slightly better than the lab path (0.347 vs 0.363). With force-plate
   contact the suit adds nothing for any seed (−0.011, −0.000, −0.007).
4. **Seed variance is as large as the noise effect.** qc_s2, the seed stopped at epoch 8, is the least
   input-sensitive: U vs L 0.060 on P010 against 0.165 and 0.143 for s1 and s3. It also has the flattest output
   (Moore lab-path late-maximum error −0.94 BW against −0.73 and −0.75), so lower suit sensitivity here comes with a weaker
   model, not a better pipeline.

### Quoting the back-dating result in the paper

The paper's earlier "cutting the F_medial MAE from 0.39 to 0.23 BW" was an error against labels on one trial. No
number here can replace it one for one: P010 has no kept labels, and on Moore back-dating does not change the label
error. Name the reference, give both trials and the seed range, e.g. "back-dating cut the suit's deviation from the
mocap-fed surrogate from 0.24 to 0.17 BW on one trial (seed mean 0.18 to 0.12) but only from 0.23 to 0.22 on the
other, and left the error against quality-checked labels unchanged (0.43 to 0.43; seed mean 0.41 to 0.40)". The
sentence's "one held-out trial" becomes two.

### Other checkpoints (1°, back-dated, stance MAE; `table_all.csv`)

| ckpt | P010 U vs L | Moore U vs L | Moore U vs kept labels | Moore L vs kept labels |
|---|---|---|---|---|
| qc_origin | 0.160 | 0.187 | 0.397 | 0.317 |
| fpopt_legacy | 0.087 | 0.038 | 1.180 (bias −1.175) | 1.152 |
| sept_legacy | 0.066 | 0.037 | 1.179 | 1.148 |

The legacy models are about 1.15 BW below SO+JR under the physical sign, as the audit predicts. Their small U-vs-L
differences mean little, because their outputs under that sign are low and flat.

## Caveats and tool issues found

- **`pn3_redis.evaluate` leaves pelvis_tilt out of `angle_rmse_deg`.** Its filter `c.startswith("pelvis_t")` was meant
  for the pelvis translations but also matches pelvis_tilt. This script recomputes all 16 rotational coordinates
  (`rmse_deg__*` in runs.csv). The worst-angle numbers are unaffected here.
- **`pn3_redis.evaluate` assumes walking along +x.** The Moore subject faces 93° (pelvis_rotation mean), i.e. along
  ground z, so its `forward_speed_mps` (0.009) and `pelvis_lateral_rmse_m` (4.2 m) are meaningless for Moore. This script
  projects on the stance-heel belt direction instead: 89.0° for Moore and 0.2° for P010, where the values match
  evaluate's. The evaluate figures `driver_eval.png` for Moore show the wrong speed in their title.
- **The synthetic calibration pose has heading 0.** Moore therefore starts with a 93° turn at the first walking frame.
  The UKF settles within the 0.5 s skip (pelvis_rotation RMSE 1.03° clean), but a real calibration would face the
  walking direction.
- **The scope is two trials and one person per lab.** These are sanity checks of the path, not population estimates.

## Files

- `runs.csv`: one row per run (120), every metric.
- `headline.csv`: seed 1 and seeds 1-3 per trial × take × config.
- `seed_spread.csv`: mean, min and max over seeds, all metrics.
- `seed_detail.csv`: per-seed values (s1, s2, s3, mean, min, max) of the F_medial stance MAEs quoted above,
  including `suit_added_kept_stance_mae`.
- `backdating.csv`: no back-dating vs 0.2 s vs force-plate contact.
- `table_all.csv`: all checkpoints.
- `label_coverage.csv`: label-mask coverage of both trials.
- `run_info.json`: settings.
- `fig_series_qc_s1_noise1deg.png`: F_medial over the first 4 s of both trials.
- `fig_noise_sweep.png`: worst angle and stance MAE vs take, seeds 1-3.
- `takes/`: the generated takes.

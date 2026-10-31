# IMU pipeline: portable capture → knee-load surrogate

Goal: estimate medial knee load (F_medial) from a wearable IMU suit (Perception Neuron 3) instead of a
mocap lab with force plates, using the GRF-free MLP surrogates. Code in `imu/`, tests in `tests/test_imu.py`.

```
Axis Studio BVH ─► from_bvh.py ─► orientations.sto ─► opensense_ik.py ─► joint angles (.mot)
                                         │               (IMUPlacer + IMU IK)        │
                                         │                                           ├─► FK: COM, feet rel. pelvis
                                         └─► contact.detect_contact_orientation ─────┤
                                              (foot sensors, no force plates)        └─► JCFPostureSurrogate ─► F_medial
```

## Two constraints that decide the design

**Orientation sensors do not observe pelvis translation.** OpenSense IK solves joint angles that match
segment orientations; `pelvis_tx/ty/tz` stay at the calibration pose. Everything the surrogate needs
besides that is translation-invariant: joint angles, and COM and feet *relative to the pelvis* from forward
kinematics. So the IMU path must use a no-pelvis-translation checkpoint (`*_notrans_*`). Checkpoints with
pelvis-translation inputs are scored on the mocap path only.

**No force plates, so no contact flags from GRF.** The contact-flag surrogates were trained with flags
from vertical GRF > 5 % BW. On an IMU suit the flags come from the foot sensor's orientation
(`contact.detect_contact_orientation`): mid-stance is quiet (foot angular speed ~20–80 °/s); heel strike
(~300 °/s) and push-off (~500 °/s) are fast but inside force-plate stance, so each quiet run is extended
back and forward to the speed peak within 0.12 s. The world-frame heel/toe detector
(`contact.detect_contact`) needs foot positions in the world frame, which the suit does not give, and it fails on treadmill
trials, where the stance foot moves with the belt.

## Modules

| Module | Does | Env |
|---|---|---|
| `bvh.py` | BVH parser and forward kinematics (intrinsic rotations in channel order) | numpy |
| `segments.py` | suit segment → OpenSense label (`pelvis_imu`, `femur_r_imu`, …), `LOWER_BODY` set | numpy |
| `opensense_io.py` | read/write OpenSense quaternion `.sto` (w,x,y,z) | numpy |
| `from_bvh.py` | CLI: BVH → orientations, starting at the calibration frame | numpy |
| `synthetic.py` | virtual IMUs from a mocap trial (body orientations via OpenSim), with noise and heading drift | jcf |
| `opensense_ik.py` | IMUPlacer calibration + IMU inverse kinematics | jcf |
| `contact.py` | contact from foot positions or from foot-IMU orientation; scoring against force-plate flags | numpy |
| `evaluate_contact.py` | tune/score the orientation detector over many trials | jcf |
| `validate_synthetic.py` | end-to-end: trial → virtual IMUs → OpenSense → angles, contact, F_medial vs mocap and OpenSim | jcf |

`jcf` = run with `conda run -n jcf python -m imu.<module>` from the repo root (OpenSim 4.5 + torch).

## Validation on mocap trials (virtual IMUs)

`validate_synthetic.py` replays a recorded trial as if it had been captured with the suit and scores the
surrogate three ways against OpenSim's SO+JR labels:

| path | inputs | isolates |
|---|---|---|
| `mocap+grf` | mocap IK angles, force-plate contact | the surrogate itself (reference) |
| `imu+grf` | OpenSense IK angles, force-plate contact | IMU kinematics |
| `imu+imu_contact` | OpenSense IK angles, orientation-based contact | the fully portable pipeline |

Noise model: `--noise-deg` is RMS rotation noise per axis, band-limited (AR(1), `--noise-corr-s`,
default 0.1 s), because sensor-fusion output errs slowly; white per-frame noise (`--noise-corr-s 0`)
puts hundreds of °/s of fake angular speed into the contact detector. `--drift-deg-per-s` is a
per-sensor heading drift about vertical, growing linearly with time.

First results, **one test-split treadmill trial** (Carter P010 t10, 8 s, 100 Hz), F_medial MAE vs
OpenSim (BW), whole trial:

| sensor error | worst joint-angle RMSE | contact agreement R / L | walking model: mocap+grf · imu+grf · imu+imu_contact | all-activity model: same three |
|---|---|---|---|---|
| none | 0.05° | 0.96 / 0.98 | 0.207 · 0.212 · 0.245 | 0.297 · 0.296 · 0.327 |
| 1° noise | 2.1° (ankle) | 0.96 / 0.97 | 0.207 · 0.236 · 0.270 | 0.297 · 0.302 · 0.348 |
| 1° noise, 0.5 °/s drift | 3.3° (hip rotation) | 0.97 / 0.98 | 0.207 · 0.228 · 0.261 | 0.297 · 0.316 · 0.348 |
| 2° noise | 4.3° (ankle) | 0.81 / 0.84 | 0.207 · 0.267 · 0.378 | 0.297 · 0.322 · 0.516 |
| 1° noise, 2 °/s drift | 13.2° (hip rotation) | 0.97 / 0.98 | 0.207 · 0.302 · 0.332 | 0.297 · 0.344 · 0.370 |

Walking model = `best_model_mlp_walking_qinst_notrans_pelvisrel_winpos_contact_w20_fmW0p5_e100_2`, all-activity
model = `best_model_mlp_both_qinst_notrans_pelvisrel_winpos_contact_w20_fmW0p5_e100`. Contact agreement is
the fraction of frames where the orientation detector matches force-plate contact.

Read:
- With clean sensors OpenSense recovers the 13 surrogate joint angles to ≤0.05°; the pipeline itself
  (calibration, IK, FK features without pelvis translation) costs almost nothing.
- At a plausible sensor error (1° noise, 0.5 °/s drift) the portable path adds ~0.05 BW over mocap with
  force plates (walking model 0.207 → 0.261): ~0.02 from kinematics, ~0.03 from contact flags.
- Contact detection degrades quickly once orientation noise reaches ~2°, because the detector
  differentiates orientation. Real suits also give raw gyro rates, which would avoid that.
- Heading drift mostly hits hip rotation (relative heading between pelvis and femur); 2 °/s over 8 s
  is the pessimistic end for a modern suit.

These are one trial; the thesis numbers should come from the test split (commands below).

## Commands

Validation on one trial, clean and with sensor error:

```bash
conda run -n jcf python -m imu.validate_synthetic \
    --trial jcf/full_duration/testing/walking/carter_P010_split0_t10 \
    --out results/imu/val/P010_t10_n1_d0.5 --fs 100 --noise-deg 1 --drift-deg-per-s 0.5
```

Writes `summary.json` (angle RMSE per coordinate, contact agreement and edge errors, F_medial errors per
checkpoint and path, whole trial and right stance), `angles_rmse.csv`, `fmedial.csv`, `fmedial.png`.
Default checkpoints are the two `*_notrans_*_contact_*` runs in `models/runs/mlp/`; pass others with
`--ckpt`.

Tune the contact detector on training trials (one per person, up to 10 people per lab), then score the
chosen setting once on the test split:

```bash
conda run -n jcf python -m imu.evaluate_contact --root jcf/full_duration/training/walking \
    --per-lab 10 --out results/imu/contact_tuning
conda run -n jcf python -m imu.evaluate_contact --root jcf/full_duration/testing/walking \
    --per-lab 0 --thresh 150 --extend 0.12 --out results/imu/contact_test
```

Real suit take (once recorded): stand still in the neutral pose for a few seconds, then walk.

```bash
python -m imu.from_bvh --bvh take01.bvh --out results/imu/take01/orientations.sto --start 1.0
conda run -n jcf python -m imu.opensense_ik --model <subject scaled_model.osim> \
    --orientations results/imu/take01/orientations.sto --out results/imu/take01 --t0 2.0 --heading-axis z
```

## Open items

- **Real-suit conventions are untested.** The Axis Studio export settings (Euler order, units, whether
  the export includes a rest pose), the base sensor's forward axis (`--heading-axis`, likely `z` for a
  BVH character facing +z) and the segment names must be checked on the first recorded take.
- **Scaled model for a suit subject.** Without markers, the model is scaled from height and mass or from
  the suit's segment lengths (BVH OFFSETs); not implemented yet.
- **Batch validation over the test split** (a loop over `validate_synthetic.py`) and a sensitivity sweep
  over noise and drift for the thesis.
- **Live use.** OpenSai has no IMU-suit driver; a live demo needs a separate bridge (Axis Studio stream →
  Redis keys). Offline analysis does not need it.

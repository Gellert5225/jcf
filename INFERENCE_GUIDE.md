# Inference Guide — `n_d05_m_s` Static-Aware Knee JCF Model

This document describes how to integrate and run the trained knee JCF surrogate model. The model is trained jointly on walking and quasi-static postures from the AddBiomechanics database (Carter, Moore, Lencioni/Tiziana, Falisse subsets), and the same predictor handles both regimes. All inference logic lives in a single self-contained module: **`inference.py`**.

## Files needed for integration

| File | Purpose |
|---|---|
| `inference.py` | Self-contained inference module (model class, preprocessing, `JCFPredictor`) |
| `best_model_n_d05_m_s2_s2.pt` | Trained model checkpoint (~3 MB) |

That's it. Drop both files into your project directory. The only Python dependencies are `numpy` and `torch`.

## What you need at runtime

| Input | Source | Format |
|---|---|---|
| **Joint angles** | OpenSim IK / motion capture | `[T, 16]` array of lower-body DOFs in radians |
| **Ground reaction forces** | Force plates / IMU-based estimate | `[T, 6]` array in Newtons |
| **Subject mass** | Subject record | Scalar in kg |
| **Subject height** | Subject record | Scalar in meters (only for Mx denormalization and the medial compartment estimate) |

You **do not** need to provide velocities, accelerations, or the gait-speed feature separately. The preprocessing in `inference.py` derives them from the raw inputs.

The model expects **100 Hz sampling by default**, but supports any rate if you pass the correct `dt` to `predict()`. The training data included Carter at 250 Hz and Moore at 100 Hz; derivatives are computed using the actual `dt`, so the model is rate-agnostic as long as the right `dt` is supplied.

**Critical:** if your data is not at 100 Hz, pass `dt=actual_dt_in_seconds` to `predict()`. Wrong `dt` scales all derivative channels (joint velocities, accelerations, GRF derivatives) and produces severely degraded predictions.

## The 16 lower-body joint angles (in order)

| Index | DOF | Notes |
|---|---|---|
| 0 | pelvis_tilt | radians |
| 1 | pelvis_list | radians |
| 2 | pelvis_rotation | radians |
| 3 | pelvis_tx | meters (zero-centered automatically by `preprocess` unless you pass `zero_center_pelvis=False`) |
| 4 | pelvis_ty | meters (zero-centered) |
| 5 | pelvis_tz | meters (zero-centered) |
| 6 | hip_flexion_r | radians |
| 7 | hip_adduction_r | radians |
| 8 | hip_rotation_r | radians |
| 9 | knee_angle_r | radians |
| 10 | ankle_angle_r | radians |
| 11 | hip_flexion_l | radians |
| 12 | hip_adduction_l | radians |
| 13 | hip_rotation_l | radians |
| 14 | knee_angle_l | radians |
| 15 | ankle_angle_l | radians |

**Dead joints excluded:** `subtalar_r`, `mtp_r`, `subtalar_l`, `mtp_l`. If your IK output includes them, drop them before constructing the input.

The constant `JOINT_ORDER` is exported from `inference.py` for reference.

## GRF channel order

| Index | Channel |
|---|---|
| 0 | `calcn_r_force_vx` (anterior-posterior, right) |
| 1 | `calcn_r_force_vy` (vertical, right) |
| 2 | `calcn_r_force_vz` (medio-lateral, right) |
| 3 | `calcn_l_force_vx` |
| 4 | `calcn_l_force_vy` |
| 5 | `calcn_l_force_vz` |

GRF should be in **Newtons**. The preprocessor normalizes by body weight internally.

## Quick start

```python
from inference import JCFPredictor

predictor = JCFPredictor("best_model_n_d05_m_s2_s2.pt")

out = predictor.predict(
    joint_angles,    # [T, 16] radians
    grf,             # [T, 6] Newtons
    mass_kg=70.0,
    height_m=1.75,   # optional, enables Mx in N·m and f_medial output
    dt=0.01,         # sampling period in seconds. Default 0.01 (100 Hz). Use 0.004 for 250 Hz.
)

# out["fx"], out["fy"], out["fz"]   # forces in BW
# out["mx"]                         # adduction moment in BW × Height (dimensionless)
# out["fy_n"]                       # axial force in Newtons
# out["mx_nm"]                      # adduction moment in N·m   (requires height_m)
# out["f_medial"]                   # medial compartment force in BW (requires height_m)
```

That's the entire API. The predictor handles preprocessing, normalization, forward pass, and denormalization.

## What `predict()` returns

A dict with these keys:

| Key | Shape | Units | Notes |
|---|---|---|---|
| `fx` | `[T]` | BW | Knee anterior-posterior force in tibia frame |
| `fy` | `[T]` | BW | Knee axial compressive force (negative = compression) |
| `fz` | `[T]` | BW | Knee medio-lateral force |
| `mx` | `[T]` | dimensionless (BW × H) | Knee adduction moment |
| `fy_n` | `[T]` | Newtons | Axial force × BW |
| `mx_nm` | `[T]` | N·m | Adduction moment × BW × Height (only if `height_m` was passed) |
| `f_medial` | `[T]` | BW | Medial compartment force estimate (only if `height_m` was passed) |

## Sign convention (important)

The model's `mx` output is the **external knee adduction moment** ($M_{add}$), signed so that **`mx > 0` corresponds to medial-compartment loading during right-leg stance** — matching the convention used throughout the clinical biomechanics literature (Andriacchi 2004, Zhao 2007).

Concretely: the OpenSim `JointReaction` analysis configured with `apply_on_bodies=child` and `express_in_frame=child` for the Rajagopal `walker_knee_r` joint already returns the moment in this convention. The model is trained on this label directly, with no sign flip. Downstream consumers (e.g., the OpenSAI controller, Schipplein–Andriacchi medial proxy) can therefore plug `out["mx"]` straight into their formulas without any negation.

For a healthy walking trial, expect `out["mx"]` to be predominantly **positive** during the stance phase of the right leg (~0.005–0.04 BW·H typically; peaks ~0.05 BW·H).

## Static posture inference

The same predictor handles single held postures via `predict_static()`. Internally, the static feature vector is replicated across 240 frames (the bidirectional receptive field) and the prediction at the central frame is returned:

```python
static_pose = np.array([...])   # shape (16,) — single posture in JOINT_ORDER
static_grf  = np.array([...])   # shape (6,)  — single GRF vector in Newtons

out = predictor.predict_static(
    joint_angles_static=static_pose,
    grf_static=static_grf,
    mass_kg=70.0,
    height_m=1.75,
)
# out["fy"], out["mx"], out["f_medial"] are SCALARS (not arrays)
```

This is the inference pattern for the multi-objective posture-optimization use case described in the paper. The replicated-input strategy matches the training-time data distribution: the model has seen near-constant input windows from the static training trials and learned to map them to the corresponding static JCF.

**Performance note:** within-laboratory cross-subject static prediction is excellent (~0.07 BW MAE for $F_y$). Cross-laboratory static prediction is more uncertain (~0.19 BW MAE on the held-out Tiziana subject). For high-stakes single-subject deployments, plan for a 5–10-frame calibration step.

## Medial compartment estimate (Eq. 5 from paper)

`predict()` computes this automatically when `height_m` is provided:

```
F_medial / BW = α × |Fy| + (Mx × Height_m) / d
```

with defaults α = 0.5 and d = 0.045 m. To override:

```python
out = predictor.predict(joint_angles, grf, mass_kg=70.0, height_m=1.75,
                        alpha=0.6, d_m=0.040)
```

For varus-aligned / OA knees, α can be calibrated to a higher value (0.6-0.7) per subject if radiographic alignment data is available.

## Important notes

1. **Input normalization is automatic.** `JCFPredictor` applies the per-feature mean/std stored in the checkpoint. You don't need to handle this yourself.

2. **Pelvis translations are auto-zero-centered** by `preprocess`. If your input is already zero-centered (e.g., relative to a reference frame), you can pass `zero_center_pelvis=False` to `preprocess` directly.

3. **Minimum sequence length: 100 frames** (or use `predict_static()` for single postures). The bidirectional CNN has a receptive field of ~120 frames each side. Shorter sequences produce edge artifacts. For real-time streaming, maintain a rolling buffer of at least 100 frames before requesting predictions; for single static queries, `predict_static()` handles the receptive-field padding internally.

4. **Sampling rate must match `dt`.** Default is 100 Hz (`dt=0.01`). For 250 Hz data, pass `dt=0.004`; for 1000 Hz, downsample to 100 Hz first. Wrong `dt` produces visibly attenuated predictions because the joint velocities/accelerations get the wrong scale.

5. **Mass is per-subject, not per-frame.** Pass it as a scalar; the predictor broadcasts it as a constant feature internally.

6. **Height is only used at the output stage.** The model itself does not consume height as an input.

## Validation against the training-test pipeline

`test_inference.py` (in this repo) compares `inference.py`'s output against the official `test_cnn.py` pipeline on real held-out trials from Carter, Moore, and Tiziana. Across 7 representative trials (4 walking, 3 static), the two pipelines agree on per-frame predictions to within **~0.0001 BW (floating-point noise)**.

This exact parity is achieved because the model is trained with `native_time_grid=True` in `load_subject` — IK and GRF are kept at their native sample times rather than being interpolated onto the JCF timestamp grid. `inference.py` consumes IK and GRF identically, so the network sees inputs in exactly the same statistical regime at training and inference.

To run the verification yourself:
```bash
python test_inference.py
```

## Performance

| Hardware | Latency for 2000-frame sequence | Throughput |
|---|---|---|
| RTX 3090 (GPU) | 0.97 ms | ~2M frames/sec |
| CPU | ~37 ms | ~54k frames/sec |

Model size: ~720K parameters, ~2.9 MB checkpoint. The `JCFPredictor` automatically uses CUDA if available; pass `device="cpu"` to force CPU.

## Smoke test

To verify the integration works on synthetic data:

```bash
python inference.py best_model_n_d05_m_s2_s2.pt
```

This loads the checkpoint, runs inference on 5 seconds of fake data, and prints output ranges. If it completes without errors, the integration is correct.

## Lower-level API (if needed)

For more control, the following are exported from `inference.py`:

| Symbol | Purpose |
|---|---|
| `JCF_CNN_v2` | The model class |
| `_ResBlock1d` | The residual block (internal, but exported for completeness) |
| `preprocess(joint_angles, grf, mass_kg, ...)` | Standalone preprocessing function |
| `JOINT_ORDER` | List of the 16 DOF names in correct order |
| `N_FEATURES` (62), `N_OUTPUTS` (4), `SAMPLING_HZ` (100) | Constants |

Most users only need `JCFPredictor`. The rest are available if you want to plug the model into a custom pipeline.

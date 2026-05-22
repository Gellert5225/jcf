# Posture-Optimizer Knee JCF Surrogate — Integration Guide

A GRF-free MLP surrogate that predicts knee joint contact force (JCF) and the external knee adduction moment (KAM) from joint posture plus a short window of body-segment positions. Designed for use inside a posture optimizer where the optimizer picks one configuration at a time and queries the surrogate for a scalar load estimate. Trained jointly on quasi-static and walking trials from the AddBiomechanics database (Carter, Moore, Lencioni/Tiziana, Falisse).

All inference logic lives in one self-contained module: **`inference_posture_surrogate.py`**.

## Why this variant

Standard JCF surrogates take ground reaction forces as input. That's fine for treadmill or instrumented-floor recordings, but not for an over-actuated dynamic simulation where there is no real ground contact and therefore no GRF signal to feed the model. This variant was trained without GRF; the surrogate compensates by reading the geometric pose of the body (COM and feet expressed relative to the pelvis) over a short temporal window.

For a single-posture optimizer query, the position window collapses to a snapshot replicated across the window — exactly the regime training saw on quiet-stance trials. Static accuracy is the best of any variant tested (see [Performance](#performance)).

## Files needed for integration

| File | Purpose |
|---|---|
| `inference_posture_surrogate.py` | Self-contained inference module |
| `best_model_mlp_both_qinst_pelvisrel_winpos_w20.pt` | Trained checkpoint (~0.2 MB) |

Drop both files into your project directory. Only dependencies are `numpy` and `torch`.

## What you need at runtime

| Input | Source | Format |
|---|---|---|
| **Joint angles** | optimizer state / IK | `(16,)` lower-body DOFs in radians (one snapshot) |
| **COM world position** | model state / centroidal dynamics | `(3,)` meters |
| **Right foot (`calcn_r`) world position** | model FK at the same posture | `(3,)` meters |
| **Left foot (`calcn_l`) world position** | model FK at the same posture | `(3,)` meters |
| **Pelvis world position** | model FK at the same posture | `(3,)` meters |
| **Subject mass** | subject record | scalar kg |
| **Subject height** | subject record | scalar m |

The model consumes **pelvis-relative** positions:
- `com_rel = com_world - pelvis_world`
- `r_rel   = calcn_r_world - pelvis_world`
- `l_rel   = calcn_l_world - pelvis_world`

You can pass either the relatives directly (preferred — most optimizers compute them anyway) or compute the relatives in your wrapper before calling the predictor. Absolute lab position is intentionally removed because the model was trained on pelvis-zero-centered joint angles and never sees absolute world coordinates.

GRF is **not** used. Mass and height are required for output denormalization.

## The 16 lower-body joint angles (in order)

| Index | DOF |
|---|---|
| 0 | pelvis_tilt |
| 1 | pelvis_list |
| 2 | pelvis_rotation |
| 3 | pelvis_tx |
| 4 | pelvis_ty |
| 5 | pelvis_tz |
| 6 | hip_flexion_r |
| 7 | hip_adduction_r |
| 8 | hip_rotation_r |
| 9 | knee_angle_r |
| 10 | ankle_angle_r |
| 11 | hip_flexion_l |
| 12 | hip_adduction_l |
| 13 | hip_rotation_l |
| 14 | knee_angle_l |
| 15 | ankle_angle_l |

Angles in radians, translations in meters. The constant `JOINT_ORDER` is exported from `inference_posture_surrogate.py`. Pelvis translations (indices 3–5) are auto-zero-centered inside the predictor; you can pass them as 0 or as raw lab coordinates — the predictor handles it.

**Dead joints excluded:** `subtalar_r`, `mtp_r`, `subtalar_l`, `mtp_l`. Drop these from your IK output before constructing the input.

## Output channels

The predictor returns a dict with these keys (scalars for a single posture query, shape `(N,)` for batched):

| Key | Meaning | Units |
|---|---|---|
| `fx` | Knee JCF anterior-posterior | BW (body weights) |
| `fy` | Knee JCF axial (compressive) | BW |
| `fz` | Knee JCF medio-lateral | BW |
| `mx` | External knee adduction moment | BW · H (height-normalized) |
| `fy_n` | Axial force in absolute units | N |
| `mx_nm` | KAM in absolute units | N · m |
| `f_medial` | Schipplein-Andriacchi medial proxy | BW |

`f_medial = α · |fy| + (mx · H) / d_m`, with defaults `α = 0.5`, `d_m = 0.045 m`. Override per call if your subject is varus/valgus-aligned: `predict_static(..., alpha=0.6, d_m=0.05)`.

**Sign convention for `mx`:** positive = medial compartment loading during right-leg stance. No post-hoc negation needed.

## Quick start — static posture (optimizer query)

```python
from inference_posture_surrogate import JCFPostureSurrogate

predictor = JCFPostureSurrogate(
    "best_model_mlp_both_qinst_pelvisrel_winpos_w20.pt"
)

out = predictor.predict_static(
    joint_angles=q,                  # (16,) radians
    com_rel_pelvis=com_rel,          # (3,) com - pelvis, meters
    calcn_r_rel_pelvis=r_rel,        # (3,) right foot - pelvis
    calcn_l_rel_pelvis=l_rel,        # (3,) left foot - pelvis
    mass_kg=70.0,
    height_m=1.75,
)

print(out["fy_n"])      # axial knee force in N
print(out["f_medial"])  # medial compartment proxy in BW
```

Internally `predict_static` replicates the single position snapshot across the 20-frame window the model expects. This matches the quiet-stance regime in training — positions barely move on the 200 ms timescale of one window, so a constant snapshot is the correct interpretation of "static."

## Batched optimizer inner loop

Evaluate many candidate postures in one call:

```python
N = 256
q_batch       = ...   # (N, 16)
com_rel_batch = ...   # (N, 3)
r_rel_batch   = ...   # (N, 3)
l_rel_batch   = ...   # (N, 3)

out = predictor.predict_static(
    q_batch, com_rel_batch, r_rel_batch, l_rel_batch,
    mass_kg=70.0, height_m=1.75,
)
# out["fy"] shape (N,), out["f_medial"] shape (N,), etc.
```

Single subject per call (one `mass_kg`/`height_m` value). For multi-subject batched calls, loop over subjects.

## Walking with a position history

If you have a temporal trajectory and want a prediction at the current frame, pass an actual 20-frame window of positions. The prediction corresponds to the **center frame** of the window:

```python
out = predictor.predict_windowed(
    joint_angles=q,                  # (16,) current snapshot
    com_rel_window=W_com,            # (20, 3) com - pelvis over 20 frames
    calcn_r_rel_window=W_r,          # (20, 3)
    calcn_l_rel_window=W_l,          # (20, 3)
    mass_kg=70.0,
    height_m=1.75,
)
```

Window convention: indices 0..9 are past frames, index 10 is current, indices 11..19 are future (or edge-replicated if you're streaming and don't have lookahead). The model was trained with a centered window; if you only have past data you can replicate the current frame for indices 10..19.

Sampling rate during training was the native IK rate of each subject (90–250 Hz). The window is interpreted in samples, not seconds — if your simulator runs at 100 Hz, 20 frames = 200 ms. For very different rates you may want to resample your trajectory to ~100 Hz before windowing for best accuracy.

## What the model does NOT do

- **No absolute lab position.** Pelvis translations get zero-centered on the way in. Two postures that differ only by translation produce identical predictions.
- **No GRF.** Don't try to feed force-plate data — there's no input channel for it.
- **No automatic kinematics.** You must compute COM and `calcn_r`/`calcn_l` positions yourself from your model state (forward kinematics on the Rajagopal model, or whatever model your optimizer uses, then take world positions of those body frames).
- **No pelvis_tilt sign flip / coordinate-axis conversion.** Use the OpenSim convention exactly as listed in the joint-order table.

## Performance

Test-set MAE on held-out subjects, separated by activity. Lab labels indicate cross-lab generalization: subjects with that lab prefix were unseen during training.

| Regime | All MAE (BW) | Within-lab Carter | Cross-lab Moore | Cross-lab Tiziana |
|---|---|---|---|---|
| Static F_y | **0.045** | — | 0.035 | 0.093 |
| Walking F_y | 0.153 | 0.148 | 0.157 | 0.137 |

For static postures this is the best-performing variant across everything tested, including the GRF-based MLP (`qfmh`, Moore static MAE 0.073) and the windowed CNN (Moore static MAE 0.059). For walking, accuracy is comparable to the GRF-based surrogate.

Correlation (Pearson r) on walking F_y: 0.899 overall, 0.948 within-lab Carter, 0.767 cross-lab Moore.

## Architecture and training details (FYI)

- Per-frame MLP: 3 hidden layers × 128 units, LayerNorm + ReLU + Dropout 0.3 (set to 0.0 at inference).
- Input dimension: 198 = 16 (q) + 20 × 9 (windowed positions) + 2 (mass, height).
- Output dimension: 4 = Fx, Fy, Fz, Mx.
- Forces normalized by body weight (`m · g`, g = 9.81). Mx normalized by BW · height.
- Trained on the union of `training/walking` and `training/static` splits from `./jcf/full_duration/`.
- Best checkpoint at epoch 90 of 100 (val_loss 0.012415).
- Model file `qinst_pelvisrel_winpos_w20` is short for: q-instantaneous, pelvis-relative positions, position-only windowing, window size 20.

## Troubleshooting

**Outputs are huge / negative axial force.** Check units: joint angles in radians, positions in meters, mass in kg, height in meters. Negative `fy` is expected — it's the axial compressive force on the tibial plateau and the model convention is that compression is negative (matches the OpenSim Joint Reaction output sign).

**Outputs barely change with posture.** Make sure you're passing `joint_angles` for the actual posture being evaluated, not the resting pose. Also verify the position vectors are pelvis-relative (subtract pelvis from each before passing in).

**Wrong shape error.** `predict_static` expects positions as `(3,)` (single posture) or `(N, 3)` (batched). `predict_windowed` expects `(20, 3)` or `(N, 20, 3)`. Mismatched batch dims between `joint_angles` and the position arrays will error.

**Predictor reports a different `input_set`.** You loaded the wrong checkpoint. This module only accepts `qinst_pelvisrel_winpos_w20`. For the GRF-based variant use `inference_static_simple.py`; for streaming walking control with COM velocity use the `qvcomrel_w20` checkpoint (separate helper not yet written).

## Smoke test

```bash
python inference_posture_surrogate.py best_model_mlp_both_qinst_pelvisrel_winpos_w20.pt
```

Should print sensible numbers for a synthetic upright-stance posture (axial force around −0.8 BW per knee, medial proxy around +0.3 BW). If you get NaNs or absurd magnitudes, the model is being fed inputs in the wrong units or order.

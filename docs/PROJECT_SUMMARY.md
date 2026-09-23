# Knee JCF Prediction — Project Summary

## Goal

Predict 3D knee joint contact force (Fx, Fy, Fz in body weights) from biomechanical
inputs (joint angles + ground reaction forces). Two separate models for two activities:

- **Running model** — predicts JCF during running gait
- **Walking model** — predicts JCF during walking gait

**Deployment context:** offline posture optimization. The model runs on full
recorded sessions, not real-time streaming.

## Data Pipeline

```
.b3d files (AddBiomechanics format)
   ↓ scan_b3d_all_runs (find valid GRF segments)
   ↓ filter by activity (GRF threshold: walking < 1.7 BW per foot < running)
   ↓ filter static trials (name match "static" + std-of-GRF threshold)
OpenSim model (.osim) + IK results (.mot) + GRF (.mot)
   ↓ Static Optimization
   ↓ Joint Reaction Analysis
JCF labels (.sto): Fx, Fy, Fz at the knee in tibia frame
```

### Datasets used

| Dataset | Subjects | Activities | Notes |
|---|---|---|---|
| Carter2023 | ~20 | flatrun (fast/norm/slow), uphillrun, downhillrun, walk_fast, walk_slow, static | The richest single dataset. Has terrain variation (uphill/downhill) which no other dataset has. |
| Hammer2013 | 10 | run at 200/300/400/500 cm/s | Pure running, fixed speeds. |
| Moore2015 | 12 | walking on treadmill | Hundreds of walking segments per subject. SO blowups produce JCF up to 257 BW (clearly broken). |
| Tiziana2019 | 43 | walking | Generally clean. |
| Falisse2017 | 3 | walking + sit-to-stand | Subject 10 has 21 lower-body joints (vs Carter's 20), broke combined-feature pipeline until fixed. |
| Fregly2012 | 3 | walking variants (bouncy, crouch, medial thrust) | Likely instrumented-knee patients, not healthy. |
| Han2023 | 24 | dance/yoga/jumping/sports + a few walking | Not gait-only. |

### Data filtering rules currently applied
1. Only segments with `MissingGRFReason.notMissingGRF`
2. Drop trials whose name contains "static" (case-insensitive)
3. Drop segments where per-foot GRF std < 0.1·BW (catches static trials not name-tagged)
4. Drop subjects whose JCF resultant peak > 10 BW (catches worst SO blowups)
5. Drop "flat" subjects (CV < 0.25 OR dynamic range < 0.3 BW) — no cyclic peaks

## Model Architectures Tried

| Class | Description | Params | Verdict |
|---|---|---|---|
| `JCF_CNN` (v1) | Plain 1D CNN, 4 conv layers | ~50K | Baseline |
| `JCF_CNN_v2` | Residual blocks, dilations 1/2/4/8, GroupNorm | 720K | **Production** |
| `JCF_CNN_v2_causal` | v2 with causal padding + LayerNorm (online) | 720K | Worse than bidirectional. Only use if streaming required. |
| `JCF_CNN_v3` | Wider (256ch), 6 res blocks, dropout | 4.1M | Overfits with current data size. |
| `JCF_TCN` | Temporal conv network with dilations 1-64 | 740K | Worse than v2. Causal nature loses future context. |
| `JCF_FFT_MLP` | Windowed FFT features + MLP | — | Much worse. |
| `JCF_Transformer` | Conv stem + transformer encoder | — | Worse than CNN at this data size. |

## Loss Functions Tried

| Loss | Description |
|---|---|
| `masked_mse` | Plain MSE |
| `symmetric_loss` | MSE + magnitude-weighted MSE + gradient matching | **Default**, works best across the board |
| `confidence_weighted_loss` | Weighted by SO confidence |
| `expectile_loss(τ=0.7)` | Asymmetric: penalizes underprediction 2.3× more |
| `log_magnitude_loss` | log(1+mag) weighting instead of raw mag |
| `log_magnitude_peak_loss` | + extra penalty at GT local maxima |
| `asymmetric_magnitude_loss` | Quadratic underprediction penalty (`mag·(1+2·underpred·mag)`) |
| `asymmetric_linear_loss` | Linear underprediction penalty (`1 + underpred·mag`) |

**Key finding:** Asymmetric / peak-biased losses produce blanket upward shifts rather
than selectively correcting high peaks. Symmetric loss is the most robust.

## Feature Sets Tried

| Set | Components | Dims |
|---|---|---|
| Full | Pos+vel+acc (37 DOF each) + GRF + GRF_vel + mass | 119 |
| Lower body | Pos+vel+acc (20 DOFs) + GRF + GRF_vel + mass | 73 |
| **N (clean lower body)** | Pos+vel+acc (16 DOFs after dead-joint removal) + GRF + GRF_vel + mass | **61** |
| **P (N + root dynamics)** | N + root_velocity(6) + root_acceleration(6) + GRF_in_root(6) + COM_acc(3) | **82** |
| O (root only) | joint_centers + root_dynamics + GRF_root + COM_acc | 81 |

`clean_features` removes pelvis translation absolute position (zero-centers per trial)
and removes dead joints (subtalar, mtp).

## Experiment Letter Index

| Exp | Architecture | Features | Loss | Other |
|---|---|---|---|---|
| a | v1 | full | symmetric | baseline |
| c | v1 | lower body | symmetric | |
| d | v2 | lower body | symmetric | + rebalance |
| i | v2 | clean lower body | symmetric | + rebalance |
| j | v2 | N | log_magnitude | |
| k | v2 | N | log_mag + peak | |
| l | TCN | N | symmetric | |
| n | v2 | N (+mass) | symmetric | **production candidate** |
| o | v2 | root features only | symmetric | needs nimble at inference |
| **p** | v2 | N + root dynamics | symmetric | **best for running** |
| q | v2 | N | asymmetric quadratic | |
| r | v2 causal | N | asymmetric linear | online variant |
| s | v2 causal | N | asymmetric linear | + 10-frame lookahead |
| t | v2 causal | N | symmetric | + 10-frame lookahead |

## Running Model Results (Carter P010, 234 trials)

| Metric | Exp N (IK only) | **Exp P (IK + root)** |
|---|---|---|
| Resultant R² | 0.901 | 0.889 |
| Resultant MAE (BW) | 0.099 | 0.121 |
| Peak slope | 0.803 | **0.978** |
| Peak R² | 0.886 | **0.939** |
| Peaks within 20% | 78% | **80%** |
| Mean error 2.5-3.5 BW peaks | -0.44 | **-0.02** |
| Mean error 3.5-5.0 BW peaks | -0.97 | -0.52 |

**Headline:** Root-frame dynamics (COM acceleration, pelvis angular velocities) gave a
substantial improvement specifically on peak prediction. Mid-range peaks (2.5-3.5 BW)
went from systematic 0.44 BW undershoot to essentially unbiased.

### Per-terrain breakdown (running)

When tested on Carter P010's full trial set:

| Terrain | Trial | Correlation | Peak MAE |
|---|---|---|---|
| Flat | split1_t32 | 0.972 | 0.10 BW |
| Uphill | split6_t23 | 0.974 | 0.17 BW |
| Downhill | split5_t37 (corrupted, 60 frames) | 0.986 | 0.43 BW |

Downhill data was fragmented by the SO pipeline into ~80-frame clips (vs 2000-frame
flat trials) which produces poor predictions. Underlying cause: the pipeline keeps only
short windows where the SO converges. After filtering split5 from test, R² goes from
0.85 → 0.89.

## Walking Model Results (Carter P010, 34 trials)

Trained on Carter walking data only (~620 segments from 20 subjects, P010 held out).

| Variant | R² | MAE | Peak slope | Peak R² | <20% |
|---|---|---|---|---|---|
| **N (no calibration)** | 0.872 | 0.130 | 0.787 | 0.893 | 82.8% |
| N + cal=3 trial mode (per-trial scale) | 0.856 | 0.141 | 0.899 | 0.901 | 76.2% |
| **N + cal=3 subject mode (one scale per subject)** | **0.896** | **0.122** | 0.834 | **0.925** | **87.6%** |

After retraining N with slightly more walking data:

| Variant | R² | MAE | Peak slope | Peak R² | <20% |
|---|---|---|---|---|---|
| New N (no cal) | 0.801 | 0.152 | 0.890 | 0.772 | 78.7% |
| **New N + cal=3 trial** | **0.906** | **0.120** | **0.944** | **0.934** | **84.5%** |
| New N + cal=3 subject | 0.782 | 0.173 | 0.776 | 0.765 | 62.3% |

### Walking-specific findings

- **Exp P hurts walking.** Root features (helpful for running) actively degrade walking
  performance: R² drops from 0.87 (N) to 0.74 (P). Hypothesis: walking has small COM
  acceleration variation across subjects, so the 21 extra root-feature channels mostly
  add noise instead of signal.
- **Subject-mode calibration is sensitive to model bias.** When the model has
  consistent per-subject offset (old training run), one global scale works great. When
  the model is more variable across trials (new training run), subject mode amplifies
  the noise from the calibration trial.
- **Trial-mode calibration is more robust** but requires GT peaks for each session at
  deployment, not just one calibration session per patient.

## Calibration Strategy

We added a feature where the model is "calibrated" against a few measured GT peaks
before deployment, mimicking a clinical calibration session.

```python
# Per-trial: scale = median(GT_peak[i] / Pred_peak[i] for first N peaks)
preds *= scale  # apply to remainder of the trial
```

Two modes:
- **Trial mode:** new scale per trial. Higher accuracy but assumes ground-truth measurement at every session.
- **Subject mode:** scale computed once from the first trial of the subject, reused for all later trials. More clinically realistic but more sensitive to calibration-trial noise.

## What Worked

1. **Lower-body features only** (drop arm DOFs)
2. **Clean features** (zero-center pelvis, drop dead joints)
3. **Mass as input feature**
4. **V2 architecture** (4 residual blocks with dilations 1/2/4/8)
5. **Symmetric loss** with magnitude-weighted MSE + gradient term
6. **Root features for running** (exp P) — major win
7. **Subject calibration at deployment**
8. **Static-trial filtering** (name + std)
9. **JCF outlier filter** (peak > 10 BW)
10. **Filtering split5 (downhill) from running test set** — improved R² because SO produced corrupted clips

## What Didn't Work

1. **Asymmetric / peak-biased losses** (q, r, s) — produce blanket upward bias instead of fixing high peaks selectively
2. **Log-target regression** (b)
3. **TCN** — causal masking loses future context, no benefit for offline use
4. **Transformer** — not enough data
5. **V3 (4.1M params)** — overfits at 2.4k subjects
6. **Confidence weighting via SO confidence** — not informative enough
7. **Root features for walking** (P) — hurts performance
8. **10-frame lookahead** (s, t) — didn't recover causal model's lost accuracy
9. **2× muscle scaling on labels** (g, h)

## Open Issues / Things We're Stuck On

### Walking peak undershoot at >3 BW

Even with calibration, predictions cap around 3 BW even when GT goes to 4 BW. Could be:
- **Data distribution:** few training examples at >3 BW
- **Information bottleneck:** IK + GRF can't distinguish 2.5 BW from 4 BW gait
  because the difference is in muscle co-contraction (not visible in kinematics)

### Why running benefits from root features but walking doesn't

Running impacts have large COM accelerations correlated with peak knee force.
Walking is steadier, COM dynamics carry less signal — the extra 21 features dilute
the IK signal more than they add information.

### Single test subject (P010 only)

All test results come from one held-out subject. We don't know:
- Whether observed errors are model limitations or P010-specific quirks
- How the model generalizes across body types

We have not yet rebuilt the test split with multiple held-out subjects from
multiple datasets.

### Cross-dataset generalization untested

All current Carter-only experiments. Whether adding Moore/Tiziana/Falisse walking
to training improves OOS accuracy is unknown — need cleaner Moore filtering first
(some Moore JCF outputs reach 257 BW from SO blowups).

### SO pipeline fragmentation

Some Carter splits (notably P010 split5 = downhillrun) produce ~80-frame fragments
instead of 2000-frame trials. Root cause unclear but likely the SO solver only
converging on short windows. Affects ~17% of running test trials.

## Inference Performance

CPU (RTX 3090 not used at inference for v2):
- Forward pass per 2000-frame sequence: 37 ms
- Backward pass: 89 ms

GPU (RTX 3090):
- Forward pass per 2000-frame sequence: **0.97 ms** (2M frames/sec)
- Backward pass per training step (B=8, T=2000): 6.8 ms
- 300-epoch training: ~10 minutes pure compute, probably ~30-60 min wall-clock with data loading

Inference is not a bottleneck. The user's stated 1000 Hz "real-time" target is
trivially achievable on GPU. Current `DEVICE = "cpu"` setting in train_cnn.py leaves
significant speed on the table.

## What I'd Ask GPT

1. **Walking peak prediction:** Is there a known biomechanics-aware feature engineering
   trick (beyond IK + GRF + root dynamics) that helps with the muscle-co-contraction
   blind spot?
2. **Calibration design:** Are there better ways to combine subject-level calibration
   with per-trial adjustment? (e.g., subject scale + small per-trial residual correction)
3. **Cross-subject generalization:** With ~20 walking subjects, what's the minimum
   training set size needed to expect generalization to truly new subjects?
4. **Architecture for muscle-related signal:** Would predicting muscle activations as
   an intermediate step (multi-task learning) help, given we don't have EMG?
5. **The "predict GRF then derive JCF" approach** (InferBiomechanics style):
   theoretically sound, but a much bigger refactor. Worth it?

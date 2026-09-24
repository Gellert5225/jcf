# mlp-grffree-contact — GRF-free knee JCF surrogate with binary contact flags (recommended)

**File:** `mlp-grffree-contact.pt` · 263 KB · sha256 `1a9e4158164a7a2e…` · promoted 2026-09-23
**Source run:** `runs/mlp/best_model_mlp_both_qinst_pelvisrel_winpos_contact_w20_fmW0p5.pt` (identical bytes; the run file is kept for `test_*.py` lookups)
**Checkpoint metadata:** model_type=mlp, input_set=qinst_pelvisrel_winpos_contact_w20, activity=both, fmedial_weight=0.5, activation=relu, dropout=0.4, epoch=46, val_loss=0.1043, n_features=238, dq_method=central

## What it is
Per-frame MLP predicting right-knee (Fx, Fy, Fz) in BW and the knee adduction moment Mx in BW·H. Inputs: 16 joint angles (centre frame), 20-frame window of COM / right-foot / left-foot positions relative to the pelvis, 20-frame binary right/left contact flags, mass, height — 238 channels. No GRF needed at inference.

## How it was trained
```bash
python train_mlp.py --activity both --input-set qinst_pelvisrel_winpos_contact_w20 --epochs 100 --lr-patience 5 --fmedial-weight 0.5 --dropout 0.4
```
Data: AddBiomechanics trials with OpenSim SO+JR labels (`jcf/full_duration/training/`), 80/20 subject-level split.

## Performance
Held-out (323 walking trials, 18 unseen subjects): F_medial MAE **0.195 BW** (base model 0.268, −27 %, better on 264/323 trials); Mx MAE 0.0043 BW·H (−31 %); late-stance push-off spike recovered **91 %** of GT magnitude (base 81 %) over 1,315 entries. Evidence: `figures/heldout/contact_vs_baseline_heldout_summary.png`.

## Known issues / caveats
**Ground-truth check on a synthetic optimizer trajectory (Sep 23, 2026):** OpenSim SO+JR on William Chong's dynamic-stage gait (his q + contact wrenches, model borrowed from a 74.6 kg/1.71 m subject) gives right-stance F_medial ≈ 1.4 BW mean, peaks 2.2 / 1.8 BW. This model predicts 0.44 / 0.51 BW there — a ~3× under-prediction. `mlp-grffree-base` reads 1.86 BW in the second stance but 0.44 in the first (corr with GT 0.04), and `contact-MSE` 0.59 / 1.35. Cause: that gait walks on locked knees — ≈5° total knee range over the whole cycle on both legs vs 15–60° in mocap (verified by the joint mapping, a scan of all 37 q columns, and a mapping-free pelvis-to-foot distance check: 5 cm variation vs 20 cm in real gait; `figures/gt/trajectory_sep23_knee_kinematics.png`) — with half the hip adduction; with no GRF input, every GRF-free surrogate reads the loaded straight leg as quiet standing. The contact flags do not fix this because the training data contains no straight-leg single-stance at ~2.5 BW axial load. Remedies: `docs/LABEL_AND_FINETUNE_GUIDE.md` (SO+JR labelling of optimizer trajectories + fine-tuning). On real mocap gait this model remains the most accurate (held-out MAE 0.195 BW vs 0.268). Use it for natural gait; for optimizer trajectories with contact wrenches, run SO+JR directly (minutes) or use a GRF-consuming model. Evidence: `figures/heldout/trajectory_sep23_groundtruth_vs_surrogates.png`. GT caveat: SO saturated >5 muscles in 42 % of frames and used large reserves, so treat the magnitudes as indicative. Contact flags must be 0/1 per frame (|F_vertical| > 5 % BW or the stance schedule). ReLU activation.

## Download from Hugging Face
```python
from huggingface_hub import hf_hub_download
path = hf_hub_download("gellert5225/jcf-knee-surrogates", "mlp-grffree-contact/mlp-grffree-contact.pt", revision="v1-isrr2026")
```

## Load
```python
from inference_posture_surrogate import JCFPostureSurrogate
m = JCFPostureSurrogate("models/mlp-grffree-contact/mlp-grffree-contact.pt")   # see docs/INFERENCE_GUIDE_MLP.md
```
Full integration guide: `docs/INFERENCE_GUIDE_MLP.md`.

**Pelvis-centering (Sep 23, 2026):** remove the `pelvis_tx/ty/tz` drift at inference for overground trajectories (running mean ≈1 s, or zero; see `docs/INFERENCE_GUIDE_MLP.md`, *Pelvis centering*). On the Sep 23 trajectory this changes this model from 0.33/0.40 BW to 0.57/0.52 BW in the two right stances (ground truth 1.76/1.17); the plain-MSE contact run (`runs/mlp/…contact_w20.pt`) goes to 0.93/0.91 with correlation 0.88 to ground truth. The locked-knee under-prediction remains in all cases.

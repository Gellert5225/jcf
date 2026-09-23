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
On a synthetic optimizer trajectory it returned peak |F_medial| ≈ 0.67 BW where `mlp-grffree-base` returned much higher values; not yet validated against OpenSim ground truth on that kind of motion (open, Sep 2026). Contact flags must be 0/1 per frame, derived with |F_vertical| > 5 % BW or from the stance schedule. ReLU activation.

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

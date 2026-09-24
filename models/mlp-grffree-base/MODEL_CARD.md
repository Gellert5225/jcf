# mlp-grffree-base — GRF-free knee JCF surrogate (paper model, no contact flags)

**File:** `mlp-grffree-base.pt` · 242 KB · sha256 `1ee13b5725c8dfcc…` · promoted 2026-09-23
**Source run:** `runs/mlp/best_model_mlp_both_qinst_pelvisrel_winpos_w20.pt` (identical bytes; the run file is kept for `test_*.py` lookups)
**Checkpoint metadata:** model_type=mlp, input_set=qinst_pelvisrel_winpos_w20, activity=both, dropout=0.3, epoch=89, val_loss=0.0124, n_features=198, dq_method=central

## What it is
Per-frame MLP, same outputs as the contact variant. Inputs: 16 joint angles, 20-frame window of COM / foot positions relative to the pelvis, mass, height — 198 channels. Plain per-channel MSE loss. This is the checkpoint used for the gait-optimization results in the ISRR 2026 paper.

## How it was trained
```bash
python train_mlp.py --activity both --input-set qinst_pelvisrel_winpos_w20        # defaults: dropout 0.3, no F_medial loss term
```
Data: AddBiomechanics trials with OpenSim SO+JR labels (`jcf/full_duration/training/`), 80/20 subject-level split.

## Performance
Held-out (same 323 trials): F_medial MAE 0.268 BW; Mx MAE 0.0062 BW·H; late-stance spike recovered 81 %. Paper: walking MAE 0.153 BW (r = 0.899), quasi-static pooled MAE 0.045 BW.

## Known issues / caveats
No stance information: cannot distinguish single from double support, so it under-predicts the push-off KAM reversal. Reads higher than the contact model on synthetic optimizer trajectories (see the contact card).

## Download from Hugging Face
```python
from huggingface_hub import hf_hub_download
path = hf_hub_download("gellert5225/jcf-knee-surrogates", "mlp-grffree-base/mlp-grffree-base.pt", revision="v1-isrr2026")
```

## Load
```python
from inference_posture_surrogate import JCFPostureSurrogate
m = JCFPostureSurrogate("models/mlp-grffree-base/mlp-grffree-base.pt")
```
Full integration guide: `docs/INFERENCE_GUIDE_MLP.md (API identical; omit the contact arguments)`.

**Pelvis-centering artefact (Sep 23, 2026):** on overground trajectories centered on their own mean, this model's output drifts with walking distance (training is ~90 % treadmill, centered pelvis range ≤ 0.1 m). On the ISRR reference motion the 1.86 BW peak at 4.2 s and the −0.6 BW start are that artefact; with `pelvis_tx/ty/tz` zeroed the model reads 0.5–0.8 BW throughout. Remove the pelvis drift at inference — stride-scale running mean, or zero it (`docs/INFERENCE_GUIDE_MLP.md`, *Pelvis centering*); zeroing costs ~10 % on real gait, the running mean ~1 %. Evidence: `figures/gt/trajectory_sep23_pelvis_centering_effect.png`.

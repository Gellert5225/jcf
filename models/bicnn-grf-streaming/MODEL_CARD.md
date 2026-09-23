# bicnn-grf-streaming — streaming BiCNN knee JCF surrogate (with GRF)

**File:** `bicnn-grf-streaming.pt` · 8523 KB · sha256 `169e0b623a433a31…` · promoted 2026-09-23
**Source run:** `runs/bicnn/best_model_n_d05_m_s2_s2.pt` (identical bytes; the run file is kept for `test_*.py` lookups)
**Checkpoint metadata:** epoch=39, val_loss=0.0142, n_features=62

## What it is
Bidirectional dilated 1-D CNN over a ~240-frame receptive field. Inputs per frame: 16 joint angles, per-foot GRF (6, Newtons), gait speed, mass — 62 channels after preprocessing. Outputs (Fx, Fy, Fz, Mx). For offline / streaming analysis of recorded gait where force-plate data exists.

## How it was trained
```bash
python train_cnn.py --exp n_d05_m_s2 --seed 2        # experiment code n_d05_m_s2, seed suffix _s2
```
Data: AddBiomechanics trials with OpenSim SO+JR labels (`jcf/full_duration/training/`), 80/20 subject-level split.

## Performance
Held-out walking (33 Carter + 142 Moore trials, 100,688 frames): Fy r = 0.923, MAE 0.135 BW. Quasi-static: within-lab (Moore) MAE 0.059 BW, cross-lab (Tiziana) 0.238 BW. ~0.72 M parameters; 0.83 ms per 240-frame pass on an RTX 3090.

## Known issues / caveats
Requires measured GRF — not usable inside the GRF-free posture optimizer. Regression-to-mean above ~3 BW.

## Download from Hugging Face
```python
from huggingface_hub import hf_hub_download
path = hf_hub_download("gellert5225/jcf-knee-surrogates", "bicnn-grf-streaming/bicnn-grf-streaming.pt", revision="v1-isrr2026")
```

## Load
```python
from inference import JCFPredictor
p = JCFPredictor("models/bicnn-grf-streaming/bicnn-grf-streaming.pt")   # see docs/INFERENCE_GUIDE.md
```
Full integration guide: `docs/INFERENCE_GUIDE.md`.

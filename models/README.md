---
license: cc-by-4.0
library_name: pytorch
tags:
- biomechanics
- gait-analysis
- knee-joint-contact-force
- surrogate-model
- opensim
- addbiomechanics
---
# models/

> **Weights are hosted on Hugging Face:** https://huggingface.co/gellert5225/jcf-knee-surrogates (private during review; tagged revision `v1-isrr2026`).
> **Code:** https://github.com/Gellert5225/jcf — the `.pt` files are *not* in the GitHub repo.
> **Training data:** https://huggingface.co/datasets/gellert5225/jcf-knee-labels (OpenSim IK/GRF/SO/JR outputs; `pipeline/fetch_data.py` in the code repo reconstructs `jcf/`).
>
> ```python
> from huggingface_hub import hf_hub_download
> ckpt = hf_hub_download("gellert5225/jcf-knee-surrogates", "mlp-grffree-contact/mlp-grffree-contact.pt", revision="v1-isrr2026")
> ```
> (private repo → run `hf auth login` once with a token that has read access.)
>
> Models predict right-knee contact force and adduction moment from kinematics; training data derive from
> AddBiomechanics (CC BY 4.0), hence the CC-BY-4.0 license. Cite: Chong, Li, Billi, Khatib, Demircan,
> *Antalgic Gait Synthesis*, ISRR 2026.

All trained checkpoints live here; the data folder `jcf/` no longer holds any `.pt` files.
`*.pt` is git-ignored (≈400 MB total); the READMEs and model cards are tracked.

## Curated models (use these)

| Folder | Model | Use it for |
|---|---|---|
| `mlp-grffree-contact/` | GRF-free MLP + binary contact flags | Posture / gait optimizer. Most accurate on held-out gait. **Recommended.** |
| `mlp-grffree-base/` | GRF-free MLP, no contact flags | The checkpoint behind the ISRR 2026 optimization results; fallback if the contact variant misbehaves on synthetic trajectories. |
| `bicnn-grf-streaming/` | Bidirectional CNN, needs measured GRF | Offline / streaming analysis of recorded gait. |

Each folder holds `<name>.pt` and a `MODEL_CARD.md` (inputs, training command, held-out metrics, caveats, load snippet).
Loading: `inference_posture_surrogate.JCFPostureSurrogate(path)` for the MLPs, `inference.JCFPredictor(path)` for the BiCNN.

## Naming scheme

`<family>-<inputs>-<variant>`: family = `mlp` | `bicnn`; inputs = `grffree` (kinematics + pelvis-relative positions only) | `grf` (consumes force-plate GRF); variant = `contact` (adds binary contact flags), `base`, `streaming`, … Folder name = file stem.

## Old → new names (for collaborators holding the original files)

| Original run file | Curated name |
|---|---|
| `best_model_mlp_both_qinst_pelvisrel_winpos_contact_w20_fmW0p5.pt` | `mlp-grffree-contact.pt` |
| `best_model_mlp_both_qinst_pelvisrel_winpos_w20.pt` | `mlp-grffree-base.pt` |
| `best_model_n_d05_m_s2_s2.pt` | `bicnn-grf-streaming.pt` |

Bytes are identical — the inference modules read the `input_set` tag inside the file, so renaming is safe.

## Training runs (`runs/`)

`train_mlp.py` writes to `runs/mlp/`, `train_cnn.py` to `runs/bicnn/`, using their script-generated names so that
`test_mlp.py` / `test_cnn.py --exp <code>` keep finding them. To promote a run: copy it into a new `models/<name>/`
folder with a `MODEL_CARD.md`.

| Run file | Size | Date | What it is | Checkpoint metadata |
|---|---|---|---|---|
| `runs/bicnn/best_model_n.pt` | 8509 KB | 2026-04-28 | BiCNN experiment `n` (see `--exp` in train_cnn.py) | epoch=56, val_loss=0.0099, n_features=61 |
| `runs/bicnn/best_model_n3.pt` | 47937 KB | 2026-04-28 | BiCNN experiment `n3` (see `--exp` in train_cnn.py) | epoch=94, val_loss=0.0101, n_features=61 |
| `runs/bicnn/best_model_n3_bin.pt` | 47959 KB | 2026-04-28 | BiCNN experiment `n3_bin` (see `--exp` in train_cnn.py) | epoch=70, val_loss=0.0109, n_features=62 |
| `runs/bicnn/best_model_n3_d05.pt` | 47959 KB | 2026-04-29 | BiCNN experiment `n3_d05` (see `--exp` in train_cnn.py) | epoch=28, val_loss=0.0132, n_features=62 |
| `runs/bicnn/best_model_n3_d05_stance.pt` | 48003 KB | 2026-04-29 | BiCNN experiment `n3_d05_stance` (see `--exp` in train_cnn.py) | epoch=64, val_loss=0.0155, n_features=64 |
| `runs/bicnn/best_model_n3_d05_w15.pt` | 47960 KB | 2026-04-28 | BiCNN experiment `n3_d05_w15` (see `--exp` in train_cnn.py) | epoch=78, val_loss=0.0103, n_features=62 |
| `runs/bicnn/best_model_n3_w15.pt` | 47959 KB | 2026-04-28 | BiCNN experiment `n3_w15` (see `--exp` in train_cnn.py) | epoch=78, val_loss=0.0105, n_features=62 |
| `runs/bicnn/best_model_n3s.pt` | 47958 KB | 2026-04-28 | BiCNN experiment `n3s` (see `--exp` in train_cnn.py) | epoch=73, val_loss=0.0104, n_features=62 |
| `runs/bicnn/best_model_n_d05.pt` | 8520 KB | 2026-04-29 | BiCNN experiment `n_d05` (see `--exp` in train_cnn.py) | epoch=18, val_loss=0.0126, n_features=62 |
| `runs/bicnn/best_model_n_d05_m.pt` | 8522 KB | 2026-04-30 | BiCNN experiment `n_d05_m` (see `--exp` in train_cnn.py) | epoch=72, val_loss=0.0119, n_features=62 |
| `runs/bicnn/best_model_n_d05_m2.pt` | 8523 KB | 2026-04-30 | BiCNN experiment `n_d05_m2` (see `--exp` in train_cnn.py) | epoch=58, val_loss=0.0117, n_features=62 |
| `runs/bicnn/best_model_n_d05_m_s.pt` | 8522 KB | 2026-05-06 | BiCNN experiment `n_d05_m_s` (see `--exp` in train_cnn.py) | epoch=66, val_loss=0.0131, n_features=62 |
| `runs/bicnn/best_model_n_d05_m_s2.pt` | 8522 KB | 2026-05-06 | BiCNN experiment `n_d05_m_s2` (see `--exp` in train_cnn.py) | epoch=66, val_loss=0.0131, n_features=62 |
| `runs/bicnn/best_model_n_d05_m_s2_s2.pt` | 8523 KB | 2026-05-06 | BiCNN experiment `n_d05_m_s2`, seed 2 → **curated `bicnn-grf-streaming`** (paper model) | epoch=39, val_loss=0.0142, n_features=62 |
| `runs/bicnn/best_model_p.pt` | 8730 KB | 2026-04-27 | BiCNN experiment `p` (see `--exp` in train_cnn.py) | epoch=61, val_loss=0.0099, n_features=82 |
| `runs/mlp/best_model_mlp_both_qdqfmh_forward.pt` | 161 KB | 2026-05-19 | MLP **with GRF**: (q, dq, F, mass, height), forward-difference dq (40 ch) | input_set=qdqfmh, dropout=0.3, epoch=21, val_loss=0.011, n_features=40 |
| `runs/mlp/best_model_mlp_both_qfmh_forward.pt` | 153 KB | 2026-05-19 | MLP **with GRF**: (q, F, mass, height), forward-difference (24 ch) | input_set=qfmh, dropout=0.3, epoch=1, val_loss=0.0111, n_features=24 |
| `runs/mlp/best_model_mlp_both_qinst_pelvisrel_w20.pt` | 415 KB | 2026-05-21 | GRF-free MLP, q replicated across the 20-frame window (540 ch) — superseded by winpos | input_set=qinst_pelvisrel_w20, dropout=0.3, epoch=54, val_loss=0.0125, n_features=540 |
| `runs/mlp/best_model_mlp_both_qinst_pelvisrel_winpos_contact_w20_fmW0p5.pt` | 263 KB | 2026-09-17 | GRF-free MLP + windowed contact flags, F_medial-weighted loss 0.5 → **curated `mlp-grffree-contact`** | input_set=qinst_pelvisrel_winpos_contact_w20, fmedial_weight=0.5, activation=relu, dropout=0.4, epoch=46, val_loss=0.1043, n_features=238 |
| `runs/mlp/best_model_mlp_both_qinst_pelvisrel_winpos_w20.pt` | 242 KB | 2026-05-21 | GRF-free MLP, plain MSE → **curated `mlp-grffree-base`** (paper model) | input_set=qinst_pelvisrel_winpos_w20, dropout=0.3, epoch=89, val_loss=0.0124, n_features=198 |
| `runs/mlp/best_model_mlp_both_qinst_pelvisrel_winpos_w20_fmW0p5_silu.pt` | 242 KB | 2026-05-26 | GRF-free MLP, F_medial loss 0.5, SiLU activation (smooth gradients for Ipopt) | input_set=qinst_pelvisrel_winpos_w20, fmedial_weight=0.5, activation=silu, dropout=0.3, epoch=10, val_loss=0.1111, n_features=198 |
| `runs/mlp/best_model_mlp_both_qinst_pelvisrel_winpos_w20_fmW1_silu.pt` | 242 KB | 2026-05-26 | GRF-free MLP, F_medial loss 1.0, SiLU | input_set=qinst_pelvisrel_winpos_w20, fmedial_weight=1.0, activation=silu, dropout=0.4, epoch=13, val_loss=0.1827, n_features=198 |
| `runs/mlp/best_model_mlp_both_qinst_pelvisrel_winpos_w20_fmW2.pt` | 242 KB | 2026-05-22 | GRF-free MLP, F_medial loss 2.0 | input_set=qinst_pelvisrel_winpos_w20, fmedial_weight=2.0, dropout=0.3, epoch=24, val_loss=0.2915, n_features=198 |
| `runs/mlp/best_model_mlp_both_qinst_pelvisrel_winpos_winctr_w20_fmW0p5.pt` | 242 KB | 2026-05-28 | GRF-free MLP, per-window pelvis centering, F_medial loss 0.5 — not recommended (val worse than per-trial centering) | input_set=qinst_pelvisrel_winpos_winctr_w20, fmedial_weight=0.5, activation=relu, dropout=0.2, epoch=45, val_loss=0.1119, n_features=198 |
| `runs/mlp/best_model_mlp_both_qvcomrel.pt` | 154 KB | 2026-05-21 | GRF-free MLP, COM velocity + feet relative to COM, single frame (27 ch) | input_set=qvcomrel, dropout=0.3, epoch=14, val_loss=0.0128, n_features=27 |
| `runs/mlp/best_model_mlp_both_qvcomrel_w20.pt` | 415 KB | 2026-05-21 | GRF-free MLP, COM velocity + feet relative to COM, 20-frame window (540 ch) | input_set=qvcomrel_w20, dropout=0.3, epoch=140, val_loss=0.011, n_features=540 |
| `runs/mlp/best_model_mlp_static.pt` | 173 KB | 2026-05-05 | MLP **with GRF**, full 64-ch feature set, static trials only | dropout=0.3, epoch=35, val_loss=0.0021, n_features=64 |
| `runs/mlp/incomplete/best_model_mlp_both_qinst_pelvisrel_winpos_w20_fmW0p5.pt` | 242 KB | 2026-05-28 | GRF-free MLP, F_medial loss 0.5 — **INCOMPLETE 2-epoch stub (overwrote the May run), do not use** | input_set=qinst_pelvisrel_winpos_w20, fmedial_weight=0.5, activation=relu, dropout=0.4, epoch=2, val_loss=0.1643, n_features=198 |

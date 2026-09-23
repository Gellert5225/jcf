# Knee Joint Contact Force (JCF) Surrogate Models

Neural-network surrogates for the right-knee joint contact force (Fx, Fy, Fz) and knee adduction
moment (Mx), trained on OpenSim Static Optimization + Joint Reaction labels computed from
[AddBiomechanics](https://addbiomechanics.org/) data. They drive the pain-risk constraint in
the antalgic gait synthesis work (Chong et al., ISRR 2026).

**Pipeline**: `.b3d` → OpenSim IK/GRF → Static Optimization → Joint Reaction → labels → surrogate training

## Repository layout

All scripts are run from the repository root (data paths are relative to `./jcf/...`).

| Path | Contents |
|---|---|
| `train_cnn.py`, `test_cnn.py` | Streaming BiCNN (consumes GRF): training and evaluation. Also hosts `load_subject`, used by the MLP pipeline. |
| `train_mlp.py`, `test_mlp.py` | GRF-free per-frame MLP variants, selected with `--input-set` (incl. the contact-flag variant `qinst_pelvisrel_winpos_contact_w20`). |
| `inference.py` | BiCNN inference module (`JCFPredictor`) — `docs/INFERENCE_GUIDE.md` |
| `inference_posture_surrogate.py` | GRF-free MLP inference (`JCFPostureSurrogate`) — `docs/INFERENCE_GUIDE_MLP.md` |
| `infer_trajectory_csv.py` | Runs an exported trajectory CSV through the MLP (frame rotation, q reorder, contact flags, windowing) |
| `test_inference.py` | Parity check of `inference.py` against the `test_cnn.py` pipeline |
| `pipeline/` | Label generation: `.b3d` → OpenSim → SO+JR (`batch_process.py`, `batch_process_parallel.py`, `b3d_to_opensim.py`, `process_jcf.py`, `label_trajectory_csv.py` (optimizer CSV → SO+JR labels, `docs/LABEL_AND_FINETUNE_GUIDE.md`), `compute_com_features.py`, `extract_root_features.py`, `validate_training_data.py`, …) |
| `analysis/` | Diagnostics and plots (`feature_importance.py`, `grad_analysis.py`, `plot_jcf.py`, `plot_f_medial_overlay.py`) |
| `legacy/` | Superseded inference modules |
| `docs/` | Integration guides and project notes |
| `figures/` | Generated figures: `heldout/` (surrogate evaluation), `gt/` (ground-truth analyses), `legacy/` — git-ignored |
| `assets/` | `Rajagopal2015.urdf` |
| `models/` | Curated checkpoints (`mlp-grffree-contact/`, `mlp-grffree-base/`, `bicnn-grf-streaming/`, each with a model card) and raw training runs (`runs/mlp/`, `runs/bicnn/`); `.pt` files git-ignored — see `models/README.md` |
| `jcf/` | Processed trials (`jcf/full_duration/training/{static,walking}/`) — git-ignored |
| `with_arm/` | Raw AddBiomechanics `.b3d` data — git-ignored |
| `paper/` | ISRR manuscript (LaTeX) — git-ignored |

## Setup

```bash
conda create -n jcf python=3.11
conda activate jcf
conda install -c opensim-org opensim
pip install nimblephysics torch numpy pandas matplotlib
```

## Data

Source data comes from [AddBiomechanics](https://addbiomechanics.org/) `.b3d` files (Rajagopal model with arms, ~37 DOFs).

```
with_arm/
├── training/                    # 7 datasets, ~355 subjects
│   ├── Moore2015_Formatted_With_Arm/
│   ├── Tiziana2019_Formatted_With_Arm/
│   ├── Carter2023_Formatted_With_Arm/
│   ├── Falisse2017_Formatted_With_Arm/
│   ├── Fregly2012_Formatted_With_Arm/
│   ├── Hammer2013_Formatted_With_Arm/
│   └── Han2023_Formatted_With_Arm/
└── testing/
    ├── Moore2015_Formatted_With_Arm/
    └── Tiziana2019_Formatted_With_Arm/
```

Processed output with dataset-prefixed names (to avoid macOS case-insensitive collisions):

```
jcf/
├── training/           # e.g. moore_subject3/, tiziana_Subject1/
│   └── <prefix_subject>/
│       ├── scaled_model.osim
│       ├── ik_results.mot
│       ├── grf_data.mot
│       ├── grf_loads.xml
│       ├── metadata.json
│       └── jcf_output/
│           └── BatchJCF_JointReaction_ReactionLoads.sto
└── testing/            # same structure
```

## Getting the processed data

The OpenSim outputs used for training (IK, GRF, Static Optimization, Joint Reaction labels — ~45 GB on
disk) are hosted as a private Hugging Face dataset, `gellert5225/jcf-knee-labels` (revision `v1-isrr2026`).
`jcf/` is git-ignored; reconstruct it with:

```bash
hf auth login                                   # once; needs read access to the dataset
python pipeline/fetch_data.py                   # all labs, both activities
python pipeline/fetch_data.py --labs carter     # or a subset
```

The raw AddBiomechanics `.b3d` inputs (194 GB) are not re-hosted; `MANIFEST_b3d.csv` in the dataset
repo lists every file with its SHA-256 so the identical inputs can be fetched from AddBiomechanics and
re-processed with `pipeline/`.

## Label generation

```bash
conda run -n jcf python pipeline/batch_process_parallel.py            # .b3d → OpenSim → SO+JR, parallel
conda run -n jcf python pipeline/compute_com_features.py              # COM / foot position features for the MLP
```

Key parameters live at the top of `pipeline/batch_process.py` (`B3D_ROOT`, `OUTPUT_ROOT`, `DATASETS_TO_PROCESS`).

## Training

```bash
python train_cnn.py ...                                                        # BiCNN (with GRF)
python train_mlp.py --activity both --input-set qinst_pelvisrel_winpos_contact_w20   # GRF-free MLP + contact flags
```

Training writes checkpoints to `models/runs/mlp/` (MLP) and `models/runs/bicnn/` (BiCNN). Curated, documented
models live in `models/<name>/` with a `MODEL_CARD.md` — see `models/README.md`.

Checkpoints are **not stored in this repo** (`*.pt` is git-ignored). They are hosted on Hugging Face at
`gellert5225/jcf-knee-surrogates` (revision `v1-isrr2026`); see `models/README.md` for a download snippet. Evaluate with `test_cnn.py` / `test_mlp.py`.

## Inference

- BiCNN: `docs/INFERENCE_GUIDE.md`
- GRF-free MLP (posture / gait optimizer): `docs/INFERENCE_GUIDE_MLP.md`

## Technical Notes

- **SWIG crash workaround**: OpenSim's `AnalyzeTool` must be invoked via XML setup file, not Python API directly (causes double-free crash).
- **macOS filesystem**: Output directories use dataset prefix (`moore_`, `tiziana_`) to avoid case-insensitive name collisions.
- **Contact bodies**: 336/348 subjects have feet-only contacts (`calcn_r`, `calcn_l`). 10 have pelvis, 2 have hands (Han2023). The scanner filters to foot contacts only.
- **Short trials**: Tiziana2019 subjects have ~200 Hz single-step force plate trials (0.3–0.5s valid GRF). Adaptive windowing handles these automatically.

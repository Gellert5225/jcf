# Knee Joint Contact Force (JCF) Surrogate Model

Neural network surrogate model for predicting knee joint contact forces from ground reaction forces, trained on musculoskeletal simulation data from [AddBiomechanics](https://addbiomechanics.org/).

## Overview

This project builds a lightweight MLP that maps 6-channel GRF inputs (per-foot force vectors) to 3-axis knee JCF outputs (Fx, Fy, Fz in the tibial frame), enabling real-time JCF estimation for operational space control applications.

**Pipeline**: `.b3d` → OpenSim IK/GRF → Static Optimization → Joint Reaction → Training labels

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

## Pipeline

### 1. Batch Processing (`batch_process.py`)

Scans `.b3d` files for valid walking windows, converts to OpenSim format, and runs Static Optimization + Joint Reaction:

```bash
conda run -n jcf python batch_process.py
```

Key parameters:
- **Window duration**: 2.0s (falls back to shorter windows for single-step trials)
- **GRF cap**: ≤1.7 BW per foot
- **Heel strike detection**: 2% BW off-threshold, 10% BW on-threshold
- **Reserve actuators**: 1 Nm rotational, 10 N translational

Configure `B3D_ROOT`, `OUTPUT_ROOT`, and `DATASETS_TO_PROCESS` at the top of the file.

### 2. Training (`train_jcf.py`)

Trains an MLP surrogate: 6 GRF inputs (normalized by BW) → 3 JCF outputs (Fx, Fy, Fz in BW):

```bash
conda run -n jcf python train_jcf.py
```

- **Architecture**: MLP [64, 64, 32] → 3 outputs
- **Input**: `calcn_r` and `calcn_l` force vectors (vx, vy, vz), z-score normalized
- **Output**: Fx (shear), Fy (compressive), Fz (shear) on `walker_knee_r` in tibial child frame
- **Outlier filter**: Subjects with peak resultant > 10 BW are excluded
- Saves `jcf_model.pt` and `jcf_norm_params.npz`

### 3. Testing (`test_jcf.py`)

Evaluates the trained model on held-out test subjects:

```bash
conda run -n jcf python test_jcf.py
```

Produces per-subject time-series plots and RMSE/R² metrics.

## Supporting Scripts

| Script | Purpose |
|--------|---------|
| `b3d_to_opensim.py` | Converts `.b3d` trial slices to OpenSim files |
| `inspect_b3d.py` | Inspects `.b3d` file metadata and trial structure |
| `extract_jcf.py` | Extracts JCF data from `.sto` result files |
| `plot_jcf.py` | Plots JCF time series |
| `check_knee_moment.py` | Diagnostic for knee moment analysis |

## Technical Notes

- **SWIG crash workaround**: OpenSim's `AnalyzeTool` must be invoked via XML setup file, not Python API directly (causes double-free crash).
- **macOS filesystem**: Output directories use dataset prefix (`moore_`, `tiziana_`) to avoid case-insensitive name collisions.
- **Contact bodies**: 336/348 subjects have feet-only contacts (`calcn_r`, `calcn_l`). 10 have pelvis, 2 have hands (Han2023). The scanner filters to foot contacts only.
- **Short trials**: Tiziana2019 subjects have ~200 Hz single-step force plate trials (0.3–0.5s valid GRF). Adaptive windowing handles these automatically.

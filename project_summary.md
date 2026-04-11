# Project: Knee Joint Contact Force (JCF) Surrogate Model

**Goal**: Train a neural network to predict knee JCF from kinematics/torques/GRF, using AddBiomechanics .b3d data processed through OpenSim's Static Optimization + Joint Reaction pipeline. For an ISRR paper.

## Environment
- **OS**: Linux (WSL)
- **Conda env**: `jcf` (Python 3.10)
- **Key packages**: opensim 4.5.2, nimblephysics, torch, numpy, pandas, matplotlib
- **Workspace**: `/home/gellert/Developer/medical_dt`

## Data
- **Source**: AddBiomechanics .b3d files (binary format, read via `nimblephysics`)
- **Training data**: `with_arm/training/With_Arm/` — 7 datasets, 355 subjects total:
  - Carter2023, Falisse2017, Fregly2012, Hammer2013, Han2023, Moore2015 (11 subjects), Tiziana2019
- **B3D structure**: Each subject has multiple trials (e.g., Moore subject13 has 96 trials × 2000 frames × 0.01s = 20s each). Contains 3 processing passes: kinematics → dynamics → low-pass filter
- **Model**: Rajagopal-based skeleton with arms, ~37 DOFs, contact bodies = `['calcn_r', 'calcn_l']`

## Pipeline (working end-to-end)

### Step 1: Scan b3d for walking (`batch_process.py: scan_b3d_for_walking()`)
- Reads b3d header directly with nimblephysics (fast, no conversion)
- Uses `getMissingGRF()` to find contiguous valid GRF runs
- Samples vertical GRF to score walking quality (mean near BW, high variation)
- Returns `(trial, start_frame, num_frames, mass_kg)`

### Step 2: Convert b3d slice to OpenSim (`b3d_to_opensim.py`)
- Takes `--trial`, `--start-frame`, `--num-frames`, `--output-name` args
- Exports only the needed slice (seconds instead of minutes)
- Outputs: `scaled_model.osim`, `ik_results.mot`, `grf_data.mot`, `grf_loads.xml`, `metadata.json`
- Uses dynamics pass (last pass) for kinematics and model; filters frames where GRF is missing

### Step 3: Run SO + JR (`batch_process.py: run_jcf()`)
- Adds reserve actuators (CoordinateActuator on all unlocked DOFs: 1 Nm rotational, 10 N translational)
- Writes full XML setup file for AnalyzeTool (MUST use XML-based approach — Python API causes SWIG double-free crash)
- `AnalyzeTool(setup_xml_path)` runs StaticOptimization + JointReaction
- JR configured for `walker_knee_r` joint, `child` body, `child` frame
- Output: `BatchJCF_JointReaction_ReactionLoads.sto` (9 columns: fx,fy,fz,mx,my,mz,px,py,pz)
- Lowpass cutoff = 6 Hz on coordinates

## Validated Result
- **Moore2015 subject13** (64 kg): Trial 71, 1 gait cycle (1.91s) + 0.3s buffer each side
- **Peak knee JCF = 3.30 BW** (expected: 2.5–3.5 BW for normal walking) ✅
- Fy (axial) dominates at -3.27 BW, Fx/Fz small (±0.3 BW)
- Output at `jcf/training/subject13/jcf_output/`

## Key Technical Gotchas
- **SWIG crash**: Never use `AnalyzeTool(model)` with Python API. Always write XML, then `AnalyzeTool(xml_path)`
- **nimblephysics readFrames**: positional args: `readFrames(trial, startFrame, numFramesToRead, ...)`
- **Processing passes**: Use kinematics pass (`processingPasses[0]`) for NN inputs (hasn't seen force data). Use dynamics pass (`processingPasses[-1]`) for labels/OpenSim conversion
- **MissingGRF filtering**: Frames with `MissingGRFReason != notMissingGRF` have unreliable `tau` and `acc`
- **.sto parsing**: `skiprows=11` for JointReaction output files
- **Reserve actuators**: Required to prevent SO failure. Contribution is ~1.7% of total force

## NN Architecture Plan
- **Inputs** (from b3d kinematics pass directly): joint positions (37), velocities (37), accelerations (37), GRF (6), joint torques from dynamics pass
- **Labels** (from OpenSim): knee JCF resultant (or 3-component Fx/Fy/Fz)
- **Reference**: [InferBiomechanics repo](https://github.com/keenon/InferBiomechanics) — uses feedforward MLP with windowed frames, PyTorch DataLoader with multi-worker loading, `SubjectOnDisk` pickle workaround via `__getstate__`/`__setstate__`
- Draft training script exists at `train_jcf.py` (needs updating for full input features)

## File Inventory

| File | Status | Purpose |
|---|---|---|
| `batch_process.py` | Working | Scan b3d → convert slice → SO+JR. Currently filtered to Moore2015 only (`DATASETS_TO_PROCESS`) |
| `b3d_to_opensim.py` | Working | Convert b3d (or slice) to .osim/.mot/grf files |
| `JCF_singleFile.py` | Working | Single-subject SO+JR (original script, points to `./jcf/subject10`) |
| `plot_jcf.py` | Working | Two-subplot JCF plot (components + resultant with expected range) |
| `train_jcf.py` | Draft | PyTorch MLP, needs update: currently only uses GRF as input |
| `check_knee_moment.py` | Working | Extract knee tau from b3d for validation |
| `inspect_b3d.py` | Working | General b3d file inspector |
| `extract_jcf.py` | Exists | Feature extractor |

## Next Steps
1. Batch process all 11 Moore2015 subjects (or expand to all 355)
2. Update `train_jcf.py` with full inputs: pos + vel + acc + tau + GRF from b3d, JCF labels from .sto
3. Implement PyTorch Dataset that loads inputs from b3d directly + labels from processed .sto files
4. Train and validate the surrogate model

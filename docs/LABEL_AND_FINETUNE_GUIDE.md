# Labelling optimizer trajectories with OpenSim and fine-tuning the surrogate

Two workflows added on 2026-09-23 after the ground-truth check of William's Sep 23 dynamic-stage export
(`docs/INFERENCE_GUIDE_MLP.md`, "Known issue"): the GRF-free contact surrogate under-predicts F_medial on
optimizer output whose knees stay locked (≈0–5° knee flexion through stance and swing, against 15–60° in the
training gait). Real walking never looks like that, so the surrogate has never seen it.

1. **Direct evaluation (no surrogate).** Run a dynamic-stage trajectory through Static Optimization + JointReaction
   and read F_medial off the result. This is the reference the surrogate is trained to imitate, so it is the number
   to trust for a *finished* trajectory. It cannot go inside the optimizer loop (no gradients, minutes per trajectory).
2. **Fine-tuning.** Use those SO+JR labels as extra training data so the surrogate learns the optimizer's own gait
   family and can be used inside the kinematic stage again.

Both start from the same script.

## 1. `pipeline/label_trajectory_csv.py` — CSV → SO+JR labels

Input: the optimizer's CSV export (`time`, `q0..q36` in Rajagopal URDF order, `com_position_*`, `pelvis_position_*`,
`right/left_foot_position_*`, `right/left_foot_wrench_0..5`), robot frame (x forward, y left, z up).
Run inside the OpenSim conda environment:

```bash
conda run -n jcf python pipeline/label_trajectory_csv.py \
    --csv robot_q_log.csv --mass 75 --height 1.70 \
    --out jcf/external/william_2026-09-23 --split-at 3.3 --plot
```

What it does:

- rotates everything to the OpenSim frame (`(x, y, z) → (x, z, −y)`) and reorders the first six q from
  `[tx, ty, tz, tilt, list, rot]` to `[tilt, list, rot, tx, ty, tz]`;
- writes a training-format trial folder: `ik_results.mot`, `grf_data.mot` (foot wrench as force + point + torque),
  `grf_loads.xml`, `scaled_model.osim`, `metadata.json`, `com_features.npz`;
- picks the training subject whose mass/height are nearest as the scaled model unless `--model` is given
  (**ask for the optimizer's own `.osim` export when possible — the borrowed skeleton is the largest single
  source of label error**);
- runs `process_jcf.run_jcf` (SO + JointReaction), then cleans the labels exactly like the training pipeline:
  frames with |M_x| > 0.15 BW·H or resultant > 6 BW are linearly interpolated (`label_frame_mask.npy` records them,
  `*_raw.csv` keeps the originals);
- writes `labels.csv` (time, phase from the wrench contact threshold, F_y, M_x, F_medial in BW, clean flag),
  `label_quality.json` (corrupted frames, SO saturation fraction, right-stance mean/peak) and, with `--plot`,
  `labels.png`;
- with `--split-at T…`, also writes `<out>_seg1`, `<out>_seg2`, … sliced at those times, so one trajectory can be
  fine-tuned on and tested on disjoint stances.

Read `label_quality.json` before using a folder. Saturation fractions above ~0.4 mean the borrowed model's muscles
could not produce the trajectory's accelerations, and the labels there are lower bounds.

## 2. Fine-tuning with `train_mlp.py`

New options (all optional, default behaviour unchanged):

| flag | meaning |
|---|---|
| `--init-from CKPT` | start from this checkpoint's weights **and input normalization** (the input set must match) |
| `--extra-dirs D…` | labelled trial folders added to training as their own sampling bucket; not subject-split, not flat-filtered |
| `--extra-weight W` | sampling weight of that bucket relative to each main bucket (static = 1, walking = 1). With `--activity both`, the fraction of batches drawn from the extra trials is `W / (2 + W)` |
| `--extra-val-dirs D…` | labelled folders reported as a separate `ext-val` loss each epoch (checkpoint selection still uses the main val loss, so forgetting shows up there and gain shows up in `ext-val`) |
| `--lr` | learning-rate override; use ~1e-4 for fine-tuning |
| `--tag` | suffix on the checkpoint name so the base run is not overwritten |

Fine-tune the contact model on the labelled optimizer trajectories, holding one segment out:

```bash
python train_mlp.py --activity both --input-set qinst_pelvisrel_winpos_contact_w20 --fmedial-weight 0.5 \
    --init-from models/mlp-grffree-contact/mlp-grffree-contact.pt \
    --extra-dirs jcf/external/*_seg1 --extra-val-dirs jcf/external/*_seg2 \
    --extra-weight 0.25 --lr 1e-4 --epochs 15 --tag ft_sep23
# → models/runs/mlp/best_model_mlp_both_qinst_pelvisrel_winpos_contact_w20_fmW0p5_ft_sep23.pt
```

Then check both sides:

```bash
# on the optimizer trajectory (compare with jcf/external/<traj>/labels.csv)
python infer_trajectory_csv.py --csv robot_q_log.csv --mass 75 --height 1.70 \
    --ckpt models/runs/mlp/best_model_mlp_both_qinst_pelvisrel_winpos_contact_w20_fmW0p5_ft_sep23.pt
```
and the held-out real-gait evaluation used for the model cards (`figures/heldout/`), which must not degrade.

Expect one trajectory (~600 frames, three right stances) to be a smoke test only. To actually cover the
optimizer's gait family the extra bucket needs on the order of 20–30 trajectories from both stages
(kinematic and dynamic) with their wrenches, ideally with the optimizer's own scaled model.

## Provenance

Every fine-tuned checkpoint stores `init_from`, `extra_dirs`, `extra_weight`, `lr` and `tag`, so
`torch.load(ckpt)["extra_dirs"]` tells you what it was tuned on.

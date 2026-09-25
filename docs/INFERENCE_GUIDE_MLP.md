# Inference Guide — GRF-free Knee JCF Surrogate (MLP, contact-flag variant)

This is the model for the **posture / gait optimizer**: it needs no ground-reaction forces, only
joint angles, the geometric pose of the COM and feet relative to the pelvis, and (new) binary
foot-contact flags. For the streaming BiCNN that consumes measured GRF, see `INFERENCE_GUIDE.md`.

## Files needed

| File | Purpose |
|---|---|
| `inference_posture_surrogate.py` | Self-contained module: model class, feature construction, `JCFPostureSurrogate` |
| `mlp-grffree-contact.pt` | **Recommended** checkpoint (in `models/mlp-grffree-contact/`; 238-channel input, contact flags). Run file: `runs/mlp/mlp-grffree-contact.pt` |

Dependencies: `numpy`, `torch`.

Get the checkpoint from Hugging Face (private during review — `hf auth login` first):

```python
from huggingface_hub import hf_hub_download
ckpt = hf_hub_download("gellert5225/jcf-knee-surrogates", "mlp-grffree-contact/mlp-grffree-contact.pt", revision="v1-isrr2026")
```
Runs that are not part of the tagged release live under `runs/mlp/` on `main` (no `revision`):

```python
ckpt = hf_hub_download("gellert5225/jcf-knee-surrogates",
                       "runs/mlp/best_model_mlp_both_qinst_pelvisrel_winpos_contact_w20_e90_2.pt")
```
The same module loads every checkpoint in the repo and adapts automatically from the `input_set` tag stored in
the file; a contact-flag checkpoint requires the two `*_contact_window` arguments, a 198-channel one rejects them.

## Which checkpoint

Held-out numbers are on 323 walking trials from 18 subjects never seen in training (1,315
late-stance → double-support entries; static numbers on 119 quasi-static trials). "Spike recovered" = mean predicted / mean ground-truth
|F_medial| at those entries. Validation losses are not comparable across rows (different loss
terms), so they are omitted.

| Checkpoint | Input | Held-out F_medial MAE | Spike recovered | Notes |
|---|---|---|---|---|
| `mlp-grffree-contact.pt` | 238 ch: + windowed R/L contact flags | **0.195 BW** | **91 %** | Most accurate on real gait; better than `mlp-grffree-base.pt` on 264 of 323 trials, M_x error −31 %. See *Known issue* below. |
| `mlp-grffree-base.pt` | 198 ch, plain MSE | 0.268 BW | 81 % | The model used for the paper's optimization results. No stance information. |
| `runs/mlp/…_contact_w20.pt` (contact-MSE) | 238 ch, plain MSE, all 6 labs | 0.252 BW | 80 % | Same inputs as `mlp-grffree-contact.pt`, no F_medial loss term. Static held-out 0.286 BW. On the Sep 23 locked-knee motion with pelvis drift removed: 0.93 / 0.91 BW in the two right stances (truth 1.76 / 1.17), never crosses 1.4 BW. |
| `runs/mlp/…_both_…_contact_w20_e90_2.pt` | 238 ch, plain MSE, `--exclude han fregly` (paper's 4-lab pool) | 0.261 BW | 86 % | Static held-out 0.298 BW. On the locked-knee motion with drift removed: 1.37 / 1.38 BW, crosses 1.4 BW in both stances, MAE 0.46 — closest of any checkpoint there, but through a moment overshoot (peaks 15–30 % high). No gain on real gait over the base. Experimental. |
| `runs/mlp/…_both_…_contact_w20_fmW0p5_e100_2.pt` | 238 ch, F_medial loss 0.5, `--exclude han fregly` | 0.225 BW | 92 % | Static held-out 0.229 BW. On the locked-knee motion: 0.58 / 0.61 BW with Mx ≈ 0, never crosses 1.4 BW — same behaviour as the curated fmW model, no advantage over it. |
| `runs/mlp/…_walking_…_contact_w20_e90_2.pt` | 238 ch, plain MSE, **walking trials only** | 0.277 BW | 88 % | Static held-out 0.422 BW (never saw static — do not use for quasi-static queries). On the locked-knee motion with drift removed: 1.63 / 1.61 BW, crosses in both stances, MAE 0.64, peaks 30–70 % high. Specialist for gait windows. |
| `runs/mlp/incomplete/…_w20_fmW0p5.pt` | 198 ch | 0.311 BW | 61 % | The file on disk is a 2-epoch checkpoint — do not use. |
| `runs/mlp/…_winctr_w20_fmW0p5.pt` | 198 ch, per-window pelvis centering | — | — | Not recommended (see *Pelvis centering*). |

Evidence for the held-out numbers: `figures/heldout/contact_vs_baseline_heldout_summary.png` (aggregate), `figures/heldout/contact_vs_baseline_heldout_carterP019.png` (one subject), and the table twin `figures/heldout/contact_vs_baseline_heldout_summary.csv`.

**Known issue (diagnosed Sep 23, 2026):** two separate effects were found on a synthetic optimizer trajectory. First, trajectory-mean pelvis centering on overground motion drives all checkpoints out of distribution (see *Pelvis centering* below; removing the pelvis drift (running mean or zeroing) removes the base model's spurious peak and lifts the contact-MSE run's correlation with ground truth from 0.0 to 0.88). Second, on that trajectory the knees stay locked (≈5° total knee range over the whole cycle, both legs, vs 15–60° in mocap; `figures/gt/trajectory_sep23_knee_kinematics.png`), OpenSim ground truth gives right-stance F_medial peaks of 1.8–2.3 BW while this model predicts ~0.5 BW and `mlp-grffree-base` is unreliable (1.86 in one stance, 0.44 in the other). GRF-free surrogates cannot see the load on a straight, upright stance leg; the training data has no such postures at load. On natural gait this model remains the most accurate. Two remedies are implemented in `docs/LABEL_AND_FINETUNE_GUIDE.md`: label optimizer trajectories with SO+JR directly (`pipeline/label_trajectory_csv.py`) and fine-tune this model on those labels (`train_mlp.py --init-from … --extra-dirs …`).

## What you need at runtime

| Input | Shape (single / batched) | Units | Notes |
|---|---|---|---|
| `joint_angles` | `(16,)` / `(N,16)` | rad, m | Center-frame snapshot, order below |
| `com_rel_window` | `(20,3)` / `(N,20,3)` | m | COM − pelvis, OpenSim ground frame, 20-frame window centered on the query |
| `calcn_r_rel_window` | `(20,3)` / `(N,20,3)` | m | right foot (calcn_r origin) − pelvis |
| `calcn_l_rel_window` | `(20,3)` / `(N,20,3)` | m | left foot − pelvis |
| `r_contact_window` | `(20,)` / `(N,20)` | 0/1 | right foot in contact, per frame **(contact checkpoint only)** |
| `l_contact_window` | `(20,)` / `(N,20)` | 0/1 | left foot in contact, per frame **(contact checkpoint only)** |
| `mass_kg`, `height_m` | scalars | kg, m | one subject per call |

Window = 20 frames at **100 Hz** (200 ms), frames `k−10 … k+9` for a query at frame `k`.
The model was trained at 100 Hz; resample if your optimizer runs at a different rate.

### Joint order (16)

```
0 pelvis_tilt   1 pelvis_list   2 pelvis_rotation
3 pelvis_tx     4 pelvis_ty     5 pelvis_tz
6 hip_flexion_r 7 hip_adduction_r 8 hip_rotation_r 9 knee_angle_r 10 ankle_angle_r
11 hip_flexion_l 12 hip_adduction_l 13 hip_rotation_l 14 knee_angle_l 15 ankle_angle_l
```

Rotations first, translations second. The Rajagopal **URDF** orders the floating base as
`[tx, ty, tz, tilt, list, rot]` — reorder to `[tilt, list, rot, tx, ty, tz]` before calling.
`subtalar_*` and `mtp_*` are excluded.

### Coordinate frame

All positions and `pelvis_tx/ty/tz` are in the **OpenSim ground frame: x forward, y up, z right**.
If your data is in a robot frame (x forward, **y left, z up**) apply one rotation to every 3-vector
*and* to the `[tx, ty, tz]` slice of `q`:

```python
def to_opensim(v):                      # (..., 3)
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    return np.stack([x, z, -y], axis=-1)
```

Sanity check after conversion: a standing right foot relative to the pelvis should come out
near `(0, −0.95, +0.08)` m — about a metre *below* the pelvis in y, slightly to the right in z.
The most common integration bug is skipping this step; the symptom is medial-force peaks ~3× too low.

### Contact flags

Binary per frame, derived from the vertical foot force:

```python
BW = mass_kg * 9.81
r_contact = (np.abs(F_vertical_right) > 0.05 * BW).astype(np.float32)   # ≈37 N at 75 kg
l_contact = (np.abs(F_vertical_left)  > 0.05 * BW).astype(np.float32)
```

The training labels used the same 5 % BW rule, so keep it. Any contact estimate your optimizer
already has for stance constraints is fine as long as it is expressed as 0/1 per frame. The
flags carry the single-stance ↔ double-stance information that kinematics alone cannot resolve,
which is what lets the model place the push-off medial spike.

## Quick start

```python
import numpy as np
from inference_posture_surrogate import JCFPostureSurrogate

m = JCFPostureSurrogate("mlp-grffree-contact.pt")
# m.include_contact -> True, m.n_features -> 238

out = m.predict_windowed(
    joint_angles       = q16,          # (16,)  center frame, order above, OpenSim frame
    com_rel_window     = com_w,        # (20,3) COM - pelvis over the window
    calcn_r_rel_window = rfoot_w,      # (20,3)
    calcn_l_rel_window = lfoot_w,      # (20,3)
    r_contact_window   = rc_w,         # (20,)  0/1
    l_contact_window   = lc_w,         # (20,)  0/1
    mass_kg=75.0, height_m=1.70,
)
# scalars for a single query, (N,) arrays when batched
out["fy"]        # axial knee force, BW  (negative = compression)
out["mx"]        # knee adduction moment, BW·H  (mx > 0 loads the medial compartment)
out["f_medial"]  # medial-compartment proxy, BW  = alpha*|fy| + mx*H/d   (alpha=0.5, d=0.045 m)
out["fy_n"], out["mx_nm"]   # same in N and N·m
```

Batched: pass `(N,16)`, `(N,20,3)` ×3, `(N,20)` ×2 — one forward pass for a whole horizon.

### Static posture

For a single held posture, `predict_static` replicates the snapshot across the window
(exactly what quiet-stance training samples look like):

```python
out = m.predict_static(q16, com_rel, rfoot_rel, lfoot_rel, mass_kg=75, height_m=1.70,
                       r_contact=1, l_contact=0)      # scalar 0/1 flags; default 1,1
```

### φ (alpha) and d overrides

```python
out = m.predict_windowed(..., alpha=0.6, d_m=0.045)
```

`fy` and `mx` are returned separately, so a φ sensitivity sweep is pure post-processing —
no re-inference needed.

## Pelvis centering (read this before running any overground trajectory)

The model never sees absolute lab position: at training time `pelvis_tx/ty/tz` were zero-centered
**per trial**, and `predict_windowed` reproduces this by subtracting the **batch mean** of `q[:, 3:6]`.
About 90 % of the training frames are treadmill trials (Carter, Moore, Han), where the centered pelvis
translation stays within ±0.1 m. An overground trajectory centered on its own mean feeds the model
pelvis positions of ±0.9 m, which it has never seen, and the output drifts with the walking distance.
On the Sep 23 optimizer trajectory this drift alone moved the base model's F_medial by ~1 BW across the
trajectory and produced a spurious 1.86 BW peak in one stance
(`figures/gt/trajectory_sep23_pelvis_centering_effect.png`, attribution: `pelvis_tx` +0.74 BW, `pelvis_ty` +0.50 BW).

**Rule: remove the walking-distance drift from the pelvis translation, keep the within-stride oscillation.**
Subtract a stride-scale running mean (about 1 s, centred) from `pelvis_tx/ty/tz` along the trajectory and pass
`skip_pelvis_centering=True`:

```python
from scipy.ndimage import uniform_filter1d
q_traj[:, 3:6] -= uniform_filter1d(q_traj[:, 3:6], size=120, axis=0, mode="nearest")   # 120 frames = 1.2 s at 100 Hz
out = m.predict_windowed(q_traj[k], ..., skip_pelvis_centering=True)
```

Measured on the 323-trial walking held-out set and the 119-trial static set (F_medial MAE, BW):

| treatment | base | contact-fmW0.5 | contact-MSE | static (all models) |
|---|---|---|---|---|
| per-trial centering (training-time, treadmill) | 0.268 | 0.195 | 0.252 | unchanged |
| running mean 1.2 s | 0.270 | 0.213 | 0.258 | unchanged |
| pelvis zeroed | 0.295 | 0.235 | 0.279 | +0.01 |

- The running mean removes the ±0.9 m drift on the Sep 23 trajectory (fed range becomes 0.09 m, inside the
  treadmill training range) and gives the same predictions there as zeroing, while costing 0–9 % on real gait
  instead of 10–20 %. The within-stride pelvis oscillation is a real signal the models learned.
- Zeroing (`q_in[3:6] = 0`) is the simplest fallback and gives a constant input (block-diagonal Jacobian). Inside an
  optimizer, compute the running mean once from the *reference* trajectory and freeze it per knot to get the same
  sparsity with the running-mean treatment.
- Do **not** subtract a single trajectory mean for overground motion. Earlier versions of this guide recommended it;
  it is what produces the artefact. For treadmill-like motion (pelvis range < 0.1 m) all treatments agree.
- The `…_winctr_…` *checkpoint* is a different thing: a retrained model that validated worse. The advice above is about
  the input treatment for the standard checkpoints, not about that file.

## Getting gradients (Ipopt / autodiff)

`predict_windowed` / `predict_static` run under `torch.no_grad()` and take NumPy — they are for
evaluation. For backpropagation build the feature tensor yourself and call the network directly.
Feature layout (238):

```
[0:16]     q16                    (pelvis tx/ty/tz drift-removed as above)
[16:236]   window, frame-major: for f in 0..19:
               com_rel[f] (3), rfoot_rel[f] (3), lfoot_rel[f] (3), r_contact[f] (1), l_contact[f] (1)
[236:238]  mass_kg, height_m
```

```python
import torch
x = torch.cat([q16_t, window_t.reshape(-1), torch.tensor([mass, height])])   # (238,), requires_grad upstream
x = (x - m.input_mean) / m.input_std           # stored in the checkpoint, on m.device
fx, fy, fz, mx = m.model(x[None])[0]           # BW, BW, BW, BW·H
f_medial = 0.5 * fy.abs() + mx * height / 0.045
f_medial.backward()
```

The contact flags are constants in the optimization (0/1 from your contact schedule), so they
carry no gradient; everything else flows through normally. Activation is ReLU in this checkpoint
(`m.activation`); if a solver needs C¹ smoothness, retrain with `--activation silu` — the module
loads either.

## Output conventions

| Key | Units | Notes |
|---|---|---|
| `fx`, `fy`, `fz` | BW | tibia frame; `fy` negative = compression, typical stance −0.5 … −2.5 |
| `mx` | BW·H | external knee adduction moment; `> 0` = medial loading during right stance; peaks ~0.03–0.06 |
| `f_medial` | BW | `alpha*|fy| + mx*H/d` |

Note `f_medial` is **signed**: at push-off `mx` reverses and `f_medial` goes negative (≈ −2 BW
in real gait). A constraint on `|f_medial|` will fire on that push-off abduction event; if you
only want medial overload, constrain `max(f_medial, 0)`.

## Validation

`infer_trajectory_csv.py` runs a full exported trajectory CSV through the model (frame rotation,
q reordering, contact flags from foot wrenches, windowing) and plots the result:

```bash
python infer_trajectory_csv.py --csv my_trajectory.csv \
    --ckpt mlp-grffree-contact.pt --mass 75 --height 1.7
```

Built-in smoke test on synthetic inputs:

```bash
python inference_posture_surrogate.py mlp-grffree-contact.pt
```

## Performance

~0.05 M parameters, ~250 KB checkpoint. A single-posture prediction is ~0.1 ms on CPU; batched
horizons are effectively free. Latency is not the bottleneck.

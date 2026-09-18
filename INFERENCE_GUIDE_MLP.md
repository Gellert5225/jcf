# Inference Guide — GRF-free Knee JCF Surrogate (MLP, contact-flag variant)

This is the model for the **posture / gait optimizer**: it needs no ground-reaction forces, only
joint angles, the geometric pose of the COM and feet relative to the pelvis, and (new) binary
foot-contact flags. For the streaming BiCNN that consumes measured GRF, see `INFERENCE_GUIDE.md`.

## Files needed

| File | Purpose |
|---|---|
| `inference_posture_surrogate.py` | Self-contained module: model class, feature construction, `JCFPostureSurrogate` |
| `best_model_mlp_both_qinst_pelvisrel_winpos_contact_w20_fmW0p5.pt` | **Recommended** checkpoint (238-channel input, contact flags) |

Dependencies: `numpy`, `torch`. The same module also loads the two older checkpoints
(`…_winpos_w20_fmW0p5.pt` baseline, `…_winctr_w20_fmW0p5.pt`) and adapts automatically from the
`input_set` tag stored in the file.

## Which checkpoint

| Checkpoint | Input | Held-out val loss | Notes |
|---|---|---|---|
| `…_contact_w20_fmW0p5.pt` | 238 ch: + windowed R/L contact flags | **0.104** | Recovers ~80 % of the late-stance (push-off) medial spike vs ~45 % for the baseline. Use this. |
| `…_winpos_w20_fmW0p5.pt` | 198 ch | 0.164 | Baseline; no stance information |
| `…_winctr_w20_fmW0p5.pt` | 198 ch, per-window pelvis centering | 0.112 | Not recommended (see *Pelvis centering*) |

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

m = JCFPostureSurrogate("best_model_mlp_both_qinst_pelvisrel_winpos_contact_w20_fmW0p5.pt")
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

## Pelvis centering (read this if you use gradients)

The model never sees absolute lab position: at training time `pelvis_tx/ty/tz` were
zero-centered **per trial**. `predict_windowed` reproduces this by subtracting the **batch mean**
of `q[:, 3:6]` inside the call.

- A single-frame call therefore sees `tx = ty = tz = 0`. That is fine for prediction (the model
  saw every trial-relative offset in training) but discards a weak signal.
- Inside an optimizer the batch-mean creates a dense Jacobian (`∂q_in[i]/∂q[j] = δ_ij − 1/N`).
  For a block-diagonal Jacobian, **freeze the constant**: compute the mean once on your initial
  trajectory, subtract it yourself, and pass `skip_pelvis_centering=True`. This is bit-identical
  to the internal centering when the constant equals the batch mean.

```python
pelvis_const = q_traj[:, 3:6].mean(axis=0)          # once
q_in = q16.copy(); q_in[3:6] -= pelvis_const
out = m.predict_windowed(q_in, ..., skip_pelvis_centering=True)
```

Do **not** center per 20-frame window — the `…_winctr_…` checkpoint was trained that way and is
measurably worse.

## Getting gradients (Ipopt / autodiff)

`predict_windowed` / `predict_static` run under `torch.no_grad()` and take NumPy — they are for
evaluation. For backpropagation build the feature tensor yourself and call the network directly.
Feature layout (238):

```
[0:16]     q16                    (pelvis tx/ty/tz already centered as above)
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
    --ckpt best_model_mlp_both_qinst_pelvisrel_winpos_contact_w20_fmW0p5.pt --mass 75 --height 1.7
```

Built-in smoke test on synthetic inputs:

```bash
python inference_posture_surrogate.py best_model_mlp_both_qinst_pelvisrel_winpos_contact_w20_fmW0p5.pt
```

## Performance

~0.05 M parameters, ~250 KB checkpoint. A single-posture prediction is ~0.1 ms on CPU; batched
horizons are effectively free. Latency is not the bottleneck.

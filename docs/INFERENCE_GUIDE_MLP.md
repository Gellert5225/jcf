# Inference Guide — GRF-free Knee JCF Surrogate (MLP, contact-flag variant)

This is the model for the **posture / gait optimizer**: it needs no ground-reaction forces, only
joint angles, the geometric pose of the COM and feet relative to the pelvis, and (new) binary
foot-contact flags. For the streaming BiCNN that consumes measured GRF, see `INFERENCE_GUIDE.md`.

## Start here: the corrected checkpoints (Nov 2026)

Every checkpoint before November 2026, including the one behind the ISRR 2026 optimization results and
`…_fpopt_w20_fmW0p5_e100.pt`, has two problems:

1. **No muscle forces in the labels.** The labelling pipeline ran OpenSim's JointReaction with an empty forces file,
   so it computed muscle forces from the model states instead of using the static-optimization solution. The
   "knee contact force" these models learned is close to the net joint force: typical walking stance peaks about
   1.3 BW, where instrumented knees measure 2.2–3 BW of total force.
2. **The moment sign.** `f_medial` used `+mx` of `walker_knee_r`. In the tibia frame z points to the subject's right,
   which is lateral on the right knee, so the medial moment is `-mx` (*M_x sign* below). With φ = 0.5,
   `0.5|F_y| + mx·H/d` is the **lateral** compartment load.

For the gait optimizer, use the three corrected checkpoints below. They are the same network as
`…_fpopt_w20_fmW0p5_e100.pt` (same 235 inputs, optimizer foot point, contact flags), trained on corrected data with a
plain per-output loss:

| File (`models/runs/mlp/`) | Best epoch | Val loss |
|---|---|---|
| `best_model_mlp_walking_qinst_notrans_pelvisrel_winpos_contact_fpopt_w20_t100_sojrqc_e100_s1.pt` | 8 | 0.189 |
| `best_model_mlp_walking_qinst_notrans_pelvisrel_winpos_contact_fpopt_w20_t100_sojrqc_e100_s2.pt` | 9 | 0.185 |
| `best_model_mlp_walking_qinst_notrans_pelvisrel_winpos_contact_fpopt_w20_t100_sojrqc_e100_s3.pt` | 33 | 0.183 |

s1, s2 and s3 are three training runs of the same model: identical data, split and settings, different random
starting weights and batch order (`--init-seed 1/2/3`). They agree within about 0.3 BW on the optimizer's reference
motion. **Average the three** (below) rather than picking one.

| | `…_fpopt_w20_fmW0p5_e100.pt` (old) | `…_fpopt_w20_t100_sojrqc_e100_s{1,2,3}.pt` |
|---|---|---|
| Labels | JointReaction without muscle forces | JointReaction with the static-optimization muscle forces |
| Training frames | all | only frames passing a label-quality filter (23 %; drops trials with a fixed centre of pressure and frames where the muscle solver failed) |
| Loss | per-output squared error + F_medial term with +mx | per-output squared error only (no F_medial term, so no sign in training) |
| Sign used by `f_medial` | +mx (stored nowhere; module default) | −mx, stored in the checkpoint (`m.mx_medial_sign = -1`) |
| Input timing | positions up to 0.3 s out of sync with the labels; window 80–200 ms depending on the lab | exact, 100 Hz, every window 200 ms |
| Network, inputs, foot point, contact flags | 235 channels, 3×128 MLP, optimizer foot point, no pelvis translation | **same** |

Why the plain per-output loss: on the muscle-force labels it was compared with three variants of the F_medial loss term
(+mx, −mx, sign-free; 3 runs each, all scored with −mx). It had the lowest held-out errors (stance F_medial 0.37 BW,
compression 0.49 BW, against 0.44 / 0.64 for the −mx term), no unstable runs, and the only above-chance score on the
implants' gait modifications (6–7 of 10). `results/thesis/sign_ablation/fig_loss_variants.png` has the comparison;
the earlier `…_sojrqc_fmW0p5_e100_s{1,2,3}.pt` (−mx F_medial term) remain available.

### What changes in your code

1. **Load the new file. Nothing else in the call changes:** the same 16 joint angles, COM and foot windows at the
   optimizer's foot point, contact flags, mass and height, 20 frames at 100 Hz.
   ```python
   m = JCFPostureSurrogate("best_model_mlp_walking_qinst_notrans_pelvisrel_winpos_contact_fpopt_w20_t100_sojrqc_e100_s1.pt")
   m.mx_medial_sign   # -1.0: out["f_medial"] already uses the medial moment
   ```
2. **Do not compute F_medial with a hard-coded `+mx`.** Use `out["f_medial"]`, or `m.mx_medial_sign` in your own
   formula: `f_medial = 0.5*|fy| + m.mx_medial_sign * mx * H / d`. For the right knee that is `0.5*|fy| - mx*H/d`.
   In stance `fy` is negative, so `-fy` equals `|fy|` there and has no kink at zero (better for Ipopt).
3. **Convert the pelvis angles** if your `q` comes from `Rajagopal2015.urdf` / OpenSai. The URDF's three pelvis
   rotation coordinates turn about the URDF's own z (up), x and y axes, not OpenSim's; copying them across
   mis-orients the pelvis by up to 13° on the Sep 23 trajectory. Translations are rotated as before
   (*Coordinate frame*).
   ```python
   from inference_posture_surrogate import urdf_pelvis_to_opensim
   q16[..., 0:3] = urdf_pelvis_to_opensim(q_urdf[..., 3:6])   # URDF q3..q5 -> OpenSim tilt, list, rotation
   ```
   To first order this is tilt = −q5, list = q4, rotation = q3. Check: with the converted angles, the generic URDF
   reproduces the optimizer's exported foot points to millimetres (about 13 cm off with the copied angles).
4. **Use a centred window.** The query frame k needs positions and contact flags for frames k−10 … k+9 (100 ms of
   the future). A trailing window (k−19 … k) shifts every prediction about 100 ms late.
5. **Average the three runs.** All three share the same input normalisation, so build the inputs once:
   ```python
   paths = [f"best_model_mlp_walking_qinst_notrans_pelvisrel_winpos_contact_fpopt_w20_t100_sojrqc_e100_s{k}.pt" for k in (1, 2, 3)]
   ms = [JCFPostureSurrogate(p) for p in paths]
   outs = [mm.predict_windowed(q16, com_w, rfoot_w, lfoot_w, mass_kg=75, height_m=1.70,
                               r_contact_window=rc_w, l_contact_window=lc_w) for mm in ms]
   fy = np.mean([o["fy"] for o in outs], axis=0)
   mx = np.mean([o["mx"] for o in outs], axis=0)
   f_medial = 0.5 * np.abs(fy) - mx * 1.70 / 0.045          # right knee, medial moment = -mx
   ```
   For gradients, sum the three networks' outputs inside your autodiff graph (*Getting gradients*).
6. **Use a relative limit, not 1.4 BW.** These models are on a different scale from the old ones and read high: on the
   Sep 23 and Nov 14 exports they average 2.5–2.7 BW over right single stance (peaks about 3.0–3.2 BW) and exceed
   1.4 BW in every single-stance frame. Limit, for example, the stance peak to a chosen percentage of the reference's
   own peak (sweep −10/20/30 %).
7. **Check your integration** on the Sep 23 export (*Validation*): you should get the numbers in the table there.

### What they can and cannot do

| | Result |
|---|---|
| Held-out people (one per lab, never trained on), against their OpenSim labels | stance F_medial r 0.80, mean error 0.36 BW; compression r 0.83, error 0.47 BW. Worse on labs not in training (Han 0.73, Fregly 0.76 BW) |
| Real knees (Knee Grand Challenge implants), medial force | early-stance peak about right in size (1.78 vs 1.94 BW) but about 80 ms late; push-off peak about 2× too high (3.33 vs 1.52 BW); r 0.18–0.43 within stance |
| Gait modifications on the implants (trunk sway, poles, medial thrust; does the predicted change follow the measured one?) | 6–7 of 10 directions right, r +0.39. The knee adduction moment (KAM) from the ground force gets 7 of 10 (r +0.51 to +0.66), OpenSim with muscle forces 10 of 10 |
| Optimizer exports | reads the Sep 23 and Nov 14 motions almost the same (2.5–2.7 BW in stance) although they differ (knee motion, KAM about 20 % lower on Nov 14) |

So they give a reasonable typical stance load but respond weakly to posture changes, which is what an antalgic
optimization exploits. For results, constraining the KAM (computed from the optimizer's own foot wrench,
`tools/kam_trajectory_csv.py`) is better validated; use these models as a demonstration or alongside it.

**Locked knee.** In the optimizer's exports so far the stance knee stays at 0° flexion (normal walking: 13–21° in
single stance). The models never saw a loaded, fully straight knee, and OpenSim cannot label it reliably either: with
no ligaments in the model, static optimization holds the hyperextension load with the gastrocnemius at full
activation (`pipeline/label_trajectory_csv.py` on the Nov 14 export: 33–43 % of single-stance frames saturated). A
reference with normal stance knee flexion makes both far more trustworthy.

**OpenSim labels for optimizer trajectories:** `pipeline/label_trajectory_csv.py` (fixed Nov 14: pelvis conversion,
JointReaction with the muscle forces in `jcf_output_sojr/`, −mx in `labels.csv`). It needs physically plausible foot
wrenches (no pulling, centre of pressure inside the foot, gradual weight transfer).

## Files needed

| File | Purpose |
|---|---|
| `inference_posture_surrogate.py` | Self-contained module: model class, feature construction, `JCFPostureSurrogate`, `urdf_pelvis_to_opensim`. The Nov 2026 version is needed for the corrected checkpoints and the pelvis conversion |
| `…_fpopt_w20_t100_sojrqc_e100_s{1,2,3}.pt` | **For the gait optimizer** (see *Start here*). ~310 KB each. Not on Hugging Face yet |
| `infer_trajectory_csv.py` | Runs a checkpoint on an exported trajectory CSV (*Validation*); converts the pelvis angles since Nov 2026 |
| `runs/mlp/best_model_mlp_walking_qinst_notrans_pelvisrel_winpos_contact_fpopt_w20_fmW0p5_e100.pt` | Old labels, +mx. Keep only to reproduce the ISRR-era optimizer runs |
| `mlp-grffree-contact.pt` | Old labels, +mx (in `models/mlp-grffree-contact/`; 238-channel input, contact flags, uses pelvis translation) |

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

> **Historical (pre-Nov 2026).** Every row below was trained on the labels without muscle forces and scored with
> `+mx`. The "OpenSim" / "truth" / "GT" numbers quoted for the locked-knee (Sep 23) motion are those same labels, read
> with `+mx`, computed on pelvis angles copied from the URDF; they are not a reference. For the optimizer, use the
> corrected checkpoints in *Start here*.

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
| `runs/mlp/…_walking_notrans_…_contact_w20_fmW0p5_e100_2.pt` | 235 ch: as the walking-only fmW0.5 row below, **without pelvis_tx/ty/tz** | 0.232 BW | 77 % | Static 0.398 BW — gait windows only. Not exploitable through pelvis height (1.3 cm lift: 2.16 → 2.15 BW). On the locked-knee motion: stance means 1.46 / 1.02 (OpenSim 1.76 / 1.16), peaks 2.42 / 2.16 (OpenSim 2.15 / 1.77), 44 % of frames over 1.4 BW. For an optimizer whose foot input is the **calcn origin**; no pelvis centering needed. |
| `runs/mlp/…_walking_notrans_…_contact_fpopt_w20_fmW0p5_e100.pt` (fpopt) | 235 ch: as the row above, foot = the optimizer's point, calcn (0.13, −0.01, 0) in OpenSim axes = (0.13, 0, −0.01) in URDF axes | same as the row above on the newer 6-person test split (0.249 vs 0.247 BW; stance peaks read 0.07 BW lower; `results/fmedial_eval/fpopt_vs_fp13`) | — | Same trials and recipe as the row above; only the foot point differs. On the locked-knee motion: stance means 1.27 / 0.94 (OpenSim 1.76 / 1.16), peaks 1.99 / 1.79, 44 % of frames over 1.4 BW, MAE 0.53 (row above: 0.57). **Recommended inside the gait optimizer when its foot input is the URDF point (0.13, 0, −0.01)** (William's setup, see *Foot point*). |
| `runs/mlp/…_walking_notrans_…_contact_fp13_w20_fmW0p5_e100.pt` (fp13) | 235 ch: foot = calcn (0.13, 0, −0.01) in OpenSim axes, the URDF numbers read in the wrong axes | 0.246 BW on the same split | — | Superseded by fpopt. On the locked-knee motion: stance means 1.71 / 1.45, peaks 2.56 / 2.70, 58 % over 1.4 BW, MAE 0.59. |
| `runs/mlp/…_walking_…_contact_w20_fmW0p5_e100_2.pt` | 238 ch, F_medial loss 0.5, **walking only**, `--exclude han fregly` | 0.216 BW | 85 % | Static held-out 0.394 BW — **gait windows only**. On the locked-knee motion with drift removed: peaks 2.30 / 1.85 (OpenSim 2.15 / 1.77), moment calibrated, force ~30 % low, MAE 0.45 — closest of any checkpoint. Interim recommendation for the gait optimizer until fine-tuned weights exist. |
| `runs/mlp/…_walking_…_contact_w20_e90_2.pt` | 238 ch, plain MSE, **walking trials only** | 0.277 BW | 88 % | Static held-out 0.422 BW (never saw static — do not use for quasi-static queries). On the locked-knee motion with drift removed: 1.63 / 1.61 BW, crosses in both stances, MAE 0.64, peaks 30–70 % high. Specialist for gait windows. |
| `runs/mlp/incomplete/…_w20_fmW0p5.pt` | 198 ch | 0.311 BW | 61 % | The file on disk is a 2-epoch checkpoint — do not use. |
| `runs/mlp/…_winctr_w20_fmW0p5.pt` | 198 ch, per-window pelvis centering | — | — | Not recommended (see *Pelvis centering*). |

**Run-to-run spread on the locked-knee motion.** The September, fpopt and fp13 rows tie on real gait (0.247 / 0.249 /
0.246 BW), yet read the first locked-knee stance as 1.46 / 1.27 / 1.71 BW (OpenSim 1.76) and differ by up to 1 BW at
single frames. The foot point is not the cause: feeding fp13 the exact optimizer point instead of its own changes it by
0.016 BW on average. On postures outside the training data, the training run itself decides the reading, so do not
trust any single checkpoint there (`figures/gt/trajectory_sep23_fpopt_vs_gt.png`). Fine-tuning on SO+JR labels of
optimizer trajectories (*Known issue* below) is the remedy.

Evidence for the held-out numbers: `figures/heldout/contact_vs_baseline_heldout_summary.png` (aggregate), `figures/heldout/contact_vs_baseline_heldout_carterP019.png` (one subject), and the table twin `figures/heldout/contact_vs_baseline_heldout_summary.csv`.

**Known issue (diagnosed Sep 23, 2026; the "OpenSim ground truth" below is superseded, see the note at the top of
this section):** two separate effects were found on a synthetic optimizer trajectory. First, trajectory-mean pelvis centering on overground motion drives all checkpoints out of distribution (see *Pelvis centering* below; removing the pelvis drift (running mean or zeroing) removes the base model's spurious peak and lifts the contact-MSE run's correlation with ground truth from 0.0 to 0.88). Second, on that trajectory the knees stay locked (≈5° total knee range over the whole cycle, both legs, vs 15–60° in mocap; `figures/gt/trajectory_sep23_knee_kinematics.png`), OpenSim ground truth gives right-stance F_medial peaks of 1.8–2.3 BW while this model predicts ~0.5 BW and `mlp-grffree-base` is unreliable (1.86 in one stance, 0.44 in the other). GRF-free surrogates cannot see the load on a straight, upright stance leg; the training data has no such postures at load. On natural gait this model remains the most accurate. Two remedies are implemented in `docs/LABEL_AND_FINETUNE_GUIDE.md`: label optimizer trajectories with SO+JR directly (`pipeline/label_trajectory_csv.py`) and fine-tune this model on those labels (`train_mlp.py --init-from … --extra-dirs …`).

## What you need at runtime

| Input | Shape (single / batched) | Units | Notes |
|---|---|---|---|
| `joint_angles` | `(16,)` / `(N,16)` | rad, m | Center-frame snapshot, order below |
| `com_rel_window` | `(20,3)` / `(N,20,3)` | m | COM − pelvis, OpenSim ground frame, 20-frame window centered on the query |
| `calcn_r_rel_window` | `(20,3)` / `(N,20,3)` | m | right foot point − pelvis: the calcn_r origin, or `m.foot_point` on calcn_r (see *Foot point*) |
| `calcn_l_rel_window` | `(20,3)` / `(N,20,3)` | m | left foot point − pelvis, same point on calcn_l |
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
`[tx, ty, tz, tilt, list, rot]`: reorder to `[tilt, list, rot, tx, ty, tz]`, convert the three rotations with
`urdf_pelvis_to_opensim` (the URDF's coordinates named tilt / list / rot turn about its own z-up, x and y axes; see
*Start here*, step 3) and rotate the translations to OpenSim axes (*Coordinate frame*). Earlier versions of this guide
said to reorder only; that copies the angles and mis-orients the pelvis. `subtalar_*` and `mtp_*` are excluded.

### Coordinate frame

All positions and `pelvis_tx/ty/tz` are in the **OpenSim ground frame: x forward, y up, z right**.
If your data is in a robot frame (x forward, **y left, z up**) apply one rotation to every 3-vector
*and* to the `[tx, ty, tz]` slice of `q`:

```python
def to_opensim(v):                      # (..., 3)
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    return np.stack([x, z, -y], axis=-1)
```

Sanity check after conversion: a standing right foot (calcn origin) relative to the pelvis should come out
near `(0, −0.95, +0.08)` m — about a metre *below* the pelvis in y, slightly to the right in z. For the fpopt
point, x is about 0.13 m further forward.
The most common integration bug is skipping this step; the symptom is medial-force peaks ~3× too low.

### Foot point

Each checkpoint expects the position of one fixed point on each calcn body, stored in the checkpoint and exposed as
`m.foot_point` (metres, calcn frame, OpenSim axes: x forward, y up, z right). Compute it in the OpenSim ground frame
and subtract the pelvis origin:

```python
p_foot = p_calcn + R_calcn @ m.foot_point        # R_calcn: calcn orientation in ground, p_calcn: calcn origin
rfoot_rel = p_foot - p_pelvis                     # OpenSim ground axes, not rotated into the pelvis frame
```

| | OpenSim calcn axes | Rajagopal URDF link axes (Z-up) |
|---|---|---|
| every checkpoint except fpopt and fp13 | (0, 0, 0), calcn origin (= subtalar joint centre, back of the heel) | (0, 0, 0) |
| fpopt checkpoint = foot point in William's Sep 23 optimizer log | (0.13, −0.01, 0) | (0.13, 0, −0.01) |
| fp13 checkpoint (superseded) | (0.13, 0, −0.01) | (0.13, 0.01, 0) |

The URDF re-expresses each OpenSim frame with C = [[1,0,0],[0,0,−1],[0,1,0]]: a point (x, y, z) in OpenSim axes is
(x, −z, y) in URDF link axes, and (x, y, z) in URDF axes is (x, z, −y) in OpenSim axes. fp13 was trained with the
URDF numbers read as OpenSim axes, 1.4 cm from the optimizer's point (1 cm to the side instead of 1 cm below); fpopt
uses the exact point, centred and the same on both feet. Its features come from
`pipeline/compute_com_features.py --foot-point 0.13 -0.01 0`.

Match the point to the checkpoint: feeding a point 13 cm off (the URDF point into a calcn-origin checkpoint, or the
reverse) shifts F_medial by about 0.05 BW on average and up to 0.35 BW at stance peaks on the same motion.

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

m = JCFPostureSurrogate("best_model_mlp_walking_qinst_notrans_pelvisrel_winpos_contact_fpopt_w20_t100_sojrqc_e100_s1.pt")
# m.include_contact -> True, m.n_features -> 235 (no pelvis translation), m.mx_medial_sign -> -1

out = m.predict_windowed(
    joint_angles       = q16,          # (16,)  center frame, order above, OpenSim frame, pelvis angles converted
    com_rel_window     = com_w,        # (20,3) COM - pelvis over the window
    calcn_r_rel_window = rfoot_w,      # (20,3)
    calcn_l_rel_window = lfoot_w,      # (20,3)
    r_contact_window   = rc_w,         # (20,)  0/1
    l_contact_window   = lc_w,         # (20,)  0/1
    mass_kg=75.0, height_m=1.70,
)
# scalars for a single query, (N,) arrays when batched
out["fy"]        # axial knee force, BW  (negative = compression)
out["mx"]        # raw right-knee JR moment about the tibia x axis, BW·H (see *M_x sign* below)
out["f_medial"]  # medial-compartment proxy, BW  = alpha*|fy| + s*mx*H/d   (alpha=0.5, d=0.045 m, s = m.mx_medial_sign)
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

> **`…_notrans_…` checkpoints (input set `qinst_notrans_pelvisrel_winpos_contact_w20`, 235 channels) do not use
> `pelvis_tx/ty/tz` at all.** Pass the same 16 joint angles as for any other checkpoint; the three translations are
> dropped inside the module, have no effect on the output and receive no gradient. Nothing in this section applies
> to them. Pelvis tilt/list/rotation and all COM/foot-relative-to-pelvis windows are unchanged.

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
from inference_posture_surrogate import remove_pelvis_drift
q_traj[:, 3:6] = remove_pelvis_drift(q_traj[:, 3:6], fs_hz=100.0)   # centred 1.2 s running mean, ends extrapolated
out = m.predict_windowed(q_traj[k], ..., skip_pelvis_centering=True)
```

`remove_pelvis_drift` extends the pelvis path at constant velocity (fitted over the first/last 0.5 s) 0.6 s beyond
each end before averaging, so the first and last 0.6 s carry no residual drift. Padding with the edge value or
averaging only the available frames leaves a residual of ~0.15·v or ~0.3·v (v = walking speed) in those frames;
identical results for trajectories that start and stop at rest, but outside the training range for a cut mid-walk.

Measured on the 323-trial walking held-out set and the 119-trial static set (F_medial MAE, BW):

| treatment | base | contact-fmW0.5 | contact-MSE | static (all models) |
|---|---|---|---|---|
| per-trial centering (training-time, treadmill) | 0.268 | 0.195 | 0.252 | unchanged |
| running mean 1.2 s | 0.270 | 0.213 | 0.258 | unchanged |
| pelvis zeroed | 0.295 | 0.235 | 0.279 | +0.01 |

- The running mean removes the ±0.9 m drift on the Sep 23 trajectory (fed range becomes 0.09 m, inside the
  treadmill training range) and gives the same predictions there as zeroing, while costing 0–9 % on real gait
  instead of 10–20 %. The within-stride pelvis oscillation is a real signal the models learned.
- Zeroing (`q_in[3:6] = 0`) is the simplest fallback and gives a constant input (block-diagonal Jacobian).
- **Inside an optimizer, do not back-propagate through `pelvis_tx/ty/tz`.** Feed the drift-removed values of the
  *reference* trajectory as constants (forward pass unchanged, gradient zero). The models are very sensitive to pelvis
  height relative to its local mean (walking fmW0.5 model: 0.27 BW per cm; paper base model with global-mean
  centering: 0.49 BW per cm, against ~0.02 BW per degree of knee or hip motion). In the training data that channel
  tracks gait phase (the within-stride bob), so pushing it changes the model's reading of *where in the step* the
  posture is, not the knee load. On the Sep 23 reference, raising only the pelvis input by 1.3 cm (the lift in the
  paper's optimized trajectory), with no joint change at all, drops the base model's stance-2 peak from 1.86 to 1.23 BW.
  Pelvis rotations and the COM/foot windows are posture-driven and keep their gradients.
- Do **not** subtract a single trajectory mean for overground motion. Earlier versions of this guide recommended it;
  it is what produces the artefact. For treadmill-like motion (pelvis range < 0.1 m) all treatments agree.
- The `…_winctr_…` *checkpoint* is a different thing: a retrained model that validated worse. The advice above is about
  the input treatment for the standard checkpoints, not about that file.

## Getting gradients (Ipopt / autodiff)

`predict_windowed` / `predict_static` run under `torch.no_grad()` and take NumPy — they are for
evaluation. For backpropagation build the feature tensor yourself and call the network directly.
Feature layout (238):

```
[0:16]     q16                    (pelvis tx/ty/tz drift-removed as above; for `…_notrans_…`: [0:13] q13 without them, total 235)
[16:236]   window, frame-major: for f in 0..19:
               com_rel[f] (3), rfoot_rel[f] (3), lfoot_rel[f] (3), r_contact[f] (1), l_contact[f] (1)
[236:238]  mass_kg, height_m
```

```python
import torch
x = torch.cat([q16_t, window_t.reshape(-1), torch.tensor([mass, height])])   # (238,), requires_grad upstream
x = (x - m.input_mean) / m.input_std           # stored in the checkpoint, on m.device
fx, fy, fz, mx = m.model(x[None])[0]           # BW, BW, BW, BW·H
f_medial = 0.5 * fy.abs() + m.mx_medial_sign * mx * height / 0.045   # +1 for pre-Nov-2026 checkpoints
f_medial.backward()
```

For the corrected checkpoints (235 channels: `q13` = `q16` without `pelvis_tx/ty/tz`), averaged over the three runs:

```python
q13_t = torch.cat([q16_t[0:3], q16_t[6:16]])                  # drop pelvis_tx/ty/tz
x = torch.cat([q13_t, window_t.reshape(-1), torch.tensor([mass, height])])    # (235,)
outs = [mm.model(((x.to(mm.device) - mm.input_mean) / mm.input_std)[None])[0] for mm in ms]   # ms: s1, s2, s3
fy = torch.stack([o[1] for o in outs]).mean()
mx = torch.stack([o[3] for o in outs]).mean()
f_medial = 0.5 * fy.abs() + ms[0].mx_medial_sign * mx * height / 0.045        # sign -1 for all three
f_medial.backward()
```

If the pelvis rotations enter your graph in URDF form, the conversion has to be differentiable too: port
`urdf_pelvis_to_opensim` to torch (it is a few matrix products, `asin` and `atan2`), or treat the converted angles
as the decision variables.

The contact flags are constants in the optimization (0/1 from your contact schedule), so they
carry no gradient; everything else flows through normally. Activation is ReLU in this checkpoint
(`m.activation`); if a solver needs C¹ smoothness, retrain with `--activation silu` — the module
loads either.

## Output conventions

| Key | Units | Notes |
|---|---|---|
| `fx`, `fy`, `fz` | BW | tibia frame; `fy` negative = compression, typical stance −0.5 … −2.5 |
| `mx` | BW·H | raw `walker_knee_r` JointReaction moment about the tibia x axis (the training label); physically the medial-positive moment is `-mx` (see *M_x sign*); peaks ~0.03–0.06 |
| `mx_medial` | BW·H | `mx_medial_sign * mx`, the moment term inside `f_medial` |
| `f_medial` | BW | `alpha*|fy| + mx_medial_sign*mx*H/d` |

### M_x sign

The pipeline and the ISRR 2026 formula took `+mx` of `walker_knee_r` as the medial (adduction) moment. It is the
other way round: the medial-positive moment is `-mx` at `walker_knee_r` and `+mx` at `walker_knee_l` (JointReaction on
and in the tibia; verified with Kaneda's condyle forces, the model's muscle geometry and two-point contact algebra in
`analysis/thesis/knee_mx_sign.py`). The networks predict the raw `mx` either way. `f_medial` uses the sign stored in
the checkpoint, `m.mx_medial_sign`:

- checkpoints from before Nov 2026 (every row of the table above) store none and keep **+1**, so their `f_medial` is
  exactly what it was (William's optimizer results stay reproducible), but it reads the lateral moment term;
- `*_sojrqc` checkpoints (`qinst_notrans_pelvisrel_winpos_contact[_fpopt]_w20_t100_sojrqc`, trained on QC-passing
  SO+JR labels) store **−1**: their F_medial loss and `f_medial` use the physical medial moment.

`tools/evaluate_fmedial.py` scores labels and raw outputs with one convention for every checkpoint
(`--mx-convention physical`, the default; `pipeline` reproduces the pre-Nov-2026 numbers).

`f_medial` is **signed**. With the corrected checkpoints it stays positive in practice: the three-run average never
went below +0.23 BW on the Grand Challenge frames or +0.38 BW on the Sep 23 trajectory, so `|f_medial|` and
`max(f_medial, 0)` give the same constraint. With the old checkpoints and `+mx` it goes negative at push-off; a
constraint on `|f_medial|` then fires on that event too.

## Validation

`infer_trajectory_csv.py` runs a full exported trajectory CSV through the model (frame rotation,
q reordering, pelvis-angle conversion, contact flags from foot wrenches, windowing) and plots the result:

```bash
python infer_trajectory_csv.py --csv my_trajectory.csv --mass 75 --height 1.7 \
    --ckpt best_model_mlp_walking_qinst_notrans_pelvisrel_winpos_contact_fpopt_w20_t100_sojrqc_e100_s1.pt \
    --out-csv s1.csv
```

`--copy-pelvis-angles` reproduces the pre-Nov-2026 behaviour (angles copied, wrong). `--out-csv` saves the per-frame
`fy`, `mx`, `mx_medial`, `f_medial`; run it once per checkpoint and average `fy` and `mx` for the three-run value.

**Integration check on the Sep 23 export** (`robot_q_log.csv`, 75 kg, 1.70 m). F_medial over right single stance
(stance 1: 1.98–2.77 s, stance 2: 4.19–4.83 s), BW, for the plain-loss checkpoints `…_t100_sojrqc_e100_s{1,2,3}.pt`
through this script (it reproduces the thesis audit's own feature path to within 1e-5 BW):

| Checkpoint | Stance 1 mean / peak | Stance 2 mean / peak |
|---|---|---|
| s1 | 2.56 / 3.23 | 2.77 / 3.23 |
| s2 | 2.50 / 3.20 | 2.68 / 3.17 |
| s3 | 2.48 / 3.09 | 2.53 / 3.05 |
| average of the three (`fy`, `mx` averaged) | 2.51 / 3.17 | 2.66 / 3.15 |

For the earlier `…_sojrqc_fmW0p5_e100_s{1,2,3}.pt` the averages are 2.22 / 2.69 and 2.32 / 2.67.

If your own pipeline gives different numbers on the same export, the usual causes are copied pelvis angles, the
calcn origin instead of the optimizer's foot point, positions not rotated to OpenSim axes, or `+mx`.

Built-in smoke test on synthetic inputs:

```bash
python inference_posture_surrogate.py mlp-grffree-contact.pt
```

## Performance

~0.05 M parameters, ~250 KB checkpoint. A single-posture prediction is ~0.1 ms on CPU; batched
horizons are effectively free. Latency is not the bottleneck.

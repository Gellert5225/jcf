# Grand Challenge in vivo validation: pilot_AB_native_jr

Run 2026-11-10 13:40 by `analysis/thesis/grand_challenge.py`.
Features: `native` (deploy = exact time alignment, 20 frames at 100 Hz). Scored frames: `jr` window; surrogate input context: `jr`.
F_medial = phi*|F_y| + M_x*H/d with phi = 0.5, d = 45.0 mm unless stated (sweep: phi [0.4, 0.5, 0.6], d [40.0, 45.0, 50.0] mm).
BW = AddBiomechanics metadata mass (JW 64.6 kg, DM 67.0 kg, SC 78.4 kg); the published masses (JW 68, DM 70, SC 78.4 kg) would scale every BW number of JW and DM by about 0.95.

**Every number from a surrogate below is PROVISIONAL (surrogate output; rerun after the *_t100 retraining).**

## Checkpoints

- `fpopt`: `models/runs/mlp/best_model_mlp_walking_qinst_notrans_pelvisrel_winpos_contact_fpopt_w20_fmW0p5_e100.pt` (input set `qinst_notrans_pelvisrel_winpos_contact_fpopt_w20`, exclude ['han', 'fregly'], git 6687dc293dd10c4b2d76ee703c3ff7f55e549812)
- `sept`: `models/runs/mlp/best_model_mlp_walking_qinst_notrans_pelvisrel_winpos_contact_w20_fmW0p5_e100_2.pt` (input set `qinst_notrans_pelvisrel_winpos_contact_w20`, exclude None, git None)

## Alignment

Segment vertical GRF vs GC plates (sum |Fz| at marker rows), searched over all trials of the subject. Accepted: r >= 0.98, amplitude ratio 0.9-1.1, and agreement with the marker/IK identity map (or, without it, sharpness (1 - r_second)/(1 - r_best) >= 5.0).

```
subject    status  segments
     DM duplicate         1
     DM        ok        50
     JW        ok        45
     SC        ok        43
```

```
subject  segments  trials  r_min  r_median  r_second_max  sharpness_min  lag_frames_median
     DM        50      20  1.000     1.000         0.999         14.429                0.0
     JW        45      18  0.998     1.000         0.997          9.171                0.0
     SC        43      29  0.994     0.999         0.999          2.269               -1.0
```

grf_lag_frames = GRF-match row minus the identity (kinematic) row: expected 0 for JW and DM and -1 for SC (AddBiomechanics stores SC's GRF one frame late).

## OpenSim label sources

- `osim_state` = OpenSim JR, state-derived muscle forces (stored labels), from `<segment>/jcf_output/`: 95 of 138 accepted segments (DM 50, JW 45, SC 0).
- `osim_sojr` = OpenSim SO+JR (ours, SO forces applied), from `<segment>/jcf_output_sojr/`: 0 of 138 accepted segments (DM 0, JW 0, SC 0).

`osim_state` are the stored labels, which are also the surrogates' training labels. pipeline/batch_process.py ran StaticOptimization and JointReaction in one AnalyzeTool with an empty `<forces_file />`; OpenSim 4.5.2 documents that JR then computes the actuator forces from the states instead of applying the SO result (SO solves on its own copy of the model). They exist for walker_knee_r only, so SC (left knee) has none.

Stored-label check (`results/thesis/gc/knee_jr/stored_check.csv`, 139 GC folders, no OpenSim): in the 2322 frames where the SO vasti force (vaslat + vasmed + vasint) exceeds 1 BW, the stored right-knee |F| is below that vasti force alone in a median 1.00 of each folder's frames (minimum 0.81); stored knee |F| peaks at 0.81 BW (median; max 1.65) while the SO vasti reach 8.4 BW. A knee label that applied the SO forces would carry the vasti pull.

The same check on every walking folder with SO output (`results/thesis/gc/knee_jr/stored_check_all_walking.csv`, the training and test labels): in 95% of the 8037 folders with vasti > 1 BW frames, the stored knee |F| is below the vasti force in most of them.

`osim_sojr` is not generated yet (JR-only OpenSim rerun, user-run). Decisive one-folder probe, then all GC folders (both knees):

```
conda run -n jcf python analysis/thesis/gc_knee_jr.py --pattern 'fregly_3GC_t18_r00' --workers 1 --summary results/thesis/gc/knee_jr/probe_t18.csv
conda run -n jcf python analysis/thesis/gc_knee_jr.py --workers 20
```

**Until `osim_sojr` exists, no OpenSim row below is SO+JR, and the surrogates' force bias and missing late peak cannot be attributed to the surrogate: they learnt `osim_state`-type labels.**

## Frame metrics (instrumented-limb stance frames, BW; lag 0 = primary result)

Subsets: `all` = every accepted segment, each source on the frames it has; `common` = frames where every OpenSim source the subject has exists and passes the training label filters, identical frames for every source; `statop_overlap` = SC frames inside Kaneda's windows. total_* = |F_y| vs measured axial force.
DM rows at lag 0 are confounded with DM's implant timing (see the DM timing section); state them conditionally and do not pool DM into headline statements on lag 0 alone.

```
        subset subject     source  n_frames  n_segments   mae  rmse   bias      r  total_mae  total_bias  total_r
           all      DM osim_state       878          24 0.680 0.813 -0.664 -0.392      1.371      -1.370   -0.209
           all      DM      fpopt       878          24 0.684 0.772 -0.404  0.423      1.250      -1.223    0.236
           all      DM       sept       878          24 0.571 0.659 -0.286  0.462      1.153      -1.122    0.292
           all      JW osim_state       889          27 0.680 0.785 -0.522 -0.060      0.833      -0.802    0.529
           all      JW      fpopt       889          27 0.496 0.643 -0.198  0.718      0.690      -0.687    0.676
           all      JW       sept       889          27 0.463 0.602 -0.187  0.736      0.665      -0.662    0.696
           all      SC      fpopt      1244          43 0.942 1.115 -0.883  0.400      1.215      -1.209    0.360
           all      SC       sept      1244          43 0.894 1.053 -0.813  0.451      1.154      -1.144    0.411
           all      SC     statop      1021          43 0.279 0.353 -0.027  0.861        NaN         NaN      NaN
        common      DM osim_state       878          24 0.680 0.813 -0.664 -0.392      1.371      -1.370   -0.209
        common      DM      fpopt       878          24 0.684 0.772 -0.404  0.423      1.250      -1.223    0.236
        common      DM       sept       878          24 0.571 0.659 -0.286  0.462      1.153      -1.122    0.292
        common      JW osim_state       889          27 0.680 0.785 -0.522 -0.060      0.833      -0.802    0.529
        common      JW      fpopt       889          27 0.496 0.643 -0.198  0.718      0.690      -0.687    0.676
        common      JW       sept       889          27 0.463 0.602 -0.187  0.736      0.665      -0.662    0.696
        common      SC      fpopt      1244          43 0.942 1.115 -0.883  0.400      1.215      -1.209    0.360
        common      SC       sept      1244          43 0.894 1.053 -0.813  0.451      1.154      -1.144    0.411
        common      SC     statop      1021          43 0.279 0.353 -0.027  0.861        NaN         NaN      NaN
statop_overlap      SC      fpopt      1021          43 0.926 1.112 -0.854  0.356      1.205      -1.198    0.327
statop_overlap      SC       sept      1021          43 0.878 1.048 -0.779  0.414      1.140      -1.128    0.380
statop_overlap      SC     statop      1021          43 0.279 0.353 -0.027  0.861        NaN         NaN      NaN
```

## Stance peaks

Windows of stance: early 5-45 %, late 55-95 %. A window is covered when the segment spans >= 90 % of it, and scored only when the measured maximum is an interior local maximum, at least 3 % of stance from the first and last covered frame of the window (a maximum at the edge is the tail of a peak outside the window, e.g. the first peak's descent). pred_interior_frac = share of the scored windows in which the source has an interior maximum of its own.

```
subject  part  n_covered  n_edge_rejected  n_scored
     DM early          3                0         3
     DM  late          8                5         3
     JW early          5                0         5
     JW  late          4                0         4
     SC early          4                0         4
     SC  late          6                0         6
```

Late-peak statements cannot include DM (3 of 8 late windows scored): most of their late-window maxima sit on the window edge.

```
subject  part             source  n_halves   mae   bias      r  meas_mean  pred_mean  pred_interior_frac
     DM early         osim_state         3 0.985 -0.985 -0.613      1.700      0.715               0.000
     DM early              fpopt         3 0.578  0.578 -0.800      1.700      2.278               1.000
     DM early               sept         3 0.766  0.766 -0.137      1.700      2.466               1.000
     DM  late         osim_state         3 0.624 -0.624 -0.332      1.090      0.467               1.000
     DM  late              fpopt         3 0.628 -0.628 -0.853      1.090      0.462               0.000
     DM  late               sept         3 0.482 -0.482 -0.870      1.090      0.608               0.000
     JW early         osim_state         5 1.105 -1.105  0.078      1.963      0.858               0.000
     JW early              fpopt         5 0.635  0.635 -0.426      1.963      2.598               1.000
     JW early               sept         5 0.530  0.530 -0.529      1.963      2.493               1.000
     JW  late         osim_state         4 0.486 -0.371 -0.630      1.361      0.990               1.000
     JW  late              fpopt         4 0.958 -0.958  0.778      1.361      0.403               0.000
     JW  late               sept         4 0.912 -0.912  0.647      1.361      0.449               0.000
     SC early              fpopt         4 0.216 -0.190  0.464      2.037      1.847               1.000
     SC early               sept         4 0.201  0.033  0.134      2.037      2.070               1.000
     SC early             statop         4 0.190 -0.188  0.521      2.037      1.849               1.000
     SC  late              fpopt         6 1.514 -1.514  0.620      1.777      0.262               0.000
     SC  late               sept         6 1.453 -1.453  0.460      1.777      0.324               0.000
     SC  late             statop         6 0.213  0.160  0.532      1.777      1.937               1.000
     SC early statop_own_windows        31 0.278 -0.060  0.927      2.521      2.461               0.935
     SC  late statop_own_windows        27 0.334  0.244  0.346      1.751      1.995               1.000
```

## DM timing (model-independent) and DM sensitivity at a fixed lag

Time from a heel strike seen on the segment's force plate to the measured first peak (interior maximum of the full GC trial's force in the early window):

```
subject signal  n_stances  median_ms  min_ms  max_ms                                       values_ms
     JW medial          5      140.0   120.0   160.0                             155 160 140 130 120
     JW  total          5      160.0   120.0   190.0                             160 165 190 130 120
     DM medial          3      225.0   225.0   240.0                                     240 225 225
     DM  total          3      240.0   235.0   250.0                                     250 240 235
     SC medial         12      187.5   116.7   241.7 200 242 242 175 133 200 150 142 117 133 208 200
     SC  total         12      154.2   108.3   241.7 200 183 242 183 125 117 125 125 108 125 217 192
```

DM minus JW + SC, rounded to 5 ms (gait-matched = median over DM stances of DM minus the reference stances of the same gait type; pooled = median minus median over all stances): medial force gait-matched +95 ms; medial force pooled +70 ms; total force gait-matched +110 ms; total force pooled +80 ms.

Lag used for the DM sensitivity rows: **+95 ms** (medial force, gait-matched). It uses no model, unlike the lag sweep's best-r lags, which also contain each model's own phase error. The GRF-edge timing points the same way (EXPLORE_NOTES section 4.4, implant minus plate edge at heel strike / toe-off: DM +40 / +55 ms, JW -100 / -37 ms, SC +21 / -8 ms). Gait differences between the patients also move the first peak, so this is a sensitivity check, not a correction.

DM frame metrics (stance frames), lag 0 vs the fixed lag, every source:

```
 lag_ms subset     source   n   mae   bias      r  total_bias
    0.0    all osim_state 878 0.680 -0.664 -0.392      -1.370
    0.0    all      fpopt 878 0.684 -0.404  0.423      -1.223
    0.0    all       sept 878 0.571 -0.286  0.462      -1.122
    0.0 common osim_state 878 0.680 -0.664 -0.392      -1.370
    0.0 common      fpopt 878 0.684 -0.404  0.423      -1.223
    0.0 common       sept 878 0.571 -0.286  0.462      -1.122
   95.0    all osim_state 878 0.585 -0.530 -0.175      -1.215
   95.0    all      fpopt 878 0.475 -0.270  0.747      -1.068
   95.0    all       sept 878 0.371 -0.151  0.777      -0.967
   95.0 common osim_state 878 0.585 -0.530 -0.175      -1.215
   95.0 common      fpopt 878 0.475 -0.270  0.747      -1.068
   95.0 common       sept 878 0.371 -0.151  0.777      -0.967
```

DM stance peaks under the same shift (windows and interior rule as above):

```
 lag_ms  part     source  n  n_scored  n_edge_rejected   mae   bias  pred_interior_frac
    0.0 early osim_state  3       3.0              0.0 0.985 -0.985                 0.0
    0.0 early      fpopt  3       3.0              0.0 0.578  0.578                 1.0
    0.0 early       sept  3       3.0              0.0 0.766  0.766                 1.0
    0.0  late osim_state  3       3.0              5.0 0.624 -0.624                 1.0
    0.0  late      fpopt  3       3.0              5.0 0.628 -0.628                 0.0
    0.0  late       sept  3       3.0              5.0 0.482 -0.482                 0.0
   95.0 early osim_state  3       3.0              0.0 0.985 -0.985                 0.0
   95.0 early      fpopt  3       3.0              0.0 0.577  0.577                 1.0
   95.0 early       sept  3       3.0              0.0 0.766  0.766                 1.0
   95.0  late osim_state  6       6.0              2.0 0.467 -0.467                 1.0
   95.0  late      fpopt  6       6.0              2.0 0.551 -0.551                 0.0
   95.0  late       sept  6       6.0              2.0 0.387 -0.387                 0.0
```

## StatOp_Val (Kaneda et al. 2023) SO+JR medial vs measured, SC, own windows

62 windows; pooled over windows: mean RMSE 0.361 BW, mean r 0.874. Our recomputation; Kaneda's own metric definition is not verified, so do not quote this as the published value.

## Kinematic stance events vs segment force-plate events (used for events outside a segment)

```
subject  n_onsets  onset_bias_ms  onset_iqr_ms  n_offsets  offset_bias_ms  offset_iqr_ms
     JW         5           45.0          15.0         15           -30.0            5.0
     DM         3           45.0           7.5         15           -25.0            2.5
     SC        12            8.3          27.1         18           -16.7            8.3
```

## phi / d sweep: best MAE per subject x source

```
subject     source  phi  d_mm   mae   bias      r
     DM      fpopt  0.6  50.0 0.630 -0.329  0.435
     DM osim_state  0.6  50.0 0.613 -0.585 -0.359
     DM       sept  0.6  50.0 0.521 -0.207  0.474
     JW      fpopt  0.5  50.0 0.460 -0.242  0.720
     JW osim_state  0.6  40.0 0.613 -0.399 -0.040
     JW       sept  0.5  50.0 0.433 -0.231  0.738
     SC      fpopt  0.6  45.0 0.880 -0.792  0.406
     SC       sept  0.6  45.0 0.833 -0.715  0.457
```

## Implied phi, d (least squares, no intercept)

```
subject     source  n_frames   phi  k_moment  d_mm  rmse_fit    r2
     DM osim_state       878 1.240   -83.075   NaN     0.336 0.119
     DM      fpopt       878 1.244   -27.549   NaN     0.275 0.403
     DM       sept       878 1.184   -28.707   NaN     0.261 0.464
     JW osim_state       889 1.668  -103.079   NaN     0.306 0.681
     JW      fpopt       889 1.223   -14.930   NaN     0.338 0.610
     JW       sept       889 1.212   -15.913   NaN     0.321 0.650
     SC      fpopt      1244 1.816   -34.809   NaN     0.602 0.210
     SC       sept      1244 1.747   -36.233   NaN     0.575 0.280
```
d = H / k_moment is undefined (NaN) when the fitted moment coefficient is not positive.

## Lag sweep (positive lag = measured force taken later than the prediction)

lag_ms = lag of minimum MAE, lag_ms_best_r = lag of maximum r. A model's systematic phase error moves both, so a sync offset shows only if every source (incl. SO+JR) agrees.

```
subject     source  lag_ms   mae  mae_at_0  lag_ms_best_r  best_r  r_at_0
     DM      fpopt   150.0 0.427     0.684          130.0   0.760   0.423
     DM osim_state   150.0 0.541     0.680         -150.0   0.036  -0.392
     DM       sept   150.0 0.343     0.571          110.0   0.780   0.462
     JW      fpopt    10.0 0.495     0.496           10.0   0.719   0.718
     JW osim_state   150.0 0.559     0.680           40.0  -0.030  -0.060
     JW       sept    10.0 0.460     0.463           10.0   0.740   0.736
     SC      fpopt   150.0 0.766     0.942           60.0   0.545   0.400
     SC       sept   150.0 0.726     0.894           50.0   0.585   0.451
     SC     statop   -10.0 0.269     0.279          -10.0   0.870   0.861
```

## Caveats

- Segments are mostly half stances (SO windows), so frame metrics weight the covered phases.
- JW medial = posterior-medial + anterior-medial cell (no regression supplied); unverified split.
- DM's implant timing is suspect (DM timing section); lag 0 is the primary result, DM statements are conditional.
- SC is a LEFT knee: surrogate inputs are mirrored; its only OpenSim labels are `osim_sojr` (walker_knee_l, `analysis/thesis/gc_knee_jr.py`, user-run).
- JW and DM sit in the training folders but every checkpoint here was trained with `--exclude han fregly`; SC is in the test split.
- The README regressions were fitted on gait and filtered data; bouncy/crouch exceed that range.

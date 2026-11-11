# Grand Challenge in vivo validation: pilot_AB_sojr

Run 2026-11-10 15:10 by `analysis/thesis/grand_challenge.py`.
Features: `deploy` (deploy = exact time alignment, 20 frames at 100 Hz). Scored frames: `jr` window; surrogate input context: `ik`.
F_medial = phi*|F_y| + M_x*H/d with phi = 0.5, d = 45.0 mm unless stated (sweep: phi [0.4, 0.5, 0.6], d [40.0, 45.0, 50.0] mm).
M_x convention `physical` (medial-positive M_x = -mx at walker_knee_r, +mx at walker_knee_l; the surrogates' M_x = -output). Settled by `analysis/thesis/knee_mx_sign.py` (results/thesis/labels_qc/mx_sign/) and checked against the implant in the M_x sign section below.
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
- `osim_sojr` = OpenSim SO+JR (ours, SO forces applied), from `<segment>/jcf_output_sojr/`: 138 of 138 accepted segments (DM 50, JW 45, SC 43).

`osim_state` are the stored labels, which are also the surrogates' training labels. pipeline/batch_process.py ran StaticOptimization and JointReaction in one AnalyzeTool with an empty `<forces_file />`; OpenSim 4.5.2 documents that JR then computes the actuator forces from the states instead of applying the SO result (SO solves on its own copy of the model). They exist for walker_knee_r only, so SC (left knee) has none.

Stored-label check (`results/thesis/gc/knee_jr/stored_check.csv`, 139 GC folders, no OpenSim): in the 2322 frames where the SO vasti force (vaslat + vasmed + vasint) exceeds 1 BW, the stored right-knee |F| is below that vasti force alone in a median 1.00 of each folder's frames (minimum 0.81); stored knee |F| peaks at 0.81 BW (median; max 1.65) while the SO vasti reach 8.4 BW. A knee label that applied the SO forces would carry the vasti pull.

The same check on every walking folder with SO output (`results/thesis/gc/knee_jr/stored_check_all_walking.csv`, the training and test labels): in 95% of the 8037 folders with vasti > 1 BW frames, the stored knee |F| is below the vasti force in most of them.

`osim_sojr` segments failing the training label filters (JW 7 of 45, DM 6 of 50, SC 27 of 43; reasons in segments.csv `osim_sojr_filter`) are left out of the `common` subset, so `common` is not the same frames as `all` for a subject with SO+JR labels.

## Frame metrics (instrumented-limb stance frames, BW; lag 0 = primary result)

Subsets: `all` = every accepted segment, each source on the frames it has; `common` = frames where every OpenSim source the subject has exists and passes the training label filters, identical frames for every source; `statop_overlap` = SC frames inside Kaneda's windows. total_* = |F_y| vs measured axial force.
DM rows at lag 0 are confounded with DM's implant timing (see the DM timing section); state them conditionally and do not pool DM into headline statements on lag 0 alone.

```
        subset subject     source  n_frames  n_segments   mae  rmse   bias      r  total_mae  total_bias  total_r
           all      DM osim_state       878          24 0.627 0.702 -0.627  0.491      1.371      -1.370   -0.209
           all      DM  osim_sojr       878          24 0.640 0.843  0.603  0.449      1.007       0.899    0.406
           all      DM      fpopt       878          24 0.745 0.905 -0.736 -0.134      1.259      -1.234    0.194
           all      DM       sept       878          24 0.755 0.888 -0.753 -0.127      1.164      -1.127    0.250
           all      JW osim_state       889          27 0.872 0.970 -0.871  0.821      0.833      -0.802    0.529
           all      JW  osim_sojr       889          27 0.649 0.958  0.617  0.644      1.275       1.250    0.718
           all      JW      fpopt       889          27 1.103 1.340 -1.070 -0.536      0.705      -0.704    0.678
           all      JW       sept       889          27 1.086 1.302 -1.055 -0.515      0.664      -0.658    0.677
           all      SC  osim_sojr      1244          43 0.661 1.018  0.502  0.563      1.249       1.183    0.679
           all      SC      fpopt      1244          43 1.243 1.454 -1.235 -0.254      1.201      -1.197    0.391
           all      SC       sept      1244          43 1.250 1.462 -1.242 -0.286      1.135      -1.125    0.432
           all      SC     statop      1021          43 0.279 0.353 -0.027  0.861        NaN         NaN      NaN
        common      DM osim_state       593          18 0.565 0.624 -0.565  0.549      1.220      -1.219   -0.059
        common      DM  osim_sojr       593          18 0.622 0.817  0.608  0.392      0.875       0.768    0.170
        common      DM      fpopt       593          18 0.683 0.822 -0.680 -0.075      1.151      -1.117    0.102
        common      DM       sept       593          18 0.690 0.801 -0.689 -0.034      1.083      -1.037    0.139
        common      JW osim_state       540          20 0.818 0.932 -0.816  0.835      0.790      -0.744    0.422
        common      JW  osim_sojr       540          20 0.538 0.747  0.529  0.757      1.230       1.220    0.788
        common      JW      fpopt       540          20 1.059 1.315 -1.014 -0.619      0.575      -0.573    0.808
        common      JW       sept       540          20 1.044 1.281 -1.001 -0.604      0.533      -0.527    0.812
        common      SC  osim_sojr       308          16 0.475 0.606  0.271  0.782      0.895       0.809    0.808
        common      SC      fpopt       308          16 0.963 1.222 -0.939 -0.067      1.157      -1.157    0.546
        common      SC       sept       308          16 0.969 1.233 -0.945 -0.107      1.098      -1.095    0.573
        common      SC     statop       268          16 0.315 0.416  0.035  0.819        NaN         NaN      NaN
statop_overlap      SC  osim_sojr      1021          43 0.610 0.915  0.460  0.644      1.179       1.106    0.715
statop_overlap      SC      fpopt      1021          43 1.231 1.432 -1.221 -0.175      1.187      -1.182    0.354
statop_overlap      SC       sept      1021          43 1.237 1.439 -1.227 -0.206      1.119      -1.107    0.398
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
     DM early         osim_state         3 1.138 -1.138  0.874      1.700      0.562               1.000
     DM early          osim_sojr         3 1.354  1.354  0.648      1.700      3.054               1.000
     DM early              fpopt         3 1.208 -1.208  0.966      1.700      0.492               0.000
     DM early               sept         3 1.318 -1.318  0.941      1.700      0.382               0.333
     DM  late         osim_state         3 0.563 -0.563  0.213      1.090      0.527               1.000
     DM  late          osim_sojr         3 1.156  1.156  0.649      1.090      2.246               1.000
     DM  late              fpopt         3 0.466 -0.466 -0.213      1.090      0.624               1.000
     DM  late               sept         3 0.435 -0.435 -0.963      1.090      0.655               0.667
     JW early         osim_state         5 1.508 -1.508 -0.439      1.963      0.455               1.000
     JW early          osim_sojr         5 0.576  0.576  0.021      1.963      2.538               1.000
     JW early              fpopt         5 1.577 -1.577  0.120      1.963      0.386               0.000
     JW early               sept         5 1.617 -1.617 -0.100      1.963      0.346               0.000
     JW  late         osim_state         4 0.940 -0.940  0.880      1.361      0.421               0.750
     JW  late          osim_sojr         4 2.925  2.925  0.977      1.361      4.286               1.000
     JW  late              fpopt         4 0.633 -0.633 -0.640      1.361      0.727               1.000
     JW  late               sept         4 0.719 -0.719 -0.306      1.361      0.642               1.000
     SC early          osim_sojr         4 0.653  0.653  0.793      2.037      2.689               1.000
     SC early              fpopt         4 1.624 -1.624  0.202      2.037      0.413               0.000
     SC early               sept         4 1.647 -1.647  0.353      2.037      0.389               0.000
     SC early             statop         4 0.190 -0.188  0.521      2.037      1.849               1.000
     SC  late          osim_sojr         6 1.638  1.638  0.471      1.777      3.415               0.500
     SC  late              fpopt         6 1.221 -1.221 -0.631      1.777      0.555               1.000
     SC  late               sept         6 1.179 -1.179 -0.361      1.777      0.597               1.000
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
    0.0    all osim_state 878 0.627 -0.627  0.491      -1.370
    0.0    all  osim_sojr 878 0.640  0.603  0.449       0.899
    0.0    all      fpopt 878 0.745 -0.736 -0.134      -1.234
    0.0    all       sept 878 0.755 -0.753 -0.127      -1.127
    0.0 common osim_state 593 0.565 -0.565  0.549      -1.219
    0.0 common  osim_sojr 593 0.622  0.608  0.392       0.768
    0.0 common      fpopt 593 0.683 -0.680 -0.075      -1.117
    0.0 common       sept 593 0.690 -0.689 -0.034      -1.037
   95.0    all osim_state 878 0.523 -0.493  0.621      -1.215
   95.0    all  osim_sojr 878 0.769  0.737  0.722       1.054
   95.0    all      fpopt 878 0.662 -0.602 -0.521      -1.079
   95.0    all       sept 878 0.664 -0.619 -0.509      -0.972
   95.0 common osim_state 593 0.454 -0.416  0.666      -1.041
   95.0 common  osim_sojr 593 0.766  0.756  0.714       0.946
   95.0 common      fpopt 593 0.582 -0.531 -0.373      -0.938
   95.0 common       sept 593 0.587 -0.540 -0.340      -0.859
```

DM stance peaks under the same shift (windows and interior rule as above):

```
 lag_ms  part     source  n  n_scored  n_edge_rejected   mae   bias  pred_interior_frac
    0.0 early osim_state  3       3.0              0.0 1.138 -1.138               1.000
    0.0 early  osim_sojr  3       3.0              0.0 1.354  1.354               1.000
    0.0 early      fpopt  3       3.0              0.0 1.208 -1.208               0.000
    0.0 early       sept  3       3.0              0.0 1.318 -1.318               0.333
    0.0  late osim_state  3       3.0              5.0 0.563 -0.563               1.000
    0.0  late  osim_sojr  3       3.0              5.0 1.156  1.156               1.000
    0.0  late      fpopt  3       3.0              5.0 0.466 -0.466               1.000
    0.0  late       sept  3       3.0              5.0 0.435 -0.435               0.667
   95.0 early osim_state  3       3.0              0.0 1.139 -1.139               1.000
   95.0 early  osim_sojr  3       3.0              0.0 1.353  1.353               1.000
   95.0 early      fpopt  3       3.0              0.0 1.209 -1.209               0.000
   95.0 early       sept  3       3.0              0.0 1.318 -1.318               0.333
   95.0  late osim_state  6       6.0              2.0 0.457 -0.457               1.000
   95.0  late  osim_sojr  6       6.0              2.0 1.063  1.063               1.000
   95.0  late      fpopt  6       6.0              2.0 0.405 -0.322               1.000
   95.0  late       sept  6       6.0              2.0 0.376 -0.334               0.833
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
     DM      fpopt  0.6  50.0 0.641 -0.629 -0.036
     DM  osim_sojr  0.4  50.0 0.422  0.287  0.445
     DM osim_state  0.6  40.0 0.548 -0.547  0.479
     DM       sept  0.6  50.0 0.633 -0.628  0.021
     JW      fpopt  0.6  50.0 0.954 -0.917 -0.437
     JW  osim_sojr  0.4  50.0 0.397  0.282  0.635
     JW osim_state  0.6  50.0 0.762 -0.752  0.822
     JW       sept  0.6  50.0 0.934 -0.896 -0.386
     SC      fpopt  0.6  50.0 1.134 -1.125 -0.138
     SC  osim_sojr  0.4  50.0 0.519  0.136  0.549
     SC       sept  0.6  50.0 1.129 -1.120 -0.140
```

## Implied phi, d (least squares, no intercept)

```
subject     source  n_frames   phi  k_moment    d_mm  rmse_fit     r2
     DM osim_state       878 1.240    83.075  20.704     0.336  0.119
     DM  osim_sojr       878 0.301    11.051 155.641     0.390 -0.146
     DM      fpopt       878 1.259    28.480  60.393     0.275  0.403
     DM       sept       878 1.202    31.767  54.144     0.267  0.439
     JW osim_state       889 1.668   103.079  16.104     0.306  0.681
     JW  osim_sojr       889 0.381    -0.426     NaN     0.403  0.453
     JW      fpopt       889 1.267    17.105  97.049     0.320  0.652
     JW       sept       889 1.232    17.983  92.310     0.318  0.656
     SC  osim_sojr      1244 0.431    -3.252     NaN     0.570  0.322
     SC      fpopt      1244 1.789    31.654  52.757     0.590  0.242
     SC       sept      1244 1.709    32.978  50.640     0.567  0.299
```
d = H / k_moment is undefined (NaN) when the fitted moment coefficient is not positive.

## Lag sweep (positive lag = measured force taken later than the prediction)

lag_ms = lag of minimum MAE, lag_ms_best_r = lag of maximum r. A model's systematic phase error moves both, so a sync offset shows only if every source (incl. SO+JR) agrees.

```
subject     source  lag_ms   mae  mae_at_0  lag_ms_best_r  best_r  r_at_0
     DM      fpopt   150.0 0.626     0.745         -150.0   0.715  -0.134
     DM  osim_sojr   -10.0 0.640     0.640           80.0   0.735   0.449
     DM osim_state   150.0 0.477     0.627           60.0   0.670   0.491
     DM       sept   150.0 0.617     0.755         -150.0   0.686  -0.127
     JW      fpopt   150.0 0.942     1.103         -150.0   0.434  -0.536
     JW  osim_sojr   -20.0 0.630     0.649            0.0   0.644   0.644
     JW osim_state   150.0 0.681     0.872          -10.0   0.829   0.821
     JW       sept   150.0 0.914     1.086         -150.0   0.457  -0.515
     SC      fpopt   150.0 1.106     1.243         -150.0   0.552  -0.254
     SC  osim_sojr   -10.0 0.655     0.661          -10.0   0.566   0.563
     SC       sept   150.0 1.112     1.250         -150.0   0.497  -0.286
     SC     statop   -10.0 0.269     0.279          -10.0   0.870   0.861
```

## M_x sign against the implant (stance frames)

r_mx_vs_meas_med_minus_lat > 0 means the source's medial-positive M_x rises when the implant's medial - lateral force rises. medial_moment_frac = share of frames with a medial moment; medial_share = mean(F_medial) / mean(|F_y|) against the implant's mean(medial) / mean(total).

```
subject     source  n_frames  r_mx_vs_meas_med_minus_lat  medial_moment_frac  moment_term_mean_bw  medial_share  meas_medial_share
     DM osim_state       878                       0.410               0.544                0.019         0.524              0.482
     DM  osim_sojr       878                       0.413               0.642                0.114         0.537              0.482
     DM      fpopt       878                      -0.445               0.552               -0.159         0.326              0.482
     DM       sept       878                      -0.437               0.423               -0.228         0.276              0.482
     JW osim_state       889                       0.482               0.083               -0.175         0.328              0.663
     JW  osim_sojr       889                       0.287               0.730                0.288         0.594              0.663
     JW      fpopt       889                      -0.720               0.336               -0.424         0.119              0.663
     JW       sept       889                      -0.719               0.305               -0.431         0.127              0.663
     SC  osim_sojr      1244                       0.321               0.766                0.362         0.610              0.713
     SC      fpopt      1244                      -0.348               0.403               -0.186         0.299              0.713
     SC       sept      1244                      -0.417               0.342               -0.228         0.271              0.713
```

## Caveats

- Segments are mostly half stances (SO windows), so frame metrics weight the covered phases.
- JW medial = posterior-medial + anterior-medial cell (no regression supplied); unverified split.
- DM's implant timing is suspect (DM timing section); lag 0 is the primary result, DM statements are conditional.
- SC is a LEFT knee: surrogate inputs are mirrored; its only OpenSim labels are `osim_sojr` (walker_knee_l, `analysis/thesis/gc_knee_jr.py`, user-run).
- JW and DM sit in the training folders but every checkpoint here was trained with `--exclude han fregly`; SC is in the test split.
- The README regressions were fitted on gait and filtered data; bouncy/crouch exceed that range.

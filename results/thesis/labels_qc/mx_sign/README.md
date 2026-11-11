# Knee M_x sign: which sign is medial

Written by `analysis/thesis/knee_mx_sign.py` (reads files only).

**Settled: medial-positive M_x = -mx at walker_knee_r and +mx at walker_knee_l** (`physical`). The pipeline convention (+mx right, -mx left: train_cnn.load_subject comment, train_mlp F_medial loss, tools/evaluate_fmedial, analysis/thesis/label_qc.MX_SIGN) is inverted.

Why: JR reports the load on the tibia at the knee centre in the tibia body frame (z to the subject's right on both legs). A compressive condyle force (0, -f, 0) at z_c gives M_x = z_c f, and the medial condyle is at z < 0 on the right tibia and z > 0 on the left.

## 1. Two-point contact (synthetic)

```
side  f_med_n  f_lat_n  z_med_m  fy_n  mx_nm  fmed_physical_n  fmed_pipeline_n
   r     1500      600  -0.0225 -2100 -20.25             1500              600
   l     1500      600   0.0225 -2100  20.25             1500              600
```
F_medial = 0.5|F_y| + s M_x / d with contacts at z = -/+ d/2: the physical s returns the medial force, the pipeline s returns the lateral force.

## 2. Model geometry (our models, one per lab)

Median z (m, relative to the walker_knee frame origin) of the medial points (gasmed origin, grac / semiten / sart) and the lateral points (gaslat origin, bfsh insertion) on femur and tibia.

```
    lab                  folder side  n_medial  n_lateral  z_medial_m  z_lateral_m  medial_all_one_side  lateral_all_one_side  implied_medial_sign  physical_sign  pipeline_sign
 carter  carter_P010_split0_t00    l         9          3     0.02417      -0.0354                 True                  True                    1              1             -1
 carter  carter_P010_split0_t00    r         9          3    -0.02417       0.0354                 True                  True                   -1             -1              1
falisse   falisse_subject_3_t00    l         9          3     0.02387     -0.03044                 True                  True                    1              1             -1
falisse   falisse_subject_3_t00    r         9          3    -0.02387      0.03044                 True                  True                   -1             -1              1
 fregly      fregly_3GC_t00_r00    l         9          3      0.0237     -0.03294                 True                  True                    1              1             -1
 fregly      fregly_3GC_t00_r00    r         9          3     -0.0237      0.03294                 True                  True                   -1             -1              1
    han     han_s003_split1_t03    l         9          3     0.01854     -0.02876                 True                  True                    1              1             -1
    han     han_s003_split1_t03    r         9          3    -0.01854      0.02876                 True                  True                   -1             -1              1
  moore moore_subject10_t01_r01    l         9          3     0.02211     -0.03395                 True                  True                    1              1             -1
  moore moore_subject10_t01_r01    r         9          3    -0.02211      0.03395                 True                  True                   -1             -1              1
tiziana   tiziana_Subject10_t00    l         9          3     0.01759     -0.02651                 True                  True                    1              1             -1
tiziana   tiziana_Subject10_t00    r         9          3    -0.01759      0.02651                 True                  True                   -1             -1              1
```

## 3. Kaneda et al.'s condyle split (StatOp_Val SO+JR, same JR table)

Rows with medial + lateral contact force > 200 N. mx = a F_med + b F_lat + c (least squares).

```
subject code side  n_windows  n_rows  r_mx_vs_med_minus_lat  a_med_m  b_lat_m    c_nm  resid_rms_nm  mx_sd_nm  fy_compressive_frac  implied_medial_sign  physical_sign  pipeline_sign
     SC  3GC    l         62    3835                 0.9934  0.04831  -0.0505   1.021         1.489     15.57                    1                    1              1             -1
     JW  4GC    r        112    5962                -0.9911 -0.03951  0.03869  0.1364        0.9868     7.543                    1                   -1             -1              1
     DM  6GC    r         36    2399                -0.9111 -0.02786  0.01775 -0.5054        0.9345     8.003                    1                   -1             -1              1
```

## 4. Population: stance-mean M_x per walking folder

Stance = |GRF_y| of that foot > 0.05 BW. Up to --max-per-lab folders per lab with SO+JR labels, every k-th in sorted order. medial_frac_* = share of folders whose stance-mean moment is medial under that sign.

```
    lab source side  n_folders  median_mx_bwh  medial_frac_physical  medial_frac_pipeline
 carter   sojr    r        247       -0.01019                0.8664                0.1336
 carter   sojr    l        247        0.01429                0.9555               0.04453
 carter stored    r        247      0.0004624                0.4737                0.5263
falisse   sojr    r         31      -0.009201                0.8387                0.1613
falisse   sojr    l         31        0.01023                 0.871                 0.129
falisse stored    r         31     -0.0009662                0.6452                0.3548
 fregly   sojr    r         92     -0.0004497                0.5326                0.4674
 fregly   sojr    l        138       0.006397                0.6667                0.3333
 fregly stored    r         92      0.0005764                0.4565                0.5435
    han   sojr    r        241       0.001584                0.4398                0.5602
    han   sojr    l        257     -0.0005172                0.4864                0.5136
    han stored    r        241       0.005647                0.1867                0.8133
  moore   sojr    r        296      -0.006668                0.8041                0.1959
  moore   sojr    l        296        0.01332                0.9493               0.05068
  moore stored    r        296      0.0005599                0.4493                0.5507
tiziana   sojr    r        179       -0.01225                 0.838                 0.162
tiziana   sojr    l        102          0.013                 0.902               0.09804
tiziana stored    r        179      -0.005025                0.8324                0.1676
```

Mean medial-positive M_x (physical sign, BW*H x 1000) per 10 % of stance, complete stances (0.3-1.6 s) only. A walking knee adduction moment is medial through stance with an early peak; a negative early-stance value is a LATERAL moment and is not physiological.

```
    lab source side  n_folders    pct5    pct15   pct25  pct35   pct45   pct55   pct65  pct75   pct85   pct95
 carter   sojr    r        245  -28.74   -34.95   -21.1  -3.65   7.299   27.06   49.18  52.62   37.05   9.896
 carter   sojr    l        245  -24.16   -25.09     -11    4.6    11.3   27.63   48.61  57.69   41.87   14.33
 carter stored    r        245  -28.42   -38.84  -30.29 -14.47  -3.613   5.105   16.94  32.28   40.49   13.02
falisse   sojr    r          4  -5.089   -9.134 -0.3318  5.385   6.227   9.761   16.82  17.94   8.051 -0.1964
falisse stored    r          4   1.023   0.7732   1.436  1.025 -0.1169  -1.698  -1.223 0.8729    1.51  -2.147
 fregly   sojr    r          5  -6.717    3.893   1.925  5.706   3.725  -1.557   12.21  10.39  -1.433  -5.602
 fregly   sojr    l          1  -15.67    5.939   8.966  16.98   11.63   7.048   1.425 -1.764 -0.7887  -3.824
 fregly stored    r          5  -4.526   -1.396   2.478  2.814  0.4406   -1.04  -1.388 -1.005  -1.325  -3.378
    han   sojr    r          7   3.789 -0.01518   3.326  4.243   2.573  -7.637 -0.4131 -2.695  -2.515  -1.678
    han   sojr    l          5  -2.698   -1.751   2.263  3.888   4.999   13.44   7.054  13.77 -0.1885  -4.421
    han stored    r          7  -4.467   -3.613  -3.038 -2.511  -2.899  -3.388  -3.453 -3.781  -3.512  -2.958
  moore   sojr    r         83 -0.9541    7.364   8.676  10.39   11.69   11.56   6.441  1.946     1.2  -10.04
  moore   sojr    l         83  -4.553    7.768   9.629  12.53   9.834   23.09   19.44  21.76    14.3    -4.6
  moore stored    r         83  -8.942 -0.09086   1.496  1.734  -6.106 -0.8909  -1.344 -0.626   1.059  -2.878
```

## 5. Implant: medial-positive M_x against the measured medial - lateral force

From `results/thesis/gc/pilot_AB_sojr/mx_sign_check.csv` (M_x convention `physical`), instrumented-limb stance frames. r > 0: the source's medial moment rises with the implant's medial - lateral force. medial_share = mean(F_medial) / mean(|F_y|) vs the implant's mean(medial) / mean(total).

```
subject     source  n_frames  r_mx_vs_meas_med_minus_lat  medial_moment_frac  moment_term_mean_bw  medial_share  meas_medial_share
     DM osim_state       878                      0.4103              0.5444              0.01862         0.524             0.4818
     DM  osim_sojr       878                      0.4125              0.6424               0.1141        0.5375             0.4818
     DM      fpopt       878                      -0.445              0.5524              -0.1586        0.3262             0.4818
     DM       sept       878                     -0.4373              0.4226              -0.2283        0.2759             0.4818
     JW osim_state       889                       0.482             0.08324              -0.1746        0.3275             0.6626
     JW  osim_sojr       889                      0.2869                0.73               0.2875        0.5938             0.6626
     JW      fpopt       889                     -0.7198              0.3363              -0.4235        0.1189             0.6626
     JW       sept       889                     -0.7192              0.3048              -0.4309        0.1274             0.6626
     SC  osim_sojr      1244                      0.3212              0.7661               0.3622        0.6097              0.713
     SC      fpopt      1244                     -0.3477              0.4027              -0.1855         0.299              0.713
     SC       sept      1244                     -0.4165              0.3424              -0.2283        0.2705              0.713
```

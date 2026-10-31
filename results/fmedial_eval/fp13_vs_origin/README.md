# F_medial evaluation · fp13_vs_origin

- created: 2026-10-29T21:13:01
- command: `python tools/evaluate_fmedial.py --ckpt models/runs/mlp/best_model_mlp_walking_qinst_notrans_pelvisrel_winpos_contact_w20_fmW0p5_e100_2.pt models/runs/mlp/best_model_mlp_walking_qinst_notrans_pelvisrel_winpos_contact_fp13_w20_fmW0p5_e100.pt --labels september fp13 --plots --name fp13_vs_origin`
- git: 6687dc293dd10c4b2d76ee703c3ff7f55e549812 (DIRTY)
- split: test; activities: walking, static; excluded labs: none; pelvis: trial; phi=0.5, d=45.0 mm; limit 1.4 BW; no flat filter
- trials evaluated: 653; rejected by the label filters: 59 (moore 59)

## Checkpoints

| name | input set | trained on | F_medial loss weight | epoch | trained at commit |
|---|---|---|---|---|---|
| september | qinst_notrans_pelvisrel_winpos_contact_w20 | walking, exclude unknown | 0.5 | 32 | no provenance |
| fp13 | qinst_notrans_pelvisrel_winpos_contact_fp13_w20 | walking, exclude ['han', 'fregly'] | 0.5 | 27 | 6687dc293d |

## Headline: frame-pooled error, all labs (BW; moment in BW·H)

| model | activity | subset | persons | F_medial MAE | person-mean ± sd | F_y MAE | M_x MAE | force-part bias | moment-part bias | peak error |
|---|---|---|---|---|---|---|---|---|---|---|
| september | static | all | 3 | 0.634 | 0.358 ± 0.278 | 0.517 | 0.0111 | -0.237 | -0.389 |  |
| september | static | stance | 3 | 0.652 | 0.415 ± 0.267 | 0.530 | 0.0114 | -0.244 | -0.402 | -0.321 |
| september | walking | all | 6 | 0.247 | 0.466 ± 0.301 | 0.172 | 0.0053 | -0.013 | +0.014 |  |
| september | walking | stance | 6 | 0.328 | 0.620 ± 0.344 | 0.207 | 0.0071 | -0.021 | -0.000 | -0.337 |
| fp13 | static | all | 3 | 0.563 | 0.334 ± 0.229 | 0.500 | 0.0099 | -0.205 | -0.352 |  |
| fp13 | static | stance | 3 | 0.578 | 0.389 ± 0.218 | 0.512 | 0.0102 | -0.211 | -0.363 | -0.348 |
| fp13 | walking | all | 6 | 0.246 | 0.469 ± 0.285 | 0.177 | 0.0052 | -0.021 | -0.001 |  |
| fp13 | walking | stance | 6 | 0.332 | 0.630 ± 0.325 | 0.217 | 0.0072 | -0.030 | -0.015 | -0.346 |

## Paper check

Axial-force (F_y) MAE over all walking frames, Carter + Moore test trials. The paper reports 0.153 BW for its GRF-free MLP.

- september: 0.162 BW over 540 trials
- fp13: 0.165 BW over 540 trials

## Sign check

F_y < 0 (compression) in right stance: median 1.000 of frames; 28 trial(s) below 95 %: carter_P010_split0_t04, carter_P010_split0_t05, carter_P010_split0_t06, carter_P010_split0_t07, carter_P010_split0_t08, carter_P010_split0_t09, carter_P010_split0_t10, carter_P010_split0_t11, carter_P010_split0_t12, carter_P010_split0_t13

Files: summary.csv, persons.csv, trials.csv, sweep.csv, rejected.csv, plots/

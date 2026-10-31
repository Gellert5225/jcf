# F_medial evaluation · test_split_v1

- created: 2026-10-09T16:09:02
- command: `python tools/evaluate_fmedial.py --ckpt models/mlp-grffree-base/mlp-grffree-base.pt models/mlp-grffree-contact/mlp-grffree-contact.pt models/runs/mlp/best_model_mlp_both_qinst_pelvisrel_winpos_contact_w20.pt models/runs/mlp/best_model_mlp_walking_qinst_pelvisrel_winpos_contact_w20_fmW0p5_e100_2.pt models/runs/mlp/best_model_mlp_walking_qinst_notrans_pelvisrel_winpos_contact_w20_fmW0p5_e100_2.pt --labels paper contact-fmW contact-mse walk-fmW september --phi 0.4 0.5 0.6 --d-mm 40 45 50 --plots --name test_split_v1`
- git: 6687dc293dd10c4b2d76ee703c3ff7f55e549812 (clean)
- split: test; activities: walking, static; excluded labs: none; pelvis: trial; phi=0.5, d=45.0 mm; limit 1.4 BW; no flat filter
- trials evaluated: 653; rejected by the label filters: 59 (moore 59)

## Checkpoints

| name | input set | trained on | F_medial loss weight | epoch | trained at commit |
|---|---|---|---|---|---|
| paper | qinst_pelvisrel_winpos_w20 | both, exclude unknown | 0.0 | 90 | no provenance |
| contact-fmW | qinst_pelvisrel_winpos_contact_w20 | both, exclude unknown | 0.5 | 47 | no provenance |
| contact-mse | qinst_pelvisrel_winpos_contact_w20 | both, exclude unknown | 0.0 | 30 | no provenance |
| walk-fmW | qinst_pelvisrel_winpos_contact_w20 | walking, exclude unknown | 0.5 | 52 | no provenance |
| september | qinst_notrans_pelvisrel_winpos_contact_w20 | walking, exclude unknown | 0.5 | 32 | no provenance |

## Headline: frame-pooled error, all labs (BW; moment in BW·H)

| model | activity | subset | persons | F_medial MAE | person-mean ± sd | F_y MAE | M_x MAE | force-part bias | moment-part bias | peak error |
|---|---|---|---|---|---|---|---|---|---|---|
| paper | static | all | 3 | 0.703 | 0.363 ± 0.329 | 0.501 | 0.0131 | -0.234 | -0.455 |  |
| paper | static | stance | 3 | 0.721 | 0.383 ± 0.325 | 0.515 | 0.0134 | -0.241 | -0.471 | -0.464 |
| paper | walking | all | 6 | 0.255 | 0.425 ± 0.203 | 0.160 | 0.0059 | -0.026 | +0.058 |  |
| paper | walking | stance | 6 | 0.325 | 0.572 ± 0.267 | 0.198 | 0.0076 | -0.044 | +0.062 | -0.285 |
| contact-fmW | static | all | 3 | 1.316 | 0.586 ± 0.715 | 1.396 | 0.0177 | +0.686 | +0.610 |  |
| contact-fmW | static | stance | 3 | 1.355 | 0.618 ± 0.716 | 1.437 | 0.0182 | +0.707 | +0.629 | +0.796 |
| contact-fmW | walking | all | 6 | 0.208 | 0.228 ± 0.048 | 0.152 | 0.0046 | -0.006 | +0.057 |  |
| contact-fmW | walking | stance | 6 | 0.273 | 0.298 ± 0.104 | 0.182 | 0.0061 | -0.007 | +0.074 | -0.173 |
| contact-mse | static | all | 3 | 0.240 | 0.157 ± 0.080 | 0.323 | 0.0030 | +0.145 | +0.022 |  |
| contact-mse | static | stance | 3 | 0.245 | 0.174 ± 0.072 | 0.330 | 0.0030 | +0.149 | +0.023 | +0.172 |
| contact-mse | walking | all | 6 | 0.272 | 0.330 ± 0.072 | 0.153 | 0.0064 | -0.010 | +0.087 |  |
| contact-mse | walking | stance | 6 | 0.370 | 0.446 ± 0.129 | 0.189 | 0.0088 | -0.010 | +0.138 | -0.182 |
| walk-fmW | static | all | 3 | 0.807 | 0.392 ± 0.406 | 0.762 | 0.0123 | -0.369 | -0.431 |  |
| walk-fmW | static | stance | 3 | 0.830 | 0.421 ± 0.401 | 0.783 | 0.0126 | -0.380 | -0.444 | -0.528 |
| walk-fmW | walking | all | 6 | 0.223 | 0.312 ± 0.194 | 0.162 | 0.0047 | -0.021 | +0.028 |  |
| walk-fmW | walking | stance | 6 | 0.289 | 0.395 ± 0.194 | 0.193 | 0.0062 | -0.034 | +0.022 | -0.352 |
| september | static | all | 3 | 0.634 | 0.358 ± 0.278 | 0.517 | 0.0111 | -0.237 | -0.389 |  |
| september | static | stance | 3 | 0.652 | 0.415 ± 0.267 | 0.530 | 0.0114 | -0.244 | -0.402 | -0.321 |
| september | walking | all | 6 | 0.247 | 0.466 ± 0.301 | 0.172 | 0.0053 | -0.013 | +0.014 |  |
| september | walking | stance | 6 | 0.328 | 0.620 ± 0.344 | 0.207 | 0.0071 | -0.021 | -0.000 | -0.337 |

## Paper check

Axial-force (F_y) MAE over all walking frames, Carter + Moore test trials. The paper reports 0.153 BW for its GRF-free MLP.

- paper: 0.152 BW over 540 trials
- contact-fmW: 0.151 BW over 540 trials
- contact-mse: 0.150 BW over 540 trials
- walk-fmW: 0.155 BW over 540 trials
- september: 0.162 BW over 540 trials

## Sign check

F_y < 0 (compression) in right stance: median 1.000 of frames; 28 trial(s) below 95 %: carter_P010_split0_t04, carter_P010_split0_t05, carter_P010_split0_t06, carter_P010_split0_t07, carter_P010_split0_t08, carter_P010_split0_t09, carter_P010_split0_t10, carter_P010_split0_t11, carter_P010_split0_t12, carter_P010_split0_t13

Files: summary.csv, persons.csv, trials.csv, sweep.csv, rejected.csv, plots/

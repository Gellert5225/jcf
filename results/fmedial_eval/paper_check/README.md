# F_medial evaluation · paper_check

- created: 2026-10-09T16:01:50
- command: `python tools/evaluate_fmedial.py --ckpt models/mlp-grffree-base/mlp-grffree-base.pt --exclude han fregly --labels paper --name paper_check`
- git: 023ffb83edc72b6cacf0b51739d029693d6fd5ba (DIRTY)
- split: test; activities: walking, static; excluded labs: ['han', 'fregly']; pelvis: trial; phi=0.5, d=45.0 mm; limit 1.4 BW; no flat filter
- trials evaluated: 581; rejected by the label filters: 59 (moore 59)

## Checkpoints

| name | input set | trained on | F_medial loss weight | epoch | trained at commit |
|---|---|---|---|---|---|
| paper | qinst_pelvisrel_winpos_w20 | both, exclude unknown | 0.0 | 90 | no provenance |

## Headline: frame-pooled error, all labs (BW; moment in BW·H)

| model | activity | subset | persons | F_medial MAE | person-mean ± sd | F_y MAE | M_x MAE | force-part bias | moment-part bias | peak error |
|---|---|---|---|---|---|---|---|---|---|---|
| paper | static | all | 2 | 0.139 | 0.130 ± 0.014 | 0.045 | 0.0034 | +0.001 | -0.119 |  |
| paper | static | stance | 2 | 0.146 | 0.153 ± 0.009 | 0.043 | 0.0036 | +0.001 | -0.130 | -0.093 |
| paper | walking | all | 4 | 0.238 | 0.353 ± 0.147 | 0.153 | 0.0055 | -0.025 | +0.070 |  |
| paper | walking | stance | 4 | 0.303 | 0.506 ± 0.295 | 0.186 | 0.0072 | -0.041 | +0.085 | -0.335 |

## Paper check

Axial-force (F_y) MAE over all walking frames, Carter + Moore test trials. The paper reports 0.153 BW for its GRF-free MLP.

- paper: 0.152 BW over 540 trials

## Sign check

F_y < 0 (compression) in right stance: median 1.000 of frames; 28 trial(s) below 95 %: carter_P010_split0_t04, carter_P010_split0_t05, carter_P010_split0_t06, carter_P010_split0_t07, carter_P010_split0_t08, carter_P010_split0_t09, carter_P010_split0_t10, carter_P010_split0_t11, carter_P010_split0_t12, carter_P010_split0_t13

Files: summary.csv, persons.csv, trials.csv, sweep.csv, rejected.csv

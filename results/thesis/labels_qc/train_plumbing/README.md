# SO+JR training set: coverage and label filters on the same trials

Created 2026-11-10T15:03:56 (git 36ae775) by analysis/thesis/sojr_trainset.py; reads files only.
Scope: {'split': ['training', 'testing'], 'activity': ['walking', 'static'], 'exclude': [], 'labs': None}; 9529 folders with jcf_output/ labels.

**READY**: 9529 of 9529 have a complete jcf_output_sojr/ file; 1 of those hold non-finite values (dropped by the loaders).

## Coverage (split x activity x lab)

| split | activity | lab | n_stored | sojr_complete | sojr_incomplete | sojr_missing | sojr_nonfinite |
|---|---|---|---|---|---|---|---|
| testing | static | han | 19 | 19 | 0 | 0 | 0 |
| testing | static | moore | 6 | 6 | 0 | 0 | 0 |
| testing | static | tiziana | 11 | 11 | 0 | 0 | 0 |
| testing | walking | carter | 33 | 33 | 0 | 0 | 0 |
| testing | walking | falisse | 5 | 5 | 0 | 0 | 0 |
| testing | walking | fregly | 43 | 43 | 0 | 0 | 0 |
| testing | walking | han | 10 | 10 | 0 | 0 | 0 |
| testing | walking | moore | 566 | 566 | 0 | 0 | 0 |
| testing | walking | tiziana | 19 | 19 | 0 | 0 | 0 |
| training | static | carter | 159 | 159 | 0 | 0 | 0 |
| training | static | fregly | 4 | 4 | 0 | 0 | 0 |
| training | static | han | 108 | 108 | 0 | 0 | 0 |
| training | static | moore | 39 | 39 | 0 | 0 | 0 |
| training | static | tiziana | 165 | 165 | 0 | 0 | 0 |
| training | walking | carter | 1200 | 1200 | 0 | 0 | 0 |
| training | walking | falisse | 26 | 26 | 0 | 0 | 0 |
| training | walking | fregly | 98 | 98 | 0 | 0 | 0 |
| training | walking | han | 253 | 253 | 0 | 0 | 0 |
| training | walking | moore | 6222 | 6222 | 0 | 0 | 0 |
| training | walking | tiziana | 543 | 543 | 0 | 0 | 1 |
| ALL | ALL | ALL | 9529 | 9529 | 0 | 0 | 1 |

## Relabel summary statuses (lab x status)

Rows are keyed by folder name, which repeats across roots, so these are counts, not a per-folder test.

| summary | lab | status | n |
|---|---|---|---|
| results/thesis/gc/knee_jr/relabel_all.csv | carter | exists | 1392 |
| results/thesis/gc/knee_jr/relabel_all.csv | carter | no SO output | 1 |
| results/thesis/gc/knee_jr/relabel_all.csv | falisse | exists | 5 |
| results/thesis/gc/knee_jr/relabel_all.csv | falisse | no SO output | 9 |
| results/thesis/gc/knee_jr/relabel_all.csv | falisse | ok | 26 |
| results/thesis/gc/knee_jr/relabel_all.csv | fregly | exists | 48 |
| results/thesis/gc/knee_jr/relabel_all.csv | fregly | no SO output | 55 |
| results/thesis/gc/knee_jr/relabel_all.csv | fregly | ok | 97 |
| results/thesis/gc/knee_jr/relabel_all.csv | grand | no SO output | 1 |
| results/thesis/gc/knee_jr/relabel_all.csv | han | exists | 145 |
| results/thesis/gc/knee_jr/relabel_all.csv | han | ok | 245 |
| results/thesis/gc/knee_jr/relabel_all.csv | moore | exists | 963 |
| results/thesis/gc/knee_jr/relabel_all.csv | moore | no SO output | 2387 |
| results/thesis/gc/knee_jr/relabel_all.csv | moore | ok | 5870 |
| results/thesis/gc/knee_jr/relabel_all.csv | tiziana | exists | 738 |
| results/thesis/gc/knee_jr/relabel_all.csv | tiziana | no SO output | 223 |
| results/thesis/gc/knee_jr/relabel_all.csv | william | no JR output | 1 |
| results/thesis/gc/knee_jr/relabel_all.csv | william | no SO output | 2 |

## Flat filter on the same walking trials (CV < 0.25 or range < 0.3 BW)

Trials with finite labels in both sets. flat_only_stored = removed under the stored labels but kept under SO+JR.

| split | lab | n_same | flat_stored | flat_sojr | flat_both | flat_only_stored | flat_only_sojr | kept_stored | kept_sojr |
|---|---|---|---|---|---|---|---|---|---|
| testing | carter | 33 | 4 | 2 | 2 | 2 | 0 | 29 | 31 |
| testing | falisse | 5 | 0 | 0 | 0 | 0 | 0 | 5 | 5 |
| testing | fregly | 43 | 30 | 3 | 3 | 27 | 0 | 13 | 40 |
| testing | han | 10 | 6 | 0 | 0 | 6 | 0 | 4 | 10 |
| testing | moore | 566 | 85 | 28 | 23 | 62 | 5 | 481 | 538 |
| testing | tiziana | 19 | 19 | 0 | 0 | 19 | 0 | 0 | 19 |
| testing | ALL | 676 | 144 | 33 | 28 | 116 | 5 | 532 | 643 |
| training | carter | 1200 | 0 | 0 | 0 | 0 | 0 | 1200 | 1200 |
| training | falisse | 26 | 4 | 0 | 0 | 4 | 0 | 22 | 26 |
| training | fregly | 98 | 44 | 12 | 8 | 36 | 4 | 54 | 86 |
| training | han | 253 | 192 | 147 | 142 | 50 | 5 | 61 | 106 |
| training | moore | 6222 | 1002 | 83 | 76 | 926 | 7 | 5220 | 6139 |
| training | tiziana | 542 | 520 | 89 | 87 | 433 | 2 | 22 | 453 |
| training | ALL | 8341 | 1762 | 331 | 313 | 1449 | 18 | 6579 | 8010 |

## Flat filter on all stored-label walking trials

| split | lab | n | flat_stored | nonfinite_stored |
|---|---|---|---|---|
| testing | carter | 33 | 4 | 0 |
| testing | falisse | 5 | 0 | 0 |
| testing | fregly | 43 | 30 | 0 |
| testing | han | 10 | 6 | 0 |
| testing | moore | 566 | 85 | 0 |
| testing | tiziana | 19 | 19 | 0 |
| testing | ALL | 676 | 144 | 0 |
| training | carter | 1200 | 0 | 0 |
| training | falisse | 26 | 4 | 0 |
| training | fregly | 98 | 44 | 0 |
| training | han | 253 | 192 | 0 |
| training | moore | 6222 | 1002 | 0 |
| training | tiziana | 543 | 520 | 0 |
| training | ALL | 8342 | 1762 | 0 |

## Peak tests on the same walking trials (resultant > 6 BW, |M_x| > 0.15 BW*H in any label frame)

| split | lab | n_same | stored_peak_gt | stored_mx_gt | stored_rejected | stored_peak_median_bw | sojr_peak_gt | sojr_mx_gt | sojr_rejected | sojr_peak_median_bw |
|---|---|---|---|---|---|---|---|---|---|---|
| testing | carter | 33 | 0 | 0 | 0 | 2.37 | 30 | 0 | 30 | 10.96 |
| testing | falisse | 5 | 0 | 0 | 0 | 0.71 | 3 | 0 | 3 | 6.07 |
| testing | fregly | 43 | 0 | 0 | 0 | 0.54 | 9 | 0 | 9 | 3.56 |
| testing | han | 10 | 0 | 0 | 0 | 1.34 | 4 | 0 | 4 | 4.81 |
| testing | moore | 566 | 36 | 57 | 59 | 1.12 | 222 | 118 | 231 | 5.52 |
| testing | tiziana | 19 | 0 | 0 | 0 | 0.57 | 8 | 0 | 8 | 5.17 |
| testing | ALL | 676 | 36 | 57 | 59 | 1.11 | 276 | 118 | 285 | 5.55 |
| training | carter | 1200 | 0 | 0 | 0 | 2.19 | 1185 | 2 | 1185 | 10.95 |
| training | falisse | 26 | 0 | 0 | 0 | 1.00 | 5 | 0 | 5 | 5.49 |
| training | fregly | 98 | 0 | 0 | 0 | 0.89 | 13 | 0 | 13 | 3.41 |
| training | han | 253 | 2 | 0 | 2 | 1.01 | 45 | 0 | 45 | 3.93 |
| training | moore | 6222 | 401 | 733 | 768 | 1.26 | 4834 | 1370 | 4870 | 7.75 |
| training | tiziana | 542 | 3 | 2 | 5 | 0.81 | 222 | 0 | 222 | 5.30 |
| training | ALL | 8341 | 406 | 735 | 775 | 1.30 | 6304 | 1372 | 6340 | 7.94 |

Files: coverage.csv, relabel_status.csv, trials.csv, flat_same_trials.csv, peak_same_trials.csv, run_info.json

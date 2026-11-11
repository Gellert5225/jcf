# SO+JR label masks (label_mask.npz)

- command: `python -I analysis/thesis/so_label_qc.py masks --workers 20`; 9530 folders, 0 errors, 34 s on 20 workers
- keep_r = right foot COP not fixed (< 0.2 m/s) and no frame flag of sat_knee, sat_leg3, res_leg, pelvis_m, jump, coact (frame_flags in analysis/thesis/so_label_qc.py) and finite label/SO rows. fy5 (|F_y| > 5 BW) is stored, not in keep_r.
- by_lab.csv (right knee; *_seconds and *_frac are weighted by the JR time step): kept_stance_frac_r = share of stance time keep_r keeps; kept_stance_frac_any_so5_r = the same with the rules.csv any_so union (no sat_leg3); fy5_after_so_* = kept frames the 5 BW rule would remove in addition; kept_stance_frac_with_fy5_r = stance share left if it were added.

| split | activity | lab | trials | fixed COP (r) | trials with kept frames | kept stance | kept stance (any_so, no sat_leg3) | +5 BW removes (stance frames) | of kept stance | stance left with 5 BW |
|---|---|---|---|---|---|---|---|---|---|---|
| external | external | william | 1 | 1 | 0 | 0.0% | 0.0% | 0 | 0.0% | 0.0% |
| testing | static | han | 19 | 0 | 15 | 23.1% | 23.1% | 0 | 0.0% | 23.1% |
| testing | static | moore | 6 | 0 | 3 | 3.1% | 3.1% | 0 | 0.0% | 3.1% |
| testing | static | tiziana | 11 | 0 | 11 | 62.9% | 62.9% | 0 | 0.0% | 62.9% |
| testing | walking | carter | 33 | 30 | 3 | 5.9% | 5.9% | 0 | 0.0% | 5.9% |
| testing | walking | falisse | 5 | 0 | 5 | 54.8% | 54.8% | 0 | 0.0% | 54.8% |
| testing | walking | fregly | 43 | 5 | 38 | 69.8% | 69.8% | 82 | 10.0% | 62.8% |
| testing | walking | han | 10 | 3 | 7 | 70.5% | 70.7% | 2 | 0.1% | 70.4% |
| testing | walking | moore | 566 | 0 | 565 | 52.2% | 52.2% | 1867 | 5.3% | 49.4% |
| testing | walking | tiziana | 19 | 14 | 5 | 0.0% | 0.0% | 0 | 0.0% | 0.0% |
| training | static | carter | 159 | 0 | 147 | 46.6% | 46.7% | 27 | 0.0% | 46.6% |
| training | static | fregly | 4 | 0 | 4 | 84.2% | 84.2% | 0 | 0.0% | 84.2% |
| training | static | han | 108 | 0 | 92 | 44.6% | 47.9% | 678 | 1.7% | 43.9% |
| training | static | moore | 39 | 0 | 26 | 47.0% | 47.0% | 0 | 0.0% | 47.0% |
| training | static | tiziana | 165 | 0 | 161 | 64.7% | 64.7% | 200 | 6.9% | 60.3% |
| training | walking | carter | 1200 | 1140 | 60 | 3.0% | 3.0% | 2643 | 6.6% | 2.8% |
| training | walking | falisse | 26 | 8 | 18 | 37.5% | 37.5% | 1 | 0.2% | 37.4% |
| training | walking | fregly | 98 | 3 | 95 | 74.6% | 74.6% | 207 | 4.8% | 71.0% |
| training | walking | han | 253 | 75 | 162 | 41.8% | 43.9% | 2044 | 3.2% | 40.5% |
| training | walking | moore | 6222 | 22 | 6186 | 43.7% | 43.7% | 6535 | 2.2% | 42.7% |
| training | walking | tiziana | 543 | 282 | 259 | 7.7% | 7.7% | 116 | 8.7% | 7.0% |
| training | walking | pool | 7991 | 1452 | 6523 | 25.4% | 25.4% | 9295 | 2.5% | 24.7% |

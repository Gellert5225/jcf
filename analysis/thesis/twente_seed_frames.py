"""Twente real-IMU check: seed-to-seed F_medial disagreement on identical inputs against the IMU-vs-optical difference.

Post-processing of a twente_imu.py run that saved per-frame outputs for every trial (`--save-frames all`); no model,
no OpenSim. Reads <run_dir>/frames/<Subj>_<walk_NN>_<variant>.npz (keys fmed|<ckpt>|<path>, fmed|<ckpt>|t,
fmed|<ckpt>|stance_r for the checkpoints qc_s1, qc_s2, qc_s3 and qc_origin, and the paths O, V, U:<variant>:gyro,
U:<variant>:grf). Frames are the scored frames of each trial (the same as fmedial.csv). Aggregation as in
twente_imu.py: trial -> subject mean over speeds -> median [IQR] over subjects; per speed: median over subjects.

Every comparison is also given per seed pair, because the 3-seed mean pairwise disagreement is dominated by the
undertrained qc_s2 (best epoch 8, stored 0-based as 7). The well-trained pair is qc_s1 / qc_s3. Rows:
  - seedpair_mae_* (mean over pairs, s1-s3, s2 against the others, s1-s2, s2-s3, s1-s3 on stance frames),
    foot_s1_vs_origin, seed_vs_ens_mae_mean, seed_sd_* on each path;
  - <ckpt>_vs_O and ens_vs_O (3-seed ensemble) of every test path against the optical path, all and stance frames;
  - ens13_*: the qc_s1 + qc_s3 ensemble against the same ensemble on the optical path (MAE, stance MAE, bias, r).
seed_ratio_table.csv divides each seed's IMU-vs-optical difference by the s1-s3 disagreement on optical input, as a
ratio of the subject medians and as the median of per-subject ratios.

Outputs (in --out, default <run_dir>): seed_disagreement_trials.csv (one row per trial and variant),
seed_disagreement_table.csv (median [IQR] over subjects, overall and per speed), seed_ratio_table.csv.
results/thesis/twente/audit_twente/ holds the copies the paper cites. The per-frame npz stay local (the dataset
states no licence; *.npz is git-ignored).

    python -I analysis/thesis/twente_seed_frames.py <run_dir> [--out results/thesis/twente/audit_twente]

The audit run's per-frame rerun (seg-static and seg-mvnmodel, 108 npz) lives outside the repo, e.g.
<scratchpad>/thesis/twente_frames/audit_twente_frames; it takes about 5 s with OMP/OPENBLAS/MKL_NUM_THREADS=1.
"""
import argparse
import glob
import itertools
import os
import re

import numpy as np
import pandas as pd

SEEDS = ["qc_s1", "qc_s2", "qc_s3"]
GOOD = ["qc_s1", "qc_s3"]          # well-trained pair (best epochs 56 / 90); qc_s2 stopped at epoch 8
TRIALS_CSV, TABLE_CSV, RATIO_CSV = "seed_disagreement_trials.csv", "seed_disagreement_table.csv", "seed_ratio_table.csv"


def _mae(a, b, mask=None):
    d = np.abs(a - b)
    return float(d[mask].mean()) if mask is not None else float(d.mean())


def _corr(a, b):
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 and a.std() > 1e-9 and b.std() > 1e-9 else np.nan


def trial_rows(run_dir):
    """One row per frames/*.npz file (trial x variant) with every per-trial metric."""
    files = sorted(glob.glob(os.path.join(run_dir, "frames", "*.npz")))
    if not files:
        raise SystemExit(f"no per-frame npz in {os.path.join(run_dir, 'frames')} (twente_imu.py --save-frames all)")
    rows = []
    for f in files:
        m = re.match(r"Subj(\d+)_(walk_\d+)_(.+)\.npz", os.path.basename(f))
        subj, trial, var = m.groups()
        z = np.load(f)
        speed = int(trial.split("_")[1]) / 36.0
        st = z["fmed|qc_s1|stance_r"].astype(bool)
        t0 = z["fmed|qc_s1|t"]
        for ck in SEEDS + ["qc_origin"]:
            assert np.array_equal(z[f"fmed|{ck}|t"], t0), (f, ck)
            assert np.array_equal(z[f"fmed|{ck}|stance_r"].astype(bool), st), (f, ck)

        def F(ck, p):
            return z[f"fmed|{ck}|{p}"].astype(float)

        paths = ["O", "V", f"U:{var}:gyro", f"U:{var}:grf"]
        r = dict(subject=subj, trial=trial, speed=round(speed, 2), variant=var)
        for p in paths:
            key = p.replace(f"U:{var}:", "U_")
            pair = [_mae(F(a, p), F(b, p)) for a, b in itertools.combinations(SEEDS, 2)]
            r[f"seedpair_mae_mean|{key}"] = float(np.mean(pair))
            r[f"seedpair_mae_s1s3|{key}"] = _mae(F("qc_s1", p), F("qc_s3", p))
            r[f"seedpair_mae_s2x|{key}"] = float(np.mean([_mae(F("qc_s2", p), F(o, p)) for o in ("qc_s1", "qc_s3")]))
            r[f"foot_s1_vs_origin|{key}"] = _mae(F("qc_s1", p), F("qc_origin", p))
            ens = np.mean([F(s, p) for s in SEEDS], axis=0)
            r[f"seed_vs_ens_mae_mean|{key}"] = float(np.mean([_mae(F(s, p), ens) for s in SEEDS]))
            r[f"seed_sd_mean|{key}"] = float(np.mean(np.std([F(s, p) for s in SEEDS], axis=0)))
            r[f"seed_sd_stance|{key}"] = float(np.mean(np.std([F(s, p) for s in SEEDS], axis=0)[st]))
            r[f"seedpair_mae_s1s2|{key}"] = _mae(F("qc_s1", p), F("qc_s2", p))
            r[f"seedpair_mae_s2s3|{key}"] = _mae(F("qc_s2", p), F("qc_s3", p))
            r[f"seedpair_mae_s1s3_stance|{key}"] = _mae(F("qc_s1", p), F("qc_s3", p), st)
        # IMU / vendor vs optical, per seed and for the 3-seed and the s1+s3 ensembles
        ensO = np.mean([F(s, "O") for s in SEEDS], axis=0)
        ens13O = np.mean([F(s, "O") for s in GOOD], axis=0)
        for p in paths[1:]:
            key = p.replace(f"U:{var}:", "U_")
            ens = np.mean([F(s, p) for s in SEEDS], axis=0)
            r[f"ens_vs_O|{key}"] = _mae(ens, ensO)
            for s in SEEDS + ["qc_origin"]:
                r[f"{s}_vs_O|{key}"] = _mae(F(s, p), F(s, "O"))
            ens13 = np.mean([F(s, p) for s in GOOD], axis=0)
            r[f"ens13_vs_O|{key}"] = _mae(ens13, ens13O)
            r[f"ens13_vs_O_stance|{key}"] = _mae(ens13, ens13O, st)
            r[f"ens13_bias_vs_O|{key}"] = float(np.mean(ens13 - ens13O))
            r[f"ens13_r_vs_O|{key}"] = _corr(ens13, ens13O)
            r[f"ens_vs_O_stance|{key}"] = _mae(ens, ensO, st)
            r[f"ens_bias_vs_O|{key}"] = float(np.mean(ens - ensO))
            r[f"ens_r_vs_O|{key}"] = _corr(ens, ensO)
            for s in SEEDS + ["qc_origin"]:
                r[f"{s}_vs_O_stance|{key}"] = _mae(F(s, p), F(s, "O"), st)
        rows.append(r)
    return pd.DataFrame(rows)


def _iqr(x):
    return f"{np.median(x):.3f} [{np.percentile(x, 25):.3f}, {np.percentile(x, 75):.3f}]"


def summary_table(df):
    """median [IQR] over subjects (subject mean over speeds) of every metric, overall and per speed."""
    out = []
    for var, g in df.groupby("variant"):
        metrics = [c for c in g.columns if "|" in c]
        for mtr in metrics:
            # the O / V paths do not depend on the IMU variant: list their seed-disagreement rows under seg-static only
            if var != "seg-static" and mtr.endswith(("|O", "|V")) and not mtr.startswith(("ens_", "ens13_", "qc_")):
                continue
            row = dict(variant=var, metric=mtr, all=_iqr(g.groupby("subject")[mtr].mean()))
            for sp, gs in g.groupby("speed"):
                row[f"{sp:.2f}"] = f"{np.median(gs.groupby('subject')[mtr].mean()):.3f}"
            out.append(row)
    return pd.DataFrame(out)


def ratio_table(df):
    """Each IMU-vs-optical difference over the well-trained seed pair's disagreement on optical input."""
    den_cols = ["seedpair_mae_s1s3|O", "seedpair_mae_s1s3_stance|O"]
    ref = df[df["variant"] == "seg-static"].set_index(["subject", "trial"])[den_cols]
    rat = []
    for var, g in df.groupby("variant"):
        # optical path is identical in every variant's npz, so its denominators must match seg-static exactly
        assert np.array_equal(g.set_index(["subject", "trial"])[den_cols].loc[ref.index].to_numpy(), ref.to_numpy())
        for s in GOOD + ["qc_s2", "qc_origin"]:
            for key in ["U_gyro", "U_grf", "V"]:
                for stance in (False, True):
                    num_col = f"{s}_vs_O{'_stance' if stance else ''}|{key}"
                    dcol = den_cols[1] if stance else den_cols[0]
                    per = g.groupby("subject")[[num_col, dcol]].mean()
                    ratio = per[num_col] / per[dcol]
                    rat.append(dict(variant=var, numerator=num_col, denominator=dcol,
                                    numerator_median=round(float(per[num_col].median()), 4),
                                    denominator_median=round(float(per[dcol].median()), 4),
                                    ratio_of_medians=round(float(per[num_col].median() / per[dcol].median()), 3),
                                    median_subject_ratio=round(float(ratio.median()), 3),
                                    ratio_q25=round(float(ratio.quantile(.25)), 3),
                                    ratio_q75=round(float(ratio.quantile(.75)), 3),
                                    ratio_min=round(float(ratio.min()), 3), n_subjects=int(len(ratio))))
    return pd.DataFrame(rat)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run_dir", help="twente_imu.py output folder whose frames/ holds the per-frame npz of every trial")
    ap.add_argument("--out", default=None, help="folder for the three CSVs (default: run_dir)")
    ap.add_argument("--quiet", action="store_true", help="do not print the tables")
    a = ap.parse_args(argv)
    out = a.out or a.run_dir
    os.makedirs(out, exist_ok=True)

    df = trial_rows(a.run_dir)
    df.to_csv(os.path.join(out, TRIALS_CSV), index=False, float_format="%.5g")
    res = summary_table(df)
    res.to_csv(os.path.join(out, TABLE_CSV), index=False)
    rt = ratio_table(df)
    rt.to_csv(os.path.join(out, RATIO_CSV), index=False)
    if not a.quiet:
        pd.set_option("display.width", 250)
        pd.set_option("display.max_rows", 500)
        pd.set_option("display.max_colwidth", 60)
        print(res.to_string(index=False))
        print()
        print(rt.to_string(index=False))
    print(f"{len(df)} trial files -> {os.path.join(out, TABLE_CSV)}, {TRIALS_CSV}, {RATIO_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

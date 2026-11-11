"""Dry build of the *_sojrqc training and validation sets, exactly as train_mlp.train() selects and loads them
(walking, --exclude han fregly, seed 1, default flat filter on the SO+JR labels), without training.

Per trial: FrameDataset([trial]) with the trial-level peak/moment filters at the values train() resolves for the input
set when --max-peak / --max-moment are not given (train_mlp.label_filter_defaults: off, 1000 / 1000, for a label-mask
set; the same as passing --max-peak 1000 --max-moment 1000); the rows it keeps are checked against
_build_com_features(...)[label_keep]. A second FrameDataset with --max-peak 6 --max-moment 0.15 (the other sets'
defaults) checks the loader's kept-frame judging against the per-trial maxima.

trial_filters.csv: candidate --max-peak / --max-moment values, judged two ways. judged_on=kept_frames is what the
loader does for a label-mask set since Nov 10 2026 (train_mlp.kept_labels_pass: max over the kept frames' labels, the
training targets, of the right-knee force resultant / BW and |M_x| / (BW*H)); judged_on=all_jr_rows is the earlier
behaviour, train_cnn.load_subject over every JR row, masked rows included.

    python -I <this> --workers 20 --out <dir>
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd

REPO = "/home/gellert/Developer/medical_dt"
sys.path.insert(0, REPO)
os.chdir(REPO)

INPUT_SET = os.environ.get("DRY_INPUT_SET", "qinst_notrans_pelvisrel_winpos_contact_fpopt_w20_t100_sojrqc")
PRE = "walker_knee_r_on_tibia_r_in_tibia_r_"
PEAKS = (6.0, 8.0, 10.0, 15.0, 20.0, 30.0, 1e9)
MOMENTS = (0.15, 0.2, 0.3, 0.5, 1.0, 1e9)


def one(job):
    import torch
    torch.set_num_threads(1)
    import train_mlp as tm
    from train_cnn import load_sto, jr_label_path
    d, split = job
    cfg = tm.INPUT_SETS[INPUT_SET]
    meta = json.load(open(os.path.join(d, "metadata.json")))
    bw, h = meta["mass_kg"] * 9.81, float(meta.get("height_m", 1.7))
    out = dict(trial=os.path.basename(d), lab=os.path.basename(d).split("_")[0], person=tm._subject_id(d), split=split,
               H=h)
    peak0, mom0 = tm.label_filter_defaults(cfg)          # what train() uses without --max-peak / --max-moment
    ds = tm.FrameDataset([d], max_peak_bw=peak0, max_moment_bwh=mom0, input_set=INPUT_SET)
    out.update(loaded=not ds.rejected_dirs, fully_masked=bool(ds.fully_masked_dirs), n_frames=int(ds.n_frames_unmasked),
               n_kept=len(ds))
    ds6 = tm.FrameDataset([d], max_peak_bw=6.0, max_moment_bwh=0.15, input_set=INPUT_SET)
    out.update(rejected_at_6_015=bool(ds6.rejected_dirs), n_kept_at_6_015=len(ds6))
    # JR rows: what load_subject's trial-level tests see, on all rows and on the kept (keep_r) rows
    jr = load_sto(jr_label_path(d, tm.SOJR_LABEL_SUBDIR))
    f = jr[[PRE + c for c in ("fx", "fy", "fz")]].to_numpy(float) / bw
    res = np.sqrt((f ** 2).sum(1))
    mx = np.abs(jr[PRE + "mx"].to_numpy(float)) / (bw * h)
    _, keep_jr = tm.load_label_mask(d, cfg)
    out.update(peak_all=float(res.max()), mx_all=float(mx.max()),
               peak_kept=float(res[keep_jr].max()) if keep_jr.any() else np.nan,
               mx_kept=float(mx[keep_jr].max()) if keep_jr.any() else np.nan)
    arrays = None
    out.update(peak_kept_frames=np.nan, mx_kept_frames=np.nan)
    if not ds.rejected_dirs:
        b = tm._build_com_features(d, cfg, peak0, 1, True, return_meta=True, max_moment_bwh=mom0)
        _, labels, _, m = b
        keep = m["label_keep"]
        assert np.array_equal(ds.frames_out, labels[keep]) if keep.any() else len(ds) == 0
        st = m["r_contact"][keep]
        out.update(n_stance=int(m["r_contact"].sum()), n_kept_stance=int(st.sum()))
        lk = labels[keep]
        if keep.any():
            out.update(peak_kept_frames=float(np.sqrt((lk[:, :3].astype(np.float64) ** 2).sum(1)).max()),
                       mx_kept_frames=float(np.abs(lk[:, 3]).max()))
        arrays = dict(fy=lk[:, 1].astype(np.float32), mx=lk[:, 3].astype(np.float32), stance=st,
                      res=np.sqrt((lk[:, :3] ** 2).sum(1)).astype(np.float32))
    return out, arrays


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    import train_mlp as tm
    from train_cnn import SEED, TRAIN_SPLIT
    from pipeline.compute_com_features import foot_point_filename
    t0 = time.time()
    cfg = tm.INPUT_SETS[INPUT_SET]
    label_subdir = tm.label_subdir_of(cfg)
    # train(): label config, then discovery with the flat filter (walking only), then the person split by seed
    dirs, scan = tm._find_trials(tm.DATA_ROOTS["walking"], dataset=None, exclude=["han", "fregly"],
                                 label_subdir=label_subdir, flat_filter=True, flat_cv=tm.FLAT_CV, flat_dr=tm.FLAT_DR,
                                 flat_label_subdir=None, allow_partial_labels=False, label_mask=cfg["label_mask"])
    td, vd, ns_t, ns_v = tm._split_dirs(dirs, TRAIN_SPLIT, SEED)
    fp = cfg.get("foot_point")
    fp_name = foot_point_filename(fp) if fp is not None else None
    missing_fp = [d for d in td + vd if fp_name and os.path.exists(os.path.join(d, "com_features.npz"))
                  and not os.path.exists(os.path.join(d, fp_name))]
    disc = dict(n_found=scan["n_found"], n_stored=scan["n_stored"], missing=len(scan["missing"]),
                nonfinite=[os.path.basename(x) for x in scan["nonfinite"]], n_flat=scan["n_flat"],
                n_trials=len(dirs), train_persons=ns_t, val_persons=ns_v, n_train=len(td), n_val=len(vd),
                missing_foot_point=len(missing_fp), seed=SEED, train_split=TRAIN_SPLIT)
    print(json.dumps(disc), flush=True)
    jobs = [(d, "train") for d in td] + [(d, "val") for d in vd]
    rows, arrs = [], {}
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for k, (r, ar) in enumerate(pool.imap_unordered(one, jobs, chunksize=8)):
            rows.append(r)
            if ar is not None:
                arrs[(r["split"], r["trial"])] = (r["lab"], r["H"], ar)
            if (k + 1) % 1000 == 0:
                print(f"  {k + 1}/{len(jobs)} {time.time() - t0:.0f} s", flush=True)
    per = pd.DataFrame(rows)
    per.to_csv(os.path.join(a.out, "per_trial.csv"), index=False)
    # label distribution of the kept frames per lab and split (and ALL)
    dist = []
    for split in ("train", "val"):
        for lab in sorted(per.lab.unique()) + ["ALL"]:
            items = [v for (sp, _t), v in arrs.items() if sp == split and (lab == "ALL" or v[0] == lab)]
            if not items:
                continue
            fy = np.concatenate([v[2]["fy"] for v in items]); mx = np.concatenate([v[2]["mx"] for v in items])
            st = np.concatenate([v[2]["stance"] for v in items])
            H = np.concatenate([np.full(len(v[2]["fy"]), v[1]) for v in items])
            fm_phys = tm.fmedial_term(fy, mx, H, tm.MX_SIGN_PHYSICAL)
            fm_pipe = tm.fmedial_term(fy, mx, H, tm.MX_SIGN_PIPELINE)
            afy = np.abs(fy)
            row = dict(split=split, lab=lab, kept_frames=len(fy), kept_stance_frames=int(st.sum()))
            if st.any():
                row.update(fy_stance_median=float(np.median(afy[st])), fy_stance_p95=float(np.percentile(afy[st], 95)),
                           fy_stance_max=float(afy[st].max()), fy_stance_gt5_frac=float((afy[st] > 5).mean()),
                           fmed_phys_stance_median=float(np.median(fm_phys[st])),
                           fmed_phys_stance_p95=float(np.percentile(fm_phys[st], 95)),
                           fmed_phys_stance_gt14_frac=float((fm_phys[st] > 1.4).mean()),
                           fmed_pipe_stance_median=float(np.median(fm_pipe[st])),
                           fmed_pipe_stance_p95=float(np.percentile(fm_pipe[st], 95)),
                           mx_stance_median_medial=float(np.median(-mx[st])))
            row.update(fy_all_median=float(np.median(afy)), fy_all_p95=float(np.percentile(afy, 95)),
                       fmed_phys_all_median=float(np.median(fm_phys)), fmed_phys_all_p95=float(np.percentile(fm_phys, 95)))
            dist.append(row)
    dist = pd.DataFrame(dist)
    dist.to_csv(os.path.join(a.out, "label_distribution.csv"), index=False)
    # frames / trials / people kept per lab and split
    g = []
    for (split, lab), x in list(per.groupby(["split", "lab"])) + [((s, "ALL"), per[per.split == s]) for s in ("train", "val")]:
        g.append(dict(split=split, lab=lab, trials=len(x), people=x.person.nunique(),
                      trials_with_kept=int((x.n_kept > 0).sum()), people_with_kept=x[x.n_kept > 0].person.nunique(),
                      fully_masked=int(x.fully_masked.sum()), loader_rejected=int((~x.loaded).sum()),
                      frames_100hz=int(x.n_frames.sum()), kept_frames=int(x.n_kept.sum()),
                      kept_frac=float(x.n_kept.sum() / max(x.n_frames.sum(), 1)),
                      stance_frames=int(x.n_stance.sum()), kept_stance_frames=int(x.n_kept_stance.sum()),
                      kept_stance_frac=float(x.n_kept_stance.sum() / max(x.n_stance.sum(), 1))))
    g = pd.DataFrame(g)
    g.to_csv(os.path.join(a.out, "kept_by_lab.csv"), index=False)
    # trial-level filters on top of the masks: trials and kept frames they would still drop
    filt = []
    for split in ("train", "val", "both"):
        x = per if split == "both" else per[per.split == split]
        for judged in ("kept_frames", "all_jr_rows"):
            for mp_ in PEAKS:
                for mm in MOMENTS:
                    if judged == "kept_frames":     # NaN (no kept frame) compares False: a fully masked trial passes
                        drop = (x.peak_kept_frames > mp_) | (x.mx_kept_frames > mm)
                    else:
                        drop = (x.peak_all > mp_) | (x.mx_all > mm)
                    dk = drop & (x.n_kept > 0)
                    filt.append(dict(split=split, judged_on=judged, max_peak=mp_, max_moment=mm,
                                     trials_dropped=int(drop.sum()), trials_with_kept_dropped=int(dk.sum()),
                                     kept_frames_dropped=int(x.n_kept[drop].sum()),
                                     kept_frames_dropped_frac=float(x.n_kept[drop].sum() / max(x.n_kept.sum(), 1)),
                                     by_lab="; ".join(f"{k} {v}" for k, v in x[dk].lab.value_counts().items()),
                                     # trials dropped because of masked frames only (kept JR rows pass)
                                     dropped_for_masked_rows_only=int((dk & (x.peak_kept <= mp_)
                                                                       & (x.mx_kept <= mm)).sum())))
    filt = pd.DataFrame(filt)
    # the loader at 6 BW / 0.15 BW*H against the per-trial maxima (kept-frame judging)
    pred6 = ((per.peak_kept_frames > 6.0) | (per.mx_kept_frames > 0.15)).to_numpy()
    info_check = dict(loader_rejected_at_6_015=int(per.rejected_at_6_015.sum()),
                      predicted_from_kept_frames=int(pred6.sum()),
                      agree=bool(np.array_equal(per.rejected_at_6_015.to_numpy(), pred6)),
                      kept_frames_at_6_015=int(per.n_kept_at_6_015.sum()),
                      kept_frames_at_6_015_frac=float(per.n_kept_at_6_015.sum() / max(per.n_kept.sum(), 1)))
    print(json.dumps(info_check))
    filt.to_csv(os.path.join(a.out, "trial_filters.csv"), index=False)
    kept = per[per.n_kept > 0]
    q = {c: kept[c].quantile([0.5, 0.95, 0.99, 1.0]).round(3).tolist()
         for c in ("peak_all", "peak_kept", "peak_kept_frames", "mx_all", "mx_kept", "mx_kept_frames")}
    peak0, mom0 = tm.label_filter_defaults(cfg)
    info = dict(discovery=disc, runtime_s=round(time.time() - t0, 1), workers=a.workers, input_set=INPUT_SET,
                filters_loaded=dict(max_peak_bw=peak0, max_moment_bwh=mom0),
                quantiles_over_trials_with_kept_frames=q, loader_check_6_015=info_check)
    json.dump(info, open(os.path.join(a.out, "run_info.json"), "w"), indent=1)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(g.to_string(index=False))
    print(dist.round(3).to_string(index=False))
    print(filt[(filt.split == "both") & filt.max_moment.isin([0.15, 1e9])].round(4).to_string(index=False))
    print(json.dumps(q))
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()

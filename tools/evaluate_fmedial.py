#!/usr/bin/env python3
"""F_medial evaluation harness (thesis Task 1).

Scores one or more GRF-free MLP checkpoints against OpenSim labels on the
held-out test split, and reports accuracy of the derived medial-compartment load

    F_medial = phi * F_jcf + M_x * H / d          (F_jcf = |F_y|, compression)

together with its two parts, so errors that cancel inside F_medial stay visible.

Usage (run from the repo root):

    python tools/evaluate_fmedial.py --ckpt models/mlp-grffree-base/mlp-grffree-base.pt \
        models/runs/mlp/best_model_mlp_walking_qinst_notrans_pelvisrel_winpos_contact_w20_fmW0p5_e100_2.pt \
        --exclude han fregly --name sept_vs_paper

    # phi / d sensitivity, seed ensembles, other trial folders:
    python tools/evaluate_fmedial.py --ckpt A.pt --phi 0.4 0.5 0.6 --d-mm 40 45 50
    python tools/evaluate_fmedial.py --ckpt seed1.pt seed2.pt seed3.pt --ensemble
    python tools/evaluate_fmedial.py --ckpt A.pt --dirs jcf/external/william_2026-09-23 --pelvis running

    # score any checkpoint against the SO+JR labels (jcf_output_sojr/) instead of the ones it was trained on:
    python tools/evaluate_fmedial.py --ckpt A.pt --label-subdir jcf_output_sojr --max-peak <BW> --max-moment <BW*H>

    # only the frames that pass the SO+JR label QC (jcf_output_sojr/label_mask.npz, keep_r); the peak/moment
    # filters are then off unless given (1000 = off) and a given value is judged on the kept frames only:
    python tools/evaluate_fmedial.py --ckpt A.pt --label-subdir jcf_output_sojr --label-mask \
        --max-peak 1000 --max-moment 1000 ...

    # the pre-Nov-2026 numbers (M_x taken as +mx of walker_knee_r):
    python tools/evaluate_fmedial.py --ckpt A.pt --mx-convention pipeline ...

M_x sign (--mx-convention): every network predicts the raw walker_knee_r mx (the label). F_medial needs the medial
moment, which is -mx at walker_knee_r (`physical`, the default; analysis/thesis/knee_mx_sign.py). The pipeline and
the ISRR 2026 numbers used +mx (`pipeline`). The chosen sign multiplies the label and every model's raw M_x output
alike, before any metric, so the M_x columns are the medial-positive moment under that convention and a checkpoint's
own training sign (mx_medial_sign, shown in the README) does not enter the scoring.

Label mask (--label-mask, off by default): score only the frames whose label passes the SO+JR label QC
(<trial>/<--label-mask-subdir>/label_mask.npz, keep_r; analysis/thesis/so_label_qc.py masks), carried onto the
scored frame grid exactly as training does (train_mlp.carry_label_mask). Windows are built on the full trial first.
Folders without a mask stop the run; trials whose mask keeps no frame go to rejected.csv. With --label-mask the
trial-level --max-peak / --max-moment filters default to off (train_mlp.MASKED_MAX_PEAK_BW / MASKED_MAX_MOMENT_BWH)
and, when given, are judged on the kept frames' labels (train_mlp.kept_labels_pass), as in a *_sojrqc training run.

Labels: each checkpoint records the label folder it was trained on (label_subdir; older checkpoints: their input
set's, i.e. jcf_output). Without --label-subdir all checkpoints must agree on it; --label-subdir scores every
checkpoint against that folder's BatchJCF_JointReaction_ReactionLoads.sto. A --split run stops when some folders have
the stored jcf_output/ labels but none in that folder (unfinished relabel) unless --allow-partial-labels is given. A
trial whose label file holds a NaN/inf is rejected ('non-finite labels' in rejected.csv).

Outputs go to results/fmedial_eval/<name>/:
    summary.csv   frame-pooled metrics per model x activity x lab (and ALL), all frames and right stance
    persons.csv   the same per held-out person
    trials.csv    the same per trial
    sweep.csv     phi / d sweep: F_medial error and limit-crossing agreement
    rejected.csv  trials the label filters dropped, with the reason
    README.md     the run, the checkpoints, the headline table and the paper check

Data conventions follow training: labels and features come from the same
loaders train_mlp.py uses (per-trial pelvis centering, trial-level rejection of
|M_x| above --max-moment BW*H and resultant peaks above --max-peak BW). Right stance = right
foot vertical GRF above 5 % of body weight.
"""
import argparse
import datetime
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

PAPER_FY_MAE_WALKING = 0.153            # ISRR 2026, GRF-free MLP, walking test set (Carter + Moore)
CONTACT_BW = 0.05
SPLIT_ROOTS = {"test": "jcf/full_duration/testing", "train": "jcf/full_duration/training"}
LABEL_FILE = "BatchJCF_JointReaction_ReactionLoads.sto"
LABEL_FILE_DEFAULT_SUBDIR = "jcf_output"   # stored labels; jcf_output_sojr/ = SO+JR (analysis/thesis/gc_knee_jr.py)
MX_CONVENTION_SIGN = {"physical": -1.0, "pipeline": 1.0}   # factor on the raw walker_knee_r mx (= train_mlp.MX_CONVENTIONS)
MASK_SUBDIR_DEFAULT = "jcf_output_sojr"                     # where so_label_qc.py writes label_mask.npz


def medial_moment(mx, convention="physical"):
    """Medial-positive M_x (BW*H) from the raw walker_knee_r mx of a label or a network output."""
    if convention not in MX_CONVENTION_SIGN:
        raise ValueError(f"convention must be one of {sorted(MX_CONVENTION_SIGN)}, got {convention!r}")
    return MX_CONVENTION_SIGN[convention] * np.asarray(mx, dtype=float)


def to_convention(arr, convention="physical"):
    """Copy of a [T, 4] (Fx, Fy, Fz, Mx) label or prediction array with column 3 the medial-positive moment."""
    out = np.array(arr, dtype=np.float64, copy=True)
    out[:, 3] = medial_moment(out[:, 3], convention)
    return out


# ─── Formula and small helpers (pure functions, unit-tested) ─────────────────

def fmedial(Fjcf, Madd, phi=0.5, d_mm=45.0, H_m=1.7, m_kg=70.0, units="normalized"):
    """Medial-compartment load, F_medial = phi * F_jcf + M_add / d.

    units="normalized": Fjcf in BW (compression positive), Madd in BW*H  -> BW.
    units="physical":   Fjcf in N  (compression positive), Madd in N*m   -> N.
    The two agree: physical = normalized * m_kg * 9.81 for the same motion.
    """
    d = np.asarray(d_mm, dtype=float) / 1000.0
    if units == "normalized":
        return phi * np.asarray(Fjcf, dtype=float) + np.asarray(Madd, dtype=float) * H_m / d
    if units == "physical":
        return phi * np.asarray(Fjcf, dtype=float) + np.asarray(Madd, dtype=float) / d
    raise ValueError(f"units must be 'normalized' or 'physical', got {units!r}")


def person_id(path):
    """Person behind a trial folder: strips _tNN, _rNN and _splitN."""
    s = re.sub(r"_t\d+(_r\d+)?$", "", os.path.basename(os.path.normpath(path)))
    return re.sub(r"_split\d+$", "", s)


def lab_of(path):
    return os.path.basename(os.path.normpath(path)).split("_")[0]


def basic_metrics(g, p):
    """MAE, RMSE, bias (pred - GT) and Pearson r; NaN where undefined."""
    g = np.asarray(g, dtype=float); p = np.asarray(p, dtype=float)
    n = len(g)
    if n == 0:
        return dict(n=0, mae=np.nan, rmse=np.nan, bias=np.nan, r=np.nan)
    e = p - g
    r = np.nan
    if n >= 3 and np.std(g) > 1e-12 and np.std(p) > 1e-12:
        r = float(np.corrcoef(g, p)[0, 1])
    return dict(n=n, mae=float(np.mean(np.abs(e))), rmse=float(np.sqrt(np.mean(e ** 2))),
                bias=float(np.mean(e)), r=r)


def sign_check(fy_gt, stance, min_frac=0.95):
    """F_jcf = -F_y should be positive (compression) in stance. Returns (fraction, ok)."""
    fy = np.asarray(fy_gt)[np.asarray(stance, dtype=bool)]
    if len(fy) == 0:
        return np.nan, True
    frac = float(np.mean(fy < 0))
    return frac, frac >= min_frac


def frame_block(gt, pred, H, stance, phi, d_mm):
    """Per-frame quantities for one trial and one model."""
    fy_g, fy_p, mx_g, mx_p = gt[:, 1], pred[:, 1], gt[:, 3], pred[:, 3]
    fj_g, fj_p = np.abs(fy_g), np.abs(fy_p)
    return dict(
        fy_g=fy_g, fy_p=fy_p, mx_g=mx_g, mx_p=mx_p,
        force_g=phi * fj_g, force_p=phi * fj_p,
        moment_g=fmedial(0, mx_g, 0, d_mm, H), moment_p=fmedial(0, mx_p, 0, d_mm, H),
        fmed_g=fmedial(fj_g, mx_g, phi, d_mm, H), fmed_p=fmedial(fj_p, mx_p, phi, d_mm, H),
        fjcf_g=fj_g, fjcf_p=fj_p, stance=np.asarray(stance, dtype=bool), H=np.full(len(fy_g), H))


def block_metrics(b, mask):
    """All metrics for the frames in mask."""
    out = {}
    for key, (g, p) in {"fy": ("fy_g", "fy_p"), "mx": ("mx_g", "mx_p"), "force_part": ("force_g", "force_p"),
                        "moment_part": ("moment_g", "moment_p"), "fmed": ("fmed_g", "fmed_p")}.items():
        m = basic_metrics(b[g][mask], b[p][mask])
        for k in ("mae", "rmse", "bias", "r"):
            out[f"{key}_{k}"] = m[k]
    out["n_frames"] = int(mask.sum())
    for side in ("g", "p"):
        fm, fj = b[f"fmed_{side}"][mask], b[f"fjcf_{side}"][mask]
        lab = "gt" if side == "g" else "pred"
        out[f"invalid_frac_{lab}"] = float(np.mean((fm < 0) | (fm > fj))) if len(fm) else np.nan
        ok = fj > 0.2
        out[f"mr_median_{lab}"] = float(np.median(fm[ok] / fj[ok])) if ok.any() else np.nan
    return out


def peak_metrics(b):
    s = b["stance"]
    if not s.any():
        return dict(peak_gt=np.nan, peak_pred=np.nan, peak_err=np.nan)
    pg, pp = float(b["fmed_g"][s].max()), float(b["fmed_p"][s].max())
    return dict(peak_gt=pg, peak_pred=pp, peak_err=pp - pg)


def concat_blocks(blocks):
    return {k: np.concatenate([b[k] for b in blocks]) for k in blocks[0]}


# ─── Data and models ──────────────────────────────────────────────────────────

def load_model(path, device):
    from train_mlp import JCF_MLP, INPUT_SETS
    ck = torch.load(path, map_location=device, weights_only=True)
    iset = ck.get("input_set", "full")
    if iset not in INPUT_SETS or not INPUT_SETS[iset].get("uses_com"):
        raise SystemExit(f"{path}: input_set {iset!r} is not a GRF-free (COM) variant; this harness evaluates that family only")
    if int(ck.get("n_outputs", 4)) < 4:
        raise SystemExit(f"{path}: predicts {ck.get('n_outputs')} channels, M_x is needed")
    m = JCF_MLP(n_features=ck["n_features"], n_outputs=4, hidden=ck.get("hidden", 128), dropout=0.0,
                activation=ck.get("activation", "relu"))
    m.load_state_dict(ck["model_state_dict"]); m.to(device).eval()
    prov = ck.get("provenance") or {}
    from train_mlp import label_subdir_of
    info = dict(path=path, name=os.path.splitext(os.path.basename(path))[0], input_set=iset,
                activity=ck.get("activity"), fmedial_weight=ck.get("fmedial_weight", 0.0),
                epoch=int(ck.get("epoch", -1)) + 1, git=(prov.get("git") or {}).get("commit"),
                exclude=(prov.get("call_args") or {}).get("exclude"),
                label_subdir=ck.get("label_subdir") or label_subdir_of(INPUT_SETS[iset]),
                mx_medial_sign=ck.get("mx_medial_sign"), label_mask=ck.get("label_mask"))
    return m, ck["input_mean"].to(device).float(), ck["input_std"].to(device).float(), info


def resolve_label_subdir(trained, override=None):
    """Label folder to score against: override (--label-subdir), else the one all checkpoints were trained on."""
    if override:
        return override.strip("/")
    found = sorted(set(trained))
    if len(found) > 1:
        raise SystemExit(f"the checkpoints were trained on different labels ({', '.join(found)}); "
                         "pass --label-subdir to choose the labels they are all scored against")
    return found[0] if found else LABEL_FILE_DEFAULT_SUBDIR


def trial_dirs(args, label_subdir=LABEL_FILE_DEFAULT_SUBDIR, skipped=None):
    """(activity, folder) pairs to score: folders with <label_subdir>/ labels. A folder with the stored jcf_output/
    labels but none in label_subdir/ is appended to skipped (when given) instead."""
    out = []
    if args.dirs:
        for d in args.dirs:
            out.append(("external", d.rstrip("/")))
        return out
    root = SPLIT_ROOTS[args.split]
    for act in args.activity:
        base = os.path.join(REPO, root, act)
        for name in sorted(os.listdir(base)) if os.path.isdir(base) else []:
            d = os.path.join(base, name)
            if not os.path.isdir(d):
                continue
            lab = name.split("_")[0]
            if args.exclude and lab in args.exclude:
                continue
            if args.labs and lab not in args.labs:
                continue
            if not os.path.exists(_label_path(d, label_subdir)):
                if skipped is not None and os.path.exists(_label_path(d, LABEL_FILE_DEFAULT_SUBDIR)):
                    skipped.append((act, d))
                continue
            out.append((act, d))
    return out


def _label_path(d, label_subdir):
    """<d>/<label_subdir>/BatchJCF_JointReaction_ReactionLoads.sto, or the FullDuration_ file (train_cnn.jr_label_path)."""
    from train_cnn import jr_label_path
    return jr_label_path(d, label_subdir)


def _read_sto(path):
    """OpenSim table with every column as float (train_cnn.load_sto: a padded '-nan' or other text becomes NaN)."""
    from train_cnn import load_sto
    with open(path) as f:
        lines = f.read().split("\n")
    hdr = next(i for i, l in enumerate(lines) if l.strip() == "endheader") + 1
    return load_sto(path, skiprows=hdr)


def rejection_reason(d, max_peak, foot_points=(), label_subdir=LABEL_FILE_DEFAULT_SUBDIR, max_moment=0.15,
                     mask_subdir=None):
    """Why the training loaders dropped a trial (best effort). mask_subdir (--label-mask): the peak and moment tests
    look at the JR rows its label_mask.npz keeps, as the loaders do (on the carried kept frames)."""
    try:
        meta = json.load(open(os.path.join(d, "metadata.json")))
    except Exception:
        return "missing or unreadable metadata.json"
    if not os.path.exists(os.path.join(d, "com_features.npz")):
        return "missing com_features.npz"
    from pipeline.compute_com_features import foot_point_filename
    for fp in foot_points:
        if fp is not None and not os.path.exists(os.path.join(d, foot_point_filename(fp))):
            return f"missing {foot_point_filename(fp)}"
    label_path = _label_path(d, label_subdir)
    if not os.path.exists(label_path):
        return f"no {label_subdir}/ labels"
    try:
        from train_cnn import nonfinite_rows
        j = _read_sto(label_path)
        n_bad = nonfinite_rows(j)
        if n_bad:
            return f"non-finite labels ({n_bad} of {len(j)} rows in {label_subdir}/)"
        BW = float(meta["mass_kg"]) * 9.81; H = float(meta.get("height_m", 1.7))
        pre = "walker_knee_r_on_tibia_r_in_tibia_r_"
        mx = np.abs(j[pre + "mx"].to_numpy()) / (BW * H)
        res = np.sqrt(sum(j[pre + c].to_numpy() ** 2 for c in ("fx", "fy", "fz"))) / BW
        where = ""
        if mask_subdir is not None:
            from train_mlp import LABEL_MASK_FILE, LABEL_MASK_KEY
            with np.load(os.path.join(d, mask_subdir, LABEL_MASK_FILE)) as z:
                keep = np.asarray(z[LABEL_MASK_KEY], bool)
            if len(keep) != len(j):
                return f"label mask does not match the {label_subdir}/ label rows"
            if not keep.any():
                return "label mask keeps no frame"
            mx, res, where = mx[keep], res[keep], " on QC-kept rows"
        if mx.max() > max_moment:
            return f"|Mx| peak {mx.max():.2f} BW*H > {max_moment:g}{where} (label corruption filter)"
        if res.max() > max_peak:
            return f"resultant peak {res.max():.1f} BW > {max_peak:g}{where}"
    except Exception as e:
        return f"labels unreadable ({type(e).__name__})"
    return "rejected by loader (length, alignment or missing channels)"


def sampling_rate(d, T):
    try:
        ik = pd.read_csv(os.path.join(d, "ik_results.mot"), sep=r"\s+", skiprows=6)
        t = ik["time"].to_numpy()
        return (T - 1) / (t[-1] - t[0]) if t[-1] > t[0] else 100.0
    except Exception:
        return 100.0


def features_for(d, cfg, args):
    """Windowed features + labels + frame meta for one trial, exactly as train_mlp.FrameDataset builds them.

    With --deploy-features every checkpoint gets deployment-style inputs (exact time alignment, 20 frames at 100 Hz,
    as the optimizer and the PN3 driver feed them), whatever input set it was trained with. Labels come from cfg's
    label_subdir (main() sets it to the resolved --label-subdir)."""
    from train_mlp import _build_com_features, _windowize_positions_only, _windowize, LABEL_MASK_FILE
    if getattr(args, "deploy_features", False):
        cfg = dict(cfg, time_exact=True, resample_hz=100.0)
    # the frame mask is a scoring choice here (--label-mask), not a property of the checkpoint's input set
    cfg = {k: v for k, v in cfg.items() if k not in ("label_mask", "label_mask_subdir")}
    if getattr(args, "label_mask", False):
        cfg.update(label_mask=LABEL_MASK_FILE, label_mask_subdir=getattr(args, "label_mask_subdir", MASK_SUBDIR_DEFAULT))
    built = _build_com_features(d, cfg, max_peak_bw=args.max_peak, min_length=1, predict_moment=True, return_meta=True,
                                max_moment_bwh=getattr(args, "max_moment", 0.15))
    if built is None:
        return None
    feat, labels, mass, fmeta = built
    if args.pelvis == "running" and not cfg.get("drop_pelvis_translation") and not cfg.get("pelvis_per_window_centering"):
        from inference_posture_surrogate import remove_pelvis_drift
        feat = feat.copy()
        fs = fmeta.get("fs") or sampling_rate(d, len(feat))
        feat[:, 3:6] = remove_pelvis_drift(feat[:, 3:6], fs_hz=fs).astype(feat.dtype)
    w = cfg.get("window", 1)
    if w > 1:
        if cfg.get("window_only_positions"):
            feat = _windowize_positions_only(feat, w, q_dim=cfg.get("q_dim", 16), n_pos=cfg.get("n_pos", 9),
                                             pelvis_t_cols=(3, 4, 5) if cfg.get("pelvis_per_window_centering") else None,
                                             heading_norm=cfg.get("heading_norm", False))
        else:
            feat = _windowize(feat, w, replicate_q=cfg.get("replicate_q", False))
    return feat, labels, fmeta


def right_stance(d, args, T, label_subdir=LABEL_FILE_DEFAULT_SUBDIR):
    """Right-foot contact from the same load_subject call the feature builder uses."""
    from train_cnn import load_subject
    # with --label-mask the peak/moment tests were already made on the kept frames (features_for); not on every row here
    lm = getattr(args, "label_mask", False)
    r = load_subject(d, lower_body_only=True, clean_features=True, include_mass=True, include_speed=False,
                     jcf_subdir=label_subdir, predict_moment=True, max_peak_bw=np.inf if lm else args.max_peak,
                     max_moment_bwh=np.inf if lm else getattr(args, "max_moment", 0.15), min_length=1,
                     native_time_grid=True)
    if r is None or r[0].shape[1] < 53 or len(r[0]) != T:
        return None
    return np.abs(r[0][:, 49]) > CONTACT_BW


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", nargs="+", required=True, help="GRF-free MLP checkpoint(s)")
    ap.add_argument("--split", choices=list(SPLIT_ROOTS), default="test")
    ap.add_argument("--dirs", nargs="+", default=None, help="evaluate these trial folders instead of a split")
    ap.add_argument("--activity", nargs="+", default=["walking", "static"], choices=["walking", "static"])
    ap.add_argument("--exclude", nargs="+", default=None, help="labs to skip, e.g. han fregly")
    ap.add_argument("--labs", nargs="+", default=None, help="only these labs")
    ap.add_argument("--max-peak", type=float, default=None,
                    help="drop trials whose right-knee force resultant exceeds this many BW in any frame (default 6; "
                         "with --label-mask: off (1000) by default, and judged on the kept frames)")
    ap.add_argument("--max-moment", type=float, default=None,
                    help="drop trials whose |M_x| exceeds this many BW*H in any frame (label corruption filter; "
                         "default 0.15; with --label-mask: off (1000) by default, and judged on the kept frames)")
    ap.add_argument("--label-subdir", default=None,
                    help="score against <trial>/<this>/BatchJCF_JointReaction_ReactionLoads.sto, e.g. jcf_output "
                         "(stored) or jcf_output_sojr (SO+JR); default: the labels the checkpoints were trained on")
    ap.add_argument("--mx-convention", choices=sorted(MX_CONVENTION_SIGN), default="physical",
                    help="medial moment from the raw walker_knee_r mx, for labels and model outputs alike: physical "
                         "= -mx (default), pipeline = +mx (the pre-Nov-2026 numbers)")
    ap.add_argument("--label-mask", action="store_true",
                    help="score only frames whose label passes the SO+JR QC (<trial>/<--label-mask-subdir>/"
                         "label_mask.npz, keep_r); default off")
    ap.add_argument("--label-mask-subdir", default=MASK_SUBDIR_DEFAULT,
                    help=f"--label-mask: folder of label_mask.npz (default {MASK_SUBDIR_DEFAULT}; its time column must "
                         "match the scored label file)")
    ap.add_argument("--phi", type=float, nargs="+", default=[0.5])
    ap.add_argument("--d-mm", type=float, nargs="+", default=[45.0])
    ap.add_argument("--limit", type=float, default=1.4, help="pain limit in BW for the crossing agreement")
    ap.add_argument("--pelvis", choices=["trial", "running"], default="trial",
                    help="trial = per-trial centering as in training; running = 1.2 s drift removal (overground trajectories)")
    ap.add_argument("--deploy-features", action="store_true",
                    help="feed every checkpoint deployment-style inputs: exact time alignment, 20 frames at 100 Hz")
    ap.add_argument("--ensemble", action="store_true", help="also score the mean of all checkpoints, with a +-2 sigma coverage")
    ap.add_argument("--labels", nargs="+", default=None, help="short display names, one per --ckpt (e.g. paper september)")
    ap.add_argument("--flat-filter", action="store_true",
                    help="drop flat (standing-like) walking trials with training's filter; off by default, as in the paper's test")
    ap.add_argument("--flat-cv", type=float, default=0.25, help="--flat-filter: coefficient-of-variation threshold")
    ap.add_argument("--flat-dr", type=float, default=0.3, help="--flat-filter: range (max - min) threshold in BW")
    ap.add_argument("--flat-labels", default=None, metavar="SUBDIR",
                    help="--flat-filter judges this label folder (default: the labels being scored); jcf_output keeps "
                         "the stored-label trial selection when scoring against jcf_output_sojr")
    ap.add_argument("--allow-partial-labels", action="store_true",
                    help="--split runs: score the folders that have the labels even when some folders have only the "
                         "stored jcf_output/ labels (default: stop)")
    ap.add_argument("--max-trials", type=int, default=None, help="cap trials per activity (smoke tests)")
    ap.add_argument("--plots", action="store_true", help="one F_medial time-series plot per person")
    ap.add_argument("--name", default=None, help="results/fmedial_eval/<name> (default: timestamp)")
    ap.add_argument("--out-root", default=os.path.join(REPO, "results", "fmedial_eval"))
    args = ap.parse_args(argv)

    from train_mlp import INPUT_SETS, label_filter_defaults, LABEL_MASK_FILE
    import provenance as prov
    # 6 BW / 0.15 BW*H, or off with --label-mask (train_mlp.label_filter_defaults, as for a *_sojrqc training run)
    args.max_peak, args.max_moment = label_filter_defaults(dict(label_mask=LABEL_MASK_FILE if args.label_mask else None),
                                                           args.max_peak, args.max_moment)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = [load_model(p, device) for p in args.ckpt]
    if args.labels:
        if len(args.labels) != len(models):
            raise SystemExit(f"--labels needs {len(models)} names, got {len(args.labels)}")
        for m, lab in zip(models, args.labels):
            m[3]["name"] = lab
    names = [m[3]["name"] for m in models]
    if len(set(names)) != len(names):
        names = [f"{i}:{n}" for i, n in enumerate(names)]
        for (m, n) in zip(models, names):
            m[3]["name"] = n
    # headline numbers use the nominal phi = 0.5, d = 45 mm when they are in the sweep
    phi0 = 0.5 if 0.5 in args.phi else args.phi[0]
    d0 = 45.0 if 45.0 in args.d_mm else args.d_mm[0]
    out_dir = os.path.join(args.out_root, args.name or datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))

    label_subdir = resolve_label_subdir([m[3]["label_subdir"] for m in models], args.label_subdir)
    no_labels = []
    dirs = trial_dirs(args, label_subdir, skipped=no_labels)
    if no_labels:
        by_lab = pd.Series([lab_of(d) for _a, d in no_labels]).value_counts().sort_index()
        msg = (f"{len(no_labels)} folders have jcf_output/ labels but no {label_subdir}/ labels ("
               + ", ".join(f"{lab} {n}" for lab, n in by_lab.items()) + f"; e.g. {os.path.basename(no_labels[0][1])})")
        if not getattr(args, "allow_partial_labels", False):
            raise SystemExit(f"{msg}. Scoring the rest would change the trial set; finish the relabel (check: python "
                             "analysis/thesis/sojr_trainset.py --gate) or pass --allow-partial-labels.")
        print(f"WARNING (--allow-partial-labels): {msg}; skipped")
    rejected = []
    n_flat_removed = 0
    flat_sub = (args.flat_labels or label_subdir).strip("/")
    if args.flat_filter:
        from train_cnn import filter_flat_subjects
        walk = [d for act, d in dirs if act == "walking"]
        bad = []
        kept, _n = filter_flat_subjects(walk, cv_threshold=args.flat_cv, dr_threshold=args.flat_dr,
                                        jcf_subdir=flat_sub, nonfinite=bad)
        kept, bad = set(kept), set(bad)
        n_flat_removed = len(walk) - len(kept) - len(bad)
        for act, d in dirs:     # non-finite trials are reported, not counted as flat
            if d in bad:
                rejected.append(dict(activity=act, lab=lab_of(d), person=person_id(d), trial=os.path.basename(d),
                                     reason=f"non-finite labels in {flat_sub}/ (flat filter)"))
        dirs = [(act, d) for act, d in dirs if act != "walking" or d in kept]
        print(f"flat filter ({flat_sub}/ labels): removed {n_flat_removed} of {len(walk)} walking trials"
              + (f"; {len(bad)} more have non-finite labels" if bad else ""))
    if args.max_trials:
        capped, seen = [], {}
        for act, d in dirs:
            seen[act] = seen.get(act, 0) + 1
            if seen[act] <= args.max_trials:
                capped.append((act, d))
        dirs = capped
    if args.label_mask:
        from train_mlp import LABEL_MASK_FILE
        no_mask = [d for _a, d in dirs if not os.path.exists(os.path.join(d, args.label_mask_subdir, LABEL_MASK_FILE))]
        if no_mask:
            by_lab = pd.Series([lab_of(d) for d in no_mask]).value_counts().sort_index()
            raise SystemExit(f"--label-mask: {len(no_mask)} of {len(dirs)} folders have no {args.label_mask_subdir}/"
                             f"{LABEL_MASK_FILE} (" + ", ".join(f"{lab} {n}" for lab, n in by_lab.items())
                             + f"; e.g. {os.path.basename(no_mask[0])}); write them with: python -I "
                               "analysis/thesis/so_label_qc.py masks")
    n_mask_total = n_mask_kept = 0
    print(f"{len(dirs)} trial folders; {len(models)} checkpoint(s); labels {label_subdir}/; "
          f"phi={args.phi} d_mm={args.d_mm}; pelvis={args.pelvis}; M_x convention {args.mx_convention}"
          + (f"; frames: {args.label_mask_subdir}/label_mask.npz keep_r" if args.label_mask else ""))
    os.makedirs(out_dir, exist_ok=True)     # after the checks above, so a stopped run leaves no empty folder

    model_names = names + (["ensemble"] if args.ensemble and len(models) >= 2 else [])
    blocks = {n: [] for n in model_names}       # per-frame blocks at (phi0, d0), with meta
    raw = {n: [] for n in model_names}          # (gt, pred, H, stance, meta) for the sweep
    trial_rows, sign_rows = [], []
    plot_first = {}

    for i, (act, d) in enumerate(dirs):
        meta = dict(activity=act, lab=lab_of(d), person=person_id(d), trial=os.path.basename(d))
        try:
            H = float(json.load(open(os.path.join(d, "metadata.json"))).get("height_m", 1.7))
        except Exception:
            H = 1.7
        preds, gt, T, fmeta = {}, None, None, None
        cache = {}
        for (m, mu, sd, info), n in zip(models, names):
            iset = info["input_set"]
            if iset not in cache:
                cache[iset] = features_for(d, dict(INPUT_SETS[iset], label_subdir=label_subdir), args)
            built = cache[iset]
            if built is None:
                break
            feat, labels, fm_ = built
            with torch.no_grad():
                x = (torch.from_numpy(feat).float().to(device) - mu) / sd
                preds[n] = m(x).cpu().numpy()
            if gt is None:
                gt, T, fmeta = labels, len(labels), fm_
            elif len(labels) != T:
                raise RuntimeError(f"{d}: input sets disagree on trial length ({len(labels)} vs {T}); "
                                   "mixing native-rate and _t100 checkpoints needs --deploy-features")
        if gt is None or len(preds) < len(models):
            fps = [INPUT_SETS[info["input_set"]].get("foot_point") for (_m, _mu, _sd, info) in models]
            rejected.append(dict(meta, reason=rejection_reason(d, args.max_peak, fps, label_subdir, args.max_moment,
                                                               args.label_mask_subdir if args.label_mask else None)))
            continue
        stance = fmeta["r_contact"] if fmeta and fmeta.get("r_contact") is not None and len(fmeta["r_contact"]) == T \
            else right_stance(d, args, T, label_subdir)
        if stance is None:
            rejected.append(dict(meta, reason="could not align ground-force channels for the stance gate"))
            continue
        if args.label_mask:             # predictions were made on full windows; score the QC-passing frames only
            keep = np.asarray(fmeta["label_keep"], dtype=bool)
            n_mask_total += T
            n_mask_kept += int(keep.sum())
            if not keep.any():
                rejected.append(dict(meta, reason=f"label mask keeps no frame ({args.label_mask_subdir}/label_mask.npz)"))
                continue
            gt, stance, T = gt[keep], np.asarray(stance, bool)[keep], int(keep.sum())
            preds = {n: p[keep] for n, p in preds.items()}
            fmeta = dict(fmeta, t=np.asarray(fmeta["t"])[keep]) if fmeta.get("t") is not None else fmeta
        # one M_x convention for the label and every raw model output (column 3 becomes the medial-positive moment)
        gt = to_convention(gt, args.mx_convention)
        preds = {n: to_convention(p, args.mx_convention) for n, p in preds.items()}
        frac, ok = sign_check(gt[:, 1], stance)
        sign_rows.append(dict(meta, fy_negative_frac_in_stance=frac, ok=ok))
        if args.ensemble and len(models) >= 2:
            stack = np.stack([preds[n] for n in names])
            preds["ensemble"] = stack.mean(0)
            fm_members = np.stack([fmedial(np.abs(s[:, 1]), s[:, 3], phi0, d0, H) for s in stack])
            ens_sd = fm_members.std(0)
        for n in model_names:
            b = frame_block(gt, preds[n], H, stance, phi0, d0)
            if n == "ensemble":
                b["fmed_sd"] = ens_sd
            for k, v in meta.items():
                b[k] = np.full(T, v, dtype=object)
            blocks[n].append(b)
            raw[n].append((gt, preds[n], H, stance, meta))
            for subset, mask in (("all", np.ones(T, bool)), ("stance", stance)):
                if mask.sum() == 0:
                    continue
                row = dict(model=n, **meta, subset=subset, H=H, **block_metrics(b, mask))
                if subset == "stance":
                    row.update(peak_metrics(b))
                trial_rows.append(row)
        if args.plots and (act, meta["person"]) not in plot_first:
            plot_first[(act, meta["person"])] = (d, gt, preds, H, stance, act, (fmeta or {}).get("fs"),
                                                 (fmeta or {}).get("t"))
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(dirs)}", flush=True)

    if not trial_rows:
        reasons = pd.Series([r["reason"] for r in rejected]).value_counts() if rejected else None
        raise SystemExit("no trials could be evaluated" + ("" if reasons is None else "; rejected (first reasons): " +
                         "; ".join(f"{n} x {why}" for why, n in reasons.head(5).items())))

    # ── frame-pooled summaries ──
    def pooled(bl_list, mask_key):
        b = concat_blocks(bl_list)
        mask = np.ones(len(b["fy_g"]), bool) if mask_key == "all" else b["stance"]
        return b, mask

    summ_rows, person_rows = [], []
    for n in model_names:
        for act in sorted({bb["activity"][0] for bb in blocks[n]}):
            act_blocks = [bb for bb in blocks[n] if bb["activity"][0] == act]
            groups = {"ALL": act_blocks}
            for lab in sorted({bb["lab"][0] for bb in act_blocks}):
                groups[lab] = [bb for bb in act_blocks if bb["lab"][0] == lab]
            for gname, gl in groups.items():
                for subset in ("all", "stance"):
                    b, mask = pooled(gl, subset)
                    if mask.sum() == 0:
                        continue
                    row = dict(model=n, activity=act, group=gname, subset=subset,
                               n_trials=len(gl), n_persons=len({bb["person"][0] for bb in gl}), **block_metrics(b, mask))
                    persons = sorted({bb["person"][0] for bb in gl})
                    pm = []
                    for p in persons:
                        pb, pmask = pooled([bb for bb in gl if bb["person"][0] == p], subset)
                        if pmask.sum():
                            pm.append(basic_metrics(pb["fmed_g"][pmask], pb["fmed_p"][pmask])["mae"])
                    row["fmed_mae_person_mean"] = float(np.mean(pm)) if pm else np.nan
                    row["fmed_mae_person_sd"] = float(np.std(pm)) if len(pm) > 1 else np.nan
                    if n == "ensemble":
                        row["coverage_2sd"] = float(np.mean(np.abs(b["fmed_g"][mask] - b["fmed_p"][mask]) <= 2 * b["fmed_sd"][mask]))
                    if subset == "stance":
                        tr = [r for r in trial_rows if r["model"] == n and r["activity"] == act and r["subset"] == "stance"
                              and (gname == "ALL" or r["lab"] == gname)]
                        row["peak_err_mean"] = float(np.nanmean([r["peak_err"] for r in tr])) if tr else np.nan
                    summ_rows.append(row)
            for p in sorted({bb["person"][0] for bb in act_blocks}):
                pl = [bb for bb in act_blocks if bb["person"][0] == p]
                for subset in ("all", "stance"):
                    b, mask = pooled(pl, subset)
                    if mask.sum():
                        person_rows.append(dict(model=n, activity=act, lab=pl[0]["lab"][0], person=p, subset=subset,
                                                n_trials=len(pl), **block_metrics(b, mask)))

    # ── phi / d sweep ──
    sweep_rows = []
    for n in model_names:
        for act in sorted({r[4]["activity"] for r in raw[n]}):
            items = [r for r in raw[n] if r[4]["activity"] == act]
            for phi in args.phi:
                for d_mm in args.d_mm:
                    fg, fp = [], []
                    for gt_, pr_, H_, st_, _ in items:
                        fg.append(fmedial(np.abs(gt_[st_, 1]), gt_[st_, 3], phi, d_mm, H_))
                        fp.append(fmedial(np.abs(pr_[st_, 1]), pr_[st_, 3], phi, d_mm, H_))
                    fg, fp = np.concatenate(fg), np.concatenate(fp)
                    over_g, over_p = fg > args.limit, fp > args.limit
                    m = basic_metrics(fg, fp)
                    sweep_rows.append(dict(model=n, activity=act, phi=phi, d_mm=d_mm, n_stance_frames=len(fg),
                                           fmed_mae=m["mae"], fmed_bias=m["bias"], fmed_r=m["r"],
                                           over_limit_frac_gt=float(over_g.mean()), over_limit_frac_pred=float(over_p.mean()),
                                           crossing_recall=float((over_g & over_p).sum() / max(over_g.sum(), 1)),
                                           crossing_precision=float((over_g & over_p).sum() / max(over_p.sum(), 1))))

    # ── write ──
    pd.DataFrame(trial_rows).to_csv(os.path.join(out_dir, "trials.csv"), index=False)
    pd.DataFrame(person_rows).to_csv(os.path.join(out_dir, "persons.csv"), index=False)
    summ = pd.DataFrame(summ_rows); summ.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    pd.DataFrame(sweep_rows).to_csv(os.path.join(out_dir, "sweep.csv"), index=False)
    rej = pd.DataFrame(rejected, columns=["activity", "lab", "person", "trial", "reason"]); rej.to_csv(os.path.join(out_dir, "rejected.csv"), index=False)
    signs = pd.DataFrame(sign_rows)

    # paper check: Fy MAE over all walking frames, Carter + Moore
    paper = []
    for n in model_names:
        wb = [bb for bb in blocks[n] if bb["activity"][0] in ("walking", "external") and bb["lab"][0] in ("carter", "moore")]
        if wb:
            b = concat_blocks(wb)
            paper.append((n, basic_metrics(b["fy_g"], b["fy_p"])["mae"], len(wb)))

    if args.plots and plot_first:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        os.makedirs(os.path.join(out_dir, "plots"), exist_ok=True)
        for (_, p), (d, gt, preds, H, stance, act, fs, tf) in plot_first.items():
            t = (np.asarray(tf) - tf[0]) if tf is not None and len(tf) == len(gt) else \
                np.arange(len(gt)) / (fs or sampling_rate(d, len(gt)))
            fig, ax = plt.subplots(figsize=(11, 3.6))
            ax.fill_between(t, -0.5, 3.0, where=stance, color="#EEEEEE", lw=0, step="mid")
            ax.plot(t, fmedial(np.abs(gt[:, 1]), gt[:, 3], phi0, d0, H), color="#000000", lw=2, label="OpenSim")
            for n in model_names:
                ax.plot(t, fmedial(np.abs(preds[n][:, 1]), preds[n][:, 3], phi0, d0, H), lw=1.4, label=n[:40])
            ax.axhline(args.limit, color="#888888", ls="--", lw=1)
            ax.set_xlabel("time (s)"); ax.set_ylabel("F_medial (BW)"); ax.set_title(f"{p} · {os.path.basename(d)}", fontsize=10)
            ax.legend(fontsize=7, loc="upper right"); ax.grid(alpha=0.3); fig.tight_layout()
            fig.savefig(os.path.join(out_dir, "plots", f"{act}_{p}.png"), dpi=110); plt.close(fig)

    g = prov.git_info()
    lines = [f"# F_medial evaluation · {os.path.basename(out_dir)}", "",
             f"- created: {datetime.datetime.now().isoformat(timespec='seconds')}",
             f"- command: `{' '.join(['python'] + sys.argv)}`",
             f"- git: {g.get('commit')} ({'DIRTY' if g.get('dirty') else 'clean'})",
             f"- split: {'external folders' if args.dirs else args.split}; activities: {', '.join(args.activity)}; "
             f"excluded labs: {args.exclude or 'none'}; pelvis: {args.pelvis}; features: {'deployment-style (exact time, 100 Hz)' if args.deploy_features else 'as trained'}; phi={phi0}, d={d0} mm; limit {args.limit} BW"
             + (f"; flat filter (CV < {args.flat_cv:g} or range < {args.flat_dr:g} BW of the {flat_sub}/ labels) "
                f"removed {n_flat_removed} walking trials" if args.flat_filter else "; no flat filter"),
             f"- M_x convention: {args.mx_convention} (medial moment = "
             f"{'-' if args.mx_convention == 'physical' else '+'}mx of walker_knee_r, applied to the labels and to every "
             f"model's raw M_x output; the M_x columns are that medial moment"
             + ("; pre-Nov-2026 runs used pipeline)" if args.mx_convention == "physical" else
                "; reproduces the pre-Nov-2026 numbers, physically the lateral moment)"),
             f"- frames: " + (f"label mask {args.label_mask_subdir}/label_mask.npz (keep_r), {n_mask_kept} of "
                              f"{n_mask_total} frames of the scored trials kept "
                              f"({n_mask_kept / max(n_mask_total, 1):.1%})" if args.label_mask else
                              "all frames (no label mask)"),
             f"- labels: {label_subdir}/{LABEL_FILE}"
             + (" (--label-subdir)" if args.label_subdir else " (as trained)")
             + f"; trial filters: resultant peak <= {args.max_peak:g} BW, |M_x| <= {args.max_moment:g} BW*H"
             + (" on the QC-kept frames" if args.label_mask else "")
             + (f"; {len(no_labels)} folders with jcf_output/ labels but none in {label_subdir}/ were skipped" if no_labels else ""),
             f"- trials evaluated: {len({(r['activity'], r['trial']) for r in trial_rows})}; rejected by the label filters: {len(rej)} "
             f"({', '.join(f'{k} {v}' for k, v in rej['lab'].value_counts().items()) if len(rej) else 'none'})", "",
             "## Checkpoints", "", "| name | input set | trained on | trained labels | F_medial loss weight | "
             "loss M_x sign | epoch | trained at commit |",
             "|---|---|---|---|---|---|---|---|"]
    for (_, _, _, info) in models:
        sgn = info.get("mx_medial_sign")
        lines.append(f"| {info['name']} | {info['input_set']} | {info['activity']}, exclude {info['exclude'] or 'unknown'} | "
                     f"{info['label_subdir']}{' + ' + info['label_mask'] if info.get('label_mask') else ''} | "
                     f"{info['fmedial_weight']} | {'+1 (legacy, not stored)' if sgn is None else format(sgn, '+g')} | "
                     f"{info['epoch']} | {info['git'][:10] if info['git'] else 'no provenance'} |")
    lines += ["", "## Headline: frame-pooled error, all labs (BW; moment in BW·H)", "",
              "| model | activity | subset | persons | F_medial MAE | person-mean ± sd | F_y MAE | M_x MAE | force-part bias | moment-part bias | peak error |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in summ[summ["group"] == "ALL"].iterrows():
        lines.append(f"| {r['model'][:48]} | {r['activity']} | {r['subset']} | {r['n_persons']} | {r['fmed_mae']:.3f} | "
                     f"{r['fmed_mae_person_mean']:.3f} ± {0 if pd.isna(r['fmed_mae_person_sd']) else r['fmed_mae_person_sd']:.3f} | "
                     f"{r['fy_mae']:.3f} | {r['mx_mae']:.4f} | {r['force_part_bias']:+.3f} | {r['moment_part_bias']:+.3f} | "
                     f"{'' if pd.isna(r.get('peak_err_mean', np.nan)) else format(r['peak_err_mean'], '+.3f')} |")
    if paper:
        lines += ["", "## Paper check", "",
                  f"Axial-force (F_y) MAE over all walking frames, Carter + Moore test trials. The paper reports {PAPER_FY_MAE_WALKING} BW for its GRF-free MLP.", ""]
        lines += [f"- {n}: {mae:.3f} BW over {k} trials" for n, mae, k in paper]
    if len(signs):
        bad = signs[~signs["ok"]]
        lines += ["", "## Sign check", "",
                  f"F_y < 0 (compression) in right stance: median {signs['fy_negative_frac_in_stance'].median():.3f} of frames; "
                  f"{len(bad)} trial(s) below 95 %" + (": " + ", ".join(bad["trial"].head(10)) if len(bad) else ".")]
    lines += ["", "Files: summary.csv, persons.csv, trials.csv, sweep.csv, rejected.csv" + (", plots/" if args.plots else "")]
    open(os.path.join(out_dir, "README.md"), "w").write("\n".join(lines) + "\n")

    print("\n".join(lines[lines.index("## Headline: frame-pooled error, all labs (BW; moment in BW·H)"):]))
    print(f"\nwritten to {out_dir}")
    if len(signs) and (~signs["ok"]).mean() > 0.05:
        print("WARNING: more than 5 % of trials fail the compression sign check; inspect before using these numbers")
    return out_dir


if __name__ == "__main__":
    main()

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

Outputs go to results/fmedial_eval/<name>/:
    summary.csv   frame-pooled metrics per model x activity x lab (and ALL), all frames and right stance
    persons.csv   the same per held-out person
    trials.csv    the same per trial
    sweep.csv     phi / d sweep: F_medial error and limit-crossing agreement
    rejected.csv  trials the label filters dropped, with the reason
    README.md     the run, the checkpoints, the headline table and the paper check

Data conventions follow training: labels and features come from the same
loaders train_mlp.py uses (per-trial pelvis centering, trial-level rejection of
|M_x| > 0.15 BW*H and resultant peaks above --max-peak). Right stance = right
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
    info = dict(path=path, name=os.path.splitext(os.path.basename(path))[0], input_set=iset,
                activity=ck.get("activity"), fmedial_weight=ck.get("fmedial_weight", 0.0),
                epoch=int(ck.get("epoch", -1)) + 1, git=(prov.get("git") or {}).get("commit"),
                exclude=(prov.get("call_args") or {}).get("exclude"))
    return m, ck["input_mean"].to(device).float(), ck["input_std"].to(device).float(), info


def trial_dirs(args):
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
            if not os.path.isdir(d) or not os.path.exists(os.path.join(d, "jcf_output", "BatchJCF_JointReaction_ReactionLoads.sto")):
                continue
            lab = name.split("_")[0]
            if args.exclude and lab in args.exclude:
                continue
            if args.labs and lab not in args.labs:
                continue
            out.append((act, d))
    return out


def _read_sto(path):
    with open(path) as f:
        lines = f.read().split("\n")
    hdr = next(i for i, l in enumerate(lines) if l.strip() == "endheader") + 1
    return pd.read_csv(path, sep="\t", skiprows=hdr)


def rejection_reason(d, max_peak, foot_points=()):
    """Why the training loaders dropped a trial (best effort)."""
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
    try:
        j = _read_sto(os.path.join(d, "jcf_output", "BatchJCF_JointReaction_ReactionLoads.sto"))
        BW = float(meta["mass_kg"]) * 9.81; H = float(meta.get("height_m", 1.7))
        pre = "walker_knee_r_on_tibia_r_in_tibia_r_"
        mx = np.abs(j[pre + "mx"].to_numpy()) / (BW * H)
        res = np.sqrt(sum(j[pre + c].to_numpy() ** 2 for c in ("fx", "fy", "fz"))) / BW
        if mx.max() > 0.15:
            return f"|Mx| peak {mx.max():.2f} BW*H > 0.15 (label corruption filter)"
        if res.max() > max_peak:
            return f"resultant peak {res.max():.1f} BW > {max_peak:g}"
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
    """Windowed features + labels for one trial, exactly as train_mlp.FrameDataset builds them."""
    from train_mlp import _build_com_features, _windowize_positions_only, _windowize
    built = _build_com_features(d, cfg, max_peak_bw=args.max_peak, min_length=1, predict_moment=True)
    if built is None:
        return None
    feat, labels, mass = built
    if args.pelvis == "running" and not cfg.get("drop_pelvis_translation") and not cfg.get("pelvis_per_window_centering"):
        from inference_posture_surrogate import remove_pelvis_drift
        feat = feat.copy()
        feat[:, 3:6] = remove_pelvis_drift(feat[:, 3:6], fs_hz=sampling_rate(d, len(feat))).astype(feat.dtype)
    w = cfg.get("window", 1)
    if w > 1:
        if cfg.get("window_only_positions"):
            feat = _windowize_positions_only(feat, w, q_dim=cfg.get("q_dim", 16), n_pos=cfg.get("n_pos", 9),
                                             pelvis_t_cols=(3, 4, 5) if cfg.get("pelvis_per_window_centering") else None)
        else:
            feat = _windowize(feat, w, replicate_q=cfg.get("replicate_q", False))
    return feat, labels


def right_stance(d, args, T):
    """Right-foot contact from the same load_subject call the feature builder uses."""
    from train_cnn import load_subject
    r = load_subject(d, lower_body_only=True, clean_features=True, include_mass=True, include_speed=False,
                     predict_moment=True, max_peak_bw=args.max_peak, min_length=1, native_time_grid=True)
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
    ap.add_argument("--max-peak", type=float, default=6.0)
    ap.add_argument("--phi", type=float, nargs="+", default=[0.5])
    ap.add_argument("--d-mm", type=float, nargs="+", default=[45.0])
    ap.add_argument("--limit", type=float, default=1.4, help="pain limit in BW for the crossing agreement")
    ap.add_argument("--pelvis", choices=["trial", "running"], default="trial",
                    help="trial = per-trial centering as in training; running = 1.2 s drift removal (overground trajectories)")
    ap.add_argument("--ensemble", action="store_true", help="also score the mean of all checkpoints, with a +-2 sigma coverage")
    ap.add_argument("--labels", nargs="+", default=None, help="short display names, one per --ckpt (e.g. paper september)")
    ap.add_argument("--flat-filter", action="store_true",
                    help="drop flat (standing-like) walking trials with training's filter; off by default, as in the paper's test")
    ap.add_argument("--max-trials", type=int, default=None, help="cap trials per activity (smoke tests)")
    ap.add_argument("--plots", action="store_true", help="one F_medial time-series plot per person")
    ap.add_argument("--name", default=None, help="results/fmedial_eval/<name> (default: timestamp)")
    ap.add_argument("--out-root", default=os.path.join(REPO, "results", "fmedial_eval"))
    args = ap.parse_args(argv)

    from train_mlp import INPUT_SETS
    import provenance as prov

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
    os.makedirs(out_dir, exist_ok=True)

    dirs = trial_dirs(args)
    n_flat_removed = 0
    if args.flat_filter:
        from train_cnn import filter_flat_subjects
        walk = [d for act, d in dirs if act == "walking"]
        kept = filter_flat_subjects(walk)
        kept = set(kept[0] if isinstance(kept, tuple) else kept)
        n_flat_removed = len(walk) - len(kept)
        dirs = [(act, d) for act, d in dirs if act != "walking" or d in kept]
        print(f"flat filter: removed {n_flat_removed} of {len(walk)} walking trials")
    if args.max_trials:
        capped, seen = [], {}
        for act, d in dirs:
            seen[act] = seen.get(act, 0) + 1
            if seen[act] <= args.max_trials:
                capped.append((act, d))
        dirs = capped
    print(f"{len(dirs)} trial folders; {len(models)} checkpoint(s); phi={args.phi} d_mm={args.d_mm}; pelvis={args.pelvis}")

    model_names = names + (["ensemble"] if args.ensemble and len(models) >= 2 else [])
    blocks = {n: [] for n in model_names}       # per-frame blocks at (phi0, d0), with meta
    raw = {n: [] for n in model_names}          # (gt, pred, H, stance, meta) for the sweep
    trial_rows, rejected, sign_rows = [], [], []
    plot_first = {}

    for i, (act, d) in enumerate(dirs):
        meta = dict(activity=act, lab=lab_of(d), person=person_id(d), trial=os.path.basename(d))
        try:
            H = float(json.load(open(os.path.join(d, "metadata.json"))).get("height_m", 1.7))
        except Exception:
            H = 1.7
        preds, gt, T = {}, None, None
        cache = {}
        for (m, mu, sd, info), n in zip(models, names):
            iset = info["input_set"]
            if iset not in cache:
                cache[iset] = features_for(d, INPUT_SETS[iset], args)
            built = cache[iset]
            if built is None:
                break
            feat, labels = built
            with torch.no_grad():
                x = (torch.from_numpy(feat).float().to(device) - mu) / sd
                preds[n] = m(x).cpu().numpy()
            if gt is None:
                gt, T = labels, len(labels)
            elif len(labels) != T:
                raise RuntimeError(f"{d}: input sets disagree on trial length ({len(labels)} vs {T})")
        if gt is None or len(preds) < len(models):
            fps = [INPUT_SETS[info["input_set"]].get("foot_point") for (_m, _mu, _sd, info) in models]
            rejected.append(dict(meta, reason=rejection_reason(d, args.max_peak, fps)))
            continue
        stance = right_stance(d, args, T)
        if stance is None:
            rejected.append(dict(meta, reason="could not align ground-force channels for the stance gate"))
            continue
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
            plot_first[(act, meta["person"])] = (d, gt, preds, H, stance, act)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(dirs)}", flush=True)

    if not trial_rows:
        raise SystemExit("no trials could be evaluated")

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
        for (_, p), (d, gt, preds, H, stance, act) in plot_first.items():
            t = np.arange(len(gt)) / sampling_rate(d, len(gt))
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
             f"excluded labs: {args.exclude or 'none'}; pelvis: {args.pelvis}; phi={phi0}, d={d0} mm; limit {args.limit} BW"
             + (f"; flat filter removed {n_flat_removed} walking trials" if args.flat_filter else "; no flat filter"),
             f"- trials evaluated: {len({(r['activity'], r['trial']) for r in trial_rows})}; rejected by the label filters: {len(rej)} "
             f"({', '.join(f'{k} {v}' for k, v in rej['lab'].value_counts().items()) if len(rej) else 'none'})", "",
             "## Checkpoints", "", "| name | input set | trained on | F_medial loss weight | epoch | trained at commit |", "|---|---|---|---|---|---|"]
    for (_, _, _, info) in models:
        lines.append(f"| {info['name']} | {info['input_set']} | {info['activity']}, exclude {info['exclude'] or 'unknown'} | "
                     f"{info['fmedial_weight']} | {info['epoch']} | {info['git'][:10] if info['git'] else 'no provenance'} |")
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

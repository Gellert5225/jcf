#!/usr/bin/env python3
"""Which sign of the OpenSim knee moment M_x means MEDIAL compartment loading.

F_medial = phi*|F_y| + s*M_x*H/d needs s, the sign that makes M_x medial-positive. M_x is the JointReaction moment
of walker_knee_{r,l} "on tibia, in tibia": the load from the femur on the tibia at the knee centre, expressed in the
tibia body frame. In Rajagopal-family models every body frame has x anterior, y superior and z to the subject's RIGHT,
on both legs. The models mirror the geometry, not the axes. A compressive contact force F = (0, -f, 0) at the
condyle r = (0, 0, z_c) gives M_x = r_y F_z - r_z F_y = z_c * f. The medial condyle is at z < 0 on the right tibia
and at z > 0 on the left, so:

    physical  (settled here):          medial-positive M_x = -mx (walker_knee_r),  +mx (walker_knee_l)
    pipeline  (until 2026-11-10: train_cnn comment, train_mlp F_medial loss, tools/evaluate_fmedial,
               analysis/thesis/label_qc.MX_SIGN; now kept only as train_mlp's legacy mx_medial_sign +1 and the
               evaluators' --mx-convention pipeline):
                                       medial-positive M_x = +mx (walker_knee_r),  -mx (walker_knee_l)

The pipeline sign was asserted in a code comment (train_cnn.load_subject) and never tested. It is inverted. The
evidence, written by this script to results/thesis/labels_qc/mx_sign/:

  1. two_point   synthetic medial / lateral contact forces -> (F_y, M_x) -> F_medial recovers the medial force only
                 with the physical sign; the pipeline sign returns the lateral force (tests/test_knee_mx_sign.py).
  2. geometry    our own models (jcf_output/_model_with_reserves.osim, one per lab). The medial knee muscles
                 (gastrocnemius medialis origin on the medial femoral condyle; gracilis, semitendinosus and sartorius
                 insertions) sit at z < 0 on femur_r / tibia_r and at z > 0 on femur_l / tibia_l. The lateral ones
                 (gastrocnemius lateralis origin, biceps femoris short head insertion) sit on the other side.
  3. kaneda      Kaneda et al.'s StatOp_Val SO+JR output (Grand Challenge package) has, in the same table, the total
                 load on the tibia (tibial_plat_weld_*_on_tibia_*_in_tibia_*) and the medial and lateral condyle
                 contact forces. A least-squares fit mx = a*F_med + b*F_lat + c recovers the condyle lever arms with
                 a residual of about 1 N*m (SD of mx 8-16 N*m). a > 0 > b at SC's left knee, a < 0 < b at JW's
                 and DM's right knees.
  4. population  (--population) stance-mean M_x of SO+JR (both knees) and of the stored labels (right knee) on walking
                 folders, per lab: the share of folders whose stance-mean moment is medial under each sign.
  5. implant     every analysis/thesis/grand_challenge.py run writes mx_sign_check.csv: r of each source's
                 medial-positive M_x with the measured medial - lateral force. The README copies the --gc-run table.

Usage (repo root; reads files only, runs no OpenSim; --population with 4 workers takes a few minutes):
    python -I analysis/thesis/knee_mx_sign.py --out results/thesis/labels_qc/mx_sign [--population]
"""
import argparse
import glob
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

MX_CONVENTIONS = ("physical", "pipeline")
_PHYSICAL = {"r": -1.0, "l": 1.0}           # medial-positive M_x = _PHYSICAL[side] * raw mx
MEDIAL_MUSCLES = ("gasmed", "grac", "semiten", "sart")
LATERAL_MUSCLES = ("gaslat", "bfsh")
KANEDA = (("3GC", "l", "SC"), ("4GC", "r", "JW"), ("6GC", "r", "DM"))
GC_ROOT = os.path.join(REPO, "jcf", "external", "grand_challenge")
LABS = ("carter", "falisse", "fregly", "han", "moore", "tiziana")


def knee_side(joint):
    """'r' or 'l' from walker_knee_r / tibia_l / 'r' ..."""
    s = str(joint).strip().lower()
    side = s if s in ("r", "l") else s[-1] if s[-2:] in ("_r", "_l") else None
    if side is None:
        raise ValueError(f"cannot tell the side of {joint!r}")
    return side


def medial_mx_sign(joint, convention="physical"):
    """Factor that turns the raw JR mx of this knee into a medial-positive moment."""
    if convention not in MX_CONVENTIONS:
        raise ValueError(f"convention must be one of {MX_CONVENTIONS}, not {convention!r}")
    s = _PHYSICAL[knee_side(joint)]
    return s if convention == "physical" else -s


def two_point_reaction(f_med, f_lat, z_med, z_lat, x_med=0.0, x_lat=0.0):
    """(F_y, M_x) on the tibia about the knee centre from compressive condyle contact forces f >= 0 (N) acting on the
    tibia as (0, -f, 0) at (x, 0, z) (m), with M = sum r x F as OpenSim reports it."""
    f_med, f_lat = np.asarray(f_med, float), np.asarray(f_lat, float)
    m = np.zeros(np.broadcast(f_med, f_lat).shape + (3,))
    for f, x, z in ((f_med, x_med, z_med), (f_lat, x_lat, z_lat)):
        F = np.stack(np.broadcast_arrays(np.zeros_like(f), -f, np.zeros_like(f)), axis=-1)
        m = m + np.cross(np.array([x, 0.0, z]), F)
    return -(f_med + f_lat), m[..., 0]


def fmedial_n(fy, mx, sign, phi=0.5, d_m=0.045):
    """F_medial in N from F_y (N) and raw M_x (N*m) with the given medial sign."""
    return phi * np.abs(fy) + sign * np.asarray(mx, float) / d_m


# ─── Model geometry ───────────────────────────────────────────────────────────

def knee_geometry(osim_path):
    """z (m) of the medial and lateral knee-muscle path points on femur_s / tibia_s, relative to the walker_knee frame
    origin in that body, one row per point."""
    root = ET.parse(osim_path).getroot()
    centre = {}
    for j in root.iter("CustomJoint"):
        if not (j.get("name") or "").startswith("walker_knee_"):
            continue
        for fr in j.iter("PhysicalOffsetFrame"):
            body = (fr.findtext("socket_parent") or "").rsplit("/", 1)[-1]
            tr = fr.findtext("translation")
            if body and tr:
                centre[body] = float(tr.split()[2])
    rows = []
    for m in root.iter():
        if m.find("GeometryPath") is None:
            continue
        base, _, side = (m.get("name") or "").rpartition("_")
        group = "medial" if base in MEDIAL_MUSCLES else "lateral" if base in LATERAL_MUSCLES else None
        if group is None or side not in ("r", "l"):
            continue
        for pp in m.iter("PathPoint"):
            body = (pp.findtext("socket_parent_frame") or "").rsplit("/", 1)[-1]
            loc = pp.findtext("location")
            if body not in (f"femur_{side}", f"tibia_{side}") or not loc:
                continue
            rows.append(dict(muscle=m.get("name"), point=pp.get("name"), body=body, side=side, group=group,
                             z_rel=float(loc.split()[2]) - centre.get(body, 0.0)))
    return pd.DataFrame(rows)


def geometry_check(paths):
    """Per model x side: median z of the medial and of the lateral points, and the medial sign of M_x they imply
    (M_x = z_med * f for a compressive medial contact)."""
    rows = []
    for lab, p in paths.items():
        g = knee_geometry(p)
        for side, gs in g.groupby("side"):
            zm = float(gs.loc[gs.group == "medial", "z_rel"].median())
            zl = float(gs.loc[gs.group == "lateral", "z_rel"].median())
            rows.append(dict(lab=lab, folder=os.path.basename(os.path.dirname(os.path.dirname(p))), side=side,
                             n_medial=int((gs.group == "medial").sum()), n_lateral=int((gs.group == "lateral").sum()),
                             z_medial_m=zm, z_lateral_m=zl,
                             medial_all_one_side=bool(np.all(np.sign(gs.loc[gs.group == "medial", "z_rel"]) == np.sign(zm))),
                             lateral_all_one_side=bool(np.all(np.sign(gs.loc[gs.group == "lateral", "z_rel"]) == np.sign(zl))),
                             implied_medial_sign=float(np.sign(zm)) if np.sign(zm) == -np.sign(zl) else np.nan,
                             physical_sign=medial_mx_sign(side), pipeline_sign=medial_mx_sign(side, "pipeline")))
    return pd.DataFrame(rows)


def one_model_per_lab(repo=REPO):
    out = {}
    for lab in LABS:
        g = sorted(glob.glob(os.path.join(repo, "jcf", "full_duration", "*", "walking", f"{lab}_*", "jcf_output",
                                          "_model_with_reserves.osim")))
        if g:
            out[lab] = g[0]
    return out


# ─── Kaneda's condyle split ───────────────────────────────────────────────────

def _read_ws(path):
    with open(path) as f:
        for i, line in enumerate(f):
            if line.strip().lower() == "endheader":
                break
    return pd.read_csv(path, sep=r"\s+", skiprows=i + 1)


def kaneda_frames(root=GC_ROOT):
    """Rows of Kaneda et al.'s instrumented-knee JR files: raw tibia mx, fy (N, N*m) and the medial / lateral condyle
    compressive forces (N, positive)."""
    out = []
    for code, side, subj in KANEDA:
        for p in sorted(glob.glob(os.path.join(root, "StatOp_Val_Data", code, "SO_JRA", "*", "instKnee",
                                               "results_JointReaction_JointRxn_ReactionLoads.sto"))):
            j = _read_ws(p)
            pre = f"tibial_plat_weld_{side}_on_tibia_{side}_in_tibia_{side}_"
            med = f"med_cond_joint_{side}_on_med_cond_{side}_in_med_cond_{side}_fy"
            lat = f"lat_cond_joint_{side}_on_lat_cond_{side}_in_lat_cond_{side}_fy"
            if not all(c in j for c in (pre + "mx", pre + "fy", med, lat)):
                continue
            out.append(pd.DataFrame(dict(subject=subj, code=code, side=side, window=os.path.basename(os.path.dirname(
                os.path.dirname(p))), mx=j[pre + "mx"], fy=j[pre + "fy"], f_med=-j[med], f_lat=-j[lat])))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def kaneda_check(frames, min_total_n=200.0):
    """Per knee: r(mx, F_med - F_lat), least squares mx = a F_med + b F_lat + c and the medial sign it implies."""
    rows = []
    for (subj, code, side), g in frames.groupby(["subject", "code", "side"], sort=False):
        g = g[(g.f_med + g.f_lat) > min_total_n]
        X = np.column_stack([g.f_med, g.f_lat, np.ones(len(g))])
        coef, *_ = np.linalg.lstsq(X, g.mx.to_numpy(), rcond=None)
        res = g.mx.to_numpy() - X @ coef
        r = float(np.corrcoef(g.mx, g.f_med - g.f_lat)[0, 1])
        rows.append(dict(subject=subj, code=code, side=side, n_windows=g.window.nunique(), n_rows=len(g),
                         r_mx_vs_med_minus_lat=r, a_med_m=coef[0], b_lat_m=coef[1], c_nm=coef[2],
                         resid_rms_nm=float(np.sqrt(np.mean(res ** 2))), mx_sd_nm=float(g.mx.std()),
                         fy_compressive_frac=float((g.fy < 0).mean()),
                         implied_medial_sign=float(np.sign(r)) if np.sign(coef[0]) == np.sign(r) == -np.sign(coef[1])
                         else np.nan,
                         physical_sign=medial_mx_sign(side), pipeline_sign=medial_mx_sign(side, "pipeline")))
    return pd.DataFrame(rows)


# ─── Population (optional) ────────────────────────────────────────────────────

PROFILE_BINS = np.arange(5.0, 100.0, 10.0)        # % of stance, centres of 10 % bins


def stance_profile(t, on, x, dur_s=(0.3, 1.6)):
    """Mean of x over complete contact runs (rising to falling edge of `on`, duration within dur_s), sampled at
    PROFILE_BINS % of each run; NaN without a complete run."""
    e = np.flatnonzero(np.diff(on.astype(int)))
    prof = []
    for a, b in zip(e[:-1], e[1:]):
        if on[a + 1] and dur_s[0] <= t[b] - t[a + 1] <= dur_s[1]:
            seg = x[a + 1:b + 1]
            prof.append(np.interp(PROFILE_BINS, np.linspace(0, 100, len(seg)), seg))
    return np.nanmean(prof, axis=0) if prof else np.full(len(PROFILE_BINS), np.nan)


def _stance_mean_mx(folder, contact_bw=0.05):
    """Stance-mean raw M_x (BW*H) per knee for SO+JR (both knees) and the stored labels (right knee), and its mean
    profile over complete stances (prof_<src>_<side>_<bin>)."""
    import json
    out = dict(folder=os.path.basename(folder), lab=os.path.basename(folder).split("_")[0])
    try:
        meta = json.load(open(os.path.join(folder, "metadata.json")))
        bw, H = meta["mass_kg"] * 9.81, meta.get("height_m") or np.nan
        grf = _read_ws(os.path.join(folder, "grf_data.mot"))
        for src, sub in (("sojr", "jcf_output_sojr"), ("stored", "jcf_output")):
            p = os.path.join(folder, sub, "BatchJCF_JointReaction_ReactionLoads.sto")
            if not os.path.exists(p):
                continue
            j = _read_ws(p)
            for s in "rl":
                pre = f"walker_knee_{s}_on_tibia_{s}_in_tibia_{s}_mx"
                if pre not in j:
                    continue
                t = j["time"].to_numpy(float)
                st = np.interp(t, grf["time"], np.abs(grf[f"calcn_{s}_force_vy"])) > contact_bw * bw
                mx = j[pre].to_numpy(float)[st] / (bw * H)
                if np.isfinite(mx).sum() >= 5:
                    out[f"{src}_{s}_n"] = int(np.isfinite(mx).sum())
                    out[f"{src}_{s}_mx_mean"] = float(np.nanmean(mx))
                    prof = stance_profile(t, st, j[pre].to_numpy(float) / (bw * H))
                    out.update({f"prof_{src}_{s}_{int(b)}": float(v) for b, v in zip(PROFILE_BINS, prof)})
    except Exception as e:                       # one bad folder must not stop the scan
        out["error"] = f"{type(e).__name__}: {e}"
    return out


def population(max_per_lab=150, workers=4, repo=REPO):
    import multiprocessing as mp
    folders = []
    for lab in LABS:
        fs = sorted(os.path.dirname(os.path.dirname(p)) for p in glob.glob(os.path.join(
            repo, "jcf", "full_duration", "*", "walking", f"{lab}_*", "jcf_output_sojr",
            "BatchJCF_JointReaction_ReactionLoads.sto")))
        step = max(1, int(np.ceil(len(fs) / max_per_lab))) if max_per_lab else 1
        folders += fs[::step]
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"                      # inherited by the spawned workers before they import numpy
    with mp.get_context("spawn").Pool(workers) as pool:
        return pd.DataFrame(pool.map(_stance_mean_mx, folders, chunksize=8))


def profile_summary(pop):
    """Per lab x source x side: mean MEDIAL-positive (physical sign) M_x in BW*H x 1000 per 10 % of stance, over
    the folders with a complete stance."""
    rows = []
    for lab, g in pop.groupby("lab"):
        for src, s in (("sojr", "r"), ("sojr", "l"), ("stored", "r")):
            cols = [f"prof_{src}_{s}_{int(b)}" for b in PROFILE_BINS]
            if not all(c in g for c in cols):
                continue
            v = g[cols].dropna()
            if v.empty:
                continue
            m = 1000 * medial_mx_sign(s) * v.mean(axis=0).to_numpy()
            rows.append(dict(lab=lab, source=src, side=s, n_folders=len(v),
                             **{f"pct{int(b)}": float(x) for b, x in zip(PROFILE_BINS, m)}))
    return pd.DataFrame(rows)


def population_summary(pop):
    rows = []
    for lab, g in pop.groupby("lab"):
        for src, s in (("sojr", "r"), ("sojr", "l"), ("stored", "r")):
            c = f"{src}_{s}_mx_mean"
            if c not in g or g[c].notna().sum() == 0:
                continue
            v = g[c].dropna()
            rows.append(dict(lab=lab, source=src, side=s, n_folders=len(v), median_mx_bwh=float(v.median()),
                             medial_frac_physical=float((medial_mx_sign(s) * v > 0).mean()),
                             medial_frac_pipeline=float((medial_mx_sign(s, "pipeline") * v > 0).mean())))
    return pd.DataFrame(rows)


# ─── Main ─────────────────────────────────────────────────────────────────────

def gc_implant_table(run_dir):
    """mx_sign_check.csv of a grand_challenge.py run and the convention it used (None when missing)."""
    import json
    p = os.path.join(run_dir, "mx_sign_check.csv")
    if not os.path.exists(p):
        return None, None
    try:
        with open(os.path.join(run_dir, "run.json")) as f:
            conv = json.load(f).get("args", {}).get("mx_convention", "pipeline")
    except (OSError, ValueError):
        conv = "unknown"
    return pd.read_csv(p), conv


def write_readme(out, geo, kan, pop_sum, two, implant=None, implant_conv=None, implant_run=None, prof=None):
    def tab(df):
        return df.to_string(index=False, float_format=lambda v: f"{v:.4g}")
    lines = ["# Knee M_x sign: which sign is medial", "",
             "Written by `analysis/thesis/knee_mx_sign.py` (reads files only).", "",
             "**Settled: medial-positive M_x = -mx at walker_knee_r and +mx at walker_knee_l** (`physical`). The "
             "pipeline convention (+mx right, -mx left: train_cnn.load_subject comment, train_mlp F_medial loss, "
             "tools/evaluate_fmedial, analysis/thesis/label_qc.MX_SIGN) is inverted.", "",
             "Why: JR reports the load on the tibia at the knee centre in the tibia body frame (z to the subject's "
             "right on both legs). A compressive condyle force (0, -f, 0) at z_c gives M_x = z_c f, and the medial "
             "condyle is at z < 0 on the right tibia and z > 0 on the left.", "",
             "## 1. Two-point contact (synthetic)", "", "```", tab(two), "```",
             "F_medial = 0.5|F_y| + s M_x / d with contacts at z = -/+ d/2: the physical s returns the medial force, the "
             "pipeline s returns the lateral force.", "",
             "## 2. Model geometry (our models, one per lab)", "",
             "Median z (m, relative to the walker_knee frame origin) of the medial points (gasmed origin, grac / "
             "semiten / sart) and the lateral points (gaslat origin, bfsh insertion) on femur and tibia.", "",
             "```", tab(geo), "```", ""]
    if kan is not None and not kan.empty:
        lines += ["## 3. Kaneda et al.'s condyle split (StatOp_Val SO+JR, same JR table)", "",
                  "Rows with medial + lateral contact force > 200 N. mx = a F_med + b F_lat + c (least squares).", "",
                  "```", tab(kan), "```", ""]
    if pop_sum is not None and not pop_sum.empty:
        lines += ["## 4. Population: stance-mean M_x per walking folder", "",
                  "Stance = |GRF_y| of that foot > 0.05 BW. Up to --max-per-lab folders per lab with SO+JR labels, "
                  "every k-th in sorted order. medial_frac_* = share of folders whose stance-mean moment is medial "
                  "under that sign.", "", "```", tab(pop_sum), "```", ""]
    if prof is not None and not prof.empty:
        lines += ["Mean medial-positive M_x (physical sign, BW*H x 1000) per 10 % of stance, complete stances "
                  "(0.3-1.6 s) only. A walking knee adduction moment is medial through stance with an early peak; a "
                  "negative early-stance value is a LATERAL moment and is not physiological.", "",
                  "```", tab(prof), "```", ""]
    if implant is not None and not implant.empty:
        lines += ["## 5. Implant: medial-positive M_x against the measured medial - lateral force", "",
                  f"From `{os.path.relpath(implant_run, REPO)}/mx_sign_check.csv` (M_x convention `{implant_conv}`), "
                  "instrumented-limb stance frames. r > 0: the source's medial moment rises with the implant's "
                  "medial - lateral force. medial_share = mean(F_medial) / mean(|F_y|) vs the implant's "
                  "mean(medial) / mean(total).", "", "```", tab(implant), "```", ""]
    with open(os.path.join(out, "README.md"), "w") as f:
        f.write("\n".join(lines))


def two_point_table(d_m=0.045):
    rows = []
    for side in "rl":
        z_med = -d_m / 2 if side == "r" else d_m / 2
        fy, mx = two_point_reaction(1500.0, 600.0, z_med, -z_med)
        rows.append(dict(side=side, f_med_n=1500.0, f_lat_n=600.0, z_med_m=z_med, fy_n=float(fy), mx_nm=float(mx),
                         fmed_physical_n=float(fmedial_n(fy, mx, medial_mx_sign(side), d_m=d_m)),
                         fmed_pipeline_n=float(fmedial_n(fy, mx, medial_mx_sign(side, "pipeline"), d_m=d_m))))
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(REPO, "results", "thesis", "labels_qc", "mx_sign"))
    ap.add_argument("--gc-root", default=GC_ROOT)
    ap.add_argument("--population", action="store_true", help="also scan walking folders (stance-mean M_x)")
    ap.add_argument("--max-per-lab", type=int, default=150)
    ap.add_argument("--gc-run", default=os.path.join(REPO, "results", "thesis", "gc", "pilot_AB_sojr"),
                    help="grand_challenge.py run whose mx_sign_check.csv goes into the README")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    two = two_point_table()
    two.to_csv(os.path.join(args.out, "two_point.csv"), index=False)
    geo = geometry_check(one_model_per_lab())
    geo.to_csv(os.path.join(args.out, "model_geometry.csv"), index=False)
    kf = kaneda_frames(args.gc_root)
    kan = kaneda_check(kf) if not kf.empty else pd.DataFrame()
    kan.to_csv(os.path.join(args.out, "kaneda_condyle.csv"), index=False)
    pop_sum = None
    if args.population:
        pop = population(args.max_per_lab, args.workers)
        pop.to_csv(os.path.join(args.out, "population.csv"), index=False)
        pop_sum = population_summary(pop)
        pop_sum.to_csv(os.path.join(args.out, "population_summary.csv"), index=False)
        prof = profile_summary(pop)
        prof.to_csv(os.path.join(args.out, "population_profile.csv"), index=False)
    else:
        prof = None
        for name in ("population_summary", "population_profile"):
            p_ = os.path.join(args.out, f"{name}.csv")
            if os.path.exists(p_):
                if name == "population_summary":
                    pop_sum = pd.read_csv(p_)
                else:
                    prof = pd.read_csv(p_)
    implant, conv = gc_implant_table(args.gc_run)
    write_readme(args.out, geo, kan, pop_sum, two, implant, conv, args.gc_run, prof)
    pd.set_option("display.width", 220)
    for name, df in (("two_point", two), ("geometry", geo), ("kaneda", kan), ("population", pop_sum),
                     ("profile", prof)):
        if df is not None and not df.empty:
            print(f"--- {name}\n{df.round(4).to_string(index=False)}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

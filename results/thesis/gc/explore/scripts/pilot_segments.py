"""Pilot (exploration only): JW/DM labelled segments, right knee. Surrogate (deploy features) and OpenSim SO+JR
F_medial vs measured medial force, right stance frames. Uses contentmap.csv + align_explore.csv.

Run: python -I pilot_segments.py CONTENTMAP ALIGN OUT_CSV CKPT...
"""
import sys, os, glob, json, re, types
REPO = "/home/gellert/Developer/medical_dt"
sys.path.insert(0, REPO)
import numpy as np
import pandas as pd
import torch
from tools.evaluate_fmedial import features_for, load_model, fmedial, basic_metrics
from train_mlp import INPUT_SETS

GC = f"{REPO}/jcf/external/grand_challenge"
LB, INLB = 4.4482216152605, 112.98482933
SUBJ = {"1GC": ("GC3", "EKnee Data", 1 / 200), "2GC": ("GC2", "eTibia Data", 1 / 200)}


def measured(code, kp):
    kd = pd.read_csv(kp, index_col=False)
    kd = kd.loc[:, ~kd.columns.str.startswith("Unnamed")]
    t = kd.iloc[:, 0].to_numpy(float)
    if code == "1GC":
        s = kd[["Sensor1", "Sensor2", "Sensor3", "Sensor4"]].to_numpy(float) * LB
        return t, s[:, 0] + s[:, 1], s.sum(1)
    fx, fz, my = kd.Sensor1 * LB, kd.Sensor3 * LB, kd.Sensor5 * INLB
    return t, (-0.876 * fx + 0.494 * fz - 0.0189 * my).to_numpy(), fz.to_numpy()


def main(cmap_csv, align_csv, out_csv, ckpts):
    cm = pd.read_csv(cmap_csv); al = pd.read_csv(align_csv).set_index(["subj", "trial"])
    args = types.SimpleNamespace(deploy_features=True, max_peak=6.0, pelvis="trial")
    models = [load_model(p, "cpu") for p in ckpts]
    rows = []
    for _, c in cm[cm.subj.isin(SUBJ)].iterrows():
        code = c.subj; fold, kdir, dt_gc = SUBJ[code]
        split = "training"
        d = f"{REPO}/jcf/full_duration/{split}/walking/{c.folder}"
        a = al.loc[(code, int(c.trial))]
        cands = {re.sub(r"[^a-z0-9]", "", os.path.basename(p).replace("_knee_forces.csv", "").lower()): p
                 for p in glob.glob(f"{GC}/{fold}/Gait Trials/{kdir}/*_knee_forces.csv")}
        tk, med, tot = measured(code, cands[re.sub(r"[^a-z0-9]", "", a.orig.lower())])
        meta = json.load(open(f"{d}/metadata.json")); m_kg, H = meta["mass_kg"], meta["height_m"]; BW = m_kg * 9.81
        for (m, mu, sd, info) in models:
            built = features_for(d, INPUT_SETS[info["input_set"]], args)
            if built is None:
                rows.append(dict(folder=c.folder, model=info["name"], status="no_features")); continue
            feat, labels, fm = built
            with torch.no_grad():
                pr = m((torch.from_numpy(feat).float() - mu) / sd).numpy()
            # segment time -> b3d frame (AddB dt) -> GC row -> GC time
            frame = int(c.start_frame) + fm["t"] / a.dt_b3d
            t_gc = a.gc_t0 + (frame + int(a.offset)) * dt_gc
            mm = np.interp(t_gc, tk, med) / BW; mt = np.interp(t_gc, tk, tot) / BW
            st = np.asarray(fm["r_contact"], bool)
            if st.sum() < 5:
                continue
            fm_p = fmedial(np.abs(pr[:, 1]), pr[:, 3], 0.5, 45.0, H)
            fm_g = fmedial(np.abs(labels[:, 1]), labels[:, 3], 0.5, 45.0, H)
            r = dict(subj=code, folder=c.folder, orig=a.orig, model=info["name"], n_stance=int(st.sum()))
            for k, (x, y) in {"sur_vs_meas": (fm_p, mm), "osim_vs_meas": (fm_g, mm), "sur_vs_osim": (fm_p, fm_g),
                              "surFy_vs_measTot": (np.abs(pr[:, 1]), mt), "osimFy_vs_measTot": (np.abs(labels[:, 1]), mt)}.items():
                b = basic_metrics(y[st], x[st])
                r.update({f"{k}_mae": b["mae"], f"{k}_bias": b["bias"], f"{k}_r": b["r"]})
            rows.append(r)
    df = pd.DataFrame(rows); df.to_csv(out_csv, index=False)
    ok = df[df.get("n_stance", pd.Series(dtype=float)).notna()]
    w = lambda g: pd.Series({c: np.average(g[c], weights=g.n_stance) for c in g.columns if c.endswith(("_mae", "_bias"))} | {"n": g.n_stance.sum(), "segs": len(g)})
    pd.set_option("display.width", 250)
    print(ok.groupby(["subj", "model"]).apply(w).round(3).T.to_string())


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:])

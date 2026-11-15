"""External knee adduction moment (KAM) on the Grand Challenge segments, scored like the other sources.

KAM = external moment of the instrumented foot's GRF (force at the COP plus free torque, grf_data.mot) about the knee
centre (tibia_<side>_offset origin, by forward kinematics of the segment's scaled model, analysis/thesis/osim_fk.py),
projected on the tibia's anterior (x) axis; medial-positive: +x on the right knee, -x on the left; in BW*H.
Scored (1) frame by frame against the measured medial force in stance and (2) on the gait-modification test exactly as
grand_challenge.gait_mods() does (the script asserts it reproduces gait_mods.csv for the measured, SO+JR and qc_s1 rows).
Output (local, per-frame GC data): results/thesis/gc/audit_gc/kam_frames.csv.gz, kam_gait_mods.csv.
Usage (repo root): python -I analysis/thesis/gc_kam.py
"""
import glob, os, re, sys
import numpy as np, pandas as pd
sys.path.insert(0, "."); sys.path.insert(0, "analysis/thesis")
from analysis.thesis.osim_fk import OsimChain
from fig_defects import read_sto
G = 9.81
fr = pd.read_csv("results/thesis/gc/audit_gc/frames.csv.gz")
side = {"JW": "r", "DM": "r", "SC": "l"}
fr["kam"] = np.nan; fr["kfm"] = np.nan
for folder, g in fr.groupby("folder"):
    d = glob.glob(f"jcf/full_duration/*/*/{folder}")[0]
    s = side[g.subject.iloc[0]]
    ik = pd.read_csv(os.path.join(d, "ik_results.mot"), sep=r"\s+", skiprows=6)
    coords = {c: ik[c].to_numpy(float) for c in ik.columns if c != "time"}
    t = ik.time.to_numpy(float)
    osim = os.path.join(d, "scaled_model.osim"); fk = OsimChain(osim)
    xml = open(osim).read(); i = xml.find(f'<PhysicalOffsetFrame name="tibia_{s}_offset">')
    off = np.array(re.findall(r"<translation>([^<]+)<", xml[i:i + 2500])[0].split(), float)
    R, p = fk.body_pose(f"tibia_{s}", coords)
    knee = p + R @ off
    grf = read_sto(os.path.join(d, "grf_data.mot"))
    F = np.column_stack([np.interp(t, grf.time, grf[f"calcn_{s}_force_v{a}"]) for a in "xyz"])
    P = np.column_stack([np.interp(t, grf.time, grf[f"calcn_{s}_force_p{a}"]) for a in "xyz"])
    T = np.column_stack([np.interp(t, grf.time, grf[f"calcn_{s}_torque_{a}"]) for a in "xyz"])
    Mg = np.cross(P - knee, F) + T                      # external moment of the GRF about the knee centre, ground axes
    Mt = np.einsum("tji,tj->ti", R, Mg)                 # in tibia axes (R^T M)
    bwh = g.mass.iloc[0] * G * g.H.iloc[0]
    kam = (1.0 if s == "r" else -1.0) * Mt[:, 0] / bwh  # medial-positive (adduction) on either knee, BW*H
    kfm = Mt[:, 2] / bwh
    fr.loc[g.index, "kam"] = np.interp(g.t_seg, t, kam); fr.loc[g.index, "kfm"] = np.interp(g.t_seg, t, kfm)
st = fr[fr.stance]
print("KAM in stance (BW*H, % below): median peak per subject", (st.groupby("subject").kam.quantile(0.98) * 100).round(2).to_dict())
print("frame-level r(KAM, measured medial force) in stance:", {k: round(float(np.corrcoef(v.kam, v.meas_med)[0, 1]), 2) for k, v in st.groupby("subject")})
# window peaks, keyed like stances.csv
pk = fr[fr.part.notna()].groupby(["gc_key", "stance_id", "part"]).kam.max().rename("peak_kam").reset_index()
S = pd.read_csv("results/thesis/gc/audit_gc/stances.csv").merge(pk, left_on=["gc_key", "stance", "part"], right_on=["gc_key", "stance_id", "part"], how="left")
S = S[S.meas_interior.astype(bool)]
print("scored windows", len(S), "with a KAM peak", int(S.peak_kam.notna().sum()))
rows = []
for (subj, part), g in S.groupby(["subject", "part"]):
    base = g[g.gait == "ngait"]
    if base.empty: continue
    for gait, gg in g.groupby("gait"):
        if gait == "ngait": continue
        r = dict(subject=subj, part=part, gait=gait, meas=gg.meas_peak.mean() - base.meas_peak.mean())
        for src in ("osim_sojr", "qc_s1", "kam"):
            r[src] = gg[f"peak_{src}"].mean() - base[f"peak_{src}"].mean()
        rows.append(r)
R_ = pd.DataFrame(rows)
ref = pd.read_csv("results/thesis/gc/audit_gc/gait_mods.csv")
ok_ = (np.allclose(R_.meas.values, ref.meas_change.values), np.allclose(R_.osim_sojr.values, ref.osim_sojr_change.values),
      np.allclose(R_.qc_s1.values, ref.qc_s1_change.values))
assert all(ok_), "does not reproduce gait_mods.csv"
print("reproduces gait_mods.csv: True")
m = R_.meas.to_numpy(); k = R_.kam.to_numpy()
print(f"KAM: direction right {int((np.sign(k) == np.sign(m)).sum())} of {len(m)}, r(change) {np.corrcoef(k, m)[0, 1]:+.2f}")
print(R_.assign(kam_pct=R_.kam * 100).drop(columns="kam").round(3).to_string(index=False))
out = "results/thesis/gc/audit_gc"
fr[["subject", "gc_key", "stance_id", "part", "t_seg", "kam", "kfm"]].to_csv(f"{out}/kam_frames.csv.gz", index=False)
R_.to_csv(f"{out}/kam_gait_mods.csv", index=False)
print("wrote", out)

#!/usr/bin/env python3
"""Label an exported trajectory CSV with OpenSim Static Optimization + JointReaction.

Turns a posture-optimizer export (time, q0..q36 in Rajagopal URDF order, com/pelvis/foot positions,
foot wrenches; robot frame x-fwd / y-left / z-up) into a training-format trial folder:

    <out>/ik_results.mot  grf_data.mot  grf_loads.xml  scaled_model.osim  metadata.json
          com_features.npz  jcf_output/BatchJCF_JointReaction_ReactionLoads.sto (+ _raw.sto)
          jcf_output/BatchJCF_StaticOptimization_activation.sto  label_quality.json  labels.csv

Run inside the OpenSim environment:

    conda run -n jcf python pipeline/label_trajectory_csv.py --csv traj.csv --mass 75 --height 1.70 \
        --out jcf/external/traj_name [--model path.osim] [--split-at 3.3] [--plot]

Label cleaning: frames whose |Mx| exceeds 0.15 BW·H or whose resultant exceeds 6 BW (the pipeline's own
corruption thresholds, typically double-support transitions) are linearly interpolated; the raw file is kept
and the mask is recorded. --split-at writes additional <out>_seg1, _seg2, … folders sliced at those times.
"""
import argparse, os, sys, json, glob, shutil, re
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

URDF_ORDER_FIRST6 = ["pelvis_tx", "pelvis_ty", "pelvis_tz", "pelvis_tilt", "pelvis_list", "pelvis_rotation"]
ALPHA, D_M = 0.5, 0.045
TEMPLATE_TRIAL = "jcf/full_duration/training/walking/carter_P019_split0_t19"   # for .mot headers + grf_loads.xml


def to_opensim(v):
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    return np.stack([x, z, -y], axis=-1)


def read_header_names(path):
    txt = open(path).read().split("endheader\n", 1)[1]
    return txt.split("\n", 1)[0].split("\t")


def write_mot(path, names, data, kind):
    with open(path, "w") as f:
        f.write(f"{kind}\nversion=1\nnRows={len(data)}\nnColumns={len(names)}\ninDegrees=no\nendheader\n" + "\t".join(names) + "\n")
        for row in data:
            f.write("\t".join(f"{v:.8f}" for v in row) + "\n")


def read_sto(path):
    hdr = open(path).read().split("endheader\n")[0].count("\n") + 1
    return pd.read_csv(path, sep="\t", skiprows=hdr)


def read_header(path):
    """Verbatim OpenSim header up to and including the endheader line. The training loaders skip a FIXED number
    of header lines (11 for JointReaction, 8 for activation, 6 for .mot), so rewritten files must keep it."""
    txt = open(path).read(); i = txt.index("endheader\n") + len("endheader\n"); return txt[:i]


def write_sto(path, df, header):
    header = re.sub(r"nRows=\d+", f"nRows={len(df)}", header); header = re.sub(r"nColumns=\d+", f"nColumns={len(df.columns)}", header)
    with open(path, "w") as f:
        f.write(header); df.to_csv(f, sep="\t", index=False, float_format="%.8f")


def pick_model(mass, height):
    best = None
    for m in glob.glob("jcf/full_duration/training/*/*/metadata.json"):
        d = json.load(open(m)); mm, hh = float(d.get("mass_kg", 0)), float(d.get("height_m", 0))
        if not (mm and hh): continue
        score = abs(mm - mass) / mass + abs(hh - height) / height
        p = os.path.join(os.path.dirname(m), "scaled_model.osim")
        if os.path.exists(p) and (best is None or score < best[0]): best = (score, p, mm, hh)
    return best


def build_folder(csv, out, mass, height, model_path):
    os.makedirs(out, exist_ok=True)
    df = pd.read_csv(csv).dropna(subset=["com_position_0"]).reset_index(drop=True)
    t = df["time"].to_numpy(); N = len(df); BW = mass * 9.81
    q = df[[f"q{i}" for i in range(37)]].to_numpy()
    ik = np.zeros((N, 37)); ik[:, 0:3] = q[:, 3:6]; ik[:, 3:6] = to_opensim(q[:, 0:3]); ik[:, 6:] = q[:, 6:]
    names = read_header_names(f"{TEMPLATE_TRIAL}/ik_results.mot")
    write_mot(f"{out}/ik_results.mot", names, np.column_stack([t, ik]), "Coordinates")
    gnames = read_header_names(f"{TEMPLATE_TRIAL}/grf_data.mot"); cols = {"time": t}
    pos = {}
    for side, tag in (("r", "right"), ("l", "left")):
        W = df[[f"{tag}_foot_wrench_{k}" for k in range(6)]].to_numpy(float)
        F, M = to_opensim(W[:, 0:3]), to_opensim(W[:, 3:6])
        P = to_opensim(df[[f"{tag}_foot_position_{k}" for k in range(3)]].to_numpy()); pos[side] = P
        for j, a in enumerate("xyz"):
            cols[f"calcn_{side}_force_v{a}"] = F[:, j]; cols[f"calcn_{side}_force_p{a}"] = P[:, j]; cols[f"calcn_{side}_torque_{a}"] = M[:, j]
    write_mot(f"{out}/grf_data.mot", gnames, np.column_stack([cols[n] for n in gnames]), "GRF")
    shutil.copy2(f"{TEMPLATE_TRIAL}/grf_loads.xml", f"{out}/grf_loads.xml")
    shutil.copy2(model_path, f"{out}/scaled_model.osim")
    com = to_opensim(df[[f"com_position_{k}" for k in range(3)]].to_numpy())
    np.savez(f"{out}/com_features.npz", time=t, com_pos=com.astype(np.float32),
             com_vel=np.gradient(com, t, axis=0).astype(np.float32),
             calcn_r_pos=pos["r"].astype(np.float32), calcn_l_pos=pos["l"].astype(np.float32))
    json.dump({"subject": os.path.basename(out), "mass_kg": mass, "height_m": height, "n_dofs": 37,
               "dof_names": names[1:], "n_valid_frames": int(N), "contact_bodies": ["calcn_r", "calcn_l"],
               "source_csv": os.path.abspath(csv), "scaled_model_from": model_path,
               "note": "synthetic optimizer trajectory labelled with pipeline/label_trajectory_csv.py"},
              open(f"{out}/metadata.json", "w"), indent=1)
    return t, BW, df


def run_so_jr(out, t0, t1):
    import process_jcf
    ok = process_jcf.run_jcf(out, t0, t1)
    if not ok: raise SystemExit("SO+JR failed")
    jo = f"{out}/jcf_output"
    for src, dst in (("FullDuration_JointReaction_ReactionLoads.sto", "BatchJCF_JointReaction_ReactionLoads.sto"),
                     ("FullDuration_StaticOptimization_activation.sto", "BatchJCF_StaticOptimization_activation.sto"),
                     ("FullDuration_StaticOptimization_force.sto", "BatchJCF_StaticOptimization_force.sto")):
        cands = [f for f in os.listdir(jo) if f.endswith(src.split("FullDuration")[1])]
        if cands: shutil.copy2(os.path.join(jo, cands[0]), os.path.join(jo, dst))


def clean_labels(out, mass, height, df_src):
    BW = mass * 9.81; jo = f"{out}/jcf_output"; p = f"{jo}/BatchJCF_JointReaction_ReactionLoads.sto"
    raw_p = f"{jo}/FullDuration_JointReaction_ReactionLoads.sto"          # untouched SO+JR output
    raw = read_sto(raw_p); hdr = read_header(raw_p); raw.to_csv(f"{jo}/BatchJCF_JointReaction_ReactionLoads_raw.csv", index=False)
    pre = "walker_knee_r_on_tibia_r_in_tibia_r_"
    fy = raw[pre + "fy"].to_numpy() / BW; mx = raw[pre + "mx"].to_numpy() / (BW * height)
    res = np.sqrt(sum(raw[pre + c].to_numpy() ** 2 for c in ("fx", "fy", "fz"))) / BW
    bad = (np.abs(mx) > 0.15) | (res > 6.0)
    cleaned = raw.copy(); t = raw["time"].to_numpy()
    for c in cleaned.columns:
        if c == "time": continue
        v = cleaned[c].to_numpy(float).copy()
        if bad.any() and (~bad).sum() > 2: v[bad] = np.interp(t[bad], t[~bad], v[~bad])
        cleaned[c] = v
    write_sto(p, cleaned, hdr); np.save(f"{jo}/label_frame_mask.npy", ~bad)
    act = read_sto(f"{jo}/BatchJCF_StaticOptimization_activation.sto")
    mus = [c for c in act.columns if c != "time" and not c.startswith("reserve")]
    sat = (act[mus].to_numpy() >= 0.999).sum(1)
    fy_c = cleaned[pre + "fy"].to_numpy() / BW; mx_c = cleaned[pre + "mx"].to_numpy() / (BW * height)
    fmed = ALPHA * np.abs(fy_c) + mx_c * height / D_M
    rc = np.abs(df_src["right_foot_wrench_2"].to_numpy(float)) > 0.05 * BW; lc = np.abs(df_src["left_foot_wrench_2"].to_numpy(float)) > 0.05 * BW
    n = min(len(t), len(rc)); ph = np.where(rc[:n] & lc[:n], "dual", np.where(rc[:n], "right", np.where(lc[:n], "left", "flight")))
    pd.DataFrame({"time": t[:n], "phase": ph, "fy_bw": fy_c[:n], "mx_bwh": mx_c[:n], "f_medial_bw": fmed[:n], "clean": (~bad)[:n]}).to_csv(f"{out}/labels.csv", index=False)
    q = {"frames": int(len(t)), "corrupted_frames_interpolated": int(bad.sum()),
         "frames_with_gt5_saturated_muscles": int((sat > 5).sum()), "saturation_fraction": float((sat > 5).mean()),
         "right_stance_fmedial_mean_bw": float(fmed[:n][ph == "right"].mean()) if (ph == "right").any() else None,
         "right_stance_fmedial_peak_bw": float(np.abs(fmed[:n][ph == "right"]).max()) if (ph == "right").any() else None,
         "right_stance_fy_mean_bw": float(fy_c[:n][ph == "right"].mean()) if (ph == "right").any() else None}
    json.dump(q, open(f"{out}/label_quality.json", "w"), indent=1)
    return q


def slice_folder(src, dst, t0, t1):
    os.makedirs(f"{dst}/jcf_output", exist_ok=True)
    for f in ("ik_results.mot", "grf_data.mot"):
        names = read_header_names(f"{src}/{f}"); df = read_sto(f"{src}/{f}"); m = (df["time"] >= t0) & (df["time"] <= t1)
        write_mot(f"{dst}/{f}", names, df[m].to_numpy(float), "Coordinates" if f.startswith("ik") else "GRF")
    for f in os.listdir(f"{src}/jcf_output"):
        if f.endswith(".sto"):
            df = read_sto(f"{src}/jcf_output/{f}"); m = (df["time"] >= t0) & (df["time"] <= t1)
            write_sto(f"{dst}/jcf_output/{f}", df[m].reset_index(drop=True), read_header(f"{src}/jcf_output/{f}"))
    z = np.load(f"{src}/com_features.npz"); m = (z["time"] >= t0) & (z["time"] <= t1)
    np.savez(f"{dst}/com_features.npz", **{k: z[k][m] for k in z.files})
    for f in ("grf_loads.xml", "scaled_model.osim"): shutil.copy2(f"{src}/{f}", f"{dst}/{f}")
    meta = json.load(open(f"{src}/metadata.json")); meta.update(subject=os.path.basename(dst), n_valid_frames=int(m.sum()), segment=[t0, t1])
    json.dump(meta, open(f"{dst}/metadata.json", "w"), indent=1)
    lab = pd.read_csv(f"{src}/labels.csv"); lab[(lab["time"] >= t0) & (lab["time"] <= t1)].to_csv(f"{dst}/labels.csv", index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--mass", type=float, required=True); ap.add_argument("--height", type=float, required=True)
    ap.add_argument("--model", default=None, help="scaled .osim to use; default = nearest mass/height subject in jcf/")
    ap.add_argument("--t0", type=float, default=None); ap.add_argument("--t1", type=float, default=None)
    ap.add_argument("--split-at", type=float, nargs="*", default=[], help="also write <out>_segN folders sliced at these times")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--relabel-only", action="store_true", help="skip folder build and SO+JR; redo cleaning/segments/plot from the existing jcf_output")
    a = ap.parse_args()
    model = a.model
    if model is None and not a.relabel_only:
        best = pick_model(a.mass, a.height); model = best[1]; print(f"borrowed scaled model: {model} ({best[2]:.1f} kg, {best[3]:.2f} m)")
    if a.relabel_only:
        df = pd.read_csv(a.csv).dropna(subset=["com_position_0"]).reset_index(drop=True); t = df["time"].to_numpy()
        t0 = a.t0 if a.t0 is not None else float(t[0]); t1 = a.t1 if a.t1 is not None else float(t[-1])
        print(f"relabel-only: reusing {a.out}/jcf_output")
    else:
        t, BW, df = build_folder(a.csv, a.out, a.mass, a.height, model)
        t0 = a.t0 if a.t0 is not None else float(t[0]); t1 = a.t1 if a.t1 is not None else float(t[-1])
        print(f"folder built: {a.out} ({len(t)} frames); running SO+JR on [{t0:.2f}, {t1:.2f}] s ...", flush=True)
        run_so_jr(a.out, t0, t1)
    q = clean_labels(a.out, a.mass, a.height, df); print("label quality:", json.dumps(q))
    edges = [t0] + sorted(a.split_at) + [t1]
    for i in range(len(edges) - 1):
        if len(edges) > 2:
            d = f"{a.out}_seg{i+1}"; slice_folder(a.out, d, edges[i], edges[i+1]); print(f"segment {i+1}: {d}  [{edges[i]:.2f}, {edges[i+1]:.2f}] s")
    if a.plot:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        lab = pd.read_csv(f"{a.out}/labels.csv"); fig, ax = plt.subplots(figsize=(11, 3.6))
        ax.plot(lab["time"], lab["f_medial_bw"], color="#0b0b0b"); ax.axhline(1.4, color="#898781", lw=1)
        for ph, col in (("dual", "#e1e0d9"),):
            for tt in lab["time"][lab["phase"] == ph]: ax.axvspan(tt - 0.005, tt + 0.005, color=col, lw=0, alpha=0.7)
        ax.set_xlabel("time (s)"); ax.set_ylabel("F_medial (BW)"); ax.set_title(f"SO+JR labels — {os.path.basename(a.out)}"); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(f"{a.out}/labels.png", dpi=120); print("plot:", f"{a.out}/labels.png")


if __name__ == "__main__":
    main()

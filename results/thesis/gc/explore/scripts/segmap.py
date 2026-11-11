"""Exploration: rebuild fregly segment folders -> (b3d trial, start frame) -> GC row range.

Run: conda run -n jcf python -I segmap.py ALIGN_CSV OUT_CSV
"""
import sys, os, json
sys.path.append(os.path.expanduser("~/.local/lib/python3.10/site-packages"))
REPO = "/home/gellert/Developer/medical_dt"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "pipeline"))
import numpy as np
import pandas as pd
from pipeline.batch_process import scan_b3d_all_runs, GRF_CAP_WALKING

B3D = {
    "1GC": (f"{REPO}/with_arm/training/Fregly2012_Formatted_With_Arm/1GC/1GC.b3d", "training"),
    "2GC": (f"{REPO}/with_arm/training/Fregly2012_Formatted_With_Arm/2GC/2GC.b3d", "training"),
    "3GC": (f"{REPO}/with_arm/testing/Fregly2012_Formatted_With_Arm/3GC/3GC.b3d", "testing"),
}
SIDE = {"1GC": "r", "2GC": "r", "3GC": "l"}


def read_mot(p):
    with open(p) as f:
        for i, l in enumerate(f):
            if l.strip() == "endheader":
                break
    return pd.read_csv(p, sep="\t", skiprows=i + 1)


def main(align_csv, out_csv):
    al = pd.read_csv(align_csv).set_index(["subj", "trial"])
    rows = []
    for code, (b3d, split) in B3D.items():
        segs = scan_b3d_all_runs(b3d, mode="motion")
        counts = {}
        for sg in segs:
            counts[sg["trial"]] = counts.get(sg["trial"], 0) + 1
        seen = {}
        for sg in segs:
            t = sg["trial"]
            ri = seen.get(t, 0)
            seen[t] = ri + 1
            name = f"fregly_{code}_t{t:02d}" if counts[t] == 1 else f"fregly_{code}_t{t:02d}_r{ri:02d}"
            act = "running" if sg["peak_foot_grf_bw"] > GRF_CAP_WALKING else "walking"
            d = os.path.join(REPO, "jcf", "full_duration", split, act, name)
            a = al.loc[(code, t)]
            row = dict(subj=code, folder=name, activity=act, exists=os.path.isdir(d), trial=t, orig=a.orig,
                       start_frame=sg["start_frame"], num_frames=sg["num_frames"], dt_b3d=sg["dt"],
                       peak_foot_bw=sg["peak_foot_grf_bw"], gc_offset=int(a.offset),
                       gc_row_start=sg["start_frame"] + int(a.offset),
                       gc_row_end=sg["start_frame"] + int(a.offset) + sg["num_frames"] - 1)
            ik = os.path.join(d, "ik_results.mot")
            row["has_ik"] = os.path.exists(ik)
            row["has_jr"] = os.path.exists(os.path.join(d, "jcf_output", "BatchJCF_JointReaction_ReactionLoads.sto"))
            row["has_com"] = os.path.exists(os.path.join(d, "com_features.npz"))
            if row["has_ik"]:
                m = read_mot(ik)
                row["ik_rows"] = len(m)
                g = read_mot(os.path.join(d, "grf_data.mot"))
                bw = json.load(open(os.path.join(d, "metadata.json")))["mass_kg"] * 9.81
                side = SIDE[code]
                vy = g[f"calcn_{side}_force_vy"].to_numpy() / bw
                row["inst_vy0"], row["inst_vyN"], row["inst_vymax"] = vy[0], vy[-1], vy.max()
                row["inst_full_stance"] = bool(vy[0] < 0.05 and vy[-1] < 0.05 and vy.max() > 0.8)
                row["inst_contains_hs"] = bool(vy[0] < 0.05 and vy.max() > 0.5)
                row["inst_contains_to"] = bool(vy[-1] < 0.05 and vy.max() > 0.5)
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    pd.set_option("display.width", 250)
    print(df.round(3).to_string())
    # folders on disk not reproduced
    for code, (b3d, split) in B3D.items():
        for act in ("walking", "running"):
            root = os.path.join(REPO, "jcf", "full_duration", split, act)
            disk = {f for f in os.listdir(root) if f.startswith(f"fregly_{code}_")} if os.path.isdir(root) else set()
            mine = set(df[(df.subj == code) & (df.activity == act)].folder)
            print(code, act, "disk", len(disk), "rebuilt", len(mine), "disk-not-rebuilt", sorted(disk - mine)[:10],
                  "rebuilt-not-disk", sorted(mine - disk)[:10])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

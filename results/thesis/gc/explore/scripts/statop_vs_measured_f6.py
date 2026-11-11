"""Exploration: Kaneda StatOp SO+JR medial/lateral (left knee) vs measured eTibia, on GC rows (tau -> row 1 + round(120 tau))."""
import sys, os, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
GC = "/home/gellert/Developer/medical_dt/jcf/external/grand_challenge"
SO = f"{GC}/StatOp_Val_Data/3GC/SO_JRA"
LB, INLB = 4.4482216152605, 112.98482933
BW = 78.4 * 9.81

def read_sto(p):
    with open(p) as f:
        for i, l in enumerate(f):
            if l.strip().lower() == "endheader":
                break
    return pd.read_csv(p, sep=r"\s+", skiprows=i + 1)

rows = []
fig, axs = plt.subplots(2, 2, figsize=(10, 6)); axs = axs.ravel(); k = 0
for d in sorted(glob.glob(f"{SO}/SC_*_stance*")):
    tag = os.path.basename(d); trial = tag.rsplit("_stance", 1)[0]
    j = read_sto(f"{d}/instKnee/results_JointReaction_JointRxn_ReactionLoads.sto")
    tau = j.time.to_numpy(float)
    med_s = j["med_cond_joint_l_on_med_cond_l_in_med_cond_l_fy"].to_numpy(float)
    lat_s = j["lat_cond_joint_l_on_lat_cond_l_in_lat_cond_l_fy"].to_numpy(float)
    kd = pd.read_csv(f"{GC}/GC1/Gait Trials/eTibia Data/{trial}_knee_forces.csv", index_col=False)
    from scipy.signal import butter, filtfilt
    bb, aa = butter(4, 6 / 60)
    fx, fz, ty = [pd.Series(filtfilt(bb, aa, kd[c].to_numpy(float)) * k) for c, k in (("Fx", LB), ("Fz", LB), ("Ty", INLB))]
    med_m = (0.942 * fx + 0.497 * fz + 0.0184 * ty).to_numpy()
    lat_m = (-0.942 * fx + 0.503 * fz - 0.0184 * ty).to_numpy()
    for shift in (-1, 0, 1, 2):
        r = 1 + np.rint(tau * 120).astype(int) + shift
        ok = (r >= 0) & (r < len(med_m))
        mm, ms = med_m[r[ok]] / BW, np.abs(med_s[ok]) / BW
        lm, ls = lat_m[r[ok]] / BW, np.abs(lat_s[ok]) / BW
        rows.append(dict(tag=tag, shift=shift, n=int(ok.sum()), sign_med=np.sign(np.median(med_s)),
                         med_rmse=np.sqrt(np.mean((ms - mm) ** 2)), med_r=np.corrcoef(ms, mm)[0, 1],
                         med_peak_meas=mm.max(), med_peak_so=ms.max(),
                         lat_rmse=np.sqrt(np.mean((ls - lm) ** 2))))
    if k < 4 and "ngait" in tag:
        r = 1 + np.rint(tau * 120).astype(int)
        axs[k].plot(tau, med_m[r] / BW, label="measured medial"); axs[k].plot(tau, np.abs(med_s) / BW, label="StatOp SO medial")
        axs[k].plot(tau, lat_m[r] / BW, label="measured lateral"); axs[k].plot(tau, np.abs(lat_s) / BW, label="StatOp SO lateral")
        axs[k].set_title(tag); axs[k].legend(fontsize=7); k += 1
fig.tight_layout(); fig.savefig(sys.argv[1] + "/statop_vs_measured_f6.png", dpi=100)
df = pd.DataFrame(rows); df.to_csv(sys.argv[1] + "/statop_vs_measured_f6.csv", index=False)
print(df.groupby("shift")[["med_rmse", "med_r", "lat_rmse"]].mean().round(3))
print("sign of SO med fy:", df.sign_med.value_counts().to_dict())
s0 = df[df["shift"] == 0]
print(s0[["med_rmse", "med_r", "med_peak_meas", "med_peak_so", "lat_rmse"]].describe().round(3))

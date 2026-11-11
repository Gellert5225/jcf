"""Exploration: measured medial/lateral force per GC gait trial; sanity vs instrumented-side GRF.

Run: python -I measured_explore.py OUT_DIR
"""
import sys, os, glob, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GC = "/home/gellert/Developer/medical_dt/jcf/external/grand_challenge"
LB = 4.4482216152605      # N per lbf
INLB = 112.98482933       # N*mm per in*lbf
SUBJ = {  # code: folder, knee dir, mass_b3d, side
    "1GC": ("GC3", "EKnee Data", 64.6, "R"),
    "2GC": ("GC2", "eTibia Data", 67.0, "R"),
    "3GC": ("GC1", "eTibia Data", 78.4, "L"),
}


def rcsv(p):
    d = pd.read_csv(p, index_col=False)
    return d.loc[:, ~d.columns.str.startswith("Unnamed")]


def measured(code, kd):
    if code == "1GC":   # eKnee: 4 uniaxial load cells (lbf); S1 PM, S2 AM, S3 AL, S4 PL
        s = kd[["Sensor1", "Sensor2", "Sensor3", "Sensor4"]].to_numpy(float) * LB
        med, lat = s[:, 0] + s[:, 1], s[:, 2] + s[:, 3]
        return med, lat, med + lat
    if code == "2GC":   # eTibia right; Sensor1..6 = Fx Fy Fz Mx My Mz (lbf, in-lbf)
        fx, fz, my = kd.Sensor1 * LB, kd.Sensor3 * LB, kd.Sensor5 * INLB
        med = -0.876 * fx + 0.494 * fz - 0.0189 * my
        lat = 0.876 * fx + 0.505 * fz + 0.0189 * my
        return med.to_numpy(), lat.to_numpy(), fz.to_numpy()
    fx, fz, ty = kd.Fx * LB, kd.Fz * LB, kd.Ty * INLB   # eTibia left
    med = 0.942 * fx + 0.497 * fz + 0.0184 * ty
    lat = -0.942 * fx + 0.503 * fz - 0.0184 * ty
    return med.to_numpy(), lat.to_numpy(), fz.to_numpy()


def plate_feet(g, traj, side):
    """Vertical GRF under the instrumented foot: assign each plate sample to the nearer heel marker."""
    tg = g.iloc[:, 0].to_numpy(float)
    tt = traj[[c for c in traj.columns if c.lower().startswith("time")][0]].to_numpy(float)
    h = {s: np.c_[np.interp(tg, tt, traj[f"{s}.HeelX"]), np.interp(tg, tt, traj[f"{s}.HeelY"])] for s in "RL"}
    tot = np.zeros(len(tg))
    for p in range(1, 5):
        if f"Fz{p}" not in g:
            continue
        fz = np.abs(g[f"Fz{p}"].to_numpy(float))
        cop = np.c_[g[f"COPx{p}"], g[f"COPy{p}"]]
        mine = np.linalg.norm(cop - h[side], axis=1) < np.linalg.norm(cop - h["L" if side == "R" else "R"], axis=1)
        tot += np.where(mine & (fz > 20), fz, 0)
    return tg, tot


def main(out):
    os.makedirs(out, exist_ok=True)
    rows = []
    fig, axs = plt.subplots(3, 1, figsize=(9, 9))
    for ai, (code, (fold, kdir, mass, side)) in enumerate(SUBJ.items()):
        bw = mass * 9.81
        for kp in sorted(glob.glob(f"{GC}/{fold}/Gait Trials/{kdir}/*_knee_forces.csv")):
            kname = os.path.basename(kp).replace("_knee_forces.csv", "")
            vm = glob.glob(f"{GC}/{fold}/Gait Trials/Video Motion Data/*_trajectories.csv")
            tp = [p for p in vm if re.sub(r"[^a-z0-9]", "", os.path.basename(p).replace("_trajectories.csv", "").lower())
                  == re.sub(r"[^a-z0-9]", "", kname.lower())][0]
            trial = os.path.basename(tp).replace("_trajectories.csv", "")
            kd = rcsv(kp)
            tk = kd.iloc[:, 0].to_numpy(float)
            med, lat, tot = measured(code, kd)
            traj = pd.read_csv(tp).interpolate(limit_direction="both")
            g = rcsv(tp.replace("_trajectories.csv", "_grf.csv"))
            tg, vg = plate_feet(g, traj, side)
            vgk = np.interp(tk, tg, vg)
            # lag (ms) maximizing correlation between total axial force and instrumented-foot GRF
            dt = np.median(np.diff(tk))
            lags = np.arange(-20, 21)
            a = (tot - tot.mean()); b = (vgk - vgk.mean())
            cc = [np.nan if b.std() == 0 else np.corrcoef(a[max(0, l):len(a) + min(0, l)], b[max(0, -l):len(b) - max(0, l)])[0, 1] for l in lags]
            st = vgk > 0.05 * bw
            rows.append(dict(subj=code, trial=trial, knee_file=os.path.basename(kp), n=len(tk), dt=dt,
                             med_max_bw=med.max() / bw, lat_max_bw=lat.max() / bw, tot_max_bw=tot.max() / bw,
                             med_min_bw=med.min() / bw, med_share_stance=np.mean(med[st]) / np.mean((med + lat)[st]) if st.any() else np.nan,
                             grf_inst_max_bw=vgk.max() / bw, lag_ms_tot_vs_grf=(lags[int(np.nanargmax(cc))] * dt * 1000 if np.isfinite(cc).any() else np.nan),
                             r_tot_vs_grf=(np.nanmax(cc) if np.isfinite(cc).any() else np.nan), stance_frames=int(st.sum())))
            if ("ngait_og5" in trial or trial in ("jw_ngait_2", "DM_ngait_4")):
                ax = axs[ai]
                ax.plot(tk, med / bw, label="medial"); ax.plot(tk, lat / bw, label="lateral")
                ax.plot(tk, tot / bw, label="axial/total"); ax.plot(tk, vgk / bw, "k--", label=f"GRF {side} foot")
                ax.set_title(f"{code} {trial}"); ax.set_ylabel("BW"); ax.legend(fontsize=7)
    axs[-1].set_xlabel("GC time (s)")
    fig.tight_layout(); fig.savefig(os.path.join(out, "measured_examples.png"), dpi=110)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "measured_summary.csv"), index=False)
    pd.set_option("display.width", 250)
    print(df.round(3).to_string())
    print(df.groupby("subj")[["med_max_bw", "lat_max_bw", "tot_max_bw", "med_share_stance", "grf_inst_max_bw",
                              "lag_ms_tot_vs_grf", "r_tot_vs_grf"]].median().round(3))


if __name__ == "__main__":
    main(sys.argv[1])


def stance_lag(out):
    """Per instrumented-foot plate stance: lag (ms) maximizing corr(total knee force, foot GRF) in a +-100 ms padded window."""
    rows = []
    for code, (fold, kdir, mass, side) in SUBJ.items():
        bw = mass * 9.81
        for kp in sorted(glob.glob(f"{GC}/{fold}/Gait Trials/{kdir}/*_knee_forces.csv")):
            kname = os.path.basename(kp).replace("_knee_forces.csv", "")
            vm = glob.glob(f"{GC}/{fold}/Gait Trials/Video Motion Data/*_trajectories.csv")
            tp = [p for p in vm if re.sub(r"[^a-z0-9]", "", os.path.basename(p).replace("_trajectories.csv", "").lower())
                  == re.sub(r"[^a-z0-9]", "", kname.lower())][0]
            kd = rcsv(kp); tk = kd.iloc[:, 0].to_numpy(float)
            med, lat, tot = measured(code, kd)
            traj = pd.read_csv(tp).interpolate(limit_direction="both")
            tg, vg = plate_feet(rcsv(tp.replace("_trajectories.csv", "_grf.csv")), traj, side)
            on = vg > 0.05 * bw
            edges = np.flatnonzero(np.diff(on.astype(int)))
            if on[0]:
                edges = edges[1:]
            for a, b in zip(edges[::2], edges[1::2]):
                ta, tb = tg[a + 1], tg[b]
                if tb - ta < 0.35 or ta - 0.1 < tk[0] or tb + 0.1 > tk[-1]:
                    continue
                fine = np.arange(ta - 0.1, tb + 0.1, 0.001)
                x = np.interp(fine, tk, tot); y = np.interp(fine, tg, vg)
                lags = np.arange(-80, 81)
                cc = [np.corrcoef(x[max(0, l):len(x) + min(0, l)], y[max(0, -l):len(y) - max(0, l)])[0, 1] for l in lags]
                rows.append(dict(subj=code, trial=kname, t_hs=ta, t_to=tb, lag_ms=int(lags[int(np.argmax(cc))]),
                                 r=float(np.max(cc)), r0=float(cc[80])))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "stance_lag.csv"), index=False)
    print(df.groupby("subj").agg(n=("lag_ms", "size"), lag_med=("lag_ms", "median"), lag_q25=("lag_ms", lambda s: s.quantile(.25)),
                                 lag_q75=("lag_ms", lambda s: s.quantile(.75)), r_med=("r", "median"), r0_med=("r0", "median")))


if __name__ == "__main__" and len(sys.argv) > 2 and sys.argv[2] == "lag":
    stance_lag(sys.argv[1])


def edge_timing(out):
    """Rising (heel strike) and falling (toe-off) edge timing of total knee force vs instrumented-foot GRF."""
    rows = []
    for code, (fold, kdir, mass, side) in SUBJ.items():
        bw = mass * 9.81
        for kp in sorted(glob.glob(f"{GC}/{fold}/Gait Trials/{kdir}/*_knee_forces.csv")):
            kname = os.path.basename(kp).replace("_knee_forces.csv", "")
            vm = glob.glob(f"{GC}/{fold}/Gait Trials/Video Motion Data/*_trajectories.csv")
            tp = [p for p in vm if re.sub(r"[^a-z0-9]", "", os.path.basename(p).replace("_trajectories.csv", "").lower())
                  == re.sub(r"[^a-z0-9]", "", kname.lower())][0]
            kd = rcsv(kp); tk = kd.iloc[:, 0].to_numpy(float)
            med, lat, tot = measured(code, kd)
            traj = pd.read_csv(tp).interpolate(limit_direction="both")
            tg, vg = plate_feet(rcsv(tp.replace("_trajectories.csv", "_grf.csv")), traj, side)
            fs = 1 / np.median(np.diff(tg))
            on = vg > 0.05 * bw
            # close dropouts shorter than 80 ms (plate-to-plate transitions)
            idx = np.flatnonzero(on)
            if len(idx) == 0:
                continue
            gaps = np.flatnonzero(np.diff(idx) > 1)
            for gi in gaps:
                if (idx[gi + 1] - idx[gi]) / fs < 0.08:
                    on[idx[gi]:idx[gi + 1]] = True
            d = np.diff(on.astype(int))
            ups, downs = np.flatnonzero(d == 1) + 1, np.flatnonzero(d == -1) + 1
            for u in ups:
                dn = downs[downs > u]
                if len(dn) == 0:
                    continue
                dn = dn[0]
                ta, tb = tg[u], tg[dn]
                if not (0.45 < tb - ta < 1.2) or ta - 0.3 < tk[0] or tb + 0.3 > tk[-1]:
                    continue
                def cross(t, y, t0, t1, rising):
                    m = (t >= t0) & (t <= t1)
                    tt, yy = t[m], y[m]
                    lo, hi = (yy[:len(yy) // 4].mean(), yy[-len(yy) // 4:].mean()) if rising else (yy[-len(yy) // 4:].mean(), yy[:len(yy) // 4].mean())
                    half = 0.5 * (lo + hi)
                    k = np.flatnonzero(yy >= half) if rising else np.flatnonzero(yy <= half)
                    return tt[k[0]] if len(k) else np.nan
                g_hs = cross(tg, vg, ta - 0.1, ta + 0.1, True)
                k_hs = cross(tk, tot, ta - 0.2, ta + 0.2, True)
                g_to = cross(tg, vg, tb - 0.1, tb + 0.1, False)
                k_to = cross(tk, tot, tb - 0.2, tb + 0.25, False)
                rows.append(dict(subj=code, trial=kname, t_hs=ta, t_to=tb, d_hs_ms=1000 * (k_hs - g_hs),
                                 d_to_ms=1000 * (k_to - g_to)))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "edge_timing.csv"), index=False)
    print(df.groupby("subj")[["d_hs_ms", "d_to_ms"]].describe().round(1).T)


if __name__ == "__main__" and len(sys.argv) > 2 and sys.argv[2] == "edges":
    edge_timing(sys.argv[1])

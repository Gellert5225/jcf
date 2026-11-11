"""Compare ways of carrying the JR-grid label mask onto the 100 Hz training grid (sample of training walking trials)."""
import glob, os, sys, random
import numpy as np, pandas as pd
REPO = "/home/gellert/Developer/medical_dt"
sys.path.insert(0, REPO); os.chdir(REPO)
import train_mlp as tm
random.seed(0)
rows = []
for lab in ("carter", "moore", "tiziana", "falisse", "fregly", "han"):
    ds = sorted(glob.glob(f"jcf/full_duration/training/walking/{lab}_*"))
    ds = [d for d in ds if os.path.exists(f"{d}/jcf_output_sojr/label_mask.npz")]
    for d in random.sample(ds, min(80, len(ds))):
        z = np.load(f"{d}/jcf_output_sojr/label_mask.npz")
        t_jr, keep = z["time"], z["keep_r"]
        ik = pd.read_csv(f"{d}/ik_results.mot", sep=r"\s+", skiprows=6)["time"].to_numpy()
        t_lab = tm._common_time_grid(d, ik, "jcf_output_sojr")
        if len(t_lab) < 2:
            continue
        hz = 100.0
        t_out = t_lab[0] + np.arange(int(np.floor((t_lab[-1] - t_lab[0]) * hz + 1e-9)) + 1) / hz
        bad = np.interp(t_out, t_lab, np.interp(t_lab, t_jr, (~keep).astype(float)))
        # nearest JR frame kept
        j = np.abs(t_jr[None, :] - t_out[:, None]).argmin(1)
        nearest = keep[j]
        # all JR frames within +-5 ms kept (and the nearest)
        win = np.array([keep[(np.abs(t_jr - t) <= 0.005 + 1e-9)].all() for t in t_out]) & nearest
        rows.append(dict(lab=lab, n=len(t_out), dt_jr=np.median(np.diff(t_jr)),
                         tol0=(bad <= 1e-12).sum(), tol01=(bad <= 0.02).sum(), tol05=(bad <= 0.05).sum(), tol10=(bad <= 0.1).sum(),
                         nearest=nearest.sum(), win5=win.sum(),
                         max_bad_in_nearest=float(bad[nearest].max()) if nearest.any() else np.nan,
                         p95_bad_in_nearest=float(np.percentile(bad[nearest], 99)) if nearest.any() else np.nan,
                         max_bad_in_win5=float(bad[win].max()) if win.any() else np.nan))
df = pd.DataFrame(rows)
g = df.groupby("lab").agg(trials=("n", "size"), frames=("n", "sum"), dt_jr=("dt_jr", "median"), tol0=("tol0", "sum"),
                          tol01=("tol01", "sum"), tol05=("tol05", "sum"), tol10=("tol10", "sum"), nearest=("nearest", "sum"), win5=("win5", "sum"),
                          max_bad_nearest=("max_bad_in_nearest", "max"), max_bad_win5=("max_bad_in_win5", "max"))
for c in ("tol0", "tol01", "tol05", "tol10", "win5"):
    g[c + "_vs_nearest"] = (g[c] / g["nearest"]).round(3)
pd.set_option("display.width", 250)
print(g.to_string())

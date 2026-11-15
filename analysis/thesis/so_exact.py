#!/usr/bin/env python3
"""Exact static optimization for a labelled trial folder: re-solve every SO frame to its exact optimum, write it as SO
force / activation files, run OpenSim's JointReaction with those forces, and write the label-QC frame mask.

OpenSim's StaticOptimization (Ipopt) often stops short of the optimum without reporting a failure (William's Nov 14
export: 69 % of right-single-stance frames with cost > 1.1x the optimum, max 2.05x), which shows up as dorsiflexor /
plantarflexor co-activation and saturation that the SO label rules reject. Tightening its tolerance does not converge
(Ipopt "Restoration failed"). This solves the same per-frame problem exactly (as so_label_qc.resolve_so_frame):

    min sum a_i^2 + sum u_k^2   s.t.  a_i in [0.01, 1],
    sum_i (a_i C_i + P_i) r_ij + sum_k u_k F_k [k acts on j] = tau_j   for every unconstrained hip, knee, ankle, lumbar j

with C_i the rigid-tendon active force at full activation (IK low-passed like the AnalyzeTool), r_ij the moment arms,
F_k the coordinate actuators' optimal forces, and tau_j and the passive forces P_i taken from OpenSim's stored solution
(so the stored solution is feasible and the targets are OpenSim's inverse dynamics). SLSQP from the stored solution and
from the previous frame's optimum, the better kept. Pelvis and upper-body actuators keep their stored values; frames
that do not solve keep the stored solution (counted in problem.json; "not converged" counts frames where SLSQP ended at a
feasible point no worse than the stored one without reporting convergence, typically reserve-dominated frames where the
stored solution already is the optimum).

Writes <folder>/<out>/FullDuration_StaticOptimization_{force,activation}.sto, problem.json and per_frame.csv.gz
(time, cost_so, cost_opt, solved), and <folder>/<out>_sojr/ (JointReaction on/in tibia, child; label_mask.npz).
Usage (repo root): python -I analysis/thesis/so_exact.py <folder> [--source jcf_output] [--out jcf_output_exact]
                   [--capsule N_M]   (--capsule: the source run has so_variant.py's capsule actuator)
"""
import argparse
import json
import os
import re
import site
import sys
import tempfile

sys.path.append(site.getusersitepackages())
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from analysis.thesis import so_label_qc as q  # noqa: E402
from analysis.thesis.gc_knee_jr import jr_only_setup, read_sto  # noqa: E402

SO_FORCE_OUT = "FullDuration_StaticOptimization_force.sto"
SO_ACT_OUT = "FullDuration_StaticOptimization_activation.sto"


def lowpass_hz(setup_xml):
    m = re.search(r"<lowpass_cutoff_frequency_for_coordinates>\s*([-0-9.eE+]+)\s*<", open(setup_xml).read())
    return float(m.group(1)) if m else -1.0


def frame_geometry(folder, times, model_file=None, cutoff=6.0):
    """Per time: moment arms (muscles x coordinates) and rigid-tendon active fiber force at full activation, from
    ik_results.mot low-passed at cutoff Hz (4th-order Butterworth, zero lag; none if cutoff <= 0). Returns (muscles,
    coordinates, {actuator: (coordinate, optimal force)}, [(R, cap)] in the order of times)."""
    import opensim as osm
    from scipy.signal import butter, filtfilt
    osm.Logger.setLevelString("error")
    model = osm.Model(model_file or os.path.join(folder, "jcf_output", "_model_with_reserves.osim"))
    ms = model.getMuscles()
    for k in range(ms.getSize()):
        ms.get(k).set_ignore_tendon_compliance(True)
    state = model.initSystem()
    cs = model.getCoordinateSet()
    ik = read_sto(os.path.join(folder, "ik_results.mot"))
    t = ik["time"].to_numpy(float)
    nyq = 0.5 / float(np.median(np.diff(t)))
    filt = cutoff > 0 and cutoff < nyq and len(t) > 15
    if filt:
        bb, aa = butter(4, cutoff / nyq)
    Q = {c: (filtfilt(bb, aa, ik[c].to_numpy(float)) if filt else ik[c].to_numpy(float))
         for c in ik.columns[1:] if cs.contains(c)}
    U = {c: np.gradient(v, t) for c, v in Q.items()}
    mus = [ms.get(k).getName() for k in range(ms.getSize())]
    dofs = [cs.get(k).getName() for k in range(cs.getSize()) if not cs.get(k).isConstrained(state)
            and cs.get(k).getName().startswith(("hip_", "knee_angle", "ankle_angle", "lumbar_"))]
    fs = model.getForceSet()
    opt = {}
    for k in range(fs.getSize()):
        ca = osm.CoordinateActuator.safeDownCast(fs.get(k))
        if ca is not None:
            opt[ca.getName()] = (ca.get_coordinate(), float(ca.getOptimalForce()))
    geo = []
    for time in times:
        i = int(np.argmin(np.abs(t - time)))
        for c in Q:
            cs.get(c).setValue(state, float(Q[c][i]), False)
            cs.get(c).setSpeedValue(state, float(U[c][i]))
        model.assemble(state)
        for m in mus:
            ms.get(m).setActivation(state, 1.0)
        model.realizeVelocity(state)
        R = np.array([[ms.get(m).computeMomentArm(state, cs.get(c)) for c in dofs] for m in mus])
        cap = np.array([ms.get(m).calcInextensibleTendonActiveFiberForce(state, 1.0) for m in mus])
        geo.append((R, cap))
    return mus, dofs, opt, geo


def solve(so, act, mus, dofs, opt, geo):
    """Exact optimum of every frame of a stored solution (so forces, act activations; geo in their row order).
    Returns (per frame: time, cost_so, cost_opt, solved; activations table; forces table; actuators of the problem)."""
    from scipy.optimize import minimize
    acts = [(n, dofs.index(c), f) for n, (c, f) in opt.items() if c in dofs and n in so]
    names = [n for n, _, _ in acts]
    fopt = np.array([f for _, _, f in acts])
    B = np.zeros((len(dofs), len(acts)))
    for k, (_, jj, f) in enumerate(acts):
        B[jj, k] = f
    summ, a_rows, f_rows, x_prev = [], [], [], None
    for j, (R, cap) in enumerate(geo):
        f_so = so.loc[j, mus].to_numpy(float)
        a_so = act.loc[j, mus].to_numpy(float)
        u_so = so.loc[j, names].to_numpy(float) / fopt
        passive = f_so - a_so * cap
        A = np.hstack([R.T * cap[None, :], B])
        rhs = R.T @ f_so + B @ u_so - R.T @ passive
        x0 = np.r_[a_so, u_so]
        best = None
        for start in (x0, x_prev):
            if start is None:
                continue
            s = minimize(lambda x: float(x @ x), start, jac=lambda x: 2 * x, method="SLSQP",
                         bounds=[(0.01, 1.0)] * len(mus) + [(None, None)] * len(acts),
                         constraints=[dict(type="eq", fun=lambda x: A @ x - rhs, jac=lambda x: A)],
                         options=dict(maxiter=1000, ftol=1e-12))
            # a feasible point no worse than the stored solution is kept even when SLSQP does not report convergence:
            # on reserve-dominated frames (stored cost in the thousands) it stops with "positive directional derivative"
            # at the stored cost, which trust-constr confirms is the optimum (Sep 23 export, 4 frames checked)
            if np.abs(A @ s.x - rhs).max() < 1e-3 and s.fun <= x0 @ x0 * (1 + 1e-9) and (best is None or s.fun < best.fun):
                best = s
        x = best.x if best is not None else x0
        x_prev = best.x if best is not None else None
        a, u = x[:len(mus)], x[len(mus):]
        t = float(so.time[j])
        summ.append(dict(time=t, cost_so=float(x0 @ x0), cost_opt=float(x @ x), solved=best is not None,
                         converged=bool(best is not None and best.success)))
        a_rows.append(dict(time=t, **dict(zip(mus, a)), **dict(zip(names, u))))
        f_rows.append(dict(time=t, **dict(zip(mus, a * cap + passive)), **dict(zip(names, u * fopt))))
    return pd.DataFrame(summ), pd.DataFrame(a_rows), pd.DataFrame(f_rows), acts


def write_like(src, values, dst):
    """Copy the .sto src to dst with the columns of values (same rows) replaced."""
    lines = open(src).read().split("\n")
    n = next(i for i, ln in enumerate(lines) if ln.strip().lower() == "endheader")
    cols = lines[n + 1].split()
    tab = read_sto(src)
    if len(tab) != len(values) or not np.allclose(tab.time.to_numpy(), values.time.to_numpy()):
        raise ValueError(f"{src}: rows do not match")
    for c in values.columns:
        if c != "time":
            tab[c] = values[c].to_numpy()
    with open(dst, "w") as fh:
        fh.write("\n".join(lines[:n + 1]) + "\n" + "\t".join(cols) + "\n")
        np.savetxt(fh, tab[cols].to_numpy(float), fmt="%.8f", delimiter="\t")


def view(tmp, folder, so_dir, jr_dir):
    """A folder laid out like a labelled trial (jcf_output/, sojr/) whose SO and JR files are so_dir and jr_dir."""
    d = os.path.join(tmp, os.path.basename(os.path.normpath(folder)))
    os.makedirs(d)
    for name, target in (("metadata.json", "metadata.json"), ("grf_data.mot", "grf_data.mot"),
                         ("ik_results.mot", "ik_results.mot"), ("jcf_output", so_dir), ("sojr", jr_dir)):
        os.symlink(os.path.join(folder, target), os.path.join(d, name))
    return d


def build(folder, source="jcf_output", out="jcf_output_exact", capsule=0.0, geometry=None):
    """Exact SO of <folder>/<source> -> <folder>/<out>/ and the JointReaction + label mask in <folder>/<out>_sojr/.
    geometry: frame_geometry() output to reuse (same folder and SO times). Returns the per-frame summary."""
    import opensim as osm
    folder = os.path.abspath(folder)
    src = os.path.join(folder, source)
    setup = os.path.join(src, "_analyze_setup.xml")
    so = read_sto(q.results_file(src, ".", q.SO_FORCES))
    act = read_sto(q.results_file(src, ".", q.SO_ACT))
    if geometry is None:
        geometry = frame_geometry(folder, so.time.to_numpy(float), cutoff=lowpass_hz(setup))
    mus, dofs, opt, geo = geometry
    opt = dict(opt)
    if capsule > 0:
        opt["capsule_knee_angle_r"] = ("knee_angle_r", float(capsule))
    summ, a, f, acts = solve(so, act, mus, dofs, opt, geo)
    od, jd = os.path.join(folder, out), os.path.join(folder, out + "_sojr")
    os.makedirs(od, exist_ok=True)
    os.makedirs(jd, exist_ok=True)
    fpath = os.path.join(od, SO_FORCE_OUT)
    write_like(q.results_file(src, ".", q.SO_FORCES), f, fpath)
    write_like(q.results_file(src, ".", q.SO_ACT), a, os.path.join(od, SO_ACT_OUT))
    summ.to_csv(os.path.join(od, "per_frame.csv.gz"), index=False)
    ratio = summ.cost_so / summ.cost_opt
    json.dump(dict(source=source, capsule_n_m=capsule, muscles=mus, coordinates=dofs,
                   actuators=[(n, fo) for n, _, fo in acts], frames=len(summ),
                   unsolved_frames=int((~summ.solved).sum()), unsolved_times=summ.time[~summ.solved].round(4).tolist(),
                   not_converged_frames=int((summ.solved & ~summ.converged).sum()),
                   source_nonoptimal_pct=float(100 * np.mean(ratio[summ.solved] > q.COST_RATIO_NONOPT))),
              open(os.path.join(od, "problem.json"), "w"), indent=1)
    xml = os.path.join(jd, "_jr_setup.xml")
    jr_only_setup(setup, fpath, jd).write(xml, encoding="UTF-8", xml_declaration=True)
    osm.Logger.setLevelString("error")
    osm.AnalyzeTool(xml).run()
    with tempfile.TemporaryDirectory() as tmp:
        d = view(tmp, folder, out, out + "_sojr")
        m = q.mask_arrays(d, "sojr")
        m["label_file"] = np.array(os.path.join(out + "_sojr", os.path.basename(str(m["label_file"]))))
        q.write_mask(d, m, "sojr")
    print(f"{os.path.basename(folder)}: {len(summ)} frames, {int((~summ.solved).sum())} not solved (stored kept), "
          f"{int((summ.solved & ~summ.converged).sum())} without a convergence report; "
          f"source not at its optimum on {np.mean(ratio[summ.solved] > q.COST_RATIO_NONOPT):.0%} of solved frames "
          f"-> {od}, {jd}")
    return summ


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("folder")
    ap.add_argument("--source", default="jcf_output", help="SO run to re-solve (sub-folder with _analyze_setup.xml)")
    ap.add_argument("--out", default="jcf_output_exact")
    ap.add_argument("--capsule", type=float, default=0.0, metavar="N_M",
                    help="optimal force of the source run's capsule_knee_angle_r actuator (so_variant.py --capsule)")
    a = ap.parse_args()
    build(a.folder, a.source, a.out, a.capsule)


if __name__ == "__main__":
    main()

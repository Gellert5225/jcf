#!/usr/bin/env python3
"""Rerun StaticOptimization (and the JointReaction pass with the SO forces) on a labelled trial folder with changed SO
settings and/or a hyperextension 'capsule' at the right knee. Writes <folder>/so_variants/<tag>/ (SO outputs,
opensim.log) and <folder>/so_variants/<tag>/sojr/ (JointReaction).

  python analysis/thesis/so_variant.py <folder> <tag> [--set optimizer_convergence_criterion=0.01 ...] [--capsule 1000]

--set     replaces any element of the folder's jcf_output/_analyze_setup.xml (e.g. optimizer_convergence_criterion,
          optimizer_max_iterations, use_muscle_physiology, activation_exponent)
--capsule adds CoordinateActuator capsule_knee_angle_r (optimal force F N*m, control in [0, inf), so it can only apply
          a flexing torque, i.e. resist hyperextension). The Rajagopal knee has no passive hyperextension stop
          (knee_angle range 0-120 deg); with a large optimal force the actuator is cheap in the SO cost, so SO uses it
          for any knee-flexion torque the right knee needs (stance and swing alike). A sensitivity test, not a model.
Scored by analysis/thesis/so_tuning_nov14.py.
"""
import argparse
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "analysis", "thesis"))
import opensim as osm  # noqa: E402
from gc_knee_jr import jr_only_setup  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("folder")
    ap.add_argument("tag")
    ap.add_argument("--set", nargs="*", default=[], metavar="ELEMENT=VALUE")
    ap.add_argument("--capsule", type=float, default=0.0, metavar="N_M")
    a = ap.parse_args()
    folder = os.path.abspath(a.folder)
    out = os.path.join(folder, "so_variants", a.tag)
    os.makedirs(os.path.join(out, "sojr"), exist_ok=True)
    osm.Logger.addFileSink(os.path.join(out, "opensim.log"))
    x = open(os.path.join(folder, "jcf_output", "_analyze_setup.xml")).read()
    x = re.sub(r"<results_directory>.*?</results_directory>", f"<results_directory>{out}</results_directory>", x)
    for kv in a.set:
        k, v = kv.split("=", 1)
        x, n = re.subn(rf"<{k}>.*?</{k}>", f"<{k}>{v}</{k}>", x)
        if n != 1:
            raise SystemExit(f"--set {k}: {n} matches in the setup")
    if a.capsule > 0:
        m = osm.Model(re.search(r"<model_file>(.*?)</model_file>", x).group(1))
        ca = osm.CoordinateActuator("knee_angle_r")
        ca.setName("capsule_knee_angle_r")
        ca.setOptimalForce(a.capsule)
        ca.setMinControl(0.0)
        ca.setMaxControl(float("inf"))
        m.addForce(ca)
        m.finalizeConnections()
        mf = os.path.join(out, "_model_capsule.osim")
        m.printToXML(mf)
        x = re.sub(r"<model_file>.*?</model_file>", f"<model_file>{mf}</model_file>", x)
    setup = os.path.join(out, "_analyze_setup.xml")
    open(setup, "w").write(x)
    osm.AnalyzeTool(setup).run()
    force = next(f for f in os.listdir(out) if f.endswith("StaticOptimization_force.sto"))
    jr_xml = os.path.join(out, "sojr", "_jr_setup.xml")
    jr_only_setup(setup, os.path.join(out, force), os.path.join(out, "sojr")).write(jr_xml, encoding="UTF-8",
                                                                                   xml_declaration=True)
    osm.AnalyzeTool(jr_xml).run()
    print("done", a.tag)


if __name__ == "__main__":
    main()

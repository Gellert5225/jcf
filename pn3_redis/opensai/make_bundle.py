"""Build the OpenSai files that visualize the PN3 driver's human model (SimViz in vizOnly mode).

    python -m pn3_redis.opensai.make_bundle --out results/imu/opensai_bundle [--geometry <OpenSim Geometry dir>] \
        [--opensai-root ~/OpenSai]

Writes, laid out like OpenSai/config_folder (OpenSai_main only loads configs from its own config folder):
    robot_files/rajagopal/Rajagopal2015.urdf + Geometry/*.obj    (meshes converted from OpenSim's .vtp)
    world_files/world_pn3.urdf                                    (robot "pn3", floor, light, camera)
    xml_config_files/pn3_viz.xml                                  (prefix "sai", simviz mode="vizOnly")
    ui/pn3_live.html                                              (sai-interfaces page: F_medial, contact, plot)
With --opensai-root, the three config subfolders are also copied into <root>/config_folder/.

Then: redis-server; ./bin/OpenSai_main pn3_viz.xml (from the OpenSai root); and the driver with --redis and
--viz-wrap-m 4 (see docs/PN3_REDIS_DRIVER.md, "Running it in OpenSai").
"""
import argparse
import glob
import os
import re
import shutil
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from pn3_redis.model import URDF                          # noqa: E402
from pn3_redis.opensai.vtp_to_obj import convert          # noqa: E402

TEMPLATES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")


def find_geometry():
    """OpenSim's Geometry folder in the active or the jcf conda env."""
    roots = [os.environ.get("CONDA_PREFIX", ""), os.path.expanduser("~/miniforge3/envs/jcf"), os.path.expanduser("~/anaconda3/envs/jcf")]
    for r in roots:
        for g in glob.glob(os.path.join(r, "share", "doc", "OpenSim", "**", "Geometry"), recursive=True):
            if glob.glob(os.path.join(g, "femur_r.vtp")):
                return g
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--geometry", default=None, help="folder with OpenSim .vtp meshes (default: search conda envs)")
    ap.add_argument("--opensai-root", default=None, help="OpenSai checkout; copies the config files into its config_folder")
    a = ap.parse_args(argv)
    geo = a.geometry or find_geometry()
    if not geo:
        raise SystemExit("OpenSim Geometry folder not found; pass --geometry")
    robot_dir = os.path.join(a.out, "robot_files", "rajagopal")
    os.makedirs(os.path.join(robot_dir, "Geometry"), exist_ok=True)
    shutil.copy(URDF, os.path.join(robot_dir, "Rajagopal2015.urdf"))
    meshes = sorted(set(re.findall(r'filename="Geometry/([^"]+)\.obj"', open(URDF).read())))
    missing, nv, nt = [], 0, 0
    for m in meshes:
        src = os.path.join(geo, m + ".vtp")
        if not os.path.exists(src):
            missing.append(m)
            continue
        v, t = convert(src, os.path.join(robot_dir, "Geometry", m + ".obj"))
        nv, nt = nv + v, nt + t
    if missing:
        raise SystemExit(f"{len(missing)} meshes missing in {geo}: {missing[:5]} ... (sai-graphics aborts on a missing mesh)")
    for sub, name in (("world_files", "world_pn3.urdf"), ("xml_config_files", "pn3_viz.xml"), ("ui", "pn3_live.html")):
        os.makedirs(os.path.join(a.out, sub), exist_ok=True)
        shutil.copy(os.path.join(TEMPLATES, name), os.path.join(a.out, sub, name))
    print(f"{len(meshes)} meshes converted ({nv} vertices, {nt} triangles) from {geo}")
    if a.opensai_root:
        cf = os.path.join(os.path.expanduser(a.opensai_root), "config_folder")
        if not os.path.isdir(cf):
            raise SystemExit(f"{cf} not found: is --opensai-root an OpenSai checkout?")
        for sub in ("robot_files", "world_files", "xml_config_files"):
            shutil.copytree(os.path.join(a.out, sub), os.path.join(cf, sub), dirs_exist_ok=True)
        print("copied into", cf)
    print("written to", a.out)


if __name__ == "__main__":
    main()

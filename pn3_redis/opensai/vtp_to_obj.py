"""Convert OpenSim .vtp meshes (VTK XML PolyData, ASCII data arrays) to Wavefront .obj, without VTK.

OpenSim ships its bone meshes as .vtp; Rajagopal2015.urdf references the same basenames as .obj, and sai-graphics
aborts on a missing mesh. Vertices are copied unchanged: the URDF's visual origin (rpy = pi/2, 0, 0) already rotates
OpenSim's y-up mesh coordinates into its z-up link frames. Polygons are fan-triangulated.
"""
import os
import xml.etree.ElementTree as ET

import numpy as np


def _array(piece, path, name=None):
    for da in piece.findall(path):
        if name is None or da.get("Name") == name:
            if da.get("format", "ascii") != "ascii":
                raise ValueError(f"only ASCII data arrays are supported, got format={da.get('format')}")
            return np.array((da.text or "").split(), float)
    return None


def read_vtp(path):
    """-> (points (n, 3), list of polygons as vertex-index arrays, normals (n, 3) or None)."""
    piece = ET.parse(path).getroot().find("PolyData/Piece")
    pts = _array(piece, "Points/DataArray").reshape(-1, 3)
    conn = _array(piece, "Polys/DataArray", "connectivity").astype(int)
    offs = _array(piece, "Polys/DataArray", "offsets").astype(int)
    polys = np.split(conn, offs[:-1])
    nrm = _array(piece, "PointData/DataArray", "Normals")
    nrm = nrm.reshape(-1, 3) if nrm is not None and nrm.size == pts.size else None
    return pts, polys, nrm


def write_obj(path, pts, polys, nrm=None):
    with open(path, "w") as f:
        f.write(f"# converted from OpenSim .vtp: {len(pts)} vertices\n")
        for p in pts:
            f.write(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
        if nrm is not None:
            for n in nrm:
                f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")
        nt = 0
        for poly in polys:
            for k in range(1, len(poly) - 1):                       # fan triangulation
                tri = (poly[0] + 1, poly[k] + 1, poly[k + 1] + 1)
                f.write("f " + " ".join(f"{i}//{i}" if nrm is not None else f"{i}" for i in tri) + "\n")
                nt += 1
    return nt


def convert(vtp_path, obj_path):
    pts, polys, nrm = read_vtp(vtp_path)
    os.makedirs(os.path.dirname(os.path.abspath(obj_path)), exist_ok=True)
    return len(pts), write_obj(obj_path, pts, polys, nrm)

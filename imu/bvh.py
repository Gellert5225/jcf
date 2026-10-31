"""Minimal BVH reader and forward kinematics, aimed at Perception Neuron / Axis Studio exports.

    bvh = read("walk.bvh")
    fk = forward_kinematics(bvh, scale=0.01)        # Axis Studio BVH is in centimetres
    pos, rot = fk["RightLeg"]                        # pos (F, 3) in metres, rot: scipy Rotation of length F

Conventions (standard BVH):
- Each joint's local rotation is the product of its rotation channels in the order they are listed,
  applied as intrinsic rotations: CHANNELS Zrotation Xrotation Yrotation  ->  R = Rz @ Rx @ Ry.
- Global rotation  R_g(joint) = R_g(parent) @ R_local(joint).
- Global position  p_g(joint) = p_g(parent) + R_g(parent) @ t_local(joint), where t_local is the OFFSET,
  or, for joints that carry position channels (Axis Studio "with displacement"), the channel values.
- The root's position is OFFSET + its position channels.
Global rotations are what OpenSense needs (segment orientations in a fixed world frame).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation

_ROT = {"Xrotation": "X", "Yrotation": "Y", "Zrotation": "Z"}
_POS = ("Xposition", "Yposition", "Zposition")


@dataclass
class Joint:
    name: str
    parent: str | None
    offset: np.ndarray
    channels: list = field(default_factory=list)
    children: list = field(default_factory=list)
    end_site: np.ndarray | None = None


@dataclass
class BVH:
    joints: dict            # name -> Joint
    order: list             # joint names in file order (channel layout order)
    frame_time: float
    frames: np.ndarray      # (n_frames, n_channels)
    columns: dict           # name -> (start, stop) column range in `frames`

    @property
    def fs(self):
        return 1.0 / self.frame_time

    @property
    def n_frames(self):
        return self.frames.shape[0]

    @property
    def root(self):
        return self.order[0]

    def channel(self, joint, name):
        """All frames of one channel of one joint, e.g. channel("Hips", "Yposition")."""
        start, _ = self.columns[joint]
        return self.frames[:, start + self.joints[joint].channels.index(name)]


def parse(text):
    tokens = text.split()
    i = 0

    def take():
        nonlocal i
        tok = tokens[i]
        i += 1
        return tok

    if take().upper() != "HIERARCHY":
        raise ValueError("not a BVH file: missing HIERARCHY")
    joints, order, columns, stack = {}, [], {}, []
    n_cols = 0
    current = None
    while True:
        tok = take()
        up = tok.upper()
        if up in ("ROOT", "JOINT"):
            name = take()
            if name in joints:
                raise ValueError(f"duplicate joint name {name!r}")
            parent = stack[-1] if stack else None
            joints[name] = Joint(name, parent, np.zeros(3))
            if parent:
                joints[parent].children.append(name)
            order.append(name)
            current = name
        elif up == "END":
            take()                                   # "Site"
            if take() != "{":
                raise ValueError("expected '{' after End Site")
            if take().upper() != "OFFSET":
                raise ValueError("expected OFFSET in End Site")
            joints[stack[-1]].end_site = np.array([float(take()) for _ in range(3)])
            if take() != "}":
                raise ValueError("expected '}' closing End Site")
        elif tok == "{":
            stack.append(current)
        elif tok == "}":
            stack.pop()
            if not stack:
                break
        elif up == "OFFSET":
            joints[current].offset = np.array([float(take()) for _ in range(3)])
        elif up == "CHANNELS":
            n = int(take())
            chans = [take() for _ in range(n)]
            for c in chans:
                if c not in _ROT and c not in _POS:
                    raise ValueError(f"unknown channel {c!r} on joint {current!r}")
            joints[current].channels = chans
            columns[current] = (n_cols, n_cols + n)
            n_cols += n
        else:
            raise ValueError(f"unexpected token {tok!r} in HIERARCHY")

    if take().upper() != "MOTION":
        raise ValueError("expected MOTION section")
    if take().rstrip(":").upper() != "FRAMES":
        raise ValueError("expected 'Frames:'")
    n_frames = int(take())
    if take().upper() != "FRAME" or take().rstrip(":").upper() != "TIME":
        raise ValueError("expected 'Frame Time:'")
    frame_time = float(take())
    values = np.array(tokens[i:i + n_frames * n_cols], dtype=float)
    if values.size != n_frames * n_cols:
        raise ValueError(f"MOTION has {values.size} values, expected {n_frames} x {n_cols}")
    return BVH(joints, order, frame_time, values.reshape(n_frames, n_cols), columns)


def read(path):
    with open(path) as f:
        return parse(f.read())


def local_rotations(bvh, joint):
    """scipy Rotation (n_frames) of one joint relative to its parent."""
    j = bvh.joints[joint]
    rot_ch = [c for c in j.channels if c in _ROT]
    if not rot_ch:
        return Rotation.identity(bvh.n_frames)
    seq = "".join(_ROT[c] for c in rot_ch)               # uppercase = intrinsic, as BVH intends
    angles = np.column_stack([bvh.channel(joint, c) for c in rot_ch])
    return Rotation.from_euler(seq, angles, degrees=True)


def local_translations(bvh, joint):
    """(n_frames, 3) translation of one joint in its parent's frame (BVH units)."""
    j = bvh.joints[joint]
    pos_ch = [c for c in _POS if c in j.channels]
    if len(pos_ch) == 3:
        t = np.column_stack([bvh.channel(joint, c) for c in _POS])
        return t + j.offset if j.parent is None else t
    return np.tile(j.offset, (bvh.n_frames, 1))


def forward_kinematics(bvh, scale=1.0, joints=None):
    """Global (positions [F, 3] * scale, rotations) for every joint, or only for `joints` and their ancestors."""
    wanted = set(bvh.order) if joints is None else set(joints)
    if joints is not None:
        for name in list(wanted):
            p = bvh.joints[name].parent
            while p:
                wanted.add(p)
                p = bvh.joints[p].parent
    out = {}
    for name in bvh.order:                              # parents always come before children
        if name not in wanted:
            continue
        j = bvh.joints[name]
        r_loc, t_loc = local_rotations(bvh, name), local_translations(bvh, name)
        if j.parent is None:
            out[name] = (t_loc * scale, r_loc)
        else:
            p_pos, p_rot = out[j.parent]
            out[name] = (p_pos + p_rot.apply(t_loc) * scale, p_rot * r_loc)
    return out if joints is None else {k: out[k] for k in joints}

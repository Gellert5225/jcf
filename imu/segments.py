"""Which suit segment drives which OpenSim body (Rajagopal model), as OpenSense IMU labels.

OpenSense attaches an IMU labelled "<body>_imu" to the OpenSim body "<body>". The BVH names below are
Axis Studio's skeleton naming; spine and head naming differs between Axis Studio versions, so each
OpenSim body lists candidates and the first one present in the file is used.

Confirm against a real Perception Neuron 3 export before relying on the spine, head and arm entries.
The lower-limb and pelvis entries are what the knee surrogate needs.
"""

# OpenSense label -> candidate BVH joint names, in order of preference
PERCEPTION_NEURON = {
    "pelvis_imu":   ["Hips"],
    "femur_r_imu":  ["RightUpLeg"],
    "tibia_r_imu":  ["RightLeg"],
    "calcn_r_imu":  ["RightFoot"],
    "femur_l_imu":  ["LeftUpLeg"],
    "tibia_l_imu":  ["LeftLeg"],
    "calcn_l_imu":  ["LeftFoot"],
    "torso_imu":    ["Spine2", "Spine3", "Spine1", "Chest", "Spine"],
    "humerus_r_imu": ["RightArm"],
    "radius_r_imu":  ["RightForeArm"],
    "hand_r_imu":    ["RightHand"],
    "humerus_l_imu": ["LeftArm"],
    "radius_l_imu":  ["LeftForeArm"],
    "hand_l_imu":    ["LeftHand"],
}

# the subset the knee surrogate depends on (pelvis, both legs, trunk for the COM)
LOWER_BODY = ["pelvis_imu", "femur_r_imu", "tibia_r_imu", "calcn_r_imu",
              "femur_l_imu", "tibia_l_imu", "calcn_l_imu", "torso_imu"]


def resolve(available_joints, labels=None, table=PERCEPTION_NEURON):
    """OpenSense label -> BVH joint actually present. Raises if a requested label has no candidate."""
    labels = list(table) if labels is None else labels
    avail = set(available_joints)
    out, missing = {}, []
    for lab in labels:
        hit = next((c for c in table[lab] if c in avail), None)
        if hit is None:
            missing.append(f"{lab} (tried {table[lab]})")
        else:
            out[lab] = hit
    if missing:
        raise KeyError("no BVH joint for: " + "; ".join(missing))
    return out


def body_of(label):
    """OpenSense label -> OpenSim body name ("femur_r_imu" -> "femur_r")."""
    if not label.endswith("_imu"):
        raise ValueError(f"OpenSense IMU labels end in '_imu': {label!r}")
    return label[:-4]

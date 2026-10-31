"""IMU groundwork: portable capture (Perception Neuron 3) -> OpenSim joint angles -> knee-load surrogate.

Pipeline, one module per step:

    bvh.py            read Axis Studio BVH exports; forward kinematics to per-segment orientations
    segments.py       map suit segments to OpenSim bodies (pelvis, femur, tibia, calcn, torso, ...)
    opensense_io.py   read/write OpenSense orientation files (.sto of quaternions)
    from_bvh.py       CLI: BVH export -> orientation file, starting at the calibration (standing) frame
    synthetic.py      make virtual IMU data from existing mocap trials, for validation    [needs opensim]
    opensense_ik.py   OpenSense calibration (IMUPlacer) and IMU inverse kinematics         [needs opensim]
    contact.py        foot contact without force plates: from foot positions, or from foot-IMU orientation alone
    evaluate_contact.py     tune/score the orientation contact detector against force-plate flags [needs opensim]
    validate_synthetic.py   mocap trial -> virtual IMUs -> OpenSense IK -> angles, contact and F_medial vs mocap
                            and OpenSim labels                                                   [needs opensim]

Modules marked [needs opensim] run in the OpenSim conda env (conda run -n jcf ...); the rest are numpy/scipy only.
See docs/IMU_PIPELINE.md.
"""

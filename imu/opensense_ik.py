"""OpenSense calibration and IMU inverse kinematics.  [needs opensim]

    conda run -n jcf python -m imu.opensense_ik --model scaled_model.osim --orientations orientations.sto \
        --out results/imu/run1 [--t0 1.0] [--heading-axis x] [--sensor-to-opensim 0 0 0]

1. Calibration (IMUPlacer): assumes the subject holds the model's default pose at the first frame of the
   orientation file, and places each "<body>_imu" sensor on its body so the two agree. The base IMU's
   heading axis is aligned with the model's forward (x) direction.
2. IMU IK (IMUInverseKinematicsTool): solves joint angles that best match the sensor orientations.

Orientation sensors give orientations only: the pelvis translations are not observed and come back
unchanged from the calibration pose. Positions relative to the pelvis (COM, feet) are still correct,
which is what the no-pelvis-translation surrogate uses.

--sensor-to-opensim rotates the sensor world frame into OpenSim's (x forward, y up, z right), as X Y Z
rotations in radians. Synthetic data is already in OpenSim's frame (0 0 0). BVH from Axis Studio is y-up,
so only the heading needs aligning, which --heading-axis does.
"""
import argparse
import os


def calibrate(model_path, orientations_path, out_model_path, base_imu="pelvis_imu", heading_axis="x",
              sensor_to_opensim=(0.0, 0.0, 0.0)):
    import opensim as osim
    placer = osim.IMUPlacer()
    placer.set_model_file(os.path.abspath(model_path))
    placer.set_orientation_file_for_calibration(os.path.abspath(orientations_path))
    placer.set_sensor_to_opensim_rotations(osim.Vec3(*sensor_to_opensim))
    if base_imu:
        placer.set_base_imu_label(base_imu)
        placer.set_base_heading_axis(heading_axis)
    placer.run(False)
    placer.getCalibratedModel().printToXML(os.path.abspath(out_model_path))
    return out_model_path


def run_ik(calibrated_model_path, orientations_path, out_dir, t0=None, t1=None, sensor_to_opensim=(0.0, 0.0, 0.0),
           out_name="imu_ik.mot"):
    import opensim as osim
    os.makedirs(out_dir, exist_ok=True)
    ik = osim.IMUInverseKinematicsTool()
    ik.set_model_file(os.path.abspath(calibrated_model_path))
    ik.set_orientations_file(os.path.abspath(orientations_path))
    ik.set_sensor_to_opensim_rotations(osim.Vec3(*sensor_to_opensim))
    ik.set_results_directory(os.path.abspath(out_dir))
    ik.set_output_motion_file(out_name)
    if t0 is not None:
        ik.set_time_range(0, float(t0))
    if t1 is not None:
        ik.set_time_range(1, float(t1))
    ik.run(False)
    path = os.path.join(out_dir, out_name)
    if not os.path.exists(path):                       # some OpenSim versions ignore the name and prefix "ik_"
        base = os.path.splitext(os.path.basename(orientations_path))[0]
        alt = os.path.join(out_dir, f"ik_{base}.mot")
        if os.path.exists(alt):
            os.replace(alt, path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"IMU IK finished but wrote no motion file in {out_dir}")
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="scaled OpenSim model")
    ap.add_argument("--orientations", required=True, help="OpenSense orientation .sto; first frame = calibration pose")
    ap.add_argument("--out", required=True)
    ap.add_argument("--t0", type=float, default=None, help="IK start time (e.g. after the calibration segment)")
    ap.add_argument("--t1", type=float, default=None)
    ap.add_argument("--base-imu", default="pelvis_imu")
    ap.add_argument("--heading-axis", default="x", help="axis of the base IMU that points forward in the calibration pose")
    ap.add_argument("--sensor-to-opensim", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    cal = calibrate(a.model, a.orientations, os.path.join(a.out, "calibrated_model.osim"), a.base_imu,
                    a.heading_axis, tuple(a.sensor_to_opensim))
    print("calibrated model:", cal)
    print("IK motion:", run_ik(cal, a.orientations, a.out, a.t0, a.t1, tuple(a.sensor_to_opensim)))


if __name__ == "__main__":
    main()

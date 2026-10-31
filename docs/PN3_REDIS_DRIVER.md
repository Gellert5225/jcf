# PN3 → Redis driver for OpenSai

OpenSai has no plug-in mechanism for IMUs, so the Perception Neuron 3 suit gets a separate driver process that
publishes to Redis, like `optitrack_redis`. Code in `pn3_redis/`, tests in `tests/test_pn3_redis.py`.

```
Axis Studio ──UDP──► source ──► calibration ──► contact ──► UKF (19 joints + pelvis) ──► Redis keys ──► OpenSai / monitor
 (or a BVH / .sto replay)        (standing)     (foot gyro      contact-anchored base      + stream
                                                 or Axis flags)       │
                                                                       └─► knee-load surrogate ─► f_medial (optional)
```

## What the suit gives, and what the driver adds

Each PN3 sensor fuses its own gyroscope, accelerometer and magnetometer on board (AHRS), and Axis Studio fits a
skeleton, so the stream already carries fused segment orientations. A Kalman filter is still needed. The driver
runs a UKF that turns those orientations into the joint state of the Rajagopal model used by OpenSai and the
surrogate, and estimates the floating base, which the suit does not measure:

- **State:** 19 joint coordinates (pelvis 6, hips 3+3, knees, ankles, lumbar 3) and their velocities, with a
  constant-velocity process model.
- **Orientation measurements:** one per sensor (pelvis, thighs, shanks, feet, torso). The residual is
  `log(R_meas^T R_pred(q))`; the prediction is forward kinematics of the Rajagopal URDF re-expressed in OpenSim's
  convention, with the walker knee tabulated from OpenSim (`knee_table.npz`). It matches OpenSim's body orientations
  exactly.
- **Contact measurements:** while a foot is planted, its stance point (heel, then toe tip once the heel lifts) stays
  where it touched down, at floor height. This carries the pelvis forward through the planted leg (legged odometry).
- **Contact source:** Axis Studio's grounding flags when the source provides them. Otherwise two online detectors on
  the foot sensor's angular speed:
  - "planted": foot flat, used for anchoring
  - "stance": tuned toward force-plate stance, published and fed to the surrogate

  The stance detector turns on at foot-flat, about 90 ms after heel strike, which is when the first knee-load peak
  builds. The surrogate was trained on force-plate flags, so it would read that peak low. Each time stance turns on,
  the detector therefore also dates the heel strike after the fact. It walks back from the switch over the
  foot-slap spike to the speed dip where the foot's rotation reverses at initial contact. The surrogate then relabels
  its buffered window from that frame on. This is possible because the surrogate already waits ~90 ms for its centred
  window. The published `::contact` stays causal. `--backdate-s` sets the look-back (default 0.2 s, 0 = off).

## Run it

Requirements: numpy, scipy, `pip install redis` for Redis output, torch for `--surrogate`, and Noitom's MocapApi
Python wrapper for live data (`pip install .` in `demo/demo-py/MocapApi` of https://github.com/pnmocap/MocapApi).

**Demo without the suit** (a held-out mocap trial replayed as a 60 Hz suit with 1° orientation noise). Start a
Redis server, then in one terminal:

```bash
python -m pn3_redis.driver --source sto results/imu/driver_demo/take_P010_t10_noise1deg/orientations.sto \
    --resample-hz 60 --realtime --loop --height 1.7 --mass 54 \
    --surrogate models/runs/mlp/best_model_mlp_walking_qinst_notrans_pelvisrel_winpos_contact_w20_fmW0p5_e100_2.pt \
    --redis redis://localhost:6379/0
```

And in a second terminal:

```bash
python -m pn3_redis.monitor --redis redis://localhost:6379/0
```

**Live from the suit:**
1. In Axis Studio: Settings › BVH Broadcasting, enable **BVH‑Capture**, UDP, destination = this machine's IP,
   port 7012, displacement on. For a recorded take, play it and enable **BVH‑Edit** instead.
2. The subject stands still, upright, feet straight, for the first second (calibration).
3. Run:

```bash
python -m pn3_redis.driver --source mocapapi --udp-port 7012 --sensor-frame axis --heading-axis z \
    --height 1.75 --mass 70 --redis redis://localhost:6379/0 [--surrogate ...]
```

**Replay an Axis Studio BVH export:**

```bash
python -m pn3_redis.driver --source bvh take01.bvh --sensor-frame axis --heading-axis z --realtime --redis ...
```

**Running it in OpenSai** (3D view of the human model; OpenSai's SimViz in `vizOnly` mode, no simulation or
controller). Tested: the model walks correctly in OpenSai's 3D view with the demo take (October 2026).

1. Build the OpenSai files. This converts OpenSim's 81 bone meshes to `.obj`, which sai-graphics needs and aborts
   without:
   ```bash
   python -m pn3_redis.opensai.make_bundle --out results/imu/opensai_bundle --opensai-root ~/OpenSai
   ```
   It writes `robot_files/rajagopal/` (URDF + meshes), `world_files/world_pn3.urdf` (robot `pn3`, floor, cameras)
   and `xml_config_files/pn3_viz.xml` (prefix `sai`, `mode="vizOnly"`). `--opensai-root` copies them into
   OpenSai's `config_folder`, the only place `OpenSai_main` loads configs from.
2. Start `redis-server` yourself. OpenSai's `launch.sh` kills a Redis it started when it exits.
3. From the OpenSai root: `./bin/OpenSai_main pn3_viz.xml`.
4. Run the driver against the same Redis:
   ```bash
   python -m pn3_redis.driver --source sto results/imu/driver_demo/take_P010_t10_noise1deg/orientations.sto \
       --resample-hz 60 --realtime --loop --height 1.7 --mass 54 --viz-wrap-m 4 \
       --surrogate models/runs/mlp/best_model_mlp_walking_qinst_notrans_pelvisrel_winpos_contact_w20_fmW0p5_e100_2.pt \
       --redis redis://localhost:6379/0
   ```
   SimViz reads `sai::sensors::pn3::joint_positions` and draws at 30 Hz. `--viz-wrap-m 4` keeps the walking body
   within the camera's view (it only affects that key). The window opens on the side camera; N / B switch
   cameras.
5. Optional live numbers (`::f_medial` exists only when the driver runs with `--surrogate`). Two ways:
   - **Our page:** `python3 <sai-interfaces>/ui/server.py results/imu/opensai_bundle/ui/pn3_live.html -hp 8001`,
     then open `http://localhost:8001`.
   - **OpenSai's own UI:** after step 3 has generated it, run from the OpenSai root
     `python3 bin/ui/server.py config_folder/xml_config_files/webui_generated_file/webui.html`. Open
     `http://localhost:8000`, click "New online plot tab", and pick `sai::sensors::pn3::f_medial` / `::contact`.

   The driver must run without `--stream`: a Redis stream key breaks the web UI's key cache for the whole
   database, and the UI server must be restarted if it saw one.

Notes:
- Run only one driver per Redis server. Two instances write the same keys, and the model flickers between their
  two poses. Check with `ps aux | grep pn3_redis`.
- OpenSai seeds `joint_positions` with the model's mid-limit pose at start-up and on every reset, until the next
  driver frame.
- It needs a display (GLFW window).
- Closing the 3D window stops OpenSai.

**Score a replay against ground truth** (virtual-suit data only):

```bash
python -m pn3_redis.driver --source sto <synth>/orientations.sto --resample-hz 60 --log run.csv ...
python -m pn3_redis.evaluate --log run.csv --synthetic <synth> --trial <trial folder> --out <dir>
```

## Redis keys

Namespace `sai::sensors::pn3` (`--namespace`). Vectors are JSON arrays, the representation sai-common's
`RedisClient` decodes. It splits on commas, so a space-separated vector would silently decode as one number.
Non-finite values are never written.

| Key | Content |
|---|---|
| `::timestamp` | source time, s |
| `::joint_names` | the 19 coordinates of `::q` (written once) |
| `::q`, `::dq`, `::q_std` | joint positions, velocities and filter std, OpenSim convention (rad, m) |
| `::q_urdf` (+ `::urdf_joint_names`) | 37-vector in `assets/Rajagopal2015.urdf` order; pelvis translation in the robot frame (x fwd, y left, z up), rotations copied, the convention of `infer_trajectory_csv.py` |
| `::joint_positions`, `::joint_velocities` | what OpenSai's SimViz reads for robot `pn3`: the same 37 joints in sai-model order (= URDF file order). Differences from `::q_urdf`: the pelvis rotations are re-expressed for the URDF's literal pelvis axes, so OpenSai renders the real pelvis orientation; translations are in the unscaled URDF's units; and `pelvis_tx` is wrapped with `--viz-wrap-m` |
| `::pelvis_position`, `::com_position` | OpenSim ground frame (x fwd, y up, z right), m |
| `::contact` | `[right, left]` 0/1 |
| `::f_medial`, `::fy`, `::mx`, `::surrogate_time` | knee-load surrogate output and the time it refers to (≈ 90 ms behind) |
| `::sensor::<label>::orientation` | raw sensor quaternion w,x,y,z (`--publish-sensors`) |
| `::status` | JSON `{calibrated, frame, rate_hz}` |
| `::stream` | only with `--stream`: Redis stream with every frame (t, q, pelvis_position, contact, f_medial) for `pn3_redis.monitor`. Off by default, since it breaks the sai-interfaces web UI; the monitor otherwise samples the latest-value keys |

## Validation on virtual-suit data

The data is held-out Carter P010, trial 10: treadmill walking replayed as a 60 Hz suit, online contact, no force
plates. Metrics are in `results/imu/driver_demo/`.

| | clean | 1° noise |
|---|---|---|
| worst joint-angle RMSE (16 surrogate inputs) | 0.9° | 2.1° |
| pelvis height / lateral RMSE | 0.9 / 1.7 cm | 1.1 / 1.6 cm |
| forward speed vs treadmill belt | 1.34 vs 1.30 m/s | 1.33 vs 1.30 m/s |
| contact agreement with force plates, published (causal) | 0.85–0.90 | 0.87–0.88 |
| contact agreement, as fed to the surrogate (heel strike back-dated) | 0.94–0.96 | 0.96 |
| live F_medial MAE vs OpenSim | 0.23 BW (r 0.95) | 0.23 BW (r 0.96) |
| cost per frame (one CPU thread) | ≈ 3.5 ms + 1 ms surrogate | |

Contact timing was the main error source. Without back-dating (`--backdate-s 0`), the stance flags come on ~90 ms
after heel strike and the live MAE is 0.35 BW clean and 0.39 BW at 1° noise. With force-plate contact it is
0.22 BW. Back-dating gets to 0.23 BW. For comparison, the surrogate fed lab mocap and force plates scores 0.21 BW on
this trial. It also holds at 2° noise (0.24 vs 0.41 BW without) and with 0.5°/s drift (0.27 vs 0.36). On a
training trial (Carter P004 t13, where the rule was tuned) it scores 0.32 vs 0.48. All of this is one held-out trial;
a wider check across held-out people is still to do.

## Limitations and open questions

- **Untested on hardware.** The live MocapApi source is written against Noitom's documented Python API and tested
  with a fake; the frame conventions (`--sensor-frame axis`, `--heading-axis z`) need checking on the first take.
- **Generic body model.** The Rajagopal URDF is scaled uniformly by height, not a subject-specific model.
- **Rigid foot.** Subtalar and MTP are locked, so the pelvis dips by ~3 cm when the trailing foot pivots onto its
  toes.
- **Knee in OpenSai's view.** OpenSai draws the URDF's hinge knee. That hinge cannot show the walker knee's coupled
  twist, so the rendered shank and foot are twisted about the shank axis by up to ~15° at deep flexion (3.5° at
  10°, 13° at 60°) compared with the estimator and OpenSim. This is visual only: flexion is correct, and `::q` is
  unaffected.
- **No pelvis translation on the treadmill.** Planted-foot anchoring turns treadmill walking into
  overground-equivalent progress (which is what the suit would see overground).
- **URDF pelvis axes (for William).** `Rajagopal2015.urdf` keeps OpenSim's raw pelvis rotation axes in a z-up frame,
  so the coordinate named `pelvis_tilt` rotates about the vertical. The repo's converters copy those three values
  as OpenSim's. `::q` is OpenSim-correct; `::joint_positions` is correct for the URDF as OpenSai interprets it;
  `::q_urdf` follows the existing copy convention until this is confirmed.

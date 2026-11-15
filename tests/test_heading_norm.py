"""Heading removal for the *_hd input sets (inference_posture_surrogate.remove_heading).

A trial rotated about the vertical must give the same training windows (train_mlp._windowize_positions_only with
heading_norm) and the same predictions (JCFPostureSurrogate), and training and inference must build the same features.
Run: python -I tests/test_heading_norm.py   (or pytest)
"""
import os
import site
import sys
import tempfile

sys.path.append(site.getusersitepackages())
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import numpy as np  # noqa: E402

from inference_posture_surrogate import (_rot, opensim_pelvis_angles, opensim_pelvis_matrix,  # noqa: E402
                                         pelvis_heading, remove_heading)

HD_SET = "qinst_notrans_pelvisrel_winpos_contact_fpopt_w20_t100_sojrqc_hd"
T, WIN = 80, 20


def synthetic_trial(seed=0, heading_deg=37.0):
    """q16 (pelvis tilt / list / rotation with a gait-like yaw oscillation about a heading, zero translations, legs),
    COM / right / left foot relative to the pelvis (T, 3) in ground axes, contact flags (T, 2)."""
    rng = np.random.default_rng(seed)
    t = np.arange(T) * 0.01
    q = np.zeros((T, 16))
    q[:, 0] = np.radians(5 + 3 * np.sin(2 * np.pi * t))              # tilt
    q[:, 1] = np.radians(4 * np.sin(2 * np.pi * t + 0.4))            # list
    q[:, 2] = np.radians(heading_deg + 6 * np.sin(2 * np.pi * t))    # rotation (yaw about the heading)
    q[:, 6:] = rng.normal(0, 0.3, (T, 10))
    vec = [rng.normal(0, 0.3, (T, 3)) for _ in range(3)]
    contact = (rng.random((T, 2)) > 0.5).astype(float)
    return q, vec, contact


def yaw_trial(q, vec, psi):
    """The same trial rotated about the vertical by psi (rad): pelvis orientation and all ground vectors."""
    Ry = _rot(1, np.full(len(q), psi))
    q = q.copy()
    q[:, 0:3] = opensim_pelvis_angles(Ry @ opensim_pelvis_matrix(q[:, 0:3]))
    return q, [(Ry @ v[..., None])[..., 0] for v in vec]


def training_windows(q, vec, contact, heading_norm=True):
    from train_mlp import _windowize_positions_only
    feat = np.hstack([np.delete(q, [3, 4, 5], axis=1), *vec, contact, np.full((T, 1), 70.0), np.full((T, 1), 1.75)])
    return _windowize_positions_only(feat.astype(np.float32), WIN, q_dim=13, n_pos=11, heading_norm=heading_norm)


def window_of(x, k):
    return x[np.clip(np.arange(k - WIN // 2, k - WIN // 2 + WIN), 0, T - 1)]


def test_matrix_roundtrip():
    rng = np.random.default_rng(1)
    a = np.column_stack([rng.uniform(-0.5, 0.5, 200), rng.uniform(-0.5, 0.5, 200), rng.uniform(-3.1, 3.1, 200)])
    assert np.allclose(opensim_pelvis_angles(opensim_pelvis_matrix(a)), a, atol=1e-10)


def test_remove_heading_zeroes_heading_and_keeps_vectors():
    q, vec, _ = synthetic_trial()
    ang, (v,) = remove_heading(q[:, 0:3], vec[0])
    assert np.allclose(pelvis_heading(ang), 0.0, atol=1e-10)
    assert np.allclose(v[:, 1], vec[0][:, 1])                                     # vertical untouched
    assert np.allclose(np.linalg.norm(v, axis=1), np.linalg.norm(vec[0], axis=1))
    # OpenSim's tilt (about ground z) and list (ground x) come before the yaw, so they mean sagittal / frontal only for
    # a subject facing ~+x; at a 0-deg heading (yaw oscillation only) removing the heading leaves them nearly unchanged
    q0, _, _ = synthetic_trial(heading_deg=0.0)
    ang0, _ = remove_heading(q0[:, 0:3])
    assert np.allclose(ang0[:, 0:2], q0[:, 0:2], atol=0.03)   # mixing ~ sin(6 deg yaw) x 8 deg tilt ~ 0.015 rad


def test_training_windows_invariant_to_yaw():
    q, vec, contact = synthetic_trial()
    ref = training_windows(q, vec, contact)
    for psi in np.radians([-90.0, 45.0, 90.0, 180.0]):
        qr, vr = yaw_trial(q, vec, psi)
        assert np.allclose(training_windows(qr, vr, contact), ref, atol=1e-5), psi
        assert not np.allclose(training_windows(qr, vr, contact, heading_norm=False),
                               training_windows(q, vec, contact, heading_norm=False), atol=1e-3)


def _dummy_hd_checkpoint(path):
    import torch
    from inference_posture_surrogate import _JCF_MLP
    torch.manual_seed(0)
    m = _JCF_MLP(n_features=235, n_outputs=4, hidden=128, dropout=0.0, activation="relu")
    torch.save(dict(input_set=HD_SET, n_features=235, n_outputs=4, hidden=128, activation="relu",
                    model_state_dict=m.state_dict(), input_mean=torch.zeros(235), input_std=torch.ones(235),
                    foot_point=[0.13, -0.01, 0.0], mx_medial_sign=-1.0, heading_norm=True), path)


def _inference_inputs(q, vec, contact, ks):
    return dict(joint_angles=q[ks], com_rel_window=np.stack([window_of(vec[0], k) for k in ks]),
                calcn_r_rel_window=np.stack([window_of(vec[1], k) for k in ks]),
                calcn_l_rel_window=np.stack([window_of(vec[2], k) for k in ks]),
                r_contact_window=np.stack([window_of(contact[:, 0], k) for k in ks]),
                l_contact_window=np.stack([window_of(contact[:, 1], k) for k in ks]))


def test_inference_matches_training_and_is_invariant():
    from inference_posture_surrogate import JCFPostureSurrogate
    q, vec, contact = synthetic_trial(seed=2)
    ks = np.arange(0, T, 7)
    with tempfile.TemporaryDirectory() as tmp:
        ck = os.path.join(tmp, "hd.pt")
        _dummy_hd_checkpoint(ck)
        sur = JCFPostureSurrogate(ck, device="cpu")
        assert sur.heading_norm
        kw = _inference_inputs(q, vec, contact, ks)
        x = sur._build_feature_windowed(kw["joint_angles"], kw["com_rel_window"], kw["calcn_r_rel_window"],
                                        kw["calcn_l_rel_window"], 70.0, 1.75, r_contact_w=kw["r_contact_window"],
                                        l_contact_w=kw["l_contact_window"]).cpu().numpy()
        assert np.allclose(x, training_windows(q, vec, contact)[ks], atol=1e-5)
        ref = sur.predict_windowed(mass_kg=70.0, height_m=1.75, **kw)
        for psi in np.radians([90.0, -135.0]):
            qr, vr = yaw_trial(q, vec, psi)
            out = sur.predict_windowed(mass_kg=70.0, height_m=1.75, **_inference_inputs(qr, vr, contact, ks))
            for key in ("fy", "mx", "f_medial"):
                assert np.allclose(out[key], ref[key], atol=1e-4), (psi, key)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)

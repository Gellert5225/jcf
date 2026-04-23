import os, torch, numpy as np
from train_cnn import load_subject, JCF_CNN_v2

MODEL_PATH = "./jcf/full_duration/training/running/best_model_i.pt"
TEST_ROOT = "./jcf/testing/running"

checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
model = JCF_CNN_v2(n_features=checkpoint['n_features'], n_outputs=3)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

input_mean = checkpoint['input_mean']
input_std = checkpoint['input_std']
lower_body_only = checkpoint.get('lower_body_only', False)
clean_features = checkpoint.get('clean_features', False)
jcf_subdir = checkpoint.get('jcf_subdir', 'jcf_output')

n_feat = checkpoint['n_features']
print(f"n_features = {n_feat}, lower_body={lower_body_only}, clean={clean_features}")

n_joints = 16
right_leg_pos = list(range(6, 11))
left_leg_pos = list(range(11, 16))

groups = {
    'pelvis_pos':     list(range(0, 6)),
    'right_leg_pos':  right_leg_pos,
    'left_leg_pos':   left_leg_pos,
    'pelvis_vel':     [i + n_joints for i in range(0, 6)],
    'right_leg_vel':  [i + n_joints for i in right_leg_pos],
    'left_leg_vel':   [i + n_joints for i in left_leg_pos],
    'pelvis_acc':     [i + 2*n_joints for i in range(0, 6)],
    'right_leg_acc':  [i + 2*n_joints for i in right_leg_pos],
    'left_leg_acc':   [i + 2*n_joints for i in left_leg_pos],
    'grf_right':      [48, 49, 50],
    'grf_left':       [51, 52, 53],
    'grf_vel_right':  [54, 55, 56],
    'grf_vel_left':   [57, 58, 59],
}

grad_accum = {k: 0.0 for k in groups}
total_frames = 0

for name in sorted(os.listdir(TEST_ROOT)):
    subj_dir = os.path.join(TEST_ROOT, name)
    result = load_subject(subj_dir, lower_body_only=lower_body_only,
                          jcf_subdir=jcf_subdir, clean_features=clean_features)
    if result is None:
        continue
    inputs, labels, mass = result
    inp = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
    inp = (inp - input_mean) / input_std
    inp.requires_grad_(True)

    out = model(inp)
    fy_sum = out[0, :, 1].sum()
    fy_sum.backward()

    grad = inp.grad[0].abs()
    T = grad.shape[0]
    total_frames += T

    for gname, indices in groups.items():
        grad_accum[gname] += grad[:, indices].sum().item() / len(indices)

    print(f"  {name}: {T} frames")

print(f"\nTotal frames: {total_frames}")
print(f"\nMean |gradient| of Fy w.r.t. input feature groups:")
print(f"{'Group':25s}  {'Mean |grad|':>12s}  {'Relative':>10s}")
print("-" * 50)

grad_means = {k: v / total_frames for k, v in grad_accum.items()}
max_grad = max(grad_means.values())
for gname, gm in sorted(grad_means.items(), key=lambda x: -x[1]):
    print(f"{gname:25s}  {gm:12.6f}  {gm/max_grad:10.3f}")

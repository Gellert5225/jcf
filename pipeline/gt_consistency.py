import os, json, numpy as np
from scipy.signal import find_peaks

TEST_ROOT = "./jcf/testing/running"
TRAIN_ROOT = "./jcf/full_duration/training/running"


def load_sto(path, skiprows=11):
    import pandas as pd
    return pd.read_csv(path, sep='\t', skiprows=skiprows)


def analyze_subject(subj_dir, name, jcf_subdir='jcf_output'):
    jcf_path = os.path.join(subj_dir, jcf_subdir,
                            'BatchJCF_JointReaction_ReactionLoads.sto')
    meta_path = os.path.join(subj_dir, 'metadata.json')
    grf_path = os.path.join(subj_dir, 'grf_data.mot')
    if not all(os.path.exists(p) for p in [jcf_path, meta_path, grf_path]):
        return None

    with open(meta_path) as f:
        mass = json.load(f)['mass_kg']
    BW = mass * 9.81

    jcf = load_sto(jcf_path)
    fx = jcf['walker_knee_r_on_tibia_r_in_tibia_r_fx'].values / BW
    fy = jcf['walker_knee_r_on_tibia_r_in_tibia_r_fy'].values / BW
    fz = jcf['walker_knee_r_on_tibia_r_in_tibia_r_fz'].values / BW
    resultant = np.sqrt(fx**2 + fy**2 + fz**2)
    time = jcf['time'].values

    # Load GRF to identify stance vs swing
    import pandas as pd
    grf = pd.read_csv(grf_path, sep='\t', skiprows=6)
    grf_time = grf['time'].values
    grf_ry = grf['calcn_r_force_vy'].values  # right foot vertical GRF
    grf_ry_interp = np.interp(time, grf_time, grf_ry)
    right_stance = grf_ry_interp > 20  # threshold for foot contact

    # Find all peaks in resultant
    all_peaks, props = find_peaks(resultant, height=0.3, distance=10)
    if len(all_peaks) < 2:
        return None

    # Classify peaks: stance (right foot on ground) vs swing (right foot off ground)
    stance_peaks = []
    swing_peaks = []
    for p in all_peaks:
        if right_stance[p]:
            stance_peaks.append(resultant[p])
        else:
            swing_peaks.append(resultant[p])

    return {
        'name': name,
        'resultant': resultant,
        'time': time,
        'stance_peaks': np.array(stance_peaks),
        'swing_peaks': np.array(swing_peaks),
        'all_peaks': resultant[all_peaks],
        'right_stance': right_stance,
    }


print("=" * 70)
print("GROUND TRUTH CONSISTENCY: STANCE vs SWING PEAKS")
print("=" * 70)

# Analyze test subjects
print("\n--- TEST SUBJECTS ---")
for name in sorted(os.listdir(TEST_ROOT)):
    subj_dir = os.path.join(TEST_ROOT, name)
    if not os.path.isdir(subj_dir):
        continue
    res = analyze_subject(subj_dir, name)
    if res is None:
        continue

    sp = res['stance_peaks']
    sw = res['swing_peaks']
    print(f"\n  {name}:")
    if len(sp) > 0:
        print(f"    Stance peaks ({len(sp)}): mean={sp.mean():.3f} std={sp.std():.3f} "
              f"CV={sp.std()/(sp.mean()+1e-8):.3f}  range=[{sp.min():.3f}, {sp.max():.3f}]")
    if len(sw) > 0:
        print(f"    Swing peaks  ({len(sw)}): mean={sw.mean():.3f} std={sw.std():.3f} "
              f"CV={sw.std()/(sw.mean()+1e-8):.3f}  range=[{sw.min():.3f}, {sw.max():.3f}]")
    else:
        print(f"    Swing peaks: NONE detected")
    if len(sp) > 0 and len(sw) > 0:
        print(f"    Ratio swing/stance: {sw.mean()/sp.mean():.3f}")

# Analyze a sample of training subjects for comparison
print("\n\n--- TRAINING SUBJECTS (sample) ---")
train_dirs = []
for name in sorted(os.listdir(TRAIN_ROOT)):
    subj_dir = os.path.join(TRAIN_ROOT, name)
    if os.path.isdir(subj_dir) and not name.startswith('best_model'):
        train_dirs.append((name, subj_dir))

all_stance_cvs = []
all_swing_cvs = []
all_swing_means = []
all_stance_means = []
n_no_swing = 0

for name, subj_dir in train_dirs[:200]:
    res = analyze_subject(subj_dir, name)
    if res is None:
        continue
    sp = res['stance_peaks']
    sw = res['swing_peaks']
    if len(sp) >= 2:
        all_stance_cvs.append(sp.std() / (sp.mean() + 1e-8))
        all_stance_means.append(sp.mean())
    if len(sw) >= 2:
        all_swing_cvs.append(sw.std() / (sw.mean() + 1e-8))
        all_swing_means.append(sw.mean())
    elif len(sw) == 0:
        n_no_swing += 1

print(f"\n  Analyzed {min(200, len(train_dirs))} training subjects")
print(f"  Subjects with NO swing peaks: {n_no_swing}")
if all_stance_cvs:
    print(f"\n  Stance peak consistency (CV):")
    print(f"    mean CV={np.mean(all_stance_cvs):.3f}  median={np.median(all_stance_cvs):.3f}")
    print(f"    mean peak height: {np.mean(all_stance_means):.3f} BW")
if all_swing_cvs:
    print(f"\n  Swing peak consistency (CV):")
    print(f"    mean CV={np.mean(all_swing_cvs):.3f}  median={np.median(all_swing_cvs):.3f}")
    print(f"    mean peak height: {np.mean(all_swing_means):.3f} BW")
if all_stance_cvs and all_swing_cvs:
    print(f"\n  Swing peaks are {np.mean(all_swing_cvs)/np.mean(all_stance_cvs):.1f}x more variable than stance peaks")

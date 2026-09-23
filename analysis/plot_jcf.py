import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sto_path = "jcf/subject10/test_output/SingleSubjTest_JointReaction_ReactionLoads.sto"
mass = 55.3  # kg
BW = mass * 9.81

df = pd.read_csv(sto_path, sep='\t', skiprows=11)

fx = df.filter(like='_fx').iloc[:, 0] / BW
fy = df.filter(like='_fy').iloc[:, 0] / BW
fz = df.filter(like='_fz').iloc[:, 0] / BW
resultant = np.sqrt(fx**2 + fy**2 + fz**2)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Top: components
ax1.plot(df['time'], fx, label='Fx (medial-lateral)')
ax1.plot(df['time'], fy, label='Fy (axial/compression)')
ax1.plot(df['time'], fz, label='Fz (anterior-posterior)')
ax1.axhline(0, color='black', linewidth=0.5)
ax1.set_ylabel('Force (BW)')
ax1.set_title('Knee Contact Force Components (walker_knee_r)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Bottom: resultant with expected range
ax2.plot(df['time'], resultant, color='black', linewidth=2, label='Resultant')
ax2.axhline(2.5, color='red', linestyle='--', alpha=0.7, label='Expected range (2.5-3.5 BW)')
ax2.axhline(3.5, color='red', linestyle='--', alpha=0.7)
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Force (BW)')
ax2.set_title('Knee Contact Force Resultant')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('knee_jcf_plot.png', dpi=150)
plt.show()
print(f"Peak resultant: {resultant.max():.2f} BW")
print("Saved to knee_jcf_plot.png")

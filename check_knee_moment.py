import nimblephysics as nimble
import numpy as np
import sys, os

try:
    b3d = nimble.biomechanics.SubjectOnDisk('./with_arm/Carter2023_Formatted_With_Arm/P010_split0/P010_split0.b3d')
    skel = b3d.readSkel(b3d.getNumProcessingPasses() - 1)
    dof_names = [skel.getDofByIndex(i).getName() for i in range(skel.getNumDofs())]
    knee_r_idx = dof_names.index('knee_angle_r')
    knee_l_idx = dof_names.index('knee_angle_l')

    n_passes = b3d.getNumProcessingPasses()
    
    # Read from multiple trials to get moments across the dataset
    mass = 55.3
    all_tau_r = []
    all_tau_l = []
    all_times = []
    cum_time = 0.0
    
    for trial in range(b3d.getNumTrials()):
        dt = b3d.getTrialTimestep(trial)
        n_frames = b3d.getTrialLength(trial)
        frames = b3d.readFrames(trial, 0, n_frames, contactThreshold=20)
        for j, f in enumerate(frames):
            pp = f.processingPasses[n_passes-1]
            all_tau_r.append(pp.tau[knee_r_idx])
            all_tau_l.append(pp.tau[knee_l_idx])
            all_times.append(cum_time + j * dt)
        cum_time += n_frames * dt
    
    tau_r = np.array(all_tau_r)
    tau_l = np.array(all_tau_l)
    times = np.array(all_times)

    print('Total frames: ' + str(len(tau_r)))
    print('Time range: ' + str(round(times[0],1)) + ' to ' + str(round(times[-1],1)) + ' s')
    print('')
    print('knee_angle_r tau: min=' + str(round(tau_r.min(),2)) + ' max=' + str(round(tau_r.max(),2)) + ' Nm')
    print('  normalized: min=' + str(round(tau_r.min()/mass,3)) + ' max=' + str(round(tau_r.max()/mass,3)) + ' Nm/kg')
    print('knee_angle_l tau: min=' + str(round(tau_l.min(),2)) + ' max=' + str(round(tau_l.max(),2)) + ' Nm')
    print('  normalized: min=' + str(round(tau_l.min()/mass,3)) + ' max=' + str(round(tau_l.max()/mass,3)) + ' Nm/kg')

    # Also check the 50-52s window specifically
    mask = (times >= 50.0) & (times <= 52.0)
    if mask.any():
        print('')
        print('50-52s window:')
        print('  knee_angle_r: min=' + str(round(tau_r[mask].min(),2)) + ' max=' + str(round(tau_r[mask].max(),2)) + ' Nm')
        print('  normalized: ' + str(round(tau_r[mask].min()/mass,3)) + ' to ' + str(round(tau_r[mask].max()/mass,3)) + ' Nm/kg')

    np.savez('knee_moment_data.npz', times=times, tau_r=tau_r, tau_l=tau_l, mass=mass)
    print('Saved knee_moment_data.npz')
except Exception as e:
    print('ERROR: ' + str(e), file=sys.stdout)

"""
Step 1: Inspect B3D Files
=========================
Run this first to see what's in your downloaded .b3d files.

Install:
    pip3 install nimblephysics

Usage:
    python 01_inspect_b3d.py /path/to/your/data/
    python 01_inspect_b3d.py /path/to/single_file.b3d
"""

import nimblephysics as nimble
import os
import sys
import glob


def inspect_subject(b3d_path: str):
    """Print everything useful about one .b3d file."""
    subject = nimble.biomechanics.SubjectOnDisk(b3d_path)
    
    fname = os.path.basename(b3d_path)
    print(f"\n{'='*60}")
    print(f"  {fname}")
    print(f"{'='*60}")
    print(f"  Mass:       {subject.getMassKg():.1f} kg")
    print(f"  Height:     {subject.getHeightM():.2f} m")
    print(f"  Sex:        {subject.getBiologicalSex()}")
    print(f"  Age:        {subject.getAgeYears()}")
    print(f"  DOFs:       {subject.getNumDofs()}")
    print(f"  Joints:     {subject.getNumJoints()}")
    print(f"  Trials:     {subject.getNumTrials()}")
    print(f"  Passes:     {subject.getNumProcessingPasses()}")
    print(f"  GRF bodies: {subject.getGroundForceBodies()}")
    print(f"  Href:       {subject.getHref()}")
    
    # Show processing pass types
    for p in range(subject.getNumProcessingPasses()):
        ptype = subject.getProcessingPassType(p)
        print(f"  Pass {p}:     {ptype}")
    
    # Show each trial
    total_frames = 0
    for t in range(subject.getNumTrials()):
        length = subject.getTrialLength(t)
        total_frames += length
        dt = subject.getTrialTimestep(t)
        duration = length * dt
        name = subject.getTrialName(t)
        tags = subject.getTrialTags(t)
        n_passes = subject.getTrialNumProcessingPasses(t)
        
        # Check how many frames have valid GRF
        missing = subject.getMissingGRF(t)
        n_valid = sum(1 for r in missing 
                      if r == nimble.biomechanics.MissingGRFReason.notMissingGRF)
        
        print(f"  Trial {t:2d}: {name:40s} | {length:5d} frames | "
              f"{duration:5.1f}s | dt={dt:.4f} | "
              f"GRF valid: {n_valid}/{length} | "
              f"passes: {n_passes} | tags: {tags}")
    
    print(f"  TOTAL:  {total_frames} frames")
    
    # Peek at one frame to show data dimensions
    if subject.getNumTrials() > 0 and subject.getTrialLength(0) > 0:
        last_pass = subject.getNumProcessingPasses() - 1
        frames = subject.readFrames(trial=0, startFrame=0, 
                                     numFramesToRead=1)
        if len(frames) > 0:
            fp = frames[0].processingPasses[min(last_pass, len(frames[0].processingPasses) - 1)]
            print(f"\n  --- Sample Frame (trial 0, frame 0, pass {last_pass}) ---")
            print(f"  pos shape:  ({len(fp.pos)},)  (joint positions)")
            print(f"  vel shape:  ({len(fp.vel)},)  (joint velocities)")
            print(f"  acc shape:  ({len(fp.acc)},)  (joint accelerations)")
            print(f"  tau shape:  ({len(fp.tau)},)  (joint torques)")
            print(f"  GRF force:  ({len(fp.groundContactForce)},)")
            print(f"  GRF CoP:    ({len(fp.groundContactCenterOfPressure)},)")
            print(f"  COM pos:    ({len(fp.comPos)},)")
    
    return {
        'file': fname,
        'mass_kg': subject.getMassKg(),
        'n_trials': subject.getNumTrials(),
        'total_frames': total_frames,
        'n_dofs': subject.getNumDofs(),
    }


def inspect_directory(data_dir: str):
    """Inspect all .b3d files in a directory."""
    b3d_files = sorted(glob.glob(os.path.join(data_dir, "**/*.b3d"), recursive=True))
    
    if not b3d_files:
        print(f"No .b3d files found in {data_dir}")
        return
    
    print(f"Found {len(b3d_files)} .b3d files in {data_dir}")
    
    summaries = []
    for path in b3d_files:
        try:
            info = inspect_subject(path)
            summaries.append(info)
        except Exception as e:
            print(f"  ERROR reading {path}: {e}")
    
    # Summary table
    print(f"\n\n{'='*60}")
    print(f"  SUMMARY: {len(summaries)} subjects loaded")
    print(f"{'='*60}")
    total = sum(s['total_frames'] for s in summaries)
    print(f"  Total frames: {total:,}")
    print(f"  DOFs:         {summaries[0]['n_dofs'] if summaries else '?'}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python 01_inspect_b3d.py <path_to_b3d_or_directory>")
        sys.exit(1)
    
    path = sys.argv[1]
    if os.path.isfile(path) and path.endswith('.b3d'):
        inspect_subject(path)
    elif os.path.isdir(path):
        inspect_directory(path)
    else:
        print(f"Not a .b3d file or directory: {path}")
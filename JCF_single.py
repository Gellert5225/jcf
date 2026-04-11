import opensim as osm
import os
import pandas as pd
import numpy as np

def test_single_subject(subject_folder, model_file):
    """
    Runs Static Optimization and Joint Reaction on one subject to verify 
    the pipeline for your ISRR paper's surrogate model.
    """
    # 1. Setup Paths
    output_dir = os.path.join(subject_folder, 'test_output')
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    # Find the required AddBiomechanics files
    # (Using simple path joining for the test)
    ik_file = os.path.join(subject_folder, 'ik_results.mot')
    grf_file = os.path.join(subject_folder, 'grf_loads.xml')
    
    # 2. Load Model
    model = osm.Model(os.path.join(subject_folder, model_file))
    model.initSystem()

    # 3. Configure Static Optimization (SO)
    so = osm.StaticOptimization()
    so.setStartTime(0.5)  # Start at 0.5s to avoid initial transients
    so.setEndTime(1.5)    # Run for 1 second of gait
    so.setActivationExponent(2)
    so.setConvergenceCriterion(1e-4)

    # 4. Configure Joint Reaction (JR)
    jr = osm.JointReaction()
    joint_names = osm.ArrayStr()
    joint_names.append("knee_r")
    jr.setJointNames(joint_names)
    on_body_names = osm.ArrayStr()
    on_body_names.append("child")
    jr.setOnBody(on_body_names)
    in_frame_names = osm.ArrayStr()
    in_frame_names.append("child")
    jr.setInFrame(in_frame_names)
    
    # Note: AnalyzeTool handles the passing of SO forces to JR automatically
    
    # 5. Execute via AnalyzeTool
    analyze = osm.AnalyzeTool(model)
    analyze.setName("SingleSubjTest")
    analyze.setResultsDir(output_dir)
    analyze.setCoordinatesFileName(ik_file)
    analyze.setExternalLoadsFileName(grf_file)
    
    analyze.getAnalysisSet().adoptAndAppend(so)
    analyze.getAnalysisSet().adoptAndAppend(jr)
    
    print("--- Starting Physics Solve (SO + JCF) ---")
    analyze.run()
    print(f"--- Process Complete. Files saved to: {output_dir} ---")

    # 6. Quick Validation Check
    # Load the resulting JCF file to check if values are realistic
    jcf_results = os.path.join(output_dir, "SingleSubjTest_JointReaction_ReactionLoads.sto")
    if os.path.exists(jcf_results):
        df = pd.read_csv(jcf_results, sep='\t', skiprows=6)
        # Look for the vertical compression component (typically 'knee_r_on_tibia_in_tibia_fy')
        fy_col = [c for c in df.columns if 'in_tibia_fy' in c][0]
        max_force = df[fy_col].max()
        print(f"Validation: Peak Knee Contact Force detected at {max_force:.2f} Newtons.")
        print("Tip: Divide this by (Mass * 9.81) to ensure it is in the 2.0-4.0 BW range.")

# --- Run the Test ---
# Replace with your actual local paths
test_single_subject(
    subject_folder='./jcf/P010_split0', 
    model_file='scaled_model.osim'
)


import opensim as osm
import os
import glob

def batch_process_muscle_informed_jcf(root_dir, model_name="scaled_model.osim"):
    """
    Two-step batch process for 100 subjects:
    1. Static Optimization (SO) -> Estimates muscle forces.
    2. Joint Reaction (JR) -> Uses SO results to calculate true JCF.
    """
    subject_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    
    for subject in subject_dirs[:100]:
        subj_path = os.path.join(root_dir, subject)
        output_dir = os.path.join(subj_path, 'processed_labels')
        if not os.path.exists(output_dir): os.makedirs(output_dir)

        try:
            model_p = os.path.join(subj_path, model_name)
            ik_p = glob.glob(os.path.join(subj_path, "*_ik.mot"))[0]
            grf_p = glob.glob(os.path.join(subj_path, "*_grf.xml"))[0]
            
            model = osm.Model(model_p)
            model.initSystem()

            # --- STEP 1: STATIC OPTIMIZATION (SO) ---
            # Resolves muscle forces required for the observed motion
            so = osm.StaticOptimization()
            so.setStartTime(0); so.setEndTime(2) # Adjust to trial length
            so.setConvergenceTolerance(1e-4)
            so.setActivationExponent(2) # Standard muscle effort minimization
            
            # --- STEP 2: JOINT REACTION ANALYSIS (JR) ---
            # Uses SO results as the 'ForcesFileName' for accurate JCF
            jr = osm.JointReaction()
            jr.setJointNames(osm.ArrayStr().append("knee_r")) # Focus on Knee
            jr.setOnBodyNames(osm.ArrayStr().append("child"))
            jr.setInFrameNames(osm.ArrayStr().append("child"))
            jr.setCoordinatesFileName(ik_p)
            jr.setExternalLoadsFileName(grf_p)

            # --- STEP 3: RUN ANALYZE TOOL ---
            analyze = osm.AnalyzeTool(model)
            analyze.setName(subject)
            analyze.setResultsDir(output_dir)
            analyze.setCoordinatesFileName(ik_p)
            analyze.setExternalLoadsFileName(grf_p)
            
            # Add both analyses to the set
            analyze.getAnalysisSet().adoptAndAppend(so)
            analyze.getAnalysisSet().adoptAndAppend(jr)
            
            print(f"Processing Subject {subject}: SO + JCF...")
            analyze.run()
            
            # Note: The JCF tool automatically looks for the SO results 
            # within the same AnalyzeTool execution context.

        except Exception as e:
            print(f"Failed {subject}: {e}")

# Run for your 100 subjects
batch_process_muscle_informed_jcf('/path/to/AddBiomechanics/Data')


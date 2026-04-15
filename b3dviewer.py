import nimblephysics as nimble

# Load the model
your_subject = nimble.biomechanics.SubjectOnDisk("/home/gellert/Developer/medical_dt/with_arm/training/Carter2023_Formatted_With_Arm/P002_split0/P002_split0.b3d")
# Read the skeleton that was optimized by the first process pass (always kinematics)
# Use the geometryFolder argument to specify where to load the bone mesh geometry from
skeleton: nimble.dynamics.Skeleton = your_subject.readSkel(
    processingPass=0,
    geometryFolder="/home/gellert/Developer/medical_dt/with_arm/training/Carter2023_Formatted_With_Arm/P002_split0/Geometry")

# Create a GUI
gui = nimble.NimbleGUI()

# Serve the GUI on port 8080
gui.serve(8080)

# Render the skeleton to the GUI
gui.nativeAPI().renderSkeleton(skeleton)

# Block until the GUI is closed
gui.blockWhileServing()
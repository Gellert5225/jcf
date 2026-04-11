"""Test: load AnalyzeTool from existing working XML with analyses."""
import opensim as osm
import os, sys

setup_xml = os.path.abspath('./jcf/P010_split0/test_output/_analyze_setup.xml')
output_dir = os.path.abspath('./jcf/P010_split0/test_output')

print(f"Loading AnalyzeTool from {setup_xml}")
analyze = osm.AnalyzeTool(setup_xml)
print(f"  Analysis set size: {analyze.getAnalysisSet().getSize()}")
for i in range(analyze.getAnalysisSet().getSize()):
    a = analyze.getAnalysisSet().get(i)
    print(f"  Analysis {i}: {a.getConcreteClassName()} ({a.getName()})")

print("--- Running ---")
sys.stdout.flush()
analyze.run()
print("--- run() completed ---")
analyze.printResults(analyze.getName(), output_dir)
print("--- printResults done ---")
sys.stdout.flush()
os._exit(0)

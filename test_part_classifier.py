import sys
import os

# Ensure we're in the right directory
os.chdir(r"E:\File storage for 3d printing\woodring_print_files\Meshfinder")
sys.path.insert(0, r"E:\File storage for 3d printing\woodring_print_files\Meshfinder")

print("Working directory:", os.getcwd())
print("Python executable:", sys.executable)
print()

try:
    from sklearn.ensemble import RandomForestClassifier
    print("[OK] sklearn.ensemble.RandomForestClassifier imported")
except Exception as e:
    print("[FAIL] sklearn import error:", e)
    import traceback
    traceback.print_exc()

print()

try:
    from src.ml.part_classifier import PartClassifier
    print("[OK] PartClassifier imported")
except Exception as e:
    print("[FAIL] PartClassifier import error:", e)
    import traceback
    traceback.print_exc()


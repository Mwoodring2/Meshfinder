import sys
print("Python:", sys.executable)
print("Version:", sys.version)

try:
    import scipy
    print("[OK] scipy:", scipy.__version__)
except Exception as e:
    print("[FAIL] scipy ERROR:", e)

try:
    import sklearn
    print("[OK] sklearn:", sklearn.__version__)
except Exception as e:
    print("[FAIL] sklearn ERROR:", e)

try:
    import numpy
    print("[OK] numpy:", numpy.__version__)
except Exception as e:
    print("[FAIL] numpy ERROR:", e)

try:
    from src.dataio.db import ensure_user_corrections
    print("[OK] ensure_user_corrections imported")
except Exception as e:
    print("[FAIL] ensure_user_corrections ERROR:", e)


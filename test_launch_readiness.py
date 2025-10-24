#!/usr/bin/env python
"""
Comprehensive stress test for launch.bat readiness
Tests all critical dependencies and systems before running the full application
"""
import sys
import os
from pathlib import Path

print("="*70)
print("LAUNCH READINESS STRESS TEST")
print("="*70)

# Track results
tests_passed = 0
tests_failed = 0
warnings = []

def test(name, func):
    """Run a test and track results"""
    global tests_passed, tests_failed
    try:
        print(f"\n[TEST] {name}...", end=" ")
        result = func()
        if result:
            print("PASS")
            tests_passed += 1
        else:
            print("FAIL")
            tests_failed += 1
        return result
    except Exception as e:
        print(f"FAIL - {e}")
        tests_failed += 1
        return False

def warn(message):
    """Add a warning"""
    warnings.append(message)
    print(f"  [WARN] {message}")

# ============================================================================
# TEST 1: Core Python Libraries
# ============================================================================
def test_core_imports():
    """Test core Python libraries"""
    import sqlite3
    import json
    import datetime
    return True

# ============================================================================
# TEST 2: Scientific Computing Stack
# ============================================================================
def test_scientific_stack():
    """Test numpy, scipy, scikit-learn"""
    import numpy as np
    import scipy
    from scipy.spatial import ConvexHull
    import sklearn
    from sklearn.ensemble import RandomForestClassifier
    
    # Quick scipy test
    points = np.random.rand(10, 3)
    hull = ConvexHull(points)
    
    # Quick sklearn test
    clf = RandomForestClassifier(n_estimators=10)
    
    print(f"\n    numpy: {np.__version__}")
    print(f"    scipy: {scipy.__version__}")
    print(f"    scikit-learn: {sklearn.__version__}")
    
    return True

# ============================================================================
# TEST 3: 3D Processing Libraries
# ============================================================================
def test_3d_libraries():
    """Test trimesh and related libraries"""
    import trimesh
    import pyglet
    import OpenGL
    
    print(f"\n    trimesh: {trimesh.__version__}")
    print(f"    pyglet: {pyglet.version}")
    
    return True

# ============================================================================
# TEST 4: UI Framework
# ============================================================================
def test_ui_framework():
    """Test PySide6"""
    from PySide6 import QtCore, QtWidgets, QtGui
    
    print(f"\n    PySide6: {QtCore.__version__}")
    
    return True

# ============================================================================
# TEST 5: Data Processing
# ============================================================================
def test_data_processing():
    """Test pandas and openpyxl"""
    import pandas as pd
    import openpyxl
    import joblib
    
    print(f"\n    pandas: {pd.__version__}")
    print(f"    openpyxl: {openpyxl.__version__}")
    
    return True

# ============================================================================
# TEST 6: Database Module
# ============================================================================
def test_database_module():
    """Test database module imports and functions"""
    from src.dataio.db import (
        ensure_user_corrections,
        add_user_correction,
        get_user_corrections,
        clear_user_corrections,
        update_file_record,
        batch_update_proposals,
        update_proposal,
        log_op,
        get_file_records
    )
    
    # Test that functions are callable
    assert callable(ensure_user_corrections)
    assert callable(add_user_correction)
    assert callable(update_proposal)
    assert callable(log_op)
    assert callable(get_file_records)
    
    print(f"\n    All database functions present")
    
    return True

# ============================================================================
# TEST 7: Database Connectivity
# ============================================================================
def test_database_connectivity():
    """Test database connection and basic operations"""
    import sqlite3
    
    db_paths = [
        "db/modelfinder.db",
        "data/db/modelfinder.db"
    ]
    
    found = False
    for db_path in db_paths:
        if Path(db_path).exists():
            print(f"\n    Found database: {db_path}")
            con = sqlite3.connect(db_path)
            cur = con.cursor()
            
            # Check files table
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='files'")
            if cur.fetchone():
                print(f"    Files table exists")
                cur.execute("SELECT COUNT(*) FROM files")
                count = cur.fetchone()[0]
                print(f"    Files in database: {count}")
                found = True
            
            con.close()
            break
    
    if not found:
        warn("No database found, but this is OK for first run")
    
    return True

# ============================================================================
# TEST 8: Feature Extraction (scipy-dependent)
# ============================================================================
def test_feature_extraction():
    """Test feature extraction with scipy"""
    import numpy as np
    from scipy.spatial import ConvexHull
    import trimesh
    
    # Create a simple test mesh
    vertices = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0.5, 0.5, 1]
    ])
    faces = np.array([
        [0, 1, 4],
        [1, 2, 4],
        [2, 3, 4],
        [3, 0, 4],
        [0, 1, 2],
        [0, 2, 3]
    ])
    
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    # Test basic mesh operations
    volume = mesh.volume
    area = mesh.area
    bounds = mesh.bounds
    
    # Test scipy-dependent operations
    hull = ConvexHull(vertices)
    
    print(f"\n    Mesh volume: {volume:.4f}")
    print(f"    Mesh area: {area:.4f}")
    print(f"    Convex hull vertices: {len(hull.vertices)}")
    
    return True

# ============================================================================
# TEST 9: ML Classifier
# ============================================================================
def test_ml_classifier():
    """Test that we can create and use ML classifiers"""
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    
    # Create dummy data
    X, y = make_classification(n_samples=100, n_features=10, n_classes=3, 
                               n_informative=8, random_state=42)
    
    # Train a quick classifier
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X[:80], y[:80])
    
    # Test prediction
    accuracy = clf.score(X[80:], y[80:])
    
    print(f"\n    Trained RandomForest classifier")
    print(f"    Test accuracy: {accuracy:.2%}")
    
    return True

# ============================================================================
# TEST 10: Proposal System
# ============================================================================
def test_proposal_system():
    """Test proposal system imports"""
    try:
        from src.features.propose_from_reference import propose_for_rows, RowMeta
        from src.ui.workers import ProposeWorker
        from src.dataio.reference_parts import load_reference_parts, get_all_projects
        
        print(f"\n    Proposal system modules loaded")
        return True
    except ImportError as e:
        warn(f"Proposal system not fully available: {e}")
        return True  # Not critical for basic operation

# ============================================================================
# TEST 11: Indexer Module
# ============================================================================
def test_indexer():
    """Test indexer module"""
    try:
        from src.indexer.modelfinder_indexer import scan_directory
        print(f"\n    Indexer module loaded")
        return True
    except ImportError as e:
        warn(f"Indexer module issue: {e}")
        return False

# ============================================================================
# Run All Tests
# ============================================================================

test("Core Python Libraries", test_core_imports)
test("Scientific Computing Stack (numpy, scipy, sklearn)", test_scientific_stack)
test("3D Processing Libraries (trimesh, pyglet, OpenGL)", test_3d_libraries)
test("UI Framework (PySide6)", test_ui_framework)
test("Data Processing (pandas, openpyxl)", test_data_processing)
test("Database Module Functions", test_database_module)
test("Database Connectivity", test_database_connectivity)
test("Feature Extraction with scipy", test_feature_extraction)
test("ML Classifier Training", test_ml_classifier)
test("Proposal System", test_proposal_system)
test("Indexer Module", test_indexer)

# ============================================================================
# Summary
# ============================================================================

print("\n" + "="*70)
print("STRESS TEST RESULTS")
print("="*70)
print(f"Tests Passed: {tests_passed}")
print(f"Tests Failed: {tests_failed}")

if warnings:
    print(f"\nWarnings ({len(warnings)}):")
    for w in warnings:
        print(f"  - {w}")

print("\n" + "="*70)

if tests_failed == 0:
    print("STATUS: ALL SYSTEMS GO! Ready to launch application.")
    print("="*70)
    sys.exit(0)
else:
    print("STATUS: SOME TESTS FAILED. Review errors above.")
    print("="*70)
    sys.exit(1)

